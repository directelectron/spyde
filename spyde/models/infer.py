"""Parameter-free inference: scale-norm -> local norm -> model -> decode -> map back.

``load_model`` + ``detect`` are vendored from the ``yoloDiffraction`` research
project (``yolodiffraction/model/infer.py``), rewired to import from the sibling
modules in this package. ``detect_batch`` is new to SpyDE: it runs a whole stack
of frames through one forward pass (the batched path the 4D-STEM chunk compute
needs), reusing the GPU-resident ``decode_batch``.

The full single-frame pipeline a user runs needs no parameters: disk size is
estimated automatically and the frame rescaled to the canonical size before the
model; predicted positions are mapped back to the original frame coordinates.
"""
from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch

from .decode import decode, decode_batch, split_by_batch
from .preprocess import estimate_disk_diameter, normalize_input, scale_to_canonical
from .unet import SpotUNet

log = logging.getLogger(__name__)


# ── Apple-MPS robustness gates (Mac-only) ─────────────────────────────────────
# SpotUNet uses nn.ConvTranspose2d + BatchNorm; ConvTranspose on MPS has a long
# crash/correctness bug history, and the multi-worker batch path runs several
# MPS forwards concurrently across worker processes. On Mac, some MPS op failures
# raise a catchable RuntimeError, but the ones that take the *process* down are
# uncatchable native Metal aborts (SIGABRT/segfault) — no try/except catches
# those. The gates below make the Mac path non-crashing by default while keeping
# MPS for the safe single-thread preview.
#
# ── THE ONE FLAG a user flips after validating MPS on their Mac ──────────────
# ``SPYDE_NEURAL_MPS_BATCH`` controls whether the multi-worker BATCH path uses
# MPS on Mac at all:
#     unset / "0" / "off"  → batch runs on CPU (the SAFE default; we cannot test
#                            MPS on the Windows dev box, so the shipped default
#                            must not crash). The single-frame PREVIEW still uses
#                            MPS (one thread in the main process — far safer).
#     "1" / "on"           → batch is ALLOWED to use MPS again (once the user has
#                            confirmed ConvTranspose/BatchNorm are stable on their
#                            Mac). Fixes 1-4 (MPS fallback env, catchable-error CPU
#                            retry, device serialization, worker-death demotion)
#                            all still apply, so re-enabling is much safer than
#                            before.
# On non-Mac (CUDA) this gate is IRRELEVANT — ``mps_batch_allowed`` only ever
# returns False on darwin; CUDA behaviour is completely unchanged.
def _is_mac() -> bool:
    return sys.platform == "darwin"


def mps_batch_allowed() -> bool:
    """True iff the multi-worker neural BATCH may use MPS. Off-Mac this is always
    True (the gate is a no-op — CUDA/CPU decide via the device chain). On Mac it is
    False unless ``SPYDE_NEURAL_MPS_BATCH`` is explicitly enabled (see the module
    comment: the safe default is CPU batch on Mac, one flag re-enables MPS batch)."""
    if not _is_mac():
        return True
    return os.environ.get("SPYDE_NEURAL_MPS_BATCH", "").lower() in ("1", "on", "true", "yes")


def enable_mps_cpu_fallback() -> None:
    """Set ``PYTORCH_ENABLE_MPS_FALLBACK=1`` on Mac so an unsupported MPS op (e.g.
    a ConvTranspose gap in a given torch build) transparently runs on CPU per-op
    instead of raising or aborting. MUST be called BEFORE torch initialises MPS
    (i.e. before the first MPS tensor/op) to take effect — so this is invoked at
    process startup (main process and every worker). No-op off Mac; idempotent."""
    if not _is_mac():
        return
    # Only set if unset so a user override (e.g. explicitly "0") is respected.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# On import (main process + every worker that imports this module) make the
# unsupported-op → CPU fallback active before any MPS work. Cheap + idempotent.
enable_mps_cpu_fallback()


def _default_device():
    """Best device for inference: CUDA → Apple-MPS → CPU (mirrors
    ``find_vectors_torch.torch_gpu_device``; previously Macs silently ran the
    model on CPU while the batch path believed it had a GPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def load_model(ckpt_path, device=None, arch: dict | None = None):
    """Load a SpotUNet checkpoint. ``arch`` (from the model registry) overrides the
    architecture hyperparams when present; otherwise they're read from the
    checkpoint (which stores ``base``/``in_ch``/``levels``)."""
    device = device or _default_device()
    # weights_only=True: checkpoints are plain state dicts + scalar hyperparams;
    # never unpickle arbitrary objects from a (possibly downloaded) file.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    arch = arch or {}
    in_ch = arch.get("in_ch", ck.get("in_ch", 1))
    model = SpotUNet(
        in_ch=in_ch,
        base=arch.get("base", ck["base"]),
        levels=arch.get("levels", ck.get("levels", 2)),
    )
    model.load_state_dict(ck["state_dict"])
    model = model.to(device)
    model.eval()
    if device.type == "mps":
        # Smoke-test the forward ONCE at load: an MPS build missing an op this
        # net uses should degrade to CPU here, not fail on every chunk.
        try:
            with torch.no_grad():
                model(torch.zeros(1, in_ch, 32, 32, device=device))
        except Exception:
            device = torch.device("cpu")
            model = model.to(device)
    return model, device


def _pad_to_multiple(nrm: np.ndarray, levels: int):
    """Reflect-pad H,W up to a multiple of 2**levels so the U-Net pool/upsample
    line up. Returns the padded array (unchanged if already aligned)."""
    H0, W0 = nrm.shape
    mult = 2 ** levels
    ph = (mult - H0 % mult) % mult
    pw = (mult - W0 % mult) % mult
    if ph or pw:
        nrm = np.pad(nrm, ((0, ph), (0, pw)), mode="reflect")
    return nrm


# Local-norm high-pass scale (px). The default suits normal data; VERY diffuse
# backgrounds (beam stop / amorphous halo) need a SMALLER value — set automatically
# by ``calibrate`` / ``auto_bg_sigma`` rather than by hand.
DEFAULT_BG_SIGMA = 12.0


def _estimate_work_diam(frame: np.ndarray, factor: float,
                        spot_diameter: float | None) -> float:
    """Disk diameter at WORKING (post-scale) resolution — drives BOTH the
    background-subtraction scale and the NMS window for big disks run near-native
    (the SCALE_CLIP floor keeps very large disks well above canonical). Ported from
    yoloDiffraction detect(): the autocorrelation estimator high-passes at
    sigma=20 (canonical-tuned) and SATURATES on big disks, so iterate with a
    size-adaptive high-pass, then correct its measured ~0.83 undershoot."""
    if spot_diameter:
        return float(spot_diameter) * factor
    wd = estimate_disk_diameter(frame)
    if np.isfinite(wd) and wd > 20:
        for _ in range(3):
            wd2 = estimate_disk_diameter(frame, hp_sigma=max(20.0, wd))
            if abs(wd2 - wd) < 1:
                wd = wd2
                break
            wd = max(wd, wd2)
        wd = wd / 0.83
    return wd * factor


def _big_disk_params(md: int, bg_sigma: float | None, work_diam: float):
    """Resolve the NMS window + background sigma for the working disk size.

    Canonical-size data (work_diam <= 20) is untouched, so existing (production-
    model) behaviour is unchanged; this only engages on big disks where the fixed
    values were wrong two ways (measured in yoloDiffraction, 24-96px disks):
      - a fixed ~4px NMS window fires a CLOUD of detections across one broad disk
        -> md ~ 0.55*diameter collapses it to ONE (center err <1.3px);
      - a fixed 12px background sigma high-passes INSIDE the disk (carves the flat
        interior into a ring the model fires on) -> bg ~ 0.6*diameter keeps the
        interior intact. An explicitly-passed bg_sigma (e.g. from calibrate) is
        respected; only the None default is size-scaled."""
    big = np.isfinite(work_diam) and work_diam > 20
    if big:
        md = max(md, int(round(work_diam * 0.55)))
    if bg_sigma is None:
        bg_sigma = max(DEFAULT_BG_SIGMA, 0.6 * work_diam) if big else DEFAULT_BG_SIGMA
    return md, float(bg_sigma)


@torch.no_grad()
def auto_bg_sigma(model, work: np.ndarray, device,
                  candidates=(4, 5, 6, 8, 10, 12, 16, 20)) -> tuple[float, float]:
    """Pick the local-norm ``bg_sigma`` that MAXIMISES the model's heatmap confidence.

    The model's own peak confidence is a reliable, hands-off signal for how well the
    diffuse background was removed (confidence has a clean maximum vs bg_sigma). Used
    by ``calibrate`` on a few representative frames; reused across the scan. Returns
    (best_bg_sigma, best_confidence)."""
    levels = int(getattr(model, "levels", 2))
    best_conf, best_sigma = -1.0, DEFAULT_BG_SIGMA
    for s in candidates:
        nrm = _pad_to_multiple(normalize_input(work, local=True, bg_sigma=float(s)), levels)
        hm, _ = model(torch.from_numpy(nrm[None, None].astype(np.float32)).to(device))
        c = float(torch.sigmoid(hm).max())
        if c > best_conf:
            best_conf, best_sigma = c, float(s)
    return best_sigma, best_conf


def bg_sigma_from_peak_size(peak_diameter_px: float) -> float:
    """Map a known/estimated peak diameter (px) -> bg_sigma (high-pass scale): a few
    x the peak so the high-pass removes the diffuse background but not the peaks."""
    return float(np.clip(1.2 * peak_diameter_px, 4.0, 24.0))


@torch.no_grad()
def detect(model, frame: np.ndarray, device, thresh: float = 0.3,
           min_distance: int = 4, auto_scale: bool = True,
           bg_sigma: float | None = None,
           spot_diameter: float | None = None):
    """Detect spots in a single frame. Returns (N,3) [y,x,score] in ORIGINAL coords.

    Estimates disk size, rescales to canonical, runs the model, maps positions back.
    ``bg_sigma`` is the local-norm high-pass scale (set by ``calibrate`` for diffuse
    data; ``None`` = automatic: the default 12 for normal data, scaled up with the
    disk for big disks run near-native). ``thresh`` is the heatmap confidence.
    ``spot_diameter`` (px) overrides the autocorrelation disk-size estimate — the
    user-facing "Spot size" knob for when the estimate gets it wrong; the canonical
    rescale then derives from it (both directions, bounded by SCALE_CLIP).
    """
    factor = 1.0
    work = frame
    if auto_scale:
        work, factor = scale_to_canonical(frame, diameter=spot_diameter)
    md = max(2, int(round(min_distance * factor)))
    work_diam = _estimate_work_diam(frame, factor, spot_diameter)
    md, bg_sigma = _big_disk_params(md, bg_sigma, work_diam)
    nrm = normalize_input(work, local=True, bg_sigma=bg_sigma)
    levels = int(getattr(model, "levels", 2))
    nrm = _pad_to_multiple(nrm, levels)
    x = torch.from_numpy(nrm[None, None]).to(device)
    hm, off = model(x)
    pred = decode(hm[0], off[0], thresh=thresh, min_distance=md)
    if len(pred) and factor != 1.0:
        pred = pred.copy()
        pred[:, :2] = pred[:, :2] / factor          # map back to original coords
    return pred


def _neural_sub_batch_size() -> int:
    """Sub-batch size (frames per forward pass) for ``detect_batch``. A single
    forward pass over a whole nav chunk (1000+ frames) allocates activations for
    every frame at once — tens of GB for a big U-Net. Looping in fixed sub-batches
    bounds peak activation memory to O(K) instead of O(N) while keeping identical
    results (the loop is purely a memory/throughput knob; the per-frame maths is
    unchanged). ``SPYDE_NEURAL_BATCH`` overrides the default; parsed once per call,
    clamped >= 1."""
    import os
    try:
        k = int(os.environ.get("SPYDE_NEURAL_BATCH", "64"))
    except ValueError:
        k = 64
    return max(1, k)


def _forward_with_cpu_retry(model, device, chunk_np: np.ndarray):
    """Run ``model`` on ``chunk_np`` ((K,H,W)); on a torch/MPS ``RuntimeError``
    move the model to CPU, retry on CPU, and return the CPU model+device so the
    caller stays on CPU (fix 2 — catchable MPS op failure → per-op CPU retry).

    Returns ``(hm, off, model, device)`` — the (possibly CPU-moved) model+device
    are what the caller must use for the REST of this call. On CUDA/CPU the retry
    path never triggers, so behaviour there is unchanged."""
    x = torch.from_numpy(chunk_np[:, None]).to(device)
    try:
        hm, off = model(x)
        return hm, off, model, device
    except RuntimeError as e:
        # A catchable MPS (or other device) op failure. Re-move the model to CPU
        # ONCE and retry this sub-batch there; subsequent sub-batches reuse the
        # CPU model. (PYTORCH_ENABLE_MPS_FALLBACK should already route unsupported
        # ops to CPU per-op; this catches the ones that still raise.)
        if device.type == "cpu":
            raise
        log.warning("[models] %s forward raised (%s); retrying this sub-batch on "
                    "CPU and staying on CPU for the rest of this call",
                    device.type, e)
        del x
        cpu = torch.device("cpu")
        model = model.to(cpu)
        model.eval()
        # Also demote the SHARED registry cache so the next get_model() caller in
        # this process doesn't re-select the just-failed device and crash again.
        try:
            from .registry import demote_cached_models_to_cpu
            demote_cached_models_to_cpu()
        except Exception:
            pass
        x = torch.from_numpy(chunk_np[:, None]).to(cpu)
        hm, off = model(x)
        return hm, off, model, cpu


@torch.no_grad()
def detect_batch(model, frames, device, thresh: float = 0.3,
                 min_distance: int = 4, auto_scale: bool = True,
                 shared_scale: bool = True, bg_sigma: float | None = None,
                 spot_diameter: float | None = None):
    """Detect spots in a STACK of frames in one forward pass.

    ``frames`` is an (N,H,W) array (or a sequence of (H,W) arrays). Returns a list
    of N (Ni,3) [y,x,score] arrays in ORIGINAL frame coordinates — the same per-frame
    contract as ``detect``, but the model runs once on the whole batch and decoding
    stays GPU-resident (``decode_batch``).

    All frames in a scan share the same physical disk size, so ``shared_scale``
    (default) estimates the scale factor ONCE from the first frame and applies it
    uniformly (cheaper, and what the scale-norm design recommends — "reuse one
    estimate across a stack"). The whole batch must share one H,W for a single
    tensor, so a shared factor is also required to keep frames the same shape.

    Memory: the (N,H,W) stack is run through the U-Net in fixed sub-batches of
    ``SPYDE_NEURAL_BATCH`` frames (default 64), NOT as one (N,1,H,W) tensor — a
    single pass over a full nav chunk (1000+ frames) held ~16 GB of activations.
    The scale/NMS-window/bg_sigma are resolved ONCE from ``frames[0]`` (below,
    outside the sub-batch loop) so results are bit-identical to a single pass;
    only the forward+decode is chunked.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 2:
        frames = frames[None]
    N = frames.shape[0]
    if N == 0:
        return []

    factor = 1.0
    if auto_scale:
        # One estimate for the whole stack (shared physical disk size); a known
        # ``spot_diameter`` (the user's Spot-size override) replaces the estimate.
        _ref, factor = scale_to_canonical(frames[0], diameter=spot_diameter)

    # Shared physical disk size across the scan -> resolve the size-dependent
    # NMS window + background sigma ONCE from the first frame (like the scale).
    md = max(2, int(round(min_distance * factor)))
    work_diam = _estimate_work_diam(frames[0], factor, spot_diameter)
    md, bg_sigma = _big_disk_params(md, bg_sigma, work_diam)

    levels = int(getattr(model, "levels", 2))
    nrm_list = []
    for f in frames:
        if auto_scale and factor != 1.0:
            from scipy.ndimage import zoom
            f = zoom(f, factor, order=1)
        nrm_list.append(normalize_input(f, local=True, bg_sigma=bg_sigma))
    # All frames now share one shape; pad the stack to the U-Net multiple.
    stack = np.stack(nrm_list, 0)
    stack = np.stack([_pad_to_multiple(n, levels) for n in stack], 0)

    K = _neural_sub_batch_size()
    per_frame: list = []
    # ``device`` / ``model`` may flip to CPU mid-call if an MPS op raises a
    # catchable RuntimeError (fix 2): re-move the model to CPU ONCE, retry the
    # failing sub-batch on CPU, and stay on CPU for the remaining sub-batches so
    # a flaky MPS op degrades cleanly at the finest grain instead of bubbling up
    # to the coarser ``_neural_block`` catch (which drops to a slow per-frame CPU
    # loop for the whole chunk). ``cur_*`` is the live model/device the loop uses.
    cur_model, cur_device = model, device
    for i0 in range(0, N, K):
        chunk = stack[i0:i0 + K]
        # If even the CPU retry inside _forward_with_cpu_retry fails, the error
        # propagates to the caller's coarser fallback (per-frame CPU in
        # _neural_block) — no extra handling needed here.
        hm, off, cur_model, cur_device = _forward_with_cpu_retry(
            cur_model, cur_device, chunk)
        res = decode_batch(hm, off, thresh=thresh, min_distance=md)
        per_frame.extend(split_by_batch(res, chunk.shape[0]))
        del hm, off, res
        if cur_device.type == "cuda":
            torch.cuda.empty_cache()

    if factor != 1.0:
        for i, p in enumerate(per_frame):
            if len(p):
                p = p.copy()
                p[:, :2] = p[:, :2] / factor
                per_frame[i] = p
    return per_frame


@torch.no_grad()
def calibrate(model, sample_frames, device, tune_threshold: bool = True,
              spot_diameter: float | None = None):
    """Calibration step (run ONCE on a few representative frames before a full scan).

    All frames in a 4D-STEM scan share the same physical disk size and background
    character, so we optimise the dataset-dependent knobs ONCE here and reuse them
    across the whole scan (cheap + hands-off). Optimises:
      - ``bg_sigma``: confidence-maximising local-norm high-pass scale (handles
        diffuse / beam-stop backgrounds without manual tuning).
      - ``scale_factor``: the disk-size -> canonical upscale (so tiny disks like
        SPED-Ag are seen at the trained size).
      - ``threshold``: lowered for faint-peak datasets where the confident response
        is weak (e.g. amorphous SRO / in-SEM), else the default 0.3.

    ``sample_frames``: a few (H,W) frames spread across the scan (ideally nav-blurred
    the same way the full compute will be). Returns a dict to pass into
    ``detect``/``detect_batch`` (bg_sigma, thresh) and ``scale_factor`` for reference.
    """
    frames = [np.asarray(f, np.float32) for f in sample_frames]
    if not frames:
        return {"bg_sigma": DEFAULT_BG_SIGMA, "thresh": 0.3, "scale_factor": 1.0}

    # 1) scale: one estimate (shared physical disk size). scale_to_canonical returns
    #    the upscale-only factor; a known spot_diameter overrides the estimate.
    _ref, factor = scale_to_canonical(frames[0], diameter=spot_diameter)
    work0 = _ref if factor != 1.0 else frames[0]

    # 2) bg_sigma: confidence-max over the sample (median best-sigma is robust).
    #    Big disks run near-native need bg ~ 0.6*diameter (a 12px high-pass carves
    #    the disk interior into a ring), so extend the candidate list with
    #    size-scaled values — the confidence-max picks them only when they win.
    candidates = [4, 5, 6, 8, 10, 12, 16, 20]
    wd = _estimate_work_diam(frames[0], factor, spot_diameter)
    if np.isfinite(wd) and wd > 20:
        candidates += [round(r * wd, 1) for r in (0.5, 0.6, 0.7)]
    sigmas = []
    confs = []
    for f in frames:
        w = f
        if factor != 1.0:
            from scipy.ndimage import zoom
            w = zoom(f, factor, order=1)
        s, c = auto_bg_sigma(model, w, device, candidates=tuple(candidates))
        sigmas.append(s)
        confs.append(c)
    bg = float(np.median(sigmas))
    best_conf = float(np.median(confs))

    # 3) threshold: if even the best-bg confidence is weak (faint-peak regime), lower
    #    the threshold so real faint peaks survive; otherwise keep the default.
    thresh = 0.3
    if tune_threshold and best_conf < 0.45:
        thresh = float(np.clip(0.5 * best_conf, 0.1, 0.3))

    return {"bg_sigma": bg, "thresh": thresh, "scale_factor": float(factor),
            "confidence": best_conf}

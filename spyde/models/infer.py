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

import numpy as np
import torch

from .decode import decode, decode_batch, split_by_batch
from .preprocess import normalize_input, scale_to_canonical
from .unet import SpotUNet


def load_model(ckpt_path, device=None, arch: dict | None = None):
    """Load a SpotUNet checkpoint. ``arch`` (from the model registry) overrides the
    architecture hyperparams when present; otherwise they're read from the
    checkpoint (which stores ``base``/``in_ch``/``levels``)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    arch = arch or {}
    model = SpotUNet(
        in_ch=arch.get("in_ch", ck.get("in_ch", 1)),
        base=arch.get("base", ck["base"]),
        levels=arch.get("levels", ck.get("levels", 2)),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
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
           bg_sigma: float = DEFAULT_BG_SIGMA):
    """Detect spots in a single frame. Returns (N,3) [y,x,score] in ORIGINAL coords.

    Estimates disk size, rescales to canonical, runs the model, maps positions back.
    ``bg_sigma`` is the local-norm high-pass scale (set by ``calibrate`` for diffuse
    data; default suits normal data). ``thresh`` is the heatmap confidence.
    """
    factor = 1.0
    work = frame
    if auto_scale:
        work, factor = scale_to_canonical(frame)
    nrm = normalize_input(work, local=True, bg_sigma=bg_sigma)
    levels = int(getattr(model, "levels", 2))
    nrm = _pad_to_multiple(nrm, levels)
    x = torch.from_numpy(nrm[None, None]).to(device)
    hm, off = model(x)
    md = max(2, int(round(min_distance * factor)))
    pred = decode(hm[0], off[0], thresh=thresh, min_distance=md)
    if len(pred) and factor != 1.0:
        pred = pred.copy()
        pred[:, :2] = pred[:, :2] / factor          # map back to original coords
    return pred


@torch.no_grad()
def detect_batch(model, frames, device, thresh: float = 0.3,
                 min_distance: int = 4, auto_scale: bool = True,
                 shared_scale: bool = True, bg_sigma: float = DEFAULT_BG_SIGMA):
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
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 2:
        frames = frames[None]
    N = frames.shape[0]
    if N == 0:
        return []

    factor = 1.0
    if auto_scale:
        # One estimate for the whole stack (shared physical disk size).
        _ref, factor = scale_to_canonical(frames[0])

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

    x = torch.from_numpy(stack[:, None]).to(device)
    hm, off = model(x)
    md = max(2, int(round(min_distance * factor)))
    res = decode_batch(hm, off, thresh=thresh, min_distance=md)
    per_frame = split_by_batch(res, N)
    if factor != 1.0:
        for i, p in enumerate(per_frame):
            if len(p):
                p = p.copy()
                p[:, :2] = p[:, :2] / factor
                per_frame[i] = p
    return per_frame


@torch.no_grad()
def calibrate(model, sample_frames, device, tune_threshold: bool = True):
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
    #    the upscale-only factor.
    _ref, factor = scale_to_canonical(frames[0])
    work0 = _ref if factor != 1.0 else frames[0]

    # 2) bg_sigma: confidence-max over the sample (median best-sigma is robust).
    sigmas = []
    confs = []
    for f in frames:
        w = f
        if factor != 1.0:
            from scipy.ndimage import zoom
            w = zoom(f, factor, order=1)
        s, c = auto_bg_sigma(model, w, device)
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

"""find_vectors_neural.py — the SpotUNet neural disk detector as a find-vectors method.

A third detection method alongside NXCORR and DoG (see ``find_vectors.py``). The
network (``spyde.models``) is a parameter-free detector: it auto-estimates the
disk size, runs a small U-Net, and returns subpixel spot positions + a confidence
score, with its own local-max NMS baked into the decode. So this module is thin —
it adapts the model's ``(N,3) [y,x,score]`` output to the find-vectors
``(corr_map, raw_response, peaks)`` contract, applies the beam-stop rejection the
same way the other methods do, and (critically) replaces the confidence column
with the raw disk-mean frame intensity every method stores.

GPU/CPU: the batch path runs the whole nav chunk through one forward pass on the
torch GPU when available (``torch_gpu_device()``), per-frame on CPU otherwise —
mirroring ``_find_vectors_chunk_dog``. The model itself is loaded once and cached
by the registry, so a 13k-pattern scan loads the net a single time.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# Heatmap confidence below which detections are discarded (the model's natural,
# dataset-independent operating point). Mirrored in find_vectors.DEFAULT_NEURAL_THRESHOLD.
DEFAULT_NEURAL_THRESHOLD = 0.3
# Footprint (px) over which the raw disk-mean intensity is averaged for the value
# column. The model works at a ~20 px canonical disk; ~5 px in native space is a
# robust, method-agnostic default (matches the NXCORR/DoG kernel scale).
_INTENSITY_RADIUS = 5.0


def _apply_beamstop(peaks_yx: np.ndarray, beamstop_mask: Optional[np.ndarray],
                    shape) -> np.ndarray:
    """Drop detections whose nearest pixel falls inside the beam-stop mask. ``peaks_yx``
    is (N, >=2) with float [y, x] in the first two columns; returns the kept rows."""
    if beamstop_mask is None or peaks_yx.size == 0 or not beamstop_mask.any():
        return peaks_yx
    H, W = shape
    iy = np.clip(np.rint(peaks_yx[:, 0]).astype(np.intp), 0, H - 1)
    ix = np.clip(np.rint(peaks_yx[:, 1]).astype(np.intp), 0, W - 1)
    keep = ~beamstop_mask[iy, ix]
    return peaks_yx[keep]


def _heatmap_response(frame_shape, peaks_yx, threshold) -> np.ndarray:
    """A sparse confidence map for the "show transform" preview toggle: the model's
    per-peak score painted at each detected pixel. (The full dense heatmap isn't
    returned by ``detect``; this is enough for the overlay, which only needs to show
    where/how strongly the net fired.)"""
    resp = np.zeros(frame_shape, dtype=np.float32)
    if peaks_yx.size:
        iy = np.clip(np.rint(peaks_yx[:, 0]).astype(np.intp), 0, frame_shape[0] - 1)
        ix = np.clip(np.rint(peaks_yx[:, 1]).astype(np.intp), 0, frame_shape[1] - 1)
        resp[iy, ix] = peaks_yx[:, 2]
    return resp


def _find_vectors_single_frame_neural(
    frame: np.ndarray,
    threshold: float = DEFAULT_NEURAL_THRESHOLD,
    min_distance: int = 4,
    *,
    subpixel: bool = True,          # accepted for signature parity; the net always
                                    # emits subpixel offsets (no integer-only mode).
    beamstop_mask: Optional[np.ndarray] = None,
    model_id: Optional[str] = None,
    bg_sigma: Optional[float] = None,  # local-norm high-pass scale (from calibrate());
                                       # None → auto (12, size-scaled for big disks).
    spot_radius: Optional[float] = None,   # user Spot-size (px radius) override for
                                           # the canonical rescale; None → auto.
):
    """Neural detector for one diffraction pattern.

    Returns ``(corr_map, raw_response, peaks)`` mirroring
    :func:`find_vectors._find_vectors_single_frame_dog`. ``peaks`` is ``(N, 3)``
    float32 ``[ky, kx, raw_intensity]`` (the model's position with its confidence
    replaced by the robust disk-mean frame intensity)."""
    from spyde import models
    from spyde.actions.find_vectors import _disk_mean_intensity

    f = np.asarray(frame, dtype=np.float32)
    model, device = models.get_model(model_id)
    pred = models.detect(model, f, device, thresh=float(threshold),
                         min_distance=int(min_distance),
                         bg_sigma=(float(bg_sigma) if bg_sigma is not None else None),
                         spot_diameter=(2.0 * spot_radius) if spot_radius else None)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 3)
    pred = _apply_beamstop(pred, beamstop_mask, f.shape)

    raw_response = _heatmap_response(f.shape, pred, threshold)
    corr_map = raw_response  # already thresholded (only kept peaks are painted)

    if pred.size == 0:
        return corr_map, raw_response, np.zeros((0, 3), dtype=np.float32)

    pos = pred[:, :2]
    intens = _disk_mean_intensity(f, pos[:, 0], pos[:, 1], _INTENSITY_RADIUS)
    peaks = np.column_stack([pos, intens]).astype(np.float32)
    return corr_map, raw_response, peaks


def _persistence_filter(peaks_grid, ny, nx, tol=3.0, min_neighbors=2):
    """Drop EXTRANEOUS (non-persistent) peaks: a real diffraction peak recurs at ~the
    same position in adjacent real-space scan positions; a false positive does not.

    ``peaks_grid``: list of (Ni,3) arrays in flat (iy*nx+ix) order. For each frame,
    keep only peaks that have a match within ``tol`` px in >= ``min_neighbors`` of its
    4-connected scan neighbours. Frames on the block edge use whatever neighbours
    exist (>=2 still required, so block-corner frames with <2 neighbours are left
    unfiltered to avoid dropping real peaks for lack of evidence)."""
    filtered = []
    for idx, peaks in enumerate(peaks_grid):
        iy, ix = divmod(idx, nx)
        nbr_idx = [(iy + dy) * nx + (ix + dx)
                   for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                   if 0 <= iy + dy < ny and 0 <= ix + dx < nx]
        if len(peaks) == 0 or len(nbr_idx) < min_neighbors:
            filtered.append(peaks)
            continue
        nbrs = [peaks_grid[j] for j in nbr_idx]
        keep = []
        for p in peaks:
            hits = sum(1 for nb in nbrs
                       if len(nb) and np.min(np.hypot(nb[:, 0] - p[0],
                                                      nb[:, 1] - p[1])) <= tol)
            if hits >= min_neighbors:
                keep.append(p)
        filtered.append(np.asarray(keep, np.float32).reshape(-1, 3)
                        if keep else np.zeros((0, 3), np.float32))
    return filtered


def _refine_block(peaks_grid, ny, nx, flat, beamstop_mask):
    """Stage-2 refinement of a greedy (low-threshold) candidate grid -> real vectors.

    Reduces over-detected candidates by encoding the physical constraints (overlap
    merge + scan-neighbour PERSISTENCE + soft FRIEDEL), using the block's own scan
    neighbours. This is the 'refine' half of propose-then-refine: stage 1 ran the
    model at a low threshold (high recall); this drops false positives while keeping
    faint real peaks. ``flat`` is the (Nframes,H,W) blurred stack (for the frame
    centre / shape)."""
    from spyde.models import refine as _refine_mod

    H, W = flat.shape[1], flat.shape[2]
    out = []
    for idx, peaks in enumerate(peaks_grid):
        iy, ix = divmod(idx, nx)
        nbr_idx = [(iy + dy) * nx + (ix + dx)
                   for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                   if 0 <= iy + dy < ny and 0 <= ix + dx < nx]
        if len(peaks) == 0 or len(nbr_idx) < 2:
            out.append(peaks)            # too few neighbours -> don't risk dropping real
            continue
        nbrs = [peaks_grid[j] for j in nbr_idx]
        out.append(_refine_mod.refine(
            peaks, nbrs, frame_shape=(H, W),
            w_conf=0.2, w_persist=1.0, w_friedel=0.2,
            keep_score=0.55, min_persist=0.5))
    return out


def _neural_block(b4d, threshold, min_dist, subpixel, beamstop_mask, model_id,
                  bg_sigma=None, persistence=False, spot_radius=None):
    """Run the neural detector on a (ny, nx, KY, KX) block → NaN-padded
    (ny, nx, MAX_PEAKS, 3). Batches the whole block through the torch GPU when
    available (internally sub-batched by ``detect_batch``, see infer.py, so a
    1000+ frame nav chunk never allocates activations for more than
    ``SPYDE_NEURAL_BATCH`` frames at once); per-frame CPU otherwise. ``bg_sigma``
    is the calibrated local-norm high-pass scale (see calibrate_neural).
    ``persistence`` drops extraneous (non-neighbour-confirmed) peaks (uses the
    block's scan neighbours).

    GPU use honours the SPYDE_FV_GPU worker policy (``_gpu_task_allowed``), with a
    neural-specific unset-default of "2": two CUDA-submitting workers keep the
    device fed while the rest run the per-frame CPU path. Real-data A/B
    (2026-07-16, 48-core/1-GPU box): all-workers CUDA was SLOWER (context
    thrash) and lagged the desktop; 2 workers was faster and smooth. Must match
    orchestrate's lane_mode. Set SPYDE_FV_GPU=one/N/all/off to override. The
    forward pass itself is additionally bounded by ``_gpu_slots()``
    (SPYDE_FV_GPU_CONC, default 2) — the same device-concurrency semaphore the
    numba NXCORR path uses — so concurrent CUDA-submitting workers don't each
    hold a full activation set on the device simultaneously."""
    from spyde import models
    from spyde.actions.find_vectors import MAX_PEAKS, _with_raw_intensity
    from spyde.actions.find_vectors.gpu_runtime import _gpu_slots, _gpu_task_allowed
    from spyde.actions.find_vectors_torch import torch_gpu_device

    out = np.full((b4d.shape[0], b4d.shape[1], MAX_PEAKS, 3), np.nan, dtype=np.float32)
    flat = b4d.reshape(-1, b4d.shape[2], b4d.shape[3]).astype(np.float32, copy=False)

    model, device = models.get_model(model_id)

    peaks_list = None
    try:
        if torch_gpu_device() is not None and _gpu_task_allowed(default_mode="4"):
            # Bound concurrent forward passes per process (same semaphore the
            # numba NXCORR path uses — chunk.py's _device_section — so a mixed
            # run shares one device-concurrency budget, SPYDE_FV_GPU_CONC).
            with _gpu_slots():
                # Sub-batched internally (SPYDE_NEURAL_BATCH) — NOT one forward
                # pass for the whole chunk; see infer.detect_batch.
                raw = models.detect_batch(
                    model, flat, device, thresh=float(threshold),
                    min_distance=int(min_dist),
                    bg_sigma=(float(bg_sigma) if bg_sigma is not None else None),
                    spot_diameter=(2.0 * spot_radius) if spot_radius else None)
            peaks_list = []
            for i, p in enumerate(raw):
                p = _apply_beamstop(np.asarray(p, np.float32).reshape(-1, 3),
                                    beamstop_mask, flat[i].shape)
                # Model col 2 is confidence → replace with raw disk-mean intensity.
                peaks_list.append(_with_raw_intensity(flat[i], p, radius=_INTENSITY_RADIUS))
    except Exception as _e:
        log.warning("[find_vectors] torch neural GPU path failed (%s); CPU per-frame", _e)
        peaks_list = None

    if peaks_list is None:
        peaks_list = [
            _find_vectors_single_frame_neural(
                frame, threshold, min_dist, subpixel=subpixel,
                beamstop_mask=beamstop_mask, model_id=model_id, bg_sigma=bg_sigma,
                spot_radius=spot_radius)[2]
            for frame in flat
        ]

    if persistence:
        peaks_list = _refine_block(peaks_list, b4d.shape[0], b4d.shape[1],
                                   flat, beamstop_mask)

    for i, peaks in enumerate(peaks_list):
        iy, ix = divmod(i, b4d.shape[1])
        n = min(len(peaks), MAX_PEAKS)
        if n > 0:
            out[iy, ix, :n, :] = peaks[:n]
    return out


def _find_vectors_chunk_neural(
    ghost_block, depth_px, nav_dim, sigma,
    threshold, min_dist, subpixel, beamstop_mask, model_id=None, bg_sigma=None,
    persistence=False, spot_radius=None,
):
    """Neural variant of ``_find_vectors_chunk``: nav-blur + ghost-trim (shared with
    the other methods), then the neural detector per frame (GPU-batched when torch
    CUDA/MPS is present). Same output structure as the NXCORR/DoG chunk fns —
    ``(nav_y, nav_x, MAX_PEAKS, 3)`` (4D) or ``(t, ..., MAX_PEAKS, 3)`` (5D)."""
    import time

    from spyde.actions.find_vectors import MAX_PEAKS, _nav_blur_trim

    t_start = time.perf_counter()
    blurred = _nav_blur_trim(ghost_block, depth_px, nav_dim, sigma)
    nav_shape = blurred.shape[:nav_dim]
    ny, nx = nav_shape[-2:]
    if nav_dim == 2:
        result = _neural_block(blurred, threshold, min_dist, subpixel,
                               beamstop_mask, model_id, bg_sigma, persistence,
                               spot_radius)
        core_shape = result.shape[:2]
    else:
        n_lead = nav_shape[0]
        out = np.full((n_lead, ny, nx, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        for t in range(n_lead):
            out[t] = _neural_block(blurred[t], threshold, min_dist, subpixel,
                                   beamstop_mask, model_id, bg_sigma, persistence,
                                   spot_radius)
        result = out
        core_shape = (n_lead, ny, nx)
    log.debug("[find_vectors] neural chunk core=%s total=%.0fms",
              tuple(int(s) for s in core_shape), (time.perf_counter() - t_start) * 1e3)
    return result


def calibrate_neural(sample_frames, *, sigma=1.0, model_id=None, spot_radius=None):
    """One-time calibration on a few representative frames before a full-scan run.

    All frames in a scan share disk size + background character, so this optimises the
    dataset-dependent knobs ONCE (hands-off) and the result is reused across the scan:
    ``bg_sigma`` (confidence-max high-pass — handles diffuse/beam-stop backgrounds),
    the scale factor, and a lowered ``threshold`` for faint-peak datasets. Frames
    should be nav-blurred the same way the compute will be (pass already-blurred frames
    or rely on the model's own robustness — here we blur lightly per frame for parity).

    Returns the dict from ``models.calibrate`` (keys: bg_sigma, thresh, scale_factor,
    confidence) — pass bg_sigma/thresh into the compute call.
    """
    from scipy.ndimage import gaussian_filter

    from spyde import models

    model, device = models.get_model(model_id)
    frames = []
    for f in sample_frames:
        f = np.asarray(f, np.float32)
        # light single-frame smoothing as a stand-in for nav-blur on isolated samples
        frames.append(gaussian_filter(f, 0.6) if sigma else f)
    cal = models.calibrate(model, frames, device,
                           spot_diameter=(2.0 * spot_radius) if spot_radius else None)
    log.info("[find_vectors] neural calibration: bg_sigma=%.1f thresh=%.2f "
             "scale=%.2f conf=%.2f", cal["bg_sigma"], cal["thresh"],
             cal["scale_factor"], cal.get("confidence", float("nan")))
    return cal

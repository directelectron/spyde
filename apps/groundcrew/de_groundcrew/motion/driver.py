"""
driver.py — the Qt-free counterpart of Ground Crew's motion QThread workers.

This is GLUE, not numerics. Everything that computes lives in
`de_groundcrew.external.gc_motion`, vendored verbatim from the upstream repo;
this module only turns a `QThread` + `Signal` into a call with `progress` and
`should_cancel` callbacks, which is exactly the swap upstream's own
QThread-removal spec describes:

    submit(callable, *args) -> Future     # concurrent.futures
    job.cancel_event                       # cooperative, checked in-loop
    emit(message: dict)                    # neutral callback, no Qt

Two things here are load-bearing.

**The sign reconciliation.** `motion_correct_v3` returns ``shifts_y`` as its
internal ``sy`` but ``shifts_x`` as ``-sx`` (MotionCor3's sign convention). The
rest of Ground Crew — `local_motion`, `_apply_shifts_fullres` — uses v3's
INTERNAL convention, where both are applied as ``-shift``. So x must be
un-negated on the way out. Get this wrong and the alignment still runs, the
tests still pass, and the drift plot is mirrored in x. Upstream marks it
LOAD-BEARING; it is copied here with the same care.

**Fail-loud is a result, not an exception.** v3 assesses its own confidence and
reports `low_confidence` with a `failure_reason` rather than returning a
plausible-looking wrong answer. That verdict is passed straight through to the
UI: an implausible alignment must be refused visibly, not displayed.
"""
from __future__ import annotations

import logging
import os
from typing import Callable

import numpy as np

from de_groundcrew.external.gc_motion import _image_ops
from de_groundcrew.external.gc_motion._motion_correction_v3 import motion_correct_v3
from de_groundcrew.external.gc_motion._worker_extracts import (
    _DEFAULT_MODE, _MODE_PRESETS, _compute_log_fft)

log = logging.getLogger(__name__)

Progress = Callable[[str], None]
ShouldCancel = Callable[[], bool]

#: Alignment quality presets, upstream's names. "fast" caps the pyramid at 6 Å,
#: "fine" pushes to 3 Å.
MODES = tuple(_MODE_PRESETS)

#: The eight gain orientations, in upstream's order — the index is a stored
#: setting, so it must not be reordered.
ORIENTATION_LABELS = tuple(getattr(
    _image_ops, "ORIENTATION_LABELS",
    ("Identity", "Rot90", "Rot180", "Rot270",
     "FlipH", "FlipV", "Transpose", "Transverse")))


def _noop(_msg: str) -> None: ...
def _never() -> bool: return False


class Cancelled(Exception):
    """Raised when a caller's `should_cancel` asked for a stop."""


def log_fft(image: np.ndarray) -> np.ndarray:
    """SerialEM-style power spectrum, cropped to a centred square.

    The crop is why Thon rings render circular on a non-square sensor. Only
    the SPECTRUM is cropped; the real-space image is displayed separately and
    is unaffected.
    """
    return _compute_log_fft(np.asarray(image))


def apply_orientation(img: np.ndarray, idx: int) -> np.ndarray:
    return _image_ops._apply_orientation(np.asarray(img), int(idx))


def match_gain_to_frame(gain: np.ndarray, h: int, w: int) -> np.ndarray:
    return _image_ops._match_gain_to_frame(np.asarray(gain), int(h), int(w))


# ── Loading ───────────────────────────────────────────────────────────────────

def load_movie_stack(path: str) -> tuple[np.ndarray, dict]:
    """Load an MRC or TIFF movie stack as ``(stack, metadata)``.

    A single-frame file loads as ``(1, h, w)`` rather than 2-D, so everything
    downstream can assume the frame axis exists.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mrc", ".mrcs"):
        import mrcfile
        with mrcfile.open(path, permissive=True) as mrc:
            data = mrc.data
            if data is None:
                # Upstream fix: a truncated MRC yields None here, and indexing
                # it produced a bare NoneType crash rather than a diagnosis.
                raise ValueError(
                    f"{os.path.basename(path)} is truncated or unreadable — "
                    "the MRC contains no image data")
            stack = np.asarray(data)
    elif ext in (".tif", ".tiff"):
        import tifffile
        stack = np.asarray(tifffile.imread(path))
    else:
        raise ValueError(f"Unsupported movie format: {ext or path!r} "
                         "(expected .mrc, .mrcs, .tif or .tiff)")

    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D stack, got shape {stack.shape}")

    n, h, w = stack.shape
    return stack, {"n_frames": int(n), "height": int(h), "width": int(w),
                   "filename": os.path.basename(path), "path": path,
                   "dtype": str(stack.dtype)}


def load_gain(path: str) -> np.ndarray:
    stack, _ = load_movie_stack(path)
    gain = stack[0] if stack.shape[0] == 1 else stack.mean(axis=0)
    return np.asarray(gain, dtype=np.float32)


def save_image(image: np.ndarray, path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    arr = np.asarray(image, dtype=np.float32)
    if ext in (".mrc", ".mrcs"):
        import mrcfile
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(arr)
    elif ext in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(path, arr)
    else:
        raise ValueError(f"Unsupported output format: {ext or path!r}")
    return path


def rank_gain_orientations(frame: np.ndarray, gain: np.ndarray):
    """Score all eight orientations. Returns ``(scores, separation)``.

    Straight through to upstream's `_image_ops.rank_gain_orientations`, and
    deliberately NOT reimplemented: its metric is ``row_std + col_std`` on a
    binned frame, which cancels the readout structure two units of the same
    camera model share, and its thresholds were calibrated on 219 real gain
    pairings. A plausible-looking substitute — mean-relative spread, say —
    scores those cases wrong and nothing here would show it.

    Note upstream's warning: the tiers assume all EIGHT orientations were
    scored, because the separation is a median over the eight. Score fewer and
    the thresholds need re-measuring, not inheriting.
    """
    return _image_ops.rank_gain_orientations(
        np.asarray(frame, dtype=np.float32), np.asarray(gain, dtype=np.float32))


def classify_gain_tier(separation: float) -> str:
    """``"ok"`` / ``"weak"`` / ``"fail"`` for a separation ratio.

    A WEAK detection is worse than none — the app used to adopt whichever of
    eight near-identical scores sorted first, and a wrong gain orientation
    ruins every frame — so the tier is surfaced rather than reduced to a
    boolean.
    """
    return _image_ops.classify_gain_tier(float(separation))


# ── Alignment ─────────────────────────────────────────────────────────────────

def align_stack(stack: np.ndarray, *, gain: np.ndarray | None = None,
                gain_orientation: int = 0, apix: float = 1.0,
                mode: str = _DEFAULT_MODE, throw: int = 0,
                dose_weight=None, unaligned_sum: np.ndarray | None = None,
                params: dict | None = None,
                progress: Progress = _noop,
                should_cancel: ShouldCancel = _never) -> dict:
    """Whole-frame coarse-to-fine alignment (`motion_correct_v3`).

    The Qt-free equivalent of upstream's `MotionCorrectionWorker` +
    `_align_v3_adapter`. v3 owns its own binning schedule, so there is no
    bin_factor here — passing one would imply a control that does nothing.

    *throw* discards leading frames before aligning; early frames carry the
    beam-induced initial burst. At least two frames are always kept.
    """
    params = dict(params or {})
    full = np.asarray(stack)
    throw = max(0, min(int(throw), max(0, full.shape[0] - 2)))
    stack = full[throw:] if throw else full
    if stack.shape[0] < 2:
        raise ValueError(f"need at least 2 frames to align, got {stack.shape[0]}")

    cancelled = False

    def _progress(msg: str) -> None:
        # v3 has no cancel hook of its own, so the progress callback doubles as
        # one: raising out of it unwinds the solve at the next report. Crude,
        # but it is the only cooperative point upstream exposes, and it beats
        # letting a minutes-long fit run on after the user pressed Stop.
        nonlocal cancelled
        progress(msg)
        if should_cancel():
            cancelled = True
            raise Cancelled()

    preset = _MODE_PRESETS.get(mode, _MODE_PRESETS[_DEFAULT_MODE])
    try:
        result = motion_correct_v3(
            stack, gain=gain, gain_orientation=gain_orientation, apix=apix,
            dose_weight=dose_weight, progress_cb=_progress,
            fine_cap_apx=preset["fine_cap_apx"],
            **{k: params[k] for k in ("max_path_A", "max_per_frame_A")
               if k in params})
    except Cancelled:
        raise
    except Exception:
        if cancelled:
            raise Cancelled()
        raise

    # SIGN RECONCILIATION — load-bearing. See the module docstring: v3 returns
    # y in its internal convention but x negated, and the rest of the pipeline
    # wants both internal.
    shifts_y = list(result["shifts_y"])
    shifts_x = [-1.0 * v for v in result["shifts_x"]]

    aligned_sum = result["aligned_sum"]
    return {
        "aligned_sum": aligned_sum,
        "unaligned_sum": (unaligned_sum if unaligned_sum is not None
                          else result["unaligned_sum"]),
        "aligned_fft": log_fft(aligned_sum),
        "dw_sum": result.get("dw_sum"),
        # v3 emits raw per-frame shifts with no separate smoothed array (RELION
        # does the same), so raw aliases smooth rather than inventing a second
        # curve the solver never produced.
        "shifts_x_raw": shifts_x, "shifts_y_raw": shifts_y,
        "shifts_x_smooth": shifts_x, "shifts_y_smooth": shifts_y,
        "n_frames": int(result["n_frames"]),
        "bin_factor": int(result["bin_factor"]),
        "throw": throw,
        # Fail-loud: an implausible result is refused, not displayed.
        "low_confidence": bool(result.get("low_confidence", False)),
        "failure_reason": result.get("failure_reason", ""),
        "confidence_signals": result.get("confidence_signals", {}),
    }


def correct_local_motion(stack: np.ndarray, *, gain: np.ndarray | None = None,
                         gain_orientation: int = 0,
                         shifts_y=None, shifts_x=None,
                         bin_factor: int = 2, patch_size: int = 512,
                         throw: int = 0, dose_weight=None,
                         progress: Progress = _noop,
                         should_cancel: ShouldCancel = _never) -> dict:
    """Per-patch local motion (Phase 2), on Phase 1's output.

    The Qt-free equivalent of upstream's `LocalMotionWorker`, following its
    call sequence exactly — correlation on BINNED, globally-aligned frames;
    the polynomial field fitted in binned coordinates; compositing at full
    resolution with `shift_scale=bin_factor` to bring the field back up.

    Unlike Phase 1, this one keeps a `bin_factor`: v3 owns its own schedule,
    but the patch correlation here does not.
    """
    from de_groundcrew.external.gc_motion.local_motion import (
        apply_local_shifts, build_cosine_blend_weights, correlate_patches,
        fit_motion_field, generate_patch_grid, smooth_patch_shifts)
    from de_groundcrew.external.gc_motion._motion_correction_v2 import _bin_image
    from de_groundcrew.external.gc_motion._worker_extracts import _apply_shift_fourier

    full = np.asarray(stack)
    throw = max(0, min(int(throw), max(0, full.shape[0] - 2)))
    stack = full[throw:] if throw else full
    n_frames, fh, fw = stack.shape

    oriented_gain = None
    if gain is not None:
        oriented_gain = match_gain_to_frame(
            apply_orientation(gain, gain_orientation), fh, fw)

    def _cancelled() -> bool:
        return bool(should_cancel())

    def _check() -> None:
        if _cancelled():
            raise Cancelled()

    progress("Phase 2: binning frames for local CC…")
    bh = fh // bin_factor if bin_factor > 1 else fh
    bw = fw // bin_factor if bin_factor > 1 else fw
    binned = []
    for i in range(n_frames):
        _check()
        f = stack[i]
        binned.append((_bin_image(f, bin_factor) if bin_factor > 1 else f
                       ).astype(np.float32))

    progress("Phase 2: applying global shifts…")
    sy = np.asarray(shifts_y, dtype=np.float64) / bin_factor
    sx = np.asarray(shifts_x, dtype=np.float64) / bin_factor
    aligned_binned = []
    for i in range(n_frames):
        _check()
        aligned_binned.append(_apply_shift_fourier(binned[i], -sy[i], -sx[i]))
    reference = np.mean(aligned_binned, axis=0).astype(np.float32)

    # A CC patch below 128 px has too little signal for a reliable peak, so the
    # binned patch size has a floor regardless of the requested full-res size.
    ps_cc = max(patch_size // bin_factor, 128)
    patches_cc, centers_cc = generate_patch_grid(bh, bw, ps_cc, 0.5)
    progress(f"Phase 2: {len(patches_cc)} patches (ps={ps_cc} at bin {bin_factor})")

    local_shifts = correlate_patches(
        aligned_binned, reference, patches_cc, upsample_factor=10,
        progress_fn=lambda m: progress(f"Phase 2 — {m}"),
        cancel_fn=_cancelled)
    if local_shifts is None:
        raise Cancelled()

    progress("Phase 2: smoothing + fitting motion field…")
    smoothed = smooth_patch_shifts(local_shifts, outlier_sigma=3.0)
    coefficients, norm_params = fit_motion_field(centers_cc, smoothed, degree=3)

    patches_full, centers_full = generate_patch_grid(fh, fw, patch_size, 0.5)
    blend_weights = build_cosine_blend_weights(patch_size, 0.5)
    # The field was fitted in BINNED coordinates, so the full-res centres are
    # scaled down to match `norm_params` before evaluation.
    eval_centers = centers_full / bin_factor if bin_factor > 1 else centers_full

    progress("Phase 2: compositing at full resolution…")
    composited = apply_local_shifts(
        stack, oriented_gain, None,
        np.asarray(shifts_y, dtype=np.float64),
        np.asarray(shifts_x, dtype=np.float64),
        coefficients, norm_params,
        patches_full, eval_centers, blend_weights, degree=3,
        shift_scale=float(bin_factor) if bin_factor > 1 else 1.0,
        progress_fn=lambda m: progress(f"Phase 2 — {m}"),
        cancel_fn=_cancelled, dose_weight=dose_weight)
    if composited is None:
        raise Cancelled()
    corrected_sum, dw_sum = composited

    return {
        "corrected_sum": corrected_sum,
        "corrected_fft": log_fft(corrected_sum),
        "dw_sum": dw_sum,
        "coefficients": coefficients,
        "norm_params": norm_params,
        "centers_full": centers_full,
        "eval_centers": eval_centers,
        "n_patches": len(patches_full),
        "patch_size": patch_size,
        "ps_cc": ps_cc,
        "bin_factor": bin_factor,
    }

"""
local.py — patch-based local motion correction (the old app's Phase 2).

Ported from `workers/local_motion.py` and `LocalMotionWorker`. Whole-frame
alignment removes the rigid part of the drift; what remains is a motion field
that varies ACROSS the frame — the specimen does not move as a rigid body under
the beam. This estimates that field and composites a locally-corrected sum.

The shape of it:

1. tile the frame with 50%-overlapping patches
2. cross-correlate each patch against the same patch of the reference
3. reject per-frame outliers by MAD, then spline-smooth each patch trajectory
4. fit a degree-3 polynomial surface to the patch shifts, per frame
5. composite at full resolution, shifting each patch by the field evaluated at
   its centre and blending with a Hann window

Two details that are easy to get wrong:

**The division by the weight map is what makes the blend seamless — not the
window.** It is tempting to say "Hann windows at 50% overlap sum to 1, so the
blend reconstructs perfectly". That is true of the PERIODIC Hann;
``np.hanning`` is the SYMMETRIC one (endpoints exactly zero) and misses it by
about 1%. Dividing the accumulator by the accumulated weights is what actually
flattens it, and it also handles the frame edges, where fewer patches overlap.
Do not drop that division on the strength of the window's reputation.

**The polynomial is fitted on NORMALISED coordinates.** A degree-3 Vandermonde
in raw pixel coordinates on an 8192² frame has columns spanning 10^12; the fit
is numerically hopeless. Centres are standardised first, and the normalisation
travels with the coefficients — which is why `evaluate_motion_field` needs
`norm_params` and cannot be called without them.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline

from de_groundcrew.motion.align import (
    Cancelled, bandpass_filter, cross_correlate, log_fft)
from de_groundcrew.motion.frames import apply_orientation, bin_image, match_gain_to_frame

#: Local patch correlation is noisier than whole-frame, so it refines less far.
PATCH_UPSAMPLE = 10

#: Degree of the polynomial motion field. 10 terms in 2-D.
DEGREE = 3

Progress = Callable[[str], None]
ShouldCancel = Callable[[], bool]


def _noop(_msg: str) -> None: ...
def _never() -> bool: return False


def generate_patch_grid(height: int, width: int, patch_size: int = 512,
                        overlap_frac: float = 0.5):
    """Overlapping patch boxes and their centres.

    The last row and column are pulled back to the frame edge rather than
    dropped, so the grid always covers the whole frame even when the size is
    not a multiple of the stride.
    """
    stride = max(int(patch_size * (1.0 - overlap_frac)), 1)
    y_starts = list(range(0, height - patch_size + 1, stride))
    x_starts = list(range(0, width - patch_size + 1, stride))
    if not y_starts or y_starts[-1] + patch_size < height:
        y_starts.append(max(0, height - patch_size))
    if not x_starts or x_starts[-1] + patch_size < width:
        x_starts.append(max(0, width - patch_size))
    y_starts = list(dict.fromkeys(y_starts))
    x_starts = list(dict.fromkeys(x_starts))

    patches, centers = [], []
    for y0 in y_starts:
        for x0 in x_starts:
            patches.append((y0, x0, y0 + patch_size, x0 + patch_size))
            centers.append((y0 + patch_size / 2, x0 + patch_size / 2))
    return patches, np.array(centers, dtype=np.float64)


def build_cosine_blend_weights(patch_size: int) -> np.ndarray:
    """Separable 2-D Hann window.

    NOT a partition of unity — see the module docstring. `apply_local_shifts`
    divides by the accumulated weights, which is what flattens the composite.
    """
    w = np.hanning(patch_size).astype(np.float32)
    return np.outer(w, w)


def correlate_patches(frames, reference, patches, *,
                      upsample_factor: int = PATCH_UPSAMPLE,
                      progress: Progress = _noop,
                      should_cancel: ShouldCancel = _never) -> np.ndarray:
    """Per-patch shifts for every frame. Returns ``(n_frames, n_patches, 2)``.

    The bandpass low cut is higher than whole-frame alignment's (0.02 vs 0.01):
    a patch is a smaller window, so a given low frequency spans fewer cycles
    within it and contributes more spurious correlation.
    """
    n_frames = len(frames)
    n_patches = len(patches)
    ps = patches[0][2] - patches[0][0]
    bp = bandpass_filter((ps, ps), low_freq=0.02, high_freq=0.5).astype(np.float32)

    ref = np.asarray(reference, dtype=np.float32)
    ref_ffts = [np.fft.fft2(ref[y0:y1, x0:x1]) * bp
                for (y0, x0, y1, x1) in patches]

    out = np.zeros((n_frames, n_patches, 2), dtype=np.float64)
    for i in range(n_frames):
        if should_cancel():
            raise Cancelled()
        frame = np.asarray(frames[i], dtype=np.float32)
        for j, (y0, x0, y1, x1) in enumerate(patches):
            dy, dx = cross_correlate(
                None, frame[y0:y1, x0:x1], upsample_factor=upsample_factor,
                ref_fft=ref_ffts[j], bp_filter=bp)
            out[i, j] = (dy, dx)
        progress(f"Local correlation: frame {i + 1}/{n_frames}")
    return out


def smooth_patch_shifts(local_shifts: np.ndarray,
                        outlier_sigma: float = 3.0) -> np.ndarray:
    """Reject per-frame outlier patches by MAD, then spline-smooth in time.

    A patch that happens to land on empty ice or a support bar produces a
    meaningless correlation peak. Replacing it with the frame's median — rather
    than dropping it — keeps the grid rectangular for the polynomial fit.
    """
    n_frames, n_patches, _ = local_shifts.shape
    cleaned = local_shifts.copy()
    for axis in range(2):
        for f in range(n_frames):
            vals = cleaned[f, :, axis]
            med = np.median(vals)
            mad = max(float(np.median(np.abs(vals - med))), 1e-6)
            vals[np.abs(vals - med) > outlier_sigma * 1.4826 * mad] = med

    if n_frames < 4:
        return cleaned
    smoothed = np.zeros_like(cleaned)
    t = np.arange(n_frames, dtype=np.float64)
    for j in range(n_patches):
        for axis in range(2):
            smoothed[:, j, axis] = CubicSpline(t, cleaned[:, j, axis])(t)
    return smoothed


def _vandermonde(centers: np.ndarray, degree: int = DEGREE) -> np.ndarray:
    """2-D polynomial basis: [1, y, x, y², yx, x², y³, y²x, yx², x³] for degree 3."""
    cy, cx = centers[:, 0], centers[:, 1]
    cols = []
    for p in range(degree + 1):
        for qy in range(p, -1, -1):
            cols.append((cy ** qy) * (cx ** (p - qy)))
    return np.column_stack(cols)


def fit_motion_field(centers: np.ndarray, smoothed_shifts: np.ndarray,
                     degree: int = DEGREE):
    """Least-squares polynomial surface per frame.

    Returns ``(coefficients, norm_params)`` where coefficients is
    ``(n_frames, 2, n_terms)``. The normalisation is NOT cosmetic — see the
    module docstring — and must be carried to `evaluate_motion_field`.
    """
    n_frames = smoothed_shifts.shape[0]
    cy_mean, cx_mean = centers.mean(axis=0)
    cy_std = max(float(centers[:, 0].std()), 1.0)
    cx_std = max(float(centers[:, 1].std()), 1.0)

    A = _vandermonde(np.column_stack([(centers[:, 0] - cy_mean) / cy_std,
                                      (centers[:, 1] - cx_mean) / cx_std]), degree)
    coeffs = np.zeros((n_frames, 2, A.shape[1]), dtype=np.float64)
    for f in range(n_frames):
        for axis in range(2):
            coeffs[f, axis, :] = np.linalg.lstsq(
                A, smoothed_shifts[f, :, axis], rcond=None)[0]
    return coeffs, (cy_mean, cx_mean, cy_std, cx_std)


def evaluate_motion_field(coefficients: np.ndarray, norm_params: tuple,
                          points: np.ndarray, degree: int = DEGREE) -> np.ndarray:
    """Evaluate one frame's field at arbitrary ``(y, x)`` points → ``(N, 2)``."""
    cy_mean, cx_mean, cy_std, cx_std = norm_params
    A = _vandermonde(np.column_stack([(points[:, 0] - cy_mean) / cy_std,
                                      (points[:, 1] - cx_mean) / cx_std]), degree)
    return np.column_stack([A @ coefficients[0], A @ coefficients[1]])


def apply_local_shifts(stack, gain, global_shifts_y, global_shifts_x,
                       coefficients, norm_params, patches, centers,
                       blend_weights, *, degree: int = DEGREE,
                       shift_scale: float = 1.0,
                       progress: Progress = _noop,
                       should_cancel: ShouldCancel = _never) -> np.ndarray:
    """Composite the locally-corrected sum at full resolution.

    The global and local shifts are COMBINED and applied once per patch, rather
    than shifting the whole frame and then each patch. On an 8192² frame that
    replaces a 67-megapoint FFT per frame with a handful of small ones.
    """
    n_frames, fh, fw = stack.shape
    ps = patches[0][2] - patches[0][0]

    weight_sum = np.zeros((fh, fw), dtype=np.float32)
    for y0, x0, y1, x1 in patches:
        weight_sum[y0:y1, x0:x1] += blend_weights
    weight_sum = np.maximum(weight_sum, 1e-6)

    fy = np.fft.fftfreq(ps).astype(np.float32).reshape(-1, 1)
    fx = np.fft.fftfreq(ps).astype(np.float32).reshape(1, -1)
    twopi_j = np.float64(-2.0 * np.pi) * 1j

    accum = np.zeros((fh, fw), dtype=np.float32)
    for i in range(n_frames):
        if should_cancel():
            raise Cancelled()
        frame = stack[i].astype(np.float32)
        if gain is not None:
            frame = frame * gain

        shifts = evaluate_motion_field(coefficients[i], norm_params,
                                       np.asarray(centers), degree)
        if shift_scale != 1.0:
            shifts = shifts * shift_scale
        if global_shifts_y is not None:
            shifts[:, 0] += float(-global_shifts_y[i])
            shifts[:, 1] += float(-global_shifts_x[i])

        for j, (y0, x0, y1, x1) in enumerate(patches):
            dy, dx = float(shifts[j, 0]), float(shifts[j, 1])
            patch = frame[y0:y1, x0:x1].copy()
            if abs(dy) > 1e-6 or abs(dx) > 1e-6:
                phase = np.exp(np.complex64(twopi_j) * (fy * dy + fx * dx))
                patch = np.real(np.fft.ifft2(
                    np.fft.fft2(patch) * phase)).astype(np.float32)
            accum[y0:y1, x0:x1] += patch * blend_weights
        progress(f"Local correction: frame {i + 1}/{n_frames}")

    return accum / weight_sum / n_frames


def correct_local_motion(stack: np.ndarray, *, gain: np.ndarray | None = None,
                         gain_orientation: int = 0,
                         shifts_y=None, shifts_x=None,
                         bin_factor: int = 2, patch_size: int = 512,
                         throw: int = 0,
                         progress: Progress = _noop,
                         should_cancel: ShouldCancel = _never) -> dict:
    """Phase 2 end to end, given Phase 1's full-resolution shifts.

    Patch correlation runs on BINNED frames (cheap), the field is fitted in
    binned coordinates, and compositing happens at full resolution — hence
    `shift_scale=bin_factor` when the field is evaluated for the final sum.
    """
    full = np.asarray(stack)
    throw = max(0, min(int(throw), max(0, full.shape[0] - 2)))
    stack = full[throw:] if throw else full
    n_frames, fh, fw = stack.shape

    oriented_gain = None
    if gain is not None:
        oriented_gain = match_gain_to_frame(
            apply_orientation(gain, gain_orientation), fh, fw)

    progress("Binning frames for local correlation…")
    binned = []
    for i in range(n_frames):
        if should_cancel():
            raise Cancelled()
        f = stack[i]
        binned.append((bin_image(f, bin_factor) if bin_factor > 1 else f
                       ).astype(np.float32))

    # Apply Phase 1's shifts at BINNED scale, so what remains for the patches
    # to find is only the non-rigid part.
    from de_groundcrew.motion.align import apply_shift_fourier
    if shifts_y is not None:
        scale = 1.0 / (float(bin_factor) if bin_factor > 1 else 1.0)
        binned = [apply_shift_fourier(b, -shifts_y[i] * scale, -shifts_x[i] * scale)
                  for i, b in enumerate(binned)]

    bh, bw = binned[0].shape
    bin_patch = max(int(patch_size // max(bin_factor, 1)), 32)
    if bin_patch > min(bh, bw):
        bin_patch = int(min(bh, bw))

    patches, centers = generate_patch_grid(bh, bw, patch_size=bin_patch)
    progress(f"{len(patches)} patches of {bin_patch}px…")

    reference = np.mean(binned, axis=0).astype(np.float32)
    raw = correlate_patches(binned, reference, patches,
                            progress=progress, should_cancel=should_cancel)
    smoothed = smooth_patch_shifts(raw)
    coefficients, norm_params = fit_motion_field(centers, smoothed)

    # Full-resolution patch grid, matching the binned one one-for-one.
    full_patches, full_centers = generate_patch_grid(
        fh, fw, patch_size=int(bin_patch * max(bin_factor, 1)))
    blend = build_cosine_blend_weights(full_patches[0][2] - full_patches[0][0])

    # Evaluate the field in the coordinates it was FITTED in (binned), so the
    # centres are divided back down before evaluation.
    eval_centers = np.asarray(full_centers) / float(max(bin_factor, 1))

    progress("Compositing locally-corrected sum…")
    corrected = apply_local_shifts(
        stack, oriented_gain, shifts_y, shifts_x, coefficients, norm_params,
        full_patches, eval_centers, blend,
        shift_scale=float(max(bin_factor, 1)),
        progress=progress, should_cancel=should_cancel)

    return {
        "corrected_sum": corrected,
        "corrected_fft": log_fft(corrected),
        "coefficients": coefficients,
        "norm_params": norm_params,
        "centers_full": np.asarray(full_centers),
        "eval_centers": eval_centers,
        "n_patches": len(full_patches),
        "patch_size": full_patches[0][2] - full_patches[0][0],
        "bin_factor": bin_factor,
    }

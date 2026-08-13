"""
Per-patch local motion correction for S.T.A.C.K.

Phase 2: After whole-frame alignment (Phase 1), this module refines
shifts on a per-patch basis and applies them with cosine-blend
compositing for seamless reconstruction.

References:
  - MotionCor2: Zheng et al., Nature Methods 14, 331–332 (2017)
  - RELION: Zivanov et al., eLife 7, e42166 (2018)
"""

import numpy as np
from scipy.interpolate import CubicSpline

try:
    import cupy as _cp
    _cp.cuda.Device(0).compute_capability
    _GPU = True
except Exception:
    _GPU = False


# ═════════════════════════════════════════════════════════════════
# Patch grid generation
# ═════════════════════════════════════════════════════════════════

def generate_patch_grid(height: int, width: int,
                        patch_size: int = 512,
                        overlap_frac: float = 0.5):
    """Generate a grid of overlapping patches.

    Returns
    -------
    patches : list of (y0, x0, y1, x1)
        Patch bounding boxes.
    centers : ndarray, shape (N, 2)
        Patch center coordinates (cy, cx).
    """
    stride = int(patch_size * (1.0 - overlap_frac))
    stride = max(stride, 1)

    patches = []
    centers = []

    # Generate start positions
    y_starts = list(range(0, height - patch_size + 1, stride))
    x_starts = list(range(0, width - patch_size + 1, stride))

    # Ensure we cover the full image
    if not y_starts or y_starts[-1] + patch_size < height:
        y_starts.append(max(0, height - patch_size))
    if not x_starts or x_starts[-1] + patch_size < width:
        x_starts.append(max(0, width - patch_size))

    # Remove duplicates while preserving order
    y_starts = list(dict.fromkeys(y_starts))
    x_starts = list(dict.fromkeys(x_starts))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = y0 + patch_size
            x1 = x0 + patch_size
            patches.append((y0, x0, y1, x1))
            centers.append((y0 + patch_size / 2, x0 + patch_size / 2))

    return patches, np.array(centers, dtype=np.float64)


def build_cosine_blend_weights(patch_size: int, overlap_frac: float = 0.5):
    """Build 2-D Hann-window blend weights for overlap-add compositing.

    The weights taper from 1.0 at center to 0.0 at edges with a
    cosine profile. With 50% overlap, overlapping Hann windows sum
    to exactly 1.0 (perfect reconstruction).

    Returns float32 array of shape (patch_size, patch_size).
    """
    # 1-D Hann window
    w = np.hanning(patch_size).astype(np.float32)
    # 2-D separable: outer product
    return np.outer(w, w)


# ═════════════════════════════════════════════════════════════════
# Per-patch cross-correlation
# ═════════════════════════════════════════════════════════════════

def correlate_patches(frames, reference, patches, upsample_factor=10,
                      progress_fn=None, cancel_fn=None):
    """Estimate local shifts for every patch in every frame.

    Uses batched FFT on GPU for performance: all patches from one frame
    are extracted, FFT'd, and cross-correlated in parallel.

    Parameters
    ----------
    frames : list of ndarray (H, W) float32
        Globally-aligned binned frames.
    reference : ndarray (H, W) float32
        Reference image (average of globally-aligned frames).
    patches : list of (y0, x0, y1, x1)
        Patch grid from generate_patch_grid.
    upsample_factor : int
        Subpixel refinement factor.
    progress_fn : callable(str) or None
    cancel_fn : callable() -> bool or None

    Returns
    -------
    local_shifts : ndarray, shape (n_frames, n_patches, 2)
        Per-patch (dy, dx) shifts for each frame.
    """
    from de_groundcrew.external.gc_motion._worker_extracts import (
        _cross_correlate, _bandpass_filter,
    )

    n_frames = len(frames)
    n_patches = len(patches)
    local_shifts = np.zeros((n_frames, n_patches, 2), dtype=np.float64)

    ps = patches[0][2] - patches[0][0]
    bp = _bandpass_filter((ps, ps), low_freq=0.02, high_freq=0.5)

    # Pre-compute patch coordinate arrays
    p_y0 = np.array([p[0] for p in patches], dtype=np.intp)
    p_x0 = np.array([p[1] for p in patches], dtype=np.intp)

    if _GPU:
        bp_gpu = _cp.asarray(bp, dtype=_cp.float32)

        # Pre-compute reference patch FFTs: upload full ref → extract on GPU
        ref_gpu = _cp.asarray(reference, dtype=_cp.float32)
        ref_batch = _cp.zeros((n_patches, ps, ps), dtype=_cp.float32)
        for j in range(n_patches):
            ref_batch[j] = ref_gpu[p_y0[j]:p_y0[j]+ps, p_x0[j]:p_x0[j]+ps]
        ref_ffts_gpu = _cp.fft.fft2(ref_batch, axes=(-2, -1))
        ref_ffts_gpu *= bp_gpu[None, :, :]
        del ref_batch, ref_gpu  # free VRAM

        # Pre-allocate patch extraction buffer
        patch_buf = _cp.zeros((n_patches, ps, ps), dtype=_cp.float32)
        idx_arr = _cp.arange(n_patches)
        half_ps = ps // 2

        for i in range(n_frames):
            if cancel_fn and cancel_fn():
                return None

            # Upload full frame to GPU, extract patches there
            # (avoids CPU extraction loop + separate CPU→GPU patch transfer)
            frame_gpu = _cp.asarray(frames[i], dtype=_cp.float32)
            for j in range(n_patches):
                patch_buf[j] = frame_gpu[p_y0[j]:p_y0[j]+ps,
                                         p_x0[j]:p_x0[j]+ps]

            # Batch FFT + bandpass + CC
            frame_ffts = _cp.fft.fft2(patch_buf, axes=(-2, -1))
            frame_ffts *= bp_gpu[None, :, :]
            CC_batch = _cp.conj(ref_ffts_gpu) * frame_ffts
            cc_batch = _cp.real(_cp.fft.ifft2(CC_batch, axes=(-2, -1)))

            # Find integer peaks
            cc_flat = cc_batch.reshape(n_patches, -1)
            peak_flat = _cp.argmax(cc_flat, axis=1)
            peak_y_raw = peak_flat // ps
            peak_x_raw = peak_flat % ps

            if upsample_factor <= 1:
                # Wrap-around + transfer to CPU
                py = peak_y_raw.get().astype(np.float64)
                px = peak_x_raw.get().astype(np.float64)
                py[py > half_ps] -= ps
                px[px > half_ps] -= ps
                local_shifts[i, :, 0] = py
                local_shifts[i, :, 1] = px
            else:
                # Batch subpixel via 3-point parabola (all on GPU)
                py_m1 = (peak_y_raw - 1) % ps
                py_p1 = (peak_y_raw + 1) % ps
                px_m1 = (peak_x_raw - 1) % ps
                px_p1 = (peak_x_raw + 1) % ps

                cc_center = cc_batch[idx_arr, peak_y_raw, peak_x_raw]
                cc_ym1 = cc_batch[idx_arr, py_m1, peak_x_raw]
                cc_yp1 = cc_batch[idx_arr, py_p1, peak_x_raw]
                cc_xm1 = cc_batch[idx_arr, peak_y_raw, px_m1]
                cc_xp1 = cc_batch[idx_arr, peak_y_raw, px_p1]

                denom_y = 2.0 * (cc_ym1 - 2.0 * cc_center + cc_yp1)
                denom_x = 2.0 * (cc_xm1 - 2.0 * cc_center + cc_xp1)

                safe_y = _cp.abs(denom_y) > 1e-10
                safe_x = _cp.abs(denom_x) > 1e-10

                sub_y = _cp.where(
                    safe_y,
                    (cc_ym1 - cc_yp1) / _cp.where(safe_y, denom_y, _cp.ones_like(denom_y)),
                    _cp.zeros_like(denom_y),
                )
                sub_x = _cp.where(
                    safe_x,
                    (cc_xm1 - cc_xp1) / _cp.where(safe_x, denom_x, _cp.ones_like(denom_x)),
                    _cp.zeros_like(denom_x),
                )
                sub_y = _cp.clip(sub_y, -0.5, 0.5)
                sub_x = _cp.clip(sub_x, -0.5, 0.5)

                # Single GPU→CPU transfer at end of frame
                # Wrap-around on GPU before transfer
                peak_y_wrap = _cp.where(peak_y_raw > half_ps,
                                        peak_y_raw.astype(_cp.float32) - ps,
                                        peak_y_raw.astype(_cp.float32))
                peak_x_wrap = _cp.where(peak_x_raw > half_ps,
                                        peak_x_raw.astype(_cp.float32) - ps,
                                        peak_x_raw.astype(_cp.float32))

                local_shifts[i, :, 0] = (peak_y_wrap + sub_y).get()
                local_shifts[i, :, 1] = (peak_x_wrap + sub_x).get()

            if progress_fn:
                progress_fn(f"Local CC: frame {i+1}/{n_frames}")
    else:
        # CPU path: same but with numpy
        bp_cpu = bp.astype(np.float32)

        ref_ffts = []
        for y0, x0, y1, x1 in patches:
            ref_patch = reference[y0:y1, x0:x1].astype(np.float32)
            ref_ffts.append(np.fft.fft2(ref_patch) * bp_cpu)

        for i in range(n_frames):
            if cancel_fn and cancel_fn():
                return None
            frame = frames[i]
            for j, (y0, x0, y1, x1) in enumerate(patches):
                patch = frame[y0:y1, x0:x1].astype(np.float32)
                frame_fft = np.fft.fft2(patch) * bp_cpu

                dy, dx = _cross_correlate(
                    None, None,
                    upsample_factor=upsample_factor,
                    bandpass=False,
                    ref_fft=ref_ffts[j],
                    frame_fft=frame_fft,
                )
                local_shifts[i, j, 0] = dy
                local_shifts[i, j, 1] = dx

            if progress_fn:
                progress_fn(f"Local CC: frame {i+1}/{n_frames}")

    return local_shifts


# ═════════════════════════════════════════════════════════════════
# Temporal smoothing + outlier rejection
# ═════════════════════════════════════════════════════════════════

def smooth_patch_shifts(local_shifts: np.ndarray,
                        outlier_sigma: float = 3.0):
    """Temporally smooth per-patch shifts and reject outliers.

    Parameters
    ----------
    local_shifts : ndarray (n_frames, n_patches, 2)
    outlier_sigma : float
        MAD threshold for outlier rejection.

    Returns
    -------
    smoothed : ndarray (n_frames, n_patches, 2)
    """
    n_frames, n_patches, _ = local_shifts.shape
    cleaned = local_shifts.copy()

    # Vectorized outlier rejection: per-frame, compare each patch to median
    for axis in range(2):
        for f in range(n_frames):
            vals = cleaned[f, :, axis]  # (n_patches,)
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            mad = max(mad, 1e-6)
            outlier_mask = np.abs(vals - med) > outlier_sigma * 1.4826 * mad
            vals[outlier_mask] = med

    # Cubic spline smoothing per-patch trajectory
    smoothed = np.zeros_like(cleaned)
    if n_frames >= 4:
        t = np.arange(n_frames, dtype=np.float64)
        for j in range(n_patches):
            for axis in range(2):
                cs = CubicSpline(t, cleaned[:, j, axis])
                smoothed[:, j, axis] = cs(t)
    else:
        smoothed[:] = cleaned

    return smoothed


# ═════════════════════════════════════════════════════════════════
# Polynomial motion field fitting
# ═════════════════════════════════════════════════════════════════

def _build_vandermonde(centers: np.ndarray, degree: int = 3):
    """Build 2-D polynomial Vandermonde matrix.

    For degree 3, columns are:
    [1, y, x, y², yx, x², y³, y²x, yx², x³]
    (10 terms)

    Parameters
    ----------
    centers : ndarray (N, 2), columns are (cy, cx)
    degree : int

    Returns
    -------
    A : ndarray (N, n_terms)
    """
    cy = centers[:, 0]
    cx = centers[:, 1]
    cols = []
    for p in range(degree + 1):
        for qy in range(p, -1, -1):
            qx = p - qy
            cols.append((cy ** qy) * (cx ** qx))
    return np.column_stack(cols)


def fit_motion_field(centers: np.ndarray,
                     smoothed_shifts: np.ndarray,
                     degree: int = 3):
    """Fit polynomial motion field per frame.

    Parameters
    ----------
    centers : ndarray (n_patches, 2)
    smoothed_shifts : ndarray (n_frames, n_patches, 2)
    degree : int

    Returns
    -------
    coefficients : ndarray (n_frames, 2, n_terms)
    """
    n_frames = smoothed_shifts.shape[0]
    A = _build_vandermonde(centers, degree)
    n_terms = A.shape[1]

    # Normalize coordinates for numerical stability
    cy_mean, cx_mean = centers.mean(axis=0)
    cy_std = max(centers[:, 0].std(), 1.0)
    cx_std = max(centers[:, 1].std(), 1.0)

    A_norm = _build_vandermonde(
        np.column_stack([
            (centers[:, 0] - cy_mean) / cy_std,
            (centers[:, 1] - cx_mean) / cx_std,
        ]),
        degree,
    )

    coeffs = np.zeros((n_frames, 2, n_terms), dtype=np.float64)
    for f in range(n_frames):
        for axis in range(2):
            b = smoothed_shifts[f, :, axis]
            result = np.linalg.lstsq(A_norm, b, rcond=None)
            coeffs[f, axis, :] = result[0]

    return coeffs, (cy_mean, cx_mean, cy_std, cx_std)


def evaluate_motion_field(coefficients: np.ndarray,
                          norm_params: tuple,
                          points: np.ndarray,
                          degree: int = 3):
    """Evaluate polynomial motion field at arbitrary points.

    Parameters
    ----------
    coefficients : ndarray (2, n_terms) for one frame
    norm_params : (cy_mean, cx_mean, cy_std, cx_std)
    points : ndarray (N, 2) — (y, x) coordinates
    degree : int

    Returns
    -------
    shifts : ndarray (N, 2) — (dy, dx) at each point
    """
    cy_mean, cx_mean, cy_std, cx_std = norm_params
    norm_pts = np.column_stack([
        (points[:, 0] - cy_mean) / cy_std,
        (points[:, 1] - cx_mean) / cx_std,
    ])
    A = _build_vandermonde(norm_pts, degree)
    dy = A @ coefficients[0]
    dx = A @ coefficients[1]
    return np.column_stack([dy, dx])


# ═════════════════════════════════════════════════════════════════
# Per-patch Fourier shift + cosine-blend compositing
# ═════════════════════════════════════════════════════════════════

def apply_local_shifts(stack, gain, gain_orientation,
                       global_shifts_y, global_shifts_x,
                       coefficients, norm_params,
                       patches, centers, blend_weights,
                       degree=3,
                       shift_scale=1.0,
                       progress_fn=None,
                       cancel_fn=None,
                       dose_weight=None):
    """Apply local motion correction and composite the final sum.

    For each frame:
      1. Gain-correct the full-resolution frame
      2. Extract patches, apply combined (global + local) shift per-patch
         via batched FFT + phase ramps, multiply by blend weights
      3. Scatter-add blended patches into accumulator

    Parameters
    ----------
    stack : ndarray (n_frames, H, W) — raw frames (any dtype)
    gain : ndarray (H, W) float32 or None — oriented+matched gain
    global_shifts_y, global_shifts_x : arrays (n_frames,) or None — Phase 1 shifts
    coefficients : ndarray (n_frames, 2, n_terms) — polynomial coeffs
    norm_params : tuple — normalization params from fit_motion_field
    patches : list of (y0, x0, y1, x1)
    centers : ndarray (n_patches, 2) — centers for motion field evaluation
        (must be in same coordinate system as norm_params)
    blend_weights : ndarray (ps, ps) float32
    degree : int
    shift_scale : float
        Multiplier for local shifts from the motion field. Use bin_factor
        when the motion field was fit at binned resolution but compositing
        is at full resolution.
    progress_fn : callable(str) or None
    cancel_fn : callable() -> bool or None
    dose_weight : DoseWeightParams or None
        When not None, accumulates a parallel dose-weighted sum in Fourier space.
        The non-DW corrected_sum is bit-for-bit identical whether this is on or off.

    Returns
    -------
    corrected_sum : ndarray (H, W) float32
    dw_sum : ndarray (H, W) float32 or None
        Dose-weighted sum. None if dose_weight is None.
    """
    n_frames, fh, fw = stack.shape
    n_patches = len(patches)
    ps = patches[0][2] - patches[0][0]

    # Pre-compute weight map (sum of all blend windows at their positions)
    weight_sum = np.zeros((fh, fw), dtype=np.float32)
    for y0, x0, y1, x1 in patches:
        weight_sum[y0:y1, x0:x1] += blend_weights
    weight_sum = np.maximum(weight_sum, 1e-6)

    patch_centers_arr = np.array(centers)
    twopi_j = np.float64(-2.0 * np.pi) * 1j

    # Pre-compute patch coordinates as arrays for fast extraction
    p_y0 = np.array([p[0] for p in patches], dtype=np.intp)
    p_x0 = np.array([p[1] for p in patches], dtype=np.intp)

    # ── Dose-weighting setup (additive, strictly separate from non-DW path) ──
    do_dw = dose_weight is not None
    if do_dw:
        from de_groundcrew.external.gc_motion.dose_weighting import (
            dose_weight_map, frame_doses)
        _dw_doses = frame_doses(n_frames, dose_weight.total_dose,
                                dose_weight.n_total_frames, dose_weight.frame_offset)
        # Accumulator and weight-sum are allocated after the _GPU check below,
        # using the appropriate array module.

    if _GPU:
        # ── GPU path: sub-batched patches + Fourier-space phase shifts ──
        # Optimization: combine global + local shifts per-patch to avoid
        # an expensive full-frame FFT+IFFT (8K = 67M-point FFT) per frame.
        # Instead, extract patches from the un-shifted frame and apply the
        # combined (global + local) shift via a single patch-level FFT.
        blend_gpu = _cp.asarray(blend_weights, dtype=_cp.float32)
        accum_gpu = _cp.zeros((fh, fw), dtype=_cp.float32)
        gain_gpu = _cp.asarray(gain, dtype=_cp.float32) if gain is not None else None

        # DW GPU accumulators (additive, strictly separate from accum_gpu path)
        if do_dw:
            _dw_accum_fft = _cp.zeros((fh, fw), dtype=_cp.complex64)
            _dw_wsum = _cp.zeros((fh, fw), dtype=_cp.float32)
            weight_sum_gpu = _cp.asarray(weight_sum, dtype=_cp.float32)

        # Frequency grids for patches (batched: broadcast over batch dim)
        fy_p = _cp.fft.fftfreq(ps).astype(_cp.float32).reshape(1, -1, 1)
        fx_p = _cp.fft.fftfreq(ps).astype(_cp.float32).reshape(1, 1, -1)

        # Sub-batch size: limit GPU memory for patch arrays
        # Each patch needs ~20 bytes/pixel (float32 + complex64 FFT + phase)
        # Limit to ~2 GB for patch workspace
        bytes_per_patch = ps * ps * 20
        max_batch = max(min(n_patches, int(2e9 / bytes_per_patch)), 16)

        for i in range(n_frames):
            if cancel_fn and cancel_fn():
                return None, None

            # 1. Gain-correct (no global shift applied to full frame)
            frame_gpu = _cp.asarray(stack[i], dtype=_cp.float32)
            if gain_gpu is not None:
                frame_gpu *= gain_gpu

            # 2. Evaluate all local shifts, scale, and combine with global
            local_shifts = evaluate_motion_field(
                coefficients[i], norm_params, patch_centers_arr, degree
            )
            if shift_scale != 1.0:
                local_shifts *= shift_scale
            if global_shifts_y is not None:
                local_shifts[:, 0] += float(-global_shifts_y[i])
                local_shifts[:, 1] += float(-global_shifts_x[i])

            # Upload all shifts for this frame at once
            all_dy = _cp.asarray(local_shifts[:, 0], dtype=_cp.float32)
            all_dx = _cp.asarray(local_shifts[:, 1], dtype=_cp.float32)

            # DW: build a per-frame image buffer in parallel (additive, separate).
            # accum_gpu always receives the same direct scatter-adds as the non-DW path
            # (bit-identical regardless of do_dw). frame_accum_gpu is an extra buffer
            # used only for the DW FFT — it does NOT feed into accum_gpu.
            frame_accum_gpu = _cp.zeros((fh, fw), dtype=_cp.float32) if do_dw else None

            # 3. Process patches in sub-batches
            for b_start in range(0, n_patches, max_batch):
                b_end = min(b_start + max_batch, n_patches)
                b_size = b_end - b_start

                # Extract patches from un-shifted frame
                patch_batch = _cp.zeros((b_size, ps, ps), dtype=_cp.float32)
                for j in range(b_size):
                    jj = b_start + j
                    patch_batch[j] = frame_gpu[p_y0[jj]:p_y0[jj]+ps,
                                               p_x0[jj]:p_x0[jj]+ps]

                # Batch Fourier shift with combined global+local
                dy_sub = all_dy[b_start:b_end].reshape(-1, 1, 1)
                dx_sub = all_dx[b_start:b_end].reshape(-1, 1, 1)

                phase_batch = _cp.exp(
                    _cp.complex64(twopi_j) * (
                        fy_p * dy_sub + fx_p * dx_sub
                    )
                )
                patch_fft = _cp.fft.fft2(patch_batch, axes=(-2, -1))
                patch_batch = _cp.real(
                    _cp.fft.ifft2(patch_fft * phase_batch, axes=(-2, -1))
                ).astype(_cp.float32)

                # Blend and scatter-add
                patch_batch *= blend_gpu[None, :, :]
                for j in range(b_size):
                    jj = b_start + j
                    accum_gpu[p_y0[jj]:p_y0[jj]+ps,
                              p_x0[jj]:p_x0[jj]+ps] += patch_batch[j]
                    if do_dw:
                        # Also scatter into the per-frame DW buffer (separate from accum_gpu)
                        frame_accum_gpu[p_y0[jj]:p_y0[jj]+ps,
                                        p_x0[jj]:p_x0[jj]+ps] += patch_batch[j]

            if do_dw:
                frame_corr = frame_accum_gpu / _cp.maximum(weight_sum_gpu, 1e-6)  # 1e-6 matches weight_sum floor above
                W = dose_weight_map((fh, fw), dose_weight.apix,
                                    float(_dw_doses[i]), dose_weight.voltage_kv,
                                    xp=_cp)
                _dw_accum_fft += _cp.fft.fft2(frame_corr) * W
                _dw_wsum += W

            if progress_fn:
                progress_fn(f"Local correction: frame {i+1}/{n_frames}")

        accum = _cp.asnumpy(accum_gpu)
    else:
        # ── CPU path ──
        # Same optimization: combine global + local shifts per-patch
        accum = np.zeros((fh, fw), dtype=np.float32)

        # DW CPU accumulators (additive, strictly separate from accum path)
        if do_dw:
            _dw_accum_fft = np.zeros((fh, fw), dtype=np.complex64)
            _dw_wsum = np.zeros((fh, fw), dtype=np.float32)

        fy_p = np.fft.fftfreq(ps).astype(np.float32).reshape(-1, 1)
        fx_p = np.fft.fftfreq(ps).astype(np.float32).reshape(1, -1)

        for i in range(n_frames):
            if cancel_fn and cancel_fn():
                return None, None

            frame = stack[i].astype(np.float32)
            if gain is not None:
                frame = frame * gain

            # Evaluate local shifts, scale, and add global shift
            local_shifts = evaluate_motion_field(
                coefficients[i], norm_params, patch_centers_arr, degree
            )
            if shift_scale != 1.0:
                local_shifts *= shift_scale
            if global_shifts_y is not None:
                local_shifts[:, 0] += float(-global_shifts_y[i])
                local_shifts[:, 1] += float(-global_shifts_x[i])

            frame_accum = np.zeros((fh, fw), dtype=np.float32)
            for j, (y0, x0, y1, x1) in enumerate(patches):
                dy_l = float(local_shifts[j, 0])
                dx_l = float(local_shifts[j, 1])

                patch = frame[y0:y1, x0:x1].copy()

                if abs(dy_l) > 1e-6 or abs(dx_l) > 1e-6:
                    phase = np.exp(
                        np.complex64(twopi_j) * (fy_p * dy_l + fx_p * dx_l)
                    )
                    patch = np.real(
                        np.fft.ifft2(np.fft.fft2(patch) * phase)
                    ).astype(np.float32)

                frame_accum[y0:y1, x0:x1] += patch * blend_weights

            accum += frame_accum

            if do_dw:
                frame_corr = frame_accum / np.maximum(weight_sum, 1e-6)  # 1e-6 matches weight_sum floor above
                W = dose_weight_map((fh, fw), dose_weight.apix,
                                    float(_dw_doses[i]), dose_weight.voltage_kv,
                                    xp=np)
                _dw_accum_fft += np.fft.fft2(frame_corr) * W
                _dw_wsum += W

            if progress_fn:
                progress_fn(f"Local correction: frame {i+1}/{n_frames}")

    # Normalize by weight sum and frame count
    corrected_sum = accum / weight_sum / n_frames

    if do_dw:
        xp = _cp if _GPU else np
        dw_sum = xp.real(xp.fft.ifft2(_dw_accum_fft / xp.maximum(_dw_wsum, 1e-8)))  # per-frequency norm BEFORE inverse FFT; 1e-8 Fourier-space guard (sum_w >= ~3, never engages)
        dw_sum = (_cp.asnumpy(dw_sum) if _GPU else dw_sum).astype(np.float32)
    else:
        dw_sum = None

    return corrected_sum, dw_sum

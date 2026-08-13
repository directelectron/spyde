"""
_worker_extracts.py — functions lifted VERBATIM from Ground Crew's
``workers/motion_correction_worker.py`` at e9e21de.

That module cannot be vendored whole because it imports PySide6, but four of
its functions are pure numerics that the rest of the vendored code needs:

    _bandpass_filter      (upstream lines 241–275)
    _cross_correlate      (277–383)  — used by local_motion.correlate_patches
    _apply_shift_fourier  (385–403)
    _compute_log_fft      (73–204)

They are copied unchanged rather than re-typed, for the same reason the rest is
vendored. `_GPU`/`_cp` are bound to the CPU path — this app has no CuPy
dependency — which is the only change.

`_MODE_PRESETS` rides along because it is the same file's data and the driver
needs it to select v3's resolution cap.
"""
import numpy as np

_GPU = False
_cp = None

#: v3 resolution caps. "fast" stops at 6 Å, "fine" pushes to 3 Å.
_MODE_PRESETS = {
    "fast": {"fine_cap_apx": 6.0},
    "fine": {"fine_cap_apx": 3.0},
}
_DEFAULT_MODE = "fast"


def _compute_log_fft(image: np.ndarray,
                     log_base: float = 5.0,
                     trunc_diam: float = 0.01,
                     bkgd_gray: float = 4.0) -> np.ndarray:
    """SerialEM-style power spectrum display.

    Pipeline (following SerialEM's ProcFFT + FindPctStretch):
      1. Cosine edge-taper (2.5%) to reduce FFT edge artifacts
      2. FFT → magnitude
      3. Normalize log-scale factor by Nyquist-edge mean magnitude
      4. Apply ``log(logScale * mag + 1)`` with center-peak scaling
      5. Two-anchor contrast stretch:
         - White point = mean of a small ring near center (clips bright core)
         - Black point adjusted so high-freq edges map to *bkgd_gray*

    Returns float32 image ready for display panel (0–255 range, approx.).
    The display panel's sigma stretch provides additional fine-tuning.

    Parameters
    ----------
    log_base : float
        Initial log-scale multiplier (SerialEM uses 5.0).
    trunc_diam : float
        Fractional ring diameter whose mean becomes the white point
        (SerialEM default range 0.002–0.025; 0.01 works well).
    bkgd_gray : float
        Target gray level (0–255) for the high-frequency background.
    """
    h, w = image.shape[-2:]

    # ── 0. Crop to a centred square ──────────────────────────
    # The displayed power spectrum must be square so Thon rings render circular
    # for non-square sensors; the full real-space image is shown separately and
    # is unaffected by this crop. (Mirrors calibration_worker's crop-to-square.)
    if h != w:
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        image = image[..., y0:y0 + s, x0:x0 + s]
        h, w = s, s

    # ── 1. Edge taper (2.5% cosine) ──────────────────────────
    taper_frac = 0.025
    img = image.astype(np.float64)
    # Build 1-D taper ramps for rows and columns
    ty = int(max(h * taper_frac, 1))
    tx = int(max(w * taper_frac, 1))
    ramp_y = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, ty)))
    ramp_x = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, tx)))
    # Apply to edges (multiplicative fade to mean)
    mean_val = float(np.mean(img))
    img_tapered = img - mean_val
    img_tapered[:ty, :]  *= ramp_y[:, None]
    img_tapered[-ty:, :] *= ramp_y[::-1, None]
    img_tapered[:, :tx]  *= ramp_x[None, :]
    img_tapered[:, -tx:] *= ramp_x[None, ::-1]
    img_tapered += mean_val

    # ── 2. FFT → magnitude ───────────────────────────────────
    if _GPU:
        g = _cp.asarray(img_tapered, dtype=_cp.float64)
        ft = _cp.fft.fftshift(_cp.fft.fft2(g))
        mag = _cp.asnumpy(_cp.abs(ft)).astype(np.float64)
    else:
        ft = np.fft.fftshift(np.fft.fft2(img_tapered))
        mag = np.abs(ft)

    cy, cx = h // 2, w // 2

    # ── 3. Normalize logScale by Nyquist-edge mean ───────────
    # Mean magnitude along the border (top + bottom rows, left + right cols)
    border = np.concatenate([
        mag[0, :], mag[-1, :], mag[:, 0], mag[:, -1],
    ])
    border_mean = float(np.mean(border))
    if border_mean < 1e-12:
        border_mean = 1.0
    log_scale = log_base / border_mean

    # ── 4. Log scaling ───────────────────────────────────────
    # Find peak near center (excluding DC)
    r_search = max(min(h, w) // 10, 5)
    center_patch = mag[cy - r_search:cy + r_search,
                       cx - r_search:cx + r_search].copy()
    # Zero out DC pixel in the patch
    center_patch[r_search, r_search] = 0.0
    cen_max = float(np.max(center_patch))
    if cen_max < 1e-12:
        cen_max = 1.0

    scale = 32000.0 / np.log(log_scale * cen_max + 1.0)
    ps = (scale * np.log(log_scale * mag + 1.0)).astype(np.float32)
    # Clamp DC to max
    ps[cy, cx] = 32000.0

    # ── 5. Two-anchor contrast stretch ───────────────────────
    # White point: mean of a small ring near center
    ring_rad = np.sqrt(h * w) * trunc_diam / 2.0
    ring_rad = max(ring_rad, 3.0)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2.0 + (x - cx) ** 2.0)
    ring_mask = (r >= 0.7 * ring_rad) & (r <= ring_rad + 1.0)
    if np.any(ring_mask):
        max_scale = float(np.mean(ps[ring_mask]))
    else:
        max_scale = float(np.percentile(ps, 99.9))

    # Black point: adjust so border pixels map to bkgd_gray
    border_ps = np.concatenate([
        ps[0, :], ps[-1, :], ps[1:-1, 0], ps[1:-1, -1],
    ])
    edge_mean = float(np.mean(border_ps))
    target_frac = bkgd_gray / 255.0
    # Solve: (edge_mean - min_scale) / (max_scale - min_scale) = target_frac
    if abs(1.0 - target_frac) > 1e-6:
        min_scale = (edge_mean - max_scale * target_frac) / (1.0 - target_frac)
    else:
        min_scale = edge_mean

    # Linear stretch to 0–255
    denom = max_scale - min_scale
    if denom < 1e-6:
        denom = 1.0
    result = (ps - min_scale) / denom * 255.0
    result = np.clip(result, 0, 255).astype(np.float32)

    return result


# ═════════════════════════════════════════════════════════════════
# Cross-correlation alignment (Guizar-Sicairos subpixel refinement)
# ═════════════════════════════════════════════════════════════════


def _upsampled_dft(data, upsampled_region_size, upsample_factor, axis_offsets):
    """
    Upsampled DFT by matrix multiplication (Guizar-Sicairos et al., 2008).

    Follows the scikit-image reference implementation.
    `data` is the product conj(F1)*F2 in Fourier space.
    """
    xp = _cp if _GPU and isinstance(data, _cp.ndarray) else np
    ups = upsampled_region_size

    # Row kernel: operates on axis 0 (rows ↔ y)
    # Uses +2j*pi (inverse DFT) since we're evaluating cc(τ) from CC(k)
    row_idx = xp.arange(ups, dtype=np.float64) - axis_offsets[0]
    freq_row = xp.fft.ifftshift(
        xp.arange(data.shape[0], dtype=np.float64) - data.shape[0] // 2
    )
    kernel_r = xp.exp(
        +2j * np.pi / (data.shape[0] * upsample_factor)
        * row_idx[:, None] * freq_row[None, :]
    )  # shape (ups, h)

    # Col kernel: operates on axis 1 (cols ↔ x)
    col_idx = xp.arange(ups, dtype=np.float64) - axis_offsets[1]
    freq_col = xp.fft.ifftshift(
        xp.arange(data.shape[1], dtype=np.float64) - data.shape[1] // 2
    )
    kernel_c = xp.exp(
        +2j * np.pi / (data.shape[1] * upsample_factor)
        * col_idx[:, None] * freq_col[None, :]
    )  # shape (ups, w)

    # kernel_r @ data @ kernel_c.T  →  shape (ups, ups)
    return kernel_r @ data @ kernel_c.T


def _bandpass_filter(shape, low_freq=0.005, high_freq=0.5):
    """Cosine-edge bandpass filter in Fourier space.

    Suppresses very-low-frequency (uneven illumination) and high-frequency noise.
    Frequencies are in fractional Nyquist units (0..1 where 1 = Nyquist).
    Returns a real 2-D array suitable for multiplying Fourier coefficients.
    """
    h, w = shape
    fy = np.fft.fftfreq(h).astype(np.float64)
    fx = np.fft.fftfreq(w).astype(np.float64)
    freq_r = np.sqrt(fy[:, None]**2 + fx[None, :]**2)  # radial frequency 0..0.5

    # Normalize so 0.5 → 1.0 (Nyquist)
    freq_norm = freq_r / 0.5

    # Cosine roll-off edges
    lo_width = max(low_freq * 0.5, 0.002)
    hi_width = max((1.0 - high_freq) * 0.5, 0.01)

    filt = np.ones_like(freq_norm)
    # Low-frequency suppression
    mask_lo = freq_norm < low_freq
    mask_lo_edge = (freq_norm >= low_freq - lo_width) & (freq_norm < low_freq)
    filt[mask_lo] = 0.0
    if np.any(mask_lo_edge):
        filt[mask_lo_edge] = 0.5 * (1.0 + np.cos(np.pi * (freq_norm[mask_lo_edge] - low_freq) / lo_width))
    # High-frequency suppression
    mask_hi = freq_norm > high_freq
    mask_hi_edge = (freq_norm > high_freq) & (freq_norm <= high_freq + hi_width)
    filt[mask_hi] = 0.0
    if np.any(mask_hi_edge):
        filt[mask_hi_edge] = 0.5 * (1.0 + np.cos(np.pi * (freq_norm[mask_hi_edge] - high_freq) / hi_width))

    return filt.astype(np.float64)


def _cross_correlate(ref: np.ndarray, frame: np.ndarray,
                     upsample_factor: int = 20,
                     bandpass: bool = True,
                     ref_fft=None,
                     bp_filter=None,
                     frame_fft=None) -> tuple[float, float]:
    """
    Subpixel shift estimation via cross-correlation with upsampled DFT refinement.

    Uses cross-correlation (not phase correlation) to preserve amplitude/SNR,
    following MotionCor2 (Zheng et al., 2017).

    Parameters
    ----------
    ref_fft : optional
        Pre-computed (and optionally bandpass-filtered) FFT of the reference.
        When supplied, `ref` is ignored for the FFT (saves one FFT per call).
    bp_filter : optional
        Pre-computed bandpass filter array (CPU or GPU). When supplied, avoids
        recomputing the filter on every call.
    frame_fft : optional
        Pre-computed (and bandpass-filtered) FFT of the frame. When supplied,
        `frame` is ignored for the FFT (eliminates GPU transfer + FFT per call).

    Returns (dy, dx) — the shift needed to align `frame` to `ref`.
    Positive dy means frame is shifted down relative to ref.
    """
    if _GPU:
        xp = _cp

        if frame_fft is not None:
            F2 = frame_fft
        else:
            frame_g = _cp.asarray(frame, dtype=_cp.float32)
            F2 = _cp.fft.fft2(frame_g)

        if ref_fft is not None:
            F1 = ref_fft
        else:
            ref_g = _cp.asarray(ref, dtype=_cp.float32)
            F1 = _cp.fft.fft2(ref_g)
    else:
        xp = np

        if frame_fft is not None:
            F2 = frame_fft
        else:
            F2 = np.fft.fft2(frame.astype(np.float32))

        if ref_fft is not None:
            F1 = ref_fft
        else:
            F1 = np.fft.fft2(ref.astype(np.float32))

    # Optional bandpass filter — only apply if FFTs were not pre-filtered
    if bandpass:
        if bp_filter is None:
            bp = _bandpass_filter(frame.shape, low_freq=0.01, high_freq=0.5)
            if _GPU:
                bp = _cp.asarray(bp)
        else:
            bp = bp_filter
        if ref_fft is None:
            F1 = F1 * bp
        if frame_fft is None:
            F2 = F2 * bp

    # Cross-correlation: conj(F1)*F2 peaks at the shift of frame relative to ref
    CC = xp.conj(F1) * F2

    # Integer peak from full IFFT
    cc = xp.real(xp.fft.ifft2(CC))
    h, w = cc.shape
    peak_flat = int(xp.argmax(cc))
    peak_y, peak_x = divmod(peak_flat, w)

    # Handle wrap-around
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w

    if upsample_factor <= 1:
        return float(peak_y), float(peak_x)

    # Subpixel refinement via upsampled DFT around the integer peak
    ups_region = int(np.ceil(upsample_factor * 1.5))
    dft_shift = int(np.floor(ups_region / 2.0))

    offset_y = dft_shift - peak_y * upsample_factor
    offset_x = dft_shift - peak_x * upsample_factor

    upsampled = _upsampled_dft(
        CC,
        upsampled_region_size=ups_region,
        upsample_factor=upsample_factor,
        axis_offsets=[offset_y, offset_x],
    )
    cc_up = xp.abs(upsampled)
    peak_flat_up = int(xp.argmax(cc_up))
    uy, ux = divmod(peak_flat_up, ups_region)

    shift_y = peak_y + (uy - dft_shift) / upsample_factor
    shift_x = peak_x + (ux - dft_shift) / upsample_factor

    return float(shift_y), float(shift_x)


def _apply_shift_fourier(frame: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift image by (dy, dx) using Fourier phase ramp (exact for band-limited data)."""
    if _GPU:
        g = _cp.asarray(frame, dtype=_cp.float32)
        h, w = g.shape
        fy = _cp.fft.fftfreq(h).astype(_cp.float32).reshape(-1, 1)
        fx = _cp.fft.fftfreq(w).astype(_cp.float32).reshape(1, -1)
        phase = _cp.exp(_cp.complex64(-2j * np.pi) * (fy * dy + fx * dx))
        shifted = _cp.real(_cp.fft.ifft2(_cp.fft.fft2(g) * phase))
        return _cp.asnumpy(shifted).astype(np.float32)
    else:
        f = frame.astype(np.float32)
        h, w = f.shape
        fy = np.fft.fftfreq(h).astype(np.float32).reshape(-1, 1)
        fx = np.fft.fftfreq(w).astype(np.float32).reshape(1, -1)
        phase = np.exp(np.complex64(-2j * np.pi) * (fy * dy + fx * dx))
        return np.real(np.fft.ifft2(np.fft.fft2(f) * phase)).astype(np.float32)



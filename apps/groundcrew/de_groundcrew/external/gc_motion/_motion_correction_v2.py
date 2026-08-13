"""
Unblur-style motion correction — v2 prototype.

Derived from:
- cisTEM/Unblur (Grant & Grigorieff 2015) — leave-one-out reference, running-average input,
  iterative convergence, B-factor envelope inside CC.
- MotionCor3 (Zheng 2023) — progressive B-factor decay across iterations.

Key differences from existing DE GC Phase 1 (workers/motion_correction_worker.py):
- Gain-corrected frames feed CC (was: raw frames, per comment assumption)
- Mean-subtracted + cosine-edge-tapered before FFT (was: neither)
- Leave-one-out reference per frame (was: single static central frame)
- Running-average input window (was: single frame)
- Iterative refinement with convergence (was: fixed 2 passes)
- Progressive B-factor envelope replaces hard bandpass (was: hard [0.01, 0.5] cosine mask)
- Normalized CC (implicitly via envelope weighting)
- Real FFT (rfft) — halves GPU memory footprint

This is a STANDALONE prototype for validation — not yet integrated into the worker.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from scipy.signal import savgol_filter

try:
    import cupy as _cp
    _GPU = True
except Exception:
    _GPU = False
    _cp = None


def _release_gpu_blocks():
    """Shape-boundary GPU release: the outgoing shape's cuFFT plans are dead, and
    free_all_blocks() alone cannot return the pool blocks their workspace still
    pins (order measured 2026-07-18) — clear the plan cache first.
    NOT for solver iteration loops: same-shape plans stay hot there, and a clear
    would re-pay plan builds every iteration."""
    _cp.fft.config.get_plan_cache().clear()
    _cp.get_default_memory_pool().free_all_blocks()


# ════════════════════════════════════════════════════════════════════
# Core helpers
# ════════════════════════════════════════════════════════════════════

def _bin_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Bin by summing factor×factor blocks (preserves count statistics)."""
    if factor == 1:
        return img
    h, w = img.shape
    nh = (h // factor) * factor
    nw = (w // factor) * factor
    return img[:nh, :nw].reshape(
        nh // factor, factor, nw // factor, factor
    ).sum(axis=(1, 3))


def _apply_orientation(img: np.ndarray, idx: int) -> np.ndarray:
    if idx == 0: return img
    if idx == 1: return np.rot90(img, 1)
    if idx == 2: return np.rot90(img, 2)
    if idx == 3: return np.rot90(img, 3)
    if idx == 4: return np.fliplr(img)
    if idx == 5: return np.flipud(img)
    if idx == 6: return img.T
    if idx == 7: return np.rot90(img, 2).T
    return img


def _match_gain_to_frame(gain: np.ndarray, h: int, w: int) -> np.ndarray:
    gh, gw = gain.shape
    if gh == h and gw == w:
        return gain
    if gh == h * 2 and gw == w * 2:
        return _bin_image(gain, 2) / 4.0
    ratio = gh // h
    if ratio == gw // w and ratio >= 2 and gh == h * ratio and gw == w * ratio:
        return _bin_image(gain, ratio) / float(ratio * ratio)
    raise ValueError(f"Gain {gh}x{gw} doesn't match frame {h}x{w}")


def _cosine_taper_2d(shape: tuple[int, int], taper_frac: float = 0.025) -> np.ndarray:
    """Cosine edge taper (2.5% default, Unblur convention). Always CPU numpy."""
    h, w = shape
    ty = np.ones(h, dtype=np.float32)
    tx = np.ones(w, dtype=np.float32)
    th = max(int(h * taper_frac), 1)
    tw = max(int(w * taper_frac), 1)
    ramp_y = 0.5 * (1 - np.cos(np.linspace(0, np.pi, th, dtype=np.float32)))
    ramp_x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, tw, dtype=np.float32)))
    ty[:th] = ramp_y
    ty[-th:] = ramp_y[::-1]
    tx[:tw] = ramp_x
    tx[-tw:] = ramp_x[::-1]
    return np.outer(ty, tx)


def _b_factor_envelope_rfft(shape: tuple[int, int], b_factor: float, xp=np):
    """Gaussian B-factor envelope for rfft layout (H, W//2+1).
    Weight = exp(-B * s²), s in cycles/pixel.
    """
    h, w = shape
    fy = xp.fft.fftfreq(h).astype(xp.float32).reshape(-1, 1)      # (h, 1)
    fx = xp.fft.rfftfreq(w).astype(xp.float32).reshape(1, -1)     # (1, w//2+1)
    s2 = fy * fy + fx * fx
    return xp.exp(-b_factor * s2).astype(xp.float32)


# ════════════════════════════════════════════════════════════════════
# Upsampled-DFT subpixel (Guizar-Sicairos) — operates on rfft output
# ════════════════════════════════════════════════════════════════════

def _upsampled_dft_peak(cc_real, upsample_factor: int, xp=np):
    """Find CC peak with subpixel refinement via Guizar-Sicairos upsampled DFT.

    cc_real : 2D real CC image (from irfft).
    Returns (dy, dx) in pixel units (signed, wrap-around handled).

    Uses the production-proven kernel pattern from workers/motion_correction_worker.py
    (which follows scikit-image's reference implementation).
    """
    h, w = cc_real.shape
    # Integer peak with wrap-around
    peak_flat = int(xp.argmax(cc_real))
    peak_y, peak_x = divmod(peak_flat, w)
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w

    if upsample_factor <= 1:
        return float(peak_y), float(peak_x)

    # Full complex FFT of the real CC image (needed for the Guizar-Sicairos kernels)
    cc_fft = xp.fft.fft2(cc_real)

    ups = int(np.ceil(upsample_factor * 1.5))
    dft_shift = int(np.floor(ups / 2.0))
    offset_y = dft_shift - peak_y * upsample_factor
    offset_x = dft_shift - peak_x * upsample_factor

    # Row kernel: shape (ups, h)
    row_idx = xp.arange(ups, dtype=np.float64) - offset_y
    freq_row = xp.fft.ifftshift(xp.arange(h, dtype=np.float64) - h // 2)
    kernel_r = xp.exp(
        +2j * np.pi / (h * upsample_factor)
        * row_idx[:, None] * freq_row[None, :]
    )

    # Col kernel: shape (ups, w)
    col_idx = xp.arange(ups, dtype=np.float64) - offset_x
    freq_col = xp.fft.ifftshift(xp.arange(w, dtype=np.float64) - w // 2)
    kernel_c = xp.exp(
        +2j * np.pi / (w * upsample_factor)
        * col_idx[:, None] * freq_col[None, :]
    )

    # (ups, h) @ (h, w) @ (w, ups) → (ups, ups)
    upsampled = kernel_r @ cc_fft @ kernel_c.T
    cc_up = xp.abs(upsampled)
    peak_flat_up = int(xp.argmax(cc_up))
    uy, ux = divmod(peak_flat_up, ups)

    shift_y = peak_y + (uy - dft_shift) / upsample_factor
    shift_x = peak_x + (ux - dft_shift) / upsample_factor
    return float(shift_y), float(shift_x)


# ════════════════════════════════════════════════════════════════════
# Main algorithm
# ════════════════════════════════════════════════════════════════════

def motion_correct_v2(
    stack: np.ndarray,
    gain: Optional[np.ndarray] = None,
    gain_orientation: int = 5,
    *,
    bin_factor: int = 2,
    max_iter: int = 5,
    tol_px: float = 0.05,
    running_avg_N: int = 3,
    b_factor_init: float = 150.0,
    b_factor_min: float = 20.0,
    b_factor_decay: float = 0.8,
    upsample_factor: int = 10,
    taper_frac: float = 0.025,
    progress_cb: Optional[Callable[[str], None]] = None,
    # Stabilization (V9 defaults — empirically best on FSU apoferritin)
    damping: float = 0.5,               # Blend α·old + (1-α)·new each iter → anti-divergence
    leave_one_out: bool = True,         # False = use full S as reference (V3 was close 2nd)
    shift_window: bool = True,          # Pre-align window frames before summing
    b_factor_fixed: bool = True,        # Keep b_factor_init throughout iterations
    track_cc_quality: bool = False,     # True = log CC peak-to-background ratio per frame
) -> dict:
    """Unblur-style iterative motion correction, V9 configuration.

    Default parameters reflect the **empirically best variant** from the divergence
    diagnostic on FSU apoferritin (15eps + 60eps). The combination
    `damping=0.5 + b_factor_fixed=True + shift_window=True` stabilizes iteration
    to within ~12% of MotionCor3-measured shifts on 60eps data and gives RMS
    error 2.4× lower than pure iterative Unblur-style (V0 baseline).

    The default config converges in ~5 iterations with sub-0.05 px tolerance.

    Returns dict with keys:
      shifts_y, shifts_x : final per-frame shifts in SUPER-RES pixels (len N)
      iterations         : number of iterations run
      convergence        : list of max |Δshift| per iteration (super-res px)
      aligned_sum        : (H, W) float32, gain-corrected, motion-corrected sum (super-res)
      unaligned_sum      : (H, W) float32, gain-corrected raw sum
      cc_quality         : optional list of mean CC peak heights per iteration (if track_cc_quality)

    Stabilization parameters (V9 defaults):
      damping            : 0.5 = 50/50 blend with previous iter (prevents overshoot)
      leave_one_out      : True (minor effect vs V3 NO-L1O; keep for now)
      shift_window       : True = phase-shift window frames by current shifts before summing
      b_factor_fixed     : True = keep B constant across iterations (vs decay)
    """
    assert running_avg_N % 2 == 1, "running_avg_N must be odd"
    xp = _cp if _GPU else np
    log = progress_cb or (lambda msg: None)

    n_frames, fh, fw = stack.shape
    log(f"Stack: {stack.shape} {stack.dtype}  binning=×{bin_factor}  GPU={_GPU}")

    # ── 1. Prepare gain at binned and full resolution ──
    bh, bw = fh // bin_factor, fw // bin_factor

    if gain is not None:
        g_full = _apply_orientation(gain, gain_orientation)
        g_full = _match_gain_to_frame(g_full, fh, fw).astype(np.float32)
        g_binned = (_bin_image(g_full, bin_factor) / (bin_factor * bin_factor)).astype(np.float32)
    else:
        g_full = None
        g_binned = None

    # ── 2. Build rfft of prepared frames, streaming to control memory ──
    # Memory budget for 76 × 4096² × complex64 rfft: 76 × 4096 × 2049 × 8 = 5.1 GB (fits 11 GB GPU)
    log("Preparing CC frames + streaming rfft...")
    taper_cpu = _cosine_taper_2d((bh, bw), taper_frac)
    F_shape = (n_frames, bh, bw // 2 + 1)
    F = xp.empty(F_shape, dtype=xp.complex64)

    t0 = time.time()
    for i in range(n_frames):
        f = stack[i].astype(np.float32)
        if bin_factor > 1:
            f = _bin_image(f, bin_factor)
        if g_binned is not None:
            f = f * g_binned
        f = f - f.mean()
        f = f * taper_cpu
        f_xp = xp.asarray(f, dtype=xp.float32)
        F[i] = xp.fft.rfft2(f_xp)
        del f_xp
    if _GPU:
        _release_gpu_blocks()
    log(f"  rfft done in {time.time()-t0:.1f}s — F shape={F.shape}, "
        f"GPU mem ~{F.nbytes/1e9:.1f} GB")

    # Frequency grids for phase ramps (rfft layout)
    fy = xp.fft.fftfreq(bh).astype(xp.float32).reshape(-1, 1)     # (bh, 1)
    fx = xp.fft.rfftfreq(bw).astype(xp.float32).reshape(1, -1)    # (1, bw//2+1)

    # Initial shifts (binned pixels)
    sy = xp.zeros(n_frames, dtype=xp.float32)
    sx = xp.zeros(n_frames, dtype=xp.float32)

    convergence_history: list[float] = []
    cc_quality_history: list[float] = []
    t_start = time.time()

    # ── 3. Iterative refinement ──
    for it in range(max_iter):
        # B-factor envelope for this iteration
        if b_factor_fixed:
            b_f = b_factor_init
        else:
            b_f = max(b_factor_init * (b_factor_decay ** it), b_factor_min)
        env = _b_factor_envelope_rfft((bh, bw), b_f, xp=xp)

        # Build aligned sum S by streaming accumulation (avoids N-frame intermediate)
        # Convention A: sy[j] = motion; to align, shift by -sy[j]
        # Fourier multiplier to apply shift -sy: exp(-2πi·fy·(-sy)) = exp(+2πi·fy·sy)
        S = xp.zeros((bh, bw // 2 + 1), dtype=xp.complex64)
        for j in range(n_frames):
            sy_j = float(sy[j]); sx_j = float(sx[j])
            phase_j = xp.exp(xp.complex64(+2j * xp.pi) * (fy * sy_j + fx * sx_j))
            S += F[j] * phase_j
            del phase_j

        # For each frame, CC its running-average window against the reference.
        new_sy = np.zeros(n_frames, dtype=np.float32)
        new_sx = np.zeros(n_frames, dtype=np.float32)
        cc_peaks = np.zeros(n_frames, dtype=np.float32) if track_cc_quality else None
        r = running_avg_N // 2

        for i in range(n_frames):
            # Recompute shifted_F[i] instead of caching (memory win)
            sy_i = float(sy[i]); sx_i = float(sx[i])
            phase_i = xp.exp(xp.complex64(+2j * xp.pi) * (fy * sy_i + fx * sx_i))
            shifted_Fi = F[i] * phase_i

            # Reference FFT: leave-one-out or full aligned sum
            if leave_one_out:
                ref_fft = (S - shifted_Fi) * env
            else:
                ref_fft = S * env

            # Running-average window input
            lo = max(0, i - r)
            hi = min(n_frames, i + r + 1)
            if (hi - lo) == 1:
                win_F = F[i]
            elif shift_window:
                # Pre-align each window frame by its current shift estimate, then sum
                win_F = xp.zeros_like(F[i])
                for k in range(lo, hi):
                    sy_k = float(sy[k]); sx_k = float(sx[k])
                    phase_k = xp.exp(xp.complex64(+2j * xp.pi) * (fy * sy_k + fx * sx_k))
                    win_F = win_F + F[k] * phase_k
                    del phase_k
            else:
                # Default: sum unshifted window FFTs (matches Unblur)
                win_F = xp.sum(F[lo:hi], axis=0)
            win_F_env = win_F * env

            # Cross-correlation (Fourier product → real inverse)
            cc_fft = xp.conj(ref_fft) * win_F_env
            cc_real = xp.fft.irfft2(cc_fft, s=(bh, bw))

            # Subpixel peak = motion of frame i relative to reference
            dy, dx = _upsampled_dft_peak(cc_real, upsample_factor, xp=xp)
            new_sy[i] = dy
            new_sx[i] = dx

            # CC quality metric: peak-to-background ratio
            if track_cc_quality:
                cc_peak_val = float(xp.max(cc_real))
                cc_bg = float(xp.median(xp.abs(cc_real)))
                cc_peaks[i] = cc_peak_val / (cc_bg + 1e-12)

            del phase_i, shifted_Fi, ref_fft, win_F, win_F_env, cc_fft, cc_real

        del S
        if _GPU:
            _cp.get_default_memory_pool().free_all_blocks()

        if track_cc_quality:
            cc_quality_history.append(float(np.mean(cc_peaks)))

        # Savitzky-Golay smoothing (polyorder 4, odd window auto)
        polyorder = 4
        w_sg = min(n_frames - (n_frames + 1) % 2, 15)  # odd, ≤15 for short sequences
        if w_sg % 2 == 0:
            w_sg -= 1
        if w_sg < polyorder + 2:
            w_sg = polyorder + 3 if (polyorder + 3) % 2 == 1 else polyorder + 2
        if w_sg > n_frames:
            smooth_y = new_sy.copy()
            smooth_x = new_sx.copy()
        else:
            smooth_y = savgol_filter(new_sy, window_length=w_sg, polyorder=polyorder)
            smooth_x = savgol_filter(new_sx, window_length=w_sg, polyorder=polyorder)

        # Optional damping: blend new shifts with previous iteration's shifts.
        # damping = 0 means pure replace (default); damping=0.5 means 50/50 blend.
        sy_cpu = _cp.asnumpy(sy) if _GPU else np.asarray(sy)
        sx_cpu = _cp.asnumpy(sx) if _GPU else np.asarray(sx)
        if damping > 0.0 and it > 0:
            a = float(damping)
            smooth_y = (1 - a) * smooth_y + a * sy_cpu
            smooth_x = (1 - a) * smooth_x + a * sx_cpu

        # Convergence (super-res)
        max_change_super = max(
            float(np.max(np.abs(smooth_y - sy_cpu))),
            float(np.max(np.abs(smooth_x - sx_cpu))),
        ) * bin_factor
        convergence_history.append(max_change_super)

        # Commit
        sy = xp.asarray(smooth_y, dtype=xp.float32)
        sx = xp.asarray(smooth_x, dtype=xp.float32)

        elapsed = time.time() - t_start
        max_abs_sy = float(xp.max(xp.abs(sy))) * bin_factor
        max_abs_sx = float(xp.max(xp.abs(sx))) * bin_factor
        log(f"  iter {it+1:2d}/{max_iter}  [{elapsed:6.1f}s]  B={b_f:6.1f}  "
            f"w_sg={w_sg:2d}  max|sy|={max_abs_sy:6.3f}  max|sx|={max_abs_sx:6.3f}  "
            f"Δ={max_change_super:.4f} px")

        if max_change_super < tol_px and it > 0:
            log(f"Converged at iter {it+1} (Δ={max_change_super:.4f} < tol={tol_px}).")
            break

    # Free big FFT array before final sum (saves GPU memory for full-res accum)
    del F
    if _GPU:
        _release_gpu_blocks()

    final_sy_super = (_cp.asnumpy(sy) if _GPU else np.asarray(sy)) * bin_factor
    final_sx_super = (_cp.asnumpy(sx) if _GPU else np.asarray(sx)) * bin_factor

    # Preserve INTERNAL sx for the final-sum application step (which is correct).
    # Output sx gets negated to match MotionCor3 column-2 sign convention.
    # This does not affect the aligned sum — just the reported shifts_x value.
    internal_sx_super = final_sx_super.copy()

    # ── 4. Apply final shifts to full-res gain-corrected frames ──
    log("Applying final shifts to full-resolution frames...")
    if _GPU:
        fy_f = _cp.fft.fftfreq(fh).astype(_cp.float32).reshape(-1, 1)
        fx_f = _cp.fft.fftfreq(fw).astype(_cp.float32).reshape(1, -1)
        gain_gpu = _cp.asarray(g_full, dtype=_cp.float32) if g_full is not None else None
        accum = _cp.zeros((fh, fw), dtype=_cp.complex64)
        unalign = _cp.zeros((fh, fw), dtype=_cp.float32)
        for i in range(n_frames):
            frame_gpu = _cp.asarray(stack[i], dtype=_cp.float32)
            if gain_gpu is not None:
                frame_gpu = frame_gpu * gain_gpu
            unalign += frame_gpu
            Fi = _cp.fft.fft2(frame_gpu)
            dy_apply = -float(final_sy_super[i])
            dx_apply = -float(internal_sx_super[i])
            ph = _cp.exp(_cp.complex64(-2j * np.pi) * (fy_f * dy_apply + fx_f * dx_apply))
            accum += Fi * ph
            del frame_gpu, Fi, ph
        aligned_sum = _cp.asnumpy(_cp.real(_cp.fft.ifft2(accum)) / n_frames).astype(np.float32)
        unaligned_sum = _cp.asnumpy(unalign / n_frames).astype(np.float32)
        del accum, unalign, gain_gpu
        _release_gpu_blocks()
    else:
        aligned_sum = np.zeros((fh, fw), dtype=np.float32)
        unaligned_sum = np.zeros((fh, fw), dtype=np.float32)
        fy_f = np.fft.fftfreq(fh).astype(np.float32).reshape(-1, 1)
        fx_f = np.fft.fftfreq(fw).astype(np.float32).reshape(1, -1)
        for i in range(n_frames):
            frame = stack[i].astype(np.float32)
            if g_full is not None:
                frame = frame * g_full
            unaligned_sum += frame
            Fi = np.fft.fft2(frame)
            dy_apply = -float(final_sy_super[i])
            dx_apply = -float(internal_sx_super[i])
            ph = np.exp(-2j * np.pi * (fy_f * dy_apply + fx_f * dx_apply))
            shifted = np.real(np.fft.ifft2(Fi * ph)).astype(np.float32)
            aligned_sum += shifted
        aligned_sum /= n_frames
        unaligned_sum /= n_frames

    # X-sign convention: MotionCor3's column-2 "x shift" is negative of our
    # internal convention. Negate at output so shifts_x matches MC3 for
    # apples-to-apples comparison. Does NOT affect the aligned sum.
    output_sx = -final_sx_super

    return {
        "shifts_y": final_sy_super.tolist(),
        "shifts_x": output_sx.tolist(),
        "iterations": len(convergence_history),
        "convergence": convergence_history,
        "cc_quality": cc_quality_history,
        "aligned_sum": aligned_sum,
        "unaligned_sum": unaligned_sum,
        "bin_factor": bin_factor,
        "n_frames": n_frames,
    }

# _motion_correction_v3.py
"""Coarse-to-fine whole-frame motion correction — rederivation.

Replaces the single-reference iterative aligner (_motion_correction_v2.py) on the
high-motion tail. Two fixes vs v2: (1) the per-frame update ACCUMULATES the measured
residual shift (sy += dy) instead of replacing (sy = dy) — removing v2's 0.5*T damping
fixed point; (2) a coarse-to-fine binning pyramid makes hundreds-of-px motion findable.
The data term is bandpassed (high-pass kills the ice/illumination gradient; low-pass ~5 A).
No whole-frame smoothing model (RELION uses raw per-frame shifts; coord #117).

Imports v2's grounded primitives so _motion_correction_v2.py stays byte-identical (A/B baseline).
Co-designed with Linux (cryoSPARC oracle) + RELION 5.0.1 OSS source; see
notes/specs/2026-06-18-motion-aligner-rederivation-design.md and coord bus #114/#117.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import BSpline as _BSpline
from scipy.ndimage import median_filter
from scipy.optimize import fmin_l_bfgs_b as _fmin_lbfgs

from de_groundcrew.external.gc_motion._motion_correction_v2 import (
    _apply_orientation,
    _bin_image,
    _cosine_taper_2d,
    _match_gain_to_frame,
    _upsampled_dft_peak,
)

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


def _declean_hot_pixels(stack, *, sigma=8.0):
    """Conservative per-frame hot-pixel / X-ray-hit replacement before alignment.

    Flag pixels > median + sigma*1.4826*MAD (robust); replace with a 3x3 median.
    Conservative sigma -> clean frames untouched (equivalence). Host/NumPy.
    0 replaced -> the original `stack` object is returned. Motivated by movie 007385.
    """
    a = np.asarray(stack)
    out = None; total = 0
    for i in range(a.shape[0]):
        fr = a[i]
        med = np.median(fr); mad = np.median(np.abs(fr - med))
        thr = med + float(sigma) * 1.4826 * mad
        hot = fr > thr
        c = int(hot.sum())
        if c:
            if out is None:
                out = a.copy()
            out[i][hot] = median_filter(fr, size=3)[hot]
            total += c
    return (stack, 0) if out is None else (out, total)


def _adaptive_schedule(apix, *, coarse_apx=12.0, fine_cap_apx=3.0):
    """Bin schedule in PHYSICAL Å/px: coarsest ≈ coarse_apx, finest capped at fine_cap_apx
    (don't refine onto a grid finer than ~fine_cap_apx — where low-dose super-res fabricates).
    Geometric ×2 ladder, coarse→fine, deduped. apix-driven (TRM-free)."""
    finest = max(1, int(round(float(fine_cap_apx) / float(apix))))
    coarsest = max(finest, int(round(float(coarse_apx) / float(apix))))
    sched, b = [], coarsest
    while b > finest:
        sched.append(b)
        b = max(finest, b // 2)
    sched.append(finest)
    return tuple(dict.fromkeys(sched))


def _resolve_K_Z(n_frames):
    """Cubic-B-spline control-point count: round(n/12)+2 clamped to [3, 7] (more knots for
    longer movies). Single source of truth for motion_correct_v3 and _pyramid_solve."""
    return min(7, max(3, round(n_frames / 12) + 2))


def _prep_level_rfft(stack, gain_full, bin_factor, taper_frac, xp=np):
    """Bin -> gain -> mean-subtract -> cosine-taper -> rfft2, streamed per frame.

    gain_full is full-resolution (already oriented + matched) or None. At bin>1
    the gain is binned and averaged (divide by bin^2) to stay a multiplicative
    gain after the frames are summed by bin^2.
    """
    n, fh, fw = stack.shape
    bh, bw = fh // bin_factor, fw // bin_factor
    if gain_full is not None and bin_factor > 1:
        g = (_bin_image(gain_full, bin_factor) / (bin_factor * bin_factor)).astype(np.float32)
    else:
        g = gain_full
    taper = _cosine_taper_2d((bh, bw), taper_frac)
    F = xp.empty((n, bh, bw // 2 + 1), dtype=xp.complex64)
    for i in range(n):
        f = stack[i].astype(np.float32)
        if bin_factor > 1:
            f = _bin_image(f, bin_factor)
        if g is not None:
            f = f * g
        f = f - f.mean()
        f = f * taper
        F[i] = xp.fft.rfft2(xp.asarray(f, dtype=xp.float32))
    if _GPU and xp is _cp:
        _release_gpu_blocks()
    return F, bw


def _prep_level_dog_hann(stack, gain_full, bin_factor, *, sigma_lo=1.0, sigma_hi=6.0, xp=np):
    """Bin -> gain -> difference-of-Gaussians(sigma_lo,sigma_hi) -> Hann window -> rfft2.

    The super-res fallback prep (Linux #158). The HANN WINDOW is load-bearing: its strong
    real-space apodization kills the FFT edge-wrap / static zero-shift correlation that
    collapses the reference-to-sum on SNR-empty super-res frames (DoG+Hann -> 231px vs
    DoG+weak-taper -> 34px collapse). The DoG(s1-s6) sets the magnitude AND bakes the narrow
    band into the frames, so the fallback reference-to-sum must NOT apply an additional CC
    band (no double-banding). Mirrors _prep_level_rfft's CPU-prep / GPU-rfft split.
    """
    from scipy.ndimage import gaussian_filter
    n, fh, fw = stack.shape
    bh, bw = fh // bin_factor, fw // bin_factor
    if gain_full is not None and bin_factor > 1:
        g = (_bin_image(gain_full, bin_factor) / (bin_factor * bin_factor)).astype(np.float32)
    else:
        g = gain_full
    han = np.outer(np.hanning(bh), np.hanning(bw)).astype(np.float32)
    F = xp.empty((n, bh, bw // 2 + 1), dtype=xp.complex64)
    for i in range(n):
        f = stack[i].astype(np.float32)
        if bin_factor > 1:
            f = _bin_image(f, bin_factor)
        if g is not None:
            f = f * g
        f = (gaussian_filter(f, sigma_lo) - gaussian_filter(f, sigma_hi)).astype(np.float32)
        f = f * han
        F[i] = xp.fft.rfft2(xp.asarray(f, dtype=xp.float32))
    if _GPU and xp is _cp:
        _release_gpu_blocks()
    return F, bw


def _cos_edge(s, cut, width, rising, xp):
    """Cosine ramp over [cut-width, cut+width]. rising: 0->1; falling: 1->0."""
    t = xp.clip((s - (cut - width)) / (2.0 * width + 1e-12), 0.0, 1.0)
    ramp = 0.5 * (1.0 - xp.cos(xp.pi * t))          # 0 -> 1 across the window
    return ramp if rising else (1.0 - ramp)


def _bandpass_rfft(shape, apix, res_low_A, res_high_A, xp=np, soft=0.5):
    """Soft cosine bandpass on the rfft grid (H, W//2+1).

    Passes spatial frequencies between 1/res_low_A (high-pass; kills the broad
    ice/illumination gradient) and 1/res_high_A (low-pass; ~5 A). Frequency in
    cycles/pixel = apix / res_A. `soft` is the cosine roll-off as a fraction of
    each cutoff.
    """
    h, w = shape
    fy = xp.fft.fftfreq(h).astype(xp.float32).reshape(-1, 1)
    fx = xp.fft.rfftfreq(w).astype(xp.float32).reshape(1, -1)
    s = xp.sqrt(fy * fy + fx * fx)
    s_lo = float(apix) / float(res_low_A)            # high-pass cutoff (low freq)
    s_hi = float(apix) / float(res_high_A)           # low-pass cutoff (high freq)
    hp = _cos_edge(s, s_lo, max(s_lo * soft, 1e-6), rising=True, xp=xp)
    lp = _cos_edge(s, s_hi, max(s_hi * soft, 1e-6), rising=False, xp=xp)
    return (hp * lp).astype(xp.float32)


def _signal_band_weight(shape, apix, *, B=500.0, res_hp_A=35.0, res_lp_A=5.5, soft=0.5, xp=np):
    """B-factor-weighted signal band on the rfft grid (H, W//2+1).

    w(k) = highpass(1/res_hp) * lowpass(1/res_lp) * exp(-2 B f^2), f in 1/Angstrom.
    The attractor-breaker (R&B-2015 / cryoSPARC rigid): the high-pass kills the static
    low-freq (illumination/carbon), the B-envelope + low-pass keep the mid-res specimen
    band where the moving signal beats the static. Used for BOTH the solve objective and
    the half-sum confidence. B/res_hp/res_lp are calibrated (swept on the 36 + held-out).
    """
    h, w = shape
    fy = xp.fft.fftfreq(h).astype(xp.float32).reshape(-1, 1)
    fx = xp.fft.rfftfreq(w).astype(xp.float32).reshape(1, -1)
    s = xp.sqrt(fy * fy + fx * fx)                      # cycles/pixel
    f_per_A = s / float(apix)                           # cycles/Angstrom
    s_hp = float(apix) / float(res_hp_A)
    s_lp = float(apix) / float(res_lp_A)
    # cutoffs may exceed the grid Nyquist at coarse bins; the cos-edge clip yields
    # all-pass, which is correct -- do not clamp.
    hp = _cos_edge(s, s_hp, max(s_hp * soft, 1e-6), rising=True, xp=xp)
    lp = _cos_edge(s, s_lp, max(s_lp * soft, 1e-6), rising=False, xp=xp)
    env = xp.exp(-2.0 * float(B) * f_per_A * f_per_A)
    return (hp * lp * env).astype(xp.float32)


def _mask_cc_to_radius(cc_real, radius, xp):
    """Zero the CC outside a ±radius wrap-around window of the origin so the peak
    search is bounded — rejects far-field noise locks (RELION search_range analog).
    radius is in this level's binned pixels; no-op if it covers the whole frame."""
    bh, bw = cc_real.shape
    r = int(radius)
    if r <= 0 or (2 * r + 1) >= min(bh, bw):
        return cc_real
    mask = xp.zeros((bh, bw), dtype=cc_real.dtype)
    mask[:r + 1, :r + 1] = 1.0
    mask[:r + 1, bw - r:] = 1.0
    mask[bh - r:, :r + 1] = 1.0
    mask[bh - r:, bw - r:] = 1.0
    return cc_real * mask


def _solve_level(F, bandpass, sy, sx, *, max_iter, tol_px, upsample_factor,
                 max_shift_px=64, leave_one_out=True, bw=None, xp=np, log=lambda m: None):
    """Leave-one-out Fourier-CC solve at one bin level, accumulate-residual update.

    For each frame i: CC its current-shift-applied FFT against the leave-one-out
    aligned sum (sum of all OTHER frames, rebuilt each iter), take the global
    subpixel peak (the residual misalignment dy,dx), and ACCUMULATE: sy[i] += dy.
    Iterate until max|residual| < tol_px. No damping — the accumulate update has its
    fixed point at the true trajectory, not 0.5*T. (RELION alignPatch objective.)

    sy/sx are in this level's binned pixels. bandpass is the rfft-grid CC weight.
    bw is the TRUE binned width (fw // bin_factor); when None, fall back to
    (bwh-1)*2 (legacy callers passing only F). The true bw is odd-safe.
    """
    n, bh, bwh = F.shape
    if bw is None:
        bw = (bwh - 1) * 2
    fy = xp.fft.fftfreq(bh).astype(xp.float32).reshape(-1, 1)
    fx = xp.fft.rfftfreq(bw).astype(xp.float32).reshape(1, -1)
    sy = xp.asarray(sy, dtype=xp.float32).copy()
    sx = xp.asarray(sx, dtype=xp.float32).copy()
    convergence = []
    for it in range(max_iter):
        # Aligned sum S = sum_j F[j] * exp(+2pi i (fy*sy_j + fx*sx_j))  (shift by -sy_j)
        S = xp.zeros((bh, bwh), dtype=xp.complex64)
        for j in range(n):
            ph = xp.exp(xp.complex64(2j * xp.pi) * (fy * float(sy[j]) + fx * float(sx[j])))
            S += F[j] * ph
            del ph
        new_sy = sy.copy()
        new_sx = sx.copy()
        max_resid = 0.0
        for i in range(n):
            ph_i = xp.exp(xp.complex64(2j * xp.pi) * (fy * float(sy[i]) + fx * float(sx[i])))
            shifted_Fi = F[i] * ph_i
            ref = (S - shifted_Fi) if leave_one_out else S
            cc_fft = xp.conj(ref) * shifted_Fi * bandpass
            cc_real = xp.fft.irfft2(cc_fft, s=(bh, bw))
            cc_real = _mask_cc_to_radius(cc_real, max_shift_px, xp)
            dy, dx = _upsampled_dft_peak(cc_real, upsample_factor, xp=xp)
            new_sy[i] = float(sy[i]) + dy          # ACCUMULATE residual (the fix)
            new_sx[i] = float(sx[i]) + dx
            if abs(dy) > max_resid:
                max_resid = abs(dy)
            if abs(dx) > max_resid:
                max_resid = abs(dx)
            del ph_i, shifted_Fi, ref, cc_fft, cc_real
        sy, sx = new_sy, new_sx
        convergence.append(float(max_resid))
        log(f"    level iter {it + 1}/{max_iter}: max_resid={max_resid:.3f} px")
        del S
        if _GPU and xp is _cp:
            _cp.get_default_memory_pool().free_all_blocks()
        if max_resid < tol_px and it > 0:
            break
    return sy, sx, convergence


def _pyramid_solve(stack, gain_full, apix, *, schedule, coarse_seed,
                   res_low_A, res_high_A, max_iter, tol_px, upsample_factor,
                   taper_frac, lp_C=2.75, hp_cyc_per_px=0.03, max_shift_px=64,
                   K_Z=None, B=500.0, res_hp_A=35.0, res_lp_A=5.5, lam=2e-3,
                   trust_px=None, knot_blend_w=0.0, knot_floor_frames=1.5,
                   seed_damping=0.6,
                   seed_reject=True, seed_reject_floor_A=60.0, seed_reject_k_mad=5.0,
                   regime_detector="excursion", excursion_max_A=1200.0,
                   prominence_z_thresh=15.0, fallback_mask_px=16, fallback_iters=12,
                   xp=np, log=lambda m: None):
    """Coarse-to-fine binning pyramid. Per level: build the B-factor signal-band
    weight and run the JOINT SMOOTH solve (_solve_level_smooth — cubic-B-spline
    trajectory, banded aligned-sum power, 2nd-deriv penalty) seeded from the coarser
    level (or the cumulative-CC coarse seed on the coarsest). Rescale the trajectory
    into the next finer level's pixels between levels. Returns shifts in input (bin=1)
    pixels.

    Replaces the round-1 greedy leave-one-out _solve_level (now legacy; still unit
    tested). res_low_A/res_high_A/max_iter/tol_px/upsample_factor/max_shift_px are
    retained only for the coarse-seed CC step (a primitive that still uses a bandpass).
    The solve objective uses the signal-band weight (B/res_hp_A/res_lp_A) instead.
    """
    n = stack.shape[0]
    if K_Z is None:
        K_Z = _resolve_K_Z(n)
    # The trajectory (sy/sx) is a HOST quantity: the smooth solve runs L-BFGS over the
    # B-spline coeffs on the CPU (only the aligned-power FFT sum runs on xp/GPU), and
    # _solve_level_smooth returns NumPy. Keep the seed/init on the host so it feeds the
    # solver's np.linalg.lstsq cleanly (CuPy refuses implicit np.asarray conversion).
    to_host = (lambda a: _cp.asnumpy(a)) if (_GPU and xp is _cp) else np.asarray
    sy = np.zeros(n, dtype=np.float32)
    sx = np.zeros(n, dtype=np.float32)
    all_conv = []
    prev_bin = None
    Phi = None
    seed_sy = seed_sx = None                            # populated at coarsest level if coarse_seed
    for b in schedule:                                  # coarse -> fine
        t0 = time.time()
        F, bw = _prep_level_rfft(stack, gain_full, b, taper_frac, xp=xp)
        bh, bwh = F.shape[1], F.shape[2]
        level_apix = apix * b
        weight = _signal_band_weight((bh, bw), level_apix, B=B,
                                     res_hp_A=res_hp_A, res_lp_A=res_lp_A, xp=xp)
        if prev_bin is None:
            if coarse_seed:
                res_hi_eff = max(res_high_A, lp_C * level_apix)
                res_lo_eff = level_apix / hp_cyc_per_px
                seed_bp = _bandpass_rfft((bh, bw), level_apix, res_lo_eff, res_hi_eff, xp=xp)
                # DEFAULT: consecutive seed (seats standard large motion).
                sy, sx = _coarse_seed_consecutive(F, seed_bp,
                                                  upsample_factor=max(2, upsample_factor // 2),
                                                  max_shift_px=max_shift_px,
                                                  reject_outliers=seed_reject,
                                                  level_apix=level_apix,
                                                  reject_floor_A=seed_reject_floor_A,
                                                  reject_k_mad=seed_reject_k_mad, bw=bw, xp=xp)
                sy, sx = to_host(sy), to_host(sx)
                excursion_A = float(max(np.abs(sy).max(), np.abs(sx).max())) * level_apix
                if regime_detector == "prominence":
                    zmed = _consecutive_prominence_z(F, seed_bp, max_shift_px=max_shift_px, bw=bw, xp=xp)
                    super_res = zmed < prominence_z_thresh
                    detlog = f"prominence z={zmed:.1f}<{prominence_z_thresh}"
                else:
                    super_res = excursion_A > excursion_max_A
                    detlog = f"excursion={excursion_A:.0f}A>{excursion_max_A:.0f}A"
                if super_res:
                    # FALLBACK: reference-to-sum on DoG+Hann frames (no CC band — DoG is in the prep).
                    F_dh, _bw_dh = _prep_level_dog_hann(stack, gain_full, b, xp=xp)
                    flat = xp.ones((bh, bwh), dtype=xp.float32)
                    sy, sx = _coarse_seed(F_dh, flat,
                                          upsample_factor=max(2, upsample_factor // 2),
                                          max_shift_px=fallback_mask_px, n_iter=fallback_iters,
                                          damping=seed_damping, bw=bw, xp=xp)
                    sy, sx = to_host(sy), to_host(sx)
                    log(f"  coarse seed=fallback (ref-to-sum, DoG+Hann); {detlog} -> super-res")
                    del F_dh, flat
                else:
                    log(f"  coarse seed=consecutive; {detlog} -> standard")
                del seed_bp
                # snapshot the coarse seed in INPUT (bin=1) px for the fail-loud collapse signal
                seed_sy = (to_host(sy) * float(b)).astype(np.float32)
                seed_sx = (to_host(sx) * float(b)).astype(np.float32)
            # Build the spline basis ONCE from the seed (arc-length placement needs the
            # seed's motion profile) and reuse it at every level for a consistent basis.
            n_interior = max(int(K_Z), 4) - 3 - 1
            if knot_blend_w > 0.0 and n_interior > 0:
                interior = _arclength_blended_knots(sy, sx, n_interior,
                                                    w=knot_blend_w,
                                                    floor_frames=knot_floor_frames)
            else:
                interior = None
            Phi = _bspline_basis(n, K_Z, interior=interior)
        else:
            ratio = float(prev_bin) / float(b)          # >1 going finer
            sy = sy * ratio
            sx = sx * ratio
        sy, sx, fval = _solve_level_smooth(F, weight, sy, sx, Phi=Phi, lam=lam,
                                           trust_px=trust_px, bw=bw, xp=xp, log=log)
        all_conv.append(float(fval))                    # one objective value per level
        log(f"  level bin={b}: smooth solve, fval={fval:.4g}, "
            f"{time.time() - t0:.1f}s")
        prev_bin = b
        del F, weight
        if _GPU and xp is _cp:
            _release_gpu_blocks()
    finest = schedule[-1]
    return sy * finest, sx * finest, all_conv, seed_sy, seed_sx   # -> input (bin=1) px


def _apply_shifts_fullres(stack, gain_full, sy, sx, *, dose_weight=None, xp=np):
    """Apply -shift to each full-res (optionally gain-corrected) frame via Fourier
    phase and accumulate the aligned + unaligned sums. sy/sx in input pixels.

    Both CPU and GPU phase-shift each frame's FFT and accumulate in the Fourier domain,
    then call ifft2 once on the sum — mathematically equivalent (by linearity of the
    IFFT) to _motion_correction_v2.py's per-frame real-domain accumulation, with a
    single inverse transform.

    dose_weight: optional DoseWeightParams. When set, a SECOND pair of Fourier-domain
    accumulators (the per-frequency weighted sum and the normalizer map Σ_i W_i) is
    filled in the SAME loop / SAME phase as the aligned sum, then
    dw_sum = real(ifft2(weighted_sum / Σ_i W_i)) — the per-frequency normalized
    weighted average (matches workers/dose_weighting.py). DW-off leaves the aligned/
    unaligned sums byte-identical (the dw_* buffers are independent). Returns
    (aligned_sum, unaligned_sum, dw_sum); dw_sum is None when dose_weight is None.
    """
    n, fh, fw = stack.shape
    sy = np.asarray(sy, dtype=np.float64)
    sx = np.asarray(sx, dtype=np.float64)
    fy = xp.fft.fftfreq(fh).astype(xp.float32).reshape(-1, 1)
    fx = xp.fft.fftfreq(fw).astype(xp.float32).reshape(1, -1)
    g = xp.asarray(gain_full, dtype=xp.float32) if gain_full is not None else None
    accum = xp.zeros((fh, fw), dtype=xp.complex64)
    unalign = xp.zeros((fh, fw), dtype=xp.float32)
    do_dw = dose_weight is not None
    if do_dw:
        from de_groundcrew.external.gc_motion.dose_weighting import (
            dose_weight_map, frame_doses)
        _doses = frame_doses(n, dose_weight.total_dose,
                             dose_weight.n_total_frames, dose_weight.frame_offset)
        accum_dw = xp.zeros((fh, fw), dtype=xp.complex64)
        wsum_dw = xp.zeros((fh, fw), dtype=xp.float32)
    for i in range(n):
        frame = xp.asarray(stack[i], dtype=xp.float32)
        if g is not None:
            frame = frame * g
        unalign += frame
        Fi = xp.fft.fft2(frame)
        ph = xp.exp(xp.complex64(-2j * np.pi) * (fy * (-float(sy[i])) + fx * (-float(sx[i]))))
        shifted = Fi * ph
        accum += shifted
        if do_dw:
            W = dose_weight_map((fh, fw), dose_weight.apix, float(_doses[i]),
                                dose_weight.voltage_kv, xp=xp)
            accum_dw += shifted * W
            wsum_dw += W
            del W
        del frame, Fi, ph, shifted
    aligned = xp.real(xp.fft.ifft2(accum)) / n
    aligned_sum = (_cp.asnumpy(aligned) if (_GPU and xp is _cp) else np.asarray(aligned)).astype(np.float32)
    unaligned_sum = (_cp.asnumpy(unalign) if (_GPU and xp is _cp) else np.asarray(unalign)).astype(np.float32) / n
    if do_dw:
        dw = xp.real(xp.fft.ifft2(accum_dw / xp.maximum(wsum_dw, 1e-8)))
        dw_sum = (_cp.asnumpy(dw) if (_GPU and xp is _cp) else np.asarray(dw)).astype(np.float32)
        del accum_dw, wsum_dw, dw
    else:
        dw_sum = None
    del accum, unalign
    if g is not None:
        del g
    if _GPU and xp is _cp:
        _release_gpu_blocks()
    return aligned_sum, unaligned_sum, dw_sum


def _assess_confidence(sy, sx, seed_sy, seed_sx, aligned_sum, unaligned_sum, apix, *,
                       max_path_A=1500.0, max_per_frame_A=120.0,
                       collapse_floor_A=12.0, collapse_seed_min_A=15.0, xp=np):
    """Post-solve internal-signal confidence (fail-loud; ADVISORY -- never alters the shifts).
    Returns (low_confidence: bool, reason: str, signals: dict) — `signals` exposes the 4 RAW
    magnitude values per movie (max_step_A, total_path_A, seed_path_A, collapse_frac) for
    distribution-based threshold calibration. Ground-truth-free; flags the gross 'wrong
    correction' classes by MAGNITUDE. reason='' when confident. (aligned_sum/unaligned_sum are
    retained for signature stability + a possible future signal; not read here -- see below.)
    Collapse uses a LOW SEED-FLOOR gate (Task 9): flag when the seed found real motion
    (seed_path > collapse_seed_min_A) yet the output is near-zero (total_path < collapse_floor_A)
    -- catches small-seed collapses while sparing genuinely-still movies (seed ~0) and the
    noisy-runaway-seed-but-correct-output class.
    NOT caught (accepted fail-safe misses): wrong-direction collapses whose output MAGNITUDE is
    normal. A reference-free aligned-sum sharpness signal (band-power ratio / spectral slope / CTF)
    was tested to catch these and proven mechanistically unable to (Linux smoke 2026-06-26): the
    aligner optimizes sum sharpness, so a wrong-direction local optimum reads as sharp (pa/pu>1; the
    CTF fit even inverts) -- any signal off that sum inherits the aligner's own blind spot. The only
    viable lever is a signal ORTHOGONAL to the aligner objective (a future arc). Detectability tracks
    motion MAGNITUDE, not direction; the rails + collapse gate cover everything magnitude-detectable."""
    sy = np.asarray(sy, np.float64); sx = np.asarray(sx, np.float64)
    dpath = np.hypot(np.diff(sy), np.diff(sx)) * float(apix)
    total_path_A = float(dpath.sum()); max_step_A = float(dpath.max()) if dpath.size else 0.0
    sdy = np.diff(np.asarray(seed_sy, np.float64)); sdx = np.diff(np.asarray(seed_sx, np.float64))
    seed_path_A = float(np.hypot(sdy, sdx).sum()) * float(apix)
    collapse_frac = (total_path_A / seed_path_A) if seed_path_A > 0 else float("inf")  # informational-only: recorded, not gated
    # JSON-safe signals: Inf (degenerate) -> None so a JSON/IPC sink can serialize them.
    signals = {"max_step_A": max_step_A, "total_path_A": total_path_A, "seed_path_A": seed_path_A,
               "collapse_frac": collapse_frac if np.isfinite(collapse_frac) else None}
    if max_step_A > max_per_frame_A:
        return True, f"per_frame_shift {max_step_A:.0f}A>{max_per_frame_A:.0f}A", signals
    if total_path_A > max_path_A:
        return True, f"path {total_path_A:.0f}A>{max_path_A:.0f}A", signals
    if seed_path_A > collapse_seed_min_A and total_path_A < collapse_floor_A:
        return True, f"collapse: output {total_path_A:.0f}A<{collapse_floor_A:.0f}A (seed {seed_path_A:.0f}A)", signals
    return False, "", signals


def motion_correct_v3(
    stack: np.ndarray,
    gain: Optional[np.ndarray] = None,
    gain_orientation: int = 0,
    *,
    apix: float = 1.0,
    schedule="auto",
    coarse_seed: bool = True,
    res_low_A: float = 40.0,
    res_high_A: float = 5.0,
    max_iter: int = 10,
    tol_px: float = 0.05,
    upsample_factor: int = 10,
    taper_frac: float = 0.025,
    lp_C: float = 2.75,
    hp_cyc_per_px: float = 0.03,
    max_shift_px: int = 64,
    coarse_apx: float = 12.0,
    fine_cap_apx: float = 3.0,
    B: float = 500.0,
    res_hp_A: float = 35.0,
    res_lp_A: float = 5.5,
    lam: float = 2e-3,
    K_Z: Optional[int] = None,
    trust_px: Optional[float] = 64.0,
    knot_blend_w: float = 0.8,
    knot_floor_frames: float = 1.5,
    seed_damping: float = 0.6,
    regime_detector: str = "excursion",
    excursion_max_A: float = 1200.0,
    prominence_z_thresh: float = 15.0,
    fallback_mask_px: int = 16,
    fallback_iters: int = 12,
    fail_loud: bool = True,
    max_path_A: float = 1500.0,
    max_per_frame_A: float = 120.0,
    collapse_floor_A: float = 12.0,
    collapse_seed_min_A: float = 15.0,
    hot_pixel_clean: bool = False,
    hot_pixel_sigma: float = 8.0,
    dose_weight=None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Coarse-to-fine whole-frame motion correction. Drop-in for the validation
    harness (motion_correct_v2-shaped). Outputs raw per-frame shifts (no smoothing).
    shifts_x is output-negated to match the MotionCor3 sign (harness GC_X_NEGATED=True).

    schedule="auto" (default) resolves the bin ladder from the pixel size via
    _adaptive_schedule(apix, coarse_apx, fine_cap_apx); an explicit tuple overrides.
    K_Z=None resolves to round(n_frames/12)+2 clamped to [3, 7] (more knots for longer
    movies). Each level runs the joint smooth solve (B-factor signal-band weight +
    cubic-B-spline trajectory under a 2nd-deriv penalty), replacing round-1's greedy CC.
    trust_px=64.0 (default-on): per-frame trust-region bound (px) passed to each level's
    L-BFGS solve — prevents runaway steps under low-contrast or noisy frames.
    knot_blend_w=0.8 (default-on): arc-length blending weight for interior knot placement
    (0=uniform, 1=pure arc-length); floor_frames=1.5 keeps knots from crowding on short movies.
    seed_damping=0.6 (default-on): the reference-to-sum fallback seed's damped-update factor
    (Linux pre-val: converges in 3-8 iters). The seed's per-step CC is bounded by max_shift_px
    (consecutive) / fallback_mask_px (fallback); runaways are caught by the excursion gate and
    the seed outlier-step rejection -- the validated guards.
    HYBRID coarse seed (default-on): the consecutive seed runs by default (seats standard
    large smooth motion); if it RAN AWAY (excursion_max_A, default 1200 A — above the largest
    legitimate motion ~716 A, below super-res runaways ~1700-2600 A) the seed falls back to the
    reference-to-sum on DoG+Hann-prepped frames (the super-res regime). regime_detector="prominence"
    selects the median consecutive-CC prominence-z gate (prominence_z_thresh) instead, for A/B.
    """
    xp = _cp if _GPU else np
    log = progress_cb or (lambda msg: None)
    n, fh, fw = stack.shape

    # DEFAULT-OFF (correctness-arc A/B finding, Linux-concurred): the per-frame
    # sigma-threshold clean flags real bright features (carbon edges / fiducials /
    # contamination) as defects on some movies -- it corrupts in-lane alignments
    # (023103 magY 1.01->7.65) and flips super-res safe-collapse into a runaway
    # (041921 magY 0.008->36). Proper hot-pixel handling = a fixed-pattern defect
    # map (DE cameras ship one) / temporal-outlier test, a follow-on -- not this.
    if hot_pixel_clean:
        stack, n_hot = _declean_hot_pixels(stack, sigma=hot_pixel_sigma)
        if n_hot:
            log(f"hot-pixel pre-clean: replaced {n_hot} px across {n} frames")

    if schedule == "auto":
        schedule = _adaptive_schedule(apix, coarse_apx=coarse_apx, fine_cap_apx=fine_cap_apx)
    else:
        schedule = tuple(schedule)
    if K_Z is None:
        K_Z = _resolve_K_Z(n)
    log(f"v3 aligner: stack {stack.shape} {stack.dtype}  schedule={schedule}  "
        f"K_Z={K_Z}  GPU={_GPU}")

    if gain is not None:
        g_full = _apply_orientation(gain, gain_orientation)
        g_full = _match_gain_to_frame(g_full, fh, fw).astype(np.float32)
    else:
        g_full = None

    sy, sx, convergence, seed_sy, seed_sx = _pyramid_solve(
        stack, g_full, apix, schedule=schedule, coarse_seed=coarse_seed,
        res_low_A=res_low_A, res_high_A=res_high_A, max_iter=max_iter, tol_px=tol_px,
        upsample_factor=upsample_factor, taper_frac=taper_frac,
        lp_C=lp_C, hp_cyc_per_px=hp_cyc_per_px, max_shift_px=max_shift_px,
        K_Z=K_Z, B=B, res_hp_A=res_hp_A, res_lp_A=res_lp_A, lam=lam,
        trust_px=trust_px, knot_blend_w=knot_blend_w, knot_floor_frames=knot_floor_frames,
        seed_damping=seed_damping,
        regime_detector=regime_detector, excursion_max_A=excursion_max_A,
        prominence_z_thresh=prominence_z_thresh, fallback_mask_px=fallback_mask_px,
        fallback_iters=fallback_iters,
        xp=xp, log=log)

    sy = _cp.asnumpy(sy) if _GPU else np.asarray(sy)        # input px
    sx = _cp.asnumpy(sx) if _GPU else np.asarray(sx)

    aligned_sum, unaligned_sum, dw_sum = _apply_shifts_fullres(
        stack, g_full, sy, sx, dose_weight=dose_weight, xp=xp)

    # seed_sy/seed_sx are None when coarse_seed=False (no seed snapshot) -> skip the seed-based guard
    if fail_loud and seed_sy is not None:
        low_conf, reason, conf_signals = _assess_confidence(
            sy, sx, seed_sy, seed_sx, aligned_sum, unaligned_sum, apix,
            max_path_A=max_path_A, max_per_frame_A=max_per_frame_A,
            collapse_floor_A=collapse_floor_A, collapse_seed_min_A=collapse_seed_min_A, xp=xp)
        if low_conf:
            log(f"FAIL-LOUD: low confidence -- {reason}")
    else:
        low_conf, reason, conf_signals = False, "", {}

    return {
        "shifts_y": np.asarray(sy, dtype=np.float64).tolist(),
        "shifts_x": (-np.asarray(sx, dtype=np.float64)).tolist(),   # MC3 sign
        "aligned_sum": aligned_sum,
        "unaligned_sum": unaligned_sum,
        "dw_sum": dw_sum,
        "convergence": convergence,
        "iterations": len(convergence),
        "n_frames": int(n),
        "bin_factor": int(schedule[-1]),
        "schedule": list(schedule),
        "low_confidence": bool(low_conf),
        "failure_reason": reason,
        "confidence_signals": conf_signals,
    }


def _reject_outlier_steps(steps_y, steps_x, level_apix, *, floor_A=60.0, k_mad=5.0):
    """Robustly clean outlier per-step consecutive-CC increments (gate-gap fix).

    Flag steps whose magnitude (Å) exceeds max(floor_A, median + k_mad*1.4826*MAD); replace
    each flagged step with the componentwise median of the un-flagged steps. Host/NumPy.
    0 flagged -> inputs returned unchanged (byte-identical seed downstream).
    Validated recipe: D:\\validation-handoff\\gate_fix_RESULTS.md (Linux #169).
    """
    sy = np.asarray(steps_y, dtype=np.float64)
    sx = np.asarray(steps_x, dtype=np.float64)
    mag = np.hypot(sy, sx) * float(level_apix)
    if mag.size < 2:                                 # 0/1-frame degenerate: nothing to reject
        return steps_y, steps_x, np.zeros(mag.shape, dtype=bool)
    # Exclude the synthetic frame-0 zero (steps[0]=0, frame 0 has no predecessor) from
    # BOTH the threshold statistics and the replacement median -- including it biases
    # the median/MAD low and the threshold down. Net effect: threshold rises slightly
    # -> rejection is more conservative (safer). Linux #194.
    mag1 = mag[1:]
    med = np.median(mag1)
    mad = 1.4826 * np.median(np.abs(mag1 - med))    # 1.4826 = σ-consistent MAD scale (matches _declean_hot_pixels; Linux #174)
    thresh = max(float(floor_A), med + float(k_mad) * mad)
    flagged = mag > thresh
    flagged[0] = False                               # never flag the synthetic frame-0 zero
    if not flagged.any():
        return steps_y, steps_x, flagged
    keep = ~flagged
    keep[0] = False                                  # exclude frame-0 zero from the replacement median too
    repl_y = float(np.median(sy[keep])) if keep.any() else 0.0
    repl_x = float(np.median(sx[keep])) if keep.any() else 0.0
    out_y = sy.copy(); out_x = sx.copy()
    out_y[flagged] = repl_y; out_x[flagged] = repl_x
    return out_y.astype(np.float32), out_x.astype(np.float32), flagged


def _coarse_seed_consecutive(F, bandpass, *, upsample_factor, max_shift_px=64,
                             reject_outliers=True, level_apix=1.0,
                             reject_floor_A=60.0, reject_k_mad=5.0, bw=None, xp=np):
    """Cumulative consecutive-frame global-CC seed (the hybrid DEFAULT).

    CC frame i against frame i-1 (bounded ±max_shift_px), cumulatively sum -> an initial
    trajectory anchored at frame 0. Seats hundreds-of-px standard large smooth motion that
    the reference-to-sum seed intrinsically under-seats (Linux #154: 131393 shape-corr 1.00).
    Runs away on SNR-empty super-res (the hybrid detects that by excursion and falls back).
    sy/sx in this level's binned px.

    Two-pass: pass 1 collects per-step increments; pass 2 (when reject_outliers=True) cleans
    them via _reject_outlier_steps before cumsum (gate-gap fix). When 0 steps are flagged the
    cleaned steps equal the raw steps -> byte-identical seed downstream (equivalence invariant).
    """
    n, bh, bwh = F.shape
    if bw is None:
        bw = (bwh - 1) * 2
    steps_y = np.zeros(n, dtype=np.float64)                # f64 accumulation == the pre-refactor seed
    steps_x = np.zeros(n, dtype=np.float64)                # (rounding steps to f32 before cumsum perturbs ~1e-6 px)
    for i in range(1, n):                                  # pass 1: per-step increments
        cc_fft = xp.conj(F[i - 1]) * F[i] * bandpass
        cc_real = xp.fft.irfft2(cc_fft, s=(bh, bw))
        cc_real = _mask_cc_to_radius(cc_real, max_shift_px, xp)
        dy, dx = _upsampled_dft_peak(cc_real, upsample_factor, xp=xp)
        steps_y[i] = float(dy); steps_x[i] = float(dx)
        del cc_fft, cc_real
    if reject_outliers:                                    # pass 2: clean outliers (gate-gap fix)
        steps_y, steps_x, _flag = _reject_outlier_steps(
            steps_y, steps_x, level_apix, floor_A=reject_floor_A, k_mad=reject_k_mad)
    sy = xp.asarray(np.cumsum(np.asarray(steps_y, dtype=np.float64)), dtype=xp.float32)
    sx = xp.asarray(np.cumsum(np.asarray(steps_x, dtype=np.float64)), dtype=xp.float32)
    return sy, sx


def _consecutive_prominence_z(F, bandpass, *, max_shift_px=64, bw=None, xp=np):
    """Median (over consecutive frames) prominence z-score of the consecutive-frame CC.

    Per pair: z = (peak - mean)/std over the ±max_shift_px centered (fftshifted) disk.
    A coarse-bin signal-quality / motion-magnitude proxy (Linux #158): high on standard
    large motion (~19-41), low on SNR-empty super-res (~4-5). Selectable regime detector
    (the excursion detector is the default). Movie-level (median), never per-frame.
    """
    n, bh, bwh = F.shape
    if bw is None:
        bw = (bwh - 1) * 2
    yy, xx = np.indices((bh, bw))
    valid = np.hypot(yy - bh // 2, xx - bw // 2) <= int(max_shift_px)
    zs = []
    for i in range(1, n):
        cc_fft = xp.conj(F[i - 1]) * F[i] * bandpass
        cc_real = xp.fft.irfft2(cc_fft, s=(bh, bw))
        cc = np.fft.fftshift(_cp.asnumpy(cc_real) if (_GPU and xp is _cp) else np.asarray(cc_real))
        v = cc[valid]
        zs.append((float(v.max()) - float(v.mean())) / (float(v.std()) + 1e-9))
        del cc_fft, cc_real
    return float(np.median(zs))


def _coarse_seed(F, bandpass, *, upsample_factor, max_shift_px=64,
                 n_iter=8, damping=0.6, tol_px=0.05, bw=None, xp=np):
    """Reference-to-sum coarse seed (replaces consecutive-frame CC integration).

    Each iteration: register every frame by its current shift (apply -shift via the
    +2pi i phase, matching _solve_level / _aligned_power), form the leave-one-out
    registered SUM as a high-SNR motion-free reference, and CC each RAW frame against
    it for its ABSOLUTE per-frame displacement (never integrated). Gauge-fix (subtract
    the mean shift) to keep the trajectory off the mask edge, then damped-update toward
    the new estimate. Returns frame-0-anchored (sy[0]=sx[0]=0), in this level's binned px.

    Why (revises the old consecutive seed): per-frame motion is sub-resolution at the
    coarse bin (~3 input px ≈ 0.1 bin30-px), so consecutive-frame CC returns noise and
    the cumulative sum random-walks to thousands of px (041921: +9282 input px vs 215 px
    truth). Reference-to-sum has no integration -> bounded at truth-scale. Pre-validated
    on real 041921 frames (Linux, coord motion-validation #150;
    round2_seed_prevalidation.{py,md}): consecutive 6192px (28x) -> reference-to-sum 206px
    (truth |max| 224), converges (max-delta 6.9 -> 0.03 bin30-px). The runaway is cured
    at the seed; the finer pyramid levels recover the fine burst shape from the bounded seed.

    max_shift_px is the ABSOLUTE per-frame displacement bound (this level's px). The caller
    (_pyramid_solve) passes a PHYSICAL bound (sized to the dataset's max motion) converted
    to level-px, so it never clips a high-motion passer while still rejecting far-field locks.
    """
    n, bh, bwh = F.shape
    if bw is None:
        bw = (bwh - 1) * 2
    fy = xp.fft.fftfreq(bh).astype(xp.float32).reshape(-1, 1)
    fx = xp.fft.rfftfreq(bw).astype(xp.float32).reshape(1, -1)
    sy = xp.zeros(n, dtype=xp.float32)
    sx = xp.zeros(n, dtype=xp.float32)
    for it in range(n_iter):
        reg = xp.empty_like(F)                                  # motion-registered frames
        for j in range(n):
            ph = xp.exp(xp.complex64(2j * xp.pi) * (fy * float(sy[j]) + fx * float(sx[j])))
            reg[j] = F[j] * ph
            del ph
        S = reg.sum(axis=0)
        new_y = xp.zeros(n, dtype=xp.float32)
        new_x = xp.zeros(n, dtype=xp.float32)
        for i in range(n):
            ref = S - reg[i]                                    # leave-one-out (motion-free) reference
            cc_fft = xp.conj(ref) * F[i] * bandpass             # CC the RAW frame -> ABSOLUTE displacement
            cc_real = xp.fft.irfft2(cc_fft, s=(bh, bw))
            cc_real = _mask_cc_to_radius(cc_real, max_shift_px, xp)
            dy, dx = _upsampled_dft_peak(cc_real, upsample_factor, xp=xp)
            new_y[i] = dy
            new_x[i] = dx
            del ref, cc_fft, cc_real
        new_y = new_y - new_y.mean()                            # gauge-fix (keep off the mask edge)
        new_x = new_x - new_x.mean()
        delta = float(max(float(xp.abs(new_y - sy).max()),
                          float(xp.abs(new_x - sx).max())))
        sy = sy + damping * (new_y - sy)                        # damped update for stability
        sx = sx + damping * (new_x - sx)
        del reg, S, new_y, new_x
        if _GPU and xp is _cp:
            _cp.get_default_memory_pool().free_all_blocks()
        if delta < tol_px and it > 0:
            break
    sy = sy - sy[0]                                             # frame-0 anchor (contract; gauge-invariant downstream)
    sx = sx - sx[0]
    return sy, sx


def _enforce_min_spacing(knots, floor):
    """Spread sorted interior knots to keep >= floor spacing and clear of the 0/1
    endpoints; fall back to uniform if they cannot fit."""
    k = np.array(knots, dtype=float)
    m = len(k)
    if m == 0:
        return k
    for i in range(m):
        lo = floor if i == 0 else k[i - 1] + floor
        k[i] = max(k[i], lo)
    if k[-1] > 1.0 - floor:                              # overflow -> uniform fallback
        return np.linspace(0.0, 1.0, m + 2)[1:-1]
    return k


def _arclength_blended_knots(sy, sx, n_interior, *, w=0.8, floor_frames=1.5):
    """Interior knot positions in (0,1) over the frame axis: a blend of frame-uniform
    and arc-length-uniform placement, pos = (1-w)*uniform + w*arclength. Arc length =
    cumulative |per-frame motion| of the (coarse-seed) trajectory -> concentrates knots
    where the motion happens (round-2 diagnosis). floor_frames keeps a minimum knot
    spacing so a burst stays resolvable and the basis non-singular. Uniform when there
    is no motion (degenerate-safe)."""
    n = len(sy)
    targets = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]   # interior fractions (uniform)
    if n_interior <= 0:
        return np.empty(0)
    if n < 3:
        return targets
    sy = np.asarray(sy, dtype=float); sx = np.asarray(sx, dtype=float)
    step = np.sqrt(np.diff(sy) ** 2 + np.diff(sx) ** 2)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    total = float(cum[-1])
    t_frame = np.linspace(0.0, 1.0, n)
    if total > 1e-9:
        arc = np.interp(targets, cum / total, t_frame)      # invert arc-length -> frame param
    else:
        arc = targets
    knots = (1.0 - float(w)) * targets + float(w) * arc
    floor = float(floor_frames) / (n - 1)
    return _enforce_min_spacing(np.sort(knots), floor)


def _bspline_basis(n_frames, K_Z, degree=3, interior=None):
    """Clamped cubic B-spline basis over frame index, K control points. interior:
    optional array of interior knot positions in (0,1) (length K-degree-1) for adaptive
    (arc-length) placement; None = uniform. Returns Phi (n_frames, K);
    per-frame shift trajectory = Phi @ coeffs."""
    K = max(int(K_Z), degree + 1)
    n_interior = K - degree - 1
    if interior is None:
        interior = (np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
                    if n_interior > 0 else np.empty(0))
    else:
        interior = np.asarray(interior, dtype=float)
        if len(interior) != n_interior:
            raise ValueError(f"interior knots {len(interior)} != n_interior {n_interior}")
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    t = np.linspace(0.0, 1.0, n_frames)
    Phi = np.empty((n_frames, K), dtype=np.float64)
    for j in range(K):
        c = np.zeros(K); c[j] = 1.0
        Phi[:, j] = _BSpline(knots, c, degree)(t)
    return Phi


def _aligned_power(sy, sx, F, weight, fy, fx, xp):
    """Signal-band power of the aligned sum: Σ weight·|Σ_i F_i e^{+2πi(fy·sy_i+fx·sx_i)}|².
    Maximizing this ≡ maximizing the sum of pairwise frame CCs (R&B-2015 data term).
    F[i] encodes a frame whose real-space content is shifted by +sy_i (Fourier shift
    theorem: a real-space shift of +s gives FFT phase exp(-2πi*f*s)), so the phase
    correction to align is exp(+2πi*(fy*sy_i+fx*sx_i)). This matches _solve_level and
    _apply_shifts_fullres (both use +2πi). sy/sx are the per-frame motions.

    Batched over the frame axis (broadcast the complex exponential, single sum), with
    the complex aligned-sum AND the band-power reduction in FLOAT64: a batched float32
    sum can reorder vs the per-frame loop and perturb the L-BFGS objective, so the
    reductions are float64 for reproducibility/equivalence (Linux #194).

    Value-only reference path: used by _smooth_objective + the FD-equivalence test, NOT the
    live L-BFGS solve (that calls _smooth_objective_and_grad, which fuses value+gradient and
    is the memory-optimized hot path). Keep float64 here as the bit-exact oracle."""
    sy = xp.asarray(sy, dtype=xp.float64).reshape(-1, 1, 1)
    sx = xp.asarray(sx, dtype=xp.float64).reshape(-1, 1, 1)
    fy = fy.astype(xp.float64)[None, :, :]
    fx = fx.astype(xp.float64)[None, :, :]
    ph = xp.exp(2j * xp.pi * (fy * sy + fx * sx))            # (n, bh, bwh) complex128
    S = xp.sum(F.astype(xp.complex128) * ph, axis=0)         # float64 reduction over frames
    return float(xp.sum(weight.astype(xp.float64) * (S.real * S.real + S.imag * S.imag)))


def _smooth_objective(c_flat, Phi, F, weight, fy, fx, lam, xp):
    """-aligned_power + λ·(2nd-difference curvature of the spline coeffs), for minimization.
    Phi: (n,K) basis; c_flat: [cy(K), cx(K)]. Per-frame shifts = Phi @ c."""
    K = Phi.shape[1]
    cy = np.asarray(c_flat[:K], dtype=np.float64)
    cx = np.asarray(c_flat[K:], dtype=np.float64)
    sy = Phi @ cy
    sx = Phi @ cx
    power = _aligned_power(sy, sx, F, weight, fy, fx, xp)
    d2y = cy[:-2] - 2.0 * cy[1:-1] + cy[2:]
    d2x = cx[:-2] - 2.0 * cx[1:-1] + cx[2:]
    reg = float(lam) * float(np.sum(d2y * d2y) + np.sum(d2x * d2x))
    return -power + reg


def _second_diff_matrix(K):
    """(K-2, K) second-difference operator; row m has 1,-2,1 at m,m+1,m+2.
    D2 @ c == c[:-2] - 2*c[1:-1] + c[2:] (matches _smooth_objective's curvature)."""
    D2 = np.zeros((max(K - 2, 0), K), dtype=np.float64)
    for m in range(K - 2):
        D2[m, m] = 1.0
        D2[m, m + 1] = -2.0
        D2[m, m + 2] = 1.0
    return D2


def _smooth_objective_and_grad(c_flat, Phi, F, weight, fy, fx, lam, xp, D2):
    """(f, grad) for L-BFGS: f = -aligned_power + lam*curvature; grad is the EXACT
    analytic gradient w.r.t. the spline coeffs c=[cy(K), cx(K)].

    F is the per-level complex64 rfft spectrum; it is multiplied in-place into the
    phase buffer (complex128), so no separate complex128 hoist array is held. f mirrors
    _smooth_objective exactly (verified by the FD test). The power and its gradient
    share the per-frame phase-shifted frames (`shifted`) and the aligned sum S, so
    the gradient costs ~one extra batched reduction instead of ~2K finite-difference
    objective evals. All reductions are float64 -> reproducible (Linux #194)."""
    K = Phi.shape[1]
    cy = np.asarray(c_flat[:K], dtype=np.float64)
    cx = np.asarray(c_flat[K:], dtype=np.float64)
    sy = Phi @ cy                                          # (n,) float64 (host)
    sx = Phi @ cx
    fy_d = fy.astype(xp.float64)                           # (bh,1)
    fx_d = fx.astype(xp.float64)                           # (1,bwh)
    w_d = weight.astype(xp.float64)                        # (bh,bwh)
    sy_c = xp.asarray(sy, dtype=xp.float64).reshape(-1, 1, 1)   # (n,1,1)
    sx_c = xp.asarray(sx, dtype=xp.float64).reshape(-1, 1, 1)
    # PEAK VRAM is here: this materializes the (n,bh,bwh) float64 phase argument and then the
    # c128 result -- it, NOT the gemv below, is the OOM ceiling on the biggest movies; coarser
    # Fast-mode binning is the mitigation (Linux #209).
    shifted = xp.exp(2j * xp.pi * (fy_d[None, :, :] * sy_c + fx_d[None, :, :] * sx_c))
    shifted *= F        # in place (F is the complex64 spectrum; widening mul, no separate ph array)
    S = xp.sum(shifted, axis=0)                            # (bh,bwh) c128
    power = float(xp.sum(w_d * (S.real * S.real + S.imag * S.imag)))
    # d(power)/d(sy_i) = 2 Re[ sum_k (2*pi*i*fy_k) * w_k * conj(S_k) * shifted_{i,k} ]
    G = w_d * xp.conj(S)                                   # (bh,bwh) c128
    coef = xp.asarray(2j * xp.pi)
    Ay = (coef * fy_d) * G                                 # (bh,bwh) c128  (fy_d (bh,1) broadcasts)
    Ax = (coef * fx_d) * G                                 # (bh,bwh)
    s2 = shifted.reshape(shifted.shape[0], -1)   # (n, bh*bwh) VIEW; shifted is C-contiguous
    gy = 2.0 * xp.real(s2 @ Ay.reshape(-1))      # (n,) via gemv — no (n,bh,bwh) temporary
    gx = 2.0 * xp.real(s2 @ Ax.reshape(-1))      # (n,)
    gy_h = _cp.asnumpy(gy) if (_GPU and xp is _cp) else np.asarray(gy)  # -> host
    gx_h = _cp.asnumpy(gx) if (_GPU and xp is _cp) else np.asarray(gx)
    # objective is -power, so chain & negate: d(-power)/dc = -Phi^T g
    grad_cy = -(Phi.T @ gy_h)                              # (K,)
    grad_cx = -(Phi.T @ gx_h)
    # curvature: reg = lam*(||D2 cy||^2 + ||D2 cx||^2) -> grad += 2*lam*D2^T D2 c
    d2y = D2 @ cy
    d2x = D2 @ cx
    reg = float(lam) * float(d2y @ d2y + d2x @ d2x)
    grad_cy = grad_cy + 2.0 * float(lam) * (D2.T @ d2y)
    grad_cx = grad_cx + 2.0 * float(lam) * (D2.T @ d2x)
    f = -power + reg
    grad = np.concatenate([grad_cy, grad_cx]).astype(np.float64)
    return f, grad


def _solve_level_smooth(F, weight, sy_init, sx_init, *, K_Z=None, Phi=None,
                        lam=2e-3, trust_px=None, bw=None, xp=np, log=lambda m: None):
    """Joint smooth solve at one bin level (R&B-2015 / cryoSPARC rigid): cubic-B-spline
    trajectory, maximize B-factor-weighted aligned-sum power under a 2nd-deriv penalty,
    via L-BFGS-B. Returns per-frame shifts in this level's binned px.

    Phi: optional precomputed (n, K) basis (arc-length-placed; Task 3). If None, a
    uniform clamped-cubic basis is built from K_Z (legacy callers).
    trust_px: if set, each spline coeff is box-bounded to +/-trust_px around its
    seed-fit init -> because the clamped cubic basis is a non-negative partition of
    unity, every per-frame shift is bounded to +/-trust_px of the (rescaled) init
    (RELION motioncorr +/-search_range analog). None = unbounded (legacy)."""
    n, bh, bwh = F.shape
    if bw is None:
        bw = (bwh - 1) * 2
    fy = xp.fft.fftfreq(bh).astype(xp.float32).reshape(-1, 1)
    fx = xp.fft.rfftfreq(bw).astype(xp.float32).reshape(1, -1)
    if Phi is None:
        Phi = _bspline_basis(n, K_Z if K_Z is not None else 4)
    cy0, *_ = np.linalg.lstsq(Phi, np.asarray(sy_init, dtype=np.float64), rcond=None)
    cx0, *_ = np.linalg.lstsq(Phi, np.asarray(sx_init, dtype=np.float64), rcond=None)
    c0 = np.concatenate([cy0, cx0])
    bounds = None
    if trust_px is not None and float(trust_px) > 0:
        r = float(trust_px)
        bounds = [(float(ci) - r, float(ci) + r) for ci in c0]
    K = Phi.shape[1]
    D2 = _second_diff_matrix(K)
    c_opt, fval, info = _fmin_lbfgs(
        _smooth_objective_and_grad, c0,
        args=(Phi, F, weight, fy, fx, float(lam), xp, D2),
        bounds=bounds, maxiter=60, pgtol=1e-5)
    sy = (Phi @ c_opt[:K]).astype(np.float32)
    sx = (Phi @ c_opt[K:]).astype(np.float32)
    log(f"    smooth solve: K={K} lam={lam} trust={trust_px} "
        f"fval={fval:.4g} calls={info.get('funcalls','?')}")
    return sy, sx, float(fval)

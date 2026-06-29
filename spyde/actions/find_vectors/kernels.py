"""
kernels.py — numba @cuda.jit kernels + CuPy/cuFFT NXCORR for find_vectors.

The device kernels (separable nav-blur, NXCORR reflect/tiled, local-max,
subpixel parabola/CoM, trim-copy, per-frame reduce, beam-stop fill, f32 cast)
and the CuPy FFT cross-correlation that replaces the brute-force numba NXCORR
when CuPy is available.  Compiled once at import; guarded so a missing
numba/CUDA sets _GPU_KERNELS_AVAILABLE = False and callers fall back to CPU.
"""

from __future__ import annotations

import functools
import logging

import numpy as np

from spyde.actions.find_vectors.detectors import _get_disk_fft
from spyde.actions.find_vectors.gpu_runtime import (
    _gpu_cache_lock,
    _gpu_disk_fft_conj_cache,
    _stream_ptr,
)

log = logging.getLogger(__name__)


# ── CuPy / cuFFT NXCORR ───────────────────────────────────────────────────────

def _gpu_disk_fft_conj(kernel_r: int, pH: int, pW: int):
    """Cached CuPy upload of conj(rfft2(disk)) for (radius, pH, pW)."""
    import cupy as cp
    key = (int(kernel_r), int(pH), int(pW))
    with _gpu_cache_lock:
        arr = _gpu_disk_fft_conj_cache.get(key)
        if arr is None:
            arr = cp.asarray(
                np.ascontiguousarray(np.conj(_get_disk_fft(kernel_r, pH, pW)))
            )
            _gpu_disk_fft_conj_cache[key] = arr
        return arr


def _nxcorr_fft_cupy(frames_d, kernel_r: int, disk_stats, numba_stream):
    """
    Batched FFT-based NXCORR via CuPy/cuFFT.

    Mirrors the CPU Lewis-formula path (_find_vectors_single_frame): rFFT
    cross-correlation against the cached disk FFT plus integral-image window
    statistics — O(N·H·W·log HW) instead of the brute-force kernels'
    O(N·H·W·(disk² + win²)), which dominates at large patterns and radii.

    frames_d : numba device array (N, H, W) float32, blurred + beamstop-filled
    Returns a CuPy (N, H, W) float32 raw_corr in [-1, 1].  All work is bound
    to `numba_stream` via ExternalStream, so the numba local-max / subpixel
    kernels that consume the result on the same stream stay correctly ordered
    (CuPy arrays pass to numba kernels zero-copy via the CUDA array interface).
    """
    import cupy as cp
    from scipy.fft import next_fast_len as _nfl

    n_disk, t_mean, t_std = disk_stats
    kr = int(kernel_r)
    krw = kr + 1  # kernel_window_pad = 1, same as the CPU and numba paths

    with cp.cuda.ExternalStream(_stream_ptr(numba_stream)):
        frames = cp.asarray(frames_d)  # zero-copy view of the numba buffer
        N, H, W = frames.shape
        pH = _nfl(H + 2 * kr)
        pW = _nfl(W + 2 * kr)
        disk_fft_conj = _gpu_disk_fft_conj(kr, pH, pW)
        kwin = 2 * krw + 1
        n_win = np.float32(kwin * kwin)

        raw_out = cp.empty((N, H, W), dtype=cp.float32)

        # Sub-batch so peak temporary memory stays ~constant regardless of N
        # (full-chunk batches put ~1 GB of CuPy temporaries in flight per
        # task thread — several concurrent streams then thrash the pool).
        nb = max(8, int(96e6 / (pH * pW * 4)))

        for s0 in range(0, N, nb):
            s1 = min(N, s0 + nb)
            fb = frames[s0:s1]

            # ── Cross-correlation numerator via batched rFFT ─────────────────
            padded = cp.pad(fb, ((0, 0), (kr, kr), (kr, kr)), mode="reflect")
            xcorr = cp.fft.irfft2(
                cp.fft.rfft2(padded, s=(pH, pW)) * disk_fft_conj, s=(pH, pW)
            )[:, :H, :W]
            padded = None

            # ── Window statistics via integral images (kr_win window) ────────
            # float64 accumulators: float32 integral images of 512^2 frames
            # lose ~7 digits to cancellation and put ~1% noise on the scores.
            sp_ = cp.pad(fb, ((0, 0), (krw, krw), (krw, krw)), mode="reflect")
            nB = sp_.shape[0]
            ph, pw = sp_.shape[1], sp_.shape[2]

            cum = cp.zeros((nB, ph + 1, pw + 1), dtype=cp.float64)
            cum[:, 1:, 1:] = sp_.cumsum(axis=1, dtype=cp.float64) \
                                .cumsum(axis=2, dtype=cp.float64)
            ws1 = (cum[:, kwin:H + kwin, kwin:W + kwin]
                   - cum[:, :H, kwin:W + kwin]
                   - cum[:, kwin:H + kwin, :W]
                   + cum[:, :H, :W])
            cum[:, 1:, 1:] = (sp_ * sp_).cumsum(axis=1, dtype=cp.float64) \
                                        .cumsum(axis=2, dtype=cp.float64)
            ws2 = (cum[:, kwin:H + kwin, kwin:W + kwin]
                   - cum[:, :H, kwin:W + kwin]
                   - cum[:, kwin:H + kwin, :W]
                   + cum[:, :H, :W])
            sp_ = None
            cum = None

            win_mean = (ws1 / n_win).astype(cp.float32)
            win_var = cp.maximum(
                (ws2 / n_win).astype(cp.float32) - win_mean * win_mean, 0.0
            )
            win_std = cp.sqrt(win_var)
            ws1 = ws2 = win_var = None

            # ── Normalise with the per-frame denominator floor ────────────────
            gstd = fb.std(axis=(1, 2)).astype(cp.float32)
            gstd = cp.where(gstd == 0, np.float32(1.0), gstd)
            denom_floor = (np.float32(0.01 * t_std)) * gstd[:, None, None]
            denom = cp.maximum(win_std * np.float32(t_std), denom_floor)
            raw = (xcorr / np.float32(n_disk)
                   - win_mean * np.float32(t_mean)) / denom
            raw_out[s0:s1] = cp.clip(raw, -1.0, 1.0)

        return raw_out


# ── GPU kernels (numba.cuda) ──────────────────────────────────────────────────
# These are defined at module level so numba can JIT-compile them once and
# reuse across calls.  They are guarded: if numba is not installed or CUDA is
# not available the try/except below sets _GPU_KERNELS_AVAILABLE = False so the
# caller falls back to the CPU path transparently.

try:
    from numba import cuda as _numba_cuda
    import math as _math

    from numba import float32 as _nb_f32

    # Disk-cache compiled kernels when supported (numba >= 0.57): each dask
    # worker process otherwise pays a multi-second JIT compile per kernel on
    # its first chunk.  Probe in isolation so an unsupported kwarg can't
    # disable the whole GPU path.
    try:
        @_numba_cuda.jit(cache=True)
        def _cuda_cache_probe(x):  # pragma: no cover - never launched
            pass
        _CUDA_JIT = functools.partial(_numba_cuda.jit, cache=True)
    except Exception:
        _CUDA_JIT = _numba_cuda.jit

    # ── Separable 1D Gaussian blur along one nav axis ─────────────────────────
    # Two passes (nav_y then nav_x) replace scipy.ndimage.gaussian_filter.
    # Input/output: float32 (N_nav_y, N_nav_x, KY, KX).
    # Each thread handles one (iy, ix, ky, kx) element.
    # kern: 1D Gaussian weights, length = 2*radius+1, pre-normalised.
    # radius: half-width of the kernel in nav pixels.
    # axis: 0 = blur along nav_y, 1 = blur along nav_x.

    @_CUDA_JIT()
    def _gaussian_blur_1d_kernel(src, dst, kern, radius, axis):
        # Grid layout: x/y span the signal pixels (kx, ky), z is the flattened
        # nav index.  gridDim.z is capped at 65535, so the *small* nav grid
        # must live on z — putting KY*KX there fails to launch for >=256x256
        # patterns (CUDA_ERROR_INVALID_VALUE).  This layout also makes warp
        # reads coalesced (consecutive threads -> consecutive kx).
        kx_i = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        ky_i = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        nav  = _numba_cuda.blockIdx.z  # flattened nav index iy*NX + ix

        NY = src.shape[0]
        NX = src.shape[1]
        KY = src.shape[2]
        KX = src.shape[3]

        if kx_i >= KX or ky_i >= KY:
            return
        iy = nav // NX
        ix = nav - iy * NX
        if iy >= NY:
            return

        acc = _nb_f32(0.0)
        klen = 2 * radius + 1
        if axis == 0:
            for k in range(klen):
                src_y = iy + k - radius
                if src_y < 0:
                    src_y = -src_y
                elif src_y >= NY:
                    src_y = 2 * NY - src_y - 2
                acc += kern[k] * src[src_y, ix, ky_i, kx_i]
        else:
            for k in range(klen):
                src_x = ix + k - radius
                if src_x < 0:
                    src_x = -src_x
                elif src_x >= NX:
                    src_x = 2 * NX - src_x - 2
                acc += kern[k] * src[iy, src_x, ky_i, kx_i]

        dst[iy, ix, ky_i, kx_i] = acc

    @_CUDA_JIT()
    def _local_max_kernel(raw_corr, peak_mask, threshold, min_d, H, W):
        """
        Mark pixels that are local maxima above threshold within ±min_d.

        raw_corr  : float32 (N, H, W)
        peak_mask : uint8   (N, H, W)  — output (1=peak, 0=not)
        """
        out_x = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        out_y = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        n     = _numba_cuda.blockIdx.z

        if out_x >= W or out_y >= H:
            return

        center = raw_corr[n, out_y, out_x]
        if center < threshold:
            peak_mask[n, out_y, out_x] = 0
            return

        # Check all neighbours within ±min_d; suppress if any are strictly greater
        y0 = out_y - min_d
        if y0 < 0:
            y0 = 0
        y1 = out_y + min_d
        if y1 >= H:
            y1 = H - 1
        x0 = out_x - min_d
        if x0 < 0:
            x0 = 0
        x1 = out_x + min_d
        if x1 >= W:
            x1 = W - 1

        is_max = 1
        for ny in range(y0, y1 + 1):
            for nx in range(x0, x1 + 1):
                if raw_corr[n, ny, nx] > center:
                    is_max = 0
                    break
            if is_max == 0:
                break

        peak_mask[n, out_y, out_x] = is_max

    @_CUDA_JIT()
    def _subpixel_com_kernel(raw_corr, peak_mask, peaks_out, n_peaks, half_win, H, W):
        """
        Centre-of-mass subpixel refinement for detected peaks.

        raw_corr  : float32 (N, H, W)
        peak_mask : uint8   (N, H, W)
        peaks_out : float32 (N, MAX_PEAKS, 3)  — [ky_subpx, kx_subpx, score]
        n_peaks   : int32   (N,)               — atomic counter per frame
        """
        out_x = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        out_y = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        n     = _numba_cuda.blockIdx.z

        if out_x >= W or out_y >= H:
            return
        if peak_mask[n, out_y, out_x] == 0:
            return

        # Claim a slot atomically
        slot = _numba_cuda.atomic.add(n_peaks, n, 1)
        max_peaks = peaks_out.shape[1]
        if slot >= max_peaks:
            return  # overflow guard — silently discard

        # CoM window bounds
        y0 = out_y - half_win
        if y0 < 0:
            y0 = 0
        y1 = out_y + half_win + 1
        if y1 > H:
            y1 = H
        x0 = out_x - half_win
        if x0 < 0:
            x0 = 0
        x1 = out_x + half_win + 1
        if x1 > W:
            x1 = W

        s  = 0.0
        sy = 0.0
        sx = 0.0
        for py in range(y0, y1):
            for px in range(x0, x1):
                v = raw_corr[n, py, px]
                s  += v
                sy += py * v
                sx += px * v

        if s > 0.0:
            fy = sy / s
            fx = sx / s
        else:
            fy = float(out_y)
            fx = float(out_x)

        peaks_out[n, slot, 0] = fy
        peaks_out[n, slot, 1] = fx
        peaks_out[n, slot, 2] = raw_corr[n, out_y, out_x]

    @_CUDA_JIT()
    def _subpixel_parabola_kernel(raw_corr, peak_mask, peaks_out, n_peaks, H, W):
        """3-point parabolic subpixel refinement — the GPU port of the CPU
        ``_subpixel_parabola``. Fits a parabola through the peak and its two
        neighbours on each axis of the NXCORR surface and locates the vertex.
        This is the standard low-bias subpixel estimator and matches the CPU /
        torch paths; centre-of-mass (the old default here) is less accurate.

        raw_corr  : float32 (N, H, W)
        peak_mask : uint8   (N, H, W)
        peaks_out : float32 (N, MAX_PEAKS, 3)  — [ky_subpx, kx_subpx, score]
        n_peaks   : int32   (N,)               — atomic counter per frame
        """
        out_x = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        out_y = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        n     = _numba_cuda.blockIdx.z

        if out_x >= W or out_y >= H:
            return
        if peak_mask[n, out_y, out_x] == 0:
            return

        slot = _numba_cuda.atomic.add(n_peaks, n, 1)
        max_peaks = peaks_out.shape[1]
        if slot >= max_peaks:
            return  # overflow guard — silently discard

        fy = float(out_y)
        fx = float(out_x)

        # Parabolic vertex along Y (rows). Mirror the CPU clamp to [-1, 1].
        if 0 < out_y < H - 1:
            a = raw_corr[n, out_y - 1, out_x]
            b = raw_corr[n, out_y, out_x]
            c = raw_corr[n, out_y + 1, out_x]
            den = a - 2.0 * b + c
            if den != 0.0:
                d = 0.5 * (a - c) / den
                if d > 1.0:
                    d = 1.0
                elif d < -1.0:
                    d = -1.0
                fy = out_y + d

        # Parabolic vertex along X (cols).
        if 0 < out_x < W - 1:
            a = raw_corr[n, out_y, out_x - 1]
            b = raw_corr[n, out_y, out_x]
            c = raw_corr[n, out_y, out_x + 1]
            den = a - 2.0 * b + c
            if den != 0.0:
                d = 0.5 * (a - c) / den
                if d > 1.0:
                    d = 1.0
                elif d < -1.0:
                    d = -1.0
                fx = out_x + d

        peaks_out[n, slot, 0] = fy
        peaks_out[n, slot, 1] = fx
        peaks_out[n, slot, 2] = raw_corr[n, out_y, out_x]

    @_CUDA_JIT()
    def _trim_copy_kernel(src, dst, lo_y, lo_x, NXc, H, W):
        """
        Compact the ghost-trimmed core of a blurred (NY_g, NX_g, KY, KX) block
        into a contiguous (N, KY, KX) frame stack — device-to-device, no host
        round trip.  NXc = number of core nav columns; n = iy*NXc + ix.
        """
        x = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        y = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        n = _numba_cuda.blockIdx.z
        if x >= W or y >= H:
            return
        iy = n // NXc
        ix = n - iy * NXc
        dst[n, y, x] = src[lo_y + iy, lo_x + ix, y, x]

    @_CUDA_JIT()
    def _frame_reduce_kernel(frames, mask, use_mask, sums, sumsqs, HW, W):
        """
        Per-frame sum and sum-of-squares reduction.

        One block of 256 threads per frame, fixed-order strided accumulation
        plus a shared-memory tree reduction — deterministic, unlike float
        atomics, so repeated runs produce bit-identical NXCORR scores.

        frames : float32 (N, H, W)
        mask   : uint8 (H, W) — pixels skipped when use_mask == 1
        sums, sumsqs : float32 (N,) — outputs (overwritten)

        Launch as _frame_reduce_kernel[N, 256](...).
        """
        sm_s = _numba_cuda.shared.array(256, dtype=_nb_f32)
        sm_ss = _numba_cuda.shared.array(256, dtype=_nb_f32)
        tid = _numba_cuda.threadIdx.x
        n = _numba_cuda.blockIdx.x

        s = _nb_f32(0.0)
        ss = _nb_f32(0.0)
        i = tid
        while i < HW:
            y = i // W
            x = i - y * W
            if use_mask == 0 or mask[y, x] == 0:
                v = frames[n, y, x]
                s += v
                ss += v * v
            i += 256
        sm_s[tid] = s
        sm_ss[tid] = ss
        _numba_cuda.syncthreads()

        stride = 128
        while stride > 0:
            if tid < stride:
                sm_s[tid] += sm_s[tid + stride]
                sm_ss[tid] += sm_ss[tid + stride]
            _numba_cuda.syncthreads()
            stride //= 2

        if tid == 0:
            sums[n] = sm_s[0]
            sumsqs[n] = sm_ss[0]

    @_CUDA_JIT()
    def _beamstop_fill_kernel(frames, mask, fills, H, W):
        """Replace masked pixels with the frame's unmasked mean — on device."""
        x = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        y = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        n = _numba_cuda.blockIdx.z
        if x >= W or y >= H:
            return
        if mask[y, x] != 0:
            frames[n, y, x] = fills[n]

    @_CUDA_JIT()
    def _nxcorr_reflect_kernel(
        frames, disk, raw_corr, global_stds,
        n_disk, t_mean, t_std, kr, kr_win, H, W,
    ):
        """
        Window-normalised cross-correlation on UNPADDED frames.

        Folds the host-side reflect-pad into the kernel: out-of-bounds taps
        are reflected in-index (same convention as np.pad mode="reflect" and
        the separable blur kernel), so the blurred frames never leave the
        device between blur and correlation.

        frames      : float32 (N, H, W)
        disk        : float32 (2*kr+1, 2*kr+1)
        raw_corr    : float32 (N, H, W) — output
        global_stds : float32 (N,)
        """
        out_x = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        out_y = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        n     = _numba_cuda.blockIdx.z

        if out_x >= W or out_y >= H:
            return

        # ── Cross-correlation: disk taps with reflected indexing ──────────────
        disk_w = 2 * kr + 1
        # float32 accumulator — a 0.0 literal would type as float64, which
        # Pascal-class GPUs execute at 1/32 the float32 rate
        xcorr = _nb_f32(0.0)
        for dr in range(disk_w):
            py = out_y + dr - kr
            if py < 0:
                py = -py
            elif py >= H:
                py = 2 * H - py - 2
            if py < 0:
                py = 0
            elif py >= H:
                py = H - 1
            for dc in range(disk_w):
                px = out_x + dc - kr
                if px < 0:
                    px = -px
                elif px >= W:
                    px = 2 * W - px - 2
                if px < 0:
                    px = 0
                elif px >= W:
                    px = W - 1
                xcorr += disk[dr, dc] * frames[n, py, px]

        # ── Window statistics over the (2*kr_win+1)^2 neighbourhood ──────────
        win_size = 2 * kr_win + 1
        sum1 = _nb_f32(0.0)
        sum2 = _nb_f32(0.0)
        for dr in range(win_size):
            for dc in range(win_size):
                v = frames[n, out_y + dr, out_x + dc]
                sum1 += v
                sum2 += v * v

        n_win = win_size * win_size
        win_mean = sum1 / n_win
        win_var = sum2 / n_win - win_mean * win_mean
        if win_var < 0.0:
            win_var = 0.0
        win_std = _math.sqrt(win_var)

        # ── Normalise (same floor logic as the padded kernel / CPU path) ─────
        denom_floor = 0.01 * global_stds[n] * t_std
        denom = win_std * t_std
        if denom < denom_floor:
            denom = denom_floor
        num = xcorr / n_disk - win_mean * t_mean

        if denom >= 1e-8:
            score = num / denom
            if score > 1.0:
                score = 1.0
            elif score < -1.0:
                score = -1.0
        else:
            score = 0.0

        raw_corr[n, out_y, out_x] = score

    @_CUDA_JIT()
    def _convert_f32_kernel(src, dst):
        """
        Elementwise cast of a (NY, NX, KY, KX) block to float32 on device.
        Grid layout matches the blur kernel: x/y = signal pixels, z = nav.
        """
        kx_i = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        ky_i = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        nav  = _numba_cuda.blockIdx.z

        NY = dst.shape[0]
        NX = dst.shape[1]
        KY = dst.shape[2]
        KX = dst.shape[3]

        if kx_i >= KX or ky_i >= KY:
            return
        iy = nav // NX
        ix = nav - iy * NX
        if iy >= NY:
            return
        dst[iy, ix, ky_i, kx_i] = src[iy, ix, ky_i, kx_i]

    @_CUDA_JIT()
    def _nxcorr_tiled_kernel(
        frames, disk, raw_corr, global_stds,
        n_disk, t_mean, t_std, kr, kr_win, H, W,
    ):
        """
        Shared-memory tiled NXCORR — same math as _nxcorr_reflect_kernel.

        Each 16x16 block cooperatively loads its (16 + 2*kr_win)^2 input tile
        (edge-reflected) into shared memory once; the ~(2*kr+1)^2 + (2*kr_win+1)^2
        taps per output pixel then hit shared memory instead of global,
        cutting global traffic by ~2 orders of magnitude.

        Only valid for kr_win <= 24 (tile <= 64x64 floats = 16 KB shared);
        the caller falls back to _nxcorr_reflect_kernel for larger kernels.
        Launch with block (16, 16, 1).
        """
        sm = _numba_cuda.shared.array((64, 64), dtype=_nb_f32)
        tx = _numba_cuda.threadIdx.x
        ty = _numba_cuda.threadIdx.y
        bx0 = _numba_cuda.blockIdx.x * 16
        by0 = _numba_cuda.blockIdx.y * 16
        n = _numba_cuda.blockIdx.z

        tile = 16 + 2 * kr_win

        # Cooperative tile load — every thread participates before any exits
        j = ty
        while j < tile:
            gy = by0 + j - kr_win
            if gy < 0:
                gy = -gy
            elif gy >= H:
                gy = 2 * H - gy - 2
            if gy < 0:
                gy = 0
            elif gy >= H:
                gy = H - 1
            i = tx
            while i < tile:
                gx = bx0 + i - kr_win
                if gx < 0:
                    gx = -gx
                elif gx >= W:
                    gx = 2 * W - gx - 2
                if gx < 0:
                    gx = 0
                elif gx >= W:
                    gx = W - 1
                sm[j, i] = frames[n, gy, gx]
                i += 16
            j += 16
        _numba_cuda.syncthreads()

        out_x = bx0 + tx
        out_y = by0 + ty
        if out_x >= W or out_y >= H:
            return

        # ── Cross-correlation: disk taps from shared memory ──────────────────
        disk_w = 2 * kr + 1
        off = kr_win - kr
        # float32 accumulator — a 0.0 literal would type as float64, which
        # Pascal-class GPUs execute at 1/32 the float32 rate
        xcorr = _nb_f32(0.0)
        for dr in range(disk_w):
            for dc in range(disk_w):
                xcorr += disk[dr, dc] * sm[ty + off + dr, tx + off + dc]

        # ── Window statistics from shared memory ─────────────────────────────
        win_size = 2 * kr_win + 1
        sum1 = _nb_f32(0.0)
        sum2 = _nb_f32(0.0)
        for dr in range(win_size):
            for dc in range(win_size):
                v = sm[ty + dr, tx + dc]
                sum1 += v
                sum2 += v * v

        n_win = win_size * win_size
        win_mean = sum1 / n_win
        win_var = sum2 / n_win - win_mean * win_mean
        if win_var < 0.0:
            win_var = 0.0
        win_std = _math.sqrt(win_var)

        denom_floor = 0.01 * global_stds[n] * t_std
        denom = win_std * t_std
        if denom < denom_floor:
            denom = denom_floor
        num = xcorr / n_disk - win_mean * t_mean

        if denom >= 1e-8:
            score = num / denom
            if score > 1.0:
                score = 1.0
            elif score < -1.0:
                score = -1.0
        else:
            score = 0.0

        raw_corr[n, out_y, out_x] = score

    _GPU_KERNELS_AVAILABLE = True

except Exception:
    _GPU_KERNELS_AVAILABLE = False

"""
find_vectors.py — Find Diffraction Vectors action for SpyDE.

Adds a caret popout to 4D/5D-STEM signal plots that:
  1. Lets the user tune real-space Gaussian σ (≤2 px), disk kernel radius (linked
     to a draggable CircleROI on the pattern), correlation threshold,
     min-distance separation, and subpixel CoM refinement.
  2. Overlays peak markers (+) and circles directly on the signal plot, updating
     live every time the navigator moves.
  3. Optionally swaps the signal image for the correlation map via a checkbox.
  4. On "Compute" runs the full batch pipeline and adds a DiffractionVectors
     node to the signal tree.

Nav-space Gaussian blur uses two paths:
  - Live preview: NavBlurCache — async per-chunk blur that piggybacks on
    CachedDaskArray's already-resident chunk data (O(1) pattern access when warm,
    ~1ms single-frame fallback when cold).
  - Batch compute: dask.array.map_overlap with ghost zones (depth=ceil(3σ)) so
    chunk boundaries are handled correctly.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.fft import rfft2, irfft2, next_fast_len

from spyde.drawing.toolbars.toolbar import RoundedToolBar

# Cache of pre-computed disk FFTs keyed by (radius, padded_H, padded_W)
_DISK_FFT_CACHE: dict = {}

# ── Module-level guard (one caret per toolbar) ─────────────────────────────────
_FV_BUILT_TOOLBARS: set = set()

# Maximum peaks per frame in GPU subpixel output buffer
MAX_PEAKS: int = 512

# Cache of device-side disk kernel arrays keyed by kernel_r
_gpu_disk_cache: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# NavBlurCache — async per-chunk Gaussian blur
# ─────────────────────────────────────────────────────────────────────────────

class NavBlurCache:
    """
    Async per-chunk Gaussian blur cache for live diffraction-vector preview.

    When a new dask chunk is loaded by CachedDaskArray, call update_chunk().
    It reflect-pads the chunk by depth=ceil(3σ) in both nav dims and blurs it
    in a daemon thread.  Subsequent calls to get_blurred() return the pre-blurred
    pattern at O(1) cost.  While the blur is computing, a single-frame fallback
    is used instead (~1ms for 256×256).
    """

    def __init__(self, sigma: float):
        self.sigma = sigma
        self._depth = int(np.ceil(3 * sigma))
        self._blurred: Optional[np.ndarray] = None  # (cy, cx, ky, kx)
        self._raw_chunk: Optional[np.ndarray] = None
        self._chunk_id: Optional[tuple] = None
        self._blur_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def update_chunk(self, chunk_array: np.ndarray, chunk_id: tuple):
        """Call when CachedDaskArray loads a new chunk. Starts async blur."""
        with self._lock:
            if chunk_id == self._chunk_id:
                return
            self._chunk_id = chunk_id
            self._raw_chunk = chunk_array
            self._blurred = None

        t = threading.Thread(
            target=self._do_blur, args=(chunk_array, chunk_id), daemon=True
        )
        self._blur_thread = t
        t.start()

    def get_blurred(
        self, iy_local: int, ix_local: int, raw_pattern: np.ndarray
    ) -> np.ndarray:
        """
        Return nav-blurred (ky, kx) pattern at local chunk position (iy_local, ix_local).

        Uses the cached blurred chunk (O(1)) when available; falls back to a
        single-frame scipy Gaussian when the async blur hasn't finished.
        """
        with self._lock:
            blurred = self._blurred
        if blurred is not None:
            return blurred[iy_local, ix_local]
        return gaussian_filter(raw_pattern, sigma=(self.sigma, self.sigma))

    def invalidate(self, sigma: float):
        """Clear cached state and update σ (call when the user changes σ)."""
        with self._lock:
            self.sigma = sigma
            self._depth = int(np.ceil(3 * sigma))
            self._blurred = None
            self._chunk_id = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _do_blur(self, chunk_array: np.ndarray, chunk_id: tuple):
        d = self._depth
        if d == 0:
            with self._lock:
                if self._chunk_id == chunk_id:
                    self._blurred = chunk_array
            return
        padded = np.pad(
            chunk_array, ((d, d), (d, d), (0, 0), (0, 0)), mode="reflect"
        )
        blurred_padded = gaussian_filter(
            padded, sigma=(self.sigma, self.sigma, 0, 0)
        )
        trimmed = blurred_padded[d:-d, d:-d]
        with self._lock:
            if self._chunk_id == chunk_id:
                self._blurred = trimmed


# ─────────────────────────────────────────────────────────────────────────────
# Core single-frame pipeline
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def _make_disk(radius: int) -> np.ndarray:
    """Build a normalized flat-disk kernel; cached by integer radius."""
    r = int(radius)
    disk = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
    yy, xx = np.ogrid[-r: r + 1, -r: r + 1]
    disk[yy ** 2 + xx ** 2 <= r ** 2] = 1.0
    disk /= disk.sum()
    return disk


def _get_disk_fft(radius: int, H: int, W: int) -> np.ndarray:
    """
    Return the pre-computed rfft2 of the disk kernel embedded in a (H, W) array.

    Cached globally — computed once per (radius, H, W) combination.
    """
    key = (radius, H, W)
    if key not in _DISK_FFT_CACHE:
        disk = _make_disk(radius)
        d_full = np.zeros((H, W), dtype=np.float32)
        d_full[:disk.shape[0], :disk.shape[1]] = disk
        _DISK_FFT_CACHE[key] = rfft2(d_full)
    return _DISK_FFT_CACHE[key]


def _subpixel_com(
    corr: np.ndarray, peaks_px: np.ndarray, half_win: int = 2
) -> np.ndarray:
    """Center-of-mass subpixel refinement within ±half_win of each integer peak."""
    H, W = corr.shape
    out = np.empty((len(peaks_px), 3), dtype=np.float32)
    for i, (py, px) in enumerate(peaks_px):
        y0 = max(0, int(py) - half_win)
        y1 = min(H, int(py) + half_win + 1)
        x0 = max(0, int(px) - half_win)
        x1 = min(W, int(px) + half_win + 1)
        patch = corr[y0:y1, x0:x1]
        if patch.size == 0:
            out[i] = [py, px, float(corr[int(py), int(px)])]
            continue
        s = float(patch.sum())
        if s > 0:
            wy = patch.sum(axis=1)
            wx = patch.sum(axis=0)
            dy = float(np.dot(np.arange(len(wy), dtype=np.float32), wy)) / s
            dx = float(np.dot(np.arange(len(wx), dtype=np.float32), wx)) / s
        else:
            dy, dx = float(py) - y0, float(px) - x0
        out[i, 0] = y0 + dy
        out[i, 1] = x0 + dx
        out[i, 2] = float(corr[int(py), int(px)])
    return out


def _find_vectors_single_frame(
    frame: np.ndarray,
    kernel_radius: int,
    threshold: float,
    min_distance: int,
    *,
    subpixel: bool = True,
    beamstop_mask: Optional[np.ndarray] = None,
    kernel_window_pad: int = 1,
    _disk_fft=None,
    _disk_stats=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Window-normalised cross-correlation (NXCORR) peak-finding on one frame.

    Implements the Lewis (1995) fast NXCORR formula:

        score(y,x) = (xcorr(y,x)/n - mean_win(y,x) * mean_T)
                     / (std_win(y,x) * std_T)

    Output is in [-1, 1] — threshold is meaningful and intensity-independent.
    This is exactly equivalent to skimage.match_template but ~3x faster via:
      - Pre-computed disk FFT (cached globally, reused across all frames)
      - Integral images via cumsum for O(H*W) window stats
      - maximum_filter + greedy NMS instead of peak_local_max

    Parameters
    ----------
    frame : (ky, kx) float32
    kernel_radius : disk kernel radius in pixels
    threshold : NXCORR threshold in (-1, 1); 0.3-0.5 typical
    min_distance : minimum peak separation in pixels
    subpixel : if True, apply CoM subpixel refinement
    beamstop_mask : (ky, kx) bool — masked pixels excluded before correlation
    kernel_window_pad : extra pixels added to the window radius used for
        computing local mean/std (not the correlation template).  A pad of 1
        means the statistics window is (kr+1) rather than kr, sampling a ring
        of background around the disk and making the denominator more robust
        against spurious single-pixel intensity spikes.  Positions where the
        padded window std is still zero are set to score=0 (not inflated).
    _disk_fft : pre-computed rfft2 of the disk at the padded frame size
    _disk_stats : (n, t_mean, t_std) pre-computed disk statistics

    Returns
    -------
    corr_map : (ky, kx) thresholded NXCORR (for display)
    raw_corr : (ky, kx) NXCORR in [-1, 1]
    peaks    : (N, 3) float32 — [ky_subpx, kx_subpx, nxcorr_value]
    """
    kr = int(kernel_radius)
    H, W = frame.shape

    if beamstop_mask is not None and beamstop_mask.any():
        fill = float(frame[~beamstop_mask].mean()) if (~beamstop_mask).any() else 0.0
        frame = frame.copy()
        frame[beamstop_mask] = fill

    # Reflect-pad the frame by kr on each side so peaks near edges are detected
    # correctly.  Then zero-extend to the next FFT-efficient size.
    padded_h = H + 2 * kr
    padded_w = W + 2 * kr
    pH = next_fast_len(padded_h)
    pW = next_fast_len(padded_w)

    padded_full = np.pad(frame, kr, mode="reflect")  # (padded_h, padded_w)

    buf = np.zeros((pH, pW), dtype=np.float32)
    buf[:padded_h, :padded_w] = padded_full

    if _disk_fft is None:
        _disk_fft = _get_disk_fft(kr, pH, pW)

    # --- Step 1: cross-correlation numerator via FFT ---
    # xcorr[y,x] = sum_u T(u) * I(y+u)   (unnormalised, in raw intensity units)
    xcorr = irfft2(rfft2(buf) * _disk_fft.conj())[:H, :W].astype(np.float32)

    # --- Step 2: window statistics via integral images ---
    # The statistics window uses kr_win = kr + kernel_window_pad so it samples
    # a slightly larger region than the correlation template.  This makes the
    # local std estimate more robust: a single bright pixel at the disk edge
    # raises the std of the padded window without affecting the correlation
    # numerator, preventing spurious near-1 scores.
    # t_mean / t_std always come from the actual disk template (kr), not kr_win.
    disk = _make_disk(kr)
    kH, kW = disk.shape
    n = kH * kW

    if _disk_stats is None:
        t_mean = float(disk.mean())
        t_std = float(np.sqrt(np.sum((disk - t_mean) ** 2) / n))
    else:
        n, t_mean, t_std = _disk_stats

    kr_win = kr + int(kernel_window_pad)
    kH_win = 2 * kr_win + 1
    kW_win = kH_win
    n_win = kH_win * kW_win

    # Build integral images on a frame padded by kr_win (not kr) so the larger
    # statistics window fits within bounds.  The extra reflect-pad beyond kr
    # costs one np.pad call but keeps the indexing simple and correct.
    stat_padded = np.pad(frame, kr_win, mode="reflect")
    stat_ph, stat_pw = stat_padded.shape  # H + 2*kr_win, W + 2*kr_win

    cum1 = np.empty((stat_ph + 1, stat_pw + 1), dtype=np.float32)
    cum1[0, :] = 0.0
    cum1[:, 0] = 0.0
    cum1[1:, 1:] = np.cumsum(np.cumsum(stat_padded, axis=0), axis=1)

    cum2 = np.empty((stat_ph + 1, stat_pw + 1), dtype=np.float32)
    cum2[0, :] = 0.0
    cum2[:, 0] = 0.0
    cum2[1:, 1:] = np.cumsum(np.cumsum(stat_padded ** 2, axis=0), axis=1)

    # Window sums: H+kH_win ≤ stat_ph = H+2*kr_win, so always in bounds.
    ws1 = (cum1[kH_win:H + kH_win, kW_win:W + kW_win]
           - cum1[0:H, kW_win:W + kW_win]
           - cum1[kH_win:H + kH_win, 0:W]
           + cum1[0:H, 0:W])
    ws2 = (cum2[kH_win:H + kH_win, kW_win:W + kW_win]
           - cum2[0:H, kW_win:W + kW_win]
           - cum2[kH_win:H + kH_win, 0:W]
           + cum2[0:H, 0:W])

    win_mean = ws1 / n_win
    win_var = ws2 / n_win - win_mean ** 2
    np.maximum(win_var, 0.0, out=win_var)
    win_std = np.sqrt(win_var)

    # --- Step 3: normalise ---
    # Floor win_std at 1% of the frame's global std before multiplying by t_std.
    # This prevents near-zero denominators in sparse/flat regions (where nav-space
    # blur can smear a tiny amount of intensity into an otherwise empty background,
    # causing win_std ≈ 0 and a falsely inflated NXCORR score).  The 1% fraction
    # is small enough that it has no effect in regions with genuine local variation.
    global_std = float(frame.std()) or 1.0
    denom_floor = 0.01 * global_std * t_std
    numerator = xcorr / n - win_mean * t_mean
    denom = np.maximum(win_std * t_std, denom_floor)
    raw_corr = (numerator / denom).astype(np.float32)
    np.clip(raw_corr, -1.0, 1.0, out=raw_corr)

    if beamstop_mask is not None and beamstop_mask.any():
        raw_corr[beamstop_mask] = -1.0

    corr_map = np.where(raw_corr >= threshold, raw_corr, 0.0).astype(np.float32)

    # Fast peak detection: local maximum filter enforces min_distance separation,
    # then threshold.  Using size = 2*min_distance+1 means a pixel is a local max
    # only if no neighbor within min_distance has a higher value.
    if not (raw_corr >= threshold).any():
        return corr_map, raw_corr, np.zeros((0, 3), dtype=np.float32)

    min_d = int(min_distance)
    local_max = maximum_filter(raw_corr, size=2 * min_d + 1)
    peaks_mask = (raw_corr == local_max) & (raw_corr >= threshold)

    if beamstop_mask is not None and beamstop_mask.any():
        peaks_mask &= ~beamstop_mask

    peaks_px = np.argwhere(peaks_mask)
    # Greedy NMS: sort by intensity descending and suppress any peak within
    # min_distance of a higher-intensity peak already accepted.
    if len(peaks_px) > 1:
        intensities = raw_corr[peaks_px[:, 0], peaks_px[:, 1]]
        order = np.argsort(-intensities)
        peaks_px = peaks_px[order]
        kept = np.ones(len(peaks_px), dtype=bool)
        min_d2 = min_d * min_d
        for i in range(len(peaks_px)):
            if not kept[i]:
                continue
            dy = peaks_px[i + 1:, 0] - peaks_px[i, 0]
            dx = peaks_px[i + 1:, 1] - peaks_px[i, 1]
            too_close = (dy * dy + dx * dx) <= min_d2
            kept[i + 1:][too_close] = False
        peaks_px = peaks_px[kept]

    if len(peaks_px) == 0:
        return corr_map, raw_corr, np.zeros((0, 3), dtype=np.float32)

    if subpixel:
        peaks = _subpixel_com(raw_corr, peaks_px)
    else:
        peaks = np.column_stack(
            [peaks_px.astype(np.float32), raw_corr[peaks_px[:, 0], peaks_px[:, 1]]]
        )
    return corr_map, raw_corr, peaks


def _estimate_disk_radius(frame: np.ndarray) -> int:
    """
    Estimate the diffraction disk radius in pixels from a single pattern.

    Strategy: LoG scale-space blob detection on the bright-field–suppressed frame.
    The LoG response |∇²G_σ * I| peaks at σ ≈ r/√2 for a filled disk of radius r.
    We sweep σ over a range, find the scale with maximum integrated response, and
    convert back to radius.  Falls back to 5% of frame width if detection fails.
    """
    from skimage.filters import laplace

    frame_f = frame.astype(np.float64)
    min_dim = min(frame.shape)
    fallback = max(3, int(min_dim * 0.05))

    # Suppress DC / background: subtract a heavily blurred version
    bg = gaussian_filter(frame_f, sigma=min_dim * 0.15)
    fg = np.clip(frame_f - bg, 0, None)
    if fg.max() == 0:
        return fallback

    fg /= fg.max()

    # Sweep blob scales: σ from 1 px to 15% of frame
    sigma_min = 1.0
    sigma_max = max(sigma_min + 1, min_dim * 0.15)
    n_steps = 20
    sigmas = np.geomspace(sigma_min, sigma_max, n_steps)

    best_sigma = sigma_min
    best_response = -np.inf
    for s in sigmas:
        blurred = gaussian_filter(fg, sigma=s)
        # LoG via scale-normalised Laplacian: multiply by σ² to compare across scales
        response = (s ** 2) * np.abs(laplace(blurred)).max()
        if response > best_response:
            best_response = response
            best_sigma = s

    # σ_LoG ≈ r / √2  →  r ≈ σ * √2
    r_est = int(round(best_sigma * 1.414))
    return max(3, min(r_est, int(min_dim * 0.25)))


def _auto_params(frame: np.ndarray) -> dict:
    """Estimate reasonable starting parameters from the current diffraction pattern."""
    r_px = _estimate_disk_radius(frame)
    return dict(
        sigma=1.0,
        kernel_radius=r_px,
        threshold=0.5,
        min_distance=max(1, r_px // 2),
        subpixel=True,
    )


def _nav_chunk_size(
    sigma: float, max_ram_mb: float = 200, sig_shape: tuple = (256, 256),
    vram_mb: float = 0.0,
) -> int:
    """
    Compute the nav chunk size so the ghost-padded chunk fits within the memory budget.

    When vram_mb > 0 (GPU path) the budget is VRAM, allowing much larger chunks and
    fewer Dask task dispatches.  For CPU the budget is max_ram_mb.
    """
    depth = int(np.ceil(3 * sigma))
    sig_pixels = sig_shape[0] * sig_shape[1]
    budget = vram_mb if vram_mb > 0 else max_ram_mb
    max_padded = int(np.sqrt(budget * 1e6 / (sig_pixels * 4)))
    return max(depth + 1, max_padded - 2 * depth)


# ─────────────────────────────────────────────────────────────────────────────
# Batch compute
# ─────────────────────────────────────────────────────────────────────────────

# ── GPU kernels (numba.cuda) ──────────────────────────────────────────────────
# These are defined at module level so numba can JIT-compile them once and
# reuse across calls.  All three are guarded: if numba is not installed or
# CUDA is not available the try/except in _find_vectors_batch_gpu catches the
# ImportError / CudaSupportError and returns None so the caller falls back to
# the CPU path transparently.

try:
    from numba import cuda as _numba_cuda
    import math as _math

    # ── Separable 1D Gaussian blur along one nav axis ─────────────────────────
    # Two passes (nav_y then nav_x) replace scipy.ndimage.gaussian_filter.
    # Input/output: float32 (N_nav_y, N_nav_x, KY, KX).
    # Each thread handles one (iy, ix, ky, kx) element.
    # kern: 1D Gaussian weights, length = 2*radius+1, pre-normalised.
    # radius: half-width of the kernel in nav pixels.
    # axis: 0 = blur along nav_y, 1 = blur along nav_x.

    @_numba_cuda.jit
    def _gaussian_blur_1d_kernel(src, dst, kern, radius, axis):
        ix  = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        iy  = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        kxy = _numba_cuda.blockIdx.z  # flattened signal pixel index

        NY = src.shape[0]
        NX = src.shape[1]
        KY = src.shape[2]
        KX = src.shape[3]

        if iy >= NY or ix >= NX:
            return
        ky_i = kxy // KX
        kx_i = kxy  % KX
        if ky_i >= KY:
            return

        acc = 0.0
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

    @_numba_cuda.jit
    def _nxcorr_kernel(
        frames_padded, disk, raw_corr, global_stds,
        n_disk, t_mean, t_std, kr, kr_win, threshold, H, W,
    ):
        """
        Window-normalised cross-correlation kernel.

        frames_padded : float32 (N, H+2*kr_win, W+2*kr_win)
        disk          : float32 (2*kr+1, 2*kr+1)
        raw_corr      : float32 (N, H, W)  — output
        global_stds   : float32 (N,)
        """
        out_x = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        out_y = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        n     = _numba_cuda.blockIdx.z

        if out_x >= W or out_y >= H:
            return

        PW = W + 2 * kr_win  # padded width (unused in indexing but kept for clarity)

        # ── Cross-correlation: convolve disk over padded frame ────────────────
        disk_h = 2 * kr + 1
        disk_w = 2 * kr + 1
        xcorr = 0.0
        for dr in range(disk_h):
            for dc in range(disk_w):
                py = out_y + kr_win + dr - kr
                px = out_x + kr_win + dc - kr
                val = frames_padded[n, py, px]
                xcorr += disk[dr, dc] * val

        # ── Window statistics: loop over (2*kr_win+1)^2 neighbourhood ────────
        win_size = 2 * kr_win + 1
        sum1 = 0.0
        sum2 = 0.0
        for dr in range(win_size):
            for dc in range(win_size):
                v = frames_padded[n, out_y + dr, out_x + dc]
                sum1 += v
                sum2 += v * v

        n_win = win_size * win_size
        win_mean = sum1 / n_win
        win_var = sum2 / n_win - win_mean * win_mean
        if win_var < 0.0:
            win_var = 0.0
        win_std = _math.sqrt(win_var)

        # ── Normalise ─────────────────────────────────────────────────────────
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

    @_numba_cuda.jit
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

    @_numba_cuda.jit
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

    _GPU_KERNELS_AVAILABLE = True

except Exception:
    _GPU_KERNELS_AVAILABLE = False


def _find_vectors_batch_gpu(
    blurred_block,
    kernel_r,
    threshold,
    min_dist,
    subpixel,
    beamstop_mask,
    disk_d,
    disk_stats,
):
    """
    GPU-accelerated batch vector finding using numba.cuda.

    Parameters
    ----------
    blurred_block : CPU float32 ndarray (..., H, W)
        Nav-blurred diffraction patterns; any number of leading nav dims.
    kernel_r : int
        Disk kernel radius in pixels.
    threshold : float
        NXCORR threshold.
    min_dist : int
        Minimum peak separation in pixels.
    subpixel : bool
        Whether to apply CoM subpixel refinement.
    beamstop_mask : (H, W) bool ndarray | None
        Pixels to exclude.
    disk_d : device array | None
        Pre-uploaded float32 disk kernel for this kernel_r.  If None the
        function uploads it and stores it in _gpu_disk_cache.
    disk_stats : (n_disk, t_mean, t_std)
        Pre-computed disk statistics (same values as in the CPU path).

    Returns
    -------
    object ndarray of shape nav_shape, each element (N_peaks, 3) float32,
    or None if CUDA is unavailable / any error occurs (caller falls back to CPU).
    """
    try:
        from numba import cuda as _cuda
    except ImportError:
        return None
    if not _cuda.is_available():
        return None
    if not _GPU_KERNELS_AVAILABLE:
        return None

    try:
        n_disk, t_mean, t_std = disk_stats
        n_disk  = np.int32(n_disk)
        t_mean  = np.float32(t_mean)
        t_std   = np.float32(t_std)
        kr      = np.int32(kernel_r)
        kr_win  = np.int32(kernel_r + 1)   # kernel_window_pad = 1, same as CPU
        thr     = np.float32(threshold)
        min_d   = np.int32(min_dist)

        nav_shape = blurred_block.shape[:-2]
        H, W = int(blurred_block.shape[-2]), int(blurred_block.shape[-1])
        iH, iW = np.int32(H), np.int32(W)
        N = int(np.prod(nav_shape)) if len(nav_shape) > 0 else 1

        # Flatten to (N, H, W) float32
        frames = blurred_block.reshape(N, H, W).astype(np.float32)

        # Apply beamstop mask fill on CPU before H2D
        if beamstop_mask is not None and beamstop_mask.any():
            fill_vals = []
            for i in range(N):
                unmasked = frames[i][~beamstop_mask]
                fill = float(unmasked.mean()) if unmasked.size > 0 else 0.0
                fill_vals.append(fill)
            for i in range(N):
                frames[i][beamstop_mask] = fill_vals[i]

        # Per-frame global std (on CPU before H2D)
        global_stds = frames.std(axis=(-1, -2)).astype(np.float32)
        # Guard against all-zero frames
        global_stds[global_stds == 0.0] = 1.0

        # Reflect-pad each frame by kr_win on all sides
        frames_padded = np.pad(
            frames,
            ((0, 0), (int(kr_win), int(kr_win)), (int(kr_win), int(kr_win))),
            mode="reflect",
        ).astype(np.float32)

        # Upload to device
        frames_d       = _cuda.to_device(frames_padded)
        global_stds_d  = _cuda.to_device(global_stds)

        # Disk kernel — upload once per kernel_r, then reuse
        if disk_d is None:
            disk_cpu = _make_disk(kernel_r)
            disk_d   = _cuda.to_device(disk_cpu)
            _gpu_disk_cache[kernel_r] = disk_d

        # Allocate output arrays on device
        raw_corr_d  = _cuda.device_array((N, H, W), dtype=np.float32)
        peak_mask_d = _cuda.device_array((N, H, W), dtype=np.uint8)

        # Grid / block configuration
        bx, by = 16, 16
        grid = (
            int(np.ceil(W / bx)),
            int(np.ceil(H / by)),
            N,
        )
        block = (bx, by, 1)

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")  # suppress numba low-occupancy warnings for small chunks

            _nxcorr_kernel[grid, block](
                frames_d, disk_d, raw_corr_d, global_stds_d,
                n_disk, t_mean, t_std, kr, kr_win, thr, iH, iW,
            )

            _local_max_kernel[grid, block](
                raw_corr_d, peak_mask_d, thr, min_d, iH, iW,
            )

            if subpixel:
                peaks_out_d = _cuda.device_array((N, MAX_PEAKS, 3), dtype=np.float32)
                n_peaks_d   = _cuda.to_device(np.zeros(N, dtype=np.int32))
                half_win    = np.int32(2)
                _subpixel_com_kernel[grid, block](
                    raw_corr_d, peak_mask_d, peaks_out_d, n_peaks_d, half_win, iH, iW,
                )

        if subpixel:
            peaks_out = peaks_out_d.copy_to_host()
            n_peaks   = n_peaks_d.copy_to_host()
        else:
            peak_mask = peak_mask_d.copy_to_host()
            raw_corr  = raw_corr_d.copy_to_host()

        # Build per-frame result list, then run greedy NMS (CPU — very fast,
        # typically <30 peaks per frame)
        result_flat = np.empty(N, dtype=object)
        min_d2 = int(min_dist) * int(min_dist)

        for i in range(N):
            if subpixel:
                np_i   = int(n_peaks[i])
                np_i   = min(np_i, MAX_PEAKS)
                if np_i == 0:
                    result_flat[i] = np.zeros((0, 3), dtype=np.float32)
                    continue
                frame_peaks = peaks_out[i, :np_i, :].copy()  # (np_i, 3): [ky, kx, score]
            else:
                pm  = peak_mask[i]   # (H, W) uint8
                rc  = raw_corr[i]    # (H, W) float32
                yx  = np.argwhere(pm > 0)
                if len(yx) == 0:
                    result_flat[i] = np.zeros((0, 3), dtype=np.float32)
                    continue
                scores = rc[yx[:, 0], yx[:, 1]]
                frame_peaks = np.column_stack([yx.astype(np.float32), scores])

            # Apply beamstop mask exclusion (on surviving peaks)
            if beamstop_mask is not None and beamstop_mask.any() and len(frame_peaks) > 0:
                ky_px = frame_peaks[:, 0].astype(int)
                kx_px = frame_peaks[:, 1].astype(int)
                np.clip(ky_px, 0, H - 1, out=ky_px)
                np.clip(kx_px, 0, W - 1, out=kx_px)
                keep = ~beamstop_mask[ky_px, kx_px]
                frame_peaks = frame_peaks[keep]

            if len(frame_peaks) == 0:
                result_flat[i] = np.zeros((0, 3), dtype=np.float32)
                continue

            # Greedy NMS (matches CPU path)
            if len(frame_peaks) > 1:
                order = np.argsort(-frame_peaks[:, 2])
                frame_peaks = frame_peaks[order]
                kept = np.ones(len(frame_peaks), dtype=bool)
                for j in range(len(frame_peaks)):
                    if not kept[j]:
                        continue
                    dy = frame_peaks[j + 1:, 0] - frame_peaks[j, 0]
                    dx = frame_peaks[j + 1:, 1] - frame_peaks[j, 1]
                    too_close = (dy * dy + dx * dx) <= min_d2
                    kept[j + 1:][too_close] = False
                frame_peaks = frame_peaks[kept]

            result_flat[i] = frame_peaks.astype(np.float32)

        result = result_flat.reshape(nav_shape) if len(nav_shape) > 0 else result_flat
        return result

    except Exception:
        return None


def _find_vectors_chunk(
    ghost_block: np.ndarray,
    depth_px: int,
    nav_dim: int,
    sigma: float,
    kernel_r: int,
    threshold: float,
    min_dist: int,
    subpixel: bool,
    beamstop_mask,
    disk_fft,
    disk_stats,
) -> np.ndarray:
    """
    Full pipeline for one ghost-padded nav chunk.

    Passed to dask.array.map_overlap with trim=False.  Receives the ghost-padded
    block from map_overlap, does everything on GPU when available:

        CPU → [H2D] → GPU blur (nav-space Gaussian, real-space separable)
                     → NXCORR kernel (xcorr + window stats + normalise)
                     → local-max kernel (NMS)
                     → subpixel CoM kernel
                     → [D2H sparse peaks]
        → pack into (nav_y, nav_x, MAX_PEAKS, 3) NaN-padded float32

    Falls back to scipy + CPU per-frame loop if CUDA is unavailable.

    The ghost zone covers depth_px = ceil(3σ) nav rows/cols on each edge.
    The blur uses these ghost rows so chunk-boundary blur values are correct;
    we trim them back before NXCORR so no ghost pixels appear in the output.

    For 5D (nav_dim=3) the leading dimension is time; we process each t-slice
    as an independent 4D block and stack into (t, ny, nx, MAX_PEAKS, 3).

    Returns
    -------
    float32 ndarray (nav_y, nav_x, MAX_PEAKS, 3)          [4D]
                 or (t, nav_y, nav_x, MAX_PEAKS, 3)        [5D]
    """
    # ── Try GPU path ──────────────────────────────────────────────────────────
    if _GPU_KERNELS_AVAILABLE:
        try:
            from numba import cuda as _cuda
            if _cuda.is_available():
                return _find_vectors_chunk_gpu(
                    ghost_block, depth_px, nav_dim, sigma,
                    kernel_r, threshold, min_dist, subpixel,
                    beamstop_mask, disk_stats,
                )
        except Exception:
            pass

    # ── CPU fallback ──────────────────────────────────────────────────────────
    from scipy.ndimage import gaussian_filter as _gf

    sigma_tuple = tuple([0.0] * (nav_dim - 2) + [sigma, sigma, 0.0, 0.0])
    blurred = _gf(ghost_block.astype(np.float32), sigma=sigma_tuple)

    # Trim ghost zones
    trim = [slice(None)] * ghost_block.ndim
    for d in range(nav_dim):
        s = blurred.shape[d]
        lo = depth_px if depth_px < s else 0
        hi = s - depth_px if depth_px < s else s
        trim[d] = slice(lo, hi)
    blurred = blurred[tuple(trim)]

    nav_shape = blurred.shape[:nav_dim]
    ny, nx = nav_shape[-2:]

    def _cpu_block(b4d):
        out = np.full((b4d.shape[0], b4d.shape[1], MAX_PEAKS, 3), np.nan, dtype=np.float32)
        flat = b4d.reshape(-1, b4d.shape[2], b4d.shape[3])
        for i, frame in enumerate(flat):
            iy, ix = divmod(i, b4d.shape[1])
            _, _, peaks = _find_vectors_single_frame(
                frame, kernel_r, threshold, min_dist,
                subpixel=subpixel, beamstop_mask=beamstop_mask,
                _disk_fft=disk_fft, _disk_stats=disk_stats,
            )
            n = min(len(peaks), MAX_PEAKS)
            if n > 0:
                out[iy, ix, :n, :] = peaks[:n]
        return out

    if nav_dim == 2:
        return _cpu_block(blurred)
    else:
        n_lead = nav_shape[0]
        out = np.full((n_lead, ny, nx, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        for t in range(n_lead):
            out[t] = _cpu_block(blurred[t])
        return out


def _find_vectors_chunk_gpu(
    ghost_block: np.ndarray,
    depth_px: int,
    nav_dim: int,
    sigma: float,
    kernel_r: int,
    threshold: float,
    min_dist: int,
    subpixel: bool,
    beamstop_mask,
    disk_stats,
) -> np.ndarray:
    """
    GPU implementation of _find_vectors_chunk.

    Single H2D transfer of the ghost-padded block, then entirely on-device:
      1. Separable Gaussian blur in nav-space (two 1D kernel passes)
      2. Trim ghost zones (device-side slice — no copy)
      3. NXCORR + local-max + subpixel kernels
      4. D2H only the sparse padded peak result

    For 5D, processes each t-slice sequentially on GPU (same device context).
    """
    from numba import cuda as _cuda

    block_f32 = ghost_block.astype(np.float32)
    nav_shape_ghost = block_f32.shape[:nav_dim]
    KY, KX = block_f32.shape[-2], block_f32.shape[-1]

    # ── Pre-compute 1D Gaussian kernel (CPU, tiny) ────────────────────────────
    if sigma > 0:
        radius = int(np.ceil(3 * sigma))
        xs = np.arange(-radius, radius + 1, dtype=np.float32)
        kern_cpu = np.exp(-0.5 * (xs / sigma) ** 2).astype(np.float32)
        kern_cpu /= kern_cpu.sum()
    else:
        radius = 0
        kern_cpu = np.ones(1, dtype=np.float32)
    kern_d = _cuda.to_device(kern_cpu)

    # ── Upload disk kernel once per kernel_r ──────────────────────────────────
    if kernel_r not in _gpu_disk_cache:
        _gpu_disk_cache[kernel_r] = _cuda.to_device(_make_disk(kernel_r))
    disk_d = _gpu_disk_cache[kernel_r]

    n_disk, t_mean, t_std = disk_stats
    n_disk = np.int32(n_disk)
    t_mean = np.float32(t_mean)
    t_std  = np.float32(t_std)
    kr     = np.int32(kernel_r)
    kr_win = np.int32(kernel_r + 1)
    thr    = np.float32(threshold)
    min_d  = np.int32(min_dist)

    def _process_4d(block4d_cpu):
        """Full GPU pipeline for one (ny_ghost, nx_ghost, KY, KX) block."""
        NY_g, NX_g = block4d_cpu.shape[0], block4d_cpu.shape[1]

        # H2D — single transfer of the ghost-padded block
        src_d = _cuda.to_device(block4d_cpu)
        tmp_d = _cuda.device_array_like(src_d)

        # ── Gaussian blur: two separable 1D passes ────────────────────────────
        # Grid: x=NX, y=NY, z=KY*KX
        bx, by = 16, 16
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            if sigma > 0:
                grid_blur = (
                    int(np.ceil(NX_g / bx)),
                    int(np.ceil(NY_g / by)),
                    KY * KX,
                )
                block_blur = (bx, by, 1)
                _gaussian_blur_1d_kernel[grid_blur, block_blur](
                    src_d, tmp_d, kern_d, np.int32(radius), np.int32(0)
                )  # blur along nav_y into tmp
                _gaussian_blur_1d_kernel[grid_blur, block_blur](
                    tmp_d, src_d, kern_d, np.int32(radius), np.int32(1)
                )  # blur along nav_x back into src_d
                blurred_d = src_d
            else:
                blurred_d = src_d  # no blur needed

        # ── Trim ghost zones (device-side slice — zero-copy view) ─────────────
        lo = depth_px
        hi_y = NY_g - depth_px
        hi_x = NX_g - depth_px
        if lo >= hi_y or lo >= hi_x:
            lo = 0; hi_y = NY_g; hi_x = NX_g
        valid_d = blurred_d[lo:hi_y, lo:hi_x, :, :]
        NY, NX = valid_d.shape[0], valid_d.shape[1]
        N = NY * NX
        iH, iW = np.int32(KY), np.int32(KX)

        # ── Reshape to (N, KY, KX) for NXCORR kernels ────────────────────────
        flat_d = valid_d.reshape(N, KY, KX)

        # Apply beamstop fill on GPU slice (CPU side, negligible for mask)
        if beamstop_mask is not None and beamstop_mask.any():
            flat_cpu = flat_d.copy_to_host()
            for i in range(N):
                unmasked = flat_cpu[i][~beamstop_mask]
                fill = float(unmasked.mean()) if unmasked.size > 0 else 0.0
                flat_cpu[i][beamstop_mask] = fill
            flat_d = _cuda.to_device(flat_cpu)

        # Per-frame global std for denom_floor
        flat_cpu_for_std = flat_d.copy_to_host()
        global_stds = flat_cpu_for_std.std(axis=(-1, -2)).astype(np.float32)
        global_stds[global_stds == 0.0] = 1.0
        global_stds_d = _cuda.to_device(global_stds)

        # Reflect-pad each frame by kr_win for NXCORR window stats
        pad = int(kr_win)
        frames_padded_cpu = np.pad(
            flat_cpu_for_std,
            ((0, 0), (pad, pad), (pad, pad)),
            mode="reflect",
        ).astype(np.float32)
        frames_d = _cuda.to_device(frames_padded_cpu)

        raw_corr_d  = _cuda.device_array((N, KY, KX), dtype=np.float32)
        peak_mask_d = _cuda.device_array((N, KY, KX), dtype=np.uint8)

        grid_px = (int(np.ceil(KX / 16)), int(np.ceil(KY / 16)), N)
        blk_px  = (16, 16, 1)

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            _nxcorr_kernel[grid_px, blk_px](
                frames_d, disk_d, raw_corr_d, global_stds_d,
                n_disk, t_mean, t_std, kr, kr_win, thr, iH, iW,
            )
            _local_max_kernel[grid_px, blk_px](
                raw_corr_d, peak_mask_d, thr, min_d, iH, iW,
            )
            if subpixel:
                peaks_out_d = _cuda.device_array((N, MAX_PEAKS, 3), dtype=np.float32)
                n_peaks_d   = _cuda.to_device(np.zeros(N, dtype=np.int32))
                _subpixel_com_kernel[grid_px, blk_px](
                    raw_corr_d, peak_mask_d, peaks_out_d, n_peaks_d,
                    np.int32(2), iH, iW,
                )

        # D2H — only sparse results
        if subpixel:
            peaks_out = peaks_out_d.copy_to_host()
            n_peaks   = n_peaks_d.copy_to_host()
        else:
            peak_mask = peak_mask_d.copy_to_host()
            raw_corr  = raw_corr_d.copy_to_host()

        # Pack into (NY, NX, MAX_PEAKS, 3) NaN-padded
        out = np.full((NY, NX, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        min_d2 = int(min_dist) * int(min_dist)
        for i in range(N):
            iy, ix = divmod(i, NX)
            if subpixel:
                np_i = min(int(n_peaks[i]), MAX_PEAKS)
                frame_peaks = peaks_out[i, :np_i].copy() if np_i > 0 else None
            else:
                yx = np.argwhere(peak_mask[i] > 0)
                if len(yx) == 0:
                    continue
                scores = raw_corr[i][yx[:, 0], yx[:, 1]]
                frame_peaks = np.column_stack([yx.astype(np.float32), scores])

            if frame_peaks is None or len(frame_peaks) == 0:
                continue

            # Beamstop exclusion on surviving peaks
            if beamstop_mask is not None and beamstop_mask.any():
                ky_px = np.clip(frame_peaks[:, 0].astype(int), 0, KY - 1)
                kx_px = np.clip(frame_peaks[:, 1].astype(int), 0, KX - 1)
                frame_peaks = frame_peaks[~beamstop_mask[ky_px, kx_px]]

            # Greedy NMS
            if len(frame_peaks) > 1:
                order = np.argsort(-frame_peaks[:, 2])
                frame_peaks = frame_peaks[order]
                kept = np.ones(len(frame_peaks), dtype=bool)
                for j in range(len(frame_peaks)):
                    if not kept[j]:
                        continue
                    dy = frame_peaks[j+1:, 0] - frame_peaks[j, 0]
                    dx = frame_peaks[j+1:, 1] - frame_peaks[j, 1]
                    kept[j+1:][(dy*dy + dx*dx) <= min_d2] = False
                frame_peaks = frame_peaks[kept]

            n = min(len(frame_peaks), MAX_PEAKS)
            if n > 0:
                out[iy, ix, :n, :] = frame_peaks[:n]

        return out

    # ── Dispatch: 4D or 5D ────────────────────────────────────────────────────
    if nav_dim == 2:
        return _process_4d(block_f32)
    else:
        n_lead = nav_shape_ghost[0]
        KY2, KX2 = block_f32.shape[-2], block_f32.shape[-1]
        # Ghost block is (t, ny_ghost, nx_ghost, KY, KX) for 5D
        ny_ghost = nav_shape_ghost[1]
        nx_ghost = nav_shape_ghost[2]
        ny = max(1, ny_ghost - 2 * depth_px)
        nx = max(1, nx_ghost - 2 * depth_px)
        out5 = np.full((n_lead, ny, nx, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        for t in range(n_lead):
            out5[t] = _process_4d(block_f32[t])
        return out5


def _do_compute_vectors(
    signal, params: dict, main_window, signal_tree,
    shm_name: str = None,
    beamstop_mask: np.ndarray = None,
    on_chunk_done=None,
    stopped_flag=None,
):
    """
    Batch compute via dask.array.map_overlap.

    Nav-space Gaussian blur is applied with ghost zones (depth=ceil(3σ)) so
    chunk boundaries are handled correctly, then _find_vectors_single_frame
    is applied to every pattern.  Results are collected as a flat buffer with
    per-position offsets.

    signal.data may be a numpy array or a lazy dask array.  For numpy inputs
    map_overlap operates on a trivially-chunked array (one chunk = full nav
    block) so no data is copied.  NEVER call .compute() on the full dataset.

    Parameters
    ----------
    shm_name : str | None
        Pre-existing float32 SharedMemory segment written live per chunk.
    on_chunk_done : callable(nav_slice_2d, count_subarray) | None
        Called after the full compute completes (shm update path).
    stopped_flag : list[bool] | None
        Checked after map_overlap returns; early-exit not supported mid-compute.
    """
    import functools
    import dask.array as da
    from spyde.signals.diffraction_vectors import (
        N_COLS, COL_KX, COL_KY, COL_TIME, COL_INTENSITY, SpyDEDiffractionVectors
    )

    nav_dim = signal.axes_manager.navigation_dimension
    sig_dim = signal.axes_manager.signal_dimension
    sig_ax = signal.axes_manager.signal_axes
    sig_shape = signal.axes_manager.signal_shape  # (ky, kx) in HS order

    sigma = float(params["sigma"])
    depth_px = int(np.ceil(3 * sigma))

    kernel_r = int(params["kernel_radius"])
    threshold = float(params["threshold"])
    min_dist = int(params["min_distance"])
    subpixel = bool(params.get("subpixel", True))

    ky_scale = float(sig_ax[1].scale)
    ky_offset = float(sig_ax[1].offset)
    kx_scale = float(sig_ax[0].scale)
    kx_offset = float(sig_ax[0].offset)

    # ── Chunk size: larger chunks on GPU to amortise Dask scheduling overhead ─
    # On GPU the bottleneck is H2D latency (~1 ms) not RAM, so we use as many
    # patterns per chunk as fit in available VRAM (leaving 2 GB headroom).
    # On CPU we stay within max_ram_mb.
    vram_mb = 0.0
    if _GPU_KERNELS_AVAILABLE:
        try:
            from numba import cuda as _nc
            if _nc.is_available():
                mem = _nc.current_context().get_memory_info()
                # leave 2 GB headroom for working buffers (raw_corr, peak_mask, etc.)
                vram_mb = max(0.0, mem.free / 1e6 - 2048)
        except Exception:
            pass
    chunk_nav = _nav_chunk_size(sigma, max_ram_mb=200, sig_shape=sig_shape,
                                vram_mb=vram_mb)

    raw = signal.data
    nav_shape_full = raw.shape[:nav_dim]
    nav_2d_shape = nav_shape_full[-2:]
    n_nav_y, n_nav_x = nav_2d_shape

    # Disk stats: passed to chunk_fn so workers don't recompute them.
    _disk = _make_disk(kernel_r)
    _n = _disk.shape[0] * _disk.shape[1]
    _t_mean = float(_disk.mean())
    _t_std = float(np.sqrt(np.sum((_disk - _t_mean) ** 2) / _n))
    disk_stats = (_n, _t_mean, _t_std)
    # CPU fallback also needs disk_fft
    pH = next_fast_len(sig_shape[0] + 2 * kernel_r)
    pW = next_fast_len(sig_shape[1] + 2 * kernel_r)
    disk_fft = _get_disk_fft(kernel_r, pH, pW)

    # ── Build chunked dask array ──────────────────────────────────────────────
    nav_chunks_tuple = tuple(chunk_nav for _ in range(nav_dim))
    sig_chunks_tuple = tuple(s for s in raw.shape[nav_dim:])
    if isinstance(raw, np.ndarray):
        da_data = da.from_array(raw.astype(np.float32),
                                chunks=nav_chunks_tuple + sig_chunks_tuple)
    else:
        da_data = raw.rechunk(nav_chunks_tuple + sig_chunks_tuple)

    # ── Build the map_overlap graph ───────────────────────────────────────────
    # trim=False: chunk_fn receives the full ghost-padded block and handles
    # trimming itself so the blur can use all ghost rows for correct boundaries.
    # drop_axis removes (ky, kx); new_axis adds (MAX_PEAKS, 3).
    # Output: (nav_y, nav_x, MAX_PEAKS, 3) [4D] or (t, ny, nx, MAX_PEAKS, 3) [5D].
    depth_dict = {i: depth_px for i in range(nav_dim)}
    sig_axes_idx = list(range(nav_dim, nav_dim + sig_dim))
    new_axes_idx = list(range(nav_dim, nav_dim + 2))
    out_chunks = da_data.chunks[:nav_dim] + ((MAX_PEAKS,), (3,))

    chunk_fn = functools.partial(
        _find_vectors_chunk,
        depth_px=depth_px,
        nav_dim=nav_dim,
        sigma=sigma,
        kernel_r=kernel_r,
        threshold=threshold,
        min_dist=min_dist,
        subpixel=subpixel,
        beamstop_mask=beamstop_mask,
        disk_fft=disk_fft,
        disk_stats=disk_stats,
    )

    # Annotate with GPU resource so the Dask distributed scheduler routes chunk
    # tasks to workers started with --resources GPU=1.  Best-effort — silently
    # ignored if no GPU workers are registered or no distributed client is active.
    import dask
    with dask.annotate(resources={"GPU": 1}):
        peaks_padded = da.map_overlap(
            chunk_fn,
            da_data,
            depth=depth_dict,
            boundary="reflect",
            dtype=np.float32,
            trim=False,
            drop_axis=sig_axes_idx,
            new_axis=new_axes_idx,
            chunks=out_chunks,
        )

    result_padded = peaks_padded.compute()

    if stopped_flag is not None and stopped_flag[0]:
        return None

    # ── Unpack padded result into flat buffer ─────────────────────────────────
    # result_padded shape: (nav_y, nav_x, MAX_PEAKS, 3)  [4D]
    #                   or (t, nav_y, nav_x, MAX_PEAKS, 3) [5D]
    # Valid peaks have finite ky (col 0); NaN-padded slots are ignored.
    def _count_valid(arr_mp3):
        """Number of finite rows in (MAX_PEAKS, 3) — first NaN ends the run."""
        valid = np.isfinite(arr_mp3[:, 0])
        return int(valid.sum())

    if nav_dim == 2:
        N_total = sum(
            _count_valid(result_padded[iy, ix])
            for iy in range(n_nav_y) for ix in range(n_nav_x)
        )
        flat_buffer = np.zeros((N_total, N_COLS), dtype=np.float32)
        flat_buffer[:, COL_TIME] = -1.0
        cursor = 0
        for iy in range(n_nav_y):
            for ix in range(n_nav_x):
                slot = result_padded[iy, ix]          # (MAX_PEAKS, 3)
                valid = np.isfinite(slot[:, 0])
                peaks = slot[valid]
                n = len(peaks)
                if n == 0:
                    continue
                flat_buffer[cursor:cursor+n, 0] = ix
                flat_buffer[cursor:cursor+n, 1] = iy
                flat_buffer[cursor:cursor+n, COL_KX] = peaks[:, 1] * kx_scale + kx_offset
                flat_buffer[cursor:cursor+n, COL_KY] = peaks[:, 0] * ky_scale + ky_offset
                flat_buffer[cursor:cursor+n, COL_INTENSITY] = peaks[:, 2]
                cursor += n
    else:
        leading_size = nav_shape_full[0]
        N_total = sum(
            _count_valid(result_padded[t, iy, ix])
            for t in range(leading_size)
            for iy in range(n_nav_y) for ix in range(n_nav_x)
        )
        flat_buffer = np.zeros((N_total, N_COLS), dtype=np.float32)
        cursor = 0
        for t in range(leading_size):
            for iy in range(n_nav_y):
                for ix in range(n_nav_x):
                    slot = result_padded[t, iy, ix]    # (MAX_PEAKS, 3)
                    valid = np.isfinite(slot[:, 0])
                    peaks = slot[valid]
                    n = len(peaks)
                    if n == 0:
                        continue
                    flat_buffer[cursor:cursor+n, 0] = ix
                    flat_buffer[cursor:cursor+n, 1] = iy
                    flat_buffer[cursor:cursor+n, COL_KX] = peaks[:, 1] * kx_scale + kx_offset
                    flat_buffer[cursor:cursor+n, COL_KY] = peaks[:, 0] * ky_scale + ky_offset
                    flat_buffer[cursor:cursor+n, COL_TIME] = float(t)
                    flat_buffer[cursor:cursor+n, COL_INTENSITY] = peaks[:, 2]
                    cursor += n

    # ── Live shm / progress callback ──────────────────────────────────────────
    if shm_name is not None or on_chunk_done is not None:
        count_map = np.zeros(nav_2d_shape, dtype=np.int32)
        if nav_dim == 2:
            for iy in range(n_nav_y):
                for ix in range(n_nav_x):
                    count_map[iy, ix] = _count_valid(result_padded[iy, ix])
        else:
            for iy in range(n_nav_y):
                for ix in range(n_nav_x):
                    count_map[iy, ix] = sum(
                        _count_valid(result_padded[t, iy, ix])
                        for t in range(leading_size)
                    )
        if shm_name is not None:
            from multiprocessing import shared_memory as _shm_mod
            try:
                shm_handle = _shm_mod.SharedMemory(name=shm_name, create=False)
                shm_buf = np.ndarray(nav_2d_shape, dtype=np.float32, buffer=shm_handle.buf)
                shm_buf[:] = count_map.astype(np.float32)
                shm_handle.close()
            except Exception:
                pass
        if on_chunk_done is not None:
            on_chunk_done((slice(None), slice(None)), count_map)

    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat_buffer,
        full_nav_shape=nav_shape_full,
        sig_shape=sig_shape,
        sig_axes=sig_ax,
        kernel_radius_px=float(kernel_r),
        kernel_radius_data=float(kernel_r) * sig_ax[0].scale,
        params=dict(params),
    )


def _copy_nav_axes_to(source_signal, target_signal):
    """Copy navigation axis calibration from source to target."""
    src_axes = source_signal.axes_manager.navigation_axes
    tgt_axes = target_signal.axes_manager.navigation_axes
    for i, ax in enumerate(src_axes):
        if i < len(tgt_axes):
            tgt_axes[i].scale = ax.scale
            tgt_axes[i].offset = ax.offset
            tgt_axes[i].units = ax.units
            tgt_axes[i].name = ax.name


def _update_scatter(circ_item, plus_item, vecs, iy: int, ix: int, r_base: float):
    """
    Update scatter overlay items for navigation position (iy, ix).

    Uses the CSR flat buffer — O(1) lookup, no dense array needed.
    Columns in vecs.at(): [nav_x, nav_y, kx_data, ky_data, intensity]
    pyqtgraph scene convention: x=column, y=row → pos=(ky, kx).
    """
    rows = vecs.at(iy, ix)
    if len(rows) == 0:
        circ_item.setData([])
        plus_item.setData([])
        return
    kx_vals = rows[:, 2]  # calibrated kx (Å⁻¹)
    ky_vals = rows[:, 3]  # calibrated ky (Å⁻¹)
    size = r_base * 2
    spots_c = [{"pos": (float(ky), float(kx)), "size": size}
               for kx, ky in zip(kx_vals, ky_vals)]
    spots_p = [{"pos": s["pos"]} for s in spots_c]
    circ_item.setData(spots_c)
    plus_item.setData(spots_p)


# ─────────────────────────────────────────────────────────────────────────────
# Live Virtual Vector Imaging caret
# ─────────────────────────────────────────────────────────────────────────────

def _add_vvi_caret(vecs, new_tree, nav_plot, sig_ax_ref):
    """
    Install a live virtual-image caret on the signal plot of `new_tree`.

    Places two concentric CircleROIs (outer=blue, inner=cyan) on the signal
    (diffraction) plot.  On every drag, recomputes the virtual image using
    the GPU (virtual_image_from_roi_gpu) when available, otherwise falls back
    to the direct numpy path.  For 5D datasets uses nav_offsets[0] to isolate
    the current time frame in O(1) before the distance test.

    The result is a live HAADF/BF-style virtual image that updates at
    ~5-20 ms per drag event for typical datasets.
    """
    from pyqtgraph import CircleROI, mkPen
    from PySide6 import QtCore as _QC

    if not new_tree.signal_plots:
        return
    sig_plot = new_tree.signal_plots[0]

    # ── Initial ROI geometry ─────────────────────────────────────────────────
    # Place outer ROI at 3× kernel radius, inner at 0 (filled disk by default)
    r_outer_data = vecs.kernel_radius_data * 3.0
    r_inner_data = 0.0

    # Centre in signal data coords
    if sig_ax_ref is not None:
        cx0 = float(sig_ax_ref[0].offset + sig_ax_ref[0].scale * sig_ax_ref[0].size / 2)
        cy0 = float(sig_ax_ref[1].offset + sig_ax_ref[1].scale * sig_ax_ref[1].size / 2)
    else:
        cx0 = cy0 = 0.0

    def _make_circle_roi(cx, cy, r, pen):
        return CircleROI(
            pos=(cx - r, cy - r),
            size=(2 * r, 2 * r),
            pen=pen,
            removable=False,
        )

    roi_outer = _make_circle_roi(cx0, cy0, r_outer_data, mkPen("c", width=2))
    roi_inner = _make_circle_roi(cx0, cy0, r_inner_data + 1e-6, mkPen("y", width=1.5))
    roi_inner.setVisible(False)  # hidden until user explicitly enables annulus

    sig_plot.addItem(roi_outer)
    sig_plot.addItem(roi_inner)

    # Keep latest levels for consistent display during drag
    _levels = [None]
    _inner_active = [False]

    # ── Update function ──────────────────────────────────────────────────────
    def _recompute_vvi():
        # Read ROI centres and radii from data-unit positions
        def _roi_centre_radius(roi):
            pos = roi.pos()
            size = roi.size()
            cx = float(pos.x() + size.x() / 2)
            cy = float(pos.y() + size.y() / 2)
            r = float(size.x() / 2)
            return cx, cy, r

        cx, cy, r_out = _roi_centre_radius(roi_outer)
        r_in = 0.0
        if _inner_active[0] and roi_inner.isVisible():
            _, _, r_in = _roi_centre_radius(roi_inner)
            r_in = min(r_in, r_out * 0.99)

        # For 5D datasets read the current time index from the signal tree
        t_cur = None
        if vecs.n_time > 0:
            try:
                t_cur = int(new_tree.root.axes_manager.indices[0])
            except Exception:
                t_cur = None

        # GPU path when available; falls back to CPU automatically
        img = vecs.virtual_image_from_roi_gpu(cx, cy, r_out, r_in, t=t_cur,
                                              intensity_weighted=True)

        finite = img[img > 0]
        if finite.size > 0:
            hi = float(finite.max())
            lo = 0.0
            _levels[0] = (lo, hi if hi > lo else lo + 1)

        lvl = _levels[0] if _levels[0] is not None else (0.0, 1.0)
        nav_plot.image_item.setImage(img, autoLevels=False, levels=lvl)

    # Throttle: at most one recompute per 30 ms while dragging
    _timer = _QC.QTimer()
    _timer.setInterval(30)
    _timer.setSingleShot(True)
    _timer.timeout.connect(_recompute_vvi)

    def _schedule():
        _timer.start()

    roi_outer.sigRegionChanged.connect(_schedule)
    roi_inner.sigRegionChanged.connect(_schedule)

    # ── Inner ring toggle ────────────────────────────────────────────────────
    # Double-click outer ROI to toggle the inner exclusion ring
    def _toggle_inner(ev=None):
        active = not _inner_active[0]
        _inner_active[0] = active
        roi_inner.setVisible(active)
        _recompute_vvi()

    roi_outer.sigClicked = getattr(roi_outer, "sigClicked", None)
    try:
        roi_outer.sigClicked.connect(_toggle_inner)
    except Exception:
        pass

    # Initial render
    _recompute_vvi()

    # Store refs on the tree so they aren't GC'd
    new_tree._vvi_rois = (roi_outer, roi_inner)
    new_tree._vvi_timer = _timer


# ─────────────────────────────────────────────────────────────────────────────
# Main caret action entry point
# ─────────────────────────────────────────────────────────────────────────────

def find_diffraction_vectors(
    toolbar: RoundedToolBar,
    action_name: str = "Find Diffraction Vectors",
    *args,
    **kwargs,
):
    """
    Build the Find Diffraction Vectors caret popout on a signal plot toolbar.

    Called once per toolbar by plot_control_toolbar.get_toolbar_actions_for_plot().
    Guard against duplicate builds with _FV_BUILT_TOOLBARS.
    """
    from PySide6 import QtCore as _QC, QtWidgets as _QW
    from pyqtgraph import CircleROI, mkPen, ScatterPlotItem, ImageItem
    import pyqtgraph as pg

    from spyde.drawing.toolbars.caret_group import CaretGroup

    tid = id(toolbar)
    if tid in _FV_BUILT_TOOLBARS:
        return
    _FV_BUILT_TOOLBARS.add(tid)

    plot = toolbar.plot
    signal = plot.plot_state.current_signal
    main_window = plot.main_window
    sig_ax = signal.axes_manager.signal_axes
    sig_scale = sig_ax[0].scale  # Å⁻¹/px

    # Grab a snapshot of the current diffraction pattern for auto-param estimation
    try:
        current_frame = np.asarray(plot.current_data, dtype=np.float32)
    except Exception:
        current_frame = np.ones(signal.axes_manager.signal_shape, dtype=np.float32)

    auto = _auto_params(current_frame)

    # ── State ─────────────────────────────────────────────────────────────────
    state = {
        "plot": plot,
        "signal": signal,
        "main_window": main_window,
        "nav_blur_cache": [NavBlurCache(sigma=auto["sigma"])],
        "circle_roi": [None],
        "corr_overlay": [None],       # pg.ImageItem — corr map on the signal plot
        "scatter_plus": [None],       # pg.ScatterPlotItem — peak centres
        "scatter_circles": [None],    # pg.ScatterPlotItem — disk circles
        "beamstop_overlay": [None],   # pg.ImageItem — translucent mask overlay
        "beamstop_mask": [None],      # (ky, kx) bool ndarray | None
        "masking_active": [False],    # True while "Mask beam stop" mode is on
        "show_corr": [False],
        "refit_timer": [None],
        "refit_generation": [0],
        "relay": [None],
        "active": [False],            # True while the action is toggled on
    }
    toolbar._fv_state = state

    # ── Build CaretGroup ──────────────────────────────────────────────────────
    caret = CaretGroup(title=action_name, toolbar=toolbar, action_name=action_name)
    toolbar.add_action_widget(action_name, caret, None)

    layout = caret.layout()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _lbl(text, parent):
        l = _QW.QLabel(text, parent)
        l.setStyleSheet("color: white; font-size: 10px;")
        l.setWordWrap(True)
        return l

    def _btn(text, parent, enabled=True):
        b = _QW.QPushButton(text, parent)
        b.setEnabled(enabled)
        b.setStyleSheet(
            "QPushButton { color: white; background: rgba(255,255,255,30); "
            "border: 1px solid rgba(255,255,255,60); padding: 3px 6px; }"
            "QPushButton:disabled { color: rgba(255,255,255,60); "
            "background: rgba(255,255,255,10); }"
        )
        return b

    def _make_slider_spin(parent, lo, hi, val, decimals, label_text, suffix=""):
        SCALE = 10 ** decimals
        row = _QW.QWidget(parent)
        h = _QW.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        lbl = _QW.QLabel(label_text, row)
        lbl.setStyleSheet("color: white; font-size: 10px;")
        spin = _QW.QDoubleSpinBox(row)
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSingleStep(10 ** -decimals)
        spin.setValue(val)
        spin.setFixedWidth(72 if suffix else 64)
        if suffix:
            spin.setSuffix(suffix)
        spin.setStyleSheet(
            "QDoubleSpinBox { color: white; background: rgba(255,255,255,40); "
            "border: 1px solid black; font-size: 10px; }"
        )
        slider = _QW.QSlider(_QC.Qt.Orientation.Horizontal, row)
        slider.setRange(int(lo * SCALE), int(hi * SCALE))
        slider.setValue(int(val * SCALE))

        def _spin_to_sl(v, _s=slider, _sc=SCALE):
            _s.blockSignals(True)
            _s.setValue(int(v * _sc))
            _s.blockSignals(False)

        def _sl_to_spin(v, _sp=spin, _sc=SCALE):
            _sp.blockSignals(True)
            _sp.setValue(v / _sc)
            _sp.blockSignals(False)

        spin.valueChanged.connect(_spin_to_sl)
        slider.valueChanged.connect(_sl_to_spin)
        h.addWidget(lbl)
        h.addWidget(slider, 1)
        h.addWidget(spin)
        return row, spin, slider

    # ── Parameter controls ────────────────────────────────────────────────────
    container = _QW.QWidget(caret)
    vbox = _QW.QVBoxLayout(container)
    vbox.setContentsMargins(4, 4, 4, 4)
    vbox.setSpacing(4)

    sigma_row, sigma_spin, sigma_slider = _make_slider_spin(
        container, 0.1, 2.0, auto["sigma"], 1, "Real-space σ", " px"
    )
    radius_row, radius_spin, radius_slider = _make_slider_spin(
        container, 1.0, 50.0, float(auto["kernel_radius"]), 0, "Kernel radius", " px"
    )
    thresh_row, thresh_spin, thresh_slider = _make_slider_spin(
        container, 0.0, 1.0, 0.5, 2, "Threshold"
    )
    mindist_row, mindist_spin, mindist_slider = _make_slider_spin(
        container, 1.0, 100.0, float(auto["min_distance"]), 0, "Min distance", " px"
    )

    subpixel_chk = _QW.QCheckBox("Subpixel refinement (CoM)", container)
    subpixel_chk.setChecked(auto["subpixel"])
    subpixel_chk.setStyleSheet("QCheckBox { color: white; font-size: 10px; }")

    show_corr_chk = _QW.QCheckBox("Show correlation image", container)
    show_corr_chk.setChecked(False)
    show_corr_chk.setStyleSheet("QCheckBox { color: white; font-size: 10px; }")

    mask_bs_chk = _QW.QCheckBox("Mask beam stop (click dark region)", container)
    mask_bs_chk.setChecked(False)
    mask_bs_chk.setStyleSheet("QCheckBox { color: white; font-size: 10px; }")

    clear_mask_btn = _btn("Clear beam stop mask", container)
    clear_mask_btn.setEnabled(False)

    status_lbl = _lbl("", container)

    compute_btn = _btn("Compute", container)

    for w in [sigma_row, radius_row, thresh_row, mindist_row,
               subpixel_chk, show_corr_chk, mask_bs_chk, clear_mask_btn,
               compute_btn, status_lbl]:
        vbox.addWidget(w)

    layout.addWidget(container)

    # ── Relay for thread → GUI marshal ────────────────────────────────────────
    class _VectorRelay(_QC.QObject):
        vectors_ready = _QC.Signal(object, object, object, float)
        compute_done = _QC.Signal(object, object, int)   # vecs, new_tree, kernel_r
        chunk_done = _QC.Signal(object, object)           # nav_2d_slices, count_subarray
        status_update = _QC.Signal(str)

    relay = _VectorRelay(toolbar)  # parented → stays on GUI thread
    state["relay"][0] = relay

    # ── Overlay items on the signal plot ─────────────────────────────────────
    # Correlation ImageItem — sits above image_item but below ROIs
    corr_overlay = ImageItem()
    corr_overlay.setZValue(5)
    corr_overlay.setVisible(False)
    # Match the same transform as image_item so data coords align
    corr_overlay.setTransform(plot.image_item.transform())
    plot.addItem(corr_overlay)
    state["corr_overlay"][0] = corr_overlay

    scatter_circles = ScatterPlotItem(
        symbol="o", pen=mkPen("r", width=2.0), brush=None, pxMode=False
    )
    scatter_circles.setZValue(10)
    scatter_plus = ScatterPlotItem(
        symbol="+", size=12, pen=mkPen("r", width=2.0), brush=None
    )
    scatter_plus.setZValue(11)
    plot.addItem(scatter_circles)
    plot.addItem(scatter_plus)
    state["scatter_circles"][0] = scatter_circles
    state["scatter_plus"][0] = scatter_plus

    # Beam stop mask overlay — translucent red over masked pixels
    beamstop_overlay = ImageItem()
    beamstop_overlay.setZValue(6)
    beamstop_overlay.setVisible(False)
    beamstop_overlay.setTransform(plot.image_item.transform())
    beamstop_overlay.setOpacity(0.45)
    plot.addItem(beamstop_overlay)
    state["beamstop_overlay"][0] = beamstop_overlay

    # ── CircleROI on the signal plot ──────────────────────────────────────────
    r_data_init = auto["kernel_radius"] * sig_scale
    cx = sig_ax[1].size / 2.0 * sig_ax[1].scale + sig_ax[1].offset
    cy = sig_ax[0].size / 2.0 * sig_ax[0].scale + sig_ax[0].offset

    circle_roi = CircleROI(
        pos=(cx - r_data_init, cy - r_data_init),
        size=(2 * r_data_init, 2 * r_data_init),
        pen=mkPen("r", width=1.5),
    )
    plot.addItem(circle_roi)
    state["circle_roi"][0] = circle_roi

    # Two-way binding: ROI ↔ radius spinbox
    _updating_roi = [False]
    _updating_spin = [False]

    def _roi_to_spin():
        if _updating_roi[0]:
            return
        r_px = circle_roi.size().x() / 2.0 / sig_scale
        _updating_spin[0] = True
        radius_spin.blockSignals(True)
        radius_spin.setValue(max(1.0, r_px))
        radius_spin.blockSignals(False)
        _updating_spin[0] = False
        _schedule_recompute()

    def _spin_to_roi(r_px):
        if _updating_spin[0]:
            return
        r_d = r_px * sig_scale
        _updating_roi[0] = True
        circle_roi.blockSignals(True)
        circle_roi.setPos(cx - r_d, cy - r_d)
        circle_roi.setSize((2 * r_d, 2 * r_d))
        circle_roi.blockSignals(False)
        _updating_roi[0] = False

    circle_roi.sigRegionChanged.connect(_roi_to_spin)
    radius_spin.valueChanged.connect(_spin_to_roi)

    # ── Beam stop mask: flood-fill on click ───────────────────────────────────
    def _flood_fill_mask(frame: np.ndarray, seed_yx: tuple) -> np.ndarray:
        """
        Return a bool mask of connected pixels below the Otsu threshold that
        are reachable from seed_yx.  Used to isolate the beam stop region.
        """
        from skimage.filters import threshold_otsu
        from skimage.segmentation import flood

        try:
            thresh = threshold_otsu(frame)
        except Exception:
            thresh = float(frame.mean())
        # Beam stop pixels are dark — below the threshold
        dark_mask = frame < thresh
        sy, sx = int(round(seed_yx[0])), int(round(seed_yx[1]))
        sy = max(0, min(sy, frame.shape[0] - 1))
        sx = max(0, min(sx, frame.shape[1] - 1))
        if not dark_mask[sy, sx]:
            # Clicked on a bright pixel — expand threshold to include it
            thresh = float(frame[sy, sx]) * 1.05
            dark_mask = frame <= thresh
        filled = flood(dark_mask, (sy, sx), connectivity=1)
        return filled

    def _update_beamstop_overlay():
        mask = state["beamstop_mask"][0]
        overlay = state["beamstop_overlay"][0]
        if overlay is None:
            return
        if mask is None or not mask.any():
            overlay.setVisible(False)
            return
        # Build RGBA: red channel only, fully saturated where masked
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[mask, 0] = 255   # red
        rgba[mask, 3] = 180   # alpha
        overlay.setTransform(plot.image_item.transform())
        overlay.setImage(rgba)
        overlay.setVisible(True)

    def _on_plot_clicked(event):
        if not state["masking_active"][0]:
            return
        # scene().sigMouseClicked passes a QGraphicsSceneMouseEvent; position
        # is in scene (data) coordinates already for pyqtgraph's GraphicsScene.
        try:
            scene_pos = event.scenePos()
        except AttributeError:
            scene_pos = event.pos()
        # Map scene → image-item pixel coords via the inverse image transform
        tr = plot.image_item.transform()
        inv_tr, ok = tr.inverted()
        if not ok:
            return
        data_pt = inv_tr.map(scene_pos)
        px_y, px_x = int(data_pt.y()), int(data_pt.x())
        frame = plot.current_data
        if frame is None or not isinstance(frame, np.ndarray):
            return
        if frame.ndim != 2:
            return
        if not (0 <= px_y < frame.shape[0] and 0 <= px_x < frame.shape[1]):
            return
        new_mask = _flood_fill_mask(frame.astype(np.float32), (px_y, px_x))
        existing = state["beamstop_mask"][0]
        if existing is not None:
            new_mask = existing | new_mask
        state["beamstop_mask"][0] = new_mask
        _update_beamstop_overlay()
        clear_mask_btn.setEnabled(True)
        _schedule_recompute()

    # Install click handler on the viewbox
    plot.getViewBox().scene().sigMouseClicked.connect(_on_plot_clicked)

    def _on_mask_bs_toggled(checked):
        state["masking_active"][0] = bool(checked)
        if not checked and state["beamstop_mask"][0] is None:
            beamstop_overlay.setVisible(False)

    mask_bs_chk.stateChanged.connect(_on_mask_bs_toggled)

    def _on_clear_mask():
        state["beamstop_mask"][0] = None
        _update_beamstop_overlay()
        clear_mask_btn.setEnabled(False)
        _schedule_recompute()

    clear_mask_btn.clicked.connect(_on_clear_mask)

    # ── Apply results from worker thread ──────────────────────────────────────
    def _apply_results(corr_map, raw_corr, peaks, elapsed_ms):
        circ = state["scatter_circles"][0]
        plus = state["scatter_plus"][0]
        cov = state["corr_overlay"][0]

        # Update correlation overlay (always keep the latest map, visibility is toggled)
        if cov is not None:
            # Sync transform in case image transform changed (e.g. first data load)
            cov.setTransform(plot.image_item.transform())
            cov.setImage(corr_map)

        r_data = radius_spin.value() * sig_scale
        if peaks is not None and len(peaks) > 0:
            # peaks columns: [ky_px, kx_px, intensity]
            # data coords: scene_x = col = ky*scale+off, scene_y = row = kx*scale+off
            spots_circles = [
                {
                    "pos": (
                        float(p[0]) * sig_ax[1].scale + sig_ax[1].offset,
                        float(p[1]) * sig_ax[0].scale + sig_ax[0].offset,
                    ),
                    "size": r_data * 2,
                }
                for p in peaks
            ]
            spots_plus = [{"pos": s["pos"]} for s in spots_circles]
            if circ is not None:
                circ.setData(spots_circles)
            if plus is not None:
                plus.setData(spots_plus)
        else:
            if circ is not None:
                circ.setData([])
            if plus is not None:
                plus.setData([])

        n = len(peaks) if peaks is not None else 0
        status_lbl.setText(f"{n} peaks · {elapsed_ms:.1f} ms")

    relay.vectors_ready.connect(_apply_results)

    # ── Compute-done handler (GUI thread) ─────────────────────────────────────
    def _on_compute_done(vecs, new_tree, k_r):
        import traceback as _tb
        try:
            _on_compute_done_impl(vecs, new_tree, k_r)
        except Exception as _exc:
            _tb.print_exc()
            status_lbl.setText(f"Tree error: {_exc}")
            compute_btn.setEnabled(True)

    def _on_compute_done_impl(vecs, new_tree, k_r):
        sig_ax_ref = signal.axes_manager.signal_axes
        r_base = k_r * float(sig_ax_ref[0].scale)

        # ── 1. Final count map on the nav plot ────────────────────────────────
        final_count_map = vecs.count_map().astype(np.float32)
        nav_plot_windows = list(new_tree.navigator_plot_manager.plot_windows.keys())
        nav_plot_ref2 = None
        if nav_plot_windows:
            nav_pw2 = nav_plot_windows[0]
            nav_plots2 = new_tree.navigator_plot_manager.plots.get(nav_pw2, [])
            if nav_plots2:
                nav_plot_ref2 = nav_plots2[0]
        if nav_plot_ref2 is not None:
            lo = float(final_count_map.min())
            hi = float(final_count_map.max())
            if hi <= lo:
                hi = lo + 1
            nav_plot_ref2.image_item.setImage(
                final_count_map, autoLevels=False, levels=(lo, hi)
            )
            nav_plot_ref2.needs_auto_level = True

        # ── 2. Attach scatter overlays + hook each signal plot's update_data ──
        #
        # CrosshairSelector → delayed_update_data → update_from_navigation_selection
        #   → child_plot.update_data(new_data)
        #
        # We wrap update_data on each signal plot so the scatter fires every
        # time the diffraction pattern updates.  The (ix, iy) is read from the
        # parent_selector at call time — always in sync with the image update.
        def _read_position(signal_plot):
            """
            Return (ix, iy) from the selector driving signal_plot.

            pyqtgraph displays count_map[ny, nx] in col-major order, so the
            scene x-axis maps to the first array index (iy = row) and y-axis to
            the second (ix = col).  CrosshairSelector returns [[scene_x, scene_y]]
            = [[iy_pixel, ix_pixel]], so we swap to get (ix, iy).
            """
            try:
                selector = signal_plot.plot_window.parent_selector
                if selector is not None:
                    raw_idx = selector.get_selected_indices()
                    idx = np.mean(raw_idx, axis=0).astype(int)
                    # idx[0] = scene_x = first array dim = iy (row)
                    # idx[1] = scene_y = second array dim = ix (col)
                    iy, ix = int(idx[0]), int(idx[1])
                    return ix, iy
                # Fallback: HyperSpy axes_manager indices = (ix, iy) convention
                nav_idx = new_tree.root.axes_manager.indices
                return int(nav_idx[0]), int(nav_idx[1])
            except Exception:
                return 0, 0

        def _make_hooked(orig_ud, circ, plus, signal_plot):
            def _hooked_ud(new_data, force=False):
                orig_ud(new_data, force=force)
                ix, iy = _read_position(signal_plot)
                _update_scatter(circ, plus, vecs, iy, ix, r_base)
            return _hooked_ud

        for sp in new_tree.signal_plots:
            circ_item = ScatterPlotItem(
                symbol="o", pen=mkPen("r", width=2.0), brush=None, pxMode=False
            )
            plus_item = ScatterPlotItem(
                symbol="+", size=12, pen=mkPen("r", width=2.0), brush=None
            )
            circ_item.setZValue(10)
            plus_item.setZValue(11)
            sp.addItem(circ_item)
            sp.addItem(plus_item)
            sp.update_data = _make_hooked(sp.update_data, circ_item, plus_item, sp)

            # Initial draw at current crosshair position
            ix, iy = _read_position(sp)
            _update_scatter(circ_item, plus_item, vecs, iy, ix, r_base)

        status_lbl.setText(
            f"Done. {int(vecs.flat_buffer.shape[0])} total vectors."
        )
        compute_btn.setEnabled(True)

        # ── 3. Build KDTree in background, then add live VVI caret ───────────
        # Upload flat_buffer to GPU (preferred) or fall back to CPU.
        # Done off the GUI thread; caret is installed once ready.
        def _build_tree_and_install():
            gpu_ok = vecs.upload_to_gpu()
            if not gpu_ok:
                vecs.build_kdtree()  # CPU fallback for small ROIs
            _QC.QMetaObject.invokeMethod(
                relay, "_install_vvi_caret",
                _QC.Qt.ConnectionType.QueuedConnection,
            )

        # Store refs needed by the deferred install
        relay._vvi_vecs = vecs
        relay._vvi_new_tree = new_tree
        relay._vvi_nav_plot_ref = nav_plot_ref2
        relay._vvi_sig_ax = sig_ax_ref

        import threading as _threading
        _threading.Thread(target=_build_tree_and_install, daemon=True).start()

    relay.compute_done.connect(_on_compute_done)

    # ── Live virtual image caret (installed after KDTree is ready) ────────────
    def _install_vvi_caret():
        vecs = relay._vvi_vecs
        new_tree = relay._vvi_new_tree
        nav_plot = relay._vvi_nav_plot_ref
        sig_ax_ref = relay._vvi_sig_ax
        if nav_plot is None or not new_tree.signal_plots:
            return
        _add_vvi_caret(vecs, new_tree, nav_plot, sig_ax_ref)

    # Attach as a slot so QMetaObject.invokeMethod can reach it
    _VectorRelay._install_vvi_caret = _QC.Slot()(_install_vvi_caret)
    relay._install_vvi_caret = _install_vvi_caret

    # ── Show-correlation toggle ───────────────────────────────────────────────
    def _on_show_corr_changed(checked):
        state["show_corr"][0] = bool(checked)
        cov = state["corr_overlay"][0]
        if cov is not None:
            cov.setVisible(bool(checked))
        plot.image_item.setVisible(not bool(checked))

    show_corr_chk.stateChanged.connect(_on_show_corr_changed)

    # ── Refit (live preview) ──────────────────────────────────────────────────
    refit_timer = _QC.QTimer()
    refit_timer.setInterval(50)
    refit_timer.setSingleShot(True)
    state["refit_timer"][0] = refit_timer

    def _schedule_recompute():
        if state["active"][0]:
            refit_timer.start()

    def _do_refit():
        from dask.distributed import Future as _Future
        current = plot.current_data
        if isinstance(current, _Future):
            # Data hasn't arrived yet — re-schedule once the future resolves.
            def _retry_on_gui(_fut):
                _QC.QMetaObject.invokeMethod(
                    refit_timer, "start",
                    _QC.Qt.ConnectionType.QueuedConnection,
                )
            current.add_done_callback(_retry_on_gui)
            return
        try:
            raw_frame = np.asarray(current, dtype=np.float32).copy()
        except Exception:
            return

        # Feed the current dask chunk to NavBlurCache (triggers async blur)
        nav_blur_cache = state["nav_blur_cache"][0]
        try:
            cached_dask = getattr(signal, "cached_dask_array", None)
            if cached_dask is not None and cached_dask.core_cached_blocks:
                block = cached_dask.core_cached_blocks[0]
                from dask.distributed import Future
                if not isinstance(block, Future):
                    chunk_id = tuple(cached_dask.core_cached_block_inds[0])
                    nav_blur_cache.update_chunk(block, chunk_id)
        except Exception:
            pass

        state["refit_generation"][0] += 1
        my_gen = state["refit_generation"][0]

        sigma_val = sigma_spin.value()
        radius_val = int(radius_spin.value())
        thresh_val = thresh_spin.value()
        mindist_val = int(mindist_spin.value())
        subpixel_val = subpixel_chk.isChecked()

        # Compute local chunk position for NavBlurCache lookup
        try:
            nav_idx = signal.axes_manager.indices
            iy_global, ix_global = int(nav_idx[0]), int(nav_idx[1])
            chunk_sizes = signal.data.chunks
            chunk_nav_y = chunk_sizes[0][0]
            chunk_nav_x = chunk_sizes[1][0]
            iy_local = iy_global % chunk_nav_y
            ix_local = ix_global % chunk_nav_x
        except Exception:
            iy_local, ix_local = 0, 0

        beamstop_mask_val = state["beamstop_mask"][0]

        def _run():
            if state["refit_generation"][0] != my_gen:
                return
            t0 = time.perf_counter()
            blurred = nav_blur_cache.get_blurred(iy_local, ix_local, raw_frame)
            corr_map, raw_corr, peaks = _find_vectors_single_frame(
                blurred, radius_val, thresh_val, mindist_val,
                subpixel=subpixel_val, beamstop_mask=beamstop_mask_val,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if state["refit_generation"][0] == my_gen:
                relay.vectors_ready.emit(corr_map, raw_corr, peaks, elapsed_ms)

        threading.Thread(target=_run, daemon=True).start()

    refit_timer.timeout.connect(_do_refit)

    # ── Hook plot.update_data for live cursor updates ─────────────────────────
    _orig_update_data = plot.update_data

    def _hooked_update_data(new_data, force=False):
        _orig_update_data(new_data, force=force)
        _schedule_recompute()

    plot.update_data = _hooked_update_data

    # ── Trigger recompute on any param change ─────────────────────────────────
    # Connect both spin and slider: slider→spin suppresses spin.valueChanged via
    # blockSignals, so the slider must also connect directly.
    for sp in [sigma_spin, radius_spin, thresh_spin, mindist_spin]:
        sp.valueChanged.connect(_schedule_recompute)
    for sl in [sigma_slider, radius_slider, thresh_slider, mindist_slider]:
        sl.valueChanged.connect(lambda _: _schedule_recompute())
    subpixel_chk.stateChanged.connect(lambda _: _schedule_recompute())

    def _on_sigma_changed(val):
        state["nav_blur_cache"][0].invalidate(sigma=val)
        _schedule_recompute()

    sigma_spin.valueChanged.connect(_on_sigma_changed)
    sigma_slider.valueChanged.connect(lambda _: state["nav_blur_cache"][0].invalidate(sigma=sigma_spin.value()))

    # ── Compute (batch) ───────────────────────────────────────────────────────
    def _on_compute_clicked():
        import hyperspy.api as hs
        compute_btn.setEnabled(False)
        status_lbl.setText("Computing…")

        params = dict(
            sigma=sigma_spin.value(),
            kernel_radius=int(radius_spin.value()),
            threshold=thresh_spin.value(),
            min_distance=int(mindist_spin.value()),
            subpixel=subpixel_chk.isChecked(),
        )

        sig_ref = signal
        kernel_r = int(radius_spin.value())
        nav_shape_2d = tuple(sig_ref.axes_manager.navigation_shape[::-1])[-2:]
        shm_name = f"spyde_fv_{id(plot)}"

        from spyde.drawing.update_functions import ensure_live_buffer
        shm = ensure_live_buffer(nav_shape_2d, shm_name)
        state["_compute_shm"] = shm  # keep alive

        # ── Create result tree immediately ─────────────────────────────────────
        new_sig = sig_ref._deepcopy_with_new_data(sig_ref.data)

        new_sig.metadata.General.title = (
            sig_ref.metadata.get_item("General.title", "Signal") + " — Vectors"
        )

        # Zero-filled count map navigator — correct spatial extent, no data yet
        zero_count = np.zeros(nav_shape_2d, dtype=np.float32)
        nav_sig = hs.signals.BaseSignal(zero_count).T
        nav_sig.metadata.General.title = "Vector count map (computing…)"
        _copy_nav_axes_to(sig_ref, nav_sig)

        from spyde.drawing.selectors import CrosshairSelector
        main_window.add_signal(new_sig, navigators=[nav_sig],
                               selector_type=CrosshairSelector)
        new_tree = main_window.signal_trees[-1]

        # Locate the new tree's nav PlotWindow and its Plot for live updates
        nav_plot_windows = list(new_tree.navigator_plot_manager.plot_windows.keys())
        nav_plot_ref = [None]
        nav_pw_ref = [None]
        if nav_plot_windows:
            nav_pw_ref[0] = nav_plot_windows[0]
            nav_plots = new_tree.navigator_plot_manager.plots.get(nav_pw_ref[0], [])
            if nav_plots:
                nav_plot_ref[0] = nav_plots[0]

        # Show zeros immediately (correct extent, no NaN warnings)
        if nav_plot_ref[0] is not None:
            nav_plot_ref[0].image_item.setImage(
                zero_count, autoLevels=False, levels=(0, 1)
            )

        # ── Chunk-done relay: background thread → GUI thread ──────────────────
        # _do_compute_vectors calls on_chunk_done(nav_2d_slices, count_subarray)
        # from the background compute thread.  We relay through a Qt signal so
        # the image_item.setImage call happens on the GUI thread.
        _levels = [None]

        def _apply_chunk(nav_2d_slices, count_sub):
            nav_p = nav_plot_ref[0]
            if nav_p is None:
                return
            # Read the count_sub (already float-castable int32) and update levels
            sub = count_sub.astype(np.float32)
            finite = sub[np.isfinite(sub) & (sub > 0)]
            if finite.size > 0:
                hi = float(finite.max())
                if _levels[0] is None:
                    _levels[0] = (0.0, hi if hi > 0 else 1.0)
                elif hi > _levels[0][1]:
                    _levels[0] = (_levels[0][0], hi)
            # Read full current count buffer from shm for display
            from spyde.drawing.update_functions import read_live_buffer
            arr = read_live_buffer(nav_shape_2d, shm_name)
            lvl = _levels[0] if _levels[0] is not None else (0.0, 1.0)
            nav_p.image_item.setImage(arr, autoLevels=False, levels=lvl)

        relay.chunk_done.connect(_apply_chunk)

        def _on_status(msg):
            status_lbl.setText(msg)

        relay.status_update.connect(_on_status)

        _stopped = [False]

        def _stop():
            _stopped[0] = True
            if nav_pw_ref[0] is not None:
                _QC.QMetaObject.invokeMethod(
                    nav_pw_ref[0], "hide_stop_button",
                    _QC.Qt.ConnectionType.QueuedConnection,
                )
            compute_btn.setEnabled(True)
            status_lbl.setText("Stopped.")

        # Stop button on the new nav window (not the original signal window)
        if nav_pw_ref[0] is not None:
            nav_pw_ref[0].set_stop_fn(_stop)
        else:
            plot.plot_window.set_stop_fn(_stop)

        beamstop_mask_snap = state["beamstop_mask"][0]
        n_chunks_total = [0]
        n_chunks_done = [0]

        # Pre-count chunks for progress display
        import itertools as _it
        _nav_est = nav_shape_2d[0] * nav_shape_2d[1]
        _cn = _nav_chunk_size(params["sigma"], sig_shape=sig_ref.axes_manager.signal_shape)
        _n_cy = max(1, int(np.ceil(nav_shape_2d[0] / _cn)))
        _n_cx = max(1, int(np.ceil(nav_shape_2d[1] / _cn)))
        n_chunks_total[0] = _n_cy * _n_cx

        def _on_chunk_from_bg(nav_2d_slices, count_sub):
            """Called from background thread — relay to GUI via Qt signal."""
            if _stopped[0]:
                return
            n_chunks_done[0] += 1
            relay.chunk_done.emit(nav_2d_slices, count_sub)
            relay.status_update.emit(
                f"Computing… {n_chunks_done[0]}/{n_chunks_total[0]} chunks"
            )

        def _run():
            try:
                vecs = _do_compute_vectors(
                    sig_ref, params, main_window, None,
                    shm_name=shm_name,
                    beamstop_mask=beamstop_mask_snap,
                    on_chunk_done=_on_chunk_from_bg,
                    stopped_flag=_stopped,
                )
                if _stopped[0]:
                    return
                # Hide stop button on nav window
                if nav_pw_ref[0] is not None:
                    _QC.QMetaObject.invokeMethod(
                        nav_pw_ref[0], "hide_stop_button",
                        _QC.Qt.ConnectionType.QueuedConnection,
                    )
                relay.compute_done.emit(vecs, new_tree, kernel_r)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                if nav_pw_ref[0] is not None:
                    _QC.QMetaObject.invokeMethod(
                        nav_pw_ref[0], "hide_stop_button",
                        _QC.Qt.ConnectionType.QueuedConnection,
                    )
                _err_msg = f"Error: {exc}"
                _QC.QMetaObject.invokeMethod(
                    status_lbl, "setText",
                    _QC.Qt.ConnectionType.QueuedConnection,
                    _QC.Q_ARG(str, _err_msg),
                )
                _QC.QMetaObject.invokeMethod(
                    compute_btn, "setEnabled",
                    _QC.Qt.ConnectionType.QueuedConnection,
                    _QC.Q_ARG(bool, True),
                )

        threading.Thread(target=_run, daemon=True).start()

    compute_btn.clicked.connect(_on_compute_clicked)

    # ── Register CircleROI with the toolbar action (auto show/hide) ──────────
    toolbar.register_action_plot_item(action_name, circle_roi, key="disk_roi")

    # ── Activate / deactivate overlays when caret is toggled ─────────────────
    def _on_caret_toggled(visible):
        state["active"][0] = visible
        for item in (scatter_circles, scatter_plus, corr_overlay, circle_roi):
            item.setVisible(visible)
        if visible:
            show_corr = state["show_corr"][0]
            corr_overlay.setVisible(show_corr)
            plot.image_item.setVisible(not show_corr)
            # Restore beam stop overlay if a mask exists
            if state["beamstop_mask"][0] is not None:
                _update_beamstop_overlay()
            _schedule_recompute()
        else:
            plot.image_item.setVisible(True)
            beamstop_overlay.setVisible(False)
            state["masking_active"][0] = False
            mask_bs_chk.setChecked(False)
            plot.update_data = _orig_update_data

    try:
        act = toolbar._find_action(action_name)
        if act is not None:
            act.toggled.connect(_on_caret_toggled)
    except Exception:
        pass

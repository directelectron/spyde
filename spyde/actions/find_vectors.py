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
import logging
import threading
import time
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.fft import rfft2, irfft2, next_fast_len

log = logging.getLogger(__name__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Qt — only needed as a type hint on the legacy Qt entry point. Importing it
    # at runtime would pull PySide6 into the Qt-free Electron backend (which
    # reuses this module's compute core), so keep it deferred.
    from spyde.drawing.toolbars.toolbar import RoundedToolBar

# Cache of pre-computed disk FFTs keyed by (radius, padded_H, padded_W)
_DISK_FFT_CACHE: dict = {}

# ── Module-level guard (one caret per toolbar) ─────────────────────────────────
_FV_BUILT_TOOLBARS: set = set()

# Maximum peaks per frame in GPU subpixel output buffer
MAX_PEAKS: int = 512

# Cache of device-side disk kernel arrays keyed by kernel_r
_gpu_disk_cache: dict = {}

# Native-endian integer dtypes the GPU converts to float32 on device.
# Uploading the raw integers halves H2D bytes for 16-bit detectors and removes
# the host-side astype pass.  float64 and big-endian (e.g. mrc '>u2') data are
# converted on the host instead.
_GPU_NATIVE_DTYPES = {
    np.dtype(t) for t in (np.uint8, np.int8, np.uint16, np.int16,
                          np.uint32, np.int32)
}

# ── Device buffer pool ────────────────────────────────────────────────────────
# numba-cuda has no caching allocator: every device_array() is a cudaMalloc,
# and allocation/free are context-wide sync points.  With several chunk tasks
# in flight on the GPU worker's threads the malloc/free storm serialises the
# whole device.  Chunks within a run share a handful of shapes, so a small
# free-list pool removes nearly all allocations.
_GPU_POOL_MAX_PER_KEY = 8
_gpu_buffer_pool: dict = {}
_gpu_pool_bytes: list = [0]
_gpu_pool_max_bytes: list = [None]  # resolved lazily from device VRAM
_gpu_pool_lock = threading.Lock()


def _gpu_pool_cap() -> int:
    """Pool byte cap: half of total VRAM (a too-small cap rejects returns and
    the resulting cudaFree storm device-syncs every thread)."""
    if _gpu_pool_max_bytes[0] is None:
        cap = 2_000_000_000
        try:
            from numba import cuda as _cuda
            _free, total = _cuda.current_context().get_memory_info()
            cap = int(total * 0.5)
        except Exception as e:
            log.debug("VRAM probe failed, using default GPU pool cap: %s", e)
        _gpu_pool_max_bytes[0] = cap
    return _gpu_pool_max_bytes[0]


def _gpu_pool_get(shape, dtype):
    """Reuse a pooled device array of this shape/dtype, or allocate one."""
    from numba import cuda as _cuda
    key = (tuple(int(s) for s in shape), np.dtype(dtype).str)
    with _gpu_pool_lock:
        lst = _gpu_buffer_pool.get(key)
        if lst:
            arr = lst.pop()
            _gpu_pool_bytes[0] -= arr.nbytes
            return arr
    return _cuda.device_array(shape, dtype=dtype)


def _gpu_pool_put(*arrays):
    """Return device arrays to the pool (over-cap buffers are just dropped)."""
    with _gpu_pool_lock:
        for arr in arrays:
            if arr is None:
                continue
            key = (tuple(int(s) for s in arr.shape), np.dtype(arr.dtype).str)
            lst = _gpu_buffer_pool.setdefault(key, [])
            if (len(lst) < _GPU_POOL_MAX_PER_KEY
                    and _gpu_pool_bytes[0] + arr.nbytes <= _gpu_pool_cap()):
                lst.append(arr)
                _gpu_pool_bytes[0] += arr.nbytes


# ── Pinned (page-locked) host staging buffers ─────────────────────────────────
# H2D copies from pageable numpy (what dask hands us) block the calling
# thread and run at reduced bandwidth.  Staging through a reused pinned
# buffer makes copy_to_device(..., stream=) truly asynchronous: the call
# returns immediately and the DMA overlaps another chunk's kernels.
# cudaHostAlloc is expensive, so buffers are pooled; pinned memory is wired
# RAM, so total allocation is capped and failure degrades to pageable.
_PINNED_POOL_MAX_PER_KEY = 4
_PINNED_POOL_MAX_BYTES = 3_000_000_000
_pinned_pool: dict = {}
_pinned_alloc_bytes = [0]  # allocated total: pooled + in flight
_pinned_failed = [False]


def _pinned_pool_get(shape, dtype):
    """A reused page-locked staging buffer, or None when unavailable."""
    if _pinned_failed[0]:
        return None
    key = (tuple(int(s) for s in shape), np.dtype(dtype).str)
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    with _gpu_pool_lock:
        lst = _pinned_pool.get(key)
        if lst:
            return lst.pop()
        if _pinned_alloc_bytes[0] + nbytes > _PINNED_POOL_MAX_BYTES:
            return None
        _pinned_alloc_bytes[0] += nbytes
    try:
        from numba import cuda as _cuda
        return _cuda.pinned_array(shape, dtype=dtype)
    except Exception:
        with _gpu_pool_lock:
            _pinned_alloc_bytes[0] -= nbytes
        _pinned_failed[0] = True
        log.debug("[find_vectors] pinned allocation failed — using pageable H2D")
        return None


def _pinned_pool_put(arr):
    if arr is None:
        return
    key = (tuple(int(s) for s in arr.shape), np.dtype(arr.dtype).str)
    with _gpu_pool_lock:
        lst = _pinned_pool.setdefault(key, [])
        if len(lst) < _PINNED_POOL_MAX_PER_KEY:
            lst.append(arr)
        else:
            _pinned_alloc_bytes[0] -= arr.nbytes  # dropped — freed by GC


# ── Per-thread CUDA streams ───────────────────────────────────────────────────
# Each chunk task thread gets its own stream: one chunk's H2D/D2H overlaps
# another's kernels instead of everything serialising on the legacy default
# stream.  CuPy work is bound to the same stream via ExternalStream so the
# whole per-chunk pipeline stays in-order on one queue.
_thread_streams = threading.local()
# Bumped by _reset_gpu_state so threads recreate streams after cuda.close()
_gpu_context_gen = [0]
# Fixed pool of streams handed out round-robin.  Creating a stream per thread
# leaks CuPy per-stream arenas and plan caches when threads churn (each dead
# thread's arena pins VRAM until a full pool reset) — a bounded set of
# long-lived streams keeps device state constant regardless of thread count.
_GPU_STREAM_POOL_SIZE = 4
_gpu_stream_pool: list = []
_gpu_stream_rr = [0]


def _get_thread_stream():
    s = getattr(_thread_streams, "stream", None)
    if s is not None and getattr(_thread_streams, "gen", -1) == _gpu_context_gen[0]:
        return s
    from numba import cuda as _cuda
    with _gpu_pool_lock:
        if _gpu_stream_pool and _gpu_stream_pool[0][0] != _gpu_context_gen[0]:
            _gpu_stream_pool.clear()  # context was reset — streams are dead
        if len(_gpu_stream_pool) < _GPU_STREAM_POOL_SIZE:
            s = _cuda.stream()
            _gpu_stream_pool.append((_gpu_context_gen[0], s))
        else:
            _gpu_stream_rr[0] = (_gpu_stream_rr[0] + 1) % _GPU_STREAM_POOL_SIZE
            s = _gpu_stream_pool[_gpu_stream_rr[0]][1]
    _thread_streams.stream = s
    _thread_streams.gen = _gpu_context_gen[0]
    return s


def _stream_ptr(stream) -> int:
    """Raw cudaStream_t pointer of a numba stream (for CuPy ExternalStream)."""
    h = stream.handle
    return int(getattr(h, "value", h) or 0)


# ── CuPy / cuFFT availability ─────────────────────────────────────────────────
_gpu_cache_lock = threading.Lock()
_gpu_disk_fft_conj_cache: dict = {}
_cupy_state = {"checked": False, "ok": False}


def _cupy_available() -> bool:
    """True when CuPy with a working CUDA runtime is importable.  Set
    SPYDE_FV_GPU_FFT=0 to force the brute-force numba NXCORR kernels."""
    import os
    if os.environ.get("SPYDE_FV_GPU_FFT", "") in ("0", "off"):
        return False
    if not _cupy_state["checked"]:
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
            _cupy_state["ok"] = True
        except Exception:
            _cupy_state["ok"] = False
        _cupy_state["checked"] = True
    return _cupy_state["ok"]


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
            # Clamp: the % chunk-size mapping uses the first chunk's size, so
            # positions in trailing partial chunks (non-square nav shapes) can
            # land one row/col past this chunk's extent.
            iy = min(max(int(iy_local), 0), blurred.shape[0] - 1)
            ix = min(max(int(ix_local), 0), blurred.shape[1] - 1)
            return blurred[iy, ix]
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


def _subpixel_parabola(corr: np.ndarray, peaks_px: np.ndarray) -> np.ndarray:
    """3-point parabolic sub-pixel peak interpolation ON the NXCORR surface.

    The peak position stays defined by the (window-normalised) cross-correlation
    — we just locate the surface's sub-pixel vertex by fitting a parabola through
    the peak and its two neighbours in each axis. This is the standard, low-bias
    subpixel estimator (and matches the torch GPU path), unlike the previous
    window centre-of-mass which barely moved off the integer pixel.

    Returns ``(N, 2)`` float32 ``[ky_subpx, kx_subpx]``.
    """
    H, W = corr.shape
    out = peaks_px.astype(np.float32).copy()
    for i in range(len(peaks_px)):
        py, px = int(peaks_px[i, 0]), int(peaks_px[i, 1])
        if 0 < py < H - 1:
            a, b, c = float(corr[py - 1, px]), float(corr[py, px]), float(corr[py + 1, px])
            den = a - 2.0 * b + c
            if den != 0.0:
                out[i, 0] = py + min(1.0, max(-1.0, 0.5 * (a - c) / den))
        if 0 < px < W - 1:
            a, b, c = float(corr[py, px - 1]), float(corr[py, px]), float(corr[py, px + 1])
            den = a - 2.0 * b + c
            if den != 0.0:
                out[i, 1] = px + min(1.0, max(-1.0, 0.5 * (a - c) / den))
    return out


def _sample_raw_bilinear(frame: np.ndarray, ys, xs) -> np.ndarray:
    """Bilinear sample of the (experimental) frame at sub-pixel positions.

    This is the RAW disk intensity in the original image at each peak — what
    virtual imaging / orientation weighting want — as opposed to the NXCORR
    score (which is ≈1 for every well-matched disk regardless of brightness)."""
    H, W = frame.shape
    ys = np.clip(np.asarray(ys, dtype=np.float64), 0.0, H - 1.0001)
    xs = np.clip(np.asarray(xs, dtype=np.float64), 0.0, W - 1.0001)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    fy = (ys - y0).astype(np.float32)
    fx = (xs - x0).astype(np.float32)
    f = frame.astype(np.float32, copy=False)
    v = ((1 - fy) * ((1 - fx) * f[y0, x0] + fx * f[y0, x0 + 1])
         + fy * ((1 - fx) * f[y0 + 1, x0] + fx * f[y0 + 1, x0 + 1]))
    return v.astype(np.float32)


def _with_raw_intensity(frame: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Overwrite a peak set's value column (col 2) with the raw frame intensity
    at each peak — keeping the NXCORR-derived position. For the GPU/torch paths,
    which return correlation scores."""
    peaks = np.asarray(peaks, dtype=np.float32)
    if peaks.size == 0:
        return peaks.reshape(-1, 3)
    out = peaks.copy()
    out[:, 2] = _sample_raw_bilinear(frame, peaks[:, 0], peaks[:, 1])
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
    subpixel : if True, refine each peak to the parabolic (3-point) vertex of
        the NXCORR surface.  The stored value column (intensity) is always the
        RAW experimental frame intensity sampled bilinearly at the peak — not
        the correlation score — so virtual imaging / weighting see true disk
        brightness.
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
    # NB: pass s=(pH, pW) explicitly. Without it irfft2 infers the last-axis
    # length as 2*(n-1), which is WRONG when pW is odd (next_fast_len can return
    # an odd value, e.g. 112+2*5 -> 125 for sped_ag) — it reconstructs width
    # pW-1 and the correlation is computed with the wrong period, shifting every
    # peak in X by a fraction of a pixel. (Even widths were unaffected, which is
    # why this only showed on real detector sizes.)
    xcorr = irfft2(rfft2(buf) * _disk_fft.conj(), s=(pH, pW))[:H, :W].astype(np.float32)

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

    # Position from the NXCORR surface (parabolic sub-pixel vertex when enabled);
    # INTENSITY from the raw experimental frame at that position (not the corr
    # score, which is ≈1 for every matched disk).
    if subpixel:
        pos = _subpixel_parabola(raw_corr, peaks_px)
    else:
        pos = peaks_px.astype(np.float32)
    intens = _sample_raw_bilinear(frame, pos[:, 0], pos[:, 1])
    peaks = np.column_stack([pos, intens]).astype(np.float32)
    return corr_map, raw_corr, peaks


def _dilate_mask(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary-dilate a 2D mask by ``r`` px (used to swallow the beam-stop rim)."""
    if mask is None or r <= 0:
        return mask
    return maximum_filter(mask.astype(np.uint8), size=2 * int(r) + 1) > 0


def detect_beamstop(mean_pattern: np.ndarray, *, frac: float = 0.15,
                    dilate: int = 5) -> Optional[np.ndarray]:
    """Auto-detect a physical beam stop from a scan-mean / navigator pattern.

    A beam stop blocks electrons, so in the time-averaged pattern it is a stable,
    connected **low-intensity** region.  We threshold at ``frac`` of the mean and
    keep the result, then **dilate by ``dilate`` px** — crucially, the brightest
    feature in a beam-stopped frame is the diffraction halo (the "rim") hugging
    the stop edge, NOT the occluded core, so the mask must extend a few px past
    the geometric stop to keep peak finders off the rim (benchmarked: ~5 px
    removes the rim with no loss of real spots).

    Returns a (H, W) bool mask, or ``None`` if no plausible stop is found
    (a featureless / no-stop pattern thresholds to almost nothing or almost
    everything).
    """
    m = np.asarray(mean_pattern, dtype=np.float64)
    if m.ndim != 2 or m.size == 0:
        return None
    mean_val = float(m.mean())
    if mean_val <= 0:
        return None
    mask = m < (mean_val * frac)
    f = mask.mean()
    # Reject degenerate masks: nothing (no stop) or almost everything (a blank
    # / near-empty pattern, where "< 0.15*mean" catches the whole background).
    if f < 0.001 or f > 0.45:
        return None
    return _dilate_mask(mask, dilate)


def _auto_beamstop_from_signal(signal, nav_dim: int, *, max_samples: int = 64,
                               dilate: int = 5) -> Optional[np.ndarray]:
    """Detect the beam stop from a small sample of patterns (memory-safe + FAST).

    A physical beam stop is STATIC across the scan, so a few dozen frames
    reproduce it cleanly. The previous version read ~400 frames one-by-one
    (`flat[i].compute()` per frame); on a chunked lazy MRC each such read decodes
    the WHOLE enclosing chunk (e.g. 134 MB for a 32x32x256x256 chunk) just to
    pull one frame — so it re-read many GB and was painfully slow.

    Instead we take ONE contiguous nav block sized to a single storage chunk and
    mean it in a single `.compute()` (one chunk read). The stop is everywhere, so
    a corner block is representative.
    """
    raw = signal.data
    nav_shape = raw.shape[:nav_dim]
    sig_shape = raw.shape[nav_dim:]
    if len(sig_shape) != 2:
        return None
    n_nav = int(np.prod(nav_shape))
    if n_nav == 0:
        return None
    flat = raw.reshape((n_nav,) + tuple(sig_shape))

    # A single contiguous slice → one chunk read for lazy data. Align to the
    # stored chunking when available so we touch exactly one chunk.
    take = min(max_samples, n_nav)
    try:
        chunks0 = getattr(flat, "chunks", None)
        if chunks0:
            take = min(take, int(chunks0[0][0]))   # first nav-chunk length
    except Exception:
        pass
    take = max(1, take)
    block = flat[:take]                              # (take, ky, kx) view
    if hasattr(block, "compute"):
        # local threaded scheduler — not the distributed cluster (shares the
        # navigator's CachedDaskArray; its cancel_surrounding would kill the
        # read). See CLAUDE.md Live-Display Core Patterns.
        try:
            block = block.compute(scheduler="threads")
        except Exception:
            block = np.asarray(block.compute())
    block = np.asarray(block, dtype=np.float64)
    if block.size == 0:
        return None
    mean_pattern = block.mean(axis=0)
    return detect_beamstop(mean_pattern, dilate=dilate)


# ── Difference-of-Gaussians (DoG) blob detector ────────────────────────────────
# Cache of separable 1D Gaussian kernels keyed by rounded sigma.
@functools.lru_cache(maxsize=64)
def _gauss_kernel_1d(sigma: float) -> np.ndarray:
    """Normalised 1D Gaussian kernel truncated at 3σ (matches scipy default)."""
    s = float(sigma)
    if s <= 0:
        return np.array([1.0], dtype=np.float32)
    r = max(1, int(round(3.0 * s)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * s * s))
    k /= k.sum()
    return k.astype(np.float32)


def _find_vectors_single_frame_dog(
    frame: np.ndarray,
    sigma1: float,
    sigma2: float,
    threshold: float,
    min_distance: int,
    *,
    subpixel: bool = True,
    beamstop_mask: Optional[np.ndarray] = None,
    bs_dilate: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Difference-of-Gaussians band-pass blob detector for small (2-3 px) spots.

    ``response = G(σ₁)*I − G(σ₂)*I`` (σ₁<σ₂) is the band-pass matched to a small
    Gaussian spot: it removes both pixel noise (≤σ₁) and the smooth diffuse
    background / beam tails (≥σ₂) in one separable real-space pass, so a faint
    spot stands out without a radius-matched template (the failure mode of the
    NXCORR disk on this kind of data).  Both Gaussians are real-space separable
    convolutions — no FFT.

    Beam stop: masked pixels are **background-filled** with the σ₂ blur (NOT
    zero-filled).  Zeroing makes a hard signal→0 step at the mask edge and the
    band-pass fires on it; filling with the smooth background removes the step.
    Detections inside the (optionally further-dilated) mask are excluded.

    Threshold is an **absolute band-pass SNR**: ``response / (1.4826·MAD)`` of the
    response, so it is intensity-independent (like the NXCORR [-1,1] score) and
    does not collapse on blank frames the way a per-frame-max-relative threshold
    would.  Typical values 3-8.

    Returns ``(corr_map, response_snr, peaks)`` mirroring
    :func:`_find_vectors_single_frame` so all overlay / batch code is unchanged.
    ``peaks`` is ``(N, 3)`` float32 ``[ky, kx, raw_intensity]``.
    """
    from scipy.ndimage import correlate1d

    f = np.asarray(frame, dtype=np.float32)
    H, W = f.shape

    excl = None
    if beamstop_mask is not None and beamstop_mask.any():
        excl = _dilate_mask(beamstop_mask, bs_dilate) if bs_dilate else beamstop_mask
        # background-fill the stop with the wide blur so no step is introduced
        bg_fill = gaussian_filter(f, sigma2)
        f = f.copy()
        f[excl] = bg_fill[excl]

    k1 = _gauss_kernel_1d(sigma1)
    k2 = _gauss_kernel_1d(sigma2)
    # separable real-space blur (reflect boundary, like the batch nav blur)
    g1 = correlate1d(correlate1d(f, k1, axis=0, mode="reflect"), k1, axis=1, mode="reflect")
    g2 = correlate1d(correlate1d(f, k2, axis=0, mode="reflect"), k2, axis=1, mode="reflect")
    resp = g1 - g2

    # absolute SNR normalisation via median absolute deviation (robust to spots)
    # Robust SNR scale from UNMASKED pixels only: the bright beam-stop rim, even
    # after background-fill, leaves a strong band-pass response in a thin ring at
    # the mask edge; including it would inflate the MAD and bury faint real spots.
    stat_src = resp[~excl] if excl is not None else resp.ravel()
    med = float(np.median(stat_src))
    mad = float(np.median(np.abs(stat_src - med)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        # near-zero MAD: a (near-)flat band-pass (low-texture frame). Fall back
        # to the std, then to the response peak, so the SNR stays finite and a
        # lone spot on a flat field is still detectable rather than div-by-zero.
        scale = float(stat_src.std())
    if scale <= 1e-12:
        scale = float(np.abs(stat_src).max()) or 1.0
    snr = ((resp - med) / scale).astype(np.float32)
    if excl is not None:
        snr[excl] = 0.0

    corr_map = np.where(snr >= threshold, snr, 0.0).astype(np.float32)

    if not (snr >= threshold).any():
        return corr_map, snr, np.zeros((0, 3), dtype=np.float32)

    min_d = int(min_distance)
    local_max = maximum_filter(snr, size=2 * min_d + 1)
    peaks_mask = (snr == local_max) & (snr >= threshold)
    if excl is not None:
        peaks_mask &= ~excl

    peaks_px = np.argwhere(peaks_mask)
    if len(peaks_px) > 1:
        inten = snr[peaks_px[:, 0], peaks_px[:, 1]]
        order = np.argsort(-inten)
        peaks_px = peaks_px[order]
        kept = np.ones(len(peaks_px), dtype=bool)
        min_d2 = min_d * min_d
        for i in range(len(peaks_px)):
            if not kept[i]:
                continue
            dy = peaks_px[i + 1:, 0] - peaks_px[i, 0]
            dx = peaks_px[i + 1:, 1] - peaks_px[i, 1]
            kept[i + 1:][(dy * dy + dx * dx) <= min_d2] = False
        peaks_px = peaks_px[kept]

    if len(peaks_px) == 0:
        return corr_map, snr, np.zeros((0, 3), dtype=np.float32)

    if subpixel:
        pos = _subpixel_parabola(snr, peaks_px)
    else:
        pos = peaks_px.astype(np.float32)
    # intensity column = RAW experimental frame brightness (for virtual imaging /
    # orientation weighting), sampled at the sub-pixel position — same convention
    # as the NXCORR path.
    intens = _sample_raw_bilinear(np.asarray(frame, dtype=np.float32), pos[:, 0], pos[:, 1])
    peaks = np.column_stack([pos, intens]).astype(np.float32)
    return corr_map, snr, peaks


# Method names accepted across the find-vectors stack.
METHOD_NXCORR = "nxcorr"
METHOD_DOG = "dog"
DEFAULT_DOG_SIGMA1 = 0.8
DEFAULT_DOG_SIGMA2 = 2.0
# Absolute band-pass SNR (response / 1.4826·MAD).  ~10 balances recall/precision
# on real small-spot data (benchmarked on the 3 nm DESEMCam scan); raise toward
# 15 for cleaner / fewer peaks, lower toward 6 for more recall.
DEFAULT_DOG_THRESHOLD = 10.0


def _find_peaks_single_frame(frame, params, *, beamstop_mask=None,
                             _disk_fft=None, _disk_stats=None,
                             with_response=False):
    """Dispatch a single frame to the configured detector, returning peaks
    ``(N,3)`` ``[ky, kx, intensity]``.  ``params`` is the find-vectors param
    dict (``method`` selects ``nxcorr`` or ``dog``).

    ``with_response=True`` returns ``(peaks, response_map)`` where response_map is
    the detector's transformed image — the DoG band-pass SNR or the NXCORR
    correlation surface — for the "show transform" preview toggle."""
    method = str(params.get("method", METHOD_NXCORR)).lower()
    if method == METHOD_DOG:
        out = _find_vectors_single_frame_dog(
            frame,
            float(params.get("dog_sigma1", DEFAULT_DOG_SIGMA1)),
            float(params.get("dog_sigma2", DEFAULT_DOG_SIGMA2)),
            float(params.get("threshold", DEFAULT_DOG_THRESHOLD)),
            int(params.get("min_distance", 3)),
            subpixel=bool(params.get("subpixel", True)),
            beamstop_mask=beamstop_mask,
        )
    else:
        out = _find_vectors_single_frame(
            frame,
            int(params.get("kernel_radius", 5)),
            float(params.get("threshold", 0.5)),
            int(params.get("min_distance", 5)),
            subpixel=bool(params.get("subpixel", True)),
            beamstop_mask=beamstop_mask,
            _disk_fft=_disk_fft, _disk_stats=_disk_stats,
        )
    # out = (corr_map_thresholded, raw_response, peaks)
    return (out[2], out[1]) if with_response else out[2]


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
    # Keep the ghost-padded chunk side <= 255 so the flattened nav grid
    # (side + 2*depth)^2 stays under the CUDA gridDim.z limit of 65535.
    max_side = 255 - 2 * depth
    # Floor the core at 3*depth: a tiny core under a big halo multiplies IO
    # and blur work by ((core+2*depth)/core)^2 — e.g. 512x512 patterns with
    # sigma=1 gave core=4, a 6.2x overhead and 4096 chunks per 256x256 scan.
    # The floor caps overlap overhead at ~2.8x; pushing it lower needs bigger
    # ghost blocks, and per-task RAM (ghost + blur copy, x n_threads) is what
    # drives workers into spill on RAM-limited machines.
    core = max(max_padded - 2 * depth, 3 * depth)
    return max(depth + 1, min(core, max_side))


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
        # float32 accumulator — a 0.0 literal would type as float64, which
        # Pascal-class GPUs execute at 1/32 the float32 rate
        xcorr = _nb_f32(0.0)
        for dr in range(disk_h):
            for dc in range(disk_w):
                py = out_y + kr_win + dr - kr
                px = out_x + kr_win + dc - kr
                val = frames_padded[n, py, px]
                xcorr += disk[dr, dc] * val

        # ── Window statistics: loop over (2*kr_win+1)^2 neighbourhood ────────
        win_size = 2 * kr_win + 1
        sum1 = _nb_f32(0.0)
        sum2 = _nb_f32(0.0)
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

        Replaces _nxcorr_kernel + the host-side reflect-pad: out-of-bounds taps
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
            py = out_y + dr - kr_win
            if py < 0:
                py = -py
            elif py >= H:
                py = 2 * H - py - 2
            if py < 0:
                py = 0
            elif py >= H:
                py = H - 1
            for dc in range(win_size):
                px = out_x + dc - kr_win
                if px < 0:
                    px = -px
                elif px >= W:
                    px = 2 * W - px - 2
                if px < 0:
                    px = 0
                elif px >= W:
                    px = W - 1
                v = frames[n, py, px]
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
        # Host frames for raw-intensity sampling (the GPU kernel stores the corr
        # score; NMS below uses it, then we overwrite the value with the raw
        # experimental intensity at each peak — matching the CPU path).
        host_frames = blurred_block.reshape(-1, H, W)

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
                # Deterministic order: the GPU subpixel kernel fills slots via
                # atomics in arbitrary order, so tie-break equal scores by
                # position to keep results reproducible run-to-run.
                order = np.lexsort(
                    (frame_peaks[:, 1], frame_peaks[:, 0], -frame_peaks[:, 2])
                )
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

            result_flat[i] = _with_raw_intensity(host_frames[i], frame_peaks)

        result = result_flat.reshape(nav_shape) if len(nav_shape) > 0 else result_flat
        return result

    except Exception:
        return None


def _gpu_task_allowed() -> bool:
    """
    Decide whether the current chunk task may use the GPU.

    With opportunistic GPU use, every dask worker funnels its chunks through
    the single device, the kernels serialise, and the whole cluster collapses
    to GPU throughput while the CPU cores idle.  Default policy: exactly ONE
    worker (LocalCluster worker name "1") uses the GPU; all others run the
    CPU path in parallel, so total throughput is GPU rate + CPU rate.

    Override with the SPYDE_FV_GPU environment variable:
        "one" (default) — GPU on worker "1" only
        "<N>" (integer) — GPU on workers "1".."N" (overlaps one chunk's
                          H2D/pack stages with another's kernels)
        "all"           — every worker may use the GPU (single-GPU contention)
        "off"           — CPU everywhere
    Outside a distributed worker (threaded scheduler, tests) the GPU is allowed.
    """
    import os
    if _gpu_warm_failures[0] >= _GPU_MAX_WARM_FAILURES and not _gpu_warmed[0]:
        return False  # GPU disabled in this process after warmup failures
    mode = os.environ.get("SPYDE_FV_GPU", "one").lower()
    if mode == "off":
        return False
    if mode == "all":
        return True
    try:
        n_gpu_workers = max(0, int(mode))
    except ValueError:
        n_gpu_workers = 1  # "one" or anything unparseable
    try:
        from distributed import get_worker
        name = str(get_worker().name)
    except Exception:
        return True  # not running on a dask worker
    try:
        return 1 <= int(name) <= n_gpu_workers
    except ValueError:
        return name == "1"  # non-integer worker names: single GPU worker


def _nav_blur_trim(ghost_block, depth_px, nav_dim, sigma):
    """Nav-space Gaussian blur (sigma over the 2 spatial nav dims, 0 elsewhere)
    of a ghost-padded block, then trim the ghost zones.  Shared by the NXCORR CPU
    fallback and the DoG path."""
    from scipy.ndimage import gaussian_filter as _gf
    sigma_tuple = tuple([0.0] * (nav_dim - 2) + [sigma, sigma, 0.0, 0.0])
    blurred = _gf(np.asarray(ghost_block, dtype=np.float32), sigma=sigma_tuple)
    trim = [slice(None)] * ghost_block.ndim
    for d in (nav_dim - 2, nav_dim - 1):
        s = blurred.shape[d]
        lo = depth_px if depth_px < s else 0
        hi = s - depth_px if depth_px < s else s
        trim[d] = slice(lo, hi)
    return blurred[tuple(trim)]


def _dog_block(b4d, sigma1, sigma2, threshold, min_dist, subpixel, beamstop_mask):
    """Run the DoG detector on a (ny, nx, KY, KX) block → NaN-padded
    (ny, nx, MAX_PEAKS, 3).  Batches on the torch GPU when available; otherwise
    the numpy per-frame core."""
    out = np.full((b4d.shape[0], b4d.shape[1], MAX_PEAKS, 3), np.nan, dtype=np.float32)
    flat = b4d.reshape(-1, b4d.shape[2], b4d.shape[3])
    peaks_list = None
    try:
        from spyde.actions.find_vectors_torch import (
            torch_gpu_device, find_vectors_dog_torch_batch)
        if torch_gpu_device() is not None:
            peaks_list = find_vectors_dog_torch_batch(
                flat, sigma1, sigma2, threshold, min_dist,
                subpixel=subpixel, beamstop_mask=beamstop_mask)
            # GPU returns the SNR in col 2; replace with raw frame intensity.
            peaks_list = [_with_raw_intensity(flat[i], p)
                          for i, p in enumerate(peaks_list)]
    except Exception as _e:
        log.warning("[find_vectors] torch DoG GPU path failed (%s); CPU per-frame", _e)
        peaks_list = None
    if peaks_list is None:
        peaks_list = [
            _find_vectors_single_frame_dog(
                frame, sigma1, sigma2, threshold, min_dist,
                subpixel=subpixel, beamstop_mask=beamstop_mask)[2]
            for frame in flat
        ]
    for i, peaks in enumerate(peaks_list):
        iy, ix = divmod(i, b4d.shape[1])
        n = min(len(peaks), MAX_PEAKS)
        if n > 0:
            out[iy, ix, :n, :] = peaks[:n]
    return out


def _find_vectors_chunk_dog(
    ghost_block, depth_px, nav_dim, sigma,
    sigma1, sigma2, threshold, min_dist, subpixel, beamstop_mask,
):
    """DoG variant of _find_vectors_chunk: nav-blur + trim, then per-frame DoG
    band-pass (GPU-batched when torch CUDA is present).  Same output structure
    as the NXCORR chunk fn."""
    t_start = time.perf_counter()
    blurred = _nav_blur_trim(ghost_block, depth_px, nav_dim, sigma)
    nav_shape = blurred.shape[:nav_dim]
    ny, nx = nav_shape[-2:]
    if nav_dim == 2:
        result = _dog_block(blurred, sigma1, sigma2, threshold, min_dist,
                            subpixel, beamstop_mask)
        core_shape = result.shape[:2]
    else:
        n_lead = nav_shape[0]
        out = np.full((n_lead, ny, nx, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        for t in range(n_lead):
            out[t] = _dog_block(blurred[t], sigma1, sigma2, threshold, min_dist,
                                subpixel, beamstop_mask)
        result = out
        core_shape = (n_lead, ny, nx)
    log.debug("[find_vectors] DoG chunk core=%s total=%.0fms",
              tuple(int(s) for s in core_shape),
              (time.perf_counter() - t_start) * 1e3)
    return result


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
    method: str = METHOD_NXCORR,
    dog_sigma1: float = DEFAULT_DOG_SIGMA1,
    dog_sigma2: float = DEFAULT_DOG_SIGMA2,
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

    Falls back to scipy + CPU per-frame loop if CUDA is unavailable or this
    worker is not the designated GPU worker (see _gpu_task_allowed /
    SPYDE_FV_GPU).  Both paths print one timing line per chunk.

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
    # Zero-size blocks (dask meta inference calls the chunk fn on empty
    # arrays in the CLIENT process) — return an empty result of the right
    # structure; an empty grid is an invalid CUDA launch.
    if ghost_block.size == 0:
        if nav_dim == 2:
            return np.empty((0, 0, MAX_PEAKS, 3), dtype=np.float32)
        return np.empty((0, 0, 0, MAX_PEAKS, 3), dtype=np.float32)

    # ── DoG band-pass detector ────────────────────────────────────────────────
    # The numba-CUDA NXCORR kernels below are disk-matched-filter specific, so
    # DoG takes its own route: nav-blur + trim (CPU), then the per-frame
    # band-pass batched on the torch GPU (Pascal CUDA) when available, numpy
    # otherwise.  Both share the same nav-blur as NXCORR.
    if str(method).lower() == METHOD_DOG:
        return _find_vectors_chunk_dog(
            ghost_block, depth_px, nav_dim, sigma,
            dog_sigma1, dog_sigma2, threshold, min_dist, subpixel,
            beamstop_mask,
        )

    # ── Try GPU path ──────────────────────────────────────────────────────────
    if _GPU_KERNELS_AVAILABLE and _gpu_task_allowed():
        try:
            from numba import cuda as _cuda
            cuda_ok = _cuda.is_available()
        except Exception:
            cuda_ok = False
        if cuda_ok:
            for _attempt in range(2):
                try:
                    return _find_vectors_chunk_gpu(
                        ghost_block, depth_px, nav_dim, sigma,
                        kernel_r, threshold, min_dist, subpixel,
                        beamstop_mask, disk_stats,
                    )
                except Exception as exc:
                    try:
                        from distributed import get_worker
                        _wname = f"worker {get_worker().name}"
                    except Exception:
                        _wname = "main process"
                    if _attempt == 0:
                        # The first CUDA touch in a fresh dask worker thread
                        # can fail transiently (driver init / cache-load
                        # race) — settles immediately after.  Retry once.
                        log.warning("[find_vectors] GPU attempt failed on %s (%r) — retrying", _wname, exc)
                        time.sleep(0.5)
                    else:
                        # A silent fallback hides real throughput problems.
                        import traceback as _tb
                        log.warning("[find_vectors] GPU path failed on %s (%r) — falling back to CPU", _wname, exc)
                        _tb.print_exc()

    # ── CPU fallback ──────────────────────────────────────────────────────────
    from scipy.ndimage import gaussian_filter as _gf

    t_start = time.perf_counter()
    sigma_tuple = tuple([0.0] * (nav_dim - 2) + [sigma, sigma, 0.0, 0.0])
    # asarray: no copy when the block is already float32
    blurred = _gf(np.asarray(ghost_block, dtype=np.float32), sigma=sigma_tuple)
    blur_ms = (time.perf_counter() - t_start) * 1e3

    # Trim ghost zones — only the two spatial nav dims are ghosted
    # (map_overlap depth is 0 on any leading time axis).
    trim = [slice(None)] * ghost_block.ndim
    for d in (nav_dim - 2, nav_dim - 1):
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
        # Fast path: batch the whole block's NXCORR on the torch GPU (Apple-MPS on
        # a MacBook, CUDA otherwise). Reached when the numba.cuda kernels above
        # aren't available — exactly the Mac case the user wants accelerated. The
        # surface + peak set match the numpy reference to float precision (the GPU
        # only accelerates the correlation; peak detection stays identical).
        peaks_list = None
        try:
            from spyde.actions.find_vectors_torch import (
                torch_gpu_device, find_vectors_torch_batch)
            if torch_gpu_device() is not None:
                peaks_list = find_vectors_torch_batch(
                    flat, kernel_r, threshold, min_dist,
                    subpixel=subpixel, beamstop_mask=beamstop_mask)
                # torch returns NXCORR positions + correlation score; replace the
                # value column with the raw frame intensity at each peak.
                peaks_list = [_with_raw_intensity(flat[i], p)
                              for i, p in enumerate(peaks_list)]
        except Exception as _e:
            log.warning("[find_vectors] torch GPU path failed (%s); CPU per-frame", _e)
            peaks_list = None
        if peaks_list is None:
            peaks_list = [
                _find_vectors_single_frame(
                    frame, kernel_r, threshold, min_dist,
                    subpixel=subpixel, beamstop_mask=beamstop_mask,
                    _disk_fft=disk_fft, _disk_stats=disk_stats)[2]
                for frame in flat
            ]
        for i, peaks in enumerate(peaks_list):
            iy, ix = divmod(i, b4d.shape[1])
            n = min(len(peaks), MAX_PEAKS)
            if n > 0:
                out[iy, ix, :n, :] = peaks[:n]
        return out

    t_find = time.perf_counter()
    if nav_dim == 2:
        result = _cpu_block(blurred)
        core_shape = result.shape[:2]
    else:
        n_lead = nav_shape[0]
        out = np.full((n_lead, ny, nx, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        for t in range(n_lead):
            out[t] = _cpu_block(blurred[t])
        result = out
        core_shape = (n_lead, ny, nx)
    find_ms = (time.perf_counter() - t_find) * 1e3
    total_ms = (time.perf_counter() - t_start) * 1e3
    log.debug(
        f"[find_vectors] CPU chunk core={tuple(int(s) for s in core_shape)} "
        f"sig=({ghost_block.shape[-2]},{ghost_block.shape[-1]}) "
        f"blur={blur_ms:.0f}ms find={find_ms:.0f}ms total={total_ms:.0f}ms"
    )
    return result


# First-use serialisation: concurrent first-time kernel cache-loading /
# compilation and CUDA context initialisation from multiple worker threads
# has produced corrupt launches (CUDA_ERROR_INVALID_VALUE) — run the first
# chunk in each process alone, then go fully concurrent.  The first CUDA
# touch in a fresh dask worker can also fail transiently and leave numba's
# context state poisoned, so warmup resets it and only counts a SUCCESSFUL
# chunk as warmed; after repeated failures the GPU is disabled per-process.
_gpu_warmup_lock = threading.Lock()
_gpu_warmed = [False]
_gpu_warm_failures = [0]
_GPU_MAX_WARM_FAILURES = 3

# Optional serialisation of GPU chunk execution across this process's task
# threads.  With per-thread streams the default is full concurrency (the
# historical "concurrent launch" failures traced back to dask meta-inference
# launching empty grids, not to numba thread-safety); SPYDE_FV_GPU_SERIAL=1
# restores the old whole-chunk lock as an escape hatch.
_gpu_exec_lock = threading.Lock()

# Per-thread streams allow chunk overlap, but unbounded concurrency thrashes
# VRAM (each in-flight 512^2 chunk holds ~1 GB of buffers, FFT temporaries
# and per-thread cuFFT plan workspaces).  The semaphore bounds how many
# chunks may occupy the device section at once; the CPU pack stage runs
# outside it.  Tune with SPYDE_FV_GPU_CONC (default 2).
_gpu_slots_state: dict = {"sem": None, "n": None}


def _gpu_slots():
    import os
    try:
        n = max(1, int(os.environ.get("SPYDE_FV_GPU_CONC", "2")))
    except ValueError:
        n = 2
    with _gpu_cache_lock:
        if _gpu_slots_state["sem"] is None or _gpu_slots_state["n"] != n:
            _gpu_slots_state["sem"] = threading.BoundedSemaphore(n)
            _gpu_slots_state["n"] = n
        return _gpu_slots_state["sem"]


def _gpu_serial_mode() -> bool:
    import os
    return os.environ.get("SPYDE_FV_GPU_SERIAL", "") not in ("", "0", "off")


import contextlib


@contextlib.contextmanager
def _interprocess_warmup_lock():
    """
    Cross-process file lock held while a process compiles / cache-loads the
    CUDA kernels (its first chunk).  numba's on-disk kernel cache is not safe
    against concurrent writers on Windows: several worker processes compiling
    a cold cache simultaneously produce torn reads and corrupt launches
    (CUDA_ERROR_INVALID_VALUE).  Serialising first chunks across processes
    makes the cache single-writer; steady-state chunks never touch this.
    """
    import os
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "spyde_fv_cuda_warmup.lock")
    fd = None
    deadline = time.time() + 300.0
    while fd is None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Remove stale locks (e.g. a killed worker)
            try:
                if time.time() - os.path.getmtime(path) > 120.0:
                    os.remove(path)
                    continue
            except OSError as e:
                log.debug("stale-lock check on %s failed: %s", path, e)
            if time.time() > deadline:
                break  # give up on locking rather than deadlock
            time.sleep(0.1)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
                os.remove(path)
            except OSError as e:
                log.debug("releasing lock file %s failed: %s", path, e)


def _reset_gpu_state():
    """Tear down numba's CUDA context state and our device-side caches so the
    next attempt starts from a clean slate (cached handles die with the
    context)."""
    try:
        from numba import cuda as _cuda
        _cuda.close()
    except Exception as e:
        log.debug("numba CUDA context teardown failed: %s", e)
    with _gpu_pool_lock:
        _gpu_buffer_pool.clear()
        _gpu_pool_bytes[0] = 0
    _gpu_disk_cache.clear()
    with _gpu_cache_lock:
        _gpu_disk_fft_conj_cache.clear()
    _gpu_pool_max_bytes[0] = None
    with _gpu_pool_lock:
        _pinned_pool.clear()
        _pinned_alloc_bytes[0] = 0
    # Invalidate per-thread streams created on the dead context
    _gpu_context_gen[0] += 1


def _gpu_warmup_probe():
    """
    Compile and exercise every CUDA kernel on a tiny synthetic chunk, for
    both the float32 and the uint16 (device-convert) paths.

    Run under the warmup locks so (a) a cursed first context costs a few
    milliseconds instead of a real chunk, and (b) no kernel specialisation
    is ever first-compiled outside the locks by concurrent task threads.
    """
    kernel_r = 2
    d = _make_disk(kernel_r)
    n = d.shape[0] * d.shape[1]
    t_mean = float(d.mean())
    t_std = float(np.sqrt(np.sum((d - t_mean) ** 2) / n))
    base = np.zeros((6, 6, 16, 16), dtype=np.float32)
    base[:, :, 8, 8] = 100.0
    for block in (base, (base * 40).astype(np.uint16)):
        _find_vectors_chunk_gpu_impl(
            block, 1, 2, 1.0, kernel_r, 0.5, 2, True, None, (n, t_mean, t_std),
        )


def _find_vectors_chunk_gpu(*args, **kwargs) -> np.ndarray:
    """Thin wrapper around _find_vectors_chunk_gpu_impl that serialises the
    first use per process (kernel compile + context init are not safe to
    race from multiple threads) and recovers from transient init failures."""
    if _gpu_warmed[0]:
        if _gpu_serial_mode():
            with _gpu_exec_lock:
                return _find_vectors_chunk_gpu_impl(*args, **kwargs)
        return _find_vectors_chunk_gpu_impl(*args, **kwargs)

    with _gpu_warmup_lock:
        if not _gpu_warmed[0]:
            if _gpu_warm_failures[0] >= _GPU_MAX_WARM_FAILURES:
                raise RuntimeError(
                    "GPU disabled in this process after repeated warmup failures"
                )
            from numba import cuda as _cuda
            # The first CUDA context in a fresh dask worker can come up
            # broken (kernel launches fail with INVALID_VALUE) and only
            # works after a full cuda.close() + re-init.  Retry the cheap
            # probe a few times with resets in between.
            last_exc = None
            for attempt in range(3):
                try:
                    with _interprocess_warmup_lock():
                        try:
                            _cuda.current_context()
                        except Exception:
                            _reset_gpu_state()
                            time.sleep(0.5)
                            _cuda.current_context()
                        with _gpu_exec_lock:
                            _gpu_warmup_probe()
                    if attempt:
                        log.debug("[find_vectors] GPU warmup succeeded on attempt %d", attempt + 1)
                    _gpu_warmed[0] = True
                    break
                except Exception as exc:
                    last_exc = exc
                    _reset_gpu_state()
                    time.sleep(1.0)
            else:
                _gpu_warm_failures[0] += 1
                raise last_exc

    if _gpu_serial_mode():
        with _gpu_exec_lock:
            return _find_vectors_chunk_gpu_impl(*args, **kwargs)
    return _find_vectors_chunk_gpu_impl(*args, **kwargs)


def _find_vectors_chunk_gpu_impl(
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
    Device-resident GPU implementation of _find_vectors_chunk.

    Single H2D transfer of the ghost-padded block, then entirely on-device:
      1. Separable Gaussian blur in nav-space (two 1D kernel passes)
      2. Trim-compaction of the core into contiguous (N, KY, KX) frames
      3. Beamstop fill (per-frame unmasked-mean reduction + fill kernel)
      4. Per-frame std reduction for the NXCORR denominator floor
      5. NXCORR with reflected indexing (no host reflect-pad round trip)
         + local-max + subpixel kernels
      6. D2H only the tiny stats vectors and the sparse padded peak result

    For 5D, processes each t-slice sequentially on GPU (same device context).
    Prints one timing line per chunk so GPU vs CPU throughput can be compared.
    """
    import os
    from numba import cuda as _cuda

    t_start = time.perf_counter()
    timings = {"stage": 0.0, "h2d": 0.0, "blur": 0.0, "stats": 0.0,
               "nxcorr": 0.0, "peaks": 0.0, "d2h": 0.0, "pack": 0.0}

    # Per-stage device syncs make the stage timings accurate but act as
    # barriers — they prevent the GPU worker's threads from overlapping their
    # chunks.  Off by default; SPYDE_FV_TIMING=1 enables.
    stage_timing = os.environ.get("SPYDE_FV_TIMING", "") not in ("", "0", "off")

    # Per-thread stream: chunk tasks on different worker threads overlap
    # their H2D/D2H and kernels instead of serialising on the legacy default
    # stream.  CuPy work joins the same stream via ExternalStream.
    stream = _get_thread_stream()
    use_fft = _cupy_available()
    nx_path = ["fft" if use_fft else "numba"]

    def _sync_if_timing():
        if stage_timing:
            stream.synchronize()

    # Native integer data is uploaded as-is and cast to float32 on device:
    # half the H2D bytes for 16-bit detectors and no host astype pass.
    # float32 passes straight through (ascontiguousarray is a no-op copy-wise);
    # anything else (float64, big-endian) converts on the host.
    if ghost_block.dtype in _GPU_NATIVE_DTYPES:
        block_host = np.ascontiguousarray(ghost_block)
        convert_on_device = True
    else:
        block_host = np.ascontiguousarray(ghost_block, dtype=np.float32)
        convert_on_device = False

    nav_shape_ghost = block_host.shape[:nav_dim]
    KY, KX = block_host.shape[-2], block_host.shape[-1]
    HW = KY * KX

    # ── Pre-compute 1D Gaussian kernel (CPU, tiny) ────────────────────────────
    if sigma > 0:
        radius = int(np.ceil(3 * sigma))
        xs = np.arange(-radius, radius + 1, dtype=np.float32)
        kern_cpu = np.exp(-0.5 * (xs / sigma) ** 2).astype(np.float32)
        kern_cpu /= kern_cpu.sum()
    else:
        radius = 0
        kern_cpu = np.ones(1, dtype=np.float32)
    kern_d = _cuda.to_device(kern_cpu, stream=stream)

    # ── Upload disk kernel once per kernel_r (lock: shared across threads) ───
    with _gpu_cache_lock:
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
    iH, iW = np.int32(KY), np.int32(KX)

    # Beamstop mask: uploaded once per chunk.  A 1x1 dummy keeps the reduction
    # kernel signature uniform when no mask is active (use_mask=0 short-circuits
    # before indexing it).
    have_mask = beamstop_mask is not None and beamstop_mask.any()
    if have_mask:
        mask_d = _cuda.to_device(beamstop_mask.astype(np.uint8))
        n_unmasked = max(1, HW - int(beamstop_mask.sum()))
    else:
        mask_d = _cuda.to_device(np.zeros((1, 1), dtype=np.uint8))
        n_unmasked = HW

    import warnings as _w

    def _device_section(block4d_cpu):
        """Device-resident pipeline for one (ny_ghost, nx_ghost, KY, KX) block.
        Runs under the GPU slot semaphore; returns host-side results only."""
        NY_g, NX_g = block4d_cpu.shape[0], block4d_cpu.shape[1]

        bx, by = 16, 16
        # x/y span signal pixels, z spans nav positions (z <= 65535)
        grid_sig_nav = (
            int(np.ceil(KX / bx)),
            int(np.ceil(KY / by)),
            NY_g * NX_g,
        )

        # H2D — the only full-block transfer in the whole pipeline.
        # Integer data goes up in its native width and is cast on device.
        # All device buffers come from the pool (no cudaMalloc per chunk).
        raw_d = None
        tmp_d = None
        t0 = time.perf_counter()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            if convert_on_device:
                raw_d = _gpu_pool_get(block4d_cpu.shape, block4d_cpu.dtype)
                raw_d.copy_to_device(block4d_cpu, stream=stream)
                src_d = _gpu_pool_get(block4d_cpu.shape, np.float32)
                _convert_f32_kernel[grid_sig_nav, (bx, by, 1), stream](raw_d, src_d)
            else:
                src_d = _gpu_pool_get(block4d_cpu.shape, np.float32)
                src_d.copy_to_device(block4d_cpu, stream=stream)
        _sync_if_timing()
        timings["h2d"] += time.perf_counter() - t0

        # ── Gaussian blur: two separable 1D passes ────────────────────────────
        t0 = time.perf_counter()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            if sigma > 0:
                tmp_d = _gpu_pool_get(block4d_cpu.shape, np.float32)
                grid_blur = grid_sig_nav
                block_blur = (bx, by, 1)
                _gaussian_blur_1d_kernel[grid_blur, block_blur, stream](
                    src_d, tmp_d, kern_d, np.int32(radius), np.int32(0)
                )  # blur along nav_y into tmp
                _gaussian_blur_1d_kernel[grid_blur, block_blur, stream](
                    tmp_d, src_d, kern_d, np.int32(radius), np.int32(1)
                )  # blur along nav_x back into src_d
        _sync_if_timing()
        timings["blur"] += time.perf_counter() - t0

        # ── Trim bounds (compacted on device by _trim_copy_kernel) ───────────
        lo = depth_px
        hi_y = NY_g - depth_px
        hi_x = NX_g - depth_px
        if lo >= hi_y or lo >= hi_x:
            lo, hi_y, hi_x = 0, NY_g, NX_g
        NY = hi_y - lo
        NX = hi_x - lo
        N = NY * NX

        grid_px = (int(np.ceil(KX / 16)), int(np.ceil(KY / 16)), N)
        blk_px = (16, 16, 1)

        t0 = time.perf_counter()
        sums_d = None
        sumsq_d = None
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            # Compact the trimmed core into contiguous (N, KY, KX) frames —
            # device-to-device; the strided ghost view never touches the host.
            frames_d = _gpu_pool_get((N, KY, KX), np.float32)
            _trim_copy_kernel[grid_px, blk_px, stream](
                src_d, frames_d, np.int32(lo), np.int32(lo), np.int32(NX), iH, iW
            )

            # Beamstop fill: per-frame unmasked mean on device, tiny D2H
            if have_mask:
                sums_d = _gpu_pool_get((N,), np.float32)
                sumsq_d = _gpu_pool_get((N,), np.float32)
                _frame_reduce_kernel[N, 256, stream](
                    frames_d, mask_d, np.int32(1), sums_d, sumsq_d,
                    np.int32(HW), iW,
                )
                sums_host = sums_d.copy_to_host(stream=stream)
                stream.synchronize()
                fills = (sums_host / n_unmasked).astype(np.float32)
                _beamstop_fill_kernel[grid_px, blk_px, stream](
                    frames_d, mask_d, _cuda.to_device(fills, stream=stream),
                    iH, iW,
                )
        _sync_if_timing()
        timings["stats"] += time.perf_counter() - t0

        raw_corr_d = None
        peak_mask_d = _gpu_pool_get((N, KY, KX), np.uint8)

        # ── NXCORR: CuPy/cuFFT when available, numba kernels otherwise ───────
        t0 = time.perf_counter()
        raw_corr_obj = None
        if use_fft:
            try:
                raw_corr_obj = _nxcorr_fft_cupy(
                    frames_d, int(kernel_r), disk_stats, stream
                )
                nx_path[0] = "fft"
            except Exception as exc:
                # Don't retry CuPy on every chunk of this process
                _cupy_state["ok"] = False
                nx_path[0] = "numba"
                log.warning("[find_vectors] CuPy NXCORR failed (%r) — "
                            "falling back to numba kernels", exc)

        if raw_corr_obj is None:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                # Per-frame global std for the denominator floor
                if sums_d is None:
                    sums_d = _gpu_pool_get((N,), np.float32)
                    sumsq_d = _gpu_pool_get((N,), np.float32)
                _frame_reduce_kernel[N, 256, stream](
                    frames_d, mask_d, np.int32(0), sums_d, sumsq_d,
                    np.int32(HW), iW,
                )
                sums = sums_d.copy_to_host(stream=stream)
                sumsqs = sumsq_d.copy_to_host(stream=stream)
                stream.synchronize()
                mean = sums / HW
                var = np.maximum(sumsqs / HW - mean * mean, 0.0)
                stds = np.sqrt(var).astype(np.float32)
                stds[stds == 0.0] = 1.0
                global_stds_d = _cuda.to_device(stds, stream=stream)

                raw_corr_d = _gpu_pool_get((N, KY, KX), np.float32)
                # Tiled shared-memory kernel when the halo fits the 64x64
                # tile (kr_win <= 24); naive reflected-index kernel beyond.
                if int(kr_win) <= 24:
                    _nxcorr_tiled_kernel[grid_px, blk_px, stream](
                        frames_d, disk_d, raw_corr_d, global_stds_d,
                        n_disk, t_mean, t_std, kr, kr_win, iH, iW,
                    )
                else:
                    _nxcorr_reflect_kernel[grid_px, blk_px, stream](
                        frames_d, disk_d, raw_corr_d, global_stds_d,
                        n_disk, t_mean, t_std, kr, kr_win, iH, iW,
                    )
            raw_corr_obj = raw_corr_d
        _sync_if_timing()
        timings["nxcorr"] += time.perf_counter() - t0

        peaks_out_d = None
        n_peaks_d = None
        t0 = time.perf_counter()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            # raw_corr_obj is a numba device array or a CuPy array — numba
            # kernels accept both via the CUDA array interface, and both were
            # produced on `stream`, so ordering is preserved.
            _local_max_kernel[grid_px, blk_px, stream](
                raw_corr_obj, peak_mask_d, thr, min_d, iH, iW,
            )
            if subpixel:
                peaks_out_d = _gpu_pool_get((N, MAX_PEAKS, 3), np.float32)
                n_peaks_d = _gpu_pool_get((N,), np.int32)
                n_peaks_d.copy_to_device(np.zeros(N, dtype=np.int32),
                                         stream=stream)
                _subpixel_com_kernel[grid_px, blk_px, stream](
                    raw_corr_obj, peak_mask_d, peaks_out_d, n_peaks_d,
                    np.int32(2), iH, iW,
                )
        _sync_if_timing()
        timings["peaks"] += time.perf_counter() - t0

        # D2H — only sparse results
        t0 = time.perf_counter()
        if subpixel:
            peaks_out = peaks_out_d.copy_to_host(stream=stream)
            n_peaks   = n_peaks_d.copy_to_host(stream=stream)
        else:
            peak_mask = peak_mask_d.copy_to_host(stream=stream)
            if raw_corr_d is not None:
                raw_corr = raw_corr_d.copy_to_host(stream=stream)
            else:
                import cupy as _cp
                with _cp.cuda.ExternalStream(_stream_ptr(stream)):
                    raw_corr = _cp.asnumpy(raw_corr_obj)
        # One sync per chunk: host reads below are safe and the pooled
        # buffers are idle, so cross-thread reuse on other streams is safe.
        stream.synchronize()
        timings["d2h"] += time.perf_counter() - t0

        _gpu_pool_put(raw_d, src_d, tmp_d, frames_d, sums_d, sumsq_d,
                      raw_corr_d, peak_mask_d, peaks_out_d, n_peaks_d)

        if subpixel:
            return NY, NX, N, peaks_out, n_peaks, None, None
        return NY, NX, N, None, None, peak_mask, raw_corr

    def _process_4d(block4d_cpu):
        """One block: stage into pinned memory, device section under the GPU
        slot semaphore, then the CPU pack outside it — staging and packing
        overlap another thread's GPU ownership."""
        # Stage into a pinned buffer BEFORE taking a GPU slot: the memcpy is
        # CPU work, and a pinned source makes the H2D truly asynchronous.
        t0 = time.perf_counter()
        pinned = _pinned_pool_get(block4d_cpu.shape, block4d_cpu.dtype)
        if pinned is not None:
            np.copyto(pinned, block4d_cpu)
            upload_src = pinned
        else:
            upload_src = block4d_cpu
        timings["stage"] += time.perf_counter() - t0

        try:
            with _gpu_slots():
                (NY, NX, N, peaks_out, n_peaks,
                 peak_mask, raw_corr) = _device_section(upload_src)
        finally:
            # Safe to recycle: the device section ends with a stream sync,
            # so the DMA from this buffer has completed.
            _pinned_pool_put(pinned)

        # Pack into (NY, NX, MAX_PEAKS, 3) NaN-padded
        t0 = time.perf_counter()
        out = np.full((NY, NX, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        min_d2 = int(min_dist) * int(min_dist)
        host_frames = block4d_cpu.reshape(-1, KY, KX)   # raw frames for intensity
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
            if have_mask:
                ky_px = np.clip(frame_peaks[:, 0].astype(int), 0, KY - 1)
                kx_px = np.clip(frame_peaks[:, 1].astype(int), 0, KX - 1)
                frame_peaks = frame_peaks[~beamstop_mask[ky_px, kx_px]]

            # Greedy NMS
            if len(frame_peaks) > 1:
                # Deterministic order: the GPU subpixel kernel fills slots via
                # atomics in arbitrary order, so tie-break equal scores by
                # position to keep results reproducible run-to-run.
                order = np.lexsort(
                    (frame_peaks[:, 1], frame_peaks[:, 0], -frame_peaks[:, 2])
                )
                frame_peaks = frame_peaks[order]
                kept = np.ones(len(frame_peaks), dtype=bool)
                for j in range(len(frame_peaks)):
                    if not kept[j]:
                        continue
                    dy = frame_peaks[j+1:, 0] - frame_peaks[j, 0]
                    dx = frame_peaks[j+1:, 1] - frame_peaks[j, 1]
                    kept[j+1:][(dy*dy + dx*dx) <= min_d2] = False
                frame_peaks = frame_peaks[kept]

            # Value column → raw experimental intensity at each peak (after NMS,
            # which used the corr score, matching the CPU path).
            frame_peaks = _with_raw_intensity(host_frames[i], frame_peaks)
            n = min(len(frame_peaks), MAX_PEAKS)
            if n > 0:
                out[iy, ix, :n, :] = frame_peaks[:n]
        timings["pack"] += time.perf_counter() - t0

        return out

    # ── Dispatch: 4D or 5D ────────────────────────────────────────────────────
    if nav_dim == 2:
        result = _process_4d(block_host)
        core_shape = result.shape[:2]
    else:
        n_lead = nav_shape_ghost[0]
        # Ghost block is (t, ny_ghost, nx_ghost, KY, KX) for 5D
        ny_ghost = nav_shape_ghost[1]
        nx_ghost = nav_shape_ghost[2]
        ny = max(1, ny_ghost - 2 * depth_px)
        nx = max(1, nx_ghost - 2 * depth_px)
        out5 = np.full((n_lead, ny, nx, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        for t in range(n_lead):
            out5[t] = _process_4d(block_host[t])
        result = out5
        core_shape = (n_lead, ny, nx)

    total_ms = (time.perf_counter() - t_start) * 1e3
    if stage_timing:
        log.debug(
            f"[find_vectors] GPU chunk core={tuple(int(s) for s in core_shape)} sig=({KY},{KX}) "
            f"path={nx_path[0]} "
            + " ".join(f"{k}={v * 1e3:.0f}ms" for k, v in timings.items())
            + f" total={total_ms:.0f}ms"
        )
    else:
        # Without per-stage syncs only host-blocking stages are meaningful.
        log.debug(
            f"[find_vectors] GPU chunk core={tuple(int(s) for s in core_shape)} sig=({KY},{KX}) "
            f"path={nx_path[0]} "
            f"stage={timings['stage'] * 1e3:.0f}ms h2d={timings['h2d'] * 1e3:.0f}ms "
            f"d2h={timings['d2h'] * 1e3:.0f}ms "
            f"pack={timings['pack'] * 1e3:.0f}ms total={total_ms:.0f}ms"
        )
    return result


def _balanced_nav_chunks(dim: int, target: int, depth: int) -> tuple:
    """
    Explicit chunk sizes for one nav dimension: ~target each, but never
    smaller than the ghost depth (map_overlap would silently rechunk and
    desync the declared output chunks), with any short remainder folded
    into the last chunk.
    """
    target = max(int(target), int(depth), 1)
    dim = int(dim)
    if dim <= target:
        return (dim,)
    n = dim // target
    chunks = [target] * n
    rem = dim - n * target
    if rem:
        if rem < max(int(depth), 1):
            chunks[-1] += rem
        else:
            chunks.append(rem)
    return tuple(chunks)


def _split_workers_for_gpu(client) -> tuple:
    """Lane split per SPYDE_FV_GPU — shared implementation in compute_dispatch."""
    from spyde.compute_dispatch import split_workers_for_gpu
    return split_workers_for_gpu(client)


def _compact_padded_chunk(arr: np.ndarray) -> np.ndarray:
    """
    Trim the NaN-padded MAX_PEAKS axis of one chunk result down to the
    longest actual peak list in the chunk.

    The dispatcher must hold every chunk future until the run ends —
    releasing futures mid-run while sibling graphs that share input keys are
    still being submitted races the scheduler (KeyError on forgotten keys).
    Compacting makes holding them cheap: ~30 real peaks instead of 512 slots.
    """
    valid = np.isfinite(arr[..., 0])
    n_max = int(valid.sum(axis=-1).max()) if valid.size else 0
    return np.ascontiguousarray(arr[..., :max(1, n_max), :])




def _dispatch_chunks_gpu_aware(
    client,
    result_array,
    nav_dim: int,
    gpu_addrs: list,
    cpu_addrs: list,
    stopped_flag=None,
):
    """Greedy dual-lane chunk dispatch — shared implementation in
    compute_dispatch; vectors-specific NaN-slot compaction via postprocess."""
    from spyde.compute_dispatch import dispatch_chunks
    return dispatch_chunks(
        client, result_array, nav_dim, gpu_addrs, cpu_addrs,
        stopped_flag=stopped_flag, postprocess=_compact_padded_chunk,
        fill_value=np.nan, label="find_vectors",
    )


def _count_chunk_to_shm(
    block: np.ndarray,
    block_info=None,
    shm_name: str = None,
    nav_2d_shape: tuple = None,
    nav_dim: int = 2,
) -> np.ndarray:
    """
    Passthrough map_blocks stage over the padded-peaks array.

    Counts finite peaks per nav position in this block and writes them into
    the live shared-memory count buffer at the block's global location, so
    the GUI (polling the buffer) sees the count image fill in chunk by chunk
    while the batch compute runs.  The block itself is returned unchanged.

    Runs on dask workers (threads or local subprocesses) — SharedMemory is
    attached by name, writes are best-effort and never raise.

    For 5D the buffer is NaN-initialised; chunks at different time indices
    accumulate into the same (y, x) region, treating NaN as "not written yet".
    """
    # Meta-inference calls pass empty blocks / no block_info — do nothing.
    if block_info is None or not isinstance(block_info, dict) or 0 not in block_info:
        return block
    try:
        loc = block_info[0]["array-location"]
    except Exception:
        return block

    try:
        counts = np.isfinite(block[..., 0]).sum(axis=-1).astype(np.float32)
        if nav_dim == 3 and counts.ndim == 3:
            counts_2d = counts.sum(axis=0)
        else:
            counts_2d = counts
        ys, xs = loc[nav_dim - 2], loc[nav_dim - 1]

        from multiprocessing import shared_memory as _shm_mod
        shm = _shm_mod.SharedMemory(name=shm_name, create=False)
        try:
            buf = np.ndarray(nav_2d_shape, dtype=np.float32, buffer=shm.buf)
            region = buf[ys[0]:ys[1], xs[0]:xs[1]]
            if nav_dim == 3:
                buf[ys[0]:ys[1], xs[0]:xs[1]] = (
                    np.where(np.isfinite(region), region, 0.0) + counts_2d
                )
            else:
                buf[ys[0]:ys[1], xs[0]:xs[1]] = counts_2d
            del region, buf
        finally:
            shm.close()
    except Exception as e:
        log.debug("live count-map shm update for block failed: %s", e)
    return block


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

    Submission: with a distributed client and a designated GPU worker the
    per-chunk futures go through _dispatch_chunks_gpu_aware (greedy dual-lane
    placement — see its docstring); otherwise the whole graph is one dask
    future.  Either way a passthrough map_blocks stage writes per-chunk peak
    counts into the shared-memory buffer as chunk tasks finish, so a GUI
    polling the buffer sees the count image update live.

    Parameters
    ----------
    shm_name : str | None
        Pre-existing float32 SharedMemory segment; chunk counts are written
        into it from worker tasks while the compute runs, plus a final
        authoritative write when the compute completes.
    on_chunk_done : callable(nav_slice_2d, count_subarray) | None
        Called once after the full compute completes with the full-nav slice.
    stopped_flag : list[bool] | None
        Polled while waiting on the future; setting it cancels the compute
        and returns None.
    """
    import functools
    import dask.array as da
    from spyde.signals.diffraction_vectors import (
        N_COLS, COL_KX, COL_KY, COL_TIME, COL_INTENSITY, SpyDEDiffractionVectors
    )
    import time

    tic = time.time()

    nav_dim = signal.axes_manager.navigation_dimension
    sig_dim = signal.axes_manager.signal_dimension
    sig_ax = signal.axes_manager.signal_axes
    sig_shape = signal.axes_manager.signal_shape  # (ky, kx) in HS order

    sigma = float(params["sigma"])
    depth_px = int(np.ceil(3 * sigma))

    method = str(params.get("method", METHOD_NXCORR)).lower()
    kernel_r = int(params["kernel_radius"])
    threshold = float(params["threshold"])
    min_dist = int(params["min_distance"])
    subpixel = bool(params.get("subpixel", True))
    dog_sigma1 = float(params.get("dog_sigma1", DEFAULT_DOG_SIGMA1))
    dog_sigma2 = float(params.get("dog_sigma2", DEFAULT_DOG_SIGMA2))
    log.debug("[do_compute_vectors] START method=%s thr=%s md=%s sigma=%s "
              "nav_dim=%s sig_shape=%s lazy=%s beamstop=%s", method, threshold,
              min_dist, sigma, nav_dim, tuple(sig_shape),
              getattr(signal, "_lazy", "?"),
              beamstop_mask is not None)

    ky_scale = float(sig_ax[1].scale)
    ky_offset = float(sig_ax[1].offset)
    kx_scale = float(sig_ax[0].scale)
    kx_offset = float(sig_ax[0].offset)

    # ── Chunk size: fixed ~100 MB RAM budget per ghost-padded chunk ──────────
    # Uniform small chunks keep every worker's memory bounded, stream a steady
    # supply of tasks (GPU included), and let the live count map fill in
    # progressively.  Sizing chunks to free VRAM collapsed small datasets into
    # a single chunk — a full rechunk-shuffle through one worker, no live
    # updates — and exploded per-worker RAM on large datasets.
    chunk_nav = _nav_chunk_size(sigma, max_ram_mb=100, sig_shape=sig_shape)

    raw = signal.data
    nav_shape_full = raw.shape[:nav_dim]
    nav_2d_shape = nav_shape_full[-2:]
    n_nav_y, n_nav_x = nav_2d_shape

    # Ghost depth cannot exceed the spatial nav extent (map_overlap rejects
    # depth > axis size); tiny grids just lose cross-boundary blur support.
    depth_px = max(0, min(depth_px, min(nav_2d_shape) - 1))

    # ── Auto beam-stop detection ──────────────────────────────────────────────
    # When no mask was supplied and the user asked for one (or it's requested by
    # default), detect a physical beam stop from a SPARSE sample of patterns
    # (never the full dataset — memory rule).  The stop is static, so a few
    # hundred frames give a clean low-intensity mask; dilated to clear the rim.
    if beamstop_mask is None and params.get("beamstop_auto", False):
        try:
            beamstop_mask = _auto_beamstop_from_signal(signal, nav_dim)
            if beamstop_mask is not None:
                log.debug("[find_vectors] auto beam-stop: %d px masked",
                          int(beamstop_mask.sum()))
        except Exception as e:
            log.debug("[find_vectors] auto beam-stop detection failed: %s", e)
            beamstop_mask = None

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
    # Leading (time) dims are chunked at 1 so a 5D chunk has the same memory
    # footprint as a 4D one.  Every spatial chunk must also be >= depth_px so
    # map_overlap never rechunks internally (that would desync out_chunks).
    # float32 conversion happens per chunk inside chunk_fn — never materialise
    # a converted copy of the full dataset here.
    min_chunk = max(depth_px, 1)
    nav_chunks_tuple = (1,) * (nav_dim - 2) + tuple(
        _balanced_nav_chunks(d, chunk_nav, depth_px) for d in nav_2d_shape
    )
    sig_chunks_tuple = tuple(s for s in raw.shape[nav_dim:])
    if isinstance(raw, np.ndarray):
        da_data = da.from_array(raw, chunks=nav_chunks_tuple + sig_chunks_tuple)
    else:
        # Avoid a rechunk shuffle when the stored chunking is already usable:
        # spatial nav chunks within [depth, ~2x budget], leading dims chunked
        # at 1, signal axes unchunked.  Alignment beats the theoretical chunk
        # budget: rechunking to a misaligned "better" size makes every ghost
        # block gather split pieces from many source chunks — measured 2.3x
        # slower (419 s vs 184 s on the 64 GiB benchmark) from the transfers
        # and memory churn alone.
        keep_limit = min(max(2 * chunk_nav, chunk_nav + 2 * depth_px),
                         255 - 2 * depth_px)
        nav_ok = all(
            min(c) >= min_chunk and max(c) <= keep_limit
            for c in raw.chunks[max(0, nav_dim - 2):nav_dim]
        ) and all(
            max(c) <= 1 for c in raw.chunks[:max(0, nav_dim - 2)]
        )
        sig_ok = all(len(c) == 1 for c in raw.chunks[nav_dim:])
        if nav_ok and sig_ok:
            da_data = raw
        else:
            log.debug("Rechunking to %s and %s", nav_chunks_tuple, sig_chunks_tuple)
            da_data = raw.rechunk(nav_chunks_tuple + sig_chunks_tuple)

    toc  = time.time()
    log.debug("Prepared dask array with chunks %s in %.1f s", da_data.chunks, toc - tic)

    # ── Build the map_overlap graph ───────────────────────────────────────────
    # trim=False: chunk_fn receives the full ghost-padded block and handles
    # trimming itself so the blur can use all ghost rows for correct boundaries.
    # drop_axis removes (ky, kx); new_axis adds (MAX_PEAKS, 3).
    # Output: (nav_y, nav_x, MAX_PEAKS, 3) [4D] or (t, ny, nx, MAX_PEAKS, 3) [5D].
    # Ghost zones only on the two spatial nav dims — the leading (time) axis
    # has blur sigma 0, so overlapping it would be pure overhead (and the
    # chunk fn does not trim it).
    depth_dict = {i: 0 for i in range(nav_dim)}
    depth_dict[nav_dim - 2] = depth_px
    depth_dict[nav_dim - 1] = depth_px
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
        method=method,
        dog_sigma1=dog_sigma1,
        dog_sigma2=dog_sigma2,
    )

    # Resolve the distributed client up front — needed both to decide on GPU
    # routing and to submit the future below.
    client = None
    if signal_tree is not None:
        client = getattr(signal_tree, "client", None)
    if client is None and main_window is not None:
        client = getattr(getattr(main_window, "dask_manager", None), "client", None)

    # Only annotate with the GPU resource when at least one worker actually
    # advertises it.  Annotating without such workers leaves every chunk task
    # stuck in no-worker state — the compute never starts.
    gpu_workers = False
    if client is not None:
        try:
            workers_info = client.scheduler_info(n_workers=-1)["workers"]
            gpu_workers = any(
                (w.get("resources") or {}).get("GPU")
                for w in workers_info.values()
            )
        except Exception:
            gpu_workers = False

    import dask
    import contextlib
    tic = time.time()
    annotate_ctx = (
        dask.annotate(resources={"GPU": 1}) if gpu_workers
        else contextlib.nullcontext()
    )
    with annotate_ctx:
        # meta= prevents dask from calling chunk_fn on empty arrays in the
        # client process for type inference.
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
            meta=np.empty((0,) * (nav_dim + 2), dtype=np.float32),
        )
    toc = time.time()
    log.debug("Built Dask graph with map_overlap in %.1f s", toc - tic)

    # ── Live count map: write per-chunk counts into shm from the graph ───────
    # A passthrough map_blocks stage counts finite peaks per nav position and
    # writes them into the shared-memory buffer as each chunk task finishes,
    # so the GUI can poll the buffer and watch the count image fill in live.
    if shm_name is not None:
        counted = da.map_blocks(
            functools.partial(
                _count_chunk_to_shm,
                shm_name=shm_name,
                nav_2d_shape=tuple(nav_2d_shape),
                nav_dim=nav_dim,
            ),
            peaks_padded,
            dtype=np.float32,
            meta=np.empty((0,) * peaks_padded.ndim, dtype=np.float32),
        )
    else:
        counted = peaks_padded

    # ── Submit as a dask future — only this (background) thread blocks ───────
    if stopped_flag is not None and stopped_flag[0]:
        return None

    log.debug("computing...")
    tic = time.time()
    if client is not None:
        # GPU-aware dual-lane dispatcher when the cluster has a designated
        # GPU worker: dask's locality-driven placement starves the (much
        # faster) GPU worker, so we place per-chunk futures ourselves.
        gpu_addrs, cpu_addrs = _split_workers_for_gpu(client)
        if gpu_addrs and cpu_addrs:
            log.debug(
                f"[find_vectors] dispatcher lanes: "
                f"GPU={len(gpu_addrs)} worker(s), CPU={len(cpu_addrs)} worker(s)"
            )
            result_padded = _dispatch_chunks_gpu_aware(
                client, counted, nav_dim, gpu_addrs, cpu_addrs,
                stopped_flag=stopped_flag,
            )
            if result_padded is None:
                return None
        else:
            future = client.compute(counted)
            while not future.done():
                if stopped_flag is not None and stopped_flag[0]:
                    try:
                        future.cancel()
                    except Exception as e:
                        log.debug("cancelling find-vectors compute future failed: %s", e)
                    return None
                time.sleep(0.1)
            result_padded = future.result()
    else:
        # No distributed client (e.g. unit tests): local threaded scheduler,
        # pinned explicitly — a bare .compute() silently runs on any ambient
        # distributed Client (dask's global default when one exists).
        # This computes the small padded-peaks output, never the raw dataset.
        result_padded = counted.compute(scheduler="threads")
    toc = time.time()
    log.debug("Computed vectors in %.1f s", toc - tic)

    if stopped_flag is not None and stopped_flag[0]:
        return None

    # ── Unpack padded result into flat buffer (vectorised) ───────────────────
    # result_padded shape: (nav_y, nav_x, MAX_PEAKS, 3)  [4D]
    #                   or (t, nav_y, nav_x, MAX_PEAKS, 3) [5D]
    # Valid peaks have finite ky (col 0); NaN-padded slots are ignored.
    # np.nonzero returns C-order indices, so the flat buffer comes out already
    # sorted outermost-nav-dim first as SpyDEDiffractionVectors requires.
    valid = np.isfinite(result_padded[..., 0])
    nz = np.nonzero(valid)
    peaks_flat = result_padded[valid]              # (N_total, 3)
    N_total = peaks_flat.shape[0]

    flat_buffer = np.zeros((N_total, N_COLS), dtype=np.float32)
    if nav_dim == 2:
        iy_idx, ix_idx = nz[0], nz[1]
        flat_buffer[:, COL_TIME] = -1.0
    else:
        t_idx, iy_idx, ix_idx = nz[0], nz[1], nz[2]
        flat_buffer[:, COL_TIME] = t_idx.astype(np.float32)
    flat_buffer[:, 0] = ix_idx
    flat_buffer[:, 1] = iy_idx
    flat_buffer[:, COL_KX] = peaks_flat[:, 1] * kx_scale + kx_offset
    flat_buffer[:, COL_KY] = peaks_flat[:, 0] * ky_scale + ky_offset
    flat_buffer[:, COL_INTENSITY] = peaks_flat[:, 2]

    # ── Final shm write / completion callback ─────────────────────────────────
    if shm_name is not None or on_chunk_done is not None:
        counts_2d = valid.sum(axis=-1)
        if nav_dim == 3:
            counts_2d = counts_2d.sum(axis=0)
        count_map = counts_2d.astype(np.int32)
        if shm_name is not None:
            from multiprocessing import shared_memory as _shm_mod
            try:
                shm_handle = _shm_mod.SharedMemory(name=shm_name, create=False)
                shm_buf = np.ndarray(nav_2d_shape, dtype=np.float32, buffer=shm_handle.buf)
                shm_buf[:] = count_map.astype(np.float32)
                shm_handle.close()
            except Exception as e:
                log.debug("final count-map shm write to %s failed: %s", shm_name, e)
        if on_chunk_done is not None:
            on_chunk_done((slice(None), slice(None)), count_map)

    log.debug("[do_compute_vectors] DONE: %d vectors over %s nav positions "
              "(%.2f per pattern)", N_total, int(np.prod(nav_shape_full)),
              N_total / max(1, int(np.prod(nav_shape_full))))

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

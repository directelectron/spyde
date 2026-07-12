"""
detectors.py — per-frame peak-finding algorithm cores for find_vectors.

The single-frame detectors (NXCORR + DoG), their subpixel refinement, disk
kernels, raw-intensity sampling, beam-stop detection and the per-frame
dispatch + auto-parameter helpers.  No dask / distributed orchestration and no
GPU-chunk machinery live here — those are in chunk.py / orchestrate.py.

The live-preview NavBlurCache (async per-chunk nav-space Gaussian blur) also
lives here as it piggybacks on the single-frame fallback.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.fft import rfft2, irfft2, next_fast_len

log = logging.getLogger(__name__)

# Cache of pre-computed disk FFTs keyed by (radius, padded_H, padded_W)
_DISK_FFT_CACHE: dict = {}


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


def _disk_mean_intensity(frame: np.ndarray, ys, xs, radius: float) -> np.ndarray:
    """Mean of the (experimental) frame over a small DISK of ``radius`` px around
    each peak — a robust per-disk brightness for virtual imaging / weighting.

    Why not a single bilinear point (the old ``_sample_raw_bilinear`` used for the
    intensity column): a 1-pixel sample is dominated by shot noise and by exactly
    where the sub-pixel centre landed, so the stored "intensity" jumps sharply
    frame-to-frame even when the underlying disk brightness is smooth — which
    shows up as BANDING/stripes in an intensity-weighted virtual image. Averaging
    over the disk (the same footprint the peak represents) removes that: the
    value tracks true disk brightness and varies continuously between neighbours.

    Falls back to a bilinear point sample when radius < 1 (degenerate)."""
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    r = int(max(0, round(float(radius))))
    if r < 1 or ys.size == 0:
        return _sample_raw_bilinear(frame, ys, xs)
    H, W = frame.shape
    f = frame.astype(np.float32, copy=False)
    # Offsets covering the disk footprint once (shared across all peaks).
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    inside = (oy * oy + ox * ox) <= r * r
    oy, ox = oy[inside], ox[inside]                      # (K,) disk offsets
    cy = np.rint(ys).astype(np.intp)[:, None] + oy[None]  # (N, K)
    cx = np.rint(xs).astype(np.intp)[:, None] + ox[None]
    valid = (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)   # clip at frame edges
    cy = np.clip(cy, 0, H - 1)
    cx = np.clip(cx, 0, W - 1)
    samp = f[cy, cx]
    samp = np.where(valid, samp, np.nan)
    # nanmean over the disk → robust to edge clipping; all-nan (shouldn't happen)
    # falls back to 0.
    with np.errstate(invalid="ignore"):
        out = np.nanmean(samp, axis=1)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def _with_raw_intensity(frame: np.ndarray, peaks: np.ndarray,
                        radius: float = 0.0) -> np.ndarray:
    """Overwrite a peak set's value column (col 2) with the raw frame intensity
    at each peak — keeping the NXCORR-derived position. For the GPU/torch paths,
    which return correlation scores. ``radius`` > 0 → robust disk-mean intensity
    (recommended; avoids the single-pixel banding); 0 → legacy bilinear point."""
    peaks = np.asarray(peaks, dtype=np.float32)
    if peaks.size == 0:
        return peaks.reshape(-1, 3)
    out = peaks.copy()
    if radius and radius >= 1:
        out[:, 2] = _disk_mean_intensity(frame, peaks[:, 0], peaks[:, 1], radius)
    else:
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

    # Beam stop = a PEAK-REJECTION region, NOT an image edit. Do NOT fill the
    # masked pixels: a fill creates a sharp step at the mask boundary that the
    # disk correlator scores as bright spots along the rim. Run NXCORR on the
    # UNMODIFIED frame; below we force the score to -1 inside the mask and drop
    # any peaks there, so the stop region contributes no detections without
    # introducing an artificial edge.

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
    # Disk-MEAN brightness over the kernel footprint (robust), not a single
    # sub-pixel sample — a 1-px value swings frame-to-frame and bands an
    # intensity-weighted virtual image. See _disk_mean_intensity.
    intens = _disk_mean_intensity(frame, pos[:, 0], pos[:, 1], kernel_radius)
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

    Beam stop is a **peak-rejection region, not an image edit**: the frame is
    left UNMODIFIED (no fill/zero — any fill creates a step at the mask edge that
    the band-pass fires on as spurious rim spots). Detections inside the
    (optionally further-dilated) mask are dropped, and the masked rim is excluded
    from the robust SNR scale so it can't inflate the MAD.

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

    # Beam stop = a PEAK-REJECTION region, NOT an image edit. We deliberately do
    # NOT fill/zero the masked pixels: filling introduces a sharp intensity step
    # at the mask boundary that the band-pass itself fires on (spurious bright
    # spots along the rim). Instead the detector runs on the UNMODIFIED frame and
    # any peak landing inside the (dilated) mask is dropped below. `excl` is also
    # used to exclude the rim from the robust SNR scale so it can't inflate MAD.

    excl = None
    if beamstop_mask is not None and beamstop_mask.any():
        excl = _dilate_mask(beamstop_mask, bs_dilate) if bs_dilate else beamstop_mask

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
    intens = _disk_mean_intensity(
        np.asarray(frame, dtype=np.float32), pos[:, 0], pos[:, 1],
        max(1.0, np.ceil(sigma2)))
    peaks = np.column_stack([pos, intens]).astype(np.float32)
    return corr_map, snr, peaks


# Method names accepted across the find-vectors stack.
METHOD_NXCORR = "nxcorr"
METHOD_DOG = "dog"
METHOD_NEURAL = "neural"
DEFAULT_DOG_SIGMA1 = 0.8
DEFAULT_DOG_SIGMA2 = 2.0
# Absolute band-pass SNR (response / 1.4826·MAD).  ~10 balances recall/precision
# on real small-spot data (benchmarked on the 3 nm DESEMCam scan); raise toward
# 15 for cleaner / fewer peaks, lower toward 6 for more recall.
DEFAULT_DOG_THRESHOLD = 10.0
# Heatmap confidence below which neural (SpotUNet) detections are discarded —
# the model's natural, dataset-independent operating point (mirrors
# find_vectors_neural.DEFAULT_NEURAL_THRESHOLD; kept literal here to avoid a
# circular import — find_vectors_neural imports back from this package).
DEFAULT_NEURAL_THRESHOLD = 0.3


def _find_peaks_single_frame(frame, params, *, beamstop_mask=None,
                             _disk_fft=None, _disk_stats=None,
                             with_response=False):
    """Dispatch a single frame to the configured detector, returning peaks
    ``(N,3)`` ``[ky, kx, intensity]``.  ``params`` is the find-vectors param
    dict (``method`` selects ``neural``, ``nxcorr`` or ``dog``).

    ``with_response=True`` returns ``(peaks, response_map)`` where response_map is
    the detector's transformed image — the neural confidence heatmap, the DoG
    band-pass SNR or the NXCORR correlation surface — for the "show transform"
    preview toggle."""
    method = str(params.get("method", METHOD_NXCORR)).lower()
    if method == METHOD_NEURAL:
        # Lazy import — find_vectors_neural imports back from this package.
        from spyde.actions.find_vectors_neural import (
            _find_vectors_single_frame_neural,
        )
        out = _find_vectors_single_frame_neural(
            frame,
            float(params.get("threshold", DEFAULT_NEURAL_THRESHOLD)),
            int(params.get("min_distance", 3)),
            subpixel=bool(params.get("subpixel", True)),
            beamstop_mask=beamstop_mask,
            model_id=(params.get("model_id") or None),
        )
    elif method == METHOD_DOG:
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

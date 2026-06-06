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
from scipy.ndimage import center_of_mass, gaussian_filter, maximum_filter
from scipy.fft import rfft2, irfft2, next_fast_len

from spyde.drawing.toolbars.toolbar import RoundedToolBar

# Cache of pre-computed disk FFTs keyed by (radius, padded_H, padded_W)
_DISK_FFT_CACHE: dict = {}

# ── Module-level guard (one caret per toolbar) ─────────────────────────────────
_FV_BUILT_TOOLBARS: set = set()


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
    # Where the local window std is zero the region is flat — no real peak can
    # exist there, so score is 0.  Flooring with 1e-8 would inflate these to
    # arbitrary large values after dividing a non-zero numerator.
    numerator = xcorr / n - win_mean * t_mean
    denom = win_std * t_std
    valid = denom >= 1e-8
    raw_corr = np.where(valid, numerator / np.where(valid, denom, 1.0), 0.0).astype(np.float32)
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
        min_distance=max(1, 2 * r_px),
        subpixel=True,
    )


def _nav_chunk_size(
    sigma: float, max_ram_mb: float = 200, sig_shape: tuple = (256, 256)
) -> int:
    """Compute the nav chunk size so the ghost-padded chunk fits within max_ram_mb."""
    depth = int(np.ceil(3 * sigma))
    sig_pixels = sig_shape[0] * sig_shape[1]
    max_padded = int(np.sqrt(max_ram_mb * 1e6 / (sig_pixels * 4)))
    return max(depth + 1, max_padded - 2 * depth)


# ─────────────────────────────────────────────────────────────────────────────
# Worker-side chunk function (module-level so it is picklable)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_find_vectors_chunk(
    full_data,          # numpy array (the full dataset, local on the worker)
    y0, y1, x0, x1,    # nav 2D chunk bounds within nav_2d
    depth_px,           # reflect-pad depth in nav dims
    n_nav_y, n_nav_x,   # full nav grid size (for padding clamps)
    sigma_tuple,        # gaussian_filter sigma tuple (all dims)
    kernel_r,           # disk radius in pixels
    threshold,          # NXCORR threshold
    min_dist,           # min peak separation in pixels
    subpixel,           # subpixel CoM refinement
    beamstop_mask,      # (ky, kx) bool array or None
):
    """
    Run on a dask.distributed worker that already holds `full_data`.

    Slices a reflect-padded nav chunk, blurs it, peak-finds every frame, and
    returns only the compact results — no bulk data leaves the worker.

    Returns
    -------
    count_2d  : (cy, cx) int32  — peaks per probe position
    peaks_out : list of (N_i, 3) float32 arrays, one per position in raster order
                columns: [ky_px, kx_px, nxcorr_value]
    """
    import numpy as _np
    from scipy.ndimage import gaussian_filter as _gf

    # Reflect-pad in nav dims with boundary clamps
    py0 = max(0, y0 - depth_px); py1 = min(n_nav_y, y1 + depth_px)
    px0 = max(0, x0 - depth_px); px1 = min(n_nav_x, x1 + depth_px)

    # Slice and blur (all local — zero TCP)
    chunk_raw = _np.asarray(full_data[py0:py1, px0:px1], dtype=_np.float32)
    chunk_blurred = _gf(chunk_raw, sigma=sigma_tuple)

    # Trim to valid region
    vy0 = y0 - py0; vy1 = vy0 + (y1 - y0)
    vx0 = x0 - px0; vx1 = vx0 + (x1 - x0)
    chunk_valid = chunk_blurred[vy0:vy1, vx0:vx1]  # (cy, cx, ky, kx)
    cy, cx = chunk_valid.shape[:2]
    sig_shape = chunk_valid.shape[2:]

    # Pre-compute disk FFT and stats once per chunk call
    from scipy.fft import next_fast_len as _nfl, rfft2 as _rfft2, irfft2 as _irfft2
    from scipy.ndimage import maximum_filter as _mf

    kH = 2 * kernel_r + 1
    kW = kH
    _n = kH * kW
    disk = _np.zeros((kH, kW), dtype=_np.float32)
    yy, xx = _np.ogrid[-kernel_r: kernel_r + 1, -kernel_r: kernel_r + 1]
    disk[yy ** 2 + xx ** 2 <= kernel_r ** 2] = 1.0
    disk /= disk.sum()
    t_mean = float(disk.mean())
    t_std = float(_np.sqrt(_np.sum((disk - t_mean) ** 2) / _n))

    ky, kx = sig_shape
    pH = _nfl(ky + 2 * kernel_r); pW = _nfl(kx + 2 * kernel_r)
    d_buf = _np.zeros((pH, pW), dtype=_np.float32)
    d_buf[:kH, :kW] = disk
    disk_fft = _rfft2(d_buf)

    count_2d = _np.zeros((cy, cx), dtype=_np.int32)
    peaks_out = []  # one entry per position (ly, lx) in raster order

    for ly in range(cy):
        for lx in range(cx):
            frame = chunk_valid[ly, lx]

            # NXCORR
            padded_full = _np.pad(frame, kernel_r, mode="reflect")
            buf = _np.zeros((pH, pW), dtype=_np.float32)
            buf[:ky + 2*kernel_r, :kx + 2*kernel_r] = padded_full
            xcorr = _irfft2(_rfft2(buf) * disk_fft.conj())[:ky, :kx].astype(_np.float32)

            cum1 = _np.empty((ky + 2*kernel_r + 1, kx + 2*kernel_r + 1), dtype=_np.float32)
            cum1[0, :] = 0.0; cum1[:, 0] = 0.0
            cum1[1:, 1:] = _np.cumsum(_np.cumsum(padded_full, axis=0), axis=1)
            cum2 = _np.empty_like(cum1)
            cum2[0, :] = 0.0; cum2[:, 0] = 0.0
            cum2[1:, 1:] = _np.cumsum(_np.cumsum(padded_full ** 2, axis=0), axis=1)

            ws1 = (cum1[kH:ky+kH, kW:kx+kW] - cum1[0:ky, kW:kx+kW]
                   - cum1[kH:ky+kH, 0:kx] + cum1[0:ky, 0:kx])
            ws2 = (cum2[kH:ky+kH, kW:kx+kW] - cum2[0:ky, kW:kx+kW]
                   - cum2[kH:ky+kH, 0:kx] + cum2[0:ky, 0:kx])
            win_mean = ws1 / _n
            win_var = ws2 / _n - win_mean ** 2
            _np.maximum(win_var, 0.0, out=win_var)
            win_std = _np.sqrt(win_var)
            numerator = xcorr / _n - win_mean * t_mean
            denom = win_std * t_std
            _np.maximum(denom, 1e-8, out=denom)
            raw_corr = _np.clip(numerator / denom, -1.0, 1.0).astype(_np.float32)

            if beamstop_mask is not None and beamstop_mask.any():
                raw_corr[beamstop_mask] = -1.0

            # Peak detection
            if not (raw_corr >= threshold).any():
                peaks_out.append(_np.zeros((0, 3), dtype=_np.float32))
                continue

            min_d = int(min_dist)
            local_max = _mf(raw_corr, size=2 * min_d + 1)
            peaks_mask = (raw_corr == local_max) & (raw_corr >= threshold)
            if beamstop_mask is not None and beamstop_mask.any():
                peaks_mask &= ~beamstop_mask
            peaks_px = _np.argwhere(peaks_mask)

            if len(peaks_px) == 0:
                peaks_out.append(_np.zeros((0, 3), dtype=_np.float32))
                continue

            # Greedy NMS
            if len(peaks_px) > 1:
                intensities = raw_corr[peaks_px[:, 0], peaks_px[:, 1]]
                order = _np.argsort(-intensities)
                peaks_px = peaks_px[order]
                kept = _np.ones(len(peaks_px), dtype=bool)
                min_d2 = min_d * min_d
                for i in range(len(peaks_px)):
                    if not kept[i]:
                        continue
                    dy = peaks_px[i + 1:, 0] - peaks_px[i, 0]
                    dx = peaks_px[i + 1:, 1] - peaks_px[i, 1]
                    kept[i + 1:][(dy * dy + dx * dx) <= min_d2] = False
                peaks_px = peaks_px[kept]

            # Subpixel CoM
            if subpixel and len(peaks_px) > 0:
                half = 2
                out = _np.empty((len(peaks_px), 3), dtype=_np.float32)
                for i, (py, px) in enumerate(peaks_px):
                    y0p = max(0, int(py) - half); y1p = min(ky, int(py) + half + 1)
                    x0p = max(0, int(px) - half); x1p = min(kx, int(px) + half + 1)
                    patch = raw_corr[y0p:y1p, x0p:x1p]
                    s = float(patch.sum())
                    if s > 0:
                        wy = patch.sum(axis=1); wx = patch.sum(axis=0)
                        dy = float(_np.dot(_np.arange(len(wy), dtype=_np.float32), wy)) / s
                        dx = float(_np.dot(_np.arange(len(wx), dtype=_np.float32), wx)) / s
                    else:
                        dy, dx = float(py) - y0p, float(px) - x0p
                    out[i] = [y0p + dy, x0p + dx, raw_corr[int(py), int(px)]]
                peaks_arr = out
            else:
                peaks_arr = _np.column_stack([
                    peaks_px.astype(_np.float32),
                    raw_corr[peaks_px[:, 0], peaks_px[:, 1]],
                ])

            count_2d[ly, lx] = len(peaks_arr)
            peaks_out.append(peaks_arr)

    return count_2d, peaks_out


# ─────────────────────────────────────────────────────────────────────────────
# Batch compute
# ─────────────────────────────────────────────────────────────────────────────


def _do_compute_vectors(
    signal, params: dict, main_window, signal_tree,
    shm_name: str = None,
    beamstop_mask: np.ndarray = None,
    on_chunk_done=None,
    stopped_flag=None,
):
    """
    Single-pass batch compute — pure numpy, no dask round-trips.

    Key design: materialize the signal data to a numpy array **once** before
    the loop.  This eliminates two major sources of latency that appeared when
    using dask.array.map_overlap + per-chunk .compute():

      1. da.from_array on a large numpy array calls dask.base.tokenize() which
         hashes the entire array (~1 s for a 1 GB dataset).
      2. Each blurred_lazy[slice].compute() call traverses a 270-task graph and,
         when using dask.distributed, incurs TCP serialization round-trips for
         every chunk (~2-5 s overhead per chunk on localhost).

    Instead we:
      - Materialize data to float32 numpy once (zero-copy if already numpy/float32).
      - Iterate nav chunks manually, reflect-padding each chunk with scipy.
      - Run NXCORR peak-finding on each frame inline — no dask scheduler involved.

    Nav-space Gaussian blur is applied per chunk with reflect boundary padding
    of depth=ceil(3σ) on each edge so chunk boundaries are handled correctly
    (equivalent to dask.array.map_overlap's ghost zones).

    Parameters
    ----------
    shm_name : str | None
        Pre-existing float32 SharedMemory segment of size nav_2d_shape.
        Written live as each chunk finishes.
    on_chunk_done : callable(nav_slice_2d, count_subarray) | None
        Called from this thread after each nav chunk completes.
    stopped_flag : list[bool] | None
        If stopped_flag[0] becomes True, the loop exits after the current chunk.
    """
    import itertools
    from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors

    nav_dim = signal.axes_manager.navigation_dimension
    sig_dim = signal.axes_manager.signal_dimension
    sig_ax = signal.axes_manager.signal_axes
    sig_shape = signal.axes_manager.signal_shape  # (ky, kx) in HS order

    sigma = float(params["sigma"])
    depth_px = int(np.ceil(3 * sigma))
    # sigma_tuple: blur only the nav dimensions, not the signal dimensions
    sigma_tuple = tuple([0.0] * (nav_dim - 2) + [sigma, sigma] + [0.0] * sig_dim)

    chunk_nav = _nav_chunk_size(sigma, max_ram_mb=200, sig_shape=sig_shape)

    # ── Resolve the data — never pull the full dataset over TCP ──────────────
    # signal.data may be:
    #   (a) numpy array — already local, use directly
    #   (b) dask.array.Array (lazy) — keep lazy, use client.submit per chunk
    #   (c) dask.distributed.Future — data is on a worker; use client.submit
    #       so work runs on that worker, only small results return
    #
    # In ALL cases we must NOT call raw.result() or raw.compute() on the full
    # dataset — that would pull hundreds of GB across TCP or into RAM.
    raw = signal.data
    from dask.distributed import Future as _DistFuture
    import dask.array as _da

    _is_numpy = isinstance(raw, np.ndarray)
    _is_dask = isinstance(raw, _da.Array)
    _is_future = isinstance(raw, _DistFuture) or (not _is_numpy and not _is_dask and hasattr(raw, 'result'))

    # Derive shape without materialising
    if _is_numpy:
        numpy_data = raw.astype(np.float32) if raw.dtype != np.float32 else raw
        nav_shape_full = numpy_data.shape[:nav_dim]
    elif _is_dask:
        nav_shape_full = raw.shape[:nav_dim]
    else:
        # Future: shape is not directly available, read from axes_manager
        nav_shape_full = tuple(
            signal.axes_manager.navigation_shape[::-1]
        )  # HS stores (nx, ny), data stored (ny, nx)
    nav_2d_shape = nav_shape_full[-2:]
    n_nav_y, n_nav_x = nav_2d_shape
    n_patterns = n_nav_y * n_nav_x

    kernel_r = int(params["kernel_radius"])
    threshold = float(params["threshold"])
    min_dist = int(params["min_distance"])
    subpixel = bool(params.get("subpixel", True))

    ky_scale = float(sig_ax[1].scale)
    ky_offset = float(sig_ax[1].offset)
    kx_scale = float(sig_ax[0].scale)
    kx_offset = float(sig_ax[0].offset)

    # Pre-compute disk FFT and statistics once — reused for every frame.
    pH = next_fast_len(sig_shape[0] + 2 * kernel_r)
    pW = next_fast_len(sig_shape[1] + 2 * kernel_r)
    disk_fft = _get_disk_fft(kernel_r, pH, pW)
    _disk = _make_disk(kernel_r)
    _n = _disk.shape[0] * _disk.shape[1]
    _t_mean = float(_disk.mean())
    _t_std = float(np.sqrt(np.sum((_disk - _t_mean) ** 2) / _n))
    disk_stats = (_n, _t_mean, _t_std)

    # ── Chunk boundaries (shared by both paths) ───────────────────────────────
    def _chunk_ranges(dim_size):
        starts = list(range(0, dim_size, chunk_nav))
        return [(s, min(s + chunk_nav, dim_size)) for s in starts]

    chunks_y = _chunk_ranges(n_nav_y)
    chunks_x = _chunk_ranges(n_nav_x)
    if nav_dim == 2:
        chunk_combos = list(itertools.product(chunks_y, chunks_x))
    else:
        leading_size = nav_shape_full[0]
        chunk_combos = [
            (t, cy, cx)
            for t in range(leading_size)
            for cy in chunks_y
            for cx in chunks_x
        ]

    # ── Shared memory handle ──────────────────────────────────────────────────
    shm_handle = None
    if shm_name is not None:
        from multiprocessing import shared_memory as _shm_mod
        try:
            shm_handle = _shm_mod.SharedMemory(name=shm_name, create=False)
            shm_buf = np.ndarray(nav_2d_shape, dtype=np.float32, buffer=shm_handle.buf)
        except Exception:
            shm_handle = None
            shm_buf = None
    else:
        shm_buf = None

    peaks_by_pos = [None] * n_patterns
    count_map = np.zeros(nav_2d_shape, dtype=np.int32)

    def _write_chunk_results(y0, y1, x0, x1, count_2d, peaks_list):
        """Unpack one chunk's results into count_map and peaks_by_pos."""
        cy, cx = count_2d.shape
        raster_idx = 0
        for ly in range(cy):
            for lx in range(cx):
                iy, ix = y0 + ly, x0 + lx
                peaks = peaks_list[raster_idx]
                raster_idx += 1
                flat_idx = iy * n_nav_x + ix
                peaks_by_pos[flat_idx] = peaks
                count_map[iy, ix] = len(peaks)
        nav_2d_slices = (slice(y0, y1), slice(x0, x1))
        if shm_buf is not None:
            try:
                shm_buf[nav_2d_slices] = count_map[nav_2d_slices].astype(np.float32)
            except Exception:
                pass
        if on_chunk_done is not None:
            try:
                on_chunk_done(nav_2d_slices, count_map[nav_2d_slices].copy())
            except Exception:
                pass

    # ── Path A: numpy — run peak-finding in this thread ───────────────────────
    if _is_numpy:
        for combo in chunk_combos:
            if stopped_flag is not None and stopped_flag[0]:
                break
            if nav_dim == 2:
                (y0, y1), (x0, x1) = combo
                py0 = max(0, y0 - depth_px); py1 = min(n_nav_y, y1 + depth_px)
                px0 = max(0, x0 - depth_px); px1 = min(n_nav_x, x1 + depth_px)
                chunk_raw = numpy_data[py0:py1, px0:px1]
            else:
                t, (y0, y1), (x0, x1) = combo
                py0 = max(0, y0 - depth_px); py1 = min(n_nav_y, y1 + depth_px)
                px0 = max(0, x0 - depth_px); px1 = min(n_nav_x, x1 + depth_px)
                chunk_raw = numpy_data[t, py0:py1, px0:px1]

            chunk_blurred = gaussian_filter(chunk_raw.astype(np.float32), sigma=sigma_tuple)
            vy0 = y0 - py0; vy1 = vy0 + (y1 - y0)
            vx0 = x0 - px0; vx1 = vx0 + (x1 - x0)
            chunk_valid = chunk_blurred[vy0:vy1, vx0:vx1]
            cy, cx = chunk_valid.shape[:2]

            count_2d = np.zeros((cy, cx), dtype=np.int32)
            peaks_list = []
            for ly in range(cy):
                for lx in range(cx):
                    _, _, peaks = _find_vectors_single_frame(
                        chunk_valid[ly, lx].astype(np.float32),
                        kernel_r, threshold, min_dist,
                        subpixel=subpixel, beamstop_mask=beamstop_mask,
                        _disk_fft=disk_fft, _disk_stats=disk_stats,
                    )
                    count_2d[ly, lx] = len(peaks)
                    peaks_list.append(peaks)

            _write_chunk_results(y0, y1, x0, x1, count_2d, peaks_list)

    # ── Path B: lazy dask or distributed Future — submit to worker ────────────
    else:
        # Get the distributed client from main_window
        client = getattr(main_window, 'dask_manager', None)
        client = getattr(client, 'client', None) if client else None

        # Convert dask array or Future to a single Future on the worker.
        # For a dask.array, client.compute() submits the graph and returns a
        # Future pointing to the result ON the worker — no data comes back yet.
        # For an already-computed Future, just use it directly.
        if _is_dask:
            data_future = client.compute(raw.astype(np.float32))
        else:
            data_future = raw  # already a Future on a worker

        # Submit each chunk as a separate task to the worker that holds data_future.
        # dask.distributed routes client.submit(fn, future, ...) to the worker
        # holding `future`, so the slice is purely local — zero TCP for bulk data.
        # Only the small result (count_2d + peaks_list, ~170KB per chunk) returns.
        # Map future -> (y0,y1,x0,x1) for O(1) lookup in the completion loop
        fut_to_coords = {}
        for combo in chunk_combos:
            if nav_dim == 2:
                (y0, y1), (x0, x1) = combo
            else:
                _, (y0, y1), (x0, x1) = combo  # ignore t for now (4D path)
            fut = client.submit(
                _worker_find_vectors_chunk,
                data_future,
                y0, y1, x0, x1,
                depth_px, n_nav_y, n_nav_x,
                sigma_tuple, kernel_r, threshold, min_dist, subpixel,
                beamstop_mask,
                pure=False,
            )
            fut_to_coords[fut] = (y0, y1, x0, x1)

        # Collect results as they complete — live shm updates happen here
        from dask.distributed import as_completed as _as_completed
        for fut in _as_completed(list(fut_to_coords)):
            if stopped_flag is not None and stopped_flag[0]:
                break
            y0, y1, x0, x1 = fut_to_coords[fut]
            count_2d, peaks_list = fut.result()
            _write_chunk_results(y0, y1, x0, x1, count_2d, peaks_list)

    if shm_handle is not None:
        try:
            shm_handle.close()
        except Exception:
            pass

    # ── Pack into CSR flat buffer ─────────────────────────────────────────────
    counts_flat = count_map.reshape(-1).astype(np.int64)
    offsets = np.zeros(n_patterns + 1, dtype=np.int64)
    np.cumsum(counts_flat, out=offsets[1:])
    N_total = int(offsets[-1])
    flat_buffer = np.zeros((N_total, 5), dtype=np.float32)

    if N_total > 0:
        # Concatenate all peak arrays in raster order and compute nav coords
        # for the whole buffer in one vectorised pass — O(N_total) numpy ops,
        # no Python loop over nav positions.
        all_peaks = []
        nav_x_col = np.empty(N_total, dtype=np.float32)
        nav_y_col = np.empty(N_total, dtype=np.float32)
        for flat_idx in range(n_patterns):
            peaks = peaks_by_pos[flat_idx]
            if peaks is None or len(peaks) == 0:
                continue
            s, e = offsets[flat_idx], offsets[flat_idx + 1]
            if e <= s:
                continue
            iy, ix = divmod(flat_idx, n_nav_x)
            nav_x_col[s:e] = ix
            nav_y_col[s:e] = iy
            all_peaks.append((s, e, peaks))

        # Bulk-write peak data for each run — still a loop but only over
        # non-empty positions and purely numpy slices inside.
        for s, e, peaks in all_peaks:
            flat_buffer[s:e, 2] = peaks[:, 1] * kx_scale + kx_offset  # kx_data
            flat_buffer[s:e, 3] = peaks[:, 0] * ky_scale + ky_offset  # ky_data
            flat_buffer[s:e, 4] = peaks[:, 2]                          # intensity

        flat_buffer[:, 0] = nav_x_col
        flat_buffer[:, 1] = nav_y_col

    return SpyDEDiffractionVectors(
        flat_buffer=flat_buffer,
        offsets=offsets,
        nav_shape=nav_2d_shape,
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

    relay.compute_done.connect(_on_compute_done)

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
        # Independent copy of the 4D signal (fresh axes_manager, independent cursor)
        try:
            raw_data = sig_ref.data
            if hasattr(raw_data, 'result'):
                raw_data = raw_data.result()
            new_sig = sig_ref._deepcopy_with_new_data(raw_data)
        except Exception:
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

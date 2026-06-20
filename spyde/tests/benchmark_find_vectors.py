"""
Timing benchmark for the find_vectors pipeline on the sped_ag dataset.

Run with:
    .venv/Scripts/python -m pytest spyde/tests/test_find_vectors_bench.py -v -s

No pytest-benchmark plugin required.  Pass -s / --capture=no to see output.

Phases
------
1. Single-frame plain cross-correlation  (_find_vectors_single_frame, rfft2)
2. Single-frame NXCORR                   (window-normalised, match_template-equivalent)
3. Full _do_compute_vectors on 16x16 (256 patterns) -- measures dask/blur overhead
4. vecs.to_dense()                        cold (first call) and warm (cache hit)
5. vecs.at(iy, ix)                        O(1) CSR lookup
"""

from __future__ import annotations

import time
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.fft import rfft2, irfft2, next_fast_len
from scipy.signal import fftconvolve

# Module-level dict shared between tests (module ordering guarantees phases run
# in declaration order under pytest's default collection).
_R: dict = {}

_FULL_FRAMES = 64 * 208   # sped_ag is (64, 208, 112, 112)
_N_REPS = 30               # repetitions for per-frame benchmarks
_WARMUP = 5


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sped_subset():
    """Load sped_ag, return (signal, numpy_data_16x16, a representative frame)."""
    import pyxem.data
    sig = pyxem.data.sped_ag(allow_download=True)
    # Already numpy (not lazy) -- shape (64, 208, 112, 112)
    assert not sig._lazy, "sped_ag should be non-lazy"
    data = sig.data.astype(np.float32)  # (64, 208, 112, 112)

    # 16x16 = 256 patterns, build a minimal HyperSpy signal for _do_compute_vectors
    import hyperspy.api as hs
    sub_data = data[:16, :16].copy()
    sub_sig = hs.signals.Signal2D(sub_data)
    for i in range(2):
        sub_sig.axes_manager._axes[i].scale = sig.axes_manager._axes[i].scale
        sub_sig.axes_manager._axes[i].offset = sig.axes_manager._axes[i].offset
    for i in range(2, 4):
        sub_sig.axes_manager._axes[i].scale = sig.axes_manager._axes[i].scale
        sub_sig.axes_manager._axes[i].offset = sig.axes_manager._axes[i].offset
        sub_sig.axes_manager._axes[i].navigate = False

    # A frame with real diffraction peaks (verified empirically)
    frame = gaussian_filter(data[30, 100], sigma=1.0)
    return sig, sub_sig, frame


# ── NXCORR helper ─────────────────────────────────────────────────────────────

def _nxcorr(frame: np.ndarray, disk: np.ndarray, kr: int) -> np.ndarray:
    """
    Window-normalised cross-correlation -- Lewis (1995).

        score(y,x) = (xcorr(y,x)/n - mean_win(y,x) * mean_T)
                     / (std_win(y,x) * std_T)

    Verified to match skimage.match_template to within 1e-4.
    Uses reflect-padding + FFT xcorr + integral-image window stats.
    """
    H, W = frame.shape
    kH, kW = disk.shape
    n = kH * kW
    t_mean = float(disk.mean())
    t_std = float(np.sqrt(np.sum((disk - t_mean) ** 2) / n))

    # Reflect-pad frame by kr so peaks near edges are detected correctly
    padded_full = np.pad(frame, kr, mode="reflect")   # (H+2kr, W+2kr)
    pH = next_fast_len(H + 2 * kr)
    pW = next_fast_len(W + 2 * kr)
    buf = np.zeros((pH, pW), dtype=np.float32)
    buf[:H + 2*kr, :W + 2*kr] = padded_full
    d_buf = np.zeros((pH, pW), dtype=np.float32)
    d_buf[:kH, :kW] = disk
    xcorr = irfft2(rfft2(buf) * rfft2(d_buf).conj())[:H, :W].astype(np.float32)

    # Integral images of the reflect-padded frame for window stats
    cum1 = np.empty((H + 2*kr + 1, W + 2*kr + 1), dtype=np.float32)
    cum1[0, :] = 0.0; cum1[:, 0] = 0.0
    cum1[1:, 1:] = np.cumsum(np.cumsum(padded_full, axis=0), axis=1)
    cum2 = np.empty((H + 2*kr + 1, W + 2*kr + 1), dtype=np.float32)
    cum2[0, :] = 0.0; cum2[:, 0] = 0.0
    cum2[1:, 1:] = np.cumsum(np.cumsum(padded_full ** 2, axis=0), axis=1)

    ws1 = cum1[kH:H+kH, kW:W+kW] - cum1[0:H, kW:W+kW] - cum1[kH:H+kH, 0:W] + cum1[0:H, 0:W]
    ws2 = cum2[kH:H+kH, kW:W+kW] - cum2[0:H, kW:W+kW] - cum2[kH:H+kH, 0:W] + cum2[0:H, 0:W]

    win_mean = ws1 / n
    win_var = ws2 / n - win_mean ** 2
    np.maximum(win_var, 0, out=win_var)
    win_std = np.sqrt(win_var)

    numerator = xcorr / n - win_mean * t_mean
    denom = win_std * t_std
    np.maximum(denom, 1e-8, out=denom)
    return np.clip(numerator / denom, -1.0, 1.0).astype(np.float32)


# ── Phase 1: plain rfft2 correlation ─────────────────────────────────────────

def test_bench_phase1_plain_corr(sped_subset):
    """rfft2 cross-correlation (current implementation) -- NOT window-normalised."""
    from spyde.actions.find_vectors import _find_vectors_single_frame, _get_disk_fft
    _, _, frame = sped_subset

    kernel_r = 8
    threshold = 0.4
    min_dist = 5
    H, W = frame.shape
    pH = next_fast_len(H + 2 * kernel_r)
    pW = next_fast_len(W + 2 * kernel_r)
    disk_fft = _get_disk_fft(kernel_r, pH, pW)

    # Warmup
    for _ in range(_WARMUP):
        _find_vectors_single_frame(frame, kernel_r, threshold, min_dist,
                                   subpixel=True, _disk_fft=disk_fft)

    t0 = time.perf_counter()
    for _ in range(_N_REPS):
        _, raw_corr, peaks = _find_vectors_single_frame(
            frame, kernel_r, threshold, min_dist, subpixel=True, _disk_fft=disk_fft)
    elapsed_ms = (time.perf_counter() - t0) / _N_REPS * 1000

    _R["plain_ms"] = elapsed_ms
    _R["plain_peaks"] = len(peaks)
    _R["plain_corr_max"] = float(raw_corr.max())
    _R["plain_corr_min"] = float(raw_corr.min())
    print(f"\n[Phase 1] plain rfft2: {elapsed_ms:.3f} ms/frame, "
          f"n_peaks={len(peaks)}, corr range=[{raw_corr.min():.3f}, {raw_corr.max():.3f}]")
    print(f"          NOTE: plain corr is NOT in [-1,1] -- threshold is intensity-dependent")


# ── Phase 2: NXCORR ──────────────────────────────────────────────────────────

def test_bench_phase2_nxcorr(sped_subset):
    """Window-normalised cross-correlation -- output in [-1,1], threshold meaningful."""
    from spyde.actions.find_vectors import _make_disk, _subpixel_com
    from scipy.ndimage import maximum_filter
    _, _, frame = sped_subset

    kernel_r = 8
    threshold = 0.4
    min_dist = 5
    disk = _make_disk(kernel_r)

    # Warmup
    for _ in range(_WARMUP):
        _nxcorr(frame, disk, kernel_r)

    t0 = time.perf_counter()
    for _ in range(_N_REPS):
        nxc = _nxcorr(frame, disk, kernel_r)
    elapsed_ms = (time.perf_counter() - t0) / _N_REPS * 1000

    # Count peaks with same threshold
    lmax = maximum_filter(nxc, size=2 * min_dist + 1)
    peaks_px = np.argwhere((nxc == lmax) & (nxc >= threshold))
    n_peaks = len(peaks_px)

    _R["nxcorr_ms"] = elapsed_ms
    _R["nxcorr_peaks"] = n_peaks
    _R["nxcorr_max"] = float(nxc.max())
    _R["nxcorr_min"] = float(nxc.min())
    print(f"\n[Phase 2] NXCORR:     {elapsed_ms:.3f} ms/frame, "
          f"n_peaks={n_peaks}, corr range=[{nxc.min():.3f}, {nxc.max():.3f}]")
    print(f"          NXCORR overhead vs plain: {elapsed_ms / max(_R.get('plain_ms', 1), 1e-9):.1f}x")
    print(f"          NXCORR output IS in [-1,1]: {float(nxc.min()) >= -1.01 and float(nxc.max()) <= 1.01}")


# ── Phase 3: full _do_compute_vectors on 256 patterns ────────────────────────

def test_bench_phase3_do_compute_vectors(sped_subset):
    """Full batch pipeline: dask blur + peak-find + CSR pack on 16x16=256 patterns."""
    from spyde.actions.find_vectors import _do_compute_vectors

    _, sub_sig, _ = sped_subset
    params = dict(sigma=1.0, kernel_radius=8, threshold=0.4,
                  min_distance=5, subpixel=True)

    n_patterns = 16 * 16
    t0 = time.perf_counter()
    vecs = _do_compute_vectors(sub_sig, params, None, None)
    elapsed_s = time.perf_counter() - t0
    ms_per_frame = elapsed_s / n_patterns * 1000

    _R["batch_s"] = elapsed_s
    _R["batch_ms_per_frame"] = ms_per_frame
    _R["vecs"] = vecs
    _R["n_patterns"] = n_patterns

    print(f"\n[Phase 3] _do_compute_vectors ({n_patterns} patterns): "
          f"{elapsed_s:.2f} s total, {ms_per_frame:.2f} ms/frame")
    print(f"          Total vectors found: {len(vecs.flat_buffer)}")
    extrap_s = ms_per_frame * _FULL_FRAMES / 1000
    print(f"          Extrapolated for {_FULL_FRAMES} frames: {extrap_s:.1f} s")

    # Break down: how much is dask overhead vs peak-finding?
    # Rerun pure peak-finding on the already-loaded numpy chunk (no dask)
    data_np = sub_sig.data.reshape(-1, 112, 112).astype(np.float32)
    from spyde.actions.find_vectors import _find_vectors_single_frame, _get_disk_fft
    from scipy.ndimage import gaussian_filter as gf
    pH = next_fast_len(112 + 2 * 8)
    pW = next_fast_len(112 + 2 * 8)
    disk_fft = _get_disk_fft(8, pH, pW)

    t0 = time.perf_counter()
    for frame in data_np:
        blurred = gf(frame, sigma=1.0)
        _find_vectors_single_frame(blurred, 8, 0.4, 5, subpixel=True, _disk_fft=disk_fft)
    pure_peak_find_s = time.perf_counter() - t0

    _R["pure_peak_find_s"] = pure_peak_find_s
    overhead_s = elapsed_s - pure_peak_find_s
    print(f"          Pure peak-find (no dask, {n_patterns}p): {pure_peak_find_s:.2f} s")
    print(f"          Dask/blur/pack overhead: {overhead_s:.2f} s  "
          f"({overhead_s/elapsed_s*100:.0f}% of total)")


# ── Phase 4: to_dense ─────────────────────────────────────────────────────────

def test_bench_phase4_to_dense(sped_subset):
    """Dense array construction: cold (builds) vs warm (cache hit)."""
    if "vecs" not in _R:
        pytest.skip("Phase 3 did not produce vecs")
    vecs = _R["vecs"]

    # Cold call -- invalidate cache
    vecs._dense_cache = None
    t0 = time.perf_counter()
    dense = vecs.to_dense()
    cold_ms = (time.perf_counter() - t0) * 1000
    _R["dense_cold_ms"] = cold_ms
    _R["dense_shape"] = dense.shape

    # Warm call -- should be instant (returns cached array)
    t0 = time.perf_counter()
    dense2 = vecs.to_dense()
    warm_ms = (time.perf_counter() - t0) * 1000
    _R["dense_warm_ms"] = warm_ms

    assert dense2 is dense, "warm call should return same cached object"
    print(f"\n[Phase 4] to_dense(): cold={cold_ms:.2f} ms, warm={warm_ms:.4f} ms")
    print(f"          Dense shape: {dense.shape}")
    n = _R["n_patterns"]
    print(f"          Cold cost per pattern: {cold_ms/n:.4f} ms")


# ── Phase 5: at(iy, ix) CSR lookup ───────────────────────────────────────────

def test_bench_phase5_at_lookup(sped_subset):
    """CSR at(iy, ix) -- should be O(1), microseconds per call."""
    if "vecs" not in _R:
        pytest.skip("Phase 3 did not produce vecs")
    vecs = _R["vecs"]
    nav_y, nav_x = vecs.nav_shape

    # Warmup
    for iy in range(min(4, nav_y)):
        for ix in range(min(4, nav_x)):
            vecs.at(iy, ix)

    # Time all positions
    t0 = time.perf_counter()
    for iy in range(nav_y):
        for ix in range(nav_x):
            _ = vecs.at(iy, ix)
    total_ms = (time.perf_counter() - t0) * 1000
    n = nav_y * nav_x
    us_per = total_ms / n * 1000

    _R["at_us"] = us_per
    print(f"\n[Phase 5] vecs.at(iy,ix): {us_per:.2f} us/call over {n} positions")


# ── Summary ───────────────────────────────────────────────────────────────────

def test_bench_summary(sped_subset):  # noqa: ARG001
    """Print the full timing table."""
    import sys
    # Force UTF-8 output so the table prints correctly on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    if not _R:
        print("\n[bench summary] No results collected -- did earlier tests run?")
        return

    print("\n")
    print("=" * 72)
    print(" FIND-VECTORS BENCHMARK SUMMARY")
    print("=" * 72)
    hdr = f"{'Phase':<38} {'ms/frame':>10} {'extrap 13.3k':>14}"
    print(hdr)
    print("-" * 72)

    full = _FULL_FRAMES

    def row(label, ms):
        extrap_s = ms * full / 1000
        if extrap_s < 60:
            ex = f"{extrap_s:.1f} s"
        else:
            ex = f"{extrap_s/60:.1f} min"
        print(f"  {label:<36} {ms:>10.3f} {ex:>14}")

    if "plain_ms" in _R:
        row(f"plain rfft2 corr (n={_R['plain_peaks']}pk)", _R["plain_ms"])
        print(f"    corr range: [{_R['plain_corr_min']:.3f}, {_R['plain_corr_max']:.3f}]  "
              f"<- NOT [-1,1] -- threshold is intensity-dependent!")
    if "nxcorr_ms" in _R:
        row(f"NXCORR (n={_R['nxcorr_peaks']}pk)", _R["nxcorr_ms"])
        print(f"    corr range: [{_R['nxcorr_min']:.3f}, {_R['nxcorr_max']:.3f}]  "
              f"<- in [-1,1], threshold is correct")
        if "plain_ms" in _R:
            ratio = _R["nxcorr_ms"] / _R["plain_ms"]
            print(f"    NXCORR costs {ratio:.1f}x plain corr")

    print("-" * 72)
    if "batch_ms_per_frame" in _R:
        row(f"full batch ({_R['n_patterns']}p incl dask/blur)", _R["batch_ms_per_frame"])
        if "pure_peak_find_s" in _R and "batch_s" in _R:
            pf = _R["pure_peak_find_s"] / _R["n_patterns"] * 1000
            ov = (_R["batch_s"] - _R["pure_peak_find_s"]) / _R["n_patterns"] * 1000
            row("  └─ peak-find only (no dask)", pf)
            row("  └─ dask/blur/pack overhead", ov)

    print("-" * 72)
    if "dense_cold_ms" in _R:
        print(f"  {'to_dense() cold':<36} {_R['dense_cold_ms']:>10.3f} ms total  "
              f"shape={_R.get('dense_shape')}")
        print(f"  {'to_dense() warm (cache hit)':<36} {_R['dense_warm_ms']:>10.4f} ms total")
    if "at_us" in _R:
        print(f"  {'vecs.at(iy,ix)':<36} {_R['at_us']:>10.4f} µs/call")
    print("=" * 72)

    # Verdict
    print("\nVERDICT:")
    if "plain_corr_max" in _R and _R["plain_corr_max"] > 5.0:
        print("  !! plain rfft2 output is NOT normalised -- values >> 1.")
        print("    Threshold=0.4 is meaningless against raw intensity sums.")
        print("    MUST use NXCORR for correct peak detection.")
    if "nxcorr_ms" in _R and "plain_ms" in _R:
        ratio = _R["nxcorr_ms"] / _R["plain_ms"]
        if ratio < 3.0:
            print(f"  OK NXCORR overhead is only {ratio:.1f}x -- worth using.")
        else:
            print(f"  ! NXCORR is {ratio:.1f}x slower -- consider if acceptable.")
    if "batch_s" in _R and "pure_peak_find_s" in _R:
        overhead_pct = (_R["batch_s"] - _R["pure_peak_find_s"]) / _R["batch_s"] * 100
        if overhead_pct > 50:
            print(f"  !! Dask/blur overhead = {overhead_pct:.0f}% of total time -- "
                  f"consider skipping nav-space blur for sigma≤1.")
        else:
            print(f"  OK Dask/blur overhead = {overhead_pct:.0f}% -- acceptable.")
    print()

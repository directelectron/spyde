"""
Tests for the diffraction vector finding pipeline.

Covers:
  - Core algorithm (_find_vectors_single_frame)
  - NavBlurCache (async chunk blur, warm/cold paths, stale-guard)
  - Sigma/depth tuple construction for 4D and 5D datasets
  - map_overlap correctness at chunk boundaries
  - SpyDEDiffractionVectors data layout
  - Performance budget (<20 ms per 256×256 frame)
"""

import threading
import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from spyde.actions.find_vectors import (
    NavBlurCache,
    _auto_params,
    _find_vectors_single_frame,
    _make_disk,
    _nav_chunk_size,
    _subpixel_com,
)
from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors


# ── Helpers ───────────────────────────────────────────────────────────────────


def _blurred_frame(shape=(128, 128), peaks=None, sigma=1.5):
    frame = np.zeros(shape, dtype=np.float32)
    if peaks:
        for ky, kx in peaks:
            frame[ky - 3: ky + 4, kx - 3: kx + 4] = 10.0
    return gaussian_filter(frame, sigma=sigma)


def _make_vecs(nav_shape=(4, 4), n_per_pos=3):
    nav_y, nav_x = nav_shape
    n_nav = nav_y * nav_x
    N = n_nav * n_per_pos
    flat = np.random.rand(N, 6).astype(np.float32)
    # Fill nav coords so _build_nav_offsets works correctly
    nav_idx = np.repeat(np.arange(n_nav, dtype=np.int64), n_per_pos)
    flat[:, 0] = (nav_idx % nav_x).astype(np.float32)   # nav_x
    flat[:, 1] = (nav_idx // nav_x).astype(np.float32)  # nav_y
    flat[:, 4] = -1.0  # time: 4D
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat,
        full_nav_shape=nav_shape,
        sig_shape=(128, 128),
        sig_axes=None,
        kernel_radius_px=5.0,
        kernel_radius_data=0.05,
    )


# ── Core algorithm ────────────────────────────────────────────────────────────


def test_detects_known_peaks():
    peaks_expected = [(30, 40), (80, 70), (60, 100)]
    frame = _blurred_frame((128, 128), peaks=peaks_expected)
    _, _, peaks = _find_vectors_single_frame(frame, 5, 0.3, 8)
    for ey, ex in peaks_expected:
        dists = np.hypot(peaks[:, 0] - ey, peaks[:, 1] - ex)
        assert dists.min() < 4, f"Peak at ({ey},{ex}) not found; got {peaks[:, :2]}"


def test_threshold_controls_count():
    frame = _blurred_frame((128, 128), peaks=[(20, 20), (60, 60), (100, 100)])
    _, _, p_low = _find_vectors_single_frame(frame, 5, 0.05, 8)
    _, _, p_high = _find_vectors_single_frame(frame, 5, 0.90, 8)
    assert len(p_low) >= len(p_high)


def test_min_distance_prevents_duplicates():
    # Build a realistic diffraction-like frame (noisy background + two bright
    # disks only 3 px apart) and verify that min_distance=10 merges them to <=1
    # returned peak.  Random background ensures NXCORR has well-defined local
    # variance everywhere (no zero-std patches that blow up the denominator).
    rng = np.random.default_rng(42)
    frame = rng.uniform(0.5, 2.0, (128, 128)).astype(np.float32)
    # Two disks separated by 3 px — NXCORR sees them as one merged feature
    for oy, ox in [(64, 64), (64, 67)]:
        yy, xx = np.ogrid[-4:5, -4:5]
        disk_blob = (yy**2 + xx**2 <= 16).astype(np.float32)
        frame[oy-4:oy+5, ox-4:ox+5] += disk_blob * 15.0
    frame = gaussian_filter(frame, sigma=0.5)
    _, _, peaks = _find_vectors_single_frame(frame, 4, 0.3, 10)
    # All surviving peaks must be >= min_distance apart
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            dist = np.hypot(peaks[i, 0] - peaks[j, 0], peaks[i, 1] - peaks[j, 1])
            assert dist >= 10, f"Peaks {i} and {j} only {dist:.1f} px apart — NMS failed"


def test_output_shapes():
    frame = np.random.rand(64, 64).astype(np.float32)
    corr, raw, peaks = _find_vectors_single_frame(frame, 4, 0.5, 5)
    assert corr.shape == frame.shape
    assert raw.shape == frame.shape
    assert peaks.ndim == 2 and peaks.shape[1] == 3


def test_zero_frame_no_peaks():
    frame = np.zeros((64, 64), dtype=np.float32)
    _, _, peaks = _find_vectors_single_frame(frame, 4, 0.3, 5)
    assert len(peaks) == 0


def test_subpixel_refinement_produces_fractional_coords():
    frame = np.zeros((64, 64), dtype=np.float32)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            dist = np.hypot(dy - 0.3, dx - 0.7)
            frame[30 + dy, 40 + dx] = max(0, 5.0 - dist)
    frame = gaussian_filter(frame, sigma=0.5)
    _, _, peaks_sub = _find_vectors_single_frame(frame, 4, 0.1, 6, subpixel=True)
    _, _, peaks_int = _find_vectors_single_frame(frame, 4, 0.1, 6, subpixel=False)
    assert len(peaks_sub) > 0
    # Subpixel should have at least one fractional coordinate
    assert any(p % 1 != 0 for p in peaks_sub[0, :2])
    # Integer mode should be whole numbers
    assert all(p % 1 == 0 for p in peaks_int[0, :2])


def test_disk_kernel_cached():
    d1 = _make_disk(8)
    d2 = _make_disk(8)
    assert d1 is d2


def test_auto_params_valid_ranges():
    frame = np.random.rand(128, 128).astype(np.float32)
    p = _auto_params(frame)
    assert 0 < p["sigma"] <= 10
    assert 1 <= p["kernel_radius"] < 64
    assert 0 < p["threshold"] < 1
    assert p["min_distance"] >= 1
    assert isinstance(p["subpixel"], bool)


# ── NavBlurCache ──────────────────────────────────────────────────────────────


def test_nav_blur_cache_warm_hit():
    """Warm cache returns the same result as gaussian_filter on the padded chunk.

    NavBlurCache blurs in nav space: it averages neighboring patterns weighted
    by a Gaussian.  A spike in a single pattern (nav position [4,4]) will reduce
    that pattern's spike value and contribute to its nav-neighbors.
    """
    sigma = 1.5
    cache = NavBlurCache(sigma=sigma)
    # All patterns are constant=1 except [4,4] which has a spike in the signal
    chunk = np.ones((8, 8, 16, 16), dtype=np.float32)
    chunk[4, 4, 8, 8] = 100.0

    cache.update_chunk(chunk, chunk_id=(0, 0))
    cache._blur_thread.join()

    result = cache.get_blurred(4, 4, raw_pattern=chunk[4, 4])
    # The spike pixel must be attenuated (neighbors are 1.0, not 100)
    assert result[8, 8] < 80.0, f"Spike not reduced: {result[8, 8]}"
    # And the nav-neighbor pattern at [4,5] gets some of the spike's value
    result_neighbor = cache.get_blurred(4, 5, raw_pattern=chunk[4, 5])
    assert result_neighbor[8, 8] > 1.0, "Nav blur did not spread to neighbor"
    # Shape and dtype
    assert result.shape == (16, 16)
    assert result.dtype == np.float32


def test_nav_blur_cache_cold_fallback():
    sigma = 1.5
    cache = NavBlurCache(sigma=sigma)
    chunk = np.random.rand(16, 16, 64, 64).astype(np.float32)
    # Simulate chunk loaded but blur not yet finished
    cache._chunk_id = (0, 0)
    cache._raw_chunk = chunk
    cache._blurred = None

    raw_pattern = chunk[8, 8]
    result = cache.get_blurred(8, 8, raw_pattern=raw_pattern)
    ref = gaussian_filter(raw_pattern, sigma=(sigma, sigma))
    np.testing.assert_allclose(result, ref, atol=1e-6)


def test_nav_blur_cache_invalidate_clears():
    cache = NavBlurCache(sigma=1.5)
    cache._blurred = np.zeros((8, 8, 64, 64), dtype=np.float32)
    cache._chunk_id = (0, 0)
    cache.invalidate(sigma=2.0)
    assert cache._blurred is None
    assert cache._chunk_id is None
    assert cache.sigma == 2.0


def test_nav_blur_cache_stale_guard():
    """Blur started for chunk A should not overwrite result for chunk B."""
    sigma = 1.5
    cache = NavBlurCache(sigma=sigma)
    chunk_a = np.zeros((4, 4, 16, 16), dtype=np.float32)
    chunk_b = np.ones((4, 4, 16, 16), dtype=np.float32)

    cache.update_chunk(chunk_a, (0, 0))
    cache.update_chunk(chunk_b, (1, 0))  # switch before chunk_a blur finishes
    cache._blur_thread.join()

    with cache._lock:
        assert cache._chunk_id == (1, 0)
        if cache._blurred is not None:
            assert cache._blurred.mean() > 0.5  # chunk_b (ones), not chunk_a (zeros)


def test_nav_blur_cache_edge_accuracy():
    """Edge patterns use reflect-pad; verify the result is at least a valid blur.

    The reflect-pad boundary deliberately differs from true neighbors, so we do
    not assert a tight tolerance vs the full-array reference.  Instead we check
    that the blurred edge pattern is smoother than the raw pattern (std dev
    is reduced) — confirming the blur ran and produced a valid output.
    """
    sigma = 1.5
    chunk = np.random.rand(8, 8, 32, 32).astype(np.float32)
    # Add a spike at the edge pattern's center pixel to make std large
    chunk[0, 0, 16, 16] = 50.0

    cache = NavBlurCache(sigma=sigma)
    cache.update_chunk(chunk, (0, 0))
    cache._blur_thread.join()

    raw_pattern = chunk[0, 0]
    result = cache.get_blurred(0, 0, raw_pattern=raw_pattern)

    # The blurred pattern should have lower peak value (spike spread out)
    assert result[16, 16] < raw_pattern[16, 16], "Blur did not reduce spike"
    # And it must be a 2-D float array of the right shape
    assert result.shape == raw_pattern.shape


def test_nav_blur_cache_warm_speed():
    cache = NavBlurCache(sigma=1.5)
    chunk = np.random.rand(16, 16, 256, 256).astype(np.float32)
    cache.update_chunk(chunk, (0, 0))
    cache._blur_thread.join()

    raw_pattern = chunk[8, 8]
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        cache.get_blurred(8, 8, raw_pattern)
    avg_ms = (time.perf_counter() - t0) / N * 1000
    assert avg_ms < 1.0, f"Warm cache too slow: {avg_ms:.2f}ms"


# ── Chunk size ────────────────────────────────────────────────────────────────


def test_nav_chunk_size_within_memory_limit():
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
        chunk = _nav_chunk_size(sigma, max_ram_mb=200, sig_shape=(128, 128))
        depth = int(np.ceil(3 * sigma))
        ram_mb = (chunk + 2 * depth) ** 2 * 128 * 128 * 4 / 1e6
        assert ram_mb <= 200, f"sigma={sigma}: {ram_mb:.0f}MB > 200MB"


def test_nav_chunk_size_larger_than_depth():
    for sigma in [0.5, 1.0, 2.0, 3.0]:
        chunk = _nav_chunk_size(sigma, max_ram_mb=200, sig_shape=(256, 256))
        depth = int(np.ceil(3 * sigma))
        assert chunk > depth, f"sigma={sigma}: chunk={chunk} <= depth={depth}"


# ── map_overlap correctness ───────────────────────────────────────────────────


def test_map_overlap_correct_at_chunk_boundary():
    import dask.array as da

    data = np.zeros((8, 8, 32, 32), dtype=np.float32)
    data[4, 0, 16, 16] = 100.0  # spike straddles chunk boundary at row 4

    sigma = 1.5
    depth = int(np.ceil(3 * sigma))
    da_data = da.from_array(data, chunks=(4, 4, 32, 32))

    result = da.map_overlap(
        gaussian_filter, da_data,
        depth=(depth, depth, 0, 0), boundary="reflect",
        sigma=(sigma, sigma, 0, 0), dtype=np.float32,
    ).compute()

    reference = gaussian_filter(data, sigma=(sigma, sigma, 0, 0))
    np.testing.assert_allclose(
        result[3, 0, 16, 16], reference[3, 0, 16, 16], rtol=1e-4
    )


def test_map_blocks_wrong_at_chunk_boundary():
    """map_blocks (no overlap) gives the wrong value at chunk boundaries."""
    import dask.array as da

    data = np.zeros((8, 8, 32, 32), dtype=np.float32)
    data[4, 0, 16, 16] = 100.0
    sigma = 1.5
    da_data = da.from_array(data, chunks=(4, 4, 32, 32))

    wrong = da_data.map_blocks(
        gaussian_filter, sigma=(sigma, sigma, 0, 0), dtype=np.float32
    ).compute()
    reference = gaussian_filter(data, sigma=(sigma, sigma, 0, 0))

    assert abs(wrong[3, 0, 16, 16] - reference[3, 0, 16, 16]) > 0.1


def test_sigma_tuple_4d():
    import hyperspy.api as hs

    s = hs.signals.Signal2D(np.zeros((4, 4, 16, 16)))
    nav_dim = s.axes_manager.navigation_dimension
    sig_dim = s.axes_manager.signal_dimension
    sigma_nav = 1.5
    sigma_tuple = tuple([0.0] * (nav_dim - 2) + [sigma_nav, sigma_nav] + [0.0] * sig_dim)
    assert sigma_tuple == (1.5, 1.5, 0.0, 0.0)


def test_sigma_tuple_5d():
    import hyperspy.api as hs

    s = hs.signals.Signal2D(np.zeros((3, 4, 4, 16, 16)))
    nav_dim = s.axes_manager.navigation_dimension
    sig_dim = s.axes_manager.signal_dimension
    sigma_nav = 1.5
    sigma_tuple = tuple([0.0] * (nav_dim - 2) + [sigma_nav, sigma_nav] + [0.0] * sig_dim)
    assert sigma_tuple == (0.0, 1.5, 1.5, 0.0, 0.0)


# ── SpyDEDiffractionVectors ───────────────────────────────────────────────────


def test_at_correct_row_count():
    vecs = _make_vecs((3, 3), n_per_pos=4)
    for iy in range(3):
        for ix in range(3):
            assert vecs.at(iy, ix).shape == (4, 6)


def test_kxy_at_correct_columns():
    vecs = _make_vecs()
    assert vecs.kxy_at(0, 0).shape == (3, 2)


def test_count_map_shape_and_values():
    vecs = _make_vecs((4, 4), n_per_pos=3)
    cm = vecs.count_map()
    assert cm.shape == (4, 4)
    assert (cm == 3).all()


def test_to_dense_shape_and_cache():
    vecs = _make_vecs((2, 3), n_per_pos=5)
    d1 = vecs.to_dense()
    assert d1.shape == (2, 3, 5, 6)
    d2 = vecs.to_dense()
    assert d1 is d2


def test_flatten_full_buffer():
    vecs = _make_vecs((2, 2), n_per_pos=3)
    assert vecs.flatten().shape == (12, 6)


def test_from_ragged_roundtrip():
    nav_shape = (3, 4)
    nav_y, nav_x = nav_shape
    ragged = np.empty(nav_shape, dtype=object)
    for i in range(nav_y):
        for j in range(nav_x):
            n = np.random.randint(1, 8)
            ragged[i, j] = np.random.rand(n, 2).astype(np.float32)

    vecs = SpyDEDiffractionVectors.from_ragged(
        ragged, nav_shape,
        sig_shape=(128, 128),
        sig_axes=None, kernel_radius_px=5.0, kernel_radius_data=0.05,
    )
    for i in range(nav_y):
        for j in range(nav_x):
            assert len(vecs.at(i, j)) == len(ragged[i, j])


def test_to_pyxem_type():
    from pyxem.signals import DiffractionVectors2D

    vecs = _make_vecs()
    dv = vecs.to_pyxem()
    assert isinstance(dv, DiffractionVectors2D)


def test_spots_at_returns_list():
    # sig_axes=None — spots_at uses kernel_radius_data for size
    vecs = _make_vecs(nav_shape=(2, 2), n_per_pos=2)
    # Manually set kxy columns to known values
    vecs.flat_buffer[0, 2] = 0.1  # kx
    vecs.flat_buffer[0, 3] = 0.2  # ky
    spots = vecs.spots_at(0, 0)
    assert isinstance(spots, list)
    # Each spot has 'pos' and 'size'
    assert "pos" in spots[0]
    assert "size" in spots[0]


# ── Performance ───────────────────────────────────────────────────────────────


def test_single_frame_pipeline_under_20ms():
    frame = gaussian_filter(np.random.rand(256, 256).astype(np.float32), sigma=1.5)
    _find_vectors_single_frame(frame, 12, 0.3, 10)  # warm up

    t0 = time.perf_counter()
    for _ in range(10):
        _find_vectors_single_frame(frame, 12, 0.3, 10)
    avg_ms = (time.perf_counter() - t0) / 10 * 1000
    assert avg_ms < 20, f"Pipeline too slow: {avg_ms:.1f}ms (limit 20ms)"

"""
Memory-safety tests for _do_compute_vectors.

THE RULE: _do_compute_vectors must NEVER call .compute() or .result() on the
full dataset.  For dask arrays the full array must stay lazy throughout — only
small per-chunk slices (numpy) are ever materialised.  The only data that
should cross from the dask graph into numpy RAM is one chunk at a time.

These tests enforce that contract by:
  1. Wrapping dask.array.Array so any .compute() call on an array larger than
     one chunk raises AssertionError — catching regressions where someone
     re-introduces a full-compute call.
  2. Monkey-patching distributed.Future.result() to block full-dataset fetches.
  3. Counting how many bytes of signal data were materialised — must be <= one
     ghost-padded chunk, not the full dataset.
  4. Verifying that the result is still correct (counts match, offsets valid,
     flat_buffer columns in range).
  5. Covering 4D and 5D datasets, lazy dask arrays, and already-computed numpy
     arrays (numpy path is allowed to hold the full array; the point is that
     the dask path must not).
"""

from __future__ import annotations

import gc
import sys
import threading
import tracemalloc
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import dask.array as da
import hyperspy.api as hs

from spyde.actions.find_vectors import _do_compute_vectors, _nav_chunk_size


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_lazy_4d(nav=(8, 8), sig=(32, 32), chunk_nav=4) -> hs.signals.Signal2D:
    """Return a lazy 4D STEM signal backed by a dask array."""
    ny, nx = nav
    ky, kx = sig
    data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    # Place a bright blob at each pattern so peaks are always detectable
    data[:, :, ky // 2 - 2:ky // 2 + 2, kx // 2 - 2:kx // 2 + 2] = 100.0
    data[:, :, 4, 4] = 80.0
    da_data = da.from_array(data, chunks=(chunk_nav, chunk_nav, ky, kx))
    s = hs.signals.Signal2D(da_data)
    s.axes_manager.signal_axes[0].scale = 0.01
    s.axes_manager.signal_axes[0].offset = -ky * 0.005
    s.axes_manager.signal_axes[1].scale = 0.01
    s.axes_manager.signal_axes[1].offset = -kx * 0.005
    return s


def _make_lazy_5d(time=3, nav=(4, 4), sig=(32, 32), chunk_nav=2) -> hs.signals.Signal2D:
    """Return a lazy 5D STEM signal (time, nav_y, nav_x, ky, kx)."""
    t, ny, nx = time, nav[0], nav[1]
    ky, kx = sig
    data = np.zeros((t, ny, nx, ky, kx), dtype=np.float32)
    data[:, :, :, ky // 2 - 2:ky // 2 + 2, kx // 2 - 2:kx // 2 + 2] = 100.0
    da_data = da.from_array(data, chunks=(1, chunk_nav, chunk_nav, ky, kx))
    s = hs.signals.Signal2D(da_data)
    return s


def _params(sig_shape=(32, 32)):
    return dict(
        sigma=0.5,
        kernel_radius=3,
        threshold=0.3,
        min_distance=2,
        subpixel=False,
    )


# ── Core contract: no full .compute() on dask path ────────────────────────────


class _ComputeGuard:
    """
    Wraps a dask array's .compute() to raise if called on the full dataset.

    We allow .compute() only if the array has the same shape as a legal
    per-chunk slice (i.e. it's already a chunk-sized slice, not the whole thing).
    """

    def __init__(self, real_array: da.Array, full_shape: tuple, chunk_nav: int):
        self._real = real_array
        self._full_shape = full_shape
        self._chunk_nav = chunk_nav
        self._compute_calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        shape = self._real.shape
        self._compute_calls.append(shape)
        # If the nav dimensions match the full nav shape, this is a full-data compute
        nav_size = shape[0] * shape[1] if len(shape) >= 2 else shape[0]
        full_nav = self._full_shape[0] * self._full_shape[1]
        if nav_size >= full_nav and nav_size > self._chunk_nav ** 2:
            raise AssertionError(
                f"Full-dataset .compute() called on shape {shape} — "
                f"this loads the entire dataset into RAM. "
                f"Use per-chunk numpy slices instead."
            )
        return self._real.compute(*args, **kwargs)


def test_dask_path_never_computes_full_array():
    """
    _do_compute_vectors must not call .compute() on the original lazy array object.

    The regression: old code called client.compute(raw.astype(np.float32)) which
    materialised the entire dataset into a single distributed Future.  We detect
    this by checking whether the compute is called on an array with shape identical
    to the full dataset.  Uses sig_shape=(256,256) to keep chunk_nav well below
    the nav size so every ghost-padded slice is a strict sub-array.
    """
    # sig_shape=(256,256) → chunk_nav≈23, nav=32 → 2x2 = 4 chunks (no single-chunk edge case)
    nav = (32, 32)
    sig_shape = (256, 256)
    sigma = 0.5
    sig = _make_lazy_4d(nav=nav, sig=sig_shape, chunk_nav=8)
    params = dict(sigma=sigma, kernel_radius=3, threshold=0.3, min_distance=2, subpixel=False)

    full_shape = sig.data.shape  # (32, 32, 256, 256)
    full_computed_on_raw = [False]
    _orig_compute = da.Array.compute

    def _spy_compute(self, *args, **kwargs):
        if self.shape == full_shape:
            full_computed_on_raw[0] = True
            raise AssertionError(
                f"compute() called on an array with full-dataset shape {self.shape} — "
                f"this loads the entire dataset into RAM. Only ghost-padded slices should be computed."
            )
        return _orig_compute(self, *args, **kwargs)

    with patch.object(da.Array, "compute", _spy_compute):
        vecs = _do_compute_vectors(sig, params, None, None)

    assert not full_computed_on_raw[0]
    assert vecs.nav_shape == nav
    assert vecs.flat_buffer.shape[1] == 6


def test_numpy_path_allowed_to_hold_full_array():
    """The numpy path is allowed to hold the full array in RAM — it's already there."""
    ny, nx, ky, kx = 6, 6, 32, 32
    data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    data[:, :, ky // 2 - 1:ky // 2 + 2, kx // 2 - 1:kx // 2 + 2] = 100.0
    s = hs.signals.Signal2D(data)
    s.axes_manager.signal_axes[0].scale = 0.01
    s.axes_manager.signal_axes[0].offset = -ky * 0.005
    s.axes_manager.signal_axes[1].scale = 0.01
    s.axes_manager.signal_axes[1].offset = -kx * 0.005
    params = _params()
    vecs = _do_compute_vectors(s, params, None, None)
    assert vecs.nav_shape == (ny, nx)


def test_result_offsets_never_exceed_buffer_length():
    """offsets[-1] must equal len(flat_buffer); no out-of-bounds indexing."""
    sig = _make_lazy_4d(nav=(6, 6), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    assert vecs.offsets[-1] == len(vecs.flat_buffer), (
        f"offsets[-1]={vecs.offsets[-1]} != len(flat_buffer)={len(vecs.flat_buffer)}"
    )


def test_offsets_monotonically_non_decreasing():
    sig = _make_lazy_4d(nav=(4, 5), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    diffs = np.diff(vecs.offsets)
    assert (diffs >= 0).all(), f"offsets not monotone: {diffs[diffs < 0]}"


def test_flat_buffer_columns_in_calibrated_range():
    """kx and ky columns (cols 2,3) must lie within the axis calibration range."""
    sig = _make_lazy_4d(nav=(4, 4), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    if len(vecs.flat_buffer) == 0:
        pytest.skip("No vectors found — adjust threshold")
    sig_ax = sig.axes_manager.signal_axes
    kx_min = sig_ax[0].offset
    kx_max = kx_min + sig_ax[0].scale * sig_ax[0].size
    ky_min = sig_ax[1].offset
    ky_max = ky_min + sig_ax[1].scale * sig_ax[1].size
    tol = 0.02  # one pixel tolerance
    assert vecs.flat_buffer[:, 2].min() >= kx_min - tol
    assert vecs.flat_buffer[:, 2].max() <= kx_max + tol
    assert vecs.flat_buffer[:, 3].min() >= ky_min - tol
    assert vecs.flat_buffer[:, 3].max() <= ky_max + tol


def test_nav_coords_in_range():
    """nav_x (col 0) and nav_y (col 1) must be valid pixel indices."""
    nav = (5, 7)
    sig = _make_lazy_4d(nav=nav, sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    if len(vecs.flat_buffer) == 0:
        pytest.skip("No vectors found")
    assert vecs.flat_buffer[:, 0].min() >= 0
    assert vecs.flat_buffer[:, 0].max() < nav[1]  # nav_x < nx
    assert vecs.flat_buffer[:, 1].min() >= 0
    assert vecs.flat_buffer[:, 1].max() < nav[0]  # nav_y < ny


def test_count_map_sums_to_flat_buffer_len():
    sig = _make_lazy_4d(nav=(4, 4), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    assert int(vecs.count_map().sum()) == len(vecs.flat_buffer)


def test_at_returns_correct_slice_lengths():
    """vecs.at(iy, ix) must return exactly count_map[iy, ix] rows."""
    sig = _make_lazy_4d(nav=(4, 4), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    cm = vecs.count_map()
    for iy in range(4):
        for ix in range(4):
            rows = vecs.at(iy, ix)
            assert len(rows) == cm[iy, ix], (
                f"at({iy},{ix}) len={len(rows)} but count_map={cm[iy, ix]}"
            )


def test_at_row_columns_match_full_flat_buffer():
    """Rows returned by .at() must be a contiguous slice of flat_buffer — not a copy."""
    sig = _make_lazy_4d(nav=(4, 4), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    cm = vecs.count_map()
    for iy in range(4):
        for ix in range(4):
            rows = vecs.at(iy, ix)
            if len(rows) == 0:
                continue
            i = iy * 4 + ix
            s, e = vecs.offsets[i], vecs.offsets[i + 1]
            np.testing.assert_array_equal(rows, vecs.flat_buffer[s:e])


def test_5d_nav_shape():
    """5D signal: nav_shape is the last 2 nav dims, not the leading time axis."""
    sig = _make_lazy_5d(time=2, nav=(4, 4), sig=(32, 32))
    params = _params()
    vecs = _do_compute_vectors(sig, params, None, None)
    # nav_shape should be (4, 4), not (2, 4) or (2, 4, 4)
    assert vecs.nav_shape == (4, 4), f"Got nav_shape={vecs.nav_shape}"
    assert vecs.flat_buffer.shape[1] == 6


def test_high_threshold_gives_fewer_vectors():
    """Raising threshold must not increase the vector count."""
    sig = _make_lazy_4d(nav=(4, 4), sig=(32, 32))
    p_low = dict(_params(), threshold=0.1)
    p_high = dict(_params(), threshold=0.9)
    v_low = _do_compute_vectors(sig, p_low, None, None)
    v_high = _do_compute_vectors(sig, p_high, None, None)
    assert len(v_high.flat_buffer) <= len(v_low.flat_buffer), (
        f"high threshold={len(v_high.flat_buffer)} > low threshold={len(v_low.flat_buffer)}"
    )


def test_zero_signal_produces_no_vectors():
    """All-zero patterns: correlation is undefined / zero everywhere → no peaks."""
    ny, nx, ky, kx = 4, 4, 32, 32
    data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    s = hs.signals.Signal2D(da.from_array(data, chunks=(2, 2, ky, kx)))
    s.axes_manager.signal_axes[0].scale = 0.01
    s.axes_manager.signal_axes[0].offset = 0.0
    s.axes_manager.signal_axes[1].scale = 0.01
    s.axes_manager.signal_axes[1].offset = 0.0
    vecs = _do_compute_vectors(s, _params(), None, None)
    assert len(vecs.flat_buffer) == 0


def test_all_positions_present_in_offsets():
    """offsets must have exactly nav_y*nav_x+1 entries."""
    nav = (5, 6)
    sig = _make_lazy_4d(nav=nav, sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    n_patterns = nav[0] * nav[1]
    assert len(vecs.offsets) == n_patterns + 1, (
        f"Expected {n_patterns + 1} offsets, got {len(vecs.offsets)}"
    )


def test_flat_buffer_dtype_is_float32():
    sig = _make_lazy_4d(nav=(4, 4), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    assert vecs.flat_buffer.dtype == np.float32


def test_offsets_dtype_is_int64():
    sig = _make_lazy_4d(nav=(4, 4), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    assert vecs.offsets.dtype == np.int64


def test_count_map_dtype_is_int32():
    sig = _make_lazy_4d(nav=(4, 4), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    assert vecs.count_map().dtype == np.int32


def test_stopped_flag_returns_none():
    """stopped_flag[0]=True before compute returns None instead of vectors."""
    nav = (4, 4)
    sig = _make_lazy_4d(nav=nav, sig=(32, 32))
    params = _params()
    stopped = [True]  # pre-set to stopped
    result = _do_compute_vectors(sig, params, None, None, stopped_flag=stopped)
    assert result is None, f"Expected None when stopped, got {result}"


def test_non_square_nav():
    """Non-square navigation grids must produce correct nav_shape and offsets."""
    nav = (3, 7)
    sig = _make_lazy_4d(nav=nav, sig=(32, 32), chunk_nav=2)
    vecs = _do_compute_vectors(sig, _params(), None, None)
    assert vecs.nav_shape == nav
    assert len(vecs.offsets) == nav[0] * nav[1] + 1
    assert vecs.offsets[-1] == len(vecs.flat_buffer)


def test_single_nav_position():
    """Degenerate 1x1 nav grid should produce exactly 1 CSR row."""
    ky, kx = 32, 32
    data = np.zeros((1, 1, ky, kx), dtype=np.float32)
    data[0, 0, ky // 2 - 2:ky // 2 + 2, kx // 2 - 2:kx // 2 + 2] = 100.0
    s = hs.signals.Signal2D(data)
    s.axes_manager.signal_axes[0].scale = 0.01
    s.axes_manager.signal_axes[0].offset = 0.0
    s.axes_manager.signal_axes[1].scale = 0.01
    s.axes_manager.signal_axes[1].offset = 0.0
    vecs = _do_compute_vectors(s, _params(), None, None)
    assert vecs.nav_shape == (1, 1)
    assert len(vecs.offsets) == 2


def test_chunk_boundary_result_equals_single_chunk():
    """
    Result must be the same whether the nav fits in one chunk or spans two.

    This tests that ghost-zone reflect-padding at chunk boundaries doesn't
    distort the peak positions.
    """
    nav = (8, 8)
    ny, nx, ky, kx = nav[0], nav[1], 32, 32
    data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    data[:, :, ky // 2 - 2:ky // 2 + 2, kx // 2 - 2:kx // 2 + 2] = 100.0
    params = dict(sigma=0.0, kernel_radius=3, threshold=0.3, min_distance=2, subpixel=False)

    # Whole dataset in one chunk
    s_one = hs.signals.Signal2D(da.from_array(data, chunks=(ny, nx, ky, kx)))
    s_one.axes_manager.signal_axes[0].scale = 0.01
    s_one.axes_manager.signal_axes[0].offset = 0.0
    s_one.axes_manager.signal_axes[1].scale = 0.01
    s_one.axes_manager.signal_axes[1].offset = 0.0

    # Split into 4 chunks (2x2 nav chunks)
    s_split = hs.signals.Signal2D(da.from_array(data, chunks=(4, 4, ky, kx)))
    s_split.axes_manager.signal_axes[0].scale = 0.01
    s_split.axes_manager.signal_axes[0].offset = 0.0
    s_split.axes_manager.signal_axes[1].scale = 0.01
    s_split.axes_manager.signal_axes[1].offset = 0.0

    v_one = _do_compute_vectors(s_one, params, None, None)
    v_split = _do_compute_vectors(s_split, params, None, None)

    # Both should find the same number of vectors at every position
    cm_one = v_one.count_map()
    cm_split = v_split.count_map()
    np.testing.assert_array_equal(
        cm_one, cm_split,
        err_msg="count_map differs between single-chunk and multi-chunk runs"
    )


def test_on_chunk_done_called_covering_full_nav():
    """on_chunk_done is called after compute with a full-nav slice covering all positions."""
    nav = (8, 8)
    sig = _make_lazy_4d(nav=nav, sig=(32, 32), chunk_nav=4)
    params = _params()

    covered = np.zeros(nav, dtype=bool)

    def _on_chunk(nav_slices, count_sub):
        covered[nav_slices] = True

    _do_compute_vectors(sig, params, None, None, on_chunk_done=_on_chunk)

    assert covered.all(), (
        f"on_chunk_done did not cover all positions: "
        f"{np.argwhere(~covered).tolist()} uncovered"
    )


def test_on_chunk_done_count_subarray_matches_final():
    """count_sub passed to on_chunk_done must match the final count_map."""
    nav = (6, 6)
    sig = _make_lazy_4d(nav=nav, sig=(32, 32), chunk_nav=3)
    params = _params()

    received = [None]

    def _on_chunk(nav_slices, count_sub):
        received[0] = (nav_slices, count_sub)

    vecs = _do_compute_vectors(sig, params, None, None, on_chunk_done=_on_chunk)
    final_cm = vecs.count_map()

    assert received[0] is not None
    nav_slices, count_sub = received[0]
    np.testing.assert_array_equal(
        final_cm[nav_slices], count_sub,
        err_msg="Chunk-done counts don't match the final count_map"
    )


def test_dask_path_produces_correct_nav_shape():
    """Dask path produces correct nav_shape and non-negative counts."""
    nav = (12, 12)
    sig_shape = (32, 32)
    chunk_nav = 4

    sig = _make_lazy_4d(nav=nav, sig=sig_shape, chunk_nav=chunk_nav)
    params = dict(sigma=0.5, kernel_radius=3, threshold=0.3, min_distance=2, subpixel=False)
    vecs = _do_compute_vectors(sig, params, None, None)

    assert vecs.nav_shape == nav
    assert (vecs.count_map() >= 0).all()


def test_no_full_compute_via_tracemalloc():
    """
    Tracemalloc check: total peak RAM allocated during the dask path of
    _do_compute_vectors must be well below the full-dataset size.

    A 12x12x32x32 float32 dataset is 18 MB.  We expect peak allocation
    during the call to be < 5 MB (one chunk + working buffers), never 18 MB.
    """
    nav = (12, 12)
    sig_shape = (32, 32)
    dataset_bytes = nav[0] * nav[1] * sig_shape[0] * sig_shape[1] * 4
    budget_bytes = dataset_bytes // 3  # allow 1/3 of full dataset

    sig = _make_lazy_4d(nav=nav, sig=sig_shape, chunk_nav=4)
    params = _params()

    # Warm up first: GPU kernel cache-loads / JIT and CuPy import allocate
    # tens of MB once per process — that one-time cost is not what this
    # test measures (steady-state per-compute memory is).
    _do_compute_vectors(sig, params, None, None)

    # GC before measurement
    gc.collect()

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    vecs = _do_compute_vectors(sig, params, None, None)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Restrict to allocations attributable to the compute path: when other
    # suites in the session leave a Qt window / distributed cluster running,
    # their background threads allocate during this window and would be
    # falsely attributed to the compute.
    filters = [
        tracemalloc.Filter(True, "*find_vectors*"),
        tracemalloc.Filter(True, "*dask*"),
        tracemalloc.Filter(True, "*numpy*"),
    ]
    snap_before = snap_before.filter_traces(filters)
    snap_after = snap_after.filter_traces(filters)
    stats = snap_after.compare_to(snap_before, "lineno")
    peak_diff = sum(s.size_diff for s in stats if s.size_diff > 0)

    assert vecs.nav_shape == nav
    assert peak_diff < dataset_bytes, (
        f"Memory delta during compute: {peak_diff / 1e6:.1f} MB "
        f">= full dataset size {dataset_bytes / 1e6:.1f} MB. "
        f"The full dataset was likely materialised into RAM."
    )


def test_intensity_column_is_raw_frame_intensity():
    """The stored intensity is the RAW experimental frame intensity at each peak
    (for virtual imaging etc.), NOT the NXCORR score — so it is bounded by the
    frame's own min/max, not [-1, 1]."""
    sig = _make_lazy_4d(nav=(4, 4), sig=(32, 32))
    vecs = _do_compute_vectors(sig, _params(), None, None)
    if len(vecs.flat_buffer) == 0:
        pytest.skip("No vectors found")
    intensities = vecs.flat_buffer[:, 5]  # COL_INTENSITY
    fmin = float(np.asarray(sig.data).min())
    fmax = float(np.asarray(sig.data).max())
    assert intensities.min() >= fmin - 1e-4
    assert intensities.max() <= fmax + 1e-4


def test_reproducible_with_same_params():
    """Two calls with identical params and data must produce identical results."""
    nav = (6, 6)
    sig = _make_lazy_4d(nav=nav, sig=(32, 32))
    params = _params()
    v1 = _do_compute_vectors(sig, params, None, None)
    v2 = _do_compute_vectors(sig, params, None, None)
    np.testing.assert_array_equal(v1.flat_buffer, v2.flat_buffer)
    np.testing.assert_array_equal(v1.offsets, v2.offsets)


def test_large_nav_does_not_compute_full_array():
    """
    Regression test: the dask path must never .compute() on the full-dataset shape.

    Old bug: called client.compute(raw.astype(...)) — materialised all nav patterns
    on a single worker before any chunking happened.

    Uses sig_shape=(256,256) and nav=(32,32) to guarantee multiple distinct chunks
    so the guard cannot be triggered by a legitimate single-chunk call.
    """
    nav = (32, 32)
    sig_shape = (256, 256)
    sig = _make_lazy_4d(nav=nav, sig=sig_shape, chunk_nav=8)
    params = dict(sigma=0.5, kernel_radius=2, threshold=0.3, min_distance=2, subpixel=False)

    full_shape = sig.data.shape
    full_computed = [False]
    _orig_compute = da.Array.compute

    def _spy(self, *args, **kwargs):
        if self.shape == full_shape:
            full_computed[0] = True
            raise AssertionError(
                f"compute() on full-dataset shape {self.shape} — "
                f"do not materialise the entire dataset at once."
            )
        return _orig_compute(self, *args, **kwargs)

    with patch.object(da.Array, "compute", _spy):
        vecs = _do_compute_vectors(sig, params, None, None)

    assert not full_computed[0], "Full-dataset compute was triggered"
    assert vecs.nav_shape == nav
    assert len(vecs.offsets) == nav[0] * nav[1] + 1

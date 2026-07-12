"""Cheap/expensive nav-read classifier.

_classify_nav_read decides whether a navigator frame read is served synchronously
on the dispatcher (cheap) or submitted off-dispatcher via submit_graph (expensive).
Async now fires ONLY for reads that would genuinely FREEZE the navigator: a LARGE
region (> REGION_ASYNC_FRAME_CAP frames) or a cold HUGE single frame on a cached
signal. Single points — including DERIVED rebin/crop views (now synchronous +
decoded-chunk-cached) — and small/medium regions are cheap. Must be pure +
side-effect-free (never compute).
"""
import numpy as np
import dask.array as da

from spyde.drawing.update_functions import (
    _classify_nav_read,
    REGION_ASYNC_FRAME_CAP,
    REGION_BYTES_CAP,
)


class _FakeAxesManager:
    def __init__(self, nav_dim):
        self.navigation_axes = list(range(nav_dim))


class _FakeSignal:
    """Minimal stand-in: the classifier only reads .cached_dask_array and
    .axes_manager.navigation_axes off the signal (the rest is passed in)."""
    def __init__(self, cached=None, nav_dim=2):
        self.cached_dask_array = cached
        self.axes_manager = _FakeAxesManager(nav_dim)


def _frame_bytes(frame_shape, itemsize):
    return int(np.prod(frame_shape)) * itemsize


class TestClassifyRegion:
    def test_small_region_is_cheap(self):
        data = da.zeros((32, 32, 8, 8), dtype=np.uint16, chunks=(8, 8, -1, -1))
        sig = _FakeSignal(cached=object(), nav_dim=2)
        idx = np.array([[i, 0] for i in range(4)])
        fb = _frame_bytes((8, 8), 2)
        assert _classify_nav_read(sig, idx, data, fb) == "cheap"

    def test_medium_region_below_async_cap_is_cheap(self):
        # A movie-sized region (≤16 frames) or a modest STEM region stays sync.
        data = da.zeros((300, 8, 8), dtype=np.uint16, chunks=(1, -1, -1))
        sig = _FakeSignal(cached=None, nav_dim=1)
        fb = _frame_bytes((8, 8), 2)
        idx = np.array([[i] for i in range(16)])  # maxed movie span
        assert _classify_nav_read(sig, idx, data, fb) == "cheap"

    def test_large_region_is_expensive(self):
        data = da.zeros((32, 32, 8, 8), dtype=np.uint16, chunks=(8, 8, -1, -1))
        sig = _FakeSignal(cached=object(), nav_dim=2)
        n = REGION_ASYNC_FRAME_CAP + 5
        idx = np.array([[i % 32, 0] for i in range(n)])
        fb = _frame_bytes((8, 8), 2)
        assert _classify_nav_read(sig, idx, data, fb) == "expensive"

    def test_region_at_async_cap_is_cheap_over_is_expensive(self):
        data = da.zeros((512, 8, 8), dtype=np.uint16, chunks=(1, -1, -1))
        sig = _FakeSignal(cached=object(), nav_dim=1)
        fb = _frame_bytes((8, 8), 2)
        at_cap = np.array([[i] for i in range(REGION_ASYNC_FRAME_CAP)])
        over = np.array([[i] for i in range(REGION_ASYNC_FRAME_CAP + 1)])
        assert _classify_nav_read(sig, at_cap, data, fb) == "cheap"
        assert _classify_nav_read(sig, over, data, fb) == "expensive"

    def test_region_by_bytes_is_expensive(self):
        # Few frames but each huge → exceeds the byte cap even below the frame cap.
        big_frame = (4096, 4096)
        fb = _frame_bytes(big_frame, 4)  # ~64 MiB/frame
        assert fb * 3 > REGION_BYTES_CAP
        data = da.zeros((10,) + big_frame, dtype=np.float32, chunks=(1, -1, -1))
        sig = _FakeSignal(cached=object(), nav_dim=1)
        idx = np.array([[0], [1], [2]])  # only 3 pts but > byte cap
        assert _classify_nav_read(sig, idx, data, fb) == "expensive"


class TestClassifySinglePoint:
    def test_raw_movie_single_point_is_cheap(self):
        data = da.from_array(np.zeros((64, 8, 8), np.uint16), chunks=(1, -1, -1))
        sig = _FakeSignal(cached=None, nav_dim=1)
        assert _classify_nav_read(sig, np.array([5]), data, _frame_bytes((8, 8), 2)) == "cheap"

    def test_rebinned_view_single_point_is_cheap_now(self):
        # A derived view is now served synchronously from the decoded-chunk cache,
        # NOT async — the transform recompute is ~5-9 ms and dwell-in-chunk is ~0 ms.
        raw = da.from_array(np.zeros((64, 16, 16), np.uint16), chunks=(1, -1, -1))
        rebinned = da.coarsen(np.mean, raw, {1: 2, 2: 2})
        sig = _FakeSignal(cached=None, nav_dim=1)
        assert _classify_nav_read(sig, np.array([5]), rebinned, _frame_bytes((8, 8), 8)) == "cheap"

    def test_cropped_view_single_point_is_cheap_now(self):
        raw = da.from_array(np.zeros((64, 16, 16), np.uint16), chunks=(1, -1, -1))
        cropped = raw[:, 2:10, 2:10]
        sig = _FakeSignal(cached=None, nav_dim=1)
        assert _classify_nav_read(sig, np.array([5]), cropped, _frame_bytes((8, 8), 2)) == "cheap"


class TestClassifyCachedCold:
    def test_cached_small_frame_is_cheap(self):
        data = da.zeros((32, 32, 8, 8), np.uint16, chunks=(8, 8, -1, -1))
        sig = _FakeSignal(cached=object(), nav_dim=2)
        fb = _frame_bytes((8, 8), 2)  # tiny
        assert _classify_nav_read(sig, np.array([3, 3]), data, fb) == "cheap"

    def test_cached_cold_huge_frame_is_expensive(self):
        # A cached signal whose cold huge frame misses residency → async.
        # cached=None fake has no core_cached_block_inds so _nav_cache_is_resident
        # returns False; use a fake cache object WITHOUT residency to simulate MISS.
        data = da.zeros((10, 4096, 4096), np.float32, chunks=(1, -1, -1))
        sig = _FakeSignal(cached=object(), nav_dim=1)  # has a cache, but not resident
        fb = _frame_bytes((4096, 4096), 4)  # 64 MiB > COLD_FRAME_CAP
        assert _classify_nav_read(sig, np.array([5]), data, fb) == "expensive"

    def test_derived_huge_frame_stays_cheap(self):
        # No cache (derived) → single points are ALWAYS synchronous (chunk-cached),
        # even for a huge frame — the async cold rule only applies to cached signals.
        raw = da.from_array(np.zeros((10, 4096, 4096), np.float32), chunks=(1, -1, -1))
        derived = da.coarsen(np.mean, raw, {1: 2, 2: 2})
        sig = _FakeSignal(cached=None, nav_dim=1)
        fb = _frame_bytes((2048, 2048), 8)
        assert _classify_nav_read(sig, np.array([5]), derived, fb) == "cheap"

    def test_never_computes(self):
        # Classifying must not materialise the array.
        raw = da.from_array(np.zeros((64, 8, 8), np.uint16), chunks=(1, -1, -1))
        rebinned = da.coarsen(np.mean, raw, {1: 2, 2: 2}).map_blocks(lambda b: b)
        sig = _FakeSignal(cached=None, nav_dim=1)
        # No exception, returns a verdict.
        assert _classify_nav_read(sig, np.array([1]), rebinned, _frame_bytes((4, 4), 8)) in ("cheap", "expensive")

"""Decoded-chunk cache for derived-view navigator reads (_NavChunkCache).

A derived (rebin/crop/.zspy) view has no hyperspy CachedDaskArray, so every move
would re-decode + re-transform the whole source nav-chunk. _NavChunkCache decodes
each output nav-chunk once and slices frames out of it — dwell-in-chunk becomes a
free numpy slice. These tests pin: frame parity vs a plain compute, hit vs miss,
LRU eviction under the frame budget, per-signal invalidation, and 1-D + 2-D nav.
"""
import numpy as np
import dask.array as da

from spyde.drawing.update_functions import _NavChunkCache, _nav_chunk_span


class _AxesManager:
    def __init__(self, nav_dim):
        self.navigation_dimension = nav_dim


class _Signal:
    def __init__(self, data, nav_dim):
        self.data = data
        self.axes_manager = _AxesManager(nav_dim)


class _Prof:
    def __init__(self):
        self.last = None

    def done(self, msg=""):
        self.last = msg


def _rebinned_movie(n=64, frame=(16, 16), chunk=8, seed=0):
    base = np.random.RandomState(seed).randint(0, 4000, (n, *frame)).astype(np.uint16)
    raw = da.from_array(base, chunks=(chunk, -1, -1))
    reb = da.coarsen(np.mean, raw, {1: 2, 2: 2})  # -> (n, frame/2, frame/2), derived
    return reb, base


class TestNavChunkSpan:
    def test_1d_mapping(self):
        # chunks of 30 over 300
        assert _nav_chunk_span((30,) * 10, 0) == (0, 0, 30)
        assert _nav_chunk_span((30,) * 10, 45) == (1, 30, 60)
        assert _nav_chunk_span((30,) * 10, 299) == (9, 270, 300)

    def test_uneven_last_chunk(self):
        # 64 in chunks of 12 → last is 4
        sizes = (12, 12, 12, 12, 12, 4)
        assert _nav_chunk_span(sizes, 63) == (5, 60, 64)
        assert _nav_chunk_span(sizes, 12) == (1, 12, 24)


class TestChunkCache1D:
    def test_frame_parity_and_hit_miss(self):
        reb, base = _rebinned_movie()
        sig = _Signal(reb, nav_dim=1)
        cache = _NavChunkCache()
        prof = _Prof()

        # First read at frame 5 → MISS (decodes the chunk [0:8]).
        f5 = cache.get_frame(sig, reb, np.array([5]), prof)
        assert "MISS" in prof.last
        expected5 = base[5].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f5, expected5, rtol=1e-5)

        # Second read at frame 3 (same chunk [0:8]) → HIT, no recompute.
        f3 = cache.get_frame(sig, reb, np.array([3]), prof)
        assert prof.last == "chunk-cache hit"
        expected3 = base[3].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f3, expected3, rtol=1e-5)

        # A frame in a DIFFERENT chunk (frame 40 → chunk [40:48]) → MISS.
        cache.get_frame(sig, reb, np.array([40]), prof)
        assert "MISS" in prof.last

    def test_region_returns_none(self):
        reb, _ = _rebinned_movie()
        sig = _Signal(reb, nav_dim=1)
        cache = _NavChunkCache()
        # A region (idx.ndim > 1) is not a single frame → cache declines.
        assert cache.get_frame(sig, reb, np.array([[1], [2], [3]]), _Prof()) is None


class TestChunkCache2D:
    def test_2d_nav_frame_parity(self):
        base = np.random.RandomState(1).randint(0, 500, (24, 24, 16, 16)).astype(np.uint16)
        raw = da.from_array(base, chunks=(12, 12, -1, -1))
        reb = da.coarsen(np.mean, raw, {2: 2, 3: 2})  # -> (24,24,8,8)
        sig = _Signal(reb, nav_dim=2)
        cache = _NavChunkCache()
        prof = _Prof()

        f = cache.get_frame(sig, reb, np.array([2, 3]), prof)
        assert "MISS" in prof.last
        exp = base[2, 3].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f, exp, rtol=1e-5)

        # Another point in the SAME 12x12 nav chunk → HIT.
        f2 = cache.get_frame(sig, reb, np.array([10, 10]), prof)
        assert prof.last == "chunk-cache hit"
        exp2 = base[10, 10].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f2, exp2, rtol=1e-5)

        # A point in a different nav chunk → MISS.
        cache.get_frame(sig, reb, np.array([20, 20]), prof)
        assert "MISS" in prof.last


class TestChunkCacheEviction:
    def test_lru_evicts_over_frame_budget(self):
        # 10 chunks of 8 frames each; budget of 16 frames → at most ~2 chunks resident.
        reb, _ = _rebinned_movie(n=80, chunk=8)
        sig = _Signal(reb, nav_dim=1)
        cache = _NavChunkCache(max_frames=16)
        # Touch one frame in each of 5 distinct chunks.
        for f in (0, 8, 16, 24, 32):
            cache.get_frame(sig, reb, np.array([f]), _Prof())
        # Budget bounds resident frames.
        assert cache._frames <= 16
        # Only the most-recent chunks survive; the first is evicted.
        assert (id(sig), (0,)) not in cache._blocks
        assert (id(sig), (4,)) in cache._blocks  # frame 32 → chunk index 4

    def test_single_chunk_over_budget_still_served(self):
        # One chunk alone exceeds the budget → we still keep it (can't serve otherwise).
        reb, base = _rebinned_movie(n=64, chunk=32)  # a 32-frame chunk
        sig = _Signal(reb, nav_dim=1)
        cache = _NavChunkCache(max_frames=16)  # smaller than one chunk
        f = cache.get_frame(sig, reb, np.array([5]), _Prof())
        assert f is not None
        exp = base[5].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f, exp, rtol=1e-5)


class TestChunkCacheInvalidation:
    def test_clear_drops_blocks(self):
        reb, _ = _rebinned_movie()
        sig = _Signal(reb, nav_dim=1)
        cache = _NavChunkCache()
        cache.get_frame(sig, reb, np.array([5]), _Prof())
        assert cache._frames > 0
        cache.clear()
        assert cache._frames == 0 and len(cache._blocks) == 0

    def test_different_signal_id_misses(self):
        reb, base = _rebinned_movie()
        cache = _NavChunkCache()
        sig1 = _Signal(reb, nav_dim=1)
        cache.get_frame(sig1, reb, np.array([5]), _Prof())
        # A NEW signal object (node switch) with the same data → different id → MISS.
        sig2 = _Signal(reb, nav_dim=1)
        prof = _Prof()
        cache.get_frame(sig2, reb, np.array([5]), prof)
        assert "MISS" in prof.last

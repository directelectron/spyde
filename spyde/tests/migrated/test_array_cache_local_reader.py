"""ArrayCache reader kind 4: signal-tree local-transform (spyde/array_cache/
readers/local_transform.py), generalizing the old _NavChunkCache
(spyde/drawing/update_functions.py, removed).

Pins: nav_chunk_span's chunk-index arithmetic (migrated from
test_nav_chunk_cache.py verbatim), LocalTransformReader frame parity vs a
plain compute + its internal one-chunk memo (no redundant decode when
dwelling in the same source chunk), and — through the real action/tree/plot
flow — that _direct_read_frame only takes the ArrayCache fast path for a
locality-resolved-local signal, falling through to a plain compute otherwise
(the opaque-fallback path, exercised here with a synthetic opaque action
since no real one exists yet).
"""
from __future__ import annotations

import time

import numpy as np
import dask.array as da

from spyde.array_cache.readers.local_transform import (
    LocalTransformReader,
    nav_chunk_span,
)
from spyde.array_cache import ArrayCache, BlockCache, get_local_frame
from spyde.tests.migrated._async import quiesce, why_busy


class _AxesManager:
    def __init__(self, nav_dim):
        self.navigation_dimension = nav_dim


class _Signal:
    def __init__(self, data, nav_dim):
        self.data = data
        self.axes_manager = _AxesManager(nav_dim)


def _rebinned_movie(n=64, frame=(16, 16), chunk=8, seed=0):
    base = np.random.RandomState(seed).randint(0, 4000, (n, *frame)).astype(np.uint16)
    raw = da.from_array(base, chunks=(chunk, -1, -1))
    reb = da.coarsen(np.mean, raw, {1: 2, 2: 2})  # -> (n, frame/2, frame/2), derived
    return reb, base


class TestNavChunkSpan:
    def test_1d_mapping(self):
        assert nav_chunk_span((30,) * 10, 0) == (0, 0, 30)
        assert nav_chunk_span((30,) * 10, 45) == (1, 30, 60)
        assert nav_chunk_span((30,) * 10, 299) == (9, 270, 300)

    def test_uneven_last_chunk(self):
        sizes = (12, 12, 12, 12, 12, 4)
        assert nav_chunk_span(sizes, 63) == (5, 60, 64)
        assert nav_chunk_span(sizes, 12) == (1, 12, 24)


class TestLocalTransformReader1D:
    def test_frame_parity_and_chunk_memo_reuse(self):
        reb, base = _rebinned_movie()
        sig = _Signal(reb, nav_dim=1)
        reader = LocalTransformReader(sig, reb)

        f5 = reader.read_frame((5,))
        expected5 = base[5].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f5, expected5, rtol=1e-5)
        block_after_first = reader._memo[1]

        # Same chunk [0:8] -> internal memo reused, no re-decode (same block object).
        f3 = reader.read_frame((3,))
        expected3 = base[3].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f3, expected3, rtol=1e-5)
        assert reader._memo[1] is block_after_first

        # A different chunk (frame 40 -> chunk [40:48]) -> memo replaced.
        reader.read_frame((40,))
        assert reader._memo[1] is not block_after_first

    def test_returned_frame_does_not_retain_the_decoded_chunk(self):
        """read_frame must COPY the frame out of the decoded chunk: caching a
        VIEW would keep the whole chunk alive while ArrayCache accounts only one
        frame (honest byte budget / Memory-Safety rule)."""
        reb, _ = _rebinned_movie(n=64, frame=(16, 16), chunk=8)
        sig = _Signal(reb, nav_dim=1)
        reader = LocalTransformReader(sig, reb)

        frame = reader.read_frame((5,))
        block = reader._memo[1]
        assert frame.base is None, "frame is a view — it retains the whole chunk"
        assert frame.nbytes < block.nbytes  # sanity: the chunk really is bigger

    def test_is_chunk_resident_tracks_the_memo(self):
        reb, _ = _rebinned_movie(n=64, frame=(16, 16), chunk=8)
        sig = _Signal(reb, nav_dim=1)
        reader = LocalTransformReader(sig, reb)

        assert reader.is_chunk_resident((5,)) is False   # nothing decoded yet
        reader.read_frame((5,))
        assert reader.is_chunk_resident((5,)) is True
        assert reader.is_chunk_resident((3,)) is True    # same chunk [0:8]
        assert reader.is_chunk_resident((40,)) is False  # a different chunk

    def test_frame_bytes(self):
        reb, _ = _rebinned_movie(frame=(16, 16))  # rebinned to (8, 8)
        sig = _Signal(reb, nav_dim=1)
        reader = LocalTransformReader(sig, reb)
        assert reader.frame_bytes == 8 * 8 * reb.dtype.itemsize


class TestLocalTransformReader2D:
    def test_2d_nav_frame_parity_and_memo(self):
        base = np.random.RandomState(1).randint(0, 500, (24, 24, 16, 16)).astype(np.uint16)
        raw = da.from_array(base, chunks=(12, 12, -1, -1))
        reb = da.coarsen(np.mean, raw, {2: 2, 3: 2})  # -> (24,24,8,8)
        sig = _Signal(reb, nav_dim=2)
        reader = LocalTransformReader(sig, reb)

        f = reader.read_frame((2, 3))
        exp = base[2, 3].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f, exp, rtol=1e-5)
        block_after_first = reader._memo[1]

        # Same 12x12 nav chunk -> memo reused.
        f2 = reader.read_frame((10, 10))
        exp2 = base[10, 10].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f2, exp2, rtol=1e-5)
        assert reader._memo[1] is block_after_first

        # Different nav chunk -> memo replaced.
        reader.read_frame((20, 20))
        assert reader._memo[1] is not block_after_first


class TestGetLocalFrameThroughArrayCache:
    """The nav_read.py bridge (used by both _direct_read_frame and
    overlay.py) — a lightweight fake plot standing in for the real Plot,
    since only signal_tree/_array_cache/_local_transform_readers matter."""

    class _FakeTree:
        def resolve_locality(self, signal):
            return getattr(signal, "_is_local", True)

    class _FakePlot:
        def __init__(self):
            self.signal_tree = TestGetLocalFrameThroughArrayCache._FakeTree()
            self._array_cache = ArrayCache()
            # Readers share the plot's BlockCache for decoded nav-chunks; without
            # it they fall back to a private one-entry memo and the block-level
            # behaviour under test isn't exercised.
            self._block_cache = BlockCache()
            self._local_transform_readers = {}

    def test_local_signal_populates_array_cache(self):
        reb, base = _rebinned_movie()
        sig = _Signal(reb, nav_dim=1)
        sig._is_local = True
        plot = self._FakePlot()

        frame = get_local_frame(plot, sig, reb, (5,))
        expected = base[5].reshape(8, 2, 8, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(frame, expected, rtol=1e-5)
        assert plot._array_cache.is_resident(id(sig), (5,)) is True

    def test_opaque_signal_returns_none(self):
        reb, _ = _rebinned_movie()
        sig = _Signal(reb, nav_dim=1)
        sig._is_local = False  # e.g. an opaque tree node
        plot = self._FakePlot()

        assert get_local_frame(plot, sig, reb, (5,)) is None
        assert len(plot._array_cache) == 0

    def test_region_returns_frame_wise_mean(self):
        """An integrating region is served THROUGH the cache, not bailed on.

        It used to return None, so region integration bypassed the whole array
        cache and paid a dask compute per point — each materialising the enclosing
        nav-chunk to keep one frame."""
        reb, src = _rebinned_movie()
        sig = _Signal(reb, nav_dim=1)
        sig._is_local = True
        plot = self._FakePlot()

        out = get_local_frame(plot, sig, reb, np.array([[1], [2]]))
        assert out is not None
        expected = np.asarray(reb[[1, 2]].compute()).mean(axis=0)
        if np.issubdtype(reb.dtype, np.integer):
            expected = np.rint(expected).astype(reb.dtype)
        np.testing.assert_array_equal(out, expected)

    def test_region_sums_from_the_block_not_frame_by_frame(self):
        """A region sums straight out of the decoded BLOCK.

        The per-frame path has to COPY each frame out of its block so a cached
        frame doesn't pin the whole thing; for a region that copy dominates
        (measured 165 ms vs 61 ms on a resident 537 MB block, 16x16 ROI) and buys
        nothing, since region frames are summed immediately and never wanted
        individually. So ``read_frame`` must NOT be called, and the region's
        frames must NOT be inserted into the frame cache — the BLOCK is what
        stays resident, which is what makes the next drag step cheap."""
        reb, _ = _rebinned_movie(n=64, frame=(8, 8), chunk=8)
        sig = _Signal(reb, nav_dim=1)
        sig._is_local = True
        plot = self._FakePlot()

        get_local_frame(plot, sig, reb, np.array([[10], [11], [12], [13]]))
        reader = plot._local_transform_readers[id(sig)]
        assert hasattr(reader, "sum_points")
        calls = []
        orig = reader.read_frame
        reader.read_frame = lambda idx, _o=orig: (calls.append(tuple(idx)), _o(idx))[1]

        out = get_local_frame(plot, sig, reb, np.array([[11], [12], [13], [14]]))
        assert out is not None
        assert calls == [], calls          # summed from the block, no frame reads
        # the block is resident, so the next overlapping step is cheap
        assert plot._block_cache.nbytes > 0

    def test_region_matches_per_frame_sum_across_a_block_boundary(self):
        """The block path groups points per block; an ROI straddling two blocks
        must still equal the plain per-frame mean."""
        reb, _ = _rebinned_movie(n=64, frame=(8, 8), chunk=8)
        sig = _Signal(reb, nav_dim=1)
        sig._is_local = True
        plot = self._FakePlot()

        pts = np.array([[6], [7], [8], [9]])       # chunk=8 -> spans two blocks
        out = get_local_frame(plot, sig, reb, pts)
        expected = np.asarray(reb[[6, 7, 8, 9]].compute()).mean(axis=0)
        if np.issubdtype(reb.dtype, np.integer):
            expected = np.rint(expected).astype(reb.dtype)
        np.testing.assert_array_equal(out, expected)

    def test_residency_probe_is_chunk_granular(self):
        """overlay.py's cheap/expensive gate: a NEW position inside an already
        DECODED chunk is cheap (the decode is the cost, the slice is free). If
        the probe only asked ArrayCache — which knows only frames it already
        returned — every move would be misclassified as a cold read and the
        overlay layer would skip + re-warm on each one."""
        from spyde.array_cache import is_local_frame_resident

        reb, _ = _rebinned_movie(n=64, frame=(16, 16), chunk=8)
        sig = _Signal(reb, nav_dim=1)
        sig._is_local = True
        plot = self._FakePlot()

        assert is_local_frame_resident(plot, sig, reb, np.array([5])) is False
        get_local_frame(plot, sig, reb, (5,))
        assert is_local_frame_resident(plot, sig, reb, np.array([5])) is True
        # frame 3 was never read, but its chunk [0:8] is decoded -> cheap
        assert plot._array_cache.is_resident(id(sig), (3,)) is False
        assert is_local_frame_resident(plot, sig, reb, np.array([3])) is True
        # frame 40 lives in an undecoded chunk -> genuinely cold
        assert is_local_frame_resident(plot, sig, reb, np.array([40])) is False

    def test_data_swap_on_the_same_signal_drops_stale_frames(self):
        """SpyDE swaps ``signal.data`` in place (progressive nav fill, a result
        window's computed data landing) without a new signal object, and the
        cache keys on the signal's identity — so the old array's frames must be
        evicted when the reader is rebuilt, not served forever."""
        first, base_a = _rebinned_movie(n=64, frame=(16, 16), chunk=8, seed=0)
        second, base_b = _rebinned_movie(n=64, frame=(16, 16), chunk=8, seed=99)
        sig = _Signal(first, nav_dim=1)
        sig._is_local = True
        plot = self._FakePlot()

        got_a = get_local_frame(plot, sig, first, (5,))
        np.testing.assert_allclose(
            got_a, base_a[5].reshape(8, 2, 8, 2).mean(axis=(1, 3)), rtol=1e-5)

        sig.data = second                      # in-place swap, same signal object
        got_b = get_local_frame(plot, sig, second, (5,))
        np.testing.assert_allclose(
            got_b, base_b[5].reshape(8, 2, 8, 2).mean(axis=(1, 3)), rtol=1e-5)
        assert not np.allclose(got_a, got_b)   # sanity: the arrays really differ

    def test_residency_probe_never_creates_a_reader(self):
        from spyde.array_cache import is_local_frame_resident

        reb, _ = _rebinned_movie()
        sig = _Signal(reb, nav_dim=1)
        sig._is_local = True
        plot = self._FakePlot()

        assert is_local_frame_resident(plot, sig, reb, np.array([5])) is False
        assert plot._local_transform_readers == {}, \
            "a side-effect-free probe must not allocate a reader"


class TestDirectReadFrameLocalityGating:
    """End-to-end through the real Session/Plot/action flow: _direct_read_frame
    only engages the ArrayCache fast path for a locality-resolved-local node."""

    def test_rebin_of_lazy_movie_uses_array_cache(self, movie_dataset):
        from spyde.actions.base import Rebin2DAction
        from spyde.drawing.update_functions import _direct_read_frame, NavProfile

        session = movie_dataset["window"]
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        act = Rebin2DAction.for_plot(plot, scale_x=2, scale_y=2)
        new = act.run()
        time.sleep(0.2)
        assert new is not None and new._lazy

        prof = NavProfile("TEST", np.array([0]))
        frame = _direct_read_frame(new, None, np.array([0]), prof, child=plot)
        assert frame is not None
        assert plot._array_cache.is_resident(id(new), (0,)) is True

    def test_untagged_transform_falls_through_not_cached(self, movie_dataset):
        """A synthetic same-tree node with no locality tag (the fail-safe
        default) must still read correctly via the plain-compute fallback,
        but must NOT populate the ArrayCache — this is the opaque-fallback
        path Phase 2/3 built, exercised here since no real opaque
        TransformAction exists in the codebase yet."""
        from spyde.drawing.update_functions import _direct_read_frame, NavProfile

        session = movie_dataset["window"]
        tree = session.signal_trees[0]
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        root = tree.root

        # add_node with no local= kwarg -> defaults to None -> resolves opaque.
        rebinned = root.rebin(scale=[1, 2, 2])
        tree.add_node(root, rebinned, "UntaggedRebin")  # local defaults to None
        assert quiesce(session), why_busy(session)

        prof = NavProfile("TEST", np.array([0]))
        frame = _direct_read_frame(rebinned, None, np.array([0]), prof, child=plot)
        assert frame is not None  # still served, just not via ArrayCache
        assert plot._array_cache.is_resident(id(rebinned), (0,)) is False

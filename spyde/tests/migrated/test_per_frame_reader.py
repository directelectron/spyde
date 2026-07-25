"""PerFrameReader — a derived view's frames built in numpy from the PARENT's.

Asking dask for one rebinned frame materialises the whole enclosing source
nav-chunk and re-runs the graph. Measured on a real .zspy 4D-STEM
(64x64x512^2, 32x32 nav chunks = 537 MB):

    rebinned[3,3].compute()                    2403 ms
    parent frame read + numpy rebin             417 ms
      - of which the parent chunk decode        435 ms   (unavoidable, cached)
      - of which the numpy rebin                1.8 ms
    rebinned frame, parent block already warm     1.8 ms

So the dask path is ~2 s of overhead per scrub position on top of a read that
has to happen anyway, and the transform itself is noise.

Correctness matters more than speed here: a silently WRONG frame is far worse
than a slow one, so the reader declines anything it can't reproduce exactly and
lets LocalTransformReader serve it.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import pytest

from spyde.array_cache import ArrayCache, BlockCache, get_local_frame
from spyde.array_cache.nav_read import _get_local_region
from spyde.array_cache.readers.per_frame import (
    PerFrameReader, build_per_frame_reader,
)


# ── scaffolding: a two-node tree over a lazy base + its rebin ────────────────

class _Node:
    def __init__(self, signal, parent=None, transformation=None, kwargs=None):
        self.signal = signal
        self.parent = parent
        self.transformation = transformation
        self.kwargs = kwargs or {}
        self.local = True
        self._resolved_local = True


class _Tree:
    def __init__(self, base, derived, transformation, kwargs=None):
        self._base_node = _Node(base)
        self._node = _Node(derived, self._base_node, transformation, kwargs)
        self._base, self._derived = base, derived

    def get_node(self, signal):
        if signal is self._derived:
            return self._node
        if signal is self._base:
            return self._base_node
        return None

    def resolve_locality(self, signal):
        return True


class _Plot:
    def __init__(self, tree):
        self._array_cache = ArrayCache()
        self._block_cache = BlockCache()
        self._local_transform_readers = {}
        self.signal_tree = tree


def _lazy_4d(nav=8, sig=8, chunk=4, seed=0):
    import hyperspy.api as hs
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 500, (nav, nav, sig, sig)).astype(np.uint16)
    d = da.from_array(arr, chunks=(chunk, chunk, sig, sig))
    return hs.signals.Signal2D(d).as_lazy(), arr


def _rebinned(nav=8, sig=8, chunk=4, factor=2):
    base, arr = _lazy_4d(nav, sig, chunk)
    reb = base.rebin(scale=[1, 1, factor, factor])
    return base, reb, arr


class TestRebinIsServedPerFrame:
    def test_reader_resolves_to_per_frame(self):
        base, reb, _ = _rebinned()
        plot = _Plot(_Tree(base, reb, "rebin"))
        get_local_frame(plot, reb, reb.data, np.array([1, 1]))
        reader = plot._local_transform_readers[id(reb)]
        assert isinstance(reader, PerFrameReader)

    def test_frame_matches_the_dask_view(self):
        """THE correctness bar: identical to what dask would have produced."""
        base, reb, _ = _rebinned()
        plot = _Plot(_Tree(base, reb, "rebin"))
        for point in [(0, 0), (1, 3), (5, 6), (7, 7)]:
            got = get_local_frame(plot, reb, reb.data, np.array(point))
            expected = np.asarray(reb.data[point].compute(scheduler="synchronous"))
            np.testing.assert_array_equal(np.asarray(got), expected)

    def test_dtype_matches_the_derived_signal(self):
        base, reb, _ = _rebinned()
        plot = _Plot(_Tree(base, reb, "rebin"))
        got = get_local_frame(plot, reb, reb.data, np.array([2, 2]))
        assert np.asarray(got).dtype == reb.data.dtype

    def test_region_matches_a_plain_per_frame_mean(self):
        """sum_points sums the PARENT's frames and transforms ONCE, which is
        only valid because rebin is LINEAR. This pins that."""
        base, reb, _ = _rebinned()
        plot = _Plot(_Tree(base, reb, "rebin"))
        yy, xx = np.meshgrid(np.arange(0, 4), np.arange(0, 4), indexing="ij")
        idx = np.stack([yy.ravel(), xx.ravel()], axis=1)

        got = _get_local_region(plot, reb, reb.data, idx)
        acc = None
        for p in idx:
            f = np.asarray(reb.data[p[0], p[1]].compute(scheduler="synchronous"))
            acc = f.astype(np.float64) if acc is None else acc + f
        expected = acc / len(idx)
        if np.issubdtype(reb.data.dtype, np.integer):
            expected = np.rint(expected).astype(reb.data.dtype)
        np.testing.assert_array_equal(got, expected)

    def test_region_spanning_parent_chunks(self):
        """The parent's block grouping must not change the answer."""
        base, reb, _ = _rebinned(nav=8, chunk=4)     # ROI straddles 4 blocks
        plot = _Plot(_Tree(base, reb, "rebin"))
        yy, xx = np.meshgrid(np.arange(2, 6), np.arange(2, 6), indexing="ij")
        idx = np.stack([yy.ravel(), xx.ravel()], axis=1)

        got = _get_local_region(plot, reb, reb.data, idx)
        acc = None
        for p in idx:
            f = np.asarray(reb.data[p[0], p[1]].compute(scheduler="synchronous"))
            acc = f.astype(np.float64) if acc is None else acc + f
        expected = np.rint(acc / len(idx)).astype(reb.data.dtype)
        np.testing.assert_array_equal(got, expected)


class TestDeclinesWhatItCannotReproduce:
    """Every decline falls back to LocalTransformReader — correct, just slower.
    These pin that the gate is conservative rather than optimistic."""

    def test_unknown_transformation_declines(self):
        base, reb, _ = _rebinned()
        node = _Node(reb, _Node(base), "some_new_action")
        r = build_per_frame_reader(reb, reb.data, node, object(), base)
        assert r is None

    def test_non_integer_rebin_factor_declines(self):
        base, _ = _lazy_4d(nav=4, sig=6)
        # 6 -> 4 is not an integer down-factor
        fake = type("S", (), {})()
        fake.data = da.zeros((4, 4, 4, 4), dtype=np.uint16, chunks=(2, 2, 4, 4))
        fake.axes_manager = base.axes_manager
        node = _Node(fake, _Node(base), "rebin")
        assert build_per_frame_reader(fake, fake.data, node, object(), base) is None

    def test_identity_rebin_declines(self):
        """Nothing to gain, and it would just add a layer."""
        base, _ = _lazy_4d()
        node = _Node(base, _Node(base), "rebin")
        assert build_per_frame_reader(base, base.data, node, object(), base) is None

    def test_changed_nav_grid_declines(self):
        """A nav crop remaps WHICH parent frame each index means — an index
        remap, not a per-frame function."""
        base, _ = _lazy_4d(nav=8)
        cropped = type("S", (), {})()
        cropped.data = da.zeros((4, 4, 8, 8), dtype=np.uint16, chunks=(2, 2, 8, 8))
        cropped.axes_manager = base.axes_manager
        node = _Node(cropped, _Node(base), "rebin")
        assert build_per_frame_reader(
            cropped, cropped.data, node, object(), base) is None

    def test_no_parent_declines(self):
        base, reb, _ = _rebinned()
        assert build_per_frame_reader(reb, reb.data, None, object(), base) is None


class TestParentBlockDoesTheWork:
    def test_residency_follows_the_parent(self):
        """The transform is ~ms, so 'is this cheap?' is really 'is the PARENT's
        block resident?' — the overlay's cheap/expensive gate depends on it."""
        base, reb, _ = _rebinned()
        plot = _Plot(_Tree(base, reb, "rebin"))
        get_local_frame(plot, reb, reb.data, np.array([1, 1]))
        reader = plot._local_transform_readers[id(reb)]
        assert reader.is_chunk_resident((1, 1)) is True

    def test_parent_reader_is_shared_not_rebuilt(self):
        """The parent must be resolved through the normal path so it gets the
        plot's BlockCache — that cache is the whole point."""
        base, reb, _ = _rebinned()
        plot = _Plot(_Tree(base, reb, "rebin"))
        get_local_frame(plot, reb, reb.data, np.array([1, 1]))
        reader = plot._local_transform_readers[id(reb)]
        assert plot._local_transform_readers.get(id(base)) is reader.parent_reader

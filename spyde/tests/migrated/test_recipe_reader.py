"""RecipeReader — a mapped node's frames from the mapped function on the
PARENT's frame, with no dask block in the loop.

hyperspy's ``map`` runs, inside every block, exactly
``out[i] = function(block[i], **per_position[i], **constants)``; so frame i of
a mapped node is that call on the parent's frame i. Asking dask for
``derived.data[i]`` computes the whole enclosing block instead. Measured on a
real 5-D .zspy (64x64 navigation chunks of 128^2 uint16), constant-shift
centring:

    first frame in a chunk through the dask block    2196 ms
    the same frame through the recipe                0.55 ms   (bit-identical)

The correctness bar is bit-for-bit parity with the block (``array_equal``,
never ``allclose``): a silently wrong frame is far worse than a slow one, so
anything the recipe path cannot reproduce exactly must decline to the block
path, and the tests below pin both halves.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest
from scipy import ndimage

from spyde.tests.migrated.conftest import _settle, close_session, make_session
from spyde.external.hyperspy.map_recipe import apply as apply_map_recipe, recipe_for
from spyde.array_cache import ArrayCache, BlockCache, get_local_frame
from spyde.array_cache.nav_read import _get_local_region
from spyde.array_cache.readers.recipe import RecipeReader, chain_reaches
from spyde.array_cache.readers.per_frame import PerFrameReader
from spyde.array_cache.readers.local_transform import LocalTransformReader

assert apply_map_recipe()   # the patch under test; idempotent


# ── scaffolding: a chain of nodes over a lazy base ───────────────────────────

class _Node:
    def __init__(self, signal, parent=None, transformation=None, kwargs=None):
        self.signal = signal
        self.parent = parent
        self.transformation = transformation
        self.kwargs = kwargs or {}
        self.local = True
        self._resolved_local = True


class _Tree:
    """``chain`` is [(signal, transformation), ...] from the base upwards."""

    def __init__(self, base, chain):
        self._nodes = {id(base): _Node(base)}
        parent = self._nodes[id(base)]
        for signal, transformation in chain:
            node = _Node(signal, parent, transformation)
            self._nodes[id(signal)] = node
            parent = node

    def get_node(self, signal):
        return self._nodes.get(id(signal))

    def resolve_locality(self, signal):
        return True


class _Plot:
    def __init__(self, tree):
        self._array_cache = ArrayCache()
        self._block_cache = BlockCache()
        self._local_transform_readers = {}
        self.signal_tree = tree


def _lazy_4d(nav=8, sig=8, chunk=4, dtype=np.uint16, seed=0):
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 500, (nav, nav, sig, sig)).astype(dtype)
    d = da.from_array(arr, chunks=(chunk, chunk, sig, sig))
    return hs.signals.Signal2D(d).as_lazy(), arr


def _shifts_signal(nav, seed=1):
    """A per-position (dx, dy) shift table with sub-pixel values."""
    rng = np.random.RandomState(seed)
    table = rng.uniform(-1.5, 1.5, (nav, nav, 2)).astype(np.float32)
    return hs.signals.Signal1D(table), table


def _shift_frame(frame, shifts, order=1):
    return ndimage.shift(frame, np.asarray(shifts)[::-1], order=order)


def _scale_frame(frame, factor=1.0):
    return frame * factor


def _centred(base, nav, order=1):
    """A per-position shifted node built through the public map, as an action would."""
    shifts, table = _shifts_signal(nav)
    out = base.map(_shift_frame, shifts=shifts, order=order, inplace=False,
                   output_signal_size=base.axes_manager.signal_shape[::-1],
                   output_dtype=base.data.dtype)
    return out, table


POINTS = [(0, 0), (1, 3), (3, 3), (4, 4), (5, 6), (7, 7)]   # interior, chunk corners, edge


def _block_frame(signal, point):
    return np.asarray(signal.data[point].compute(scheduler="synchronous"))


# ── the recorder ──────────────────────────────────────────────────────────────

class TestRecorder:
    def test_map_output_carries_its_recipe(self):
        base, _ = _lazy_4d()
        out = base.map(_scale_frame, factor=2.0, inplace=False,
                       output_signal_size=(8, 8), output_dtype=np.uint16)
        recipe = recipe_for(out)
        assert recipe is not None
        assert recipe.function is _scale_frame
        assert recipe.static == {"factor": 2.0}
        assert recipe.iterating == {}
        assert recipe.source is base
        assert recipe.output_shape == (8, 8) and recipe.output_dtype == np.uint16

    def test_eager_per_position_argument_is_snapshotted(self):
        """map rebinds an eager argument signal to a lazy copy; the recipe keeps
        the array the caller passed."""
        base, _ = _lazy_4d()
        out, table = _centred(base, 8)
        recipe = recipe_for(out)
        assert isinstance(recipe.iterating["shifts"], np.ndarray)
        np.testing.assert_array_equal(recipe.iterating["shifts"], table)
        assert recipe.static == {"order": 1}

    def test_inplace_map_records_nothing_and_clears_a_stale_recipe(self):
        base, _ = _lazy_4d()
        out = base.map(_scale_frame, factor=2.0, inplace=False,
                       output_signal_size=(8, 8), output_dtype=np.uint16)
        assert recipe_for(out) is not None
        out.map(_scale_frame, factor=3.0, inplace=True,
                output_signal_size=(8, 8), output_dtype=np.uint16)
        assert recipe_for(out) is None

    def test_swapped_data_invalidates_the_recipe(self):
        base, _ = _lazy_4d()
        out = base.map(_scale_frame, factor=2.0, inplace=False,
                       output_signal_size=(8, 8), output_dtype=np.uint16)
        out.data = da.zeros_like(out.data)
        assert recipe_for(out) is None

    def test_recorder_is_observe_only(self):
        """The wrapped map builds the same graph, with the same name, as the
        original: the batch path cannot see the recorder."""
        from hyperspy.signal import BaseSignal
        base, _ = _lazy_4d()
        kwargs = dict(factor=2.0, inplace=False,
                      output_signal_size=(8, 8), output_dtype=np.uint16)
        wrapped = base.map(_scale_frame, **kwargs)
        original = BaseSignal.map._spyde_original(base, _scale_frame, **kwargs)
        assert wrapped.data.name == original.data.name
        assert recipe_for(original) is None

    def test_eager_output_carries_nothing(self):
        base, _ = _lazy_4d()
        out = base.map(_scale_frame, factor=2.0, inplace=False, lazy_output=False,
                       output_signal_size=(8, 8), output_dtype=np.uint16)
        assert recipe_for(out) is None


# ── the reader: parity with the block ─────────────────────────────────────────

class TestParityWithTheBlock:
    @pytest.mark.parametrize("dtype", [np.uint16, np.float32])
    def test_frame_matches_the_dask_block(self, dtype):
        base, _ = _lazy_4d(dtype=dtype)
        out, _ = _centred(base, 8)
        plot = _Plot(_Tree(base, [(out, "map")]))
        for point in POINTS:
            got = get_local_frame(plot, out, out.data, np.array(point))
            np.testing.assert_array_equal(np.asarray(got), _block_frame(out, point))
        assert isinstance(plot._local_transform_readers[id(out)], RecipeReader)

    def test_float_result_truncates_like_the_block(self):
        """The block path writes the function's float result into a uint16
        array by assignment; the recipe must cast the same way."""
        base, _ = _lazy_4d(dtype=np.uint16)
        out = base.map(_scale_frame, factor=0.37, inplace=False,
                       output_signal_size=(8, 8), output_dtype=np.uint16)
        plot = _Plot(_Tree(base, [(out, "map")]))
        for point in POINTS:
            got = get_local_frame(plot, out, out.data, np.array(point))
            np.testing.assert_array_equal(np.asarray(got), _block_frame(out, point))

    def test_scalar_per_position_argument(self):
        """A per-position argument with signal shape (1,) reaches the function
        as a scalar in the block; the recipe squeezes it the same way."""
        base, _ = _lazy_4d(dtype=np.float32)
        gains = hs.signals.Signal1D(np.linspace(0.5, 2.0, 64, dtype=np.float32).reshape(8, 8, 1))

        def _gain(frame, gain):
            assert np.ndim(gain) == 0
            return frame * gain

        out = base.map(_gain, gain=gains, inplace=False,
                       output_signal_size=(8, 8), output_dtype=np.float32)
        plot = _Plot(_Tree(base, [(out, "map")]))
        for point in POINTS:
            got = get_local_frame(plot, out, out.data, np.array(point))
            np.testing.assert_array_equal(np.asarray(got), _block_frame(out, point))

    def test_lazy_per_position_argument_with_its_own_recipe(self):
        """Shifts computed lazily by a map on the same parent (the automatic
        centre) resolve through their own recipe: one call of the shift finder
        per frame read, not one block of them."""
        base, _ = _lazy_4d(dtype=np.float32)
        calls = []

        def _find_shift(frame):
            calls.append(1)
            centre = np.array(frame.shape[::-1]) / 2.0
            return (centre - np.array([1.0, 2.0])).astype(np.float32)

        shifts = base.map(_find_shift, inplace=False, lazy_output=True,
                          output_signal_size=(2,), output_dtype=np.float32)
        out = base.map(_shift_frame, shifts=shifts, order=1, inplace=False,
                       output_signal_size=(8, 8), output_dtype=np.float32)
        assert chain_reaches(out, base)

        expected = _block_frame(out, (2, 5))
        calls.clear()
        plot = _Plot(_Tree(base, [(out, "map")]))
        got = get_local_frame(plot, out, out.data, np.array((2, 5)))
        np.testing.assert_array_equal(np.asarray(got), expected)
        assert len(calls) == 1

    def test_chain_leaving_the_tree_declines(self):
        """A method that mapped an intermediate the tree never saw has no
        per-frame definition rooted at the parent: the block path serves it."""
        base, _ = _lazy_4d()
        intermediate = base.rebin(scale=[1, 1, 2, 2])
        out = intermediate.map(_scale_frame, factor=2.0, inplace=False,
                               output_signal_size=(4, 4), output_dtype=np.uint16)
        assert not chain_reaches(out, base)
        plot = _Plot(_Tree(base, [(out, "map")]))
        got = get_local_frame(plot, out, out.data, np.array((3, 3)))
        assert isinstance(plot._local_transform_readers[id(out)], LocalTransformReader)
        np.testing.assert_array_equal(np.asarray(got), _block_frame(out, (3, 3)))

    def test_rebin_over_a_mapped_node(self):
        """Centre, then bin: the per-frame rebin reader composes over the
        recipe reader, and neither computes a block."""
        base, _ = _lazy_4d()
        out, _ = _centred(base, 8)
        binned = out.rebin(scale=[1, 1, 2, 2])
        plot = _Plot(_Tree(base, [(out, "map"), (binned, "rebin")]))
        for point in POINTS:
            got = get_local_frame(plot, binned, binned.data, np.array(point))
            np.testing.assert_array_equal(np.asarray(got), _block_frame(binned, point))
        reader = plot._local_transform_readers[id(binned)]
        assert isinstance(reader, PerFrameReader)
        assert isinstance(reader.parent_reader, RecipeReader)

    def test_map_over_a_rebinned_node(self):
        """Bin, then centre: the recipe's source is the rebin node, whose own
        reader is the per-frame rebin over the root."""
        base, _ = _lazy_4d()
        binned = base.rebin(scale=[1, 1, 2, 2])
        shifts, _ = _shifts_signal(8)
        out = binned.map(_shift_frame, shifts=shifts, order=1, inplace=False,
                         output_signal_size=(4, 4), output_dtype=np.uint16)
        plot = _Plot(_Tree(base, [(binned, "rebin"), (out, "map")]))
        for point in POINTS:
            got = get_local_frame(plot, out, out.data, np.array(point))
            np.testing.assert_array_equal(np.asarray(got), _block_frame(out, point))

    def test_region_mean_matches_the_per_frame_mean(self):
        base, _ = _lazy_4d()
        out, _ = _centred(base, 8)
        plot = _Plot(_Tree(base, [(out, "map")]))
        yy, xx = np.meshgrid(np.arange(2, 6), np.arange(2, 6), indexing="ij")
        idx = np.stack([yy.ravel(), xx.ravel()], axis=1)     # straddles 4 blocks

        got = _get_local_region(plot, out, out.data, idx)
        acc = None
        for p in idx:
            f = _block_frame(out, (p[0], p[1]))
            acc = f.astype(np.float64) if acc is None else acc + f
        expected = np.rint(acc / len(idx)).astype(out.data.dtype)
        np.testing.assert_array_equal(got, expected)

    def test_concurrent_reads_agree(self):
        """The dispatcher and the overlay warm read the same node at once."""
        base, _ = _lazy_4d()
        out, _ = _centred(base, 8)
        plot = _Plot(_Tree(base, [(out, "map")]))
        expected = {p: _block_frame(out, p) for p in POINTS}
        failures = []

        def _worker():
            for _ in range(5):
                for p in POINTS:
                    got = get_local_frame(plot, out, out.data, np.array(p))
                    if not np.array_equal(np.asarray(got), expected[p]):
                        failures.append(p)

        threads = [threading.Thread(target=_worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert not failures

    def test_eager_root_is_served(self):
        """The bundled synthetic datasets are in RAM; the recipe reads the
        parent's frame by index."""
        arr = np.random.RandomState(0).randint(0, 500, (6, 6, 8, 8)).astype(np.uint16)
        base = hs.signals.Signal2D(arr)
        shifts, _ = _shifts_signal(6)
        out = base.map(_shift_frame, shifts=shifts, order=1, inplace=False,
                       output_signal_size=(8, 8), output_dtype=np.uint16)
        assert out._lazy is False   # an eager map yields eager output
        lazy_out = base.map(_shift_frame, shifts=shifts, order=1, inplace=False,
                            lazy_output=True, output_signal_size=(8, 8),
                            output_dtype=np.uint16)
        plot = _Plot(_Tree(base, [(lazy_out, "map")]))
        for point in [(0, 0), (2, 4), (5, 5)]:
            got = get_local_frame(plot, lazy_out, lazy_out.data, np.array(point))
            np.testing.assert_array_equal(np.asarray(got), _block_frame(lazy_out, point))

    def test_pyxem_center_direct_beam_is_served(self):
        """The real method the Center Zero Beam action calls."""
        base, _ = _lazy_4d(dtype=np.float32)
        base.set_signal_type("electron_diffraction")
        shifts, _ = _shifts_signal(8)
        out = base.center_direct_beam(shifts=shifts, inplace=False)
        assert chain_reaches(out, base)
        plot = _Plot(_Tree(base, [(out, "center_direct_beam")]))
        for point in POINTS:
            got = get_local_frame(plot, out, out.data, np.array(point))
            np.testing.assert_array_equal(np.asarray(got), _block_frame(out, point))
        assert isinstance(plot._local_transform_readers[id(out)], RecipeReader)


# ── the tree and the node switch, through a real Session ──────────────────────

def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=25.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _lazy_beam_4d(nav=(4, 4), sig=(32, 32), beam=(18, 14)):
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - beam[0]) ** 2 + (yy - beam[1]) ** 2 <= 9).astype(np.float32)
    data = np.zeros(nav + sig, dtype=np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * (100.0 + idx[0] * 10 + idx[1])
    s = hs.signals.Signal2D(da.from_array(data, chunks=(2, 2) + sig)).as_lazy()
    s.set_signal_type("electron_diffraction")
    return s


class TestThroughTheSession:
    def test_manual_centre_node_is_local_and_reads_per_frame(self):
        from spyde.actions.center_zero_beam import czb_open, czb_pick
        session = make_session()
        try:
            session._add_signal(_lazy_beam_4d())
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree
            root = src.plot_state.current_signal

            czb_open(session, src, {})
            tree._czb_cross.set(cx=18.0, cy=14.0)
            czb_pick(session, src, {})
            assert _wait(lambda: src.plot_state.current_signal is not root)
            centred = src.plot_state.current_signal

            node = tree.get_node(centred)
            assert node.local is True
            assert tree.resolve_locality(centred)
            for point in [(0, 0), (1, 2), (3, 3)]:
                got = get_local_frame(src, centred, centred.data, np.array(point))
                np.testing.assert_array_equal(np.asarray(got), _block_frame(centred, point))
            assert isinstance(src._local_transform_readers[id(centred)], RecipeReader)
        finally:
            close_session(session)

    def test_node_switch_keeps_the_ancestor_chain(self):
        from spyde.actions.center_zero_beam import czb_open, czb_pick
        from spyde.actions.lifecycle import show_tree_node
        session = make_session()
        try:
            session._add_signal(_lazy_beam_4d())
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree
            root = src.plot_state.current_signal

            # Decode a root block on the root node.
            get_local_frame(src, root, root.data, np.array((1, 1)))
            root_reader = src._local_transform_readers[id(root)]
            assert root_reader.is_chunk_resident((1, 1))

            czb_open(session, src, {})
            tree._czb_cross.set(cx=18.0, cy=14.0)
            czb_pick(session, src, {})
            assert _wait(lambda: src.plot_state.current_signal is not root)
            centred = src.plot_state.current_signal

            # The switch kept the root's reader and its decoded block.
            assert src._local_transform_readers.get(id(root)) is root_reader
            assert root_reader.is_chunk_resident((1, 1))
            get_local_frame(src, centred, centred.data, np.array((1, 1)))
            assert id(centred) in src._local_transform_readers

            # Back to the root: the root stays, the centred node's reader goes.
            show_tree_node(src, tree, root)
            assert _wait(lambda: src.plot_state.current_signal is root)
            assert src._local_transform_readers.get(id(root)) is root_reader
            assert root_reader.is_chunk_resident((1, 1))
            assert id(centred) not in src._local_transform_readers
        finally:
            close_session(session)

"""One route for everything that follows the navigator.

An overlay is a child node of the node a window displays. It holds no data: it
holds a recipe, evaluated at the navigator's position through the readers the
base frame already uses, and drawn by the painter thread. These tests pin the
mechanism rather than any particular overlay: what a recipe may read, where the
work runs, and when the drawing is cleared.

Every read runs on the lazy, chunked fixture, so the reader chain and the caches
are the ones the app uses rather than a numpy index.
"""
from __future__ import annotations

import itertools
import threading
import time

import dask.array as da
import numpy as np
import hyperspy.api as hs

from spyde.array_cache import reader_for_overlay
from spyde.array_cache.readers.binary import BinaryReader
from spyde.array_cache.readers.recipe import chain_reaches, evaluate
from spyde.drawing.overlay_node import OverlaySignal
from spyde.drawing.overlays import refresh_overlays
from spyde.external.hyperspy.map_recipe import (
    FrameRecipe, apply as apply_map_recipe, recipe_for,
)
from spyde.tests.migrated.conftest import _settle
from spyde.tests.migrated.test_array_cache_binary_reader import _write_synthetic_mrc
from spyde.tests.migrated.test_center_zero_beam import _signal_plot, _wait

# The map wrapper is applied with the heavy imports in the real app; a test
# that builds a mapped node itself has to ask for it. Idempotent.
assert apply_map_recipe()

BEAM = (18, 14)          # column, row of the un-centred disk
CENTRE = (16.0, 16.0)    # where centring puts it on a 32x32 frame


def _off_centre_lazy(nav=(8, 8), sig=(32, 32), beam=BEAM, chunk=4):
    """A lazy scan in ``chunk`` x ``chunk`` navigation blocks, so a mapped child
    has a real block to avoid computing."""
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - beam[0]) ** 2 + (yy - beam[1]) ** 2 <= 9).astype(np.float32)
    data = np.zeros(nav + sig, dtype=np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    signal = hs.signals.Signal2D(data).as_lazy()
    signal.data = signal.data.rechunk((chunk, chunk, -1, -1))
    signal.set_signal_type("electron_diffraction")
    return signal


def _indexed_lazy(nav=(8, 8), sig=(8, 8), chunk=4):
    """The same shape of scan with a different constant per position, so a
    window of frames can be told apart from any other window."""
    data = np.zeros(nav + sig, dtype=np.float32)
    for y, x in np.ndindex(*nav):
        data[y, x] = float(y * 10 + x)
    signal = hs.signals.Signal2D(data).as_lazy()
    signal.data = signal.data.rechunk((chunk, chunk, -1, -1))
    signal.set_signal_type("electron_diffraction")
    return signal


def _spectra_lazy(nav=(4, 4), channels=16):
    """A lazy spectrum image, for the curve group kind."""
    data = np.tile(np.arange(channels, dtype=np.float32), nav + (1,))
    signal = hs.signals.Signal1D(data).as_lazy()
    signal.data = signal.data.rechunk((2, 2, -1))
    return signal


def _centre_of_mass(frame):
    frame = np.asarray(frame, dtype=np.float64)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    total = frame.sum()
    return float((xx * frame).sum() / total), float((yy * frame).sum() / total)


def _centre(session, src):
    """Run Center Zero Beam on ``src`` and return the centred node's signal."""
    from spyde.actions.center_zero_beam import czb_run
    before = src.plot_state.current_signal
    czb_run(session, src, {"method": "center_of_mass"})
    assert _wait(lambda: src.plot_state.current_signal is not before, 30), \
        "centering never produced a new signal"
    centred = src.plot_state.current_signal
    assert _wait(lambda: (src.current_data is not None
                          and np.allclose(_centre_of_mass(src.current_data), CENTRE,
                                          atol=0.5)),
                 20), "the window never painted the centred frame"
    return centred


def _open_session(signal):
    """A settled session showing ``signal``, with its signal plot."""
    from spyde.backend.session import Session
    session = Session(n_workers=1, threads_per_worker=1)
    session._add_signal(signal)
    _settle(session)
    return session, _signal_plot(session)


def _navigator_selectors(tree):
    manager = tree.navigator_plot_manager
    return [sel for sels in manager.navigation_selectors.values() for sel in sels]


def _move_navigator(session, tree):
    for selector in _navigator_selectors(tree):
        selector.delayed_update_data(force=True)
    _settle(session)


def _group_keys(plot, node):
    return [key for key in plot._overlay_groups if key[0] == id(node)]


def _workflow_tree(session, plot) -> dict:
    """The workflow tree as the renderer would receive it."""
    from spyde.backend import session as session_module

    payload = {}

    def capture(message):
        if message.get("type") == "signal_tree":
            payload.update(message["tree"])

    original = session_module.emit
    session_module.emit = capture
    try:
        session._reemit_signal_tree(plot)
    finally:
        session_module.emit = original
    return payload


def _node_names(payload) -> list:
    names = [payload.get("name")]
    for child in payload.get("children", []):
        names.extend(_node_names(child))
    return names


class _RowStore:
    """A per-position source: it answers one navigation index at a time."""

    def __init__(self, rows):
        self._rows = rows
        self.asked = []

    def at(self, *index):
        self.asked.append(tuple(index))
        return self._rows.get(tuple(index))


class _ConstantReader:
    """A reader override serving one frame everywhere, or nothing at all."""

    def __init__(self, frame):
        self.frame = frame
        self.data = None

    @property
    def frame_bytes(self) -> int:
        return 0 if self.frame is None else int(self.frame.nbytes)

    def read_frame(self, indices):
        return self.frame


class _RampReader(_ConstantReader):
    """A reader override whose frame value is the navigation position, so an
    integrating region has a mean worth asserting on."""

    def __init__(self, shape, dtype=np.float32):
        super().__init__(None)
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def read_frame(self, indices):
        return np.full(self.shape, float(sum(indices)), dtype=self.dtype)


class _RegionReader(_RampReader):
    """The same, with a region rule of its own: the frames summed rather than
    averaged, which is what a store that renders counts wants."""

    def region_frame(self, points):
        total = np.zeros(self.shape, dtype=np.float64)
        for row in np.asarray(points):
            total += self.read_frame(tuple(int(v) for v in row))
        return total


class _ThreadRecorder:
    """Wraps a bound method and records the thread each call ran on."""

    def __init__(self, owner, name):
        self.threads = []
        self._wrapped = getattr(owner, name)
        setattr(owner, name, self)

    def __call__(self, *args, **kwargs):
        self.threads.append(threading.current_thread().name)
        return self._wrapped(*args, **kwargs)


class TestRecipeSources:
    def test_a_recipe_with_no_source_reads_its_per_position_store(self):
        rows = {(0, 0): np.array([[1.0, 2.0]]),
                (1, 2): np.array([[3.0, 4.0], [5.0, 6.0]])}
        recipe = FrameRecipe(
            function=lambda *, spots: spots,
            static={}, iterating={"spots": _RowStore(rows)}, source=None,
            output_name=None, output_shape=None, output_dtype=None,
        )
        assert np.array_equal(evaluate(recipe, (1, 2), None, None), rows[(1, 2)])
        assert np.array_equal(evaluate(recipe, (0, 0), None, None), rows[(0, 0)])

    def test_a_recipe_with_no_source_needs_no_chain_to_the_parent(self):
        parent = _off_centre_lazy(nav=(2, 2), sig=(8, 8))
        recipe = FrameRecipe(
            function=lambda *, spots: spots,
            static={}, iterating={"spots": _RowStore({})}, source=None,
            output_name=None, output_shape=None, output_dtype=None,
        )
        assert chain_reaches(OverlaySignal(parent, recipe), parent)

    def test_a_five_dimensional_store_is_asked_time_first(self):
        signal = hs.signals.Signal2D(
            np.zeros((3, 4, 5, 8, 8), dtype=np.float32)).as_lazy()
        signal.data = signal.data.rechunk((1, 2, 2, -1, -1))
        session, plot = _open_session(signal)
        try:
            tree = plot.signal_tree
            store = _RowStore({(2, 1, 3): np.array([[7.0, 8.0]])})
            node = tree.add_overlay(tree.root, lambda *, spots: spots,
                                    name="spots", groups={}, source=False,
                                    iterating={"spots": store})
            value = reader_for_overlay(plot, node).read_frame((2, 1, 3))
            assert store.asked == [(2, 1, 3)], store.asked
            assert np.array_equal(value, [[7.0, 8.0]])
        finally:
            session.shutdown()

    def test_a_ragged_map_node_evaluates_to_its_block_value(self):
        signal = _indexed_lazy(nav=(4, 4), sig=(4, 4), chunk=2)

        def spots(frame):
            return np.array([[float(frame[0, 0]), 1.0], [2.0, 3.0]])

        mapped = signal.map(spots, inplace=False, lazy_output=True, ragged=True)
        recipe = recipe_for(mapped)
        assert recipe is not None and recipe.output_shape is None

        def parent_frame(index):
            return np.asarray(signal.data[index].compute())

        blocks = mapped.deepcopy()
        blocks.compute()
        for index in ((0, 0), (1, 2), (3, 3)):
            value = evaluate(recipe, index, signal, parent_frame)
            assert np.array_equal(value, blocks.data[index]), index


class TestNavigationDepth:
    """``depth`` asks for a window of source frames instead of one."""

    def _window_at(self, session, plot, index, depth=1):
        """The shape and centre index of the window a depth recipe receives at
        ``index`` on the CENTRED node, which is read through its recipe."""
        tree = plot.signal_tree
        centred = _centre(session, plot)
        seen = []

        def record(window, centre, *, sink):
            sink.append((tuple(window.shape), tuple(centre)))
            return {}

        node = tree.add_overlay(centred, record, name="window", groups={},
                                static={"sink": seen}, depth=depth)
        reader_for_overlay(plot, node).read_frame(index)
        assert len(seen) == 1, seen
        return seen[0]

    def test_an_interior_position_gets_the_full_window(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            shape, centre = self._window_at(session, plot, (4, 4))
            assert shape == (3, 3, 32, 32), shape
            assert centre == (1, 1), centre
        finally:
            session.shutdown()

    def test_every_corner_is_clipped_on_the_sides_it_touches(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            corners = {(0, 0): (0, 0), (0, 7): (0, 1),
                       (7, 0): (1, 0), (7, 7): (1, 1)}
            for index, expected_centre in corners.items():
                shape, centre = self._window_at(session, plot, index)
                assert shape == (2, 2, 32, 32), (index, shape)
                assert centre == expected_centre, (index, centre)
        finally:
            session.shutdown()

    def test_the_window_holds_the_source_frames_around_the_position(self):
        session, plot = _open_session(_indexed_lazy())
        try:
            tree = plot.signal_tree
            frames = []

            def record(window, centre, *, sink):
                sink.append(np.asarray(window))
                return {}

            node = tree.add_overlay(tree.root, record, name="window", groups={},
                                    static={"sink": frames}, depth=1)
            reader_for_overlay(plot, node).read_frame((4, 4))
            expected = np.asarray(tree.root.data[3:6, 3:6].compute())
            assert np.array_equal(frames[0], expected)
        finally:
            session.shutdown()

    def test_an_opaque_parent_is_read_as_its_block(self):
        """A node the locality gate rejects has its dask block for a
        footprint, which is slower than a recipe and still a real frame."""
        session, plot = _open_session(_indexed_lazy())
        try:
            tree = plot.signal_tree
            opaque = tree.add_transformation(
                tree.root, function=lambda signal: signal.deepcopy(),
                node_name="Opaque", local=False)
            assert not tree.resolve_locality(opaque)

            single = tree.add_overlay(opaque, lambda frame: {"doubled": frame * 2.0},
                                      name="single", groups={})
            value = reader_for_overlay(plot, single).read_frame((2, 3))
            expected = np.asarray(opaque.data[2, 3].compute()) * 2.0
            assert np.array_equal(value["doubled"], expected)

            frames = []
            window = tree.add_overlay(
                opaque, lambda w, centre, *, sink: sink.append(np.asarray(w)) or {},
                name="window", groups={}, static={"sink": frames}, depth=1)
            reader_for_overlay(plot, window).read_frame((2, 3))
            assert np.array_equal(
                frames[0], np.asarray(opaque.data[1:4, 2:5].compute()))
        finally:
            session.shutdown()

    def test_a_window_re_reads_only_the_frames_that_moved(self, tmp_path):
        """On a memmap movie the reader keeps nothing, so a window that did not
        go through the frame cache would read every frame again on every move."""
        data = (np.arange(30 * 8 * 8, dtype=np.uint16).reshape(30, 8, 8))
        path = str(tmp_path / "movie.mrc")
        _write_synthetic_mrc(path, data)

        session, plot = _open_session(hs.load(path, lazy=True))
        try:
            tree = plot.signal_tree

            def take_centre(window, centre):
                return {}

            node = tree.add_overlay(tree.root, take_centre, name="window",
                                    groups={}, depth=3)
            reader = reader_for_overlay(plot, node)
            reader.read_frame((10,))          # resolves the parent's reader

            source = plot._local_transform_readers[id(tree.root)]
            assert isinstance(source, BinaryReader), type(source).__name__
            counter = _ThreadRecorder(source, "read_frame")

            plot._array_cache.clear()
            reader.read_frame((10,))
            cold = len(counter.threads)
            reader.read_frame((11,))
            moved = len(counter.threads) - cold

            assert cold == 7, cold
            assert moved == 1, moved
        finally:
            session.shutdown()


class TestOverlayNodeLifecycle:
    GROUPS = {"found": ("circles", {"radius": 3.0}),
              "bands": ("lines", {"linewidths": 1.0})}

    def test_a_group_is_created_and_removed_per_entry(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            node = tree.add_overlay(tree.root, lambda frame: {}, name="markers",
                                    groups=self.GROUPS)
            keys = _group_keys(plot, node)
            assert sorted(key[1] for key in keys) == ["bands", "found"], keys
            assert tree.overlay_children(tree.root) == [node]

            tree.remove_overlay(node)
            assert _group_keys(plot, node) == []
            assert tree.overlay_children(tree.root) == []
        finally:
            session.shutdown()

    def test_close_removes_every_group(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            node = tree.add_overlay(tree.root, lambda frame: {}, name="markers",
                                    groups=self.GROUPS)
            assert len(_group_keys(plot, node)) == 2
            tree.close()
            assert _group_keys(plot, node) == []
        finally:
            session.shutdown()

    def test_an_overlay_is_neither_a_plot_state_nor_a_workflow_node(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            node = tree.add_overlay(tree.root, lambda frame: {}, name="markers",
                                    groups=self.GROUPS)
            assert node.signal not in plot.plot_states
            names = _node_names(_workflow_tree(session, plot))
            assert "markers" not in names, names
        finally:
            session.shutdown()

    def test_a_plot_opened_after_the_overlay_still_draws_it(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            node = tree.add_overlay(
                tree.root,
                lambda frame: {"found": np.array([[4.0, 5.0]], dtype=np.float32)},
                name="markers", groups={"found": ("circles", {"radius": 3.0})})

            tree.add_signal_plot()
            later = tree.signal_plots[-1]
            assert later is not plot
            assert _group_keys(later, node) == []

            refresh_overlays(later, _navigator_selectors(tree)[0].current_indices)
            assert _wait(lambda: (id(node), "found") in later._overlay_groups, 10)
            group = later._overlay_groups[(id(node), "found")]
            assert _wait(lambda: len(np.asarray(group._data["offsets"])) == 1, 10)
        finally:
            session.shutdown()


class TestGroupKinds:
    """Every kind pushes on the painter thread and clears on a missing value."""

    def _draw_and_clear(self, session, plot, node, recorder, value):
        plot.enqueue_overlay(node, value)
        assert _wait(lambda: recorder.threads, 10), "the value never drew"
        assert set(recorder.threads) == {"nav-paint"}, recorder.threads
        drawn = len(recorder.threads)
        plot.enqueue_overlay(node, {})
        assert _wait(lambda: len(recorder.threads) > drawn, 10), "the clear never ran"
        assert set(recorder.threads) == {"nav-paint"}, recorder.threads

    def test_lines_and_arrows_and_layer_and_transform(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            node = tree.add_overlay(
                tree.root, lambda frame: {}, name="kinds",
                groups={"bands": ("lines", {"linewidths": 1.0}),
                        "vectors": ("arrows", {}),
                        "sheet": ("layer", {"alpha": 0.5}),
                        "detector": ("transform", {})})

            bands = plot._overlay_groups[(id(node), "bands")]
            self._draw_and_clear(session, plot, node, _ThreadRecorder(bands, "set"),
                                 {"bands": np.array([[[0.0, 0.0], [4.0, 4.0]]])})
            assert len(np.asarray(bands._data["segments"])) == 0

            arrows = plot._overlay_groups[(id(node), "vectors")]
            self._draw_and_clear(
                session, plot, node, _ThreadRecorder(arrows, "set"),
                {"vectors": (np.array([[1.0, 2.0]]), np.array([3.0]),
                             np.array([4.0]))})
            assert len(np.asarray(arrows._data["offsets"])) == 0

            # A layer is built on its first image, so record the push after
            # that, and a cleared layer is hidden rather than redrawn empty.
            plot.enqueue_overlay(node, {"sheet": np.zeros((32, 32), np.float32)})
            assert _wait(lambda: plot._overlay_groups[(id(node), "sheet")] is not None,
                         10)
            sheet = plot._overlay_groups[(id(node), "sheet")]
            pushed = _ThreadRecorder(sheet, "set_data")
            plot.enqueue_overlay(node, {"sheet": np.ones((32, 32), np.float32)})
            assert _wait(lambda: pushed.threads, 10), "the layer never drew"
            assert set(pushed.threads) == {"nav-paint"}, pushed.threads
            hidden = _ThreadRecorder(sheet, "set")
            plot.enqueue_overlay(node, {})
            assert _wait(lambda: hidden.threads, 10), "the layer never cleared"
            assert set(hidden.threads) == {"nav-paint"}, hidden.threads

            painted = _ThreadRecorder(plot, "set_transform_image")
            plot.enqueue_overlay(node, {"detector": np.ones((32, 32), np.float32)})
            assert _wait(lambda: painted.threads, 10)
            assert set(painted.threads) == {"nav-paint"}, painted.threads
            assert bool(plot._live_transform_groups)
            plot.enqueue_overlay(node, {})
            assert _wait(lambda: not plot._live_transform_groups, 10)
        finally:
            session.shutdown()

    def test_a_value_may_carry_the_appearance_to_draw_it_with(self):
        """A slider that changes a marker radius rides the next value: the
        group is pushed with the new appearance, never rebuilt for it."""
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            node = tree.add_overlay(
                tree.root, lambda frame: {}, name="sized",
                groups={"found": ("circles", {"radius": 3.0})})
            group = plot._overlay_groups[(id(node), "found")]
            pushed = _ThreadRecorder(group, "set")

            plot.enqueue_overlay(node, {"found": {
                "data": np.array([[4.0, 5.0]], dtype=np.float32), "radius": 9.0}})
            assert _wait(lambda: pushed.threads, 10), "the value never drew"
            assert set(pushed.threads) == {"nav-paint"}, pushed.threads
            assert plot._overlay_groups[(id(node), "found")] is group, \
                "the group was rebuilt for an appearance change"
            assert float(group._data["radius"]) == 9.0
            assert len(np.asarray(group._data["offsets"])) == 1
        finally:
            session.shutdown()

    def test_a_live_transform_holds_the_base_frame_back(self):
        """A transform group's image IS the frame the plot shows, so the
        navigator's own paint waits until the group's value is None again."""
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            node = tree.add_overlay(
                tree.root, lambda frame: {}, name="detector",
                groups={"response": ("transform", {})})
            response = np.ones((32, 32), np.float32)
            plot.enqueue_overlay(node, {"response": response})
            assert _wait(lambda: bool(plot._live_transform_groups), 10)

            raw = np.full((32, 32), 7.0, np.float32)
            painted = _ThreadRecorder(plot, "_set_array")
            plot.enqueue_paint(raw)
            assert _wait(lambda: plot.current_data is raw, 10)
            time.sleep(0.3)                       # give a wrong paint time to land
            assert painted.threads == [], "the raw frame painted over the transform"

            plot.enqueue_overlay(node, {})
            assert _wait(lambda: not plot._live_transform_groups, 10)
            plot.enqueue_paint(raw)
            assert _wait(lambda: painted.threads, 10), "the raw frame never came back"
        finally:
            session.shutdown()

    def test_curves_grow_and_trim_with_the_value(self):
        session, plot = _open_session(_spectra_lazy())
        try:
            tree = plot.signal_tree
            node = tree.add_overlay(tree.root, lambda frame: {}, name="model",
                                    groups={"components": ("curves", {})})
            x = np.arange(16, dtype=float)

            plot.enqueue_overlay(node, {"components": [(x, x), (x, 2 * x)]})
            key = (id(node), "components")
            assert _wait(lambda: len(plot._overlay_groups[key]) == 2, 10), \
                plot._overlay_groups[key]

            drawn = _ThreadRecorder(plot._overlay_groups[key][0], "set_data")
            plot.enqueue_overlay(node, {"components": [(x, 3 * x)]})
            assert _wait(lambda: len(plot._overlay_groups[key]) == 1, 10)
            assert set(drawn.threads) == {"nav-paint"}, drawn.threads

            plot.enqueue_overlay(node, {})
            assert _wait(lambda: len(plot._overlay_groups[key]) == 0, 10)
        finally:
            session.shutdown()


class TestTransformOrdering:
    """A transform group's image IS the frame the window shows, so the raw
    pattern must never appear on the way in or on the way out."""

    def _painted(self, plot):
        """Every array pushed to the figure, in order."""
        pushed = []
        plot._set_array = lambda data, levels=None: pushed.append(np.asarray(data))
        return pushed

    def _transform_overlay(self, tree, response):
        def detector(frame, *, show):
            return {"response": response if show else None}

        return tree.add_overlay(tree.root, detector, name="detector",
                                groups={"response": ("transform", {})},
                                static={"show": True})

    def test_entering_the_transform_view_never_shows_the_raw_frame(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            response = np.ones((32, 32), np.float32)
            node = self._transform_overlay(tree, response)
            pushed = self._painted(plot)

            _move_navigator(session, tree)
            assert _wait(lambda: pushed, 10), "nothing was painted"
            assert all(np.array_equal(one, response) for one in pushed),                 "the raw pattern was painted under the transform"
            assert plot.displayed_data is response
            assert node.visible
        finally:
            session.shutdown()

    def test_a_paint_arriving_before_the_first_value_still_waits(self):
        """The painter runs on its own thread, and `_run_update` stages the
        base frame before it evaluates the overlays — so a pass can reach the
        painter with nothing staged and the transform not yet live. It used to
        paint the raw pattern there, about one move in three on this machine.
        A transform group that has not answered holds the base back."""
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            response = np.ones((32, 32), np.float32)
            node = self._transform_overlay(tree, response)
            pushed = self._painted(plot)

            # The pass the navigator would have woken, with the overlay
            # attached and not yet evaluated: exactly the losing interleaving.
            plot.paint_pass(np.zeros((32, 32), np.float32))
            assert pushed == [], "the raw pattern was painted under the transform"

            _move_navigator(session, tree)
            assert _wait(lambda: pushed, 10), "nothing was painted"
            assert all(np.array_equal(one, response) for one in pushed)
            assert node.visible
        finally:
            session.shutdown()

    def test_a_transform_that_cannot_evaluate_hands_the_frame_back(self):
        """Nothing may wait on a transform for good. An overlay whose function
        raises clears its groups, so the base frame paints instead of the
        window staying on whatever was last up."""
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree

            def detector(frame, *, show):
                raise RuntimeError("no response today")

            tree.add_overlay(tree.root, detector, name="detector",
                             groups={"response": ("transform", {})},
                             static={"show": True})
            pushed = self._painted(plot)
            _move_navigator(session, tree)
            assert _wait(lambda: pushed, 10), "the base frame never came back"
        finally:
            session.shutdown()

    def test_leaving_it_paints_the_raw_frame_once(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            response = np.ones((32, 32), np.float32)
            node = self._transform_overlay(tree, response)
            _move_navigator(session, tree)
            assert _wait(lambda: bool(plot._live_transform_groups), 10)

            pushed = self._painted(plot)
            tree.replace_overlay_static(node, show=False)
            assert _wait(lambda: not plot._live_transform_groups, 10)
            _settle(session)
            assert len(pushed) == 1,                 f"the raw frame was painted {len(pushed)} times, not once"
            assert np.array_equal(pushed[0], plot.current_data)
            assert plot.displayed_data is plot.current_data
        finally:
            session.shutdown()

    def test_dropping_the_node_brings_the_pattern_back(self):
        """Closing the caret restores the pattern without waiting for a move:
        the node owned the image on screen, so it stages a repaint as it goes."""
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            node = self._transform_overlay(tree, np.ones((32, 32), np.float32))
            _move_navigator(session, tree)
            assert _wait(lambda: bool(plot._live_transform_groups), 10)
            raw = plot.current_data

            pushed = self._painted(plot)
            tree.remove_overlay(node)
            assert _wait(lambda: pushed, 10), "the pattern never came back"
            assert np.array_equal(pushed[-1], raw)
            assert not plot._live_transform_groups
        finally:
            session.shutdown()


class TestOverlaySourceOverride:
    def test_a_frame_overlay_reads_through_a_pinned_reader(self):
        """A window whose frames come from a pinned reader has no array to
        slice, so an overlay ON that node must read the reader's frame rather
        than the placeholder underneath it."""
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            frame = np.full((32, 32), 7.0, dtype=np.float32)
            tree.set_reader_override(tree.root, _ConstantReader(frame))

            seen = []
            node = tree.add_overlay(
                tree.root, lambda source: seen.append(np.asarray(source)) or {},
                name="reads", groups={"found": ("circles", {})})

            _move_navigator(session, tree)
            assert _wait(lambda: seen, 10), "the overlay never evaluated"
            assert np.array_equal(seen[-1], frame), seen[-1]
            assert node.visible
        finally:
            session.shutdown()


class TestWhereTheWorkRuns:
    def test_a_move_evaluates_on_the_dispatcher_and_draws_on_the_painter(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            evaluated = []

            def offsets(frame):
                evaluated.append(threading.current_thread().name)
                return {"found": np.array([[4.0, 5.0]], dtype=np.float32)}

            node = tree.add_overlay(tree.root, offsets, name="markers",
                                    groups={"found": ("circles", {"radius": 3.0})})
            group = plot._overlay_groups[(id(node), "found")]
            drawn = _ThreadRecorder(group, "set")

            _move_navigator(session, tree)

            assert _wait(lambda: evaluated and drawn.threads, 10), \
                (evaluated, drawn.threads)
            assert set(evaluated) == {"nav-dispatch"}, evaluated
            assert set(drawn.threads) == {"nav-paint"}, drawn.threads
            assert np.array_equal(np.asarray(group._data["offsets"]),
                                  np.array([[4.0, 5.0]], dtype=np.float32))
        finally:
            session.shutdown()

    def test_a_value_draws_on_the_painter_with_no_base_frame(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            node = tree.add_overlay(tree.root, lambda frame: {}, name="markers",
                                    groups={"found": ("circles", {"radius": 3.0})})
            group = plot._overlay_groups[(id(node), "found")]
            drawn = _ThreadRecorder(group, "set")

            plot.current_data = None
            plot.enqueue_overlay(
                node, {"found": np.array([[1.0, 2.0]], dtype=np.float32)})
            assert _wait(lambda: drawn.threads, 10), "the value never drew"
            assert set(drawn.threads) == {"nav-paint"}, drawn.threads
        finally:
            session.shutdown()

    def _blocking_overlay(self, tree, painted, gate):
        """An expensive overlay whose FIRST evaluation blocks on ``gate``."""
        calls = itertools.count()

        def blocking(frame, *, wait_for):
            call = next(calls)
            if call == 0:
                wait_for.wait(20)
            return {"found": np.array([[float(call), 0.0]], dtype=np.float32)}

        return tree.add_overlay(
            tree.root, blocking, name="slow", expensive=True,
            groups={"found": ("circles", {"radius": 3.0})},
            static={"wait_for": gate}, on_value=painted.append)

    def test_an_expensive_child_leaves_the_dispatcher_at_once(self):
        session, plot = _open_session(_off_centre_lazy())
        gate = threading.Event()
        try:
            tree = plot.signal_tree
            painted = []
            node = self._blocking_overlay(tree, painted, gate)
            indices = _navigator_selectors(tree)[0].current_indices

            # Resolve the reader first: what is being timed is a navigator MOVE,
            # not the one-off cost of building the reader chain.
            reader_for_overlay(plot, node)
            refresh_overlays(plot, indices)
            assert _wait(lambda: plot._overlay_futures.get(id(node)) is not None, 5)

            started = time.perf_counter()
            refresh_overlays(plot, indices)
            elapsed = time.perf_counter() - started
            assert elapsed < 0.002, elapsed
            assert not painted, painted

            gate.set()
            assert _wait(lambda: painted, 10), painted
        finally:
            gate.set()
            session.shutdown()

    def test_a_superseded_expensive_future_does_not_paint(self):
        session, plot = _open_session(_off_centre_lazy())
        gate = threading.Event()
        try:
            tree = plot.signal_tree
            painted = []
            node = self._blocking_overlay(tree, painted, gate)
            indices = _navigator_selectors(tree)[0].current_indices

            reader_for_overlay(plot, node)
            refresh_overlays(plot, indices)
            assert _wait(lambda: plot._overlay_futures.get(id(node)) is not None, 5)
            refresh_overlays(plot, indices)

            gate.set()
            assert _wait(lambda: painted, 10), painted
            time.sleep(0.3)
            superseded = np.array([[0.0, 0.0]], dtype=np.float32)
            assert not any(np.array_equal(value["found"], superseded)
                           for value in painted), painted
        finally:
            gate.set()
            session.shutdown()

    def test_a_hidden_child_does_not_draw_when_its_future_lands(self):
        session, plot = _open_session(_off_centre_lazy())
        gate = threading.Event()
        try:
            tree = plot.signal_tree
            painted = []
            node = self._blocking_overlay(tree, painted, gate)
            group = plot._overlay_groups[(id(node), "found")]
            indices = _navigator_selectors(tree)[0].current_indices

            reader_for_overlay(plot, node)
            refresh_overlays(plot, indices)
            assert _wait(lambda: plot._overlay_futures.get(id(node)) is not None, 5)

            tree.set_overlay_visible(node, False)
            gate.set()
            time.sleep(0.5)

            assert painted == [], painted
            assert len(np.asarray(group._data["offsets"])) == 0
        finally:
            gate.set()
            session.shutdown()


class TestReaderOverride:
    def test_an_override_serves_the_displayed_frame(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            frame = np.full((32, 32), 7.0, dtype=np.float32)
            tree.set_reader_override(tree.root, _ConstantReader(frame))
            _move_navigator(session, tree)
            assert _wait(lambda: np.array_equal(plot.current_data, frame), 10), \
                plot.current_data

            tree.set_reader_override(tree.root, None)
            _move_navigator(session, tree)
            assert _wait(lambda: not np.array_equal(plot.current_data, frame), 10)
        finally:
            session.shutdown()

    def test_no_frame_from_the_override_keeps_the_last_one(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            _move_navigator(session, tree)
            before = np.asarray(plot.current_data).copy()

            tree.set_reader_override(tree.root, _ConstantReader(None))
            _move_navigator(session, tree)
            time.sleep(0.3)
            assert np.array_equal(plot.current_data, before)
        finally:
            session.shutdown()

    def test_an_override_answers_for_data_that_cannot_be_sliced(self):
        """The progressive case: the node's array is a placeholder, which
        every ordinary read declines, and the override still has a frame."""
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            frame = np.full((32, 32), 7.0, dtype=np.float32)
            tree.set_reader_override(tree.root, _ConstantReader(frame))
            tree.root.data = None          # hyperspy's one-element placeholder

            _move_navigator(session, tree)
            assert _wait(lambda: np.array_equal(plot.current_data, frame), 10), \
                plot.current_data
        finally:
            session.shutdown()

    def test_no_frame_from_the_override_computes_nothing(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            _move_navigator(session, tree)
            before = np.asarray(plot.current_data).copy()
            tree.set_reader_override(tree.root, _ConstantReader(None))

            computed = []
            original = da.Array.compute

            def counting_compute(self, *args, **kwargs):
                if threading.current_thread().name == "nav-dispatch":
                    computed.append(self.shape)
                return original(self, *args, **kwargs)

            da.Array.compute = counting_compute
            try:
                _move_navigator(session, tree)
                time.sleep(0.3)
            finally:
                da.Array.compute = original

            assert computed == [], computed
            assert np.array_equal(plot.current_data, before)
        finally:
            session.shutdown()

    def test_a_region_integrates_through_the_override(self):
        """Without a rule of its own an override's region is the mean of its
        frames; with one, the region is whatever that rule returns, used as
        is."""
        from spyde.drawing.update_functions import _read_through_override

        points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        mean = _read_through_override(_RampReader((4, 4)), points)
        assert np.array_equal(mean, np.full((4, 4), 3.0))   # mean of 0, 2, 4, 6

        summed = _read_through_override(_RegionReader((4, 4)), points)
        assert np.array_equal(summed, np.full((4, 4), 12.0))

    def test_an_integer_region_rounds_back_to_its_dtype(self):
        """The dtype parity rule of the base read: an integer source integrates
        to a ROUNDED integer frame, not a truncated one."""
        from spyde.drawing.update_functions import _read_through_override

        # Values 1 and 2 over two points: the mean is 1.5, which truncates to
        # 1 and rounds to 2 (numpy rounds a half to even).
        counts = _RampReader((4, 4), dtype=np.uint16)
        got = _read_through_override(counts, np.array([[0, 1], [1, 1]]))
        assert got.dtype == np.uint16
        assert np.array_equal(got, np.full((4, 4), 2, dtype=np.uint16)), got[0, 0]


class TestNodeSwitch:
    def test_switching_to_a_sibling_clears_the_overlay(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            centred = _centre(session, plot)
            sibling = tree.add_transformation(
                tree.root, function=lambda signal: signal.deepcopy(),
                node_name="Copy", local=True)

            node = tree.add_overlay(
                centred,
                lambda frame: {"found": np.array([[4.0, 5.0]], dtype=np.float32)},
                name="markers", groups={"found": ("circles", {"radius": 3.0})})
            group = plot._overlay_groups[(id(node), "found")]

            _move_navigator(session, tree)
            assert _wait(lambda: len(np.asarray(group._data["offsets"])) == 1, 10)

            plot.set_plot_state(sibling)
            assert _wait(lambda: len(np.asarray(group._data["offsets"])) == 0, 10), \
                group._data["offsets"]
        finally:
            session.shutdown()

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

import numpy as np
import hyperspy.api as hs

from spyde.array_cache import get_local_frame, reader_for_overlay
from spyde.array_cache.readers.recipe import chain_reaches, evaluate
from spyde.drawing.overlay_node import OverlaySignal
from spyde.drawing.overlays import refresh_overlays
from spyde.external.hyperspy.map_recipe import FrameRecipe
from spyde.tests.migrated.conftest import _settle
from spyde.tests.migrated.test_center_zero_beam import _signal_plot, _wait

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

    def at(self, *index):
        return self._rows[tuple(index)]


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

    def test_a_corner_position_gets_a_clipped_window(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            shape, centre = self._window_at(session, plot, (0, 0))
            assert shape == (2, 2, 32, 32), shape
            assert centre == (0, 0), centre
        finally:
            session.shutdown()

    def test_the_far_corner_is_clipped_on_the_other_side(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            shape, centre = self._window_at(session, plot, (7, 7))
            assert shape == (2, 2, 32, 32), shape
            assert centre == (1, 1), centre
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


class TestWhereTheWorkRuns:
    def test_a_move_evaluates_on_the_dispatcher_and_draws_on_the_painter(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            evaluated, drawn = [], []

            def offsets(frame):
                evaluated.append(threading.current_thread().name)
                return {"found": np.array([[4.0, 5.0]], dtype=np.float32)}

            node = tree.add_overlay(tree.root, offsets, name="markers",
                                    groups={"found": ("circles", {"radius": 3.0})})
            group = plot._overlay_groups[(id(node), "found")]
            push = group.set

            def recording_set(**kwargs):
                drawn.append(threading.current_thread().name)
                push(**kwargs)

            group.set = recording_set

            for selector in _navigator_selectors(tree):
                selector.delayed_update_data(force=True)
            _settle(session)

            assert _wait(lambda: evaluated and drawn, 10), (evaluated, drawn)
            assert set(evaluated) == {"nav-dispatch"}, evaluated
            assert set(drawn) == {"nav-paint"}, drawn
            assert np.array_equal(np.asarray(group._data["offsets"]),
                                  np.array([[4.0, 5.0]], dtype=np.float32))
        finally:
            session.shutdown()

    def test_an_expensive_child_leaves_the_dispatcher_at_once(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            painted = []
            release = threading.Event()
            calls = itertools.count()

            def blocking(frame, *, gate):
                call = next(calls)
                if call == 0:
                    gate.wait(20)
                return {"found": np.array([[float(call), 0.0]], dtype=np.float32)}

            node = tree.add_overlay(
                tree.root, blocking, name="slow", expensive=True,
                groups={"found": ("circles", {"radius": 3.0})},
                static={"gate": release}, on_value=painted.append)
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

            assert _wait(lambda: painted, 10), painted
            assert np.array_equal(painted[-1]["found"],
                                  np.array([[1.0, 0.0]], dtype=np.float32)), painted
        finally:
            release.set()
            session.shutdown()

    def test_a_superseded_expensive_future_does_not_paint(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            painted = []
            release = threading.Event()
            calls = itertools.count()

            def blocking(frame, *, gate):
                call = next(calls)
                if call == 0:
                    gate.wait(20)
                return {"found": np.array([[float(call), 0.0]], dtype=np.float32)}

            node = tree.add_overlay(
                tree.root, blocking, name="slow", expensive=True,
                groups={"found": ("circles", {"radius": 3.0})},
                static={"gate": release}, on_value=painted.append)
            indices = _navigator_selectors(tree)[0].current_indices

            reader_for_overlay(plot, node)
            refresh_overlays(plot, indices)
            assert _wait(lambda: plot._overlay_futures.get(id(node)) is not None, 5)
            refresh_overlays(plot, indices)
            assert _wait(lambda: painted, 10), painted

            release.set()
            time.sleep(0.3)
            first_call = np.array([[0.0, 0.0]], dtype=np.float32)
            assert not any(np.array_equal(value["found"], first_call)
                           for value in painted), painted
        finally:
            release.set()
            session.shutdown()


class TestReaderOverride:
    def test_an_override_serves_the_frame(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            signal = tree.root
            frame = np.full((32, 32), 7.0, dtype=np.float32)
            tree.set_reader_override(signal, _ConstantReader(frame))
            assert np.array_equal(
                get_local_frame(plot, signal, signal.data, (2, 3)), frame)

            tree.set_reader_override(signal, None)
            assert not np.array_equal(
                get_local_frame(plot, signal, signal.data, (2, 3)), frame)
        finally:
            session.shutdown()

    def test_a_frame_the_override_has_no_value_for_is_not_cached(self):
        session, plot = _open_session(_off_centre_lazy())
        try:
            tree = plot.signal_tree
            signal = tree.root
            tree.set_reader_override(signal, _ConstantReader(None))

            assert get_local_frame(plot, signal, signal.data, (2, 3)) is None
            assert not plot._array_cache.is_resident(id(signal), (2, 3))
        finally:
            session.shutdown()


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

            for selector in _navigator_selectors(tree):
                selector.delayed_update_data(force=True)
            _settle(session)
            assert _wait(lambda: len(np.asarray(group._data["offsets"])) == 1, 10)

            plot.set_plot_state(sibling)
            assert _wait(lambda: len(np.asarray(group._data["offsets"])) == 0, 10), \
                group._data["offsets"]
        finally:
            session.shutdown()

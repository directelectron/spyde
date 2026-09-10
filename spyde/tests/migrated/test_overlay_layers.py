"""
test_overlay_layers.py: MDI live image layering.

Exercises spyde.actions.overlay against a real Qt-free ``Session`` built on a 4-D
STEM tree (navigator + signal windows share one tree). A second same-shape signal
window is added so a layer's SOURCE differs from its TARGET.

Covered: add / same-shape validation / mismatch refusal / self refusal,
``layers_state`` emissions, overlay_set reflected in the anyplotlib layer state,
nav-move → the layer refreshes with the SOURCE's frame at the new indices (driven
via the selector's ``_run_update``, like test_navigator_race), a move whose base
read declines refreshing the layer without error, teardown on window close, tile-
mode refusal, the painter's atomic take-and-clear of staged overlay values, and
an expensive overlay painting from its future's callback.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from spyde.actions import overlay as ov
from spyde.tests.migrated.conftest import _settle
from spyde.tests.migrated._async import quiesce, why_busy


# ── setup helpers ───────────────────────────────────────────────────────────────


def _prime(session):
    for p in session._plots:
        if isinstance(getattr(p, "current_data", None), np.ndarray):
            continue
        try:
            sig = p.plot_state.current_signal
            frame = np.asarray(sig.data)
            if frame.ndim > 2:
                frame = frame.reshape(-1, *frame.shape[-2:])[0]
            p.current_data = np.ascontiguousarray(frame.astype(np.float32))
            p._last_levels = (float(np.nanmin(p.current_data)),
                              float(np.nanmax(p.current_data)))
        except Exception:
            pass


def _two_signal_windows(session):
    """Add a SECOND same-shape signal window to the 4-D tree and return
    (target_plot, source_plot, navigator_plot, multiplot_manager, nav_plot_window).
    Both signal plots read the same signal at the same nav indices (same tree)."""
    nav = [p for p in session._plots if p.is_navigator][0]
    mm = nav.multiplot_manager
    navpw = nav.plot_window
    mm.add_navigation_selector_and_signal_plot(navpw)
    _settle(session)
    _prime(session)
    sigs = sorted((p for p in session._plots if not p.is_navigator),
                  key=lambda p: p.window_id)
    return sigs[0], sigs[1], nav, mm, navpw


def _layers_states(messages):
    return [m for m in messages if m.get("type") == "layers_state"]


def _layer_handle(plot, node):
    """The anyplotlib Layer the node's group draws with, once the painter has
    built it from the first value."""
    return plot._overlay_groups.get((id(node), ov.LAYER_GROUP))


def _wait_for_layer(plot, node, timeout=5.0):
    """Block until the painter has built the node's anyplotlib layer."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        handle = _layer_handle(plot, node)
        if handle is not None:
            return handle
        time.sleep(0.02)
    return None


def _add_layer(session, tgt, src, timeout=5.0):
    """Add a layer and wait for the painter to build it."""
    ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                  "source_window_id": src.window_id})
    nodes = ov.layer_nodes(tgt)
    if nodes:
        _wait_for_layer(tgt, nodes[0], timeout)
    return nodes[0] if nodes else None


def _region_points(height, width):
    """The widget (cx, cy) pairs an integrating region of this size reports."""
    return np.array([[x, y] for y in range(height) for x in range(width)])


def _drive_region_to(mm, navpw, target_plot, points):
    """Run the selector that drives ``target_plot`` over an integrating region
    covering ``points`` (widget (cx, cy) pairs)."""
    sel = [s for s in mm.navigation_selectors[navpw] if target_plot in s.children][0]
    inner = getattr(sel, "selector", sel)
    inner.get_selected_indices = lambda: np.asarray(points)
    inner.is_integrating = True
    inner._run_update(force=True)
    time.sleep(0.6)
    return sel


def _drive_selector_to(mm, navpw, target_plot, y, x):
    """Run the selector that drives ``target_plot`` at nav position (y, x) via
    ``_run_update`` (the same code path a real drag takes), forcing the update.

    ``IntegratingSSelector2D`` is a COMPOSITE that delegates ``_run_update`` to its
    ACTIVE inner sub-selector (the crosshair), so we override + drive the inner
    selector — its ``_run_update``'s ``self`` is the inner one."""
    sel = [s for s in mm.navigation_selectors[navpw] if target_plot in s.children][0]
    inner = getattr(sel, "selector", sel)   # active sub-selector of the composite
    inner.get_selected_indices = lambda: np.array([[int(x), int(y)]])  # widget (cx, cy)
    inner._run_update(force=True)
    time.sleep(0.6)   # the layer is evaluated on the overlay lane, then painted
    return sel


# ── add / validation ────────────────────────────────────────────────────────────


class TestOverlayAdd:
    def test_add_layer_same_shape(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        messages.clear()
        node = _add_layer(session, tgt, src)
        assert node is not None
        assert len(ov.layer_nodes(tgt)) == 1
        # A real anyplotlib layer exists on the target's plot2d.
        assert len(tgt._plot2d._state.get("layers", [])) == 1
        # layers_state emitted for the target.
        st = _layers_states(messages)
        assert st and st[-1]["window_id"] == tgt.window_id
        assert len(st[-1]["layers"]) == 1
        layer = st[-1]["layers"][0]
        assert set(layer) >= {"id", "title", "cmap", "alpha", "clim", "visible"}
        assert layer["visible"] is True

    def test_layer_draws_only_on_its_target(self, stem_4d_dataset):
        """A layer belongs to the window it was dropped on, not to every window
        of the tree showing the same node."""
        session = stem_4d_dataset["window"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        node = _add_layer(session, tgt, src)
        assert ov.layer_nodes(tgt) == [node]
        assert ov.layer_nodes(src) == []
        assert src._plot2d._state.get("layers", []) == []

    def test_refuse_shape_mismatch(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        nav = [p for p in session._plots if p.is_navigator][0]
        sig = [p for p in session._plots if not p.is_navigator][0]
        # navigator (4x5) vs signal (16x16) → refused with a status, no layer.
        messages.clear()
        ov.overlay_add(session, sig, {"window_id": sig.window_id,
                                      "source_window_id": nav.window_id})
        assert ov.layer_nodes(sig) == []
        assert any(m.get("type") == "status" for m in messages)

    def test_refuse_self(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        sig = [p for p in session._plots if not p.is_navigator][0]
        messages.clear()
        ov.overlay_add(session, sig, {"window_id": sig.window_id,
                                      "source_window_id": sig.window_id})
        assert ov.layer_nodes(sig) == []
        assert any(m.get("type") == "status" for m in messages)

    def test_query_reemits_state(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        _add_layer(session, tgt, src)
        messages.clear()
        ov.overlay_query(session, tgt, {"window_id": tgt.window_id})
        st = _layers_states(messages)
        assert st and len(st[-1]["layers"]) == 1


# ── overlay_set ─────────────────────────────────────────────────────────────────


class TestShapeChangeDropsLayers:
    def test_shape_changing_paint_drops_layers_cleanly(self, stem_4d_dataset):
        """anyplotlib raises on a shape-changing set_data while layers exist; the
        plot must drop its layers FIRST (status + empty layers_state), not raise."""
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        _add_layer(session, tgt, src)
        assert len(ov.layer_nodes(tgt)) == 1
        messages.clear()
        old_shape = tgt._plot2d._state["image_height"], tgt._plot2d._state["image_width"]
        new_frame = np.random.rand(old_shape[0] * 2, old_shape[1] * 2).astype(np.float32)
        tgt._set_array(new_frame)          # must not raise
        assert ov.layer_nodes(tgt) == []
        st = _layers_states(messages)
        assert st and st[-1]["window_id"] == tgt.window_id
        assert st[-1]["layers"] == []
        assert any(m.get("type") == "status" and "shape changed" in m.get("text", "")
                   for m in messages)


class TestOverlaySet:
    def test_set_reflected_in_layer_state(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        node = _add_layer(session, tgt, src)
        messages.clear()
        ov.overlay_set(session, tgt, {"window_id": tgt.window_id,
                                      "layer_id": node.name,
                                      "cmap": "plasma", "alpha": 0.2,
                                      "visible": False})
        # The emitted layers_state is the authoritative record.
        st = _layers_states(messages)[-1]["layers"][0]
        assert st["cmap"] == "plasma"
        assert abs(st["alpha"] - 0.2) < 1e-9
        assert st["visible"] is False
        # And the appearance rides the layer's next value, so the anyplotlib
        # entry follows once the painter has drawn it.
        deadline = time.monotonic() + 5.0
        entry = {}
        while time.monotonic() < deadline:
            entries = tgt._plot2d._state.get("layers", [])
            entry = entries[0] if entries else {}
            if entry.get("visible") is False:
                break
            time.sleep(0.02)
        assert entry.get("visible") is False


# ── live nav refresh ────────────────────────────────────────────────────────────


class TestLiveNavRefresh:
    def test_nav_move_refreshes_layer(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        node = _add_layer(session, tgt, src)
        handle = _layer_handle(tgt, node)
        assert handle is not None

        # Record every frame the layer handle receives.
        pushed = []
        orig = handle.set_data

        def _rec(frame):
            pushed.append(float(np.asarray(frame).mean()))
            return orig(frame)

        handle.set_data = _rec

        # Drive to two DISTINCT nav positions; the layer must refresh from the
        # SOURCE's frame at each (the fixture data varies per nav index).
        _drive_selector_to(mm, navpw, tgt, 0, 0)
        _drive_selector_to(mm, navpw, tgt, 3, 4)

        assert len(pushed) >= 2, f"layer not refreshed on nav move (got {pushed})"
        # The two positions produce different source frames.
        assert pushed[0] != pushed[-1], (
            f"layer frame did not change across nav positions: {pushed}")

    def test_declined_source_read_leaves_the_layer_alone(self, stem_4d_dataset,
                                                         monkeypatch):
        """A source frame the reader cannot produce must not raise and must not
        clear the layer: the last frame stays up until a readable one lands."""
        session = stem_4d_dataset["window"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        node = _add_layer(session, tgt, src)
        handle = _layer_handle(tgt, node)
        pushed = []
        handle.set_data = lambda f: pushed.append(1)

        def _no_frame(*a, **k):
            raise ValueError("no source frame here")

        import spyde.array_cache.nav_read as nav_read
        monkeypatch.setattr(nav_read._CachedParentFrames, "read_frame", _no_frame)

        from spyde.drawing.overlays import refresh_overlays
        refresh_overlays(tgt, np.array([[1, 1]]))
        assert quiesce(session), why_busy(session)
        time.sleep(0.4)
        assert pushed == [], "a declined source read still pushed a layer frame"


# ── teardown + tile-mode refusal ────────────────────────────────────────────────


class TestOverlayTeardown:
    def test_remove_layer(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        node = _add_layer(session, tgt, src)
        messages.clear()
        ov.overlay_remove(session, tgt, {"window_id": tgt.window_id,
                                         "layer_id": node.name})
        assert ov.layer_nodes(tgt) == []
        assert tgt._plot2d._state.get("layers", []) == []
        assert _layers_states(messages)[-1]["layers"] == []

    def test_target_close_drops_layers(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        _add_layer(session, tgt, src)
        assert len(ov.layer_nodes(tgt)) == 1
        # Closing the TARGET plot drops its layers cleanly.
        tgt.close()
        assert ov.layer_nodes(tgt) == []

    def test_source_close_drops_layers_on_target(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        _add_layer(session, tgt, src)
        assert len(ov.layer_nodes(tgt)) == 1
        messages.clear()
        # Closing the SOURCE plot must drop the target's layer that sourced from it.
        src.close()
        assert ov.layer_nodes(tgt) == []
        # A layers_state was re-emitted for the affected target.
        assert any(m.get("type") == "layers_state"
                   and m.get("window_id") == tgt.window_id for m in messages)

    def test_tile_mode_refused(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        # Force the target into tile mode.
        tgt._plot2d._tile_on = True
        messages.clear()
        ov.overlay_add(session, tgt, {"window_id": tgt.window_id,
                                      "source_window_id": src.window_id})
        assert ov.layer_nodes(tgt) == []
        assert any(m.get("type") == "status" and "tile" in str(m.get("text", "")).lower()
                   for m in messages)


# ── the painter's take-and-clear ─────────────────────────────────────────────────


class _StubNode:
    """An overlay node stand-in for the staging slot: it only needs to look
    attached, visible, and to declare no groups (nothing is pushed)."""

    def __init__(self):
        self.name = "layer"
        self.visible = True
        self.groups = {}
        self.on_value = None
        self.parent = self
        self.children = {"layer": self}


class TestPendingOverlayRace:
    def test_write_between_read_and_clear_is_not_lost(self, stem_4d_dataset,
                                                      monkeypatch):
        """The painter's take-and-clear of a plot's staged overlay values must be
        ATOMIC against a dispatcher write. We wedge the painter INSIDE the
        critical section and fire a newer write; with the lock the write blocks
        until the painter finishes its swap, so the newer value is applied on the
        next drain and never silently dropped."""
        import threading
        from spyde.drawing.plots import plot as plot_module

        session = stem_4d_dataset["window"]
        _prime(session)
        target = [p for p in session._plots if not p.is_navigator][0]
        node = _StubNode()

        drawn = []
        monkeypatch.setattr(
            type(target), "_push_overlay_group",
            lambda self, n, name, kind, value, base_painted=False:
                drawn.append(value))

        in_critical_section = threading.Event()
        release = threading.Event()
        acquired = {"count": 0}
        real_lock = plot_module._nav_painter._lock

        class _InstrumentedLock:
            """Wrap the real lock: on the painter's ACQUIRE (the first, i.e. the
            take-and-clear), signal that we are in the section and block until
            the writer has attempted its lock-contended write."""

            def __enter__(self):
                real_lock.acquire()
                acquired["count"] += 1
                if acquired["count"] == 1:
                    in_critical_section.set()
                    release.wait(2.0)
                return self

            def __exit__(self, *a):
                real_lock.release()
                return False

        monkeypatch.setattr(plot_module._nav_painter, "_lock", _InstrumentedLock())

        old_value = {"mark": "old"}
        new_value = {"mark": "new"}
        target._pending_overlay_values = {id(node): (node, old_value)}
        node.on_value = lambda value: drawn.append(value)

        painter = threading.Thread(target=lambda: target.paint_pass(None))
        painter.start()
        assert in_critical_section.wait(2.0), \
            "the painter never entered the critical section"

        wrote = threading.Event()

        def _writer():
            target.enqueue_overlay(node, new_value)
            wrote.set()

        writer = threading.Thread(target=_writer)
        writer.start()
        assert not wrote.wait(0.3), \
            "the writer wrote while the painter held the lock (race)"

        release.set()
        painter.join(2.0)
        assert wrote.wait(2.0)
        writer.join(2.0)

        # The painter delivered the OLD value it read; the NEW one is still
        # staged, so a second drain delivers it. Nothing is lost.
        monkeypatch.setattr(plot_module._nav_painter, "_lock", real_lock)
        target.paint_pass(None)
        assert old_value in drawn, "the old value was not delivered"
        assert new_value in drawn, "the NEWER value was lost (lost-update race)"


# ── an expensive overlay paints from its future's callback ───────────────────────


class TestExpensiveOverlayCallback:
    def test_value_arrives_from_the_future(self, stem_4d_dataset):
        """An expensive overlay returns from the dispatcher at once and its value
        reaches the plot from the compute backend's done callback, the tier a
        cold cross-window source read runs on."""
        session = stem_4d_dataset["window"]
        _prime(session)
        tree = session.signal_trees[0]
        plot = tree.signal_plots[0]
        seen = []

        def slow_value(frame):
            time.sleep(0.1)
            return {"mark": float(np.asarray(frame).mean())}

        node = tree.add_overlay(tree.root, slow_value, name="slow", groups={},
                                expensive=True, on_value=seen.append)
        try:
            from spyde.drawing.overlays import refresh_overlays
            started = time.monotonic()
            refresh_overlays(plot, np.array([[1, 1]]))
            assert time.monotonic() - started < 0.05, \
                "an expensive overlay blocked the dispatcher"
            deadline = time.monotonic() + 5.0
            while not seen and time.monotonic() < deadline:
                time.sleep(0.02)
            assert seen, "the expensive overlay's value never arrived"
        finally:
            tree.remove_overlay(node)


# ── the tier a layer takes, and the region it integrates ────────────────────────


class TestLayerTakesTheSourceTier:
    """A layer names no tier of its own: it is the source window's frame, so
    it runs wherever reading that frame would run."""

    def _threads_of(self, monkeypatch):
        """Record the thread ``layer_frame`` is called on."""
        threads = []
        original = ov.layer_frame

        def recording(frame, *, appearance):
            threads.append(threading.current_thread().name)
            return original(frame, appearance=appearance)

        monkeypatch.setattr(ov, "layer_frame", recording)
        return threads

    def test_a_resident_frame_is_read_inline(self, stem_4d_dataset, monkeypatch):
        """Read on the thread running the navigator update, so the value is
        staged before the base frame is painted and both reach the figure in
        one pass."""
        session = stem_4d_dataset["window"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        threads = self._threads_of(monkeypatch)
        node = _add_layer(session, tgt, src)
        threads.clear()                       # the drop's own seeding call

        _drive_selector_to(mm, navpw, tgt, 2, 3)
        assert threads, "the layer never evaluated"
        assert set(threads) == {threading.current_thread().name}, threads
        assert node.expensive is None

    def test_a_costly_read_arrives_from_the_overlay_lane(self, stem_4d_dataset,
                                                         monkeypatch):
        """A region wide enough to block the navigator goes to the compute
        backend's lane, exactly as the base frame's own read would."""
        session = stem_4d_dataset["window"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        threads = self._threads_of(monkeypatch)
        _add_layer(session, tgt, src)
        threads.clear()                       # the drop's own seeding call

        _drive_region_to(mm, navpw, tgt, _region_points(8, 8))
        assert quiesce(session), why_busy(session)
        time.sleep(0.4)
        assert threads, "the layer never evaluated"
        assert all(name.startswith("overlay-eval") for name in threads), threads


class TestEagerRegionParity:
    def test_a_region_layer_integrates_what_the_base_integrates(self,
                                                                stem_4d_dataset):
        """A layer composites over the base image, so it must integrate the
        same navigation positions under the same rule: the region reaches the
        source read whole rather than being reduced to its centre."""
        from spyde.array_cache import get_local_frame
        from spyde.drawing.update_functions import _prepare_nav_indices

        session = stem_4d_dataset["window"]
        _prime(session)
        tgt, src, nav, mm, navpw = _two_signal_windows(session)
        node = _add_layer(session, tgt, src)

        points = _region_points(3, 2)
        _drive_region_to(mm, navpw, tgt, points)
        assert quiesce(session), why_busy(session)
        time.sleep(0.4)

        value = tgt.last_overlay_value(node)
        assert value, "the layer drew nothing for the region"
        drawn = np.asarray(value[ov.LAYER_GROUP]["data"])

        signal = src.plot_state.current_signal
        prepared = _prepare_nav_indices(signal, points, integrating=True)
        expected = get_local_frame(src, signal, signal.data, prepared)
        assert np.array_equal(drawn, expected), \
            "the layer's region differs from the base frame's"
        centre = get_local_frame(src, signal, signal.data,
                                 _prepare_nav_indices(signal, points,
                                                      integrating=False))
        assert not np.array_equal(drawn, centre), \
            "the fixture cannot tell a region from its centre position"

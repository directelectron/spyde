"""
The offset "set origin" crosshair: toggling it on drops a draggable crosshair on
the signal plot; as it moves, both signal-axis offsets update so the crosshair
position reads (0, 0).  Toggling off removes it.

The crosshair widget itself is anyplotlib's; here we stub a minimal widget on the
plot's _plot2d so the test exercises the offset MATH and lifecycle without a GUI.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, close_session, make_session


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


class _FakeWidget:
    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.handlers = []
        self.hidden = False

    def add_event_handler(self, cb, *events):
        self.handlers.append(cb)

    def hide(self):
        self.hidden = True

    def fire(self, etype="pointer_move"):
        ev = type("E", (), {"type": etype})()
        for cb in self.handlers:
            cb(ev)


class _FakePlot2D:
    def __init__(self):
        self.widget = None
        self.removed = []

    def add_crosshair_widget(self, cx, cy, color=None):
        self.widget = _FakeWidget(cx, cy)
        return self.widget

    def remove_widget(self, w):
        self.removed.append(w)
        w.hide()


class TestOffsetCrosshair:
    def _session_with_plot(self):
        session = make_session()
        s = hs.signals.Signal2D(np.zeros((4, 5, 20, 20), np.float32))
        s.set_signal_type("electron_diffraction")
        # known calibration: scale 0.1, offset 0 on both signal axes
        for ax in s.axes_manager.signal_axes:
            ax.scale, ax.offset = 0.1, 0.0
        session._add_signal(s)
        _settle(session)
        plot = _signal_plot(session)
        return session, plot

    def test_toggle_on_sets_offset_so_crosshair_is_origin(self):
        session, plot = self._session_with_plot()
        try:
            plot._plot2d = _FakePlot2D()
            tree = plot.signal_tree
            session._set_offset_crosshair(plot, {"on": True})
            w = plot._plot2d.widget
            assert w is not None
            # anyplotlib 2-D widgets report PIXELS. Move the crosshair to pixel
            # (10, 5) under scale 0.1 → offsets -(10*0.1)=-1.0 and -(5*0.1)=-0.5.
            w.cx, w.cy = 10.0, 5.0
            w.fire("pointer_move")
            ax = tree.root.axes_manager.signal_axes
            assert abs(float(ax[0].offset) - (-1.0)) < 1e-6
            assert abs(float(ax[1].offset) - (-0.5)) < 1e-6
        finally:
            session.shutdown()

    def test_stable_across_repeated_moves(self):
        # offset must converge, not drift, when the same data position is
        # re-applied (no feedback loop from re-reading the offset).
        session, plot = self._session_with_plot()
        try:
            plot._plot2d = _FakePlot2D()
            tree = plot.signal_tree
            session._set_offset_crosshair(plot, {"on": True})
            w = plot._plot2d.widget
            w.cx, w.cy = 2.0, 2.0
            ax = tree.root.axes_manager.signal_axes
            offs = []
            for _ in range(5):
                w.fire("pointer_move")
                offs.append((float(ax[0].offset), float(ax[1].offset)))
            # all five identical (stable)
            assert all(abs(o[0] - offs[0][0]) < 1e-9 for o in offs)
            assert all(abs(o[1] - offs[0][1]) < 1e-9 for o in offs)
        finally:
            session.shutdown()

    def test_drag_release_drag_again(self):
        # after a release (final apply re-pushes the extent and re-anchors the
        # reference), a second drag from the new origin must still land correctly.
        session, plot = self._session_with_plot()
        try:
            plot._plot2d = _FakePlot2D()
            tree = plot.signal_tree
            session._set_offset_crosshair(plot, {"on": True})
            w = plot._plot2d.widget
            ax = tree.root.axes_manager.signal_axes
            # first drag to pixel (10,10) → offset -(10*0.1)=-1.0 under scale 0.1
            w.cx, w.cy = 10.0, 10.0
            w.fire("pointer_move")
            w.fire("pointer_up")
            assert abs(float(ax[0].offset) - (-1.0)) < 1e-6
            # The widget reports ABSOLUTE pixels, so the host re-push leaves it at
            # the same pixel. A second drag to pixel 13 in x → offset -(13*0.1).
            w.cx, w.cy = 13.0, 10.0
            w.fire("pointer_move")
            assert abs(float(ax[0].offset) - (-1.3)) < 1e-6
        finally:
            session.shutdown()

    def test_toggle_off_removes_crosshair(self):
        session, plot = self._session_with_plot()
        try:
            plot._plot2d = _FakePlot2D()
            session._set_offset_crosshair(plot, {"on": True})
            w = plot._plot2d.widget
            session._set_offset_crosshair(plot, {"on": False})
            # removed (not just hidden) on the FIRST toggle-off — remove_widget
            # re-pushes the panel so it disappears in one click.
            assert w in plot._plot2d.removed
            assert w.hidden
            assert getattr(plot, "_offset_cross", None) is None
        finally:
            session.shutdown()

    def test_navigator_plot_edits_navigation_axes(self):
        """On a navigator plot the tool must edit the NAVIGATION axes, leaving
        the signal axes untouched (signal->signal, nav->nav)."""
        import hyperspy.api as hs
        session = make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((6, 6, 12, 12), np.float32))
            s.set_signal_type("electron_diffraction")
            for a in s.axes_manager.navigation_axes:
                a.scale, a.offset = 2.0, 0.0
            session._add_signal(s)
            _settle(session)
            navp = next((p for p in session._plots if p.is_navigator), None)
            assert navp is not None
            navp._plot2d = _FakePlot2D()
            session._set_offset_crosshair(navp, {"on": True})
            w = navp._plot2d.widget
            # move to nav PIXELS (2, 3) under scale 2.0 → nav offsets
            # -(2*2)=-4.0 and -(3*2)=-6.0; SIGNAL axes stay 0.
            w.cx, w.cy = 2.0, 3.0
            w.fire("pointer_move")
            nav = navp.signal_tree.root.axes_manager.navigation_axes
            sig = navp.signal_tree.root.axes_manager.signal_axes
            assert abs(float(nav[0].offset) - (-4.0)) < 1e-6
            assert abs(float(nav[1].offset) - (-6.0)) < 1e-6
            assert abs(float(sig[0].offset)) < 1e-9
            assert abs(float(sig[1].offset)) < 1e-9
        finally:
            close_session(session)

    # ── the "+" toggle is BACKEND-owned: button ON ⟺ crosshair alive ────────
    #
    # The dock used to keep its own boolean, which drifted: focusing another
    # window reset it (the "+" went dark) while the crosshair stayed on the
    # first plot, and the next click then targeted the NEW active plot — so the
    # stale crosshair could never be dismissed from the UI. Every toggle now
    # answers with the state the backend actually reached.

    def test_toggle_emits_pick_state(self, captured_messages):
        session, plot = self._session_with_plot()
        try:
            plot._plot2d = _FakePlot2D()
            captured_messages.clear()
            session._set_offset_crosshair(plot, {"on": True})
            picks = [m for m in captured_messages if m.get("type") == "offset_pick"]
            assert picks and picks[-1]["on"] is True
            assert picks[-1]["window_id"] == plot.window_id

            captured_messages.clear()
            session._set_offset_crosshair(plot, {"on": False})
            picks = [m for m in captured_messages if m.get("type") == "offset_pick"]
            assert picks and picks[-1]["on"] is False
        finally:
            session.shutdown()

    def test_refused_toggle_reports_off(self, captured_messages):
        """A toggle-on the backend can't honour (no figure to hang the widget
        on) must report OFF — otherwise the dock lights a button with no
        crosshair under it, and the next click sends `on: False` (a no-op),
        which is the "have to toggle it twice" symptom."""
        session, plot = self._session_with_plot()
        try:
            plot._plot2d = None
            captured_messages.clear()
            session._set_offset_crosshair(plot, {"on": True})
            picks = [m for m in captured_messages if m.get("type") == "offset_pick"]
            assert picks and picks[-1]["on"] is False
            assert getattr(plot, "_offset_cross", None) is None
        finally:
            session.shutdown()

    def test_node_switch_clears_the_crosshair(self, captured_messages):
        """The crosshair captured THIS node's signal axes, so a node switch
        would leave it recalibrating the node the user just left. It is torn
        down, and the dock is told so the "+" goes with it."""
        session, plot = self._session_with_plot()
        try:
            plot._plot2d = _FakePlot2D()
            session._set_offset_crosshair(plot, {"on": True})
            w = plot._plot2d.widget
            assert w is not None
            captured_messages.clear()
            # Switch to the node the plot already displays — the switch path is
            # what's under test, not the transform.
            current = plot.plot_state.current_signal
            session._select_signal_node(plot, id(current))
            assert getattr(plot, "_offset_cross", None) is None
            picks = [m for m in captured_messages if m.get("type") == "offset_pick"]
            assert picks and picks[-1]["on"] is False
        finally:
            session.shutdown()

    def test_clear_is_a_noop_without_a_crosshair(self, captured_messages):
        session, plot = self._session_with_plot()
        try:
            plot._plot2d = _FakePlot2D()
            captured_messages.clear()
            session._clear_offset_crosshair(plot)
            assert not [m for m in captured_messages
                        if m.get("type") == "offset_pick"]
        finally:
            session.shutdown()

    def test_starts_at_current_offset_no_change_until_drag(self):
        """Toggling on must NOT change the offset — the crosshair starts at the
        current origin; the offset only moves when the user drags."""
        session, plot = self._session_with_plot()
        try:
            plot._plot2d = _FakePlot2D()
            ax = plot.signal_tree.root.axes_manager.signal_axes
            ax[0].offset, ax[1].offset = -0.7, -0.7   # existing origin
            before = (float(ax[0].offset), float(ax[1].offset))
            session._set_offset_crosshair(plot, {"on": True})
            after = (float(ax[0].offset), float(ax[1].offset))
            assert abs(after[0] - before[0]) < 1e-9
            assert abs(after[1] - before[1]) < 1e-9
            # the crosshair starts at the current origin PIXEL = -offset/scale =
            # 0.7/0.1 = 7 (the widget reports pixels, not data coords).
            w = plot._plot2d.widget
            assert abs(w.cx - 7.0) < 1e-6 and abs(w.cy - 7.0) < 1e-6
        finally:
            session.shutdown()

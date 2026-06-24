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
        from spyde.backend.session import Session
        session = Session(n_workers=1, threads_per_worker=1)
        s = hs.signals.Signal2D(np.zeros((4, 5, 20, 20), np.float32))
        s.set_signal_type("electron_diffraction")
        # known calibration: scale 0.1, offset 0 on both signal axes
        for ax in s.axes_manager.signal_axes:
            ax.scale, ax.offset = 0.1, 0.0
        session._add_signal(s)
        time.sleep(0.3)
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
        from spyde.backend.session import Session
        import hyperspy.api as hs
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            s = hs.signals.Signal2D(np.zeros((6, 6, 12, 12), np.float32))
            s.set_signal_type("electron_diffraction")
            for a in s.axes_manager.navigation_axes:
                a.scale, a.offset = 2.0, 0.0
            session._add_signal(s)
            time.sleep(0.3)
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

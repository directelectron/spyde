"""
Center Zero Beam (Electron, two-tab parity).

  Automatic — `czb_run` estimates the beam position (pyxem
              `get_direct_beam_position`) and applies `center_direct_beam`, adding
              a "Centered" tree node; the displayed pattern becomes centred.
  Manual    — `czb_open` drops a draggable crosshair; `czb_pick`
              centres by the picked position (constant shift).
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle
from spyde.tests.migrated._async import wait_until


def _wait(pred, timeout=25.0):
    return wait_until(pred, timeout)


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _off_center_4d(nav=(3, 3), sig=(32, 32), beam=(18, 14)):
    """A disk offset from the detector centre (beam at col=18,row=14)."""
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - beam[0]) ** 2 + (yy - beam[1]) ** 2 <= 9).astype(np.float32)
    data = np.zeros(nav + sig, dtype=np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    return s


def _com(frame):
    frame = np.asarray(frame, dtype=np.float64)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    tot = frame.sum()
    return (xx * frame).sum() / tot, (yy * frame).sum() / tot   # (col, row)


class TestCenterZeroBeam:
    def test_auto_centers_beam(self):
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_run
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(beam=(18, 14)))
            _settle(session)
            src = _signal_plot(session)
            before = src.plot_state.current_signal

            czb_run(session, src, {"method": "center_of_mass"})
            assert _wait(lambda: src.plot_state.current_signal is not before,
                         timeout=20), "centering never produced a new signal"
            centered = src.plot_state.current_signal
            frame = centered.inav[0, 0].data
            if hasattr(frame, "compute"):
                frame = frame.compute()
            cx, cy = _com(frame)
            assert abs(cx - 16) < 2 and abs(cy - 16) < 2, (cx, cy)
            # The tree records the step.
            node = session.signal_trees[0].get_node(before)
            assert any("Centered" in k for k in node.children)
        finally:
            session.shutdown()

    def test_auto_region_shows_draggable_widget_and_run_uses_it(self):
        # The Automatic tab's search-window box must be a DRAGGABLE
        # RectangleWidget (add_rectangle_widget), not a static marker (the old
        # add_squares bug: no drag handles, geometry never fed back into the
        # search). czb_set_region opens it; resizing it changes the region
        # czb_run actually searches.
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_set_region, czb_run, czb_close
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(sig=(64, 64), beam=(18, 14)))
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            czb_set_region(session, src, {"half_square_width": 20})
            widget = getattr(tree, "_czb_region_mg", None)
            assert widget is not None, "no region widget was created"
            # Draggable/resizable: has the widget attrs a plain marker lacks.
            assert hasattr(widget, "add_event_handler")
            assert hasattr(widget, "set") and hasattr(widget, "hide")
            assert float(widget.w) == float(widget.h) == 40.0  # 2*half_square_width
            # Centred on the 64x64 frame.
            assert abs(float(widget.x) - 12.0) < 1e-6
            assert abs(float(widget.y) - 12.0) < 1e-6

            # Resize the widget (simulating a user drag) — shrink it well below
            # the caret field's stale half_square_width=20.
            widget.set(x=28.0, y=28.0, w=8.0, h=8.0, _push=False)
            before = src.plot_state.current_signal
            # Run with the STALE field value (20) — the live widget (half-width
            # 4) must win.
            czb_run(session, src, {"method": "center_of_mass", "half_square_width": 20})
            assert _wait(lambda: src.plot_state.current_signal is not before,
                         timeout=20)

            # Closing the caret tears the widget down.
            czb_close(session, src, {})
            assert getattr(tree, "_czb_region_mg", None) is None
        finally:
            session.shutdown()

    def test_region_widget_torn_down_on_close(self):
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_set_region, czb_close
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(sig=(32, 32)))
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            czb_set_region(session, src, {"half_square_width": 10})
            widget = getattr(tree, "_czb_region_mg", None)
            assert widget is not None
            assert widget.visible is True

            czb_close(session, src, {})
            assert getattr(tree, "_czb_region_mg", None) is None
            # The widget itself was hidden (not just dropped from the tree).
            assert widget.visible is False

            # hw<=0 ("full frame") RESIZES the widget to the full frame rather
            # than removing it — same full-frame-box-on-open contract as Crop.
            czb_set_region(session, src, {"half_square_width": 10})
            widget2 = getattr(tree, "_czb_region_mg", None)
            assert widget2 is not None
            czb_set_region(session, src, {"half_square_width": 0})
            assert getattr(tree, "_czb_region_mg", None) is widget2
            assert (float(widget2.w), float(widget2.h)) == (32.0, 32.0)
            # Only czb_close removes it.
            czb_close(session, src, {})
            assert getattr(tree, "_czb_region_mg", None) is None
        finally:
            session.shutdown()

    def test_activating_automatic_tab_shows_full_frame_box_immediately(self):
        # Opening the Automatic tab (half_square_width defaults to 0, "full
        # frame") must show the box right away, not require the user to first
        # type a nonzero half-width — this was the actual #10 bug (the box
        # either never appeared or wasn't interactive).
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_set_region
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(sig=(48, 32)))
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            czb_set_region(session, src, {"half_square_width": 0})
            widget = getattr(tree, "_czb_region_mg", None)
            assert widget is not None, "the search box must appear on open, at half_square_width=0"
            assert (float(widget.w), float(widget.h)) == (32.0, 32.0)
        finally:
            session.shutdown()

    def test_region_drag_handler_runs_once_per_event(self):
        # REGRESSION (reviewer repro): Widget.set() fires pointer_move
        # UNCONDITIONALLY, so the re-centre handler's own set() used to
        # re-enter itself recursively (~2000 deep on ONE JS drag frame,
        # surviving only via recursionlimit + swallowed RecursionError).
        # Drive the REAL JS→Python path and count the handler's set().
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_set_region
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(sig=(32, 32)))
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            czb_set_region(session, src, {"half_square_width": 10})
            widget = tree._czb_region_mg

            calls = []
            real_set = widget.set
            def counting_set(*a, **kw):
                calls.append(1)
                return real_set(*a, **kw)
            # object.__setattr__: a plain `widget.set = ...` would route
            # through Widget.__setattr__ into set(set=...) instead.
            object.__setattr__(widget, "set", counting_set)

            widget._update_from_js(
                {"x": 2.0, "y": 5.0, "w": 12.0, "h": 9.0}, "pointer_move")

            assert len(calls) == 1, \
                f"re-centre must run exactly ONCE per JS event, ran {len(calls)}x"
            # Snapped back square + centred: side = min(12, 9) = 9 on 32x32.
            assert float(widget.w) == float(widget.h) == 9.0
            assert abs(float(widget.x) - (32 - 9) / 2.0) < 1e-6
            assert abs(float(widget.y) - (32 - 9) / 2.0) < 1e-6
        finally:
            session.shutdown()

    def test_node_switch_tears_down_region_and_markers(self):
        # A node switch (show_tree_node — any transform / Workflow click)
        # leaves the caret mounted, so czb_close never fires; the search box,
        # crosshair and found markers must be swept there instead of staying
        # painted over the new node. (czb_run/czb_pick add their found markers
        # AFTER _display, so the post-run markers are unaffected.)
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_set_region, czb_open
        from spyde.actions.lifecycle import show_tree_node
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d())
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            czb_set_region(session, src, {"half_square_width": 10})
            czb_open(session, src, {})           # Manual crosshair too
            region = tree._czb_region_mg
            assert region is not None
            assert tree._czb_cross is not None

            show_tree_node(src, tree, tree.root)
            assert getattr(tree, "_czb_region_mg", None) is None
            assert getattr(tree, "_czb_region_handler", None) is None
            assert getattr(tree, "_czb_cross", None) is None
            assert getattr(tree, "_czb_found_mgs", None) is None
            assert region.visible is False
        finally:
            session.shutdown()

    def test_tree_close_tears_down_region_widgets(self):
        # BaseSignalTree.close() must sweep the CZB widget attrs (they were
        # missing from its hard-coded teardown list).
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_set_region, czb_open
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d())
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            czb_set_region(session, src, {"half_square_width": 10})
            czb_open(session, src, {})
            region, cross = tree._czb_region_mg, tree._czb_cross

            tree.close()
            for attr in ("_czb_region_mg", "_czb_region_handler",
                         "_czb_cross", "_czb_found_mgs"):
                assert getattr(tree, attr, None) is None, attr
            assert region.visible is False
            assert cross.visible is False
        finally:
            session.shutdown()

    def test_manual_center_from_crosshair(self):
        from spyde.backend.session import Session
        from spyde.actions.center_zero_beam import czb_open, czb_pick
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_off_center_4d(beam=(18, 14)))
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree
            before = src.plot_state.current_signal

            czb_open(session, src, {})
            assert getattr(tree, "_czb_cross", None) is not None
            # Simulate the user dragging the crosshair onto the beam (18, 14).
            tree._czb_cross.set(cx=18.0, cy=14.0)

            czb_pick(session, src, {})
            assert _wait(lambda: src.plot_state.current_signal is not before,
                         timeout=20), "manual centering never produced a new signal"
            centered = src.plot_state.current_signal
            frame = centered.inav[0, 0].data
            if hasattr(frame, "compute"):
                frame = frame.compute()
            cx, cy = _com(frame)
            assert abs(cx - 16) < 2 and abs(cy - 16) < 2, (cx, cy)
            # Crosshair is cleared after applying.
            assert getattr(tree, "_czb_cross", None) is None
        finally:
            session.shutdown()

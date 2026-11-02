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
from pathlib import Path

import numpy as np
import pytest
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, close_session, make_session
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


def _assert_origin_on_beam(centered, source):
    """The centered node's detector axes read (0, 0) on the pixel the beam was
    moved to (size / 2), and the source node keeps its own calibration."""
    for axis, source_axis in zip(centered.axes_manager.signal_axes,
                                 source.axes_manager.signal_axes):
        assert axis.offset == -axis.scale * axis.size / 2, axis.offset
        assert source_axis.offset == 0.0


class TestCenterZeroBeam:
    def test_auto_centers_beam(self):
        from spyde.actions.center_zero_beam import czb_run
        session = make_session()
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
            _assert_origin_on_beam(centered, before)
        finally:
            close_session(session)

    def test_auto_region_shows_draggable_widget_and_run_uses_it(self):
        # The Automatic tab's search-window box must be a DRAGGABLE
        # RectangleWidget (add_rectangle_widget), not a static marker (the old
        # add_squares bug: no drag handles, geometry never fed back into the
        # search). czb_set_region opens it; resizing it changes the region
        # czb_run actually searches.
        from spyde.actions.center_zero_beam import czb_set_region, czb_run, czb_close
        session = make_session()
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
            close_session(session)

    def test_region_widget_torn_down_on_close(self):
        from spyde.actions.center_zero_beam import czb_set_region, czb_close
        session = make_session()
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
            close_session(session)

    def test_activating_automatic_tab_shows_full_frame_box_immediately(self):
        # Opening the Automatic tab (half_square_width defaults to 0, "full
        # frame") must show the box right away, not require the user to first
        # type a nonzero half-width — this was the actual #10 bug (the box
        # either never appeared or wasn't interactive).
        from spyde.actions.center_zero_beam import czb_set_region
        session = make_session()
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
            close_session(session)

    def test_region_drag_handler_runs_once_per_event(self):
        # REGRESSION (reviewer repro): Widget.set() fires pointer_move
        # UNCONDITIONALLY, so the re-centre handler's own set() used to
        # re-enter itself recursively (~2000 deep on ONE JS drag frame,
        # surviving only via recursionlimit + swallowed RecursionError).
        # Drive the REAL JS→Python path and count the handler's set().
        from spyde.actions.center_zero_beam import czb_set_region
        session = make_session()
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
            close_session(session)

    def test_node_switch_tears_down_region_and_markers(self):
        # A node switch (show_tree_node — any transform / Workflow click)
        # leaves the caret mounted, so czb_close never fires; the search box,
        # crosshair and found markers must be swept there instead of staying
        # painted over the new node. (czb_run/czb_pick add their found markers
        # AFTER _display, so the post-run markers are unaffected.)
        from spyde.actions.center_zero_beam import czb_set_region, czb_open
        from spyde.actions.lifecycle import show_tree_node
        session = make_session()
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
            close_session(session)

    def test_tree_close_tears_down_region_widgets(self):
        # BaseSignalTree.close() must sweep the CZB widget attrs (they were
        # missing from its hard-coded teardown list).
        from spyde.actions.center_zero_beam import czb_set_region, czb_open
        session = make_session()
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
            close_session(session)

    def test_manual_center_from_crosshair(self):
        from spyde.actions.center_zero_beam import czb_open, czb_pick
        session = make_session()
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
            _assert_origin_on_beam(centered, before)
        finally:
            close_session(session)


# ── Corners only: the descan plane from the four scan corners ───────────────

#: The scan's descan ramp, as pyxem's ``center − beam`` shift in pixels:
#: ``(offset, per column, per row)`` for x and for y.
_DESCAN_X = (1.0, 0.12, -0.08)
_DESCAN_Y = (-0.5, 0.05, 0.10)


def _gaussian(xx, yy, x, y, sigma=2.0):
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))


def _descan_with_sample(nav=24, sig=48, sample=True):
    """A scan whose direct beam follows a known linear descan, with a sample
    across the MIDDLE of the scan that the four corners never see.

    The sample is a second, weaker spot 12 px to the right of the beam. It
    drags a whole-pattern center of mass ~4.5 px off the beam wherever it is
    present, so any fit that includes the middle of the scan is biased, while
    the corners (15 % boxes here) are clean vacuum.
    """
    yy, xx = np.mgrid[0:sig, 0:sig].astype(np.float64)
    iy, ix = np.mgrid[0:nav, 0:nav].astype(np.float64)
    shift_x = _DESCAN_X[0] + _DESCAN_X[1] * ix + _DESCAN_X[2] * iy
    shift_y = _DESCAN_Y[0] + _DESCAN_Y[1] * ix + _DESCAN_Y[2] * iy
    data = np.zeros((nav, nav, sig, sig), dtype=np.float32)
    for row in range(nav):
        for column in range(nav):
            beam_x = sig / 2.0 - shift_x[row, column]
            beam_y = sig / 2.0 - shift_y[row, column]
            frame = 100.0 * _gaussian(xx, yy, beam_x, beam_y)
            if sample and 6 <= row < 18 and 6 <= column < 18:
                frame += 60.0 * _gaussian(xx, yy, beam_x + 12.0, beam_y)
            data[row, column] = frame
    signal = hs.signals.Signal2D(data)
    signal.set_signal_type("electron_diffraction")
    return signal, np.stack([shift_x, shift_y], axis=-1)


def _beam_position(frame, around, radius=6):
    """Center of mass of *frame* inside a disc of *radius* around *around* —
    the beam alone, without the sample spot 12 px away."""
    frame = np.asarray(frame, dtype=np.float64)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    inside = (xx - around[0]) ** 2 + (yy - around[1]) ** 2 <= radius ** 2
    weights = np.where(inside, frame, 0.0)
    total = weights.sum()
    return (xx * weights).sum() / total, (yy * weights).sum() / total


def _frame(signal, column, row):
    frame = signal.inav[column, row].data
    return frame.compute() if hasattr(frame, "compute") else frame


class TestCornersOnly:
    """The Automatic tab can fit the descan plane through the four scan
    corners only, the way the DPC Center tab's Corners reference does."""

    def test_the_corner_reference_ignores_the_sample(self):
        from spyde.actions.dpc import measure_corner_reference
        signal, truth = _descan_with_sample()
        reference = measure_corner_reference(
            signal, method="center_of_mass", corner_fraction=0.15)
        assert reference.shape == truth.shape
        assert np.abs(reference - truth).max() < 0.05

    def test_the_whole_scan_plane_is_biased_by_the_same_sample(self):
        """The control: the case the corners option exists for really is one
        the whole-scan plane gets wrong."""
        from spyde.actions.dpc import measure_beam_shifts, plane_through
        signal, truth = _descan_with_sample()
        whole_scan = plane_through(
            measure_beam_shifts(signal, method="center_of_mass"), None)
        assert np.abs(whole_scan - truth).max() > 1.0

    def _centered(self, plane_fit):
        from spyde.actions.center_zero_beam import czb_run
        session = make_session()
        try:
            signal, _truth = _descan_with_sample()
            session._add_signal(signal)
            _settle(session)
            src = _signal_plot(session)
            before = src.plot_state.current_signal
            czb_run(session, src, {"method": "center_of_mass",
                                   "plane_fit": plane_fit,
                                   "corner_fraction": 0.15})
            assert _wait(lambda: src.plot_state.current_signal is not before,
                         timeout=30), "centering never produced a new signal"
            centered = src.plot_state.current_signal
            node = session.signal_trees[0].get_node(before)
            names = list(node.children)
            return ({(column, row): _frame(centered, column, row)
                     for column in (0, 12, 23) for row in (0, 12, 23)}, names)
        finally:
            close_session(session)

    def test_corners_put_the_beam_on_the_pattern_center_everywhere(self):
        frames, names = self._centered("corners")
        assert any("Centered" in name for name in names)
        for key, frame in frames.items():
            position = _beam_position(frame, (24, 24))
            assert np.hypot(position[0] - 24, position[1] - 24) < 0.3, (key, position)

    def test_the_whole_scan_plane_leaves_the_beam_off_center(self):
        """The control on the full path: the sample pulls the whole-scan plane
        (and so every centered pattern) more than a pixel off."""
        frames, _names = self._centered("whole_scan")
        worst = max(np.hypot(*(np.subtract(_beam_position(frame, (24, 24)), 24)))
                    for frame in frames.values())
        assert worst > 1.0

    def test_corners_need_a_two_dimensional_scan(self, monkeypatch):
        from spyde.actions import center_zero_beam
        from spyde.actions.center_zero_beam import czb_run
        errors = []
        monkeypatch.setattr(center_zero_beam, "emit_error", errors.append)
        session = make_session()
        try:
            data = np.zeros((5, 16, 16), dtype=np.float32)
            data[:, 8, 8] = 1.0
            session._add_signal(hs.signals.Signal2D(data))
            _settle(session)
            src = _signal_plot(session)
            czb_run(session, src, {"method": "center_of_mass",
                                   "plane_fit": "corners"})
            assert _wait(lambda: bool(errors), timeout=10)
            assert "2-D scan" in errors[0]
        finally:
            close_session(session)

    def test_the_corner_boxes_show_on_the_navigator(self):
        from spyde.actions.center_zero_beam import czb_close, czb_set_region
        session = make_session()
        try:
            signal, _truth = _descan_with_sample()
            session._add_signal(signal)
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree

            czb_set_region(session, src, {"half_square_width": 0,
                                          "plane_fit": "corners",
                                          "corner_fraction": 0.15})
            boxes = getattr(tree, "_czb_corner_mg", None)
            assert boxes is not None, "no corner boxes on the navigator"
            czb_set_region(session, src, {"half_square_width": 0,
                                          "plane_fit": "corners",
                                          "corner_fraction": 0.25})
            assert tree._czb_corner_mg is boxes, "a resize must not rebuild them"
            czb_set_region(session, src, {"half_square_width": 0,
                                          "plane_fit": "none"})
            assert getattr(tree, "_czb_corner_mg", None) is None
            czb_set_region(session, src, {"half_square_width": 0,
                                          "plane_fit": "corners",
                                          "corner_fraction": 0.15})
            czb_close(session, src, {})
            assert getattr(tree, "_czb_corner_mg", None) is None
        finally:
            close_session(session)


_CARET = (Path(__file__).resolve().parents[3] / "electron" / "src" / "renderer"
          / "src" / "components" / "CenterZeroBeamWizard.tsx")


class TestCornersOnTheCaret:
    """The option is declared once, in the backend schema, and offered by the
    caret with the DPC Center tab's wording."""

    def test_the_schema_offers_corners(self):
        from spyde.actions import registry
        from spyde.corners import DEFAULT_CORNER_FRACTION
        schema = registry.wizard_parameters("czb")
        assert schema["plane_fit"]["choices"] == ["none", "whole_scan", "corners"]
        assert schema["corner_fraction"]["default"] == DEFAULT_CORNER_FRACTION

    @pytest.mark.skipif(not _CARET.exists(), reason="renderer sources not present")
    def test_the_caret_sends_the_plane_fit_and_box_size(self):
        text = _CARET.read_text(encoding="utf-8")
        assert "'corners'" in text
        assert "Corners (plane through 4 boxes)" in text
        assert "plane_fit:" in text and "corner_fraction:" in text

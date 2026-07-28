"""The background preview must follow the band as it is dragged.

The geometry of a range widget is on ``event.source``, not on the event —
anyplotlib's ``Event`` carries x/y and a few scalars, and the widget it hands
back carries ``x0``/``x1``. Reading them off the event returned None every
time, so the drag handler bailed at its first check and the preview never
moved. The same mistake the Fit handles made.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hyperspy.api as hs

from spyde.actions.background_action import bg_close, bg_open


def _spectrum_image(ny=3, nx=4, nc=192):
    """A decaying background with a peak on top of it."""
    x = np.linspace(10.0, 200.0, nc)
    y = 5e5 * x ** -2.0 + 400.0 * np.exp(-0.5 * ((x - 140.0) / 8.0) ** 2) + 5.0
    data = np.tile(y, (ny, nx, 1))
    s = hs.signals.Signal1D(data)
    ax = s.axes_manager.signal_axes[0]
    ax.offset, ax.scale = float(x[0]), float(x[1] - x[0])
    s.metadata.General.title = "bg test"
    return s


@pytest.fixture
def opened(window):
    session = window["window"]
    session._add_signal(_spectrum_image())
    tree = window["signal_trees"][0]
    plot = next(iter(tree.signal_plots))
    bg_open(session, plot, {})
    return session, plot, tree, tree._bg_wizard


class _Src:
    def __init__(self, x0, x1):
        self.x0, self.x1 = x0, x1


class _Event:
    def __init__(self, x0, x1, event_type="pointer_move"):
        self.source = _Src(x0, x1)
        self.event_type = event_type


class TestTheDragIsHeard:
    def test_the_geometry_comes_off_event_source(self, window, opened):
        _session, _plot, _tree, wiz = opened
        wiz._on_drag(_Event(60.0, 90.0))
        assert wiz.window == (60.0, 90.0)

    def test_an_event_without_a_source_is_ignored(self, window, opened):
        """A bare event carries no x0/x1 — that is the whole bug, so a stray
        one must not blank the window."""
        _session, _plot, _tree, wiz = opened
        before = wiz.window

        class _Bare:
            event_type = "pointer_move"
        wiz._on_drag(_Bare())
        assert wiz.window == before

    def test_a_dict_event_works_too(self, window, opened):
        _session, _plot, _tree, wiz = opened
        wiz._on_drag({"x0": 40.0, "x1": 70.0})
        assert wiz.window == (40.0, 70.0)

    def test_the_window_is_normalised(self, window, opened):
        """Dragging right-to-left gives x1 < x0."""
        _session, _plot, _tree, wiz = opened
        wiz._on_drag(_Event(90.0, 60.0))
        assert wiz.window == (60.0, 90.0)


class TestThePreviewFollows:
    def test_the_curve_changes_when_the_band_moves(self, window, opened):
        """The point of it: a different window fits a different background."""
        _session, _plot, _tree, wiz = opened
        wiz._on_drag(_Event(20.0, 60.0))
        first = np.array(wiz._line._entry()["data"])
        wiz._on_drag(_Event(120.0, 190.0))
        second = np.array(wiz._line._entry()["data"])
        assert not np.allclose(first, second), "the preview did not move"

    def test_the_line_is_updated_in_place_not_rebuilt(self, window, opened):
        """Removing and re-adding a line every drag frame is heavy AND does
        not repaint during the drag — the curve simply does not follow."""
        _session, _plot, _tree, wiz = opened
        wiz._on_drag(_Event(20.0, 60.0))
        line = wiz._line
        for x0 in (30.0, 40.0, 50.0):
            wiz._on_drag(_Event(x0, x0 + 40.0))
        assert wiz._line is line, "the preview line was rebuilt mid-drag"

    def test_only_one_background_line_exists(self, window, opened):
        _session, plot, _tree, wiz = opened
        for x0 in (20.0, 30.0, 40.0):
            wiz._on_drag(_Event(x0, x0 + 40.0))
        p1 = plot._plot1d
        labels = [e.get("label") for e in p1._state["extra_lines"]]
        assert labels.count("background") == 1


class TestTheCurveStaysOnScale:
    """The background is drawn — and subtracted — across the WHOLE axis, not
    just the window. A PowerLaw whose `origin` sits on an axis that reaches
    zero is singular there, so a window on the far side of a peak extrapolated
    to 3e9 at the left edge: the plot's y-range blew away and the spectrum
    vanished behind it.
    """

    @staticmethod
    def _axis_from_zero(session):
        s = _spectrum_image()
        ax = s.axes_manager.signal_axes[0]
        ax.offset = 0.0                     # the axis now reaches the origin
        session._add_signal(s)
        return s

    def test_a_window_right_of_the_peak_does_not_explode(self, window):
        session = window["window"]
        self._axis_from_zero(session)
        tree = window["signal_trees"][-1]
        plot = next(iter(tree.signal_plots))
        bg_open(session, plot, {})
        wiz = tree._bg_wizard
        wiz.model_kind = "PowerLaw"

        data = np.asarray(wiz.signal.data, float)
        peak = float(np.nanmax(data))
        x = wiz.axis()
        wiz._on_drag(_Event(float(x[len(x) * 3 // 4]), float(x[-1])))

        curve = np.array(wiz._line._entry()["data"])
        assert np.isfinite(curve).all()
        assert float(np.max(curve)) < 100 * peak, (
            f"the background extrapolated to {np.max(curve):.3g} against a "
            f"peak of {peak:.3g} — the plot's y-range is gone")

    def test_the_origin_is_moved_off_a_zero_based_axis(self, window):
        session = window["window"]
        self._axis_from_zero(session)
        tree = window["signal_trees"][-1]
        plot = next(iter(tree.signal_plots))
        bg_open(session, plot, {})
        wiz = tree._bg_wizard
        wiz.model_kind = "PowerLaw"
        spec = wiz.build_spec()
        assert spec.components[0]["origin"].value < 0.0

    def test_it_is_left_at_zero_when_the_axis_does_not_reach_it(self, window,
                                                                opened):
        """HyperSpy's convention, kept wherever it is safe — the fixture's
        axis starts at 10."""
        _session, _plot, _tree, wiz = opened
        wiz.model_kind = "PowerLaw"
        assert wiz.build_spec().components[0]["origin"].value == 0.0


class TestDragCost:
    """A pointer_move redraws; the state message waits for the release."""

    @staticmethod
    def _states(window):
        return [m for m in window["messages"] if m.get("type") == "bg_state"]

    def test_a_live_move_does_not_emit_state(self, window, opened):
        _session, _plot, _tree, wiz = opened
        before = len(self._states(window))
        wiz._on_drag(_Event(20.0, 60.0, "pointer_move"))
        assert len(self._states(window)) == before

    def test_the_release_emits_state(self, window, opened):
        _session, _plot, _tree, wiz = opened
        before = len(self._states(window))
        wiz._on_drag(_Event(20.0, 60.0, "pointer_up"))
        assert len(self._states(window)) > before


class TestScoping:
    def test_closing_removes_the_band_and_the_curve(self, window, opened):
        session, plot, tree, wiz = opened
        p1 = plot._plot1d
        wiz._on_drag(_Event(20.0, 60.0))
        assert any(e.get("label") == "background"
                   for e in p1._state["extra_lines"])
        assert p1._widgets

        bg_close(session, plot, {})
        assert not any(e.get("label") == "background"
                       for e in p1._state["extra_lines"])
        assert not p1._widgets, "the span outlived the caret"
        assert getattr(tree, "_bg_wizard", None) is None

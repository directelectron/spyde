"""
test_figure.py — the shared image pane.

`robust_levels` gets the most attention because it is the one piece of real
arithmetic here, and every one of its cases is a way a live viewer goes solid
black or solid white on real hardware.
"""
from __future__ import annotations

import numpy as np
import pytest

from de_shell.plotting.figure import FigureView, robust_levels


class TestRobustLevels:
    def test_a_hot_pixel_does_not_set_the_ceiling(self):
        # THE case this exists for: one saturated pixel under a min/max range
        # compresses everything else into the bottom of the scale and the image
        # renders black. Every real detector has hot pixels.
        frame = np.full((64, 64), 100.0)
        frame[0, 0] = 65535.0
        lo, hi = robust_levels(frame)
        assert hi < 1000.0, "a single hot pixel dominated the display range"

    def test_uniform_frame_still_gets_a_usable_range(self):
        # A zero-width window renders as a solid block, indistinguishable from a
        # broken decode.
        lo, hi = robust_levels(np.full((8, 8), 7.0))
        assert hi > lo

    def test_all_nan_frame_does_not_raise_or_return_nan(self):
        lo, hi = robust_levels(np.full((8, 8), np.nan))
        assert np.isfinite(lo) and np.isfinite(hi) and hi > lo

    def test_partially_nan_frame_uses_the_finite_values(self):
        frame = np.full((8, 8), np.nan)
        frame[0, :4] = [1.0, 2.0, 3.0, 4.0]
        lo, hi = robust_levels(frame)
        assert np.isfinite(lo) and np.isfinite(hi) and hi > lo

    def test_spans_the_bulk_of_an_ordinary_frame(self):
        frame = np.linspace(0.0, 1000.0, 10_000).reshape(100, 100)
        lo, hi = robust_levels(frame)
        assert 0.0 <= lo < 50.0 and 950.0 < hi <= 1000.0

    def test_percentiles_are_configurable(self):
        frame = np.linspace(0.0, 100.0, 10_000).reshape(100, 100)
        tight = robust_levels(frame, 25.0, 75.0)
        wide = robust_levels(frame, 0.0, 100.0)
        assert tight[0] > wide[0] and tight[1] < wide[1]

    def test_integer_frames_are_handled(self):
        lo, hi = robust_levels(np.arange(256, dtype=np.uint16).reshape(16, 16))
        assert np.isfinite(lo) and np.isfinite(hi) and hi > lo


class TestFigureViewLifecycle:
    """No anyplotlib figure is built here — these cover the guards around it,
    which are what keep teardown and pre-open calls from raising."""

    def test_starts_closed_with_no_fig_id(self):
        v = FigureView(0, title="t")
        assert v.fig_id is None and not v.is_open

    def test_painting_before_open_is_a_no_op_not_a_crash(self):
        # A frame can arrive from an acquisition thread before open() has run.
        assert FigureView(0).show(np.zeros((4, 4))) is False

    def test_colormap_and_title_before_open_are_retained(self):
        v = FigureView(0)
        v.set_colormap("viridis")
        v.set_title("later")
        assert v._colormap == "viridis" and v.title == "later"

    def test_close_is_idempotent(self):
        v = FigureView(0)
        v.close()
        v.close()
        assert not v.is_open

    def test_painting_after_close_is_a_no_op(self):
        # The teardown ordering that matters: acquisition is stopped first, but
        # a frame already in flight must not resurrect a closed window.
        v = FigureView(0)
        v.close()
        assert v.show(np.zeros((4, 4))) is False

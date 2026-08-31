"""RoiTrace — the always-on detector for an integrating ROI that moves on its own.

The value of this thing is entirely in its false-positive rate: it runs on every
pointer_move, so a rule that fires during ordinary dragging makes the log useless
and the real jump invisible. These tests are mostly about what must NOT fire.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from spyde.drawing.selectors.base_selector import MAX_REGION_EXTENT_PER_DIM
from spyde.drawing.selectors.roi_trace import RoiTrace
from spyde.tests.migrated.conftest import _settle, make_session


def _jump_records(caplog):
    return [r for r in caplog.records if "[ROI-JUMP]" in r.getMessage()]


class TestQuietDuringNormalDragging:
    """Every one of these is a thing a user does constantly."""

    def test_pure_translation_is_silent(self, caplog):
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            for k in range(6):
                g = [(1.0 + k * 0.05, 1.8 + k * 0.05)]
                t.observe(g, g, n_indices=16, n_unique=16)
        assert not _jump_records(caplog)

    def test_dragging_one_edge_is_silent(self, caplog):
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            for k in range(6):
                g = [(1.0, 1.2 + k * 0.1)]        # only the upper edge moves
                t.observe(g, g, n_indices=4 + k, n_unique=4 + k)
        assert not _jump_records(caplog)

    def test_dragging_the_lower_edge_is_silent(self, caplog):
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            for k in range(6):
                g = [(1.0 - k * 0.1, 1.8)]
                t.observe(g, g, n_indices=16, n_unique=16)
        assert not _jump_records(caplog)

    def test_a_fast_flick_is_silent(self, caplog):
        """A quick mouse move legitimately translates the ROI a long way in one
        event — distance must NOT be a rule, or every fast drag cries wolf."""
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            t.observe([(1.0, 1.8)], [(1.0, 1.8)], n_indices=16, n_unique=16)
            t.observe([(19.0, 19.8)], [(19.0, 19.8)], n_indices=16, n_unique=16)
        assert not _jump_records(caplog)

    def test_a_single_point_region_is_silent(self, caplog):
        """n==1 and unique==1 is a crosshair, not a collapsed span."""
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            t.observe([(2.0, 2.0)], [(2.0, 2.0)], n_indices=1, n_unique=1)
        assert not _jump_records(caplog)

    def test_first_event_has_nothing_to_compare_against(self, caplog):
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            t.observe([(5.0, 9.0)], [(5.0, 9.0)], n_indices=16, n_unique=16)
        assert not _jump_records(caplog)


class TestFiresOnRealAnomalies:
    def test_clamp_rewriting_geometry_fires(self, caplog):
        """The widget-side max_extent is supposed to make the fallback clamp dead
        code during a drag. If it rewrites geometry, that assumption broke."""
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            t.observe([(1.0, 7.0)], [(1.0, 1.8)], n_indices=16, n_unique=16)
        recs = _jump_records(caplog)
        assert len(recs) == 1
        assert "clamp rewrote geometry" in recs[0].getMessage()

    def test_moved_and_resized_in_one_event_fires(self, caplog):
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            t.observe([(1.0, 1.8)], [(1.0, 1.8)], n_indices=16, n_unique=16)
            # both ends move, by different amounts: neither a drag of one edge
            # nor a drag of the body
            t.observe([(1.4, 3.9)], [(1.4, 3.9)], n_indices=16, n_unique=16)
        recs = _jump_records(caplog)
        assert len(recs) == 1
        assert "moved AND resized" in recs[0].getMessage()

    def test_collapse_onto_one_position_fires(self, caplog):
        """The span ran off the end of the data, so every index clamped to the
        last frame — 16 points describing one frame."""
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            t.observe([(1.9, 2.7)], [(1.9, 2.7)], n_indices=16, n_unique=1)
        recs = _jump_records(caplog)
        assert len(recs) == 1
        assert "collapsed onto one position" in recs[0].getMessage()

    def test_the_warning_carries_the_preceding_events(self, caplog):
        """A single anomalous event says little; the run-up says what the user
        was doing. That context is the whole reason this is a ring buffer."""
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            for k in range(4):
                g = [(1.0 + k * 0.05, 1.8 + k * 0.05)]
                t.observe(g, g, n_indices=16, n_unique=16)
            t.observe([(9.0, 12.0)], [(9.0, 9.8)], n_indices=16, n_unique=16)
        msg = _jump_records(caplog)[0].getMessage()
        assert msg.count("\n") >= 4, "the ring buffer was not dumped"
        assert "1.15" in msg or "1.1" in msg, "earlier events missing from the dump"

    def test_2d_rectangle_axes_are_checked_independently(self, caplog):
        t = RoiTrace("rect")
        with caplog.at_level(logging.WARNING):
            t.observe([(0.0, 8.0), (0.0, 8.0)], [(0.0, 8.0), (0.0, 8.0)],
                      n_indices=64, n_unique=64)
            # x translates cleanly; y both moves and resizes
            t.observe([(2.0, 10.0), (3.0, 20.0)], [(2.0, 10.0), (3.0, 20.0)],
                      n_indices=64, n_unique=64)
        recs = _jump_records(caplog)
        assert len(recs) == 1
        assert "axis1" in recs[0].getMessage()
        assert "axis0" not in recs[0].getMessage()


class TestRobustness:
    def test_trace_all_env_logs_every_event(self, caplog, monkeypatch):
        monkeypatch.setenv("SPYDE_ROI_TRACE", "1")
        t = RoiTrace("test")
        with caplog.at_level(logging.INFO):
            for k in range(3):
                g = [(1.0 + k, 2.0 + k)]
                t.observe(g, g, n_indices=16, n_unique=16)
        assert len([r for r in caplog.records
                    if "[ROI-TRACE]" in r.getMessage()]) == 3

    @pytest.mark.parametrize("bad", [None, "nonsense", [(1.0,)], [()]])
    def test_never_raises_on_junk(self, bad):
        """A diagnostic that can break a drag is worse than no diagnostic."""
        t = RoiTrace("test")
        t.observe([(1.0, 2.0)], [(1.0, 2.0)], n_indices=4, n_unique=4)
        t.observe(bad, bad, n_indices=None, n_unique=None)

    def test_mismatched_axis_count_is_ignored_not_fatal(self, caplog):
        t = RoiTrace("test")
        with caplog.at_level(logging.WARNING):
            t.observe([(1.0, 2.0)], [(1.0, 2.0)], n_indices=4, n_unique=4)
            t.observe([(1.0, 2.0), (3.0, 4.0)], [(1.0, 2.0), (3.0, 4.0)],
                      n_indices=4, n_unique=4)
        assert not _jump_records(caplog)


class TestWiredIntoTheSelectors:
    """The rules are useless if nothing feeds them."""

    @staticmethod
    def _movie_session():
        import time
        import hyperspy.api as hs

        s = hs.signals.Signal2D(
            np.random.RandomState(1).rand(64, 8, 8).astype(np.float32))
        tax = s.axes_manager.navigation_axes[0]
        tax.name, tax.units, tax.scale = "time", "s", 0.05
        sess = make_session()
        sess._add_signal(s, source_path=None)
        _settle(sess)
        return sess

    def test_sliding_an_established_span_is_quiet(self, caplog):
        """The property that matters: once the span exists and is within the cap,
        sliding it must not trip the detector on ANY step. (The very first event
        after a programmatic set legitimately does — see the next test — so the
        span is established first and the log cleared.)"""
        sess = self._movie_session()
        try:
            region = next(iter(
                sess._nav_selectors.values()))._linear_region_selector
            w = region._widget
            assert w is not None
            w.x0, w.x1 = 1.0, 1.8
            region._on_pointer_up(None)
            assert region._roi_trace is not None, "the trace was never built"
            with caplog.at_level(logging.WARNING):
                caplog.clear()
                for k in range(1, 8):          # slide it, one frame per step
                    w.x0, w.x1 = 1.0 + k * 0.05, 1.8 + k * 0.05
                    region._on_pointer_up(None)
            assert not _jump_records(caplog), (
                "sliding an established span tripped the detector:\n"
                + "\n".join(r.getMessage() for r in _jump_records(caplog)))
        finally:
            sess.shutdown()

    def test_a_span_at_the_cap_does_not_rewrite_every_event(self, caplog):
        """Regression for the epsilon rewrite: a span resting EXACTLY on the cap
        computes hi-lo fractionally over it in floating point, so a bare `>` made
        _clamp_extent write the widget on nearly every pointer_move — echoing
        python-sourced geometry back into a live drag."""
        sess = self._movie_session()
        try:
            region = next(iter(
                sess._nav_selectors.values()))._linear_region_selector
            w = region._widget
            from spyde.drawing.selectors.selector1d import _signal_axis
            scale, _ = _signal_axis(region)
            cap = abs(MAX_REGION_EXTENT_PER_DIM * scale)
            w.x0, w.x1 = 1.0, 1.0 + cap
            region._on_pointer_up(None)
            with caplog.at_level(logging.WARNING):
                caplog.clear()
                for k in range(1, 8):
                    lo = 1.0 + k * scale       # exactly at the cap every step
                    w.x0, w.x1 = lo, lo + cap
                    before = (float(w.x0), float(w.x1))
                    region._clamp_extent()
                    assert (float(w.x0), float(w.x1)) == before, (
                        f"step {k}: _clamp_extent rewrote a span that is exactly "
                        f"at the cap ({before} -> "
                        f"{(float(w.x0), float(w.x1))})")
            assert not _jump_records(caplog)
        finally:
            sess.shutdown()

    def test_a_programmatic_over_cap_set_does_fire(self, caplog):
        """The clamp firing on geometry that never went through a drag is a TRUE
        positive — that is the case it exists for, and it is worth seeing."""
        sess = self._movie_session()
        try:
            region = next(iter(
                sess._nav_selectors.values()))._linear_region_selector
            w = region._widget
            w.x0, w.x1 = 1.0, 1.8
            region._on_pointer_up(None)
            with caplog.at_level(logging.WARNING):
                caplog.clear()
                region._widget.x1 = 9.0        # bypasses any drag-time cap
                region._on_pointer_up(None)
            recs = _jump_records(caplog)
            if recs:                            # widget may clamp on assign
                assert "clamp rewrote geometry" in recs[0].getMessage()
            assert float(w.x1) - float(w.x0) <= abs(
                MAX_REGION_EXTENT_PER_DIM * 0.05) + 1e-9
        finally:
            sess.shutdown()

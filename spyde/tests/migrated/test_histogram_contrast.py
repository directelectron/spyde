"""Histogram binning + the dock's Auto button.

The reported symptom: on real (Poisson) data the histogram's bars are crushed
into the leftmost sliver of the widget, so the contrast handles have nowhere
useful to grab. The cause is that a counting frame's MAX sits far above its
bulk — hot pixels, or the central beam — and the bins used to span min–max.

These pin the fix: bin over the neighbourhood of the display range, clip (never
drop) the tail into the end bins, and give the dock a way to ask for auto levels
back.
"""
import numpy as np

from spyde.drawing.plots.plot import Plot


def _poisson_frame(seed: int = 0, hot: float = 40_000.0) -> np.ndarray:
    """A realistic counting frame: Poisson background, a bright central beam and
    a couple of hot pixels an order of magnitude above everything else."""
    rng = np.random.default_rng(seed)
    frame = rng.poisson(12.0, size=(128, 128)).astype(np.float64)
    frame[60:68, 60:68] += 3_000.0        # central beam
    frame[3, 5] = hot                     # hot pixel
    frame[100, 7] = hot * 0.9
    return frame


class TestHistRange:
    """The range is derived from the DISPLAY LEVELS, so the fixture's levels are
    the ones the app would actually use (_robust_levels on a signal frame)."""

    def _levels(self, frame):
        return Plot._robust_levels(frame, signal=True)

    def test_tail_does_not_own_the_axis(self):
        frame = _poisson_frame()
        vmin, vmax = self._levels(frame)
        lo, hi, clipped = Plot._hist_range(frame.ravel(), vmin, vmax)
        assert clipped, "a frame with hot pixels must report a clipped range"
        assert hi < frame.max() / 10, \
            f"upper bin edge {hi} is still dominated by the tail (max {frame.max()})"

    def test_the_handles_land_inside_the_drawn_range(self):
        """The point of the widget: both handles must be grabbable, with room to
        move either way rather than pinned against an edge."""
        frame = _poisson_frame()
        vmin, vmax = self._levels(frame)
        lo, hi, _ = Plot._hist_range(frame.ravel(), vmin, vmax)
        assert lo < vmin < vmax < hi
        # …and neither is squashed into a corner of it.
        assert 0.05 < (vmin - lo) / (hi - lo) < 0.5
        assert 0.5 < (vmax - lo) / (hi - lo) < 0.95

    def test_bulk_is_resolved_across_many_bins(self):
        """The actual complaint, measured: how much of the widget the data uses."""
        frame = _poisson_frame()
        finite = frame.ravel()
        vmin, vmax = self._levels(frame)
        # Old behaviour: bins over the full extent.
        old, _ = np.histogram(finite, bins=64)
        lo, hi, _ = Plot._hist_range(finite, vmin, vmax)
        new, _ = np.histogram(np.clip(finite, lo, hi), bins=64, range=(lo, hi))
        assert (old > 0).sum() <= 4, "fixture no longer reproduces the squish"
        assert (new > 0).sum() > 15, \
            f"bulk still crushed into {(new > 0).sum()} bins"

    def test_clipping_never_drops_samples(self):
        """np.histogram(range=…) would DROP the tail; clipping counts it."""
        frame = _poisson_frame()
        finite = frame.ravel()
        lo, hi, _ = Plot._hist_range(finite, *self._levels(frame))
        counts, _ = np.histogram(np.clip(finite, lo, hi), bins=64, range=(lo, hi))
        assert counts.sum() == finite.size

    def test_flat_frame_falls_back_to_full_extent(self):
        flat = np.full(256, 7.0)
        lo, hi, clipped = Plot._hist_range(flat, 7.0, 7.0)
        assert not clipped
        assert hi > lo, "bins must have width even when every pixel is identical"

    def test_uniform_data_is_not_clipped(self):
        """No tail to hide → the margin reaches the extent and nothing is cut."""
        rng = np.random.default_rng(1)
        data = rng.uniform(0, 100, size=100_000)
        lo, hi, clipped = Plot._hist_range(data, *Plot._robust_levels(data))
        assert not clipped
        assert (hi - lo) > 98


class TestHistogramMessage:
    def test_carries_the_full_extent(self, stem_4d_dataset):
        hg = [m for m in stem_4d_dataset["messages"] if m.get("type") == "histogram"]
        assert hg, "no histogram emitted"
        msg = hg[-1]
        assert len(msg["counts"]) == 64
        assert "data_min" in msg and "data_max" in msg and "clipped" in msg
        assert msg["data_min"] <= msg["edges"][0]
        assert msg["data_max"] >= msg["edges"][-1]


class TestAutoClim:
    def _signal_plot(self, window):
        plots = [p for p in window["plots"]
                 if isinstance(getattr(p, "current_data", None), np.ndarray)
                 and getattr(p, "current_data").ndim == 2]
        assert plots, "no 2-D plot to work with"
        return plots[-1]

    def test_restores_robust_levels_after_a_manual_drag(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        plot = self._signal_plot(stem_4d_dataset)
        wid = plot.window_id

        session.dispatch_action({"action": "set_clim", "window_id": wid,
                                 "payload": {"vmin": -50.0, "vmax": -49.0}})
        assert plot._last_levels == (-50.0, -49.0)

        stem_4d_dataset["messages"].clear()
        session.dispatch_action({"action": "auto_clim", "window_id": wid,
                                 "payload": {}})

        expected = Plot._robust_levels(plot.current_data,
                                       signal=not plot.is_navigator)
        assert plot._last_levels == expected
        hg = [m for m in stem_4d_dataset["messages"] if m.get("type") == "histogram"]
        assert hg, "Auto must re-emit the histogram so the handles follow"
        assert (hg[-1]["vmin"], hg[-1]["vmax"]) == expected

    def test_reset_spans_the_full_range(self, stem_4d_dataset):
        """Reset is the escape hatch from a robust view: the display range
        becomes the actual min–max, tail included."""
        session = stem_4d_dataset["window"]
        plot = self._signal_plot(stem_4d_dataset)
        data = plot.current_data

        stem_4d_dataset["messages"].clear()
        session.dispatch_action({"action": "auto_clim", "window_id": plot.window_id,
                                 "payload": {"mode": "full"}})

        assert plot._last_levels == (float(data.min()), float(data.max()))
        hg = [m for m in stem_4d_dataset["messages"] if m.get("type") == "histogram"]
        assert hg
        assert (hg[-1]["vmin"], hg[-1]["vmax"]) == (float(data.min()), float(data.max()))

    def test_reset_moves_the_handles_not_the_bins(self, stem_4d_dataset):
        """Reset changes the DISPLAY RANGE, so it must not redraw the histogram
        underneath it. Binning off the live clim meant Reset re-binned over the
        whole tail and crushed the bars back into the left edge — the exact
        squish this widget's binning exists to prevent. The bars belong to the
        frame; only the handles move."""
        session = stem_4d_dataset["window"]
        plot = self._signal_plot(stem_4d_dataset)
        wid = plot.window_id

        stem_4d_dataset["messages"].clear()
        session.dispatch_action({"action": "auto_clim", "window_id": wid,
                                 "payload": {}})
        auto = [m for m in stem_4d_dataset["messages"] if m.get("type") == "histogram"][-1]

        stem_4d_dataset["messages"].clear()
        session.dispatch_action({"action": "auto_clim", "window_id": wid,
                                 "payload": {"mode": "full"}})
        reset = [m for m in stem_4d_dataset["messages"] if m.get("type") == "histogram"][-1]

        assert reset["edges"] == auto["edges"], "Reset re-binned the histogram"
        assert reset["counts"] == auto["counts"], "Reset redrew the bars"
        assert reset["vmax"] >= auto["vmax"], "Reset did not widen the display range"

    def test_keeps_the_threshold_marker(self, stem_4d_dataset):
        """Auto re-emits the histogram; the Find-Vectors threshold line must
        survive that, not silently vanish."""
        session = stem_4d_dataset["window"]
        plot = self._signal_plot(stem_4d_dataset)
        plot._emit_histogram(plot.current_data, 0.0, 1.0, threshold=3.5)

        stem_4d_dataset["messages"].clear()
        session.dispatch_action({"action": "auto_clim",
                                 "window_id": plot.window_id, "payload": {}})
        hg = [m for m in stem_4d_dataset["messages"] if m.get("type") == "histogram"]
        assert hg and hg[-1]["threshold"] == 3.5

    def test_unknown_window_is_a_no_op(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        session.dispatch_action({"action": "auto_clim", "window_id": 999_999,
                                 "payload": {}})   # must not raise

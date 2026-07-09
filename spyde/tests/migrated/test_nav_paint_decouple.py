"""Nav paint is decoupled from the dispatcher via a newest-wins painter thread.

The nav READ stays serial on the dispatcher, but PAINTING happens on _NavPainter so
a slow transport doesn't block reading the next slider position (the slider "catch"
fix). A frame superseded before it paints is dropped (newest-wins).
"""
import threading
import time

import numpy as np

from spyde.drawing.plots.plot import _nav_painter, _NavPainter


class _RecorderPlot:
    """Stand-in Plot: records painted frames; its _set_array can be made slow."""
    def __init__(self, paint_delay=0.0):
        self.current_data = None
        self.painted = []
        self._paint_delay = paint_delay
        self._lock = threading.Lock()

    def _set_array(self, data):
        if self._paint_delay:
            time.sleep(self._paint_delay)
        with self._lock:
            self.painted.append(np.asarray(data).copy())


def _wait(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        time.sleep(0.01)
    return cond()


class TestNavPainter:
    def test_paint_happens_off_caller_thread(self):
        painter = _NavPainter()
        plot = _RecorderPlot()
        caller = threading.current_thread().ident
        seen = {}

        def _capture(d):
            seen["thread"] = threading.current_thread().ident
        plot._set_array = _capture

        painter.submit(plot, np.zeros((4, 4)))
        assert _wait(lambda: "thread" in seen)
        assert seen["thread"] != caller, "paint ran on the caller thread, not the painter"

    def test_newest_wins_drops_stale_paint(self):
        painter = _NavPainter()
        # A slow first paint occupies the painter; queue several more while it runs.
        plot = _RecorderPlot(paint_delay=0.2)
        painter.submit(plot, np.full((4, 4), 1.0))   # slow one starts
        time.sleep(0.02)
        # While the first paints, submit newer frames — only the LAST should paint.
        painter.submit(plot, np.full((4, 4), 2.0))
        painter.submit(plot, np.full((4, 4), 3.0))
        painter.submit(plot, np.full((4, 4), 9.0))   # newest
        # Wait for the queue to drain.
        assert _wait(lambda: len(plot.painted) >= 2, timeout=3.0)
        time.sleep(0.1)
        vals = [float(p[0, 0]) for p in plot.painted]
        # The slow first (1.0) painted; then ONLY the newest (9.0) — not 2.0/3.0.
        assert vals[0] == 1.0
        assert 9.0 in vals
        assert 2.0 not in vals and 3.0 not in vals, f"stale frames painted: {vals}"

    def test_slow_paint_does_not_block_submitter(self):
        painter = _NavPainter()
        plot = _RecorderPlot(paint_delay=0.3)
        painter.submit(plot, np.zeros((4, 4)))
        # Submitting again returns IMMEDIATELY even though a paint is in flight.
        t0 = time.perf_counter()
        painter.submit(plot, np.ones((4, 4)))
        submit_ms = (time.perf_counter() - t0) * 1000
        assert submit_ms < 50, f"submit blocked on the in-flight paint ({submit_ms:.0f}ms)"


class TestEnqueuePaint:
    def test_enqueue_sets_current_data_immediately(self):
        # Plot.enqueue_paint sets current_data now (so a concurrent read sees intent)
        # and paints off-thread. Use the real global painter with a recorder plot.
        plot = _RecorderPlot()
        frame = np.full((8, 8), 7.0)
        # Call the real method via the class (bind our recorder as self).
        from spyde.drawing.plots.plot import Plot
        Plot.enqueue_paint(plot, frame)
        assert plot.current_data is frame, "current_data not set immediately"
        assert _wait(lambda: len(plot.painted) == 1)
        np.testing.assert_array_equal(plot.painted[0], frame)

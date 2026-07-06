"""
Interactive-preempt of the background navigator/VI fill (task #11).

For a large movie the navigator sum reads the whole file from disk; that
background read otherwise starves the crosshair's own per-frame read, so the
signal plot freezes while the navigator fills. `_InteractiveActivity` lets the
fill YIELD: the nav read pokes it on every move, the fill calls wait_if_active()
between chunks and pauses briefly while scrubbing is recent, then resumes.
"""
from __future__ import annotations

import time
import threading

from spyde.drawing.update_functions import _InteractiveActivity


class TestInteractiveActivity:
    def test_wait_returns_fast_when_idle(self):
        act = _InteractiveActivity(quiet_s=0.2)
        # No recent poke → wait_if_active returns immediately.
        t0 = time.monotonic()
        act.wait_if_active()
        assert (time.monotonic() - t0) < 0.1

    def test_stop_aborts_the_wait_immediately(self):
        act = _InteractiveActivity(quiet_s=1.0)
        act.poke()                         # would otherwise block ~1s
        stop = threading.Event()
        stop.set()
        t0 = time.monotonic()
        act.wait_if_active(max_wait_s=5.0, stop=stop)
        assert (time.monotonic() - t0) < 0.1, "stop must abort the wait at once"

    def test_wait_blocks_after_a_poke(self):
        act = _InteractiveActivity(quiet_s=0.2)
        act.poke()
        t0 = time.monotonic()
        act.wait_if_active(max_wait_s=1.0)
        waited = time.monotonic() - t0
        # Blocks until ~quiet_s of no pokes (a bit more for the 50ms poll step).
        assert 0.15 <= waited <= 0.45, f"waited {waited:.3f}s"

    def test_continuous_pokes_are_capped_by_max_wait(self):
        act = _InteractiveActivity(quiet_s=0.5)
        stop = threading.Event()

        def poker():
            while not stop.is_set():
                act.poke()
                time.sleep(0.02)

        t = threading.Thread(target=poker, daemon=True)
        t.start()
        try:
            t0 = time.monotonic()
            act.wait_if_active(max_wait_s=0.3)   # a continuous drag can't starve
            waited = time.monotonic() - t0
            assert waited <= 0.45, f"max_wait not honoured: {waited:.3f}s"
        finally:
            stop.set()

    def test_fill_resumes_after_scrub_settles(self):
        act = _InteractiveActivity(quiet_s=0.15)
        act.poke()
        # Simulate the fill loop calling wait between chunks while the user stops.
        t0 = time.monotonic()
        act.wait_if_active(max_wait_s=1.0)   # returns once idle >= quiet
        first = time.monotonic() - t0
        # Immediately calling again (still idle) returns fast — fill runs freely.
        t1 = time.monotonic()
        act.wait_if_active(max_wait_s=1.0)
        second = time.monotonic() - t1
        assert first >= 0.10
        assert second < 0.1, "fill should run freely once scrubbing settled"

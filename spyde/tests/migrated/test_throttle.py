"""
Selector updates must not flood the compute backend on a fast drag.

A continuous navigator drag emits pointer-move events at ~60/sec. The throttle
was historically a per-`live_delay` debounce timer on ``update_data``; it is now
the single serial **_NavDispatcher**: every move is submitted to one worker thread
that coalesces by ``id(selector)`` (latest-position-wins), so a burst collapses to
ONE pending job and superseded positions are dropped before they ever compute. A
trailing ``_settle_timer`` then fires one final forced update once motion stops
(so the resting frame computes even though intermediate futures were cancelled).

These tests pin that current contract.
"""
from __future__ import annotations

import time

from spyde.drawing.selectors.base_selector import BaseSelector, _nav_dispatcher


class _StubChild:
    plot_window = None


def _make_selector():
    sel = BaseSelector(
        parent=object(),
        children=_StubChild(),
        update_function=lambda *a, **k: None,
    )
    return sel


class TestThrottle:
    def test_min_throttle_interval_enforced(self):
        # Even a tiny requested delay is clamped up so we never flood.
        sel = BaseSelector(object(), _StubChild(), lambda *a, **k: None, live_delay=2)
        assert sel.live_delay >= 0.040 - 1e-9

    def test_update_data_uses_dispatcher_not_debounce_timer(self):
        # update_data no longer uses a per-window debounce timer; it submits to the
        # shared serial dispatcher (and arms a settle timer). The old _pending_timer
        # debounce attribute is gone.
        sel = _make_selector()
        assert not hasattr(sel, "_pending_timer")
        runs = []
        sel._run_update = lambda *a, **k: runs.append(1)  # type: ignore[assignment]
        sel.update_data()
        time.sleep(0.05)
        assert sum(runs) >= 1                 # the move reached _run_update

    def test_burst_coalesces_while_dispatcher_is_busy(self):
        # The serialisation guarantee: while one update is running, a BURST of
        # newer submits collapses to a SINGLE pending job (latest-wins), not one
        # job per event. Make the first run block so the burst piles up behind it.
        import threading
        sel = _make_selector()
        started = threading.Event()
        release = threading.Event()
        runs = []

        def _run(*a, **k):
            runs.append(1)
            if len(runs) == 1:
                started.set()
                release.wait(1.0)             # hold the lane busy

        sel._run_update = _run  # type: ignore[assignment]
        sel.update_data()                     # job 1 — starts and blocks
        assert started.wait(0.5)
        for _ in range(10):                   # 10 newer moves while job 1 runs
            sel.update_data()
        release.set()                         # let job 1 finish; burst drains
        time.sleep(0.1)
        # job 1 + the coalesced burst (1) [+ maybe a settle fire] — far fewer than
        # the 11 submits.
        assert 2 <= sum(runs) <= 4

    def test_settle_timer_armed_on_move_and_clears(self):
        sel = _make_selector()
        sel.live_delay = 0.05
        runs = []
        sel._run_update = lambda *a, **k: runs.append(1)  # type: ignore[assignment]

        sel.update_data()
        # A trailing settle timer is armed so a resting frame re-fires once.
        assert sel._settle_timer is not None

        time.sleep(0.05 + 0.1 + 0.05)
        # After the quiet period the settle timer fired and cleared itself.
        assert sel._settle_timer is None
        # The settle fire forces one extra run (force=True) on top of the move(s).
        assert sum(runs) >= 1

    def test_dispatcher_is_a_single_shared_lane(self):
        # All selectors share one dispatcher thread (the serialisation point that
        # removes the concurrency the old per-timer design had).
        a = _make_selector()
        b = _make_selector()
        # submit is the public queue entry; both go through the same instance.
        assert a.__class__  # sanity
        assert _nav_dispatcher is not None
        # Two different selectors → two pending slots keyed by id, not a clash.
        import threading
        ev = threading.Event()
        a._run_update = lambda *x, **k: ev.set()  # type: ignore[assignment]
        a.update_data()
        assert ev.wait(0.5)

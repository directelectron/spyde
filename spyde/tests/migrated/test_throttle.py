"""
Selector updates must THROTTLE, not fire per mouse event.

A continuous navigator drag emits pointer-move events at ~60/sec. With the old
2 ms debounce that became ~60 computes/sec, flooding Dask with superseded futures
so the lazy navigator stuttered and new chunks didn't paint. `update_data` now
coalesces a burst into a single fire per `live_delay` window.
"""
from __future__ import annotations

import time

from spyde.drawing.selectors.base_selector import BaseSelector


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

    def test_burst_coalesces_into_one_fire(self):
        sel = _make_selector()
        sel.live_delay = 0.05
        fires = []
        sel.delayed_update_data = lambda *a, **k: fires.append(1)

        sel.update_data()
        first_timer = sel._pending_timer
        assert first_timer is not None
        # Further moves during the window must NOT reset/replace the timer.
        sel.update_data()
        sel.update_data()
        assert sel._pending_timer is first_timer

        time.sleep(0.12)
        assert sum(fires) == 1                 # one fire for the whole burst
        assert sel._pending_timer is None      # window closed; ready for the next

    def test_next_window_fires_again(self):
        sel = _make_selector()
        sel.live_delay = 0.04
        fires = []
        sel.delayed_update_data = lambda *a, **k: fires.append(1)
        sel.update_data(); time.sleep(0.08)
        sel.update_data(); time.sleep(0.08)
        assert sum(fires) == 2                  # a new burst → a new fire

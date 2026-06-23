"""
Navigator update model: a SINGLE serial dispatcher thread runs every selector
update one at a time (the Qt event-loop equivalent). This removes the concurrency
that used to race hyperspy's CachedDaskArray block bookkeeping
("ValueError: (i, j) is not in list") — so there is NO per-signal cache lock and
no generation/stale-body machinery anymore. Latest-wins is handled by the
dispatcher coalescing repeated submissions for a selector into one pending job.

These tests pin that contract:
  * updates never run concurrently (the cache call is never re-entered),
  * a burst of positions ends on the LAST one,
  * a newer submission supersedes an older queued one (it isn't computed twice).
"""
import threading
import time

import numpy as np

from spyde.drawing.selectors.base_selector import BaseSelector, _nav_dispatcher


class _RecorderChild:
    """Minimal stand-in for a Plot child receiving sliced data."""

    def __init__(self):
        self.current_data = None
        self.needs_auto_level = False
        self.multiplot_manager = None
        self.plot_window = None

    def update_data(self, data):
        self.current_data = data


class _PositionSelector(BaseSelector):
    """A selector whose 'position' the test drives directly."""

    def __init__(self, child, fn):
        super().__init__(child, child, fn, live_delay=2)
        self._pos = np.array([[0, 0]])

    def set_pos(self, y, x):
        self._pos = np.array([[int(x), int(y)]])   # widget order (cx, cy)

    def get_selected_indices(self) -> np.ndarray:   # bypass clip/plot_state
        return self._pos


def _wait_idle(timeout=2.0):
    """Wait until the dispatcher has drained its pending queue."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with _nav_dispatcher._lock:
            empty = not _nav_dispatcher._pending
        if empty:
            time.sleep(0.05)   # let the in-flight job finish
            return
        time.sleep(0.01)


def test_updates_run_serially_never_concurrently():
    """The dispatcher must run selector update bodies ONE AT A TIME — the whole
    point (so the cache is never re-entered). Drive many selectors at once and
    assert the per-update body never overlaps itself."""
    meter = {"active": 0, "max": 0}
    mlock = threading.Lock()

    def slice_fn(sel, ch, indices):
        with mlock:
            meter["active"] += 1
            meter["max"] = max(meter["max"], meter["active"])
        try:
            time.sleep(0.01)   # widen the window for any overlap to show
            return np.asarray(indices).copy()
        finally:
            with mlock:
                meter["active"] -= 1

    selectors = [_PositionSelector(_RecorderChild(), slice_fn) for _ in range(6)]
    for i, sel in enumerate(selectors):
        sel.set_pos(i, i)
        sel.delayed_update_data(force=True)
    _wait_idle()

    assert meter["max"] == 1, (
        f"update bodies ran {meter['max']}x concurrently — the dispatcher is not "
        "serial, so the cache can be raced"
    )


def test_latest_position_wins_under_burst():
    """After a burst of positions on one selector the child ends on the LAST."""
    child = _RecorderChild()

    def slow_slice_fn(sel, ch, indices):
        time.sleep(0.02)
        return np.asarray(indices).copy()

    sel = _PositionSelector(child, slow_slice_fn)
    for i in range(1, 6):
        sel.set_pos(i, i)
        sel.delayed_update_data(force=True)
    _wait_idle()

    assert child.current_data is not None
    last = np.asarray(child.current_data).reshape(-1)[:2]
    assert tuple(last) == (5, 5), f"child ended on {tuple(last)}, not (5, 5)"


def test_newer_submission_coalesces_older_one():
    """Submitting a selector again while a job is queued REPLACES the pending job
    (latest-wins) rather than running both — so a fast drag doesn't compute every
    intermediate position. We block the dispatcher on the first job, queue two
    more positions, then release; only the LAST queued position should compute
    after the blocker."""
    child = _RecorderChild()
    computed = []
    first = {"flag": True}
    in_first = threading.Event()
    release = threading.Event()

    def fn(sel, ch, indices):
        pos = tuple(np.asarray(indices).reshape(-1)[:2])
        if first["flag"]:
            first["flag"] = False
            in_first.set()
            release.wait(timeout=2.0)
        computed.append(pos)
        return np.asarray(indices).copy()

    sel = _PositionSelector(child, fn)

    # Job 1 — occupies the dispatcher and blocks at the gate.
    sel.set_pos(1, 1)
    sel.delayed_update_data(force=True)
    assert in_first.wait(timeout=2.0), "first job never ran"

    # While blocked, queue two more — the 2nd should overwrite the 1st pending.
    sel.set_pos(2, 2)
    sel.delayed_update_data(force=True)
    sel.set_pos(3, 3)
    sel.delayed_update_data(force=True)

    release.set()
    _wait_idle()

    # The blocker (1,1) computed, then ONLY the latest queued (3,3) — not (2,2).
    assert (1, 1) in computed
    assert (3, 3) in computed
    assert (2, 2) not in computed, f"stale intermediate position computed: {computed}"

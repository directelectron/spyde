"""
Reproduction: the navigator update path is not mutually exclusive.

`BaseSelector.update_data` throttles with `threading.Timer` (a NEW thread per
fire), and `delayed_update_data` can block for 100s of ms inside a slow update
(in the app: the Find-Vectors peak-preview index-hook does a synchronous
`block.compute()` per frame). So two fires run `delayed_update_data` CONCURRENTLY
and race the per-child future cancel/submit in
`update_from_navigation_selection` (and `Plot.update_data`'s `current_data`),
both unlocked. The image's async future gets cancelled/superseded while the
synchronous overlay markers keep painting — "the overlay updates but the
underlying image freezes on the previous frame" (the reported, distributed-Dask
symptom).

This is the "future-cancel greedy workflow isn't being followed": the greedy
cancel assumes ONE navigator update in flight; the throttle + a slow update
break that. The fix serialises `delayed_update_data` so the body (and its
cancel/submit) runs one at a time, latest position wins.

`test_delayed_update_is_mutually_exclusive` is expected to FAIL before the fix.
"""
import threading
import time

import numpy as np

from spyde.drawing.selectors.base_selector import BaseSelector


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


def test_slow_update_does_not_block_newer_one():
    """A slow in-flight update must NOT stall a newer fire (latest-wins, no long
    lock).  The previous design serialised the whole body with an RLock, so a
    slow compute held the lock and froze every other update — with two signals
    live the navigator flashed but never tracked the crosshair.  The new model
    lets the newer fire run immediately and supersede the stale one."""
    child = _RecorderChild()
    first = {"flag": True}
    in_body = threading.Event()
    release = threading.Event()
    second_ran = threading.Event()

    def slice_fn(sel, ch, indices):
        if first["flag"]:
            first["flag"] = False
            in_body.set()
            release.wait(timeout=2.0)   # first fire blocks here
        else:
            second_ran.set()             # second fire reaches the body
        return np.asarray(indices).copy()

    sel = _PositionSelector(child, slice_fn)

    sel.set_pos(1, 1)
    t1 = threading.Thread(target=lambda: sel.delayed_update_data(force=True))
    t1.start()
    assert in_body.wait(timeout=2.0), "first update never entered its body"

    # Fire 2 while fire 1 is blocked. It must NOT wait for fire 1 to finish.
    sel.set_pos(2, 2)
    t2 = threading.Thread(target=lambda: sel.delayed_update_data(force=True))
    t2.start()
    assert second_ran.wait(timeout=1.0), (
        "newer update blocked behind the slow in-flight one — the update path is "
        "holding a lock across the compute and stalls live tracking."
    )
    t2.join(timeout=2.0)
    # Latest position won.
    assert np.array_equal(sel.current_indices, np.array([[2, 2]]))

    # The slow, now-stale first body finishes but must NOT clobber the newer
    # committed indices.
    release.set()
    t1.join(timeout=2.0)
    assert np.array_equal(sel.current_indices, np.array([[2, 2]])), (
        "the stale (superseded) update overwrote the latest position"
    )


def test_cache_critical_section_is_serialized():
    """The cache cancel/get_chunk/submit section of
    update_from_navigation_selection must not run concurrently for one signal.

    Two overlapping updates would mutate the shared CachedDaskArray block lists
    and cancel each other's chunk futures — and a cancelled chunk future kills
    the dependent get_inds/write_shared_array futures (the image freezes on the
    previous frame). This drives two updates at the same signal and asserts the
    critical section is entered one at a time.
    """
    from spyde.drawing import update_functions as uf

    meter = {"active": 0, "max": 0}
    mlock = threading.Lock()
    first = {"flag": True}
    entered = threading.Event()
    release = threading.Event()

    class _Cache:
        def cancel_surrounding(self):
            pass

    class _Signal:
        _lazy = True
        cached_dask_array = _Cache()
        data = np.zeros((4, 4, 8, 8), dtype=np.float32)

        def _get_cache_dask_chunk(self, indices, get_result=False, return_future=False):
            with mlock:
                meter["active"] += 1
                meter["max"] = max(meter["max"], meter["active"])
            try:
                if first["flag"]:
                    first["flag"] = False
                    entered.set()
                    release.wait(timeout=2.0)
                return np.zeros((8, 8), dtype=np.float32)   # numpy → no submit branch
            finally:
                with mlock:
                    meter["active"] -= 1

    sig = _Signal()

    class _PS:
        current_signal = sig

    class _Child:
        plot_state = _PS()
        _pending_shm_future = None

        def update_data(self, d):
            pass

    class _Sel:
        is_integrating = False

    child, sel = _Child(), _Sel()

    def call():
        uf.update_from_navigation_selection(sel, child, np.array([[1, 1]]))

    t1 = threading.Thread(target=call)
    t1.start()
    assert entered.wait(timeout=2.0), "first cache update never ran"

    t2 = threading.Thread(target=call)
    t2.start()
    time.sleep(0.15)   # t2 must block on the cache lock, not enter the section

    peak = meter["max"]
    release.set()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert peak == 1, (
        f"the cache critical section ran {peak}x concurrently — overlapping "
        "navigator updates can cancel a chunk future that the dependent "
        "get_inds/write_shared_array futures need, freezing the image."
    )


def test_stale_body_skips_cache_cancel():
    """A superseded body must NOT cancel_surrounding() — that kills the LATEST
    position's in-flight chunk future ('get_inds … lost dependencies'). When the
    selector reports the body is stale, update_from_navigation_selection returns
    immediately without touching the cache."""
    from spyde.drawing import update_functions as uf

    cancels = {"n": 0}

    class _Cache:
        def cancel_surrounding(self):
            cancels["n"] += 1

    class _Signal:
        _lazy = True
        cached_dask_array = _Cache()
        data = np.zeros((4, 4, 8, 8), dtype=np.float32)

        def _get_cache_dask_chunk(self, indices, **k):
            raise AssertionError("stale body must not request a chunk")

    class _PS:
        current_signal = _Signal()

    class _Child:
        plot_state = _PS()
        _pending_shm_future = None
        def update_data(self, d): pass

    class _StaleSel:
        is_integrating = False
        def is_stale_body(self):           # always stale
            return True

    out = uf.update_from_navigation_selection(_StaleSel(), _Child(),
                                              np.array([[1, 1]]))
    assert out is None, "stale body should return None (skip, no paint)"
    assert cancels["n"] == 0, "stale body cancelled surrounding blocks (kills latest future)"


def test_latest_position_wins_under_burst():
    """After a burst of position changes the child must end on the LAST one."""
    child = _RecorderChild()

    def slow_slice_fn(sel, ch, indices):
        time.sleep(0.02)
        return np.asarray(indices).copy()

    sel = _PositionSelector(child, slow_slice_fn)

    threads = []
    for i in range(1, 6):
        sel.set_pos(i, i)
        sel.delayed_update_data(force=True)
    for t in threads:
        t.join(timeout=2.0)

    time.sleep(0.2)
    assert child.current_data is not None
    last = np.asarray(child.current_data).reshape(-1)[:2]
    assert tuple(last) == (5, 5), (
        f"child ended on {tuple(last)}, not the last position (5, 5)."
    )

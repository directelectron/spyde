"""Tests for the reactive-overlay engine (spyde.drawing.live_overlay)."""
import threading
import time

from spyde.drawing.live_overlay import LiveOverlayEngine


def test_thread_mode_is_offthread_singleflight_latest_wins():
    caller = threading.get_ident()
    meter = {"active": 0, "max": 0}
    mlock = threading.Lock()
    compute_threads = []
    rendered = []

    def compute(iy, ix):
        with mlock:
            meter["active"] += 1
            meter["max"] = max(meter["max"], meter["active"])
        compute_threads.append(threading.get_ident())
        time.sleep(0.02)
        with mlock:
            meter["active"] -= 1
        return (iy, ix)

    def render(payload):
        rendered.append(payload)

    eng = LiveOverlayEngine(compute, render, mode="thread", name="t")
    try:
        t0 = time.time()
        for i in range(5):
            eng.request(i, i)        # must NOT block the caller
        assert time.time() - t0 < 0.05, "request() blocked the caller thread"

        # Wait for the worker to drain.
        for _ in range(50):
            if rendered and rendered[-1] == (4, 4):
                break
            time.sleep(0.02)
    finally:
        eng.stop()

    assert meter["max"] == 1, "compute ran concurrently (not single-flight)"
    assert compute_threads and all(t != caller for t in compute_threads), \
        "compute ran on the caller thread, not the worker"
    assert rendered[-1] == (4, 4), "latest position did not win"
    assert len(rendered) < 5, "intermediate positions were not coalesced"


def test_sync_mode_runs_inline_on_caller():
    caller = threading.get_ident()
    seen = []
    eng = LiveOverlayEngine(
        lambda iy, ix: (iy, ix, threading.get_ident()),
        lambda p: seen.append(p),
        mode="sync", name="s",
    )
    eng.request(2, 3)
    assert seen and seen[0][:2] == (2, 3)
    assert seen[0][2] == caller, "sync mode must run on the caller thread"


def test_compute_error_does_not_propagate():
    rendered = []

    def boom(iy, ix):
        raise ValueError("compute failed")

    eng = LiveOverlayEngine(boom, lambda p: rendered.append(p),
                            mode="sync", name="e")
    eng.request(0, 0)               # must not raise
    assert rendered == []           # render skipped on compute error


def test_stop_is_idempotent_and_halts_worker():
    eng = LiveOverlayEngine(lambda iy, ix: (iy, ix), lambda p: None,
                            mode="thread", name="x")
    eng.request(1, 1)
    time.sleep(0.05)
    eng.stop()
    eng.stop()                      # second stop must be harmless

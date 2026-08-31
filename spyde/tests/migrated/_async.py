"""Waiting for the session to catch up — the one place that knows how.

Every test here drives work that finishes somewhere else: on a worker thread,
on the navigator's dispatcher, on the asyncio loop the session marshals
results back onto. Before this module each test file answered "has it finished
yet?" for itself — 33 of them defined their own ``_wait``, in 24 distinct
variants — and each answered it by watching some VISIBLE SIDE EFFECT of the
work rather than the work itself.

That proxy is the bug factory, not the polling. A side effect can be true
before the work is done (``test_a_lazy_scan_fills_in_progressively`` waited for
"every position in the field is finite", which the corner seed makes true
before a single chunk has landed, so it read the progress stream three messages
early) and it can be true because something ELSE set it. Both read as
flakiness, and both are invisible until the machine is slow enough.

So there are two primitives here and they are not interchangeable:

``wait_until``  — poll a condition YOU care about. Use it when the thing you
                  are waiting for IS the assertion.
``quiesce``     — wait for the session to have nothing left to do. Use it when
                  you are about to assert on a state that background work could
                  still change, and there is no single condition that says
                  "done" — which is most of the time.
"""
from __future__ import annotations

import concurrent.futures
import threading
import time

#: Generous on purpose: for a wait that SUCCEEDS, the timeout only decides how
#: long a broken test takes to say so, and a tight one turns a slow runner into
#: a flake. Callers that expect a wait to time out — "prove nothing happens" —
#: must pass their own, because there the timeout IS the runtime.
DEFAULT_TIMEOUT = 60.0


def wait_until(pred, timeout: float = DEFAULT_TIMEOUT, poll: float = 0.02) -> bool:
    """Poll *pred* until it is true. Returns whether it became true.

    Returns rather than raises so the call site reads
    ``assert wait_until(...), "what should have happened"`` — the message
    belongs with the assertion, where the reader is.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(poll)
    return pred()


def drain_loop(session, timeout: float = 5.0) -> bool:
    """Wait for everything already queued on the session's loop to have run.

    A round trip, not an inspection: put a callback on the end of the queue and
    wait for it. Everything the loop was already holding runs first, so this
    returns only once those callbacks are DONE — which is the guarantee
    ``inflight_workers() == 0`` deliberately does not give you.
    """
    loop = getattr(session, "_main_loop", None)
    if loop is None:            # no loop registered: callbacks ran inline
        return True
    landed = threading.Event()
    try:
        loop.call_soon_threadsafe(landed.set)
    except RuntimeError:        # loop already closed (teardown)
        return True
    return landed.wait(timeout)


def _outstanding(session) -> str | None:
    """What is still to happen, or ``None`` if nothing is. The string is the
    failure message: "quiesce timed out" alone tells you nothing."""
    from de_shell.actions.lifecycle import inflight_workers
    from spyde.drawing.selectors import base_selector as bs

    n = inflight_workers(session)
    if n:
        return f"{n} worker job(s) still running"
    if not bs._nav_dispatcher.idle():
        return "the navigator dispatcher is still working"
    for _wid, sel in (getattr(session, "_nav_selectors", {}) or {}).items():
        for s in (sel, getattr(sel, "selector", None)):
            if s is not None and getattr(s, "_settle_timer", None) is not None:
                return "a selector settle timer is still armed"
    return None


def quiesce(session, timeout: float = DEFAULT_TIMEOUT) -> bool:
    """Block until the session has nothing left in flight.

    Reaching a fixpoint, not a single sweep: draining the loop RUNS callbacks,
    and a callback's whole job is usually to start more work — repaint, emit,
    dispatch the next pass. A sweep that checked once would return in the gap
    between the work finishing and its continuation being queued. So sweep
    until two consecutive passes find nothing outstanding.

    Returns whether it settled. Pair it with an assertion when you need the
    test to fail loudly rather than assert on a half-finished session::

        assert quiesce(session), "the session never went idle"
    """
    deadline = time.monotonic() + timeout
    clean = 0
    while time.monotonic() < deadline:
        if _outstanding(session) is None and drain_loop(session):
            clean += 1
            if clean >= 2:
                return True
        else:
            clean = 0
        time.sleep(0.005)
    return _outstanding(session) is None


def why_busy(session) -> str:
    """What ``quiesce`` is still waiting for — for a test's failure message."""
    return _outstanding(session) or "nothing (the session is idle)"


def call_on_loop(session, fn, timeout: float = 30.0):
    """Run *fn* on the session's loop thread and return its result.

    Where the app runs its action handlers: a message arrives on stdin and is
    dispatched on the loop. A test calls the same handler directly, so it runs
    on the test thread — usually harmless, and occasionally not.

    ``handlers.harvest_snapshots`` is the case that matters. It arms its
    fallback with ``loop.call_later``, which asyncio only supports from the
    loop's OWN thread; called from another one the timer is queued without
    waking the loop, so an idle loop can sleep straight past it. The 3 s
    fallback every report save relies on then never fires — in the test only.
    Run the handler here and it behaves as it does in the app.
    """
    loop = getattr(session, "_main_loop", None)
    if loop is None:
        return fn()
    done: concurrent.futures.Future = concurrent.futures.Future()

    def _run():
        try:
            done.set_result(fn())
        except BaseException as e:      # noqa: BLE001 - re-raised to the caller
            done.set_exception(e)

    loop.call_soon_threadsafe(_run)
    return done.result(timeout)

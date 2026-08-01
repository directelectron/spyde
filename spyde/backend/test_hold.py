"""test_hold.py — a deterministic pause point for tests that must observe a
PARTIALLY-completed compute.

The problem this exists for. Several e2e specs assert on a transient: the
diffraction pattern filling in while a batch is still running, the navigator
serving an already-computed position mid-compute. Written the obvious way they
POLL for that moment, which means they race the thing they are measuring — and
on a fast runner the batch finishes first, so a correct implementation fails
with "nothing was proven". `progressive_signal_preview.spec.ts` burned four
separate assertion fixes to that pattern before it became clear the design, not
any one assertion, was the fault: part of it needs the batch RUNNING and the
navigator over COMPUTED data at the same instant, and no amount of polling can
make that window exist reliably.

Retries do not help and actively hurt: re-running until it passes hid a real
defect (the drag walked over data that was uncomputed by construction).

So instead of hoping the window exists, the app opens one on request. With
``SPYDE_TEST_HOLD`` set, a compute pauses at a named point once it has passed a
given fraction, and stays paused until the test releases it. The partial state
is then a fact the test can take its time over, not a race.

Usage (e2e)::

    launchApp({ env: { SPYDE_TEST_HOLD: 'fv-batch@0.25' } })
    ...                                  # the batch parks at >=25% and waits
    await backendAction(page, 'test_hold_release', { name: 'fv-batch' })

Format: ``<name>@<where>``, comma-separated for several. ``<where>`` is a
fraction of the total (``0.25``) or an absolute count (``800``). A bare
``<name>`` holds at the first opportunity.

**Production cost is one module-level bool.** ``_SPEC`` is parsed once at
import; when the variable is unset ``hold_point`` returns on its first line and
nothing else in this module ever runs.

**It cannot wedge CI.** Every wait is bounded by ``MAX_HOLD_S``; a forgotten
release costs that long once and the compute continues, rather than hanging the
job until the workflow timeout.
"""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger(__name__)

#: Upper bound on any single hold. A test that forgets to release loses this
#: many seconds ONCE — it does not hang the run.
MAX_HOLD_S = 120.0


def _parse(spec: str) -> dict:
    """``"fv-batch@0.25,other@800"`` → ``{"fv-batch": 0.25, "other": 800.0}``."""
    out: dict = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        name, _, where = part.partition("@")
        name = name.strip()
        if not name:
            continue
        try:
            out[name] = float(where) if where.strip() else 0.0
        except ValueError:
            log.warning("[test-hold] ignoring unparseable spec %r", part)
    return out


_SPEC: dict = _parse(os.environ.get("SPYDE_TEST_HOLD", "") or "")
_events: dict = {}
_held: set = set()
_lock = threading.Lock()


def _event(name: str) -> threading.Event:
    with _lock:
        ev = _events.get(name)
        if ev is None:
            ev = _events[name] = threading.Event()
        return ev


def armed() -> bool:
    """True when any hold is configured — the one check production pays for."""
    return bool(_SPEC)


def hold_point(name: str, done: int, total: int) -> None:
    """Pause *name* once ``done`` has passed its configured threshold.

    Called from whatever thread is driving the compute (for find-vectors, a
    Dask callback thread). Blocking there is the point: the asyncio main thread
    keeps serving the UI, so the partially-filled display stays live and
    interactive while the test inspects it.

    EVERY caller past the threshold parks, not just the first. Holding only the
    first was the obvious design and it does not work: a per-chunk callback
    parks one thread while every later chunk sails through, so the compute runs
    to completion anyway and the "paused" state never exists. Once released the
    event stays set, so all subsequent calls return immediately and the batch
    finishes without stuttering.
    """
    if not _SPEC:
        return                                  # production: nothing to do
    where = _SPEC.get(name)
    if where is None:
        return
    ev = _event(name)
    if ev.is_set():
        return                                  # already released — sail through
    # A fraction (<= 1) is of the total; anything larger is an absolute count.
    threshold = (where * total) if 0.0 < where <= 1.0 else where
    if done < threshold:
        return
    first = False
    with _lock:
        if name not in _held:
            _held.add(name)
            first = True
    if first:                                   # log once, not per chunk
        log.info("[test-hold] %s parked at %d/%d (threshold %s) — waiting for "
                 "release", name, done, total, where)
    released = ev.wait(MAX_HOLD_S)
    if first:
        log.info("[test-hold] %s %s", name,
                 "released" if released else f"timed out after {MAX_HOLD_S:.0f}s")


def release(name: str) -> bool:
    """Let a parked hold continue. Returns True if that name was configured."""
    if name not in _SPEC:
        log.info("[test-hold] release(%r) — no such hold configured", name)
        return False
    _event(name).set()
    log.info("[test-hold] %s released", name)
    return True


def reset() -> None:
    """Forget every hold's state (unit tests only)."""
    with _lock:
        _events.clear()
        _held.clear()

"""
action_profile.py — stage timing for USER-VISIBLE ACTIONS, end to end.

The navigator already has this (``NavProfile`` in ``drawing/update_functions.py``,
``SPYDE_NAV_PROFILE=1`` → one ``[NAV-PROFILE]`` line per frame). That facility
exists because "the navigator is slow" is unanswerable without a per-stage
breakdown. Exactly the same is true of an action: "the drift check takes 60
seconds" names a symptom and indicts nothing, and the stage that actually costs
the 60 s is rarely the one anybody guesses.

This is the same idea one level up. One ``[ACTION-PROFILE]`` line per action
invocation, breaking it into the stages a user would recognise::

    [ACTION-PROFILE] drift_open total=61240.3ms  queued=12.1  frames=3.4
        sum_read=60980.2 window=190.4 preview_spawn=1.1  n_frames=240
        frame=2048x2048 sum_frames=64

**Read the line, then fix what it indicts.** That is the whole point: the project
rule is to time the arithmetic before reasoning about caches or async, and a
guessed bottleneck has been wrong here before.

Two deliberate choices:

* **It logs; it does not emit.** Backend ``emit()`` goes down the ``PLOTAPP:``
  line protocol, which the Electron main process consumes and echoes only for a
  tiny allowlist — a Playwright spec waiting on a profile MESSAGE would wait
  forever. An INFO log record reaches stderr (which the e2e harness captures),
  the Log panel, and a terminal, all at once. One channel, three readers.
* **It is off unless asked.** Gated on ``SPYDE_ACTION_PROFILE=1`` or the live
  ``action_profile`` debug flag, and every method short-circuits on a single
  boolean, so an un-profiled run pays one attribute lookup per stage.

The stage names are part of the contract — ``tests/action_latency.spec.ts``
asserts BUDGETS against them, so a regression is caught by CI instead of by
somebody sitting through a minute of dead UI. Renaming a stage breaks that
assertion, which is the intended amount of friction.
"""
from __future__ import annotations

import contextlib
import logging
import time

from spyde.backend.debug_flags import action_profile_on as _on

log = logging.getLogger(__name__)


class ActionProfile:
    """Accumulate per-stage timings for ONE action and log a single line.

    Usage::

        prof = ActionProfile("drift_open", payload=payload)
        with prof.stage("frames"):
            n, get_frame, shape = wiz.frames()
        with prof.stage("sum_read"):
            raw = _stack_sum(...)
        prof.info(n_frames=n)
        prof.done()

    Stages may be timed on different threads (an action typically hops
    dispatch → worker → main); the list append is all that crosses, and one
    profile object belongs to one invocation.
    """

    __slots__ = ("_on", "_label", "_stages", "_t0", "_info", "_queued")

    def __init__(self, label: str, payload: dict | None = None) -> None:
        self._on = _on()
        self._label = label
        self._stages: list[tuple[str, float]] = []
        self._info: dict = {}
        self._t0 = time.perf_counter() if self._on else 0.0
        self._queued = None
        if self._on and payload:
            # The renderer stamps `_t_click` (epoch ms) when the user actually
            # clicks. Everything before the handler runs — IPC hop, the backend's
            # asyncio queue, any action ahead of it — lands in this one number,
            # and it is the difference between "the compute is slow" and "the
            # button was dead for two seconds before anything started".
            t_click = payload.get("_t_click")
            if isinstance(t_click, (int, float)) and t_click > 0:
                self._queued = max(0.0, time.time() - float(t_click) / 1000.0)

    def stage(self, name: str):
        """Context manager timing one stage; a nullcontext when profiling is off."""
        if not self._on:
            return contextlib.nullcontext()
        return _Stage(self, name)

    def mark(self, name: str) -> None:
        """Record a point in time as an elapsed-since-start marker.

        For things that are not a bracketed span — "the first pixels reached the
        renderer" is an instant, not a duration, and it is the number the user
        actually experiences.
        """
        if self._on:
            self._stages.append((name + "@", time.perf_counter() - self._t0))

    def info(self, **kw) -> None:
        """Attach context that explains the timings (frame count, shape, …)."""
        if self._on:
            self._info.update(kw)

    def _record(self, name: str, dt: float) -> None:
        self._stages.append((name, dt))

    @property
    def enabled(self) -> bool:
        return self._on

    def done(self, extra: str = "") -> None:
        if not self._on:
            return
        total = (time.perf_counter() - self._t0) * 1e3
        parts = "  ".join(f"{n}={dt * 1e3:.1f}" for n, dt in self._stages)
        q = f"queued={self._queued * 1e3:.1f}  " if self._queued is not None else ""
        ctx = "  ".join(f"{k}={v}" for k, v in self._info.items())
        log.info("[ACTION-PROFILE] %s total=%.1fms  %s%s%s%s",
                 self._label, total, q, parts,
                 ("  " + ctx) if ctx else "", ("  " + extra) if extra else "")


class _Stage:
    __slots__ = ("_prof", "_name", "_t")

    def __init__(self, prof: ActionProfile, name: str) -> None:
        self._prof = prof
        self._name = name
        self._t = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._prof._record(self._name, time.perf_counter() - self._t)
        return False

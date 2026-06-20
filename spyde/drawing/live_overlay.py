"""
live_overlay.py — a reactive overlay engine.

The established workflow for "a function applied to an overlay that re-runs
whenever the underlying navigator data changes": bind a
``compute(iy, ix) -> payload`` to the navigator and a ``render(payload)`` to the
markers. Whenever the navigator position changes, ``compute`` runs **off the
navigator thread**, single-flight and latest-wins (intermediate positions are
coalesced), and ``render`` draws the result.

Why off-thread: the navigator update path is serialised (one update in flight,
so the image future isn't raced — see ``base_selector`` / ``test_navigator_race``).
If a slow overlay computed inline in that path it would HOLD that lock and gate
the image at the overlay's rate. Running it here keeps the image at full async
rate while the overlay tracks a beat behind.

Execution engines (``mode``):

* ``"thread"`` (default for heavy overlays) — a daemon worker runs ``compute``
  in-process on the latest position. Fast; for peak/pose overlays that slice a
  small in-memory / already-resident window. Naturally coalesces: the worker
  always recomputes the most recent position, dropping intermediates.
* ``"future"`` — submit ``compute`` to a Dask client and render on completion.
  Cleaner and consistent with the image future→shm path (and the way batch find
  uses ``map_overlap``); a worker round-trip slower. The previous in-flight
  future is cancelled (greedy) and superseded results are dropped (latest wins).
  ``compute`` must be picklable and slice small data on the worker.
* ``"sync"`` — run ``compute`` + ``render`` inline on the caller. The legacy
  behaviour, for cheap overlays (e.g. a CSR vector lookup) where async adds
  nothing and an immediate paint is wanted.

``render`` may run on a worker thread — that's fine for anyplotlib marker groups
(``_push`` is GIL-protected). Keep ``render`` cheap and self-guarding.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class LiveOverlayEngine:
    """Re-run ``compute(iy, ix)`` off the navigator thread on each position
    change and ``render`` the result, single-flight + latest-wins."""

    def __init__(
        self,
        compute: Callable[[int, int], object],
        render: Callable[[object], None],
        *,
        mode: str = "thread",
        client=None,
        name: str = "overlay",
    ) -> None:
        self._compute = compute
        self._render = render
        self._mode = mode if mode in ("thread", "future", "sync") else "thread"
        self._client = client
        self._name = name

        self._lock = threading.Lock()
        self._latest: tuple[int, int] | None = None
        self._gen = 0                       # request generation (latest-wins)
        self._wake = threading.Event()
        self._alive = False
        self._thread: threading.Thread | None = None
        self._pending_future = None

    # ── public API ────────────────────────────────────────────────────────────
    def request(self, iy: int, ix: int) -> None:
        """Note a new navigator position and (re)schedule the overlay compute."""
        with self._lock:
            self._latest = (int(iy), int(ix))
            self._gen += 1
            gen = self._gen
            pos = self._latest
        if self._mode == "sync":
            self._compute_and_render(pos)
        elif self._mode == "future":
            self._submit_future(gen, pos)
        else:
            self._ensure_thread()
            self._wake.set()

    def stop(self) -> None:
        """Tear down the worker / cancel any in-flight future (call on remove)."""
        self._alive = False
        self._wake.set()
        fut = self._pending_future
        if fut is not None and not fut.done():
            try:
                fut.cancel()
            except Exception as e:
                logger.debug("%s: cancelling overlay future on stop failed: %s",
                             self._name, e)

    # ── thread engine ─────────────────────────────────────────────────────────
    def _ensure_thread(self) -> None:
        with self._lock:
            if self._alive:
                return
            self._alive = True
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name=f"{self._name}-live")
            self._thread.start()

    def _loop(self) -> None:
        while self._alive:
            self._wake.wait()
            self._wake.clear()
            if not self._alive:
                break
            with self._lock:
                pos = self._latest
            if pos is None:
                continue
            # Always recompute/render the LATEST position; positions that arrived
            # between iterations are coalesced away (we never queue a backlog).
            self._compute_and_render(pos)

    # ── future engine ─────────────────────────────────────────────────────────
    def _submit_future(self, gen: int, pos: tuple[int, int]) -> None:
        client = self._client
        if client is None:
            self._compute_and_render(pos)         # no cluster → inline fallback
            return
        prev = self._pending_future               # greedy: drop the stale one
        if prev is not None and not prev.done():
            try:
                prev.cancel()
            except Exception as e:
                logger.debug("%s: cancelling prior overlay future failed: %s",
                             self._name, e)
        try:
            fut = client.submit(self._compute, pos[0], pos[1], pure=False)
        except Exception as e:
            logger.debug("%s: overlay future submit failed (%s); inline", self._name, e)
            self._compute_and_render(pos)
            return
        self._pending_future = fut

        def _done(f, gen=gen):
            with self._lock:
                if gen != self._gen:              # superseded → drop (latest wins)
                    return
            try:
                payload = f.result()
            except Exception as e:
                logger.debug("%s: overlay future failed: %s", self._name, e)
                return
            self._safe_render(payload)

        fut.add_done_callback(_done)

    # ── shared ────────────────────────────────────────────────────────────────
    def _compute_and_render(self, pos: tuple[int, int]) -> None:
        try:
            payload = self._compute(*pos)
        except Exception as e:
            logger.debug("%s: overlay compute failed: %s", self._name, e)
            return
        self._safe_render(payload)

    def _safe_render(self, payload) -> None:
        try:
            self._render(payload)
        except Exception as e:
            logger.debug("%s: overlay render failed: %s", self._name, e)

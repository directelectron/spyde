"""
stream.py — refreshing a figure from work that finishes off the main thread.

Every shell app produces frames somewhere other than the asyncio main thread: a
camera thread, an acquisition runner, a worker computing a result, a background
fill writing into a buffer. All of them need the same three things, and all
three had been written separately in each app:

1. **Marshal to the main thread.** Figures may only be touched there.
2. **Newest wins, with ONE scheduled paint.** A producer faster than the
   renderer must not queue a callback per frame — the main thread would fall
   further and further behind while the queue grew, showing ever-staler frames.
   A single pending slot means a superseded frame is dropped, not backlogged.
3. **Count what was dropped.** Dropping is correct here, but invisible dropping
   is how "the display is laggy" becomes unfalsifiable.

There is deliberately **no dask**. A future is any object with
``add_done_callback`` / ``result`` / ``cancel`` — ``concurrent.futures.Future``
from ``de_shell.compute.ThreadCompute`` is the common case, and SpyDE's
distributed adapter satisfies the same protocol without this module knowing.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


class FrameStream:
    """Newest-wins painting into a :class:`~de_shell.plotting.figure.FigureView`.

    Parameters
    ----------
    view
        The figure to paint. Anything with ``show(frame, clim=…) -> bool``.
    dispatch
        Schedules a callable on the main thread — ``SessionBase._dispatch_to_main``.
    on_painted
        Optional ``fn(frame)`` run on the MAIN thread after a successful paint,
        for whatever the app hangs off a new frame (stats, counters, a status
        line).
    on_error
        Optional ``fn(exc)`` for a paint or future that failed. Runs on the main
        thread. Without one, failures are logged and swallowed — a display error
        must never kill the producer.
    """

    def __init__(self, view, dispatch: Callable[[Callable[[], None]], None], *,
                 on_painted: Callable[[np.ndarray], None] | None = None,
                 on_error: Callable[[Exception], None] | None = None) -> None:
        self._view = view
        self._dispatch = dispatch
        self._on_painted = on_painted
        self._on_error = on_error

        self._lock = threading.Lock()
        self._pending: np.ndarray | None = None
        self._pending_clim: tuple[float, float] | None = None
        self._scheduled = False
        self._closed = False
        #: The future whose result we are still interested in. A newer submission
        #: supersedes it; its callback then no-ops on this identity check.
        self._future: Any = None

        self.shown = 0
        self.dropped = 0

    # ── Submitting ────────────────────────────────────────────────────────────

    def submit(self, frame: np.ndarray, *,
               clim: tuple[float, float] | None = None) -> None:
        """Offer a frame from ANY thread. The newest one wins."""
        if self._closed:
            return
        with self._lock:
            if self._pending is not None:
                self.dropped += 1
            self._pending = frame
            self._pending_clim = clim
            if self._scheduled:
                return          # a paint is already on its way; it will take this
            self._scheduled = True
        self._dispatch(self._paint_pending)

    def submit_future(self, future, *,
                      clim: tuple[float, float] | None = None) -> None:
        """Paint whatever *future* resolves to, when it resolves.

        Supersedes any future still outstanding: the older one is cancelled, and
        if it was already running its callback no-ops on an identity check
        rather than painting a frame the user has moved past. That check is the
        whole latest-wins guarantee for async work — a queued future cancels
        cleanly, an in-flight one cannot be stopped and must instead be ignored.
        """
        if self._closed:
            return
        with self._lock:
            prev, self._future = self._future, future
        if prev is not None and prev is not future:
            try:
                prev.cancel()
            except Exception as e:
                log.debug("cancelling superseded frame future failed: %s", e)

        def _done(fut) -> None:
            with self._lock:
                if self._future is not fut:
                    return      # superseded while running — drop it
                self._future = None
            try:
                if fut.cancelled():
                    return
                result = fut.result()
            except Exception as e:
                log.debug("frame future failed: %s", e)
                self._fail(e)
                return
            if result is not None:
                self.submit(np.asarray(result), clim=clim)

        try:
            future.add_done_callback(_done)
        except Exception as e:
            log.debug("attaching frame-future callback failed: %s", e)
            self._fail(e)

    # ── Painting (main thread) ────────────────────────────────────────────────

    def _paint_pending(self) -> None:
        with self._lock:
            frame, self._pending = self._pending, None
            clim, self._pending_clim = self._pending_clim, None
            self._scheduled = False
        if frame is None or self._closed:
            return
        try:
            if self._view.show(frame, clim=clim):
                self.shown += 1
                if self._on_painted is not None:
                    self._on_painted(frame)
        except Exception as e:
            log.exception("painting frame failed")
            self._fail(e)

    def _fail(self, exc: Exception) -> None:
        if self._on_error is None:
            return
        # Errors surface on the MAIN thread like paints do, so a handler can
        # touch UI without each caller having to remember to marshal.
        self._dispatch(lambda: self._on_error(exc))

    # ── Teardown ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop accepting and painting frames. Idempotent.

        Cancels any outstanding future and drops the pending frame, so a
        producer still winding down cannot paint into a closed window.
        """
        with self._lock:
            self._closed = True
            self._pending = None
            fut, self._future = self._future, None
        if fut is not None:
            try:
                fut.cancel()
            except Exception as e:
                log.debug("cancelling frame future on close failed: %s", e)

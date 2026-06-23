from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, Set, Dict

import numpy as np
from psygnal import Signal
from dask.distributed import Future

from spyde.drawing.update_functions import read_shared_array

log = logging.getLogger(__name__)


class PlotUpdateWorker:
    """
    Worker that periodically scans plots for completed Dask Futures and emits results.
    Runs in a daemon thread; callers connect to signals to receive results.

    Thread safety note: emit() is called from the background thread.  Slots that
    touch the IPC layer should be thread-safe (e.g. use a queue or asyncio
    run_coroutine_threadsafe); slots that mutate anyplotlib figures are fine
    because anyplotlib's _push() is GIL-protected.
    """

    plot_ready = Signal(object, object, object)   # (plot, result, future)
    signal_ready = Signal(object, object, object) # (signal, result, plot)
    debug_print = Signal(str)

    def __init__(
        self,
        get_plots_callable: Callable[[], list],
        interval_ms: int = 2,
        dispatch: "Callable[[Callable], None] | None" = None,
    ) -> None:
        self._get_plots = get_plots_callable
        self._interval = interval_ms / 1000.0
        # Dedup set of (future.key, id(plot)) already emitted — see _maybe_emit_future.
        self._seen: Set[tuple] = set()
        self._running = False
        self._thread: threading.Thread | None = None
        import os
        self._emit_timing = os.environ.get("SPYDE_NAV_TIMING") == "1"
        # Marshal the result-application onto the MAIN thread. The poll thread only
        # detects "future done" and reads the (already-computed) shm/result; the
        # actual plot.update()/set_data()/push MUST run on the main thread, exactly
        # like the Qt app marshaled plot_ready via a queued signal/slot. Without
        # this the push happens on the poll thread and races the main loop's own
        # figure pushes. `dispatch(fn)` schedules fn() on the main thread (e.g.
        # loop.call_soon_threadsafe); None → run inline (tests / no loop).
        self._dispatch = dispatch

    def start(self) -> None:
        """Start the polling loop in a daemon thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="plot-update-worker"
        )
        self._thread.start()

    def stop(self) -> None:
        """Request the worker to stop. Returns immediately; thread exits on next tick."""
        self._running = False
        self._seen.clear()

    def _loop(self) -> None:
        while self._running:
            try:
                self._check()
            except Exception as e:
                log.debug("plot-update worker tick failed: %s", e)
            time.sleep(self._interval)

    def _check(self) -> None:
        try:
            plots = self._get_plots() or []
        except Exception:
            plots = []

        for plot in plots:
            self._maybe_emit_plot_ready(plot)
            self._maybe_emit_signal_ready(plot)

    def _maybe_emit_plot_ready(self, plot) -> None:
        try:
            fut = getattr(plot, "current_data", None)
            self._maybe_emit_future(fut, self.plot_ready.emit, plot)
        except Exception as e:
            log.debug("emit plot_ready failed: %s", e)

    def _maybe_emit_signal_ready(self, plot) -> None:
        try:
            sig = getattr(plot.plot_state, "current_signal", None)
            if sig is None:
                return
            fut = self._future_from_signal(sig)
            self._maybe_emit_future(fut, self.signal_ready.emit, sig, plot)
        except Exception as e:
            log.debug("emit signal_ready failed: %s", e)

    def _future_from_signal(self, sig) -> Optional[Future]:
        data = getattr(sig, "data", None)
        if isinstance(data, Future):
            return data
        # NB: `and data` on an ndarray with >1 element raises "truth value of an
        # array is ambiguous" — check length explicitly. A lazy signal whose
        # data is a length-1 object array holding a Future is the case we want
        # (a plain numeric ndarray has data[0] that isn't a Future → None).
        if isinstance(data, (list, tuple, np.ndarray)) and len(data) > 0:
            try:
                first = data[0]
            except Exception:
                return None
            if isinstance(first, Future):
                return data[0]
        return None

    def _maybe_emit_future(
        self,
        fut: Optional[Future],
        emitter: Callable,
        plot=None,
        extra=None,
    ) -> None:
        if not isinstance(fut, Future) or not fut.done():
            return
        # NB: do NOT skip cancelled futures here. The QT worker didn't, and the
        # shared-memory buffer is read defensively (read_shared_array rejects an
        # empty/torn header) — a cancelled future whose shm was already written
        # still yields a valid frame, and one whose shm wasn't yields a harmless
        # ValueError that _on_plot_ready drops. Skipping ALL cancelled futures
        # silently dropped EVERY distributed DP frame (hyperspy's cache GC-cancels
        # the write's get_inds dependency), so the diffraction pattern never
        # painted on the distributed path.
        # Dedup by the future's DASK KEY (unique per submission), not id(fut).
        # id() reuses freed addresses: under a fast crosshair drag, futures are
        # created and GC'd rapidly, so a BRAND-NEW write_shared_array future can
        # be handed the same id() as an already-"seen" one → it was wrongly
        # skipped and never emitted → the diffraction pattern froze mid-drag while
        # the nav-update side kept logging fast timings. The key is content-stable
        # and collision-free. Track per (key, plot) so the same chunk re-displayed
        # on two plots still emits for each.
        try:
            fkey = fut.key
        except Exception:
            fkey = id(fut)
        seen_key = (fkey, id(plot))
        if seen_key in self._seen:
            return
        self._seen.add(seen_key)
        # Bound the dedup set so a long session doesn't grow it without limit
        # (keys are unique per submission, so it would otherwise grow forever).
        if len(self._seen) > 4096:
            self._seen.clear()
        try:
            if "write_shared_array" in fut.key and plot is not None:
                start = time.perf_counter()
                # Read the plot's single shared-memory buffer (the frame this
                # future wrote). A frame clobbered by a newer overlapping write is
                # dropped by the latest-wins staleness guard in _on_plot_ready, so
                # reading the single buffer is correct.
                result = read_shared_array(plot.shared_memory)
                _ms = (time.perf_counter() - start) * 1e3
            else:
                start = time.perf_counter()
                result = fut.result()
                _ms = (time.perf_counter() - start) * 1e3
            # Per-frame timing is a hot path during a drag: emitting a log line
            # for EVERY painted frame floods the stdout IPC pipe (shared with the
            # figure pushes) and itself adds lag. Off unless SPYDE_NAV_TIMING=1.
            if self._emit_timing:
                self.debug_print.emit(
                    f"NAV-DEBUG worker: delivered {fut.key} in {_ms:.2f} ms"
                )
        except Exception as e:
            result = e
        # signal_ready passes the owning plot as `extra` → emit (signal, result,
        # plot). plot_ready has no extra → emit (plot, result, future). The old
        # code ignored `extra` and always passed the future as the 3rd arg, so
        # `_on_signal_ready` got a Future where it expected the plot
        # ("'Future' object has no attribute 'parent_selector'").
        # The shm read / fut.result() above already happened (fast, off-thread).
        # Marshal only the APPLY (plot.update → set_data → push) onto the main
        # thread so it doesn't race the main loop's own figure pushes.
        if extra is not None:
            call = lambda: emitter(plot, result, extra)
        else:
            call = lambda: emitter(plot, result, fut)
        if self._dispatch is not None:
            self._dispatch(call)
        else:
            call()

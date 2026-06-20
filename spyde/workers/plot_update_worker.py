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
    ) -> None:
        self._seen_plots: Dict[int, int] = {}
        self._get_plots = get_plots_callable
        self._interval = interval_ms / 1000.0
        self._seen: Set[int] = set()
        self._running = False
        self._thread: threading.Thread | None = None

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
        if isinstance(data, (list, tuple, np.ndarray)) and data and isinstance(data[0], Future):
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
        fid = id(fut)
        if fid in self._seen and self._seen_plots.get(fid) == id(plot):
            return
        self._seen.add(fid)
        self._seen_plots[fid] = id(plot)
        try:
            self.debug_print.emit(f"Emitting Future {fut.key} for plot: {plot}")
            if "write_shared_array" in fut.key and plot is not None:
                start = time.time()
                result = read_shared_array(plot.shared_memory)
                self.debug_print.emit(f"Read shared array in {(time.time()-start)*1000:.2f} ms")
            else:
                start = time.time()
                result = fut.result()
                self.debug_print.emit(f"Transferred Future over TCP in {(time.time()-start)*1000:.2f} ms")
        except Exception as e:
            result = e
        # signal_ready passes the owning plot as `extra` → emit (signal, result,
        # plot). plot_ready has no extra → emit (plot, result, future). The old
        # code ignored `extra` and always passed the future as the 3rd arg, so
        # `_on_signal_ready` got a Future where it expected the plot
        # ("'Future' object has no attribute 'parent_selector'").
        if extra is not None:
            emitter(plot, result, extra)
        else:
            emitter(plot, result, fut)

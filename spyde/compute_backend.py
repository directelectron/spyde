"""
compute_backend.py

Uniform compute abstraction over dask.distributed.Client (distributed mode)
and concurrent.futures.ThreadPoolExecutor (threaded mode, default).

Both modes return concurrent.futures.Future objects so callers are identical
regardless of backend.  The distributed backend wraps dask Futures with a thin
adapter so the same interface works.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Callable, Any, Iterable

import numpy as np
import dask
import dask.array as da

log = logging.getLogger(__name__)


class _DistributedFutureAdapter:
    """
    Wraps a dask.distributed.Future to look like concurrent.futures.Future.

    Only the subset used by PlotUpdateWorker and update_functions is implemented:
    .done(), .result(), .add_done_callback(), .cancel().
    """

    def __init__(self, dask_future):
        self._f = dask_future

    def done(self) -> bool:
        return self._f.done()

    def result(self, timeout=None):
        return self._f.result()

    def cancel(self):
        try:
            self._f.cancel()
        except Exception as e:
            log.debug("cancelling distributed future failed: %s", e)

    def add_done_callback(self, fn: Callable) -> None:
        # dask callbacks receive the dask future; wrap so fn gets this adapter
        def _cb(dask_f):
            fn(self)
        self._f.add_done_callback(_cb)

    # Keep a reference to the underlying dask future for callers that need it
    @property
    def dask_future(self):
        return self._f


class _SyncFuture:
    """Immediately-resolved future for already-computed results."""

    def __init__(self, result):
        self._result = result
        self._callbacks: list[Callable] = []

    def done(self) -> bool:
        return True

    def result(self, timeout=None):
        return self._result

    def cancel(self):
        pass

    def add_done_callback(self, fn: Callable) -> None:
        fn(self)


class ComputeBackend:
    """
    Uniform interface for submitting dask work.

    Parameters
    ----------
    executor : ThreadPoolExecutor | None
        When provided, use threaded mode.  When None, use distributed mode.
    client : dask.distributed.Client | None
        Used when executor is None.
    """

    def __init__(
        self,
        executor: concurrent.futures.ThreadPoolExecutor | None = None,
        client=None,
    ):
        if executor is None and client is None:
            raise ValueError("Provide either executor or client")
        self._executor = executor
        self._client = client
        self._lock = threading.Lock()

    @property
    def client(self):
        """Underlying dask.distributed.Client, or None in threaded mode."""
        return self._client

    @property
    def executor(self):
        """Underlying ThreadPoolExecutor, or None in distributed mode."""
        return self._executor

    @property
    def is_distributed(self) -> bool:
        return self._client is not None

    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a callable, return a concurrent.futures.Future."""
        if self._executor is not None:
            return self._executor.submit(fn, *args, **kwargs)
        else:
            dask_fut = self._client.submit(fn, *args, **kwargs)
            return _DistributedFutureAdapter(dask_fut)

    def submit_graph(self, lazy_array: "da.Array") -> concurrent.futures.Future:
        """Compute a single lazy dask slice and return a **cancellable** Future.

        This is the low-latency, async, cancellable read the movie navigator
        needs — WITHOUT the distributed scheduler round-trip. In threaded mode it
        submits ``lazy_array.compute(scheduler="synchronous")`` to OUR
        ThreadPoolExecutor: the pool provides the concurrent.futures.Future
        (``.cancel()`` a superseded scrub frame, ``.add_done_callback()`` to
        paint off-thread), while the ``synchronous`` scheduler walks the dask
        graph on that one worker thread — so dask does NOT spawn a nested thread
        pool under ours (which would contend). A queued future cancels cleanly
        (latest-position-wins); an in-flight one runs to completion.

        Because the input is a plain lazy dask array, the SAME call reads the
        original movie, a lazy CROP (``s.inav[..].isig[..]``), a rebinned view,
        or a ``.zspy`` — cropping/rebinning stay pure graph ops and scrub through
        this one path. Result is a materialised ``np.ndarray``.

        In distributed mode it falls back to ``client.compute`` (already async +
        cancellable via the adapter) so callers are backend-agnostic.
        """
        if self._executor is not None:
            def _read(a=lazy_array):
                return np.asarray(a.compute(scheduler="synchronous"))
            return self._executor.submit(_read)
        dask_fut = self._client.compute(lazy_array)
        return _DistributedFutureAdapter(dask_fut)

    def compute(self, dask_array_or_list) -> concurrent.futures.Future:
        """
        Trigger async computation of a dask array (or list of arrays).

        Returns a concurrent.futures.Future resolving to the computed result(s).
        """
        if self._executor is not None:
            if isinstance(dask_array_or_list, (list, tuple)):
                arrays = list(dask_array_or_list)
                return self._executor.submit(dask.compute, *arrays, scheduler="threads")
            else:
                return self._executor.submit(
                    dask_array_or_list.compute, scheduler="threads"
                )
        else:
            dask_fut = self._client.compute(dask_array_or_list)
            if isinstance(dask_fut, (list, tuple)):
                # wrap list — pick first future as representative, attach callback
                # for multi-array case return a combined future
                combined = self._client.submit(lambda: [f.result() for f in dask_fut])
                return _DistributedFutureAdapter(combined)
            return _DistributedFutureAdapter(dask_fut)

    def compute_chunks_progressive(
        self,
        result_array: da.Array,
        nav_ndim: int,
        on_chunk_done: Callable | None,
    ) -> concurrent.futures.Future:
        """
        Submit per-nav-chunk computations; call on_chunk_done(chunk, slices)
        from a worker thread as each chunk finishes.

        Returns a Future that resolves to the full result array.
        """
        import itertools

        nav_chunks = result_array.chunks[:nav_ndim]

        axes_ranges = []
        for axis_chunks in nav_chunks:
            positions, start = [], 0
            for size in axis_chunks:
                positions.append((start, size))
                start += size
            axes_ranges.append(positions)

        chunk_futures = []
        chunk_slices = []
        for combo in itertools.product(*axes_ranges):
            slices = tuple(slice(s, s + n) for s, n in combo)
            full_slice = slices + (slice(None),) * (result_array.ndim - nav_ndim)
            chunk_da = result_array[full_slice]

            if self._executor is not None:
                fut = self._executor.submit(chunk_da.compute, scheduler="threads")
            else:
                dask_fut = self._client.compute(chunk_da)
                fut = _DistributedFutureAdapter(dask_fut)

            if on_chunk_done is not None:
                _slices = slices  # capture
                def _make_cb(nav_slices):
                    def _cb(f):
                        try:
                            on_chunk_done(f.result(), nav_slices)
                        except Exception as e:
                            # Live-preview callback; the whole-array future
                            # re-raises a genuine chunk error on the commit path.
                            log.debug("chunk callback %r failed: %s", nav_slices, e)
                    return _cb
                fut.add_done_callback(_make_cb(_slices))

            chunk_futures.append(fut)
            chunk_slices.append(slices)

        # Full-array future
        if self._executor is not None:
            return self._executor.submit(result_array.compute, scheduler="threads")
        else:
            return _DistributedFutureAdapter(self._client.compute(result_array))

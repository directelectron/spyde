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
        # Dedicated LOCAL pool for interactive nav frame reads. Created lazily so
        # threaded mode (which already has _executor) never pays for it. See
        # submit_graph: a nav read must NEVER go to the distributed cluster.
        self._nav_executor: concurrent.futures.ThreadPoolExecutor | None = None

    def _nav_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """The local pool interactive nav reads run on, built on first use.

        ONE worker, deliberately. ``fut.cancel()`` only takes effect on a QUEUED
        future — an already-running one runs to completion. With N>1 workers,
        several superseded reads run concurrently and complete in arbitrary
        order, so an OLDER frame can land after a newer one and the display jumps
        backwards while you drag. One worker makes the reads serial, so the only
        ordering hazard left is a single in-flight read, which the caller's
        identity check already discards."""
        with self._lock:
            if self._nav_executor is None:
                self._nav_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="nav-read")
            return self._nav_executor

    def shutdown_nav_pool(self) -> None:
        """Release the local nav-read pool (Session.shutdown)."""
        with self._lock:
            pool, self._nav_executor = self._nav_executor, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

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

        **This NEVER goes to the distributed cluster, even in distributed mode.**
        A nav read is a few MB of LOCAL file that the GUI process can read itself;
        routing it through ``client.compute`` adds graph serialization, worker
        dispatch and a result transfer to a read whose whole job is to be
        interactive. Measured in the real app on a .zspy 4D-STEM: an integrating
        region took 87-441 ms (once 3731 ms) via the cluster, while the same read
        served locally through the array cache is single-digit ms. The cluster is
        for BATCH compute (find-vectors, orientation, VI) where the work per byte
        is large; a frame read is the opposite shape.

        So both modes submit to a LOCAL ThreadPoolExecutor running
        ``compute(scheduler="synchronous")``: the pool provides the
        concurrent.futures.Future (``.cancel()`` a superseded scrub frame,
        ``.add_done_callback()`` to paint off-thread) while the ``synchronous``
        scheduler walks the graph on that one worker thread, so dask does NOT
        spawn a nested pool under ours.
        """
        def _read(a=lazy_array):
            return np.asarray(a.compute(scheduler="synchronous"))
        pool = self._executor if self._executor is not None else self._nav_pool()
        return pool.submit(_read)

    def submit_nav_read(self, fn) -> concurrent.futures.Future:
        """Run ``fn`` (a no-arg callable returning an ndarray) on the LOCAL nav
        pool — the same never-distributed, cancellable path as submit_graph.

        Exists so the expensive tier can run the CACHED read off the dispatcher.
        Async used to mean "compute a dask graph", which bypassed the array cache,
        so a region routed async could never warm and re-read everything on every
        drag step. This lets async and cached coexist."""
        pool = self._executor if self._executor is not None else self._nav_pool()
        return pool.submit(fn)

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
        stopped_flag: "list | None" = None,
    ) -> concurrent.futures.Future:
        """
        Submit per-nav-chunk computations; call on_chunk_done(chunk, slices)
        from a worker thread as each chunk finishes.

        Returns a Future that resolves to the full result array — ASSEMBLED
        from the chunks, not recomputed as a second whole-array graph.

        *stopped_flag* is the codebase's cancellation token: a 1-element
        ``[False]`` list, registered on the tree via ``register_cancel(flag=…)``
        and set to ``True`` to stop the pass. **A superseded compute is
        CANCELLED, not left to finish and have its result thrown away** — a pass
        over a whole scan is the most expensive thing the app does, and running
        one nobody is waiting for costs the cluster (or the pool) the entire
        dataset. Both modes honour it: distributed hands it to
        ``dispatch_chunks``, threaded checks it before each submit AND inside
        each chunk task, because the pool queues them all up front and most are
        still waiting when a supersede lands.

        Passing ``None`` keeps the old behaviour (nothing can stop the pass), so
        this is only safe for a compute that genuinely cannot be superseded.

        The distributed mode routes through ``compute_dispatch.dispatch_chunks``
        (the one dispatcher: batched submit, bounded in-flight window, stall
        watchdog).  This used to be a ``for combo in itertools.product(...):
        client.compute(chunk)`` loop that submitted every chunk up front, one
        blocking scheduler round trip at a time, and then computed the whole
        array AGAIN — see ``test_chunk_dispatch_guard.py``.
        """
        import itertools

        nav_chunks = result_array.chunks[:nav_ndim]
        trailing = (slice(None),) * (result_array.ndim - nav_ndim)

        def _stopped() -> bool:
            try:
                return stopped_flag is not None and bool(stopped_flag[0])
            except Exception:      # a malformed token must not kill the pass
                return False

        if self._executor is None:
            from spyde.compute_dispatch import dispatch_chunks

            fill = (np.nan if np.issubdtype(result_array.dtype, np.floating)
                    else 0)

            def _assemble(result, slices, chunk):
                result[slices + trailing] = chunk

            def _chunk_done(slices, chunk):
                # dispatch_chunks calls (slices, chunk); this API is (chunk,
                # slices).
                if on_chunk_done is not None:
                    on_chunk_done(chunk, slices)

            out: concurrent.futures.Future = concurrent.futures.Future()

            def _run():
                if not out.set_running_or_notify_cancel():
                    return
                try:
                    out.set_result(dispatch_chunks(
                        self._client, result_array, nav_ndim, [], None,
                        stopped_flag=stopped_flag,
                        assemble=_assemble, fill_value=fill,
                        on_chunk_done=_chunk_done, label="chunks-progressive",
                        lane_default_mode="off",
                    ))
                except Exception as exc:
                    out.set_exception(exc)

            threading.Thread(target=_run, daemon=True,
                             name="chunks-progressive").start()
            return out

        # Threaded mode: no scheduler to round-trip, so the per-chunk submit is
        # a local pool.submit() — cheap, and the pool already bounds concurrency.
        axes_ranges = []
        for axis_chunks in nav_chunks:
            positions, start = [], 0
            for size in axis_chunks:
                positions.append((start, size))
                start += size
            axes_ranges.append(positions)

        def _chunk_work(chunk_da):
            # Checked HERE and not only at submit time: this loop queues every
            # chunk up front, so when a supersede lands most of them are still
            # waiting in the pool. Raising is what makes the pass STOP rather
            # than compute a result nobody reads.
            if _stopped():
                raise concurrent.futures.CancelledError("superseded")
            return chunk_da.compute(scheduler="threads")

        for combo in itertools.product(*axes_ranges):
            if _stopped():
                break
            slices = tuple(slice(s, s + n) for s, n in combo)
            fut = self._executor.submit(_chunk_work,
                                        result_array[slices + trailing])
            if on_chunk_done is not None:
                def _make_cb(nav_slices):
                    def _cb(f):
                        if _stopped() or f.cancelled():
                            return
                        try:
                            on_chunk_done(f.result(), nav_slices)
                        except concurrent.futures.CancelledError:
                            pass
                        except Exception as e:
                            # Live-preview callback; the whole-array future
                            # re-raises a genuine chunk error on the commit path.
                            log.debug("chunk callback %r failed: %s", nav_slices, e)
                    return _cb
                fut.add_done_callback(_make_cb(slices))

        def _assemble_all():
            if _stopped():
                raise concurrent.futures.CancelledError("superseded")
            return result_array.compute(scheduler="threads")

        return self._executor.submit(_assemble_all)

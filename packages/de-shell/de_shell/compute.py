"""
compute.py — the dask-free half of the compute abstraction.

Every shell app needs to run work off the asyncio main thread and marshal the
result back. Only SpyDE needs a distributed cluster behind that. So the shell
owns ``ThreadCompute`` — a ``concurrent.futures``-backed submitter — and SpyDE's
``ComputeBackend`` subclasses it to add the ``dask.distributed`` branch.

The split is load-bearing, not cosmetic: importing this module must never pull in
dask, because de-groundcrew and de-autopilot are live in-memory apps that do not
install it.
"""
from __future__ import annotations

import concurrent.futures
import logging
import threading
from typing import Callable

log = logging.getLogger(__name__)


class SyncFuture:
    """Immediately-resolved future for already-computed results."""

    def __init__(self, result):
        self._result = result

    def done(self) -> bool:
        return True

    def result(self, timeout=None):
        return self._result

    def cancel(self):
        return False

    def add_done_callback(self, fn: Callable) -> None:
        fn(self)


class ThreadCompute:
    """Submit callables to a thread pool, returning ``concurrent.futures.Future``.

    Parameters
    ----------
    executor
        The general-purpose pool. When ``None``, subclasses are expected to
        provide their own routing (SpyDE's distributed mode does) — the base
        class then only offers the dedicated interactive-read pool.
    """

    def __init__(self, executor: concurrent.futures.ThreadPoolExecutor | None = None):
        self._executor = executor
        self._lock = threading.Lock()
        # Dedicated LOCAL pool for interactive frame reads. Created lazily so a
        # plain threaded app (which already has _executor) never pays for it.
        self._nav_executor: concurrent.futures.ThreadPoolExecutor | None = None

    def _nav_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """The local pool interactive reads run on, built on first use.

        ONE worker, deliberately. ``fut.cancel()`` only takes effect on a QUEUED
        future — an already-running one runs to completion. With N>1 workers,
        several superseded reads run concurrently and complete in arbitrary
        order, so an OLDER frame can land after a newer one and the display jumps
        backwards while you drag. One worker makes the reads serial, so the only
        ordering hazard left is a single in-flight read, which the caller's
        identity check already discards.
        """
        with self._lock:
            if self._nav_executor is None:
                self._nav_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="nav-read")
            return self._nav_executor

    def shutdown_nav_pool(self) -> None:
        """Release the local interactive-read pool (Session.shutdown)."""
        with self._lock:
            pool, self._nav_executor = self._nav_executor, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    @property
    def executor(self):
        """Underlying ThreadPoolExecutor, or None."""
        return self._executor

    @property
    def is_distributed(self) -> bool:
        return False

    def submit(self, fn: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit a callable, return a concurrent.futures.Future."""
        if self._executor is None:
            raise RuntimeError("ThreadCompute has no executor; provide one or override submit()")
        return self._executor.submit(fn, *args, **kwargs)

    def submit_nav_read(self, fn) -> concurrent.futures.Future:
        """Run ``fn`` (a no-arg callable returning an ndarray) on the LOCAL
        interactive-read pool — cancellable, serial, never remote."""
        pool = self._executor if self._executor is not None else self._nav_pool()
        return pool.submit(fn)

"""
Patch: blosc decodes with its thread pool on every thread, one caller at a time.

WHAT
----
Sets ``numcodecs.blosc.use_threads = True`` and ``set_nthreads(N)``, and wraps
the module-level ``numcodecs.blosc.compress``, ``decompress`` and, where the
installed version has it, ``decompress_partial`` so each call holds one
process-wide lock. ``Blosc.encode``, ``Blosc.decode`` and
``Blosc.decode_partial`` look those names up in the module at call time, so
every zarr read and write in THIS process goes through the wrappers.

Applied twice: by ``ensure_heavy_imports`` in the backend process, and by the
worker plugin (``dask_manager._WorkerTuningPlugin.setup``) in every dask worker
process, which is a separate spawned process with its own pool and its own
lock. Every task in a worker runs on a worker thread, so without the plugin a
worker decodes single-threaded and the batch's parallelism is dask's alone.

``SPYDE_BLOSC_THREADS=0`` applies nothing, which is the A/B switch. Unset uses
:data:`DEFAULT_THREAD_COUNT`; any other number sets that thread count.

WHY
---
``numcodecs`` decides whether to use blosc's thread pool from the identity of
the calling thread: with ``use_threads`` left at ``None`` it uses the pool on
the main thread and decodes single-threaded everywhere else. The navigator
reads on the ``_NavDispatcher`` thread and the block prefetcher on its own, so
in this process the fast path was never taken.

Measured on a real .zspy (blosc zstd level 1 with shuffle, 64 MiB chunks of
512^2 float32, compressed to about 50 MiB), decoding one chunk off the main
thread: 172 ms single-threaded, 29 ms at eight threads, 22 ms at sixteen. On
the main thread, where the pool was already used, 30 ms. A navigator crossing
into a new chunk paid the 172 ms.

blosc's pool is global, so two callers using it at once corrupt each other's
work. The lock is what makes turning the pool on safe when the dispatcher, the
prefetcher and the threaded scheduler all decode. It serialises decodes that used to
overlap, and the batch paths got quicker anyway: over the same square of
chunks on dask's threaded scheduler with eight threads, the navigator sum went
615 ms to 486 ms and a Find Vectors DoG batch 1201 ms to 1093 ms. Serialised
pooled decodes finish more chunks per second than eight concurrent
single-threaded ones, so nothing had to be traded for the interactive win.

The lock covers the call and nothing else. It is not held across any read of
the store, and no other lock is taken inside it. It is not fair either, so a
threaded ``.zspy`` save in this process, compressing chunk after chunk, can
keep a navigator decode waiting for as long as the save runs.

WHEN TO REMOVE
--------------
When numcodecs uses the pool off the main thread and serialises access to it
itself. The thread-identity rule is in ``numcodecs.blosc._get_use_threads``;
there is no upstream issue for it yet. Until then removing this module returns
every off-main-thread decode to single-threaded.
"""
from __future__ import annotations

import functools
import logging
import os
import threading

log = logging.getLogger(__name__)

# Threads blosc's pool uses per call. Eight: a whole navigator read of a 64 MiB
# chunk measured 83 ms at eight threads and 83 ms at sixteen on a 48-core box,
# because once the pool is on, the decode is no longer the larger half of the
# read. Eight is also the pool size numcodecs picks for itself, so it does not
# oversubscribe a machine with fewer cores than this one.
DEFAULT_THREAD_COUNT = 8

THREAD_COUNT_VARIABLE = "SPYDE_BLOSC_THREADS"

MARKER = "_spyde_serialised"

# One caller at a time, for the whole process: blosc's thread pool is global.
_POOL_LOCK = threading.Lock()


def thread_count() -> int | None:
    """Threads to give blosc's pool, or None to leave numcodecs alone."""
    setting = os.environ.get(THREAD_COUNT_VARIABLE)
    if setting is None:
        return DEFAULT_THREAD_COUNT
    try:
        count = int(setting)
    except ValueError:
        log.warning("spyde.external.numcodecs: %s is %r, which is not a number; "
                    "using %d", THREAD_COUNT_VARIABLE, setting,
                    DEFAULT_THREAD_COUNT)
        return DEFAULT_THREAD_COUNT
    return None if count <= 0 else count


def _serialised(function):
    @functools.wraps(function)
    def call(*args, **kwargs):
        with _POOL_LOCK:
            return function(*args, **kwargs)
    setattr(call, MARKER, True)
    return call


def apply() -> bool:
    """Idempotently turn blosc's thread pool on for every thread and serialise
    access to it. Returns True if the wrappers are in place, False if the
    switch is off or the upstream shape changed and it was skipped."""
    count = thread_count()
    if count is None:
        return False
    try:
        import numcodecs.blosc as blosc
    except Exception as e:
        log.warning("spyde.external.numcodecs: numcodecs.blosc import failed, "
                    "decodes stay single-threaded off the main thread: %s", e)
        return False
    for name in ("compress", "decompress", "use_threads", "set_nthreads"):
        if not hasattr(blosc, name):
            log.warning("spyde.external.numcodecs: numcodecs.blosc has no %s; "
                        "skipping the thread-pool patch", name)
            return False
    if getattr(blosc.decompress, MARKER, False):
        return True

    blosc.compress = _serialised(blosc.compress)
    blosc.decompress = _serialised(blosc.decompress)
    # Blosc.decode_partial's route to the same pool. zarr 2 calls it only with
    # partial decompression enabled, which is off by default, and older
    # numcodecs may not have it at all.
    if hasattr(blosc, "decompress_partial"):
        blosc.decompress_partial = _serialised(blosc.decompress_partial)
    blosc.use_threads = True
    blosc.set_nthreads(count)
    log.debug("spyde.external.numcodecs: blosc pool on with %d threads, "
              "serialised", count)
    return True

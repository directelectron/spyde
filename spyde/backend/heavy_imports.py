"""
heavy_imports.py — single-flight import of the heavy analysis stack.

hyperspy + pyxem have internal circular imports; importing them CONCURRENTLY
from two threads (the startup prewarm and a data-load thread) can surface
``cannot import name … from partially initialized module 'pyxem.signals'``
and permanently poison ``sys.modules`` for the session. This became likely
once the backend ran at full speed (the Electron tick fix removed the frozen
timers that accidentally serialized startup).

Every thread that touches hyperspy/pyxem for the first time calls
``ensure_heavy_imports()`` first: one thread performs the import, the others
wait on the lock, and subsequent calls are a no-op check.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_DONE = False


def ensure_heavy_imports() -> None:
    global _DONE
    if _DONE:
        return
    with _LOCK:
        if _DONE:
            return
        import hyperspy.api  # noqa: F401
        try:
            import pyxem  # noqa: F401
        except Exception as e:
            # pyxem is required by the diffraction paths but a plain-imaging
            # session can live without it — don't fail the whole import gate.
            log.warning("pyxem import failed during heavy-import warmup: %s", e)
        _patch_cached_dask_client()
        _DONE = True


def _patch_cached_dask_client() -> None:
    """Make ``CachedDaskArray.client`` honour ``_client = None`` as "no client →
    synchronous numpy-cache path", instead of falling back to the process-global
    default ``distributed.Client``.

    Why: the navigator frame read (``update_from_navigation_selection``) runs
    SERIALLY and BLOCKING on the ``_NavDispatcher`` thread and wants the fast
    synchronous chunk cache (~1-2 ms dwell-in-chunk hits), NOT a distributed
    round-trip (~16 ms dwell / ~100 ms cross-chunk). It sets ``cached_arr._client
    = None`` to request that. But the fork's ``client`` property, when
    ``_client is None``, calls ``dask.distributed.get_client()`` — which returns
    the app's default ``Client(cluster)`` from any non-worker thread (the
    ``_NavDispatcher`` thread included). So the pin was a no-op and every nav move
    still went distributed (measured, and confirmed by review). This patch removes
    that fallback so an explicit ``_client = None`` really selects the synchronous
    branch; distributed callers still work by setting ``_client`` explicitly.

    Safe/targeted: the ``CachedDaskArray`` is used (via ``_get_cache_dask_chunk``)
    only by the navigator read here; VI/orientation compute goes through
    ``ComputeBackend``/the client directly and is unaffected. Idempotent."""
    try:
        from hyperspy.misc.array_tools import CachedDaskArray
        if getattr(CachedDaskArray, "_spyde_client_patched", False):
            return

        def _client_get(self):
            # Honour an explicit client (distributed callers set it); otherwise
            # None means "synchronous cache" — do NOT auto-adopt the global client.
            return self._client

        CachedDaskArray.client = property(_client_get)
        CachedDaskArray._spyde_client_patched = True
        log.debug("patched CachedDaskArray.client to honour _client=None (sync cache)")
    except Exception as e:
        log.warning("could not patch CachedDaskArray.client: %s", e)

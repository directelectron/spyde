"""
Patch: ``CachedDaskArray.client`` must honour ``_client = None`` as
"no client → synchronous numpy-cache path".

WHAT
----
The hyperspy fork's ``hyperspy.misc.array_tools.CachedDaskArray`` exposes a
``client`` property whose getter, when ``self._client is None``, falls back to
``dask.distributed.get_client()``. This patch replaces that getter with one that
returns ``self._client`` verbatim — so an explicit ``_client = None`` really
selects the synchronous branch, and a distributed caller that sets ``_client``
still gets it back.

WHY
---
The navigator frame read (``update_from_navigation_selection``) runs SERIALLY and
BLOCKING on the ``_NavDispatcher`` thread and wants the fast synchronous chunk
cache (~1-2 ms dwell-in-chunk hits), NOT a distributed round-trip (~16 ms dwell /
~100 ms cross-chunk). It sets ``cached_arr._client = None`` to request that. But
``get_client()`` returns the app's process-global default ``Client(cluster)`` from
ANY non-worker thread (the ``_NavDispatcher`` thread included — it does NOT raise),
so the pin was a silent no-op and every nav move still went distributed (measured;
confirmed by review). Removing the fallback makes the pin real.

Targeted + safe: ``CachedDaskArray`` is used (via ``_get_cache_dask_chunk``) only
by the navigator read; VI / orientation compute go through ``ComputeBackend`` / the
client directly and are unaffected.

WHEN TO REMOVE
--------------
When the hyperspy fork's ``CachedDaskArray.client`` getter no longer auto-adopts
the global client for ``_client is None`` (i.e. the fork is fixed upstream, or the
behaviour is made opt-in). Track via the fork branch
``github.com/cssfrancis/hyperspy@slice-integrate2`` (the ``CachedDaskArray`` /
``get_index`` cache logic lives there). No public upstream issue/PR yet — this is a
fork-local getter.

History: this lived as ``heavy_imports._patch_cached_dask_client`` (still
re-exported there for back-compat) before moving here. Tests:
``spyde/tests/migrated/test_cache_client_patch.py``,
``spyde/tests/migrated/test_cache_client_patch`` +
``cache_client_ambient_threaded`` memory note.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def apply() -> bool:
    """Idempotently patch ``CachedDaskArray.client`` to honour ``_client=None``.

    Returns True if the patch is in place (applied now or already applied),
    False if the upstream shape changed and it was skipped."""
    try:
        from hyperspy.misc.array_tools import CachedDaskArray
    except Exception as e:  # upstream moved/renamed the class
        log.warning("spyde.external.hyperspy: CachedDaskArray import failed, "
                    "leaving nav-read client fallback in place: %s", e)
        return False

    if getattr(CachedDaskArray, "_spyde_client_patched", False):
        return True

    # Guard: only patch if the attribute we're replacing actually exists as we
    # expect (a property). If the fork changed its shape, skip loudly rather than
    # silently masking a real client attribute.
    existing = getattr(CachedDaskArray, "client", None)
    if existing is not None and not isinstance(existing, property):
        log.warning("spyde.external.hyperspy: CachedDaskArray.client is not a "
                    "property (%r); skipping the _client=None patch", type(existing))
        return False

    def _client_get(self):
        # Honour an explicit client (distributed callers set it); otherwise None
        # means "synchronous cache" — do NOT auto-adopt the global client.
        return self._client

    try:
        CachedDaskArray.client = property(_client_get)
        CachedDaskArray._spyde_client_patched = True
    except Exception as e:  # pragma: no cover - defensive
        log.warning("spyde.external.hyperspy: could not patch "
                    "CachedDaskArray.client: %s", e)
        return False

    log.debug("spyde.external.hyperspy: patched CachedDaskArray.client to honour "
              "_client=None (synchronous cache)")
    return True

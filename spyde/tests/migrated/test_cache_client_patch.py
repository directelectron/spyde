"""
CachedDaskArray.client patch — honour _client=None as "synchronous cache".

The navigator read sets `cached_arr._client = None` to select the fast synchronous
numpy-cache path. The hyperspy fork's `client` property otherwise falls back to
`dask.distributed.get_client()`, which adopts the app's process-global default
Client from any thread → the pin was a no-op and every nav move went distributed.
`_patch_cached_dask_client` (applied in ensure_heavy_imports) removes that
fallback so _client=None really means synchronous.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs

from spyde.backend.heavy_imports import (
    ensure_heavy_imports, _patch_cached_dask_client,
)


class TestCacheClientPatch:
    def test_patch_applied(self):
        ensure_heavy_imports()
        from hyperspy.misc.array_tools import CachedDaskArray
        assert getattr(CachedDaskArray, "_spyde_client_patched", False) is True

    def test_client_none_is_honoured(self):
        # After the patch, a cache with _client=None reports .client == None even
        # if there is a global default client — no auto-adoption.
        ensure_heavy_imports()
        s = hs.signals.Signal2D(
            da.zeros((4, 4, 8, 8), dtype=np.uint16, chunks=(2, 2, -1, -1))).as_lazy()
        s._get_cache_dask_chunk(np.array([[0, 0]]), get_result=True)  # build cache
        ca = s.cached_dask_array
        ca._client = None
        assert ca.client is None, "patched .client must not adopt a global client"

    def test_explicit_client_still_returned(self):
        # A distributed caller that sets _client explicitly still gets it back.
        ensure_heavy_imports()
        s = hs.signals.Signal2D(
            da.zeros((4, 4, 8, 8), dtype=np.uint16, chunks=(2, 2, -1, -1))).as_lazy()
        s._get_cache_dask_chunk(np.array([[0, 0]]), get_result=True)
        ca = s.cached_dask_array
        sentinel = object()
        ca._client = sentinel
        assert ca.client is sentinel

    def test_idempotent(self):
        _patch_cached_dask_client()
        _patch_cached_dask_client()   # second call is a no-op, no error
        from hyperspy.misc.array_tools import CachedDaskArray
        assert CachedDaskArray._spyde_client_patched is True

    def test_synchronous_read_returns_correct_frame(self):
        # With _client=None (synchronous branch), a frame reads correctly.
        ensure_heavy_imports()
        data = np.zeros((4, 4, 8, 8), dtype=np.uint16)
        data[1, 2] = 7                                   # marker at nav (1,2)
        s = hs.signals.Signal2D(
            da.from_array(data, chunks=(2, 2, -1, -1))).as_lazy()
        s._get_cache_dask_chunk(np.array([[0, 0]]), get_result=True)
        s.cached_dask_array._client = None
        # data order [iy, ix]; read nav (iy=1, ix=2).
        frame = np.asarray(
            s._get_cache_dask_chunk(np.array([[1, 2]]), get_result=True))
        assert frame.shape == (8, 8)
        assert float(frame[0, 0]) == 7.0

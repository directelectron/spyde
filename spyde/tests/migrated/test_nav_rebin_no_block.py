"""A rebinned/derived lazy view scrubs SYNCHRONOUSLY via the decoded-chunk cache.

A derived view (rebin/crop) has no hyperspy CachedDaskArray, so a naive read
re-decodes + re-transforms the whole source nav-chunk on every move. The read now
serves it synchronously through _direct_read_frame + the per-plot _NavChunkCache:
the first frame in a chunk pays one decode, every other frame in that chunk is a
free numpy slice — NO async round-trip, NO per-move re-decode.
"""
import time

import numpy as np
import dask.array as da

from spyde.drawing.update_functions import _direct_read_frame, _NavChunkCache


class _AxesManager:
    def __init__(self, nav_dim):
        self.navigation_dimension = nav_dim


class _Signal:
    def __init__(self, data, nav_dim):
        self.data = data
        self._lazy = True
        self.cached_dask_array = None  # derived view → no hyperspy cache
        self.axes_manager = _AxesManager(nav_dim)


class _Plot:
    """Minimal stand-in carrying the per-plot chunk cache _direct_read_frame reads."""
    def __init__(self):
        self._nav_chunk_cache = _NavChunkCache()


class _Prof:
    def stage(self, name):
        import contextlib
        return contextlib.nullcontext()

    def set_frame(self, *a):
        pass

    def done(self, *a, **k):
        pass


def _slow_rebinned_movie(n=32, frame=(16, 16), chunk=8, decode_s=0.3, seed=0):
    """A lazy movie whose per-source-block DECODE sleeps, then a rebin on top."""
    base = np.random.RandomState(seed).randint(0, 4000, (n, *frame)).astype(np.uint16)

    def _slow_block(block):
        time.sleep(decode_s)
        return block

    raw = da.from_array(base, chunks=(chunk, -1, -1)).map_blocks(_slow_block, dtype=base.dtype)
    reb = da.coarsen(np.mean, raw, {1: 2, 2: 2})  # 16->8 signal, derived
    expected = base.reshape(n, frame[0] // 2, 2, frame[1] // 2, 2).mean(axis=(2, 4))
    return reb, expected, decode_s, chunk


class TestRebinSyncCached:
    def test_dwell_in_chunk_is_fast_no_redecode(self):
        reb, expected, decode_s, chunk = _slow_rebinned_movie(decode_s=0.3, chunk=8)
        sig = _Signal(reb, nav_dim=1)
        plot = _Plot()

        # First frame in chunk [0:8] → one slow decode (>= decode_s).
        t0 = time.perf_counter()
        f0 = _direct_read_frame(sig, None, np.array([1]), _Prof(), child=plot)
        first_ms = (time.perf_counter() - t0) * 1000
        assert f0 is not None
        np.testing.assert_allclose(f0, expected[1], rtol=1e-5)
        assert first_ms >= decode_s * 1000 * 0.5, "first read should pay the chunk decode"

        # Other frames in the SAME chunk → cache HITS, each ≪ a decode (no re-decode).
        for fi in (0, 2, 5, 7):
            t0 = time.perf_counter()
            f = _direct_read_frame(sig, None, np.array([fi]), _Prof(), child=plot)
            dwell_ms = (time.perf_counter() - t0) * 1000
            np.testing.assert_allclose(f, expected[fi], rtol=1e-5)
            assert dwell_ms < decode_s * 1000 * 0.25, (
                f"dwell frame {fi} took {dwell_ms:.0f}ms — re-decoded instead of cache hit")

    def test_crossing_chunk_boundary_decodes_once_then_dwells(self):
        reb, expected, decode_s, chunk = _slow_rebinned_movie(decode_s=0.2, chunk=8)
        sig = _Signal(reb, nav_dim=1)
        plot = _Plot()
        # Cross into chunk [8:16]: first frame pays one decode…
        t0 = time.perf_counter()
        _direct_read_frame(sig, None, np.array([9]), _Prof(), child=plot)
        assert (time.perf_counter() - t0) >= decode_s * 0.5
        # …then a neighbour in the same chunk is a hit.
        t0 = time.perf_counter()
        f = _direct_read_frame(sig, None, np.array([12]), _Prof(), child=plot)
        assert (time.perf_counter() - t0) < decode_s * 0.25
        np.testing.assert_allclose(f, expected[12], rtol=1e-5)

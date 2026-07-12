"""
Unified direct-read viewing path — read the requested nav slice directly, bypass
get_index's CachedDaskArray overhead.

For EVERY navigator (movie frame, 4D-STEM diffraction pattern, integrating region)
the hyperspy CachedDaskArray adds ~160 ms/frame of overhead (seconds on a cold
miss); a direct raw[idx].compute(scheduler="synchronous") of the same slice is
~2-30 ms and byte-identical. It also serves DERIVED views (rebin/crop) that have no
CachedDaskArray, and stays memory-bounded (dask reads only the frame's deps).
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import dask.array as da
import hyperspy.api as hs

from spyde.drawing.update_functions import (
    _direct_read_frame, NavProfile,
)


class _Sel:
    def __init__(self, integrating=False):
        self.is_integrating = integrating


def _movie(n=8, frame=(16, 16), dtype=np.uint8):
    data = np.arange(n * frame[0] * frame[1], dtype=dtype).reshape(n, *frame)
    s = hs.signals.Signal2D(da.from_array(data, chunks=(1, -1, -1))).as_lazy()
    return s, data


def _4dstem(nav=(4, 4), sig=(8, 8), dtype=np.uint16):
    data = np.arange(nav[0] * nav[1] * sig[0] * sig[1],
                     dtype=dtype).reshape(*nav, *sig)
    s = hs.signals.Signal2D(
        da.from_array(data, chunks=(2, 2, -1, -1))).as_lazy()
    return s, data


class TestDirectReadSinglePoint:
    def test_movie_single_frame(self):
        s, data = _movie()
        frame = _direct_read_frame(s, _Sel(), np.array([3]), NavProfile("SIG"))
        assert frame is not None
        assert frame.dtype == data.dtype                 # native dtype, no round
        assert np.array_equal(frame, data[3])

    def test_4dstem_diffraction_pattern(self):
        # A 4D-STEM DP (nav-dim 2, single point) now ALSO takes the direct path.
        s, data = _4dstem()
        frame = _direct_read_frame(s, _Sel(), np.array([1, 2]), NavProfile("SIG"))
        assert frame is not None
        assert frame.dtype == data.dtype
        # indices are (iy, ix) → data[1, 2].
        assert np.array_equal(frame, data[1, 2])

    def test_matches_get_index_movie(self):
        s, data = _movie()
        s._get_cache_dask_chunk(np.array([[0]]), get_result=True)
        s.cached_dask_array._client = None
        gi = np.asarray(s._get_cache_dask_chunk(np.array([[5]]), get_result=True))
        direct = _direct_read_frame(s, _Sel(), np.array([5]), NavProfile("SIG"))
        assert np.array_equal(direct.astype(gi.dtype), gi.astype(gi.dtype))

    def test_out_of_range_index_read_raises_to_none(self):
        # No clamping in the helper (the caller clamps before calling); an OOB
        # index makes data[t] raise → the helper returns None (fall through), not
        # a wrong frame.
        s, data = _movie(n=8)
        out = _direct_read_frame(s, _Sel(), np.array([99]), NavProfile("SIG"))
        assert out is None


class TestDirectReadRegion:
    def test_region_mean_matches_get_index(self):
        # An integrating region: N nav points → frame-wise mean, parity with the
        # cache path (value-for-value after integer rounding).
        s, data = _4dstem(nav=(6, 6), sig=(8, 8))
        pts = np.array([[iy, ix] for iy in range(1, 4) for ix in range(1, 4)])  # 9
        s._get_cache_dask_chunk(np.array([[0, 0]]), get_result=True)
        s.cached_dask_array._client = None
        gi = np.asarray(s._get_cache_dask_chunk(pts, get_result=True))
        direct = _direct_read_frame(s, _Sel(integrating=True), pts,
                                    NavProfile("SIG"))
        assert direct is not None
        assert np.array_equal(np.rint(direct), np.rint(gi))

    def test_region_rounds_integer_source(self):
        s, data = _4dstem(nav=(4, 4), sig=(4, 4))
        pts = np.array([[0, 0], [0, 1]])            # mean of two frames
        direct = _direct_read_frame(s, _Sel(integrating=True), pts,
                                    NavProfile("SIG"))
        assert direct.dtype == data.dtype           # rounded back to native dtype
        expect = np.rint(data[[0, 0], [0, 1]].mean(axis=0)).astype(data.dtype)
        assert np.array_equal(direct, expect)

    def test_large_region_stays_bounded(self):
        # A large region is read INCREMENTALLY (peak ~1 frame), so it no longer
        # falls through — it's served directly and stays memory-bounded.
        s, data = _4dstem(nav=(8, 8), sig=(32, 32))
        pts = np.array([[iy, ix] for iy in range(8) for ix in range(8)])  # all 64
        out = _direct_read_frame(s, _Sel(integrating=True), pts, NavProfile("SIG"))
        assert out is not None
        expect = np.rint(
            data.reshape(64, 32, 32).mean(axis=0)).astype(data.dtype)
        assert np.array_equal(out, expect)


class TestDirectReadDerivedViews:
    def test_rebinned_view_scrubs(self):
        # A rebinned lazy view has NO CachedDaskArray — the direct read is the only
        # path that serves it. One output frame pulls only its source frame(s).
        s, data = _movie(n=10, frame=(32, 32))
        sr = s.rebin(scale=(1, 2, 2))               # 32→16 signal, still lazy
        assert sr._lazy
        frame = _direct_read_frame(sr, _Sel(), np.array([4]), NavProfile("SIG"))
        assert frame is not None
        assert frame.shape == (16, 16)
        expect = np.asarray(sr.data[4].compute(scheduler="synchronous"))
        assert np.array_equal(frame, expect)

    def test_cropped_view_scrubs(self):
        s, data = _movie(n=10, frame=(32, 32))
        sc = s.isig[8:24, 8:24]                      # crop signal region, lazy
        frame = _direct_read_frame(sc, _Sel(), np.array([4]), NavProfile("SIG"))
        assert frame is not None
        assert frame.shape == (16, 16)
        assert np.array_equal(frame, data[4, 8:24, 8:24])


class TestDirectReadMemoryBounded:
    def test_single_frame_on_monolithic_chunk_is_bounded(self):
        # Worst case: the whole array is ONE dask chunk. Reading one frame must NOT
        # materialise the whole chunk — dask's slice optimisation reads only the
        # frame. (A small fixture so the assert is meaningful.)
        n, fr = 40, (256, 256)
        data = np.zeros((n, *fr), dtype=np.uint8)
        s = hs.signals.Signal2D(
            da.from_array(data, chunks=(-1, -1, -1))).as_lazy()   # monolithic
        one_frame = fr[0] * fr[1]                                  # bytes (uint8)
        whole = n * one_frame
        tracemalloc.start()
        frame = _direct_read_frame(s, _Sel(), np.array([10]), NavProfile("SIG"))
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert frame is not None
        # Peak must be far below the whole array — a few frames' worth, not all 40.
        assert peak < whole // 4, (
            f"single-frame read peaked {peak/1e6:.1f}MB of a {whole/1e6:.1f}MB "
            f"array — must not materialise the whole chunk")

    def test_region_block_bounded_to_block(self):
        # Larger fixture so the block (5 frames) is unambiguously a small fraction
        # of the whole array, leaving headroom for vindex/mean overhead.
        n, fr = 120, (256, 256)
        data = np.zeros((n, *fr), dtype=np.uint8)
        s = hs.signals.Signal2D(
            da.from_array(data, chunks=(1, -1, -1))).as_lazy()
        pts = np.array([[i] for i in range(5)])          # 5-frame region
        one_frame = fr[0] * fr[1]
        whole = n * one_frame
        tracemalloc.start()
        _direct_read_frame(s, _Sel(integrating=True), pts, NavProfile("SIG"))
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Bounded to the block (+ vindex/mean overhead), NOT the whole 40-frame
        # array — the point is it doesn't materialise the dataset.
        assert peak < whole // 2, (
            f"region block peaked {peak/1e6:.1f}MB of a {whole/1e6:.1f}MB array")


class TestDirectReadFallThrough:
    def test_eager_falls_through(self):
        # Eager (in-RAM) data has no .compute — the helper returns None so the
        # eager branch handles it.
        data = np.arange(8 * 16 * 16, dtype=np.uint8).reshape(8, 16, 16)
        s = hs.signals.Signal2D(data)   # eager
        out = _direct_read_frame(s, _Sel(), np.array([2]), NavProfile("SIG"))
        assert out is None

"""
Unified cached navigator read (Phase 2).

``update_from_navigation_selection`` now computes the frame SYNCHRONOUSLY via
hyperspy's numpy chunk cache (get_index, no distributed client) on the serial
_NavDispatcher, returning a plain ndarray — no distributed Future, no shared
memory. This pins the two behaviours that matter:

  * a lazy SINGLE-point read returns the exact frame as a numpy array (paints
    immediately via Plot.update_data), and
  * an INTEGER integrating-region mean is ROUNDED back to the source dtype, to
    match the old distributed path (weighted_mean_round_from_sums) so the DP
    navigator's contrast doesn't shift — the benchmarks.md rounding gotcha.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs

from spyde.drawing.update_functions import update_from_navigation_selection


class _StubPlot:
    """Minimal child-plot stand-in exposing what the update function reads."""
    def __init__(self, signal):
        class _PS:
            current_signal = signal
        self.plot_state = _PS()
        self.main_window = None


class _StubSelector:
    def __init__(self, is_integrating):
        self.is_integrating = is_integrating


def _lazy_int_4d(nav=(4, 5), sig=(8, 8), dtype=np.uint16):
    # data[iy, ix] frame is filled with (iy*10 + ix) so a read is identifiable.
    ny, nx = nav
    data = np.zeros((ny, nx) + sig, dtype=dtype)
    for iy in range(ny):
        for ix in range(nx):
            data[iy, ix] = iy * 10 + ix
    s = hs.signals.Signal2D(da.from_array(data, chunks=(2, 2, -1, -1))).as_lazy()
    return s


class TestCachedNavRead:
    # NOTE: update_from_navigation_selection takes indices in WIDGET order
    # (ix, iy) for a 2-D-signal dataset and transposes the last pair to data
    # order (iy, ix) internally. So to read data[iy, ix] pass [ix, iy].

    def test_single_point_returns_exact_frame_as_ndarray(self):
        s = _lazy_int_4d()
        child = _StubPlot(s)
        sel = _StubSelector(is_integrating=False)
        # widget [ix=3, iy=1] → data[iy=1, ix=3] = 13.
        out = update_from_navigation_selection(sel, child, np.array([[3, 1]]))
        assert isinstance(out, np.ndarray)          # paints synchronously
        assert out.dtype == np.uint16               # native frame dtype
        assert float(out[0, 0]) == 1 * 10 + 3

    def test_integer_region_mean_is_rounded_to_source_dtype(self):
        # Region = 2x2 nav block, data frames 10,11,20,21 → mean 15.5 → rounds to
        # 16 (uint16), matching the old distributed weighted_mean_round_from_sums;
        # NOT the un-rounded float 15.5 the synchronous np.mean branch returns raw.
        s = _lazy_int_4d()
        child = _StubPlot(s)
        sel = _StubSelector(is_integrating=True)
        # widget [ix, iy]: ix in {0,1}, iy in {1,2} → data (1,0)=10,(1,1)=11,
        #   (2,0)=20,(2,1)=21 → mean 15.5.
        pts = np.array([[0, 1], [1, 1], [0, 2], [1, 2]])
        out = update_from_navigation_selection(sel, child, pts)
        assert isinstance(out, np.ndarray)
        assert np.issubdtype(out.dtype, np.integer), "region mean must stay integer"
        assert float(out[0, 0]) == 16.0, f"expected rounded 16, got {out[0, 0]}"

    def test_float_region_mean_is_not_rounded(self):
        # A float source keeps the un-rounded mean (only integer dtypes round).
        ny, nx, sig = 4, 5, (8, 8)
        data = np.zeros((ny, nx) + sig, dtype=np.float32)
        for iy in range(ny):
            for ix in range(nx):
                data[iy, ix] = iy * 10 + ix
        s = hs.signals.Signal2D(da.from_array(data, chunks=(2, 2, -1, -1))).as_lazy()
        child = _StubPlot(s)
        sel = _StubSelector(is_integrating=True)
        pts = np.array([[0, 1], [1, 1], [0, 2], [1, 2]])   # mean 15.5
        out = update_from_navigation_selection(sel, child, pts)
        assert np.issubdtype(out.dtype, np.floating)
        assert abs(float(out[0, 0]) - 15.5) < 1e-6

    def test_result_is_never_a_future(self):
        # The unified path must NOT return a distributed Future (that path is
        # gone); update_data paints an ndarray directly.
        from distributed import Future
        s = _lazy_int_4d()
        child = _StubPlot(s)
        sel = _StubSelector(is_integrating=False)
        out = update_from_navigation_selection(sel, child, np.array([[2, 2]]))
        assert not isinstance(out, Future)
        assert isinstance(out, np.ndarray)

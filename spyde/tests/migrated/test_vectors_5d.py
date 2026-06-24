"""
SpyDEDiffractionVectors 5-D (stacked) per-slice access.

A 5-D stack stores vectors with full_nav_shape = (n_stack, nav_y, nav_x); the
outer axis is the stack/time dimension. These exercise the per-slice helpers used
by the stack display: count_map_series, at_nav/kxy_at_nav (current slice only),
and the rect/disk virtual-image series.
"""
from __future__ import annotations

import numpy as np

from spyde.signals.diffraction_vectors import (
    SpyDEDiffractionVectors, N_COLS, COL_NAV_X, COL_NAV_Y, COL_KX, COL_KY,
    COL_TIME, COL_INTENSITY,
)


class _Ax:
    def __init__(self, size, scale=1.0, offset=0.0):
        self.size = size
        self.scale = scale
        self.offset = offset
        self.units = ""
        self.name = ""


def _make_5d():
    """stack=2, nav_y=2, nav_x=3 detector 8x8. One vector per (stack,y,x):
    intensity encodes the slice so we can tell slices apart; all peaks at kx=ky=4
    so an ROI at the centre catches them. Sorted outermost-first (stack,y,x)."""
    rows = []
    for st in range(2):
        for y in range(2):
            for x in range(3):
                r = np.zeros(N_COLS, dtype=np.float32)
                r[COL_NAV_X] = x
                r[COL_NAV_Y] = y
                r[COL_TIME] = st
                r[COL_KX] = 4.0
                r[COL_KY] = 4.0
                r[COL_INTENSITY] = 10.0 * (st + 1)   # slice 0 → 10, slice 1 → 20
                rows.append(r)
    flat = np.array(rows, dtype=np.float32)
    sig_axes = (_Ax(8), _Ax(8))
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat, full_nav_shape=(2, 2, 3),
        sig_shape=(8, 8), sig_axes=sig_axes,
        kernel_radius_px=1.0, kernel_radius_data=1.0,
    )


class TestVectors5D:
    def test_n_time_and_shapes(self):
        v = _make_5d()
        assert v.n_time == 2
        assert v.nav_shape == (2, 3)
        assert v.full_nav_shape == (2, 2, 3)

    def test_count_map_series_is_3d_per_slice(self):
        v = _make_5d()
        series = v.count_map_series()
        assert series.shape == (2, 2, 3)          # (stack, y, x)
        # One vector per position in every slice.
        assert np.all(series == 1)
        # And it differs from the 2-D summed map (which sums over stack → 2).
        assert v.count_map().shape == (2, 3)
        assert np.all(v.count_map() == 2)

    def test_count_map_at_t_matches_series_and_no_crash(self):
        v = _make_5d()
        # Regression: count_map_at_t walked nav_offsets[0] (vector offsets) as row
        # indices → IndexError on real data. Now it indexes the series.
        for t in range(v.n_time):
            cm = v.count_map_at_t(t)
            assert cm.shape == (2, 3)
            np.testing.assert_array_equal(cm, v.count_map_series()[t])
        # Out-of-range t is clamped, not an IndexError.
        assert v.count_map_at_t(99).shape == (2, 3)

    def test_at_nav_picks_current_slice_only(self):
        v = _make_5d()
        # at() (no lead) returns BOTH slices' vectors at (y=1, x=2).
        both = v.at(1, 2)
        assert both.shape[0] == 2
        # at_nav with lead=(stack,) returns ONLY that slice's single vector.
        s0 = v.at_nav(1, 2, lead=(0,))
        s1 = v.at_nav(1, 2, lead=(1,))
        assert s0.shape[0] == 1 and s1.shape[0] == 1
        assert float(s0[0, COL_INTENSITY]) == 10.0   # slice 0
        assert float(s1[0, COL_INTENSITY]) == 20.0   # slice 1

    def test_kxy_at_nav_shape(self):
        v = _make_5d()
        kxy = v.kxy_at_nav(0, 0, lead=(1,))
        assert kxy.shape == (1, 2)
        np.testing.assert_allclose(kxy[0], [4.0, 4.0])

    def test_at_nav_stale_lead_falls_back(self):
        v = _make_5d()
        # Wrong-length lead (e.g. a stale 2-D position) must not raise.
        out = v.at_nav(0, 0, lead=(0, 0))   # 2 lead coords but only 1 outer dim
        assert out.shape[1] == N_COLS

    def test_virtual_image_series_disk_is_3d(self):
        v = _make_5d()
        series = v.virtual_image_series(cx=4.0, cy=4.0, r_outer=2.0, r_inner=0.0,
                                        intensity_weighted=True)
        assert series.shape == (2, 2, 3)             # (stack, y, x)
        # Slice 0 sums intensity 10 per position, slice 1 sums 20.
        assert np.allclose(series[0], 10.0)
        assert np.allclose(series[1], 20.0)

    def test_virtual_image_series_rect_is_3d(self):
        v = _make_5d()
        series = v.virtual_image_series_rect(x0=2.0, y0=2.0, x1=6.0, y1=6.0,
                                             intensity_weighted=False)
        assert series.shape == (2, 2, 3)
        # count weighting → 1 vector per position per slice.
        assert np.all(series == 1.0)

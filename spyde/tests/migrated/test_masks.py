"""
Detector-mask builders (`spyde.actions.masks.widget_to_mask`).

REGRESSION GUARD for the "black virtual image on real data" bug:

anyplotlib overlay widgets report geometry (`cx/cy/r`, `x/y/w/h`) in
*image-pixel* coordinates — 0..image_width — with NO axis scale/offset applied.
The mask grid was previously built in *physical* units (`pixel*scale + offset`),
so on any calibrated detector axis (scale != 1) the ROI never overlapped the
grid → empty mask → virtual image of all zeros (black).  It stayed hidden
because every synthetic test used scale=1.

These tests use a CALIBRATED axis (scale=0.1) so the regression can't recur.
"""
from __future__ import annotations

import numpy as np
import hyperspy.api as hs

from anyplotlib.widgets._widgets2d import (
    CircleWidget, AnnularWidget, RectangleWidget,
)
from spyde.actions.masks import widget_to_mask


def _noop():
    pass


def _calibrated_dp(size=64, scale=0.1):
    """A single 2-D diffraction pattern with a CALIBRATED signal axis."""
    s = hs.signals.Signal2D(np.ones((size, size), dtype=np.float32))
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = 0.0
        ax.units = "1/nm"
    return s


class TestWidgetToMaskCalibration:
    def test_disk_mask_is_nonempty_on_calibrated_axes(self):
        """The whole bug in one assertion: a centred disk on a scale=0.1
        detector must select pixels, not an empty mask."""
        s = _calibrated_dp(size=64, scale=0.1)
        w = CircleWidget(_noop, cx=32, cy=32, r=6.0)
        mask = widget_to_mask(w, s)

        assert mask.shape == (64, 64)
        # area of a disk r=6 is ~113 px; allow generous bounds.
        assert mask.sum() > 50, "calibrated disk produced an (almost) empty mask"
        # centred: centre of mass near (32, 32) in PIXEL space.
        ys, xs = np.nonzero(mask)
        assert abs(xs.mean() - 32) < 1.5
        assert abs(ys.mean() - 32) < 1.5

    def test_mask_is_calibration_invariant(self):
        """The pixel mask must not depend on the axis scale — the widget reports
        pixels, so scale=1 and scale=0.1 give the SAME mask."""
        w = CircleWidget(_noop, cx=32, cy=32, r=8.0)
        m1 = widget_to_mask(w, _calibrated_dp(64, scale=1.0))
        m01 = widget_to_mask(w, _calibrated_dp(64, scale=0.1))
        m_big = widget_to_mask(w, _calibrated_dp(64, scale=50.0))
        assert np.array_equal(m1, m01)
        assert np.array_equal(m1, m_big)

    def test_offcentre_disk_lands_where_the_widget_is(self):
        """Off-centre ROI must select the right pixels (catches the latent x/y
        swap, which only hides for centred square detectors)."""
        s = _calibrated_dp(size=64, scale=0.1)
        # widget cx is horizontal (column), cy vertical (row).
        w = CircleWidget(_noop, cx=10, cy=50, r=4.0)
        mask = widget_to_mask(w, s)
        ys, xs = np.nonzero(mask)
        assert abs(xs.mean() - 10) < 1.5   # column ~ cx
        assert abs(ys.mean() - 50) < 1.5   # row ~ cy

    def test_annular_mask_is_a_ring_on_calibrated_axes(self):
        s = _calibrated_dp(size=64, scale=0.1)
        w = AnnularWidget(_noop, cx=32, cy=32, r_outer=12.0, r_inner=6.0)
        mask = widget_to_mask(w, s)
        assert mask.sum() > 50
        # the very centre pixel is inside r_inner → excluded.
        assert mask[32, 32] == 0.0

    def test_rectangle_mask_on_calibrated_axes(self):
        s = _calibrated_dp(size=64, scale=0.1)
        w = RectangleWidget(_noop, x=20, y=24, w=10, h=8)
        mask = widget_to_mask(w, s)
        ys, xs = np.nonzero(mask)
        assert xs.min() == 20 and xs.max() == 29   # [20, 30)
        assert ys.min() == 24 and ys.max() == 31   # [24, 32)
        assert mask.sum() == 80


class TestVirtualImageNonBlackOnCalibratedData:
    """End-to-end: a virtual image over a CALIBRATED 4-D dataset must not be all
    zeros (the user-visible 'black VI' symptom)."""

    def test_vi_is_nonzero_with_calibrated_signal_axes(self):
        nav = (4, 5)
        sig = (64, 64)
        data = np.zeros(nav + sig, dtype=np.float32)
        yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
        disk = ((xx - 32) ** 2 + (yy - 32) ** 2 <= 36).astype(np.float32)
        for k, idx in enumerate(np.ndindex(*nav)):
            data[idx] = disk * (k + 1)
        s = hs.signals.Signal2D(data)
        s.set_signal_type("electron_diffraction")
        for ax in s.axes_manager.signal_axes:
            ax.scale = 0.1
            ax.offset = 0.0

        w = CircleWidget(_noop, cx=32, cy=32, r=6.0)
        mask = widget_to_mask(w, s)
        assert mask.sum() > 0

        # Same reduction VirtualImageAction._virtual_image_array performs.
        vi = (s.data * mask).sum(axis=(-2, -1)) / float(mask.sum())
        assert vi.shape == nav
        assert float(np.asarray(vi).max()) > 0.0, "virtual image is all zeros (black)"

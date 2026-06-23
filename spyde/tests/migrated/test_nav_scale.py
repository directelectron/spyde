"""
Navigator scale → index regression tests.

The navigator image is drawn with the navigation axes' CALIBRATION (scale +
offset, e.g. a 3 nm step size with units 'nm') for the scale bar / tooltip, but
anyplotlib's 2-D overlay widgets (crosshair / rectangle) report their position in
IMAGE-PIXEL coordinates — the widget math in figure_esm.js
(``_canvasToImg2d`` / ``_imgToCanvas2d``) is pure image pixels and never applies
the axes' scale/offset (only the 1-D widgets use data coords).

So the selected pixel index IS the widget's cx/cy, rounded — the selector must
NOT divide by scale/offset. Doing so double-corrects a calibrated scan: a click on
pixel 10 of a 3 nm-step navigator would resolve to index round(10/3)=3, loading
the WRONG diffraction pattern (and clamping to the edge once the value exceeds
size). These tests pin that the mapping is calibration-INDEPENDENT across several
scales/offsets and a non-square scan.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs
import pytest


def _make_session():
    from spyde.backend.session import Session
    return Session(n_workers=1, threads_per_worker=1)


def _ramp_4d(nav, sig=(8, 8), scale=1.0, offset=0.0, units="nm"):
    """A 4-D signal whose DP MEAN encodes the flat nav index (k = iy*nx + ix),
    so the selected frame's mean tells us exactly which nav pixel was sliced.
    Navigation axes get the given scale/offset/units."""
    ny, nx = nav
    data = np.zeros((ny, nx) + sig, dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            data[iy, ix] = float(iy * nx + ix + 1)   # +1 so frame 0 is non-zero
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    # Navigation axes are the first two (y, x).
    nav_axes = s.axes_manager.navigation_axes
    for ax in nav_axes:
        ax.scale = scale
        ax.offset = offset
        ax.units = units
    return s


def _navigator_selector(session):
    """The navigation selector driving the signal plot for the single tree."""
    tree = session.signal_trees[-1]
    mgr = tree.navigator_plot_manager
    assert mgr is not None, "no navigator plot manager (not a navigated dataset?)"
    pw = next(iter(mgr.navigation_selectors.keys()))
    sel = mgr.navigation_selectors[pw][0]
    return sel


def _set_crosshair_pixels(sel, px, py):
    """Place the crosshair at IMAGE-PIXEL coordinates (what anyplotlib reports)
    and read back the integer nav indices the selector resolves."""
    cross = getattr(sel, "_crosshair_selector", sel)
    w = cross._widget
    assert w is not None, "crosshair widget not initialised"
    w.cx = float(px)
    w.cy = float(py)
    return cross.get_selected_indices()


class TestNavScaleIndexMapping:
    @pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (3.0, 0.0), (2.5, 10.0)])
    def test_crosshair_pixel_maps_to_same_index_regardless_of_calibration(
            self, scale, offset):
        """A crosshair at PIXEL (ix=2, iy=1) must resolve to array index (2, 1)
        regardless of scale/offset — the calibration must NOT be applied."""
        session = _make_session()
        try:
            s = _ramp_4d((4, 5), scale=scale, offset=offset)  # ny=4, nx=5
            session._add_signal(s, source_path=None)
            time.sleep(0.6)
            sel = _navigator_selector(session)

            ix, iy = 2, 1
            indices = _set_crosshair_pixels(sel, ix, iy)

            # Selector reports (x, y) rows. The widget pixel IS the index — the
            # scale/offset must be ignored (applying it was the bug).
            pt = np.asarray(indices).reshape(-1, 2)[0]
            assert int(pt[0]) == ix and int(pt[1]) == iy, (
                f"scale={scale} offset={offset}: crosshair at pixel "
                f"({ix},{iy}) → index {tuple(pt)}, expected ({ix},{iy})"
            )
        finally:
            session.shutdown()

    def test_scaled_navigator_selects_correct_diffraction_pattern(self):
        """End-to-end: placing the crosshair on nav PIXEL (ix=3, iy=2) of a
        scale-3 navigator must slice the DP whose mean encodes k = iy*nx+ix+1
        (not a clamped edge frame from a divided-by-scale index)."""
        session = _make_session()
        try:
            nx = 5
            s = _ramp_4d((4, nx), sig=(8, 8), scale=3.0, units="nm")
            session._add_signal(s, source_path=None)
            time.sleep(0.6)
            sel = _navigator_selector(session)

            ix, iy = 3, 2
            _set_crosshair_pixels(sel, ix, iy)
            sel.delayed_update_data(force=True)
            time.sleep(0.4)

            # The child (signal) plot must now hold the DP for frame k.
            child = next(iter(sel.children.keys()))
            data = child.current_data
            assert isinstance(data, np.ndarray), f"no DP array on child: {type(data)}"
            expected_k = float(iy * nx + ix + 1)
            assert abs(float(np.mean(data)) - expected_k) < 1e-3, (
                f"selected DP mean {float(np.mean(data))}, expected frame {expected_k}"
            )
        finally:
            session.shutdown()

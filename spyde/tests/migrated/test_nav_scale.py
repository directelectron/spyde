"""
Navigator scale → index regression tests.

The navigator image is drawn with the navigation axes' CALIBRATION (scale +
offset, e.g. a 3 nm step size with units 'nm'). The crosshair / rectangle widget
therefore reports its position in DATA coordinates (nm), not array indices. The
selector must convert those back to integer array indices via
``index = round((value - offset) / scale)`` before slicing — otherwise a click
at 30 nm on a scale-3 nm scan selects frame 30 instead of frame 10, loading the
WRONG diffraction pattern (and clamping to the edge once value > size).

This bites any non-unit-scale dataset (the 3 nm-step DE scan) and is invisible
on the synthetic scale-1 fixtures, so it regressed silently in the Qt→Electron
port. These tests pin the mapping across several scales/offsets and a
non-square scan.
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


def _set_crosshair_data_coords(sel, data_x, data_y):
    """Place the crosshair at DATA coordinates (axis units) and read back the
    integer nav indices the selector resolves."""
    cross = getattr(sel, "_crosshair_selector", sel)
    w = cross._widget
    assert w is not None, "crosshair widget not initialised"
    w.cx = float(data_x)
    w.cy = float(data_y)
    return cross.get_selected_indices()


class TestNavScaleIndexMapping:
    @pytest.mark.parametrize("scale,offset", [(1.0, 0.0), (3.0, 0.0), (2.5, 10.0)])
    def test_crosshair_data_coords_map_to_correct_index(self, scale, offset):
        """A crosshair at the DATA coordinate of nav pixel (ix=2, iy=1) must
        resolve to array index (2, 1) regardless of scale/offset."""
        session = _make_session()
        try:
            s = _ramp_4d((4, 5), scale=scale, offset=offset)  # ny=4, nx=5
            session._add_signal(s, source_path=None)
            time.sleep(0.6)
            sel = _navigator_selector(session)

            ix, iy = 2, 1
            data_x = offset + ix * scale
            data_y = offset + iy * scale
            indices = _set_crosshair_data_coords(sel, data_x, data_y)

            # Selector reports (x, y) rows. The selected pixel must be (ix, iy),
            # NOT (data_x, data_y) — that's the bug.
            pt = np.asarray(indices).reshape(-1, 2)[0]
            assert int(pt[0]) == ix and int(pt[1]) == iy, (
                f"scale={scale} offset={offset}: crosshair at data "
                f"({data_x},{data_y}) → index {tuple(pt)}, expected ({ix},{iy})"
            )
        finally:
            session.shutdown()

    def test_scaled_navigator_selects_correct_diffraction_pattern(self):
        """End-to-end: clicking nav pixel (ix=3, iy=2) on a scale-3 navigator
        must slice the DP whose mean encodes k = iy*nx+ix+1 (not a clamped edge
        frame)."""
        session = _make_session()
        try:
            nx = 5
            s = _ramp_4d((4, nx), sig=(8, 8), scale=3.0, units="nm")
            session._add_signal(s, source_path=None)
            time.sleep(0.6)
            sel = _navigator_selector(session)

            ix, iy = 3, 2
            _set_crosshair_data_coords(sel, ix * 3.0, iy * 3.0)
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

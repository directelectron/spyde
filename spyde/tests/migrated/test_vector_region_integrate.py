"""MDI region-integrate for the vector-disk display.

The find-vectors result window renders disks synchronously on every navigator
move via ``tree._render_frame_fn`` (installed by
``find_vectors_action._install_render_display``). A POINT crosshair emits one
``[[ix, iy]]`` row → ``render_frame``; a REGION selector (RectangleSelector)
emits a grid of ``[ix, iy]`` rows spanning the nav rectangle → the summed
``render_region``. These tests pin that the installed ``_fn`` branches on the
indices shape correctly, so dragging/resizing a nav region shows the summed DP.
"""
from __future__ import annotations

import numpy as np

from spyde.actions.find_vectors_action import _install_render_display
from spyde.signals.diffraction_vectors import (
    COL_INTENSITY, COL_KX, COL_KY, COL_TIME, N_COLS,
    SpyDEDiffractionVectors, _AxisLite,
)


def _vecs(nav=(4, 4), sig=64):
    """A small 4-D vectors set: one spot per position, drifting with the column
    so different positions render different frames (region sums differ)."""
    ny, nx = nav
    rows = []
    for iy in range(ny):
        for ix in range(nx):
            r = np.zeros(N_COLS, np.float32)
            r[0], r[1] = ix, iy
            r[COL_TIME] = -1.0
            # spot drifts across the detector with the nav column
            r[COL_KX] = -0.5 + 0.2 * ix
            r[COL_KY] = 0.1 * iy
            r[COL_INTENSITY] = 100.0 + ix + iy
            rows.append(r)
    flat = np.stack(rows).astype(np.float32)
    ax = _AxisLite(scale=2.0 / (sig - 1), offset=-1.0, size=sig, units="1/A", name="k")
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat, full_nav_shape=(ny, nx), sig_shape=(sig, sig),
        sig_axes=[ax, ax], kernel_radius_px=3.0, kernel_radius_data=0.0,
        params={}, nav_axes=None,
    )


class _Tree:
    """Minimal stand-in — _install_render_display only touches these attrs when
    there is no navigator_plot_manager (it stashes _render_frame_fn and bails)."""
    signal_plots = []
    navigator_plot_manager = None

    def __init__(self):
        self._render_frame_fn = None


def _installed_fn(vecs):
    tree = _Tree()
    _install_render_display(tree, vecs)
    assert tree._render_frame_fn is not None
    return tree._render_frame_fn


class TestRegionIntegrateFn:
    def test_crosshair_indices_render_single_frame(self):
        vecs = _vecs()
        fn = _installed_fn(vecs)
        # A crosshair emits [[ix, iy]] (one row) → render_frame(iy, ix).
        out = fn(None, None, np.array([[2, 1]]))          # ix=2, iy=1
        np.testing.assert_array_equal(out, vecs.render_frame(1, 2))

    def test_region_grid_indices_sum_frames(self):
        vecs = _vecs()
        fn = _installed_fn(vecs)
        # A RectangleSelector emits a grid of [ix, iy] pairs. Span nav rows 0-1,
        # cols 0-1 → the summed render_region of that rectangle.
        xs = np.array([0, 1])
        ys = np.array([0, 1])
        grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)   # (4, 2) [ix, iy]
        out = fn(None, None, grid)
        ref = vecs.render_region(0, 2, 0, 2)
        np.testing.assert_allclose(out, ref, atol=1e-4)
        # And that equals the explicit sum of the four per-position frames.
        expect = (vecs.render_frame(0, 0) + vecs.render_frame(0, 1)
                  + vecs.render_frame(1, 0) + vecs.render_frame(1, 1))
        np.testing.assert_allclose(out, expect, atol=1e-4)

    def test_single_cell_grid_equals_render_frame(self):
        vecs = _vecs()
        fn = _installed_fn(vecs)
        # A 1x1 "region" (grid with a single row) must collapse to render_frame.
        out = fn(None, None, np.array([[3, 2]]))          # ix=3, iy=2
        np.testing.assert_array_equal(out, vecs.render_frame(2, 3))

    def test_region_differs_from_pointer(self):
        vecs = _vecs()
        fn = _installed_fn(vecs)
        pointer = fn(None, None, np.array([[0, 0]]))
        xs, ys = np.array([0, 1, 2]), np.array([0, 1, 2])
        grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
        region = fn(None, None, grid)
        assert not np.array_equal(pointer, region)
        # The integrated frame is brighter (more disks summed in).
        assert region.sum() > pointer.sum()

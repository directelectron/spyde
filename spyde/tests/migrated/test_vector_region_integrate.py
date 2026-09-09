"""MDI region-integrate for the vector-disk display.

The find-vectors result window renders disks on every navigator move through
:class:`~spyde.actions.find_vectors_action.RenderedVectorsReader`, pinned on the
result tree. A POINT crosshair prepares one nav index and reads
``render_frame``; a REGION selector prepares a grid of nav indices and reads
``sum_points``, which is the summed ``render_region``. These tests pin the
reader's two answers, so dragging or resizing a nav region shows the summed
diffraction pattern.
"""
from __future__ import annotations

import numpy as np

from spyde.actions.find_vectors_action import RenderedVectorsReader
from spyde.array_cache.region_sum import finalize_sum
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


def _region(reader, points):
    """What the navigator read shows for an integrating region: the reader's
    accumulator, finalized the way every region read is."""
    points = np.asarray(points)
    return finalize_sum(reader.sum_points(points, np.float64),
                        len(points), np.float32)


class TestRenderedVectorsReader:
    def test_point_renders_a_single_frame(self):
        vecs = _vecs()
        reader = RenderedVectorsReader(vecs)
        # A prepared crosshair index is (iy, ix) in data order.
        np.testing.assert_array_equal(reader.read_frame((1, 2)),
                                      vecs.render_frame(1, 2))

    def test_region_points_sum_frames(self):
        vecs = _vecs()
        reader = RenderedVectorsReader(vecs)
        # A region selector prepares a grid of (iy, ix) rows. Span nav rows 0-1,
        # cols 0-1 → the summed render_region of that rectangle.
        grid = np.array([[iy, ix] for iy in (0, 1) for ix in (0, 1)])
        out = _region(reader, grid)
        np.testing.assert_allclose(out, vecs.render_region(0, 2, 0, 2), atol=1e-4)
        # And that equals the explicit sum of the four per-position frames.
        expect = (vecs.render_frame(0, 0) + vecs.render_frame(0, 1)
                  + vecs.render_frame(1, 0) + vecs.render_frame(1, 1))
        np.testing.assert_allclose(out, expect, atol=1e-4)

    def test_single_cell_region_equals_render_frame(self):
        vecs = _vecs()
        reader = RenderedVectorsReader(vecs)
        # A 1x1 "region" (one prepared row) must collapse to render_frame.
        out = _region(reader, np.array([[2, 3]]))
        np.testing.assert_allclose(out, vecs.render_frame(2, 3), atol=1e-4)

    def test_region_differs_from_pointer(self):
        vecs = _vecs()
        reader = RenderedVectorsReader(vecs)
        pointer = reader.read_frame((0, 0))
        grid = np.array([[iy, ix] for iy in (0, 1, 2) for ix in (0, 1, 2)])
        region = _region(reader, grid)
        assert not np.array_equal(pointer, region)
        # The integrated frame is brighter (more disks summed in).
        assert region.sum() > pointer.sum()

    def test_a_stacks_leading_index_selects_the_slice(self):
        """A 5-D scan's prepared index leads with the stack coordinate, which
        picks the slice whose vectors are drawn."""
        vecs = _vecs()
        reader = RenderedVectorsReader(vecs)
        # These vectors carry no time column, so every slice renders the same
        # frame; what is pinned is that the leading coordinate is not read as a
        # spatial one.
        np.testing.assert_array_equal(reader.read_frame((0, 1, 2)),
                                      vecs.render_frame(1, 2, t=0))

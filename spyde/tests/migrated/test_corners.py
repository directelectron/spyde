"""
The four corner blocks of a scan, and the plane through them
(:mod:`spyde.corners`).

The machinery lived in ``spyde.actions.dpc`` while DPC was its only caller.
Multi-angle reciprocal alignment is the second, so it moved here and ``dpc.py``
re-exports it. ``test_dpc.py`` still exercises every one of these through the
``dpc.`` name and is not repeated; what is pinned HERE is the move itself and
the one thing the move added — a per-corner extent.

**The re-export must be the same object, not a copy.** Two definitions that
agree on the day they are written is the failure mode a "shared" module is
supposed to remove, and a test that only checks behaviour would not see it.

**A corner block spans its OWN axis.** That is the deliberate divergence from
pyxem's ``_get_corner_slices``, which derives both block sizes from the
navigation axes and applies them to the numpy array the other way round —
harmless on a square scan, transposed on any other. It is the reason this code
exists rather than a call to ``get_linear_plane(fit_corners=…)``, so it is
pinned on a scan that is emphatically not square.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde import corners
from spyde.actions import dpc

#: Deliberately not square, and not a multiple of the other axis.
NAV = (40, 120)


class TestTheDpcReExport:
    """``dpc.corner_slices`` and friends must BE the shared objects."""

    @pytest.mark.parametrize("name", [
        "corner_slice", "corner_slices", "corner_boxes", "corner_mask",
        "corner_reference", "plane_through",
        "DEFAULT_CORNER_FRACTION", "MIN_CORNER_PX", "CORNER_NAMES",
    ])
    def test_the_name_resolves_to_the_same_object(self, name):
        assert getattr(dpc, name) is getattr(corners, name), (
            f"dpc.{name} is a second definition, not the shared one — the "
            f"two can now drift apart")

    def test_the_pyxem_divergence_is_still_documented(self):
        """The reason this is not ``get_linear_plane(fit_corners=…)``.

        A reader who does not know it will "simplify" the mask away, and the
        result is transposed corner blocks on every non-square scan — which
        looks fine, because the fit still converges.
        """
        doc = corners.corner_slices.__doc__ or ""
        assert "_get_corner_slices" in doc
        assert "square" in doc


class TestTheCornerBlocks:
    def test_there_are_four_in_a_named_order(self):
        assert len(corners.CORNER_NAMES) == 4
        assert corners.CORNER_NAMES[0] == "top left"
        assert len(corners.corner_slices(NAV)) == 4

    def test_each_block_spans_its_own_axis(self):
        """A 40 × 120 scan gets boxes that are wide and short, not square."""
        (rows, columns), *_ = corners.corner_slices(NAV, 0.1)
        assert rows.stop - rows.start == 4
        assert columns.stop - columns.start == 12

    def test_the_four_blocks_are_the_four_corners(self):
        ny, nx = NAV
        blocks = corners.corner_slices(NAV, 0.1)
        assert [(r.start, c.start) for r, c in blocks] == [
            (0, 0), (0, nx - 12), (ny - 4, 0), (ny - 4, nx - 12)]

    def test_corner_slice_indexes_the_same_four(self):
        """One corner at a time has to agree with all four at once, or a
        per-corner extent would move a different box from the one drawn."""
        for corner, block in enumerate(corners.corner_slices(NAV, 0.1)):
            assert corners.corner_slice(NAV, corner, 0.1) == block

    def test_a_tiny_fraction_still_gives_a_fittable_block(self):
        """Two planes of three parameters need 6 points; 1×1 boxes give 4."""
        (rows, columns), *_ = corners.corner_slices((8, 8), 0.001)
        assert rows.stop - rows.start == corners.MIN_CORNER_PX
        assert columns.stop - columns.start == corners.MIN_CORNER_PX

    def test_corner_boxes_are_the_slices_as_rectangles(self):
        boxes = corners.corner_boxes(NAV, 0.1)
        blocks = corners.corner_slices(NAV, 0.1)
        for (x, y, w, h), (rows, columns) in zip(boxes, blocks):
            assert (x, y) == (columns.start, rows.start)
            assert (w, h) == (columns.stop - columns.start,
                              rows.stop - rows.start)


class TestPerCornerExtents:
    """Each corner sized on its own — the multi-angle loader's addition.

    A scan whose top-left sits on vacuum and whose bottom-right sits on the
    sample needs two different boxes, and one fraction for all four cannot say
    that.
    """

    def test_one_corner_resizes_without_moving_the_others(self):
        wide = corners.corner_slice(NAV, 3, 0.25)
        for corner in range(3):
            assert corners.corner_slice(NAV, corner, 0.1) == \
                corners.corner_slices(NAV, 0.1)[corner]
        rows, columns = wide
        assert rows.stop - rows.start == 10
        assert columns.stop - columns.start == 30

    def test_a_bottom_corner_grows_upwards_and_stays_in_the_corner(self):
        """It is the CORNER that is fixed, not the block's origin — a box that
        grew downwards off the bottom edge would clip to a different shape."""
        ny, nx = NAV
        for extent in (0.05, 0.2, 0.5):
            rows, columns = corners.corner_slice(NAV, 3, extent)
            assert rows.stop == ny
            assert columns.stop == nx

    def test_an_extent_is_clipped_rather_than_refused(self):
        """``corner_slice`` is geometry, not validation: a caller that has
        already vetted its number must not have to vet it twice, and a
        traceback from a slider is worse than a clamped box."""
        rows, columns = corners.corner_slice(NAV, 0, 4.0)
        assert rows.stop - rows.start == NAV[0] // 2
        assert columns.stop - columns.start == NAV[1] // 2


class TestThePlane:
    def test_it_recovers_a_planted_ramp(self):
        ny, nx = 20, 30
        rows, columns = np.mgrid[0:ny, 0:nx]
        planted = np.stack([0.5 * columns - 0.25 * rows + 3.0,
                            -0.125 * columns + 0.75 * rows - 1.0], axis=-1)
        fitted = corners.plane_through(planted, None)
        assert np.allclose(fitted, planted)

    def test_it_fits_each_component_independently(self):
        """x and y are two separate planes; a shared fit would smear one
        component's ramp into the other's."""
        ny, nx = 12, 12
        rows, columns = np.mgrid[0:ny, 0:nx]
        planted = np.stack([columns.astype(float),
                            np.zeros((ny, nx))], axis=-1)
        fitted = corners.plane_through(planted, None)
        assert np.allclose(fitted[..., 1], 0.0)

    def test_it_ignores_the_points_it_is_told_to(self):
        """The middle of the scan is the sample; only the corners are the
        instrument. A fit that used both would subtract the signal."""
        ny, nx = 20, 20
        planted = np.full((ny, nx, 2), 2.0)
        planted[5:15, 5:15] = 100.0          # the "sample"
        fitted = corners.plane_through(planted, corners.corner_mask(
            (ny, nx), 0.2))
        assert np.allclose(fitted, 2.0)

    def test_nan_is_ignored_without_a_mask(self):
        """How the multi-angle loader fits four corner MEASUREMENTS: it fills
        the blocks and leaves the rest NaN rather than building a mask."""
        ny, nx = 16, 16
        sparse = np.full((ny, nx, 2), np.nan)
        for corner in range(4):
            rows, columns = corners.corner_slice((ny, nx), corner, 0.2)
            sparse[rows, columns] = (7.0, -3.0)
        fitted = corners.plane_through(sparse, None)
        assert np.allclose(fitted[..., 0], 7.0)
        assert np.allclose(fitted[..., 1], -3.0)

    def test_too_few_points_falls_back_to_their_mean(self):
        """A flat reference is a defensible reference; a singular-matrix
        traceback in the middle of a wizard is not."""
        values = np.full((6, 6, 2), np.nan)
        values[0, 0] = (4.0, 8.0)
        values[0, 1] = (6.0, 12.0)
        fitted = corners.plane_through(values, None)
        assert np.allclose(fitted[..., 0], 5.0)
        assert np.allclose(fitted[..., 1], 10.0)

    def test_nothing_finite_gives_zeros(self):
        fitted = corners.plane_through(np.full((4, 4, 2), np.nan), None)
        assert np.array_equal(fitted, np.zeros((4, 4, 2)))


class TestCornerReference:
    def test_it_is_the_plane_through_the_corner_blocks(self):
        ny, nx = 24, 24
        rows, columns = np.mgrid[0:ny, 0:nx]
        ramp = np.stack([0.25 * columns, 0.5 * rows], axis=-1)
        measured = ramp.copy()
        measured[8:16, 8:16] += 50.0          # the sample, off the corners
        assert np.allclose(corners.corner_reference(measured, 0.2), ramp)

    def test_the_corners_and_the_mask_come_from_one_source(self):
        """The boxes a user SEES are exactly the pixels the plane is fitted
        to — the whole reason the mask is built here rather than delegated."""
        mask = corners.corner_mask(NAV, 0.1)
        inside = np.zeros(NAV, dtype=bool)
        for rows, columns in corners.corner_slices(NAV, 0.1):
            inside[rows, columns] = True
        assert np.array_equal(mask, ~inside)

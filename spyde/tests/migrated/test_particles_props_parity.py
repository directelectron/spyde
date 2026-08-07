"""
Per-column parity between the vectorised property path and ``regionprops_table``.

:mod:`spyde.particles.props` and :mod:`spyde.particles.hull` replaced skimage's
``regionprops_table`` inside :func:`spyde.particles.measure.measure_frame` for one
reason — it costs 53.5 s on a real 4096² frame against 3.0 s to segment it, because
its cost is per REGION and a real frame has 26 566 of them (``benchmarks.md``).

**A faster measurement that quietly changes what a particle's properties ARE is a
regression, not an optimisation.** So the replacement is only defensible if every
column is checked against the implementation it replaced, on a raster with
THOUSANDS of irregular, concave, touching, edge-clipped regions — not on three
discs, which agree under any implementation and prove nothing.

That is what this module is: one scene of ~5 000 blobs from thresholded noise (so
the shapes are ragged and concave, and plenty of them run off the frame edge), and
one assertion per column. The table below states the tolerance actually ASSERTED
in ``_TOL`` — the number that gates CI — with the value measured on this module's
own scene in parentheses. They are not the same number on purpose: the gate is
kept well clear of the measured floor (roughly 1e3-1e5x looser) so that a
platform-dependent BLAS/LAPACK build, which can shift summation order without
being wrong, does not turn into a flake. What the gap is NOT is agreement to the
measured figure — read the parenthetical as "this is what we currently see", not
as a second, tighter contract:

=========================  =========================================
column                     agreement
=========================  =========================================
label, area, bbox-*        exact (integers)
centroid-0, centroid-1     exact (both sum exact integers in float64)
equivalent_diameter_area   exact (a function of ``area`` alone)
solidity                   exact (the hull is integer arithmetic)
perimeter                  gated at 1e-12 relative (measured ~2e-16;
                           same weights, summed in a different order)
major_axis_length          gated at 1e-11 relative (measured ~9e-16;
minor_axis_length          gated at 1e-11 relative (measured ~3e-15;
                           same 2x2 LAPACK eigenproblem, moments
                           summed in a different order)
eccentricity               gated at 1e-9 relative (measured ~1e-14; same
                           eigenproblem above, but eccentricity divides
                           by axis length so it amplifies that error)
=========================  =========================================

The same scene is used to pin ``measure_frame``'s own output rows, because the
columns feed calibration and circularity and a per-column check alone would not
catch a mis-wiring between the two.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.particles.measure import measure_frame, property_table
from spyde.signals.particles import COL, COLUMNS, N_COLUMNS

#: Columns that must agree to the BIT, and why (see the module docstring).
_EXACT = ("label", "area", "bbox-0", "bbox-1", "bbox-2", "bbox-3",
          "centroid-0", "centroid-1", "equivalent_diameter_area", "solidity")

#: Columns that differ only in floating-point summation order. Relative, and
#: three orders of magnitude tighter than any measurement this feeds.
_TOL = {"perimeter": 1e-12,
        "major_axis_length": 1e-11,
        "minor_axis_length": 1e-11,
        "eccentricity": 1e-9}


def _blob_field(size=1024, n_particles=4000, r_max=3, seed=0):
    """Thousands of ragged, concave, touching, edge-clipped regions.

    Each particle is the union of one to three overlapping discs, which is what
    makes the scene worth testing on: a single disc is CONVEX, so its solidity is
    ~1 and its hull is uninteresting, and a field of discs would let a wrong
    convex-hull implementation pass. Lobed unions are concave, they agglomerate
    where they overlap, and some run off the frame edge — the three places where a
    per-region crop, an erosion border value and the hull's ±0.5 offsets can each
    be wrong on their own.

    The defaults are tuned to the REAL frame this replaces ``regionprops_table``
    for: ~3 260 regions of mean area 30 px (real: 26 566 of mean 33 px), solidity
    spanning ~0.60-1.0 (real: 0.28-0.92).
    """
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size), bool)

    def _disc(r):
        y, x = np.mgrid[-r:r + 1, -r:r + 1]
        return y * y + x * x <= r * r

    discs = {r: _disc(r) for r in range(1, r_max + 1)}
    for _ in range(n_particles):
        cy = int(rng.integers(-2, size + 2))
        cx = int(rng.integers(-2, size + 2))
        for _lobe in range(int(rng.integers(1, 4))):
            r = int(rng.integers(1, r_max + 1))
            d = discs[r]
            y0, x0 = cy + int(rng.integers(-r, r + 1)) - r, \
                cx + int(rng.integers(-r, r + 1)) - r
            y1, x1 = y0 + 2 * r + 1, x0 + 2 * r + 1
            sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(size, y1), min(size, x1)
            if sy0 >= sy1 or sx0 >= sx1:
                continue
            canvas[sy0:sy1, sx0:sx1] |= d[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]

    lab, n = ndi.label(canvas)
    assert n > 2000, f"scene has only {n} regions; the parity check needs thousands"
    img = canvas.astype(np.float32) + 0.1 * rng.standard_normal(canvas.shape).astype(np.float32)
    return lab.astype(np.int32), img


@pytest.fixture(scope="module")
def scene():
    return _blob_field()


@pytest.fixture(scope="module")
def tables(scene):
    """Both property tables, computed ONCE — the legacy one is seconds even here."""
    lab, _img = scene
    return property_table(lab, fast=False), property_table(lab, fast=True)


class TestPropertyTableParity:
    def test_scene_is_actually_hard(self, scene, tables):
        """Guard the guard: a scene of near-convex blobs would not test the hull."""
        lab, _img = scene
        tbl = tables[0]
        assert len(tbl["label"]) > 2000
        # Genuinely concave shapes present, not a field of discs.
        assert tbl["solidity"].min() < 0.7
        assert np.median(tbl["solidity"]) < 0.95
        # Regions clipped by the frame edge.
        assert (tbl["bbox-0"] == 0).any() and (tbl["bbox-2"] == lab.shape[0]).any()

    def test_every_column_matches_regionprops(self, tables):
        ref, got = tables

        assert set(got) == set(ref)
        for key in sorted(ref):
            a = np.asarray(ref[key], np.float64)
            b = np.asarray(got[key], np.float64)
            assert a.shape == b.shape, key
            if key in _EXACT:
                assert np.array_equal(a, b), (
                    f"{key} must match regionprops exactly; "
                    f"max |diff| = {np.abs(a - b).max()}")
            else:
                tol = _TOL[key]
                rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
                assert rel.max() < tol, (
                    f"{key} differs by {rel.max():.3e} relative (tolerance "
                    f"{tol:.0e}) — that is more than summation order")

    def test_solidity_is_the_same_convex_hull(self, scene):
        """``area_convex`` is a pixel count, so 'close' is not the bar — the hull
        must select the SAME pixels. This is the column that was rewritten from
        Qhull to integer arithmetic, so it gets its own assertion."""
        from spyde.particles.hull import convex_areas
        from skimage.measure import regionprops_table

        lab, _img = scene
        counts = np.bincount(lab.reshape(-1))
        labels = (np.flatnonzero(counts[1:] > 0) + 1).astype(np.int64)
        got = convex_areas(lab, labels, counts)
        if got is None:
            pytest.skip("numba unavailable; solidity falls back to regionprops")
        ref = regionprops_table(lab, properties=("area_convex",))["area_convex"]
        assert np.array_equal(got, ref.astype(np.int64))


class TestSparseLabels:
    """A label image whose values are not 1..N.

    ``regionprops_table`` skips absent labels, so the vectorised path has to as
    well — and the perimeter's per-label histogram has FIFTY bins per label, so
    keying it by the raw label value would ask for 50x the label range in memory.
    Both are pinned here rather than left to a frame that happens to be dense.
    """

    def test_matches_regionprops_with_gaps(self, scene):
        lab, _img = scene
        sparse = np.where(lab > 0, lab.astype(np.int64) * 3 + 7, 0).astype(np.int32)
        ref = property_table(sparse, fast=False)
        got = property_table(sparse, fast=True)
        assert len(ref["label"]) == len(got["label"]) > 2000
        assert np.array_equal(ref["label"], got["label"])
        assert np.array_equal(ref["area"], got["area"])
        assert np.array_equal(ref["solidity"], got["solidity"])
        assert np.allclose(ref["perimeter"], got["perimeter"], rtol=1e-12, atol=0)


class TestMeasureFrameParity:
    def test_rows_match_between_paths(self, scene):
        lab, img = scene
        rows_l, cont_l = measure_frame(lab, img, t=3, scale=0.25, fast=False)
        rows_f, cont_f = measure_frame(lab, img, t=3, scale=0.25, fast=True)
        assert rows_l.shape == rows_f.shape

        loose = {"major_axis", "minor_axis", "eccentricity", "perimeter",
                 "circularity"}
        for i, name in enumerate(COLUMNS[:N_COLUMNS]):
            a, b = rows_l[:, i], rows_f[:, i]
            assert np.array_equal(np.isnan(a), np.isnan(b)), name
            fin = ~np.isnan(a)
            if name in loose:
                # float32 rows, so the comparison is at float32 resolution.
                assert np.allclose(a[fin], b[fin], rtol=1e-6, atol=0), name
            else:
                assert np.array_equal(a[fin], b[fin]), name

        # Outlines are compared by the FILLED polygon, not by vertex identity —
        # the vectorised tracer cuts a closed contour at a different vertex, and
        # everything downstream (`render_frame`, `mask_at`) fills it.
        # `test_particles_contours_parity.py` is where that gate lives; this is
        # the wiring check that `measure_frame` returns one per kept row.
        from spyde.tests.migrated.test_particles_contours_parity import (
            filled_pixels)

        assert len(cont_l) == len(cont_f) == rows_f.shape[0]
        for x, y in zip(cont_l, cont_f):
            assert np.array_equal(filled_pixels(x, lab.shape),
                                  filled_pixels(y, lab.shape))

    def test_min_area_and_empty_frame(self, scene):
        lab, img = scene
        rows_l, _ = measure_frame(lab, img, min_area_px=25, fast=False)
        rows_f, _ = measure_frame(lab, img, min_area_px=25, fast=True)
        assert rows_l.shape == rows_f.shape and rows_f.shape[0] > 100
        assert np.array_equal(rows_l[:, COL["area"]], rows_f[:, COL["area"]])

        empty = np.zeros((16, 16), np.int32)
        for fast in (False, True):
            rows, cont = measure_frame(empty, fast=fast)
            assert rows.shape == (0, N_COLUMNS) and cont == []


class TestHullEdgeCases:
    """Shapes whose hull is degenerate or whose pixels touch the frame edge —
    the cases where an integer reimplementation of Qhull is most likely to
    disagree, and where ``regionprops`` itself is at its least obvious."""

    @pytest.mark.parametrize("build", [
        pytest.param(lambda a: a.__setitem__((5, 5), 1), id="single-pixel"),
        pytest.param(lambda a: a.__setitem__((5, slice(2, 9)), 1), id="h-line"),
        pytest.param(lambda a: a.__setitem__((slice(2, 9), 5), 1), id="v-line"),
        pytest.param(lambda a: a.__setitem__((0, 0), 1), id="corner"),
        pytest.param(lambda a: a.__setitem__((slice(0, 3), slice(0, 3)), 1),
                     id="corner-block"),
        pytest.param(lambda a: [a.__setitem__((i, i), 1) for i in range(2, 9)],
                     id="diagonal"),
        pytest.param(lambda a: [a.__setitem__((slice(2, 9), 2), 1),
                                a.__setitem__((2, slice(2, 9)), 1)], id="L"),
    ])
    def test_matches_regionprops(self, build):
        from skimage.measure import regionprops_table
        from spyde.particles.hull import convex_areas

        lab = np.zeros((12, 12), np.int32)
        build(lab)
        counts = np.bincount(lab.reshape(-1))
        labels = (np.flatnonzero(counts[1:] > 0) + 1).astype(np.int64)
        got = convex_areas(lab, labels, counts)
        if got is None:
            pytest.skip("numba unavailable")
        ref = regionprops_table(lab, properties=("area_convex",))["area_convex"]
        assert np.array_equal(got, ref.astype(np.int64))

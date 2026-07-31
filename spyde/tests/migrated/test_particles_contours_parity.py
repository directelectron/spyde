"""
Parity for the two per-region loops that were left in ``measure_frame``.

:mod:`spyde.particles.props` and :mod:`spyde.particles.hull` took the property
table from 43.7 s to 1.1 s on a real 4096² frame with 26 566 particles. What
remained was **9.4 s of two Python ``for`` loops** — ``_fill_intensity`` and
``_contours`` — which is 92% of the measurement and, like ``regionprops_table``
before them, holds the GIL throughout, so a dask worker's four task slots stay
worth one core (``benchmarks.md``). :mod:`spyde.particles.intensity` and
:mod:`spyde.particles.contours` replace them. This module is the gate.

The two halves need DIFFERENT gates, and getting that wrong in either direction
is the trap
---------------------------------------------------------------------------
* **The intensity columns are exact and are asserted exactly.** The pixel sets
  are identical by construction — same crop, same finite-only filter — so the
  only freedom is floating-point summation order, and at the float32 resolution
  the rows are stored in, that is nine orders below the last bit. Every intensity
  column is asserted ``array_equal`` on the stored rows, and to ~1e-12 relative
  on the float64 intermediates.

* **The outlines are NOT asserted vertex by vertex, and it would be wrong to.**
  A closed marching-squares contour is a CYCLE; ``skimage``'s dict-and-deque
  assembly and a ``succ``-following walk cut the same cycle at different
  vertices, so the arrays differ by a rotation while describing the same shape.
  Demanding bit-identical vertices would reject a correct implementation.

  It would be equally wrong to conclude from that that outlines are cosmetic and
  a "close enough" polygon will do.
  :meth:`~spyde.signals.particles.SpyDEParticles.render_frame` FILLS them to
  rebuild the label movie, and :meth:`~…SpyDEParticles.mask_at` fills one to
  produce the per-particle mask a mean diffraction pattern is sliced with — a
  different contour is a different mask is a different measurement. So the gate
  is the thing those two consume, and nothing weaker:

      **``skimage.draw.polygon`` on the new outline must select EXACTLY the same
      pixels as on the old one, for every region.** A boolean set equality, not a
      tolerance, not an IoU.

  Asserted here per region on a scene of thousands, and separately verified on
  the real 26 566-particle frame (``benchmarks.md``).

The scene is :func:`~spyde.tests.migrated.test_particles_props_parity._blob_field`
— thousands of ragged, concave, touching, edge-clipped regions — for the reason
that module gives: three discs agree under any implementation and prove nothing.
Regions that run off the frame edge matter twice over here, because that is where
a contour is left OPEN and where the dilation that defines the background ring is
truncated.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.particles.measure import _contours, _fill_intensity, property_table
from spyde.signals.particles import COL, N_COLUMNS
from spyde.tests.migrated.test_particles_props_parity import _blob_field

#: The intensity columns, and the reason each is exact rather than close.
_INTENSITY_COLS = ("intensity_mean", "intensity_max", "intensity_std",
                   "background")


def filled_pixels(contour, shape) -> np.ndarray:
    """The pixels ``skimage.draw.polygon`` selects, as a sorted flat index.

    This IS what ``render_frame`` and ``mask_at`` compute, including their
    ``len(c) < 3`` skip. Sorted, because the fill's output ORDER follows the
    vertex order and two rotations of one cycle may enumerate the same pixels in
    a different sequence — which no consumer can observe.
    """
    from skimage.draw import polygon as sk_polygon

    c = np.asarray(contour)
    if len(c) < 3:
        return np.zeros(0, np.int64)
    rr, cc = sk_polygon(c[:, 0].astype(np.intp), c[:, 1].astype(np.intp),
                        shape=shape)
    flat = rr.astype(np.int64) * int(shape[1]) + cc.astype(np.int64)
    flat.sort()
    return flat


def filled_mask(contour, shape) -> np.ndarray:
    """:func:`filled_pixels` as a boolean image. For assertions that want one."""
    m = np.zeros(int(shape[0]) * int(shape[1]), bool)
    m[filled_pixels(contour, shape)] = True
    return m.reshape(shape)


@pytest.fixture(scope="module")
def scene():
    return _blob_field()


@pytest.fixture(scope="module")
def table(scene):
    lab, _img = scene
    return property_table(lab, fast=True)


class TestContourFillParity:
    """The filled polygon, per region, against the ``find_contours`` loop."""

    @pytest.fixture(scope="class")
    def both(self, scene, table):
        lab, _img = scene
        fast = _contours(lab, table, fast=True)
        if fast is None:                                      # pragma: no cover
            pytest.skip("numba unavailable; contours stay on find_contours")
        return _contours(lab, table, fast=False), fast

    def test_one_outline_per_region(self, both, table):
        ref, got = both
        assert len(got) == len(ref) == len(table["label"]) > 2000

    def test_filled_polygon_is_identical_per_region(self, both, scene):
        """The gate. Every region, exact pixel-set equality, no tolerance."""
        lab, _img = scene
        ref, got = both
        differing = [i for i in range(len(ref))
                     if not np.array_equal(filled_pixels(ref[i], lab.shape),
                                           filled_pixels(got[i], lab.shape))]
        assert not differing, (
            f"{len(differing)}/{len(ref)} regions fill to a different pixel set; "
            f"first at index {differing[:5]}")

    def test_vertex_count_matches_and_the_cycle_is_the_same(self, both):
        """Stronger than the gate and not required by it, but it is TRUE, and
        it is what says the tracer found the same contour rather than a
        different one that happens to fill the same.

        A closed contour repeats its first vertex last, so 'the same cycle' means
        equal after dropping that and rotating."""
        ref, got = both
        assert [len(c) for c in ref] == [len(c) for c in got]
        rotations = 0
        for a, b in zip(ref, got):
            a = np.asarray(a, np.int64)
            b = np.asarray(b, np.int64)
            closed = len(a) > 1 and np.array_equal(a[0], a[-1])
            if not closed:
                assert np.array_equal(a, b)      # open paths are bit-identical
                continue
            ca, cb = a[:-1], b[:-1]
            assert np.array_equal(cb[0], cb[-1]) is False or len(cb) == 1
            hits = [s for s in range(len(cb))
                    if np.array_equal(np.roll(cb, -s, axis=0), ca)]
            assert hits, "closed contour is not a rotation of the reference"
            rotations += 1
        assert rotations > 100, "scene has too few closed contours to be a test"

    def test_int16_csr_layout_still_holds(self, both):
        """``SpyDEParticles`` stores ``contours`` + ``contour_offsets`` as one
        int16 ``(N, 2)`` block (``particle_overlay.add_particles`` concatenates
        and re-slices it), so an outline that is not int16 ``(k, 2)`` breaks the
        store rather than the drawing."""
        _ref, got = both
        for c in got:
            assert c.dtype == np.int16 and c.ndim == 2 and c.shape[1] == 2
        pool = np.concatenate(got, axis=0)
        assert pool.dtype == np.int16
        offsets = np.concatenate([[0], np.cumsum([len(c) for c in got])])
        assert offsets[-1] == len(pool)


class TestIntensityParity:
    """The four intensity columns, exact, against the per-region crop loop."""

    @pytest.fixture(scope="class")
    def rows(self, scene, table):
        lab, img = scene
        inten = np.asarray(img, np.float64)
        n = len(table["label"])
        keep = np.ones(n, bool)

        def run(fast):
            r = np.zeros((n, N_COLUMNS), np.float32)
            for name in _INTENSITY_COLS:
                r[:, COL[name]] = np.nan
            _fill_intensity(r, lab, inten, table, keep, 3, fast=fast)
            return r

        return run(False), run(True)

    @pytest.mark.parametrize("name", _INTENSITY_COLS)
    def test_column_is_exact(self, rows, name):
        ref, got = rows
        a, b = ref[:, COL[name]], got[:, COL[name]]
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: NaN pattern"
        fin = ~np.isnan(a)
        assert fin.sum() > 2000, f"{name}: nothing measured, the test is vacuous"
        assert np.array_equal(a[fin], b[fin]), (
            f"{name}: max |diff| = {np.abs(a[fin] - b[fin]).max()}")

    def test_float64_intermediates_agree_to_summation_order(self, scene, table):
        """The stored columns are float32, so 'exact' there could in principle
        hide a real difference. This checks the float64 values the kernels
        actually produce."""
        from spyde.particles.intensity import (label_intensity_stats,
                                               ring_backgrounds)

        lab, img = scene
        inten = np.asarray(img, np.float64)
        labels = np.asarray(table["label"], np.int64)
        bb = np.stack([np.asarray(table[f"bbox-{k}"], np.int64)
                       for k in range(4)], axis=1)
        mean, mx, std = label_intensity_stats(lab, inten, labels)
        bg = ring_backgrounds(lab, inten, labels, bb, 3)
        if bg is None:                                        # pragma: no cover
            pytest.skip("numba unavailable")

        from scipy.ndimage import binary_dilation
        h, w = lab.shape
        # A subset is enough at float64 resolution and keeps the reference loop
        # (which is the slow path by construction) off the suite's critical path.
        for i in range(0, len(labels), 7):
            lbl = int(labels[i])
            y0, x0 = int(bb[i, 0]), int(bb[i, 1])
            y1, x1 = int(bb[i, 2]), int(bb[i, 3])
            py0, px0 = max(0, y0 - 4), max(0, x0 - 4)
            py1, px1 = min(h, y1 + 4), min(w, x1 + 4)
            sub_lab = lab[py0:py1, px0:px1]
            sub_int = inten[py0:py1, px0:px1]
            m = sub_lab == lbl
            vals = sub_int[m]
            vals = vals[np.isfinite(vals)]
            assert vals.size
            assert mean[i] == pytest.approx(vals.mean(), rel=1e-12, abs=0)
            assert mx[i] == vals.max()
            assert std[i] * vals.max() == pytest.approx(vals.std(), rel=1e-11,
                                                        abs=0)
            ring = binary_dilation(m, iterations=3) & ~m & (sub_lab == 0)
            bvals = sub_int[ring]
            bvals = bvals[np.isfinite(bvals)]
            if bvals.size:
                assert bg[i] == pytest.approx(bvals.mean(), rel=1e-12, abs=0)
            else:
                assert np.isnan(bg[i])

    def test_nan_pixels_are_excluded_not_coerced(self, scene, table):
        """A drift-corrected frame has a NaN-padded border. Letting NaN reach a
        plain mean makes every particle touching it report NaN; coercing it to
        zero invents a dark rim. Both paths must do neither."""
        lab, img = scene
        inten = np.asarray(img, np.float64)
        inten[:8, :] = np.nan
        inten[:, :8] = np.nan
        n = len(table["label"])
        keep = np.ones(n, bool)
        out = []
        for fast in (False, True):
            r = np.zeros((n, N_COLUMNS), np.float32)
            for name in _INTENSITY_COLS:
                r[:, COL[name]] = np.nan
            _fill_intensity(r, lab, inten, table, keep, 3, fast=fast)
            out.append(r)
        ref, got = out
        touching = np.asarray(table["bbox-0"]) < 8
        assert touching.sum() > 10, "no region touches the NaN border"
        for name in _INTENSITY_COLS:
            a, b = ref[:, COL[name]], got[:, COL[name]]
            assert np.array_equal(np.isnan(a), np.isnan(b)), name
            fin = ~np.isnan(a)
            assert np.array_equal(a[fin], b[fin]), name
        # A region entirely inside the NaN band has no finite pixel at all.
        assert np.isnan(got[:, COL["intensity_mean"]]).any()

    def test_ring_zero_leaves_background_unset(self, scene, table):
        lab, img = scene
        inten = np.asarray(img, np.float64)
        n = len(table["label"])
        r = np.zeros((n, N_COLUMNS), np.float32)
        r[:, COL["background"]] = np.nan
        _fill_intensity(r, lab, inten, table, np.ones(n, bool), 0, fast=True)
        assert np.isnan(r[:, COL["background"]]).all()
        assert not np.isnan(r[:, COL["intensity_mean"]]).all()


class TestKernelEdgeCases:
    """Shapes where an integer reimplementation is most likely to disagree."""

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
        pytest.param(lambda a: [a.__setitem__((slice(2, 9), slice(2, 9)), 1),
                                a.__setitem__((slice(4, 7), slice(4, 7)), 0)],
                     id="ring-with-hole"),
        pytest.param(lambda a: a.__setitem__((slice(0, 12), slice(0, 12)), 1),
                     id="fills-the-frame"),
        pytest.param(lambda a: a.__setitem__((slice(0, 12), 0), 1),
                     id="full-left-edge"),
    ])
    def test_contour_fill_and_intensity_match(self, build):
        lab = np.zeros((12, 12), np.int32)
        build(lab)
        rng = np.random.default_rng(3)
        inten = rng.standard_normal(lab.shape) + 5.0
        tbl = property_table(lab, fast=True)
        n = len(tbl["label"])

        ref = _contours(lab, tbl, fast=False)
        got = _contours(lab, tbl, fast=True)
        if got is None:                                       # pragma: no cover
            pytest.skip("numba unavailable")
        for i in range(n):
            assert np.array_equal(filled_pixels(ref[i], lab.shape),
                                  filled_pixels(got[i], lab.shape)), i

        out = []
        for fast in (False, True):
            r = np.zeros((n, N_COLUMNS), np.float32)
            for name in _INTENSITY_COLS:
                r[:, COL[name]] = np.nan
            _fill_intensity(r, lab, inten, tbl, np.ones(n, bool), 3, fast=fast)
            out.append(r)
        for name in _INTENSITY_COLS:
            a, b = out[0][:, COL[name]], out[1][:, COL[name]]
            assert np.array_equal(np.isnan(a), np.isnan(b)), name
            fin = ~np.isnan(a)
            assert np.array_equal(a[fin], b[fin]), name

    def test_sparse_labels(self, scene):
        """Label values that are not 1..N — every upstream filter re-tags, and
        both kernels index BY ROW, not by label value."""
        lab, img = scene
        sparse = np.where(lab > 0, lab.astype(np.int64) * 3 + 7, 0).astype(np.int32)
        tbl = property_table(sparse, fast=True)
        n = len(tbl["label"])
        assert n > 2000

        ref = _contours(sparse, tbl, fast=False)
        got = _contours(sparse, tbl, fast=True)
        if got is None:                                       # pragma: no cover
            pytest.skip("numba unavailable")
        bad = sum(1 for i in range(n)
                  if not np.array_equal(filled_pixels(ref[i], sparse.shape),
                                        filled_pixels(got[i], sparse.shape)))
        assert bad == 0

        inten = np.asarray(img, np.float64)
        out = []
        for fast in (False, True):
            r = np.zeros((n, N_COLUMNS), np.float32)
            for name in _INTENSITY_COLS:
                r[:, COL[name]] = np.nan
            _fill_intensity(r, sparse, inten, tbl, np.ones(n, bool), 3, fast=fast)
            out.append(r)
        for name in _INTENSITY_COLS:
            a, b = out[0][:, COL[name]], out[1][:, COL[name]]
            fin = ~np.isnan(a)
            assert np.array_equal(a[fin], b[fin]), name

    def test_empty_frame(self):
        lab = np.zeros((16, 16), np.int32)
        tbl = property_table(lab, fast=True)
        assert len(tbl["label"]) == 0
        assert _contours(lab, tbl, fast=True) == []


class TestNumbaUnavailable:
    """Both kernels are OPTIONAL, and the machine without numba must still be
    able to measure a frame.

    Distinct from ``fast=False``: that asks for the legacy path, this asks for
    the fast one and has it refuse. The half that does not need numba
    (``bincount`` statistics) must NOT be left half-written when the half that
    does is unavailable — a partly-filled row is worse than a slow one."""

    @pytest.fixture
    def no_numba(self, monkeypatch):
        from spyde.particles import contours as cmod
        from spyde.particles import intensity as imod
        monkeypatch.setattr(cmod, "_build_kernel", lambda: None)
        monkeypatch.setattr(imod, "_build_ring_kernel", lambda: None)

    def test_measure_frame_still_matches(self, scene, no_numba):
        from spyde.particles.measure import measure_frame

        lab, img = scene
        rows_f, cont_f = measure_frame(lab, img, t=2, scale=0.5, fast=True)
        rows_l, cont_l = measure_frame(lab, img, t=2, scale=0.5, fast=False)
        assert rows_f.shape == rows_l.shape and rows_f.shape[0] > 2000
        for name in _INTENSITY_COLS:
            a, b = rows_l[:, COL[name]], rows_f[:, COL[name]]
            assert np.array_equal(np.isnan(a), np.isnan(b)), name
            fin = ~np.isnan(a)
            assert np.array_equal(a[fin], b[fin]), name
        # Without numba the fallback IS `find_contours`, so these are identical
        # vertex for vertex, not merely equal when filled.
        assert all(np.array_equal(x, y) for x, y in zip(cont_l, cont_f))

    def test_helpers_report_unavailable_rather_than_guessing(self, scene,
                                                             no_numba):
        from spyde.particles.contours import label_contours
        from spyde.particles.intensity import ring_backgrounds

        lab, _img = scene
        tbl = property_table(lab, fast=True)
        labels = np.asarray(tbl["label"], np.int64)
        bb = np.stack([np.asarray(tbl[f"bbox-{k}"], np.int64)
                       for k in range(4)], axis=1)
        assert label_contours(lab, labels, bb,
                              np.asarray(tbl["area"], np.int64)) is None
        assert ring_backgrounds(lab, np.zeros(lab.shape), labels, bb, 3) is None

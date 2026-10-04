"""
The strain fit weights each vector by its confidence and positional sigma
(`strain_mapping.vector_weights`) when Find Vectors recorded them, and fits
vectors without them exactly as before.
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

from spyde.actions.strain_mapping import (
    WEIGHT_FLOOR_PX, _compute_strain_field_loop, _median_nn, compute_strain_field,
    vector_weights,
)
from spyde.signals.diffraction_vectors import (
    COL_CONFIDENCE, COL_SIGMA, LEGACY_N_COLS, N_COLS, SpyDEDiffractionVectors,
)

# A 7 x 7 reciprocal lattice, 20 px spacing: 48 reflections per pattern, so a
# pixel's affine fit is well over-determined.
G_REF = 20.0 * np.array([[i, j] for i in range(-3, 4) for j in range(-3, 4) if (i, j) != (0, 0)], float)
GOOD = dict(noise=0.05, sigma=0.05, confidence=0.95)
POOR = dict(noise=1.5, sigma=1.5, confidence=0.3)


def _strain(iy, ix):
    """Known symmetric (rotation-free) deformation, so strain = F - I."""
    rng = np.random.default_rng(1000 * iy + ix)
    a, b, c = rng.uniform(-0.01, 0.01, 3)
    return np.array([[1.0 + a, b], [b, 1.0 + c]])


def _vectors(ny=6, nx=6, *, columns=N_COLS, confidence=True, seed=0):
    """Every pattern: half its reflections located well, half poorly — the
    confidence and sigma columns say which (NaN when ``confidence`` is False)."""
    rng = np.random.default_rng(seed)
    rows, offsets = [], [0]
    for iy in range(ny):
        for ix in range(nx):
            T = np.linalg.inv(_strain(iy, ix)).T
            g = G_REF @ T.T
            poor = rng.random(len(g)) < 0.5
            for (kx, ky), is_poor in zip(g, poor):
                kind = POOR if is_poor else GOOD
                kx, ky = np.array([kx, ky]) + rng.normal(0.0, kind["noise"], 2)
                row = [ix, iy, kx, ky, -1.0, 1.0]
                if columns > LEGACY_N_COLS:
                    row += ([kind["confidence"], kind["sigma"]] if confidence else [np.nan, np.nan])
                rows.append(row)
            offsets.append(len(rows))
    flat = np.asarray(rows, dtype=np.float32)
    off = np.asarray(offsets, dtype=np.int64)
    return SpyDEDiffractionVectors(
        flat_buffer=flat, nav_offsets=[np.arange(ny + 1) * nx, off],
        nav_shape=(ny, nx), full_nav_shape=(ny, nx), sig_shape=(256, 256),
        sig_axes=None, kernel_radius_px=1.0, kernel_radius_data=1.0, offsets=off)


def _error(field, ny=6, nx=6):
    truth = np.array([[_strain(iy, ix) - np.eye(2) for ix in range(nx)] for iy in range(ny)])
    d = np.concatenate([(field.exx - truth[..., 0, 0]).ravel(), (field.eyy - truth[..., 1, 1]).ravel(),
                        (field.exy - truth[..., 0, 1]).ravel()])
    return float(np.sqrt(np.nanmean(d ** 2)))


class TestWeightedStrainFit:
    def test_weighting_recovers_known_strain_under_uneven_noise(self):
        """Half the reflections 0.05 px off, half 1.5 px off: the weighted fit
        follows the well-located half (0.025 % RMS strain error vs 0.31 %)."""
        v = _vectors()
        weighted = _error(compute_strain_field(v, ref_vectors=G_REF, weighted=True))
        plain = _error(compute_strain_field(v, ref_vectors=G_REF, weighted=False))
        assert weighted < 0.4 * plain, (weighted, plain)

    def test_per_pixel_and_whole_field_paths_agree(self):
        v = _vectors()
        tol = 0.25 * _median_nn(G_REF)
        whole = compute_strain_field(v, ref_vectors=G_REF, tol=tol)
        per_pixel = _compute_strain_field_loop(v, G_REF, tol, 6, 6)
        for name in ("exx", "eyy", "exy", "omega", "coverage", "residual"):
            np.testing.assert_allclose(getattr(whole, name), getattr(per_pixel, name), atol=1e-6, equal_nan=True,
                                       err_msg=name)

    @pytest.mark.parametrize("columns, confidence", [(LEGACY_N_COLS, False), (N_COLS, False)])
    def test_vectors_without_confidence_fit_exactly_as_before(self, columns, confidence):
        """Six-column vectors from an older file, and vectors from a method that
        records no confidence, give bit-identical fields."""
        v = _vectors(columns=columns, confidence=confidence)
        on = compute_strain_field(v, ref_vectors=G_REF, weighted=True)
        off = compute_strain_field(v, ref_vectors=G_REF, weighted=False)
        for name in ("exx", "eyy", "exy", "omega", "coverage", "residual"):
            assert np.array_equal(getattr(on, name), getattr(off, name), equal_nan=True), name


class TestVectorWeights:
    def test_confidence_over_sigma_squared_with_a_floor(self):
        rows = np.zeros((2, N_COLS))
        rows[:, COL_CONFIDENCE] = [0.9, 0.3]
        rows[:, COL_SIGMA] = [0.01, 0.2]
        w = vector_weights(rows, unit_per_pixel=2.0)
        floor = WEIGHT_FLOOR_PX * 2.0
        np.testing.assert_allclose(w, [0.9 / (0.0001 + floor ** 2), 0.3 / (0.04 + floor ** 2)])

    def test_a_row_without_confidence_gets_the_median_weight(self):
        rows = np.zeros((3, N_COLS))
        rows[:, COL_CONFIDENCE] = [0.9, 0.5, np.nan]
        rows[:, COL_SIGMA] = [0.1, 0.1, np.nan]
        w = vector_weights(rows)
        assert w[2] == pytest.approx(np.median(w[:2]))

    def test_none_when_no_row_has_confidence(self):
        rows = np.full((4, N_COLS), np.nan)
        assert vector_weights(rows) is None
        assert vector_weights(np.zeros((4, LEGACY_N_COLS))) is None
        assert vector_weights(np.zeros((4, N_COLS))) is None   # columns left at zero

    def test_a_pixel_whose_weights_are_all_zero_fits_unweighted(self):
        v = _vectors()
        v.flat_buffer[:, COL_CONFIDENCE] = 0.0
        weighted = compute_strain_field(v, ref_vectors=G_REF, robust=None, one_per_reflection=False)
        plain = compute_strain_field(v, ref_vectors=G_REF, robust=None, one_per_reflection=False, weighted=False)
        np.testing.assert_allclose(weighted.exx, plain.exx, atol=1e-7)


# A wider lattice (40 px) so the match radius (a quarter of it, 10 px) admits
# the 5-10 px outliers the robust loss has to deal with.
WIDE_REF = 2.0 * G_REF


def _contaminated(*, outliers=0, duplicates=0, ny=6, nx=6, seed=0):
    """Well-located patterns (0.05 px) plus, per pattern, ``outliers`` vectors
    5-10 px off their reflection and ``duplicates`` second vectors 2 px off an
    already-found reflection, both at a confidence the network would give a
    real disk."""
    rng = np.random.default_rng(seed)
    rows, offsets = [], [0]
    for iy in range(ny):
        for ix in range(nx):
            T = np.linalg.inv(_strain(iy, ix)).T
            g = WIDE_REF @ T.T
            measured = g + rng.normal(0.0, 0.05, g.shape)
            extra = []
            for k in rng.choice(len(g), outliers, replace=False):
                angle = rng.uniform(0, 2 * np.pi)
                extra.append(g[k] + rng.uniform(5.0, 10.0) * np.array([np.cos(angle), np.sin(angle)]))
            for k in rng.choice(len(g), duplicates, replace=False):
                angle = rng.uniform(0, 2 * np.pi)
                extra.append(g[k] + 2.0 * np.array([np.cos(angle), np.sin(angle)]))
            for kx, ky in list(measured) + extra:
                rows.append([ix, iy, kx, ky, -1.0, 1.0, 0.8, 0.05])
            offsets.append(len(rows))
    flat = np.asarray(rows, dtype=np.float32)
    off = np.asarray(offsets, dtype=np.int64)
    return SpyDEDiffractionVectors(
        flat_buffer=flat, nav_offsets=[np.arange(ny + 1) * nx, off],
        nav_shape=(ny, nx), full_nav_shape=(ny, nx), sig_shape=(512, 512),
        sig_axes=None, kernel_radius_px=1.0, kernel_radius_data=1.0, offsets=off)


class TestRobustStrainFit:
    def test_far_outliers_lose_their_influence(self):
        """Eight vectors per pattern 5-10 px off their reflection, inside the
        match radius: the one-pass trim leaves 0.085 % RMS strain error, Tukey
        0.008 %, Huber 0.010 %. (One vector per reflection is off here: it would
        drop these outliers on its own, since each shares its reflection with
        a good vector.)"""
        v = _contaminated(outliers=8)
        fit = functools.partial(compute_strain_field, v, ref_vectors=WIDE_REF, one_per_reflection=False)
        trimmed, tukey, huber = (_error(fit(robust=loss)) for loss in (None, "tukey", "huber"))
        assert tukey < 0.2 * trimmed and huber < 0.2 * trimmed, (trimmed, tukey, huber)

    def test_two_vectors_on_one_reflection_count_once(self):
        """Ten reflections per pattern found twice, the second 2 px off: counting
        both gives 0.085 % RMS strain error, one per reflection 0.0085 % — the
        clean patterns' 0.0090 %."""
        v = _contaminated(duplicates=10)
        both = _error(compute_strain_field(v, ref_vectors=WIDE_REF, trim=False, robust=None,
                                           one_per_reflection=False))
        one = _error(compute_strain_field(v, ref_vectors=WIDE_REF, trim=False, robust=None))
        clean = _error(compute_strain_field(_contaminated(), ref_vectors=WIDE_REF, trim=False, robust=None))
        assert one < 1.5 * clean < 0.3 * both, (both, one, clean)

    def test_vectors_beyond_the_match_radius_never_enter(self):
        far = _contaminated()
        rows = far.flat_buffer.copy()
        rows[::7, 2] += 15.0                          # past the 10 px match radius
        moved = SpyDEDiffractionVectors(
            flat_buffer=rows, nav_offsets=far.nav_offsets, nav_shape=far.nav_shape,
            full_nav_shape=far.full_nav_shape, sig_shape=far.sig_shape, sig_axes=None,
            kernel_radius_px=1.0, kernel_radius_data=1.0, offsets=far.offsets)
        diagnostics = {}
        compute_strain_field(moved, ref_vectors=WIDE_REF, diagnostics=diagnostics)
        assert not np.isin(np.arange(0, len(rows), 7), diagnostics["row"]).any()

    @pytest.mark.parametrize("one_per_reflection", [False, True])
    @pytest.mark.parametrize("robust", [None, "tukey", "huber"])
    def test_per_pixel_and_whole_field_paths_agree(self, one_per_reflection, robust):
        v = _contaminated(outliers=4, duplicates=6)
        tol = 0.25 * _median_nn(WIDE_REF)
        options = dict(one_per_reflection=one_per_reflection, robust=robust)
        whole = compute_strain_field(v, ref_vectors=WIDE_REF, tol=tol, **options)
        per_pixel = _compute_strain_field_loop(v, WIDE_REF, tol, 6, 6, **options)
        for name in ("exx", "eyy", "exy", "omega", "coverage", "residual"):
            np.testing.assert_allclose(getattr(whole, name), getattr(per_pixel, name), atol=1e-6, equal_nan=True,
                                       err_msg=name)
        assert np.array_equal(whole.n_matched, per_pixel.n_matched)

    def test_with_every_option_off_old_vectors_fit_as_the_original_per_pattern_fit(self):
        from spyde.actions.strain_mapping import fit_pattern_strain

        v = _vectors(columns=LEGACY_N_COLS, confidence=False)
        tol = 0.25 * _median_nn(G_REF)
        field = compute_strain_field(v, ref_vectors=G_REF, tol=tol, one_per_reflection=False, robust=None)
        for iy in range(6):
            for ix in range(6):
                exx, eyy, exy, omega, _ = fit_pattern_strain(v.kxy_at(iy, ix), G_REF, tol=tol)
                np.testing.assert_allclose([field.exx[iy, ix], field.eyy[iy, ix], field.exy[iy, ix]],
                                           [exx, eyy, exy], atol=1e-6)

    def test_an_unknown_loss_is_refused(self):
        with pytest.raises(ValueError):
            compute_strain_field(_vectors(), ref_vectors=G_REF, robust="cauchy")

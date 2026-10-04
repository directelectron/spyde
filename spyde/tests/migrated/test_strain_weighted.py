"""
The strain fit weights each vector by its confidence and positional sigma
(`strain_mapping.vector_weights`) when Find Vectors recorded them, and fits
vectors without them exactly as before.
"""
from __future__ import annotations

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
        rows[:, COL_SIGMA] = [0.0, 0.2]
        w = vector_weights(rows, unit_per_pixel=2.0)
        floor = WEIGHT_FLOOR_PX * 2.0
        np.testing.assert_allclose(w, [0.9 / floor ** 2, 0.3 / (0.04 + floor ** 2)])

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

"""
Strain mapping from diffraction vectors (`spyde.actions.strain_mapping`):
the −g=g, center-robust deformation-gradient fit and the per-pixel field.
"""
from __future__ import annotations

import numpy as np

from spyde.actions.strain_mapping import (
    fit_pattern_strain, compute_strain_field, principal_strain, StrainField,
)

# A small multi-ring reference lattice (square, 1st + 2nd ring; non-collinear).
G_REF = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
                  [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])


def _strained(T, g=G_REF, *, offset=(0.0, 0.0)):
    return (g @ np.asarray(T).T) + np.asarray(offset)


class TestStrainFit:
    def test_recovers_known_strain(self):
        T = np.array([[1.01, 0.0], [0.0, 0.995]])      # +1% x, −0.5% y
        exx, eyy, exy, omega, cov = fit_pattern_strain(_strained(T), G_REF, tol=0.3)
        assert abs(exx - 0.01) < 1e-4
        assert abs(eyy + 0.005) < 1e-4
        assert abs(exy) < 1e-4
        assert cov == 1.0

    def test_center_offset_does_not_leak_into_strain(self):
        # An off-centre diffraction pattern: every peak shifted by a constant.
        # The translation term must absorb it — strain unchanged (the −g=g point).
        T = np.array([[1.008, 0.002], [0.002, 0.996]])
        clean = fit_pattern_strain(_strained(T), G_REF, tol=0.3)
        shifted = fit_pattern_strain(_strained(T, offset=(0.35, -0.22)), G_REF, tol=0.3)
        for a, b in zip(clean[:4], shifted[:4]):
            assert abs(a - b) < 1e-6                    # identical despite the offset

    def test_friedel_matches_minus_g(self):
        # Pattern shows only the −g half of each reflection; ±g matching recovers T.
        T = np.array([[1.02, 0.0], [0.0, 1.0]])
        g_meas = _strained(T, g=-G_REF)
        exx, eyy, exy, omega, cov = fit_pattern_strain(g_meas, G_REF, tol=0.3)
        assert abs(exx - 0.02) < 1e-4 and abs(eyy) < 1e-4

    def test_pure_rotation_is_not_strain(self):
        th = np.deg2rad(2.0)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        exx, eyy, exy, omega, cov = fit_pattern_strain(_strained(R), G_REF, tol=0.3)
        assert abs(exx) < 1e-4 and abs(eyy) < 1e-4 and abs(exy) < 1e-4
        assert abs(omega - th) < 1e-3                   # rotation captured separately

    def test_too_few_matches_returns_none(self):
        assert fit_pattern_strain(np.array([[5.0, 5.0]]), G_REF, tol=0.1) is None


class _MockVecs:
    """Duck-typed SpyDEDiffractionVectors: nav_shape + kxy_at(iy, ix)."""
    def __init__(self, nav_shape, T_of):
        self.nav_shape = nav_shape
        self._T_of = T_of

    def kxy_at(self, iy, ix):
        return _strained(self._T_of(iy, ix))


class TestStrainField:
    def test_linear_gradient_field(self):
        ny, nx = 4, 5
        # εxx grows linearly with ix; reference at (0,0) is unstrained.
        T_of = lambda iy, ix: np.array([[1.0 + 0.01 * ix, 0.0], [0.0, 1.0]])
        field = compute_strain_field(_MockVecs((ny, nx), T_of), (0, 0), tol=0.3)
        assert isinstance(field, StrainField) and field.nav_shape == (ny, nx)
        assert abs(field.exx[0, 0]) < 1e-4                       # reference unstrained
        assert field.exx[0, 4] > field.exx[0, 1] > field.exx[0, 0]   # gradient
        assert np.allclose(field.exx[2, :], field.exx[0, :], atol=1e-5)  # no y dependence
        assert np.nanmax(field.coverage) == 1.0

    def test_principal_strain_axes(self):
        e1, e2, theta = principal_strain(np.array([0.02]), np.array([0.0]), np.array([0.0]))
        assert abs(e1[0] - 0.02) < 1e-9 and abs(e2[0]) < 1e-9
        assert abs(theta[0]) < 1e-9                              # ε1 along x
        # 45° pure shear → principal axes at 45°
        e1, e2, theta = principal_strain(np.array([0.0]), np.array([0.0]), np.array([0.01]))
        assert abs(e1[0] - 0.01) < 1e-9 and abs(e2[0] + 0.01) < 1e-9
        assert abs(abs(theta[0]) - np.deg2rad(45)) < 1e-6

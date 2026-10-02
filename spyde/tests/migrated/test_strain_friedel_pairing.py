"""Exact diffraction-vector positions must give the exact strain field.

Realistic geometry the old fit got wrong: the beam sits a few pixels off the
detector centre (so edge reflections lose their Friedel partners), every
pattern's beam drifts by up to half a pixel (descan), the calibrated centre is
a fraction of a pixel off, and a beam-stop arm hides a band on one side. The
±g matching then offered spots two candidates a few px apart and 10-50 % of
pixels came out wrong with perfect input.
"""
import numpy as np
import pytest

from spyde.actions.strain_mapping import compute_strain_field
from spyde.signals.diffraction_vectors import N_COLS, SpyDEDiffractionVectors

NAV = (12, 12)
DETECTOR = (-56.0, 64.0)            # px from the beam; the beam is 4 px off-centre
CENTRE_ERROR = np.array([0.3, -0.2])  # px; calibrated centre vs the true beam
G1 = 20.0 * np.array([np.cos(0.21), np.sin(0.21)])
G2 = 22.4 * np.array([np.cos(0.21 + 1.22), np.sin(0.21 + 1.22)])


def _deformation(iy, ix):
    """Smooth strain + rotation, exactly zero in the 3x3 reference corner (the
    region a reference radius of 2 around (0, 0) pools)."""
    y, x = max(iy - 2, 0), max(ix - 2, 0)
    exx, eyy, exy = 0.01 * x / NAV[1], -0.008 * y / NAV[0], 0.004 * x * y / (NAV[0] * NAV[1])
    angle = 0.006 * (x - y) / NAV[0]
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotation @ (np.eye(2) + np.array([[exx, exy], [exy, eyy]])), (exx, eyy, exy, angle)


def _descan(iy, ix):
    return 0.5 * np.array([np.sin(0.9 * ix + 0.3), np.cos(0.7 * iy - 0.4)])


def _pattern(iy, ix, beam_stop):
    h, k = np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), indexing="ij")
    g0 = h.reshape(-1, 1) * G1 + k.reshape(-1, 1) * G2
    deformation, _ = _deformation(iy, ix)
    g = g0 @ np.linalg.inv(deformation)          # reciprocal vectors: g = F^-T g0
    g = g + _descan(iy, ix)
    visible = np.all((g > DETECTOR[0]) & (g < DETECTOR[1]), axis=1)
    if beam_stop:                               # an arm from the beam to the left edge
        visible &= ~((g[:, 0] < 0) & (np.abs(g[:, 1]) < 8.0))
    return g[visible] + CENTRE_ERROR


def _vectors(beam_stop):
    rows, offsets = [], [0]
    for iy in range(NAV[0]):
        for ix in range(NAV[1]):
            for kx, ky in _pattern(iy, ix, beam_stop):
                rows.append([ix, iy, kx, ky, -1.0, 1.0])
            offsets.append(len(rows))
    flat = np.asarray(rows, dtype=np.float64).reshape(-1, N_COLS)
    offsets = np.asarray(offsets, dtype=np.int64)
    return SpyDEDiffractionVectors(
        flat_buffer=flat, nav_offsets=[np.arange(NAV[0] + 1) * NAV[1], offsets],
        nav_shape=NAV, full_nav_shape=NAV, sig_shape=(128, 128), sig_axes=None,
        kernel_radius_px=1.0, kernel_radius_data=1.0, offsets=offsets)


def _truth():
    out = np.zeros((4,) + NAV)
    for iy in range(NAV[0]):
        for ix in range(NAV[1]):
            out[:, iy, ix] = _deformation(iy, ix)[1]
    return out


@pytest.mark.parametrize("beam_stop", [False, True])
@pytest.mark.parametrize("reference_radius", [0, 2])
class TestStrainFriedelPairing:
    def test_exact_positions_give_exact_strain(self, beam_stop, reference_radius):
        field = compute_strain_field(_vectors(beam_stop), (0, 0), ref_radius=reference_radius)
        measured = np.stack([field.exx, field.eyy, field.exy, field.omega]).astype(float)
        assert np.all(np.isfinite(measured)), "every pixel must fit"
        error = np.abs(measured - _truth())
        assert error.max() < 1e-6, (
            f"max strain error {error.max():.2e}; "
            f"{(error.max(0) > 1e-4).mean() * 100:.0f} % of pixels off by > 1e-4")

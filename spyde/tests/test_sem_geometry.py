"""
Flat-detector / curved-Ewald-sphere geometry corrections for SEM 4D-STEM
(spyde/actions/sem_geometry.py). The corrections must be EXACT (round-trip a
forward-simulated spot), reduce to the linear map at small angle / high kV, and
recover the pattern center from Friedel pairs with no direct beam.
"""
import numpy as np
import pytest

from spyde.actions import sem_geometry as G


class TestWavelength:
    @pytest.mark.parametrize("kV, lam_A", [
        (200.0, 0.02508), (30.0, 0.06979), (10.0, 0.12204), (5.0, 0.17328),
    ])
    def test_known_wavelengths(self, kV, lam_A):
        assert G.electron_wavelength(kV) == pytest.approx(lam_A, rel=2e-3)


class TestRadiusToG:
    def test_small_angle_linear_limit(self):
        # R << L: g -> R / (lambda L). At 200 kV (flat sphere) over a modest R.
        lam = G.electron_wavelength(200.0)
        L = 1000.0
        R = np.array([0.5, 1.0, 2.0])     # R/L = 5e-4 .. 2e-3
        g = G.detector_radius_to_g(R, L, lam)
        g_lin = R / (lam * L)
        assert np.allclose(g, g_lin, rtol=1e-5)

    def test_exact_roundtrip(self):
        # forward: pick true g -> 2theta -> R; then invert and recover g.
        lam = G.electron_wavelength(5.0)
        L = 100.0
        g_true = np.array([0.2, 0.5, 0.9, 1.3])
        theta = np.arcsin(g_true * lam / 2.0)        # Bragg: g = 2 sin(th)/lam
        R = L * np.tan(2.0 * theta)                   # flat detector radius
        g_rec = G.detector_radius_to_g(R, L, lam)
        assert np.allclose(g_rec, g_true, rtol=1e-9)

    def test_linear_overestimates_at_high_angle(self):
        # the naive linear calibration over-estimates g at large R (low kV).
        lam = G.electron_wavelength(5.0)
        L = 50.0
        R = 20.0                                       # R/L = 0.4, big angle
        g_exact = G.detector_radius_to_g(R, L, lam)
        g_lin = R / (lam * L)
        assert g_lin > g_exact                          # linear too large
        # relative error ~ (R/L)^2 / 8 — sizeable here
        assert (g_lin - g_exact) / g_exact > 0.01


class TestCorrectVectors:
    def test_tem_is_near_noop_in_shape(self):
        # at 200 kV with sane geometry the correction barely moves spots;
        # azimuth is exactly preserved (radial-only remap).
        lam = G.electron_wavelength(200.0)
        L = 1000.0
        rng = np.random.RandomState(0)
        kxy = rng.uniform(-2, 2, (20, 2))
        out = G.correct_vectors_flat_detector(kxy, L, lam,
                                              center=(0, 0), pixel_to_length=1.0)
        # same direction (cross product ~ 0)
        cross = kxy[:, 0] * out[:, 1] - kxy[:, 1] * out[:, 0]
        assert np.allclose(cross, 0, atol=1e-6)

    def test_azimuth_preserved_low_kv(self):
        lam = G.electron_wavelength(5.0)
        kxy = np.array([[10.0, 0.0], [0.0, 8.0], [6.0, 6.0], [-5.0, 2.0]])
        out = G.correct_vectors_flat_detector(kxy, 80.0, lam)
        for i in range(len(kxy)):
            a_in = np.arctan2(kxy[i, 1], kxy[i, 0])
            a_out = np.arctan2(out[i, 1], out[i, 0])
            assert np.isclose(np.cos(a_in - a_out), 1.0, atol=1e-6)

    def test_center_offset_applied(self):
        lam = G.electron_wavelength(30.0)
        c = np.array([3.0, -2.0])
        # a spot exactly at the center maps to ~0
        out = G.correct_vectors_flat_detector(c[None, :], 100.0, lam, center=c)
        assert np.allclose(out, 0.0, atol=1e-9)


class TestFriedelCenter:
    def test_recovers_offset_center_no_direct_beam(self):
        rng = np.random.RandomState(1)
        g = rng.uniform(-1, 1, (8, 2))
        g = np.vstack([g, -g])               # centrosymmetric, no (0,0) spot
        true_c = np.array([2.5, -1.5])
        meas = g + true_c                     # shift off-origin (no centering)
        c = G.friedel_center(meas)
        assert c is not None
        assert np.allclose(c, true_c, atol=0.05)

    def test_robust_to_unpaired_spots(self):
        rng = np.random.RandomState(2)
        g = rng.uniform(-1, 1, (6, 2))
        g = np.vstack([g, -g])
        true_c = np.array([1.0, 1.0])
        meas = np.vstack([g + true_c, np.array([[5.0, -4.0], [4.0, 5.0]])])  # +junk
        c = G.friedel_center(meas)
        assert np.allclose(c, true_c, atol=0.1)

    def test_none_when_no_pairs(self):
        # all spots on one side, no centrosymmetric partners
        kxy = np.array([[1.0, 1.0], [1.1, 0.9], [1.2, 1.0]])
        # may return a degenerate guess but must not crash; center near cloud
        c = G.friedel_center(kxy)
        assert c is None or np.all(np.isfinite(c))


class TestPrepareSemVectors:
    def test_end_to_end_recovers_g(self):
        # forward-simulate detector spots from known g at 10 kV, off-center,
        # then prepare_sem_vectors should recover the g magnitudes.
        lam = G.electron_wavelength(10.0)
        L = 60.0
        g_true = np.array([[0.3, 0.0], [-0.3, 0.0], [0.0, 0.45], [0.0, -0.45]])
        gmag = np.linalg.norm(g_true, axis=1)
        theta = np.arcsin(gmag * lam / 2.0)
        R = L * np.tan(2 * theta)
        unit = g_true / gmag[:, None]
        center = np.array([7.0, -3.0])
        det = unit * R[:, None] + center            # flat-detector positions
        corr, c = G.prepare_sem_vectors(det, 10.0, L, pixel_to_length=1.0)
        assert np.allclose(c, center, atol=0.1)
        assert np.allclose(np.linalg.norm(corr, axis=1), gmag, atol=1e-3)

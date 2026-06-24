"""
Tests for index_and_calibrate: phase d-spacing tables, scale-free ratio
matching, 2D reciprocal-basis fitting, Friedel centering, and the end-to-end
index_and_calibrate_vectors pipeline.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest

from spyde.actions.index_and_calibrate import (
    Phase, default_ti_nb_o_phases, match_phase, _ratio_match,
    fit_reciprocal_basis, center_vectors_friedel, index_and_calibrate_vectors,
    _anatase_allowed,
)


class TestPhaseDSpacings:
    def test_anatase_first_reflection(self):
        an = Phase("anatase", a=3.7845, c=9.5143, system="tetragonal",
                   allowed=_anatase_allowed)
        # (101) is the textbook strong anatase reflection at ~3.52 A
        assert abs(an.d_spacing(1, 0, 1) - 3.517) < 0.01

    def test_cubic_d_spacing(self):
        c = Phase("cubic", a=4.0, system="cubic")
        assert abs(c.d_spacing(1, 0, 0) - 4.0) < 1e-6
        assert abs(c.d_spacing(1, 1, 0) - 4.0 / np.sqrt(2)) < 1e-6

    def test_d_list_descending_unique(self):
        ph = default_ti_nb_o_phases()[0]
        d = ph.d_list(max_index=3)
        assert (np.diff(d) < 0).all()       # strictly descending
        assert len(d) == len(set(np.round(d, 4)))


class TestRatioMatch:
    def test_exact_recovery(self):
        ph = default_ti_nb_o_phases()[0]
        scale_true = 0.0123
        d = ph.d_list(max_index=3)[:5]
        R = np.sort((1.0 / d) / scale_true)
        resid, scale, assign = _ratio_match(R, ph.d_list(max_index=3))
        assert resid < 1e-6
        assert abs(scale - scale_true) < 1e-5

    def test_match_phase_ranks_correct(self):
        ph = default_ti_nb_o_phases()[1]   # rutile
        scale_true = 0.015
        d = ph.d_list(max_index=3)[:5]
        R = np.sort((1.0 / d) / scale_true)
        matches = match_phase(R)
        assert matches[0].phase == "rutile TiO2"
        assert abs(matches[0].scale_inv_angstrom_per_px - scale_true) < 5e-4


class TestBasisFit:
    def _net(self, g1, g2, hk_max=3, jitter=0.0, seed=0):
        rng = np.random.default_rng(seed)
        pts = []
        for h in range(-hk_max, hk_max + 1):
            for k in range(-hk_max, hk_max + 1):
                if h == 0 and k == 0:
                    continue
                v = h * np.array(g1) + k * np.array(g2)
                if np.hypot(*v) < 60:
                    pts.append(v)
        pts = np.array(pts, float)
        if jitter:
            pts += rng.normal(0, jitter, pts.shape)
        return pts

    def test_square_net(self):
        rel = self._net([12, 0], [0, 12])
        fb = fit_reciprocal_basis(rel)
        assert fb is not None
        assert abs(fb.g1_px - 12) < 0.5
        assert abs(fb.angle_deg - 90) < 3
        assert fb.inlier_frac > 0.9

    def test_hex_net(self):
        rel = self._net([12, 0], [6, 12 * np.sqrt(3) / 2])
        fb = fit_reciprocal_basis(rel)
        assert fb is not None
        assert abs(fb.angle_deg - 60) < 3
        assert abs(fb.ratio - 1.0) < 0.05

    def test_noisy_net_still_fits(self):
        rel = self._net([14, 1], [-2, 13], jitter=0.4)
        fb = fit_reciprocal_basis(rel)
        assert fb is not None
        assert fb.inlier_frac > 0.7

    def test_random_points_reject(self):
        rng = np.random.default_rng(1)
        rel = rng.uniform(-40, 40, (25, 2))
        fb = fit_reciprocal_basis(rel)
        # random points should not index cleanly to a net
        assert fb is None or fb.inlier_frac < 0.6


def _vectors_from_net(g1, g2, nav=(5, 5), center=(64, 64), sig=(128, 128),
                      jitter=0.0, seed=0):
    """Build a SpyDEDiffractionVectors where every pattern is the same 2D net,
    via the real find-vectors batch path (so the CSR layout is exact)."""
    from spyde.actions.find_vectors import _do_compute_vectors
    ny, nx = nav
    ky, kx = sig
    rng = np.random.default_rng(seed)
    data = rng.normal(50, 3, (ny, nx, ky, kx)).astype(np.float32)
    yy, xx = np.mgrid[0:ky, 0:kx]
    for h in range(-3, 4):
        for k in range(-3, 4):
            v = h * np.array(g1) + k * np.array(g2)
            cy, cx = center[1] + v[1], center[0] + v[0]
            if 4 < cy < ky - 4 and 4 < cx < kx - 4:
                data[:, :, :, :] += 300 * np.exp(
                    -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.3 ** 2))
    sigd = hs.signals.Signal2D(data)
    p = dict(method="dog", sigma=0.0, kernel_radius=5, threshold=8.0,
             min_distance=3, subpixel=True, dog_sigma1=0.8, dog_sigma2=2.0)
    return _do_compute_vectors(sigd, p, None, None)


class TestCentering:
    def test_friedel_center_recovered(self):
        vecs = _vectors_from_net([14, 0], [0, 14], center=(70, 60))
        cr = center_vectors_friedel(vecs)
        assert cr.n_centered >= 1
        # global center should land near the true (70, 60) px
        assert abs(cr.global_center[0] - 70) < 2.0
        assert abs(cr.global_center[1] - 60) < 2.0


class TestEndToEnd:
    def test_pipeline_runs_and_flags_consistency(self):
        # a single consistent square net across all patterns -> high/medium conf
        vecs = _vectors_from_net([15, 0], [0, 15], nav=(6, 6), center=(64, 64))
        res = index_and_calibrate_vectors(vecs)
        assert res.center.n_centered > 0
        assert len(res.grain_bases) > 0
        # all patterns identical -> bases cluster -> consistent
        assert res.single_grain_consistent
        assert res.confidence in ("high", "medium")
        assert res.best is not None

    def test_manual_calibration_from_dspacing(self):
        vecs = _vectors_from_net([15, 0], [0, 15], center=(64, 64))
        res = index_and_calibrate_vectors(vecs)
        # if the first ring (|g1| ~ 15 px) is d=3.0 A, scale = (1/3)/15
        sc = res.angstrom_per_px_from_dspacing(0, 3.0)
        assert abs(sc - (1.0 / 3.0) / np.sort(res.ring_radii_px)[0]) < 1e-9

"""Coarse-seed template selection for the vector-OM fit, and the
PARTIAL-DETECTION asymmetry it depends on.

Real vector finding detects only the strong subset of a pattern's reflections
(~5–15 of ~25), so the correct template always has spots with no detected
counterpart. The seed's scoring is therefore deliberately ASYMMETRIC: it rewards
a template whose intensity mass sits where the measurement has mass, and does
NOT charge a template for spots that were never detected.

Making it symmetric — normalising the correlation by the template norm, i.e. a
cosine — is the obvious-looking "cleanup", and it is measurably wrong. On
synthetic full-support patterns it looks like a large win (true template ranked
top 40/40 vs 1/40, recovered field exact vs ~28° off), but on real sped_ag it
collapsed agreement with the dense raw-OM reference from ~98%/98%/100% (IPF
X/Y/Z) to ~0%/28%/0%. These tests pin the asymmetry so that regression is caught
in the headless suite instead of on a user's map.

Device is pinned to CPU: forward-only tensor maths (no autograd), so it needs no
GPU, and pinning avoids the torch-CUDA-under-pytest segfault on Windows.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions.vector_orientation import TemplateLibrary

torch = pytest.importorskip("torch")


def _ring(n, r, phase=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.stack([r * np.cos(a), r * np.sin(a)], -1).astype(np.float32)


def _library(specs):
    """specs: list of (spots_xy, intensity_scale) → a minimal TemplateLibrary."""
    spots_xy, spots_I = [], []
    for xy, scale in specs:
        spots_xy.append(np.asarray(xy, np.float32))
        spots_I.append(np.full(len(xy), scale, np.float32))
    n = len(specs)
    return TemplateLibrary(
        spots_xy=spots_xy, spots_I=spots_I,
        template_quats=np.tile(np.array([1.0, 0, 0, 0]), (n, 1)),
        template_phase=np.zeros(n, np.int16),
        phases_meta=[{"name": "synthetic", "point_group": "m-3m"}],
        cache={}, radial_range=(0.0, 0.5), r_max=0.5,
    )


def _rot(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _seed(lib, meas_xy, meas_I, n_angles=180, sigma=0.06):
    """Run the production batched seed on one pattern, on the CPU device."""
    from spyde.actions.vector_orientation_gpu import (
        _pack_templates, _coarse_seed_batched)
    dev, dt = torch.device("cpu"), torch.float32
    g, gI, gmask = _pack_templates(lib, dev, dt)
    v = torch.as_tensor(meas_xy[None].astype(np.float32), device=dev, dtype=dt)
    vI = torch.as_tensor(meas_I[None].astype(np.float32), device=dev, dtype=dt)
    vmask = torch.ones(1, meas_xy.shape[0], dtype=torch.bool, device=dev)
    bt, ba, sc = _coarse_seed_batched(g, gI, gmask, v, vI, vmask,
                                      n_angles, sigma)
    return int(bt[0]), float(ba[0]), float(sc[0])


class TestSeedToleratesPartialDetection:
    """The seed must not be talked out of the right template by the reflections
    vector finding failed to detect."""

    def test_full_template_beats_a_sparse_exactly_matching_decoy(self):
        """THE REGRESSION. Measured = 5 of the true template's 12 spots (partial
        detection). The decoy holds ONLY those 5 positions, so it "explains" the
        measurement perfectly and a template-norm-normalised (cosine) score
        prefers it. The seed must still pick the full template."""
        true_xy = _ring(12, 0.30)
        keep = [0, 2, 5, 7, 10]
        meas = true_xy[keep].copy()
        decoy_xy = meas.copy()
        lib = _library([(decoy_xy, 1.0), (true_xy, 1.0)])   # true = index 1

        bt, _ba, _sc = _seed(lib, meas, np.ones(len(meas), np.float32))
        assert bt == 1, (
            "seed preferred the sparse decoy — the template-side penalty is "
            "back; see the module docstring for the sped_ag A/B")

    def test_recovers_the_in_plane_angle(self):
        true_xy = _ring(6, 0.30)
        lib = _library([(true_xy, 1.0), (_ring(4, 0.42), 1.0)])
        theta = 37.0
        meas = (true_xy @ _rot(theta).T).astype(np.float32)
        bt, ba, _ = _seed(lib, meas, np.ones(len(meas), np.float32))
        assert bt == 0
        err = abs((np.rad2deg(ba) - theta + 180.0) % 360.0 - 180.0)
        # 6-fold ring → any multiple of 60 deg is the same match
        assert min(err, abs(err - 60.0), abs(err - 120.0)) < 6.0, \
            f"angle off: {np.rad2deg(ba)}"

    def test_picks_the_matching_geometry_over_a_different_shell(self):
        """Sanity that the seed discriminates at all: a template on a different
        radial shell must lose to the one on the measured shell."""
        true_xy = _ring(8, 0.30)
        lib = _library([(_ring(8, 0.45), 1.0), (true_xy, 1.0)])
        meas = (true_xy @ _rot(-80.0).T).astype(np.float32)
        bt, _ba, _sc = _seed(lib, meas, np.ones(len(meas), np.float32))
        assert bt == 1


class TestPoseInPlaneAngle:
    """The scipy refine's orientation must come from the pose's PHYSICAL
    rotation, not the bare theta parameter: LM freely parks part of the total
    rotation in the free 2x2 A, so theta under-reports it. Resolving from theta
    gave IPF-X/Y agreement of 44%/2% against the dense reference on sped_ag even
    with the correct template handed in (IPF-Z, blind to the in-plane angle, was
    100%); resolving from the polar-decomposition rotation gives 99%/98%/100%."""

    def _pose(self, theta, A):
        return np.concatenate([[theta], np.asarray(A, float).ravel(), [0.0, 0.0]])

    def test_pure_rotation_in_theta(self):
        from spyde.actions.vector_orientation import pose_in_plane_angle
        for deg in (-170.0, -35.0, 0.0, 42.0, 155.0):
            ang = pose_in_plane_angle(self._pose(np.deg2rad(deg), np.eye(2)))
            assert np.rad2deg(ang) == pytest.approx(deg, abs=1e-6)

    def test_rotation_parked_in_A_is_recovered(self):
        """theta=0 but A itself is a rotation — the physical angle is A's."""
        from spyde.actions.vector_orientation import pose_in_plane_angle
        for deg in (-120.0, -15.0, 25.0, 88.0):
            ang = pose_in_plane_angle(self._pose(0.0, _rot(deg)))
            assert np.rad2deg(ang) == pytest.approx(deg, abs=1e-6)

    def test_rotation_split_between_theta_and_A_adds_up(self):
        from spyde.actions.vector_orientation import pose_in_plane_angle
        ang = pose_in_plane_angle(self._pose(np.deg2rad(20.0), _rot(35.0)))
        assert np.rad2deg(ang) == pytest.approx(55.0, abs=1e-6)

    def test_symmetric_stretch_does_not_tilt_the_angle(self):
        """A pure symmetric stretch carries no rotation, so it must not shift
        the reported in-plane angle (that separation is the whole point)."""
        from spyde.actions.vector_orientation import pose_in_plane_angle
        S = np.array([[1.03, 0.01], [0.01, 0.98]])
        ang = pose_in_plane_angle(self._pose(np.deg2rad(30.0), S))
        assert np.rad2deg(ang) == pytest.approx(30.0, abs=0.6)

    def test_strain_is_unchanged_by_the_split(self):
        """Companion invariant: the strain must be the same whether the rotation
        sits in theta or in A."""
        from spyde.actions.vector_orientation import strain_from_pose
        S = np.array([[1.03, 0.01], [0.01, 0.98]])
        a = strain_from_pose(self._pose(np.deg2rad(30.0), S))
        b = strain_from_pose(self._pose(0.0, S @ _rot(30.0)))
        np.testing.assert_allclose(a, b, atol=1e-9)

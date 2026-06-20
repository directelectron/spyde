"""
Unit tests for SpyDEOrientationMap — pure container, no Qt, no dask.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from spyde.signals.orientation_map import (
    SpyDEOrientationMap, ipf_xy_for_rotations, ipf_triangle_xy,
    orix_phase_from_dict,
)


def _random_quats(rng, shape):
    """Uniform random unit quaternions of the given leading shape."""
    q = rng.normal(size=shape + (4,))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.astype(np.float32)


def _make_map(ny=6, nx=5, n_best=3, n_phases=1, seed=0):
    rng = np.random.default_rng(seed)
    quats = _random_quats(rng, (ny, nx, n_best))
    corr = np.sort(rng.random((ny, nx, n_best)).astype(np.float32),
                   axis=-1)[..., ::-1].copy()
    phase_idx = rng.integers(0, n_phases, (ny, nx, n_best)).astype(np.int16)
    mirror = rng.choice([-1, 1], (ny, nx, n_best)).astype(np.int8)
    phases = [{"name": f"phase{i}", "point_group": pg}
              for i, pg in zip(range(n_phases), ["m-3m", "6/mmm"][:n_phases])]
    return SpyDEOrientationMap(
        quats=quats, corr=corr, phase_idx=phase_idx, mirror=mirror,
        phases=phases, params={"n_best": n_best},
    )


class TestIPFHelpers:

    def test_identity_rotation_projects_z_to_origin(self):
        from orix.quaternion import Rotation
        phase = orix_phase_from_dict({"name": "al", "point_group": "m-3m"})
        x, y = ipf_xy_for_rotations(Rotation.identity(), phase, "z")
        # [001] is a sector vertex at the stereographic origin for m-3m
        assert abs(x[0]) < 1e-6 and abs(y[0]) < 1e-6

    def test_xy_inside_sector_bounds(self):
        from orix.quaternion import Rotation
        rng = np.random.default_rng(3)
        phase = orix_phase_from_dict({"name": "al", "point_group": "m-3m"})
        rots = Rotation(_random_quats(rng, (200,)))
        x, y = ipf_xy_for_rotations(rots, phase, "z")
        edges, _, _ = ipf_triangle_xy(phase)
        # All projected points must lie within the sector's bounding box
        pad = 1e-3
        assert x.min() >= edges[:, 0].min() - pad
        assert x.max() <= edges[:, 0].max() + pad
        assert y.min() >= edges[:, 1].min() - pad
        assert y.max() <= edges[:, 1].max() + pad

    def test_triangle_outline_nonempty(self):
        phase = orix_phase_from_dict({"name": "al", "point_group": "m-3m"})
        edges, label_xy, labels = ipf_triangle_xy(phase)
        assert len(edges) > 10
        assert len(labels) == len(label_xy)


class TestContainer:

    def test_shapes_and_props(self):
        om = _make_map(ny=6, nx=5, n_best=3)
        assert om.nav_shape == (6, 5)
        assert om.n_best == 3
        assert om.n_phases == 1

    def test_ipf_color_map(self):
        om = _make_map()
        rgb = om.ipf_color_map("z")
        assert rgb.shape == (6, 5, 3)
        assert rgb.dtype == np.uint8
        assert rgb.max() > 0  # not all black

    def test_color_map_directions_differ(self):
        om = _make_map(seed=7)
        rz = om.ipf_color_map("z")
        rx = om.ipf_color_map("x")
        assert not np.array_equal(rz, rx)

    def test_correlation_and_phase_maps(self):
        om = _make_map(n_phases=2, seed=2)
        cm = om.correlation_map()
        pm = om.phase_map()
        assert cm.shape == (6, 5) and pm.shape == (6, 5)
        np.testing.assert_array_equal(cm, om.corr[..., 0])
        assert set(np.unique(pm)).issubset({0, 1})

    def test_ipf_xy_matches_direct_orix(self):
        om = _make_map(seed=4)
        xy, pidx, corr = om.ipf_xy(2, 3)
        assert xy.shape == (3, 2)
        from orix.quaternion import Rotation
        x, y = ipf_xy_for_rotations(
            Rotation(om.quats[2, 3, 0]), om.orix_phase(0), "z"
        )
        np.testing.assert_allclose(xy[0], [x[0], y[0]], atol=1e-6)

    def test_ipf_xy_roi_counts_and_subsample(self):
        om = _make_map(ny=8, nx=8, n_best=4, seed=5)
        xy, c = om.ipf_xy_roi(slice(0, 4), slice(0, 8), phase=0,
                              best_only=True)
        assert len(xy) == 4 * 8
        xy_all, _ = om.ipf_xy_roi(slice(0, 4), slice(0, 8), phase=0,
                                  best_only=False)
        assert len(xy_all) == 4 * 8 * 4
        xy_sub, _ = om.ipf_xy_roi(slice(0, 8), slice(0, 8), phase=0,
                                  best_only=False, max_points=10)
        assert len(xy_sub) <= 10 + 4  # stride subsampling, roughly capped

    def test_ipf_xy_roi_multiphase_partition(self):
        om = _make_map(n_phases=2, seed=6)
        n0 = len(om.ipf_xy_roi(slice(None), slice(None), phase=0)[0])
        n1 = len(om.ipf_xy_roi(slice(None), slice(None), phase=1)[0])
        assert n0 + n1 == 6 * 5  # best-only: every position exactly once

    def test_ipf_xyz_geometry(self):
        om = _make_map(seed=8)
        v, t, pidx = om.ipf_xyz(1, 1)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)
        # tangent is unit and orthogonal to v
        assert np.isclose(np.linalg.norm(t), 1.0, atol=1e-6)
        assert abs(np.dot(v, t)) < 1e-6
        # v must be inside the fundamental sector (folding idempotent)
        from orix.vector import Vector3d
        v_again = Vector3d(v).in_fundamental_sector(
            om.orix_phase(pidx).point_group
        )
        np.testing.assert_allclose(v_again.data.ravel(), v, atol=1e-5)

    def test_ipf_xyz_encodes_inplane_rotation(self):
        """Two orientations sharing the beam direction but differing by an
        in-plane rotation must give the same v but different tangents."""
        from orix.quaternion import Rotation
        from orix.vector import Vector3d
        base = Rotation.identity()
        inplane = Rotation.from_axes_angles(Vector3d.zvector(), np.pi / 3)
        om = _make_map(ny=1, nx=2, n_best=1, seed=9)
        om.quats[0, 0, 0] = base.data.ravel().astype(np.float32)
        om.quats[0, 1, 0] = (inplane * base).data.ravel().astype(np.float32)
        om.phase_idx[...] = 0
        v1, t1, _ = om.ipf_xyz(0, 0)
        v2, t2, _ = om.ipf_xyz(0, 1)
        np.testing.assert_allclose(v1, v2, atol=1e-5)
        assert np.linalg.norm(t1 - t2) > 0.1

    def test_save_load_roundtrip(self):
        om = _make_map(n_phases=2, seed=10)
        path = os.path.join(tempfile.gettempdir(), "test_om.npz")
        try:
            om.save(path)
            om2 = SpyDEOrientationMap.load(path)
            np.testing.assert_array_equal(om2.quats, om.quats)
            np.testing.assert_array_equal(om2.corr, om.corr)
            np.testing.assert_array_equal(om2.phase_idx, om.phase_idx)
            assert om2.phases == om.phases
            # reloaded container is fully functional
            rgb = om2.ipf_color_map()
            assert rgb.shape == (6, 5, 3)
        finally:
            if os.path.exists(path):
                os.remove(path)

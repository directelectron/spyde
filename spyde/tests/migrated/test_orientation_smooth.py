"""
The within-grain orientation mean: speckle inside a grain drops, the boundary
between two grains stays where it was, and positions nothing matched are
neither moved nor averaged in.
"""
from __future__ import annotations

import numpy as np
import pytest


def _field(ny=12, nx=16, noise_deg=1.5, seed=0):
    """Two grains side by side, 30° apart, each with a tilt scatter of
    *noise_deg* about its own orientation. Returns the result and the two
    true orientations."""
    from orix.quaternion import Orientation, Rotation
    from orix.quaternion.symmetry import Oh
    from spyde.signals.orientation_map import VectorOrientationResult

    rng = np.random.default_rng(seed)
    left = Orientation.from_axes_angles([0, 0, 1], np.deg2rad(10), symmetry=Oh)
    right = Orientation.from_axes_angles([0, 0, 1], np.deg2rad(40), symmetry=Oh)
    quats = np.empty((ny, nx, 4))
    for iy in range(ny):
        for ix in range(nx):
            base = left if ix < nx // 2 else right
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            wobble = Rotation.from_axes_angles(axis, np.deg2rad(rng.normal(0, noise_deg)))
            quats[iy, ix] = (wobble * Rotation(base.data)).data
    result = VectorOrientationResult(
        quats=quats.astype(np.float32),
        phase_idx=np.zeros((ny, nx), np.int16), theta=np.zeros((ny, nx), np.float32),
        strain=np.full((ny, nx, 3), np.nan, np.float32),
        residual=np.full((ny, nx), np.nan, np.float32),
        friedel_asym=np.full((ny, nx), np.nan, np.float32),
        n_matched=np.zeros((ny, nx), np.int16),
        coarse_score=np.full((ny, nx), 0.9, np.float32),
        phases_meta=[{"name": "Ag", "point_group": "m-3m"}], nav_shape=(ny, nx))
    result.reliability = np.full((ny, nx), 0.5, np.float32)
    return result, left, right


def _scatter(quats, truth):
    """Mean symmetry-reduced angle, degrees, of each orientation to *truth*."""
    from orix.quaternion import Orientation
    from orix.quaternion.symmetry import Oh
    orientations = Orientation(np.asarray(quats, float).reshape(-1, 4), symmetry=Oh)
    return float(np.degrees(orientations.angle_with(truth)).mean())


class TestWithinGrainMean:
    def test_speckle_drops_and_the_boundary_stays(self):
        from spyde.actions.orientation_smooth import smooth_orientation_field
        result, left, right = _field()
        nx = result.nav_shape[1]
        before_left = _scatter(result.quats[:, :nx // 2], left)
        before_right = _scatter(result.quats[:, nx // 2:], right)

        smoothed = smooth_orientation_field(result, threshold_deg=5.0, iterations=2)

        after_left = _scatter(smoothed.quats[:, :nx // 2], left)
        after_right = _scatter(smoothed.quats[:, nx // 2:], right)
        assert after_left < before_left / 2, (before_left, after_left)
        assert after_right < before_right / 2, (before_right, after_right)
        # The boundary column on each side still belongs to its own grain.
        assert _scatter(smoothed.quats[:, nx // 2 - 1], left) < 3
        assert _scatter(smoothed.quats[:, nx // 2], right) < 3
        assert np.array_equal(smoothed.raw_quats, result.quats), "the raw field is kept"
        assert smoothed.params["smooth_orientations"] is True

    def test_an_unmatched_position_is_left_alone(self):
        from spyde.actions.orientation_smooth import smooth_orientation_field
        result, _left, _right = _field()
        result.coarse_score[3, 3] = 0.0
        hole = result.quats[3, 3].copy()
        smoothed = smooth_orientation_field(result)
        assert np.array_equal(smoothed.quats[3, 3], hole)

    def test_the_variant_nearest_the_centre_is_taken(self):
        """A neighbour reported as a symmetric equivalent far away in
        quaternion space is still the same orientation, and is averaged as
        such rather than pulling the mean off toward nothing."""
        from orix.quaternion import Orientation, Rotation
        from orix.quaternion.symmetry import Oh
        from spyde.actions.orientation_smooth import nearest_variant

        centre = Orientation.from_axes_angles([0, 0, 1], np.deg2rad(10), symmetry=Oh)
        equivalent = (Rotation(Oh.data[7]) * Rotation(centre.data)).data
        variant, degrees = nearest_variant(
            np.asarray(centre.data).reshape(1, 4), np.asarray(equivalent).reshape(1, 4),
            Oh.data)
        assert degrees[0] < 1e-6
        assert np.allclose(np.abs(variant[0] @ centre.data.reshape(4)), 1.0)

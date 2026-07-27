"""Atom finding, refinement and property maps (#75, #77, #79).

The synthetic lattice knows exactly where every atom is, so refinement is
scored against those positions rather than against atomap's answer or a
fixture. Sub-pixel accuracy is the whole point of the gaussian step, so the
tolerances here are tight on purpose — a test that accepts +/-1 px would pass
with the refinement removed entirely.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyde.atoms import (
    displacement_from_ideal,
    ellipticity,
    intensity,
    nearest_neighbour_distance,
    property_maps,
    refine_atoms,
    refine_center_of_mass,
    refine_gaussian,
    to_map,
)
from spyde.data import atom_lattice, ground_truth


@pytest.fixture(scope="module")
def lattice():
    s = atom_lattice(grid=(6, 8), spacing=16.0, noise=0.0)
    return np.asarray(s.data, float), np.asarray(ground_truth(s)["positions"])


def _jitter(pos, amount, seed=0):
    rng = np.random.default_rng(seed)
    return pos + rng.uniform(-amount, amount, pos.shape)


class TestCenterOfMass:
    def test_pulls_a_jittered_guess_back_toward_the_atom(self, lattice):
        img, truth = lattice
        start = _jitter(truth, 2.0)
        got = refine_center_of_mass(img, start, box=9)
        assert (np.hypot(*(got - truth).T).mean()
                < np.hypot(*(start - truth).T).mean())

    def test_does_not_move_an_already_centred_atom_much(self, lattice):
        img, truth = lattice
        got = refine_center_of_mass(img, truth, box=9)
        assert np.hypot(*(got - truth).T).max() < 0.6

    def test_handles_atoms_near_the_edge(self, lattice):
        """Box clamping must keep every patch inside the image — an atom in
        the corner would otherwise index out of bounds."""
        img, truth = lattice
        corner = truth[:1].copy()
        corner[0] = [1.0, 1.0]
        got = refine_center_of_mass(img, corner, box=11)
        assert np.isfinite(got).all()


class TestGaussianRefinement:
    def test_recovers_positions_to_sub_pixel(self, lattice):
        """The reason the gaussian step exists. +/-0.1 px would not be
        achievable from centre-of-mass alone on this data."""
        img, truth = lattice
        pos, params = refine_gaussian(img, _jitter(truth, 1.0), box=13,
                                      device="cpu")
        err = np.hypot(*(pos - truth).T)
        assert err.max() < 0.1, f"worst atom off by {err.max():.3f} px"
        assert params.shape == (len(truth), 5)

    def test_two_step_beats_the_gaussian_alone_from_a_poor_start(self, lattice):
        """Why atomap does COM first and so do we."""
        img, truth = lattice
        start = _jitter(truth, 3.0, seed=3)
        only_gauss, _ = refine_gaussian(img, start, box=13, device="cpu")
        two_step, _ = refine_atoms(img, start, box=13, device="cpu")
        assert (np.hypot(*(two_step - truth).T).mean()
                <= np.hypot(*(only_gauss - truth).T).mean())

    def test_recovers_the_known_widths(self, lattice):
        img, truth = lattice
        _, params = refine_gaussian(img, truth, box=13, sigma=2.5, device="cpu")
        # The generator used sigma=2.6 with no ellipticity.
        assert np.abs(params[:, 3] - 2.6).max() < 0.15
        assert np.abs(params[:, 4] - 2.6).max() < 0.15

    def test_pinned_atoms_keep_their_position_exactly(self, lattice):
        """The red/green refine toggle (#76). A pinned atom must be EXCLUDED
        from the fit, not fitted and then discarded."""
        img, truth = lattice
        start = _jitter(truth, 1.0)
        mask = np.ones(len(truth), bool)
        mask[::3] = False
        pos, params = refine_gaussian(img, start, box=13, device="cpu",
                                      refine_mask=mask)
        np.testing.assert_array_equal(pos[~mask], start[~mask])
        assert np.isnan(params[~mask]).all()
        assert not np.allclose(pos[mask], start[mask])

    def test_all_atoms_pinned_is_a_no_op(self, lattice):
        img, truth = lattice
        pos, params = refine_gaussian(img, truth, device="cpu",
                                      refine_mask=np.zeros(len(truth), bool))
        np.testing.assert_array_equal(pos, truth)
        assert np.isnan(params).all()

    def test_mask_length_is_checked(self, lattice):
        img, truth = lattice
        with pytest.raises(ValueError, match="refine_mask"):
            refine_gaussian(img, truth, device="cpu",
                            refine_mask=np.ones(3, bool))

    def test_a_runaway_fit_keeps_its_input_position(self):
        """A fit that leaves its own box is worse than the input, so the input
        is what comes back — never a wild coordinate."""
        img = np.zeros((40, 40))          # nothing to fit at all
        start = np.array([[20.0, 20.0]])
        pos, _ = refine_gaussian(img, start, box=11, device="cpu")
        assert np.hypot(*(pos - start).T).max() <= 5.5

    def test_ellipticity_is_recovered_when_present(self):
        s = atom_lattice(grid=(4, 6), spacing=18.0, ellipticity=0.6, noise=0.0)
        img = np.asarray(s.data, float)
        truth = np.asarray(ground_truth(s)["positions"])
        _, params = refine_gaussian(img, truth, box=15, device="cpu")
        e = ellipticity(params)
        # The generator stretches sigma_x with x, so the right-hand atoms must
        # be measurably more elliptical than the left-hand ones.
        left = truth[:, 0] < truth[:, 0].mean()
        assert np.nanmean(e[~left]) > np.nanmean(e[left]) + 0.1


class TestPropertyMaps:
    def test_ellipticity_is_at_least_one(self, lattice):
        img, truth = lattice
        _, params = refine_gaussian(img, truth, box=13, device="cpu")
        e = ellipticity(params)
        assert np.nanmin(e) >= 1.0 - 1e-9

    def test_ellipticity_does_not_flip_with_orientation(self):
        """Expressed as max/min, so an atom elongated along y reads the same as
        one elongated along x. A sigma_x/sigma_y ratio would drop below 1 and
        put a spurious boundary wherever the elongation direction changes."""
        wide_x = np.array([[1.0, 0, 0, 3.0, 1.5]])
        wide_y = np.array([[1.0, 0, 0, 1.5, 3.0]])
        assert ellipticity(wide_x)[0] == pytest.approx(ellipticity(wide_y)[0])

    def test_nn_distance_matches_the_lattice_spacing(self, lattice):
        img, truth = lattice
        d = nearest_neighbour_distance(truth)
        assert np.median(d) == pytest.approx(16.0, rel=0.02)

    def test_displacement_is_zero_on_a_perfect_lattice(self, lattice):
        """A local reference means an undistorted lattice reads zero — the
        baseline every distortion map is measured against."""
        img, truth = lattice
        inner = ((truth[:, 0] > 20) & (truth[:, 0] < truth[:, 0].max() - 20)
                 & (truth[:, 1] > 20) & (truth[:, 1] < truth[:, 1].max() - 20))
        d = displacement_from_ideal(truth)
        assert np.abs(d[inner]).max() < 1e-6

    def test_displacement_finds_a_real_distortion(self):
        s = atom_lattice(grid=(6, 8), spacing=18.0, displacement=2.0,
                         noise=0.0)
        truth = np.asarray(ground_truth(s)["positions"])
        assert np.abs(displacement_from_ideal(truth)).max() > 0.5

    def test_intensity_is_the_fitted_volume(self, lattice):
        img, truth = lattice
        _, params = refine_gaussian(img, truth, box=13, device="cpu")
        assert np.allclose(intensity(params), params[:, 0])
        assert (intensity(params) > 0).all()

    def test_property_maps_returns_one_value_per_atom(self, lattice):
        img, truth = lattice
        _, params = refine_gaussian(img, truth, box=13, device="cpu")
        maps = property_maps(truth, params)
        assert set(maps) == {"Ellipticity", "Intensity", "NN distance",
                             "Displacement"}
        for name, v in maps.items():
            assert v.shape == (len(truth),), name


class TestToMap:
    def test_places_values_at_their_atoms(self, lattice):
        img, truth = lattice
        vals = np.arange(len(truth), dtype=float)
        m = to_map(vals, truth, img.shape)
        ix = int(round(truth[5, 0]))
        iy = int(round(truth[5, 1]))
        assert m[iy, ix] == pytest.approx(5.0)

    def test_pixels_without_an_atom_are_nan(self, lattice):
        img, truth = lattice
        m = to_map(np.ones(len(truth)), truth, img.shape)
        assert np.isnan(m).any()

    def test_atoms_outside_the_shape_are_dropped_not_wrapped(self):
        m = to_map(np.array([1.0, 2.0]), np.array([[5.0, 5.0], [500.0, 5.0]]),
                   (20, 20))
        assert m[5, 5] == pytest.approx(1.0)
        assert np.nansum(m) == pytest.approx(1.0)

    def test_length_mismatch_is_caught(self, lattice):
        img, truth = lattice
        with pytest.raises(ValueError, match="values for"):
            to_map(np.ones(3), truth, img.shape)


class TestAtomapIntegration:
    def test_find_atoms_locates_the_lattice(self, lattice):
        pytest.importorskip("atomap", reason="needs the atoms extra")
        from spyde.atoms import find_atoms

        img, truth = lattice
        found = find_atoms(img, separation=10)
        assert len(found) == pytest.approx(len(truth), rel=0.15)
        # Every found atom should be near a real one.
        from scipy.spatial import cKDTree
        d, _ = cKDTree(truth).query(found)
        assert np.median(d) < 2.0

    def test_find_then_refine_recovers_the_truth(self, lattice):
        """The full pipeline, scored against known positions."""
        pytest.importorskip("atomap", reason="needs the atoms extra")
        from scipy.spatial import cKDTree
        from spyde.atoms import find_atoms

        img, truth = lattice
        found = find_atoms(img, separation=10)
        refined, _ = refine_atoms(img, found, box=13, device="cpu")
        d, _ = cKDTree(truth).query(refined)
        assert np.median(d) < 0.2, f"median error {np.median(d):.3f} px"

    def test_missing_extra_gives_the_install_line(self, monkeypatch):
        import spyde.atoms.finding as f

        def boom():
            raise f.MissingExtra(
                'atom finding needs atomap — install with: '
                'pip install "spyde[atoms]"')

        monkeypatch.setattr(f, "_require_atomap", boom)
        with pytest.raises(f.MissingExtra, match=r"spyde\[atoms\]"):
            f.find_atoms(np.zeros((10, 10)))

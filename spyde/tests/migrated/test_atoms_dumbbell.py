"""Dumbbell lattices (#78).

The synthetic lattice generates dumbbells with a KNOWN separation, so every
test here scores against that rather than against a fixture.

The properties that matter and are easy to get subtly wrong: the estimated
vector must not cancel to zero (each pair contributes +v and -v), pairing must
not claim an atom twice, and the angle must be folded to a half-turn or two
identical dumbbells differ by pi depending on which atom happened to be listed
first.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.atoms.dumbbell import (
    dumbbell_properties,
    estimate_dumbbell_vector,
    find_dumbbells,
    pair_atoms,
)
from spyde.data import atom_lattice, ground_truth

SEP = 6.0


@pytest.fixture(scope="module")
def dumbbells():
    s = atom_lattice(grid=(5, 6), spacing=20.0, dumbbell=SEP, sigma=2.0,
                     noise=0.0)
    return np.asarray(s.data, float), np.asarray(ground_truth(s)["positions"])


class TestEstimateVector:
    def test_recovers_the_known_separation(self, dumbbells):
        _, pos = dumbbells
        v = estimate_dumbbell_vector(pos)
        assert np.hypot(*v) == pytest.approx(SEP, rel=0.02)

    def test_recovers_the_known_direction(self, dumbbells):
        """The generator splits along x, so the vector must be along x."""
        _, pos = dumbbells
        v = estimate_dumbbell_vector(pos)
        assert abs(v[1]) < 0.1 * abs(v[0])

    def test_does_not_cancel_to_zero(self, dumbbells):
        """Each pair contributes +v from one atom and -v from the other. An
        implementation that averages without folding onto a half-plane gets
        exactly zero — and zero is a plausible-looking answer."""
        _, pos = dumbbells
        assert np.hypot(*estimate_dumbbell_vector(pos)) > 1.0

    def test_works_for_a_diagonal_dumbbell(self):
        """Nothing may assume the pair splits along an axis."""
        base = np.array([[x * 30.0, y * 30.0]
                         for y in range(4) for x in range(4)])
        v = np.array([4.0, 3.0])                    # length 5, diagonal
        pos = np.vstack([base, base + v])
        got = estimate_dumbbell_vector(pos)
        assert np.hypot(*got) == pytest.approx(5.0, rel=1e-6)
        assert abs(got[1] / got[0]) == pytest.approx(0.75, rel=1e-6)

    def test_max_separation_rejects_a_non_dumbbell_lattice(self):
        """A plain lattice has no pair closer than the spacing; saying so beats
        returning a 'dumbbell vector' that is really the lattice vector."""
        pos = np.array([[x * 20.0, y * 20.0]
                        for y in range(4) for x in range(4)])
        with pytest.raises(ValueError, match="not a dumbbell lattice"):
            estimate_dumbbell_vector(pos, max_separation=5.0)

    def test_too_few_atoms_is_an_error(self):
        with pytest.raises(ValueError, match="at least two"):
            estimate_dumbbell_vector(np.array([[1.0, 2.0]]))


class TestPairing:
    def test_pairs_every_atom_on_a_clean_lattice(self, dumbbells):
        _, pos = dumbbells
        v = estimate_dumbbell_vector(pos)
        pairs, unpaired = pair_atoms(pos, v)
        assert len(pairs) == len(pos) // 2
        assert len(unpaired) == 0

    def test_no_atom_is_claimed_twice(self, dumbbells):
        """A greedy matcher that forgets to mark atoms taken produces more
        pairs than atoms and every downstream count is wrong."""
        _, pos = dumbbells
        pairs, _ = pair_atoms(pos, estimate_dumbbell_vector(pos))
        flat = pairs.ravel()
        assert len(set(flat.tolist())) == len(flat)

    def test_partners_are_the_real_partners(self, dumbbells):
        """Each pair must be SEP apart — pairing neighbouring dumbbells instead
        would still return the right number of pairs."""
        _, pos = dumbbells
        pairs, _ = pair_atoms(pos, estimate_dumbbell_vector(pos))
        d = np.hypot(*(pos[pairs[:, 1]] - pos[pairs[:, 0]]).T)
        np.testing.assert_allclose(d, SEP, rtol=0.02)

    def test_a_wrong_vector_leaves_atoms_unpaired(self):
        """Surfacing a bad vector beats papering over it: an atom with no
        partner at the expected offset must be REPORTED, not force-matched."""
        base = np.array([[x * 30.0, 0.0] for x in range(4)])
        pos = np.vstack([base, base + np.array([5.0, 0.0])])
        _, unpaired = pair_atoms(pos, np.array([13.0, 0.0]))
        assert len(unpaired) > 0

    def test_zero_vector_is_rejected(self, dumbbells):
        _, pos = dumbbells
        with pytest.raises(ValueError, match="zero length"):
            pair_atoms(pos, np.array([0.0, 0.0]))

    def test_an_odd_atom_out_is_reported_not_dropped(self):
        base = np.array([[0.0, 0.0], [5.0, 0.0], [40.0, 0.0]])
        pairs, unpaired = pair_atoms(base, np.array([5.0, 0.0]))
        assert len(pairs) == 1
        assert unpaired.tolist() == [2]


class TestProperties:
    def test_separation_matches_the_truth(self, dumbbells):
        _, pos = dumbbells
        pairs, props, _ = find_dumbbells(pos)
        np.testing.assert_allclose(props["separation"], SEP, rtol=0.02)

    def test_angle_is_folded_to_a_half_turn(self):
        """+v and -v are the SAME dumbbell. Without folding, two identical
        dumbbells differ by pi depending on which atom was listed first, and an
        angle map shows a boundary that is not there."""
        pos = np.array([[0.0, 0.0], [5.0, 0.0],
                        [100.0, 0.0], [95.0, 0.0]])
        props = dumbbell_properties(pos, np.array([[0, 1], [2, 3]]))
        assert props["angle"][0] == pytest.approx(props["angle"][1], abs=1e-9)

    def test_angle_tracks_a_real_rotation(self):
        pos = np.array([[0.0, 0.0], [3.0, 3.0]])
        props = dumbbell_properties(pos, np.array([[0, 1]]))
        assert props["angle"][0] == pytest.approx(np.pi / 4)

    def test_centre_is_the_midpoint(self):
        pos = np.array([[10.0, 20.0], [16.0, 20.0]])
        props = dumbbell_properties(pos, np.array([[0, 1]]))
        assert props["centre_x"][0] == pytest.approx(13.0)
        assert props["centre_y"][0] == pytest.approx(20.0)

    def test_one_value_per_pair_not_per_atom(self, dumbbells):
        _, pos = dumbbells
        pairs, props, _ = find_dumbbells(pos)
        for name, v in props.items():
            assert len(v) == len(pairs), name
        assert len(pairs) == len(pos) // 2

    def test_intensity_ratio_distinguishes_the_two_sites(self):
        """A polar structure has unequal sites; a symmetric pair must read 1."""
        pos = np.array([[0.0, 0.0], [5.0, 0.0], [50.0, 0.0], [55.0, 0.0]])
        params = np.zeros((4, 5))
        params[:, 0] = [10.0, 20.0, 10.0, 10.0]
        props = dumbbell_properties(pos, np.array([[0, 1], [2, 3]]), params)
        assert props["intensity_ratio"][0] == pytest.approx(2.0)
        assert props["intensity_ratio"][1] == pytest.approx(1.0)


class TestFullWorkflow:
    def test_reports_the_vector_it_used(self, dumbbells):
        """A bad estimate must be visible, not show up later as a
        suspiciously small dumbbell count."""
        _, pos = dumbbells
        _, _, info = find_dumbbells(pos)
        assert np.hypot(*info["vector"]) == pytest.approx(SEP, rel=0.02)
        assert info["n_unpaired"] == 0
        assert info["n_pairs"] == len(pos) // 2

    def test_accepts_an_explicit_vector(self, dumbbells):
        """A user who knows the vector — or picked it interactively (#76) —
        can skip the estimate."""
        _, pos = dumbbells
        _, _, info = find_dumbbells(pos, vector=(SEP, 0.0))
        assert info["n_pairs"] == len(pos) // 2

    def test_survives_refinement_jitter(self, dumbbells):
        """Real refined positions are not exact, so pairing must tolerate
        sub-pixel scatter."""
        _, pos = dumbbells
        jittered = pos + np.random.default_rng(0).uniform(-0.3, 0.3, pos.shape)
        _, props, info = find_dumbbells(jittered)
        assert info["n_unpaired"] == 0
        np.testing.assert_allclose(props["separation"], SEP, atol=0.8)

    def test_independent_fits_underestimate_the_separation(self, dumbbells):
        """Documents the error joint fitting exists to remove.

        Fitting each atom on its own biases its centre TOWARDS its partner —
        the partner's tail is signal a single gaussian can only explain by
        moving. The separation therefore comes out too small, and UNIFORMLY so,
        which makes it look like a calibration rather than a bug.
        """
        img, truth = dumbbells
        from spyde.atoms import refine_gaussian

        refined, _ = refine_gaussian(img, truth, box=9, sigma=2.0,
                                     device="cpu")
        _, props, _ = find_dumbbells(refined)
        measured = float(np.median(props["separation"]))
        assert measured < 0.85 * SEP, (
            f"expected the known inward bias; got {measured:.2f} for a true "
            f"{SEP}. If this now passes, joint fitting may have become the "
            f"default and this test is obsolete.")

    def test_joint_fitting_recovers_the_true_separation(self, dumbbells):
        """The fix: both atoms in ONE box as one two-gaussian model, so each
        explains the other's tail instead of absorbing it."""
        img, truth = dumbbells
        from spyde.atoms.dumbbell import refine_pairs

        pairs, _ = pair_atoms(truth, estimate_dumbbell_vector(truth))
        refined, _ = refine_pairs(img, truth, pairs, sigma=2.0, device="cpu")
        props = dumbbell_properties(refined, pairs)
        np.testing.assert_allclose(props["separation"], SEP, atol=0.15)

    def test_joint_fitting_beats_independent_fitting(self, dumbbells):
        """The comparison stated directly, so a regression in either path
        shows up as the gap closing from the wrong side."""
        img, truth = dumbbells
        from spyde.atoms import refine_gaussian
        from spyde.atoms.dumbbell import refine_pairs

        pairs, _ = pair_atoms(truth, estimate_dumbbell_vector(truth))
        solo, _ = refine_gaussian(img, truth, box=9, sigma=2.0, device="cpu")
        joint, _ = refine_pairs(img, truth, pairs, sigma=2.0, device="cpu")

        err_solo = abs(np.median(dumbbell_properties(solo, pairs)["separation"]) - SEP)
        err_joint = abs(np.median(dumbbell_properties(joint, pairs)["separation"]) - SEP)
        assert err_joint < err_solo / 5, (err_joint, err_solo)

    def test_end_to_end_from_the_image(self, dumbbells):
        """The order a user works in: find pairs, then refine them jointly."""
        img, truth = dumbbells
        from spyde.atoms.dumbbell import refine_pairs

        pairs, props, info = find_dumbbells(truth)
        assert info["n_pairs"] == len(truth) // 2
        refined, params = refine_pairs(img, truth, pairs, sigma=2.0,
                                       device="cpu")
        final = dumbbell_properties(refined, pairs)
        np.testing.assert_allclose(final["separation"], SEP, atol=0.15)

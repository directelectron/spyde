"""CrystalMap, IPF colours, similarity map, phase merging (#73).

The display half of Wave 3. Compute only — the window wiring that puts these on
screen is UI work and gets verified by running the app and looking at pixels
(CLAUDE.md), not here.

The similarity map gets the most attention because it is the one thing here
with non-obvious semantics: it measures agreement between NEIGHBOURS' ranked
match lists, which is what exposes confidently-wrong indexing that a raw score
map cannot.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("orix")

from spyde.data import ebsd_patterns, ground_truth
from spyde.data.synthetic import simulate_patterns
from spyde.ebsd import dictionary_index, remove_background, sample_orientations
from spyde.ebsd.crystal_map import (
    COMMON_PHASES,
    ipf_colors,
    merge_phases,
    orientation_similarity_map,
    to_crystal_map,
)


@pytest.fixture(scope="module")
def scan():
    s = ebsd_patterns(nav=(8, 8), detector=(32, 32), noise=0.0)
    gt = ground_truth(s)
    return s, np.asarray(gt["euler"]), np.asarray(gt["grain2_mask"], bool)


class TestCrystalMap:
    def test_builds_from_euler_angles(self, scan):
        _, euler, _ = scan
        xm = to_crystal_map(euler)
        assert xm.size == 64
        assert xm.shape == (8, 8)

    def test_carries_scores_as_a_property(self, scan):
        _, euler, _ = scan
        xm = to_crystal_map(euler, scores=np.linspace(0, 1, 64))
        assert "scores" in xm.prop
        assert xm.prop["scores"].shape == (64,)

    def test_phase_symmetry_comes_from_the_space_group(self, scan):
        _, euler, _ = scan
        assert to_crystal_map(euler, space_group=COMMON_PHASES["fcc"]
                              ).phases[0].point_group.name == "m-3m"
        assert to_crystal_map(euler, space_group=COMMON_PHASES["hcp"]
                              ).phases[0].point_group.name == "6/mmm"

    def test_step_scales_the_coordinates(self, scan):
        _, euler, _ = scan
        assert to_crystal_map(euler, step=0.5).x.max() == pytest.approx(3.5)


class TestIpfColors:
    def test_shape_and_range(self, scan):
        _, euler, _ = scan
        rgb = ipf_colors(euler)
        assert rgb.shape == (8, 8, 3)
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0

    def test_the_two_grains_get_different_colours(self, scan):
        """The whole point of an IPF map. Identical colours would mean the map
        renders as a flat block and still passes every range check."""
        _, euler, mask = scan
        rgb = ipf_colors(euler)
        assert not np.allclose(rgb[mask].mean(0), rgb[~mask].mean(0), atol=0.05)

    def test_one_orientation_gives_one_colour(self, scan):
        """Grain 2 is a single orientation, so its colour must be uniform."""
        _, euler, mask = scan
        block = ipf_colors(euler)[mask]
        # Compared as a max deviation rather than assert_allclose against
        # block[0]: numpy no longer broadcasts shapes there, so (N, 3) vs (3,)
        # fails on shape even when every row is identical.
        assert np.abs(block - block[0]).max() < 1e-6

    def test_is_rgb_shaped_for_the_existing_display(self, scan):
        """commit_result_tree takes an (H, W, 3) RGB primary directly — this is
        why Wave 3 needs no new display code."""
        _, euler, _ = scan
        rgb = ipf_colors(euler)
        assert rgb.ndim == 3 and rgb.shape[-1] == 3


class TestOrientationSimilarityMap:
    def test_uniform_indexing_scores_one(self):
        """Every position agreeing with every neighbour is perfect agreement."""
        idx = np.tile(np.array([[3, 7, 1]]), (16, 1))
        osm = orientation_similarity_map(idx, (4, 4))
        np.testing.assert_allclose(osm, 1.0)

    def test_a_boundary_shows_as_low_agreement(self):
        """The metric exists to find boundaries and bad indexing; a map where
        half the positions match a different list must dip at the seam."""
        grid = np.zeros((4, 4, 2), int)
        grid[:, :2] = [1, 2]
        grid[:, 2:] = [8, 9]
        osm = orientation_similarity_map(grid.reshape(16, 2), (4, 4))
        assert osm[:, 1].mean() < osm[:, 0].mean()
        assert osm[:, 2].mean() < osm[:, 3].mean()

    def test_partial_overlap_is_between_zero_and_one(self):
        """Sharing some of the ranked list, but not all of it, is the normal
        case and must land in between rather than saturating."""
        grid = np.zeros((2, 2, 2), int)
        grid[0, 0] = [1, 2]
        grid[0, 1] = [1, 9]      # shares one of two
        grid[1, 0] = [1, 2]
        grid[1, 1] = [1, 2]
        osm = orientation_similarity_map(grid.reshape(4, 2), (2, 2))
        assert 0.0 < osm[0, 1] < 1.0

    def test_detects_a_confidently_wrong_position(self):
        """The case a SCORE map cannot show: one position indexed to something
        completely different from its neighbourhood."""
        idx = np.tile(np.array([[3, 7, 1]]), (25, 1))
        idx[12] = [40, 41, 42]                        # centre of a 5x5
        osm = orientation_similarity_map(idx, (5, 5))
        assert osm[2, 2] == pytest.approx(0.0)
        assert osm[0, 0] > 0.9

    def test_rejects_a_mismatched_shape(self):
        with pytest.raises(ValueError, match="does not match"):
            orientation_similarity_map(np.zeros((10, 3), int), (4, 4))

    def test_requires_a_nav_shape(self):
        with pytest.raises(ValueError, match="nav_shape"):
            orientation_similarity_map(np.zeros((16, 3), int))

    def test_works_on_a_real_indexing_result(self, scan):
        s, _, mask = scan
        dic_euler = sample_orientations(step_deg=12.0)
        dic = simulate_patterns(dic_euler, detector=(32, 32))
        r = dictionary_index(remove_background(s.data, device="cpu"),
                             remove_background(dic, device="cpu"),
                             keep=5, device="cpu")
        osm = orientation_similarity_map(r.indices, (8, 8))
        assert osm.shape == (8, 8)
        assert (osm >= 0).all() and (osm <= 1).all()
        # Inside a uniform grain neighbours agree; across the boundary they
        # cannot, so the map must not be flat.
        assert osm.std() > 0


class TestMergePhases:
    def test_picks_the_higher_scoring_phase_per_position(self):
        a = np.tile([0.1, 0.2, 0.3], (4, 1))
        b = np.tile([1.0, 1.1, 1.2], (4, 1))
        euler, phase_id, best = merge_phases(
            [a, b], [np.array([0.9, 0.1, 0.9, 0.1]),
                     np.array([0.2, 0.8, 0.2, 0.8])])
        assert phase_id.tolist() == [0, 1, 0, 1]
        np.testing.assert_allclose(euler[0], a[0])
        np.testing.assert_allclose(euler[1], b[1])
        np.testing.assert_allclose(best, [0.9, 0.8, 0.9, 0.8])

    def test_reshapes_to_the_scan(self):
        a = np.zeros((4, 3))
        b = np.ones((4, 3))
        euler, phase_id, best = merge_phases(
            [a, b], [np.zeros(4), np.ones(4)], nav_shape=(2, 2))
        assert euler.shape == (2, 2, 3)
        assert phase_id.shape == (2, 2)
        assert best.shape == (2, 2)

    def test_mismatched_coverage_is_caught(self):
        with pytest.raises(ValueError, match="same positions"):
            merge_phases([np.zeros((4, 3)), np.zeros((5, 3))],
                         [np.zeros(4), np.zeros(5)])


class TestEndToEnd:
    def test_index_then_colour_recovers_the_grain_structure(self, scan):
        """Indexing -> IPF colours must reproduce the two-grain layout the data
        was built with — the closest headless proxy for 'the map looks right'.
        The window itself still needs eyes on a screenshot."""
        s, euler, mask = scan
        dic_euler = sample_orientations(step_deg=10.0)
        dic = simulate_patterns(dic_euler, detector=(32, 32))
        r = dictionary_index(remove_background(s.data, device="cpu"),
                             remove_background(dic, device="cpu"),
                             device="cpu")
        rgb = ipf_colors(r.orientations(dic_euler, (8, 8)))
        assert rgb.shape == (8, 8, 3)
        inside = rgb[mask].mean(0)
        outside = rgb[~mask].mean(0)
        assert not np.allclose(inside, outside, atol=0.05), \
            "indexed IPF map does not distinguish the two grains"

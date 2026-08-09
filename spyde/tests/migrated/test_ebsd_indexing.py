"""GPU dictionary indexing (#71).

The test that matters is **recovering known orientations**: the synthetic EBSD
data stamps the exact Euler angles it was generated from, so indexing is scored
against the truth rather than against a fixture of its own output.

The rest pins the tiling, which is where this can silently go wrong — a
running top-k merged across tiles must give exactly the same answer as one
un-tiled matmul, or results depend on chunk size and nobody notices.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyde.data import ebsd_patterns, ground_truth
from spyde.data.synthetic import simulate_patterns
from spyde.ebsd import dictionary_index, sample_orientations


@pytest.fixture(scope="module")
def scan():
    s = ebsd_patterns(nav=(8, 8), detector=(40, 40), noise=0.0)
    gt = ground_truth(s)
    return s, np.asarray(gt["euler"]), np.asarray(gt["grain2_mask"], bool)


@pytest.fixture(scope="module")
def dic_exact(scan):
    """The exact-euler dictionary — pure data, was rebuilt by ~6 tests."""
    _s, euler, _ = scan
    return simulate_patterns(euler.reshape(-1, 3), detector=(40, 40))


@pytest.fixture(scope="module")
def step_dict():
    """Memoised (euler, dictionary) per sampling step — pure data; the 10/12/
    15 deg dictionaries were each built twice.  Consumers are read-only
    (remove_background / dictionary_index return new arrays)."""
    cache: dict = {}

    def get(step_deg):
        if step_deg not in cache:
            eul = sample_orientations(step_deg=step_deg)
            cache[step_deg] = (eul, simulate_patterns(eul, detector=(40, 40)))
        return cache[step_deg]

    return get


class TestRecoversKnownOrientations:
    def test_exact_dictionary_recovers_every_orientation(self, scan, dic_exact):
        """The strongest possible check: put the true orientations IN the
        dictionary and require indexing to find each one."""
        s, euler, _ = scan
        flat = euler.reshape(-1, 3)
        dic = dic_exact
        r = dictionary_index(s.data, dic, device="cpu")
        got = np.asarray(dic)[r.best]
        # Several positions share an orientation (grain 2 is uniform), so
        # compare the matched PATTERN, not the index.
        for i in range(len(flat)):
            assert np.allclose(got[i], dic[i], atol=1e-5), \
                f"position {i} matched a different pattern"

    def test_background_removal_is_what_makes_scores_near_one(self, scan, dic_exact):
        """NCC is invariant to gain and offset but NOT to a spatial GRADIENT.
        The detector background is present in every experimental pattern and in
        no simulated one, so it dilutes every score until it is removed — which
        is the entire practical argument for #70."""
        from spyde.ebsd import remove_background

        s, euler, _ = scan
        dic = dic_exact
        raw = dictionary_index(s.data, dic, device="cpu").best_score.min()
        # The SAME correction must be applied to BOTH sides. Removing the low
        # frequencies from the experimental patterns only would leave the
        # dictionary carrying low-frequency content its counterpart no longer
        # has, which is a different mismatch rather than a fix. kikuchipy
        # preprocesses its dictionary for the same reason.
        clean = dictionary_index(
            remove_background(s.data, method="dynamic", device="cpu"),
            remove_background(dic, method="dynamic", device="cpu"),
            device="cpu").best_score.min()
        assert clean > raw, f"background removal did not help ({clean} vs {raw})"
        assert clean > 0.99, f"exact match still only scores {clean:.4f}"

    def test_a_finer_dictionary_matches_better(self, scan, step_dict):
        """A realistic dictionary does NOT contain the exact answer — it lands
        on a nearby sample, and the finer the sampling the closer it lands.

        That MONOTONIC relationship is the real property worth pinning; an
        absolute score threshold would just encode whatever this synthetic
        crystal happens to produce. Similarity is compared rather than Euler
        distance because Euler angles are not a metric space, and cubic
        symmetry means several triples describe the same crystal.
        """
        from spyde.ebsd import remove_background

        s, euler, _ = scan
        exp = remove_background(s.data, device="cpu")
        scores = {}
        for step in (15.0, 10.0, 5.0):
            _eul, dic = step_dict(step)
            r = dictionary_index(exp, remove_background(dic, device="cpu"),
                                 device="cpu")
            scores[step] = float(r.best_score.mean())
        assert scores[5.0] > scores[10.0] > scores[15.0], scores
        assert scores[5.0] > 0.9, f"even a 5 deg dictionary only got {scores}"

    def test_the_two_grains_index_differently(self, scan, step_dict):
        """If every position matched the same entry the map would be blank and
        every score test above would still pass.  Two properties of the one
        result: the grain index-sets are disjoint, and grain 2 (a single
        orientation) picks the SAME dictionary entry at every position."""
        s, euler, mask = scan
        _eul, dic = step_dict(10.0)
        r = dictionary_index(s.data, dic, device="cpu")
        best = r.best.reshape(mask.shape)
        assert set(best[mask].tolist()).isdisjoint(set(best[~mask].tolist()))
        assert len(set(best[mask].tolist())) == 1


class TestTiling:
    """A running top-k merged across tiles must equal one un-tiled matmul."""

    def test_tiled_equals_untiled(self, scan, step_dict):
        """One whole/tiny pair at keep=5 pins both properties: the winner is
        identical, and the k-th best SCORE is right too — the similarity map
        (#73) ranks the whole list.

        The ranked list is compared on scores, not indices: cubic symmetry
        means many dictionary entries produce genuinely identical patterns, so
        which of a tied set is returned depends on tile order and is not
        meaningful. Requiring equal indices there would be asserting an
        arbitrary tie-break.
        """
        s, euler, _ = scan
        _eul, dic = step_dict(12.0)
        whole = dictionary_index(s.data, dic, device="cpu", keep=5,
                                 tile_elements=1 << 30)
        tiny = dictionary_index(s.data, dic, device="cpu", keep=5,
                                tile_elements=512)
        np.testing.assert_array_equal(whole.best, tiny.best)
        np.testing.assert_allclose(whole.best_score, tiny.best_score, rtol=1e-5)
        np.testing.assert_allclose(whole.scores, tiny.scores, atol=1e-5)

    def test_scores_are_sorted_best_first(self, scan, step_dict):
        s, _, _ = scan
        _eul, dic = step_dict(15.0)
        r = dictionary_index(s.data, dic, device="cpu", keep=4)
        assert (np.diff(r.scores, axis=1) <= 1e-6).all()

    def test_keep_larger_than_the_dictionary_is_clamped(self, scan):
        s, euler, _ = scan
        dic = simulate_patterns(euler.reshape(-1, 3)[:3], detector=(40, 40))
        r = dictionary_index(s.data, dic, device="cpu", keep=99)
        assert r.indices.shape[1] == 3


class TestNormalisation:
    def test_invariant_to_gain_and_offset(self, scan, dic_exact):
        """NCC is scale- and offset-free, which is why a dictionary needs no
        background: a detector gradient or an exposure change must not alter
        the match."""
        s, euler, _ = scan
        dic = dic_exact
        base = dictionary_index(s.data, dic, device="cpu")
        altered = dictionary_index(s.data.astype(np.float64) * 3.7 + 120.0,
                                   dic, device="cpu")
        np.testing.assert_array_equal(base.best, altered.best)
        np.testing.assert_allclose(base.best_score, altered.best_score,
                                   atol=1e-4)

    def test_survives_a_detector_background(self, scan, dic_exact):
        """The real reason the above matters — an un-corrected pattern still
        indexes, so background removal (#70) is an improvement not a
        prerequisite."""
        s, euler, _ = scan
        dic = dic_exact
        with_bg = simulate_patterns(euler.reshape(-1, 3), detector=(40, 40),
                                    background=True)
        r = dictionary_index(with_bg, dic, device="cpu")
        assert r.best_score.min() > 0.9


class TestShapesAndGuards:
    def test_accepts_flattened_or_image_shaped_patterns(self, scan, dic_exact):
        s, euler, _ = scan
        dic = dic_exact
        img = dictionary_index(s.data, dic, device="cpu")
        flat = dictionary_index(s.data.reshape(64, -1),
                                dic.reshape(len(dic), -1), device="cpu")
        np.testing.assert_array_equal(img.best, flat.best)

    def test_mismatched_detector_size_is_caught(self, scan):
        s, euler, _ = scan
        wrong = simulate_patterns(euler.reshape(-1, 3), detector=(20, 20))
        with pytest.raises(ValueError, match="pixels"):
            dictionary_index(s.data, wrong, device="cpu")

    def test_orientations_reshape_to_the_scan(self, scan, step_dict):
        s, euler, _ = scan
        dic_euler, dic = step_dict(15.0)
        r = dictionary_index(s.data, dic, device="cpu")
        assert r.orientations(dic_euler, (8, 8)).shape == (8, 8, 3)

    def test_progress_reports_completion(self, scan, dic_exact):
        s, euler, _ = scan
        dic = dic_exact
        seen = []
        dictionary_index(s.data, dic, device="cpu", tile_elements=2048,
                         progress=lambda d, t: seen.append((d, t)))
        assert seen and seen[-1] == (64, 64)


class TestSampleOrientations:
    def test_covers_the_cubic_fundamental_zone(self):
        eul = sample_orientations(step_deg=15.0)
        assert eul.shape[1] == 3
        assert eul[:, 0].max() <= np.deg2rad(360)
        assert eul[:, 1].max() <= np.deg2rad(90) + 1e-9

    def test_finer_step_gives_a_bigger_dictionary(self):
        assert len(sample_orientations(10.0)) > len(sample_orientations(20.0))


# --------------------------------------------------------------------------
# parity against kikuchipy — THE acceptance gate for #71
# --------------------------------------------------------------------------

class TestKikuchipyParity:
    """We replaced kikuchipy's dask dictionary indexing with a chunked matmul,
    so the bar is its numbers, not our own self-consistency.

    Everything else in this file shows our answer is *sensible* (it recovers
    known orientations, tiling does not change it). Only this shows it is the
    *same* answer the reference implementation gives. The comparison is made at
    the similarity-metric level rather than through ``EBSD.dictionary_indexing``
    deliberately: that wrapper also does masking, dask chunking and CrystalMap
    assembly, none of which we reimplemented — the claim under test is
    specifically that our ``E @ Dᵀ`` reproduces kikuchipy's NCC.

    Skipped without the ``ebsd`` extra; CI's `extras` job installs it.
    """

    @staticmethod
    def _reference(exp, dic):
        """kikuchipy's own (P, D) NCC matrix for the same inputs."""
        kp_indexing = pytest.importorskip("kikuchipy.indexing")
        P, D = len(exp), len(dic)
        metric = kp_indexing.NormalizedCrossCorrelationMetric(P, D)
        e = metric.prepare_experimental(exp.reshape(P, -1))
        d = metric.prepare_dictionary(dic.reshape(D, -1))
        return np.asarray(metric.match(e, d))

    @pytest.fixture(scope="class")
    def pair(self):
        """Random patterns, with a few dictionary entries made near-copies of
        experimental ones so the ranking is decided by signal rather than by
        which way ties happen to break."""
        rng = np.random.default_rng(0)
        P, D, H, W = 37, 53, 12, 10
        exp = rng.normal(50, 12, (P, H, W)).astype(np.float32)
        dic = rng.normal(50, 12, (D, H, W)).astype(np.float32)
        dic[:8] = exp[:8] + rng.normal(0, 0.4, (8, H, W))
        return exp, dic

    def test_every_similarity_matches(self, pair):
        """The whole (P, D) matrix, not just the winners."""
        exp, dic = pair
        ref = self._reference(exp, dic)
        got = dictionary_index(exp, dic, keep=len(dic), device="cpu")
        mine = np.empty_like(ref)
        for p in range(len(exp)):
            mine[p, got.indices[p]] = got.scores[p]
        # Measured max difference 1.8e-7 — float32 rounding on a quantity
        # bounded by 1, i.e. the two agree to the precision they are computed in.
        np.testing.assert_allclose(mine, ref, atol=1e-5)

    def test_the_same_pattern_wins(self, pair):
        exp, dic = pair
        ref = self._reference(exp, dic)
        got = dictionary_index(exp, dic, device="cpu")
        assert (got.best == ref.argmax(1)).all()

    def test_the_winning_score_matches(self, pair):
        exp, dic = pair
        ref = self._reference(exp, dic)
        got = dictionary_index(exp, dic, device="cpu")
        np.testing.assert_allclose(got.best_score, ref.max(1), atol=1e-5)

    def test_tiling_does_not_move_us_off_the_reference(self, pair):
        """Parity has to survive the chunking, which is the part kikuchipy
        does completely differently (dask blocks vs a running top-k)."""
        exp, dic = pair
        ref = self._reference(exp, dic)
        got = dictionary_index(exp, dic, device="cpu", tile_elements=64)
        assert (got.best == ref.argmax(1)).all()
        np.testing.assert_allclose(got.best_score, ref.max(1), atol=1e-5)

    @pytest.mark.skipif(
        not (getattr(torch.backends, "mps", None)
             and torch.backends.mps.is_available()),
        reason="no Apple-MPS device")
    def test_mps_matches_the_reference_too(self, pair):
        """Metal must land on kikuchipy's answer, not merely on the CPU's."""
        exp, dic = pair
        ref = self._reference(exp, dic)
        got = dictionary_index(exp, dic, device="mps")
        assert (got.best == ref.argmax(1)).all()
        np.testing.assert_allclose(got.best_score, ref.max(1), atol=1e-5)

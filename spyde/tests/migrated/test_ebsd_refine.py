"""Batched orientation refinement (#72).

Indexing can only return a dictionary entry, so its accuracy is capped by the
sampling step. The property under test is that refinement **lifts that cap**:
starting from a coarse indexed match it must land measurably closer to the true
orientation, which the synthetic data knows exactly.

CPU only — torch-CUDA segfaults under pytest on Windows (CLAUDE.md).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyde.data import ebsd_patterns, ground_truth
from spyde.data.synthetic import simulate_patterns
from spyde.ebsd import dictionary_index, remove_background, sample_orientations
from spyde.ebsd.refine import (
    BandSimulator,
    euler_to_matrix_torch,
    refine_orientations,
)


@pytest.fixture(scope="module")
def scan():
    s = ebsd_patterns(nav=(4, 4), detector=(32, 32), noise=0.0)
    gt = ground_truth(s)
    return s, np.asarray(gt["euler"])


def _pattern_similarity(euler_a, euler_b, detector=(32, 32)):
    """Compare orientations by the PATTERNS they produce, not by Euler
    distance: Euler angles are not a metric space, and cubic symmetry means
    different triples can describe the same crystal."""
    a = simulate_patterns(np.atleast_2d(euler_a), detector=detector)
    b = simulate_patterns(np.atleast_2d(euler_b), detector=detector)
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    a = (a - a.mean(1, keepdims=True))
    b = (b - b.mean(1, keepdims=True))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


class TestEulerToMatrix:
    def test_matches_the_numpy_reference(self):
        """The torch rotation must be the same convention as the generator's,
        or refinement optimises toward a different crystal."""
        from spyde.data.synthetic import _euler_to_matrix
        eul = np.array([[0.3, 0.7, 1.1], [2.0, 0.4, 0.9]])
        want = _euler_to_matrix(eul[:, 0], eul[:, 1], eul[:, 2])
        got = euler_to_matrix_torch(torch.as_tensor(eul)).numpy()
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_is_differentiable(self):
        eul = torch.tensor([[0.3, 0.7, 1.1]], dtype=torch.float64,
                           requires_grad=True)
        euler_to_matrix_torch(eul).sum().backward()
        assert torch.isfinite(eul.grad).all()

    def test_produces_proper_rotations(self):
        eul = torch.tensor([[0.3, 0.7, 1.1], [2.0, 0.4, 0.9]],
                           dtype=torch.float64)
        m = euler_to_matrix_torch(eul)
        eye = torch.eye(3, dtype=torch.float64).expand(2, 3, 3)
        torch.testing.assert_close(m @ m.transpose(1, 2), eye)
        torch.testing.assert_close(torch.linalg.det(m),
                                   torch.ones(2, dtype=torch.float64))


class TestBandSimulator:
    def test_matches_the_numpy_generator(self):
        """The differentiable simulator and the data generator must render the
        SAME pattern — otherwise refinement is fitting a different forward
        model than the one that made the data."""
        eul = np.array([[0.3, 0.7, 1.1], [1.5, 0.4, 0.2]])
        want = simulate_patterns(eul, detector=(24, 24)).reshape(2, -1)
        got = BandSimulator((24, 24))(torch.as_tensor(eul, dtype=torch.float32))
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-4, atol=1e-4)

    def test_is_differentiable_wrt_orientation(self):
        eul = torch.tensor([[0.3, 0.7, 1.1]], requires_grad=True)
        BandSimulator((16, 16))(eul).sum().backward()
        assert torch.isfinite(eul.grad).all()
        assert (eul.grad.abs() > 0).any(), "gradient is identically zero"


class TestRefinementImprovesOnIndexing:
    def test_recovers_a_perturbed_orientation(self):
        """The cleanest statement of the job: nudge a known orientation off by
        a few degrees and require refinement to walk it back."""
        truth = np.array([[0.3, 0.6, 1.0], [1.2, 0.5, 0.4]])
        pat = simulate_patterns(truth, detector=(32, 32))
        start = truth + np.deg2rad(4.0)

        r = refine_orientations(pat, start, detector=(32, 32), device="cpu",
                                steps=200, lr=0.02)
        before = _pattern_similarity(start, truth)
        after = _pattern_similarity(r.euler, truth)
        assert (after > before).all(), f"{after} vs {before}"
        assert after.min() > 0.99, f"only reached {after}"

    def test_beats_a_coarse_dictionary(self, scan):
        """The real workflow: index against a coarse dictionary, then refine.
        Refinement must land closer to truth than the dictionary entry could."""
        s, euler = scan
        exp = remove_background(s.data, device="cpu")
        dic_euler = sample_orientations(step_deg=15.0)
        dic = simulate_patterns(dic_euler, detector=(32, 32))
        idx = dictionary_index(exp, remove_background(dic, device="cpu"),
                               device="cpu")
        indexed = idx.orientations(dic_euler)

        r = refine_orientations(exp, indexed, detector=(32, 32), device="cpu",
                                steps=250, lr=0.02)
        flat_truth = euler.reshape(-1, 3)
        before = _pattern_similarity(indexed, flat_truth)
        after = _pattern_similarity(r.euler, flat_truth)
        assert after.mean() > before.mean(), (
            f"refinement did not improve on indexing: "
            f"{after.mean():.4f} vs {before.mean():.4f}")

    def test_reports_the_score_it_started_from(self, scan):
        s, euler = scan
        r = refine_orientations(s.data, euler.reshape(-1, 3) + 0.05,
                                detector=(32, 32), device="cpu", steps=60)
        assert r.score_before.shape == r.score.shape
        assert (r.score >= r.score_before - 1e-6).all()

    def test_never_returns_a_worse_orientation(self):
        """Refinement is an improvement step ON TOP of indexing. If Adam
        wanders off — a bad start, too large an lr — the indexed answer is
        still the better estimate and must be what comes back."""
        truth = np.array([[0.3, 0.6, 1.0]])
        pat = simulate_patterns(truth, detector=(32, 32))
        # An absurd learning rate makes the optimiser diverge on purpose.
        r = refine_orientations(pat, truth, detector=(32, 32), device="cpu",
                                steps=40, lr=5.0)
        np.testing.assert_allclose(r.euler, truth, atol=1e-6)
        assert r.improved.all()


class TestBatching:
    def test_each_pattern_gets_its_own_orientation(self):
        """A broadcasting mistake would drive every position to one answer."""
        truth = np.array([[0.2, 0.5, 0.9], [1.4, 0.8, 0.3], [2.6, 0.3, 1.2]])
        pat = simulate_patterns(truth, detector=(32, 32))
        r = refine_orientations(pat, truth + np.deg2rad(3.0),
                                detector=(32, 32), device="cpu", steps=150,
                                lr=0.02)
        sim = _pattern_similarity(r.euler, truth)
        assert (sim > 0.98).all(), sim
        assert len({tuple(np.round(e, 4)) for e in r.euler}) == 3

    def test_chunked_matches_unchunked(self):
        truth = np.array([[0.2, 0.5, 0.9], [1.4, 0.8, 0.3], [2.6, 0.3, 1.2],
                          [0.9, 0.6, 0.7]])
        pat = simulate_patterns(truth, detector=(24, 24))
        start = truth + np.deg2rad(3.0)
        kw = dict(detector=(24, 24), device="cpu", steps=80, lr=0.02)
        whole = refine_orientations(pat, start, **kw)
        pieces = refine_orientations(pat, start, chunk=2, **kw)
        np.testing.assert_allclose(whole.euler, pieces.euler, atol=1e-5)

    def test_shape_mismatch_is_caught(self):
        pat = simulate_patterns(np.array([[0.3, 0.6, 1.0]]), detector=(16, 16))
        with pytest.raises(ValueError, match="starting"):
            refine_orientations(pat, np.zeros((5, 3)), detector=(16, 16),
                                device="cpu", steps=2)

    def test_result_maps_back_to_the_scan(self, scan):
        s, euler = scan
        r = refine_orientations(s.data, euler.reshape(-1, 3),
                                detector=(32, 32), device="cpu", steps=10)
        assert r.euler_map((4, 4)).shape == (4, 4, 3)


class TestCallbacks:
    def test_yield_is_called_inside_the_step_loop(self):
        """Yielding only between stages leaves a UI frozen for seconds; the
        contract is that it fires DURING the optimisation."""
        truth = np.array([[0.3, 0.6, 1.0]])
        pat = simulate_patterns(truth, detector=(16, 16))
        calls = []
        refine_orientations(pat, truth, detector=(16, 16), device="cpu",
                            steps=60, yield_every=10,
                            on_yield=lambda: calls.append(1))
        assert len(calls) >= 5

    def test_a_failing_callback_does_not_kill_the_refine(self):
        truth = np.array([[0.3, 0.6, 1.0]])
        pat = simulate_patterns(truth, detector=(16, 16))

        def boom():
            raise RuntimeError("renderer went away")

        r = refine_orientations(pat, truth, detector=(16, 16), device="cpu",
                                steps=30, on_yield=boom,
                                progress=lambda d, t: boom())
        assert r.euler.shape == (1, 3)

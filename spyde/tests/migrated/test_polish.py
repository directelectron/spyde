"""Rescuing the positions the batched fit got wrong.

A whole-scan fit already puts ~97% of positions at the noise floor; the
headroom is the few percent that land somewhere else, and those are pixels
whose neighbours all succeeded. Restarting each from its best neighbour
recovers most of them — measured on hyperspy's two_gaussians, positions worse
than 1.5x the noise floor go 27 -> 1 and total chisq improves 9.7%.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyde.actions.fit_action import new_component_spec
from spyde.fitting import ModelSpec
from spyde.fitting.polish import neighbour_index, poor_mask, polish_scan


def _gauss_scan(ny=8, nx=8, nc=192, seed=0):
    """A smooth scan of one gaussian, so a neighbour really is a good seed."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 60.0, nc)
    centre = 30.0 + 4.0 * np.sin(np.linspace(0, 3, ny))[:, None] \
        * np.cos(np.linspace(0, 3, nx))[None, :]
    data = np.empty((ny, nx, nc))
    for iy in range(ny):
        for ix in range(nx):
            data[iy, ix] = (800 * np.exp(
                -0.5 * ((x - centre[iy, ix]) / 5.0) ** 2) + 20.0)
    data = rng.poisson(np.maximum(data, 0)).astype(float)
    return x, data


def _spec():
    spec = ModelSpec()
    c = new_component_spec("Gaussian")
    c["centre"].value, c["sigma"].value, c["A"].value = 30.0, 5.0, 10000.0
    spec.append(c)
    return spec


class TestNeighbourIndex:
    def test_a_corner_has_two_neighbours(self):
        nb = neighbour_index((4, 5))
        assert len(nb[0]) == 2
        assert len(nb[-1]) == 2

    def test_an_interior_position_has_four(self):
        nb = neighbour_index((4, 5))
        assert len(nb[6]) == 4        # (1, 1)

    def test_they_are_actual_neighbours(self):
        ny, nx = 4, 5
        nb = neighbour_index((ny, nx))
        for flat, n in enumerate(nb):
            iy, ix = divmod(flat, nx)
            for j in n:
                jy, jx = divmod(int(j), nx)
                assert abs(jy - iy) + abs(jx - ix) == 1

    def test_a_1d_scan_works(self):
        nb = neighbour_index((6,))
        assert len(nb) == 6 and len(nb[0]) == 1 and len(nb[3]) == 2


class TestPoorMask:
    def test_poor_is_local_not_absolute(self):
        """chisq scales with the counts, so a bright region legitimately has a
        larger one. A global threshold would flag the whole bright region;
        only the comparison with the neighbours says "this one went wrong"."""
        nb = neighbour_index((1, 12))
        # A hundredfold intensity gradient across the scan, all fitting fine.
        chisq = np.geomspace(10.0, 1000.0, 12)
        assert not poor_mask(chisq, nb).any()

    def test_a_corner_is_not_judged_on_chisq_alone(self):
        """The median of ONE number is that number, so at a scan corner the
        test would reduce to "worse than the single pixel beside it" — which a
        steep gradient satisfies by itself."""
        nb = neighbour_index((1, 3))
        assert not poor_mask(np.array([100., 10., 10.]), nb)[0]
        # ...but not converging still flags it; that needs no neighbours.
        assert poor_mask(np.array([100., 10., 10.]), nb,
                         converged=np.array([False, True, True]))[0]

    def test_a_sharp_edge_can_be_flagged(self):
        """The known limitation, recorded rather than hidden: at a genuine
        discontinuity a position's neighbours really are unlike it, so it
        looks like an outlier. Harmless — `polish_scan` keeps a refit only
        when it lowers chisq, so a wrongly flagged position costs one wasted
        fit and changes nothing."""
        nb = neighbour_index((1, 6))
        chisq = np.array([10., 10., 10., 1000., 1000., 1000.])
        assert poor_mask(chisq, nb)[3]

    def test_a_local_outlier_is_caught(self):
        nb = neighbour_index((1, 8))
        chisq = np.array([10., 11., 9., 900., 10., 11., 9., 10.])
        assert poor_mask(chisq, nb)[3]
        assert poor_mask(chisq, nb).sum() == 1

    def test_a_position_that_did_not_converge_is_poor_whatever_its_chisq(self):
        nb = neighbour_index((1, 4))
        chisq = np.array([10., 10., 10., 10.])
        conv = np.array([True, False, True, True])
        assert poor_mask(chisq, nb, converged=conv).tolist() == \
            [False, True, False, False]


class TestPolish:
    def test_it_rescues_a_sabotaged_position(self):
        """The direct test: take a good scan, ruin one position's parameters,
        and require the pass to bring it back from its neighbours."""
        from spyde.fitting.engine import fit_batched
        x, data = _gauss_scan()
        spec = _spec()
        nav = data.shape[:2]
        flat = data.reshape(-1, data.shape[-1])
        r = fit_batched(spec, flat, x, device="cpu", max_iter=80)

        victim = 27
        r.values[victim] = [1.0, 5.0, 0.5]     # A, centre, sigma — nonsense
        from spyde.fitting import components as tcomp
        model = tcomp.evaluate(spec, torch.as_tensor(x),
                               torch.as_tensor(r.values[victim:victim + 1])
                               ).numpy()[0]
        r.chisq[victim] = float(((flat[victim] - model) ** 2).sum())
        r.converged[victim] = False
        before = r.chisq[victim]

        out = polish_scan(spec, data, x, r, nav_shape=nav)
        assert out.chisq[victim] < before / 10, "the poor position was not rescued"
        assert out.polish_improved >= 1

    def test_it_never_makes_a_position_worse(self):
        """Reseeding is a heuristic; on a position that was already right it
        can land somewhere worse. Keeping only the improvement is what makes
        the pass safe to run automatically and safe to repeat."""
        from spyde.fitting.engine import fit_batched
        x, data = _gauss_scan()
        spec = _spec()
        flat = data.reshape(-1, data.shape[-1])
        r = fit_batched(spec, flat, x, device="cpu", max_iter=80)
        before = np.array(r.chisq)
        out = polish_scan(spec, data, x, r, nav_shape=data.shape[:2])
        assert np.all(np.asarray(out.chisq) <= before + 1e-9)

    def test_a_clean_scan_needs_no_passes(self):
        from spyde.fitting.engine import fit_batched
        x, data = _gauss_scan()
        spec = _spec()
        r = fit_batched(spec, data.reshape(-1, data.shape[-1]), x,
                        device="cpu", max_iter=200)
        out = polish_scan(spec, data, x, r, nav_shape=data.shape[:2])
        assert out.polish_improved == 0

    def test_a_signal_with_no_grid_is_left_alone(self):
        from spyde.fitting.engine import fit_batched
        x, data = _gauss_scan(ny=1, nx=2)
        spec = _spec()
        r = fit_batched(spec, data.reshape(-1, data.shape[-1]), x,
                        device="cpu", max_iter=40)
        before = np.array(r.values)
        out = polish_scan(spec, data, x, r, nav_shape=(1, 2))
        assert np.array_equal(out.values, before)

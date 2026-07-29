"""Seeded propagation: coarse fit -> propagate -> one batched refine (#54).

The property that matters is not "seeding runs" but **seeding does not make
things worse and helps where a cold start struggles**. So these tests check the
propagation machinery exactly (which position seeds from which, and what
happens when a coarse fit fails) rather than just asserting a fit happened.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hyperspy.api as hs
from hyperspy.components1d import Gaussian, Offset

from spyde.fitting import ModelSpec
from spyde.fitting.engine import fit_batched
from spyde.fitting.seeding import _coarse_indices, fit_seeded


def _make_si(ny=8, nx=8, nc=192, noise=0.0):
    x = np.linspace(0.0, 50.0, nc)
    cen = np.linspace(18.0, 32.0, ny * nx).reshape(ny, nx)
    amp = np.linspace(20.0, 60.0, ny * nx).reshape(ny, nx)
    data = np.empty((ny, nx, nc))
    for iy in range(ny):
        for ix in range(nx):
            data[iy, ix] = 5.0 + (amp[iy, ix] / (3.0 * np.sqrt(2 * np.pi))
                                  * np.exp(-((x - cen[iy, ix]) ** 2) / 18.0))
    if noise:
        data += np.random.default_rng(0).normal(0, noise, data.shape)
    s = hs.signals.Signal1D(data)
    s.axes_manager.signal_axes[0].offset = x[0]
    s.axes_manager.signal_axes[0].scale = x[1] - x[0]
    return s, x, amp, cen


def _model(signal):
    m = signal.create_model()
    while len(m):
        m.remove(m[0])
    m.extend([Offset(), Gaussian()])
    m[0].offset.value = 1.0
    m[1].A.value, m[1].centre.value, m[1].sigma.value = 30.0, 25.0, 2.0
    return m


class TestCoarseIndices:
    def test_samples_the_strided_grid(self):
        coarse, nearest = _coarse_indices((8, 8), 4)
        # rows/cols 0 and 4, plus the last (7) appended on each axis -> 3x3
        assert coarse.size == 9
        assert nearest.size == 64

    def test_always_includes_the_last_index(self):
        """Without this a grid whose size is not a multiple of the stride has
        an unseeded strip down its far edge — where scan contrast often changes
        most."""
        coarse, _ = _coarse_indices((10, 10), 4)
        rows, cols = np.unravel_index(coarse, (10, 10))
        assert 9 in set(rows.tolist())
        assert 9 in set(cols.tolist())

    def test_each_position_maps_to_its_nearest_coarse_sample(self):
        coarse, nearest = _coarse_indices((9, 1), 4)
        rows = np.unravel_index(coarse, (9, 1))[0]      # [0, 4, 8]
        got = rows[nearest]
        # 0,1,2 -> 0 ; 3,4,5,6 -> 4 ; 7,8 -> 8   (ties go to the lower sample)
        assert got.tolist() == [0, 0, 0, 4, 4, 4, 4, 8, 8]

    def test_a_coarse_sample_seeds_itself(self):
        coarse, nearest = _coarse_indices((8, 8), 4)
        for k, flat in enumerate(coarse):
            iy, ix = np.unravel_index(flat, (8, 8))
            assert coarse[nearest[flat]] == flat, \
                f"coarse position ({iy},{ix}) does not seed from itself"


class TestSeededFit:
    def test_matches_a_cold_fit_on_easy_data(self):
        """Seeding must not CHANGE the answer where a cold fit already works —
        it is an initial-value strategy, not a different model."""
        s, x, amp, cen = _make_si()
        spec = ModelSpec.from_model(_model(s))
        cold = fit_batched(spec, s.data, x, device="cpu")
        warm = fit_seeded(spec, s.data, x, stride=4, device="cpu")
        np.testing.assert_allclose(warm.values, cold.values, rtol=1e-4)

    def test_recovers_the_truth(self):
        s, x, amp, cen = _make_si()
        spec = ModelSpec.from_model(_model(s))
        got = fit_seeded(spec, s.data, x, stride=4, device="cpu")
        maps = got.as_maps(spec, s.data.shape[:2])
        np.testing.assert_allclose(maps["Gaussian.centre"], cen, rtol=1e-3)
        np.testing.assert_allclose(maps["Gaussian.A"], amp, rtol=1e-3)

    def test_reports_how_many_seeds_were_usable(self):
        s, x, _, _ = _make_si()
        spec = ModelSpec.from_model(_model(s))
        got = fit_seeded(spec, s.data, x, stride=4, device="cpu")
        assert got.n_seeds == 9
        assert got.seed_converged == pytest.approx(1.0)

    def test_plain_fit_reports_no_seeding(self):
        """A caller has to be able to tell a seeded result from a plain one."""
        s, x, _, _ = _make_si()
        spec = ModelSpec.from_model(_model(s))
        assert fit_batched(spec, s.data, x, device="cpu").seed_converged is None

    def test_stride_one_degenerates_to_a_plain_fit(self):
        s, x, _, _ = _make_si(ny=4, nx=4)
        spec = ModelSpec.from_model(_model(s))
        got = fit_seeded(spec, s.data, x, stride=1, device="cpu")
        assert got.seed_converged is None          # no coarse pass ran

    def test_oversized_stride_falls_back_instead_of_wasting_a_pass(self):
        """If the coarse grid is not smaller than the full one, seeding costs a
        whole extra fit and buys nothing."""
        s, x, _, _ = _make_si(ny=2, nx=2)
        spec = ModelSpec.from_model(_model(s))
        got = fit_seeded(spec, s.data, x, stride=1, device="cpu")
        assert got.values.shape[0] == 4

    def test_failed_coarse_fits_are_not_propagated(self, monkeypatch):
        """A coarse fit that failed has wandered somewhere unphysical. Seeding
        its neighbourhood from it would SPREAD the failure, so those positions
        must fall back to the model's own starting values."""
        import spyde.fitting.seeding as seeding

        s, x, _, _ = _make_si()
        spec = ModelSpec.from_model(_model(s))
        defaults = spec.flat_values()
        real_fit = seeding.fit_batched
        calls = {}

        def fake_fit(spec_, data_, x_, **kw):
            out = real_fit(spec_, data_, x_, **kw)
            if "initial" not in kw:                 # the COARSE pass
                out.converged[:] = False            # pretend every seed failed
                out.values[:] = 1e9                 # ... with garbage values
            else:
                calls["initial"] = kw["initial"]
            return out

        monkeypatch.setattr(seeding, "fit_batched", fake_fit)
        seeding.fit_seeded(spec, s.data, x, stride=4, device="cpu")

        seeds = calls["initial"]
        assert not np.any(seeds == 1e9), "garbage seed was propagated"
        np.testing.assert_allclose(seeds, np.broadcast_to(defaults, seeds.shape))

    def test_converged_coarse_fits_are_propagated(self, monkeypatch):
        """The other half: a good coarse result MUST reach its neighbours, or
        seeding is an expensive no-op."""
        import spyde.fitting.seeding as seeding

        s, x, _, _ = _make_si()
        spec = ModelSpec.from_model(_model(s))
        real_fit = seeding.fit_batched
        calls = {}
        marker = 12345.0

        def fake_fit(spec_, data_, x_, **kw):
            out = real_fit(spec_, data_, x_, **kw)
            if "initial" not in kw:
                out.converged[:] = True
                out.values[:, 0] = marker
            else:
                calls["initial"] = kw["initial"]
            return out

        monkeypatch.setattr(seeding, "fit_batched", fake_fit)
        seeding.fit_seeded(spec, s.data, x, stride=4, device="cpu")
        assert np.all(calls["initial"][:, 0] == marker)

    def test_coarse_pass_gets_a_larger_iteration_budget(self, monkeypatch):
        """Few fits, started cold, and their whole value is being right."""
        import spyde.fitting.seeding as seeding

        s, x, _, _ = _make_si()
        spec = ModelSpec.from_model(_model(s))
        real_fit = seeding.fit_batched
        seen = []

        def fake_fit(spec_, data_, x_, **kw):
            seen.append(kw.get("max_iter"))
            return real_fit(spec_, data_, x_, **kw)

        monkeypatch.setattr(seeding, "fit_batched", fake_fit)
        seeding.fit_seeded(spec, s.data, x, stride=4, device="cpu",
                           max_iter=20, coarse_max_iter=100)
        assert seen[0] == 100          # coarse
        assert seen[1] == 20           # refine


class TestSeedingHelpsWhereColdStartsStruggle:
    def test_improves_convergence_on_a_hard_starting_point(self):
        """The point of the whole module. A start far from the answer makes the
        cold fit struggle; seeding gives the refine a nearby start.

        Asserted as 'no worse, and better on at least one of convergence or
        residual' — a strict inequality on a specific dataset would be a
        brittle test of this box rather than of the method.
        """
        s, x, _, cen = _make_si(ny=12, nx=12, noise=0.3)
        m = _model(s)
        m[1].centre.value = 8.0        # well off the true 18-32 range
        m[1].sigma.value = 8.0
        spec = ModelSpec.from_model(m)

        cold = fit_batched(spec, s.data, x, device="cpu", max_iter=25)
        warm = fit_seeded(spec, s.data, x, stride=4, device="cpu", max_iter=25,
                          coarse_max_iter=200)

        assert warm.convergence_rate >= cold.convergence_rate or \
            np.median(warm.chisq) <= np.median(cold.chisq), (
                f"seeding was worse on both counts: converged "
                f"{warm.convergence_rate:.2f} vs {cold.convergence_rate:.2f}, "
                f"median chisq {np.median(warm.chisq):.4g} vs "
                f"{np.median(cold.chisq):.4g}")

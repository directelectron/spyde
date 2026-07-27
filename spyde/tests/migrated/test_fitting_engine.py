"""The batched engine must reproduce HyperSpy's `multifit`.

This is the acceptance gate for #53, and the reason it is written this way:
we are *replacing a reference implementation*, so "it converged" proves
nothing. The bar is that the parameters agree with the ones HyperSpy's own
optimiser finds on the same data.

Everything here runs on the CPU. torch-CUDA segfaults under pytest on Windows
(CLAUDE.md), and the engine is the same code on both devices — GPU is exercised
by the benchmark, run outside pytest.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hyperspy.api as hs
from hyperspy.components1d import Gaussian, Offset, PowerLaw

from spyde.fitting import ModelSpec
from spyde.fitting.engine import default_device, fit_batched


# --------------------------------------------------------------------------
# fixtures: a small spectrum image with known truth
# --------------------------------------------------------------------------

def _make_si(ny=4, nx=5, nc=256, noise=0.0, seed=0):
    """Offset + Gaussian, with the amplitude and centre varying across the
    scan so every pixel has a genuinely different answer (a constant field
    would let a broadcasting bug pass)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 50.0, nc)
    amp = np.linspace(20.0, 60.0, ny * nx).reshape(ny, nx)
    cen = np.linspace(20.0, 30.0, ny * nx).reshape(ny, nx)
    data = np.empty((ny, nx, nc))
    for iy in range(ny):
        for ix in range(nx):
            g = (amp[iy, ix] / (3.0 * np.sqrt(2 * np.pi))
                 * np.exp(-((x - cen[iy, ix]) ** 2) / (2 * 3.0 ** 2)))
            data[iy, ix] = 5.0 + g
    if noise:
        data += rng.normal(0.0, noise, data.shape)
    s = hs.signals.Signal1D(data)
    s.axes_manager.signal_axes[0].offset = x[0]
    s.axes_manager.signal_axes[0].scale = x[1] - x[0]
    return s, x, amp, cen


def _seed_model(signal):
    m = signal.create_model()
    m.extend([Offset(), Gaussian()])
    m[0].offset.value = 1.0
    m[1].A.value, m[1].centre.value, m[1].sigma.value = 30.0, 25.0, 2.0
    return m


class TestParityWithMultifit:
    def test_parameters_match_hyperspy_multifit(self):
        """THE acceptance test for the engine."""
        s, x, _amp, _cen = _make_si(noise=0.05)

        ref = _seed_model(s)
        ref.multifit(optimizer="lm", show_progressbar=False)
        ref_maps = np.stack([
            ref[0].offset.map["values"].ravel(),
            ref[1].A.map["values"].ravel(),
            ref[1].centre.map["values"].ravel(),
            ref[1].sigma.map["values"].ravel(),
        ], axis=1)

        spec = ModelSpec.from_model(_seed_model(s))
        got = fit_batched(spec, s.data, x, device="cpu")

        order = spec.parameter_names()
        cols = [order.index(n) for n in
                ["Offset.offset", "Gaussian.A", "Gaussian.centre", "Gaussian.sigma"]]
        np.testing.assert_allclose(got.values[:, cols], ref_maps,
                                   rtol=1e-4, atol=1e-6)

    def test_recovers_the_truth_it_was_built_from(self):
        s, x, amp, cen = _make_si(noise=0.0)
        spec = ModelSpec.from_model(_seed_model(s))
        got = fit_batched(spec, s.data, x, device="cpu")
        maps = got.as_maps(spec, s.data.shape[:2])
        np.testing.assert_allclose(maps["Gaussian.A"], amp, rtol=1e-4)
        np.testing.assert_allclose(maps["Gaussian.centre"], cen, rtol=1e-4)
        np.testing.assert_allclose(maps["Offset.offset"],
                                   np.full_like(amp, 5.0), atol=1e-3)

    def test_every_position_converges_on_clean_data(self):
        s, x, _, _ = _make_si(noise=0.0)
        got = fit_batched(ModelSpec.from_model(_seed_model(s)), s.data, x,
                          device="cpu")
        assert got.convergence_rate == 1.0

    def test_power_law_background_matches_multifit(self):
        """PowerLaw is the awkward one — a masked branch and a bounded
        exponent — and it is the EELS background, so it has to be right."""
        x = np.linspace(200.0, 800.0, 512)
        data = np.stack([1e6 * a * x ** -3.0 for a in (0.8, 1.0, 1.4, 2.0)])
        s = hs.signals.Signal1D(data)
        s.axes_manager.signal_axes[0].offset = x[0]
        s.axes_manager.signal_axes[0].scale = x[1] - x[0]

        def build():
            m = s.create_model()
            m.append(PowerLaw())
            m[0].A.value, m[0].r.value = 1e5, 2.0
            m[0].origin.free = False
            return m

        ref = build()
        ref.multifit(optimizer="lm", show_progressbar=False)
        spec = ModelSpec.from_model(build())
        got = fit_batched(spec, s.data, x, device="cpu")

        order = spec.parameter_names()
        for name, want in (("PowerLaw.A", ref[0].A.map["values"].ravel()),
                           ("PowerLaw.r", ref[0].r.map["values"].ravel())):
            np.testing.assert_allclose(got.values[:, order.index(name)], want,
                                       rtol=1e-3)


class TestFixedParametersAndBounds:
    def test_fixed_parameter_is_not_moved(self):
        s, x, _, _ = _make_si(noise=0.02)
        m = _seed_model(s)
        m[1].sigma.free = False
        m[1].sigma.value = 3.0
        spec = ModelSpec.from_model(m)
        got = fit_batched(spec, s.data, x, device="cpu")
        col = spec.parameter_names().index("Gaussian.sigma")
        np.testing.assert_allclose(got.values[:, col], 3.0, rtol=1e-12)

    def test_bounds_are_respected(self):
        s, x, _, _ = _make_si(noise=0.02)
        m = _seed_model(s)
        m[1].centre.bmin, m[1].centre.bmax = 24.0, 26.0
        spec = ModelSpec.from_model(m)
        got = fit_batched(spec, s.data, x, device="cpu")
        col = spec.parameter_names().index("Gaussian.centre")
        assert (got.values[:, col] >= 24.0 - 1e-9).all()
        assert (got.values[:, col] <= 26.0 + 1e-9).all()

    def test_all_parameters_fixed_is_a_clear_error(self):
        s, x, _, _ = _make_si()
        m = _seed_model(s)
        for c in m:
            for p in c.parameters:
                p.free = False
        with pytest.raises(ValueError, match="fixed"):
            fit_batched(ModelSpec.from_model(m), s.data, x, device="cpu")


class TestSignalRange:
    def test_masked_channels_do_not_influence_the_fit(self):
        """A channel outside the range must have NO effect — so corrupting it
        beyond recognition must not move the answer at all."""
        s, x, _, _ = _make_si(noise=0.0)
        m = _seed_model(s)
        m.set_signal_range(15.0, 40.0)
        spec = ModelSpec.from_model(m)

        clean = fit_batched(spec, s.data, x, device="cpu")
        wrecked = s.data.copy()
        wrecked[..., :int(0.2 * len(x))] += 1e6      # far outside the range
        dirty = fit_batched(spec, wrecked, x, device="cpu")
        np.testing.assert_allclose(clean.values, dirty.values, rtol=1e-8)

    def test_mask_length_mismatch_is_caught(self):
        s, x, _, _ = _make_si()
        spec = ModelSpec.from_model(_seed_model(s))
        spec.channel_mask = np.ones(7, bool)
        with pytest.raises(ValueError, match="channel mask"):
            fit_batched(spec, s.data, x, device="cpu")


class TestWeights:
    def test_poisson_weighting_runs_and_stays_sane(self):
        s, x, amp, _ = _make_si(noise=0.0)
        spec = ModelSpec.from_model(_seed_model(s))
        got = fit_batched(spec, s.data, x, weights="poisson", device="cpu")
        maps = got.as_maps(spec, s.data.shape[:2])
        np.testing.assert_allclose(maps["Gaussian.A"], amp, rtol=1e-3)

    def test_unknown_weighting_is_rejected(self):
        s, x, _, _ = _make_si()
        with pytest.raises(ValueError, match="weighting"):
            fit_batched(ModelSpec.from_model(_seed_model(s)), s.data, x,
                        weights="gaussianish", device="cpu")


class TestChunking:
    def test_chunked_and_unchunked_agree(self):
        """Chunking is a memory optimisation and must be numerically invisible;
        a chunk-boundary bug would otherwise show as a faint grid in the maps."""
        s, x, _, _ = _make_si(ny=6, nx=6, noise=0.02)
        spec = ModelSpec.from_model(_seed_model(s))
        whole = fit_batched(spec, s.data, x, device="cpu")
        pieces = fit_batched(spec, s.data, x, device="cpu", chunk=4)
        # NOT bit-identical, and it should not be asserted as such: each
        # position is an independent problem, but batched BLAS picks different
        # kernels/reduction orders for different batch sizes, so the last
        # couple of ULPs move. ~1e-8 relative is float noise; a real
        # chunk-boundary bug would show up orders of magnitude larger (and as a
        # visible grid in the maps).
        np.testing.assert_allclose(whole.values, pieces.values, rtol=1e-6)

    def test_result_shape_follows_the_navigation_grid(self):
        s, x, _, _ = _make_si(ny=3, nx=7)
        spec = ModelSpec.from_model(_seed_model(s))
        got = fit_batched(spec, s.data, x, device="cpu")
        assert got.values.shape == (21, len(spec.parameter_names()))
        assert got.as_maps(spec, (3, 7))["Gaussian.A"].shape == (3, 7)


class TestInitialValues:
    def test_per_position_seeds_are_used(self):
        """The hand-off seeded propagation (#54) depends on: each position may
        start from its own values, not one shared guess."""
        s, x, amp, cen = _make_si(noise=0.0)
        spec = ModelSpec.from_model(_seed_model(s))
        P, n = s.data.shape[0] * s.data.shape[1], len(spec.parameter_names())
        seeds = np.broadcast_to(spec.flat_values(), (P, n)).copy()
        seeds[:, spec.parameter_names().index("Gaussian.centre")] = cen.ravel()
        got = fit_batched(spec, s.data, x, device="cpu", initial=seeds)
        np.testing.assert_allclose(
            got.as_maps(spec, cen.shape)["Gaussian.centre"], cen, rtol=1e-4)

    def test_seeds_outside_bounds_are_clipped_not_rejected(self):
        s, x, _, _ = _make_si(noise=0.0)
        m = _seed_model(s)
        m[1].centre.bmin, m[1].centre.bmax = 20.0, 30.0
        spec = ModelSpec.from_model(m)
        P, n = s.data.shape[0] * s.data.shape[1], len(spec.parameter_names())
        seeds = np.broadcast_to(spec.flat_values(), (P, n)).copy()
        seeds[:, spec.parameter_names().index("Gaussian.centre")] = 1e6
        got = fit_batched(spec, s.data, x, device="cpu", initial=seeds)
        col = spec.parameter_names().index("Gaussian.centre")
        assert (got.values[:, col] <= 30.0 + 1e-9).all()


class TestGuards:
    def test_unsupported_component_refuses_rather_than_dropping_it(self):
        """Silently omitting a component would fit a DIFFERENT model and still
        return a plausible answer."""
        from spyde.fitting.spec import ComponentSpec, ParameterSpec
        s, x, _, _ = _make_si()
        spec = ModelSpec(components=[
            ComponentSpec(kind="Voigt", parameters=[ParameterSpec("area", 1.0)])])
        with pytest.raises(NotImplementedError, match="Voigt"):
            fit_batched(spec, s.data, x, device="cpu")

    def test_axis_length_mismatch_is_caught(self):
        s, _x, _, _ = _make_si()
        spec = ModelSpec.from_model(_seed_model(s))
        with pytest.raises(ValueError, match="signal axis"):
            fit_batched(spec, s.data, np.linspace(0, 1, 9), device="cpu")

    def test_progress_callback_reports_completion(self):
        s, x, _, _ = _make_si(ny=4, nx=4)
        seen = []
        fit_batched(ModelSpec.from_model(_seed_model(s)), s.data, x,
                    device="cpu", chunk=5, progress=lambda d, t: seen.append((d, t)))
        assert seen and seen[-1] == (16, 16)

    def test_a_failing_progress_callback_does_not_kill_the_fit(self):
        s, x, _, _ = _make_si(ny=2, nx=2)
        def boom(done, total):
            raise RuntimeError("renderer went away")
        got = fit_batched(ModelSpec.from_model(_seed_model(s)), s.data, x,
                          device="cpu", progress=boom)
        assert got.values.shape[0] == 4


class TestDevice:
    def test_default_device_is_reported(self):
        assert default_device() in ("cpu", "cuda")

    def test_cpu_is_always_usable(self):
        s, x, _, _ = _make_si(ny=2, nx=2)
        got = fit_batched(ModelSpec.from_model(_seed_model(s)), s.data, x,
                          device="cpu")
        assert got.device == "cpu"

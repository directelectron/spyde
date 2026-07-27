"""Every batched torch component must match the HyperSpy component it ports.

This file is the actual specification for `spyde/fitting/components.py`. The
formulas there were transcribed from HyperSpy's `expression=` strings, and the
only thing that makes that transcription trustworthy is checking it numerically
against the real component at randomised parameter values.

Two failure modes this is built to catch, both silent:

* **Wrong parameter ORDER.** HyperSpy's order is neither alphabetical nor the
  constructor's — `PowerLaw` is (A, left_cutoff, origin, r). A mismatch fits
  the wrong parameter and still converges to something.
* **Wrong convention.** A HyperSpy `Gaussian`'s `A` is the AREA, not the peak
  height. A height-based port produces a plausible curve that is simply not
  HyperSpy's.

CPU only: these are tiny tensors and torch-CUDA segfaults under pytest on
Windows (CLAUDE.md). GPU correctness is exercised by the engine's own
subprocess test.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hyperspy.components1d as c1d

from spyde.fitting import components as tc
from spyde.fitting.spec import ModelSpec, ComponentSpec, ParameterSpec

# kind -> parameter values to test at, keyed by HyperSpy parameter name.
# Chosen to be physically sensible AND to avoid the degenerate spots
# (sigma=0, x<=origin for a power law) that would make any port agree.
CASES = {
    "Gaussian":      {"A": 12.0, "centre": 3.0, "sigma": 1.7},
    "GaussianHF":    {"centre": 2.0, "fwhm": 3.1, "height": 5.0},
    "Lorentzian":    {"A": 8.0, "centre": -1.0, "gamma": 2.2},
    "PowerLaw":      {"A": 900.0, "left_cutoff": 0.0, "origin": -12.0, "r": 2.6},
    "Offset":        {"offset": 3.25},
    "Exponential":   {"A": 40.0, "tau": 6.5},
    "Arctan":        {"A": 2.5, "k": 1.3, "x0": 1.0},
    "Erf":           {"A": 7.0, "origin": 0.5, "sigma": 2.0},
    "HeavisideStep": {"A": 4.0, "n": 1.5},
    "Logistic":      {"a": 5.0, "b": 2.0, "c": 0.7, "origin": 1.0},
}

X = np.linspace(-8.0, 12.0, 401)


def _hyperspy_component(kind: str):
    comp = getattr(c1d, kind)()
    for name, value in CASES[kind].items():
        getattr(comp, name).value = value
    return comp


def _torch_eval(kind: str, x: np.ndarray) -> np.ndarray:
    comp = _hyperspy_component(kind)
    batched = tc.get_component(kind)
    values = np.array([[getattr(comp, n).value for n in batched.params]])
    y = batched(torch.as_tensor(x, dtype=torch.float64),
                torch.as_tensor(values, dtype=torch.float64))
    return y.detach().numpy()[0]


@pytest.mark.parametrize("kind", sorted(CASES))
class TestParity:
    def test_parameter_order_matches_hyperspy(self, kind):
        """The port's tuple must BE HyperSpy's `component.parameters` order."""
        hs_order = tuple(p.name for p in _hyperspy_component(kind).parameters)
        assert tc.get_component(kind).params == hs_order

    def test_linear_flags_match_hyperspy(self, kind):
        """Variable projection (#53) trusts these; they must come from HyperSpy."""
        comp = _hyperspy_component(kind)
        hs_linear = tuple(bool(getattr(p, "_linear", False))
                          for p in comp.parameters)
        assert tc.get_component(kind).linear == hs_linear

    def test_values_match_hyperspy(self, kind):
        """The whole point: same parameters, same curve."""
        expected = np.asarray(_hyperspy_component(kind).function(X), float)
        got = _torch_eval(kind, X)
        assert got.shape == expected.shape
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_is_differentiable(self, kind):
        """The engine gets its Jacobian from autograd, so every parameter must
        carry a finite gradient — a NaN in a discarded `where` branch would
        poison it even where the mask throws the value away."""
        batched = tc.get_component(kind)
        comp = _hyperspy_component(kind)
        values = torch.tensor(
            [[getattr(comp, n).value for n in batched.params]],
            dtype=torch.float64, requires_grad=True)
        y = batched(torch.as_tensor(X, dtype=torch.float64), values)
        y.sum().backward()
        assert values.grad is not None
        assert torch.isfinite(values.grad).all(), \
            f"{kind} produced a non-finite gradient: {values.grad}"


@pytest.mark.parametrize("kind", sorted(CASES))
class TestAnalyticGradient:
    """Analytic derivatives must equal autodiff's.

    Autodiff is the perfect oracle here: it is derived mechanically from the
    value function, so it cannot share a mistake with a hand-written formula.
    The analytic path exists purely for speed (autodiff costs one forward pass
    per parameter — 51.6 ms vs 3.4 ms for a residual on a 13-parameter model),
    so it must be numerically indistinguishable, never an approximation.
    """

    def test_matches_autodiff(self, kind):
        batched = tc.get_component(kind)
        if not batched.has_analytic_grad:
            pytest.skip(f"{kind} has no analytic gradient")
        comp = _hyperspy_component(kind)
        vals = torch.tensor([[getattr(comp, n).value for n in batched.params]],
                            dtype=torch.float64, requires_grad=True)
        xt = torch.as_tensor(X, dtype=torch.float64)

        from torch.func import jacfwd
        auto = jacfwd(lambda p: batched(xt, p.unsqueeze(0)).squeeze(0))(
            vals.detach()[0])                       # (C, n)
        got = batched.grad(xt, vals.detach())[0]    # (C, n)
        assert got.shape == auto.shape
        torch.testing.assert_close(got, auto, rtol=1e-9, atol=1e-9)

    def test_gradient_is_finite_everywhere(self, kind):
        batched = tc.get_component(kind)
        if not batched.has_analytic_grad:
            pytest.skip(f"{kind} has no analytic gradient")
        comp = _hyperspy_component(kind)
        vals = torch.tensor([[getattr(comp, n).value for n in batched.params]],
                            dtype=torch.float64)
        g = batched.grad(torch.as_tensor(X, dtype=torch.float64), vals)
        assert torch.isfinite(g).all(), f"{kind} analytic gradient not finite"


class TestWholeModelGradient:
    def test_evaluate_with_grad_matches_autodiff(self):
        """The assembled model Jacobian, not just one component's block."""
        from torch.func import jacfwd
        from spyde.fitting.spec import ComponentSpec, ParameterSpec

        spec = ModelSpec(components=[
            ComponentSpec(kind="PowerLaw", parameters=[
                ParameterSpec("A", 900.0), ParameterSpec("left_cutoff", 0.0),
                ParameterSpec("origin", -12.0), ParameterSpec("r", 2.6)]),
            ComponentSpec(kind="Gaussian", parameters=[
                ParameterSpec("A", 12.0), ParameterSpec("centre", 3.0),
                ParameterSpec("sigma", 1.7)]),
        ])
        xt = torch.as_tensor(X, dtype=torch.float64)
        v = torch.as_tensor(spec.flat_values()[None, :], dtype=torch.float64)

        value, jac = tc.evaluate_with_grad(spec, xt, v)
        auto_v = tc.evaluate(spec, xt, v)
        auto_j = jacfwd(lambda p: tc.evaluate(spec, xt, p.unsqueeze(0)
                                              ).squeeze(0))(v[0])
        torch.testing.assert_close(value, auto_v)
        torch.testing.assert_close(jac[0], auto_j, rtol=1e-9, atol=1e-9)

    def test_columns_follow_the_packed_order(self):
        """Column j must be parameter j — a shifted block would fit the wrong
        parameter and still converge to something."""
        from spyde.fitting.spec import ComponentSpec, ParameterSpec
        spec = ModelSpec(components=[
            ComponentSpec(kind="Offset",
                          parameters=[ParameterSpec("offset", 2.0)]),
            ComponentSpec(kind="Gaussian", parameters=[
                ParameterSpec("A", 12.0), ParameterSpec("centre", 3.0),
                ParameterSpec("sigma", 1.7)]),
        ])
        xt = torch.as_tensor(X, dtype=torch.float64)
        _, jac = tc.evaluate_with_grad(
            spec, xt, torch.as_tensor(spec.flat_values()[None, :],
                                      dtype=torch.float64))
        assert spec.parameter_names()[0] == "Offset.offset"
        # d(model)/d(offset) is 1 everywhere; nothing else is.
        torch.testing.assert_close(jac[0, :, 0], torch.ones_like(xt))
        assert not torch.allclose(jac[0, :, 1], torch.ones_like(xt))

    def test_reports_analytic_availability(self):
        from spyde.fitting.spec import ComponentSpec, ParameterSpec
        ok = ModelSpec(components=[ComponentSpec(
            kind="Gaussian", parameters=[ParameterSpec(n) for n in
                                         ("A", "centre", "sigma")])])
        assert tc.has_analytic_grad(ok) is True
        unsupported = ModelSpec(components=[ComponentSpec(
            kind="Voigt", parameters=[ParameterSpec("area")])])
        assert tc.has_analytic_grad(unsupported) is False


class TestBatching:
    def test_one_call_evaluates_every_position_independently(self):
        """P rows in, P different curves out — no broadcasting mistake that
        makes every pixel share one parameter set."""
        batched = tc.get_component("Gaussian")
        vals = torch.tensor([[10.0, 0.0, 1.0],
                             [10.0, 4.0, 1.0],
                             [20.0, 0.0, 2.0]], dtype=torch.float64)
        y = batched(torch.as_tensor(X, dtype=torch.float64), vals)
        assert y.shape == (3, len(X))
        assert X[int(y[0].argmax())] == pytest.approx(0.0, abs=0.1)
        assert X[int(y[1].argmax())] == pytest.approx(4.0, abs=0.1)
        assert not torch.allclose(y[0], y[2])

    def test_batched_equals_looping_one_at_a_time(self):
        rng = np.random.default_rng(0)
        vals = np.stack([rng.uniform([1, -3, 0.5], [20, 6, 3.0]) for _ in range(16)])
        x = torch.as_tensor(X, dtype=torch.float64)
        batched = tc.get_component("Gaussian")
        together = batched(x, torch.as_tensor(vals, dtype=torch.float64))
        for i in range(len(vals)):
            one = batched(x, torch.as_tensor(vals[i:i + 1], dtype=torch.float64))
            torch.testing.assert_close(together[i], one[0])

    def test_offset_broadcasts_over_the_signal_axis(self):
        """Offset ignores x, which makes it the easy one to get wrong: it must
        still return a full (P, C) block, not a (P, 1) that broadcasts later."""
        y = tc.get_component("Offset")(
            torch.as_tensor(X, dtype=torch.float64),
            torch.tensor([[2.0], [5.0]], dtype=torch.float64))
        assert y.shape == (2, len(X))
        assert torch.allclose(y[0], torch.full_like(y[0], 2.0))
        assert torch.allclose(y[1], torch.full_like(y[1], 5.0))


class TestPolynomial:
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_matches_hyperspy_at_each_order(self, order):
        comp = c1d.Polynomial(order=order)
        for k in range(order + 1):
            getattr(comp, f"a{k}").value = 0.5 * (k + 1)
        batched = tc.get_component("Polynomial", n_params=order + 1)
        assert batched.params == tuple(p.name for p in comp.parameters)
        vals = np.array([[getattr(comp, n).value for n in batched.params]])
        got = batched(torch.as_tensor(X, dtype=torch.float64),
                      torch.as_tensor(vals, dtype=torch.float64)).numpy()[0]
        np.testing.assert_allclose(got, np.asarray(comp.function(X), float),
                                   rtol=1e-10, atol=1e-12)

    def test_needs_an_explicit_order(self):
        with pytest.raises(ValueError, match="n_params"):
            tc.get_component("Polynomial")


class TestPowerLawEdgeCases:
    def test_zero_below_the_cutoff(self):
        y = _torch_eval("PowerLaw", np.array([-5.0, -1.0, 0.0, 1.0, 5.0]))
        assert y[0] == 0.0 and y[1] == 0.0 and y[2] == 0.0
        assert y[3] > 0 and y[4] > 0

    def test_no_nan_gradient_from_the_dead_branch(self):
        """`(x - origin) ** -r` is inf/NaN where x <= origin. A naive
        implementation masks the VALUE but still back-propagates the NaN."""
        batched = tc.get_component("PowerLaw")
        vals = torch.tensor([[900.0, 0.0, 2.0, 2.6]], dtype=torch.float64,
                            requires_grad=True)
        x = torch.linspace(-5, 20, 200, dtype=torch.float64)   # spans origin=2
        batched(x, vals).sum().backward()
        assert torch.isfinite(vals.grad).all()


class TestRegistry:
    def test_unknown_component_names_what_is_available(self):
        with pytest.raises(NotImplementedError) as e:
            tc.get_component("Flurbulator")
        assert "Gaussian" in str(e.value)

    def test_supports_reports_true_for_a_portable_model(self):
        spec = ModelSpec(components=[
            ComponentSpec(kind="PowerLaw", parameters=[
                ParameterSpec(n) for n in ("A", "left_cutoff", "origin", "r")]),
            ComponentSpec(kind="Gaussian", parameters=[
                ParameterSpec(n) for n in ("A", "centre", "sigma")]),
        ])
        assert tc.supports(spec) is True

    def test_supports_reports_false_rather_than_dropping_a_component(self):
        """The engine must fall back to HyperSpy for a model it cannot fully
        evaluate — silently omitting the unsupported component would fit a
        different model and still look like it worked."""
        spec = ModelSpec(components=[
            ComponentSpec(kind="Gaussian", parameters=[
                ParameterSpec(n) for n in ("A", "centre", "sigma")]),
            ComponentSpec(kind="Voigt", parameters=[ParameterSpec("area")]),
        ])
        assert tc.supports(spec) is False

    def test_inactive_unsupported_component_does_not_block(self):
        spec = ModelSpec(components=[
            ComponentSpec(kind="Gaussian", parameters=[
                ParameterSpec(n) for n in ("A", "centre", "sigma")]),
            ComponentSpec(kind="Voigt", active=False,
                          parameters=[ParameterSpec("area")]),
        ])
        assert tc.supports(spec) is True


class TestEvaluateSpec:
    def test_sums_active_components_and_matches_hyperspy(self):
        """A whole model, not just one component — the sum is what the engine
        actually fits."""
        import hyperspy.api as hs
        sig = hs.signals.Signal1D(np.zeros((1, len(X))))
        sig.axes_manager.signal_axes[0].offset = X[0]
        sig.axes_manager.signal_axes[0].scale = X[1] - X[0]

        m = sig.create_model()
        m.extend([c1d.Offset(), c1d.Gaussian()])
        m[0].offset.value = 2.0
        m[1].A.value, m[1].centre.value, m[1].sigma.value = 12.0, 3.0, 1.7
        expected = np.asarray(m[0].function(X), float) + \
            np.asarray(m[1].function(X), float)

        spec = ModelSpec.from_model(m)
        got = tc.evaluate(spec, torch.as_tensor(X, dtype=torch.float64),
                          torch.as_tensor(spec.flat_values()[None, :],
                                          dtype=torch.float64)).numpy()[0]
        np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    def test_inactive_component_contributes_nothing(self):
        spec = ModelSpec(components=[
            ComponentSpec(kind="Offset",
                          parameters=[ParameterSpec("offset", value=2.0)]),
            ComponentSpec(kind="Offset", name="off2", active=False,
                          parameters=[ParameterSpec("offset", value=100.0)]),
        ])
        y = tc.evaluate(spec, torch.as_tensor(X, dtype=torch.float64),
                        torch.as_tensor(spec.flat_values()[None, :],
                                        dtype=torch.float64))
        assert torch.allclose(y, torch.full_like(y, 2.0))

"""The batched EELS edge must agree with exspy, including its fine structure.

`EELSCLEdge` is not a formula — exspy integrates a GOS table. But that integral
depends on the ELEMENT and the MICROSCOPE, not the pixel, so it is the same
curve for every spectrum in a scan and is computed once. What is left per pixel
is linear or nearly so: `intensity` scales the edge, and the fine-structure
coefficients are cubic B-spline weights, which means the edge is EXACTLY
`base + sum(c_i * basis_i)`.

exspy is the oracle throughout. If the batched evaluation and exspy's own
`function` disagree, the batched one is wrong.

> This replaced a private `TabulatedShape` component that fitted intensity and
> a shift and discarded the fine structure — which was both unnecessary (the
> fine structure is the LINEAR part, the cheapest thing to batch) and harmful
> (a private component cannot be stored by HyperSpy, so a fitted EELS model
> could not be saved with its own dataset).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("exspy")

import hyperspy.api as hs

from spyde.data import eels_si
from spyde.fitting import ModelSpec
from spyde.fitting import components as tcomp
from spyde.spectroscopy import model_for_composition, prepare_eels_edges


@pytest.fixture(scope="module")
def eels():
    """An EELS model with fine structure ON — otherwise the coefficients do
    nothing and the tests that matter here would all skip."""
    s = eels_si(nav=(3, 3), n_channels=512)
    s.add_elements(["C", "N", "O"])
    spec, _info = model_for_composition(s, ["C", "N", "O"])
    model = spec.to_model(s)
    for c in model:
        if type(c).__name__ == "EELSCLEdge":
            c.fine_structure_active = True
    return s, ModelSpec.from_model(model)


def _edges(spec):
    return [c for c in spec.components if c.kind == "EELSCLEdge"]


class TestItStaysAnExspyEdge:
    def test_the_kind_is_unchanged(self, eels):
        _s, spec = eels
        prepared, info = prepare_eels_edges(spec, _s)
        assert info["prepared"], "no edges were prepared"
        assert {c.kind for c in _edges(prepared)} == {"EELSCLEdge"}

    def test_the_model_still_rebuilds_in_hyperspy(self, eels):
        """The whole point of keeping the kind: a prepared model is still a
        real exspy model, so it stores on the signal and loads back."""
        s, spec = eels
        prepared, _ = prepare_eels_edges(spec, s)
        model = prepared.to_model(s)
        kinds = [type(c).__name__ for c in model]
        assert kinds.count("EELSCLEdge") == len(_edges(prepared))

    def test_the_batched_engine_accepts_it(self, eels):
        s, spec = eels
        prepared, _ = prepare_eels_edges(spec, s)
        assert tcomp.supports(prepared), "the prepared model is still unfittable"

    def test_the_private_component_is_gone(self):
        assert not hasattr(tcomp, "TABULATED_KIND")
        assert not hasattr(tcomp, "tabulated_component")
        assert "TabulatedShape" not in tcomp.available()


class TestParityWithExspy:
    """The batched evaluation against exspy's own `function`."""

    @staticmethod
    def _one(s, spec):
        prepared, _ = prepare_eels_edges(spec, s)
        edge = _edges(prepared)[0]
        model = prepared.to_model(s)
        live = [c for c in model if c.name == edge.name][0]
        x = np.asarray(s.axes_manager.signal_axes[0].axis, float)
        return prepared, edge, live, x

    def _evaluate(self, edge, x, values):
        comp = tcomp.component_for(edge)
        return comp(torch.as_tensor(x),
                    torch.as_tensor(np.array([values], float))).numpy()[0]

    def test_the_base_shape_matches(self, eels):
        s, spec = eels
        _prepared, edge, live, x = self._one(s, spec)
        names = [p.name for p in edge.scalar_parameters]
        values = [float(p.value) for p in edge.scalar_parameters]
        values[names.index("intensity")] = 1.0
        for i, n in enumerate(names):
            if n.startswith("fine_structure_coeff_"):
                values[i] = 0.0

        live.intensity.value = 1.0
        live.onset_energy.value = values[names.index("onset_energy")]
        if hasattr(live, "fine_structure_coeff") and live.fine_structure_active:
            live.fine_structure_coeff.value = tuple(
                0.0 for _ in np.ravel(live.fine_structure_coeff.value))
        want = np.asarray(live.function(x), float)
        got = self._evaluate(edge, x, values)
        scale = max(float(np.max(np.abs(want))), 1e-30)
        assert np.max(np.abs(got - want)) / scale < 1e-9

    def test_intensity_scales_it_linearly(self, eels):
        s, spec = eels
        _prepared, edge, _live, x = self._one(s, spec)
        names = [p.name for p in edge.scalar_parameters]
        base = [float(p.value) for p in edge.scalar_parameters]
        base[names.index("intensity")] = 1.0
        one = self._evaluate(edge, x, base)
        base[names.index("intensity")] = 7.0
        seven = self._evaluate(edge, x, base)
        assert np.allclose(seven, 7.0 * one, rtol=1e-12, atol=1e-12)

    def test_the_fine_structure_coefficients_are_fitted_not_frozen(self, eels):
        """The regression this whole change exists for: changing a coefficient
        must change the curve. Under `TabulatedShape` it could not."""
        s, spec = eels
        _prepared, edge, _live, x = self._one(s, spec)
        names = [p.name for p in edge.scalar_parameters]
        coeffs = [i for i, n in enumerate(names)
                  if n.startswith("fine_structure_coeff_")]
        if not coeffs:
            pytest.skip("this edge has fine structure switched off")
        values = [float(p.value) for p in edge.scalar_parameters]
        before = self._evaluate(edge, x, values)
        values[coeffs[0]] = values[coeffs[0]] + 5.0
        after = self._evaluate(edge, x, values)
        assert not np.allclose(before, after), \
            "a fine-structure coefficient did nothing — it is frozen"

    def test_it_matches_exspy_for_RANDOM_coefficients(self, eels):
        """The linearity claim, checked rather than assumed: the edge is
        exactly `base + sum(c_i * basis_i)`, so arbitrary coefficients must
        reproduce exspy's own curve."""
        s, spec = eels
        _prepared, edge, live, x = self._one(s, spec)
        names = [p.name for p in edge.scalar_parameters]
        coeffs = [i for i, n in enumerate(names)
                  if n.startswith("fine_structure_coeff_")]
        if not coeffs:
            pytest.skip("this edge has fine structure switched off")

        rng = np.random.default_rng(0)
        values = [float(p.value) for p in edge.scalar_parameters]
        values[names.index("intensity")] = 1.0
        live.intensity.value = 1.0
        live.onset_energy.value = values[names.index("onset_energy")]
        for _ in range(3):
            c = rng.normal(0.0, 3.0, len(coeffs))
            for j, i in enumerate(coeffs):
                values[i] = float(c[j])
            live.fine_structure_coeff.value = tuple(float(v) for v in c)
            want = np.asarray(live.function(x), float)
            got = self._evaluate(edge, x, values)
            scale = max(float(np.max(np.abs(want))), 1e-30)
            assert np.max(np.abs(got - want)) / scale < 1e-9

    def test_the_onset_slides_the_edge(self, eels):
        s, spec = eels
        _prepared, edge, _live, x = self._one(s, spec)
        names = [p.name for p in edge.scalar_parameters]
        i_on = names.index("onset_energy")
        values = [float(p.value) for p in edge.scalar_parameters]
        before = self._evaluate(edge, x, values)
        values[i_on] = values[i_on] + 10.0
        after = self._evaluate(edge, x, values)
        # Moving the onset UP in energy moves the edge to the right, so the
        # curve just past the original onset drops.
        assert not np.allclose(before, after)
        rise_before = int(np.argmax(before > 0.5 * np.max(before)))
        rise_after = int(np.argmax(after > 0.5 * np.max(after)))
        assert rise_after > rise_before, "the edge moved the wrong way"


class TestGradients:
    """Analytic derivatives against autodiff, which is the ideal oracle."""

    def test_every_column_matches_autodiff(self, eels):
        s, spec = eels
        prepared, _ = prepare_eels_edges(spec, s)
        edge = _edges(prepared)[0]
        comp = tcomp.component_for(edge)
        assert comp.has_analytic_grad

        x = torch.as_tensor(
            np.asarray(s.axes_manager.signal_axes[0].axis, float))
        p = torch.as_tensor(
            np.array([[float(q.value) for q in edge.scalar_parameters]]),
            dtype=torch.float64).requires_grad_(True)

        analytic = comp.grad(x, p).detach().numpy()[0]     # (C, n)
        # jacobian of (C,) w.r.t. (1, n) is (C, 1, n) — squeeze the batch.
        auto = torch.autograd.functional.jacobian(
            lambda q: comp(x, q)[0], p).detach().numpy()
        auto = auto.reshape(auto.shape[0], -1)             # (C, n)
        assert analytic.shape == auto.shape
        scale = max(float(np.max(np.abs(auto))), 1e-30)
        names = [p.name for p in edge.scalar_parameters]
        i_on = names.index("onset_energy")
        lin = [i for i in range(len(names)) if i != i_on]
        # intensity and every coefficient are linear — exact.
        assert np.max(np.abs(analytic[:, lin] - auto[:, lin])) / scale < 1e-9

        # The onset column is a CENTRAL DIFFERENCE by design: the exact
        # piecewise-linear derivative is a step function that jumps at every
        # segment boundary, so LM chatters as the onset crosses a channel. The
        # two therefore differ AT the kinks — the edge onset, the ends of the
        # fine-structure window — and nowhere else. Assert that shape rather
        # than loosening the tolerance across the board, which would hide a
        # genuinely wrong column.
        d = np.abs(analytic[:, i_on] - auto[:, i_on]) / scale
        differing = int((d > 1e-3).sum())
        assert differing <= 6, (
            f"the onset derivative differs from autodiff at {differing} "
            f"channels — expected only the handful of kinks")
        assert float(np.median(d)) < 1e-9


class TestRoundTrip:
    def test_the_coefficients_survive_the_trip_to_hyperspy(self, eels):
        """They are carried as one scalar per column (`fine_structure_coeff_0`,
        `_1`, …) because the packed vector holds one scalar per column, and
        reassembled into the vector on the way back."""
        s, spec = eels
        prepared, _ = prepare_eels_edges(spec, s)
        edge = _edges(prepared)[0]
        names = [p.name for p in edge.scalar_parameters]
        coeffs = [n for n in names if n.startswith("fine_structure_coeff_")]
        if not coeffs:
            pytest.skip("this edge has fine structure switched off")
        for i, n in enumerate(coeffs):
            edge[n].value = float(i + 1)

        model = prepared.to_model(s)
        live = [c for c in model if c.name == edge.name][0]
        got = list(np.ravel(np.asarray(live.fine_structure_coeff.value, float)))
        assert got[:len(coeffs)] == pytest.approx(
            [float(i + 1) for i in range(len(coeffs))])

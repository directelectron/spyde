"""Fitted intensities -> composition maps (#66).

Split the way the code is, because the two halves fail differently:

* The BRIDGE is where a naming or ordering mistake sends iron's intensity into
  copper's map. That is silent — every downstream number stays plausible — so
  it gets the most tests.
* The PHYSICS is where a wrong k-factor lives, which is at least a number a
  microscopist can sanity-check.

The end-to-end test scores the result against the concentration maps the
synthetic data was generated from, so "quantification works" means "it recovers
the composition we put in", not "it produced numbers that sum to 1".
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spyde.fitting import ModelSpec
from spyde.fitting.engine import FitResult
from spyde.fitting.spec import ComponentSpec, ParameterSpec
from spyde.spectroscopy.quantify import (
    element_intensity_maps,
    parse_line,
    quantify,
    quantify_result,
)


def _spec(*names_kinds):
    comps = []
    for name, kind in names_kinds:
        pname = {"Gaussian": "A", "EELSCLEdge": "intensity"}[kind]
        comps.append(ComponentSpec(
            kind=kind, name=name,
            parameters=[ParameterSpec(pname, 1.0, linear=True)]
            + ([ParameterSpec("centre", 0.0), ParameterSpec("sigma", 1.0)]
               if kind == "Gaussian" else [ParameterSpec("onset_shift", 0.0)])))
    return ModelSpec(components=comps)


def _result(spec, values):
    v = np.asarray(values, float)
    return FitResult(v, np.ones(len(v), bool), np.zeros(len(v)), 1, "cpu")


class TestParseLine:
    @pytest.mark.parametrize("name,want", [
        ("Fe_Ka", ("Fe", "Ka")), ("O_K", ("O", "K")),
        ("Cu_Kb1", ("Cu", "Kb1")), ("C_K", ("C", "K")),
    ])
    def test_parses_element_and_line(self, name, want):
        assert parse_line(name) == want

    @pytest.mark.parametrize("name", ["background_order_6", "PowerLaw",
                                      "", "Offset", "not-a-line"])
    def test_rejects_non_lines(self, name):
        assert parse_line(name) is None


class TestBridge:
    def test_one_map_per_element(self):
        spec = _spec(("Fe_Ka", "Gaussian"), ("Cu_Ka", "Gaussian"))
        res = _result(spec, [[10.0, 0, 1, 4.0, 0, 1],
                             [20.0, 0, 1, 8.0, 0, 1]])
        maps = element_intensity_maps(spec, res)
        assert set(maps) == {"Fe", "Cu"}
        np.testing.assert_allclose(maps["Fe"], [10.0, 20.0])
        np.testing.assert_allclose(maps["Cu"], [4.0, 8.0])

    def test_sums_a_family(self):
        """Ka + Kb belong to the same element and must add, not overwrite."""
        spec = _spec(("Fe_Ka", "Gaussian"), ("Fe_Kb", "Gaussian"))
        res = _result(spec, [[10.0, 0, 1, 1.3, 0, 1]])
        assert element_intensity_maps(spec, res)["Fe"][0] == pytest.approx(11.3)

    def test_lines_filter_selects_one_line(self):
        """Summing a family is usually right, but a contaminated member is
        worse than none — so a clean line can be selected instead."""
        spec = _spec(("Fe_Ka", "Gaussian"), ("Fe_Kb", "Gaussian"))
        res = _result(spec, [[10.0, 0, 1, 1.3, 0, 1]])
        maps = element_intensity_maps(spec, res, lines={"Fe": "Ka"})
        assert maps["Fe"][0] == pytest.approx(10.0)

    def test_background_is_ignored(self):
        spec = ModelSpec(components=[
            ComponentSpec(kind="Offset", name="background_order_6",
                          parameters=[ParameterSpec("offset", 5.0)]),
            ComponentSpec(kind="Gaussian", name="Fe_Ka", parameters=[
                ParameterSpec("A", 1.0), ParameterSpec("centre", 0.0),
                ParameterSpec("sigma", 1.0)]),
        ])
        res = _result(spec, [[99.0, 7.0, 0, 1]])
        assert set(element_intensity_maps(spec, res)) == {"Fe"}

    def test_negative_intensities_are_clamped(self):
        """A fit can drive a line slightly negative on noise. A negative
        'amount of an element' is not physical AND it corrupts the
        normalisation for every other element, so it is clamped at source."""
        spec = _spec(("Fe_Ka", "Gaussian"), ("Cu_Ka", "Gaussian"))
        res = _result(spec, [[-3.0, 0, 1, 10.0, 0, 1]])
        maps = element_intensity_maps(spec, res)
        assert maps["Fe"][0] == 0.0

    def test_reshapes_to_the_scan(self):
        spec = _spec(("Fe_Ka", "Gaussian"))
        res = _result(spec, np.column_stack([np.arange(6.0),
                                             np.zeros(6), np.ones(6)]))
        assert element_intensity_maps(spec, res, (2, 3))["Fe"].shape == (2, 3)

    def test_reads_a_tabulated_edge_intensity(self):
        """EELS edges carry `intensity`, EDS lines carry `A` — one bridge must
        handle both or the EELS path silently produces no maps."""
        spec = _spec(("O_K", "EELSCLEdge"))
        res = _result(spec, [[7.0, 0.0]])
        assert element_intensity_maps(spec, res)["O"][0] == pytest.approx(7.0)

    def test_inactive_components_are_excluded(self):
        spec = _spec(("Fe_Ka", "Gaussian"), ("Cu_Ka", "Gaussian"))
        spec["Cu_Ka"].active = False
        res = _result(spec, [[10.0, 0, 1]])
        assert set(element_intensity_maps(spec, res)) == {"Fe"}


class TestQuantify:
    def test_relative_normalises_to_one(self):
        frac, info = quantify({"Fe": np.array([2.0]), "Cu": np.array([6.0])})
        assert frac["Fe"][0] == pytest.approx(0.25)
        assert frac["Cu"][0] == pytest.approx(0.75)
        assert info["method"] == "relative"

    def test_cliff_lorimer_applies_k_factors(self):
        """Equal intensities with a 2x k-factor must give a 2:1 composition."""
        frac, _ = quantify({"Fe": np.array([10.0]), "Cu": np.array([10.0])},
                           method="cliff_lorimer",
                           kfactors={"Fe": 2.0, "Cu": 1.0})
        assert frac["Fe"][0] == pytest.approx(2 / 3)
        assert frac["Cu"][0] == pytest.approx(1 / 3)

    def test_eels_divides_by_cross_section(self):
        frac, _ = quantify({"C": np.array([10.0]), "O": np.array([10.0])},
                           method="eels",
                           cross_sections={"C": 1.0, "O": 2.0})
        assert frac["C"][0] == pytest.approx(2 / 3)

    def test_missing_factors_default_and_are_reported(self):
        """Defaulting silently would make an uncalibrated number look
        calibrated."""
        _, info = quantify({"Fe": np.array([1.0]), "Cu": np.array([1.0])},
                           method="cliff_lorimer", kfactors={"Fe": 1.0})
        assert info["defaulted_factors"] == ["Cu"]

    def test_result_says_it_is_relative(self):
        """Cliff-Lorimer cannot know about an element you did not fit — the
        fractions are relative, and leaving an element out redistributes its
        share rather than causing a small error."""
        _, info = quantify({"Fe": np.array([1.0])})
        assert info["relative_to"] == ["Fe"]
        assert "redistributed" in info["note"]

    def test_zero_signal_is_nan_not_a_divide_error(self):
        frac, _ = quantify({"Fe": np.array([0.0, 5.0]),
                            "Cu": np.array([0.0, 5.0])})
        assert np.isnan(frac["Fe"][0])
        assert frac["Fe"][1] == pytest.approx(0.5)

    def test_fractions_sum_to_one_across_a_map(self):
        rng = np.random.default_rng(0)
        maps = {e: rng.random((4, 5)) + 0.1 for e in ("Fe", "Ni", "Cu")}
        frac, _ = quantify(maps)
        total = sum(frac.values())
        np.testing.assert_allclose(total, np.ones((4, 5)), atol=1e-12)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="unknown quantification"):
            quantify({"Fe": np.array([1.0])}, method="magic")

    def test_no_maps_is_an_error(self):
        with pytest.raises(ValueError, match="no intensity maps"):
            quantify({})


class TestEndToEnd:
    def test_recovers_the_composition_the_data_was_built_from(self):
        """The real bar. Fit the synthetic EDS SI, quantify, and compare with
        the concentration maps the generator used."""
        pytest.importorskip("exspy", reason="needs the eels extra")
        from spyde.data import eds_si, ground_truth
        from spyde.fitting.engine import fit_batched
        from spyde.spectroscopy import model_for_composition

        s = eds_si(nav=(6, 6), n_channels=2048, noise=False)
        spec, info = model_for_composition(s, ["Fe", "Ni", "Cu"],
                                           energy_range=(5.5, 9.5))
        assert info["engine_supported"]

        x = np.asarray(s.axes_manager.signal_axes[0].axis, float)
        res = fit_batched(spec, s.data, x, device="cpu", max_iter=120)
        frac, qinfo = quantify_result(spec, res, (6, 6))

        assert set(frac) == {"Fe", "Ni", "Cu"}
        truth = ground_truth(s)["concentration"]
        for el in ("Fe", "Ni", "Cu"):
            got = frac[el].ravel()
            want = np.asarray(truth[el]).ravel()
            r = np.corrcoef(got, want)[0, 1]
            assert r > 0.9, f"{el}: fitted composition does not track truth (r={r:.3f})"

    def test_quantify_result_carries_the_raw_maps(self):
        spec = _spec(("Fe_Ka", "Gaussian"), ("Cu_Ka", "Gaussian"))
        res = _result(spec, [[10.0, 0, 1, 30.0, 0, 1]])
        frac, info = quantify_result(spec, res)
        assert info["intensity_maps"]["Fe"][0] == pytest.approx(10.0)
        assert frac["Fe"][0] == pytest.approx(0.25)

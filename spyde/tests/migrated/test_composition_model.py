"""Composition -> auto-populated model (#62).

Skipped wholesale without the `eels` extra, which is the point of the extra:
the code must be importable and the tests must skip cleanly rather than error,
because CI runs one job with the extras and one without.
"""
from __future__ import annotations

import numpy as np
import pytest

exspy = pytest.importorskip("exspy", reason="needs the eels extra")

from spyde.data import eds_si, eels_si
from spyde.fitting import components as tcomp
from spyde.spectroscopy import MissingExtra, model_for_composition, prune_to_range


@pytest.fixture(scope="module")
def eds():
    return eds_si(nav=(2, 2), n_channels=1024)


@pytest.fixture(scope="module")
def eels():
    return eels_si(nav=(2, 2), n_channels=512)


class TestEDS:
    def test_builds_a_component_per_line(self, eds):
        spec, info = model_for_composition(eds, ["Fe", "Ni", "Cu"])
        assert info["kind"] == "EDS"
        assert set(info["elements"]) == {"Fe", "Ni", "Cu"}
        names = [c.name for c in spec]
        assert any(n.startswith("Fe_K") for n in names), names
        assert any(n.startswith("Cu_K") for n in names), names

    def test_lines_land_at_their_real_energies(self, eds):
        """The whole value of using exspy: the energies are right without us
        maintaining a table."""
        from spyde.data.synthetic import EDS_LINES

        spec, _ = model_for_composition(eds, ["Fe", "Ni", "Cu"])
        by_name = {c.name: c for c in spec}
        for el, lines in EDS_LINES.items():
            comp = by_name.get(f"{el}_Ka")
            assert comp is not None, f"{el}_Ka missing from {list(by_name)}"
            assert comp["centre"].value == pytest.approx(lines[0][1], abs=0.02)

    def test_the_batched_engine_can_fit_an_EDS_model(self, eds):
        """EDS builds from Polynomial + Gaussian, both of which the engine
        implements — so EDS gets the GPU path with no fallback. This is the
        asymmetry with EELS worth pinning."""
        spec, info = model_for_composition(eds, ["Fe", "Ni", "Cu"])
        assert info["engine_supported"] is True
        assert tcomp.supports(spec) is True

    def test_only_lines_restricts_the_model(self, eds):
        """`add_lines` only APPENDS, so restricting needs `set_lines` — the
        obvious call silently leaves every default line in place."""
        spec, _ = model_for_composition(eds, ["Fe", "Cu"],
                                        only_lines=["Fe_Ka", "Cu_Ka"])
        lines = [c.name for c in spec
                 if "_" in c.name and not c.name.startswith("background")]
        assert set(lines) <= {"Fe_Ka", "Cu_Ka"}, lines
        assert lines, "every line was dropped"


class TestEELS:
    def test_builds_an_edge_per_element(self, eels):
        spec, info = model_for_composition(eels, ["C", "N", "O"])
        assert info["kind"] == "EELS"
        kinds = [c.kind for c in spec]
        assert kinds.count("EELSCLEdge") == 3, kinds
        assert "PowerLaw" in kinds          # background

    def test_edges_land_at_their_real_onsets(self, eels):
        from spyde.data.synthetic import EELS_EDGES

        spec, _ = model_for_composition(eels, ["C", "N", "O"])
        by_name = {c.name: c for c in spec}
        for name, onset in EELS_EDGES.items():
            comp = by_name.get(name)
            assert comp is not None, f"{name} missing from {list(by_name)}"
            assert comp["onset_energy"].value == pytest.approx(onset, abs=5.0)

    def test_reports_that_the_engine_cannot_fit_it_yet(self, eels):
        """EELSCLEdge is a tabulated GOS lookup with no batched port (#63).
        The fallback to HyperSpy is correct, just slower — but it must be
        REPORTED, not a surprise."""
        spec, info = model_for_composition(eels, ["C", "N", "O"])
        assert info["engine_supported"] is False
        assert "EELSCLEdge" in info["unsupported_components"]


class TestPruning:
    def test_drops_components_outside_the_measured_range(self, eds):
        """A component with no data under it is not harmless — its amplitude is
        unconstrained, so the optimiser trades it against everything else and
        degrades the parameters that ARE measurable."""
        spec, info = model_for_composition(eds, ["Fe", "Ni", "Cu"],
                                           energy_range=(5.0, 9.0))
        for c in spec:
            if c.kind == "Gaussian":
                assert 5.0 <= c["centre"].value <= 9.0, c.name
        assert info["dropped"], "nothing was pruned from a narrowed range"

    def test_pruning_can_be_turned_off(self, eds):
        wide, _ = model_for_composition(eds, ["Fe", "Ni", "Cu"], prune=False)
        narrow, _ = model_for_composition(eds, ["Fe", "Ni", "Cu"],
                                          energy_range=(5.0, 9.0))
        assert len(wide) > len(narrow)

    def test_keeps_components_whose_position_is_unknown(self):
        """Dropping something we do not understand is the worse error — a
        background has no 'position' and must survive."""
        from spyde.fitting.spec import ComponentSpec, ParameterSpec

        spec = ModelSpecShim = None  # noqa: F841  (readability of the next line)
        from spyde.fitting import ModelSpec
        spec = ModelSpec(components=[
            ComponentSpec(kind="PowerLaw", parameters=[ParameterSpec("A", 1.0)]),
            ComponentSpec(kind="Gaussian", name="far",
                          parameters=[ParameterSpec("centre", 999.0)]),
        ])
        kept, dropped = prune_to_range(spec, 0.0, 20.0)
        assert [c.kind for c in kept] == ["PowerLaw"]
        assert dropped == ["far"]

    def test_margin_widens_the_window(self):
        from spyde.fitting import ModelSpec
        from spyde.fitting.spec import ComponentSpec, ParameterSpec

        spec = ModelSpec(components=[ComponentSpec(
            kind="Gaussian", name="edge",
            parameters=[ParameterSpec("centre", 20.5)])])
        assert prune_to_range(spec, 0.0, 20.0)[1] == ["edge"]
        assert prune_to_range(spec, 0.0, 20.0, margin=1.0)[1] == []


class TestGuards:
    def test_a_plain_signal_is_rejected_with_guidance(self):
        import hyperspy.api as hs
        s = hs.signals.Signal1D(np.zeros((2, 2, 32)))
        with pytest.raises(ValueError, match="neither EELS nor EDS"):
            model_for_composition(s, ["Fe"])

    def test_no_elements_is_a_clear_error(self, eds):
        bare = eds.deepcopy()
        try:
            del bare.metadata.Sample
        except Exception:
            pass
        with pytest.raises(ValueError, match="no elements"):
            model_for_composition(bare)

    def test_the_source_signal_is_not_mutated(self, eds):
        """Building a model must not silently add elements to the user's
        signal — they may be exploring several compositions."""
        before = list(getattr(eds.metadata, "Sample", {}).get_item("elements", [])
                      if hasattr(getattr(eds.metadata, "Sample", None), "get_item")
                      else [])
        model_for_composition(eds, ["Fe", "Ni", "Cu", "Zn"])
        after = list(getattr(eds.metadata, "Sample", {}).get_item("elements", [])
                     if hasattr(getattr(eds.metadata, "Sample", None), "get_item")
                     else [])
        assert before == after


class TestEndToEnd:
    def test_an_EDS_composition_model_actually_fits(self, eds):
        """The point of the whole wave: elements in, fitted maps out, through
        the batched engine."""
        from spyde.fitting.engine import fit_batched

        spec, info = model_for_composition(eds, ["Fe", "Ni", "Cu"],
                                           energy_range=(5.5, 9.5))
        assert info["engine_supported"]
        x = np.asarray(eds.axes_manager.signal_axes[0].axis, float)
        got = fit_batched(spec, eds.data, x, device="cpu", max_iter=40)
        assert got.values.shape[0] == 4
        assert np.isfinite(got.values).all()

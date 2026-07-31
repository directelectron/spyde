"""Adding ONE ``EELSCLEdge`` from the Fit caret's component picker.

The bug this pins: the picker offered nine analytic shapes and no way to reach
an EELS core-loss edge at all. ``fit_from_composition`` could build a whole
model of edges, but it REPLACES the model — so adding a single O-K edge to a
background you had already tuned was not possible.

An edge is unlike every other entry in the picker in three ways, and each one
is a test here:

* it takes a constructor argument (``EELSCLEdge("O_K")``), so it is one button
  per SUBSHELL, not one for the kind;
* it needs the microscope geometry, and the native failure without it is
  ``AttributeError('Acquisition_instrument')`` raised from inside exspy;
* it only resolves once appended to a model of a real EELS signal.

Skipped wholesale without the ``eels`` extra — CI runs one job with the extras
and one without, and the code must import cleanly in both.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
exspy = pytest.importorskip("exspy", reason="needs the eels extra")

import hyperspy.api as hs

from spyde.actions.fit_action import (
    CATALOGUE,
    eels_offer,
    fit_add_component,
    fit_open,
)
from spyde.data import eels_si
from spyde.fitting import components as tcomp
from spyde.spectroscopy import edges as eels_edges


@pytest.fixture(scope="module")
def eels():
    """The bundled synthetic EELS SI — 200-800 eV, C/N/O K edges, and the
    microscope parameters already stamped on it (`spyde.data.synthetic`)."""
    return eels_si(nav=(2, 2), n_channels=512)


def _bare_eels():
    """An EELS signal over the same range with NO microscope parameters."""
    s = hs.signals.Signal1D(np.random.default_rng(0).random((2, 2, 512)))
    ax = s.axes_manager.signal_axes[0]
    ax.offset, ax.scale, ax.units = 200.0, (800.0 - 200.0) / 512, "eV"
    s.set_signal_type("EELS")
    return s


class TestSignalGating:
    def test_an_eels_signal_is_recognised(self, eels):
        assert eels_edges.is_eels(eels) is True

    def test_a_plain_signal_is_not(self):
        assert eels_edges.is_eels(hs.signals.Signal1D(np.zeros((2, 8)))) is False

    def test_declared_metadata_counts_even_without_an_EELS_class(self):
        """Without exspy there IS no EELS class — `set_signal_type("EELS")`
        writes the metadata and leaves a plain Signal1D. Reading only the class
        would hide the edge section from exactly the user who needs to be told
        to install the extra."""
        s = hs.signals.Signal1D(np.zeros((2, 8)))
        s.metadata.set_item("Signal.signal_type", "EELS")
        assert eels_edges.is_eels(s) is True

    def test_the_offer_is_empty_off_EELS(self):
        """A non-EELS signal gets no edge section at all — the picker must not
        show a control that can only produce an error."""
        offer = eels_offer(hs.signals.Signal1D(np.zeros((2, 8))))
        assert offer["eels"] is False
        assert offer["edges"] == []

    def test_the_offer_lists_edges_on_EELS(self, eels):
        offer = eels_offer(eels)
        assert offer["eels"] is True and offer["exspy"] is True
        assert offer["microscope_missing"] == []
        assert {e["subshell"] for e in offer["edges"]} >= {"C_K", "N_K", "O_K"}


class TestMicroscopeParameters:
    def test_the_synthetic_signal_has_them_all(self, eels):
        assert eels_edges.missing_microscope_parameters(eels) == []

    def test_missing_ones_are_named(self):
        """Named, not counted: "set the collection angle" is actionable and
        `AttributeError('Acquisition_instrument')` is not."""
        missing = eels_edges.missing_microscope_parameters(_bare_eels())
        assert missing == ["beam energy", "convergence angle",
                           "collection angle"]

    def test_a_non_numeric_value_counts_as_missing(self):
        s = _bare_eels()
        s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", "n/a")
        assert "beam energy" in eels_edges.missing_microscope_parameters(s)

    def test_building_an_edge_without_them_raises_ours_not_exspys(self):
        with pytest.raises(eels_edges.MissingMicroscopeParameters) as e:
            eels_edges.edge_component_spec(_bare_eels(), "O_K")
        assert "collection angle" in str(e.value)
        assert e.value.missing  # the fields, for a caller that wants them

    def test_the_offer_reports_them_instead_of_the_edges(self):
        offer = eels_offer(_bare_eels())
        assert offer["eels"] is True
        assert "collection angle" in offer["microscope_missing"]


class TestAvailableEdges:
    def test_every_offered_onset_is_inside_the_measured_range(self, eels):
        """An edge whose onset is off-screen has an unconstrained intensity,
        which makes every other parameter worse rather than merely wasting
        one — the same rule `prune_to_range` enforces."""
        lo, hi = 200.0, 800.0
        for e in eels_edges.available_edges(eels):
            assert lo <= e["onset"] <= hi, e

    def test_onsets_match_the_tabulated_values(self, eels):
        from spyde.data.synthetic import EELS_EDGES

        by_name = {e["subshell"]: e for e in eels_edges.available_edges(eels)}
        for subshell, onset in EELS_EDGES.items():
            assert by_name[subshell]["onset"] == pytest.approx(onset, abs=1.0)

    def test_the_composition_seeds_the_suggestions(self, eels):
        """`metadata.Sample.elements` is what Plot Control's Composition panel
        writes, so setting the composition seeds the picker with no further
        wiring."""
        s = eels.deepcopy()
        s.metadata.set_item("Sample.elements", ["O"])
        suggested = {e["subshell"] for e in eels_edges.available_edges(s)
                     if e["suggested"]}
        assert suggested == {"O_K"}

    def test_without_a_composition_nothing_is_suggested_but_edges_remain(self, eels):
        got = eels_edges.available_edges(eels, elements=[])
        assert got, "the catalogue must still list the window's edges"
        assert not any(e["suggested"] for e in got)

    def test_suggested_edges_sort_first(self, eels):
        s = eels.deepcopy()
        s.metadata.set_item("Sample.elements", ["O"])
        got = eels_edges.available_edges(s)
        assert got[0]["subshell"] == "O_K"

    def test_the_kind_is_absent_from_the_analytic_catalogue(self):
        """EELSCLEdge must NOT be a plain CATALOGUE entry: every one of those
        is a bare `Kind()` the picker draws as one button, and there is no
        bare edge."""
        assert "EELSCLEdge" not in {kind for kind, _ in CATALOGUE}


class TestEdgeComponentSpec:
    def test_it_is_a_real_exspy_edge(self, eels):
        cspec = eels_edges.edge_component_spec(eels, "O_K")
        assert cspec.kind == "EELSCLEdge"
        assert cspec.init_args["element_subshell"] == "O_K"
        names = {p.name for p in cspec.parameters}
        assert {"intensity", "onset_energy"} <= names

    def test_it_lands_at_the_tabulated_onset(self, eels):
        cspec = eels_edges.edge_component_spec(eels, "O_K")
        assert cspec["onset_energy"].value == pytest.approx(532.0, abs=1.0)

    def test_a_non_eels_signal_is_refused_with_a_readable_reason(self):
        with pytest.raises(ValueError, match="EELS"):
            eels_edges.edge_component_spec(
                hs.signals.Signal1D(np.zeros((2, 8))), "O_K")


class TestAddingOneFromTheCaret:
    @pytest.fixture
    def opened(self, window, eels):
        session = window["window"]
        session._add_signal(eels.deepcopy())
        tree = window["signal_trees"][0]
        plot = next(iter(tree.signal_plots))
        fit_open(session, plot)
        return session, plot, tree

    def _states(self, window):
        return [m for m in window["messages"] if m.get("type") == "fit_state"]

    def test_adding_an_edge_puts_it_in_the_model(self, window, opened):
        session, plot, tree = opened
        fit_add_component(session, plot,
                          {"kind": "EELSCLEdge", "element_subshell": "O_K"})
        kinds = [c.kind for c in tree.fit_spec.components]
        assert kinds == ["EELSCLEdge"]
        assert tree.fit_spec.components[0].name == "O_K"

    def test_the_added_edge_is_fittable_by_the_batched_engine(self, window, opened):
        """The GOS integral depends on the element and the microscope, not the
        pixel, so it is precomputed on the way in (#63). Without that the whole
        model drops onto HyperSpy's one-pixel-at-a-time path."""
        session, plot, tree = opened
        fit_add_component(session, plot,
                          {"kind": "EELSCLEdge", "element_subshell": "O_K"})
        assert tcomp.unsupported(tree.fit_spec) == {}
        assert tcomp.supports(tree.fit_spec) is True

    def test_the_edge_is_scaled_onto_the_data(self, window, opened):
        """A component that arrives at intensity 1 against counts of 1e5 draws
        as a flat line on the axis and reads as "the model does nothing"."""
        session, plot, tree = opened
        fit_add_component(session, plot,
                          {"kind": "EELSCLEdge", "element_subshell": "O_K"})
        assert tree.fit_spec.components[0]["intensity"].value > 1.0

    def test_an_edge_joins_a_background_rather_than_replacing_it(self, window, opened):
        """The whole point against `fit_from_composition`, which replaces the
        model wholesale."""
        session, plot, tree = opened
        fit_add_component(session, plot, {"kind": "PowerLaw"})
        fit_add_component(session, plot,
                          {"kind": "EELSCLEdge", "element_subshell": "O_K"})
        assert [c.kind for c in tree.fit_spec.components] \
            == ["PowerLaw", "EELSCLEdge"]

    def test_two_edges_stay_separately_addressable(self, window, opened):
        session, plot, tree = opened
        for sub in ("O_K", "N_K"):
            fit_add_component(session, plot,
                              {"kind": "EELSCLEdge", "element_subshell": sub})
        assert [c.name for c in tree.fit_spec.components] == ["O_K", "N_K"]

    def test_the_state_the_caret_renders_carries_the_edge(self, window, opened):
        session, plot, tree = opened
        fit_add_component(session, plot,
                          {"kind": "EELSCLEdge", "element_subshell": "O_K"})
        comps = self._states(window)[-1]["components"]
        assert [c["name"] for c in comps] == ["O_K"]
        assert comps[0]["kind"] == "EELSCLEdge"

    def test_no_subshell_is_an_error_not_a_silent_no_op(self, window, opened):
        session, plot, tree = opened
        fit_add_component(session, plot, {"kind": "EELSCLEdge"})
        errors = [m for m in window["messages"] if m.get("type") == "error"]
        assert errors and "edge" in str(errors[-1]).lower()
        assert list(tree.fit_spec.components) == []

    def test_an_unknown_subshell_is_an_error_not_a_crash(self, window, opened):
        session, plot, tree = opened
        fit_add_component(session, plot,
                          {"kind": "EELSCLEdge", "element_subshell": "Zz_Q9"})
        errors = [m for m in window["messages"] if m.get("type") == "error"]
        assert errors, "a bad subshell must be reported"
        assert list(tree.fit_spec.components) == []

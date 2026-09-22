"""
test_composition.py — the sample's phases and the COD structure search.

A sample is a list of phases, each its elements (with optional percentages and
trace marks) and the ``.cif`` that indexes it. ``metadata.Sample.elements`` is
their union, for EELS and EDS. The COD search/normalise/fetch logic is
unit-tested with mocked network; one network-guarded test hits the live COD API
(skipped offline).
"""
import os

import numpy as np
import pytest
import hyperspy.api as hs

from spyde.actions import composition as comp

CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")


class _Tree:
    def __init__(self, signal):
        self.root = signal
        self.signal_plots = []


class _Session:
    """Only what the handlers ask a session for: the windows to update. With no
    ``_dispatch_to_main``, work meant for a worker thread runs inline."""

    def _tree_window_ids(self, tree):
        return [3]


_SESSION = _Session()


def _tree(*phases, signal=None):
    tree = _Tree(signal if signal is not None
                 else hs.signals.Signal2D(np.zeros((4, 4), dtype=np.float32)))
    if phases:
        comp.write_phases(tree, list(phases))
    return tree


def _elements(tree):
    return [phase["elements"] for phase in comp.read_phases(tree)]


def _phase(tree, phase_id):
    return next(phase for phase in comp.read_phases(tree) if phase["id"] == phase_id)


@pytest.fixture
def emitted(monkeypatch):
    """Every message the composition module sends."""
    messages = []
    monkeypatch.setattr(comp, "emit", messages.append)
    monkeypatch.setattr(comp, "emit_status", lambda text: messages.append(
        {"type": "status", "text": text}))
    monkeypatch.setattr(comp, "emit_error", lambda text: messages.append(
        {"type": "error", "text": text}))
    return messages


def _act(monkeypatch, tree, handler, payload):
    monkeypatch.setattr(comp, "_src_plot_tree", lambda session, plot: (None, tree))
    handler(_SESSION, None, payload)


# ── the stored model ────────────────────────────────────────────────────────────
class TestPhaseModel:
    def test_sample_elements_are_the_union_of_the_phases(self):
        # EELS and EDS fit every element of every phase, and they read it here.
        tree = _tree({"elements": ["Cu"], "percentages": {"Cu": 60.0}},
                     {"elements": ["Nb"]})
        assert list(tree.root.metadata.Sample.elements) == ["Cu", "Nb"]

    def test_the_union_lists_a_shared_element_once(self):
        tree = _tree({"elements": ["Zr", "O"]}, {"elements": ["Zr"]})
        assert list(tree.root.metadata.Sample.elements) == ["Zr", "O"]
        assert _elements(tree) == [["Zr", "O"], ["Zr"]]

    def test_elements_without_a_phase_list_read_as_one_phase(self):
        # Set by a file reader, or by SpyDE before phases existed.
        tree = _tree()
        tree.root.metadata.set_item("Sample.elements", ["Fe", "Ni"])
        tree.root.metadata.set_item("Sample.composition", {"Fe": 70.0})
        phases = comp.read_phases(tree)
        assert [phase["elements"] for phase in phases] == [["Fe", "Ni"]]
        assert phases[0]["percentages"] == {"Fe": 70.0}
        assert phases[0]["trace"] == []
        assert phases[0]["cif_path"] is None

    def test_old_percentages_that_are_not_a_table_are_ignored(self):
        tree = _tree()
        tree.root.metadata.set_item("Sample.elements", ["Fe"])
        tree.root.metadata.set_item("Sample.composition", "Fe70Ni30")
        assert comp.read_phases(tree)[0]["percentages"] == {}

    def test_a_single_symbol_is_one_element(self):
        tree = _tree()
        tree.root.metadata.set_item("Sample.elements", "Fe")
        assert _elements(tree) == [["Fe"]]

    def test_an_element_added_elsewhere_joins_phase_one_as_trace(self):
        # The console or exspy can add to Sample.elements directly. The element
        # still counts for EELS and EDS, but is not Phase 1's structure.
        tree = _tree({"elements": ["Fe"]}, {"elements": ["Cr"]})
        tree.root.metadata.set_item("Sample.elements", ["Fe", "Cr", "O"])
        first = comp.read_phases(tree)[0]
        assert first["elements"] == ["Fe", "O"]
        assert first["trace"] == ["O"]
        # …and the next edit keeps it rather than dropping it.
        comp.write_phases(tree, comp.read_phases(tree))
        assert list(tree.root.metadata.Sample.elements) == ["Fe", "O", "Cr"]

    def test_the_first_read_gives_stable_ids(self):
        tree = _tree()
        tree.root.metadata.set_item("Sample.elements", ["Fe"])
        assert comp.read_phases(tree)[0]["id"] == comp.read_phases(tree)[0]["id"]

    def test_writing_phases_drops_the_old_flat_percentages(self):
        # They now live on the phases; keeping both is two answers that drift.
        tree = _tree()
        tree.root.metadata.set_item("Sample.elements", ["Fe"])
        tree.root.metadata.set_item("Sample.composition", {"Fe": 70.0})
        comp.write_phases(tree, comp.read_phases(tree))
        assert not tree.root.metadata.has_item("Sample.composition")
        assert comp.read_phases(tree)[0]["percentages"] == {"Fe": 70.0}

    def test_removing_every_phase_empties_the_sample(self):
        tree = _tree({"elements": ["Cu"]})
        comp.write_phases(tree, [])
        assert list(tree.root.metadata.Sample.elements) == []
        assert comp.read_phases(tree) == []

    def test_a_removed_element_loses_its_xray_lines(self):
        # exspy adds back the element of every X-ray line it finds, so a line
        # left behind would keep the element in the EDS fit.
        tree = _tree()
        tree.root.metadata.set_item("Sample.elements", ["Cr", "Fe"])
        tree.root.metadata.set_item("Sample.xray_lines", ["Cr_Ka", "Fe_Ka"])
        comp.write_phases(tree, [{"elements": ["Fe"]}])
        assert list(tree.root.metadata.Sample.xray_lines) == ["Fe_Ka"]

    def test_a_phase_remembers_its_structure(self):
        tree = _tree({"id": "cu", "elements": ["Cu"], "cif_path": "/x/Cu.cif",
                      "label": "Cu Fm-3m", "cod_id": "9008468",
                      "structure_elements": ["Cu"]})
        phase = _phase(tree, "cu")
        assert (phase["cif_path"], phase["label"], phase["cod_id"]) == \
            ("/x/Cu.cif", "Cu Fm-3m", "9008468")
        assert phase["structure_elements"] == ["Cu"]
        assert comp.phase_label(phase) == "Cu Fm-3m"

    def test_a_structureless_phase_reads_as_its_elements(self):
        assert comp.phase_label({"elements": ["Zr", "O"]}) == "Zr-O"

    def test_a_stored_phase_keeps_a_fixed_shape(self):
        # It round-trips through file metadata.
        phase = comp.read_phases(_tree({"elements": ["Cu"], "scribble": 1}))[0]
        assert set(phase) == {"id", "elements", "percentages", "trace", "cif_path",
                              "label", "cod_id", "structure_elements"}

    def test_marks_are_kept_only_for_the_phases_own_elements(self):
        phase = comp.read_phases(_tree({
            "elements": ["Cu"], "percentages": {"Cu": 60, "Nb": 40, "Zn": "x"},
            "trace": ["Nb"]}))[0]
        assert phase["percentages"] == {"Cu": 60.0}
        assert phase["trace"] == []


# ── editing ─────────────────────────────────────────────────────────────────────
class TestPhaseEdits:
    def test_add_and_remove(self, monkeypatch, emitted):
        tree = _tree({"id": "cu", "elements": ["Cu"]})
        _act(monkeypatch, tree, comp.add_phase, {"phase": "empty"})
        _act(monkeypatch, tree, comp.add_phase, {"phase": "nb", "elements": ["Nb"]})
        assert _elements(tree) == [["Cu"], [], ["Nb"]]
        _act(monkeypatch, tree, comp.remove_phase, {"phase": "cu"})
        assert _elements(tree) == [[], ["Nb"]]

    def test_adding_an_existing_phase_does_nothing(self, monkeypatch, emitted):
        tree = _tree({"id": "cu", "elements": ["Cu"]})
        _act(monkeypatch, tree, comp.add_phase, {"phase": "cu"})
        assert _elements(tree) == [["Cu"]]
        assert emitted == []

    def test_toggling_adds_then_removes_an_element(self, monkeypatch, emitted):
        tree = _tree({"id": "p", "elements": ["Cu"], "percentages": {"Cu": 60.0},
                      "trace": ["Cu"]})
        _act(monkeypatch, tree, comp.toggle_phase_element, {"phase": "p", "element": "Zn"})
        assert _elements(tree) == [["Cu", "Zn"]]
        _act(monkeypatch, tree, comp.toggle_phase_element, {"phase": "p", "element": "Cu"})
        assert _elements(tree) == [["Zn"]]
        # Its marks go with it, or they would come back with the element.
        assert _phase(tree, "p")["percentages"] == {}
        assert _phase(tree, "p")["trace"] == []

    def test_an_element_can_be_in_two_phases(self, monkeypatch, emitted):
        tree = _tree({"id": "zirconia", "elements": ["Zr", "O"]}, {"id": "alpha"})
        _act(monkeypatch, tree, comp.toggle_phase_element,
             {"phase": "alpha", "element": "Zr"})
        assert _elements(tree) == [["Zr", "O"], ["Zr"]]

    def test_a_new_phase_id_is_created_once(self, monkeypatch, emitted):
        # Two clicks on the not-yet-created phase, the second sent before the
        # first reply, land in the same new phase.
        tree = _tree()
        _act(monkeypatch, tree, comp.toggle_phase_element, {"phase": "new", "element": "Fe"})
        _act(monkeypatch, tree, comp.toggle_phase_element, {"phase": "new", "element": "O"})
        assert _elements(tree) == [["Fe", "O"]]
        assert list(tree.root.metadata.Sample.elements) == ["Fe", "O"]

    def test_an_unknown_phase_is_ignored(self, monkeypatch, emitted):
        tree = _tree({"id": "cu", "elements": ["Cu"]})
        _act(monkeypatch, tree, comp.remove_phase, {"phase": "gone"})
        _act(monkeypatch, tree, comp.set_phase_percentages,
             {"phase": "gone", "percentages": {"Cu": 1}})
        _act(monkeypatch, tree, comp.set_phase_trace,
             {"phase": "gone", "element": "Cu", "trace": True})
        _act(monkeypatch, tree, comp.toggle_phase_element, {"element": "Ag"})
        assert _elements(tree) == [["Cu"]]
        assert emitted == []

    def test_percentages_merge(self, monkeypatch, emitted):
        # Two saves made before the first reply keep both values.
        tree = _tree({"id": "p", "elements": ["Fe", "Ni"]})
        _act(monkeypatch, tree, comp.set_phase_percentages,
             {"phase": "p", "percentages": {"Fe": 60}})
        _act(monkeypatch, tree, comp.set_phase_percentages,
             {"phase": "p", "percentages": {"Ni": 40}})
        assert _phase(tree, "p")["percentages"] == {"Fe": 60.0, "Ni": 40.0}
        _act(monkeypatch, tree, comp.set_phase_percentages,
             {"phase": "p", "percentages": {"Fe": None}})
        assert _phase(tree, "p")["percentages"] == {"Ni": 40.0}

    def test_marking_trace(self, monkeypatch, emitted):
        tree = _tree({"id": "p", "elements": ["Fe", "O"]})
        _act(monkeypatch, tree, comp.set_phase_trace,
             {"phase": "p", "element": "O", "trace": True})
        assert _phase(tree, "p")["trace"] == ["O"]
        assert {"type": "status", "text": "Phase 1: O is trace"} in emitted
        _act(monkeypatch, tree, comp.set_phase_trace,
             {"phase": "p", "element": "O", "trace": False})
        assert _phase(tree, "p")["trace"] == []

    def test_changing_elements_leaves_the_structure_alone(self, monkeypatch, emitted):
        # Silently dropping the .cif would hide that the two now disagree.
        tree = _tree({"id": "p", "elements": ["Cu"], "cif_path": "/x/Cu.cif", "label": "Cu"})
        _act(monkeypatch, tree, comp.toggle_phase_element, {"phase": "p", "element": "Zn"})
        assert _phase(tree, "p")["cif_path"] == "/x/Cu.cif"

    def test_an_edit_shows_the_phases_and_their_union(self, monkeypatch, emitted):
        tree = _tree({"id": "cu", "elements": ["Cu"]}, {"id": "nb", "elements": ["Nb"]})
        _act(monkeypatch, tree, comp.toggle_phase_element, {"phase": "nb", "element": "O"})
        sent = [message for message in emitted if message["type"] == "composition"][-1]
        assert sent["window_ids"] == [3]
        assert [phase["elements"] for phase in sent["phases"]] == [["Cu"], ["Nb", "O"]]
        assert sent["elements"] == ["Cu", "Nb", "O"]
        assert {"type": "status", "text": "Phase 2: Nb-O"} in emitted


class TestPhaseStructure:
    def test_a_structure_names_itself_from_the_file(self, monkeypatch, emitted):
        tree = _tree({"id": "p", "elements": ["Nb"]})
        _act(monkeypatch, tree, comp.set_phase_structure,
             {"phase": "p", "cif_path": "/x/beta_Nb_cod4000948.cif"})
        assert _phase(tree, "p")["label"] == "beta_Nb_cod4000948"

    def test_a_structure_can_be_cleared(self, monkeypatch, emitted):
        tree = _tree({"id": "p", "elements": ["Nb"], "cif_path": "/x/Nb.cif",
                      "label": "Nb", "cod_id": "1", "structure_elements": ["Nb"]})
        _act(monkeypatch, tree, comp.set_phase_structure, {"phase": "p", "cif_path": None})
        phase = _phase(tree, "p")
        assert (phase["cif_path"], phase["label"], phase["cod_id"],
                phase["structure_elements"]) == (None, None, None, None)
        assert phase["elements"] == ["Nb"]

    def test_clearing_an_unknown_phase_creates_nothing(self, monkeypatch, emitted):
        tree = _tree()
        _act(monkeypatch, tree, comp.set_phase_structure, {"phase": "new", "cif_path": None})
        assert comp.read_phases(tree) == []

    def test_the_elements_come_from_the_file_when_the_phase_has_none(self, monkeypatch, emitted):
        tree = _tree({"id": "p"})
        _act(monkeypatch, tree, comp.set_phase_structure, {"phase": "p", "cif_path": CIF})
        assert _elements(tree) == [["Ag"]]
        assert _phase(tree, "p")["structure_elements"] == ["Ag"]

    def test_loading_a_structure_for_a_new_phase_creates_it(self, monkeypatch, emitted):
        tree = _tree()
        _act(monkeypatch, tree, comp.set_phase_structure, {"phase": "new", "cif_path": CIF})
        assert _elements(tree) == [["Ag"]]

    def test_the_elements_follow_a_swapped_structure(self, monkeypatch, emitted):
        # Elements that came from the old file are replaced by the new file's;
        # a trace element stays.
        monkeypatch.setattr(comp, "elements_from_cif", lambda path: {
            "/x/Ag.cif": ["Ag"], "/x/Cu.cif": ["Cu"]}[path])
        tree = _tree({"id": "p"})
        _act(monkeypatch, tree, comp.set_phase_structure, {"phase": "p", "cif_path": "/x/Ag.cif"})
        _act(monkeypatch, tree, comp.toggle_phase_element, {"phase": "p", "element": "O"})
        _act(monkeypatch, tree, comp.set_phase_trace,
             {"phase": "p", "element": "O", "trace": True})
        _act(monkeypatch, tree, comp.set_phase_structure, {"phase": "p", "cif_path": "/x/Cu.cif"})
        phase = _phase(tree, "p")
        assert phase["elements"] == ["Cu", "O"]
        assert phase["trace"] == ["O"]

    def test_chosen_elements_are_never_overwritten_by_the_file(self, monkeypatch, emitted):
        # The person may have said something the file cannot — a solid
        # solution, a measured percentage.
        tree = _tree({"id": "p", "elements": ["Ag", "Cu"], "percentages": {"Ag": 90.0}})
        _act(monkeypatch, tree, comp.set_phase_structure, {"phase": "p", "cif_path": CIF})
        phase = _phase(tree, "p")
        assert phase["elements"] == ["Ag", "Cu"]
        assert phase["percentages"] == {"Ag": 90.0}

    def test_an_unreadable_file_changes_no_elements(self, monkeypatch, emitted):
        monkeypatch.setattr(comp, "elements_from_cif", lambda path: [])
        tree = _tree({"id": "p", "elements": ["Ag"], "cif_path": "/x/Ag.cif",
                      "structure_elements": ["Ag"]})
        _act(monkeypatch, tree, comp.set_phase_structure, {"phase": "p", "cif_path": "/x/bad.cif"})
        assert _elements(tree) == [["Ag"]]


class TestElementsFromCif:
    def test_reads_the_elements(self):
        assert comp.elements_from_cif(CIF) == ["Ag"]

    def test_an_unreadable_file_is_not_an_error(self):
        assert comp.elements_from_cif("/no/such/file.cif") == []


class TestEditsReachEveryWindowOfTheSample:
    """The dock shows the FOCUSED window's phases, and on a scan that is
    usually the navigator, so an edit must reach every window of the tree."""

    def _send_from_navigator(self, dataset, monkeypatch, action, payload):
        """Dispatch *action* as the focused navigator would; return the last
        composition message it produced and the two plots."""
        messages = []
        # This module bound `emit` at import, before the fixture could patch it.
        monkeypatch.setattr(comp, "emit", messages.append)
        navigator = next(plot for plot in dataset["plots"] if plot.is_navigator)
        signal = next(plot for plot in dataset["plots"] if not plot.is_navigator)
        dataset["window"].dispatch_action({"action": action, "payload": payload,
                                           "window_id": navigator.window_id})
        sent = [message for message in messages if message.get("type") == "composition"]
        assert sent, f"{action} did not push the composition"
        return sent[-1], navigator, signal

    def test_a_phase_added_from_the_navigator_reaches_the_navigator(
            self, stem_4d_dataset, monkeypatch):
        sent, navigator, signal = self._send_from_navigator(
            stem_4d_dataset, monkeypatch, "add_phase", {"phase": "p"})
        assert navigator.window_id in sent["window_ids"]
        assert signal.window_id in sent["window_ids"]
        assert len(sent["phases"]) == 1

    def test_an_element_click_reaches_the_navigator(self, stem_4d_dataset, monkeypatch):
        sent, navigator, _signal = self._send_from_navigator(
            stem_4d_dataset, monkeypatch, "toggle_phase_element",
            {"phase": "p", "element": "Cu"})
        assert navigator.window_id in sent["window_ids"]
        assert sent["elements"] == ["Cu"]


# ── COD ─────────────────────────────────────────────────────────────────────────
_SILVER_ROW = {"file": "9", "a": "4.08", "b": "4.08", "c": "4.08", "alpha": "90",
               "beta": "90", "gamma": "90", "sg": "Fm-3m", "sgNumber": "225",
               "formula": "- Ag -", "mineral": "Silver", "vol": "68"}


class TestCodTidy:
    def test_tidy_dedupes_and_sorts_by_cell(self):
        raw = [
            {"file": "1", "a": "4.08", "b": "4.08", "c": "4.08", "alpha": "90",
             "beta": "90", "gamma": "90", "sg": "F m -3 m", "sgNumber": "225",
             "formula": "- Ag -", "mineral": "Silver", "vol": "68.2"},
            {"file": "2", "a": "4.08", "b": "4.08", "c": "4.08", "alpha": "90",
             "beta": "90", "gamma": "90", "sg": "F m -3 m", "sgNumber": "225",
             "formula": "- Ag -", "mineral": "Silver", "vol": "68.2"},   # dup
            {"file": "3", "a": "2.88", "b": "2.88", "c": "2.88", "alpha": "90",
             "beta": "90", "gamma": "90", "sg": "I m -3 m", "formula": "- Fe -",
             "mineral": "Iron", "vol": "23.9"},
        ]
        out = comp._tidy_results(raw)
        assert len(out) == 2                       # duplicate dropped
        assert out[0]["volume"] < out[1]["volume"]  # smaller cell first
        assert out[0]["formula"] == "Fe" and out[0]["a"] == 2.88 and out[0]["sg"]
        assert out[0]["phase"] == "Iron"

    def test_tidy_skips_rows_without_a_cell(self):
        assert comp._tidy_results([{"file": "x", "formula": "Ag"}]) == []


class TestCodSearch:
    """COD is asked for a structure containing EXACTLY these elements, so the
    query is one phase's non-trace elements, never the whole sample's."""

    def _search(self, monkeypatch, tree, payload, query=lambda elements: []):
        asked = []

        def recording_query(elements):
            asked.append(list(elements))
            return query(elements)
        monkeypatch.setattr(comp, "_cod_query", recording_query)
        _act(monkeypatch, tree, comp.cod_search, payload)
        return asked

    def test_the_query_is_the_named_phases_elements(self, monkeypatch, emitted):
        tree = _tree({"id": "cu", "elements": ["Cu"]}, {"id": "nb", "elements": ["Nb"]})
        assert self._search(monkeypatch, tree, {"phase": "cu"}) == [["Cu"]]
        assert self._search(monkeypatch, tree, {"phase": "nb"}) == [["Nb"]]

    def test_trace_elements_are_left_out(self, monkeypatch, emitted):
        tree = _tree({"id": "fe", "elements": ["Fe", "O"], "trace": ["O"]})
        assert self._search(monkeypatch, tree, {"phase": "fe"}) == [["Fe"]]

    def test_results_come_back_tagged_with_their_phase(self, monkeypatch, emitted):
        tree = _tree({"id": "cu", "elements": ["Cu"]}, {"id": "ag", "elements": ["Ag"]})
        self._search(monkeypatch, tree, {"phase": "ag"}, query=lambda elements: [_SILVER_ROW])
        results = [message for message in emitted if message["type"] == "cod_results"][-1]
        assert results["phase"] == "ag"
        assert results["results"][0]["id"] == "9"
        assert results["error"] is None

    def test_a_network_failure_is_reported_not_raised(self, monkeypatch, emitted):
        def offline(elements):
            raise OSError("no network")
        tree = _tree({"id": "ag", "elements": ["Ag"]})
        self._search(monkeypatch, tree, {"phase": "ag"}, query=offline)
        results = [message for message in emitted if message["type"] == "cod_results"][-1]
        assert results["results"] == [] and results["error"]

    def test_a_phase_with_only_trace_elements_is_not_searched(self, monkeypatch, emitted):
        tree = _tree({"id": "p", "elements": ["O"], "trace": ["O"]})
        assert self._search(monkeypatch, tree, {"phase": "p"}) == []
        assert any(message["type"] == "error" for message in emitted)


class TestCodPick:
    def test_the_download_becomes_the_phases_structure(self, monkeypatch, emitted):
        monkeypatch.setattr(comp, "fetch_cod_cif", lambda cod_id: CIF)
        tree = _tree({"id": "cu", "elements": ["Cu"]}, {"id": "new"})
        _act(monkeypatch, tree, comp.cod_pick,
             {"phase": "new", "cod_id": "9", "label": "Ag Fm-3m"})
        phase = _phase(tree, "new")
        assert (phase["cif_path"], phase["label"], phase["cod_id"]) == (CIF, "Ag Fm-3m", "9")
        assert phase["elements"] == ["Ag"]

    def test_the_structure_finds_its_phase_after_the_list_changed(self, monkeypatch, emitted):
        # The download can take seconds; removing an earlier phase meanwhile
        # must not send the structure to a different one.
        tree = _tree({"id": "cu", "elements": ["Cu"]}, {"id": "ag", "elements": ["Ag"]})

        def download_while_cu_is_removed(cod_id):
            comp.write_phases(tree, [phase for phase in comp.read_phases(tree)
                                     if phase["id"] != "cu"])
            return CIF
        monkeypatch.setattr(comp, "fetch_cod_cif", download_while_cu_is_removed)
        _act(monkeypatch, tree, comp.cod_pick, {"phase": "ag", "cod_id": "9"})
        assert [(phase["id"], phase["cif_path"]) for phase in comp.read_phases(tree)] \
            == [("ag", CIF)]

    def test_a_removed_phase_gets_no_structure(self, monkeypatch, emitted):
        tree = _tree({"id": "cu", "elements": ["Cu"]})

        def download_while_the_phase_is_removed(cod_id):
            comp.write_phases(tree, [])
            return CIF
        monkeypatch.setattr(comp, "fetch_cod_cif", download_while_the_phase_is_removed)
        _act(monkeypatch, tree, comp.cod_pick, {"phase": "cu", "cod_id": "9"})
        assert comp.read_phases(tree) == []
        assert any(message["type"] == "error" for message in emitted)

    def test_a_pick_without_a_phase_is_reported(self, monkeypatch, emitted):
        monkeypatch.setattr(comp, "fetch_cod_cif",
                            lambda cod_id: pytest.fail("downloaded with no phase to bind"))
        _act(monkeypatch, _tree(), comp.cod_pick, {"cod_id": "9"})
        assert any(message["type"] == "error" for message in emitted)

    def test_a_failed_download_leaves_the_phase_alone(self, monkeypatch, emitted):
        def offline(cod_id):
            raise OSError("no network")
        monkeypatch.setattr(comp, "fetch_cod_cif", offline)
        tree = _tree({"id": "ag", "elements": ["Ag"]})
        _act(monkeypatch, tree, comp.cod_pick, {"phase": "ag", "cod_id": "9"})
        assert _phase(tree, "ag")["cif_path"] is None
        assert any(message["type"] == "error" for message in emitted)


class TestCodFetch:
    def test_fetch_cod_cif_writes_file(self, monkeypatch):
        cif = "data_test\n_cell_length_a 4.08\nloop_\n_atom_site_label\nAg1\n"

        class _Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return cif.encode("utf-8")
        monkeypatch.setattr(comp.urllib.request, "urlopen", lambda *args, **kwargs: _Response())
        path = comp.fetch_cod_cif("1100136")
        assert path.endswith("cod_1100136.cif")
        with open(path) as handle:
            assert "_cell_length_a" in handle.read()

    @pytest.mark.network
    def test_cod_live_search_silver(self):
        """Live COD smoke test — pure-Ag search returns FCC silver (a≈4.09).
        Skipped automatically if the network/COD is unavailable."""
        try:
            raw = comp._cod_query(["Ag"])
        except Exception:
            pytest.skip("COD/network unavailable")
        results = comp._tidy_results(raw)
        if not results:
            pytest.skip("COD returned nothing (rate-limited?)")
        assert any(abs((result["a"] or 0) - 4.09) < 0.2 for result in results)

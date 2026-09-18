"""
test_composition.py — sample composition metadata + the COD "easy CIF" picker.

Composition is stored at the HyperSpy-canonical ``metadata.Sample.elements`` /
``Sample.composition``. The COD search/normalise/fetch logic is unit-tested with
mocked network; one network-guarded test hits the live COD API (skipped offline).
"""
import os
import io
import numpy as np
import pytest
import hyperspy.api as hs

from spyde.actions import composition as comp


class _Tree:
    def __init__(self, sig):
        self.root = sig
        self.signal_plots = []


def _sig():
    return hs.signals.Signal2D(np.zeros((4, 4), dtype=np.float32))


# (cod_search/cod_pick ride lifecycle.run_on_worker, which runs INLINE when the
# session has no _dispatch_to_main — passing session=None below makes the
# handler's emit observable synchronously, no thread stub needed.)

# ── metadata round-trip ─────────────────────────────────────────────────────────
class TestCompositionMetadata:
    def test_write_read_roundtrip(self):
        t = _Tree(_sig())
        comp.write_composition(t, ["Fe", "Ni"], {"Fe": 70.0, "Ni": 30.0})
        els, pct = comp.read_composition(t)
        assert els == ["Fe", "Ni"]
        assert pct == {"Fe": 70.0, "Ni": 30.0}
        assert list(t.root.metadata.Sample.elements) == ["Fe", "Ni"]   # canonical

    def test_write_without_percentages(self):
        t = _Tree(_sig())
        comp.write_composition(t, ["Ag"])
        els, pct = comp.read_composition(t)
        assert els == ["Ag"] and pct == {}

    def test_read_empty(self):
        els, pct = comp.read_composition(_Tree(_sig()))
        assert els == [] and pct == {}

    def test_set_composition_handler_writes_and_emits(self, monkeypatch):
        captured = []
        monkeypatch.setattr(comp, "emit", lambda m: captured.append(m))
        monkeypatch.setattr(comp, "emit_status", lambda *a, **k: None)
        t = _Tree(_sig())

        class _Plot:
            window_id = 3
            signal_tree = t
        comp.set_composition(None, _Plot(), {"elements": ["Si", "O"],
                                             "percentages": {"Si": 33.3, "O": 66.7}})
        assert list(t.root.metadata.Sample.elements) == ["Si", "O"]
        comps = [m for m in captured if m.get("type") == "composition"]
        assert comps and comps[-1]["elements"] == ["Si", "O"]
        assert comps[-1]["percentages"]["O"] == 66.7


# ── COD result normalisation ────────────────────────────────────────────────────
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


# ── COD search / fetch (mocked network) ──────────────────────────────────────────
class TestCodNetwork:
    def test_cod_search_emits_results(self, monkeypatch):
        captured = []
        monkeypatch.setattr(comp, "emit", lambda m: captured.append(m))
        monkeypatch.setattr(comp, "emit_status", lambda *a, **k: None)
        monkeypatch.setattr(comp, "_cod_query", lambda els: [
            {"file": "9", "a": "4.08", "b": "4.08", "c": "4.08", "alpha": "90",
             "beta": "90", "gamma": "90", "sg": "Fm-3m", "sgNumber": "225",
             "formula": "- Ag -", "mineral": "Silver", "vol": "68"}])

        class _Plot:
            window_id = 5
            signal_tree = None
        comp.cod_search(None, _Plot(), {"elements": ["Ag"]})
        res = [m for m in captured if m.get("type") == "cod_results"]
        assert res and res[-1]["window_id"] == 5
        assert res[-1]["results"][0]["id"] == "9"

    def test_cod_search_handles_network_error(self, monkeypatch):
        captured = []
        monkeypatch.setattr(comp, "emit", lambda m: captured.append(m))
        monkeypatch.setattr(comp, "emit_status", lambda *a, **k: None)

        def _boom(_els):
            raise OSError("no network")
        monkeypatch.setattr(comp, "_cod_query", _boom)

        class _Plot:
            window_id = 5
            signal_tree = None
        comp.cod_search(None, _Plot(), {"elements": ["Ag"]})
        res = [m for m in captured if m.get("type") == "cod_results"]
        assert res and res[-1]["results"] == [] and res[-1].get("error")

    def test_fetch_cod_cif_writes_file(self, monkeypatch, tmp_path):
        cif = "data_test\n_cell_length_a 4.08\nloop_\n_atom_site_label\nAg1\n"

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return cif.encode("utf-8")
        monkeypatch.setattr(comp.urllib.request, "urlopen", lambda *a, **k: _Resp())
        path = comp.fetch_cod_cif("1100136")
        assert path.endswith("cod_1100136.cif")
        with open(path) as fh:
            assert "_cell_length_a" in fh.read()

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
        assert any(abs((r["a"] or 0) - 4.09) < 0.2 for r in results)


# ── phases ─────────────────────────────────────────────────────────────────────
class TestPhases:
    """A sample is made of PHASES — what each is made of AND what indexes it.

    The flat composition could not say "Cu and Nb are two separate phases", so
    the COD query — which asks for a structure containing EXACTLY these
    elements — was asked for a Cu-Nb compound. It returns nothing, while the
    two elemental phases it should have found are one query each.
    """

    def test_a_flat_composition_reads_as_one_phase(self):
        # Nothing written by an older SpyDE has a phase list, and every caller
        # is written against one — so the composition has to answer as a phase.
        t = _Tree(_sig())
        comp.write_composition(t, ["Fe", "Ni"], {"Fe": 70.0})
        phases = comp.read_phases(t)
        assert len(phases) == 1
        assert phases[0]["elements"] == ["Fe", "Ni"]
        assert phases[0]["percentages"] == {"Fe": 70.0}
        assert phases[0]["cif_path"] is None

    def test_no_composition_is_no_phases(self):
        assert comp.read_phases(_Tree(_sig())) == []

    def test_phases_keep_the_canonical_flat_fields_true(self):
        # EELS edge suggestion and EDS quantification read Sample.elements, so
        # it has to stay the union of everything present, in the order a person
        # would list it — whatever the grouping is.
        t = _Tree(_sig())
        comp.write_phases(t, [
            {"elements": ["Cu"], "percentages": {"Cu": 60.0}},
            {"elements": ["Nb"], "percentages": {"Nb": 40.0}},
        ])
        elements, percentages = comp.read_composition(t)
        assert elements == ["Cu", "Nb"]
        assert percentages == {"Cu": 60.0, "Nb": 40.0}
        assert list(t.root.metadata.Sample.elements) == ["Cu", "Nb"]

    def test_the_union_dedupes_across_phases(self):
        t = _Tree(_sig())
        comp.write_phases(t, [
            {"elements": ["Zr", "O"]},     # zirconia
            {"elements": ["Zr"]},          # alpha-Zr
        ])
        assert comp.read_composition(t)[0] == ["Zr", "O"]
        assert [p["elements"] for p in comp.read_phases(t)] == [["Zr", "O"], ["Zr"]]

    def test_a_phase_remembers_its_structure(self):
        t = _Tree(_sig())
        comp.write_phases(t, [
            {"elements": ["Cu"], "cif_path": "/x/Cu.cif", "label": "Cu Fm-3m",
             "cod_id": "9008468"},
        ])
        phase = comp.read_phases(t)[0]
        assert phase["cif_path"] == "/x/Cu.cif"
        assert phase["label"] == "Cu Fm-3m"
        assert phase["cod_id"] == "9008468"
        assert comp.phase_label(phase) == "Cu Fm-3m"

    def test_a_structureless_phase_reads_as_its_elements(self):
        assert comp.phase_label({"elements": ["Zr", "O"]}) == "Zr-O"

    def test_a_stored_phase_keeps_a_fixed_shape(self):
        # This dict round-trips through file metadata, so an unknown key must
        # not be carried into it and a missing one must not be absent.
        t = _Tree(_sig())
        comp.write_phases(t, [{"elements": ["Cu"], "scribble": 1}])
        phase = comp.read_phases(t)[0]
        assert set(phase) == {"elements", "percentages", "cif_path", "label", "cod_id"}


class TestPhaseHandlers:
    def _tree(self, *phases):
        t = _Tree(_sig())
        if phases:
            comp.write_phases(t, list(phases))
        return t

    def test_add_remove_and_set(self, monkeypatch):
        t = self._tree({"elements": ["Cu"]})
        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, t))

        comp.add_phase(None, None, {})                      # the "&" button
        assert [p["elements"] for p in comp.read_phases(t)] == [["Cu"], []]

        comp.set_phase(None, None, {"index": 1, "elements": ["Nb"]})
        assert [p["elements"] for p in comp.read_phases(t)] == [["Cu"], ["Nb"]]

        comp.remove_phase(None, None, {"index": 0})
        assert [p["elements"] for p in comp.read_phases(t)] == [["Nb"]]

    def test_setting_a_phase_leaves_its_structure_alone(self, monkeypatch):
        # Changing what a phase is made of does not by itself invalidate the
        # .cif chosen for it, and silently dropping one hides the disagreement.
        t = self._tree({"elements": ["Cu"], "cif_path": "/x/Cu.cif", "label": "Cu"})
        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, t))
        comp.set_phase(None, None, {"index": 0, "elements": ["Cu", "Zn"]})
        phase = comp.read_phases(t)[0]
        assert phase["elements"] == ["Cu", "Zn"]
        assert phase["cif_path"] == "/x/Cu.cif"

    def test_a_structure_names_itself_from_the_file_when_unlabelled(self, monkeypatch):
        t = self._tree({"elements": ["Nb"]})
        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, t))
        comp.set_phase_structure(None, None,
                                 {"index": 0, "cif_path": "/x/beta_Nb_cod4000948.cif"})
        assert comp.read_phases(t)[0]["label"] == "beta_Nb_cod4000948"

    def test_out_of_range_is_ignored_not_crashed(self, monkeypatch):
        t = self._tree({"elements": ["Cu"]})
        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, t))
        comp.remove_phase(None, None, {"index": 7})
        comp.set_phase(None, None, {"index": -1, "elements": ["X"]})
        assert [p["elements"] for p in comp.read_phases(t)] == [["Cu"]]


class TestSearchIsScopedToOnePhase:
    """The whole point: COD is asked for a structure containing EXACTLY these
    elements, so the query has to be one phase's elements, not the sample's."""

    def _capture(self, monkeypatch, tree, payload):
        seen = {}
        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, tree))
        monkeypatch.setattr(comp, "_cod_query",
                            lambda els: seen.setdefault("elements", list(els)) and [])
        monkeypatch.setattr(comp, "emit", lambda m: None)
        monkeypatch.setattr(comp, "emit_status", lambda m: None)
        monkeypatch.setattr(comp, "emit_error", lambda m: None)
        comp.cod_search(None, None, payload)
        return seen.get("elements")

    def test_a_phase_index_scopes_the_query(self, monkeypatch):
        t = _Tree(_sig())
        comp.write_phases(t, [{"elements": ["Cu"]}, {"elements": ["Nb"]}])
        assert self._capture(monkeypatch, t, {"phase": 0}) == ["Cu"]
        assert self._capture(monkeypatch, t, {"phase": 1}) == ["Nb"]

    def test_explicit_elements_still_win(self, monkeypatch):
        t = _Tree(_sig())
        comp.write_phases(t, [{"elements": ["Cu"]}])
        assert self._capture(monkeypatch, t, {"elements": ["Fe"]}) == ["Fe"]

    def test_no_phase_given_falls_back_to_the_whole_composition(self, monkeypatch):
        # A sample that really is one phase keeps working as it did.
        t = _Tree(_sig())
        comp.write_composition(t, ["Fe", "Ni"], {})
        assert self._capture(monkeypatch, t, {}) == ["Fe", "Ni"]


class TestElementsFromCif:
    """A structure file knows what it is made of, so adding one should not also
    require typing the composition in."""

    CIF = os.path.join(os.path.dirname(__file__), "..", "Silver__0011135.cif")

    def test_reads_the_elements(self):
        assert comp.elements_from_cif(self.CIF) == ["Ag"]

    def test_an_unreadable_file_is_not_an_error(self):
        # The phase keeps its structure and simply has no composition — the
        # state it would have been in anyway.
        assert comp.elements_from_cif("/no/such/file.cif") == []

    def test_binding_a_structure_fills_an_empty_composition(self, monkeypatch):
        t = _Tree(_sig())
        comp.write_phases(t, [{"elements": []}])
        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, t))
        comp.set_phase_structure(None, None, {"index": 0, "cif_path": self.CIF})
        assert comp.read_phases(t)[0]["elements"] == ["Ag"]

    def test_an_existing_composition_is_never_overwritten(self, monkeypatch):
        # The user may have said something the file cannot — a solid solution,
        # a measured percentage — and the file does not get to argue with it.
        t = _Tree(_sig())
        comp.write_phases(t, [{"elements": ["Ag", "Cu"], "percentages": {"Ag": 90.0}}])
        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, t))
        comp.set_phase_structure(None, None, {"index": 0, "cif_path": self.CIF})
        phase = comp.read_phases(t)[0]
        assert phase["elements"] == ["Ag", "Cu"]
        assert phase["percentages"] == {"Ag": 90.0}


class TestSampleElementsOutsideAnyPhase:
    """An element can belong to the sample without belonging to a phase — the
    extra oxygen that is in neither structure being indexed against."""

    def test_an_extra_element_survives_a_phase_edit(self, monkeypatch):
        t = _Tree(_sig())
        comp.write_composition(t, ["Ti", "O", "C"], {})      # C is in no phase
        comp.write_phases(t, [{"elements": ["Ti", "O"]}])
        assert comp.read_composition(t)[0] == ["Ti", "O", "C"]

        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, t))
        comp.set_phase(None, None, {"index": 0, "elements": ["Ti"]})
        # Rebuilding the sample from the phases alone would have dropped C.
        assert comp.read_composition(t)[0] == ["Ti", "O", "C"]

    def test_a_phase_element_joins_the_sample(self, monkeypatch):
        t = _Tree(_sig())
        comp.write_composition(t, ["Ti"], {})
        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, t))
        comp.set_phase(None, None, {"index": 0, "elements": ["Ti", "O"]})
        assert comp.read_composition(t)[0] == ["Ti", "O"]

    def test_unticking_an_element_removes_it_from_the_phases_too(self, monkeypatch):
        # A phase is a SUBSET of the sample, so an element the sample no longer
        # has cannot stay in one — and if it did, the next phase write would put
        # it back and the removal would look broken.
        t = _Tree(_sig())
        comp.write_phases(t, [{"elements": ["Ti", "O"]}, {"elements": ["O"]}])
        monkeypatch.setattr(comp, "_src_plot_tree", lambda s, p: (None, t))
        comp.set_composition(None, None, {"elements": ["Ti"], "percentages": {}})
        assert comp.read_composition(t)[0] == ["Ti"]
        assert [p["elements"] for p in comp.read_phases(t)] == [["Ti"], []]

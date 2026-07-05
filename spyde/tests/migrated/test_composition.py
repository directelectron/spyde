"""
test_composition.py — sample composition metadata + the COD "easy CIF" picker.

Composition is stored at the HyperSpy-canonical ``metadata.Sample.elements`` /
``Sample.composition``. The COD search/normalise/fetch logic is unit-tested with
mocked network; one network-guarded test hits the live COD API (skipped offline).
"""
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

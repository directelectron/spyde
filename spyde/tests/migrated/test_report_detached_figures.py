"""
test_report_detached_figures.py — a saved figure comes back INTERACTIVE.

A figure cell persists as a RECIPE (``figures/<id>.yaml``) plus a baked still
(``assets/<id>.png``). The recipe points at the SOURCE signal, so reopening a
report without that data loaded gave you a flat PNG: you could look at the
figure but not pan, zoom, or touch a widget. Since
``figure_builder.build_figure`` wants a spec AND a per-layer snapshot map, and
the spec already round-tripped, only the pixels were missing.

``data/<id>.npz`` supplies them. On open, a cell whose sources don't resolve but
whose pixels WERE saved is rebuilt into a real anyplotlib figure and marked
DETACHED — every interaction works; the one thing it cannot do is refresh from a
signal that isn't there.

Covered:
* the npz pack/unpack round-trip, including what it refuses to pickle,
* ``read_report_snapshots`` as a SIBLING of read_report (whose 2-tuple is
  unpacked at ~20 call sites and must not change),
* a report written WITHOUT data still opens (back-compat) and lands offline,
* a report written WITH data opens DETACHED, not offline,
* re-saving a detached report round-trips the pixels it was opened with,
* the per-cell size cap.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from spyde.actions.report import handlers as h
from spyde.actions.report import model as m
from spyde.actions.report.model import (
    Cell, FigureSpec, LayerSpec, PanelSpec, ReportDoc, read_report,
    read_report_snapshots, write_report,
)


def _tmp(name: str) -> str:
    return os.path.join(tempfile.mkdtemp(), name)


def _states(messages):
    return [msg for msg in messages if msg.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


class TestSnapshotNpzRoundTrip:
    def test_arrays_survive_exactly(self):
        snap = {
            ("p1", "l1"): np.arange(64, dtype=np.uint16).reshape(8, 8),
            ("p1", "l2"): np.linspace(0, 1, 32, dtype=np.float32).reshape(4, 8),
        }
        back = m.npz_from_snapshots(m.snapshots_to_npz(snap))
        assert set(back) == set(snap)
        for key in snap:
            assert np.array_equal(back[key], snap[key])
            assert back[key].dtype == snap[key].dtype

    def test_object_arrays_are_dropped_not_pickled(self):
        """A report is a file people email each other, so loading one must never
        be a code-execution path (np.load runs with allow_pickle=False)."""
        snap = {("p", "obj"): np.array([{"a": 1}, None], dtype=object),
                ("p", "ok"): np.ones((2, 2), dtype=np.uint8)}
        back = m.npz_from_snapshots(m.snapshots_to_npz(snap))
        assert set(back) == {("p", "ok")}

    def test_empty_map_packs_to_nothing(self):
        assert m.snapshots_to_npz({}) == b""
        assert m.npz_from_snapshots(b"") == {}

    def test_corrupt_blob_degrades_to_empty(self):
        """A damaged data blob must cost the figure its interactivity, never the
        whole open."""
        assert m.npz_from_snapshots(b"not an npz at all") == {}

    def test_ids_containing_odd_characters_round_trip(self):
        snap = {("panel with space", "layer-1.2"): np.zeros((2, 2), np.uint8)}
        back = m.npz_from_snapshots(m.snapshots_to_npz(snap))
        assert set(back) == set(snap)


class TestZipCarriesTheData:
    def _doc(self):
        spec = FigureSpec(panels=[PanelSpec(id="p1", layers=[LayerSpec(id="l1")])])
        doc = ReportDoc(title="t")
        doc.cells = [Cell(id="c1", cell_type="figure", spec=spec)]
        return doc

    def test_data_written_and_read_back(self):
        arr = np.arange(100, dtype=np.uint16).reshape(10, 10)
        path = _tmp("with_data.spyde-report")
        write_report(self._doc(), path, assets={"c1": b"PNG"},
                     snapshots={"c1": m.snapshots_to_npz({("p1", "l1"): arr})})

        got = read_report_snapshots(path)
        assert set(got) == {"c1"}
        assert np.array_equal(got["c1"][("p1", "l1")], arr)

    def test_read_report_arity_is_unchanged(self):
        """~20 call sites unpack this 2-tuple; the pixels ride in a sibling
        reader precisely so none of them had to change."""
        path = _tmp("arity.spyde-report")
        write_report(self._doc(), path, assets={"c1": b"PNG"},
                     snapshots={"c1": m.snapshots_to_npz(
                         {("p1", "l1"): np.zeros((4, 4), np.uint8)})})
        doc, assets = read_report(path)
        assert doc.cells[0].id == "c1"
        assert assets["c1"] == b"PNG"

    def test_report_written_without_data_reads_empty(self):
        """Back-compat: every report saved before this existed."""
        path = _tmp("no_data.spyde-report")
        write_report(self._doc(), path, assets={"c1": b"PNG"})
        assert read_report_snapshots(path) == {}

    def test_a_non_report_zip_does_not_raise(self):
        import zipfile
        path = _tmp("junk.spyde-report")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("data/bad.npz", b"garbage")
        assert read_report_snapshots(path) == {}


class TestAssembleSnapshots:
    def test_packs_live_snapshots_for_figure_cells(self, window):
        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        spec = FigureSpec(panels=[PanelSpec(id="p1", layers=[LayerSpec(id="l1")])])
        mgr.doc.cells.append(Cell(id="c1", cell_type="figure", spec=spec))
        mgr.set_snapshot("c1", "p1", "l1", np.ones((4, 4), np.uint8))

        packed = mgr.assemble_snapshots()
        assert set(packed) == {"c1"}
        back = m.npz_from_snapshots(packed["c1"])
        assert np.array_equal(back[("p1", "l1")], np.ones((4, 4), np.uint8))

    def test_falls_back_to_the_snapshots_the_report_was_opened_with(self, window):
        """Re-saving a report whose sources were never available must not
        silently drop the pixels it came with."""
        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        spec = FigureSpec(panels=[PanelSpec(id="p1", layers=[LayerSpec(id="l1")])])
        mgr.doc.cells.append(Cell(id="c1", cell_type="figure", spec=spec))
        arr = np.full((3, 3), 7, np.uint8)
        mgr._loaded_snapshots["c1"] = {("p1", "l1"): arr}

        packed = mgr.assemble_snapshots()
        assert np.array_equal(
            m.npz_from_snapshots(packed["c1"])[("p1", "l1")], arr)

    def test_a_cell_over_the_cap_saves_without_data(self, window):
        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        spec = FigureSpec(panels=[PanelSpec(id="p1", layers=[LayerSpec(id="l1")])])
        mgr.doc.cells.append(Cell(id="c1", cell_type="figure", spec=spec))
        # Random noise so compression can't shrink it under the cap.
        rng = np.random.default_rng(0)
        mgr.set_snapshot("c1", "p1", "l1",
                         rng.integers(0, 255, (2048, 2048), dtype=np.uint8))
        mgr.SNAPSHOT_MAX_BYTES = 1024

        assert mgr.assemble_snapshots() == {}

    def test_cells_without_a_spec_are_skipped(self, window):
        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        mgr.doc.cells.append(Cell(id="cmd", cell_type="markdown", source="x"))
        mgr._snapshots["cmd"] = {("p", "l"): np.zeros((2, 2), np.uint8)}
        assert mgr.assemble_snapshots() == {}


class TestOpenRestoresDetached:
    """The behaviour the whole change exists for."""

    def _write(self, path, *, with_data: bool):
        spec = FigureSpec(panels=[PanelSpec(id="p1", layers=[LayerSpec(id="l1")])])
        doc = ReportDoc(title="detached")
        doc.cells = [Cell(id="c1", cell_type="figure", spec=spec)]
        arr = np.arange(256, dtype=np.uint8).reshape(16, 16)
        snaps = ({"c1": m.snapshots_to_npz({("p1", "l1"): arr})}
                 if with_data else None)
        write_report(doc, path, assets={"c1": b"PNG"}, snapshots=snaps)
        return arr

    def test_without_saved_data_the_cell_is_offline(self, window):
        """The old behaviour, unchanged: no pixels → the flat PNG."""
        session, messages = window["window"], window["messages"]
        path = _tmp("offline.spyde-report")
        self._write(path, with_data=False)
        messages.clear()
        h.report_open(session, None, {"path": path})

        mgr = session._report
        assert "c1" in mgr._offline
        assert "c1" not in mgr._detached
        entry = _last_state(messages)["cells"][0]
        assert entry["data_offline"] is True
        assert entry["data_detached"] is False

    def test_with_saved_data_the_cell_is_detached_and_live(self, window):
        session, messages = window["window"], window["messages"]
        path = _tmp("detached.spyde-report")
        arr = self._write(path, with_data=True)
        messages.clear()
        h.report_open(session, None, {"path": path})

        mgr = session._report
        # THE assertion: interactive, not a static fallback.
        assert "c1" in mgr._detached, "saved pixels should rebuild a live figure"
        assert "c1" not in mgr._offline
        assert mgr._window_by_cell.get("c1") is not None, "no live figure window"
        # The pixels really are the ones that were saved.
        assert np.array_equal(mgr.snapshot_map("c1")[("p1", "l1")], arr)

        entry = _last_state(messages)["cells"][0]
        assert entry["data_detached"] is True
        assert entry["data_offline"] is False

    def test_a_detached_report_re_saves_its_pixels(self, window):
        """Open detached → save → the next open is detached too, not offline."""
        session = window["window"]
        first = _tmp("d1.spyde-report")
        arr = self._write(first, with_data=True)
        h.report_open(session, None, {"path": first})

        second = _tmp("d2.spyde-report")
        packed = session._report.assemble_snapshots()
        write_report(session._report.doc, second,
                     assets={"c1": b"PNG"}, snapshots=packed)

        again = read_report_snapshots(second)
        assert np.array_equal(again["c1"][("p1", "l1")], arr)

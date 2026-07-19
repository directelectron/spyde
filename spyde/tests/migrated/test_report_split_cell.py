"""
test_report_split_cell.py — Wave A: the SPLIT-block primitive.

A split cell (``cell_type="split"``) is ONE atomic block: a TEXT side (``source``)
BESIDE a FIGURE/PHOTO side (the SAME ``spec``/snapshot OR ``image_ext``/bytes a
figure/image cell uses), plus ``split_layout`` ("text-left" | "text-right").

Covered here:
* report.md round-trip (text + figure/image ref + ``spyde:split`` marker;
  split_layout preserved; back-compat — a plain markdown/figure/image cell is NOT
  mis-parsed as a split);
* ``report_add_split_cell`` creates one; ``report_set_split_layout`` sets it;
  ``report_set_split_figure`` fills the figure side (snapshot captured);
* a ``.spyde-report`` zip round-trip restores the split cell + its asset;
* ``slide_columns`` yields a ``split`` row; export mode:'slides'/'static' renders
  the 2-col grid; the LEGACY adjacency ``column`` 2-col path still loads/renders.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from spyde.actions.report import export_html as ex
from spyde.actions.report import handlers as h
from spyde.actions.report.model import (
    Cell, FigureSpec, PanelSpec, ReportDoc, parse_report_md, read_report,
    serialize_report_md, slide_columns, write_report,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _cells(messages):
    return _last_state(messages)["cells"]


def _prime_plot_data(session):
    for p in session._plots:
        if isinstance(getattr(p, "current_data", None), np.ndarray):
            continue
        try:
            sig = p.plot_state.current_signal
            frame = np.asarray(sig.data)
            if frame.ndim > 2:
                frame = frame.reshape(-1, *frame.shape[-2:])[0]
            p.current_data = np.ascontiguousarray(frame.astype(np.float32))
            p._last_levels = (float(np.nanmin(p.current_data)),
                              float(np.nanmax(p.current_data)))
        except Exception:
            pass


def _signal_wid(session):
    for p in session._plots:
        if not getattr(p, "is_navigator", False) and p.window_id is not None:
            return p.window_id
    return session._plots[0].window_id


# ── model: report.md round-trip ─────────────────────────────────────────────────


class TestSplitSerialization:
    def test_image_side_round_trips(self):
        c = Cell(cell_type="split", source="describe the photo",
                 image_ext="jpg", caption="micrograph",
                 split_layout="text-right")
        doc = ReportDoc(doc_type="presentation")
        doc.cells = [c]
        md = serialize_report_md(doc)
        assert "spyde:split text-right" in md
        assert f"![micrograph](assets/{c.id}.jpg)" in md
        doc2 = parse_report_md(md)
        sc = doc2.cells[0]
        assert sc.cell_type == "split"
        assert sc.source == "describe the photo"
        assert sc.image_ext == "jpg"
        assert sc.caption == "micrograph"
        assert sc.split_layout == "text-right"
        assert sc.id == c.id  # the marker id re-associates both sides

    def test_figure_side_round_trips(self):
        spec = FigureSpec(panels=[PanelSpec()])
        c = Cell(cell_type="split", source="the fit result", spec=spec,
                 caption="strain", split_layout="text-left")
        doc = ReportDoc()
        doc.cells = [c]
        md = serialize_report_md(doc)
        assert "spyde:split text-left" in md
        assert f"![strain](assets/{c.id}.png)" in md
        doc2 = parse_report_md(md)
        sc = doc2.cells[0]
        # parse alone leaves spec None (attached from figures/<id>.yaml by
        # read_report), but the cell_type + text + caption + id survive.
        assert sc.cell_type == "split"
        assert sc.source == "the fit result"
        assert sc.caption == "strain"
        assert sc.split_layout == "text-left"
        assert sc.id == c.id

    def test_empty_figure_side_round_trips(self):
        # A split with no figure/photo yet (a drop zone) — text only, no ref.
        c = Cell(cell_type="split", source="just text here",
                 split_layout="text-left")
        doc = ReportDoc()
        doc.cells = [c]
        doc2 = parse_report_md(serialize_report_md(doc))
        sc = doc2.cells[0]
        assert sc.cell_type == "split"
        assert sc.source == "just text here"
        assert sc.spec is None
        assert not sc.image_ext

    def test_split_layout_normalizes(self):
        c = Cell(cell_type="split", source="x", image_ext="png",
                 split_layout="garbage")
        # Serialization normalises an unknown layout to text-left.
        md = serialize_report_md(ReportDoc(cells=[c]))
        assert "spyde:split text-left" in md

    def test_plain_cells_not_mis_parsed_as_split(self):
        # A plain markdown + figure + image body has NO spyde:split marker, so
        # nothing is a split cell (full back-compat).
        legacy = (
            "---\nversion: 1\ntitle: X\n---\n"
            "some paragraph text\n\n"
            "![a fig](assets/cabc12345.png)\n\n"
            "![a photo](assets/cdef67890.jpg)\n"
        )
        doc = parse_report_md(legacy)
        types = [c.cell_type for c in doc.cells]
        assert types == ["markdown", "figure", "image"]
        assert all(c.cell_type != "split" for c in doc.cells)

    def test_split_amid_plain_cells(self):
        # A split cell surrounded by plain cells parses each correctly.
        c_split = Cell(cell_type="split", source="split text", image_ext="png",
                       caption="cap", split_layout="text-right")
        doc = ReportDoc()
        doc.cells = [
            Cell(cell_type="markdown", source="intro para"),
            c_split,
            Cell(cell_type="markdown", source="outro para"),
        ]
        doc2 = parse_report_md(serialize_report_md(doc))
        types = [c.cell_type for c in doc2.cells]
        assert types == ["markdown", "split", "markdown"]
        assert doc2.cells[0].source == "intro para"
        assert doc2.cells[1].source == "split text"
        assert doc2.cells[1].split_layout == "text-right"
        assert doc2.cells[2].source == "outro para"


# ── model: zip round-trip ───────────────────────────────────────────────────────


class TestSplitZipRoundTrip:
    def test_figure_side_zip(self):
        spec = FigureSpec(panels=[PanelSpec()])
        c = Cell(cell_type="split", source="sp text", spec=spec,
                 caption="fig cap", split_layout="text-right")
        doc = ReportDoc(doc_type="presentation", cells=[c])
        tmp = os.path.join(tempfile.gettempdir(), "split_fig.spyde-report")
        try:
            write_report(doc, tmp, assets={c.id: b"PNGBYTES"})
            doc2, assets = read_report(tmp)
            sc = doc2.cells[0]
            assert doc2.doc_type == "presentation"
            assert sc.cell_type == "split"
            assert sc.spec is not None        # attached from figures/<id>.yaml
            assert sc.split_layout == "text-right"
            assert assets.get(sc.id) == b"PNGBYTES"
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_photo_side_zip(self):
        c = Cell(cell_type="split", source="ph text", image_ext="jpg",
                 caption="photo", split_layout="text-left")
        doc = ReportDoc(cells=[c])
        tmp = os.path.join(tempfile.gettempdir(), "split_photo.spyde-report")
        try:
            write_report(doc, tmp, assets={c.id: b"JPGBYTES"})
            doc2, assets = read_report(tmp)
            sc = doc2.cells[0]
            assert sc.cell_type == "split"
            assert sc.spec is None
            assert sc.image_ext == "jpg"
            assert assets.get(sc.id) == b"JPGBYTES"
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_png_photo_side_zip(self):
        # A ``.png`` PHOTO side (no yaml) must NOT be mistaken for a figure and
        # its cell_type must stay "split" (never promoted to image).
        c = Cell(cell_type="split", source="t", image_ext="png",
                 split_layout="text-left")
        doc = ReportDoc(cells=[c])
        tmp = os.path.join(tempfile.gettempdir(), "split_png.spyde-report")
        try:
            write_report(doc, tmp, assets={c.id: b"PNGPHOTO"})
            doc2, assets = read_report(tmp)
            sc = doc2.cells[0]
            assert sc.cell_type == "split"
            assert sc.spec is None
            assert sc.image_ext == "png"
            assert assets.get(sc.id) == b"PNGPHOTO"
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


# ── model: slide_columns ────────────────────────────────────────────────────────


class TestSplitSlideColumns:
    def test_split_cell_is_own_cols_row(self):
        c = Cell(cell_type="split", source="t", image_ext="png")
        rows = slide_columns([c])
        assert len(rows) == 1
        assert rows[0]["kind"] == "split"
        assert rows[0]["cell"] is c

    def test_legacy_column_path_still_works(self):
        # The legacy adjacency 2-col path (column="left"/"right") is unchanged.
        left = Cell(cell_type="markdown", source="L", column="left")
        right = Cell(cell_type="figure", caption="R", column="right",
                     spec=FigureSpec(panels=[PanelSpec()]))
        rows = slide_columns([left, right])
        assert len(rows) == 1
        assert rows[0]["kind"] == "cols"
        assert rows[0]["left"] == [left]
        assert rows[0]["right"] == [right]

    def test_split_closes_open_legacy_cols_row(self):
        left = Cell(cell_type="markdown", source="L", column="left")
        split = Cell(cell_type="split", source="s", image_ext="png")
        rows = slide_columns([left, split])
        assert [r["kind"] for r in rows] == ["cols", "split"]


# ── handlers ────────────────────────────────────────────────────────────────────


class TestSplitHandlers:
    def test_add_split_cell(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {"type": "presentation"})
        h.report_add_split_cell(session, None,
                                {"source": "hello", "layout": "text-right"})
        cells = _cells(messages)
        assert len(cells) == 1
        c = cells[0]
        assert c["cell_type"] == "split"
        assert c["source"] == "hello"
        assert c["split_layout"] == "text-right"
        assert c["split_empty"] is True        # no figure side yet

    def test_add_split_cell_default_layout(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        assert _cells(messages)[0]["split_layout"] == "text-left"

    def test_set_split_layout(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        cid = _cells(messages)[0]["id"]
        h.report_set_split_layout(session, None,
                                  {"cell_id": cid, "layout": "text-right"})
        assert _cells(messages)[0]["split_layout"] == "text-right"

    def test_set_split_layout_unknown_normalizes(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        cid = _cells(messages)[0]["id"]
        h.report_set_split_layout(session, None,
                                  {"cell_id": cid, "layout": "bogus"})
        assert _cells(messages)[0]["split_layout"] == "text-left"

    def test_update_cell_edits_split_text(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "old"})
        cid = _cells(messages)[0]["id"]
        h.report_update_cell(session, None, {"cell_id": cid, "source": "new"})
        assert _cells(messages)[0]["source"] == "new"

    def test_set_split_figure_fills_figure_side(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {"type": "presentation"})
        h.report_add_split_cell(session, None, {"source": "text side"})
        cid = _cells(messages)[0]["id"]
        h.report_set_split_figure(
            session, None, {"cell_id": cid, "source_window_id": wid})
        c = next(x for x in _cells(messages) if x["id"] == cid)
        assert c["cell_type"] == "split"
        assert c["source"] == "text side"        # text side untouched
        assert c.get("figure") is not None       # figure side filled (spec dict)
        assert c["split_empty"] is False
        # the snapshot was captured for the split cell's id
        mgr = h._manager(session)
        assert cid in mgr._snapshots and mgr._snapshots[cid]

    def test_set_split_figure_non_split_is_noop(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "m"})
        cid = _cells(messages)[0]["id"]
        errs_before = [m for m in messages if m.get("type") == "error"]
        h.report_set_split_figure(
            session, None, {"cell_id": cid, "source_window_id": wid})
        errs_after = [m for m in messages if m.get("type") == "error"]
        # It refuses (emits an error) rather than mutating the markdown cell.
        assert len(errs_after) > len(errs_before)
        assert _cells(messages)[0]["cell_type"] == "markdown"


# ── handlers: full zip save/reopen through the manager ──────────────────────────


class TestSplitManagerZip:
    def test_save_reopen_photo_split(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {"type": "presentation"})
        # Add a split cell then attach a PHOTO side by hand (mgr._images), the way
        # a photo drop will (Wave B): here we exercise the persistence contract.
        h.report_add_split_cell(session, None,
                                {"source": "cap text", "layout": "text-right"})
        cid = _cells(messages)[0]["id"]
        mgr = h._manager(session)
        cell = mgr.doc.cell_by_id(cid)
        cell.image_ext = "png"
        cell.caption = "the photo"
        mgr._images[cid] = b"PHOTOBYTES"

        tmp = os.path.join(tempfile.gettempdir(), "mgr_split.spyde-report")
        try:
            from spyde.actions.report.model import write_report as _wr
            assets = mgr.assemble_assets({})
            _wr(mgr.doc, tmp, assets=assets)
            doc2, assets2 = read_report(tmp)
            sc = doc2.cells[0]
            assert doc2.doc_type == "presentation"
            assert sc.cell_type == "split"
            assert sc.source == "cap text"
            assert sc.split_layout == "text-right"
            assert sc.image_ext == "png"
            assert assets2.get(sc.id) == b"PHOTOBYTES"
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


# ── export ──────────────────────────────────────────────────────────────────────


def _render_static(mgr):
    return ex._render_body(mgr, mgr.assemble_assets({}), interactive=False,
                           session=mgr.session)


def _render_slides(mgr):
    return ex._render_slides(mgr, mgr.assemble_assets({}), interactive=False,
                             session=mgr.session)


class TestSplitExport:
    def _mgr_with_photo_split(self, session, messages, layout="text-left"):
        h.report_new(session, None, {"type": "presentation"})
        h.report_add_split_cell(session, None,
                                {"source": "**bold** text side",
                                 "layout": layout})
        cid = _cells(messages)[0]["id"]
        mgr = h._manager(session)
        cell = mgr.doc.cell_by_id(cid)
        cell.image_ext = "png"
        cell.caption = "photo cap"
        # a 1x1 red PNG so the <img> has real bytes
        import base64
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAA"
            "C0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
        mgr._images[cid] = png
        return mgr, cid

    def test_static_export_renders_split_block(self, window):
        session, messages = window["window"], window["messages"]
        mgr, _cid = self._mgr_with_photo_split(session, messages)
        html = _render_static(mgr)
        assert "split-block" in html
        assert "split-text" in html and "split-fig" in html
        assert "<img" in html                   # the photo side

    def test_slides_export_renders_split_block(self, window):
        session, messages = window["window"], window["messages"]
        mgr, _cid = self._mgr_with_photo_split(session, messages)
        html = _render_slides(mgr)
        assert "split-block" in html
        assert "<img" in html

    def test_split_layout_orders_columns(self, window):
        session, messages = window["window"], window["messages"]
        # text-left → text column BEFORE figure column in source order.
        mgr, _cid = self._mgr_with_photo_split(session, messages,
                                               layout="text-left")
        html = _render_static(mgr)
        assert html.index("split-text") < html.index("split-fig")

        session2, messages2 = window["window"], window["messages"]
        mgr2, _cid2 = self._mgr_with_photo_split(session2, messages2,
                                                 layout="text-right")
        html2 = _render_static(mgr2)
        assert html2.index("split-fig") < html2.index("split-text")

    def test_legacy_column_export_still_renders(self, window):
        # A LEGACY 2-col slide (column left/right, NOT a split cell) still renders
        # its .slide-cols grid — the split cell doesn't break the old path.
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {"type": "presentation"})
        h.report_add_cell(session, None, {"cell_type": "markdown",
                                          "source": "left text",
                                          "column": "left"})
        h.report_add_image_cell(session, None, {
            "image_b64": (
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1"
                "HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="),
            "image_ext": "png", "column": "right"})
        # tag the image cell to the right column
        cells = _cells(messages)
        img_id = cells[-1]["id"]
        h.report_set_cell_column(session, None,
                                 {"cell_id": img_id, "column": "right"})
        mgr = h._manager(session)
        html = _render_slides(mgr)
        assert "slide-cols" in html

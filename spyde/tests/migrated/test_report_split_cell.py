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
  the 2-col grid.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from spyde.actions.report import export_html as ex
from spyde.actions.report import handlers as h
from spyde.actions.report.handlers import _is_figure_like
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

    def test_empty_split_does_not_swallow_following_cell(self):
        # REGRESSION (critical data loss): an empty-figure-side split followed by
        # a real figure/image cell must NOT bind that following cell's ref as its
        # own figure side — both cells must survive. Previously the split's open
        # figure side swallowed the next figure ref anywhere later in the doc.
        for follower in (
            Cell(cell_type="figure", caption="REAL FIG"),
            Cell(cell_type="image", image_ext="png", caption="REAL PHOTO"),
        ):
            doc = ReportDoc(cells=[
                Cell(cell_type="split", source="describe"),   # empty figure side
                follower,
            ])
            doc2 = parse_report_md(serialize_report_md(doc))
            assert len(doc2.cells) == 2, (
                f"empty split swallowed the following {follower.cell_type} cell")
            assert doc2.cells[0].cell_type == "split"
            assert doc2.cells[0].spec is None and not doc2.cells[0].image_ext
            # The follower keeps its own identity (a .png figure ref parses as a
            # figure pre-promotion; a non-png image parses as an image).
            assert doc2.cells[1].cell_type in ("figure", "image")

    def test_split_text_with_image_ref_lookalike_stays_intact(self):
        # REGRESSION: a standalone image-ref lookalike inside a split's TEXT must
        # NOT fragment the text or be stolen as the figure side (the text is
        # base64-encoded into the marker, so it is fully opaque).
        src = "intro\n\n![sneaky](assets/otherid.png)\n\nmore text"
        doc = ReportDoc(cells=[
            Cell(cell_type="split", source=src, image_ext="jpg", caption="realcap"),
        ])
        doc2 = parse_report_md(serialize_report_md(doc))
        assert len(doc2.cells) == 1
        sc = doc2.cells[0]
        assert sc.cell_type == "split"
        assert sc.source == src            # text preserved EXACTLY, not truncated
        assert sc.image_ext == "jpg"        # its real figure side, not "otherid"

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

    def test_non_split_cells_are_full_rows(self):
        # Every non-split cell emits its own full-width row (order preserved).
        a = Cell(cell_type="markdown", source="A")
        split = Cell(cell_type="split", source="s", image_ext="png")
        b = Cell(cell_type="figure", caption="B",
                 spec=FigureSpec(panels=[PanelSpec()]))
        rows = slide_columns([a, split, b])
        assert [r["kind"] for r in rows] == ["full", "split", "full"]
        assert [r["cell"] for r in rows] == [a, split, b]


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

    def test_set_split_layout_stacked(self, window):
        # The stacked layouts (text-top / text-bottom) are accepted + round-trip.
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        cid = _cells(messages)[0]["id"]
        for lay in ("text-top", "text-bottom"):
            h.report_set_split_layout(session, None, {"cell_id": cid, "layout": lay})
            assert _cells(messages)[0]["split_layout"] == lay


class TestFigurePlaceholderSlide:
    def test_add_figure_placeholder(self, window):
        # "Add figure slide" creates an empty placeholder figure cell (dashed
        # drop-zone) — with slide_break so it's its own slide.
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {"type": "presentation"})
        h.report_add_figure_placeholder(session, None, {"slide_break": True})
        cells = _cells(messages)
        fig = next(c for c in cells if c["cell_type"] == "figure")
        assert fig["placeholder"] is True
        assert fig["slide_break"] is True

    def test_figure_placeholder_roundtrips(self, window, tmp_path):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {"type": "presentation"})
        h.report_add_figure_placeholder(session, None, {"slide_break": True})
        mgr = session._report
        p = str(tmp_path / "p.spyde-report")
        write_report(mgr.doc, p, mgr.assemble_assets({}))
        doc2, _ = read_report(p)
        fig = next(c for c in doc2.cells if c.cell_type == "figure")
        assert fig.placeholder is True and fig.slide_break is True

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

    def test_split_figure_side_is_refreshable(self, stem_4d_dataset):
        # REGRESSION: a split cell's FIGURE side (once filled) must be refreshable
        # — report_refresh_figure previously rejected any cell_type != "figure",
        # so a split's figure could never be pulled fresh from live data. The fix
        # admits a split-with-a-spec (_is_figure_like).
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {"type": "presentation"})
        h.report_add_split_cell(session, None, {"source": "txt"})
        cid = _cells(messages)[0]["id"]
        h.report_set_split_figure(
            session, None, {"cell_id": cid, "source_window_id": wid})
        assert _is_figure_like(h._manager(session).doc.cell_by_id(cid))
        errs_before = len([m for m in messages if m.get("type") == "error"])
        # Refresh must run (not silently return on the type gate) and keep the cell
        # a split with its figure side intact — no crash, no error.
        h.report_refresh_figure(session, None, {"cell_id": cid})
        errs_after = len([m for m in messages if m.get("type") == "error"])
        assert errs_after == errs_before
        c = next(x for x in _cells(messages) if x["id"] == cid)
        assert c["cell_type"] == "split"
        assert c["source"] == "txt"
        assert c.get("figure") is not None
        assert c["split_empty"] is False

    def test_photo_split_is_not_figure_like(self):
        # A split whose side is a PHOTO (no spec) is NOT figure-like — you can't
        # repfig-edit a photo. Guards the _is_figure_like boundary.
        assert not _is_figure_like(Cell(cell_type="split", image_ext="png"))
        assert _is_figure_like(Cell(cell_type="split", spec=FigureSpec()))
        assert _is_figure_like(Cell(cell_type="figure"))
        assert not _is_figure_like(Cell(cell_type="markdown"))
        assert not _is_figure_like(None)

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

    def test_stacked_layout_export(self, window):
        # text-top → text before figure + the --stacked modifier class (rows).
        session, messages = window["window"], window["messages"]
        mgr, _cid = self._mgr_with_photo_split(session, messages, layout="text-top")
        html = _render_static(mgr)
        assert "split-block--stacked" in html
        assert html.index("split-text") < html.index("split-fig")
        # text-bottom → figure before text, still stacked.
        session2, messages2 = window["window"], window["messages"]
        mgr2, _cid2 = self._mgr_with_photo_split(session2, messages2, layout="text-bottom")
        html2 = _render_static(mgr2)
        assert "split-block--stacked" in html2
        assert html2.index("split-fig") < html2.index("split-text")


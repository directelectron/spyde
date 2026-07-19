"""
test_report_slide_columns.py — Report Builder 2-column slide layout.

Covers the per-cell ``column`` field (text BESIDE a figure/photo on one slide):

* ``column`` round-trips through report.md serialization (left/right markers;
  "full"/unknown/absent → "" == full width),
* the ``slide_columns`` grouping helper turns a slide's cells into full / cols
  rows (the rule the renderer + export share),
* the ``report_set_cell_column`` verb mutates + emits (and normalises),
* ``report_add_cell`` accepts ``column`` inline,
* ``report_export_html {mode:'slides'}`` produces the 2-col grid divs for a
  text+figure slide.
"""
from __future__ import annotations

import numpy as np

from spyde.actions.report import export_html as ex
from spyde.actions.report import handlers as h
from spyde.actions.report import model as m


# ── helpers (mirror test_report_present) ───────────────────────────────────────


def _states(messages):
    return [msg for msg in messages if msg.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _exported(messages):
    return [msg for msg in messages if msg.get("type") == "report_exported"]


def _errors(messages):
    return [msg for msg in messages if msg.get("type") == "error"]


def _signal_window_id(session):
    for p in session._plots:
        if not getattr(p, "is_navigator", False) and p.window_id is not None:
            return p.window_id
    return session._plots[0].window_id


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


# ── column round-trip ──────────────────────────────────────────────────────────


class TestColumnRoundTrip:
    def test_left_right_round_trip(self):
        doc = m.ReportDoc(title="Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="Notes", column="left"))
        doc.cells.append(m.Cell(id="cf1", cell_type="figure", caption="Fig",
                                column="right"))
        text = m.serialize_report_md(doc)
        assert "<!-- spyde:column left -->" in text
        assert "<!-- spyde:column right -->" in text
        back = m.parse_report_md(text)
        assert [c.column for c in back.cells] == ["left", "right"]
        # Cell types + order preserved through the markers.
        assert [c.cell_type for c in back.cells] == ["markdown", "figure"]
        assert back.cells[0].source == "Notes"

    def test_full_and_default_emit_no_marker(self):
        """A "full"/"" cell writes NO column marker (default full width). The two
        cells are kept distinct by a slide-break so parse doesn't coalesce them."""
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="A"))          # ""
        doc.cells.append(m.Cell(cell_type="markdown", source="B", column="full",
                                slide_break=True))
        text = m.serialize_report_md(doc)
        assert "spyde:column" not in text
        back = m.parse_report_md(text)
        assert [c.column for c in back.cells] == ["", ""]

    def test_absent_column_is_empty_on_old_file(self):
        """An OLD report.md (no column markers) loads every cell as column=''."""
        old = ("---\nversion: 1\ntitle: Old\n---\n\n"
               "# Heading\n\n![Cap](assets/cf9.png)\n")
        back = m.parse_report_md(old)
        assert all(c.column == "" for c in back.cells)

    def test_unknown_column_value_collapses_to_empty(self):
        raw = ("---\nversion: 1\ntitle: X\n---\n\n"
               "<!-- spyde:column middle -->\n\n# Heading\n")
        back = m.parse_report_md(raw)
        assert back.cells[0].column == ""

    def test_column_survives_zip(self, tmp_path):
        doc = m.ReportDoc(title="Zipped Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="Text", column="left"))
        doc.cells.append(m.Cell(cell_type="markdown", source="More", column="right"))
        path = str(tmp_path / "deck.spyde-report")
        m.write_report(doc, path)
        back, _assets = m.read_report(path)
        assert [c.column for c in back.cells] == ["left", "right"]

    def test_double_round_trip_stable_with_column(self):
        doc = m.ReportDoc(title="Stable")
        doc.created = doc.modified = "2020-01-01T00:00:00+00:00"
        doc.cells.append(m.Cell(cell_type="markdown", source="A", column="left"))
        doc.cells.append(m.Cell(cell_type="markdown", source="B", column="right",
                                slide_break=True))
        t1 = m.serialize_report_md(doc)
        back = m.parse_report_md(t1)
        back.created = back.modified = "2020-01-01T00:00:00+00:00"
        t2 = m.serialize_report_md(back)
        assert t1 == t2

    def test_column_and_slide_break_and_live_action_on_one_cell(self):
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="First"))
        doc.cells.append(m.Cell(cell_type="markdown", source="Left body",
                                slide_break=True, column="left",
                                live_action={"tutorial": "strain"}))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert back.cells[1].column == "left"
        assert back.cells[1].slide_break is True
        assert back.cells[1].live_action == {"tutorial": "strain"}


# ── slide_columns grouping ──────────────────────────────────────────────────────


class TestSlideColumnsGrouping:
    def test_full_left_right_full(self):
        cells = [
            m.Cell(id="a", cell_type="markdown", source="head"),          # full
            m.Cell(id="b", cell_type="markdown", source="text", column="left"),
            m.Cell(id="c", cell_type="figure", column="right"),
            m.Cell(id="d", cell_type="markdown", source="foot"),          # full
        ]
        rows = m.slide_columns(cells)
        assert [r["kind"] for r in rows] == ["full", "cols", "full"]
        assert rows[0]["cell"].id == "a"
        assert [c.id for c in rows[1]["left"]] == ["b"]
        assert [c.id for c in rows[1]["right"]] == ["c"]
        assert rows[2]["cell"].id == "d"

    def test_all_full_is_one_row_each(self):
        cells = [m.Cell(cell_type="markdown", source=str(i)) for i in range(3)]
        rows = m.slide_columns(cells)
        assert [r["kind"] for r in rows] == ["full", "full", "full"]

    def test_text_left_figure_right_is_one_cols_row(self):
        """The canonical text+figure slide → a single side-by-side row."""
        cells = [
            m.Cell(id="t", cell_type="markdown", source="notes", column="left"),
            m.Cell(id="f", cell_type="figure", column="right"),
        ]
        rows = m.slide_columns(cells)
        assert len(rows) == 1 and rows[0]["kind"] == "cols"
        assert [c.id for c in rows[0]["left"]] == ["t"]
        assert [c.id for c in rows[0]["right"]] == ["f"]

    def test_multiple_cells_per_column(self):
        cells = [
            m.Cell(id="l1", cell_type="markdown", column="left"),
            m.Cell(id="l2", cell_type="markdown", column="left"),
            m.Cell(id="r1", cell_type="figure", column="right"),
        ]
        rows = m.slide_columns(cells)
        assert len(rows) == 1 and rows[0]["kind"] == "cols"
        assert [c.id for c in rows[0]["left"]] == ["l1", "l2"]
        assert [c.id for c in rows[0]["right"]] == ["r1"]

    def test_full_between_two_cols_rows_splits_them(self):
        cells = [
            m.Cell(id="a", cell_type="markdown", column="left"),
            m.Cell(id="b", cell_type="markdown", column="right"),
            m.Cell(id="c", cell_type="markdown"),               # full closes row 1
            m.Cell(id="d", cell_type="markdown", column="left"),
            m.Cell(id="e", cell_type="markdown", column="right"),
        ]
        rows = m.slide_columns(cells)
        assert [r["kind"] for r in rows] == ["cols", "full", "cols"]
        assert [c.id for c in rows[0]["left"]] == ["a"]
        assert [c.id for c in rows[2]["right"]] == ["e"]

    def test_every_cell_lands_in_one_row(self):
        cells = [
            m.Cell(id="a", cell_type="markdown"),
            m.Cell(id="b", cell_type="markdown", column="left"),
            m.Cell(id="c", cell_type="markdown", column="right"),
            m.Cell(id="d", cell_type="markdown", column="left"),
        ]
        flat = []
        for r in m.slide_columns(cells):
            if r["kind"] == "full":
                flat.append(r["cell"].id)
            else:
                flat.extend(c.id for c in r["left"])
                flat.extend(c.id for c in r["right"])
        assert sorted(flat) == ["a", "b", "c", "d"]

    def test_empty_cells_no_rows(self):
        assert m.slide_columns([]) == []


# ── the report_set_cell_column verb ─────────────────────────────────────────────


class TestSetColumnVerb:
    def test_set_left_then_right_then_full(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id

        messages.clear()
        h.report_set_cell_column(session, None, {"cell_id": cid, "column": "left"})
        cell = next(c for c in _last_state(messages)["cells"] if c["id"] == cid)
        assert cell["column"] == "left"

        h.report_set_cell_column(session, None, {"cell_id": cid, "column": "right"})
        assert session._report.doc.cells[0].column == "right"

        # "full" clears it back to "".
        h.report_set_cell_column(session, None, {"cell_id": cid, "column": "full"})
        assert session._report.doc.cells[0].column == ""

    def test_unknown_value_normalises_to_full(self, window):
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id
        h.report_set_cell_column(session, None, {"cell_id": cid, "column": "left"})
        h.report_set_cell_column(session, None, {"cell_id": cid, "column": "bogus"})
        assert session._report.doc.cells[0].column == ""

    def test_set_column_unknown_cell_is_noop(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        h.report_set_cell_column(session, None, {"cell_id": "nope", "column": "left"})
        assert not _errors(messages)

    def test_add_cell_accepts_column_inline(self, window):
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "Left", "column": "left"})
        assert session._report.doc.cells[0].column == "left"

    def test_state_ships_column_field(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id
        cell = next(c for c in _last_state(messages)["cells"] if c["id"] == cid)
        # Default full width ships as "".
        assert cell["column"] == ""


# ── slides HTML export with columns ─────────────────────────────────────────────


class TestSlideColumnsExport:
    def test_text_plus_figure_slide_emits_2col_grid(self, tem_2d_dataset, tmp_path):
        """A slide with a LEFT text cell + a RIGHT figure cell exports a
        .slide-cols grid with both children."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        h.report_new(session, None, {})
        # Left: a text cell. Right: a figure. One slide (no break between).
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "## Notes\n\nDescribe the pattern.",
            "html": "<h2>Notes</h2><p>Describe the pattern.</p>", "column": "left"})
        h.report_add_figure(session, None, {
            "source_window_id": wid, "caption": "A pattern"})
        fig_cell = next(c for c in session._report.doc.cells
                        if c.cell_type == "figure")
        h.report_set_cell_column(session, None,
                                 {"cell_id": fig_cell.id, "column": "right"})

        path = str(tmp_path / "cols_deck.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        assert not _errors(messages)
        exp = _exported(messages)
        assert exp and exp[0]["kind"] == "html-slides"

        html = open(path, encoding="utf-8").read()
        # One slide, one 2-col grid inside it, with a left + right column.
        assert html.count('<section class="slide">') == 1
        assert 'class="slide-cols"' in html
        assert html.count('class="slide-col"') == 2
        assert 'grid-template-columns: 1fr 1fr' in html   # the grid CSS
        # The text rode into the LEFT column and the figure into the RIGHT.
        assert "<h2>Notes</h2>" in html
        assert "A pattern" in html
        assert ("<iframe" in html
                or '<img src="data:image/png;base64,' in html)
        # The text's column div precedes the figure's (left before right).
        left_idx = html.find("<h2>Notes</h2>")
        fig_idx = html.find("A pattern")
        assert 0 < left_idx < fig_idx

    def test_full_width_slide_has_no_grid(self, window, tmp_path):
        """A slide of only full-width cells emits NO .slide-cols grid."""
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Title", "html": "<h1>Title</h1>"})
        path = str(tmp_path / "full.html")
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        html = open(path, encoding="utf-8").read()
        assert html.count('<section class="slide">') == 1
        assert 'class="slide-cols"' not in html
        assert "<h1>Title</h1>" in html

    def test_static_export_ignores_column_and_stacks(self, window, tmp_path):
        """Static (article) export stacks column-assigned cells (no grid) — the
        2-column layout is a slides feature."""
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "Left", "html": "<p>Left</p>",
            "column": "left"})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "Right", "html": "<p>Right</p>",
            "column": "right"})
        path = str(tmp_path / "article.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "static", "path": path})
        assert not _errors(messages)
        html = open(path, encoding="utf-8").read()
        assert 'class="slide-cols"' not in html
        assert "<p>Left</p>" in html and "<p>Right</p>" in html

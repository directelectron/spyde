"""
test_report_present.py — Report Builder Phase 6 (Present mode / slides).

Covers the slide-grouping model + serialization + the slides HTML export against
a real Qt-free ``Session`` (the ``window`` / ``tem_2d_dataset`` fixtures):

* ``slide_break`` round-trips through report.md serialization (set on a cell,
  save+reload, flag preserved; absent on old files = False),
* ``live_action`` round-trips likewise,
* ``ReportDoc.slides()`` groups cells into slides correctly,
* ``report_toggle_slide_break`` / ``report_set_live_action`` verbs mutate + emit,
* ``report_add_cell`` accepts the Present-mode fields inline (deck seeding),
* ``report_export_html {mode:'slides'}`` produces the slide shell + one ``.slide``
  per group + the cells' content.
"""
from __future__ import annotations

import numpy as np

from spyde.actions.report import export_html as ex
from spyde.actions.report import handlers as h
from spyde.actions.report import model as m


# ── helpers (mirror test_report_export / test_report_model) ────────────────────


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


# ── slide_break + live_action model round-trip ─────────────────────────────────


class TestSlideMarkersRoundTrip:
    def test_slide_break_round_trips(self):
        doc = m.ReportDoc(title="Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="# Slide one"))
        doc.cells.append(m.Cell(cell_type="markdown", source="## Slide two",
                                slide_break=True))
        doc.cells.append(m.Cell(id="cf1", cell_type="figure", caption="Fig",
                                slide_break=True))
        text = m.serialize_report_md(doc)
        assert "<!-- spyde:slide-break -->" in text
        back = m.parse_report_md(text)
        assert [c.slide_break for c in back.cells] == [False, True, True]
        # Cell types + order preserved through the markers.
        assert [c.cell_type for c in back.cells] == ["markdown", "markdown", "figure"]
        assert back.cells[1].source == "## Slide two"

    def test_absent_slide_break_is_false_on_old_file(self):
        """An OLD report.md (no slide-break markers) loads every cell as
        slide_break=False."""
        old = ("---\nversion: 1\ntitle: Old\n---\n\n"
               "# Heading\n\n![Cap](assets/cf9.png)\n")
        back = m.parse_report_md(old)
        assert all(c.slide_break is False for c in back.cells)

    def test_live_action_round_trips(self):
        doc = m.ReportDoc(title="Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="Intro",
                                live_action={"tutorial": "strain", "guide": "strain"}))
        text = m.serialize_report_md(doc)
        assert "<!-- spyde:live-action" in text
        back = m.parse_report_md(text)
        assert back.cells[0].live_action == {"tutorial": "strain", "guide": "strain"}

    def test_absent_live_action_is_none(self):
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="No action"))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert back.cells[0].live_action is None

    def test_both_markers_on_one_cell(self):
        doc = m.ReportDoc(title="Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="First"))
        doc.cells.append(m.Cell(cell_type="markdown", source="Live slide",
                                slide_break=True,
                                live_action={"tutorial": "find_vectors"}))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert back.cells[1].slide_break is True
        assert back.cells[1].live_action == {"tutorial": "find_vectors"}

    def test_double_round_trip_stable_with_markers(self):
        doc = m.ReportDoc(title="Stable")
        doc.created = doc.modified = "2020-01-01T00:00:00+00:00"
        doc.cells.append(m.Cell(cell_type="markdown", source="A"))
        doc.cells.append(m.Cell(cell_type="markdown", source="B", slide_break=True,
                                live_action={"guide": "orientation"}))
        t1 = m.serialize_report_md(doc)
        back = m.parse_report_md(t1)
        back.created = back.modified = "2020-01-01T00:00:00+00:00"
        t2 = m.serialize_report_md(back)
        assert t1 == t2

    def test_slide_break_survives_zip(self, tmp_path):
        doc = m.ReportDoc(title="Zipped Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="One"))
        doc.cells.append(m.Cell(cell_type="markdown", source="Two", slide_break=True,
                                live_action={"tutorial": "movie"}))
        path = str(tmp_path / "deck.spyde-report")
        m.write_report(doc, path)
        back, _assets = m.read_report(path)
        assert [c.slide_break for c in back.cells] == [False, True]
        assert back.cells[1].live_action == {"tutorial": "movie"}


# ── slides() grouping ──────────────────────────────────────────────────────────


class TestSlidesGrouping:
    def test_groups_by_slide_break(self):
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="A"))
        doc.cells.append(m.Cell(cell_type="markdown", source="A2"))
        doc.cells.append(m.Cell(cell_type="markdown", source="B", slide_break=True))
        doc.cells.append(m.Cell(id="cf", cell_type="figure", slide_break=True))
        doc.cells.append(m.Cell(cell_type="markdown", source="C trailing"))
        slides = doc.slides()
        assert len(slides) == 3
        assert [c.source for c in slides[0]] == ["A", "A2"]
        assert [c.source for c in slides[1]] == ["B"]
        # The figure slide keeps its trailing markdown too.
        assert [c.cell_type for c in slides[2]] == ["figure", "markdown"]

    def test_empty_doc_no_slides(self):
        assert m.ReportDoc().slides() == []

    def test_no_breaks_single_slide(self):
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="A"))
        doc.cells.append(m.Cell(cell_type="markdown", source="B"))
        slides = doc.slides()
        assert len(slides) == 1 and len(slides[0]) == 2

    def test_leading_break_on_first_cell_is_noop(self):
        """A slide_break on cell 0 can't split before the start — one slide."""
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="A", slide_break=True))
        doc.cells.append(m.Cell(cell_type="markdown", source="B"))
        slides = doc.slides()
        assert len(slides) == 1
        assert [c.source for c in slides[0]] == ["A", "B"]

    def test_every_cell_lands_in_exactly_one_slide(self):
        doc = m.ReportDoc()
        for i in range(6):
            doc.cells.append(m.Cell(cell_type="markdown", source=str(i),
                                    slide_break=(i % 2 == 0)))
        flat = [c for s in doc.slides() for c in s]
        assert len(flat) == len(doc.cells)
        assert [c.source for c in flat] == [str(i) for i in range(6)]


# ── the toggle / set-live-action verbs ─────────────────────────────────────────


class TestPresentVerbs:
    def test_toggle_slide_break_toggles_and_emits(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id

        messages.clear()
        h.report_toggle_slide_break(session, None, {"cell_id": cid})
        st = _last_state(messages)
        cell = next(c for c in st["cells"] if c["id"] == cid)
        assert cell["slide_break"] is True

        messages.clear()
        h.report_toggle_slide_break(session, None, {"cell_id": cid})
        cell = next(c for c in _last_state(messages)["cells"] if c["id"] == cid)
        assert cell["slide_break"] is False

    def test_toggle_slide_break_explicit_value(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id
        h.report_toggle_slide_break(session, None, {"cell_id": cid, "value": True})
        assert session._report.doc.cells[0].slide_break is True
        h.report_toggle_slide_break(session, None, {"cell_id": cid, "value": False})
        assert session._report.doc.cells[0].slide_break is False

    def test_set_live_action_and_clear(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id

        messages.clear()
        h.report_set_live_action(session, None, {
            "cell_id": cid, "live_action": {"tutorial": "strain", "guide": "strain"}})
        cell = next(c for c in _last_state(messages)["cells"] if c["id"] == cid)
        assert cell["live_action"] == {"tutorial": "strain", "guide": "strain"}

        # Clear it.
        h.report_set_live_action(session, None, {"cell_id": cid, "live_action": None})
        assert session._report.doc.cells[0].live_action is None

    def test_add_cell_accepts_present_fields_inline(self, window):
        """Deck seeding: report_add_cell can create a cell already-marked."""
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Slide",
            "slide_break": True, "live_action": {"guide": "find-vectors"}})
        cell = session._report.doc.cells[0]
        assert cell.slide_break is True
        assert cell.live_action == {"guide": "find-vectors"}

    def test_toggle_unknown_cell_is_noop(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        h.report_toggle_slide_break(session, None, {"cell_id": "nope"})
        # No crash; no state change emitted for a missing cell.
        assert not _errors(messages)


# ── slides HTML export ─────────────────────────────────────────────────────────


class TestSlidesExport:
    def test_slides_export_shell_and_one_slide_per_group(self, window, tmp_path):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_set_title(session, None, {"title": "My Talk"})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Intro",
            "html": "<h1>Intro</h1>"})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "## Method",
            "html": "<h2>Method</h2>", "slide_break": True})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "## Results",
            "html": "<h2>Results</h2>", "slide_break": True})

        path = str(tmp_path / "deck.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "slides", "path": path})

        exp = _exported(messages)
        assert exp and exp[0]["kind"] == "html-slides"
        assert exp[0]["path"] == path
        assert not _errors(messages)

        html = open(path, encoding="utf-8").read()
        # Deck shell present.
        assert "<title>My Talk</title>" in html
        assert 'id="deck"' in html
        assert 'id="deck-counter"' in html
        # The vanilla-JS switcher (no external CDN).
        assert "ArrowRight" in html and "PageDown" in html
        assert "http://" not in html and "cdn" not in html.lower()
        # One .slide section per slide group (3).
        assert html.count('<section class="slide">') == 3
        # The cells' rendered content rode into the deck.
        assert "<h1>Intro</h1>" in html
        assert "<h2>Method</h2>" in html
        assert "<h2>Results</h2>" in html

    def test_slides_export_multiple_cells_per_slide(self, window, tmp_path):
        """Two markdown cells with NO break between them share ONE slide."""
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Title",
            "html": "<h1>Title</h1>"})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "subtitle body",
            "html": "<p>subtitle body</p>"})
        path = str(tmp_path / "one_slide.html")
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        html = open(path, encoding="utf-8").read()
        # One slide, both cells inside it.
        assert html.count('<section class="slide">') == 1
        assert "<h1>Title</h1>" in html
        assert "<p>subtitle body</p>" in html

    def test_slides_export_with_figure(self, tem_2d_dataset, tmp_path):
        """A figure slide's baked snapshot (or interactive embed) lands in the
        deck — static <img> fallback is fine for this headless path."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# DP", "html": "<h1>DP</h1>"})
        h.report_add_figure(session, None, {
            "source_window_id": wid, "caption": "A pattern"})
        # Mark the figure as its own slide.
        fig_cell = next(c for c in session._report.doc.cells
                        if c.cell_type == "figure")
        h.report_toggle_slide_break(session, None,
                                    {"cell_id": fig_cell.id, "value": True})

        path = str(tmp_path / "fig_deck.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        assert not _errors(messages)
        html = open(path, encoding="utf-8").read()
        assert html.count('<section class="slide">') == 2
        # The figure rode in as an interactive iframe OR a static <img> — either
        # way the figure caption is present.
        assert "A pattern" in html
        assert ("<iframe" in html
                or '<img src="data:image/png;base64,' in html)

    def test_slides_export_no_open_report_errors(self, window):
        session, messages = window["window"], window["messages"]
        ex.report_export_html(session, None, {"mode": "slides", "path": "x.html"})
        assert _errors(messages)

    def test_slides_export_empty_slide_dropped(self, window, tmp_path):
        """A slide whose only cell renders empty (a lone placeholder) is dropped
        rather than shown blank."""
        from spyde.actions.report.model import Cell
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {"template": True})
        mgr = session._report
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Real", "html": "<h1>Real</h1>"})
        # A placeholder figure on its own slide → renders empty → dropped.
        mgr.doc.cells.append(Cell(cell_type="figure", caption="slot",
                                  placeholder=True, slide_break=True))
        path = str(tmp_path / "tpl_deck.html")
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        html = open(path, encoding="utf-8").read()
        # Only the real slide remains.
        assert html.count('<section class="slide">') == 1
        assert "<h1>Real</h1>" in html

"""
test_report_slide_style.py — Report Builder presentation POLISH (title / section
slides + per-slide styling).

Covers the per-slide ``slide_kind`` (``""`` content / ``"title"``) + ``slide_style``
(``""`` default / ``"plain"`` / ``"accent"``) attributes — carried on a slide's
FIRST cell (the slide-break cell) — against a real Qt-free ``Session``:

* ``slide_kind`` / ``slide_style`` round-trip through report.md serialization +
  the zip (markers on the first cell; absent on old files = default; combined with
  slide_break / column so the marker order stays stable),
* ``model.slide_meta`` reads the slide's kind/style off its first cell,
* the ``report_set_slide_kind`` / ``report_set_slide_style`` verbs mutate + emit
  (and apply to the slide's FIRST cell even when fired from a later cell),
* ``report_add_cell`` accepts the fields inline (deck seeding),
* ``report_export_html {mode:'slides'}`` marks a title slide
  (``data-kind="title"`` + the big-title CSS) + a styled slide
  (``slide-style-<preset>``) + confirms a caption renders in the slides output.
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


# ── slide_kind / slide_style model round-trip ──────────────────────────────────


class TestSlideKindStyleRoundTrip:
    def test_slide_kind_round_trips(self):
        doc = m.ReportDoc(title="Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="# Talk title",
                                slide_kind="title"))
        doc.cells.append(m.Cell(cell_type="markdown", source="Body",
                                slide_break=True))
        text = m.serialize_report_md(doc)
        assert "<!-- spyde:slide-kind title -->" in text
        back = m.parse_report_md(text)
        assert [c.slide_kind for c in back.cells] == ["title", ""]
        assert back.cells[0].source == "# Talk title"

    def test_slide_style_round_trips(self):
        doc = m.ReportDoc(title="Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="# S",
                                slide_style="accent"))
        text = m.serialize_report_md(doc)
        assert "<!-- spyde:slide-style accent -->" in text
        back = m.parse_report_md(text)
        assert back.cells[0].slide_style == "accent"

    def test_kind_and_style_and_break_on_one_cell(self):
        """The three slide markers coexist on a slide's first cell in a stable
        order (break, kind, style)."""
        doc = m.ReportDoc(title="Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="Intro"))
        doc.cells.append(m.Cell(cell_type="markdown", source="# Section",
                                slide_break=True, slide_kind="title",
                                slide_style="accent"))
        back = m.parse_report_md(m.serialize_report_md(doc))
        c = back.cells[1]
        assert c.slide_break is True
        assert c.slide_kind == "title"
        assert c.slide_style == "accent"

    def test_kind_with_column_coexists(self):
        """A title cell can also carry a column assignment — both markers parse."""
        doc = m.ReportDoc(title="Deck")
        doc.cells.append(m.Cell(cell_type="markdown", source="Left",
                                slide_kind="title", column="left"))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert back.cells[0].slide_kind == "title"
        assert back.cells[0].column == "left"

    def test_absent_markers_default_on_old_file(self):
        old = ("---\nversion: 1\ntitle: Old\n---\n\n"
               "# Heading\n\n![Cap](assets/cf9.png)\n")
        back = m.parse_report_md(old)
        assert all(c.slide_kind == "" for c in back.cells)
        assert all(c.slide_style == "" for c in back.cells)

    def test_unknown_values_normalize_to_default(self):
        doc = m.ReportDoc()
        # Unknown kind/style values collapse to "" (content / default).
        doc.cells.append(m.Cell(cell_type="markdown", source="A",
                                slide_kind="banner", slide_style="neon"))
        text = m.serialize_report_md(doc)
        # Nothing emitted for the unknown values.
        assert "spyde:slide-kind" not in text
        assert "spyde:slide-style" not in text
        back = m.parse_report_md(text)
        assert back.cells[0].slide_kind == ""
        assert back.cells[0].slide_style == ""

    def test_double_round_trip_stable(self):
        doc = m.ReportDoc(title="Stable")
        doc.created = doc.modified = "2020-01-01T00:00:00+00:00"
        doc.cells.append(m.Cell(cell_type="markdown", source="# Cover",
                                slide_kind="title", slide_style="plain"))
        doc.cells.append(m.Cell(cell_type="markdown", source="B", slide_break=True))
        t1 = m.serialize_report_md(doc)
        back = m.parse_report_md(t1)
        back.created = back.modified = "2020-01-01T00:00:00+00:00"
        t2 = m.serialize_report_md(back)
        assert t1 == t2

    def test_survives_zip(self, tmp_path):
        doc = m.ReportDoc(title="Zipped")
        doc.cells.append(m.Cell(cell_type="markdown", source="# Cover",
                                slide_kind="title", slide_style="accent"))
        doc.cells.append(m.Cell(cell_type="markdown", source="Body",
                                slide_break=True))
        path = str(tmp_path / "deck.spyde-report")
        m.write_report(doc, path)
        back, _assets = m.read_report(path)
        assert back.cells[0].slide_kind == "title"
        assert back.cells[0].slide_style == "accent"
        assert back.cells[1].slide_kind == ""


# ── slide_meta ─────────────────────────────────────────────────────────────────


class TestSlideMeta:
    def test_reads_kind_style_off_first_cell(self):
        cells = [
            m.Cell(cell_type="markdown", source="# Title", slide_kind="title",
                   slide_style="accent"),
            m.Cell(cell_type="markdown", source="not read"),
        ]
        meta = m.slide_meta(cells)
        assert meta == {"kind": "title", "style": "accent"}

    def test_defaults_for_plain_content_slide(self):
        cells = [m.Cell(cell_type="markdown", source="body")]
        assert m.slide_meta(cells) == {"kind": "", "style": ""}

    def test_empty_slide(self):
        assert m.slide_meta([]) == {"kind": "", "style": ""}

    def test_slides_meta_combines_with_grouping(self):
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="# Cover",
                                slide_kind="title"))
        doc.cells.append(m.Cell(cell_type="markdown", source="## Sec",
                                slide_break=True, slide_style="plain"))
        slides = doc.slides()
        assert len(slides) == 2
        assert m.slide_meta(slides[0])["kind"] == "title"
        assert m.slide_meta(slides[1])["style"] == "plain"


# ── the set-kind / set-style verbs ─────────────────────────────────────────────


class TestSlideKindStyleVerbs:
    def test_toggle_slide_kind(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id

        messages.clear()
        h.report_set_slide_kind(session, None, {"cell_id": cid})
        cell = next(c for c in _last_state(messages)["cells"] if c["id"] == cid)
        assert cell["slide_kind"] == "title"

        h.report_set_slide_kind(session, None, {"cell_id": cid})
        assert session._report.doc.cells[0].slide_kind == ""

    def test_set_slide_kind_explicit(self, window):
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id
        h.report_set_slide_kind(session, None, {"cell_id": cid, "slide_kind": "title"})
        assert session._report.doc.cells[0].slide_kind == "title"
        h.report_set_slide_kind(session, None, {"cell_id": cid, "slide_kind": "content"})
        assert session._report.doc.cells[0].slide_kind == ""

    def test_set_slide_kind_applies_to_slide_first_cell(self, window):
        """Firing the verb on a LATER cell of a slide applies the kind to the
        slide's FIRST cell (where the per-slide attribute lives)."""
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "A"})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "B"})
        first_id = session._report.doc.cells[0].id
        later_id = session._report.doc.cells[1].id
        # No break between them → one slide; the later cell is NOT its own start.
        h.report_set_slide_kind(session, None, {"cell_id": later_id, "slide_kind": "title"})
        assert session._report.doc.cells[0].slide_kind == "title"
        assert session._report.doc.cells[1].slide_kind == ""
        # A break makes the later cell start its OWN slide → it takes the kind.
        h.report_toggle_slide_break(session, None, {"cell_id": later_id, "value": True})
        h.report_set_slide_kind(session, None, {"cell_id": later_id, "slide_kind": "title"})
        assert session._report.doc.cells[1].slide_kind == "title"
        # The first slide's kind is unchanged.
        assert session._report.doc.cell_by_id(first_id).slide_kind == "title"

    def test_set_slide_style(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id
        messages.clear()
        h.report_set_slide_style(session, None, {"cell_id": cid, "slide_style": "accent"})
        cell = next(c for c in _last_state(messages)["cells"] if c["id"] == cid)
        assert cell["slide_style"] == "accent"
        # An unknown style clears back to default.
        h.report_set_slide_style(session, None, {"cell_id": cid, "slide_style": "zzz"})
        assert session._report.doc.cells[0].slide_style == ""

    def test_add_cell_accepts_kind_style_inline(self, window):
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Cover",
            "slide_kind": "title", "slide_style": "accent"})
        cell = session._report.doc.cells[0]
        assert cell.slide_kind == "title"
        assert cell.slide_style == "accent"

    def test_unknown_cell_is_noop(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        h.report_set_slide_kind(session, None, {"cell_id": "nope"})
        h.report_set_slide_style(session, None, {"cell_id": "nope", "slide_style": "plain"})
        assert not _errors(messages)


# ── slides HTML export (title slide + styling + caption) ────────────────────────


class TestSlidesExportStyling:
    def test_title_slide_marked_and_big_title_css(self, window, tmp_path):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_set_title(session, None, {"title": "My Talk"})
        # Slide 1: a title slide.
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# The Big Title",
            "html": "<h1>The Big Title</h1>", "slide_kind": "title"})
        # Slide 2: a normal content slide.
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "## Body",
            "html": "<h2>Body</h2>", "slide_break": True})

        path = str(tmp_path / "deck.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        exp = _exported(messages)
        assert exp and exp[0]["kind"] == "html-slides"
        assert not _errors(messages)

        html = open(path, encoding="utf-8").read()
        # The title slide carries data-kind="title"; the content slide does not.
        assert '<section class="slide" data-kind="title">' in html
        assert html.count('<section class="slide">') == 1        # the content slide
        # The big-title CSS rule is present.
        assert '.slide[data-kind="title"] h1' in html
        # The title's rendered content rode in.
        assert "<h1>The Big Title</h1>" in html

    def test_slide_style_preset_class(self, window, tmp_path):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Cover",
            "html": "<h1>Cover</h1>", "slide_style": "accent"})
        path = str(tmp_path / "styled.html")
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        html = open(path, encoding="utf-8").read()
        assert 'class="slide slide-style-accent"' in html
        assert ".slide-style-accent" in html          # the CSS rule

    def test_caption_renders_in_slides(self, tem_2d_dataset, tmp_path):
        """A figure's caption reads on a slide — italic/muted figcaption present."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        h.report_new(session, None, {})
        h.report_add_figure(session, None, {
            "source_window_id": wid, "caption": "Figure 1. A pattern caption"})
        fig_cell = next(c for c in session._report.doc.cells
                        if c.cell_type == "figure")
        h.report_toggle_slide_break(session, None,
                                    {"cell_id": fig_cell.id, "value": True})

        path = str(tmp_path / "cap_deck.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        assert not _errors(messages)
        html = open(path, encoding="utf-8").read()
        # The caption text is present, and the figcaption styling rule too.
        assert "Figure 1. A pattern caption" in html
        assert "figure.report-figure figcaption" in html
        assert "font-style: italic" in html

    def test_default_slide_markup_unchanged(self, window, tmp_path):
        """A deck with NO kind/style markers emits the same plain
        <section class="slide"> markup as before (back-compat)."""
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# A", "html": "<h1>A</h1>"})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# B", "html": "<h1>B</h1>",
            "slide_break": True})
        path = str(tmp_path / "plain.html")
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        html = open(path, encoding="utf-8").read()
        assert html.count('<section class="slide">') == 2
        # No <section> carries a kind attr or a style-preset class (the CSS block
        # still DEFINES the rules — that's expected, so check the section tags).
        import re
        sections = re.findall(r"<section [^>]*>", html)
        assert sections == ['<section class="slide">', '<section class="slide">']

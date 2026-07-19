"""
test_report_slide_notes.py — Report Builder presenter-view SPEAKER NOTES.

Covers the per-slide ``notes`` attribute — free multi-line markdown text carried
on a slide's FIRST cell (like ``slide_kind`` / ``slide_style``), shown only in the
presenter view and NEVER to the audience — against a real Qt-free ``Session``:

* the base64 marker round-trips through report.md serialization + the zip
  (newlines, markdown syntax, ``-->`` sequences, and unicode all preserved;
  absent → "" on old files),
* ``_encode_notes`` / ``_decode_notes`` are inverse + tolerant,
* ``model.slide_meta`` / ``model.slide_notes`` read the notes off the first cell,
* the ``report_set_slide_notes`` verb sets notes on the slide's FIRST cell (even
  when fired from a later cell) + emits ``notes`` in report_state,
* ``report_add_cell`` accepts ``notes`` inline (deck seeding),
* ``report_export_html {mode:'slides'}`` NEVER renders notes as visible slide
  content (only a hidden ``data-notes`` attribute for a future web presenter view).
"""
from __future__ import annotations

import re

import numpy as np

from spyde.actions.report import export_html as ex
from spyde.actions.report import handlers as h
from spyde.actions.report import model as m


# ── helpers (mirror test_report_slide_style) ───────────────────────────────────


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


# ── encode / decode helpers ─────────────────────────────────────────────────────


class TestEncodeDecodeNotes:
    def test_roundtrip_multiline_markdown_unicode(self):
        notes = ("# Say this\n\n- point *one*\n- point two → résumé, Ω, 日本語\n\n"
                 "even a comment-looking token <!-- x --> and an end -->")
        token = m._encode_notes(notes)
        # A single-line, whitespace-free base64 token (comment-safe).
        assert "\n" not in token and " " not in token
        assert m._decode_notes(token) == notes

    def test_empty_and_whitespace_encode_to_empty(self):
        assert m._encode_notes("") == ""
        assert m._encode_notes("   \n  ") == ""
        assert m._encode_notes(None) == ""

    def test_decode_tolerates_garbage(self):
        # A malformed / non-base64 token decodes to "" rather than raising.
        assert m._decode_notes("not!base64!!") == ""
        assert m._decode_notes("") == ""
        assert m._decode_notes(None) == ""


# ── notes model round-trip ───────────────────────────────────────────────────────


class TestNotesRoundTrip:
    def test_notes_round_trip_report_md(self):
        doc = m.ReportDoc(title="Deck")
        notes = "Line one\nLine two with **bold** and $x^2$\nüñî"
        doc.cells.append(m.Cell(cell_type="markdown", source="# Slide",
                                notes=notes))
        text = m.serialize_report_md(doc)
        assert "<!-- spyde:notes " in text
        # The raw multi-line notes text is NOT present verbatim (it's base64'd).
        assert "Line two with" not in text
        back = m.parse_report_md(text)
        assert back.cells[0].notes == notes

    def test_notes_survive_zip(self, tmp_path):
        doc = m.ReportDoc(title="Zipped")
        notes = "Remember to mention the beamstop\n\n- dwell 5s\n- then advance"
        doc.cells.append(m.Cell(cell_type="markdown", source="# Cover",
                                notes=notes, slide_kind="title"))
        doc.cells.append(m.Cell(cell_type="markdown", source="Body",
                                slide_break=True))
        path = str(tmp_path / "deck.spyde-report")
        m.write_report(doc, path)
        back, _assets = m.read_report(path)
        assert back.cells[0].notes == notes
        assert back.cells[1].notes == ""      # no notes on the second slide
        # Coexists with slide_kind on the same first cell.
        assert back.cells[0].slide_kind == "title"

    def test_notes_on_slide_first_cell_only(self):
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="A", notes="my notes"))
        # A slide_break keeps the second cell distinct (an unmarked markdown run
        # would coalesce into the first cell) — notes stay only on the first.
        doc.cells.append(m.Cell(cell_type="markdown", source="B", slide_break=True))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert len(back.cells) == 2
        assert back.cells[0].notes == "my notes"
        assert back.cells[1].notes == ""

    def test_absent_notes_default_on_old_file(self):
        old = ("---\nversion: 1\ntitle: Old\n---\n\n"
               "# Heading\n\n![Cap](assets/cf9.png)\n")
        back = m.parse_report_md(old)
        assert all(c.notes == "" for c in back.cells)

    def test_double_round_trip_stable(self):
        doc = m.ReportDoc(title="Stable")
        doc.created = doc.modified = "2020-01-01T00:00:00+00:00"
        doc.cells.append(m.Cell(cell_type="markdown", source="# Cover",
                                notes="speaker\nnotes\nhere", slide_style="plain"))
        doc.cells.append(m.Cell(cell_type="markdown", source="B", slide_break=True))
        t1 = m.serialize_report_md(doc)
        back = m.parse_report_md(t1)
        back.created = back.modified = "2020-01-01T00:00:00+00:00"
        t2 = m.serialize_report_md(back)
        assert t1 == t2

    def test_notes_coexist_with_all_markers(self):
        """Notes coexist with break + kind + style on one cell."""
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="Intro"))
        doc.cells.append(m.Cell(cell_type="markdown", source="# Section",
                                slide_break=True, slide_kind="title",
                                slide_style="accent",
                                notes="the section notes"))
        back = m.parse_report_md(m.serialize_report_md(doc))
        c = back.cells[1]
        assert c.slide_break is True
        assert c.slide_kind == "title"
        assert c.slide_style == "accent"
        assert c.notes == "the section notes"


# ── slide_meta / slide_notes ─────────────────────────────────────────────────────


class TestSlideNotesAccessors:
    def test_slide_meta_exposes_notes(self):
        cells = [
            m.Cell(cell_type="markdown", source="# Title", notes="hello notes"),
            m.Cell(cell_type="markdown", source="not read", notes="ignored"),
        ]
        meta = m.slide_meta(cells)
        assert meta["notes"] == "hello notes"

    def test_slide_notes_off_first_cell(self):
        cells = [
            m.Cell(cell_type="markdown", source="# Title", notes="first notes"),
            m.Cell(cell_type="markdown", source="body", notes="later, ignored"),
        ]
        assert m.slide_notes(cells) == "first notes"

    def test_slide_notes_empty(self):
        assert m.slide_notes([]) == ""
        assert m.slide_notes([m.Cell(cell_type="markdown", source="x")]) == ""

    def test_slide_notes_via_grouping(self):
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="# Cover",
                                notes="cover notes"))
        doc.cells.append(m.Cell(cell_type="markdown", source="## Sec",
                                slide_break=True, notes="sec notes"))
        slides = doc.slides()
        assert m.slide_notes(slides[0]) == "cover notes"
        assert m.slide_notes(slides[1]) == "sec notes"


# ── the set-notes verb ───────────────────────────────────────────────────────────


class TestSetSlideNotesVerb:
    def test_set_notes_emits(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id
        messages.clear()
        h.report_set_slide_notes(session, None,
                                 {"cell_id": cid, "notes": "my private notes"})
        cell = next(c for c in _last_state(messages)["cells"] if c["id"] == cid)
        assert cell["notes"] == "my private notes"
        assert session._report.doc.cells[0].notes == "my private notes"

    def test_clear_notes(self, window):
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        cid = session._report.doc.cells[0].id
        h.report_set_slide_notes(session, None, {"cell_id": cid, "notes": "abc"})
        assert session._report.doc.cells[0].notes == "abc"
        h.report_set_slide_notes(session, None, {"cell_id": cid, "notes": ""})
        assert session._report.doc.cells[0].notes == ""
        # A missing notes key also clears.
        h.report_set_slide_notes(session, None, {"cell_id": cid, "notes": "abc"})
        h.report_set_slide_notes(session, None, {"cell_id": cid})
        assert session._report.doc.cells[0].notes == ""

    def test_notes_applies_to_slide_first_cell(self, window):
        """Firing the verb on a LATER cell of a slide applies the notes to the
        slide's FIRST cell (where the per-slide attribute lives)."""
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "A"})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "B"})
        first_id = session._report.doc.cells[0].id
        later_id = session._report.doc.cells[1].id
        # No break between → one slide; the later cell isn't its own start.
        h.report_set_slide_notes(session, None,
                                 {"cell_id": later_id, "notes": "slide-1 notes"})
        assert session._report.doc.cells[0].notes == "slide-1 notes"
        assert session._report.doc.cells[1].notes == ""
        # A break makes the later cell start its OWN slide → it takes the notes.
        h.report_toggle_slide_break(session, None,
                                    {"cell_id": later_id, "value": True})
        h.report_set_slide_notes(session, None,
                                 {"cell_id": later_id, "notes": "slide-2 notes"})
        assert session._report.doc.cells[1].notes == "slide-2 notes"
        assert session._report.doc.cell_by_id(first_id).notes == "slide-1 notes"

    def test_add_cell_accepts_notes_inline(self, window):
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Cover",
            "slide_kind": "title", "notes": "seeded notes"})
        cell = session._report.doc.cells[0]
        assert cell.notes == "seeded notes"
        assert cell.slide_kind == "title"

    def test_unknown_cell_is_noop(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        h.report_set_slide_notes(session, None, {"cell_id": "nope", "notes": "x"})
        assert not _errors(messages)

    def test_state_emits_notes_field(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "X"})
        st = _last_state(messages)
        # Every cell carries a notes field (default "").
        assert all("notes" in c for c in st["cells"])
        assert st["cells"][0]["notes"] == ""


# ── slides HTML export: notes NOT audience-visible ───────────────────────────────


class TestNotesInvisibleInExport:
    def test_notes_not_rendered_as_visible_content(self, window, tmp_path):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        secret = "SECRET_SPEAKER_NOTE_DO_NOT_SHOW"
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# The Big Title",
            "html": "<h1>The Big Title</h1>", "notes": secret})
        path = str(tmp_path / "deck.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        exp = _exported(messages)
        assert exp and exp[0]["kind"] == "html-slides"
        assert not _errors(messages)

        html = open(path, encoding="utf-8").read()
        # The visible slide BODY (inside .slide-inner) must NOT contain the notes.
        # The notes may appear ONLY inside a hidden data-notes="..." attribute.
        # Strip the data-notes attribute values, then assert the secret is gone.
        without_attr = re.sub(r'data-notes="[^"]*"', 'data-notes=""', html)
        assert secret not in without_attr, "speaker notes leaked into the audience deck"
        # The title still rendered normally.
        assert "<h1>The Big Title</h1>" in html

    def test_notes_stashed_in_hidden_data_attr(self, window, tmp_path):
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Cover",
            "html": "<h1>Cover</h1>", "notes": "for a future web presenter view"})
        path = str(tmp_path / "deck2.html")
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        html = open(path, encoding="utf-8").read()
        # The (escaped) notes ride ONLY in the section's data-notes attribute.
        assert 'data-notes="for a future web presenter view"' in html

    def test_no_notes_no_attr(self, window, tmp_path):
        """A notes-free deck's section markup is unchanged (no data-notes)."""
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# A", "html": "<h1>A</h1>"})
        path = str(tmp_path / "plain.html")
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        html = open(path, encoding="utf-8").read()
        assert "data-notes" not in html
        assert '<section class="slide">' in html

    def test_notes_escaped_in_attr(self, window, tmp_path):
        """Notes with quotes / angle brackets are HTML-escaped in the attribute
        so they can't break out or inject markup."""
        session, _ = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# A", "html": "<h1>A</h1>",
            "notes": 'say "hi" & <b>note</b>'})
        path = str(tmp_path / "esc.html")
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        html = open(path, encoding="utf-8").read()
        # No raw unescaped angle bracket / quote inside the attribute value.
        assert "&quot;hi&quot;" in html
        assert "&lt;b&gt;note&lt;/b&gt;" in html

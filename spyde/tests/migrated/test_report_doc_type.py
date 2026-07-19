"""
test_report_doc_type.py — Wave A: the report/presentation ``doc_type`` field.

``ReportDoc.doc_type`` ("report" | "presentation" | "movie") mirrors ``template``:
it serializes as a front-matter ``type:`` key, reads back tolerantly (absent →
"report"), rides through the ``.spyde-report`` zip, is set by ``report_new
{type}``, and is emitted in ``report_state`` as ``type``. SCHEMA_VERSION stays 1,
so an older file with no ``type:`` loads as a plain report.
"""
from __future__ import annotations

import os
import tempfile

from spyde.actions.report import handlers as h
from spyde.actions.report.model import (
    Cell, ReportDoc, parse_report_md, read_report, serialize_report_md,
    write_report,
)


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


class TestDocTypeModel:
    def test_front_matter_round_trip_presentation(self):
        doc = ReportDoc(title="Deck", doc_type="presentation")
        doc.cells = [Cell(cell_type="markdown", source="hi")]
        md = serialize_report_md(doc)
        assert "type: presentation" in md
        doc2 = parse_report_md(md)
        assert doc2.doc_type == "presentation"

    def test_front_matter_round_trip_report(self):
        doc = ReportDoc(title="Article", doc_type="report")
        md = serialize_report_md(doc)
        assert parse_report_md(md).doc_type == "report"

    def test_absent_type_defaults_to_report(self):
        # A legacy report.md with NO ``type:`` front-matter loads as "report".
        legacy = "---\nversion: 1\ntitle: Old\ntemplate: false\n---\nbody\n"
        assert parse_report_md(legacy).doc_type == "report"

    def test_unknown_type_normalizes_to_report(self):
        weird = "---\nversion: 1\ntitle: X\ntype: banana\n---\nbody\n"
        assert parse_report_md(weird).doc_type == "report"

    def test_movie_type_tolerated(self):
        doc = ReportDoc(doc_type="movie")
        assert parse_report_md(serialize_report_md(doc)).doc_type == "movie"

    def test_zip_round_trip(self):
        doc = ReportDoc(title="Deck", doc_type="presentation")
        doc.cells = [Cell(cell_type="markdown", source="one")]
        tmp = os.path.join(tempfile.gettempdir(), "doctype_rt.spyde-report")
        try:
            write_report(doc, tmp)
            doc2, _assets = read_report(tmp)
            assert doc2.doc_type == "presentation"
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


class TestDocTypeHandlers:
    def test_report_new_sets_type(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {"type": "presentation"})
        st = _last_state(messages)
        assert st["type"] == "presentation"

    def test_report_new_default_type_is_report(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        assert _last_state(messages)["type"] == "report"

    def test_report_new_unknown_type_normalizes(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {"type": "nonsense"})
        assert _last_state(messages)["type"] == "report"

    def test_state_emits_type(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {"type": "presentation"})
        mgr = h._manager(session)
        assert mgr.state()["type"] == "presentation"

    def test_closed_state_has_report_type(self, window):
        session = window["window"]
        mgr = h._manager(session)
        # A closed manager reports the default type (open=False).
        st = mgr.state()
        assert st["open"] is False
        assert st["type"] == "report"

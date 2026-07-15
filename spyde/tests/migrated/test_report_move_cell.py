"""
test_report_move_cell.py — off-by-one regression guard for ``report_move_cell``.

The renderer sends ``index`` as the drop TARGET cell's own pre-removal array
position (see ``ReportSidebar.tsx`` ``cells.map((cell, i) => ... index={i}``
feeding ``makeDragProps``), i.e. "insert before the cell currently at this
index". ``report_move_cell`` used to pop the dragged cell FIRST and then use
``index`` unmodified against the now-shorter (and shifted) list, so a forward
drag (dragging a cell to a LATER position) landed one slot later than the
drop target: dragging A onto C in [A,B,C] produced [B,C,A] instead of the
intended "insert A right before C" -> [B,A,C].

This file is intentionally separate from ``test_report_handlers.py`` (owned by
another concurrent change) to avoid touching shared test infrastructure.
"""
from __future__ import annotations

from spyde.actions.report import handlers as h


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _sources(messages):
    return [c["source"] for c in _last_state(messages)["cells"]]


def _seed_abc(session, messages):
    """New report with three markdown cells A, B, C (in that order); returns
    the cell ids in list order."""
    h.report_new(session, None, {})
    for s in ("A", "B", "C"):
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": s})
    return [c["id"] for c in _last_state(messages)["cells"]]


class TestReportMoveCellOffByOne:
    def test_forward_drag_lands_before_target_not_after(self, window):
        """[A,B,C], drag A (index 0) onto C (pre-removal index 2) -> [B,A,C].

        Before the fix this produced [B,C,A] (A landed one slot too late)."""
        session, messages = window["window"], window["messages"]
        a_id, _b_id, _c_id = _seed_abc(session, messages)
        h.report_move_cell(session, None, {"cell_id": a_id, "index": 2})
        assert _sources(messages) == ["B", "A", "C"]

    def test_forward_drag_onto_immediate_next_neighbor_is_noop(self, window):
        """[A,B,C], drag A (index 0) onto B (pre-removal index 1): A is
        already immediately before B, so nothing should move."""
        session, messages = window["window"], window["messages"]
        a_id, _b_id, _c_id = _seed_abc(session, messages)
        h.report_move_cell(session, None, {"cell_id": a_id, "index": 1})
        assert _sources(messages) == ["A", "B", "C"]

    def test_backward_drag_lands_before_target(self, window):
        """[A,B,C], drag C (index 2) onto A (pre-removal index 0) -> [C,A,B].

        Backward drags never went through the pop-then-insert shift, so this
        pins that the fix doesn't disturb the already-correct case."""
        session, messages = window["window"], window["messages"]
        _a_id, _b_id, c_id = _seed_abc(session, messages)
        h.report_move_cell(session, None, {"cell_id": c_id, "index": 0})
        assert _sources(messages) == ["C", "A", "B"]

    def test_same_position_is_noop(self, window):
        """Dragging a cell and dropping it back on itself changes nothing."""
        session, messages = window["window"], window["messages"]
        _a_id, b_id, _c_id = _seed_abc(session, messages)
        h.report_move_cell(session, None, {"cell_id": b_id, "index": 1})
        assert _sources(messages) == ["A", "B", "C"]

    def test_drag_to_end_no_index(self, window):
        """``index: None`` (drop past the last cell) appends at the end."""
        session, messages = window["window"], window["messages"]
        a_id, _b_id, _c_id = _seed_abc(session, messages)
        h.report_move_cell(session, None, {"cell_id": a_id, "index": None})
        assert _sources(messages) == ["B", "C", "A"]

    def test_unknown_cell_id_is_noop(self, window):
        session, messages = window["window"], window["messages"]
        _seed_abc(session, messages)
        before = _sources(messages)
        h.report_move_cell(session, None, {"cell_id": "not-a-real-id", "index": 1})
        assert _sources(messages) == before

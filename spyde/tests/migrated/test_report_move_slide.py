"""
test_report_move_slide.py — the ``report_move_slide`` verb + the ``move_slide``
pure helper (the Slide Overview grid's drag-reorder of WHOLE slides).

A slide is a contiguous RUN of cells (a cell with ``slide_break=True`` STARTS a
slide; cells accumulate until the next break — see ``ReportDoc.slides``). Moving
a slide means splicing that whole cell-run to a new position in ``doc.cells``
while keeping the slide GROUPING intact (the slide_break invariant). These tests
pin:

* the cell order after a block move (both forward + backward),
* that ``slides()`` regroups into the SAME slides just reordered,
* the slide_break invariant at the boundaries (the displaced formerly-first
  slide gains a break; the new first slide's leading break is a harmless no-op),
* to/from the first slide keeps grouping valid,
* out-of-range / no-op indices leave the deck unchanged.
"""
from __future__ import annotations

from spyde.actions.report import handlers as h
from spyde.actions.report.model import Cell, ReportDoc, move_slide


# ── pure helper (move_slide) ──────────────────────────────────────────────────


def _mk(cells_spec):
    """Build a flat cell list from ``[(source, slide_break), ...]``."""
    return [Cell(id=f"c{i}", cell_type="markdown", source=src, slide_break=brk)
            for i, (src, brk) in enumerate(cells_spec)]


def _slides_sources(cells):
    """Group ``cells`` into slides (via ReportDoc.slides) and return each slide's
    cell sources as a list-of-lists."""
    doc = ReportDoc(cells=list(cells))
    return [[c.source for c in slide] for slide in doc.slides()]


def _breaks(cells):
    return [c.slide_break for c in cells]


class TestMoveSlideHelper:
    def test_three_slides_move_last_to_first(self):
        # 3 slides: [A],[B],[C]. Move slide 2 (C) to position 0.
        cells = _mk([("A", True), ("B", True), ("C", True)])
        # Normalise the FIRST cell's break to False (as a real deck has) to prove
        # the helper fixes the invariant regardless of the source flag state.
        cells[0].slide_break = False
        out = move_slide(cells, 2, 0)
        assert [c.source for c in out] == ["C", "A", "B"]
        # slides() regroups into [C],[A],[B] — same slides, reordered.
        assert _slides_sources(out) == [["C"], ["A"], ["B"]]
        # New first slide (C) leading-break is a no-op → False; the displaced
        # A + B must now carry breaks so they stay distinct slides.
        assert _breaks(out) == [False, True, True]

    def test_move_first_slide_to_last(self):
        cells = _mk([("A", False), ("B", True), ("C", True)])
        out = move_slide(cells, 0, 2)
        assert [c.source for c in out] == ["B", "C", "A"]
        assert _slides_sources(out) == [["B"], ["C"], ["A"]]
        # B is now first (break False), C + A distinct slides (break True).
        assert _breaks(out) == [False, True, True]

    def test_move_multicell_slide_block(self):
        # Slide 0 = [A1, A2], slide 1 = [B], slide 2 = [C1, C2].
        cells = _mk([("A1", False), ("A2", False),
                     ("B", True),
                     ("C1", True), ("C2", False)])
        # Move slide 2 (C1,C2) to position 1 → [A1,A2],[C1,C2],[B].
        out = move_slide(cells, 2, 1)
        assert [c.source for c in out] == ["A1", "A2", "C1", "C2", "B"]
        assert _slides_sources(out) == [["A1", "A2"], ["C1", "C2"], ["B"]]
        # Slide starts: A1 (first, no break), C1 (break), B (break). The
        # non-start cells (A2, C2) keep break False.
        assert _breaks(out) == [False, False, True, False, True]

    def test_move_middle_forward(self):
        # 4 slides, move slide 1 (B) to position 3 (last).
        cells = _mk([("A", False), ("B", True), ("C", True), ("D", True)])
        out = move_slide(cells, 1, 3)
        assert [c.source for c in out] == ["A", "C", "D", "B"]
        assert _slides_sources(out) == [["A"], ["C"], ["D"], ["B"]]

    def test_noop_same_index(self):
        cells = _mk([("A", False), ("B", True), ("C", True)])
        out = move_slide(cells, 1, 1)
        assert [c.source for c in out] == ["A", "B", "C"]

    def test_out_of_range_indices(self):
        cells = _mk([("A", False), ("B", True)])
        assert [c.source for c in move_slide(cells, 5, 0)] == ["A", "B"]
        assert [c.source for c in move_slide(cells, 0, 9)] == ["A", "B"]
        assert [c.source for c in move_slide(cells, -1, 0)] == ["A", "B"]

    def test_input_list_not_mutated_in_place(self):
        cells = _mk([("A", False), ("B", True), ("C", True)])
        original_ids = [id(c) for c in cells]
        move_slide(cells, 2, 0)
        # The input LIST object is unchanged (a new list is returned); the same
        # Cell objects are reused (only their break flags may be normalised).
        assert [id(c) for c in cells] == original_ids

    def test_empty(self):
        assert move_slide([], 0, 0) == []


# ── the verb (report_move_slide via the manager) ──────────────────────────────


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_cells(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]["cells"]


def _seed_three_slides(session, messages):
    """New report with three 1-cell slides titled A, B, C (each its own slide)."""
    h.report_new(session, None, {})
    h.report_add_cell(session, None, {"cell_type": "markdown", "source": "A"})
    h.report_add_cell(session, None,
                      {"cell_type": "markdown", "source": "B", "slide_break": True})
    h.report_add_cell(session, None,
                      {"cell_type": "markdown", "source": "C", "slide_break": True})


class TestReportMoveSlideVerb:
    def test_move_slide_three_to_one(self, window):
        session, messages = window["window"], window["messages"]
        _seed_three_slides(session, messages)
        # Move slide index 2 (C) to slide position 0.
        h.report_move_slide(session, None, {"from": 2, "to": 0})
        cells = _last_cells(messages)
        assert [c["source"] for c in cells] == ["C", "A", "B"]
        # Grouping preserved: [C],[A],[B].
        assert [c["slide_break"] for c in cells] == [False, True, True]

    def test_move_slide_emits_state_and_marks_dirty(self, window):
        session, messages = window["window"], window["messages"]
        _seed_three_slides(session, messages)
        n_before = len(_states(messages))
        h.report_move_slide(session, None, {"from": 0, "to": 2})
        assert len(_states(messages)) == n_before + 1
        assert _states(messages)[-1]["report"]["dirty"] is True
        assert [c["source"] for c in _last_cells(messages)] == ["B", "C", "A"]

    def test_move_slide_out_of_range_noop(self, window):
        session, messages = window["window"], window["messages"]
        _seed_three_slides(session, messages)
        before = [c["source"] for c in _last_cells(messages)]
        h.report_move_slide(session, None, {"from": 9, "to": 0})
        assert [c["source"] for c in _last_cells(messages)] == before

    def test_move_slide_missing_indices_noop(self, window):
        session, messages = window["window"], window["messages"]
        _seed_three_slides(session, messages)
        before = [c["source"] for c in _last_cells(messages)]
        h.report_move_slide(session, None, {"from": 1})
        h.report_move_slide(session, None, {})
        assert [c["source"] for c in _last_cells(messages)] == before

    def test_move_multicell_slides_via_verb(self, window):
        """A slide with 2 cells moves as a BLOCK; grouping stays exact."""
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "A1"})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "A2"})
        h.report_add_cell(session, None,
                          {"cell_type": "markdown", "source": "B", "slide_break": True})
        h.report_add_cell(session, None,
                          {"cell_type": "markdown", "source": "C1", "slide_break": True})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "C2"})
        # Slides: [A1,A2],[B],[C1,C2]. Move slide 2 → position 0.
        h.report_move_slide(session, None, {"from": 2, "to": 0})
        cells = _last_cells(messages)
        assert [c["source"] for c in cells] == ["C1", "C2", "A1", "A2", "B"]
        assert [c["slide_break"] for c in cells] == [False, False, True, False, True]

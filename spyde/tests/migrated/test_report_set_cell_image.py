"""
test_report_set_cell_image.py — filling an EXISTING cell's figure slot with a
dropped/pasted PHOTO.

The bug this pins: dropping a PNG onto a split cell's empty figure side (or onto
an empty figure placeholder) had no verb that could fill the slot. The renderer's
drop zones only recognised a figure/window PILL, so a file drag failed their
test, bubbled up to the sidebar body, and was APPENDED as a brand-new image cell
BELOW — the slot the user aimed at stayed empty and the picture landed somewhere
else on the page.

``report_set_cell_image`` is the image counterpart of ``report_set_split_figure``:

* a SPLIT cell takes the photo as its figure side, TEXT and layout untouched,
* a FIGURE PLACEHOLDER converts IN PLACE to an ``image`` cell — same cell id, so
  the slide attributes riding on a slide's first cell (break / kind / style /
  speaker notes) survive; re-creating the cell would silently lose them,
* an already-filled slot is REPLACED (the prior figure's snapshot is dropped),
* a cell with no figure slot at all (markdown) is a no-op with an error,
* the shared decode path still caps size and normalises the extension.
"""
from __future__ import annotations

import base64

from spyde.actions.report import handlers as h


# A 1×1 red PNG (67 bytes) — the smallest real PNG, so the bytes round-trip
# through a real data URL without any faking.
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhV"
    "AAAAAElFTkSuQmCC")
_GIF_1x1 = base64.b64decode(
    "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")

_PNG_URL = "data:image/png;base64," + base64.b64encode(_PNG_1x1).decode("ascii")
_GIF_URL = "data:image/gif;base64," + base64.b64encode(_GIF_1x1).decode("ascii")


def _states(messages):
    return [msg for msg in messages if msg.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _errors(messages):
    return [msg for msg in messages if msg.get("type") == "error"]


class TestSetCellImageOnSplit:
    def test_fills_the_split_figure_side_in_place(self, window):
        """The photo lands in the EXISTING cell — no new cell is appended."""
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {
            "source": "## Motivation\n\n- a bullet\n", "layout": "text-left"})
        cell_id = session._report.doc.cells[0].id
        messages.clear()

        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "png"})
        assert not _errors(messages)

        cells = _last_state(messages)["cells"]
        # THE assertion: still ONE cell. The old path produced two.
        assert len(cells) == 1
        assert cells[0]["id"] == cell_id
        assert cells[0]["cell_type"] == "split"
        assert cells[0]["image"].startswith("data:image/png;base64,")
        assert session._report._images[cell_id] == _PNG_1x1

    def test_text_side_and_layout_are_untouched(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {
            "source": "## Keep me\n", "layout": "text-bottom"})
        cell_id = session._report.doc.cells[0].id
        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "png"})

        cell = session._report.doc.cells[0]
        assert cell.source == "## Keep me\n"
        assert cell.split_layout == "text-bottom"

    def test_caption_is_optional_and_applied_when_given(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x", "caption": "before"})
        cell_id = session._report.doc.cells[0].id

        # No caption in the payload → the existing one survives.
        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "png"})
        assert session._report.doc.cells[0].caption == "before"

        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "png",
            "caption": "after"})
        assert session._report.doc.cells[0].caption == "after"

    def test_replacing_a_photo_swaps_the_bytes(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        cell_id = session._report.doc.cells[0].id
        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "png"})
        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _GIF_URL, "image_ext": "gif"})

        assert session._report.doc.cells[0].image_ext == "gif"
        assert session._report._images[cell_id] == _GIF_1x1
        assert len(session._report.doc.cells) == 1

    def test_photo_clears_a_figure_previously_in_the_slot(self, window):
        """A slot holds a figure OR a photo, never both — the mirror of
        report_set_split_figure clearing a prior photo."""
        session = window["window"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        cell = session._report.doc.cells[0]
        # Stand in for a filled figure side without needing a real plot window.
        cell.spec = object()
        session._report._snapshots[cell.id] = {"panel": b""}

        h.report_set_cell_image(session, None, {
            "cell_id": cell.id, "image_b64": _PNG_URL, "image_ext": "png"})

        assert cell.spec is None
        assert cell.id not in session._report._snapshots
        assert session._report._images[cell.id] == _PNG_1x1


class TestSetCellImageOnPlaceholder:
    def test_placeholder_becomes_an_image_cell_with_the_same_id(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_figure_placeholder(session, None, {})
        cell_id = session._report.doc.cells[0].id
        messages.clear()

        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "png"})
        assert not _errors(messages)

        cells = _last_state(messages)["cells"]
        assert len(cells) == 1
        assert cells[0]["id"] == cell_id          # SAME cell, converted in place
        assert cells[0]["cell_type"] == "image"
        assert not session._report.doc.cells[0].placeholder

    def test_slide_attributes_survive_the_conversion(self, window):
        """The regression a re-created cell would cause: a slide's first cell
        carries the break and the speaker notes, so losing the cell merges the
        slide into the previous one and drops the notes."""
        session = window["window"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"source": "slide one"})
        h.report_add_figure_placeholder(session, None, {"slide_break": True})
        cell = session._report.doc.cells[1]
        cell.slide_kind, cell.slide_style = "", "accent"
        cell.notes = "remember to breathe"

        h.report_set_cell_image(session, None, {
            "cell_id": cell.id, "image_b64": _PNG_URL, "image_ext": "png"})

        after = session._report.doc.cells[1]
        assert after.slide_break is True
        assert after.slide_style == "accent"
        assert after.notes == "remember to breathe"
        assert len(session._report.doc.slides()) == 2


class TestReplaceAnExistingPhoto:
    """Drop another image onto a picture that is already there."""

    def test_image_cell_bytes_are_swapped_in_place(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_URL, "image_ext": "png", "caption": "keep me"})
        cell_id = session._report.doc.cells[0].id
        messages.clear()

        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _GIF_URL, "image_ext": "gif"})
        assert not _errors(messages)

        cells = _last_state(messages)["cells"]
        assert len(cells) == 1, "replacing must not append a second picture"
        assert cells[0]["id"] == cell_id
        assert cells[0]["cell_type"] == "image"
        assert cells[0]["image_ext"] == "gif"
        assert session._report._images[cell_id] == _GIF_1x1
        # The identity that hangs off the cell survives the swap.
        assert cells[0]["caption"] == "keep me"

    def test_slide_attributes_survive_an_image_swap(self, window):
        session = window["window"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"source": "slide one"})
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_URL, "image_ext": "png", "slide_break": True})
        cell = session._report.doc.cells[1]
        cell.notes = "say the thing"

        h.report_set_cell_image(session, None, {
            "cell_id": cell.id, "image_b64": _GIF_URL, "image_ext": "gif"})

        after = session._report.doc.cells[1]
        assert after.slide_break is True
        assert after.notes == "say the thing"
        assert len(session._report.doc.slides()) == 2


class TestTextSlideBecomesASplit:
    """Drop a picture on a TEXT slide and it becomes a SPLIT slide."""

    def test_markdown_converts_in_place_and_keeps_its_prose(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"source": "## Motivation\n\n- a point\n"})
        cell_id = session._report.doc.cells[0].id
        messages.clear()

        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "png"})
        assert not _errors(messages)

        cells = _last_state(messages)["cells"]
        assert len(cells) == 1, "a layout change must not add a second cell"
        cell = session._report.doc.cells[0]
        assert cell.id == cell_id
        assert cell.cell_type == "split"
        assert cell.source == "## Motivation\n\n- a point\n"   # prose preserved
        assert cell.image_ext == "png"
        assert session._report._images[cell_id] == _PNG_1x1
        assert cell.split_layout == "text-left"               # sane default

    def test_the_layout_can_be_chosen_at_conversion(self, window):
        session = window["window"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"source": "text"})
        cell_id = session._report.doc.cells[0].id
        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "png",
            "layout": "text-top"})
        assert session._report.doc.cells[0].split_layout == "text-top"

    def test_the_slide_stays_one_slide(self, window):
        """The conversion is a LAYOUT change, so a slide that was one text block
        must not become two slides — and its notes must survive."""
        session = window["window"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"source": "slide one"})
        h.report_add_cell(session, None, {"source": "## Two\n", "slide_break": True})
        cell = session._report.doc.cells[1]
        cell.notes = "remember the aside"
        cell.slide_style = "accent"

        h.report_set_cell_image(session, None, {
            "cell_id": cell.id, "image_b64": _PNG_URL, "image_ext": "png"})

        after = session._report.doc.cells[1]
        assert after.slide_break is True
        assert after.notes == "remember the aside"
        assert after.slide_style == "accent"
        assert len(session._report.doc.slides()) == 2


class TestSetCellImageRejects:
    def test_a_movie_cell_has_no_slot(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"source": "x"})
        cell = session._report.doc.cells[0]
        cell.cell_type = "movie"
        messages.clear()

        h.report_set_cell_image(session, None, {
            "cell_id": cell.id, "image_b64": _PNG_URL, "image_ext": "png"})

        assert _errors(messages), "a slot-less cell must report why, not no-op"
        assert session._report.doc.cells[0].cell_type == "movie"

    def test_unknown_cell_id_errors(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        h.report_set_cell_image(session, None, {
            "cell_id": "nope", "image_b64": _PNG_URL, "image_ext": "png"})
        assert _errors(messages)

    def test_undecodable_payload_errors_and_leaves_the_slot_empty(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        cell_id = session._report.doc.cells[0].id
        messages.clear()

        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": "", "image_ext": "png"})

        assert _errors(messages)
        assert cell_id not in session._report._images

    def test_oversized_image_is_refused(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        cell_id = session._report.doc.cells[0].id
        big = base64.b64encode(b"\x00" * (h._IMAGE_CELL_MAX_BYTES + 1)).decode("ascii")
        messages.clear()

        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": big, "image_ext": "png"})

        assert _errors(messages)
        assert cell_id not in session._report._images

    def test_unknown_extension_normalises_to_png(self, window):
        session = window["window"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        cell_id = session._report.doc.cells[0].id
        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "tiff"})
        assert session._report.doc.cells[0].image_ext == "png"

    def test_jpeg_normalises_to_jpg(self, window):
        session = window["window"]
        h.report_new(session, None, {})
        h.report_add_split_cell(session, None, {"source": "x"})
        cell_id = session._report.doc.cells[0].id
        h.report_set_cell_image(session, None, {
            "cell_id": cell_id, "image_b64": _PNG_URL, "image_ext": "jpeg"})
        assert session._report.doc.cells[0].image_ext == "jpg"


class TestSetCellImageIsRegistered:
    def test_the_verb_dispatches(self, window):
        """The handler is reachable through the action registry — the renderer
        calls it by name, so an unregistered verb is a silent dead drop."""
        from spyde.actions import registry
        assert "report_set_cell_image" in registry.STAGED_HANDLERS

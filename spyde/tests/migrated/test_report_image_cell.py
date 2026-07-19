"""
test_report_image_cell.py — Report Builder PHOTO/IMAGE cells.

A user can drop / paste / browse an image into a report; it becomes a
``cell_type="image"`` cell whose raw bytes are stored as an asset and rendered
inline (with a caption). Exercises against a real Qt-free ``Session``:

* ``report_add_image_cell`` creates an image cell, holds the bytes, and emits the
  data URL in report_state,
* the size cap rejects an oversized image,
* report.md serialize/parse distinguishes an IMAGE cell (``assets/<id>.<ext>``,
  NO figures yaml) from a FIGURE cell (``assets/<id>.png`` WITH a yaml) — and a
  bare ``.png`` image (a pasted screenshot) round-trips as an image cell via the
  yaml-presence promotion in read_report,
* save to a ``.spyde-report`` zip + reopen restores the image bytes,
* the caption edits through report_update_cell / report_set_caption,
* ``report_export_html {mode:'static'|'slides'}`` inlines the image data URL.
"""
from __future__ import annotations

import base64

from spyde.actions.report import export_html as ex
from spyde.actions.report import handlers as h
from spyde.actions.report import model as m


# ── tiny valid image fixtures ──────────────────────────────────────────────────

# A 1×1 red PNG (67 bytes). The smallest real PNG so the bytes round-trip through
# a real zip / data URL without any faking.
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhV"
    "AAAAAElFTkSuQmCC")
# A tiny GIF (a 1×1 transparent GIF, 43 bytes) to prove the non-png ext path.
_GIF_1x1 = base64.b64decode(
    "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")

_PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(_PNG_1x1).decode("ascii")
_GIF_DATA_URL = "data:image/gif;base64," + base64.b64encode(_GIF_1x1).decode("ascii")


# ── helpers ────────────────────────────────────────────────────────────────────


def _states(messages):
    return [msg for msg in messages if msg.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _errors(messages):
    return [msg for msg in messages if msg.get("type") == "error"]


def _exported(messages):
    return [msg for msg in messages if msg.get("type") == "report_exported"]


# ── report_add_image_cell handler ──────────────────────────────────────────────


class TestAddImageCell:
    def test_add_image_cell_creates_cell_and_emits_data_url(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_DATA_URL, "image_ext": "png", "caption": "My photo"})
        assert not _errors(messages)

        st = _last_state(messages)
        assert len(st["cells"]) == 1
        cell = st["cells"][0]
        assert cell["cell_type"] == "image"
        assert cell["caption"] == "My photo"
        assert cell["image_ext"] == "png"
        # The bytes ride back as a data URL so the renderer draws the <img>.
        assert cell["image"].startswith("data:image/png;base64,")
        assert base64.b64decode(cell["image"].split(",", 1)[1]) == _PNG_1x1

        # Held on the manager keyed by the cell id.
        doc_cell = session._report.doc.cells[0]
        assert session._report._images[doc_cell.id] == _PNG_1x1

    def test_add_image_cell_accepts_bare_base64(self, window):
        """The handler decodes a bare base64 payload too (no data: prefix)."""
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        raw = base64.b64encode(_PNG_1x1).decode("ascii")
        h.report_add_image_cell(session, None, {"image_b64": raw, "image_ext": "png"})
        assert session._report.doc.cells[0].cell_type == "image"

    def test_add_image_cell_gif_ext_and_mime(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        h.report_add_image_cell(session, None, {
            "image_b64": _GIF_DATA_URL, "image_ext": "gif", "caption": "anim"})
        cell = _last_state(messages)["cells"][0]
        assert cell["image_ext"] == "gif"
        assert cell["image"].startswith("data:image/gif;base64,")

    def test_add_image_cell_jpeg_normalises_to_jpg(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_DATA_URL, "image_ext": "jpeg"})
        # jpeg → jpg for the asset ext, but the emitted MIME stays image/jpeg.
        assert session._report.doc.cells[0].image_ext == "jpg"

    def test_add_image_cell_unknown_ext_falls_back_to_png(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_DATA_URL, "image_ext": "tiff"})
        assert session._report.doc.cells[0].image_ext == "png"

    def test_add_image_cell_at_index(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "top"})
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_DATA_URL, "image_ext": "png", "index": 0})
        types = [c.cell_type for c in session._report.doc.cells]
        assert types == ["image", "markdown"]

    def test_add_image_cell_no_data_errors(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        h.report_add_image_cell(session, None, {"image_b64": "", "image_ext": "png"})
        assert _errors(messages)
        assert not session._report.doc.cells

    def test_size_cap_rejects_oversized_image(self, window, monkeypatch):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        # Shrink the cap so we don't have to build a 10 MB payload in the test.
        monkeypatch.setattr(h, "_IMAGE_CELL_MAX_BYTES", 32)
        big = base64.b64encode(b"x" * 64).decode("ascii")
        messages.clear()
        h.report_add_image_cell(session, None, {"image_b64": big, "image_ext": "png"})
        assert _errors(messages)
        assert "too large" in _errors(messages)[0]["text"].lower()
        assert not session._report.doc.cells


# ── serialization: image cell distinct from figure cell ─────────────────────────


class TestImageCellSerialization:
    def test_image_cell_serializes_with_ext_no_yaml(self):
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(id="cimg0001", cell_type="image",
                                caption="A gif", image_ext="gif"))
        text = m.serialize_report_md(doc)
        assert "![A gif](assets/cimg0001.gif)" in text
        # A figure cell would carry a figures/<id>.yaml — an image cell never does.

    def test_nonpng_image_ref_parses_as_image_cell(self):
        text = ("---\nversion: 1\ntitle: T\n---\n\n"
                "![A photo](assets/cimg0002.jpg)\n")
        back = m.parse_report_md(text)
        assert [c.cell_type for c in back.cells] == ["image"]
        assert back.cells[0].id == "cimg0002"
        assert back.cells[0].caption == "A photo"
        assert back.cells[0].image_ext == "jpg"

    def test_png_ref_defaults_to_figure_in_parse(self):
        """A bare ``.png`` ref parses as a FIGURE cell (back-compat); the
        image-vs-figure split for pngs is resolved by read_report's yaml check."""
        text = ("---\nversion: 1\ntitle: T\n---\n\n"
                "![Cap](assets/cf0001.png)\n")
        back = m.parse_report_md(text)
        assert [c.cell_type for c in back.cells] == ["figure"]

    def test_image_cell_round_trips_through_md(self):
        doc = m.ReportDoc(title="Photos")
        doc.cells.append(m.Cell(cell_type="markdown", source="Intro"))
        doc.cells.append(m.Cell(id="cimg9", cell_type="image",
                                caption="Sample micrograph", image_ext="webp"))
        doc.cells.append(m.Cell(cell_type="markdown", source="Outro"))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert [c.cell_type for c in back.cells] == ["markdown", "image", "markdown"]
        img = back.cells[1]
        assert img.id == "cimg9"
        assert img.caption == "Sample micrograph"
        assert img.image_ext == "webp"

    def test_image_and_figure_interleaved_distinguished(self):
        """An image ref and a figure ref in the same doc parse to distinct types
        by extension (jpg=image, png=figure)."""
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(id="cimgA", cell_type="image", caption="photo",
                                image_ext="jpg"))
        doc.cells.append(m.Cell(id="cfigB", cell_type="figure", caption="fig"))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert [(c.cell_type, c.id) for c in back.cells] == [
            ("image", "cimgA"), ("figure", "cfigB")]

    def test_double_round_trip_stable(self):
        doc = m.ReportDoc(title="Stable")
        doc.created = doc.modified = "2020-01-01T00:00:00+00:00"
        doc.cells.append(m.Cell(id="cimgZ", cell_type="image", caption="pic",
                                image_ext="jpg"))
        t1 = m.serialize_report_md(doc)
        back = m.parse_report_md(t1)
        back.created = back.modified = "2020-01-01T00:00:00+00:00"
        t2 = m.serialize_report_md(back)
        assert t1 == t2


# ── zip save + reopen ───────────────────────────────────────────────────────────


class TestImageCellZip:
    def test_save_and_reopen_restores_image_bytes(self, tmp_path):
        doc = m.ReportDoc(title="Zipped Photos")
        doc.cells.append(m.Cell(id="cimg1", cell_type="image", caption="jpg photo",
                                image_ext="jpg"))
        doc.cells.append(m.Cell(id="cimg2", cell_type="image", caption="gif",
                                image_ext="gif"))
        assets = {"cimg1": _PNG_1x1, "cimg2": _GIF_1x1}
        path = str(tmp_path / "photos.spyde-report")
        m.write_report(doc, path, assets=assets)

        back, back_assets = m.read_report(path)
        assert [c.cell_type for c in back.cells] == ["image", "image"]
        assert back.cells[0].image_ext == "jpg"
        assert back.cells[1].image_ext == "gif"
        assert back_assets["cimg1"] == _PNG_1x1
        assert back_assets["cimg2"] == _GIF_1x1

    def test_png_image_without_yaml_promoted_to_image_cell(self, tmp_path):
        """A pasted PNG screenshot is stored as an image cell — its ``.png`` asset
        has NO sibling figures yaml, so read_report promotes it from the default
        figure parse back to an image cell (the yaml-presence rule)."""
        doc = m.ReportDoc(title="Screenshot")
        doc.cells.append(m.Cell(id="cshot1", cell_type="image", caption="shot",
                                image_ext="png"))
        path = str(tmp_path / "shot.spyde-report")
        m.write_report(doc, path, assets={"cshot1": _PNG_1x1})

        back, back_assets = m.read_report(path)
        assert [c.cell_type for c in back.cells] == ["image"]
        assert back.cells[0].image_ext == "png"
        assert back.cells[0].spec is None
        assert back_assets["cshot1"] == _PNG_1x1

    def test_figure_png_with_yaml_stays_figure(self, tmp_path):
        """A real figure cell (PNG + figures yaml) is NOT mis-promoted to image."""
        ref = m.SignalRef(title="scan")
        layer = m.LayerSpec(source=ref, cmap="viridis")
        panel = m.PanelSpec(id="p1", grid_pos=[0, 0], kind="image", layers=[layer])
        spec = m.FigureSpec(layout={"kind": "single"}, panels=[panel])
        doc = m.ReportDoc(title="Real Figure")
        doc.cells.append(m.Cell(id="cfig1", cell_type="figure", caption="DP",
                                spec=spec))
        path = str(tmp_path / "fig.spyde-report")
        m.write_report(doc, path, assets={"cfig1": _PNG_1x1})

        back, _assets = m.read_report(path)
        assert [c.cell_type for c in back.cells] == ["figure"]
        assert back.cells[0].spec is not None

    def test_end_to_end_add_save_reopen_via_handlers(self, window, tmp_path):
        """The full flow: add via handler → save → reopen → the image cell +
        bytes come back and re-emit their data URL."""
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_image_cell(session, None, {
            "image_b64": _GIF_DATA_URL, "image_ext": "gif", "caption": "held"})
        path = str(tmp_path / "roundtrip.spyde-report")
        h.report_save(session, None, {"path": path})

        # Reopen into a fresh manager state.
        messages.clear()
        h.report_open(session, None, {"path": path})
        st = _last_state(messages)
        cell = st["cells"][0]
        assert cell["cell_type"] == "image"
        assert cell["caption"] == "held"
        assert cell["image"].startswith("data:image/gif;base64,")
        assert base64.b64decode(cell["image"].split(",", 1)[1]) == _GIF_1x1


# ── caption edit ────────────────────────────────────────────────────────────────


class TestImageCaption:
    def test_update_cell_edits_image_caption(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_DATA_URL, "image_ext": "png", "caption": "old"})
        cid = session._report.doc.cells[0].id
        messages.clear()
        h.report_update_cell(session, None, {"cell_id": cid, "caption": "new caption"})
        cell = next(c for c in _last_state(messages)["cells"] if c["id"] == cid)
        assert cell["caption"] == "new caption"

    def test_set_caption_edits_image_caption(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_DATA_URL, "image_ext": "png", "caption": "a"})
        cid = session._report.doc.cells[0].id
        h.report_set_caption(session, None, {"cell_id": cid, "caption": "b"})
        assert session._report.doc.cells[0].caption == "b"

    def test_remove_image_cell_clears_held_bytes(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_DATA_URL, "image_ext": "png"})
        cid = session._report.doc.cells[0].id
        h.report_remove_cell(session, None, {"cell_id": cid})
        assert not session._report.doc.cells
        assert cid not in session._report._images


# ── HTML export inlines the image ───────────────────────────────────────────────


class TestImageExport:
    def _add_photo(self, session):
        h.report_new(session, None, {})
        h.report_add_image_cell(session, None, {
            "image_b64": _PNG_DATA_URL, "image_ext": "png", "caption": "Fig 1"})

    def test_static_export_inlines_image_data_url(self, window, tmp_path):
        session, messages = window["window"], window["messages"]
        self._add_photo(session)
        path = str(tmp_path / "static.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "static", "path": path})
        assert _exported(messages) and not _errors(messages)

        html = open(path, encoding="utf-8").read()
        b64 = base64.b64encode(_PNG_1x1).decode("ascii")
        assert f"data:image/png;base64,{b64}" in html
        assert "<figcaption>Fig 1</figcaption>" in html
        # Self-contained — no external fetches.
        assert "http://" not in html

    def test_slides_export_inlines_image_data_url(self, window, tmp_path):
        session, messages = window["window"], window["messages"]
        self._add_photo(session)
        path = str(tmp_path / "deck.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "slides", "path": path})
        assert _exported(messages) and _exported(messages)[0]["kind"] == "html-slides"

        html = open(path, encoding="utf-8").read()
        b64 = base64.b64encode(_PNG_1x1).decode("ascii")
        assert f"data:image/png;base64,{b64}" in html
        assert '<section class="slide">' in html

    def test_gif_export_uses_gif_mime(self, window, tmp_path):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_image_cell(session, None, {
            "image_b64": _GIF_DATA_URL, "image_ext": "gif", "caption": "g"})
        path = str(tmp_path / "gif.html")
        ex.report_export_html(session, None, {"mode": "static", "path": path})
        html = open(path, encoding="utf-8").read()
        assert "data:image/gif;base64," in html


# ── paste ───────────────────────────────────────────────────────────────────────


class TestImagePaste:
    def test_paste_image_cell(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        ex.report_paste_cell(session, None, {"cell": {
            "cell_type": "image", "caption": "pasted", "image_ext": "gif",
            "image": _GIF_DATA_URL}})
        cell = session._report.doc.cells[0]
        assert cell.cell_type == "image"
        assert cell.caption == "pasted"
        assert cell.image_ext == "gif"
        assert session._report._images[cell.id] == _GIF_1x1

    def test_paste_image_cell_missing_bytes_errors(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        ex.report_paste_cell(session, None, {"cell": {
            "cell_type": "image", "image_ext": "png"}})
        assert _errors(messages)

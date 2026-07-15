"""
test_report_model.py — the Report data model + ``.spyde-report`` container.

Round-trip stability of report.md (incl. odd user markdown: inline images,
non-spyde HTML comments, code fences with figure-ref lookalikes), FigureSpec
YAML, the atomic zip write, fingerprint match/mismatch, placeholder parsing, and
the baked PNG fallback.
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pytest

from spyde.actions.report import model as m


# ── report.md round-trip ──────────────────────────────────────────────────────


class TestMarkdownRoundTrip:
    def test_basic_markdown_and_figure(self):
        doc = m.ReportDoc(title="Analysis")
        doc.cells.append(m.Cell(cell_type="markdown", source="# Heading\n\nBody."))
        doc.cells.append(m.Cell(id="cfig00001", cell_type="figure",
                                caption="A pattern"))
        text = m.serialize_report_md(doc)
        back = m.parse_report_md(text)

        assert back.title == "Analysis"
        assert [c.cell_type for c in back.cells] == ["markdown", "figure"]
        assert back.cells[0].source == "# Heading\n\nBody."
        assert back.cells[1].id == "cfig00001"
        assert back.cells[1].caption == "A pattern"
        assert back.cells[1].placeholder is False

    def test_front_matter_fields(self):
        doc = m.ReportDoc(title="T", template=True)
        text = m.serialize_report_md(doc)
        back = m.parse_report_md(text)
        assert back.title == "T"
        assert back.template is True
        assert back.version == m.SCHEMA_VERSION
        assert back.created and back.modified

    def test_inline_image_is_not_a_figure_cell(self):
        """An inline image inside a text paragraph is NOT a figure cell — only a
        standalone-paragraph ``assets/<id>.png`` ref is."""
        src = ("Here is an inline ![logo](https://x/y.png) image, and a local "
               "one ![z](assets/notreally.png) mid-sentence.")
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source=src))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert [c.cell_type for c in back.cells] == ["markdown"]
        assert back.cells[0].source == src

    def test_non_spyde_html_comment_survives(self):
        src = "<!-- TODO: revisit this -->\n\nSome text."
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source=src))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert [c.cell_type for c in back.cells] == ["markdown"]
        assert "TODO: revisit" in back.cells[0].source

    def test_code_fence_with_figure_lookalike_not_parsed(self):
        """A fenced code block containing an ``![x](assets/c1.png)`` line must
        pass through as markdown, NOT parse as a figure cell."""
        fence = "```"
        src = (f"Example markdown:\n\n{fence}markdown\n"
               f"![x](assets/c1.png)\n"
               f"{fence}\n\nEnd.")
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source=src))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert [c.cell_type for c in back.cells] == ["markdown"]
        assert "assets/c1.png" in back.cells[0].source
        assert back.cells[0].source.count("```") == 2

    def test_tilde_fence_lookalike_not_parsed(self):
        fence = "~~~"
        src = f"{fence}\n![y](assets/c2.png)\n{fence}"
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source=src))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert [c.cell_type for c in back.cells] == ["markdown"]

    def test_four_backtick_fence_with_inner_triple_fence(self):
        """A 4-backtick fence that CONTAINS a nested ```python line + an
        ``![x](assets/cX.png)`` figure lookalike must NOT close at the inner ```
        (CommonMark: a closing fence needs length >= the opener). It round-trips
        intact and spawns NO phantom figure cell."""
        src = (
            "````\n"
            "```python\n"
            "img = '![x](assets/cX.png)'\n"
            "```\n"
            "still inside the outer fence\n"
            "````"
        )
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source=src))
        back = m.parse_report_md(m.serialize_report_md(doc))
        # ONE markdown cell, no figure cell leaked from inside the fence.
        assert [c.cell_type for c in back.cells] == ["markdown"]
        body = back.cells[0].source
        assert "assets/cX.png" in body
        assert "```python" in body
        assert body.count("````") == 2       # outer fence intact (open + close)

    def test_four_backtick_fence_round_trips_stable(self):
        """Serialize→parse→serialize is a fixed point for a nested-fence block."""
        src = (
            "````\n"
            "```\n"
            "![z](assets/cZZ.png)\n"
            "```\n"
            "````"
        )
        doc = m.ReportDoc(title="Nested")
        doc.created = doc.modified = "2020-01-01T00:00:00+00:00"
        doc.cells.append(m.Cell(cell_type="markdown", source=src))
        t1 = m.serialize_report_md(doc)
        back = m.parse_report_md(t1)
        back.created = back.modified = "2020-01-01T00:00:00+00:00"
        t2 = m.serialize_report_md(back)
        assert t1 == t2
        assert [c.cell_type for c in back.cells] == ["markdown"]

    def test_four_tilde_fence_with_inner_triple_tilde(self):
        """The same nested-fence rule for tilde fences."""
        src = (
            "~~~~\n"
            "~~~\n"
            "![t](assets/cTT.png)\n"
            "~~~\n"
            "~~~~"
        )
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source=src))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert [c.cell_type for c in back.cells] == ["markdown"]
        assert "assets/cTT.png" in back.cells[0].source

    def test_placeholder_parsing_with_and_without_caption(self):
        doc = m.ReportDoc(template=True)
        doc.cells.append(m.Cell(id="cph1", cell_type="figure",
                                caption="Cap here", placeholder=True))
        doc.cells.append(m.Cell(id="cph2", cell_type="figure",
                                caption="", placeholder=True))
        text = m.serialize_report_md(doc)
        assert "spyde:placeholder cph1 Cap here" in text
        assert "spyde:placeholder cph2 -->" in text
        back = m.parse_report_md(text)
        assert all(c.placeholder for c in back.cells)
        assert back.cells[0].caption == "Cap here"
        assert back.cells[1].caption == ""

    def test_interleaved_cells_order_preserved(self):
        doc = m.ReportDoc()
        doc.cells.append(m.Cell(cell_type="markdown", source="Intro"))
        doc.cells.append(m.Cell(id="cf1", cell_type="figure", caption="F1"))
        doc.cells.append(m.Cell(cell_type="markdown", source="Middle"))
        doc.cells.append(m.Cell(id="cf2", cell_type="figure", caption="F2"))
        doc.cells.append(m.Cell(cell_type="markdown", source="Outro"))
        back = m.parse_report_md(m.serialize_report_md(doc))
        assert [c.cell_type for c in back.cells] == [
            "markdown", "figure", "markdown", "figure", "markdown"]
        assert [c.id for c in back.cells if c.cell_type == "figure"] == ["cf1", "cf2"]
        assert back.cells[0].source == "Intro"
        assert back.cells[2].source == "Middle"
        assert back.cells[4].source == "Outro"

    def test_double_round_trip_stable(self):
        """Serialize → parse → serialize yields identical text (modulo the
        modified timestamp, which we hold fixed)."""
        doc = m.ReportDoc(title="Stable")
        doc.created = doc.modified = "2020-01-01T00:00:00+00:00"
        doc.cells.append(m.Cell(cell_type="markdown", source="Para one.\n\nPara two."))
        doc.cells.append(m.Cell(id="cf9", cell_type="figure", caption="Fig"))
        t1 = m.serialize_report_md(doc)
        back = m.parse_report_md(t1)
        back.created = back.modified = "2020-01-01T00:00:00+00:00"
        t2 = m.serialize_report_md(back)
        assert t1 == t2


# ── FigureSpec YAML ───────────────────────────────────────────────────────────


class TestFigureSpecYaml:
    def _spec(self):
        ref = m.SignalRef(file_path="/data/scan.hspy",
                          fingerprint={"size": 100, "mtime": 1.5},
                          tree_node="scan", view="Signal", title="scan")
        layer = m.LayerSpec(source=ref, cmap="inferno", clim=[0.0, 5.0],
                            alpha=0.8, visible=True)
        panel = m.PanelSpec(id="p1", grid_pos=[0, 0], kind="image",
                            layers=[layer], axes={"units": "1/nm"},
                            scalebar=True, title="DP")
        return m.FigureSpec(layout={"kind": "single"}, panels=[panel],
                            nav_context={"indices": [3, 4]})

    def test_yaml_round_trip(self):
        spec = self._spec()
        text = spec.to_yaml()
        back = m.FigureSpec.from_yaml(text)
        assert back.to_dict() == spec.to_dict()

    def test_yaml_has_no_python_object_tags(self):
        """Human-readable requirement: plain YAML, no ``!!python`` object tags,
        no JSON."""
        text = self._spec().to_yaml()
        assert "!!python" not in text
        assert "!!" not in text
        assert "{" not in text.replace("{}", "")  # block style, no flow maps

    def test_primary_layer_helper(self):
        spec = self._spec()
        assert spec.primary_layer is not None
        assert spec.primary_layer.cmap == "inferno"
        assert m.FigureSpec().primary_layer is None

    def test_figure_annotations_default_empty(self):
        """A FigureSpec has no figure-level annotations by default."""
        assert m.FigureSpec().annotations == []
        assert m.FigureSpec().to_dict()["annotations"] == []

    def test_figure_annotations_yaml_round_trip(self):
        """Figure-level annotations (figure-fraction markers, NO calibration
        conversion) survive to_dict / YAML / zip round-trips verbatim."""
        anns = [
            {"id": "m1", "kind": "text", "x": 0.5, "y": 0.5, "text": "Label",
             "color": "#ff9800", "fontsize": 14},
            {"id": "m2", "kind": "circle", "x": 0.2, "y": 0.3, "r": 0.08},
            {"id": "m3", "kind": "rect", "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.15},
            {"id": "m4", "kind": "arrow", "x": 0.35, "y": 0.35, "u": 0.15, "v": 0.15},
        ]
        spec = self._spec()
        spec.annotations = [dict(a) for a in anns]
        back = m.FigureSpec.from_yaml(spec.to_yaml())
        assert back.annotations == anns
        assert back.to_dict() == spec.to_dict()

    def test_figure_annotations_absent_key_yields_empty(self):
        """A spec dict from an OLDER file (no ``annotations`` key) → []."""
        d = {"layout": {"kind": "single"}, "panels": []}
        assert m.FigureSpec.from_dict(d).annotations == []

    def test_figure_annotations_zip_round_trip(self, tmp_path):
        """Figure-level annotations survive the .spyde-report zip write/read."""
        doc = m.ReportDoc(title="FigAnn")
        anns = [{"id": "z1", "kind": "text", "x": 0.4, "y": 0.6, "text": "Hi"}]
        spec = m.FigureSpec(panels=[m.PanelSpec(layers=[m.LayerSpec()])],
                            annotations=anns)
        doc.cells.append(m.Cell(id="cfa00001", cell_type="figure", spec=spec))
        path = str(tmp_path / "figann.spyde-report")
        m.write_report(doc, path, assets={"cfa00001": b""})
        back, _ = m.read_report(path)
        assert back.cells[0].spec.annotations == anns


# ── zip container (atomic) ────────────────────────────────────────────────────


class TestZipContainer:
    def _doc_with_fig(self):
        doc = m.ReportDoc(title="Zipped")
        doc.cells.append(m.Cell(cell_type="markdown", source="Report body."))
        layer = m.LayerSpec(source=m.SignalRef(file_path="/x.hspy"))
        spec = m.FigureSpec(panels=[m.PanelSpec(layers=[layer])])
        doc.cells.append(m.Cell(id="czip0001", cell_type="figure",
                                caption="Cap", spec=spec))
        return doc

    def test_write_read_round_trip(self, tmp_path):
        doc = self._doc_with_fig()
        png = m.bake_fallback_png(np.random.RandomState(0).rand(48, 48))
        path = str(tmp_path / "r.spyde-report")
        m.write_report(doc, path, assets={"czip0001": png})

        # The container is a real zip with the expected member layout.
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        assert "report.md" in names
        assert "figures/czip0001.yaml" in names
        assert "assets/czip0001.png" in names

        back, assets = m.read_report(path)
        assert back.title == "Zipped"
        assert [c.cell_type for c in back.cells] == ["markdown", "figure"]
        assert back.cells[1].spec is not None
        assert back.cells[1].spec.primary_layer.source.file_path == "/x.hspy"
        assert assets["czip0001"] == png

    def test_no_json_anywhere_in_container(self, tmp_path):
        doc = self._doc_with_fig()
        path = str(tmp_path / "r.spyde-report")
        m.write_report(doc, path, assets={"czip0001": b""})
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                assert not n.endswith(".json"), n
            md = zf.read("report.md").decode("utf-8")
            spec = zf.read("figures/czip0001.yaml").decode("utf-8")
        # No JSON object-literal syntax leaking into the human-readable files.
        assert "!!python" not in spec
        assert md.lstrip().startswith("---")

    def test_atomic_write_no_partial_file_on_failure(self, tmp_path, monkeypatch):
        """A failure mid-write must leave NO file at the target path and NO tmp
        debris (atomic tmp + os.replace contract, like nav_sidecar)."""
        doc = self._doc_with_fig()
        path = str(tmp_path / "r.spyde-report")

        # Force os.replace to blow up AFTER the tmp zip has been written.
        import spyde.actions.report.model as mm

        def _boom(src, dst):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(mm.os, "replace", _boom)
        with pytest.raises(OSError):
            m.write_report(doc, path, assets={"czip0001": b"x"})

        assert not os.path.exists(path), "target file must not exist after failure"
        # No leftover tmp file in the directory.
        leftovers = [p for p in os.listdir(tmp_path) if ".tmp" in p]
        assert leftovers == [], f"tmp debris left behind: {leftovers}"

    def test_atomic_write_overwrites_existing(self, tmp_path):
        doc = self._doc_with_fig()
        path = str(tmp_path / "r.spyde-report")
        m.write_report(doc, path, assets={"czip0001": b"a"})
        doc.title = "Second"
        m.write_report(doc, path, assets={"czip0001": b"b"})
        back, assets = m.read_report(path)
        assert back.title == "Second"
        assert assets["czip0001"] == b"b"


# ── fingerprint + rebind resolution ───────────────────────────────────────────


class TestFingerprint:
    def test_fingerprint_match_and_mismatch(self, tmp_path):
        f = tmp_path / "scan.hspy"
        f.write_bytes(b"0123456789")
        path = str(f)
        fp1 = m.fingerprint_file(path)
        assert fp1 is not None and fp1["size"] == 10

        # Same file → same fingerprint.
        assert m.fingerprint_file(path) == fp1

        # Change the size → mismatch.
        f.write_bytes(b"0123456789EXTRA")
        fp2 = m.fingerprint_file(path)
        assert fp2 != fp1
        assert fp2["size"] == 15

    def test_fingerprint_missing_file(self):
        assert m.fingerprint_file(None) is None
        assert m.fingerprint_file("/no/such/file.hspy") is None

    def test_signalref_dict_round_trip(self):
        ref = m.SignalRef(file_path="/a.hspy", fingerprint={"size": 1, "mtime": 2.0},
                          tree_node="a", view="Signal", title="a")
        assert m.SignalRef.from_dict(ref.to_dict()).to_dict() == ref.to_dict()

    def test_signalref_resolve_none_session(self):
        ref = m.SignalRef(file_path="/a.hspy")
        assert ref.resolve(None) is None


# ── title-based rebind with shape disambiguation (finding 3) ───────────────────


class _FakeMeta:
    def __init__(self, title):
        self._title = title

    def get_item(self, key, default=""):
        if key == "General.title":
            return self._title
        return default


class _FakeSignal:
    def __init__(self, shape):
        self.metadata = _FakeMeta("")
        self.data = np.zeros(shape, dtype=np.float32)


class _FakeAxesRoot:
    def __init__(self, title):
        self.metadata = _FakeMeta(title)


class _FakeTree:
    def __init__(self, title, shape, source_path=None):
        self.root = _FakeAxesRoot(title)
        self.source_path = source_path


class _FakePlotState:
    def __init__(self, shape):
        self.current_signal = _FakeSignal(shape)


class _FakePlot:
    def __init__(self, title, shape, source_path=None, is_navigator=False):
        self.signal_tree = _FakeTree(title, shape, source_path)
        self.plot_state = _FakePlotState(shape)
        self.is_navigator = is_navigator


class _FakeSession:
    def __init__(self, plots):
        self._plots = plots


class TestRebindShape:
    def test_same_title_different_shape_rebinds_correct(self):
        """Two open trees share a title but differ in shape → the ref (which
        recorded a shape) rebinds to the tree with the MATCHING shape, never the
        wrong one."""
        p_small = _FakePlot("Scan", (16, 16))
        p_big = _FakePlot("Scan", (64, 64))
        session = _FakeSession([p_small, p_big])
        # A ref pinned to the small-shape tree by title + shape.
        ref = m.SignalRef(tree_node="Scan", title="Scan", shape=[16, 16])
        assert ref.resolve(session) is p_small
        # And the big one.
        ref_big = m.SignalRef(tree_node="Scan", title="Scan", shape=[64, 64])
        assert ref_big.resolve(session) is p_big

    def test_same_title_same_shape_is_ambiguous_offline(self):
        """Two open trees share BOTH title and shape → genuinely ambiguous; the
        ref resolves to None (offline) rather than binding the wrong one."""
        p1 = _FakePlot("Scan", (32, 32))
        p2 = _FakePlot("Scan", (32, 32))
        session = _FakeSession([p1, p2])
        ref = m.SignalRef(tree_node="Scan", title="Scan", shape=[32, 32])
        assert ref.resolve(session) is None

    def test_title_match_no_shape_still_binds(self):
        """An OLD ref with no shape (older file) still binds by title alone when
        unambiguous — shape is only a disambiguator, not a hard requirement."""
        p = _FakePlot("Scan", (16, 16))
        session = _FakeSession([p])
        ref = m.SignalRef(tree_node="Scan", title="Scan", shape=None)
        assert ref.resolve(session) is p

    def test_shape_serialized_and_tolerant_of_absence(self):
        """``shape`` round-trips through to_dict/from_dict, and from_dict tolerates
        an old dict that lacks it."""
        ref = m.SignalRef(tree_node="Scan", title="Scan", shape=[8, 8])
        assert ref.to_dict()["shape"] == [8, 8]
        assert m.SignalRef.from_dict(ref.to_dict()).shape == [8, 8]
        # Old file: no "shape" key → None, no error.
        old = {"tree_node": "Scan", "title": "Scan"}
        assert m.SignalRef.from_dict(old).shape is None


# ── baked PNG fallback ────────────────────────────────────────────────────────


class TestBakeFallbackPng:
    def test_returns_decodable_png(self):
        arr = np.random.RandomState(1).rand(80, 80).astype(np.float32)
        png = m.bake_fallback_png(arr, cmap="viridis", clim=[0.0, 1.0])
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        from PIL import Image
        img = Image.open(io.BytesIO(png))
        img.verify()

    def test_downsamples_huge_frame(self):
        arr = np.random.RandomState(2).rand(4000, 4000).astype(np.float32)
        png = m.bake_fallback_png(arr, max_edge=1200)
        from PIL import Image
        img = Image.open(io.BytesIO(png))
        assert max(img.size) <= 1200

    def test_rgb_passthrough(self):
        rgb = (np.random.RandomState(3).rand(30, 30, 3) * 255).astype(np.uint8)
        png = m.bake_fallback_png(rgb)
        from PIL import Image
        img = Image.open(io.BytesIO(png))
        img.verify()

    def test_auto_clim_when_none(self):
        arr = np.linspace(0, 100, 64 * 64).reshape(64, 64).astype(np.float32)
        png = m.bake_fallback_png(arr, clim=None)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

"""
test_report_export.py — Report Builder Phase 3 export + copy/paste.

Exercises the export handlers (``export_html.py``) against a real Qt-free
``Session`` (the ``window`` / ``tem_2d_dataset`` fixtures + ``captured_messages``):

* static HTML export: title, one ``<img src="data:image/png>`` per figure cell,
  captions, the cached-``html`` path AND the ``<pre class="md-src">`` fallback, no
  ``\\x00bin:`` bytes, no ``<iframe>``.
* interactive HTML export: N sandboxed ``srcdoc`` iframes, no ``\\x00bin:`` (even
  with binary transport ON), and an offline figure falling back to ``<img>``.
* markdown-folder export: dir contents match the zip serialization; a non-empty
  foreign directory is refused.
* paste: markdown + figure (resolvable → live rebuild; unresolvable → offline with
  the provided PNG).
"""
from __future__ import annotations

import base64
import os

import numpy as np

from spyde.actions.report import export_html as ex
from spyde.actions.report import handlers as h
from spyde.actions.report.model import bake_fallback_png
from spyde.tests.migrated._report import answer_harvest


# ── helpers (mirrors test_report_handlers) ─────────────────────────────────────


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _exported(messages, session=None):
    """The export replies — answering the snapshot handshake first.

    A report with live figure cells does not write inline any more: it asks the
    renderer for fresh PNGs and waits. Pass the session so this answers that
    request the way the renderer does; without one it is the plain filter it
    always was (fine for a report with nothing live to harvest).
    """
    if session is not None:
                answer_harvest(session, messages)
    return [m for m in messages if m.get("type") == "report_exported"]


def _errors(messages):
    return [m for m in messages if m.get("type") == "error"]


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


def _fig_cell_id(session):
    """The id of the (single) figure cell in the open report."""
    mgr = session._report
    for c in mgr.doc.cells:
        if c.cell_type == "figure":
            return c.id
    return None


# ── static HTML export ─────────────────────────────────────────────────────────


class TestStaticExport:
    def test_static_html_has_title_imgs_captions(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        h.report_new(session, None, {"template": False})
        h.report_set_title(session, None, {"title": "Grain Analysis"})
        # A markdown cell WITH a renderer-cached html fragment (the common path).
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Intro\n\nSome text.",
            "html": "<h1>Intro</h1>\n<p>Some text.</p>",
        })
        # A markdown cell WITHOUT html (fallback path → <pre class="md-src">).
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "raw & <unescaped> body"})
        h.report_add_figure(session, None, {"source_window_id": wid,
                                            "caption": "My DP"})

        path = str(tmp_path / "report.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "static", "path": path})

        exp = _exported(messages, session)
        assert exp and exp[0]["kind"] == "html-static"
        assert exp[0]["path"] == path
        assert not _errors(messages)

        html = open(path, encoding="utf-8").read()
        # Title in <title> AND the article heading.
        assert "<title>Grain Analysis</title>" in html
        assert ">Grain Analysis</h1>" in html
        # One <img src="data:image/png per figure cell.
        assert html.count('<img src="data:image/png;base64,') == 1
        # Caption present in a <figcaption>.
        assert "<figcaption>My DP</figcaption>" in html
        # Cached-html path: the rendered fragment is embedded verbatim.
        assert "<h1>Intro</h1>" in html
        # Fallback path: raw markdown escaped inside <pre class="md-src">.
        assert '<pre class="md-src">' in html
        assert "raw &amp; &lt;unescaped&gt; body" in html
        # No binary tokens, no iframes.
        assert "\x00bin:" not in html
        assert "<iframe" not in html

    def test_static_skips_placeholder(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)

        from spyde.actions.report.model import Cell
        h.report_new(session, None, {"template": True})
        mgr = session._report
        mgr.doc.cells.append(Cell(cell_type="figure", caption="empty slot",
                                  placeholder=True))
        path = str(tmp_path / "tpl.html")
        ex.report_export_html(session, None, {"mode": "static", "path": path})

        html = open(path, encoding="utf-8").read()
        # A placeholder contributes no <img> and no caption.
        assert '<img src="data:image/png' not in html
        assert "empty slot" not in html

    def test_static_no_open_report_errors(self, window):
        session, messages = window["window"], window["messages"]
        ex.report_export_html(session, None, {"mode": "static", "path": "x.html"})
        assert _errors(messages)

    def test_static_temp_writes_unique_tempfile(self, tem_2d_dataset):
        """`temp:true` (the PDF-export first leg) writes into the OS temp dir and
        emits `report_exported` with THAT generated path — `path` is ignored."""
        import tempfile

        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        h.report_new(session, None, {})
        h.report_set_title(session, None, {"title": "PDF Source"})
        h.report_add_cell(session, None, {
            "cell_type": "markdown", "source": "# Body",
            "html": "<h1>Body</h1>"})
        h.report_add_figure(session, None, {"source_window_id": wid,
                                            "caption": "DP"})

        messages.clear()
        # No `path` — the temp branch generates its own.
        ex.report_export_html(session, None, {"mode": "static", "temp": True})

        exp = _exported(messages, session)
        assert exp and exp[0]["kind"] == "html-static"
        assert not _errors(messages)
        out = exp[0]["path"]
        # A unique file under the OS temp dir (not a caller-supplied path).
        assert os.path.dirname(out) == tempfile.gettempdir()
        assert os.path.basename(out).startswith("spyde-report-")
        assert out.endswith(".html")
        assert os.path.isfile(out)

        html = open(out, encoding="utf-8").read()
        assert "<title>PDF Source</title>" in html
        assert html.count('<img src="data:image/png;base64,') == 1
        assert "\x00bin:" not in html
        assert "<iframe" not in html
        try:
            os.remove(out)
        except OSError:
            pass


# ── interactive HTML export ────────────────────────────────────────────────────


class TestInteractiveExport:
    def test_interactive_has_sandboxed_iframes_no_bin(self, tem_2d_dataset, tmp_path,
                                                      monkeypatch):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        # Turn binary transport ON so the pixel-resolve path is actually exercised
        # (otherwise base64 is already inline and "no \x00bin:" is trivially true).
        monkeypatch.setenv("APL_BINARY_TRANSPORT", "1")

        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "Body",
                                          "html": "<p>Body</p>"})
        h.report_add_figure(session, None, {"source_window_id": wid, "caption": "F"})

        path = str(tmp_path / "interactive.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "interactive", "path": path})

        exp = _exported(messages, session)
        assert exp and exp[0]["kind"] == "html-interactive"
        assert not _errors(messages)

        html = open(path, encoding="utf-8").read()
        # One sandboxed srcdoc iframe per (rebuildable) figure cell.
        assert html.count("<iframe sandbox=\"allow-scripts\" srcdoc=") == 1
        # The pixel tokens were materialised — no binary tokens leak into the page.
        assert "\x00bin:" not in html
        # The srcdoc content is HTML-escaped (can't break out of the attribute).
        assert "&lt;" in html

    def test_interactive_offline_falls_back_to_img(self, tem_2d_dataset, tmp_path):
        """A figure cell with no rebuildable live figure (offline) falls back to
        the static <img> in interactive mode."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid, "caption": "F"})
        cid = _fig_cell_id(session)
        mgr = session._report
        # Simulate an offline cell: drop the live snapshot map (so no rebuild),
        # keep a baked PNG so the static <img> fallback has pixels.
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        mgr._baked[cid] = bake_fallback_png(arr)
        mgr._snapshots.pop(cid, None)

        path = str(tmp_path / "offline_interactive.html")
        ex.report_export_html(session, None, {"mode": "interactive", "path": path})
        # The cell is still MOUNTED (only its snapshot was dropped), so the
        # export asks the renderer for pixels before it writes. Answer, or the
        # file does not exist until the 3 s fallback fires.
        answer_harvest(session, messages)

        html = open(path, encoding="utf-8").read()
        assert "<iframe" not in html
        assert html.count('<img src="data:image/png;base64,') == 1


# ── markdown-folder export ─────────────────────────────────────────────────────


class TestMarkdownFolderExport:
    def test_folder_matches_zip_serialization(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        h.report_new(session, None, {})
        h.report_set_title(session, None, {"title": "Folder Export"})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "Notes"})
        h.report_add_figure(session, None, {"source_window_id": wid, "caption": "DP"})
        cid = _fig_cell_id(session)

        out = str(tmp_path / "export_dir")
        messages.clear()
        ex.report_export_markdown(session, None, {"path": out})

        exp = _exported(messages, session)
        assert exp and exp[0]["kind"] == "markdown-folder"
        assert exp[0]["path"] == out
        assert not _errors(messages)

        # The directory holds exactly the unzipped container layout.
        assert os.path.isfile(os.path.join(out, "report.md"))
        assert os.path.isfile(os.path.join(out, "figures", f"{cid}.yaml"))
        asset = os.path.join(out, "assets", f"{cid}.png")
        assert os.path.isfile(asset)
        assert open(asset, "rb").read()[:8] == b"\x89PNG\r\n\x1a\n"

        # report.md content matches what the zip serializer would write, and the
        # spec round-trips.
        from spyde.actions.report import model as m
        md = open(os.path.join(out, "report.md"), encoding="utf-8").read()
        parsed = m.parse_report_md(md)
        assert parsed.title == "Folder Export"
        assert [c.cell_type for c in parsed.cells] == ["markdown", "figure"]

    def test_reexport_over_prior_export_ok(self, tem_2d_dataset, tmp_path):
        """Re-exporting into a directory that already looks like a prior export
        (report.md / figures / assets) is allowed."""
        session = tem_2d_dataset["window"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})

        out = str(tmp_path / "reexport")
        ex.report_export_markdown(session, None, {"path": out})
        messages = tem_2d_dataset["messages"]
        messages.clear()
        # Second export into the same dir — no refusal.
        ex.report_export_markdown(session, None, {"path": out})
        assert _exported(messages, session) and not _errors(messages)

    def test_refuses_non_empty_foreign_dir(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})

        out = tmp_path / "populated"
        out.mkdir()
        (out / "important.txt").write_text("do not clobber")
        messages.clear()
        ex.report_export_markdown(session, None, {"path": str(out)})

        assert not _exported(messages, session)
        assert _errors(messages)
        # The foreign file is untouched.
        assert (out / "important.txt").read_text() == "do not clobber"


# ── paste cell ─────────────────────────────────────────────────────────────────


class TestPasteCell:
    def test_paste_markdown_cell(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        messages.clear()
        ex.report_paste_cell(session, None, {
            "cell": {"cell_type": "markdown", "source": "pasted body",
                     "html": "<p>pasted body</p>"}})
        st = _last_state(messages)
        assert len(st["cells"]) == 1
        assert st["cells"][0]["cell_type"] == "markdown"
        assert st["cells"][0]["source"] == "pasted body"

    def test_paste_figure_resolvable_rebuilds_live(self, tem_2d_dataset):
        """A figure cell whose SignalRef resolves to an open plot rebuilds live
        (fresh ids, a report figure emitted, not offline)."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        # Build a real figure cell to get a genuine, resolvable FigureSpec dict.
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid, "caption": "src"})
        src_cid = _fig_cell_id(session)
        src_state = [c for c in _last_state(messages)["cells"]
                     if c["id"] == src_cid][0]
        fig_dict = src_state["figure"]
        assert fig_dict is not None

        messages.clear()
        ex.report_paste_cell(session, None, {
            "cell": {"cell_type": "figure", "caption": "pasted DP",
                     "figure": fig_dict}})

        st = _last_state(messages)
        fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
        assert len(fig_cells) == 2   # source + pasted
        pasted = fig_cells[-1]
        assert pasted["id"] != src_cid          # fresh cell id
        assert pasted["caption"] == "pasted DP"
        assert pasted["placeholder"] is False
        assert pasted["data_offline"] is False
        # Fresh panel/layer ids (not colliding with the source spec).
        src_layer_id = fig_dict["panels"][0]["layers"][0]["id"]
        new_layer_id = pasted["figure"]["panels"][0]["layers"][0]["id"]
        assert new_layer_id != src_layer_id
        # A live report figure was emitted for the pasted cell.
        rep_figs = [m for m in messages if m.get("type") == "figure"
                    and m.get("host") == "report"]
        assert any(m.get("cell_id") == pasted["id"] for m in rep_figs)

    def test_paste_figure_unresolvable_is_offline_with_png(self, window):
        """A figure cell whose SignalRef resolves to NOTHING (no matching plot)
        becomes an offline cell using the provided png data URL as the fallback."""
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})

        # A minimal FigureSpec dict pointing at a non-existent source, plus a png.
        arr = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        png = bake_fallback_png(arr)
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        fig_dict = {
            "layout": {"kind": "single"},
            "panels": [{
                "id": "p1", "grid_pos": [0, 0], "kind": "image",
                "layers": [{
                    "id": "lZZZ",
                    "source": {"file_path": "/nope/gone.hspy",
                               "tree_uid": "tNOPE", "tree_node": "ghost"},
                    "cmap": "viridis", "clim": None, "alpha": 1.0, "visible": True,
                }],
            }],
            "nav_context": None,
        }
        messages.clear()
        ex.report_paste_cell(session, None, {
            "cell": {"cell_type": "figure", "caption": "ghost",
                     "figure": fig_dict, "png": data_url}})

        st = _last_state(messages)
        fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
        assert len(fig_cells) == 1
        cell = fig_cells[0]
        assert cell["data_offline"] is True
        assert isinstance(cell.get("png"), str)
        assert cell["png"].startswith("data:image/png;base64,")
        # No live report figure emitted for an offline paste.
        assert not [m for m in messages if m.get("type") == "figure"
                    and m.get("host") == "report"]


# ── export token correlation (cross-agent renderer contract, finding 9) ─────────


class TestExportToken:
    def test_html_export_echoes_token_verbatim(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid, "caption": "F"})

        path = str(tmp_path / "tok.html")
        messages.clear()
        ex.report_export_html(session, None, {
            "mode": "static", "path": path, "token": "req-42"})
        exp = _exported(messages, session)
        assert exp and exp[0]["kind"] == "html-static"
        assert exp[0]["path"] == path
        # The token rides back VERBATIM.
        assert exp[0]["token"] == "req-42"

    def test_html_export_omits_token_when_absent(self, tem_2d_dataset, tmp_path):
        """No token in the request → no token key in the reply (backward compat)."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})

        path = str(tmp_path / "notok.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "static", "path": path})
        exp = _exported(messages, session)
        assert exp and "token" not in exp[0]

    def test_markdown_export_echoes_token_verbatim(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})

        out = str(tmp_path / "tok_dir")
        messages.clear()
        ex.report_export_markdown(session, None, {"path": out, "token": "md-tok-7"})
        exp = _exported(messages, session)
        assert exp and exp[0]["kind"] == "markdown-folder"
        assert exp[0]["token"] == "md-tok-7"

    def test_markdown_export_omits_token_when_absent(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})

        out = str(tmp_path / "notok_dir")
        messages.clear()
        ex.report_export_markdown(session, None, {"path": out})
        exp = _exported(messages, session)
        assert exp and "token" not in exp[0]


# ── every cell kind survives every export mode ────────────────────────────────


def _tiny_png() -> bytes:
    """A 2x2 PNG standing in for a baked poster or snapshot."""
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


class TestCellKindCoverage:
    """Export dispatch must be TOTAL over the document model's cell types.

    Movie cells shipped for a release exporting as nothing at all, image and
    caption both, in static HTML, interactive HTML, the slides deck and the PDF
    that renders from the static file. An unhandled type is indistinguishable
    from an empty report, so the dispatch is pinned against ``model.CELL_TYPES``
    rather than a list copied into the test."""

    def test_every_cell_type_has_an_export_branch(self, window):
        from spyde.actions.report.model import CELL_TYPES, Cell

        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        # One cell of every declared type, each with a distinctive caption/source.
        for kind in CELL_TYPES:
            if kind == "markdown":
                mgr.doc.cells.append(Cell(cell_type="markdown",
                                          source=f"body-{kind}"))
            else:
                mgr.doc.cells.append(Cell(cell_type=kind, caption=f"cap-{kind}"))
        assets = {c.id: _tiny_png() for c in mgr.doc.cells}

        for interactive in (False, True):
            for c in mgr.doc.cells:
                frag = ex._render_cell_html(mgr, c, assets,
                                            interactive=interactive)
                needle = (f"body-{c.cell_type}" if c.cell_type == "markdown"
                          else f"cap-{c.cell_type}")
                assert needle in frag, (
                    f"{c.cell_type} cell exported nothing "
                    f"(interactive={interactive}): dispatch is not total")

    def test_movie_cell_exports_poster_and_caption(self, window, tmp_path):
        from spyde.actions.report.model import Cell

        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        cell = Cell(cell_type="movie", caption="Growth at 400 C")
        mgr.doc.cells.append(cell)
        mgr._baked[cell.id] = _tiny_png()

        path = str(tmp_path / "movie_report.html")
        ex.report_export_html(session, None, {"mode": "static", "path": path})
        html = open(path, encoding="utf-8").read()

        assert "<figcaption>Growth at 400 C</figcaption>" in html
        assert '<img src="data:image/png;base64,' in html
        # Badged so a still of a movie doesn't read as a static figure.
        assert "movie-badge" in html

    def test_movie_without_a_poster_still_keeps_its_caption(self, window,
                                                            tmp_path):
        from spyde.actions.report.model import Cell

        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        mgr.doc.cells.append(Cell(cell_type="movie", caption="Never rendered"))

        path = str(tmp_path / "unrendered.html")
        ex.report_export_html(session, None, {"mode": "static", "path": path})
        html = open(path, encoding="utf-8").read()
        assert "Never rendered" in html

    def test_unrendered_movie_is_reported_as_a_dropped_asset(self, window):
        from spyde.actions.report.model import Cell

        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        cell = Cell(cell_type="movie", caption="Never rendered")
        mgr.doc.cells.append(cell)

        mgr.assemble_assets({})
        # write_report still writes this cell's image ref, so a poster-less movie
        # is the same dangling-ref hazard a pixel-less figure is: the save must
        # warn rather than report clean.
        assert [c.id for c in mgr._dropped_assets] == [cell.id]

    def test_a_rendered_movie_is_inlined_in_an_interactive_export(self, window,
                                                                  tmp_path):
        from spyde.actions.report.model import Cell

        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        cell = Cell(cell_type="movie", caption="Growth")
        mgr.doc.cells.append(cell)
        mgr._baked[cell.id] = _tiny_png()
        gif = tmp_path / "growth.gif"
        gif.write_bytes(b"GIF89a stands in for the rendered animation")
        mgr._movie_files[cell.id] = str(gif)

        path = str(tmp_path / "inlined.html")
        ex.report_export_html(session, None,
                              {"mode": "interactive", "path": path})
        html = open(path, encoding="utf-8").read()
        assert "data:image/gif;base64," in html

    def test_a_movie_over_the_budget_exports_its_still_and_says_so(
            self, window, tmp_path, monkeypatch):
        from spyde.actions.report.model import Cell

        session = window["window"]
        h.report_new(session, None, {})
        mgr = session._report
        cell = Cell(cell_type="movie", caption="Long run")
        mgr.doc.cells.append(cell)
        mgr._baked[cell.id] = _tiny_png()
        movie = tmp_path / "long.mp4"
        movie.write_bytes(b"x" * (3 * 2 ** 20))
        mgr._movie_files[cell.id] = str(movie)
        monkeypatch.setattr(ex, "EMBED_BUDGET_BYTES", 2 ** 20)

        path = str(tmp_path / "oversize.html")
        ex.report_export_html(session, None,
                              {"mode": "interactive", "path": path})
        html = open(path, encoding="utf-8").read()

        assert "movie-badge" in html, "the still should stand in for the video"
        assert "data:video/mp4" not in html
        # Both numbers, so the reader can judge how far over it is.
        assert "3.0 MB" in html and "1.0 MB" in html

    def test_figure_without_pixels_keeps_its_caption(self):
        # A scene3d cell nobody harvested, or an offline figure that never baked.
        # Dropping the whole <figure> deleted the caption with it, so the reader
        # saw no trace that anything was meant to be there.
        frag = ex._figure_img_html("Orientation, IPF-Z", None)
        assert "Orientation, IPF-Z" in frag
        assert "report-figure--missing" in frag

    def test_no_caption_and_no_pixels_still_renders_nothing(self):
        assert ex._figure_img_html("", None) == ""


class TestFigureBoxMatchesTheSidebar:
    """The exported figure box is sized the way the sidebar cell is.

    A fixed pixel height gave every exported figure the same tall box whatever
    its shape: a wide 1x3 row was letterboxed and a square pattern stretched.
    Reading a report and reading its export should not be two experiences.
    """

    def test_a_single_panel_uses_the_default_ratio(self):
        from spyde.actions.report.model import FigureSpec

        assert ex._figure_aspect(FigureSpec()) == ex._DEFAULT_ASPECT

    def test_a_grid_scales_by_cols_over_rows(self):
        from spyde.actions.report.model import FigureSpec

        wide = FigureSpec(layout={"kind": "grid", "rows": 1, "cols": 3})
        tall = FigureSpec(layout={"kind": "grid", "rows": 3, "cols": 1})
        assert ex._figure_aspect(wide) == ex._PANEL_ASPECT * 3
        assert ex._figure_aspect(tall) == ex._PANEL_ASPECT / 3

    def test_a_vectors_explorer_gets_room_for_its_chrome(self):
        from spyde.actions.report.model import FigureSpec

        spec = FigureSpec()
        spec.vectors_mode = "viewer"
        assert ex._figure_aspect(spec) == ex._VECTORS_ASPECT

    def test_the_export_sizes_by_aspect_not_a_fixed_height(self, tem_2d_dataset,
                                                           tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None,
                            {"source_window_id": _signal_window_id(session)})

        path = str(tmp_path / "aspect.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "interactive", "path": path})
        assert _exported(messages, session)
        html = open(path, encoding="utf-8").read()
        assert "aspect-ratio:" in html
        assert "height:480px" not in html

    def test_the_ratios_match_the_sidebar_component(self):
        # The renderer has its own copy (ReportFigureCell.tsx) and the two must
        # agree, or the export silently disagrees with the thing it is a copy of.
        # PARSE the TSX rather than trust a comment: the caret-defaults trap was
        # exactly a TSX value drifting from its Python twin and winning silently.
        import re
        from pathlib import Path

        tsx = (Path(__file__).resolve().parents[3] / "electron" / "src"
               / "renderer" / "src" / "components" / "ReportFigureCell.tsx")
        src = tsx.read_text(encoding="utf-8")

        def ratio(pattern: str) -> float:
            m = re.search(pattern, src)
            assert m, f"could not find {pattern!r} in ReportFigureCell.tsx"
            return int(m.group(1)) / int(m.group(2))

        assert ratio(r"PANEL_ASPECT\s*=\s*(\d+)\s*/\s*(\d+)") == ex._PANEL_ASPECT
        assert ratio(r"return\s+(\d+)\s*/\s*(\d+)\s*\n\s*\}\s*\n\s*const layout") \
            == ex._VECTORS_ASPECT
        assert ratio(r"layout\.kind !== 'grid'\) return (\d+) / (\d+)") \
            == ex._DEFAULT_ASPECT

    def test_the_guard_would_catch_a_drift(self):
        # The check above is only worth having if it fails on a changed value.
        import re
        src = "const PANEL_ASPECT = 5 / 3"
        m = re.search(r"PANEL_ASPECT\s*=\s*(\d+)\s*/\s*(\d+)", src)
        assert m and int(m.group(1)) / int(m.group(2)) != ex._PANEL_ASPECT

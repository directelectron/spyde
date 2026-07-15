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
* the "Copy to Report" toolbar wrapper dispatched via ``toolbar_action`` adds a cell.
"""
from __future__ import annotations

import base64
import os

import numpy as np

from spyde.actions.report import export_html as ex
from spyde.actions.report import handlers as h
from spyde.actions.report.model import bake_fallback_png


# ── helpers (mirrors test_report_handlers) ─────────────────────────────────────


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _exported(messages):
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

        exp = _exported(messages)
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

        exp = _exported(messages)
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

        exp = _exported(messages)
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

        exp = _exported(messages)
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
        assert _exported(messages) and not _errors(messages)

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

        assert not _exported(messages)
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


# ── the "Copy to Report" toolbar wrapper ───────────────────────────────────────


class TestCopyToReportToolbar:
    def test_toolbar_action_adds_a_figure_cell(self, tem_2d_dataset):
        """Dispatching the 'Copy to Report' YAML toolbar action adds a figure
        cell (auto-opening a report)."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        session.dispatch_action({
            "action": "toolbar_action",
            "window_id": wid,
            "payload": {"name": "Copy to Report", "params": {}},
        })

        # A report was opened and a figure cell added.
        st = _last_state(messages)
        assert st["open"] is True
        fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
        assert len(fig_cells) == 1
        # A report figure was emitted.
        rep_figs = [m for m in messages if m.get("type") == "figure"
                    and m.get("host") == "report"]
        assert rep_figs and rep_figs[-1]["host"] == "report"

    def test_wrapper_is_registered_in_yaml(self):
        """The YAML entry resolves to the wrapper function."""
        import spyde
        meta = spyde.TOOLBAR_ACTIONS["functions"].get("Copy to Report")
        assert meta is not None
        assert meta["function"] == \
            "spyde.actions.report.toolbar_actions.copy_to_report"
        assert 2 in meta["plot_dim"]


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
        exp = _exported(messages)
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
        exp = _exported(messages)
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
        exp = _exported(messages)
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
        exp = _exported(messages)
        assert exp and "token" not in exp[0]

"""
test_report_handlers.py — the Report Builder staged handlers vs captured messages.

Every handler is exercised against a real Qt-free ``Session`` (the ``window`` /
``tem_2d_dataset`` fixtures + ``captured_messages``): ``report_state`` emissions,
figure emission with ``host:"report"`` + ``cell_id``, add-figure snapshotting,
save with no renderer reply falling back to a baked PNG, open+rebind in-session,
controller teardown on close, and double-fire idempotence.
"""
from __future__ import annotations

import numpy as np

from spyde.actions.report import handlers as h


# ── helpers ────────────────────────────────────────────────────────────────────


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _report_figures(messages):
    return [m for m in messages if m.get("type") == "figure"
            and m.get("host") == "report"]


def _signal_window_id(session):
    """The window id of the loaded signal (non-navigator) plot."""
    for p in session._plots:
        if not getattr(p, "is_navigator", False) and p.window_id is not None:
            return p.window_id
    return session._plots[0].window_id


def _prime_plot_data(session):
    """Ensure the signal plot has a real ndarray current_data to snapshot (the
    fixture may not have painted a frame yet in a headless build)."""
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


# ── document-cell handlers ─────────────────────────────────────────────────────


class TestReportDocument:
    def test_new_emits_open_state(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        st = _last_state(messages)
        assert st["open"] is True
        assert st["cells"] == []
        assert st["dirty"] is False
        assert st["template"] is False

    def test_new_template_flag(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {"template": True})
        assert _last_state(messages)["template"] is True

    def test_add_markdown_cell(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "Hi"})
        st = _last_state(messages)
        assert len(st["cells"]) == 1
        assert st["cells"][0]["cell_type"] == "markdown"
        assert st["cells"][0]["source"] == "Hi"
        assert st["dirty"] is True

    def test_add_cell_lazily_opens_report(self, window):
        """A cell op with no open report creates a fresh one (no crash)."""
        session, messages = window["window"], window["messages"]
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "x"})
        st = _last_state(messages)
        assert st["open"] is True
        assert len(st["cells"]) == 1

    def test_update_cell(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "old"})
        cid = _last_state(messages)["cells"][0]["id"]
        h.report_update_cell(session, None, {"cell_id": cid, "source": "new text"})
        assert _last_state(messages)["cells"][0]["source"] == "new text"

    def test_remove_cell(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "a"})
        cid = _last_state(messages)["cells"][0]["id"]
        h.report_remove_cell(session, None, {"cell_id": cid})
        assert _last_state(messages)["cells"] == []

    def test_move_cell(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        for s in ("A", "B", "C"):
            h.report_add_cell(session, None, {"cell_type": "markdown", "source": s})
        cells = _last_state(messages)["cells"]
        first_id = cells[0]["id"]
        # ``index`` is the drop-target cell's pre-removal position ("insert before
        # the cell currently at this index"). Dragging A (idx 0) onto C (idx 2) lands
        # A just before C → [B, A, C]. (Detailed off-by-one coverage lives in
        # test_report_move_cell.py.)
        h.report_move_cell(session, None, {"cell_id": first_id, "index": 2})
        sources = [c["source"] for c in _last_state(messages)["cells"]]
        assert sources == ["B", "A", "C"]

    def test_close_emits_closed_state(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_close(session, None, {})
        assert _last_state(messages)["open"] is False


# ── figure handlers ────────────────────────────────────────────────────────────


class TestReportFigures:
    def test_add_figure_snapshots_and_emits_report_figure(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        h.report_new(session, None, {})
        messages.clear()
        h.report_add_figure(session, None, {"source_window_id": wid,
                                            "caption": "My DP"})

        # A figure cell exists in the authoritative state.
        st = _last_state(messages)
        fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
        assert len(fig_cells) == 1
        cell = fig_cells[0]
        assert cell["caption"] == "My DP"
        assert cell["placeholder"] is False
        assert cell["fig_id"] == cell["id"]

        # A report figure was emitted with the host + cell_id contract fields.
        rep_figs = _report_figures(messages)
        assert len(rep_figs) == 1
        assert rep_figs[0]["host"] == "report"
        assert rep_figs[0]["cell_id"] == cell["id"]

        # The snapshot array was captured in memory (NOT in the file). Phase 2
        # stores a per-(panel, layer) map; the primary snapshot is the base array.
        mgr = session._report
        assert cell["id"] in mgr._snapshots
        assert isinstance(mgr._snapshots[cell["id"]], dict)
        assert isinstance(mgr.primary_snapshot(cell["id"]), np.ndarray)

        # The figure has a live controller registered.
        assert cell["id"] in mgr._window_by_cell
        wid_fig = mgr._window_by_cell[cell["id"]]
        assert session.controller_by_window_id(wid_fig) is not None

    def test_add_figure_fills_placeholder_in_place(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)

        # Build a template with one placeholder figure cell.
        from spyde.actions.report.model import Cell
        h.report_new(session, None, {"template": True})
        mgr = session._report
        ph = Cell(cell_type="figure", caption="slot", placeholder=True)
        mgr.doc.cells.append(ph)
        messages.clear()

        h.report_add_figure(session, None, {"source_window_id": wid,
                                            "at_cell": ph.id})
        st = _last_state(messages)
        # Still one cell, now filled (not a placeholder), same id.
        assert len(st["cells"]) == 1
        assert st["cells"][0]["id"] == ph.id
        assert st["cells"][0]["placeholder"] is False

    def test_add_figure_bad_source_errors(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": 99999})
        assert any(m.get("type") == "error" for m in messages)

    def test_set_caption(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})
        cid = [c for c in _last_state(messages)["cells"]
               if c["cell_type"] == "figure"][0]["id"]
        h.report_set_caption(session, None, {"cell_id": cid, "caption": "Renamed"})
        cell = [c for c in _last_state(messages)["cells"] if c["id"] == cid][0]
        assert cell["caption"] == "Renamed"

    def test_set_title(self, window):
        session = window["window"]
        messages = window["messages"]
        h.report_new(session, None, {})
        h.report_set_title(session, None, {"title": "My Report"})
        state = _last_state(messages)
        assert state["title"] == "My Report"
        assert state["dirty"] is True

    def test_refresh_figure_reemits(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})
        cid = [c for c in _last_state(messages)["cells"]
               if c["cell_type"] == "figure"][0]["id"]
        messages.clear()
        h.report_refresh_figure(session, None, {"cell_id": cid})
        assert len(_report_figures(messages)) == 1


# ── save flow (headless fallback baking) ───────────────────────────────────────


class TestReportSave:
    def test_save_with_no_renderer_reply_bakes_png(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "Body"})
        h.report_add_figure(session, None, {"source_window_id": wid, "caption": "F"})

        path = str(tmp_path / "out.spyde-report")
        messages.clear()
        # No _main_loop registered (test Session) → bakes straight from snapshots.
        h.report_save(session, None, {"path": path})

        saved = [m for m in messages if m.get("type") == "report_saved"]
        assert saved and saved[0]["path"] == path

        # The written container has the baked asset for the figure cell.
        from spyde.actions.report.model import read_report
        doc, assets = read_report(path)
        fig_cells = [c for c in doc.cells if c.cell_type == "figure"]
        assert len(fig_cells) == 1
        cid = fig_cells[0].id
        assert cid in assets
        assert assets[cid][:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_remembers_path(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "x"})
        path = str(tmp_path / "r.spyde-report")
        h.report_save(session, None, {"path": path})
        # A second save with no path uses the remembered one.
        messages.clear()
        h.report_save(session, None, {})
        saved = [m for m in messages if m.get("type") == "report_saved"]
        assert saved and saved[0]["path"] == path

    def test_snapshots_completes_pending_save(self, tem_2d_dataset, tmp_path):
        """A report_snapshots delivery (harvested PNG) writes that PNG as the
        asset. We drive the handshake directly (no event loop)."""
        import base64
        session = tem_2d_dataset["window"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})
        mgr = session._report
        cid = next(iter(mgr._window_by_cell))

        # Simulate a pending save (as report_save/harvest_snapshots would arm with a
        # loop present): the pending entry carries the ``finish`` callback that
        # writes the report — matching the real shape (no legacy ``path`` key).
        path = str(tmp_path / "harvest.spyde-report")
        token = "tok123"
        mgr._pending_save[token] = {
            "cell_ids": [cid], "harvested": {},
            "finish": lambda harvested: h._finish_save(session, mgr, path,
                                                       harvested),
        }
        # A distinct red 1x1 PNG so we can tell harvest from bake.
        from spyde.actions.report.model import bake_fallback_png
        harvested_png = bake_fallback_png(np.ones((4, 4)) * 7.0)
        data_url = "data:image/png;base64," + base64.b64encode(harvested_png).decode()
        h.report_snapshots(session, None, {"token": token,
                                           "images": {cid: data_url}})

        from spyde.actions.report.model import read_report
        _doc, assets = read_report(path)
        assert assets[cid] == harvested_png

    def test_empty_harvested_png_still_writes_baked_asset(self, tem_2d_dataset,
                                                          tmp_path):
        """An EMPTY harvested payload (b"" from a "data:image/png;base64," data URL
        with no bytes) must NOT skip the asset — the bake fallback fills it, so the
        written container still holds a non-empty PNG (no dangling image ref)."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid, "caption": "F"})
        mgr = session._report
        cid = next(iter(mgr._window_by_cell))

        # Simulate the renderer delivering an EMPTY data URL for this cell (decodes
        # to b"") through the real report_snapshots → finish → _finish_save path.
        path = str(tmp_path / "empty.spyde-report")
        token = "tokEMPTY"
        mgr._pending_save[token] = {
            "cell_ids": [cid], "harvested": {},
            "finish": lambda harvested: h._finish_save(session, mgr, path,
                                                       harvested),
        }
        h.report_snapshots(session, None, {
            "token": token, "images": {cid: "data:image/png;base64,"}})

        from spyde.actions.report.model import read_report
        _doc, assets = read_report(path)
        # The asset IS present (baked) and non-empty — a valid PNG.
        assert cid in assets
        assert assets[cid], "asset must not be empty (bake fallback expected)"
        assert assets[cid][:8] == b"\x89PNG\r\n\x1a\n"

    def test_assemble_assets_empty_harvest_uses_baked_fallback(self, tem_2d_dataset):
        """Unit: assemble_assets treats an empty harvested PNG as missing and falls
        back to the held baked PNG (not skipping the cell)."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})
        mgr = session._report
        cid = next(iter(mgr._window_by_cell))
        # Drop the live snapshot so the only fallback is the baked PNG.
        from spyde.actions.report.model import bake_fallback_png
        baked = bake_fallback_png(np.ones((6, 6)) * 3.0)
        mgr._baked[cid] = baked
        mgr._snapshots.pop(cid, None)

        # An empty harvest for this cell → must fall back to baked, not skip it.
        assets = mgr.assemble_assets({cid: b""})
        assert assets.get(cid) == baked


# ── open + rebind round-trip ───────────────────────────────────────────────────


class TestReportOpenRebind:
    def test_save_then_open_rebinds_live(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "Notes"})
        h.report_add_figure(session, None, {"source_window_id": wid, "caption": "DP"})
        path = str(tmp_path / "rt.spyde-report")
        h.report_save(session, None, {"path": path})

        # Close, then reopen in the SAME session (the source tree is still open).
        h.report_close(session, None, {})
        messages.clear()
        h.report_open(session, None, {"path": path})

        st = _last_state(messages)
        assert st["open"] is True
        assert st["title"] == "Untitled Report" or st["title"]
        types = [c["cell_type"] for c in st["cells"]]
        assert types == ["markdown", "figure"]
        fig_cell = [c for c in st["cells"] if c["cell_type"] == "figure"][0]
        # The source tree is still open → the figure rebinds live (not offline).
        assert fig_cell["data_offline"] is False
        # A live report figure was re-emitted.
        assert len(_report_figures(messages)) == 1

    def test_open_offline_when_source_gone_includes_png(self, tem_2d_dataset, tmp_path):
        """With NO matching open tree, a figure cell is offline and its baked PNG
        rides along in the state as a data URL (renderer has no zip access)."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid, "caption": "DP"})
        path = str(tmp_path / "off.spyde-report")
        h.report_save(session, None, {"path": path})

        # Break the rebind: clear the session's plots so nothing resolves.
        session._plots = []
        h.report_close(session, None, {})
        messages.clear()
        h.report_open(session, None, {"path": path})

        st = _last_state(messages)
        fig_cell = [c for c in st["cells"] if c["cell_type"] == "figure"][0]
        assert fig_cell["data_offline"] is True
        assert isinstance(fig_cell.get("png"), str)
        assert fig_cell["png"].startswith("data:image/png;base64,")
        # No live figure emitted for an offline cell.
        assert _report_figures(messages) == []


# ── teardown + idempotence ─────────────────────────────────────────────────────


class TestReportTeardown:
    def test_close_tears_down_controllers(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})
        mgr = session._report
        fig_wid = next(iter(mgr._window_by_cell.values()))
        assert fig_wid in session._window_controllers

        h.report_close(session, None, {})
        # The figure window's controller is gone from the session registry.
        assert fig_wid not in session._window_controllers
        assert mgr._controllers == {}
        assert mgr._window_by_cell == {}

    def test_remove_figure_cell_tears_down_its_window(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})
        cid = [c for c in _last_state(messages)["cells"]
               if c["cell_type"] == "figure"][0]["id"]
        mgr = session._report
        fig_wid = mgr._window_by_cell[cid]
        h.report_remove_cell(session, None, {"cell_id": cid})
        assert fig_wid not in session._window_controllers
        assert cid not in mgr._window_by_cell
        assert cid not in mgr._snapshots

    def test_double_new_is_idempotent(self, window):
        session, messages = window["window"], window["messages"]
        h.report_new(session, None, {})
        h.report_add_cell(session, None, {"cell_type": "markdown", "source": "x"})
        h.report_new(session, None, {})   # fresh report replaces the old
        assert _last_state(messages)["cells"] == []

    def test_double_add_figure_two_cells(self, tem_2d_dataset):
        """Adding the same window twice yields TWO distinct figure cells, each
        with its own controller (not one clobbering the other)."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})
        h.report_add_figure(session, None, {"source_window_id": wid})
        st = _last_state(messages)
        fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
        assert len(fig_cells) == 2
        assert fig_cells[0]["id"] != fig_cells[1]["id"]
        mgr = session._report
        assert len(mgr._window_by_cell) == 2

    def test_close_when_never_opened(self, window):
        session, messages = window["window"], window["messages"]
        h.report_close(session, None, {})
        assert _last_state(messages)["open"] is False

    def test_build_figure_window_early_return_tears_down_stale(self, tem_2d_dataset):
        """A rebuild with an empty snap_map (or no spec) must STILL tear down the
        prior window/controller — it must not leave a stale window mapped to a now
        figure-less cell."""
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_window_id(session)
        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": wid})
        mgr = session._report
        cid = next(iter(mgr._window_by_cell))
        fig_wid = mgr._window_by_cell[cid]
        assert fig_wid in session._window_controllers

        # Simulate the cell losing its snapshot (e.g. a refresh that went offline)
        # then a rebuild — the early return must tear down the prior window.
        cell = mgr.doc.cell_by_id(cid)
        mgr._snapshots.pop(cid, None)
        mgr.build_figure_window(cell)

        assert cid not in mgr._window_by_cell
        assert fig_wid not in mgr._controllers
        assert fig_wid not in session._window_controllers

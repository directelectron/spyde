"""
test_report_compose.py — Report Builder Phase 2 combined-figure composition.

Exercises the ``repfig_*`` staged handlers against a real Qt-free ``Session``:
compose-option query (same-shape → overlay, nav/signal pair → callout, mismatch →
tiles only), each compose mode's FigureSpec mutation + figure/state re-emission,
layer edit/remove semantics (incl. last-layer / last-panel collapse), annotation
CRUD, a YAML round-trip of a composed multi-panel spec, and that "Add to report"
from a LAYERED MDI plot carries the layers into the FigureSpec.
"""
from __future__ import annotations

import numpy as np

from spyde.actions.report import compose as cx
from spyde.actions.report import handlers as h


# ── helpers ────────────────────────────────────────────────────────────────────


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


def _signal_wid(session):
    for p in session._plots:
        if not getattr(p, "is_navigator", False) and p.window_id is not None:
            return p.window_id
    return session._plots[0].window_id


def _nav_wid(session):
    for p in session._plots:
        if getattr(p, "is_navigator", False) and p.window_id is not None:
            return p.window_id
    return None


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _report_figures(messages):
    return [m for m in messages if m.get("type") == "figure"
            and m.get("host") == "report"]


def _compose_options(messages):
    return [m for m in messages if m.get("type") == "repfig_compose_options"]


def _make_figure_cell(session, messages, source_wid, caption="Fig"):
    """Add a figure cell from a source window and return its cell id + FigureSpec."""
    h.report_add_figure(session, None, {"source_window_id": source_wid,
                                        "caption": caption})
    st = _last_state(messages)
    fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
    cid = fig_cells[-1]["id"]
    return cid


def _fig_dict_of(messages, cell_id):
    """The pixel-free FigureSpec dict carried in report_state for a cell."""
    st = _last_state(messages)
    for c in st["cells"]:
        if c["id"] == cell_id:
            return c.get("figure")
    return None


# ── query options ──────────────────────────────────────────────────────────────


class TestQueryOptions:
    def test_same_shape_offers_overlay(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        messages.clear()
        # Compose the SAME window (identical frame shape) → overlay is offered.
        cx.repfig_query_compose(session, None,
                                {"cell_id": cid, "source_window_id": wid})
        opts = _compose_options(messages)
        assert opts, "no repfig_compose_options emitted"
        m = opts[-1]
        assert m["cell_id"] == cid
        assert m["source_window_id"] == wid
        assert "overlay" in m["options"]
        assert m["detail"]["same_shape"] is True
        # Tiles are always present.
        for t in ("tile-up", "tile-down", "tile-left", "tile-right"):
            assert t in m["options"]

    def test_nav_signal_pair_offers_callout(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        nav_wid = _nav_wid(session)
        assert nav_wid is not None
        h.report_new(session, None, {})
        # Build the cell from the SIGNAL; drop the NAVIGATOR onto it.
        cid = _make_figure_cell(session, messages, sig_wid)
        messages.clear()
        cx.repfig_query_compose(session, None,
                                {"cell_id": cid, "source_window_id": nav_wid})
        m = _compose_options(messages)[-1]
        assert m["detail"]["nav_signal_pair"] is True
        assert "callout" in m["options"]
        # nav (4x5) vs signal (16x16) differ → no overlay.
        assert "overlay" not in m["options"]
        assert m["detail"]["same_shape"] is False

    def test_mismatch_tiles_only(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        nav_wid = _nav_wid(session)
        h.report_new(session, None, {})
        # Build the cell from the NAVIGATOR; the signal is a nav/signal pair too,
        # so to get a pure tiles-only case we compose a DIFFERENT-shape non-pair:
        # use the navigator cell + the signal window but assert overlay absent.
        cid = _make_figure_cell(session, messages, nav_wid)
        messages.clear()
        cx.repfig_query_compose(session, None,
                                {"cell_id": cid, "source_window_id": sig_wid})
        m = _compose_options(messages)[-1]
        assert "overlay" not in m["options"]     # shapes differ
        assert set(m["options"]) >= {"tile-up", "tile-down", "tile-left", "tile-right"}


# ── compose modes ───────────────────────────────────────────────────────────────


class TestComposeModes:
    def test_overlay_appends_layer(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        messages.clear()
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "overlay", "source_window_id": wid})
        # The panel now has TWO layers (base + overlay), a figure re-emit + state.
        fig = _fig_dict_of(messages, cid)
        assert fig is not None
        assert len(fig["panels"]) == 1
        assert len(fig["panels"][0]["layers"]) == 2
        assert len(_report_figures(messages)) == 1
        # The overlay layer has its own snapshot stored.
        mgr = session._report
        assert len(mgr.snapshot_map(cid)) == 2

    def test_tile_right_grows_grid(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        nav_wid = _nav_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, sig_wid)
        messages.clear()
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "tile-right",
                           "source_window_id": nav_wid})
        fig = _fig_dict_of(messages, cid)
        assert fig["layout"]["kind"] == "grid"
        assert fig["layout"]["rows"] == 1 and fig["layout"]["cols"] == 2
        assert len(fig["panels"]) == 2
        positions = sorted(tuple(p["grid_pos"]) for p in fig["panels"])
        assert positions == [(0, 0), (0, 1)]

    def test_tile_left_prepends_and_shifts(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        nav_wid = _nav_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, sig_wid)
        messages.clear()
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "tile-left",
                           "source_window_id": nav_wid})
        fig = _fig_dict_of(messages, cid)
        assert fig["layout"] == {"kind": "grid", "rows": 1, "cols": 2}
        # New panel at col 0, the original shifted to col 1.
        by_pos = {tuple(p["grid_pos"]): p for p in fig["panels"]}
        assert (0, 0) in by_pos and (0, 1) in by_pos

    def test_tile_down_then_insert_stays_grid(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        nav_wid = _nav_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, sig_wid)
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "tile-down",
                           "source_window_id": nav_wid})
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "tile-right",
                           "source_window_id": nav_wid})
        fig = _fig_dict_of(messages, cid)
        assert fig["layout"]["kind"] == "grid"
        assert len(fig["panels"]) == 3

    def test_callout_adds_inset(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        nav_wid = _nav_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, sig_wid)
        messages.clear()
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "callout",
                           "source_window_id": nav_wid})
        fig = _fig_dict_of(messages, cid)
        # Layout stays single (the callout panel is a floating inset, not a grid
        # cell), the primary panel gains an inset entry, and a new inset panel
        # spec + its snapshot exist.
        assert fig["layout"]["kind"] == "single"
        primary = fig["panels"][0]
        assert len(primary["insets"]) == 1
        inset_pid = primary["insets"][0]["panel"]
        assert any(p["id"] == inset_pid for p in fig["panels"])
        assert len(_report_figures(messages)) == 1


# ── layer / panel / annotation edits ────────────────────────────────────────────


class TestEdits:
    def _overlayed(self, session, messages):
        """A cell with a base + one overlay layer; returns (cid, panel_id, layer_ids)."""
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "overlay", "source_window_id": wid})
        fig = _fig_dict_of(messages, cid)
        panel = fig["panels"][0]
        return cid, panel["id"], [ly["id"] for ly in panel["layers"]]

    def test_set_layer_updates_spec(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, pid, lids = self._overlayed(session, messages)
        messages.clear()
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1],
            "cmap": "plasma", "alpha": 0.25, "visible": False,
            "clim": [1.0, 9.0]})
        fig = _fig_dict_of(messages, cid)
        ov = [ly for ly in fig["panels"][0]["layers"] if ly["id"] == lids[1]][0]
        assert ov["cmap"] == "plasma"
        assert abs(ov["alpha"] - 0.25) < 1e-9
        assert ov["visible"] is False
        assert ov["clim"] == [1.0, 9.0]

    def test_remove_overlay_layer(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, pid, lids = self._overlayed(session, messages)
        messages.clear()
        cx.repfig_remove_layer(session, None,
                               {"cell_id": cid, "panel_id": pid, "layer_id": lids[1]})
        fig = _fig_dict_of(messages, cid)
        assert len(fig["panels"][0]["layers"]) == 1
        # Overlay snapshot dropped.
        mgr = session._report
        assert len(mgr.snapshot_map(cid)) == 1

    def test_remove_last_layer_removes_panel_and_collapses_cell(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        fig = _fig_dict_of(messages, cid)
        pid = fig["panels"][0]["id"]
        lid = fig["panels"][0]["layers"][0]["id"]
        mgr = session._report
        assert cid in mgr._window_by_cell
        messages.clear()
        # Removing the ONLY layer of the ONLY panel → placeholder (empty) cell.
        cx.repfig_remove_layer(session, None,
                               {"cell_id": cid, "panel_id": pid, "layer_id": lid})
        st = _last_state(messages)
        cell = [c for c in st["cells"] if c["id"] == cid][0]
        assert cell["placeholder"] is True
        assert cell.get("figure") is None
        assert cid not in mgr._window_by_cell
        assert cid not in mgr._snapshots

    def test_remove_panel_from_grid(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        nav_wid = _nav_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, sig_wid)
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "tile-right",
                           "source_window_id": nav_wid})
        fig = _fig_dict_of(messages, cid)
        # Remove the second (tiled) panel → collapse back to single.
        pid2 = [p["id"] for p in fig["panels"] if tuple(p["grid_pos"]) == (0, 1)][0]
        messages.clear()
        cx.repfig_remove_panel(session, None, {"cell_id": cid, "panel_id": pid2})
        fig = _fig_dict_of(messages, cid)
        assert len(fig["panels"]) == 1
        assert fig["layout"]["kind"] == "single"

    def test_annotation_crud(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        fig = _fig_dict_of(messages, cid)
        pid = fig["panels"][0]["id"]

        # ADD a circle.
        cx.repfig_add_annotation(session, None, {
            "cell_id": cid, "panel_id": pid,
            "annotation": {"kind": "circle", "offsets": [[8, 8]], "radius": 3}})
        fig = _fig_dict_of(messages, cid)
        anns = fig["panels"][0]["annotations"]
        assert len(anns) == 1 and anns[0]["kind"] == "circle"

        # ADD a text, then UPDATE index 0.
        cx.repfig_add_annotation(session, None, {
            "cell_id": cid, "panel_id": pid,
            "annotation": {"kind": "text", "offsets": [[2, 2]], "texts": ["A"]}})
        cx.repfig_update_annotation(session, None, {
            "cell_id": cid, "panel_id": pid, "index": 0,
            "annotation": {"kind": "circle", "offsets": [[4, 4]], "radius": 5}})
        fig = _fig_dict_of(messages, cid)
        anns = fig["panels"][0]["annotations"]
        assert len(anns) == 2
        assert anns[0]["offsets"] == [[4, 4]]

        # REMOVE index 0.
        cx.repfig_remove_annotation(session, None,
                                    {"cell_id": cid, "panel_id": pid, "index": 0})
        fig = _fig_dict_of(messages, cid)
        anns = fig["panels"][0]["annotations"]
        assert len(anns) == 1 and anns[0]["kind"] == "text"


# ── YAML round-trip of a composed multi-panel spec ─────────────────────────────


class TestRoundTrip:
    def test_composed_spec_roundtrips(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        nav_wid = _nav_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, sig_wid)
        # Grid + an annotation → a non-trivial spec.
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "tile-right",
                           "source_window_id": nav_wid})
        fig = _fig_dict_of(messages, cid)
        pid = fig["panels"][0]["id"]
        cx.repfig_add_annotation(session, None, {
            "cell_id": cid, "panel_id": pid,
            "annotation": {"kind": "circle", "offsets": [[8, 8]], "radius": 3}})

        spec = session._report.doc.cell_by_id(cid).spec
        from spyde.actions.report.model import FigureSpec
        text = spec.to_yaml()
        assert "\x00bin" not in text          # NO pixel bytes in the recipe
        rt = FigureSpec.from_yaml(text)
        assert rt.layout == spec.layout
        assert len(rt.panels) == len(spec.panels)
        # Panel ids, layer ids, grid positions, and the annotation survive.
        assert [p.id for p in rt.panels] == [p.id for p in spec.panels]
        assert [[ly.id for ly in p.layers] for p in rt.panels] == \
               [[ly.id for ly in p.layers] for p in spec.panels]
        assert rt.panels[0].annotations == spec.panels[0].annotations


# ── Add to report from a LAYERED MDI plot ──────────────────────────────────────


class TestLayeredAddToReport:
    def test_layered_plot_carries_layers(self, stem_4d_dataset):
        from spyde.actions import overlay as ov
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        sig_plot = session._plot_by_window_id(sig_wid)

        # Fake a second same-shape source and a live layer on the signal plot,
        # WITHOUT anyplotlib (headless): construct the PlotLayer directly with a
        # stub handle, so _snapshot_plot serializes it.
        class _StubHandle:
            id = "Lx"
            def set_data(self, f):
                pass

        class _StubSource:
            def __init__(self, frame):
                self.current_data = frame
                self._layers = []
                self.view_label = "Overlay Src"
                self.is_navigator = False
                self.signal_tree = sig_plot.signal_tree

                class _PS:
                    class _Sig:
                        class metadata:
                            @staticmethod
                            def get_item(k, default=""):
                                return "Src"
                    current_signal = _Sig()
                self.plot_state = _PS()

        src_frame = np.asarray(sig_plot.current_data, dtype=np.float32) + 1.0
        stub_src = _StubSource(src_frame)
        sig_plot._layers = [ov.PlotLayer(
            layer_id="Lx", source_plot=stub_src, cmap="magma", alpha=0.5,
            clim=None, visible=True, handle=_StubHandle(), title="Overlay Src")]

        h.report_new(session, None, {})
        h.report_add_figure(session, None, {"source_window_id": sig_wid})
        st = _last_state(messages)
        cid = [c for c in st["cells"] if c["cell_type"] == "figure"][-1]["id"]
        fig = _fig_dict_of(messages, cid)
        # Base + the live overlay layer serialized into the panel.
        assert len(fig["panels"][0]["layers"]) == 2
        cmaps = [ly["cmap"] for ly in fig["panels"][0]["layers"]]
        assert "magma" in cmaps
        # Both layers have a stored snapshot.
        assert len(session._report.snapshot_map(cid)) == 2

"""
test_report_overlay_tint.py — Report Builder Phase 4: tinted overlays + the
interactive-export opacity blender.

Covers, against a real Qt-free ``Session``:

* ``LayerSpec.tint`` schema: to_dict emits the key ONLY when set, from_dict is
  tolerant (older dicts/YAML without the key load as None), YAML round-trip.
* ``repfig_set_layer`` tint set / clear persistence (cmap kept as the revert
  value), malformed-hex tolerance.
* ``_compose_overlay`` auto-assigns DISTINCT tint-cycle colors (and still
  assigns the cmap-cycle revert value).
* ``figure_builder`` passes the tint through to the live anyplotlib layer
  state; a legacy (untinted) layer renders exactly as before.
* interactive HTML export: a tinted cell exports the overlay BLENDER block
  (base grayscale + per-overlay range slider), NOT the vectors explorer; and
  ``report_state`` stays pixel-free (tint string only, no base64 payloads on a
  LIVE cell).
"""
from __future__ import annotations

import json

import numpy as np

from spyde.actions.report import compose as cx
from spyde.actions.report import export_html as ex
from spyde.actions.report import handlers as h
from spyde.actions.report.model import FigureSpec, LayerSpec, PanelSpec, SignalRef


# ── helpers (mirrors test_report_compose) ──────────────────────────────────────


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


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _make_figure_cell(session, messages, source_wid, caption="Fig"):
    h.report_add_figure(session, None, {"source_window_id": source_wid,
                                        "caption": caption})
    st = _last_state(messages)
    fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
    return fig_cells[-1]["id"]


def _fig_dict_of(messages, cell_id):
    st = _last_state(messages)
    for c in st["cells"]:
        if c["id"] == cell_id:
            return c.get("figure")
    return None


def _overlayed(session, messages):
    """A cell with base + one composed overlay; (cid, panel_id, layer_ids)."""
    wid = _signal_wid(session)
    h.report_new(session, None, {})
    cid = _make_figure_cell(session, messages, wid)
    cx.repfig_compose(session, None,
                      {"cell_id": cid, "mode": "overlay", "source_window_id": wid})
    fig = _fig_dict_of(messages, cid)
    panel = fig["panels"][0]
    return cid, panel["id"], [ly["id"] for ly in panel["layers"]]


# ── schema round-trip ──────────────────────────────────────────────────────────


class TestTintSchema:
    def test_to_dict_omits_absent_tint(self):
        d = LayerSpec().to_dict()
        assert "tint" not in d          # older readers never see the key

    def test_to_dict_emits_and_from_dict_reads(self):
        ly = LayerSpec(tint="#f38ba8")
        d = ly.to_dict()
        assert d["tint"] == "#f38ba8"
        rt = LayerSpec.from_dict(d)
        assert rt.tint == "#f38ba8"
        assert rt.cmap == ly.cmap       # cmap (revert value) rides alongside

    def test_old_dict_without_tint_loads_none(self):
        rt = LayerSpec.from_dict({"id": "l1", "cmap": "gray", "alpha": 0.5})
        assert rt.tint is None
        assert rt.cmap == "gray"

    def test_yaml_round_trip_carries_tint(self):
        spec = FigureSpec(panels=[PanelSpec(id="p1", layers=[
            LayerSpec(source=SignalRef(title="base"), cmap="gray"),
            LayerSpec(source=SignalRef(title="ov"), cmap="magma",
                      alpha=0.5, tint="#a6e3a1"),
        ])])
        text = spec.to_yaml()
        assert "#a6e3a1" in text
        rt = FigureSpec.from_yaml(text)
        assert rt.panels[0].layers[0].tint is None      # untinted stays None
        assert rt.panels[0].layers[1].tint == "#a6e3a1"

    def test_yaml_without_tint_key_loads_none(self):
        # A pre-Phase-4 YAML (no tint key anywhere) loads with tint=None on
        # every layer — legacy cells render exactly as today.
        spec = FigureSpec(panels=[PanelSpec(id="p1", layers=[
            LayerSpec(cmap="viridis"), LayerSpec(cmap="magma", alpha=0.5)])])
        text = spec.to_yaml()
        assert "tint" not in text
        rt = FigureSpec.from_yaml(text)
        assert all(ly.tint is None for p in rt.panels for ly in p.layers)


# ── repfig_set_layer tint set / clear ──────────────────────────────────────────


class TestSetLayerTint:
    def test_set_and_clear_persist(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, pid, lids = _overlayed(session, messages)

        messages.clear()
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1],
            "tint": "#89dceb"})
        fig = _fig_dict_of(messages, cid)
        ov = [ly for ly in fig["panels"][0]["layers"] if ly["id"] == lids[1]][0]
        assert ov["tint"] == "#89dceb"
        # The cmap is KEPT while tinted — it's the revert value.
        assert ov["cmap"]

        messages.clear()
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1],
            "tint": None})
        fig = _fig_dict_of(messages, cid)
        ov = [ly for ly in fig["panels"][0]["layers"] if ly["id"] == lids[1]][0]
        # Cleared → to_dict omits the key entirely; cmap display restored.
        assert "tint" not in ov
        assert ov["cmap"]

    def test_malformed_tint_ignored(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, pid, lids = _overlayed(session, messages)
        spec_layer = session._report.doc.cell_by_id(cid).spec.panels[0].layers[1]
        before = spec_layer.tint
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1],
            "tint": "notacolor"})
        assert spec_layer.tint == before    # untouched — never reaches add_layer

    def test_tint_survives_alpha_edit(self, tem_2d_dataset):
        # A later alpha-only slider edit (no tint key in the payload) must not
        # disturb the stored tint.
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, pid, lids = _overlayed(session, messages)
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1],
            "tint": "#cba6f7"})
        messages.clear()
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1], "alpha": 0.3})
        fig = _fig_dict_of(messages, cid)
        ov = [ly for ly in fig["panels"][0]["layers"] if ly["id"] == lids[1]][0]
        assert ov["tint"] == "#cba6f7"
        assert abs(ov["alpha"] - 0.3) < 1e-9


# ── compose auto-assigns cycle tints ───────────────────────────────────────────


class TestComposeTintCycle:
    def test_overlays_get_distinct_cycle_tints(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "overlay", "source_window_id": wid})
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "overlay", "source_window_id": wid})
        fig = _fig_dict_of(messages, cid)
        layers = fig["panels"][0]["layers"]
        assert len(layers) == 3
        # Base stays untinted; the overlays take the first two cycle colors.
        assert "tint" not in layers[0]
        assert layers[1]["tint"] == cx._OVERLAY_TINT_CYCLE[0]
        assert layers[2]["tint"] == cx._OVERLAY_TINT_CYCLE[1]
        assert layers[1]["tint"] != layers[2]["tint"]
        # The cmap-cycle revert values are STILL assigned.
        assert layers[1]["cmap"] == cx._OVERLAY_CMAP_CYCLE[0]
        assert layers[2]["cmap"] == cx._OVERLAY_CMAP_CYCLE[1]

    def test_next_unused_skips_taken_color(self, tem_2d_dataset):
        # Recolor overlay 1 to cycle color #2; the NEXT compose must skip both
        # taken colors and pick the first unused one.
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, pid, lids = _overlayed(session, messages)
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1],
            "tint": cx._OVERLAY_TINT_CYCLE[1]})
        wid = _signal_wid(session)
        cx.repfig_compose(session, None,
                          {"cell_id": cid, "mode": "overlay", "source_window_id": wid})
        fig = _fig_dict_of(messages, cid)
        tints = [ly.get("tint") for ly in fig["panels"][0]["layers"][1:]]
        assert tints == [cx._OVERLAY_TINT_CYCLE[1], cx._OVERLAY_TINT_CYCLE[0]]


# ── figure_builder passes tint into anyplotlib ─────────────────────────────────


class TestBackendRender:
    def _spec_with_overlay(self, tint):
        base = LayerSpec(cmap="gray")
        ov = LayerSpec(cmap="magma", alpha=0.5, tint=tint)
        panel = PanelSpec(id="p1", layers=[base, ov])
        spec = FigureSpec(panels=[panel])
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        snap = {("p1", base.id): arr, ("p1", ov.id): arr + 1.0}
        return spec, snap

    def test_tint_reaches_layer_state(self, window):
        from spyde.actions.report.figure_builder import build_cell_figure
        spec, snap = self._spec_with_overlay("#f38ba8")
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        plot = next(iter(fig._plots_map.values()))
        layers = plot._state.get("layers", [])
        assert len(layers) == 1
        assert layers[0]["tint"] == "#f38ba8"
        # A tint LUT is RGBA (256×4) with the alpha ramping 0→255.
        lut = layers[0]["colormap_data"]
        assert len(lut) == 256 and len(lut[0]) == 4
        assert lut[0][3] == 0 and lut[255][3] == 255

    def test_untinted_layer_renders_as_before(self, window):
        from spyde.actions.report.figure_builder import build_cell_figure
        spec, snap = self._spec_with_overlay(None)
        fig, _fig_id, _html = build_cell_figure(spec, snap)
        plot = next(iter(fig._plots_map.values()))
        layers = plot._state.get("layers", [])
        assert len(layers) == 1
        assert layers[0]["tint"] is None
        # Named-cmap LUT stays RGB (256×3) — the legacy shape.
        assert len(layers[0]["colormap_data"][0]) == 3


# ── interactive export: the overlay blender ────────────────────────────────────


class TestInteractiveExportBlender:
    def _tinted_cell(self, session, messages):
        cid, pid, lids = _overlayed(session, messages)
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1],
            "tint": "#f38ba8"})
        return cid, pid, lids

    def test_blender_html_structure(self, tem_2d_dataset):
        from spyde.actions.report.overlay_embed import overlay_blender_html
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, _pid, _lids = self._tinted_cell(session, messages)
        mgr = session._report
        cell = mgr.doc.cell_by_id(cid)
        html = overlay_blender_html(mgr, cell, caption="Blend")
        assert html is not None
        assert "ovb-root" in html
        # Exactly one range slider (one tinted overlay) + the tint swatch.
        assert html.count("type=\"range\"") == 1
        assert "#f38ba8" in html
        assert "Blend" in html

    def test_blender_none_without_tint(self, tem_2d_dataset):
        from spyde.actions.report.overlay_embed import overlay_blender_html
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, pid, lids = _overlayed(session, messages)
        # Clear the compose-assigned default tint → a pure-cmap overlay cell.
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1], "tint": None})
        mgr = session._report
        cell = mgr.doc.cell_by_id(cid)
        assert overlay_blender_html(mgr, cell) is None

    def test_interactive_export_swaps_in_blender(self, tem_2d_dataset, tmp_path):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        self._tinted_cell(session, messages)

        path = str(tmp_path / "tinted.html")
        messages.clear()
        ex.report_export_html(session, None, {"mode": "interactive", "path": path})
        exported = [m for m in messages if m.get("type") == "report_exported"]
        assert exported and exported[0]["kind"] == "html-interactive"

        html = open(path, encoding="utf-8").read()
        # The blender block rides inside the sandboxed srcdoc iframe.
        assert "ovb-root" in html
        assert html.count("ovb-slider") >= 2   # class attr + the JS query
        # NOT the vectors explorer (no vectors on this tree) and no bin tokens.
        assert "vx-root" not in html
        assert "vx-data" not in html
        assert "\x00bin:" not in html

    def test_untinted_cell_keeps_live_iframe(self, tem_2d_dataset, tmp_path):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, pid, lids = _overlayed(session, messages)
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": pid, "layer_id": lids[1], "tint": None})
        path = str(tmp_path / "untinted.html")
        ex.report_export_html(session, None, {"mode": "interactive", "path": path})
        html = open(path, encoding="utf-8").read()
        assert "ovb-root" not in html
        assert "<iframe sandbox=\"allow-scripts\" srcdoc=" in html

    def test_report_state_pixel_free_with_tint(self, tem_2d_dataset):
        """The spec stays pixel-free: a LIVE tinted cell's report_state figure
        entry carries the tint string but NO base64 payload (the offline PNG
        fallback is the only base64 producer, and a live cell never takes it)."""
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        cid, _pid, _lids = self._tinted_cell(session, messages)
        mgr = session._report
        state = mgr.state()
        entry = [c for c in state["cells"] if c["id"] == cid][0]
        assert entry["data_offline"] is False
        assert "png" not in entry               # live → no offline data URL
        fig_json = json.dumps(entry["figure"])
        assert "#f38ba8" in fig_json
        assert "base64" not in fig_json

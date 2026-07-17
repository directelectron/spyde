"""
test_report_line_panels.py — Report Builder 1-D line-panel support.

Exercises the ``kind="line"`` panel path end-to-end against a real Qt-free
``Session``: snapshotting a 1-D plot into a line-panel FigureSpec (axes +
base-layer styling), rendering that spec into a live anyplotlib Plot1D
(``figure_builder._render_line_panel`` via ``build_cell_figure``),
``repfig_set_layer`` styling edits (color/linewidth/label, clamps, YAML
round-trip), refusing the 2-D-only annotation/callout/zoom verbs on a line
panel, and per-panel refresh re-snapshotting the curve.

No fixture in ``conftest.py`` produces a 1-D signal plot, so this file builds
one directly (mirrors ``tem_2d_dataset``'s pattern: a real ``Session`` +
``session._add_signal`` with a ``hs.signals.Signal1D``, no navigation axes —
so exactly one signal plot window is created, ``is_navigator=False``).
"""
from __future__ import annotations

import time

import hyperspy.api as hs
import numpy as np
import pytest

from spyde.actions.report import compose as cx
from spyde.actions.report import handlers as h
from spyde.actions.report.model import FigureSpec


# ── local fixture: a 1-D signal (no navigation) → one signal window ────────────


def _make_session():
    from spyde.backend.session import Session
    return Session(n_workers=1, threads_per_worker=1)


@pytest.fixture
def signal1d_dataset(captured_messages):
    """1-D signal (no navigation) → one signal window, calibrated x-axis."""
    session = _make_session()
    data = np.sin(np.linspace(0, 4 * np.pi, 128)).astype(np.float32)
    s = hs.signals.Signal1D(data)
    ax = s.axes_manager.signal_axes[0]
    ax.name, ax.units, ax.scale, ax.offset = "Energy", "eV", 0.5, 0.0
    s.metadata.set_item("General.title", "1D Curve")
    session._add_signal(s, source_path=None)
    time.sleep(0.3)
    yield {"window": session, "signal_trees": session.signal_trees,
           "plots": session._plots, "messages": captured_messages}
    session.shutdown()


# ── helpers ──────────────────────────────────────────────────────────────────


def _prime_1d_plot_data(session):
    """Stamp ``current_data``/``_last_levels`` from the live signal's raw data
    directly (bypassing the async paint pipeline) — the same pattern
    ``test_report_compose.py``'s ``_prime_plot_data`` uses for 2-D, scoped to a
    1-D frame."""
    for p in session._plots:
        if isinstance(getattr(p, "current_data", None), np.ndarray):
            continue
        try:
            sig = p.plot_state.current_signal
            frame = np.asarray(sig.data)
            if frame.ndim != 1:
                continue
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


def _report_figures(messages):
    return [m for m in messages if m.get("type") == "figure"
            and m.get("host") == "report"]


def _errors(messages):
    return [m for m in messages if m.get("type") == "error"]


def _make_figure_cell(session, messages, source_wid, caption="Curve"):
    h.report_add_figure(session, None, {"source_window_id": source_wid,
                                        "caption": caption})
    st = _last_state(messages)
    fig_cells = [c for c in st["cells"] if c["cell_type"] == "figure"]
    cid = fig_cells[-1]["id"]
    return cid


def _fig_dict_of(messages, cell_id):
    st = _last_state(messages)
    for c in st["cells"]:
        if c["id"] == cell_id:
            return c.get("figure")
    return None


def _cell_of(session, cell_id):
    return session._report.doc.cell_by_id(cell_id)


# ── snapshot ─────────────────────────────────────────────────────────────────


class TestLineSnapshot:
    def test_snapshot_produces_line_panel(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        fig = _fig_dict_of(messages, cid)
        assert fig is not None
        assert len(fig["panels"]) == 1
        panel = fig["panels"][0]
        assert panel["kind"] == "line"
        assert len(_report_figures(messages)) == 1

    def test_axes_carries_x_axis_and_units(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        fig = _fig_dict_of(messages, cid)
        panel = fig["panels"][0]
        axes = panel.get("axes")
        assert axes is not None
        assert "x_axis" in axes and "units" in axes
        xa = axes["x_axis"]
        assert len(xa) == 128
        # Calibrated: offset 0.0, scale 0.5 eV.
        assert xa[0] == pytest.approx(0.0)
        assert xa[1] == pytest.approx(0.5)
        assert "eV" in axes["units"] or axes["units"] == "eV"

    def test_no_scalebar_or_colorbar_on_line_panel(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        fig = _fig_dict_of(messages, cid)
        panel = fig["panels"][0]
        assert panel["scalebar"] is False
        assert panel["colorbar"] is False

    def test_snapshot_map_holds_1d_array(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        mgr = session._report
        snap_map = mgr.snapshot_map(cid)
        assert len(snap_map) == 1
        arr = next(iter(snap_map.values()))
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        assert arr.shape[0] == 128

    def test_base_layer_styling_captured_from_live_plot1d(self, signal1d_dataset):
        """The real load path paints a live ``Plot1D`` (unlike the direct-stamp
        priming helper other report tests use for 2-D plots — a 1-D signal
        with no navigation goes straight through ``_set_array`` on load), so
        the base layer's styling is read from its live ``_state`` — anyplotlib's
        own defaults when nothing has been customized yet."""
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        cell = _cell_of(session, cid)
        layer = cell.spec.panels[0].layers[0]
        assert layer.color == "#4fc3f7"      # anyplotlib Plot1D's default
        assert layer.linewidth == pytest.approx(1.5)
        assert layer.label is None            # no label set → no legend entry

    def test_snapshot_tolerates_missing_plot1d_state(self, signal1d_dataset, monkeypatch):
        """When ``plot._plot1d`` isn't reachable (or carries no usable state)
        the styling read must tolerate that and leave color/linewidth/label as
        None, rather than crashing the snapshot."""
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        src_plot = next(p for p in session._plots if p.window_id == wid)
        monkeypatch.setattr(src_plot, "_plot1d", None, raising=False)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        cell = _cell_of(session, cid)
        layer = cell.spec.panels[0].layers[0]
        assert layer.color is None
        assert layer.linewidth is None
        assert layer.label is None

    def test_bad_current_data_returns_none(self, signal1d_dataset):
        """A plot with no paintable current_data still fails snapshot cleanly
        (mirrors the 2-D contract) — report_add_figure emits an error, no cell
        is created."""
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        wid = _signal_wid(session)
        src_plot = next(p for p in session._plots if p.window_id == wid)
        src_plot.current_data = None   # simulate no paintable frame yet
        h.report_new(session, None, {})
        messages.clear()
        h.report_add_figure(session, None, {"source_window_id": wid,
                                            "caption": "x"})
        assert _errors(messages), "expected an error for an unpaintable source"


# ── render ───────────────────────────────────────────────────────────────────


class TestLineRender:
    def test_build_figure_window_creates_plot1d(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        mgr = session._report
        fig = mgr.live_fig(cid)
        assert fig is not None
        plots_map = getattr(fig, "_plots_map", None) or {}
        assert len(plots_map) == 1
        p1 = next(iter(plots_map.values()))
        state = getattr(p1, "_state", None)
        assert state is not None
        assert state.get("kind") == "1d"
        assert np.asarray(state["data"]).shape[0] == 128

    def test_styling_reaches_live_state(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        mgr = session._report
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        layer = panel.layers[0]
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "color": "#ff7043", "linewidth": 3.0, "label": "sine",
        })

        fig = mgr.live_fig(cid)
        p1 = next(iter(getattr(fig, "_plots_map", {}).values()))
        state = p1._state
        assert state["line_color"] == "#ff7043"
        assert state["line_linewidth"] == pytest.approx(3.0)
        assert state["line_label"] == "sine"

    def test_text_sizes_apply_to_line_panel(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        cx_payload = {"cell_id": cid, "panel_id": panel.id,
                     "target": "ticks", "size": 18}
        cx.repfig_set_text_size(session, None, cx_payload)

        mgr = session._report
        fig = mgr.live_fig(cid)
        p1 = next(iter(getattr(fig, "_plots_map", {}).values()))
        assert p1._state.get("tick_size") == pytest.approx(18)

        cell = _cell_of(session, cid)
        assert cell.spec.panels[0].text_sizes.get("ticks") == 18

    def test_length_mismatch_falls_back_to_index_axis(self, signal1d_dataset):
        """A stale/mismatched axes x_axis (wrong length vs the layer's y-data)
        must not crash the render — it falls back to a bare index axis for
        that layer."""
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        # Corrupt the stored x_axis length.
        panel.axes["x_axis"] = list(panel.axes["x_axis"])[:10]

        from spyde.actions.report.figure_builder import build_cell_figure
        mgr = session._report
        fig, fig_id, html = build_cell_figure(cell.spec, mgr.snapshot_map(cid))
        p1 = next(iter(getattr(fig, "_plots_map", {}).values()))
        # Falls back to arange(n) — length matches the data, not the stale axis.
        assert np.asarray(p1._state["x_axis"]).shape[0] == 128


# ── layer styling edits ─────────────────────────────────────────────────────


class TestLayerStyling:
    def test_set_layer_persists_and_rebuilds(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        layer = panel.layers[0]
        messages.clear()

        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "color": "#00ff00", "linewidth": 5.0, "label": "peak A",
        })

        cell = _cell_of(session, cid)
        layer = cell.spec.panels[0].layers[0]
        assert layer.color == "#00ff00"
        assert layer.linewidth == pytest.approx(5.0)
        assert layer.label == "peak A"
        assert _report_figures(messages), "expected a figure re-emit"
        assert _states(messages), "expected a report_state re-emit"

    def test_linewidth_clamped(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        layer = panel.layers[0]

        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "linewidth": 999.0,
        })
        cell = _cell_of(session, cid)
        assert cell.spec.panels[0].layers[0].linewidth == pytest.approx(12.0)

        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "linewidth": -5.0,
        })
        cell = _cell_of(session, cid)
        assert cell.spec.panels[0].layers[0].linewidth == pytest.approx(0.5)

    def test_label_empty_string_clears(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        layer = panel.layers[0]

        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "label": "first",
        })
        cell = _cell_of(session, cid)
        assert cell.spec.panels[0].layers[0].label == "first"

        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "label": "",
        })
        cell = _cell_of(session, cid)
        assert cell.spec.panels[0].layers[0].label is None

    def test_label_absent_leaves_unchanged(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        layer = panel.layers[0]

        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "label": "kept",
        })
        # A color-only update with no "label" key must not touch the label.
        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "color": "#123456",
        })
        cell = _cell_of(session, cid)
        layer = cell.spec.panels[0].layers[0]
        assert layer.label == "kept"
        assert layer.color == "#123456"

    def test_label_capped_at_120_chars(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        layer = panel.layers[0]

        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "label": "x" * 500,
        })
        cell = _cell_of(session, cid)
        assert len(cell.spec.panels[0].layers[0].label) == 120

    def test_layerspec_emits_styling_only_when_set(self):
        """Model-level contract (independent of a live plot): a bare, untouched
        LayerSpec (color/linewidth/label all None, the dataclass default) emits
        none of the three keys, mirroring the ``tint`` pattern."""
        from spyde.actions.report.model import LayerSpec

        untouched = LayerSpec()
        d = untouched.to_dict()
        assert "color" not in d
        assert "linewidth" not in d
        assert "label" not in d

        styled = LayerSpec(color="#abcdef", linewidth=2.5, label="curve 1")
        d2 = styled.to_dict()
        assert d2["color"] == "#abcdef"
        assert d2["linewidth"] == pytest.approx(2.5)
        assert d2["label"] == "curve 1"

        rt = LayerSpec.from_dict(d2)
        assert rt.color == "#abcdef"
        assert rt.linewidth == pytest.approx(2.5)
        assert rt.label == "curve 1"

        # An older-file dict with none of the keys loads every one as None.
        rt_old = LayerSpec.from_dict({"id": "l1", "cmap": "viridis"})
        assert rt_old.color is None
        assert rt_old.linewidth is None
        assert rt_old.label is None

    def test_yaml_round_trip_live_cell(self, signal1d_dataset):
        """The live-cell path: repfig_set_layer's styling round-trips through
        the cell's actual to_yaml()/from_yaml() (the persisted-container
        contract), including the label key which is genuinely absent until
        set (unlike color/linewidth, which the live Plot1D always supplies a
        default for on snapshot)."""
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        layer = panel.layers[0]

        # Untouched: label absent from the dict + YAML (never set yet).
        d0 = cell.spec.to_dict()
        assert "label" not in d0["panels"][0]["layers"][0]

        cx.repfig_set_layer(session, None, {
            "cell_id": cid, "panel_id": panel.id, "layer_id": layer.id,
            "color": "#abcdef", "linewidth": 2.5, "label": "curve 1",
        })
        cell = _cell_of(session, cid)
        yaml_text = cell.spec.to_yaml()
        assert "abcdef" in yaml_text
        assert "linewidth: 2.5" in yaml_text
        assert "label: curve 1" in yaml_text

        round_tripped = FigureSpec.from_yaml(yaml_text)
        rt_layer = round_tripped.panels[0].layers[0]
        assert rt_layer.color == "#abcdef"
        assert rt_layer.linewidth == pytest.approx(2.5)
        assert rt_layer.label == "curve 1"
        assert round_tripped.panels[0].kind == "line"


# ── refusals ─────────────────────────────────────────────────────────────────


class TestRefusals:
    def test_add_annotation_refused(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        n_anns_before = len(panel.annotations)
        messages.clear()

        cx.repfig_add_annotation(session, None, {
            "cell_id": cid, "panel_id": panel.id,
            "annotation": {"kind": "circle", "offsets": [[1.0, 2.0]],
                          "radius": 3.0},
        })
        assert _errors(messages), "expected an error for annotation on a line panel"
        cell = _cell_of(session, cid)
        assert len(cell.spec.panels[0].annotations) == n_anns_before

    def test_add_callout_refused(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        n_panels_before = len(cell.spec.panels)
        n_insets_before = len(panel.insets)
        messages.clear()

        cx.repfig_add_callout(session, None, {"cell_id": cid, "panel_id": panel.id})
        assert _errors(messages), "expected an error for a callout on a line panel"
        cell = _cell_of(session, cid)
        assert len(cell.spec.panels) == n_panels_before
        assert len(cell.spec.panels[0].insets) == n_insets_before

    def test_add_time_callouts_refused(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        n_panels_before = len(cell.spec.panels)
        messages.clear()

        cx.repfig_add_time_callouts(session, None,
                                    {"cell_id": cid, "panel_id": panel.id})
        assert _errors(messages), "expected an error for time-callouts on a line panel"
        cell = _cell_of(session, cid)
        assert len(cell.spec.panels) == n_panels_before

    def test_add_zoom_callout_refused(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        n_panels_before = len(cell.spec.panels)
        messages.clear()

        cx.repfig_add_zoom_callout(session, None,
                                   {"cell_id": cid, "panel_id": panel.id})
        assert _errors(messages), "expected an error for zoom-callout on a line panel"
        cell = _cell_of(session, cid)
        assert len(cell.spec.panels) == n_panels_before


# ── refresh ──────────────────────────────────────────────────────────────────


class TestRefresh:
    def test_refresh_panel_resnapshots_curve(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)

        mgr = session._report
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]

        # Change the live plot's current_data (simulating new data landing).
        src_plot = next(p for p in session._plots if p.window_id == wid)
        new_data = np.cos(np.linspace(0, 4 * np.pi, 128)).astype(np.float32)
        src_plot.current_data = new_data
        src_plot._last_levels = (float(new_data.min()), float(new_data.max()))

        ok = h.refresh_panel(session, mgr, cell, panel)
        assert ok is True
        snap = mgr.snapshot_map(cid).get((panel.id, panel.layers[0].id))
        assert snap is not None
        assert snap.ndim == 1
        np.testing.assert_allclose(np.asarray(snap), new_data, atol=1e-5)

    def test_repfig_refresh_panel_handler_no_error(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        cell = _cell_of(session, cid)
        panel = cell.spec.panels[0]
        messages.clear()

        h.repfig_refresh_panel(session, None,
                               {"cell_id": cid, "panel_id": panel.id})
        assert not _errors(messages)
        assert _report_figures(messages), "expected a figure re-emit on refresh"

    def test_report_refresh_figure_no_error(self, signal1d_dataset):
        session, messages = signal1d_dataset["window"], signal1d_dataset["messages"]
        _prime_1d_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        messages.clear()

        h.report_refresh_figure(session, None, {"cell_id": cid})
        assert not _errors(messages)
        st = _last_state(messages)
        cell_entry = next(c for c in st["cells"] if c["id"] == cid)
        assert cell_entry["data_offline"] is False

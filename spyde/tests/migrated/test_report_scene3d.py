"""
test_report_scene3d.py — Phase 5: 3-D IPF report cells (kind='scene3d').

A window pill dragged while its 3-D IPF explorer is shown drops with
``view:'3d'``; ``report_add_figure`` then snapshots the SCENE — a single
scene3d PanelSpec whose small ``scene`` params ride the spec while the point
cloud (xyz/rgb) lives ONLY in the backend snapshot map under the pseudo layer
keys ``(panel_id, "xyz") / (panel_id, "rgb")``. These tests pin: the drop
branch, spec/YAML round-trip tolerance, refresh-recompute, save→reopen
rebind + offline, compose refusal, and the bake-fallback contract (the Agg
bake can NEVER see a point cloud).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from spyde.actions.report import compose as cx
from spyde.actions.report import handlers as h
from spyde.actions.report.model import FigureSpec, LayerSpec, PanelSpec


# ── helpers ────────────────────────────────────────────────────────────────────


def _orientation_map(ny=4, nx=5, seed=0):
    """A tiny SpyDEOrientationMap with random orientations (m-3m point group —
    rebuilt straight from the phase dict, no CIF / diffpy needed)."""
    from spyde.signals.orientation_map import SpyDEOrientationMap

    rng = np.random.RandomState(seed)
    q = rng.randn(ny, nx, 1, 4).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    corr = np.ones((ny, nx, 1), np.float32)
    phase_idx = np.zeros((ny, nx, 1), np.int16)
    mirror = np.ones((ny, nx, 1), np.int8)
    return SpyDEOrientationMap(q, corr, phase_idx, mirror,
                               [{"name": "Al", "point_group": "m-3m"}])


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _errors(messages):
    return [m for m in messages if m.get("type") == "error"]


def _report_figures(messages):
    return [m for m in messages if m.get("type") == "figure"
            and m.get("host") == "report"]


def _signal_plot(session):
    for p in session._plots:
        if not getattr(p, "is_navigator", False) and p.window_id is not None:
            return p
    return session._plots[0]


def _drop_3d(session, messages, caption="IPF 3D"):
    """Attach an OM to the signal plot's tree and drop it with view:'3d'.
    Returns (cell_id, tree)."""
    plot = _signal_plot(session)
    tree = plot.signal_tree
    tree.orientation_map = _orientation_map()
    h.report_new(session, None, {})
    messages.clear()
    h.report_add_figure(session, None, {
        "source_window_id": plot.window_id, "view": "3d", "caption": caption,
    })
    cells = [c for c in _last_state(messages)["cells"]
             if c["cell_type"] == "figure"]
    assert len(cells) == 1, "3-D drop did not create exactly one figure cell"
    return cells[0]["id"], tree


# ── the drop branch ────────────────────────────────────────────────────────────


class TestScene3DDrop:
    def test_drop_3d_creates_scene3d_cell_with_scene_and_snapshots(
            self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)

        st = _last_state(messages)
        cell = [c for c in st["cells"] if c["id"] == cid][0]
        fig = cell["figure"]
        assert len(fig["panels"]) == 1
        panel = fig["panels"][0]
        assert panel["kind"] == "scene3d"
        scene = panel["scene"]
        assert scene["kind"] == "ipf3d"
        assert scene["direction"] == "z"
        assert scene["point_size"] > 0
        assert len(scene["bounds"]) == 3
        # The rebind handle: one LayerSpec whose SignalRef points at the tree.
        assert len(panel["layers"]) == 1
        assert panel["layers"][0]["source"]["tree_uid"]
        # Callout buttons stay hidden (scene3d never navigates).
        assert panel["nav_dims"] == 0

        # The point cloud is held backend-side under the pseudo layer keys.
        mgr = session._report
        snaps = mgr._snapshots[cid]
        xyz = snaps[("p1", "xyz")]
        rgb = snaps[("p1", "rgb")]
        assert isinstance(xyz, np.ndarray) and xyz.shape[1] == 3 and len(xyz) > 0
        assert isinstance(rgb, np.ndarray) and rgb.shape == xyz.shape
        # Points sit on the unit sphere (the same data emit_ipf_3d draws).
        assert np.allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-3)

        # A live report figure was emitted for the cell.
        figs = _report_figures(messages)
        assert figs and figs[-1]["cell_id"] == cid

    def test_report_state_stays_pixel_free(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)
        st = _last_state(messages)
        cell = [c for c in st["cells"] if c["id"] == cid][0]
        # The shipped figure dict is the SMALL recipe: no point-cloud keys, and
        # tiny when serialized (arrays would be orders of magnitude larger).
        blob = json.dumps(cell["figure"])
        assert "xyz" not in cell["figure"]["panels"][0]
        assert "rgb" not in cell["figure"]["panels"][0]
        assert len(blob) < 5000
        json.dumps(st)   # the whole state must remain JSON-serializable

    def test_drop_direction_follows_tree(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        plot = _signal_plot(session)
        tree = plot.signal_tree
        tree.orientation_map = _orientation_map()
        tree._ipf_direction = "x"      # the X/Y/Z selector state at drop time
        h.report_new(session, None, {})
        messages.clear()
        h.report_add_figure(session, None, {
            "source_window_id": plot.window_id, "view": "3d"})
        cell = [c for c in _last_state(messages)["cells"]
                if c["cell_type"] == "figure"][0]
        assert cell["figure"]["panels"][0]["scene"]["direction"] == "x"

    def test_drop_3d_without_orientation_result_errors(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        plot = _signal_plot(session)      # tree carries NO orientation map
        h.report_new(session, None, {})
        messages.clear()
        h.report_add_figure(session, None, {
            "source_window_id": plot.window_id, "view": "3d"})
        # The handler errors and returns BEFORE mutating the doc (no state
        # re-emit — same contract as the "no image to snapshot" case).
        assert _errors(messages), "expected the no-3-D-view error"
        assert not any(c.cell_type == "figure"
                       for c in session._report.doc.cells)


# ── spec / YAML round-trip tolerance ───────────────────────────────────────────


class TestScene3DSpecRoundTrip:
    def _scene_spec(self):
        scene = {"kind": "ipf3d", "direction": "y", "point_size": 6.0,
                 "bounds": [[-1.0, 1.0]] * 3}
        panel = PanelSpec(id="p1", kind="scene3d", layers=[LayerSpec()],
                          scene=scene)
        return FigureSpec(layout={"kind": "single"}, panels=[panel])

    def test_dict_roundtrip(self):
        spec = self._scene_spec()
        d = spec.to_dict()
        assert d["panels"][0]["kind"] == "scene3d"
        assert d["panels"][0]["scene"]["direction"] == "y"
        back = FigureSpec.from_dict(d)
        assert back.panels[0].kind == "scene3d"
        assert back.panels[0].scene == d["panels"][0]["scene"]

    def test_yaml_roundtrip(self):
        spec = self._scene_spec()
        back = FigureSpec.from_yaml(spec.to_yaml())
        assert back.panels[0].kind == "scene3d"
        assert back.panels[0].scene == spec.panels[0].scene

    def test_scene_key_omitted_when_unset(self):
        # Old readers never see the key; old dicts without it load as None.
        assert "scene" not in PanelSpec().to_dict()
        old = PanelSpec.from_dict({"id": "p1", "kind": "image", "layers": []})
        assert old.scene is None
        assert old.kind == "image"


# ── refresh recompute ─────────────────────────────────────────────────────────


class TestScene3DRefresh:
    def test_panel_refresh_recomputes_point_cloud(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, tree = _drop_3d(session, messages)
        mgr = session._report
        xyz0 = np.array(mgr._snapshots[cid][("p1", "xyz")], copy=True)

        # New orientations on the tree → refresh must RECOMPUTE, not keep.
        tree.orientation_map = _orientation_map(seed=7)
        tree._ipf_result = None if not hasattr(tree, "_ipf_result") else None
        messages.clear()
        h.repfig_refresh_panel(session, None, {"cell_id": cid, "panel_id": "p1"})
        xyz1 = mgr._snapshots[cid][("p1", "xyz")]
        assert xyz1.shape[1] == 3 and len(xyz1) > 0
        assert not (xyz1.shape == xyz0.shape and np.allclose(xyz1, xyz0)), \
            "refresh did not recompute the scene3d point cloud"
        # The refresh rebuilt + re-emitted the cell.
        assert _report_figures(messages)

    def test_whole_figure_refresh_keeps_cell_live(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)
        messages.clear()
        h.report_refresh_figure(session, None, {"cell_id": cid})
        st = _last_state(messages)
        cell = [c for c in st["cells"] if c["id"] == cid][0]
        assert cell["data_offline"] is False
        assert _report_figures(messages)


# ── save → reopen: rebind + offline ───────────────────────────────────────────


class TestScene3DReopen:
    def test_save_reopen_rebinds_live(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)
        path = os.path.join(str(tmp_path), "scene.spyde-report")
        # Headless save: no renderer harvest → finish runs synchronously; the
        # scene3d cell gets NO baked asset (the Agg bake can't render it) and
        # that must not crash the save.
        h.report_save(session, None, {"path": path})
        assert os.path.exists(path)

        h.report_close(session, None, {})
        messages.clear()
        h.report_open(session, None, {"path": path})
        st = _last_state(messages)
        cell = [c for c in st["cells"] if c["cell_type"] == "figure"][0]
        assert cell["figure"]["panels"][0]["kind"] == "scene3d"
        # The tree is still open → the scene recomputes and the cell is LIVE.
        assert cell["data_offline"] is False
        mgr = session._report
        assert ("p1", "xyz") in mgr._snapshots[cell["id"]]
        assert _report_figures(messages)

    def test_reopen_without_result_goes_offline_gracefully(
            self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, tree = _drop_3d(session, messages)
        path = os.path.join(str(tmp_path), "scene-offline.spyde-report")
        h.report_save(session, None, {"path": path})
        h.report_close(session, None, {})

        # The orientation result is gone (e.g. a fresh session where OM was
        # never recomputed) → the cell must open OFFLINE, never crash.
        tree.orientation_map = None
        if hasattr(tree, "_ipf_result"):
            tree._ipf_result = None
        messages.clear()
        h.report_open(session, None, {"path": path})
        st = _last_state(messages)
        cell = [c for c in st["cells"] if c["cell_type"] == "figure"][0]
        assert cell["data_offline"] is True
        json.dumps(st)   # state still serializable (badge path, maybe no png)


# ── compose refusal ───────────────────────────────────────────────────────────


class TestScene3DCompose:
    def test_query_compose_offers_nothing(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)
        src_wid = _signal_plot(session).window_id
        messages.clear()
        cx.repfig_query_compose(session, None, {
            "cell_id": cid, "source_window_id": src_wid})
        opts = [m for m in messages if m.get("type") == "repfig_compose_options"]
        assert opts and opts[-1]["options"] == []
        assert opts[-1]["detail"] == {"same_shape": False,
                                      "nav_signal_pair": False}

    def test_compose_tile_is_refused_without_mutation(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)
        src_wid = _signal_plot(session).window_id
        mgr = session._report
        n_panels = len(mgr.doc.cell_by_id(cid).spec.panels)
        messages.clear()
        cx.repfig_compose(session, None, {
            "cell_id": cid, "mode": "tile-right", "source_window_id": src_wid})
        assert _errors(messages), "compose onto a scene3d cell must refuse"
        assert len(mgr.doc.cell_by_id(cid).spec.panels) == n_panels

    def test_add_annotation_is_refused(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)
        mgr = session._report
        messages.clear()
        cx.repfig_add_annotation(session, None, {
            "cell_id": cid, "panel_id": "p1",
            "annotation": {"kind": "circle", "offsets": [[1, 1]], "radius": 1}})
        assert _errors(messages)
        assert mgr.doc.cell_by_id(cid).spec.panels[0].annotations == []


# ── asset assembly (bake fallback contract) ───────────────────────────────────


class TestScene3DAssets:
    def test_assemble_assets_without_harvest_skips_gracefully(
            self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)
        mgr = session._report
        # No harvested PNG, no baked PNG: the Agg bake cannot render a point
        # cloud, so the cell is skipped — never baked from the xyz array.
        assets = mgr.assemble_assets({})
        assert cid not in assets

    def test_harvested_png_is_used_and_reused(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)
        mgr = session._report
        png = b"\x89PNG-fake-3d-pixels"
        assets = mgr.assemble_assets({cid: png})
        assert assets[cid] == png
        # The harvest is stashed as the baked fallback: a LATER headless
        # assemble (no renderer reply) reuses the last real 3-D pixels.
        assets2 = mgr.assemble_assets({})
        assert assets2[cid] == png

    def test_primary_snapshot_never_returns_point_cloud(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        cid, _tree = _drop_3d(session, messages)
        mgr = session._report
        assert mgr.primary_snapshot(cid) is None

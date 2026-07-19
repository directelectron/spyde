"""
test_report_callouts.py — Report Builder Phase 3 fresh-slice + zoom-region
inset callouts, and inset drag-geometry persistence.

A fresh-slice callout stores WHERE it was sliced (``nav_indices`` x-first /
``time_index``) and every re-read goes through ``slicing.read_frame_at`` —
slice ``sig.inav[...]`` FIRST, compute ONLY the slice. A ZOOM-REGION callout
(``zoom_region``) instead crops the BASE panel's OWN already-held snapshot —
it never touches the dataset, so it works on a plain 2-D image with no
navigation axes at all. Covered against a real Qt-free ``Session``:

  1. ``repfig_add_callout`` on a 4-D dataset: inset carries nav_indices
     (default = nav-space center), the hidden inset panel is NOT a grid panel,
     its snapshot equals ``sig.inav[...]``, connector present when the base
     panel is the navigator image.
  2. Edit-mode marker drag: a simulated ``pointer_up`` at a new position
     re-slices (nav_indices + snapshot + connector updated, figure rebuilt);
     clamped out-of-range drops; unchanged position skips the rebuild.
  3. ``repfig_add_time_callouts`` on a 1-D-nav movie: three insets at
     t = 0 / n//2 / n-1 with spread anchors, each snapshot the lazy frame.
  4. Refresh re-slices at the STORED position (never the live current frame).
  5. Memory safety: ``da.Array.compute`` is spied (test_find_vectors_memory
     pattern) — only single-frame slices ever compute, never the full dataset.
  6. ``nav_dims`` is stamped on the SHIPPED panel dicts only (ephemeral).
  7. ``repfig_add_zoom_callout`` on a plain 2-D image (no nav axes): crops the
     base snapshot, refuses on scene3d/missing-snapshot, default region is
     centered + clamped.
  8. Edit-mode zoom-region rectangle drag: a simulated ``pointer_up`` re-crops
     the base snapshot at the new rect; unchanged region skips the rebuild.
  9. Inset drag/resize geometry (``inset_geometry_change``) persists anchor +
     w_frac/h_frac into the owning inset dict WITHOUT a rebuild.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from spyde.actions.report import compose as cx
from spyde.actions.report import handlers as h
from spyde.actions.report.slicing import read_frame_at


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


def _current_signal(session, wid):
    return session._plot_by_window_id(wid).plot_state.current_signal


def _frame_at(sig, idx):
    """The expected frame at x-first *idx* — computed the reference way."""
    data = sig.inav[tuple(int(i) for i in idx)].data
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data)


def _nav_cell(session, messages):
    """A figure cell built from the NAVIGATOR window (base = the nav image, so
    callouts get a connector + edit-mode marker). Returns (cid, base_panel)."""
    _prime_plot_data(session)
    nav_wid = _nav_wid(session)
    assert nav_wid is not None
    h.report_new(session, None, {})
    cid = _make_figure_cell(session, messages, nav_wid)
    cell = session._report.doc.cell_by_id(cid)
    return cid, cell.spec.panels[0]


def _2d_cell(session, messages):
    """A figure cell built from a plain 2-D image window (no navigation axes
    at all — the zoom-callout's "works with no nav" case). Returns (cid,
    base_panel)."""
    _prime_plot_data(session)
    wid = _signal_wid(session)
    h.report_new(session, None, {})
    cid = _make_figure_cell(session, messages, wid)
    cell = session._report.doc.cell_by_id(cid)
    return cid, cell.spec.panels[0]


# ── 1. repfig_add_callout on a 4-D dataset ─────────────────────────────────────


class TestAddCallout:
    def test_default_center_callout(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel = _nav_cell(session, messages)
        sig = _current_signal(session, _signal_wid(session))
        # nav_shape (nx, ny) = (5, 4) → center [2, 2] (x-first, inav order).
        assert tuple(sig.axes_manager.navigation_shape) == (5, 4)
        messages.clear()

        cx.repfig_add_callout(session, None,
                              {"cell_id": cid, "panel_id": base_panel.id})

        fig = _fig_dict_of(messages, cid)
        insets = fig["panels"][0]["insets"]
        assert len(insets) == 1
        inset = insets[0]
        assert inset["nav_indices"] == [2, 2]
        # The hidden inset panel exists but is NOT a grid panel; layout single.
        inset_pid = inset["panel"]
        cell = session._report.doc.cell_by_id(cid)
        assert any(p.id == inset_pid for p in cell.spec.panels)
        assert inset_pid not in [p.id for p in cx._grid_panels(cell.spec)]
        assert fig["layout"]["kind"] == "single"
        # Snapshot = the frame sliced at the stored position.
        inset_panel = next(p for p in cell.spec.panels if p.id == inset_pid)
        snap = session._report.snapshot_map(cid).get(
            (inset_pid, inset_panel.layers[0].id))
        np.testing.assert_array_equal(snap, _frame_at(sig, [2, 2]))
        # Base is the navigator image → connector rect around the point.
        assert inset["connector"] is not None
        assert inset["connector"]["region"] == list(
            cx._index_region_to_data(base_panel, (1.5, 1.5, 1.0, 1.0)))
        # Same source ref as the base layer; a figure re-emit happened.
        assert inset_panel.layers[0].source is base_panel.layers[0].source
        assert len(_report_figures(messages)) == 1

    def test_explicit_indices_clamped(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel = _nav_cell(session, messages)
        sig = _current_signal(session, _signal_wid(session))
        messages.clear()
        # (99, -3) clamps to (4, 0) — nx=5, ny=4.
        cx.repfig_add_callout(session, None,
                              {"cell_id": cid, "panel_id": base_panel.id,
                               "nav_indices": [99, -3]})
        fig = _fig_dict_of(messages, cid)
        inset = fig["panels"][0]["insets"][0]
        assert inset["nav_indices"] == [4, 0]
        cell = session._report.doc.cell_by_id(cid)
        inset_panel = next(p for p in cell.spec.panels
                           if p.id == inset["panel"])
        snap = session._report.snapshot_map(cid).get(
            (inset["panel"], inset_panel.layers[0].id))
        np.testing.assert_array_equal(snap, _frame_at(sig, [4, 0]))

    def test_no_nav_axes_errors(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        wid = _signal_wid(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, wid)
        panel_id = _fig_dict_of(messages, cid)["panels"][0]["id"]
        messages.clear()
        cx.repfig_add_callout(session, None,
                              {"cell_id": cid, "panel_id": panel_id})
        errors = [m for m in messages if m.get("type") == "error"]
        assert errors and "navigation axes" in errors[-1]["text"]
        assert not _states(messages)   # no mutation

    def test_rank_mismatch_errors(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel = _nav_cell(session, messages)
        messages.clear()
        cx.repfig_add_callout(session, None,
                              {"cell_id": cid, "panel_id": base_panel.id,
                               "nav_indices": [1]})
        errors = [m for m in messages if m.get("type") == "error"]
        assert errors and "rank mismatch" in errors[-1]["text"]
        assert not _states(messages)


# ── 2. edit-mode marker drag → re-slice ────────────────────────────────────────


class TestMarkerDrag:
    def _callout_in_edit(self, session, messages):
        """A nav-based cell with one center callout, in edit mode. Returns
        (cid, base_panel, fig, widget)."""
        cid, base_panel = _nav_cell(session, messages)
        cx.repfig_add_callout(session, None,
                              {"cell_id": cid, "panel_id": base_panel.id})
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        mgr = session._report
        fig = mgr._controllers[mgr._window_by_cell[cid]].fig
        wiring = list(getattr(fig, "_report_callout_wiring", []))
        assert len(wiring) == 1, "edit rebuild must create ONE callout marker"
        widget, wired_pid, inset_index, wired_panel = wiring[0]
        assert wired_pid == base_panel.id and inset_index == 0
        assert wired_panel is base_panel
        return cid, base_panel, fig, widget

    def _drop(self, fig, base_panel, widget, cx_px, cy_px):
        disp = fig._report_panel_map[base_panel.id]
        fig._dispatch_event(json.dumps({
            "panel_id": disp, "event_type": "pointer_up",
            "widget_id": widget.id, "cx": cx_px, "cy": cy_px, "r": 0.5}))

    def test_marker_drop_reslices(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel, fig, widget = self._callout_in_edit(session, messages)
        sig = _current_signal(session, _signal_wid(session))
        messages.clear()

        # Drop near nav pixel (0, 1) — rounds to index [0, 1].
        self._drop(fig, base_panel, widget, 0.2, 1.2)

        inset = base_panel.insets[0]
        assert inset["nav_indices"] == [0, 1]
        cell = session._report.doc.cell_by_id(cid)
        inset_panel = next(p for p in cell.spec.panels
                           if p.id == inset["panel"])
        snap = session._report.snapshot_map(cid).get(
            (inset["panel"], inset_panel.layers[0].id))
        np.testing.assert_array_equal(snap, _frame_at(sig, [0, 1]))
        # Connector followed the point; rebuild + state emitted.
        assert inset["connector"]["region"] == list(
            cx._index_region_to_data(base_panel, (-0.5, 0.5, 1.0, 1.0)))
        assert len(_report_figures(messages)) == 1
        assert _states(messages)

    def test_marker_drop_clamps(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel, fig, widget = self._callout_in_edit(session, messages)
        sig = _current_signal(session, _signal_wid(session))
        messages.clear()
        self._drop(fig, base_panel, widget, 99.0, -5.0)   # → clamp [4, 0]
        inset = base_panel.insets[0]
        assert inset["nav_indices"] == [4, 0]
        cell = session._report.doc.cell_by_id(cid)
        inset_panel = next(p for p in cell.spec.panels
                           if p.id == inset["panel"])
        snap = session._report.snapshot_map(cid).get(
            (inset["panel"], inset_panel.layers[0].id))
        np.testing.assert_array_equal(snap, _frame_at(sig, [4, 0]))

    def test_unchanged_position_skips_rebuild(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel, fig, widget = self._callout_in_edit(session, messages)
        messages.clear()
        # Drop back onto the CURRENT index ([2, 2]) → no rebuild, no state.
        self._drop(fig, base_panel, widget, 2.1, 1.9)
        assert base_panel.insets[0]["nav_indices"] == [2, 2]
        assert not _report_figures(messages)
        assert not _states(messages)

    def test_no_marker_outside_edit_mode(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel = _nav_cell(session, messages)
        cx.repfig_add_callout(session, None,
                              {"cell_id": cid, "panel_id": base_panel.id})
        mgr = session._report
        fig = mgr._controllers[mgr._window_by_cell[cid]].fig
        assert getattr(fig, "_report_callout_wiring", None) == []


# ── 3. time callouts on a 1-D-nav movie ────────────────────────────────────────


class TestTimeCallouts:
    def test_three_insets_start_mid_end(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        sig = _current_signal(session, sig_wid)
        n = int(sig.axes_manager.navigation_shape[0])
        assert n == 8
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, sig_wid)
        panel_id = _fig_dict_of(messages, cid)["panels"][0]["id"]
        messages.clear()

        cx.repfig_add_time_callouts(session, None,
                                    {"cell_id": cid, "panel_id": panel_id})

        fig = _fig_dict_of(messages, cid)
        insets = fig["panels"][0]["insets"]
        assert [i["time_index"] for i in insets] == [0, 4, 7]
        assert [i["anchor"] for i in insets] == \
            [[0.03, 0.03], [0.37, 0.03], [0.71, 0.03]]
        assert all(i["connector"] is None for i in insets)
        # Each hidden inset panel holds the frame at its time index; none of
        # them count as grid panels (layout stays single).
        cell = session._report.doc.cell_by_id(cid)
        grid_ids = [p.id for p in cx._grid_panels(cell.spec)]
        assert grid_ids == [panel_id]
        assert fig["layout"]["kind"] == "single"
        for ins in insets:
            inset_panel = next(p for p in cell.spec.panels
                               if p.id == ins["panel"])
            snap = session._report.snapshot_map(cid).get(
                (ins["panel"], inset_panel.layers[0].id))
            np.testing.assert_array_equal(
                snap, _frame_at(sig, [ins["time_index"]]))
        assert len(_report_figures(messages)) == 1

    def test_2d_nav_refused(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel = _nav_cell(session, messages)
        messages.clear()
        cx.repfig_add_time_callouts(session, None,
                                    {"cell_id": cid, "panel_id": base_panel.id})
        errors = [m for m in messages if m.get("type") == "error"]
        assert errors and "1-D" in errors[-1]["text"]
        assert not _states(messages)


# ── 4. refresh re-slices at the STORED position ────────────────────────────────


class TestRefreshStoredPosition:
    def test_4d_refresh_keeps_position(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel = _nav_cell(session, messages)
        sig_plot = session._plot_by_window_id(_signal_wid(session))
        sig = sig_plot.plot_state.current_signal
        cx.repfig_add_callout(session, None,
                              {"cell_id": cid, "panel_id": base_panel.id})
        inset = base_panel.insets[0]
        assert inset["nav_indices"] == [2, 2]
        # Make the LIVE current frame something else entirely — a refresh that
        # (wrongly) snapshots the live plot would store zeros.
        sig_plot.current_data = np.zeros((16, 16), dtype=np.float32)
        messages.clear()

        h.report_refresh_figure(session, None, {"cell_id": cid})

        cell = session._report.doc.cell_by_id(cid)
        inset_panel = next(p for p in cell.spec.panels
                           if p.id == inset["panel"])
        snap = session._report.snapshot_map(cid).get(
            (inset["panel"], inset_panel.layers[0].id))
        expected = _frame_at(sig, [2, 2])
        assert np.any(expected != 0)
        np.testing.assert_array_equal(snap, expected)

    def test_movie_refresh_keeps_time_frames(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        _prime_plot_data(session)
        sig_wid = _signal_wid(session)
        sig_plot = session._plot_by_window_id(sig_wid)
        sig = sig_plot.plot_state.current_signal
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, sig_wid)
        panel_id = _fig_dict_of(messages, cid)["panels"][0]["id"]
        cx.repfig_add_time_callouts(session, None,
                                    {"cell_id": cid, "panel_id": panel_id})
        sig_plot.current_data = np.zeros((32, 32), dtype=np.float32)
        messages.clear()

        h.report_refresh_figure(session, None, {"cell_id": cid})

        cell = session._report.doc.cell_by_id(cid)
        base = next(p for p in cell.spec.panels if p.id == panel_id)
        for ins in base.insets:
            inset_panel = next(p for p in cell.spec.panels
                               if p.id == ins["panel"])
            snap = session._report.snapshot_map(cid).get(
                (ins["panel"], inset_panel.layers[0].id))
            expected = _frame_at(sig, [ins["time_index"]])
            assert np.any(expected != 0)
            np.testing.assert_array_equal(snap, expected)


# ── 5. memory safety: only single-frame slices ever compute ───────────────────


class _FullComputeGuard:
    """Context helper: patch ``da.Array.compute`` to record every computed
    shape and RAISE on a full-dataset compute (test_find_vectors_memory's
    ``patch.object`` pattern)."""

    def __init__(self, full_shape):
        import dask.array as da
        self.da = da
        self.full_shape = tuple(full_shape)
        self.shapes: list[tuple] = []
        self._orig = da.Array.compute
        guard = self

        def _spy(arr_self, *args, **kwargs):
            guard.shapes.append(tuple(arr_self.shape))
            if tuple(arr_self.shape) == guard.full_shape:
                raise AssertionError(
                    f"full-dataset .compute() on shape {arr_self.shape} — "
                    "callout slicing must compute the SLICE only")
            return guard._orig(arr_self, *args, **kwargs)

        self._patch = patch.object(da.Array, "compute", _spy)

    def __enter__(self):
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        return self._patch.__exit__(*exc)


class TestMemorySafety:
    def test_time_callouts_compute_single_frames_only(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        _prime_plot_data(session)   # BEFORE the guard (may realise tiny arrays)
        sig_wid = _signal_wid(session)
        sig = _current_signal(session, sig_wid)
        full_shape = tuple(sig.data.shape)   # (8, 32, 32)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, sig_wid)
        panel_id = _fig_dict_of(messages, cid)["panels"][0]["id"]

        with _FullComputeGuard(full_shape) as guard:
            cx.repfig_add_time_callouts(session, None,
                                        {"cell_id": cid, "panel_id": panel_id})

        fig = _fig_dict_of(messages, cid)
        assert len(fig["panels"][0]["insets"]) == 3
        assert guard.shapes, "the lazy movie slices must go through compute()"
        frame_px = int(np.prod(full_shape[1:]))
        assert all(int(np.prod(s)) <= frame_px for s in guard.shapes), \
            f"a compute exceeded one frame: {guard.shapes}"

    def test_read_frame_at_lazy_4d_slices_only(self):
        import hyperspy.api as hs
        data = np.random.RandomState(1).rand(6, 7, 16, 16).astype(np.float32)
        s = hs.signals.Signal2D(data).as_lazy()
        plot = SimpleNamespace(plot_state=SimpleNamespace(current_signal=s))
        with _FullComputeGuard(data.shape):
            frame = read_frame_at(plot, [3, 2])
        assert frame is not None and frame.shape == (16, 16)
        np.testing.assert_array_equal(frame, data[2, 3])   # inav is x-first

    def test_read_frame_at_guards(self):
        import hyperspy.api as hs
        s = hs.signals.Signal2D(np.zeros((4, 5, 8, 8), dtype=np.float32))
        plot = SimpleNamespace(plot_state=SimpleNamespace(current_signal=s))
        assert read_frame_at(plot, [1]) is None            # rank mismatch
        assert read_frame_at(plot, [99, 99]) is None       # out of range
        s2 = hs.signals.Signal2D(np.zeros((8, 8), dtype=np.float32))
        plot2 = SimpleNamespace(plot_state=SimpleNamespace(current_signal=s2))
        assert read_frame_at(plot2, []) is None            # no nav axes

    def test_zoom_callout_and_drag_never_compute_dataset(self, stem_4d_dataset):
        """A zoom callout crops an already-in-memory SNAPSHOT array — it must
        never reach ``da.Array.compute`` at all (not even a slice), since it
        never touches the dataset in the first place."""
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel = _nav_cell(session, messages)
        sig = _current_signal(session, _signal_wid(session))
        full_shape = tuple(sig.data.shape)
        messages.clear()

        with _FullComputeGuard(full_shape) as guard:
            cx.repfig_add_zoom_callout(session, None,
                                       {"cell_id": cid, "panel_id": base_panel.id})
            h.report_refresh_figure(session, None, {"cell_id": cid})

        assert not guard.shapes, \
            f"zoom callout must never call compute() at all: {guard.shapes}"


# ── 6. nav_dims is emit-time-only ──────────────────────────────────────────────


class TestNavDims:
    def test_4d_panels_carry_nav_dims_2(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, _base = _nav_cell(session, messages)
        fig = _fig_dict_of(messages, cid)
        assert fig["panels"][0]["nav_dims"] == 2
        # Ephemeral: the persisted spec dict has NO nav_dims key.
        spec_dict = session._report.doc.cell_by_id(cid).spec.to_dict()
        assert "nav_dims" not in spec_dict["panels"][0]

    def test_movie_panel_nav_dims_1(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        _prime_plot_data(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, _signal_wid(session))
        fig = _fig_dict_of(messages, cid)
        assert fig["panels"][0]["nav_dims"] == 1

    def test_2d_image_nav_dims_0(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        _prime_plot_data(session)
        h.report_new(session, None, {})
        cid = _make_figure_cell(session, messages, _signal_wid(session))
        fig = _fig_dict_of(messages, cid)
        assert fig["panels"][0]["nav_dims"] == 0


# ── 7. repfig_add_zoom_callout — a magnified crop of the panel's OWN pixels ────


class TestZoomCallout:
    def test_default_center_region_on_plain_2d_image(self, tem_2d_dataset):
        """No navigation axes needed at all: the base panel's OWN 32x32
        snapshot is cropped centered, W/4 x H/4 pixels by default."""
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel = _2d_cell(session, messages)
        base_arr = np.asarray(session._report.snapshot_map(cid)[
            (base_panel.id, base_panel.layers[0].id)])
        assert base_arr.shape == (32, 32)
        messages.clear()

        cx.repfig_add_zoom_callout(session, None,
                                   {"cell_id": cid, "panel_id": base_panel.id})

        fig = _fig_dict_of(messages, cid)
        insets = fig["panels"][0]["insets"]
        assert len(insets) == 1
        inset = insets[0]
        # Default region: centered, W/4 x H/4 = 8x8 at (12, 12) (uncalibrated
        # 32x32 image → data coords == index coords).
        assert inset["zoom_region"] == [12.0, 12.0, 8.0, 8.0]
        assert inset["connector"]["region"] == inset["zoom_region"]
        assert inset["corner"] == "bottom-right"
        # Hidden inset panel exists, NOT a grid panel; layout unaffected.
        inset_pid = inset["panel"]
        cell = session._report.doc.cell_by_id(cid)
        assert any(p.id == inset_pid for p in cell.spec.panels)
        assert inset_pid not in [p.id for p in cx._grid_panels(cell.spec)]
        assert fig["layout"]["kind"] == "single"
        # Snapshot equals the crop of the BASE snapshot (never a re-slice).
        inset_panel = next(p for p in cell.spec.panels if p.id == inset_pid)
        snap = session._report.snapshot_map(cid).get(
            (inset_pid, inset_panel.layers[0].id))
        np.testing.assert_array_equal(snap, base_arr[12:20, 12:20])
        # Same source ref as the base layer (rebind/refresh handle).
        assert inset_panel.layers[0].source is base_panel.layers[0].source
        assert len(_report_figures(messages)) == 1

    def test_explicit_region_clamped_to_bounds(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel = _2d_cell(session, messages)
        base_arr = np.asarray(session._report.snapshot_map(cid)[
            (base_panel.id, base_panel.layers[0].id)])
        messages.clear()
        # A region that overhangs the bottom-right corner clamps into bounds
        # (28,28,10,10) on a 32x32 image -> x0/y0 pulled back to 22.
        cx.repfig_add_zoom_callout(session, None,
                                   {"cell_id": cid, "panel_id": base_panel.id,
                                    "region": [28, 28, 10, 10]})
        fig = _fig_dict_of(messages, cid)
        inset = fig["panels"][0]["insets"][0]
        assert inset["zoom_region"] == [22.0, 22.0, 10.0, 10.0]
        cell = session._report.doc.cell_by_id(cid)
        inset_panel = next(p for p in cell.spec.panels
                           if p.id == inset["panel"])
        snap = session._report.snapshot_map(cid).get(
            (inset["panel"], inset_panel.layers[0].id))
        np.testing.assert_array_equal(snap, base_arr[22:32, 22:32])

    def test_refused_on_scene3d_panel(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        cid, base_panel = _nav_cell(session, messages)
        cell = session._report.doc.cell_by_id(cid)
        cell.spec.panels[0].kind = "scene3d"
        messages.clear()
        cx.repfig_add_zoom_callout(session, None,
                                   {"cell_id": cid, "panel_id": base_panel.id})
        errors = [m for m in messages if m.get("type") == "error"]
        assert errors and "3-D" in errors[-1]["text"]
        assert not _states(messages)

    def test_refused_without_base_snapshot(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel = _2d_cell(session, messages)
        # Drop the base snapshot entirely (simulates a panel with no image yet).
        session._report._snapshots[cid].pop(
            (base_panel.id, base_panel.layers[0].id), None)
        messages.clear()
        cx.repfig_add_zoom_callout(session, None,
                                   {"cell_id": cid, "panel_id": base_panel.id})
        errors = [m for m in messages if m.get("type") == "error"]
        assert errors and "no image" in errors[-1]["text"]
        assert not _states(messages)

    def test_default_region_clamped_on_small_image(self, window):
        """A base image smaller than 4px on a side still gets a valid
        (>=1px) centered region, clamped into the image bounds."""
        session, messages = window["window"], window["messages"]
        from spyde.actions.report.model import (
            Cell, FigureSpec, LayerSpec, PanelSpec, SignalRef, new_cell_id,
        )
        h.report_new(session, None, {})
        mgr = session._report
        layer = LayerSpec(source=SignalRef())
        panel = PanelSpec(id="p1", grid_pos=[0, 0], kind="image", layers=[layer])
        cell = Cell(id=new_cell_id(), cell_type="figure", placeholder=False,
                   spec=FigureSpec(layout={"kind": "single"}, panels=[panel]))
        mgr.doc.cells.append(cell)
        tiny = np.arange(9, dtype=np.float32).reshape(3, 3)
        mgr.set_snapshot(cell.id, panel.id, layer.id, tiny)
        messages.clear()

        cx.repfig_add_zoom_callout(session, None,
                                   {"cell_id": cell.id, "panel_id": panel.id})

        errors = [m for m in messages if m.get("type") == "error"]
        assert not errors
        inset = panel.insets[0]
        x, y, w, h_ = inset["zoom_region"]
        assert w >= 1.0 and h_ >= 1.0
        assert 0.0 <= x <= 3.0 - w + 1e-9
        assert 0.0 <= y <= 3.0 - h_ + 1e-9


# ── 8. edit-mode zoom-region rectangle drag → re-crop ──────────────────────────


class TestZoomRegionDrag:
    def _zoom_in_edit(self, session, messages):
        """A plain 2-D cell with one centered zoom callout, in edit mode.
        Returns (cid, base_panel, fig, widget)."""
        cid, base_panel = _2d_cell(session, messages)
        cx.repfig_add_zoom_callout(session, None,
                                   {"cell_id": cid, "panel_id": base_panel.id})
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        mgr = session._report
        fig = mgr._controllers[mgr._window_by_cell[cid]].fig
        wiring = list(getattr(fig, "_report_zoom_wiring", []))
        assert len(wiring) == 1, "edit rebuild must create ONE zoom rectangle"
        widget, wired_pid, inset_index = wiring[0]
        assert wired_pid == base_panel.id and inset_index == 0
        return cid, base_panel, fig, widget

    def _drop(self, fig, base_panel, widget, x, y, w, h):
        disp = fig._report_panel_map[base_panel.id]
        fig._dispatch_event(json.dumps({
            "panel_id": disp, "event_type": "pointer_up",
            "widget_id": widget.id, "x": x, "y": y, "w": w, "h": h}))

    def test_rect_drop_recrops(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel, fig, widget = self._zoom_in_edit(session, messages)
        base_arr = np.asarray(session._report.snapshot_map(cid)[
            (base_panel.id, base_panel.layers[0].id)])
        messages.clear()

        self._drop(fig, base_panel, widget, 4.0, 4.0, 10.0, 6.0)

        inset = base_panel.insets[0]
        assert inset["zoom_region"] == [4.0, 4.0, 10.0, 6.0]
        assert inset["connector"]["region"] == [4.0, 4.0, 10.0, 6.0]
        cell = session._report.doc.cell_by_id(cid)
        inset_panel = next(p for p in cell.spec.panels
                           if p.id == inset["panel"])
        snap = session._report.snapshot_map(cid).get(
            (inset["panel"], inset_panel.layers[0].id))
        np.testing.assert_array_equal(snap, base_arr[4:10, 4:14])
        assert len(_report_figures(messages)) == 1
        assert _states(messages)

    def test_rect_drop_clamps_to_bounds(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel, fig, widget = self._zoom_in_edit(session, messages)
        base_arr = np.asarray(session._report.snapshot_map(cid)[
            (base_panel.id, base_panel.layers[0].id)])
        messages.clear()
        # w/h larger than the 32x32 image clamp to the image size; x/y clamp
        # to keep the rect fully inside.
        self._drop(fig, base_panel, widget, 30.0, 30.0, 40.0, 40.0)
        inset = base_panel.insets[0]
        assert inset["zoom_region"] == [0.0, 0.0, 32.0, 32.0]
        cell = session._report.doc.cell_by_id(cid)
        inset_panel = next(p for p in cell.spec.panels
                           if p.id == inset["panel"])
        snap = session._report.snapshot_map(cid).get(
            (inset["panel"], inset_panel.layers[0].id))
        np.testing.assert_array_equal(snap, base_arr)

    def test_unchanged_region_skips_rebuild(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel, fig, widget = self._zoom_in_edit(session, messages)
        region = list(base_panel.insets[0]["zoom_region"])
        messages.clear()
        self._drop(fig, base_panel, widget, *region)
        assert base_panel.insets[0]["zoom_region"] == region
        assert not _report_figures(messages)
        assert not _states(messages)

    def test_no_rect_outside_edit_mode(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel = _2d_cell(session, messages)
        cx.repfig_add_zoom_callout(session, None,
                                   {"cell_id": cid, "panel_id": base_panel.id})
        mgr = session._report
        fig = mgr._controllers[mgr._window_by_cell[cid]].fig
        assert getattr(fig, "_report_zoom_wiring", None) == []


# ── 9. inset drag/resize geometry persistence ──────────────────────────────────


class TestInsetGeometryPersist:
    def _cell_with_inset_in_edit(self, session, messages):
        """A plain 2-D cell with one zoom callout inset, in edit mode. Returns
        (cid, base_panel, fig, inset_disp_id)."""
        cid, base_panel = _2d_cell(session, messages)
        cx.repfig_add_zoom_callout(session, None,
                                   {"cell_id": cid, "panel_id": base_panel.id})
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        mgr = session._report
        fig = mgr._controllers[mgr._window_by_cell[cid]].fig
        inset_spec_pid = base_panel.insets[0]["panel"]
        inset_disp_id = fig._report_inset_map[inset_spec_pid]
        return cid, base_panel, fig, inset_disp_id

    def test_geometry_change_persists_without_rebuild(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel, fig, inset_disp_id = \
            self._cell_with_inset_in_edit(session, messages)
        mgr = session._report
        window_before = mgr._window_by_cell[cid]
        messages.clear()

        fig._dispatch_event(json.dumps({
            "source": "js", "panel_id": inset_disp_id,
            "event_type": "inset_geometry_change",
            "anchor": [0.1, 0.2], "w_frac": 0.4, "h_frac": 0.3,
        }))

        inset = base_panel.insets[0]
        assert inset["anchor"] == [0.1, 0.2]
        assert inset["w_frac"] == 0.4
        assert inset["h_frac"] == 0.3
        assert "corner" not in inset
        # SAME window id — no rebuild happened (targeted persist only).
        assert mgr._window_by_cell[cid] == window_before
        assert not _report_figures(messages)
        assert _states(messages)
        # The moved geometry rides in the emitted state's figure recipe too.
        fig_dict = _fig_dict_of(messages, cid)
        shipped_inset = fig_dict["panels"][0]["insets"][0]
        assert shipped_inset["anchor"] == [0.1, 0.2]
        assert shipped_inset["w_frac"] == 0.4
        assert shipped_inset["h_frac"] == 0.3

    def test_spec_round_trips_new_inset_keys(self, tem_2d_dataset):
        """anchor/w_frac/h_frac/zoom_region are plain inset-dict entries — a
        to_dict()/from_dict() round trip preserves them verbatim."""
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel, fig, inset_disp_id = \
            self._cell_with_inset_in_edit(session, messages)
        fig._dispatch_event(json.dumps({
            "source": "js", "panel_id": inset_disp_id,
            "event_type": "inset_geometry_change",
            "anchor": [0.15, 0.25], "w_frac": 0.35, "h_frac": 0.22,
        }))
        cell = session._report.doc.cell_by_id(cid)
        d = cell.spec.to_dict()
        from spyde.actions.report.model import FigureSpec
        spec2 = FigureSpec.from_dict(d)
        inset2 = spec2.panels[0].insets[0]
        assert inset2["anchor"] == [0.15, 0.25]
        assert inset2["w_frac"] == 0.35
        assert inset2["h_frac"] == 0.22
        assert inset2["zoom_region"] == base_panel.insets[0]["zoom_region"]

    def test_unknown_inset_id_is_silent_noop(self, tem_2d_dataset):
        session, messages = tem_2d_dataset["window"], tem_2d_dataset["messages"]
        cid, base_panel, fig, _inset_disp_id = \
            self._cell_with_inset_in_edit(session, messages)
        region_before = dict(base_panel.insets[0])
        messages.clear()
        fig._dispatch_event(json.dumps({
            "source": "js", "panel_id": "not-a-real-id",
            "event_type": "inset_geometry_change",
            "anchor": [0.9, 0.9], "w_frac": 0.5, "h_frac": 0.5,
        }))
        assert base_panel.insets[0] == region_before
        assert not _states(messages)

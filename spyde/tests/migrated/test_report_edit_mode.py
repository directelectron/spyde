"""test_report_edit_mode.py — Report Builder draggable-annotation EDIT MODE.

Edit mode renders a figure cell's annotations as draggable anyplotlib WIDGETS
(shape widgets show resize handles; the label is handle-free) instead of static
markers; dragging/resizing one and releasing (``pointer_up``) persists the new
geometry back into ``PanelSpec.annotations`` in DATA coords WITHOUT rebuilding the
iframe (the widget already moved JS-side).

Covered (against a real Qt-free ``Session`` + ``captured_messages``):
  1. Toggle edit ON → the rebuilt figure's panel plots carry widgets matching the
     annotations and NO static markers; toggle OFF → static markers again.
  2. Drag round-trip WITHOUT rebuild: dispatch a ``pointer_up`` event with moved
     px geometry → ``panel.annotations[idx]`` updated in DATA coords, ``mgr.dirty``
     True, a ``report_state`` emitted, and NO new figure build after the drag.
  3. Rect center↔top-left and arrow U/V conversions round-trip exactly (px↔data↔px).
  4. Guards: pointer_up for a removed panel / out-of-range index → no crash/change.
  5. Uncalibrated panel (axes None): drag persists px values identically (identity).
"""
from __future__ import annotations

import json

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


def _states(messages):
    return [m for m in messages if m.get("type") == "report_state"]


def _last_state(messages):
    st = _states(messages)
    assert st, "no report_state emitted"
    return st[-1]["report"]


def _report_figures(messages):
    return [m for m in messages if m.get("type") == "figure"
            and m.get("host") == "report"]


# A 128-sample calibrated axis 0..12 nm (step = 12/127); index i ↔ data i*step.
_STEP = 12.0 / 127.0
_CAL_AXES = {
    "units": "nm",
    "x_axis": [float(v) for v in np.linspace(0.0, 12.0, 128)],
    "y_axis": [float(v) for v in np.linspace(0.0, 12.0, 128)],
}


def _make_calibrated_cell(session, messages, *, axes=_CAL_AXES, annotations=None):
    """Add a figure cell from the signal window, then overwrite its panel's axes +
    annotations to a controlled calibration/geometry so px↔data conversions are exact.
    Returns (mgr, cell_id, panel)."""
    _prime_plot_data(session)
    wid = _signal_wid(session)
    h.report_new(session, None, {})
    h.report_add_figure(session, None, {"source_window_id": wid, "caption": "F"})
    st = _last_state(messages)
    cid = [c for c in st["cells"] if c["cell_type"] == "figure"][-1]["id"]
    mgr = session._report
    cell = mgr.doc.cell_by_id(cid)
    panel = cell.spec.panels[0]
    panel.axes = (dict(axes) if axes is not None else None)
    panel.annotations = [dict(a) for a in (annotations or [])]
    # Rebuild (non-interactive) so the live figure reflects the injected
    # axes/annotations as STATIC markers before any edit-mode toggle.
    mgr.build_figure_window(cell)
    return mgr, cid, panel


def _cell_figure(mgr, cell_id):
    """The live anyplotlib Figure for a cell (from its window controller)."""
    wid = mgr._window_by_cell.get(cell_id)
    assert wid is not None, "cell has no live figure window"
    return mgr._controllers[wid].fig


def _first_plot(fig):
    plots_map = getattr(fig, "_plots_map", None)
    assert plots_map, "figure has no panels"
    return next(iter(plots_map.values()))


def _markers(p2, type_):
    return [m for m in p2._state["markers"] if m["type"] == type_]


def _widgets(p2):
    return list(p2._widgets.values())


def _dispatch_up(fig, p2, widget, fields):
    """Dispatch a pointer_up event carrying moved px geometry to one widget."""
    msg = {"panel_id": p2._id, "event_type": "pointer_up",
           "widget_id": widget.id, **fields}
    fig._dispatch_event(json.dumps(msg))


# ── 1. toggle: widgets in edit mode, static markers otherwise ──────────────────


class TestEditModeToggle:
    def test_edit_on_makes_widgets_off_makes_markers(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "text", "offsets": [[6.0, 6.0]], "texts": ["hi"],
                "color": "#ff9800", "fontsize": 12},
               {"kind": "circle", "offsets": [[6.0, 6.0]], "radius": _STEP * 5.0,
                "edgecolors": "#ff9800"}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)

        # Not editing yet: the current build is static markers, no widgets.
        p2 = _first_plot(_cell_figure(mgr, cid))
        assert not _widgets(p2)
        assert len(_markers(p2, "texts")) == 1
        assert len(_markers(p2, "circles")) == 1

        # Toggle edit ON → rebuild in interactive mode: widgets, no static markers.
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        p2 = _first_plot(_cell_figure(mgr, cid))
        w = _widgets(p2)
        assert len(w) == 2
        kinds = sorted(x._type for x in w)
        assert kinds == ["circle", "label"]   # text → label widget
        assert _markers(p2, "texts") == []
        assert _markers(p2, "circles") == []
        # SHAPE widgets (circle) show resize handles — the nodes ARE the resize
        # affordance; the LABEL keeps handles hidden (reposition-only, no resize
        # DOF → a bare anchor dot would just clutter the text).
        circle = next(x for x in w if x._type == "circle")
        label0 = next(x for x in w if x._type == "label")
        assert circle.get("show_handles") is True
        assert label0.get("show_handles") is False
        # Widget geometry is the pixel-converted center (data 6 → index 63.5).
        label = next(x for x in w if x._type == "label")
        assert abs(label.get("x") - 63.5) < 0.6
        assert abs(label.get("y") - 63.5) < 0.6

        # Toggle edit OFF → back to static markers, no widgets.
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": False})
        p2 = _first_plot(_cell_figure(mgr, cid))
        assert not _widgets(p2)
        assert len(_markers(p2, "texts")) == 1
        assert len(_markers(p2, "circles")) == 1

    def test_toggle_same_value_is_noop(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(
            session, messages,
            annotations=[{"kind": "text", "offsets": [[6.0, 6.0]], "texts": ["x"]}])
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        messages.clear()
        # Setting editing True again → no membership change → no rebuild/emit.
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        assert _report_figures(messages) == []
        assert _states(messages) == []


# ── 2. drag round-trip WITHOUT rebuild ──────────────────────────────────────────


class TestDragPersistNoRebuild:
    def test_text_drag_updates_data_no_rebuild(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "text", "offsets": [[6.0, 6.0]], "texts": ["hi"],
                "color": "#ff9800", "fontsize": 12}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        messages.clear()
        mgr.dirty = False
        # Drag the label to pixel (95.25, 31.75): data = index * step.
        _dispatch_up(fig, p2, widget, {"x": 95.25, "y": 31.75})

        off = panel.annotations[0]["offsets"][0]
        assert abs(off[0] - 95.25 * _STEP) < 1e-6   # 9.0 nm
        assert abs(off[1] - 31.75 * _STEP) < 1e-6   # 3.0 nm
        # Text + color untouched.
        assert panel.annotations[0]["texts"] == ["hi"]
        assert panel.annotations[0]["color"] == "#ff9800"
        # dirty flipped, a report_state emitted, and NO new figure build.
        assert mgr.dirty is True
        assert len(_states(messages)) >= 1
        assert _report_figures(messages) == []

    def test_circle_drag_updates_offset_and_radius(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "circle", "offsets": [[6.0, 6.0]], "radius": _STEP * 5.0,
                "edgecolors": "#ff9800", "linewidths": 1.5}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        messages.clear()
        _dispatch_up(fig, p2, widget, {"cx": 20.0, "cy": 40.0, "r": 10.0})

        off = panel.annotations[0]["offsets"][0]
        assert abs(off[0] - 20.0 * _STEP) < 1e-6
        assert abs(off[1] - 40.0 * _STEP) < 1e-6
        # radius uses the mean axis scale (== step here since x/y equal).
        assert abs(panel.annotations[0]["radius"] - 10.0 * _STEP) < 1e-6
        # cosmetic keys untouched.
        assert panel.annotations[0]["edgecolors"] == "#ff9800"
        assert panel.annotations[0]["linewidths"] == 1.5
        assert _report_figures(messages) == []


# ── 3. rect center↔top-left and arrow U/V round-trip ────────────────────────────


class TestGeometryRoundTrip:
    def test_rect_center_topleft_roundtrip(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        # data-coord rect: center (6,6), width step*40, height step*20.
        ann = [{"kind": "rect", "offsets": [[6.0, 6.0]],
                "widths": [_STEP * 40.0], "heights": [_STEP * 20.0],
                "edgecolors": "#ff9800"}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        # The widget was built from the spec: its px top-left is center_px - size/2.
        # center data 6 → index 63.5; w px 40, h px 20 → x = 63.5-20 = 43.5,
        # y = 63.5 - 10 = 53.5.
        assert abs(widget.get("x") - 43.5) < 0.6
        assert abs(widget.get("y") - 53.5) < 0.6
        assert abs(widget.get("w") - 40.0) < 1e-6
        assert abs(widget.get("h") - 20.0) < 1e-6

        # Drag: new top-left px (10, 12), new size px (60, 30).
        _dispatch_up(fig, p2, widget, {"x": 10.0, "y": 12.0, "w": 60.0, "h": 30.0})
        # spec offset is the CENTER: (10+30, 12+15) px = (40, 27) index → data.
        off = panel.annotations[0]["offsets"][0]
        assert abs(off[0] - 40.0 * _STEP) < 1e-6
        assert abs(off[1] - 27.0 * _STEP) < 1e-6
        assert abs(panel.annotations[0]["widths"][0] - 60.0 * _STEP) < 1e-6
        assert abs(panel.annotations[0]["heights"][0] - 30.0 * _STEP) < 1e-6

    def test_arrow_uv_roundtrip_signed(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "arrow", "offsets": [[4.0, 4.0]],
                "U": [_STEP * 10.0], "V": [_STEP * -8.0], "edgecolors": "#ff9800"}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        # Built widget: tail px = 4/step ≈ 42.333, u=10, v=-8.
        assert abs(widget.get("x") - 4.0 / _STEP) < 0.6
        assert abs(widget.get("u") - 10.0) < 1e-6
        assert abs(widget.get("v") - (-8.0)) < 1e-6

        _dispatch_up(fig, p2, widget, {"x": 50.0, "y": 60.0, "u": 15.0, "v": -12.0})
        off = panel.annotations[0]["offsets"][0]
        assert abs(off[0] - 50.0 * _STEP) < 1e-6
        assert abs(off[1] - 60.0 * _STEP) < 1e-6
        assert abs(panel.annotations[0]["U"][0] - 15.0 * _STEP) < 1e-6
        assert abs(panel.annotations[0]["V"][0] - (-12.0 * _STEP)) < 1e-6
        assert panel.annotations[0]["V"][0] < 0


# ── 3b. RESIZE (not just move) persistence ──────────────────────────────────────
#
# Edit-mode widgets now show resize NODES; a resize emits the same pointer_up with
# the widget's FINAL geometry, so the drag-persist path must write the resized
# radius / size / arrow-vector — not merely a translated offset.


class TestResizePersistence:
    def test_circle_radius_resize_persists_data_radius(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "circle", "offsets": [[6.0, 6.0]], "radius": _STEP * 5.0,
                "edgecolors": "#ff9800"}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        messages.clear()
        # Pure radius grow: the east-edge node dragged out → r 5px → 22px, center
        # UNCHANGED (cx/cy identical to the built position, index 63.5).
        _dispatch_up(fig, p2, widget, {"cx": 63.5, "cy": 63.5, "r": 22.0})
        # Center in data coords is unchanged (6 nm); radius is the NEW 22px * step.
        off = panel.annotations[0]["offsets"][0]
        assert abs(off[0] - 6.0) < 1e-6 and abs(off[1] - 6.0) < 1e-6
        assert abs(panel.annotations[0]["radius"] - 22.0 * _STEP) < 1e-6
        assert _report_figures(messages) == []   # no rebuild

    def test_rect_corner_resize_persists_center_and_size(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        # data-coord rect: center (6,6), width step*40, height step*20 → px built
        # top-left (43.5, 53.5), w40 h20.
        ann = [{"kind": "rect", "offsets": [[6.0, 6.0]],
                "widths": [_STEP * 40.0], "heights": [_STEP * 20.0],
                "edgecolors": "#ff9800"}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        messages.clear()
        # Anchor the TOP-LEFT corner (43.5, 53.5) and drag the BOTTOM-RIGHT out:
        # new size 80x50 px (anyplotlib's opposite-corner anchor keeps x/y fixed).
        _dispatch_up(fig, p2, widget,
                     {"x": 43.5, "y": 53.5, "w": 80.0, "h": 50.0})
        # New CENTER = top-left + size/2 = (43.5+40, 53.5+25) px = (83.5, 78.5).
        off = panel.annotations[0]["offsets"][0]
        assert abs(off[0] - 83.5 * _STEP) < 1e-6
        assert abs(off[1] - 78.5 * _STEP) < 1e-6
        # BOTH width AND height grew (a resize, not a move).
        assert abs(panel.annotations[0]["widths"][0] - 80.0 * _STEP) < 1e-6
        assert abs(panel.annotations[0]["heights"][0] - 50.0 * _STEP) < 1e-6
        assert _report_figures(messages) == []

    def test_arrow_tail_reshape_keeps_head_fixed(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        # data arrow: tail (4,4), U/V = step*10 / step*8 → head data (4+step*10,
        # 4+step*8). px: tail 4/step, u=10, v=8 → head px (4/step+10, 4/step+8).
        ann = [{"kind": "arrow", "offsets": [[4.0, 4.0]],
                "U": [_STEP * 10.0], "V": [_STEP * 8.0], "edgecolors": "#ff9800"}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        tail_px = 4.0 / _STEP
        head_px_x = tail_px + 10.0
        head_px_y = tail_px + 8.0
        # anyplotlib's tail reshape: tail moves to the cursor, HEAD STAYS — it emits
        # x,y = new tail px and u,v = (head - new tail) so head px is invariant.
        # Move the tail to (tail+15, tail-6) px; the widget re-solves u,v to keep
        # the head at (head_px_x, head_px_y).
        new_tail_x = tail_px + 15.0
        new_tail_y = tail_px - 6.0
        new_u = head_px_x - new_tail_x
        new_v = head_px_y - new_tail_y

        messages.clear()
        _dispatch_up(fig, p2, widget,
                     {"x": new_tail_x, "y": new_tail_y, "u": new_u, "v": new_v})
        # Tail offset moved (in data coords).
        off = panel.annotations[0]["offsets"][0]
        assert abs(off[0] - new_tail_x * _STEP) < 1e-6
        assert abs(off[1] - new_tail_y * _STEP) < 1e-6
        # U/V CHANGED from the originals.
        U = panel.annotations[0]["U"][0]
        V = panel.annotations[0]["V"][0]
        assert abs(U - new_u * _STEP) < 1e-6
        assert abs(V - new_v * _STEP) < 1e-6
        assert U != _STEP * 10.0 and V != _STEP * 8.0
        # THE HEAD (offset + U, offset + V) in DATA coords is UNCHANGED — the tail
        # reshape must pivot about a fixed head.
        head_x = off[0] + U
        head_y = off[1] + V
        assert abs(head_x - head_px_x * _STEP) < 1e-6
        assert abs(head_y - head_px_y * _STEP) < 1e-6
        assert _report_figures(messages) == []


# ── 4. guards ───────────────────────────────────────────────────────────────────


class TestGuards:
    def test_pointer_up_after_panel_removed_no_change(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "text", "offsets": [[6.0, 6.0]], "texts": ["hi"]}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        # Detach the panel from the cell's spec (simulate a concurrent removal).
        cell = mgr.doc.cell_by_id(cid)
        cell.spec.panels = []
        before = json.dumps(panel.annotations)
        messages.clear()
        mgr.dirty = False
        # Should be a no-op guarded by "panel_spec not in cell.spec.panels".
        _dispatch_up(fig, p2, widget, {"x": 99.0, "y": 99.0})
        assert json.dumps(panel.annotations) == before
        assert mgr.dirty is False

    def test_pointer_up_index_out_of_range_no_change(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "text", "offsets": [[6.0, 6.0]], "texts": ["hi"]}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        # Truncate the annotations list so the wired ann_index (0) is now OOB.
        panel.annotations = []
        messages.clear()
        mgr.dirty = False
        _dispatch_up(fig, p2, widget, {"x": 99.0, "y": 99.0})
        assert panel.annotations == []
        assert mgr.dirty is False


# ── 5. uncalibrated panel: identity px==data ────────────────────────────────────


class TestUncalibratedIdentity:
    def test_drag_persists_pixel_values_identically(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "text", "offsets": [[6.0, 6.0]], "texts": ["hi"]}]
        # axes=None → uncalibrated → index == pixel == data.
        mgr, cid, panel = _make_calibrated_cell(session, messages, axes=None,
                                                annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]
        # Built widget position is the raw data value (identity).
        assert widget.get("x") == 6.0
        assert widget.get("y") == 6.0

        _dispatch_up(fig, p2, widget, {"x": 42.0, "y": 88.0})
        off = panel.annotations[0]["offsets"][0]
        assert off == [42.0, 88.0]   # px persisted verbatim


class TestRectArrowUncalibratedRoundTrip:
    """Rect center↔top-left and arrow U/V px↔data↔px must be exact even without
    calibration (the identity case still must reconcile the center/top-left frame)."""

    def test_rect_center_topleft_identity(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "rect", "offsets": [[50.0, 50.0]],
                "widths": [40.0], "heights": [20.0]}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, axes=None,
                                                annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]
        # center (50,50) w40 h20 → top-left (30,40).
        assert widget.get("x") == 30.0
        assert widget.get("y") == 40.0
        _dispatch_up(fig, p2, widget, {"x": 30.0, "y": 40.0, "w": 40.0, "h": 20.0})
        # Round-trips back to the original center + size.
        off = panel.annotations[0]["offsets"][0]
        assert off == [50.0, 50.0]
        assert panel.annotations[0]["widths"] == [40.0]
        assert panel.annotations[0]["heights"] == [20.0]


# ── 6. build sets edit_chrome + figure markers ──────────────────────────────────


def _panel_selected(messages):
    return [m for m in messages if m.get("type") == "report_panel_selected"]


class TestFigureBuildEditChrome:
    def test_interactive_sets_edit_chrome_and_markers(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        fig_anns = [{"id": "fm1", "kind": "text", "x": 0.5, "y": 0.5,
                     "text": "Fig label"}]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        # Put figure-level annotations on the cell's spec, then rebuild interactive.
        mgr.doc.cell_by_id(cid).spec.annotations = [dict(a) for a in fig_anns]
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        assert fig.edit_chrome is True
        got = fig.figure_markers
        assert [m["kind"] for m in got] == ["text"]
        assert got[0]["text"] == "Fig label"

    def test_non_interactive_sets_markers_not_chrome(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        mgr.doc.cell_by_id(cid).spec.annotations = [
            {"id": "c1", "kind": "circle", "x": 0.2, "y": 0.3, "r": 0.1}]
        # Rebuild NON-interactive (default).
        mgr.build_figure_window(mgr.doc.cell_by_id(cid))
        fig = _cell_figure(mgr, cid)
        assert fig.edit_chrome is False
        assert [m["kind"] for m in fig.figure_markers] == ["circle"]

    def test_panel_map_stashed(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        fig = _cell_figure(mgr, cid)
        pmap = getattr(fig, "_report_panel_map", None)
        assert isinstance(pmap, dict) and pmap
        # p1 (the single panel's spec id) maps to the base plot's dispatch id.
        p2 = _first_plot(fig)
        assert pmap.get("p1") == p2._id


# ── 7. panel selection via events + action ──────────────────────────────────────


class TestPanelSelection:
    def test_panel_pointer_down_selects_and_outlines(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        messages.clear()
        # A genuine panel click (misses widgets) → pointer_down on the panel plot.
        fig._dispatch_event(json.dumps(
            {"panel_id": p2._id, "event_type": "pointer_down"}))
        sel = _panel_selected(messages)
        assert sel and sel[-1]["cell_id"] == cid
        assert sel[-1]["panel_id"] == "p1"           # the SPEC panel id
        assert fig.selected_panel == p2._id           # dispatch id on the trait
        assert mgr._selected[cid] == "p1"

    def test_figure_background_deselects(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        # First select the panel.
        fig._dispatch_event(json.dumps(
            {"panel_id": p2._id, "event_type": "pointer_down"}))
        assert fig.selected_panel == p2._id
        messages.clear()
        # Then click the figure background → deselect (figure-level).
        fig._dispatch_event(json.dumps(
            {"panel_id": "", "event_type": "pointer_down",
             "figure_background": True}))
        sel = _panel_selected(messages)
        assert sel and sel[-1]["panel_id"] is None
        assert fig.selected_panel == ""
        assert mgr._selected[cid] is None

    def test_repfig_select_panel_action(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        messages.clear()
        cx.repfig_select_panel(session, None, {"cell_id": cid, "panel_id": "p1"})
        sel = _panel_selected(messages)
        assert sel and sel[-1]["panel_id"] == "p1"
        assert fig.selected_panel == p2._id
        # And deselect via the action (panel_id null).
        messages.clear()
        cx.repfig_select_panel(session, None, {"cell_id": cid, "panel_id": None})
        sel = _panel_selected(messages)
        assert sel and sel[-1]["panel_id"] is None
        assert fig.selected_panel == ""

    def test_select_unknown_panel_is_figure_level(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        messages.clear()
        cx.repfig_select_panel(session, None,
                               {"cell_id": cid, "panel_id": "does-not-exist"})
        sel = _panel_selected(messages)
        assert sel and sel[-1]["panel_id"] is None      # normalised to figure-level

    def test_selection_survives_rebuild(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        cx.repfig_select_panel(session, None, {"cell_id": cid, "panel_id": "p1"})
        # A rebuild (e.g. an annotation edit) must re-apply the outline.
        mgr.build_figure_window(mgr.doc.cell_by_id(cid))
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        assert fig.selected_panel == p2._id


# ── 8. figure-level marker drag persists (no rebuild) ───────────────────────────


class TestFigureMarkerDrag:
    def test_marker_drag_persists_no_rebuild(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        mgr.doc.cell_by_id(cid).spec.annotations = [
            {"id": "am1", "kind": "arrow", "x": 0.1, "y": 0.1, "u": 0.2, "v": 0.2}]
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        messages.clear()
        mgr.dirty = False
        # Simulate anyplotlib's figure-marker drag end: it has ALREADY merged the
        # moved fraction fields into fig.figure_markers before firing pointer_up.
        fig._dispatch_event(json.dumps({
            "panel_id": "", "event_type": "pointer_up",
            "figure_marker": True, "marker_id": "am1",
            "x": 0.5, "y": 0.6, "u": 0.3, "v": 0.3,
        }))
        ann = mgr.doc.cell_by_id(cid).spec.annotations[0]
        assert (ann["x"], ann["y"], ann["u"], ann["v"]) == (0.5, 0.6, 0.3, 0.3)
        assert mgr.dirty is True
        assert len(_states(messages)) >= 1
        # No figure rebuild (the marker already moved JS-side).
        assert _report_figures(messages) == []


# ── 9. layout spacing + figure-annotation add/update/remove actions ─────────────


class TestLayoutAndFigureAnnotationActions:
    def test_set_layout_applies_hspace_wspace(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        cx.repfig_set_layout(session, None,
                             {"cell_id": cid, "hspace": 0.3, "wspace": 0.25})
        spec = mgr.doc.cell_by_id(cid).spec
        assert spec.layout["hspace"] == 0.3
        assert spec.layout["wspace"] == 0.25
        # The rebuilt figure has them applied.
        fig = _cell_figure(mgr, cid)
        assert fig._hspace == 0.3
        assert fig._wspace == 0.25

    def test_set_layout_clamps_to_unit_range(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        cx.repfig_set_layout(session, None,
                             {"cell_id": cid, "hspace": 5.0, "wspace": -2.0})
        spec = mgr.doc.cell_by_id(cid).spec
        assert spec.layout["hspace"] == 1.0     # clamped high
        assert spec.layout["wspace"] == 0.0     # clamped low

    def test_add_update_remove_fig_annotation_round_trip(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        spec = mgr.doc.cell_by_id(cid).spec

        # Add — an id is assigned when missing.
        cx.repfig_add_fig_annotation(session, None, {
            "cell_id": cid,
            "annotation": {"kind": "text", "x": 0.5, "y": 0.5, "text": "L"}})
        assert len(spec.annotations) == 1
        assert spec.annotations[0]["id"]                 # id auto-assigned
        assert spec.annotations[0]["text"] == "L"
        # The live figure drew it as a figure marker.
        fig = _cell_figure(mgr, cid)
        assert [m["kind"] for m in fig.figure_markers] == ["text"]

        # Update index 0 — id preserved when the incoming dict omits it.
        old_id = spec.annotations[0]["id"]
        cx.repfig_update_fig_annotation(session, None, {
            "cell_id": cid, "index": 0,
            "annotation": {"kind": "text", "x": 0.7, "y": 0.8, "text": "L2"}})
        assert spec.annotations[0]["text"] == "L2"
        assert spec.annotations[0]["x"] == 0.7
        assert spec.annotations[0]["id"] == old_id

        # Bad kind → rejected (no change).
        cx.repfig_update_fig_annotation(session, None, {
            "cell_id": cid, "index": 0,
            "annotation": {"kind": "polygon", "x": 0.1, "y": 0.1}})
        assert spec.annotations[0]["text"] == "L2"

        # Remove.
        cx.repfig_remove_fig_annotation(session, None, {"cell_id": cid, "index": 0})
        assert spec.annotations == []

    def test_add_fig_annotation_rejects_bad_kind(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        spec = mgr.doc.cell_by_id(cid).spec
        cx.repfig_add_fig_annotation(session, None, {
            "cell_id": cid, "annotation": {"kind": "polygon", "x": 0.5, "y": 0.5}})
        assert spec.annotations == []


# ── 10. panel drag-swap (figure-level event → grid_pos swap + rebuild) ──────────


def _make_two_panel_cell(session, messages):
    """A 2-panel (1×2 grid) figure cell in EDIT MODE. Builds a single-panel cell
    then tile-rights a navigator into it. Returns (mgr, cell_id, fig)."""
    _prime_plot_data(session)
    sig_wid = _signal_wid(session)
    nav_wid = None
    for p in session._plots:
        if getattr(p, "is_navigator", False) and p.window_id is not None:
            nav_wid = p.window_id
            break
    h.report_new(session, None, {})
    h.report_add_figure(session, None, {"source_window_id": sig_wid, "caption": "F"})
    st = _last_state(messages)
    cid = [c for c in st["cells"] if c["cell_type"] == "figure"][-1]["id"]
    src = nav_wid if nav_wid is not None else sig_wid
    cx.repfig_compose(session, None,
                      {"cell_id": cid, "mode": "tile-right", "source_window_id": src})
    mgr = session._report
    cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
    return mgr, cid, _cell_figure(mgr, cid)


def _dispatch_panel_swap(fig, src_disp, tgt_disp):
    fig._dispatch_event(json.dumps({
        "panel_id": "", "event_type": "pointer_up", "panel_swap": True,
        "source_panel_id": src_disp, "target_panel_id": tgt_disp,
    }))


class TestPanelSwap:
    def test_swap_exchanges_grid_pos_and_rebuilds(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        mgr, cid, fig = _make_two_panel_cell(session, messages)
        spec = mgr.doc.cell_by_id(cid).spec
        assert len(spec.panels) == 2
        pmap = dict(fig._report_panel_map)   # spec_pid → dispatch id
        # The two panels' spec ids and their current grid positions.
        pos_before = {p.id: list(p.grid_pos) for p in spec.panels}
        (pid_a, disp_a), (pid_b, disp_b) = list(pmap.items())[:2]
        assert pos_before[pid_a] != pos_before[pid_b]

        messages.clear()
        _dispatch_panel_swap(fig, disp_a, disp_b)

        # grid_pos EXCHANGED between the two panels.
        panel_a = next(p for p in spec.panels if p.id == pid_a)
        panel_b = next(p for p in spec.panels if p.id == pid_b)
        assert panel_a.grid_pos == pos_before[pid_b]
        assert panel_b.grid_pos == pos_before[pid_a]
        assert mgr.dirty is True
        # A rebuild WAS emitted (unlike the no-rebuild annotation-drag path).
        assert _report_figures(messages), "panel swap must rebuild the figure"

    def test_swap_same_panel_is_noop(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        mgr, cid, fig = _make_two_panel_cell(session, messages)
        spec = mgr.doc.cell_by_id(cid).spec
        pmap = dict(fig._report_panel_map)
        disp_a = next(iter(pmap.values()))
        pos_before = {p.id: list(p.grid_pos) for p in spec.panels}
        messages.clear()
        mgr.dirty = False
        _dispatch_panel_swap(fig, disp_a, disp_a)   # same → no-op
        assert {p.id: list(p.grid_pos) for p in spec.panels} == pos_before
        assert mgr.dirty is False
        assert _report_figures(messages) == []

    def test_swap_unknown_dispatch_id_is_noop(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        mgr, cid, fig = _make_two_panel_cell(session, messages)
        spec = mgr.doc.cell_by_id(cid).spec
        pmap = dict(fig._report_panel_map)
        disp_a = next(iter(pmap.values()))
        pos_before = {p.id: list(p.grid_pos) for p in spec.panels}
        messages.clear()
        mgr.dirty = False
        _dispatch_panel_swap(fig, disp_a, 999999)   # target not a panel
        assert {p.id: list(p.grid_pos) for p in spec.panels} == pos_before
        assert mgr.dirty is False
        assert _report_figures(messages) == []


# ── 11. in-place updates skip the figure rebuild (no iframe flash) ──────────────


class TestFigAnnotationInPlaceUpdate:
    """Figure-level annotation add/update/remove push fig.set_figure_markers on the
    LIVE figure instead of rebuilding — a report_state is emitted, but NO new figure
    build message (so the iframe never reloads / flashes)."""

    def test_update_fig_annotation_no_rebuild(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        cx.repfig_add_fig_annotation(session, None, {
            "cell_id": cid,
            "annotation": {"kind": "circle", "x": 0.5, "y": 0.5, "r": 0.1,
                           "color": "#ff9800"}})
        fig = _cell_figure(mgr, cid)
        messages.clear()
        mgr.dirty = False

        # Update the color → in-place set_figure_markers, no rebuild.
        cx.repfig_update_fig_annotation(session, None, {
            "cell_id": cid, "index": 0,
            "annotation": {"kind": "circle", "x": 0.5, "y": 0.5, "r": 0.1,
                           "color": "#00b0ff"}})
        assert len(_states(messages)) >= 1
        assert _report_figures(messages) == []
        assert mgr.dirty is True
        # The LIVE figure's marker layer reflects the new color.
        assert fig.figure_markers[0]["color"] == "#00b0ff"

    def test_add_and_remove_fig_annotation_no_rebuild(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        mgr, cid, _panel = _make_calibrated_cell(session, messages)
        fig = _cell_figure(mgr, cid)
        messages.clear()
        cx.repfig_add_fig_annotation(session, None, {
            "cell_id": cid, "annotation": {"kind": "text", "x": 0.3, "y": 0.3,
                                           "text": "hi"}})
        assert _report_figures(messages) == []          # no rebuild on add
        assert [m["kind"] for m in fig.figure_markers] == ["text"]
        messages.clear()
        cx.repfig_remove_fig_annotation(session, None, {"cell_id": cid, "index": 0})
        assert _report_figures(messages) == []          # no rebuild on remove
        assert fig.figure_markers == []


class TestPanelAnnotationInPlaceUpdate:
    """A panel-annotation edit (color/text/geometry) WHILE in edit mode pushes
    widget.set on the matching live widget — report_state but NO figure rebuild, and
    the live widget's _data carries the new value. A NON-edit-mode edit rebuilds."""

    def test_edit_mode_color_update_no_rebuild(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "circle", "offsets": [[6.0, 6.0]], "radius": _STEP * 5.0,
                "edgecolors": "#ff9800", "linewidths": 1.5}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        messages.clear()
        mgr.dirty = False
        # Change ONLY the color — an in-place widget.set, no rebuild.
        new_ann = dict(ann[0])
        new_ann["edgecolors"] = "#00b0ff"
        cx.repfig_update_annotation(session, None, {
            "cell_id": cid, "panel_id": panel.id, "index": 0, "annotation": new_ann})

        # Spec updated, state emitted, NO figure rebuild.
        assert panel.annotations[0]["edgecolors"] == "#00b0ff"
        assert len(_states(messages)) >= 1
        assert _report_figures(messages) == []
        assert mgr.dirty is True
        # The LIVE widget's _data reflects the new color (mapped edgecolors→color).
        assert widget.get("color") == "#00b0ff"

    def test_edit_mode_text_update_no_rebuild(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "text", "offsets": [[6.0, 6.0]], "texts": ["hi"],
                "color": "#ff9800", "fontsize": 12}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        fig = _cell_figure(mgr, cid)
        p2 = _first_plot(fig)
        widget = _widgets(p2)[0]

        messages.clear()
        new_ann = dict(ann[0])
        new_ann["texts"] = ["updated"]
        cx.repfig_update_annotation(session, None, {
            "cell_id": cid, "panel_id": panel.id, "index": 0, "annotation": new_ann})
        assert panel.annotations[0]["texts"] == ["updated"]
        assert _report_figures(messages) == []
        assert widget.get("text") == "updated"

    def test_non_edit_mode_update_rebuilds(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "circle", "offsets": [[6.0, 6.0]], "radius": _STEP * 5.0,
                "edgecolors": "#ff9800"}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        # NOT in edit mode → the update takes the rebuild path.
        messages.clear()
        new_ann = dict(ann[0])
        new_ann["edgecolors"] = "#00b0ff"
        cx.repfig_update_annotation(session, None, {
            "cell_id": cid, "panel_id": panel.id, "index": 0, "annotation": new_ann})
        assert panel.annotations[0]["edgecolors"] == "#00b0ff"
        assert _report_figures(messages), "non-edit update must rebuild the figure"

    def test_kind_change_rebuilds_even_in_edit_mode(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        ann = [{"kind": "circle", "offsets": [[6.0, 6.0]], "radius": _STEP * 5.0,
                "edgecolors": "#ff9800"}]
        mgr, cid, panel = _make_calibrated_cell(session, messages, annotations=ann)
        cx.repfig_set_edit_mode(session, None, {"cell_id": cid, "editing": True})
        messages.clear()
        # Circle → rect: a kind change restructures the overlay → must rebuild.
        cx.repfig_update_annotation(session, None, {
            "cell_id": cid, "panel_id": panel.id, "index": 0,
            "annotation": {"kind": "rect", "offsets": [[6.0, 6.0]],
                           "widths": [_STEP * 8.0], "heights": [_STEP * 8.0],
                           "edgecolors": "#ff9800"}})
        assert panel.annotations[0]["kind"] == "rect"
        assert _report_figures(messages), "kind change must rebuild"

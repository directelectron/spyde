"""
compose.py — Report Builder Phase 2: combined report figures.

The user builds a COMBINED figure by dragging a source window's figure/pill onto a
report figure cell. Depending on compatibility the drop can OVERLAY (add a layer to
the target panel), TILE (grow the grid with the source as a new panel on a side), or
CALLOUT (add a small inset panel referencing the source). Once combined, the layers,
annotations, and panels are edited through the ``repfig_*`` staged handlers here.

All handlers share the uniform ``fn(session, plot, payload)`` signature and are
registered in :data:`spyde.actions.registry.STAGED_HANDLERS`; ``plot`` is ignored
(the report sidebar isn't tied to a signal window — the cell is addressed by
``cell_id``). Every MUTATION re-emits the cell's figure (via
``ReportManager.build_figure_window``) AND the authoritative ``report_state`` (whose
figure cells now carry the pixel-free ``figure`` recipe dict).

Message contracts (the renderer is written against these EXACT shapes):

* ``repfig_compose_options`` — the query reply:
    ``{"type":"repfig_compose_options","cell_id","source_window_id",
       "options":[...subset of "overlay"/"callout"/"tile-up"/"tile-down"/
                  "tile-left"/"tile-right"...],
       "detail":{"same_shape":bool,"nav_signal_pair":bool}}``
* ``report_state`` — re-emitted after every mutation (see handlers.ReportManager).
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.backend import ipc
from spyde.actions.report.handlers import _manager, _resolve_source_plot, _snapshot_plot
from spyde.actions.report.model import (
    LayerSpec, PanelSpec, SignalRef, new_layer_id,
)

log = logging.getLogger(__name__)

# The default overlay-layer colormap cycle (distinct from a typical gray/viridis
# base) so a composed overlay reads as a separate image.
_OVERLAY_CMAP_CYCLE = ["magma", "cividis", "plasma", "inferno", "cool", "spring"]

_TILE_MODES = ("tile-up", "tile-down", "tile-left", "tile-right")


# ── shared helpers ────────────────────────────────────────────────────────────


def _cell(mgr, cell_id):
    if not mgr.open:
        return None
    cell = mgr.doc.cell_by_id(cell_id)
    if cell is None or cell.cell_type != "figure" or cell.placeholder:
        return None
    return cell


def _target_base_shape(mgr, cell):
    """The (H, W) of the target cell's BASE layer snapshot, or None."""
    arr = mgr.primary_snapshot(cell.id)
    if isinstance(arr, np.ndarray) and arr.ndim >= 2:
        return tuple(arr.shape[:2])
    return None


def _target_base_source(cell):
    """The SignalRef of the target cell's base (primary) layer — the plot the cell
    was snapshotted from."""
    pl = cell.spec.primary_layer if cell.spec else None
    return pl.source if pl is not None else None


def _is_nav_signal_pair(session, cell, source_plot):
    """True when the source plot and the target cell's base source are a
    NAVIGATOR ↔ SIGNAL pair of the SAME signal tree (so a callout makes sense —
    e.g. a navigator cell with a diffraction-pattern callout, or vice versa)."""
    if source_plot is None or cell.spec is None:
        return False
    ref = _target_base_source(cell)
    target_plot = ref.resolve(session) if ref is not None else None
    if target_plot is None:
        return False
    src_tree = getattr(source_plot, "signal_tree", None)
    tgt_tree = getattr(target_plot, "signal_tree", None)
    if src_tree is None or src_tree is not tgt_tree:
        return False
    src_nav = bool(getattr(source_plot, "is_navigator", False))
    tgt_nav = bool(getattr(target_plot, "is_navigator", False))
    # One navigator + one signal of the same tree.
    return src_nav != tgt_nav


def _source_nav_region(session, source_plot):
    """The source's current nav-selector region ``(x, y, w, h)`` when it has an
    integrating navigator selector — used as the callout connector region. None
    when there's no region selector (a crosshair) or it can't be read."""
    try:
        mm = getattr(source_plot, "multiplot_manager", None)
        if mm is None:
            return None
        # Search every navigator's selectors for one whose child is this plot.
        for nav_pw, selectors in getattr(mm, "navigation_selectors", {}).items():
            for sel in selectors:
                if not getattr(sel, "is_integrating", False):
                    continue
                idx = np.asarray(sel.get_selected_indices())
                if idx.ndim == 2 and idx.shape[0] >= 1 and idx.shape[1] >= 2:
                    xs, ys = idx[:, 0], idx[:, 1]
                    x0, y0 = int(xs.min()), int(ys.min())
                    return (x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1))
    except Exception as e:
        log.debug("reading source nav region failed: %s", e)
    return None


def _rebuild_and_emit(mgr, cell) -> None:
    """Rebuild the cell's live figure window and emit the authoritative state."""
    mgr._offline.discard(cell.id)
    mgr.build_figure_window(cell)
    mgr.dirty = True
    mgr.emit_state()


# ── query: which compose modes are compatible for this drop ───────────────────


def repfig_query_compose(session, plot, payload) -> None:
    """Reply with the compose modes compatible for dropping ``source_window_id``
    onto figure cell ``cell_id``: overlay when the source frame shape matches the
    target panel's base; callout when they're a navigator↔signal pair of one tree;
    tiles ALWAYS."""
    mgr = _manager(session)
    cell_id = payload.get("cell_id")
    source_window_id = payload.get("source_window_id")
    cell = _cell(mgr, cell_id)
    src = _resolve_source_plot(session, source_window_id)

    same_shape = False
    nav_signal_pair = False
    options = list(_TILE_MODES)   # tiles always available

    if cell is not None and src is not None:
        base_shape = _target_base_shape(mgr, cell)
        src_frame = getattr(src, "current_data", None)
        src_shape = (tuple(src_frame.shape[:2])
                     if isinstance(src_frame, np.ndarray) and src_frame.ndim >= 2
                     else None)
        if base_shape is not None and src_shape is not None and base_shape == src_shape:
            same_shape = True
            options.insert(0, "overlay")
        if _is_nav_signal_pair(session, cell, src):
            nav_signal_pair = True
            options.append("callout")

    ipc.emit({
        "type": "repfig_compose_options",
        "cell_id": cell_id,
        "source_window_id": source_window_id,
        "options": options,
        "detail": {"same_shape": bool(same_shape),
                   "nav_signal_pair": bool(nav_signal_pair)},
    })


# ── compose: mutate the FigureSpec per the chosen mode ────────────────────────


def repfig_compose(session, plot, payload) -> None:
    """Combine ``source_window_id`` into figure cell ``cell_id`` in ``mode``
    (``overlay`` / ``tile-up|down|left|right`` / ``callout``). Snapshots the source
    NOW and mutates the cell's FigureSpec, then rebuilds + re-emits."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None:
        ipc.emit_error("repfig_compose: figure cell not found.")
        return
    src = _resolve_source_plot(session, payload.get("source_window_id"))
    if src is None:
        ipc.emit_error("repfig_compose: source window not found.")
        return
    snap = _snapshot_plot(src)
    if snap is None:
        ipc.emit_error("repfig_compose: source window has no image to snapshot.")
        return
    src_spec, src_map = snap
    src_panel = src_spec.panels[0] if src_spec.panels else None
    if src_panel is None or not src_panel.layers:
        ipc.emit_error("repfig_compose: source has no layer to compose.")
        return
    mode = str(payload.get("mode", "")).lower()

    if mode == "overlay":
        _compose_overlay(mgr, cell, src_panel, src_map)
    elif mode in _TILE_MODES:
        _compose_tile(mgr, cell, src_panel, src_map, mode)
    elif mode == "callout":
        _compose_callout(session, mgr, cell, src, src_panel, src_map)
    else:
        ipc.emit_error(f"repfig_compose: unknown mode {mode!r}.")
        return

    _rebuild_and_emit(mgr, cell)


def _compose_overlay(mgr, cell, src_panel, src_map) -> None:
    """Append the source's base layer to the TARGET panel as an overlay layer
    (distinct cmap, alpha 0.5)."""
    target_panel = cell.spec.panels[0] if cell.spec.panels else None
    if target_panel is None:
        return
    src_base = src_panel.layers[0]
    src_arr = src_map.get((src_panel.id, src_base.id))
    if src_arr is None:
        return
    n_over = len(target_panel.layers)   # base is [0]; overlays start at 1
    cmap = _OVERLAY_CMAP_CYCLE[(n_over - 1) % len(_OVERLAY_CMAP_CYCLE)]
    new_layer = LayerSpec(source=src_base.source, cmap=cmap, clim=src_base.clim,
                          alpha=0.5, visible=True, id=new_layer_id())
    target_panel.layers.append(new_layer)
    mgr.set_snapshot(cell.id, target_panel.id, new_layer.id, np.asarray(src_arr))


def _compose_tile(mgr, cell, src_panel, src_map, mode) -> None:
    """Grow the grid layout, placing the source as a NEW panel on the given side.

    single → 1×2 / 2×1 with the new panel left/right/up/down of the existing one;
    already-grid → insert a row/col on that side and place the new panel there.
    Existing panels' ``grid_pos`` are shifted as needed."""
    spec = cell.spec
    layout = dict(spec.layout or {"kind": "single"})
    if str(layout.get("kind")) != "grid":
        rows, cols = 1, 1
    else:
        rows = int(layout.get("rows", 1) or 1)
        cols = int(layout.get("cols", 1) or 1)

    horizontal = mode in ("tile-left", "tile-right")
    prepend = mode in ("tile-up", "tile-left")

    if horizontal:
        new_cols = cols + 1
        new_rows = max(rows, 1)
        if prepend:
            for p in spec.panels:
                p.grid_pos = [p.grid_pos[0], p.grid_pos[1] + 1]
            new_pos = [0, 0]
        else:
            new_pos = [0, cols]
        rows, cols = new_rows, new_cols
    else:
        new_rows = rows + 1
        new_cols = max(cols, 1)
        if prepend:
            for p in spec.panels:
                p.grid_pos = [p.grid_pos[0] + 1, p.grid_pos[1]]
            new_pos = [0, 0]
        else:
            new_pos = [rows, 0]
        rows, cols = new_rows, new_cols

    # Build the new panel from the source (a fresh id; copy its base layer).
    src_base = src_panel.layers[0]
    src_arr = src_map.get((src_panel.id, src_base.id))
    new_panel_id = _next_panel_id(spec)
    new_base = LayerSpec(source=src_base.source, cmap=src_base.cmap,
                         clim=src_base.clim, alpha=1.0, visible=True,
                         id=new_layer_id())
    new_panel = PanelSpec(id=new_panel_id, grid_pos=new_pos, kind=src_panel.kind,
                          layers=[new_base], axes=(dict(src_panel.axes)
                                                   if src_panel.axes else None),
                          title=src_panel.title,
                          scalebar=src_panel.scalebar)
    spec.panels.append(new_panel)
    if src_arr is not None:
        mgr.set_snapshot(cell.id, new_panel_id, new_base.id, np.asarray(src_arr))

    spec.layout = {"kind": "grid", "rows": rows, "cols": cols}


def _compose_callout(session, mgr, cell, src, src_panel, src_map) -> None:
    """Add a small callout INSET on the target's primary panel that references a NEW
    small panel spec rendered from the source snapshot. The connector region is the
    source's current nav-selector region on the navigator (when applicable)."""
    target_panel = cell.spec.panels[0] if cell.spec.panels else None
    if target_panel is None:
        return
    src_base = src_panel.layers[0]
    src_arr = src_map.get((src_panel.id, src_base.id))
    if src_arr is None:
        return
    inset_panel_id = _next_panel_id(cell.spec)
    inset_layer = LayerSpec(source=src_base.source, cmap=src_base.cmap,
                            clim=src_base.clim, alpha=1.0, visible=True,
                            id=new_layer_id())
    inset_panel = PanelSpec(id=inset_panel_id, grid_pos=[0, 0], kind=src_panel.kind,
                            layers=[inset_layer], title=src_panel.title)
    # The inset panel is NOT placed in the grid (it's an overlay); we keep its spec
    # in panels so its snapshot resolves + it round-trips, and reference it by id.
    cell.spec.panels.append(inset_panel)
    mgr.set_snapshot(cell.id, inset_panel_id, inset_layer.id, np.asarray(src_arr))

    region = _source_nav_region(session, src)
    inset_entry = {
        "panel": inset_panel_id,
        "corner": "top-right",
        "w_frac": 0.3,
        "h_frac": 0.3,
        "connector": ({"region": list(region)} if region is not None else None),
    }
    target_panel.insets.append(inset_entry)


def _next_panel_id(spec) -> str:
    """A fresh panel id (``p<N>``) not already used in the spec."""
    used = {p.id for p in spec.panels}
    n = len(spec.panels) + 1
    while f"p{n}" in used:
        n += 1
    return f"p{n}"


# ── layer / panel / annotation edits ──────────────────────────────────────────


def _find_panel(spec, panel_id):
    for p in spec.panels:
        if p.id == panel_id:
            return p
    return None


def _find_layer(panel, layer_id):
    for ly in panel.layers:
        if ly.id == layer_id:
            return ly
    return None


def repfig_set_layer(session, plot, payload) -> None:
    """Update one layer's appearance (cmap / alpha / clim / visible) in a panel."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    panel = _find_panel(cell.spec, payload.get("panel_id"))
    if panel is None:
        return
    layer = _find_layer(panel, payload.get("layer_id"))
    if layer is None:
        return
    if "cmap" in payload and payload["cmap"]:
        layer.cmap = str(payload["cmap"])
    if "alpha" in payload and payload["alpha"] is not None:
        try:
            layer.alpha = float(payload["alpha"])
        except (TypeError, ValueError):
            pass
    if "visible" in payload and payload["visible"] is not None:
        layer.visible = bool(payload["visible"])
    if "clim" in payload:
        clim = payload["clim"]
        if clim is None:
            layer.clim = None
        else:
            try:
                layer.clim = [float(clim[0]), float(clim[1])]
            except (TypeError, ValueError, IndexError):
                pass
    _rebuild_and_emit(mgr, cell)


def repfig_remove_layer(session, plot, payload) -> None:
    """Remove one layer from a panel. Removing the LAST layer removes the panel;
    removing the last panel empties the cell back to a placeholder."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    panel = _find_panel(cell.spec, payload.get("panel_id"))
    if panel is None:
        return
    layer = _find_layer(panel, payload.get("layer_id"))
    if layer is None:
        return
    panel.layers = [ly for ly in panel.layers if ly.id != layer.id]
    mgr._snapshots.get(cell.id, {}).pop((panel.id, layer.id), None)
    if not panel.layers:
        _remove_panel(mgr, cell, panel)
    _finalize_edit(mgr, cell)


def repfig_remove_panel(session, plot, payload) -> None:
    """Remove an entire panel from the cell (dropping its layers + snapshots)."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    panel = _find_panel(cell.spec, payload.get("panel_id"))
    if panel is None:
        return
    _remove_panel(mgr, cell, panel)
    _finalize_edit(mgr, cell)


def _remove_panel(mgr, cell, panel) -> None:
    """Drop a panel: its layers' snapshots, any insets that reference it (on any
    panel), and re-normalise the grid layout."""
    cell.spec.panels = [p for p in cell.spec.panels if p.id != panel.id]
    snap = mgr._snapshots.get(cell.id, {})
    for lyr in panel.layers:
        snap.pop((panel.id, lyr.id), None)
    # Drop any inset referencing the removed panel.
    for p in cell.spec.panels:
        p.insets = [ins for ins in (p.insets or [])
                    if ins.get("panel") != panel.id]
    _renormalise_layout(cell.spec)


def _renormalise_layout(spec) -> None:
    """Collapse the grid layout to fit the remaining GRID panels; ``single`` when ≤1
    grid panel remains. Inset-only panels (referenced by a callout) are floating and
    don't count toward the grid."""
    inset_ids = set()
    for p in spec.panels:
        for ins in (p.insets or []):
            if ins.get("panel"):
                inset_ids.add(ins["panel"])
    grid_panels = [p for p in spec.panels if p.id not in inset_ids]
    n = len(grid_panels)
    if n <= 1:
        spec.layout = {"kind": "single"}
        if n == 1:
            grid_panels[0].grid_pos = [0, 0]
        return
    rows = max(int(p.grid_pos[0]) for p in grid_panels) + 1
    cols = max(int(p.grid_pos[1]) for p in grid_panels) + 1
    spec.layout = {"kind": "grid", "rows": max(1, rows), "cols": max(1, cols)}


def _finalize_edit(mgr, cell) -> None:
    """Rebuild + emit after an edit; if the cell has NO panels left, empty it back
    to a placeholder (tear down the window, drop snapshots)."""
    if not cell.spec.panels:
        wid = mgr._window_by_cell.get(cell.id)
        if wid is not None:
            mgr._forget(wid)
        mgr._snapshots.pop(cell.id, None)
        cell.spec = None
        cell.placeholder = True
        mgr.dirty = True
        mgr.emit_state()
        return
    _rebuild_and_emit(mgr, cell)


def repfig_add_annotation(session, plot, payload) -> None:
    """Add an annotation (``{kind, ...anyplotlib-marker kwargs, data coords}``) to a
    panel."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    panel = _find_panel(cell.spec, payload.get("panel_id"))
    if panel is None:
        return
    ann = payload.get("annotation")
    if not isinstance(ann, dict) or not ann.get("kind"):
        return
    panel.annotations.append(dict(ann))
    _rebuild_and_emit(mgr, cell)


def repfig_update_annotation(session, plot, payload) -> None:
    """Replace the annotation at ``index`` on a panel."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    panel = _find_panel(cell.spec, payload.get("panel_id"))
    if panel is None:
        return
    ann = payload.get("annotation")
    idx = payload.get("index")
    if not isinstance(ann, dict) or idx is None:
        return
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return
    if 0 <= i < len(panel.annotations):
        panel.annotations[i] = dict(ann)
        _rebuild_and_emit(mgr, cell)


def repfig_remove_annotation(session, plot, payload) -> None:
    """Remove the annotation at ``index`` from a panel."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    panel = _find_panel(cell.spec, payload.get("panel_id"))
    if panel is None:
        return
    idx = payload.get("index")
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return
    if 0 <= i < len(panel.annotations):
        del panel.annotations[i]
        _rebuild_and_emit(mgr, cell)

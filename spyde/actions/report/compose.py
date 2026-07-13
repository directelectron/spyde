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


def _target_panel(cell, target_panel_id=None):
    """The panel to treat as the compose TARGET: the panel matching
    ``target_panel_id`` when given and found, else the primary (first) panel —
    preserves single-panel behaviour and is the safe default for an unknown id."""
    if cell.spec is None or not cell.spec.panels:
        return None
    if target_panel_id is not None:
        for p in cell.spec.panels:
            if p.id == target_panel_id:
                return p
    return cell.spec.panels[0]


def _target_base_shape(mgr, cell, target_panel_id=None):
    """The (H, W) of the target panel's BASE layer snapshot, or None."""
    panel = _target_panel(cell, target_panel_id)
    if panel is None or not panel.layers:
        return None
    arr = mgr.snapshot_map(cell.id).get((panel.id, panel.layers[0].id))
    if isinstance(arr, np.ndarray) and arr.ndim >= 2:
        return tuple(arr.shape[:2])
    return None


def _target_base_source(cell, target_panel_id=None):
    """The SignalRef of the target panel's base layer — the plot the panel was
    snapshotted from."""
    panel = _target_panel(cell, target_panel_id)
    return panel.layers[0].source if (panel is not None and panel.layers) else None


def _is_nav_signal_pair(session, cell, source_plot, target_panel_id=None):
    """True when the source plot and the target panel's base source are a
    NAVIGATOR ↔ SIGNAL pair of the SAME signal tree (so a callout makes sense —
    e.g. a navigator cell with a diffraction-pattern callout, or vice versa)."""
    if source_plot is None or cell.spec is None:
        return False
    ref = _target_base_source(cell, target_panel_id)
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


def _region_from_selector(sel):
    """The integrating ``(x, y, w, h)`` nav region of one selector, or None when
    it's a crosshair (non-integrating) or its indices can't be read."""
    if not getattr(sel, "is_integrating", False):
        return None
    try:
        idx = np.asarray(sel.get_selected_indices())
    except Exception as e:
        log.debug("reading selector indices failed: %s", e)
        return None
    if idx.ndim == 2 and idx.shape[0] >= 1 and idx.shape[1] >= 2:
        xs, ys = idx[:, 0], idx[:, 1]
        x0, y0 = int(xs.min()), int(ys.min())
        return (x0, y0, int(xs.max() - x0 + 1), int(ys.max() - y0 + 1))
    return None


def _selectors_for_source(mm, source_plot):
    """The nav selectors ACTUALLY linked to *source_plot* (NOT every selector on
    any navigator). A selector is linked when *source_plot* is one of its driven
    children (source is the signal plot the selector slices) OR when the source is
    the NAVIGATOR the selector lives on (its ``parent`` plot_window holds the
    source). Returns them in registration order."""
    nav_selectors = getattr(mm, "navigation_selectors", {}) or {}
    src_pw = getattr(source_plot, "plot_window", None)
    out = []
    for nav_pw, selectors in nav_selectors.items():
        for sel in selectors:
            children = getattr(sel, "children", {}) or {}
            drives_source = source_plot in children
            on_source_nav = src_pw is not None and (
                nav_pw is src_pw or getattr(sel, "parent", None) is src_pw)
            if drives_source or on_source_nav:
                out.append(sel)
    return out


def _source_nav_region(session, source_plot):
    """The nav-selector region ``(x, y, w, h)`` of the integrating selector that
    is actually linked to *source_plot* — used as the callout connector region.

    Only selectors driving/attached to *source_plot* are considered (NOT the first
    integrating selector across any navigator, which would return an unrelated
    ROI). None when *source_plot* has no linked integrating region selector or it
    can't be read."""
    try:
        mm = getattr(source_plot, "multiplot_manager", None)
        if mm is None:
            return None
        for sel in _selectors_for_source(mm, source_plot):
            region = _region_from_selector(sel)
            if region is not None:
                return region
    except Exception as e:
        log.debug("reading source nav region failed: %s", e)
    return None


def _rebuild_and_emit(mgr, cell) -> None:
    """Rebuild the cell's live figure window and emit the authoritative state."""
    mgr._offline.discard(cell.id)
    mgr.build_figure_window(cell)
    mgr.dirty = True
    mgr.emit_state()


def _emit_fig_markers_or_rebuild(mgr, cell) -> None:
    """Persist a FIGURE-LEVEL annotation change with the LEAST disruption: try an
    in-place ``fig.set_figure_markers`` on the live figure (targeted redraw, no
    iframe reload); if there's no live figure, fall back to a full rebuild. Either
    way, flag dirty + emit the authoritative state so the renderer mirror updates."""
    mgr.dirty = True
    if mgr.push_fig_markers(cell):
        mgr.emit_state()
    else:
        _rebuild_and_emit(mgr, cell)


# ── query: which compose modes are compatible for this drop ───────────────────


def repfig_query_compose(session, plot, payload) -> None:
    """Reply with the compose modes compatible for dropping ``source_window_id``
    onto figure cell ``cell_id``: overlay when the source frame shape matches the
    target panel's base; callout when they're a navigator↔signal pair of one tree;
    tiles ALWAYS. ``target_panel_id`` (optional) selects which panel is the
    compose target on a multi-panel cell; defaults to the primary (first) panel."""
    mgr = _manager(session)
    cell_id = payload.get("cell_id")
    source_window_id = payload.get("source_window_id")
    target_panel_id = payload.get("target_panel_id")
    cell = _cell(mgr, cell_id)
    src = _resolve_source_plot(session, source_window_id)

    same_shape = False
    nav_signal_pair = False
    options = list(_TILE_MODES)   # tiles always available

    if cell is not None and src is not None:
        base_shape = _target_base_shape(mgr, cell, target_panel_id)
        src_frame = getattr(src, "current_data", None)
        src_shape = (tuple(src_frame.shape[:2])
                     if isinstance(src_frame, np.ndarray) and src_frame.ndim >= 2
                     else None)
        if base_shape is not None and src_shape is not None and base_shape == src_shape:
            same_shape = True
            options.insert(0, "overlay")
        if _is_nav_signal_pair(session, cell, src, target_panel_id):
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
    NOW and mutates the cell's FigureSpec, then rebuilds + re-emits.

    ``target_panel_id`` (optional) is the panel the compose is relative to —
    for ``overlay`` the panel that gains the layer, for a ``tile-*`` mode the
    panel the new panel is placed next to. Defaults to the primary (first)
    panel / the legacy edge-of-grid placement when absent or unresolvable, so
    existing (whole-figure) drop behaviour is unchanged."""
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
    target_panel_id = payload.get("target_panel_id")

    if mode == "overlay":
        _compose_overlay(mgr, cell, src_panel, src_map, target_panel_id)
    elif mode in _TILE_MODES:
        _compose_tile(mgr, cell, src_panel, src_map, mode, target_panel_id)
    elif mode == "callout":
        _compose_callout(session, mgr, cell, src, src_panel, src_map, target_panel_id)
    else:
        ipc.emit_error(f"repfig_compose: unknown mode {mode!r}.")
        return

    _rebuild_and_emit(mgr, cell)


def _compose_overlay(mgr, cell, src_panel, src_map, target_panel_id=None) -> None:
    """Append the source's base layer to the TARGET panel as an overlay layer
    (distinct cmap, alpha 0.5).

    Refuses (no-op + ``emit_error``) when the source frame's shape doesn't match
    the target panel's existing base shape — the popover's query-time gate
    (``repfig_query_compose``) keeps the UI from OFFERING overlay in that case,
    but ``mode`` is caller-supplied, so the execute path re-checks independently
    (a stale popover click, a race, or a hand-built payload must not be able to
    smuggle a mismatched layer into the FigureSpec — anyplotlib layers require
    matching shapes; see ``Plot._set_array``'s shape-change layer-drop guard)."""
    target_panel = _target_panel(cell, target_panel_id)
    if target_panel is None:
        return
    src_base = src_panel.layers[0]
    src_arr = src_map.get((src_panel.id, src_base.id))
    if src_arr is None:
        return
    base_shape = _target_base_shape(mgr, cell, target_panel_id)
    src_shape = tuple(np.asarray(src_arr).shape[:2])
    if base_shape is not None and base_shape != src_shape:
        ipc.emit_error(
            f"repfig_compose: overlay needs matching image sizes: "
            f"{src_shape} vs {base_shape}.")
        return
    n_over = len(target_panel.layers)   # base is [0]; overlays start at 1
    cmap = _OVERLAY_CMAP_CYCLE[(n_over - 1) % len(_OVERLAY_CMAP_CYCLE)]
    new_layer = LayerSpec(source=src_base.source, cmap=cmap, clim=src_base.clim,
                          alpha=0.5, visible=True, id=new_layer_id())
    target_panel.layers.append(new_layer)
    mgr.set_snapshot(cell.id, target_panel.id, new_layer.id, np.asarray(src_arr))


_DIRECTION_DELTA = {
    "tile-up": (-1, 0), "tile-down": (1, 0),
    "tile-left": (0, -1), "tile-right": (0, 1),
}


def _grid_panels(spec):
    """The panels that occupy a GRID cell — i.e. every panel NOT referenced as a
    callout inset (an inset panel is a floating overlay, not a grid cell). Mirrors
    the inset-id scan in :func:`_renormalise_layout`."""
    inset_ids = set()
    for p in spec.panels:
        for ins in (p.insets or []):
            if ins.get("panel"):
                inset_ids.add(ins["panel"])
    return [p for p in spec.panels if p.id not in inset_ids]


def _resolve_tile_target(grid_panels, mode, target_panel_id):
    """The grid panel the tile should be placed relative to.

    Prefers the panel matching ``target_panel_id``; falls back to the legacy
    edge-of-grid default (preserves pre-targeting behaviour when no/unknown
    target is given): tile-right → max col (tie-break min row), tile-left → min
    col, tile-down → max row (tie-break min col), tile-up → min row (tie-break
    min col)."""
    if target_panel_id is not None:
        for p in grid_panels:
            if p.id == target_panel_id:
                return p
    if not grid_panels:
        return None
    if mode == "tile-right":
        return min(grid_panels, key=lambda p: (-int(p.grid_pos[1]), int(p.grid_pos[0])))
    if mode == "tile-left":
        return min(grid_panels, key=lambda p: (int(p.grid_pos[1]), int(p.grid_pos[0])))
    if mode == "tile-down":
        return min(grid_panels, key=lambda p: (-int(p.grid_pos[0]), int(p.grid_pos[1])))
    # tile-up
    return min(grid_panels, key=lambda p: (int(p.grid_pos[0]), int(p.grid_pos[1])))


def _compose_tile(mgr, cell, src_panel, src_map, mode, target_panel_id=None) -> None:
    """Place the source as a NEW panel in the grid, relative to a TARGET panel
    (2-D grid-aware; see module docstring / CLAUDE task notes for the full
    contract):

    * Resolve the target panel (``target_panel_id``, else the legacy
      edge-of-grid default for *mode*).
    * The neighbor cell = target.grid_pos + the direction delta for *mode*.
    * If that cell is within the current grid AND unoccupied → HOLE FILL (no
      grid growth, no shifting).
    * Otherwise INSERT a row/column at the neighbor position: every grid panel
      at/after the insertion index is shifted by one, and the new panel lands
      at the freed slot next to the target.

    ``spec.layout`` is recomputed as the bounding grid over all grid panels
    (including the new one) once placement is decided."""
    spec = cell.spec
    layout = dict(spec.layout or {"kind": "single"})
    grid_panels = _grid_panels(spec)
    if str(layout.get("kind")) != "grid":
        # Single (0 or 1 grid panel) — normalise every existing grid panel to
        # [0, 0] so the legacy/target resolution below sees a coherent 1x1 grid.
        rows, cols = 1, 1
        for p in grid_panels:
            p.grid_pos = [0, 0]
    else:
        rows = int(layout.get("rows", 1) or 1)
        cols = int(layout.get("cols", 1) or 1)

    target = _resolve_tile_target(grid_panels, mode, target_panel_id)
    dr, dc = _DIRECTION_DELTA[mode]

    if target is None:
        # No existing grid panel at all — place the new one at the origin.
        new_pos = [0, 0]
        rows, cols = 1, 1
    else:
        tr, tc = int(target.grid_pos[0]), int(target.grid_pos[1])
        nr, nc = tr + dr, tc + dc
        occupied = {(int(p.grid_pos[0]), int(p.grid_pos[1])) for p in grid_panels}
        if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in occupied:
            # Hole fill: the neighbor cell exists and is empty.
            new_pos = [nr, nc]
        else:
            horizontal = dc != 0
            if horizontal:
                insert_at = tc + 1 if dc > 0 else tc
                for p in grid_panels:
                    if int(p.grid_pos[1]) >= insert_at:
                        p.grid_pos = [p.grid_pos[0], p.grid_pos[1] + 1]
                new_pos = [tr, insert_at]
                cols += 1
            else:
                insert_at = tr + 1 if dr > 0 else tr
                for p in grid_panels:
                    if int(p.grid_pos[0]) >= insert_at:
                        p.grid_pos = [p.grid_pos[0] + 1, p.grid_pos[1]]
                new_pos = [insert_at, tc]
                rows += 1

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

    # Recompute the bounding grid over all grid panels (including the new one).
    all_grid = _grid_panels(spec)
    final_rows = max((int(p.grid_pos[0]) for p in all_grid), default=0) + 1
    final_cols = max((int(p.grid_pos[1]) for p in all_grid), default=0) + 1
    spec.layout = {"kind": "grid", "rows": max(rows, final_rows),
                   "cols": max(cols, final_cols)}


def _axis_offset_scale(axis_vals):
    """(offset, scale) for a calibrated 1-D axis array (offset = first sample,
    scale = uniform step). None when the array is missing / too short."""
    try:
        a = np.asarray(axis_vals, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 1 or a.size < 1:
        return None
    offset = float(a[0])
    scale = float(a[1] - a[0]) if a.size >= 2 else 1.0
    return offset, scale


def _index_region_to_data(panel, region):
    """Convert a nav-INDEX-space ``(x, y, w, h)`` region to the ``panel``'s DATA
    coords using the panel's calibrated x/y axes (offset + index*scale for the
    origin, w/h scaled). Falls back to the raw index region if the panel carries
    no usable axes (uncalibrated → index == data)."""
    x, y, w, h = region
    axes = panel.axes if panel is not None else None
    if not axes:
        return (float(x), float(y), float(w), float(h))
    xo_sc = _axis_offset_scale(axes.get("x_axis"))
    yo_sc = _axis_offset_scale(axes.get("y_axis"))
    if xo_sc is None or yo_sc is None:
        return (float(x), float(y), float(w), float(h))
    (ox, sx), (oy, sy) = xo_sc, yo_sc
    return (ox + x * sx, oy + y * sy, w * sx, h * sy)


def _compose_callout(session, mgr, cell, src, src_panel, src_map,
                     target_panel_id=None) -> None:
    """Add a small callout INSET on the target panel that references a NEW
    small panel spec rendered from the source snapshot.

    The connector (the dashed source-region rectangle drawn on the BASE panel) is
    attached ONLY when the base panel is the NAVIGATOR whose selector produced the
    region — i.e. the base shows the navigator and the callout inset shows the
    signal. In that case the nav-INDEX-space region is converted to the base
    panel's DATA coords (offset + index*scale). When the base panel is the SIGNAL
    (the navigator was dropped as the inset) there is no meaningful source region
    on the diffraction-pattern panel, so the connector is SKIPPED."""
    target_panel = _target_panel(cell, target_panel_id)
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

    # A connector is meaningful only when the BASE panel is the navigator (so the
    # region rectangle lands on nav axes) AND the inset (the dropped source) is the
    # signal driven by that navigator's selector. Determine the base plot and skip
    # the connector otherwise.
    connector = None
    base_ref = _target_base_source(cell, target_panel_id)
    base_plot = base_ref.resolve(session) if base_ref is not None else None
    base_is_nav = bool(getattr(base_plot, "is_navigator", False)) if base_plot else False
    src_is_nav = bool(getattr(src, "is_navigator", False))
    if base_is_nav and not src_is_nav:
        # The selector that produced the region lives on the BASE navigator; read
        # its region and convert index → the base panel's data coords.
        region = _source_nav_region(session, base_plot)
        if region is not None:
            data_region = _index_region_to_data(target_panel, region)
            connector = {"region": list(data_region)}

    inset_entry = {
        "panel": inset_panel_id,
        "corner": "top-right",
        "w_frac": 0.3,
        "h_frac": 0.3,
        "connector": connector,
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
        mgr._editing.discard(cell.id)
        mgr._edit_wiring.pop(cell.id, None)
        mgr._ann_widgets.pop(cell.id, None)
        mgr._selected.pop(cell.id, None)
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


# The widget kinds an in-place ``widget.set`` update supports (mirrors
# figure_builder._WIDGET_KINDS — ellipse/line have no draggable widget, so an
# edit to them always rebuilds).
_INPLACE_WIDGET_KINDS = {"text", "circle", "rect", "arrow"}


def _spec_ann_to_widget_fields(ann: dict, axes) -> "dict | None":
    """Map a PANEL annotation dict (DATA coords, spec schema) → the anyplotlib
    edit-WIDGET field schema (image PIXELS), the SAME forward mapping
    ``figure_builder._add_annotation_widget`` applies at build time. Returns the
    widget-attr dict to push via ``widget.set(...)`` (color/text/fontsize +
    geometry), or None if the kind has no widget / the geometry can't be read.

    Field mapping (spec → widget), post data→px conversion:
      * text   → ``x, y`` (offset), ``text`` (texts[0]), ``fontsize``, ``color``
      * circle → ``cx, cy`` (offset), ``r`` (radius), ``color`` (edgecolors)
      * rect   → ``x, y`` (offset - size/2 → TOP-LEFT), ``w, h``, ``color``
      * arrow  → ``x, y`` (tail offset), ``u, v`` (U/V), ``color`` (edgecolors)"""
    from spyde.actions.report import coords
    from spyde.actions.report.figure_builder import (
        _first_offset, _scalar0, _first_color,
    )

    kind = str(ann.get("kind", "")).lower()
    if kind not in _INPLACE_WIDGET_KINDS:
        return None
    conv = coords.annotation_data_to_pixel(ann, axes)
    pt = _first_offset(conv.get("offsets"))
    if pt is None:
        return None
    cx, cy = pt
    if kind == "text":
        texts = conv.get("texts")
        text = str(texts[0]) if isinstance(texts, (list, tuple)) and texts \
            else str(conv.get("text", "Label"))
        return {"x": cx, "y": cy, "text": text,
                "fontsize": int(conv.get("fontsize", 14) or 14),
                "color": _first_color(conv.get("color"), "#00e5ff")}
    if kind == "circle":
        r = _scalar0(conv.get("radius"))
        if r is None:
            return None
        return {"cx": cx, "cy": cy, "r": float(r),
                "color": _first_color(conv.get("edgecolors"), "#00e5ff")}
    if kind == "rect":
        w = _scalar0(conv.get("widths"))
        hh = _scalar0(conv.get("heights"))
        if w is None or hh is None:
            return None
        # spec offset is the CENTER; widget x/y is the TOP-LEFT.
        return {"x": cx - float(w) / 2.0, "y": cy - float(hh) / 2.0,
                "w": float(w), "h": float(hh),
                "color": _first_color(conv.get("edgecolors"), "#00e5ff")}
    if kind == "arrow":
        u = _scalar0(conv.get("U"))
        v = _scalar0(conv.get("V"))
        if u is None or v is None:
            return None
        return {"x": cx, "y": cy, "u": float(u), "v": float(v),
                "color": _first_color(conv.get("edgecolors"), "#00e5ff")}
    return None


def repfig_update_annotation(session, plot, payload) -> None:
    """Replace the annotation at ``index`` on a panel.

    Fast path (NO figure rebuild → no iframe flash): when the cell is in EDIT MODE,
    the KIND is unchanged (color/text/fontsize/geometry edit only), and a live edit
    widget exists for this annotation, push the changed fields onto that widget via
    ``widget.set(...)`` — a targeted JS merge + redraw. Otherwise (not editing /
    kind changed / widget missing) fall back to the full rebuild."""
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
    if not (0 <= i < len(panel.annotations)):
        return
    prev = panel.annotations[i]
    new_ann = dict(ann)
    panel.annotations[i] = new_ann

    # In-place update ONLY when editing, the kind is unchanged, and a live widget
    # exists + accepts the pushed fields. A kind change (or a non-widget kind, or a
    # missing widget) restructures the overlay → full rebuild.
    same_kind = str(prev.get("kind", "")) == str(new_ann.get("kind", ""))
    if cell.id in mgr._editing and same_kind:
        fields = _spec_ann_to_widget_fields(new_ann, panel.axes)
        if fields is not None and mgr.push_ann_widget(cell.id, panel.id, i, fields):
            # Widget updated live (no rebuild) — just persist + emit state.
            mgr.dirty = True
            mgr.emit_state()
            return
    # Fallback: not editing / kind changed / widget not found → rebuild.
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


def repfig_set_edit_mode(session, plot, payload) -> None:
    """Toggle EDIT MODE for a figure cell (``{cell_id, editing: bool}``). In edit
    mode the cell's annotations render as draggable widgets (drag → persist), out of
    it they render as static markers. On ANY change to the ``_editing`` membership
    the cell's figure is rebuilt (so it re-renders in the right mode) and the
    authoritative state re-emitted."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None:
        return
    editing = bool(payload.get("editing"))
    was = cell.id in mgr._editing
    if editing == was:
        return
    if editing:
        mgr._editing.add(cell.id)
    else:
        mgr._editing.discard(cell.id)
        # Leaving edit mode clears the selection (the dock unmounts); the outline
        # is gone anyway once edit_chrome is off on the rebuilt figure.
        mgr._selected.pop(cell.id, None)
    _rebuild_and_emit(mgr, cell)


# ── selection + figure-level layout / annotations ─────────────────────────────


def repfig_select_panel(session, plot, payload) -> None:
    """Select a panel (``{cell_id, panel_id|null}``) — the dock chips drive the
    same selection source of truth as a click on the live figure. ``panel_id`` null
    → figure-level (deselect). Delegates to ``ReportManager.select_panel`` (which
    records it, pushes the outline, and emits ``report_panel_selected``)."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None:
        return
    mgr.select_panel(cell.id, payload.get("panel_id"))


# Figure layout spacing is clamped to a sane fraction range (a whole-figure inter-
# panel gap of >1 mean cell is nonsensical; negatives collapse panels).
_LAYOUT_MIN, _LAYOUT_MAX = 0.0, 1.0


def _clamp_layout(val):
    try:
        return max(_LAYOUT_MIN, min(_LAYOUT_MAX, float(val)))
    except (TypeError, ValueError):
        return None


def repfig_set_layout(session, plot, payload) -> None:
    """Set the figure-level layout spacing (``{cell_id, hspace?, wspace?}``) — the
    whole-figure gap between grid panels. Only the provided keys are stored (clamped
    to 0..1); then the figure is rebuilt so ``subplots_adjust`` takes effect."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    layout = dict(cell.spec.layout or {"kind": "single"})
    changed = False
    for key in ("hspace", "wspace"):
        if key in payload and payload[key] is not None:
            v = _clamp_layout(payload[key])
            if v is not None:
                layout[key] = v
                changed = True
    if not changed:
        return
    cell.spec.layout = layout
    _rebuild_and_emit(mgr, cell)


_LAYOUT_PRESETS = ("row", "column", "grid")


def repfig_apply_layout_preset(session, plot, payload) -> None:
    """Reassign ``grid_pos`` for every GRID panel (inset-referenced panels are
    excluded — see :func:`_grid_panels`) to one of three presets, in the
    panels' CURRENT visual order (sorted by ``(row, col)`` first):

    * ``row``    — 1 × N (all panels in a single row)
    * ``column`` — N × 1 (all panels in a single column)
    * ``grid``   — 2 columns, ``ceil(N/2)`` rows, filled row-major

    ``{cell_id, preset}``. Updates ``spec.layout`` rows/cols (preserving
    ``hspace``/``wspace`` when present) and rebuilds + re-emits. An unknown
    preset (or a cell/spec that can't be resolved) emits an error and makes NO
    change."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        ipc.emit_error("repfig_apply_layout_preset: figure cell not found.")
        return
    preset = str(payload.get("preset", "")).lower()
    if preset not in _LAYOUT_PRESETS:
        ipc.emit_error(f"repfig_apply_layout_preset: unknown preset {preset!r}.")
        return

    grid_panels = _grid_panels(cell.spec)
    if not grid_panels:
        return
    ordered = sorted(grid_panels, key=lambda p: (int(p.grid_pos[0]), int(p.grid_pos[1])))
    n = len(ordered)

    if preset == "row":
        rows, cols = 1, n
        for i, p in enumerate(ordered):
            p.grid_pos = [0, i]
    elif preset == "column":
        rows, cols = n, 1
        for i, p in enumerate(ordered):
            p.grid_pos = [i, 0]
    else:   # grid — 2 columns, ceil(N/2) rows, row-major
        cols = 2
        rows = (n + cols - 1) // cols
        for i, p in enumerate(ordered):
            p.grid_pos = [i // cols, i % cols]

    layout = dict(cell.spec.layout or {"kind": "single"})
    layout["kind"] = "grid"
    layout["rows"] = rows
    layout["cols"] = cols
    # hspace/wspace (if the caller previously set them) are preserved as-is —
    # nothing above touches those keys.
    cell.spec.layout = layout
    _rebuild_and_emit(mgr, cell)


def _valid_fig_annotation(ann) -> bool:
    """A figure-level annotation is a dict whose ``kind`` is one anyplotlib's
    figure-marker layer accepts (text/circle/rect/arrow). Positions are FIGURE
    FRACTIONS — no data-coord conversion — so we only validate the kind here."""
    return isinstance(ann, dict) and str(ann.get("kind", "")) in _FIG_ANN_KINDS


_FIG_ANN_KINDS = {"text", "circle", "rect", "arrow"}


def repfig_add_fig_annotation(session, plot, payload) -> None:
    """Add a FIGURE-LEVEL annotation (``{cell_id, annotation}``) — a fraction-coord
    marker in the anyplotlib figure-marker schema. An ``id`` is assigned when
    missing so a later drag can persist back by id."""
    import uuid as _uuid
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    ann = payload.get("annotation")
    if not _valid_fig_annotation(ann):
        return
    ann = dict(ann)
    if not ann.get("id"):
        ann["id"] = _uuid.uuid4().hex[:8]
    cell.spec.annotations.append(ann)
    _emit_fig_markers_or_rebuild(mgr, cell)


def repfig_update_fig_annotation(session, plot, payload) -> None:
    """Replace the FIGURE-LEVEL annotation at ``index`` (``{cell_id, index,
    annotation}``). Preserves the existing id when the incoming dict omits one so
    drag-persistence keeps matching."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    ann = payload.get("annotation")
    idx = payload.get("index")
    if not _valid_fig_annotation(ann) or idx is None:
        return
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return
    anns = cell.spec.annotations
    if 0 <= i < len(anns):
        new = dict(ann)
        if not new.get("id"):
            new["id"] = anns[i].get("id")
        anns[i] = new
        _emit_fig_markers_or_rebuild(mgr, cell)


def repfig_remove_fig_annotation(session, plot, payload) -> None:
    """Remove the FIGURE-LEVEL annotation at ``index`` (``{cell_id, index}``)."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        return
    idx = payload.get("index")
    try:
        i = int(idx)
    except (TypeError, ValueError):
        return
    anns = cell.spec.annotations
    if 0 <= i < len(anns):
        del anns[i]
        _emit_fig_markers_or_rebuild(mgr, cell)

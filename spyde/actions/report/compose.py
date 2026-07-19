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
import re

import numpy as np

from spyde.backend import ipc
from spyde.actions.report.handlers import _manager, _resolve_source_plot, _snapshot_plot
from spyde.actions.report.model import (
    LayerSpec, PanelSpec, SignalRef, new_layer_id,
)

log = logging.getLogger(__name__)

# The default overlay-layer colormap cycle (distinct from a typical gray/viridis
# base) so a composed overlay reads as a separate image. Still assigned to every
# composed overlay — it's the stored REVERT value when the user clears a tint
# back to colormap display.
_OVERLAY_CMAP_CYCLE = ["magma", "cividis", "plasma", "inferno", "cool", "spring"]

# The default overlay TINT cycle: the renderer's preset palette minus
# white/black (a white/black clear→colour ramp is invisible over a gray base).
# A newly composed overlay renders as a clear→tint intensity ramp by default;
# legacy cells (tint None) keep colormap display unchanged.
_OVERLAY_TINT_CYCLE = ["#f38ba8", "#ff9800", "#f9e2af",
                       "#a6e3a1", "#89dceb", "#cba6f7"]

# A tint must be a #rgb / #rrggbb hex — the same shapes anyplotlib's tint LUT
# parses. Anything else is ignored at the handler (an invalid colour would
# raise ValueError inside add_layer at rebuild time and silently DROP the
# layer from the figure — the guarded add_layer logs + skips).
_TINT_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

_TILE_MODES = ("tile-up", "tile-down", "tile-left", "tile-right")

# repfig_set_text_size: valid PanelSpec.text_sizes keys, and the event-style
# target aliases the edit dock sends (x_ticks/y_ticks share one "ticks" size —
# anyplotlib's set_tick_label_size applies to both axes; colorbar_label is the
# dock's event name for the "colorbar" spec key).
_TEXT_SIZE_KEYS = {"title", "x_label", "y_label", "ticks", "legend", "colorbar"}
_TEXT_SIZE_TARGET_ALIASES = {
    "x_ticks": "ticks", "y_ticks": "ticks", "colorbar_label": "colorbar",
}
_TEXT_SIZE_MIN, _TEXT_SIZE_MAX = 6, 96


# ── shared helpers ────────────────────────────────────────────────────────────


def _cell(mgr, cell_id):
    if not mgr.open:
        return None
    cell = mgr.doc.cell_by_id(cell_id)
    if cell is None or cell.placeholder:
        return None
    # A plain figure cell OR a split cell whose figure side is a real figure
    # (has a spec) — both use the same FigureSpec/panel edit machinery.
    if cell.cell_type == "figure" or (
            cell.cell_type == "split" and cell.spec is not None):
        return cell
    return None


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

    # A scene3d target composes with nothing: overlay needs matching image
    # shapes, a callout needs nav slicing, and tiling next to a 3-D scene is
    # unsupported (repfig_compose refuses it too — the renderer's no-rich-
    # options fallback auto-fires tile-right). Reply with NO options.
    if cell is not None:
        tgt = _target_panel(cell, target_panel_id)
        if tgt is not None and str(tgt.kind) == "scene3d":
            ipc.emit({
                "type": "repfig_compose_options",
                "cell_id": cell_id,
                "source_window_id": source_window_id,
                "options": [],
                "detail": {"same_shape": False, "nav_signal_pair": False},
            })
            return

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
    # ``mode`` is caller-supplied (an edge drop / the renderer's tile-right
    # fallback bypasses the query), so the execute path must refuse a scene3d
    # target independently — no overlay/tile/callout onto a 3-D scene panel.
    tgt = _target_panel(cell, payload.get("target_panel_id"))
    if tgt is not None and str(tgt.kind) == "scene3d":
        ipc.emit_error("repfig_compose: a 3-D scene panel can't be combined "
                       "with another figure.")
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


def _next_overlay_tint(panel) -> str:
    """The first :data:`_OVERLAY_TINT_CYCLE` colour not already used by one of
    *panel*'s layers, so stacked overlays stay visually distinct; wraps by
    overlay count once every cycle colour is taken."""
    used = {str(getattr(ly, "tint", None) or "").lower() for ly in panel.layers}
    for hexc in _OVERLAY_TINT_CYCLE:
        if hexc not in used:
            return hexc
    return _OVERLAY_TINT_CYCLE[(len(panel.layers) - 1) % len(_OVERLAY_TINT_CYCLE)]


def _compose_overlay(mgr, cell, src_panel, src_map, target_panel_id=None) -> None:
    """Append the source's base layer to the TARGET panel as an overlay layer
    (clear→tint ramp from the tint cycle, alpha 0.5; the cmap cycle value is
    kept alongside as the revert-to-colormap value).

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
                          alpha=0.5, visible=True,
                          tint=_next_overlay_tint(target_panel),
                          id=new_layer_id())
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
    # Record WHERE the inset was sliced so the callout becomes refreshable
    # (a fresh-slice callout: refresh/marker-drag re-slice the dataset at this
    # position instead of re-snapshotting the live frame). Only when the inset
    # shows the SIGNAL (a navigator dropped as the inset has no nav position).
    # Convention: hyperspy ``axes_manager.indices`` is x-first ((ix, iy)) and
    # ``sig.inav[...]`` consumes the SAME order, so the list is stored as-is —
    # exactly what ``slicing.read_frame_at`` expects.
    if not src_is_nav:
        try:
            idx = [int(i) for i in
                   src.plot_state.current_signal.axes_manager.indices]
            if idx:
                inset_entry["nav_indices"] = idx
        except Exception as e:
            log.debug("callout nav_indices read failed: %s", e)
    target_panel.insets.append(inset_entry)


def _next_panel_id(spec) -> str:
    """A fresh panel id (``p<N>``) not already used in the spec."""
    used = {p.id for p in spec.panels}
    n = len(spec.panels) + 1
    while f"p{n}" in used:
        n += 1
    return f"p{n}"


# ── fresh-slice zoom-inset callouts (Phase 3) ─────────────────────────────────
#
# A fresh-slice callout inset carries WHERE it was sliced (``nav_indices``
# x-first, or ``time_index`` for a movie) so refresh / marker-drag re-slice the
# dataset at that position via ``slicing.read_frame_at`` — never a snapshot of
# whatever frame the live plot happens to show.

# Top-left anchors (figure fractions) spreading the t=0 / t=n//2 / t=n-1 time
# callouts left → center → right along the top edge.
_TIME_CALLOUT_ANCHORS = ([0.03, 0.03], [0.37, 0.03], [0.71, 0.03])


def _callout_connector_region(panel, ix, iy):
    """Connector dict for a callout marked at nav index ``(ix, iy)``: a
    1-nav-pixel rect centered on the point, in the BASE panel's DATA coords
    (same conversion the drop-time callout connector uses)."""
    region = _index_region_to_data(panel, (ix - 0.5, iy - 0.5, 1.0, 1.0))
    return {"region": list(region)}


def _resolve_panel_nav_source(session, panel):
    """``(src_plot, nav_shape)`` for a panel's layer-0 source: the resolved live
    plot (``SignalRef.resolve`` prefers the non-navigator plot of the tree, so
    even a navigator-snapshotted panel yields the plot whose ``current_signal``
    carries the full navigation space) and its x-first ``navigation_shape``.
    ``(None, ())`` when unresolvable / no nav axes."""
    if panel is None or not panel.layers or panel.layers[0].source is None:
        return None, ()
    src_plot = panel.layers[0].source.resolve(session)
    if src_plot is None:
        return None, ()
    try:
        am = src_plot.plot_state.current_signal.axes_manager
        nav_shape = tuple(int(n) for n in am.navigation_shape)
    except Exception as e:
        log.debug("callout nav source read failed: %s", e)
        return None, ()
    return src_plot, nav_shape


def _append_callout_inset(mgr, cell, target_panel, frame, entry) -> str:
    """Create the hidden inset panel (same source ref as the target's base
    layer, NOT placed in the grid), store *frame* as its snapshot, and append
    *entry* (completed with the new panel id) to the target's insets. Returns
    the new inset panel id."""
    base_layer = target_panel.layers[0]
    inset_panel_id = _next_panel_id(cell.spec)
    inset_layer = LayerSpec(source=base_layer.source, cmap=base_layer.cmap,
                            clim=None, alpha=1.0, visible=True,
                            id=new_layer_id())
    inset_panel = PanelSpec(id=inset_panel_id, grid_pos=[0, 0], kind="image",
                            layers=[inset_layer])
    cell.spec.panels.append(inset_panel)
    mgr.set_snapshot(cell.id, inset_panel_id, inset_layer.id, frame)
    entry = dict(entry)
    entry["panel"] = inset_panel_id
    target_panel.insets.append(entry)
    return inset_panel_id


def repfig_add_callout(session, plot, payload) -> None:
    """Add a FRESH-SLICE callout inset to a panel (``{cell_id, panel_id,
    nav_indices?}``): slice the panel's source signal at ``nav_indices``
    (default: the center of the navigation space) and show that frame as a
    floating inset. ``nav_indices`` is x-first (hyperspy ``inav`` order),
    clamped into range. When the base panel IS the navigator image (its
    snapshot spans the nav space) the inset also gets a connector rect around
    the marked point, and edit mode renders a draggable marker there."""
    from spyde.actions.report.slicing import read_frame_at

    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        ipc.emit_error("repfig_add_callout: figure cell not found.")
        return
    panel = _target_panel(cell, payload.get("panel_id"))
    if panel is None or not panel.layers:
        ipc.emit_error("repfig_add_callout: target panel not found.")
        return
    if str(panel.kind) == "line":
        ipc.emit_error("repfig_add_callout: callouts aren't supported on a "
                       "line panel.")
        return
    src_plot, nav_shape = _resolve_panel_nav_source(session, panel)
    if src_plot is None or not nav_shape:
        ipc.emit_error("repfig_add_callout: panel source has no live signal "
                       "with navigation axes.")
        return
    raw = payload.get("nav_indices")
    if raw is None:
        nav_indices = [int(n) // 2 for n in nav_shape]
    else:
        try:
            nav_indices = [int(i) for i in raw]
        except (TypeError, ValueError):
            ipc.emit_error("repfig_add_callout: bad nav_indices.")
            return
        if len(nav_indices) != len(nav_shape):
            ipc.emit_error("repfig_add_callout: nav_indices rank mismatch "
                           f"({len(nav_indices)} vs {len(nav_shape)} nav axes).")
            return
        nav_indices = [max(0, min(i, n - 1))
                       for i, n in zip(nav_indices, nav_shape)]
    frame = read_frame_at(src_plot, nav_indices)
    if frame is None:
        ipc.emit_error("repfig_add_callout: slicing the frame failed.")
        return

    # Connector (and, in edit mode, the draggable marker) only when the base
    # panel is the NAVIGATOR image — its pixels ARE the nav space, so the
    # marked point has a spatial anchor. Detected by shape: the base snapshot
    # spans (ny, nx) of a 2-D nav.
    connector = None
    if len(nav_shape) == 2:
        base_arr = mgr.snapshot_map(cell.id).get((panel.id, panel.layers[0].id))
        if isinstance(base_arr, np.ndarray) and base_arr.ndim >= 2 \
                and tuple(base_arr.shape[:2]) == (nav_shape[1], nav_shape[0]):
            connector = _callout_connector_region(
                panel, nav_indices[0], nav_indices[1])

    _append_callout_inset(mgr, cell, panel, frame, {
        "corner": "top-right",
        "w_frac": 0.3,
        "h_frac": 0.3,
        "connector": connector,
        "nav_indices": [int(i) for i in nav_indices],
    })
    _rebuild_and_emit(mgr, cell)


def repfig_add_time_callouts(session, plot, payload) -> None:
    """Add THREE fresh-slice callouts at t = 0, n//2, n-1 of a 1-D (time)
    navigation axis (``{cell_id, panel_id}``), anchored top-left / top-center /
    top-right. All frames are sliced BEFORE any spec mutation so a failed slice
    leaves the cell untouched. Duplicate t values (tiny movies) collapse."""
    from spyde.actions.report.slicing import read_frame_at

    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        ipc.emit_error("repfig_add_time_callouts: figure cell not found.")
        return
    panel = _target_panel(cell, payload.get("panel_id"))
    if panel is None or not panel.layers:
        ipc.emit_error("repfig_add_time_callouts: target panel not found.")
        return
    if str(panel.kind) == "line":
        ipc.emit_error("repfig_add_time_callouts: callouts aren't supported "
                       "on a line panel.")
        return
    src_plot, nav_shape = _resolve_panel_nav_source(session, panel)
    if src_plot is None or not nav_shape:
        ipc.emit_error("repfig_add_time_callouts: panel source has no live "
                       "signal with navigation axes.")
        return
    if len(nav_shape) != 1:
        ipc.emit_error("repfig_add_time_callouts: needs a 1-D (time) "
                       "navigation axis.")
        return
    n = nav_shape[0]
    ts = list(dict.fromkeys([0, n // 2, n - 1]))
    frames = []
    for t in ts:
        f = read_frame_at(src_plot, [t])
        if f is None:
            ipc.emit_error(f"repfig_add_time_callouts: slicing frame t={t} "
                           "failed.")
            return
        frames.append(f)
    for slot, (t, f) in enumerate(zip(ts, frames)):
        _append_callout_inset(mgr, cell, panel, f, {
            "anchor": list(_TIME_CALLOUT_ANCHORS[slot % len(_TIME_CALLOUT_ANCHORS)]),
            "w_frac": 0.26,
            "h_frac": 0.26,
            "connector": None,
            "time_index": int(t),
            "title": f"t={t}",
        })
    _rebuild_and_emit(mgr, cell)


# ── zoom-region callouts (a magnified crop of the BASE panel itself, no nav) ──
#
# Unlike a fresh-slice callout (which re-slices a DIFFERENT dataset position),
# a zoom callout crops the panel's OWN currently-held base snapshot — it never
# touches the dataset, so it works on a plain 2-D image with no navigation axes
# at all (a scene3d panel is still refused: there's no 2-D pixel grid to crop).


def _base_snapshot_hw(mgr, cell, panel):
    """The target panel's base-layer snapshot as ``(arr, H, W)``, or
    ``(None, 0, 0)`` when there's no usable 2-D/RGB base snapshot."""
    if panel is None or not panel.layers:
        return None, 0, 0
    arr = mgr.snapshot_map(cell.id).get((panel.id, panel.layers[0].id))
    arr = np.asarray(arr) if arr is not None else None
    if arr is None or arr.ndim < 2:
        return None, 0, 0
    h, w = arr.shape[0], arr.shape[1]
    return arr, h, w


def _crop_region_px(arr, ix0, iy0, w, h):
    """Crop *arr* (2-D or HxWxC) to the pixel-index rect ``[ix0, ix0+w) x
    [iy0, iy0+h)``, clamped to the array bounds. Returns a detached copy."""
    H, W = arr.shape[0], arr.shape[1]
    x0 = max(0, min(int(round(ix0)), W - 1))
    y0 = max(0, min(int(round(iy0)), H - 1))
    x1 = max(x0 + 1, min(int(round(ix0 + w)), W))
    y1 = max(y0 + 1, min(int(round(iy0 + h)), H))
    return np.array(arr[y0:y1, x0:x1], copy=True)


def repfig_add_zoom_callout(session, plot, payload) -> None:
    """Add a ZOOM-REGION callout to a panel (``{cell_id, panel_id,
    region?:[x,y,w,h] index-space}``): crop a rectangular region out of the
    panel's OWN base snapshot (never a dataset re-slice — works on a plain 2-D
    image with no navigation axes) and show the crop as a floating inset,
    magnified. Default region (when ``region`` is omitted): centered, W/4 x
    H/4 pixel indices. The inset carries ``zoom_region`` (the region in DATA
    coords, the write-back key for a later drag-resize) and a ``connector``
    pointing at the same region on the base panel.

    Refused (no-op + ``emit_error``) for a scene3d panel (no 2-D pixel grid to
    crop), a line panel (no 2-D pixel grid either — a curve has no region to
    zoom into), or when the target panel has no usable base snapshot yet."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        ipc.emit_error("repfig_add_zoom_callout: figure cell not found.")
        return
    panel = _target_panel(cell, payload.get("panel_id"))
    if panel is None or not panel.layers:
        ipc.emit_error("repfig_add_zoom_callout: target panel not found.")
        return
    if str(panel.kind) == "scene3d":
        ipc.emit_error("repfig_add_zoom_callout: a 3-D scene panel has no "
                       "2-D region to zoom into.")
        return
    if str(panel.kind) == "line":
        ipc.emit_error("repfig_add_zoom_callout: a line panel has no 2-D "
                       "region to zoom into.")
        return
    base_arr, H, W = _base_snapshot_hw(mgr, cell, panel)
    if base_arr is None or H < 1 or W < 1:
        ipc.emit_error("repfig_add_zoom_callout: panel has no image to zoom "
                       "into yet.")
        return

    raw_region = payload.get("region")
    if raw_region is not None:
        try:
            ix0, iy0, iw, ih = (float(v) for v in raw_region)
        except (TypeError, ValueError):
            ipc.emit_error("repfig_add_zoom_callout: bad region.")
            return
    else:
        iw, ih = max(1.0, W / 4.0), max(1.0, H / 4.0)
        ix0, iy0 = (W - iw) / 2.0, (H - ih) / 2.0
    iw = max(1.0, min(iw, W))
    ih = max(1.0, min(ih, H))
    ix0 = max(0.0, min(ix0, W - iw))
    iy0 = max(0.0, min(iy0, H - ih))

    crop = _crop_region_px(base_arr, ix0, iy0, iw, ih)
    data_region = _index_region_to_data(panel, (ix0, iy0, iw, ih))

    _append_callout_inset(mgr, cell, panel, crop, {
        "corner": "bottom-right",
        "w_frac": 0.3,
        "h_frac": 0.3,
        "connector": {"region": list(data_region)},
        "zoom_region": list(data_region),
    })
    _rebuild_and_emit(mgr, cell)


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


def repfig_set_text_size(session, plot, payload) -> None:
    """Set one text element's font size (title/x_label/y_label/ticks/legend/
    colorbar) on a panel, persist it to ``PanelSpec.text_sizes``, and push it
    to the LIVE figure in place (no rebuild / iframe flash) when the cell has a
    mounted window — via ``figure_builder._apply_text_sizes`` on the live plot
    object for the panel's anyplotlib dispatch id. Falls back to a full rebuild
    when there's no live figure (or the in-place push raises).

    ``{cell_id, panel_id, target, size}`` — ``target`` accepts both a spec key
    (``title``/``x_label``/``y_label``/``ticks``/``legend``/``colorbar``) and
    the edit dock's event-style names (``x_ticks``/``y_ticks`` → ``ticks``,
    ``colorbar_label`` → ``colorbar``). ``panel_id`` may be either a spec panel
    id or an anyplotlib dispatch panel id — both are resolved to the spec
    panel. ``size`` is clamped to ``[6, 96]`` (int). An unknown target or a
    non-numeric size emits an error and makes NO change."""
    mgr = _manager(session)
    cell = _cell(mgr, payload.get("cell_id"))
    if cell is None or cell.spec is None:
        ipc.emit_error("repfig_set_text_size: figure cell not found.")
        return

    raw_target = str(payload.get("target", "") or "")
    key = _TEXT_SIZE_TARGET_ALIASES.get(raw_target, raw_target)
    if key not in _TEXT_SIZE_KEYS:
        ipc.emit_error(f"repfig_set_text_size: unknown target {raw_target!r}.")
        return

    try:
        size = int(round(float(payload.get("size"))))
    except (TypeError, ValueError):
        ipc.emit_error("repfig_set_text_size: size must be numeric.")
        return
    size = max(_TEXT_SIZE_MIN, min(_TEXT_SIZE_MAX, size))

    # panel_id may be either a spec panel id OR an anyplotlib dispatch id (the
    # fig._report_panel_map key vs value) — resolve to the spec PanelSpec and
    # remember the dispatch id for the live push below.
    raw_panel_id = payload.get("panel_id")
    panel = _find_panel(cell.spec, raw_panel_id)
    disp_id = None
    fig = mgr.live_fig(cell.id)
    if panel is None and fig is not None:
        panel_map = dict(getattr(fig, "_report_panel_map", None) or {})
        spec_by_dispatch = {disp: spec for spec, disp in panel_map.items()}
        spec_pid = spec_by_dispatch.get(raw_panel_id)
        if spec_pid is not None:
            panel = _find_panel(cell.spec, spec_pid)
            disp_id = raw_panel_id
    if panel is None:
        ipc.emit_error("repfig_set_text_size: panel not found.")
        return
    if disp_id is None and fig is not None:
        panel_map = dict(getattr(fig, "_report_panel_map", None) or {})
        disp_id = panel_map.get(panel.id)

    panel.text_sizes = {**(panel.text_sizes or {}), key: size}

    pushed = False
    if fig is not None and disp_id is not None:
        plots_map = getattr(fig, "_plots_map", None) or {}
        live_plot = plots_map.get(disp_id)
        if live_plot is not None:
            try:
                from spyde.actions.report.figure_builder import _apply_text_sizes
                _apply_text_sizes(live_plot, {key: size})
                pushed = True
            except Exception as e:
                log.debug("repfig_set_text_size in-place push failed "
                          "(cell %s panel %s): %s", cell.id, panel.id, e)
                pushed = False

    mgr.dirty = True
    if pushed:
        mgr.emit_state()
    else:
        _rebuild_and_emit(mgr, cell)


# repfig_set_layer's LINE-panel styling clamps/caps.
_LINEWIDTH_MIN, _LINEWIDTH_MAX = 0.5, 12.0
_LABEL_MAX_LEN = 120


def repfig_set_layer(session, plot, payload) -> None:
    """Update one layer's appearance (cmap / alpha / clim / visible / tint /
    color / linewidth / label) in a panel.

    ``tint`` (key present): a ``#rgb``/``#rrggbb`` string switches the layer to
    the clear→tint intensity ramp; ``null``/``""`` clears it back to colormap
    display. The stored ``cmap`` is NEVER dropped by a tint change — it's the
    revert value the clear falls back to. (anyplotlib's ``Layer.set`` rejects
    cmap+tint together, but this handler persists to the SPEC and rebuilds the
    figure — ``add_layer(cmap=..., tint=...)`` accepts both, keeping the cmap
    as the revert value — so no live ``set`` call ever carries both.)

    ``color`` / ``linewidth`` / ``label`` are LINE-PANEL curve styling
    (unused on an image layer, but harmless to set — they simply ride along
    unread until/unless the layer's panel becomes a line panel). ``color`` is
    a pass-through string (no validation — any CSS colour anyplotlib accepts).
    ``linewidth`` is clamped to ``[0.5, 12]``. ``label`` is capped to
    ``_LABEL_MAX_LEN`` chars; an explicit empty string CLEARS it to ``None``
    (removing the legend entry), while an absent key leaves it unchanged."""
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
    if "tint" in payload:
        tint = payload["tint"]
        if tint is None or tint == "":
            layer.tint = None              # back to cmap display (cmap kept)
        elif isinstance(tint, str) and _TINT_RE.match(tint):
            layer.tint = tint
        # A malformed tint string is ignored (same tolerance as the fields
        # below) — see _TINT_RE for why it must never reach the spec.
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
    if "color" in payload:
        color = payload["color"]
        layer.color = str(color) if color else None
    if "linewidth" in payload and payload["linewidth"] is not None:
        try:
            lw = float(payload["linewidth"])
            layer.linewidth = max(_LINEWIDTH_MIN, min(_LINEWIDTH_MAX, lw))
        except (TypeError, ValueError):
            pass
    if "label" in payload:
        label = payload["label"]
        if label is None:
            pass                            # absent/None = leave unchanged
        elif label == "":
            layer.label = None              # explicit empty = clear
        else:
            layer.label = str(label)[:_LABEL_MAX_LEN]
    # Layer appearance changes reach the live figure via the REBUILD path (the
    # same route cmap/alpha take today); tint/color/linewidth/label ride
    # identically.
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
    # 2-D marker geometry has no meaning on a 3-D scene panel OR a 1-D line
    # panel (the renderer hides the add buttons; this guards a stale/hand-built
    # payload).
    if str(panel.kind) == "scene3d":
        ipc.emit_error("repfig_add_annotation: annotations aren't supported "
                       "on a 3-D scene panel.")
        return
    if str(panel.kind) == "line":
        ipc.emit_error("repfig_add_annotation: annotations aren't supported "
                       "on a line panel.")
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
      * circle → ``cx, cy`` (offset), ``r`` (radius), ``color`` (edgecolors),
                 ``linewidth`` (when the annotation carries a scalar-able
                 ``linewidths``)
      * rect   → ``x, y`` (offset - size/2 → TOP-LEFT), ``w, h``, ``color``,
                 ``linewidth`` (same as circle)
      * arrow  → ``x, y`` (tail offset), ``u, v`` (U/V), ``color`` (edgecolors),
                 ``linewidth`` (defaults to 2.0 when unset)"""
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
        out = {"cx": cx, "cy": cy, "r": float(r),
               "color": _first_color(conv.get("edgecolors"), "#00e5ff")}
        lw = _scalar0(conv.get("linewidths"))
        if lw is not None:
            out["linewidth"] = float(lw)
        return out
    if kind == "rect":
        w = _scalar0(conv.get("widths"))
        hh = _scalar0(conv.get("heights"))
        if w is None or hh is None:
            return None
        # spec offset is the CENTER; widget x/y is the TOP-LEFT.
        out = {"x": cx - float(w) / 2.0, "y": cy - float(hh) / 2.0,
               "w": float(w), "h": float(hh),
               "color": _first_color(conv.get("edgecolors"), "#00e5ff")}
        lw = _scalar0(conv.get("linewidths"))
        if lw is not None:
            out["linewidth"] = float(lw)
        return out
    if kind == "arrow":
        u = _scalar0(conv.get("U"))
        v = _scalar0(conv.get("V"))
        if u is None or v is None:
            return None
        lw = _scalar0(conv.get("linewidths"))
        return {"x": cx, "y": cy, "u": float(u), "v": float(v),
                "color": _first_color(conv.get("edgecolors"), "#00e5ff"),
                "linewidth": float(lw) if lw else 2.0}
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

"""
handlers.py — the Report Builder staged action handlers.

All handlers share the uniform ``fn(session, plot, payload)`` signature and are
registered in :data:`spyde.actions.registry.STAGED_HANDLERS`. Every handler
tolerates ``plot=None`` (the report sidebar isn't tied to a signal window). The
report document lives at ``session._report`` (a :class:`ReportManager`, created
lazily by :func:`_manager`); after every MUTATING handler we emit the
authoritative ``report_state`` so the renderer mirror stays in sync.

Message contracts (the renderer is written against these EXACT shapes):

* ``report_state`` — the authoritative document (see :meth:`ReportManager.state`).
* cell figures — emitted through the normal bare-figure path
  (``finalize_figure_html``) with EXTRA top-level ``host:"report"`` + ``cell_id``.
* ``report_need_snapshots`` — the save handshake (renderer harvests PNGs via 0f).
* ``report_saved`` — save success.
* errors via ``emit_error``.
"""
from __future__ import annotations

import base64
import logging

import numpy as np

from spyde.backend import ipc
from spyde.actions.figure_registry import keep_alive
from spyde.actions.report.figure_builder import (
    ReportFigureController, build_cell_figure,
)
from spyde.actions.report.model import (
    Cell, FigureSpec, LayerSpec, PanelSpec, ReportDoc, SignalRef,
    bake_fallback_png, new_cell_id, read_report, write_report,
)

log = logging.getLogger(__name__)

# Fallback-bake if the renderer's harvested PNGs don't arrive within this window.
_SNAPSHOT_TIMEOUT_S = 3.0
# Cap the inline offline-fallback PNG (data URL in report_state) so a long
# offline report doesn't balloon the state message.
_OFFLINE_PNG_MAX_EDGE = 640


# ── the per-session report manager ────────────────────────────────────────────


class ReportManager:
    """Backend owner of the live report document. Holds the :class:`ReportDoc`,
    the in-memory figure snapshots (numpy arrays — NEVER written to the file), the
    open figure windows (window_id ↔ cell), the last save path, and the dirty
    flag."""

    def __init__(self, session):
        self.session = session
        self.doc: "ReportDoc | None" = None
        self.path: str | None = None
        self.dirty = False
        # cell_id -> { (panel_id, layer_id) : ndarray } in-memory snapshots (baked to
        # PNG only on save; NEVER written to the container's spec). One entry per
        # panel-layer so a composed multi-panel / multi-layer figure holds every
        # layer's pixels. Use the accessors (primary_snapshot / snapshot_map) below.
        self._snapshots: dict[str, dict] = {}
        # cell_id -> baked PNG bytes read from an opened report (offline fallback)
        self._baked: dict[str, bytes] = {}
        # window_id -> ReportFigureController, and cell_id -> window_id
        self._controllers: dict[int, ReportFigureController] = {}
        self._window_by_cell: dict[str, int] = {}
        # cell ids currently in EDIT MODE (annotations rendered as draggable
        # widgets). Membership drives ``build_figure_window``'s ``interactive`` flag.
        self._editing: set[str] = set()
        # cell_id -> the live annotation drag-persist wiring list from the last
        # interactive build (kept so the widget handlers aren't GC'd); replaced
        # wholesale on each rebuild, dropped when the cell is removed / report closed.
        self._edit_wiring: dict[str, list] = {}
        # cell_id -> { (panel_id, ann_index) : widget } for the current interactive
        # build — lets a live color/text/geometry edit push widget.set(...) onto the
        # exact widget instead of rebuilding the figure (edit-mode in-place update).
        # Rebuilt alongside _edit_wiring; empty for non-interactive builds.
        self._ann_widgets: dict[str, dict] = {}
        # cell_id -> the currently SELECTED spec panel id (or None = figure-level
        # selection). The single source of truth the edit dock mirrors; cleared
        # alongside _editing (new / close / remove-cell / edit-off).
        self._selected: dict[str, "str | None"] = {}
        # cell_id -> True while a figure's rebuild is pending
        self._offline: set[str] = set()
        # pending save handshake: token -> {cells, path, remaining}
        self._pending_save: dict[str, dict] = {}

    # ── per-(cell, panel, layer) snapshot accessors ─────────────────────────────

    @staticmethod
    def _layer_key(panel_id: str, layer_id: str) -> tuple:
        return (str(panel_id), str(layer_id))

    def set_snapshot(self, cell_id: str, panel_id: str, layer_id: str, arr) -> None:
        self._snapshots.setdefault(cell_id, {})[self._layer_key(panel_id, layer_id)] = arr

    def snapshot_map(self, cell_id: str) -> dict:
        """The ``{(panel_id, layer_id): ndarray}`` snapshot map for a cell (may be
        empty). The builder consumes this to paint each panel-layer."""
        return self._snapshots.get(cell_id, {})

    def primary_snapshot(self, cell_id: str):
        """The FIRST panel's FIRST layer snapshot for a cell — the offline-bake +
        legacy single-image path. None if the cell has no snapshots."""
        cell = self.doc.cell_by_id(cell_id) if self.doc else None
        if cell is not None and cell.spec is not None:
            for panel in cell.spec.panels:
                for layer in panel.layers:
                    arr = self._snapshots.get(cell_id, {}).get(
                        self._layer_key(panel.id, layer.id))
                    if arr is not None:
                        return arr
        # Fallback: any array in the map.
        m = self._snapshots.get(cell_id, {})
        for arr in m.values():
            return arr
        return None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    @property
    def open(self) -> bool:
        return self.doc is not None

    def new(self, template: bool = False) -> None:
        self.close_windows()
        self.doc = ReportDoc(template=template)
        self.path = None
        self.dirty = False
        self._snapshots.clear()
        self._baked.clear()
        self._offline.clear()
        self._editing.clear()
        self._edit_wiring.clear()
        self._ann_widgets.clear()
        self._selected.clear()

    def close_windows(self) -> None:
        """Tear down every open figure window (through the session's forget path
        so controllers + kept-alive figures are evicted)."""
        for wid in list(self._controllers):
            forget = getattr(self.session, "_forget_window", None)
            if forget is not None:
                try:
                    forget(int(wid))
                except Exception as e:
                    log.debug("report close_windows forget failed: %s", e)
            else:
                self._controllers.pop(wid, None)
        self._controllers.clear()
        self._window_by_cell.clear()

    def close(self) -> None:
        self.close_windows()
        self.doc = None
        self.path = None
        self.dirty = False
        self._snapshots.clear()
        self._baked.clear()
        self._offline.clear()
        self._pending_save.clear()
        self._editing.clear()
        self._edit_wiring.clear()
        self._ann_widgets.clear()
        self._selected.clear()

    # ── state emission ─────────────────────────────────────────────────────────

    def state(self) -> dict:
        """The authoritative ``report_state`` message body."""
        if self.doc is None:
            return {"open": False, "path": None, "title": "", "template": False,
                    "dirty": False, "cells": []}
        cells = []
        for c in self.doc.cells:
            entry = {
                "id": c.id,
                "cell_type": c.cell_type,
                "source": c.source if c.cell_type == "markdown" else None,
                "caption": c.caption if c.cell_type == "figure" else None,
                "placeholder": bool(c.placeholder),
                "fig_id": c.id if c.cell_type == "figure" else None,
                "data_offline": bool(c.cell_type == "figure" and c.id in self._offline),
            }
            # For a figure cell, ship the PIXEL-FREE FigureSpec recipe (as a plain
            # dict) so the renderer's edit toolbar can list panels / layers /
            # annotations. It carries NO image bytes — only the YAML-shaped structure.
            if c.cell_type == "figure" and c.spec is not None:
                entry["figure"] = c.spec.to_dict()
            # For an OFFLINE figure cell only, ship the baked PNG as a data URL so
            # the renderer (which has no zip access) can still show the snapshot.
            if entry["data_offline"]:
                png = self._offline_png(c.id)
                if png is not None:
                    entry["png"] = "data:image/png;base64," + \
                        base64.b64encode(png).decode("ascii")
            cells.append(entry)
        return {
            "open": True,
            "path": self.path,
            "title": self.doc.title,
            "template": bool(self.doc.template),
            "dirty": bool(self.dirty),
            "cells": cells,
        }

    def emit_state(self) -> None:
        ipc.emit({"type": "report_state", "report": self.state()})

    def _offline_png(self, cell_id: str) -> "bytes | None":
        """A size-capped PNG for an offline cell: the baked PNG from the opened
        report, else a bake of the held snapshot."""
        png = self._baked.get(cell_id)
        if png is not None:
            return png
        arr = self.primary_snapshot(cell_id)
        if arr is not None:
            try:
                cell = self.doc.cell_by_id(cell_id) if self.doc else None
                layer = cell.spec.primary_layer if (cell and cell.spec) else None
                cmap = layer.cmap if layer else "viridis"
                clim = layer.clim if layer else None
                return bake_fallback_png(arr, cmap=cmap, clim=clim,
                                         max_edge=_OFFLINE_PNG_MAX_EDGE)
            except Exception as e:
                log.debug("offline png bake failed: %s", e)
        return None

    # ── figure cell build / rebuild ────────────────────────────────────────────

    def build_figure_window(self, cell: Cell) -> None:
        """Build (or rebuild) the live figure window for a figure cell and emit
        it through the bare-figure path with ``host:"report"`` + ``cell_id``."""
        snap_map = self.snapshot_map(cell.id)
        if not snap_map or cell.spec is None:
            # Nothing to build — but STILL tear down any prior window/controller for
            # this cell (a refresh that lost its snapshot, or a spec-cleared cell),
            # so a stale live figure is never left mapped to a now-figure-less cell.
            # _forget clears both _controllers and _window_by_cell for the cell.
            prev_wid = self._window_by_cell.get(cell.id)
            if prev_wid is not None:
                self._forget(prev_wid)
                self._window_by_cell.pop(cell.id, None)
            self._edit_wiring.pop(cell.id, None)
            self._ann_widgets.pop(cell.id, None)
            return
        # Tear down any prior window for this cell first (re-snapshot / refresh).
        prev_wid = self._window_by_cell.get(cell.id)
        if prev_wid is not None:
            self._forget(prev_wid)
        interactive = cell.id in self._editing
        try:
            fig, fig_id, html = build_cell_figure(cell.spec, snap_map,
                                                  interactive=interactive)
        except Exception as e:
            log.exception("report figure build failed for cell %s: %s", cell.id, e)
            ipc.emit_error(f"Report figure build failed: {e}")
            return
        # Attach a pointer_up drag-persist handler to each annotation widget and
        # keep the wiring alive on the manager (replaced wholesale each rebuild, so
        # a stale build's handlers/widgets are dropped). Non-interactive → empty.
        wiring = list(getattr(fig, "_report_annotation_wiring", None) or [])
        self._wire_annotation_drag(cell.id, wiring)
        # In edit mode, also wire the SELECTION handlers: a genuine click on a
        # panel selects it; a figure-background click deselects (→ figure-level);
        # a figure-marker drag persists into spec.annotations. Appended to the same
        # _edit_wiring keep-alive list so nothing is GC'd (dropped on next rebuild).
        if interactive:
            self._wire_selection_handlers(cell.id, fig)
        wid = self.session.next_window_id()
        keep_alive(int(wid), fig)
        ctrl = ReportFigureController(self.session, self, cell.id, wid, fig=fig)
        reg = getattr(self.session, "register_window_controller", None)
        if reg is not None:
            reg(int(wid), ctrl)
        self._controllers[int(wid)] = ctrl
        self._window_by_cell[cell.id] = int(wid)
        self._offline.discard(cell.id)
        ipc.emit({
            "type": "figure", "fig_id": fig_id, "window_id": int(wid),
            "html": html, "title": cell.caption or "Figure",
            "is_navigator": False,
            "host": "report", "cell_id": cell.id,
        })

    def _wire_annotation_drag(self, cell_id: str, wiring: list) -> None:
        """Attach a ``pointer_up`` drag-persist handler to each annotation widget in
        *wiring* and store the (widget, handler) list on the manager so nothing is
        garbage-collected. Replaces this cell's prior wiring wholesale (a rebuild's
        widgets supersede the old ones). ``wiring`` empty (non-interactive) → the
        cell's wiring is dropped."""
        stored = []
        ann_widgets: dict = {}
        for (widget, panel_id, ann_index, panel_spec) in wiring:
            kind = str((panel_spec.annotations[ann_index] or {}).get("kind", "")) \
                if (0 <= ann_index < len(panel_spec.annotations)) else ""
            handler = _make_annotation_drag_handler(
                self, cell_id, panel_spec, ann_index, kind)
            try:
                widget.add_event_handler(handler, "pointer_up")
            except Exception as e:
                log.debug("wiring annotation drag handler failed: %s", e)
                continue
            stored.append((widget, handler))
            # Key by (SPEC panel id, ann_index) so a live edit finds the widget.
            ann_widgets[(panel_spec.id, ann_index)] = widget
        if stored:
            self._edit_wiring[cell_id] = stored
            self._ann_widgets[cell_id] = ann_widgets
        else:
            self._edit_wiring.pop(cell_id, None)
            self._ann_widgets.pop(cell_id, None)

    def _wire_selection_handlers(self, cell_id: str, fig) -> None:
        """Wire the EDIT-MODE selection handlers onto *fig* (interactive builds
        only). Registers, per panel base plot, a module-level-closure
        ``pointer_down`` handler → ``select_panel(cell_id, spec_panel_id)``; and a
        FIGURE-level ``pointer_down``/``pointer_up`` handler that routes a
        ``figure_background`` click → deselect (figure-level) and a
        ``figure_marker`` drop → persist the moved marker into ``spec.annotations``.

        All handlers are appended to this cell's ``_edit_wiring`` keep-alive list
        (this runs AFTER ``_wire_annotation_drag`` set it) so none are GC'd; the
        whole list is replaced on the next rebuild. After wiring, re-applies the
        cell's CURRENT selection so the persistent outline survives a rebuild."""
        stored = self._edit_wiring.get(cell_id) or []
        panel_map = dict(getattr(fig, "_report_panel_map", None) or {})
        plots_map = getattr(fig, "_plots_map", None) or {}
        # spec-panel id keyed by anyplotlib plot dispatch id (inverse of panel_map).
        spec_by_dispatch = {disp: spec for spec, disp in panel_map.items()}

        # Per panel: a pointer_down handler that selects the SPEC panel id.
        for spec_pid, disp_id in panel_map.items():
            plot = plots_map.get(disp_id)
            if plot is None:
                continue
            handler = _make_panel_select_handler(self, cell_id, spec_pid)
            try:
                plot.add_event_handler(handler, "pointer_down")
            except Exception as e:
                log.debug("wiring panel-select handler failed: %s", e)
                continue
            stored.append((plot, handler))

        # One figure-level handler for background-deselect + figure-marker persist.
        fig_handler = _make_figure_edit_handler(self, cell_id)
        try:
            fig.add_event_handler(fig_handler, "pointer_down", "pointer_up")
            # Keep a reference alive; the figure itself outlives the wiring, but
            # the handler closure must not be GC'd.
            stored.append((fig, fig_handler))
        except Exception as e:
            log.debug("wiring figure-level edit handler failed: %s", e)

        self._edit_wiring[cell_id] = stored
        # Re-apply the current selection so the outline persists across rebuilds.
        self._apply_selected_panel(cell_id, fig)

    def _apply_selected_panel(self, cell_id: str, fig) -> None:
        """Push ``fig.selected_panel`` to match this cell's recorded selection: the
        anyplotlib dispatch id for the selected SPEC panel, or "" for figure-level
        (None). No-op if the figure can't accept the trait."""
        spec_pid = self._selected.get(cell_id)
        panel_map = dict(getattr(fig, "_report_panel_map", None) or {})
        disp_id = panel_map.get(spec_pid) if spec_pid is not None else None
        try:
            fig.selected_panel = disp_id or ""
        except Exception as e:
            log.debug("applying selected_panel failed: %s", e)

    def select_panel(self, cell_id: str, panel_id: "str | None") -> None:
        """Record the selected SPEC panel id (or None = figure-level) for *cell_id*,
        push the persistent outline onto the live figure (``selected_panel`` = the
        panel's anyplotlib dispatch id, "" for None), and emit
        ``report_panel_selected`` so the dock mirrors it. A ``panel_id`` that isn't
        one of the cell's panels is treated as None (figure-level)."""
        cell = self.doc.cell_by_id(cell_id) if self.doc else None
        if cell is None or cell.cell_type != "figure":
            return
        # Normalise: only accept a panel id that actually exists on the spec.
        pid = None
        if panel_id is not None and cell.spec is not None:
            if any(p.id == panel_id for p in cell.spec.panels):
                pid = panel_id
        self._selected[cell_id] = pid
        fig = self.live_fig(cell_id)
        if fig is not None:
            self._apply_selected_panel(cell_id, fig)
        ipc.emit({"type": "report_panel_selected", "cell_id": cell_id,
                  "panel_id": pid})

    # ── in-place live updates (skip the figure rebuild / iframe flash) ──────────

    def live_fig(self, cell_id: str):
        """The live anyplotlib Figure for a cell (via its window controller), or
        None if the cell has no mounted figure window."""
        wid = self._window_by_cell.get(cell_id)
        ctrl = self._controllers.get(wid) if wid is not None else None
        return getattr(ctrl, "fig", None) if ctrl is not None else None

    def push_fig_markers(self, cell) -> bool:
        """Re-sync a cell's figure-level annotations onto the LIVE figure's marker
        layer (``fig.set_figure_markers``) — a targeted redraw, NO rebuild / iframe
        reload. Returns True when it reached a live figure, False otherwise (caller
        should fall back to a rebuild). ``spec.annotations`` are already in the
        anyplotlib figure-marker fraction schema, so they pass straight through."""
        if cell is None or cell.spec is None:
            return False
        fig = self.live_fig(cell.id)
        if fig is None or not hasattr(fig, "set_figure_markers"):
            return False
        try:
            fig.set_figure_markers(list(cell.spec.annotations))
            return True
        except Exception as e:
            log.debug("push_fig_markers failed (cell %s): %s", cell.id, e)
            return False

    def push_ann_widget(self, cell_id: str, panel_id: str, ann_index: int,
                        widget_fields: dict) -> bool:
        """Push *widget_fields* (already in the WIDGET's image-pixel / attr schema)
        onto the live edit widget for ``(panel_id, ann_index)`` via ``widget.set``
        — a targeted JS merge + redraw, NO figure rebuild. Returns True when the
        widget was found and updated, False otherwise (caller rebuilds instead).

        Only valid while the cell is in EDIT MODE (widgets exist); a non-edit cell
        has no ``_ann_widgets`` entry → returns False."""
        widget = (self._ann_widgets.get(cell_id, {}) or {}).get((panel_id, ann_index))
        if widget is None or not widget_fields:
            return False
        try:
            widget.set(**widget_fields)
            return True
        except Exception as e:
            log.debug("push_ann_widget failed (cell %s panel %s idx %s): %s",
                      cell_id, panel_id, ann_index, e)
            return False

    def assemble_assets(self, harvested: dict) -> dict:
        """Build ``{cell_id -> PNG bytes}`` for every non-placeholder figure cell,
        preferring (in order): the renderer-harvested PNG, a fresh bake of the held
        snapshot, then the PNG loaded from an opened report. The shared basis for
        the zip save AND every export path so all writes get identical pixels."""
        assets: dict[str, bytes] = {}
        for c in self.doc.cells:
            if c.cell_type != "figure" or c.placeholder:
                continue
            png = harvested.get(c.id)
            # Treat an EMPTY harvest (b"" from a "data:image/png;base64," data URL
            # with no payload) as missing, so we still bake / fall back — otherwise
            # write_report's ``if png:`` guard would silently skip the asset, leaving
            # a dangling image ref in report.md.
            if not png:
                arr = self.primary_snapshot(c.id)
                if arr is not None:
                    try:
                        layer = c.spec.primary_layer if c.spec else None
                        cmap = layer.cmap if layer else "viridis"
                        clim = layer.clim if layer else None
                        png = bake_fallback_png(arr, cmap=cmap, clim=clim)
                    except Exception as e:
                        log.debug("asset bake failed for cell %s: %s", c.id, e)
                if not png:
                    png = self._baked.get(c.id)
            if png is not None:
                assets[c.id] = png
        return assets

    def _forget(self, window_id: int) -> None:
        forget = getattr(self.session, "_forget_window", None)
        if forget is not None:
            try:
                forget(int(window_id))
            except Exception as e:
                log.debug("report _forget failed: %s", e)
        self._controllers.pop(int(window_id), None)
        for cid, wid in list(self._window_by_cell.items()):
            if wid == window_id:
                self._window_by_cell.pop(cid, None)


# ── edit-mode annotation drag persistence ──────────────────────────────────────


def _make_annotation_drag_handler(mgr, cell_id, panel_spec, ann_index, kind):
    """A module-level closure factory (NOT a bound method — anyplotlib's
    ``add_event_handler`` sets ``fn._event_types`` on the handler, which fails on a
    bound method) that returns a ``pointer_up`` handler for one annotation widget.

    On drop (runs on the asyncio main thread; the widget's ``_data`` already carries
    the final DRAGGED geometry in image pixels) it converts px → DATA and rewrites
    ONLY the geometric keys of ``panel_spec.annotations[ann_index]`` in place
    (offsets; radius; widths/heights; U/V — never texts/colors), sets
    ``mgr.dirty`` + emits the authoritative state. It does NOT rebuild the figure
    (the widget already moved JS-side; a rebuild would flash the iframe).

    Guards (no-op, no crash): the cell must still exist in ``mgr.doc``; the panel
    must still be one of the cell's panels; ``ann_index`` must be in range; the
    widget geometry must be readable."""
    from spyde.actions.report import coords

    def _on_drag_end(event):
        try:
            widget = getattr(event, "source", None)
            if widget is None:
                return
            # Guards: cell + panel + index must still be valid.
            cell = mgr.doc.cell_by_id(cell_id) if mgr.doc else None
            if cell is None or cell.spec is None:
                return
            if panel_spec not in cell.spec.panels:
                return
            anns = panel_spec.annotations
            if not (0 <= ann_index < len(anns)):
                return
            ann = anns[ann_index]
            axes = panel_spec.axes
            new_geom = _widget_geometry_to_data(kind, widget, axes, coords)
            if not new_geom:
                return
            changed = False
            for key, val in new_geom.items():
                if ann.get(key) != val:
                    ann[key] = val
                    changed = True
            # A drag on a panel annotation also SELECTS that panel (the dock
            # follows the last-touched panel). Best-effort; independent of the
            # geometry change so a no-move drag still selects.
            try:
                mgr.select_panel(cell_id, panel_spec.id)
            except Exception as e:
                log.debug("annotation drag panel-select failed: %s", e)
            if changed:
                mgr.dirty = True
                mgr.emit_state()
        except Exception as e:
            log.debug("annotation drag persist failed (cell %s idx %s): %s",
                      cell_id, ann_index, e)

    return _on_drag_end


def _make_panel_select_handler(mgr, cell_id, spec_panel_id):
    """A module-level closure (NOT a bound method — anyplotlib sets
    ``fn._event_types`` on it) returning a ``pointer_down`` handler that selects
    one panel. Fires on a genuine panel click that misses widgets (anyplotlib only
    fires ``pointer_down`` on the panel plot when the click is not on a widget), so
    a click on a panel → its spec panel becomes the selection."""
    def _on_panel_down(event):
        try:
            mgr.select_panel(cell_id, spec_panel_id)
        except Exception as e:
            log.debug("panel-select handler failed (cell %s panel %s): %s",
                      cell_id, spec_panel_id, e)
    return _on_panel_down


def _make_figure_edit_handler(mgr, cell_id):
    """A module-level closure returning a FIGURE-level handler (registered for
    ``pointer_down`` + ``pointer_up``) that routes the three figure-scoped edit
    events:

    * ``panel_swap`` (``event.source_panel_id`` + ``event.target_panel_id`` set —
      the user dragged one panel's move-grip onto another) → swap the two spec
      panels' ``grid_pos`` and REBUILD (anyplotlib reorders nothing itself).
    * ``figure_background`` (a click on the bare figure, no panel underneath) →
      ``select_panel(cell_id, None)`` (deselect → figure-level dock).
    * ``figure_marker`` on ``pointer_up`` (a figure annotation was dragged) →
      persist the marker's updated FRACTION fields into ``spec.annotations``
      (matched by id — anyplotlib has ALREADY merged them into its own
      ``figure_markers`` before firing), set ``mgr.dirty``, re-emit the state,
      and do NOT rebuild (the marker already moved JS-side).

    anyplotlib's figure event carries the flat fields on the ``Event``; the marker
    id rides in ``event.last_widget_id`` and the authoritative moved marker is read
    back off ``fig.figure_markers`` so we persist exactly what the JS committed."""
    def _on_figure_event(event):
        try:
            fig = getattr(event, "source", None)
            etype = getattr(event, "event_type", "")
            marker_id = getattr(event, "last_widget_id", None)
            cell = mgr.doc.cell_by_id(cell_id) if mgr.doc else None
            if cell is None or cell.cell_type != "figure":
                return
            # panel drag-swap: two panel DISPATCH ids on the event (mapped back to
            # spec panel ids via fig._report_panel_map). Handled BEFORE the
            # background/marker branches (a swap carries no marker id).
            src_disp = getattr(event, "source_panel_id", None)
            tgt_disp = getattr(event, "target_panel_id", None)
            if src_disp is not None and tgt_disp is not None:
                _handle_panel_swap(mgr, cell_id, fig, src_disp, tgt_disp)
                return
            # figure-background click → deselect (any pointer event; a click
            # arrives as pointer_down). No marker id on a background click.
            if marker_id is None:
                if etype == "pointer_down":
                    mgr.select_panel(cell_id, None)
                return
            # figure-marker drag end → persist the moved marker.
            if etype != "pointer_up" or cell.spec is None or fig is None:
                return
            markers = getattr(fig, "figure_markers", None) or []
            moved = next((dict(m) for m in markers
                          if m.get("id") == marker_id), None)
            if moved is None:
                return
            anns = cell.spec.annotations
            # Match the stored annotation by id; fall back to positional index if
            # ids weren't assigned yet (set_figure_markers assigns them on build).
            target = None
            for a in anns:
                if a.get("id") == marker_id:
                    target = a
                    break
            if target is None:
                # No id match — append the moved marker (a new figure annotation
                # created + dragged before its id round-tripped to the spec).
                return
            changed = False
            for key, val in moved.items():
                if target.get(key) != val:
                    target[key] = val
                    changed = True
            if changed:
                mgr.dirty = True
                mgr.emit_state()
        except Exception as e:
            log.debug("figure-level edit handler failed (cell %s): %s",
                      cell_id, e)
    return _on_figure_event


def _handle_panel_swap(mgr, cell_id, fig, src_disp, tgt_disp) -> None:
    """Swap two panels' ``grid_pos`` in response to a panel drag-swap event.

    *src_disp* / *tgt_disp* are anyplotlib PLOT dispatch ids; invert
    ``fig._report_panel_map`` (spec_pid → dispatch id) to get the two SPEC panel
    ids, find both ``PanelSpec``s, exchange their ``grid_pos``, mark dirty, and
    REBUILD the figure (anyplotlib performs no layout change itself — it only
    reports the intent).

    Guards (no-op, no crash): the cell/spec must exist; both dispatch ids must map
    to a distinct spec panel; both panels must still be on the spec. An unknown id
    or a same-panel drop is a no-op."""
    cell = mgr.doc.cell_by_id(cell_id) if mgr.doc else None
    if cell is None or cell.cell_type != "figure" or cell.spec is None:
        return
    if src_disp == tgt_disp:
        return
    panel_map = dict(getattr(fig, "_report_panel_map", None) or {})
    # dispatch id → spec panel id (inverse of the stashed panel_map).
    spec_by_disp = {disp: pid for pid, disp in panel_map.items()}
    src_pid = spec_by_disp.get(src_disp)
    tgt_pid = spec_by_disp.get(tgt_disp)
    if src_pid is None or tgt_pid is None or src_pid == tgt_pid:
        return
    src_panel = next((p for p in cell.spec.panels if p.id == src_pid), None)
    tgt_panel = next((p for p in cell.spec.panels if p.id == tgt_pid), None)
    if src_panel is None or tgt_panel is None:
        return
    # Exchange grid positions (swap in place; keep every other panel attr).
    src_panel.grid_pos, tgt_panel.grid_pos = (
        list(tgt_panel.grid_pos), list(src_panel.grid_pos))
    mgr.dirty = True
    mgr.build_figure_window(cell)
    mgr.emit_state()


def _widget_geometry_to_data(kind, widget, axes, coords) -> "dict | None":
    """Read one edit widget's final IMAGE-PIXEL geometry from its ``_data`` and
    convert to the DATA-coordinate annotation keys (the inverse of
    ``figure_builder._add_annotation_widget``'s field mapping). Returns a dict of
    ONLY the geometric keys to rewrite (offsets / radius / widths+heights / U+V),
    or None if the geometry can't be read.

    Inverse field mapping (widget px → spec data):
      * text (label)  → ``x, y``            → ``offsets: [[dx, dy]]``
      * circle        → ``cx, cy, r``       → ``offsets: [[dx, dy]]``, ``radius``
      * rect          → ``x, y, w, h`` (TOP-LEFT) → CENTER + ``widths/heights``
      * arrow         → ``x, y, u, v``      → ``offsets: [[tail]]``, ``U``, ``V``"""
    g = getattr(widget, "_data", None)
    if not isinstance(g, dict):
        return None
    if kind == "text":
        dx, dy = coords.pixel_to_data_point(g.get("x"), g.get("y"), axes)
        return {"offsets": [[dx, dy]]}
    if kind == "circle":
        dx, dy = coords.pixel_to_data_point(g.get("cx"), g.get("cy"), axes)
        r = coords.pixel_to_data_radius(g.get("r"), axes)
        return {"offsets": [[dx, dy]], "radius": r}
    if kind == "rect":
        # widget x/y is the TOP-LEFT; the spec offset is the CENTER.
        px, py = float(g.get("x", 0.0)), float(g.get("y", 0.0))
        pw, ph = float(g.get("w", 0.0)), float(g.get("h", 0.0))
        cx_px, cy_px = px + pw / 2.0, py + ph / 2.0
        dx, dy = coords.pixel_to_data_point(cx_px, cy_px, axes)
        w = coords.pixel_to_data_width(pw, axes)
        hh = coords.pixel_to_data_height(ph, axes)
        return {"offsets": [[dx, dy]], "widths": [w], "heights": [hh]}
    if kind == "arrow":
        dx, dy = coords.pixel_to_data_point(g.get("x"), g.get("y"), axes)
        u = coords.pixel_to_data_u(g.get("u"), axes)
        v = coords.pixel_to_data_v(g.get("v"), axes)
        return {"offsets": [[dx, dy]], "U": [u], "V": [v]}
    return None


def _manager(session) -> ReportManager:
    """Return (creating lazily) the session's ReportManager."""
    mgr = getattr(session, "_report", None)
    if mgr is None:
        mgr = ReportManager(session)
        session._report = mgr
    return mgr


def _ensure_open(session) -> ReportManager:
    """Return the manager, creating a fresh empty report if none is open."""
    mgr = _manager(session)
    if not mgr.open:
        mgr.new()
    return mgr


# ── snapshotting a live Plot into a FigureSpec + held array ────────────────────


def _snapshot_plot(plot) -> "tuple[FigureSpec, dict] | None":
    """Snapshot a live ``Plot`` NOW into a single-panel FigureSpec + a
    ``{(panel_id, layer_id): ndarray}`` snapshot map. Reads ``current_data``,
    ``_last_levels``, colormap, axes, title, nav indices, and the view label.

    If the plot carries live MDI overlay layers (``plot._layers``), each is
    serialized into the same panel as an extra LayerSpec (same cmap / alpha, its
    own source ref) with its current frame in the snapshot map — so "Add to report"
    on a layered plot captures the whole composite. Returns None when the plot has
    no paintable base frame."""
    data = getattr(plot, "current_data", None)
    if not isinstance(data, np.ndarray) or data.dtype == object:
        return None
    arr = np.array(data, copy=True)   # detach from the live buffer

    # colormap
    cmap = "gray"
    try:
        ps = getattr(plot, "plot_state", None)
        if ps is not None and getattr(ps, "colormap", None):
            cmap = str(ps.colormap)
    except Exception:
        pass

    # contrast (held levels), skip for RGB
    clim = None
    lv = getattr(plot, "_last_levels", None)
    is_rgb = arr.ndim == 3 and arr.shape[-1] in (3, 4)
    if lv is not None and not is_rgb:
        try:
            clim = [float(lv[0]), float(lv[1])]
        except Exception:
            clim = None

    # axes / units / title
    axes_dict = None
    title = ""
    try:
        axes, units = plot._axes_info(arr)
        if axes is not None:
            axes_dict = {
                "units": units,
                "x_axis": [float(v) for v in np.asarray(axes[0])],
                "y_axis": [float(v) for v in np.asarray(axes[1])],
            }
    except Exception as e:
        log.debug("snapshot axes read failed: %s", e)
    try:
        title = plot._plot_title()
    except Exception:
        title = ""

    # nav indices (position snapshot)
    nav_context = None
    try:
        sig = plot.plot_state.current_signal
        idx = tuple(int(i) for i in sig.axes_manager.indices)
        if idx:
            nav_context = {"indices": list(idx)}
    except Exception:
        nav_context = None

    base_layer = LayerSpec(source=SignalRef.from_plot(plot), cmap=cmap,
                           clim=clim, alpha=1.0, visible=True)
    layers = [base_layer]
    snap_map = {("p1", base_layer.id): arr}

    # Live MDI overlay layers → extra LayerSpecs on the same panel (base + overlays).
    for live in list(getattr(plot, "_layers", None) or []):
        src = getattr(live, "source_plot", None)
        frame = getattr(src, "current_data", None) if src is not None else None
        if not isinstance(frame, np.ndarray) or frame.dtype == object or frame.ndim != 2:
            continue
        ov = LayerSpec(source=(SignalRef.from_plot(src) if src is not None else SignalRef()),
                       cmap=str(getattr(live, "cmap", "magma")),
                       clim=(list(live.clim) if getattr(live, "clim", None) else None),
                       alpha=float(getattr(live, "alpha", 0.5)),
                       visible=bool(getattr(live, "visible", True)))
        layers.append(ov)
        snap_map[("p1", ov.id)] = np.array(frame, copy=True)

    panel = PanelSpec(id="p1", grid_pos=[0, 0], kind="image", layers=layers,
                      axes=axes_dict, title=title,
                      scalebar=bool(axes_dict is not None))
    spec = FigureSpec(layout={"kind": "single"}, panels=[panel],
                      nav_context=nav_context)
    return spec, snap_map


def _snapshot_layer_now(plot) -> "tuple[np.ndarray, str, list | None] | None":
    """Read a live ``Plot`` NOW into ``(arr, cmap, clim)`` for an EXISTING
    LayerSpec refresh (per-panel / per-layer re-snapshot) — the pixel + contrast
    half of :func:`_snapshot_plot`'s base-layer logic, factored out so a panel
    refresh can update one layer's pixels/cmap/clim without rebuilding the whole
    FigureSpec (axes/title/annotations/alpha/visible are left untouched — a
    refresh keeps the panel's edited chrome). Returns None when the plot has no
    paintable frame."""
    data = getattr(plot, "current_data", None)
    if not isinstance(data, np.ndarray) or data.dtype == object:
        return None
    arr = np.array(data, copy=True)   # detach from the live buffer

    cmap = "gray"
    try:
        ps = getattr(plot, "plot_state", None)
        if ps is not None and getattr(ps, "colormap", None):
            cmap = str(ps.colormap)
    except Exception:
        pass

    clim = None
    lv = getattr(plot, "_last_levels", None)
    is_rgb = arr.ndim == 3 and arr.shape[-1] in (3, 4)
    if lv is not None and not is_rgb:
        try:
            clim = [float(lv[0]), float(lv[1])]
        except Exception:
            clim = None

    return arr, cmap, clim


def refresh_panel(session, mgr: "ReportManager", cell: Cell, panel: PanelSpec) -> bool:
    """Re-snapshot ONE panel's layers from their resolved live plots, IN PLACE.

    Every layer's ``source`` ref must resolve to a live plot for the panel to
    refresh; if ANY layer is unresolvable the panel is left untouched (its
    existing snapshot/spec stay exactly as they were) — matching the whole-figure
    refresh's "skip on unresolved source" behaviour, just scoped to one panel.
    On success, each layer's pixels/cmap/clim are updated (alpha/visible/id and
    the panel's axes/title/annotations are left alone — a refresh pulls fresh
    data, it doesn't discard the user's edits) and the manager's snapshot map is
    updated. Returns True when the panel was refreshed, False otherwise (caller
    decides whether that means "mark offline" — see ``report_refresh_figure`` —
    or "leave silently as-is" — ``repfig_refresh_panel``)."""
    if not panel.layers:
        return False
    resolved: list[tuple[LayerSpec, np.ndarray, str, "list | None"]] = []
    for layer in panel.layers:
        src_plot = layer.source.resolve(session) if layer.source else None
        if src_plot is None:
            return False
        snap = _snapshot_layer_now(src_plot)
        if snap is None:
            return False
        arr, cmap, clim = snap
        resolved.append((layer, arr, cmap, clim))
    for layer, arr, cmap, clim in resolved:
        layer.cmap = cmap
        layer.clim = clim
        mgr.set_snapshot(cell.id, panel.id, layer.id, arr)
    return True


def _resolve_source_plot(session, source_window_id):
    """The Plot for a source window id (the plot the user dragged into the report)."""
    if source_window_id is None:
        return None
    return session._plot_by_window_id(int(source_window_id))


# ── handlers ───────────────────────────────────────────────────────────────────


def report_new(session, plot, payload) -> None:
    mgr = _manager(session)
    mgr.new(template=bool(payload.get("template", False)))
    mgr.emit_state()


def report_open(session, plot, payload) -> None:
    path = payload.get("path")
    if not path:
        ipc.emit_error("report_open: no path.")
        return
    mgr = _manager(session)
    try:
        doc, assets = read_report(path)
    except Exception as e:
        ipc.emit_error(f"Opening report failed: {e}")
        return
    mgr.close_windows()
    mgr.doc = doc
    mgr.path = path
    mgr.dirty = False
    mgr._snapshots.clear()
    mgr._baked = dict(assets)
    mgr._offline.clear()
    # Rebind each figure cell: resolve EVERY layer of EVERY panel against open trees
    # / files. The cell rebinds live only when ALL its layers resolve; if any layer's
    # source is offline the whole cell is offline (renderer shows the baked PNG).
    for c in doc.cells:
        if c.cell_type != "figure" or c.placeholder or c.spec is None:
            continue
        all_resolved = bool(c.spec.panels)
        for panel in c.spec.panels:
            for layer in panel.layers:
                src_plot = layer.source.resolve(session) if layer.source else None
                arr = None
                if src_plot is not None:
                    frame = getattr(src_plot, "current_data", None)
                    if isinstance(frame, np.ndarray) and frame.dtype != object:
                        arr = np.array(frame, copy=True)
                if arr is None:
                    all_resolved = False
                else:
                    # Keep the SAVED spec (cmap/clim/axes/title) but the live pixels.
                    mgr.set_snapshot(c.id, panel.id, layer.id, arr)
        if all_resolved:
            mgr.build_figure_window(c)
        else:
            # Unresolved → offline: renderer shows the baked PNG (data URL in state).
            mgr._snapshots.pop(c.id, None)
            mgr._offline.add(c.id)
    mgr.emit_state()


def harvest_snapshots(session, mgr: ReportManager, finish) -> None:
    """Run the renderer PNG-harvest handshake, then call ``finish(harvested)``.

    Shared by ``report_save`` AND the HTML/markdown exporters so every write path
    gets FRESH live-figure pixels (0f export protocol) with the SAME 3 s
    fallback-bake safety: emit ``report_need_snapshots``, accept a matching
    ``report_snapshots`` reply, and — if it's slow/missing — fire ``finish`` with
    whatever arrived (``finish`` fills the gaps from held / baked snapshots).

    ``finish`` is ``finish(harvested: dict[cell_id -> png bytes]) -> None``. When
    there are no mounted figures OR no event loop (headless / tests), ``finish``
    is called synchronously with an empty dict — the write happens NOW."""
    fig_cells = [c for c in mgr.doc.cells
                 if c.cell_type == "figure" and not c.placeholder]
    live_cells = [c for c in fig_cells if c.id in mgr._window_by_cell]

    if not live_cells or getattr(session, "_main_loop", None) is None:
        finish({})
        return

    import uuid as _uuid
    token = _uuid.uuid4().hex
    mgr._pending_save[token] = {
        "cell_ids": [c.id for c in live_cells],
        "harvested": {},
        "finish": finish,
    }
    ipc.emit({
        "type": "report_need_snapshots", "token": token,
        "cells": [{"cell_id": c.id, "fig_id": c.id} for c in live_cells],
    })
    loop = session._main_loop

    def _timeout():
        pend = mgr._pending_save.pop(token, None)
        if pend is None:
            return   # already completed by report_snapshots
        pend["finish"](pend["harvested"])

    try:
        loop.call_later(_SNAPSHOT_TIMEOUT_S, _timeout)
    except Exception as e:
        log.debug("snapshot-harvest timeout arm failed, finishing now: %s", e)
        mgr._pending_save.pop(token, None)
        finish({})


def report_save(session, plot, payload) -> None:
    """Save the report. Ask the renderer to harvest live-figure PNGs (0f export
    protocol); fall back to a baked PNG for any cell whose image doesn't arrive
    within a short timeout so a save NEVER hangs or fails for want of pixels."""
    mgr = _manager(session)
    if not mgr.open:
        ipc.emit_error("report_save: no open report.")
        return
    path = payload.get("path") or mgr.path
    if not path:
        ipc.emit_error("report_save: no path (use Save As).")
        return
    mgr.path = path
    harvest_snapshots(session, mgr,
                      lambda harvested: _finish_save(session, mgr, path, harvested))


def report_snapshots(session, plot, payload) -> None:
    """Renderer-harvested PNGs for a pending harvest handshake (save OR export).
    Decode the data URLs and hand them to the pending entry's ``finish``
    callback (fallback-baking any still-missing cell)."""
    mgr = _manager(session)
    token = payload.get("token")
    pend = mgr._pending_save.pop(token, None) if token else None
    images = payload.get("images") or {}
    harvested: dict[str, bytes] = {}
    for cell_id, data_url in images.items():
        png = _decode_data_url(data_url)
        if png is not None:
            harvested[cell_id] = png
    if pend is None:
        # Timeout already fired (or unknown token) — nothing to complete. The
        # write already happened via the timeout path.
        return
    pend["harvested"].update(harvested)
    finish = pend.get("finish")
    if finish is None:
        # Every pending entry armed by ``harvest_snapshots`` carries a ``finish``
        # callback; a pend with none is malformed. Log and return rather than
        # KeyError on the old ``pend["path"]`` shape (which no longer exists).
        log.debug("report_snapshots: pending entry has no finish callback; "
                  "ignoring (token=%s)", token)
        return
    finish(pend["harvested"])


def _finish_save(session, mgr: ReportManager, path: str,
                 harvested: dict) -> None:
    """Assemble the asset PNGs (harvested → held-baked → offline-baked) and write
    the zip atomically, then emit ``report_saved`` + refresh state."""
    assets = mgr.assemble_assets(harvested)
    try:
        mgr.doc.touch()
        write_report(mgr.doc, path, assets=assets)
    except Exception as e:
        ipc.emit_error(f"Saving report failed: {e}")
        return
    mgr.path = path
    mgr.dirty = False
    ipc.emit({"type": "report_saved", "path": path})
    mgr.emit_state()


def report_save_as_template(session, plot, payload) -> None:
    """Save the report as a TEMPLATE (figure cells become placeholders on load,
    ready to be filled). Marks the doc template=True for the saved copy."""
    mgr = _manager(session)
    if not mgr.open:
        ipc.emit_error("report_save_as_template: no open report.")
        return
    path = payload.get("path")
    if not path:
        ipc.emit_error("report_save_as_template: no path.")
        return
    # A template keeps the cell structure but ships placeholders (no baked
    # pixels), so filling it later starts from an empty drop zone.
    assets: dict[str, bytes] = {}
    template_doc = ReportDoc(
        title=mgr.doc.title, template=True, version=mgr.doc.version,
        created=mgr.doc.created,
    )
    for c in mgr.doc.cells:
        if c.cell_type == "markdown":
            template_doc.cells.append(Cell(id=c.id, cell_type="markdown",
                                           source=c.source))
        elif c.cell_type == "figure":
            template_doc.cells.append(Cell(id=c.id, cell_type="figure",
                                           caption=c.caption, placeholder=True))
    try:
        write_report(template_doc, path, assets=assets)
    except Exception as e:
        ipc.emit_error(f"Saving template failed: {e}")
        return
    ipc.emit({"type": "report_saved", "path": path})
    mgr.emit_state()


def report_close(session, plot, payload) -> None:
    mgr = _manager(session)
    mgr.close()
    mgr.emit_state()


def report_add_cell(session, plot, payload) -> None:
    mgr = _ensure_open(session)
    cell_type = str(payload.get("cell_type", "markdown"))
    if cell_type != "markdown":
        ipc.emit_error(f"report_add_cell: unsupported cell_type {cell_type!r}.")
        return
    cell = Cell(id=new_cell_id(), cell_type="markdown",
                source=str(payload.get("source", "") or ""))
    # OPTIONAL sanitized HTML fragment from the renderer (marked+DOMPurify) —
    # a derived, non-persisted field used only by HTML export.
    if payload.get("html") is not None:
        cell.html = str(payload.get("html") or "")
    _insert_cell(mgr.doc, cell, payload.get("index"))
    mgr.dirty = True
    mgr.emit_state()


def report_update_cell(session, plot, payload) -> None:
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None or cell.cell_type != "markdown":
        return
    cell.source = str(payload.get("source", "") or "")
    # Refresh the derived (non-persisted) rendered-HTML fragment when the
    # renderer sends one; otherwise the stale fragment is dropped so export can't
    # emit HTML that no longer matches the source.
    if "html" in payload:
        cell.html = str(payload.get("html") or "")
    else:
        cell.html = ""
    mgr.dirty = True
    mgr.emit_state()


def report_remove_cell(session, plot, payload) -> None:
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None:
        return
    # Tear down the figure window (if any) so nothing leaks.
    if cell.cell_type == "figure":
        wid = mgr._window_by_cell.get(cell.id)
        if wid is not None:
            mgr._forget(wid)
        mgr._snapshots.pop(cell.id, None)
        mgr._baked.pop(cell.id, None)
        mgr._offline.discard(cell.id)
        mgr._editing.discard(cell.id)
        mgr._edit_wiring.pop(cell.id, None)
        mgr._ann_widgets.pop(cell.id, None)
        mgr._selected.pop(cell.id, None)
    mgr.doc.cells = [c for c in mgr.doc.cells if c.id != cell.id]
    mgr.dirty = True
    mgr.emit_state()


def report_move_cell(session, plot, payload) -> None:
    mgr = _manager(session)
    if not mgr.open:
        return
    cell_id = payload.get("cell_id")
    cur = mgr.doc.index_of(cell_id)
    if cur < 0:
        return
    idx = payload.get("index")
    idx = len(mgr.doc.cells) if idx is None else int(idx)
    if cur < idx:
        idx -= 1
    cell = mgr.doc.cells.pop(cur)
    idx = max(0, min(idx, len(mgr.doc.cells)))
    mgr.doc.cells.insert(idx, cell)
    mgr.dirty = True
    mgr.emit_state()


def report_set_caption(session, plot, payload) -> None:
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None or cell.cell_type != "figure":
        return
    cell.caption = str(payload.get("caption", "") or "")
    mgr.dirty = True
    mgr.emit_state()


def report_set_title(session, plot, payload) -> None:
    mgr = _manager(session)
    if not mgr.open:
        return
    mgr.doc.title = str(payload.get("title", "") or "")
    mgr.dirty = True
    mgr.emit_state()


def report_add_figure(session, plot, payload) -> None:
    """Snapshot the source window's Plot NOW into a figure cell. ``at_cell`` fills
    an existing placeholder in place; otherwise a new figure cell is inserted."""
    mgr = _ensure_open(session)
    src = _resolve_source_plot(session, payload.get("source_window_id"))
    if src is None:
        ipc.emit_error("report_add_figure: source window not found.")
        return
    snap = _snapshot_plot(src)
    if snap is None:
        ipc.emit_error("report_add_figure: source window has no image to snapshot.")
        return
    spec, snap_map = snap
    caption = str(payload.get("caption", "") or "")

    at_cell = payload.get("at_cell")
    cell = mgr.doc.cell_by_id(at_cell) if at_cell else None
    if cell is not None and cell.cell_type == "figure":
        # Fill a placeholder (or replace an existing figure) in place.
        cell.placeholder = False
        cell.spec = spec
        if caption:
            cell.caption = caption
    else:
        cell = Cell(id=new_cell_id(), cell_type="figure", caption=caption,
                    placeholder=False, spec=spec)
        _insert_cell(mgr.doc, cell, payload.get("index"))

    mgr._snapshots[cell.id] = dict(snap_map)
    mgr._baked.pop(cell.id, None)
    mgr._offline.discard(cell.id)
    mgr.build_figure_window(cell)
    mgr.dirty = True
    mgr.emit_state()


def report_refresh_figure(session, plot, payload) -> None:
    """Re-snapshot EVERY panel of a figure cell from its resolved live plot(s)
    and re-emit — i.e. ``refresh_panel`` run for each panel in turn (one code
    path shared with the single-panel ``repfig_refresh_panel``). A panel whose
    source(s) can't be resolved keeps its existing snapshot; if NO panel
    resolved (nothing refreshed at all) the whole cell is marked offline, same
    as before this was panel-scoped."""
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None or cell.cell_type != "figure" or cell.spec is None:
        return
    any_refreshed = False
    for panel in cell.spec.panels:
        if refresh_panel(session, mgr, cell, panel):
            any_refreshed = True
    if not any_refreshed:
        # Nothing resolved — can't refresh; mark offline so the UI reflects it.
        mgr._offline.add(cell.id)
        wid = mgr._window_by_cell.get(cell.id)
        if wid is not None:
            mgr._forget(wid)
        mgr.emit_state()
        return
    mgr._offline.discard(cell.id)
    mgr.build_figure_window(cell)
    mgr.dirty = True
    mgr.emit_state()


def repfig_refresh_panel(session, plot, payload) -> None:
    """Re-snapshot ONLY ONE panel (``{cell_id, panel_id}``) of a figure cell from
    its resolved live plot(s), then rebuild + re-emit the whole cell (the
    renderer swaps the iframe seamlessly, so a full rebuild for a one-panel
    change is fine — same rebuild path every other repfig edit uses).

    If the panel's source(s) can't be resolved, the panel keeps its existing
    snapshot untouched (no offline flag on the panel/cell — the OTHER panels are
    still live) and this is a silent no-op (no error; a transient source can
    reconnect on a later refresh). An unknown ``cell_id``/``panel_id`` is
    likewise a no-op — no crash."""
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None or cell.cell_type != "figure" or cell.spec is None:
        return
    panel = next((p for p in cell.spec.panels
                  if p.id == payload.get("panel_id")), None)
    if panel is None:
        return
    if not refresh_panel(session, mgr, cell, panel):
        return
    mgr.build_figure_window(cell)
    mgr.dirty = True
    mgr.emit_state()


def report_cell_from_window(session, plot, payload) -> None:
    """Minimal 'copy this window as a report figure': build the cell + spec and
    append it to the report (Phase 3 copy/paste reuses this)."""
    payload = {**payload}
    payload.pop("at_cell", None)   # always append
    report_add_figure(session, plot, payload)


# ── helpers ────────────────────────────────────────────────────────────────────


def _insert_cell(doc: ReportDoc, cell: Cell, index) -> None:
    if index is None:
        doc.cells.append(cell)
    else:
        idx = max(0, min(int(index), len(doc.cells)))
        doc.cells.insert(idx, cell)


def _decode_data_url(data_url) -> "bytes | None":
    """Decode a ``data:image/png;base64,...`` URL to PNG bytes (or raw base64)."""
    if not isinstance(data_url, str) or not data_url:
        return None
    try:
        if "," in data_url and data_url.strip().lower().startswith("data:"):
            data_url = data_url.split(",", 1)[1]
        return base64.b64decode(data_url)
    except Exception as e:
        log.debug("decoding snapshot data URL failed: %s", e)
        return None

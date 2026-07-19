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
    IMAGE_EXTS, Cell, FigureSpec, LayerSpec, PanelSpec, ReportDoc, SignalRef,
    bake_fallback_png, bake_line_fallback_png, new_cell_id, read_report,
    write_report,
)

log = logging.getLogger(__name__)

# Fallback-bake if the renderer's harvested PNGs don't arrive within this window.
_SNAPSHOT_TIMEOUT_S = 3.0
# Cap the inline offline-fallback PNG (data URL in report_state) so a long
# offline report doesn't balloon the state message.
_OFFLINE_PNG_MAX_EDGE = 640
# Cap the DECODED bytes of an image (photo) cell — refuse a giant photo so a
# single drop/paste can't balloon the report file / the report_state message.
_IMAGE_CELL_MAX_BYTES = 10 * 1024 * 1024
# The pseudo layer ids a scene3d panel's snapshots are keyed under:
# (panel_id, "xyz") float32 (M,3) sphere points, (panel_id, "rgb") uint8 (M,3)
# IPF colours. NOT LayerSpec ids — a scene3d panel has no image layers.
_SCENE3D_SNAPSHOT_KEYS = ("xyz", "rgb")


def _is_scene3d_panel(panel) -> bool:
    return str(getattr(panel, "kind", "")) == "scene3d"


def _is_scene3d_cell(cell) -> bool:
    """True when *cell* is a figure cell whose spec contains a scene3d panel
    (the matplotlib Agg bake CANNOT render these — fallbacks must reuse a
    harvested/baked PNG or skip gracefully)."""
    spec = getattr(cell, "spec", None)
    return spec is not None and any(_is_scene3d_panel(p) for p in spec.panels)


def _is_line_panel(panel) -> bool:
    return str(getattr(panel, "kind", "")) == "line"


def _bake_primary_snapshot(cell, arr, *, max_edge: int) -> "bytes | None":
    """Bake a fallback PNG for *cell*'s primary-panel snapshot *arr*, routing
    to the LINE-panel Agg renderer (a real ``ax.plot``) when the cell's
    primary panel is ``kind="line"``, else the image heatmap renderer. Shared
    by ``_offline_png`` and ``assemble_assets`` so both bake paths agree."""
    panel = cell.spec.panels[0] if (cell.spec and cell.spec.panels) else None
    if panel is not None and _is_line_panel(panel):
        layer = panel.layers[0] if panel.layers else None
        x_axis = (panel.axes or {}).get("x_axis") if panel.axes else None
        x_units = str((panel.axes or {}).get("units") or "") if panel.axes else ""
        return bake_line_fallback_png(
            arr, x_axis=x_axis,
            color=(layer.color if layer and layer.color else "#4fc3f7"),
            linewidth=(layer.linewidth if layer and layer.linewidth else 1.5),
            label=(layer.label if layer else None), x_units=x_units)
    layer = cell.spec.primary_layer if cell.spec else None
    cmap = layer.cmap if layer else "viridis"
    clim = layer.clim if layer else None
    return bake_fallback_png(arr, cmap=cmap, clim=clim, max_edge=max_edge)


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
        # cell_id -> raw image bytes for an IMAGE (photo) cell — dropped / pasted /
        # browsed in. Held here so save can write them to assets/<id>.<ext> and
        # state() can emit them as a data URL; loaded back from the zip on open.
        self._images: dict[str, bytes] = {}
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
        legacy single-image path. None if the cell has no snapshots.

        A scene3d panel's snapshots live under the pseudo layer ids ``"xyz"`` /
        ``"rgb"`` (point-cloud arrays, not an image) — those are NEVER a valid
        primary IMAGE snapshot, so both the spec walk (its layer ids don't
        collide) and the any-array fallback skip them: baking an (M, 3) point
        array through matplotlib would produce garbage, not a figure."""
        cell = self.doc.cell_by_id(cell_id) if self.doc else None
        if cell is not None and cell.spec is not None:
            for panel in cell.spec.panels:
                for layer in panel.layers:
                    arr = self._snapshots.get(cell_id, {}).get(
                        self._layer_key(panel.id, layer.id))
                    if arr is not None:
                        return arr
        # Fallback: any IMAGE array in the map (scene3d point clouds excluded).
        m = self._snapshots.get(cell_id, {})
        for key, arr in m.items():
            if key[1] in _SCENE3D_SNAPSHOT_KEYS:
                continue
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
        self._images.clear()
        self._offline.clear()
        self._editing.clear()
        self._edit_wiring.clear()
        self._ann_widgets.clear()
        self._selected.clear()
        _clear_vectors_explorer_cache()

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
        self._images.clear()
        self._offline.clear()
        self._pending_save.clear()
        self._editing.clear()
        self._edit_wiring.clear()
        self._ann_widgets.clear()
        self._selected.clear()
        _clear_vectors_explorer_cache()

    # ── state emission ─────────────────────────────────────────────────────────

    def _panel_nav_dims(self, panel, cache: dict) -> int:
        """Navigation dimensionality of a panel's layer-0 source signal (0 when
        unresolvable) — an EPHEMERAL emit-time field the renderer uses to gate
        the fresh-slice callout buttons. Stamped onto the SHIPPED panel dicts
        only; never part of PanelSpec/to_dict/YAML. *cache* is per-emit, keyed
        by the SignalRef identity, so one emit resolves each source once."""
        # A scene3d panel has no image layers to slice — callouts never apply,
        # so skip the resolve entirely (cheap, and keeps the buttons hidden).
        if _is_scene3d_panel(panel):
            return 0
        if not panel.layers or panel.layers[0].source is None:
            return 0
        ref = panel.layers[0].source
        key = id(ref)
        if key in cache:
            return cache[key]
        dims = 0
        try:
            p = ref.resolve(self.session)
            if p is not None:
                dims = int(p.plot_state.current_signal
                           .axes_manager.navigation_dimension)
        except Exception as e:
            log.debug("panel nav_dims resolve failed: %s", e)
        cache[key] = dims
        return dims

    def state(self) -> dict:
        """The authoritative ``report_state`` message body."""
        if self.doc is None:
            return {"open": False, "path": None, "title": "", "template": False,
                    "dirty": False, "cells": []}
        cells = []
        nav_dims_cache: dict = {}
        for c in self.doc.cells:
            entry = {
                "id": c.id,
                "cell_type": c.cell_type,
                "source": c.source if c.cell_type == "markdown" else None,
                "caption": (c.caption
                            if c.cell_type in ("figure", "image") else None),
                "placeholder": bool(c.placeholder),
                "fig_id": c.id if c.cell_type == "figure" else None,
                "data_offline": bool(c.cell_type == "figure" and c.id in self._offline),
                # Present-mode fields (Phase 6): slide grouping + go-live handle
                # + 2-column layout ("" | "left" | "right") + per-slide kind/style
                # (title/section slide + background preset — carried on the
                # slide's first cell).
                "slide_break": bool(getattr(c, "slide_break", False)),
                "live_action": (dict(c.live_action)
                                if getattr(c, "live_action", None) else None),
                "column": str(getattr(c, "column", "") or ""),
                "slide_kind": str(getattr(c, "slide_kind", "") or ""),
                "slide_style": str(getattr(c, "slide_style", "") or ""),
                # Speaker notes (presenter view) — per-slide, carried on the
                # slide's first cell; free multi-line markdown, shown only in the
                # presenter view, never to the audience.
                "notes": str(getattr(c, "notes", "") or ""),
            }
            # An IMAGE (photo) cell ships its bytes as a data URL so the renderer
            # can draw the <img> with no round trip (mirrors the offline-figure
            # PNG emission below). The extension picks the MIME so jpg/gif/webp
            # render correctly.
            if c.cell_type == "image":
                img = self._images.get(c.id)
                if img is not None:
                    ext = (c.image_ext or "png").lower()
                    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
                    entry["image_ext"] = ext
                    entry["image"] = (f"data:image/{mime};base64,"
                                      + base64.b64encode(img).decode("ascii"))
            # For a figure cell, ship the PIXEL-FREE FigureSpec recipe (as a plain
            # dict) so the renderer's edit toolbar can list panels / layers /
            # annotations. It carries NO image bytes — only the YAML-shaped structure.
            if c.cell_type == "figure" and c.spec is not None:
                entry["figure"] = c.spec.to_dict()
                # Stamp nav_dims onto the SHIPPED panel dicts (ephemeral —
                # to_dict() built fresh dicts, so the spec is never touched).
                for pd, ps in zip(entry["figure"]["panels"], c.spec.panels):
                    pd["nav_dims"] = self._panel_nav_dims(ps, nav_dims_cache)
                # While the cell is in EDIT MODE, also ship the live widget-id →
                # annotation mapping so the renderer can resolve a clicked edit
                # widget (awi_event widget_id) to its spec annotation and open
                # the floating style popover. Ephemeral — never persisted.
                if c.id in self._editing:
                    amap = self._ann_widgets.get(c.id) or {}
                    entry["ann_widgets"] = {
                        str(w.id): {"panel_id": pid, "index": int(idx)}
                        for (pid, idx), w in amap.items()
                    }
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
                if cell is not None and cell.spec is not None:
                    return _bake_primary_snapshot(
                        cell, arr, max_edge=_OFFLINE_PNG_MAX_EDGE)
            except Exception as e:
                log.debug("offline png bake failed: %s", e)
        return None

    # ── figure cell build / rebuild ────────────────────────────────────────────

    def _vectors_explorer_for_cell(self, cell: Cell) -> "tuple[str, str] | None":
        """For a VIEWER-vectors figure cell (its resolved source tree carries
        ``diffraction_vectors`` and ``spec.vectors_mode != "image"``), build the
        self-contained interactive vectors explorer — the SAME page the HTML
        export emits — and return ``(fig_id, html)``, else None.

        This is Approach A: the sidebar cell hosts the EXACT export explorer
        (navigator + DP, crosshair/rectangle, pointer/integrate) as its figure
        iframe. It renders client-side (overlay-canvas disk splat) and is fully
        self-contained (inlines the anyplotlib ESM + the packed vectors blob), so
        the existing figure-emit path (``host:"report"`` → main writes
        ``spyde_fig_<figId>.html`` → served over ``spyde-fig://`` → mounted by
        SeamlessFigureFrame) shows it with NO renderer changes and guaranteed
        export/sidebar parity. Over the embed cap / empty / unavailable →
        None, and ``build_figure_window`` falls back to the anyplotlib snapshot.

        The fig_id is a fresh uuid per rebuild (there is no live anyplotlib
        figure to ``_electron.register``); a unique id per build is exactly what
        SeamlessFigureFrame keys its seamless swap on, and the explorer receives
        no state-push replay (self-contained), so an unregistered id is safe."""
        spec = cell.spec
        if spec is None:
            return None
        # Drop-time choice: "image" pins the static snapshot even when the tree
        # carries vectors. Default "" and "viewer" both embed the explorer
        # (mirrors export_html._render_body's `!= "image"` gate exactly).
        if str(getattr(spec, "vectors_mode", "") or "") == "image":
            return None
        # A scene3d cell is never a vectors-explorer candidate.
        if _is_scene3d_cell(cell):
            return None
        try:
            from spyde.actions.report.vectors_embed import (
                vectors_explorer_html, vectors_for_cell,
            )
            vecs = vectors_for_cell(self.session, cell)
            if vecs is None:
                return None
            # cache_key=cell.id memoizes the built page per (cell, vectors
            # identity) so a rebuild for the same cell + same vectors reuses the
            # packed blob + serialized figure instead of re-encoding (fix #6).
            html = vectors_explorer_html(vecs, caption=cell.caption or "",
                                         cache_key=cell.id)
            if html is None:            # over the embed cap / empty dataset
                return None
        except Exception as e:
            log.debug("[report] sidebar vectors explorer build failed "
                      "(cell %s): %s", cell.id, e)
            return None
        import uuid
        return f"vx_{cell.id}_{uuid.uuid4().hex[:8]}", html

    def build_figure_window(self, cell: Cell) -> None:
        """Build (or rebuild) the live figure window for a figure cell and emit
        it through the bare-figure path with ``host:"report"`` + ``cell_id``.

        A VIEWER-vectors cell (source tree carries diffraction vectors and the
        drop chose the interactive viewer, or defaulted to it) is emitted as the
        LIVE 2-panel vectors explorer instead of the plain anyplotlib snapshot —
        the same page the interactive HTML export embeds (see
        :meth:`_vectors_explorer_for_cell`). Over the embed cap / no vectors it
        falls through to the anyplotlib figure below."""
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
        # VIEWER-vectors cell → host the interactive explorer (Approach A). Not in
        # EDIT mode (the annotation editor targets the anyplotlib figure, so a cell
        # being edited falls back to the plain snapshot figure it can annotate).
        if cell.id not in self._editing:
            explorer = self._vectors_explorer_for_cell(cell)
            if explorer is not None:
                self._emit_vectors_explorer(cell, explorer)
                return
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
        # Fresh-slice callout markers (edit mode): pointer_up → re-slice the
        # inset at the dropped nav position. Appended AFTER _wire_annotation_drag
        # (which replaces the keep-alive list wholesale) so both survive.
        callout_wiring = list(getattr(fig, "_report_callout_wiring", None) or [])
        if callout_wiring:
            self._wire_callout_drag(cell.id, callout_wiring)
        # Zoom-region callout rectangles (edit mode): pointer_up → re-crop the
        # base panel's OWN snapshot at the dragged/resized rect. Appended to the
        # same _edit_wiring keep-alive list (after callout drag, same reasoning).
        zoom_wiring = list(getattr(fig, "_report_zoom_wiring", None) or [])
        if zoom_wiring:
            self._wire_zoom_region_drag(cell.id, zoom_wiring)
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

    def _emit_vectors_explorer(self, cell: Cell, explorer: "tuple[str, str]") -> None:
        """Emit a VIEWER-vectors cell's self-contained explorer as its figure
        iframe through the SAME ``host:"report"`` figure path
        (``build_figure_window`` picked it — see :meth:`_vectors_explorer_for_cell`).

        Unlike the anyplotlib branch there is NO live backend figure to
        ``keep_alive`` and NO ``ReportFigureController`` to register — the explorer
        is fully client-side (self-contained page; disks splatted in the iframe),
        so nothing on the backend drives it. We still allocate a window id and map
        it into ``_window_by_cell`` so a later rebuild / refresh / cell-close tears
        the cell's window down through the normal ``_forget`` path. Edit-mode
        annotation wiring is cleared (a vectors cell isn't annotation-edited)."""
        fig_id, html = explorer
        self._edit_wiring.pop(cell.id, None)
        self._ann_widgets.pop(cell.id, None)
        wid = self.session.next_window_id()
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

    def _wire_callout_drag(self, cell_id: str, wiring: list) -> None:
        """Attach a ``pointer_up`` re-slice handler to each fresh-slice callout
        marker widget in *wiring* and APPEND the (widget, handler) pairs to this
        cell's keep-alive list (``_wire_annotation_drag`` has already replaced
        it wholesale for this rebuild, so appending here never mixes stale
        builds)."""
        stored = self._edit_wiring.get(cell_id) or []
        for (widget, panel_id, inset_index, panel_spec) in wiring:
            handler = _make_callout_drag_handler(self, cell_id, panel_spec,
                                                 inset_index)
            try:
                widget.add_event_handler(handler, "pointer_up")
            except Exception as e:
                log.debug("wiring callout drag handler failed: %s", e)
                continue
            stored.append((widget, handler))
        if stored:
            self._edit_wiring[cell_id] = stored

    def _wire_zoom_region_drag(self, cell_id: str, wiring: list) -> None:
        """Attach a ``pointer_up`` re-crop handler to each zoom-region
        rectangle widget in *wiring* (``[(widget, base_panel_id,
        inset_index)]`` — see ``figure_builder._apply_insets``) and APPEND the
        (widget, handler) pairs to this cell's keep-alive list (mirrors
        ``_wire_callout_drag``; runs after it so both survive a rebuild)."""
        stored = self._edit_wiring.get(cell_id) or []
        for (widget, base_panel_id, inset_index) in wiring:
            handler = _make_zoom_region_drag_handler(
                self, cell_id, base_panel_id, inset_index)
            try:
                widget.add_event_handler(handler, "pointer_up")
            except Exception as e:
                log.debug("wiring zoom region drag handler failed: %s", e)
                continue
            stored.append((widget, handler))
        if stored:
            self._edit_wiring[cell_id] = stored

    def _wire_selection_handlers(self, cell_id: str, fig) -> None:
        """Wire the EDIT-MODE selection handlers onto *fig* (interactive builds
        only). Registers, per panel base plot, a module-level-closure
        ``pointer_down`` handler → ``select_panel(cell_id, spec_panel_id)``; a
        FIGURE-level ``pointer_down``/``pointer_up`` handler that routes a
        ``figure_background`` click → deselect (figure-level) and a
        ``figure_marker`` drop → persist the moved marker into ``spec.annotations``;
        and a FIGURE-level ``inset_geometry_change`` handler that persists a
        dragged/resized inset's anchor + w_frac/h_frac (see
        :func:`_make_inset_geometry_handler` — anyplotlib has already applied the
        geometry to its own InsetAxes, so this only writes the spec back).

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

        # One figure-level handler for inset drag/resize persistence.
        inset_geom_handler = _make_inset_geometry_handler(self, cell_id)
        try:
            fig.add_event_handler(inset_geom_handler, "inset_geometry_change")
            stored.append((fig, inset_geom_handler))
        except Exception as e:
            log.debug("wiring inset-geometry handler failed: %s", e)

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
        """Build ``{cell_id -> bytes}`` for every non-placeholder figure cell AND
        every image (photo) cell. For a figure it prefers (in order): the
        renderer-harvested PNG, a fresh bake of the held snapshot, then the PNG
        loaded from an opened report. For an image cell it is simply the held raw
        image bytes. The shared basis for the zip save AND every export path so all
        writes get identical pixels."""
        assets: dict[str, bytes] = {}
        for c in self.doc.cells:
            # Image (photo) cells: the held raw bytes go straight to the asset dict.
            if c.cell_type == "image":
                data = self._images.get(c.id)
                if data:
                    assets[c.id] = data
                continue
            if c.cell_type != "figure" or c.placeholder:
                continue
            png = harvested.get(c.id)
            # Treat an EMPTY harvest (b"" from a "data:image/png;base64," data URL
            # with no payload) as missing, so we still bake / fall back — otherwise
            # write_report's ``if png:`` guard would silently skip the asset, leaving
            # a dangling image ref in report.md.
            if not png:
                # The Agg bake cannot render a 3-D scene: primary_snapshot skips
                # the scene3d point-cloud arrays, so a scene3d cell falls straight
                # through to the last harvested/loaded PNG (below) — or is skipped
                # gracefully (no asset) rather than baking garbage / crashing.
                arr = self.primary_snapshot(c.id)
                if arr is not None and c.spec is not None:
                    try:
                        png = _bake_primary_snapshot(c, arr, max_edge=1200)
                    except Exception as e:
                        log.debug("asset bake failed for cell %s: %s", c.id, e)
                if not png:
                    png = self._baked.get(c.id)
            elif _is_scene3d_cell(c):
                # Keep the renderer-harvested 3-D pixels as this cell's baked
                # fallback: a later HEADLESS save (no renderer reply) has no way
                # to re-render the scene, so "the last harvested PNG" is the
                # best truthful asset available.
                self._baked[c.id] = png
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


def _make_callout_drag_handler(mgr, cell_id, panel_spec, inset_index):
    """A module-level closure factory (NOT a bound method — anyplotlib sets
    ``fn._event_types`` on the handler) returning a ``pointer_up`` handler for
    one fresh-slice callout MARKER widget on the base (navigator) panel.

    On drop: the widget's ``_data`` carries the final dragged center in IMAGE
    PIXELS, which on a navigator image ARE nav indices (index == pixel, so no
    axes inversion is needed — anyplotlib 2-D widgets are pixel-convention).
    Nearest index, clamped into the nav space; then the INSET panel's layer-0
    source is resolved and re-sliced at the new position
    (``slicing.read_frame_at`` — slice-then-compute, never the full dataset),
    the snapshot + ``nav_indices`` + connector region are updated, and the cell
    is REBUILT (the inset image must repaint) + state re-emitted.

    Guards (no-op, no crash): cell/panel/inset must still exist; the inset must
    carry ``nav_indices``; the source must resolve; an unchanged index skips
    the rebuild (no iframe flash — the marker merely snaps on the next one)."""
    def _on_marker_drop(event):
        try:
            from spyde.actions.report.compose import _callout_connector_region
            from spyde.actions.report.slicing import read_frame_at

            widget = getattr(event, "source", None)
            g = getattr(widget, "_data", None) if widget is not None else None
            if not isinstance(g, dict):
                return
            cell = mgr.doc.cell_by_id(cell_id) if mgr.doc else None
            if cell is None or cell.spec is None \
                    or panel_spec not in cell.spec.panels:
                return
            insets = panel_spec.insets or []
            if not (0 <= inset_index < len(insets)):
                return
            inset = insets[inset_index]
            if inset.get("nav_indices") is None:
                return
            inset_panel = next((p for p in cell.spec.panels
                                if p.id == inset.get("panel")), None)
            if inset_panel is None or not inset_panel.layers:
                return
            layer = inset_panel.layers[0]
            src_plot = layer.source.resolve(mgr.session) if layer.source else None
            if src_plot is None:
                return
            try:
                am = src_plot.plot_state.current_signal.axes_manager
                nav_shape = tuple(int(n) for n in am.navigation_shape)
            except Exception:
                return
            if len(nav_shape) != 2:
                return
            ix = int(np.clip(round(float(g.get("cx", 0.0))), 0, nav_shape[0] - 1))
            iy = int(np.clip(round(float(g.get("cy", 0.0))), 0, nav_shape[1] - 1))
            if [ix, iy] == [int(v) for v in inset["nav_indices"]]:
                return
            frame = read_frame_at(src_plot, [ix, iy])
            if frame is None:
                return
            inset["nav_indices"] = [ix, iy]
            if inset.get("connector") is not None:
                inset["connector"] = _callout_connector_region(panel_spec, ix, iy)
            mgr.set_snapshot(cell_id, inset_panel.id, layer.id, frame)
            mgr.dirty = True
            mgr._offline.discard(cell_id)
            mgr.build_figure_window(cell)
            mgr.emit_state()
        except Exception as e:
            log.debug("callout marker drop failed (cell %s inset %s): %s",
                      cell_id, inset_index, e)
    return _on_marker_drop


def _make_zoom_region_drag_handler(mgr, cell_id, base_panel_id, inset_index):
    """A module-level closure factory (NOT a bound method — anyplotlib sets
    ``fn._event_types`` on the handler) returning a ``pointer_up`` handler for
    one zoom-region callout RECTANGLE widget on the base panel.

    On drop: the widget's ``_data`` carries the final dragged/resized rect in
    IMAGE PIXELS (``x, y, w, h`` — top-left + size, the anyplotlib rect-widget
    convention). Clamp to the base image bounds, convert px → the base panel's
    DATA coords (``coords.data_region_to_index``'s inverse —
    ``compose._index_region_to_data``, the SAME conversion the drop-time
    zoom-callout add used), and if the region is UNCHANGED skip everything (no
    re-crop, no rebuild — the rect merely snaps back on the next repaint).
    Otherwise: update the inset dict's ``zoom_region`` + ``connector.region``,
    re-crop the BASE panel's OWN held snapshot from ``mgr._snapshots`` (never
    any dataset — this is a pixel crop of an already-in-memory array, so it
    can't violate the memory-safety rule), store the crop as the inset panel's
    layer-0 snapshot, and REBUILD (the inset image must repaint) + re-emit.

    Guards (no-op, no crash): cell/base panel/inset must still exist; the
    inset must carry ``zoom_region``; the base snapshot must still resolve."""
    def _on_rect_drop(event):
        try:
            from spyde.actions.report.compose import (
                _crop_region_px, _index_region_to_data,
            )

            widget = getattr(event, "source", None)
            g = getattr(widget, "_data", None) if widget is not None else None
            if not isinstance(g, dict):
                return
            cell = mgr.doc.cell_by_id(cell_id) if mgr.doc else None
            if cell is None or cell.spec is None:
                return
            base_panel = next((p for p in cell.spec.panels
                               if p.id == base_panel_id), None)
            if base_panel is None or not base_panel.layers:
                return
            insets = base_panel.insets or []
            if not (0 <= inset_index < len(insets)):
                return
            inset = insets[inset_index]
            if inset.get("zoom_region") is None:
                return
            base_layer = base_panel.layers[0]
            base_arr = mgr.snapshot_map(cell_id).get(
                (base_panel.id, base_layer.id))
            if base_arr is None:
                return
            base_arr = np.asarray(base_arr)
            H, W = base_arr.shape[0], base_arr.shape[1]
            px = float(g.get("x", 0.0))
            py = float(g.get("y", 0.0))
            pw = max(1.0, float(g.get("w", 1.0)))
            ph = max(1.0, float(g.get("h", 1.0)))
            pw = min(pw, W)
            ph = min(ph, H)
            px = max(0.0, min(px, W - pw))
            py = max(0.0, min(py, H - ph))

            new_region = list(_index_region_to_data(
                base_panel, (px, py, pw, ph)))
            old_region = [float(v) for v in (inset.get("zoom_region") or [])]
            if len(old_region) == 4 and all(
                    abs(a - b) < 1e-6 for a, b in zip(old_region, new_region)):
                return

            inset_panel = next((p for p in cell.spec.panels
                                if p.id == inset.get("panel")), None)
            if inset_panel is None or not inset_panel.layers:
                return
            crop = _crop_region_px(base_arr, px, py, pw, ph)
            inset["zoom_region"] = new_region
            if inset.get("connector") is not None:
                inset["connector"] = {"region": list(new_region)}
            layer = inset_panel.layers[0]
            mgr.set_snapshot(cell_id, inset_panel.id, layer.id, crop)
            mgr.dirty = True
            mgr._offline.discard(cell_id)
            mgr.build_figure_window(cell)
            mgr.emit_state()
        except Exception as e:
            log.debug("zoom region drop failed (cell %s panel %s inset %s): %s",
                      cell_id, base_panel_id, inset_index, e)
    return _on_rect_drop


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


def _make_inset_geometry_handler(mgr, cell_id):
    """A module-level closure (NOT a bound method — anyplotlib sets
    ``fn._event_types`` on it) returning a FIGURE-level ``inset_geometry_change``
    handler that persists a dragged/resized inset's geometry.

    anyplotlib has ALREADY applied the geometry to its own ``InsetAxes`` before
    firing this event (see ``Figure._dispatch_event``'s ``inset_geometry_change``
    branch, which calls ``inset_ax.set_geometry(...)`` before
    ``_fire_figure_event``) — this handler ONLY persists the same values into
    the spec so a save/rebuild reproduces the position, and does NOT rebuild
    the figure (the inset already moved JS-side; a rebuild would flash it).

    ``event.inset_id`` is the moved inset's anyplotlib Plot2D dispatch id;
    resolve it to the SPEC inset-panel id via the inverted
    ``fig._report_inset_map`` (built by ``figure_builder._apply_insets``), then
    find the grid panel whose ``insets`` entry references that spec panel id
    (``inset["panel"] == spec_pid``) and rewrite its ``anchor``/``w_frac``/
    ``h_frac`` in place, dropping ``corner`` (an explicit ``anchor`` wins over
    ``corner`` at render time — see ``_apply_insets`` — so a stale ``corner``
    would be dead weight, not a conflicting value).

    Guards (no-op, no crash; every failure just returns after a debug log):
    the cell must still exist and be a figure cell; ``inset_id`` must resolve
    through ``_report_inset_map``; a grid panel with a matching inset entry
    must still exist on the spec."""
    def _on_inset_geometry(event):
        try:
            fig = getattr(event, "source", None)
            inset_disp_id = getattr(event, "inset_id", None)
            if inset_disp_id is None or fig is None:
                return
            cell = mgr.doc.cell_by_id(cell_id) if mgr.doc else None
            if cell is None or cell.cell_type != "figure" or cell.spec is None:
                return
            inset_map = dict(getattr(fig, "_report_inset_map", None) or {})
            spec_by_disp = {disp: pid for pid, disp in inset_map.items()}
            spec_pid = spec_by_disp.get(inset_disp_id)
            if spec_pid is None:
                log.debug("inset geometry event for unknown inset id %s "
                          "(cell %s)", inset_disp_id, cell_id)
                return
            owner = None
            for p in cell.spec.panels:
                for ins in (p.insets or []):
                    if ins.get("panel") == spec_pid:
                        owner = ins
                        break
                if owner is not None:
                    break
            if owner is None:
                log.debug("inset geometry event: no inset entry references "
                          "panel %s (cell %s)", spec_pid, cell_id)
                return
            anchor = getattr(event, "anchor", None)
            w_frac = getattr(event, "w_frac", None)
            h_frac = getattr(event, "h_frac", None)
            if anchor is not None:
                try:
                    owner["anchor"] = [float(anchor[0]), float(anchor[1])]
                except (TypeError, ValueError, IndexError):
                    pass
            if w_frac is not None:
                try:
                    owner["w_frac"] = float(w_frac)
                except (TypeError, ValueError):
                    pass
            if h_frac is not None:
                try:
                    owner["h_frac"] = float(h_frac)
                except (TypeError, ValueError):
                    pass
            owner.pop("corner", None)
            mgr.dirty = True
            mgr.emit_state()
        except Exception as e:
            log.debug("inset geometry persist failed (cell %s): %s",
                      cell_id, e)
    return _on_inset_geometry


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


def _clear_vectors_explorer_cache(cell_id: "str | None" = None) -> None:
    """Drop the memoized vectors-explorer page(s) (fix #6). Best-effort; a lazy
    import so vectors_embed (which pulls anyplotlib) isn't loaded at handler
    import time. ``cell_id`` clears one cell; ``None`` clears all."""
    try:
        from spyde.actions.report.vectors_embed import clear_explorer_cache
        clear_explorer_cache(cell_id)
    except Exception as e:
        log.debug("clear vectors explorer cache failed: %s", e)


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


def _snapshot_line_state(plot) -> dict:
    """Read the live 1-D anyplotlib state (``line_color``/``line_linewidth``/
    ``line_label``) off ``plot._plot1d`` for the base-layer styling snapshot.
    Every key is independently tolerant — a plot with no ``_plot1d`` (e.g. a
    test that stamps ``current_data`` directly without going through the real
    paint pipeline) simply yields an empty dict, and the caller's LayerSpec
    fields stay ``None`` (figure_builder falls back to anyplotlib's own
    defaults)."""
    out: dict = {}
    p1 = getattr(plot, "_plot1d", None)
    state = getattr(p1, "_state", None) if p1 is not None else None
    if not isinstance(state, dict):
        return out
    try:
        if state.get("line_color") is not None:
            out["color"] = str(state["line_color"])
    except Exception:
        pass
    try:
        if state.get("line_linewidth") is not None:
            out["linewidth"] = float(state["line_linewidth"])
    except Exception:
        pass
    try:
        lbl = state.get("line_label")
        if lbl:
            out["label"] = str(lbl)
    except Exception:
        pass
    return out


def _snapshot_line_extras(plot) -> list:
    """Extra overlay curves from the live plot's ``extra_lines`` state, each as
    ``(ndarray, {color, linewidth, label})``. Best-effort: an entry whose
    y-data isn't cleanly readable as a 1-D numpy array is skipped silently (a
    base-curve-only snapshot is acceptable v1) rather than failing the whole
    snapshot."""
    out: list = []
    p1 = getattr(plot, "_plot1d", None)
    state = getattr(p1, "_state", None) if p1 is not None else None
    if not isinstance(state, dict):
        return out
    for entry in list(state.get("extra_lines", None) or []):
        try:
            y = np.asarray(entry.get("data"), dtype=float)
            if y.ndim != 1 or y.size == 0:
                continue
        except Exception:
            continue
        style: dict = {}
        try:
            if entry.get("color") is not None:
                style["color"] = str(entry["color"])
        except Exception:
            pass
        try:
            if entry.get("linewidth") is not None:
                style["linewidth"] = float(entry["linewidth"])
        except Exception:
            pass
        try:
            if entry.get("label"):
                style["label"] = str(entry["label"])
        except Exception:
            pass
        out.append((np.array(y, copy=True), style))
    return out


def _snapshot_plot(plot) -> "tuple[FigureSpec, dict] | None":
    """Snapshot a live ``Plot`` NOW into a single-panel FigureSpec + a
    ``{(panel_id, layer_id): ndarray}`` snapshot map. Reads ``current_data``,
    ``_last_levels``, colormap, axes, title, nav indices, and the view label.

    A 1-D ``current_data`` (a line-profile / spectrum plot) takes the
    LINE-PANEL branch (``kind="line"``): axes carries ``x_axis`` + ``units``
    (from ``plot._axes_info_1d``, falling back to ``range(n)`` when
    unavailable — never crashing the snapshot), and the base LayerSpec's
    ``color``/``linewidth``/``label`` are read from the live anyplotlib 1-D
    state when reachable (see :func:`_snapshot_line_state`). Extra overlay
    curves (``plot._plot1d``'s ``extra_lines``) become extra LayerSpecs when
    their y-data is cleanly readable; MDI overlay-LAYER harvesting
    (``plot._layers``, the 2-D compositing path) does NOT apply to a 1-D plot
    and is skipped entirely.

    If the plot carries live MDI overlay layers (``plot._layers``, 2-D only),
    each is serialized into the same panel as an extra LayerSpec (same
    cmap / alpha, its own source ref) with its current frame in the snapshot
    map — so "Add to report" on a layered plot captures the whole composite.
    Returns None when the plot has no paintable base frame."""
    data = getattr(plot, "current_data", None)
    if not isinstance(data, np.ndarray) or data.dtype == object:
        return None
    arr = np.array(data, copy=True)   # detach from the live buffer

    if arr.ndim == 1:
        return _snapshot_line_plot(plot, arr)

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


def _snapshot_line_plot(plot, arr: np.ndarray) -> "tuple[FigureSpec, dict]":
    """The ``kind="line"`` branch of :func:`_snapshot_plot`: build a
    single-panel FigureSpec from a 1-D ``current_data`` array. Always
    succeeds (a 1-D array is always paintable) — unlike the 2-D branch there
    is no "no paintable frame" case to reject."""
    axes_dict = None
    try:
        xa, x_units, _y_label = plot._axes_info_1d(arr)
        if xa is not None:
            axes_dict = {"units": x_units,
                         "x_axis": [float(v) for v in np.asarray(xa)]}
    except Exception as e:
        log.debug("snapshot 1d axes read failed: %s", e)
    if axes_dict is None:
        # No calibrated axis reachable (or the length didn't match) — fall back
        # to a bare index axis so the panel still renders with real ticks.
        axes_dict = {"units": "px",
                     "x_axis": [float(i) for i in range(arr.shape[0])]}

    title = ""
    try:
        title = plot._plot_title()
    except Exception:
        title = ""

    nav_context = None
    try:
        sig = plot.plot_state.current_signal
        idx = tuple(int(i) for i in sig.axes_manager.indices)
        if idx:
            nav_context = {"indices": list(idx)}
    except Exception:
        nav_context = None

    style = _snapshot_line_state(plot)
    base_layer = LayerSpec(source=SignalRef.from_plot(plot), alpha=1.0,
                           visible=True, color=style.get("color"),
                           linewidth=style.get("linewidth"),
                           label=style.get("label"))
    layers = [base_layer]
    snap_map = {("p1", base_layer.id): arr}

    for extra_arr, extra_style in _snapshot_line_extras(plot):
        ov = LayerSpec(source=SignalRef.from_plot(plot), alpha=1.0,
                       visible=True, color=extra_style.get("color"),
                       linewidth=extra_style.get("linewidth"),
                       label=extra_style.get("label"))
        layers.append(ov)
        snap_map[("p1", ov.id)] = extra_arr

    panel = PanelSpec(id="p1", grid_pos=[0, 0], kind="line", layers=layers,
                      axes=axes_dict, title=title)
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


def _snapshot_scene3d(session, src_plot) -> "tuple[FigureSpec, dict] | None":
    """Snapshot a window's 3-D IPF view into a single scene3d-panel FigureSpec +
    a ``{(panel_id, "xyz"/"rgb"): ndarray}`` snapshot map.

    Recomputes the sphere points EXACTLY the way ``emit_ipf_3d`` does (the
    shared ``ipf_view.ipf_scene_data`` builder) from the orientation result on
    the plot's tree, at the tree's CURRENT direction (the X/Y/Z selector state,
    ``tree._ipf_direction``). Point size mirrors the live ``Plot3D`` when one is
    cached on the tree. The panel carries one LayerSpec whose SignalRef points
    at the tree — the rebind/refresh handle — but the pixels are the xyz/rgb
    arrays under the pseudo layer keys, never a LayerSpec image. Returns None
    when the tree has no orientation result (caller emits the no-image error)."""
    from spyde.actions.ipf_view import (
        IPF3D_POINT_SIZE, ipf_scene_data, tree_orientation_result,
    )

    tree = getattr(src_plot, "signal_tree", None)
    result = tree_orientation_result(tree)
    if result is None:
        return None
    direction = str(getattr(tree, "_ipf_direction", "z") or "z")
    data = ipf_scene_data(result, direction)
    if data is None:
        return None
    xyz, rgb, scene = data
    # Mirror the live explorer's point size when the tree holds the live Plot3D
    # (a user-tuned size would ride along); otherwise the shared default.
    try:
        p3d = getattr(tree, "_ipf_p3d", None)
        ps = (getattr(p3d, "_state", {}) or {}).get("point_size")
        scene["point_size"] = float(ps) if ps else float(IPF3D_POINT_SIZE)
    except Exception:
        scene["point_size"] = float(IPF3D_POINT_SIZE)

    ref_layer = LayerSpec(source=SignalRef.from_plot(src_plot))
    panel = PanelSpec(id="p1", grid_pos=[0, 0], kind="scene3d",
                      layers=[ref_layer], scene=scene)
    spec = FigureSpec(layout={"kind": "single"}, panels=[panel])
    snap_map = {("p1", "xyz"): np.asarray(xyz), ("p1", "rgb"): np.asarray(rgb)}
    return spec, snap_map


def _scene3d_snap_entries(session, panel: PanelSpec) -> "dict | None":
    """Recompute a scene3d panel's point cloud from its resolved source tree at
    the panel's STORED scene direction → ``{(panel_id, "xyz"/"rgb"): ndarray}``,
    or None (source offline / no orientation result). The one recompute path
    behind report_open rebind, paste rebind, and refresh_panel."""
    from spyde.actions.ipf_view import ipf_scene_data, tree_orientation_result

    if not panel.layers or panel.layers[0].source is None:
        return None
    src_plot = panel.layers[0].source.resolve(session)
    if src_plot is None:
        return None
    result = tree_orientation_result(getattr(src_plot, "signal_tree", None))
    if result is None:
        return None
    direction = str((panel.scene or {}).get("direction", "z") or "z")
    data = ipf_scene_data(result, direction)
    if data is None:
        return None
    xyz, rgb, _scene = data
    return {(panel.id, "xyz"): np.asarray(xyz),
            (panel.id, "rgb"): np.asarray(rgb)}


def _recompute_scene3d_panel(session, mgr: "ReportManager", cell: Cell,
                             panel: PanelSpec) -> bool:
    """Refresh a scene3d panel's xyz/rgb snapshots in place via
    :func:`_scene3d_snap_entries`. Returns True on success (False = source
    offline / no orientation result — caller decides offline vs no-op)."""
    entries = _scene3d_snap_entries(session, panel)
    if entries is None:
        return False
    for (pid, key), arr in entries.items():
        mgr.set_snapshot(cell.id, pid, key, arr)
    return True


def _stored_position_inset(spec, panel_id):
    """The inset dict referencing *panel_id* that carries a STORED slice
    position (``nav_indices`` / ``time_index``), or None. Such a panel is a
    FRESH-SLICE callout: refresh re-slices the dataset at the stored position
    instead of re-snapshotting whatever frame the live plot currently shows."""
    for p in spec.panels:
        for ins in (p.insets or []):
            if ins.get("panel") == panel_id and (
                    ins.get("nav_indices") is not None
                    or ins.get("time_index") is not None):
                return ins
    return None


def _refresh_callout_panel(session, mgr: "ReportManager", cell: Cell,
                           panel: PanelSpec, inset: dict) -> bool:
    """Refresh a fresh-slice callout panel: resolve its layer-0 source and
    RE-SLICE at the inset's stored position (never the live current frame).
    cmap/clim are left alone — the stored-position slice isn't the live view,
    so the live plot's display state doesn't apply. Returns True on success."""
    from spyde.actions.report.slicing import read_frame_at

    if not panel.layers:
        return False
    layer = panel.layers[0]
    src_plot = layer.source.resolve(session) if layer.source else None
    if src_plot is None:
        return False
    indices = inset.get("nav_indices")
    if indices is None:
        indices = [inset.get("time_index")]
    frame = read_frame_at(src_plot, indices)
    if frame is None:
        return False
    mgr.set_snapshot(cell.id, panel.id, layer.id, frame)
    return True


def _stored_zoom_inset(spec, panel_id):
    """``(base_panel, inset)`` for the inset referencing *panel_id* that
    carries a ``zoom_region`` (a ZOOM-REGION callout — a magnified crop of its
    BASE panel's own pixels, never a dataset re-slice), or ``(None, None)``."""
    for p in spec.panels:
        for ins in (p.insets or []):
            if ins.get("panel") == panel_id and ins.get("zoom_region") is not None:
                return p, ins
    return None, None


def _refresh_zoom_panel(mgr: "ReportManager", cell: Cell, panel: PanelSpec,
                        base_panel: PanelSpec, inset: dict) -> bool:
    """Refresh a zoom-region callout panel: re-crop the (possibly just-
    refreshed) BASE panel's OWN held snapshot at ``inset["zoom_region"]``
    (DATA coords → pixel index via ``coords.data_region_to_index``) — never a
    dataset re-slice, never the live plot. Returns True on success (the base
    panel's snapshot must already be resolvable; False leaves the zoom
    panel's existing snapshot untouched)."""
    from spyde.actions.report import coords
    from spyde.actions.report.compose import _crop_region_px

    if not panel.layers or not base_panel.layers:
        return False
    base_layer = base_panel.layers[0]
    base_arr = mgr.snapshot_map(cell.id).get((base_panel.id, base_layer.id))
    if base_arr is None:
        return False
    base_arr = np.asarray(base_arr)
    region = inset.get("zoom_region")
    try:
        x, y, w, h = coords.data_region_to_index(region, base_panel.axes)
    except Exception as e:
        log.debug("zoom panel refresh region convert failed: %s", e)
        return False
    crop = _crop_region_px(base_arr, x, y, w, h)
    layer = panel.layers[0]
    mgr.set_snapshot(cell.id, panel.id, layer.id, crop)
    return True


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
    or "leave silently as-is" — ``repfig_refresh_panel``).

    A LINE panel's layers refresh through this same generic path:
    ``_snapshot_layer_now`` places no ndim gate on ``current_data``, so a 1-D
    array flows through unchanged (cmap/clim are updated but stay unused —
    ``figure_builder`` ignores them for ``kind="line"``); the curve's
    color/linewidth/label styling is left as-is (a refresh pulls fresh DATA,
    not a re-read of the live line style).

    A FRESH-SLICE callout panel (referenced by an inset carrying
    ``nav_indices``/``time_index``) takes the re-slice path instead: its pixels
    come from the dataset at the STORED position, not the live current frame.
    A ZOOM-REGION callout panel (referenced by an inset carrying
    ``zoom_region``) takes the re-crop path instead: its pixels come from
    cropping its BASE panel's OWN (already-refreshed, if the base panel
    precedes it in ``cell.spec.panels`` — the normal creation order) held
    snapshot, never a dataset re-slice or the live plot. A SCENE3D panel
    likewise recomputes its point cloud from the resolved orientation result
    at the STORED scene direction (never image layers)."""
    if not panel.layers:
        return False
    if _is_scene3d_panel(panel):
        return _recompute_scene3d_panel(session, mgr, cell, panel)
    if cell.spec is not None:
        ins = _stored_position_inset(cell.spec, panel.id)
        if ins is not None:
            return _refresh_callout_panel(session, mgr, cell, panel, ins)
        base_panel, zoom_ins = _stored_zoom_inset(cell.spec, panel.id)
        if zoom_ins is not None:
            return _refresh_zoom_panel(mgr, cell, panel, base_panel, zoom_ins)
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
    # Image (photo) cells re-hydrate their raw bytes from the same assets dict
    # (read_report returns image bytes keyed by cell id). Held so state() can emit
    # the data URL and a re-save round-trips them.
    mgr._images = {c.id: assets[c.id] for c in doc.cells
                   if c.cell_type == "image" and c.id in assets}
    mgr._offline.clear()
    # Rebind each figure cell: resolve EVERY layer of EVERY panel against open trees
    # / files. The cell rebinds live only when ALL its layers resolve; if any layer's
    # source is offline the whole cell is offline (renderer shows the baked PNG).
    for c in doc.cells:
        if c.cell_type != "figure" or c.placeholder or c.spec is None:
            continue
        all_resolved = bool(c.spec.panels)
        for panel in c.spec.panels:
            # A scene3d panel rebinds by RECOMPUTING its point cloud from the
            # resolved orientation result (there is no image layer to read);
            # an unresolvable source / missing result → the whole cell offline
            # (baked PNG badge), same rule as image layers.
            if _is_scene3d_panel(panel):
                if not _recompute_scene3d_panel(session, mgr, c, panel):
                    all_resolved = False
                continue
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
    # OPTIONAL Present-mode fields (Phase 6) so a seeded deck (report_from_guide)
    # can create each cell already-marked without a follow-up round trip.
    if payload.get("slide_break") is not None:
        cell.slide_break = bool(payload.get("slide_break"))
    la = payload.get("live_action")
    if isinstance(la, dict) and la:
        cell.live_action = dict(la)
    if payload.get("column") is not None:
        from spyde.actions.report.model import _normalize_column
        cell.column = _normalize_column(payload.get("column"))
    # Per-slide kind/style (presentation polish) so a seeded deck can create a
    # title / styled slide's first cell already-marked without a round trip.
    if payload.get("slide_kind") is not None:
        from spyde.actions.report.model import _normalize_slide_kind
        cell.slide_kind = _normalize_slide_kind(payload.get("slide_kind"))
    if payload.get("slide_style") is not None:
        from spyde.actions.report.model import _normalize_slide_style
        cell.slide_style = _normalize_slide_style(payload.get("slide_style"))
    # Speaker notes (presenter view) — a seeded deck can create a slide's first
    # cell already carrying notes without a follow-up round trip.
    if payload.get("notes") is not None:
        cell.notes = str(payload.get("notes") or "")
    _insert_cell(mgr.doc, cell, payload.get("index"))
    mgr.dirty = True
    mgr.emit_state()


def report_add_image_cell(session, plot, payload) -> None:
    """Add an IMAGE (photo) cell from a base64-encoded image — a file dropped,
    a clipboard paste, or a browse. Decodes the bytes, size-caps them, stores them
    on the manager keyed by a fresh cell id, inserts an ``image`` cell at ``index``,
    and re-emits state (the bytes ride back as a data URL).

    ``payload``: ``{image_b64, image_ext, caption?, index?, slide_break?}``. The
    ext is normalised to one of :data:`IMAGE_EXTS` (unknown → png), and the bytes
    are refused over :data:`_IMAGE_CELL_MAX_BYTES` so a giant photo can't bloat the
    report."""
    mgr = _ensure_open(session)
    raw = payload.get("image_b64")
    data = _decode_data_url(raw) if raw else None
    if not data:
        ipc.emit_error("report_add_image_cell: no / undecodable image data.")
        return
    if len(data) > _IMAGE_CELL_MAX_BYTES:
        mb = _IMAGE_CELL_MAX_BYTES / (1024 * 1024)
        ipc.emit_error(
            f"Image is too large ({len(data) / (1024 * 1024):.1f} MB) — the limit "
            f"is {mb:.0f} MB. Resize it and try again.")
        return
    ext = str(payload.get("image_ext", "") or "").lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in IMAGE_EXTS:
        ext = "png"
    cell = Cell(id=new_cell_id(), cell_type="image",
                caption=str(payload.get("caption", "") or ""), image_ext=ext)
    if payload.get("slide_break") is not None:
        cell.slide_break = bool(payload.get("slide_break"))
    mgr._images[cell.id] = data
    _insert_cell(mgr.doc, cell, payload.get("index"))
    mgr.dirty = True
    mgr.emit_state()


def report_update_cell(session, plot, payload) -> None:
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None:
        return
    # An IMAGE (photo) cell edits only its caption through this path (its bytes are
    # immutable once added). Mirror report_set_caption so either wire works.
    if cell.cell_type == "image":
        if "caption" in payload:
            cell.caption = str(payload.get("caption", "") or "")
            mgr.dirty = True
            mgr.emit_state()
        return
    if cell.cell_type != "markdown":
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
        _clear_vectors_explorer_cache(cell.id)
    elif cell.cell_type == "image":
        mgr._images.pop(cell.id, None)
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


def report_toggle_slide_break(session, plot, payload) -> None:
    """Toggle (or set) a cell's ``slide_break`` flag — Present mode / the slides
    export group cells into slides by it (a cell with ``slide_break=True`` STARTS
    a new slide).

    ``{cell_id}`` alone TOGGLES; an explicit ``{cell_id, value: bool}`` sets it.
    An unknown cell is a no-op (no crash)."""
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None:
        return
    if "value" in payload:
        cell.slide_break = bool(payload.get("value"))
    else:
        cell.slide_break = not bool(getattr(cell, "slide_break", False))
    mgr.dirty = True
    mgr.emit_state()


def report_set_live_action(session, plot, payload) -> None:
    """Set (or clear) a cell's ``live_action`` — the optional "go live" excursion
    Present mode surfaces as a "Launch live ▶" button.

    ``{cell_id, live_action: {tutorial?, guide?}}`` sets it; a null / empty
    ``live_action`` clears it. An unknown cell is a no-op."""
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None:
        return
    la = payload.get("live_action")
    cell.live_action = dict(la) if isinstance(la, dict) and la else None
    mgr.dirty = True
    mgr.emit_state()


def report_set_cell_column(session, plot, payload) -> None:
    """Assign a cell to a COLUMN within its slide — the 2-column layout so a text
    cell can sit BESIDE a figure/photo (see :func:`spyde.actions.report.model.slide_columns`).

    ``{cell_id, column}`` where ``column`` is ``"left"`` / ``"right"`` (place in
    the 2-col grid) or ``""`` / ``"full"`` (span the whole slide — the default).
    Any other value collapses to ``""`` (full width). An unknown cell is a no-op
    (no crash)."""
    from spyde.actions.report.model import _normalize_column
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None:
        return
    cell.column = _normalize_column(payload.get("column"))
    mgr.dirty = True
    mgr.emit_state()


def _slide_start_cell(mgr, cell):
    """The FIRST cell of the slide *cell* belongs to — walk BACK from *cell* until
    a slide_break cell (or the document start). Per-slide attributes (kind/style)
    live on the slide's first cell, so an authoring toggle fired on ANY cell of a
    slide is applied there. A missing cell → None."""
    cells = mgr.doc.cells
    idx = mgr.doc.index_of(cell.id)
    if idx < 0:
        return None
    j = idx
    while j > 0 and not bool(getattr(cells[j], "slide_break", False)):
        j -= 1
    return cells[j]


def report_set_slide_kind(session, plot, payload) -> None:
    """Set a SLIDE's ``slide_kind`` — ``"title"`` makes the whole slide a
    TITLE / SECTION slide (big centered title block in Present mode + the slides
    export); ``""`` / ``"content"`` is a normal slide.

    Applied to the slide's FIRST cell (the slide-break cell) even when
    ``cell_id`` names a later cell of the slide — the per-slide attribute lives
    there (:func:`_slide_start_cell`). ``{cell_id}`` alone TOGGLES title↔content;
    an explicit ``{cell_id, slide_kind}`` sets it. Unknown value → ""; unknown
    cell → no-op."""
    from spyde.actions.report.model import _normalize_slide_kind
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None:
        return
    target = _slide_start_cell(mgr, cell) or cell
    if "slide_kind" in payload:
        target.slide_kind = _normalize_slide_kind(payload.get("slide_kind"))
    else:
        cur = _normalize_slide_kind(getattr(target, "slide_kind", ""))
        target.slide_kind = "" if cur == "title" else "title"
    mgr.dirty = True
    mgr.emit_state()


def report_set_slide_style(session, plot, payload) -> None:
    """Set a SLIDE's ``slide_style`` background/heading preset — ``""`` /
    ``"default"`` the standard dark stage, ``"plain"`` a flat darker stage,
    ``"accent"`` a subtle accent-tinted gradient.

    Applied to the slide's FIRST cell like :func:`report_set_slide_kind`.
    ``{cell_id, slide_style}`` sets it; unknown value → "" (default). Unknown
    cell → no-op."""
    from spyde.actions.report.model import _normalize_slide_style
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None:
        return
    target = _slide_start_cell(mgr, cell) or cell
    target.slide_style = _normalize_slide_style(payload.get("slide_style"))
    mgr.dirty = True
    mgr.emit_state()


def report_set_slide_notes(session, plot, payload) -> None:
    """Set a SLIDE's SPEAKER NOTES — free multi-line markdown text the presenter
    sees in the presenter view but the audience never does.

    Applied to the slide's FIRST cell (the slide-break cell) even when ``cell_id``
    names a later cell of the slide — the per-slide attribute lives there
    (:func:`_slide_start_cell`, like :func:`report_set_slide_kind`). ``{cell_id,
    notes}`` sets it (any string; empty clears); a missing ``notes`` clears.
    Unknown cell → no-op."""
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None:
        return
    target = _slide_start_cell(mgr, cell) or cell
    target.notes = str(payload.get("notes", "") or "")
    mgr.dirty = True
    mgr.emit_state()


def report_set_caption(session, plot, payload) -> None:
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None or cell.cell_type not in ("figure", "image"):
        return
    cell.caption = str(payload.get("caption", "") or "")
    # A vectors-explorer page bakes the caption in; drop its memoized page so the
    # next rebuild picks up the new caption (fix #6 cache correctness).
    _clear_vectors_explorer_cache(cell.id)
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
    an existing placeholder in place; otherwise a new figure cell is inserted.

    A source whose tree carries diffraction vectors can export either as the
    static snapshot or as the interactive vectors explorer. When the payload
    has no ``vectors_mode`` yet, the drop is deferred: a
    ``report_vectors_choice`` message asks the renderer, which re-sends this
    action with ``vectors_mode`` ("viewer" | "image") once the user picks.

    ``view: "3d"`` (the pill was dragged while the window showed its 3-D IPF
    explorer) snapshots the SCENE instead of the 2-D image: a single scene3d
    panel whose point cloud is recomputed from the tree's orientation result
    (``_snapshot_scene3d``). This branch runs FIRST — a 3-D drop is never a
    vectors-explorer candidate."""
    mgr = _ensure_open(session)
    src = _resolve_source_plot(session, payload.get("source_window_id"))
    if src is None:
        ipc.emit_error("report_add_figure: source window not found.")
        return
    vectors_mode = str(payload.get("vectors_mode", "") or "")
    if str(payload.get("view", "") or "") == "3d":
        snap = _snapshot_scene3d(session, src)
        if snap is None:
            ipc.emit_error("report_add_figure: source window has no 3-D "
                           "orientation view to snapshot.")
            return
        vectors_mode = ""              # a scene cell never embeds the explorer
    else:
        if not vectors_mode:
            vecs = getattr(getattr(src, "signal_tree", None),
                           "diffraction_vectors", None)
            if vecs is not None:
                try:
                    count = int(len(vecs.flat_buffer))
                except Exception:
                    count = 0
                ipc.emit({
                    "type": "report_vectors_choice",
                    "source_window_id": payload.get("source_window_id"),
                    "index": payload.get("index"),
                    "at_cell": payload.get("at_cell"),
                    "caption": str(payload.get("caption", "") or ""),
                    "count": count,
                })
                return
        snap = _snapshot_plot(src)
        if snap is None:
            ipc.emit_error("report_add_figure: source window has no image to snapshot.")
            return
    spec, snap_map = snap
    spec.vectors_mode = vectors_mode
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
    # Explicit refresh → always rebuild the explorer page from fresh vectors.
    _clear_vectors_explorer_cache(cell.id)
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
    # Explicit refresh → always rebuild the explorer page from fresh vectors.
    _clear_vectors_explorer_cache(cell.id)
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

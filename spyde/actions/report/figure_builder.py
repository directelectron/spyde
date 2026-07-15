"""
figure_builder.py — build a live anyplotlib figure for a report figure cell.

A report figure cell owns a :class:`~spyde.actions.report.model.FigureSpec`
(recipe) + an in-memory snapshot map ``{(panel_id, layer_id): ndarray}``. This
module renders that into a bare anyplotlib figure and returns its
``(fig, fig_id, html)`` so the handler can emit it through the normal bare-figure
path (``finalize_figure_html``), plus a :class:`ReportFigureController`
implementing the WindowController protocol so the figure is reachable by dispatch
and torn down by ``Session._forget_window``.

Phase 2 consumes the FULL spec:

* ``layout={kind:single}`` → one panel; ``{kind:grid, rows, cols, width_ratios}``
  → an ``apl.subplots`` grid, panels placed by ``grid_pos``.
* per panel: base ``imshow`` (layer 0) + extra layers via ``plot2d.add_layer``
  (overlay), annotations mapped to ``add_texts/add_circles/add_ellipses/
  add_rectangles/add_arrows/add_lines``, and callout insets via
  ``fig.add_inset`` (+ ``inset.indicate_region`` when anyplotlib provides it).
* ``_resolve_pixels_for_standalone`` materialises the binary pixel tokens for the
  BASE image AND every layer so the standalone iframe is self-contained.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


# ── snapshot-map helpers ──────────────────────────────────────────────────────


def _as_snapshot_map(spec, snapshots):
    """Accept either a single ndarray (legacy single-panel/single-layer) or a
    ``{(panel_id, layer_id): ndarray}`` map, and return the map form keyed by
    ``(panel_id, layer_id)`` for uniform lookup."""
    if isinstance(snapshots, dict):
        return dict(snapshots)
    # Legacy: a bare array → the primary panel's primary layer.
    arr = np.asarray(snapshots)
    panel = spec.panels[0] if (spec is not None and spec.panels) else None
    layer = panel.layers[0] if (panel is not None and panel.layers) else None
    if panel is not None and layer is not None:
        return {(panel.id, layer.id): arr}
    return {}


def _resolve_cmap(name):
    from spyde.drawing.colormaps import COLORMAPS
    return COLORMAPS.get(name, name)


def _panel_axes_kw(panel):
    """(axes_kw, units) for calibrated ticks / scale bar from a panel's axes."""
    axes_kw = {}
    units = "px"
    if panel is not None and panel.axes:
        try:
            ax_units = panel.axes.get("units")
            xa = panel.axes.get("x_axis")
            ya = panel.axes.get("y_axis")
            if xa is not None and ya is not None:
                axes_kw["axes"] = [np.asarray(xa, dtype=float),
                                   np.asarray(ya, dtype=float)]
                units = str(ax_units or "px")
                axes_kw["units"] = units
        except Exception as e:
            log.debug("report figure axes from spec failed: %s", e)
    return axes_kw, units


# ── annotation rendering ──────────────────────────────────────────────────────


def _first_offset(offsets):
    """The first ``(x, y)`` of an annotation's ``offsets`` — accepts a flat
    ``[x, y]`` OR an ``(N, 2)`` list; returns None if unusable."""
    if offsets is None:
        return None
    arr = np.asarray(offsets, dtype=float)
    if arr.ndim == 1 and arr.size >= 2:
        return float(arr[0]), float(arr[1])
    if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] >= 2:
        return float(arr[0, 0]), float(arr[0, 1])
    return None


def _scalar0(val):
    """First element of a scalar-or-list size field (radius/widths/heights/U/V)."""
    if val is None:
        return None
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[0]) if arr.size else None


def _first_color(val, default="#00e5ff"):
    """First colour from an edgecolors/color field (scalar or list)."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return str(val[0]) if val else default
    return str(val)


# Kinds that map cleanly onto a draggable EDIT widget. Everything else
# (ellipse/line — not creatable from the UI) stays on the static-marker path.
_WIDGET_KINDS = {"text", "circle", "rect", "arrow"}


def _apply_annotations(p2, annotations, axes=None, *, interactive=False,
                       panel_spec=None):
    """Map each annotation dict (``{kind, ...anyplotlib-marker kwargs, data coords}``)
    onto the panel. Unknown kinds and bad kwargs are logged and skipped so one
    malformed annotation can't blank the whole figure.

    Annotation dicts store offsets/sizes in calibrated DATA coordinates (the
    on-disk spec/YAML contract), but anyplotlib's 2-D path renders them as IMAGE
    PIXELS — so each dict is converted (on a COPY; the spec's stored dicts are
    never mutated) via ``coords.annotation_data_to_pixel`` using the panel's
    ``axes`` (``{units, x_axis, y_axis}``). ``axes=None``/unusable → identity
    passthrough (an uncalibrated panel already has index == pixel).

    ``interactive=True`` (edit mode) renders every ``text/circle/rect/arrow``
    annotation as a draggable anyplotlib WIDGET (shape widgets carry visible
    resize handles; the label is handle-free — see ``_add_annotation_widget``)
    instead of a static marker, and RETURNS a wiring list
    ``[(widget, panel_id, ann_index, panel_spec)]`` so the caller can attach a
    ``pointer_up`` drag-persist handler to each. Non-widget kinds (ellipse/line)
    still render as static markers. ``interactive=False`` returns ``[]``."""
    from spyde.actions.report.coords import annotation_data_to_pixel

    wiring = []
    panel_id = getattr(p2, "_id", None)
    for ann_index, ann in enumerate(annotations or []):
        try:
            conv = annotation_data_to_pixel(ann, axes)
            kind = str(conv.get("kind", "")).lower()
            if interactive and kind in _WIDGET_KINDS:
                widget = _add_annotation_widget(p2, kind, conv)
                if widget is not None:
                    wiring.append((widget, panel_id, ann_index, panel_spec))
                continue
            kw = {k: v for k, v in conv.items() if k != "kind"}
            if kind == "text":
                offsets = kw.pop("offsets", None)
                texts = kw.pop("texts", None)
                if offsets is None or texts is None:
                    continue
                p2.add_texts(offsets, texts, **kw)
            elif kind == "circle":
                offsets = kw.pop("offsets", None)
                if offsets is None:
                    continue
                p2.add_circles(offsets, **kw)
            elif kind == "ellipse":
                offsets = kw.pop("offsets", None)
                widths = kw.pop("widths", None)
                heights = kw.pop("heights", None)
                if offsets is None or widths is None or heights is None:
                    continue
                p2.add_ellipses(offsets, widths, heights, **kw)
            elif kind == "rect":
                offsets = kw.pop("offsets", None)
                widths = kw.pop("widths", None)
                heights = kw.pop("heights", None)
                if offsets is None or widths is None or heights is None:
                    continue
                p2.add_rectangles(offsets, widths, heights, **kw)
            elif kind == "arrow":
                offsets = kw.pop("offsets", None)
                U = kw.pop("U", None)
                V = kw.pop("V", None)
                if offsets is None or U is None or V is None:
                    continue
                p2.add_arrows(offsets, U, V, **kw)
            elif kind == "line":
                segments = kw.pop("segments", None)
                if segments is None:
                    continue
                p2.add_lines(segments, **kw)
            else:
                log.debug("report figure: unknown annotation kind %r", kind)
        except Exception as e:
            log.debug("report figure annotation %r failed: %s", ann, e)
    return wiring


def _add_annotation_widget(p2, kind, conv):
    """Add ONE draggable edit widget for a PIXEL-converted annotation dict *conv*
    (``kind`` in :data:`_WIDGET_KINDS`). Returns the created widget, or None if the
    geometry can't be read (logged + skipped, mirroring the static path's robustness).

    Field mapping (spec annotation → widget), all image-pixel:
      * text   → ``add_label_widget(x, y, text, fontsize, color)`` (offset = anchor)
      * circle → ``add_circle_widget(cx, cy, r, color)`` (offset = center, radius = r)
      * rect   → ``add_rectangle_widget(x, y, w, h, color)`` — the spec rect offset
                 is the CENTER + widths/heights; the widget x/y is the TOP-LEFT, so
                 ``x = cx - w/2``, ``y = cy - h/2``.
      * arrow  → ``add_arrow_widget(x, y, u, v, color)`` (offset = tail, U/V = vector)

    SHAPE widgets (circle/rect/arrow) are added with ``show_handles=True`` so the
    resize NODES are visible — they ARE the resize affordance (circle: center=move,
    east-edge=resize radius; rect: 4 corners resize with opposite-corner anchor;
    arrow: tail=reshape-tail, head=reshape-head, shaft=move). The LABEL widget keeps
    ``show_handles=False``: its only node is a plain anchor dot (a label has no
    resize DOF — it's reposition-only), so the dot is pure clutter over the text.

    KNOWN EXPORT CAVEAT: anyplotlib's ``drawOverlay2d`` draws these handle dots on the
    overlay canvas whenever ``show_handles`` is truthy, and ``exportPNG`` with
    ``includeWidgets:true`` blits that canvas verbatim (NO handle-suppression, unlike
    the figure-marker layer's ``forceNoHandles`` export path). So a PNG harvested from
    a cell WHILE it is in edit mode will bake the handle dots in. The HTML/standalone
    export is unaffected (it rebuilds NON-interactive → static markers, never widgets).
    See the SpyDE report-edit report for the flagged anyplotlib follow-up."""
    pt = _first_offset(conv.get("offsets"))
    if pt is None:
        return None
    cx, cy = pt
    if kind == "text":
        texts = conv.get("texts")
        text = str(texts[0]) if isinstance(texts, (list, tuple)) and texts \
            else str(conv.get("text", "Label"))
        color = _first_color(conv.get("color"), "#00e5ff")
        fontsize = int(conv.get("fontsize", 14) or 14)
        return p2.add_label_widget(x=cx, y=cy, text=text, fontsize=fontsize,
                                   color=color, show_handles=False)
    if kind == "circle":
        r = _scalar0(conv.get("radius"))
        if r is None:
            return None
        color = _first_color(conv.get("edgecolors"), "#00e5ff")
        return p2.add_circle_widget(cx=cx, cy=cy, r=float(r), color=color,
                                    show_handles=True)
    if kind == "rect":
        w = _scalar0(conv.get("widths"))
        hh = _scalar0(conv.get("heights"))
        if w is None or hh is None:
            return None
        color = _first_color(conv.get("edgecolors"), "#00e5ff")
        # spec rect offset is the CENTER; widget x/y is the TOP-LEFT.
        return p2.add_rectangle_widget(x=cx - float(w) / 2.0, y=cy - float(hh) / 2.0,
                                       w=float(w), h=float(hh), color=color,
                                       show_handles=True)
    if kind == "arrow":
        u = _scalar0(conv.get("U"))
        v = _scalar0(conv.get("V"))
        if u is None or v is None:
            return None
        color = _first_color(conv.get("edgecolors"), "#00e5ff")
        return p2.add_arrow_widget(x=cx, y=cy, u=float(u), v=float(v), color=color,
                                   show_handles=True)
    return None


# ── one panel: base image + overlay layers ────────────────────────────────────


def _render_panel(ax, panel, snap_map, *, interactive=False, wiring=None):
    """Render one panel onto ``ax``: the base image (layer 0) + any overlay layers
    (``add_layer``) + annotations. Returns the base ``Plot2D`` (or None if the panel
    has no paintable base snapshot).

    ``interactive=True`` renders the panel's annotations as draggable EDIT widgets
    and appends their ``(widget, panel_id, ann_index, panel_spec)`` wiring tuples to
    the passed ``wiring`` list (for the caller to attach drag-persist handlers)."""
    if not panel.layers:
        return None
    base_layer = panel.layers[0]
    base_arr = snap_map.get((panel.id, base_layer.id))
    if base_arr is None:
        return None
    base_arr = np.asarray(base_arr)
    is_rgb = base_arr.ndim == 3 and base_arr.shape[-1] in (3, 4)
    frame = base_arr if is_rgb else np.nan_to_num(np.asarray(base_arr, dtype=np.float32))

    axes_kw, units = _panel_axes_kw(panel)
    cmap = None if is_rgb else _resolve_cmap(base_layer.cmap)
    p2 = ax.imshow(frame, cmap=cmap, tile=False, **axes_kw)

    if not is_rgb and base_layer.clim and base_layer.clim[0] is not None \
            and base_layer.clim[1] is not None:
        try:
            p2.set_clim(float(base_layer.clim[0]), float(base_layer.clim[1]))
        except Exception as e:
            log.debug("report figure base set_clim failed: %s", e)

    # Overlay layers (layer 1..N) — each an add_layer over the base.
    for layer in panel.layers[1:]:
        if not layer.visible:
            continue
        arr = snap_map.get((panel.id, layer.id))
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim != 2:
            continue
        clim = None
        if layer.clim and layer.clim[0] is not None and layer.clim[1] is not None:
            clim = (float(layer.clim[0]), float(layer.clim[1]))
        try:
            p2.add_layer(arr, cmap=_resolve_cmap(layer.cmap),
                         alpha=float(layer.alpha), clim=clim,
                         visible=bool(layer.visible))
        except Exception as e:
            log.debug("report figure add_layer failed (panel %s layer %s): %s",
                      panel.id, layer.id, e)

    if panel.title and hasattr(p2, "set_title"):
        try:
            p2.set_title(str(panel.title))
        except Exception as e:
            log.debug("report figure set_title failed: %s", e)

    panel_wiring = _apply_annotations(p2, panel.annotations, panel.axes,
                                      interactive=interactive, panel_spec=panel)
    if interactive and wiring is not None:
        wiring.extend(panel_wiring)
    return p2


# ── callout insets ────────────────────────────────────────────────────────────

_CORNER_ALIASES = {
    "top-right": "top-right", "top-left": "top-left",
    "bottom-right": "bottom-right", "bottom-left": "bottom-left",
}


def _apply_insets(fig, panel, base_plot, snap_map):
    """Render each of a panel's callout insets: a small floating axes
    (``fig.add_inset``) showing the referenced panel's base snapshot, plus a
    connector to the source region when anyplotlib exposes ``indicate_region``."""
    for inset in panel.insets or []:
        try:
            ref_panel_id = inset.get("panel")
            corner = _CORNER_ALIASES.get(str(inset.get("corner", "top-right")),
                                         "top-right")
            w_frac = float(inset.get("w_frac", 0.3))
            h_frac = float(inset.get("h_frac", 0.3))
            # The inset image is the referenced panel's FIRST layer snapshot.
            arr = None
            for (pid, _lid), a in snap_map.items():
                if pid == ref_panel_id:
                    arr = a
                    break
            if arr is None:
                continue
            arr = np.asarray(arr)
            inset_ax = fig.add_inset(w_frac, h_frac, corner=corner,
                                     title=str(inset.get("title", "")))
            is_rgb = arr.ndim == 3 and arr.shape[-1] in (3, 4)
            ip = inset_ax.imshow(
                arr if is_rgb else np.nan_to_num(np.asarray(arr, dtype=np.float32)),
                cmap=(None if is_rgb else "gray"), tile=False)
            # Connector (dashed source rect + leader lines) — only if anyplotlib
            # provides it (added in parallel). Region = the snapshot nav-selector
            # region, if the spec recorded one.
            connector = inset.get("connector") or {}
            region = connector.get("region")
            if region is not None and base_plot is not None \
                    and hasattr(inset_ax, "indicate_region"):
                try:
                    inset_ax.indicate_region(base_plot, tuple(region))
                except Exception as e:
                    log.debug("report inset indicate_region failed: %s", e)
        except Exception as e:
            log.debug("report figure inset %r failed: %s", inset, e)


# ── the figure build ──────────────────────────────────────────────────────────


def _grid_shape(spec):
    """(rows, cols, width_ratios, height_ratios) for the figure grid from the
    layout spec. A ``single`` layout is 1×1."""
    layout = spec.layout or {"kind": "single"}
    if str(layout.get("kind")) == "grid":
        rows = int(layout.get("rows", 1) or 1)
        cols = int(layout.get("cols", 1) or 1)
        wr = layout.get("width_ratios")
        hr = layout.get("height_ratios")
        return max(1, rows), max(1, cols), wr, hr
    return 1, 1, None, None


def _apply_layout_spacing(fig, layout) -> None:
    """Apply the figure-level ``hspace``/``wspace`` from *layout* to the anyplotlib
    figure (whole-figure inter-panel gaps). Only the keys present in *layout* are
    passed; absent → anyplotlib leaves its current value unchanged. No-op when the
    figure has no ``subplots_adjust`` or the values aren't numeric."""
    if not layout or not hasattr(fig, "subplots_adjust"):
        return
    kw = {}
    for key in ("hspace", "wspace"):
        val = layout.get(key)
        if val is not None:
            try:
                kw[key] = float(val)
            except (TypeError, ValueError):
                pass
    if kw:
        try:
            fig.subplots_adjust(**kw)
        except Exception as e:
            log.debug("report figure subplots_adjust failed: %s", e)


def build_cell_figure(spec, snapshots, *, standalone: bool = False,
                      interactive: bool = False):
    """Render *spec* + *snapshots* → ``(fig, fig_id, html)``.

    ``snapshots`` is a ``{(panel_id, layer_id): ndarray}`` map (or, legacy, a single
    ndarray for a single-panel/single-layer cell). Builds the grid, each panel's
    base image + overlay layers + annotations, and callout insets, then materialises
    the pixel tokens for a self-contained standalone embed.

    ``standalone=True`` (the interactive-EXPORT path) returns HTML with the JS
    bundle fully INLINED — no machine-local ``file://`` ESM reference — so the
    figure renders on any machine, in any browser, inside a sandboxed ``srcdoc``
    iframe. ``standalone=False`` (default, the live report cell) keeps the shared
    ``file://`` ESM optimization used by the MDI iframes.

    ``interactive=True`` (EDIT MODE, the live report cell only — NEVER an export /
    standalone / bake path) renders each panel's annotations as draggable anyplotlib
    widgets instead of static markers, and stashes the drag-persist wiring list
    ``[(widget, panel_id, ann_index, panel_spec)]`` on ``fig._report_annotation_wiring``
    for the caller (``ReportManager.build_figure_window``) to attach ``pointer_up``
    handlers to. Non-interactive builds set it to ``[]``.

    Figure-level state applied regardless of *interactive*: ``spec.annotations``
    (figure-fraction markers) → ``fig.set_figure_markers`` (drawn + exported
    always), and ``spec.layout`` hspace/wspace → ``fig.subplots_adjust``. In edit
    mode ONLY, ``fig.edit_chrome`` is turned on (hover outlines / background click /
    figure-marker drag). The spec-panel-id → anyplotlib plot dispatch id map is
    stashed on ``fig._report_panel_map`` so the caller can push ``selected_panel``
    from a spec-panel id."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    snap_map = _as_snapshot_map(spec, snapshots)
    rows, cols, wr, hr = _grid_shape(spec)

    fig, axes = apl.subplots(rows, cols, width_ratios=wr, height_ratios=hr)

    def _ax_at(r, c):
        # Normalise apl.subplots' return (scalar / 1-D / 2-D) to a grid getter.
        if rows == 1 and cols == 1:
            return axes
        if rows == 1:
            return axes[c]
        if cols == 1:
            return axes[r]
        return axes[r][c]

    # Panels referenced ONLY as callout insets are drawn as floating insets, not
    # placed in the grid (they'd otherwise overwrite a grid cell).
    inset_panel_ids = set()
    for panel in spec.panels:
        for ins in (panel.insets or []):
            if ins.get("panel"):
                inset_panel_ids.add(ins["panel"])

    base_by_panel = {}
    used = set()
    wiring: list = []
    for panel in spec.panels:
        if panel.id in inset_panel_ids:
            continue
        try:
            gp = panel.grid_pos or [0, 0]
            r = int(gp[0]) if len(gp) > 0 else 0
            c = int(gp[1]) if len(gp) > 1 else 0
            r = max(0, min(r, rows - 1))
            c = max(0, min(c, cols - 1))
        except Exception:
            r, c = 0, 0
        ax = _ax_at(r, c)
        p2 = _render_panel(ax, panel, snap_map, interactive=interactive,
                           wiring=wiring)
        if p2 is not None:
            base_by_panel[panel.id] = p2
        used.add((r, c))

    # Callout insets (rendered after all panels so their referenced panel exists).
    for panel in spec.panels:
        if panel.insets:
            _apply_insets(fig, panel, base_by_panel.get(panel.id), snap_map)

    # Apply the figure-level layout spacing (hspace/wspace) from the layout dict
    # when the user has tuned it (the whole-figure gap between grid panels).
    _apply_layout_spacing(fig, spec.layout)

    # Figure-level annotations (fraction-coord markers). These are CONTENT —
    # drawn + exported ALWAYS, interactive or not — so they ride straight into
    # anyplotlib's figure-marker layer regardless of edit mode.
    fig_anns = list(getattr(spec, "annotations", None) or [])
    if fig_anns:
        try:
            fig.set_figure_markers(fig_anns)
        except Exception as e:
            log.debug("report figure set_figure_markers failed: %s", e)

    # In EDIT MODE, turn on anyplotlib's figure edit chrome (JS-local hover
    # outlines + background-click + figure-marker dragging).
    if interactive:
        try:
            fig.edit_chrome = True
        except Exception as e:
            log.debug("report figure edit_chrome set failed: %s", e)

    # Stash the SPEC-panel-id → anyplotlib plot dispatch id map so the manager can
    # push ``fig.selected_panel`` (which is keyed by the anyplotlib plot id) from a
    # spec-panel id. base_by_panel already keys the rendered base Plot2D per spec id.
    fig._report_panel_map = {pid: getattr(p2, "_id", None)
                             for pid, p2 in base_by_panel.items()
                             if getattr(p2, "_id", None) is not None}

    # Materialise pixel tokens (base + every layer) so the standalone HTML embed is
    # self-contained even when the app's binary transport is active.
    _resolve_pixels_for_standalone(fig)

    # Stash the edit-mode drag-persist wiring on the figure so the caller can
    # attach pointer_up handlers to each widget (and the wiring is kept alive
    # alongside the figure). Empty for non-interactive builds.
    fig._report_annotation_wiring = wiring

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id, standalone=standalone)
    return fig, fig_id, html


def _resolve_pixels_for_standalone(fig) -> None:
    """Rewrite every panel trait of *fig* with pixel change-tokens materialised to
    inline base64, so the standalone HTML embed is self-contained even when the
    app's Electron binary transport is active (base image AND every layer's
    ``layer_<id>_b64`` are resolved by anyplotlib's ``resolve_pixel_tokens``).

    Does this WITHOUT mutating ``APL_BINARY_TRANSPORT``: the old approach cleared
    the env around a ``fig._push`` and restored it, but the process-global env is
    shared with the live ``_NavPainter`` thread pushing MDI figures concurrently —
    a genuine race that could resolve a live figure's token to base64 (corrupting
    the binary-transport dedup) or leave a token unresolved in the export.

    Instead we replicate ``Figure._push``'s COLD path per panel directly: serialise
    the panel state, resolve its pixel tokens to real base64, split the heavy geom
    keys onto the ``panel_<id>_geom`` trait (where ``build_standalone_html`` reads
    them for the initial render), and write the light state onto ``panel_<id>_json``
    — so ``build_standalone_html`` serialises real pixels regardless of the env.
    The live MDI figures are untouched (we only write THIS export figure's traits).
    """
    import json

    plots_map = getattr(fig, "_plots_map", None)
    if not plots_map:
        return
    for panel_id, plot in list(plots_map.items()):
        try:
            tname = f"panel_{panel_id}_json"
            if not fig.has_trait(tname):
                continue
            state = plot.to_state_dict()
            # Materialise "\x00bin:…" tokens (base image + overlay mask + every
            # layer's layer_<id>_b64) to real base64, in place.
            if hasattr(plot, "resolve_pixel_tokens"):
                plot.resolve_pixel_tokens(state)
            geom_keys = getattr(plot, "_GEOM_KEYS", None)
            gname = f"panel_{panel_id}_geom"
            if geom_keys and fig.has_trait(gname):
                # Mirror _push's geom split so the JS reassembles the panel from the
                # geom trait (its initial render reads panel_<id>_geom). Bump the
                # revision so the resolved geom is treated as fresh.
                geom = {k: state.pop(k) for k in list(geom_keys) if k in state}
                rev = getattr(fig, "_geom_rev", {})
                geom_last = getattr(fig, "_geom_last", {})
                if geom != (geom_last.get(panel_id) if isinstance(geom_last, dict)
                            else None):
                    if isinstance(geom_last, dict):
                        geom_last[panel_id] = geom
                    if isinstance(rev, dict):
                        rev[panel_id] = rev.get(panel_id, 0) + 1
                    setattr(fig, gname, json.dumps(geom, sort_keys=True))
                state["_geom_rev"] = (rev.get(panel_id, 0)
                                      if isinstance(rev, dict) else 0)
                setattr(fig, tname, json.dumps(state))
            else:
                setattr(fig, tname, json.dumps(state))
        except Exception as e:
            log.debug("report figure pixel-resolve failed for %s: %s",
                      panel_id, e)


class ReportFigureController:
    """WindowController for a report figure cell's bare figure window.

    Registered via ``session.register_window_controller`` so the window has a
    dispatch + teardown identity; ``close()`` (called by
    ``Session._forget_window``) evicts the kept-alive figure and drops the
    report's back-reference. Idempotent."""

    def __init__(self, session, report, cell_id: str, window_id: int, fig=None):
        self.session = session
        self.report = report
        self.cell_id = cell_id
        self.window_id = int(window_id)
        self.fig = fig
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Drop the figure keep-alive for this window.
        try:
            from spyde.actions.figure_registry import forget_window
            forget_window(self.window_id)
        except Exception as e:
            log.debug("report figure controller keep-alive evict failed: %s", e)
        # Detach from the report so a later rebuild starts clean.
        try:
            mgr = getattr(self.session, "_report", None)
            if mgr is not None:
                mgr._controllers.pop(self.window_id, None)
                if mgr._window_by_cell.get(self.cell_id) == self.window_id:
                    mgr._window_by_cell.pop(self.cell_id, None)
        except Exception as e:
            log.debug("report figure controller detach failed: %s", e)

    def handle_action(self, name: str, payload: dict) -> bool:
        # Report figures carry no per-window actions of their own; the compose
        # edits are cell-scoped staged actions (spyde.actions.report.compose).
        return False

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


def _apply_text_sizes(plot, text_sizes: dict) -> None:
    """Apply per-element font-size overrides from a panel's ``text_sizes`` dict
    onto a live anyplotlib *plot* via its existing live setters (mutate state +
    push, no rebuild). Reads the CURRENT text from ``plot._state`` (falling back
    to ``""``) so a pure size change never clobbers the label/title text.

    Each key is ``hasattr``-guarded (a kind/backend that lacks the setter is
    silently skipped, mirroring the ``indicate_region`` guard) and individually
    try/excepted so one bad key can't block the rest."""
    if not text_sizes:
        return
    state = getattr(plot, "_state", None) or {}

    def _apply(key, fn_name, cur_text_key=None):
        if key not in text_sizes or text_sizes[key] is None:
            return
        fn = getattr(plot, fn_name, None)
        if fn is None:
            return
        try:
            size = float(text_sizes[key])
            if cur_text_key is None:
                fn(size)
            else:
                fn(str(state.get(cur_text_key, "") or ""), fontsize=size)
        except Exception as e:
            log.debug("report figure text_sizes[%s] via %s failed: %s",
                      key, fn_name, e)

    _apply("title", "set_title", "title")
    _apply("ticks", "set_tick_label_size")
    _apply("x_label", "set_xlabel", "x_label")
    _apply("y_label", "set_ylabel", "y_label")
    _apply("colorbar", "set_colorbar_label", "colorbar_label")
    _apply("legend", "set_legend_fontsize")


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
        lw = _scalar0(conv.get("linewidths"))
        return p2.add_circle_widget(cx=cx, cy=cy, r=float(r), color=color,
                                    linewidth=float(lw) if lw else 2.0,
                                    show_handles=True)
    if kind == "rect":
        w = _scalar0(conv.get("widths"))
        hh = _scalar0(conv.get("heights"))
        if w is None or hh is None:
            return None
        color = _first_color(conv.get("edgecolors"), "#00e5ff")
        lw = _scalar0(conv.get("linewidths"))
        # spec rect offset is the CENTER; widget x/y is the TOP-LEFT.
        return p2.add_rectangle_widget(x=cx - float(w) / 2.0, y=cy - float(hh) / 2.0,
                                       w=float(w), h=float(hh), color=color,
                                       linewidth=float(lw) if lw else 2.0,
                                       show_handles=True)
    if kind == "arrow":
        u = _scalar0(conv.get("U"))
        v = _scalar0(conv.get("V"))
        if u is None or v is None:
            return None
        color = _first_color(conv.get("edgecolors"), "#00e5ff")
        lw = _scalar0(conv.get("linewidths"))
        return p2.add_arrow_widget(x=cx, y=cy, u=float(u), v=float(v), color=color,
                                   linewidth=float(lw) if lw else 2.0,
                                   show_handles=True)
    return None


# ── one panel: base image + overlay layers ────────────────────────────────────


def _render_scene3d_panel(ax, panel, snap_map):
    """Render a scene3d panel (the 3-D IPF sphere) onto ``ax`` from its
    point-cloud snapshots — ``(panel_id, "xyz")`` / ``(panel_id, "rgb")`` — and
    the small ``panel.scene`` params. Mirrors ``build_ipf_3d_figure``'s scatter
    via the shared ``ipf_view.scatter_ipf_sphere`` call, so a report cell shows
    the SAME sphere the live explorer does. Returns the ``Plot3D`` (or None
    when the point cloud is missing/empty — the panel is skipped, no crash).

    Annotations/insets/edit widgets never apply here: 2-D marker geometry has
    no meaning on the 3-D scene, so the caller's annotation path is bypassed
    entirely for this panel kind."""
    from spyde.actions.ipf_view import (
        IPF3D_BOUNDS, IPF3D_POINT_SIZE, IPF3D_ZOOM, scatter_ipf_sphere,
    )

    xyz = snap_map.get((panel.id, "xyz"))
    rgb = snap_map.get((panel.id, "rgb"))
    if xyz is None or rgb is None or len(np.asarray(xyz)) == 0:
        return None
    scene = panel.scene or {}
    try:
        point_size = float(scene.get("point_size", IPF3D_POINT_SIZE))
    except (TypeError, ValueError):
        point_size = float(IPF3D_POINT_SIZE)
    bounds = scene.get("bounds") or IPF3D_BOUNDS
    try:
        return scatter_ipf_sphere(ax, np.asarray(xyz), np.asarray(rgb),
                                  point_size=point_size, bounds=bounds,
                                  zoom=IPF3D_ZOOM)
    except Exception as e:
        log.debug("report figure scene3d render failed (panel %s): %s",
                  panel.id, e)
        return None


def _line_axes_kw(panel):
    """(x, units) for a line panel's calibrated x-axis from ``panel.axes``
    (``{units, x_axis}`` — the 1-D snapshot's axes dict, distinct from the 2-D
    ``{units, x_axis, y_axis}`` shape ``_panel_axes_kw`` reads). Returns
    ``(None, "px")`` when the panel carries no usable x-axis (the caller falls
    back to ``ax.plot``'s own ``arange`` default)."""
    if panel is not None and panel.axes:
        try:
            xa = panel.axes.get("x_axis")
            if xa is not None:
                units = str(panel.axes.get("units") or "px")
                return np.asarray(xa, dtype=float), units
        except Exception as e:
            log.debug("report figure line axes from spec failed: %s", e)
    return None, "px"


def _render_line_panel(ax, panel, snap_map):
    """Render a line panel (``kind="line"``) onto ``ax``: the base curve
    (layer 0, via ``ax.plot``) + any extra overlay curves (layers 1..N, via
    ``Plot1D.add_line``), styled from each LayerSpec's ``color``/``linewidth``/
    ``label`` (unset → anyplotlib's own defaults). Sets the x-axis label from
    the panel's calibrated units and the panel title. Returns the base
    ``Plot1D`` (or None if the panel has no paintable base snapshot).

    A LENGTH MISMATCH between the panel's stored x-axis and a layer's y-data
    (e.g. a stale axes dict after a data-shape change) falls back to a bare
    index axis for THAT layer rather than raising — logged at debug.

    No scalebar / colorbar / clim apply to a line panel (1-D has no
    calibrated area or intensity range to bar/bar-label); annotations/insets
    are the caller's responsibility to skip (see ``_render_panel``)."""
    if not panel.layers:
        return None
    base_layer = panel.layers[0]
    base_y = snap_map.get((panel.id, base_layer.id))
    if base_y is None:
        return None
    base_y = np.asarray(base_y, dtype=np.float64).reshape(-1)

    xa, units = _line_axes_kw(panel)
    if xa is not None and xa.shape[0] != base_y.shape[0]:
        log.debug("report line panel %s: x_axis length %d != data length %d, "
                  "falling back to index axis", panel.id, xa.shape[0],
                  base_y.shape[0])
        xa = None

    plot_kw = {}
    if base_layer.color:
        plot_kw["color"] = str(base_layer.color)
    if base_layer.linewidth:
        plot_kw["linewidth"] = float(base_layer.linewidth)
    if base_layer.label:
        plot_kw["label"] = str(base_layer.label)
    p1 = ax.plot(base_y, axes=([xa] if xa is not None else None),
                units=units, **plot_kw)

    for layer in panel.layers[1:]:
        if not layer.visible:
            continue
        y = snap_map.get((panel.id, layer.id))
        if y is None:
            continue
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        lxa, _lunits = _line_axes_kw(panel)
        if lxa is not None and lxa.shape[0] != y.shape[0]:
            lxa = None
        line_kw = {}
        if layer.color:
            line_kw["color"] = str(layer.color)
        if layer.linewidth:
            line_kw["linewidth"] = float(layer.linewidth)
        if layer.label:
            line_kw["label"] = str(layer.label)
        try:
            p1.add_line(y, x_axis=lxa, **line_kw)
        except Exception as e:
            log.debug("report figure line add_line failed (panel %s layer "
                      "%s): %s", panel.id, layer.id, e)

    if units and units != "px" and hasattr(p1, "set_xlabel"):
        try:
            p1.set_xlabel(units)
        except Exception as e:
            log.debug("report line panel set_xlabel failed: %s", e)

    if panel.title and hasattr(p1, "set_title"):
        try:
            p1.set_title(str(panel.title))
        except Exception as e:
            log.debug("report figure line set_title failed: %s", e)

    if panel.text_sizes:
        _apply_text_sizes(p1, panel.text_sizes)

    return p1


def _render_panel(ax, panel, snap_map, *, interactive=False, wiring=None):
    """Render one panel onto ``ax``: the base image (layer 0) + any overlay layers
    (``add_layer``) + annotations. Returns the base ``Plot2D`` (or None if the panel
    has no paintable base snapshot).

    A ``scene3d`` panel takes the 3-D scatter path instead (returns a ``Plot3D``);
    image layers/annotations/edit widgets don't apply to it. A ``line`` panel takes
    the 1-D curve path instead (returns a ``Plot1D``); annotations/insets don't
    apply to it either — no scalebar/colorbar/clim, and the caller's annotation
    call is skipped entirely for this kind (see the ``kind == "line"`` branch below,
    which returns BEFORE the shared image-annotation call at the end of this
    function).

    ``interactive=True`` renders the panel's annotations as draggable EDIT widgets
    and appends their ``(widget, panel_id, ann_index, panel_spec)`` wiring tuples to
    the passed ``wiring`` list (for the caller to attach drag-persist handlers)."""
    if str(getattr(panel, "kind", "")) == "scene3d":
        return _render_scene3d_panel(ax, panel, snap_map)
    if str(getattr(panel, "kind", "")) == "line":
        # Annotations/callouts are refused on a line panel (compose.py's
        # repfig_add_annotation/add_callout/add_time_callouts/add_zoom_callout
        # all check panel.kind and error out before mutating the spec), so
        # there is never a wiring list to populate here — just render + return.
        return _render_line_panel(ax, panel, snap_map)
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
            # tint only when SET: a legacy (untinted) layer issues the exact
            # pre-tint call, and the cmap always rides along as the stored
            # revert value (add_layer keeps both; only Layer.set() rejects the
            # cmap+tint combination). The base layer never tints — it's the
            # panel's imshow, not an add_layer.
            add_kw = {}
            if getattr(layer, "tint", None):
                add_kw["tint"] = str(layer.tint)
            p2.add_layer(arr, cmap=_resolve_cmap(layer.cmap),
                         alpha=float(layer.alpha), clim=clim,
                         visible=bool(layer.visible), **add_kw)
        except Exception as e:
            log.debug("report figure add_layer failed (panel %s layer %s): %s",
                      panel.id, layer.id, e)

    if panel.title and hasattr(p2, "set_title"):
        try:
            p2.set_title(str(panel.title))
        except Exception as e:
            log.debug("report figure set_title failed: %s", e)

    if panel.text_sizes:
        _apply_text_sizes(p2, panel.text_sizes)

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


# Edit-mode callout markers are drawn in a distinct accent so they don't read
# as content annotations (annotation widgets default to #00e5ff).
_CALLOUT_MARKER_COLOR = "#89b4fa"


def _add_callout_marker(base_plot, inset, base_shape):
    """A small draggable circle widget on the BASE panel marking a fresh-slice
    callout's nav position (edit mode only). Only for an inset carrying 2-D
    ``nav_indices`` AND a connector — the connector is created exactly when the
    base panel IS the navigator image, so the nav point maps onto base-panel
    pixels (a ``time_index`` inset on an image base panel has no spatial
    anchor → skip). anyplotlib 2-D widgets take IMAGE PIXELS and a navigator
    image has index == pixel, so the nav indices are used directly. Returns the
    widget, or None (unusable geometry — logged + skipped)."""
    if base_plot is None:
        return None
    nav_indices = inset.get("nav_indices")
    if not isinstance(nav_indices, (list, tuple)) or len(nav_indices) != 2:
        return None
    if inset.get("connector") is None:
        return None
    try:
        ix, iy = float(nav_indices[0]), float(nav_indices[1])
        # Radius scaled to the base image so the handle stays visible on a tiny
        # nav map yet unobtrusive on a large one.
        r = max(0.5, 0.02 * max(base_shape)) if base_shape else 3.0
        return base_plot.add_circle_widget(cx=ix, cy=iy, r=float(r),
                                           color=_CALLOUT_MARKER_COLOR,
                                           show_handles=False)
    except Exception as e:
        log.debug("callout marker widget failed: %s", e)
        return None


def _add_zoom_region_widget(base_plot, panel, inset):
    """A draggable RECTANGLE widget on the BASE panel marking a zoom-region
    callout's source rect (edit mode only) — the resize-handle affordance for
    ``repfig_add_zoom_callout``. Only for an inset carrying ``zoom_region``
    (DATA coords); converted to the base panel's IMAGE PIXELS via
    ``coords.data_region_to_index`` (the inverse of ``compose.
    _index_region_to_data``, which built ``zoom_region`` in the first place).
    Returns the widget, or None (no zoom_region / unusable geometry — logged +
    skipped, mirroring :func:`_add_callout_marker`)."""
    from spyde.actions.report import coords

    if base_plot is None:
        return None
    region = inset.get("zoom_region")
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        return None
    try:
        x, y, w, h = coords.data_region_to_index(region, panel.axes)
        return base_plot.add_rectangle_widget(
            x=float(x), y=float(y), w=float(w), h=float(h),
            color=_CALLOUT_MARKER_COLOR, linewidth=2, show_handles=True)
    except Exception as e:
        log.debug("zoom region widget failed: %s", e)
        return None


def _apply_insets(fig, panel, base_plot, snap_map, *, interactive=False,
                  wiring=None, zoom_wiring=None, inset_id_map=None):
    """Render each of a panel's callout insets: a small floating axes
    (``fig.add_inset``) showing the referenced panel's base snapshot, plus a
    connector to the source region when anyplotlib exposes ``indicate_region``.
    An inset may carry an ``anchor`` ([fx, fy] figure fractions, top-left
    corner) for free placement — it wins over ``corner``.

    A ZOOM-REGION inset (``inset["zoom_region"]`` set — a magnified crop of the
    BASE panel's own pixels, from ``repfig_add_zoom_callout``) renders with the
    BASE layer-0's cmap/clim instead of the flat "gray" every other (fresh-
    slice) callout uses, so the magnified crop matches the parent's display.

    ``interactive=True`` (edit mode) also adds, per inset: a draggable MARKER
    widget on the base panel for a fresh-slice callout (see
    :func:`_add_callout_marker`, appended to *wiring* as ``(widget, panel_id,
    inset_index, panel_spec)``), OR a draggable RECTANGLE widget for a zoom
    callout (see :func:`_add_zoom_region_widget`, appended to *zoom_wiring* as
    ``(widget, panel_id, inset_index)``) — never both, since the two kinds are
    mutually exclusive per inset.

    *inset_id_map*, when given, is populated with ``{ref_panel_id:
    inset_dispatch_id}`` for every rendered inset — the SPEC inset-panel id →
    the anyplotlib Plot2D dispatch id ``fig.add_inset(...).imshow(...)``
    creates. The caller stashes this SEPARATELY from ``fig._report_panel_map``
    (as ``fig._report_inset_map``) — inset panels are floating callouts, not
    grid panels, so they must NOT enter the panel-select / text-size dispatch
    path that iterates ``_report_panel_map``; an ``inset_geometry_change``
    handler inverts ``_report_inset_map`` instead to resolve its ``inset_id``
    (a dispatch id) back to the owning spec panel id."""
    base_shape = None
    base_layer = panel.layers[0] if panel.layers else None
    if base_layer is not None:
        a0 = snap_map.get((panel.id, base_layer.id))
        if a0 is not None:
            base_shape = np.asarray(a0).shape[:2]
    for inset_index, inset in enumerate(panel.insets or []):
        try:
            ref_panel_id = inset.get("panel")
            corner = _CORNER_ALIASES.get(str(inset.get("corner", "top-right")),
                                         "top-right")
            w_frac = float(inset.get("w_frac", 0.3))
            h_frac = float(inset.get("h_frac", 0.3))
            inset_kw = {}
            anchor = inset.get("anchor")
            if anchor is not None:
                try:
                    inset_kw["anchor"] = (float(anchor[0]), float(anchor[1]))
                except (TypeError, ValueError, IndexError):
                    pass
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
                                     title=str(inset.get("title", "")),
                                     **inset_kw)
            is_zoom = inset.get("zoom_region") is not None
            is_rgb = arr.ndim == 3 and arr.shape[-1] in (3, 4)
            if is_rgb:
                inset_cmap = None
            elif is_zoom and base_layer is not None:
                # Match the parent's display so the magnified crop reads as
                # "the same image, zoomed in" rather than a re-tinted copy.
                inset_cmap = _resolve_cmap(base_layer.cmap)
            else:
                inset_cmap = "gray"
            ip = inset_ax.imshow(
                arr if is_rgb else np.nan_to_num(np.asarray(arr, dtype=np.float32)),
                cmap=inset_cmap, tile=False)
            disp_id = getattr(ip, "_id", None)
            if inset_id_map is not None and ref_panel_id is not None \
                    and disp_id is not None:
                inset_id_map[ref_panel_id] = disp_id
            if is_zoom and not is_rgb and base_layer is not None \
                    and base_layer.clim and base_layer.clim[0] is not None \
                    and base_layer.clim[1] is not None:
                try:
                    ip.set_clim(float(base_layer.clim[0]), float(base_layer.clim[1]))
                except Exception as e:
                    log.debug("report zoom inset set_clim failed: %s", e)
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
            if interactive:
                if is_zoom:
                    if zoom_wiring is not None:
                        w = _add_zoom_region_widget(base_plot, panel, inset)
                        if w is not None:
                            zoom_wiring.append((w, panel.id, inset_index))
                elif wiring is not None:
                    w = _add_callout_marker(base_plot, inset, base_shape)
                    if w is not None:
                        wiring.append((w, panel.id, inset_index, panel))
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

    # Callout insets (rendered after all panels so their referenced panel
    # exists). In edit mode this also creates the draggable fresh-slice
    # markers (wired by the caller via ``fig._report_callout_wiring``) and the
    # draggable zoom-region RECTANGLES (via ``fig._report_zoom_wiring``) — the
    # two lists are disjoint per inset (see ``_apply_insets``).
    callout_wiring: list = []
    zoom_wiring: list = []
    inset_id_map: dict = {}
    for panel in spec.panels:
        if panel.insets:
            _apply_insets(fig, panel, base_by_panel.get(panel.id), snap_map,
                          interactive=interactive, wiring=callout_wiring,
                          zoom_wiring=zoom_wiring, inset_id_map=inset_id_map)

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
    # Inset panels are kept OUT of this map (they aren't grid-selectable / don't
    # take a text-size target) — see ``fig._report_inset_map`` below instead.
    fig._report_panel_map = {pid: getattr(p2, "_id", None)
                             for pid, p2 in base_by_panel.items()
                             if getattr(p2, "_id", None) is not None}

    # SPEC inset-panel id → its anyplotlib Plot2D dispatch id (from
    # ``_apply_insets``'s inset_id_map). Lets an ``inset_geometry_change``
    # event's ``inset_id`` (a dispatch id) resolve back to the spec panel that
    # owns the moved/resized inset, without polluting ``_report_panel_map``'s
    # grid-panel-only contract (panel-select / text-size dispatch).
    fig._report_inset_map = dict(inset_id_map)

    # Materialise pixel tokens (base + every layer) so the standalone HTML embed is
    # self-contained even when the app's binary transport is active.
    _resolve_pixels_for_standalone(fig)

    # Stash the edit-mode drag-persist wiring on the figure so the caller can
    # attach pointer_up handlers to each widget (and the wiring is kept alive
    # alongside the figure). Empty for non-interactive builds. The callout
    # wiring is the same contract for the fresh-slice marker widgets:
    # ``[(widget, panel_id, inset_index, panel_spec)]``. The zoom wiring is the
    # rectangle-widget contract for zoom-region callouts:
    # ``[(widget, base_panel_id, inset_index)]`` — the handler side
    # (``handlers._wire_zoom_region_drag``) re-resolves the PanelSpec from
    # ``cell.spec`` by id at drop time (the same defensive re-lookup every
    # other drag handler here does, so a rebuilt/removed panel can't leave a
    # handler holding a stale spec object).
    fig._report_annotation_wiring = wiring
    fig._report_callout_wiring = callout_wiring
    fig._report_zoom_wiring = zoom_wiring

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
            # A 3-D panel (scene3d) has no pixel tokens: its geometry is already
            # self-contained b64 in the state the normal push wrote. Skip it —
            # nothing to resolve, and this loop's geom rewrite is Plot2D-shaped.
            if str((getattr(plot, "_state", None) or {}).get("kind", "")) == "3d":
                continue
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

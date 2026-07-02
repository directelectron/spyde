"""
ipf_view.py — the 3-D IPF explorer (anyplotlib ``scatter3d`` on the unit sphere).

An orientation IPF window normally shows the 2-D RGB map. This adds a SECOND
figure to that same window — a 3-D scatter of every position's reduced crystal
direction on the unit sphere, coloured by its IPF colour — tagged ``view="3d"``
so the frontend renders a 2D ⇄ 3D toggle (the
``cssfrancis.github.io/anyplotlib`` IPF-explorer view).

Works for BOTH the dense Orientation Mapping result (``SpyDEOrientationMap``)
and the Vector Orientation result (which exposes ``to_orientation_map()``).
No Qt.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.figure_registry import keep_alive

log = logging.getLogger(__name__)


def _as_orientation_map(result):
    """Normalise either result type to a SpyDEOrientationMap (has
    ``ipf_sphere_points``)."""
    if hasattr(result, "to_orientation_map"):
        return result.to_orientation_map()
    return result


def build_ipf_3d_figure(xyz: np.ndarray, rgb: np.ndarray, highlight=None):
    """Build the 3-D IPF scatter figure → ``(fig, fig_id, html)``. If
    ``highlight`` is a 3-vector, draw a large black-ringed white marker there (the
    orientation of the pixel picked by the map's point selector)."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    fig, axes = apl.subplots(1, 1)
    ax = axes[0][0] if isinstance(axes, list) else axes
    colors = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    p3d = ax.scatter3d(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        colors=colors, point_size=6,
        x_label="[100]", y_label="[010]", z_label="[001]",
        bounds=((-1, 1),) * 3, zoom=1.4,
    )
    try:
        p3d.set_sphere(1.0)
    except Exception as e:
        log.debug("setting IPF sphere failed: %s", e)
    if highlight is not None:
        try:
            p3d.set_highlight(float(highlight[0]), float(highlight[1]),
                              float(highlight[2]), color="#ffffff", size=11)
        except Exception as e:
            log.debug("setting IPF highlight failed: %s", e)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html, p3d


def emit_ipf_3d(window_id: int, result, direction: str = "z",
                highlight_iyix=None, tree=None) -> bool:
    """Compute the sphere points from *result* and emit a ``view="3d"`` figure for
    *window_id*. ``highlight_iyix=(iy,ix)`` marks that pixel's orientation. When a
    ``tree`` is given the live ``Plot3D`` is cached on ``tree._ipf_p3d`` so a later
    point-pick updates the highlight IN PLACE (camera preserved) instead of
    re-emitting. Returns True if a figure was emitted."""
    from spyde.backend.ipc import emit

    om = _as_orientation_map(result)
    try:
        xyz, rgb = om.ipf_sphere_points(direction)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("ipf_sphere_points failed: %s", e)
        return False
    if xyz is None or len(xyz) == 0:
        return False

    highlight = None
    if highlight_iyix is not None:
        try:
            iy, ix = int(highlight_iyix[0]), int(highlight_iyix[1])
            highlight = om.ipf_xyz(iy, ix, 0, direction)[0]      # the sphere point
        except Exception:
            highlight = None

    _fig, fig_id, html, p3d = build_ipf_3d_figure(
        np.asarray(xyz), np.asarray(rgb), highlight=highlight)
    keep_alive(int(window_id), _fig)
    if tree is not None:
        tree._ipf_p3d = p3d
    emit({
        "type": "figure", "fig_id": fig_id, "window_id": window_id,
        "html": html, "title": "IPF (3D)", "is_navigator": False, "view": "3d",
    })
    return True


def _ipf_key_color_grid(phase, direction: str, n: int):
    """Build the IPF colour-KEY raster over the fundamental sector → ``(xc, yc,
    colors, xy_edges, label_xy, labels)``.

    ``xc``/``yc`` are the ``(n+1, n+1)`` cell-CORNER stereographic grids (for
    ``pcolormesh``); ``colors`` is the ``(n, n)`` array of ``#rrggbb`` strings —
    the orix IPF colour at each cell-centre direction, masked (``""``) outside the
    fundamental sector so the mesh clips itself to the triangle.
    """
    import numpy as np
    from orix.plot import IPFColorKeyTSL
    from orix.projections import InverseStereographicProjection

    from spyde.signals.orientation_map import _direction_vector, ipf_triangle_xy

    key = IPFColorKeyTSL(phase.point_group.laue,
                         direction=_direction_vector(direction))
    dck = key.direction_color_key
    sector = phase.point_group.laue.fundamental_sector

    xy_edges, label_xy, labels = ipf_triangle_xy(phase)
    ex, ey = np.asarray(xy_edges)[:, 0], np.asarray(xy_edges)[:, 1]
    xmin, xmax = float(ex.min()), float(ex.max())
    ymin, ymax = float(ey.min()), float(ey.max())

    # Cell-corner grids (n+1) and cell-centre grids (n) for the colour eval.
    xe = np.linspace(xmin, xmax, n + 1)
    ye = np.linspace(ymin, ymax, n + 1)
    xc, yc = np.meshgrid(xe, ye)
    cx = 0.5 * (xe[:-1] + xe[1:])
    cy = 0.5 * (ye[:-1] + ye[1:])
    CX, CY = np.meshgrid(cx, cy)

    inv = InverseStereographicProjection()
    v = inv.xy2vector(CX.ravel(), CY.ravel())
    inside = np.asarray(v < sector).reshape(CX.shape)        # fundamental sector
    rgb = np.asarray(dck.direction2color(v)).reshape(CX.shape + (3,))
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    colors = np.empty(CX.shape, dtype=object)
    for i in range(CX.shape[0]):
        for j in range(CX.shape[1]):
            if inside[i, j]:
                r, g, b = rgb8[i, j]
                colors[i, j] = f"#{r:02x}{g:02x}{b:02x}"
            else:
                colors[i, j] = ""                            # skipped by pcolormesh
    masked = np.ma.masked_array(colors, mask=~inside)
    return xc, yc, masked, np.asarray(xy_edges), np.asarray(label_xy), labels


def build_ipf_key_figure(result, direction: str = "z", *, n: int = 120):
    """Build the IPF colour-KEY triangle legend as a NATIVE anyplotlib figure →
    ``(fig, fig_id, html)``.

    The standard stereographic fundamental-sector colour key (e.g. cubic
    [001]/[101]/[111]) rendered with the same anyplotlib primitives as
    :func:`spyde.actions.ipf_density.build_ipf_density_figure` — a ``pcolormesh``
    quad mesh whose per-cell face colours come directly from the orix IPF colour
    key (``direction2color``), clipped to the curved sector boundary, with the
    white sector outline and the ``[hkl]`` corner labels. Same key for sample
    X/Y/Z (it's the crystal-direction colour map)."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron

    from spyde.actions.ipf_density import _sector_limits
    from spyde.drawing.plots.plot import finalize_figure_html

    om = _as_orientation_map(result)
    phase = om.orix_phase(0)                       # primary phase's point group

    xc, yc, colors, xy_edges, label_xy, labels = _ipf_key_color_grid(
        phase, direction, n)
    xlim, ylim = _sector_limits(xy_edges)

    fig, axes = apl.subplots(1, 1)
    ax = axes[0][0] if isinstance(axes, list) else axes
    xy = ax.axes2d(xlim=xlim, ylim=ylim, aspect="equal")
    # `colors` is an array of CSS colour strings → pcolormesh draws each cell with
    # that exact face colour (the direct-RGB path); clip to the sector boundary.
    xy.pcolormesh(xc, yc, colors, clip_path=xy_edges)
    xy.plot(xy_edges[:, 0], xy_edges[:, 1], color="#ffffff", linewidth=1.5)
    for (lx, ly), txt in zip(np.asarray(label_xy, dtype=float), labels):
        xy.text(float(lx), float(ly), str(txt), color="#ffffff", fontsize=12)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html


def emit_ipf_key(window_id: int, result, direction: str = "z") -> bool:
    """Emit the IPF colour-key triangle legend for *window_id* as a native
    anyplotlib ``view="ipf_key"`` figure (pinned in a corner of the IPF map)."""
    from spyde.backend.ipc import emit
    try:
        _fig, fig_id, html = build_ipf_key_figure(result, direction)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("ipf key figure failed: %s", e)
        return False
    keep_alive(int(window_id), _fig)
    emit({
        "type": "figure", "fig_id": fig_id, "window_id": int(window_id),
        "html": html, "title": "IPF colour key", "is_navigator": False,
        "view": "ipf_key",
    })
    return True


def attach_ipf_3d(tree, result, direction: str = "z") -> bool:
    """Add the 3-D IPF view + the colour-key triangle legend to *tree*'s IPF
    window (its first signal plot)."""
    sp = next(iter(getattr(tree, "signal_plots", []) or []), None)
    wid = getattr(sp, "window_id", None)
    if wid is None:
        return False
    tree._ipf_result = result          # remember it for X/Y/Z re-colouring
    emit_ipf_key(wid, result, direction)
    ok = emit_ipf_3d(wid, result, direction, tree=tree)
    try:                               # native IPDF density heatmap (3rd toggle)
        from spyde.actions.ipf_density import emit_ipf_density
        emit_ipf_density(wid, result, direction)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("ipf density attach failed: %s", e)
    return ok


def attach_ipf_point_selector(tree, result, direction: str = "z") -> None:
    """Add a white crosshair POINT SELECTOR to the IPF map's first plot. Picking a
    pixel re-emits the 3-D IPF sphere with that orientation marked — the "pick on
    the map, see it on the IPF legend" interaction."""
    sp = next(iter(getattr(tree, "signal_plots", []) or []), None)
    plot2d = getattr(sp, "_plot2d", None) if sp is not None else None
    wid = getattr(sp, "window_id", None)
    if plot2d is None or wid is None:
        return
    try:
        ny, nx = int(result.nav_shape[0]), int(result.nav_shape[1])
        widget = plot2d.add_crosshair_widget(color="#ffffff")
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("ipf point selector init failed: %s", e)
        return
    tree._ipf_picker = widget
    tree._ipf_result = result

    def _on_pick(event=None):
        try:
            ix = int(round(float(widget.cx)))
            iy = int(round(float(widget.cy)))
        except Exception:
            return
        if not (0 <= iy < ny and 0 <= ix < nx):
            return
        res = getattr(tree, "_ipf_result", result)
        d = getattr(tree, "_ipf_direction", direction)
        p3d = getattr(tree, "_ipf_p3d", None)
        if p3d is not None:
            # Camera-preserving: move the highlight on the existing 3-D figure in
            # place (a view-only push) rather than rebuilding the whole sphere.
            try:
                v = _as_orientation_map(res).ipf_xyz(iy, ix, 0, d)[0]
                p3d.set_highlight(float(v[0]), float(v[1]), float(v[2]),
                                  color="#ffffff", size=11)
                return
            except Exception as e:
                log.debug("in-place IPF highlight failed, rebuilding: %s", e)
        emit_ipf_3d(wid, res, d, (iy, ix), tree=tree)   # fallback: rebuild

    try:
        widget.add_event_handler(_on_pick, "pointer_up")
    except Exception as e:
        log.debug("wiring IPF point-selector pick handler failed: %s", e)


def ipf_set_direction(session, plot, payload) -> None:
    """Re-colour an IPF window's 2-D map AND its 3-D explorer by sample direction
    x | y | z (the IPF axis selector). The orientation result is cached on the
    tree by ``attach_ipf_3d``; works for raw-OM (`orientation_map`) and vector-OM
    (`vector_orientation`)."""
    direction = str(payload.get("direction", "z")).lower()
    if direction not in ("x", "y", "z"):
        return
    tree = getattr(plot, "signal_tree", None)
    if tree is None:
        return
    result = (getattr(tree, "_ipf_result", None)
              or getattr(tree, "orientation_map", None)
              or getattr(tree, "vector_orientation", None))
    if result is None:
        return
    try:
        om = _as_orientation_map(result)
        ipf = om.ipf_color_map(direction)                 # (ny,nx,3) uint8
        for sp in list(getattr(tree, "signal_plots", [])):
            try:
                sp.needs_auto_level = True
                sp.set_data(ipf)
            except Exception as e:
                log.debug("painting IPF map for new direction failed: %s", e)
        tree._ipf_direction = direction
        wid = getattr(plot, "window_id", None)
        if wid is not None:
            emit_ipf_3d(wid, result, direction, tree=tree)   # frontend replaces the old 3-D
            try:                                              # refresh density heatmap too
                from spyde.actions.ipf_density import emit_ipf_density
                emit_ipf_density(wid, result, direction)
            except Exception as e:
                log.debug("refreshing IPF density heatmap failed: %s", e)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("ipf_set_direction failed: %s", e)

"""
ipf_window.py — the IPF EXPLORER window (window 2 of the orientation result pair).

An orientation-mapping result opens **two** windows:

* **Window 1 — the orientation MAPS.** The IPF-X / IPF-Y / IPF-Z projections of
  the scan, as chip views on the result tree's own plot window (built by
  :func:`spyde.actions.ipf_view.attach_ipf_projections`, using the ordinary
  ``view_label`` chip strip: click one, ⌘-click to tile all three).
* **Window 2 — THIS one.** The inverse pole figure itself, with two INDEPENDENT
  toggles — ``[2D | 3D]`` and ``[Points | Heatmap]`` — i.e. four figures
  pre-built and swapped client-side, exactly the way the old 2D/3D/PDF toggle
  already worked (no backend round-trip to switch).

The four figures and their ``view`` tags:

===============  ==================  ==================================================
toggle state     ``view``            what it is
===============  ==================  ==================================================
2D · Points      ``ipf2d``           stereographic scatter of every position's reduced
                                     crystal direction, IPF-coloured, in the
                                     fundamental-sector triangle
2D · Heatmap     ``density``         the inverse pole DENSITY function (orix
                                     ``pole_density_function``) — :mod:`ipf_density`
3D · Points      ``3d``              the same directions as a point cloud on the unit
                                     sphere — :func:`ipf_view.build_ipf_3d_figure`
3D · Heatmap     ``density3d``       the density field sampled ON the sphere: the
                                     sector's equal-area cells lifted back to unit
                                     vectors and coloured by MRD
===============  ==================  ==================================================

``3d`` and ``density`` keep their historical tag values so the Report Builder's
``scene3d`` drag payload and the existing density tests keep working unchanged.

**The crosshair drives this window.** ``ipf_view.attach_ipf_point_selector``
puts a white crosshair on the map window; every pick calls
:meth:`IpfWindowController.show_orientation`, which

1. moves the highlight marker on all four figures, and
2. **rotates the sphere** so that direction faces the camera
   (:func:`face_camera` — the turntable-camera aim from anyplotlib's
   ``plot_ipf_explorer`` gallery example).

3-D · Points stays a point cloud (it *is* one orientation per pixel). 3-D ·
Heatmap is a **continuous textured surface** — ``Plot3D.set_texture``,
anyplotlib >= 0.6.0 — with the density painted on as a skin rather than
approximated by a dense dot grid. No Qt.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.figure_registry import keep_alive
from spyde.actions.ipf_view import _as_orientation_map

logger = logging.getLogger(__name__)

#: Angular bin size (degrees) for the 3-D density sampling. This now sets BOTH
#: the surface mesh and the texture resolution — one texel per equal-area cell
#: — and the texture is interpolated across each triangle, so it no longer has
#: to be fine enough to hide a dot grid the way the old point patch did.
DENSITY3D_RESOLUTION = 1.0
#: 2-D scatter cap — a big EBSD map is millions of positions and the IPF is a
#: DISTRIBUTION, so a uniform subsample shows the same picture far faster.
POINTS2D_MAX = 30_000
POINTS2D_SIZE = 5

# The four toggle states → their figure `view` tag.
VIEW_POINTS_2D = "ipf2d"
VIEW_HEATMAP_2D = "density"
VIEW_POINTS_3D = "3d"
VIEW_HEATMAP_3D = "density3d"


def face_camera(v) -> tuple[float, float]:
    """``(azimuth°, elevation°)`` that aim the turntable camera straight down the
    unit vector *v* — i.e. rotate the sphere so *v* faces the viewer.

    Lifted from anyplotlib's ``plot_ipf_explorer`` gallery example: the view
    faces ``v`` when ``el = asin(vz)`` and ``az = atan2(vx, -vy)``.
    """
    vx, vy, vz = (float(v[0]), float(v[1]), float(v[2]))
    el = float(np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0))))
    az = float(np.degrees(np.arctan2(vx, -vy)))
    return az, el


def _aim_at(xyz: np.ndarray) -> tuple[float, float]:
    """Opening camera angles for a cloud of unit vectors: point at their mean
    direction.

    The default turntable camera (-60°, 30°) looks at a spot the fundamental
    sector may not even reach — a cubic IPF cloud sits in one small cap, so the
    sphere opened showing its blank back side. Aiming at the centroid means the
    window opens on the data.
    """
    v = np.asarray(xyz, dtype=float)
    if v.size == 0:
        return -60.0, 30.0
    m = np.nanmean(v, axis=0)
    n = float(np.linalg.norm(m))
    return face_camera(m / n) if n > 1e-9 else (-60.0, 30.0)


def _hex_colors(rgb: np.ndarray) -> list[str]:
    """(N, 3) uint8 → the ``['#rrggbb', …]`` list anyplotlib's per-point colour
    argument takes."""
    a = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
    packed = (a[:, 0].astype(np.uint32) << 16) | (a[:, 1].astype(np.uint32) << 8) \
        | a[:, 2].astype(np.uint32)
    return ["#%06x" % int(p) for p in packed]


def _present_phases(om) -> list[int]:
    """Phase indices that actually occur in the map (never empty)."""
    pm = np.asarray(om.phase_map()).reshape(-1)
    pidxs = [p for p in range(int(om.n_phases)) if np.any(pm == p)]
    return pidxs or [0]


def _draw_sector(xy, xy_edges, label_xy, labels) -> None:
    """White fundamental-sector outline + ``[hkl]`` corner labels on a PlotXY."""
    xy.plot(xy_edges[:, 0], xy_edges[:, 1], color="#ffffff", linewidth=1.5)
    for (lx, ly), txt in zip(np.asarray(label_xy, dtype=float), labels):
        xy.text(float(lx), float(ly), str(txt), color="#ffffff", fontsize=12)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D · Points — the stereographic scatter
# ─────────────────────────────────────────────────────────────────────────────

def build_ipf_points_2d_figure(result, direction: str = "z", *,
                               max_points: int = POINTS2D_MAX):
    """Build the 2-D IPF POINT scatter → ``(fig, fig_id, html, panels)``.

    One axis per phase present in the map: every position's best-match crystal
    direction, folded into that phase's fundamental sector, stereographically
    projected and drawn in its own IPF colour — the flat counterpart of the 3-D
    sphere cloud. ``panels`` maps ``phase_index → {"xy", "marker"}`` so the pick
    handler can move the highlight without rebuilding the figure.
    """
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from orix.quaternion import Rotation

    from spyde.actions.ipf_density import _sector_limits
    from spyde.drawing.plots.plot import finalize_figure_html
    from spyde.signals.orientation_map import ipf_triangle_xy, ipf_xy_for_rotations

    om = _as_orientation_map(result)
    quats = np.asarray(om.quats)[..., 0, :].reshape(-1, 4)
    phase_map = np.asarray(om.phase_map()).reshape(-1)
    rgb_all = np.asarray(om.ipf_color_map(direction)).reshape(-1, 3)
    pidxs = _present_phases(om)

    fig, axes = apl.subplots(1, len(pidxs))
    arr = np.array(axes, dtype=object).ravel()
    panels: dict[int, dict] = {}
    for ax, pidx in zip(arr, pidxs):
        phase = om.orix_phase(pidx)
        sel = phase_map == pidx
        q = quats[sel] if np.any(sel) else quats
        c = rgb_all[sel] if np.any(sel) else rgb_all
        if len(q) > max_points:                       # uniform subsample
            stride = int(np.ceil(len(q) / max_points))
            q, c = q[::stride], c[::stride]
        x, y = ipf_xy_for_rotations(Rotation(q), phase, direction)

        xy_edges, label_xy, labels = ipf_triangle_xy(phase)
        xlim, ylim = _sector_limits(xy_edges)
        xy = ax.axes2d(xlim=xlim, ylim=ylim, aspect="equal")
        # Transparent stroke: at ~5 px an outline would swamp the fill colour,
        # and the fill IS the information here (the IPF key colour).
        xy.scatter(x, y, s=POINTS2D_SIZE, c=_hex_colors(c),
                   edgecolors="rgba(0,0,0,0)")
        _draw_sector(xy, xy_edges, label_xy, labels)
        # The picked-orientation marker: one point, moved in place by
        # IpfWindowController.show_orientation (never re-emitted).
        cx = float(np.mean(np.asarray(xy_edges)[:, 0]))
        cy = float(np.mean(np.asarray(xy_edges)[:, 1]))
        marker = xy.scatter([cx], [cy], s=13, c=["#ffffff"], edgecolors="#000000")
        panels[int(pidx)] = {"xy": xy, "marker": marker}
        if len(pidxs) > 1:
            try:
                ax.set_title(str(getattr(phase, "name", "") or f"phase {pidx}"))
            except Exception as e:
                logger.debug("set_title on IPF points panel failed: %s", e)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html, panels


# ─────────────────────────────────────────────────────────────────────────────
# 3-D · Heatmap — the density field sampled ON the sphere
# ─────────────────────────────────────────────────────────────────────────────

def _density_sphere_grid(om, pidx: int, direction: str, *,
                         resolution: float, sigma: float, cmap: str):
    """``(X, Y, Z, rgba (H,W,4) uint8)`` — the phase's inverse-pole density as a
    SURFACE grid on the unit sphere plus the texture that paints it.

    ``pole_density_function`` bins the crystal directions on an EQUAL-AREA grid
    in stereographic coordinates. Each cell centre inverse-projects to the unit
    vector it came from, and because the grid keeps its ``(H, W)`` shape here
    (rather than being flattened to a point list) it *is* a surface mesh —
    which is what lets the density be painted on as a skin.

    The fundamental sector is not rectangular and the grid is, so the mask
    travels in the texture's **alpha channel** rather than in the geometry:
    cells outside the sector, and cells whose density is undefined, get
    ``alpha = 0``. The surface is built everywhere and simply not painted
    there. Keeping the geometry complete matters — a NaN vertex would tear the
    mesh, and anyplotlib's textured-surface pipeline blends per-texel alpha
    (``srcFactor: 'src-alpha'``), so an unpainted cell costs nothing but is
    genuinely invisible.

    Returns ``None`` when nothing survives.
    """
    from anyplotlib._utils import _build_colormap_lut
    from orix.measure import pole_density_function
    from orix.projections import InverseStereographicProjection
    from orix.quaternion import Rotation

    from spyde.signals.orientation_map import _direction_vector

    phase = om.orix_phase(pidx)
    phase_map = np.asarray(om.phase_map()).reshape(-1)
    quats = np.asarray(om.quats)[..., 0, :].reshape(-1, 4)
    sel = phase_map == pidx
    q = quats[sel] if np.any(sel) else quats

    t = Rotation(q) * _direction_vector(direction)
    hist, (x, y) = pole_density_function(
        t, symmetry=phase.point_group, resolution=resolution, sigma=sigma,
        log=False, hemisphere="upper")

    xc = np.asarray(x, dtype=float)
    yc = np.asarray(y, dtype=float)
    hc = np.ma.filled(np.ma.asarray(hist, dtype=float), np.nan)
    if xc.shape[0] == hc.shape[0] + 1 and xc.shape[1] == hc.shape[1] + 1:
        xc = 0.25 * (xc[:-1, :-1] + xc[1:, :-1] + xc[:-1, 1:] + xc[1:, 1:])
        yc = 0.25 * (yc[:-1, :-1] + yc[1:, :-1] + yc[:-1, 1:] + yc[1:, 1:])
    elif xc.shape != hc.shape:
        logger.debug("IPF 3-D density grid/hist mismatch: x=%s hist=%s",
                     xc.shape, hc.shape)
        return None

    shape = hc.shape
    xr, yr, hr = np.ravel(xc), np.ravel(yc), np.ravel(hc)
    vec = InverseStereographicProjection().xy2vector(xr, yr)
    inside = np.asarray(vec < phase.point_group.fundamental_sector).reshape(-1)
    xyz = np.asarray(vec.data, dtype=np.float64).reshape(-1, 3)

    # A cell can fall outside the projection's unit disk, where the inverse is
    # undefined. Those vertices must still be FINITE or the mesh tears, so give
    # them a harmless point on the sphere and mask them in alpha with the rest.
    bad_xyz = ~np.isfinite(xyz).all(axis=1)
    if bad_xyz.any():
        xyz[bad_xyz] = (0.0, 0.0, 1.0)
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = xyz / np.where(norm > 0, norm, 1.0)

    painted = inside & np.isfinite(hr) & ~bad_xyz
    if not np.any(painted):
        return None

    vals = np.where(painted, hr, 0.0)
    hi = float(np.nanmax(vals[painted])) or 1.0
    lut = np.asarray(_build_colormap_lut(cmap), dtype=np.uint8)      # (256, 3)
    idx = np.rint(np.clip(vals / hi, 0.0, 1.0) * 255).astype(np.intp)
    rgba = np.zeros((idx.size, 4), dtype=np.uint8)
    rgba[:, :3] = lut[idx]
    rgba[:, 3] = np.where(painted, 255, 0).astype(np.uint8)

    X = xyz[:, 0].reshape(shape)
    Y = xyz[:, 1].reshape(shape)
    Z = xyz[:, 2].reshape(shape)
    return X, Y, Z, rgba.reshape(shape + (4,))


def build_ipf_density_3d_figure(result, direction: str = "z", *,
                                resolution: float = DENSITY3D_RESOLUTION,
                                sigma: float = 5.0, cmap: str = "fire"):
    """Build the 3-D IPF DENSITY heatmap → ``(fig, fig_id, html, plots)``.

    The inverse pole density function drawn *on the sphere*: the fundamental
    sector's equal-area density cells lifted back to unit vectors and rendered
    as a dense IPF-density-coloured point patch inside the reference sphere, one
    axis per phase. ``plots`` maps ``phase_index → Plot3D``.

    **A textured skin, not a point patch** (anyplotlib >= 0.6.0). Earlier
    versions' ``plot_surface`` colour-mapped by the Z COORDINATE, so a sphere
    patch could only be coloured by height and the density had to be faked with
    a dense point cloud — visibly a dot grid at any real zoom, and it cost one
    vertex per cell. ``Plot3D.set_texture`` paints an image across the mesh
    instead, so the density is now a continuous surface: the grid carries the
    geometry and the raster carries the colour.

    The mapping is parametric — image column ``j`` to grid column ``j`` — and
    :func:`_density_sphere_grid` returns both at the same ``(H, W)``, so they
    line up by construction with no ``uv`` needed.
    """
    import anyplotlib as apl
    import anyplotlib._electron as _electron

    from spyde.actions.ipf_view import IPF3D_BOUNDS, IPF3D_ZOOM
    from spyde.drawing.plots.plot import finalize_figure_html

    om = _as_orientation_map(result)
    pidxs = _present_phases(om)
    built = []
    for pidx in pidxs:
        grid = _density_sphere_grid(om, pidx, direction, resolution=resolution,
                                    sigma=sigma, cmap=cmap)
        if grid is not None:
            built.append((pidx, grid))
    if not built:
        return None

    fig, axes = apl.subplots(1, len(built))
    arr = np.array(axes, dtype=object).ravel()
    plots: dict[int, object] = {}
    for ax, (pidx, (X, Y, Z, rgba)) in zip(arr, built):
        # Aim at the PAINTED cells only — the grid spans the whole hemisphere
        # but the sector is a small part of it, and aiming at the full grid's
        # centroid would open the view on blank sphere.
        lit = rgba[..., 3] > 0
        az, el = _aim_at(np.stack([X[lit], Y[lit], Z[lit]], axis=1))
        p3d = ax.plot_surface(
            X, Y, Z,
            x_label="[100]", y_label="[010]", z_label="[001]",
            bounds=IPF3D_BOUNDS, zoom=IPF3D_ZOOM, gpu=True,
            azimuth=az, elevation=el,
        )
        try:
            # cull_backfaces stays FALSE: this is an open patch, not a closed
            # solid, so its far side is exactly what you look at after the
            # sphere rotates a picked orientation to the centre.
            p3d.set_texture(rgba, cull_backfaces=False, shade=False)
        except Exception as e:
            logger.debug("texturing the IPF density sphere failed: %s", e)
        try:
            p3d.set_sphere(1.0)
        except Exception as e:
            logger.debug("setting IPF density sphere failed: %s", e)
        plots[int(pidx)] = p3d
        if len(built) > 1:
            try:
                ax.set_title(str(getattr(om.orix_phase(pidx), "name", "")
                                 or f"phase {pidx}"))
            except Exception as e:
                logger.debug("set_title on IPF density-3D panel failed: %s", e)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html, plots


# ─────────────────────────────────────────────────────────────────────────────
# The window controller
# ─────────────────────────────────────────────────────────────────────────────

class IpfWindowController:
    """Owns the IPF EXPLORER window (window 2): its four figures, the X/Y/Z
    direction state, and the crosshair-driven orientation highlight.

    Registered with ``session.register_window_controller`` so ✕-closing the
    window tears it down (the WindowController protocol in
    :mod:`spyde.actions.registry`) and so ``ipf_set_direction`` — dispatched
    against a BARE figure window with no registered ``Plot`` — can still find it.
    """

    def __init__(self, session, window_id: int, tree, result,
                 direction: str = "z", title: str = "IPF"):
        self.session = session
        self.window_id = int(window_id)
        self.tree = tree
        self.result = result
        self.direction = str(direction).lower()
        self.title = title
        self._p3d_points: dict[int, object] = {}     # phase → Plot3D (points)
        self._p3d_density: dict[int, object] = {}    # phase → Plot3D (heatmap)
        self._panels_2d: dict[int, dict] = {}        # phase → {"xy", "marker"}
        self._closed = False

    # ── emit ────────────────────────────────────────────────────────────────
    def _emit(self, fig, fig_id: str, html: str, view: str, title: str) -> None:
        from de_shell.ipc import emit
        keep_alive(self.window_id, fig)
        emit({
            "type": "figure", "fig_id": fig_id, "window_id": self.window_id,
            "html": html, "title": title, "is_navigator": False, "view": view,
        })

    def emit_all(self) -> bool:
        """(Re)build and emit all four toggle figures. Returns True if the 3-D
        point cloud — the one view that must exist — came up."""
        ok = self._emit_points_3d()
        self._emit_points_2d()
        self._emit_heatmap_2d()
        self._emit_heatmap_3d()
        return ok

    def _emit_points_3d(self) -> bool:
        from spyde.actions.ipf_view import build_ipf_3d_figure, ipf_scene_data
        scene = ipf_scene_data(self.result, self.direction)
        if scene is None:
            return False
        xyz, rgb, _params = scene
        try:
            fig, fig_id, html, p3d = build_ipf_3d_figure(xyz, rgb)
        except Exception as e:
            logger.debug("IPF 3-D points figure failed: %s", e)
            return False
        self._p3d_points = {0: p3d}
        # Legacy single-sphere handle: the map window's pick handler and the
        # Report scene3d snapshot both look for tree._ipf_p3d.
        if self.tree is not None:
            self.tree._ipf_p3d = p3d
        self._emit(fig, fig_id, html, VIEW_POINTS_3D, "IPF (3D points)")
        return True

    def _emit_points_2d(self) -> bool:
        try:
            fig, fig_id, html, panels = build_ipf_points_2d_figure(
                self.result, self.direction)
        except Exception as e:
            logger.debug("IPF 2-D points figure failed: %s", e)
            return False
        self._panels_2d = panels
        self._emit(fig, fig_id, html, VIEW_POINTS_2D, "IPF (2D points)")
        return True

    def _emit_heatmap_2d(self) -> bool:
        from spyde.actions.ipf_density import emit_ipf_density
        return bool(emit_ipf_density(self.window_id, self.result, self.direction))

    def _emit_heatmap_3d(self) -> bool:
        try:
            built = build_ipf_density_3d_figure(self.result, self.direction)
        except Exception as e:
            logger.debug("IPF 3-D density figure failed: %s", e)
            return False
        if built is None:
            return False
        fig, fig_id, html, plots = built
        self._p3d_density = plots
        self._emit(fig, fig_id, html, VIEW_HEATMAP_3D, "IPF density (3D)")
        return True

    # ── interaction ─────────────────────────────────────────────────────────
    def set_direction(self, direction: str) -> None:
        """Re-colour every view for sample direction x | y | z."""
        d = str(direction).lower()
        if d not in ("x", "y", "z") or self._closed:
            return
        self.direction = d
        if self.tree is not None:
            self.tree._ipf_direction = d
        self.emit_all()

    def show_orientation(self, iy: int, ix: int) -> bool:
        """The crosshair landed on scan pixel ``(iy, ix)``: mark that orientation
        on all four views and **rotate both spheres to face it**.

        In-place pushes only (``set_highlight`` / ``set_view`` / ``marker.set``)
        so the camera the user orbited to survives — the figures are never
        re-emitted for a pick.
        """
        if self._closed:
            return False
        om = _as_orientation_map(self.result)
        try:
            v = om.ipf_xyz(int(iy), int(ix), 0, self.direction)[0]
        except Exception as e:
            logger.debug("resolving picked orientation failed: %s", e)
            return False
        try:
            pidx = int(np.asarray(om.phase_map())[int(iy), int(ix)])
        except Exception:
            pidx = 0

        az, el = face_camera(v)
        for handles in (self._p3d_points, self._p3d_density):
            for key, p3d in handles.items():
                # The points sphere is a single all-phase cloud (key 0); the
                # density sphere is one panel PER phase — only aim the panel the
                # picked pixel belongs to.
                if handles is self._p3d_density and key != pidx:
                    continue
                try:
                    p3d.set_highlight(float(v[0]), float(v[1]), float(v[2]),
                                      color="#ffffff", size=11)
                    p3d.set_view(azimuth=az, elevation=el)
                except Exception as e:
                    logger.debug("updating IPF sphere highlight failed: %s", e)

        panel = self._panels_2d.get(pidx) or next(iter(self._panels_2d.values()), None)
        if panel is not None:
            try:
                xy2 = om.ipf_xy(int(iy), int(ix), self.direction)[0][0]
                panel["marker"].set(offsets=[[float(xy2[0]), float(xy2[1])]])
            except Exception as e:
                logger.debug("updating IPF 2-D marker failed: %s", e)
        return True

    # ── WindowController protocol ───────────────────────────────────────────
    @property
    def source_plot(self):
        """The MAP window's Plot — this window's stand-in wherever a real
        ``Plot``/tree is required.

        The Report Builder resolves a dragged window to a Plot (its rebind
        handle is a ``SignalRef`` to a tree), and a bare figure window has none;
        without this, dragging the 3-D IPF pill into a report failed with
        "source window not found"."""
        return next(iter(getattr(self.tree, "signal_plots", []) or []), None)

    def handle_action(self, name: str, payload: dict) -> bool:
        if name == "ipf_set_direction":
            self.set_direction(str(payload.get("direction", "z")))
            return True
        return False

    def close(self) -> None:
        self._closed = True
        self._p3d_points = {}
        self._p3d_density = {}
        self._panels_2d = {}
        if self.tree is not None and getattr(self.tree, "_ipf_window", None) is self:
            try:
                self.tree._ipf_window = None
            except Exception as e:
                logger.debug("clearing tree._ipf_window failed: %s", e)


def open_ipf_window(session, tree, result, direction: str = "z", *,
                    title: str | None = None):
    """Open the IPF EXPLORER window for *tree*'s orientation *result*.

    A bare-figure window (no ``Plot``) holding the four toggle views, wired to a
    :class:`IpfWindowController` registered on the session so ✕-close and
    ``ipf_set_direction`` both reach it. Cached on ``tree._ipf_window``; a second
    call reuses the existing window rather than piling up duplicates.
    Returns the controller, or None.
    """
    if session is None:
        return None
    existing = getattr(tree, "_ipf_window", None)
    if existing is not None and not getattr(existing, "_closed", False):
        existing.result = result
        existing.set_direction(direction)
        return existing

    base = ""
    try:
        base = str(getattr(tree, "root").metadata.get_item("General.title", "") or "")
    except Exception as e:
        logger.debug("resolving IPF window title failed: %s", e)
    # "<src> — Orientation (IPF-Z)" is the MAP window's name; window 2 is the
    # inverse pole figure OF that map, so drop the parenthetical.
    base = base.split(" — Orientation")[0] if " — Orientation" in base else base
    wid = session.next_window_id()
    win_title = title or (f"{base} — IPF" if base else "IPF")
    ctrl = IpfWindowController(session, wid, tree, result, direction,
                               title=win_title)
    if not ctrl.emit_all():
        return None
    try:                                # `view`-tagged figures never rename a
        from de_shell.ipc import emit    # window, so name it explicitly
        emit({"type": "window_title", "window_ids": [wid], "title": win_title})
    except Exception as e:
        logger.debug("titling the IPF window failed: %s", e)
    try:
        session.register_window_controller(wid, ctrl)
    except Exception as e:
        logger.debug("registering IPF window controller failed: %s", e)
    try:
        tree._ipf_window = ctrl
    except Exception as e:
        logger.debug("caching IPF window on the tree failed: %s", e)
    return ctrl

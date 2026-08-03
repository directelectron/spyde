"""orientation_embed.py — embed a FULL orientation-mapping result in an HTML report.

The sibling of :mod:`vectors_embed`, and the same bargain: an orientation result
is small enough to inline into a self-contained page, and rich enough to drive
the whole IPF explorer client-side. Per nav position it is one best-match
orientation — a stereographic ``(x, y)`` in the fundamental sector, a reduced
crystal direction on the unit sphere, and an IPF colour. 20 bytes per position
per sample direction.

The page is the app's TWO WINDOWS side by side:

- **the map** (window 1) — the IPF colour map with a draggable **crosshair**,
  its own mount, always on screen.
- **the explorer** (window 2) — ONE view at a time, chosen by the app's two
  independent toggle pairs, ``[2D | 3D]`` and ``[Points | Heatmap]``:

  =========  ==============================  ==============================
  \\          Points                          Heatmap
  =========  ==============================  ==============================
  **2D**     sector scatter, IPF-coloured    inverse pole density raster
  **3D**     unit-sphere scatter             density painted on the sphere
  =========  ==============================  ==============================

Picking on the map marks that orientation on the 2-D scatter and turns BOTH
spheres to face it, exactly as ``IpfWindowController.show_orientation`` does.
**X / Y / Z** re-colours every view.

Four explorer views means four figures, each MOUNTED LAZILY the first time its
toggle is picked — a figure mounted into a ``display:none`` box measures 0×0 and
draws nothing, and mounting all four up front would pay for three the reader may
never look at. The map is a separate mount from the explorer so that swapping
the explorer never resets the crosshair.

Nothing recomputes: a pick is an index into the packed arrays, so the page needs
no orix, no backend and no network, and still works years later. The density
fields are the one thing that cannot be recomputed client-side (they are orix's
``pole_density_function``, an equal-area binning in the fundamental sector — a
plain histogram of the stereographic coordinates is a different, uncorrected
quantity), so they are computed at build time and shipped as pixels. Their
GEOMETRY is direction-independent — the same grid bins any direction — so only
the raster and the texture travel per direction, not the mesh.

Packing (little-endian, one base64 blob) — see :func:`pack_orientation`:
    uint8   phase[M]                      per-position best-match phase
    per direction d in (x, y, z):
      uint8   rgb[M*3]                    IPF colour  → the map image
      float32 xy[M*2]                     sector (x, y) → the 2-D scatter
      float32 xyz[M*3]                    unit-sphere direction → the 3-D scatter
M = ny*nx. 20 bytes per position per direction (+ M for the phase map). The
density images do NOT ride the blob — they are PNG data URLs on the header, one
per phase per direction, because a density field is smooth and mostly
transparent and PNG crushes it.

Hooked into ``export_html._render_body`` (interactive mode) and the report
sidebar's live figure build, exactly like the vectors explorer.
"""
from __future__ import annotations

import base64
import html as _html
import json
import logging

import numpy as np

log = logging.getLogger(__name__)

DIRECTIONS = ("x", "y", "z")
VIEWS = ("2d-points", "2d-heat", "3d-points", "3d-heat")

# The embed refuses a scan bigger than this. 20 bytes/position/direction x 3
# directions = 60 B/position, so 2 M positions is ~120 MB packed — already far
# past what belongs in an HTML file, and base64 makes it 160 MB.
MAX_EMBED_POSITIONS = 2_000_000

# Cloud points actually DRAWN in the scatter views. The full per-position arrays
# stay in the packed blob (a pick indexes them), but scatter offsets ride the
# figure's own state as JSON text — at ~30 chars a point that is the single
# biggest thing in the page, and it is re-serialised on every direction switch.
# Same ceiling the live 2-D view uses (``ipf_window.POINTS2D_MAX``).
CLOUD_MAX = 30_000

# Density resolution. The 2-D raster is `res x res` RGBA (256 -> 262 kB per
# direction per phase, raw); the 3-D grid is coarser because it is a MESH, and
# every cell costs a vertex as well as a texel.
DENSITY_RASTER_RES = 256
DENSITY_2D_RESOLUTION = 2.0
DENSITY_3D_RESOLUTION = 3.0
DENSITY_SIGMA = 5.0
DENSITY_CMAP = "fire"

POINT_SIZE_2D = 5
POINT_SIZE_3D = 6.0
FIG_PX = 300                  # each panel's CSS height

_ESM_TEXT: "str | None" = None

# Memoized page per cell id, holding a STRONG reference to the result it was
# built from (identity-compared with `is`) — the vectors explorer's cache
# contract, and for the same reason: an int `id()` is reused once the original
# is collected, which would serve a stale page for a different result.
_EXPLORER_CACHE: "dict[str, tuple[object, str]]" = {}


def _esm_text() -> str:
    """anyplotlib's ESM, read once per process (it is ~400 KB and never
    changes)."""
    global _ESM_TEXT
    if _ESM_TEXT is None:
        from anyplotlib import embed as apl_embed
        _ESM_TEXT = apl_embed.esm_path().read_text(encoding="utf-8")
    return _ESM_TEXT


def clear_explorer_cache(cell_id: "str | None" = None) -> None:
    """Drop the memoized page(s) — one cell's, or all of them."""
    if cell_id is None:
        _EXPLORER_CACHE.clear()
    else:
        _EXPLORER_CACHE.pop(cell_id, None)


def _as_orientation_map(result):
    """Normalise a raw-OM or vector-OM result to a ``SpyDEOrientationMap``.

    Delegates to the live IPF window's own normaliser rather than repeating the
    rule, so a third producer only has to be taught to one of them. Lazy: this
    module is imported from the report handlers, and ``ipf_view`` pulls the
    figure registry."""
    from spyde.actions.ipf_view import _as_orientation_map as _norm
    return _norm(result)


def _hex_colors(rgb: np.ndarray) -> list:
    """(N,3) uint8 → ``#rrggbb`` strings (anyplotlib's per-point colour form)."""
    a = np.clip(np.asarray(rgb), 0, 255).astype(np.uint8)
    packed = (a[:, 0].astype(np.uint32) << 16) | (a[:, 1].astype(np.uint32) << 8) \
        | a[:, 2].astype(np.uint32)
    return ["#%06x" % int(p) for p in packed]


# ── packing ───────────────────────────────────────────────────────────────────

def _per_position_direction(om, direction: str):
    """``(rgb (M,3) uint8, xy (M,2) float32, xyz (M,3) float32)`` for every nav
    position, in nav raster order.

    Deliberately NOT ``om.ipf_sphere_points`` / ``om.ipf_color_map``: the former
    drops non-finite rows and subsamples, so its row i is not nav position i —
    and an index that does not mean "nav position" is exactly what a crosshair
    pick needs. Everything here stays full length and nav-aligned; a position
    with no valid orientation keeps a NaN row and the page skips it.
    """
    from orix.plot import IPFColorKeyTSL
    from orix.projections import StereographicProjection
    from orix.quaternion import Orientation, Rotation
    from spyde.signals.orientation_map import _direction_vector

    best_phase = np.asarray(om.phase_idx)[..., 0].reshape(-1)
    best_q = np.asarray(om.quats)[..., 0, :].reshape(-1, 4)
    m = best_q.shape[0]
    d = _direction_vector(direction)

    rgb = np.zeros((m, 3), dtype=np.uint8)
    xy = np.full((m, 2), np.nan, dtype=np.float32)
    xyz = np.full((m, 3), np.nan, dtype=np.float32)
    proj = StereographicProjection()
    for i in range(om.n_phases):
        mask = best_phase == i
        if not mask.any():
            continue
        pg = om.orix_phase(i).point_group
        rot = Rotation(best_q[mask])
        v = (rot * d).in_fundamental_sector(pg)
        xyz[mask] = np.asarray(v.data, dtype=np.float32)
        vx, vy = proj.vector2xy(v)
        xy[mask] = np.column_stack([np.atleast_1d(vx),
                                    np.atleast_1d(vy)]).astype(np.float32)
        key = IPFColorKeyTSL(pg.laue, direction=d)
        colors = key.orientation2color(Orientation(rot, symmetry=pg))
        rgb[mask] = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return rgb, xy, xyz


def _phase_geometry(om) -> list:
    """One record per phase PRESENT in the map: its sector outline, corner
    labels and axis limits — the static furniture every 2-D view draws."""
    from spyde.actions.ipf_density import _sector_limits
    from spyde.signals.orientation_map import ipf_triangle_xy

    present = sorted({int(p) for p in np.asarray(om.phase_map()).reshape(-1)})
    out = []
    for pidx in present:
        try:
            phase = om.orix_phase(pidx)
            edges, label_xy, labels = ipf_triangle_xy(phase)
            xlim, ylim = _sector_limits(np.asarray(edges))
        except Exception as e:
            log.debug("[report] sector geometry for phase %s failed: %s", pidx, e)
            continue
        e = np.asarray(edges, dtype=float)
        if e.size == 0:
            continue
        out.append({
            "idx": int(pidx),
            "name": str(getattr(phase, "name", "") or f"phase {pidx}"),
            "edges": e.tolist(),
            "labels": [{"x": float(p[0]), "y": float(p[1]), "text": str(t)}
                       for p, t in zip(np.asarray(label_xy, dtype=float), labels)],
            "xlim": [float(xlim[0]), float(xlim[1])],
            "ylim": [float(ylim[0]), float(ylim[1])],
        })
    return out


def _png_urls(arrays) -> "dict | None":
    """``{direction: "data:image/png;base64,…"}`` for one per-direction list of
    RGBA arrays, or None if any of them will not encode."""
    from anyplotlib._utils import _image_to_data_url
    out = {}
    for d, arr in zip(DIRECTIONS, arrays):
        try:
            out[d] = _image_to_data_url(np.ascontiguousarray(arr, dtype=np.uint8))
        except Exception as e:
            log.debug("[report] PNG-encoding a density image failed: %s", e)
            return None
    return out


def _density_2d(om, pidx: int, direction: str, xlim, ylim):
    """``(rgba (R,R,4) uint8, extent)`` — the phase's inverse pole density
    resampled onto the regular raster ``add_raster`` draws, or None."""
    from orix.measure import pole_density_function
    from orix.quaternion import Rotation
    from spyde.actions.ipf_density import _resample_density_to_raster
    from spyde.signals.orientation_map import _direction_vector

    phase_map = np.asarray(om.phase_map()).reshape(-1)
    quats = np.asarray(om.quats)[..., 0, :].reshape(-1, 4)
    sel = phase_map == pidx
    q = quats[sel] if np.any(sel) else quats
    t = Rotation(q) * _direction_vector(direction)
    hist, (x, y) = pole_density_function(
        t, symmetry=om.orix_phase(pidx).point_group,
        resolution=DENSITY_2D_RESOLUTION, sigma=DENSITY_SIGMA, log=False,
        hemisphere="upper")
    return _resample_density_to_raster(x, y, hist, xlim, ylim, DENSITY_CMAP,
                                       None, res=DENSITY_RASTER_RES)


def pack_orientation(result) -> "dict | None":
    """Pack the whole result into ``{"header": …, "b64": …, "arrays": …}``, or
    None when it is empty / over :data:`MAX_EMBED_POSITIONS`.

    ``arrays`` and ``density`` are kept for the figure builders (which need the
    same numbers to draw the initial views); only ``header`` and ``b64`` reach
    the page.
    """
    try:
        om = _as_orientation_map(result)
        ny, nx = (int(v) for v in om.nav_shape)
    except Exception as e:
        log.debug("[report] orientation result not packable: %s", e)
        return None
    m = ny * nx
    if m <= 0:
        return None
    if m > MAX_EMBED_POSITIONS:
        log.debug("[report] orientation embed refused: %d positions > cap %d",
                  m, MAX_EMBED_POSITIONS)
        return None

    phases = _phase_geometry(om)
    if not phases:
        return None
    try:
        phase = np.ascontiguousarray(
            np.clip(np.asarray(om.phase_map()).reshape(-1), 0, 255), dtype=np.uint8)
    except Exception as e:
        log.debug("[report] orientation phase map failed: %s", e)
        return None

    arrays: dict[str, tuple] = {}
    blocks = [phase.tobytes()]
    for d in DIRECTIONS:
        try:
            rgb, xy, xyz = _per_position_direction(om, d)
        except Exception as e:
            log.debug("[report] orientation direction %s failed: %s", d, e)
            return None
        arrays[d] = (rgb, xy, xyz)
        blocks.append(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        blocks.append(np.ascontiguousarray(xy, dtype=np.float32).tobytes())
        blocks.append(np.ascontiguousarray(xyz, dtype=np.float32).tobytes())

    # DENSITY. Both grids are a property of the point group and the resolution,
    # not of the sample direction — measured identical across x/y/z — so the
    # mesh and the raster extent are built once per phase and only the PIXELS
    # travel per direction.
    density: dict = {}
    for rec in phases:
        pidx = rec["idx"]
        rasters, extent = [], None
        for d in DIRECTIONS:
            try:
                got = _density_2d(om, pidx, d, rec["xlim"], rec["ylim"])
            except Exception as e:
                log.debug("[report] 2-D density (phase %s, %s) failed: %s",
                          pidx, d, e)
                got = None
            if got is None:
                rasters = []
                break
            rgba, extent = got
            rasters.append(np.ascontiguousarray(rgba, dtype=np.uint8))

        grid, textures = None, []
        from spyde.actions.ipf_window import _density_sphere_grid
        for d in DIRECTIONS:
            try:
                got = _density_sphere_grid(om, pidx, d,
                                           resolution=DENSITY_3D_RESOLUTION,
                                           sigma=DENSITY_SIGMA,
                                           cmap=DENSITY_CMAP)
            except Exception as e:
                log.debug("[report] 3-D density (phase %s, %s) failed: %s",
                          pidx, d, e)
                got = None
            if got is None:
                textures = []
                break
            X, Y, Z, tex = got
            grid = (X, Y, Z)
            textures.append(np.ascontiguousarray(tex, dtype=np.uint8))

        density[pidx] = {"rasters": rasters, "extent": extent,
                         "grid": grid, "textures": textures}
        # PNG data URLs, not raw bytes in the blob. A density field is smooth
        # and mostly transparent, so PNG crushes it — measured 73x on a flat
        # test image — and the 3-D texture wants a data URL anyway
        # (``Plot3D.set_texture`` → ``_image_to_data_url``). The 2-D raster
        # wants RAW RGBA base64 (``image_b64``), so the page decodes the PNG
        # through a canvas; that is a browser primitive and it is worth ~1 MB.
        rec["raster"] = _png_urls(rasters) if rasters else None
        rec["texture"] = _png_urls(textures) if textures else None

    # The cloud is uniformly strided, not sliced: a head-slice of a raster-order
    # scan is its top rows, which is a crop of the sample rather than a sample of
    # the orientations.
    stride = max(1, int(np.ceil(m / CLOUD_MAX)))
    header = {
        "nav": [ny, nx],
        "m": m,
        "dirs": list(DIRECTIONS),
        "phases": phases,
        "stride": stride,
        "views": list(VIEWS),
    }
    return {
        "header": header,
        "b64": base64.b64encode(b"".join(blocks)).decode("ascii"),
        "arrays": arrays,
        "density": density,
        "phase": phase,
    }


# ── figures ───────────────────────────────────────────────────────────────────

def _panel_ids(state: dict) -> list:
    """Panel ids in the state, in the order the axes were created.

    Read out of ``layout_json``'s ``panel_specs``, which IS that order. The
    state dict's own key order is not — and every per-phase update here maps
    panel *i* to phase *i*, so an unordered list silently paints one phase's
    density into another's panel."""
    ids = []
    try:
        layout = json.loads(state.get("layout_json") or "{}")
        for spec in (layout.get("panel_specs") or []):
            pid = spec.get("id")
            if pid:
                ids.append(str(pid))
    except Exception as e:
        log.debug("[report] reading the figure layout failed: %s", e)
    if ids:
        return ids
    return [k[len("panel_"):-len("_json")] for k in state
            if k.startswith("panel_") and k.endswith("_json")]


def _figure_state(fig) -> dict:
    from anyplotlib import embed as apl_embed
    return apl_embed.figure_state(fig)


def _build_map_figure(payload):
    """The IPF MAP (window 1): the colour image plus the crosshair that drives
    everything. Its own figure, so swapping the explorer never resets it."""
    import anyplotlib as apl

    hdr = payload["header"]
    ny, nx = hdr["nav"]
    rgb = payload["arrays"]["z"][0]
    aspect = min(3.0, max(1 / 3.0, (nx / ny) if ny else 1.0))
    fig, axes = apl.subplots(1, 1, figsize=(int(round(FIG_PX * aspect)) + 12,
                                            FIG_PX))
    ax = axes[0][0] if isinstance(axes, list) else axes
    p = ax.imshow(np.asarray(rgb, dtype=np.uint8).reshape(ny, nx, 3))
    p.add_widget("crosshair", color="#ffffff", cx=nx / 2, cy=ny / 2)
    state = _figure_state(fig)
    ids = _panel_ids(state)
    return (state, ids[0]) if ids else None


def _sector_furniture(xyp, rec) -> None:
    """The white sector outline + ``[hkl]`` corner labels every 2-D view draws
    (``ipf_window._draw_sector``, which this deliberately matches)."""
    edges = np.asarray(rec["edges"], dtype=float)
    xyp.plot(edges[:, 0], edges[:, 1], color="#ffffff", linewidth=1.5)
    for lbl in rec["labels"]:
        try:
            xyp.text(float(lbl["x"]), float(lbl["y"]), str(lbl["text"]),
                     color="#ffffff", fontsize=12)
        except Exception as e:
            log.debug("[report] sector label failed: %s", e)


def _build_points_2d(payload):
    """``[2D] · [Points]`` — the sector scatter, one panel per present phase,
    each point in its own IPF colour, plus the white pick marker."""
    import anyplotlib as apl

    hdr = payload["header"]
    stride = int(hdr["stride"])
    rgb, xy, _xyz = payload["arrays"]["z"]
    phase = payload["phase"]
    recs = hdr["phases"]

    fig, axes = apl.subplots(1, len(recs), figsize=(FIG_PX * len(recs), FIG_PX))
    arr = np.array(axes, dtype=object).ravel()
    for ax, rec in zip(arr, recs):
        sel = np.flatnonzero((phase == rec["idx"])
                             & np.isfinite(xy).all(axis=1))[::stride]
        xyp = ax.axes2d(xlim=tuple(rec["xlim"]), ylim=tuple(rec["ylim"]),
                        aspect="equal")
        # Transparent stroke: at ~5 px an outline swamps the fill, and the fill
        # IS the information here (the IPF key colour).
        xyp.scatter(xy[sel, 0], xy[sel, 1], s=POINT_SIZE_2D,
                    c=_hex_colors(rgb[sel]), edgecolors="rgba(0,0,0,0)")
        _sector_furniture(xyp, rec)
        seed = sel[0] if len(sel) else 0
        xyp.scatter([float(xy[seed, 0])], [float(xy[seed, 1])], s=13,
                    c=["#ffffff"], edgecolors="#000000")
        if len(recs) > 1:
            try:
                ax.set_title(rec["name"])
            except Exception as e:
                log.debug("[report] phase title failed: %s", e)
    state = _figure_state(fig)
    return state, _panel_ids(state)


def _build_heat_2d(payload):
    """``[2D] · [Heatmap]`` — the inverse pole density raster, one panel per
    phase. The image bytes are swapped per direction from the packed blob."""
    import anyplotlib as apl

    hdr = payload["header"]
    recs = [r for r in hdr["phases"] if r.get("raster")]
    if not recs:
        return None
    fig, axes = apl.subplots(1, len(recs), figsize=(FIG_PX * len(recs), FIG_PX))
    arr = np.array(axes, dtype=object).ravel()
    for ax, rec in zip(arr, recs):
        den = payload["density"][rec["idx"]]
        xyp = ax.axes2d(xlim=tuple(rec["xlim"]), ylim=tuple(rec["ylim"]),
                        aspect="equal")
        edges = np.asarray(rec["edges"], dtype=float)
        # Clipped to the curved sector boundary, like the live view: the
        # equal-area grid is coarse at the edge and would otherwise spill.
        xyp.add_raster(den["rasters"][DIRECTIONS.index("z")],
                       extent=den["extent"], clip_path=edges, smooth=True,
                       name="density")
        _sector_furniture(xyp, rec)
        if len(recs) > 1:
            try:
                ax.set_title(rec["name"])
            except Exception as e:
                log.debug("[report] phase title failed: %s", e)
    state = _figure_state(fig)
    return state, _panel_ids(state)


def _build_points_3d(payload):
    """``[3D] · [Points]`` — the unit-sphere scatter, aimed at the cloud."""
    import anyplotlib as apl
    from spyde.actions.ipf_view import IPF3D_BOUNDS, IPF3D_ZOOM
    from spyde.actions.ipf_window import _aim_at

    hdr = payload["header"]
    stride = int(hdr["stride"])
    rgb, _xy, xyz = payload["arrays"]["z"]
    sel = np.flatnonzero(np.isfinite(xyz).all(axis=1))[::stride]
    if not len(sel):
        return None
    fig, axes = apl.subplots(1, 1, figsize=(FIG_PX, FIG_PX))
    ax = axes[0][0] if isinstance(axes, list) else axes
    az, el = _aim_at(xyz[sel])
    p3d = ax.scatter3d(
        xyz[sel, 0], xyz[sel, 1], xyz[sel, 2],
        colors=np.clip(np.asarray(rgb[sel], dtype=np.float32) / 255.0, 0.0, 1.0),
        point_size=POINT_SIZE_3D, x_label="[100]", y_label="[010]",
        z_label="[001]", bounds=IPF3D_BOUNDS, zoom=IPF3D_ZOOM, gpu=True,
        azimuth=az, elevation=el,
    )
    try:
        p3d.set_sphere(1.0)
    except Exception as e:
        log.debug("[report] IPF reference sphere failed: %s", e)
    seed = int(sel[0])
    p3d.set_highlight(float(xyz[seed, 0]), float(xyz[seed, 1]),
                      float(xyz[seed, 2]), color="#ffffff", size=11)
    state = _figure_state(fig)
    return state, _panel_ids(state)


def _build_heat_3d(payload):
    """``[3D] · [Heatmap]`` — the density painted onto the sphere as a textured
    mesh (``ipf_window.build_ipf_density_3d_figure``'s skin, rebuilt here so the
    state can be serialized instead of registered with the app)."""
    import anyplotlib as apl
    from spyde.actions.ipf_view import IPF3D_BOUNDS, IPF3D_ZOOM
    from spyde.actions.ipf_window import _aim_at

    hdr = payload["header"]
    recs = [r for r in hdr["phases"] if r.get("texture")]
    if not recs:
        return None
    fig, axes = apl.subplots(1, len(recs), figsize=(FIG_PX * len(recs), FIG_PX))
    arr = np.array(axes, dtype=object).ravel()
    for ax, rec in zip(arr, recs):
        den = payload["density"][rec["idx"]]
        X, Y, Z = den["grid"]
        tex = den["textures"][DIRECTIONS.index("z")]
        # Aim at the PAINTED cells only: the grid spans the hemisphere but the
        # sector is a small part of it, so the full grid's centroid opens the
        # view on blank sphere.
        lit = tex[..., 3] > 0
        az, el = _aim_at(np.stack([X[lit], Y[lit], Z[lit]], axis=1))
        p3d = ax.plot_surface(X, Y, Z, x_label="[100]", y_label="[010]",
                              z_label="[001]", bounds=IPF3D_BOUNDS,
                              zoom=IPF3D_ZOOM, gpu=True,
                              azimuth=az, elevation=el)
        try:
            # cull_backfaces stays FALSE: this is an open patch, not a closed
            # solid, so its far side is exactly what you look at once a picked
            # orientation rotates to the centre.
            p3d.set_texture(tex, cull_backfaces=False, shade=False)
        except Exception as e:
            log.debug("[report] texturing the density sphere failed: %s", e)
        try:
            p3d.set_sphere(1.0)
        except Exception as e:
            log.debug("[report] density reference sphere failed: %s", e)
        if len(recs) > 1:
            try:
                ax.set_title(rec["name"])
            except Exception as e:
                log.debug("[report] phase title failed: %s", e)
    state = _figure_state(fig)
    return state, _panel_ids(state)


def _build_all(payload) -> "dict | None":
    """Every figure the page can show → ``{"map": …, "2d-points": …, …}`` of
    ``{"state", "panels"}``. A view that cannot be built is simply absent and
    its toggle is disabled — a scan with no density (scipy missing, an empty
    sector) still gets the scatters."""
    try:
        import anyplotlib  # noqa: F401
    except Exception as e:
        log.debug("[report] anyplotlib unavailable for orientation embed: %s", e)
        return None

    built: dict = {}
    m = _build_map_figure(payload)
    if m is None:
        log.debug("[report] orientation embed: the map figure did not build")
        return None
    built["map"] = {"state": m[0], "panels": [m[1]]}

    for name, fn in (("2d-points", _build_points_2d),
                     ("2d-heat", _build_heat_2d),
                     ("3d-points", _build_points_3d),
                     ("3d-heat", _build_heat_3d)):
        try:
            got = fn(payload)
        except Exception as e:
            log.debug("[report] orientation view %s failed: %s", name, e)
            got = None
        if got is not None:
            built[name] = {"state": got[0], "panels": list(got[1])}
    if "2d-points" not in built:
        log.debug("[report] orientation embed: no 2-D scatter, refusing")
        return None
    return built


# Dark theme, matching vectors_embed (Catppuccin Mocha) so the two explorers
# read as the same product inside one report.
_EXPLORER_CSS = """
:root { color-scheme: dark; }
body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
       background: #1e1e2e; color: #cdd6f4; font-size: 12px; }
.ox-wrap { display: flex; flex-direction: column; gap: 8px; padding: 8px;
           align-items: flex-start; }
.ox-wrap h4 { margin: 0; font-size: 12px; font-weight: 600; color: #bac2de; }
.ox-row { display: flex; gap: 8px; align-items: flex-start;
          background: #181825; border-radius: 6px; padding: 4px; }
.ox-slot { position: relative; }
.ox-slot > div { display: none; }
.ox-slot > div.ox-on { display: block; }
.ox-controls { display: flex; gap: 14px; align-items: center; font-size: 11px;
               flex-wrap: wrap; }
.ox-seg { display: inline-flex; align-items: center; gap: 6px; }
.ox-seg-label { color: #6c7086; }
.ox-seg-group { display: inline-flex; border: 1px solid #313244;
                border-radius: 999px; overflow: hidden; }
.ox-seg-btn { appearance: none; border: 0; padding: 3px 11px; cursor: pointer;
              background: #1e1e2e; color: #a6adc8; font: inherit; }
.ox-seg-btn[aria-pressed="true"] { background: #89b4fa; color: #11111b;
                                   font-weight: 600; }
.ox-seg-btn[disabled] { opacity: 0.35; cursor: not-allowed; }
.ox-status-dot { width: 7px; height: 7px; border-radius: 50%;
                 background: #a6e3a1; display: inline-block; }
.ox-meta { color: #6c7086; }
"""

_EXPLORER_JS_TMPL = r"""
const HDR = JSON.parse(document.getElementById('ox-header').textContent);
const B64 = document.getElementById('ox-data').textContent.trim();
const FIGS = JSON.parse(document.getElementById('ox-figs').textContent);
const ESM = document.getElementById('ox-esm').textContent;

const esmUrl = URL.createObjectURL(new Blob([ESM], { type: 'text/javascript' }));
const { mount } = await import(esmUrl);

const NY = HDR.nav[0], NX = HDR.nav[1], M = HDR.m, STRIDE = HDR.stride;
const DIRS = HDR.dirs, PHASES = HDR.phases;

const ab = await (await fetch('data:application/octet-stream;base64,' + B64))
  .arrayBuffer();
let off = 0;
const PHASE = new Uint8Array(ab, off, M); off += M;
// Per direction: rgb(uint8 M*3) | xy(float32 M*2) | xyz(float32 M*3). Float
// views need their byte offset 4-aligned; M*3 uint8 bytes need not be, so the
// float blocks are COPIED out rather than viewed in place.
const D = {};
for (const d of DIRS) {
  const rgb = new Uint8Array(ab, off, M * 3); off += M * 3;
  const xy = new Float32Array(ab.slice(off, off + 4 * M * 2)); off += 4 * M * 2;
  const xyz = new Float32Array(ab.slice(off, off + 4 * M * 3)); off += 4 * M * 3;
  D[d] = { rgb, xy, xyz };
}
// The density images are PNG data URLs on the header (see _png_urls), not raw
// bytes in the blob: a density field is smooth and mostly transparent, so PNG
// crushes it, and the 3-D texture wants a data URL anyway.

/** A PNG data URL → RAW RGBA base64, which is what a raster marker's
 *  `image_b64` is (add_raster base64s the array, it does not PNG it). Decoded
 *  through a canvas — a browser primitive — and memoised, because a reader
 *  flipping X/Y/Z repeatedly should not re-decode. */
const _rawCache = new Map();
async function pngToRawB64(url) {
  const hit = _rawCache.get(url);
  if (hit) return hit;
  const img = await new Promise((res, rej) => {
    const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url;
  });
  const cv = document.createElement('canvas');
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  const cx = cv.getContext('2d');
  cx.drawImage(img, 0, 0);
  const d = cx.getImageData(0, 0, cv.width, cv.height).data;
  const out = { b64: b64(new Uint8Array(d.buffer)), w: cv.width, h: cv.height };
  _rawCache.set(url, out);
  return out;
}

let dir = 'z';                 // the direction the page was built for
let dim = '2d', style = 'points';
let pick = 0;                  // flat nav index of the marked position
const H = {};                  // view name -> mount handle (lazy)

function viewName() { return dim + '-' + (style === 'points' ? 'points' : 'heat'); }
function has(name) { return !!FIGS[name]; }

function b64(bytes) {
  let s = '';
  const CH = 0x8000;           // chunked: apply() blows the argument limit
  for (let i = 0; i < bytes.length; i += CH) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
  }
  return btoa(s);
}

// Patch through the LIVE model state (handle.get), never the page-load parse:
// re-applying the stale load-time json resets the fitted view and collapses the
// panel to zero width (the vectors explorer learned this the hard way).
function live(h, key, fallback) {
  try { return JSON.parse(h.get(key)); }
  catch (e) { return JSON.parse(fallback || '{}'); }
}
function panelKey(p) { return 'panel_' + p + '_json'; }
function geomKey(p) { return 'panel_' + p + '_geom'; }

/** The flat nav indices drawn in phase `pidx`'s cloud, in build order: that
 *  phase's finite rows, then strided. Rebuilt per direction because which rows
 *  are finite is a property of the direction's fold. */
function cloudIndices(d, pidx) {
  const xy = D[d].xy, out = [];
  for (let i = 0; i < M; i++) {
    if (PHASE[i] !== pidx) continue;
    if (Number.isFinite(xy[2 * i]) && Number.isFinite(xy[2 * i + 1])) out.push(i);
  }
  const strided = [];
  for (let i = 0; i < out.length; i += STRIDE) strided.push(out[i]);
  return strided;
}

function hex(rgb, i) {
  const s = (v) => v.toString(16).padStart(2, '0');
  return '#' + s(rgb[3 * i]) + s(rgb[3 * i + 1]) + s(rgb[3 * i + 2]);
}

/** A 2-D scatter panel's marker groups, in creation order: [0] the cloud, then
 *  the sector outline and the corner labels, and LAST the pick marker. Picked
 *  out by type and position rather than index so an added label cannot shift
 *  the lookup. */
function markerGroups(pj) {
  const pts = (pj.markers || []).filter((g) => g.type === 'points');
  return { cloud: pts[0] || null, pick: pts[pts.length - 1] || null };
}

// ── the explorer slot ────────────────────────────────────────────────────────
// Each view is its own figure, mounted the FIRST time it is shown. A figure
// mounted into a display:none box measures 0x0 and draws nothing, so mounting
// them all up front would produce three blank panels and pay for views the
// reader may never open.
function show(name) {
  for (const el of document.querySelectorAll('#ox-slot > div')) {
    el.classList.toggle('ox-on', el.dataset.view === name);
  }
  if (!H[name] && FIGS[name]) {
    const host = document.querySelector(`#ox-slot > div[data-view="${name}"]`);
    H[name] = mount(host, FIGS[name].state, {});
    // The freshly mounted view was built for IPF-Z; bring it up to the current
    // direction and pick before the reader sees it.
    requestAnimationFrame(() => {
      applyDirection(name);
      applyPick(name);
    });
  }
}

function applyDirection(name) {
  const h = H[name];
  if (!h) return;
  const panels = FIGS[name].panels, { rgb, xy, xyz } = D[dir];

  if (name === '2d-points') {
    panels.forEach((p, i) => {
      const rec = PHASES[i]; if (!rec) return;
      const idx = cloudIndices(dir, rec.idx);
      const pj = live(h, panelKey(p), FIGS[name].state[panelKey(p)]);
      const g = markerGroups(pj).cloud;
      if (!g) return;
      g.offsets = idx.map((k) => [xy[2 * k], xy[2 * k + 1]]);
      // `fill_color`, NOT `facecolors`: `facecolors` is the PYTHON kwarg, and
      // MarkerGroup.to_wire renames it. Writing the python name here is
      // silently ignored — the points move and keep the previous direction's
      // colours, which reads as "the IPF triangle is all one colour".
      g.fill_color = idx.map((k) => hex(rgb, k));
      g.fill_alpha = 1.0;
      g.sizes = idx.map(() => HDR.point_size_2d);
      h.applyUpdate(panelKey(p), JSON.stringify(pj));
    });
    return;
  }
  if (name === '2d-heat') {
    panels.forEach(async (p, i) => {
      const rec = PHASES[i]; if (!rec || !rec.raster) return;
      const raw = await pngToRawB64(rec.raster[dir]);
      const gj = live(h, geomKey(p), FIGS[name].state[geomKey(p)]);
      const rg = gj.raster_geom || {};
      for (const id of Object.keys(rg)) {
        rg[id].image_b64 = raw.b64;
        rg[id].image_width = raw.w;
        rg[id].image_height = raw.h;
      }
      gj.raster_geom = rg;
      h.applyUpdate(geomKey(p), JSON.stringify(gj));
      const pj = live(h, panelKey(p), FIGS[name].state[panelKey(p)]);
      pj._geom_rev = (pj._geom_rev || 0) + 1;
      h.applyUpdate(panelKey(p), JSON.stringify(pj));
    });
    return;
  }
  if (name === '3d-points') {
    const p = panels[0];
    const idx = [];
    for (let i = 0; i < M; i += 1) {
      if (Number.isFinite(xyz[3 * i])) idx.push(i);
    }
    const use = [];
    for (let i = 0; i < idx.length; i += STRIDE) use.push(idx[i]);
    const n = use.length;
    const v = new Float32Array(n * 3), c = new Uint8Array(n * 3),
      z = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const i = use[k];
      v[3 * k] = xyz[3 * i]; v[3 * k + 1] = xyz[3 * i + 1]; v[3 * k + 2] = xyz[3 * i + 2];
      c[3 * k] = rgb[3 * i]; c[3 * k + 1] = rgb[3 * i + 1]; c[3 * k + 2] = rgb[3 * i + 2];
      z[k] = xyz[3 * i + 2];
    }
    // A 3-D panel's cloud does NOT live in the view json — anyplotlib hoists
    // vertices and per-point colours into `panel_<id>_geom` so a camera nudge
    // never re-transmits them. Writing `vertices` into the view json is a no-op.
    const gj = live(h, geomKey(p), FIGS[name].state[geomKey(p)]);
    gj.vertices_b64 = b64(new Uint8Array(v.buffer));
    gj.point_colors_b64 = b64(c);
    gj.z_values_b64 = b64(new Uint8Array(z.buffer));
    h.applyUpdate(geomKey(p), JSON.stringify(gj));
    const pj = live(h, panelKey(p), FIGS[name].state[panelKey(p)]);
    pj.vertices_count = n;
    pj._geom_rev = (pj._geom_rev || 0) + 1;
    h.applyUpdate(panelKey(p), JSON.stringify(pj));
    return;
  }
  if (name === '3d-heat') {
    panels.forEach((p, i) => {
      const rec = PHASES[i]; if (!rec || !rec.texture) return;
      // `texture_url` is a data: URL, so the PNG goes straight in — no decode.
      const gj = live(h, geomKey(p), FIGS[name].state[geomKey(p)]);
      gj.texture_url = rec.texture[dir];
      h.applyUpdate(geomKey(p), JSON.stringify(gj));
      const pj = live(h, panelKey(p), FIGS[name].state[panelKey(p)]);
      pj._geom_rev = (pj._geom_rev || 0) + 1;
      h.applyUpdate(panelKey(p), JSON.stringify(pj));
    });
  }
}

/** Mark the picked orientation on `name`. The 2-D scatter moves its white
 *  marker; both spheres move the highlight AND turn to face it. The 2-D heatmap
 *  has no marker — neither does the live view it mirrors. */
function applyPick(name) {
  const h = H[name];
  if (!h) return;
  const { xy, xyz } = D[dir];
  const panels = FIGS[name].panels;

  if (name === '2d-points') {
    const x = xy[2 * pick], y = xy[2 * pick + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const i = PHASES.findIndex((r) => r.idx === PHASE[pick]);
    const p = panels[i < 0 ? 0 : i];
    const pj = live(h, panelKey(p), FIGS[name].state[panelKey(p)]);
    const g = markerGroups(pj).pick;
    if (!g) return;
    g.offsets = [[x, y]];
    h.applyUpdate(panelKey(p), JSON.stringify(pj));
    return;
  }
  if (name === '3d-points' || name === '3d-heat') {
    const vx = xyz[3 * pick], vy = xyz[3 * pick + 1], vz = xyz[3 * pick + 2];
    if (!Number.isFinite(vx)) return;
    let sel = panels;
    if (name === '3d-heat') {          // one sphere per phase — aim only the
      const i = PHASES.findIndex((r) => r.idx === PHASE[pick]);  // picked one
      sel = [panels[i < 0 ? 0 : i]];
    }
    for (const p of sel) {
      const pj = live(h, panelKey(p), FIGS[name].state[panelKey(p)]);
      pj.highlight = { x: vx, y: vy, z: vz, color: '#ffffff', size: 11 };
      // Rotate to bring the picked orientation to the centre. This is
      // `ipf_window.face_camera` verbatim: el = asin(vz), az = atan2(vx, -vy).
      // Not "some angle that points that way" — atan2(vy, vx) - 90 is the same
      // direction 180 out, and lands the highlight on the sphere's far edge.
      // `_view_from_python` is what tells the renderer this camera is INTENDED;
      // without it the panel preserves the camera the reader orbited to and the
      // rotation is silently dropped.
      const r = Math.hypot(vx, vy, vz) || 1;
      pj.azimuth = Math.atan2(vx, -vy) * 180 / Math.PI;
      pj.elevation = Math.asin(Math.max(-1, Math.min(1, vz / r))) * 180 / Math.PI;
      pj._view_from_python = true;
      h.applyUpdate(panelKey(p), JSON.stringify(pj));
    }
  }
}

function setPick(i) {
  pick = i;
  applyPick(viewName());
  readout();
}

function readout() {
  const iy = Math.floor(pick / NX), ix = pick % NX;
  const x = D[dir].xy[2 * pick], y = D[dir].xy[2 * pick + 1];
  const el = document.getElementById('ox-readout');
  if (!el) return;
  el.textContent = Number.isFinite(x)
    ? `IPF-${dir.toUpperCase()} · scan (${ix}, ${iy}) · sector (${x.toFixed(3)}, `
      + `${y.toFixed(3)}) · phase ${PHASE[pick]}`
    : `IPF-${dir.toUpperCase()} · scan (${ix}, ${iy}) · no indexed orientation`;
}

// ── the map ──────────────────────────────────────────────────────────────────
// The map image is repainted onto a PASS-THROUGH OVERLAY canvas over the map
// panel's image rect — the same primitive the vectors explorer's DP repaint
// uses, and for the same reason: pushing pixels back through the figure's own
// state keys is not debuggable in the standalone shim, while a drawImage on top
// is. anyplotlib keeps the crosshair, the axes and the theme underneath.
let ovMap = null;
function makeOverlay(panel, iw, ih) {
  const host = panel.plotCanvas.parentElement;
  const scale = Math.min(panel.imgW / iw, panel.imgH / ih);
  const w = iw * scale, h = ih * scale;
  const x = (panel.imgW - w) / 2, y = (panel.imgH - h) / 2;
  const c = document.createElement('canvas');
  c.width = iw; c.height = ih;
  c.style.cssText = 'position:absolute;pointer-events:none;z-index:2;'
    + 'image-rendering:pixelated;'
    + `left:${panel.plotCanvas.offsetLeft + x}px;`
    + `top:${panel.plotCanvas.offsetTop + y}px;`
    + `width:${w}px;height:${h}px;`;
  if (getComputedStyle(host).position === 'static') host.style.position = 'relative';
  host.insertBefore(c, panel.plotCanvas.nextSibling);
  return c.getContext('2d');
}

function pushMap() {
  if (!ovMap) return;
  const rgb = D[dir].rgb;
  const img = ovMap.createImageData(NX, NY);
  const d = img.data;
  for (let i = 0; i < M; i++) {
    d[4 * i] = rgb[3 * i]; d[4 * i + 1] = rgb[3 * i + 1];
    d[4 * i + 2] = rgb[3 * i + 2]; d[4 * i + 3] = 255;
  }
  ovMap.putImageData(img, 0, 0);
}

// The crosshair on the MAP is the only interaction that changes the pick.
function onMapEvent(ev) {
  if (ev.event_type !== 'pointer_move' && ev.event_type !== 'pointer_up') return;
  if (typeof ev.cx !== 'number') return;
  const ix = Math.max(0, Math.min(NX - 1, Math.round(ev.cx)));
  const iy = Math.max(0, Math.min(NY - 1, Math.round(ev.cy)));
  setPick(iy * NX + ix);
}

// ── toggles ──────────────────────────────────────────────────────────────────
function syncSeg() {
  document.querySelectorAll('.ox-seg-btn').forEach((b) => {
    const on = (b.dataset.dim && b.dataset.dim === dim)
      || (b.dataset.style && b.dataset.style === style)
      || (b.dataset.dir && b.dataset.dir === dir);
    b.setAttribute('aria-pressed', String(!!on));
  });
  // A view that did not build (no scipy, an empty sector) disables the toggle
  // that would select it rather than showing an empty box.
  document.querySelectorAll('.ox-seg-btn[data-style]').forEach((b) => {
    b.disabled = !has(dim + '-' + (b.dataset.style === 'points' ? 'points' : 'heat'));
  });
  document.querySelectorAll('.ox-seg-btn[data-dim]').forEach((b) => {
    b.disabled = !has(b.dataset.dim + '-points') && !has(b.dataset.dim + '-heat');
  });
}

function select(next) {
  if (next.dim) dim = next.dim;
  if (next.style) style = next.style;
  // Fall back within the chosen projection when the other half is missing.
  if (!has(viewName())) style = (style === 'points') ? 'heatmap' : 'points';
  if (!has(viewName())) return;
  show(viewName());
  syncSeg();
}

function setDirection(d) {
  if (!D[d]) return;
  dir = d;
  pushMap();
  for (const name of Object.keys(H)) { applyDirection(name); applyPick(name); }
  syncSeg();
  readout();
}

// The map is its OWN mount, kept out of `H`: `H` is the explorer's lazy-mount
// table, and every loop over it means "each explorer view that exists".
const mapHandle = mount(document.getElementById('ox-map'), FIGS.map.state,
                        { onEvent: onMapEvent });
await new Promise((res) => requestAnimationFrame(() => setTimeout(res, 30)));
const mapPanel = mapHandle.api.panels.get(FIGS.map.panels[0]);
if (mapPanel) { ovMap = makeOverlay(mapPanel, NX, NY); pushMap(); }

document.querySelectorAll('.ox-seg-btn').forEach((b) => {
  b.addEventListener('click', () => {
    if (b.disabled) return;
    if (b.dataset.dir) setDirection(b.dataset.dir);
    else select({ dim: b.dataset.dim, style: b.dataset.style });
  });
});
select({ dim: '2d', style: 'points' });
setPick(Math.floor(M / 2));

// Test hook: the SAME code paths the crosshair and the toggles call.
window.__ox = {
  setPick(iy, ix) { setPick(iy * NX + ix); },
  setDirection,
  select,
  views: () => VIEW_NAMES.filter(has),
  state() {
    const name = viewName(), h = H[name], panels = FIGS[name].panels;
    const out = { dir, dim, style, view: name, pick,
                  iy: Math.floor(pick / NX), ix: pick % NX,
                  mounted: Object.keys(H) };
    if (h && name === '2d-points') {
      const i = PHASES.findIndex((r) => r.idx === PHASE[pick]);
      const pj = live(h, panelKey(panels[i < 0 ? 0 : i]));
      const g = markerGroups(pj);
      out.marker = g.pick ? g.pick.offsets[0] : null;
      out.cloudColors = g.cloud ? (g.cloud.fill_color || []).slice(0, 4) : null;
      out.cloudN = g.cloud ? g.cloud.offsets.length : 0;
    }
    if (h && (name === '3d-points' || name === '3d-heat')) {
      const pj = live(h, panelKey(panels[0]));
      out.highlight = pj.highlight || null;
      out.azimuth = pj.azimuth; out.elevation = pj.elevation;
    }
    return out;
  },
  // The CURRENT explorer view's box on screen, so a test can clip to the panel
  // it means instead of guessing at a fraction of the row — a guessed fraction
  // still "passes" while measuring the wrong pixels.
  viewRect() {
    const el = document.querySelector('#ox-slot > div.ox-on');
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return { x: r.left, y: r.top, width: r.width, height: r.height };
  },
  // Where nav (iy, ix) is on SCREEN, so a test can press exactly on the
  // crosshair instead of guessing at a fraction of the figure.
  navToPage(iy, ix) {
    const p = mapHandle.api.panels.get(FIGS.map.panels[0]);
    if (!p || !p.overlayCanvas) return null;
    const r = p.overlayCanvas.getBoundingClientRect();
    return { x: r.left + ((ix + 0.5) / NX) * r.width,
             y: r.top + ((iy + 0.5) / NY) * r.height };
  },
  _h: () => ({ H, FIGS }),
};
document.getElementById('ox-root').dataset.ready = '1';
"""


def orientation_explorer_html(result, caption: str = "",
                              cache_key: "str | None" = None) -> "str | None":
    """The self-contained IPF explorer page for one orientation result, or None
    when it is empty / over the embed cap.

    ``cache_key`` (a cell id) memoizes the built page per ``(cell_id, result
    identity)``: a rebuild for the same cell and the same result reuses the
    packed blob and the serialized figures instead of re-running orix over every
    position three times and the density six, which is the expensive part here.
    """
    if cache_key is not None:
        hit = _EXPLORER_CACHE.get(cache_key)
        if hit is not None and hit[0] is result:
            return hit[1]

    payload = pack_orientation(result)
    if payload is None:
        return None
    built = _build_all(payload)
    if built is None:
        return None

    hdr = dict(payload["header"])
    hdr["point_size_2d"] = POINT_SIZE_2D
    hdr["built"] = [k for k in VIEWS if k in built]
    cap = _html.escape(caption or "")

    def _json_script(el_id: str, obj) -> str:
        # </script> cannot appear inside a script element — escape the slash.
        txt = json.dumps(obj).replace("</", "<\\/")
        return f"<script type=\"application/json\" id=\"{el_id}\">{txt}</script>"

    esm_safe = _esm_text().replace("</script>", "<\\/script>")
    slots = "".join(f"<div data-view=\"{v}\"></div>"
                    for v in VIEWS if v in built)
    dim_seg = "".join(
        f"<button type=\"button\" class=\"ox-seg-btn\" data-dim=\"{v}\" "
        f"data-testid=\"ox-dim-{v}\" aria-pressed=\"{str(v == '2d').lower()}\">"
        f"{v.upper()}</button>" for v in ("2d", "3d"))
    style_seg = "".join(
        f"<button type=\"button\" class=\"ox-seg-btn\" data-style=\"{s}\" "
        f"data-testid=\"ox-style-{s}\" aria-pressed=\"{str(s == 'points').lower()}\">"
        f"{s.capitalize()}</button>" for s in ("points", "heatmap"))
    dir_seg = "".join(
        f"<button type=\"button\" class=\"ox-seg-btn\" data-dir=\"{d}\" "
        f"data-testid=\"ox-dir-{d}\" aria-pressed=\"{str(d == 'z').lower()}\">"
        f"{d.upper()}</button>" for d in DIRECTIONS)

    page = (
        "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<style>{_EXPLORER_CSS}</style></head><body>"
        "<div id=\"ox-root\" class=\"ox-wrap\">"
        "<h4>Orientation map — the crosshair marks that pixel's orientation "
        "and turns the sphere to face it</h4>"
        "<div class=\"ox-row\">"
        "<div id=\"ox-map\"></div>"
        f"<div id=\"ox-slot\" class=\"ox-slot\">{slots}</div>"
        "</div>"
        "<div class=\"ox-controls\">"
        "<span class=\"ox-status-dot\" title=\"live\"></span>"
        "<span class=\"ox-seg\"><span class=\"ox-seg-label\">view</span>"
        f"<span class=\"ox-seg-group\" role=\"group\">{dim_seg}</span>"
        f"<span class=\"ox-seg-group\" role=\"group\">{style_seg}</span></span>"
        "<span class=\"ox-seg\"><span class=\"ox-seg-label\">direction</span>"
        f"<span class=\"ox-seg-group\" role=\"group\">{dir_seg}</span></span>"
        "</div>"
        "<div id=\"ox-readout\" class=\"ox-meta\"></div>"
        f"<div class=\"ox-meta\">{cap}</div>"
        "</div>"
        f"{_json_script('ox-header', hdr)}"
        f"<script type=\"application/json\" id=\"ox-data\">{payload['b64']}</script>"
        f"{_json_script('ox-figs', built)}"
        f"<script type=\"text/plain\" id=\"ox-esm\">{esm_safe}</script>"
        "<script type=\"module\">const VIEW_NAMES = "
        f"{json.dumps(list(VIEWS))};\n{_EXPLORER_JS_TMPL}</script>"
        "</body></html>"
    )
    if cache_key is not None:
        _EXPLORER_CACHE[cache_key] = (result, page)
    return page


def orientation_for_cell(session, cell) -> "object | None":
    """The orientation result behind a figure cell's base layer, or None.

    Resolution goes through the cell spec's SignalRef → live plot → tree, then
    the SAME chain the live IPF window uses (``ipf_view.tree_orientation_result``)
    so a cell dragged from a raw-OM, vector-OM or EBSD result all resolve.
    """
    spec = getattr(cell, "spec", None)
    if spec is None or not spec.panels:
        return None
    try:
        layers = spec.panels[0].layers
        if not layers:
            return None
        plot = layers[0].source.resolve(session)
        if plot is None:
            return None
        from spyde.actions.ipf_view import tree_orientation_result
        return tree_orientation_result(getattr(plot, "signal_tree", None))
    except Exception as e:
        log.debug("[report] orientation resolve for cell failed: %s", e)
        return None

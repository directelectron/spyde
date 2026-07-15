"""
ipf_density.py — inverse pole **density** function (IPDF) heatmap.

A native-anyplotlib view for an IPF window: the density of the per-pixel crystal
directions across the fundamental sector (orix
:func:`orix.measure.pole_density_function`), one triangle per phase. Tagged
``view="density"`` so the frontend offers it as a third toggle next to the 2-D
RGB map and the 3-D sphere.

The orix density grid is EQUAL-AREA (non-uniform, non-axis-aligned), so it can't
use anyplotlib's ``pcolormesh`` raster fast path directly (that needs a regular
``np.meshgrid`` of separable edges). ``_resample_density_to_raster`` resamples it
onto a regular grid (nearest-neighbour, so displayed MRD values are exact bin
values) and draws it as a single :meth:`anyplotlib.Plot1D.add_raster` image —
one ``drawImage`` instead of thousands of per-cell polygons. Falls back to
``pcolormesh`` (still correct, just the slower polygon path) if the resample is
unavailable. No Qt.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.ipf_view import _as_orientation_map

# NB: named `logger` (not `log`) — build_ipf_density_figure has a `log: bool`
# param for log-scale density that would otherwise shadow a module-level `log`.
from spyde.actions.figure_registry import keep_alive

logger = logging.getLogger(__name__)


def _sector_limits(xy_edges: np.ndarray):
    """``(xlim, ylim)`` from the fundamental-sector outline, padded a little."""
    ex = np.asarray(xy_edges)[:, 0]
    ey = np.asarray(xy_edges)[:, 1]
    xmin, xmax = float(ex.min()), float(ex.max())
    ymin, ymax = float(ey.min()), float(ey.max())
    px = 0.08 * ((xmax - xmin) or 1.0)
    py = 0.12 * ((ymax - ymin) or 1.0)
    return (xmin - px, xmax + px), (ymin - py, ymax + py)


def _resample_density_to_raster(x, y, hist, xlim, ylim, cmap: str,
                                vmax, *, res: int = 256):
    """Resample an orix equal-area (non-uniform, non-axis-aligned) density
    grid onto a REGULAR raster over ``(xlim, ylim)`` at ``res x res``, then
    colour-map it to an ``(res, res, 4)`` uint8 RGBA image.

    ``pcolormesh`` can only auto-raster a regular axis-aligned mesh; the
    ``pole_density_function`` grid is equal-area (cell corners/centres are
    NOT a ``np.meshgrid`` of separable 1-D edges), so it always falls to the
    slow per-cell-polygon path. Nearest-neighbour interpolation is used
    (not linear) so displayed MRD values are exact histogram bin values, not
    a blend — important since this is a quantitative density map, not just a
    pretty picture.

    ``x``/``y`` are the ``(nr+1, nc+1)`` cell-CORNER grids (the same
    ``pcolormesh`` convention orix returns them in) and ``hist`` the
    ``(nr, nc)`` per-cell field — cell centres are averaged from the corners
    before interpolating so the point cloud matches the value array shape.

    Returns ``(rgba, extent)`` with the same row/col orientation convention
    as :meth:`anyplotlib.Plot1D.add_raster` (row 0 = top / max y, col 0 =
    left / min x), or ``None`` if scipy is unavailable or the interpolation
    fails (caller should fall back to ``pcolormesh``).
    """
    from anyplotlib._utils import _build_colormap_lut

    try:
        from scipy.interpolate import griddata
    except Exception as e:
        logger.debug("scipy unavailable for IPF density raster resample: %s", e)
        return None

    xc = np.asarray(x, dtype=float)
    yc = np.asarray(y, dtype=float)
    hc = np.ma.asarray(hist, dtype=float)
    if xc.shape[0] == hc.shape[0] + 1 and xc.shape[1] == hc.shape[1] + 1:
        # Cell-corner grids -> cell centres (average the 4 corners), matching
        # hist's (nr, nc) shape.
        xc = 0.25 * (xc[:-1, :-1] + xc[1:, :-1] + xc[:-1, 1:] + xc[1:, 1:])
        yc = 0.25 * (yc[:-1, :-1] + yc[1:, :-1] + yc[:-1, 1:] + yc[1:, 1:])
    elif xc.shape != hc.shape:
        logger.debug("IPF density grid/hist shape mismatch: x=%s hist=%s",
                     xc.shape, hc.shape)
        return None

    xr = np.ravel(xc)
    yr = np.ravel(yc)
    hr = np.ravel(np.ma.filled(hc, np.nan))
    finite = np.isfinite(hr)
    if not np.any(finite):
        return None
    xr, yr, hr = xr[finite], yr[finite], hr[finite]

    xg = np.linspace(xlim[0], xlim[1], res)
    yg = np.linspace(ylim[0], ylim[1], res)      # row i = yg[i], ascending
    XG, YG = np.meshgrid(xg, yg)
    try:
        field = griddata((xr, yr), hr, (XG, YG), method="nearest")
    except Exception as e:
        logger.debug("griddata resample failed: %s", e)
        return None

    lo = 0.0
    hi = float(vmax) if vmax is not None else float(np.nanmax(hr))
    span = (hi - lo) or 1.0
    lut = np.asarray(_build_colormap_lut(cmap), dtype=np.uint8)   # (256, 3)
    t = np.clip((field - lo) / span, 0.0, 1.0)
    idx = np.rint(t * 255).astype(np.intp)

    rgba = np.zeros((res, res, 4), dtype=np.uint8)
    rgba[..., :3] = lut[idx]
    rgba[..., 3] = 255                            # opacity comes from clip_path

    # row 0 currently = yg[0] = ylim[0] (bottom); add_raster wants row 0 = top.
    rgba = rgba[::-1, :, :]
    extent = (float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
    return np.ascontiguousarray(rgba), extent


def build_ipf_density_figure(result, direction: str = "z", *,
                             resolution: float = 2.0, sigma: float = 5.0,
                             log: bool = False, vmax=None, cmap: str = "fire"):
    """Build the IPDF heatmap figure → ``(fig, fig_id, html)``.

    One axis per phase present in the map. For each, the best-match orientations
    rotate the sample *direction* into crystal frame; ``pole_density_function``
    folds those directions into the point-group fundamental sector and bins them
    (MRD). The equal-area histogram is resampled onto a regular raster and drawn
    as a single stretched RGBA image (:meth:`anyplotlib.Plot1D.add_raster`)
    clipped to the sector, with the sector outline + ``[hkl]`` corner labels. If
    the resample is unavailable, falls back to the ``pcolormesh`` per-cell-polygon
    path (still correct, just slower to draw).
    """
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from orix.measure import pole_density_function
    from orix.quaternion import Rotation

    from spyde.drawing.plots.plot import finalize_figure_html
    from spyde.signals.orientation_map import _direction_vector, ipf_triangle_xy

    om = _as_orientation_map(result)
    quats = np.asarray(om.quats)[:, :, 0, :].reshape(-1, 4)   # best match / pixel
    phase_map = np.asarray(om.phase_map()).reshape(-1)
    v = _direction_vector(direction)

    pidxs = [p for p in range(int(om.n_phases)) if np.any(phase_map == p)]
    if not pidxs:
        pidxs = [0]

    fig, axes = apl.subplots(1, len(pidxs))
    arr = np.array(axes, dtype=object).ravel()
    for ax, pidx in zip(arr, pidxs):
        phase = om.orix_phase(pidx)
        sel = phase_map == pidx
        q = quats[sel] if np.any(sel) else quats
        t = Rotation(q) * v                                  # crystal directions
        hist, (x, y) = pole_density_function(
            t, symmetry=phase.point_group, resolution=resolution,
            sigma=sigma, log=log, hemisphere="upper",
        )
        xy_edges, label_xy, labels = ipf_triangle_xy(phase)
        xlim, ylim = _sector_limits(xy_edges)
        xy = ax.axes2d(xlim=xlim, ylim=ylim, aspect="equal")
        # Clip to the curved sector boundary so edge cells (the equal-area
        # grid is coarse there) don't overflow the triangle.
        raster = _resample_density_to_raster(x, y, hist, xlim, ylim, cmap, vmax)
        if raster is not None:
            rgba, extent = raster
            xy.add_raster(rgba, extent=extent, clip_path=xy_edges, smooth=True)
        else:
            xy.pcolormesh(x, y, hist, cmap=cmap, vmin=0.0,
                          vmax=(None if vmax is None else float(vmax)),
                          clip_path=xy_edges)
        xy.plot(xy_edges[:, 0], xy_edges[:, 1], color="#ffffff", linewidth=1.5)
        for (lx, ly), txt in zip(np.asarray(label_xy, dtype=float), labels):
            xy.text(float(lx), float(ly), str(txt), color="#ffffff", fontsize=12)
        if len(pidxs) > 1:
            try:
                ax.set_title(str(getattr(phase, "name", "") or f"phase {pidx}"))
            except Exception as e:
                logger.debug("set_title on IPF density panel failed: %s", e)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html


def emit_ipf_density(window_id: int, result, direction: str = "z", **kw) -> bool:
    """Build + emit the IPDF heatmap as a ``view="density"`` figure for
    *window_id*. Returns True if a figure was emitted."""
    from spyde.backend.ipc import emit
    try:
        _fig, fig_id, html = build_ipf_density_figure(result, direction, **kw)
    except Exception as e:
        logger.debug("ipf density build failed: %s", e)
        return False
    keep_alive(int(window_id), _fig)
    emit({
        "type": "figure", "fig_id": fig_id, "window_id": int(window_id),
        "html": html, "title": "IPF density", "is_navigator": False,
        "view": "density",
    })
    return True

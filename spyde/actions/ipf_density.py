"""
ipf_density.py — inverse pole **density** function (IPDF) heatmap.

A native-anyplotlib view for an IPF window: the density of the per-pixel crystal
directions across the fundamental sector (orix
:func:`orix.measure.pole_density_function` → :meth:`anyplotlib.PlotXY.pcolormesh`),
one triangle per phase. Tagged ``view="density"`` so the frontend offers it as a
third toggle next to the 2-D RGB map and the 3-D sphere.

This is the orix ``inverse_pole_density_function`` example rendered with pure
anyplotlib primitives (a data-coordinate quad mesh) — no matplotlib raster.
No Qt.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.ipf_view import _as_orientation_map

# NB: named `logger` (not `log`) — build_ipf_density_figure has a `log: bool`
# param for log-scale density that would otherwise shadow a module-level `log`.
logger = logging.getLogger(__name__)

# Keep figures alive past the emit (the _electron registry holds only a weak ref).
_ALIVE: list = []


def _sector_limits(xy_edges: np.ndarray):
    """``(xlim, ylim)`` from the fundamental-sector outline, padded a little."""
    ex = np.asarray(xy_edges)[:, 0]
    ey = np.asarray(xy_edges)[:, 1]
    xmin, xmax = float(ex.min()), float(ex.max())
    ymin, ymax = float(ey.min()), float(ey.max())
    px = 0.08 * ((xmax - xmin) or 1.0)
    py = 0.12 * ((ymax - ymin) or 1.0)
    return (xmin - px, xmax + px), (ymin - py, ymax + py)


def build_ipf_density_figure(result, direction: str = "z", *,
                             resolution: float = 2.0, sigma: float = 5.0,
                             log: bool = False, vmax=None, cmap: str = "fire"):
    """Build the IPDF heatmap figure → ``(fig, fig_id, html)``.

    One axis per phase present in the map. For each, the best-match orientations
    rotate the sample *direction* into crystal frame; ``pole_density_function``
    folds those directions into the point-group fundamental sector and bins them
    (MRD); the masked histogram is drawn as a ``pcolormesh`` quad mesh with the
    sector outline + ``[hkl]`` corner labels.
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
        # Clip the mesh to the curved sector boundary so edge cells (the
        # equal-area grid is coarse there) don't overflow the triangle.
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
    _ALIVE.append(fig)
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
    emit({
        "type": "figure", "fig_id": fig_id, "window_id": int(window_id),
        "html": html, "title": "IPF density", "is_navigator": False,
        "view": "density",
    })
    return True

"""
ipf_refine.py — the live per-orientation correlation HEATMAP on the IPF triangle
for the Orientation-Mapping refine step (Qt parity, Qt-free).

During refine, instead of a static colour-key triangle, each phase's IPF
fundamental-sector triangle is painted with the template-match correlation for
the CURRENT pattern: every template orientation is projected into the sector and
its correlation interpolated over the triangle. It updates as the navigator
moves. Double-clicking adds a mask circle that LIMITS which orientations the
refine considers (``rot_mask``) — useful to lock onto a known orientation family.
Multi-phase libraries get one triangle per phase.

The heavy maths mirrors ``actions.pyxem`` (the Qt refine widget) but imports no
Qt — only orix / pyxem.signals / scipy (+ ``matplotlib.path.Path`` for the
point-in-triangle test, geometry only). The heatmap itself is rendered natively
by :mod:`ipf_refine_render` (anyplotlib ``PlotXY`` — no matplotlib raster).
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

GRID_N = 96      # heatmap raster resolution per phase


def _triangle_xy(phase):
    """(edge_xy (M,2), label_xy (K,2), labels) for the phase's fundamental sector."""
    from orix.projections import StereographicProjection
    from pyxem.signals.indexation_results import (
        _closed_edges_in_hemisphere, _get_ipf_axes_labels,
    )
    s = StereographicProjection()
    sector = phase.point_group.fundamental_sector
    edges = _closed_edges_in_hemisphere(sector.edges, sector)
    ex, ey = s.vector2xy(edges)
    xy_edges = np.vstack((np.atleast_1d(ex), np.atleast_1d(ey))).T.astype(float)
    try:
        raw = _get_ipf_axes_labels(sector.vertices, symmetry=phase.point_group)
        labels = [str(l).replace("$", "") for l in raw]
        lx, ly = s.vector2xy(sector.vertices)
        lx = np.atleast_1d(np.array(lx, float))
        ly = np.atleast_1d(np.array(ly, float))
        cx, cy = float(lx.mean()), float(ly.mean())
        D = 0.16
        label_xy = np.vstack([lx + D * (lx - cx), ly + D * (ly - cy)]).T
    except Exception:
        labels, label_xy = [], np.empty((0, 2))
    return xy_edges, label_xy, labels


def build_phase_ipf(sim) -> list[dict]:
    """Per-phase IPF geometry (geometry-only → compute ONCE per library): each
    phase's template stereographic positions, global template indices, triangle
    outline + corner labels, and a Delaunay interpolation grid for the heatmap."""
    from orix.quaternion import Rotation
    from orix.vector import Vector3d
    from orix.projections import StereographicProjection
    from scipy.spatial import Delaunay
    from matplotlib.path import Path
    from spyde.actions.orientation_compute import template_tables, sim_phases_list

    quats, phase_of = template_tables(sim)
    phases = sim_phases_list(sim)
    sp = StereographicProjection()
    infos: list[dict] = []
    for p, phase in enumerate(phases):
        lib = np.where(phase_of == p)[0]                 # global template indices
        if lib.size == 0:
            continue
        vecs = (Rotation(quats[lib]) * Vector3d.zvector()
                ).in_fundamental_sector(phase.point_group)
        xs, ys = sp.vector2xy(vecs)
        xs = np.atleast_1d(np.array(xs, float))
        ys = np.atleast_1d(np.array(ys, float))
        tri_xy, label_xy, labels = _triangle_xy(phase)

        # Bound the panel by the triangle AND the corner labels (so labels, which
        # sit OUTSIDE the vertices, aren't clipped) + a small margin.
        bound_xy = np.vstack([tri_xy, label_xy]) if len(label_xy) else tri_xy
        mins, maxs = bound_xy.min(0), bound_xy.max(0)
        pad = 0.05 * (maxs - mins + 1e-9)
        mins, maxs = mins - pad, maxs + pad
        gx = np.linspace(mins[0], maxs[0], GRID_N)
        gy = np.linspace(mins[1], maxs[1], GRID_N)
        xx, yy = np.meshgrid(gx, gy)
        flat = np.vstack((xx.ravel(), yy.ravel())).T

        verts = weights = None
        outside = np.ones(flat.shape[0], dtype=bool)
        try:
            tri = Delaunay(np.vstack((xs, ys)).T)
            simplex = tri.find_simplex(flat)
            inside_hull = simplex >= 0
            v = np.take(tri.simplices, simplex, axis=0)
            T = np.take(tri.transform, simplex, axis=0)
            delta = flat - T[:, -1]
            bary = np.einsum("njk,nk->nj", T[:, :-1, :], delta)
            weights = np.hstack((bary, 1 - bary.sum(1, keepdims=True)))
            verts = v
            outside = ~inside_hull
        except Exception as e:
            log.debug("Delaunay barycentric setup failed (using fallback): %s", e)
        try:                                              # also clip to the triangle polygon
            outside = outside | (~Path(tri_xy).contains_points(flat))
        except Exception as e:
            log.debug("triangle-polygon clip failed: %s", e)

        infos.append(dict(
            phase_index=p,
            name=(getattr(phase, "name", None) or f"phase {p}"),
            lib_idx=lib, xs=xs, ys=ys,
            tri_xy=tri_xy, label_xy=label_xy, labels=labels,
            mins=mins, maxs=maxs, grid_n=GRID_N,
            verts=verts, weights=weights, outside=outside,
        ))
    return infos


def match_correlations(pattern_data, sim, cache, *, gamma: float = 1.0,
                       normalize_templates: bool = False, rot_mask=None):
    """Full per-template correlation for ONE pattern, GLOBAL-indexed (length =
    n_templates). ``rot_mask`` (global bool) restricts which templates compete.
    Returns ``(corr (n_templates,), best_row [lib_idx, corr, angle, mirror])``."""
    from pyxem.utils.indexation_utils import _mixed_matching_lib_to_polar
    from pyxem.utils._azimuthal_integrations import _slice_radial_integrate
    from spyde.actions.orientation_compute import PYXEM_LOCK

    integrated = cache["integrated"]
    n_tmpl = integrated.shape[0]
    int_t = cache["intensities_norm"] if normalize_templates else cache["intensities_raw"]
    rt, tt = cache["r_templates"], cache["theta_templates"]

    # pyxem's numba matcher is not thread-safe — share the one global lock with
    # the spot overlay + the whole-field compute (which may run concurrently).
    with PYXEM_LOCK:
        polar = _slice_radial_integrate(
            np.asarray(pattern_data, float), cache["factors"], cache["factors_slice"],
            cache["slices"], cache["NR"], cache["NA"], mean=True)
        polar = np.nan_to_num(polar ** gamma).T.astype(float)
        if rot_mask is not None and np.any(rot_mask):
            idx = np.where(rot_mask)[0]
            res = _mixed_matching_lib_to_polar(
                polar, integrated_templates=integrated[idx], r_templates=rt[idx],
                theta_templates=tt[idx], intensities_templates=int_t[idx],
                n_keep=None, frac_keep=1.0, n_best=len(idx), transpose=False)
            res[:, 0] = idx[res[:, 0].astype(int)]
        else:
            res = _mixed_matching_lib_to_polar(
                polar, integrated_templates=integrated, r_templates=rt,
                theta_templates=tt, intensities_templates=int_t,
                n_keep=None, frac_keep=1.0, n_best=n_tmpl, transpose=False)

    corr = np.zeros(n_tmpl, dtype=float)
    li = res[:, 0].astype(int)
    cv = res[:, 1].astype(float)
    ok = (li >= 0) & (li < n_tmpl)
    corr[li[ok]] = cv[ok]
    return corr, res[0]


def interp_grid(corr_global: np.ndarray, info: dict, *, vmax: float | None = None
                ) -> np.ndarray:
    """(GRID, GRID) float in [0,1] of this phase's per-template correlation
    interpolated over the triangle; ``np.nan`` outside the triangle."""
    c = corr_global[info["lib_idx"]]
    vmax = float(c.max()) if vmax is None else float(vmax)
    vmax = vmax if vmax > 0 else 1.0
    cn = np.clip(c / vmax, 0.0, 1.0)
    n = info["grid_n"]
    if info["verts"] is not None and info["weights"] is not None:
        vals = (cn[info["verts"]] * info["weights"]).sum(axis=1)
    else:
        vals = np.zeros(n * n)
    vals = np.clip(vals, 0.0, 1.0)
    vals[info["outside"]] = np.nan
    return vals.reshape(n, n)


def rot_mask_from_circles(infos: list[dict], circles_per_phase: dict,
                          n_templates: int):
    """Global boolean (n_templates,) of templates allowed by the IPF mask circles.
    A phase WITH circles keeps only templates inside one of them; a phase with NO
    circles keeps ALL its templates. Returns ``None`` when no circles exist (→
    every template competes)."""
    if not any(circles_per_phase.get(i["phase_index"]) for i in infos):
        return None
    mask = np.zeros(int(n_templates), dtype=bool)
    for info in infos:
        circs = circles_per_phase.get(info["phase_index"]) or []
        lib = info["lib_idx"]
        if not circs:
            mask[lib] = True
            continue
        xs, ys = info["xs"], info["ys"]
        inside = np.zeros(xs.shape[0], dtype=bool)
        for cx, cy, r in circs:
            inside |= ((xs - cx) ** 2 + (ys - cy) ** 2) <= r * r
        mask[lib[inside]] = True
    return mask

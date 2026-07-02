"""
ipf_refine_render.py — the per-phase IPF correlation heatmap triangles for the OM
refine step, drawn **natively** with anyplotlib ``PlotXY`` (no matplotlib raster).

One ``subplots(1, n_phases)`` figure; each phase panel is a data-coordinate axis
(``ax.axes2d``) holding, in z-order:

* the heatmap — one polygon per inside-sector grid cell (built ONCE, clipped to
  the curved sector via ``clip_path``); the live update just re-pushes the
  per-cell **face colours** (``MarkerGroup.set(facecolors=…)``) so the geometry
  and z-order are stable and each navigator move is a single push;
* the mask circles (region-restriction), updated in place via ``vertices_list``;
* the white triangle outline + ``[hkl]`` corner labels (static);
* the red best-match marker (a scatter point), updated via ``offsets``.

Everything is in stereographic DATA coordinates (like ``ipf_density``), so the
triangle / labels / markers line up with the heatmap with no pixel transform, and
``double_click`` reports the IPF data coords used for the mask.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.ipf_refine import interp_grid

log = logging.getLogger(__name__)

# Keep figures alive past the emit (the _electron registry holds a weak ref).
_ALIVE: list = []

# "fire" (black→red→white) is anyplotlib's correlation-friendly map, same as the
# IPF density heatmap. (anyplotlib's "inferno" is a wrong black→blue ramp.)
_CMAP = "fire"


def _lut(cmap: str = _CMAP) -> np.ndarray:
    from anyplotlib._utils import _build_colormap_lut
    return np.asarray(_build_colormap_lut(cmap))      # (256, 3)


def _hex(rgb) -> str:
    r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def _mesh_geometry(info: dict):
    """Quad polygons (data coords) for every inside-sector grid cell + their
    ``(i, j)`` grid indices (so live updates colour them in the same order)."""
    n = int(info["grid_n"])
    mins, maxs = info["mins"], info["maxs"]
    ex = np.linspace(float(mins[0]), float(maxs[0]), n + 1)
    ey = np.linspace(float(mins[1]), float(maxs[1]), n + 1)
    outside = np.asarray(info["outside"]).reshape(n, n)
    verts, cells = [], []
    for i in range(n):
        for j in range(n):
            if outside[i, j]:
                continue
            verts.append([[ex[j], ey[i]], [ex[j + 1], ey[i]],
                          [ex[j + 1], ey[i + 1]], [ex[j], ey[i + 1]]])
            cells.append((i, j))
    return verts, np.asarray(cells, dtype=int).reshape(-1, 2)


def _cell_colors(vals: np.ndarray, cells: np.ndarray, lut: np.ndarray) -> list:
    """Per-cell hex colours for the (grid_n, grid_n) [0,1] correlation grid."""
    if cells.size == 0:
        return []
    v = np.clip(np.nan_to_num(vals[cells[:, 0], cells[:, 1]]), 0.0, 1.0)
    idx = np.clip(np.round(v * 255).astype(int), 0, 255)
    return [_hex(c) for c in lut[idx]]


def _circle_polys(circles, k: int = 40) -> list:
    """Mask circles → closed N-gon vertex lists (data coords)."""
    if not circles:
        return []
    th = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    cos, sin = np.cos(th), np.sin(th)
    return [np.column_stack([cx + r * cos, cy + r * sin]).tolist()
            for cx, cy, r in circles]


def build_refine_figure(infos: list[dict], *, cmap: str = _CMAP):
    """Build the multi-phase refine figure. Returns ``(fig, fig_id, html,
    panels)`` where each panel carries the live marker groups to update."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    lut = _lut(cmap)
    blank = _hex(lut[0])
    n = max(1, len(infos))
    fig, axes = apl.subplots(1, n)
    arr = np.array(axes, dtype=object).ravel()

    panels = []
    for ax, info in zip(arr, infos):
        mins, maxs = info["mins"], info["maxs"]
        xy = ax.axes2d(xlim=(float(mins[0]), float(maxs[0])),
                       ylim=(float(mins[1]), float(maxs[1])), aspect="equal")

        verts, cells = _mesh_geometry(info)
        faces = [blank] * len(verts)
        # Heatmap (bottom): one polygon per inside-sector cell, clipped to the
        # curved sector boundary; only its face colours change per frame.
        mesh = xy.add_polygons(verts, facecolors=faces, edgecolors=faces,
                               alpha=1.0, linewidths=0.4, clip_path=info["tri_xy"])
        # Mask circles (above the heatmap, below the outline).
        circ = xy.add_polygons([], facecolors="#00e5ff", edgecolors="#00e5ff",
                               alpha=0.18, linewidths=1.6)
        # Static sector outline + corner labels.
        tri = np.asarray(info["tri_xy"], float)
        xy.plot(tri[:, 0], tri[:, 1], color="#ffffff", linewidth=1.4)
        for (lx, ly), txt in zip(np.asarray(info["label_xy"], float), info["labels"]):
            xy.text(float(lx), float(ly), str(txt), color="#ffffff", fontsize=11)
        # Best-match marker (top), initially empty. s = marker radius in px.
        best = xy.scatter([], [], s=9, c="#ff3030", edgecolors="#ffffff")

        if len(infos) > 1:
            try:
                ax.set_title(str(info["name"]))
            except Exception as e:
                log.debug("set_title on refine panel failed: %s", e)

        panels.append({"xy": xy, "plot2d": xy, "info": info, "mesh": mesh,
                       "cells": cells, "circle_grp": circ, "best": best, "lut": lut})

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html, panels


def update_panels(panels, corr_global, circles_per_phase, best_xy_per_phase=None,
                  *, cmap: str = _CMAP) -> None:
    """Re-colour every phase panel from the current correlation + mask circles
    (in place — the polygon geometry and z-order are fixed)."""
    best_xy_per_phase = best_xy_per_phase or {}
    for panel in panels:
        info = panel["info"]
        vals = interp_grid(corr_global, info)
        colors = _cell_colors(vals, panel["cells"], panel["lut"])
        try:
            panel["mesh"].set(facecolors=colors, edgecolors=colors)
        except Exception as e:
            log.debug("updating refine heatmap colours failed: %s", e)

        bxy = best_xy_per_phase.get(info["phase_index"])
        try:
            panel["best"].set(
                offsets=[[float(bxy[0]), float(bxy[1])]] if bxy is not None else [])
        except Exception as e:
            log.debug("updating refine best-match marker failed: %s", e)

        try:
            panel["circle_grp"].set(
                vertices_list=_circle_polys(circles_per_phase.get(info["phase_index"], [])))
        except Exception as e:
            log.debug("updating refine mask circles failed: %s", e)


def best_xy_for(infos, best_lib_idx: int):
    """{phase_index: (x, y)} for the phase that owns ``best_lib_idx`` (the IPF
    point of the best-matching template) — for the red best-match marker."""
    for info in infos:
        hit = np.where(info["lib_idx"] == int(best_lib_idx))[0]
        if hit.size:
            j = int(hit[0])
            return {info["phase_index"]: (float(info["xs"][j]), float(info["ys"][j]))}
    return {}


def emit_refine_window(session, fig, fig_id: str, html: str, *, title: str = "IPF Refine"):
    """Emit the refine-heatmap figure to a fresh window. Returns the window id."""
    from spyde.backend.ipc import emit
    from spyde.actions.figure_registry import keep_alive
    wid = session.next_window_id()
    keep_alive(int(wid), fig)
    emit({"type": "figure", "fig_id": fig_id, "window_id": int(wid),
          "html": html, "title": title, "is_navigator": False})
    return int(wid)


class RefineIpfController:
    """Drives the live per-phase IPF correlation heatmaps during refine: on every
    navigator move (and gamma/normalize change) it re-matches the current pattern
    and recolours each phase's triangle; a double-click on a panel adds/removes a
    mask circle that LIMITS which orientations the match considers (``rot_mask``).
    """

    def __init__(self, dp_plot, signal, sim, cache, infos, panels, *,
                 gamma: float = 1.0, normalize: bool = False):
        import threading
        from spyde.actions.orientation_compute import template_tables
        self.dp_plot = dp_plot
        self.signal = signal
        self.sim = sim
        self.cache = cache
        self.infos = infos
        self.panels = panels
        self.gamma = float(gamma)
        self.normalize = bool(normalize)
        self.n_templates = int(template_tables(sim)[0].shape[0])
        self.circles = {info["phase_index"]: [] for info in infos}   # per-phase masks
        self._last_iyix = (0, 0)
        self._lock = threading.Lock()
        self._selectors: list = []

    def attach(self, tree):
        from spyde.actions.vector_overlay import (
            _navigator_selectors_for, _indices_to_iyix,
        )
        self._to_iyix = _indices_to_iyix
        self._selectors = _navigator_selectors_for(tree, self.dp_plot)
        for sel in self._selectors:
            sel.index_hooks.append(self._on_indices)
            if sel.current_indices is not None:
                self._last_iyix = _indices_to_iyix(sel.current_indices)
        for panel in self.panels:
            self._wire_double_click(panel)
        self._recompute()                       # seed the heatmaps
        return self

    def _on_indices(self, indices):
        self._last_iyix = self._to_iyix(indices)
        self._recompute()

    def set_refine_params(self, *, gamma=None, normalize=None) -> None:
        if gamma is not None:
            self.gamma = float(gamma)
        if normalize is not None:
            self.normalize = bool(normalize)
        self._recompute()

    def _frame(self, iy, ix):
        f = self.signal.data[iy, ix]
        if hasattr(f, "compute"):
            f = f.compute()
        return np.asarray(f, dtype=float)

    def _recompute(self):
        from spyde.actions.ipf_refine import match_correlations, rot_mask_from_circles
        if not self.panels:
            return
        with self._lock:
            try:
                iy, ix = self._last_iyix
                mask = rot_mask_from_circles(self.infos, self.circles, self.n_templates)
                corr, best = match_correlations(
                    self._frame(iy, ix), self.sim, self.cache,
                    gamma=self.gamma, normalize_templates=self.normalize, rot_mask=mask)
                update_panels(self.panels, corr, self.circles,
                              best_xy_for(self.infos, int(best[0])))
            except Exception as e:
                log.debug("refine ipf recompute failed: %s", e)

    def toggle_circle(self, phase_index: int, x: float, y: float) -> None:
        """Double-click action: remove the mask circle the click lands in, else
        add a new one centred there (radius ≈ 9 % of the triangle extent), then
        re-match with the updated region restriction."""
        info = next((i for i in self.infos if i["phase_index"] == phase_index), None)
        if info is None:
            return
        circs = self.circles[phase_index]
        for k, (cx, cy, cr) in enumerate(circs):
            if (x - cx) ** 2 + (y - cy) ** 2 <= cr * cr:
                circs.pop(k)
                self._recompute()
                return
        circs.append((float(x), float(y),
                      0.09 * float((info["maxs"] - info["mins"]).mean())))
        self._recompute()

    def _wire_double_click(self, panel):
        pidx = panel["info"]["phase_index"]

        def _on_dbl(ev=None):
            x = getattr(ev, "xdata", None)
            y = getattr(ev, "ydata", None)
            if x is None and isinstance(ev, dict):
                x, y = ev.get("xdata"), ev.get("ydata")
            if x is not None and y is not None:
                self.toggle_circle(pidx, float(x), float(y))

        try:
            panel["xy"].add_event_handler(_on_dbl, "double_click")
        except Exception as e:
            log.debug("wiring refine panel double-click failed: %s", e)

    def remove(self):
        for sel in self._selectors:
            if self._on_indices in sel.index_hooks:
                sel.index_hooks.remove(self._on_indices)
        self._selectors = []

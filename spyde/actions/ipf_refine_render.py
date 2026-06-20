"""
ipf_refine_render.py — render the per-phase IPF correlation heatmap triangles for
the OM refine step, one anyplotlib figure (subplots(1, n_phases)).

Each phase panel's CONTENT (heatmap + triangle outline + corner labels +
best-match marker + mask circles) is composed with matplotlib-Agg into an RGBA
raster and shown through an anyplotlib ``imshow`` — so it (a) updates live via
``set_data`` as the navigator moves and (b) reports ``double_click`` events in
IPF data coords (for the region-limiting mask). One panel per phase.

NOTE: this panel is the one place SpyDE still composes a frame with matplotlib
rather than native anyplotlib primitives. anyplotlib *can* draw it natively
(imshow + add_polygons/add_texts/add_points/add_circles), but its marker overlays
use IMAGE-PIXEL coords (not the stereographic data axis), so the triangle/labels/
circles need a stereo→pixel transform + a y-flip + a double-click inverse to line
up. The matplotlib raster sidesteps that and renders pixel-correct; revisit if a
fully-native panel is wanted.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.ipf_refine import interp_grid

log = logging.getLogger(__name__)

# Keep figures alive past the emit (the _electron registry holds a weak ref).
_ALIVE: list = []


def render_panel_rgba(vals: np.ndarray, info: dict, *, circles=(), best_xy=None,
                      size_px: int = 300, cmap: str = "inferno") -> np.ndarray:
    """A full IPF-heatmap panel → (size, size, 4) uint8 RGBA (row 0 = TOP)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    mins, maxs = info["mins"], info["maxs"]
    fig = Figure(figsize=(size_px / 100.0, size_px / 100.0), dpi=100)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_axis_off()
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_aspect("equal")

    ax.imshow(vals, extent=[mins[0], maxs[0], mins[1], maxs[1]], origin="lower",
              cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear", zorder=0)
    tri = info["tri_xy"]
    ax.plot(tri[:, 0], tri[:, 1], color="white", lw=1.4, zorder=4)
    for (lx, ly), txt in zip(info["label_xy"], info["labels"]):
        ax.text(lx, ly, txt, color="white", ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=5)
    for cx, cy, r in circles:
        ax.add_patch(Circle((cx, cy), r, fc="#00e5ff", alpha=0.18, zorder=2))
        ax.add_patch(Circle((cx, cy), r, fill=False, ec="#00e5ff", lw=1.6, zorder=3))
    if best_xy is not None:
        ax.plot([best_xy[0]], [best_xy[1]], marker="o", ms=9,
                mec="white", mfc="#ff3030", mew=1.4, zorder=6)

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _gx_gy(info, h, w):
    # imshow row 0 = TOP = max y, so y_axis descends; origin='upper'.
    return (np.linspace(info["mins"][0], info["maxs"][0], w),
            np.linspace(info["maxs"][1], info["mins"][1], h))


def build_refine_figure(infos: list[dict]):
    """Build the multi-phase refine figure. Returns ``(fig, fig_id, html,
    panels)`` where ``panels[i] = {plot2d, info}`` (one per phase)."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    n = max(1, len(infos))
    fig, axes = apl.subplots(1, n)
    arr = np.array(axes, dtype=object).ravel()
    panels = []
    for ax, info in zip(arr, infos):
        blank = np.full((info["grid_n"], info["grid_n"]), np.nan)
        rgba = render_panel_rgba(blank, info)
        gx, gy = _gx_gy(info, rgba.shape[0], rgba.shape[1])
        p = ax.imshow(rgba, axes=[gx, gy], origin="upper")
        try:
            ax.set_axis_off()
        except Exception as e:
            log.debug("set_axis_off on refine panel failed: %s", e)
        try:
            ax.set_title(str(info["name"]))
        except Exception as e:
            log.debug("set_title on refine panel failed: %s", e)
        panels.append({"plot2d": p, "info": info})

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    _ALIVE.append(fig)
    return fig, fig_id, html, panels


def update_panels(panels, corr_global, circles_per_phase, best_xy_per_phase=None,
                  *, cmap: str = "inferno") -> None:
    """Re-render every phase panel from the current correlation + mask circles."""
    best_xy_per_phase = best_xy_per_phase or {}
    for panel in panels:
        info = panel["info"]
        vals = interp_grid(corr_global, info)
        rgba = render_panel_rgba(
            vals, info,
            circles=circles_per_phase.get(info["phase_index"], []),
            best_xy=best_xy_per_phase.get(info["phase_index"]),
            cmap=cmap)
        try:
            panel["plot2d"].set_data(rgba)
        except Exception as e:
            log.debug("updating refine panel raster failed: %s", e)


def best_xy_for(infos, best_lib_idx: int):
    """{phase_index: (x, y)} for the phase that owns ``best_lib_idx`` (the IPF
    point of the best-matching template) — for the red best-match marker."""
    for info in infos:
        hit = np.where(info["lib_idx"] == int(best_lib_idx))[0]
        if hit.size:
            j = int(hit[0])
            return {info["phase_index"]: (float(info["xs"][j]), float(info["ys"][j]))}
    return {}


def emit_refine_window(session, fig_id: str, html: str, *, title: str = "IPF Refine"):
    """Emit the refine-heatmap figure to a fresh window. Returns the window id."""
    from spyde.backend.ipc import emit
    wid = session.next_window_id()
    emit({"type": "figure", "fig_id": fig_id, "window_id": int(wid),
          "html": html, "title": title, "is_navigator": False})
    return int(wid)


class RefineIpfController:
    """Drives the live per-phase IPF correlation heatmaps during refine: on every
    navigator move (and gamma/normalize change) it re-matches the current pattern
    and repaints each phase's triangle; a double-click on a panel adds/removes a
    mask circle that LIMITS which orientations the match considers (``rot_mask``).
    """

    def __init__(self, dp_plot, signal, sim, cache, infos, panels, *,
                 gamma: float = 1.0, normalize: bool = False):
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
        self._lock = __import__("threading").Lock()
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
                import logging
                logging.getLogger(__name__).debug("refine ipf recompute failed: %s", e)

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
            panel["plot2d"].add_event_handler(_on_dbl, "double_click")
        except Exception as e:
            log.debug("wiring refine panel double-click failed: %s", e)

    def remove(self):
        for sel in self._selectors:
            if self._on_indices in sel.index_hooks:
                sel.index_hooks.remove(self._on_indices)
        self._selectors = []

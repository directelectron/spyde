"""
figure_builder.py — build a live anyplotlib figure for a report figure cell.

A report figure cell owns a :class:`~spyde.actions.report.model.FigureSpec`
(recipe) + an in-memory snapshot ndarray. This module renders that into a bare
anyplotlib figure and returns its ``(fig, fig_id, html)`` so the handler can
emit it through the normal bare-figure path (``finalize_figure_html``), plus a
:class:`ReportFigureController` implementing the WindowController protocol so
the figure is reachable by dispatch and torn down by ``Session._forget_window``.

Phase 1 renders a single panel with a single image layer (clim + cmap + axes +
title from the spec). Phase 2 grows grid layouts, multi-layer panels, and
annotations here.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def build_cell_figure(spec, snapshot_array):
    """Render *spec* + *snapshot_array* → ``(fig, fig_id, html)``.

    Phase 1: single panel / single image layer. ``cmap``/``clim`` come from the
    primary layer, axes/title/units from the primary panel. RGB(A) snapshots
    render directly (no colormap)."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html
    from spyde.drawing.colormaps import COLORMAPS

    arr = np.asarray(snapshot_array)
    layer = spec.primary_layer if spec is not None else None
    panel = spec.panels[0] if (spec is not None and spec.panels) else None

    cmap = None
    clim = None
    if layer is not None:
        cmap = COLORMAPS.get(layer.cmap, layer.cmap)
        clim = layer.clim

    fig, axes = apl.subplots(1, 1)
    ax = axes[0][0] if isinstance(axes, list) else axes

    is_rgb = arr.ndim == 3 and arr.shape[-1] in (3, 4)
    frame = arr if is_rgb else np.nan_to_num(np.asarray(arr, dtype=np.float32))

    # Calibrated axes / units from the spec (if present) so the report figure
    # draws the same ticks + scale bar the live plot showed.
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

    p = ax.imshow(frame, cmap=(None if is_rgb else cmap), tile=False, **axes_kw)
    if not is_rgb and clim is not None and clim[0] is not None and clim[1] is not None:
        try:
            p.set_clim(float(clim[0]), float(clim[1]))
        except Exception as e:
            log.debug("report figure set_clim failed: %s", e)

    title = ""
    if panel is not None and panel.title:
        title = str(panel.title)
    if title and hasattr(p, "set_title"):
        try:
            p.set_title(title)
        except Exception as e:
            log.debug("report figure set_title failed: %s", e)

    # The report figure is a COLD standalone embed: its iframe HTML must be
    # self-contained (no PLOTBIN binary channel, no live re-push). When the app's
    # binary transport is active (APL_BINARY_TRANSPORT=1), imshow left the panel
    # pixels as a "\x00bin:<checksum>" change-token in the panel_*_json / _geom
    # traits (the real bytes ride PLOTBIN for MDI windows). A standalone iframe
    # can't resolve that token → BLANK figure. Force each panel to re-serialise
    # with the tokens materialised to inline base64 (Figure._push already does
    # this whenever _binary_wire() is False) so the embedded state paints on load.
    _resolve_pixels_for_standalone(fig)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html


def _resolve_pixels_for_standalone(fig) -> None:
    """Rewrite every panel trait of *fig* with pixel change-tokens materialised
    to inline base64, so the standalone HTML embed is self-contained even when
    the app's Electron binary transport is active.

    Temporarily clears ``APL_BINARY_TRANSPORT`` so ``Figure._push`` takes its
    cold path (``resolve_pixel_tokens`` → real base64 in both the ``panel_*_json``
    and ``panel_*_geom`` traits), then restores the env for the live MDI figures.
    A no-op when binary transport is off (the traits already hold base64)."""
    import os

    prev = os.environ.get("APL_BINARY_TRANSPORT")
    if prev != "1":
        return   # base64 already inline — nothing to resolve.
    plots_map = getattr(fig, "_plots_map", None)
    if not plots_map:
        return
    os.environ["APL_BINARY_TRANSPORT"] = "0"
    try:
        for panel_id in list(plots_map):
            try:
                fig._push(panel_id)
            except Exception as e:
                log.debug("report figure pixel-resolve push failed for %s: %s",
                          panel_id, e)
    finally:
        os.environ["APL_BINARY_TRANSPORT"] = prev


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
        # Phase 1 report figures carry no per-window actions of their own.
        return False

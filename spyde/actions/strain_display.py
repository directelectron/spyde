"""
strain_display.py — the strain-field visualization: a diverging component map
(εxx / εyy / εxy / ω) with the unstrained-reference crosshair.

The background colour encodes one component (diverging colormap centred on 0).
A component toggle swaps the shown map in place. (The old white principal-strain
ellipse glyphs were removed — they cluttered the map and bought nothing the
component colour didn't already show.)

The map is shown in PERCENT strain (rotation in degrees) with a colorbar that
says so, over the scan's own calibrated axes. ``display_component`` is the one
place the fit's fractional values are scaled, so the histogram, the contrast
handles, the colorbar and a committed tree all agree.

Contrast is the PLOT WIDGET's job, not a wizard knob: the strain window emits
the standard sidebar histogram (``emit_strain_histogram`` — same message the
Plot class sends) and the dock's drag-handles / colormap picker reach the
StrainController via the session's ``set_clim`` / ``set_colormap`` controller
fallback. Failed-fit pixels are NaN in the DATA (excluded from the histogram +
auto-levels) and render as the neutral zero-strain colour.

No Qt. Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.strain_mapping import StrainField
from spyde.actions._common import (
    STRAIN_DISPLAY_SCALE, STRAIN_TITLES as _COMPONENTS, robust_map_limits,
    strain_quantity,
)

logger = logging.getLogger(__name__)


def display_component(field: StrainField, component: str) -> np.ndarray:
    """One component of *field* in DISPLAY units: percent strain, degrees of
    rotation. The fit itself stays fractional and in radians."""
    raw = {
        "exx": field.exx, "eyy": field.eyy, "exy": field.exy, "omega": field.omega,
    }[component]
    return np.asarray(raw, dtype=np.float64) * STRAIN_DISPLAY_SCALE[component]


def _auto_clim(arr: np.ndarray) -> tuple[float, float]:
    """Symmetric colour limits — every strain component's zero is meaningful."""
    return robust_map_limits(arr, symmetric=True)


def emit_strain_histogram(window_id, field: StrainField, component: str,
                          clim: tuple[float, float], *,
                          colormap: str = "coolwarm") -> None:
    """Send the sidebar histogram for the strain window — the same message shape
    ``Plot._emit_histogram`` uses, so the dock's contrast handles just work.
    ``symmetric`` tells the dock the two handles are one number (zero stays at
    the middle of the diverging map) and ``colormap`` what its picker should
    show."""
    if window_id is None:
        return
    data = display_component(field, component)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    try:
        counts, edges = np.histogram(finite, bins=64)
        from de_shell.ipc import emit
        emit({
            "type": "histogram",
            "window_id": int(window_id),
            "counts": counts.astype(int).tolist(),
            "edges": [float(e) for e in edges],
            "vmin": float(clim[0]),
            "vmax": float(clim[1]),
            "threshold": None,
            "symmetric": True,
            "colormap": str(colormap),
        })
    except Exception as e:
        logger.debug("strain histogram emit failed: %s", e)


def build_strain_figure(field: StrainField, *, component: str = "exx",
                        ref_yx=None, clim: tuple[float, float] | None = None,
                        cmap: str = "coolwarm", axes=None):
    """Build the strain view → ``(fig, fig_id, html, plot2d)``.
    ``plot2d`` is returned so a controller can live-update the component map /
    reference crosshair / contrast.

    ``axes`` is the scan calibration ``(x, y, units)`` from
    ``commit.navigation_extent`` — with it the map draws real-distance ticks
    and a scale bar; without it the axes are scan pixels. The colorbar is
    always on, labelled with the component's display unit."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    data = display_component(field, component)
    lo, hi = clim if clim is not None else _auto_clim(data)

    fig, figure_axes = apl.subplots(1, 1)
    ax = figure_axes[0][0] if isinstance(figure_axes, list) else figure_axes
    p = ax.imshow(np.nan_to_num(data, nan=0.0).astype(np.float32), cmap=cmap)
    try:
        p.set_clim(lo, hi)                      # diverging, centred on zero strain
    except Exception as e:
        logger.debug("set_clim on strain map failed: %s", e)
    if axes is not None:
        try:
            p.set_extent(axes[0], axes[1], units=axes[2])
        except Exception as e:
            logger.debug("calibrating the strain map failed: %s", e)
    try:
        p.set_colorbar_label(strain_quantity(component))
        p.set_colorbar_visible(True)
    except Exception as e:
        logger.debug("labelling the strain colorbar failed: %s", e)

    if ref_yx is not None:
        ry, rx = int(ref_yx[0]), int(ref_yx[1])
        L = max(2.0, 0.05 * max(field.nav_shape))     # crosshair half-length (px)
        # Markers are placed in data coordinates, which are scan pixels only
        # while the map is uncalibrated.
        cx, cy, lx, ly = float(rx), float(ry), L, L
        if axes is not None:
            x, y = np.asarray(axes[0], float), np.asarray(axes[1], float)
            cx = float(x[min(max(rx, 0), len(x) - 1)])
            cy = float(y[min(max(ry, 0), len(y) - 1)])
            lx = L * float(abs(x[1] - x[0])) if len(x) > 1 else L
            ly = L * float(abs(y[1] - y[0])) if len(y) > 1 else L
        p.add_lines([[[cx - lx, cy], [cx + lx, cy]], [[cx, cy - ly], [cx, cy + ly]]],
                    name="strain_ref", edgecolors="#00e5ff", linewidths=2.0)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html, p


def update_strain_view(p, field: StrainField, component: str, *,
                       clim: tuple[float, float] | None = None) -> None:
    """Live-update an existing strain plot in place: swap the component map
    and relabel the colorbar for it.
    ``clim`` = the user's dock-set contrast (None → fresh symmetric auto)."""
    data = display_component(field, component)
    lo, hi = clim if clim is not None else _auto_clim(data)
    try:
        p.set_data(np.nan_to_num(data, nan=0.0).astype(np.float32))
        p.set_clim(lo, hi)
        p.set_colorbar_label(strain_quantity(component))
    except Exception as e:
        logger.debug("updating strain map data failed: %s", e)

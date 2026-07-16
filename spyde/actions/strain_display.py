"""
strain_display.py — the strain-field visualization: a diverging component map
(εxx / εyy / εxy / ω) with the unstrained-reference crosshair.

The background colour encodes one component (diverging colormap centred on 0).
A component toggle swaps the shown map in place. (The old white principal-strain
ellipse glyphs were removed — they cluttered the map and bought nothing the
component colour didn't already show.)

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
from spyde.actions._common import STRAIN_TITLES as _COMPONENTS

logger = logging.getLogger(__name__)


def _component_map(field: StrainField, component: str) -> np.ndarray:
    return {
        "exx": field.exx, "eyy": field.eyy, "exy": field.exy, "omega": field.omega,
    }[component]


def _auto_clim(arr: np.ndarray) -> tuple[float, float]:
    """Symmetric colour limits from the robust (98th-pct) magnitude of the FINITE
    values (failed pixels are NaN and excluded, so they can't stretch the scale)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (-1.0, 1.0)
    v = float(np.percentile(np.abs(finite), 98))
    v = v if v > 0 else 1.0
    return (-v, v)


def emit_strain_histogram(window_id, field: StrainField, component: str,
                          clim: tuple[float, float]) -> None:
    """Send the sidebar histogram for the strain window — the same message shape
    ``Plot._emit_histogram`` uses, so the dock's contrast handles just work."""
    if window_id is None:
        return
    data = np.asarray(_component_map(field, component), float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    try:
        counts, edges = np.histogram(finite, bins=64)
        from spyde.backend.ipc import emit
        emit({
            "type": "histogram",
            "window_id": int(window_id),
            "counts": counts.astype(int).tolist(),
            "edges": [float(e) for e in edges],
            "vmin": float(clim[0]),
            "vmax": float(clim[1]),
            "threshold": None,
        })
    except Exception as e:
        logger.debug("strain histogram emit failed: %s", e)


def build_strain_figure(field: StrainField, *, component: str = "exx",
                        ref_yx=None, clim: tuple[float, float] | None = None,
                        cmap: str = "coolwarm"):
    """Build the strain view → ``(fig, fig_id, html, plot2d)``.
    ``plot2d`` is returned so a controller can live-update the component map /
    reference crosshair / contrast."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    data = _component_map(field, component)
    lo, hi = clim if clim is not None else _auto_clim(data)

    fig, axes = apl.subplots(1, 1)
    ax = axes[0][0] if isinstance(axes, list) else axes
    p = ax.imshow(np.nan_to_num(data, nan=0.0).astype(np.float32), cmap=cmap)
    try:
        p.set_clim(lo, hi)                      # diverging, centred on zero strain
    except Exception as e:
        logger.debug("set_clim on strain map failed: %s", e)

    if ref_yx is not None:
        ry, rx = int(ref_yx[0]), int(ref_yx[1])
        L = max(2.0, 0.05 * max(field.nav_shape))     # crosshair half-length (px)
        p.add_lines([[[rx - L, ry], [rx + L, ry]], [[rx, ry - L], [rx, ry + L]]],
                    name="strain_ref", edgecolors="#00e5ff", linewidths=2.0)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html, p


def update_strain_view(p, field: StrainField, component: str, *,
                       clim: tuple[float, float] | None = None) -> None:
    """Live-update an existing strain plot in place: swap the component map.
    ``clim`` = the user's dock-set contrast (None → fresh symmetric auto)."""
    data = _component_map(field, component)
    lo, hi = clim if clim is not None else _auto_clim(data)
    try:
        p.set_data(np.nan_to_num(data, nan=0.0).astype(np.float32))
        p.set_clim(lo, hi)
    except Exception as e:
        logger.debug("updating strain map data failed: %s", e)

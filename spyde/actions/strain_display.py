"""
strain_display.py — the strain-field visualization: a diverging component map
(εxx / εyy / εxy / ω) with the unstrained-reference crosshair.

The background colour encodes one component (diverging colormap centred on 0).
A component toggle swaps the shown map in place. (The old white principal-strain
ellipse glyphs were removed — they cluttered the map and bought nothing the
component colour didn't already show.)

No Qt. Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.strain_mapping import StrainField
from spyde.actions._common import STRAIN_TITLES as _COMPONENTS

logger = logging.getLogger(__name__)

_ALIVE: list = []


def _component_map(field: StrainField, component: str) -> np.ndarray:
    return {
        "exx": field.exx, "eyy": field.eyy, "exy": field.exy, "omega": field.omega,
    }[component]


def _auto_vmax(arr: np.ndarray) -> float:
    """Symmetric colour limit from the robust (98th-pct) magnitude."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(finite), 98))
    return v if v > 0 else 1.0


def build_strain_figure(field: StrainField, *, component: str = "exx",
                        ref_yx=None, vmax: float | None = None,
                        cmap: str = "coolwarm"):
    """Build the strain view → ``(fig, fig_id, html, plot2d)``.
    ``plot2d`` is returned so a controller can live-update the component map /
    reference crosshair."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    cmap_data = _component_map(field, component)
    v = float(vmax) if vmax is not None else _auto_vmax(cmap_data)

    fig, axes = apl.subplots(1, 1)
    ax = axes[0][0] if isinstance(axes, list) else axes
    p = ax.imshow(np.nan_to_num(cmap_data, nan=0.0).astype(np.float32), cmap=cmap)
    try:
        p.set_clim(-v, v)                       # diverging, centred on zero strain
    except Exception as e:
        logger.debug("set_clim on strain map failed: %s", e)

    if ref_yx is not None:
        ry, rx = int(ref_yx[0]), int(ref_yx[1])
        L = max(2.0, 0.05 * max(field.nav_shape))     # crosshair half-length (px)
        p.add_lines([[[rx - L, ry], [rx + L, ry]], [[rx, ry - L], [rx, ry + L]]],
                    name="strain_ref", edgecolors="#00e5ff", linewidths=2.0)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    _ALIVE.append(fig)
    return fig, fig_id, html, p


def update_strain_view(p, field: StrainField, component: str, *,
                       vmax: float | None = None) -> None:
    """Live-update an existing strain plot in place: swap the component map with
    a fresh symmetric colour limit."""
    data = _component_map(field, component)
    v = float(vmax) if vmax is not None else _auto_vmax(data)
    try:
        p.set_data(np.nan_to_num(data, nan=0.0).astype(np.float32))
        p.set_clim(-v, v)
    except Exception as e:
        logger.debug("updating strain map data failed: %s", e)

"""
strain_display.py — the strain-field visualization: a diverging component map
(εxx / εyy / εxy / ω) with the unstrained-reference crosshair.

The background colour encodes one component (diverging colormap centred on 0).
A component toggle swaps the shown map in place. (The old white principal-strain
ellipse glyphs were removed — they cluttered the map and bought nothing the
component colour didn't already show.)

Rendering is an RGBA composition (anyplotlib imshow renders (H, W, 4) floats
true-colour), which buys three things a plain colormapped array couldn't:
  * FAILED / masked pixels render neutral GRAY, not "zero strain" white
    (``np.nan_to_num`` used to paint them as perfectly unstrained);
  * an optional per-pixel CONFIDENCE weighting — alpha from the fit's coverage
    or RMS residual — fades unreliable pixels into the background instead of
    letting them shout over the signal;
  * a user-controllable colour range (``vmax``; 0/None = robust auto).

No Qt. Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.strain_mapping import StrainField
from spyde.actions._common import STRAIN_TITLES as _COMPONENTS

logger = logging.getLogger(__name__)

# Neutral gray for failed/masked pixels — visibly "no data" on the dark theme,
# distinct from the colormap's white (= zero strain) centre.
_MASK_RGBA = (0.26, 0.28, 0.32, 1.0)
WEIGHT_MODES = ("none", "coverage", "error")


def _component_map(field: StrainField, component: str) -> np.ndarray:
    return {
        "exx": field.exx, "eyy": field.eyy, "exy": field.exy, "omega": field.omega,
    }[component]


def _auto_vmax(arr: np.ndarray) -> float:
    """Symmetric colour limit from the robust (98th-pct) magnitude of the FINITE
    values (failed pixels are NaN and excluded, so they can't stretch the scale)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(finite), 98))
    return v if v > 0 else 1.0


def _confidence(field: StrainField, weight: str):
    """Per-pixel fit confidence in [0, 1] for ``weight`` ('coverage'|'error'),
    or None for 'none'/unavailable."""
    if weight == "coverage" and field.coverage is not None:
        return np.clip(np.nan_to_num(np.asarray(field.coverage, float), nan=0.0), 0.0, 1.0)
    if weight == "error" and getattr(field, "residual", None) is not None:
        r = np.asarray(field.residual, float)
        finite = r[np.isfinite(r)]
        # Soft Lorentzian roll-off scaled to the field's own typical residual:
        # a pixel at the median residual keeps q≈0.8; 2× the median q≈0.5.
        r0 = 2.0 * float(np.median(finite)) if finite.size else 0.0
        if r0 <= 0:
            return None
        return 1.0 / (1.0 + (np.nan_to_num(r, nan=np.inf) / r0) ** 2)
    return None


def compose_strain_rgba(field: StrainField, component: str, *,
                        vmax: float | None = None, weight: str = "none",
                        cmap: str = "coolwarm"):
    """Compose the display image → ``((H, W, 4) float32 RGBA, vmax_used)``.

    ``vmax`` = symmetric colour limit (None/0 → robust auto). ``weight`` = 'none'
    | 'coverage' | 'error': per-pixel alpha from the fit quality, so unreliable
    pixels fade instead of dominating. NaN (failed-fit) pixels are gray."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    data = np.asarray(_component_map(field, component), float)
    finite = np.isfinite(data)
    v = float(vmax) if vmax else _auto_vmax(data)
    rgba = colormaps[cmap](Normalize(vmin=-v, vmax=v, clip=True)(
        np.nan_to_num(data, nan=0.0)))
    q = _confidence(field, weight)
    if q is not None:
        rgba[..., 3] = 0.15 + 0.85 * np.clip(q, 0.0, 1.0)
    rgba[~finite] = _MASK_RGBA
    return rgba.astype(np.float32), v


def build_strain_figure(field: StrainField, *, component: str = "exx",
                        ref_yx=None, vmax: float | None = None,
                        weight: str = "none", cmap: str = "coolwarm"):
    """Build the strain view → ``(fig, fig_id, html, plot2d)``.
    ``plot2d`` is returned so a controller can live-update the component map /
    reference crosshair."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    rgba, _v = compose_strain_rgba(field, component, vmax=vmax, weight=weight,
                                   cmap=cmap)
    fig, axes = apl.subplots(1, 1)
    ax = axes[0][0] if isinstance(axes, list) else axes
    p = ax.imshow(rgba)

    if ref_yx is not None:
        ry, rx = int(ref_yx[0]), int(ref_yx[1])
        L = max(2.0, 0.05 * max(field.nav_shape))     # crosshair half-length (px)
        p.add_lines([[[rx - L, ry], [rx + L, ry]], [[rx, ry - L], [rx, ry + L]]],
                    name="strain_ref", edgecolors="#00e5ff", linewidths=2.0)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html, p


def update_strain_view(p, field: StrainField, component: str, *,
                       vmax: float | None = None, weight: str = "none",
                       cmap: str = "coolwarm") -> None:
    """Live-update an existing strain plot in place: recompose the RGBA image for
    the component / colour range / confidence weighting."""
    rgba, _v = compose_strain_rgba(field, component, vmax=vmax, weight=weight,
                                   cmap=cmap)
    try:
        p.set_data(rgba)
    except Exception as e:
        logger.debug("updating strain map data failed: %s", e)

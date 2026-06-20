"""
strain_display.py — the strain-field visualization: a diverging component map
(εxx / εyy / εxy / ω) overlaid with principal-strain **ellipse glyphs** and the
unstrained-reference crosshair.

The background colour encodes one component (diverging colormap centred on 0);
the white ellipse glyphs encode the *full* local strain tensor — each is the
(amplified) deformation of a unit circle, so its elongation direction is the
principal-strain axis and its shape shows tension vs compression at a glance.
This is the "component map + glyph overlay" view chosen in the design pass.

No Qt. Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

import numpy as np

from spyde.actions.strain_mapping import StrainField, principal_strain

_ALIVE: list = []

# Diverging colormap (colorcet alias → matplotlib-free) centred on zero strain.
_COMPONENTS = {
    "exx": "εxx", "eyy": "εyy", "exy": "εxy", "omega": "ω",
}


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


def _glyph_ellipses(field: StrainField, *, step: int, amp: float):
    """Principal-strain ellipse glyphs on a coarse grid.

    Returns ``(offsets, widths, heights, angles_deg)`` for ``add_ellipses`` —
    image-pixel coordinates (x=col, y=row). A unit circle of radius ``base`` is
    deformed by the (amplified) principal strains, so the ellipse axis lengths
    are ``base·(1 + amp·ε1)`` and ``base·(1 + amp·ε2)`` along the principal angle.
    """
    e1, e2, theta = principal_strain(field.exx, field.eyy, field.exy)
    ny, nx = field.nav_shape
    base = 0.42 * step
    offs, wid, hei, ang = [], [], [], []
    for iy in range(step // 2, ny, step):
        for ix in range(step // 2, nx, step):
            a, b, t = e1[iy, ix], e2[iy, ix], theta[iy, ix]
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(t)):
                continue
            offs.append([float(ix), float(iy)])
            wid.append(float(2.0 * base * max(0.1, 1.0 + amp * a)))   # full width
            hei.append(float(2.0 * base * max(0.1, 1.0 + amp * b)))
            ang.append(float(np.degrees(t)))
    return np.array(offs, float).reshape(-1, 2), np.array(wid), np.array(hei), np.array(ang)


def build_strain_figure(field: StrainField, *, component: str = "exx",
                        ref_yx=None, glyph_step: int | None = None,
                        glyph_amp: float = 60.0, vmax: float | None = None,
                        cmap: str = "coolwarm", glyphs: bool = True):
    """Build the strain view → ``(fig, fig_id, html, plot2d)``. ``plot2d`` is
    returned so a controller can live-update the data / glyphs / reference."""
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
    except Exception:
        pass

    glyph_group = None
    if glyphs:
        ny, nx = field.nav_shape
        step = glyph_step or max(1, int(round(max(ny, nx) / 16)))
        offs, wid, hei, ang = _glyph_ellipses(field, step=step, amp=glyph_amp)
        if len(offs):
            glyph_group = p.add_ellipses(offs, wid, hei, name="strain_glyphs",
                                         angles=ang, facecolors=None,
                                         edgecolors="#ffffff", linewidths=1.3,
                                         alpha=0.0)

    if ref_yx is not None:
        ry, rx = int(ref_yx[0]), int(ref_yx[1])
        L = max(2.0, 0.05 * max(field.nav_shape))     # crosshair half-length (px)
        p.add_lines([[[rx - L, ry], [rx + L, ry]], [[rx, ry - L], [rx, ry + L]]],
                    name="strain_ref", edgecolors="#00e5ff", linewidths=2.0)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    _ALIVE.append(fig)
    return fig, fig_id, html, p, glyph_group


def update_strain_view(p, field: StrainField, component: str, glyph_group, *,
                       vmax: float | None = None, glyph_step: int | None = None,
                       glyph_amp: float = 60.0) -> None:
    """Live-update an existing strain plot in place: swap the component map (with
    a fresh symmetric colour limit) and re-draw the principal-strain glyphs."""
    data = _component_map(field, component)
    v = float(vmax) if vmax is not None else _auto_vmax(data)
    try:
        p.set_data(np.nan_to_num(data, nan=0.0).astype(np.float32))
        p.set_clim(-v, v)
    except Exception:
        pass
    if glyph_group is not None:
        ny, nx = field.nav_shape
        step = glyph_step or max(1, int(round(max(ny, nx) / 16)))
        offs, wid, hei, ang = _glyph_ellipses(field, step=step, amp=glyph_amp)
        try:
            glyph_group.set(offsets=offs, widths=wid, heights=hei, angles=ang)
        except Exception:
            pass

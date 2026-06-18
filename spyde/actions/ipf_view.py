"""
ipf_view.py — the 3-D IPF explorer (anyplotlib ``scatter3d`` on the unit sphere).

An orientation IPF window normally shows the 2-D RGB map. This adds a SECOND
figure to that same window — a 3-D scatter of every position's reduced crystal
direction on the unit sphere, coloured by its IPF colour — tagged ``view="3d"``
so the frontend renders a 2D ⇄ 3D toggle (the
``cssfrancis.github.io/anyplotlib`` IPF-explorer view).

Works for BOTH the dense Orientation Mapping result (``SpyDEOrientationMap``)
and the Vector Orientation result (which exposes ``to_orientation_map()``).
No Qt.
"""
from __future__ import annotations

import numpy as np

# Keep figures alive past the emit so they are not garbage-collected while the
# iframe is still mounted (the _electron registry holds a weak association only).
_ALIVE: list = []


def _as_orientation_map(result):
    """Normalise either result type to a SpyDEOrientationMap (has
    ``ipf_sphere_points``)."""
    if hasattr(result, "to_orientation_map"):
        return result.to_orientation_map()
    return result


def build_ipf_3d_figure(xyz: np.ndarray, rgb: np.ndarray):
    """Build the 3-D IPF scatter figure → ``(fig, fig_id, html)``."""
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    fig, axes = apl.subplots(1, 1)
    ax = axes[0][0] if isinstance(axes, list) else axes
    colors = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    p3d = ax.scatter3d(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        colors=colors, point_size=6,
        x_label="[100]", y_label="[010]", z_label="[001]",
        bounds=((-1, 1),) * 3, zoom=1.4,
    )
    try:
        p3d.set_sphere(1.0)
    except Exception:
        pass

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    _ALIVE.append(fig)
    return fig, fig_id, html


def emit_ipf_3d(window_id: int, result, direction: str = "z") -> bool:
    """Compute the sphere points from *result* and emit a ``view="3d"`` figure for
    *window_id*. Returns True if a 3-D figure was emitted (False if no points)."""
    from spyde.backend.ipc import emit

    om = _as_orientation_map(result)
    try:
        xyz, rgb = om.ipf_sphere_points(direction)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("ipf_sphere_points failed: %s", e)
        return False
    if xyz is None or len(xyz) == 0:
        return False

    _fig, fig_id, html = build_ipf_3d_figure(np.asarray(xyz), np.asarray(rgb))
    emit({
        "type": "figure", "fig_id": fig_id, "window_id": window_id,
        "html": html, "title": "IPF (3D)", "is_navigator": False, "view": "3d",
    })
    return True


def attach_ipf_3d(tree, result, direction: str = "z") -> bool:
    """Add the 3-D IPF view to *tree*'s IPF window (its first signal plot)."""
    sp = next(iter(getattr(tree, "signal_plots", []) or []), None)
    wid = getattr(sp, "window_id", None)
    if wid is None:
        return False
    return emit_ipf_3d(wid, result, direction)

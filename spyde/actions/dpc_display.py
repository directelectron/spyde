"""
dpc_display.py — the DPC result view.

The map is either the RGB direction image (hue = field direction, brightness =
magnitude) or one scalar component (Ex, Ey, |E|, divergence, curl). Over it sits
the colour wheel: the legend saying which hue means which direction.

Three things here are deliberate and will look wrong to change:

* **The wheel is a key, not an inset.** A key is a static picture floating in
  screen space with no chrome, like a scale bar, and it lands in a PNG export.
  An inset is a window with a title bar and its own canvas to keep painted.
* **The wheel is always shown, not on hover.** The hues mean nothing without it.
* **The wheel is built ONCE**, at ``dpc.DISPLAY_ROTATION``. The rotation is
  applied to the shift vectors before colouring, so the colour→direction mapping
  never changes. Rebuilding it at ``result.rotation`` applies that angle twice
  and points the legend away from the map it describes.

Contrast belongs to the plot dock, not the wizard — same as ``strain_display``.

No Qt. Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions import dpc as _dpc
from spyde.actions._common import robust_map_limits

logger = logging.getLogger(__name__)

#: Key geometry: width as a fraction of the plot area's SHORTER side, which
#: corner it pins to, and the rendered resolution of the wheel image itself.
WHEEL_SIZE = 0.26
WHEEL_CORNER = "bottom-right"
WHEEL_PX = 192
#: The scale caption is centred under the key and is WIDER than it, so it
#: overhangs on both sides. At the default 10 px margin that overhang runs off
#: the panel and the units get cut in half. Keep the gap bigger than the
#: overhang, and keep the caption short — see `wheel_scale_label`.
WHEEL_MARGIN = 16.0
WHEEL_LABEL_SIZE = 8.0
#: No card behind it — the wheel's own alpha makes it a disc on the map rather
#: than a rectangle sitting on top of it.
WHEEL_BG = "none"
#: Compass points, in key-image fractions.
#:
#: The wheel is drawn in the map's own SCREEN frame, so these sit exactly where
#: they read. They name the scan AXES rather than "up"/"down": +y points DOWN
#: on screen (image convention, the same direction the navigator's y axis
#: increases), so a wheel labelled "up" would be describing −y — and a legend
#: that needs a footnote is not a legend.
#:
#: Pulled INSIDE the disc rather than sitting on its edge, where they hung off
#: the wheel and had the map behind them. Black, because the rim is bright and
#: fully saturated at every angle — white reads on the blues and disappears on
#: the yellows.
WHEEL_LABELS = [
    {"x": 0.50, "y": 0.15, "text": "−y", "size": 10, "color": "#000000",
     "align": "center"},
    {"x": 0.50, "y": 0.88, "text": "+y", "size": 10, "color": "#000000",
     "align": "center"},
    {"x": 0.13, "y": 0.54, "text": "−x", "size": 10, "color": "#000000",
     "align": "left"},
    {"x": 0.87, "y": 0.54, "text": "+x", "size": 10, "color": "#000000",
     "align": "right"},
]

#: The RGB direction map isn't one of `dpc.COMPONENTS` — it's the default view.
RGB_VIEW = "rgb"
VIEWS: tuple[str, ...] = (RGB_VIEW,) + _dpc.COMPONENTS

#: Components whose zero is meaningful → diverging map, symmetric limits.
_SIGNED = ("fx", "fy", "divergence", "curl")


def view_titles(mode: str, units: str) -> dict[str, str]:
    """Display label per view, including the RGB one."""
    titles = {RGB_VIEW: "Direction + magnitude"}
    titles.update(_dpc.component_titles(mode, units))
    return titles


def _auto_clim(arr: np.ndarray, signed: bool) -> tuple[float, float]:
    """Robust display limits — symmetric about 0 for a signed component."""
    return robust_map_limits(arr, symmetric=signed)


def view_array(result: "_dpc.DpcResult", view: str
               ) -> tuple[np.ndarray, tuple[float, float] | None, str]:
    """``(image, clim, colormap)`` for *view* — RGB or a scalar component.

    Non-finite values are painted as 0 (the same choice ``strain_display``
    makes for failed fits) but are EXCLUDED from the contrast, so positions
    still streaming in during a progressive fill neither blank the map nor
    stretch its scale.
    """
    if view == RGB_VIEW:
        return result.rgb, None, "gray"
    arr = np.asarray(result.component(view), dtype=np.float32)
    signed = view in _SIGNED
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    cmap = "coolwarm" if signed else ("hsv" if view == "phase" else "viridis")
    if view == "phase":
        return clean, (0.0, float(2 * np.pi)), cmap
    # Contrast from `arr` (non-finite excluded by `_auto_clim`), pixels from
    # `clean` — so an unmeasured position is drawn, not scaled against.
    return clean, _auto_clim(arr, signed), cmap


def emit_dpc_histogram(window_id, result: "_dpc.DpcResult", view: str,
                       clim: tuple[float, float] | None) -> None:
    """Sidebar histogram for a scalar view — the same message ``Plot`` sends, so
    the dock's contrast handles work with no special casing. The RGB view has no
    scalar distribution, so it sends nothing."""
    if window_id is None or view == RGB_VIEW:
        return
    data = np.asarray(result.component(view), dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return
    lo, hi = clim if clim is not None else _auto_clim(finite, view in _SIGNED)
    try:
        counts, edges = np.histogram(finite, bins=64)
        from de_shell.ipc import emit
        emit({"type": "histogram", "window_id": int(window_id),
              "counts": counts.astype(int).tolist(),
              "edges": [float(e) for e in edges],
              "vmin": float(lo), "vmax": float(hi), "threshold": None})
    except Exception as e:                                   # pragma: no cover
        logger.debug("DPC histogram emit failed: %s", e)


def build_dpc_figure(result: "_dpc.DpcResult", *, view: str = RGB_VIEW,
                     title: str = "DPC"):
    """Build the DPC window → ``(fig, fig_id, html, plot2d, wheel_key)``.

    *plot2d* is returned so the controller can live-update the map without
    rebuilding the figure (a rebuild would drop the user's zoom and flash the
    window on every slider tick). *wheel_key* is the legend's handle, used only
    to show/hide it — its picture never changes.
    """
    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    data, clim, cmap = view_array(result, view)
    fig, axes = apl.subplots(1, 1)
    ax = axes[0][0] if isinstance(axes, list) else axes
    p = ax.imshow(data, cmap=None if view == RGB_VIEW else cmap)
    if clim is not None:
        try:
            p.set_clim(*clim)
        except Exception as e:                               # pragma: no cover
            logger.debug("set_clim on DPC map failed: %s", e)

    wheel_key = attach_wheel_key(p, visible=(view == RGB_VIEW),
                                 scale=wheel_scale_label(result))

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html, p, wheel_key


def wheel_scale_label(result: "_dpc.DpcResult | None") -> str | None:
    """Caption for the wheel: what its centre and rim are worth in real units.

    Hue answers "which way"; without this the picture never answers "how much".
    ``None`` when the ceiling cannot be derived — see ``dpc.magnitude_ceiling``.
    """
    if result is None:
        return None
    sigma = float((result.params or {}).get("autolim_sigma", 4.0))
    ceiling = _dpc.magnitude_ceiling(result.field, autolim_sigma=sigma)
    if ceiling is None:
        return None
    # Two significant figures, no spaces around the dash. The caption has to
    # stay narrower than the key plus its margin or it is clipped by the panel
    # edge, and a third figure on a display ceiling is false precision anyway.
    return f"0–{ceiling:.2g} {result.units}"


def attach_wheel_key(plot2d, *, visible: bool = True, scale: str | None = None):
    """Pin the direction legend over *plot2d*. ``None`` on an anyplotlib without
    ``add_key`` (< 0.7.0), which is a missing legend, not a broken window — so
    the caller carries on."""
    add_key = getattr(plot2d, "add_key", None)
    if add_key is None:                                      # pragma: no cover
        logger.debug("this anyplotlib has no add_key; skipping the DPC wheel")
        return None
    try:
        # Built ONCE, at the map's own constant display rotation — see
        # dpc.DISPLAY_ROTATION for why this never has to track anything.
        return add_key(
            _dpc.color_wheel_rgba(WHEEL_PX, rotation=_dpc.DISPLAY_ROTATION),
            corner=WHEEL_CORNER, size=WHEEL_SIZE, bgcolor=WHEEL_BG,
            margin=WHEEL_MARGIN, hover_only=False, visible=bool(visible),
            labels=WHEEL_LABELS, label=scale, label_size=WHEEL_LABEL_SIZE,
            name="dpc_wheel")
    except Exception as e:                                   # pragma: no cover
        logger.debug("attaching the DPC colour-wheel key failed: %s", e)
        return None


def attach_wheel_key_to_tree(tree, result: "_dpc.DpcResult") -> None:
    """Put the legend on a COMMITTED tree's map as well as the live one.

    A committed tree is the artefact that gets saved, exported and shown to
    someone else, so it is the copy that most needs to say what its hues mean.
    """
    signal_plot = next(iter(getattr(tree, "signal_plots", []) or []), None)
    plot2d = getattr(signal_plot, "_plot2d", None)
    if plot2d is None:
        logger.debug("committed DPC tree has no live 2-D plot; no wheel added")
        return
    attach_wheel_key(plot2d, visible=True, scale=wheel_scale_label(result))


def update_dpc_view(plot2d, wheel_key, result: "_dpc.DpcResult", view: str,
                    *, clim: tuple[float, float] | None = None,
                    cmap: str | None = None) -> None:
    """Repaint an existing DPC window in place: swap the view and/or the field.

    *clim* is the user's dock-set contrast (``None`` → re-derive). The wheel is
    hidden for a scalar view — a hue legend left over a divergence map describes
    something that isn't on screen — but its picture is never re-sent.
    """
    data, auto_clim, auto_cmap = view_array(result, view)
    try:
        plot2d.set_data(data)
    except Exception as e:                                   # pragma: no cover
        logger.debug("updating the DPC map failed: %s", e)
        return
    if view != RGB_VIEW:
        lo, hi = clim if clim is not None else (auto_clim or (0.0, 1.0))
        try:
            plot2d.set_clim(float(lo), float(hi))
            plot2d.set_colormap(cmap or auto_cmap)
        except Exception as e:                               # pragma: no cover
            logger.debug("updating DPC contrast failed: %s", e)
    show_wheel_key(wheel_key, visible=(view == RGB_VIEW))


def show_wheel_key(wheel_key, *, visible: bool) -> None:
    """Show or hide the legend. Its PICTURE is never re-sent — restyling a key
    rides the geometry channel, so a hover/visibility toggle costs nothing."""
    if wheel_key is None:
        return
    try:
        wheel_key.set(visible=bool(visible))
    except Exception as e:                                   # pragma: no cover
        logger.debug("toggling the DPC colour-wheel key failed: %s", e)

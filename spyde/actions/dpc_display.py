"""
dpc_display.py — the DPC result view: an RGB direction map with a circular
phase-direction legend, plus the scalar component maps.

The window has two parts:

* the **map** — either the magnitude/phase RGB image (hue = field direction,
  brightness = magnitude) or one scalar component (Ex, Ey, |E|, divergence,
  curl) on a diverging colormap;
* the **colour wheel** — the legend saying which hue means which direction,
  pinned over the map as a native anyplotlib **key** (``Plot2D.add_key``), the
  same primitive the IPF colour triangle uses.

A key, not an inset. An inset is a floating window with a title bar and its own
canvas stack — the right tool for a live sub-plot, and the wrong one for a
legend: it sat on the map like a panel, and it was one more thing to keep
painted. A key is a static picture that floats in screen space with no chrome,
reads as part of the figure the way a scale bar does, and is included in a PNG
export.

It is shown ALWAYS, not on hover. A direction map is unreadable without its
key — the hues mean nothing on their own — so hiding it until the pointer
arrives hides the thing that makes the picture interpretable. It costs a corner
of a map whose corners carry no data anyway. (``hover_only`` is still what a
dense, self-explanatory map like the IPF triangle's wants; this one is not
that.)

The wheel is built ONCE. The scan↔detector rotation is applied to the shift
VECTORS before the field is coloured, so the colour→direction mapping is the
same at every rotation (``dpc.DISPLAY_ROTATION``). Re-rendering the legend at
``result.rotation`` — the obvious-looking thing, and what an earlier version
did — applies that angle a second time and leaves the legend pointing
``rotation`` degrees away from the map it describes.

Contrast follows the same rule as ``strain_display``: it belongs to the plot
widget dock, not to the wizard. Scalar components emit the standard sidebar
histogram and honour the dock's ``set_clim`` / ``set_colormap``. The RGB map has
no scalar contrast to set, so it emits none.

No Qt. Host-agnostic (Electron + Jupyter).
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions import dpc as _dpc

logger = logging.getLogger(__name__)

#: Key geometry: width as a fraction of the plot area's SHORTER side, which
#: corner it pins to, and the rendered resolution of the wheel image itself.
WHEEL_SIZE = 0.26
WHEEL_CORNER = "bottom-right"
WHEEL_PX = 192
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
WHEEL_LABELS = [
    {"x": 0.5, "y": 0.045, "text": "−y", "size": 9, "align": "center"},
    {"x": 0.5, "y": 0.985, "text": "+y", "size": 9, "align": "center"},
    {"x": 0.02, "y": 0.53, "text": "−x", "size": 9, "align": "left"},
    {"x": 0.98, "y": 0.53, "text": "+x", "size": 9, "align": "right"},
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
    """Robust display limits — symmetric about 0 for a signed component.

    The 98th percentile of |value| rather than the extremes: a handful of
    failed-fit or edge pixels otherwise stretch the scale until the real
    structure is a flat mid-tone.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if signed:
        v = float(np.percentile(np.abs(finite), 98)) or 1.0
        return (-v, v)
    lo = float(np.percentile(finite, 2))
    hi = float(np.percentile(finite, 98))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


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

    wheel_key = attach_wheel_key(p, visible=(view == RGB_VIEW))

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    return fig, fig_id, html, p, wheel_key


def attach_wheel_key(plot2d, *, visible: bool = True):
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
            hover_only=False, visible=bool(visible), labels=WHEEL_LABELS,
            name="dpc_wheel")
    except Exception as e:                                   # pragma: no cover
        logger.debug("attaching the DPC colour-wheel key failed: %s", e)
        return None


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

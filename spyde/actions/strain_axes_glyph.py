"""
strain_axes_glyph.py — the scan's x and y drawn on the reference diffraction
pattern, and turnable by hand.

The strain fit lives in the detector's frame; the map is read in the scan's.
Which way the scan's axes point on the detector is the one number that ties
the two, and a number in a slider is hard to sanity-check. Two arrows on the
pattern are not: drag the head of either and the tensor turns with it (the
same ``rotation`` / ``flip`` the caret shows), so a reflection you know to lie
along the scan's x can simply be pointed at.

Widgets are placed in image pixels (anyplotlib's contract for 2-D widgets),
so the glyph needs no calibration — the origin is the pattern's centre and
the arrows a fixed fraction of its size.
"""
from __future__ import annotations

import logging
import math

from spyde.drawing.selectors.base_selector import event_handler_fn

logger = logging.getLogger(__name__)

X_COLOR = "#ff6ec7"          # pink — distinct from the cyan crosshair and the
Y_COLOR = "#ffd23f"          # gold   green / grey / orange selection overlay
LABEL_COLOR = "#f8f8f2"      # the letters: neutral, so an arrow's colour is
                             # the arrow alone (a test finds its head by it)
LENGTH_FRACTION = 0.18       # arrow length as a fraction of the pattern's size


def scan_axes_on_detector(rotation: float, flip: bool) -> tuple[tuple[float, float],
                                                                tuple[float, float]]:
    """Unit vectors of the SCAN's x and y expressed in detector (image-pixel)
    coordinates, for the frame ``rotate_strain_basis`` defines.

    That transform takes a detector vector to the scan frame with the clockwise
    ``R = [[c, -s], [s, c]]`` after an optional x/y swap, so a scan axis on the
    detector is the inverse: ``Rᵀ`` on the unit vector, then the swap.
    """
    theta = math.radians(float(rotation))
    c, s = math.cos(theta), math.sin(theta)
    x_axis, y_axis = (c, -s), (s, c)
    if flip:
        x_axis, y_axis = (x_axis[1], x_axis[0]), (y_axis[1], y_axis[0])
    return x_axis, y_axis


def rotation_from_direction(u: float, v: float, *, axis: str, flip: bool) -> float:
    """The ``rotation`` (degrees) whose scan *axis* (``"x"`` or ``"y"``) points
    along the detector vector ``(u, v)`` — the inverse of
    :func:`scan_axes_on_detector`, for a dragged arrow."""
    if axis == "x":
        theta = math.atan2(-u, v) if flip else math.atan2(-v, u)
    else:
        theta = math.atan2(v, u) if flip else math.atan2(u, v)
    return math.degrees(theta)


class ScanAxesGlyph:
    """Two draggable arrows (and their labels) on a diffraction-pattern plot.

    ``on_change(rotation_deg)`` is called when the user drags either arrow;
    :meth:`set_rotation` moves the glyph when the angle changes elsewhere.
    """

    def __init__(self, dp_plot, *, rotation: float = 0.0, flip: bool = False,
                 on_change=None):
        plot2d = getattr(dp_plot, "_plot2d", None)
        if plot2d is None:
            raise ValueError("the reference pattern has no 2-D figure to draw on")
        self.plot2d = plot2d
        self.rotation = float(rotation)
        self.flip = bool(flip)
        self.on_change = on_change
        state = plot2d._state
        width = float(state.get("image_width") or 1)
        height = float(state.get("image_height") or 1)
        self.origin = (width / 2.0, height / 2.0)
        self.length = LENGTH_FRACTION * min(width, height)

        (ux, vx), (uy, vy) = self._vectors()
        ox, oy = self.origin
        self.x_arrow = plot2d.add_arrow_widget(x=ox, y=oy, u=ux, v=vx,
                                               color=X_COLOR, linewidth=2.5)
        self.y_arrow = plot2d.add_arrow_widget(x=ox, y=oy, u=uy, v=vy,
                                               color=Y_COLOR, linewidth=2.5)
        # The letters are MARKERS, not widgets: a label widget is grabbable
        # within 15 px of its anchor, which on a small pattern is the arrow
        # head itself — the drag went to the letter and the arrow never turned.
        self.labels = plot2d.add_circles(
            self._label_offsets(), name="strain_axes_labels", radius=0.5,
            size_units="px", facecolors=None, edgecolors=LABEL_COLOR, alpha=0.0,
            labels=["x", "y"], transform="axes")
        # Bound methods cannot carry the attribute add_event_handler sets, and
        # the wrappers must stay referenced or the registration is collected.
        self._x_handler = event_handler_fn(self._on_x_drag)
        self._y_handler = event_handler_fn(self._on_y_drag)
        for event in ("pointer_move", "pointer_up"):
            self.x_arrow.add_event_handler(self._x_handler, event)
            self.y_arrow.add_event_handler(self._y_handler, event)

    # ── geometry ─────────────────────────────────────────────────────────────
    def _vectors(self):
        (xx, xy), (yx, yy) = scan_axes_on_detector(self.rotation, self.flip)
        L = self.length
        return (xx * L, xy * L), (yx * L, yy * L)

    def _label_offsets(self) -> list:
        """Where the letters go, as axes fractions (x right, y UP — the
        markers' "axes" transform), a little past each arrow head."""
        state = self.plot2d._state
        width = float(state.get("image_width") or 1)
        height = float(state.get("image_height") or 1)
        ox, oy = self.origin
        out = []
        for (u, v) in self._vectors():
            px, py = ox + u * 1.3, oy + v * 1.3
            out.append([px / width, 1.0 - py / height])
        return out

    def _place(self) -> None:
        (ux, vx), (uy, vy) = self._vectors()
        ox, oy = self.origin
        # `_notify=False`: this is us moving the widgets, not the user.
        self.x_arrow.set(_notify=False, x=ox, y=oy, u=ux, v=vx)
        self.y_arrow.set(_notify=False, x=ox, y=oy, u=uy, v=vy)
        self.labels.set(offsets=self._label_offsets(), labels=["x", "y"])

    def set_rotation(self, rotation: float, flip: bool | None = None) -> None:
        self.rotation = float(rotation)
        if flip is not None:
            self.flip = bool(flip)
        self._place()

    # ── user drags ───────────────────────────────────────────────────────────
    def _dragged(self, widget, axis: str) -> None:
        try:
            u, v = float(widget.u), float(widget.v)
        except Exception:
            return
        if math.hypot(u, v) < 1e-9:
            self._place()
            return
        self.rotation = rotation_from_direction(u, v, axis=axis, flip=self.flip)
        self._place()               # re-pin the tail and the other arrow
        if self.on_change is not None:
            try:
                self.on_change(self.rotation)
            except Exception as e:
                logger.debug("axes glyph on_change failed: %s", e)

    def _on_x_drag(self, _event=None) -> None:
        self._dragged(self.x_arrow, "x")

    def _on_y_drag(self, _event=None) -> None:
        self._dragged(self.y_arrow, "y")

    # ── lifecycle ────────────────────────────────────────────────────────────
    @property
    def widgets(self):
        return (self.x_arrow, self.y_arrow)

    def set_visible(self, visible: bool) -> None:
        for widget in self.widgets:
            try:
                widget.show() if visible else widget.hide()
            except Exception as e:
                logger.debug("axes glyph visibility failed: %s", e)
        try:
            if visible:
                self.labels.set(offsets=self._label_offsets(), labels=["x", "y"])
            else:
                self.labels.set(offsets=[], labels=[])
        except Exception as e:
            logger.debug("axes glyph label visibility failed: %s", e)

    def remove(self) -> None:
        for widget in self.widgets:
            try:
                widget.remove()
            except Exception as e:
                logger.debug("removing an axes glyph widget failed: %s", e)
        try:
            self.labels.remove()
        except Exception as e:
            logger.debug("removing the axes glyph labels failed: %s", e)

"""2D Selectors using anyplotlib interactive widgets."""
from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING, Union, List

from spyde.drawing.selectors.base_selector import (
    BaseSelector,
    IntegratingSelectorMixin,
    event_handler_fn,
)

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow

logger = logging.getLogger(__name__)


class CrosshairSelector(BaseSelector):
    """Point selector — wraps anyplotlib CrosshairWidget."""

    def __init__(
        self,
        parent: Union["PlotWindow", "Plot"],
        children: Union["Plot", List["Plot"]],
        update_function: Union[callable, List[callable]],
        live_delay: int = 2,
        multi_selector: bool = False,
        **kwargs,
    ):
        super().__init__(
            parent,
            children,
            update_function,
            live_delay=live_delay,
            multi_selector=multi_selector,
            color=kwargs.get("color", "green"),
        )
        self._widget = None
        self.roi = None
        plot2d = self._get_plot2d()
        if plot2d is not None:
            try:
                self._widget = plot2d.add_crosshair_widget(color=self.color)
                self.roi = self._widget
                self._event_cb = event_handler_fn(self._on_pointer_up)
                self._widget.add_event_handler(self._event_cb, "pointer_move", "pointer_up")
            except Exception as e:
                logger.debug("CrosshairSelector widget init failed: %s", e)

    def _get_plot2d(self):
        plot = self.current_plot
        return getattr(plot, "_plot2d", None) if plot is not None else None

    def _on_pointer_up(self, event):
        self.update_data()

    def _get_selected_indices(self) -> np.ndarray:
        if self._widget is None:
            return np.array([[0, 0]])
        # Widget cx/cy are already IMAGE-PIXEL coordinates (anyplotlib's 2-D
        # widgets report pixels, not calibrated data units), so they ARE the
        # array index — just round. See BaseSelector._data_to_index.
        cx, cy = self._data_to_index(float(self._widget.cx), float(self._widget.cy))
        return np.array([[cx, cy]])

    def add_linked_roi(self, plot: "Plot") -> None:
        pass

    def translate_pixels(self, shift_x: int, shift_y: int) -> None:
        if self._widget is not None:
            try:
                self._widget.cx = float(self._widget.cx) + shift_x
                self._widget.cy = float(self._widget.cy) + shift_y
            except Exception as e:
                logger.debug("translating crosshair selector failed: %s", e)


class RectangleSelector(BaseSelector):
    """Area selector — wraps anyplotlib RectangleWidget."""

    def __init__(
        self,
        parent: Union["PlotWindow", "Plot"],
        children: Union["Plot", List["Plot"]],
        update_function: Union[callable, List[callable]],
        live_delay: int = 2,
        multi_selector: bool = False,
        **kwargs,
    ):
        super().__init__(
            parent,
            children,
            update_function,
            live_delay=live_delay,
            multi_selector=multi_selector,
            color=kwargs.get("color", "green"),
        )
        self._widget = None
        self.roi = None
        plot2d = self._get_plot2d()
        if plot2d is not None:
            try:
                self._widget = plot2d.add_rectangle_widget(color=self.color)
                self.roi = self._widget
                self._event_cb = event_handler_fn(self._on_pointer_up)
                self._widget.add_event_handler(self._event_cb, "pointer_move", "pointer_up")
            except Exception as e:
                logger.debug("RectangleSelector widget init failed: %s", e)

    def _get_plot2d(self):
        plot = self.current_plot
        return getattr(plot, "_plot2d", None) if plot is not None else None

    def _on_pointer_up(self, event):
        self.update_data()

    def _get_selected_indices(self) -> np.ndarray:
        if self._widget is None:
            return np.array([[0, 0]])
        x = float(self._widget.x)
        y = float(self._widget.y)
        w = float(self._widget.w)
        h = float(self._widget.h)

        # The rectangle's x/y/w/h are already in IMAGE-PIXEL coordinates
        # (anyplotlib 2-D widgets report pixels), so the bounds map straight to
        # array indices — see BaseSelector._data_to_index.
        x0, y0 = self._data_to_index(x, y)
        x1, y1 = self._data_to_index(x + w, y + h)

        x_indices = np.arange(x0, max(x0 + 1, x1), dtype=int)
        y_indices = np.arange(y0, max(y0 + 1, y1), dtype=int)

        grid = np.array(np.meshgrid(x_indices, y_indices)).T.reshape(-1, 2)
        return grid

    def add_linked_roi(self, plot: "Plot") -> None:
        pass

    def translate_pixels(self, shift_x: int, shift_y: int) -> None:
        if self._widget is not None:
            try:
                self._widget.x = float(self._widget.x) + shift_x
                self._widget.y = float(self._widget.y) + shift_y
            except Exception as e:
                logger.debug("translating rectangle selector failed: %s", e)


class CircleSelector(BaseSelector):
    """Disk selector — wraps anyplotlib CircleWidget."""

    def __init__(self, parent, children, update_function,
                 live_delay: int = 2, multi_selector: bool = False, **kwargs):
        super().__init__(parent, children, update_function,
                         live_delay=live_delay, multi_selector=multi_selector,
                         color=kwargs.get("color", "green"))
        self._widget = None
        self.roi = None
        plot2d = self._get_plot2d()
        if plot2d is not None:
            try:
                self._widget = plot2d.add_circle_widget(color=self.color)
                self.roi = self._widget
                self._event_cb = event_handler_fn(self._on_pointer_up)
                self._widget.add_event_handler(self._event_cb, "pointer_move", "pointer_up")
            except Exception as e:
                logger.debug("CircleSelector widget init failed: %s", e)

    def _get_plot2d(self):
        plot = self.current_plot
        return getattr(plot, "_plot2d", None) if plot is not None else None

    def _on_pointer_up(self, event):
        self.update_data()

    def _get_selected_indices(self) -> np.ndarray:
        if self._widget is None:
            return np.array([[0, 0]])
        # Geometry encoded as ints so change-detection fires on move/resize.
        cx = int(round(float(self._widget.cx)))
        cy = int(round(float(self._widget.cy)))
        r = int(round(float(self._widget.r)))
        return np.array([[cx, cy, r]])

    def add_linked_roi(self, plot) -> None:
        pass


class AnnularSelector(BaseSelector):
    """Ring selector — wraps anyplotlib AnnularWidget."""

    def __init__(self, parent, children, update_function,
                 live_delay: int = 2, multi_selector: bool = False, **kwargs):
        super().__init__(parent, children, update_function,
                         live_delay=live_delay, multi_selector=multi_selector,
                         color=kwargs.get("color", "green"))
        self._widget = None
        self.roi = None
        plot2d = self._get_plot2d()
        if plot2d is not None:
            try:
                self._widget = plot2d.add_annular_widget(color=self.color)
                self.roi = self._widget
                self._event_cb = event_handler_fn(self._on_pointer_up)
                self._widget.add_event_handler(self._event_cb, "pointer_move", "pointer_up")
            except Exception as e:
                logger.debug("AnnularSelector widget init failed: %s", e)

    def _get_plot2d(self):
        plot = self.current_plot
        return getattr(plot, "_plot2d", None) if plot is not None else None

    def _on_pointer_up(self, event):
        self.update_data()

    def _get_selected_indices(self) -> np.ndarray:
        if self._widget is None:
            return np.array([[0, 0]])
        cx = int(round(float(self._widget.cx)))
        cy = int(round(float(self._widget.cy)))
        r_in = int(round(float(self._widget.r_inner)))
        r_out = int(round(float(self._widget.r_outer)))
        return np.array([[cx, cy, r_in, r_out]])

    def add_linked_roi(self, plot) -> None:
        pass


class LineProfileSelector(BaseSelector):
    """Line-profile selector: the profile path is a LINE WIDGET — a 2-vertex
    anyplotlib "polygon" widget, which the renderer draws as a solid segment
    with a native control point at each end (drag a control point to move that
    end). A DASHED marker line perpendicular to it at the midpoint shows the
    integration WIDTH; the bare control point at its tip (an empty-text label
    widget — just the handle dot, no extra chrome) drags to widen/narrow the
    band. The two lines are linked: moving the profile line re-derives the
    perpendicular and carries the width control along; moving the width
    control re-derives the width and snaps back onto the perpendicular.

    (Only anyplotlib's *Python* ``PolygonWidget`` wrapper insists on ≥3
    vertices — the JS renderer draws and hit-tests any count ≥2, so the line
    widget is a bare ``Widget("polygon", …)`` registered the same way
    ``add_polygon_widget`` does.)
    """

    def __init__(self, parent, children, update_function,
                 live_delay: int = 2, multi_selector: bool = False, **kwargs):
        super().__init__(parent, children, update_function,
                         live_delay=live_delay, multi_selector=multi_selector,
                         color=kwargs.get("color", "#ffd166"))
        self._line = None    # the 2-vertex line widget (the solid profile path)
        self._hw = None      # width control point (empty-label widget)
        self._dash_mg = None
        self.roi = None
        self.width = 1.0     # integration width (px, across the line)
        # Programmatic widget syncs fire the widgets' own pointer callbacks
        # (Widget.set → callbacks.fire) — guard against re-entry.
        self._syncing = False

        plot2d = self._get_plot2d()
        if plot2d is None:
            return
        try:
            iw = float(plot2d._state.get("image_width") or 100)
            ih = float(plot2d._state.get("image_height") or 100)
            self.width = max(2.0, 0.1 * min(iw, ih))
            self._line = self._add_line_widget(
                plot2d, [[0.3 * iw, 0.5 * ih], [0.7 * iw, 0.5 * ih]])
            mx, my, nx, ny = self._mid_and_normal()
            self._hw = plot2d.add_label_widget(
                x=mx + nx * self.width / 2.0, y=my + ny * self.width / 2.0,
                text="", fontsize=1, color=self.color)
            self.roi = self._line
            self._cb_line = event_handler_fn(self._on_line_moved)
            self._cb_width = event_handler_fn(self._on_width_moved)
            self._line.add_event_handler(self._cb_line, "pointer_move", "pointer_up")
            self._hw.add_event_handler(self._cb_width, "pointer_move", "pointer_up")
            self._redraw()
        except Exception as e:
            logger.debug("LineProfileSelector widget init failed: %s", e)

    def _add_line_widget(self, plot2d, vertices):
        """Register a 2-vertex polygon widget (the draggable line segment)."""
        from anyplotlib.widgets._base import Widget
        w = Widget("polygon", lambda: None,
                   vertices=[[float(x), float(y)] for x, y in vertices],
                   color=self.color)
        w._push_fn = plot2d._make_widget_push_fn(w)
        plot2d._widgets[w.id] = w
        plot2d._push()
        return w

    def _get_plot2d(self):
        plot = self.current_plot
        return getattr(plot, "_plot2d", None) if plot is not None else None

    # ── Geometry ────────────────────────────────────────────────────────────────

    @property
    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        v = self._line.vertices
        return ((float(v[0][0]), float(v[0][1])),
                (float(v[1][0]), float(v[1][1])))

    def _mid_and_normal(self) -> tuple[float, float, float, float]:
        """Midpoint + unit normal of the profile line (widget pixel coords)."""
        (x0, y0), (x1, y1) = self.endpoints
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy)) or 1.0
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0, -dy / length, dx / length

    # ── Events ──────────────────────────────────────────────────────────────────

    def _on_line_moved(self, event=None):
        # The line moved — carry the width control along (same width, new
        # perpendicular) and redraw the dashed width line.
        if self._syncing:
            return
        self._syncing = True
        try:
            mx, my, nx, ny = self._mid_and_normal()
            self._hw.set(x=mx + nx * self.width / 2.0,
                         y=my + ny * self.width / 2.0)
        except Exception as e:
            logger.debug("repositioning width control point failed: %s", e)
        finally:
            self._syncing = False
        self._redraw()
        self.update_data()

    def _on_width_moved(self, event=None):
        # Width = 2 × the control point's projection onto the perpendicular;
        # snap it back onto the perpendicular axis.
        if self._syncing:
            return
        self._syncing = True
        try:
            mx, my, nx, ny = self._mid_and_normal()
            px, py = float(self._hw.x) - mx, float(self._hw.y) - my
            proj = px * nx + py * ny
            self.width = max(1.0, 2.0 * abs(proj))
            side = 1.0 if proj >= 0 else -1.0
            self._hw.set(x=mx + side * nx * self.width / 2.0,
                         y=my + side * ny * self.width / 2.0)
        except Exception as e:
            logger.debug("updating line-profile width failed: %s", e)
        finally:
            self._syncing = False
        self._redraw()
        self.update_data()

    # ── Drawing ─────────────────────────────────────────────────────────────────

    def _dash_segments(self) -> list:
        """The dashed perpendicular as a run of short segments through the
        midpoint (length = the integration width)."""
        mx, my, nx, ny = self._mid_and_normal()
        half = self.width / 2.0
        dash, gap = 4.0, 3.0
        segs = []
        t = -half
        while t < half:
            t2 = min(t + dash, half)
            segs.append([[mx + nx * t, my + ny * t], [mx + nx * t2, my + ny * t2]])
            t = t2 + gap
        return segs

    def _redraw(self) -> None:
        """Redraw the dashed width line (the solid line IS the widget — the
        renderer draws it and its control points natively)."""
        plot2d = self._get_plot2d()
        if plot2d is None:
            return
        try:
            if self._dash_mg is None:
                self._dash_mg = plot2d.add_lines(
                    self._dash_segments(), name=f"lp_dash_{id(self)}",
                    edgecolors=self.color, linewidths=1.2)
            else:
                self._dash_mg.set(segments=self._dash_segments())
        except Exception as e:
            logger.debug("line-profile redraw failed: %s", e)

    # ── BaseSelector contract ───────────────────────────────────────────────────

    def _get_selected_indices(self) -> np.ndarray:
        if self._line is None:
            return np.array([[0, 0]])
        (x0, y0), (x1, y1) = self.endpoints
        # Geometry row (not an index grid) — ints so change-detection fires.
        return np.array([[int(round(x0)), int(round(y0)),
                          int(round(x1)), int(round(y1)),
                          int(round(self.width))]])

    def add_linked_roi(self, plot: "Plot") -> None:
        pass

    def translate_pixels(self, shift_x: int, shift_y: int) -> None:
        if self._line is None:
            return
        try:
            (x0, y0), (x1, y1) = self.endpoints
            self._line.set(vertices=[[x0 + shift_x, y0 + shift_y],
                                     [x1 + shift_x, y1 + shift_y]])
        except Exception as e:
            logger.debug("translating line-profile selector failed: %s", e)

    def hide(self) -> None:
        for wdg in (self._line, self._hw):
            if wdg is not None:
                try:
                    wdg.hide()
                except Exception as e:
                    logger.debug("hiding line-profile widget failed: %s", e)
        if self._dash_mg is not None:
            try:
                self._dash_mg.set(segments=[])
            except Exception as e:
                logger.debug("hiding line-profile dash failed: %s", e)

    def show(self) -> None:
        for wdg in (self._line, self._hw):
            if wdg is not None:
                try:
                    wdg.show()
                except Exception as e:
                    logger.debug("showing line-profile widget failed: %s", e)
        self._redraw()

    def close(self) -> None:
        if self._dash_mg is not None:
            try:
                self._dash_mg.remove()
            except Exception as e:
                logger.debug("removing line-profile dash failed: %s", e)
        self._dash_mg = None
        super().close()   # hides the widgets (our hide override) + panel push


# Back-compat alias (the old stub's export name).
LineSelector = LineProfileSelector


class IntegratingSSelector2D(IntegratingSelectorMixin):
    """Composite selector switching between crosshair (point) and rectangle (area)."""

    def __init__(
        self,
        parent: "PlotWindow",
        children: Union["Plot", List["Plot"]],
        update_function: Union[callable, List[callable]],
        live_delay: int = 3,
        multi_selector: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._rect_selector = RectangleSelector(
            parent, children, update_function,
            live_delay=live_delay, multi_selector=multi_selector, **kwargs,
        )
        self._crosshair_selector = CrosshairSelector(
            parent, children, update_function,
            live_delay=live_delay, multi_selector=multi_selector, **kwargs,
        )
        self.parent = parent

        if not isinstance(children, list):
            self.children = {children: update_function}
            self.active_children = [children]
            if hasattr(children, "plot_window") and children.plot_window is not None:
                children.plot_window.parent_selector = self
        else:
            self.children = {}
            self.active_children = []
            for child, fn in zip(children, update_function):
                self.children[child] = fn
                self.active_children.append(child)
                if hasattr(child, "parent_selector"):
                    child.parent_selector = self

        self.selector = self._crosshair_selector
        self._crosshair_selector.is_integrating = False
        self._rect_selector.is_integrating = True
        if self._rect_selector._widget is not None:
            try:
                self._rect_selector._widget.hide()
            except Exception as e:
                logger.debug("hiding rectangle selector on init failed: %s", e)
        # Reflect the initial hidden rectangle in the panel overlay state.
        self._force_overlay_repaint()

    def __getattr__(self, name):
        """Delegate any BaseSelector method/attribute not defined on the
        composite to the currently-active sub-selector (update_data,
        get_selected_indices, upstream_selectors, multi_selector, …)."""
        if name in ("selector", "_rect_selector", "_crosshair_selector"):
            raise AttributeError(name)
        selector = self.__dict__.get("selector")
        if selector is None:
            raise AttributeError(name)
        return getattr(selector, name)

    @property
    def roi(self):
        return self.selector.roi

    @property
    def current_indices(self):
        return self.selector.current_indices

    def _get_selected_indices(self) -> np.ndarray:
        return self.selector._get_selected_indices()

    def delayed_update_data(self, force: bool = False, update_contrast: bool = False) -> None:
        self.selector.delayed_update_data(force=force, update_contrast=update_contrast)

    def _force_overlay_repaint(self) -> None:
        """Push the full panel state so the new widget visibility is reflected in
        overlay_widgets (replayable + reliably repainted), not only in the
        single-shot event_json targeted update which the second hide/show
        overwrites."""
        try:
            plot2d = self._crosshair_selector._get_plot2d()
            if plot2d is not None:
                plot2d._push()
        except Exception as e:
            logger.debug("pushing 2-D panel overlay state failed: %s", e)

    def set_integrating(self, enabled: bool) -> None:
        if enabled:
            if self._crosshair_selector._widget is not None:
                try:
                    self._crosshair_selector._widget.hide()
                except Exception as e:
                    logger.debug("hiding crosshair selector failed: %s", e)
            if self._rect_selector._widget is not None:
                try:
                    self._rect_selector._widget.show()
                except Exception as e:
                    logger.debug("showing rectangle selector failed: %s", e)
            self.selector = self._rect_selector
        else:
            if self._rect_selector._widget is not None:
                try:
                    self._rect_selector._widget.hide()
                except Exception as e:
                    logger.debug("hiding rectangle selector failed: %s", e)
            if self._crosshair_selector._widget is not None:
                try:
                    self._crosshair_selector._widget.show()
                except Exception as e:
                    logger.debug("showing crosshair selector failed: %s", e)
            self.selector = self._crosshair_selector
        self.is_integrating = enabled
        self._force_overlay_repaint()
        self.selector.delayed_update_data(force=True)

    def hide(self) -> None:
        self._crosshair_selector.hide()
        self._rect_selector.hide()

    def show(self) -> None:
        self.selector.show()

    def close(self) -> None:
        self._crosshair_selector.close()
        self._rect_selector.close()

    def add_linked_roi(self, plot: "Plot") -> None:
        pass


# Keep old name as alias
IntegratingSSelector2D = IntegratingSSelector2D

"""2D Selectors using anyplotlib interactive widgets."""
from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING, Union, List

from spyde.drawing.selectors.base_selector import BaseSelector, IntegratingSelectorMixin

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
                self._widget.add_event_handler(self._on_pointer_up, "pointer_move", "pointer_up")
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
        cx = int(round(float(self._widget.cx)))
        cy = int(round(float(self._widget.cy)))
        return np.array([[cx, cy]])

    def add_linked_roi(self, plot: "Plot") -> None:
        pass

    def translate_pixels(self, shift_x: int, shift_y: int) -> None:
        if self._widget is not None:
            try:
                self._widget.cx = float(self._widget.cx) + shift_x
                self._widget.cy = float(self._widget.cy) + shift_y
            except Exception:
                pass


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
                self._widget.add_event_handler(self._on_pointer_up, "pointer_move", "pointer_up")
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

        x0, x1 = int(round(x)), int(round(x + w))
        y0, y1 = int(round(y)), int(round(y + h))

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
            except Exception:
                pass


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
                self._widget.add_event_handler(self._on_pointer_up, "pointer_move", "pointer_up")
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
                self._widget.add_event_handler(self._on_pointer_up, "pointer_move", "pointer_up")
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


class LineSelector(BaseSelector):
    """Line selector stub — no pyqtgraph LineROI equivalent in anyplotlib yet."""

    def __init__(self, parent, children, update_function, **kwargs):
        super().__init__(parent, children, update_function)

    def _get_selected_indices(self) -> np.ndarray:
        return np.array([[0, 0]])

    def add_linked_roi(self, plot: "Plot") -> None:
        pass


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
            live_delay=live_delay, multi_selector=multi_selector,
        )
        self._crosshair_selector = CrosshairSelector(
            parent, children, update_function,
            live_delay=live_delay, multi_selector=multi_selector,
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
            except Exception:
                pass
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
        except Exception:
            pass

    def set_integrating(self, enabled: bool) -> None:
        if enabled:
            if self._crosshair_selector._widget is not None:
                try:
                    self._crosshair_selector._widget.hide()
                except Exception:
                    pass
            if self._rect_selector._widget is not None:
                try:
                    self._rect_selector._widget.show()
                except Exception:
                    pass
            self.selector = self._rect_selector
        else:
            if self._rect_selector._widget is not None:
                try:
                    self._rect_selector._widget.hide()
                except Exception:
                    pass
            if self._crosshair_selector._widget is not None:
                try:
                    self._crosshair_selector._widget.show()
                except Exception:
                    pass
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

"""2D Selectors for a Plot."""

from spyde.drawing.selectors.base_selector import BaseSelector, IntegratingSelectorMixin


import time

from pyqtgraph import LinearRegionItem, RectROI, LineROI, ROI

from PySide6 import QtCore, QtWidgets, QtGui
import numpy as np

import pyqtgraph as pg
import logging

from typing import TYPE_CHECKING, Union, List, Type, Iterable

from spyde.drawing.selectors.utils import create_linked_rect_roi
from spyde.external.pyqtgraph.crosshair_roi import CrosshairROI

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow

Logger = logging.getLogger(__name__)

class CrosshairSelector(BaseSelector):
    def __init__(
        self,
        parent: "PlotWindow",
        children: Union["Plot", List["Plot"]],
        update_function: Union[callable, List[callable]],
        live_delay: int = 2,
        multi_selector: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            parent,
            children,
            update_function,
            live_delay=live_delay,
            multi_selector=multi_selector,
            *args,
            **kwargs,
        )
        # Place the crosshair at the image center. pixel_size is in screen pixels;
        # _update_for_zoom converts it to data units on first view range change.
        CROSSHAIR_SCREEN_PX = 10
        image_item = parent.current_plot_item.image_item
        try:
            bounding = image_item.mapRectToScene(image_item.boundingRect())
            center = bounding.center()
            scene_px_size = max(bounding.width(), bounding.height()) / max(
                image_item.width() or 1, image_item.height() or 1
            )
            init_scene_half = scene_px_size * CROSSHAIR_SCREEN_PX / 2
        except Exception:
            # Image not yet rendered — place at origin, will be corrected on first zoom
            center = QtCore.QPointF(0.0, 0.0)
            init_scene_half = 0.01
        pos = QtCore.QPointF(center.x() - init_scene_half, center.y() - init_scene_half)
        self.roi = CrosshairROI(
            pos=pos,
            pixel_size=CROSSHAIR_SCREEN_PX,
            view=parent.current_plot_item.getViewBox(),
            pen=self.roi_pen,
            handlePen=self.handlePen,
            hoverPen=self.hoverPen,
            handleHoverPen=self.handleHoverPen,
            *args,
            **kwargs,
        )

        self._last_size_sig = (0, 0)
        self.roi.sigRegionChangeFinished.connect(self._on_region_change_finished)

        for plot in parent.plots:
            # The selector isn't actually added to any plot??
            self.add_linked_roi(plot)
        self.roi.sigRegionChanged.connect(self.update_data)

    def update_data(self, ev=None):
        """Ignore region changes that are purely zoom-driven resizes."""
        if getattr(self.roi, '_zoom_resize', False):
            return
        super().update_data(ev)

    def _get_selected_indices(self):
        """
        Get the currently selected indices from the crosshair selector.
        """
        inverted_transform, _ = self.parent.current_plot_item.image_item.transform().inverted()
        pos = self.roi.pos()
        size = self.roi.size()
        center = pos + QtCore.QPointF(size[0] / 2, size[1] / 2)
        center_pixel = inverted_transform.map(center)

        x = int(np.round(center_pixel.x()))
        y = int(np.round(center_pixel.y()))
        indices = np.array([[x, y]])
        return indices

    def add_linked_roi(self, plot: "Plot"):

        if self.roi is not None:
            new_selector = create_linked_rect_roi(self.roi)
            plot.addItem(new_selector)
            self.linked_selectors.append(new_selector)


class RectangleSelector(BaseSelector):
    def __init__(
            self,
            parent: Union["PlotWindow", "Plot"],
            children: Union["Plot", List["Plot"]],
            update_function: Union[callable, List[callable]],
            live_delay: int = 2,
            multi_selector: bool = False,
            *args,
            **kwargs,
    ):
        from spyde.drawing.plots.plot import Plot
        from spyde.drawing.plots.plot_window import PlotWindow
        super().__init__(
            parent,
            children,
            update_function,
            live_delay=live_delay,
            multi_selector=multi_selector,
            *args,
            **kwargs,
        )
        print("Creating Rectangle Selector")
        print(args, kwargs)

        # auto position and size — centered on the image FOV, 10% of image width
        # RectROI pos is its lower-left corner, so subtract half the size to center it
        image_item = self.current_plot.image_item
        transform = image_item.transform()
        img_w = image_item.width()
        img_h = image_item.height()
        center = transform.map(QtCore.QPointF(img_w / 2, img_h / 2))
        width = max(1, img_w // 10)
        pos = center - QtCore.QPointF(width / 2, width / 2)

        self.roi = RectROI(
            pos=pos,
            size=(width, width),
            pen=self.roi_pen,
            handlePen=self.handlePen,
            hoverPen=self.hoverPen,
            handleHoverPen=self.handleHoverPen,
            *args,
            **kwargs,
        )
        self._last_size_sig = (0, 0)
        self.roi.sigRegionChangeFinished.connect(self._on_region_change_finished)

        if isinstance(parent, PlotWindow):
            for plot in parent.plots:
                # The selector isn't actually added to any plot??
                self.add_linked_roi(plot)
        else:
            parent.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_data)

    def _get_selected_indices(self):
        """
        Get the currently selected indices from the selector.
        """
        if self.multi_selector:
            [
                parent_selector.get_selected_indices()
                for parent_selector in self.upstream_selectors()
            ]

        lower_left = self.roi.pos()
        size = self.roi.size()

        # pyqtgraph only knows one coordinate system.  We need to map the scene
        # to pixels.
        inverted_transform, is_inversion = (
            self.current_plot.image_item.transform().inverted()
        )

        lower_left_pixel = inverted_transform.map(lower_left)

        size_pixels = inverted_transform.map(size) - inverted_transform.map(
            QtCore.QPointF(0, 0)
        )

        # ignore rotation for now...
        rotation = self.roi.angle()

        y_indices = np.arange(0, np.round(size_pixels.y()), dtype=int)
        x_indices = np.arange(0, np.round(size_pixels.x()), dtype=int)

        indices = np.reshape(np.array(np.meshgrid(x_indices, y_indices)).T, (-1, 2))
        indices[:, 0] += np.round(lower_left_pixel.x()).astype(int)
        indices[:, 1] += np.round(lower_left_pixel.y()).astype(int)
        indices = indices.astype(int)
        return indices

    def add_linked_roi(self, plot: "Plot"):

        if self.roi is not None:
            new_selector = create_linked_rect_roi(self.roi)
            plot.addItem(new_selector)
            self.linked_selectors.append(new_selector)

    def translate_pixels(self, shift_x: int, shift_y: int):
        """
        Translate the selector by the given amount in pixels.
        """
        if self.roi is not None:
            shift = QtCore.QPointF(shift_x, shift_y)
            transform = self.parent.current_plot_item.image_item.transform()
            shift = transform.map(shift)
            self.roi.translate(shift.x(), shift.y())

    def move_selector(self, key: QtCore.Qt.Key):
        """
        Move the selector based on the key pressed.
        """
        self.timer = time.time()
        if key == QtCore.Qt.Key.Key_Left:
            self.translate_pixels(-1, 0)
        elif key == QtCore.Qt.Key.Key_Right:
            self.translate_pixels(1, 0)
        elif key == QtCore.Qt.Key.Key_Up:
            self.translate_pixels(0, 1)
        elif key == QtCore.Qt.Key.Key_Down:
            self.translate_pixels(0, -1)



class IntegratingSelector2D(IntegratingSelectorMixin):
    def __init__(
            self,
            parent: "PlotWindow",
            children: Union["Plot", List["Plot"]],
            update_function: Union[callable, List[callable]],
            live_delay: int = 3,
            multi_selector: bool = False,
            *args,
            **kwargs,
    ):
        # initialize the rect and the crosshair
        # Store the rectangle selector created by parents
        super().__init__()
        self._rect_selector = RectangleSelector(parent,
                                                children,
                                                update_function,
                                                live_delay=live_delay,
                                                multi_selector=multi_selector,
                                                *args,
                                                **kwargs
        )
        self._crosshair_selector = CrosshairSelector(parent,
                                                     children,
                                                     update_function,
                                                     live_delay=live_delay,
                                                     multi_selector=multi_selector,
                                                    *args,
                                                     **kwargs
        )

        self.parent = parent  # type: PlotWindow
        if not isinstance(children, list):
            self.children = {children: update_function}  # type: dict[Plot, callable]
            self.active_children = [
                children,
            ]  # type: list[Plot]
            children.plot_window.parent_selector = self
            # children.parent_selector = self

        else:
            self.children = {}  # type: dict[Plot, callable]
            for child, function in zip(children, update_function):
                self.children[child] = function
                child.parent_selector = self

        self._hide_rect_selector()
        self.selector = self._crosshair_selector # type: BaseSelector
        # connect the is_integrating property to the selector
        self._crosshair_selector.is_integrating = False
        self._rect_selector.is_integrating = True

    @property
    def roi(self) -> ROI:
        """Get the current ROI."""
        return self.selector.roi

    def _hide_crosshair_selector(self):
        """Hide the crosshair selector."""
        print(f"Hiding {self._crosshair_selector.linked_selectors}")
        for linked in self._crosshair_selector.linked_selectors:
                linked.hide()
    def _show_crosshair_selector(self):
        """Show the crosshair selector."""
        for linked in self._crosshair_selector.linked_selectors:
                linked.show()
        self.selector = self._crosshair_selector
    def _hide_rect_selector(self):
        """Hide the rectangle selector."""
        for linked in self._rect_selector.linked_selectors:
                linked.hide()

    def _show_rect_selector(self):
        """Show the rectangle selector."""
        for linked in self._rect_selector.linked_selectors:
                linked.show()
        self.selector = self._crosshair_selector

    def add_linked_selector(self, plot: "Plot"):
        """Add both selectors to the new plot."""
        self._rect_selector.add_linked_roi(plot)
        self._crosshair_selector.add_linked_roi(plot)
        if not self.is_integrating:
            self._show_crosshair_selector()
            self._hide_rect_selector()
        else:
            self._show_rect_selector()
            self._hide_crosshair_selector()

    def on_integrate_toggled(self, checked):
        """Switch between crosshair and rectangle."""
        if not checked:
            self._hide_rect_selector()
            self._show_crosshair_selector()
        else:
            self._hide_crosshair_selector()
            self._show_rect_selector()
        super().on_integrate_toggled(checked)

    def _get_selected_indices(self):
        """Handle both crosshair and rectangle selection."""
        if not self.is_integrating:
            # Use crosshair implementation
            return self._crosshair_selector._get_selected_indices()
        else:
            # Use parent implementation for rectangle
            return self._rect_selector._get_selected_indices()

    def delayed_update_data(self, force: bool = False):
        """Update data with a delay."""
        self.selector.delayed_update_data(force=force)

    def hide(self):
        self._crosshair_selector.hide()
        self._rect_selector.hide()

    def close(self):
        self._crosshair_selector.close()
        self._rect_selector.close()

    def add_linked_roi(self, plot: "Plot"):
        """Add both selectors to the new plot."""
        self._rect_selector.add_linked_roi(plot)
        self._crosshair_selector.add_linked_roi(plot)

class LineSelector(BaseSelector):
    """
    A selector which uses a LineROI to select a region along one axis.
    """

    def __init__(
        self,
        parent: "PlotWindow",
        children: Union["Plot", List["Plot"]],
        update_function: Union[callable, List[callable]],
        *args,
        **kwargs,
    ):
        super().__init__(parent, children, update_function, *args, **kwargs)
        self.selector = LineROI(
            pen=self.roi_pen,
            handlePen=self.handlePen,
            hoverPen=self.hoverPen,
            handleHoverPen=self.handleHoverPen,
            *args,
            **kwargs,
        )
        for plot in parent.plots:
            plot.addItem(self.selector)
        self.selector.sigRegionChanged.connect(self.update_data)

    def _get_selected_indices(self):
        """
        Get the currently selected indices from the selector.
        """
        pos = self.selector.getArraySlice(
            np.arange(self.parent.current_plot_item.data.shape[0]),
            self.parent.data.shape,
        )[0]
        indices = np.array(
            [[np.round(pos[0][i]).astype(int)] for i in range(len(pos[0]))]
        )
        return indices


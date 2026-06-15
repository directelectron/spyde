import time

from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from pyqtgraph import GraphicsItem, GraphicsLayoutWidget

import numpy as np
from typing import TYPE_CHECKING, Union, List, Dict, Tuple, Optional

from spyde.drawing.plots.multiplot_manager import MultiplotManager
from spyde.drawing.selectors.base_selector import IntegratingSelectorMixin
from spyde.qt.subwindow import FramelessSubWindow

if TYPE_CHECKING:
    from spyde.signal_tree import BaseSignalTree
    from spyde.__main__ import MainWindow
    from spyde.drawing.selectors import BaseSelector
    from spyde.drawing.plots.plot_states import PlotState
    from spyde.drawing.plots.plot import Plot

import logging

logger = logging.getLogger(__name__)


class PlotWindow(FramelessSubWindow):
    """
     A QMdi sub-window that contains either a single Plot or potentially multiple plots in some layout.

     The ``PlotWindow`` has a graphics layout widget which has an ``items`` dictionary which maps a GraphicsItem
     to a (row, col) position in the layout.  In the instance of a single Plot, the items dictionary will
     have a single entry mapping the Plot's plot_item to (0, 0).

     In addition to the items dictionary, the ``PlotWindow`` also has a plot_state dictionary which maps
     each GraphicsItem to its corresponding ``PlotState``.

    When multiple plots are present, the items dictionary will have multiple entries mapping each Plot's
    plot_item to its (row, col) position in the layout.

    In the case that the PlotWindow is a member of a NavigationPlotManager,  a navigation selector will be
    added to each plot in the window.  Each `Plot` (or GraphicsItem) can be updated independently based on an
    associated update function. This allows for more complex plots (real/img FFT side-by-side,  update all
    virtual images at once)

    Each PlotWindow will also have toolbars, but these are all associated with individual Plots within the window
    (more specifically, with the PlotStates of each Plot).  Clicking on some plot will activate its toolbars and hide
    the toolbars of other plots within the same window. This can be slightly confusing, but it allows for the
    flexibility of multiple plots within the same window without needing to manage multiple sets of toolbars. The active
    `Plot` within the `PlotWindow` is shown by a small highlight around the plot area.
    """

    def __init__(
        self,
        is_navigator: bool = False,  # if navigator then it will share the navigation selectors
        plot_manager: Union["MultiplotManager", None] = None,
        signal_tree: Union["BaseSignalTree", None] = None,
        main_window: Union["MainWindow", None] = None,
        parent_selector: Union["BaseSelector", None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.signal_tree = signal_tree
        self.is_navigator = is_navigator  # type: bool
        # the plot manager for managing multiple plot windows.  Different from Multiplexed Plots as
        # this is for managing multiple Plot windows.
        self.multiplot_manager = plot_manager  # type: MultiplotManager | None

        # Instance state: track the currently active Plot (or None)
        self._current_plot_item = None  # type: Plot | None
        # The primary plot item: used for linking axes when new plots are added.
        self._primary_plot_item = None  # type: Plot | None

        # Previous layout state: used when restoring saved multiplexed layouts
        self.previous_subplots_pos = (
            dict()
        )  # type: Dict[pg.PlotItem, Tuple[int, int]] | Dict
        self.previous_graphics_layout_widget = (
            None
        )  # type: pg.GraphicsLayoutWidget | None
        self.parent_selector = parent_selector  # type: BaseSelector | None | IntegratingSelectorMixin

        # UI: container widget + layout that will host the GraphicsLayoutWidget.
        # Use a QVBoxLayout with zero margins/spacing so the plot fills the subwindow.
        self.container = QtWidgets.QWidget()
        self.container.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        # Thin themed frame around the plot (was an opaque light-grey block that
        # clashed with the dark theme). The 1px container margins below show it
        # as a subtle border.
        from spyde.qt.style import BORDER_FAINT
        self.container.setStyleSheet(f"background-color: {BORDER_FAINT};")
        container_layout = QtWidgets.QVBoxLayout(self.container)
        container_layout.setContentsMargins(1, 0, 1, 1)
        container_layout.setSpacing(0)

        # The central plot widget: a pyqtgraph GraphicsLayoutWidget parented to our container.
        self.plot_widget = pg.GraphicsLayoutWidget(self.container)
        container_layout.addWidget(self.plot_widget)
        self.setWidget(self.container)

        self.last_used_selector = None # Type: BaseSelector | None

        # Explicitly store the main window reference provided at construction.
        self.main_window = main_window  # type: "MainWindow"
        self.timer = None
        self._compute_indicator = None  # type: ComputeStatusIndicator | None
        self._commit_fn = None  # type: callable | None
        self._commit_connection = None  # type: QtCore.QMetaObject.Connection | None
        self.owner_plot_window = None  # type: "PlotWindow | None"
        self.controlling_action = None  # type: QtGui.QAction | None
        # Action-owned windows (IPF, VI previews, …) set this to intercept the
        # title-bar X: QMdiSubWindow.closeEvent would close the inner container
        # widget, leaving a bare title bar if the action later re-shows the
        # window. The callable hides/untoggles instead of destroying.
        self.on_close_request = None  # type: callable | None
        # Optional gate consulted by MDIManager._update_3state_visibility:
        # when set, the window is only auto-shown if it returns True.
        self.visibility_gate = None  # type: callable | None

        # Debounce toolbar repositioning: coalesce rapid moveEvent/resizeEvent
        # calls (every pixel of drag) into a single reposition at the end.
        self._reposition_timer = QtCore.QTimer(self)
        self._reposition_timer.setInterval(16)  # ~1 frame at 60 Hz
        self._reposition_timer.setSingleShot(True)
        self._reposition_timer.timeout.connect(self.reposition_toolbars)

    def set_compute_indicator(self, indicator) -> None:
        """Insert the ComputeStatusIndicator into the title bar left zone."""
        self._compute_indicator = indicator
        layout = self.title_bar_layout
        old = self._status_placeholder
        idx = layout.indexOf(old)
        layout.removeWidget(old)
        old.deleteLater()
        self._status_placeholder = indicator
        indicator.setParent(self.title_bar)
        layout.insertWidget(idx, indicator)
        indicator.show()

    def set_stop_fn(self, fn: callable) -> None:
        """Wire a stop/cancel function and show the title-bar Stop button."""
        btn = self.title_bar.stop_button
        try:
            btn.clicked.disconnect()
        except Exception:
            pass
        btn.clicked.connect(fn)
        btn.show()

    def set_compute_fn(self, fn: callable) -> None:
        """Wire a compute function and show the title-bar Compute button.

        Sits beside Commit so actions can offer compute-on-demand from the
        plot itself rather than from their caret popout.
        """
        btn = self.title_bar.compute_button
        try:
            btn.clicked.disconnect()
        except Exception:
            pass
        btn.clicked.connect(fn)
        btn.show()

    @QtCore.Slot()
    def hide_stop_button(self) -> None:
        self.title_bar.stop_button.hide()

    def set_commit_fn(self, fn: callable, label: str = "Commit") -> None:
        """Wire a commit function and show the title-bar Commit button.

        The button starts disabled — call set_commit_enabled(True) once
        the first data is ready.
        """
        btn = self.title_bar.commit_button
        btn.setText(label)
        # Disconnect the previous function if one was connected
        if self._commit_connection is not None:
            btn.clicked.disconnect(self._commit_connection)
        # Store the function and connect it
        self._commit_fn = fn
        self._commit_connection = btn.clicked.connect(self._commit_fn)
        btn.setEnabled(False)
        btn.show()

    @QtCore.Slot(bool)
    def set_commit_enabled(self, enabled: bool) -> None:
        """Enable or disable the title-bar Commit button.

        Decorated as a Slot(bool) so it can be called safely from dask
        callback threads via QMetaObject.invokeMethod(..., QueuedConnection).
        """
        btn = self.title_bar.commit_button
        btn.setEnabled(enabled)

    def hideEvent(self, ev: QtGui.QHideEvent) -> None:
        """Called when the widget is hidden."""
        super().hideEvent(ev)
        if self.current_plot_state is not None:
            self.current_plot_state.hide_toolbars()

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        """Called when the widget is shown."""
        super().showEvent(ev)
        if self.current_plot_state is not None:
            self.current_plot_state.show_toolbars()

    @property
    def current_plot_item(self) -> Union["Plot", None]:
        """Get or set the currently active Plot in this PlotWindow."""
        if self._current_plot_item is None and len(self.plots) > 0:
            self._current_plot_item = self.plots[0]
        return self._current_plot_item

    @current_plot_item.setter
    def current_plot_item(self, plot: "Plot"):
        """Set the currently active Plot in this PlotWindow."""
        if plot in self.plots:
            self._current_plot_item = plot
            # Notify the main window that this subwindow is active
            self.main_window.on_subwindow_activated(self)
        else:
            raise ValueError("Plot is not in this PlotWindow.")

    @property
    def current_plot_state(self) -> Union["PlotState", None]:
        """Get the PlotState of the currently active Plot in this PlotWindow."""
        if self.current_plot_item is not None:
            return self.current_plot_item.plot_state
        return None

    @property
    def plots(self) -> List["Plot"]:
        """Get the list of Plot objects in this PlotWindow."""
        from spyde.drawing.plots.plot import Plot
        plots = []
        for item in self.plot_widget.ci.items.keys():
            if isinstance(item, Plot):
                plots.append(item)
        return plots

    @property
    def dimensions(self) -> Optional[int]:
        """Get the dimensionality of the current plot state."""
        if self.is_navigator and self.current_plot_state is not None:
            return self.current_plot_state.dimensions
        else:
            return None

    @property
    def mdi_area(self):
        return self.main_window.mdi_area

    @property
    def navigation_selectors(self) -> List["BaseSelector"]:
        """Get the list of navigation selectors associated with this PlotWindow."""
        if self.multiplot_manager is not None:
            return self.multiplot_manager.navigation_selectors.get(self, [])
        return []

    def add_new_plot(
        self,
        row: int = 0,
        col: int = 0,
    ) -> "Plot":
        """Creates and returns a new (empty) Plot."""
        from spyde.drawing.plots.plot import Plot

        plot = Plot(
            signal_tree=self.signal_tree,
            is_navigator=self.is_navigator,
            plot_window=self,
            multiplot_manager=self.multiplot_manager,
        )
        self.add_item(plot, row, col)
        if self.current_plot_item is None:
            self.current_plot_item = plot
        if self._primary_plot_item is None:
            self._primary_plot_item = plot



        return plot

    def _arrange_graphics_layout_preview(
        self,
        graphics_layout_dict: dict,
        graphics_layout: GraphicsLayoutWidget,
        drop_pos: QtCore.QPointF,
    ):
        """
        Calculate where to place a new plot without modifying the layout.
        Returns (row, col) tuple.

        Parameters
        ----------
        graphics_layout_dict : dict
            Current layout as a dictionary mapping plots to (col, row) positions.
        graphics_layout : GraphicsLayoutWidget
            The graphics layout widget.
        drop_pos : QtCore.QPointF
            The position where the navigator is being dropped.
        """
        # Get existing plot positions (excluding placeholder)
        col_inds = []
        row_inds = []
        for plot, position in graphics_layout_dict.items():
            if isinstance(position, list):
                position = position[0]
            col_inds.append(position[1])
            row_inds.append(position[0])
        max_col = max(col_inds) + 1 if col_inds else 1
        max_row = max(row_inds) + 1 if row_inds else 1
        print("MaxCol:", max_col, " MaxRow:", max_row)
        # Get drop position
        x_pos = drop_pos.x() if hasattr(drop_pos, "x") else drop_pos[0]
        y_pos = drop_pos.y() if hasattr(drop_pos, "y") else drop_pos[1]

        layout_width = graphics_layout.width()
        layout_height = graphics_layout.height()

        cell_width = layout_width / max_col
        cell_height = layout_height / max_row

        drop_col = min(int(x_pos / cell_width), max_col)
        drop_row = min(int(y_pos / cell_height), max_row)

        cell_x = x_pos - (drop_col * cell_width)
        cell_y = y_pos - (drop_row * cell_height)

        norm_x = cell_x / cell_width
        norm_y = cell_y / cell_height

        print(
            "NormX:",
            norm_x,
            " NormY:",
            norm_y,
            " CellX",
            drop_col,
            " CellY:",
            drop_row,
        )
        # split cell into zones in an x
        angle = np.arctan2(norm_y - 0.5, norm_x - 0.5)  # -pi to pi

        if angle >= -3 * np.pi / 4 and angle < -np.pi / 4:
            zone = "top"
        elif angle >= -np.pi / 4 and angle < np.pi / 4:
            zone = "right"
        elif angle >= np.pi / 4 and angle < 3 * np.pi / 4:
            zone = "bottom"
        else:
            zone = "left"

        # Calculate target position
        new_pos = None
        if zone == "top" or zone == "left":
            new_pos = (drop_col, drop_row)
        elif zone == "bottom":
            new_pos = (drop_col, drop_row + 1)
        else:  # zone == 'right':
            new_pos = (drop_col + 1, drop_row)
        print("zone:", zone, " new pos:", new_pos)
        return new_pos, zone


    def _build_new_layout(
            self,
            drop_pos: QtCore.QPointF,
            plot_to_add: "Plot",
    ):
        """Build a new layout with the new plot added at the drop position."""
        from spyde.drawing.plots.plot import Plot
        new_pos, zone = self._arrange_graphics_layout_preview(
            self.previous_subplots_pos, self.previous_graphics_layout_widget, drop_pos
        )

        # build a new layout based on previous layout and the new position
        # new_layout_dictionary = active_plot.previous_subplots_pos.copy()
        new_layout_dictionary = {}
        if new_pos is not None:
            # reset to the previous layout
            col, row = new_pos

            # cycle though all the items and shift them if needed

            # old: [0,0]  pos[0], pos[1]
            # new: [1,0] col, row

            for plot, position in self.previous_subplots_pos.items():
                # Columns should add to the right.
                prev_pos = position[0]  # this is a list of (col, row) positions
                # rows should shift that column down
                if (
                        col == prev_pos[1]
                        and row <= prev_pos[0]
                        and zone in ("top", "bottom")
                ):
                    new_layout_dictionary[plot] = (prev_pos[0] + 1, prev_pos[1])
                # columns to the right should shift right
                elif col <= prev_pos[1] and zone in ("left", "right"):
                    new_layout_dictionary[plot] = (prev_pos[0], prev_pos[1] + 1)
                # columns to the right/bottom should stay the same
                else:
                    new_layout_dictionary[plot] = (prev_pos[0], prev_pos[1])
            # add the placeholder at the new position
            new_layout_dictionary[plot_to_add] = (new_pos[1], new_pos[0])

            # Find the maximum row to determine bottom plots
            max_row = max(pos[0] for pos in new_layout_dictionary.values())

            # Hide x-axis for all plots except those in the bottom row for each column
            if self.dimensions == 1:
                for plot, pos in new_layout_dictionary.items():
                    if isinstance(plot, Plot):
                        if pos[0] == max_row:
                            plot.showAxis('bottom')
                        else:
                            plot.hideAxis('bottom')

            self.set_graphics_layout_widget(new_layout_dictionary)
            self.plot_widget.ci.layout.setContentsMargins(0, 0, 0, 0)
            self.plot_widget.ci.layout.setSpacing(2)  # Small positive spacing
            for plot in self.plots:
                plot.getViewBox().setDefaultPadding(0.0)

            self.plot_widget.ci.layout.setSpacing(-2) # axes should be

    def insert_new_plot(
        self,
        drop_pos: QtCore.QPointF,
    ) -> "Plot":
        """Inserts and returns a new (empty) Plot at the drop position.

        If this Plot window has a MultiplotManager and it is a navigation plot then the plots have to
        be all the same size... That is handled when the plot states are added though.....

        """
        from spyde.drawing.plots.plot import Plot

        new_plot = Plot(
            signal_tree=self.signal_tree,
            multiplot_manager=self.multiplot_manager,
            is_navigator=self.is_navigator,
            plot_window=self,
        )
        self._build_new_layout(drop_pos, new_plot)
        self.multiplot_manager.plots[self].append(new_plot)
        # add all active selectors
        print("Adding linked selectors to new plot:", new_plot)
        print("Current selectors:", self.navigation_selectors)
        for selector in self.navigation_selectors:
            selector.add_linked_roi(plot=new_plot)
            print("Newplot items:", new_plot.items)
            print(new_plot.items[0].isVisible())
        # link the plot
        if self.multiplot_manager is not None and self.is_navigator and self.multiplot_manager.nav_dim == 1:
            new_plot.setXLink(self._primary_plot_item)
            primary_vb = self._primary_plot_item.getViewBox()
            new_vb = new_plot.getViewBox()

            def sync_y_relative(view_box):
                """Synchronize Y range relative to maximum values."""
                primary_range = primary_vb.viewRange()[1]
                primary_data_max = getattr(self._primary_plot_item, 'data_max_y', 1.0)
                primary_data_min = getattr(self._primary_plot_item, 'data_min_y', 0.0)
                new_data_max = getattr(new_plot, 'data_max_y', 1.0)
                new_data_min = getattr(new_plot, 'data_min_y', 1.0)
                print("Syncing Y range:", primary_range,
                      " Primary max:", [primary_data_min, primary_data_max],
                      " New max:", [new_data_min, new_data_max])

                if primary_data_max != 0 and new_data_max != 0:
                    scale_factor = (new_data_max-new_data_min) / (primary_data_max-primary_data_min)
                    shift = new_data_min - primary_data_min * scale_factor
                    new_range = [primary_range[0] * scale_factor + shift,
                                 primary_range[1] * scale_factor + shift]
                    print("Setting new Y range:", new_range)
                    view_box.setYRange(*new_range, padding=0)

            primary_vb.sigRangeChanged.connect(lambda: sync_y_relative(new_vb))
        elif self.multiplot_manager is not None and self.is_navigator and self.multiplot_manager.nav_dim == 2:
            new_plot.setXLink(self._primary_plot_item)
            new_plot.setYLink(self._primary_plot_item)

        return new_plot

    def add_item(self, item: GraphicsItem, row: int = 0, col: int = 0):
        """Add a GraphicsItem to the graphics layout at the specified row and column."""
        from spyde.drawing.plots.plot import Plot

        self.plot_widget.addItem(item, row, col)

        print(self.plot_widget.ci.items, " after adding item at (", row, ",", col, ")")

        if isinstance(item, Plot):
            item.plot_window = self
            self._current_plot_item = item

    def set_graphics_layout_widget(
        self,
        layout_dictionary: Dict[GraphicsItem, List[Tuple[int, int]]],
    ):
        """
        Set the graphics layout widget based on a layout dictionary. This will compare the current
        layout with the layout dictionary and rearrange the subplots accordingly.

        This is mostly used for restoring saved layouts when multiplexing `Plot` objects. This could also
        be extended to support any GraphicsItem, but currently it is only tested against `Plot` objects.

        Parameters
        ----------
        layout_dictionary : Dict[GraphicsItem, List[Tuple[int, int]]]
            A dictionary mapping plot items to their (row, col) positions in the layout.
        """
        # first remove any subplots that are not in the layout dictionary or are in the wrong position

        print(
            "The items", self.plot_widget.ci.items, " layout dict:", layout_dictionary
        )

        # only clear if there are changes
        needs_update = False
        for plot_item, pos in self.plot_widget.ci.items.items():
            if isinstance(pos, list):
                pos = pos[0]
            if (
                plot_item not in layout_dictionary
                or layout_dictionary[plot_item] != pos
            ):
                needs_update = True

        for plot_item in layout_dictionary:
            if plot_item not in self.plot_widget.ci.items:
                needs_update = True

        if needs_update:
            print(
                "Old layout:",
                self.plot_widget.ci.items,
                "New layout:",
                layout_dictionary,
            )
            try:
                self.plot_widget.clear()  # clear all items first
            except ValueError as e:
                print("Error clearing plot widget:", e)
                print("Continuing...")
            for plot_item, pos in layout_dictionary.items():
                if isinstance(pos, list):
                    pos = pos[0]
                self.add_item(plot_item, pos[0], pos[1])

    def reposition_toolbars(self):
        """Reposition the floating toolbars around the subwindow."""
        if self.current_plot_state is None:
            return
        else:
            for tb in (
                getattr(self.current_plot_state, "toolbar_right", None),
                getattr(self.current_plot_state, "toolbar_left", None),
                getattr(self.current_plot_state, "toolbar_top", None),
                getattr(self.current_plot_state, "toolbar_bottom", None),
            ):
                if tb is not None and tb.isVisible():
                    tb.move_next_to_plot()

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        self.reposition_toolbars()

    def moveEvent(self, ev: QtGui.QMoveEvent) -> None:
        super().moveEvent(ev)
        self._reposition_timer.start()

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        super().resizeEvent(ev)
        self._reposition_timer.start()

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        """Handle arrow keys to move the active selector."""
        if ev.key() not in (
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Down,
        ):
            return

        if self.last_used_selector is not None:
            # change the position of the selector
            self.timer = time.time()
            self.last_used_selector.move_roi(ev.key())
            return

    def close_window(self):
        """Close the plot window and clean up toolbars and selectors."""
        # Mark as torn down so action bindings never re-show this shell
        # (QMdiSubWindow close hides the inner container; re-showing the
        # subwindow afterwards would display only the title bar).
        self._spyde_closed = True
        for plot in self.plots:
            plot.close_plot()

        # Tree-level teardown is owned by the MDIManager: closing a navigator
        # (level-1 window) tears down the WHOLE tree — every signal/preview
        # window, toolbars, selectors — via close_signal_tree. Action-spawned
        # previews (VI, IPF, vector-OM maps) all carry .signal_tree so they are
        # included; the old level==1 path missed them. The _spyde_tree_teardown
        # guard prevents infinite recursion when close_signal_tree calls us back.
        if (self.multiplot_manager is not None
                and not getattr(self, "_spyde_tree_teardown", False)):
            level, _children = self.multiplot_manager.get_plot_window_level(
                plot_window=self
            )
            if level == 1 and self.signal_tree is not None:
                mdi = getattr(self.main_window, "mdi_manager", None)
                if mdi is not None:
                    mdi.close_signal_tree(self.signal_tree)
                    return  # the manager closed us (and everything else)

        logger.info("Closing parent selector if exists")
        if self.parent_selector is not None:
            logger.info("Closing parent selector")
            print("Removing parent selector:", self.parent_selector)
            nav_selectors = self.multiplot_manager.navigation_selectors.get(
                self.parent_selector.parent, []
            )
            if self.parent_selector in nav_selectors:
                nav_selectors.remove(self.parent_selector)
            self.parent_selector.hide()
            self.parent_selector.close()

        # Remove from main window tracking
        if hasattr(self.main_window, "plot_subwindows"):
            try:
                self.main_window.plot_subwindows.remove(self)
                logger.info("MultiPlot: Removed plot from main window tracking.")
            except ValueError:
                pass

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        if callable(self.on_close_request):
            ev.ignore()
            self.on_close_request()
            return
        self.close_window()
        super().closeEvent(ev)

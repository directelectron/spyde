from __future__ import annotations
from PySide6 import QtCore
from hyperspy.signal import BaseSignal

from typing import TYPE_CHECKING, Optional, List

from spyde.drawing.toolbars.plot_control_toolbar import get_toolbar_actions_for_plot
from spyde.drawing.toolbars.toolbar import RoundedToolBar

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import MultiplotManager, Plot


class PlotState:
    """
    Represents the complete visualization state for a (Plot, Signal) pair.

    Stores:
      - Current signal and dimensionality
      - Contrast / brightness (levels and percentiles)
      - Colormap choice
      - Dynamic / selector-driven plotting flag
      - Active selectors and any child plots they spawned
      - Four side toolbars (top/bottom/left/right) whose actions depend on signal type & dimensionality

    A Plot holds (and switches between) multiple PlotState instances when the user changes
    the active signal. Restoring a PlotState re-applies its selectors, toolbars, and visual parameters.
    """

    def __init__(
        self,
        signal: BaseSignal,
        plot: "Plot",
        dimensions: Optional[int] = None,
        dynamic: bool = True,
    ):
        # Each PlotState is tied to a particular signal and a particular Plot instance
        # This allows us a unique state monitor for each signal/plot combination and
        # allows us to save things like brightness/contrast settings as well as the
        # current toolbars/selectors and child plots associated with this state.

        # This is ultimately what needs to be saved/restored when switching signals in a plot
        # and what needs to be serialized when (eventually) saving/loading a project.

        self.current_signal: BaseSignal = signal
        self.plot: "Plot" = plot
        # self.plot_window: "PlotWindow" = plot.

        # Visualization parameters. The min/max percentile are used to determine the contrast/brightness
        self.min_percentile = 100
        self.max_percentile = 0
        self.min_level = 0
        self.max_level = 1
        self.colormap = "gray"  # default colormap

        self.dynamic: bool = (
            dynamic  # if the image/plot will update based on some selector.
        )

        # Selectors which are tied to this particular "State" of the signal...
        # When the state is changed these selectors should be removed from the plot
        # And the children plots should be hidden. When the state is restored these
        # selectors and children plots should be restored/shown.
        # plot_selectors include things like Virtual Images.

        self.plot_selectors: List[object] = []
        self.signal_tree_selectors: List[object] = []

        self.toolbar_top: Optional[RoundedToolBar] = None
        self.toolbar_bottom: Optional[RoundedToolBar] = None
        self.toolbar_left: Optional[RoundedToolBar] = None
        self.toolbar_right: Optional[RoundedToolBar] = None

        self.plot_selectors_children: List["Plot"] = []
        self.signal_tree_selectors_children: List["Plot"] = []

        # for navigation plots make sure we transpose to get the plot dimensions correct...
        self.dimensions = self.current_signal.axes_manager.signal_dimension

        # Initialize toolbars for this plot state so that we can add actions to them
        self._initialize_toolbars()
        print("Initialized PlotState:", self)

    def __repr__(self):
        return (
            f"<PlotState signal={self.current_signal}, "
            f"dimensions={self.dimensions},"
            f" dynamic={self.dynamic}>"
        )

    def _initialize_toolbars(self) -> None:
        """Create (or recreate) the four side toolbars for this state and populate them with actions."""
        self.toolbar_right = RoundedToolBar(
            title="Plot Controls",
            plot_state=self,
            parent=self.plot.main_window,
            position="right",
        )
        self.toolbar_left = RoundedToolBar(
            title="Plot Controls",
            plot_state=self,
            parent=self.plot.main_window,
            position="left",
        )
        self.toolbar_top = RoundedToolBar(
            title="Plot Controls",
            plot_state=self,
            parent=self.plot.main_window,
            position="top",
        )
        self.toolbar_bottom = RoundedToolBar(
            title="Plot Controls",
            plot_state=self,
            parent=self.plot.main_window,
            position="bottom",
        )

        # Ensure they are visible
        for tb in (
            self.toolbar_right,
            self.toolbar_left,
            self.toolbar_top,
            self.toolbar_bottom,
        ):
            tb.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
            tb.show()

        functions, icons, names, toolbar_sides, toggles, params, sub_functions, setup_fns = (
            get_toolbar_actions_for_plot(self)
        )

        # Add actions to the appropriate toolbars
        for func, icon, name, side, toggle, param, sub_function, setup_fn in zip(
            functions, icons, names, toolbar_sides, toggles, params, sub_functions, setup_fns
        ):
            print(f"Adding toolbar action: {name} to {side} toolbar")
            print(f"Function: {func}, Icon: {icon}, Toggle: {toggle}, Params: {param}")
            if side == "right":
                _, action_widget = self.toolbar_right.add_action(
                    name, icon, func, toggle, param, sub_function
                )
            elif side == "left":
                _, action_widget = self.toolbar_left.add_action(
                    name, icon, func, toggle, param, sub_function
                )
            elif side == "top":
                _, action_widget = self.toolbar_top.add_action(
                    name, icon, func, toggle, param, sub_function
                )
            elif side == "bottom":
                _, action_widget = self.toolbar_bottom.add_action(
                    name, icon, func, toggle, param, sub_function
                )
            else:
                action_widget = None

            # Wire optional one-time setup callback onto the CaretParams widget.
            if setup_fn is not None and action_widget is not None:
                from spyde.drawing.toolbars.caret_group import CaretParams
                if isinstance(action_widget, CaretParams):
                    tb = (
                        self.toolbar_right if side == "right"
                        else self.toolbar_left if side == "left"
                        else self.toolbar_top if side == "top"
                        else self.toolbar_bottom
                    )
                    action_widget._on_first_show_cb = lambda fn=setup_fn, t=tb, n=name: fn(t, action_name=n)

        for tb in [
            self.toolbar_right,
            self.toolbar_left,
            self.toolbar_top,
            self.toolbar_bottom,
        ]:
            tb.set_size()
            # if there are no actions hide the toolbar
            if tb.num_actions() == 0:
                tb.hide()
            else:
                tb.show()
            tb.raise_()
        # start with toolbars hidden
        self.hide_toolbars()

    def show_toolbars(self) -> None:
        """Show all toolbars that have at least one action."""
        for tb in [
            self.toolbar_right,
            self.toolbar_left,
            self.toolbar_top,
            self.toolbar_bottom,
        ]:
            if tb is None:
                pass
            elif tb.num_actions() == 0:
                tb.hide()
            else:
                tb.show()
            tb.raise_()

            # Add all the plot items to the plot. This is needed when restoring a PlotState
            # This adds things like Selectors, ROIs, etc back to the plot associated with
            # specific toolbar actions.
            for action in tb.action_widgets:
                if "plot_items" in tb.action_widgets[action]:
                    for key in tb.action_widgets[action]["plot_items"]:
                        self.plot.addItem(tb.action_widgets[action]["plot_items"][key])

    def update_toolbars(self) -> None:
        """Recompute size/visibility for each toolbar after external changes to actions."""
        for tb in [
            self.toolbar_right,
            self.toolbar_left,
            self.toolbar_top,
            self.toolbar_bottom,
        ]:
            tb.set_size()
            # if there are no actions hide the toolbar
            if tb.num_actions() == 0:
                tb.hide()
            else:
                tb.show()
            tb.raise_()

    def hide_toolbars(self) -> None:
        """Hide all toolbars for this state (used when the PlotState becomes inactive)."""
        for tb in [
            self.toolbar_right,
            self.toolbar_left,
            self.toolbar_top,
            self.toolbar_bottom,
        ]:
            if tb:  # check if toolbar exists
                tb.hide()
                for action in tb.action_widgets:
                    if "plot_items" in tb.action_widgets[action]:
                        print("Restoring plot items for action:", action)
                        for key in tb.action_widgets[action]["plot_items"]:
                            self.plot.removeItem(
                                tb.action_widgets[action]["plot_items"][key]
                            )

    def close(self) -> None:
        self.hide_toolbars()
        for attr in ("toolbar_right", "toolbar_left", "toolbar_top", "toolbar_bottom"):
            tb = getattr(self, attr, None)
            if tb is not None:
                tb.plot = None
                tb.close()
                setattr(self, attr, None)

    def rebuild_toolbars(self) -> None:
        """
        Tear down and rebuild the side toolbars from TOOLBAR_ACTIONS.

        Needed when the action filter inputs change after construction —
        e.g. ``set_signal_type`` retypes ``current_signal``, which can add
        or remove signal-class-gated actions (Virtual Imaging, Orientation
        Mapping). Action state held in the old toolbars (carets, ROIs) is
        discarded.
        """
        was_visible = any(
            tb is not None and tb.isVisible()
            for tb in (
                self.toolbar_right,
                self.toolbar_left,
                self.toolbar_top,
                self.toolbar_bottom,
            )
        )
        self.close()
        self._initialize_toolbars()
        if was_visible:
            self.show_toolbars()


class MultiImageManager:
    """The MultiImageManager manages multiple images within a single plotting context.

    The idea is that there are cases where you have multiple images which represent
    things like different Virtual Images, different channels, etc.  These images
    can be toggled between in the same plot area. In addition, you might want to overlay
    multiple image at once, adjusting their opacities independently. Or you might want
    to create a grid of images.

    In the case of 1D Spectra this could be multiple spectra overlaid, or stacked, etc.
    In the case of 2D images this could be multiple images in a grid.

    The idea is that this functionality is shared between multiple classes. The most common
    use case is the NavigationPlotManager, but it is also used to manage multiple Virtual
    Images.

    Parameters
    ----------
    plot_states : List[PlotState]
        A list of PlotState instances managed by this MultiImageManager.
    plot : Plot
        The Plot instance associated with this MultiImageManager.

    """

    def __init__(
        self,
        plot_states: List[PlotState],
        plot: "Plot",
    ):

        # initialize the MultiImageManager with the provided plot states and plot
        self.plot_states: List[PlotState] = plot_states
        self.plot: "Plot" = plot

    def add_plot_state(self, plot_state: PlotState):
        """
        Add a new PlotState to the MultiImageManager.

        Parameters
        ----------
        plot_state : PlotState
            The PlotState to add.

        Returns
        -------
        None
        """
        self.plot_states.append(plot_state)

    def remove_plot_state(self, plot_state: PlotState):
        """
        Remove a PlotState from the MultiImageManager.

        Parameters
        ----------
        plot_state : PlotState
            The PlotState to remove.
        Returns
        -------
        None
        """
        self.plot_states.remove(plot_state)


class NavigationManagerState:
    """State container for a NavigationPlotManager.

    Wraps a navigation-capable signal and builds the required PlotState list
    corresponding to (signal_dimension, navigation_dimension). Only up to 4D total supported.
    """

    def __init__(
        self,
        signal: BaseSignal,
        plot_manager: "MultiplotManager",
    ):
        # only up to 4 navigation dimensions supported for now...
        dimensions = (
            signal.axes_manager.signal_dimension
            + signal.axes_manager.navigation_dimension
        )
        if dimensions > 4:
            raise ValueError(
                "NavigationManagerState only supports up to 4D signals for now."
            )

        if (
            signal.axes_manager.signal_dimension > 2
            or signal.axes_manager.navigation_dimension > 2
        ):
            #  Force it to be 2D signals for now...
            signal = signal.transpose(2)

        self.current_signal: BaseSignal = signal
        self.plot_manager: "MultiplotManager" = plot_manager

        self.dimensions: List[int] = [
            signal.axes_manager.signal_dimension,
            signal.axes_manager.navigation_dimension,
        ]

        # The list of plot states for each plot in the navigation manager

        print("this is the signal!", signal)

        # need to update for multiple dimensional navigators (5D STEM etc)
        self.plot_states: List[PlotState] = [
            PlotState(signal=signal, plot=self.plot_manager.plots[0], dimensions=dim)
            for dim in self.dimensions
            if dim > 0
        ]

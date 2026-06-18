from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Tuple, Optional

if TYPE_CHECKING:
    from spyde.signal_tree import BaseSignalTree
    from spyde.backend.session import Session
    from spyde.drawing.selectors import BaseSelector
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow

from hyperspy.signal import BaseSignal
from spyde.drawing.update_functions import update_from_navigation_selection

import logging

logger = logging.getLogger(__name__)


def _plot_window_dims(plot_window: "PlotWindow") -> int:
    """Derive the displayed dimensionality of a PlotWindow's current plot.

    Prefer the active plot state's ``dimensions``; if that's missing or 0
    (e.g. a fully-reduced navigator image), fall back to the displayed
    signal's ``signal_dimension``.
    """
    try:
        plot = plot_window.current_plot_item
        state = plot.plot_state
        if state is not None and state.dimensions:
            return state.dimensions
        return state.current_signal.axes_manager.signal_dimension
    except Exception:
        return 2


class MultiplotManager:
    """Manages multiple Plot instances for navigation plots in a signal tree."""

    def __init__(
        self,
        signal_tree: "BaseSignalTree",
        selector_type=None,
        # Accept both 'session' (new) and 'main_window' (legacy) for compat
        session: "Session | None" = None,
        main_window=None,
    ):
        self.session = session or main_window
        self.plots: Dict["PlotWindow", List["Plot"]] = {}
        self.plot_windows: Dict["PlotWindow", Dict] = {}
        self.navigation_selectors: Dict["PlotWindow", List["BaseSelector"]] = {}
        self.signal_tree = signal_tree
        self.navigation_depth = 1

        if self.nav_dim < 1:
            raise ValueError(
                "MultiplotManager requires at least 1 navigation dimension."
            )
        elif self.nav_dim < 3:
            nav_plot_window = self.session.add_plot_window(
                is_navigator=True, plot_manager=self, signal_tree=self.signal_tree
            )
            nav_plot = nav_plot_window.add_new_plot()
            self.plot_windows[nav_plot_window] = {}
            self.plots[nav_plot_window] = [nav_plot]
            self.navigation_selectors[nav_plot_window] = []
            for signal in self.signal_tree.navigator_signals.values():
                self.add_plot_states_for_navigation_signals(signal)
            self.add_navigation_selector_and_signal_plot(
                nav_plot_window, selector_type=selector_type
            )

        elif self.nav_dim < 5:
            self.navigation_depth = 2
            plot_window = self.session.add_plot_window(
                is_navigator=True, plot_manager=self, signal_tree=self.signal_tree
            )
            nav_plot_1d = plot_window.add_new_plot()
            self.plot_windows[plot_window] = {}
            self.plots[plot_window] = [nav_plot_1d]
            self.navigation_selectors[plot_window] = []
            for signal in self.signal_tree.navigator_signals.values():
                self.add_plot_states_for_navigation_signals(signal)
            new_window = self.add_navigation_selector_and_signal_plot(plot_window)
            self.add_navigation_selector_and_signal_plot(new_window)

    @property
    def all_navigation_selectors(self) -> List["BaseSelector"]:
        selectors = []
        for sel_list in self.navigation_selectors.values():
            selectors.extend(sel_list)
        return selectors

    def add_plot_states_for_navigation_signals(self, signals: List[BaseSignal]) -> None:
        for plot_window in self.plot_windows:
            for plot in self.plots[plot_window]:
                # The navigator image is displayed via its *signal* axes (a 2D
                # navigator → an imshow; a 1D navigator → a line). That displayed
                # dimensionality is what both the figure type and the selector
                # choice must key off of — not navigation_dimension (which is 0
                # for a fully-reduced navigator image).
                dim = signals[0].axes_manager.signal_dimension
                plot.add_plot_state(
                    signal=signals[0],
                    dimensions=dim,
                    dynamic=False,
                )
            if len(signals) > 1:
                for sub_plot_window in self.plot_windows[plot_window]:
                    for plot in self.plots.get(sub_plot_window, []):
                        dim = signals[1].axes_manager.signal_dimension
                        plot.add_plot_state(
                            signal=signals[1],
                            dimensions=dim,
                            dynamic=True,
                        )

    @property
    def navigation_signals(self) -> dict:
        return self.signal_tree.navigator_signals

    @property
    def nav_dim(self) -> int:
        return self.signal_tree.nav_dim

    def get_plot_window_level(
        self, plot_window: "PlotWindow"
    ) -> Tuple[int, Optional[dict]]:
        level = 1
        current_window = plot_window
        children_dictionary = self.plot_windows.get(plot_window, None)
        while True:
            parent_found = False
            for pw, children in self.plot_windows.items():
                if current_window in children:
                    level += 1
                    if children_dictionary is None:
                        children_dictionary = children
                    current_window = pw
                    parent_found = True
                    break
            if not parent_found:
                break
        return level, children_dictionary

    def add_navigation_selector_and_signal_plot(
        self, plot_window: "PlotWindow", selector_type=None
    ) -> "PlotWindow":
        from spyde.drawing.selectors import (
            IntegratingSelector1D,
            IntegratingSSelector2D,
            BaseSelector,
        )

        dim = _plot_window_dims(plot_window)

        if dim == 1 and selector_type is None:
            selector_type = IntegratingSelector1D
        elif dim == 2 and selector_type is None:
            selector_type = IntegratingSSelector2D
        elif selector_type is not None and not (
            isinstance(selector_type, type) and issubclass(selector_type, BaseSelector)
        ):
            raise ValueError("selector_type must be a BaseSelector subclass.")

        window_level, children_dict = self.get_plot_window_level(plot_window=plot_window)
        is_navigator = window_level < self.navigation_depth

        if plot_window not in self.navigation_selectors:
            self.navigation_selectors[plot_window] = []

        window = self.session.add_plot_window(
            is_navigator=is_navigator,
            plot_manager=self,
            signal_tree=self.signal_tree,
        )

        children_dict[window] = {}

        child = window.add_new_plot()
        if window not in self.plots:
            self.plots[window] = []
        self.plots[window].append(child)

        selector = selector_type(
            parent=plot_window,
            children=child,
            multi_selector=True,
            update_function=update_from_navigation_selection,
        )

        self.navigation_selectors[plot_window].append(selector)
        plot_window.last_used_selector = selector

        # Register the selector so the right-dock toggle can switch it between
        # crosshair (point) and integrating (region) modes, and tell the dock
        # it exists. Composite selectors start in crosshair mode.
        try:
            mode = "integrate" if getattr(selector, "is_integrating", False) else "crosshair"
            self.session.register_nav_selector(plot_window.window_id, selector)
            from spyde.backend.ipc import emit
            emit({
                "type": "selector_info",
                "window_id": plot_window.window_id,
                "mode": mode,
                "title": "Navigator",
            })
        except Exception:
            pass

        if is_navigator:
            for signal in self.signal_tree.navigator_signals.values():
                self.add_plot_states_for_navigation_signals(signal)
        else:
            self.signal_tree.create_plot_states(plot=child)

        selector.update_data()
        if child.current_data is not None:
            child.update_data(child.current_data)
        self.signal_tree.signal_plots.append(child)
        child.needs_auto_level = True
        return window

    @property
    def all_plot_windows(self) -> List["PlotWindow"]:
        windows: List["PlotWindow"] = []

        def collect_windows(windows_dict: dict) -> None:
            for win, children in windows_dict.items():
                if win not in windows:
                    windows.append(win)
                if children:
                    collect_windows(children)

        collect_windows(self.plot_windows)

        for win in list(windows):
            plot = getattr(win, "current_plot_item", None)
            state = getattr(plot, "plot_state", None) if plot else None
            if state is None:
                continue
            for toolbar in [
                state.toolbar_right, state.toolbar_left,
                state.toolbar_bottom, state.toolbar_top,
            ]:
                plot_windows_attr = getattr(toolbar, "plot_windows", None)
                if plot_windows_attr is None:
                    continue
                for item in plot_windows_attr:
                    windows.append(item[1] if isinstance(item, tuple) else item)
        return windows

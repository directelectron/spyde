"""
base_selector.py — BaseSelector using anyplotlib widgets.

Replaces the pyqtgraph ROI-based selectors.  Each selector wraps an
anyplotlib interactive widget and uses its callback events to drive
slice-and-update of child plots.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np

from spyde.drawing.selectors.utils import broadcast_rows_cartesian

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow


class _StubWidget:
    """Shim so sidebar code that calls widget.hide() doesn't raise."""
    def hide(self) -> None: pass
    def show(self) -> None: pass
    def setVisible(self, v: bool) -> None: pass


class BaseSelector:
    """
    Base class for anyplotlib-backed selectors.

    Parameters
    ----------
    parent : Plot | PlotWindow
        The data-source plot.
    children : Plot | list[Plot]
        Child plots that receive sliced data when the selector moves.
    update_function : callable | list[callable]
        Called as ``fn(selector, child_plot, indices)`` → new data for child.
    live_delay : int
        Debounce delay in milliseconds before dispatching the update.
    multi_selector : bool
        If True, chain-compose with upstream selectors.
    """

    def __init__(
        self,
        parent: Union["PlotWindow", "Plot"],
        children: Union["Plot", List["Plot"]],
        update_function: Union[Callable, List[Callable]],
        width: int = 3,
        color: str = "green",
        hover_color: str = "red",
        live_delay: int = 2,
        resize_on_move: bool = False,
        multi_selector: bool = False,
    ):
        self.parent = parent
        self.color = color
        self.hover_color = hover_color
        # Used as a THROTTLE interval (see update_data). Enforce a sane minimum so
        # a continuous drag submits at most ~25 computes/sec instead of one per
        # mouse event (~60/sec) — the latter floods Dask with superseded futures
        # and the navigator stutters / new chunks don't paint while loading.
        self.live_delay = max(float(live_delay), 40.0) / 1000.0  # ms → s
        self.multi_selector = multi_selector

        if not isinstance(children, list):
            self.children: Dict["Plot", Callable] = {children: update_function}
            self.active_children: List["Plot"] = [children]
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

        # No Qt widget — stub so sidebar compatibility shims don't break
        self.widget = _StubWidget()
        self.is_integrating = False
        self.current_indices: np.ndarray | None = None
        self.linked_selectors: list = []

        # Hooks fired (with the new indices) whenever the selection changes —
        # used by feature overlays (e.g. Find Vectors / Orientation markers) to
        # redraw on the signal plot as the navigator moves. Each is called
        # ``hook(indices)``; exceptions are swallowed so one bad overlay can't
        # break navigation.
        self.index_hooks: list = []

        # anyplotlib widget (set by subclasses)
        self.roi = None   # Alias for the anyplotlib Widget
        self._widget = None  # type: anyplotlib Widget

        # Debounce: timer replaced by threading.Timer
        self._pending_timer: threading.Timer | None = None
        self.update_function = update_function
        self._last_size_sig = None

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _get_selected_indices(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement _get_selected_indices.")

    def _get_selected_indices_and_clip(self) -> np.ndarray:
        indices = self._get_selected_indices()
        plot = self.current_plot
        if plot is None or plot.plot_state is None:
            return indices
        axes_manager = plot.plot_state.current_signal.axes_manager
        if axes_manager.navigation_dimension > 0:
            clip_shape = axes_manager.navigation_shape
        else:
            clip_shape = axes_manager.signal_shape
        # Region selectors (circle/annular) encode geometry rows whose column
        # count doesn't match the nav/signal dimensionality — clipping is
        # meaningless there, so only clip true index arrays.
        idx = np.asarray(indices)
        if idx.ndim == 2 and idx.shape[-1] == len(clip_shape):
            return np.clip(idx, 0, np.array(clip_shape) - 1)
        return idx

    def get_selected_indices(self) -> np.ndarray:
        if self.multi_selector:
            upstream = [s._get_selected_indices_and_clip()
                        for s in self.upstream_selectors()]
            current = self._get_selected_indices_and_clip()
            return broadcast_rows_cartesian(*(upstream + [current]))
        return self._get_selected_indices_and_clip()

    @property
    def current_plot(self) -> "Plot | None":
        from spyde.drawing.plots.plot_window import PlotWindow
        from spyde.drawing.plots.plot import Plot
        if isinstance(self.parent, Plot):
            return self.parent
        if isinstance(self.parent, PlotWindow):
            return self.parent.current_plot_item
        return None

    def upstream_selectors(self) -> list:
        selectors = []
        current = self.parent
        while hasattr(current, "parent_selector") and current.parent_selector is not None:
            selectors.append(current.parent_selector)
            current = current.parent_selector.parent
        return selectors

    # ── Update dispatch ───────────────────────────────────────────────────────

    def update_data(self, ev=None) -> None:
        """Trigger a THROTTLED update.

        Coalesce rapid drag events: if a fire is already scheduled in this window,
        do nothing (the timer reads the LATEST widget position when it fires). A
        plain debounce (cancel + restart on every move) with a tiny delay instead
        fires once per mouse event, swamping Dask with a compute per event — so a
        lazy navigator drag clogs and the new chunk paints poorly. Throttling
        bounds it to ~1 compute / live_delay while still tracking the cursor.
        """
        if self._pending_timer is not None:
            return
        self._pending_timer = threading.Timer(
            self.live_delay, self._throttled_fire
        )
        self._pending_timer.daemon = True
        self._pending_timer.start()

    def _throttled_fire(self) -> None:
        self._pending_timer = None
        self.delayed_update_data()

    def delayed_update_data(self, force: bool = False, update_contrast: bool = False) -> None:
        """Perform the actual data update if indices changed."""
        self._pending_timer = None
        try:
            indices = self.get_selected_indices()
        except Exception:
            return
        if not np.array_equal(indices, self.current_indices) or force:
            for child, fn in self.children.items():
                try:
                    new_data = fn(self, child, indices)
                    child.update_data(new_data)
                    if update_contrast:
                        child.needs_auto_level = True
                    if (child.multiplot_manager is not None
                            and child.plot_window in child.multiplot_manager.navigation_selectors):
                        for child_sel in child.multiplot_manager.navigation_selectors[child.plot_window]:
                            child_sel.delayed_update_data()
                except Exception as e:
                    logger.debug("selector update failed: %s", e)
            for hook in self.index_hooks:
                try:
                    hook(indices)
                except Exception as e:
                    logger.debug("index hook failed: %s", e)
            self.current_indices = indices

    # ── Visibility ─────────────────────────────────────────────────────────────

    def add_linked_roi(self, plot: "Plot") -> None:
        pass

    def hide(self) -> None:
        if self._widget is not None:
            try:
                self._widget.hide()
            except Exception as e:
                logger.debug("hiding selector widget failed: %s", e)

    def show(self) -> None:
        if self._widget is not None:
            try:
                self._widget.show()
            except Exception as e:
                logger.debug("showing selector widget failed: %s", e)

    def close(self) -> None:
        if self._pending_timer is not None:
            self._pending_timer.cancel()
        self.hide()
        # A bare widget.hide() only emits a targeted event that a later full
        # repaint overwrites, so the ROI lingers. Re-push the panel so the
        # hidden selector actually disappears (e.g. when its output window is
        # closed). Matches set_integrating's _force_overlay_repaint.
        try:
            get = getattr(self, "_get_plot2d", None) or getattr(self, "_get_plot1d", None)
            if get is not None:
                panel = get()
                if panel is not None:
                    panel._push()
        except Exception as e:
            logger.debug("re-pushing panel on selector close failed: %s", e)


class IntegratingSelectorMixin:
    """Mixin that provides integrate-over-region behaviour."""
    is_integrating: bool = False

    def set_integrating(self, enabled: bool) -> None:
        self.is_integrating = enabled
        self.delayed_update_data(force=True)

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
        # Latest-position-wins, future-cancel model (NOT a long lock).
        #
        # A previous version held an RLock across the WHOLE update body — but the
        # body calls the per-child update fn, which can block for 100s of ms on a
        # compute. With two signals / selectors live at once that lock stays held
        # across the slow compute and STALLS every other update: the navigator
        # flashes but the image won't track the crosshair (the lock isn't free).
        #
        # Instead we follow the greedy future-cancel workflow the distributed
        # path was built for: each fire bumps a generation counter; the update
        # body runs UNLOCKED so a newer fire starts immediately and (via
        # update_from_navigation_selection) cancels the in-flight future and
        # submits its own. A body only commits its result (current_indices +
        # index hooks) if it is still the latest generation — a superseded body
        # drops its now-stale work. The short per-signal cache lock
        # (_cache_lock_ctx) still serialises just the cancel+submit critical
        # section, and the "plot.current_data is future" guard drops stale
        # frames. See test_navigator_race.
        self._gen_lock = threading.Lock()
        self._update_gen = 0
        self._fn_gen = 0          # generation of the body currently in fn()
        self.update_function = update_function
        self._last_size_sig = None

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _display_axes(self):
        """The (x_axis, y_axis) HyperSpy axes the parent plot is drawn against,
        in widget-coordinate order (axis 0 = x = columns, axis 1 = y = rows).

        The plot's image is rendered with these axes' CALIBRATION (scale +
        offset + units), so the interactive widget reports its position in their
        DATA coordinates — e.g. nanometres for a 3 nm-step navigator. Returns
        ``None`` when calibration is unavailable (then coords are pixel
        indices already)."""
        plot = self.current_plot
        if plot is None or getattr(plot, "plot_state", None) is None:
            return None
        try:
            sig = plot.plot_state.current_signal
            sig_axes = sig.axes_manager.signal_axes
            # anyplotlib draws against signal_axes (for a navigator plot these
            # ARE the navigation axes). signal_axes is in (x, y) order.
            if len(sig_axes) >= 2:
                return sig_axes[0], sig_axes[1]
        except Exception as e:
            logger.debug("reading display axes for index mapping failed: %s", e)
        return None

    def _data_to_index(self, value_x: float, value_y: float) -> tuple[int, int]:
        """Convert a widget position in DATA coordinates to integer array
        indices, dividing out the displayed axes' scale/offset:
        ``index = round((value - offset) / scale)``.

        With unit scale and zero offset (the synthetic fixtures, and any
        uncalibrated image) this is just ``round(value)`` — so the change is a
        no-op there and only corrects calibrated images, where using the raw
        data coordinate as an index loads the WRONG pixel (the 3 nm-step bug)."""
        axes = self._display_axes()
        if axes is None:
            return int(round(value_x)), int(round(value_y))
        x_ax, y_ax = axes
        try:
            sx = float(getattr(x_ax, "scale", 1.0)) or 1.0
            sy = float(getattr(y_ax, "scale", 1.0)) or 1.0
            ox = float(getattr(x_ax, "offset", 0.0))
            oy = float(getattr(y_ax, "offset", 0.0))
            return int(round((value_x - ox) / sx)), int(round((value_y - oy) / sy))
        except Exception as e:
            logger.debug("data→index conversion failed: %s", e)
            return int(round(value_x)), int(round(value_y))

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

    def is_stale_body(self) -> bool:
        """True when the update body currently running ``fn`` has been superseded
        by a newer navigator position. Used by the cache critical section to skip
        cancelling/submitting on behalf of a stale position (which would cancel
        the latest position's in-flight chunk future)."""
        return self._fn_gen != self._update_gen

    def delayed_update_data(self, force: bool = False, update_contrast: bool = False) -> None:
        """Perform the actual data update if indices changed.

        Latest-position-wins (no long lock): bump a generation counter, run the
        per-child update fn UNLOCKED (so a newer fire can start and supersede
        this one through the future-cancel path), and only commit the result if
        this body is still the latest generation. The per-signal cache lock
        inside the update fn serialises the short cancel+submit section; this
        method never blocks a concurrent selector across a slow compute."""
        self._pending_timer = None
        try:
            indices = self.get_selected_indices()
        except Exception:
            return

        # Reserve this update's generation. A staleness check against it after
        # the (slow, unlocked) child compute lets a newer fire win.
        with self._gen_lock:
            if np.array_equal(indices, self.current_indices) and not force:
                return
            self._update_gen += 1
            my_gen = self._update_gen

        for child, fn in self.children.items():
            # A newer fire started while we were working — its future-cancel has
            # already superseded ours; drop this stale body.
            if my_gen != self._update_gen:
                return
            # Let the child fn (update_from_navigation_selection) re-check, INSIDE
            # its cache lock, that we are still the latest before it cancels
            # surrounding blocks + submits — otherwise a stale body's
            # cancel_surrounding() races and kills the LATEST position's future
            # ("get_inds cancelled: lost dependencies"). See test_navigator_race.
            self._fn_gen = my_gen
            try:
                new_data = fn(self, child, indices)
                # None == the body detected it was superseded inside fn and
                # skipped the cache touch; do NOT clobber current_data (that would
                # drop the latest future's pending result).
                if new_data is None:
                    return
                child.update_data(new_data)
                if update_contrast:
                    child.needs_auto_level = True
                if (child.multiplot_manager is not None
                        and child.plot_window in child.multiplot_manager.navigation_selectors):
                    for child_sel in child.multiplot_manager.navigation_selectors[child.plot_window]:
                        child_sel.delayed_update_data()
            except Exception as e:
                logger.debug("selector update failed: %s", e)

        # Commit only if still latest — otherwise the newer fire owns the state.
        with self._gen_lock:
            if my_gen != self._update_gen:
                return
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

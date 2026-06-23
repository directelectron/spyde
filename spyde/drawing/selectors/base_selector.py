"""
base_selector.py — BaseSelector using anyplotlib widgets.

Replaces the pyqtgraph ROI-based selectors.  Each selector wraps an
anyplotlib interactive widget and uses its callback events to drive
slice-and-update of child plots.
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

import numpy as np

from spyde.drawing.selectors.utils import broadcast_rows_cartesian

logger = logging.getLogger(__name__)

import os as _os
# Per-frame navigator trace logs are gated behind this (they fire on every
# crosshair move and flood the IPC log/panel at DEBUG, which adds real lag).
_NAV_TIMING = _os.environ.get("SPYDE_NAV_TIMING") == "1"


class _NavDispatcher:
    """A single serial worker thread that runs ALL navigator selector updates,
    one at a time — recreating the Qt app's behaviour, where every selector
    update ran on the one GUI event loop and never overlapped.

    Why this exists: the Electron port fired each update on its OWN
    ``threading.Timer`` thread, so updates ran CONCURRENTLY and raced hyperspy's
    CachedDaskArray block bookkeeping (``ValueError: (i, j) is not in list``).
    The fix was a per-signal lock around the cache call — but that lock then
    serialised the work in a way that STALLED the drag. Running every update on
    one dedicated thread removes the concurrency at the source: no race, so no
    lock, and the cache call is never re-entered. The drag stays responsive
    because the dispatcher is LATEST-WINS — a newer update for a selector
    replaces any still-queued older one (coalescing), exactly like the throttle
    intended.

    Per (selector, kwargs) we keep at most ONE pending job; submitting again
    overwrites it. The worker pulls the current job, runs it to completion, then
    looks for the next. So at any instant the in-flight job is the latest the
    user asked for, and stale intermediate positions are dropped without ever
    being computed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: "dict[int, tuple]" = {}   # selector id -> (selector, kwargs)
        self._wake = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="nav-dispatch", daemon=True
        )
        self._thread.start()

    def submit(self, selector, **kwargs) -> None:
        """Queue (or replace) this selector's pending update. Latest wins."""
        with self._lock:
            self._pending[id(selector)] = (selector, kwargs)
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait()
            # Drain everything currently pending; new submissions that arrive
            # while we run are picked up on the next loop (their wake re-fires).
            with self._lock:
                jobs = list(self._pending.values())
                self._pending.clear()
                self._wake.clear()
            for selector, kwargs in jobs:
                try:
                    selector._run_update(**kwargs)
                except Exception as e:
                    logger.debug("nav dispatch update failed: %s", e)


# One dispatcher for the whole process — the single serial lane all navigator
# updates flow through (the Qt event-loop equivalent).
_nav_dispatcher = _NavDispatcher()


def event_handler_fn(method: Callable) -> Callable:
    """Wrap a BOUND METHOD in a plain function for ``add_event_handler``.

    anyplotlib's ``add_event_handler`` does ``fn._event_types = …`` on the
    handler (callbacks.py), but a Python *bound method* is read-only and can't
    take attributes → ``'method' object has no attribute '_event_types'``, so the
    whole selector widget fails to init (no crosshair/rectangle, dead navigator).
    A module-level function wrapper accepts the attribute. The caller MUST keep a
    reference to the returned wrapper (e.g. store it on the selector) or it (and
    anyplotlib's weak callback registration) may be garbage-collected."""
    @functools.wraps(method)
    def _handler(*args, **kwargs):
        return method(*args, **kwargs)
    return _handler

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

        # Throttle: drop ROI/crosshair drag events that arrive within live_delay
        # of the last fire (the dispatcher reads the latest position when it
        # runs). All updates then run serially on the single _nav_dispatcher
        # thread, so there is NO concurrency to guard — no generation counters,
        # no per-signal cache lock. Latest-wins is handled by the dispatcher
        # coalescing repeated submissions for this selector into one pending job.
        self._last_fire_t = 0.0
        # Settle re-fire: a single trailing timer, (re)armed on every move and
        # cancelled by the next move, that fires ONE forced update once motion
        # stops. See update_data — it guarantees the resting frame computes even
        # though every intermediate future got cancelled by latest-wins.
        self._settle_timer: threading.Timer | None = None
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
        """Trigger a navigator update on the single serial dispatcher.

        Every selector move just queues the LATEST position on the dispatcher
        (coalescing: a newer submit replaces any still-queued one for this
        selector). The dispatcher runs one update at a time, and the per-future
        staleness guard in ``_on_plot_ready`` drops any frame that a newer
        position has already superseded — so the DP always converges on the
        cursor's final position without us tracking in-flight frames here.

        This mirrors the proven Qt design: submit-latest + drop-stale. The earlier
        Electron port added self-pacing (skip while a frame is in flight, then a
        trailing re-fire) on top of this, which created a self-perpetuating timer
        loop that re-emitted the same handful of shm buffers forever — removed.

        Settle re-fire: during a fast drag, every intermediate get_inds /
        write_shared_array future is CANCELLED by the next move (latest-wins +
        hyperspy's cache GC dropping its deps), so they never compute a real frame
        — the buffer keeps the last frame that happened to survive. When motion
        stops (including just holding still mid-drag, no mouse-up), nothing
        re-submits the resting position, so the DP is stuck on that stale frame
        even though a single re-fire would resolve it. We therefore (re)arm ONE
        trailing timer here; the next move cancels it, and when the user finally
        rests it fires a single FORCED update for the resting position. Because no
        newer move follows, that future is not cancelled and paints. This timer has
        NO in-flight gate (unlike the removed self-pacing), so it cannot wedge.
        """
        _nav_dispatcher.submit(self)
        self._arm_settle()

    def _arm_settle(self) -> None:
        """(Re)arm the single settle timer — fires one forced update once motion
        stops (see update_data). Cancelled and replaced by the next move."""
        t = self._settle_timer
        if t is not None:
            t.cancel()
        # Wait a touch longer than live_delay (already in SECONDS) so a continuing
        # drag always cancels this before it fires; ~120 ms of stillness = settled.
        delay = max(0.12, (self.live_delay or 0.0) + 0.1)
        nt = threading.Timer(delay, self._settle_fire)
        nt.daemon = True
        self._settle_timer = nt
        nt.start()

    def _settle_fire(self) -> None:
        """Motion has stopped — force one final update for the resting position so
        the frame that all the cancelled in-flight futures never produced actually
        computes and paints. force=True bypasses the dup short-circuit in
        _run_update (an earlier move already committed current_indices here)."""
        self._settle_timer = None
        _nav_dispatcher.submit(self, force=True)

    def delayed_update_data(self, force: bool = False, update_contrast: bool = False) -> None:
        """Public entry point: queue an update on the single serial dispatcher.

        All navigator updates run on ONE worker thread (the Qt event-loop
        equivalent), so they never overlap and the cache is never re-entered —
        hence no per-signal cache lock is needed. Latest-wins: a newer submission
        for this selector replaces any still-queued one."""
        _nav_dispatcher.submit(self, force=force, update_contrast=update_contrast)

    def _run_update(self, force: bool = False, update_contrast: bool = False) -> None:
        """The actual update body — ALWAYS runs on the dispatcher thread, one at a
        time. Because nothing else can be running an update concurrently, the
        cache call inside ``fn`` is safe without a lock and the generation /
        stale-body gymnastics are unnecessary (a superseded position was already
        dropped from the dispatcher's pending slot before it ever ran)."""
        try:
            indices = self.get_selected_indices()
        except Exception:
            return

        # Short-circuit a repeat of the SAME position. anyplotlib fires BOTH
        # pointer_move and pointer_up for a single release, so one interaction
        # produces two submits with identical indices; without this they'd both
        # recompute the same frame. Commit current_indices UP FRONT (not at the
        # end) so the duplicate skips here even if it runs back-to-back before the
        # body finishes. force=True bypasses (repaint after a signal/contrast
        # change at an unchanged position).
        if np.array_equal(indices, self.current_indices) and not force:
            return
        self.current_indices = indices

        # TEMP (ungated): trace each selector's computed indices + how many
        # children/chained selectors it drives, to debug the 5-D time→DP chain.
        logger.debug("[NAV-IDX] %s indices=%s force=%s nchildren=%d multi=%s",
                     type(self).__name__, np.asarray(indices).tolist(), force,
                     len(self.children), self.multi_selector)

        for child, fn in self.children.items():
            try:
                new_data = fn(self, child, indices)
                if new_data is None:
                    continue
                child.update_data(new_data)
                if update_contrast:
                    child.needs_auto_level = True
                # TEMP (ungated): does this update CHAIN to a downstream navigator?
                mm = child.multiplot_manager
                gate = (mm is not None
                        and child.plot_window in mm.navigation_selectors)
                logger.debug("[CHAIN] %s updated child win=%s gate=%s nav_keys=%s",
                             type(self).__name__, getattr(child, "window_id", None),
                             gate,
                             [getattr(k, "window_id", None) for k in mm.navigation_selectors]
                             if mm else None)
                if gate:
                    downstream = mm.navigation_selectors[child.plot_window]
                    logger.debug("[CHAIN]  → re-firing %d downstream selector(s): %s",
                                 len(downstream),
                                 [type(s).__name__ for s in downstream])
                    for child_sel in downstream:
                        child_sel.delayed_update_data()
            except Exception as e:
                logger.debug("selector update failed: %s", e)

        # Fire index hooks (overlays) with the committed position.
        for hook in self.index_hooks:
            try:
                hook(indices)
            except Exception as e:
                logger.debug("index hook failed: %s", e)

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
        self.hide()
        # Stop a pending settle re-fire so it can't submit against a closed plot.
        t = getattr(self, "_settle_timer", None)
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass
            self._settle_timer = None
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

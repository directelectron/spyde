"""1D Selectors using anyplotlib interactive widgets."""
from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING, Union, List

from spyde.drawing.selectors.base_selector import (
    BaseSelector,
    IntegratingSelectorMixin,
    MAX_REGION_EXTENT_PER_DIM,
    event_handler_fn,
)

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow

logger = logging.getLogger(__name__)


def _signal_axis(selector: BaseSelector):
    """Return (scale, offset) for the first signal axis, or (1.0, 0.0)."""
    try:
        plot = selector.current_plot
        signal = plot.plot_state.current_signal
        axs = signal.axes_manager.signal_axes[0]
        return float(axs.scale), float(axs.offset)
    except Exception:
        return 1.0, 0.0


class InfiniteLineSelector(BaseSelector):
    """Single-index selector — wraps anyplotlib VLineWidget."""

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
            parent, children, update_function,
            live_delay=live_delay, multi_selector=multi_selector,
        )
        self._widget = None
        self.roi = None
        plot1d = self._get_plot1d()
        if plot1d is not None:
            try:
                from anyplotlib.widgets import VLineWidget
                widget = VLineWidget(lambda: None, x=0.0, color=self.color)
                widget._push_fn = plot1d._make_widget_push_fn(widget)
                plot1d._widgets[widget.id] = widget
                plot1d._push()
                self._widget = widget
                self.roi = widget
                self._event_cb = event_handler_fn(self._on_pointer_up)
                widget.add_event_handler(self._event_cb, "pointer_move", "pointer_up")
            except Exception as e:
                logger.debug("InfiniteLineSelector widget init failed: %s", e)

    def _get_plot1d(self):
        plot = self.current_plot
        return getattr(plot, "_plot1d", None) if plot is not None else None

    def _on_pointer_up(self, event):
        self.update_data()

    def _get_selected_indices(self) -> np.ndarray:
        if self._widget is None:
            return np.array([[0]])
        scale, offset = _signal_axis(self)
        pos = float(self._widget.x)
        index = int(round((pos - offset) / scale))
        return np.array([[index]])

    def translate_pixels(self, shift_x: int) -> None:
        if self._widget is not None:
            scale, offset = _signal_axis(self)
            try:
                self._widget.x = float(self._widget.x) + shift_x * scale
            except Exception as e:
                logger.debug("translating line selector failed: %s", e)


class LinearRegionSelector(BaseSelector):
    """Range selector — wraps anyplotlib RangeWidget."""

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
            parent, children, update_function,
            live_delay=live_delay, multi_selector=multi_selector,
        )
        self._widget = None
        self.roi = None
        self._roi_trace = None          # built lazily on the first pointer event
        plot1d = self._get_plot1d()
        if plot1d is not None:
            try:
                from anyplotlib.widgets import RangeWidget
                # max_extent makes the widget itself STOP at the cap while
                # dragging. The span is in DATA units, so the real cap depends on
                # the axis scale — which usually isn't known yet here (the signal
                # attaches later). Seed it from the current scale and let
                # _clamp_extent re-derive it on every update.
                scale, _ = _signal_axis(self)
                span_cap = abs(MAX_REGION_EXTENT_PER_DIM * scale)
                widget = RangeWidget(lambda: None, x0=0.0, x1=10.0,
                                     color=self.color, max_extent=span_cap)
                widget._push_fn = plot1d._make_widget_push_fn(widget)
                plot1d._widgets[widget.id] = widget
                plot1d._push()
                self._widget = widget
                self.roi = widget
                self._event_cb = event_handler_fn(self._on_pointer_up)
                widget.add_event_handler(self._event_cb, "pointer_move", "pointer_up")
            except Exception as e:
                logger.debug("LinearRegionSelector widget init failed: %s", e)

    def _get_plot1d(self):
        plot = self.current_plot
        return getattr(plot, "_plot1d", None) if plot is not None else None

    def _on_pointer_up(self, event):
        before = self._spans()
        self._clamp_extent()
        after = self._spans()
        self._trace_geometry(before, after)
        self.update_data()

    def _spans(self):
        """Geometry as ``[(lo, hi)]`` for RoiTrace — normalised so x0>x1 (the
        widget keeps whichever orientation the drag produced) doesn't read as a
        jump on its own."""
        if self._widget is None:
            return []
        x0, x1 = float(self._widget.x0), float(self._widget.x1)
        return [(x0, x1) if x0 <= x1 else (x1, x0)]

    def _trace_geometry(self, before, after) -> None:
        """Feed the ROI-jump detector. Silent unless the geometry moved in a way
        no single pointer gesture explains — see roi_trace.py."""
        try:
            if self._roi_trace is None:
                from spyde.drawing.selectors.roi_trace import RoiTrace
                self._roi_trace = RoiTrace("1-D span")
            idx = self._get_selected_indices()
            scale, offset = _signal_axis(self)
            self._roi_trace.observe(
                before, after, n_indices=int(idx.shape[0]),
                n_unique=int(np.unique(idx).size),
                extra=f"scale={scale:.4g} offset={offset:.4g} "
                      f"cap={getattr(self._widget, 'max_extent', None)}")
        except Exception as e:
            logger.debug("roi trace (1-D) failed: %s", e)

    def _clamp_extent(self) -> None:
        """Keep the widget's own span cap in sync, then clamp as a fallback.

        The WIDGET enforces the cap during the drag (anyplotlib ``max_extent``),
        which is what makes the span physically stop instead of snapping back.
        This method's jobs are the two things the widget can't do for itself:

        1. Re-derive the cap from the CURRENT axis scale. The span is in DATA
           units, so the cap is ``MAX_REGION_EXTENT_PER_DIM * scale`` — and at
           construction time the signal often isn't attached yet (``_signal_axis``
           falls back to scale 1.0), so a cap fixed then would be wrong for any
           calibrated axis.
        2. Clamp geometry that never went through a drag (a programmatic set) —
           the widget's cap only applies to interactive drags.

        The fallback clamp anchors on the lower edge, which can move the edge the
        user is holding — that is the phantom-movement the widget-side cap exists
        to avoid, so it should now be a no-op in normal dragging."""
        if self._widget is None:
            return
        try:
            scale, _ = _signal_axis(self)
            span_cap = abs(MAX_REGION_EXTENT_PER_DIM * scale)
            # Push the live cap down to the widget (no-op if unchanged / absent).
            try:
                if getattr(self._widget, "max_extent", None) != span_cap:
                    self._widget.max_extent = span_cap
            except Exception:
                pass                    # older anyplotlib: fallback clamp below
            x0 = float(self._widget.x0)
            x1 = float(self._widget.x1)
            lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
            # Relative tolerance, NOT a bare >. A span sitting exactly on the cap
            # computes hi-lo fractionally OVER it in floating point (1.95-1.15 =
            # 0.8000000000000003 > 0.8), so a bare comparison rewrote the widget
            # on nearly every pointer_move of a capped drag. Each rewrite pushes a
            # python-sourced geometry echo back to the renderer mid-drag, which is
            # exactly the feedback loop this clamp is supposed to stay out of.
            if (hi - lo) > span_cap * (1.0 + 1e-9):
                hi = lo + span_cap
                # Preserve the widget's x0/x1 orientation when writing back.
                if x0 <= x1:
                    self._widget.x1 = hi
                else:
                    self._widget.x0 = hi
        except Exception as e:
            logger.debug("clamping region span extent failed: %s", e)

    def _get_selected_indices(self) -> np.ndarray:
        if self._widget is None:
            return np.array([[0]])
        scale, offset = _signal_axis(self)
        x0 = float(self._widget.x0)
        x1 = float(self._widget.x1)
        if x0 > x1:
            x0, x1 = x1, x0
        start = (x0 - offset) / scale
        end = (x1 - offset) / scale
        first = int(np.floor(start))
        last = int(np.ceil(end))
        # Belt-and-suspenders: cap the span length even if the widget geometry
        # wasn't clamped (e.g. a programmatic set that bypassed _on_pointer_up).
        last = min(last, first + MAX_REGION_EXTENT_PER_DIM)
        indices = np.arange(first, last).reshape(-1, 1)
        if len(indices) == 0:
            indices = np.array([[int(round(start))]])
        return indices

    def translate_pixels(self, shift_x: int) -> None:
        if self._widget is not None:
            scale, _ = _signal_axis(self)
            try:
                self._widget.x0 = float(self._widget.x0) + shift_x * scale
                self._widget.x1 = float(self._widget.x1) + shift_x * scale
            except Exception as e:
                logger.debug("translating region selector failed: %s", e)


class IntegratingSelector1D(IntegratingSelectorMixin):
    """Composite selector switching between single-index and range selection."""

    def __init__(
        self,
        parent: Union["PlotWindow", "Plot"],
        children: Union["Plot", List["Plot"]],
        update_function: Union[callable, List[callable]],
        live_delay: int = 2,
        multi_selector: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._inf_line_selector = InfiniteLineSelector(
            parent, children, update_function,
            live_delay=live_delay, multi_selector=multi_selector,
        )
        self._linear_region_selector = LinearRegionSelector(
            parent, children, update_function,
            live_delay=live_delay, multi_selector=multi_selector,
        )
        self.parent = parent
        self.children = self._inf_line_selector.children
        self.active_children = self._inf_line_selector.active_children

        # CRITICAL: point each child window's parent_selector at the COMPOSITE
        # (self), not at one of the two inner selectors. Each inner selector's
        # __init__ set children.plot_window.parent_selector = <inner>, and the
        # LinearRegionSelector (constructed second) WON — so a downstream selector
        # walking upstream_selectors() found the hidden region selector instead of
        # the active crosshair, composed the wrong index, and never tracked this
        # axis. (5-D bug: moving the time axis updated the real-space image but not
        # the DP.) The composite delegates _get_selected_indices to the ACTIVE
        # sub-selector, so resolving upstream to `self` is what makes the chain
        # see the live position. Mirrors IntegratingSSelector2D, which already
        # does this.
        for child in self.active_children:
            pw = getattr(child, "plot_window", None)
            if pw is not None:
                pw.parent_selector = self
            elif hasattr(child, "parent_selector"):
                child.parent_selector = self

        self._inf_line_selector.is_integrating = False
        self._linear_region_selector.is_integrating = True
        self.selector = self._inf_line_selector
        if self._linear_region_selector._widget is not None:
            try:
                self._linear_region_selector._widget.hide()
            except Exception as e:
                logger.debug("hiding region selector on init failed: %s", e)
        # Reflect the initial hidden region in the panel overlay state.
        try:
            plot1d = self._inf_line_selector._get_plot1d()
            if plot1d is not None:
                plot1d._push()
        except Exception as e:
            logger.debug("pushing 1-D panel overlay state failed: %s", e)

    def __getattr__(self, name):
        """Delegate undefined attributes to the active sub-selector."""
        if name in ("selector", "_inf_line_selector", "_linear_region_selector"):
            raise AttributeError(name)
        selector = self.__dict__.get("selector")
        if selector is None:
            raise AttributeError(name)
        return getattr(selector, name)

    @property
    def roi(self):
        return self.selector.roi

    def _get_selected_indices(self) -> np.ndarray:
        return self.selector._get_selected_indices()

    def delayed_update_data(self, force: bool = False, update_contrast: bool = False) -> None:
        self.selector.delayed_update_data(force=force, update_contrast=update_contrast)

    def set_integrating(self, enabled: bool) -> None:
        if enabled:
            if self._inf_line_selector._widget is not None:
                try:
                    self._inf_line_selector._widget.hide()
                except Exception as e:
                    logger.debug("hiding line selector widget failed: %s", e)
            if self._linear_region_selector._widget is not None:
                try:
                    self._linear_region_selector._widget.show()
                except Exception as e:
                    logger.debug("showing region selector widget failed: %s", e)
            self.selector = self._linear_region_selector
        else:
            if self._linear_region_selector._widget is not None:
                try:
                    self._linear_region_selector._widget.hide()
                except Exception as e:
                    logger.debug("hiding region selector widget failed: %s", e)
            if self._inf_line_selector._widget is not None:
                try:
                    self._inf_line_selector._widget.show()
                except Exception as e:
                    logger.debug("showing line selector widget failed: %s", e)
            self.selector = self._inf_line_selector
        self.is_integrating = enabled
        # Force a full panel re-push so the new widget visibility is reflected in
        # overlay_widgets (replayable + reliably repainted).
        try:
            plot1d = self._inf_line_selector._get_plot1d()
            if plot1d is not None:
                plot1d._push()
        except Exception as e:
            logger.debug("pushing 1-D panel overlay state failed: %s", e)
        self.selector.delayed_update_data(force=True)

    def hide(self) -> None:
        self._inf_line_selector.hide()
        self._linear_region_selector.hide()

    def show(self) -> None:
        self.selector.show()

    def close(self) -> None:
        self._inf_line_selector.close()
        self._linear_region_selector.close()

    def move_roi(self, key) -> None:
        if hasattr(self.selector, "translate_pixels"):
            pass

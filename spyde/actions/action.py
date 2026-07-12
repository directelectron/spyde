"""
action.py — declarative, host-agnostic action templates.

The goal: a new action is a small subclass that overrides one method, not a
pile of bespoke UI plumbing.  Because these templates only touch the signal
tree and anyplotlib (selectors / figures) — never the Electron IPC layer
directly — the same action classes run unchanged in:

  * the SpyDE Electron app (anyplotlib renders to iframes), and
  * a Jupyter notebook (anyplotlib renders as an anywidget).

Three shapes cover nearly everything:

  * :class:`TransformAction`  — ``signal + params -> new signal node``
    (rebin, FFT-of-whole, azimuthal integration, …)
  * :class:`RegionAction`     — ``signal + interactive ROI -> linked output plot``
    that recomputes as the ROI moves (virtual imaging, live FFT, line profile)
  * :class:`Action`           — the bare base, for anything custom.

Usage from a notebook::

    from spyde.actions.virtual_image import VirtualImageAction
    VirtualImageAction.for_plot(plot).run(type="disk", calculation="mean")

Usage from the toolbar dispatcher is identical — it builds an
:class:`~spyde.actions.context.ActionContext` and calls ``.run(**params)``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import hyperspy.api as hs

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spyde.actions.context import ActionContext
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.selectors import BaseSelector


class Action:
    """Base template. Subclasses declare ``name`` / ``parameters`` and override
    :meth:`run`.  ``parameters`` uses the same dict spec as the YAML config, so
    a host (Electron panel or an ipywidgets form) can render an input form."""

    name: str = ""
    parameters: dict[str, dict] = {}

    def __init__(self, ctx: "ActionContext"):
        self.ctx = ctx

    # ── Convenience constructors ───────────────────────────────────────────────

    @classmethod
    def for_plot(cls, plot: "Plot", **params: Any) -> "Action":
        """Build an action directly from a Plot (notebook-friendly)."""
        from spyde.actions.context import ActionContext
        return cls(ActionContext(plot=plot, params=params, action_name=cls.name))

    # ── Shared accessors ───────────────────────────────────────────────────────

    @property
    def plot(self) -> "Plot":
        return self.ctx.plot

    @property
    def signal(self):
        return self.plot.plot_state.current_signal

    @property
    def signal_tree(self):
        return self.plot.signal_tree

    @property
    def session(self):
        return self.plot.session

    def _resolved_params(self, overrides: dict) -> dict:
        """Merge declared defaults < ctx.params < explicit call kwargs."""
        merged: dict[str, Any] = {}
        for key, spec in self.parameters.items():
            if isinstance(spec, dict) and "default" in spec:
                merged[key] = spec["default"]
        merged.update(self.ctx.params or {})
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return merged

    # ── Entry point ────────────────────────────────────────────────────────────

    def run(self, **params: Any):
        raise NotImplementedError("Subclasses must implement run().")


class TransformAction(Action):
    """``signal + params -> new signal node``.

    Override :meth:`build_kwargs` to translate UI params into the kwargs of a
    HyperSpy ``method`` (or a free ``function``).  The new node is added to the
    signal tree and a plot is created automatically by ``add_transformation``.
    """

    method: str | None = None          # HyperSpy method name on the signal
    function: Any = None               # or function(signal, **kwargs) -> signal
    node_name: str | None = None

    def build_kwargs(self, signal, **params) -> dict:
        """Default: pass resolved params straight through. Override to adapt."""
        return params

    def run(self, **params):
        resolved = self._resolved_params(params)
        kwargs = self.build_kwargs(self.signal, **resolved)
        new = self.signal_tree.add_transformation(
            parent_signal=self.signal,
            method=self.method,
            function=self.function,
            node_name=self.node_name or self.name or None,
            **kwargs,
        )
        # Switch the display to the new node and force a navigator re-slice —
        # add_transformation only registers the PlotState, so without this the
        # plot keeps showing the OLD data until the crosshair moves (the
        # "Rebin didn't do anything" bug). Also refreshes the Workflow panel.
        if new is not None:
            from spyde.actions.lifecycle import show_tree_node
            show_tree_node(self.plot, self.signal_tree, new)
        return new


class RegionAction(Action):
    """``signal + interactive ROI -> linked output plot``.

    Adds an anyplotlib selector to the source plot and a new output plot that
    recomputes via :meth:`reduce` whenever the ROI moves.  This is the live
    virtual-imaging / FFT / line-profile shape.
    """

    #: Selector class to place on the source plot. Override or set via
    #: ``selector_for_params`` for param-dependent selector choice.
    selector_type: Any = None
    #: Dimensionality of the output plot (2 = image, 1 = line).
    output_dims: int = 2
    output_node_name: str = "Region"
    #: Y-axis label for a 1-D output plot (``output_dims == 1``). Stamped onto the
    #: output signal's ``metadata.Signal.quantity`` so the line plot draws it —
    #: see ``Plot._axes_info_1d``. ``None`` falls back to the generic "Intensity".
    output_y_label: str | None = None
    #: Optional ROI colour (e.g. multi virtual-image "red"/"green"/…).
    roi_color: str | None = None

    def selector_for_params(self, **params):
        """Return the selector class to use given the resolved params.
        Default: ``self.selector_type`` (or RectangleSelector)."""
        if self.selector_type is not None:
            return self.selector_type
        from spyde.drawing.selectors import RectangleSelector
        return RectangleSelector

    def placeholder_signal(self):
        """Signal shown in the output plot before the first reduce."""
        if self.output_dims == 1:
            sig = hs.signals.Signal1D(np.zeros(10))
            # Stamp the y-axis label so the 1-D line plot shows it (read from
            # metadata.Signal.quantity by Plot._axes_info_1d). Default when the
            # action doesn't declare one is handled downstream ("Intensity").
            if self.output_y_label:
                try:
                    sig.metadata.set_item("Signal.quantity", self.output_y_label)
                except Exception:
                    pass
            return sig
        return hs.signals.Signal2D(np.zeros((10, 10)))

    def reduce(self, signal, selector: "BaseSelector", indices, **params):
        """Compute the output data for the current ROI. Must override."""
        raise NotImplementedError("RegionAction subclasses must implement reduce().")

    def reduce_to(self, signal, selector: "BaseSelector", child, indices, **params):
        """Compute output for ``child`` and return the data the selector should
        push. Default delegates to :meth:`reduce` (synchronous). Override to
        drive ``child`` directly — e.g. progressive chunked streaming — and
        return a blank/initial frame to display while the stream fills in."""
        return self.reduce(signal, selector, indices, **params)

    def run(self, **params) -> "BaseSelector":
        resolved = self._resolved_params(params)
        # Live params: the selector reads these on every recompute, so a caret
        # edit (update_live_params) takes effect without rebuilding the selector.
        self._live_params = dict(resolved)

        out_window = self.session.add_plot_window(
            is_navigator=False,
            signal_tree=self.signal_tree,
        )
        out_window.owner_plot_window = self.plot.plot_window
        out_plot = out_window.add_new_plot()
        out_plot.add_plot_state(
            signal=self.placeholder_signal(),
            dimensions=self.output_dims,
            dynamic=True,
        )
        # Create the output figure NOW (push the placeholder) so its iframe
        # starts loading immediately. Otherwise — for LAZY data, where the result
        # is pushed later from the PlotUpdateWorker thread — that single push
        # races the iframe load and is lost, leaving the output stuck on the
        # placeholder ("the virtual image is just black"). The navigator avoids
        # this only because it pushes repeatedly.
        try:
            ph = np.asarray(self.placeholder_signal().data, dtype=np.float32)
            out_plot.set_data(ph)
        except Exception as e:
            log.debug("painting action output placeholder failed: %s", e)

        # Tag the output window so the renderer can label/colour it.
        try:
            out_window.vi_color = self.roi_color
        except Exception as e:
            log.debug("tagging output window colour failed: %s", e)
        self._out_plot = out_plot
        selector = self._make_selector(out_plot)
        self._selector = selector
        # Trigger an initial compute so the output isn't blank.
        try:
            selector.delayed_update_data(force=True)
        except Exception as e:
            log.debug("initial action compute failed: %s", e)
        return selector

    def _make_selector(self, out_plot):
        """Create the ROI selector for the current live params (selector class is
        param-dependent, so this is also how a detector-type change rebuilds it)."""
        selector_cls = self.selector_for_params(**self._live_params)

        # The update hook reads the CURRENT live params each recompute, so caret
        # edits (calculation, etc.) take effect live without a rebuild.
        def _update_fn(selector, child, indices):
            return self.reduce_to(self.signal, selector, child, indices, **self._live_params)

        sel_kwargs = {}
        if self.roi_color:
            sel_kwargs["color"] = self.roi_color
        return selector_cls(
            parent=self.plot,
            children=out_plot,
            update_function=_update_fn,
            multi_selector=False,
            **sel_kwargs,
        )

    def update_live_params(self, params: dict) -> None:
        """Apply new params (from a per-output caret edit) and recompute live.

        If the param change alters which *selector class* is used (e.g. detector
        type disk↔annular↔rectangle), the on-plot ROI is REBUILT in the new shape;
        otherwise (calculation, etc.) it just recomputes with the existing ROI.
        """
        params = params or {}
        old_cls = self.selector_for_params(**self._live_params)
        self._live_params = {**getattr(self, "_live_params", {}), **params}
        new_cls = self.selector_for_params(**self._live_params)

        sel = getattr(self, "_selector", None)
        if new_cls is not old_cls and getattr(self, "_out_plot", None) is not None:
            # Detector shape changed → swap the ROI on the source plot.
            if sel is not None:
                try:
                    sel.close()   # removes the old ROI (hide + panel repaint)
                except Exception as e:
                    log.debug("closing old ROI on detector-shape change failed: %s", e)
            self._selector = self._make_selector(self._out_plot)
            try:
                self._selector.delayed_update_data(force=True, update_contrast=True)
            except Exception as e:
                log.debug("recompute after ROI swap failed: %s", e)
        elif sel is not None:
            try:
                sel.delayed_update_data(force=True, update_contrast=True)
            except Exception as e:
                log.debug("recompute after param change failed: %s", e)

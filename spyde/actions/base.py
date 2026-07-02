"""
base.py — core toolbar actions (Electron architecture).

Action functions receive an :class:`~spyde.actions.context.ActionContext`
(historically named ``toolbar`` in the call sites) and operate on the new
Plot / PlotWindow / Session objects.  UI that used to be Qt CaretGroups /
button trees is now emitted to Electron as IPC messages; the frontend renders
the controls and sends back parameter values.
"""
from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

log = logging.getLogger(__name__)
import hyperspy.api as hs

from spyde.actions.action import TransformAction
from spyde.drawing.update_functions import get_fft
from spyde.drawing.selectors import RectangleSelector

if TYPE_CHECKING:
    from spyde.actions.context import ActionContext

ZOOM_STEP = 0.8
NAVIGATOR_DRAG_MIME = "application/x-spyde-navigator"


def _emit(obj: dict) -> None:
    try:
        from spyde.backend.ipc import emit
        emit(obj)
    except Exception as e:
        log.debug("IPC emit of %r failed: %s", obj.get("type"), e)


# ── View actions ────────────────────────────────────────────────────────────

def zoom_in(toolbar: "ActionContext", *args, **kwargs):
    """Zoom the plot in (handled by anyplotlib's view via IPC)."""
    _emit({"type": "plot_view", "window_id": toolbar.plot.window_id,
           "command": "zoom", "factor": ZOOM_STEP})


def zoom_out(toolbar: "ActionContext", *args, **kwargs):
    """Zoom the plot out."""
    _emit({"type": "plot_view", "window_id": toolbar.plot.window_id,
           "command": "zoom", "factor": 1.0 / ZOOM_STEP})


def reset_view(toolbar: "ActionContext", *args, **kwargs):
    """Reset the plot view to auto-range."""
    _emit({"type": "plot_view", "window_id": toolbar.plot.window_id,
           "command": "reset"})


# ── Selector actions ────────────────────────────────────────────────────────

def add_selector(toolbar: "ActionContext", toggled=None, *args, **kwargs):
    """Add a navigation selector + linked signal plot."""
    mgr = toolbar.plot.multiplot_manager
    if mgr is not None:
        mgr.add_navigation_selector_and_signal_plot(toolbar.plot_window)


def add_fft_selector(toolbar: "ActionContext", action_name="", *args, **kwargs):
    """Add an FFT selector: a RectangleSelector on the parent that computes the
    FFT of the selected region into a new plot window."""
    widgets = toolbar.action_widgets
    if (action_name in widgets
            and "plot_windows" in widgets[action_name]
            and "FFT_Plot_Window" in widgets[action_name]["plot_windows"]):
        return  # already initialised

    plot = toolbar.plot
    session = plot.session
    signal_tree = plot.signal_tree

    plot_window = session.add_plot_window(
        is_navigator=False,
        signal_tree=signal_tree,
    )
    plot_window.owner_plot_window = plot.plot_window

    fft_plot = plot_window.add_new_plot()
    place_holder_signal = hs.signals.Signal2D(data=np.zeros((10, 10)))

    selector = RectangleSelector(
        parent=plot,
        children=fft_plot,
        multi_selector=False,
        update_function=get_fft,
    )

    fft_plot.add_plot_state(
        signal=place_holder_signal,
        dimensions=2,
        dynamic=True,
    )
    toolbar.register_action_plot_item(
        action_name=action_name, item=selector.roi, key="RectangleSelector_FFT"
    )
    toolbar.register_action_plot_window(
        action_name=action_name, plot_window=plot_window, key="FFT_Plot_Window"
    )


# ── Toggle / navigation actions (UI emitted to Electron) ────────────────────

def toggle_navigation_plots(toolbar: "ActionContext", action_name="", toggle=None,
                            *args, **kwargs):
    """Emit the available navigation signals so Electron can render a switcher."""
    mgr = toolbar.plot.multiplot_manager
    if mgr is None:
        raise RuntimeError("Plot does not have a navigation plot manager.")

    signal_options = mgr.navigation_signals
    options = []
    for name, signal in signal_options.items():
        options.append({"name": name})

    _emit({
        "type": "navigation_options",
        "window_id": toolbar.plot.window_id,
        "action_name": action_name,
        "options": options,
        "visible": bool(toggle) if toggle is not None else True,
    })


def toggle_signal_tree(toolbar: "ActionContext", action_name="", toggle=None,
                       *args, **kwargs):
    """Emit the signal tree structure so Electron can render a node switcher."""
    root_node = toolbar.plot.signal_tree.root_node

    def node_to_dict(node) -> dict:
        return {
            "name": node.name,
            "signal_id": id(node.signal),
            "children": [node_to_dict(c) for c in node.children.values()],
        }

    active = None
    try:
        active = id(toolbar.plot.plot_state.current_signal)
    except Exception as e:
        log.debug("resolving active signal id failed: %s", e)
    _emit({
        "type": "signal_tree",
        "window_id": toolbar.plot.window_id,
        "action_name": action_name,
        "tree": node_to_dict(root_node),
        "active_signal_id": active,
        "visible": bool(toggle) if toggle is not None else True,
    })


def select_signal_node(toolbar: "ActionContext", signal_id: int = None, *args, **kwargs):
    """Switch the active plot to the signal node identified by signal_id.

    Called when the user picks a node in the Electron signal-tree switcher.
    """
    if signal_id is None:
        signal_id = toolbar.params.get("signal_id")
    if signal_id is None:
        return
    for sig in toolbar.plot.plot_states.keys():
        if id(sig) == signal_id:
            toolbar.plot.set_plot_state(sig)
            return


# ── Rebin ────────────────────────────────────────────────────────────────────

class Rebin2DAction(TransformAction):
    """Rebin the 2-D signal by (scale_x, scale_y) — a TransformAction: the
    template resolves the params, runs hyperspy ``rebin`` and adds the
    "Binned" node (+ PlotState) to the SAME tree automatically."""

    name = "Rebin"
    method = "rebin"
    node_name = "Binned"
    parameters = {
        "scale_x": {"default": 2},
        "scale_y": {"default": 2},
    }

    def build_kwargs(self, signal, scale_x=2, scale_y=2, **_):
        if signal.axes_manager.signal_dimension != 2:
            raise RuntimeError("Current signal is not 2D, cannot rebin2d.")
        nav = signal.axes_manager.navigation_dimension
        return {"scale": [1] * nav + [int(scale_x), int(scale_y)]}

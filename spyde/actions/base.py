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


# ── Movie playback (on the 1-D time navigator) ───────────────────────────────

def _session_of(toolbar: "ActionContext"):
    return getattr(getattr(toolbar, "plot", None), "session", None)


def play_pause(toolbar: "ActionContext", toggled=None, *args, **kwargs):
    """Toggle movie playback: start the frame clock (or pause it). A toggle
    action — ``toggled`` is the requested on/off state from the renderer."""
    session = _session_of(toolbar)
    if session is None:
        return
    pb = session.playback
    if toggled is None:
        pb.toggle(**{k: v for k, v in kwargs.items() if k in ("fps", "step", "loop")})
    elif toggled:
        pb.play(**{k: v for k, v in kwargs.items() if k in ("fps", "step", "loop")})
    else:
        pb.pause()


def fast_forward(toolbar: "ActionContext", toggled=None, *args, **kwargs):
    """Toggle fast playback — a larger frame step (default 5x)."""
    session = _session_of(toolbar)
    if session is None:
        return
    step = kwargs.get("step", 5)
    fps = kwargs.get("fps")
    pb = session.playback
    if toggled is False:
        pb.pause()
    else:
        pb.toggle(fps=fps, step=int(step) if step else 5)


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
# (The old "Select Navigator" / "Navigate Signal Tree" toolbar toggles are
# gone: navigators are switched via the chip strip on the navigator window,
# and the Workflow tree is always shown in the right-hand dock — the session
# pushes `signal_tree` messages on tree creation and after every transform.)

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


# ── Crop ─────────────────────────────────────────────────────────────────────

def _crop_signal(signal, x0=0, x1=0, y0=0, y1=0, t0=0, t1=0, **_):
    """Crop a 2-D-signal dataset to a spatial (image) box and, for a movie /
    navigated dataset, an optional leading-nav (time/first-nav-axis) range.

    All slicing is by PIXEL INDEX via hyperspy ``isig`` / ``inav`` — a lazy dask
    view (a graph op, no materialise), so a huge in-situ movie is trimmed to a
    smaller lazy movie with no data read (memory-safety rule respected). Empty /
    zero ranges mean "keep the full extent" on that axis, so a pure spatial crop
    leaves the nav axis whole and vice-versa.

    ``x0:x1`` / ``y0:y1`` are signal-axis (image column / row) pixel bounds;
    ``t0:t1`` is the FIRST navigation axis in DISPLAY order — a movie's time axis
    (nav-dim 1) or, on a 4-D scan, the fast (x) scan axis. An ``end`` of 0 (the
    default) means "keep the full extent" on that axis, so a pure spatial crop
    leaves the nav axis whole; if every bound is 0 the signal is returned
    UNCHANGED (no redundant node).
    """
    am = signal.axes_manager
    sig_shape = tuple(int(s) for s in am.signal_shape)   # (x, y) display order
    nav_shape = tuple(int(s) for s in am.navigation_shape)

    def _bounds(lo, hi, n):
        # An `end` of 0 (or out of range) means "to the end". An inverted /
        # degenerate box is clamped to a >=1-px slice rather than raising.
        lo = int(lo or 0)
        hi = int(hi or 0)
        if hi <= 0 or hi > n:
            hi = n
        lo = max(0, min(lo, n - 1))
        hi = max(lo + 1, min(hi, n))
        return lo, hi

    want_spatial = any(int(v or 0) for v in (x0, x1, y0, y1))
    want_time = bool(int(t0 or 0) or int(t1 or 0))
    if not want_spatial and not want_time:
        return signal          # all-zero crop → no-op, don't add a redundant node

    out = signal
    if am.signal_dimension >= 2 and want_spatial:
        sx0, sx1 = _bounds(x0, x1, sig_shape[0])
        sy0, sy1 = _bounds(y0, y1, sig_shape[1])
        # isig indexes signal axes in display (x, y) order → X=columns, Y=rows.
        out = out.isig[sx0:sx1, sy0:sy1]
    if am.navigation_dimension >= 1 and want_time:
        nt0, nt1 = _bounds(t0, t1, nav_shape[0])
        # inav indexes the FIRST navigation axis (display order): a movie's time
        # axis, or a 4-D scan's fast (x) axis.
        out = out.inav[nt0:nt1]
    return out


class CropAction(TransformAction):
    """Crop the dataset to a spatial (image) box + optional time range — a
    TransformAction that adds a lazy "Cropped" node to the SAME tree. Nothing is
    materialised (isig/inav are dask-view slices), so a multi-GB movie crops for
    free. Zero ranges keep the full extent on that axis."""

    name = "Crop"
    function = staticmethod(_crop_signal)
    node_name = "Cropped"
    parameters = {
        "x0": {"default": 0},
        "x1": {"default": 0},
        "y0": {"default": 0},
        "y1": {"default": 0},
        "t0": {"default": 0},
        "t1": {"default": 0},
    }

    def build_kwargs(self, signal, x0=0, x1=0, y0=0, y1=0, t0=0, t1=0, **_):
        return {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "t0": t0, "t1": t1}

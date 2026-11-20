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
        from de_shell.ipc import emit
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


def _bind_playback_tree(toolbar: "ActionContext", pb) -> None:
    """Bias the shared clock toward the tree of the plot whose button was clicked,
    so the RIGHT movie plays when several are open (falls back to a full scan)."""
    tree = getattr(getattr(toolbar, "plot", None), "signal_tree", None)
    if tree is not None:
        pb.set_preferred_tree(tree)


def play_pause(toolbar: "ActionContext", toggled=None, *args, **kwargs):
    """Toggle real-time movie playback: start the wall-clock frame clock (or pause
    it). A plain on/off toggle — pressing Play while fast-forwarding PAUSES (Fast
    Forward owns the speed cycle). Real-time pacing is derived from the time axis'
    scale/units. Playback ALWAYS loops (wraps to frame 0 at the end)."""
    session = _session_of(toolbar)
    if session is None:
        return
    pb = session.playback
    _bind_playback_tree(toolbar, pb)
    if toggled is None:
        pb.toggle(loop=True)
    elif toggled:
        pb.play(loop=True)
    else:
        pb.pause()


def fast_forward(toolbar: "ActionContext", toggled=None, *args, **kwargs):
    """Fast-forward = speed multiplier. Cycles 2x → 4x → 8x → back to 1x. Pressed
    while stopped, starts playback at 2x; while playing, bumps the speed one notch
    (staying at 1x after 8x)."""
    session = _session_of(toolbar)
    if session is None:
        return
    pb = session.playback
    _bind_playback_tree(toolbar, pb)
    if toggled is False:
        pb.pause()
    else:
        pb.fast_forward(loop=True)


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
    """Rebin the scan by (scan_x, scan_y) and the diffraction pattern by
    (scale_x, scale_y) — a TransformAction: the template resolves the params,
    runs hyperspy ``rebin`` and adds the "Binned" node (+ PlotState) to the
    SAME tree automatically.

    The parameters are declared scan first, as a 4-D dataset is described;
    the toolbar YAML lists them in the same order, which is the order the
    caret shows them in."""

    name = "Rebin"
    method = "rebin"
    node_name = "Binned"
    # Frame N of the binned output is a deterministic downsample of frame N of
    # the input only — a bounded, local per-frame transform. Binning the SCAN
    # is not: an output position is built from several input ones, so it is
    # left off that promise.
    is_local_per_frame = True
    parameters = {
        "scan_x": {"default": 1},
        "scan_y": {"default": 1},
        "scale_x": {"default": 2},
        "scale_y": {"default": 2},
    }

    def build_kwargs(self, signal, scale_x=2, scale_y=2, scan_x=1, scan_y=1,
                     **_):
        """Bin the detector, the scan, or both.

        The scan was not binnable at all before, which left no way to shrink a
        4-D dataset in the direction that usually dominates its size — and no
        way to save a reduced one out of the app.

        ``scale`` is in the axes manager's own order, which is display order:
        navigation axes first, fastest first, then the signal axes. For a 4-D
        scan that is (scan x, scan y, kx, ky), and an array shaped
        (scan y, scan x, ky, kx).
        """
        manager = signal.axes_manager
        if manager.signal_dimension != 2:
            raise RuntimeError("Current signal is not 2D, cannot rebin2d.")
        navigation = int(manager.navigation_dimension)
        scan = [1] * navigation
        if navigation >= 1:
            scan[0] = max(1, int(scan_x))
        if navigation >= 2:
            scan[1] = max(1, int(scan_y))
        # hyperspy rebins by SUMMING, and every axis divides exactly or it
        # raises. Refusing here says which axis and by how much, where the
        # library's message names neither.
        factors = scan + [int(scale_x), int(scale_y)]
        # Labels from the axes, not a list of four: on a 5-D stack a fixed
        # list ran out one axis early, so the last detector axis was never
        # checked and the other two were named wrongly.
        labels = ([f"scan {_axis_label(axis, index)}"
                   for index, axis in enumerate(manager.navigation_axes)]
                  + [f"detector {_axis_label(axis, index)}"
                     for index, axis in enumerate(manager.signal_axes)])
        for size, factor, label in zip(
                list(manager.navigation_shape) + list(manager.signal_shape),
                factors, labels):
            if factor > 1 and int(size) % factor:
                raise RuntimeError(
                    f"{label} is {int(size)} px, which {factor} does not "
                    f"divide — crop it to a multiple of {factor} first")
        return {"scale": factors,
                "dtype": _rebin_dtype(signal.data.dtype, factors)}

    def run(self, **params):
        # Taken before the run: it switches the window to the new node, and
        # `self.signal` follows the window.
        source = self.signal
        new = super().run(**params)
        if new is not None:
            _record_binning(new, source)
        return new


def _axis_label(axis, index: int) -> str:
    """An axis's name, or its position when hyperspy has none for it."""
    name = str(getattr(axis, "name", "") or "")
    if name and name != "<undefined>":
        return name
    return ("x", "y")[index] if index < 2 else f"axis {index}"


def _record_binning(binned, source) -> None:
    """Note on a multi-angle stack how far it has been reduced.

    Its recorded member offsets are in the pixels of the composition and are
    not rescaled — they would stop being whole pixels — so the record says
    what they are now measured against. Cumulative across repeated rebins.
    """
    from spyde.signals.multiangle import MULTIANGLE_METADATA

    metadata = binned.metadata
    if not metadata.has_item(MULTIANGLE_METADATA):
        return
    navigation = binned.axes_manager.navigation_dimension
    before, after = source.data.shape, binned.data.shape
    factors = [max(1, int(b) // max(1, int(a))) for b, a in zip(before, after)]
    scan = factors[navigation - 2:navigation] if navigation >= 2 else [1, 1]
    detector = factors[navigation:navigation + 2]
    key = f"{MULTIANGLE_METADATA}.binned_by"
    previous = metadata.get_item(key, None)
    previous = (previous.as_dictionary() if hasattr(previous, "as_dictionary")
                else previous) or {"scan": [1, 1], "detector": [1, 1]}
    metadata.set_item(key, {
        "scan": [int(p) * int(f) for p, f in zip(previous["scan"], scan)],
        "detector": [int(p) * int(f)
                     for p, f in zip(previous["detector"], detector)]})


def _rebin_dtype(source_dtype, factors):
    """The narrowest dtype that holds a sum of this many source pixels.

    Left to itself hyperspy sums into numpy's default accumulator, which turns
    a uint16 scan binned 2x2x2x2 into uint64 — four times the bytes that were
    asked for, and a dtype nothing downstream can widen for a further sum.
    Here the sum of 16 uint16 pixels lands in uint32, which is exact and half
    the size. A float stays a float; a dtype with no notion of a sum (bool,
    strings) is left to hyperspy's default.
    """
    from spyde.multiangle.compose import sum_dtype

    count = 1
    for factor in factors:
        count *= max(1, int(factor))
    try:
        return sum_dtype(source_dtype, count)
    except (TypeError, ValueError):
        return None


# ── Crop ─────────────────────────────────────────────────────────────────────

def _crop_signal(signal, x0=0, x1=0, y0=0, y1=0, t0=0, t1=0,
                 scan_x0=0, scan_x1=0, scan_y0=0, scan_y1=0, **_):
    """:func:`spyde.actions.crop.crop_signal` with the Crop fields as keywords,
    for scripts and the in-tree :class:`CropAction`.

    ``x0:x1`` / ``y0:y1`` cut the image or detector, ``scan_x0:scan_x1`` /
    ``scan_y0:scan_y1`` the scan, and ``t0:t1`` the time axis — a movie's
    only navigation axis, the slowest of a time series of scans (on a 4-D
    scan, which has none, the first scan axis). An ``end`` of 0 keeps the axis
    to its end, so a pure detector crop leaves the scan whole; when every
    bound is 0 the signal is returned unchanged.
    """
    from spyde.actions.crop import crop_signal, time_axis_index

    am = signal.axes_manager

    def wanted(*bounds):
        return any(int(bound or 0) for bound in bounds)

    def bounds(low, high, size):
        high = int(high or 0)
        return int(low or 0), (size if high <= 0 or high > size else high)

    ranges = {}
    if am.signal_dimension >= 2 and wanted(x0, x1, y0, y1):
        width, height = (int(axis.size) for axis in am.signal_axes[:2])
        ranges["sig0"] = bounds(x0, x1, width)
        ranges["sig1"] = bounds(y0, y1, height)
    out = signal
    if am.navigation_dimension >= 1 and wanted(t0, t1):
        index = time_axis_index(signal)
        index = 0 if index is None else index
        out = crop_signal(out, {f"nav{index}": bounds(
            t0, t1, int(am.navigation_axes[index].size))})
    if ranges:
        out = crop_signal(out, ranges)
    if am.navigation_dimension >= 1 and wanted(scan_x0, scan_x1, scan_y0, scan_y1):
        # Against the axes as the time cut left them: on a movie the "scan"
        # and the time axis are the same one.
        axes = out.axes_manager.navigation_axes
        scan = {"nav0": bounds(scan_x0, scan_x1, int(axes[0].size))}
        if am.navigation_dimension >= 2:
            scan["nav1"] = bounds(scan_y0, scan_y1, int(axes[1].size))
        out = crop_signal(out, scan)
    return out


class CropAction(TransformAction):
    """Crop to a box on any axis as a lazy "Cropped" node of the SAME tree —
    for scripts. The app's Crop panel (``spyde.actions.crop``) opens the
    result as a new dataset instead, so its navigators match the crop.
    Nothing is materialised: ``isig`` / ``inav`` are dask-view slices. Zero
    ranges keep the full extent on that axis."""

    name = "Crop"
    function = staticmethod(_crop_signal)
    node_name = "Cropped"
    # Pure isig/inav dask-view slicing: frame N of the output is a spatial crop
    # of a single (possibly index-shifted, for the t0/t1 nav crop) input frame —
    # local either way.
    is_local_per_frame = True
    parameters = {
        "x0": {"default": 0},
        "x1": {"default": 0},
        "y0": {"default": 0},
        "y1": {"default": 0},
        "t0": {"default": 0},
        "t1": {"default": 0},
        # Listed for the DEFAULTS. What actually carries a field through is
        # `build_kwargs` naming it in the dict it returns: `_resolved_params`
        # merges `ctx.params` wholesale, so the toolbar's values arrive here
        # either way, and then a key the returned dict omits is dropped. That
        # omission is what made a scan crop run, succeed, and return the whole
        # dataset.
        "scan_x0": {"default": 0},
        "scan_x1": {"default": 0},
        "scan_y0": {"default": 0},
        "scan_y1": {"default": 0},
    }

    def build_kwargs(self, signal, x0=0, x1=0, y0=0, y1=0, t0=0, t1=0,
                     scan_x0=0, scan_x1=0, scan_y0=0, scan_y1=0, **_):
        return {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "t0": t0, "t1": t1,
                "scan_x0": scan_x0, "scan_x1": scan_x1,
                "scan_y0": scan_y0, "scan_y1": scan_y1}

    def run(self, **params):
        # A no-op crop returns the source itself; add_transformation has no
        # identity check and would hang a redundant "Cropped" node on it.
        resolved = self._resolved_params(params)
        kwargs = self.build_kwargs(self.signal, **resolved)
        if not any(int(v or 0) for v in kwargs.values()):
            return self.signal
        return super().run(**params)

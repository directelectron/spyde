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
    """Rebin the 2-D signal by (scale_x, scale_y) — a TransformAction: the
    template resolves the params, runs hyperspy ``rebin`` and adds the
    "Binned" node (+ PlotState) to the SAME tree automatically."""

    name = "Rebin"
    method = "rebin"
    node_name = "Binned"
    # Frame N of the binned output is a deterministic downsample of frame N of
    # the input only — a bounded, local per-frame transform.
    is_local_per_frame = True
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


_CROP_COLOR = "#f9e2af"


def _crop_widget_bounds(tree, w: int, h: int) -> "tuple[int, int, int, int] | None":
    """Read the live crop-rectangle widget's bounds as signal-axis pixel indices
    (x0, x1, y0, y1), clamped to the [0, w] x [0, h] frame — a resize that drags
    a handle past the detector edge, or leaves the box partially off-frame, is
    clamped rather than producing an inverted/out-of-range crop. Returns
    ``None`` when there is no widget (typed-field-only flow)."""
    widget = getattr(tree, "_crop_widget", None) if tree is not None else None
    if widget is None:
        return None
    x0 = max(0, min(int(round(float(widget.x))), w))
    y0 = max(0, min(int(round(float(widget.y))), h))
    x1 = max(0, min(int(round(float(widget.x) + float(widget.w))), w))
    y1 = max(0, min(int(round(float(widget.y) + float(widget.h))), h))
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return x0, x1, y0, y1


def _crop_clamp_widget(widget, w: int, h: int) -> None:
    """Clamp the crop widget's geometry to stay inside the [0,w] x [0,h] frame
    (handle-drag past the edge, or a whole-box drag off-frame, is pinned back
    in rather than allowed to produce an inverted/out-of-range crop). Callers
    invoking this from a widget event handler MUST hold the ``_crop_clamping``
    re-entrancy guard (see ``_on_drag`` in crop_open) — the ``set()`` here
    fires ``pointer_move`` unconditionally and would recurse the handler."""
    if widget is None:
        return
    try:
        x = max(0.0, min(float(widget.x), float(w)))
        y = max(0.0, min(float(widget.y), float(h)))
        ww = max(1.0, min(float(widget.w), float(w) - x))
        hh = max(1.0, min(float(widget.h), float(h) - y))
        widget.set(x=x, y=y, w=ww, h=hh)
    except Exception as e:
        log.debug("crop widget clamp failed: %s", e)


def crop_open(session, plot, payload) -> None:
    """Activating Crop shows a draggable/resizable rectangle widget on the
    source plot outlining the spatial crop box (image pixel coords). Starts
    covering the FULL frame (a no-op box) unless the caret already has typed
    field values, so the user drags handles in from the full extent rather
    than starting from a to-be-guessed default."""
    from spyde.actions.context import src_plot_tree as _src_plot_tree, current_signal as _current_signal
    from spyde.backend.ipc import emit_error

    src, tree = _src_plot_tree(session, plot)
    signal = _current_signal(src)
    plot2d = getattr(src, "_plot2d", None) if src is not None else None
    if plot2d is None or signal is None or tree is None:
        emit_error("Crop: no active 2-D plot to place the crop box on")
        return
    am = signal.axes_manager
    if am.signal_dimension < 2:
        emit_error("Crop: current signal is not 2-D")
        return
    sig_ax = am.signal_axes
    w, h = int(sig_ax[0].size), int(sig_ax[1].size)

    x0 = int(payload.get("x0", 0) or 0)
    x1 = int(payload.get("x1", 0) or 0) or w
    y0 = int(payload.get("y0", 0) or 0)
    y1 = int(payload.get("y1", 0) or 0) or h
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 <= x0:
        x0, x1 = 0, w
    if y1 <= y0:
        y0, y1 = 0, h

    crop_close(session, plot, payload)   # replace any prior widget
    try:
        widget = plot2d.add_rectangle_widget(
            x=float(x0), y=float(y0), w=float(x1 - x0), h=float(y1 - y0),
            color=_CROP_COLOR, show_handles=True,
        )
        tree._crop_widget = widget

        def _on_drag(event, _tree=tree, _w=w, _h=h):
            # RE-ENTRANCY GUARD: anyplotlib Widget.set() fires pointer_move
            # UNCONDITIONALLY (even when nothing changed, regardless of
            # _push), so the clamp's set() re-invokes this handler
            # synchronously — unguarded, ONE JS drag frame recursed ~2000
            # deep before RecursionError. A hard per-tree flag breaks the
            # cycle; compare-before-set is NOT sufficient (set() fires on a
            # no-change write too).
            if getattr(_tree, "_crop_clamping", False):
                return
            _tree._crop_clamping = True
            try:
                _crop_clamp_widget(getattr(_tree, "_crop_widget", None), _w, _h)
            finally:
                _tree._crop_clamping = False

        from spyde.drawing.selectors.base_selector import event_handler_fn
        handler = event_handler_fn(_on_drag)
        widget.add_event_handler(handler, "pointer_move", "pointer_up")
        tree._crop_widget_handler = handler   # keep a ref alive (weak callback)
    except Exception as e:
        log.debug("crop widget add failed: %s", e)


def _crop_remove_widget(tree) -> None:
    """Hide + drop the crop-box widget on *tree* (harmless when absent).
    Shared by crop_close, the node-switch teardown in
    ``lifecycle.show_tree_node``, and ``BaseSignalTree.close()``. Widgets have
    no ``remove()``, only ``hide()`` (see the CZB precedent in
    center_zero_beam.py)."""
    widget = getattr(tree, "_crop_widget", None) if tree is not None else None
    if widget is not None:
        try:
            widget.hide()
        except Exception as e:
            log.debug("hiding crop widget failed: %s", e)
        tree._crop_widget = None
        tree._crop_widget_handler = None


def crop_close(session, plot, payload=None) -> None:
    """Caret closed / action deselected / window closed → remove the crop box."""
    from spyde.actions.context import src_plot_tree as _src_plot_tree
    _src, tree = _src_plot_tree(session, plot)
    _crop_remove_widget(tree)


def crop_set_region(session, plot, payload) -> None:
    """A typed-field edit (x0/x1/y0/y1) moves the live widget to match, so the
    numeric fields and the on-plot box stay in sync in BOTH directions (drag →
    field sync happens client-side by reading the widget's own pointer_move /
    pointer_up events — see CropWizard.tsx)."""
    from spyde.actions.context import src_plot_tree as _src_plot_tree, current_signal as _current_signal
    src, tree = _src_plot_tree(session, plot)
    signal = _current_signal(src)
    widget = getattr(tree, "_crop_widget", None) if tree is not None else None
    if widget is None or signal is None:
        return
    am = signal.axes_manager
    if am.signal_dimension < 2:
        return
    sig_ax = am.signal_axes
    w, h = int(sig_ax[0].size), int(sig_ax[1].size)
    x0 = max(0, min(int(payload.get("x0", 0) or 0), w))
    x1 = max(0, min(int(payload.get("x1", 0) or 0) or w, w))
    y0 = max(0, min(int(payload.get("y0", 0) or 0), h))
    y1 = max(0, min(int(payload.get("y1", 0) or 0) or h, h))
    if x1 <= x0 or y1 <= y0:
        return
    try:
        widget.set(x=float(x0), y=float(y0), w=float(x1 - x0), h=float(y1 - y0))
    except Exception as e:
        log.debug("crop widget field-sync failed: %s", e)


class CropAction(TransformAction):
    """Crop the dataset to a spatial (image) box + optional time range — a
    TransformAction that adds a lazy "Cropped" node to the SAME tree. Nothing is
    materialised (isig/inav are dask-view slices), so a multi-GB movie crops for
    free. Zero ranges keep the full extent on that axis.

    The spatial box (x0/x1/y0/y1) is set by an interactive rectangle widget on
    the source plot (see crop_open/crop_close/crop_set_region) rather than
    typed-in-only values — the widget is the PRIMARY input (matches the CZB
    search-window precedent): when the caret's widget is present, its LIVE
    geometry overrides the (possibly stale) typed fields. The optional t0/t1
    nav-range stays a plain typed field (no on-plot widget for it)."""

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
    }

    def build_kwargs(self, signal, x0=0, x1=0, y0=0, y1=0, t0=0, t1=0, **_):
        am = signal.axes_manager
        if am.signal_dimension >= 2:
            sig_ax = am.signal_axes
            w, h = int(sig_ax[0].size), int(sig_ax[1].size)
            live = _crop_widget_bounds(self.signal_tree, w, h)
            if live is not None:
                x0, x1, y0, y1 = live
                if (x0, x1, y0, y1) == (0, w, 0, h):
                    # Widget covers the full frame (opened, never adjusted) —
                    # a real no-op crop. Zero it out so _crop_signal's existing
                    # all-zero-means-noop contract applies and no redundant
                    # "Cropped" node is created.
                    x0 = x1 = y0 = y1 = 0
        return {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "t0": t0, "t1": t1}

    def run(self, **params):
        # True no-op (full-frame box / all-zero fields): skip the tree
        # entirely — add_transformation has no identity check, so letting it
        # run would add a redundant "Cropped" node pointing at the SAME
        # signal. The box stays up so the user can adjust and try again.
        resolved = self._resolved_params(params)
        kwargs = self.build_kwargs(self.signal, **resolved)
        if not any(int(v or 0) for v in kwargs.values()):
            return self.signal
        new = super().run(**params)
        if new is not None:
            # The box no longer describes the (now different) displayed node —
            # show_tree_node already hid it on the node switch; this covers
            # any path that returned a node without switching. Idempotent.
            crop_close(self.session, self.plot, {})
        return new

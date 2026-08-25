"""
center_zero_beam.py — Electron-native Center Zero Beam (two-tab parity).

Mirrors the Qt action (``pyxem.center_zero_beam`` / ``..._setup``):

  Automatic — pyxem ``get_direct_beam_position(method, half_square_width)`` →
              optional linear-plane flat field → ``center_direct_beam`` → a
              "Centered" child node (the current DP updates in place).
  Manual    — drop a draggable crosshair on the DP at the zero beam, Apply a
              CONSTANT shift (``centre − picked``) → "Centered (Manual)" node.

No Qt: this is the Electron-native action; it is import-safe in the backend
(mirrors the find_vectors / orientation split). The old pyqtgraph caret was
removed in the Qt-removal cleanup.
"""
from __future__ import annotations

import logging

import numpy as np

from de_shell.ipc import emit, emit_status, emit_error
from spyde.actions.context import src_plot_tree as _src_plot_tree, current_signal as _current_signal

log = logging.getLogger(__name__)

DEFAULTS = dict(method="center_of_mass", half_square_width=0, make_flat_field=False)

# Declared parameter schema (single source of truth for every host — the
# Electron CenterZeroBeamWizard.tsx caret mirrors the Automatic tab; the
# Manual tab is a crosshair interaction, not a parameter). Same dict spec as
# toolbars.yaml `parameters:`. CZB has no controller class (pure staged
# handlers), so the schema lives module-level; resolved via
# registry.wizard_parameters("czb").
PARAMETERS = {
    "method": {
        "name": "Method", "type": "enum", "default": DEFAULTS["method"],
        # everything pyxem get_direct_beam_position accepts
        "choices": ["center_of_mass", "cross_correlate", "blur", "interpolate"],
        "tab": "Automatic",
    },
    "half_square_width": {
        "name": "Half window (px, 0=full)", "type": "int",
        "default": DEFAULTS["half_square_width"], "min": 0, "max": 256,
        "tab": "Automatic",
    },
    "make_flat_field": {
        "name": "Plane-fit shifts", "type": "bool",
        "default": DEFAULTS["make_flat_field"], "tab": "Automatic",
    },
}
_CROSS_COLOR = "#ffcc00"
_REGION_COLOR = "#ffcc00"    # the half_square_width centering window outline
_FOUND_COLOR = "#a6e3a1"     # the found beam-centre marker


def _czb_remove_region(tree) -> None:
    """Tear down the Automatic-tab search-window widget (see czb_open/
    czb_set_region). ``_czb_region_mg`` is the historical name (it used to hold
    a static MarkerGroup from add_squares); it now holds a draggable
    RectangleWidget, hidden the same way the Manual-tab crosshair is (widgets
    have no ``remove()``, only ``hide()`` — see _czb_manual_stop_obj)."""
    widget = getattr(tree, "_czb_region_mg", None) if tree is not None else None
    if widget is not None:
        try:
            widget.hide()
        except Exception as e:
            log.debug("hiding czb region widget failed: %s", e)
        tree._czb_region_mg = None
        tree._czb_region_handler = None


def _czb_remove_found(tree) -> None:
    for mg in (getattr(tree, "_czb_found_mgs", None) or []) if tree is not None else []:
        try:
            mg.remove()
        except Exception as e:
            log.debug("removing czb found-centre marker failed: %s", e)
    if tree is not None:
        tree._czb_found_mgs = None


def _czb_region_from_widget(tree, w: int, h: int) -> int | None:
    """Read the current search-window half-width off the live widget (None if
    there is no widget). ``get_direct_beam_position`` always centres the search
    box on the frame, so only the SIZE is meaningful — position drift from a
    resize-drag is corrected back to centred by the drag handler. A widget that
    (still) covers the FULL frame reports 0 ("full frame", the same as no
    override) rather than a half-width that would clip nothing anyway."""
    widget = getattr(tree, "_czb_region_mg", None) if tree is not None else None
    if widget is None:
        return None
    side = min(float(widget.w), float(widget.h))
    if side >= min(float(w), float(h)) - 1e-6:
        return 0
    return max(1, int(round(side / 2.0)))


def _czb_on_region_drag(tree, w: int, h: int, event) -> None:
    """Drag handler for the Automatic-tab search-window widget: pyxem's
    ``get_direct_beam_position(half_square_width=...)`` always centres the
    search box on the frame, so a resize is allowed but the box is snapped back
    to centred on every move (dragging the whole box, or an asymmetric resize,
    would otherwise silently do nothing — worse, it would look like it did).
    Clamped to the frame so it can never grow past the detector edges.

    RE-ENTRANCY GUARD: anyplotlib ``Widget.set()`` fires ``pointer_move``
    UNCONDITIONALLY (even when nothing changed, and regardless of ``_push``),
    so the ``set()`` below re-invokes this handler synchronously — unguarded,
    ONE JS drag frame recursed ~2000 deep before RecursionError. A hard
    per-tree flag breaks the cycle; compare-before-set is NOT sufficient
    (``set()`` fires on a no-change write too)."""
    widget = getattr(tree, "_czb_region_mg", None) if tree is not None else None
    if widget is None:
        return
    if getattr(tree, "_czb_region_clamping", False):
        return
    tree._czb_region_clamping = True
    try:
        side = min(float(widget.w), float(widget.h), float(w), float(h))
        side = max(2.0, side)
        widget.set(x=(w - side) / 2.0, y=(h - side) / 2.0, w=side, h=side)
    except Exception as e:
        log.debug("czb region widget re-centre failed: %s", e)
    finally:
        tree._czb_region_clamping = False


def czb_set_region(session, plot, payload) -> None:
    """Automatic tab: show a DRAGGABLE (resize-only, re-centred) rectangle
    widget outlining the centring search window (the centred
    ``half_square_width`` box pyxem's ``get_direct_beam_position`` uses), so
    the user can see AND adjust what region drives the fit — activating the
    Automatic tab shows it immediately, covering the FULL frame at
    ``half_square_width=0`` (pyxem's "search everywhere" default) exactly like
    the full-frame box Crop opens with. Resizing the widget is the primary
    input — the numeric half-width field stays in sync via the widget's own
    drag events (see CenterZeroBeamWizard.tsx), and czb_run reads the live
    widget size. The box is only ever REMOVED by czb_close (caret closed / tab
    left)."""
    src, tree = _src_plot_tree(session, plot)
    signal = _current_signal(src)
    plot2d = getattr(src, "_plot2d", None) if src is not None else None
    if plot2d is None or signal is None or tree is None:
        return
    sig_ax = signal.axes_manager.signal_axes
    w, h = int(sig_ax[0].size), int(sig_ax[1].size)
    hw = int(payload.get("half_square_width", 0) or 0)
    side = float(min(2 * hw, w, h)) if hw > 0 else float(min(w, h))
    existing = getattr(tree, "_czb_region_mg", None)
    if existing is not None:
        # Re-centre + resize the LIVE widget in place (a caret field edit while
        # the box is already up) instead of tearing it down and losing the
        # user's in-progress drag state.
        try:
            existing.set(x=(w - side) / 2.0, y=(h - side) / 2.0, w=side, h=side)
        except Exception as e:
            log.debug("czb region widget resize failed: %s", e)
        return
    try:
        widget = plot2d.add_rectangle_widget(
            x=(w - side) / 2.0, y=(h - side) / 2.0, w=side, h=side,
            color=_REGION_COLOR, show_handles=True,
        )
        tree._czb_region_mg = widget
        from spyde.drawing.selectors.base_selector import event_handler_fn
        handler = event_handler_fn(
            lambda event: _czb_on_region_drag(tree, w, h, event)
        )
        widget.add_event_handler(handler, "pointer_move", "pointer_up")
        tree._czb_region_handler = handler   # keep a ref alive (weak callback)
    except Exception as e:
        log.debug("czb region widget draw failed: %s", e)


def _czb_show_found(src, tree, signal, beam_xy) -> None:
    """Mark the found beam centre on the DP: a ring at the ORIGINAL beam
    position (mean over the scan for Automatic; the picked spot for Manual)
    plus a small cross at the target centre the pattern is now centred on."""
    plot2d = getattr(src, "_plot2d", None) if src is not None else None
    if plot2d is None or tree is None:
        return
    try:
        _czb_remove_found(tree)
        sig_ax = signal.axes_manager.signal_axes
        w, h = int(sig_ax[0].size), int(sig_ax[1].size)
        bx, by = float(beam_xy[0]), float(beam_xy[1])
        arm = max(3.0, min(w, h) * 0.03)
        mgs = [
            plot2d.add_circles([[bx, by]], name="czb_found", radius=5,
                               edgecolors=_FOUND_COLOR, facecolors=None,
                               linewidths=2.0, alpha=0.95),
            plot2d.add_lines(
                [[[w / 2.0 - arm, h / 2.0], [w / 2.0 + arm, h / 2.0]],
                 [[w / 2.0, h / 2.0 - arm], [w / 2.0, h / 2.0 + arm]]],
                name="czb_centre", edgecolors=_FOUND_COLOR, linewidths=1.2),
        ]
        tree._czb_found_mgs = mgs
    except Exception as e:
        log.debug("czb found-centre marker failed: %s", e)


def _display(src, tree, new_signal) -> None:
    """Switch the source DP to the new (centered) node, re-slice from the
    navigator, and refresh the Workflow panel (the shared lifecycle helper)."""
    from spyde.actions.lifecycle import show_tree_node
    show_tree_node(src, tree, new_signal)


def center_zero_beam(ctx, action_name: str = "Center Zero Beam", **kwargs):
    """Parent toolbar action — a no-op; the Electron toolbar opens the staged
    Center-Zero-Beam wizard (Automatic / Manual) which drives the ``czb_*``
    handlers."""
    return None


def czb_run(session, plot, payload) -> None:
    """Automatic tab: estimate the beam position per pattern and centre it."""
    src, tree = _src_plot_tree(session, plot)
    signal = _current_signal(src)
    if src is None or tree is None or signal is None:
        emit_error("Center Zero Beam: no active dataset")
        return
    method = str(payload.get("method", DEFAULTS["method"]))
    hw = int(payload.get("half_square_width", 0) or 0)
    # The rectangle widget (see czb_set_region) is the PRIMARY input once it's
    # on screen — a drag/resize does not round-trip back into the caret's
    # numeric field on every frame, so the typed field can be stale at Run
    # time. Always prefer the live widget when one is up. A full-frame widget
    # (its opening state) reads back as hw=0, and at hw=0 the
    # half_square_width kwarg is OMITTED below entirely (pyxem then searches
    # the full frame) — it is never passed as 0.
    sig_ax = signal.axes_manager.signal_axes
    w, h = int(sig_ax[0].size), int(sig_ax[1].size)
    live_hw = _czb_region_from_widget(tree, w, h)
    if live_hw is not None:
        hw = live_hw
    flat = bool(payload.get("make_flat_field", False))
    emit_status("Centering zero beam…")

    from spyde.actions.lifecycle import supersede
    handle = supersede(getattr(tree, "_czb_compute", None), tree)
    tree._czb_compute = handle

    def _work():
        if handle.stopped:
            return
        try:
            try:
                signal.set_signal_type("electron_diffraction")
            except Exception as e:
                log.debug("set_signal_type(electron_diffraction) failed: %s", e)
            kw = {"method": method, "lazy_output": False}
            if hw > 0:
                kw["half_square_width"] = hw
            shifts = signal.get_direct_beam_position(**kw)
            if getattr(shifts, "_lazy", False):
                shifts.compute()
            if flat:
                try:
                    lp = shifts.get_linear_plane()
                    if lp is not None:
                        shifts = lp
                except Exception as e:
                    log.debug("flat-field plane failed: %s", e)
            new = tree.add_transformation(
                parent_signal=signal, method="center_direct_beam",
                node_name="Centered", shifts=shifts, inplace=False,
                # Per-pattern shift estimates are local; flat-field mode fits one
                # plane across ALL shifts first, so that specific run is opaque.
                local=(not flat),
            )
            if new is None:
                emit_error("Center Zero Beam: centering failed")
                return
            _display(src, tree, new)
            # Mark where the beam was found (mean over the scan): shift
            # convention is (centre − beam), so beam = centre − shift.
            try:
                s = np.asarray(shifts.data, dtype=np.float64)
                sig_ax = signal.axes_manager.signal_axes
                w, h = int(sig_ax[0].size), int(sig_ax[1].size)
                beam = (w / 2.0 - float(np.nanmean(s[..., 0])),
                        h / 2.0 - float(np.nanmean(s[..., 1])))
                _czb_show_found(src, tree, signal, beam)
            except Exception as e:
                log.debug("czb found-centre (auto) failed: %s", e)
            emit_status("Zero beam centered")
            emit({"type": "czb_done",
                  "window_id": getattr(src, "window_id", None), "mode": "auto"})
        except Exception as e:
            emit_error(f"Center Zero Beam (auto) failed: {e}")
            log.exception("Center Zero Beam (auto) failed")
        finally:
            handle.retire()

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="czb-auto")


def czb_open(session, plot, payload) -> None:
    """Manual tab: drop a draggable crosshair at the centre of the DP for the
    user to drag onto the zero beam."""
    src, tree = _src_plot_tree(session, plot)
    signal = _current_signal(src)
    plot2d = getattr(src, "_plot2d", None) if src is not None else None
    if plot2d is None or signal is None:
        emit_error("Center Zero Beam: no active diffraction plot to place the "
                   "crosshair on")
        return
    sig_ax = signal.axes_manager.signal_axes
    w, h = int(sig_ax[0].size), int(sig_ax[1].size)
    _czb_manual_stop_obj(tree)   # replace any prior crosshair
    try:
        cross = plot2d.add_crosshair_widget(cx=w / 2.0, cy=h / 2.0, color=_CROSS_COLOR)
        tree._czb_cross = cross
        emit_status("Drag the crosshair onto the zero beam, then Apply")
    except Exception as e:
        log.debug("czb crosshair add failed: %s", e)


def _czb_manual_stop_obj(tree) -> None:
    cross = getattr(tree, "_czb_cross", None) if tree is not None else None
    if cross is not None:
        try:
            cross.hide()
        except Exception as e:
            log.debug("hiding czb crosshair failed: %s", e)
        tree._czb_cross = None


def _czb_clear_widgets(tree) -> None:
    """Remove every CZB on-plot artefact from *tree* — the Manual crosshair,
    the Automatic search-window widget, and the found-centre markers. Harmless
    when absent. Shared by czb_close, the node-switch teardown in
    ``lifecycle.show_tree_node``, and ``BaseSignalTree.close()``."""
    _czb_manual_stop_obj(tree)
    _czb_remove_region(tree)
    _czb_remove_found(tree)


def czb_close(session, plot, payload=None) -> None:
    """Caret closed / left the Manual tab → remove the crosshair, the region
    box, and the found-centre marker."""
    _src, tree = _src_plot_tree(session, plot)
    _czb_clear_widgets(tree)


def czb_pick(session, plot, payload) -> None:
    """Manual tab Apply: centre by the picked crosshair position (constant shift
    ``centre − picked`` over the whole scan)."""
    src, tree = _src_plot_tree(session, plot)
    signal = _current_signal(src)
    if src is None or tree is None or signal is None:
        emit_error("Center Zero Beam: no active dataset")
        return
    cross = getattr(tree, "_czb_cross", None)
    if cross is not None:
        cx, cy = float(cross.cx), float(cross.cy)
    else:
        cx, cy = payload.get("cx"), payload.get("cy")
    if cx is None or cy is None:
        emit_error("Center Zero Beam: place the crosshair first")
        return

    from spyde.actions.lifecycle import supersede
    handle = supersede(getattr(tree, "_czb_compute", None), tree)
    tree._czb_compute = handle

    def _work():
        if handle.stopped:
            return
        try:
            import hyperspy.api as hs
            try:
                signal.set_signal_type("electron_diffraction")
            except Exception as e:
                log.debug("set_signal_type(electron_diffraction) failed: %s", e)
            am = signal.axes_manager
            sig_ax = am.signal_axes
            w, h = int(sig_ax[0].size), int(sig_ax[1].size)
            # shift convention matches get_direct_beam_position: (centre − beam),
            # [x=col, y=row] in pixels.
            sx, sy = (w / 2.0 - float(cx)), (h / 2.0 - float(cy))
            nav_shape = tuple(int(n) for n in am.navigation_shape)[::-1]  # (ny, nx)
            data = np.zeros(nav_shape + (2,), dtype=np.float32)
            data[..., 0] = sx
            data[..., 1] = sy
            shifts = hs.signals.Signal1D(data)
            for i, ax in enumerate(am.navigation_axes):
                oax = shifts.axes_manager.navigation_axes[i]
                oax.scale, oax.offset = ax.scale, ax.offset
                oax.units, oax.name = ax.units, ax.name
            new = tree.add_transformation(
                parent_signal=signal, method="center_direct_beam",
                node_name="Centered (Manual)", shifts=shifts, inplace=False,
                # A constant shift over the whole scan — trivially local.
                local=True,
            )
            if new is None:
                emit_error("Center Zero Beam: centering failed")
                return
            _display(src, tree, new)
            _czb_manual_stop_obj(tree)
            _czb_show_found(src, tree, signal, (float(cx), float(cy)))
            emit_status("Zero beam centered (manual)")
            emit({"type": "czb_done",
                  "window_id": getattr(src, "window_id", None), "mode": "manual"})
        except Exception as e:
            emit_error(f"Center Zero Beam (manual) failed: {e}")
            log.exception("Center Zero Beam (manual) failed")
        finally:
            handle.retire()

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="czb-manual")

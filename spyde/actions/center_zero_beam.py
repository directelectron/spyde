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

from spyde.backend.ipc import emit, emit_status, emit_error
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
    mg = getattr(tree, "_czb_region_mg", None) if tree is not None else None
    if mg is not None:
        try:
            mg.remove()
        except Exception as e:
            log.debug("removing czb region box failed: %s", e)
        tree._czb_region_mg = None


def _czb_remove_found(tree) -> None:
    for mg in (getattr(tree, "_czb_found_mgs", None) or []) if tree is not None else []:
        try:
            mg.remove()
        except Exception as e:
            log.debug("removing czb found-centre marker failed: %s", e)
    if tree is not None:
        tree._czb_found_mgs = None


def czb_set_region(session, plot, payload) -> None:
    """Automatic tab: outline the centering search window (the centred
    ``half_square_width`` box pyxem's ``get_direct_beam_position`` uses) live
    on the DP so the user sees what region drives the fit. ``hw <= 0`` (full
    frame) removes the box."""
    src, tree = _src_plot_tree(session, plot)
    signal = _current_signal(src)
    plot2d = getattr(src, "_plot2d", None) if src is not None else None
    if plot2d is None or signal is None or tree is None:
        return
    _czb_remove_region(tree)
    hw = int(payload.get("half_square_width", 0) or 0)
    if hw <= 0:
        return
    sig_ax = signal.axes_manager.signal_axes
    w, h = int(sig_ax[0].size), int(sig_ax[1].size)
    side = float(min(2 * hw, w, h))
    try:
        tree._czb_region_mg = plot2d.add_squares(
            [[w / 2.0, h / 2.0]], [side], name="czb_region",
            edgecolors=_REGION_COLOR, facecolors=None, linewidths=1.5, alpha=0.9,
        )
    except Exception as e:
        log.debug("czb region box draw failed: %s", e)


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
    flat = bool(payload.get("make_flat_field", False))
    emit_status("Centering zero beam…")

    def _work():
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


def czb_close(session, plot, payload=None) -> None:
    """Caret closed / left the Manual tab → remove the crosshair, the region
    box, and the found-centre marker."""
    _src, tree = _src_plot_tree(session, plot)
    _czb_manual_stop_obj(tree)
    _czb_remove_region(tree)
    _czb_remove_found(tree)


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

    def _work():
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

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="czb-manual")

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
_CROSS_COLOR = "#ffcc00"


def _display(src, tree, new_signal) -> None:
    """Switch the source DP to display the new (centered) node and re-slice from
    the navigator so the centered frame shows immediately. ``add_transformation``
    only REGISTERS the new PlotState; switching the view is a separate step."""
    try:
        src.set_plot_state(new_signal)
    except Exception as e:
        log.debug("switching source plot to centered node failed: %s", e)
    npm = getattr(tree, "navigator_plot_manager", None)
    if npm is None:
        return
    for sels in getattr(npm, "navigation_selectors", {}).values():
        for sel in sels:
            try:
                sel.delayed_update_data(force=True)
            except Exception as e:
                log.debug("re-slicing navigator after centering failed: %s", e)


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
            try:
                session._reemit_signal_tree(src)
            except Exception as e:
                log.debug("re-emitting signal tree after centering failed: %s", e)
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
    """Caret closed / left the Manual tab → remove the crosshair."""
    _src, tree = _src_plot_tree(session, plot)
    _czb_manual_stop_obj(tree)


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
            try:
                session._reemit_signal_tree(src)
            except Exception as e:
                log.debug("re-emitting signal tree after centering failed: %s", e)
            _czb_manual_stop_obj(tree)
            emit_status("Zero beam centered (manual)")
            emit({"type": "czb_done",
                  "window_id": getattr(src, "window_id", None), "mode": "manual"})
        except Exception as e:
            emit_error(f"Center Zero Beam (manual) failed: {e}")
            log.exception("Center Zero Beam (manual) failed")

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="czb-manual")

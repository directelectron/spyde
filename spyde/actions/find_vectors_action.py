"""
find_vectors_action.py — Electron-native "Find Diffraction Vectors".

Reuses the Qt-free, memory-safe compute core (`_do_compute_vectors`) from
`find_vectors.py` and produces a new *vectors image* window
(signal_type ``spyde_diffraction_vectors_image``) with
``tree.diffraction_vectors`` attached — which unlocks the vector actions
(Vector Virtual Imaging / Vector Orientation Mapping).

No Qt, no live Qt preview. A result window appears immediately with a zero
count-map navigator; the compute runs on a background thread and, when done,
swaps in the rendered vectors + the real count map.

MEMORY: `_do_compute_vectors` uses `dask.array.map_overlap` and NEVER calls
`.compute()` on the full dataset (see its docstring + test_find_vectors_memory).
"""
from __future__ import annotations

import threading

import numpy as np
import hyperspy.api as hs

from spyde.backend.ipc import emit_status, emit_error
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.find_vectors import _do_compute_vectors, _copy_nav_axes_to

# Defaults mirror the old Qt CaretGroup sliders.
DEFAULTS: dict = dict(
    sigma=1.0, kernel_radius=5, threshold=0.5, min_distance=5, subpixel=True,
)


def _coerce(params: dict) -> dict:
    p = dict(DEFAULTS)
    for k, default in DEFAULTS.items():
        v = params.get(k)
        if v is None or v == "":
            continue
        try:
            p[k] = bool(v) if isinstance(default, bool) else type(default)(v)
        except (TypeError, ValueError):
            pass
    return p


def find_diffraction_vectors(ctx, action_name: str = "Find Diffraction Vectors", **params):
    """Toolbar entry point (ActionContext convention) — one-shot batch compute."""
    plot = ctx.plot
    session = ctx.session
    src_tree = getattr(plot, "signal_tree", None)
    if src_tree is None or session is None:
        emit_error("Find Vectors: no active dataset")
        return None
    src = src_tree.root
    am = src.axes_manager
    if am.signal_dimension != 2 or am.navigation_dimension < 2:
        emit_error("Find Vectors needs a 4D-STEM dataset (2-D nav + 2-D signal)")
        return None
    return _start_batch(session, plot, src_tree, _coerce(params))


def _start_batch(session, plot, src_tree, p: dict):
    """Build the result window, then run the full-dataset compute on a background
    thread. Shared by the toolbar one-shot and the staged-wizard ``fv_run``."""
    src = src_tree.root
    am = src.axes_manager

    # Drop any live tuning preview — the final overlay replaces it.
    prev = getattr(src_tree, "_fv_preview", None)
    if prev is not None:
        try:
            prev.remove()
        except Exception:
            pass
        src_tree._fv_preview = None

    # ── Build the result tree up front: a lazy zero placeholder with the
    #    source's axes (so we never reference the raw dataset) + a zero
    #    count-map navigator override (so the base navigator is NOT recomputed
    #    from the full dataset).
    import dask.array as da
    nav_dim = am.navigation_dimension
    data_shape = tuple(src.data.shape)
    nav_shape_full = data_shape[:nav_dim]
    nav_shape_2d = nav_shape_full[-2:]
    nav_chunks = tuple(min(32, int(s)) for s in nav_shape_full)
    placeholder = da.zeros(
        data_shape, chunks=nav_chunks + tuple(data_shape[nav_dim:]), dtype=np.float32,
    )
    new_sig = src._deepcopy_with_new_data(placeholder)
    if not new_sig._lazy:
        new_sig._lazy = True
        new_sig._assign_subclass()
    try:
        new_sig.set_signal_type("spyde_diffraction_vectors_image")
    except Exception:
        pass
    base_title = src.metadata.get_item("General.title", "Signal")
    new_sig.metadata.General.title = f"{base_title} — Vectors"

    nav_sig = hs.signals.BaseSignal(np.zeros(nav_shape_full, dtype=np.float32)).T
    nav_sig.metadata.General.title = "Vector count map"
    _copy_nav_axes_to(src, nav_sig)

    from spyde.drawing.selectors import CrosshairSelector
    new_tree = session._add_signal(
        new_sig, navigator_override=nav_sig, selector_type=CrosshairSelector,
    )

    emit_status("Finding diffraction vectors…")
    src_dp_plot = plot   # overlay the found vectors on the live DP we ran from

    def _work():
        try:
            vecs = _do_compute_vectors(src, p, main_window=session, signal_tree=src_tree)
            if vecs is None:
                emit_error("Find Vectors: compute returned no result")
                return
            _finalize(new_tree, vecs)
            _overlay_on_source(src_tree, src_dp_plot, vecs)
        except Exception as e:
            import traceback
            emit_error(f"Find Vectors failed: {e}")
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True, name="find-vectors").start()
    return None


def _overlay_on_source(src_tree, dp_plot, vecs) -> None:
    """Overlay the found vectors as live circle markers on the SOURCE diffraction
    pattern (Qt parity: peaks tracked the navigator). Replaces any prior overlay
    from an earlier run so re-running Find Vectors doesn't stack markers."""
    if dp_plot is None or src_tree is None:
        return
    from spyde.actions.vector_overlay import attach_vector_overlay
    old = getattr(src_tree, "_vector_overlay", None)
    if old is not None:
        try:
            old.remove()
        except Exception:
            pass
    try:
        src_tree._vector_overlay = attach_vector_overlay(dp_plot, vecs, src_tree)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("vector overlay attach failed: %s", e)


def _finalize(tree, vecs) -> None:
    """Swap the placeholder for vectors-rendered frames, attach the vectors,
    fill the count-map navigator, and unlock the vector toolbar actions."""
    tree.root.data = vecs.to_rendered_dask()
    tree.diffraction_vectors = vecs

    count_map = vecs.count_map().astype(np.float32)
    nav_plot = _first_nav_plot(tree)
    if nav_plot is not None:
        try:
            nav_plot.needs_auto_level = True
            nav_plot.set_data(count_map)
        except Exception:
            pass

    # Re-slice the signal plot from the new (rendered) root + re-send the
    # toolbar config so the now-available vector actions appear.
    for sp in list(getattr(tree, "signal_plots", [])):
        try:
            state = getattr(sp, "plot_state", None)
            if state is not None and hasattr(state, "_send_toolbar_config"):
                state._send_toolbar_config()
        except Exception:
            pass
    _refresh_signal_from_navigator(tree)

    total = int(count_map.sum())
    emit_status(f"Found {total} diffraction vectors")


def _first_nav_plot(tree):
    npm = getattr(tree, "navigator_plot_manager", None)
    if npm is None:
        return None
    for pw in list(npm.plot_windows.keys()):
        plots = npm.plots.get(pw, [])
        if plots:
            return plots[0]
    return None


def _refresh_signal_from_navigator(tree) -> None:
    """Force the navigator selector to re-slice so the signal plot shows a
    vectors-rendered frame instead of the placeholder zeros."""
    npm = getattr(tree, "navigator_plot_manager", None)
    if npm is None:
        return
    for selectors in getattr(npm, "navigation_selectors", {}).values():
        for sel in selectors:
            try:
                sel.delayed_update_data(force=True)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Staged "wizard" workflow (Qt parity): a live found-peaks PREVIEW on the source
# DP while you tune the sliders, then Compute → the full-dataset batch. The
# preview overlay lives on the source tree as `_fv_preview`.
# ─────────────────────────────────────────────────────────────────────────────

def fv_preview(session, plot, payload) -> None:
    """'Tune' step: attach the LIVE found-peaks preview to the source DP so the
    red circles update as you tune the sliders / move the navigator (Qt parity).
    Idempotent — replaces any existing preview."""
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Find Vectors: no active dataset")
        return
    am = tree.root.axes_manager
    if am.signal_dimension != 2 or am.navigation_dimension < 2:
        emit_error("Find Vectors needs a 4D-STEM dataset (2-D nav + 2-D signal)")
        return
    p = _coerce(payload)

    def _work():
        try:
            from spyde.actions.vector_overlay import attach_find_vectors_preview
            old = getattr(tree, "_fv_preview", None)
            if old is not None:
                try:
                    old.remove()
                except Exception:
                    pass
            tree._fv_preview = attach_find_vectors_preview(
                src, tree.root, tree, sigma=p["sigma"],
                kernel_radius=p["kernel_radius"], threshold=p["threshold"],
                min_distance=p["min_distance"], subpixel=p["subpixel"],
            )
            emit_status("Find Vectors: tune the parameters — peaks preview under "
                        "the crosshair, then Compute")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("fv_preview attach failed: %s", e)

    threading.Thread(target=_work, daemon=True, name="fv-preview").start()


def fv_tune(session, plot, payload) -> None:
    """'Tune' step: live-update the preview sliders and redraw the found peaks at
    the current crosshair position."""
    src, tree = _src_plot_tree(session, plot)
    prev = getattr(tree, "_fv_preview", None) if tree is not None else None
    if prev is None:
        return

    def _work():
        try:
            prev.set_params(**_coerce(payload))
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("fv_tune failed: %s", e)

    threading.Thread(target=_work, daemon=True, name="fv-tune").start()


def fv_run(session, plot, payload) -> None:
    """'Compute' step: full-dataset batch with the tuned params → a new vectors
    image window. Drops the live preview first."""
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Find Vectors: no active dataset")
        return
    am = tree.root.axes_manager
    if am.signal_dimension != 2 or am.navigation_dimension < 2:
        emit_error("Find Vectors needs a 4D-STEM dataset (2-D nav + 2-D signal)")
        return
    _start_batch(session, src, tree, _coerce(payload))


def fv_stop(session, plot, payload=None) -> None:
    """Caret closed: remove the live preview overlay."""
    src, tree = _src_plot_tree(session, plot)
    prev = getattr(tree, "_fv_preview", None) if tree is not None else None
    if prev is not None:
        try:
            prev.remove()
        except Exception:
            pass
        tree._fv_preview = None

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

import logging
import threading
import time

import numpy as np
import hyperspy.api as hs

log = logging.getLogger(__name__)

from spyde.backend.ipc import emit, emit_status, emit_error
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.find_vectors import _do_compute_vectors, _copy_nav_axes_to

# Defaults mirror the old Qt CaretGroup sliders, plus the DoG band-pass method
# (better for small 2-3 px spots / beam-stopped data — see the 3 nm benchmark).
# `threshold` is shared but means different things per method: NXCORR uses an
# [-1,1] correlation score (~0.5); DoG uses an absolute band-pass SNR (~10).
DEFAULTS: dict = dict(
    sigma=1.0, kernel_radius=5, threshold=0.5, min_distance=5, subpixel=True,
    method="nxcorr", dog_sigma1=0.8, dog_sigma2=2.0, beamstop_auto=False,
)

# Per-method threshold default applied when the user switches method without
# having explicitly set a threshold for it.
_METHOD_THRESHOLD = {"nxcorr": 0.5, "dog": 10.0}


def _coerce(params: dict) -> dict:
    p = dict(DEFAULTS)
    for k, default in DEFAULTS.items():
        v = params.get(k)
        if v is None or v == "":
            continue
        try:
            p[k] = bool(v) if isinstance(default, bool) else type(default)(v)
        except (TypeError, ValueError) as e:
            log.debug("find-vectors param %r=%r not coercible, using default: %s", k, v, e)
    p["method"] = str(p.get("method", "nxcorr")).lower()
    if p["method"] not in _METHOD_THRESHOLD:
        p["method"] = "nxcorr"
    # DoG uses an SNR threshold on a different scale than NXCORR's [-1,1] score;
    # if the caller left threshold at the NXCORR default while asking for DoG,
    # substitute the DoG default so the first preview isn't empty/flooded.
    if (params.get("threshold") in (None, "")) and p["method"] == "dog":
        p["threshold"] = _METHOD_THRESHOLD["dog"]
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
        except Exception as e:
            log.debug("dropping find-vectors preview failed: %s", e)
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
    except Exception as e:
        log.debug("set_signal_type(diffraction_vectors_image) failed: %s", e)
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

    # ── Progressive (live) count map: the compute writes per-chunk vector counts
    #    into a shared-memory buffer as chunks finish; a poller paints them into
    #    the count-map navigator so it fills in live instead of all-at-the-end.
    from spyde.drawing.update_functions import ensure_live_buffer, read_live_buffer
    shm_name = f"spyde_fv_{id(plot)}"
    try:
        shm = ensure_live_buffer(nav_shape_2d, shm_name)
    except Exception:
        shm, shm_name = None, None
    stop_poll = [False]

    def _poller():
        nav_plot = _first_nav_plot(new_tree)
        while not stop_poll[0]:
            try:
                arr = read_live_buffer(nav_shape_2d, shm_name)
                if nav_plot is not None and np.isfinite(arr).any():
                    nav_plot.needs_auto_level = True
                    nav_plot.set_data(np.nan_to_num(arr).astype(np.float32))
            except Exception as e:
                log.debug("live count-map poll paint failed: %s", e)
            time.sleep(0.35)

    def _work():
        try:
            vecs = _do_compute_vectors(src, p, main_window=session,
                                       signal_tree=src_tree, shm_name=shm_name)
            stop_poll[0] = True                      # final paint owns the nav plot
            if vecs is None:
                emit_error("Find Vectors: compute returned no result")
                return
            _finalize(new_tree, vecs)
            _overlay_on_source(src_tree, src_dp_plot, vecs)
        except Exception as e:
            emit_error(f"Find Vectors failed: {e}")
            log.exception("Find Vectors compute failed")
        finally:
            stop_poll[0] = True
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    log.debug("shared-memory cleanup failed: %s", e)

    if shm_name is not None:
        threading.Thread(target=_poller, daemon=True, name="fv-poll").start()
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
        except Exception as e:
            log.debug("removing prior vector overlay failed: %s", e)
    try:
        src_tree._vector_overlay = attach_vector_overlay(dp_plot, vecs, src_tree)
    except Exception as e:
        log.debug("vector overlay attach failed: %s", e)


def _finalize(tree, vecs) -> None:
    """Swap the placeholder for vectors-rendered frames, attach the vectors,
    fill the count-map navigator, and unlock the vector toolbar actions."""
    tree.root.data = vecs.to_rendered_dask()
    # The signal plot's CachedDaskArray captured the OLD placeholder array when
    # the window first rendered (zeros). Drop it so navigation renders the new
    # disk frames — otherwise the result window stays black (Qt parity: the
    # computed-vectors window shows each position's vectors as flat disks).
    try:
        tree.root.cached_dask_array = None
        tree.root._clear_cache_dask_data()
    except Exception as e:
        log.debug("clearing stale cached dask array failed: %s", e)
    tree.diffraction_vectors = vecs

    count_map = vecs.count_map().astype(np.float32)
    nav_plot = _first_nav_plot(tree)
    if nav_plot is not None:
        try:
            nav_plot.needs_auto_level = True
            nav_plot.set_data(count_map)
        except Exception as e:
            log.debug("painting final count map onto navigator failed: %s", e)

    # Re-send the toolbar config so the now-available vector actions appear.
    for sp in list(getattr(tree, "signal_plots", [])):
        try:
            sp.needs_auto_level = True
            state = getattr(sp, "plot_state", None)
            if state is not None and hasattr(state, "_send_toolbar_config"):
                state._send_toolbar_config()
        except Exception as e:
            log.debug("re-sending toolbar config after find-vectors failed: %s", e)
    _install_render_display(tree, vecs)
    _overlay_on_result(tree, vecs)

    total = int(count_map.sum())
    emit_status(f"Found {total} diffraction vectors")


def _install_render_display(tree, vecs) -> None:
    """Drive the result window's signal plot by rendering vectors frames
    IN-PROCESS on every navigator move (Qt parity) — ``render_frame`` is an O(1)
    CSR slice. This REPLACES the navigator's slice function so navigation never
    touches the lazy ``to_rendered_dask`` root, whose chunks are delivered
    asynchronously (Future → shared-memory) and can leave the window black on
    real distributed data. Each navigated position now paints its disks
    synchronously, exactly like the Qt ``_make_hooked`` update."""
    from spyde.actions.vector_overlay import _indices_to_iyix
    H = int(vecs.sig_axes[1].size)
    W = int(vecs.sig_axes[0].size)

    def _fn(selector, child, indices):
        iy, ix = _indices_to_iyix(indices)
        try:
            return vecs.render_frame(iy, ix)
        except Exception as e:
            log.debug("render_frame(%d, %d) failed, showing blank: %s", iy, ix, e)
            return np.zeros((H, W), dtype=np.float32)

    sig_plots = set(getattr(tree, "signal_plots", []))
    npm = getattr(tree, "navigator_plot_manager", None)
    touched = set()
    if npm is not None:
        for sel in getattr(npm, "all_navigation_selectors", []):
            for child in list(getattr(sel, "children", {}).keys()):
                if child in sig_plots:
                    sel.children[child] = _fn
                    child.needs_auto_level = True
                    touched.add(sel)
    for sel in touched:
        try:
            sel.delayed_update_data(force=True)
        except Exception as e:
            log.debug("forcing navigator re-slice after find-vectors failed: %s", e)
    if not touched:                      # fallback: the lazy nav path
        _refresh_signal_from_navigator(tree)


def _overlay_on_result(tree, vecs) -> None:
    """Overlay the found vectors as red circle markers on the RESULT window's
    rendered diffraction pattern, tracking its count-map navigator (Qt parity:
    the computed-vectors window drew red circles over the rendered disks).
    Replaces any prior overlay so re-running doesn't stack markers."""
    from spyde.actions.vector_overlay import attach_vector_overlay
    old = getattr(tree, "_result_vector_overlay", None)
    if old is not None:
        try:
            old.remove()
        except Exception as e:
            log.debug("removing prior vector overlay failed: %s", e)
        tree._result_vector_overlay = None
    for sp in list(getattr(tree, "signal_plots", [])):
        try:
            tree._result_vector_overlay = attach_vector_overlay(sp, vecs, tree)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "result vector overlay attach failed: %s", e)


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
            except Exception as e:
                log.debug("navigator re-slice (lazy fallback) failed: %s", e)


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
                except Exception as e:
                    log.debug("dropping prior find-vectors preview failed: %s", e)
            tree._fv_preview = attach_find_vectors_preview(
                src, tree.root, tree, sigma=p["sigma"],
                kernel_radius=p["kernel_radius"], threshold=p["threshold"],
                min_distance=p["min_distance"], subpixel=p["subpixel"],
                method=p["method"], dog_sigma1=p["dog_sigma1"],
                dog_sigma2=p["dog_sigma2"],
                beamstop_mask=_preview_beamstop(tree, p),
            )
            # Qt parity: estimate the disk radius from the data (once) so the
            # wizard's defaults match the pattern instead of a fixed 5.
            if not getattr(tree, "_fv_auto_sent", False):
                tree._fv_auto_sent = True
                _emit_auto_params(src, tree)
            emit_status("Find Vectors: tune the parameters — peaks preview under "
                        "the crosshair, then Compute")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("fv_preview attach failed: %s", e)

    threading.Thread(target=_work, daemon=True, name="fv-preview").start()


def _preview_beamstop(tree, p):
    """Beam-stop mask for the live preview, cached on the tree. Detected from a
    sparse sample of the dataset (memory-safe) only when the user enabled
    beamstop_auto; None otherwise."""
    if not p.get("beamstop_auto"):
        return None
    cached = getattr(tree, "_fv_beamstop", None)
    if cached is not None:
        return cached
    try:
        from spyde.actions.find_vectors import _auto_beamstop_from_signal
        nav_dim = tree.root.axes_manager.navigation_dimension
        mask = _auto_beamstop_from_signal(tree.root, nav_dim)
        tree._fv_beamstop = mask
        return mask
    except Exception as e:
        log.debug("preview beam-stop detection failed: %s", e)
        return None


def _emit_auto_params(plot, tree) -> None:
    """Estimate the diffraction-disk radius (Qt's LoG blob detection) from a
    representative pattern and emit it so the wizard seeds its sliders — matching
    Qt, which auto-sizes per dataset rather than using a fixed radius."""
    try:
        from spyde.actions.find_vectors import _auto_params
        root = tree.root
        nav_dim = root.axes_manager.navigation_dimension
        nav_shape = tuple(root.data.shape[:nav_dim])
        idx = tuple(int(s) // 2 for s in nav_shape)        # centre pattern
        frame = root.data[idx]
        if hasattr(frame, "compute"):                      # lazy: one small frame
            frame = frame.compute()
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim != 2:
            return
        ap = _auto_params(frame)
        emit({"type": "fv_auto_params",
              "window_id": getattr(plot, "window_id", None),
              "kernel_radius": int(ap["kernel_radius"]),
              "min_distance": int(ap["min_distance"])})
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("fv auto-params failed: %s", e)


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
        except Exception as e:
            log.debug("removing find-vectors preview on stop failed: %s", e)
        tree._fv_preview = None

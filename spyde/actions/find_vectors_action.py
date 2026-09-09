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

import numpy as np
import hyperspy.api as hs

log = logging.getLogger(__name__)

from de_shell.ipc import emit, emit_status, emit_error
from spyde.backend import test_hold as _hold
from spyde.actions.context import src_plot_tree as _src_plot_tree, current_signal as _current_signal
from spyde.actions.find_vectors import _do_compute_vectors, _copy_nav_axes_to

# Defaults mirror the old Qt CaretGroup sliders, plus the detection method:
# "neural" (the SpotUNet disk detector — parameter-free, the default), "nxcorr"
# (window-normalised cross-correlation against a flat disk) and "dog" (band-pass,
# better for small 2-3 px spots / beam-stopped data — see the 3 nm benchmark).
# `threshold` is shared but means different things per method: neural uses the
# model's confidence (~0.3); NXCORR an [-1,1] correlation score (~0.5); DoG an
# absolute band-pass SNR (~10). `model_id` selects a registry model for the
# neural method ("" → the registry default). `bg_sigma` is the neural local-norm
# high-pass scale — auto-set by the one-shot calibration (fv_open emits
# `fv_calibration`) and threaded identically through preview AND batch.
DEFAULTS: dict = dict(
    # Nav blur defaults OFF (user decision 2026-07-16): the slider remains for
    # NXCORR/DoG (weak-signal data benefits) but starts at 0; for NEURAL it is
    # never applied at all (_coerce forces sigma=0 — the net is trained on
    # single frames).
    sigma=0.0, kernel_radius=5, threshold=0.3, min_distance=5, subpixel=True,
    method="neural", model_id="", bg_sigma=12.0, dog_sigma1=0.8, dog_sigma2=2.0,
    # Spot size (px RADIUS) for the neural canonical rescale: 0 = the model's
    # own autocorrelation estimate; the wizard always sends its auto-seeded
    # Spot-size slider so the UI knob is the single source of truth.
    spot_radius=0.0,
    # Neural stage-2 refine: drop peaks not confirmed by scan neighbours
    # (models/refine.py). BATCH-only (the preview has no neighbours) and
    # default-off until the eval benchmark says it should be on (plan Phase 3).
    persistence=False,
    beamstop_auto=False, beamstop_dilate=5, show_transform=False,
)

# Per-method threshold default applied when the user switches method without
# having explicitly set a threshold for it.
_METHOD_THRESHOLD = {"neural": 0.3, "nxcorr": 0.5, "dog": 10.0}


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
    p["method"] = str(p.get("method", DEFAULTS["method"])).lower()
    if p["method"] not in _METHOD_THRESHOLD:
        p["method"] = DEFAULTS["method"]
    # Thresholds are per-method scales (neural confidence ~0.3, NXCORR score
    # ~0.5, DoG SNR ~10); when the caller didn't set one explicitly, substitute
    # the method's own default so the first preview isn't empty/flooded.
    if params.get("threshold") in (None, ""):
        p["threshold"] = _METHOD_THRESHOLD[p["method"]]
    # Nav blur is NEVER applied for the neural method (user decision — the net
    # is trained on single frames; blur only smears the disks it was trained
    # to see). Forced here, the single choke point for wizard/toolbar/api.
    if p["method"] == "neural":
        p["sigma"] = 0.0
    return p


def find_diffraction_vectors(ctx, action_name: str = "Find Diffraction Vectors", **params):
    """Toolbar entry point (ActionContext convention) — one-shot batch compute."""
    plot = ctx.plot
    session = ctx.session
    src_tree = getattr(plot, "signal_tree", None)
    if src_tree is None or session is None:
        emit_error("Find Vectors: no active dataset")
        return None
    src = _current_signal(plot) or src_tree.root
    am = src.axes_manager
    if am.signal_dimension != 2 or am.navigation_dimension < 2:
        emit_error("Find Vectors needs a 4D-STEM dataset (2-D nav + 2-D signal)")
        return None
    return _start_batch(session, plot, src_tree, _coerce(params))


def _ensure_model_local(p: dict) -> None:
    """Resolve the neural model's weights to a LOCAL file before the compute is
    submitted, so dask workers never touch the network (a first-use HF-hosted
    model downloads once, with a status line, instead of N times concurrently).
    No-op for non-neural methods; on failure ``get_model``'s bundled-default
    fallback takes over on the workers."""
    if str(p.get("method", "")).lower() != "neural":
        return
    try:
        from spyde import models
        mid = p.get("model_id") or None
        if not models.is_cached(mid):
            emit_status(f"Find Vectors: downloading model {mid or 'default'}…")
        models.ensure_local(mid)
    except Exception as e:
        log.debug("ensure_local(%r) failed: %s", p.get("model_id"), e)


def _start_batch(session, plot, src_tree, p: dict, *, overlay_visible: bool = True):
    """Build the result window, then run the full-dataset compute on a background
    thread. Shared by the toolbar one-shot and the staged-wizard ``fv_run``
    (which passes ``overlay_visible=False`` — after Compute the source DP stays
    clean; reopening the caret toggles the overlay back via ``set_overlay``)."""
    src = _current_signal(plot) or src_tree.root
    am = src.axes_manager

    # Pin the CONCRETE model id for neural runs ("" = registry default) so the
    # provenance stamped below records exactly which model produced the vectors,
    # and workers keep loading the same model even if the user manifest is
    # refreshed mid-run.
    if p.get("method") == "neural" and not p.get("model_id"):
        try:
            from spyde.models import default_model_id
            p["model_id"] = default_model_id() or ""
        except Exception as e:
            log.debug("resolving default model id failed: %s", e)

    # Drop the live tuning preview (the final overlay replaces it) and the
    # prior run's persistent source-pattern overlay. The replace inside
    # _overlay_on_source only runs at the TAIL of the async batch, so without
    # this a second Compute leaves run 1's circles on the pattern all run.
    from spyde.actions.vector_overlay import (
        remove_find_vectors_preview, replace_tree_overlay,
    )
    remove_find_vectors_preview(src_tree, plot)
    replace_tree_overlay(src_tree, "_vector_overlay", None)

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
    base_title = src.metadata.get_item("General.title", "Signal")

    nav_sig = hs.signals.BaseSignal(np.zeros(nav_shape_full, dtype=np.float32)).T
    nav_sig.metadata.General.title = "Vector count map"
    _copy_nav_axes_to(src, nav_sig)

    from spyde.drawing.selectors import CrosshairSelector
    from spyde.actions.commit import open_result_tree
    new_tree = open_result_tree(
        session, title=f"{base_title} — Vectors", signal=new_sig,
        signal_type="spyde_diffraction_vectors_image",
        navigator_override=nav_sig, selector_type=CrosshairSelector,
        provenance={"action": "Find Diffraction Vectors",
                    "source_title": base_title,
                    "source_node": _node_name(src_tree, src), "params": dict(p)},
    )

    emit_status("Finding diffraction vectors…")

    # ── Progressive (live) count map: the compute writes per-chunk vector counts
    #    into a shared-memory buffer as chunks finish; a poller paints them into
    #    the count-map navigator so it fills in live instead of all-at-the-end.
    from spyde.actions.lifecycle import live_fill_poller
    from spyde.drawing.update_functions import ensure_live_buffer
    shm_name = f"spyde_fv_{id(plot)}"
    try:
        # Allocate the buffer with the FULL nav shape so 5D data gets a 3D
        # buffer (n_t, n_y, n_x); the compute's per-chunk callback writes each
        # chunk to its correct (t, y, x) location without time-axis collapsing.
        shm = ensure_live_buffer(nav_shape_full, shm_name)
    except Exception:
        shm, shm_name = None, None

    def _spatial_nav_plot():
        """The navigator plot whose displayed shape is the 2-D spatial grid
        (nav_shape_2d). For a 5-D stack the first nav plot is the OUTER (1-D
        stack) plot — painting the 2-D live count map there leaves the spatial
        navigator black AND wedges the time navigator, so there is deliberately
        NO fallback: no match means skip this paint and try again next poll.
        Uses _display_shape, which resolves before the first async paint lands
        (a 4-D tree has one nav plot and matches immediately either way)."""
        return _nav_plot_with_shape(new_tree, nav_shape_2d)

    def _paint(arr):
        nav_plot = _spatial_nav_plot()
        # For 5D (3D buffer), show the t=0 slice on the spatial navigator.
        display = arr[0] if arr.ndim == 3 else arr
        if nav_plot is not None and np.isfinite(display).any():
            nav_plot.needs_auto_level = True
            nav_plot.set_data(np.nan_to_num(display).astype(np.float32))

    # ── Progressive (live) SIGNAL plot: the result window is a navigator + a
    #    signal plot, and only the navigator used to come alive during the run —
    #    the diffraction pattern sat on its zero placeholder until _finalize.
    #    The client already receives every chunk's raw peaks block (that is what
    #    the count map above is derived from), so keep them and render from them:
    #    each landing block paints one sample position (so the DP visibly updates
    #    alongside the count map), and the navigator→signal slice function reads
    #    any ALREADY-computed position on demand. _finalize's real render display
    #    takes over at the end; preview.close() never clobbers it. See
    #    spyde/actions/live_signal.py.
    from spyde.actions.find_vectors import LiveVectorFrames
    from spyde.actions.live_signal import attach_signal_preview
    # (H, W) from the SIGNAL AXES, exactly like SpyDEDiffractionVectors.
    # render_frame (axis 0 = kx = columns, axis 1 = ky = rows) — so a
    # non-square detector previews at the same shape it finalizes at.
    live_frames = LiveVectorFrames(
        sig_hw=(int(am.signal_axes[1].size), int(am.signal_axes[0].size)),
        kernel_radius_px=float(p.get("kernel_radius", 5)),
    )
    preview = attach_signal_preview(
        session, new_tree, render=live_frames.render,
        nav_shape=nav_shape_full, name="fv-signal",
    )

    def _on_chunk_block(nav_slices, block):
        """Per-chunk hand-off from the compute (a Dask callback thread)."""
        if preview is None:
            return
        if live_frames.add(nav_slices, block):
            preview.note_block(nav_slices)
            # Optional deterministic pause so a test can inspect a PARTIALLY
            # filled result (see backend/test_hold.py). No-op unless
            # SPYDE_TEST_HOLD names this point; blocking this Dask callback
            # thread is what keeps the half-filled display up and interactive
            # while the asyncio main thread carries on serving the UI.
            if _hold.armed():
                _hold.hold_point("fv-batch", preview.ready_count,
                                 preview.n_positions)

    # Marks the batch as in-flight so a downstream action opened in the gap
    # (e.g. Strain Mapping) can tell "still computing, keep waiting" apart from
    # "nothing is running, give up" instead of guessing off a fixed timeout —
    # see lifecycle.wait_for_vectors.
    new_tree._fv_batch_running = True
    src_tree._fv_batch_running = True

    # Cancellation: register a stopped_flag on BOTH trees (compute reads the
    # source, results land on the new tree — closing either should stop it).
    # _do_compute_vectors polls this flag and cancels its dask futures.
    stopped_flag = [False]
    for _t in {id(src_tree): src_tree, id(new_tree): new_tree}.values():
        if hasattr(_t, "register_cancel"):
            _t.register_cancel(flag=stopped_flag)

    stop_poll = live_fill_poller(nav_shape_full, shm_name, _paint,
                                 interval=0.35, name="fv-poll")

    def _work():
        try:
            # Timing logs at INFO so a "batch looks stuck" report can be
            # localized from the app log: cluster compute vs finalize/paint.
            # (Chronic symptom under investigation: the batch sometimes only
            # completes after the user clicks a plot — i.e. after unrelated
            # navigator traffic reaches the same dask client.)
            import time as _time
            t0 = _time.monotonic()
            _ensure_model_local(p)       # download once HERE, not on N workers
            log.info("[fv-batch] compute starting (shm=%s)", shm_name)
            vecs = _do_compute_vectors(src, p, main_window=session,
                                       signal_tree=src_tree, shm_name=shm_name,
                                       on_chunk_block=_on_chunk_block,
                                       stopped_flag=stopped_flag)
            log.info("[fv-batch] compute returned in %.1fs (vecs=%s)",
                     _time.monotonic() - t0, "none" if vecs is None else "ok")
            stop_poll()                              # final paint owns the nav plot
            if vecs is None:
                # None also means "cancelled" (tree closed mid-compute) — stay
                # quiet in that case; only surface an error for a real failure.
                if not stopped_flag[0]:
                    emit_error("Find Vectors: compute returned no result")
                return
            _finalize(new_tree, vecs)
            log.info("[fv-batch] finalized in %.1fs total", _time.monotonic() - t0)
            _overlay_on_source(src_tree, vecs, visible=overlay_visible,
                               signal=src)
        except Exception as e:
            emit_error(f"Find Vectors failed: {e}")
            log.exception("Find Vectors compute failed")
        finally:
            stop_poll()
            # Hand the signal plot back. AFTER _finalize on purpose: close()
            # restores the placeholder slice function only while the preview's
            # own one is still installed, so the real render display that
            # _finalize just wired stays put and a cancelled/failed run still
            # gets its original slice function back.
            if preview is not None:
                preview.close()
            live_frames.clear()
            new_tree._fv_batch_running = False
            src_tree._fv_batch_running = False
            # Reset the workers' RSS accounting after the batch churn — see
            # dask_stats.trim_cluster_memory (spill thresholds act on process
            # memory, so this batch's allocator retention would otherwise eat
            # the NEXT compute's headroom).
            try:
                from spyde.backend.dask_stats import trim_cluster_memory
                trim_cluster_memory(session)
            except Exception as e:
                log.debug("post-batch cluster trim failed: %s", e)
            for _t in {id(src_tree): src_tree, id(new_tree): new_tree}.values():
                if hasattr(_t, "unregister_cancel"):
                    _t.unregister_cancel(flag=stopped_flag)
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    log.debug("shared-memory cleanup failed: %s", e)

    threading.Thread(target=_work, daemon=True, name="find-vectors").start()
    return None


def _node_name(tree, signal):
    """The tree node name of ``signal``, for provenance; None if not a node."""
    try:
        node = tree.get_node(signal)
    except Exception:
        node = None
    return getattr(node, "name", None)


def _overlay_on_source(src_tree, vecs, *, visible: bool = True,
                       signal=None) -> None:
    """Overlay the found vectors as live circle markers on the SOURCE
    diffraction pattern. Replaces any prior overlay from an earlier run so
    re-running Find Vectors doesn't stack markers. ``signal`` is the node the
    vectors were found on; the overlay draws only while a window displays it.

    ``visible=False`` (the wizard path) attaches it hidden: the pattern stays
    clean after Compute, and reopening the caret shows it again through the
    renderer's ``set_overlay`` toggle."""
    if src_tree is None:
        return
    from spyde.actions.vector_overlay import (
        attach_vector_overlay, replace_tree_overlay,
    )
    replace_tree_overlay(src_tree, "_vector_overlay", None)
    node = attach_vector_overlay(vecs, src_tree, signal=signal)
    src_tree._vector_overlay = node
    if not visible:
        src_tree.set_overlay_visible(node, False)


def _apply_axes_from_vecs(new_sig, nav_sig, vecs) -> None:
    """Calibrate the result signal's nav + signal axes from the vectors' stored
    axis records (used when there's no source signal — i.e. loading a saved
    vectors file)."""
    nav_axes = list(getattr(vecs, "nav_axes", []) or [])
    sig_axes = list(getattr(vecs, "sig_axes", []) or [])

    def _set(axes, recs):
        for ax, rec in zip(axes, recs):
            try:
                ax.scale = float(rec.scale)
                ax.offset = float(rec.offset)
                ax.units = str(getattr(rec, "units", "") or "")
                ax.name = str(getattr(rec, "name", "") or "")
            except Exception as e:
                log.debug("applying axis calibration failed: %s", e)

    # new_sig nav axes are in axes-manager order (matches navigation_axes order
    # used to build nav_axes); signal axes likewise.
    _set(new_sig.axes_manager.navigation_axes, nav_axes)
    _set(new_sig.axes_manager.signal_axes, sig_axes)
    _set(nav_sig.axes_manager.navigation_axes, nav_axes)


def build_vectors_result_tree(session, vecs, title: str = "Diffraction Vectors"):
    """Build a Find-Vectors *result tree* from a reconstructed
    :class:`SpyDEDiffractionVectors`, with no source dataset.

    This is the load-side twin of the tree the toolbar Find-Vectors action builds:
    a lazy zero-placeholder image (signal_type ``spyde_diffraction_vectors_image``)
    + a count-map navigator override, with ``tree.diffraction_vectors`` attached
    and the render-on-demand display wired (``vecs.render_frame`` per nav move).
    Used by Save→Load round-trip so a saved ``.zspy`` vectors file reopens exactly
    like a fresh Find-Vectors result (vector toolbar actions unlocked, calibrated
    scan grid, rendered disks).
    """
    import dask.array as da
    from spyde.drawing.selectors import CrosshairSelector

    full_nav_shape = tuple(int(s) for s in vecs.full_nav_shape)
    H = int(vecs.sig_axes[1].size)
    W = int(vecs.sig_axes[0].size)
    sig_hw = (H, W)
    data_shape = full_nav_shape + sig_hw
    nav_chunks = tuple(min(32, int(s)) for s in full_nav_shape)
    placeholder = da.zeros(
        data_shape, chunks=nav_chunks + sig_hw, dtype=np.float32,
    )
    new_sig = hs.signals.Signal2D(placeholder).as_lazy()

    nav_sig = hs.signals.BaseSignal(
        np.zeros(full_nav_shape, dtype=np.float32)).T
    nav_sig.metadata.General.title = "Vector count map"

    _apply_axes_from_vecs(new_sig, nav_sig, vecs)

    from spyde.actions.commit import open_result_tree
    tree = open_result_tree(
        session, title=title, signal=new_sig,
        signal_type="spyde_diffraction_vectors_image",
        navigator_override=nav_sig, selector_type=CrosshairSelector,
        provenance={"action": "Find Diffraction Vectors", "source": "loaded"},
    )
    _finalize(tree, vecs)
    return tree


def _finalize(tree, vecs) -> None:
    """Attach the vectors, fill the count-map navigator, wire the render-on-demand
    display, and unlock the vector toolbar actions.

    The result window's frames are produced on every navigator move by
    ``vecs.render_frame`` (an O(1) slice of the vector buffer; see
    :class:`RenderedVectorsReader`), not by reading the signal's lazy data. So
    the root keeps its cheap zero placeholder array, with the right shape and
    axes for the window, and no rendered-frame dask graph is ever built or
    stored, which is what makes Save tiny: the vectors are serialised, not the
    frames they draw."""
    # The signal plot's CachedDaskArray captured the placeholder array when the
    # window first rendered (zeros). Drop it so the render-display re-slice paints
    # the disk frames instead of the cached zeros.
    try:
        tree.root.cached_dask_array = None
        tree.root._clear_cache_dask_data()
    except Exception as e:
        log.debug("clearing stale cached dask array failed: %s", e)
    from spyde.actions.lifecycle import attach_container
    attach_container(tree, vecs, name="diffraction_vectors")

    # Paint the count map onto the SPATIAL (2-D) navigator plot. For a 5-D stack
    # the navigator is multi-level; _first_nav_plot may return the OUTER (1-D
    # stack) plot, and a 2-D count map painted there mismatches → the navigator
    # stays black. Find the nav plot whose displayed shape is the 2-D spatial grid
    # and paint that one; for a stack use the current slice's per-slice counts so
    # it's meaningful (not the stack-summed map).
    spatial_2d = vecs.nav_shape                                  # (nav_y, nav_x)
    if vecs.n_time > 0:
        count_map = vecs.count_map_at_t(0).astype(np.float32)    # slice 0 to start
    else:
        count_map = vecs.count_map().astype(np.float32)
    # For a stack, the vectors-per-slice curve — what the 1-D TIME navigator
    # should show. Without it that window stayed on whatever it held (an
    # all-zero flat line), so a 5-D result gave no indication of how many
    # vectors each slice held, or which slice you were looking at.
    per_slice = None
    if vecs.n_time > 0:
        try:
            series = np.asarray(vecs.count_map_series(), dtype=np.float32)
            per_slice = series.reshape(series.shape[0], -1).sum(axis=1)
        except Exception as e:
            log.debug("building the per-slice vector counts failed: %s", e)

    if vecs.n_time > 0:
        # Logged at INFO because it is the signal the 5-D e2e reads; see the note
        # on the time-nav paint. Shapes come from _display_shape, so an unpainted
        # plot reports its real display shape rather than ().
        log.info("[fv-5d] nav plots: %s | navsigs=%s | spatial=%s n_time=%d",
                 [_display_shape(p) for p in _all_nav_plots(tree)],
                 list(getattr(tree, "navigator_signals", {}) or {}),
                 tuple(spatial_2d), vecs.n_time)

    painted = False
    for nav_plot in _all_nav_plots(tree):
        try:
            exp = _display_shape(nav_plot)
            if exp is not None and tuple(exp) == tuple(spatial_2d):
                nav_plot.needs_auto_level = True
                nav_plot.set_data(count_map)
                painted = True
            elif (per_slice is not None and exp is not None
                    and tuple(exp) == (int(per_slice.shape[0]),)):
                # The 1-D stack navigator: total vectors in each slice.
                nav_plot.needs_auto_level = True
                nav_plot.set_data(per_slice)
                # INFO, not debug: this is the signal the 5-D e2e waits on (a
                # 1-D line plot's values can't be read back from a screenshot).
                log.info("[fv-5d] time-nav painted: %d slices, totals %s",
                         int(per_slice.shape[0]),
                         [int(x) for x in per_slice[:8]])
        except Exception as e:
            log.debug("painting final count map onto navigator failed: %s", e)
    if not painted and per_slice is None:
        # Fallback for a 4-D tree only: one navigator, shape unresolvable.
        # NEVER for a stack — its first nav plot is the 1-D time line, and
        # pushing a 2-D count map there is what left it blank (see _display_shape).
        np0 = _first_nav_plot(tree)
        if np0 is not None:
            try:
                np0.needs_auto_level = True
                np0.set_data(count_map)
            except Exception as e:
                log.debug("fallback count-map paint failed: %s", e)

    # Re-send the toolbar config so the now-available vector actions appear.
    for sp in list(getattr(tree, "signal_plots", [])):
        try:
            sp.needs_auto_level = True
            state = getattr(sp, "plot_state", None)
            if state is not None and hasattr(state, "_send_toolbar_config"):
                state._send_toolbar_config()
        except Exception as e:
            log.debug("re-sending toolbar config after find-vectors failed: %s", e)
    _install_result_readers(tree, vecs)
    _overlay_on_result(tree, vecs)
    _attach_time_slice_repaint(tree, vecs)

    total = int(len(vecs.flat_buffer))   # total over ALL slices, not one slice
    emit_status(f"Found {total} diffraction vectors")


def _split_nav_index(index):
    """A prepared navigation index as ``(time, iy, ix)``.

    The spatial pair is the last two coordinates in data order; a leading
    coordinate is the stack slice a 5-D scan's vectors were found in, and a
    4-D scan has none.
    """
    values = tuple(int(v) for v in np.atleast_1d(np.asarray(index)).ravel())
    if len(values) < 2:
        return None, 0, 0
    return (int(values[0]) if len(values) > 2 else None), values[-2], values[-1]


class RenderedVectorsReader:
    """Frames of a Find Vectors result window, drawn from the vector store.

    The window's root is a zero placeholder with the right shape and axes; the
    frame at a position is that position's vectors drawn as flat disks, an
    O(1) slice of the compact vector buffer. A region is the store's own rule:
    each position's disks are drawn with the intra-frame maximum and those
    frames are summed.
    """

    def __init__(self, vecs):
        self.vecs = vecs
        self.frame_shape = (int(vecs.sig_axes[1].size), int(vecs.sig_axes[0].size))

    @property
    def frame_bytes(self) -> int:
        return int(np.prod(self.frame_shape)) * np.dtype(np.float32).itemsize

    def read_frame(self, indices):
        slice_index, iy, ix = _split_nav_index(indices)
        try:
            return self.vecs.render_frame(iy, ix, t=slice_index)
        except Exception as e:
            log.debug("rendering the vectors at (%s, %s, t=%s) failed: %s",
                      iy, ix, slice_index, e)
            return np.zeros(self.frame_shape, dtype=np.float32)

    def sum_points(self, points, dtype):
        """The region's rendered frame, scaled by its point count.

        A region read divides the accumulator it is given by the number of
        points, and a vectors region is a sum rather than a mean, so the sum
        is scaled up to survive that divide. float64 makes the scale and the
        divide exact for any frame the store renders."""
        points = np.asarray(points)
        slice_index = int(points[0][0]) if points.shape[1] > 2 else None
        rows = points[:, -2].astype(np.int64)
        columns = points[:, -1].astype(np.int64)
        region = self.vecs.render_region(
            int(rows.min()), int(rows.max()) + 1,
            int(columns.min()), int(columns.max()) + 1, t=slice_index)
        return np.asarray(region, dtype=np.float64) * float(len(points))


class CountMapReader:
    """The vector count map of one slice of a stack, for the real-space
    navigator of a Find Vectors result window.

    That navigator's own array is a placeholder of zeros the time axis would
    otherwise slice, so scrubbing time showed zeros over the map.
    """

    def __init__(self, vecs):
        self.vecs = vecs
        self.shape = tuple(int(s) for s in vecs.nav_shape)
        self.n_time = int(getattr(vecs, "n_time", 0) or 0)

    @property
    def frame_bytes(self) -> int:
        return int(np.prod(self.shape)) * np.dtype(np.float32).itemsize

    def _slice(self, t: int) -> int:
        return max(0, min(int(t), self.n_time - 1))

    def read_frame(self, indices):
        values = np.atleast_1d(np.asarray(indices)).ravel()
        t = self._slice(values[-1] if values.size else 0)
        try:
            return np.asarray(self.vecs.count_map_at_t(t), dtype=np.float32)
        except Exception as e:
            log.debug("the count map of slice %s failed to build: %s", t, e)
            return np.zeros(self.shape, dtype=np.float32)

    def sum_points(self, points, dtype):
        """The count maps of the slices a span covers, summed and scaled by
        the point count (see :meth:`RenderedVectorsReader.sum_points`)."""
        points = np.asarray(points)
        slices = sorted({self._slice(row[-1]) for row in points})
        total = np.zeros(self.shape, dtype=np.float64)
        for t in slices:
            total += np.asarray(self.vecs.count_map_at_t(t), dtype=np.float64)
        return total * float(len(points))


def _install_result_readers(tree, vecs) -> None:
    """Read the result window's frames from the vector store.

    The diffraction pattern is the navigated position's disks and, for a
    stack, the real-space navigator is that slice's count map. Both are
    pinned on the tree as reader overrides, so every window of the tree draws
    them, including a signal plot opened later by "Add Selector"."""
    tree.set_reader_override(tree.root, RenderedVectorsReader(vecs))
    for plot in list(getattr(tree, "signal_plots", [])):
        plot.needs_auto_level = True
    if int(getattr(vecs, "n_time", 0) or 0) > 0:
        spatial_2d = tuple(int(s) for s in vecs.nav_shape)
        for nav_plot in _all_nav_plots(tree):
            if not getattr(nav_plot, "is_navigator", False):
                continue
            if _display_shape(nav_plot) != spatial_2d:
                continue
            state = getattr(nav_plot, "plot_state", None)
            signal = getattr(state, "current_signal", None)
            if signal is not None:
                tree.set_reader_override(signal, CountMapReader(vecs))
                nav_plot.needs_auto_level = True
    _refresh_signal_from_navigator(tree)


def _attach_time_slice_repaint(tree, vecs) -> None:
    """Repaint the 2-D count map when the TIME axis moves (5-D stacks only).

    The count map a navigator shows belongs to one slice, so it has to follow
    the time axis wherever that axis is driven from. This rides the navigator
    ``index_hooks`` the vector overlay follows, and hands the map to the plot's
    painter thread like every other frame. Idempotent: re-running Find Vectors
    removes the previous hook first, or a second run would paint twice per move.
    """
    from spyde.actions.vector_overlay import (
        _indices_lead_nav, _navigator_selectors_for,
    )

    _detach_time_slice_repaint(tree)
    if getattr(vecs, "n_time", 0) <= 0:
        return                                   # 4-D: nothing to slice
    spatial_2d = tuple(int(s) for s in vecs.nav_shape)
    n_t = int(vecs.n_time)
    state = {"t": 0}

    def _targets():
        out = []
        for nav_plot in _all_nav_plots(tree):
            cur = getattr(nav_plot, "current_data", None)
            exp = tuple(cur.shape) if hasattr(cur, "shape") else None
            if exp is not None and exp == spatial_2d:
                out.append(nav_plot)
        return out

    def _on_indices(indices):
        lead = _indices_lead_nav(indices)
        if not lead:
            return
        t = int(lead[0])
        if not (0 <= t < n_t) or t == state["t"]:
            return                               # same slice — nothing to redo
        state["t"] = t
        try:
            cm = np.asarray(vecs.count_map_at_t(t), dtype=np.float32)
        except Exception as e:
            log.debug("count_map_at_t(%s) failed: %s", t, e)
            return
        n_painted = 0
        for nav_plot in _targets():
            try:
                nav_plot.needs_auto_level = True
                nav_plot.enqueue_paint(cm)
                n_painted += 1
            except Exception as e:
                log.debug("repainting the count map for t=%s failed: %s", t, e)
        # INFO because it is the only honest handle the e2e has on "the map
        # followed the time axis".
        log.info("[fv-5d] count map -> slice %d (%d plot(s))", t, n_painted)

    hooked = []
    for sp in list(getattr(tree, "signal_plots", [])):
        for sel in _navigator_selectors_for(tree, sp):
            if _on_indices not in sel.index_hooks:
                sel.index_hooks.append(_on_indices)
                hooked.append(sel)
    tree._vectors_time_repaint = (_on_indices, hooked)


def _detach_time_slice_repaint(tree) -> None:
    """Drop a previous run's time-slice hook (see the idempotence note above)."""
    prev = getattr(tree, "_vectors_time_repaint", None)
    if not prev:
        return
    fn, sels = prev
    for sel in sels:
        try:
            if fn in sel.index_hooks:
                sel.index_hooks.remove(fn)
        except Exception as e:
            log.debug("detaching the time-slice repaint failed: %s", e)
    tree._vectors_time_repaint = None


def _overlay_on_result(tree, vecs) -> None:
    """Overlay the found vectors as red circle markers on the RESULT window's
    rendered diffraction pattern, tracking its count-map navigator. Replaces
    any prior overlay so re-running doesn't stack markers.

    One node covers every window the result tree opens, including a signal plot
    added later by "Add Selector"."""
    from spyde.actions.vector_overlay import (
        attach_vector_overlay, replace_tree_overlay,
    )
    replace_tree_overlay(tree, "_result_vector_overlay", None)
    tree._result_vector_overlay = attach_vector_overlay(vecs, tree)


def _first_nav_plot(tree):
    npm = getattr(tree, "navigator_plot_manager", None)
    if npm is None:
        return None
    for pw in list(npm.plot_windows.keys()):
        plots = npm.plots.get(pw, [])
        if plots:
            return plots[0]
    return None


def _all_nav_plots(tree):
    """All navigator plots across every navigator window (a 5-D stack has more
    than one: an outer stack navigator + the 2-D spatial count map).

    ``MultiplotManager.plot_windows`` is a dict of PARENT window -> its
    SUB-windows (see ``add_plot_states_for_navigation_signals``, which fills the
    top level from ``signals[0]`` and the nested level from ``signals[1]``).
    Iterating only ``.keys()`` therefore returned ONE level — which is why this
    function, despite its docstring, handed back a single plot for a 5-D stack
    and the time navigator was never found, let alone painted."""
    npm = getattr(tree, "navigator_plot_manager", None)
    if npm is None:
        return []
    out, seen = [], set()

    def _add(pw):
        for p in npm.plots.get(pw, []) or []:
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)

    for pw, subs in list(npm.plot_windows.items()):
        _add(pw)
        for sub in list(subs or []):
            _add(sub)
    return out


def _display_shape(plot):
    """The numpy shape this plot DISPLAYS, or None if it can't be resolved.

    ``current_data`` is the honest answer once the plot has painted — but the
    first paint is dispatched ASYNCHRONOUSLY on the nav thread, so for the first
    fraction of a second after a window opens it is still None. Matching a
    navigator purely on ``current_data.shape`` therefore MISSES during exactly
    the window in which Find-Vectors starts its live count-map poller, and the
    "no match" fallback then painted a 2-D count map onto the 1-D TIME navigator
    of a 5-D stack (set_data raises on the shape mismatch, leaving that plot's
    ``current_data`` None forever — which is why the time navigator stayed a
    flat zero line even after the batch finished).

    So fall back to the plot state's own signal: its SIGNAL axes are what the
    figure draws, and they exist the moment the state is created. ``signal_shape``
    is fastest-axis-first, so reverse it for numpy row-major order."""
    cur = getattr(plot, "current_data", None)
    if hasattr(cur, "shape"):
        return tuple(int(s) for s in cur.shape)
    am = getattr(getattr(getattr(plot, "plot_state", None),
                         "current_signal", None), "axes_manager", None)
    if am is None:
        return None
    try:
        return tuple(int(s) for s in reversed(am.signal_shape))
    except Exception as e:
        log.debug("resolving plot display shape failed: %s", e)
        return None


def _nav_plot_with_shape(tree, want) -> "object | None":
    """The navigator plot that displays exactly ``want`` — or None. Never
    substitutes a differently-shaped plot (see :func:`_display_shape`)."""
    want = tuple(int(s) for s in want)
    for npl in _all_nav_plots(tree):
        if _display_shape(npl) == want:
            return npl
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
# Staged "wizard" workflow: a live found-peaks PREVIEW on the source pattern
# while you tune the sliders, then Compute runs the full-dataset batch. The
# preview overlay lives on the source tree as `_fv_preview`.
# ─────────────────────────────────────────────────────────────────────────────

def fv_open(session, plot, payload) -> None:
    """'Tune' step: attach the LIVE found-peaks preview to the source pattern
    so the red circles update as you tune the sliders or move the navigator.
    Idempotent: it replaces any existing preview."""
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Find Vectors: no active dataset")
        return
    source = _current_signal(src) or tree.root
    am = source.axes_manager
    if am.signal_dimension != 2 or am.navigation_dimension < 2:
        emit_error("Find Vectors needs a 4D-STEM dataset (2-D nav + 2-D signal)")
        return
    p = _coerce(payload)
    log.debug("[fv-preview] ATTACH method=%s thr=%s show_transform=%s beamstop=%s "
              "data.shape=%s lazy=%s", p["method"], p["threshold"],
              p.get("show_transform"), p.get("beamstop_auto"),
              tuple(source.data.shape), getattr(source, "_lazy", "?"))

    # Run/stop generation guard (React StrictMode mounts the wizard twice
    # synchronously: fv_open, fv_close, fv_open — before either worker
    # lands). Bumped here BEFORE the worker spawns; fv_close bumps it too, so a
    # superseded preview attach is dropped instead of stacking a second overlay.
    from spyde.actions.lifecycle import bump_generation, is_current
    gen = bump_generation(tree, "_fv_run_gen")

    def _work():
        try:
            from spyde.actions.vector_overlay import (
                attach_find_vectors_preview, remove_overlay_node,
                replace_tree_overlay,
            )
            if not is_current(tree, "_fv_run_gen", gen):
                return                     # superseded by fv_close / newer preview
            if p["method"] == "neural":
                _ensure_model_local(p)   # a first-use HF model downloads here,
                                         # not inside the preview's frame compute
            new_prev = attach_find_vectors_preview(src, source, tree, p)
            # Superseded while attaching (fv_close or a newer fv_open bumped the
            # generation after the check above): tear down what was just
            # attached instead of installing a stale overlay.
            if not is_current(tree, "_fv_run_gen", gen):
                remove_overlay_node(tree, new_prev)
                return
            old = getattr(tree, "_fv_preview", None)
            if old is not None and old is not new_prev:
                remove_overlay_node(tree, old)
            tree._fv_preview = new_prev
            # The live preview supersedes any persistent overlay from an
            # earlier Compute: both drawing at once duplicates every peak.
            replace_tree_overlay(tree, "_vector_overlay", None)
            # Estimate the disk radius from the data, once, so the wizard's
            # defaults match the pattern instead of a fixed 5.
            if not getattr(tree, "_fv_auto_sent", False):
                tree._fv_auto_sent = True
                _emit_auto_params(src, tree)
            emit_status("Find Vectors: tune the parameters — peaks preview under "
                        "the crosshair, then Compute")
            # One-shot neural auto-calibration (bg_sigma / threshold): runs on
            # this worker AFTER the preview is up, cached on the tree, emitted
            # to the wizard, which adopts the values and re-tunes — so preview
            # and batch run with identical, dataset-tuned parameters.
            if p["method"] == "neural":
                try:
                    _emit_calibration(src, tree, p, gen)
                except Exception as e:
                    log.debug("[fv-cal] auto-calibration failed: %s", e)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("fv_open attach failed: %s", e)

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="fv-preview")


def _emit_auto_params(plot, tree) -> None:
    """Estimate the diffraction-disk radius (Qt's LoG blob detection) from a
    representative pattern and emit it so the wizard seeds its sliders — matching
    Qt, which auto-sizes per dataset rather than using a fixed radius."""
    try:
        from spyde.actions.find_vectors import _auto_params
        root = _current_signal(plot) or tree.root
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


# Skip auto-calibration on very large signal frames: 8 candidate σ × a few
# frames of CPU forward passes is seconds on a ≤512² DP but minutes on a 4k²
# frame — and find-vectors targets diffraction patterns, not full-frame movies.
_CAL_MAX_FRAME_PX = 1024 * 1024


def _calibration_frames(root, n: int = 3) -> list:
    """A few representative diffraction patterns spread across the scan —
    each a single small per-frame read (never the full dataset)."""
    nav_dim = root.axes_manager.navigation_dimension
    nav_shape = tuple(int(s) for s in root.data.shape[:nav_dim])
    frames, seen = [], set()
    for f in (0.25, 0.5, 0.75)[:max(1, int(n))]:
        idx = tuple(min(s - 1, max(0, int(round(s * f)))) for s in nav_shape)
        if idx in seen:
            continue
        seen.add(idx)
        frame = root.data[idx]
        if hasattr(frame, "compute"):
            frame = frame.compute()
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim == 2:
            frames.append(frame)
    return frames


def _emit_calibration(plot, tree, p: dict, gen) -> None:
    """One-shot neural auto-calibration for this dataset (see
    ``find_vectors_neural.calibrate_neural``): optimise ``bg_sigma`` (diffuse /
    beam-stopped backgrounds) and lower the threshold for faint-peak data.
    Cached on the tree (reopening the caret re-emits without recomputing);
    emitted as ``fv_calibration`` for the wizard to adopt (user-overridable).
    Generation-guarded like the preview attach — a closed wizard gets nothing."""
    from spyde.actions.lifecycle import is_current

    cal = getattr(tree, "_fv_calibration", None)
    if cal is None:
        root = _current_signal(plot) or tree.root
        sig_shape = root.axes_manager.signal_shape
        if int(sig_shape[0]) * int(sig_shape[1]) > _CAL_MAX_FRAME_PX:
            log.debug("[fv-cal] signal frame too large — keeping defaults")
            return
        frames = _calibration_frames(root)
        if not frames:
            return
        from spyde.actions.find_vectors_neural import calibrate_neural
        cal = calibrate_neural(frames, sigma=p.get("sigma", 0.0),
                               model_id=p.get("model_id") or None,
                               spot_radius=p.get("spot_radius") or None)
        tree._fv_calibration = cal
    if not is_current(tree, "_fv_run_gen", gen):
        return                       # wizard closed / superseded while calibrating
    msg = {"type": "fv_calibration",
           "window_id": getattr(plot, "window_id", None),
           "bg_sigma": float(cal["bg_sigma"]),
           "thresh": float(cal["thresh"]),
           "scale_factor": float(cal.get("scale_factor", 1.0))}
    conf = cal.get("confidence")
    if conf is not None and np.isfinite(conf):   # NaN is not valid JSON
        msg["confidence"] = float(conf)
    emit(msg)


def fv_tune(session, plot, payload) -> None:
    """'Tune' step: live-update the preview sliders and redraw the found peaks at
    the current crosshair position."""
    src, tree = _src_plot_tree(session, plot)
    prev = getattr(tree, "_fv_preview", None) if tree is not None else None
    coerced = _coerce(payload)
    log.info("[fv-tune] RECV plot=%s tree=%s preview=%s | thr=%s md=%s kr=%s method=%s",
             getattr(plot, "window_id", None), tree is not None,
             "present" if prev is not None else "MISSING",
             coerced.get("threshold"), coerced.get("min_distance"),
             coerced.get("kernel_radius"), coerced.get("method"))
    if prev is None:
        log.info("[fv-tune] DROPPED — no _fv_preview on tree (params not applied)")
        return

    def _work():
        from spyde.actions.vector_overlay import tune_find_vectors_preview
        try:
            tune_find_vectors_preview(tree, prev, src, coerced)
            log.info("[fv-tune] parameters APPLIED thr=%s md=%s kr=%s",
                     coerced.get("threshold"), coerced.get("min_distance"),
                     coerced.get("kernel_radius"))
        except Exception as e:
            log.exception("[fv-tune] applying the parameters FAILED: %s", e)

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="fv-tune")


def fv_run(session, plot, payload) -> None:
    """'Compute' step: full-dataset batch with the tuned params → a new vectors
    image window. Drops the live preview first."""
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Find Vectors: no active dataset")
        return
    source = _current_signal(src) or tree.root
    am = source.axes_manager
    if am.signal_dimension != 2 or am.navigation_dimension < 2:
        emit_error("Find Vectors needs a 4D-STEM dataset (2-D nav + 2-D signal)")
        return
    p = _coerce(payload)
    log.debug("[fv-run] COMPUTE full dataset method=%s thr=%s md=%s "
              "kr=%s dog=(%s,%s) beamstop=%s", p["method"], p["threshold"],
              p["min_distance"], p["kernel_radius"], p["dog_sigma1"],
              p["dog_sigma2"], p.get("beamstop_auto"))
    # The wizard caret closes on Compute; attach the final source-DP overlay
    # hidden so the pattern is clean until the caret is reopened.
    _start_batch(session, src, tree, p, overlay_visible=False)


def fv_models(session, plot, payload) -> None:
    """Emit the available neural models for the wizard's Model dropdown.

    Payload: ``{type: "fv_models", window_id, default, models: [{id, label,
    version, notes}]}`` — straight from the model registry (bundled manifest
    merged with any user-installed models)."""
    from spyde.models import available_models
    msg = {"type": "fv_models",
           "window_id": (payload or {}).get("window_id",
                                            getattr(plot, "window_id", None))}
    msg.update(available_models())
    emit(msg)


def fv_refresh_models(session, plot, payload) -> None:
    """'Check for new models': pull the latest ``registry.json`` from Hugging
    Face (the ship-a-model-without-re-releasing path — the author-side contract
    is in ``spyde/models/registry.py``'s module docstring) on a worker thread — never the main loop (network) — then re-emit
    ``fv_models`` with ``refreshed: true`` so the wizard dropdown updates in
    place. Offline-safe: a failed refresh keeps the current merged manifest."""
    window_id = (payload or {}).get("window_id", getattr(plot, "window_id", None))

    def _work():
        try:
            from spyde import models
            avail = models.refresh_remote_registry()
            msg = {"type": "fv_models", "window_id": window_id, "refreshed": True}
            msg.update(avail)
            emit(msg)
        except Exception as e:
            log.debug("fv_refresh_models failed: %s", e)

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="fv-refresh-models")


def fv_close(session, plot, payload=None) -> None:
    """Caret closed: remove the live preview overlay."""
    src, tree = _src_plot_tree(session, plot)
    if tree is not None:
        # Invalidate any fv_open still in flight FIRST (StrictMode fires
        # preview/stop/preview synchronously — see fv_open's gen guard).
        from spyde.actions.lifecycle import bump_generation
        bump_generation(tree, "_fv_run_gen")
    prev = getattr(tree, "_fv_preview", None) if tree is not None else None
    log.debug("[fv-stop] removing preview=%s", prev is not None)
    if prev is not None:
        from spyde.actions.vector_overlay import remove_find_vectors_preview
        remove_find_vectors_preview(tree, src)


def test_hold_release(session, plot, payload=None) -> None:
    """Release a deterministic test hold (``SPYDE_TEST_HOLD``) by name.

    Test-only wiring, and inert in production: with the env var unset there is
    no hold to release and this reports as much. See backend/test_hold.py.
    """
    name = str((payload or {}).get("name") or "fv-batch")
    ok = _hold.release(name)
    emit_status(f"test hold {name}: {'released' if ok else 'not configured'}")

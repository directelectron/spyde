"""
vector_orientation_om.py — Electron-native Vector Orientation Mapping.

REUSES the staged Orientation-Mapping wizard pattern (see ``orientation_action``):
a CIF → template-library → whole-field fit flow, but driven by the sparse-vector
matcher (`vector_orientation.compute_vector_orientation`) on a tree's
``diffraction_vectors`` instead of the dense template match. Produces an
orientation IPF-Z map plus εxx / εyy / εxy strain maps.

Staged handlers (mirroring `om_generate_library` / `om_run`):
  vom_generate_library — load the .cif, build the diffsims simulation + the
                         per-template g-vector library, cache on the tree.
  vom_run              — fit every position → IPF-Z + 3 strain map windows.

No Qt: the heavy Qt caret lives in `vector_orientation_action.py` (pyqtgraph);
this module is import-safe in the Electron backend.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
import hyperspy.api as hs

from spyde.backend.ipc import emit, emit_status, emit_error
from spyde.actions.context import src_plot_tree as _src_plot_tree

log = logging.getLogger(__name__)

DEFAULTS = dict(
    accelerating_voltage=200.0,
    resolution=1.0,
    minimum_intensity=1e-4,
    strain_cap=0.05,
    smooth=False,
)


def _reciprocal_radius(signal) -> float:
    ax = signal.axes_manager.signal_axes
    return float(min(a.scale * a.size / 2.0 for a in ax))


def vector_orientation_mapping(ctx, action_name: str = "Vector Orientation Mapping", **kwargs):
    """Parent toolbar action — a no-op; the Electron toolbar opens the staged
    Vector Orientation wizard (which drives the ``vom_*`` handlers) instead."""
    return None


def vom_generate_library(session, plot, payload) -> None:
    """'Generate Library': load the .cif, build the diffsims simulation and the
    per-template g-vector library used by the sparse-vector fit. Cached on the
    source tree as ``_vom_wizard``; emits ``vom_library_ready``."""
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Vector Orientation: no active dataset")
        return
    if getattr(tree, "diffraction_vectors", None) is None:
        emit_error("Vector Orientation: run Find Diffraction Vectors first")
        return
    cif_path = payload.get("cif_path")
    if not cif_path:
        emit_error("Vector Orientation: choose a .cif crystal first")
        return
    voltage = float(payload.get("accelerating_voltage", DEFAULTS["accelerating_voltage"]))
    resolution = float(payload.get("resolution", DEFAULTS["resolution"]))
    min_int = float(payload.get("minimum_intensity", DEFAULTS["minimum_intensity"]))
    emit_status("Vector Orientation: generating template library…")
    # Warm the CUDA autograd engine on this (dispatch) thread so the batched GPU
    # field fit below — run on the worker thread — is safe (no-op on MPS/CPU).
    try:
        from spyde.actions.vector_orientation_gpu import warmup_autograd
        warmup_autograd()
    except Exception as e:
        log.debug("CUDA autograd warmup failed: %s", e)

    def _work():
        try:
            from orix.crystal_map import Phase
            from spyde.actions.orientation_compute import generate_library_from_phases
            from spyde.actions.vector_orientation import build_template_library
            root = tree.root
            vecs = tree.diffraction_vectors
            phase = Phase.from_cif(cif_path)
            recip_r = _reciprocal_radius(root)
            sim = generate_library_from_phases(
                [phase], accelerating_voltage=voltage, resolution=resolution,
                minimum_intensity=min_int, reciprocal_radius=recip_r,
            )
            lib = build_template_library(sim, root, r_max=recip_r)
            n_templates = len(lib.spots_xy)

            # Activate the LIVE refine overlay: the fitted template (green) over
            # the measured vectors (red) under the crosshair (Qt parity). The
            # overlay's on_fit callback streams the strain/residual readout to
            # the wizard's Refine tab as the crosshair (or a slider) moves.
            overlay = None
            old = getattr(tree, "_vom_wizard", None)
            if old is not None and old.get("overlay") is not None:
                try:
                    old["overlay"].remove()
                except Exception as e:
                    log.debug("removing prior vom overlay failed: %s", e)
            wid = getattr(src, "window_id", None)
            try:
                from spyde.actions.vector_overlay import attach_vector_orientation_overlay
                overlay = attach_vector_orientation_overlay(
                    src, vecs, lib, tree, on_fit=lambda fit: _emit_vom_fit(wid, fit))
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("vom overlay attach failed: %s", e)

            tree._vom_wizard = {
                "phase": phase, "sim": sim, "lib": lib, "overlay": overlay,
                "voltage": voltage, "recip_r": recip_r,
                "strain_cap": DEFAULTS["strain_cap"], "sink_bw": None,
            }
            emit_status(f"Vector Orientation: library ready ({n_templates} templates) "
                        f"— computing live IPF map…")
            emit({"type": "vom_library_ready",
                  "window_id": getattr(src, "window_id", None),
                  "n_templates": n_templates})

            # LIVE IPF heatmap (Qt parity, "super nice"): fit the WHOLE field on
            # the GPU right away — it's seconds on a real scan — and show the
            # IPF-Z map immediately so you see the orientation result while you
            # refine. Cached on the tree so Compute Maps is then instant.
            field = _fit_field(vecs, lib, dict(strain_cap=DEFAULTS["strain_cap"]))
            if field is not None:
                tree._vom_field = field
                tree._vom_field_cap = DEFAULTS["strain_cap"]
                tree._vom_wizard["field"] = field
                _build_ipf_heatmap(session, root, field)
                emit_status("Vector Orientation: live IPF ready — refine, or "
                            "Compute Maps for the strain maps")
        except Exception as e:
            emit_error(f"Generate Library failed: {e}")
            log.exception("Generate Library failed")

    threading.Thread(target=_work, daemon=True, name="vom-generate-library").start()


def _build_ipf_heatmap(session, src, result, title="Orientation (IPF-Z, live)"):
    """Open just the IPF-Z map window (the live refine heatmap) + its 3-D
    explorer. The strain windows are added later by Compute Maps."""
    ny, nx = result.nav_shape
    base = src.metadata.get_item("General.title", "Signal")
    new_sig = hs.signals.Signal2D(np.zeros((ny, nx), dtype=np.float32))
    new_sig.metadata.General.title = f"{base} — {title}"
    tree = session._add_signal(new_sig)
    tree.vector_orientation = result
    try:
        ipf = result.ipf_color_map("z")
        for sp in list(getattr(tree, "signal_plots", [])):
            sp.needs_auto_level = True
            sp.set_data(ipf)
        from spyde.actions.ipf_view import attach_ipf_3d, attach_ipf_point_selector
        attach_ipf_3d(tree, result, "z")
        attach_ipf_point_selector(tree, result, "z")
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("ipf heatmap build failed: %s", e)
    return tree


def _emit_vom_fit(window_id, fit) -> None:
    """Stream the live single-pattern fit metrics to the wizard's Refine tab
    (Qt parity: εxx/εyy/εxy + residual + Friedel asymmetry + matched count)."""
    if fit is None:
        emit({"type": "vom_fit", "window_id": window_id, "ok": False})
        return
    import numpy as _np
    fr = fit.friedel_asym
    emit({
        "type": "vom_fit", "window_id": window_id, "ok": True,
        "exx": float(fit.strain[0, 0]), "eyy": float(fit.strain[1, 1]),
        "exy": float(fit.strain[0, 1]), "residual": float(fit.residual),
        "friedel": (None if fr is None or _np.isnan(fr) else float(fr)),
        "matched": int(fit.n_matched),
    })


def vom_refine(session, plot, payload) -> None:
    """'Refine' tab: live-update the strain cap / match tolerance and re-fit the
    pattern under the crosshair (Qt parity). Updates the green-template overlay
    and streams the new strain readout via ``_emit_vom_fit``."""
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_vom_wizard", None) if tree is not None else None
    if not wiz or wiz.get("overlay") is None:
        return
    params = {}
    if payload.get("strain_cap") is not None:
        params["strain_cap"] = float(payload["strain_cap"])
        wiz["strain_cap"] = params["strain_cap"]
    if payload.get("sink_bw") is not None:
        params["sink_bw"] = float(payload["sink_bw"])
        wiz["sink_bw"] = params["sink_bw"]

    def _work():
        try:
            wiz["overlay"].set_params(**params)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("vom_refine failed: %s", e)

    threading.Thread(target=_work, daemon=True, name="vom-refine").start()


def vom_run(session, plot, payload) -> None:
    """'Compute Maps': fit orientation + strain for every position using the
    already-built library → an IPF-Z window + εxx / εyy / εxy strain windows."""
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_vom_wizard", None) if tree is not None else None
    if not wiz or wiz.get("lib") is None:
        emit_error("Compute Maps: generate the library first")
        return
    vecs = getattr(tree, "diffraction_vectors", None)
    if vecs is None:
        emit_error("Compute Maps: no diffraction vectors on this tree")
        return
    lib = wiz["lib"]
    strain_cap = float(payload.get("strain_cap", DEFAULTS["strain_cap"]))
    sink_bw = payload.get("sink_bw", wiz.get("sink_bw"))
    smooth = bool(payload.get("smooth", DEFAULTS["smooth"]))
    fit_params = dict(strain_cap=strain_cap)
    if sink_bw is not None:
        fit_params["sink_bw"] = float(sink_bw)
    emit_status("Vector Orientation: fitting the field…")

    # Initialise the CUDA autograd engine on THIS (dispatch) thread before the
    # fit runs on the worker — torch's CUDA backward segfaults the first time it
    # runs on an un-warmed thread on Windows (no-op on MPS/CPU).
    try:
        from spyde.actions.vector_orientation_gpu import warmup_autograd
        warmup_autograd()
    except Exception as e:
        log.debug("CUDA autograd warmup failed: %s", e)

    def _work():
        try:
            # Reuse the field already fit at Generate (the live IPF heatmap) when
            # nothing material changed — Compute Maps is then instant and just
            # adds the strain windows next to the existing IPF heatmap.
            cached = getattr(tree, "_vom_field", None)
            reuse = cached is not None and abs(
                float(getattr(tree, "_vom_field_cap", -1)) - strain_cap) < 1e-9
            if reuse:
                result, with_ipf = cached, False
            else:
                result = _fit_field(vecs, lib, fit_params)
                with_ipf = True
            if result is None:
                emit_error("Vector Orientation: fit returned no result")
                return
            tree.vector_orientation = result
            _build_result_windows(session, tree.root, result, smooth=smooth,
                                  with_ipf=with_ipf)
            emit_status("Vector Orientation map complete")
        except Exception as e:
            emit_error(f"Compute Maps failed: {e}")
            log.exception("Compute Maps failed")

    threading.Thread(target=_work, daemon=True, name="vom-run").start()


def _fit_field(vecs, lib, params):
    """Whole-field vector-orientation fit. Prefers the BATCHED GPU path (the Qt
    production path) — it fits every nav position in one pass, seconds on a real
    13k-pattern scan; the serial per-pattern CPU scipy fit is ~minutes and only
    a fallback when torch is unavailable (or the GPU fit errors)."""
    ny, nx = vecs.nav_shape
    total = ny * nx

    def _progress(done, total_):
        if total_:
            emit_status(f"Vector Orientation: fitting… {int(100 * done / total_)}%")

    try:
        from spyde.actions.vector_orientation_gpu import (
            compute_vector_orientation_gpu, select_device, torch_available)
        dev = select_device() if torch_available() else None
    except Exception:
        dev = None
    if dev is not None:
        emit_status(f"Vector Orientation: fitting on {dev.type} ({total} patterns)…")
        try:
            res = compute_vector_orientation_gpu(vecs, lib, params, progress=_progress)
            if res is not None:
                return res
        except Exception as e:
            import traceback
            traceback.print_exc()
            emit_status(f"Vector Orientation: GPU fit failed ({e}); CPU fallback…")

    from spyde.actions.vector_orientation import compute_vector_orientation
    emit_status(f"Vector Orientation: fitting on CPU ({total} patterns, slower)…")
    return compute_vector_orientation(vecs, lib, params, progress=_progress)


def _build_result_windows(session, src, result, *, smooth=False, with_ipf=True) -> None:
    """Open three strain windows (εxx/εyy/εxy) and — unless the live IPF heatmap
    already exists (``with_ipf=False``) — an IPF-Z orientation window (RGB)."""
    ny, nx = result.nav_shape
    base = src.metadata.get_item("General.title", "Signal")

    def _window(title, data, rgb=False, levels=None):
        new_sig = hs.signals.Signal2D(np.zeros((ny, nx), dtype=np.float32))
        new_sig.metadata.General.title = f"{base} — {title}"
        tree = session._add_signal(new_sig)
        for sp in list(getattr(tree, "signal_plots", [])):
            try:
                sp.needs_auto_level = True
                if levels is not None:
                    sp.set_clim(*levels)
                sp.set_data(data)
            except Exception as e:
                log.debug("painting vom window signal plot failed: %s", e)
        return tree

    if with_ipf:
        ipf = result.ipf_color_map(direction="z")        # (ny, nx, 3) uint8
        otree = _window("Orientation (IPF-Z)", ipf, rgb=True)
        otree.vector_orientation = result
        # 3-D IPF explorer as a second figure → the window gets a 2D/3D toggle,
        # plus the point selector → 3-D highlight.
        try:
            from spyde.actions.ipf_view import attach_ipf_3d, attach_ipf_point_selector
            attach_ipf_3d(otree, result, direction="z")
            attach_ipf_point_selector(otree, result, "z")
        except Exception as e:
            log.debug("attaching 3-D IPF explorer to vom result failed: %s", e)

    # ── Strain: ONE window with εxx / εyy / εxy as chip-selectable views (the
    #    unified chip-strip selector — ⌘-click to tile + compare). εxx is the
    #    window's signal plot; εyy / εxy are extra tagged view figures.
    strain = result.smoothed_strain() if smooth else result.strain
    mats = [(lbl, np.nan_to_num(strain[..., i].astype(np.float32)))
            for lbl, i in (("εxx", 0), ("εyy", 1), ("εxy", 2))]
    finite = [float(np.nanmax(np.abs(m))) for _, m in mats if np.isfinite(m).any()]
    lim = (max(finite) if finite else 1.0) or 1.0
    stree = _window("Strain", mats[0][1], levels=(-lim, lim))
    sp = next(iter(getattr(stree, "signal_plots", [])), None)
    if sp is not None:
        sp.set_view_tag("εxx", "2d")
        wid = getattr(sp, "window_id", None)
        if wid is not None:
            from spyde.actions.views import emit_view_figure, register_views
            # Stash every component so ⌘-tiling can rebuild a side-by-side
            # (anyplotlib multi-axis) figure for any selected subset.
            register_views(wid, mats, levels=(-lim, lim))
            for lbl, m in mats[1:]:
                emit_view_figure(wid, m, lbl, kind="2d", levels=(-lim, lim))

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

No Qt: this is the Electron-native staged action; it is import-safe in the
backend. The old pyqtgraph caret (vector_orientation_action.py) was removed in
the Qt-removal cleanup.
"""
from __future__ import annotations

import logging

from spyde.backend.ipc import emit, emit_status, emit_error
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions._common import reciprocal_radius as _reciprocal_radius

log = logging.getLogger(__name__)

DEFAULTS = dict(
    accelerating_voltage=200.0,
    resolution=1.0,
    minimum_intensity=1e-4,
    strain_cap=0.05,
    smooth=False,
)

from spyde.actions.wizard import WizardController


class VomWizard(WizardController):
    """Owns the Vector-Orientation wizard state: the .cif phase, the diffsims
    simulation + per-template g-vector library, the live refine overlay on the
    source DP, the Refine-tab weights, and the Generate-time field cache."""

    key = "vom"

    # Declared parameter schema (single source of truth for every host — the
    # Electron VectorOrientationWizard.tsx caret mirrors these; note the caret
    # shows strain_cap/sink_bw/gamma as PERCENT sliders but dispatches the
    # fractional values declared here). Same dict spec as toolbars.yaml.
    parameters = {
        "cif_path": {
            "name": "Crystal (.cif)", "type": "file", "default": "",
            "extensions": [".cif"], "tab": "Library",
        },
        "accelerating_voltage": {
            "name": "Voltage (kV)", "type": "float", "default": 200.0,
            "min": 20.0, "max": 1000.0, "tab": "Library",
        },
        "resolution": {
            "name": "Angle res (°)", "type": "float", "default": 1.0,
            "min": 0.1, "max": 10.0, "step": 0.1, "tab": "Library",
        },
        "minimum_intensity": {
            "name": "Min intensity", "type": "float", "default": 1e-4,
            "min": 0.0, "max": 0.05, "step": 0.0005, "tab": "Library",
        },
        "strain_cap": {
            "name": "Strain cap", "type": "float", "default": 0.05,
            "min": 0.005, "max": 0.20, "step": 0.005, "tab": "Refine",
        },
        "sink_bw": {
            "name": "Tolerance (Å⁻¹)", "type": "float", "default": 0.04,
            "min": 0.005, "max": 0.15, "step": 0.005, "tab": "Refine",
        },
        "gamma": {
            "name": "Intensity γ", "type": "float", "default": 0.5,
            "min": 0.0, "max": 1.0, "step": 0.05, "tab": "Refine",
        },
        "k_power": {
            "name": "High-k weight", "type": "float", "default": 0.0,
            "min": 0.0, "max": 2.0, "step": 0.25, "tab": "Refine",
        },
        "smooth": {
            "name": "Smooth strain (TV)", "type": "bool", "default": False,
            "tab": "Run",
        },
    }

    def __init__(self, session, tree, *, phase, sim, lib, overlay,
                 voltage, recip_r, strain_cap, sink_bw=None):
        super().__init__(session, tree)
        self.phase = phase
        self.sim = sim
        self.lib = lib
        self.overlay = overlay
        self.voltage = voltage
        self.recip_r = recip_r
        self.strain_cap = strain_cap
        self.sink_bw = sink_bw
        self.gamma = None
        self.k_power = None
        self.field = None

    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.overlay is not None and hasattr(self.overlay, "remove"):
            try:
                self.overlay.remove()
            except Exception as e:
                log.debug("removing VOM wizard overlay failed: %s", e)
        self.overlay = None
        if getattr(self.tree, "_vom_wizard", None) is self:
            self.tree._vom_wizard = None


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
        # Find Vectors may still be attaching (its batch finalizes on a worker
        # thread) — wait it out and re-dispatch instead of erroring in the gap.
        from spyde.actions.lifecycle import wait_for_vectors
        if wait_for_vectors(session, plot,
                            lambda: vom_generate_library(session, plot, payload),
                            what="Vector Orientation", strict=True):
            return
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
            # A regenerated library replaces the previous wizard wholesale.
            old = getattr(tree, "_vom_wizard", None)
            if old is not None and hasattr(old, "remove"):
                try:
                    old.remove()
                except Exception as e:
                    log.debug("removing prior VOM wizard failed: %s", e)
            overlay = None
            wid = getattr(src, "window_id", None)
            try:
                from spyde.actions.vector_overlay import attach_vector_orientation_overlay
                overlay = attach_vector_orientation_overlay(
                    src, vecs, lib, tree, on_fit=lambda fit: _emit_vom_fit(wid, fit))
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("vom overlay attach failed: %s", e)

            wiz = VomWizard(
                session, tree, phase=phase, sim=sim, lib=lib, overlay=overlay,
                voltage=voltage, recip_r=recip_r,
                strain_cap=DEFAULTS["strain_cap"], sink_bw=None,
            )
            tree._vom_wizard = wiz
            emit_status(f"Vector Orientation: library ready ({n_templates} templates) "
                        f"— computing live IPF map…")
            emit({"type": "vom_library_ready",
                  "window_id": getattr(src, "window_id", None),
                  "n_templates": n_templates})

            # LIVE IPF heatmap (Qt parity, "super nice"): fit the WHOLE field on
            # the GPU right away — it's seconds on a real scan — and show the
            # IPF-Z map immediately so you see the orientation result while you
            # refine. Cached on the tree so Compute Maps is then instant.
            field = _fit_field(vecs, lib, dict(strain_cap=DEFAULTS["strain_cap"]),
                               tree=tree)
            if field is not None:
                tree._vom_field = field
                tree._vom_field_cap = DEFAULTS["strain_cap"]
                # Field cached with default weights → (strain_cap, gamma, k_power,
                # sink_bw); Compute re-fits only if the Refine tab changed any.
                tree._vom_field_sig = (DEFAULTS["strain_cap"], None, None, None)
                wiz.field = field
                _build_ipf_heatmap(session, root, field)
                emit_status("Vector Orientation: live IPF ready — refine, or "
                            "Compute Maps for the strain maps")
        except Exception as e:
            emit_error(f"Generate Library failed: {e}")
            log.exception("Generate Library failed")

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="vom-generate-library")


def _build_ipf_heatmap(session, src, result, title="Orientation (IPF-Z, live)"):
    """Open just the IPF-Z map window (the live refine heatmap) + its 3-D
    explorer. The strain windows are added later by Compute Maps."""
    from spyde.actions.commit import commit_result_tree
    base = src.metadata.get_item("General.title", "Signal")

    def _attach(tree):
        from spyde.actions.ipf_view import attach_ipf_3d, attach_ipf_point_selector
        attach_ipf_3d(tree, result, "z")
        attach_ipf_point_selector(tree, result, "z")

    return commit_result_tree(
        session, title=f"{base} — {title}",
        primary=result.ipf_color_map("z"),
        attrs={"vector_orientation": result},
        provenance={"action": "Vector Orientation Mapping",
                    "source_title": base},
        on_tree=_attach,
    )


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
    if wiz is None or wiz.overlay is None:
        return
    params = {}
    for key in ("strain_cap", "sink_bw", "gamma", "k_power"):
        if payload.get(key) is not None:
            params[key] = float(payload[key])
            setattr(wiz, key, params[key])

    def _work():
        try:
            wiz.overlay.set_params(**params)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("vom_refine failed: %s", e)

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="vom-refine")


def vom_run(session, plot, payload) -> None:
    """'Compute Maps': fit orientation + strain for every position using the
    already-built library → an IPF-Z window + εxx / εyy / εxy strain windows."""
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_vom_wizard", None) if tree is not None else None
    if wiz is None or wiz.lib is None:
        emit_error("Compute Maps: generate the library first")
        return
    vecs = getattr(tree, "diffraction_vectors", None)
    if vecs is None:
        from spyde.actions.lifecycle import wait_for_vectors
        if wait_for_vectors(session, plot,
                            lambda: vom_run(session, plot, payload),
                            what="Compute Maps", strict=True):
            return
        emit_error("Compute Maps: no diffraction vectors on this tree")
        return
    lib = wiz.lib
    strain_cap = float(payload.get("strain_cap", DEFAULTS["strain_cap"]))
    sink_bw = payload.get("sink_bw", wiz.sink_bw)
    smooth = bool(payload.get("smooth", DEFAULTS["smooth"]))
    fit_params = dict(strain_cap=strain_cap)
    if sink_bw is not None:
        fit_params["sink_bw"] = float(sink_bw)
    # Reflection-weighting knobs (gamma / |g| lever-arm) from the Refine tab.
    for key in ("gamma", "k_power"):
        val = payload.get(key, getattr(wiz, key, None))
        if val is not None:
            fit_params[key] = float(val)
    # Signature for the Generate-time field cache: re-fit if any weight changed.
    fit_sig = (strain_cap, fit_params.get("gamma"), fit_params.get("k_power"),
               fit_params.get("sink_bw"))
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
            reuse = cached is not None and (
                getattr(tree, "_vom_field_sig", None) == fit_sig)
            if reuse:
                result, with_ipf = cached, False
            else:
                result = _fit_field(vecs, lib, fit_params, tree=tree)
                with_ipf = True
            if result is None:
                # None also means "cancelled" (tree closed mid-fit) — no toast.
                if not getattr(tree, "_spyde_closed", False):
                    emit_error("Vector Orientation: fit returned no result")
                return
            tree.vector_orientation = result
            _build_result_windows(session, tree.root, result, smooth=smooth,
                                  with_ipf=with_ipf)
            emit_status("Vector Orientation map complete")
        except Exception as e:
            emit_error(f"Compute Maps failed: {e}")
            log.exception("Compute Maps failed")

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="vom-run")


def _fit_field(vecs, lib, params, *, tree=None):
    """Whole-field vector-orientation fit. Prefers the BATCHED GPU path (the Qt
    production path) — it fits every nav position in one pass, seconds on a real
    13k-pattern scan; the serial per-pattern CPU scipy fit is ~minutes and only
    a fallback when torch is unavailable (or the GPU fit errors).

    ``tree`` (when given) registers a stopped_flag so closing the tree mid-fit
    interrupts the GPU anneal / CPU loop instead of running it to completion."""
    ny, nx = vecs.nav_shape
    total = ny * nx

    def _progress(done, total_):
        if total_:
            emit_status(f"Vector Orientation: fitting… {int(100 * done / total_)}%")

    # Cancellation: both the GPU anneal and the CPU per-pattern loop poll this.
    stopped_flag = [False]
    if tree is not None and hasattr(tree, "register_cancel"):
        tree.register_cancel(flag=stopped_flag)
    try:
        try:
            from spyde.actions.vector_orientation_gpu import (
                compute_vector_orientation_gpu, select_device, torch_available)
            dev = select_device() if torch_available() else None
        except Exception:
            dev = None
        if dev is not None:
            emit_status(f"Vector Orientation: fitting on {dev.type} ({total} patterns)…")
            try:
                res = compute_vector_orientation_gpu(
                    vecs, lib, params, progress=_progress, stopped_flag=stopped_flag)
                if res is not None:
                    return res
                if stopped_flag[0]:
                    return None          # cancelled — don't fall through to CPU
            except Exception as e:
                import traceback
                traceback.print_exc()
                emit_status(f"Vector Orientation: GPU fit failed ({e}); CPU fallback…")

        if stopped_flag[0]:
            return None
        from spyde.actions.vector_orientation import compute_vector_orientation
        emit_status(f"Vector Orientation: fitting on CPU ({total} patterns, slower)…")
        return compute_vector_orientation(
            vecs, lib, params, progress=_progress, stopped_flag=stopped_flag)
    finally:
        if tree is not None and hasattr(tree, "unregister_cancel"):
            tree.unregister_cancel(flag=stopped_flag)


def _build_result_windows(session, src, result, *, smooth=False, with_ipf=True) -> None:
    """Commit the fitted field: a strain window (εxx signal plot + εyy/εxy as
    chip-selectable views) and — unless the live IPF heatmap already exists
    (``with_ipf=False``) — an IPF-Z orientation window (RGB)."""
    from spyde.actions.commit import commit_result_tree
    base = src.metadata.get_item("General.title", "Signal")

    if with_ipf:
        _build_ipf_heatmap(session, src, result, title="Orientation (IPF-Z)")

    strain = result.smoothed_strain() if smooth else result.strain
    commit_result_tree(
        session, title=f"{base} — Strain",
        primary=strain[..., 0], primary_label="εxx",
        views=[("εyy", strain[..., 1]), ("εxy", strain[..., 2])],
        levels="auto_sym",
        provenance={"action": "Vector Orientation Mapping",
                    "source_title": base, "params": {"smooth": bool(smooth)}},
    )

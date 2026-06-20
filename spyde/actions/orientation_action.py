"""
orientation_action.py — Electron-native "Orientation Mapping".

Reuses the Qt-free compute core (`orientation_compute._do_compute_orientations`)
and the diffsims library helpers. Loads a crystal structure from a .cif, builds
the template library, runs batch template matching on a background thread, then
opens an IPF-Z orientation-map window (RGB) and attaches ``tree.orientation_map``.

No Qt. The interactive per-pixel IPF *refinement* (the live crosshair refine) is
NOT ported here — this is the full-field map. `_do_compute_orientations` is
memory-safe (per-chunk slices only; never computes the full dataset).
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import hyperspy.api as hs

from spyde.backend.ipc import emit, emit_status, emit_error
from spyde.actions.context import src_plot_tree as _src_plot_tree

log = logging.getLogger(__name__)

DEFAULTS = dict(accelerating_voltage=200.0, resolution=1.0, n_best=5, gamma=1.0,
                minimum_intensity=1e-4)


def _count_templates(sim) -> int:
    """Total template count across a possibly multi-phase Simulation2D (its
    ``rotations`` is a list, one Rotation per phase, when multi-phase)."""
    try:
        rots = sim.rotations
        if isinstance(rots, (list, tuple)):
            return int(sum(np.asarray(r.data).reshape(-1, 4).shape[0] for r in rots))
        return int(np.asarray(rots.data).reshape(-1, 4).shape[0])
    except Exception:
        return 0


def _reciprocal_radius(signal) -> float:
    """Max reciprocal radius from the signal-axis calibration."""
    sig_axes = signal.axes_manager.signal_axes
    return float(min(ax.scale * ax.size / 2.0 for ax in sig_axes))


def orientation_mapping(ctx, action_name: str = "Orientation Mapping",
                        cif_path: str | None = None, **params):
    """Toolbar entry point (ActionContext convention)."""
    plot = ctx.plot
    session = ctx.session
    src_tree = getattr(plot, "signal_tree", None)
    if src_tree is None or session is None:
        emit_error("Orientation Mapping: no active dataset")
        return None

    src = src_tree.root
    am = src.axes_manager
    if am.signal_dimension != 2 or am.navigation_dimension != 2:
        emit_error("Orientation Mapping needs a 4D-STEM dataset (2-D nav + 2-D signal)")
        return None
    if not cif_path:
        emit_error("Orientation Mapping: choose a .cif crystal structure first")
        return None

    sim_params = dict(
        accelerating_voltage=float(params.get("accelerating_voltage",
                                              DEFAULTS["accelerating_voltage"])),
        resolution=float(params.get("resolution", DEFAULTS["resolution"])),
        minimum_intensity=float(params.get("minimum_intensity",
                                           DEFAULTS["minimum_intensity"])),
    )
    match_params = dict(
        n_best=int(params.get("n_best", DEFAULTS["n_best"])),
        gamma=float(params.get("gamma", DEFAULTS["gamma"])),
        normalize_templates=bool(params.get("normalize_templates", False)),
    )

    emit_status("Orientation: loading crystal structure…")
    src_dp_plot = plot   # overlay the matched template on the live DP

    def _work():
        try:
            from orix.crystal_map import Phase
            phase = Phase.from_cif(cif_path)
            run_orientation(session, src, src_tree, [phase], sim_params,
                            match_params, src_dp_plot=src_dp_plot)
        except Exception as e:
            import traceback
            emit_error(f"Orientation Mapping failed: {e}")
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True, name="orientation").start()
    return None


def run_orientation(session, src, src_tree, phases, sim_params, match_params,
                    src_dp_plot=None):
    """Library → compute → IPF window. Synchronous (call from a worker thread).
    Factored out so tests can pass a programmatically-built phase."""
    from spyde.actions.orientation_compute import (
        generate_library_from_phases, _do_compute_orientations,
    )
    emit_status("Orientation: generating template library…")
    sim = generate_library_from_phases(
        phases=phases,
        accelerating_voltage=sim_params["accelerating_voltage"],
        resolution=sim_params["resolution"],
        minimum_intensity=float(sim_params.get("minimum_intensity", 1e-4)),
        reciprocal_radius=_reciprocal_radius(src),
    )
    gamma = float(match_params.get("gamma", DEFAULTS["gamma"]))
    emit_status("Orientation: matching templates…")
    om = _compute_with_live_ipf(
        session, src, src_tree, sim,
        dict(n_best=match_params.get("n_best", 5), gamma=gamma,
             normalize_templates=bool(match_params.get("normalize_templates", True))))
    if om is None:
        emit_error("Orientation: compute returned no result")
        return None
    _overlay_template_on_source(src_tree, src_dp_plot, src, sim, gamma)
    emit_status("Orientation map complete")
    return om


def _overlay_template_on_source(src_tree, dp_plot, src, sim, gamma) -> None:
    """Overlay the best-matching template's simulated spots on the SOURCE
    diffraction pattern, tracking the navigator (Qt-parity orientation overlay).
    Replaces any prior overlay so re-running doesn't stack markers."""
    if dp_plot is None or src_tree is None:
        return
    from spyde.actions.orientation_compute import build_matching_cache
    from spyde.actions.vector_overlay import attach_orientation_overlay
    old = getattr(src_tree, "_orientation_overlay", None)
    if old is not None:
        try:
            old.remove()
        except Exception as e:
            log.debug("removing prior orientation overlay failed: %s", e)
    try:
        cache = build_matching_cache(src, sim)
        src_tree._orientation_overlay = attach_orientation_overlay(
            dp_plot, src, sim, cache, src_tree,
            gamma=gamma, max_radius=_reciprocal_radius(src),
            normalize_templates=True,
        )
    except Exception as e:
        log.debug("orientation overlay attach failed: %s", e)


def _open_refine_ipf(session, dp_plot, signal, sim, cache, tree):
    """Open the live per-phase IPF correlation-heatmap window for refine and wire
    a controller to the navigator. Returns the controller (or None)."""
    try:
        from spyde.actions.ipf_refine import build_phase_ipf
        from spyde.actions.ipf_refine_render import (
            build_refine_figure, emit_refine_window, RefineIpfController,
        )
        infos = build_phase_ipf(sim)
        if not infos:
            return None
        _fig, fig_id, html, panels = build_refine_figure(infos)
        base = signal.metadata.get_item("General.title", "Signal")
        emit_refine_window(session, fig_id, html, title=f"{base} — IPF Refine")
        return RefineIpfController(
            dp_plot, signal, sim, cache, infos, panels,
            gamma=DEFAULTS["gamma"], normalize=False).attach(tree)
    except Exception as e:
        log.debug("refine IPF window failed: %s", e)
        return None


def _create_blank_ipf_window(session, src, ny, nx):
    """Open a blank IPF-Z window up front (for the progressive fill-in)."""
    new_sig = hs.signals.Signal2D(np.zeros((ny, nx), dtype=np.float32))
    base = src.metadata.get_item("General.title", "Signal")
    new_sig.metadata.General.title = f"{base} — Orientation (IPF-Z)"
    return session._add_signal(new_sig)


def _finalize_ipf_window(tree, om) -> None:
    """Paint the final IPF-Z map, attach the map + 3-D explorer + point selector."""
    tree.orientation_map = om
    ipf = om.ipf_color_map(direction="z")        # (ny, nx, 3) uint8
    for sp in list(getattr(tree, "signal_plots", [])):
        try:
            sp.needs_auto_level = True
            sp.set_data(ipf)   # RGB → Plot._set_array routes it to anyplotlib
        except Exception as e:
            log.debug("painting IPF map onto signal plot failed: %s", e)
    try:
        from spyde.actions.ipf_view import attach_ipf_3d, attach_ipf_point_selector
        attach_ipf_3d(tree, om, direction="z")
        attach_ipf_point_selector(tree, om, "z")
    except Exception as e:
        log.debug("attaching 3-D IPF explorer failed: %s", e)


def _compute_with_live_ipf(session, src, src_tree, sim, params):
    """Dense orientation match with a PROGRESSIVE IPF: open the IPF-Z window blank
    up front, poll the per-chunk shared-memory buffer to fill the map in live (Qt
    live-buffer parity), then finalize. Returns the SpyDEOrientationMap (or None).
    """
    from spyde.actions.orientation_compute import _do_compute_orientations
    from spyde.drawing.update_functions import ensure_live_buffer, read_live_buffer
    nav_dim = src.axes_manager.navigation_dimension
    ny, nx = tuple(int(s) for s in src.data.shape[:nav_dim])[-2:]
    om_tree = _create_blank_ipf_window(session, src, ny, nx)
    shm_name = f"spyde_om_{id(src)}"
    try:
        shm = ensure_live_buffer((ny, nx, 9), shm_name)
    except Exception:
        shm, shm_name = None, None
    stop_poll = [False]

    def _poller():
        sp = next(iter(getattr(om_tree, "signal_plots", []) or []), None)
        while not stop_poll[0]:
            try:
                z = read_live_buffer((ny, nx, 9), shm_name)[..., 6:9]   # Z RGB
                if sp is not None and np.isfinite(z).any():
                    sp.needs_auto_level = True
                    sp.set_data(np.nan_to_num(z).clip(0, 255).astype(np.uint8))
            except Exception as e:
                log.debug("live IPF poll paint failed: %s", e)
            time.sleep(0.4)

    if shm_name is not None:
        threading.Thread(target=_poller, daemon=True, name="om-poll").start()
    try:
        om = _do_compute_orientations(src, sim, params, main_window=session,
                                      signal_tree=src_tree, shm_name=shm_name)
    finally:
        stop_poll[0] = True
        if shm is not None:
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                log.debug("cleaning up OM live-buffer shared memory failed: %s", e)
    if om is not None:
        _finalize_ipf_window(om_tree, om)
    return om


# ─────────────────────────────────────────────────────────────────────────────
# Staged "wizard" workflow (Qt 4-tab parity): Generate Library → live Refine →
# Compute Map. State lives on the source tree as `_om_wizard`.
# ─────────────────────────────────────────────────────────────────────────────


def om_generate_library(session, plot, payload) -> None:
    """Tab 2 'Generate Library': load the .cif + build the diffsims library and
    matching cache, then activate the LIVE refine overlay on the source DP (the
    matched template tracks the crosshair). Reuses an existing library if already
    built for this tree."""
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Generate Library: no active dataset")
        return
    # Accept one .cif (`cif_path`) or several (`cif_paths`) for multi-phase OM.
    cif_paths = list(payload.get("cif_paths") or [])
    if not cif_paths and payload.get("cif_path"):
        cif_paths = [payload["cif_path"]]
    if not cif_paths:
        emit_error("Generate Library: choose a .cif crystal first")
        return
    voltage = float(payload.get("accelerating_voltage", DEFAULTS["accelerating_voltage"]))
    resolution = float(payload.get("resolution", DEFAULTS["resolution"]))
    min_int = float(payload.get("minimum_intensity", DEFAULTS["minimum_intensity"]))
    emit_status("Orientation: generating template library…")

    def _work():
        try:
            from orix.crystal_map import Phase
            from spyde.actions.orientation_compute import (
                generate_library_from_phases, build_matching_cache,
            )
            from spyde.actions.vector_overlay import attach_orientation_overlay
            src_root = tree.root
            phases = [Phase.from_cif(p) for p in cif_paths]
            recip_r = _reciprocal_radius(src_root)
            sim = generate_library_from_phases(phases, voltage, resolution,
                                               min_int, recip_r)
            n_templates = _count_templates(sim)

            old = getattr(tree, "_om_wizard", None)
            if old is not None and old.get("overlay") is not None:
                try:
                    old["overlay"].remove()
                except Exception as e:
                    log.debug("removing prior OM wizard overlay failed: %s", e)
            # Build the matching cache once (used by both the single-pattern
            # best-match overlay AND the per-phase IPF correlation heatmap).
            cache = build_matching_cache(src_root, sim)

            # The single-pattern best-match SPOT overlay is single-phase; skip it
            # for multi-phase (the whole-field Run handles multi-phase).
            overlay = None
            if len(phases) == 1:
                overlay = attach_orientation_overlay(
                    src, src_root, sim, cache, tree,
                    gamma=DEFAULTS["gamma"], max_radius=recip_r, normalize_templates=False,
                )

            # Live IPF correlation-heatmap window (one triangle per phase): updates
            # as the navigator moves; double-click a triangle to limit the refined
            # IPF region. Multi-phase elegantly → multiple triangles.
            refine_ipf = _open_refine_ipf(session, src, src_root, sim, cache, tree)

            old_ipf = getattr(getattr(tree, "_om_wizard", None) or {}, "get", lambda *_: None)("refine_ipf")
            if old_ipf is not None:
                try:
                    old_ipf.remove()
                except Exception as e:
                    log.debug("removing prior refine IPF window failed: %s", e)

            tree._om_wizard = {
                "phases": phases, "sim": sim, "cache": cache, "overlay": overlay,
                "refine_ipf": refine_ipf, "voltage": voltage, "recip_r": recip_r,
            }
            n_ph = len(phases)
            extra = "move the crosshair to refine" if n_ph == 1 else f"{n_ph} phases"
            emit_status(f"Orientation: library ready ({n_templates} templates) — {extra}")
            emit({"type": "om_library_ready", "window_id": getattr(src, "window_id", None),
                  "n_templates": n_templates})
        except Exception as e:
            import traceback
            emit_error(f"Generate Library failed: {e}")
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True, name="om-generate-library").start()


def om_refine(session, plot, payload) -> None:
    """Tab 3 'Refine': live-update the gamma / scale / min-intensity / normalize
    sliders → redraw the matched template at the current crosshair position."""
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_om_wizard", None) if tree is not None else None
    if not wiz or (wiz.get("overlay") is None and wiz.get("refine_ipf") is None):
        return

    def _work():
        if wiz.get("overlay") is not None:
            try:
                wiz["overlay"].set_refine_params(
                    gamma=payload.get("gamma"),
                    scale_override=payload.get("scale_override"),
                    min_intensity=payload.get("min_intensity"),
                    normalize_templates=payload.get("normalize_templates"),
                )
                wiz["refine"] = dict(payload)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("om_refine failed: %s", e)
        # The IPF heatmap follows the same gamma / normalize knobs (and exists for
        # multi-phase too, where the spot overlay does not).
        rip = wiz.get("refine_ipf")
        if rip is not None:
            try:
                rip.set_refine_params(gamma=payload.get("gamma"),
                                      normalize=payload.get("normalize_templates"))
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("refine ipf params failed: %s", e)

    threading.Thread(target=_work, daemon=True, name="om-refine").start()


def om_run(session, plot, payload) -> None:
    """Tab 4 'Compute Map': full-field template match using the already-built
    library → IPF-Z window + attached map."""
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_om_wizard", None) if tree is not None else None
    if not wiz or wiz.get("sim") is None:
        emit_error("Compute Map: generate the library first")
        return
    sim = wiz["sim"]
    n_best = int(payload.get("n_best", DEFAULTS["n_best"]))
    gamma = float(payload.get("gamma", DEFAULTS["gamma"]))
    normalize = bool(payload.get("normalize_templates", False))
    emit_status("Orientation: computing map…")

    def _work():
        try:
            om = _compute_with_live_ipf(
                session, tree.root, tree, sim,
                dict(n_best=n_best, gamma=gamma, normalize_templates=normalize))
            if om is None:
                emit_error("Orientation: compute returned no result")
                return
            emit_status("Orientation map complete")
        except Exception as e:
            import traceback
            emit_error(f"Compute Map failed: {e}")
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True, name="om-run").start()

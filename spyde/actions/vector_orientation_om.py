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

import threading

import numpy as np
import hyperspy.api as hs

from spyde.backend.ipc import emit, emit_status, emit_error
from spyde.actions.context import src_plot_tree as _src_plot_tree

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
            # the measured vectors (red) under the crosshair (Qt parity).
            overlay = None
            old = getattr(tree, "_vom_wizard", None)
            if old is not None and old.get("overlay") is not None:
                try:
                    old["overlay"].remove()
                except Exception:
                    pass
            try:
                from spyde.actions.vector_overlay import attach_vector_orientation_overlay
                overlay = attach_vector_orientation_overlay(src, vecs, lib, tree)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("vom overlay attach failed: %s", e)

            tree._vom_wizard = {
                "phase": phase, "sim": sim, "lib": lib, "overlay": overlay,
                "voltage": voltage, "recip_r": recip_r,
            }
            emit_status(f"Vector Orientation: library ready ({n_templates} templates) "
                        f"— move the crosshair to refine, or Compute Maps")
            emit({"type": "vom_library_ready",
                  "window_id": getattr(src, "window_id", None),
                  "n_templates": n_templates})
        except Exception as e:
            import traceback
            emit_error(f"Generate Library failed: {e}")
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True, name="vom-generate-library").start()


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
    smooth = bool(payload.get("smooth", DEFAULTS["smooth"]))
    emit_status("Vector Orientation: fitting the field…")

    def _work():
        try:
            from spyde.actions.vector_orientation import compute_vector_orientation

            def _progress(done, total):
                emit_status(f"Vector Orientation: fitting… {done}/{total}")

            result = compute_vector_orientation(
                vecs, lib, dict(strain_cap=strain_cap),
                progress=_progress,
            )
            if result is None:
                emit_error("Vector Orientation: fit returned no result")
                return
            tree.vector_orientation = result
            _build_result_windows(session, tree.root, result, smooth=smooth)
            emit_status("Vector Orientation map complete")
        except Exception as e:
            import traceback
            emit_error(f"Compute Maps failed: {e}")
            print(traceback.format_exc())

    threading.Thread(target=_work, daemon=True, name="vom-run").start()


def _build_result_windows(session, src, result, *, smooth=False) -> None:
    """Open an IPF-Z orientation window (RGB) + three strain windows (εxx/εyy/εxy),
    each a nav-shaped image painted from the fit result."""
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
            except Exception:
                pass
        return tree

    ipf = result.ipf_color_map(direction="z")        # (ny, nx, 3) uint8
    otree = _window("Orientation (IPF-Z)", ipf, rgb=True)
    otree.vector_orientation = result
    # Add the 3-D IPF explorer as a second figure → the window gets a 2D/3D toggle.
    try:
        from spyde.actions.ipf_view import attach_ipf_3d
        attach_ipf_3d(otree, result, direction="z")
    except Exception:
        pass

    strain = result.smoothed_strain() if smooth else result.strain
    for comp, ttl in (("exx", "Strain εxx"), ("eyy", "Strain εyy"), ("exy", "Strain εxy")):
        m = strain[..., {"exx": 0, "eyy": 1, "exy": 2}[comp]].astype(np.float32)
        lim = float(np.nanmax(np.abs(m))) if np.isfinite(m).any() else 1.0
        lim = lim or 1.0
        _window(ttl, np.nan_to_num(m), levels=(-lim, lim))

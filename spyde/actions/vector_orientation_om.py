"""
vector_orientation_om.py — Electron-native Vector Orientation Mapping.

REUSES the staged Orientation-Mapping wizard pattern (see ``orientation_action``):
a phase → library → whole-field fit flow driven by the sparse diffraction
vectors on a tree rather than the dense template match. Both the crosshair
preview and the map come from the vendored quantem correlation matcher
(`vector_orientation_quantem`), so Compute Maps produces the map of the fit the
preview was showing. Produces an orientation IPF-Z map plus εxx / εyy / εxy
strain maps.

Staged handlers (mirroring `om_generate_library` / `om_run`):
  vom_generate_library — load the phases, build one correlation plan each,
                         turn on the crosshair preview, cache on the tree.
  vom_run              — fit every position → IPF-Z + 3 strain map windows.

No Qt: this is the Electron-native staged action; it is import-safe in the
backend. The old pyqtgraph caret (vector_orientation_action.py) was removed in
the Qt-removal cleanup.
"""
from __future__ import annotations

import logging

import numpy as np

from de_shell.ipc import emit, emit_status, emit_error
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions._common import reciprocal_radius as _reciprocal_radius

log = logging.getLogger(__name__)

DEFAULTS = dict(
    accelerating_voltage=200.0,
    resolution=1.0,
    minimum_intensity=1e-4,
    smooth=False,
)

from de_shell.actions.wizard import WizardController


class VomWizard(WizardController):
    """Owns the Vector-Orientation wizard state: the phases, the matcher whose
    plans they were built into, the live preview and heat map on the source DP,
    and the Refine-tab settings.

    Several phases may be loaded at once. The fit picks the best-matching
    template per pattern and every template already knows which phase it came
    from, so a two-phase library answers "which crystal structure is this, in
    what orientation, under what strain" in ONE pass — which is the question a
    precipitate in a matrix actually poses."""

    key = "vom"

    # Declared parameter schema (single source of truth for every host — the
    # Electron VectorOrientationWizard.tsx caret mirrors these). Same dict spec
    # as toolbars.yaml.
    parameters = {
        "cif_paths": {
            "name": "Crystal phases (.cif)", "type": "file_list", "default": [],
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
        # Refine tunes the MATCHER, so it carries the matcher's own two
        # arguments and nothing else. The strain cap, no-match tolerance,
        # intensity gamma and high-k lever arm that used to live here are
        # parameters of the pose fit; the correlation matcher has no such
        # thing, and leaving them would have been four dials wired to nothing.
        # Sent in Å⁻¹ and read in Å⁻¹ — the old sliders declared fractions here
        # and dispatched percentages, so any host but this one sent the wrong
        # number.
        "pair_distance": {
            "name": "Pair distance (Å⁻¹)", "type": "float", "default": 0.05,
            "min": 0.01, "max": 0.15, "step": 0.005, "tab": "Refine",
        },
        "sigma_excitation": {
            "name": "Excitation σ (Å⁻¹)", "type": "float", "default": 0.04,
            "min": 0.01, "max": 0.12, "step": 0.005, "tab": "Refine",
        },
        # No strain cap: the matcher solves the deformation in closed form from
        # the paired peaks and has nothing to bound. It was the pose fit's, and
        # Run no longer uses the pose fit.
        "smooth": {
            "name": "Smooth strain (TV)", "type": "bool", "default": False,
            "tab": "Run",
        },
    }

    def __init__(self, session, tree, *, phases, overlay,
                 voltage, recip_r, refine_ipf=None, fitter=None):
        super().__init__(session, tree)
        self.refine_ipf = refine_ipf
        # Kept so the heat map can be rebuilt: it is the expensive part, and
        # re-opening a closed window must not mean rebuilding the library.
        self.fitter = fitter
        self.phases = list(phases)
        #: Å⁻¹ per detector axis unit, the join between stored vectors and
        #: the matcher, which works entirely in Å⁻¹.
        self.inverse_angstrom_factor = 1.0
        self.overlay = overlay
        self.voltage = voltage
        self.recip_r = recip_r
        # Refine's live matcher settings; None means the plan's own values.
        self.pair_distance = None
        self.sigma_excitation = None

    def ensure_refine_ipf(self, session) -> None:
        """Re-open the IPF heat map if it is not up.

        Closing that window with ✕ tears down its overlay, which is right —
        nothing should keep correlating for a window nobody is looking at. But
        it used to retire the heat map for good, with only a library rebuild to
        get it back. Re-selecting the action brings it back instead, which is
        what a toggle is supposed to mean.
        """
        if self.fitter is None or self._closed:
            return
        if self.refine_ipf is not None and not getattr(
                self.refine_ipf, "_closed", False):
            return
        vectors = getattr(self.tree, "diffraction_vectors", None)
        if vectors is None:
            return
        from spyde.actions.vector_refine_ipf import open_refine_ipf
        self.refine_ipf = open_refine_ipf(
            session, self.tree.root, self.fitter, self.phases, vectors,
            self.tree, fit_overlay=self.overlay)

    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        from spyde.actions.vector_overlay import remove_overlay_node
        remove_overlay_node(self.tree, self.overlay)
        self.overlay = None
        # Everything this wizard put on the tree comes off with it, the heat
        # map's own overlay included — leaving it behind would keep correlating
        # against a library that no longer exists.
        # The IPF window outlives the wizard (a regenerated library gets it
        # back); only the heat map that was drawing into it goes.
        from spyde.actions.ipf_panel import panel_for
        panel = panel_for(self.tree, "vom")
        if panel is not None and panel.controller is self.refine_ipf:
            panel.controller = None
        if self.refine_ipf is not None:
            try:
                self.refine_ipf.remove()
            except Exception as e:
                log.debug("removing the vector refine IPF failed: %s", e)
            self.refine_ipf = None
        if getattr(self.tree, "_vom_wizard", None) is self:
            self.tree._vom_wizard = None


def vector_orientation_mapping(ctx, action_name: str = "Vector Orientation Mapping", **kwargs):
    """Parent toolbar action — a no-op; the Electron toolbar opens the staged
    Vector Orientation wizard (which drives the ``vom_*`` handlers) instead."""
    return None


def vom_generate_library(session, plot, payload) -> None:
    """'Generate Library': load the phases and build one correlation plan each,
    then turn on the crosshair preview and the IPF heat map. Cached on the
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
    # One .cif (`cif_path`) or several (`cif_paths`) for a multi-phase fit,
    # the same shape the dense `om_generate_library` accepts.
    cif_paths = [p for p in (payload.get("cif_paths") or []) if p]
    if not cif_paths and payload.get("cif_path"):
        cif_paths = [payload["cif_path"]]
    if not cif_paths:
        emit_error("Vector Orientation: choose a .cif crystal first")
        return
    voltage = float(payload.get("accelerating_voltage", DEFAULTS["accelerating_voltage"]))
    resolution = float(payload.get("resolution", DEFAULTS["resolution"]))
    min_int = float(payload.get("minimum_intensity", DEFAULTS["minimum_intensity"]))
    emit_status("Vector Orientation: generating template library…")
    # Warm the CUDA autograd engine on this (dispatch) thread so the batched GPU
    # field fit below — run on the worker thread — is safe (no-op on MPS/CPU).
    try:
        from spyde.torch_device import warmup_autograd
        warmup_autograd()
    except Exception as e:
        log.debug("CUDA autograd warmup failed: %s", e)

    def _work():
        try:
            from orix.crystal_map import Phase

            from spyde.reciprocal_units import axis_unit_factor
            root = tree.root
            vecs = tree.diffraction_vectors
            phases = [Phase.from_cif(p) for p in cif_paths]
            recip_r = _reciprocal_radius(root)
            # No diffsims library: the matcher samples zone axes from the
            # crystal itself and leaves the in-plane angle to an FFT, so a
            # library of per-orientation template spots is not something it
            # reads. Building one anyway cost about 17 s of every Generate.
            factor = float(axis_unit_factor(root) or 1.0)

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
            fitter = None
            n_orientations = 0
            wid = getattr(src, "window_id", None)
            # The IPF window shows the phases' triangles filling in while the
            # plans build — see ipf_panel.
            from spyde.actions.ipf_panel import ensure_panel
            panel = ensure_panel(session, tree, "vom", root)
            panel.controller = None
            panel.set_phases(phases)
            panel.show()
            panel.start_filling()
            try:
                # The live preview is the quantem correlation matcher: it
                # returns a continuous orientation rather than the nearest
                # library node, and carries a correlation and a reliability
                # that the pose fit has no equivalent of.
                from spyde.actions.vector_orientation_quantem import (
                    PeaksAdapter, SinglePatternFitter, phase_to_crystal,
                )
                from spyde.actions.vector_overlay import (
                    attach_quantem_orientation_overlay,
                )
                fitter = SinglePatternFitter(
                    [phase_to_crystal(p) for p in phases],
                    PeaksAdapter(vecs, inverse_angstrom_factor=factor),
                    energy_ev=float(voltage) * 1e3, k_max=float(recip_r),
                    inverse_angstrom_factor=factor,
                    angle_step_zone_axis_deg=float(resolution))
                overlay = attach_quantem_orientation_overlay(
                    vecs, fitter, tree, on_fit=lambda fit: _emit_vom_fit(wid, fit))
                n_orientations = sum(int(m.zone_axes.shape[0])
                                     for m in fitter._maps)
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("vom overlay attach failed: %s", e)
            finally:
                panel.stop_filling()

            # The correlation surface the matcher picks its answer off, one
            # triangle per phase. A single best number cannot say whether the
            # position was confident, ambiguous between variants, or the wrong
            # phase entirely; this can, and it costs one correlation per move.
            refine_ipf = None
            if fitter is not None:
                from spyde.actions.vector_refine_ipf import open_refine_ipf
                refine_ipf = open_refine_ipf(session, root, fitter, phases,
                                             vecs, tree, fit_overlay=overlay)

            wiz = VomWizard(
                session, tree, phases=phases, overlay=overlay,
                voltage=voltage, recip_r=recip_r,
                refine_ipf=refine_ipf, fitter=fitter,
            )
            tree._vom_wizard = wiz
            phase_names = ", ".join(str(p.name) for p in phases)
            # Generate builds the library and turns on the live preview, and
            # stops there. Fitting the whole field here as well cost a scan's
            # compute before the user had chosen anything, put a result window
            # on screen that only Run is supposed to produce, and needed three
            # tree attributes and a cache signature to decide whether Run could
            # reuse it — a signature that never matched, so Run refit the same
            # field and opened a second IPF window beside the first. The map is
            # Run's job; the crosshair preview is what refining looks at.
            emit_status(f"Vector Orientation: ready ({n_orientations} "
                        f"orientations, {phase_names}) — move the crosshair "
                        f"to refine, or Compute Maps")
            emit({"type": "vom_library_ready",
                  "window_id": getattr(src, "window_id", None),
                  "ok": True, "n_templates": n_orientations,
                  "phases": [str(p.name) for p in phases]})
        except Exception as e:
            emit_error(f"Generate Library failed: {e}")
            log.exception("Generate Library failed")
            emit({"type": "vom_library_ready", "window_id": getattr(src, "window_id", None),
                  "ok": False, "error": str(e)})

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="vom-generate-library")


def _ipf_title(src, title: str) -> str:
    base = src.metadata.get_item("General.title", "Signal")
    return f"{base} — {title}"


def _open_ipf_window(session, src, nav_shape, title="Orientation (IPF-Z)"):
    """Open the IPF-Z window before the fit, every position in the unfit
    grey, so the map has somewhere to fill in. :func:`_build_ipf_heatmap`
    finishes it once the result exists."""
    from spyde.actions.commit import commit_result_tree
    ny, nx = nav_shape
    blank = np.empty((ny, nx, 3), np.uint8)
    blank[...] = _UNFIT_COLOR
    return commit_result_tree(
        session, title=_ipf_title(src, title), primary=blank,
        provenance={"action": "Vector Orientation Mapping",
                    "source_title": src.metadata.get_item(
                        "General.title", "Signal")},
        source_signal=src,
    )


class _BandPainter:
    """Paints each matched band of rows into the early IPF window.

    Holds the whole map, grey where nothing has landed, and repaints it as
    each band arrives; the paint is marshalled onto the main loop because the
    bands come off the fit's worker thread.
    """

    def __init__(self, session, tree, nav_shape, phases):
        self.session = session
        self.tree = tree
        self.phases = list(phases)
        ny, nx = nav_shape
        self.rgb = np.empty((ny, nx, 3), np.uint8)
        self.rgb[...] = _UNFIT_COLOR

    def on_band(self, row_start, row_stop, quats, phase_index) -> None:
        from spyde.signals.orientation_map import ipf_rgb
        band = np.empty((row_stop - row_start,) + self.rgb.shape[1:], np.uint8)
        band[...] = _UNFIT_COLOR
        for index, phase in enumerate(self.phases):
            mask = phase_index == index
            if mask.any():
                band[mask] = ipf_rgb(quats[mask], phase, "z")
        self.rgb[row_start:row_stop] = band
        snapshot = self.rgb.copy()

        def _paint():
            from spyde.actions.lifecycle import paint_signal_plots
            paint_signal_plots(self.tree, snapshot)

        dispatch = getattr(self.session, "_dispatch_to_main", None)
        if dispatch is not None:
            dispatch(_paint)
        else:
            _paint()


def _build_ipf_heatmap(session, src, result, title="Orientation (IPF-Z, live)",
                       tree=None):
    """Open just the IPF-Z map window (the live refine heatmap) + its 3-D
    explorer. The strain windows are added later by Compute Maps.

    With *tree* — the window :func:`_open_ipf_window` opened before the fit —
    that window is finished instead: the final map painted, the result
    attached, the explorers wired."""
    from spyde.actions.commit import commit_result_tree

    def _attach(tree):
        from spyde.actions.ipf_view import attach_ipf_3d, attach_ipf_point_selector
        attach_ipf_3d(tree, result, "z", session=session)
        attach_ipf_point_selector(tree, result, "z")

    if tree is not None:
        from spyde.actions.lifecycle import paint_signal_plots
        tree.vector_orientation = result
        paint_signal_plots(tree, result.ipf_color_map("z"))
        try:
            _attach(tree)
        except Exception as e:
            log.debug("attaching the IPF explorers failed: %s", e)
        return tree

    base = src.metadata.get_item("General.title", "Signal")
    return commit_result_tree(
        session, title=_ipf_title(src, title),
        primary=result.ipf_color_map("z"),
        attrs={"vector_orientation": result},
        provenance={"action": "Vector Orientation Mapping",
                    "source_title": base},
        on_tree=_attach, source_signal=src,
    )


def _emit_vom_fit(window_id, fit) -> None:
    """Stream the live single-pattern fit metrics to the wizard's Refine tab
    (Qt parity: εxx/εyy/εxy + residual + Friedel asymmetry + matched count)."""
    if fit is None:
        emit({"type": "vom_fit", "window_id": window_id, "ok": False})
        return
    import numpy as _np

    def _number(value):
        """JSON has no NaN; the caret reads a missing field as 'no value'."""
        value = float(value)
        return None if _np.isnan(value) else value

    # The pose fit reports strain as a 2x2 tensor, the correlation matcher as
    # [exx, eyy, exy]. Both reach this readout.
    strain = _np.asarray(fit.strain, float)
    if strain.ndim == 2:
        exx, eyy, exy = strain[0, 0], strain[1, 1], strain[0, 1]
    else:
        exx, eyy, exy = strain[0], strain[1], strain[2]

    message = {
        "type": "vom_fit", "window_id": window_id, "ok": True,
        "exx": _number(exx), "eyy": _number(eyy), "exy": _number(exy),
        "residual": _number(fit.residual),
        "matched": int(getattr(fit, "n_matched", 0) or 0),
    }
    friedel = getattr(fit, "friedel_asym", None)
    if friedel is not None:
        message["friedel"] = _number(friedel)
    # Only the correlation matcher has these; they are the discriminability the
    # pose fit never offered.
    for name in ("correlation", "reliability"):
        if hasattr(fit, name):
            message[name] = _number(getattr(fit, name))
    emit(message)


#: What the Refine tab can change, in the units it sends them (Å⁻¹). Both are
#: arguments of the REFINEMENT stage, so a nudge re-fits the pattern under the
#: crosshair without rebuilding the correlation plan. The zone and in-plane
#: sampling steps would, so they stay on the Library tab where a rebuild is
#: expected.
REFINE_KEYS = ("pair_distance", "sigma_excitation")


def vom_refine(session, plot, payload) -> None:
    """'Refine' tab: live-update the matcher's pairing distance / excitation
    width and re-fit the pattern under the crosshair. Updates the green matched
    pattern on the overlay and streams the new readout via ``_emit_vom_fit``."""
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_vom_wizard", None) if tree is not None else None
    if wiz is None or wiz.overlay is None:
        return
    params = {}
    for key in REFINE_KEYS:
        if payload.get(key) is not None:
            params[key] = float(payload[key])
            setattr(wiz, key, params[key])

    def _work():
        from spyde.actions.vector_overlay import set_vector_orientation_params
        try:
            set_vector_orientation_params(tree, wiz.overlay, **params)
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
    if wiz is None or wiz.fitter is None:
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
    smooth = bool(payload.get("smooth", DEFAULTS["smooth"]))
    # Whatever Refine was last set to, so Compute Maps produces the map of the
    # fit the crosshair was showing rather than one at the defaults.
    fit_params = {}
    for key in REFINE_KEYS:
        value = payload.get(key, getattr(wiz, key, None))
        if value is not None:
            fit_params[key] = float(value)
    emit_status("Vector Orientation: fitting the field…")

    # Initialise the CUDA autograd engine on THIS (dispatch) thread before the
    # fit runs on the worker — torch's CUDA backward segfaults the first time it
    # runs on an un-warmed thread on Windows (no-op on MPS/CPU).
    try:
        from spyde.torch_device import warmup_autograd
        warmup_autograd()
    except Exception as e:
        log.debug("CUDA autograd warmup failed: %s", e)

    # The orientation window opens NOW, blank, and fills in a band of rows at
    # a time as the match lands — the same early window every other long
    # compute here opens, instead of a status line and then everything at once.
    ipf_tree = _open_ipf_window(session, tree.root, vecs.nav_shape)
    painter = _BandPainter(session, ipf_tree, vecs.nav_shape, wiz.phases)

    def _work():
        try:
            result = _fit_field(vecs, wiz, fit_params, tree=tree,
                                on_band=painter.on_band)
            if result is None:
                # None also means "cancelled" (tree closed mid-fit) — no toast.
                if not getattr(tree, "_spyde_closed", False):
                    emit_error("Vector Orientation: fit returned no result")
                return
            tree.vector_orientation = result
            _build_result_windows(session, tree.root, result, smooth=smooth,
                                  ipf_tree=ipf_tree)
            emit_status("Vector Orientation map complete")
        except Exception as e:
            emit_error(f"Compute Maps failed: {e}")
            log.exception("Compute Maps failed")

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="vom-run")


def _fit_field(vecs, wiz, params, *, tree=None, on_band=None):
    """Whole-field fit: orientation, phase and strain, by correlation match.

    The same matcher the crosshair preview uses, run over every position — so
    what Compute Maps produces is the map of what the preview was showing,
    rather than a second method's opinion of it.

    There is deliberately no fallback to the pose fit. The two report strain in
    opposite senses (the matcher solves the deformation in reciprocal space and
    inverts it to real space, which is the one that returns an applied strain
    with the sign it was applied), so falling back would make the sign of every
    strain map depend on whether the matcher happened to succeed. A failure is
    reported instead.

    ``tree`` (when given) registers a stopped_flag, so closing the tree mid-fit
    stops the run rather than leaving a scan's compute to finish into nothing.
    ``on_band`` receives each band of rows as it is matched — see
    :func:`compute_vector_orientation_quantem`.
    """
    ny, nx = vecs.nav_shape
    total = ny * nx

    def _progress(done, total_):
        if total_:
            emit_status(f"Vector Orientation: fitting… {int(100 * done / total_)}%")

    stopped_flag = [False]
    if tree is not None and hasattr(tree, "register_cancel"):
        tree.register_cancel(flag=stopped_flag)
    try:
        from spyde.actions.vector_orientation_quantem import (
            compute_vector_orientation_quantem,
        )
        try:
            from spyde.torch_device import select_device, torch_available
            device = select_device() if torch_available() else None
        except Exception:
            device = None
        device_name = device.type if device is not None else "cpu"
        factor = float(getattr(wiz, "inverse_angstrom_factor", 1.0))
        emit_status(f"Vector Orientation: fitting on {device_name} "
                    f"({total} patterns)…")
        return compute_vector_orientation_quantem(
            vecs, wiz.phases, energy_ev=float(wiz.voltage) * 1e3,
            k_max=float(wiz.recip_r), inverse_angstrom_factor=factor,
            device=device_name, progress=_progress,
            stopped_flag=stopped_flag, params=params, on_band=on_band)
    finally:
        if tree is not None and hasattr(tree, "unregister_cancel"):
            tree.unregister_cancel(flag=stopped_flag)


#: Phase colours, in library order. Chosen to stay distinguishable in both
#: themes and under the common forms of colour blindness — a phase map is read
#: as a set of regions, so the only thing the colours must do is separate.
_PHASE_COLORS = (
    (0x4C, 0x9B, 0xE8),   # blue
    (0xE8, 0x7D, 0x3C),   # orange
    (0x5C, 0xC8, 0x6E),   # green
    (0xC6, 0x6B, 0xD8),   # purple
    (0xE8, 0xC8, 0x4C),   # yellow
)
#: Where no template fit at all. Neutral grey rather than a sixth phase colour,
#: so an unindexed region cannot be mistaken for a structure.
_UNFIT_COLOR = (0x55, 0x58, 0x60)


def phase_map_rgb(result) -> np.ndarray:
    """``(ny, nx, 3)`` uint8 — which crystal structure best explains each pattern.

    A position whose fit did not converge is grey, not phase 0: the phase index
    defaults to zero everywhere, so colouring it by index alone would paint
    every unindexed pixel as the first phase and invent a region that is not
    there.
    """

    phase_idx = np.asarray(result.phase_idx, dtype=int)
    rgb = np.empty(phase_idx.shape + (3,), dtype=np.uint8)
    rgb[...] = _UNFIT_COLOR
    fitted = np.isfinite(np.asarray(result.residual, dtype=float))
    for index in range(len(getattr(result, "phases_meta", None) or [1])):
        selected = fitted & (phase_idx == index)
        if selected.any():
            rgb[selected] = _PHASE_COLORS[index % len(_PHASE_COLORS)]
    return rgb


def _phase_legend(result) -> list[dict]:
    """``[{name, color, fraction}]`` — what each phase colour means and how much
    of the scan it claims, for the status line and the window's provenance."""

    phase_idx = np.asarray(result.phase_idx, dtype=int)
    fitted = np.isfinite(np.asarray(result.residual, dtype=float))
    total = max(int(fitted.sum()), 1)
    legend = []
    for index, meta in enumerate(getattr(result, "phases_meta", None) or []):
        count = int((fitted & (phase_idx == index)).sum())
        red, green, blue = _PHASE_COLORS[index % len(_PHASE_COLORS)]
        legend.append({"name": str(meta.get("name", f"phase {index}")),
                       "color": f"#{red:02x}{green:02x}{blue:02x}",
                       "fraction": count / total})
    return legend


def _build_result_windows(session, src, result, *, smooth=False,
                          ipf_tree=None) -> None:
    """Commit the fitted field: an IPF-Z orientation window (RGB), a strain
    window (εxx signal plot + εyy/εxy as chip-selectable views), and a phase map
    when more than one structure was in the library. *ipf_tree* is the
    orientation window opened before the fit, finished here rather than
    opened again."""
    from spyde.actions.commit import commit_result_tree
    base = src.metadata.get_item("General.title", "Signal")

    _build_ipf_heatmap(session, src, result, title="Orientation (IPF-Z)",
                       tree=ipf_tree)

    # One phase is the whole scan by construction, so a map of it says nothing.
    if len(getattr(result, "phases_meta", None) or []) > 1:
        legend = _phase_legend(result)
        commit_result_tree(
            session, title=f"{base} — Phase",
            primary=phase_map_rgb(result), primary_label="Phase",
            source_signal=src,
            provenance={"action": "Vector Orientation Mapping",
                        "source_title": base, "params": {"phases": legend}},
        )
        emit_status("Phase: " + ", ".join(
            f"{entry['name']} {entry['fraction']:.0%}" for entry in legend))

    _build_strain_window(session, src, result, smooth=smooth)


def _build_strain_window(session, src, result, *, smooth=False) -> None:
    """One window holding εxx, εyy and εxy — εxx is its signal plot and the
    other two are chip-selectable views of it, not separate trees."""
    from spyde.actions.commit import commit_result_tree
    from spyde.actions._common import (
        STRAIN_DISPLAY_SCALE, STRAIN_TITLES, strain_quantity,
    )
    base = src.metadata.get_item("General.title", "Signal")
    strain = np.asarray(result.smoothed_strain() if smooth else result.strain)
    components = ("exx", "eyy", "exy")
    # The fit is fractional; the maps read in percent, like the Strain window.
    maps = {c: strain[..., i] * STRAIN_DISPLAY_SCALE[c]
            for i, c in enumerate(components)}
    commit_result_tree(
        session, title=f"{base} — Strain",
        primary=maps["exx"], primary_label=STRAIN_TITLES["exx"],
        views=[(STRAIN_TITLES[c], maps[c]) for c in components[1:]],
        levels="auto_sym", cmap="coolwarm", source_signal=src,
        value_units={STRAIN_TITLES[c]: strain_quantity(c) for c in components},
        provenance={"action": "Vector Orientation Mapping",
                    "source_title": base, "params": {"smooth": bool(smooth)}},
    )

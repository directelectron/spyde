"""
ebsd_action.py — the staged "EBSD Indexing" wizard.

**This is the 4D-STEM orientation wizard, with bands instead of spots.** The
workflow is deliberately the same one, stage for stage, because it is the same
job: pick a crystal, build a library of simulated patterns, watch the match
under the crosshair until the parameters are right, then run the whole field
and get an IPF map.

    Orientation Mapping (4D-STEM)      EBSD Indexing (here)
    ------------------------------     -------------------------------------
    1 Load     .cif + voltage          1 Load     .cif + voltage + detector PC
    2 Library  angle res -> diffsims   2 Library  angle step -> simulated
               template library                   pattern dictionary
    3 Refine   matched template's      3 Refine   matched orientation's Kikuchi
               SPOTS on the live DP               BANDS on the live pattern
    4 Run      whole-field match ->    4 Run      whole-field index (+ optional
               IPF-Z map                          refinement) -> IPF-Z map

Everything downstream of the fit is literally shared code: the result is packed
into a :class:`~spyde.signals.orientation_map.SpyDEOrientationMap`, so the IPF
colouring, the 3-D IPF explorer, the point selector and the direction toggle
built for 4D-STEM all work here untouched. That reuse is the reason Wave 3's
display half is small (RELEASE_0_3_0_PLAN.md, 3.5).

The compute lives in :mod:`spyde.ebsd` (dictionary indexing, refinement,
preprocessing, band geometry); this module is only the interactive wiring.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.wizard import WizardController
from spyde.backend.ipc import emit, emit_error, emit_status

log = logging.getLogger(__name__)

DEFAULTS = dict(
    cif_path="",
    space_group=225,
    accelerating_voltage=20.0,
    min_dspacing=0.7,
    pc_x=0.5, pc_y=0.5, pc_z=0.55,
    step_deg=4.0,
    background="dynamic",
    background_sigma=8.0,
    n_bands=12,
    show_zone_axes=False,
    keep=4,
    refine=True,
    refine_steps=120,
)

#: Live-preview dictionaries above this many entries make each navigator move
#: cost more than a frame, so the Refine tab stops feeling live. The step size
#: is the user's lever; this only decides when to warn them.
_LIVE_DICT_WARN = 200_000


class EbsdWizard(WizardController):
    """Owns the EBSD wizard state: the phase and its reflectors, the simulated
    dictionary (resident on the compute device for the live match), the band
    overlay on the pattern plot, and the background correction that both the
    live match and the whole-field run must apply identically."""

    key = "ebsd"

    #: Declared parameter schema — one source of truth for every host (the
    #: Electron caret mirrors these; see registry._WIZARD_SCHEMAS).
    parameters = {
        "cif_path": {
            "name": "Crystal (.cif)", "type": "file", "default": "",
            "extensions": [".cif"], "tab": "Load",
        },
        "space_group": {
            "name": "Space group", "type": "int", "default": 225,
            "min": 1, "max": 230, "tab": "Load",
        },
        "accelerating_voltage": {
            "name": "Voltage (kV)", "type": "float", "default": 20.0,
            "min": 1.0, "max": 50.0, "step": 1.0, "tab": "Load",
        },
        "pc_x": {
            "name": "PC x", "type": "float", "default": 0.5,
            "min": 0.0, "max": 1.0, "step": 0.005, "tab": "Load",
        },
        "pc_y": {
            "name": "PC y", "type": "float", "default": 0.5,
            "min": 0.0, "max": 1.0, "step": 0.005, "tab": "Load",
        },
        "pc_z": {
            "name": "Detector distance", "type": "float", "default": 0.55,
            "min": 0.05, "max": 2.0, "step": 0.005, "tab": "Load",
        },
        "step_deg": {
            "name": "Angle step (°)", "type": "float", "default": 4.0,
            "min": 0.5, "max": 15.0, "step": 0.5, "tab": "Library",
        },
        "min_dspacing": {
            "name": "Min d-spacing (Å)", "type": "float", "default": 0.7,
            "min": 0.3, "max": 3.0, "step": 0.05, "tab": "Library",
        },
        "background": {
            "name": "Background", "type": "enum", "default": "dynamic",
            "choices": ["dynamic", "static", "both", "none"], "tab": "Library",
        },
        "background_sigma": {
            "name": "Background σ (px)", "type": "float", "default": 8.0,
            "min": 1.0, "max": 40.0, "step": 0.5, "tab": "Library",
        },
        "n_bands": {
            "name": "Bands drawn", "type": "int", "default": 12,
            "min": 1, "max": 80, "tab": "Refine",
        },
        "show_zone_axes": {
            "name": "Zone axes", "type": "bool", "default": False,
            "tab": "Refine",
        },
        "keep": {
            "name": "N best", "type": "int", "default": 4,
            "min": 1, "max": 20, "tab": "Run",
        },
        "refine": {
            "name": "Refine orientations", "type": "bool", "default": True,
            "tab": "Run",
        },
        "refine_steps": {
            "name": "Refine steps", "type": "int", "default": 120,
            "min": 10, "max": 600, "step": 10, "tab": "Run",
        },
    }

    def __init__(self, session, tree, *, phase, reflectors, indexer, euler,
                 detector, pc, background, background_sigma, static_ref,
                 voltage, overlay=None):
        super().__init__(session, tree)
        self.phase = phase
        self.reflectors = reflectors
        self.indexer = indexer            # SinglePatternIndexer (resident)
        self.euler = euler                # (D, 3) the dictionary's orientations
        self.detector = detector
        self.pc = pc
        self.background = background
        self.background_sigma = float(background_sigma)
        self.static_ref = static_ref
        self.voltage = float(voltage)
        self.overlay = overlay
        self.refine: dict = {}

    # ── the background correction, applied identically everywhere ─────────────
    @property
    def sim_sigma(self):
        """The high-pass the SIMULATED patterns must also get.

        Both sides of a cross-correlation have to go through the same filter.
        The ``dynamic`` pass is a high-pass and applies to both; the ``static``
        pass subtracts a detector artefact that simulated patterns do not have,
        so it applies to the experimental side only.
        """
        return (self.background_sigma if self.background in ("dynamic", "both")
                else None)

    def correct(self, patterns):
        """Apply the wizard's background correction to one pattern or a stack.

        The live match and the whole-field run MUST correct the same way: NCC
        is invariant to gain and offset but not to a spatial gradient, so a
        pattern corrected differently from the dictionary's expectations scores
        differently, and the crosshair preview would stop predicting the map
        (:mod:`spyde.ebsd.preprocess`).
        """
        if self.background in (None, "none"):
            return np.asarray(patterns, float)
        from spyde.ebsd.preprocess import remove_background
        arr = np.asarray(patterns, float)
        single = arr.ndim == 2
        if single:
            arr = arr[None]
        out = remove_background(arr, method=self.background,
                                sigma=self.background_sigma,
                                static_reference=self.static_ref)
        return out[0] if single else out

    def remove(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.overlay is not None and hasattr(self.overlay, "remove"):
            try:
                self.overlay.remove()
            except Exception as e:
                log.debug("removing EBSD band overlay failed: %s", e)
        self.overlay = None
        self.indexer = None
        if getattr(self.tree, "_ebsd_wizard", None) is self:
            self.tree._ebsd_wizard = None


def ebsd_indexing(ctx, action_name: str = "EBSD Indexing", **kwargs):
    """Parent toolbar action — a no-op; the Electron toolbar opens the staged
    EBSD wizard (which drives the ``ebsd_*`` handlers) instead."""
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detector_shape(signal) -> tuple[int, int]:
    """``(dy, dx)`` of one pattern."""
    axes = signal.axes_manager.signal_axes
    return (int(axes[1].size), int(axes[0].size))


def _stamped(signal, key, default=None):
    """A value stamped on synthetic data by ``spyde.data.synthetic``, if any.

    The bundled EBSD scan records the projection centre it was rendered with,
    so the wizard can open already pointing at the right geometry instead of
    making the user guess a PC before anything can possibly line up.
    """
    try:
        rec = signal.metadata.get_item("Spyde.synthetic")
    except Exception:
        return default
    if rec is None:
        return default
    try:
        val = rec[key] if not hasattr(rec, "get") else rec.get(key, default)
    except Exception:
        return default
    return default if val is None else val


def _resolve_pc(signal, payload) -> tuple[float, float, float]:
    """The projection centre: explicit payload values win, else the stamp on
    the data, else the centred default."""
    stamp = _stamped(signal, "pc")
    base = ([float(v) for v in np.asarray(stamp, float).reshape(3)]
            if stamp is not None
            else [DEFAULTS["pc_x"], DEFAULTS["pc_y"], DEFAULTS["pc_z"]])
    for i, key in enumerate(("pc_x", "pc_y", "pc_z")):
        if payload.get(key) is not None:
            base[i] = float(payload[key])
    return tuple(base)


def _nav_shape(signal) -> tuple[int, int]:
    nav = signal.axes_manager.navigation_shape       # (x, y) in hyperspy order
    return (int(nav[1]), int(nav[0]))


def _read_scan(signal) -> np.ndarray:
    """The whole scan as ``(ny, nx, dy, dx)`` float32.

    Deliberately materialised: every step of :mod:`spyde.ebsd` — background
    correction, the ADP map, indexing, refinement — is written whole-field, so
    there is nothing to stream into. EBSD patterns are small (a 60x60 detector
    is 3.6 kB), which is what makes that reasonable where the 4D-STEM memory
    rule forbids it.

    It is not free, though: at float32 a 256x256 scan of 60x60 patterns is
    940 MB and a 512x512 one is 3.8 GB. Indexing a scan that large wants the
    read chunked over navigation and fed to ``dictionary_index`` a block at a
    time — the function already tiles internally and takes a ``stopped_flag``,
    so the change is here rather than there.
    """
    data = signal.data
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data, np.float32)


def _phase_for(payload, signal):
    """The orix Phase to index with: a .cif if one was chosen, else a bare
    space-group phase (which still gives the right IPF colour key, just a
    generic cubic band set)."""
    from orix.crystal_map import Phase
    cif = payload.get("cif_path") or ""
    if cif:
        return Phase.from_cif(cif)
    sg = int(payload.get("space_group") or DEFAULTS["space_group"])
    return Phase(name="phase", space_group=sg)


def _emit_match(window_id, euler, score) -> None:
    """Stream the live single-pattern match to the caret's Refine tab."""
    emit({"type": "ebsd_match", "window_id": window_id, "ok": True,
          "phi1": float(np.rad2deg(euler[0])), "Phi": float(np.rad2deg(euler[1])),
          "phi2": float(np.rad2deg(euler[2])), "score": float(score)})


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — "Build Dictionary" (the analogue of om_generate_library)
# ─────────────────────────────────────────────────────────────────────────────

def ebsd_build_dictionary(session, plot, payload) -> None:
    """Sample orientation space, simulate a pattern for each, and switch on the
    LIVE band overlay: from here the matched orientation's Kikuchi bands are
    drawn on whatever pattern the navigator is sitting on."""
    src, tree = _src_plot_tree(session, plot)
    if src is None or tree is None:
        emit_error("Build Dictionary: no active dataset")
        return
    root = tree.root
    if root.axes_manager.signal_dimension != 2 or \
            root.axes_manager.navigation_dimension != 2:
        emit_error("EBSD Indexing needs a 2-D scan of 2-D patterns")
        return

    step = float(payload.get("step_deg", DEFAULTS["step_deg"]))
    voltage = float(payload.get("accelerating_voltage",
                                DEFAULTS["accelerating_voltage"]))
    min_d = float(payload.get("min_dspacing", DEFAULTS["min_dspacing"]))
    background = str(payload.get("background", DEFAULTS["background"]))
    bg_sigma = float(payload.get("background_sigma", DEFAULTS["background_sigma"]))
    n_bands = int(payload.get("n_bands", DEFAULTS["n_bands"]))
    zone_axes = bool(payload.get("show_zone_axes", DEFAULTS["show_zone_axes"]))
    pc = _resolve_pc(root, payload)
    detector = _detector_shape(root)
    window_id = getattr(src, "window_id", None) or payload.get("window_id")

    emit_status("EBSD: building the pattern dictionary…")

    def _work():
        try:
            from spyde.ebsd.bands import cubic_reflectors, reflectors_from_phase
            from spyde.ebsd.indexing import (
                SinglePatternIndexer, sample_orientations, simulate_dictionary,
            )

            phase = _phase_for(payload, root)
            try:
                reflectors = reflectors_from_phase(
                    phase, min_dspacing=min_d, voltage_kv=voltage)
            except Exception as e:
                log.debug("reflectors from phase failed (%s); generic cubic", e)
                reflectors = cubic_reflectors()

            # Sample the PHASE's fundamental zone: every distinct orientation
            # once. Falls back to an Euler grid if the phase has no usable
            # point group (sample_orientations owns that decision).
            euler = sample_orientations(
                step, point_group=getattr(phase, "point_group", None))
            n = len(euler)
            if n > _LIVE_DICT_WARN:
                emit_status(f"EBSD: {n:,} orientations at {step}° — the live "
                            f"preview will lag; raise the angle step to speed "
                            f"it up")
            emit_status(f"EBSD: simulating {n:,} patterns "
                        f"({len(reflectors)} reflectors)…")

            def _progress(done, total):
                if total:
                    emit_status(f"EBSD: simulating dictionary… "
                                f"{int(100 * done / total)}%")

            # The dictionary is high-passed as it is simulated, with the same
            # sigma the experimental patterns get — both sides of an NCC have
            # to come through the same filter (see EbsdWizard.sim_sigma).
            sim_sigma = (bg_sigma if background in ("dynamic", "both") else None)
            dic = simulate_dictionary(euler, detector, pc,
                                      reflectors=reflectors,
                                      background_sigma=sim_sigma,
                                      progress=_progress)
            if dic is None:
                return
            indexer = SinglePatternIndexer(dic, euler)

            # The static reference for background correction is the scan mean;
            # compute it once here, not per preview frame.
            static_ref = None
            if background in ("static", "both"):
                static_ref = _read_scan(root).mean(axis=(0, 1))

            # A rebuilt dictionary replaces the previous wizard wholesale, so
            # its overlay tears down with it rather than stacking a second set
            # of lines on the pattern.
            old = getattr(tree, "_ebsd_wizard", None)
            if old is not None and hasattr(old, "remove"):
                try:
                    old.remove()
                except Exception as e:
                    log.debug("removing prior EBSD wizard failed: %s", e)

            wiz = EbsdWizard(
                session, tree, phase=phase, reflectors=reflectors,
                indexer=indexer, euler=euler, detector=detector, pc=pc,
                background=background, background_sigma=bg_sigma,
                static_ref=static_ref, voltage=voltage,
            )
            from spyde.actions.ebsd_overlay import attach_ebsd_band_overlay
            wiz.overlay = attach_ebsd_band_overlay(
                src, root, indexer, reflectors, tree,
                detector=detector, pc=pc, correct=wiz.correct,
                n_bands=n_bands, show_zone_axes=zone_axes,
                on_match=lambda e, s: _emit_match(window_id, e, s),
            )
            # Published only once it is COMPLETE: `tree._ebsd_wizard` is what
            # every other stage tests for, and a refine or an overlay toggle
            # arriving while the overlay was still attaching would find None
            # and silently do nothing.
            tree._ebsd_wizard = wiz

            emit_status(f"EBSD: dictionary ready ({n:,} orientations) — "
                        f"move the crosshair to check the bands")
            emit({"type": "ebsd_dictionary_ready", "window_id": window_id,
                  "n_orientations": int(n), "n_reflectors": int(len(reflectors)),
                  "pc": [float(v) for v in pc]})
        except Exception as e:
            emit_error(f"Build Dictionary failed: {e}")
            log.exception("Build Dictionary failed")

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="ebsd-build-dictionary")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — "Refine" (the analogue of om_refine)
# ─────────────────────────────────────────────────────────────────────────────

def ebsd_refine(session, plot, payload) -> None:
    """Live-update the band-overlay knobs — how many bands to draw, whether to
    mark the zone axes, and the projection centre — and redraw at the current
    crosshair position.

    The PC belongs here rather than only on the Load tab because it is the one
    parameter you cannot set from first principles: you nudge it until the
    drawn lines sit on the bands, which is only possible with the overlay live
    in front of you.
    """
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_ebsd_wizard", None) if tree is not None else None
    if wiz is None or wiz.overlay is None:
        return
    params: dict = {}
    if payload.get("n_bands") is not None:
        params["n_bands"] = int(payload["n_bands"])
    if payload.get("show_zone_axes") is not None:
        params["show_zone_axes"] = bool(payload["show_zone_axes"])
    if any(payload.get(k) is not None for k in ("pc_x", "pc_y", "pc_z")):
        pc = _resolve_pc(tree.root, {**{"pc_x": wiz.pc[0], "pc_y": wiz.pc[1],
                                        "pc_z": wiz.pc[2]}, **payload})
        wiz.pc = pc
        params["pc"] = pc

    def _work():
        try:
            wiz.overlay.set_refine_params(**params)
            wiz.refine = dict(payload)
        except Exception as e:
            log.debug("ebsd_refine failed: %s", e)

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="ebsd-refine")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — "Index Map" (the analogue of om_run)
# ─────────────────────────────────────────────────────────────────────────────

def ebsd_run(session, plot, payload) -> None:
    """Index every pattern in the scan against the built dictionary, optionally
    refine each orientation off its dictionary entry, and open the IPF-Z map."""
    src, tree = _src_plot_tree(session, plot)
    wiz = getattr(tree, "_ebsd_wizard", None) if tree is not None else None
    if wiz is None or wiz.indexer is None:
        emit_error("Index Map: build the dictionary first")
        return
    keep = int(payload.get("keep", DEFAULTS["keep"]))
    do_refine = bool(payload.get("refine", DEFAULTS["refine"]))
    steps = int(payload.get("refine_steps", DEFAULTS["refine_steps"]))
    emit_status("EBSD: indexing the scan…")

    # torch's CUDA autograd backward segfaults the first time it runs on a
    # thread whose engine was never initialised; the refinement runs on the
    # worker, so warm the engine here on the dispatch thread first (the same
    # fix vector-orientation mapping needed — CLAUDE.md, GPU Computing).
    if do_refine:
        try:
            from spyde.actions.vector_orientation_gpu import warmup_autograd
            warmup_autograd()
        except Exception as e:
            log.debug("CUDA autograd warmup failed: %s", e)

    def _work():
        try:
            _run_indexing(session, tree, wiz, keep=keep, refine=do_refine,
                          steps=steps)
        except Exception as e:
            emit_error(f"Index Map failed: {e}")
            log.exception("Index Map failed")

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="ebsd-run")


def _run_indexing(session, tree, wiz, *, keep, refine, steps):
    """Whole-field index (+ optional refine) → the IPF window. Synchronous;
    call from a worker."""
    from spyde.ebsd.crystal_map import orientation_similarity_map
    from spyde.ebsd.indexing import dictionary_index
    from spyde.ebsd.preprocess import average_dot_product_map

    root = tree.root
    ny, nx = _nav_shape(root)
    scan = wiz.correct(_read_scan(root))
    dict_euler = wiz.euler

    ipf_tree = _open_ipf_window(session, root, ny, nx)
    # Closing EITHER the source or the result window stops the run — an index
    # of a large scan is long enough that "close it and it keeps going" reads
    # as a hang.
    stopped = [False]
    trees = {id(tree): tree, id(ipf_tree): ipf_tree}
    for t in trees.values():
        if t is not None and hasattr(t, "register_cancel"):
            t.register_cancel(flag=stopped)

    try:
        painter = _ProgressivePainter(session, ipf_tree, wiz, (ny, nx), dict_euler)
        # Cap the pattern tile so the map fills in as it goes: left alone the
        # tiling puts the whole scan in one chunk and nothing paints until the
        # end (see dictionary_index's pattern_chunk).
        chunk = max(1, (ny * nx) // 16)
        result = dictionary_index(
            scan, wiz.indexer, keep=keep, pattern_chunk=chunk,
            on_chunk=painter, stopped_flag=stopped,
            progress=lambda d, t: emit_status(
                f"EBSD: indexing… {int(100 * d / max(t, 1))}%"),
        )
        if result is None:
            if not getattr(tree, "_spyde_closed", False) and not stopped[0]:
                emit_error("EBSD: indexing returned no result")
            return None

        euler = dict_euler[result.best].reshape(ny, nx, 3)
        score = result.best_score.reshape(ny, nx)

        # Refinement is the tail of the SAME run, so it stays inside the
        # cancellation registration — unregistering after the index would let
        # a window closed during refinement carry on to paint into it.
        if refine and not stopped[0]:
            emit_status(f"EBSD: refining {ny * nx:,} orientations…")
            from spyde.ebsd.refine import refine_orientations
            ref = refine_orientations(
                scan, euler, detector=wiz.detector, pc=wiz.pc,
                reflectors=wiz.reflectors, background_sigma=wiz.sim_sigma,
                steps=steps,
                progress=lambda d, t: emit_status(
                    f"EBSD: refining… {int(100 * d / max(t, 1))}%"))
            euler = ref.euler_map((ny, nx))
            score = ref.score.reshape(ny, nx)
            log.debug("refinement improved %d/%d orientations",
                      int(ref.improved.sum()), ref.improved.size)

        if stopped[0]:
            return None

        om = _orientation_map(euler, score, wiz, root)
        osm = (orientation_similarity_map(result.indices, (ny, nx))
               if keep > 1 else None)
        try:
            adp = average_dot_product_map(scan)
        except Exception as e:
            log.debug("ADP map failed: %s", e)
            adp = None
    finally:
        for t in trees.values():
            if t is not None and hasattr(t, "unregister_cancel"):
                t.unregister_cancel(flag=stopped)

    tree.orientation_map = om

    _finalize_ipf_window(session, ipf_tree, om, score=score, osm=osm, adp=adp)
    emit_status(f"EBSD orientation map complete (mean NCC "
                f"{float(np.nanmean(score)):.3f})")
    return om


class _ProgressivePainter:
    """Paint the IPF map as indexing completes each slice of patterns.

    A dictionary index reports per chunk of PATTERNS, which in scan order is a
    contiguous run of positions — so the map fills top to bottom and you can
    see the grain structure emerge instead of watching a blank window. Costs
    one IPF colouring per chunk over the positions done so far.
    """

    def __init__(self, session, ipf_tree, wiz, nav_shape, dict_euler):
        from orix.plot import IPFColorKeyTSL

        self.session = session
        self.tree = ipf_tree
        self.wiz = wiz
        self.nav_shape = nav_shape
        self.dict_euler = dict_euler
        ny, nx = nav_shape
        self.rgb = np.zeros((ny, nx, 3), np.uint8)
        self.plot = next(iter(getattr(ipf_tree, "signal_plots", []) or []), None)
        # The SAME key SpyDEOrientationMap.ipf_color_map uses (the point
        # group's LAUE group). Built from anything else, the fill-in would be
        # coloured differently from the final map and the whole thing would
        # visibly change hue the moment the run finished.
        self._pg = wiz.phase.point_group
        self._key = IPFColorKeyTSL(self._pg.laue)

    def __call__(self, lo, hi, indices, scores) -> None:
        if self.plot is None:
            return
        try:
            from orix.quaternion import Orientation, Rotation
            rot = Rotation.from_euler(self.dict_euler[indices[:, 0]])
            rgb = self._key.orientation2color(Orientation(rot, symmetry=self._pg))
            self.rgb.reshape(-1, 3)[lo:hi] = np.clip(
                np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
        except Exception as e:
            log.debug("progressive IPF colouring failed: %s", e)
            return
        # Painting touches a live plot, so it has to happen on the main thread.
        frame = self.rgb.copy()

        def _paint():
            try:
                self.plot.needs_auto_level = True
                self.plot.set_data(frame)
            except Exception as e:
                log.debug("progressive IPF paint failed: %s", e)

        dispatch = getattr(self.session, "_dispatch_to_main", None)
        if dispatch is not None:
            dispatch(_paint)
        else:
            _paint()


def _orientation_map(euler, score, wiz, src):
    """Pack the indexed field into the SAME result object 4D-STEM orientation
    mapping produces, so every existing IPF view works on it unchanged."""
    from orix.quaternion import Rotation
    from spyde.signals.orientation_map import SpyDEOrientationMap, phase_to_dict

    ny, nx = euler.shape[:2]
    quats = np.asarray(Rotation.from_euler(euler.reshape(-1, 3)).data,
                       np.float32).reshape(ny, nx, 1, 4)
    corr = np.asarray(score, np.float32).reshape(ny, nx, 1)
    return SpyDEOrientationMap(
        quats=quats,
        corr=corr,
        phase_idx=np.zeros((ny, nx, 1), np.int16),
        mirror=np.ones((ny, nx, 1), np.int8),
        phases=[phase_to_dict(wiz.phase)],
        nav_axes=list(src.axes_manager.navigation_axes),
        params={"action": "EBSD Indexing", "pc": list(wiz.pc),
                "detector": list(wiz.detector),
                "voltage_kv": wiz.voltage,
                "n_dictionary": int(len(wiz.euler)),
                "background": wiz.background},
    )


def _open_ipf_window(session, src, ny, nx):
    """Open the IPF-Z window blank up front so the map has somewhere to fill."""
    from spyde.actions.commit import open_result_tree
    base = src.metadata.get_item("General.title", "Signal")
    return open_result_tree(
        session, title=f"{base} — Orientation (IPF-Z)",
        data=np.zeros((ny, nx), dtype=np.float32),
        provenance={"action": "EBSD Indexing", "source_title": base},
    )


def _finalize_ipf_window(session, tree, om, *, score=None, osm=None,
                         adp=None) -> None:
    """Paint the final IPF-Z map and attach the shared orientation views."""
    tree.orientation_map = om
    ipf = om.ipf_color_map(direction="z")            # (ny, nx, 3) uint8
    for sp in list(getattr(tree, "signal_plots", [])):
        try:
            sp.needs_auto_level = True
            sp.set_data(ipf)
        except Exception as e:
            log.debug("painting the EBSD IPF map failed: %s", e)
    try:
        from spyde.actions.ipf_view import attach_ipf_3d, attach_ipf_point_selector
        attach_ipf_3d(tree, om, direction="z")
        attach_ipf_point_selector(tree, om, "z")
    except Exception as e:
        log.debug("attaching the 3-D IPF explorer failed: %s", e)
    _attach_quality_views(session, tree, score=score, osm=osm, adp=adp)


def _attach_quality_views(session, tree, *, score=None, osm=None,
                          adp=None) -> None:
    """Add the quality maps as chip-selectable views on the IPF window.

    An IPF map alone cannot tell you where it is WRONG — every position gets a
    colour whether or not the match meant anything. The NCC map shows how well
    each pattern matched something; the orientation-similarity map is the one
    that exposes confidently-wrong indexing (a position that scored well but
    disagrees with its whole neighbourhood); the ADP map shows where the
    patterns were worth indexing at all.

    Same shape as ``commit.commit_result_tree``'s views, but that door needs
    the data up front and this window was opened blank to fill progressively —
    so the views are added here, once the run is done.
    """
    import hyperspy.api as hs
    from spyde.actions.views import emit_view_figure, register_views

    views = [(lbl, np.nan_to_num(np.asarray(m, np.float32)))
             for lbl, m in (("NCC", score), ("Similarity", osm), ("ADP", adp))
             if m is not None]
    if not views:
        return
    title = tree.root.metadata.get_item("General.title", "Orientation")
    for lbl, m in views:
        try:
            child = hs.signals.Signal2D(m.copy())
            child.metadata.General.title = f"{title} {lbl}"
            tree.add_node(tree.root, child, lbl)
            tree.update_plot_states(child)
        except Exception as e:
            log.debug("adding EBSD %r view node failed: %s", lbl, e)
    try:
        session._reemit_signal_tree(tree)
    except Exception as e:
        log.debug("re-emitting the EBSD result tree failed: %s", e)

    sp = next(iter(getattr(tree, "signal_plots", []) or []), None)
    wid = getattr(sp, "window_id", None) if sp is not None else None
    if wid is None:
        return
    try:
        sp.set_view_tag("IPF-Z", "2d")
    except Exception as e:
        log.debug("tagging the EBSD IPF view failed: %s", e)
    register_views(wid, views)
    for lbl, m in views:
        emit_view_figure(wid, m, lbl, kind="2d")

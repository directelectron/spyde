"""
dpc_action.py — the DPC wizard (``dpc_`` staged actions).

Differential phase contrast: the direct beam is deflected by the electric or
magnetic field it passes through, so tracking where it lands at every scan point
maps that field. The physics is in :mod:`spyde.actions.dpc`; the figures in
:mod:`spyde.actions.dpc_display`; this module is the interaction.

    dpc_open           caret mounted → measure the beam shifts once, open the
                       result window, report whether centering is even needed
    dpc_close          caret unmounted → tear it all down
    dpc_set_center     Center tab: none | manual | vacuum | corners
    dpc_set_beam       the beam region: off | circle | ring, and its geometry
    dpc_pick_center    Manual: adopt the beam region's centre
    dpc_load_vacuum    Vacuum tab: measure a second (vacuum) dataset
    dpc_auto_rotation  solve the scan↔detector rotation from the data
    dpc_tune           any live parameter → re-derive and repaint (cheap)
    dpc_set_view       swap the displayed map (RGB / Ex / Ey / |E| / div / curl)
    dpc_run            re-measure with a different method / search window
    dpc_commit         freeze the field as a new SignalTree

**Measure once, tune forever.** The only expensive step is the beam-shift pass
over the dataset. It runs at open (and again only if the *method*, the search
window or the beam region changes) and the ``(ny, nx, 2)`` result is cached.
Centering, rotation, handedness and calibration are then all pure arithmetic on
that small array, which is what lets the rotation slider be genuinely live
instead of a click-and-wait. Do not move the measure into ``dpc_tune``.

**That one pass STREAMS on lazy data.** It is dispatched per navigation chunk
through ``ComputeBackend.compute_chunks_progressive`` and the map repaints as
each chunk lands, so a scan that takes minutes shows a field filling in rather
than a spinner. Two properties make that work and are worth not breaking:

* the lazy graph keeps the dataset's own nav chunking (no rechunk layer), so a
  streamed "chunk" is a STORAGE chunk and the dispatch granularity matches what
  the reader actually reads — Live-Display §1;
* partial state is ``NaN``, which every downstream stage already tolerates: the
  plane fits mask on ``isfinite``, the rotation estimator drops non-finite
  gradients, and the display paints non-finite black instead of letting one
  poison the contrast. So the map genuinely fills in rather than appearing at
  the end.

Eager data (already in RAM) has nothing to stream and runs in one go.

**The three ways to find zero.** A DPC map is a map of DIFFERENCES from the
undeflected beam position, so it is only as good as the zero it is measured
against — and the instrument's own descan drifts across a scan, which looks
exactly like a slowly varying field. The Center tab offers, in increasing order
of trustworthiness:

* **Manual** — one centre for every pattern, taken from the beam region below.
  Removes a constant offset, nothing else. Fine when the descan is already good.
* **Corners** — the beam centre is measured in four boxes at the corners of the
  scan and a plane is fitted through them. Assumes the corners are off the
  feature of interest. No extra data needed, and it removes a RAMP, not just an
  offset.
* **Vacuum** — a second dataset acquired in vacuum with the same scan settings.
  Contains only descan, so subtracting it is exact. The gold standard, at the
  cost of acquiring it.

``dpc_open`` measures the residual descan first (:func:`dpc.centering_report`)
and says when there is nothing to remove, so an already-centred dataset (Center
Zero Beam has run, or the microscope was well set up) skips the step instead of
having a correction applied to it for no reason.

**The beam region is one shape doing two jobs**, and is deliberately not owned
by any single Center mode. Its AREA is the centre-of-mass mask — a diffracted
disc inside the search area drags the centroid, often by more than the field
being measured — and its CENTRE is what Manual subtracts. Two separate controls
for one physical question ("where is the direct beam?") could disagree with each
other; one cannot. ``ring`` is for a saturated or beam-stopped beam, where the
centroid has to come from the disc edge instead of its core.

**Rotation is not cosmetic.** The detector's x/y and the scan's x/y are related
by an unknown rotation — and possibly a handedness flip. Get it wrong and every
direction on the map is wrong, which is the single easiest way to publish a
wrong DPC figure. :func:`dpc.estimate_rotation` solves it from the data using
the symmetry the field must have (electric fields are curl-free, magnetic
deflections divergence-free); the caret shows the improvement so the user can
see whether the fit actually found anything. The remaining 180° ambiguity is
physics the data cannot settle, so it stays a user toggle.

The result window is a bare ``figure`` (not a registered ``Plot``), so it
registers a controller via ``own_window`` and keeps its figure referenced with
``figure_registry.keep_alive`` — ``actions/README.md`` §6.
"""
from __future__ import annotations

import concurrent.futures
import logging

import numpy as np

from spyde.actions import dpc as _dpc
from spyde.actions import dpc_display as _display
from spyde.actions.context import current_signal as _current_signal
from spyde.actions.context import src_plot_tree as _src_plot_tree
from de_shell.actions.wizard import WizardController
from de_shell.ipc import emit, emit_error, emit_progress, emit_status

log = logging.getLogger(__name__)

#: Every live parameter, with the value the caret opens on. The renderer's
#: DpcWizard.tsx DEFAULTS must agree key-for-key — a drifting TSX default wins
#: silently (see the caret-defaults trap in CLAUDE.md), which is why
#: ``test_dpc_action.py`` parses the TSX and compares.
DEFAULTS: dict = {
    "method": "center_of_mass",
    "half_square_width": 0,
    "center_mode": "corners",
    "corner_fraction": 0.05,
    # The beam region (see BeamRegion): "off" | "circle" | "ring". It opens OFF
    # so the measurement is byte-for-byte what it was before this control
    # existed; the radii are filled in from the detector size the first time it
    # is switched on, because a default in pixels cannot know the frame size.
    "beam_shape": "off",
    "beam_cx": 0.0,
    "beam_cy": 0.0,
    "beam_r": 0.0,
    "beam_r_inner": 0.0,
    "mode": "magnetic",
    "rotation": 0.0,
    "flip": False,
    "reverse": False,
    "thickness_nm": 60.0,
    "beam_energy_kv": 200.0,
    "mrad_per_px": 0.0,
    "view": "rgb",
    "autolim_sigma": 4.0,
}

#: Colours for the on-plot furniture. Distinct from the navigator's green
#: crosshair and from Center Zero Beam's yellow, so two open carets never look
#: like one.
_CORNER_COLOR = "#ff3030"      # the four corner boxes on the navigator (as
                               # in vector_overlay — this repo's on-plot red)
_BEAM_COLOR = "#94e2d5"        # the beam region (circle / ring) on the DP

#: The corner boxes sit on a navigator that is usually busy, so they carry a
#: heavier edge than the other furniture to stay findable against it.
_CORNER_LINEWIDTH = 3.0

#: Re-measuring the whole scan is the one expensive step, so a DRAG must not
#: trigger it per frame. The widget and the readouts follow the pointer live;
#: the re-measure waits this long after motion stops. Same shape as the drift
#: caret's ROI settle, for the same reason.
_REGION_SETTLE_S = 0.45

#: Bare-figure window geometry. A bare figure never receives ``resize_figure``,
#: so its initial px size is the one it keeps and anything drawn outside is
#: CLIPPED by the subwindow — see the same note in ``drift_action``.
_FIG_WIDTH, _FIG_HEIGHT = 340, 300


class DpcWizard(WizardController):
    """Owns one live DPC analysis: the cached beam shifts, the current
    parameters, the result window, and the overlays on the source windows."""

    key = "dpc"

    #: The declared schema — one source of truth for every host (the Electron
    #: caret, a notebook form, generated docs). Same spec as toolbars.yaml
    #: ``parameters:``; resolved via ``registry.wizard_parameters("dpc")``.
    parameters = {
        "method": {
            "name": "Beam finder", "type": "enum",
            "default": DEFAULTS["method"], "choices": list(_dpc.BEAM_METHODS),
            "tab": "Center",
        },
        "half_square_width": {
            "name": "Search window (px, 0=full)", "type": "int",
            "default": DEFAULTS["half_square_width"], "min": 0, "max": 512,
            "tab": "Center",
        },
        "center_mode": {
            "name": "Reference", "type": "enum",
            "default": DEFAULTS["center_mode"], "choices": list(_dpc.CENTER_MODES),
            "tab": "Center",
        },
        "corner_fraction": {
            "name": "Corner box size", "type": "float",
            "default": DEFAULTS["corner_fraction"], "min": 0.01, "max": 0.45,
            "step": 0.01, "tab": "Center",
        },
        "beam_shape": {
            "name": "Beam region", "type": "enum",
            "default": DEFAULTS["beam_shape"], "choices": list(_dpc.BEAM_SHAPES),
            "tab": "Center",
        },
        "beam_cx": {
            "name": "Beam x (px)", "type": "float", "default": DEFAULTS["beam_cx"],
            "min": 0.0, "max": 100000.0, "step": 0.5, "tab": "Center",
            "display_condition": {"beam_shape": ["circle", "ring"]},
        },
        "beam_cy": {
            "name": "Beam y (px)", "type": "float", "default": DEFAULTS["beam_cy"],
            "min": 0.0, "max": 100000.0, "step": 0.5, "tab": "Center",
            "display_condition": {"beam_shape": ["circle", "ring"]},
        },
        "beam_r": {
            "name": "Radius (px)", "type": "float", "default": DEFAULTS["beam_r"],
            "min": 0.0, "max": 100000.0, "step": 0.5, "tab": "Center",
            "display_condition": {"beam_shape": ["circle", "ring"]},
        },
        "beam_r_inner": {
            "name": "Inner radius (px)", "type": "float",
            "default": DEFAULTS["beam_r_inner"], "min": 0.0, "max": 100000.0,
            "step": 0.5, "tab": "Center",
            "display_condition": {"beam_shape": "ring"},
        },
        "mode": {
            "name": "Field", "type": "enum", "default": DEFAULTS["mode"],
            "choices": list(_dpc.FIELD_MODES), "tab": "Field",
        },
        "thickness_nm": {
            "name": "Thickness (nm)", "type": "float",
            "default": DEFAULTS["thickness_nm"], "min": 0.1, "max": 10000.0,
            "step": 1.0, "tab": "Field",
            "display_condition": {"mode": "electric"},
        },
        "beam_energy_kv": {
            "name": "Beam energy (kV)", "type": "float",
            "default": DEFAULTS["beam_energy_kv"], "min": 1.0, "max": 1000.0,
            "step": 1.0, "tab": "Field",
            "display_condition": {"mode": "electric"},
        },
        "mrad_per_px": {
            "name": "Detector scale (mrad/px, 0=auto)", "type": "float",
            "default": DEFAULTS["mrad_per_px"], "min": 0.0, "max": 100.0,
            "step": 0.001, "tab": "Field",
        },
        "rotation": {
            "name": "Rotation (deg)", "type": "float",
            "default": DEFAULTS["rotation"], "min": 0.0, "max": 360.0,
            "step": 0.5, "tab": "Rotation",
        },
        "flip": {
            "name": "Flip handedness", "type": "bool",
            "default": DEFAULTS["flip"], "tab": "Rotation",
        },
        "reverse": {
            "name": "Reverse (+180°)", "type": "bool",
            "default": DEFAULTS["reverse"], "tab": "Rotation",
        },
        # Both of these live on the Map tab in the caret. The schema is what a
        # host that builds its own form reads (docs, notebook), so a tab here
        # that disagrees with the caret puts the same control in two places
        # depending on who renders it.
        "view": {
            "name": "Map", "type": "enum", "default": DEFAULTS["view"],
            "choices": list(_display.VIEWS), "tab": "Map",
        },
        "autolim_sigma": {
            "name": "Colour limit (σ)", "type": "float",
            "default": DEFAULTS["autolim_sigma"], "min": 0.5, "max": 10.0,
            "step": 0.5, "tab": "Map",
        },
    }

    def __init__(self, session, tree, src_plot, *, params: dict | None = None):
        super().__init__(session, tree)
        self.src_plot = src_plot
        self.params = dict(DEFAULTS)
        self.params.update(params or {})
        self.shifts: np.ndarray | None = None       # the cached (ny, nx, 2)
        self.vacuum_shifts: np.ndarray | None = None
        self.vacuum_label: str = ""
        self.report: _dpc.CenteringReport | None = None
        self.estimate: _dpc.RotationEstimate | None = None
        self.result: _dpc.DpcResult | None = None
        self.window_id: int | None = None
        self.plot = None                            # the map Plot2D
        self.wheel = None                           # the colour-wheel KeyOverlay
        self.clim: tuple[float, float] | None = None
        self.cmap: str | None = None
        self._corner_mg = None                      # navigator corner boxes
        self._beam_widget = None                    # the circle/ring on the DP
        self._beam_handler = None                   # kept alive (weak callback)
        self._beam_dragging = False                 # re-entrancy guard
        self._settle_timer = None                   # drag → debounced re-measure
        self._measure_stop: list | None = None      # in-flight pass's cancel token
        self._measure_future = None                 # …and its future, if any
        self._last_brightness = None                # re-sent during a drag

    # ── the source signal ────────────────────────────────────────────────────

    @property
    def signal(self):
        return _current_signal(self.src_plot)

    def _nav_shape(self) -> tuple[int, int]:
        if self.shifts is not None:
            return tuple(int(n) for n in self.shifts.shape[:2])
        am = self.signal.axes_manager
        return tuple(int(n) for n in am.navigation_shape)[::-1]

    def _sig_shape(self) -> tuple[int, int]:
        ax = self.signal.axes_manager.signal_axes
        return (int(ax[1].size), int(ax[0].size))

    # ── stage 1: measure (the only expensive step) ───────────────────────────

    def region(self) -> _dpc.BeamRegion:
        """The beam region the caret's parameters describe."""
        p = self.params
        return _dpc.BeamRegion(shape=str(p.get("beam_shape", "off")),
                               cx=float(p.get("beam_cx") or 0.0),
                               cy=float(p.get("beam_cy") or 0.0),
                               r=float(p.get("beam_r") or 0.0),
                               r_inner=float(p.get("beam_r_inner") or 0.0))

    def measure(self, *, on_done=None) -> None:
        """Measure the direct-beam position over the whole scan, off-thread.

        On a LAZY dataset this streams: the pass is dispatched per nav chunk and
        the map is repainted as each lands, so a multi-minute scan shows a field
        filling in rather than a spinner. On eager data (already in RAM) there
        is nothing to stream and it runs in one go.
        """
        if self.signal is None:
            emit_error("DPC: no active dataset")
            return
        # A superseded pass is CANCELLED, not left running to have its result
        # discarded — a beam-shift pass reads the whole scan, so one nobody is
        # waiting for costs the cluster the entire dataset. Same contract as
        # virtual_image (cancel prior → unregister → register new).
        self._cancel_measure()
        # A pass gets its OWN signal object, taken here on the dispatch thread.
        # The worker below runs hyperspy `map` on it, and a new pass can still
        # overlap the TAIL of the one it cancels: a queued future cancels
        # cleanly, one already inside `map` does not. See dpc.private_view.
        signal = _dpc.private_view(self.signal)
        method = str(self.params["method"])
        hw = int(self.params["half_square_width"] or 0)
        region = self.region()
        sig_shape = self._sig_shape()
        gen = self.guard()
        # Only the EAGER branch below uses this one; the progressive branch owns
        # its own token, because that is where a pass can actually be stopped
        # part-way (see _measure_progressive).
        eager_stop: list = [False]
        emit_status("DPC: locating the direct beam…")

        def _finish(shifts):
            self._retire_measure(eager_stop)
            if eager_stop[0] or not self.still(gen) or self._closed:
                return
            self.shifts = np.asarray(shifts, dtype=np.float64)
            self.report = _dpc.centering_report(self.shifts)
            self._warn_if_region_missed(self.shifts, region, sig_shape)
            self._set_computing(False)
            emit_progress(1, 1, "DPC")
            self.emit_state()
            self.refresh()
            if on_done is not None:
                on_done()

        if self._measure_progressive(signal, method, hw, region, gen, _finish):
            return

        # Eager data: ONE hyperspy `map` over an array already in RAM. There is
        # no interruption point inside it, so the token can only stop the pass
        # BEFORE it starts and drop the result if it lands late — which is also
        # all `virtual_image` does for eager data. Registering it still matters:
        # closing the tree flips it, so a pass in the queue never begins.
        self._track_measure(eager_stop)

        def _work():
            if eager_stop[0]:
                return None
            return _dpc.measure_beam_shifts(signal, method=method,
                                            half_square_width=hw, region=region)

        def _done(shifts):
            if shifts is not None:
                _finish(shifts)

        self.run_on_worker(_work, name="dpc-measure", on_done=_done,
                           on_error=lambda e: self._measure_failed(gen, e))

    def _track_measure(self, stop: list, future=None) -> None:
        """Adopt *stop*/*future* as the in-flight pass and register them on the
        tree, so closing the tree mid-pass stops it (``register_cancel``)."""
        self._measure_stop, self._measure_future = stop, future
        tree = self.tree
        reg = getattr(tree, "register_cancel", None)
        if reg is not None:
            reg(flag=stop, future=future)

    def _retire_measure(self, stop: list) -> None:
        """Drop a FINISHED pass's token. Without this the tree's cancel registry
        gains an entry per measure — and every drag settle is a measure."""
        if self._measure_stop is not stop:
            return                      # already superseded; not ours to drop
        future = self._measure_future
        self._measure_stop = self._measure_future = None
        unreg = getattr(self.tree, "unregister_cancel", None)
        if unreg is not None:
            try:
                unreg(flag=stop, future=future)
            except Exception as e:                           # pragma: no cover
                log.debug("retiring the DPC measure token failed: %s", e)

    def _cancel_measure(self) -> None:
        """Stop the in-flight beam-shift pass, if any.

        Setting the flag is what actually stops it: the progressive path checks
        it before each chunk submit and inside each chunk task, so a superseded
        pass stops dispatching instead of computing a result nobody reads.
        Cancelling the future kills one still queued; one already running ends
        at its next flag check. Then unregister both, or the tree's cancel list
        grows by one entry per drag.
        """
        stop, future = self._measure_stop, self._measure_future
        self._measure_stop = self._measure_future = None
        if stop is None and future is None:
            return
        if stop is not None:
            stop[0] = True
        if future is not None:
            try:
                if not future.done():
                    future.cancel()
            except Exception as e:
                log.debug("cancelling the prior DPC measure failed: %s", e)
        unreg = getattr(self.tree, "unregister_cancel", None)
        if unreg is not None:
            try:
                unreg(flag=stop, future=future)
            except Exception as e:                           # pragma: no cover
                log.debug("unregistering the prior DPC measure failed: %s", e)

    def _measure_failed(self, gen: int, exc: Exception) -> None:
        """Report a failed pass — unless nobody is waiting for it any more.

        A measure that is still running when the caret closes (or the app quits)
        fails on the way down: the executor it is submitting into is already
        shut down. Reporting that shows the user "locating the direct beam
        failed" for something they caused deliberately and that has no
        consequence. A superseded run is the same story — a newer measure has
        already replaced it.
        """
        if self._closed or not self.still(gen):
            log.debug("DPC measure abandoned after close/supersede: %s", exc)
            return
        emit_error(f"DPC: locating the direct beam failed: {exc}")

    def _measure_progressive(self, signal, method, hw, region, gen, on_finish
                             ) -> bool:
        """Stream the beam-shift pass per nav chunk. False → not applicable.

        Uses ``ComputeBackend.compute_chunks_progressive``, which dispatches one
        task per nav chunk and calls back from a worker thread as each lands.
        The lazy graph keeps the dataset's own nav chunking (no rechunk layer),
        so a "chunk" here is a storage chunk — the streaming granularity matches
        what the reader actually reads (Live-Display §1).

        Partial state is NaN, which every stage downstream already tolerates:
        the plane fits mask on ``isfinite``, the rotation estimator drops
        non-finite gradients, and the display paints non-finite black rather
        than letting one poison the contrast. So the map genuinely fills in.
        """
        backend = getattr(self.session, "compute_backend", None)
        if backend is None or not hasattr(backend, "compute_chunks_progressive"):
            return False
        try:
            graph = _dpc.beam_shift_graph(signal, method=method,
                                          half_square_width=hw, region=region)
        except Exception as e:
            log.debug("building the DPC beam-shift graph failed: %s", e)
            return False
        if graph is None:                       # eager data — nothing to stream
            return False

        ny, nx = int(graph.shape[0]), int(graph.shape[1])
        partial = np.full((ny, nx, 2), np.nan, dtype=np.float64)
        total = max(1, len(graph.chunks[0]) * len(graph.chunks[1]))
        done = [0]
        # The cancel token travels WITH the dispatch: the backend checks it
        # before each chunk submit and inside each chunk task, so superseding
        # this pass stops it rather than letting it finish unread.
        stop: list = [False]
        self.shifts = partial
        self._set_computing(True)

        def _on_chunk(chunk, slices):
            """Worker thread — marshal the paint onto the main loop."""
            if stop[0] or self._closed or not self.still(gen):
                return
            try:
                partial[slices] = np.asarray(chunk, dtype=np.float64)
            except Exception as e:                           # pragma: no cover
                log.debug("storing a DPC chunk failed: %s", e)
                return
            done[0] += 1
            n = done[0]
            dispatch = getattr(self.session, "_dispatch_to_main", None)
            paint = lambda: self._on_partial(gen, n, total)   # noqa: E731
            dispatch(paint) if dispatch is not None else paint()

        try:
            future = backend.compute_chunks_progressive(graph, 2, _on_chunk,
                                                        stopped_flag=stop)
        except Exception as e:
            log.debug("progressive DPC dispatch failed (%s) — running in one "
                      "pass instead", e)
            return False
        self._track_measure(stop, future)

        def _settled(fut):
            self._retire_measure(stop)
            try:
                result = fut.result()
            except concurrent.futures.CancelledError:
                # Cancelled deliberately (superseded, or the caret/tree closed).
                # Not a failure and not the user's problem — say nothing.
                self._set_computing(False)
                return
            except Exception as e:
                if stop[0]:              # torn down mid-pass; same story
                    self._set_computing(False)
                    return
                self._measure_failed(gen, e)
                self._set_computing(False)
                return
            dispatch = getattr(self.session, "_dispatch_to_main", None)
            finish = lambda: on_finish(result)                # noqa: E731
            dispatch(finish) if dispatch is not None else finish()

        future.add_done_callback(_settled)
        return True

    def _on_partial(self, gen: int, done: int, total: int) -> None:
        """Repaint from what has landed so far (main thread)."""
        if self._closed or not self.still(gen):
            return
        emit_progress(done, total, "DPC: locating the direct beam")
        self.refresh()

    def _set_computing(self, computing: bool) -> None:
        """Drive the window's "Calculating…" overlay. Every True is paired."""
        try:
            from de_shell.ipc import emit_window_computing
            emit_window_computing(self.window_id, bool(computing))
        except Exception as e:                               # pragma: no cover
            log.debug("DPC computing marker failed: %s", e)

    def _warn_if_region_missed(self, shifts, region, sig_shape) -> None:
        """Say so when the region is not actually on the beam.

        This cannot be left to the numbers looking wrong, because they do not:
        a region sitting on empty detector still returns the centroid of
        whatever is inside it — a finite, plausible position — and every map
        downstream is then confidently wrong.

        BRIGHTNESS is the check that works. Containment does NOT: the centroid
        of a non-negative distribution over a CONVEX region always lands inside
        it, so a disc anywhere on the detector passes that test trivially (see
        ``dpc.beam_inside_region``). Containment still earns its place for a
        RING, where a centroid in the hole means it is not concentric with the
        beam.
        """
        if not region.active:
            return
        if np.isnan(np.asarray(shifts)).all():
            emit_error("DPC: the beam region captured no intensity anywhere — "
                       "drag it onto the direct beam.")
            return
        brightness = _dpc.region_brightness(self.signal, region)
        if np.isfinite(brightness) and brightness < _dpc.BEAM_DIM_THRESHOLD:
            emit_status(
                f"DPC: warning — the beam region is dimmer than the detector "
                f"average ({brightness:.2f}×), so it is not on the direct beam. "
                f"Drag it over, or widen it.")
            return
        if region.shape == "ring" and not _dpc.beam_inside_region(
                shifts, region, sig_shape):
            emit_status("DPC: warning — the beam was found in the ring's HOLE, "
                        "so the ring is not centred on it.")

    # ── stage 2: derive (pure arithmetic on the cached shifts) ───────────────

    def manual_center(self) -> tuple[float, float] | None:
        """The Manual reference position: an explicit pick, else the region's
        own centre.

        A region the user has already dragged onto the beam HAS answered "where
        is the undeflected beam" — asking them to place a second marker saying
        the same thing would be busywork, and the two could then disagree.
        """
        cx, cy = self.params.get("cx"), self.params.get("cy")
        if cx is not None and cy is not None:
            return (float(cx), float(cy))
        r = self.region()
        return r.center if r.active else None

    def reference(self) -> np.ndarray | None:
        """The descan reference for the current Center mode, or ``None``.

        ``strict=False``: a caret sitting on Manual before a centre has been
        placed, or on Vacuum before a dataset has been chosen, is
        mid-interaction — "no reference yet" is a valid state that must render,
        not an error that blanks the window.
        """
        if self.shifts is None:
            return None
        return _dpc.resolve_reference(
            self.shifts, center_mode=str(self.params["center_mode"]),
            corner_fraction=float(self.params["corner_fraction"]),
            center_xy=self.manual_center(),
            vacuum_shifts=self.vacuum_shifts, sig_shape=self._sig_shape(),
            strict=False)

    def derive(self) -> _dpc.DpcResult | None:
        """Re-run everything downstream of the measure. Milliseconds."""
        if self.shifts is None:
            return None
        p = self.params
        scale = float(p.get("mrad_per_px") or 0.0) or None
        try:
            result = _dpc.compute_dpc(
                self.signal, shifts=self.shifts, reference=self.reference(),
                mode=str(p["mode"]), center_mode=str(p["center_mode"]),
                corner_fraction=float(p["corner_fraction"]),
                rotation=float(p["rotation"]), flip=bool(p["flip"]),
                reverse=bool(p["reverse"]),
                thickness_nm=float(p["thickness_nm"]),
                beam_energy_kev=float(p["beam_energy_kv"]),
                mrad_per_px=scale, autolim_sigma=float(p["autolim_sigma"]),
                # Carried for provenance only — `shifts` is already measured, so
                # this cannot change the numbers, but a committed tree should
                # record which pixels the centroid was taken over.
                region=self.region())
        except Exception as e:
            emit_error(f"DPC: {e}")
            log.exception("DPC derive failed")
            return None
        result.estimate = self.estimate
        result.centering = self.report
        self.result = result
        return result

    def refresh(self) -> None:
        """Derive and repaint the map (opening the window on the first call)."""
        result = self.derive()
        if result is None:
            return
        if self.window_id is None:
            self._open_window(result)
        else:
            _display.update_dpc_view(self.plot, self.wheel, result,
                                     str(self.params["view"]),
                                     clim=self.clim, cmap=self.cmap)
        self._emit_histogram()
        self.emit_result()

    # ── the result window ────────────────────────────────────────────────────

    def _open_window(self, result: _dpc.DpcResult) -> None:
        from de_shell.actions.figure_registry import keep_alive
        try:
            fig, fig_id, html, plot, wheel = _display.build_dpc_figure(
                result, view=str(self.params["view"]), title=self._title())
        except Exception as e:
            emit_error(f"DPC: building the result window failed: {e}")
            log.exception("DPC window build failed")
            return
        wid = int(self.session.next_window_id())
        keep_alive(wid, fig)
        self.window_id, self.plot, self.wheel = wid, plot, wheel
        emit({"type": "figure", "fig_id": fig_id, "window_id": wid,
              "html": html, "title": self._title(), "is_navigator": False,
              "aspect": _FIG_WIDTH / float(_FIG_HEIGHT)})
        self.own_window(wid)

    #: The live window's title. Deliberately does NOT name the field type.
    #:
    #: It used to read "DPC (B field)" / "DPC (E field)", set once at open — and
    #: switching Magnetic→Electric left the old label in place, because the
    #: title only travels with a full ``figure`` message and re-sending the
    #: whole HTML on every tune to fix a caption is not a trade worth making.
    #: A stale label is worse than no label: the readout in the caret already
    #: names the units (MV/cm vs mrad), and the COMMITTED tree does get the
    #: specific title.
    WINDOW_TITLE = "DPC Field Map"

    def _title(self) -> str:
        return self.WINDOW_TITLE

    def _emit_histogram(self) -> None:
        if self.result is None:
            return
        _display.emit_dpc_histogram(self.window_id, self.result,
                                    str(self.params["view"]), self.clim)

    # ── plot-widget dock integration (session controller fallback) ───────────

    def set_clim(self, vmin, vmax) -> None:
        try:
            self.clim = (float(vmin), float(vmax))
            self.plot.set_clim(*self.clim)
        except Exception as e:                               # pragma: no cover
            log.debug("DPC set_clim failed: %s", e)

    def auto_clim(self, mode: str = "robust") -> None:
        """Dock Auto / Reset — drop the manual override and re-derive."""
        self.clim = None
        if self.result is None:
            return
        view = str(self.params["view"])
        if mode == "full" and view != _display.RGB_VIEW:
            arr = np.asarray(self.result.component(view), float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                self.clim = (float(finite.min()), float(finite.max()))
        _display.update_dpc_view(self.plot, self.wheel, self.result, view,
                                 clim=self.clim, cmap=self.cmap)
        self._emit_histogram()

    def set_colormap(self, name: str) -> None:
        try:
            self.cmap = str(name)
            self.plot.set_colormap(self.cmap)
        except Exception as e:                               # pragma: no cover
            log.debug("DPC set_colormap failed: %s", e)

    # ── overlays on the SOURCE windows ───────────────────────────────────────

    def _navigator_plot2d(self):
        """The source tree's navigator plot — where the corner boxes go.

        The corner boxes select SCAN positions, so they belong on the navigator,
        not on the diffraction pattern. A tree with no navigator (a single 2-D
        image) simply gets no boxes.
        """
        tree = self.tree
        npm = getattr(tree, "navigator_plot_manager", None) if tree else None
        if npm is None:
            return None
        pw = next(iter(getattr(npm, "plot_windows", {}) or {}), None)
        if pw is None:
            return None
        plots = npm.plots.get(pw) or []
        return getattr(plots[0], "_plot2d", None) if plots else None

    def show_corner_boxes(self) -> None:
        """Draw (or resize) the four corner boxes the plane is fitted through.

        Static markers, not draggable widgets: their geometry IS
        ``corner_fraction``, so the slider is the only sensible way to change
        them and a drag would have nowhere to write back to. Geometry comes
        from :func:`dpc.corner_boxes`, the same source the fit mask does, so
        what is drawn is exactly what is fitted.
        """
        plot2d = self._navigator_plot2d()
        if plot2d is None:
            return
        boxes = _dpc.corner_boxes(self._nav_shape(),
                                  float(self.params["corner_fraction"]))
        # add_rectangles takes CENTRES + sizes; corner_boxes gives (x, y, w, h)
        # where x and y are pixel INDICES. Pixel i covers [i - 0.5, i + 0.5], so
        # a block over indices 0..1 spans [-0.5, 1.5] and is centred on 0.5 —
        # not on x + w/2, which is 1.0. Without the half-pixel every box sits
        # shifted toward the bottom-right: a gap inside the top-left corner, an
        # overhang past the bottom-right edge, and the drawn box no longer
        # covers the pixels the plane is actually fitted through.
        offsets = [[x + w / 2.0 - 0.5, y + h / 2.0 - 0.5]
                   for (x, y, w, h) in boxes]
        widths = [w for (_x, _y, w, _h) in boxes]
        heights = [h for (_x, _y, _w, h) in boxes]
        if self._corner_mg is not None:
            try:
                self._corner_mg.set(offsets=offsets, widths=widths,
                                    heights=heights)
                return
            except Exception as e:                           # pragma: no cover
                log.debug("resizing the DPC corner boxes failed: %s", e)
        try:
            self._corner_mg = plot2d.add_rectangles(
                offsets, widths, heights, name="dpc_corners",
                edgecolors=_CORNER_COLOR, facecolors=_CORNER_COLOR,
                linewidths=_CORNER_LINEWIDTH, alpha=0.22)
        except Exception as e:                               # pragma: no cover
            log.debug("drawing the DPC corner boxes failed: %s", e)

    def hide_corner_boxes(self) -> None:
        if self._corner_mg is not None:
            try:
                self._corner_mg.remove()
            except Exception as e:                           # pragma: no cover
                log.debug("removing the DPC corner boxes failed: %s", e)
            self._corner_mg = None

    # ── the beam region (one shape, two jobs) ────────────────────────────────

    def ensure_region_defaults(self) -> None:
        """Fill in radii the first time the region is switched on.

        They cannot be declared in ``DEFAULTS`` because a sensible radius is a
        fraction of the DETECTOR, whose size is not known until a dataset is
        open.
        """
        p = self.params
        if str(p.get("beam_shape", "off")) == "off" or float(p.get("beam_r") or 0) > 0:
            return
        d = _dpc.default_beam_region(self._sig_shape(), str(p["beam_shape"]))
        p["beam_cx"], p["beam_cy"] = d.cx, d.cy
        p["beam_r"], p["beam_r_inner"] = d.r, d.r_inner

    def show_beam_region(self) -> None:
        """Draw (or reshape) the draggable circle / ring on the pattern.

        The SAME widget answers both of the Center step's questions: its area is
        the centre-of-mass mask, and its centre is the Manual reference. Two
        separate controls for one physical thing (where is the beam?) is what
        this replaces.

        Switching shape rebuilds the widget — a circle and an annulus are
        different anyplotlib widget types, so there is nothing to mutate.
        """
        plot2d = getattr(self.src_plot, "_plot2d", None)
        shape = str(self.params.get("beam_shape", "off"))
        if plot2d is None or shape == "off":
            self.hide_beam_region()
            return
        self.ensure_region_defaults()
        r = self.region()
        if self._beam_widget is not None:
            if getattr(self._beam_widget, "_dpc_shape", None) == shape:
                try:
                    kw = ({"cx": r.cx, "cy": r.cy, "r_outer": r.r,
                           "r_inner": r.r_inner} if shape == "ring"
                          else {"cx": r.cx, "cy": r.cy, "r": r.r})
                    self._beam_widget.set(**kw)
                    return
                except Exception as e:                       # pragma: no cover
                    log.debug("resizing the DPC beam region failed: %s", e)
            self.hide_beam_region()
        try:
            if shape == "ring":
                w = plot2d.add_annular_widget(cx=r.cx, cy=r.cy, r_outer=r.r,
                                              r_inner=r.r_inner,
                                              color=_BEAM_COLOR)
            else:
                w = plot2d.add_circle_widget(cx=r.cx, cy=r.cy, r=r.r,
                                             color=_BEAM_COLOR)
            w._dpc_shape = shape
            from spyde.drawing.selectors.base_selector import event_handler_fn
            handler = event_handler_fn(lambda event: self._on_region_drag(event))
            w.add_event_handler(handler, "pointer_move", "pointer_up")
            self._beam_widget, self._beam_handler = w, handler
        except Exception as e:                               # pragma: no cover
            log.debug("adding the DPC beam region failed: %s", e)

    def hide_beam_region(self) -> None:
        """Widgets have no ``remove()``, only ``hide()`` (same as CZB's)."""
        if self._beam_widget is not None:
            try:
                self._beam_widget.hide()
            except Exception as e:                           # pragma: no cover
                log.debug("hiding the DPC beam region failed: %s", e)
            self._beam_widget = self._beam_handler = None

    def _on_region_drag(self, event=None) -> None:
        """Read the widget back, echo its geometry, and on RELEASE re-measure.

        Everything expensive waits for ``pointer_up``. A drag frame only reads
        the widget and echoes numbers the caret already has in hand.

        The brightness readout is what makes this necessary. It reads a frame,
        which on a lazy signal is a dask compute, and one per pointer frame
        queues work faster than it drains — the caret's own radius then keeps
        climbing for a while after the pointer stops, because the messages
        behind it are still landing. Same reason the Fit caret sends its state
        on release only (see ``background_action._on_window_drag``).

        RE-ENTRANCY GUARD: anyplotlib ``Widget.set()`` fires ``pointer_move``
        unconditionally — even on a no-change write — so anything here that
        writes back to the widget re-invokes this handler synchronously. The
        same recursion Crop and CZB both hit; compare-before-set is NOT enough.
        """
        w = self._beam_widget
        if w is None or self._beam_dragging or self._closed:
            return
        self._beam_dragging = True
        try:
            self.params["beam_cx"] = float(w.cx)
            self.params["beam_cy"] = float(w.cy)
            if str(self.params.get("beam_shape")) == "ring":
                self.params["beam_r"] = float(w.r_outer)
                self.params["beam_r_inner"] = float(w.r_inner)
            else:
                self.params["beam_r"] = float(w.r)
        except Exception as e:                               # pragma: no cover
            log.debug("reading the DPC beam region failed: %s", e)
        finally:
            self._beam_dragging = False
        released = str(getattr(event, "event_type", "") or "") == "pointer_up"
        self.emit_region(with_brightness=released)
        if released:
            self.arm_region_settle()

    def arm_region_settle(self) -> None:
        """(Re)start the debounce that re-measures once the drag stops.

        The region changes the centre of mass, so it can only take effect by
        re-measuring the whole scan — the one expensive step. Doing that per
        drag frame would make the widget unusable on any real dataset, so the
        widget and the brightness readout track the pointer and the measurement
        follows once motion stops.
        """
        import threading
        if self._settle_timer is not None:
            try:
                self._settle_timer.cancel()
            except Exception as e:                           # pragma: no cover
                log.debug("cancelling the DPC settle timer failed: %s", e)
        if self._closed:
            return

        def _fire():
            self._settle_timer = None
            if self._closed:
                return
            dispatch = getattr(self.session, "_dispatch_to_main", None)
            if dispatch is not None:
                dispatch(self.measure)
            else:
                self.measure()

        self._settle_timer = threading.Timer(_REGION_SETTLE_S, _fire)
        self._settle_timer.daemon = True
        self._settle_timer.start()

    def emit_region(self, with_brightness: bool = True) -> None:
        """Live region geometry + how bright it is, for the caret's readout.

        ``with_brightness=False`` re-sends the last measured value instead of
        reading a frame for a new one. Mid-drag the geometry is what the caret
        needs to track the pointer; the brightness costs a frame read and is
        recomputed on release. Re-sending the last value rather than ``None``
        keeps the readout from blanking on every drag.
        """
        region = self.region()
        signal = self.signal
        if with_brightness:
            brightness = (_dpc.region_brightness(signal, region)
                          if signal is not None else float("inf"))
            self._last_brightness = (None if not np.isfinite(brightness)
                                     else float(brightness))
        emit({"type": "dpc_region", "window_id": self.caret_window_id,
              "result_window_id": self.window_id,
              **region.as_dict(),
              "brightness": self._last_brightness})

    def sync_overlays(self) -> None:
        """Show exactly the furniture the current state needs.

        The corner boxes belong to one Center MODE; the beam region does not —
        it defines the centre of mass for every mode, so it is shown whenever
        it is switched on.
        """
        mode = str(self.params["center_mode"])
        if mode == "corners":
            self.show_corner_boxes()
        else:
            self.hide_corner_boxes()
        self.show_beam_region()

    # ── rotation ─────────────────────────────────────────────────────────────

    def solve_rotation(self) -> None:
        """Fit the scan↔detector rotation + handedness from the field itself."""
        if self.shifts is None:
            emit_error("DPC: no beam shifts to fit a rotation to yet.")
            return
        centered = _dpc.apply_reference(self.shifts, self.reference())
        est = _dpc.estimate_rotation(centered, mode=str(self.params["mode"]),
                                     nav_scale=_dpc._nav_scale(self.signal))
        self.estimate = est
        self.params["rotation"] = est.angle
        self.params["flip"] = est.flip
        emit({"type": "dpc_estimate", "window_id": self.caret_window_id,
              "result_window_id": self.window_id, **est.as_dict()})
        target = "curl" if est.mode == "electric" else "divergence"
        emit_status(f"DPC: rotation {est.angle:.1f}°"
                    f"{' (flipped)' if est.flip else ''} — "
                    f"{target} down {est.improvement:.1f}×")
        self.refresh()

    # ── vacuum reference ─────────────────────────────────────────────────────

    def load_vacuum(self, *, path: str | None = None,
                    tree_index: int | None = None) -> None:
        """Measure the beam shifts of a second (vacuum) dataset, off-thread."""
        signal, label = self._resolve_vacuum(path, tree_index)
        if signal is None:
            emit_error("DPC: pick a vacuum dataset first.")
            return
        method = str(self.params["method"])
        hw = int(self.params["half_square_width"] or 0)
        emit_status(f"DPC: measuring the vacuum reference ({label})…")

        def _work():
            return _dpc.measure_beam_shifts(signal, method=method,
                                            half_square_width=hw)

        def _apply(vac):
            if self._closed:
                return
            self.vacuum_shifts = vac
            self.vacuum_label = label
            self.params["center_mode"] = "vacuum"
            if self.shifts is not None and vac.shape[:2] != self.shifts.shape[:2]:
                # dpc.vacuum_reference assumes the same field of view at a
                # different sampling. It cannot check that, so say so.
                emit_status(
                    f"DPC: vacuum scan is {vac.shape[1]}×{vac.shape[0]}, "
                    f"sample is {self.shifts.shape[1]}×{self.shifts.shape[0]} — "
                    f"assuming the same field of view and rescaling the descan "
                    f"plane to fit.")
            else:
                emit_status(f"DPC: vacuum reference from {label}.")
            self.emit_state()
            self.sync_overlays()
            self.refresh()

        self.run_on_worker(_work, name="dpc-vacuum", on_done=_apply,
                           on_error=lambda e: emit_error(
                               f"DPC: reading the vacuum reference failed: {e}"))

    def _resolve_vacuum(self, path, tree_index):
        """A vacuum reference from a file on disk or an already-open dataset."""
        if tree_index is not None:
            trees = list(getattr(self.session, "signal_trees", []) or [])
            i = int(tree_index)
            if 0 <= i < len(trees) and trees[i] is not self.tree:
                return trees[i].root, _tree_title(trees[i])
            return None, ""
        if path:
            try:
                import hyperspy.api as hs
                sig = hs.load(str(path), lazy=True)
            except Exception as e:
                emit_error(f"DPC: could not open {path}: {e}")
                return None, ""
            import os
            return sig, os.path.basename(str(path))
        return None, ""

    # ── messages to the caret ────────────────────────────────────────────────

    @property
    def caret_window_id(self):
        """The window every ``dpc_*`` message must be addressed to.

        **The SOURCE window, not the result window.** ``useWizardEvent`` drops
        any message whose ``window_id`` is not the one the caret is mounted on,
        and the caret lives on the diffraction pattern — so addressing these to
        the result window (which has its own, different id) made every one of
        them silently vanish: the descan readout never arrived and Solve looked
        like it had hung. The result window's id rides along separately as
        ``result_window_id``.
        """
        return getattr(self.src_plot, "window_id", None)

    def emit_state(self) -> None:
        """Everything the caret needs to render itself honestly."""
        signal = self.signal
        auto_scale = _dpc.mrad_per_pixel(signal) if signal is not None else None
        energy = _dpc.beam_energy_kv(signal) if signal is not None else None
        emit({
            "type": "dpc_state",
            "window_id": self.caret_window_id,
            "result_window_id": self.window_id,
            "measured": self.shifts is not None,
            "nav_shape": list(self._nav_shape()) if self.shifts is not None else None,
            "centering": self.report.as_dict() if self.report else None,
            "mrad_per_px": float(auto_scale) if auto_scale else None,
            "beam_energy_kv": float(energy) if energy else None,
            "vacuum": self.vacuum_label or None,
            "datasets": self._dataset_choices(),
            "params": {k: v for k, v in self.params.items()
                       if not isinstance(v, np.ndarray)},
        })

    def emit_result(self) -> None:
        if self.result is None:
            return
        r = self.result
        div, curl = _dpc.field_symmetry(r.field, _dpc._nav_scale(self.signal))
        mag = r.magnitude
        finite = mag[np.isfinite(mag)]
        emit({
            "type": "dpc_result", "window_id": self.caret_window_id,
            "result_window_id": self.window_id,
            "units": r.units, "mode": r.mode, "rotation": r.rotation,
            "flip": r.flip, "reverse": r.reverse,
            "calibrated": bool(r.params.get("calibrated")),
            "max": float(finite.max()) if finite.size else 0.0,
            "mean": float(finite.mean()) if finite.size else 0.0,
            "divergence": float(div), "curl": float(curl),
        })

    def _dataset_choices(self) -> list[dict]:
        """Open datasets that could actually SERVE as the vacuum reference.

        Only 4D scans qualify — 2-D navigation over a 2-D detector — because
        anything else has no per-scan-point beam position to measure. The list
        used to be every open tree, which offered the user this action's own
        committed result maps as "vacuum scans": picking one produced a failed
        measure and an error, for a choice that was never valid.

        The scan shape is appended to the label because these are usually near-
        duplicates of each other (a sample scan and its vacuum scan, both named
        for the same session), and two identical rows are not a choice.
        """
        out = []
        for i, t in enumerate(getattr(self.session, "signal_trees", []) or []):
            if t is self.tree:
                continue
            root = getattr(t, "root", None)
            try:
                am = root.axes_manager
                if am.navigation_dimension != 2 or am.signal_dimension != 2:
                    continue
                ny, nx = tuple(int(n) for n in am.navigation_shape)[::-1]
            except Exception:                                # pragma: no cover
                continue
            out.append({"index": i, "title": f"{_tree_title(t)} ({nx}×{ny})"})
        return out

    # ── commit ───────────────────────────────────────────────────────────────

    def commit(self):
        """Freeze the current field as a NEW SignalTree.

        The RGB direction map is the primary (it is the picture people mean by
        "the DPC map"); every scalar component rides along as a chip-selectable
        view AND a real child node, so a saved tree carries Ex, Ey, magnitude,
        phase, divergence and curl rather than a picture of them.
        """
        if self.result is None or self.session is None:
            emit_error("DPC: nothing to commit yet.")
            return None
        from spyde.actions.commit import commit_result_tree
        r = self.result
        titles = _dpc.component_titles(r.mode, r.units)
        sym = "E" if r.mode == "electric" else "B"

        def _attach_wheel(tree):
            # The committed tree is what gets saved and shown to someone else,
            # so it is the copy that most needs to say what its hues mean.
            _display.attach_wheel_key_to_tree(tree, r)

        return commit_result_tree(
            self.session, title=f"DPC ({sym})",
            # The primary is the RGB direction+magnitude image, so label it that
            # way — calling it "Ex" put a chip next to the real "Ex (MV/cm)"
            # view claiming to be the same map.
            primary=r.rgb, primary_label=f"{sym} direction",
            views=[(titles[c], r.component(c)) for c in _dpc.COMPONENTS],
            levels=None, cmap="coolwarm",
            attrs={"dpc_result": r},
            provenance={
                "action": "DPC",
                "params": {**{k: v for k, v in r.params.items()},
                           "mode": r.mode, "rotation": r.rotation,
                           "flip": r.flip, "reverse": r.reverse,
                           "units": r.units,
                           "vacuum_reference": self.vacuum_label or None},
                "source_title": _tree_title(self.tree),
            },
            on_tree=_attach_wheel,
        )

    # ── teardown ─────────────────────────────────────────────────────────────

    def remove(self) -> None:
        """Tear down everything the wizard added. Idempotent — re-entry through
        remove → _forget_window → close → remove is a no-op."""
        if self._closed:
            return
        self._closed = True
        # Stop the in-flight pass. Closing the caret is the clearest case of
        # "nobody is waiting for this any more", and a beam-shift pass reads the
        # whole scan — it must not keep running for a wizard that is gone.
        self._cancel_measure()
        # Cancel the drag debounce FIRST: a timer that fires after teardown
        # would re-measure a torn-down wizard on a worker thread.
        if self._settle_timer is not None:
            try:
                self._settle_timer.cancel()
            except Exception as e:                           # pragma: no cover
                log.debug("cancelling the DPC settle timer failed: %s", e)
            self._settle_timer = None
        self.hide_corner_boxes()
        self.hide_beam_region()
        if self.window_id is not None:
            forget = getattr(self.session, "_forget_window", None)
            if forget is not None:
                try:
                    forget(int(self.window_id))
                except Exception as e:                       # pragma: no cover
                    log.debug("forgetting the DPC window failed: %s", e)
            else:                                            # pragma: no cover
                emit({"type": "window_closed", "window_id": int(self.window_id)})
                reg = getattr(self.session, "_window_controllers", None)
                if isinstance(reg, dict):
                    reg.pop(int(self.window_id), None)
        self.window_id = self.plot = self.wheel = None
        if getattr(self.tree, "_dpc_wizard", None) is self:
            self.tree._dpc_wizard = None


def _tree_title(tree) -> str:
    try:
        return str(tree.root.metadata.General.title) or "untitled"
    except Exception:                                        # pragma: no cover
        return "untitled"


# ── toolbar entry (ActionContext convention: fn(ctx, ...)) ────────────────────

def dpc(ctx, action_name: str = "DPC", **params) -> None:
    """Parent toolbar action — a no-op; the Electron toolbar opens the staged
    DPC wizard, which drives the ``dpc_*`` handlers."""
    return None


# ── staged handlers (fn(session, plot, payload)) ──────────────────────────────

def _ctrl_for(session, plot, payload) -> DpcWizard | None:
    """Resolve the live wizard for an action message.

    The result window is a bare ``figure``, so a ``window_id`` on it does not
    resolve to a ``Plot`` — look in the controller registry first, then fall
    back to the source tree's back-reference (the caret sends the SOURCE
    window's id, which does resolve to a Plot).
    """
    wid = (payload or {}).get("window_id")
    if wid is not None:
        lookup = getattr(session, "controller_by_window_id", None)
        ctrl = lookup(int(wid)) if lookup is not None else None
        if isinstance(ctrl, DpcWizard):
            return ctrl
    tree = getattr(plot, "signal_tree", None)
    ctrl = getattr(tree, "_dpc_wizard", None) if tree is not None else None
    if isinstance(ctrl, DpcWizard):
        return ctrl
    for cand in getattr(session, "signal_trees", []) or []:
        ctrl = getattr(cand, "_dpc_wizard", None)
        if isinstance(ctrl, DpcWizard) and not ctrl._closed:
            return ctrl
    return None


def dpc_open(session, plot, payload) -> None:
    """Caret mounted: cache the beam shifts and open the result window."""
    src, tree = _src_plot_tree(session, plot)
    signal = _current_signal(src)
    if src is None or tree is None or signal is None:
        emit_error("DPC: no active dataset")
        return
    if signal.axes_manager.navigation_dimension != 2:
        emit_error("DPC needs a 2-D scan (a 4D-STEM dataset): this signal has "
                   f"{signal.axes_manager.navigation_dimension} navigation "
                   f"dimension(s).")
        return

    # Idempotent: re-opening must not build a second wizard. React StrictMode
    # fires open→close→open synchronously, before the first measure lands — the
    # generation guard inside DpcWizard.measure() catches that race, this catches
    # a genuine re-open of a still-live wizard.
    existing = getattr(tree, "_dpc_wizard", None)
    if isinstance(existing, DpcWizard) and not existing._closed:
        existing.params.update(_clean(payload))
        existing.sync_overlays()
        existing.emit_state()
        return

    ctrl = DpcWizard(session, tree, src, params=_clean(payload))
    tree._dpc_wizard = ctrl
    ctrl.sync_overlays()
    ctrl.measure()


def dpc_close(session, plot, payload=None) -> None:
    """Caret unmounted: remove the windows and the overlays."""
    # Bump the generation FIRST and unconditionally, so a measure still in
    # flight (whose wizard isn't on the tree yet) is invalidated on arrival —
    # the StrictMode open/close/open race, exactly as in strain_close.
    tree = getattr(plot, "signal_tree", None)
    if tree is None:
        for cand in getattr(session, "signal_trees", []) or []:
            if getattr(cand, "_dpc_wizard", None) is not None:
                tree = cand
                break
    if tree is not None:
        from spyde.actions.lifecycle import bump_generation
        bump_generation(tree, "_dpc_run_gen")
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is not None:
        ctrl.remove()


def dpc_set_center(session, plot, payload) -> None:
    """Center tab: switch reference mode / resize the corner boxes."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    ctrl.params.update(_clean(payload))
    ctrl.sync_overlays()
    # Re-send the state, not just the map. The list of datasets that could serve
    # as a vacuum reference is part of it, and the user may well have OPENED
    # that vacuum scan since the caret mounted — a list captured once at open
    # shows them an empty picker and no way to refresh it.
    ctrl.emit_state()
    ctrl.refresh()


def dpc_set_beam(session, plot, payload) -> None:
    """Beam region: switch between off / circle / ring, or set its geometry.

    Changing the region changes the CENTRE OF MASS, so it can only take effect
    by re-measuring — which this does immediately for a discrete change (a
    shape toggle, a typed radius). A DRAG goes through the debounce instead
    (``_on_region_drag``), because the whole scan cannot be re-measured per
    pointer frame.
    """
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    before = ctrl.region().as_dict()
    ctrl.params.update(_clean(payload))
    ctrl.ensure_region_defaults()
    ctrl.sync_overlays()
    ctrl.emit_region()
    if ctrl.region().as_dict() != before:
        ctrl.measure()


def dpc_pick_center(session, plot, payload) -> None:
    """Manual mode: adopt the beam region's centre as the undeflected position.

    The region has already been dragged onto the beam, so it IS the answer —
    this just promotes it to the Manual reference (and accepts an explicit
    ``cx``/``cy`` for scripted callers)."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    cx, cy = payload.get("cx"), payload.get("cy")
    if cx is None or cy is None:
        picked = ctrl.region().center if ctrl.region().active else None
        if picked is None:
            emit_error("DPC: turn on the beam region (circle or ring) and drag "
                       "it onto the direct beam first.")
            return
        cx, cy = picked
    ctrl.params.update({"center_mode": "manual", "cx": float(cx), "cy": float(cy)})
    emit_status(f"DPC: beam centre set to ({cx:.1f}, {cy:.1f}) px.")
    ctrl.emit_state()
    ctrl.refresh()


def dpc_load_vacuum(session, plot, payload) -> None:
    """Vacuum tab: measure a second dataset as the descan reference."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    ctrl.load_vacuum(path=payload.get("path"),
                     tree_index=payload.get("tree_index"))


def dpc_auto_rotation(session, plot, payload) -> None:
    """Solve the scan↔detector rotation (and handedness) from the data."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    ctrl.params.update(_clean(payload))
    ctrl.solve_rotation()


def dpc_tune(session, plot, payload) -> None:
    """Any live parameter changed → re-derive and repaint (no re-measure)."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    before = str(ctrl.params.get("mode"))
    ctrl.params.update(_clean(payload))
    if str(ctrl.params.get("mode")) != before:
        # Electric and magnetic have different units AND different window
        # titles; the estimator's target symmetry changes too, so a stale
        # estimate would describe the wrong physics.
        ctrl.estimate = None
    ctrl.refresh()


def dpc_set_view(session, plot, payload) -> None:
    """Swap the displayed map. The colour wheel folds away for a scalar view."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    view = str(payload.get("view", DEFAULTS["view"]))
    if view not in _display.VIEWS:
        return
    ctrl.params["view"] = view
    ctrl.clim = None                    # each view gets its own fresh scale
    if ctrl.result is not None:
        _display.update_dpc_view(ctrl.plot, ctrl.wheel, ctrl.result, view,
                                 cmap=ctrl.cmap)
        ctrl._emit_histogram()


def dpc_run(session, plot, payload) -> None:
    """Re-measure the beam positions (a different finder or search window)."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    ctrl.params.update(_clean(payload))
    ctrl.measure()


def dpc_commit(session, plot, payload) -> None:
    """Freeze the current field as a new SignalTree."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        emit_error("DPC: no live field to commit.")
        return
    ctrl.commit()


def _clean(payload: dict | None) -> dict:
    """Keep only recognised parameters out of a caret payload.

    ``window_id`` and friends ride along on every staged message; letting them
    into ``params`` would put transport plumbing into the committed provenance.
    """
    allowed = set(DEFAULTS) | {"cx", "cy"}
    return {k: v for k, v in (payload or {}).items() if k in allowed}

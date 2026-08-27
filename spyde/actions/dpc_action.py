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
as ONE cancellable ``client.compute`` future, and the map repaints as
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
import os
import time

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
    # The beam region (see BeamRegion): "circle" | "ring". There is no "off" —
    # the region is what the centre of mass is taken over, so a pattern without
    # one is just a region the size of the whole frame, drawn nowhere the user
    # can grab it. The radii are filled in from the detector size once a dataset
    # is open, because a default in pixels cannot know the frame size.
    "beam_shape": "circle",
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
        self._beam_selector = None                  # the circle/ring on the DP
        self._lane = None                           # the serial pass-setup lane
        self._measure_stop: list | None = None      # in-flight pass's cancel token
        self._measure_future = None                 # …and its future, if any
        self._measured_region = None                # the region a pass last ran
        self._measure_event = None                  # …and what the stream waits on
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
        return _dpc.BeamRegion(shape=str(p.get("beam_shape", "circle")),
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

        Only the BOOKKEEPING happens on the caller's thread: cancel the pass
        this one replaces, take the parameters it will run with, and adopt its
        cancel token so a supersede arriving a moment later still stops it.
        Everything after that — copying the signal, building the graph, handing
        the chunks to the backend — runs on :meth:`_pass_lane`. Measured, that
        part costs 13-33 ms (hyperspy's ``map`` / ``get_direct_beam_position``
        graph build dominates), and a dragged region runs it several times a
        second: inline, it stuttered the very gesture it exists to serve.
        """
        if self.signal is None:
            emit_error("DPC: no active dataset")
            return
        # A superseded pass is CANCELLED, not left running to have its result
        # discarded — a beam-shift pass reads the whole scan, so one nobody is
        # waiting for costs the cluster the entire dataset. Same contract as
        # virtual_image (cancel prior → unregister → register new).
        self._cancel_measure()
        method = str(self.params["method"])
        hw = int(self.params["half_square_width"] or 0)
        region = self.region()
        sig_shape = self._sig_shape()
        gen = self.guard()
        # Adopted BEFORE the lane picks the pass up, so a supersede that lands
        # while it is still queued stops it before it costs anything: the lane
        # checks the token first and drops the job. During a drag that is most
        # of them.
        stop: list = [False]
        self._track_measure(stop)
        self._measured_region = region.as_dict()
        emit_status("DPC: locating the direct beam…")

        def _finish(shifts):
            self._retire_measure(stop)
            if stop[0] or not self.still(gen) or self._closed:
                return
            self.shifts = np.asarray(shifts, dtype=np.float64)
            self.report = _dpc.centering_report(self.shifts)
            self._warn_if_region_missed(self.shifts, region, sig_shape)
            self._set_computing(False)
            emit_progress(1, 1, "DPC")
            self.emit_state()
            self.refresh()
            # Now, not per pointer frame: this reads a frame off the dataset.
            self.emit_region(with_brightness=True)
            if on_done is not None:
                on_done()

        try:
            self._pass_lane().submit(self._begin_pass, stop, gen, method, hw,
                                     region, _finish)
        except RuntimeError as e:       # lane shut down under a closing wizard
            log.debug("DPC pass lane refused the job: %s", e)
            self._retire_measure(stop)

    def _pass_lane(self):
        """The ONE thread every beam-shift pass is set up on.

        Serial for the reason :func:`dpc.private_view` gives: hyperspy parks a
        length-1 placeholder on ``data`` for the width of the copy, so two
        threads setting a pass up on the same signal read each other's
        placeholder. One lane means the copy, and the graph build after it, can
        never overlap another pass's — however fast the region is dragged.

        A superseded job drops itself at the top of :meth:`_begin_pass`, so what
        queues behind a slow set-up is no-ops, not stacked passes.
        """
        if self._lane is None:
            self._lane = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="dpc-pass")
        return self._lane

    def _begin_pass(self, stop, gen, method, hw, region, on_finish) -> None:
        """Set the pass up and hand it to the cluster — on the pass lane.

        ONE handle, kept on ``_measure_future`` so the next move can cancel it,
        and the chunks paint into the display array as they land. Both come from
        ``compute_with_live_buffer``, which is what the virtual-image stream
        uses: a superseded whole-scan reduction has to be CANCELLED (its
        ``cancel`` stops outstanding work through the dispatcher's ``on_start``
        hook, not on a 0.5 s poll), and a scan big enough to be worth cancelling
        is big enough that the user should watch it fill in.

        Data already in RAM has no graph and so no handle, and runs to the end
        uninterrupted — the token can only drop its result. That is fine now
        and was not before: the centre of mass is one vectorised contraction
        per block rather than a Python call per frame, which measured 1429 ms →
        38 ms on a 64x64 scan of 64x64 frames. A pass that finishes in the time
        of two pointer frames does not need stopping.
        """
        if stop[0] or self._closed:
            return
        live = self.signal
        if live is None:
            return
        try:
            # Its OWN signal object: the graph runs hyperspy `map` on it, and a
            # new pass can still overlap the TAIL of the one it cancels.
            signal = _dpc.private_view(live)
            graph = _dpc.beam_shift_graph(signal, method=method,
                                          half_square_width=hw, region=region)
        except Exception as e:
            self._retire_measure(stop)
            self._measure_failed(gen, e)
            self._set_computing(False)
            return
        if stop[0] or self._closed:
            return

        client = getattr(self.tree, "client", None)
        backend = getattr(self.session, "compute_backend", None)
        if graph is None or (client is None and backend is None):
            # Nothing to chunk and so nothing to cancel: the data is already in
            # RAM, or there is neither a cluster nor a compute backend to hand
            # the chunks to. The token can only drop the result.
            try:
                shifts = _dpc.measure_beam_shifts(signal, method=method,
                                                  half_square_width=hw,
                                                  region=region)
            except Exception as e:
                self._retire_measure(stop)
                self._measure_failed(gen, e)
                self._set_computing(False)
                return
            self._dispatch(lambda: on_finish(shifts))
            return

        ny, nx = int(graph.shape[0]), int(graph.shape[1])
        field = self._display_field(signal, (ny, nx), method, hw, region, gen)
        if stop[0] or self._closed:
            return
        self.shifts = field
        self._set_computing(True)
        total = max(1, len(graph.chunks[0]) * len(graph.chunks[1]))
        landed = [0]
        # The cluster stream waits on an Event; the backend and the tree's
        # cancel registry work in `[False]` tokens. Both are set by the same
        # _cancel_measure, so there is one decision and two ways of hearing it.
        import threading
        event = threading.Event()

        def _on_chunk(chunk, slices):
            """A dask callback thread — store, then marshal the repaint."""
            if stop[0] or self._closed or not self.still(gen):
                return
            try:
                field[slices] = np.asarray(chunk, dtype=np.float64)
            except Exception as e:                           # pragma: no cover
                log.debug("storing a DPC chunk failed: %s", e)
                return
            landed[0] += 1
            n = landed[0]
            if n >= total:
                # The last chunk IS the completed pass. Counting them is how the
                # finish is known: the handle's value is assembled client-side
                # from these same chunks, so waiting on it as well would only
                # add a hop.
                self._dispatch(lambda: on_finish(field))
                return
            self._dispatch(lambda: self._on_partial(gen, n, total))

        try:
            if client is not None:
                # The cluster path, and the virtual-image stream's own call:
                # ONE handle, per-chunk callbacks, and a `cancel` that stops
                # outstanding work through the dispatcher's `on_start` hook
                # rather than on its next 0.5 s poll.
                from spyde.drawing.update_functions import (
                    compute_with_live_buffer)
                handle = compute_with_live_buffer(
                    graph, (ny, nx), client, "", on_chunk_done=_on_chunk,
                    windowed=True, stop_event=event)
            else:
                # No cluster (it takes ~10 s to come up, and a scan can be open
                # sooner; tests run with SPYDE_NO_DASK). The backend still
                # chunks and still stops — the threaded mode checks the token
                # before each submit AND inside each chunk task. Routing this
                # through a plain blocking compute instead is what made a
                # superseded pass run to the end.
                handle = backend.compute_chunks_progressive(
                    graph, 2, _on_chunk, stopped_flag=stop)
        except Exception as e:
            self._retire_measure(stop)
            self._measure_failed(gen, e)
            self._set_computing(False)
            return
        if self._measure_stop is stop:
            self._measure_event = event
        self._attach_future(stop, handle)

    def _display_field(self, signal, nav_shape, method, hw, region, gen):
        """The ``(ny, nx, 2)`` array the pass paints into.

        BLANK, deliberately, and this is the visible half of cancellation: the
        map goes dark the instant a new pass starts and fills back in as its
        chunks land, so stopping one pass and starting another is something you
        can SEE rather than something you have to trust. It is what the virtual
        image does, and it is the reason a superseded compute there is never in
        doubt. Carrying the previous field over instead looks smoother and
        hides exactly the thing worth showing.

        The FIRST pass after the caret opens is the exception: there is nothing
        to supersede and nothing on screen, so it seeds from the CORNERS. Those
        are a few percent of the scan, so they measure in a fraction of the time
        the full pass takes, and the plane through them is the descan ramp — a
        real map immediately, rather than an empty window with a spinner.
        """
        ny, nx = int(nav_shape[0]), int(nav_shape[1])
        field = np.full((ny, nx, 2), np.nan, dtype=np.float64)
        if self.shifts is not None:
            return field                 # a supersede: go dark and refill
        try:
            corners = self._measure_corners(signal, (ny, nx), method, hw, region)
        except Exception as e:
            log.debug("the DPC corner seed failed: %s", e)
            return field
        if corners is not None and self.still(gen):
            field[:] = corners
        return field

    def _measure_corners(self, signal, nav_shape, method, hw, region):
        """A whole-field plane fitted through the four scan corners only.

        The corners carry the instrument descan and (by assumption) none of the
        sample's field, which is what makes them both cheap to measure and worth
        showing on their own.
        """
        fraction = float(self.params["corner_fraction"])
        sparse = np.full((int(nav_shape[0]), int(nav_shape[1]), 2), np.nan,
                         dtype=np.float64)
        for rows, cols in _dpc.corner_slices(nav_shape, fraction):
            block = _dpc.measure_beam_shifts(signal.inav[cols, rows],
                                             method=method,
                                             half_square_width=hw, region=region)
            sparse[rows, cols] = block
        return _dpc.corner_reference(sparse, fraction)

    def _dispatch(self, fn) -> None:
        """Run *fn* on the event loop — figures and IPC belong there."""
        dispatch = getattr(self.session, "_dispatch_to_main", None)
        dispatch(fn) if dispatch is not None else fn()

    def _on_partial(self, gen: int, done: int, total: int) -> None:
        """Repaint from what has landed so far (event loop)."""
        if self._closed or not self.still(gen):
            return
        emit_progress(done, total, "DPC: locating the direct beam")
        self.refresh()


    def _track_measure(self, stop: list, future=None) -> None:
        """Adopt *stop*/*future* as the in-flight pass and register them on the
        tree, so closing the tree mid-pass stops it (``register_cancel``)."""
        self._measure_stop, self._measure_future = stop, future
        tree = self.tree
        reg = getattr(tree, "register_cancel", None)
        if reg is not None:
            reg(flag=stop, future=future)

    def _attach_future(self, stop: list, future) -> None:
        """Add the backend future to an ALREADY-adopted token.

        The token is adopted on the caller's thread; the future only exists once
        the pass has been set up on the lane, by which time a newer pass may own
        the slot. Attaching to a token that is no longer current would make the
        next ``_cancel_measure`` cancel the WRONG pass — so if this one has been
        superseded, cancel its future here instead.
        """
        if self._measure_stop is not stop:
            try:
                if not future.done():
                    future.cancel()
            except Exception as e:                           # pragma: no cover
                log.debug("cancelling a superseded DPC future failed: %s", e)
            return
        self._measure_future = future
        reg = getattr(self.tree, "register_cancel", None)
        if reg is not None:
            reg(flag=stop, future=future)

    def _retire_measure(self, stop: list) -> None:
        """Drop a FINISHED pass's token. Without this the tree's cancel registry
        gains an entry per measure — and every drag settle is a measure."""
        if self._measure_stop is not stop:
            return                      # already superseded; not ours to drop
        future = self._measure_future
        self._measure_stop = self._measure_future = self._measure_event = None
        unreg = getattr(self.tree, "unregister_cancel", None)
        if unreg is not None:
            try:
                unreg(flag=stop, future=future)
            except Exception as e:                           # pragma: no cover
                log.debug("retiring the DPC measure token failed: %s", e)

    def _cancel_measure(self) -> None:
        """Stop the in-flight beam-shift pass, if any.

        **Cancelling the future is what stops it.** The pass is ONE
        ``client.compute`` graph, so this is the same cancellation
        ``virtual_image`` uses and the scheduler drops the tasks. The token
        beside it covers only the window BEFORE that future exists — the graph
        is built on the pass lane, and a move can land while it is — and stops a
        late result being painted. Unregister both afterwards, or the tree's
        cancel list grows by one entry per drag.
        """
        stop, future = self._measure_stop, self._measure_future
        event = self._measure_event
        self._measure_stop = self._measure_future = self._measure_event = None
        if stop is None and future is None:
            return
        if stop is not None:
            stop[0] = True
        # The stream waits on the Event, so setting it is what stops the
        # dispatcher NOW rather than on its next 0.5 s poll.
        if event is not None:
            event.set()
        # At INFO because "is it actually cancelling, or just restarting after
        # it finishes?" is the one question this path raises and the one the
        # code cannot answer by inspection.
        log.info("DPC: cancelled the in-flight beam-shift pass")

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
        # With the traceback: the message alone names a symptom, and the pass
        # runs on a worker, so the frames are gone by the time anyone looks.
        log.exception("DPC beam-shift pass failed", exc_info=exc)
        emit_error(f"DPC: locating the direct beam failed: {exc}")

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
        """Fill in the radii the first time a dataset is open.

        They cannot be declared in ``DEFAULTS`` because a sensible radius is a
        fraction of the DETECTOR, whose size is not known until then — which is
        also why ``BeamRegion.active`` still exists with the "off" shape gone.
        """
        p = self.params
        if float(p.get("beam_r") or 0) > 0:
            return
        d = _dpc.default_beam_region(self._sig_shape(), str(p["beam_shape"]))
        p["beam_cx"], p["beam_cy"] = d.cx, d.cy
        p["beam_r"], p["beam_r_inner"] = d.r, d.r_inner

    def sync_beam_region(self) -> None:
        """Put the selector for the current shape on the pattern.

        The beam region is a :class:`~spyde.drawing.selectors.CircleSelector` /
        :class:`~spyde.drawing.selectors.AnnularSelector` — the same selectors a
        virtual image puts on the same plot — so it inherits the whole live
        path: every pointer frame submits to the one serial navigator
        dispatcher, a newer position replaces a still-queued older one, and a
        trailing settle re-fires once motion stops. It used to be a raw widget
        with a hand-rolled debounce, which is the only reason the map did not
        track the region the way a virtual image tracks its detector ROI.

        Switching shape REBUILDS it — a circle and an annulus are different
        anyplotlib widget types, so there is nothing to mutate.
        """
        from spyde.drawing.selectors import AnnularSelector, CircleSelector

        if self._closed or getattr(self.src_plot, "_plot2d", None) is None:
            return
        self.ensure_region_defaults()
        region = self.region()
        shape = region.shape
        selector = self._beam_selector
        if selector is not None and getattr(selector, "_dpc_shape", "") != shape:
            self.hide_beam_region()
            selector = None
        if selector is None:
            cls = AnnularSelector if shape == "ring" else CircleSelector
            try:
                # No children: this selector drives a whole-scan re-measure, not
                # a sliced child plot, so the work hangs off `index_hooks` — the
                # same seam the vector overlays use. `_run_update`'s child loop
                # is then a no-op and its geometry de-duplication still applies.
                selector = cls(parent=self.src_plot, children=[],
                               update_function=[], color=_BEAM_COLOR)
            except Exception as e:                           # pragma: no cover
                log.debug("building the DPC beam selector failed: %s", e)
                return
            selector._dpc_shape = shape
            selector.index_hooks.append(self._on_region_moved)
            self._beam_selector = selector
        self._write_region_to_widget(region)

    def _write_region_to_widget(self, region) -> None:
        """Push *region* onto the widget (a typed radius, a shape rebuild).

        Safe to call from anywhere: unlike the old raw widget this does NOT
        re-enter synchronously. ``set`` fires ``pointer_move``, but the selector
        answers that by submitting to the dispatcher, so the read-back lands on
        another thread and one write cannot recurse into another.
        """
        widget = getattr(self._beam_selector, "roi", None)
        if widget is None:
            return
        try:
            widget.set(**({"cx": region.cx, "cy": region.cy,
                           "r_outer": region.r, "r_inner": region.r_inner}
                          if region.shape == "ring"
                          else {"cx": region.cx, "cy": region.cy, "r": region.r}))
        except Exception as e:                               # pragma: no cover
            log.debug("writing the DPC beam region to its widget failed: %s", e)

    def hide_beam_region(self) -> None:
        """Take the selector off the pattern (shape change, or teardown)."""
        selector = self._beam_selector
        self._beam_selector = None
        if selector is None:
            return
        try:
            selector.close()
        except Exception as e:                               # pragma: no cover
            log.debug("closing the DPC beam selector failed: %s", e)

    def _on_region_moved(self, indices=None) -> None:
        """The region moved — track it and re-measure. On the dispatcher thread.

        This is the virtual-image update function's job, in the virtual-image
        update function's place: the selector fires it once per COMMITTED
        position (repeats at the same geometry are de-duplicated, and a
        superseded position was dropped from the pending slot before it ever
        ran), and it cancels the compute it replaces and starts a new one.
        There is no pacing on top of that — the dispatcher's latest-wins
        coalescing IS the pacing, exactly as it is for a virtual image.
        """
        if self._closed:
            return
        widget = getattr(self._beam_selector, "roi", None)
        if widget is None:
            return
        try:
            self.params["beam_cx"] = float(widget.cx)
            self.params["beam_cy"] = float(widget.cy)
            if str(self.params.get("beam_shape")) == "ring":
                self.params["beam_r"] = float(widget.r_outer)
                self.params["beam_r_inner"] = float(widget.r_inner)
            else:
                self.params["beam_r"] = float(widget.r)
        except Exception as e:                               # pragma: no cover
            log.debug("reading the DPC beam region failed: %s", e)
            return
        # Geometry only: the brightness readout reads a FRAME, and one dask
        # compute per pointer frame queues work faster than it drains however
        # far off-thread it runs. It is refreshed when the pass lands.
        self.emit_region(with_brightness=False)
        if self.region().as_dict() == self._measured_region:
            # The region the running (or last) pass already used. Reached on
            # every OPEN, because placing the selector writes its geometry and
            # the widget reports that write as a move — which superseded the
            # opening pass with an identical one, throwing away a whole scan's
            # work and the progressive fill with it. Also covers a drag that
            # returns to where it started.
            return
        self.measure()

    def emit_region(self, with_brightness: bool = True) -> None:
        """Live region geometry + how bright it is, for the caret's readout.

        The geometry goes out NOW, carrying whatever brightness was last
        measured — the caret has to track the pointer, and re-sending the last
        value rather than ``None`` keeps the readout from blanking on every
        drag. ``with_brightness`` additionally asks for a fresh reading, which
        costs a frame read (a dask compute on a lazy signal), so it happens on a
        worker and lands in a second message. Doing it inline stalled the event
        loop on exactly the gesture that must not stall.
        """
        region = self.region()
        self._send_region(region)
        signal = self.signal
        if not with_brightness or signal is None:
            return

        def _work():
            return _dpc.region_brightness(signal, region)

        def _done(brightness):
            # Drop a reading for a region the user has already moved on from —
            # a stale multiplier next to a region it does not describe reads as
            # a wrong answer, not a late one.
            if self._closed or self.region().as_dict() != region.as_dict():
                return
            self._last_brightness = (None if not np.isfinite(brightness)
                                     else float(brightness))
            self._send_region(region)

        self.run_on_worker(_work, name="dpc-brightness", on_done=_done,
                           on_error=lambda e: log.debug(
                               "the DPC brightness probe failed: %s", e))

    def _send_region(self, region) -> None:
        """One ``dpc_region`` message for *region* + the last known brightness."""
        emit({"type": "dpc_region", "window_id": self.caret_window_id,
              "result_window_id": self.window_id,
              **region.as_dict(),
              "brightness": self._last_brightness})

    def sync_overlays(self) -> None:
        """Show exactly the furniture the current state needs.

        The corner boxes belong to one Center MODE; the beam region does not —
        it defines the centre of mass for every mode, so it is always there.
        """
        mode = str(self.params["center_mode"])
        if mode == "corners":
            self.show_corner_boxes()
        else:
            self.hide_corner_boxes()
        self.sync_beam_region()

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
            # way â€” calling it "Ex" put a chip next to the real "Ex (MV/cm)"
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

    # â”€â”€ teardown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        # The selector owns the only remaining drag timer (its settle re-fire)
        # and cancels it in `close`; `_closed` is already True, so a hook that
        # fires in the meantime bails out on its own.
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
        # Last, and without waiting: a job still on the lane has already been
        # stopped by the cancel above and returns on its next token check, but
        # `remove` runs on the event loop and must not block on it.
        lane, self._lane = self._lane, None
        if lane is not None:
            lane.shutdown(wait=False)


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
    """Beam region: switch circle/ring, or type a radius.

    This only writes the geometry onto the widget. The re-measure follows from
    the widget the same way it does for a drag — ``sync_beam_region`` pushes the
    value, the selector reports the move, and ``_on_region_moved`` decides. One
    path for "the region changed", whether a pointer or a number moved it;
    measuring here as well would run the pass twice for every typed radius.
    """
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    ctrl.params.update(_clean(payload))
    ctrl.ensure_region_defaults()
    ctrl.sync_overlays()
    ctrl.emit_region(with_brightness=False)


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

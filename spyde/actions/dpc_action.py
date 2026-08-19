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
    dpc_pick_center    Manual tab: adopt the crosshair as the beam centre
    dpc_load_vacuum    Vacuum tab: measure a second (vacuum) dataset
    dpc_auto_rotation  solve the scan↔detector rotation from the data
    dpc_tune           any live parameter → re-derive and repaint (cheap)
    dpc_set_view       swap the displayed map (RGB / Ex / Ey / |E| / div / curl)
    dpc_run            re-measure with a different method / search window
    dpc_commit         freeze the field as a new SignalTree

**Measure once, tune forever.** The only expensive step is
``get_direct_beam_position``, a full pass over the dataset. It runs at open (and
again only if the *method* or search window changes) and the ``(ny, nx, 2)``
result is cached. Centering, rotation, handedness and calibration are then all
pure arithmetic on that small array, which is what lets the rotation slider be
genuinely live instead of a click-and-wait. Do not move the measure into
``dpc_tune``.

**The three ways to find zero.** A DPC map is a map of DIFFERENCES from the
undeflected beam position, so it is only as good as the zero it is measured
against — and the instrument's own descan drifts across a scan, which looks
exactly like a slowly varying field. The Center tab offers, in increasing order
of trustworthiness:

* **Manual** — one crosshair, one centre for every pattern. Removes a constant
  offset, nothing else. Fine when the descan is already good.
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

import logging

import numpy as np

from spyde.actions import dpc as _dpc
from spyde.actions import dpc_display as _display
from spyde.actions.context import current_signal as _current_signal
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.wizard import WizardController
from spyde.backend.ipc import emit, emit_error, emit_status

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
_CORNER_COLOR = "#f9e2af"      # the four corner boxes on the navigator
_CROSS_COLOR = "#94e2d5"       # the manual beam-centre crosshair on the DP

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
        "view": {
            "name": "Map", "type": "enum", "default": DEFAULTS["view"],
            "choices": list(_display.VIEWS), "tab": "Rotation",
        },
        "autolim_sigma": {
            "name": "Colour limit (σ)", "type": "float",
            "default": DEFAULTS["autolim_sigma"], "min": 0.5, "max": 10.0,
            "step": 0.5, "tab": "Rotation",
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
        self._cross = None                          # manual centre crosshair

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

    def measure(self, *, on_done=None) -> None:
        """Measure the direct-beam position over the whole scan, off-thread."""
        signal = self.signal
        if signal is None:
            emit_error("DPC: no active dataset")
            return
        method = str(self.params["method"])
        hw = int(self.params["half_square_width"] or 0)
        gen = self.guard()
        emit_status("DPC: locating the direct beam…")

        def _work():
            return _dpc.measure_beam_shifts(signal, method=method,
                                            half_square_width=hw)

        def _apply(shifts):
            if not self.still(gen) or self._closed:
                return
            self.shifts = shifts
            self.report = _dpc.centering_report(shifts)
            self.emit_state()
            self.refresh()
            if on_done is not None:
                on_done()

        self.run_on_worker(_work, name="dpc-measure", on_done=_apply,
                           on_error=lambda e: emit_error(f"DPC: locating the "
                                                         f"direct beam failed: {e}"))

    # ── stage 2: derive (pure arithmetic on the cached shifts) ───────────────

    def reference(self) -> np.ndarray | None:
        """The descan reference for the current Center mode, or ``None``.

        ``strict=False``: a caret sitting on Manual before the crosshair has
        been dragged, or on Vacuum before a dataset has been chosen, is
        mid-interaction — "no reference yet" is a valid state that must render,
        not an error that blanks the window.
        """
        if self.shifts is None:
            return None
        return _dpc.resolve_reference(
            self.shifts, center_mode=str(self.params["center_mode"]),
            corner_fraction=float(self.params["corner_fraction"]),
            center_xy=(self.params.get("cx"), self.params.get("cy")),
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
                mrad_per_px=scale, autolim_sigma=float(p["autolim_sigma"]))
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
        from spyde.actions.figure_registry import keep_alive
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
        # add_rectangles takes CENTRES + sizes; corner_boxes gives (x, y, w, h).
        offsets = [[x + w / 2.0, y + h / 2.0] for (x, y, w, h) in boxes]
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
                linewidths=1.5, alpha=0.22)
        except Exception as e:                               # pragma: no cover
            log.debug("drawing the DPC corner boxes failed: %s", e)

    def hide_corner_boxes(self) -> None:
        if self._corner_mg is not None:
            try:
                self._corner_mg.remove()
            except Exception as e:                           # pragma: no cover
                log.debug("removing the DPC corner boxes failed: %s", e)
            self._corner_mg = None

    def show_crosshair(self) -> None:
        """Drop a draggable crosshair on the DP for the Manual centre."""
        plot2d = getattr(self.src_plot, "_plot2d", None)
        if plot2d is None or self._cross is not None:
            return
        sy, sx = self._sig_shape()
        cx = float(self.params.get("cx") or sx / 2.0)
        cy = float(self.params.get("cy") or sy / 2.0)
        try:
            self._cross = plot2d.add_crosshair_widget(cx=cx, cy=cy,
                                                      color=_CROSS_COLOR)
            emit_status("DPC: drag the crosshair onto the undeflected beam, "
                        "then Use this centre.")
        except Exception as e:                               # pragma: no cover
            log.debug("adding the DPC crosshair failed: %s", e)

    def hide_crosshair(self) -> None:
        """Widgets have no ``remove()``, only ``hide()`` (same as CZB's)."""
        if self._cross is not None:
            try:
                self._cross.hide()
            except Exception as e:                           # pragma: no cover
                log.debug("hiding the DPC crosshair failed: %s", e)
            self._cross = None

    def sync_overlays(self) -> None:
        """Show exactly the furniture the current Center mode needs."""
        mode = str(self.params["center_mode"])
        if mode == "corners":
            self.show_corner_boxes()
        else:
            self.hide_corner_boxes()
        if mode == "manual":
            self.show_crosshair()
        else:
            self.hide_crosshair()

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
        )

    # ── teardown ─────────────────────────────────────────────────────────────

    def remove(self) -> None:
        """Tear down everything the wizard added. Idempotent — re-entry through
        remove → _forget_window → close → remove is a no-op."""
        if self._closed:
            return
        self._closed = True
        self.hide_corner_boxes()
        self.hide_crosshair()
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


def dpc_pick_center(session, plot, payload) -> None:
    """Manual tab: adopt the crosshair's position as the beam centre."""
    ctrl = _ctrl_for(session, plot, payload)
    if ctrl is None:
        return
    cross = ctrl._cross
    cx = float(cross.cx) if cross is not None else payload.get("cx")
    cy = float(cross.cy) if cross is not None else payload.get("cy")
    if cx is None or cy is None:
        emit_error("DPC: place the crosshair on the undeflected beam first.")
        return
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

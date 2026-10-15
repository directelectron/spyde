"""
phase_viewer.py — the phase viewer: a phase's master pattern, unit cell and
detector pattern, and the microscope set-up they are simulated in.

Clicking a phase in the Composition dock pops this out beside the sidebar.
It is part of the phase's set-up, not a dataset, so it is never a window:
the figure is sent tagged ``host: "phase"`` and the renderer shows it in the
pop-out, with the set-up form (``phase_setup`` messages) under it.

The three panes share one crystal orientation:

* **master pattern** — the phase's band sphere in the crystal frame, with
  the detector's footprint on it. The footprint belongs to the microscope
  (a ``"world"`` patch), so it stays put while a drag turns the crystal;
* **unit cell** — the atoms and edges of the cell, turning with it;
* **detector** — what the screen records at that orientation, with the
  kinematic spots on it on a TEM.

The 3-D panes are drawn in the DETECTOR frame, looking straight at the
screen with its own "up", so the front of the sphere reads like the screen
does whatever the detector's angles are; the crystal's orientation is kept
in the sample frame and carried into the detector frame for display.

Dragging either 3-D pane turns the crystal in both and repaints the
detector. The master pattern is either the kinematic bands
(:func:`spyde.ebsd.bands.band_intensity`, the profile the EBSD wizard's
dictionary uses) or a full dynamical simulation from de_ebsd when it is
installed; both are sampled at the same directions, so switching changes
the intensities and never the geometry.

Which microscope is read off the signal, not chosen. The set-up's numbers
are read from, and written back to, the signal's metadata
(:mod:`spyde.ebsd.microscope`), so the indexing wizards see them too. On an
EBSD scan the pattern centre can instead be solved from the scan's own
patterns (:mod:`spyde.ebsd.calibrate`).
"""
from __future__ import annotations

import importlib.util
import logging

import numpy as np

from de_shell.ipc import emit, emit_error, emit_status
from spyde.actions.composition import _sample, phase_label
from spyde.actions.lifecycle import run_on_worker
from spyde.drawing.selectors.base_selector import event_handler_fn

log = logging.getLogger(__name__)

#: The band sphere's image, longitude × latitude, and the mesh it is wrapped
#: on. The image carries the detail, so the mesh can stay coarse.
SPHERE_TEXTURE = (512, 1024)
SPHERE_MESH = (49, 97)
#: The detector preview's longer side, in pixels. A drag repaints it every
#: frame, so it is sized for speed, not for the detector's full resolution.
PREVIEW_PIXELS = 160
#: Three square panes. A figure that is not a window is never resized, so the
#: pop-out hosts it at exactly this size.
FIGURE_SIZE = (744, 248)
#: The full simulation's Lambert half-width and shortest d-spacing (nm).
FULL_NPX = 150
FULL_DMIN = 0.1

#: Looks along the detector frame's +z, with its +y up and +x to the right.
CAMERA = dict(azimuth=0.0, elevation=90.0)

DETECTOR_COLOUR = "#f38ba8"
SPOT_COLOUR = "#f9e2af"


def full_simulation_available() -> bool:
    """Whether de_ebsd, which computes the dynamical master pattern, is installed."""
    return importlib.util.find_spec("de_ebsd") is not None


def preview_shape(signal, microscope: str) -> tuple[int, int]:
    """``(rows, cols)`` of the detector preview: in SEM the signal's frame
    shape scaled to :data:`PREVIEW_PIXELS` on its longer side, so the aspect
    (which the pattern centre is measured against) is right; in TEM square."""
    rows = cols = PREVIEW_PIXELS
    if microscope == "SEM":
        try:
            axes = signal.axes_manager.signal_axes
            if len(axes) == 2:
                rows, cols = int(axes[1].size), int(axes[0].size)
        except Exception as e:
            log.debug("reading the detector shape failed: %s", e)
    scale = PREVIEW_PIXELS / max(rows, cols)
    return max(8, round(rows * scale)), max(8, round(cols * scale))


def _grey_rgb(image: np.ndarray) -> np.ndarray:
    """A float image as an 8-bit grey RGB texture, stretched over its 1–99.5 %
    range so a few very bright zone axes do not wash the bands out."""
    low, high = np.percentile(image, [1.0, 99.5])
    grey = np.clip((image - low) / max(high - low, 1e-12), 0, 1)
    return np.repeat((grey * 255).astype(np.uint8)[..., None], 3, axis=-1)


class PhaseViewer:
    """One phase's viewer: its figure, its set-up, and the crystal orientation.

    Kept on the sample's tree, one per phase (:func:`_viewers`), so popping
    the phase out again re-shows it as it was left.
    """

    def __init__(self, session, tree, phase: dict, structure, setup):
        from spyde.ebsd.spots import reflections_from_phase

        self.session = session
        self.tree = tree
        self.phase = phase
        self.structure = structure              # orix Phase, from the CIF
        self.setup = setup
        self.master = "bands"                   # or "full"
        self.full_state = "none"                # computing | ready | error
        self.full_note = ""
        self.solve_state = "none"               # solving | done | error
        self.solve_note = ""
        self.show_spots = setup.microscope == "TEM"
        self.rotation = np.eye(3)               # crystal → sample
        self._textures: dict = {}               # ("bands", kV) | ("full", key) → image
        self._reflectors: dict = {}             # kV → Reflectors
        self._full_patterns: dict = {}          # (kV, tilt) → de_ebsd MasterPattern
        self._spot_reflections = reflections_from_phase(structure)
        self._generation = 0
        self._closed = False
        self._registered = None
        self._geometry()

    # ── compute ──────────────────────────────────────────────────────────────
    @property
    def voltage(self) -> float:
        return float(self.setup["beam_energy"])

    def reflectors(self):
        from spyde.ebsd.bands import reflectors_from_phase

        if self.voltage not in self._reflectors:
            self._reflectors[self.voltage] = reflectors_from_phase(
                self.structure, voltage_kv=self.voltage)
        return self._reflectors[self.voltage]

    def _full_key(self):
        return (self.voltage, float(self.setup.values.get("sample_tilt", 70.0)))

    def _texture_key(self):
        if self.master == "full":
            return ("full",) + self._full_key()
        return ("bands", self.voltage)

    def texture(self) -> np.ndarray:
        """The master-pattern image for the current mode and voltage.

        Slow (~1 s for the bands, longer for a full simulation), so callers
        on the main thread go through :meth:`_refresh_texture`.
        """
        from spyde.ebsd.bands import band_sphere, sphere_directions

        key = self._texture_key()
        if key not in self._textures:
            if key[0] == "full":
                import torch
                pattern = self._full_patterns[self._full_key()]
                directions = torch.as_tensor(sphere_directions(*SPHERE_TEXTURE),
                                             dtype=torch.float32)
                image = pattern.sample(directions).cpu().numpy()
            else:
                image = band_sphere(self.reflectors(), SPHERE_TEXTURE)
            self._textures[key] = image
        return self._textures[key]

    def _geometry(self) -> None:
        from spyde.ebsd.microscope import detector_rays, detector_to_sample

        self.shape = preview_shape(self.tree.root, self.setup.microscope)
        rays = detector_rays(self.setup, self.shape)
        self._to_sample = detector_to_sample(self.setup)
        self._rays = rays.reshape(-1, 3)
        self._corners = np.array([rays[0, 0], rays[0, -1], rays[-1, -1], rays[-1, 0]])
        # The screen's gnomonic extent, for placing spots in pixels.
        gnomonic = rays[..., :2] / rays[..., 2:]
        self._extent = (float(gnomonic[0, 0, 0]), float(gnomonic[0, -1, 0]),
                        float(gnomonic[-1, 0, 1]), float(gnomonic[0, 0, 1]))

    def displayed(self) -> np.ndarray:
        """The crystal's rotation in the detector frame the panes are drawn in."""
        return self._to_sample.T @ self.rotation

    def pattern(self) -> np.ndarray:
        """What the detector records at the current orientation.

        :meth:`displayed` carries crystal directions into the detector frame,
        so a ray ``r`` came from crystal direction ``displayedᵀ r`` — as rows,
        ``r @ displayed``.
        """
        from spyde.ebsd.bands import band_intensity

        crystal_rays = self._rays @ self.displayed()
        if self.master == "full" and self._full_key() in self._full_patterns:
            import torch
            image = self._full_patterns[self._full_key()].sample(
                torch.as_tensor(crystal_rays, dtype=torch.float32)).cpu().numpy()
        else:
            image = band_intensity(crystal_rays, self.reflectors())
        return np.asarray(image, np.float32).reshape(self.shape)

    def spots(self):
        """``(pixels, radii)`` of the kinematic spots, or empty arrays."""
        from spyde.ebsd.bands import wavelength
        from spyde.ebsd.spots import spot_pixels

        if not (self.show_spots and self.setup.microscope == "TEM"):
            return np.empty((0, 2)), np.empty(0)
        pixels, weights = spot_pixels(
            self._spot_reflections, self.rotation, self._to_sample,
            wavelength(self.voltage), self._extent, self.shape)
        return pixels, 1.0 + 3.0 * np.sqrt(weights)

    # ── figure ───────────────────────────────────────────────────────────────
    def build_figure(self, texture: np.ndarray) -> None:
        import anyplotlib as apl
        from spyde.ebsd.bands import sphere_directions
        from spyde.ebsd.unit_cell import ELEMENT_COLOURS, unit_cell

        self.figure, (sphere_axes, cell_axes, detector_axes) = apl.subplots(
            1, 3, figsize=FIGURE_SIZE)

        mesh = sphere_directions(*SPHERE_MESH)
        self.sphere = sphere_axes.plot_surface(
            mesh[..., 0], mesh[..., 1], mesh[..., 2], bounds=((-1, 1),) * 3,
            zoom=1.3, **CAMERA)
        self.sphere.set_axis_off()
        self.sphere.set_title("Master pattern")
        self.sphere.set_texture(_grey_rgb(texture), shade=True, cull_backfaces=True)
        self.sphere.set_drag_mode("rotate")
        self.sphere.set_rotation(self.displayed())
        self._footprint = self.sphere.add_patch(
            self._corners, frame="world", facecolor=DETECTOR_COLOUR, alpha=0.12,
            linewidth=2, name="detector")

        positions, elements, edges = unit_cell(self.structure)
        centre = (edges.reshape(-1, 3).max(0) + edges.reshape(-1, 3).min(0)) / 2
        reach = 1.05 * float(np.linalg.norm(edges.reshape(-1, 3) - centre, axis=1).max())
        self.cell = cell_axes.scatter3d(
            positions[:, 0], positions[:, 1], positions[:, 2],
            colors=[ELEMENT_COLOURS.get(e, "#9e9e9e") for e in elements],
            point_size=6, bounds=tuple((c - reach, c + reach) for c in centre),
            **CAMERA)
        self.cell.set_axis_off()
        self.cell.set_title("Unit cell")
        self.cell.set_drag_mode("rotate")
        self.cell.set_rotation(self.displayed())
        for number, edge in enumerate(edges):
            self.cell.add_patch(edge, closed=False, geodesic=False,
                                edgecolor="#a6adc8", alpha=0, linewidth=1,
                                name=f"edge{number}")

        self.detector = detector_axes.imshow(self.pattern(), cmap="gray")
        self.detector.set_axis_off()
        self._spot_markers = None
        self._paint_detector()

        # Held on self: the callback registry may keep only a weak reference.
        self._handlers = {
            "sphere": event_handler_fn(lambda event: self._on_turn(event, self.sphere)),
            "cell": event_handler_fn(lambda event: self._on_turn(event, self.cell)),
        }
        self.sphere.add_event_handler(self._handlers["sphere"], "pointer_move", "pointer_up")
        self.cell.add_event_handler(self._handlers["cell"], "pointer_move", "pointer_up")

    def _paint_detector(self) -> None:
        from spyde.ebsd.bands import matrix_to_euler

        self.detector.set_data(self.pattern())
        pixels, radii = self.spots()
        if self._spot_markers is None and len(pixels):
            self._spot_markers = self.detector.add_circles(
                pixels, name="spots", radius=radii, facecolors=SPOT_COLOUR,
                edgecolors=SPOT_COLOUR, alpha=0.8, linewidths=0.5,
                size_units="px")
        elif self._spot_markers is not None:
            self._spot_markers.set(offsets=pixels if len(pixels) else np.empty((0, 2)),
                                   radius=radii if len(radii) else 1.0)
        euler = np.degrees(matrix_to_euler(self.rotation.T))
        self.detector.set_title("φ1 {:.0f}°  Φ {:.0f}°  φ2 {:.0f}°".format(*euler))

    def _on_turn(self, event, source) -> None:
        if self._closed or getattr(event, "rotation", None) is None:
            return
        self.rotation = self._to_sample @ source.rotation
        other = self.cell if source is self.sphere else self.sphere
        try:
            other.set_rotation(source.rotation)
            self._paint_detector()
        except Exception as e:
            log.debug("repainting after a turn failed: %s", e)

    def _apply_geometry(self) -> None:
        """Redraw after a set-up change. The crystal stays put in the sample
        frame, so moving the detector turns it as the detector sees it."""
        self._geometry()
        self.sphere.set_rotation(self.displayed())
        self.cell.set_rotation(self.displayed())
        self._footprint.set(points=self._corners)
        self._paint_detector()

    def _refresh_texture(self) -> None:
        """Repaint the sphere for a new voltage or master pattern, computing
        the image off the main thread when it is not cached."""
        if self._texture_key() in self._textures:
            self.sphere.set_texture(_grey_rgb(self._textures[self._texture_key()]),
                                    shade=True, cull_backfaces=True)
            self._paint_detector()
            return
        self._generation += 1
        generation = self._generation

        def painted(texture):
            if self._closed or generation != self._generation:
                return
            self.sphere.set_texture(_grey_rgb(texture), shade=True, cull_backfaces=True)
            self._paint_detector()

        run_on_worker(self.session, self.texture, name="phase-viewer-texture",
                      on_done=painted,
                      on_error=lambda e: emit_error(f"Could not redraw the master pattern: {e}"))

    # ── set-up changes ───────────────────────────────────────────────────────
    def set_field(self, key: str, value: float) -> None:
        from spyde.ebsd.microscope import write_field

        write_field(self.tree.root, self.setup.microscope, key, value)
        old_key = self._texture_key()
        self.setup.values[key] = float(value)
        self.setup.sources[key] = "metadata"
        self._apply_geometry()
        if self._texture_key() != old_key:
            if self.master == "full" and self._full_key() not in self._full_patterns:
                self.master, self.full_state = "bands", "none"
            self._refresh_texture()

    def set_master(self, master: str) -> None:
        if master == "bands":
            self.master = "bands"
            self._refresh_texture()
            return
        if not full_simulation_available() or self.setup.microscope != "SEM":
            return
        if self._full_key() in self._full_patterns:
            self.master, self.full_state = "full", "ready"
            self._refresh_texture()
            return
        self._compute_full()

    def set_spots(self, show: bool) -> None:
        self.show_spots = bool(show)
        self._paint_detector()

    def _simulate_full(self, key):
        """The dynamical master pattern at *key* (kV, sample tilt), on the CPU.
        Slow: run off the main thread."""
        import de_ebsd
        from spyde.device_lock import accelerator_lock

        device = _device()
        crystal = de_ebsd.Crystal.from_cif(self.phase["cif_path"])
        microscope = de_ebsd.Microscope(energy=key[0] * 1e3, sample_tilt=key[1])
        simulator = de_ebsd.MasterPatternSimulator(
            crystal, microscope, npx=FULL_NPX, dmin=FULL_DMIN, backend="bloch_prism")
        with accelerator_lock(device):
            return simulator.compute(device=device).to("cpu")

    def _compute_full(self) -> None:
        key = self._full_key()
        self.full_state, self.full_note = "computing", "Simulating…"
        self.emit_setup()

        def work():
            return self._simulate_full(key)

        def done(pattern):
            if self._closed:
                return
            self._full_patterns[key] = pattern
            self.full_state, self.full_note = "ready", ""
            if self._full_key() == key:
                self.master = "full"
                self._refresh_texture()
            self.emit_setup()

        def failed(error):
            self.full_state, self.full_note = "error", str(error)
            self.emit_setup()

        run_on_worker(self.session, work, name="phase-viewer-full",
                      on_done=done, on_error=failed)

    # ── pattern centre from the data ─────────────────────────────────────────
    def can_solve(self) -> bool:
        """An SEM scan with several patterns, and de_ebsd to fit them."""
        axes = self.tree.root.axes_manager
        return (self.setup.microscope == "SEM" and full_simulation_available()
                and axes.signal_dimension == 2 and axes.navigation_dimension >= 1)

    def solve_pattern_centre(self) -> None:
        """Fit the pattern centre to patterns read across the scan, and write
        it to the metadata when the fit is good enough to trust.

        The crystal is turned to the centre pattern's orientation, so the
        detector preview shows what that pattern should look like.
        """
        from spyde.ebsd.calibrate import (MIN_FIT, calibration_positions,
                                          read_patterns, solve_pattern_centre)

        if not self.can_solve() or self.solve_state == "solving":
            return
        key = self._full_key()
        setup = self.setup
        signal = self.tree.root
        self.solve_state, self.solve_note = "solving", "Solving from the scan's patterns…"
        self.emit_setup()

        def work():
            from spyde.device_lock import accelerator_lock

            master = self._full_patterns.get(key) or self._simulate_full(key)
            positions = calibration_positions(signal.axes_manager.navigation_shape)
            patterns = read_patterns(signal, positions)
            device = _device()
            with accelerator_lock(device):
                calibration = solve_pattern_centre(
                    patterns, master, setup, self.structure.point_group,
                    device=device, positions=positions)
            return master, calibration

        def done(result):
            master, calibration = result
            if self._closed:
                return
            self._full_patterns[key] = master
            summary = "{} patterns · fit {:.2f} (from {:.2f})".format(
                len(calibration.positions), calibration.fit, calibration.fit_before)
            if not calibration.trusted:
                self.solve_state = "error"
                self.solve_note = (f"{summary} — below {MIN_FIT}, so the centre "
                                   "was left as it was")
                self.emit_setup()
                return
            for axis, value in zip(("pc_x", "pc_y", "pc_z"), calibration.pc):
                self.set_field(axis, value)
            self.rotation = calibration.rotations[0]
            self._apply_geometry()
            self.solve_state, self.solve_note = "done", f"Solved from {summary}"
            self.emit_setup()

        def failed(error):
            self.solve_state, self.solve_note = "error", f"Could not solve: {error}"
            self.emit_setup()

        run_on_worker(self.session, work, name="phase-viewer-solve-pc",
                      on_done=done, on_error=failed)

    # ── messages ─────────────────────────────────────────────────────────────
    def show(self) -> None:
        """Send the figure and the set-up to the pop-out."""
        from spyde.actions.figure_window import register_figure

        if self._registered is None:
            self._registered = register_figure(self.figure)
        figure_id, html = self._registered
        emit({"type": "figure", "host": "phase", "phase_id": self.phase["id"],
              "fig_id": figure_id, "html": html, "title": phase_label(self.phase)})
        self.emit_setup()

    def emit_setup(self) -> None:
        lattice = self.structure.structure.lattice
        emit({
            "type": "phase_setup",
            "phase_id": self.phase["id"],
            "title": phase_label(self.phase),
            "subtitle": "{} · a {:.3f} Å".format(
                self.structure.space_group.short_name, float(lattice.a)),
            "microscope": self.setup.microscope,
            "fields": [{"key": f.key, "label": f.label, "unit": f.unit,
                        "value": self.setup.values[f.key], "step": f.step,
                        "source": self.setup.sources[f.key]}
                       for f in self.setup.fields],
            "master": self.master,
            "full": {"available": full_simulation_available(),
                     "applies": self.setup.microscope == "SEM",
                     "state": self.full_state, "note": self.full_note},
            "solve": {"available": self.can_solve(), "state": self.solve_state,
                      "note": self.solve_note},
            "spots": self.show_spots,
            "spots_apply": self.setup.microscope == "TEM",
        })

    def close(self) -> None:
        self._closed = True


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _pc_fallback(signal) -> dict:
    """The pattern centre the EBSD wizard would use when the metadata has
    none: the one stamped on synthetic data, else its centred default."""
    from spyde.actions.ebsd_action import _resolve_pc

    pc = _resolve_pc(signal, {})
    return {"pc_x": pc[0], "pc_y": pc[1], "pc_z": pc[2]}


def _viewers(tree) -> dict:
    """The tree's phase viewers by phase id — state that lives and dies with
    the sample (the README's ownership map), not with a window."""
    viewers = getattr(tree, "_phase_viewers", None)
    if viewers is None:
        viewers = {}
        tree._phase_viewers = viewers
    return viewers


def _viewer_for(session, plot, payload):
    tree, phases = _sample(session, plot)
    phase = next((p for p in phases if p["id"] == payload.get("phase")), None)
    if tree is None or phase is None:
        return None, None, None
    return tree, phase, _viewers(tree).get(phase["id"])


# ── staged handlers ──────────────────────────────────────────────────────────
def show_phase(session, plot, payload) -> None:
    """Pop out the viewer of phase ``payload['phase']``.

    A viewer already built for the phase's current structure is re-sent as
    it is. Otherwise reading the structure and computing the band sphere take
    about a second, so they run off the main thread.
    """
    from spyde.ebsd.microscope import read_setup

    tree, phase, existing = _viewer_for(session, plot, payload)
    if tree is None:
        emit_error("Phase viewer: that phase is no longer on this sample.")
        return
    if not phase.get("cif_path"):
        emit_error(f"{phase_label(phase)} has no structure yet — choose a CIF "
                   "or a COD entry for it first.")
        return
    if existing is not None and existing.phase.get("cif_path") == phase["cif_path"]:
        existing.show()
        return
    setup = read_setup(tree.root, fallback=_pc_fallback(tree.root))
    emit_status(f"Simulating {phase_label(phase)}…")

    def work():
        from orix.crystal_map import Phase

        viewer = PhaseViewer(session, tree, phase, Phase.from_cif(phase["cif_path"]),
                             setup)
        return viewer, viewer.texture()

    def opened(result):
        viewer, texture = result
        viewer.build_figure(texture)
        replaced = _viewers(tree).get(phase["id"])
        if replaced is not None:
            replaced.close()
        _viewers(tree)[phase["id"]] = viewer
        viewer.show()
        emit_status(f"{phase_label(phase)}: drag the sphere or the cell to turn the crystal.")

    def failed(error):
        emit_error(f"Could not simulate {phase_label(phase)}: {error}")

    run_on_worker(session, work, name="phase-viewer", on_done=opened, on_error=failed)


def phase_setup_set(session, plot, payload) -> None:
    """Change one thing about a phase viewer's set-up.

    *payload* carries ``phase`` and exactly one of ``field`` + ``value``,
    ``master`` (bands | full), ``spots`` (bool) or ``solve_pc`` (solve the
    pattern centre from the scan).
    """
    _tree, _phase, viewer = _viewer_for(session, plot, payload)
    if viewer is None:
        return
    try:
        if "field" in payload:
            viewer.set_field(str(payload["field"]), float(payload["value"]))
        elif "master" in payload:
            viewer.set_master(str(payload["master"]))
        elif "spots" in payload:
            viewer.set_spots(bool(payload["spots"]))
        elif payload.get("solve_pc"):
            viewer.solve_pattern_centre()
    except Exception as e:
        log.exception("phase set-up change failed")
        emit_error(f"Could not apply that change: {e}")
    viewer.emit_setup()

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
  kinematic spots on it in TEM mode.

Dragging either 3-D pane turns the crystal in both and repaints the
detector. The master pattern is either the kinematic bands
(:func:`spyde.ebsd.bands.band_intensity`, the profile the EBSD wizard's
dictionary uses) or a full dynamical simulation from de_ebsd when it is
installed; both are sampled at the same directions, so switching changes
the intensities and never the geometry.

The set-up's numbers are read from, and written back to, the signal's
metadata (:mod:`spyde.ebsd.microscope`), so the indexing wizards see them too.
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

DETECTOR_COLOUR = "#f38ba8"
SPOT_COLOUR = "#f9e2af"


def full_simulation_available() -> bool:
    """Whether de_ebsd, which computes the dynamical master pattern, is installed."""
    return importlib.util.find_spec("de_ebsd") is not None


def preview_shape(signal, mode: str) -> tuple[int, int]:
    """``(rows, cols)`` of the detector preview: in SEM the signal's frame
    shape scaled to :data:`PREVIEW_PIXELS` on its longer side, so the aspect
    (which the pattern centre is measured against) is right; in TEM square."""
    rows = cols = PREVIEW_PIXELS
    if mode == "SEM":
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
        self.show_spots = setup.mode == "TEM"
        self.rotation = np.eye(3)
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

        self.shape = preview_shape(self.tree.root, self.setup.mode)
        rays = detector_rays(self.setup, self.shape)
        self._to_sample = detector_to_sample(self.setup)
        self._rays_sample = rays.reshape(-1, 3) @ self._to_sample.T
        corners = np.array([rays[0, 0], rays[0, -1], rays[-1, -1], rays[-1, 0]])
        self._corners = corners @ self._to_sample.T
        # The screen's gnomonic extent, for placing spots in pixels.
        gnomonic = rays[..., :2] / rays[..., 2:]
        self._extent = (float(gnomonic[0, 0, 0]), float(gnomonic[0, -1, 0]),
                        float(gnomonic[-1, 0, 1]), float(gnomonic[0, 0, 1]))

    def pattern(self) -> np.ndarray:
        """What the detector records at the current orientation.

        :attr:`rotation` carries crystal directions into the sample frame,
        so a ray ``r`` came from crystal direction ``rotationᵀ r`` — as rows,
        ``r @ rotation``.
        """
        from spyde.ebsd.bands import band_intensity

        crystal_rays = self._rays_sample @ self.rotation
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

        if not (self.show_spots and self.setup.mode == "TEM"):
            return np.empty((0, 2)), np.empty(0)
        pixels, weights = spot_pixels(
            self._spot_reflections, self.rotation, self._to_sample,
            wavelength(self.voltage), self._extent, self.shape)
        return pixels, 1.0 + 3.0 * np.sqrt(weights)

    # ── figure ───────────────────────────────────────────────────────────────
    def build_figure(self, texture: np.ndarray) -> None:
        import anyplotlib as apl
        from spyde.ebsd.bands import sphere_directions
        from spyde.ebsd.microscope import camera_angles
        from spyde.ebsd.unit_cell import ELEMENT_COLOURS, unit_cell

        azimuth, elevation = camera_angles(self.setup)
        self.figure, (sphere_axes, cell_axes, detector_axes) = apl.subplots(
            1, 3, figsize=FIGURE_SIZE)

        mesh = sphere_directions(*SPHERE_MESH)
        self.sphere = sphere_axes.plot_surface(
            mesh[..., 0], mesh[..., 1], mesh[..., 2], bounds=((-1, 1),) * 3,
            azimuth=azimuth, elevation=elevation, zoom=1.3)
        self.sphere.set_axis_off()
        self.sphere.set_title("Master pattern")
        self.sphere.set_texture(_grey_rgb(texture), shade=True, cull_backfaces=True)
        self.sphere.set_drag_mode("rotate")
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
            azimuth=azimuth, elevation=elevation)
        self.cell.set_axis_off()
        self.cell.set_title("Unit cell")
        self.cell.set_drag_mode("rotate")
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
        self.rotation = source.rotation
        other = self.cell if source is self.sphere else self.sphere
        try:
            other.set_rotation(self.rotation)
            self._paint_detector()
        except Exception as e:
            log.debug("repainting after a turn failed: %s", e)

    def _apply_geometry(self) -> None:
        """Re-aim everything after a set-up change that moves the detector."""
        from spyde.ebsd.microscope import camera_angles

        self._geometry()
        azimuth, elevation = camera_angles(self.setup)
        self.sphere.set_view(azimuth=azimuth, elevation=elevation)
        self.cell.set_view(azimuth=azimuth, elevation=elevation)
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
    def set_mode(self, mode: str) -> None:
        from spyde.ebsd.microscope import read_setup

        self.setup = read_setup(self.tree.root, mode, fallback=_pc_fallback(self.tree.root))
        self.show_spots = mode == "TEM"
        if mode == "TEM" and self.master == "full":
            self.master = "bands"
        self._apply_geometry()
        self._refresh_texture()

    def set_field(self, key: str, value: float) -> None:
        from spyde.ebsd.microscope import write_field

        write_field(self.tree.root, self.setup.mode, key, value)
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
        if not full_simulation_available() or self.setup.mode != "SEM":
            return
        if self._full_key() in self._full_patterns:
            self.master, self.full_state = "full", "ready"
            self._refresh_texture()
            return
        self._compute_full()

    def set_spots(self, show: bool) -> None:
        self.show_spots = bool(show)
        self._paint_detector()

    def _compute_full(self) -> None:
        key = self._full_key()
        cif = self.phase["cif_path"]
        self.full_state, self.full_note = "computing", "Simulating…"
        self.emit_setup()

        def work():
            import de_ebsd
            import torch
            from spyde.device_lock import accelerator_lock

            device = "cuda" if torch.cuda.is_available() else "cpu"
            crystal = de_ebsd.Crystal.from_cif(cif)
            microscope = de_ebsd.Microscope(energy=key[0] * 1e3, sample_tilt=key[1])
            simulator = de_ebsd.MasterPatternSimulator(
                crystal, microscope, npx=FULL_NPX, dmin=FULL_DMIN,
                backend="bloch_prism")
            with accelerator_lock(device):
                return simulator.compute(device=device)

        def done(pattern):
            if self._closed:
                return
            self._full_patterns[key] = pattern.to("cpu")
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
            "mode": self.setup.mode,
            "fields": [{"key": f.key, "label": f.label, "unit": f.unit,
                        "value": self.setup.values[f.key], "step": f.step,
                        "source": self.setup.sources[f.key]}
                       for f in self.setup.fields],
            "master": self.master,
            "full": {"available": full_simulation_available(),
                     "applies": self.setup.mode == "SEM",
                     "state": self.full_state, "note": self.full_note},
            "spots": self.show_spots,
            "spots_apply": self.setup.mode == "TEM",
        })

    def close(self) -> None:
        self._closed = True


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
    setup = read_setup(tree.root, payload.get("mode"), fallback=_pc_fallback(tree.root))
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

    *payload* carries ``phase`` and exactly one of ``mode`` (SEM | TEM),
    ``field`` + ``value``, ``master`` (bands | full) or ``spots`` (bool).
    """
    _tree, _phase, viewer = _viewer_for(session, plot, payload)
    if viewer is None:
        return
    try:
        if "mode" in payload:
            viewer.set_mode(str(payload["mode"]))
        elif "field" in payload:
            viewer.set_field(str(payload["field"]), float(payload["value"]))
        elif "master" in payload:
            viewer.set_master(str(payload["master"]))
        elif "spots" in payload:
            viewer.set_spots(bool(payload["spots"]))
    except Exception as e:
        log.exception("phase set-up change failed")
        emit_error(f"Could not apply that change: {e}")
    viewer.emit_setup()

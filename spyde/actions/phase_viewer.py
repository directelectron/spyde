"""
phase_viewer.py — a phase's Kikuchi band sphere and the pattern it makes on
the detector.

Clicking a phase's structure in the Composition dock expands the phase to
show this figure in the dock itself — it is part of the phase's setup, not a
dataset, so it never becomes a window. The left panel is the phase's
kinematic master pattern on a sphere — every band the structure makes, in
the crystal frame — with the detector's footprint drawn on it. The
footprint belongs to the microscope, so it stays put while a drag turns the
crystal under it. The right panel is what the detector sees at that
orientation, repainted on every drag frame.

Both panels are :func:`spyde.ebsd.bands.band_intensity` sampled at different
directions — the same profile the EBSD wizard's dictionary and its live band
overlay use — so the sphere, the preview and the overlay cannot disagree.

The geometry comes from the open signal the way the EBSD wizard reads it: the
pattern centre from :func:`spyde.actions.ebsd_action._resolve_pc`, the
voltage from the signal's beam energy (20 kV when it has none), and the
detector's shape from the signal's frames.
"""
from __future__ import annotations

import logging

import numpy as np

from de_shell.ipc import emit_error, emit_status
from spyde.actions.composition import _sample, phase_label
from spyde.actions.lifecycle import run_on_worker
from spyde.drawing.selectors.base_selector import event_handler_fn

log = logging.getLogger(__name__)

#: The band sphere's image, longitude × latitude. The image carries the
#: detail; the mesh it is wrapped on can stay coarse.
SPHERE_TEXTURE = (512, 1024)
SPHERE_MESH = (49, 97)
#: The detector preview's longer side, in pixels. A drag repaints it every
#: frame, so it is sized for speed, not for the detector's full resolution.
PREVIEW_PIXELS = 160
DEFAULT_VOLTAGE_KV = 20.0
#: The figure fills the dock's width under the phase's row.
FIGURE_SIZE = (276, 150)

#: The camera that looks at the sphere from the detector. The detector sits
#: along +z of the sample frame with +y up on its screen (see
#: ``detector_directions``); azimuth 0, elevation 90 puts +z toward the viewer,
#: +x to the right and +y up, so the sphere reads like the detector does.
CAMERA = dict(azimuth=0.0, elevation=90.0)


def beam_voltage_kv(signal) -> float:
    """The accelerating voltage recorded on *signal*, in kV."""
    for key in ("Acquisition_instrument.SEM.beam_energy",
                "Acquisition_instrument.TEM.beam_energy"):
        try:
            value = signal.metadata.get_item(key, None)
        except Exception:
            value = None
        if value:
            return float(value)
    return DEFAULT_VOLTAGE_KV


def preview_shape(signal) -> tuple[int, int]:
    """``(rows, cols)`` of the detector preview: the signal's frame shape when
    it has 2-D frames, scaled so the longer side is :data:`PREVIEW_PIXELS`."""
    rows = cols = PREVIEW_PIXELS
    try:
        axes = signal.axes_manager.signal_axes
        if len(axes) == 2:
            rows, cols = int(axes[1].size), int(axes[0].size)
    except Exception as e:
        log.debug("reading the detector shape failed: %s", e)
    scale = PREVIEW_PIXELS / max(rows, cols)
    return max(8, round(rows * scale)), max(8, round(cols * scale))


class PhaseViewerController:
    """One phase's viewer: the figure, its two panels, and the geometry.

    Kept on the sample's tree, one per phase (:func:`_viewers`), so expanding
    the phase again re-shows it without recomputing the sphere.
    """

    def __init__(self, phase: dict, reflectors, texture: np.ndarray, *,
                 voltage_kv: float, pc, shape):
        from spyde.ebsd.bands import detector_directions

        self.phase = phase
        self.reflectors = reflectors
        self.voltage_kv = float(voltage_kv)
        self.pc = tuple(float(v) for v in pc)
        self.shape = tuple(int(v) for v in shape)
        directions = detector_directions(self.shape, self.pc)
        self._rays = directions.reshape(-1, 3)
        self._corners = np.array([directions[0, 0], directions[0, -1],
                                  directions[-1, -1], directions[-1, 0]])
        self._closed = False
        self._build_figure(texture)

    # ── figure ───────────────────────────────────────────────────────────────
    def _build_figure(self, texture: np.ndarray) -> None:
        import anyplotlib as apl
        from spyde.ebsd.bands import sphere_directions

        self.figure, (sphere_axes, detector_axes) = apl.subplots(
            1, 2, figsize=FIGURE_SIZE)
        mesh = sphere_directions(*SPHERE_MESH)
        self.sphere = sphere_axes.plot_surface(
            mesh[..., 0], mesh[..., 1], mesh[..., 2],
            bounds=((-1, 1),) * 3, zoom=1.3, **CAMERA)
        self.sphere.set_axis_off()
        self.sphere.set_texture(_grey_rgb(texture), shade=True,
                                cull_backfaces=True)
        self.sphere.set_drag_mode("rotate")
        self.sphere.add_patch(self._corners, frame="world",
                              facecolor="#f38ba8", alpha=0.12, linewidth=2,
                              name="detector")

        self.detector = detector_axes.imshow(self.pattern(np.eye(3)),
                                             cmap="gray")
        self._title(np.eye(3))
        # Held on self: the callback registry may keep only a weak reference.
        self._turn_handler = event_handler_fn(self._on_turn)
        self.sphere.add_event_handler(self._turn_handler,
                                      "pointer_move", "pointer_up")

    def pattern(self, rotation: np.ndarray) -> np.ndarray:
        """The detector pattern with the crystal turned by *rotation*.

        *rotation* is the sphere's model rotation, which carries crystal
        directions into the sample frame — the transpose of the Bunge matrix
        ``bands`` works with, so the plane normals in the sample frame are
        ``normals @ rotation.T``.
        """
        from spyde.ebsd.bands import band_intensity

        normals = self.reflectors.normals @ np.asarray(rotation).T
        return band_intensity(self._rays, self.reflectors,
                              normals=normals).reshape(self.shape)

    def _title(self, rotation: np.ndarray) -> None:
        from spyde.ebsd.bands import matrix_to_euler

        euler = np.degrees(matrix_to_euler(np.asarray(rotation).T))
        self.detector.set_title(
            "φ1 {:.0f}°  Φ {:.0f}°  φ2 {:.0f}°".format(*euler))

    def _on_turn(self, event) -> None:
        if self._closed or getattr(event, "rotation", None) is None:
            return
        rotation = self.sphere.rotation
        try:
            self.detector.set_data(self.pattern(rotation))
            self._title(rotation)
        except Exception as e:
            log.debug("repainting the detector preview failed: %s", e)

    def matches(self, phase: dict, voltage_kv: float, pc, shape) -> bool:
        """Whether this viewer still shows *phase* in this geometry."""
        return (self.phase.get("cif_path") == phase.get("cif_path")
                and self.voltage_kv == float(voltage_kv)
                and self.pc == tuple(float(v) for v in pc)
                and self.shape == tuple(int(v) for v in shape))

    def show(self) -> None:
        """Send the figure to the dock, addressed to the phase it belongs to.

        Registered once: a second show re-sends the same figure id, so the
        dock's iframe keeps its state.
        """
        from de_shell import ipc
        from spyde.actions.figure_window import register_figure

        if getattr(self, "_registered", None) is None:
            self._registered = register_figure(self.figure)
        figure_id, html = self._registered
        ipc.emit({"type": "figure", "host": "phase",
                  "phase_id": self.phase["id"], "fig_id": figure_id,
                  "html": html, "title": phase_label(self.phase)})

    def close(self) -> None:
        self._closed = True


def _viewers(tree) -> dict:
    """The tree's phase viewers by phase id — state that lives and dies with
    the sample (the README's ownership map), not with a window."""
    viewers = getattr(tree, "_phase_viewers", None)
    if viewers is None:
        viewers = {}
        tree._phase_viewers = viewers
    return viewers


def _grey_rgb(image: np.ndarray) -> np.ndarray:
    """A float image as an 8-bit grey RGB texture, stretched over its 1–99.5 %
    range so a few very bright zone axes do not wash the bands out."""
    low, high = np.percentile(image, [1.0, 99.5])
    grey = np.clip((image - low) / max(high - low, 1e-12), 0, 1)
    return np.repeat((grey * 255).astype(np.uint8)[..., None], 3, axis=-1)


# ── staged handler ───────────────────────────────────────────────────────────
def show_phase(session, plot, payload) -> None:
    """Show the viewer of phase ``payload['phase']`` in the dock.

    A viewer already built for this phase in this geometry is re-sent as it
    is. Otherwise reading the structure and computing the band sphere take
    about a second, so they run off the main thread and the figure is sent
    when they finish.
    """
    from spyde.actions.ebsd_action import _resolve_pc

    tree, phases = _sample(session, plot)
    phase = next((p for p in phases if p["id"] == payload.get("phase")), None)
    if tree is None or phase is None:
        emit_error("Phase viewer: that phase is no longer on this sample.")
        return
    if not phase.get("cif_path"):
        emit_error(f"{phase_label(phase)} has no structure yet — choose a CIF "
                   "or a COD entry for it first.")
        return
    signal = tree.root
    voltage = beam_voltage_kv(signal)
    pc = _resolve_pc(signal, {})
    shape = preview_shape(signal)
    viewers = _viewers(tree)
    existing = viewers.get(phase["id"])
    if existing is not None and existing.matches(phase, voltage, pc, shape):
        existing.show()
        return
    emit_status(f"Simulating the bands of {phase_label(phase)} at {voltage:g} kV…")

    def work():
        from orix.crystal_map import Phase
        from spyde.ebsd.bands import band_sphere, reflectors_from_phase

        reflectors = reflectors_from_phase(Phase.from_cif(phase["cif_path"]),
                                           voltage_kv=voltage)
        return reflectors, band_sphere(reflectors, SPHERE_TEXTURE)

    def opened(result):
        reflectors, texture = result
        viewer = PhaseViewerController(phase, reflectors, texture,
                                       voltage_kv=voltage, pc=pc, shape=shape)
        replaced = viewers.get(phase["id"])
        if replaced is not None:
            replaced.close()
        viewers[phase["id"]] = viewer
        viewer.show()
        emit_status(f"{phase_label(phase)}: drag the sphere to turn the crystal.")

    def failed(error):
        emit_error(f"Could not simulate {phase_label(phase)}: {error}")

    run_on_worker(session, work, name="phase-viewer", on_done=opened,
                  on_error=failed)

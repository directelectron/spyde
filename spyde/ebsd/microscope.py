"""microscope.py — the microscope set-up a phase is simulated in: which
microscope, its geometry fields, and the rays its detector sees.

A phase's band sphere is fixed to the crystal; what the detector records
depends on where the detector is and how the sample sits. Those are a
handful of numbers that already live (or should live) with the data, so
each field names the metadata key it reads and writes. Editing one in the
phase viewer writes it back, so the indexing wizards see the same set-up.

**Which microscope** is not a choice: :func:`microscope_of` reads it off the
signal — an EBSD scan or SEM metadata is SEM, anything else TEM.

**Frames.** The detector frame is the one
:func:`spyde.ebsd.bands.detector_directions` uses (x right, y up on the
screen, z from the sample towards the screen); orientations are given in the
SAMPLE frame. :func:`detector_to_sample` turns one into the other:

* **SEM** — kikuchipy's convention, the reference for EBSD geometry, with all
  four of its angles: sample tilt, detector tilt, azimuthal and twist
  (:func:`_sample_to_detector`, a port of kikuchipy's own, pinned against it
  in ``test_phase_viewer``). ``spyde.ebsd.bands`` uses the same detector
  frame (``test_ebsd_bands_kikuchipy_parity``).
* **TEM** — the screen is normal to the beam, so only the stage tilts turn
  the sample relative to it: α about x, then β about y.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MICROSCOPES = ("SEM", "TEM")


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    unit: str
    metadata_key: str
    default: float
    step: float = 1.0


_EBSD = "Acquisition_instrument.SEM.Detector.EBSD."

FIELDS: dict[str, tuple[Field, ...]] = {
    "SEM": (
        Field("beam_energy", "Beam energy", "kV",
              "Acquisition_instrument.SEM.beam_energy", 20.0, 0.5),
        Field("sample_tilt", "Sample tilt", "°",
              "Acquisition_instrument.SEM.Stage.tilt_alpha", 70.0, 0.5),
        Field("detector_tilt", "Detector tilt", "°", _EBSD + "tilt", 0.0, 0.5),
        Field("azimuthal", "Azimuthal", "°", _EBSD + "azimuthal", 0.0, 0.5),
        Field("twist", "Twist", "°", _EBSD + "twist", 0.0, 0.5),
        Field("pc_x", "PC x", "", _EBSD + "pc_x", 0.5, 0.005),
        Field("pc_y", "PC y", "", _EBSD + "pc_y", 0.5, 0.005),
        Field("pc_z", "PC z", "", _EBSD + "pc_z", 0.55, 0.005),
    ),
    "TEM": (
        Field("beam_energy", "Beam energy", "kV",
              "Acquisition_instrument.TEM.beam_energy", 200.0, 10.0),
        Field("camera_length", "Camera length", "mm",
              "Acquisition_instrument.TEM.camera_length", 300.0, 10.0),
        Field("detector_size", "Detector width", "mm",
              "Acquisition_instrument.TEM.Detector.Diffraction.size", 20.0, 1.0),
        Field("tilt_alpha", "Stage tilt α", "°",
              "Acquisition_instrument.TEM.Stage.tilt_alpha", 0.0, 0.5),
        Field("tilt_beta", "Stage tilt β", "°",
              "Acquisition_instrument.TEM.Stage.tilt_beta", 0.0, 0.5),
    ),
}


@dataclass
class Setup:
    """One microscope set-up: which microscope and every field's value.

    ``sources[key]`` says where a value came from: ``"metadata"``, ``"data"``
    (recorded with the scan — a kikuchipy detector, a synthetic stamp) or
    ``"default"`` when the signal had none, which the viewer flags because a
    default is a guess the person should check.
    """
    microscope: str
    values: dict
    sources: dict

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    @property
    def fields(self) -> tuple[Field, ...]:
        return FIELDS[self.microscope]


def microscope_of(signal) -> str:
    """SEM for an EBSD scan or a signal with SEM metadata, else TEM."""
    try:
        if "EBSD" in str(getattr(signal, "_signal_type", "") or "").upper():
            return "SEM"
        if signal.metadata.has_item("Acquisition_instrument.SEM"):
            return "SEM"
    except Exception:
        pass
    return "TEM"


def recorded_detector(signal) -> dict:
    """The SEM detector geometry recorded with the scan, if any: a kikuchipy
    EBSD signal carries its ``EBSDDetector``."""
    detector = getattr(signal, "detector", None)
    if detector is None:
        return {}
    try:
        pc = np.asarray(detector.pc_average, float).reshape(3)
        return {"sample_tilt": float(detector.sample_tilt),
                "detector_tilt": float(detector.tilt),
                "azimuthal": float(detector.azimuthal),
                "twist": float(detector.twist),
                "pc_x": pc[0], "pc_y": pc[1], "pc_z": pc[2]}
    except Exception:
        return {}


def read_setup(signal, *, fallback=None) -> Setup:
    """The set-up recorded on *signal*.

    Each field is, in order: the metadata value, the value recorded with the
    scan (:func:`recorded_detector`, then *fallback* — the pattern centre
    stamped on synthetic data, say), then the field's default.
    """
    microscope = microscope_of(signal)
    recorded = {**(fallback or {}), **recorded_detector(signal)}
    values, sources = {}, {}
    for field in FIELDS[microscope]:
        try:
            stored = signal.metadata.get_item(field.metadata_key, None)
        except Exception:
            stored = None
        if stored is not None:
            values[field.key], sources[field.key] = float(stored), "metadata"
        elif field.key in recorded:
            values[field.key], sources[field.key] = float(recorded[field.key]), "data"
        else:
            values[field.key], sources[field.key] = field.default, "default"
    return Setup(microscope, values, sources)


def write_field(signal, microscope: str, key: str, value: float) -> None:
    """Record one field in *signal*'s metadata."""
    field = next(f for f in FIELDS[microscope] if f.key == key)
    signal.metadata.set_item(field.metadata_key, float(value))


def _rotation_about(axis, degrees: float) -> np.ndarray:
    """Right-handed rotation by *degrees* about unit *axis* (Rodrigues)."""
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    a = np.radians(degrees)
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def _sample_to_detector(sample_tilt, detector_tilt, azimuthal, twist) -> np.ndarray:
    """kikuchipy's passive sample → detector rotation (``M @ v_sample``).

    Ported from kikuchipy (``detectors/_ebsd_detector.py``,
    ``_sample_to_detector_matrix``, GPL-3.0-or-later): the rows start as the
    detector axes written in sample coordinates, and each angle turns all of
    them about one of them — sample tilt (negated) and detector tilt about the
    detector x axis, then azimuthal about y and twist about z, both negated.
    """
    basis = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])
    for row, angle in zip((0, 0, 1, 2),
                          (-sample_tilt, detector_tilt, -azimuthal, -twist)):
        basis = basis @ _rotation_about(basis[row], angle).T
    return basis


def detector_to_sample(setup: Setup) -> np.ndarray:
    """The rotation taking detector-frame directions into the sample frame."""
    if setup.microscope == "SEM":
        return _sample_to_detector(setup["sample_tilt"], setup["detector_tilt"],
                                   setup["azimuthal"], setup["twist"]).T
    return _rotation_about([1, 0, 0], setup["tilt_alpha"]) @ \
        _rotation_about([0, 1, 0], setup["tilt_beta"])


def detector_rays(setup: Setup, shape) -> np.ndarray:
    """Unit directions from the sample to every detector pixel, in the
    DETECTOR frame, ``(rows, cols, 3)``."""
    from spyde.ebsd.bands import detector_directions

    rows, cols = int(shape[0]), int(shape[1])
    if setup.microscope == "SEM":
        return detector_directions((rows, cols),
                                   (setup["pc_x"], setup["pc_y"], setup["pc_z"]))
    # TEM: the screen is centred on the beam; its half-width subtends
    # atan(width / 2 / camera length).
    half = setup["detector_size"] / 2.0 / max(setup["camera_length"], 1e-6)
    x = np.linspace(-half, half, cols)
    y = np.linspace(half, -half, rows) * rows / cols
    gx, gy = np.meshgrid(x, y)
    rays = np.stack([gx, gy, np.ones_like(gx)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

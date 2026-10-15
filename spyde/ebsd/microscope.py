"""microscope.py — the microscope set-up a phase is simulated in: fields, metadata,
and the rays a detector sees.

A phase's band sphere is fixed to the crystal; what the detector records
depends on where the detector is and how the sample sits. Those are a
handful of numbers that already live (or should live) in the signal's
metadata, so each field here names the metadata key it reads and writes.
Editing one in the phase viewer writes it back, so the indexing wizards see
the same set-up.

**Frames.** The sphere is drawn in the SAMPLE frame: x along the tilt axis,
z along the sample normal. The detector frame is the one
:func:`spyde.ebsd.bands.detector_directions` uses (x right, y up on the
screen, z from the sample towards the screen). :func:`detector_to_sample`
turns one into the other:

* **SEM** — the sample is tilted towards the detector, so the angle between
  the sample normal and the screen normal is ``90° − sample tilt + detector
  tilt`` (20° for the usual 70° tilt), a rotation about x. The sample normal
  sits ABOVE the pattern centre, as it does on a real EBSD screen.
* **TEM** — the screen is normal to the beam, so only the stage tilts turn
  the sample relative to it: α about x, then β about y.

At zero relative tilt the two frames coincide, which is the frame the EBSD
wizard's dictionary is built in.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODES = ("SEM", "TEM")


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    unit: str
    metadata_key: str
    default: float
    step: float = 1.0


FIELDS: dict[str, tuple[Field, ...]] = {
    "SEM": (
        Field("beam_energy", "Beam energy", "kV",
              "Acquisition_instrument.SEM.beam_energy", 20.0, 0.5),
        Field("sample_tilt", "Sample tilt", "°",
              "Acquisition_instrument.SEM.Stage.tilt_alpha", 70.0, 0.5),
        Field("detector_tilt", "Detector tilt", "°",
              "Acquisition_instrument.SEM.Detector.EBSD.tilt", 0.0, 0.5),
        Field("pc_x", "PC x", "", "Acquisition_instrument.SEM.Detector.EBSD.pc_x",
              0.5, 0.005),
        Field("pc_y", "PC y", "", "Acquisition_instrument.SEM.Detector.EBSD.pc_y",
              0.5, 0.005),
        Field("pc_z", "PC z", "", "Acquisition_instrument.SEM.Detector.EBSD.pc_z",
              0.55, 0.005),
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
    """One microscope set-up: the mode and every field's value.

    ``sources[key]`` is ``"metadata"`` when the value was read from the
    signal and ``"default"`` when the signal had none — the viewer shows
    which, because a default is a guess the person should check.
    """
    mode: str
    values: dict
    sources: dict

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    @property
    def fields(self) -> tuple[Field, ...]:
        return FIELDS[self.mode]


def default_mode(signal) -> str:
    """SEM for an EBSD scan or a signal with SEM metadata, else TEM."""
    try:
        if "EBSD" in str(getattr(signal, "_signal_type", "") or "").upper():
            return "SEM"
        if signal.metadata.has_item("Acquisition_instrument.SEM"):
            return "SEM"
    except Exception:
        pass
    return "TEM"


def read_setup(signal, mode: str | None = None, *, fallback=None) -> Setup:
    """The set-up recorded on *signal* in *mode* (its default mode if None).

    *fallback* maps field keys to values to use before the field's own
    default when the metadata has none — the EBSD wizard's stamped pattern
    centre, for instance.
    """
    mode = mode or default_mode(signal)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    values, sources = {}, {}
    for field in FIELDS[mode]:
        try:
            stored = signal.metadata.get_item(field.metadata_key, None)
        except Exception:
            stored = None
        if stored is not None:
            values[field.key], sources[field.key] = float(stored), "metadata"
        elif fallback and field.key in fallback:
            values[field.key], sources[field.key] = float(fallback[field.key]), "data"
        else:
            values[field.key], sources[field.key] = field.default, "default"
    return Setup(mode, values, sources)


def write_field(signal, mode: str, key: str, value: float) -> None:
    """Record one field in *signal*'s metadata."""
    field = next(f for f in FIELDS[mode] if f.key == key)
    signal.metadata.set_item(field.metadata_key, float(value))


def _rotation_x(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _rotation_y(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def detector_to_sample(setup: Setup) -> np.ndarray:
    """The rotation taking detector-frame directions into the sample frame."""
    if setup.mode == "SEM":
        return _rotation_x(90.0 - setup["sample_tilt"] + setup["detector_tilt"])
    return _rotation_x(setup["tilt_alpha"]) @ _rotation_y(setup["tilt_beta"])


def detector_rays(setup: Setup, shape) -> np.ndarray:
    """Unit directions from the sample to every detector pixel, in the
    DETECTOR frame, ``(rows, cols, 3)``."""
    from spyde.ebsd.bands import detector_directions

    rows, cols = int(shape[0]), int(shape[1])
    if setup.mode == "SEM":
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


def camera_angles(setup: Setup) -> tuple[float, float]:
    """``(azimuth, elevation)`` in degrees that look at the sphere from the
    detector, so the sphere's front reads like the screen does."""
    v = detector_to_sample(setup) @ np.array([0.0, 0.0, 1.0])
    elevation = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(v[0], -v[1])))
    return azimuth, elevation

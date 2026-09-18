"""
reciprocal_units.py — what a diffraction axis measures, and how to re-express it.

A 4D-STEM detector axis can be calibrated four ways that all describe the same
geometry: raw pixels, a scattering angle in milliradians, or a reciprocal
spacing in nm⁻¹ or Å⁻¹. Files disagree — the ZrNb precipitate scan ships in
nm⁻¹, pyxem's own examples in Å⁻¹ — while every crystallographic library SpyDE
talks to (diffsims, orix, pyxem's matcher) works in **Å⁻¹** and none of them
checks. A template library built from an nm⁻¹ half-extent is a factor of ten
wrong and still returns an answer, so the mistake shows up as a bad orientation
map rather than as an error.

So Å⁻¹ is the internal unit, and this module is the only place that knows it:

* :func:`inverse_angstrom_scale` — what one detector pixel is worth in Å⁻¹,
  whatever the axis is labelled. Physics code asks this instead of reading
  ``ax.scale``, which is why a scan *displayed* in nm⁻¹ still indexes correctly.
* :func:`convert_signal_axes` — re-express the axes in another unit. It
  RESCALES; it never relabels. Relabelling is how a calibration becomes a lie.

Reciprocal-to-reciprocal (nm⁻¹ ↔ Å⁻¹) is a factor of ten and needs nothing
else. Only ``mrad`` involves the electron wavelength, and only ``px`` discards
information — which is why the Å⁻¹ calibration is stashed in metadata before a
conversion can lose it, so the toggle is reversible in every direction.

The unit spellings are pyxem's (``px`` / ``mrad`` / ``nm^-1`` / ``A^-1``) so the
same names mean the same thing in a notebook, and every LaTeX variant a file
might carry is accepted on input.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: The four ways a detector axis can be calibrated.
PIXELS = "px"
MILLIRADIAN = "mrad"
PER_NANOMETRE = "nm^-1"
PER_ANGSTROM = "A^-1"

#: Å⁻¹ is what diffsims, orix and pyxem's matcher all work in, so it is what
#: SpyDE stores and what every physics helper here returns.
CANONICAL = PER_ANGSTROM

#: Offered by the units toggle, in the order it shows them.
TOGGLE_ORDER = (PIXELS, MILLIRADIAN, PER_NANOMETRE, PER_ANGSTROM)

#: The reciprocal units, in Å⁻¹ per unit. ``mrad`` is absent because it needs
#: the wavelength, and ``px`` because it has no physical scale at all.
_INVERSE_ANGSTROM_PER = {PER_ANGSTROM: 1.0, PER_NANOMETRE: 0.1}

#: Every spelling of a unit that a file, a library or a person might write.
#: HyperSpy stores raw LaTeX, pyxem has its own equivalents table, and
#: instrument vendors have their own habits again.
_SPELLINGS = {
    PIXELS: ("px", "pixel", "pixels"),
    MILLIRADIAN: ("mrad", "mrads", "milliradian", "milliradians"),
    PER_NANOMETRE: (
        "nm^-1", "nm-1", "1/nm", "nm⁻¹", "k_nm^-1", "nm$^{-1}$", "$nm^{-1}$",
        "nm^{-1}", "nm¯¹", "per nm", "1 / nm",
    ),
    PER_ANGSTROM: (
        "a^-1", "a-1", "1/a", "a⁻¹", "å^-1", "å-1", "1/å", "å⁻¹", "k_a^-1",
        r"$\aa^{-1}$", "$a^{-1}$", "a^{-1}", "å$^{-1}$", "angstrom^-1",
        "angstrom-1", "1/angstrom", "per angstrom", "ang^-1",
    ),
}

_CANONICAL_BY_SPELLING = {
    spelling: unit
    for unit, spellings in _SPELLINGS.items()
    for spelling in spellings
}

#: Where the Å⁻¹ calibration is kept so a conversion through ``px`` (which has
#: no scale to convert back from) stays reversible. In ``metadata`` rather than
#: on the object so it survives a save and reload.
_STASH = "Signal.spyde_reciprocal_scale"

# Electron rest energy (keV) and hc, for the relativistic wavelength. Same
# constants as ``actions/sem_geometry``, which cannot be imported here — this
# module has to stay dependency-light enough for the backend to import it on
# the load path.
_ELECTRON_REST_ENERGY_KEV = 510.998950
_HC_KEV_ANGSTROM = 12.3984198


def parse_unit(text) -> str | None:
    """Which of the four units *text* names, or ``None`` if it names none.

    Case- and whitespace-insensitive. ``None`` covers both a blank label and an
    unrecognised one — those mean *unspecified*, which is not the same as
    ``px``: plenty of files carry a real reciprocal scale and never say what it
    is in, whereas ``px`` is a positive claim that the axis counts pixels.
    Callers reading a signal treat unspecified as :data:`CANONICAL`, which is
    the behaviour the app had before units were tracked at all.
    """
    if text is None:
        return None
    key = str(text).strip().lower().replace(" ", "")
    if key in _CANONICAL_BY_SPELLING:
        return _CANONICAL_BY_SPELLING[key]
    # Retry without the whitespace-stripping, for the few spellings that
    # contain a meaningful space ("per nm").
    return _CANONICAL_BY_SPELLING.get(str(text).strip().lower())


def is_reciprocal(unit) -> bool:
    """Is *unit* a reciprocal spacing (as opposed to pixels or an angle)?"""
    return parse_unit(unit) in _INVERSE_ANGSTROM_PER


def electron_wavelength_angstrom(accelerating_voltage_kv: float) -> float:
    """Relativistic electron wavelength (Å) for an accelerating voltage (kV).

    ``λ = hc / √(eV(2m₀c² + eV))`` — the standard relativistic form, and the
    one diffsims uses, so a library and the data it is matched against agree.
    """
    voltage = float(accelerating_voltage_kv)
    return _HC_KEV_ANGSTROM / np.sqrt(
        voltage * (2.0 * _ELECTRON_REST_ENERGY_KEV + voltage))


def beam_energy_kv(signal) -> float | None:
    """Accelerating voltage (kV) from *signal*'s metadata, or ``None``.

    Tries the places the value actually lives across the stack: pyxem's
    calibration object, HyperSpy's standard metadata tree, and the attribute
    pyxem's own signals expose.
    """
    for getter in (
        lambda: signal.metadata.get_item("Acquisition_instrument.TEM.beam_energy"),
        lambda: signal.calibration.beam_energy,
        lambda: signal.beam_energy,
    ):
        try:
            value = getter()
        except Exception:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return value
    return None


def wavelength_angstrom(signal) -> float | None:
    """Electron wavelength (Å) for *signal*, or ``None`` if unknowable.

    Prefers a wavelength recorded directly (pyxem writes one) over deriving it
    from the accelerating voltage, so a dataset that carries both stays
    self-consistent.
    """
    try:
        recorded = signal.metadata.get_item("Acquisition_instrument.TEM.wavelength")
    except Exception:
        recorded = None
    if recorded is not None:
        # pyxem may store this as a pint Quantity.
        recorded = getattr(recorded, "m", recorded)
        try:
            recorded = float(recorded)
        except (TypeError, ValueError):
            recorded = None
        if recorded is not None and np.isfinite(recorded) and recorded > 0:
            return recorded
    energy = beam_energy_kv(signal)
    return None if energy is None else electron_wavelength_angstrom(energy)


def inverse_angstrom_factor(unit, *, wavelength: float | None = None
                            ) -> float | None:
    """Multiply a value in *unit* by this to get Å⁻¹; ``None`` if impossible.

    ``px`` always returns ``None``: a pixel index has no physical size, so the
    caller has to go to the signal's stashed calibration instead of guessing.
    ``mrad`` needs *wavelength* (Å) and returns ``None`` without it.
    """
    canonical = parse_unit(unit)
    if canonical in _INVERSE_ANGSTROM_PER:
        return _INVERSE_ANGSTROM_PER[canonical]
    if canonical == MILLIRADIAN:
        if wavelength is None or not np.isfinite(wavelength) or wavelength <= 0:
            return None
        # θ = k·λ, with θ in radians. pyxem writes the flat-Ewald form
        # tan(θ)/λ instead, but a detector PIXEL subtends well under a
        # milliradian, so the two agree to about one part in 10⁷ — far below
        # any calibration this converts. The exact remap, which does matter
        # across a whole SEM detector, is sem_geometry.correct_vectors_flat_detector.
        return float(1.0 / (1e3 * wavelength))
    return None


def is_unlabelled(units) -> bool:
    """Did this axis simply never say what it is in?

    Distinct from carrying a label that is not one of the four: ``"nm"`` on a
    scan axis is a real unit and a definite "not a reciprocal axis", whereas a
    blank is an absence, and the two are read differently.
    """
    return str(units or "").strip().lower() in ("", "<undefined>", "none")


def _factor_for_axis(units, scale, wavelength) -> float | None:
    """:func:`inverse_angstrom_factor` for one axis, resolving the two labels
    that do not mean what they say.

    An axis with **no** label is read as :data:`CANONICAL` — that is what the
    app did before it tracked units, so a dataset that never recorded what its
    detector was calibrated in keeps indexing exactly as it did.

    An axis labelled ``px`` whose scale is not 1 is read the same way. ``px`` is
    what pyxem stamps on a signal's axes by DEFAULT when it types it as
    electron diffraction, so on a calibrated detector the label is left over
    rather than asserted; a genuine pixel axis always has a scale of one.

    A label that IS present but is none of the four (``"nm"``, ``"eV"``) gets
    ``None``: it is a positive statement that this is not a reciprocal axis.
    """
    unit = parse_unit(units)
    if unit is None:
        return 1.0 if is_unlabelled(units) else None
    if unit == PIXELS and float(scale) != 1.0:
        return 1.0
    return inverse_angstrom_factor(units, wavelength=wavelength)


def _signal_axes(signal):
    try:
        return list(signal.axes_manager.signal_axes)
    except Exception:
        return []


def _stashed_scales(signal) -> list[float] | None:
    """The Å⁻¹-per-pixel scales remembered for *signal*, if any."""
    try:
        stashed = signal.metadata.get_item(_STASH)
    except Exception:
        return None
    if stashed is None:
        return None
    try:
        scales = [float(s) for s in stashed]
    except (TypeError, ValueError):
        return None
    return scales if scales and all(np.isfinite(s) and s > 0 for s in scales) else None


def _stash_scales(signal, scales) -> None:
    try:
        signal.metadata.set_item(_STASH, [float(s) for s in scales])
    except Exception as e:
        log.debug("stashing the reciprocal calibration failed: %s", e)


def inverse_angstrom_scales(signal) -> list[float] | None:
    """Å⁻¹ per pixel for each signal axis, or ``None`` if not knowable.

    The answer does not depend on how the axes are currently *labelled*: an
    axis in nm⁻¹ is converted, one in mrad is converted through the wavelength,
    and one in px falls back to the calibration stashed before the conversion
    that discarded it. ``None`` means the detector genuinely has no physical
    calibration, and the caller must say so rather than assume Å⁻¹.
    """
    axes = _signal_axes(signal)
    if not axes:
        return None
    wavelength = wavelength_angstrom(signal)
    scales: list[float] = []
    for axis in axes:
        try:
            scale = float(axis.scale)
        except (TypeError, ValueError):
            return None
        factor = _factor_for_axis(getattr(axis, "units", ""), scale, wavelength)
        if factor is None:
            # px, or mrad with no beam energy — neither carries a reciprocal
            # scale of its own, so fall back to what was stashed before the
            # conversion that discarded it.
            return _stashed_scales(signal)
        value = scale * factor
        if not np.isfinite(value) or value <= 0:
            return None
        scales.append(value)
    return scales


def inverse_angstrom_scale(signal) -> float | None:
    """Å⁻¹ per pixel for *signal*, averaged over its signal axes.

    A detector's two axes share a pixel size in every dataset SpyDE handles, so
    the mean is the scalar the physics wants; it differs from either axis only
    when a file is calibrated anisotropically, which is itself worth knowing.
    """
    scales = inverse_angstrom_scales(signal)
    if not scales:
        return None
    return float(np.mean(scales))


def to_inverse_angstrom(values, signal):
    """Convert *values* expressed in *signal*'s current signal-axis units to Å⁻¹.

    Raises ``ValueError`` when the signal has no physical calibration — a
    silently unconverted vector set is a ten-times-wrong answer that still
    looks plausible.
    """
    factor = axis_unit_factor(signal)
    if factor is None:
        raise ValueError(
            "This dataset's detector axes have no reciprocal calibration, so "
            "vectors cannot be expressed in Å⁻¹. Set the detector scale (or a "
            "beam energy, for axes in mrad) in the Plot Control dock.")
    return np.asarray(values, dtype=np.float64) * factor


def from_inverse_angstrom(values, signal):
    """Convert *values* in Å⁻¹ back to *signal*'s current signal-axis units —
    the inverse of :func:`to_inverse_angstrom`, for drawing a simulated overlay
    on a diffraction pattern that is displayed in some other unit."""
    factor = axis_unit_factor(signal)
    if factor is None:
        raise ValueError("This dataset's detector axes have no reciprocal "
                         "calibration, so Å⁻¹ values cannot be placed on them.")
    return np.asarray(values, dtype=np.float64) / factor


def axis_unit_factor(signal) -> float | None:
    """Multiply a value in *signal*'s CURRENT signal-axis units by this to get
    Å⁻¹, or ``None`` if the axes carry no physical calibration.

    This is the per-unit factor (10 for an axis in nm⁻¹), not the per-pixel
    scale — use it on measured vectors and tolerances, which are already in
    axis units. For a pixel axis it is derived from the stashed calibration,
    since one pixel then IS one unit.
    """
    axes = _signal_axes(signal)
    if not axes:
        return None
    units = getattr(axes[0], "units", "")
    try:
        scale = float(axes[0].scale)
    except (TypeError, ValueError):
        return None
    factor = _factor_for_axis(units, scale, wavelength_angstrom(signal))
    if factor is not None:
        return factor
    if parse_unit(units) == PIXELS:
        # One pixel IS one unit here, so the per-pixel scale is the per-unit
        # factor — recoverable only from what was stashed before the axes were
        # converted to pixels.
        stashed = _stashed_scales(signal)
        if stashed:
            return float(np.mean(stashed))
    return None


def reciprocal_extent(signal) -> float | None:
    """The largest radius that fits inside the detector in EVERY signal
    dimension, in Å⁻¹ — the half-extent of the narrowest axis.

    This is the outer radius a simulated template library should be generated
    to: spots beyond it cannot land on the detector.
    """
    scales = inverse_angstrom_scales(signal)
    axes = _signal_axes(signal)
    if not scales or len(scales) != len(axes):
        return None
    try:
        return float(min(s * float(ax.size) / 2.0 for s, ax in zip(scales, axes)))
    except (TypeError, ValueError):
        return None


def current_unit(signal) -> str | None:
    """Which of the four units *signal*'s signal axes are currently in.

    Resolved the same way the physics resolves it, so an axis whose label is
    blank or a left-over ``px`` on a real calibration reports the Å⁻¹ it is
    actually being read as, rather than what it happens to say.
    """
    axes = _signal_axes(signal)
    if not axes:
        return None
    units = getattr(axes[0], "units", "")
    unit = parse_unit(units)
    try:
        scale = float(axes[0].scale)
    except (TypeError, ValueError):
        return unit
    if _factor_for_axis(units, scale, None) == 1.0 and unit != PER_ANGSTROM:
        return CANONICAL
    return unit


def available_units(signal) -> dict[str, str]:
    """``{unit: ""}`` for each unit *signal* can be converted to, or
    ``{unit: reason}`` naming what is missing for the ones it cannot.

    The toggle shows every unit either way — an option that is simply absent
    tells the user nothing, whereas a disabled one with "needs a beam energy"
    tells them exactly what to fill in.
    """
    axes = _signal_axes(signal)
    labelled = parse_unit(getattr(axes[0], "units", "")) if axes else None
    has_scale = inverse_angstrom_scales(signal) is not None
    has_wavelength = wavelength_angstrom(signal) is not None

    # A detector already labelled in mrad cannot be read at all without the
    # wavelength, so the beam energy blocks every unit, not just mrad — saying
    # "needs a detector scale" there would send the user off to recalibrate a
    # detector that is perfectly well calibrated.
    if labelled == MILLIRADIAN and not has_wavelength:
        missing = "needs a beam energy"
    elif not has_scale:
        missing = "needs a detector scale"
    else:
        missing = ""

    reasons = {unit: missing for unit in TOGGLE_ORDER}
    reasons[PIXELS] = ""                     # always available: it discards
    if not missing and not has_wavelength:
        reasons[MILLIRADIAN] = "needs a beam energy"
    return reasons


def convert_signal_axes(signal, to_unit: str) -> bool:
    """Re-express *signal*'s signal axes in *to_unit*, RESCALING them.

    Returns whether the conversion happened. The Å⁻¹ calibration is stashed
    first, so converting to ``px`` — the one direction that discards the
    physical scale — can still be undone.

    The origin is preserved rather than recentred: ``offset`` scales with
    ``scale``, so a detector whose direct beam sits off-centre keeps its
    measured centre instead of being snapped to the middle of the array.
    """
    target = parse_unit(to_unit)
    if target is None:
        log.warning("unknown reciprocal unit %r", to_unit)
        return False
    axes = _signal_axes(signal)
    if not axes:
        return False

    scales = inverse_angstrom_scales(signal)
    if scales is None:
        if target != PIXELS:
            return False
        scales = []                      # px → px on an uncalibrated detector
    else:
        _stash_scales(signal, scales)

    if target == PIXELS:
        new_scales = [1.0] * len(axes)
    elif not scales:
        return False
    elif target == MILLIRADIAN:
        wavelength = wavelength_angstrom(signal)
        if wavelength is None:
            return False
        factor = inverse_angstrom_factor(MILLIRADIAN, wavelength=wavelength)
        new_scales = [s / factor for s in scales]
    else:
        new_scales = [s / _INVERSE_ANGSTROM_PER[target] for s in scales]

    for axis, new_scale in zip(axes, new_scales):
        try:
            old_scale = float(axis.scale)
            old_offset = float(axis.offset)
            axis.scale = float(new_scale)
            if old_scale != 0.0:
                axis.offset = old_offset * (float(new_scale) / old_scale)
            axis.units = target
        except Exception as e:
            log.warning("converting a signal axis to %s failed: %s", target, e)
            return False
    return True


def normalize_to_canonical(signal) -> bool:
    """Put *signal*'s detector axes into Å⁻¹, the unit everything computes in.

    Called once when a diffraction dataset is opened. A scan already in Å⁻¹, an
    uncalibrated one, and a non-diffraction signal are all left exactly as they
    are — this converts a real reciprocal calibration and nothing else, so it
    can run on every file without having to ask what kind it is.
    """
    unit = current_unit(signal)
    if unit is None or unit == PIXELS or unit == CANONICAL:
        return False
    if unit == MILLIRADIAN and wavelength_angstrom(signal) is None:
        return False
    return convert_signal_axes(signal, CANONICAL)

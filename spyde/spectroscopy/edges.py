"""edges.py — offer ONE EELS core-loss edge at a time, from the component picker.

:mod:`~spyde.spectroscopy.composition` builds a whole model from a list of
elements. That is the right tool when the composition is known, but it is the
only tool there was: the Fit caret's ``+ Component`` picker offered nine
analytic shapes (Gaussian, PowerLaw, …) and no way to reach ``EELSCLEdge`` at
all, so the only route to an edge was "From Fe, Ni, Cu", which REPLACES the
model wholesale. Adding one O-K edge to a background you had already tuned was
not possible.

This module is the missing half: *which* edges make sense for this signal, and
how to turn one subshell name into a :class:`~spyde.fitting.spec.ComponentSpec`
the caret can append.

Three things make an edge unlike every other component in the picker:

* **It takes a constructor argument.** ``EELSCLEdge("Fe_L3")`` — there is no
  bare edge, so it cannot be a single button the way ``Gaussian`` is. The
  picker needs the subshell, which is why :func:`available_edges` exists.
* **It needs the MICROSCOPE, not just the axis.** The GOS integral is
  evaluated at an effective collection angle derived from the beam energy and
  the convergence/collection angles. Without them exspy raises
  ``AttributeError('Acquisition_instrument')`` from inside ``model.append`` —
  which is exactly the kind of failure a user cannot act on, so
  :func:`missing_microscope_parameters` is checked FIRST and reported by name.
* **It only resolves ON a model.** ``EELSCLEdge("O_K").function(x)`` raises
  (``GoshGOS has no attribute energy_shift``) because the cross-section is
  integrated when the edge is appended to a model of a real EELS signal. So
  :func:`edge_component_spec` builds a throwaway model to read the shape off,
  rather than sampling the bare component.

exspy stays the authority on where the edges are and what they look like —
this module only decides which of them are worth offering and does the
plumbing.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.spectroscopy.composition import MissingExtra, _require_exspy

log = logging.getLogger(__name__)

#: The microscope geometry an ``EELSCLEdge`` needs, as ``(metadata key, label)``.
#: The collection angle lives under the EELS detector, not next to the other
#: two — a detail worth spelling out, because looking for it on ``TEM`` is why
#: "the beam energy is set, so why is it still complaining" happens.
MICROSCOPE_PARAMETERS = (
    ("Acquisition_instrument.TEM.beam_energy", "beam energy"),
    ("Acquisition_instrument.TEM.convergence_angle", "convergence angle"),
    ("Acquisition_instrument.TEM.Detector.EELS.collection_angle",
     "collection angle"),
)


class MissingMicroscopeParameters(RuntimeError):
    """Raised when an edge is asked for before the microscope is described.

    Carries :attr:`missing` so a caller can name the fields rather than
    forwarding a sentence.
    """

    def __init__(self, missing):
        self.missing = list(missing)
        super().__init__(
            "an EELS edge needs the microscope geometry — set the "
            + ", ".join(self.missing)
            + " in the Metadata panel (or call "
              "signal.set_microscope_parameters(...)) and try again")


def is_eels(signal) -> bool:
    """True for an EELS signal, by CLASS or by declared metadata.

    ``_signal_type`` (the class attribute) is the idiom
    :func:`spyde.spectroscopy.composition._signal_kind` uses, and it is the
    authoritative one — but it is not sufficient here, because **without exspy
    there is no EELS class at all**: ``set_signal_type("EELS")`` writes
    ``metadata.Signal.signal_type`` and then leaves the object a plain
    ``Signal1D``, with only a log line to say so.

    Falling back to the metadata string is what makes the missing-extra case
    reachable. Otherwise a user who opens an EELS dataset without the ``eels``
    extra sees no edge section and no explanation — the picker simply has
    nothing in it, which is the failure mode this whole module exists to
    remove. With the fallback they get "EELS edges need exspy — pip install
    "spyde[eels]"" exactly where they went looking for the edge.
    """
    if "EELS" in (getattr(signal, "_signal_type", "") or "").upper():
        return True
    try:
        declared = signal.metadata.get_item("Signal.signal_type", "") or ""
    except Exception:                                    # pragma: no cover
        return False
    return "EELS" in str(declared).upper()


def missing_microscope_parameters(signal) -> list[str]:
    """Human names of the microscope parameters this signal is missing.

    Empty means an edge can be built. Anything else is the message to show —
    a non-numeric or NaN value counts as missing, because exspy fails just as
    hard on those as on an absent key and the user would have no idea why.
    """
    md = getattr(signal, "metadata", None)
    if md is None:
        return [label for _key, label in MICROSCOPE_PARAMETERS]
    out = []
    for key, label in MICROSCOPE_PARAMETERS:
        try:
            value = md.get_item(key, None)
        except Exception:                                # pragma: no cover
            value = None
        try:
            ok = value is not None and np.isfinite(float(value))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append(label)
    return out


def _energy_range(signal) -> tuple[float, float]:
    ax = signal.axes_manager.signal_axes[0].axis
    return float(np.min(ax)), float(np.max(ax))


def _elements_on(signal) -> list[str]:
    """``metadata.Sample.elements``, or an empty list.

    This is what Plot Control's Composition panel writes (see
    :mod:`spyde.actions.composition`), so setting the composition seeds the
    picker without any further wiring.
    """
    try:
        got = signal.metadata.get_item("Sample.elements", None)
    except Exception:                                    # pragma: no cover
        return []
    return [str(e) for e in got] if got else []


def _subshells_for(element: str) -> dict:
    # `exspy.material.elements`, not `exspy.misc.elements` — the latter warns
    # (VisibleDeprecationWarning) and goes away in exspy 1.0.
    from exspy.material import elements

    try:
        binding = elements[element]["Atomic_properties"]["Binding_energies"]
    except (KeyError, AttributeError):
        return {}
    try:
        return binding.as_dictionary()
    except AttributeError:                               # pragma: no cover
        return dict(binding)


def _entry(subshell: str, info: dict, *, suggested: bool) -> dict | None:
    onset = info.get("onset_energy (eV)")
    if onset is None:
        return None
    element, _, shell = str(subshell).partition("_")
    relevance = str(info.get("relevance") or "")
    return {
        "subshell": str(subshell),
        "element": element,
        "shell": shell,
        "onset": float(onset),
        "relevance": relevance,
        # What the picker puts in the tooltip. `edge` is exspy's own shape
        # description ("Delayed maximum", "Abrupt onset") — the one piece of
        # information that says what the component will actually look like.
        "description": " ".join(
            p for p in (f"{element} {shell} edge at {float(onset):g} eV",
                        f"({relevance.lower()})" if relevance else "",
                        str(info.get("edge") or "")) if p),
        "suggested": bool(suggested),
    }


def available_edges(signal, *, elements=None, energy_range=None) -> list[dict]:
    """Every EELS edge whose onset falls inside the measured range.

    Each entry is ``{subshell, element, shell, onset, relevance, description,
    suggested}``, sorted by onset. ``suggested`` marks the edges belonging to
    the elements the user has already declared
    (``metadata.Sample.elements``, i.e. Plot Control's Composition panel) —
    the picker leads with those, because "I know there is oxygen here, give me
    the O-K edge" is the common case and hunting for it among the ~130 edges a
    600 eV window contains is not.

    Only the onset is used to decide "in range", which matches
    :func:`~spyde.spectroscopy.composition.prune_to_range`: an edge whose
    onset is off-screen has an unconstrained intensity, and an unconstrained
    component makes every other parameter worse rather than merely wasting
    one.
    """
    _require_exspy()
    import exspy.utils.eels as eels_utils

    lo, hi = energy_range if energy_range else _energy_range(signal)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"the signal axis {lo!r}-{hi!r} is not an energy range")

    wanted = list(elements) if elements is not None else _elements_on(signal)

    out: dict[str, dict] = {}
    for element in wanted:
        for shell, info in _subshells_for(str(element)).items():
            entry = _entry(f"{element}_{shell}", info, suggested=True)
            if entry is not None and lo <= entry["onset"] <= hi:
                out[entry["subshell"]] = entry

    # Everything else in the window, so the picker is a real catalogue and not
    # only a mirror of the composition. `get_edges_near_energy` is exspy's own
    # lookup; `only_major=False` keeps the minor edges, which the renderer
    # hides until the user types an element symbol.
    try:
        near = eels_utils.get_edges_near_energy(
            (lo + hi) / 2.0, width=(hi - lo), only_major=False)
        for subshell, info in zip(near, eels_utils.get_info_from_edges(near)):
            if subshell in out:
                continue
            entry = _entry(subshell, info, suggested=False)
            if entry is not None and lo <= entry["onset"] <= hi:
                out[subshell] = entry
    except Exception as e:                               # pragma: no cover
        log.info("listing the edges near %g-%g eV failed: %s", lo, hi, e)

    return sorted(out.values(),
                  key=lambda d: (not d["suggested"], d["onset"]))


def edge_component_spec(signal, element_subshell: str):
    """A :class:`~spyde.fitting.spec.ComponentSpec` for one ``EELSCLEdge``.

    The edge is built ON a throwaway model of *signal* rather than bare,
    because that append is what makes exspy integrate the GOS at this
    microscope's effective angle — a bare ``EELSCLEdge`` cannot even evaluate
    its own ``function``.

    Raises :class:`MissingExtra` without exspy and
    :class:`MissingMicroscopeParameters` before the microscope is described.
    Both are checked here rather than left to fail inside exspy, because the
    native failures (``ImportError``, ``AttributeError('Acquisition_instrument')``)
    say nothing a user can act on.
    """
    _require_exspy()
    import exspy.components as exspy_components
    from spyde.fitting.spec import ModelSpec

    if not is_eels(signal):
        raise ValueError(
            f"an EELS edge needs an EELS signal; this one is "
            f"{getattr(signal, '_signal_type', None)!r} — set the signal type "
            f"to EELS in Plot Control first")
    missing = missing_microscope_parameters(signal)
    if missing:
        raise MissingMicroscopeParameters(missing)

    edge = exspy_components.EELSCLEdge(str(element_subshell))
    # `auto_add_edges=False` matters: with elements on the signal, the default
    # would populate the model with an edge per subshell and the spec read back
    # would be one of THOSE, not the edge asked for.
    model = signal.create_model(auto_background=False, auto_add_edges=False)
    while len(model):                                    # pragma: no cover
        model.remove(model[0])
    model.append(edge)

    # `ModelSpec.from_model`, NOT `spec_from_component`. The latter is the
    # PICKER's reader and flattens every parameter to `np.ravel(value)[0]` —
    # which is right for the nine analytic shapes it was written for and wrong
    # here, because `fine_structure_coeff` is a VECTOR (12 spline coefficients
    # for O-K). A scalar there survives all the way to `to_model`, which then
    # raises `ValueError('The length of the parameter must be ', 12)` from
    # inside hyperspy — so the edge silently never gets its batched GOS curves
    # and the whole model drops onto the one-pixel-at-a-time path.
    return ModelSpec.from_model(model).components[0]


__all__ = ["MICROSCOPE_PARAMETERS", "MissingExtra",
           "MissingMicroscopeParameters", "available_edges",
           "edge_component_spec", "is_eels", "missing_microscope_parameters"]

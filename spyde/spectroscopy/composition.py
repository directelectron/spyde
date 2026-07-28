"""composition.py — elements in, a fitted-ready model out (#62).

The headline of Wave 2: a user types which elements are present and gets a
populated model, instead of hand-placing a component per edge or line.

**exspy does the spectroscopy, this module does the plumbing.** Where the
ionisation edges and X-ray lines are, what a GOS-based edge shape is, which
lines belong to a family — all of that is exspy's, and reimplementing it would
mean diverging from the numbers the community publishes. What is added here:

* one call that works for EELS and EDS alike and returns a
  :class:`~spyde.fitting.spec.ModelSpec`, so the result feeds the batched
  engine, the wizard and the renderer through one type;
* **pruning to the measured range** — exspy will happily add a Cu-L line at
  0.93 keV to a spectrum whose useful range starts at 2 keV, and a component
  with no data under it is an unconstrained parameter that makes the whole fit
  worse, not merely a wasted one;
* an honest report of whether the batched engine can actually fit the result.

That last point matters and is asymmetric today:

* **EDS** builds from Polynomial + Gaussian, both of which the batched engine
  implements — so EDS gets the full GPU path immediately.
* **EELS** builds from ``EELSCLEdge``, a tabulated GOS lookup with no batched
  port yet (#63), so an EELS model falls back to HyperSpy's own fitting.

:func:`model_for_composition` reports this rather than letting it be a
surprise, and the fallback is correct — just slower.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.fitting import ModelSpec

log = logging.getLogger(__name__)

# The parameter that positions each kind of component on the energy axis.
# Used for range pruning; a component whose position parameter is unknown is
# kept, because dropping something we do not understand is the worse error.
_POSITION_PARAM = {
    "Gaussian": "centre",
    "GaussianHF": "centre",
    "Lorentzian": "centre",
    "Voigt": "centre",
    "SplitVoigt": "centre",
    "EELSCLEdge": "onset_energy",
    "Erf": "origin",
    "Arctan": "x0",
}


class MissingExtra(RuntimeError):
    """Raised when the ``eels`` extra is needed but not installed."""


def _require_exspy():
    try:
        import exspy  # noqa: F401
    except ImportError as e:
        raise MissingExtra(
            'EELS/EDS models need exspy — install with: pip install "spyde[eels]"'
        ) from e


def _signal_kind(signal) -> str:
    st = (getattr(signal, "_signal_type", "") or "").upper()
    if "EELS" in st:
        return "EELS"
    if "EDS" in st:
        return "EDS"
    raise ValueError(
        f"signal type {getattr(signal, '_signal_type', None)!r} is neither "
        f"EELS nor EDS — call set_signal_type('EELS') / ('EDS_TEM') first "
        f"(needs the eels extra)")


def _axis_range(signal) -> tuple[float, float]:
    ax = signal.axes_manager.signal_axes[0].axis
    return float(np.min(ax)), float(np.max(ax))


def _component_position(comp_spec):
    name = _POSITION_PARAM.get(comp_spec.kind)
    if name is None:
        return None
    try:
        return float(comp_spec[name].value)
    except KeyError:
        return None


def prune_to_range(spec: ModelSpec, lo: float, hi: float, *,
                   margin: float = 0.0) -> tuple[ModelSpec, list[str]]:
    """Drop components positioned outside ``[lo, hi]``.

    A component with no data under it is not harmless: its amplitude is
    unconstrained, so the optimiser is free to trade it against everything
    else, which degrades the parameters that ARE measurable.

    Returns the pruned spec and the names removed.
    """
    kept, dropped = [], []
    for c in spec.components:
        pos = _component_position(c)
        if pos is not None and not (lo - margin <= pos <= hi + margin):
            dropped.append(c.name)
            continue
        kept.append(c)
    return ModelSpec(components=kept, channel_mask=spec.channel_mask), dropped


def model_for_composition(signal, elements=None, *, prune: bool = True,
                          energy_range: tuple[float, float] | None = None,
                          only_lines=None):
    """Build a :class:`ModelSpec` for the elements present in *signal*.

    Parameters
    ----------
    signal
        An EELS or EDS signal (needs the ``eels`` extra for the signal type to
        resolve at all).
    elements : sequence of str, optional
        e.g. ``["Fe", "Ni", "Cu"]``. Defaults to whatever is already on
        ``metadata.Sample.elements``.
    prune : bool
        Drop components positioned outside the measured range (see
        :func:`prune_to_range`).
    energy_range : (lo, hi), optional
        Override the range used for pruning — e.g. to exclude a noisy
        low-energy region the axis technically covers.
    only_lines : sequence of str, optional
        EDS only: restrict to these X-ray lines (``["Fe_Ka", "Cu_Ka"]``)
        instead of every line exspy knows for the element.

    Returns
    -------
    (spec, info)
        *info* carries ``kind``, ``elements``, ``dropped`` and
        ``engine_supported`` — the last being whether
        :mod:`spyde.fitting.engine` can fit this model or whether it falls back
        to HyperSpy (#63).
    """
    _require_exspy()
    from spyde.fitting import components as tcomp

    kind = _signal_kind(signal)
    s = signal.deepcopy()

    if elements:
        s.add_elements(list(elements))
    have = list(getattr(s.metadata, "Sample", {}).get_item("elements", [])
                if hasattr(getattr(s.metadata, "Sample", None), "get_item")
                else [])
    if not have:
        raise ValueError(
            "no elements to build a model from — pass elements=[...] or set "
            "metadata.Sample.elements")

    if kind == "EDS":
        try:
            if only_lines:
                # set_lines REPLACES the line list; add_lines only appends, so
                # using it here would leave every default line in place and
                # `only_lines` would silently do nothing.
                s.set_lines(list(only_lines))
            else:
                s.add_lines()
        except Exception as e:
            log.debug("setting X-ray lines failed (%s); relying on "
                      "create_model's defaults", e)

    model = s.create_model()
    spec = ModelSpec.from_model(model)

    if kind == "EDS" and only_lines:
        # exspy's EDS model expands each ELEMENT into its whole family
        # (selecting Fe_Ka still builds Fe_Kb, Fe_La, Fe_Ln, ...) regardless of
        # metadata.Sample.xray_lines, so `set_lines` alone does not restrict
        # the model. Filter the built spec by name instead, keeping anything
        # that is not a named line (the background).
        wanted = set(only_lines)
        spec = ModelSpec(
            components=[c for c in spec.components
                        if "_" not in c.name or c.name in wanted],
            channel_mask=spec.channel_mask)

    lo, hi = energy_range if energy_range else _axis_range(s)
    dropped: list[str] = []
    if prune:
        spec, dropped = prune_to_range(spec, lo, hi)
        if dropped:
            log.info("pruned %d component(s) outside %.4g-%.4g: %s",
                     len(dropped), lo, hi, ", ".join(dropped))

    info = {
        "kind": kind,
        "elements": have,
        "dropped": dropped,
        "engine_supported": tcomp.supports(spec),
        "energy_range": (lo, hi),
    }
    if not info["engine_supported"]:
        # Why each one is unfittable, from actually trying to build it — an
        # EELS edge is a SUPPORTED kind that still needs its GOS curves
        # precomputed, so a kind-only check would call it fittable and then
        # fail inside the engine. `prepare_eels_edges` is what resolves it.
        blocked = tcomp.unsupported(spec)
        info["unsupported_components"] = sorted(
            {c.kind for c in spec.active_components if c.name in blocked})
        info["unsupported_reasons"] = blocked
        log.info("the batched engine cannot fit %s yet — %s",
                 sorted(blocked), "; ".join(sorted(set(blocked.values()))))
    return spec, info

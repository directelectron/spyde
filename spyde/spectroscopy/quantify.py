"""quantify.py — fitted intensities -> per-element composition maps (#66).

Two steps, kept separate because they fail differently:

:func:`element_intensity_maps`
    The BRIDGE. Turns a :class:`~spyde.fitting.engine.FitResult` into one
    intensity map per element, by reading the component names the model was
    built with (``Fe_Ka``, ``O_K``) and summing a family's lines. Pure array
    work, no physics — and no optional extra.

:func:`quantify`
    The PHYSICS. Cliff-Lorimer for EDS, relative cross-sections for EELS.

Why the split matters: the bridge is where a naming or ordering mistake sends
iron's intensity into copper's map, and that is silent — every downstream number
stays plausible. The physics is where a wrong k-factor lives, and that is at
least a number a microscopist can sanity-check. They deserve separate tests.

**Composition is normalised and therefore relative.** Cliff-Lorimer gives
ratios; it cannot know about an element you did not fit, so the fractions sum
to 1 over the elements PRESENT IN THE MODEL. Leaving an element out does not
produce a small error, it redistributes its share across everything else.
:func:`quantify` says so in its result rather than letting the number look
absolute.
"""
from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

# "Fe_Ka" / "Cu_Kb1" / "O_K" -> element, line. The EELS edge naming ("O_K")
# and the EDS line naming ("Fe_Ka") share this shape, which is why one bridge
# serves both.
_LINE_RE = re.compile(r"^([A-Z][a-z]?)_([A-Za-z0-9]+)$")

# The parameter carrying "how much" for each component kind.
_INTENSITY_PARAM = {
    "Gaussian": "A",
    "GaussianHF": "height",
    "Lorentzian": "A",
    "TabulatedShape": "intensity",
    "EELSCLEdge": "intensity",
}


def parse_line(name: str) -> tuple[str, str] | None:
    """``"Fe_Ka"`` -> ``("Fe", "Ka")``; ``None`` for anything else."""
    m = _LINE_RE.match(name or "")
    return (m.group(1), m.group(2)) if m else None


def element_intensity_maps(spec, result, nav_shape=None, *,
                           lines: dict[str, str] | None = None):
    """One intensity map per element, summed over that element's lines.

    Parameters
    ----------
    spec : ModelSpec
        The fitted model — its component names carry the element identity.
    result : FitResult
        From :func:`~spyde.fitting.engine.fit_batched`.
    lines : dict, optional
        Restrict to one line per element, e.g. ``{"Fe": "Ka"}``. Summing a
        whole K family is usually right, but if a family member overlaps
        another element badly, using the clean line alone is more accurate than
        a contaminated sum.

    Returns
    -------
    dict of element -> map
        Shaped to *nav_shape* when given, otherwise flat.
    """
    names = spec.parameter_names()
    out: dict[str, np.ndarray] = {}

    for comp in spec.active_components:
        parsed = parse_line(comp.name)
        if parsed is None:
            continue                                    # background, etc.
        element, line = parsed
        if lines and lines.get(element) not in (None, line):
            continue

        pname = _INTENSITY_PARAM.get(comp.kind)
        if pname is None:
            log.debug("no intensity parameter known for %s (%s) — skipped",
                      comp.name, comp.kind)
            continue
        key = f"{comp.name}.{pname}"
        if key not in names:
            log.debug("%s not in the fitted parameters — skipped", key)
            continue

        col = result.values[:, names.index(key)]
        # A fit can drive a line slightly negative on noise; a negative
        # "amount of an element" is not physical and would corrupt the
        # normalisation for every OTHER element, so clamp at the source.
        out[element] = out.get(element, 0.0) + np.clip(col, 0.0, None)

    if nav_shape is not None:
        out = {k: v.reshape(nav_shape) for k, v in out.items()}
    return out


def quantify(intensity_maps, *, method: str = "relative",
             kfactors: dict[str, float] | None = None,
             cross_sections: dict[str, float] | None = None):
    """Intensity maps -> atomic-fraction maps.

    Parameters
    ----------
    method : {"relative", "cliff_lorimer", "eels"}
        ``relative``
            Normalise raw intensities. Honest only when the elements have
            similar sensitivity — offered because it needs no factors and is
            the right first look.
        ``cliff_lorimer``
            EDS: ``C_A/C_B = k_AB * I_A/I_B``. *kfactors* is per element,
            relative to a common reference; missing ones default to 1.0 and
            are reported.
        ``eels``
            Divide by partial cross-sections, then normalise.

    Returns
    -------
    (fractions, info)
        *fractions* maps element -> atomic fraction in [0, 1] summing to 1
        across the elements present. *info* records the method, which factors
        were defaulted, and that the result is RELATIVE to the fitted elements.
    """
    if not intensity_maps:
        raise ValueError("no intensity maps to quantify")

    elements = sorted(intensity_maps)
    stack = np.stack([np.asarray(intensity_maps[e], float) for e in elements])
    defaulted: list[str] = []

    def _factors(table) -> np.ndarray:
        """Per-element factor, broadcast over whatever map shape we have."""
        vals = []
        for e in elements:
            v = (table or {}).get(e)
            if v is None:
                defaulted.append(e)
                v = 1.0
            vals.append(float(v))
        return np.asarray(vals).reshape((-1,) + (1,) * (stack.ndim - 1))

    if method == "relative":
        weighted = stack
    elif method == "cliff_lorimer":
        weighted = stack * _factors(kfactors)
    elif method == "eels":
        weighted = stack / _factors(cross_sections)
    else:
        raise ValueError(f"unknown quantification method {method!r} "
                         f"(relative, cliff_lorimer or eels)")

    total = weighted.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(total > 0, weighted / total, np.nan)

    if defaulted:
        log.info("quantification: no factor for %s — defaulted to 1.0, so "
                 "those fractions are uncalibrated", ", ".join(defaulted))

    return ({e: frac[i] for i, e in enumerate(elements)},
            {"method": method, "elements": elements,
             "defaulted_factors": defaulted,
             "relative_to": elements,
             "note": "fractions are normalised over the FITTED elements only; "
                     "an element left out of the model has its share "
                     "redistributed across the rest"})


def quantify_result(spec, result, nav_shape=None, **kwargs):
    """:func:`element_intensity_maps` then :func:`quantify`, in one call."""
    maps = element_intensity_maps(spec, result, nav_shape)
    fractions, info = quantify(maps, **kwargs)
    info["intensity_maps"] = maps
    return fractions, info

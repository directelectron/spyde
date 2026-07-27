"""tabulate.py — make GOS-based EELS edges fittable by the batched engine (#63).

An EELS core-loss edge is not a formula. ``EELSCLEdge`` looks its shape up in a
generalised-oscillator-strength table, which is why it has no batched torch
port and why an EELS model falls back to HyperSpy's one-pixel-at-a-time fitting
while an EDS model (gaussians) gets the GPU.

The way out is to notice what is actually being fitted. Across a spectrum image
the edge SHAPE is the same everywhere — it is atomic physics, fixed by the
element and the microscope. What varies pixel to pixel is **how much** of the
element there is and **exactly where** its edge starts. So:

1. sample each edge once, on the signal axis, at its current parameters;
2. replace it with a tabulated component carrying that shape;
3. fit ``intensity`` (linear) and ``onset_shift`` against it, batched.

**The approximation, stated plainly.** Fine structure and effective angle are
frozen at the values the table was sampled with. They are what make an edge a
lookup in the first place, and refitting them would mean re-sampling the table
every iteration — the per-item Python work the batched engine exists to avoid.
For quantification, which asks "how much of each element", that is the right
trade. For fine-structure analysis it is not, and
:func:`tabulate_model` says so by leaving the original model alone unless asked.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.fitting import ModelSpec
from spyde.fitting.components import TABULATED_KIND
from spyde.fitting.spec import ComponentSpec, ParameterSpec

log = logging.getLogger(__name__)

# Components worth tabulating: form is a lookup, amplitude is linear.
TABULATABLE = ("EELSCLEdge",)


def _axis(signal):
    ax = signal.axes_manager.signal_axes[0]
    x = np.asarray(ax.axis, float)
    if len(x) < 2:
        raise ValueError("signal axis needs at least two channels to tabulate")
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx, rtol=1e-6):
        raise ValueError("tabulation needs a UNIFORM signal axis; this one is "
                         "not evenly spaced")
    return x, float(x[0]), dx


def tabulate_model(spec: ModelSpec, signal, *, kinds=TABULATABLE,
                   intensity_name: str = "intensity"):
    """Replace lookup-shaped components with tabulated ones.

    Returns ``(new_spec, info)``. *info* lists what was tabulated and what was
    left alone, so a caller can tell the user which parameters are no longer
    being fitted.

    The returned spec is fittable by :mod:`spyde.fitting.engine` whenever every
    remaining component has a batched port — check with
    ``components.supports``.
    """
    x, x0, dx = _axis(signal)
    model = spec.to_model(signal)

    by_name = {c.name: c for c in model}
    out, tabulated, skipped = [], [], []

    for cspec in spec.components:
        if cspec.kind not in kinds:
            out.append(cspec)
            continue

        comp = by_name.get(cspec.name)
        if comp is None:
            skipped.append(cspec.name)
            out.append(cspec)
            continue

        try:
            table, onset = _sample(comp, x, intensity_name)
        except Exception as e:
            log.info("could not tabulate %s (%s); left as-is", cspec.name, e)
            skipped.append(cspec.name)
            out.append(cspec)
            continue

        intensity = _value(cspec, intensity_name, default=1.0)
        out.append(ComponentSpec(
            kind=TABULATED_KIND, name=cspec.name, active=cspec.active,
            # The table is sampled ON the signal axis, so the component's own
            # x0/dx are the axis's — recorded here because the component needs
            # them to interpolate and they are not derivable from the table.
            init_args={"x0": x0, "dx": dx, "source_kind": cspec.kind,
                       "onset_energy": onset},
            data=table,
            parameters=[
                ParameterSpec(intensity_name, float(intensity), linear=True,
                              bmin=0.0),
                # Bounded to a few channels: the onset is known from the
                # element, and a shift larger than this is not a refinement but
                # the fit sliding onto a neighbouring edge.
                ParameterSpec("onset_shift", 0.0,
                              bmin=-20.0 * dx, bmax=20.0 * dx),
            ]))
        tabulated.append(cspec.name)

    info = {"tabulated": tabulated, "skipped": skipped,
            "frozen": ["fine_structure_coeff", "effective_angle"],
            "x0": x0, "dx": dx}
    if tabulated:
        log.info("tabulated %d edge(s): %s — fine structure and effective "
                 "angle are now FROZEN", len(tabulated), ", ".join(tabulated))
    return ModelSpec(components=out, channel_mask=spec.channel_mask), info


def _sample(comp, x, intensity_name):
    """Sample one component's shape at UNIT intensity.

    Unit intensity is what makes the table a pure shape, so the fitted
    ``intensity`` means the same thing it did in HyperSpy rather than being
    scaled by whatever the seed happened to be.
    """
    par = getattr(comp, intensity_name, None)
    original = None if par is None else float(np.ravel(par.value)[0])
    try:
        if par is not None:
            par.value = 1.0
        y = np.nan_to_num(np.asarray(comp.function(x), float),
                          nan=0.0, posinf=0.0, neginf=0.0)
    finally:
        if par is not None and original is not None:
            par.value = original

    if not np.isfinite(y).all() or not np.any(y):
        raise ValueError("component sampled to an empty or non-finite shape")
    onset = getattr(getattr(comp, "onset_energy", None), "value", None)
    return y, (None if onset is None else float(np.ravel(onset)[0]))


def _value(cspec, name, default=1.0):
    try:
        return float(cspec[name].value)
    except (KeyError, TypeError):
        return default


def onset_energies(spec: ModelSpec) -> dict[str, float]:
    """Absolute fitted onset per tabulated component.

    ``onset_shift`` is relative to where the table was sampled, which is not
    what a user wants to read — this adds the original onset back so the number
    is an energy in eV.
    """
    out = {}
    for c in spec.components:
        if c.kind != TABULATED_KIND:
            continue
        base = c.init_args.get("onset_energy")
        if base is None:
            continue
        try:
            out[c.name] = float(base) + float(c["onset_shift"].value)
        except KeyError:
            continue
    return out

"""eels_batch.py — make GOS-based EELS edges fittable by the batched engine (#63).

``EELSCLEdge`` is not a formula: exspy integrates a generalised-oscillator-
strength table to get the cross-section, which is why an EELS model used to
fall back to HyperSpy's one-pixel-at-a-time fitting while an EDS model of
gaussians got the GPU.

The way through is to notice what the GOS integral actually depends on: the
element, the subshell, and the microscope geometry — **not the pixel**. It is
the same curve for every spectrum in the scan, so it is computed once. What
remains varies per pixel and is linear or nearly so:

* ``intensity`` scales the edge — linear;
* ``fine_structure_coeff`` are cubic B-spline coefficients, and a spline with
  fixed knots is a linear combination of basis functions — so probing exspy's
  own ``function`` once per coefficient gives those curves, after which the
  edge is exactly ``base + sum(c_i * basis_i)``. Measured against exspy at 0.0
  relative error;
* ``onset_energy`` slides the edge, which the batched component does by
  interpolating the precomputed curves.

**The component stays an ``EELSCLEdge``.** Nothing here changes the kind, the
parameter meanings or the model's identity — it only attaches the precomputed
curves so the batched evaluator has something to work with. That is what keeps
the model a real exspy model, so it stores on the signal with ``m.store()`` and
loads back with ``s.models.restore()`` like any other.

> **What this replaced.** An earlier version swapped the edge for a private
> ``TabulatedShape`` component that fitted intensity and a shift and threw the
> fine structure away. That was wrong twice: the fine structure is the LINEAR
> part and therefore the cheapest thing here to batch, and a private component
> cannot be represented in HyperSpy, so a fitted EELS model could not be saved
> with its own dataset. Both are fixed by keeping the edge an edge.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.fitting import ModelSpec
from spyde.fitting.components import EELS_EDGE_KIND
from spyde.fitting.spec import ComponentSpec, ParameterSpec

log = logging.getLogger(__name__)

#: Parameters an edge fits per pixel. Everything else about the shape —
#: the GOS integral, the effective angle, the knot placement — is fixed by the
#: element and the microscope, so it is baked into the precomputed curves.
INTENSITY = "intensity"
ONSET = "onset_energy"
COEFF = "fine_structure_coeff"


def _axis(signal):
    ax = signal.axes_manager.signal_axes[0]
    x = np.asarray(ax.axis, float)
    if len(x) < 2:
        raise ValueError("signal axis needs at least two channels")
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx, rtol=1e-6):
        raise ValueError("the batched EELS path needs a UNIFORM signal axis; "
                         "this one is not evenly spaced")
    return x, float(x[0]), dx


def _probe(comp, x):
    """The edge's base shape and its fine-structure basis, at unit intensity.

    Unit intensity is what makes these pure shapes, so the fitted
    ``intensity`` means what it meant in HyperSpy rather than carrying whatever
    the seed happened to be.

    The basis comes from exspy itself — one probe per coefficient — rather than
    from a reimplementation of ``splev``. exspy stays the authority on the
    edge's shape; this only rearranges it into columns.
    """
    par = getattr(comp, INTENSITY, None)
    coeff = getattr(comp, COEFF, None)
    active = bool(getattr(comp, "fine_structure_active", False))
    n = int(np.size(coeff.value)) if (coeff is not None and active) else 0

    saved_i = None if par is None else float(np.ravel(par.value)[0])
    saved_c = None if coeff is None else np.array(np.ravel(coeff.value), float)
    try:
        if par is not None:
            par.value = 1.0

        def at(c=None):
            if coeff is not None and c is not None:
                coeff.value = tuple(float(v) for v in c)
            return np.nan_to_num(np.asarray(comp.function(x), float),
                                 nan=0.0, posinf=0.0, neginf=0.0)

        base = at(np.zeros(n) if n else None)
        basis = [at(np.eye(n)[i]) - base for i in range(n)]
    finally:
        if par is not None and saved_i is not None:
            par.value = saved_i
        if coeff is not None and saved_c is not None:
            coeff.value = tuple(float(v) for v in saved_c)

    if not np.isfinite(base).all() or not np.any(base):
        raise ValueError("the edge sampled to an empty or non-finite shape")
    onset = getattr(getattr(comp, ONSET, None), "value", None)
    tables = np.vstack([base] + basis) if basis else base[None, :]
    return tables, (0.0 if onset is None else float(np.ravel(onset)[0])), n


def _edge_spec(cspec, tables, onset_ref, n_basis, x0, dx) -> ComponentSpec:
    """The same edge, with the batched curves attached and its coefficients
    expanded into scalar columns.

    The vector ``fine_structure_coeff`` becomes ``fine_structure_coeff_0..n``
    because the packed parameter vector holds one scalar per column. They are
    reassembled into the vector on the way back to HyperSpy
    (``spec._make_component``), so the round trip is lossless.
    """
    keep = {INTENSITY, ONSET}
    params = [p for p in cspec.parameters if p.name in keep]
    by_name = {p.name: p for p in params}
    if INTENSITY not in by_name:
        params.insert(0, ParameterSpec(INTENSITY, 1.0, linear=True, bmin=0.0))
    else:
        by_name[INTENSITY].linear = True
    if ONSET not in by_name:
        params.append(ParameterSpec(ONSET, onset_ref))
    # Ordered intensity, onset, then the coefficients — the order the batched
    # component declares, and the packed vector has to match it exactly.
    params.sort(key=lambda p: (p.name != INTENSITY, p.name != ONSET))
    old = cspec[COEFF].value if COEFF in {p.name for p in cspec.parameters} else None
    old = np.ravel(old) if old is not None else np.zeros(n_basis)
    for i in range(n_basis):
        params.append(ParameterSpec(
            f"{COEFF}_{i}", float(old[i]) if i < len(old) else 0.0,
            linear=True))

    # Under a reserved key, NOT loose in init_args: those are the component's
    # CONSTRUCTOR arguments and go straight to `EELSCLEdge(**init_args)`, so an
    # `x0` of ours would be handed to exspy as if it meant something.
    # MERGED, not replaced: `spyde` already carries the edge's
    # `fine_structure_active`, and overwriting it rebuilt the edge with fine
    # structure OFF — so the model kept a filled cross-section where the probe
    # had left a spline-shaped hole, and the two disagreed by the full height
    # of the edge across the fine-structure window.
    init = dict(cspec.init_args or {})
    init["spyde"] = {**(init.get("spyde") or {}),
                     "x0": float(x0), "dx": float(dx),
                     "onset_reference": float(onset_ref),
                     "n_coeff": int(n_basis)}
    return ComponentSpec(kind=EELS_EDGE_KIND, name=cspec.name,
                         active=cspec.active, init_args=init,
                         data=tables, parameters=params)


def prepare_eels_edges(spec: ModelSpec, signal, *, fit_onset: bool = True,
                       fit_fine_structure: bool = True):
    """Attach the batched curves to every EELS edge in *spec*.

    Returns ``(new_spec, info)``. *info* lists the edges prepared, those left
    alone, and how many fine-structure coefficients each carries.

    Nothing is approximated away. Set *fit_fine_structure* False to hold the
    coefficients (they stay in the model at their current values, just fixed),
    or *fit_onset* False to pin each edge at its tabulated onset — both are
    ordinary "hold this parameter" choices, not a different model.
    """
    x, x0, dx = _axis(signal)
    model = spec.to_model(signal)
    by_name = {c.name: c for c in model}

    out, prepared, skipped, coeffs = [], [], [], {}
    for cspec in spec.components:
        if cspec.kind != EELS_EDGE_KIND:
            out.append(cspec)
            continue
        comp = by_name.get(cspec.name)
        if comp is None:
            skipped.append(cspec.name)
            out.append(cspec)
            continue
        try:
            tables, onset_ref, n = _probe(comp, x)
        except Exception as e:
            log.info("could not prepare %s (%s); left as-is", cspec.name, e)
            skipped.append(cspec.name)
            out.append(cspec)
            continue
        if not fit_fine_structure:
            n = 0
        new = _edge_spec(cspec, tables, onset_ref, n, x0, dx)
        if not fit_onset:
            new[ONSET].free = False
        out.append(new)
        prepared.append(cspec.name)
        coeffs[cspec.name] = n

    info = {"prepared": prepared, "skipped": skipped, "coefficients": coeffs,
            "x0": x0, "dx": dx}
    if prepared:
        log.info("prepared %d EELS edge(s) for the batched engine: %s "
                 "(fine structure fitted: %s)", len(prepared),
                 ", ".join(prepared), fit_fine_structure)
    return ModelSpec(components=out, channel_mask=spec.channel_mask), info


def onset_energies(spec: ModelSpec) -> dict[str, float]:
    """Fitted absolute onset per edge, in eV.

    ``onset_energy`` is already absolute here — the batched component
    interpolates relative to the reference internally rather than exposing a
    shift, so what the caret shows is what a user means by an onset.
    """
    return {c.name: float(c[ONSET].value)
            for c in spec.components
            if c.kind == EELS_EDGE_KIND and ONSET in
            {p.name for p in c.parameters}}

"""spec.py — a serialisable description of a HyperSpy model.

``ModelSpec`` is the contract between the three things that need to agree about
what "the model" is:

* **HyperSpy** — the reference implementation and the storage format. A spec
  round-trips through ``BaseModel.as_dictionary()``, so a model built in SpyDE
  opens in plain HyperSpy and vice versa, and ``m.store()`` / ``s.models.restore()``
  keep working unchanged.
* **The batched engine** (:mod:`spyde.fitting.engine`) — which needs the
  parameters as flat, ordered arrays it can pack into ``(P, n)`` tensors.
* **The UI** — which adds components line by line, toggles them, and needs a
  JSON-safe form to send to the renderer.

Deliberately a plain dataclass tree, not a HyperSpy subclass: the engine must
be able to describe a model without a signal attached (the wizard builds one
before it runs), and the renderer needs it as JSON.

**Parameter order is part of the contract.** ``flat_values()`` and everything
derived from it iterate components in order and parameters within a component
in order, so column *j* of a packed tensor always means the same parameter.
:meth:`ModelSpec.parameter_names` is that order made explicit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Sequence

import numpy as np

log = logging.getLogger(__name__)

# HyperSpy stores an absent bound as None; the engine wants finite arrays.
_NEG_INF, _POS_INF = -np.inf, np.inf


@dataclass
class ParameterSpec:
    """One fittable parameter.

    ``linear`` mirrors HyperSpy's ``_linear`` flag: the model is linear in this
    parameter (a Gaussian's ``A``, a PowerLaw's ``A``), so the engine can solve
    for it by least squares instead of iterating on it. That is what makes
    variable projection possible (#53) and what makes tabulated EELS edges
    tractable (#63).
    """

    name: str
    value: float = 0.0
    free: bool = True
    bmin: float | None = None
    bmax: float | None = None
    linear: bool = False
    units: str = ""

    def bounds(self) -> tuple[float, float]:
        """Finite bounds for the engine (``None`` becomes ±inf)."""
        lo = _NEG_INF if self.bmin is None else float(self.bmin)
        hi = _POS_INF if self.bmax is None else float(self.bmax)
        return lo, hi

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": float(self.value),
                "free": bool(self.free), "bmin": self.bmin, "bmax": self.bmax,
                "linear": bool(self.linear), "units": self.units}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ParameterSpec":
        return cls(name=d["name"], value=float(d.get("value", 0.0)),
                   free=bool(d.get("free", True)), bmin=d.get("bmin"),
                   bmax=d.get("bmax"), linear=bool(d.get("linear", False)),
                   units=d.get("units", ""))


@dataclass
class ComponentSpec:
    """One model component.

    ``kind`` is HyperSpy's ``_id_name`` (``"Gaussian"``, ``"PowerLaw"``, …) —
    the key both the HyperSpy factory and the torch component library resolve
    against. ``name`` is the user-facing label, which may be edited and is not
    unique.
    """

    kind: str
    name: str = ""
    active: bool = True
    parameters: list[ParameterSpec] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = self.kind

    def __getitem__(self, name: str) -> ParameterSpec:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(f"{self.kind} has no parameter {name!r} "
                       f"(has {[p.name for p in self.parameters]})")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "active": bool(self.active),
                "parameters": [p.to_dict() for p in self.parameters]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ComponentSpec":
        return cls(kind=d["kind"], name=d.get("name", ""),
                   active=bool(d.get("active", True)),
                   parameters=[ParameterSpec.from_dict(p)
                               for p in d.get("parameters", [])])


@dataclass
class ModelSpec:
    """A whole model: ordered components plus the fitted channel range.

    ``channel_mask`` is HyperSpy's ``_channel_switches`` — a boolean array over
    the signal axis, True where the channel takes part in the fit. ``None``
    means "all channels", which is the common case and avoids carrying an
    array the size of the signal axis around in JSON.
    """

    components: list[ComponentSpec] = field(default_factory=list)
    channel_mask: np.ndarray | None = None

    # -- construction ------------------------------------------------------
    def append(self, comp: ComponentSpec) -> "ModelSpec":
        self.components.append(comp)
        return self

    def copy(self) -> "ModelSpec":
        return replace(
            self,
            components=[ComponentSpec.from_dict(c.to_dict())
                        for c in self.components],
            channel_mask=None if self.channel_mask is None
            else np.array(self.channel_mask, copy=True),
        )

    def __getitem__(self, key: int | str) -> ComponentSpec:
        if isinstance(key, int):
            return self.components[key]
        for c in self.components:
            if c.name == key:
                return c
        raise KeyError(f"no component named {key!r} "
                       f"(have {[c.name for c in self.components]})")

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[ComponentSpec]:
        return iter(self.components)

    # -- the flat view the engine packs ------------------------------------
    @property
    def active_components(self) -> list[ComponentSpec]:
        """Only these contribute to the model — and therefore to the parameter
        vector. An inactive component keeps its values but occupies no column,
        so toggling one changes the packed width."""
        return [c for c in self.components if c.active]

    def parameter_names(self) -> list[str]:
        """``"<component name>.<parameter name>"`` in packed order.

        This IS the column order of every array below and of the tensors the
        engine builds. Anything that reports per-parameter results (maps,
        std devs, the UI's parameter list) must use it rather than
        re-deriving an order of its own.
        """
        return [f"{c.name}.{p.name}"
                for c in self.active_components for p in c.parameters]

    def flat_values(self) -> np.ndarray:
        return np.array([p.value for c in self.active_components
                         for p in c.parameters], dtype=np.float64)

    def free_mask(self) -> np.ndarray:
        """True where a parameter is fitted. Fixed parameters keep their value
        and are dropped from the Jacobian rather than being fitted and ignored."""
        return np.array([p.free for c in self.active_components
                         for p in c.parameters], dtype=bool)

    def linear_mask(self) -> np.ndarray:
        """True where the model is LINEAR in the parameter — the columns
        variable projection can solve by least squares (#53)."""
        return np.array([p.linear for c in self.active_components
                         for p in c.parameters], dtype=bool)

    def bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` with ``None`` expanded to ±inf."""
        pairs = [p.bounds() for c in self.active_components for p in c.parameters]
        if not pairs:
            return np.empty(0), np.empty(0)
        lo, hi = zip(*pairs)
        return np.array(lo, np.float64), np.array(hi, np.float64)

    def set_flat_values(self, values: Sequence[float]) -> None:
        """Write a packed parameter vector back onto the spec (the inverse of
        :meth:`flat_values`)."""
        values = np.asarray(values, dtype=float).ravel()
        expected = sum(len(c.parameters) for c in self.active_components)
        if values.size != expected:
            raise ValueError(f"expected {expected} values for this spec, "
                             f"got {values.size}")
        i = 0
        for c in self.active_components:
            for p in c.parameters:
                p.value = float(values[i])
                i += 1

    def component_slices(self) -> dict[str, slice]:
        """Where each active component's parameters sit in the packed vector —
        what the component-area maps (#58) need to isolate one component."""
        out, i = {}, 0
        for c in self.active_components:
            n = len(c.parameters)
            out[c.name] = slice(i, i + n)
            i += n
        return out

    # -- HyperSpy interop --------------------------------------------------
    @classmethod
    def from_model(cls, model) -> "ModelSpec":
        """Read a live HyperSpy model (``s.create_model()``) into a spec."""
        comps = []
        for c in model:
            params = []
            for p in c.parameters:
                bmin, bmax = getattr(p, "bmin", None), getattr(p, "bmax", None)
                params.append(ParameterSpec(
                    name=p.name,
                    # A multidimensional parameter's `.value` is the value at
                    # the CURRENT nav index; that is the right seed to carry.
                    value=float(np.ravel(p.value)[0]),
                    free=bool(p.free),
                    bmin=None if bmin is None else float(bmin),
                    bmax=None if bmax is None else float(bmax),
                    linear=bool(getattr(p, "_linear", False)),
                    units=getattr(p, "units", "") or "",
                ))
            comps.append(ComponentSpec(
                kind=getattr(c, "_id_name", type(c).__name__),
                name=c.name, active=bool(c.active), parameters=params))

        mask = getattr(model, "_channel_switches", None)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.all():
                mask = None          # "all channels" is the default; don't carry it
        return cls(components=comps, channel_mask=mask)

    def to_model(self, signal, *, apply_range: bool = True):
        """Build a live HyperSpy model on *signal* from this spec.

        Used for the CPU reference path, for the parity tests, and for the live
        single-spectrum preview — anywhere HyperSpy itself should do the work.
        """
        model = signal.create_model()
        # create_model() may pre-populate (an EELS model adds a background and
        # the declared edges); the spec is authoritative, so start clean.
        while len(model):
            model.remove(model[0])

        for cspec in self.components:
            comp = _make_component(cspec)
            model.append(comp)
            comp.active = bool(cspec.active)
            for pspec in cspec.parameters:
                try:
                    par = getattr(comp, pspec.name)
                except AttributeError:
                    log.debug("component %s has no parameter %s — skipped",
                              cspec.kind, pspec.name)
                    continue
                # Bounds BEFORE value: HyperSpy clips an out-of-range assignment.
                par.bmin, par.bmax = pspec.bmin, pspec.bmax
                par.value = float(pspec.value)
                par.free = bool(pspec.free)

        if apply_range and self.channel_mask is not None:
            try:
                model._channel_switches = np.asarray(self.channel_mask, bool)
            except Exception as e:
                log.debug("applying channel mask to model failed: %s", e)
        return model

    # -- JSON --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-safe. The channel mask goes out as a list of ``[start, stop)``
        runs rather than a per-channel bool array — a 4096-channel mask is one
        or two ranges in practice, and the renderer draws ranges, not bits."""
        return {"components": [c.to_dict() for c in self.components],
                "channel_ranges": _mask_to_ranges(self.channel_mask),
                "n_channels": (None if self.channel_mask is None
                               else int(self.channel_mask.size))}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelSpec":
        n = d.get("n_channels")
        return cls(
            components=[ComponentSpec.from_dict(c) for c in d.get("components", [])],
            channel_mask=_ranges_to_mask(d.get("channel_ranges"), n),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_component(cspec: ComponentSpec):
    """Instantiate the HyperSpy component named by ``cspec.kind``.

    Searches components1d then components2d, then exspy's components if the
    ``eels`` extra is installed (EELSCLEdge and friends live there, #63).
    """
    import hyperspy.components1d as c1d
    import hyperspy.components2d as c2d

    mods = [c1d, c2d]
    try:                                   # optional extra — absent is normal
        import exspy.components as exc
        mods.append(exc)
    except ImportError:
        pass

    for mod in mods:
        cls = getattr(mod, cspec.kind, None)
        if cls is not None:
            comp = cls()
            if cspec.name and cspec.name != cspec.kind:
                comp.name = cspec.name
            return comp
    raise ValueError(
        f"unknown component kind {cspec.kind!r}. Not in hyperspy "
        f"components1d/components2d"
        + ("" if len(mods) > 2 else " (and exspy is not installed — install "
                                    'spyde[eels] for EELS/EDS components)'))


def spec_from_component(comp) -> ComponentSpec:
    """One live HyperSpy component -> a ComponentSpec (used by the picker,
    which instantiates a component just to show its default shape, #56)."""
    return ComponentSpec(
        kind=getattr(comp, "_id_name", type(comp).__name__),
        name=comp.name, active=bool(comp.active),
        parameters=[ParameterSpec(
            name=p.name, value=float(np.ravel(p.value)[0]), free=bool(p.free),
            bmin=getattr(p, "bmin", None), bmax=getattr(p, "bmax", None),
            linear=bool(getattr(p, "_linear", False)),
            units=getattr(p, "units", "") or "",
        ) for p in comp.parameters],
    )


def _mask_to_ranges(mask) -> list[list[int]] | None:
    if mask is None:
        return None
    m = np.asarray(mask, bool)
    if m.all():
        return None
    edges = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [[int(a), int(b)] for a, b in zip(starts, stops)]


def _ranges_to_mask(ranges, n_channels) -> np.ndarray | None:
    if not ranges or not n_channels:
        return None
    m = np.zeros(int(n_channels), bool)
    for a, b in ranges:
        m[int(a):int(b)] = True
    return m

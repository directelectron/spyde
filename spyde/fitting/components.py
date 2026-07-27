"""components.py — batched, differentiable torch ports of HyperSpy's components.

Each component evaluates **every navigation position at once**: given the shared
signal axis ``x`` of length ``C`` and a parameter block ``p`` of shape
``(P, n)``, it returns ``(P, C)``. There is no Python loop over pixels anywhere
in this module — that is the entire point (see :mod:`spyde.fitting.engine`).

Three rules, each of which is a parity bug if broken:

* **Parameter order is HyperSpy's order**, i.e. the order of
  ``component.parameters``. It is *not* alphabetical and *not* the order of the
  constructor arguments: ``PowerLaw`` is ``(A, left_cutoff, origin, r)`` and
  ``GaussianHF`` is ``(centre, fwhm, height)``. ``ModelSpec`` packs columns in
  this order, so a mismatch silently fits the wrong parameter.
* **The formula is HyperSpy's formula.** These are transcribed from the
  ``expression=`` strings in ``hyperspy/_components/``, not from memory — e.g.
  a HyperSpy ``Gaussian``'s ``A`` is the AREA, not the peak height.
  ``test_torch_components.py`` checks every component against the HyperSpy one
  numerically; that test is the real specification.
* **Everything must stay differentiable.** No ``.item()``, no in-place writes
  into a tensor that requires grad, and no branch that produces NaN on the dead
  side of a ``where`` — a NaN there poisons the gradient even where the mask
  discards the value.

torch is a core dependency, but it is imported lazily so that importing
``spyde.fitting`` (for ``ModelSpec``) never pays for it.
"""
from __future__ import annotations

import logging
import math
from typing import Callable, Sequence

log = logging.getLogger(__name__)

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)
_FOUR_LOG2 = 4.0 * math.log(2.0)

# Guards a division by a parameter the optimiser can walk to zero.
_EPS = 1e-30


class TorchComponent:
    """One batched component.

    ``params`` mirrors HyperSpy's parameter order; ``linear`` marks the columns
    the model is linear in (what variable projection solves directly).
    """

    __slots__ = ("kind", "params", "linear", "_fn")

    def __init__(self, kind: str, params: Sequence[str], linear: Sequence[bool],
                 fn: Callable):
        if len(params) != len(linear):
            raise ValueError(f"{kind}: {len(params)} params vs "
                             f"{len(linear)} linear flags")
        self.kind = kind
        self.params = tuple(params)
        self.linear = tuple(bool(b) for b in linear)
        self._fn = fn

    @property
    def n_params(self) -> int:
        return len(self.params)

    def __call__(self, x, p):
        """``x``: ``(C,)`` or ``(P, C)``. ``p``: ``(P, n_params)``. -> ``(P, C)``."""
        if p.shape[-1] != self.n_params:
            raise ValueError(f"{self.kind} expects {self.n_params} parameters "
                             f"{self.params}, got {p.shape[-1]}")
        if x.dim() == 1:
            x = x.unsqueeze(0)                      # (1, C), broadcasts over P
        return self._fn(x, [p[:, i:i + 1] for i in range(self.n_params)])

    def __repr__(self) -> str:                       # pragma: no cover
        return f"<TorchComponent {self.kind}{self.params}>"


# ---------------------------------------------------------------------------
# the formulas — transcribed from hyperspy/_components/*.py `expression=`
# ---------------------------------------------------------------------------

def _gaussian(x, p):
    # "A * (1 / (sigma * sqrt(2*pi))) * exp(-(x - centre)**2 / (2 * sigma**2))"
    # NB A is the AREA under the curve, not the height.
    A, centre, sigma = p
    s = sigma + _EPS
    return A / (s * _SQRT_2PI) * (-((x - centre) ** 2) / (2 * s ** 2)).exp()


def _gaussian_hf(x, p):
    # "height * exp(-(x - centre)**2 * 4 * log(2)/fwhm**2)"
    centre, fwhm, height = p
    w = fwhm + _EPS
    return height * (-((x - centre) ** 2) * _FOUR_LOG2 / w ** 2).exp()


def _lorentzian(x, p):
    # "A / pi * (gamma_ / ((x - centre)**2 + gamma_**2))"  (gamma_ renames gamma)
    A, centre, gamma = p
    return A / math.pi * (gamma / ((x - centre) ** 2 + gamma ** 2 + _EPS))


def _power_law(x, p):
    # "where(left_cutoff < x, A*(-origin + x)**-r, 0)"
    A, left_cutoff, origin, r = p
    import torch
    base = x - origin
    live = (left_cutoff < x) & (base > 0)
    # Clamp the base BEFORE the power: (<=0) ** -r is inf/NaN, and a NaN in the
    # discarded branch of a where() still propagates NaN through the gradient.
    safe = torch.where(live, base, torch.ones_like(base))
    return torch.where(live, A * safe ** (-r), torch.zeros_like(base))


def _offset(x, p):
    (offset,) = p
    return offset.expand(-1, x.shape[-1]) if x.shape[0] == 1 else \
        offset + 0.0 * x


def _exponential(x, p):
    # "A * exp(-x / tau)"
    A, tau = p
    return A * (-x / (tau + _EPS)).exp()


def _arctan(x, p):
    # "A * atan(k * (x - x0))"
    A, k, x0 = p
    return A * (k * (x - x0)).atan()


def _erf(x, p):
    # "A * erf((x - origin) / sqrt(2) / sigma) / 2"
    A, origin, sigma = p
    import torch
    return A * torch.erf((x - origin) / _SQRT_2 / (sigma + _EPS)) / 2.0


def _heaviside(x, p):
    # HeavisideStep: A below/above the step at n (0.5 exactly at the step).
    A, n = p
    import torch
    return A * torch.where(x > n, torch.ones_like(x),
                           torch.where(x < n, torch.zeros_like(x),
                                       torch.full_like(x, 0.5)))


def _logistic(x, p):
    # "a / (1 + b * exp(-c * (x - origin)))"
    a, b, c, origin = p
    return a / (1.0 + b * (-c * (x - origin)).exp())


def _polynomial_fn(order: int):
    """Polynomial is variable-order, so its evaluator is built per order.
    Parameters are ``a0..a{order}`` and ``aK`` multiplies ``x**K``."""

    def fn(x, p):
        out = p[0].expand_as(x) if len(p) else None
        acc = out
        for k in range(1, order + 1):
            acc = acc + p[k] * x ** k
        return acc

    return fn


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, TorchComponent] = {}


def _register(kind, params, linear, fn) -> None:
    _REGISTRY[kind] = TorchComponent(kind, params, linear, fn)


# Parameter tuples are HyperSpy's `component.parameters` ORDER — verified by
# test_torch_components.py::TestParameterOrder against live components.
_register("Gaussian",      ("A", "centre", "sigma"),        (True, False, False),  _gaussian)
_register("GaussianHF",    ("centre", "fwhm", "height"),    (False, False, True),  _gaussian_hf)
_register("Lorentzian",    ("A", "centre", "gamma"),        (True, False, False),  _lorentzian)
_register("PowerLaw",      ("A", "left_cutoff", "origin", "r"),
          (True, False, False, False), _power_law)
_register("Offset",        ("offset",),                     (True,),               _offset)
_register("Exponential",   ("A", "tau"),                    (True, False),         _exponential)
_register("Arctan",        ("A", "k", "x0"),                (True, False, False),  _arctan)
_register("Erf",           ("A", "origin", "sigma"),        (True, False, False),  _erf)
_register("HeavisideStep", ("A", "n"),                      (True, False),         _heaviside)
_register("Logistic",      ("a", "b", "c", "origin"),       (True, False, False, False), _logistic)


def get_component(kind: str, *, n_params: int | None = None) -> TorchComponent:
    """Look up a batched component by HyperSpy ``_id_name``.

    ``Polynomial`` is variable-order, so its evaluator is built on demand from
    the parameter count the spec carries.
    """
    if kind == "Polynomial":
        if n_params is None:
            raise ValueError("Polynomial needs n_params (order + 1)")
        order = int(n_params) - 1
        return TorchComponent("Polynomial",
                              tuple(f"a{k}" for k in range(order + 1)),
                              (True,) * (order + 1),
                              _polynomial_fn(order))
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise NotImplementedError(
            f"no batched torch implementation for component {kind!r}. "
            f"Available: {sorted(_REGISTRY) + ['Polynomial']}. "
            f"Fall back to the CPU/HyperSpy path, or add it here with a parity "
            f"test (#52)."
        ) from None


def available() -> list[str]:
    """Component kinds the batched engine can fit."""
    return sorted(_REGISTRY) + ["Polynomial"]


def supports(spec) -> bool:
    """True when every ACTIVE component of a ModelSpec has a batched port.

    The engine calls this to decide whether it can run at all; anything
    unsupported falls back to HyperSpy's own fitting rather than silently
    dropping a component from the model.
    """
    for c in getattr(spec, "active_components", []):
        try:
            get_component(c.kind, n_params=len(c.parameters))
        except NotImplementedError:
            return False
    return True


def evaluate(spec, x, values):
    """Evaluate a whole :class:`~spyde.fitting.spec.ModelSpec`.

    ``x``: ``(C,)``; ``values``: ``(P, n_total)`` packed in the spec's order.
    Returns ``(P, C)`` — the sum over active components.
    """
    import torch

    out = None
    i = 0
    for c in spec.active_components:
        n = len(c.parameters)
        comp = get_component(c.kind, n_params=n)
        y = comp(x, values[:, i:i + n])
        out = y if out is None else out + y
        i += n
    if out is None:
        p = values.shape[0] if values.dim() > 1 else 1
        return torch.zeros((p, x.shape[-1]), dtype=x.dtype, device=x.device)
    return out

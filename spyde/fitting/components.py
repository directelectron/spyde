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

import numpy as np

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

    ``grad``, when present, returns the analytic derivative stack
    ``(P, C, n_params)``. **This is a performance decision with a measured
    basis, not a micro-optimisation.** Autodiff on these components costs one
    forward pass per parameter — measured at 51.6 ms against 3.4 ms for a
    single residual evaluation, i.e. ~94% of the whole solver's time. The
    derivatives here are closed forms that mostly reuse the value already
    computed (``df/dA = f/A`` for every linear amplitude), so the analytic
    stack costs about as much as one residual.

    A component without ``grad`` still works — the engine falls back to
    ``jacfwd``. Correctness is not at stake either way, only speed, and
    ``test_torch_components.py`` checks every analytic gradient against
    autodiff, which is the ideal oracle for exactly this.
    """

    __slots__ = ("kind", "params", "linear", "_fn", "_grad", "ndim")

    def __init__(self, kind: str, params: Sequence[str], linear: Sequence[bool],
                 fn: Callable, grad: Callable | None = None, ndim: int = 1):
        if len(params) != len(linear):
            raise ValueError(f"{kind}: {len(params)} params vs "
                             f"{len(linear)} linear flags")
        self.kind = kind
        self.params = tuple(params)
        self.linear = tuple(bool(b) for b in linear)
        self._fn = fn
        self._grad = grad
        # 1 for a spectrum (x is the signal axis), 2 for an image (x carries
        # (x, y) coordinate PAIRS). A 2-D model is otherwise identical: the
        # image is flattened to C sample points and everything downstream —
        # packing, the Jacobian, the LM solve — is unchanged.
        self.ndim = int(ndim)

    @property
    def n_params(self) -> int:
        return len(self.params)

    @property
    def has_analytic_grad(self) -> bool:
        return self._grad is not None

    def _split(self, x, p):
        if p.shape[-1] != self.n_params:
            raise ValueError(f"{self.kind} expects {self.n_params} parameters "
                             f"{self.params}, got {p.shape[-1]}")
        cols = [p[:, i:i + 1] for i in range(self.n_params)]
        if self.ndim == 2:
            # x is (C, 2) or (P, C, 2): coordinate PAIRS, not a single axis.
            if x.dim() == 2:
                x = x.unsqueeze(0)                  # (1, C, 2)
            return (x[..., 0], x[..., 1]), cols
        if x.dim() == 1:
            x = x.unsqueeze(0)                      # (1, C), broadcasts over P
        return x, cols

    def __call__(self, x, p):
        """``x``: ``(C,)`` or ``(P, C)``. ``p``: ``(P, n_params)``. -> ``(P, C)``."""
        x, cols = self._split(x, p)
        return self._fn(x, cols)

    def grad(self, x, p):
        """Analytic ``d(value)/d(parameter)`` -> ``(P, C, n_params)``.

        Raises if this component has no analytic form; callers should check
        :attr:`has_analytic_grad` (or use :func:`evaluate_with_grad`).
        """
        if self._grad is None:
            raise NotImplementedError(
                f"{self.kind} has no analytic gradient; the engine falls back "
                f"to autodiff for it")
        x, cols = self._split(x, p)
        return self._grad(x, cols)

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
    # Broadcast rather than expand — see _polynomial_fn for why expand-based
    # shaping breaks as soon as there is more than one spectrum.
    return offset + 0.0 * x


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


def _gaussian_2d(xy, p):
    # "A * (1 / (sigma_x * sigma_y * 2 * pi))
    #  * exp(-((x - centre_x)**2 / (2*sigma_x**2)
    #        + (y - centre_y)**2 / (2*sigma_y**2)))"
    # As in 1-D, A is the VOLUME under the surface, not the peak height.
    x, y = xy
    A, cx, cy, sx, sy = p
    sxe, sye = sx + _EPS, sy + _EPS
    return A / (sxe * sye * 2.0 * math.pi) * (
        -((x - cx) ** 2 / (2 * sxe ** 2) + (y - cy) ** 2 / (2 * sye ** 2))).exp()


def _d_gaussian_2d(xy, p):
    x, y = xy
    A, cx, cy, sx, sy = p
    sxe, sye = sx + _EPS, sy + _EPS
    dx, dy = x - cx, y - cy
    shape = (-(dx ** 2 / (2 * sxe ** 2) + dy ** 2 / (2 * sye ** 2))).exp() \
        / (sxe * sye * 2.0 * math.pi)
    f = A * shape
    return _stack(shape,                                   # df/dA
                  f * dx / sxe ** 2,                       # df/dcentre_x
                  f * dy / sye ** 2,                       # df/dcentre_y
                  f * (dx ** 2 / sxe ** 3 - 1.0 / sxe),    # df/dsigma_x
                  f * (dy ** 2 / sye ** 3 - 1.0 / sye))    # df/dsigma_y


def image_coordinates(shape, *, device=None, dtype=None):
    """``(H*W, 2)`` of ``(x, y)`` sample points for a 2-D model.

    This is the "signal axis" a 2-D fit uses: the image is flattened to H*W
    sample points, exactly as a spectrum is C channels, so nothing downstream
    (packing, the Jacobian, the LM solve) needs to know it came from an image.
    """
    import torch
    h, w = int(shape[0]), int(shape[1])
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype),
                            indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


# ---------------------------------------------------------------------------
# analytic derivatives — checked against autodiff in test_torch_components.py
#
# Every one of these reuses the value where it can: for a linear amplitude the
# derivative IS the shape (df/dA = f/A), which is already the expensive part.
# ---------------------------------------------------------------------------

def _stack(*cols):
    import torch
    return torch.stack(torch.broadcast_tensors(*cols), dim=-1)


def _d_gaussian(x, p):
    A, centre, sigma = p
    s = sigma + _EPS
    shape = (-((x - centre) ** 2) / (2 * s ** 2)).exp() / (s * _SQRT_2PI)
    f = A * shape
    d = x - centre
    return _stack(shape,                      # df/dA
                  f * d / s ** 2,              # df/dcentre
                  f * (d ** 2 / s ** 3 - 1.0 / s))   # df/dsigma


def _d_gaussian_hf(x, p):
    centre, fwhm, height = p
    w = fwhm + _EPS
    d = x - centre
    shape = (-(d ** 2) * _FOUR_LOG2 / w ** 2).exp()
    f = height * shape
    return _stack(f * 2.0 * _FOUR_LOG2 * d / w ** 2,      # df/dcentre
                  f * 2.0 * _FOUR_LOG2 * d ** 2 / w ** 3,  # df/dfwhm
                  shape)                                   # df/dheight


def _d_lorentzian(x, p):
    A, centre, gamma = p
    d = x - centre
    D = d ** 2 + gamma ** 2 + _EPS
    shape = gamma / D / math.pi
    return _stack(shape,                                   # df/dA
                  A / math.pi * gamma * 2.0 * d / D ** 2,   # df/dcentre
                  A / math.pi * (d ** 2 - gamma ** 2) / D ** 2)  # df/dgamma


def _d_power_law(x, p):
    import torch
    A, left_cutoff, origin, r = p
    base = x - origin
    live = (left_cutoff < x) & (base > 0)
    safe = torch.where(live, base, torch.ones_like(base))
    shape = torch.where(live, safe ** (-r), torch.zeros_like(base))
    f = A * shape
    zero = torch.zeros_like(base)
    return _stack(shape,                                   # df/dA
                  zero,                                    # df/dleft_cutoff (step)
                  torch.where(live, f * r / safe, zero),   # df/dorigin
                  torch.where(live, -f * safe.log(), zero))  # df/dr


def _d_offset(x, p):
    import torch
    (offset,) = p
    return torch.ones_like(offset + 0.0 * x).unsqueeze(-1)


def _d_exponential(x, p):
    A, tau = p
    t = tau + _EPS
    shape = (-x / t).exp()
    return _stack(shape,                       # df/dA
                  A * shape * x / t ** 2)      # df/dtau


def _d_arctan(x, p):
    A, k, x0 = p
    d = x - x0
    u = k * d
    D = 1.0 + u ** 2
    return _stack(u.atan(),                    # df/dA
                  A * d / D,                   # df/dk
                  -A * k / D)                  # df/dx0


def _d_erf(x, p):
    import torch
    A, origin, sigma = p
    s = sigma + _EPS
    u = (x - origin) / _SQRT_2 / s
    # d/du erf(u) = 2/sqrt(pi) * exp(-u^2)
    dedu = (2.0 / math.sqrt(math.pi)) * (-(u ** 2)).exp()
    half_A = A / 2.0
    return _stack(torch.erf(u) / 2.0,                       # df/dA
                  half_A * dedu * (-1.0 / (_SQRT_2 * s)),   # df/dorigin
                  half_A * dedu * (-(x - origin) / (_SQRT_2 * s ** 2)))  # df/dsigma


def _d_heaviside(x, p):
    import torch
    A, n = p
    step = torch.where(x > n, torch.ones_like(x),
                       torch.where(x < n, torch.zeros_like(x),
                                   torch.full_like(x, 0.5)))
    # d/dn is a delta function — zero almost everywhere, which is what any
    # gradient-based optimiser can use. `n` is effectively unfittable, exactly
    # as it is in HyperSpy.
    return _stack(step, torch.zeros_like(step))


def _d_logistic(x, p):
    a, b, c, origin = p
    d = x - origin
    e = (-c * d).exp()
    E = b * e
    D = 1.0 + E
    return _stack(1.0 / D,                     # df/da
                  -a * e / D ** 2,             # df/db
                  a * E * d / D ** 2,          # df/dc
                  -a * E * c / D ** 2)         # df/dorigin


def _interp_uniform(xq, x0, dx, table):
    """Linear interpolation of ``table`` (sampled uniformly from ``x0``) at
    ``xq``, differentiable with respect to ``xq``.

    ``torch.searchsorted`` is not needed because the table is on the signal
    axis, which is uniform — index arithmetic is exact and far cheaper. The
    index itself is a floor and carries no gradient, which is correct: the
    function is piecewise linear, so the gradient comes entirely from the
    interpolation weight.

    Outside the table the value is held at the end sample rather than
    extrapolated. Extrapolating a GOS tail would invent signal where the
    measurement has none.
    """
    import torch

    n = table.shape[-1]
    pos = (xq - x0) / dx
    idx = torch.clamp(pos.floor(), 0, n - 2)
    t = torch.clamp(pos - idx, 0.0, 1.0)
    i0 = idx.long()
    y0 = table[i0]
    y1 = table[torch.clamp(i0 + 1, max=n - 1)]
    return y0 + t * (y1 - y0)


def _tabulated_fn(table, x0, dx):
    def fn(x, p):
        intensity, onset_shift = p
        # Shifting the SAMPLE point left is the same as moving the edge right,
        # so a positive onset_shift moves the edge up in energy as a user
        # expects.
        return intensity * _interp_uniform(x - onset_shift, x0, dx, table)

    return fn


def _tabulated_grad(table, x0, dx):
    def grad(x, p):
        intensity, onset_shift = p
        shape = _interp_uniform(x - onset_shift, x0, dx, table)
        # d/d(shift) is minus the local slope. Taken as a CENTRAL difference
        # over one sample rather than the exact piecewise-linear derivative,
        # deliberately: the exact one is a step function that jumps at every
        # segment boundary, so LM chatters as the shift crosses a channel. The
        # smoothed version differs from autodiff only within one channel of a
        # kink and converges to the same place.
        half = dx / 2
        slope = (_interp_uniform(x - onset_shift + half, x0, dx, table)
                 - _interp_uniform(x - onset_shift - half, x0, dx, table)) / dx
        return _stack(shape, -intensity * slope)

    return grad


def _d_polynomial_fn(order: int):
    def grad(x, p):
        return _stack(*[x ** k + 0.0 * p[0] for k in range(order + 1)])

    return grad


def _polynomial_fn(order: int):
    """Polynomial is variable-order, so its evaluator is built per order.
    Parameters are ``a0..a{order}`` and ``aK`` multiplies ``x**K``."""

    def fn(x, p):
        # `p[0] + 0.0 * x`, NOT `p[0].expand_as(x)`. The parameter block is
        # (P, 1) and the axis is (1, C), so expand_as tries to force P down to
        # 1 and raises for any batch bigger than one spectrum — which single-
        # spectrum tests never reach. Plain broadcasting gives (P, C) for both
        # a shared axis and a per-position one.
        acc = p[0] + 0.0 * x
        for k in range(1, order + 1):
            acc = acc + p[k] * x ** k
        return acc

    return fn


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, TorchComponent] = {}


def _register(kind, params, linear, fn, grad=None) -> None:
    _REGISTRY[kind] = TorchComponent(kind, params, linear, fn, grad)


# Parameter tuples are HyperSpy's `component.parameters` ORDER — verified by
# test_torch_components.py::TestParameterOrder against live components.
_register("Gaussian",      ("A", "centre", "sigma"),        (True, False, False),  _gaussian, _d_gaussian)
_register("GaussianHF",    ("centre", "fwhm", "height"),    (False, False, True),  _gaussian_hf, _d_gaussian_hf)
_register("Lorentzian",    ("A", "centre", "gamma"),        (True, False, False),  _lorentzian, _d_lorentzian)
_register("PowerLaw",      ("A", "left_cutoff", "origin", "r"),
          (True, False, False, False), _power_law, _d_power_law)
_register("Offset",        ("offset",),                     (True,),               _offset, _d_offset)
_register("Exponential",   ("A", "tau"),                    (True, False),         _exponential, _d_exponential)
_register("Arctan",        ("A", "k", "x0"),                (True, False, False),  _arctan, _d_arctan)
_register("Erf",           ("A", "origin", "sigma"),        (True, False, False),  _erf, _d_erf)
_register("HeavisideStep", ("A", "n"),                      (True, False),         _heaviside, _d_heaviside)
_register("Logistic",      ("a", "b", "c", "origin"),       (True, False, False, False), _logistic, _d_logistic)
_REGISTRY["Gaussian2D"] = TorchComponent(
    "Gaussian2D", ("A", "centre_x", "centre_y", "sigma_x", "sigma_y"),
    (True, False, False, False, False), _gaussian_2d, _d_gaussian_2d, ndim=2)


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
                              _polynomial_fn(order), _d_polynomial_fn(order))
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise NotImplementedError(
            f"no batched torch implementation for component {kind!r}. "
            f"Available: {sorted(_REGISTRY) + ['Polynomial']}. "
            f"Fall back to the CPU/HyperSpy path, or add it here with a parity "
            f"test (#52)."
        ) from None


EELS_EDGE_KIND = "EELSCLEdge"


def _interp_stack(xq, x0, dx, table):
    """:func:`_interp_uniform` for a STACK of tables — ``(K, C)`` -> ``(K, P, C)``.

    Every row is sampled at the same query points, which is the whole point:
    the base shape and each fine-structure basis function slide together when
    the onset moves, because they are one edge.
    """
    import torch

    n = table.shape[-1]
    pos = (xq - x0) / dx
    idx = torch.clamp(pos.floor(), 0, n - 2)
    t = torch.clamp(pos - idx, 0.0, 1.0)
    i0 = idx.long()
    y0 = table[:, i0]
    y1 = table[:, torch.clamp(i0 + 1, max=n - 1)]
    return y0 + t * (y1 - y0)


def _eels_bracket(tables, x0, dx, onset_ref, x, cols):
    """``shape(E - delta)`` and its fine-structure part, at unit intensity."""
    onset = cols[1]
    delta = onset - onset_ref
    stack = _interp_stack(x - delta, x0, dx, tables)     # (1+N, P, C)
    total = stack[0]
    for i, c in enumerate(cols[2:]):
        total = total + c * stack[i + 1]
    return total, stack


def eels_edge_component(tables, x0: float, dx: float, onset_ref: float, *,
                        device=None, dtype=None) -> TorchComponent:
    """A batched EELS core-loss edge — GOS shape AND fittable fine structure.

    ``EELSCLEdge`` is not a formula: exspy integrates a generalised-oscillator-
    strength table to get the cross-section. That integral is what has no place
    in a batched inner loop — but it depends on the ELEMENT and the MICROSCOPE
    GEOMETRY, not on the pixel, so it is the same curve for every spectrum in
    the scan and is computed once.

    What is left is linear or nearly so, which is why this works:

    * ``intensity`` scales the whole edge — linear.
    * ``fine_structure_coeff`` are cubic B-spline coefficients, and a spline
      with fixed knots is a LINEAR combination of basis functions. Probing
      exspy's own ``function`` once per coefficient gives those basis curves,
      after which the edge is exactly ``base + sum(c_i * basis_i)``. Verified
      against exspy at 0.0 relative error, not approximately.
    * ``onset_energy`` slides the edge. Moving it in exspy re-integrates the
      GOS and re-places the knots; here the precomputed curves are interpolated
      instead, which is exact for the shape and differs only in how the far
      tail is re-derived — well inside the few channels a fit actually moves.

    ``tables`` is ``(1 + N, C)``: the base shape at unit intensity with every
    coefficient zero, then one row per fine-structure basis function.

    This REPLACED a component that discarded the fine structure altogether and
    fitted only intensity and a shift. That was a bad trade twice over: fine
    structure is the *linear* part and so the cheapest thing here to batch, and
    dropping it produced a component HyperSpy could not represent, so a fitted
    EELS model could not be stored on its own signal.
    """
    import torch

    t = torch.as_tensor(np.asarray(tables, float), device=device,
                        dtype=dtype or torch.float64)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    n_basis = int(t.shape[0]) - 1
    params = ("intensity", "onset_energy") + tuple(
        f"fine_structure_coeff_{i}" for i in range(n_basis))
    linear = (True, False) + (True,) * n_basis
    x0, dx, ref = float(x0), float(dx), float(onset_ref)

    def fn(x, cols):
        total, _ = _eels_bracket(t, x0, dx, ref, x, cols)
        return cols[0] * total

    def grad(x, cols):
        intensity = cols[0]
        total, stack = _eels_bracket(t, x0, dx, ref, x, cols)
        # d/d(onset) is minus the local slope, as a CENTRAL difference over one
        # channel — the exact piecewise-linear derivative is a step function
        # that jumps at every segment boundary, so LM chatters as the onset
        # crosses a channel.
        half = dx / 2.0
        shifted = []
        for sign in (+1.0, -1.0):
            moved = list(cols)
            moved[1] = cols[1] - sign * half
            tot, _ = _eels_bracket(t, x0, dx, ref, x, moved)
            shifted.append(tot)
        slope = (shifted[0] - shifted[1]) / dx
        return _stack(total, -intensity * slope,
                      *[intensity * stack[i + 1] for i in range(n_basis)])

    return TorchComponent(EELS_EDGE_KIND, params, linear, fn, grad)


def component_for(cspec, *, device=None, dtype=None) -> TorchComponent:
    """Resolve the batched component for a :class:`ComponentSpec`.

    Everything analytic resolves by ``kind`` alone; an EELS edge also needs its
    own precomputed curves, which live on the spec. Callers that have a spec
    should use this rather than :func:`get_component`, so a data-bound
    component is never silently looked up as if it were stateless.
    """
    if cspec.kind == EELS_EDGE_KIND:
        if cspec.data is None:
            raise ValueError(
                f"{cspec.name}: this EELS edge has not been prepared for the "
                f"batched engine — call spyde.spectroscopy.prepare_eels_edges")
        meta = (cspec.init_args or {}).get("spyde") or {}
        return eels_edge_component(
            cspec.data, float(meta.get("x0", 0.0)), float(meta.get("dx", 1.0)),
            float(meta.get("onset_reference", 0.0)),
            device=device, dtype=dtype)
    return get_component(cspec.kind, n_params=len(cspec.scalar_parameters))


def available() -> list[str]:
    """Component kinds the batched engine can fit."""
    return sorted(_REGISTRY) + ["Polynomial", EELS_EDGE_KIND]


def unsupported(spec) -> dict[str, str]:
    """``{component name: why the batched engine cannot fit it}``.

    Resolved by actually TRYING to build each component, not by checking the
    kind against :func:`available`. The two differ for an EELS edge, which is
    a supported kind but only once its GOS curves have been precomputed — so a
    kind-only check would report a raw edge as fittable and then fail inside
    the engine.
    """
    out = {}
    for c in getattr(spec, "active_components", []):
        try:
            component_for(c)
        except (NotImplementedError, ValueError) as e:
            out[c.name] = str(e)
    return out


def supports(spec) -> bool:
    """True when every ACTIVE component of a ModelSpec has a batched port.

    The engine calls this to decide whether it can run at all; anything
    unsupported falls back to HyperSpy's own fitting rather than silently
    dropping a component from the model.
    """
    return not unsupported(spec)


def has_analytic_grad(spec) -> bool:
    """True when EVERY active component can supply an analytic derivative, so
    the engine can build the whole Jacobian without autodiff."""
    for c in getattr(spec, "active_components", []):
        try:
            if not component_for(c).has_analytic_grad:
                return False
        except (NotImplementedError, ValueError):
            return False
    return True


def evaluate_with_grad(spec, x, values):
    """``(value (P, C), jacobian (P, C, n_total))`` for a whole ModelSpec.

    The model is a SUM of components, so the Jacobian is just each component's
    derivative block written into its own columns — components do not interact,
    which is what makes this cheap.

    Columns follow the spec's packed order, the same order as
    :meth:`~spyde.fitting.spec.ModelSpec.parameter_names`.
    """
    import torch

    # Sample count is the LAST axis for a 1-D axis and the second-to-last for
    # 2-D coordinate pairs — `x` is (C,)/(P, C) or (C, 2)/(P, C, 2).
    C = x.shape[-2] if x.shape[-1] == 2 and x.dim() >= 2 else x.shape[-1]
    P = values.shape[0]
    n_total = values.shape[1]
    out = torch.zeros((P, C), dtype=values.dtype, device=values.device)
    jac = torch.zeros((P, C, n_total), dtype=values.dtype, device=values.device)

    i = 0
    for c in spec.active_components:
        n = len(c.scalar_parameters)
        comp = component_for(c, device=values.device, dtype=values.dtype)
        block = values[:, i:i + n]
        out = out + comp(x, block)
        jac[:, :, i:i + n] = comp.grad(x, block).expand(P, C, n)
        i += n
    return out, jac


def evaluate(spec, x, values):
    """Evaluate a whole :class:`~spyde.fitting.spec.ModelSpec`.

    ``x``: ``(C,)``; ``values``: ``(P, n_total)`` packed in the spec's order.
    Returns ``(P, C)`` — the sum over active components.
    """
    import torch

    out = None
    i = 0
    for c in spec.active_components:
        n = len(c.scalar_parameters)
        comp = component_for(c, device=values.device, dtype=values.dtype)
        y = comp(x, values[:, i:i + n])
        out = y if out is None else out + y
        i += n
    if out is None:
        p = values.shape[0] if values.dim() > 1 else 1
        c = x.shape[-2] if x.shape[-1] == 2 and x.dim() >= 2 else x.shape[-1]
        return torch.zeros((p, c), dtype=x.dtype, device=x.device)
    return out

"""engine.py — batched Levenberg-Marquardt over the whole navigation grid.

HyperSpy's ``multifit`` fits one pixel at a time: measured at ~110 spectra/s on
this box, i.e. ~10 minutes for a 256x256 spectrum image. This module fits every
pixel *simultaneously*, following the playbook already proven in
``spyde/actions/vector_orientation_gpu.py``.

Why it works — the shape argument, which is the whole design:

    P  navigation positions   (10^3 - 10^5)
    C  channels per spectrum  (10^3 - 10^4)
    n  free parameters        (3 - 20)     <- tiny, and that is the point

Levenberg-Marquardt solves ``(JᵀJ + λ·diag(JᵀJ)) δ = -Jᵀr`` each step. ``JᵀJ`` is
``(P, n, n)`` — a batch of *tiny* linear systems, which ``torch.linalg`` solves
natively in one call. The only large object is the Jacobian ``(P, C, n)``, so
the batch dimension is **chunked** to bound it (65536 x 2048 x 12 float32 =
6.4 GB whole; chunked it is a fixed working set).

The Jacobian comes from ``torch.func.jacfwd``, not ``jacrev``: forward-mode
costs one pass per *input* (n ≈ 10) where reverse-mode costs one per *output*
(C ≈ 2000). For this shape that is a ~100x difference.

**There is one implementation, not two.** The "CPU fallback" is this same code
with ``device="cpu"`` — torch runs everywhere. A separate scipy path would be a
second thing to keep in agreement with HyperSpy, and HyperSpy already *is* the
reference: correctness here means reproducing ``multifit``'s parameters, which
``test_fitting_engine.py`` asserts directly.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from spyde.fitting import components as tcomp

log = logging.getLogger(__name__)

# Bounds the Jacobian working set per chunk. 2**22 elements x 8 B (float64)
# x (value + tangents) lands comfortably inside a few hundred MB.
_TARGET_JACOBIAN_ELEMENTS = 1 << 22


@dataclass
class FitResult:
    """Outcome of a batched fit.

    ``values`` is ``(P, n_total)`` in the spec's packed order — the same order
    as :meth:`~spyde.fitting.spec.ModelSpec.parameter_names`, so a component
    map is a column slice (``spec.component_slices()``).
    """

    values: np.ndarray            # (P, n_total)
    converged: np.ndarray         # (P,) bool
    chisq: np.ndarray             # (P,) sum of squared (weighted) residuals
    n_iter: int
    device: str

    @property
    def convergence_rate(self) -> float:
        return float(self.converged.mean()) if self.converged.size else 0.0

    def as_maps(self, spec, nav_shape) -> dict[str, np.ndarray]:
        """Per-parameter maps, reshaped to the navigation grid and keyed by
        ``"<component>.<parameter>"``."""
        names = spec.parameter_names()
        return {name: self.values[:, i].reshape(nav_shape)
                for i, name in enumerate(names)}


def default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception as e:                                # pragma: no cover
        log.debug("probing CUDA failed: %s", e)
    return "cpu"


def _selection_matrix(free_mask, n_total, dtype, device):
    """``S`` such that ``full = fixed + S @ p_free``.

    Written as a matmul rather than index assignment because the whole step
    runs under ``vmap``/``jacfwd``, where in-place writes into a tensor that
    carries tangents are at best fragile. ``n`` is tiny, so the matmul is free.
    """
    import torch
    idx = torch.nonzero(free_mask, as_tuple=False).squeeze(-1)
    S = torch.zeros((n_total, idx.numel()), dtype=dtype, device=device)
    S[idx, torch.arange(idx.numel(), device=device)] = 1.0
    return S


def _make_residual_fn(spec, x, S):
    """Build ``residual(p_free, y_i, w_i, fixed_i) -> (C,)`` for ONE position.

    Everything that varies per position is an explicit argument so the caller
    can ``vmap`` over all four; only the model structure and the signal axis
    are closed over.
    """
    def residual(p_free, y_i, w_i, fixed_i):
        full = fixed_i + (S @ p_free)
        model = tcomp.evaluate(spec, x, full.unsqueeze(0)).squeeze(0)
        return (model - y_i) * w_i

    return residual


def fit_batched(spec, data, x, *, weights=None, device=None, max_iter=60,
                ftol=1e-8, xtol=1e-8, gtol=1e-8, chunk=None, dtype="float64",
                progress: Callable[[int, int], None] | None = None,
                initial=None):
    """Fit *spec* to every spectrum in *data*, all at once.

    Parameters
    ----------
    spec : ModelSpec
        The model. Every ACTIVE component must have a batched port
        (``components.supports(spec)``); otherwise fall back to HyperSpy.
    data : array (P, C) or (..., C)
        Spectra. Leading dimensions are flattened to ``P``.
    x : array (C,)
        The signal axis (calibrated units — the same values HyperSpy fits in,
        not channel indices, or every centre/onset comes out in the wrong unit).
    weights : array (C,) or (P, C), optional
        Residual weights. ``"poisson"`` gives ``1/sqrt(max(y, 1))``, which is
        what counting data wants (#53).
    initial : array (P, n_total), optional
        Per-position starting values — the seeded-propagation hand-off (#54).
        Defaults to the spec's values broadcast to every position.

    Returns
    -------
    FitResult
    """
    import torch

    spec_names = spec.parameter_names()
    n_total = len(spec_names)
    if n_total == 0:
        raise ValueError("model has no active parameters to fit")
    if not tcomp.supports(spec):
        unsupported = [c.kind for c in spec.active_components
                       if c.kind not in tcomp.available()]
        raise NotImplementedError(
            f"no batched implementation for {unsupported}; use the HyperSpy "
            f"path for this model (see components.supports)")

    device = device or default_device()
    tdtype = getattr(torch, dtype)

    data = np.asarray(data)
    nav_shape = data.shape[:-1]
    y_np = data.reshape(-1, data.shape[-1])
    P, C = y_np.shape
    if len(x) != C:
        raise ValueError(f"signal axis has {len(x)} points but data has {C}")

    # --- weights ----------------------------------------------------------
    if isinstance(weights, str):
        if weights != "poisson":
            raise ValueError(f"unknown weighting {weights!r}")
        w_np = 1.0 / np.sqrt(np.maximum(y_np, 1.0))
    elif weights is None:
        w_np = np.ones((1, C))
    else:
        w_np = np.asarray(weights, float)
        if w_np.ndim == 1:
            w_np = w_np[None, :]

    # --- signal range: a masked channel simply gets zero weight -----------
    if spec.channel_mask is not None:
        mask = np.asarray(spec.channel_mask, bool)
        if mask.size != C:
            raise ValueError(f"channel mask has {mask.size} entries, "
                             f"signal axis has {C}")
        w_np = w_np * mask[None, :]

    free_mask_np = spec.free_mask()
    n_free = int(free_mask_np.sum())
    if n_free == 0:
        raise ValueError("every parameter is fixed — nothing to fit")

    lo_np, hi_np = spec.bounds_arrays()
    start_np = (np.broadcast_to(spec.flat_values(), (P, n_total)).copy()
                if initial is None else
                np.asarray(initial, float).reshape(P, n_total).copy())
    # A start outside its own bounds makes the first step meaningless.
    start_np = np.clip(start_np, lo_np, hi_np)

    if chunk is None:
        chunk = max(1, min(P, _TARGET_JACOBIAN_ELEMENTS // max(C * n_free, 1)))

    x_t = torch.as_tensor(np.asarray(x, float), dtype=tdtype, device=device)
    free_t = torch.as_tensor(free_mask_np, device=device)
    S = _selection_matrix(free_t, n_total, tdtype, device)
    lo_t = torch.as_tensor(lo_np[free_mask_np], dtype=tdtype, device=device)
    hi_t = torch.as_tensor(hi_np[free_mask_np], dtype=tdtype, device=device)

    out_values = np.empty((P, n_total), np.float64)
    out_conv = np.zeros(P, bool)
    out_chisq = np.full(P, np.inf)
    iters_used = 0

    for lo_i in range(0, P, chunk):
        hi_i = min(P, lo_i + chunk)
        v, c, q, it = _fit_chunk(
            spec, x_t, S, free_t, tdtype, device,
            y=torch.as_tensor(y_np[lo_i:hi_i], dtype=tdtype, device=device),
            w=torch.as_tensor(w_np[lo_i:hi_i] if w_np.shape[0] > 1 else w_np,
                              dtype=tdtype, device=device),
            start=torch.as_tensor(start_np[lo_i:hi_i], dtype=tdtype, device=device),
            lo=lo_t, hi=hi_t, max_iter=max_iter, ftol=ftol, xtol=xtol,
            gtol=gtol,
        )
        out_values[lo_i:hi_i] = v
        out_conv[lo_i:hi_i] = c
        out_chisq[lo_i:hi_i] = q
        iters_used = max(iters_used, it)
        if progress is not None:
            try:
                progress(hi_i, P)
            except Exception as e:                        # pragma: no cover
                log.debug("fit progress callback failed: %s", e)

    return FitResult(values=out_values, converged=out_conv, chisq=out_chisq,
                     n_iter=iters_used, device=device)


def _fit_chunk(spec, x, S, free_mask, tdtype, device, *, y, w, start, lo, hi,
               max_iter, ftol, xtol, gtol):
    """One chunk of positions through the LM loop. All tensors already on
    *device*; returns numpy."""
    import torch
    from torch.func import jacfwd, vmap

    n_chunk = y.shape[0]
    if w.shape[0] == 1:
        w = w.expand(n_chunk, -1)

    # Fixed parameters contribute a constant offset; free ones come via S.
    fixed_full = start * (~free_mask).to(tdtype)
    p = start[:, free_mask].clone()

    residual = _make_residual_fn(spec, x, S)
    # in_dims: p_free batched, and y/w/fixed batched alongside it.
    res_b = vmap(residual, in_dims=(0, 0, 0, 0))
    jac_b = vmap(jacfwd(residual, argnums=0), in_dims=(0, 0, 0, 0))

    lam = torch.full((n_chunk,), 1e-3, dtype=tdtype, device=device)
    r = res_b(p, y, w, fixed_full)
    cost = 0.5 * (r * r).sum(1)
    converged = torch.zeros(n_chunk, dtype=torch.bool, device=device)
    used = 0

    eye = torch.eye(p.shape[1], dtype=tdtype, device=device)

    for it in range(max_iter):
        used = it + 1
        J = jac_b(p, y, w, fixed_full)                     # (B, C, n_free)

        # COLUMN SCALING (MINPACK's `diag`), and it is load-bearing, not a
        # refinement. Parameters routinely differ by many orders of magnitude —
        # a PowerLaw fits A ~ 1e6 alongside r ~ 3 — which makes cond(J) ~ 1e6
        # and cond(JᵀJ) ~ 1e12. Cholesky in float64 then loses almost every
        # digit in the small-eigenvalue direction, so the Gauss-Newton step is
        # numerically junk and the fit crawls: measured 247 iterations to fit a
        # TWO-parameter power law (and it stopped 1.5-3.5% short of the value
        # multifit finds), versus 46 iterations to chisq ~1e-34 once the
        # columns are normalised.
        # Normalising each column to unit norm before forming JᵀJ removes the
        # scale entirely (and makes diag(JᵀJ) ≈ 1, so the Marquardt damping
        # below is automatically scale-free).
        colnorm = J.norm(dim=1).clamp_min(1e-300)          # (B, n)
        Js = J / colnorm.unsqueeze(1)
        Jst = Js.transpose(1, 2)
        JtJ = Jst @ Js                                     # (B, n, n) — tiny
        Jtr = (Jst @ r.unsqueeze(2)).squeeze(2)            # (B, n)

        diag = torch.diagonal(JtJ, dim1=1, dim2=2)
        A = JtJ + lam[:, None, None] * torch.diag_embed(
            diag.clamp_min(1e-12)) + 1e-14 * eye

        # cholesky_ex reports failure per item instead of raising, so one
        # singular pixel cannot abort the whole batch.
        L, info = torch.linalg.cholesky_ex(A)
        solvable = info == 0
        delta = torch.zeros_like(p)
        if solvable.any():
            sol = torch.cholesky_solve((-Jtr).unsqueeze(2)[solvable],
                                       L[solvable]).squeeze(2)
            delta[solvable] = sol
        delta = delta / colnorm                            # back to real units

        p_new = torch.clamp(p + delta, lo, hi)             # bounds by projection
        r_new = res_b(p_new, y, w, fixed_full)
        cost_new = 0.5 * (r_new * r_new).sum(1)

        better = (cost_new < cost) & solvable
        # Accepted: keep the step and trust the model more (smaller λ).
        # Rejected: fall back toward gradient descent (larger λ).
        p = torch.where(better[:, None], p_new, p)
        r = torch.where(better[:, None], r_new, r)
        rel = (cost - cost_new) / cost.clamp_min(1e-300)
        cost = torch.where(better, cost_new, cost)
        lam = torch.where(better, (lam / 3.0).clamp_min(1e-12),
                          (lam * 3.0).clamp_max(1e12))

        # Convergence. The GRADIENT test is the primary one, and it has to be:
        #
        #  * A step-size test alone is wrong on a REJECTED step — lambda grows,
        #    the next delta shrinks, and "stuck against a wall" reads as
        #    "converged" (measured: a PowerLaw stopping 1.5-3.5% short).
        #  * But requiring an ACCEPTED step is *also* wrong, because AT the
        #    optimum no step improves anything, so `better` is permanently
        #    False and neither test can ever fire — the fit then burns every
        #    iteration of its budget while already converged (measured: 8% of
        #    pixels "converged" on a fit that was actually finished).
        #
        # The gradient does not care whether the last step was taken. Columns
        # of J are unit-norm here (see the scaling above), so `Jᵀr` is the
        # scaled gradient and `|Jᵀr| / ||r||` is dimensionless — a real
        # relative tolerance rather than a magic absolute number.
        rnorm = r.norm(dim=1).clamp_min(1e-300)
        grad_small = (Jtr.abs().amax(1) / rnorm) < gtol
        step_small = better & (delta.abs() <= xtol * (p.abs() + xtol)).all(1)
        cost_small = better & (rel.abs() < ftol)
        converged = converged | ((grad_small | step_small | cost_small) & solvable)
        if bool(converged.all()):
            break

    # Reassemble the full parameter vector (free values back into their slots).
    full = fixed_full + (S @ p.unsqueeze(2)).squeeze(2)
    return (full.detach().cpu().numpy().astype(np.float64),
            converged.detach().cpu().numpy(),
            (2.0 * cost).detach().cpu().numpy().astype(np.float64),
            used)

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

from spyde.device_lock import accelerator_lock
from spyde.fitting import components as tcomp

log = logging.getLogger(__name__)

# Bounds the Jacobian working set per chunk, as ELEMENTS of (P, C, n).
#
# Sized to keep a GPU busy, not to be safe: chunking is what stops the whole
# (P, C, n) Jacobian materialising, but chunk too SMALL and every LM iteration
# becomes launch overhead on a nearly idle device. Measured at 1024 spectra x
# 1024 channels x 13 params, a 2**22 cap (which chopped the batch into pieces
# of 372) ran at 178 spectra/s against 267 for the whole batch in one go — the
# "safe" setting cost a third of the throughput. 2**26 elements is ~0.5 GB in
# float64 / 0.27 GB in float32, which fits any GPU worth using and lets typical
# scans run unchunked.
_TARGET_JACOBIAN_ELEMENTS = 1 << 26


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
    # Set only by spyde.fitting.seeding.fit_seeded — the fraction of the coarse
    # grid that produced a usable seed, and how many coarse fits there were.
    # None from a plain fit_batched, which is how a caller tells them apart.
    seed_converged: float | None = None
    n_seeds: int | None = None

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
    """CUDA > MPS > CPU, overridable with ``SPYDE_FIT_DEVICE``.

    **MPS is real but modest, and it is not free.** Every op this engine needs
    works on Metal (cholesky_ex, cholesky_solve, vmap(jacfwd), batched matmul),
    and float32 reproduces ``multifit`` to ~1e-5 — see
    ``test_fitting_engine.py::TestFloat32AndMPSParity``. What it does NOT buy is
    a speed-up on every Mac. Measured on an M1 MacBook Air (8 GPU cores),
    float32 both sides, after the sync removal below:

    | raw float32 GEMM     | MPS 1747 vs CPU 868 GFLOP/s -> 2.0x ceiling |
    | this engine, P=64    | 173 ms vs 171 ms            -> 0.99x        |
    | this engine, P=16384 | 41.6 s vs 28.4 s            -> 0.68x        |

    So on THIS box the CPU still wins. The engine is not GEMM-bound — ``n <= 20``
    free parameters make ``JᵀJ`` a batch of tiny matrices, so an LM iteration is
    many small kernels, exactly the shape Metal's launch overhead punishes
    (CLAUDE.md, GPU Computing). An M1 Air pairs a strong 8-core CPU with the
    smallest Apple GPU, which is the pessimistic end of the range; a Pro/Max/
    Ultra part carries the same per-launch cost against 2-8x the GPU compute.
    EBSD indexing, which IS one big matmul, already crosses over on this same
    laptop (``spyde/ebsd/_device.py``) — the difference is the workload, not the
    backend.

    Hence: prefer the accelerator and let anyone measure their own box with
    ``SPYDE_FIT_DEVICE``, rather than bake this laptop's ratio into a threshold
    that would be wrong on every other Mac.
    """
    import os
    forced = os.environ.get("SPYDE_FIT_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception as e:                                # pragma: no cover
        log.debug("probing the accelerator failed: %s", e)
    return "cpu"


def resolve_dtype(device: str, dtype: str) -> str:
    """Metal has no float64 — asking for one raises rather than downcasting.

    Coercing here (instead of at every call site) keeps ``dtype="float64"``
    usable as the default it is everywhere else.
    """
    if str(device).startswith("mps") and dtype == "float64":
        log.debug("MPS does not support float64; fitting in float32")
        return "float32"
    return dtype


def floor_tolerances(tdtype, ftol, xtol, gtol):
    """Raise the convergence tolerances to something the dtype can express.

    THE BUG THIS FIXES: the defaults are 1e-8, but float32's epsilon is 1.19e-7.
    All three tests — gradient, step and cost — are then strictly unreachable,
    so a fit that has genuinely converged reports ``converged=False`` for every
    position, burns its whole ``max_iter`` budget, and surfaces to the user as
    "did not converge" / "0 fitted" while the parameters are in fact correct to
    ~1e-5. Measured before this floor: ``convergence_rate`` 1.00 in float64 and
    **0.00** in float32, on identical data with identical answers.

    10x epsilon is the floor: comfortably above the noise a float32 residual
    carries, and far below the 1e-8 default in float64 (eps 2.2e-16), so
    double-precision behaviour is untouched.
    """
    import torch
    floor = 10.0 * float(torch.finfo(tdtype).eps)
    return (max(ftol, floor), max(xtol, floor), max(gtol, floor))


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
    dtype = resolve_dtype(device, dtype)
    tdtype = getattr(torch, dtype)
    ftol, xtol, gtol = floor_tolerances(tdtype, ftol, xtol, gtol)

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

    # PER CHUNK, not around the whole loop. On MPS every torch user in the
    # process shares one lock (spyde/device_lock.py), and a whole-scan fit runs
    # for minutes — holding it end-to-end would stall the live navigator preview
    # for the entire run. Releasing between chunks bounds any other thread's
    # wait to a single chunk, which is the same "hand the device back at your
    # yield points" rule compute_vector_orientation_gpu follows.
    for lo_i in range(0, P, chunk):
        hi_i = min(P, lo_i + chunk)
        with accelerator_lock(device):
            v, c, q, it = _fit_chunk(
                spec, x_t, S, free_t, tdtype, device,
                y=torch.as_tensor(y_np[lo_i:hi_i], dtype=tdtype, device=device),
                w=torch.as_tensor(w_np[lo_i:hi_i] if w_np.shape[0] > 1 else w_np,
                                  dtype=tdtype, device=device),
                start=torch.as_tensor(start_np[lo_i:hi_i], dtype=tdtype,
                                      device=device),
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

    # ANALYTIC path when every component supplies derivatives, autodiff
    # otherwise. This is the single biggest cost in the solver: measured on a
    # 13-free-parameter model, jacfwd took 51.6 ms per call against 3.4 ms for
    # one residual evaluation — forward-mode AD costs one pass per parameter,
    # while the closed forms mostly reuse the value that was computed anyway.
    analytic = tcomp.has_analytic_grad(spec)
    if analytic:
        def res_b(pf, y_i, w_i, fixed_i):
            full = fixed_i + (pf @ S.T)
            return (tcomp.evaluate(spec, x, full) - y_i) * w_i

        def val_jac(pf, y_i, w_i, fixed_i):
            full = fixed_i + (pf @ S.T)
            model, jac_full = tcomp.evaluate_with_grad(spec, x, full)
            # Weight the residual and its Jacobian identically, and keep only
            # the FREE columns (S selects them, so S picks the same subset).
            return ((model - y_i) * w_i,
                    (jac_full * w_i.unsqueeze(-1)) @ S)
    else:
        residual = _make_residual_fn(spec, x, S)
        # in_dims: p_free batched, and y/w/fixed batched alongside it.
        _res_v = vmap(residual, in_dims=(0, 0, 0, 0))
        _jac_v = vmap(jacfwd(residual, argnums=0), in_dims=(0, 0, 0, 0))

        def res_b(pf, y_i, w_i, fixed_i):
            return _res_v(pf, y_i, w_i, fixed_i)

        def val_jac(pf, y_i, w_i, fixed_i):
            return (_res_v(pf, y_i, w_i, fixed_i),
                    _jac_v(pf, y_i, w_i, fixed_i))

    # The scale the residual is measured against, for the noise-floor test in
    # the loop. Weighted, because `r` is.
    ynorm = (y * w).norm(dim=1).clamp_min(1e-300)
    # A residual this small IS zero as far as the dtype is concerned. 4x eps
    # rather than 1x so the test survives the few rounding steps between `y`
    # and `r` (the model evaluation, the subtraction, the weighting).
    noise_floor = 4.0 * float(torch.finfo(tdtype).eps)

    lam = torch.full((n_chunk,), 1e-3, dtype=tdtype, device=device)
    cost = torch.full((n_chunk,), float("inf"), dtype=tdtype, device=device)
    converged = torch.zeros(n_chunk, dtype=torch.bool, device=device)
    used = 0

    eye = torch.eye(p.shape[1], dtype=tdtype, device=device)
    # Reading a device tensor on the host stalls the queue; on the CPU it is a
    # plain memory read. So poll convergence every iteration there and rarely
    # on an accelerator.
    check_every = 1 if str(device).startswith("cpu") else 8

    for it in range(max_iter):
        used = it + 1
        # One call gives both — the value and its derivatives share almost all
        # of their work (df/dA IS the shape), so computing them separately
        # would repeat the expensive part.
        r, J = val_jac(p, y, w, fixed_full)                # (B, C), (B, C, n_free)
        cost = 0.5 * (r * r).sum(1)

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
        # Solve the WHOLE batch and mask afterwards, rather than gathering the
        # solvable rows first. `if solvable.any():` and `L[solvable]` both need
        # the mask's VALUE on the host, and on MPS each such round trip costs
        # ~367 us against ~30 us for the same reduction left on the device —
        # twice per iteration, every iteration. Substituting the identity where
        # the factorisation failed is arithmetically the same answer (those rows
        # are discarded by the `where` below, exactly as `delta` stayed zero for
        # them before) and keeps the whole step on-device.
        L_safe = torch.where(solvable[:, None, None], L, eye)
        sol = torch.cholesky_solve((-Jtr).unsqueeze(2), L_safe).squeeze(2)
        delta = torch.where(solvable[:, None], sol, torch.zeros_like(sol))
        delta = delta / colnorm                            # back to real units

        p_new = torch.clamp(p + delta, lo, hi)             # bounds by projection
        r_new = res_b(p_new, y, w, fixed_full)
        cost_new = 0.5 * (r_new * r_new).sum(1)

        better = (cost_new < cost) & solvable
        # Accepted: keep the step and trust the model more (smaller λ).
        # Rejected: fall back toward gradient descent (larger λ).
        # `r`/`cost` are NOT carried forward — the top of the next iteration
        # recomputes both from the current `p` alongside the Jacobian it needs
        # anyway, so tracking them here would only risk them drifting out of
        # step with `p`.
        p = torch.where(better[:, None], p_new, p)
        rel = (cost - cost_new) / cost.clamp_min(1e-300)
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
        # NOISE FLOOR — the test the three relative ones cannot make on a
        # near-perfect fit. `grad_small` divides by `rnorm`, so once the fit is
        # exact it is 0/0: on CLEAN data in float32 the residual falls to the
        # rounding floor (~3e-6 against data of order 50), and `Jᵀr` is then
        # rounding noise of the same relative size, so the ratio is O(1) and
        # never passes. Measured: an exact fit — chisq 1e-11, parameters right
        # to 8 digits — reported 50% converged and burned all 60 iterations.
        # A residual at the representation limit cannot be improved, so it IS
        # converged, and saying so absolutely is the only way to say it.
        at_floor = rnorm <= noise_floor * ynorm
        converged = converged | ((grad_small | step_small | cost_small
                                  | at_floor) & solvable)
        # The early exit needs the answer on the HOST, so on an accelerator it
        # is a pipeline barrier (see the cholesky comment above). Amortise it:
        # checking every 8th iteration costs at most 7 wasted iterations once,
        # against a stall on all 60. On CPU the read is free, so keep it exact.
        if it % check_every == check_every - 1 and bool(converged.all()):
            break

    # Final cost from the FINAL p. The loop's `cost` belongs to the start of
    # the last iteration, so reporting it would attribute the previous point's
    # chisq to the answer actually returned.
    r_final = res_b(p, y, w, fixed_full)
    chisq = (r_final * r_final).sum(1)

    # Reassemble the full parameter vector (free values back into their slots).
    full = fixed_full + (S @ p.unsqueeze(2)).squeeze(2)
    return (full.detach().cpu().numpy().astype(np.float64),
            converged.detach().cpu().numpy(),
            chisq.detach().cpu().numpy().astype(np.float64),
            used)

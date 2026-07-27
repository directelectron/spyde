"""refine.py — batched orientation refinement (#72).

Dictionary indexing (#71) can only ever return a dictionary entry, so its
accuracy is capped by the sampling step: a 5-degree dictionary gives 5-degree
answers. Refinement lifts that cap by optimising each orientation continuously,
starting from its indexed match.

This is the same shape of problem as ``actions/vector_orientation_gpu.py`` and
is solved the same way: **the whole field at once**. Every pattern's three
Euler angles are one row of a ``(P, 3)`` tensor, the simulated patterns are one
batched forward pass, and one Adam optimiser walks all P orientations
simultaneously. No dask, no per-pattern loop.

Two things carried over from the vector-orientation work, both of which cost
real time to learn there (CLAUDE.md, GPU Computing):

* **`backward()` segfaults off the main thread under CUDA on Windows.** The
  refine loop therefore runs INLINE on the calling thread with an ``on_yield``
  callback to keep a UI responsive, rather than being pushed to a worker.
* **Yield inside the step loop, not just between stages**, or the window
  freezes for seconds and the progress bar looks stuck.

The simulator is injectable. The built-in one is the same band geometry the
synthetic data uses, which makes refinement testable end-to-end against known
Euler angles today; #69 swaps in kikuchipy master patterns without touching the
optimiser.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RefinementResult:
    euler: np.ndarray          # (P, 3) refined Bunge angles, radians
    score: np.ndarray          # (P,) NCC after refinement
    score_before: np.ndarray   # (P,) NCC at the starting orientation
    n_steps: int
    device: str

    @property
    def improved(self) -> np.ndarray:
        return self.score >= self.score_before

    def euler_map(self, nav_shape) -> np.ndarray:
        return self.euler.reshape(tuple(nav_shape) + (3,))


def euler_to_matrix_torch(euler):
    """Bunge ZXZ Euler angles -> rotation matrices, differentiable.

    ``(P, 3)`` -> ``(P, 3, 3)``. Built by stacking rather than by writing into
    an empty tensor: an in-place write into a tensor that carries gradients is
    the classic way to lose them silently here.
    """
    import torch

    phi1, Phi, phi2 = euler[:, 0], euler[:, 1], euler[:, 2]
    c1, s1 = torch.cos(phi1), torch.sin(phi1)
    c, s = torch.cos(Phi), torch.sin(Phi)
    c2, s2 = torch.cos(phi2), torch.sin(phi2)
    rows = [
        c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s,
        -c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s,
        s1 * s, -c1 * s, c,
    ]
    return torch.stack(rows, dim=-1).reshape(-1, 3, 3)


class BandSimulator:
    """Differentiable Kikuchi-band pattern simulator.

    Renders the same geometry as :func:`spyde.data.synthetic.simulate_patterns`
    — a band appears where a detector direction is near-perpendicular to a
    plane normal — but in torch, so the pattern is differentiable with respect
    to the orientation.

    Stands in for a master-pattern lookup (#69). The optimiser does not care
    which it is: anything mapping ``(P, 3)`` Euler angles to ``(P, K)`` patterns
    differentiably will do.
    """

    def __init__(self, detector=(60, 60), pc=(0.5, 0.5, 0.55), *,
                 device="cpu", dtype=None):
        import torch
        from spyde.data.synthetic import _cubic_plane_normals, detector_directions

        dtype = dtype or torch.float32
        r = detector_directions(detector, pc).reshape(-1, 3)
        normals, weights = _cubic_plane_normals()
        self.r = torch.as_tensor(r, dtype=dtype, device=device)
        self.normals = torch.as_tensor(normals, dtype=dtype, device=device)
        self.weights = torch.as_tensor(weights, dtype=dtype, device=device)
        self.widths = torch.as_tensor(
            0.055 * (weights / weights.max()) + 0.012, dtype=dtype, device=device)
        self.shape = tuple(detector)

    def __call__(self, euler):
        """``(P, 3)`` -> ``(P, K)``."""
        rot = euler_to_matrix_torch(euler)                    # (P, 3, 3)
        n_rot = self.normals @ rot.transpose(1, 2)            # (P, B, 3)
        d = n_rot @ self.r.T                                  # (P, B, K)
        band = (-0.5 * (d / self.widths[None, :, None]) ** 2).exp()
        return (band * self.weights[None, :, None]).sum(1)    # (P, K)


def _ncc(a, b, eps=1e-12):
    """Row-wise normalised cross-correlation."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(eps)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (a * b).sum(-1)


def refine_orientations(patterns, euler_start, *, simulator=None,
                        detector=None, pc=(0.5, 0.5, 0.55), device=None,
                        steps: int = 120, lr: float = 0.01, chunk=None,
                        on_yield: Callable[[], None] | None = None,
                        yield_every: int = 12,
                        progress: Callable[[int, int], None] | None = None):
    """Continuously refine one orientation per pattern.

    Parameters
    ----------
    patterns : array (..., H, W)
        Experimental patterns. Background-corrected is strongly preferred —
        NCC is not invariant to a detector gradient (see
        :mod:`spyde.ebsd.preprocess`).
    euler_start : array (..., 3)
        Starting orientations, normally ``IndexingResult.orientations(...)``.
    simulator : callable, optional
        ``(P, 3) -> (P, K)``, differentiable. Defaults to :class:`BandSimulator`
        on *detector*.
    steps, lr :
        Adam budget. The default is deliberately generous: refinement runs once
        per scan, and the cost is one batched forward+backward per step.

    Returns
    -------
    RefinementResult

    Notes
    -----
    Runs INLINE on the calling thread — ``backward()`` segfaults on a worker
    thread under CUDA on Windows. Pass *on_yield* to keep a UI alive; it is
    called every *yield_every* steps, inside the loop, because yielding only
    between stages leaves the window frozen for seconds at a time.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    exp = np.asarray(patterns)
    nav_shape = exp.shape[:-2]
    K = int(np.prod(exp.shape[-2:]))
    E = exp.reshape(-1, K)
    start = np.asarray(euler_start, float).reshape(-1, 3)
    if len(start) != len(E):
        raise ValueError(f"{len(E)} patterns but {len(start)} starting "
                         f"orientations")

    if simulator is None:
        if detector is None:
            detector = tuple(exp.shape[-2:])
        simulator = BandSimulator(detector, pc, device=device)

    P = len(E)
    chunk = int(chunk) if chunk else P
    out_euler = np.empty((P, 3), np.float64)
    out_score = np.empty(P, np.float32)
    out_before = np.empty(P, np.float32)

    for lo in range(0, P, chunk):
        hi = min(P, lo + chunk)
        e_t = torch.as_tensor(E[lo:hi], dtype=torch.float32, device=device)
        e_t = e_t - e_t.mean(-1, keepdim=True)
        e_t = e_t / e_t.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        ang = torch.as_tensor(start[lo:hi], dtype=torch.float32,
                              device=device).clone().requires_grad_(True)
        with torch.no_grad():
            out_before[lo:hi] = _ncc(simulator(ang), e_t).cpu().numpy()

        opt = torch.optim.Adam([ang], lr=lr)
        # Pin backward to this thread; see the module docstring.
        prev_mt = torch.is_grad_enabled()
        try:
            torch.autograd.set_multithreading_enabled(False)
        except Exception as exc:                              # pragma: no cover
            log.debug("set_multithreading_enabled unavailable: %s", exc)

        try:
            for step in range(steps):
                opt.zero_grad(set_to_none=True)
                # Maximise similarity == minimise its negative. Summed, not
                # averaged: each pattern's gradient is independent, so the sum
                # keeps every one at full strength regardless of batch size.
                loss = -_ncc(simulator(ang), e_t).sum()
                loss.backward()
                opt.step()
                if on_yield is not None and step % yield_every == 0:
                    try:
                        on_yield()
                    except Exception as exc:                  # pragma: no cover
                        log.debug("refine on_yield failed: %s", exc)
        finally:
            try:
                torch.autograd.set_multithreading_enabled(prev_mt)
            except Exception:                                 # pragma: no cover
                pass

        with torch.no_grad():
            final = _ncc(simulator(ang), e_t)
            better = final >= torch.as_tensor(out_before[lo:hi], device=device)
            # NEVER return a worse orientation than we were given. Refinement
            # is an improvement step on top of indexing; if Adam wandered off
            # (a bad starting point, too large an lr) the indexed answer is
            # still the better estimate.
            keep = torch.where(better.unsqueeze(1), ang,
                               torch.as_tensor(start[lo:hi],
                                               dtype=torch.float32,
                                               device=device))
            out_euler[lo:hi] = keep.detach().cpu().numpy()
            out_score[lo:hi] = torch.maximum(
                final, torch.as_tensor(out_before[lo:hi],
                                       device=device)).cpu().numpy()

        if progress is not None:
            try:
                progress(hi, P)
            except Exception as exc:                          # pragma: no cover
                log.debug("refine progress callback failed: %s", exc)

    result = RefinementResult(out_euler, out_score, out_before, steps, device)
    result.nav_shape = tuple(nav_shape)
    return result

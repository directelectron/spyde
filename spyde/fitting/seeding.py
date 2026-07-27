"""seeding.py — SAMFire's good idea, without SAMFire's scheduler.

SAMFire's insight is that **a neighbour's fitted parameters are the best
starting point** for the next pixel: a spectrum image is spatially smooth, so
by the time you have fitted one pixel you almost know the answer for the one
next to it. That insight is worth keeping.

What is *not* worth keeping is how SAMFire acts on it — a per-pixel scheduler
with markers and strategies that walks the grid one position at a time,
serialising exactly the work the batched engine exists to do all at once.

So this module uses the insight as a **seed source** and leaves the scheduling
alone:

1. fit a coarse strided grid from the model's own starting values;
2. propagate those results across the full grid as initial values;
3. run ONE batched refine over every position.

Two batched fits total, not P sequential ones. The coarse pass costs
``1/stride**2`` of a full pass (a stride of 4 is ~6%), and it buys the refine a
starting point close enough that hard models converge where a cold start
stalls.

**Only converged coarse fits are propagated.** A coarse fit that failed is not
merely useless as a seed, it is actively worse than the model's defaults — it
has wandered somewhere unphysical, and seeding a neighbourhood from it spreads
that failure instead of the answer.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.fitting.engine import FitResult, fit_batched

log = logging.getLogger(__name__)


def _coarse_indices(nav_shape, stride):
    """Positions of the coarse grid, and the map from every full-grid position
    to its nearest coarse position.

    Returns ``(flat_coarse_idx, nearest_flat_coarse_for_each_position)``.
    """
    grids = [np.arange(0, n, stride) for n in nav_shape]
    # Always include the LAST index along each axis. Without it a grid whose
    # size is not a multiple of the stride has an unseeded strip down its far
    # edge — which is where a scan's contrast often changes most.
    grids = [g if g[-1] == n - 1 else np.append(g, n - 1)
             for g, n in zip(grids, nav_shape)]

    mesh = np.meshgrid(*grids, indexing="ij")
    flat_coarse = np.ravel_multi_index([m.ravel() for m in mesh], nav_shape)

    # For each axis, which coarse sample is nearest to each full index.
    nearest_per_axis = []
    for g, n in zip(grids, nav_shape):
        full = np.arange(n)
        if len(g) == 1:
            # A singleton axis (or one shorter than the stride) has exactly one
            # candidate. The neighbour comparison below would index g[-1] and
            # produce a coordinate of -1, so short-circuit it.
            nearest_per_axis.append(np.zeros(n, dtype=int))
            continue
        # searchsorted + compare neighbours: exact nearest, no float rounding
        # surprises at the halfway point.
        pos = np.clip(np.searchsorted(g, full), 1, len(g) - 1)
        left, right = g[pos - 1], g[pos]
        nearest_per_axis.append(np.where(full - left <= right - full,
                                         pos - 1, pos))

    coarse_shape = tuple(len(g) for g in grids)
    idx_mesh = np.meshgrid(*nearest_per_axis, indexing="ij")
    nearest_flat = np.ravel_multi_index([m.ravel() for m in idx_mesh],
                                        coarse_shape)
    return flat_coarse, nearest_flat


def fit_seeded(spec, data, x, *, stride: int = 4, coarse_max_iter: int = 120,
               progress=None, **kwargs) -> FitResult:
    """Coarse fit -> propagate -> one batched refine.

    Parameters
    ----------
    stride : int
        Coarse-grid spacing in navigation positions. 4 samples ~6% of the grid.
        ``stride <= 1`` skips seeding entirely and just fits everything.
    coarse_max_iter : int
        The coarse pass gets a LARGER iteration budget than the refine: there
        are few of these fits, they start cold, and their whole value is being
        right. A bad seed is worse than none.

    Remaining keyword arguments go to
    :func:`~spyde.fitting.engine.fit_batched`.

    Returns
    -------
    FitResult
        From the refine pass, with ``seed_converged`` recording how much of the
        coarse grid produced a usable seed.
    """
    data = np.asarray(data)
    nav_shape = data.shape[:-1]
    P = int(np.prod(nav_shape)) if nav_shape else 1

    if stride <= 1 or P <= 1 or not nav_shape:
        return fit_batched(spec, data, x, progress=progress, **kwargs)

    flat = data.reshape(P, data.shape[-1])
    coarse_idx, nearest = _coarse_indices(nav_shape, int(stride))

    if coarse_idx.size >= P:                      # stride too big to help
        return fit_batched(spec, data, x, progress=progress, **kwargs)

    # -- 1. coarse pass ----------------------------------------------------
    log.debug("seeding: coarse fit of %d/%d positions (stride %d)",
              coarse_idx.size, P, stride)
    coarse_kwargs = dict(kwargs)
    coarse_kwargs["max_iter"] = coarse_max_iter
    coarse = fit_batched(spec, flat[coarse_idx], x, **coarse_kwargs)

    # -- 2. propagate ------------------------------------------------------
    n_total = coarse.values.shape[1]
    defaults = np.broadcast_to(spec.flat_values(), (P, n_total))
    seeds = np.array(coarse.values[nearest], dtype=np.float64, copy=True)

    # A failed coarse fit has wandered somewhere unphysical; seeding from it
    # would SPREAD the failure. Those positions start from the model defaults
    # instead, exactly as an unseeded fit would.
    usable = coarse.converged[nearest]
    seeds[~usable] = defaults[~usable]
    if not coarse.converged.any():
        log.warning("seeding: no coarse fit converged — every position falls "
                    "back to the model's starting values")

    # -- 3. one batched refine --------------------------------------------
    result = fit_batched(spec, data, x, initial=seeds, progress=progress,
                         **kwargs)
    result.seed_converged = float(coarse.converged.mean())
    result.n_seeds = int(coarse_idx.size)
    return result

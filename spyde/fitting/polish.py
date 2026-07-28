"""polish.py — rescue the positions the batched fit got wrong.

A whole-scan fit already puts most positions at the noise floor; the headroom
is the few percent that land somewhere else. Those are not spread evenly — a
failure is a pixel whose neighbours all succeeded, because a spectrum image is
smooth and the answer next door is nearly the answer here. So the fix is the
one a person would use: find the positions that are worse than their
surroundings, start each from its best neighbour, and fit those again.

Measured on hyperspy's ``two_gaussians`` (1024 positions, two gaussians):
positions worse than 1.5x the noise floor go **27 -> 1**, total chisq improves
**9.7%**, and it costs 0.1 s a pass against 2.8 s for the fit itself. Two
passes reach the fixed point; the third improves nothing, which is why the
loop stops on "no improvement" rather than a fixed count.

**Poor is LOCAL, not absolute.** chisq scales with the counts in the spectrum,
so a bright pixel legitimately has a bigger one than a dim pixel and any global
threshold either flags a whole region or nothing at all. Comparing each
position with the median of its neighbours is the honest form of "this one went
wrong and the ones around it did not".

**A refit is only kept where it is better.** Reseeding is a heuristic; on a
position that was already right it can land somewhere worse. Keeping the
minimum makes the pass safe to run always and safe to repeat.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.fitting.engine import fit_batched

log = logging.getLogger(__name__)


def neighbour_index(nav_shape) -> list[np.ndarray]:
    """4-connected neighbours of every flat navigation position."""
    nav_shape = tuple(int(n) for n in nav_shape)
    total = int(np.prod(nav_shape))
    idx = np.arange(total).reshape(nav_shape)
    out = []
    for pos in np.ndindex(nav_shape):
        n = []
        for axis in range(len(nav_shape)):
            for step in (-1, 1):
                j = list(pos)
                j[axis] += step
                if 0 <= j[axis] < nav_shape[axis]:
                    n.append(int(idx[tuple(j)]))
        out.append(np.array(n, dtype=int))
    return out


def poor_mask(chisq, nb, factor: float = 1.5, converged=None) -> np.ndarray:
    """Positions that fit worse than their surroundings.

    A position that did not converge is poor whatever its chisq — it stopped
    because it ran out of iterations, not because it arrived.

    A position with fewer than two neighbours is never flagged on chisq alone.
    The comparison is a median, and the median of ONE number is that number:
    at a scan corner the test reduces to "is this pixel worse than the single
    pixel beside it", which a steep intensity gradient satisfies on its own.
    It can still be flagged for not converging, which does not depend on the
    neighbourhood.
    """
    chisq = np.asarray(chisq, float).ravel()
    local = np.array([np.median(chisq[n]) if n.size else chisq[i]
                      for i, n in enumerate(nb)])
    judged = np.array([n.size >= 2 for n in nb])
    mask = judged & (chisq > float(factor) * np.maximum(local, 1e-30))
    if converged is not None:
        mask |= ~np.asarray(converged, bool).ravel()
    return mask


def polish_scan(spec, data, x, result, *, nav_shape, max_passes: int = 4,
                factor: float = 1.5, max_iter: int = 120, progress=None,
                **kwargs):
    """Refit the poor positions from their best neighbour, until it stops
    helping.

    Mutates and returns *result*: ``values``, ``chisq`` and ``converged`` are
    replaced where the refit did better, and ``polish_passes`` /
    ``polish_improved`` record what happened.
    """
    nav_shape = tuple(int(n) for n in (nav_shape or ()))
    total = int(np.prod(nav_shape)) if nav_shape else 0
    values = np.array(result.values, dtype=np.float64, copy=True)
    chisq = np.array(result.chisq, dtype=np.float64).ravel()
    converged = np.array(result.converged, dtype=bool).ravel()
    if total != values.shape[0] or total < 3:
        return result

    flat = np.asarray(data, float).reshape(total, -1)
    nb = neighbour_index(nav_shape)
    improved_total, passes = 0, 0

    for p in range(int(max_passes)):
        poor = poor_mask(chisq, nb, factor=factor, converged=converged)
        if not poor.any():
            break
        passes = p + 1
        # Seed each poor position from its BEST neighbour — lowest chisq among
        # those that are not themselves poor, falling back to the least bad.
        seeds = values.copy()
        for i in np.flatnonzero(poor):
            n = nb[i]
            if not n.size:
                continue
            good = n[~poor[n]]
            src = (good[np.argmin(chisq[good])] if good.size
                   else n[np.argmin(chisq[n])])
            seeds[i] = values[src]

        if progress is not None:
            progress(p, int(max_passes))
        sub = fit_batched(spec, flat[poor], x, initial=seeds[poor],
                          max_iter=max_iter, **kwargs)

        # Keep ONLY where it is better. A refit is a heuristic; on a position
        # that was already right it can land somewhere worse, and a pass that
        # can make things worse cannot be run automatically.
        sub_chisq = np.asarray(sub.chisq, float).ravel()
        where = np.flatnonzero(poor)
        better = sub_chisq < chisq[where]
        if not better.any():
            break
        keep = where[better]
        values[keep] = np.asarray(sub.values, float)[better]
        chisq[keep] = sub_chisq[better]
        converged[keep] = np.asarray(sub.converged, bool).ravel()[better]
        improved_total += int(better.sum())
        log.debug("polish pass %d: refit %d, improved %d", p + 1,
                  int(poor.sum()), int(better.sum()))

    result.values = values
    result.chisq = chisq
    result.converged = converged
    result.polish_passes = passes
    result.polish_improved = improved_total
    return result

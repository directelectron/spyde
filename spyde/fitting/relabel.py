"""relabel.py — keep a component's IDENTITY the same across a scan.

Two gaussians in one model are **exchangeable**: nothing says which is "the
broad one". Fitted position by position they land in whichever slot each fit
happens to pick, and on a real spectrum image they disagree — measured on
hyperspy's ``two_gaussians``, the first component was the broad peak at only
**43%** of positions, its sigma ranging from 2.2 to 25.7 across the scan.

Every visible consequence of that reads as "the fit is broken":

* scrubbing the navigator makes the caret's numbers jump between two answers,
  and at each jump one component's amplitude drops by an order of magnitude —
  it looks like the fit is suppressing a component to zero;
* a committed component map is a checkerboard of the broad peak and the narrow
  one, which is not a map of anything.

This is a RELABELLING, not a refit: it permutes fitted values between
interchangeable slots and changes no residual. Fit quality is untouched; only
which component is called what.

**Why a global discriminant and not a serpentine chain.** The obvious method is
the one a person would use — walk the grid in serpentine order and keep the
arrangement that best matches the neighbour you just came from. It was tried
and it drifts: matching is chained, so ONE bad match (at a position where the
components are genuinely similar, or one that did not converge) flips the
convention for every position after it. Measured: 425 of 1024 positions
correctly permuted, and agreement went DOWN, 43% to 28%, because the walk
adopted the opposite convention partway through and kept it.

So the ordering is decided ONCE for the whole scan instead: find the parameter
that most reliably separates the components (here ``sigma``, 2.5 against 25.5)
and sort every position by it. Global, order-independent, deterministic, and
it cannot drift because there is no chain.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Below this the parameter does not really tell the components apart, and
# sorting by it would impose an identity the data does not support.
MIN_SEPARATION = 1.5


def exchangeable_groups(spec) -> list[list[int]]:
    """Indices of components that could be swapped without changing the model.

    Same kind, same free/fixed pattern, same bounds — anything else and the
    two are not actually interchangeable, so permuting them would move a
    fitted value into a slot that means something different.
    """
    sig: dict[tuple, list[int]] = {}
    for i, c in enumerate(spec.active_components):
        key = (c.kind,
               tuple(p.name for p in c.scalar_parameters),
               tuple(bool(p.free) for p in c.scalar_parameters),
               tuple(p.bounds() for p in c.scalar_parameters))
        sig.setdefault(key, []).append(i)
    return [g for g in sig.values() if len(g) > 1]


def _slices(spec) -> list[slice]:
    """Where each active component's scalar parameters sit in a flat vector."""
    out, start = [], 0
    for c in spec.active_components:
        n = len(c.scalar_parameters)
        out.append(slice(start, start + n))
        start += n
    return out


def _discriminant(block: np.ndarray) -> tuple[int, float]:
    """Which parameter genuinely tells the components apart, and how well.

    *block* is ``(P, k, n_p)`` — every position, every component in the group,
    every parameter.

    The measure is DISJOINTNESS, not separation, and the distinction is the
    whole difficulty: sorting always separates something. Two components drawn
    from the same distribution still produce a "smaller" slot and a "larger"
    one, with a perfectly respectable gap between their means — order them by
    that and you have invented an identity the data does not support, which is
    worse than leaving them inconsistent because the resulting component map
    looks meaningful.

    So the score asks whether the slots' ranges actually stay APART across the
    scan: the bottom of the upper slot minus the top of the lower one (5th and
    95th percentiles, so a couple of bad positions cannot veto a real split),
    over the typical within-slot spread. Positive and large means a narrow peak
    and a broad one; zero or negative means they overlap and nothing here
    distinguishes them.
    """
    best_j, best_score = -1, 0.0
    for j in range(block.shape[2]):
        ordered = np.sort(block[:, :, j], axis=1)            # (P, k)
        spread = np.median(np.abs(ordered - np.median(ordered, axis=0)))
        gaps = []
        for a in range(ordered.shape[1] - 1):
            lo_top = float(np.percentile(ordered[:, a], 95))
            hi_bottom = float(np.percentile(ordered[:, a + 1], 5))
            gaps.append(hi_bottom - lo_top)
        gap = float(min(gaps)) if gaps else 0.0
        if gap <= 0:
            continue                       # the slots overlap: not a divider
        score = gap / spread if spread > 1e-12 else np.inf
        if score > best_score:
            best_j, best_score = j, score
    return best_j, best_score


def relabel_scan(spec, values, nav_shape=None, *, converged=None) -> np.ndarray:
    """Order exchangeable components consistently across every position.

    Parameters
    ----------
    values : (P, n) array
        Fitted parameters, one row per navigation position, in
        ``spec.parameter_names()`` order.
    nav_shape, converged
        Accepted and unused — the ordering is global, so it needs neither the
        grid's shape nor which positions converged. Kept in the signature
        because callers have them and an earlier neighbour-chaining version
        did use them.

    Returns
    -------
    (P, n) array
        A copy, with component blocks permuted. The residual at every position
        is unchanged: this only decides which slot each fitted peak sits in.
    """
    values = np.array(values, dtype=np.float64, copy=True)
    if values.ndim != 2 or values.shape[0] < 2:
        return values
    groups = exchangeable_groups(spec)
    if not groups:
        return values

    slices = _slices(spec)
    for group in groups:
        cols = [slices[i] for i in group]
        block = np.stack([values[:, s] for s in cols], axis=1)   # (P, k, n_p)
        j, score = _discriminant(block)
        if j < 0 or score < MIN_SEPARATION:
            # Nothing separates them reliably. Any ordering here would be an
            # invention, and a wrong one is worse than an inconsistent one —
            # it would make a component map look meaningful when it is not.
            log.debug("relabel: components %s are not separable (best score "
                      "%.2f); left as fitted", group, score)
            continue
        order = np.argsort(block[:, :, j], axis=1)               # (P, k)
        permuted = np.take_along_axis(block, order[:, :, None], axis=1)
        for slot, s in enumerate(cols):
            values[:, s] = permuted[:, slot, :]
        log.debug("relabel: ordered components %s by parameter %d "
                  "(separation %.1f)", group, j, score)
    return values


def consistency(spec, values, key: str = "sigma") -> float:
    """Fraction of positions whose components are in ascending *key* order.

    The diagnostic for the above: 1.0 means every position agrees about which
    slot holds the narrow peak. Around 0.5 means the arrangement is a coin
    flip, which is what an unrelabelled scan looks like.
    """
    groups = exchangeable_groups(spec)
    if not groups:
        return 1.0
    slices = _slices(spec)
    names = [p.name for p in spec.active_components[0].scalar_parameters]
    if key not in names:
        return 1.0
    j = names.index(key)
    fracs = []
    for group in groups:
        cols = np.stack([values[:, slices[i]][:, j] for i in group], axis=1)
        fracs.append(float(np.mean(np.all(np.diff(cols, axis=1) > 0, axis=1))))
    return float(np.mean(fracs))

"""dumbbell.py — dumbbell lattices (#78).

Many structures image as PAIRS of closely-spaced atoms — silicon down <110>
being the canonical one. Treating each pair as two independent atoms throws
away the thing the pair actually measures: the separation and orientation of
the dumbbell, which is what carries polarisation, tilt and local strain.

Following https://atomap.org/dumbbell_lattice.html, the workflow is:

1. find the **dumbbell vector** — the displacement from one atom of a pair to
   its partner, shared by every dumbbell in the field;
2. **pair** the atoms using it;
3. measure per-dumbbell properties.

The vector is estimated from the data rather than assumed, because it is a
property of the projected structure and the scan rotation, and a user should
not have to type it in. :func:`estimate_dumbbell_vector` is deliberately
separable from :func:`pair_atoms` so a user who knows the vector — or who
picked it interactively (#76) — can skip the estimate.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def estimate_dumbbell_vector(positions, *, max_separation: float | None = None):
    """The displacement between the two atoms of a dumbbell -> ``(dx, dy)``.

    Each atom's nearest neighbour IS its dumbbell partner when the pair
    separation is smaller than the lattice spacing, which is what makes a
    dumbbell a dumbbell. So the nearest-neighbour displacements form two tight
    clusters at ``+v`` and ``-v``; folding them into one half-plane and taking
    the median gives ``v`` robustly, without a fit.

    The median, not the mean: a few mis-paired atoms at the field edge would
    drag a mean noticeably, and there is no reason to be sensitive to them.
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(positions, float).reshape(-1, 2)
    if len(pos) < 2:
        raise ValueError("need at least two atoms to estimate a dumbbell vector")

    dist, idx = cKDTree(pos).query(pos, k=2)
    delta = pos[idx[:, 1]] - pos                       # to nearest neighbour
    d = dist[:, 1]

    if max_separation is not None:
        keep = d <= float(max_separation)
        if not keep.any():
            raise ValueError(
                f"no atom has a neighbour within {max_separation} px — either "
                f"the separation is wrong or this is not a dumbbell lattice")
        delta = delta[keep]

    # +v and -v describe the same dumbbell, so fold onto one half-plane before
    # averaging or the two clusters cancel to zero.
    flip = (delta[:, 0] < 0) | ((delta[:, 0] == 0) & (delta[:, 1] < 0))
    delta[flip] *= -1
    return np.median(delta, axis=0)


def pair_atoms(positions, vector, *, tolerance: float = 0.4):
    """Pair atoms into dumbbells using a known dumbbell *vector*.

    Greedy nearest-partner matching: each atom's best candidate is the one
    closest to ``position + vector``, and an atom already claimed cannot be
    claimed again. Greedy is enough because a correct dumbbell's partner is
    unambiguous — anything closer than *tolerance* × |vector| to the wrong atom
    means the vector itself is wrong, which is worth surfacing rather than
    papering over with a global assignment.

    Parameters
    ----------
    tolerance : float
        Allowed mismatch as a fraction of the dumbbell length.

    Returns
    -------
    (pairs, unpaired)
        *pairs* is ``(M, 2)`` of indices into *positions*, first atom then its
        partner along ``+vector``. *unpaired* are the indices left over —
        edge atoms and genuine singles, which are reported rather than
        silently dropped.
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(positions, float).reshape(-1, 2)
    v = np.asarray(vector, float).ravel()
    length = float(np.hypot(*v))
    if length <= 0:
        raise ValueError("dumbbell vector has zero length")

    tree = cKDTree(pos)
    targets = pos + v
    dist, idx = tree.query(targets, k=1)

    taken = np.zeros(len(pos), bool)
    pairs = []
    # Best matches first, so a confident pair claims its partner before an
    # ambiguous one can steal it.
    for i in np.argsort(dist):
        j = int(idx[i])
        if i == j or taken[i] or taken[j]:
            continue
        if dist[i] > tolerance * length:
            continue
        taken[i] = taken[j] = True
        pairs.append((int(i), j))

    unpaired = np.flatnonzero(~taken)
    if len(unpaired):
        log.debug("%d atom(s) left unpaired out of %d", len(unpaired), len(pos))
    return np.array(pairs, int).reshape(-1, 2), unpaired


def dumbbell_properties(positions, pairs, params=None) -> dict[str, np.ndarray]:
    """Per-dumbbell measurements — one value per PAIR, not per atom.

    ``separation``
        Distance between the two atoms. The primary measurement.
    ``angle``
        Orientation in radians, wrapped to ``(-pi/2, pi/2]`` because a dumbbell
        has no head or tail — reporting it over the full circle would make
        identical dumbbells differ by pi depending on which atom was listed
        first.
    ``centre_x`` / ``centre_y``
        Midpoint, which is the position to use for a lattice-level map.
    ``intensity_ratio``
        Second atom over first, when *params* is given. Distinguishes the two
        sites of a polar structure; 1.0 means the pair is symmetric.
    """
    pos = np.asarray(positions, float).reshape(-1, 2)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    a, b = pos[pr[:, 0]], pos[pr[:, 1]]
    d = b - a

    angle = np.arctan2(d[:, 1], d[:, 0])
    # Fold to a half-turn: +v and -v are the same dumbbell.
    angle = (angle + np.pi / 2) % np.pi - np.pi / 2

    out = {
        "separation": np.hypot(d[:, 0], d[:, 1]),
        "angle": angle,
        "centre_x": (a[:, 0] + b[:, 0]) / 2,
        "centre_y": (a[:, 1] + b[:, 1]) / 2,
    }
    if params is not None:
        p = np.asarray(params, float)
        i0, i1 = p[pr[:, 0], 0], p[pr[:, 1], 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            out["intensity_ratio"] = np.where(i0 > 0, i1 / i0, np.nan)
    return out


def refine_pairs(image, positions, pairs, *, sigma: float = 2.5,
                 margin: float = 3.0, device=None, max_iter: int = 80):
    """Refine both atoms of each dumbbell **jointly**, as one two-gaussian fit.

    This is not a refinement of a refinement — it corrects a systematic error.
    Fitting each atom independently biases its centre TOWARDS its partner,
    because the partner's tail is signal the single-gaussian model can only
    explain by moving. The pair separation therefore comes out too small, and
    uniformly so, which makes it look like a calibration rather than a bug:
    measured on a synthetic dumbbell of 6.0 px with sigma 2.0, independent fits
    return **4.21 px**, a 30% underestimate. Joint fitting recovers 6.0.

    The two atoms share one box and one model of two ``Gaussian2D``
    components, so each explains the other's tail instead of absorbing it. Every
    pair is still fitted in ONE batched call — the engine does not care that the
    model now has two components.

    Returns refined ``(N, 2)`` positions (unpaired atoms keep their input) and
    the per-pair parameter table.
    """
    from spyde.fitting import ModelSpec
    from spyde.fitting.components import image_coordinates
    from spyde.fitting.engine import fit_batched
    from spyde.fitting.spec import ComponentSpec, ParameterSpec

    img = np.asarray(image, float)
    pos = np.asarray(positions, float).reshape(-1, 2)
    pr = np.asarray(pairs, int).reshape(-1, 2)
    out = pos.copy()
    if not len(pr):
        return out, np.empty((0, 10))

    a, b = pos[pr[:, 0]], pos[pr[:, 1]]
    sep = float(np.median(np.hypot(*(b - a).T)))
    # The box must hold BOTH atoms and enough tail for the fit to see where
    # each one ends; too tight and the joint fit inherits the bias it exists
    # to remove.
    box = int(2 * round(sep / 2 + margin * sigma) + 1)
    r = box // 2

    h, w = img.shape
    mid = (a + b) / 2
    cx = np.clip(np.rint(mid[:, 0]).astype(int), r, w - r - 1)
    cy = np.clip(np.rint(mid[:, 1]).astype(int), r, h - r - 1)
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    patches = img[cy[:, None, None] + oy[None],
                  cx[:, None, None] + ox[None]].reshape(len(pr), -1)

    def _comp(name):
        return ComponentSpec(kind="Gaussian2D", name=name, parameters=[
            ParameterSpec("A", 1.0, linear=True),
            ParameterSpec("centre_x", float(r)),
            ParameterSpec("centre_y", float(r)),
            ParameterSpec("sigma_x", float(sigma), bmin=0.3, bmax=float(box)),
            ParameterSpec("sigma_y", float(sigma), bmin=0.3, bmax=float(box)),
        ])

    spec = ModelSpec(components=[_comp("atom0"), _comp("atom1")])
    names = spec.parameter_names()
    start = np.broadcast_to(spec.flat_values(), (len(pr), len(names))).copy()
    peak = patches.max(1) / 2.0
    for tag, xy in (("atom0", a), ("atom1", b)):
        start[:, names.index(f"{tag}.A")] = peak * 2 * np.pi * sigma * sigma
        start[:, names.index(f"{tag}.centre_x")] = xy[:, 0] - cx + r
        start[:, names.index(f"{tag}.centre_y")] = xy[:, 1] - cy + r

    xy_grid = image_coordinates((box, box)).numpy().astype(float)
    res = fit_batched(spec, patches, xy_grid, device=device,
                      max_iter=max_iter, initial=start)

    for tag, col in (("atom0", 0), ("atom1", 1)):
        px = res.values[:, names.index(f"{tag}.centre_x")] + cx - r
        py = res.values[:, names.index(f"{tag}.centre_y")] + cy - r
        keep = (np.abs(px - pos[pr[:, col], 0]) <= r) & \
               (np.abs(py - pos[pr[:, col], 1]) <= r) & \
               np.isfinite(px) & np.isfinite(py)
        out[pr[keep, col], 0] = px[keep]
        out[pr[keep, col], 1] = py[keep]
    return out, res.values


def find_dumbbells(positions, *, vector=None, max_separation=None,
                   tolerance: float = 0.4, params=None):
    """Estimate the vector, pair the atoms, measure — the whole workflow.

    Returns ``(pairs, properties, info)``; *info* carries the vector used and
    how many atoms went unpaired, so a bad estimate is visible instead of
    showing up later as a suspiciously small dumbbell count.
    """
    v = (estimate_dumbbell_vector(positions, max_separation=max_separation)
         if vector is None else np.asarray(vector, float).ravel())
    pairs, unpaired = pair_atoms(positions, v, tolerance=tolerance)
    props = dumbbell_properties(positions, pairs, params)
    info = {"vector": v, "length": float(np.hypot(*v)),
            "n_pairs": len(pairs), "n_unpaired": int(len(unpaired)),
            "unpaired": unpaired}
    return pairs, props, info

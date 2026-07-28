"""properties.py — per-atom property maps (#79).

Each function turns the refined atom table into one value per atom, which the
UI ships through the same ``commit_result_tree`` view mechanism as the strain
components (single click shows one, cmd-click tiles several). No new display
code — see :mod:`spyde.actions.views`.

Everything here is vectorised over atoms. These are cheap compared with the
fit, but a Python loop over 10k atoms is still a visible pause, and the
neighbour search is the part that would be worst.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _neighbours(positions, k: int):
    """Indices and distances of each atom's k nearest neighbours.

    Uses a KD-tree; a brute-force (N, N) distance matrix is 800 MB at 10k atoms
    and quadratic beyond that.
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(positions, float).reshape(-1, 2)
    tree = cKDTree(pos)
    # k+1 because the first hit is always the atom itself.
    dist, idx = tree.query(pos, k=min(k + 1, len(pos)))
    return idx[:, 1:], dist[:, 1:]


def ellipticity(params) -> np.ndarray:
    """``sigma_max / sigma_min`` per atom — always >= 1.

    Expressed as a ratio of the LARGER to the smaller axis rather than
    ``sigma_x / sigma_y``, so the value does not flip below 1 when an atom
    happens to be elongated along y instead of x. An orientation-dependent
    "ellipticity" would show a spurious boundary wherever the elongation
    direction changes.
    """
    p = np.asarray(params, float)
    sx, sy = np.abs(p[:, 3]), np.abs(p[:, 4])
    big = np.maximum(sx, sy)
    small = np.minimum(sx, sy)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(small > 0, big / small, np.nan)


def ellipticity_angle(params) -> np.ndarray:
    """0 where an atom is wider in x, pi/2 where it is wider in y.

    A separate map from :func:`ellipticity` on purpose: magnitude and direction
    answer different questions, and folding them together is what makes an
    ellipticity map hard to read.
    """
    p = np.asarray(params, float)
    return np.where(np.abs(p[:, 3]) >= np.abs(p[:, 4]), 0.0, np.pi / 2)


def intensity(params) -> np.ndarray:
    """Fitted gaussian VOLUME per atom (the ``A`` parameter).

    Volume, not peak height: it is what scales with scattering power and is
    insensitive to a slightly wider or narrower fit, so it is the more stable
    of the two for comparing sites.
    """
    return np.asarray(params, float)[:, 0]


def nearest_neighbour_distance(positions, k: int = 1) -> np.ndarray:
    """Mean distance to the *k* nearest neighbours, per atom."""
    _, dist = _neighbours(positions, k)
    return dist.mean(1) if dist.ndim > 1 else dist


def displacement_from_ideal(positions, *, k: int = 4) -> np.ndarray:
    """Each atom's offset from the centroid of its *k* nearest neighbours.

    Returns ``(N, 2)`` in ``(dx, dy)``. This is a *local* reference, so it
    measures a genuine local distortion and is immune to sample drift or a
    tilted scan — a globally-fitted ideal lattice would report both as
    displacement.
    """
    pos = np.asarray(positions, float).reshape(-1, 2)
    idx, _ = _neighbours(pos, k)
    return pos - pos[idx].mean(1)


def displacement_magnitude(positions, *, k: int = 4) -> np.ndarray:
    d = displacement_from_ideal(positions, k=k)
    return np.hypot(d[:, 0], d[:, 1])


def to_map(values, positions, shape, *, fill=np.nan) -> np.ndarray:
    """Scatter per-atom values onto an image-shaped array.

    Nearest-pixel placement, so the result lines up with the image the atoms
    were found in and can be shown beside it. Atoms outside the shape are
    dropped rather than wrapped.
    """
    pos = np.asarray(positions, float).reshape(-1, 2)
    vals = np.asarray(values, float).ravel()
    if len(vals) != len(pos):
        raise ValueError(f"{len(vals)} values for {len(pos)} atoms")
    out = np.full(tuple(shape), fill, float)
    ix = np.rint(pos[:, 0]).astype(int)
    iy = np.rint(pos[:, 1]).astype(int)
    inside = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
    out[iy[inside], ix[inside]] = vals[inside]
    return out


def property_maps(positions, params, *, k: int = 4) -> dict[str, np.ndarray]:
    """Every per-atom property in one dict, keyed for the view chips (#79).

    The keys are what the user sees on the toggle, so they are spelled for a
    human rather than after the function names.
    """
    return {
        "Ellipticity": ellipticity(params),
        "Intensity": intensity(params),
        "NN distance": nearest_neighbour_distance(positions),
        "Displacement": displacement_magnitude(positions, k=k),
    }

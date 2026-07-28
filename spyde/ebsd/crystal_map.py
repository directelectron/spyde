"""crystal_map.py — indexing results -> orix CrystalMap -> IPF colours (#73).

The display half of Wave 3 is nearly free, and deliberately so. SpyDE already
depends on **orix** and already has IPF views, orientation maps and a 3D IPF
toolbar from the 4D-STEM work, so the job here is to hand the EBSD result over
in the form that machinery already speaks — not to build a second orientation
display beside the first.

Two things this module does that the existing 4D-STEM path does not need:

``orientation_similarity_map``
    A quality metric peculiar to dictionary indexing: how much two neighbouring
    positions agree about their *ranked list* of best matches. Unlike a raw
    correlation score it is sensitive to indexing that is confidently wrong —
    a position that matched well but disagrees with everything around it stands
    out, which a score map cannot show.

``merge_phases``
    Multi-phase indexing runs one dictionary per phase, so the maps have to be
    combined by score afterwards.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Space groups for the phases most likely to be indexed first. A caller can
# always pass an explicit orix Phase; this is only a convenience.
COMMON_PHASES = {
    "fcc": 225,     # Fm-3m — Al, Ni, Cu, austenite
    "bcc": 229,     # Im-3m — ferrite, W, Mo
    "dc": 227,      # Fd-3m — Si, Ge, diamond
    "hcp": 194,     # P6_3/mmc — Ti, Mg, Zn
}


def _phase(phase=None, name: str = "phase", space_group: int = 225):
    from orix.crystal_map import Phase
    if phase is not None:
        return phase
    return Phase(name=name, space_group=int(space_group))


def to_crystal_map(euler, nav_shape=None, *, phase=None, space_group: int = 225,
                   phase_name: str = "phase", scores=None, step: float = 1.0):
    """Euler angles -> an orix :class:`~orix.crystal_map.CrystalMap`.

    Parameters
    ----------
    euler : array (..., 3)
        Bunge angles in radians — e.g. ``RefinementResult.euler`` or
        ``IndexingResult.orientations(dictionary_euler)``.
    scores : array, optional
        Per-position match quality, stored as the map's ``prop["scores"]`` so
        the display and :func:`merge_phases` can use it.
    step : float
        Scan step size, used for the x/y coordinates.
    """
    from orix.crystal_map import CrystalMap, PhaseList
    from orix.quaternion import Rotation

    eul = np.asarray(euler, float)
    if nav_shape is None:
        nav_shape = eul.shape[:-1]
    flat = eul.reshape(-1, 3)

    rot = Rotation.from_euler(flat)
    ph = _phase(phase, phase_name, space_group)

    if len(nav_shape) == 2:
        ny, nx = nav_shape
        yy, xx = np.mgrid[0:ny, 0:nx].astype(float) * float(step)
        x, y = xx.ravel(), yy.ravel()
    else:
        x = np.arange(len(flat), dtype=float) * float(step)
        y = None

    props = {"scores": np.asarray(scores, float).ravel()} if scores is not None else None
    return CrystalMap(rotations=rot, phase_id=np.zeros(len(flat), int),
                      x=x, y=y, phase_list=PhaseList([ph]), prop=props)


def ipf_colors(euler, nav_shape=None, *, phase=None, space_group: int = 225,
               direction=None) -> np.ndarray:
    """IPF-Z RGB for each orientation -> ``(..., 3)`` float in [0, 1].

    This is what feeds SpyDE's existing orientation display: an ``(H, W, 3)``
    RGB array is exactly what ``commit.commit_result_tree`` already accepts as
    an RGB primary (it skips contrast locking for those).
    """
    from orix.plot import IPFColorKeyTSL
    from orix.quaternion import Rotation
    from orix.vector import Vector3d

    eul = np.asarray(euler, float)
    if nav_shape is None:
        nav_shape = eul.shape[:-1]
    rot = Rotation.from_euler(eul.reshape(-1, 3))
    ph = _phase(phase, "phase", space_group)
    key = IPFColorKeyTSL(ph.point_group,
                         direction=direction or Vector3d.zvector())
    rgb = np.asarray(key.orientation2color(rot), float)
    return np.clip(rgb, 0.0, 1.0).reshape(tuple(nav_shape) + (3,))


def orientation_similarity_map(indices, nav_shape=None) -> np.ndarray:
    """How much each position's ranked match list agrees with its neighbours.

    ``indices`` is ``IndexingResult.indices`` — ``(P, k)``, best first. For each
    position the metric is the mean overlap between its top-k dictionary
    indices and each neighbour's, normalised to [0, 1].

    Why this rather than the correlation score: a score map shows how well a
    pattern matched *something*, and a confidently WRONG index scores just as
    highly as a right one. Agreement with the neighbourhood is what exposes it,
    so this is the map that finds bad indexing rather than bad patterns.

    Needs ``keep > 1`` from :func:`~spyde.ebsd.indexing.dictionary_index`;
    with k=1 it degenerates to "does my single best match equal my neighbour's",
    which is a legitimate but much coarser signal.
    """
    idx = np.asarray(indices)
    if idx.ndim != 2:
        raise ValueError("indices must be (P, k) from IndexingResult.indices")
    P, k = idx.shape
    if nav_shape is None:
        raise ValueError("nav_shape is required to find neighbours")
    ny, nx = nav_shape
    if ny * nx != P:
        raise ValueError(f"nav_shape {nav_shape} does not match {P} positions")

    grid = idx.reshape(ny, nx, k)
    total = np.zeros((ny, nx), float)
    count = np.zeros((ny, nx), float)

    # Set overlap per neighbour pair, vectorised over the whole map: compare
    # every ranked index against every one of the neighbour's, then count hits.
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ys = slice(max(0, dy), ny + min(0, dy))
        xs = slice(max(0, dx), nx + min(0, dx))
        ys2 = slice(max(0, -dy), ny + min(0, -dy))
        xs2 = slice(max(0, -dx), nx + min(0, -dx))
        a = grid[ys, xs][..., :, None]        # (h, w, k, 1)
        b = grid[ys2, xs2][..., None, :]      # (h, w, 1, k)
        overlap = (a == b).any(-1).sum(-1) / float(k)
        total[ys, xs] += overlap
        count[ys, xs] += 1.0
    return total / np.maximum(count, 1.0)


def merge_phases(maps, scores, nav_shape=None):
    """Combine per-phase indexing runs into one labelled result.

    Multi-phase indexing runs one dictionary per phase; the winner at each
    position is simply the phase whose match scored highest.

    Parameters
    ----------
    maps : sequence of array (..., 3)
        Euler angles from each phase's run, in the same order as *scores*.
    scores : sequence of array (...,)
        Best-match score per position for each phase.

    Returns
    -------
    (euler, phase_id, best_score)
    """
    eulers = [np.asarray(m, float) for m in maps]
    raw_scores = [np.asarray(s, float).ravel() for s in scores]
    # Validate BEFORE stacking: np.stack raises "all input arrays must have the
    # same shape", which says nothing about which phase disagreed or why.
    sizes = {e.reshape(-1, 3).shape[0] for e in eulers} | {s.size for s in raw_scores}
    if len(sizes) != 1:
        raise ValueError(
            f"every phase map must cover the same positions, got {sorted(sizes)}")
    if len(eulers) != len(raw_scores):
        raise ValueError(f"{len(eulers)} phase maps but {len(raw_scores)} "
                         f"score arrays")
    sc = np.stack(raw_scores)                                       # (n_phase, P)

    winner = sc.argmax(0)                                            # (P,)
    stacked = np.stack([e.reshape(-1, 3) for e in eulers])           # (n, P, 3)
    P = stacked.shape[1]
    euler = stacked[winner, np.arange(P)]
    best = sc[winner, np.arange(P)]

    if nav_shape is not None:
        euler = euler.reshape(tuple(nav_shape) + (3,))
        winner = winner.reshape(nav_shape)
        best = best.reshape(nav_shape)
    return euler, winner, best

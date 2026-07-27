"""indexing.py — dictionary indexing as one big matmul.

The insight that makes this fast: normalised cross-correlation between an
experimental pattern and a dictionary pattern is, once both are zero-mean and
unit-norm, **exactly a dot product**. So indexing P experimental patterns
against D dictionary patterns is a single matrix multiply ``E @ Dᵀ`` followed
by a top-k — the operation GPUs are built for.

    E  (P, K)   P experimental patterns, K = detector pixels
    D  (D, K)   D dictionary patterns
    S  (P, D)   similarity, = E @ Dᵀ after normalisation

The catch is that ``S`` is enormous and must never exist: 65k experimental x
100k dictionary in float32 is 26 GB. So both axes are **chunked with a running
top-k** — each tile of ``S`` is reduced against the best-so-far and thrown
away, so peak memory is one tile plus the (P, k) result.

kikuchipy does this with dask; doing it as a chunked matmul on the GPU is the
point of the wave. Its scores are the reference — ``test_ebsd_indexing.py``
checks agreement against ``kikuchipy.indexing`` when that extra is installed,
and against known ground-truth orientations always.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

# Elements per similarity tile. 2**26 float32 = 268 MB — big enough to keep a
# GPU busy, small enough that the (P, D) product never materialises.
_TILE_ELEMENTS = 1 << 26


@dataclass
class IndexingResult:
    """Best matches per experimental pattern."""

    indices: np.ndarray        # (P, k) dictionary indices, best first
    scores: np.ndarray         # (P, k) similarity in [-1, 1]
    device: str

    @property
    def best(self) -> np.ndarray:
        return self.indices[:, 0]

    @property
    def best_score(self) -> np.ndarray:
        return self.scores[:, 0]

    def orientations(self, dictionary_euler: np.ndarray,
                     nav_shape=None) -> np.ndarray:
        """Euler angles of the best match per position, shaped to the scan."""
        eul = np.asarray(dictionary_euler, float)[self.best]
        return eul.reshape(tuple(nav_shape) + (3,)) if nav_shape else eul


def _normalise(a, eps=1e-12):
    """Zero-mean, unit-norm along the last axis — after which a dot product IS
    the normalised cross-correlation (Pearson) coefficient."""
    a = a - a.mean(-1, keepdim=True)
    return a / a.norm(dim=-1, keepdim=True).clamp_min(eps)


def dictionary_index(patterns, dictionary, *, keep: int = 1, device=None,
                     dtype="float32", tile_elements: int = _TILE_ELEMENTS,
                     progress: Callable[[int, int], None] | None = None):
    """Match every pattern against every dictionary entry.

    Parameters
    ----------
    patterns : array (..., H, W) or (P, K)
        Experimental patterns. Leading dimensions are flattened to ``P``.
    dictionary : array (D, H, W) or (D, K)
        Simulated patterns, e.g. from
        :func:`spyde.data.synthetic.simulate_patterns`.
    keep : int
        How many matches to keep per pattern. >1 is what
        ``orientation_similarity_map`` needs (#73).
    dtype : str
        ``float32`` by default and deliberately: NCC is a bounded, well-
        conditioned quantity and the matmul is memory-bound, so float64 would
        halve throughput to protect digits that do not affect the ranking.

    Returns
    -------
    IndexingResult
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tdtype = getattr(torch, dtype)

    exp = np.asarray(patterns)
    dic = np.asarray(dictionary)

    # Compare pixel counts BEFORE reshaping. Reshaping the experimental stack
    # to the DICTIONARY's pixel count quietly succeeds whenever the sizes share
    # a factor — a 40x40 scan against a 20x20 dictionary reshapes to 4x as many
    # "patterns" and indexes garbage instead of raising.
    dic_k = int(np.prod(dic.shape[1:]))
    exp_k = int(np.prod(exp.shape[-2:] if exp.ndim > 2 else exp.shape[-1:]))
    if exp_k != dic_k:
        raise ValueError(f"pattern has {exp_k} pixels but dictionary entries "
                         f"have {dic_k}")

    nav_shape = exp.shape[:-2] if exp.ndim > 2 else exp.shape[:-1]
    E = exp.reshape(-1, exp_k)
    D = dic.reshape(-1, dic_k)
    P, Dn = E.shape[0], D.shape[0]
    keep = int(min(keep, Dn))

    # The dictionary is normalised ONCE and kept resident — it is reused by
    # every tile, and re-normalising it per tile would dominate the matmul.
    d_t = _normalise(torch.as_tensor(D, dtype=tdtype, device=device))

    # Tile both axes so no (P, D) block is ever larger than the budget.
    d_chunk = max(1, min(Dn, tile_elements // max(P, 1)))
    p_chunk = max(1, min(P, tile_elements // max(min(d_chunk, Dn), 1)))

    out_scores = np.empty((P, keep), np.float32)
    out_index = np.empty((P, keep), np.int64)

    for p0 in range(0, P, p_chunk):
        p1 = min(P, p0 + p_chunk)
        e_t = _normalise(torch.as_tensor(E[p0:p1], dtype=tdtype, device=device))

        best_s = torch.full((p1 - p0, keep), -2.0, dtype=tdtype, device=device)
        best_i = torch.zeros((p1 - p0, keep), dtype=torch.int64, device=device)

        for d0 in range(0, Dn, d_chunk):
            d1 = min(Dn, d0 + d_chunk)
            sim = e_t @ d_t[d0:d1].T                       # (p, d) tile
            k = min(keep, d1 - d0)
            s, i = torch.topk(sim, k, dim=1)
            # Merge this tile's best with the running best, then re-reduce —
            # so the full (P, D) similarity never has to exist at once.
            cat_s = torch.cat([best_s, s], 1)
            cat_i = torch.cat([best_i, i + d0], 1)
            best_s, order = torch.topk(cat_s, keep, dim=1)
            best_i = torch.gather(cat_i, 1, order)

        out_scores[p0:p1] = best_s.detach().cpu().numpy()
        out_index[p0:p1] = best_i.detach().cpu().numpy()
        if progress is not None:
            try:
                progress(p1, P)
            except Exception as e:                          # pragma: no cover
                log.debug("indexing progress callback failed: %s", e)

    log.debug("indexed %d patterns against %d dictionary entries on %s "
              "(tiles %dx%d)", P, Dn, device, p_chunk, d_chunk)
    result = IndexingResult(out_index, out_scores, device)
    result.nav_shape = tuple(nav_shape)
    return result


def sample_orientations(step_deg: float = 5.0, *, phi1=(0.0, 360.0),
                        Phi=(0.0, 90.0), phi2=(0.0, 90.0)) -> np.ndarray:
    """A regular Euler-space grid -> ``(N, 3)`` radians.

    Deliberately simple and NOT equal-area: a proper dictionary uses a uniform
    SO(3) sampling (kikuchipy/orix do this correctly, and #69 wires that up).
    This exists so indexing can be developed and tested without the extra
    installed, and the default ranges cover the cubic fundamental zone.
    """
    a = np.deg2rad(np.arange(phi1[0], phi1[1], step_deg))
    b = np.deg2rad(np.arange(Phi[0], Phi[1] + 1e-9, step_deg))
    c = np.deg2rad(np.arange(phi2[0], phi2[1] + 1e-9, step_deg))
    grid = np.meshgrid(a, b, c, indexing="ij")
    return np.stack([g.ravel() for g in grid], -1)

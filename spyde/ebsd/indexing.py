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

from spyde.device_lock import accelerator_lock
from spyde.ebsd._device import default_device, resolve_dtype

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
                     pattern_chunk: int | None = None,
                     progress: Callable[[int, int], None] | None = None,
                     stopped_flag=None):
    """Match every pattern against every dictionary entry.

    Parameters
    ----------
    patterns : array (..., H, W) or (P, K)
        Experimental patterns. Leading dimensions are flattened to ``P``.
    dictionary : array (D, H, W) or (D, K), or a :class:`SinglePatternIndexer`
        Simulated patterns, e.g. from :func:`simulate_dictionary`. Passing an
        indexer reuses the dictionary it already holds normalised on the
        device — which is what the interactive path does, having built one for
        the live preview: re-normalising ``D x K`` is the single most expensive
        step of a small index and there is no reason to pay it twice.
    keep : int
        How many matches to keep per pattern. >1 is what
        ``orientation_similarity_map`` needs (#73).
    dtype : str
        ``float32`` by default and deliberately: NCC is a bounded, well-
        conditioned quantity and the matmul is memory-bound, so float64 would
        halve throughput to protect digits that do not affect the ranking.
    pattern_chunk : int, optional
        Cap on the experimental tile size, on top of the *tile_elements*
        budget. Callers that already stream the scan in navigation blocks
        (``spyde.actions.ebsd_action``) do not need it; it is here for a caller
        holding a large in-memory stack that wants a tighter bound.
    stopped_flag : list[bool], optional
        Polled between chunks; a truthy first element abandons the run and
        returns None. This is how closing the window cancels an index.

    Returns
    -------
    IndexingResult, or None if cancelled.
    """
    import torch

    resident = getattr(dictionary, "normalised", None)   # SinglePatternIndexer
    if resident is not None:
        device = device or dictionary.device
    device = device or default_device()
    dtype = resolve_dtype(device, dtype)
    tdtype = getattr(torch, dtype)

    exp = np.asarray(patterns)

    # Compare pixel counts BEFORE reshaping. Reshaping the experimental stack
    # to the DICTIONARY's pixel count quietly succeeds whenever the sizes share
    # a factor — a 40x40 scan against a 20x20 dictionary reshapes to 4x as many
    # "patterns" and indexes garbage instead of raising.
    if resident is not None:
        dic_k = int(dictionary.k)
    else:
        dic = np.asarray(dictionary)
        dic_k = int(np.prod(dic.shape[1:]))
    exp_k = int(np.prod(exp.shape[-2:] if exp.ndim > 2 else exp.shape[-1:]))
    if exp_k != dic_k:
        raise ValueError(f"pattern has {exp_k} pixels but dictionary entries "
                         f"have {dic_k}")

    nav_shape = exp.shape[:-2] if exp.ndim > 2 else exp.shape[:-1]
    E = exp.reshape(-1, exp_k)

    # The dictionary is normalised ONCE and kept resident — it is reused by
    # every tile, and re-normalising it per tile would dominate the matmul.
    # Under the lock like everything else that touches the device: this uploads
    # D x K floats and reduces over them, which is the single largest transfer
    # the function makes.
    if resident is not None:
        d_t = resident
    else:
        with accelerator_lock(device):
            d_t = _normalise(torch.as_tensor(dic.reshape(-1, dic_k),
                                             dtype=tdtype, device=device))
    tdtype = d_t.dtype
    P, Dn = E.shape[0], int(d_t.shape[0])
    keep = int(min(keep, Dn))

    # Tile both axes so no (P, D) block is ever larger than the budget.
    d_chunk = max(1, min(Dn, tile_elements // max(P, 1)))
    p_chunk = max(1, min(P, tile_elements // max(min(d_chunk, Dn), 1)))
    if pattern_chunk:
        p_chunk = max(1, min(p_chunk, int(pattern_chunk)))

    out_scores = np.empty((P, keep), np.float32)
    out_index = np.empty((P, keep), np.int64)

    for p0 in range(0, P, p_chunk):
        if stopped_flag is not None and stopped_flag[0]:
            log.debug("dictionary indexing cancelled at %d/%d patterns", p0, P)
            return None
        p1 = min(P, p0 + p_chunk)
        # Held per PATTERN TILE, not around the whole index. On MPS this shares
        # one process-wide lock with every other torch user (device_lock.py), and
        # a full index runs for minutes — so releasing between tiles bounds any
        # concurrent preview's wait to a single tile. The `stopped_flag` check
        # above stays OUTSIDE the lock so cancelling never waits on the device.
        with accelerator_lock(device):
            e_t = _normalise(torch.as_tensor(E[p0:p1], dtype=tdtype,
                                             device=device))

            best_s = torch.full((p1 - p0, keep), -2.0, dtype=tdtype,
                                device=device)
            best_i = torch.zeros((p1 - p0, keep), dtype=torch.int64,
                                 device=device)

            for d0 in range(0, Dn, d_chunk):
                d1 = min(Dn, d0 + d_chunk)
                sim = e_t @ d_t[d0:d1].T                   # (p, d) tile
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


class SinglePatternIndexer:
    """The dictionary, normalised once and kept resident, for live indexing.

    :func:`dictionary_index` is the batch door: it normalises the dictionary,
    matches a whole scan and drops everything. Under the crosshair the shape of
    the problem is the opposite — ONE pattern at a time, again on every
    navigator move — so re-normalising ``D x K`` per move would cost far more
    than the match itself (25k entries of a 60x60 detector is ~370 MB; the
    match is a single mat-vec).

    So the wizard builds one of these when it builds the dictionary and the
    band overlay calls :meth:`best` per position. Same normalisation, same
    scores as the batch path — a dot product of zero-mean unit-norm vectors.
    """

    def __init__(self, dictionary, euler, *, device=None, dtype="float32"):
        import torch

        self.device = device or default_device()
        tdtype = getattr(torch, resolve_dtype(self.device, dtype))
        dic = np.asarray(dictionary)
        self.euler = np.asarray(euler, float).reshape(-1, 3)
        self.pattern_shape = tuple(dic.shape[1:])
        self.k = int(np.prod(self.pattern_shape))
        with accelerator_lock(self.device):
            self._d = _normalise(torch.as_tensor(
                dic.reshape(-1, self.k), dtype=tdtype, device=self.device))
        self._dtype = tdtype

    def __len__(self) -> int:
        return int(self._d.shape[0])

    @property
    def normalised(self):
        """The zero-mean unit-norm dictionary, resident on ``device``.
        :func:`dictionary_index` takes this instead of re-normalising."""
        return self._d

    def best(self, pattern):
        """Best-matching orientation for one pattern -> ``(euler (3,), score)``."""
        import torch

        p = np.asarray(pattern, float).reshape(-1)
        if p.size != self.k:
            raise ValueError(f"pattern has {p.size} pixels but the dictionary "
                             f"has {self.k}")
        # This is the LIVE path — it fires on every navigator move while the
        # band overlay is up, so it is exactly the concurrent submitter the
        # device lock exists to serialise against a running index or refine.
        with accelerator_lock(self.device), torch.no_grad():
            e = _normalise(torch.as_tensor(p, dtype=self._dtype,
                                           device=self.device))
            sim = self._d @ e
            score, idx = torch.max(sim, 0)
            return self.euler[int(idx.item())], float(score.item())


def simulate_dictionary(euler, detector=(60, 60), pc=(0.5, 0.5, 0.55), *,
                        reflectors=None, background_sigma=None, device=None,
                        chunk: int = 4096,
                        progress: Callable[[int, int], None] | None = None,
                        stopped_flag=None):
    """Simulate one pattern per orientation -> ``(D, H, W)`` float32.

    The dictionary side of indexing. Uses the same torch
    :class:`~spyde.ebsd.refine.BandSimulator` the refinement optimises through,
    so the dictionary and the refined patterns are the same function — and the
    same :class:`~spyde.ebsd.bands.Reflectors` the live overlay projects, so
    the drawn lines are the drawn bands' centres.

    ``spyde.ebsd.bands.simulate_patterns`` does this in numpy for a handful of
    orientations; a dictionary is thousands, so it goes through torch in
    chunks — batched over orientations, with the chunk bounding the largest
    ``(chunk, B, K)`` intermediate rather than materialising ``(D, B, K)``.

    Pass *background_sigma* to high-pass the simulated patterns exactly as the
    experimental ones were corrected. Skipping it is the classic way to get
    mediocre scores that look like bad indexing — see
    :meth:`~spyde.ebsd.refine.BandSimulator._high_pass`.
    """
    import torch
    from spyde.ebsd.refine import BandSimulator

    device = device or default_device()
    eul = np.atleast_2d(np.asarray(euler, float))
    sim = BandSimulator(detector, pc, reflectors=reflectors,
                        background_sigma=background_sigma, device=device)
    dy, dx = int(detector[0]), int(detector[1])
    out = np.empty((len(eul), dy, dx), np.float32)
    for lo in range(0, len(eul), int(chunk)):
        if stopped_flag is not None and stopped_flag[0]:
            return None
        hi = min(len(eul), lo + int(chunk))
        # Per chunk, so a long simulation hands the device back — same rule as
        # the indexing tiles above.
        with accelerator_lock(device), torch.no_grad():
            ang = torch.as_tensor(eul[lo:hi], dtype=torch.float32,
                                  device=device)
            out[lo:hi] = sim(ang).reshape(-1, dy, dx).detach().cpu().numpy()
        if progress is not None:
            try:
                progress(hi, len(eul))
            except Exception as e:                          # pragma: no cover
                log.debug("dictionary progress callback failed: %s", e)
    return out


def sample_orientations(step_deg: float = 5.0, *, point_group=None,
                        phi1=(0.0, 360.0), Phi=(0.0, 90.0),
                        phi2=(0.0, 90.0)) -> np.ndarray:
    """Orientations to build a dictionary from -> ``(N, 3)`` Bunge radians.

    With a *point_group* this is orix's uniform sampling of that group's
    FUNDAMENTAL ZONE — every distinct crystal orientation once, none of them
    twice. Prefer it: the Euler grid below is neither equal-area (it bunches
    towards Phi = 0) nor free of symmetric duplicates, and the difference is
    not academic — at 5 degrees for m-3m it is ~6.6k orientations against the
    grid's ~26k, so the dictionary is a quarter of the size, a quarter of the
    simulation time, and a quarter of the cost of every live match.

    Without one it falls back to a plain Euler grid over the given ranges, so
    indexing still works with nothing but numpy — which is what lets the rest
    of this module be developed and tested without the ``ebsd`` extra.
    """
    if point_group is not None:
        try:
            from orix.sampling import get_sample_fundamental
            rot = get_sample_fundamental(resolution=float(step_deg),
                                         point_group=point_group)
            return np.asarray(rot.to_euler(), float).reshape(-1, 3)
        except Exception as e:
            log.warning("orix fundamental-zone sampling failed (%s) — falling "
                        "back to an Euler grid", e)

    a = np.deg2rad(np.arange(phi1[0], phi1[1], step_deg))
    b = np.deg2rad(np.arange(Phi[0], Phi[1] + 1e-9, step_deg))
    c = np.deg2rad(np.arange(phi2[0], phi2[1] + 1e-9, step_deg))
    grid = np.meshgrid(a, b, c, indexing="ij")
    return np.stack([g.ravel() for g in grid], -1)

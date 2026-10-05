"""The symmetry stage: refine every disk of a frame using the frame's other disks.

Find Vectors (neural) runs three small models in a row. The detector finds the
disks; the centre stage (:mod:`spyde.models.centre_refine`) places each one on
its own crop; this stage looks at all of a frame's disks together. A crystal's
pattern is not a bag of independent spots: ``g`` and ``-g`` sit symmetrically
about the direct beam, and the spots lie on a lattice. A disk the crop alone
placed badly can be corrected by its partners.

Amorphous data has no such structure, so this stage is skipped there (the
wizard's Sample: Amorphous); detection and the centre stage still run.

A symmetry refiner is any object with

    refine_frames(frames: Sequence[FrameDisks], spot_radius) -> list of (positions, sigma)

called once per chunk with every frame's disks — positions only, never pixels.
It returns, per frame, the ``(n, 2)`` refined ``[y, x]`` positions and the
``(n,)`` positional uncertainty in pixels, or ``None`` for the uncertainty when
it has none (the caller keeps its own). A refiner declines a disk by returning
NaN for it. It may carry a ``device`` (the stage takes the accelerator lock for
it) and a ``min_spot_radius``.

:func:`refine_frames` applies a refiner the way the centre stage applies its
own: a declined disk, and any disk the refiner moves by half a spot radius or
more, keeps its incoming position.

:class:`FriedelRefiner` is the classical baseline the learned model must beat.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from .centre_refine import MAX_SHIFT_FRACTION

log = logging.getLogger(__name__)

SYMMETRY_OFF = "off"
SYMMETRY_FRIEDEL = "friedel"

SAMPLE_CRYSTALLINE = "crystalline"
SAMPLE_AMORPHOUS = "amorphous"
SAMPLE_TYPES = (SAMPLE_CRYSTALLINE, SAMPLE_AMORPHOUS)


@dataclass
class FrameDisks:
    """One frame's disks as the symmetry stage sees them, detector pixels."""

    positions: np.ndarray                     #: (n, 2) [y, x]
    sigma: np.ndarray                         #: (n,) positional uncertainty, NaN if unknown
    confidence: np.ndarray                    #: (n,) detector confidence, NaN if unknown
    beam: Optional[np.ndarray] = None         #: (2,) direct-beam position, if known
    embeddings: Optional[np.ndarray] = None   #: (n, d) per-disk features from earlier stages


class SymmetryRefiner(Protocol):
    def refine_frames(self, frames: Sequence[FrameDisks], spot_radius: float
                      ) -> list: ...


def beam_from_records(records: np.ndarray, intensity_column: int = 2) -> Optional[np.ndarray]:
    """The direct beam of a frame: its brightest disk. ``records`` are the
    stored ``[y, x, intensity, ...]`` peak rows."""
    if len(records) == 0:
        return None
    return np.asarray(records[np.nanargmax(records[:, intensity_column]), :2], np.float64)


def refine_frames(frames: Sequence[FrameDisks], spot_radius: float, refiner):
    """Run ``refiner`` over a chunk's frames. Returns ``(positions, sigmas)``:
    per frame the refined ``(n, 2)`` float32 positions and the refiner's
    ``(n,)`` uncertainty (NaN where the input was kept), or ``None`` for
    ``sigmas`` when the refiner reports none."""
    from spyde.device_lock import accelerator_lock

    radius = float(spot_radius)
    positions = [np.asarray(f.positions, np.float32).reshape(-1, 2).copy() for f in frames]
    if radius < float(getattr(refiner, "min_spot_radius", 0.0)) or not any(len(p) for p in positions):
        return positions, None
    with accelerator_lock(getattr(refiner, "device", None)):
        results = refiner.refine_frames(frames, radius)
    sigmas: list = []
    has_sigma = True
    for index, (refined, sigma) in enumerate(results):
        incoming = positions[index]
        refined = np.asarray(refined, np.float64).reshape(-1, 2)
        shift = np.hypot(*(refined - incoming).T)
        accepted = np.isfinite(refined).all(1) & (shift < MAX_SHIFT_FRACTION * radius)
        incoming[accepted] = refined[accepted]
        if sigma is None:
            has_sigma = False
            sigmas.append(None)
        else:
            kept = np.full(len(incoming), np.nan, np.float32)
            kept[accepted] = np.asarray(sigma, np.float32)[accepted]
            sigmas.append(kept)
    return positions, (sigmas if has_sigma else None)


class FriedelRefiner:
    """Baseline: make every Friedel pair exactly symmetric about the beam centre.

    Per frame:

    1. Pair. Starting from the direct beam, each disk's partner is the disk
       nearest its mirror ``2c - p``. A pair is kept when the two are each
       other's nearest mirror and closer than ``pair_tolerance x R``. Disks
       within ``2R`` of the centre (the beam itself) are left out.
    2. Fit the centre ``c``: the sigma-weighted mean of the pair midpoints,
       re-pairing around it (``iterations`` times).
    3. Symmetrise. For a pair ``i, j`` with weights ``w = 1 / sigma²``,
       ``g = (w_i (p_i - c) - w_j (p_j - c)) / (w_i + w_j)``; ``p_i = c + g``,
       ``p_j = c - g``. Both get ``sigma = 1 / sqrt(w_i + w_j)``. Without sigma
       the two weigh the same and no sigma is reported.

    An unpaired disk is declined. This is the bar a learned symmetry model has
    to beat: it uses only the Friedel relation, not the lattice, and trusts it
    exactly, so dynamical intensities or a tilted pattern bias it."""

    pair_tolerance = 0.3
    iterations = 3

    def refine_frames(self, frames, spot_radius):
        return [self._frame(frame, float(spot_radius)) for frame in frames]

    def _pairs(self, positions, centre, radius):
        relative = positions - centre
        mirror = centre - relative
        distance = np.hypot(mirror[:, None, 0] - positions[None, :, 0],
                            mirror[:, None, 1] - positions[None, :, 1])
        np.fill_diagonal(distance, np.inf)
        far = np.hypot(*relative.T) >= 2.0 * radius
        distance[~far] = np.inf
        distance[:, ~far] = np.inf
        partner = distance.argmin(1)
        index = np.arange(len(positions))
        mutual = partner[partner] == index
        close = distance[index, partner] < max(1.0, self.pair_tolerance * radius)
        keep = mutual & close & (index < partner)
        return index[keep], partner[keep]

    def _frame(self, frame: FrameDisks, radius):
        positions = np.asarray(frame.positions, np.float64).reshape(-1, 2)
        count = len(positions)
        result = np.full((count, 2), np.nan)
        sigma_out = np.full(count, np.nan, np.float32)
        if count < 3:
            return result, sigma_out
        sigma = np.asarray(frame.sigma, np.float64).reshape(-1)
        weighted = np.isfinite(sigma).all() and (sigma > 0).all()
        weight = 1.0 / sigma ** 2 if weighted else np.ones(count)
        centre = (np.asarray(frame.beam, np.float64) if frame.beam is not None
                  else positions.mean(0))
        first, second = self._pairs(positions, centre, radius)
        for _ in range(self.iterations):
            if len(first) == 0:
                return result, sigma_out
            pair_weight = 1.0 / (1.0 / weight[first] + 1.0 / weight[second])
            midpoints = 0.5 * (positions[first] + positions[second])
            centre = (pair_weight[:, None] * midpoints).sum(0) / pair_weight.sum()
            first, second = self._pairs(positions, centre, radius)
        if len(first) == 0:
            return result, sigma_out
        total = weight[first] + weight[second]
        g = (weight[first, None] * (positions[first] - centre)
             - weight[second, None] * (positions[second] - centre)) / total[:, None]
        result[first] = centre + g
        result[second] = centre - g
        if weighted:
            combined = 1.0 / np.sqrt(total)
            sigma_out[first] = combined
            sigma_out[second] = combined
        return result, sigma_out


def symmetry_refiner_for(choice: Optional[str], device=None):
    """The refiner a Find Vectors ``symmetry_refiner`` value names: ``None`` when
    off (empty or ``"off"``), the Friedel baseline, or a ``"kind": "symmetry"``
    model from the registry by id. A model that fails to load logs a warning and
    gives ``None``, so detection still completes."""
    if not choice or choice == SYMMETRY_OFF:
        return None
    if choice == SYMMETRY_FRIEDEL:
        return FriedelRefiner()
    from . import registry
    try:
        return registry.get_symmetry_refiner(choice, device)
    except Exception as error:
        log.warning("[models] symmetry refiner %r unavailable (%s); skipping the "
                    "symmetry stage", choice, error)
        return None

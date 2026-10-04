"""The centre stage: re-place each detected disk on the raw frame at native resolution.

The detector finds disks on a rescaled, locally normalised working image, so its
centre is only as good as that image. This stage looks again at the raw counts:
it cuts a window around every detection, hands a chunk's windows to a refiner
as one batch, and moves each vector to the refined centre.

A refiner is any object with

    crop_half_width(spot_radius) -> int
    refine(crops, centres, spot_radius) -> (centres, sigma)

``crops`` is ``(M, S, S)`` float32 raw counts with ``S = 2 * crop_half_width + 1``,
cut around each detection's nearest pixel; ``centres`` is ``(M, 2)`` ``[y, x]``,
the detections in crop pixels. It returns the refined ``(M, 2)`` centres in the
same crop pixels and an ``(M,)`` positional uncertainty in pixels, or ``None``
for the uncertainty when it has none (the caller keeps its own). A refiner
declines a disk by returning NaN for it.

The stage keeps the detection's centre for a declined disk and for any centre
that moves more than half a spot radius: no refiner sees enough of the frame to
justify a larger move, and a refiner that wandered onto a neighbour is worse
than the detection it replaced.

:class:`MaskCentroidRefiner` is the classical baseline, and the bar a network
refiner (:mod:`spyde.models.centre_network`) has to clear.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Protocol, Sequence

import numpy as np

log = logging.getLogger(__name__)

CENTRE_DECODE = "decode"                  # no refinement: the detector's own centre
CENTRE_MASK_CENTROID = "mask-centroid"

# The largest move the stage accepts, as a fraction of the spot radius.
MAX_SHIFT_FRACTION = 0.5

# Crops refined per batch. A chunk can hold 10^5 disks; this bounds the crop
# stack (and a network's activations) whatever the chunk size.
DEFAULT_BATCH = 4096


class CentreRefiner(Protocol):
    def crop_half_width(self, spot_radius: float) -> int: ...

    def refine(self, crops: np.ndarray, centres: np.ndarray, spot_radius: float
               ) -> tuple[np.ndarray, Optional[np.ndarray]]: ...


def masked_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``np.median(values[i][mask[i]])`` for every row in one sort (0 for an
    empty row). A frame can hold hundreds of disks, and a Python loop of
    ``np.median`` calls cost ten times the detector network itself."""
    count = mask.sum(1)
    ordered = np.sort(np.where(mask, values, np.inf), axis=1)
    rows = np.arange(len(values))
    low = ordered[rows, np.maximum(count - 1, 0) // 2]
    high = ordered[rows, count // 2 - (count == 0)]
    return np.where(count > 0, 0.5 * (low.astype(np.float64) + high), 0.0)


def extract_crops(frame: np.ndarray, centres: np.ndarray, half_width: int):
    """Windows of ``2 * half_width + 1`` px around each centre's nearest pixel,
    zero outside the frame. Returns ``(crops, local)``: the ``(M, S, S)`` float32
    stack and the centres in crop pixels."""
    frame = np.asarray(frame, np.float32)
    centres = np.asarray(centres, np.float64).reshape(-1, 2)
    size = 2 * half_width + 1
    if len(centres) == 0:
        return np.zeros((0, size, size), np.float32), np.zeros((0, 2))
    padded = np.pad(frame, half_width + 1)
    nearest = np.rint(centres).astype(np.intp)
    nearest[:, 0] = np.clip(nearest[:, 0], 0, frame.shape[0] - 1)
    nearest[:, 1] = np.clip(nearest[:, 1], 0, frame.shape[1] - 1)
    offsets = np.arange(-half_width, half_width + 1)
    rows = nearest[:, 0, None] + 1 + half_width + offsets
    columns = nearest[:, 1, None] + 1 + half_width + offsets
    crops = padded[rows[:, :, None], columns[:, None, :]]
    local = centres - (nearest - half_width)
    return crops, local


def refine_centres(frames: Sequence[np.ndarray], positions: Sequence[np.ndarray],
                   spot_radius: float, refiner: CentreRefiner,
                   batch_size: int = DEFAULT_BATCH):
    """Refine every frame's ``(n, 2)`` ``[y, x]`` detections in ``frames``.

    Crops are cut and refined a batch at a time across frames, so a chunk's
    disks go through the refiner together without ever holding all their crops
    at once. Returns ``(positions, sigmas)``: the refined ``(n, 2)`` float32
    positions per frame, and per frame the refiner's ``(n,)`` uncertainty
    (NaN where the detection was kept) or ``None`` when the refiner reports
    none. Each disk's result depends only on its own crop, so refining one
    frame alone gives the same answer as refining it inside a chunk."""
    radius = float(spot_radius)
    half_width = int(refiner.crop_half_width(radius))
    refined = [np.asarray(p, np.float32).reshape(-1, 2).copy() for p in positions]
    sigmas: list = [np.full(len(p), np.nan, np.float32) for p in refined]
    has_sigma = True

    def _flush(pending):
        nonlocal has_sigma
        crops = np.concatenate([c for _, c, _ in pending])
        local = np.concatenate([l for _, _, l in pending])
        new_local, sigma = refiner.refine(crops, local, radius)
        new_local = np.asarray(new_local, np.float64).reshape(-1, 2)
        shift = np.hypot(*(new_local - local).T)
        accepted = np.isfinite(new_local).all(1) & (shift <= MAX_SHIFT_FRACTION * radius)
        if sigma is None:
            has_sigma = False
        start = 0
        for frame_index, crop_stack, _ in pending:
            stop = start + len(crop_stack)
            keep = accepted[start:stop]
            moved = refined[frame_index] + (new_local[start:stop] - local[start:stop])
            refined[frame_index][keep] = moved[keep]
            if sigma is not None:
                sigmas[frame_index][keep] = np.asarray(sigma[start:stop])[keep]
            start = stop

    pending, pending_count = [], 0
    for frame_index, (frame, points) in enumerate(zip(frames, refined)):
        if len(points) == 0:
            continue
        crops, local = extract_crops(frame, points, half_width)
        pending.append((frame_index, crops, local))
        pending_count += len(crops)
        if pending_count >= batch_size:
            _flush(pending)
            pending, pending_count = [], 0
    if pending:
        _flush(pending)
    return refined, (sigmas if has_sigma else None)


class MaskCentroidRefiner:
    """The centroid of the disk's support: every pixel above 30 % of the way from
    the local background to the disk's plateau, with a linear ramp below that.

    A brightness-weighted centroid follows the brightest part of the disk, and a
    dynamical disk is rarely evenly lit; the support is the disk's outline, so
    its centroid is the geometric centre however the disk is filled. The ramp
    makes the edge pixels count fractionally, which is what gives a sub-pixel
    answer. Per disk, iterated around the current estimate:

    * background — median of the ring 1-4 px outside the spot radius;
    * plateau — median inside the disk, away from its edge;
    * weight — ``clip((value - background) / (0.3 * (plateau - background)), 0, 1)``
      within 3 px of the spot radius, on the crop lightly smoothed (3 x 3 mean).

    A disk with no plateau above its background is declined (NaN)."""

    fraction = 0.3
    iterations = 4

    def crop_half_width(self, spot_radius: float) -> int:
        # The support window (radius + 3) around a centre that has moved by up
        # to the stage's limit still fits, with the background ring around it.
        return int(math.ceil(spot_radius)) + 5

    def refine(self, crops, centres, spot_radius):
        from scipy.ndimage import uniform_filter

        crops = np.asarray(crops, np.float32)
        count, size, _ = crops.shape
        if count == 0:
            return np.zeros((0, 2)), None
        radius = float(spot_radius)
        smoothed = uniform_filter(crops, size=(1, 3, 3), mode="nearest")
        flat = smoothed.reshape(count, -1)
        grid = np.arange(size, dtype=np.float64)
        row, column = grid[None, :, None], grid[None, None, :]
        centre_y = np.asarray(centres, np.float64)[:, 0].copy()
        centre_x = np.asarray(centres, np.float64)[:, 1].copy()
        plateau_radius = max(radius - 3.0, 0.5 * radius, 1.0)
        for _ in range(self.iterations):
            distance = np.hypot(row - centre_y[:, None, None], column - centre_x[:, None, None])
            flat_distance = distance.reshape(count, -1)
            background = masked_median(
                flat, (flat_distance > radius + 1) & (flat_distance <= radius + 4))
            plateau = masked_median(flat, flat_distance <= plateau_radius)
            level = np.maximum(self.fraction * (plateau - background), 1e-6)
            weight = np.clip((smoothed - background[:, None, None]) / level[:, None, None], 0, 1)
            weight *= distance <= radius + 3
            total = weight.sum((1, 2))
            usable = (total > 0) & (plateau > background)
            with np.errstate(invalid="ignore", divide="ignore"):
                centre_y = np.where(usable, (weight * row).sum((1, 2)) / total, centre_y)
                centre_x = np.where(usable, (weight * column).sum((1, 2)) / total, centre_x)
        result = np.stack([centre_y, centre_x], 1)
        result[~usable] = np.nan
        return result, None


def refiner_for(choice: Optional[str], device=None) -> Optional[CentreRefiner]:
    """The refiner a Find Vectors ``centre_refiner`` value names: ``None`` for
    the detector's own decode (empty or ``"decode"``), the mask centroid, or a
    refiner model from the registry by id. A model that fails to load logs a
    warning and gives ``None``, so detection still completes with the decode."""
    if not choice or choice == CENTRE_DECODE:
        return None
    if choice == CENTRE_MASK_CENTROID:
        return MaskCentroidRefiner()
    from . import registry
    try:
        return registry.get_refiner(choice, device)
    except Exception as error:
        log.warning("[models] centre refiner %r unavailable (%s); keeping the "
                    "detector's centres", choice, error)
        return None

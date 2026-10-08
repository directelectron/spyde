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
declines a disk by returning NaN for it. A refiner may also carry a
``min_spot_radius``: below it the stage is skipped and every detection kept.

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

# Below this spot radius (native px) every refiner keeps the detector's centres.
# On disks this small the raw crop holds too few pixels to beat the network's
# decode: on SPED-Ag (R ~ 3 px) refining raised the in-grain speckle from 0.036
# to 0.047, and the mask centroid's scatter failed the CIF-referenced
# orientation check that the decode passes. A refiner's ``min_spot_radius``
# overrides it (0 turns the gate off).
DEFAULT_MIN_SPOT_RADIUS = 5.0

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


def _window_indices(centres: np.ndarray, half_width: int, shape):
    """Rows, columns and inside-mask of the ``2 * half_width + 1`` px windows
    around each centre's nearest pixel, and the centres in window pixels."""
    height, width = shape
    nearest = np.rint(centres).astype(np.intp)
    nearest[:, 0] = np.clip(nearest[:, 0], 0, height - 1)
    nearest[:, 1] = np.clip(nearest[:, 1], 0, width - 1)
    offsets = np.arange(-half_width, half_width + 1)
    rows = nearest[:, 0, None] + offsets
    columns = nearest[:, 1, None] + offsets
    inside = (((rows >= 0) & (rows < height))[:, :, None]
              & ((columns >= 0) & (columns < width))[:, None, :])
    return (np.clip(rows, 0, height - 1), np.clip(columns, 0, width - 1), inside,
            centres - (nearest - half_width))


def extract_crops(frame: np.ndarray, centres: np.ndarray, half_width: int):
    """Windows of ``2 * half_width + 1`` px around each centre's nearest pixel,
    zero outside the frame. Returns ``(crops, local)``: the ``(M, S, S)`` float32
    stack and the centres in crop pixels."""
    frame = np.asarray(frame, np.float32)
    centres = np.asarray(centres, np.float64).reshape(-1, 2)
    size = 2 * half_width + 1
    if len(centres) == 0:
        return np.zeros((0, size, size), np.float32), np.zeros((0, 2))
    rows, columns, inside, local = _window_indices(centres, half_width, frame.shape)
    # Gather only the windows. Padding the frame instead copies all of it for
    # every frame, which on 512 x 512 frames cost four times the network.
    crops = frame[rows[:, :, None], columns[:, None, :]]
    return np.where(inside, crops, np.float32(0)), local


def _extract_crops_on_device(stack, frame_of: np.ndarray, centres: np.ndarray,
                             half_width: int):
    """:func:`extract_crops` for a group of frames at once, gathered on the GPU
    from ``stack`` (the group's frames already moved there): the same windows."""
    import torch

    device = stack.device
    rows, columns, inside, local = _window_indices(centres, half_width, stack.shape[1:])
    crops = stack[torch.as_tensor(frame_of, device=device)[:, None, None],
                  torch.as_tensor(rows, device=device)[:, :, None],
                  torch.as_tensor(columns, device=device)[:, None, :]]
    return crops * torch.as_tensor(inside, device=device), local


#: On a GPU, frames are moved to the device a group at a time; this bounds a
#: group's bytes (a few 4096 x 4096 frames, hundreds of 512 x 512 ones).
DEVICE_FRAME_BYTES = 256 * 2 ** 20


def frame_beams(positions, frame_shape, spot_radius) -> np.ndarray:
    """Per frame, the centre of Friedel symmetry of its ``(n, 2)`` centres:
    start at the disk nearest the mean of all disks, pair each disk with the
    one nearest its mirror (within ``max(0.4 R, 2 px)``), re-centre on the mean
    pair midpoint; three times. A frame with fewer than 4 disks or 2 pairs takes
    the median of the other frames' beams (descanned data), or the frame centre
    when none has one — so a lone frame in the live preview can fall back
    differently from the same frame inside a chunk."""
    tolerance = max(0.4 * float(spot_radius), 2.0)
    beams = np.full((len(positions), 2), np.nan)
    for number, points in enumerate(positions):
        points = np.asarray(points, np.float64).reshape(-1, 2)
        if len(points) < 4:
            continue
        centre = points[np.argmin(np.hypot(*(points - points.mean(0)).T))]
        found = False
        for _ in range(3):
            mirror = 2 * centre - points
            distance = np.hypot(mirror[:, None, 0] - points[None, :, 0],
                                mirror[:, None, 1] - points[None, :, 1])
            np.fill_diagonal(distance, np.inf)
            partner = distance.argmin(1)
            index = np.arange(len(points))
            paired = (distance[index, partner] < tolerance) & (index < partner)
            if paired.sum() < 2:
                found = False
                break
            centre = 0.5 * (points[paired] + points[partner[paired]]).mean(0)
            found = True
        if found:
            beams[number] = centre
    if np.isfinite(beams).any():
        fallback = np.nanmedian(beams, 0)
    else:
        fallback = np.asarray(frame_shape, np.float64) / 2
    return np.where(np.isfinite(beams), beams, fallback)


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
    refined = [np.asarray(p, np.float32).reshape(-1, 2).copy() for p in positions]
    if not (float(getattr(refiner, "min_spot_radius", 0.0)) <= radius
            <= float(getattr(refiner, "max_spot_radius", math.inf))):
        return refined, None
    half_width = int(refiner.crop_half_width(radius))
    sigmas: list = [np.full(len(p), np.nan, np.float32) for p in refined]
    has_sigma = True
    needs_partner = bool(getattr(refiner, "needs_partner", False))
    beams = frame_beams(refined, np.asarray(frames[0]).shape, radius) if needs_partner else None
    device = _gpu_device(refiner)

    def _cut(pending, points_of, stack):
        """The windows of every pending frame's ``points_of(frame_index)``;
        on the GPU from ``stack``, the pending frames moved there once."""
        if stack is None:
            cut = [extract_crops(frames[i], points_of(i), half_width) for i, _ in pending]
            return (np.concatenate([c for c, _ in cut]), np.concatenate([l for _, l in cut]))
        frame_of = np.concatenate([np.full(len(points), n) for n, (_, points) in enumerate(pending)])
        centres = np.concatenate([points_of(i) for i, _ in pending]).astype(np.float64)
        return _extract_crops_on_device(stack, frame_of, centres, half_width)

    def _flush(pending):
        nonlocal has_sigma
        stack = None
        if device is not None:
            import torch

            from spyde.device_lock import accelerator_lock
            with accelerator_lock(device):
                stack = torch.stack([torch.as_tensor(np.asarray(frames[i], np.float32), device=device)
                                     for i, _ in pending])
        crops, local = _cut(pending, lambda i: refined[i].astype(np.float64), stack)
        if needs_partner:
            # the window at each disk's Friedel mirror point, 2 x beam - p
            mirror_crops, mirror_local = _cut(
                pending, lambda i: 2 * beams[i] - refined[i].astype(np.float64), stack)
            new_local, sigma = refiner.refine(crops, local, radius, mirror_crops, mirror_local)
        else:
            new_local, sigma = refiner.refine(crops, local, radius)
        new_local = np.asarray(new_local, np.float64).reshape(-1, 2)
        shift = np.hypot(*(new_local - local).T)
        accepted = np.isfinite(new_local).all(1) & (shift <= MAX_SHIFT_FRACTION * radius)
        if sigma is None:
            has_sigma = False
        start = 0
        for frame_index, points in pending:
            stop = start + len(points)
            keep = accepted[start:stop]
            moved = refined[frame_index] + (new_local[start:stop] - local[start:stop])
            refined[frame_index][keep] = moved[keep]
            if sigma is not None:
                sigmas[frame_index][keep] = np.asarray(sigma[start:stop])[keep]
            start = stop

    pending, pending_count, pending_bytes = [], 0, 0
    for frame_index, (frame, points) in enumerate(zip(frames, refined)):
        if len(points) == 0:
            continue
        # the incoming positions, before any of this batch's refinement
        pending.append((frame_index, points.copy()))
        pending_count += len(points)
        pending_bytes += 4 * np.asarray(frame).size
        if pending_count >= batch_size or (device is not None and pending_bytes >= DEVICE_FRAME_BYTES):
            _flush(pending)
            pending, pending_count, pending_bytes = [], 0, 0
    if pending:
        _flush(pending)
    return refined, (sigmas if has_sigma else None)


def _gpu_device(refiner):
    """The refiner's torch device when windows should be cut there (a GPU), else None."""
    device = getattr(refiner, "device", None)
    try:
        import torch
        device = torch.device(device) if device is not None else None
    except (ImportError, TypeError, RuntimeError):
        return None
    return device if device is not None and device.type in ("cuda", "mps") else None


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

    A disk with no plateau above its background is declined (NaN). Below
    ``min_spot_radius`` the stage is skipped (see ``DEFAULT_MIN_SPOT_RADIUS``)."""

    fraction = 0.3
    iterations = 4

    def __init__(self, min_spot_radius: float = DEFAULT_MIN_SPOT_RADIUS):
        self.min_spot_radius = float(min_spot_radius)

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

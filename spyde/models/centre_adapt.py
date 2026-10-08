"""Adapt a centre network to one scan, using the scan's own symmetry as the teacher.

A general centre network is trained on simulation and a few real datasets; on a
new scan it still makes small, scan-varying mistakes (uneven disk fills,
detector response). The scan itself says where those are: a crystal's ``g`` and
``-g`` are symmetric about the direct beam, and its reflections sit on a
lattice. This fine-tunes a copy of the network so that, across the scan, the
Friedel midpoints and the lattice residuals vary less — and keeps the copy only
if frames it never trained on agree.

1. Detect and refine every frame (read a chunk at a time, never the whole scan).
2. Beam: per frame, the detection nearest the scan's median Friedel centre.
3. Reflections: every disk's ``g = position - beam`` pooled into a histogram of
   ``0.25 R`` cells; peaks held by at least 2 % of the frames are the scan's
   reflections. Each disk is indexed to the nearest reflection within ``0.4 R``,
   at most one per reflection per frame.
4. The two shortest non-collinear reflections are a basis; reflections within
   0.15 of integer ``(h, k)`` feed a per-frame affine fit ``p = A (h, k) + t``.
   Friedel partners are reflections whose ``g`` are opposite within ``0.4 R``.
5. Fine-tune on the Friedel midpoints and affine residuals, each after removing
   its pair's / reflection's running mean over the scan (a static offset cancels
   in a referenced strain map and is not the network's to explain), plus a
   simulated-disk anchor and a weak pull toward the mask centroid. Eight frames
   per step, under a wall-clock budget.
6. Hold out every fifth scan row (a random 20 %, capped at 1500 frames, when
   there are no rows). Accept only if, on the held-out frames, the varying
   Friedel error falls to ``friedel_ratio`` (0.98) of before or less, the
   varying lattice residual rises to no more than ``lattice_ratio`` (1.02), and
   the noise the adaptation adds (half-dose split, in quadrature) is under
   ``noise_fraction`` (0.5) of the Friedel improvement. A missing measure skips
   its test; both missing rejects.

A scan with fewer than ``min_train`` (50) training or ``min_held_out`` (20)
held-out frames indexing at least 4 reflections — sparse or amorphous data — is
declined before any training.

Ported from the training code's ``adapt2.py``. Its simulation anchor renders
multislice disks on whole frames; here the anchor renders soft, unevenly filled
disks straight into crop space, which keeps the network calibrated to a known
centre at a fraction of the cost.
"""
from __future__ import annotations

import contextlib
import copy
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from .centre_refine import MaskCentroidRefiner, accepted_moves, extract_crops

log = logging.getLogger(__name__)


@dataclass
class AdaptSettings:
    """The knobs of :func:`adapt_centre_refiner`. Defaults are the training
    code's validated values."""

    budget_seconds: Optional[float] = None   #: None: 150 s on a GPU, 300 s on a CPU
    max_steps: int = 1200
    frames_per_step: int = 8
    learning_rate: float = 2e-4
    prior_weight: float = 0.05               #: pull toward the mask centroid
    simulation_weight: float = 1.0           #: the simulated-disk anchor
    friedel_ratio: float = 0.98              #: accept if Friedel <= this x before
    lattice_ratio: float = 1.02              #: ... and lattice <= this x before
    noise_fraction: float = 0.5              #: ... and added noise < this x Friedel gain
    #: Reject if the refined fraction drops by more than this (fraction, e.g.
    #: 0.02). Off by default: reported, not enforced, until it is decided.
    max_refined_drop: Optional[float] = None
    min_train: int = 50
    min_held_out: int = 20
    held_out_cap: int = 1500
    train_cap: int = 2000                    #: training frames whose crops are kept
    noise_frames: int = 300
    seed: int = 0


@dataclass
class AdaptReport:
    accepted: bool
    declined: Optional[str] = None           #: why no training was attempted
    cancelled: bool = False
    frames: int = 0
    reflections: int = 0
    on_lattice: int = 0
    friedel_pairs: int = 0
    indexed_per_frame: float = float("nan")
    train_frames: int = 0
    held_out_frames: int = 0
    steps: int = 0
    before: dict = field(default_factory=dict)   #: friedel, lattice, noise, refined
    after: dict = field(default_factory=dict)
    mask_centroid: dict = field(default_factory=dict)
    refined_drop: float = float("nan")
    prepare_seconds: float = 0.0
    train_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self):
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return {k: clean(v) for k, v in self.__dict__.items()}


class Cancelled(Exception):
    pass


# ── the scan's own index ───────────────────────────────────────────────────────

def frame_centre(points: np.ndarray, radius: float) -> Optional[np.ndarray]:
    """A frame's centre of symmetry from its Friedel pairs, or None with fewer
    than two pairs: start from the disk nearest the disks' mean, pair each disk
    with the one nearest its mirror, re-centre on the mean pair midpoint, three
    times."""
    if len(points) < 4:
        return None
    tolerance = max(0.4 * radius, 2.0)
    centre = points[np.argmin(np.hypot(*(points - points.mean(0)).T))]
    for _ in range(3):
        mirror = 2 * centre - points
        distance = np.hypot(mirror[:, None, 0] - points[None, :, 0],
                            mirror[:, None, 1] - points[None, :, 1])
        np.fill_diagonal(distance, np.inf)
        partner = distance.argmin(1)
        index = np.arange(len(points))
        keep = (distance[index, partner] < tolerance) & (index < partner)
        if keep.sum() < 2:
            return None
        centre = 0.5 * (points[keep] + points[partner[keep]]).mean(0)
    return centre


@dataclass
class ScanIndex:
    beams: np.ndarray            #: (N, 2) per frame
    reflections: np.ndarray      #: (K, 2) g vectors
    index: list                  #: per frame (n,) reflection of each disk, -1 if none
    hk: np.ndarray               #: (K, 2), NaN off the lattice
    partner: np.ndarray          #: (K,) Friedel partner reflection, -1 if none

    @property
    def pairs(self):
        return [(k, j) for k, j in enumerate(self.partner) if j > k]


def scan_index(positions, frame_shape, radius) -> ScanIndex:
    """Steps 2-4 above, on every frame's refined ``(n, 2)`` positions."""
    from scipy.ndimage import gaussian_filter, maximum_filter

    radius = float(radius)
    centres = [c for c in (frame_centre(p, radius) for p in positions) if c is not None]
    scan_beam = np.median(centres, 0) if centres else np.asarray(frame_shape, float) / 2
    beams = np.zeros((len(positions), 2))
    for n, points in enumerate(positions):
        distance = np.hypot(*(points - scan_beam).T) if len(points) else np.array([np.inf])
        beams[n] = points[np.argmin(distance)] if distance.min() < radius else scan_beam
    reflections = np.zeros((0, 2))
    g = np.concatenate([p - b for p, b in zip(positions, beams)]) if positions else np.zeros((0, 2))
    if len(g):
        cell = 0.25 * radius
        extent = np.abs(g).max() + radius
        edges = np.arange(-extent, extent + cell, cell)
        histogram, _, _ = np.histogram2d(g[:, 0], g[:, 1], bins=[edges, edges])
        smooth = gaussian_filter(histogram, 1.0)
        window = int(np.ceil(1.2 * radius / cell)) | 1
        peaks = (smooth == maximum_filter(smooth, size=window)) & (histogram >= 0.02 * len(positions))
        rows, columns = np.nonzero(peaks)
        found = []
        for centre in np.c_[edges[rows] + cell / 2, edges[columns] + cell / 2]:
            near = np.hypot(*(g - centre).T) < 0.4 * radius
            if near.sum() >= 0.02 * len(positions):
                found.append(g[near].mean(0))
        if found:
            reflections = np.array(found)
            reflections = reflections[np.hypot(*reflections.T) > 1.5 * radius]
    count = len(reflections)
    index = []
    for points, beam in zip(positions, beams):
        assigned = np.full(len(points), -1)
        if len(points) and count:
            relative = points - beam
            distance = np.hypot(relative[:, None, 0] - reflections[None, :, 0],
                                relative[:, None, 1] - reflections[None, :, 1])
            best = distance.argmin(1)
            close = distance[np.arange(len(points)), best] < 0.4 * radius
            for k in np.unique(best[close]):
                members = np.where(close & (best == k))[0]
                assigned[members[np.argmin(distance[members, k])]] = k
        index.append(assigned)
    partner = np.full(count, -1)
    for k in range(count):
        distance = np.hypot(*(reflections + reflections[k]).T)
        j = int(np.argmin(distance))
        if distance[j] < 0.4 * radius and j != k:
            partner[k] = j
    hk = np.full((count, 2), np.nan)
    if count >= 2:
        order = np.argsort(np.hypot(*reflections.T))
        a = reflections[order[0]]
        b = next((reflections[j] for j in order[1:]
                  if abs(np.cross(a, reflections[j]))
                  > 0.3 * np.linalg.norm(a) * np.linalg.norm(reflections[j])), None)
        if b is not None:
            coordinates = np.linalg.solve(np.c_[a, b], reflections.T).T
            on = np.all(np.abs(coordinates - np.round(coordinates)) < 0.15, 1)
            hk[on] = np.round(coordinates[on])
    return ScanIndex(beams, reflections, index, hk, partner)


def gather(positions, index, count):
    """Per frame ``(K, 2)``: the position of each reflection, NaN if absent."""
    out = np.full((len(positions), count, 2), np.nan)
    for n, (points, assigned) in enumerate(zip(positions, index)):
        ok = assigned >= 0
        out[n, assigned[ok]] = points[ok]
    return out


def residuals(per_reflection, hk, partner):
    """Friedel-midpoint residuals ``(N, pairs, 2)`` about each frame's mean
    midpoint (frames with two pairs or more), and affine-lattice residuals
    ``(N, K, 2)`` (frames with five on-lattice reflections or more); NaN where
    unavailable."""
    frames, count, _ = per_reflection.shape
    pairs = [(k, j) for k, j in enumerate(partner) if j > k]
    friedel = np.full((frames, len(pairs), 2), np.nan)
    if pairs:
        midpoints = np.stack([0.5 * (per_reflection[:, k] + per_reflection[:, j]) for k, j in pairs], 1)
        have = np.isfinite(midpoints[..., 0]).sum(1)
        with _quiet():
            centre = np.nanmean(midpoints, 1, keepdims=True)
        friedel = np.where((have >= 2)[:, None, None], midpoints - centre, np.nan)
    lattice = np.full((frames, count, 2), np.nan)
    on = np.isfinite(hk[:, 0])
    design = np.c_[np.nan_to_num(hk), np.ones(count)]
    for n in range(frames):
        ok = on & np.isfinite(per_reflection[n, :, 0])
        if ok.sum() >= 5:
            solution, *_ = np.linalg.lstsq(design[ok], per_reflection[n, ok], rcond=None)
            lattice[n, ok] = ((per_reflection[n, ok] - design[ok] @ solution)
                              * np.sqrt(ok.sum() / (ok.sum() - 3)))
    return friedel, lattice


@contextlib.contextmanager
def _quiet():
    """All-NaN columns are expected here (a pair absent from every frame)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def varying(values) -> float:
    """Robust per-axis scale after removing each column's (pair's / reflection's)
    median over the frames: the part of the error that varies across the scan."""
    with _quiet():
        centred = values - np.nanmedian(values, 0, keepdims=True)
    centred = centred[np.isfinite(centred)]
    return float(1.4826 * np.median(np.abs(centred))) if len(centred) else float("nan")


# ── what is kept from the read ─────────────────────────────────────────────────

@dataclass
class _Disks:
    """One frame's kept disks: native crops, the detections in crop pixels, and
    each crop's frame origin."""

    crops: np.ndarray
    local: np.ndarray
    origin: np.ndarray

    @property
    def seeds(self):
        return self.local + self.origin


def _crop_frame(frame, seeds, half_width):
    crops, local = extract_crops(frame, seeds, half_width)
    nearest = np.rint(seeds).astype(np.intp)
    nearest[:, 0] = np.clip(nearest[:, 0], 0, frame.shape[0] - 1)
    nearest[:, 1] = np.clip(nearest[:, 1], 0, frame.shape[1] - 1)
    return _Disks(crops, local, (nearest - half_width).astype(np.float64))


def _refine_disks(refiner, disks: list, radius):
    """Refined frame positions per frame and the fraction of disks refined
    (the rest keep the detection), the way the centre stage applies a refiner."""
    out, refined, total = [], 0, 0
    for item in disks:
        if len(item.local) == 0:
            out.append(np.zeros((0, 2)))
            continue
        new_local, _ = refiner.refine(item.crops, item.local, radius)
        new_local = np.asarray(new_local, np.float64).reshape(-1, 2)
        keep = accepted_moves(item.local, new_local, radius)
        local = np.where(keep[:, None], new_local, item.local)
        out.append(local + item.origin)
        refined += int(keep.sum())
        total += len(keep)
    return out, refined / max(total, 1)


def _half_dose_noise(refiner, disks: list, radius, rng, frames):
    """Per-axis robust scale of (position on half the counts - on the other
    half) / 2, over up to ``frames`` frames."""
    chosen = rng.choice(len(disks), min(frames, len(disks)), replace=False)
    differences = []
    for i in chosen:
        item = disks[i]
        if len(item.local) == 0:
            continue
        counts = np.rint(np.clip(item.crops, 0, None)).astype(np.int64)
        half = rng.binomial(counts, 0.5).astype(np.float32)
        a, _ = _refine_disks(refiner, [_Disks(half, item.local, item.origin)], radius)
        b, _ = _refine_disks(refiner, [_Disks(counts.astype(np.float32) - half, item.local,
                                              item.origin)], radius)
        differences.append(a[0] - b[0])
    if not differences:
        return float("nan")
    return float(1.4826 * np.median(np.abs(np.concatenate(differences))) / 2)


# ── the adaptation ─────────────────────────────────────────────────────────────

def _predict(refiner, crops, seeds, step):
    """Differentiable single-pass centre (crop pixels) of ``crops`` around
    ``seeds``: the refiner's own resample, normalisation and coverage centroid."""
    sampled = refiner._normalise(refiner._resample(crops, seeds, step))
    output = refiner.net(sampled)
    logits = output[0] if isinstance(output, tuple) else output
    offset, _ = refiner._coverage_centroid(logits, iterations=1)
    return seeds + offset * step


def _simulated_batch(refiner, count, generator, device):
    """Soft, unevenly filled disks rendered straight into crop space with known
    offsets: normalised crops ``(count, 1, S, S)`` and offsets in crop pixels."""
    import torch

    size = 2 * refiner.crop_half + 1
    grid = torch.arange(size, device=device, dtype=torch.float32) - refiner.crop_half
    rows, columns = grid[None, :, None], grid[None, None, :]

    def uniform(low, high):
        return low + (high - low) * torch.rand(count, device=device, generator=generator)

    radius = refiner.crop_radius * uniform(0.85, 1.15)
    offset = torch.stack([uniform(-0.4, 0.4), uniform(-0.4, 0.4)], 1) * refiner.crop_radius
    dy = rows - offset[:, 0, None, None]
    dx = columns - offset[:, 1, None, None]
    distance = torch.sqrt(dy * dy + dx * dx)
    edge = 0.5 * torch.erfc((distance - radius[:, None, None]) / (math.sqrt(2) * 0.6))
    angle = uniform(0, 2 * math.pi)
    tilt = uniform(0.0, 0.5)
    along = (dx * torch.cos(angle)[:, None, None] + dy * torch.sin(angle)[:, None, None]) \
        / radius[:, None, None]
    fill = (1 + tilt[:, None, None] * along).clamp_min(0)
    amplitude = uniform(20, 400)
    background = uniform(0, 20)
    image = background[:, None, None] + amplitude[:, None, None] * edge * fill
    image = torch.poisson(image, generator=generator)
    return refiner._normalise(image[:, None]), offset


def adapt_centre_refiner(chunks: Iterable, spot_radius: float, base_refiner,
                         detect: Callable, frame_shape, settings: AdaptSettings = None,
                         progress: Optional[Callable] = None,
                         cancel: Optional[list] = None):
    """Adapt ``base_refiner`` (a :class:`~spyde.models.centre_network.NetworkRefiner`)
    to one scan. Returns ``(adapted refiner or None, AdaptReport)``.

    ``chunks`` yields ``(frames (n, H, W), rows (n,) or None)`` — the scan a
    chunk at a time; only per-disk crops of a capped sample of frames are kept.
    ``detect(frames) -> [(m, 2) [y, x]]`` is the detector. ``progress(stage,
    done, total)`` is called as work proceeds; ``cancel`` is a one-element list
    polled throughout (``[True]`` stops with ``report.cancelled``)."""
    import torch
    import torch.nn.functional as F

    from spyde.device_lock import accelerator_lock

    from .centre_refine import refine_centres

    settings = settings or AdaptSettings()
    rng = np.random.default_rng(settings.seed)
    radius = float(spot_radius)
    device = torch.device(base_refiner.device)
    started = time.perf_counter()

    def report_progress(stage, done, total):
        if cancel is not None and cancel[0]:
            raise Cancelled
        if progress is not None:
            progress(stage, done, total)

    half_width = int(base_refiner.crop_half_width(radius))
    height, width = frame_shape
    margin = 2 * radius
    positions, rows_all, kept = [], [], {}
    try:
        # 1. detect and refine, a chunk at a time; keep crops for a sample
        frame_number = 0
        for chunk_number, (frames, rows) in enumerate(chunks):
            report_progress("reading", chunk_number, None)
            frames = np.asarray(frames, np.float32)
            seeds = [np.asarray(s, np.float64).reshape(-1, 2) for s in detect(frames)]
            seeds = [s[(s[:, 0] > margin) & (s[:, 0] < height - margin)
                       & (s[:, 1] > margin) & (s[:, 1] < width - margin)] for s in seeds]
            refined, _ = refine_centres(list(frames), seeds, radius, base_refiner)
            for i, frame in enumerate(frames):
                positions.append(np.asarray(refined[i], np.float64))
                rows_all.append(-1 if rows is None else int(rows[i]))
                # every frame is a candidate until the caps are reached; a
                # reservoir sample keeps the kept set uniform over the scan
                number = frame_number + i
                if len(kept) < settings.train_cap + settings.held_out_cap:
                    kept[number] = _crop_frame(frame, seeds[i], half_width)
                else:
                    slot = rng.integers(0, number + 1)
                    if slot < settings.train_cap + settings.held_out_cap:
                        victim = list(kept)[int(slot)]
                        del kept[victim]
                        kept[number] = _crop_frame(frame, seeds[i], half_width)
            frame_number += len(frames)
        count = len(positions)
        report_progress("indexing", 0, None)

        # 2-4. the scan's own index
        index = scan_index(positions, frame_shape, radius)
        reflections = len(index.reflections)
        rows_all = np.asarray(rows_all)
        if (rows_all >= 0).all():
            held_mask = rows_all % 5 == 4
        else:
            held_mask = rng.random(count) < 0.2
        usable = np.array([(assigned >= 0).sum() >= 4 for assigned in index.index])
        kept_numbers = np.array(sorted(kept), dtype=int)
        held = kept_numbers[held_mask[kept_numbers] & usable[kept_numbers]]
        train = kept_numbers[~held_mask[kept_numbers] & usable[kept_numbers]]
        if len(held) > settings.held_out_cap:
            held = np.sort(rng.choice(held, settings.held_out_cap, replace=False))
        report = AdaptReport(
            accepted=False, frames=count, reflections=reflections,
            on_lattice=int(np.isfinite(index.hk[:, 0]).sum()), friedel_pairs=len(index.pairs),
            indexed_per_frame=float(np.mean([(a >= 0).sum() for a in index.index])) if count else 0.0,
            train_frames=int(len(train)), held_out_frames=int(len(held)))
        if len(train) < settings.min_train or len(held) < settings.min_held_out:
            report.declined = ("too few frames index four reflections (sparse or amorphous "
                               "data, or a spot size that does not match)")
            report.prepare_seconds = report.total_seconds = time.perf_counter() - started
            return None, report

        # disks of a kept frame, restricted to the ones the index uses
        def indexed_disks(number):
            item = kept[number]
            assigned = index.index[number]
            mask = assigned >= 0
            return _Disks(item.crops[mask], item.local[mask], item.origin[mask]), assigned[mask]

        held_disks = [indexed_disks(n)[0] for n in held]
        held_index = [indexed_disks(n)[1] for n in held]

        def score(frame_positions):
            per_reflection = gather(frame_positions, held_index, reflections)
            friedel, lattice = residuals(per_reflection, index.hk, index.partner)
            return varying(friedel), varying(lattice)

        report_progress("scoring", 0, None)
        before_positions, refined_before = _refine_disks(base_refiner, held_disks, radius)
        friedel_before, lattice_before = score(before_positions)
        mask_positions = [
            np.where(np.isfinite(m), m, d.local) + d.origin if len(d.local) else np.zeros((0, 2))
            for d, m in ((d, MaskCentroidRefiner(min_spot_radius=0).refine(d.crops, d.local, radius)[0]
                          if len(d.local) else None) for d in held_disks)]
        mask_friedel, mask_lattice = score(mask_positions)
        noise_before = _half_dose_noise(base_refiner, held_disks, radius,
                                        np.random.default_rng(settings.seed + 1), settings.noise_frames)
        report.prepare_seconds = time.perf_counter() - started
        report.before = dict(friedel=friedel_before, lattice=lattice_before,
                             noise=noise_before, refined=refined_before)
        report.mask_centroid = dict(friedel=mask_friedel, lattice=mask_lattice)

        # 5. fine-tune a copy
        adapted = copy.copy(base_refiner)
        adapted.net = copy.deepcopy(base_refiner.net)
        net = adapted.net
        net.train()
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        optimiser = torch.optim.AdamW(net.parameters(), lr=settings.learning_rate,
                                      weight_decay=1e-4)
        generator = torch.Generator(device=device).manual_seed(settings.seed)
        pairs = index.pairs
        pair_first = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
        pair_second = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
        on_lattice = torch.as_tensor(np.isfinite(index.hk[:, 0]), device=device)
        design = torch.as_tensor(np.c_[np.nan_to_num(index.hk), np.ones(reflections)],
                                 device=device, dtype=torch.float32)
        pair_constant = torch.zeros(len(pairs), 2, device=device)
        reflection_constant = torch.zeros(reflections, 2, device=device)
        prior_refiner = MaskCentroidRefiner(min_spot_radius=0)
        train_items = [indexed_disks(n) for n in train]
        budget = settings.budget_seconds
        if budget is None:
            budget = 150.0 if device.type in ("cuda", "mps") else 300.0
        step_scale = radius / adapted.crop_radius
        training_started = time.perf_counter()
        step = 0
        with accelerator_lock(device):
            while step < settings.max_steps and time.perf_counter() - training_started < budget:
                report_progress("training", step, settings.max_steps)
                chosen = rng.choice(len(train_items), min(settings.frames_per_step, len(train_items)),
                                    replace=False)
                items = [train_items[i] for i in chosen]
                crops = torch.as_tensor(np.concatenate([d.crops for d, _ in items]), device=device)
                local = torch.as_tensor(np.concatenate([d.local for d, _ in items]),
                                        device=device, dtype=torch.float32)
                origin = torch.as_tensor(np.concatenate([d.origin for d, _ in items]),
                                         device=device, dtype=torch.float32)
                frame_of = torch.as_tensor(np.concatenate([np.full(len(d.local), n)
                                                           for n, (d, _) in enumerate(items)]),
                                           device=device, dtype=torch.long)
                reflection_of = torch.as_tensor(np.concatenate([a for _, a in items]),
                                                device=device, dtype=torch.long)
                jitter = (torch.rand(local.shape, device=device, generator=generator) * 2 - 1) \
                    * 0.15 * radius
                start = local + jitter
                predicted = _predict(adapted, crops, start, step_scale) + origin
                batch = len(items)
                per_reflection = torch.zeros(batch, reflections, 2, device=device).index_put(
                    (frame_of, reflection_of), predicted)
                present = torch.zeros(batch, reflections, device=device).index_put(
                    (frame_of, reflection_of), torch.ones(len(predicted), device=device))
                loss = predicted.sum() * 0
                if pairs:
                    both = present[:, pair_first] * present[:, pair_second]
                    midpoints = 0.5 * (per_reflection[:, pair_first] + per_reflection[:, pair_second])
                    paired = both.sum(1, keepdim=True).clamp_min(1)
                    friedel = midpoints - (midpoints * both[..., None]).sum(1, keepdim=True) \
                        / paired[..., None]
                    use = both * (paired >= 2).float()
                    loss = loss + (F.smooth_l1_loss((friedel - pair_constant) / radius,
                                                    torch.zeros_like(friedel), beta=0.03,
                                                    reduction="none").sum(-1) * use).sum() \
                        / use.sum().clamp_min(1) * 10
                    with torch.no_grad():
                        pair_constant.mul_(0.99).add_(
                            0.01 * (friedel * use[..., None]).sum(0) / use.sum(0).clamp_min(1)[:, None])
                lattice_weight = present * on_lattice.float()
                enough = (lattice_weight.sum(1, keepdim=True) >= 5).float() * lattice_weight
                if enough.sum() > 0:
                    weighted = design.t()[None] * lattice_weight[:, None, :]
                    solution = torch.linalg.solve(
                        weighted @ design[None].expand(batch, -1, -1)
                        + 1e-3 * torch.eye(3, device=device), weighted @ per_reflection)
                    residual = per_reflection - design[None] @ solution
                    loss = loss + (F.smooth_l1_loss((residual - reflection_constant) / radius,
                                                    torch.zeros_like(residual), beta=0.03,
                                                    reduction="none").sum(-1) * enough).sum() \
                        / enough.sum().clamp_min(1) * 10
                    with torch.no_grad():
                        reflection_constant.mul_(0.99).add_(
                            0.01 * (residual * enough[..., None]).sum(0)
                            / enough.sum(0).clamp_min(1)[:, None])
                if settings.prior_weight > 0 and step % 4 == 0:
                    prior_local, _ = prior_refiner.refine(
                        np.concatenate([d.crops for d, _ in items]),
                        np.concatenate([d.local for d, _ in items]), radius)
                    finite = torch.as_tensor(np.isfinite(prior_local).all(1), device=device)
                    if finite.any():
                        prior = torch.as_tensor(np.nan_to_num(prior_local), device=device,
                                                dtype=torch.float32) + origin
                        loss = loss + settings.prior_weight * F.smooth_l1_loss(
                            predicted[finite] / radius, prior[finite] / radius, beta=0.1)
                if settings.simulation_weight > 0 and step % 2 == 0:
                    simulated, truth = _simulated_batch(adapted, 32, generator, device)
                    output = net(simulated)
                    logits = output[0] if isinstance(output, tuple) else output
                    centre, _ = adapted._coverage_centroid(logits, iterations=1)
                    loss = loss + settings.simulation_weight * F.smooth_l1_loss(centre, truth, beta=0.2)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                step += 1
        net.eval()
        report.steps = step
        report.train_seconds = time.perf_counter() - training_started

        # 6. judge on the held-out frames
        report_progress("scoring", 1, None)
        after_positions, refined_after = _refine_disks(adapted, held_disks, radius)
        friedel_after, lattice_after = score(after_positions)
        noise_after = _half_dose_noise(adapted, held_disks, radius,
                                       np.random.default_rng(settings.seed + 1), settings.noise_frames)
        report.after = dict(friedel=friedel_after, lattice=lattice_after,
                            noise=noise_after, refined=refined_after)
        report.refined_drop = refined_before - refined_after
        report.accepted = accept(report.before, report.after, settings)
        report.total_seconds = time.perf_counter() - started
        return (adapted if report.accepted else None), report
    except Cancelled:
        report = AdaptReport(accepted=False, cancelled=True, frames=len(positions),
                             total_seconds=time.perf_counter() - started)
        return None, report


def accept(before: dict, after: dict, settings: AdaptSettings) -> bool:
    """The acceptance rule (module docstring, step 6) on the held-out measures."""
    friedel_before, friedel_after = before["friedel"], after["friedel"]
    lattice_before, lattice_after = before["lattice"], after["lattice"]
    if not math.isfinite(friedel_before) and not math.isfinite(lattice_before):
        return False
    ok = True
    if math.isfinite(friedel_before):
        ok &= friedel_after <= settings.friedel_ratio * friedel_before
    if math.isfinite(lattice_before):
        ok &= lattice_after <= settings.lattice_ratio * lattice_before
    gain = friedel_before - friedel_after if math.isfinite(friedel_before) else 0.0
    added = math.sqrt(max(after["noise"] ** 2 - before["noise"] ** 2, 0.0)) \
        if math.isfinite(after["noise"]) and math.isfinite(before["noise"]) else 0.0
    ok &= added < settings.noise_fraction * gain if math.isfinite(friedel_before) else True
    if settings.max_refined_drop is not None:
        ok &= before["refined"] - after["refined"] <= settings.max_refined_drop
    return bool(ok)

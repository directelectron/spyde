"""Adapt the disk detector to one dataset from a user's marks.

The user double-clicks wrong detections ("not a disk") and places they see a
disk the detector missed ("disk here") on a few patterns. A copy of the model
is fine-tuned on those marks for a few steps; the original is never touched.

What was measured to work, in a research spike on synthetic data with a
deliberate domain shift (5 datasets, 2 click draws each):

* All weights, 20 Adam steps at 1e-3, about a second on a GPU and two to three
  on a CPU. Training only the last 1x1 heatmap head made detection WORSE: eight
  feature channels cannot separate a faint disk from a hot pixel linearly.
* Almost all of the gain is fewer false positives (20 → 1-2 per pattern, F1
  0.75 → 0.83). "Not a disk" marks carry it; "disk here" marks barely move recall,
  and on their own they raise the response everywhere.
* Unmarked pixels are held to the base model's own output (weighted toward its
  peaks), so a mark changes the neighbourhood of what was marked, not the whole
  pattern. The sub-pixel offset head is held to the base model too: a click is
  not sub-pixel accurate, so it names the disk, never its centre.
* 20 % replay of the original synthetic training frames keeps the model a disk
  detector: without it, accuracy on the original distribution falls from F1
  0.86 to ~0.60; with it, ~0.78. :func:`original_f1` reports that number, so
  the user can see how far the model was pulled toward their data.

The adapted model is therefore a model FOR THIS DATASET, not a better default.
"""
from __future__ import annotations

import copy
import logging
import time
from functools import lru_cache
from importlib import resources

import numpy as np
import torch
import torch.nn.functional as F

from .decode import decode_batch, split_by_batch
from .infer import _big_disk_params, _estimate_work_diam, _pad_to_multiple
from .preprocess import normalize_input, scale_to_canonical

log = logging.getLogger(__name__)

#: Marks: 1 = "a disk is here", 0 = "this detection is wrong".
DISK, NOT_DISK = 1, 0

STEPS = 20
LEARNING_RATE = 1e-3
REPLAY_WEIGHT = 0.2
DISTILL_WEIGHT = 1.0
OFFSET_WEIGHT = 1.0
#: Unmarked pixels are held to the base output with this extra weight on its
#: peaks: a plain mean over ~50k background pixels lets every unmarked
#: detection drift for free.
PEAK_HOLD = 20.0
HEATMAP_SIGMA = 1.2

class Cancelled(Exception):
    """The fit was stopped (its tree closed, or a newer fit replaced it)."""


HYPERPARAMETERS = dict(steps=STEPS, learning_rate=LEARNING_RATE, replay_weight=REPLAY_WEIGHT,
                       distill_weight=DISTILL_WEIGHT, offset_weight=OFFSET_WEIGHT,
                       peak_hold=PEAK_HOLD, method="all weights, Adam, D4 augmentation")


# ── preprocessing, exactly as detect() does it ───────────────────────────────

class Working:
    """A stack of patterns at the model's working resolution, with the decode
    parameters ``detect`` would use for them."""

    def __init__(self, frames, params: dict, levels: int):
        frames = [np.asarray(f, np.float32) for f in frames]
        spot_radius = float(params.get("spot_radius") or 0)
        spot_diameter = 2.0 * spot_radius if spot_radius > 0 else None
        _, self.factor = scale_to_canonical(frames[0], diameter=spot_diameter)
        md = max(2, int(round(int(params.get("min_distance", 4)) * self.factor)))
        work_diam = _estimate_work_diam(frames[0], self.factor, spot_diameter)
        bg = params.get("bg_sigma")
        self.min_distance, self.bg_sigma = _big_disk_params(md, None if bg is None else float(bg), work_diam)
        self.threshold = float(params.get("threshold", 0.3))
        self.shape = frames[0].shape
        out = []
        for frame in frames:
            if self.factor != 1.0:
                from scipy.ndimage import zoom
                frame = zoom(frame, self.factor, order=1)
            out.append(_pad_to_multiple(normalize_input(frame, local=True, bg_sigma=self.bg_sigma), levels))
        self.x = torch.from_numpy(np.stack(out)[:, None].astype(np.float32))

    def peaks(self, hm, off) -> list[np.ndarray]:
        """(N_i, 3) [y, x, score] per pattern, in ORIGINAL pixels."""
        per = split_by_batch(decode_batch(hm, off, thresh=self.threshold,
                                          min_distance=self.min_distance), hm.shape[0])
        return [np.column_stack([p[:, :2] / self.factor, p[:, 2:]]) if len(p) else p for p in per]


# ── marks → training targets ─────────────────────────────────────────────────

@torch.no_grad()
def mark_targets(base, working: Working, marks: list[np.ndarray], disk_radius_px: float, device):
    """Heatmap target + weight from per-pattern marks ``[y, x, label]`` (original px).

    A "disk here" mark within ~2 px of an existing detection confirms it.
    Otherwise it is snapped to the base heatmap's maximum within ~1.5 px of the
    click — never onto an existing detection, since a missed disk often sits
    beside a found one. A "not a disk" mark sets a small disk of the heatmap to
    zero. Every mark weighs the same however many pixels it covers."""
    x = working.x.to(device)
    hm_b, off_b = base(x)
    n, _, h, w = x.shape
    f = working.factor
    target = torch.zeros_like(hm_b)
    weight = torch.zeros_like(hm_b)
    yy = torch.arange(h, device=device).view(h, 1).float()
    xx = torch.arange(w, device=device).view(1, w).float()
    reject_radius = max(2.0, 0.4 * disk_radius_px * f)
    probability = torch.sigmoid(hm_b)
    pooled = F.max_pool2d(probability, 2 * working.min_distance + 1, stride=1, padding=working.min_distance)
    found = ((probability == pooled) & (probability >= working.threshold)).squeeze(1).nonzero()
    for b, frame_marks in enumerate(marks):
        detections = found[found[:, 0] == b][:, 1:].float()
        for my, mx, label in np.asarray(frame_marks, np.float64).reshape(-1, 3):
            cy, cx = my * f, mx * f
            if int(label) == NOT_DISK:
                region = ((yy - cy) ** 2 + (xx - cx) ** 2) <= reject_radius ** 2
                target[b, 0][region] = 0.0
                weight[b, 0][region] += 1.0 / region.sum().clamp_min(1)
                continue
            iy, ix = int(round(cy)), int(round(cx))
            distance = torch.hypot(detections[:, 0] - cy, detections[:, 1] - cx) if len(detections) else None
            if distance is not None and float(distance.min()) <= 2.0 * f:
                iy, ix = (int(v) for v in detections[int(torch.argmin(distance))])
            else:
                reach = max(1, int(round(1.5 * f)))
                y0, y1 = max(0, iy - reach), min(h, iy + reach + 1)
                x0, x1 = max(0, ix - reach), min(w, ix + reach + 1)
                window = hm_b[b, 0, y0:y1, x0:x1].clone()
                for dy, dx in detections.tolist():
                    wy = torch.arange(y0, y1, device=device).view(-1, 1).float()
                    wx = torch.arange(x0, x1, device=device).view(1, -1).float()
                    window[((wy - dy) ** 2 + (wx - dx) ** 2) <= (2.5 * f) ** 2] = -1e9
                k = int(torch.argmax(window))
                if float(window.flatten()[k]) > -1e8:
                    iy, ix = y0 + k // window.shape[1], x0 + k % window.shape[1]
            iy, ix = min(max(iy, 0), h - 1), min(max(ix, 0), w - 1)
            blob = torch.exp(-((yy - iy) ** 2 + (xx - ix) ** 2) / (2 * HEATMAP_SIGMA ** 2))
            target[b, 0] = torch.maximum(target[b, 0], blob)
            target[b, 0, iy, ix] = 1.0
            region = ((yy - iy) ** 2 + (xx - ix) ** 2) <= 9.0
            weight[b, 0][region] += 1.0 / region.sum()
    return dict(x=x, target=target, weight=weight, hm_base=hm_b, off_base=off_b)


# ── losses ───────────────────────────────────────────────────────────────────

def _mark_loss(logit, target, weight):
    if float(weight.sum()) <= 0:
        return logit.sum() * 0
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    return (bce * weight).sum() / weight.sum()


def _hold_loss(logit, hm_base, weight):
    """KL to the base model on unmarked pixels, weighted toward its peaks."""
    base = torch.sigmoid(hm_base)
    free = (weight <= 0).float() * (1.0 + PEAK_HOLD * base)
    bce = F.binary_cross_entropy_with_logits(logit, base, reduction="none")
    entropy = F.binary_cross_entropy(base.clamp(1e-6, 1 - 1e-6), base, reduction="none")
    return ((bce - entropy) * free).sum() / free.sum().clamp_min(1)


def _offset_hold(off, off_base, hm_base):
    near = (torch.sigmoid(hm_base) > 0.1).float()
    return ((off - off_base).abs() * near).sum() / (2 * near.sum()).clamp_min(1)


def _focal(logits, target, eps=1e-6):
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    positive = (target >= 0.999).float()
    negative = 1.0 - positive
    loss = (-((1 - p) ** 2) * torch.log(p) * positive
            - (p ** 2) * torch.log(1 - p) * (1.0 - target) ** 4 * negative)
    return loss.sum() / positive.sum().clamp_min(1.0)


def _dihedral(t, k, offset=False):
    """One of the 8 symmetries of the square; ``offset`` also turns the (dy, dx)
    channels with the image."""
    if k & 1:
        t = t.transpose(-1, -2)
        if offset:
            t = t.flip(1)
    if k & 2:
        t = t.flip(-2)
        if offset:
            t = torch.cat([-t[:, :1], t[:, 1:]], 1)
    if k & 4:
        t = t.flip(-1)
        if offset:
            t = torch.cat([t[:, :1], -t[:, 1:]], 1)
    return t


# ── the original training distribution ───────────────────────────────────────

@lru_cache(maxsize=1)
def _bank():
    """32 crops of the synthetic frames the detector was trained on (yoloDiffraction
    ``EdgeDiskSpots``, disk radius 3-20 px, 128 px crops): the first 24 are replayed
    during adaptation, the last 8 are held out for :func:`original_f1`."""
    with resources.files("spyde.models.weights").joinpath("adapt_replay.npz").open("rb") as handle:
        z = np.load(handle)
        return dict(x=torch.from_numpy(z["x"].astype(np.float32))[:, None],
                    heatmap=torch.from_numpy(z["heatmap"].astype(np.float32) / 255.0)[:, None],
                    offset=torch.from_numpy(z["offset"].astype(np.float32)),
                    mask=torch.from_numpy(z["mask"].astype(np.float32)))


REPLAY, GAUGE = slice(0, 24), slice(24, 32)


def _replay_loss(model, rng, device, n=2):
    bank = _bank()
    index = rng.integers(REPLAY.start, REPLAY.stop, n)
    x = bank["x"][index].to(device)
    heatmap = bank["heatmap"][index].to(device)
    offset = bank["offset"][index].to(device)
    mask = bank["mask"][index].to(device)[:, None]
    logit, predicted = model(x)
    return _focal(logit, heatmap) + (F.l1_loss(predicted * mask, offset * mask, reduction="sum")
                                     / mask.sum().clamp_min(1))


@torch.no_grad()
def original_f1(model, device, threshold: float = 0.3) -> float:
    """F1 on held-out frames of the ORIGINAL synthetic training distribution
    (matched within 3 px). The bundled base model scores ~0.86 here; a fall
    measures how far adaptation pulled the model away from being a general
    disk detector."""
    bank = _bank()
    model.eval()
    x = bank["x"][GAUGE].to(device)
    hm, off = model(x)
    found = split_by_batch(decode_batch(hm, off, thresh=threshold, min_distance=2), x.shape[0])
    tp = fp = fn = 0
    for k, i in enumerate(range(GAUGE.start, GAUGE.stop)):
        mask = bank["mask"][i].numpy()
        offset = bank["offset"][i].numpy()
        ys, xs = np.nonzero(mask)
        centres = np.stack([ys + offset[0][ys, xs], xs + offset[1][ys, xs]], 1)
        truth = []
        for c in centres:
            if not truth or min(np.hypot(*(c - t)) for t in truth) > 2:
                truth.append(c)
        truth = np.asarray(truth).reshape(-1, 2)
        p = found[k][:, :2]
        matched = _greedy_matches(truth, p, 3.0)
        tp += matched
        fp += len(p) - matched
        fn += len(truth) - matched
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def _greedy_matches(a, b, tolerance) -> int:
    if not len(a) or not len(b):
        return 0
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    used_a, used_b, n = set(), set(), 0
    for i, j in zip(*np.unravel_index(np.argsort(d, axis=None), d.shape)):
        if d[i, j] > tolerance:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); n += 1
    return n


# ── the fine-tune ────────────────────────────────────────────────────────────

def adapt(base, device, frames, marks: list[np.ndarray], params: dict, *,
          steps: int = STEPS, learning_rate: float = LEARNING_RATE,
          replay_weight: float = REPLAY_WEIGHT, seed: int = 0, stop: list | None = None):
    """Fine-tune a copy of ``base`` on ``marks`` (one ``[y, x, label]`` array per
    pattern in ``frames``, original pixels). ``params`` are the detector
    parameters the user tuned (``spot_radius``, ``min_distance``, ``bg_sigma``,
    ``threshold``), so the copy is trained on exactly the input it will see.

    Runs on the calling thread. Every device step holds the process-wide
    accelerator lock (a no-op off Apple-MPS) and releases it between steps, so a
    live preview can interleave; backward is pinned to this thread. On CUDA the
    caller must have run ``spyde.torch_device.warmup_autograd()`` on the
    dispatching thread first (Windows: a first backward on an uninitialised
    thread segfaults). ``stop`` is a ``[False]`` flag polled between steps;
    setting it raises :class:`Cancelled`. Returns ``(model on CPU, report)``."""
    from spyde.device_lock import accelerator_lock

    started = time.perf_counter()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    disk_radius = float(params.get("spot_radius") or params.get("kernel_radius") or 5.0)
    with accelerator_lock(device):
        base = base.to(device).eval()
        model = copy.deepcopy(base).eval()           # BatchNorm statistics stay frozen
        working = Working(frames, params, int(getattr(base, "levels", 2)))
        labels = mark_targets(base, working, marks, disk_radius, device)
        before = original_f1(base, device)
    for p in model.parameters():
        p.requires_grad_(True)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    previous = torch.is_grad_enabled()
    try:
        torch.autograd.set_multithreading_enabled(False)
    except Exception as error:                      # pragma: no cover - older torch
        log.debug("set_multithreading_enabled unavailable: %s", error)
    try:
        with torch.enable_grad():
            for _ in range(int(steps)):
                if stop is not None and stop[0]:
                    raise Cancelled()
                with accelerator_lock(device):
                    k = int(rng.integers(0, 8))
                    logit, off = model(_dihedral(labels["x"], k))
                    hm_base = _dihedral(labels["hm_base"], k)
                    weight = _dihedral(labels["weight"], k)
                    loss = (_mark_loss(logit, _dihedral(labels["target"], k), weight)
                            + DISTILL_WEIGHT * _hold_loss(logit, hm_base, weight)
                            + OFFSET_WEIGHT * _offset_hold(off, _dihedral(labels["off_base"], k, offset=True),
                                                           hm_base))
                    if replay_weight:
                        loss = loss + replay_weight * _replay_loss(model, rng, device)
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    optimiser.step()
    finally:
        try:
            torch.autograd.set_multithreading_enabled(previous)
        except Exception:                           # pragma: no cover
            pass
    for p in model.parameters():
        p.requires_grad_(False)
    with accelerator_lock(device):
        model.eval()
        after = original_f1(model, device)
        model = model.to("cpu")
    marks_all = np.concatenate([np.asarray(m).reshape(-1, 3) for m in marks]) if marks else np.zeros((0, 3))
    report = dict(device=str(device), seconds=time.perf_counter() - started,
                  patterns=len(frames), disk_marks=int((marks_all[:, 2] == DISK).sum()),
                  not_disk_marks=int((marks_all[:, 2] == NOT_DISK).sum()),
                  original_f1_base=before, original_f1=after, scale_factor=float(working.factor),
                  **{**HYPERPARAMETERS, "steps": int(steps), "learning_rate": float(learning_rate),
                     "replay_weight": float(replay_weight)})
    return model, report


# ── the model's icon ─────────────────────────────────────────────────────────

ICON_SIZE = 48


def preferred_input(model, device, *, size: int = ICON_SIZE, steps: int = 200, seed: int = 0) -> np.ndarray:
    """What the model treats as an ideal disk: the input that maximises its centre
    heatmap logit, found by gradient ascent from near-zero, with an L2 and a
    total-variation penalty so the answer is a smooth picture rather than noise.
    At the model's working resolution, where every disk is ~9 px across."""
    from spyde.device_lock import accelerator_lock
    model = model.to(device).eval()
    generator = torch.Generator().manual_seed(seed)
    with accelerator_lock(device), torch.enable_grad():
        x = (0.01 * torch.randn(1, 1, size, size, generator=generator)).to(device).requires_grad_(True)
        optimiser = torch.optim.Adam([x], lr=0.05)
        for _ in range(int(steps)):
            logit, _ = model(x)
            tv = (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
            loss = -logit[0, 0, size // 2, size // 2] + 0.02 * (x ** 2).mean() + 0.3 * tv
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        return x.detach().cpu().numpy()[0, 0]


def write_icon(model, device, path: str) -> None:
    """The model's icon: its preferred input (:func:`preferred_input`), the centre
    cropped, on a diverging scale centred at zero, as a PNG."""
    from matplotlib import colormaps
    from PIL import Image
    image = preferred_input(model, device)
    margin = ICON_SIZE // 8
    image = image[margin:-margin, margin:-margin]
    scale = float(np.percentile(np.abs(image), 99.5)) or 1.0
    rgba = colormaps["RdBu_r"](np.clip(0.5 + 0.5 * image / scale, 0, 1))
    Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).resize((64, 64), Image.BICUBIC).save(path)

"""A network centre refiner: segment the disk in a raw crop, take the segment's centroid.

The network is a learned version of :class:`~spyde.models.centre_refine.MaskCentroidRefiner`:
a small U-Net turns a normalised crop into a disk-coverage map, and the centre is
that map's centroid. A convolution moves with the disk, so the centre does too,
and the centroid can never leave the disk the way a regressed offset can.

Two network layouts load, told apart by their state-dict keys:

* :class:`CentreNet` (R1, R3): a two-level U-Net, coverage logits only.
* :class:`FastCentreNet` (F3 on): a narrower U-Net with ``levels`` poolings and,
  optionally, a sigma head — a linear layer on the mean-pooled bottleneck that
  predicts the log of the centre's standard deviation. About five times faster
  than R1 for the same accuracy on real GaN.

Checkpoint contract (``torch.load(weights_only=True)``): ``state_dict`` plus the
scalars ``base``, ``crop_radius`` and ``crop_half`` (and for the fast layout
``levels``, ``depthwise``, ``sigma_head``, ``normalisation``). A registry entry
with ``"kind": "refiner"`` may override any of them under ``arch`` / ``input``
and sets the inference options below.

Input contract, all relative to the spot radius ``R`` in native pixels:

* the crop is resampled (bilinear, zeros outside the frame) so the disk spans
  ``crop_radius`` crop pixels: a ``(2 * crop_half + 1)²`` grid at
  ``R / crop_radius`` native px per step, centred on the current centre;
* the background is taken over the annulus ``annulus[0]-annulus[1] x crop_half``
  (default 0.78-1.0), then one of two normalisations:

  - ``ring-median/disk-p95`` (CentreNet): subtract the annulus median, divide by
    the 95th percentile inside ``crop_radius``, clamp to [-3, 6];
  - ``mean`` (fast layout): subtract the annulus mean, divide by
    ``|mean inside crop_radius - background|`` (at least 1e-3), clamp to [-4, 8].

Inference options (``input`` keys):

* ``passes`` (default 2) and ``recrop_over`` (default 0) — after a pass, re-crop
  at the refined centre and run again, for the disks that pass moved by more
  than ``recrop_over x R``. The network is most accurate on a disk near the
  middle of its crop: on simulated crystals, 0.49 -> 0.43 px RMS at 3e5
  electrons. F3 re-crops only past 0.15 R.
* ``uncertainty`` — ``"head"`` (default when the network has a sigma head) or
  ``"mirror"``. See below.
* ``mirror_mean`` (default true, mirror uncertainty only) — the final pass's
  centre is the mean over the crop and its three mirrors. That pass runs the
  mirrors for the uncertainty anyway, so this costs nothing.
* ``min_spot_radius`` (default ``centre_refine.DEFAULT_MIN_SPOT_RADIUS``, 5 px,
  shared with the mask centroid) — below this spot radius (native px) the
  stage is skipped and the detector's centres are kept; 0 turns it off.

Uncertainty:

* ``"head"`` — ``exp(log_sigma) x sigma_scale x R / crop_radius``, one view.
  ``sigma_scale`` is the checkpoint's own calibration factor (the trainer
  stores it per checkpoint; 1 when absent; a contract value wins). On real
  disks the head RANKS centres well (Spearman 0.42-0.66 against a
  Friedel-midpoint test, where the mirror spread scores 0.0-0.3), but its
  absolute scale is uncertain to about 2x either way: use it to rank vectors or
  weight them relative to each other, not as a calibrated error bar.
* ``"mirror"`` — the network also runs on the crop flipped in x, in y and in
  both; the three answers are flipped back, and ``sigma_per_spread`` (6) x the
  spread (mean distance of the four centres from their mean, native px) is the
  sigma. The factor is calibrated on held-out-crystal simulation, where the true
  error RMS was 6.0x the spread in every quintile (Spearman 0.76).

A disk is declined — NaN, so the stage keeps its detected centre — when it moved
``max_shift_fraction`` of a spot radius or more (0.5; a checkpoint's own
``decline_moved_over`` wins — an adapted network corrects its scan's
detections by more, and carries 0.75), when its coverage is under a fifth of the expected
disk, or when its sigma is ``max_sigma_fraction`` (0.25) of the spot radius or more.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .centre_refine import DEFAULT_MIN_SPOT_RADIUS
from .centre_refine import MAX_SHIFT_FRACTION as DEFAULT_MAX_SHIFT_FRACTION

NORMALISATION_RING_MEDIAN_DISK_P95 = "ring-median/disk-p95"
NORMALISATION_MEAN = "mean"
NORMALISATIONS = (NORMALISATION_RING_MEDIAN_DISK_P95, NORMALISATION_MEAN)

UNCERTAINTY_HEAD = "head"
UNCERTAINTY_MIRROR = "mirror"

DEFAULT_CROP_RADIUS = 10.0
DEFAULT_CROP_HALF = 16
DEFAULT_ANNULUS = (0.78, 1.0)
DEFAULT_PASSES = 2
DEFAULT_SIGMA_PER_SPREAD = 6.0
DEFAULT_MAX_SIGMA_FRACTION = 0.25

# Crops per forward. Four mirrored views of a 4096-crop batch at 41 x 41 would
# hold several GB of activations at once; this keeps one forward to ~0.5 GB.
FORWARD_BATCH = 1024

# The flips of a (..., y, x) crop and the sign that maps a [y, x] offset found
# in the flipped crop back to the original one.
_MIRRORS = (((), (1.0, 1.0)), ((-1,), (1.0, -1.0)), ((-2,), (-1.0, 1.0)),
            ((-2, -1), (-1.0, -1.0)))


def _block(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True),
        nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True))


def _pad_to_multiple(x, multiple):
    """Replicate-pad a square crop to a side divisible by ``multiple`` (the
    poolings must line up); returns the padded crop and the leading pad."""
    size = x.shape[-1]
    pad_total = (-size) % multiple
    before, after = pad_total // 2, pad_total - pad_total // 2
    return F.pad(x, (before, after, before, after), mode="replicate"), before


class CentreNet(nn.Module):
    """Two-level U-Net: one normalised crop channel in, disk-coverage logits out.
    Layer names are the checkpoint's state-dict keys; do not rename them."""

    def __init__(self, base: int = 16):
        super().__init__()
        self.enc0 = _block(1, base)
        self.enc1 = _block(base, 2 * base)
        self.enc2 = _block(2 * base, 4 * base)
        self.up1 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec1 = _block(4 * base, 2 * base)
        self.up0 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec0 = _block(2 * base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        size = x.shape[-1]
        x, before = _pad_to_multiple(x, 4)
        e0 = self.enc0(x)
        e1 = self.enc1(F.max_pool2d(e0, 2))
        e2 = self.enc2(F.max_pool2d(e1, 2))
        d1 = self.dec1(torch.cat([self.up1(e2), e1], 1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], 1))
        return self.head(d0)[:, :, before:before + size, before:before + size]


def _fast_convolution(channels_in, channels_out, depthwise):
    if depthwise and channels_in > 1:
        return nn.Sequential(
            nn.Conv2d(channels_in, channels_in, 3, padding=1, groups=channels_in, bias=False),
            nn.Conv2d(channels_in, channels_out, 1, bias=False))
    return nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False)


def _fast_block(channels_in, channels_out, depthwise):
    return nn.Sequential(
        _fast_convolution(channels_in, channels_out, depthwise),
        nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True),
        _fast_convolution(channels_out, channels_out, depthwise),
        nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True))


class FastCentreNet(nn.Module):
    """U-Net with ``levels`` poolings: coverage logits and, with a sigma head,
    the log of the centre's standard deviation in crop pixels. Layer names are
    the checkpoint's state-dict keys; do not rename them."""

    def __init__(self, base: int = 8, levels: int = 2, depthwise: bool = False,
                 sigma_head: bool = True):
        super().__init__()
        self.levels = levels
        widths = [base * 2 ** level for level in range(levels + 1)]
        self.encoders = nn.ModuleList([
            _fast_block(1 if level == 0 else widths[level - 1], widths[level], depthwise)
            for level in range(levels + 1)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(widths[level], widths[level - 1], 2, stride=2)
                                  for level in range(levels, 0, -1)])
        self.decoders = nn.ModuleList([_fast_block(2 * widths[level - 1], widths[level - 1], depthwise)
                                       for level in range(levels, 0, -1)])
        self.head = nn.Conv2d(base, 1, 1)
        self.sigma_head = nn.Linear(widths[-1], 1) if sigma_head else None

    def forward(self, x):
        size = x.shape[-1]
        x, before = _pad_to_multiple(x, 2 ** self.levels)
        features = []
        hidden = x
        for level, encoder in enumerate(self.encoders):
            hidden = encoder(hidden if level == 0 else F.max_pool2d(hidden, 2))
            features.append(hidden)
        decoded = features[-1]
        for step, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            decoded = decoder(torch.cat([up(decoded), features[self.levels - 1 - step]], 1))
        logits = self.head(decoded)[:, :, before:before + size, before:before + size]
        log_sigma = (self.sigma_head(features[-1].mean((2, 3)))[:, 0]
                     if self.sigma_head is not None else None)
        return logits, log_sigma


class NetworkRefiner:
    """A :class:`~spyde.models.centre_refine.CentreRefiner` backed by a centre
    network. The network returns coverage logits, or ``(logits, log_sigma)``."""

    def __init__(self, net, device, crop_radius: float = DEFAULT_CROP_RADIUS,
                 crop_half: int = DEFAULT_CROP_HALF, annulus=DEFAULT_ANNULUS,
                 normalisation: str = NORMALISATION_RING_MEDIAN_DISK_P95,
                 passes: int = DEFAULT_PASSES, recrop_over: float = 0.0,
                 uncertainty: str = UNCERTAINTY_MIRROR, mirror_mean: bool = True,
                 sigma_per_spread: float = DEFAULT_SIGMA_PER_SPREAD,
                 max_sigma_fraction: float = DEFAULT_MAX_SIGMA_FRACTION,
                 sigma_scale: float = 1.0,
                 max_shift_fraction: float = DEFAULT_MAX_SHIFT_FRACTION,
                 min_spot_radius: float = DEFAULT_MIN_SPOT_RADIUS):
        if normalisation not in NORMALISATIONS:
            raise ValueError(f"unknown refiner normalisation {normalisation!r}; "
                             f"known: {NORMALISATIONS}")
        if uncertainty not in (UNCERTAINTY_HEAD, UNCERTAINTY_MIRROR):
            raise ValueError(f"unknown refiner uncertainty {uncertainty!r}")
        self.net = net
        self.device = device
        self.crop_radius = float(crop_radius)
        self.crop_half = int(crop_half)
        self.annulus = (float(annulus[0]), float(annulus[1]))
        self.normalisation = normalisation
        self.passes = max(1, int(passes))
        self.recrop_over = float(recrop_over)
        self.uncertainty = uncertainty
        self.mirror_mean = bool(mirror_mean)
        self.sigma_per_spread = float(sigma_per_spread)
        self.max_sigma_fraction = float(max_sigma_fraction)
        self.sigma_scale = float(sigma_scale)
        self.max_shift_fraction = float(max_shift_fraction)
        self.min_spot_radius = float(min_spot_radius)

    def crop_half_width(self, spot_radius: float) -> int:
        # The resampled grid reaches crop_half / crop_radius radii from the
        # centre, which sits up to half a pixel off the crop centre and, on a
        # re-crop pass, may have moved by up to the largest accepted shift.
        # One more pixel keeps the bilinear samples inside the crop.
        reach = self.crop_half / self.crop_radius * float(spot_radius)
        if self.passes > 1:
            reach += self.max_shift_fraction * float(spot_radius)
        return int(math.ceil(reach + 0.5)) + 1

    def _offsets(self):
        return torch.arange(-self.crop_half, self.crop_half + 1,
                            device=self.device, dtype=torch.float32)

    def _resample(self, crops, centres, step):
        size = crops.shape[-1]
        offsets = self._offsets()
        rows = centres[:, 0, None] + offsets[None] * step          # (M, S') crop px
        columns = centres[:, 1, None] + offsets[None] * step
        samples = rows.shape[1]
        grid = torch.stack([
            (columns[:, None, :].expand(-1, samples, -1) / (size - 1)) * 2 - 1,
            (rows[:, :, None].expand(-1, -1, samples) / (size - 1)) * 2 - 1], -1)
        return F.grid_sample(crops[:, None], grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)

    def _normalise(self, sampled):
        count = sampled.shape[0]
        offsets = self._offsets()
        distance = torch.sqrt(offsets[:, None] ** 2 + offsets[None] ** 2).reshape(-1)
        inner, outer = self.annulus
        ring = (distance > inner * self.crop_half) & (distance <= outer * self.crop_half)
        disk = distance <= self.crop_radius
        flat = sampled.reshape(count, -1)
        if self.normalisation == NORMALISATION_MEAN:
            background = flat[:, ring].mean(1)
            level = (flat[:, disk].mean(1) - background).abs().clamp_min(1e-3)
            return ((flat - background[:, None]) / level[:, None]).reshape_as(sampled).clamp(-4, 8)
        background = flat[:, ring].median(1).values
        shifted = flat - background[:, None]
        top = torch.quantile(shifted[:, disk], 0.95, dim=1).clamp_min(1e-3)
        return (shifted / top[:, None]).reshape_as(sampled).clamp(-3, 6)

    def _coverage_centroid(self, logits, iterations=2):
        coverage = torch.sigmoid(logits[:, 0])
        offsets = self._offsets()
        rows, columns = offsets[None, :, None], offsets[None, None, :]
        centre_y = torch.zeros(coverage.shape[0], device=coverage.device)
        centre_x = torch.zeros_like(centre_y)
        window = torch.full_like(centre_y, 1.3 * self.crop_radius + 1.5)
        area = torch.zeros_like(centre_y)
        for _ in range(iterations):
            distance = torch.sqrt((rows - centre_y[:, None, None]) ** 2
                                  + (columns - centre_x[:, None, None]) ** 2)
            weight = (window[:, None, None] - distance + 0.5).clamp(0, 1) * coverage
            area = weight.sum((1, 2)).clamp_min(1e-3)
            centre_y = (weight * rows).sum((1, 2)) / area
            centre_x = (weight * columns).sum((1, 2)) / area
            window = torch.sqrt(area / math.pi) + 2.0
        return torch.stack([centre_y, centre_x], 1), area

    def _forward(self, x):
        """``(logits, log_sigma or None)`` for a crop stack, ``FORWARD_BATCH`` at a time."""
        logits, log_sigmas = [], []
        for start in range(0, len(x), FORWARD_BATCH):
            output = self.net(x[start:start + FORWARD_BATCH])
            if isinstance(output, tuple):
                output, log_sigma = output
                log_sigmas.append(log_sigma)
            logits.append(output)
        return torch.cat(logits), (torch.cat(log_sigmas) if log_sigmas else None)

    def _views(self, x, mirrored: bool):
        """Offsets (crop px from the crop centre) found in the crop and, when
        ``mirrored``, in its three mirrors flipped back: ``(V, M, 2)``. Also the
        plain crop's coverage area and log sigma (``None`` without a head)."""
        mirrors = _MIRRORS if mirrored else _MIRRORS[:1]
        views, area, log_sigma = [], None, None
        for dims, sign in mirrors:
            view = x.flip(dims) if dims else x
            logits, view_log_sigma = self._forward(view)
            offset, view_area = self._coverage_centroid(logits)
            views.append(offset * torch.tensor(sign, device=offset.device))
            if area is None:
                area, log_sigma = view_area, view_log_sigma
        return torch.stack(views), area, log_sigma

    def _pass(self, crop_tensor, centre, step, last):
        """One pass at ``centre``: the new centre, coverage area and sigma
        (native px; ``None`` until the final pass of the mirror uncertainty)."""
        mirrored = self.uncertainty == UNCERTAINTY_MIRROR and last
        sampled = self._normalise(self._resample(crop_tensor, centre, step))
        views, area, log_sigma = self._views(sampled, mirrored)
        offset = views.mean(0) if mirrored and self.mirror_mean else views[0]
        if self.uncertainty == UNCERTAINTY_HEAD:
            sigma = torch.exp(log_sigma) * step * self.sigma_scale
        elif mirrored:
            sigma = self.sigma_per_spread * (views - views.mean(0)).norm(dim=2).mean(0) * step
        else:
            sigma = None
        return centre + offset * step, area, sigma

    @torch.no_grad()
    def refine(self, crops, centres, spot_radius):
        from spyde.device_lock import accelerator_lock

        count = len(crops)
        if count == 0:
            return np.zeros((0, 2)), np.zeros(0, np.float32)
        radius = float(spot_radius)
        step = radius / self.crop_radius
        with accelerator_lock(self.device):
            crop_tensor = torch.as_tensor(np.asarray(crops, np.float32), device=self.device)
            seeds = torch.as_tensor(np.asarray(centres, np.float32), device=self.device)
            centre, area, sigma = self._pass(crop_tensor, seeds, step,
                                             last=self.passes == 1)
            for pass_index in range(1, self.passes):
                last = pass_index == self.passes - 1
                again = (centre - seeds).norm(dim=1) > self.recrop_over * radius
                if self.uncertainty == UNCERTAINTY_MIRROR and last:
                    again[:] = True      # the mirrors for sigma run on every disk
                if not again.any():
                    break
                moved, moved_area, moved_sigma = self._pass(
                    crop_tensor[again], centre[again], step, last)
                centre[again], area[again] = moved, moved_area
                if moved_sigma is not None:
                    if sigma is None:
                        sigma = torch.full_like(area, float("nan"))
                    sigma[again] = moved_sigma
            declined = ((centre - seeds).norm(dim=1) >= self.max_shift_fraction * radius) \
                | (area <= 0.2 * math.pi * self.crop_radius ** 2) \
                | (sigma >= self.max_sigma_fraction * radius)
            centre[declined] = float("nan")
            sigma[declined] = float("nan")
            result = centre.double().cpu().numpy()
            sigma = sigma.float().cpu().numpy()
        return result, sigma


def _is_fast_layout(state_dict) -> bool:
    return any(key.startswith("encoders.") for key in state_dict)


def load_refiner(path, device, arch: dict | None = None, contract: dict | None = None):
    """A :class:`NetworkRefiner` from a checkpoint of either layout; registry
    ``arch`` / ``input`` values win over the checkpoint's own scalars."""
    from spyde.device_lock import accelerator_lock

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    arch = arch or {}
    contract = contract or {}

    def setting(key, default):
        return arch.get(key, checkpoint.get(key, default))

    if _is_fast_layout(checkpoint["state_dict"]):
        net = FastCentreNet(int(setting("base", 8)), int(setting("levels", 2)),
                            bool(setting("depthwise", 0)), bool(setting("sigma_head", 1)))
        default_normalisation = checkpoint.get("normalisation", NORMALISATION_MEAN)
        has_head = net.sigma_head is not None
    else:
        net = CentreNet(int(setting("base", 16)))
        default_normalisation = NORMALISATION_RING_MEDIAN_DISK_P95
        has_head = False
    net.load_state_dict(checkpoint["state_dict"])
    with accelerator_lock(device):
        net = net.to(device).eval()
    return NetworkRefiner(
        net, device,
        crop_radius=float(contract.get("crop_radius",
                                       checkpoint.get("crop_radius", DEFAULT_CROP_RADIUS))),
        crop_half=int(contract.get("crop_half", checkpoint.get("crop_half", DEFAULT_CROP_HALF))),
        annulus=tuple(contract.get("annulus", DEFAULT_ANNULUS)),
        normalisation=contract.get("normalisation", default_normalisation),
        passes=int(contract.get("passes", DEFAULT_PASSES)),
        recrop_over=float(contract.get("recrop_over", 0.0)),
        uncertainty=contract.get("uncertainty",
                                 UNCERTAINTY_HEAD if has_head else UNCERTAINTY_MIRROR),
        mirror_mean=bool(contract.get("mirror_mean", True)),
        sigma_per_spread=float(contract.get("sigma_per_spread", DEFAULT_SIGMA_PER_SPREAD)),
        max_sigma_fraction=float(contract.get("max_sigma_fraction",
                                              DEFAULT_MAX_SIGMA_FRACTION)),
        min_spot_radius=float(contract.get("min_spot_radius", DEFAULT_MIN_SPOT_RADIUS)),
        sigma_scale=float(contract.get("sigma_scale", checkpoint.get("sigma_scale", 1.0))),
        max_shift_fraction=float(contract.get(
            "decline_moved_over", checkpoint.get("decline_moved_over", DEFAULT_MAX_SHIFT_FRACTION))))

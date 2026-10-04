"""A network centre refiner: segment the disk in a raw crop, take the segment's centroid.

The network is a learned version of :class:`~spyde.models.centre_refine.MaskCentroidRefiner`:
a small two-level U-Net turns a normalised crop into a disk-coverage map, and the
centre is that map's centroid. A convolution moves with the disk, so the centre
does too, and the centroid can never leave the disk the way a regressed offset can.

Checkpoint contract (``torch.load(weights_only=True)``): ``state_dict`` plus the
scalars ``base`` (channel width), ``crop_radius`` and ``crop_half``. A registry
entry with ``"kind": "refiner"`` may override any of them under ``arch`` /
``input`` and sets the inference options below; ``input.normalisation`` must
name a normalisation this module knows.

Input contract, all relative to the spot radius ``R`` in native pixels:

* the crop is resampled (bilinear) so the disk spans ``crop_radius`` crop pixels:
  a ``(2 * crop_half + 1)²`` grid at ``R / crop_radius`` native px per step,
  centred on the current centre. ``crop_half`` is per checkpoint (16, or 20 from
  R3 on);
* ``ring-median/disk-p95`` normalisation: subtract the median of the annulus
  ``annulus[0]-annulus[1] x crop_half`` from the centre (default 0.78-1.0),
  divide by the 95th percentile inside ``crop_radius``, clamp to [-3, 6].

Inference options (``input`` keys):

* ``passes`` (default 2) — after a pass, re-crop at the refined centre and run
  again. The network is most accurate on a disk near the middle of its crop:
  on simulated crystals, 0.49 -> 0.43 px RMS at 3e5 electrons.
* ``mirror_mean`` (default true) — the final pass's centre is the mean over the
  crop and its three mirrors instead of the plain crop alone. The final pass
  runs the mirrors for the uncertainty anyway, so this costs nothing; earlier
  passes only re-centre the crop and use the plain crop.

The uncertainty comes from the mirrors whatever ``mirror_mean`` says: the network
also runs on the crop flipped in x, in y and in both, the three answers are
flipped back, and the spread (mean distance of the four centres from their
mean, native px) times ``sigma_per_spread`` is the centre's sigma. The factor 6
is calibrated on held-out-crystal simulation, where the true error RMS was 6.0x
the spread in every quintile (Spearman 0.76). A disk whose sigma exceeds
``max_sigma_fraction`` (0.25) of the spot radius, or whose coverage is under a
fifth of the expected disk, is declined.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NORMALISATION_RING_MEDIAN_DISK_P95 = "ring-median/disk-p95"
NORMALISATIONS = (NORMALISATION_RING_MEDIAN_DISK_P95,)

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
        # Two poolings need a side divisible by 4; pad, then cut the pad back off.
        size = x.shape[-1]
        pad_total = (-size) % 4
        before, after = pad_total // 2, pad_total - pad_total // 2
        x = F.pad(x, (before, after, before, after), mode="replicate")
        e0 = self.enc0(x)
        e1 = self.enc1(F.max_pool2d(e0, 2))
        e2 = self.enc2(F.max_pool2d(e1, 2))
        d1 = self.dec1(torch.cat([self.up1(e2), e1], 1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], 1))
        return self.head(d0)[:, :, before:before + size, before:before + size]


class NetworkRefiner:
    """A :class:`~spyde.models.centre_refine.CentreRefiner` backed by :class:`CentreNet`."""

    def __init__(self, net: nn.Module, device, crop_radius: float = DEFAULT_CROP_RADIUS,
                 crop_half: int = DEFAULT_CROP_HALF, annulus=DEFAULT_ANNULUS,
                 normalisation: str = NORMALISATION_RING_MEDIAN_DISK_P95,
                 passes: int = DEFAULT_PASSES, mirror_mean: bool = True,
                 sigma_per_spread: float = DEFAULT_SIGMA_PER_SPREAD,
                 max_sigma_fraction: float = DEFAULT_MAX_SIGMA_FRACTION):
        if normalisation not in NORMALISATIONS:
            raise ValueError(f"unknown refiner normalisation {normalisation!r}; "
                             f"known: {NORMALISATIONS}")
        self.net = net
        self.device = device
        self.crop_radius = float(crop_radius)
        self.crop_half = int(crop_half)
        self.annulus = (float(annulus[0]), float(annulus[1]))
        self.passes = max(1, int(passes))
        self.mirror_mean = bool(mirror_mean)
        self.sigma_per_spread = float(sigma_per_spread)
        self.max_sigma_fraction = float(max_sigma_fraction)

    def crop_half_width(self, spot_radius: float) -> int:
        # The resampled grid reaches crop_half / crop_radius radii from the
        # centre, which sits up to half a pixel off the crop centre and, on a
        # re-crop pass, may have moved by the stage's largest accepted shift.
        # One more pixel keeps the bilinear samples inside the crop.
        from .centre_refine import MAX_SHIFT_FRACTION

        reach = self.crop_half / self.crop_radius * float(spot_radius)
        if self.passes > 1:
            reach += MAX_SHIFT_FRACTION * float(spot_radius)
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
        return torch.cat([self.net(x[start:start + FORWARD_BATCH])
                          for start in range(0, len(x), FORWARD_BATCH)])

    def _views(self, x, mirrored: bool):
        """Offsets (crop px from the crop centre) found in the crop and, when
        ``mirrored``, in its three mirrors flipped back: ``(V, M, 2)``. Also the
        plain crop's coverage area."""
        mirrors = _MIRRORS if mirrored else _MIRRORS[:1]
        views, area = [], None
        for dims, sign in mirrors:
            view = x.flip(dims) if dims else x
            offset, view_area = self._coverage_centroid(self._forward(view))
            views.append(offset * torch.tensor(sign, device=offset.device))
            if area is None:
                area = view_area
        return torch.stack(views), area

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
            centre = torch.as_tensor(np.asarray(centres, np.float32), device=self.device)
            for pass_index in range(self.passes):
                last = pass_index == self.passes - 1
                sampled = self._normalise(self._resample(crop_tensor, centre, step))
                views, area = self._views(sampled, mirrored=last)
                offset = views.mean(0) if self.mirror_mean and last else views[0]
                centre = centre + offset * step
            spread = (views - views.mean(0)).norm(dim=2).mean(0) * step
            sigma = self.sigma_per_spread * spread
            declined = ((area <= 0.2 * math.pi * self.crop_radius ** 2)
                        | (sigma > self.max_sigma_fraction * radius))
            centre[declined] = float("nan")
            sigma[declined] = float("nan")
            result = centre.double().cpu().numpy()
            sigma = sigma.float().cpu().numpy()
        return result, sigma


def load_refiner(path, device, arch: dict | None = None, contract: dict | None = None):
    """A :class:`NetworkRefiner` from a checkpoint; registry ``arch`` / ``input``
    values win over the checkpoint's own scalars."""
    from spyde.device_lock import accelerator_lock

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    arch = arch or {}
    contract = contract or {}
    net = CentreNet(int(arch.get("base", checkpoint.get("base", 16))))
    net.load_state_dict(checkpoint["state_dict"])
    with accelerator_lock(device):
        net = net.to(device).eval()
    return NetworkRefiner(
        net, device,
        crop_radius=float(contract.get("crop_radius",
                                       checkpoint.get("crop_radius", DEFAULT_CROP_RADIUS))),
        crop_half=int(contract.get("crop_half", checkpoint.get("crop_half", DEFAULT_CROP_HALF))),
        annulus=tuple(contract.get("annulus", DEFAULT_ANNULUS)),
        normalisation=contract.get("normalisation", NORMALISATION_RING_MEDIAN_DISK_P95),
        passes=int(contract.get("passes", DEFAULT_PASSES)),
        mirror_mean=bool(contract.get("mirror_mean", True)),
        sigma_per_spread=float(contract.get("sigma_per_spread", DEFAULT_SIGMA_PER_SPREAD)),
        max_sigma_fraction=float(contract.get("max_sigma_fraction",
                                              DEFAULT_MAX_SIGMA_FRACTION)))

"""Where a detected disk's centre is, and how well that is known.

``decode`` (vendored, frozen) places a disk at the heatmap's maximum pixel plus
the network's offset head. That is fine when the disk is evenly lit, but under
dynamical scattering the intensity inside a disk is uneven and moves, and a
single pixel plus a learned offset follows it. Taking the **soft-argmax** of
the heatmap around that pixel — the softmax-weighted mean position of the
heatmap logits within the disk — uses the whole response the network gives to
the disk instead of one pixel of it. On synthetic dynamical patterns with known
centres it lands nearer the true outline centre: 0.33 px RMS against 0.37 px
for the decode (``spyde/tests/migrated/test_vector_confidence.py``).

Each detection also gets:

* ``confidence`` — the heatmap peak (sigmoid), 0..1: how sure the network is that
  there is a disk here. A faded reflection scores low.
* ``width`` — the spread (one standard deviation, working pixels) of the same
  softmax weights. Turned into a positional uncertainty by
  :func:`positional_sigma` once the disk's counts are known.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

CENTRE_SOFTARGMAX = "softargmax"
CENTRE_OFFSET = "offset"          # the frozen decode: argmax pixel + offset head
CENTRES = (CENTRE_SOFTARGMAX, CENTRE_OFFSET)


@torch.no_grad()
def decode_batch_centres(hm_logits, off, thresh=0.3, min_distance=3, radius=3.0,
                         centre: str = CENTRE_SOFTARGMAX):
    """Peaks of a ``(B,1,H,W)`` heatmap batch, as ``(M,5)``
    ``[batch, y, x, confidence, width]`` in working pixels.

    The peaks themselves are exactly ``decode.decode_batch``'s (same max-pool,
    same threshold), so switching ``centre`` changes where a disk is placed,
    never which disks are found. ``radius`` (working px) is the window the
    soft-argmax averages over — the disk's own radius."""
    if centre not in CENTRES:
        raise ValueError(f"centre must be one of {CENTRES}; got {centre!r}")
    hm = torch.sigmoid(hm_logits)
    B, _, H, W = hm.shape
    k = 2 * min_distance + 1
    pooled = F.max_pool2d(hm, k, stride=1, padding=min_distance)
    idx = ((hm == pooled) & (hm >= thresh)).squeeze(1).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return torch.zeros((0, 5), device=hm.device)
    b, ys, xs = idx[:, 0], idx[:, 1], idx[:, 2]
    confidence = hm[b, 0, ys, xs]

    half = max(1, int(math.ceil(radius)))
    d = torch.arange(-half, half + 1, device=hm.device)
    dy, dx = torch.meshgrid(d, d, indexing="ij")
    inside = (dy * dy + dx * dx) <= radius * radius + 1e-6
    gy = (ys[:, None, None] + dy).clamp(0, H - 1)
    gx = (xs[:, None, None] + dx).clamp(0, W - 1)
    logits = hm_logits[b[:, None, None], 0, gy, gx]
    logits = torch.where(inside & (gy == ys[:, None, None] + dy) & (gx == xs[:, None, None] + dx),
                         logits, torch.full_like(logits, -float("inf")))
    w = torch.softmax(logits.flatten(1), 1).reshape(logits.shape)
    my = (w * gy).sum((1, 2))
    mx = (w * gx).sum((1, 2))
    var = 0.5 * ((w * (gy - my[:, None, None]) ** 2).sum((1, 2))
                 + (w * (gx - mx[:, None, None]) ** 2).sum((1, 2)))
    width = torch.sqrt(var.clamp_min(0))
    if centre == CENTRE_SOFTARGMAX:
        y, x = my, mx
    else:
        y = ys.float() + off[b, 0, ys, xs]
        x = xs.float() + off[b, 1, ys, xs]
    return torch.stack([b.float(), y, x, confidence, width], 1)


def positional_sigma(width_px, signal, background_per_px, pixel=1.0):
    """One standard deviation of a centre, in detector pixels.

    The localisation-precision formula for a spot of standard deviation ``s``
    measured with ``N`` signal counts over a background of ``b²`` counts per
    pixel, pixel size ``a`` (Thompson, Larson & Webb, Biophys. J. 82, 2775 (2002)):

        sigma² = (s² + a²/12) / N  +  8 π s⁴ b² / (a² N²)

    Here ``s`` is the width of the network's response to the disk (``width`` from
    :func:`decode_batch_centres`, in detector pixels), ``N`` the disk's
    background-subtracted counts and ``b²`` its local background level (Poisson,
    so the variance equals the mean). A bright, sharp disk gets a small sigma; a
    faint or diffuse one a large sigma. NaN where the disk has no net signal."""
    s = np.maximum(np.asarray(width_px, np.float64), 0.5)
    n = np.asarray(signal, np.float64)
    b2 = np.clip(np.asarray(background_per_px, np.float64), 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s * s + pixel * pixel / 12.0) / n + 8 * np.pi * s ** 4 * b2 / (pixel * pixel * n * n)
        sigma = np.sqrt(var)
    return np.where(n > 0, sigma, np.nan).astype(np.float32)

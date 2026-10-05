"""Heatmap+offset decoding for the SpotUNet detector.

Vendored from the ``yoloDiffraction`` research project
(``yolodiffraction/model/targets.py``) — only the DECODE side is carried over
(the training target-rendering and losses are not needed for inference). DO NOT
change the maths here; it must match how the checkpoints were trained.

``mask_centroid`` is SpyDE's own: for a model with a mask head it replaces the
offset-head position by the centroid of the predicted disk coverage, which is the
geometric centre however the intensity is distributed inside the disk.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def mask_centroid(mask_logits, batch, y, x, radius, iterations=2):
    """Centroid of the predicted disk coverage around each seed (working px).

    ``mask_logits`` is ``(B,1,H,W)``; ``batch``/``y``/``x`` index the seeds. The window
    is a soft disk (1 px ramp) so the centroid moves continuously with the seed: the
    first pass uses ``radius`` (the expected disk radius), the next the radius the mask
    itself implies (area / pi), centred on the previous centroid. Returns the centred
    (y, x) and the covered area, both ``(M,)``."""
    coverage = torch.sigmoid(mask_logits)
    _, _, H, W = coverage.shape
    half = int(np.ceil(1.6 * radius + 3))
    steps = torch.arange(-half, half + 1, device=coverage.device, dtype=torch.float32)
    rows = y.round()[:, None] + steps[None]                                   # (M, S)
    columns = x.round()[:, None] + steps[None]
    inside = (((rows >= 0) & (rows < H))[:, :, None]
              & ((columns >= 0) & (columns < W))[:, None, :])
    patch = coverage[batch[:, None, None], 0,
                     rows.long().clamp(0, H - 1)[:, :, None],
                     columns.long().clamp(0, W - 1)[:, None, :]] * inside
    rows, columns = rows[:, :, None], columns[:, None, :]
    centre_y, centre_x = y.clone(), x.clone()
    window = torch.full_like(y, 1.3 * radius + 1.5)
    area = torch.zeros_like(y)
    for _ in range(iterations):
        distance = torch.sqrt((rows - centre_y[:, None, None]) ** 2 + (columns - centre_x[:, None, None]) ** 2)
        weight = (window[:, None, None] - distance + 0.5).clamp(0, 1) * patch
        area = weight.sum((1, 2))
        found = area > 1e-3
        centre_y = torch.where(found, (weight * rows).sum((1, 2)) / area.clamp_min(1e-6), centre_y)
        centre_x = torch.where(found, (weight * columns).sum((1, 2)) / area.clamp_min(1e-6), centre_x)
        window = torch.sqrt(area / np.pi) + 2.0
    return centre_y, centre_x, area


def _centre_on_mask(mask_logits, batch, y, x, radius):
    """Move each offset-head position to its mask centroid, unless the mask there is
    too small to be a disk (< 15 % of the expected area) or its centroid lands more
    than 1.5 px away (a neighbour's mask, or no mask at all)."""
    centre_y, centre_x, area = mask_centroid(mask_logits, batch, y, x, radius)
    trusted = ((torch.hypot(centre_y - y, centre_x - x) < 1.5)
               & (area > 0.15 * np.pi * radius ** 2))
    return torch.where(trusted, centre_y, y), torch.where(trusted, centre_x, x)


def decode(hm_logits, off, thresh=0.3, min_distance=3, topk=None, mask_logits=None, radius=4.5):
    """Decode one frame's (1,H,W) logits + (2,H,W) offsets -> (N,3) [y,x,score].

    Heatmap local-max via max-pool (>= thresh), then add the predicted subpixel
    offset at each peak pixel. With ``mask_logits`` ((1,H,W), a mask-head model) the
    position is the mask centroid (``_centre_on_mask``; ``radius`` = expected disk
    radius in working px).
    """
    hm = torch.sigmoid(hm_logits)
    if hm.dim() == 3:
        hm = hm.unsqueeze(0)
        off = off.unsqueeze(0)
        if mask_logits is not None:
            mask_logits = mask_logits.unsqueeze(0)
    B, _, H, W = hm.shape
    k = 2 * min_distance + 1
    pooled = F.max_pool2d(hm, k, stride=1, padding=min_distance)
    peak = (hm == pooled) & (hm >= thresh)
    out = []
    for b in range(B):
        ys, xs = torch.where(peak[b, 0])
        if topk is not None and len(ys) > topk:
            scores = hm[b, 0, ys, xs]
            sel = torch.argsort(scores, descending=True)[:topk]
            ys, xs = ys[sel], xs[sel]
        dy = off[b, 0, ys, xs]
        dx = off[b, 1, ys, xs]
        s = hm[b, 0, ys, xs]
        y, x = ys.float() + dy, xs.float() + dx
        if mask_logits is not None and len(ys):
            y, x = _centre_on_mask(mask_logits, torch.full_like(ys, b), y, x, radius)
        res = torch.stack([y, x, s], dim=1)
        out.append(res.detach().cpu().numpy())
    return out if B > 1 else out[0]


@torch.no_grad()
def decode_batch(hm_logits, off, thresh=0.3, min_distance=3, return_numpy=True,
                 mask_logits=None, radius=4.5):
    """Fully-batched, GPU-resident decode. Returns (M,4) [batch, y, x, score] for ALL
    peaks across the whole batch — ONE host transfer, no per-frame python loop.

    This is the high-throughput path (the old per-frame `decode` with torch.where +
    .cpu() per frame caps throughput at a few hundred fps; this stays on-GPU and is
    ~50x+ faster batched). Split back per frame on the host via the batch column.

    hm_logits: (B,1,H,W) logits. off: (B,2,H,W). Apply subpixel offset at each peak.
    mask_logits: optional (B,1,H,W) from a mask-head model; then positions are mask
    centroids (``_centre_on_mask``, ``radius`` = expected disk radius, working px).
    """
    hm = torch.sigmoid(hm_logits)
    B, _, H, W = hm.shape
    k = 2 * min_distance + 1
    pooled = F.max_pool2d(hm, k, stride=1, padding=min_distance)
    mask = (hm == pooled) & (hm >= thresh)            # (B,1,H,W) bool
    idx = mask.squeeze(1).nonzero(as_tuple=False)     # (M,3) [b, y, x] — ONE op
    if idx.numel() == 0:
        empty = torch.zeros((0, 4), device=hm.device)
        return empty.cpu().numpy() if return_numpy else empty
    b, ys, xs = idx[:, 0], idx[:, 1], idx[:, 2]
    dy = off[b, 0, ys, xs]
    dx = off[b, 1, ys, xs]
    s = hm[b, 0, ys, xs]
    y, x = ys.float() + dy, xs.float() + dx
    if mask_logits is not None:
        y, x = _centre_on_mask(mask_logits, b, y, x, radius)
    res = torch.stack([b.float(), y, x, s], dim=1)
    return res.cpu().numpy() if return_numpy else res


def split_by_batch(res, B):
    """Split (M,4) [batch,y,x,score] from decode_batch into a list of B (Ni,3)
    [y,x,score] arrays (host-side, cheap)."""
    out = [np.zeros((0, 3), np.float32) for _ in range(B)]
    if len(res) == 0:
        return out
    res = np.asarray(res)
    order = np.argsort(res[:, 0], kind="stable")
    res = res[order]
    bounds = np.searchsorted(res[:, 0], np.arange(B + 1) - 0.5)
    for i in range(B):
        if bounds[i + 1] > bounds[i]:
            out[i] = res[bounds[i]:bounds[i + 1], 1:].astype(np.float32)
    return out

"""preprocess.py — batched EBSD pattern correction (#70).

Every operation here is per-pattern and embarrassingly parallel, so the whole
scan goes to the GPU as one tensor and comes back corrected. No Python loop
over positions.

Why this matters for indexing rather than being cosmetic: normalised cross-
correlation is invariant to a constant gain and offset, but **not** to a
spatial gradient. A real detector's intensity falls off away from the screen
centre, and that gradient is a pattern in its own right — it is present in
every experimental pattern and in none of the simulated ones, so it dilutes
every score. Measured on the synthetic scan: an exact-orientation match scores
~0.92 with the background left in and >0.999 once it is removed. Indexing
still *works* without this (the gradient is the same for every dictionary
entry, so the ranking mostly survives), which is exactly why it is easy to
skip and then wonder why the scores look mediocre.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.device_lock import accelerator_lock
from spyde.ebsd._device import default_device, resolve_dtype

log = logging.getLogger(__name__)


def _as_stack(patterns):
    a = np.asarray(patterns)
    nav_shape = a.shape[:-2]
    return a.reshape(-1, *a.shape[-2:]), nav_shape


def _gaussian_kernel1d(sigma: float, device, dtype):
    import torch
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur(stack, sigma, device, dtype):
    """Separable gaussian blur over the last two axes, batched.

    Separable on purpose: two 1-D passes are O(2r) per pixel where a 2-D kernel
    is O(r^2), and the blur is applied to the whole scan at once.
    """
    import torch
    import torch.nn.functional as F

    k = _gaussian_kernel1d(sigma, device, dtype)
    r = (k.numel() - 1) // 2
    x = stack.unsqueeze(1)                                   # (N, 1, H, W)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x.squeeze(1)


def remove_background(patterns, *, method: str = "dynamic", sigma: float = 8.0,
                      static_reference=None, device=None, dtype="float32"):
    """Correct the detector background across a whole scan.

    Parameters
    ----------
    method : {"dynamic", "static", "both"}
        ``dynamic`` subtracts a heavily blurred copy of EACH pattern, removing
        that pattern's own smooth gradient. This is the one that matters for
        indexing and it needs no reference.

        ``static`` subtracts a single reference — by default the mean of the
        scan, which is what a flat-field image approximates. It removes
        fixed-pattern detector artefacts but NOT a per-pattern gradient.

        ``both`` applies static then dynamic, as a real workflow does.
    sigma : float
        Blur width in pixels for the dynamic pass. Must be large compared with
        the Kikuchi band width or it removes the signal along with the
        background — this is the one parameter that can silently destroy the
        data, hence the default of 8 px on a typical 60 px detector.

    Returns
    -------
    ndarray
        float32, same shape as the input.
    """
    import torch

    if method not in ("dynamic", "static", "both"):
        raise ValueError(f"unknown background method {method!r} "
                         f"(dynamic, static or both)")

    device = device or default_device()
    tdtype = getattr(torch, resolve_dtype(device, dtype))
    stack, nav_shape = _as_stack(patterns)

    with accelerator_lock(device):
        t = torch.as_tensor(stack, dtype=tdtype, device=device)

        if method in ("static", "both"):
            ref = (torch.as_tensor(np.asarray(static_reference), dtype=tdtype,
                                   device=device)
                   if static_reference is not None else t.mean(0))
            t = t - ref

        if method in ("dynamic", "both"):
            t = t - _blur(t, sigma, device, tdtype)

        return t.detach().cpu().numpy().reshape(*nav_shape, *stack.shape[-2:])


def average_dot_product_map(patterns, *, device=None, dtype="float32"):
    """Mean normalised dot product between each pattern and its 4 neighbours.

    A pattern-quality map: high inside a grain where neighbours look alike, low
    at boundaries and in poorly-diffracting regions. Computed on the whole scan
    at once rather than per position.
    """
    import torch

    device = device or default_device()
    tdtype = getattr(torch, resolve_dtype(device, dtype))
    a = np.asarray(patterns)
    if a.ndim != 4:
        raise ValueError("ADP needs a 2-D scan of 2-D patterns (ny, nx, H, W)")
    ny, nx = a.shape[:2]

    with accelerator_lock(device):
        t = torch.as_tensor(a.reshape(ny, nx, -1), dtype=tdtype, device=device)
        t = t - t.mean(-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        total = torch.zeros((ny, nx), dtype=tdtype, device=device)
        count = torch.zeros((ny, nx), dtype=tdtype, device=device)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ys = slice(max(0, dy), ny + min(0, dy))
            xs = slice(max(0, dx), nx + min(0, dx))
            ys2 = slice(max(0, -dy), ny + min(0, -dy))
            xs2 = slice(max(0, -dx), nx + min(0, -dx))
            dot = (t[ys, xs] * t[ys2, xs2]).sum(-1)
            total[ys, xs] += dot
            count[ys, xs] += 1.0
        return (total / count.clamp_min(1.0)).detach().cpu().numpy()

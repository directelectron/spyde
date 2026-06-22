"""
find_vectors_torch.py — batched window-normalised cross-correlation (NXCORR)
peak finding on torch (CUDA / Apple-MPS / CPU).

A fast path for ``_find_vectors_chunk``'s per-frame CPU loop: the WHOLE pipeline
for a nav block — NXCORR, local-max peak detection, and subpixel CoM — runs
batched on the GPU (Metal on a MacBook, CUDA elsewhere); only slicing the flat
peak list back per frame happens on the host. The method mirrors
``find_vectors._find_vectors_single_frame`` (Lewis-1995 NXCORR via
cross-correlation + integral-image window stats, maximum_filter local max, CoM
subpixel), so results match on real (sharp) peaks; on a saturated plateau the GPU
ridge handling is cleaner than the CPU's split peaks (the surface is identical).

The small disk template (radius ~5 px) makes a direct ``conv2d`` cheaper and
simpler than FFT (no next-fast-len juggling); the whole block is one batched
conv. GPU access is serialised by a lock — the dask threaded scheduler runs
several chunks concurrently and a single Metal/CUDA context shouldn't be hit
from many threads at once.
"""
from __future__ import annotations

import threading
from typing import Optional

import numpy as np

_TORCH_DEV = "unset"        # cache: torch.device | None
_GPU_LOCK = threading.Lock()


def torch_gpu_device():
    """Best torch GPU device — CUDA → Apple-MPS — or None if neither (plain CPU
    torch is NOT used here; the numpy path is already fine on CPU). Cached."""
    global _TORCH_DEV
    if _TORCH_DEV != "unset":
        return _TORCH_DEV
    try:
        import torch
        if torch.cuda.is_available():
            _TORCH_DEV = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and \
                torch.backends.mps.is_available():
            _TORCH_DEV = torch.device("mps")
        else:
            _TORCH_DEV = None
    except Exception:
        _TORCH_DEV = None
    return _TORCH_DEV


def _nxcorr_torch(frames: np.ndarray, kr: int, device, kernel_window_pad: int = 1):
    """(N,H,W) window-normalised cross-correlation in [-1,1] on ``device``.

    Same Lewis-1995 NXCORR as ``find_vectors._find_vectors_single_frame``
    (numerator = xcorr/n - win_mean*t_mean; denom = max(win_std*t_std, floor)),
    but the correlation is a direct ``conv2d`` (the disk is tiny, so conv beats
    FFT — and on MPS the odd 5-smooth FFT length is slow) and the window stats use
    a ones-kernel conv. This computes the TRUE linear cross-correlation; the numpy
    reference's circular-FFT correlation has a slight boundary/normalisation quirk,
    so this is marginally MORE correct, not bit-identical to it.

    NXCORR is invariant to scaling the template, so the disk is left binary."""
    import torch
    import torch.nn.functional as F

    f = torch.as_tensor(np.ascontiguousarray(frames, np.float32), device=device)
    N, H, W = f.shape
    ar = torch.arange(-kr, kr + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ar, ar, indexing="ij")
    disk = ((yy * yy + xx * xx) <= float(kr * kr)).to(torch.float32)   # binary
    kH = 2 * kr + 1
    n = float(kH * kH)
    t_mean = disk.mean()
    t_std = ((disk - t_mean) ** 2).sum().div(n).sqrt()

    # Step 1: cross-correlation numerator (conv2d IS cross-correlation).
    fp = F.pad(f[:, None], (kr, kr, kr, kr), mode="reflect")
    xcorr = F.conv2d(fp, disk[None, None]).squeeze(1)                  # (N,H,W)

    # Step 2: window mean/std over a (kr+pad)-radius box via ones-kernel convs.
    krw = kr + int(kernel_window_pad)
    kHw = 2 * krw + 1
    nwin = float(kHw * kHw)
    sp = F.pad(f[:, None], (krw, krw, krw, krw), mode="reflect")
    ones = torch.ones(1, 1, kHw, kHw, device=device)
    win_mean = F.conv2d(sp, ones).squeeze(1) / nwin
    win_msq = F.conv2d(sp * sp, ones).squeeze(1) / nwin
    win_std = (win_msq - win_mean ** 2).clamp_min(0.0).sqrt()

    # Step 3: normalise (per-frame global-std floor; numpy std is population).
    gstd = f.reshape(N, -1).std(dim=1, unbiased=False).clamp_min(1e-6).view(N, 1, 1)
    denom_floor = 0.01 * gstd * t_std
    numer = xcorr / n - win_mean * t_mean
    denom = torch.maximum(win_std * t_std, denom_floor)
    return (numer / denom).clamp_(-1.0, 1.0)


_QFIT_PINV: dict = {}     # device-keyed cache of the (6, ks*ks) LS pseudo-inverse


def _qfit_pinv(device, hw=2):
    """Least-squares pseudo-inverse mapping a (2hw+1)^2 neighbourhood → the 6
    coefficients of z = c0 + cx·x + cy·y + cxx·x² + cyy·y² + cxy·xy. Cached."""
    import torch
    key = (str(device), hw)
    P = _QFIT_PINV.get(key)
    if P is None:
        rng = range(-hw, hw + 1)
        A = np.array([[1.0, dx, dy, dx * dx, dy * dy, dx * dy]
                      for dy in rng for dx in rng], np.float64)   # row-major (dy,dx)
        P = torch.as_tensor(np.linalg.pinv(A).astype(np.float32), device=device)
        _QFIT_PINV[key] = P
    return P


def _fit_peaks_quadratic(raw, nn, yy, xx, device, hw=2):
    """Subpixel peak location by a batched 2D quadratic least-squares fit of the
    NXCORR neighbourhood (the surface near a correlation max is parabolic, so its
    fitted vertex IS the subpixel peak — more accurate + less biased than a CoM).
    Returns (dy, dx) offsets in pixels; falls back to 0 (the integer peak) where
    the fit isn't a clean concave maximum or the vertex leaves the window."""
    import torch
    _, H, W = raw.shape
    ks = 2 * hw + 1
    ar = torch.arange(-hw, hw + 1, device=device)
    gy = (yy[:, None, None] + ar[None, :, None]).clamp(0, H - 1).expand(-1, ks, ks)
    gx = (xx[:, None, None] + ar[None, None, :]).clamp(0, W - 1).expand(-1, ks, ks)
    win = raw[nn[:, None, None], gy, gx].reshape(-1, ks * ks)        # (M, ks*ks)
    c = win @ _qfit_pinv(device, hw).t()                            # (M, 6)
    cx, cy, cxx, cyy, cxy = c[:, 1], c[:, 2], c[:, 3], c[:, 4], c[:, 5]
    det = 4.0 * cxx * cyy - cxy * cxy                                # >0 for a max
    dx = (-2.0 * cyy * cx + cxy * cy) / det
    dy = (-2.0 * cxx * cy + cxy * cx) / det
    bad = ((det <= 1e-9) | (cxx >= 0) | (dx.abs() > 1.5) | (dy.abs() > 1.5)
           | ~torch.isfinite(dx) | ~torch.isfinite(dy))
    z = torch.zeros_like(dx)
    return torch.where(bad, z, dy), torch.where(bad, z, dx)


def find_vectors_torch_batch(
    frames: np.ndarray, kernel_radius: int, threshold: float, min_distance: int,
    *, subpixel: bool = True, beamstop_mask: Optional[np.ndarray] = None,
    device=None,
):
    """Find peaks in a batch of (N,H,W) frames. Returns a list of N arrays, each
    (Ni,3) float32 ``[ky_subpx, kx_subpx, nxcorr_value]`` — same as calling
    ``_find_vectors_single_frame`` on each frame, but batched on the GPU."""
    import torch
    import torch.nn.functional as F

    dev = device or torch_gpu_device()
    if dev is None:
        raise RuntimeError("no torch GPU device available")
    kr, md, thr = int(kernel_radius), int(min_distance), float(threshold)
    frames = np.asarray(frames, np.float32)
    # Beam stop = PEAK REJECTION, not an image edit. Do NOT zero/fill the masked
    # pixels (a fill creates a sharp step the correlator scores as rim spots).
    # The NXCORR score is forced to -1 inside the mask below, which (with the
    # >= threshold peak gate) drops every detection there with no artificial edge.
    N = frames.shape[0]

    # EVERYTHING on the GPU: NXCORR → local-max (maximum_filter equivalent via
    # max_pool) → dense subpixel CoM → one batched `nonzero` extraction. The only
    # per-frame host work is slicing the flat peak list back per frame. Same
    # method as the numpy reference; results match on real (sharp) peaks and the
    # local-max ridge handling is actually cleaner than the CPU split-peak case.
    with _GPU_LOCK:
        raw = _nxcorr_torch(frames, kr, dev)               # (N,H,W)
        H, W = raw.shape[1], raw.shape[2]
        if beamstop_mask is not None and np.any(beamstop_mask):
            raw[:, torch.as_tensor(beamstop_mask, device=dev)] = -1.0

        # local maxima ≥ threshold, separated by min_distance (max-pool window).
        # A tiny monotone ramp (≤1e-5 globally → ~1e-6 within a window) breaks
        # EXACT ties on a saturated plateau so it yields one peak, not the ridge;
        # it's far below real peak-to-neighbour differences so sharp peaks are
        # unaffected.
        ramp = (torch.arange(H * W, device=dev, dtype=torch.float32)
                .reshape(1, H, W) * (1e-5 / float(H * W)))
        rawt = raw + ramp
        pooled = F.max_pool2d(rawt[:, None], kernel_size=2 * md + 1, stride=1,
                              padding=md).squeeze(1)
        mask = (rawt >= pooled) & (raw >= thr)

        idx = mask.nonzero(as_tuple=False)                 # (M,3) [n, y, x]
        if idx.numel() == 0:
            return [np.zeros((0, 3), np.float32) for _ in range(N)]
        nn, yy, xx = idx[:, 0], idx[:, 1], idx[:, 2]
        vals = raw[nn, yy, xx]
        if subpixel:
            dy, dx = _fit_peaks_quadratic(raw, nn, yy, xx, dev)
            py = yy.to(torch.float32) + dy
            px = xx.to(torch.float32) + dx
        else:
            py, px = yy.to(torch.float32), xx.to(torch.float32)
        res = torch.stack([py, px, vals], dim=1).to("cpu").numpy()
        frame_ids = nn.to("cpu").numpy()

    out = [np.zeros((0, 3), np.float32) for _ in range(N)]
    order = np.argsort(frame_ids, kind="stable")
    res, frame_ids = res[order], frame_ids[order]
    bounds = np.searchsorted(frame_ids, np.arange(N + 1))
    for i in range(N):
        if bounds[i + 1] > bounds[i]:
            out[i] = res[bounds[i]:bounds[i + 1]].astype(np.float32)
    return out


_DOG_KERN: dict = {}    # (device, round(sigma,3)) -> 1D Gaussian tensor


def _gauss1d_torch(sigma: float, device):
    """Cached normalised 1D Gaussian kernel (truncated 3σ) on ``device``."""
    import torch
    key = (str(device), round(float(sigma), 3))
    k = _DOG_KERN.get(key)
    if k is None:
        r = max(1, int(round(3.0 * sigma)))
        x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        k = k / k.sum()
        _DOG_KERN[key] = k
    return k


def _sep_blur_torch(f, sigma, device):
    """Separable real-space Gaussian blur of (N,H,W) via two conv1d passes
    (reflect padding), matching scipy.ndimage.gaussian_filter."""
    import torch
    import torch.nn.functional as F
    k = _gauss1d_torch(sigma, device)
    r = (k.numel() - 1) // 2
    x = f[:, None]                                            # (N,1,H,W)
    # blur along H (rows): pad H, conv with (1,1,K,1)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    # blur along W (cols): pad W, conv with (1,1,1,K)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    return x.squeeze(1)                                       # (N,H,W)


def find_vectors_dog_torch_batch(
    frames: np.ndarray, sigma1: float, sigma2: float, threshold: float,
    min_distance: int, *, subpixel: bool = True,
    beamstop_mask: Optional[np.ndarray] = None, device=None,
):
    """Batched Difference-of-Gaussians blob finder on the GPU.

    Mirrors :func:`find_vectors._find_vectors_single_frame_dog` (band-pass
    ``G(σ₁)−G(σ₂)``, MAD-normalised absolute-SNR threshold, local-max + quadratic
    subpixel) but runs the WHOLE nav block in one batched pass — two separable
    Gaussian blurs, a max-pool local max, and a single ``nonzero`` extraction.
    The beam-stop region is left UNMODIFIED (no fill — a fill creates a step the
    band-pass fires on); peaks inside the mask are dropped and the rim is excluded
    from the SNR scale. Returns a list of N ``(Ni,3)`` ``[ky, kx, snr]`` arrays.
    """
    import torch
    import torch.nn.functional as F

    dev = device or torch_gpu_device()
    if dev is None:
        raise RuntimeError("no torch GPU device available")
    md, thr = int(min_distance), float(threshold)
    frames = np.asarray(frames, np.float32)
    N, H, W = frames.shape

    with _GPU_LOCK:
        f = torch.as_tensor(np.ascontiguousarray(frames), device=dev)
        # Beam stop = PEAK REJECTION, not an image edit. Do NOT fill the masked
        # pixels (a fill introduces a step the band-pass fires on). Run the DoG on
        # the UNMODIFIED frame; `excl_t` only excludes the rim from the robust SNR
        # scale and drops peaks inside the mask below — no artificial edge.
        excl_t = None
        if beamstop_mask is not None and np.any(beamstop_mask):
            excl_t = torch.as_tensor(np.ascontiguousarray(beamstop_mask), device=dev)
        g2 = _sep_blur_torch(f, sigma2, dev)
        g1 = _sep_blur_torch(f, sigma1, dev)
        resp = g1 - g2                                # (N,H,W)

        # robust per-frame absolute SNR via median + MAD, over UNMASKED pixels
        # only (the bright beam-stop rim would otherwise inflate the MAD).
        if excl_t is not None:
            big = resp.max().detach() + 1.0
            stat = resp.masked_fill(excl_t[None], float("nan"))
            flat = stat.reshape(N, -1)
            med = flat.nanmedian(dim=1).values.view(N, 1, 1)
            mad = (stat - med).abs().reshape(N, -1).nanmedian(dim=1).values.view(N, 1, 1)
        else:
            flat = resp.reshape(N, -1)
            med = flat.median(dim=1).values.view(N, 1, 1)
            mad = (resp - med).abs().reshape(N, -1).median(dim=1).values.view(N, 1, 1)
        scale = 1.4826 * mad
        # std fallback where MAD ~ 0 (near-flat / low-texture frame)
        std = resp.reshape(N, -1).std(dim=1, unbiased=False).view(N, 1, 1)
        scale = torch.where(scale > 1e-6, scale, std).clamp_min(1e-6)
        snr = (resp - med) / scale
        if excl_t is not None:
            snr = snr.masked_fill(excl_t[None], 0.0)

        # local maxima ≥ threshold separated by min_distance (tie-break ramp,
        # same trick as the NXCORR path so a flat plateau yields one peak).
        ramp = (torch.arange(H * W, device=dev, dtype=torch.float32)
                .reshape(1, H, W) * (1e-4 / float(H * W)))
        snr_t = snr + ramp
        pooled = F.max_pool2d(snr_t[:, None], kernel_size=2 * md + 1, stride=1,
                              padding=md).squeeze(1)
        mask = (snr_t >= pooled) & (snr >= thr)
        if excl_t is not None:
            mask = mask & ~excl_t[None]

        idx = mask.nonzero(as_tuple=False)
        if idx.numel() == 0:
            return [np.zeros((0, 3), np.float32) for _ in range(N)]
        nn, yy, xx = idx[:, 0], idx[:, 1], idx[:, 2]
        vals = snr[nn, yy, xx]
        if subpixel:
            # hw=1 (3x3 fit) for DoG: spots are small (~3 px FWHM), so a 5x5
            # window reaches into the curved flanks and biases the vertex
            # (~0.08 px error vs ~0.02 px at hw=1, matching the CPU parabola).
            dy, dx = _fit_peaks_quadratic(snr, nn, yy, xx, dev, hw=1)
            py = yy.to(torch.float32) + dy
            px = xx.to(torch.float32) + dx
        else:
            py, px = yy.to(torch.float32), xx.to(torch.float32)
        res = torch.stack([py, px, vals], dim=1).to("cpu").numpy()
        frame_ids = nn.to("cpu").numpy()

    out = [np.zeros((0, 3), np.float32) for _ in range(N)]
    order = np.argsort(frame_ids, kind="stable")
    res, frame_ids = res[order], frame_ids[order]
    bounds = np.searchsorted(frame_ids, np.arange(N + 1))
    for i in range(N):
        if bounds[i + 1] > bounds[i]:
            out[i] = res[bounds[i]:bounds[i + 1]].astype(np.float32)
    return out

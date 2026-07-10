"""Input normalization + automatic, parameter-free disk-size scale normalization.

Vendored verbatim from the ``yoloDiffraction`` research project
(``normalize_input`` from ``yolodiffraction/data/dataset.py`` and the scale-norm
helpers from ``yolodiffraction/data/scale_norm.py``) so SpyDE preprocesses model
inputs EXACTLY as the checkpoints were trained — DO NOT change the maths here
(train/infer parity).

Within a dataset all diffraction disks are the same physical size (the central
beam only looks bigger because it's brighter). So one model trained at a CANONICAL
disk size suffices if we resample each dataset so its disks hit that size. The
disk size is estimated WITHOUT parameters from the pattern's autocorrelation: a
disk autocorrelated with itself gives a central peak whose width ≈ the disk
diameter.

Usage at inference:
    diam = estimate_disk_diameter(frame)
    scaled, factor = scale_to_canonical(frame, diam)        # disks -> ~CANONICAL px
    # run model on `scaled`, then map predicted positions back with /factor
"""
from __future__ import annotations

import numpy as np

# Canonical disk size the model is trained at. Set at the LARGER end of real disk
# sizes (ZrNb ~18-22px) because DOWNSAMPLING a large disk to a small canonical size
# destroys subpixel information (~2x worse Friedel residual on ZrNb). Policy is
# upsample-only: small disks are upsampled to ~canonical; large disks are left at
# native size (never shrunk).
CANONICAL_DIAMETER = 20.0
DOWNSAMPLE = False            # never downsample (it throws away subpixel info)


def normalize_input(frame: np.ndarray, local: bool = True,
                    bg_sigma: float = 12.0) -> np.ndarray:
    """log1p + robust standardization. Shared train+infer.

    log1p compresses the huge spot/background dynamic range; median/MAD is robust to
    the bright central beam and hot pixels.

    ``local`` (default) additionally subtracts a LOCAL background estimate (large
    Gaussian blur) before standardizing — the learned analogue of WNCC's window
    normalization. This makes a spot a local bump above ITS surroundings regardless
    of a spatially-varying diffuse background across the FOV (the hard case). bg_sigma
    should be a few times the disk size so it removes background, not the spots.
    """
    x = np.log1p(np.clip(frame, 0, None).astype(np.float32))
    if local:
        from scipy.ndimage import gaussian_filter
        x = x - gaussian_filter(x, bg_sigma)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)


def estimate_disk_diameter(frame: np.ndarray, hp_sigma: float = 20.0) -> float:
    """Estimate disk diameter (px) from the autocorrelation central-peak FWHM.

    High-passes the frame first (removes smooth background and tames the bright
    central beam), then measures the half-max width of the autocorrelation peak,
    which equals the disk diameter. Robust across sizes; no thresholds/params.
    """
    from scipy.ndimage import gaussian_filter

    f = np.clip(frame.astype(np.float64), 0, None)
    f = np.clip(f - gaussian_filter(f, hp_sigma), 0, None)
    if f.max() <= 0:
        return CANONICAL_DIAMETER
    F = np.fft.fft2(f)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    H, W = ac.shape
    cy, cx = H // 2, W // 2
    # average the horizontal & vertical central-line profiles (robust to anisotropy)
    prof = 0.5 * (ac[cy] / ac[cy, cx] + ac[:, cx] / ac[cy, cx])
    l = r = cx
    while l > 0 and prof[l] > 0.5:
        l -= 1
    while r < W - 1 and prof[r] > 0.5:
        r += 1
    return float(max(r - l, 1))


def scale_factor(frame: np.ndarray, target: float = CANONICAL_DIAMETER) -> float:
    """Upsample-only factor so this frame's disks become >= ~``target`` px.

    factor >= 1 (never shrinks): if disks are already >= target, factor = 1 (leave
    big disks at native size; downsampling them destroys subpixel info).
    """
    f = target / estimate_disk_diameter(frame)
    return float(max(f, 1.0)) if not DOWNSAMPLE else float(f)


def scale_to_canonical(frame: np.ndarray, diameter: float | None = None,
                       target: float = CANONICAL_DIAMETER):
    """Resample ``frame`` so its disks are >= ~``target`` px. Returns (scaled, factor).

    Upsample-only (never downsample — shrinking large disks ~doubles subpixel
    error). A predicted position p in the scaled frame maps back as p / factor.
    ``diameter`` may be passed to reuse one estimate across a stack.
    """
    from scipy.ndimage import zoom

    if diameter is None:
        diameter = estimate_disk_diameter(frame)
    factor = target / diameter
    if not DOWNSAMPLE:
        factor = max(factor, 1.0)            # never shrink
    if abs(factor - 1.0) < 0.05:
        return frame.astype(np.float32), 1.0
    scaled = zoom(frame.astype(np.float32), factor, order=1)
    return scaled, factor

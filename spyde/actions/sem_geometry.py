"""
sem_geometry.py — flat-detector / curved-Ewald-sphere corrections for low-kV
(SEM / t-SEM) 4D-STEM orientation mapping.

At TEM voltages the Ewald sphere is nearly flat over the detector and a flat
camera records an essentially orthographic, linear map pixel -> Angstrom^-1.
At SEM voltages (5-30 kV) two things break that:

  1. **Curved Ewald sphere.** lambda is large, so the sphere radius 1/lambda is
     small and a reflection at scattering angle 2theta has reciprocal magnitude
     g = 2 sin(theta) / lambda, NOT g = 2theta / lambda.

  2. **Flat detector (gnomonic) projection.** A flat camera at camera length L
     records a reflection at detector radius R = L tan(2theta). The naive
     calibration g = R * scale assumes g ∝ R, which over-estimates g at high
     angle by ~ (R/L)^2 / 8 = (tan 2theta)^2 / 8.

Composing the two gives the exact pixel-radius -> Angstrom^-1 map. This module
applies it as a careful per-vector RADIAL remap (direction preserved, magnitude
corrected), done once after peak finding so the downstream affine fit — which
assumes a linear reciprocal space — is valid again.

No Qt; pure numpy; unit-tested. See VECTOR_ORIENTATION_MAPPING_PLAN.md §9 (SEM).
"""
from __future__ import annotations

import numpy as np

# electron rest energy (keV) and hc constants for the relativistic wavelength
_M0C2_keV = 510.998950
_HC_keV_A = 12.3984198  # h*c in keV*Angstrom


def electron_wavelength(accelerating_voltage_kV: float) -> float:
    """Relativistic electron wavelength (Angstrom) for an accelerating voltage.

    lambda = hc / sqrt(eV (2 m0c^2 + eV)), eV and m0c^2 in keV. Matches the
    standard relativistic formula used by diffsims so library and data agree.
    """
    V = float(accelerating_voltage_kV)
    return _HC_keV_A / np.sqrt(V * (2.0 * _M0C2_keV + V))


def detector_radius_to_g(R, camera_length, wavelength):
    """Exact flat-detector radius -> reciprocal magnitude g (Angstrom^-1).

    R, camera_length in the SAME length units (e.g. mm or m or detector-frame
    units). wavelength in Angstrom.

        2theta = atan(R / L)
        g = 2 sin(theta) / lambda = (2/lambda) sin( atan(R/L) / 2 )

    Reduces to the small-angle linear map g ≈ R / (lambda L) as R/L -> 0.
    """
    R = np.asarray(R, dtype=float)
    two_theta = np.arctan2(R, float(camera_length))
    return (2.0 / float(wavelength)) * np.sin(0.5 * two_theta)


def correct_vectors_flat_detector(kxy, camera_length, wavelength,
                                  center=(0.0, 0.0), pixel_to_length=1.0):
    """Remap measured detector vectors to true reciprocal-space (Angstrom^-1).

    Each vector keeps its azimuth; only its magnitude is corrected for the
    flat-detector gnomonic projection + curved Ewald sphere.

    Parameters
    ----------
    kxy : (N, 2) detector positions. If `pixel_to_length` is 1.0 these are
        already in the detector's length units relative to `center`; otherwise
        they are pixels and are converted via `pixel_to_length`.
    camera_length : L in the same length units as `pixel_to_length * pixel`.
    wavelength : Angstrom (use electron_wavelength()).
    center : (cx, cy) detector origin in the same coords as kxy.
    pixel_to_length : detector length per pixel (so R = |kxy-center| * this).

    Returns
    -------
    (N, 2) corrected vectors in Angstrom^-1, same azimuth, centered at origin.
    """
    kxy = np.asarray(kxy, dtype=float)
    if kxy.size == 0:
        return kxy.reshape(-1, 2).copy()
    rel = kxy - np.asarray(center, dtype=float)[None, :]
    R_len = np.linalg.norm(rel, axis=1) * float(pixel_to_length)
    g = detector_radius_to_g(R_len, camera_length, wavelength)
    # preserve direction; scale magnitude from R_len -> g
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = rel / np.linalg.norm(rel, axis=1, keepdims=True)
    unit[~np.isfinite(unit)] = 0.0
    return unit * g[:, None]


def friedel_center(kxy, tol_frac=0.05, max_pairs=64):
    """Estimate the pattern center from Friedel (g <-> -g) pairs — the only way
    to find the origin when there is no direct beam (the SEM case).

    For each vector, its centrosymmetric partner is the one whose midpoint with
    it is most consistent with a shared center. Greedy: pair the closest
    opposite vectors; the mean of pair midpoints is the center estimate.

    tol_frac : a pair qualifies if ||v_i + v_j - 2c0|| < tol_frac * scale, where
        c0 is the running center guess and scale is the median |v|. Robust to
        unpaired spots (they are simply not used).

    Returns (cx, cy) or None if no usable pairs.
    """
    kxy = np.asarray(kxy, dtype=float)
    n = len(kxy)
    if n < 2:
        return None
    c0 = kxy.mean(0)                       # crude first guess
    scale = np.median(np.linalg.norm(kxy - c0, axis=1)) or 1.0

    def _pairs_about(c, tol):
        """Greedy Friedel pairs whose midpoint ~ c; return list of midpoints."""
        s = (kxy[:, None, :] + kxy[None, :, :]) - 2.0 * c[None, None, :]
        d2 = (s ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        mids, used = [], np.zeros(n, bool)
        for flat in np.argsort(d2.ravel()):
            i, j = divmod(int(flat), n)
            if used[i] or used[j] or i == j:
                continue
            if np.sqrt(d2[i, j]) > 2.0 * tol:
                break
            used[i] = used[j] = True
            mids.append(0.5 * (kxy[i] + kxy[j]))
            if len(mids) >= max_pairs:
                break
        return mids

    # Iterate: a few junk/unpaired spots pull the mean-of-all first guess off,
    # so re-estimate the center from the pairs found and re-pair. Converges in
    # 2-3 rounds; tolerance tightens as the estimate improves.
    # First round is permissive (junk/unpaired spots pull the mean-of-all guess
    # off by up to ~scale, so the true pairs' midpoints sit far from c0); later
    # rounds tighten as the center estimate converges.
    c = c0
    mids = []
    tols = [scale, 0.5 * scale, tol_frac * scale, tol_frac * scale]
    for tol in tols:
        m = _pairs_about(c, tol)
        if not m:
            continue
        # robust center: median of midpoints rejects an occasional bad pair
        c_new = np.median(np.asarray(m), axis=0)
        mids = m
        if np.linalg.norm(c_new - c) < 1e-4 * scale:
            c = c_new
            break
        c = c_new
    return c if mids else None


def prepare_sem_vectors(kxy, accelerating_voltage_kV, camera_length,
                        pixel_to_length=1.0, center=None, friedel=True):
    """Full SEM vector preparation: find the center (Friedel if not given),
    then apply the exact flat-detector / Ewald radial correction.

    Returns (corrected_kxy_in_invAngstrom, center_used).
    """
    kxy = np.asarray(kxy, dtype=float)
    lam = electron_wavelength(accelerating_voltage_kV)
    if center is None:
        center = friedel_center(kxy) if friedel else kxy.mean(0)
        if center is None:
            center = kxy.mean(0)
    corrected = correct_vectors_flat_detector(
        kxy, camera_length, lam, center=center, pixel_to_length=pixel_to_length)
    return corrected, np.asarray(center, dtype=float)

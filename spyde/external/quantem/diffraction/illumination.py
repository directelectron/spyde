"""Illumination-averaged excitation envelopes for kinematical patterns.

Precession rotates the incident beam on a cone and a convergent probe fills
a disk of directions; the recorded intensity of a reflection is the
average of its rocking curve over that support. For a ring centered on the
optic axis the excitation error of reflection g is exactly
s_g(phi) = c_g + a_g cos(phi - delta_g), and for a small disk it is affine
in the incident direction to leading order, so the average of a Gaussian
excitation envelope over ring and disk reduces to one scalar transform,

    G(c, a, b; sigma) = sqrt(2/pi) int_0^inf exp(-x^2/2) cos(c x / sigma)
                        J0(a x / sigma) jinc(b x / sigma) dx,

with jinc(x) = 2 J1(x) / x; G(c, 0, 0) = exp(-c^2 / 2 sigma^2) recovers the
static envelope. The functions here evaluate G for whole reflection lists
at once (vectorized Gauss-Legendre quadrature of the transform, exact to
~1e-9 over the parameter range of electron diffraction), give the ring and
disk coefficients (c, a, b) from the geometry, and are shared by the
pattern simulation, the orientation library and the refinements.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import j0, j1

from .._compat import electron_wavelength_angstrom

_X_NODES, _X_WEIGHTS = np.polynomial.legendre.leggauss(400)
_X_MAX = 9.0
_X = 0.5 * _X_MAX * (_X_NODES + 1)
_W = 0.5 * _X_MAX * _X_WEIGHTS * np.exp(-0.5 * _X**2) * np.sqrt(2 / np.pi)


def _jinc(x: np.ndarray) -> np.ndarray:
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = 2 * j1(x[nz]) / x[nz]
    return out


def gaussian_envelope(c, a, b, sigma: float) -> np.ndarray:
    """Illumination-averaged Gaussian excitation envelope G(c, a, b; sigma).

    Parameters
    ----------
    c, a, b : array-like
        Central excitation error, ring amplitude and disk amplitude of each
        reflection (1/Angstroms), from `excitation_coefficients`.
    sigma : float
        Width of the excitation envelope (1/Angstroms).

    Returns
    -------
    np.ndarray
        The averaged envelope in [0, 1], same shape as c.
    """
    c = np.asarray(c, dtype=float)
    a = np.broadcast_to(np.asarray(a, dtype=float), c.shape)
    b = np.broadcast_to(np.asarray(b, dtype=float), c.shape)
    x = _X / sigma
    integrand = np.cos(c[..., None] * x) * j0(a[..., None] * x) * _jinc(b[..., None] * x)
    return np.clip(integrand @ _W, 0.0, 1.0)


def excitation_coefficients(
    g_lab: torch.Tensor | np.ndarray,
    energy_ev: float,
    precession_deg: float = 0.0,
    semiconv_mrad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central excitation error and illumination amplitudes of each reflection.

    For a precession ring of radius r = k0 sin(theta_p) centered on the optic
    axis, with K = sqrt(k0^2 - r^2),

        c_g = (2 K g_z - |g|^2) / (2 (K - g_z)),   a_g = r |g_xy| / |K - g_z|,

    which is exact: s_g(phi) = c_g + a_g cos(phi - delta_g). The convergence
    disk of radius R = k0 sin(alpha) adds b_g = R |g_xy| / |K - g_z| under
    the same affine model. Without illumination, c_g is the static
    excitation error and a_g = b_g = 0.

    Returns (c, a, b) as float arrays (N,).
    """
    g = np.asarray(
        g_lab.detach().cpu().numpy() if isinstance(g_lab, torch.Tensor) else g_lab, dtype=float
    )
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    r = k0 * np.sin(np.deg2rad(precession_deg))
    R = k0 * np.sin(semiconv_mrad * 1e-3)
    K = np.sqrt(k0**2 - r**2)
    den = K - g[:, 2]
    g2 = (g**2).sum(axis=1)
    gxy = np.hypot(g[:, 0], g[:, 1])
    c = (2 * K * g[:, 2] - g2) / (2 * den)
    a = r * gxy / np.abs(den)
    b = R * gxy / np.abs(den)
    return c, a, b


def averaged_gaussian_intensity_envelope(
    g_lab, energy_ev: float, sigma: float, precession_deg: float = 0.0, semiconv_mrad: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Envelope, central excitation error and support half-width of every
    reflection under the illumination: returns (envelope, c, a, b)."""
    c, a, b = excitation_coefficients(g_lab, energy_ev, precession_deg, semiconv_mrad)
    if precession_deg <= 0 and semiconv_mrad <= 0:
        return np.exp(-0.5 * (c / sigma) ** 2), c, a, b
    return gaussian_envelope(c, a, b, sigma), c, a, b


def ring_disk_quadrature(r: float, R: float, n_phi: int = 128, n_r: int = 8, n_psi: int = 32):
    """Positive angular quadrature of the ring x disk illumination, the
    reference against which the analytic envelope is checked: returns
    in-plane tilts (M, 2) and normalized weights (M,)."""
    phi = 2 * np.pi * np.arange(n_phi) / n_phi if r > 0 else np.zeros(1)
    ring = r * np.column_stack((np.cos(phi), np.sin(phi)))
    if R > 0:
        x, w = np.polynomial.legendre.leggauss(n_r)
        radius = R * np.sqrt((x + 1) / 2)
        psi = 2 * np.pi * np.arange(n_psi) / n_psi
        disk = (radius[:, None, None] * np.column_stack((np.cos(psi), np.sin(psi)))[None]).reshape(
            -1, 2
        )
        wd = np.repeat(w / 2 / n_psi, n_psi)
    else:
        disk, wd = np.zeros((1, 2)), np.ones(1)
    t = (ring[:, None] + disk[None]).reshape(-1, 2)
    weights = np.tile(wd, len(ring)) / len(ring)
    return t, weights


def gaussian_envelope_ring_series(c, a, sigma: float, n_terms: int = 6) -> np.ndarray:
    """Ring-averaged Gaussian envelope, b = 0, by the Bessel series

        G = exp(-c^2/2 sigma^2 - v) [I0(u) I0(v) + 2 sum_n (-1)^n I_2n(u) I_n(v)],
        u = c a / sigma^2,  v = a^2 / 4 sigma^2,

    which converges in a few terms for v < 1 (the precession sweep of the
    excitation error smaller than the envelope width, the electron
    diffraction regime); larger v falls back to the transform. Accepts
    numpy arrays or torch tensors of any shape (a broadcast to c)."""
    from scipy.special import iv

    is_torch = isinstance(c, torch.Tensor)
    c_np = np.asarray(c.detach().cpu().numpy() if is_torch else c, dtype=float)
    a_np = np.broadcast_to(
        np.asarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a, dtype=float),
        c_np.shape,
    )
    u = c_np * a_np / sigma**2
    v = a_np**2 / (4 * sigma**2)
    total = iv(0, u) * iv(0, v)
    for n in range(1, n_terms + 1):
        total = total + 2 * (-1) ** n * iv(2 * n, u) * iv(n, v)
    out = np.exp(-0.5 * (c_np / sigma) ** 2 - v) * total
    big = v > 1.0
    if np.any(big):
        out[big] = gaussian_envelope(c_np[big], a_np[big], 0.0, sigma)
    out = np.clip(out, 0.0, 1.0)
    return torch.as_tensor(out, dtype=torch.float64) if is_torch else out


def excitation_amplitudes(
    g_lab: torch.Tensor, energy_ev: float, precession_deg: float, semiconv_mrad: float
):
    """Ring and disk amplitudes (a, b) as torch tensors for lab-frame g of
    any leading shape (..., 3); zero tensors when no illumination."""
    lam = electron_wavelength_angstrom(energy_ev)
    k0 = 1.0 / lam
    r = k0 * np.sin(np.deg2rad(precession_deg))
    R = k0 * np.sin(semiconv_mrad * 1e-3)
    K = np.sqrt(k0**2 - r**2)
    den = (K - g_lab[..., 2]).abs().clamp_min(1e-12)
    gxy = torch.hypot(g_lab[..., 0], g_lab[..., 1])
    return r * gxy / den, R * gxy / den


_V_NODES, _V_WEIGHTS = np.polynomial.legendre.leggauss(200)
_V = 0.5 * (_V_NODES + 1)
_WV = 0.5 * _V_WEIGHTS * 2 * (1 - _V)


def slab_envelope(c, a, b, thickness_A: float) -> np.ndarray:
    """Illumination-averaged finite-thickness (first Born) rocking curve,

        S(c, a, b; z) = 2 int_0^1 (1 - v) cos(2 pi c z v) J0(2 pi a z v)
                        jinc(2 pi b z v) dv,

    which reduces to sinc(c z)^2 without illumination; the Born intensity
    of reflection g is (pi |U_g| z / k0)^2 S. Vectorized Gauss-Legendre
    quadrature over v, exact to ~1e-10 for the phase ranges of electron
    diffraction (c z below ~20)."""
    c = np.asarray(c, dtype=float)
    a = np.broadcast_to(np.asarray(a, dtype=float), c.shape)
    b = np.broadcast_to(np.asarray(b, dtype=float), c.shape)
    x = 2 * np.pi * thickness_A * _V
    integrand = np.cos(c[..., None] * x) * j0(a[..., None] * x) * _jinc(b[..., None] * x)
    return np.clip(integrand @ _WV, 0.0, 1.0)


def gaussian_envelope_ring_torch(c: torch.Tensor, a: torch.Tensor, sigma: float) -> torch.Tensor:
    """Ring-averaged Gaussian envelope (b = 0) in torch, for the refinement
    loops: the Bessel series of gaussian_envelope_ring_series truncated
    after the I_4 term, with I_n(u) from the I_0 / I_1 recurrences (and
    their small-argument series where the recurrence would cancel). Below
    v = a^2 / 4 sigma^2 = 0.3 the truncation error is under 1e-5; larger
    sweeps fall back to the reference series."""
    c = c.to(torch.float64)
    a = torch.as_tensor(a, dtype=torch.float64)
    u = c * a / sigma**2
    v = a * a / (4 * sigma**2)
    i0u = torch.special.i0(u)
    i1u = torch.special.i1(u)
    small2 = u.abs() < 1e-3
    u_safe = torch.where(small2, torch.ones_like(u), u)
    i2u = torch.where(small2, u * u / 8, i0u - 2 * i1u / u_safe)
    small4 = u.abs() < 5e-2
    i3u = torch.where(small4, u**3 / 48, i1u - 4 * i2u / u_safe)
    i4u = torch.where(small4, u**4 / 384, i2u - 6 * i3u / u_safe)
    i0v = torch.special.i0(v)
    i1v = torch.special.i1(v)
    smallv = v < 1e-3
    v_safe = torch.where(smallv, torch.ones_like(v), v)
    i2v = torch.where(smallv, v * v / 8, i0v - 2 * i1v / v_safe)
    out = torch.exp(-0.5 * (c / sigma) ** 2 - v) * (i0u * i0v - 2 * i2u * i1v + 2 * i4u * i2v)
    big = v > 0.3
    if bool(big.any()):
        out = out.clone()
        ref = gaussian_envelope_ring_series(c[big], torch.broadcast_to(a, c.shape)[big], sigma)
        out[big] = ref.to(out.dtype)
    return out.clamp(0.0, 1.0)

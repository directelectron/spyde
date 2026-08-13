"""Dose weighting for motion correction (spec: notes/specs/2026-06-18-dose-weighting-design.md).

Voltage-scaled cryoSPARC exposure filter. Pure functions; pass xp=cupy for GPU.
Combine identity (used by the motion workers, not here): the dose-weighted sum is the
per-frequency normalized weighted average  DW(k) = sum_i w_i(k) F_i(k) / sum_i w_i(k).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# cryoSPARC's live critical-dose curve (V=300 reference), Linux-verified to machine precision.
CS_A = 1.35172          # coefficient
CS_B = -1.31489         # exponent
A_TABLE = {100: 0.312, 120: 0.328, 200: 0.368, 300: 0.490}   # calibrated a(V), e/A^2
A_REF = A_TABLE[300]    # 0.490


@dataclass
class DoseWeightParams:
    """Bundle passed to the motion workers. None => feature off."""
    total_dose: float        # e/A^2 over the WHOLE original movie
    voltage_kv: float
    apix: float              # A/px of the frames being summed
    n_total_frames: int      # ORIGINAL frame count (before any throw)
    frame_offset: int = 0    # number of leading frames thrown before alignment


def voltage_scale(voltage_kv: float) -> float:
    """s(V) = a(V)/a(300). Linear interp inside [100,300]; clamp + warn outside."""
    vs = sorted(A_TABLE)             # [100, 120, 200, 300]
    if voltage_kv <= vs[0]:
        if voltage_kv < vs[0]:
            warnings.warn(f"voltage {voltage_kv} kV below table; clamping to {vs[0]} kV")
        a = A_TABLE[vs[0]]
    elif voltage_kv >= vs[-1]:
        if voltage_kv > vs[-1]:
            warnings.warn(f"voltage {voltage_kv} kV above table; clamping to {vs[-1]} kV")
        a = A_TABLE[vs[-1]]
    else:
        hi = next(v for v in vs if v >= voltage_kv)
        lo = max(v for v in vs if v <= voltage_kv)
        if hi == lo:
            a = A_TABLE[hi]
        else:
            f = (voltage_kv - lo) / (hi - lo)
            a = A_TABLE[lo] + f * (A_TABLE[hi] - A_TABLE[lo])
    return a / A_REF


def frame_doses(n_frames, total_dose, n_total_frames=None, frame_offset=0):
    """Accumulated dose per KEPT frame. Kept frame local index j (0-based) maps to
    original 1-based position p = frame_offset + j + 1; D = p * total_dose / n_total_frames.
    Thrown frames still deposit dose, so n_total_frames (not n_frames) sets dose_per_frame."""
    if n_total_frames is None:
        n_total_frames = n_frames
    dose_per_frame = total_dose / float(n_total_frames)
    positions = np.arange(frame_offset + 1, frame_offset + n_frames + 1, dtype=np.float64)
    return positions * dose_per_frame


def radial_freq_grid(shape, apix, xp=np):
    """2D radial spatial-frequency map in 1/A. DC at [0,0] (fft layout)."""
    h, w = shape
    ky = xp.fft.fftfreq(h).astype(xp.float64) / apix      # cycles/A
    kx = xp.fft.fftfreq(w).astype(xp.float64) / apix
    return xp.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)


def critical_dose(k, voltage_kv, xp=np):
    """Nc(k,V) = s(V) * CS_A * k**CS_B, with +inf where k == 0 (so weight -> 1)."""
    s = voltage_scale(voltage_kv)
    k = xp.asarray(k)
    nc = xp.full(k.shape, xp.inf, dtype=xp.float64)
    nz = k > 0
    nc[nz] = s * CS_A * xp.power(k[nz], CS_B)
    return nc


def dose_weight_map(shape, apix, frame_dose, voltage_kv, xp=np):
    """W(k) = exp(-frame_dose / Nc(k,V)); float32. W[0,0] == 1 (Nc -> inf at DC)."""
    k = radial_freq_grid(shape, apix, xp=xp)
    nc = critical_dose(k, voltage_kv, xp=xp)
    W = xp.exp(-float(frame_dose) / nc)          # -D/inf -> -0 -> exp -> 1 at DC
    return W.astype(xp.float32)

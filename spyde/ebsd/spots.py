"""spots.py — kinematic diffraction spots on a detector.

A reflection ``g`` lights a spot where the diffracted wave ``k0 + g`` meets
the screen, if ``k0 + g`` lies close enough to the Ewald sphere: its
excitation error ``s = |k0 + g| − |k0|`` within a relrod of about the
inverse sample thickness. Spots form in transmission, so this is the TEM
picture; on an EBSD screen the diffracted beams go forward, nowhere near it.

Reflections come from the same diffsims structure-factor calculation as
:func:`spyde.ebsd.bands.reflectors_from_phase`, but with ``+g`` and ``−g``
kept apart — each is its own spot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

#: Excitation-error scale in Å⁻¹ — a relrod for a sample ~50 nm thick.
RELROD = 0.02


@dataclass
class Reflections:
    g: np.ndarray            # (N, 3) crystal-frame reciprocal vectors, Å⁻¹
    intensity: np.ndarray    # (N,) |F|², normalised to 1


def reflections_from_phase(phase, *, min_dspacing: float = 0.8) -> Reflections:
    """Every reflection of *phase* out to *min_dspacing* (Å), ±g separately."""
    from diffsims.crystallography import ReciprocalLatticeVector

    rlv = ReciprocalLatticeVector.from_min_dspacing(phase, min_dspacing=float(min_dspacing))
    rlv.sanitise_phase()
    rlv.calculate_structure_factor()
    intensity = np.abs(np.asarray(rlv.structure_factor)) ** 2
    g = np.asarray(rlv.data, float).reshape(-1, 3)
    keep = np.isfinite(intensity) & (intensity > 1e-6 * max(float(intensity.max()), 1e-12))
    g, intensity = g[keep], intensity[keep]
    return Reflections(g, intensity / max(float(intensity.max()), 1e-12))


def spot_pixels(reflections: Reflections, rotation: np.ndarray,
                to_sample: np.ndarray, wavelength: float, extent: tuple,
                shape) -> tuple[np.ndarray, np.ndarray]:
    """``(pixels, weights)`` of the spots on a flat screen.

    *rotation* carries crystal directions into the sample frame and
    *to_sample* detector directions into the sample frame. The screen spans
    gnomonic coordinates ``extent = (x0, x1, y0, y1)`` across *shape*
    ``(rows, cols)``; ``y1`` is the top row. *pixels* are ``(col, row)``.
    """
    rows, cols = int(shape[0]), int(shape[1])
    k0 = np.array([0.0, 0.0, 1.0 / wavelength])
    g_detector = (reflections.g @ np.asarray(rotation).T) @ np.asarray(to_sample)
    k = k0 + g_detector
    excitation = np.linalg.norm(k, axis=1) - 1.0 / wavelength
    weights = reflections.intensity * np.exp(-(excitation / RELROD) ** 2)
    forward = k[:, 2] > 0
    gx = np.where(forward, k[:, 0] / np.where(forward, k[:, 2], 1.0), np.nan)
    gy = np.where(forward, k[:, 1] / np.where(forward, k[:, 2], 1.0), np.nan)
    x0, x1, y0, y1 = extent
    col = (gx - x0) / (x1 - x0) * (cols - 1)
    row = (y1 - gy) / (y1 - y0) * (rows - 1)
    on_screen = (forward & (weights > 0.02) & (col >= 0) & (col <= cols - 1)
                 & (row >= 0) & (row <= rows - 1))
    pixels = np.stack([col, row], axis=1)[on_screen]
    # The direct beam is always there, and brightest.
    centre = np.array([[(0 - x0) / (x1 - x0) * (cols - 1),
                        (y1 - 0) / (y1 - y0) * (rows - 1)]])
    return (np.vstack([centre, pixels]),
            np.concatenate([[1.0], weights[on_screen]]))

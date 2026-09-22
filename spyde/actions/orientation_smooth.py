"""
orientation_smooth.py — a within-grain mean of an orientation field.

Peak positions pin the in-plane rotation well and the zone-axis tilt poorly,
so neighbouring positions in one grain come out a few degrees apart in tilt
and the IPF-X and IPF-Y maps speckle where IPF-Z is clean. That is noise in
the weakly determined direction, not mis-indexing, and the neighbour rescue
does not touch it: rescue only re-fits a position that disagrees with every
neighbour by more than its threshold.

This is the EBSD clean-up: each position becomes the reliability-weighted
mean of itself and the neighbours within a grain threshold of it. A
neighbour across a boundary is outside the threshold and never averaged in,
so grains smooth and boundaries stay where they were. The phase map and the
strain are left alone.
"""
from __future__ import annotations

import dataclasses
import logging

import numpy as np

log = logging.getLogger(__name__)

#: Neighbours farther than this, symmetry reduced, are another grain.
GRAIN_THRESHOLD_DEG = 5.0
ITERATIONS = 2

_OFFSETS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                 if (dr, dc) != (0, 0))


def smooth_orientation_field(result, threshold_deg: float = GRAIN_THRESHOLD_DEG,
                             iterations: int = ITERATIONS):
    """A copy of *result* with its orientations smoothed within grains.

    The raw orientations stay on the copy as ``raw_quats``, the way the raw
    strain stays beside its smoothed version; ``params`` records the
    smoothing. Positions nothing matched are neither changed nor averaged in.
    """
    quats = np.asarray(result.quats, np.float64)
    phase_idx = np.asarray(result.phase_idx, int)
    fitted = np.nan_to_num(np.asarray(result.coarse_score, float)) > 0
    weights = np.asarray(getattr(result, "reliability", None)
                         if getattr(result, "reliability", None) is not None
                         else result.coarse_score, float)
    weights = np.where(fitted, np.clip(np.nan_to_num(weights), 1e-6, None), 0.0)

    from spyde.signals.orientation_map import orix_phase_from_dict
    symmetries = [orix_phase_from_dict(meta).point_group.data
                  for meta in (getattr(result, "phases_meta", None) or [])]

    smoothed = quats.copy()
    for _ in range(max(1, int(iterations))):
        smoothed = _mean_pass(smoothed, phase_idx, fitted, weights, symmetries,
                              float(threshold_deg))

    out = dataclasses.replace(result, quats=smoothed.astype(np.float32),
                              params={**dict(result.params or {}),
                                      "smooth_orientations": True,
                                      "grain_threshold_deg": float(threshold_deg)})
    out.raw_quats = np.asarray(result.quats)
    for name in ("reliability", "provenance"):
        if getattr(result, name, None) is not None:
            setattr(out, name, getattr(result, name))
    return out


def _mean_pass(quats, phase_idx, fitted, weights, symmetries, threshold_deg):
    """One pass: every fitted position → the weighted mean of itself and its
    same-phase, same-grain neighbours, each neighbour taken in the symmetric
    variant nearest the position."""
    rows, columns = phase_idx.shape
    accumulated = quats * weights[..., None]
    for row_step, column_step in _OFFSETS:
        centre_rows = slice(max(0, -row_step), rows - max(0, row_step))
        centre_columns = slice(max(0, -column_step), columns - max(0, column_step))
        neighbour_rows = slice(max(0, row_step), rows - max(0, -row_step))
        neighbour_columns = slice(max(0, column_step), columns - max(0, -column_step))

        centre = quats[centre_rows, centre_columns]
        neighbour = quats[neighbour_rows, neighbour_columns]
        same = ((phase_idx[centre_rows, centre_columns]
                 == phase_idx[neighbour_rows, neighbour_columns])
                & fitted[centre_rows, centre_columns]
                & fitted[neighbour_rows, neighbour_columns])
        aligned = np.zeros_like(centre)
        angle = np.full(centre.shape[:2], np.inf)
        for index, symmetry in enumerate(symmetries):
            mask = same & (phase_idx[centre_rows, centre_columns] == index)
            if not mask.any():
                continue
            variant, degrees = nearest_variant(centre[mask], neighbour[mask], symmetry)
            aligned[mask] = variant
            angle[mask] = degrees
        within = same & (angle <= threshold_deg)
        weight = np.where(within, weights[neighbour_rows, neighbour_columns], 0.0)
        accumulated[centre_rows, centre_columns] += aligned * weight[..., None]

    norm = np.linalg.norm(accumulated, axis=-1, keepdims=True)
    mean = np.where(norm > 0, accumulated / np.where(norm > 0, norm, 1.0), quats)
    return np.where(fitted[..., None], mean, quats)


def nearest_variant(centre, neighbour, symmetry):
    """For each ``(N, 4)`` pair, the symmetric equivalent of *neighbour*
    nearest *centre* — sign-aligned so it can be averaged with it — and the
    misorientation angle in degrees.

    The equivalents of an orientation are the symmetry operators composed
    onto it the way orix composes them (``symmetry.outer(orientation)``), so
    the angle here is orix's symmetry-reduced angle.
    """
    from orix.quaternion import Rotation

    equivalents = Rotation(symmetry).outer(Rotation(neighbour)).data  # (S, N, 4)
    dots = np.einsum("snk,nk->sn", equivalents, centre)
    best = np.argmax(np.abs(dots), axis=0)
    chosen = equivalents[best, np.arange(centre.shape[0])]
    signed = dots[best, np.arange(centre.shape[0])]
    chosen = chosen * np.sign(signed)[:, None]
    cosine = np.clip(np.abs(signed), 0.0, 1.0)
    return chosen, np.degrees(2.0 * np.arccos(cosine))

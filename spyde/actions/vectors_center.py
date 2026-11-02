"""
vectors_center.py — centre the zero beam of a vectors tree.

On a vectors tree the direct beam is a vector like any other, the brightest
one near the middle of the pattern, so centring needs no frame: pick that
vector at every scan position from inside a box around the centre, fit one
plane through where it sits across the scan (the beam drifts smoothly with
the scan coils, and a plane is what a per-position estimate noisily samples),
and move every vector by centre minus plane. Positions where no vector fell
in the box take the plane too, which is what a plane is for.

This is what "centre the beam, then find vectors" costs on the frames, done
after the fact on the peaks — the same shift, without touching a frame.
"""
from __future__ import annotations

import copy
import logging

import numpy as np

from spyde.signals.diffraction_vectors import (
    COL_INTENSITY, COL_KX, COL_KY, COL_NAV_X, COL_NAV_Y,
)

log = logging.getLogger(__name__)


def _brightest_per_position(position, intensity, n_positions):
    """For rows tagged with a flat scan *position*, the row index of the
    brightest row at each position (-1 where a position has no rows)."""
    order = np.lexsort((-intensity, position))
    starts = np.searchsorted(position[order], np.arange(n_positions))
    counts = np.bincount(position, minlength=n_positions)
    return np.where(counts > 0, order[np.minimum(starts, len(order) - 1)], -1)


def _rows_in_box(vecs, center_xy, half_width):
    """``(position, inside)`` for every vector: its flat scan position
    (``iy * nx + ix``) and whether it lies in the box."""
    rows = np.asarray(vecs.flat_buffer)
    nx = int(vecs.nav_shape[1])
    position = (rows[:, COL_NAV_Y].astype(np.int64) * nx
                + rows[:, COL_NAV_X].astype(np.int64))
    half = float(half_width)
    inside = ((np.abs(rows[:, COL_KX] - float(center_xy[0])) <= half)
              & (np.abs(rows[:, COL_KY] - float(center_xy[1])) <= half))
    return position, inside


def _beam_rows(vecs, center_xy, half_width):
    """Per scan position, the row index of the brightest vector inside the
    box, or -1 where the box holds none."""
    n_positions = int(np.prod(vecs.nav_shape))
    position, inside = _rows_in_box(vecs, center_xy, half_width)
    candidates = np.flatnonzero(inside)
    best = np.full(n_positions, -1)
    if candidates.size:
        intensity = np.asarray(vecs.flat_buffer)[candidates, COL_INTENSITY]
        first = _brightest_per_position(position[candidates], intensity, n_positions)
        best[first >= 0] = candidates[first[first >= 0]]
    return best


def beam_per_position(vecs, centre_xy, half_width) -> np.ndarray:
    """``(ny, nx, 2)`` — the brightest vector inside the box at each scan
    position, in the vectors' own units; NaN where the box holds none."""
    ny, nx = vecs.nav_shape
    best = _beam_rows(vecs, centre_xy, half_width)
    beam = np.full((ny * nx, 2), np.nan)
    found = best >= 0
    beam[found] = np.asarray(vecs.flat_buffer)[best[found]][:, [COL_KX, COL_KY]]
    return beam.reshape(ny, nx, 2)


def beam_contrast(vecs, center_xy, half_width) -> float:
    """How much brighter the beam candidate is than the brightest vector of
    any other disk in the same pattern: the median over the scan of their
    intensity ratio.

    A Bragg spot has a Friedel partner of nearly equal intensity, so when a
    beam stop hides the beam and a spot is taken for it the ratio sits at ~1
    (measured 1.00–1.11 on si_grains, sped_ag and mgo_nanocrystals with the
    beam masked), while a visible beam stands well clear (6.7–16 on the same
    scans). Vectors within one disk radius of the candidate are the same disk
    found again, not another spot, and are left out. NaN when no position has
    both a candidate and another disk."""
    n_positions = int(np.prod(vecs.nav_shape))
    rows = np.asarray(vecs.flat_buffer)
    if rows.shape[0] == 0:
        return float("nan")
    best = _beam_rows(vecs, center_xy, half_width)
    position, _ = _rows_in_box(vecs, center_xy, half_width)
    intensity = rows[:, COL_INTENSITY]
    beam_row = best[position]
    beam_k = rows[np.maximum(beam_row, 0)][:, [COL_KX, COL_KY]]
    distance = np.hypot(*(rows[:, [COL_KX, COL_KY]] - beam_k).T)
    disk_radius = float(getattr(vecs, "kernel_radius_data", 0.0) or 0.0)
    elsewhere = (beam_row >= 0) & (distance > disk_radius)
    other_max = np.zeros(n_positions)
    np.maximum.at(other_max, position[elsewhere], intensity[elsewhere])
    usable = (best >= 0) & (other_max > 0)
    if not usable.any():
        return float("nan")
    return float(np.median(intensity[best[usable]] / other_max[usable]))


def fit_plane(values: np.ndarray) -> np.ndarray:
    """Least-squares plane ``a + b·ix + c·iy`` through the finite entries of
    an ``(ny, nx)`` field, evaluated everywhere. Fewer than three finite
    points give their mean; none gives zeros."""
    values = np.asarray(values, float)
    ny, nx = values.shape
    iy, ix = np.mgrid[0:ny, 0:nx]
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros_like(values)
    if finite.sum() < 3:
        return np.full_like(values, float(np.nanmean(values)))
    design = np.column_stack([np.ones(finite.sum()), ix[finite], iy[finite]])
    coefficients, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
    return coefficients[0] + coefficients[1] * ix + coefficients[2] * iy


def centring_shifts(vecs, centre_xy, half_width, plane: bool = True,
                    corner_fraction: float | None = None):
    """``(ny, nx, 2)`` — what to add to every vector at each position so the
    beam lands on *centre_xy*; with *plane* the shift is the fitted plane
    everywhere, without it each position keeps its own beam and only the
    positions with no beam take the plane.

    A *corner_fraction* fits the plane through the four scan-corner blocks
    only (:func:`spyde.corners.corner_reference`), for a scan whose middle
    holds a sample that would bias it."""
    beam = beam_per_position(vecs, centre_xy, half_width)
    shift = np.stack([float(centre_xy[0]) - beam[..., 0],
                      float(centre_xy[1]) - beam[..., 1]], axis=-1)
    if corner_fraction is not None:
        from spyde.corners import corner_reference
        fitted = corner_reference(shift, corner_fraction)
    else:
        fitted = np.stack([fit_plane(shift[..., 0]), fit_plane(shift[..., 1])],
                          axis=-1)
    if plane:
        return fitted, beam
    return np.where(np.isfinite(shift), shift, fitted), beam


def shifted_vectors(vecs, shift):
    """A copy of *vecs* with every vector moved by its position's ``shift``
    (``(ny, nx, 2)``, the vectors' units)."""
    buffer = np.array(vecs.flat_buffer, copy=True)
    if buffer.shape[0]:
        iy = buffer[:, COL_NAV_Y].astype(np.int64)
        ix = buffer[:, COL_NAV_X].astype(np.int64)
        buffer[:, COL_KX] += shift[iy, ix, 0]
        buffer[:, COL_KY] += shift[iy, ix, 1]
    out = copy.copy(vecs)
    out.flat_buffer = buffer
    if hasattr(out, "_packed"):
        out._packed = buffer
    for cache in ("_dense_cache", "_kdtree"):
        if hasattr(out, cache):
            setattr(out, cache, None)
    return out


def with_origin_at(vecs, origin_xy, sig_axes):
    """A copy of *vecs* whose coordinate frame puts *origin_xy* at (0, 0),
    with *sig_axes* (the vectors' signal-axis records) re-offset to match.

    Every vector and both axis offsets move by the same amount, so each vector
    stays on the same detector pixel; only the numbers change. Strain and
    orientation read a vector's coordinates as its scattering vector, which is
    only true once the zero beam is the origin."""
    from spyde.signals.ragged_store import _AxisLite

    shift = np.broadcast_to(-np.asarray(origin_xy, float), tuple(vecs.nav_shape) + (2,))
    out = shifted_vectors(vecs, shift)
    out.sig_axes = [
        _AxisLite(scale=float(axis.scale),
                  offset=float(axis.offset) - float(origin),
                  size=int(axis.size),
                  units=str(getattr(axis, "units", "") or ""),
                  name=str(getattr(axis, "name", "") or ""))
        for axis, origin in zip(sig_axes, origin_xy)
    ]
    return out


def center_vectors(vecs, centre_xy, half_width, plane: bool = True,
                   corner_fraction: float | None = None):
    """Centre a vectors object. Returns ``(centred, shift, beam)``."""
    shift, beam = centring_shifts(vecs, centre_xy, half_width, plane=plane,
                                  corner_fraction=corner_fraction)
    return shifted_vectors(vecs, shift), shift, beam


def detector_center(sig_axes):
    """The middle of the detector in the axes' own units."""
    return tuple(float(axis.offset) + int(axis.size) / 2.0 * float(axis.scale)
                 for axis in sig_axes[:2])


def center_on_beam(vecs, sig_axes, half_width_px: int = 0, plane: bool = True,
                   corner_fraction: float | None = None):
    """Center *vecs* on the zero beam and make the beam the origin.

    The beam is searched for in a box of half-width *half_width_px* pixels
    around the middle of the detector (0 searches the whole pattern);
    *corner_fraction* fits the plane through the scan corners alone (see
    ``centring_shifts``).
    Returns ``(centered, shift, found)`` — *found* is how many scan positions
    had a vector in the box; with none, *centered* is None."""
    center = detector_center(sig_axes)
    scale = abs(float(sig_axes[0].scale))
    half = (half_width_px if half_width_px > 0
            else max(int(sig_axes[0].size), int(sig_axes[1].size))) * scale
    centered, shift, beam = center_vectors(vecs, center, half, plane=plane,
                                          corner_fraction=corner_fraction)
    found = int(np.isfinite(beam[..., 0]).sum())
    if found == 0:
        return None, shift, 0
    return with_origin_at(centered, center, sig_axes), shift, found

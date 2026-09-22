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
import dataclasses
import logging

import numpy as np

from spyde.signals.diffraction_vectors import (
    COL_INTENSITY, COL_KX, COL_KY, COL_NAV_X, COL_NAV_Y,
)

log = logging.getLogger(__name__)


def beam_per_position(vecs, centre_xy, half_width) -> np.ndarray:
    """``(ny, nx, 2)`` — the brightest vector inside the box at each scan
    position, in the vectors' own units; NaN where the box holds none."""
    ny, nx = vecs.nav_shape
    beam = np.full((ny, nx, 2), np.nan)
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    half = float(half_width)
    for iy in range(ny):
        for ix in range(nx):
            rows = np.asarray(vecs.at(iy, ix))
            if rows.shape[0] == 0:
                continue
            inside = ((np.abs(rows[:, COL_KX] - cx) <= half)
                      & (np.abs(rows[:, COL_KY] - cy) <= half))
            if not inside.any():
                continue
            candidates = rows[inside]
            best = candidates[np.argmax(candidates[:, COL_INTENSITY])]
            beam[iy, ix] = (best[COL_KX], best[COL_KY])
    return beam


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


def centring_shifts(vecs, centre_xy, half_width, plane: bool = True):
    """``(ny, nx, 2)`` — what to add to every vector at each position so the
    beam lands on *centre_xy*; with *plane* the shift is the fitted plane
    everywhere, without it each position keeps its own beam and only the
    positions with no beam take the plane."""
    beam = beam_per_position(vecs, centre_xy, half_width)
    shift = np.stack([float(centre_xy[0]) - beam[..., 0],
                      float(centre_xy[1]) - beam[..., 1]], axis=-1)
    fitted = np.stack([fit_plane(shift[..., 0]), fit_plane(shift[..., 1])], axis=-1)
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


def center_vectors(vecs, centre_xy, half_width, plane: bool = True):
    """Centre a vectors object. Returns ``(centred, shift, beam)``."""
    shift, beam = centring_shifts(vecs, centre_xy, half_width, plane=plane)
    return shifted_vectors(vecs, shift), shift, beam

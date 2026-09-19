"""
corners.py — the four corner blocks of a scan, and the plane through them.

A scan's corners are the cheap sample. They are a few percent of the positions,
they are (by assumption) off whatever feature the middle of the scan is there to
show, and a plane fitted through them extrapolates across the whole scan. Two
quite different jobs in SpyDE are built on exactly that:

* **DPC centering** (:mod:`spyde.actions.dpc`) takes the beam shift measured in
  the corners as the instrument's descan ramp and subtracts the plane from the
  whole field.
* **Multi-angle reciprocal alignment**
  (:mod:`spyde.backend._session_multiangle_loader`) sums the diffraction
  patterns over each corner, locates the direct beam in the four sums, and takes
  the plane's value at the scan centre as that member's beam position.

The machinery lived in ``dpc.py`` while DPC was its only caller. It is here now
because neither job is more entitled to it than the other, and a module named
after one physics technique is the wrong place for a second one to import from.
``dpc.py`` re-exports every name below, so nothing that already said
``dpc.corner_slices`` has to change.

Pure numpy. No Qt, no IPC, no ``Session``, no pyxem — so a notebook, a script
and the wizard all run the same code.
"""
from __future__ import annotations

import numpy as np

#: Default corner-box size as a fraction of each scan axis (pyxem's own default
#: for ``get_linear_plane(fit_corners=…)``).
DEFAULT_CORNER_FRACTION = 0.05

#: Smallest corner block, per side, regardless of the fraction — see
#: :func:`corner_slices`.
MIN_CORNER_PX = 2

#: The four corners, in the order :func:`corner_slices` returns them and
#: :func:`corner_slice` indexes them: ROW-MAJOR, the reading order of a 2 × 2
#: grid. Named so a dialog can label a per-corner control without inventing its
#: own order and disagreeing with the fit — a mislabelled corner is a picture
#: that is simply wrong and looks entirely plausible.
CORNER_NAMES: tuple[str, ...] = ("top left", "top right",
                                 "bottom left", "bottom right")


def corner_slice(nav_shape: tuple[int, int], corner: int,
                 corner_fraction: float = DEFAULT_CORNER_FRACTION
                 ) -> tuple[slice, slice]:
    """ONE of the four corner blocks, as a ``(row_slice, col_slice)`` pair.

    *corner* indexes :data:`CORNER_NAMES`. This is the whole of
    :func:`corner_slices`' arithmetic, exposed on its own so a caller can give
    each corner its OWN extent — a scan whose top-left sits on vacuum and whose
    bottom-right sits on the sample needs two different boxes, and one fraction
    for all four cannot say that.
    """
    ny, nx = int(nav_shape[0]), int(nav_shape[1])
    fraction = float(np.clip(corner_fraction, 1e-6, 0.5))
    # Floor each side at MIN_CORNER_PX. Two planes of three parameters need 6
    # points; 1x1 boxes give exactly 4, so the fit is under-determined and the
    # blocks are invisible on screen besides. On a 256-wide scan the default 5%
    # is 13 px and the floor never binds — it exists for the small scans where
    # a fraction alone degenerates.
    hy = int(max(MIN_CORNER_PX, min(ny, round(ny * fraction))))
    hx = int(max(MIN_CORNER_PX, min(nx, round(nx * fraction))))
    rows = slice(0, hy) if int(corner) < 2 else slice(ny - hy, ny)
    columns = slice(0, hx) if int(corner) % 2 == 0 else slice(nx - hx, nx)
    return rows, columns


def corner_slices(nav_shape: tuple[int, int],
                  corner_fraction: float = DEFAULT_CORNER_FRACTION
                  ) -> list[tuple[slice, slice]]:
    """The four corner blocks of a scan as ``(row_slice, col_slice)`` pairs.

    *nav_shape* is ``(ny, nx)`` — numpy order, matching ``shifts[..., 0].shape``.
    Each block spans ``corner_fraction`` of ITS OWN axis, so a 512 × 64 scan gets
    boxes that are wide and short rather than square.

    That last point is a deliberate divergence from pyxem's
    ``_get_corner_slices``, which derives both block sizes from
    ``navigation_axes[0]``/``[1]`` and then applies them to the numpy array in
    the opposite order — harmless on a square scan, transposed on any other. We
    build the mask ourselves and hand it to ``get_linear_plane(mask=…)`` instead
    of using ``fit_corners=…``, so the boxes the user SEES on the navigator are
    exactly the pixels the plane is fitted to (``test_dpc.py`` pins that).
    """
    return [corner_slice(nav_shape, corner, corner_fraction)
            for corner in range(len(CORNER_NAMES))]


def corner_boxes(nav_shape: tuple[int, int],
                 corner_fraction: float = DEFAULT_CORNER_FRACTION
                 ) -> list[tuple[float, float, float, float]]:
    """The same four blocks as drawable ``(x, y, w, h)`` rectangles.

    IMAGE PIXELS, the frame anyplotlib's 2-D widgets and markers use (see
    ``drift_action`` — do not add a scale conversion). Same source as
    :func:`corner_slices`, so the overlay cannot disagree with the fit.
    """
    out = []
    for rs, cs in corner_slices(nav_shape, corner_fraction):
        out.append((float(cs.start), float(rs.start),
                    float(cs.stop - cs.start), float(rs.stop - rs.start)))
    return out


def corner_mask(nav_shape: tuple[int, int],
                corner_fraction: float = DEFAULT_CORNER_FRACTION) -> np.ndarray:
    """``get_linear_plane``-style mask: ``True`` where the fit must IGNORE.

    So it is ``True`` everywhere except inside the four corner blocks.
    """
    mask = np.ones((int(nav_shape[0]), int(nav_shape[1])), dtype=bool)
    for rs, cs in corner_slices(nav_shape, corner_fraction):
        mask[rs, cs] = False
    return mask


def _fit_plane(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Least-squares plane ``a·col + b·row + c`` through the UNMASKED points.

    *mask* is ``True`` = ignore (pyxem's polarity). Falls back to the masked
    mean, then to zeros, when the selected points can't determine a plane
    (fewer than 3, or all collinear) — a flat reference is a defensible
    reference; a singular-matrix traceback in the middle of a wizard is not.
    """
    ny, nx = values.shape
    rows, cols = np.mgrid[0:ny, 0:nx]
    keep = np.isfinite(values)
    if mask is not None:
        keep &= ~mask
    n = int(keep.sum())
    if n == 0:
        return np.zeros_like(values, dtype=np.float64)
    A = np.stack([cols[keep].ravel(), rows[keep].ravel(),
                  np.ones(n)], axis=1).astype(np.float64)
    b = values[keep].ravel().astype(np.float64)
    if n < 3:
        return np.full_like(values, float(b.mean()), dtype=np.float64)
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:                            # pragma: no cover
        return np.full_like(values, float(b.mean()), dtype=np.float64)
    if not np.all(np.isfinite(coef)):                        # pragma: no cover
        return np.full_like(values, float(b.mean()), dtype=np.float64)
    return (coef[0] * cols + coef[1] * rows + coef[2]).astype(np.float64)


def plane_through(shifts: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Fit an independent plane to each of the x- and y-shift components."""
    shifts = np.asarray(shifts, dtype=np.float64)
    return np.stack([_fit_plane(shifts[..., 0], mask),
                     _fit_plane(shifts[..., 1], mask)], axis=-1)


def corner_reference(shifts: np.ndarray,
                     corner_fraction: float = DEFAULT_CORNER_FRACTION
                     ) -> np.ndarray:
    """Descan reference from the four scan corners.

    The corners are assumed to be off the feature of interest, so whatever beam
    shift they carry is instrument descan. A plane through them extrapolates
    that ramp across the whole scan — pyxem's ``get_linear_plane(fit_corners=…)``
    idea, with the boxes made visible and made correct on non-square scans (see
    :func:`corner_slices`).
    """
    nav_shape = np.asarray(shifts).shape[:2]
    return plane_through(shifts, corner_mask(nav_shape, corner_fraction))

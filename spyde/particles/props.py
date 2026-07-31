"""
props.py — ``regionprops_table``'s columns as label-wise reductions over the raster.

``skimage.measure.regionprops_table`` walks a Python object per region. That is
fine at the scale its docs assume and it is the run at ours: a real 4096² in-situ
growth frame has **26 566** particles, and measuring them costs **53.5 s** against
3.0 s to segment them (``benchmarks.md`` § "The batch run at real scale"). The
cost is per REGION, so it grows with what the microscope actually produced.

Almost every property in that table is a **label-wise reduction over the pixel
raster** — O(pixels), no Python loop over regions, exactly the shape
``np.bincount`` exists for. This module computes those, and only those, in one
vectorised pass:

* ``area``, ``centroid``, ``bbox``, ``equivalent_diameter_area`` — counts, sums
  and per-label extents.
* ``major/minor_axis_length``, ``eccentricity`` — the eigenvalues of the inertia
  tensor, which is built from the second central moments ``mu20/mu02/mu11``, each
  of which is a ``bincount`` of ``dr*dr`` / ``dc*dc`` / ``dr*dc``.
* ``perimeter`` — skimage's border-crossing weighting, done on the whole frame at
  once (see :func:`label_perimeter`).

**Value parity is the whole point, and it is not approximate.** Every formula
here is skimage's own, reproduced from its source rather than from its docs, and
in skimage's own coordinate frame:

* ``centroid`` is the mean of GLOBAL integer coordinates. Both paths sum exact
  integers in float64 (the largest possible sum here is ~1e8, far inside
  float64's exact-integer range), so the two agree **bit for bit**.
* ``area``/``bbox`` are integers, so they agree **exactly**.
* The central moments are taken about ``centroid_local`` in the region's own
  bbox frame, the two-pass way ``_moments.moments_central`` does it when handed
  an explicit centre — NOT the raw-to-central expansion, which cancels. skimage
  reduces along one axis then the other (``einsum``) and this reduces in raster
  order, so the two differ only in summation order: ~1e-15 relative, and the
  eigen-decomposition that follows is the same LAPACK call on the same 2×2.
* ``perimeter`` sums the same per-region 50-bin histogram against the same
  weight vector, so it also matches to summation order.

``solidity`` is the exception and gets its own module: it is a convex hull per
region, not a raster reduction, and it is 30.5 s of that 53.5 s on its own.
:mod:`spyde.particles.hull` reproduces skimage's hull EXACTLY in integer
arithmetic (verified bit-for-bit on all 26 566 regions of the real frame) and
:func:`solidity_table` remains as the fallback when numba is unavailable.

What this does NOT do
---------------------
No GPU. The reductions land at ~1.0 s for a 4096² frame with 26 566 regions,
which is already below the per-frame budget, and a torch path would add a device
lock, a transfer and a fallback to a stage that is no longer the bottleneck.

Nothing here materialises more than the frame it was handed (CLAUDE.md § Memory
Safety): the working set is the label raster, one padded copy of it, and a handful
of arrays sized by the FOREGROUND pixel count.
"""
from __future__ import annotations

from math import sqrt

import numpy as np

#: skimage's border-crossing weights (``_regionprops.perimeter``). Index is the
#: convolution code ``1 + 2*(orthogonal border neighbours) + 10*(diagonal border
#: neighbours)``; only odd codes (i.e. those with a border pixel at the centre)
#: carry weight, which is why this module can evaluate the code at border pixels
#: only and still match a full-frame convolution.
_PERIM_WEIGHTS = np.zeros(50, dtype=np.float64)
_PERIM_WEIGHTS[[5, 7, 15, 17, 25, 27]] = 1.0
_PERIM_WEIGHTS[[21, 33]] = sqrt(2)
_PERIM_WEIGHTS[[13, 23]] = (1.0 + sqrt(2)) / 2.0

#: The keys :func:`label_props` produces, matching ``regionprops_table``'s names.
PROP_KEYS: tuple[str, ...] = (
    "label",
    "centroid-0", "centroid-1",
    "area",
    "equivalent_diameter_area",
    "major_axis_length",
    "minor_axis_length",
    "perimeter",
    "eccentricity",
    "bbox-0", "bbox-1", "bbox-2", "bbox-3",
)


def _empty_table(with_perimeter: bool = True) -> dict[str, np.ndarray]:
    out = {k: np.zeros((0,), np.float64) for k in PROP_KEYS}
    out["label"] = np.zeros((0,), np.int64)
    for k in ("bbox-0", "bbox-1", "bbox-2", "bbox-3"):
        out[k] = np.zeros((0,), np.int64)
    out["area"] = np.zeros((0,), np.float64)
    if not with_perimeter:
        out.pop("perimeter")
    return out


def label_bboxes(lab: np.ndarray, n_max: int) -> np.ndarray:
    """``(n_max, 4)`` int64 ``(y0, x0, y1, x1)``, row *i* for label ``i+1``.

    ``scipy.ndimage.find_objects`` is the same C pass ``regionprops`` itself uses
    to decide which labels exist, and it is 0.065 s at 4096² against 1.1 s for the
    ``bbox`` column of ``regionprops_table``. Absent labels get an all-zero row and
    are filtered out by the caller's ``counts > 0`` mask, which selects exactly the
    labels for which ``find_objects`` returned a slice.
    """
    from scipy import ndimage as ndi

    objs = ndi.find_objects(lab, max_label=int(n_max))
    bb = np.zeros((int(n_max), 4), np.int64)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        ys, xs = sl
        bb[i, 0] = ys.start
        bb[i, 1] = xs.start
        bb[i, 2] = ys.stop
        bb[i, 3] = xs.stop
    return bb


def label_perimeter(lab: np.ndarray, dense: np.ndarray, n_out: int) -> np.ndarray:
    """Per-label ``skimage.measure.perimeter(region_mask, 4)``, whole frame at once.

    *dense* maps a label value to its row in the output (``-1`` for absent), and
    *n_out* is how many rows that is. The indirection is not cosmetic: the
    per-label histogram has **50 bins per label**, so keying it by the raw label
    value would allocate 50x the label range. A label image whose values are
    sparse (anything filtered or re-tagged upstream) would then ask for gigabytes
    to describe a few thousand regions.

    skimage runs, for each region, a 4-connected erosion of that region's ISOLATED
    mask inside its bbox, subtracts it to get the border ring, convolves the ring
    with ``[[10,2,10],[2,1,2],[10,2,10]]`` and sums a weight per code. Two
    observations turn that into one pass over the frame:

    * A pixel survives the per-region erosion iff it and all four of its
      orthogonal neighbours carry the SAME label. The bbox is tight, so any
      neighbour outside the crop is outside the region as well, which is exactly
      what ``border_value=0`` gives — so testing "same label" on the padded FULL
      frame reproduces the per-region erosion pixel for pixel.
    * Only odd codes carry a nonzero weight, and a code is odd only when the
      centre pixel is itself a border pixel. Non-border pixels therefore
      contribute exactly 0 and need not be evaluated.

    The per-label sum is taken as a ``(n_max+1, 50)`` histogram dotted with the
    weight vector — the same reduction skimage performs, so the result agrees to
    summation order rather than to an approximation.
    """
    h, w = lab.shape
    pad = np.zeros((h + 2, w + 2), dtype=lab.dtype)
    pad[1:-1, 1:-1] = lab
    ctr = pad[1:-1, 1:-1]

    def shifted(dy: int, dx: int) -> np.ndarray:
        return pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    fg = ctr != 0
    interior = fg.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= shifted(dy, dx) == ctr
    border = fg
    border &= ~interior                        # in place; `fg` is a fresh array
    del interior

    bpad = np.zeros((h + 2, w + 2), dtype=bool)
    bpad[1:-1, 1:-1] = border

    def bshift(dy: int, dx: int) -> np.ndarray:
        return bpad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    code = border.astype(np.uint8)             # the centre pixel's own +1
    same_border = np.empty((h, w), dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        np.equal(shifted(dy, dx), ctr, out=same_border)
        same_border &= bshift(dy, dx)
        code += same_border.view(np.uint8) * np.uint8(2)
    for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        np.equal(shifted(dy, dx), ctr, out=same_border)
        same_border &= bshift(dy, dx)
        code += same_border.view(np.uint8) * np.uint8(10)

    idx = np.flatnonzero(border.reshape(-1))
    if idx.size == 0:
        return np.zeros((int(n_out),), np.float64)
    keys = dense[lab.reshape(-1)[idx]] * 50
    keys += code.reshape(-1)[idx]
    hist = np.bincount(keys, minlength=int(n_out) * 50)
    hist = hist[:int(n_out) * 50].reshape(int(n_out), 50)
    return hist.astype(np.float64) @ _PERIM_WEIGHTS


def label_props(lab: np.ndarray, *, with_perimeter: bool = True,
                with_solidity: bool = True) -> dict[str, np.ndarray]:
    """``regionprops_table``'s columns, vectorised, in the same order and dtypes.

    Drops into ``measure_frame`` where the table did. ``solidity`` comes from
    :mod:`spyde.particles.hull` when numba is available and from
    :func:`solidity_table` (skimage) when it is not — the two agree exactly.
    """
    lab = np.asarray(lab)
    if lab.ndim != 2:
        raise ValueError(f"labels must be 2-D; got shape {lab.shape}")
    h, w = lab.shape
    flat = lab.reshape(-1)
    if flat.size == 0 or int(lab.max()) <= 0:
        out = _empty_table(with_perimeter)
        if with_solidity:
            out["solidity"] = np.zeros((0,), np.float64)
        return out

    counts = np.bincount(flat)
    n_max = counts.size - 1
    keep = counts[1:] > 0
    labels = (np.flatnonzero(keep) + 1).astype(np.int64)

    bb = label_bboxes(lab, n_max)

    # Foreground pixels only. The reductions below are O(foreground), not
    # O(frame), and the raster is scanned once to find them.
    nz = np.flatnonzero(flat)
    lab_of = flat[nz].astype(np.intp)
    row = nz // w
    col = nz - row * w

    cnt_f = counts.astype(np.float64)

    # LOCAL (bbox-relative) coordinates, which is the frame skimage takes its
    # central moments in. Integers, so these sums are exact.
    y0_by = np.zeros(n_max + 1, np.int64)
    x0_by = np.zeros(n_max + 1, np.int64)
    y0_by[1:] = bb[:, 0]
    x0_by[1:] = bb[:, 1]
    dr = (row - y0_by[lab_of]).astype(np.float64)
    dc = (col - x0_by[lab_of]).astype(np.float64)
    del row, col, nz

    sr = np.bincount(lab_of, weights=dr, minlength=n_max + 1)
    sc = np.bincount(lab_of, weights=dc, minlength=n_max + 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cr_local = sr / cnt_f
        cc_local = sc / cnt_f
    np.nan_to_num(cr_local, copy=False)
    np.nan_to_num(cc_local, copy=False)

    # Two-pass central moments about `centroid_local`, matching
    # `_moments.moments_central(image, centroid_local, ...)`.
    dr -= cr_local[lab_of]
    dc -= cc_local[lab_of]
    mu20 = np.bincount(lab_of, weights=dr * dr, minlength=n_max + 1)
    mu02 = np.bincount(lab_of, weights=dc * dc, minlength=n_max + 1)
    dr *= dc
    mu11 = np.bincount(lab_of, weights=dr, minlength=n_max + 1)
    del dr, dc, lab_of

    sel = labels                                  # index into the by-label arrays
    area = cnt_f[sel]
    # GLOBAL centroid = mean of global integer coordinates, exactly as
    # `RegionProperties.centroid` computes it (`coords_scaled.mean(axis=0)`).
    cy = (sr[sel] + area * bb[sel - 1, 0]) / area
    cx = (sc[sel] + area * bb[sel - 1, 1]) / area

    # inertia_tensor: [[mu02, -mu11], [-mu11, mu20]] / mu00  (skimage's
    # convention — I_ii is the second moment of every axis EXCEPT i).
    n = sel.size
    tensor = np.empty((n, 2, 2), np.float64)
    tensor[:, 0, 0] = mu02[sel] / area
    tensor[:, 1, 1] = mu20[sel] / area
    off = -mu11[sel] / area
    tensor[:, 0, 1] = off
    tensor[:, 1, 0] = off
    ev = np.linalg.eigvalsh(tensor)               # ascending
    np.clip(ev, 0, None, out=ev)
    l1 = ev[:, 1]                                 # descending order's first
    l2 = ev[:, 0]
    major = 4.0 * np.sqrt(l1)
    minor = 4.0 * np.sqrt(l2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ecc = np.where(l1 == 0, 0.0, np.sqrt(1.0 - l2 / l1))

    out: dict[str, np.ndarray] = {
        "label": labels,
        "centroid-0": cy,
        "centroid-1": cx,
        "area": area,
        "equivalent_diameter_area": (4.0 * area / np.pi) ** (1 / 2),
        "major_axis_length": major,
        "minor_axis_length": minor,
        "eccentricity": ecc,
        "bbox-0": bb[sel - 1, 0],
        "bbox-1": bb[sel - 1, 1],
        "bbox-2": bb[sel - 1, 2],
        "bbox-3": bb[sel - 1, 3],
    }
    if with_perimeter:
        dense = np.zeros(n_max + 1, np.int64)
        dense[labels] = np.arange(labels.size, dtype=np.int64)
        out["perimeter"] = label_perimeter(lab, dense, labels.size)
    if with_solidity:
        from spyde.particles.hull import convex_areas

        conv = convex_areas(lab, labels, counts)
        out["solidity"] = (area / conv.astype(np.float64)) if conv is not None \
            else solidity_table(lab)
    return out


def solidity_table(lab: np.ndarray) -> np.ndarray:
    """``solidity`` for every region, on skimage's per-region convex hull.

    The fallback for :func:`spyde.particles.hull.convex_areas` — used when numba
    is missing or its kernel will not compile. Kept because it is the definition
    everything else is checked against, and because it must still be possible to
    measure a frame on a machine with no numba at all. It is ~170x slower (30.5 s
    against 0.18 s at 26 566 regions), so this is a correctness floor, not a
    performance one.
    """
    from skimage.measure import regionprops_table

    return regionprops_table(lab, properties=("solidity",))["solidity"]

"""
hull.py — ``area_convex`` for every region at once, exactly, without Qhull.

``solidity`` is ``area / area_convex``, and ``area_convex`` is the one property in
SpyDE's measured set that is **not** a reduction over the raster: it is a convex
hull per region. At the scale that matters — a real 4096² in-situ growth frame
with 26 566 particles — ``regionprops_table``'s ``solidity`` alone is **30.5 s**,
64% of the whole measurement (``benchmarks.md``). Every other column is now a
``bincount`` (:mod:`spyde.particles.props`); this one is why the frame is still
slow.

The cost is not the hull. A particle here averages 33 pixels, so its hull is a
dozen points — microseconds of arithmetic. The cost is **per region overhead**:
two ``scipy.spatial.ConvexHull`` (Qhull) calls, a ``unique_rows``, a
``grid_points_in_poly`` and the Python object around them, ~1.1 ms each,
26 566 times. So this module keeps skimage's DEFINITION exactly and removes only
the overhead, in a numba kernel that runs every region in one ``prange``.

Exactly skimage's definition, in exact integer arithmetic
---------------------------------------------------------
``convex_hull_image`` (with its defaults, which is what ``regionprops`` uses):

1. reduces the region to the first/last pixel of each row (``possible_hull``);
2. replaces every such pixel ``(r, c)`` with the four **diamond offsets**
   ``(r±0.5, c)``, ``(r, c±0.5)``;
3. takes the convex hull of that point set;
4. returns the grid points that are inside **or on** it
   (``grid_points_in_poly(..., binarize=False)`` then ``labels >= 1``);
5. ``area_convex`` is the count of those grid points.

Each step is reproduced here, and the half-integer coordinates are the reason it
can be done **without floating point at all**: doubling every coordinate turns the
offsets into integers (``(2r±1, 2c)``, ``(2r, 2c±1)``), so the monotone-chain
cross products and the inside test are exact ``int64``. There is no tolerance to
tune and no tie to lose — a grid point is inside iff every edge's cross product is
non-negative, which is a comparison of integers.

Reducing to row extremes first is not an approximation: a pixel strictly between
the first and last of its own row lies in their convex hull, so it can never be a
hull vertex. (skimage also keeps column extremes; a superset of the vertices gives
the same hull either way.)

Why numba and not numpy
-----------------------
A hull is a stack algorithm — sequential per region, and the regions are tiny and
numerous, which is the one shape ``bincount`` cannot express. ``numba.njit`` with
``nogil=True`` gives the two things the batch run needs from it: the arithmetic
compiled instead of interpreted, and **the GIL released**, which is what lets a
dask worker's task slots be worth more than one core (``benchmarks.md``:
``regionprops_table`` in four threads measured 0.93x of serial).

``numba`` is optional — :func:`convex_areas` returns ``None`` when it is missing
or the kernel refuses to compile, and :mod:`spyde.particles.measure` falls back to
``regionprops_table``. Nothing here is on a GPU, so there is no device lock to
take.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_KERNEL = [None]          # compiled lazily, once per process
_FAILED = [False]


def _build_kernel():
    """Compile the per-region hull kernel, or return None if numba is unusable."""
    if _KERNEL[0] is not None or _FAILED[0]:
        return _KERNEL[0]
    try:
        import numba
    except Exception as exc:                                  # pragma: no cover
        log.info("[particles] numba unavailable (%s); solidity stays on "
                 "regionprops", exc)
        _FAILED[0] = True
        return None

    @numba.njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _kernel(starts, counts, rows, cols, out):             # pragma: no cover
        n = starts.shape[0]
        for i in numba.prange(n):
            s = starts[i]
            m = counts[i]
            if m <= 0:
                out[i] = 0
                continue

            # ── step 1: first/last pixel of each row, in DOUBLED coordinates.
            # The pixels arrive in raster order, so a row's run is contiguous.
            cand_x = np.empty(2 * m, np.int64)
            cand_y = np.empty(2 * m, np.int64)
            nc = 0
            j = s
            end = s + m
            while j < end:
                r = rows[j]
                k = j
                while k + 1 < end and rows[k + 1] == r:
                    k += 1
                cand_x[nc] = 2 * cols[j]
                cand_y[nc] = 2 * r
                nc += 1
                if k != j:
                    cand_x[nc] = 2 * cols[k]
                    cand_y[nc] = 2 * r
                    nc += 1
                j = k + 1

            # ── step 2: the diamond offsets, still integers once doubled.
            np_ = 4 * nc
            px = np.empty(np_, np.int64)
            py = np.empty(np_, np.int64)
            key = np.empty(np_, np.int64)
            for a in range(nc):
                bx = cand_x[a]
                by = cand_y[a]
                px[4 * a + 0] = bx - 1
                py[4 * a + 0] = by
                px[4 * a + 1] = bx + 1
                py[4 * a + 1] = by
                px[4 * a + 2] = bx
                py[4 * a + 2] = by - 1
                px[4 * a + 3] = bx
                py[4 * a + 3] = by + 1
            # Sort by (x, y) for the monotone chain. One int64 key rather than a
            # lexsort: the doubled coordinates are bounded by 2*frame_size+1, so
            # a 2^20 stride is exact for any frame up to ~262 000 px a side.
            for a in range(np_):
                key[a] = px[a] * 1048576 + py[a]
            order = np.argsort(key)

            # ── step 3: monotone chain over the sorted points.
            hx = np.empty(2 * np_ + 1, np.int64)
            hy = np.empty(2 * np_ + 1, np.int64)
            nh = 0
            for a in range(np_):
                x = px[order[a]]
                y = py[order[a]]
                while nh >= 2 and (
                        (hx[nh - 1] - hx[nh - 2]) * (y - hy[nh - 2])
                        - (hy[nh - 1] - hy[nh - 2]) * (x - hx[nh - 2])) <= 0:
                    nh -= 1
                hx[nh] = x
                hy[nh] = y
                nh += 1
            lower = nh
            for a in range(np_ - 2, -1, -1):
                x = px[order[a]]
                y = py[order[a]]
                while nh > lower and (
                        (hx[nh - 1] - hx[nh - 2]) * (y - hy[nh - 2])
                        - (hy[nh - 1] - hy[nh - 2]) * (x - hx[nh - 2])) <= 0:
                    nh -= 1
                hx[nh] = x
                hy[nh] = y
                nh += 1
            nh -= 1                        # the closing point repeats hull[0]

            if nh < 3:
                out[i] = m               # cannot happen once offset; be safe
                continue

            # ── steps 4-5: count grid points inside or ON the hull. The hull of
            # the offsets extends 0.5 px past the region, so it can only cover
            # grid points inside the region's own bbox.
            y0 = rows[s]
            y1 = rows[end - 1]
            x0 = cols[s]
            x1 = cols[s]
            for a in range(s + 1, end):
                c = cols[a]
                if c < x0:
                    x0 = c
                if c > x1:
                    x1 = c

            total = 0
            for yy in range(y0, y1 + 1):
                gy = 2 * yy
                started = False
                for xx in range(x0, x1 + 1):
                    gx = 2 * xx
                    inside = True
                    for e in range(nh):
                        e2 = e + 1
                        if e2 == nh:
                            e2 = 0
                        if ((hx[e2] - hx[e]) * (gy - hy[e])
                                - (hy[e2] - hy[e]) * (gx - hx[e])) < 0:
                            inside = False
                            break
                    if inside:
                        total += 1
                        started = True
                    elif started:
                        break            # convex: one run per row
            out[i] = total

    _KERNEL[0] = _kernel
    return _kernel


def convex_areas(lab: np.ndarray, labels: np.ndarray, counts: np.ndarray,
                 ) -> np.ndarray | None:
    """``area_convex`` per entry of *labels*, or None if numba is unavailable.

    Parameters
    ----------
    lab
        The ``(h, w)`` int label image.
    labels
        Ascending labels present in *lab* — the same order
        ``regionprops_table`` emits.
    counts
        ``np.bincount(lab.ravel())``, i.e. pixel counts indexed BY LABEL.

    Notes
    -----
    Memory: one int64 index array and two int32 coordinate arrays sized by the
    FOREGROUND pixel count, nothing sized by the number of regions squared and
    nothing that materialises a per-region mask (CLAUDE.md § Memory Safety).
    """
    kernel = _build_kernel()
    if kernel is None:
        return None

    h, w = lab.shape
    flat = lab.reshape(-1)
    nz = np.flatnonzero(flat)
    if nz.size == 0:
        return np.zeros(labels.shape, np.int64)
    lab_of = flat[nz]
    # Stable sort groups the pixels by label while KEEPING raster order inside
    # each group, which is what the kernel's row-run scan relies on. numpy uses
    # a radix sort for integer keys here, so this is a linear pass.
    order = np.argsort(lab_of, kind="stable")
    nz = nz[order]
    rows = (nz // w).astype(np.int32)
    cols = (nz - rows.astype(np.int64) * w).astype(np.int32)

    grp = counts[labels].astype(np.int64)
    starts = np.zeros(labels.size, np.int64)
    np.cumsum(grp[:-1], out=starts[1:])

    out = np.zeros(labels.size, np.int64)
    try:
        kernel(starts, grp, rows, cols, out)
    except Exception as exc:                                  # pragma: no cover
        log.warning("[particles] hull kernel failed (%r); solidity falls back "
                    "to regionprops", exc)
        _FAILED[0] = True
        _KERNEL[0] = None
        return None
    return out


def warmup() -> bool:
    """Compile the kernel on a 3-pixel toy image. Returns True if numba is live.

    Worth calling once on a dask worker: the first ``njit`` call pays the
    compile, and paying it inside the first measured frame makes that frame look
    like a regression. ``cache=True`` means it is paid once per machine, not once
    per process, but a cold cache still has to build.
    """
    lab = np.zeros((4, 4), np.int32)
    lab[1:3, 1:3] = 1
    counts = np.bincount(lab.reshape(-1))
    return convex_areas(lab, np.array([1], np.int64), counts) is not None

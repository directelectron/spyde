"""
intensity.py — the intensity columns as label-wise reductions, and the ring as a kernel.

:mod:`spyde.particles.props` removed ``regionprops_table``'s Python loop over
regions and :mod:`spyde.particles.hull` removed ``solidity``'s. What was left of
``measure_frame`` was two loops of exactly the same shape, and this module is one
of them: ``_fill_intensity`` was **4.9 s of a 10.2 s frame** on a real 4096²
in-situ growth raster with 26 566 particles (``benchmarks.md``), spent almost
entirely on per-region overhead — a bbox crop, a mask, a boolean take and a
``scipy.ndimage.binary_dilation``, 26 566 times, all of it holding the GIL.

The four columns split cleanly in two, and only one of them is hard:

* ``intensity_mean`` / ``intensity_max`` / ``intensity_std`` are **label-wise
  reductions over the foreground pixels** — ``bincount`` with weights for the
  sums, one label-grouped ``np.maximum.reduceat`` for the max. Same shape as the
  moments in :mod:`~spyde.particles.props`, and O(foreground) rather than
  O(regions x crop).
* ``background`` is not. It is the mean intensity of the pixels a **dilation of
  this particle by ``ring``** adds and that belong to no particle — a *per
  particle* neighbourhood that overlapping neighbours may each claim, so it is
  not a partition of the raster and no ``bincount`` expresses it.
  :func:`ring_backgrounds` keeps the definition exactly (an iterated
  4-connected dilation inside the same padded bbox crop, which is what
  ``binary_dilation``'s default structure and ``border_value=0`` do) and moves
  it into a ``numba`` kernel that runs every region in one ``prange`` with the
  GIL released, the way :mod:`~spyde.particles.hull` does for the convex hull.

Parity
------
The pixel SETS are identical by construction — every statistic here is taken over
exactly the pixels the per-region crop selected, including the finite-only filter
that keeps a NaN-padded drift-corrected border from poisoning a particle that
touches it. What differs is **summation order**: ``np.mean``/``np.std`` reduce
pairwise, ``bincount`` and the kernel accumulate sequentially, so the float64
intermediates disagree at ~1e-16 relative. The rows are stored in float32, where
that is 9 orders below the last bit, and the parity test asserts the stored
columns come out **bit-identical** on a scene of thousands of ragged regions.

``intensity_std`` is computed the way ``np.std`` computes it — mean first, then
the mean of squared deviations about it — and NOT as ``E[x²] - E[x]²``, which is
algebraically equal and loses digits to cancellation exactly where the variance
is small compared to the mean, which is the normal case for a bright particle.

Memory: the working set is one float64 and one index array sized by the
FOREGROUND pixel count, plus one padded bbox crop per region at a time inside the
kernel (CLAUDE.md § Memory Safety). Nothing scales with the number of regions
squared and nothing materialises a full-frame per-region mask.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_RING_KERNEL = [None]        # compiled lazily, once per process
_RING_FAILED = [False]


def label_intensity_stats(
    lab: np.ndarray,
    inten: np.ndarray,
    labels: np.ndarray,
    *,
    n_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean, max, std_over_max)`` per entry of *labels*, over FINITE pixels.

    Reproduces, for every region at once, what the per-region crop computed::

        vals = sub_int[sub_lab == lbl]
        vals = vals[np.isfinite(vals)]
        mean, mx = vals.mean(), vals.max()
        std = vals.std() / mx if mx else nan

    A region with no finite pixel gets NaN in all three, and a region whose
    finite maximum is exactly 0 gets NaN for the normalised deviation — both are
    the per-region path's own behaviour, not a new convention.

    *labels* must be ascending and must be exactly the labels present, which is
    what :func:`spyde.particles.props.label_props` emits.
    """
    lab = np.asarray(lab)
    inten = np.asarray(inten, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    nan3 = np.full(n, np.nan)
    if n == 0:
        return nan3, nan3.copy(), nan3.copy()

    flat = lab.reshape(-1)
    nz = np.flatnonzero(flat)
    if nz.size == 0:
        return nan3, nan3.copy(), nan3.copy()

    if n_max is None:
        n_max = int(labels[-1])
    # label value -> row of the output. Sparse label images are normal here (any
    # upstream filter re-tags), so the row index is dense and the label is not.
    dense = np.zeros(int(n_max) + 1, np.int64)
    dense[labels] = np.arange(n, dtype=np.int64)

    v = inten.reshape(-1)[nz]
    fin = np.isfinite(v)
    if not fin.all():
        v = v[fin]
        li = dense[flat[nz[fin]]]
    else:
        li = dense[flat[nz]]
    del nz, fin

    cnt = np.bincount(li, minlength=n)[:n]
    have = cnt > 0
    cnt_f = cnt.astype(np.float64)

    s = np.bincount(li, weights=v, minlength=n)[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s / cnt_f
    mean[~have] = np.nan

    # Two-pass deviation, matching `np.std`: the one-pass E[x^2]-E[x]^2 is
    # algebraically the same and cancels away the digits that matter when a
    # particle is bright and uniform.
    dev = v - mean[li]
    dev *= dev
    s2 = np.bincount(li, weights=dev, minlength=n)[:n]
    del dev
    with np.errstate(divide="ignore", invalid="ignore"):
        std = np.sqrt(s2 / cnt_f)

    # Per-label max. Grouping by label is a stable (radix) sort of the
    # foreground, after which one `maximum.reduceat` covers every region; the
    # alternative, `np.maximum.at`, is an unbuffered ufunc call per pixel and is
    # ~50x slower than the sort it avoids.
    mx = np.full(n, np.nan)
    if li.size:
        order = np.argsort(li, kind="stable")
        vs = v[order]
        starts = np.zeros(n, np.int64)
        np.cumsum(cnt[:-1], out=starts[1:])
        mx[have] = np.maximum.reduceat(vs, starts[have])
        del order, vs

    with np.errstate(divide="ignore", invalid="ignore"):
        std_norm = np.where(mx != 0, std / mx, np.nan)
    std_norm[~have] = np.nan
    return mean, mx, std_norm


def _build_ring_kernel():
    """Compile the per-region ring kernel, or return None if numba is unusable."""
    if _RING_KERNEL[0] is not None or _RING_FAILED[0]:
        return _RING_KERNEL[0]
    try:
        import numba
    except Exception as exc:                                  # pragma: no cover
        log.info("[particles] numba unavailable (%s); the background ring stays "
                 "on the per-region loop", exc)
        _RING_FAILED[0] = True
        return None

    @numba.njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _kernel(lab, inten, labels, bb, ring, sums, counts):  # pragma: no cover
        h, w = lab.shape
        n = labels.shape[0]
        for i in numba.prange(n):
            lbl = labels[i]
            # The SAME crop the per-region path takes: the bbox grown by
            # ring + 1 and clipped to the frame, so a dilation by `ring` fits
            # inside it and a region at the frame edge is truncated exactly as
            # `binary_dilation` truncates it there.
            y0 = bb[i, 0] - ring - 1
            x0 = bb[i, 1] - ring - 1
            y1 = bb[i, 2] + ring + 1
            x1 = bb[i, 3] + ring + 1
            if y0 < 0:
                y0 = 0
            if x0 < 0:
                x0 = 0
            if y1 > h:
                y1 = h
            if x1 > w:
                x1 = w
            hh = y1 - y0
            ww = x1 - x0
            if hh <= 0 or ww <= 0:
                sums[i] = 0.0
                counts[i] = 0
                continue

            cur = np.zeros((hh, ww), np.uint8)
            for r in range(hh):
                for c in range(ww):
                    if lab[y0 + r, x0 + c] == lbl:
                        cur[r, c] = 1

            # `binary_dilation(m, iterations=ring)` with scipy's default
            # structure is `generate_binary_structure(2, 1)` — the 4-connected
            # cross — applied `ring` times, i.e. everything within city-block
            # distance `ring`. `border_value=0` is the bounds check below.
            if ring > 0:
                nxt = np.zeros((hh, ww), np.uint8)
                for _it in range(ring):
                    for r in range(hh):
                        for c in range(ww):
                            v = cur[r, c]
                            if v == 0:
                                if r > 0 and cur[r - 1, c] != 0:
                                    v = 1
                                elif r + 1 < hh and cur[r + 1, c] != 0:
                                    v = 1
                                elif c > 0 and cur[r, c - 1] != 0:
                                    v = 1
                                elif c + 1 < ww and cur[r, c + 1] != 0:
                                    v = 1
                            nxt[r, c] = v
                    tmp = cur
                    cur = nxt
                    nxt = tmp

            # The ring is what the dilation added MINUS anything belonging to a
            # neighbouring particle: a touching particle's body is not this
            # one's background.
            s = 0.0
            k = 0
            for r in range(hh):
                for c in range(ww):
                    if cur[r, c] != 0 and lab[y0 + r, x0 + c] == 0:
                        val = inten[y0 + r, x0 + c]
                        if np.isfinite(val):
                            s += val
                            k += 1
            sums[i] = s
            counts[i] = k

    _RING_KERNEL[0] = _kernel
    return _kernel


def ring_backgrounds(lab: np.ndarray, inten: np.ndarray, labels: np.ndarray,
                     bboxes: np.ndarray, ring: int) -> np.ndarray | None:
    """Mean background per entry of *labels*, or None if numba is unavailable.

    Parameters
    ----------
    lab
        The ``(h, w)`` int label image.
    inten
        The ``(h, w)`` float64 intensity image. May contain NaN; NaN pixels are
        excluded from the mean rather than coerced to zero, which would invent a
        dark rim on every particle touching a drift-corrected border.
    labels
        Ascending labels present in *lab*.
    bboxes
        ``(n, 4)`` int ``(y0, x0, y1, x1)`` per entry of *labels* — the same
        tight bboxes the property table reports.
    ring
        Dilation width in pixels. ``0`` returns all-NaN (the caller leaves the
        column unset), matching the per-region path's ``if ring > 0`` guard.

    Notes
    -----
    Memory: one padded bbox crop at a time per thread, plus two float/int arrays
    of length ``n``. Nothing full-frame is allocated (CLAUDE.md § Memory Safety).
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    if n == 0:
        return np.zeros((0,), np.float64)
    if int(ring) <= 0:
        return np.full(n, np.nan)

    kernel = _build_ring_kernel()
    if kernel is None:
        return None

    bb = np.ascontiguousarray(bboxes, dtype=np.int64)
    sums = np.zeros(n, np.float64)
    counts = np.zeros(n, np.int64)
    try:
        kernel(lab, np.asarray(inten, np.float64), labels, bb, int(ring),
               sums, counts)
    except Exception as exc:                                  # pragma: no cover
        log.warning("[particles] ring kernel failed (%r); background falls back "
                    "to the per-region dilation", exc)
        _RING_FAILED[0] = True
        _RING_KERNEL[0] = None
        return None

    out = np.full(n, np.nan)
    have = counts > 0
    out[have] = sums[have] / counts[have].astype(np.float64)
    return out


def warmup() -> bool:
    """Compile the ring kernel on a toy image. Returns True if numba is live.

    Worth calling once on a dask worker, for the reason
    :func:`spyde.particles.hull.warmup` exists: paying the first ``njit``
    compile inside the first measured frame makes that frame look like a
    regression.
    """
    lab = np.zeros((8, 8), np.int32)
    lab[3:5, 3:5] = 1
    inten = np.ones((8, 8), np.float64)
    bb = np.array([[3, 3, 5, 5]], np.int64)
    return ring_backgrounds(lab, inten, np.array([1], np.int64), bb, 2) is not None

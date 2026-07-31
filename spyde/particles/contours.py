"""
contours.py — ``find_contours`` for every region at once, with the same FILLED polygon.

The last of ``measure_frame``'s three per-region Python loops. On a real 4096²
in-situ growth frame with 26 566 particles it is **4.8 s of a 10.2 s frame**
(``benchmarks.md``), and, like the property table and the convex hull before it,
almost none of that is arithmetic: it is a bbox crop, a ``float32`` cast, a
``skimage.measure.find_contours`` call whose marching squares is Cython but whose
segment ASSEMBLY is a pure-Python dict-and-deque walk, a ``max``, an ``rint`` and
a ``clip`` — ~177 us per region, 26 566 times, holding the GIL throughout.

The outline is not decoration
-----------------------------
It is tempting to treat an outline as a display choice and allow a "close enough"
polygon. It is not: :meth:`spyde.signals.particles.SpyDEParticles.render_frame`
FILLS the contours to rebuild the label movie, and :meth:`~…SpyDEParticles.mask_at`
fills one to produce the per-particle mask that a mean diffraction pattern is
sliced with. A different contour is a different mask is a different measurement.

But that also says exactly what the gate is, and it is weaker than bit-identical
vertices:

    **The FILLED POLYGON must be identical, per region.** Vertex count, ordering
    and starting point may differ; ``skimage.draw.polygon`` on the new contour
    must select exactly the same pixels as on the old one.

That is what every consumer reads, so nothing downstream can observe a difference
that survives it. ``test_particles_contours_parity.py`` asserts it as a boolean
``array_equal`` of the two filled masks, per region, on a scene of thousands.

Why the cycle is the same cycle
-------------------------------
skimage's ``_get_contour_segments`` is a fixed 16-entry case table over the 2x2
cells of the crop, and on a BINARY mask at level 0.5 every interpolated vertex
lands exactly on an edge midpoint — ``_get_fraction`` is ``(0.5 - 0) / (1 - 0)``
for every edge the table actually uses. So a vertex is at ``(i + 0.5, j)`` or
``(i, j + 0.5)``, i.e. it IS the crack between two 4-adjacent pixels, and it can
be named by an integer edge index with no floating point anywhere.

Each cell emits its segments oriented so that low values are on the left, and a
crack interior to the crop is shared by exactly two cells — appearing once as a
segment's tail and once as another's head. So every vertex has in-degree <= 1 and
out-degree <= 1, and the segment set decomposes into disjoint simple paths (which
run off the crop border) and cycles. ``_assemble_contours``'s dicts and deques
recover exactly those maximal chains; following ``succ`` recovers the same ones,
in the same direction, differing only in where a CYCLE is cut — which a fill
cannot see. A cycle is emitted with its first vertex repeated at the end, which is
what ``_assemble_contours`` does when it closes a loop, so the vertex COUNT
matches too.

Two details that look like trivia and decide the answer
-------------------------------------------------------
* **``np.rint`` is round-half-to-EVEN, and it is applied in CROP coordinates.**
  Every marching-squares vertex here is a half-integer, so the rounding is
  entirely in the tie case, and it resolves on the PARITY of the crop-local
  coordinate — which depends on where the region's padded bbox happens to start.
  Two congruent particles at different positions therefore get genuinely
  different integer outlines. That is the behaviour on disk today; reproducing
  it means rounding in the crop frame and offsetting afterwards, never the
  reverse.
* **Which contour "the" contour is.** The caller takes ``max(cs, key=len)``, and
  ``cs`` is ordered by ``_assemble_contours``'s creation counter — which, after
  every merge keeps the smaller of the two keys, equals the order of each
  contour's SMALLEST segment index. So ties in length break to the chain
  containing the earliest cell in raster order, and that is reproduced here
  explicitly rather than left to whatever order a walk happens to discover.

``numba`` is optional: :func:`label_contours` returns ``None`` when it is missing
or the kernel will not compile, and :mod:`spyde.particles.measure` falls back to
the ``find_contours`` loop. Nothing here is on a GPU, so there is no device lock.

Memory: one padded bbox crop at a time per thread, plus one flat int16 buffer for
every outline, bounded by ``4 * area + 4`` per region — a region's crossed cracks
cannot exceed four per pixel (CLAUDE.md § Memory Safety).
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

_KERNEL = [None]          # compiled lazily, once per process
_FAILED = [False]


def _build_kernel():
    """Compile the per-region contour kernel, or None if numba is unusable."""
    if _KERNEL[0] is not None or _FAILED[0]:
        return _KERNEL[0]
    try:
        import numba
    except Exception as exc:                                  # pragma: no cover
        log.info("[particles] numba unavailable (%s); contours stay on "
                 "find_contours", exc)
        _FAILED[0] = True
        return None

    @numba.njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _kernel(lab, labels, bb, offsets, out_buf, out_len):  # pragma: no cover
        h, w = lab.shape
        n = labels.shape[0]
        for i in numba.prange(n):
            lbl = labels[i]
            by0 = bb[i, 0]
            bx0 = bb[i, 1]
            by1 = bb[i, 2]
            bx1 = bb[i, 3]
            base = offsets[i]
            # An outline cannot have more vertices than the region has cracks,
            # and a pixel has at most four — so `cap` is a proof, not a guess.
            # It is enforced anyway: njit does not bounds-check, and a wrong
            # bound would be silent memory corruption rather than an IndexError.
            cap = offsets[i + 1] - base

            # The same crop `_contours` takes: the bbox padded by ONE and
            # clipped to the frame. One pixel is all marching squares needs to
            # close a contour around the region; at the frame edge the crop is
            # truncated and the contour is left open, exactly as skimage leaves
            # it open against an array border.
            py0 = by0 - 1
            px0 = bx0 - 1
            py1 = by1 + 1
            px1 = bx1 + 1
            if py0 < 0:
                py0 = 0
            if px0 < 0:
                px0 = 0
            if py1 > h:
                py1 = h
            if px1 > w:
                px1 = w
            hh = py1 - py0
            ww = px1 - px0

            # `find_contours` refuses an array smaller than 2x2, and returns
            # nothing for a crop that is entirely inside or entirely outside the
            # region. Both are the caller's degenerate branch: the bbox corners,
            # so that every row still has an outline and the 1:1 correspondence
            # holds.
            m = np.zeros((1, 1), np.uint8)
            nseg = 0
            if hh >= 2 and ww >= 2:
                m = np.zeros((hh, ww), np.uint8)
                for r in range(hh):
                    for c in range(ww):
                        if lab[py0 + r, px0 + c] == lbl:
                            m[r, c] = 1

                # ── pass 1: how many segments the case table will emit.
                for r0 in range(hh - 1):
                    for c0 in range(ww - 1):
                        case = (m[r0, c0] + 2 * m[r0, c0 + 1]
                                + 4 * m[r0 + 1, c0] + 8 * m[r0 + 1, c0 + 1])
                        if case == 0 or case == 15:
                            continue
                        nseg += 2 if (case == 6 or case == 9) else 1

            if nseg == 0:
                ya = by0 if by0 < h else h - 1
                yb = by1 - 1 if by1 - 1 < h else h - 1
                xa = bx0 if bx0 < w else w - 1
                xb = bx1 - 1 if bx1 - 1 < w else w - 1
                out_buf[base + 0, 0] = ya
                out_buf[base + 0, 1] = xa
                out_buf[base + 1, 0] = ya
                out_buf[base + 1, 1] = xb
                out_buf[base + 2, 0] = yb
                out_buf[base + 2, 1] = xb
                out_buf[base + 3, 0] = yb
                out_buf[base + 3, 1] = xa
                out_len[i] = 4
                continue

            # A vertex is a CRACK between two 4-adjacent crop pixels: `nh`
            # horizontal cracks (between columns) then the vertical ones.
            nh = hh * (ww - 1)
            nv = nh + (hh - 1) * ww
            seg_from = np.empty(nseg, np.int32)
            seg_to = np.empty(nseg, np.int32)
            succ = np.full(nv, -1, np.int32)
            has_pred = np.zeros(nv, np.uint8)

            # ── pass 2: the case table itself. Segment ORDER is row-major over
            # cells, which is what fixes the tie-break below.
            k = 0
            for r0 in range(hh - 1):
                for c0 in range(ww - 1):
                    case = (m[r0, c0] + 2 * m[r0, c0 + 1]
                            + 4 * m[r0 + 1, c0] + 8 * m[r0 + 1, c0 + 1])
                    if case == 0 or case == 15:
                        continue
                    top = r0 * (ww - 1) + c0
                    bottom = (r0 + 1) * (ww - 1) + c0
                    left = nh + r0 * ww + c0
                    right = nh + r0 * ww + c0 + 1

                    a0 = -1
                    b0 = -1
                    a1 = -1
                    b1 = -1
                    if case == 1:
                        a0 = top
                        b0 = left
                    elif case == 2:
                        a0 = right
                        b0 = top
                    elif case == 3:
                        a0 = right
                        b0 = left
                    elif case == 4:
                        a0 = left
                        b0 = bottom
                    elif case == 5:
                        a0 = top
                        b0 = bottom
                    elif case == 6:
                        a0 = right
                        b0 = top
                        a1 = left
                        b1 = bottom
                    elif case == 7:
                        a0 = right
                        b0 = bottom
                    elif case == 8:
                        a0 = bottom
                        b0 = right
                    elif case == 9:
                        a0 = top
                        b0 = left
                        a1 = bottom
                        b1 = right
                    elif case == 10:
                        a0 = bottom
                        b0 = top
                    elif case == 11:
                        a0 = bottom
                        b0 = left
                    elif case == 12:
                        a0 = left
                        b0 = right
                    elif case == 13:
                        a0 = top
                        b0 = right
                    else:                                     # case == 14
                        a0 = left
                        b0 = top

                    seg_from[k] = a0
                    seg_to[k] = b0
                    succ[a0] = k
                    has_pred[b0] = 1
                    k += 1
                    if a1 >= 0:
                        seg_from[k] = a1
                        seg_to[k] = b1
                        succ[a1] = k
                        has_pred[b1] = 1
                        k += 1

            # ── pass 3: follow `succ` to recover the maximal chains, and keep
            # the longest. Open paths first (they have a vertex with no
            # predecessor to start from), then whatever is left, which is
            # cycles.
            used = np.zeros(nseg, np.uint8)
            verts = np.empty(nseg + 1, np.int32)
            best = np.empty(nseg + 1, np.int32)
            best_len = 0
            best_min = 0

            for phase in range(2):
                for j in range(nseg):
                    if used[j] != 0:
                        continue
                    if phase == 0 and has_pred[seg_from[j]] != 0:
                        continue
                    cnt = 0
                    verts[cnt] = seg_from[j]
                    cnt += 1
                    minidx = j
                    e = j
                    while True:
                        used[e] = 1
                        if e < minidx:
                            minidx = e
                        verts[cnt] = seg_to[e]
                        cnt += 1
                        nxt = succ[seg_to[e]]
                        if nxt < 0 or used[nxt] != 0:
                            break
                        e = nxt
                    if cnt > best_len or (cnt == best_len and minidx < best_min):
                        best_len = cnt
                        best_min = minidx
                        for a in range(cnt):
                            best[a] = verts[a]

            # ── pass 4: half-integer crack -> np.rint (half to EVEN) in CROP
            # coordinates, then offset to the frame and clip.
            if best_len > cap:
                best_len = cap
            for a in range(best_len):
                v = best[a]
                if v < nh:
                    rr = v // (ww - 1)
                    cc = v - rr * (ww - 1)
                    dr = 2 * rr
                    dc = 2 * cc + 1
                else:
                    u = v - nh
                    rr = u // ww
                    cc = u - rr * ww
                    dr = 2 * rr + 1
                    dc = 2 * cc

                kr = dr >> 1
                if (dr & 1) != 0 and (kr & 1) != 0:
                    kr += 1
                kc = dc >> 1
                if (dc & 1) != 0 and (kc & 1) != 0:
                    kc += 1

                y = kr + py0
                x = kc + px0
                if y < 0:
                    y = 0
                elif y > h - 1:
                    y = h - 1
                if x < 0:
                    x = 0
                elif x > w - 1:
                    x = w - 1
                out_buf[base + a, 0] = y
                out_buf[base + a, 1] = x
            out_len[i] = best_len

    _KERNEL[0] = _kernel
    return _kernel


def label_contours(lab: np.ndarray, labels: np.ndarray, bboxes: np.ndarray,
                   areas: np.ndarray) -> list[np.ndarray] | None:
    """One int16 ``(k, 2)`` outline per entry of *labels*, or None without numba.

    Parameters
    ----------
    lab
        The ``(h, w)`` int label image.
    labels
        Ascending labels present in *lab*.
    bboxes
        ``(n, 4)`` int ``(y0, x0, y1, x1)`` per entry of *labels*.
    areas
        Pixel count per entry of *labels*. Used only to size the output buffer:
        a region's outline cannot have more vertices than it has cracks, and a
        pixel has at most four.

    Returns
    -------
    A list of ``(k, 2)`` int16 arrays, one per label and in the same order — the
    1:1 correspondence ``SpyDEParticles.from_frames`` requires. Each is a VIEW
    into one contiguous buffer, which is also the layout ``SpyDEParticles``
    stores (``contours`` + ``contour_offsets``).
    """
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.size
    if n == 0:
        return []
    kernel = _build_kernel()
    if kernel is None:
        return None

    bb = np.ascontiguousarray(bboxes, dtype=np.int64)
    bound = 4 * np.asarray(areas, dtype=np.int64) + 4
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(bound, out=offsets[1:])
    buf = np.zeros((int(offsets[-1]), 2), np.int16)
    lens = np.zeros(n, np.int64)
    try:
        kernel(lab, labels, bb, offsets, buf, lens)
    except Exception as exc:                                  # pragma: no cover
        log.warning("[particles] contour kernel failed (%r); outlines fall back "
                    "to find_contours", exc)
        _FAILED[0] = True
        _KERNEL[0] = None
        return None

    # Compact into a buffer of the ACTUAL size. `bound` is ~4x what an outline
    # really needs, and the caller (`batch.py`) accumulates one contour list per
    # frame for the whole movie — holding 900 over-allocated frames alive is
    # gigabytes for nothing, whereas the compacted CSR is what
    # `SpyDEParticles` stores anyway.
    new_off = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=new_off[1:])
    total = int(new_off[-1])
    within = np.arange(total, dtype=np.int64) - np.repeat(new_off[:n], lens)
    src = np.repeat(offsets[:n], lens) + within
    flat = np.ascontiguousarray(buf[src], np.int16)
    del buf, src, within
    return [flat[int(a):int(b)] for a, b in zip(new_off[:n], new_off[1:])]


def warmup() -> bool:
    """Compile the kernel on a toy image. Returns True if numba is live.

    The counterpart of :func:`spyde.particles.hull.warmup`: a dask worker should
    pay the first ``njit`` compile before the first measured frame, not inside it.
    """
    lab = np.zeros((8, 8), np.int32)
    lab[2:5, 2:5] = 1
    out = label_contours(lab, np.array([1], np.int64),
                         np.array([[2, 2, 5, 5]], np.int64),
                         np.array([9], np.int64))
    return out is not None

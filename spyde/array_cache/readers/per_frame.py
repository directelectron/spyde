"""Reader kind 5: per-frame derived view — rebin/crop computed in NUMPY from the
PARENT's frame, with no dask in the loop.

The problem this solves. A rebinned view's frame is a deterministic function of
ONE parent frame, but asking dask for it (``rebinned.data[y, x].compute()``)
makes dask materialise the whole enclosing source nav-chunk and re-run the graph.
Measured on a real .zspy 4D-STEM (64x64x512^2, 32x32 nav chunks = 537 MB):

    rebinned[3,3].compute()                    2403 ms
    parent frame read + numpy rebin             417 ms
      - of which: the parent chunk decode       435 ms   (unavoidable, cached)
      - of which: the numpy rebin                1.8 ms
    rebinned frame with the parent block WARM     1.8 ms

So the dask path costs ~2 s per scrub position on top of a read that has to
happen anyway, and the transform itself is noise. Reading the parent's frame
through the parent's OWN reader means the parent's block cache does the work
once and every later frame in that block is ~2 ms of numpy.

This is the "per-frame streaming instead of fat blockwise" idea from
DATA_ACCESS_PATTERNS.md case 3, applied to the read path: most SpyDE "chunk"
transforms are really frame functions lifted to chunks, so a chain
``f = fn ... f1 : frame -> frame`` can be composed and applied per frame.

SCOPE — deliberately narrow. Only transforms that are (a) purely per-frame and
(b) reproducible from the node's recorded args are handled; anything else
returns None from :func:`build_per_frame_reader` and falls back to
LocalTransformReader (correct, just slower). A wrong frame is far worse than a
slow one, so the gate is conservative: shapes are checked against the derived
signal's own shape before the reader is accepted.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def _rebin_fn(src_shape, out_shape):
    """frame -> frame rebin by integer factors, or None if not a clean rebin.

    hyperspy's rebin sums over integer-factor blocks; we reproduce that with a
    reshape + sum, which is what makes it ~2 ms instead of a graph."""
    if len(src_shape) != len(out_shape) or len(src_shape) != 2:
        return None
    factors = []
    for s, o in zip(src_shape, out_shape):
        if o <= 0 or s % o != 0:
            return None                      # not an integer down-factor
        factors.append(s // o)
    if all(f == 1 for f in factors):
        return None                          # identity — nothing to gain
    fy, fx = factors
    oy, ox = out_shape

    def _apply(frame, _fy=fy, _fx=fx, _oy=oy, _ox=ox):
        return frame.reshape(_oy, _fy, _ox, _fx).sum(axis=(1, 3))

    return _apply


def _crop_fn(src_shape, out_shape, node):
    """frame -> frame signal-space crop, from the node's recorded kwargs.

    Only a SIGNAL-space crop is handled here: a NAV crop shifts which parent
    frame each output index maps to, which is an index remap rather than a
    per-frame function, and is left to the dask path."""
    kw = dict(getattr(node, "kwargs", None) or {})
    try:
        x0, x1 = int(kw.get("x0", 0)), int(kw.get("x1", 0))
        y0, y1 = int(kw.get("y0", 0)), int(kw.get("y1", 0))
        t0, t1 = int(kw.get("t0", 0)), int(kw.get("t1", 0))
    except Exception:
        return None
    if t0 or t1:
        return None                          # nav (time) crop — not per-frame
    # Zero ranges mean "keep the full extent" (CropAction's convention).
    ys = slice(y0, y1) if (y0 or y1) else slice(None)
    xs = slice(x0, x1) if (x0 or x1) else slice(None)

    def _apply(frame, _ys=ys, _xs=xs):
        return frame[_ys, _xs]

    probe = _apply(np.zeros(src_shape, dtype=np.uint8))
    if tuple(probe.shape) != tuple(out_shape):
        return None                          # our reading of the crop is wrong
    return _apply


def build_per_frame_reader(signal, data, node, parent_reader, parent_signal):
    """A :class:`PerFrameReader` for ``signal``, or None to fall back.

    ``node`` is the signal's SignalNode (carries ``transformation`` + the args it
    was built with); ``parent_reader`` serves the parent signal's frames."""
    if node is None or parent_reader is None:
        return None
    transform = getattr(node, "transformation", None)
    try:
        nav_ndim = signal.axes_manager.navigation_dimension
        src_shape = tuple(int(v) for v in parent_signal.data.shape[nav_ndim:])
        out_shape = tuple(int(v) for v in data.shape[nav_ndim:])
        if parent_signal.data.shape[:nav_ndim] != data.shape[:nav_ndim]:
            return None                      # nav grid changed — index remap
    except Exception:
        return None

    fn = None
    if transform == "rebin":
        fn = _rebin_fn(src_shape, out_shape)
    elif transform in ("_crop_signal", "crop"):
        fn = _crop_fn(src_shape, out_shape, node)
    if fn is None:
        return None

    # Final gate: the composed function must actually produce this signal's
    # frame shape. A silently wrong frame is much worse than a slow one.
    try:
        probe = fn(np.zeros(src_shape, dtype=data.dtype))
        if tuple(probe.shape) != out_shape:
            return None
    except Exception:
        return None
    return PerFrameReader(signal, data, parent_reader, fn)


class PerFrameReader:
    """Applies a per-frame numpy transform to the PARENT's frames.

    Holds no cache of its own: the parent's reader (and the plot's BlockCache
    behind it) does the decoding, so a scrub inside one parent block costs the
    transform only. Output frames are small and go in the plot's ArrayCache like
    any other, so a revisited position is a dict hit."""

    def __init__(self, signal, data, parent_reader, fn):
        self.signal = signal
        self.data = data
        self.parent_reader = parent_reader
        self._fn = fn
        self._nav_ndim = signal.axes_manager.navigation_dimension

    @property
    def frame_bytes(self) -> int:
        frame_shape = self.data.shape[self._nav_ndim:]
        return int(np.prod(frame_shape)) * self.data.dtype.itemsize

    def is_chunk_resident(self, indices) -> bool:
        """Cheap iff the PARENT's block is resident — the transform is ~ms."""
        probe = getattr(self.parent_reader, "is_chunk_resident", None)
        return bool(probe(indices)) if probe is not None else False

    def read_frame(self, indices: tuple[int, ...]) -> np.ndarray:
        parent = self.parent_reader.read_frame(indices)
        out = self._fn(np.asarray(parent))
        if out.dtype != self.data.dtype:
            out = out.astype(self.data.dtype, copy=False)
        return out

    def sum_points(self, points, out_dtype=np.float32):
        """Sum transformed frames for a region.

        Both of these transforms are LINEAR (rebin sums blocks; a crop selects a
        sub-array), so summing the PARENT's frames first and transforming once is
        identical to transforming each frame and summing — and it does the
        transform once instead of N times. Measured on a 16x16 ROI of 512^2
        frames: 529 ms per-frame vs ~60 ms via the parent's own block sum.

        Falls back to the per-frame loop if the parent can't sum for us."""
        parent_sum = getattr(self.parent_reader, "sum_points", None)
        if parent_sum is not None:
            try:
                total = parent_sum(points, out_dtype)
                if total is not None:
                    return self._fn(total)
            except Exception as e:
                log.debug("parent sum_points failed, per-frame fallback: %s", e)

        acc = None
        for p in points:
            idx = tuple(int(v) for v in np.asarray(p)[:self._nav_ndim])
            frame = self.read_frame(idx)
            if acc is None:
                acc = np.asarray(frame, dtype=out_dtype).copy()
            else:
                acc += frame
        return acc

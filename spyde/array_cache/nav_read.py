"""Drop-in replacement for _NavChunkCache's call pattern.

Both spyde/drawing/update_functions.py (_direct_read_frame, the base
navigator's synchronous single-point read) and spyde/actions/overlay.py (the
MDI live-overlay layer sync, which reads the SAME derived-view frame on a
source plot) used to call a plot's ``_nav_chunk_cache`` directly. That's now
routed through here instead, so both call sites get the locality gate (an
opaque signal-tree node — anything not tagged local — falls through to a
plain compute, exactly like _NavChunkCache.get_frame returning None for a
region it couldn't serve) without either file needing its own copy of the
locality check.

Kept as plain functions rather than a method on ArrayCache itself because the
locality check needs the owning Plot's signal_tree, which ArrayCache stays
deliberately agnostic of (it doesn't know about signals or trees, only
frames and readers).
"""
from __future__ import annotations

import logging

import numpy as np

from .resolve import resolve_reader

log = logging.getLogger(__name__)


def _reader_for(plot, signal, data):
    """Resolve (and cache on the plot, keyed by id(signal)) the best
    FrameReader for this signal's data — reusing it across calls is what
    makes a reader's internal state (e.g. LocalTransformReader's one-chunk
    memo, BinaryReader's open file descriptor) useful instead of rebuilt
    every read.

    A SUPERSEDED reader (its signal's ``data`` was swapped) is dropped, NOT
    closed here: this runs on the dispatcher thread AND on overlay.py's
    off-thread warm, so closing would risk yanking a file descriptor out from
    under an in-flight ``os.pread`` on the other thread (EBADF at best, a read
    from a recycled fd — i.e. another file's bytes — at worst). BinaryReader's
    ``__del__`` releases the fd once the last in-flight user drops it; the
    explicit close stays in :func:`close_all_readers`, which only runs on node
    switch / plot close."""
    readers = plot._local_transform_readers
    key = id(signal)
    reader = readers.get(key)
    if reader is not None and reader.data is data:
        return reader
    if reader is not None:
        # The signal's data was swapped in place (progressive nav fill, a result
        # window's computed data landing) — frames decoded from the OLD array are
        # stale, and ArrayCache keys on the signal's identity, not its data's.
        plot._array_cache.drop_key(key)
        block_cache = getattr(plot, "_block_cache", None)
        if block_cache is not None and reader is not None:
            block_cache.drop_owner(id(reader))
    reader = _try_per_frame_reader(plot, signal, data) or resolve_reader(
        signal, data, block_cache=getattr(plot, "_block_cache", None))
    if reader is not None:
        readers[key] = reader
    return reader


def _try_per_frame_reader(plot, signal, data):
    """A PerFrameReader for a derived view whose frames are a per-frame numpy
    function of the PARENT's frames (rebin, signal-space crop), or None.

    This is the big win for derived views: asking dask for one rebinned frame
    materialises the whole enclosing source chunk and re-runs the graph (measured
    2403 ms on a 537 MB-chunk .zspy), while reading the parent's frame through
    the PARENT's reader and rebinning in numpy is ~1.8 ms once the parent's block
    is cached. See readers/per_frame.py.

    Lives here rather than in resolve_reader because it needs the signal TREE (to
    find the parent node and its recorded transform args), which the array_cache
    package otherwise stays agnostic of. Returns None on anything unexpected —
    LocalTransformReader then serves it correctly, just slower."""
    tree = getattr(plot, "signal_tree", None)
    if tree is None:
        return None
    try:
        from .readers.per_frame import build_per_frame_reader

        node = tree.get_node(signal)
        parent = getattr(node, "parent", None) if node is not None else None
        parent_signal = getattr(parent, "signal", None)
        if parent_signal is None or getattr(parent_signal, "data", None) is None:
            return None
        # Resolve the PARENT's reader through the normal path so it gets the
        # plot's BlockCache — that cache is what makes the repeat reads cheap.
        parent_reader = _reader_for(plot, parent_signal, parent_signal.data)
        if parent_reader is None:
            return None
        return build_per_frame_reader(signal, data, node, parent_reader,
                                      parent_signal)
    except Exception as e:
        log.debug("per-frame reader resolve failed, using the dask view: %s", e)
        return None


def _resolve_for(plot, signal, data):
    """The reader serving this (signal, data), or None if the signal isn't
    ArrayCache-eligible (an opaque signal-tree node, or no signal_tree).

    An EXISTING reader is itself proof the locality gate already passed — only a
    local-resolved signal ever gets one. That keeps BaseSignalTree.resolve_locality
    (which walks the tree to find the node) off the per-frame path: it runs once
    per view, not once per move."""
    reader = plot._local_transform_readers.get(id(signal))
    if reader is not None and getattr(reader, "data", None) is data:
        return reader
    tree = getattr(plot, "signal_tree", None)
    if tree is None or not tree.resolve_locality(signal):
        return None
    return _reader_for(plot, signal, data)


def get_local_frame(plot, signal, data, indices, prof=None):
    """Return the decoded frame at ``indices`` via the plot's ArrayCache, or
    None if this signal isn't ArrayCache-eligible right now (an opaque
    signal-tree node, or no signal_tree available) — the caller falls back to a
    plain compute.

    ``indices`` is either a single nav point (``ndim <= 1``) or an INTEGRATING
    REGION of N points (``ndim > 1``), in which case the frame-wise mean is
    returned. A region used to bail out here, which is why region integration
    never touched any of this machinery and re-materialised a whole nav-chunk
    per point through dask — 64 chunk decodes to read the ~4 chunks an 8x8 ROI
    actually spans. Serving it through the same readers makes a drag cheap: the
    points share blocks, and the blocks are cached."""
    idx = np.asarray(indices)
    if idx.ndim > 1:
        return _get_local_region(plot, signal, data, idx, prof)
    point = tuple(int(v) for v in np.atleast_1d(idx))

    reader = _resolve_for(plot, signal, data)
    if reader is None:
        return None
    return plot._array_cache.get_frame(id(signal), reader, point, prof)


def _region_accum_dtype(dtype, n_pts):
    """Accumulator dtype for summing ``n_pts`` frames of ``dtype``.

    float32 is the fast choice (measured ~40% quicker than float64 on a 512^2
    frame) but it only holds 24 bits of integer precision, so it is EXACT only
    while ``n_pts * max(dtype)`` stays under 2^24. That is true for the common
    detector dtypes — 256 x uint16 max is 16,776,960, just under 16,777,216 —
    and NOT true for int32/uint32 or float64 data, which .hspy/.zspy can hold:
    summing 256 int32 frames in float32 was measured to lose up to 236,148 in
    absolute value. Anything else gets float64.

    NB the uint16 margin is only 256 counts, i.e. exactly the frame budget. If
    MAX_REGION_EXTENT_PER_DIM ever rises above 16, uint16 stops being exact too
    and lands here — which is why this decides from n_pts rather than a
    hard-coded dtype list."""
    dt = np.dtype(dtype)
    if dt == np.float64 or dt.itemsize > 4:
        return np.float64
    if np.issubdtype(dt, np.floating):
        # float16/float32 sources: float32 accumulate keeps the source's own
        # precision (a float64 accumulator cannot recover bits the data lacks).
        return np.float32
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
        span = max(abs(int(info.max)), abs(int(info.min)))
        if int(n_pts) * span <= (1 << 24):
            return np.float32
    return np.float64


def _get_local_region(plot, signal, data, idx, prof=None):
    """Frame-wise mean over the N nav points of an integrating region.

    Reads point-by-point through the SAME reader (and therefore the same
    ArrayCache/BlockCache) the single-point path uses, so an 8x8 ROI touches
    only the handful of nav-chunks it spans and a dragged ROI re-serves the
    ~75% of points it shares with the previous step from cache.

    The accumulator dtype is chosen by :func:`_region_accum_dtype` — float32 only
    where it is provably exact, float64 otherwise. Rounds back to an integer
    source dtype for parity with the distributed
    ``weighted_mean_round_from_sums`` the old path used.

    Memory stays bounded by construction: one frame at a time into one
    accumulator, never an N-frame stack (the Memory-Safety rule)."""
    reader = _resolve_for(plot, signal, data)
    if reader is None:
        return None
    key = id(signal)
    cache = plot._array_cache
    n_pts = int(idx.shape[0])
    if n_pts == 0:
        return None

    # FAST PATH: a reader that owns decoded blocks can sum the points directly
    # out of them. The per-frame path below has to COPY each frame out of its
    # block (so a cached frame doesn't pin the whole block) — for a region that
    # copy dominates: 165 ms vs 61 ms on a resident 537 MB block, 16x16 ROI.
    # Region frames are summed immediately and never wanted individually, so the
    # copy buys nothing here.
    acc_dtype = _region_accum_dtype(data.dtype, n_pts)
    acc = None
    how = "region"
    summer = getattr(reader, "sum_points", None)
    if summer is not None:
        try:
            acc = summer(idx, acc_dtype)
            if acc is not None:
                how = "region-block"
        except Exception as e:
            log.debug("block region sum failed, per-frame fallback: %s", e)
            acc = None

    if acc is None:
        # Per-frame fallback: contiguous sources, unchunked arrays, BinaryReader
        # (which reads exactly the frames asked for and has no block to sum from).
        for i in range(n_pts):
            point = tuple(int(v) for v in idx[i])
            frame = cache.get_frame(key, reader, point, None)
            if acc is None:
                acc = np.asarray(frame, dtype=acc_dtype).copy()
            else:
                acc += frame

    acc = acc / n_pts
    if np.issubdtype(data.dtype, np.integer):
        acc = np.rint(acc).astype(data.dtype)
    if prof is not None:
        prof.done(f"array-cache {how} x{n_pts}")
    return acc


def close_all_readers(plot) -> None:
    """Close every cached reader on ``plot`` (releasing e.g. BinaryReader's
    open file descriptor) before dropping them — called on node switch and
    on Plot.close(), mirroring where _array_cache.clear() is already called."""
    readers = getattr(plot, "_local_transform_readers", None)
    if not readers:
        return
    for reader in readers.values():
        close = getattr(reader, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
    readers.clear()


def is_local_frame_resident(plot, signal, data, indices) -> bool:
    """Side-effect-free hit probe: would reading this frame be a ~0 ms numpy
    slice? Mirrors _NavChunkCache.is_resident's contract for overlay.py's
    cheap/expensive read classification.

    Two ways to be resident, and BOTH matter. ArrayCache answers only for
    frames it has ALREADY returned; the old _NavChunkCache answered at CHUNK
    granularity, so a new position inside an already-decoded chunk counted as
    cheap (it is — the decode is what costs, the slice is free). Asking only
    ArrayCache would misclassify every new position in a warm chunk as an
    expensive cold read, making an overlay layer skip + re-warm on every single
    move. So a reader that keeps its own decoded-chunk memo (LocalTransformReader)
    gets to answer too, via the optional ``is_chunk_resident`` hook.

    Probes ONLY an EXISTING reader — it never resolves a new one, since that can
    allocate real resources (BinaryReader opens a file descriptor). It also
    doesn't re-check locality: a cache entry or a reader for this signal only
    exists because :func:`get_local_frame` already resolved it local (and the
    reader holds a strong reference to the signal, so its ``id`` can't be
    recycled underneath the cache key while it's there)."""
    try:
        idx = np.asarray(indices)
        if idx.ndim > 1:
            return False
        point = tuple(int(v) for v in np.atleast_1d(idx))
        if plot._array_cache.is_resident(id(signal), point):
            return True
        reader = plot._local_transform_readers.get(id(signal))
        if reader is None or getattr(reader, "data", None) is not data:
            return False
        probe = getattr(reader, "is_chunk_resident", None)
        return bool(probe(point)) if probe is not None else False
    except Exception:
        return False

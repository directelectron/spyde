"""Reader kinds 2 + 3: zarr+blosc (``.zspy``) and HDF5 (``.hspy``) — served by
ONE reader, because hyperspy hands both to dask the same way.

A lazy ``.hspy``/``.zspy`` signal's array is ``da.from_array(source)``, which
builds a two-layer graph: the output layer ``array-<token>`` (getter tasks) and
a sibling ``original-array-<token>`` layer whose single value IS the live source
object — an ``h5py.Dataset`` or a ``zarr.core.Array``. Both support numpy-style
integer indexing that reads ONLY the storage chunks intersecting the request, so
the store can be read without dask in the loop at all.

Measured on a 32x32x128x128 uint16 scan stored in 8x8-nav chunks (warm page
cache, per frame, frames spread one-per-chunk vs dwelling inside one chunk):

                                        cold chunk    dwell in chunk
    .zspy  per-frame from the store       0.97 ms        0.79 ms
           nav-chunk via dask (kind 4)    1.68 ms        0.009 ms
           nav-chunk from the store       0.84 ms        0.002 ms   <- this reader
    .hspy  per-frame from the store       5.12 ms        0.003 ms
           nav-chunk via dask (kind 4)    5.97 ms        0.009 ms
           nav-chunk from the store       5.10 ms        0.002 ms   <- this reader

Neither of the first two rows wins alone: reading per frame skips dask's graph
overhead on a cold chunk, but zarr re-decompresses the enclosing chunk for EVERY
frame, so dwelling inside one chunk — a 4D-STEM DP navigator's dominant pattern
— is ~88x worse than slicing an already-decoded block. So this reader does both:
for a CHUNKED source it reads the whole nav-chunk DIRECTLY from the store
(cheaper than dask's route to the same block) and memoizes it, so a cold chunk
pays the faster fill and dwell is a free numpy slice. For a CONTIGUOUS
(unchunked) source, one frame is an exact hyperslab read, so there is nothing to
amortise and it reads the frame alone.

Correctness gate — the same trap the binary reader hit: a dask HighLevelGraph
keeps every ancestor layer, so a DERIVED view still carries the source's
``original-array-*`` layer. Reading through it would silently return the
UNTRANSFORMED source frame. So the array must BE the ``from_array`` output
(``data.name`` starts with ``array-``) and the source's shape/dtype must match
it exactly; anything else declines to LocalTransformReader.

Plain in-RAM ``np.ndarray`` sources are declined too — caching a copy of data
that is already fully resident is pure duplication (``np.memmap`` IS accepted:
it is file-backed, so a cached frame saves re-faulting its pages).

Thread safety: h5py serialises reads on its own global lock and zarr reads are
safe concurrently, so the occasional off-dispatcher read from overlay.py's
source warm is fine here.
"""
from __future__ import annotations

import numpy as np


def find_source_array(data):
    """Return the live source object (``h5py.Dataset`` / ``zarr.core.Array`` /
    ``np.memmap``) that ``data`` directly wraps, or None if ``data`` is not such
    a wrap (a derived view, a computed array, or a plain in-RAM ndarray)."""
    name = getattr(data, "name", None)
    graph = getattr(data, "dask", None)
    if graph is None or not isinstance(name, str) or not name.startswith("array-"):
        return None
    try:
        layers = graph.layers
        source = None
        for lname, layer in layers.items():
            # da.from_array names the source layer "original-" + the output name.
            if lname == f"original-{name}":
                values = list(layer.values())
                if len(values) != 1:
                    return None
                source = values[0]
                break
        if source is None:
            return None
        if isinstance(source, np.ndarray) and not isinstance(source, np.memmap):
            return None  # already in RAM — nothing to cache
        if tuple(int(v) for v in getattr(source, "shape", ())) != \
                tuple(int(v) for v in data.shape):
            return None
        if np.dtype(getattr(source, "dtype", None)) != data.dtype:
            return None
        if not hasattr(source, "__getitem__"):
            return None
        return source
    except Exception:
        return None


# A COMPRESSED store's chunk is ATOMIC: you cannot read less than one.
#
# Measured on a real .zspy with 537 MB (32x32 nav x 512^2 uint16) chunks:
#
#     whole chunk 32x32   537 MB : 437 ms
#     sub-block    8x16    67 MB : 445 ms   (102% of the whole chunk)
#     ONE frame           0.5 MB : 406 ms   ( 93% of the whole chunk)
#
# Reading one frame costs the SAME as reading all 1024. So the old fallback
# ("chunk too big to cache -> read single frames") was a ~100x bug on compressed
# data: a ~50-frame integrating ROI paid ~50 whole-chunk decodes. In the app that
# was 2100-4250 ms (0.01 GB/s) for a region MRC served in 50-100 ms (0.74-1.27
# GB/s). Sub-blocking is equally wrong — same decode, 1/8 the payload cached.
#
# So for a chunked store we ALWAYS cache the whole nav-chunk, and the budget must
# be big enough to hold the few chunks an ROI spans. A 16x16 ROI touches at most
# 4 chunks; the BlockCache LRU bounds the total (CLAUDE.md Memory-Safety: this is
# a handful of chunks, never the dataset).
#
# The one case still read frame-at-a-time is an UNCHUNKED (contiguous) source,
# where a frame really is an exact hyperslab and nothing is amortised.
MAX_BLOCK_BYTES = 1 << 30            # 1 GiB — hold a chunk even when it is large


def _sum_block_points(block, nav_slices, pts, nav_ndim, out_dtype):
    """Sum the frames at ``pts`` out of a decoded ``block``.

    An integrating ROI is a RECTANGLE, so its points inside one block are almost
    always a dense sub-rectangle. Summing that as a plain slice with
    ``sum(axis=nav_axes)`` avoids materialising an intermediate — which is the
    whole cost here. Measured on a resident 537 MB block, 16x16 ROI:

        fancy-index then sum          141 ms   (copies 134 MB first)
        slice + reshape then sum      124 ms   (reshape also copies)
        slice, sum over the nav axes   67 ms   <- no intermediate

    Falls back to fancy indexing for a non-rectangular point set (which the
    selectors don't currently produce, but nothing here depends on that)."""
    local = [np.fromiter((p[ax] - nav_slices[ax].start for p in pts),
                         dtype=np.intp, count=len(pts))
             for ax in range(nav_ndim)]
    lo = [int(a.min()) for a in local]
    hi = [int(a.max()) + 1 for a in local]
    # Dense == the bounding box has exactly as many cells as we have points AND
    # the points are distinct (duplicates would make the counts match while the
    # slice silently summed different frames).
    extent = int(np.prod([h - l for l, h in zip(lo, hi)]))
    dense = extent == len(pts) and len(
        set(zip(*(a.tolist() for a in local)))) == len(pts)
    if dense:
        sl = tuple(slice(l, h) for l, h in zip(lo, hi))
        return block[sl].sum(axis=tuple(range(nav_ndim)), dtype=out_dtype)
    return block[tuple(local)].sum(axis=0, dtype=out_dtype)


class SourceArrayReader:
    """One instance per (signal, data) pair. Holds only a reference to the
    already-open source (hyperspy keeps the h5py file / zarr store open for the
    lifetime of a lazy signal), so there is nothing to close.

    Decoded blocks live in the plot's shared :class:`BlockCache` (a byte-budgeted
    LRU), NOT in a one-entry memo. A single slot could only serve a dwell inside
    one chunk: crossing a boundary and returning re-decompressed the chunk every
    move (59x on zspy), and an integrating region — which spans up to 4 nav-chunks
    — thrashed on every drag step. The LRU keeps a region's whole span plus recent
    history resident.

    A cache is optional (``block_cache=None`` falls back to a private one-entry
    memo) so a reader constructed outside a Plot still works.
    """

    def __init__(self, signal, data, source, block_cache=None):
        self.signal = signal
        self.data = data
        self.source = source
        self._nav_ndim = signal.axes_manager.navigation_dimension
        self._nav_shape = tuple(int(v) for v in data.shape[:self._nav_ndim])
        self._block_cache = block_cache
        self._memo = None                   # fallback when there's no BlockCache

        # Uniform per-axis STORAGE chunk sizes along the nav axes (h5py's
        # Dataset.chunks / zarr's Array.chunks; None for a contiguous dataset,
        # absent for a memmap).
        chunks = getattr(source, "chunks", None)
        nav_chunks = None
        if chunks is not None and len(chunks) >= self._nav_ndim:
            nav_chunks = self._fit_block(
                tuple(int(c) for c in chunks[:self._nav_ndim]))
        self._nav_chunks = nav_chunks

    def _fit_block(self, nav_chunks):
        """The nav extent to read and cache as one block — the WHOLE storage chunk
        for a chunked source, since a chunk is atomic (see the module note).

        Returns None (read frame-at-a-time) only when there is nothing to
        amortise: a 1-frame chunk, or a chunk so large that holding it would blow
        the budget outright, in which case per-frame reads at least bound memory."""
        total = int(np.prod(nav_chunks))
        if total <= 1:
            return None                      # 1 frame/chunk — nothing to amortise
        if total * self.frame_bytes > MAX_BLOCK_BYTES:
            return None                      # can't hold it; bound memory instead
        return tuple(nav_chunks)

    @property
    def frame_bytes(self) -> int:
        frame_shape = self.data.shape[self._nav_ndim:]
        return int(np.prod(frame_shape)) * self.data.dtype.itemsize

    def _locate(self, point):
        """(block_index_tuple, per-nav-axis slices) of the storage chunk holding
        ``point``, or None when this reader reads frame-at-a-time."""
        if self._nav_chunks is None:
            return None
        block_idx, nav_slices = [], []
        for ax in range(self._nav_ndim):
            c = self._nav_chunks[ax]
            b = point[ax] // c
            block_idx.append(b)
            nav_slices.append(slice(b * c, min((b + 1) * c, self._nav_shape[ax])))
        return tuple(block_idx), nav_slices

    def _cached_block(self, key):
        """The decoded block for ``key`` if resident, else None."""
        if self._block_cache is not None:
            return self._block_cache.get(id(self), key)
        memo = self._memo                   # read ONCE — torn pairs slice the wrong chunk
        return memo[1] if memo is not None and memo[0] == key else None

    def _store_block(self, key, block) -> None:
        if self._block_cache is not None:
            self._block_cache.put(id(self), key, block)
        else:
            self._memo = (key, block)

    def is_chunk_resident(self, indices) -> bool:
        """Is the decoded storage chunk holding ``indices`` already cached —
        i.e. would read_frame be a ~0 ms numpy slice? Side-effect-free; feeds
        overlay.py's cheap/expensive gate (see nav_read.is_local_frame_resident)."""
        try:
            point = tuple(int(v) for v in np.atleast_1d(np.asarray(indices)))
            located = self._locate(point)
            if located is None:
                return False
            if self._block_cache is not None:
                return self._block_cache.contains(id(self), located[0])
            memo = self._memo
            return memo is not None and located[0] == memo[0]
        except Exception:
            return False

    def sum_points(self, points, out_dtype=np.float32):
        """Sum the frames at ``points`` straight out of the resident block(s).

        The per-frame path (read_frame -> ArrayCache) has to COPY each frame out
        of its block so a cached entry doesn't pin the whole thing. For a REGION
        that copy is pure waste — the frames are summed immediately and never
        cached individually. Measured on a resident 537 MB block, 16x16 ROI:

            per-frame loop + copy   165 ms
            loop without the copy    61 ms
            vectorised block sum     99 ms

        so skipping the copy is the win, and slicing contiguous runs out of the
        block lets numpy do the accumulate. Returns None if this reader has no
        block concept (contiguous source), so the caller falls back.

        Groups points by block, so an ROI straddling several chunks still works;
        each block is fetched once through the same cache the single-point path
        uses (no extra decode, no extra residency)."""
        if self._nav_chunks is None:
            return None
        groups: dict = {}
        for p in points:
            point = tuple(int(v) for v in p[:self._nav_ndim])
            located = self._locate(point)
            if located is None:
                return None
            key, nav_slices = located
            groups.setdefault(key, (nav_slices, []))[1].append(point)

        acc = None
        for key, (nav_slices, pts) in groups.items():
            block = self._cached_block(key)
            if block is None:
                block = np.asarray(self.source[tuple(nav_slices)])
                self._store_block(key, block)
            part = _sum_block_points(block, nav_slices, pts, self._nav_ndim,
                                     out_dtype)
            acc = part if acc is None else acc + part
        return acc

    def read_frame(self, indices: tuple[int, ...]) -> np.ndarray:
        point = tuple(int(v) for v in indices[:self._nav_ndim])
        located = self._locate(point)
        if located is None:
            # Contiguous source (or a chunk too big / too small to amortise):
            # one frame is an exact read.
            frame = np.asarray(self.source[point])
        else:
            key, nav_slices = located
            block = self._cached_block(key)
            if block is None:
                block = np.asarray(self.source[tuple(nav_slices)])
                self._store_block(key, block)
            local = tuple(point[ax] - nav_slices[ax].start
                          for ax in range(self._nav_ndim))
            frame = block[local]
        if frame.base is not None:
            # A slice of the memo block (or of a memmap) is a VIEW onto a much
            # bigger buffer — copy so a cached entry retains exactly one frame
            # and ArrayCache's byte accounting stays honest.
            frame = frame.copy()
        return frame

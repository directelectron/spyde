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


# Don't memoize a nav-chunk bigger than this — read the single frame instead.
# A storage chunk spanning 32x32 nav positions of 256^2 uint16 frames is 128 MB;
# holding that per plot (on top of ArrayCache) for a dwell win isn't a trade
# worth making, and for a chunked store a single-frame read is still correct,
# just without the dwell amortisation (CLAUDE.md Memory-Safety rule).
MAX_BLOCK_BYTES = 64 << 20


class SourceArrayReader:
    """One instance per (signal, data) pair. Holds only a reference to the
    already-open source (hyperspy keeps the h5py file / zarr store open for the
    lifetime of a lazy signal), so there is nothing to close.

    Like LocalTransformReader, the decoded-block memo is ONE ``(key, block)``
    tuple so overlay.py's off-thread source warm can't observe a torn pair and
    slice a frame out of the wrong chunk.
    """

    def __init__(self, signal, data, source):
        self.signal = signal
        self.data = data
        self.source = source
        self._nav_ndim = signal.axes_manager.navigation_dimension
        self._nav_shape = tuple(int(v) for v in data.shape[:self._nav_ndim])
        self._memo = None                   # (block_index_tuple, block ndarray)

        # Uniform per-axis STORAGE chunk sizes along the nav axes (h5py's
        # Dataset.chunks / zarr's Array.chunks; None for a contiguous dataset,
        # absent for a memmap). The block read is worth it only when the store
        # decodes a whole chunk anyway.
        chunks = getattr(source, "chunks", None)
        nav_chunks = None
        if chunks is not None and len(chunks) >= self._nav_ndim:
            nav_chunks = tuple(int(c) for c in chunks[:self._nav_ndim])
            block_frames = int(np.prod(nav_chunks))
            if block_frames <= 1 or block_frames * self.frame_bytes > MAX_BLOCK_BYTES:
                nav_chunks = None           # nothing to amortise / too big to hold
        self._nav_chunks = nav_chunks

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

    def is_chunk_resident(self, indices) -> bool:
        """Is the decoded storage chunk holding ``indices`` already memoized —
        i.e. would read_frame be a ~0 ms numpy slice? Side-effect-free; feeds
        overlay.py's cheap/expensive gate (see nav_read.is_local_frame_resident)."""
        try:
            memo = self._memo
            if memo is None:
                return False
            point = tuple(int(v) for v in np.atleast_1d(np.asarray(indices)))
            located = self._locate(point)
            return located is not None and located[0] == memo[0]
        except Exception:
            return False

    def read_frame(self, indices: tuple[int, ...]) -> np.ndarray:
        point = tuple(int(v) for v in indices[:self._nav_ndim])
        located = self._locate(point)
        if located is None:
            # Contiguous source (or a chunk too big / too small to amortise):
            # one frame is an exact read.
            frame = np.asarray(self.source[point])
        else:
            key, nav_slices = located
            memo = self._memo               # read ONCE — see the class docstring
            if memo is not None and memo[0] == key:
                block = memo[1]
            else:
                block = np.asarray(self.source[tuple(nav_slices)])
                self._memo = (key, block)
            local = tuple(point[ax] - nav_slices[ax].start
                          for ax in range(self._nav_ndim))
            frame = block[local]
        if frame.base is not None:
            # A slice of the memo block (or of a memmap) is a VIEW onto a much
            # bigger buffer — copy so a cached entry retains exactly one frame
            # and ArrayCache's byte accounting stays honest.
            frame = frame.copy()
        return frame

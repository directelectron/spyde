"""Reader kind 4: signal-tree local-transform (derived lazy views).

Wraps the same lazy-slice construction the async (expensive) tier already
uses (spyde.drawing.update_functions._build_nav_lazy_slice), but computes
synchronously for a single frame — generalizing what _NavChunkCache's miss
path did, minus the whole-chunk caching (the outer ArrayCache now handles
cross-call caching at frame granularity instead).

A chunked derived view still forces dask to materialise the WHOLE source
nav-chunk to yield any one frame — that's inherent to how dask chunking
works, a getitem task depends on its enclosing chunk, not something this
reader can avoid. To keep scrubbing through several NEW frames in the same
chunk cheap before ArrayCache has seen them individually, this reader keeps
a tiny one-chunk memo of its own — a private implementation detail, not
exposed to ArrayCache or its callers.
"""
from __future__ import annotations

import numpy as np


def nav_chunk_span(chunk_sizes, pos):
    """For one navigation axis with dask ``chunk_sizes`` (a tuple of per-block
    lengths) and integer ``pos``, return ``(block_index, start, stop)`` of the
    chunk that contains ``pos``. cumsum + searchsorted; O(number of blocks)."""
    edges = np.cumsum((0,) + tuple(int(c) for c in chunk_sizes))
    b = int(np.searchsorted(edges, pos, side="right") - 1)
    b = max(0, min(b, len(chunk_sizes) - 1))
    return b, int(edges[b]), int(edges[b + 1])


class LocalTransformReader:
    """One instance per (signal, data) pair. Cheap to construct — holds only
    references and a one-chunk memo, no I/O happens until read_frame().

    The memo is stored as ONE ``(key, block)`` tuple, deliberately not two
    attributes: overlay.py's off-thread source warm can call read_frame
    concurrently with the dispatcher's own read, and a torn key/block pair
    would slice a frame out of the WRONG chunk (a silent wrong-frame bug, or an
    IndexError on a ragged last chunk). A single assignment is atomic under the
    GIL, so a concurrent reader sees either the old pair or the new one.
    """

    def __init__(self, signal, data):
        self.signal = signal
        self.data = data
        self._nav_ndim = signal.axes_manager.navigation_dimension
        self._memo = None                   # (block_index_tuple, block ndarray)

    @property
    def frame_bytes(self) -> int:
        frame_shape = self.data.shape[self._nav_ndim:]
        return int(np.prod(frame_shape)) * self.data.dtype.itemsize

    def _locate(self, indices):
        """(block_index_tuple, per-nav-axis slices) for the chunk holding
        ``indices`` — or None if this array isn't dask-chunked."""
        chunks = getattr(self.data, "chunks", None)
        if not chunks:
            return None
        block_idx, nav_slices = [], []
        for ax in range(self._nav_ndim):
            b, s, e = nav_chunk_span(chunks[ax], indices[ax])
            block_idx.append(b)
            nav_slices.append(slice(s, e))
        return tuple(block_idx), nav_slices

    def is_chunk_resident(self, indices) -> bool:
        """Is the decoded chunk holding ``indices`` already memoized — i.e.
        would read_frame be a ~0 ms numpy slice? Side-effect-free. This is the
        CHUNK-granularity residency the overlay's cheap/expensive gate needs:
        ArrayCache alone can only answer for frames it has already returned, so
        without this every NEW position in an already-decoded chunk would be
        misclassified as an expensive cold read."""
        try:
            memo = self._memo
            if memo is None:
                return False
            located = self._locate(indices)
            return located is not None and located[0] == memo[0]
        except Exception:
            return False

    def read_frame(self, indices: tuple[int, ...]) -> np.ndarray:
        located = self._locate(indices)
        if located is None:
            # Not dask-chunked (shouldn't happen for a lazy signal) — plain compute.
            return np.asarray(self.data[indices].compute(scheduler="synchronous"))
        key, nav_slices = located

        memo = self._memo                   # read ONCE — see the class docstring
        if memo is not None and memo[0] == key:
            block = memo[1]
        else:
            full_slice = tuple(nav_slices) + (slice(None),) * (self.data.ndim - self._nav_ndim)
            block = np.asarray(self.data[full_slice].compute(scheduler="synchronous"))
            self._memo = (key, block)

        local = tuple(indices[ax] - nav_slices[ax].start for ax in range(self._nav_ndim))
        frame = block[local]
        if frame.base is not None and frame.nbytes < block.nbytes:
            # COPY out of the block. ``block[local]`` is a VIEW, so caching it
            # would keep the ENTIRE decoded chunk alive while ArrayCache only
            # accounts one frame — 10 frames from each of 100 chunks would
            # account as ~32 MB while retaining ~3.2 GB (the old
            # _NavChunkCache accounted per-block, so views were sound there).
            # A frame-sized memcpy is µs-to-~1 ms; honest accounting is worth it.
            frame = frame.copy()
        return frame

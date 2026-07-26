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
    references, no I/O happens until read_frame().

    Decoded blocks live in the plot's shared :class:`BlockCache` (a byte-budgeted
    LRU). A one-entry memo — what this used to hold — only served a dwell inside
    one chunk: a derived view must re-decode AND re-run the transform over the
    whole source nav-chunk to yield any one frame, so ping-ponging across a
    boundary cost 217 ms/move instead of 0.14 ms (1520x), and an integrating
    region spanning several chunks thrashed on every drag step.

    A cache is optional (``block_cache=None`` falls back to a private one-entry
    memo, stored as ONE tuple so a concurrent reader can't observe a torn
    key/block pair and slice a frame out of the WRONG chunk).
    """

    def __init__(self, signal, data, block_cache=None):
        self.signal = signal
        self.data = data
        self._nav_ndim = signal.axes_manager.navigation_dimension
        self._block_cache = block_cache
        self._memo = None                   # fallback when there's no BlockCache

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

    def _cached_block(self, key):
        """The decoded block for ``key`` if resident, else None."""
        if self._block_cache is not None:
            return self._block_cache.get(id(self), key)
        memo = self._memo                   # read ONCE — see the class docstring
        return memo[1] if memo is not None and memo[0] == key else None

    def _store_block(self, key, block) -> None:
        if self._block_cache is not None:
            self._block_cache.put(id(self), key, block)
        else:
            self._memo = (key, block)

    def is_chunk_resident(self, indices) -> bool:
        """Is the decoded chunk holding ``indices`` already cached — i.e.
        would read_frame be a ~0 ms numpy slice? Side-effect-free. This is the
        CHUNK-granularity residency the overlay's cheap/expensive gate needs:
        ArrayCache alone can only answer for frames it has already returned, so
        without this every NEW position in an already-decoded chunk would be
        misclassified as an expensive cold read."""
        try:
            located = self._locate(indices)
            if located is None:
                return False
            if self._block_cache is not None:
                return self._block_cache.contains(id(self), located[0])
            memo = self._memo
            return memo is not None and located[0] == memo[0]
        except Exception:
            return False

    def sum_points(self, points, out_dtype=np.float32):
        """Sum the frames at ``points`` straight out of the resident block(s) —
        see SourceArrayReader.sum_points for why (skips the per-frame copy the
        cached single-point path needs). Returns None when this array isn't
        dask-chunked, so the caller falls back to the per-frame loop."""
        chunks = getattr(self.data, "chunks", None)
        if not chunks:
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
                full = tuple(nav_slices) + \
                    (slice(None),) * (self.data.ndim - self._nav_ndim)
                block = np.asarray(
                    self.data[full].compute(scheduler="synchronous"))
                self._store_block(key, block)
            from .source_array import _sum_block_points
            part = _sum_block_points(block, nav_slices, pts, self._nav_ndim,
                                     out_dtype)
            acc = part if acc is None else acc + part
        return acc

    def read_frame(self, indices: tuple[int, ...]) -> np.ndarray:
        located = self._locate(indices)
        if located is None:
            # Not dask-chunked (shouldn't happen for a lazy signal) — plain compute.
            return np.asarray(self.data[indices].compute(scheduler="synchronous"))
        key, nav_slices = located

        block = self._cached_block(key)
        if block is None:
            full_slice = tuple(nav_slices) + (slice(None),) * (self.data.ndim - self._nav_ndim)
            block = np.asarray(self.data[full_slice].compute(scheduler="synchronous"))
            self._store_block(key, block)

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

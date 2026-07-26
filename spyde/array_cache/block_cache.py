"""BlockCache — a byte-budgeted LRU of DECODED nav-chunk blocks, shared by every
reader kind that has a chunk concept (zarr/HDF5 via SourceArrayReader, derived
dask views via LocalTransformReader).

Why this exists (the bug it replaces): both readers used to hold ONE
``(key, block)`` tuple as their memo. That is enough for dwelling inside a
single chunk, and nothing else. Two access patterns SpyDE has constantly break
it:

  * **crossing a chunk boundary and coming back** — a scrub that ping-pongs
    between two chunks re-decodes on every move. Measured on a 48x48x256^2
    zspy: 0.14 ms served vs 9.3 ms re-decompressed (59x), and on a rebinned
    (derived) view 0.14 ms vs 217 ms (1520x — a derived view must re-run the
    transform over the whole source chunk to yield one frame).
  * **an integrating REGION** — an 8x8 ROI spans up to 4 nav-chunks, so a
    one-entry memo thrashes on every single step of a drag. This is what made
    region integration ~64 chunk decodes to read 4 chunks' worth of data.

A modest LRU fixes both: at the default budget a 4D-STEM's 33.6 MB nav-chunks
give ~30 resident blocks, so a region (<=4) and a wandering drag both stay warm
and eviction is rare during normal work.

Granularity note (the design principle this serves): read granularity should
match ACCESS granularity. A frame-granular backing (BinaryReader's pread, and
eventually sub-chunked zspy) does NOT use this cache at all — it reads exactly
the frames asked for. This cache is the fallback for backings where the store
must decode a whole chunk anyway (non-sub-chunked zarr/HDF5, any dask-chunked
derived view); there, caching the decoded block is what makes per-frame access
cheap. When sub-chunked zspy lands, that reader simply stops populating this.

Threading: ``ArrayCache``'s note applies here too. spyde/actions/overlay.py
warms frames from a compute-backend worker thread while the ``_NavDispatcher``
reads on its own thread, so the bookkeeping IS locked. The lock covers ONLY the
dict bookkeeping, NEVER the decode — holding a lock across a compute is the
retired ``_cache_lock_ctx`` mistake that wedged the navigator (Live-Display
Section 2). Two threads may both miss and both decode the same block; that is
harmless, last insert wins.

Cached blocks are SHARED and must be treated as READ-ONLY — readers slice a
frame out and copy it, never mutate the block.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np

# PER-PLOT budget for decoded blocks. It must hold the few STORAGE CHUNKS an ROI
# spans, and a chunk can be big: a real .zspy 4D-STEM with 32x32 nav chunks of
# 512^2 uint16 frames is 537 MB PER CHUNK, and a 16x16 ROI can straddle 4 of them
# (~2.1 GB). A compressed chunk is atomic — reading one frame costs the same as
# reading all 1024 (measured 406 ms vs 437 ms) — so the only way an ROI drag is
# not paying a full decode per frame is to keep those chunks resident.
# Smaller geometries (33.6 MB chunks) leave dozens of blocks resident here.
# The LRU still bounds this to a handful of chunks, never the dataset.
DEFAULT_BLOCK_BUDGET_BYTES = 3 << 30            # 3 GiB

# What an UNFOCUSED plot keeps. Not a purge: dropping everything would make
# clicking back to a window pay a full cold decode of the chunk you were just
# looking at. This keeps the working set (a chunk or three) and returns the rest,
# so N background windows cost ~100 MB each instead of ~1 GB each.
UNFOCUSED_BUDGET_BYTES = 100 << 20              # 100 MiB


class BlockCache:
    """Keyed by ``(owner_key, block_index)`` where ``owner_key`` identifies the
    reader/signal being served (readers pass ``id(self)``), so blocks decoded by
    a superseded reader can be dropped wholesale without disturbing others.
    """

    def __init__(self, budget_bytes: int = DEFAULT_BLOCK_BUDGET_BYTES) -> None:
        self._entries: "OrderedDict[tuple[Any, Any], np.ndarray]" = OrderedDict()
        self._nbytes = 0
        self._budget_bytes = int(budget_bytes)
        self._full_budget_bytes = int(budget_bytes)
        self._lock = threading.Lock()   # bookkeeping only — never held across a decode

    # -- lookup ------------------------------------------------------------

    def get(self, owner_key: Any, block_index: Any):
        """Return the decoded block, or None on a miss. LRU-touches on a hit."""
        key = (owner_key, block_index)
        with self._lock:
            block = self._entries.get(key)
            if block is not None:
                self._entries.move_to_end(key)
            return block

    def put(self, owner_key: Any, block_index: Any, block: np.ndarray) -> None:
        """Store a freshly decoded block, evicting LRU entries over budget."""
        key = (owner_key, block_index)
        nbytes = int(getattr(block, "nbytes", 0))
        with self._lock:
            prior = self._entries.pop(key, None)
            if prior is not None:
                self._nbytes -= int(getattr(prior, "nbytes", 0))
            self._entries[key] = block
            self._nbytes += nbytes
            self._evict_over_budget()

    def contains(self, owner_key: Any, block_index: Any) -> bool:
        """Side-effect-free residency probe — does NOT LRU-touch, so it is safe
        for the read classifier / overlay's cheap-vs-expensive gate to call
        without perturbing eviction order."""
        return (owner_key, block_index) in self._entries

    # -- lifecycle ---------------------------------------------------------

    def drop_owner(self, owner_key: Any) -> None:
        """Evict every block decoded by one reader (it was superseded — its
        signal's ``data`` was swapped in place, so its blocks are stale)."""
        with self._lock:
            stale = [k for k in self._entries if k[0] == owner_key]
            for k in stale:
                self._nbytes -= int(getattr(self._entries.pop(k), "nbytes", 0))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._nbytes = 0

    def set_budget(self, budget_bytes: int) -> None:
        """Change the budget, evicting immediately if the new one is smaller.
        Used for the focus demote/restore — NOT a purge (see the module docstring
        and ``UNFOCUSED_BUDGET_BYTES``)."""
        with self._lock:
            self._budget_bytes = int(budget_bytes)
            self._evict_over_budget()

    def demote(self) -> None:
        """Shrink to the unfocused budget (this plot lost focus)."""
        self.set_budget(min(self._full_budget_bytes, UNFOCUSED_BUDGET_BYTES))

    def restore(self) -> None:
        """Return to the full budget (this plot gained focus)."""
        self.set_budget(self._full_budget_bytes)

    # -- internals ---------------------------------------------------------

    def _evict_over_budget(self) -> None:
        # Caller holds _lock. The just-inserted entry is MRU (last) and
        # popitem(last=False) drops the LRU (first), so a block that alone
        # exceeds the budget still survives — matches ArrayCache's rule.
        while self._nbytes > self._budget_bytes and len(self._entries) > 1:
            _old, block = self._entries.popitem(last=False)
            self._nbytes -= int(getattr(block, "nbytes", 0))

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

"""ArrayCache — a byte-budget LRU of DECODED frame ndarrays, backing-agnostic
via the FrameReader protocol.

Generalizes _NavChunkCache (spyde/drawing/update_functions.py) in two ways:
the budget is in BYTES, not frame count (so it doesn't care whether the
source is a 32 KB diffraction pattern or a 32 MB 4k movie frame), and
eviction is per-FRAME, not per-CHUNK — a reader kind may have no chunk
concept at all (the binary/pread reader reads exactly one frame per call).

Almost always touched from the serial _NavDispatcher thread (Live-Display §2),
but NOT exclusively: spyde/actions/overlay.py's off-thread source warm
(``_warm_source_chunk``) populates this same cache from a compute-backend
worker thread so the next dispatcher read finds the frame resident. So the
bookkeeping IS locked — an OrderedDict's ``move_to_end``/``popitem`` and the
running ``_nbytes`` total are not safe to interleave across threads. The lock
covers ONLY the dict bookkeeping, NEVER the reader call: holding a lock across
a compute is exactly the retired ``_cache_lock_ctx`` mistake that wedged the
navigator (Live-Display §2). Two threads may therefore both miss and both
read the same frame — harmless, last insert wins.

Cached frames are SHARED, so consumers must treat them as READ-ONLY (the
binary reader's zero-copy pread frame is literally non-writeable); mutating a
frame in place would corrupt every later hit. One instance per Plot, same
ownership as the cache it replaces.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from .protocol import FrameReader

# PER-PLOT budget for SINGLE-POINT scrub history, so the real footprint is this ×
# the number of open plots — keep it modest. At 256 MiB that is ~8 frames of a
# 4096² uint16 movie (32 MB each) or ~8000 frames of a 128² uint16 diffraction
# pattern (32 KB each) — well past the point of diminishing returns for
# dwell-and-return scrubbing, and small enough that a few open windows can't
# quietly hold the whole dataset in RAM (the Memory-Safety rule in CLAUDE.md).
DEFAULT_BUDGET_BYTES = 256 << 20

# …but regions are ALSO served from here now (get_local_frame stopped declining
# them), and a region's working set is N frames AT ONCE, not a scrub history. On a
# 4096² uint16 movie a 16-frame ROI is 512 MiB against 8 slots — an LRU scan over a
# working set twice the cache, i.e. measured 0 hits / 16 misses on EVERY drag step
# (~166 ms/step of pure re-reading). ensure_budget_for grows the budget to hold the
# ROI, bounded by this ceiling so it can't follow a pathological region to
# arbitrary RAM. It never shrinks below DEFAULT_BUDGET_BYTES, and clear() puts it
# back (so a node switch doesn't inherit the last ROI's appetite).
REGION_BUDGET_CEILING_BYTES = 1 << 30

# The window alone is NOT enough headroom. A sliding ROI re-reads the LEAVING
# frame — the oldest one — while the incremental path touches only 2 frames per
# step, so the other N-2 age by one insert per step and the leaving frame is
# evicted at almost exactly the step it is next needed: measured 1.9 misses/step
# at N+1 slots versus 1.0 at N*1.5 (111 ms vs 59 ms on a 16-frame 4096² ROI).
REGION_BUDGET_HEADROOM = 1.5


class ArrayCache:
    """Keyed by ``(key, indices)`` where ``key`` is an opaque identity for the
    signal/reader currently being served (e.g. ``id(signal)``) — mirrors
    _NavChunkCache's ``id(signal)`` component so a node switch (a NEW signal
    object) misses naturally even if ``clear()`` is somehow skipped.
    """

    def __init__(self, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> None:
        self._entries: "OrderedDict[tuple[Any, tuple[int, ...]], np.ndarray]" = OrderedDict()
        self._nbytes = 0
        self._budget_bytes = int(budget_bytes)
        self._default_budget_bytes = int(budget_bytes)   # floor for ensure_budget_for
        self._lock = threading.Lock()      # bookkeeping only — never held across a read

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._nbytes = 0
            # A plot that grew to hold a big ROI must not keep that budget for
            # whatever node it shows next (ensure_budget_for re-grows on demand).
            self._budget_bytes = self._default_budget_bytes

    def get_frame(
        self,
        key: Any,
        reader: FrameReader,
        indices: tuple[int, ...],
        prof=None,
    ) -> np.ndarray:
        """Return the decoded frame at ``indices``, reading through ``reader``
        on a miss and caching the result."""
        cache_key = (key, tuple(int(v) for v in indices))
        with self._lock:
            frame = self._entries.get(cache_key)
            if frame is not None:
                self._entries.move_to_end(cache_key)  # LRU touch
        if frame is not None:
            if prof is not None:
                prof.done("array-cache hit")
            return frame

        # Read OUTSIDE the lock (it can be a real decode) — a concurrent miss on
        # the same frame just reads twice and the later insert wins.
        frame = np.asarray(reader.read_frame(cache_key[1]))
        with self._lock:
            prior = self._entries.pop(cache_key, None)
            if prior is not None:
                self._nbytes -= prior.nbytes
            self._entries[cache_key] = frame
            self._nbytes += frame.nbytes
            self._evict_over_budget()
        if prof is not None:
            prof.done("array-cache MISS read")
        return frame

    def ensure_budget_for(self, n_frames: int, frame_bytes: int) -> None:
        """Grow the budget so an ``n_frames``-point region's working set fits, up
        to :data:`REGION_BUDGET_CEILING_BYTES`. Never shrinks below the default,
        and tracks the CURRENT region size so a big ROI followed by a small one
        gives the memory back (the entries themselves go on the next insert's
        eviction pass, not here — evicting eagerly would drop frames the caller is
        about to ask for)."""
        slots = int(n_frames * REGION_BUDGET_HEADROOM) + 2
        want = max(self._default_budget_bytes,
                   min(REGION_BUDGET_CEILING_BYTES, slots * int(frame_bytes)))
        with self._lock:
            self._budget_bytes = want

    def drop_key(self, key: Any) -> None:
        """Evict every frame cached under ``key``. Needed because the key is the
        signal's identity, NOT its data's: SpyDE swaps ``signal.data`` in place
        (a progressive navigator fill, a result window's computed data landing)
        without creating a new signal object, and frames decoded from the old
        array must not survive that. Called when a reader is superseded because
        its ``data`` changed — see nav_read._reader_for."""
        with self._lock:
            stale = [k for k in self._entries if k[0] == key]
            for k in stale:
                self._nbytes -= self._entries.pop(k).nbytes

    def is_resident(self, key: Any, indices: tuple[int, ...]) -> bool:
        """Side-effect-free hit probe — never calls the reader."""
        cache_key = (key, tuple(int(v) for v in indices))
        return cache_key in self._entries

    def _evict_over_budget(self) -> None:
        # Caller holds _lock. The just-inserted entry is the MRU (last), and
        # popitem(last=False) drops the LRU (first), so the frame we were just
        # asked for survives even when it alone exceeds the budget — matches
        # _NavChunkCache's rule.
        while self._nbytes > self._budget_bytes and len(self._entries) > 1:
            _old_key, frame = self._entries.popitem(last=False)
            self._nbytes -= frame.nbytes

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        return self._nbytes

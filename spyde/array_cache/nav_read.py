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

import numpy as np

from .resolve import resolve_reader


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
    reader = resolve_reader(signal, data)
    if reader is not None:
        readers[key] = reader
    return reader


def get_local_frame(plot, signal, data, indices, prof=None):
    """Return the decoded frame at ``indices`` via the plot's ArrayCache, or
    None if this signal isn't ArrayCache-eligible right now (an opaque
    signal-tree node, a region rather than a single point, or no signal_tree
    available) — the caller falls back to a plain compute."""
    idx = np.asarray(indices)
    if idx.ndim > 1:
        return None  # region — not a single frame
    point = tuple(int(v) for v in np.atleast_1d(idx))

    # An EXISTING reader for this (signal, data) is itself proof the locality
    # gate already passed — only a local-resolved signal ever gets one. That
    # keeps BaseSignalTree.resolve_locality (which walks the tree to find the
    # node) off the per-frame path: it runs once per view, not once per move.
    reader = plot._local_transform_readers.get(id(signal))
    if reader is None or getattr(reader, "data", None) is not data:
        tree = getattr(plot, "signal_tree", None)
        if tree is None or not tree.resolve_locality(signal):
            return None
        reader = _reader_for(plot, signal, data)
        if reader is None:
            return None
    return plot._array_cache.get_frame(id(signal), reader, point, prof)


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

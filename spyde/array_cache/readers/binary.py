"""Reader kind 1: binary uncompressed — backed by rosettasciio's own
memmap_distributed primitives, not a re-derived header parser.

Getting a raw format's data offset wrong is a correctness bug, not just a
perf one — MRC in particular has an optional extended (FEI) header whose
size is read from the file itself, so a hardcoded offset (or a hand-rolled
header parser that drifts from rosettasciio's own) can silently misread
data. Instead of re-deriving it, this reader extracts the exact
``(file, dtypes, shape, offset, order, mode, key)`` rosettasciio's
``rsciio.utils._distributed.memmap_distributed`` already baked into the
lazy signal's dask graph when it built the array — confirmed by inspecting
a real hyperspy-loaded .mrc's graph: the ``slice_memmap`` task's kwargs are
exactly the parameters ``np.memmap`` needs, and a memmap built from them is
byte-identical to hyperspy's own ``.compute()``.

Those primitives are also exactly what memmap_distributed hands to a
distributed worker instead of a live memmap object (a memmap can't safely
cross a pickle/process boundary the way a filename+offset+dtype tuple can).
This reader doesn't need that property itself — it lives entirely in-process
on the dispatcher thread, never pickled anywhere — but reusing the same
primitives is what makes it correct without re-parsing anything.

Declines (resolves to None, via resolve_reader in this package) rather than
guess for anything more exotic: ``positions=True`` (an arbitrary/custom scan
pattern — no simple linear stride from nav index to file offset), a
non-C order, or a structured dtype with a ``key`` — the existing lazy-compute
path still serves those correctly, just without this fast tier.
"""
from __future__ import annotations

import os

import numpy as np


def find_memmap_source(data) -> dict | None:
    """Extract the memmap_distributed kwargs baked into ``data``'s dask graph,
    or None if this array wasn't built that way (a derived view, a reader
    that doesn't use memmap_distributed, or an arbitrary-positions scan).

    Matches on ``data.name`` — the array must BE the output of the
    ``slice_memmap`` layer, not merely have one somewhere upstream. A dask
    HighLevelGraph keeps every ancestor layer, so a DERIVED view of a
    memmap-backed file (``s.rebin(...)``, ``s.inav[a:b]``, ``center_direct_beam``
    — all of which the locality tag marks ArrayCache-eligible) still carries the
    source's ``slice_memmap`` tasks. Reading through them would return the RAW
    SOURCE frame with the transform silently skipped: verified as a wrong frame
    (no error at all) for a nav crop and a wrong-shaped frame for a rebin. So
    anything downstream of the memmap layer must decline here and be served by
    LocalTransformReader instead.

    Reading the ONE task from ``layers[name]`` also avoids materialising the
    whole graph — ``dict(graph)`` on a 13 k-chunk scan is ~200 ms.
    """
    name = getattr(data, "name", None)
    graph = getattr(data, "dask", None)
    if graph is None or not isinstance(name, str) \
            or not name.startswith("slice_memmap"):
        return None
    try:
        layer = graph.layers[name]
        task = layer[next(iter(layer))]
        kwargs = getattr(task, "kwargs", None)
        if not kwargs:
            return None
        if kwargs.get("positions", False):
            return None  # arbitrary scan pattern — decline, don't guess
        # Belt-and-braces: the layer we just read must describe THIS array. The
        # name check already implies it; this turns any future graph-shape
        # surprise into a decline (slow but correct) instead of a misread.
        shape = tuple(int(v) for v in kwargs["shape"])
        data_shape = tuple(int(v) for v in data.shape)
        if kwargs.get("key") is not None:
            # Structured dtype: ``shape`` covers the leading dims only, the
            # dtype's sub-array shape supplies the rest.
            if data_shape[:len(shape)] != shape:
                return None
        elif data_shape != shape or np.dtype(kwargs["dtypes"]) != data.dtype:
            return None
        return kwargs
    except Exception:
        return None


class BinaryReader:
    """One instance per (signal, data) pair. Opens the file once at
    construction (mmap is lazy — no data read yet); read_frame() does a
    single pread per call on the fast path, falling back to a memmap slice
    for anything the fast path doesn't support (Fortran order, a structured
    dtype with a key)."""

    def __init__(self, signal, data, memmap_kwargs: dict):
        self.signal = signal
        self.data = data
        self._nav_ndim = signal.axes_manager.navigation_dimension
        self._shape = tuple(int(s) for s in memmap_kwargs["shape"])
        # Frame/nav shapes come from the ARRAY, not the memmap kwargs: for a
        # structured dtype the kwargs' ``shape`` covers only the leading dims
        # (the dtype's sub-array supplies the frame), so slicing the kwargs
        # shape would give an empty frame shape and a zero frame_bytes.
        data_shape = tuple(int(s) for s in data.shape)
        self._frame_shape = data_shape[self._nav_ndim:]
        self._nav_shape = data_shape[:self._nav_ndim]
        self._dtype = np.dtype(memmap_kwargs["dtypes"])
        self._offset = int(memmap_kwargs["offset"])
        self._order = memmap_kwargs.get("order", "C")
        self._key = memmap_kwargs.get("key")
        self._filename = memmap_kwargs["file"]
        self._frame_nbytes = int(np.prod(self._frame_shape)) * self._dtype.itemsize

        # Always-correct fallback: a real memmap built from the exact same
        # kwargs np.memmap needs (handles order/structured-key correctly by
        # construction, since it's just numpy's own indexing).
        self._memmap = np.memmap(
            self._filename, self._dtype, mode="r",
            offset=self._offset, shape=self._shape, order=self._order,
        )
        if self._key is not None:
            self._memmap = self._memmap[self._key]

        # os.pread is Unix-only (no Windows implementation), so the pread fast
        # path is gated on it — otherwise every read_frame would raise
        # AttributeError on Windows and silently degrade to a plain compute
        # upstream. The memmap fallback below is the Windows path and is itself
        # a single page-cache-backed slice.
        self._fast_path = (
            self._order == "C" and self._key is None and hasattr(os, "pread")
        )
        self._fd = os.open(self._filename, os.O_RDONLY) if self._fast_path else None

    @property
    def frame_bytes(self) -> int:
        return self._frame_nbytes

    def _flat_nav_index(self, indices: tuple[int, ...]) -> int:
        flat, stride = 0, 1
        for ax in range(self._nav_ndim - 1, -1, -1):
            flat += int(indices[ax]) * stride
            stride *= self._nav_shape[ax]
        return flat

    def read_frame(self, indices: tuple[int, ...]) -> np.ndarray:
        """One frame, as a fresh array (never a view into a bigger buffer, so
        an ArrayCache entry retains exactly one frame). The pread result is
        zero-copy and therefore READ-ONLY — like every cached frame it must be
        treated as immutable by consumers (see ArrayCache)."""
        if self._fast_path:
            flat = self._flat_nav_index(indices)
            byte_offset = self._offset + flat * self._frame_nbytes
            buf = os.pread(self._fd, self._frame_nbytes, byte_offset)
            if len(buf) != self._frame_nbytes:      # short read → truncated file
                raise IOError(
                    f"short read: got {len(buf)} of {self._frame_nbytes} bytes "
                    f"at offset {byte_offset} in {self._filename}")
            return np.frombuffer(buf, dtype=self._dtype).reshape(self._frame_shape)
        return np.array(self._memmap[tuple(int(v) for v in indices)])

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __del__(self):
        # Backstop so a superseded reader's fd is released when the last
        # reference drops, even though _reader_for deliberately does NOT close
        # it (another thread may be mid-pread on it — see nav_read._reader_for).
        try:
            self.close()
        except Exception:
            pass

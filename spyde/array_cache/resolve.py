"""resolve_reader — pick the best FrameReader kind for a (signal, data) pair,
once, at read time (cached by the caller keyed on id(signal) — see
nav_read.py._reader_for). Tries the more specific/faster kinds first, falling
back to the universal one:

  1. binary (memmap_distributed-backed) — fastest, only applies to a base
     signal whose reader used rosettasciio's memmap_distributed utility.
  2/3. source_array (zarr+blosc / HDF5, and any other da.from_array-wrapped
     file-backed source) — reads the one frame straight from the store,
     touching only the storage chunks that hold it.
  4. local_transform (any dask-chunked array) — the universal fallback; works
     for a locality-tagged derived view and for any base signal whose backing
     none of the above recognise.

EVERY specific kind must decline for a DERIVED view (a rebin/crop of a
memmap- or zarr-backed file) rather than read through to the untransformed
source — a dask HighLevelGraph keeps its ancestor layers, so the source's
tasks are still present in a transformed array's graph. Both finders gate on
``data.name`` (the array must BE that layer's output) plus a shape/dtype match;
see their module docstrings. This function is the one place that needs to grow a
branch for a new backing, not any call site.
"""
from __future__ import annotations

from .readers.binary import BinaryReader, find_memmap_source
from .readers.local_transform import LocalTransformReader
from .readers.source_array import SourceArrayReader, find_source_array


def resolve_reader(signal, data):
    kwargs = find_memmap_source(data)
    if kwargs is not None:
        try:
            return BinaryReader(signal, data, kwargs)
        except Exception:
            pass  # fall through to the next kind

    source = find_source_array(data)
    if source is not None:
        try:
            return SourceArrayReader(signal, data, source)
        except Exception:
            pass  # fall through to the universal reader

    try:
        return LocalTransformReader(signal, data)
    except Exception:
        return None

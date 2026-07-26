"""Backing-aware, byte-budgeted frame cache for fast navigator scrubbing
across large 4D-STEM/movie datasets — raw binary, zarr+blosc, HDF5, and
signal-tree local-transform views, behind one FrameReader interface.

Reader kinds, tried in this order (see resolve.py):

  1. ``readers.binary``        raw uncompressed via rosettasciio's
                               memmap_distributed primitives (.mrc, .de5, raw)
  2/3. ``readers.source_array`` zarr+blosc (.zspy) and HDF5 (.hspy) read
                               straight from the open store, plus any other
                               ``da.from_array``-wrapped file-backed source
  4. ``readers.local_transform`` universal dask fallback — the only kind that
                               serves a locality-tagged DERIVED view

Every specific kind must DECLINE for a derived view instead of reading through
to its untransformed source; see resolve.py.
"""
from .cache import ArrayCache, DEFAULT_BUDGET_BYTES
from .block_cache import (
    BlockCache, DEFAULT_BLOCK_BUDGET_BYTES, UNFOCUSED_BUDGET_BYTES,
)
from .protocol import FrameReader
from .nav_read import get_local_frame, is_local_frame_resident, close_all_readers

__all__ = [
    "ArrayCache",
    "BlockCache",
    "FrameReader",
    "DEFAULT_BUDGET_BYTES",
    "DEFAULT_BLOCK_BUDGET_BYTES",
    "UNFOCUSED_BUDGET_BYTES",
    "get_local_frame",
    "is_local_frame_resident",
    "close_all_readers",
]

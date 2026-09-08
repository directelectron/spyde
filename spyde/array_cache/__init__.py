"""Backing-aware, byte-budgeted frame cache for fast navigator scrubbing
across large 4D-STEM/movie datasets — raw binary, zarr+blosc, HDF5, and
signal-tree local-transform views, behind one FrameReader interface.

Reader kinds, tried in this order (nav_read._reader_for, then resolve.py):

  0. ``readers.eager``         data already in RAM: a frame is an index
                               (the parent of a derived view on an eager root)
  5. ``readers.per_frame``     rebin / crop of the PARENT's frame, in numpy
  6. ``readers.recipe``        a node made by a hyperspy ``map``: the recorded
                               function on the PARENT's frame, no dask block
  1. ``readers.binary``        raw uncompressed via rosettasciio's
                               memmap_distributed primitives (.mrc, .de5, raw)
  2/3. ``readers.source_array`` zarr+blosc (.zspy) and HDF5 (.hspy) read
                               straight from the open store, plus any other
                               ``da.from_array``-wrapped file-backed source
  4. ``readers.local_transform`` universal dask fallback — serves any other
                               locality-tagged DERIVED view a block at a time

Every specific kind must DECLINE for a derived view instead of reading through
to its untransformed source; see resolve.py.
"""
from .cache import (
    ArrayCache, DEFAULT_BUDGET_BYTES, REGION_BUDGET_CEILING_BYTES,
)
from .block_cache import (
    BlockCache, DEFAULT_BLOCK_BUDGET_BYTES, UNFOCUSED_BUDGET_BYTES,
)
from .protocol import FrameReader
from .region_sum import RegionIntegrator, finalize_sum
from .nav_read import (
    get_local_frame, is_local_frame_resident, close_all_readers, retain_readers,
)

__all__ = [
    "ArrayCache",
    "BlockCache",
    "FrameReader",
    "RegionIntegrator",
    "finalize_sum",
    "DEFAULT_BUDGET_BYTES",
    "REGION_BUDGET_CEILING_BYTES",
    "DEFAULT_BLOCK_BUDGET_BYTES",
    "UNFOCUSED_BUDGET_BYTES",
    "get_local_frame",
    "is_local_frame_resident",
    "close_all_readers",
    "retain_readers",
]

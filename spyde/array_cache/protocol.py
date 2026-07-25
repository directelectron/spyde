"""FrameReader — the backing-agnostic interface every ArrayCache reader kind
implements (binary/pread, zarr+blosc, HDF5, signal-tree-local-transform).

A reader instance is scoped to ONE signal's data and resolved ONCE (per
file-open or per node-select), not per frame — everything backing-specific
(file handles, chunk indices, codec setup) lives in the reader's __init__,
so read_frame() is cheap enough to call on every ArrayCache miss.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameReader(Protocol):
    def read_frame(self, indices: tuple[int, ...]) -> np.ndarray:
        """Return the decoded frame at the given navigation indices."""
        ...

    @property
    def frame_bytes(self) -> int:
        """Size in bytes of one decoded frame — used for cache budget accounting
        and for the cheap/expensive read-size checks upstream of the cache."""
        ...

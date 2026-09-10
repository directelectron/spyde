"""Reader kind 0: data that is already in RAM. A frame is an index.

Needed as the PARENT of a derived view whose root was loaded eagerly (the
bundled synthetic datasets are): the per-frame and recipe readers read their
parent's frames through the parent's reader, and every other kind either
declines an in-RAM array or expects a dask one.
"""
from __future__ import annotations

import numpy as np


class EagerReader:
    def __init__(self, data: np.ndarray, nav_ndim: int):
        self.data = data
        self._nav_ndim = nav_ndim

    @property
    def frame_bytes(self) -> int:
        frame_shape = self.data.shape[self._nav_ndim:]
        return int(np.prod(frame_shape)) * self.data.dtype.itemsize

    def is_chunk_resident(self, indices) -> bool:
        return True

    def read_frame(self, indices: tuple[int, ...]) -> np.ndarray:
        point = tuple(int(v) for v in indices[:self._nav_ndim])
        # A copy, so a cached frame never aliases the dataset.
        return np.array(self.data[point])

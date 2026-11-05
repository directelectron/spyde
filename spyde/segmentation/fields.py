"""
fields.py — the 2-D images a segmentation runs over, read one at a time.

A *field* is any 2-D image the user paints on and the classifier labels. Three
kinds of dataset supply them, and the point of this module is that nothing
downstream can tell them apart:

* a single image — one field;
* an in-situ movie (1-D navigation over 2-D frames) — one field per frame,
  read through :func:`spyde.drift.frame_source` so a frame is computed only
  when asked for and the stack is never materialised;
* the real-space navigator of a 4-D STEM scan — one field whose pixels ARE
  the scan positions, so an instance found on it is a set of positions.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class FieldSource:
    """``count`` fields of ``shape``, read by ``get(index)``."""

    def __init__(self, count: int, get: Callable[[int], np.ndarray],
                 shape: tuple[int, int]) -> None:
        self.count = int(count)
        self.get = get
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def is_movie(self) -> bool:
        return self.count > 1


def field_source(data) -> FieldSource:
    """Fields for *data*: a 2-D array, a 2-D-signal HyperSpy signal with 0-D
    or 1-D navigation, or anything :func:`spyde.drift.frame_source` accepts.

    Raises ``TypeError`` for a signal whose frames are not 2-D images or whose
    navigation has more than one axis: a 4-D STEM scan is segmented on its
    navigator, not on its diffraction patterns.
    """
    if hasattr(data, "axes_manager"):
        manager = data.axes_manager
        if int(manager.signal_dimension) != 2:
            raise TypeError("segmentation needs 2-D image frames; got "
                            f"signal_dimension={int(manager.signal_dimension)}")
        if int(manager.navigation_dimension) == 0:
            return _single(data.data)
        if int(manager.navigation_dimension) > 1:
            raise TypeError("segment a 4-D scan on its real-space navigator window, "
                            "not on the diffraction pattern")
        from spyde.drift import frame_source
        count, get_frame, shape = frame_source(data)
        return FieldSource(count, get_frame, shape)
    if getattr(data, "ndim", None) == 2:
        return _single(data)
    from spyde.drift import frame_source
    count, get_frame, shape = frame_source(data)
    return FieldSource(count, get_frame, shape)


def _single(array) -> FieldSource:
    def get(_index: int, _array=array) -> np.ndarray:
        computed = _array.compute() if hasattr(_array, "compute") else _array
        return np.asarray(computed)
    height, width = int(array.shape[-2]), int(array.shape[-1])
    return FieldSource(1, get, (height, width))

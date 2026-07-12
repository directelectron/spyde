"""
Movie read-ahead prefetch (Phase 3).

_MoviePrefetcher warms the OS page cache for the frames a movie scrub is about to
reach: after painting frame t it reads t±1…t±radius on a background thread, off
the RAW dask array (never the CachedDaskArray, so it can't race the nav read's
cache — CLAUDE.md §4). This pins that it reads the right neighbouring frames,
stays in bounds, and is latest-center-wins.
"""
from __future__ import annotations

import time
import threading

import numpy as np
import dask.array as da

from spyde.drawing.update_functions import _MoviePrefetcher


class _RecordingArray:
    """Wraps a dask array; records every integer-frame index that gets computed."""
    def __init__(self, arr):
        self._arr = arr
        self.shape = arr.shape
        self.reads = []
        self._lock = threading.Lock()

    def __getitem__(self, i):
        with self._lock:
            self.reads.append(int(i))
        return self._arr[i]


def _wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


class TestMoviePrefetch:
    def _arr(self, n=50, frame=(16, 16)):
        return _RecordingArray(da.zeros((n,) + frame, dtype=np.float32,
                                        chunks=(1,) + frame))

    def test_reads_neighbours_around_center(self):
        pf = _MoviePrefetcher(radius=2)
        arr = self._arr()
        pf.prime(arr, center=20, n_time=50)
        # Expect frames 18,19,21,22 (t±1, t±2) to be read — not 20 itself.
        _wait_until(lambda: set([18, 19, 21, 22]).issubset(set(arr.reads)))
        got = set(arr.reads)
        assert {18, 19, 21, 22}.issubset(got), f"prefetch read {sorted(got)}"
        assert 20 not in got, "prefetch should not re-read the current frame"

    def test_stays_in_bounds(self):
        pf = _MoviePrefetcher(radius=3)
        arr = self._arr(n=5)
        pf.prime(arr, center=0, n_time=5)      # only 1,2,3 are valid ahead
        _wait_until(lambda: len(arr.reads) >= 3)
        got = set(arr.reads)
        assert all(0 <= i < 5 for i in got), f"out-of-bounds read: {sorted(got)}"
        assert -1 not in got and -2 not in got

    def test_latest_center_wins(self):
        pf = _MoviePrefetcher(radius=2)
        arr = self._arr(n=100)
        # Fire many centers quickly; the prefetcher should end up reading around
        # the LAST center (90), not pile up all of them.
        for c in (10, 30, 60, 90):
            pf.prime(arr, center=c, n_time=100)
        _wait_until(lambda: set([88, 89, 91, 92]).issubset(set(arr.reads)))
        assert {88, 89, 91, 92}.issubset(set(arr.reads))

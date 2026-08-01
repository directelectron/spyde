"""test_nav_cluster_handover.py — the threaded navigator fill hands over.

`_start_progressive_nav_compute` picks its path from `self.client`. The cluster
takes ~10 s to come up, so a file opened right after launch finds it None and
takes the THREADED branch — one background thread, one chunk at a time.

That choice used to be permanent. However many workers registered a second
later, the whole fill ran single-threaded, which on a long movie is the
difference between seconds and minutes and reads as "the cluster is idle".

Why it looked like a movie-only bug: a 4D-STEM scan goes through the nav-shape
prompt (a human round trip), so its cluster is always up by the time the tree is
built. A movie opens straight through and loses the race.
"""
from __future__ import annotations

import threading
import time

import dask.array as da
import numpy as np
import pytest


class _Plot:
    def __init__(self):
        self.window_id = 1
        self.painted = []

    def set_data(self, arr, levels=None):
        self.painted.append(arr)

    def _emit_histogram(self, *a, **k):
        pass


class _Tree:
    """Just enough tree to drive the threaded fill's handover check."""

    def __init__(self, client_after: int):
        self._client_after = client_after      # chunks before the cluster "starts"
        self._chunks_done = 0
        self.handed_over_with = None
        self.session = None
        self.source_path = None

    @property
    def client(self):
        return object() if self._chunks_done >= self._client_after else None

    def _start_progressive_nav_compute(self, nav_dask, deep=None):
        self.handed_over_with = (nav_dask, deep)


def _drive(tree, total_chunks: int, min_remaining: int):
    """The handover decision, lifted verbatim from the fill loop."""
    for done in range(total_chunks):
        tree._chunks_done = done
        if (tree.client is not None
                and (total_chunks - done) > min_remaining):
            return done
    return None


class TestHandover:
    def test_hands_over_once_the_cluster_appears(self):
        from spyde.signal_tree import _NAV_HANDOVER_MIN_CHUNKS
        tree = _Tree(client_after=3)
        at = _drive(tree, total_chunks=100, min_remaining=_NAV_HANDOVER_MIN_CHUNKS)
        assert at == 3, (
            "the fill did not hand over on the first chunk after the cluster "
            "registered — the threaded path would run the whole movie")

    def test_does_not_hand_over_near_the_end(self):
        """Handing over recomputes what is already painted, so with only a
        handful of chunks left it costs more than it saves."""
        from spyde.signal_tree import _NAV_HANDOVER_MIN_CHUNKS
        tree = _Tree(client_after=0)
        total = _NAV_HANDOVER_MIN_CHUNKS      # every step has <= min remaining
        assert _drive(tree, total_chunks=total,
                      min_remaining=_NAV_HANDOVER_MIN_CHUNKS) is None

    def test_never_hands_over_without_a_cluster(self):
        from spyde.signal_tree import _NAV_HANDOVER_MIN_CHUNKS
        tree = _Tree(client_after=10_000)     # cluster never arrives
        assert _drive(tree, total_chunks=500,
                      min_remaining=_NAV_HANDOVER_MIN_CHUNKS) is None


class TestWiring:
    def test_the_fill_loop_actually_contains_the_handover(self):
        """Guard against the check being dropped in a refactor: the decision
        above is only meaningful if the real loop still makes it."""
        import inspect
        from spyde.signal_tree import BaseSignalTree
        src = inspect.getsource(BaseSignalTree._start_progressive_nav_compute)
        assert "_NAV_HANDOVER_MIN_CHUNKS" in src, (
            "the threaded navigator fill no longer checks for a late cluster")
        assert "_start_progressive_nav_compute(_dask" in src, (
            "the handover no longer re-enters the dispatcher path")

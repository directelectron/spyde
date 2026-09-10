"""The read-ahead aims at the next chunk, and two threads never decode one
block twice.

A drag pays nothing inside a decoded chunk and about 80 ms entering an
undecoded one, so the only stall left is the crossing. Aiming a fixed number of
positions ahead misses it: a drag of one position per step is still inside its
own chunk two positions later, so the read-ahead warmed a chunk that was
already warm and every crossing decoded on the dispatcher. These tests pin the
aim, the byte warm one chunk further, and the deduplication that makes the
dispatcher wait for a decode already running rather than start a second one.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import dask.array as da

from spyde.array_cache import ArrayCache, BlockCache
from spyde.array_cache.readers.local_transform import LocalTransformReader
from spyde.drawing.update_functions import (
    BYTE_WARM_VARIABLE, _block_prefetcher, _byte_warm_enabled,
    _next_chunk_position, _step_signs, _warm_chunk_bytes,
)


class _AxesManager:
    def __init__(self, navigation_dimension):
        self.navigation_dimension = navigation_dimension


class _Signal:
    def __init__(self, data, navigation_dimension):
        self.data = data
        self.axes_manager = _AxesManager(navigation_dimension)


class _Plot:
    """The three attributes the read path touches on a plot."""

    def __init__(self, signal, reader, block_cache=None):
        self._array_cache = ArrayCache()
        self._block_cache = block_cache or BlockCache()
        self._local_transform_readers = {id(signal): reader}


class _ChunkedReader:
    """A reader with eight-wide chunks that records what it was asked for."""

    def __init__(self, data, chunk=8, source=None):
        self.data = data
        self.source = source
        self._chunk = chunk
        self.frames_read: list[tuple] = []

    @property
    def frame_bytes(self):
        return 4

    def chunk_span(self, point):
        return tuple((int(p) // self._chunk * self._chunk,
                      int(p) // self._chunk * self._chunk + self._chunk)
                     for p in point)

    def is_chunk_resident(self, point):
        return False

    def read_frame(self, point):
        self.frames_read.append(tuple(int(v) for v in point))
        return np.zeros((1, 1), dtype=np.float32)


class _RecordingStore(dict):
    def __init__(self):
        super().__init__()
        self.requested: list[str] = []

    def __getitem__(self, key):
        self.requested.append(key)
        return b""


class _FakeZarrArray:
    """The four attributes the byte warm needs off a zarr 2 array."""

    def __init__(self, chunks, grid_shape):
        self.chunks = chunks
        self.cdata_shape = grid_shape
        self.chunk_store = _RecordingStore()

    def _chunk_key(self, coordinates):
        return "data/" + ".".join(str(int(c)) for c in coordinates)


def _wait_for(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestGetOrLoad:
    def test_two_threads_load_once_and_both_get_the_block(self):
        cache = BlockCache()
        block = np.arange(64, dtype=np.float32)
        loads = []
        started = threading.Event()
        release = threading.Event()

        def load():
            loads.append(1)
            started.set()
            release.wait(5.0)
            return block

        results = []

        def ask():
            results.append(cache.get_or_load("reader", (0, 0), load))

        first = threading.Thread(target=ask)
        first.start()
        assert started.wait(5.0)
        second = threading.Thread(target=ask)
        second.start()
        # The second caller is inside the wait, not inside a second load.
        time.sleep(0.1)
        assert len(loads) == 1
        release.set()
        first.join(5.0)
        second.join(5.0)

        assert len(loads) == 1
        assert len(results) == 2
        assert all(r is block for r in results)
        assert cache.contains("reader", (0, 0))

    def test_a_hit_does_not_load(self):
        cache = BlockCache()
        block = np.zeros(4, dtype=np.float32)
        cache.put("reader", (1,), block)
        assert cache.get_or_load("reader", (1,),
                                 lambda: _must_not_load()) is block

    def test_a_failed_load_reaches_the_waiter_and_leaves_no_entry(self):
        cache = BlockCache()
        started = threading.Event()
        release = threading.Event()

        def load():
            started.set()
            release.wait(5.0)
            raise ValueError("the store said no")

        errors = []

        def ask():
            try:
                cache.get_or_load("reader", (0,), load)
            except ValueError as error:
                errors.append(error)

        first = threading.Thread(target=ask)
        first.start()
        assert started.wait(5.0)
        second = threading.Thread(target=ask)
        second.start()
        release.set()
        first.join(5.0)
        second.join(5.0)

        assert len(errors) == 2
        assert not cache.contains("reader", (0,))
        # The key is free again, so the next caller retries rather than
        # inheriting the failure for ever.
        assert cache.get_or_load("reader", (0,),
                                 lambda: np.zeros(1)) is not None


def _must_not_load():
    raise AssertionError("load must not run on a cache hit")


class TestAim:
    def test_travelling_along_x_targets_the_next_chunk(self):
        """Primed from (10, 3) to (13, 3), eight-wide chunks: (16, 3)."""
        reader = _ChunkedReader(None)
        signs = _step_signs((10, 3), (13, 3))
        assert signs == (1, 0)
        assert _next_chunk_position(reader, (13, 3), signs, (256, 256)) \
            == (16, 3)

    def test_travelling_backwards_targets_the_chunk_below(self):
        reader = _ChunkedReader(None)
        signs = _step_signs((20, 3), (17, 3))
        assert _next_chunk_position(reader, (17, 3), signs, (256, 256)) \
            == (15, 3)

    def test_a_diagonal_step_moves_both_axes(self):
        reader = _ChunkedReader(None)
        signs = _step_signs((10, 10), (11, 11))
        assert _next_chunk_position(reader, (11, 11), signs, (256, 256)) \
            == (16, 16)

    def test_the_target_is_clipped_to_the_grid(self):
        reader = _ChunkedReader(None)
        signs = _step_signs((250, 3), (253, 3))
        assert _next_chunk_position(reader, (253, 3), signs, (256, 256)) \
            == (255, 3)

    def test_a_reader_without_chunks_has_no_target(self):
        class _NoSpan:
            pass

        assert _next_chunk_position(_NoSpan(), (5, 5), (1, 0), (64, 64)) is None


class TestByteWarm:
    def test_the_chunk_key_is_requested(self):
        source = _FakeZarrArray(chunks=(8, 8, 512, 512),
                                grid_shape=(32, 32, 1, 1))
        reader = _ChunkedReader(None, source=source)
        _warm_chunk_bytes(reader, (24, 0))
        assert source.chunk_store.requested == ["data/3.0.0.0"]

    def test_every_signal_chunk_of_the_block_is_requested(self):
        source = _FakeZarrArray(chunks=(8, 8, 256, 256),
                                grid_shape=(32, 32, 2, 1))
        reader = _ChunkedReader(None, source=source)
        _warm_chunk_bytes(reader, (8, 8))
        assert source.chunk_store.requested == ["data/1.1.0.0", "data/1.1.1.0"]

    def test_a_position_off_the_grid_reads_nothing(self):
        source = _FakeZarrArray(chunks=(8, 8, 512, 512),
                                grid_shape=(4, 4, 1, 1))
        reader = _ChunkedReader(None, source=source)
        _warm_chunk_bytes(reader, (40, 0))
        assert source.chunk_store.requested == []

    def test_a_reader_without_a_store_is_skipped(self):
        _warm_chunk_bytes(_ChunkedReader(None), (0, 0))     # must not raise


class TestPrefetcher:
    def _prime_and_wait(self, monkeypatch, byte_warm):
        monkeypatch.setenv(BYTE_WARM_VARIABLE, "1" if byte_warm else "0")
        source = _FakeZarrArray(chunks=(8, 8, 4, 4), grid_shape=(32, 32, 1, 1))
        data = da.zeros((256, 256, 4, 4), chunks=(8, 8, 4, 4),
                        dtype=np.float32)
        signal = _Signal(data, 2)
        reader = _ChunkedReader(data, source=source)
        plot = _Plot(signal, reader)
        _block_prefetcher.prime(plot, signal, data, (10, 3), (13, 3), 2)
        return reader, source

    def test_it_decodes_the_chunk_the_drag_is_heading_into(self, monkeypatch):
        reader, _ = self._prime_and_wait(monkeypatch, byte_warm=False)
        assert _wait_for(lambda: reader.frames_read)
        assert reader.frames_read[0] == (16, 3)

    def test_the_byte_warm_asks_for_the_chunk_after_that(self, monkeypatch):
        """Chunk (2, 0) is decoded, so chunk (3, 0)'s bytes are requested."""
        reader, source = self._prime_and_wait(monkeypatch, byte_warm=True)
        assert _wait_for(lambda: reader.frames_read
                         and source.chunk_store.requested)
        assert reader.frames_read[0] == (16, 3)
        assert source.chunk_store.requested[0] == "data/3.0.0.0"

    def test_the_byte_warm_is_off_by_default(self, monkeypatch):
        """It measured slower on every drag tried, so it takes a switch."""
        monkeypatch.delenv(BYTE_WARM_VARIABLE, raising=False)
        assert _byte_warm_enabled() is False
        reader, source = self._prime_and_wait(monkeypatch, byte_warm=False)
        assert _wait_for(lambda: reader.frames_read)
        time.sleep(0.2)
        assert source.chunk_store.requested == []

    def test_a_crossing_is_a_hit_once_the_read_ahead_has_run(self):
        """The point of the whole change, on a lazy chunked array."""
        values = np.random.RandomState(0).rand(64, 64, 4, 4).astype(np.float32)
        data = da.from_array(values, chunks=(8, 8, -1, -1))
        signal = _Signal(data, 2)
        block_cache = BlockCache()
        reader = LocalTransformReader(signal, data, block_cache=block_cache)
        plot = _Plot(signal, reader, block_cache=block_cache)

        assert not reader.is_chunk_resident((16, 3))
        _block_prefetcher.prime(plot, signal, data, (10, 3), (13, 3), 2)
        assert _wait_for(lambda: reader.is_chunk_resident((16, 3)))

        # And the frame the drag arrives at comes out of that block correctly.
        np.testing.assert_array_equal(reader.read_frame((16, 3)),
                                      values[16, 3])

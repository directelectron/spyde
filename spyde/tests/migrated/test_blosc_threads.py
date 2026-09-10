"""Blosc decodes with its thread pool on every thread, one caller at a time.

numcodecs picks its decode mode from the identity of the calling thread, so the
navigator dispatcher, the block prefetcher and dask's workers all decoded
single-threaded: 172 ms for a 64 MiB chunk against 29 ms with the pool. Turning
the pool on for every thread is only safe if one caller uses it at a time,
because the pool is global, so these tests pin both halves together.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from numcodecs import Blosc

from spyde.external.numcodecs import blosc_threads


@pytest.fixture
def blosc_module():
    import numcodecs.blosc as blosc
    return blosc


def _chunk_bytes(codec, values):
    return codec.encode(values)


class TestApply:
    def test_wrapper_is_in_place_after_apply(self, blosc_module):
        assert blosc_threads.apply() is True
        assert getattr(blosc_module.compress, blosc_threads.MARKER, False)
        assert getattr(blosc_module.decompress, blosc_threads.MARKER, False)
        assert blosc_module.use_threads is True
        assert blosc_module.get_nthreads() == blosc_threads.thread_count()

    def test_apply_is_idempotent(self, blosc_module):
        blosc_threads.apply()
        wrapped = blosc_module.decompress
        assert blosc_threads.apply() is True
        assert blosc_module.decompress is wrapped

    def test_round_trip_still_correct(self):
        blosc_threads.apply()
        codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
        values = np.arange(200000, dtype=np.float32)
        decoded = np.frombuffer(codec.decode(codec.encode(values)),
                                dtype=np.float32)
        np.testing.assert_array_equal(decoded, values)


class TestSwitch:
    def test_zero_applies_nothing(self, monkeypatch, blosc_module):
        """SPYDE_BLOSC_THREADS=0 is the A/B switch: numcodecs is left alone."""
        monkeypatch.setenv(blosc_threads.THREAD_COUNT_VARIABLE, "0")
        assert blosc_threads.thread_count() is None
        before = (blosc_module.compress, blosc_module.decompress,
                  blosc_module.use_threads, blosc_module.get_nthreads())
        assert blosc_threads.apply() is False
        assert (blosc_module.compress, blosc_module.decompress,
                blosc_module.use_threads, blosc_module.get_nthreads()) == before

    def test_a_number_sets_that_thread_count(self, monkeypatch):
        monkeypatch.setenv(blosc_threads.THREAD_COUNT_VARIABLE, "4")
        assert blosc_threads.thread_count() == 4

    def test_unset_uses_the_default(self, monkeypatch):
        monkeypatch.delenv(blosc_threads.THREAD_COUNT_VARIABLE, raising=False)
        assert blosc_threads.thread_count() == blosc_threads.DEFAULT_THREAD_COUNT

    def test_nonsense_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv(blosc_threads.THREAD_COUNT_VARIABLE, "lots")
        assert blosc_threads.thread_count() == blosc_threads.DEFAULT_THREAD_COUNT


class TestConcurrentDecode:
    def test_eight_threads_decoding_one_chunk_agree(self):
        """The pool is global. Without the lock, concurrent callers share it and
        the output is whatever they did to each other's state."""
        blosc_threads.apply()
        codec = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
        rng = np.random.default_rng(0)
        values = rng.integers(0, 4096, 1 << 20).astype(np.float32)
        chunk = _chunk_bytes(codec, values)

        results: list[np.ndarray] = []
        failures: list[BaseException] = []
        start = threading.Barrier(8)

        def decode():
            try:
                start.wait()
                for _ in range(6):
                    results.append(np.frombuffer(codec.decode(chunk),
                                                 dtype=np.float32))
            except BaseException as error:
                failures.append(error)

        readers = [threading.Thread(target=decode, name=f"decode-{i}")
                   for i in range(8)]
        [reader.start() for reader in readers]
        [reader.join() for reader in readers]

        assert not failures
        assert len(results) == 48
        for decoded in results:
            np.testing.assert_array_equal(decoded, values)

    def test_the_registry_carries_the_patch(self):
        """apply_all is what the backend calls, so the patch has to be in it."""
        import spyde.external as external

        external.apply_all(force=True)
        assert any(name == "numcodecs" for name, _ in external._REGISTRY)

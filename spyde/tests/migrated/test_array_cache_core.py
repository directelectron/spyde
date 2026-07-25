"""ArrayCache core: byte-budget LRU of decoded frames, backing-agnostic via
FrameReader. These tests pin: hit vs miss (no redundant reader calls), LRU
touch-on-hit, byte-budget eviction, keep-at-least-one-oversized-entry, the
side-effect-free is_resident probe, clear(), and per-key (node-switch)
invalidation. Standalone — no Session, mirrors test_nav_chunk_cache.py's
fake-object style.
"""
import numpy as np

from spyde.array_cache import ArrayCache


class _Prof:
    def __init__(self):
        self.last = None

    def done(self, msg=""):
        self.last = msg


class _FakeReader:
    """Returns a deterministic frame per index and counts calls, so tests can
    assert a cache hit never reaches the reader."""

    def __init__(self, shape=(4, 4), dtype=np.uint16, seed=0):
        self.shape = shape
        self.dtype = dtype
        self.calls = []
        self._rng = np.random.RandomState(seed)
        self._known = {}

    def read_frame(self, indices):
        self.calls.append(indices)
        if indices not in self._known:
            self._known[indices] = self._rng.randint(
                0, 1000, self.shape
            ).astype(self.dtype)
        return self._known[indices]

    @property
    def frame_bytes(self):
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize


class TestHitMiss:
    def test_first_read_is_a_miss(self):
        cache = ArrayCache()
        reader = _FakeReader()
        prof = _Prof()
        frame = cache.get_frame("sig", reader, (0, 0), prof)
        assert "MISS" in prof.last
        assert reader.calls == [(0, 0)]
        np.testing.assert_array_equal(frame, reader._known[(0, 0)])

    def test_second_read_same_indices_is_a_hit_no_reader_call(self):
        cache = ArrayCache()
        reader = _FakeReader()
        cache.get_frame("sig", reader, (0, 0))
        prof = _Prof()
        frame = cache.get_frame("sig", reader, (0, 0), prof)
        assert prof.last == "array-cache hit"
        assert reader.calls == [(0, 0)]  # not called again
        np.testing.assert_array_equal(frame, reader._known[(0, 0)])

    def test_different_indices_is_a_miss(self):
        cache = ArrayCache()
        reader = _FakeReader()
        cache.get_frame("sig", reader, (0, 0))
        prof = _Prof()
        cache.get_frame("sig", reader, (1, 1), prof)
        assert "MISS" in prof.last
        assert reader.calls == [(0, 0), (1, 1)]


class TestResidentProbe:
    def test_is_resident_never_calls_reader(self):
        cache = ArrayCache()
        reader = _FakeReader()
        assert cache.is_resident("sig", (0, 0)) is False
        assert reader.calls == []
        cache.get_frame("sig", reader, (0, 0))
        assert cache.is_resident("sig", (0, 0)) is True
        assert cache.is_resident("sig", (1, 1)) is False
        assert reader.calls == [(0, 0)]  # is_resident added no calls


class TestKeyInvalidation:
    def test_different_key_misses_even_for_same_indices(self):
        # A node switch changes the key (e.g. id(signal)) but indices could
        # coincidentally repeat — must still miss.
        cache = ArrayCache()
        reader = _FakeReader()
        cache.get_frame("sig-a", reader, (0, 0))
        prof = _Prof()
        cache.get_frame("sig-b", reader, (0, 0), prof)
        assert "MISS" in prof.last
        assert reader.calls == [(0, 0), (0, 0)]


class TestClear:
    def test_clear_drops_everything(self):
        cache = ArrayCache()
        reader = _FakeReader()
        cache.get_frame("sig", reader, (0, 0))
        cache.get_frame("sig", reader, (1, 1))
        assert len(cache) == 2
        assert cache.nbytes > 0
        cache.clear()
        assert len(cache) == 0
        assert cache.nbytes == 0
        assert cache.is_resident("sig", (0, 0)) is False


class TestByteBudgetEviction:
    def test_lru_evicts_oldest_over_budget(self):
        reader = _FakeReader(shape=(4, 4), dtype=np.uint16)  # 32 bytes/frame
        budget = reader.frame_bytes * 2 + 1  # room for ~2 frames
        cache = ArrayCache(budget_bytes=budget)
        for i in range(5):
            cache.get_frame("sig", reader, (i,))
        assert cache.nbytes <= budget + reader.frame_bytes  # last insert can push over briefly
        # oldest frames evicted, most recent survive
        assert cache.is_resident("sig", (0,)) is False
        assert cache.is_resident("sig", (4,)) is True

    def test_touching_a_frame_protects_it_from_eviction(self):
        reader = _FakeReader(shape=(4, 4), dtype=np.uint16)
        budget = reader.frame_bytes * 2 + 1
        cache = ArrayCache(budget_bytes=budget)
        cache.get_frame("sig", reader, (0,))
        cache.get_frame("sig", reader, (1,))
        cache.get_frame("sig", reader, (0,))  # re-touch (0,) -> moves to end (MRU)
        cache.get_frame("sig", reader, (2,))  # forces an eviction
        # (1,) is now the oldest untouched entry and should be evicted, not (0,)
        assert cache.is_resident("sig", (0,)) is True
        assert cache.is_resident("sig", (1,)) is False
        assert cache.is_resident("sig", (2,)) is True

    def test_byte_accounting_survives_concurrent_readers(self):
        """overlay.py's off-thread source warm populates this cache from a
        compute-backend worker thread while the dispatcher reads it, so the
        OrderedDict + running byte total must stay consistent under
        concurrency (the reason cache.py takes a lock around its bookkeeping)."""
        import threading

        reader = _FakeReader(shape=(8, 8), dtype=np.uint16)   # 128 bytes/frame
        cache = ArrayCache(budget_bytes=reader.frame_bytes * 16)

        def worker(base):
            for i in range(200):
                cache.get_frame("sig", reader, ((base + i) % 40,))

        threads = [threading.Thread(target=worker, args=(b,)) for b in range(0, 40, 5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache.nbytes == sum(f.nbytes for f in cache._entries.values())
        assert cache.nbytes <= reader.frame_bytes * 17

    def test_single_entry_over_budget_still_served(self):
        reader = _FakeReader(shape=(64, 64), dtype=np.float32)  # bigger than budget
        cache = ArrayCache(budget_bytes=16)  # smaller than one frame
        frame = cache.get_frame("sig", reader, (0,))
        assert frame is not None
        assert cache.is_resident("sig", (0,)) is True
        assert len(cache) == 1

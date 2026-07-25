"""BlockCache — the byte-budgeted LRU of decoded nav-chunk blocks that replaced
each reader's one-entry memo.

The one-entry memo was correct for dwelling inside a single chunk and wrong for
everything else: crossing a chunk boundary and returning re-decoded every move
(59x on zspy, 1520x on a rebinned view), and an integrating region spanning
several chunks thrashed on every drag step. These tests pin the properties that
fix makes the read path depend on.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.array_cache import BlockCache, UNFOCUSED_BUDGET_BYTES
from spyde.array_cache.block_cache import DEFAULT_BLOCK_BUDGET_BYTES


def _block(mb, fill=0):
    return np.full(int(mb * 1024 * 1024), fill, dtype=np.uint8)


class TestBlockCacheBasics:
    def test_miss_then_hit(self):
        c = BlockCache()
        assert c.get("r", (0, 0)) is None
        b = _block(1)
        c.put("r", (0, 0), b)
        assert c.get("r", (0, 0)) is b
        assert len(c) == 1

    def test_contains_is_side_effect_free(self):
        """The read classifier probes residency before deciding sync-vs-async, so
        it must not perturb the LRU order (that would evict the wrong block)."""
        c = BlockCache(budget_bytes=3 * 1024 * 1024)
        c.put("r", "a", _block(1))
        c.put("r", "b", _block(1))
        # probing "a" must NOT make it the most-recent
        assert c.contains("r", "a")
        c.put("r", "c", _block(1))
        c.put("r", "d", _block(1))   # over budget -> evicts the true LRU ("a")
        assert not c.contains("r", "a")
        assert c.contains("r", "d")

    def test_keys_are_scoped_per_owner(self):
        """Two readers may use the same block index; they must not collide."""
        c = BlockCache()
        c.put("r1", (0,), _block(1, fill=1))
        c.put("r2", (0,), _block(1, fill=2))
        assert c.get("r1", (0,))[0] == 1
        assert c.get("r2", (0,))[0] == 2


class TestEviction:
    def test_evicts_lru_over_budget(self):
        c = BlockCache(budget_bytes=2 * 1024 * 1024)
        c.put("r", "a", _block(1))
        c.put("r", "b", _block(1))
        c.put("r", "c", _block(1))          # pushes over -> "a" goes
        assert not c.contains("r", "a")
        assert c.contains("r", "b") and c.contains("r", "c")
        assert c.nbytes <= 2 * 1024 * 1024

    def test_hit_refreshes_recency(self):
        c = BlockCache(budget_bytes=2 * 1024 * 1024)
        c.put("r", "a", _block(1))
        c.put("r", "b", _block(1))
        c.get("r", "a")                      # "a" is now MRU
        c.put("r", "c", _block(1))
        assert c.contains("r", "a")
        assert not c.contains("r", "b")

    def test_oversized_block_still_served(self):
        """A single block bigger than the whole budget must still be returned —
        matches ArrayCache's rule; the alternative is never serving it at all."""
        c = BlockCache(budget_bytes=1 * 1024 * 1024)
        big = _block(4)
        c.put("r", "big", big)
        assert c.get("r", "big") is big

    def test_region_span_stays_resident(self):
        """The concrete case this exists for: an 8x8 ROI spans up to 4 nav-chunks.
        At the real budget all 4 stay resident, so a drag step is cache hits."""
        c = BlockCache()                     # 1 GiB default
        for k in range(4):
            c.put("r", (k,), _block(33.6))   # 4D-STEM nav-chunk ~33.6 MB
        assert all(c.contains("r", (k,)) for k in range(4))

    def test_drop_owner_leaves_others(self):
        c = BlockCache()
        c.put("r1", "a", _block(1))
        c.put("r2", "a", _block(1))
        c.drop_owner("r1")
        assert not c.contains("r1", "a")
        assert c.contains("r2", "a")
        assert c.nbytes == 1024 * 1024

    def test_clear_resets_accounting(self):
        c = BlockCache()
        c.put("r", "a", _block(2))
        c.clear()
        assert len(c) == 0 and c.nbytes == 0


class TestFocusBudget:
    def test_demote_shrinks_but_keeps_working_set(self):
        """Focus loss is a DEMOTE, not a purge: a background window keeps a
        working set so returning to it is a numpy slice, not a cold re-decode."""
        c = BlockCache()
        for k in range(20):
            c.put("r", (k,), _block(33.6))
        assert c.nbytes > UNFOCUSED_BUDGET_BYTES
        c.demote()
        assert c.nbytes <= UNFOCUSED_BUDGET_BYTES
        assert len(c) >= 1                      # not purged
        # the most recently used block survived
        assert c.contains("r", (19,))

    def test_restore_returns_full_budget(self):
        c = BlockCache()
        c.demote()
        assert c.budget_bytes == UNFOCUSED_BUDGET_BYTES
        c.restore()
        assert c.budget_bytes == DEFAULT_BLOCK_BUDGET_BYTES

    def test_demote_is_idempotent(self):
        c = BlockCache()
        c.put("r", "a", _block(1))
        c.demote()
        c.demote()
        assert c.contains("r", "a")


class TestThreadSafety:
    def test_concurrent_put_keeps_accounting_consistent(self):
        """overlay.py warms frames from a worker thread while the dispatcher
        reads, so the byte bookkeeping must survive concurrent writers."""
        import threading

        c = BlockCache(budget_bytes=64 * 1024 * 1024)
        errors = []

        def worker(base):
            try:
                for k in range(40):
                    c.put(f"r{base}", (k,), _block(1))
                    c.get(f"r{base}", (k,))
            except Exception as e:      # pragma: no cover
                errors.append(e)

        ts = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]

        assert not errors
        assert c.nbytes <= 64 * 1024 * 1024
        # accounting must equal the real retained bytes
        total = sum(int(b.nbytes) for b in c._entries.values())
        assert c.nbytes == total


class TestRegionAccumulatorDtype:
    """The region accumulator must not silently lose precision.

    float32 is ~40% faster than float64 on a 512^2 frame, but it holds only 24
    bits of integer precision — exact only while n_pts * max(dtype) < 2^24. That
    holds for the common detector dtypes (256 x uint16 max = 16,776,960, just
    under 16,777,216) and NOT for int32/uint32/float64, which .hspy/.zspy can
    hold: summing 256 int32 frames in float32 loses up to ~236,148 in absolute
    value, which would show up as a visibly wrong integrated frame.
    """

    def test_uint16_at_the_cap_uses_float32(self):
        from spyde.array_cache.nav_read import _region_accum_dtype
        assert _region_accum_dtype(np.uint16, 256) == np.float32
        # ...and the margin is exactly the frame budget, so this is not slack.
        assert 256 * np.iinfo(np.uint16).max <= (1 << 24)

    def test_uint16_above_the_cap_falls_back(self):
        """If MAX_REGION_EXTENT_PER_DIM ever rises, uint16 stops being exact."""
        from spyde.array_cache.nav_read import _region_accum_dtype
        assert _region_accum_dtype(np.uint16, 4096) == np.float64

    @pytest.mark.parametrize("dt", [np.int32, np.uint32, np.float64])
    def test_wide_dtypes_use_float64(self, dt):
        from spyde.array_cache.nav_read import _region_accum_dtype
        assert _region_accum_dtype(dt, 256) == np.float64

    @pytest.mark.parametrize("dt", [np.uint8, np.int16, np.float16, np.float32])
    def test_narrow_dtypes_use_float32(self, dt):
        from spyde.array_cache.nav_read import _region_accum_dtype
        assert _region_accum_dtype(dt, 256) == np.float32

    def test_int32_region_mean_is_accurate(self):
        """End-to-end: an int32 source must not degrade through the region sum."""
        from spyde.array_cache.nav_read import _region_accum_dtype
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 2 ** 31 - 1, (256, 8, 8), dtype=np.int32)
        acc_dt = _region_accum_dtype(np.int32, 256)
        got = frames.sum(0, dtype=acc_dt) / 256
        exact = frames.sum(0, dtype=np.float64) / 256
        np.testing.assert_allclose(got, exact, rtol=1e-12)

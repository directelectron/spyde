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

"""RegionIntegrator — the threaded + incremental integrating-ROI accumulator.

A region drag on a 4096² movie cost 660 ms/step, ~500 ms of which was plain
single-threaded numpy (see spyde/array_cache/region_sum.py). The integrator splits
the accumulate into row bands and reuses the previous step's running sum when the
ROI merely slid.

Both are pure OPTIMISATIONS of a path that already had a correct answer, so the
contract these tests pin is: **bit-identical to the serial loop it replaced**, for
every dtype, every window motion, and with threading on or off. A region frame that
is silently 1 count off is exactly the kind of bug nobody would notice until it
mattered, so parity is asserted with array_equal, never allclose.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.array_cache import ArrayCache, RegionIntegrator, finalize_sum
from spyde.array_cache.cache import (
    DEFAULT_BUDGET_BYTES, REGION_BUDGET_CEILING_BYTES,
)
from spyde.array_cache.nav_read import _region_accum_dtype


class FakeReader:
    """Frames straight out of an in-memory stack; counts real reads."""

    def __init__(self, stack):
        self.stack = stack
        self.data = stack
        self.reads = 0

    @property
    def frame_bytes(self):
        return int(np.prod(self.stack.shape[1:])) * self.stack.dtype.itemsize

    def read_frame(self, indices):
        self.reads += 1
        return self.stack[tuple(int(v) for v in indices)]


class CountingCache(ArrayCache):
    """Counts frames TOUCHED, not frames read off the reader.

    That distinction is the whole point: after a 1-frame slide a full recompute
    still reads only one new frame off the reader (the other 15 are cache hits),
    so a reader-read count cannot tell a recompute from an incremental update.
    Frames touched can: 16 versus 2."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.touches = 0

    def get_frame(self, key, reader, indices, prof=None):
        self.touches += 1
        return super().get_frame(key, reader, indices, prof)


def _serial_mean(stack, points, source_dtype):
    """EXACTLY what spyde/array_cache/nav_read.py's per-frame loop does — the
    reference every fast path has to reproduce bit-for-bit."""
    n = len(points)
    acc_dtype = _region_accum_dtype(source_dtype, n)
    acc = None
    for p in points:
        f = stack[tuple(int(v) for v in p)]
        if acc is None:
            acc = np.asarray(f, dtype=acc_dtype).copy()
        else:
            acc = acc + f
    acc = acc / n
    if np.issubdtype(np.dtype(source_dtype), np.integer):
        acc = np.rint(acc).astype(source_dtype)
    return acc


def _run(integ, stack, points, cache=None, key="k"):
    reader = FakeReader(stack)
    cache = cache if cache is not None else ArrayCache()
    n = len(points)
    return integ.mean_frame(key, reader, cache, points, stack.dtype,
                            _region_accum_dtype(stack.dtype, n), None), reader


def _stack(n, edge, dtype, seed=0, lo=0, hi=4000):
    rng = np.random.default_rng(seed)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return rng.integers(lo, hi, (n, edge, edge)).astype(dtype)
    return rng.random((n, edge, edge)).astype(dtype)


def _pts(indices):
    return [(int(i),) for i in indices]


# ── parity ────────────────────────────────────────────────────────────────────
class TestParityWithTheSerialLoop:
    @pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "int32",
                                       "float32", "float64"])
    def test_full_recompute_matches(self, dtype):
        stack = _stack(24, 96, dtype, seed=1)
        pts = _pts(range(3, 19))
        got, _ = _run(RegionIntegrator(), stack, pts)
        assert got is not None
        expect = _serial_mean(stack, pts, stack.dtype)
        assert got.dtype == expect.dtype
        assert np.array_equal(got, expect), f"{dtype}: region mean drifted"

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 16, 17, 256])
    def test_every_region_size_matches(self, n):
        stack = _stack(300, 24, "uint16", seed=2)
        pts = _pts(range(n))
        got, _ = _run(RegionIntegrator(), stack, pts)
        assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))

    def test_2d_nav_points_match(self):
        """A 4D-STEM ROI: nav points are (y, x) pairs, not scalars."""
        rng = np.random.default_rng(3)
        stack = rng.integers(0, 5000, (8, 8, 32, 32)).astype(np.uint16)
        pts = [(y, x) for y in range(2, 6) for x in range(1, 5)]
        got, _ = _run(RegionIntegrator(), stack, pts)
        assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))

    def test_threading_off_gives_the_same_answer(self, monkeypatch):
        stack = _stack(24, 96, "uint16", seed=4)
        pts = _pts(range(16))
        monkeypatch.setenv("SPYDE_REGION_THREADS", "1")
        serial, _ = _run(RegionIntegrator(), stack, pts)
        monkeypatch.setenv("SPYDE_REGION_THREADS", "8")
        threaded, _ = _run(RegionIntegrator(), stack, pts)
        assert np.array_equal(serial, threaded)
        assert np.array_equal(serial, _serial_mean(stack, pts, stack.dtype))

    def test_band_split_does_not_reorder_summation(self):
        """Row banding partitions PIXELS, never the per-pixel frame order — so it
        must be exact even for float data, where addition is not associative."""
        stack = _stack(20, 64, "float64", seed=5)
        pts = _pts(range(12))
        got, _ = _run(RegionIntegrator(), stack, pts)
        assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))


# ── the incremental sliding window ────────────────────────────────────────────
class TestIncrementalSlide:
    def test_slid_window_matches_a_cold_recompute(self):
        stack = _stack(64, 96, "uint16", seed=6)
        integ = RegionIntegrator()
        cache = ArrayCache()
        reader = FakeReader(stack)
        for start in range(0, 20):
            pts = _pts(range(start, start + 16))
            got = integ.mean_frame("k", reader, cache, pts, stack.dtype,
                                   _region_accum_dtype(stack.dtype, 16), None)
            expect = _serial_mean(stack, pts, stack.dtype)
            assert np.array_equal(got, expect), f"drift at window start {start}"

    def test_slide_touches_only_the_entering_and_leaving_frames(self):
        stack = _stack(64, 96, "uint16", seed=7)
        integ, cache = RegionIntegrator(), CountingCache()
        reader = FakeReader(stack)
        acc_dt = _region_accum_dtype(stack.dtype, 16)
        integ.mean_frame("k", reader, cache, _pts(range(16)), stack.dtype,
                         acc_dt, None)
        cache.touches = 0
        integ.mean_frame("k", reader, cache, _pts(range(1, 17)), stack.dtype,
                         acc_dt, None)
        # leaving frame 0 subtracted, entering frame 16 added — nothing else.
        assert cache.touches == 2, (
            f"a 1-frame slide touched {cache.touches} frames — the running sum "
            f"is not being reused")

    def test_a_jump_with_no_overlap_still_matches(self):
        stack = _stack(64, 96, "uint16", seed=8)
        integ, cache = RegionIntegrator(), ArrayCache()
        reader = FakeReader(stack)
        acc_dt = _region_accum_dtype(stack.dtype, 8)
        integ.mean_frame("k", reader, cache, _pts(range(8)), stack.dtype, acc_dt, None)
        pts = _pts(range(40, 48))
        got = integ.mean_frame("k", reader, cache, pts, stack.dtype, acc_dt, None)
        assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))

    def test_resize_changes_the_divisor_correctly(self):
        stack = _stack(64, 96, "uint16", seed=9)
        integ, cache = RegionIntegrator(), ArrayCache()
        reader = FakeReader(stack)
        for n in (4, 9, 16, 7, 2, 16):
            pts = _pts(range(3, 3 + n))
            got = integ.mean_frame("k", reader, cache, pts, stack.dtype,
                                   _region_accum_dtype(stack.dtype, n), None)
            assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype)), \
                f"resize to {n} points"

    def test_float_source_never_goes_incremental(self):
        """Subtracting a float frame back out of a running sum does not round-trip
        (cancellation), so a float source must always recompute. Verified by the
        read count, and by the answer staying exact."""
        stack = _stack(64, 64, "float32", seed=10)
        integ, cache = RegionIntegrator(), CountingCache()
        reader = FakeReader(stack)
        acc_dt = _region_accum_dtype(stack.dtype, 16)
        integ.mean_frame("k", reader, cache, _pts(range(16)), stack.dtype, acc_dt, None)
        cache.touches = 0
        pts = _pts(range(1, 17))
        got = integ.mean_frame("k", reader, cache, pts, stack.dtype, acc_dt, None)
        assert cache.touches == 16, "float source took the incremental path"
        assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))

    def test_uint16_running_sum_is_exact_at_the_cap(self):
        """float32 holds the running sum only while n*max < 2**24. At the 16-point
        cap with saturated uint16 that is 1,048,560 — exact, and the subtract must
        not lose a count."""
        stack = np.full((32, 48, 48), 65535, np.uint16)
        stack[::2] = 65534
        integ, cache = RegionIntegrator(), ArrayCache()
        reader = FakeReader(stack)
        acc_dt = _region_accum_dtype(stack.dtype, 16)
        assert acc_dt == np.float32
        for start in range(8):
            pts = _pts(range(start, start + 16))
            got = integ.mean_frame("k", reader, cache, pts, stack.dtype, acc_dt, None)
            assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))


# ── invalidation ──────────────────────────────────────────────────────────────
class TestInvalidation:
    def test_invalidate_forces_a_recompute(self):
        stack = _stack(64, 64, "uint16", seed=11)
        integ, cache = RegionIntegrator(), ArrayCache()
        reader = FakeReader(stack)
        acc_dt = _region_accum_dtype(stack.dtype, 16)
        integ.mean_frame("k", reader, cache, _pts(range(16)), stack.dtype, acc_dt, None)
        integ.invalidate()
        before = reader.reads
        cache.clear()
        pts = _pts(range(1, 17))
        got = integ.mean_frame("k", reader, cache, pts, stack.dtype, acc_dt, None)
        assert reader.reads - before == 16
        assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))

    def test_a_different_signal_key_does_not_reuse_the_sum(self):
        a = _stack(32, 64, "uint16", seed=12)
        b = _stack(32, 64, "uint16", seed=13)
        integ, cache = RegionIntegrator(), ArrayCache()
        acc_dt = _region_accum_dtype(a.dtype, 16)
        integ.mean_frame("A", FakeReader(a), cache, _pts(range(16)), a.dtype,
                         acc_dt, None)
        pts = _pts(range(1, 17))
        got = integ.mean_frame("B", FakeReader(b), cache, pts, b.dtype, acc_dt, None)
        assert np.array_equal(got, _serial_mean(b, pts, b.dtype))

class TestDuplicatePoints:
    """A span dragged off the end of a movie clamps EVERY point to the last frame,
    so a 16-point ROI arrives as 16 copies of one index. That is a normal thing for
    a user to do, and it used to fall out of the fast path into the serial loop:
    517 ms of re-summing to display a single frame (caught in the real app by
    movie_roi_drag_perf.spec.ts, not by any headless test)."""

    @pytest.mark.parametrize("idxs", [
        [1, 2, 2, 3],                       # one repeat
        [7] * 16,                           # fully clamped — the movie-edge case
        [3, 3, 4, 4, 5, 5, 5, 9],           # mixed multiplicities
    ])
    def test_weighted_mean_matches_the_serial_loop(self, idxs):
        stack = _stack(24, 48, "uint16", seed=14)
        pts = _pts(idxs)
        got, _ = _run(RegionIntegrator(), stack, pts)
        assert got is not None, "duplicates must be served, not declined"
        assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))

    def test_float_source_with_duplicates_is_still_exact(self):
        """Repeated addition and multiplication agree for exact integers but not
        for floats in general, so a float source keeps the plain n-pass sum."""
        stack = _stack(24, 48, "float32", seed=15)
        pts = _pts([2, 2, 3, 5, 5, 5])
        got, _ = _run(RegionIntegrator(), stack, pts)
        assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))

    def test_clamped_span_reads_one_frame_not_n(self):
        stack = _stack(24, 48, "uint16", seed=16)
        integ, cache = RegionIntegrator(), CountingCache()
        reader = FakeReader(stack)
        pts = _pts([23] * 16)
        integ.mean_frame("k", reader, cache, pts, stack.dtype,
                         _region_accum_dtype(stack.dtype, 16), None)
        assert cache.touches == 1, (
            f"16 copies of one frame touched {cache.touches} frames")

    def test_duplicates_do_not_disturb_a_live_running_sum(self):
        """The weighted path must not publish into (or clear) the sliding window a
        normal drag is relying on."""
        stack = _stack(48, 48, "uint16", seed=17)
        integ, cache = RegionIntegrator(), CountingCache()
        reader = FakeReader(stack)
        acc_dt = _region_accum_dtype(stack.dtype, 16)
        integ.mean_frame("k", reader, cache, _pts(range(16)), stack.dtype,
                         acc_dt, None)
        integ.mean_frame("k", reader, cache, _pts([40] * 16), stack.dtype,
                         acc_dt, None)
        cache.touches = 0
        pts = _pts(range(1, 17))
        got = integ.mean_frame("k", reader, cache, pts, stack.dtype, acc_dt, None)
        assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))
        assert cache.touches == 2, (
            f"the running sum was lost to the duplicate read ({cache.touches} "
            f"frames touched instead of 2)")


# ── finalize_sum (the block-reader tail) ──────────────────────────────────────
class TestFinalizeSum:
    @pytest.mark.parametrize("dtype", ["uint16", "int32", "float32", "float64"])
    def test_matches_the_inline_tail(self, dtype):
        stack = _stack(12, 64, dtype, seed=15)
        n = 12
        acc_dtype = _region_accum_dtype(stack.dtype, n)
        acc = stack.astype(acc_dtype).sum(axis=0)
        expect = acc / n
        if np.issubdtype(np.dtype(dtype), np.integer):
            expect = np.rint(expect).astype(dtype)
        got = finalize_sum(acc, n, stack.dtype)
        assert got.dtype == expect.dtype
        assert np.array_equal(got, expect)


# ── the cache budget that feeds it ────────────────────────────────────────────
class TestRegionBudget:
    def test_grows_to_hold_a_big_frame_roi(self):
        c = ArrayCache()
        frame = 32 << 20                      # 4096² uint16
        c.ensure_budget_for(16, frame)
        assert c._budget_bytes > 16 * frame, (
            "the budget must exceed the window itself — the LRU evicts the "
            "leaving frame at exactly N+1 slots")
        assert c._budget_bytes <= REGION_BUDGET_CEILING_BYTES

    def test_never_drops_below_the_default(self):
        c = ArrayCache()
        c.ensure_budget_for(4, 1024)          # a tiny 4D-STEM ROI
        assert c._budget_bytes == DEFAULT_BUDGET_BYTES

    def test_is_bounded_by_the_ceiling(self):
        c = ArrayCache()
        c.ensure_budget_for(256, 64 << 20)    # 16 GiB of frames
        assert c._budget_bytes == REGION_BUDGET_CEILING_BYTES

    def test_clear_restores_the_default(self):
        c = ArrayCache()
        c.ensure_budget_for(16, 32 << 20)
        assert c._budget_bytes > DEFAULT_BUDGET_BYTES
        c.clear()
        assert c._budget_bytes == DEFAULT_BUDGET_BYTES


class TestConcurrentCallers:
    """mean_frame is NOT confined to the _NavDispatcher thread: the expensive tier
    runs get_local_frame on a compute worker (_submit_async_nav_read) and
    spyde/actions/overlay.py warms off-thread. The running sum is mutated IN
    PLACE, so a second caller must never be able to interleave a delta into it —
    and must never BLOCK waiting either (that is the retired _cache_lock_ctx
    wedge)."""

    def test_concurrent_reads_all_return_the_right_frame(self):
        import threading

        stack = _stack(80, 96, "uint16", seed=20)
        integ, cache = RegionIntegrator(), ArrayCache()
        reader = FakeReader(stack)
        acc_dt = _region_accum_dtype(stack.dtype, 16)
        errors, results = [], {}
        barrier = threading.Barrier(6)

        def worker(w):
            try:
                barrier.wait(timeout=30)
                for k in range(8):
                    start = (w * 5 + k) % 60
                    pts = _pts(range(start, start + 16))
                    got = integ.mean_frame("k", reader, cache, pts, stack.dtype,
                                           acc_dt, None)
                    want = _serial_mean(stack, pts, stack.dtype)
                    if not np.array_equal(got, want):
                        results[(w, k)] = (got, want)
            except Exception as e:                      # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(w,)) for w in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not [t for t in threads if t.is_alive()], \
            "a caller blocked — the integrator lock must be a TRY-lock"
        assert not errors, errors
        assert not results, (
            f"{len(results)} concurrent reads returned a corrupted frame — the "
            f"running sum was mutated by two threads at once")

    def test_a_contended_read_does_not_wait(self):
        """While one thread holds the sum, another must return promptly with a
        correct (just unshared) answer rather than queueing behind it."""
        import threading
        import time

        stack = _stack(32, 64, "uint16", seed=21)
        integ, cache = RegionIntegrator(), ArrayCache()
        reader = FakeReader(stack)
        acc_dt = _region_accum_dtype(stack.dtype, 8)
        pts = _pts(range(8))
        integ._lock.acquire()                    # simulate an in-flight async read
        try:
            t0 = time.perf_counter()
            got = integ.mean_frame("k", reader, cache, pts, stack.dtype, acc_dt, None)
            elapsed = time.perf_counter() - t0
            assert np.array_equal(got, _serial_mean(stack, pts, stack.dtype))
            assert elapsed < 5.0
            assert integ._sum is None, \
                "a contended read must not publish into the shared running sum"
        finally:
            integ._lock.release()


class TestGpuGating:
    """The GPU accumulator's decline rules — all decided BEFORE torch is touched,
    so they are testable without a GPU. Parity of the device path itself lives in
    test_region_integrator_gpu.py (a subprocess, per the CLAUDE.md harness note)."""

    def test_off_by_default(self, monkeypatch):
        from spyde.array_cache.region_sum_gpu import gpu_region_enabled
        monkeypatch.delenv("SPYDE_GPU_REGION", raising=False)
        assert gpu_region_enabled() is False

    @pytest.mark.parametrize("val,want", [("1", True), ("true", True),
                                          ("0", False), ("no", False)])
    def test_env_switch(self, monkeypatch, val, want):
        from spyde.array_cache.region_sum_gpu import gpu_region_enabled
        monkeypatch.setenv("SPYDE_GPU_REGION", val)
        assert gpu_region_enabled() is want

    def test_declines_a_small_frame_without_importing_torch(self, monkeypatch):
        """A 32 KB diffraction pattern is quicker to sum on the host than to ship
        to the card — and this must not pay torch's import to find that out."""
        from spyde.array_cache.region_sum_gpu import (
            GPU_MIN_FRAME_BYTES, make_gpu_accumulator)
        monkeypatch.setenv("SPYDE_GPU_REGION", "1")
        import spyde.array_cache.region_sum_gpu as rsg
        monkeypatch.setattr(rsg, "_torch_cuda", lambda: pytest.fail(
            "torch was probed for a frame below the size gate"))
        assert make_gpu_accumulator(GPU_MIN_FRAME_BYTES - 1, np.uint16,
                                    np.float32) is None

    def test_declines_a_float64_accumulator(self, monkeypatch):
        """fp64 is ~1/32 rate on the target card and a second numerical regime to
        keep in parity — int32/uint32/float64 sources stay on the CPU."""
        from spyde.array_cache.region_sum_gpu import make_gpu_accumulator
        monkeypatch.setenv("SPYDE_GPU_REGION", "1")
        import spyde.array_cache.region_sum_gpu as rsg
        monkeypatch.setattr(rsg, "_torch_cuda", lambda: pytest.fail(
            "torch was probed for a float64 accumulator"))
        assert make_gpu_accumulator(64 << 20, np.int32, np.float64) is None

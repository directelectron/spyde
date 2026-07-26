"""RegionIntegrator — turn the N nav points of an integrating ROI into ONE mean
frame, fast enough to keep up with a drag on LARGE frames.

WHY THIS EXISTS (measured 2026-07-26, 48 x 4096^2 uint16 .mrc movie, 16-frame ROI):
a drag step cost **660 ms**, and only ~166 ms of that was I/O. ~500 ms was plain
numpy — ``16 x acc(float32, 64 MiB) += frame(uint16, 32 MiB)`` plus the ``/n``,
``rint``, ``astype`` tail, all on ONE core, ~2.5 GiB of memory traffic per step.
That is why the two cache-shaped explanations both disappoint: sizing ArrayCache to
hold the ROI only gets 660 -> 460 ms, and routing the read async (Live-Display §3)
moves the 500 ms off the dispatcher without removing it. Live-Display §3 models I/O
and does not model this cost at all.

Two independent wins, both preserved bit-for-bit against the old serial path:

1. **Row-band threading.** numpy ufuncs release the GIL, so splitting the frame
   into row bands and accumulating them concurrently is a drop-in ~6.6x (502 ->
   76 ms). Each pixel still sees its frames in the SAME order, so the result is
   bit-identical — banding partitions PIXELS, never the summation order.
   It saturates around 8 threads: this is memory-bandwidth-bound, not core-bound
   (the dev box has 48 cores and gains nothing past ~8).

2. **Incremental +-1.** A dragged ROI shares most of its points with the previous
   step, so the running sum is updated by subtracting the leaving frames and adding
   the entering ones instead of re-reading and re-summing all N (502 -> 156 ms
   serial, 48 ms threaded). This is the user's own data-access model: "integration
   = cached frames + incremental +-1".

MEMORY: the full recompute streams ONE frame at a time into ONE accumulator and
fans the per-frame add out across bands with a barrier — it never materialises an
N-frame stack, so the Memory-Safety rule in CLAUDE.md holds exactly as it did for
the serial loop. The incremental path only ever holds the entering/leaving frames.

EXACTNESS: the incremental path is enabled ONLY for an INTEGER source. Every frame
value is then an exact integer in the accumulator, every partial sum stays within
the dtype's exact-integer range (:func:`_region_accum_dtype` already guarantees
``n * max < 2**24`` before it picks float32), and the leaving frames are subtracted
BEFORE the entering ones are added so no intermediate can exceed the full-window
bound. Add and subtract are therefore exact and the running sum is bit-identical to
a from-scratch one. For a FLOAT source they are not (cancellation), so a float
source always takes the full recompute.

Concurrency: one instance per Plot, driven from the serial _NavDispatcher thread
(Live-Display §2). The worker threads only touch disjoint row bands of arrays this
call owns, so there is no shared mutable state to lock.
"""
from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

log = logging.getLogger(__name__)

# Row-band threading only pays for frames big enough that a band is real work — a
# 32 KB diffraction pattern is dominated by task dispatch, so it stays serial.
THREAD_MIN_BYTES = 1 << 20          # 1 MiB/frame

# Bandwidth-bound, so this saturates well below the core count (measured: 6.6x at
# 8 threads on a 48-core box, no gain past it). SPYDE_REGION_THREADS overrides.
DEFAULT_THREADS = 8

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def _thread_count() -> int:
    try:
        n = int(os.environ.get("SPYDE_REGION_THREADS", DEFAULT_THREADS))
    except ValueError:
        n = DEFAULT_THREADS
    return max(1, min(n, os.cpu_count() or 1))


def _get_pool() -> ThreadPoolExecutor | None:
    """One shared pool for every plot — building one per call measured ~4 ms of
    the 76 ms budget. Returns None when threading is disabled (1 thread)."""
    global _pool
    n = _thread_count()
    if n <= 1:
        return None
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadPoolExecutor(n, thread_name_prefix="spyde-region")
    return _pool


def _band_bounds(nrows: int, nbands: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, nrows, nbands + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _run_bands(fn, bounds) -> None:
    """Apply ``fn(lo, hi)`` to every band, concurrently when there's a pool.

    Exceptions propagate — a partially-applied band would leave the running sum
    inconsistent, so the caller invalidates on any failure."""
    pool = _get_pool()
    if pool is None or len(bounds) == 1:
        for lo, hi in bounds:
            fn(lo, hi)
        return
    for fut in [pool.submit(fn, lo, hi) for lo, hi in bounds]:
        fut.result()


def _finalize_band(acc, out, lo, hi, n, source_dtype) -> None:
    """``acc/n`` rounded back to an integer source — the same two lines the serial
    path ran, fused into the band so the divide/round/cast reads the accumulator
    while it is still in this core's cache."""
    band = acc[lo:hi] / n
    if np.issubdtype(np.dtype(source_dtype), np.integer):
        np.rint(band, out=band)
        out[lo:hi] = band.astype(source_dtype)
    else:
        out[lo:hi] = band


def _out_dtype(source_dtype, acc_dtype):
    return source_dtype if np.issubdtype(np.dtype(source_dtype), np.integer) \
        else acc_dtype


def finalize_sum(acc, n, source_dtype):
    """Threaded ``rint(acc/n).astype(source_dtype)`` for a sum somebody else built
    — the block readers' ``sum_points`` fast path. Same tail the serial region loop
    ran, just banded: on a 4096^2 frame the divide+round+cast alone is ~3 full
    passes over 64 MiB."""
    acc = np.asarray(acc)
    out = np.empty(acc.shape, dtype=_out_dtype(source_dtype, acc.dtype))
    frame_bytes = acc.nbytes
    nbands = _thread_count() if frame_bytes >= THREAD_MIN_BYTES else 1
    bounds = _band_bounds(acc.shape[0], nbands) if acc.ndim else [(0, 1)]
    _run_bands(lambda lo, hi: _finalize_band(acc, out, lo, hi, n, source_dtype),
               bounds)
    return out


class RegionIntegrator:
    """Per-Plot integrating-ROI accumulator with a sliding-window memo.

    ``mean_frame`` is the only entry point; everything else is bookkeeping for
    the incremental path. Cleared by :meth:`invalidate` wherever the frame caches
    are cleared (node switch, data swap, plot close)."""

    def __init__(self) -> None:
        self._sum: np.ndarray | None = None       # host running sum, acc_dtype
        self._points: frozenset | None = None     # nav points currently summed in
        self._token = None                        # (key, id(reader), acc_dtype)
        self._gpu = None                          # GpuRegionAccumulator, if active
        # The running sum is mutated IN PLACE, and mean_frame is NOT confined to
        # the dispatcher: the expensive tier runs get_local_frame on a compute
        # worker (_submit_async_nav_read), and spyde/actions/overlay.py warms the
        # same cache off-thread. Two threads applying deltas to one accumulator
        # would corrupt it silently.
        #
        # This is a TRY-lock, never a blocking one. Holding a lock across a compute
        # is the retired _cache_lock_ctx mistake that wedged the navigator
        # (Live-Display §2); a contended read instead takes a STATELESS recompute
        # into its own accumulator — same answer, same threading, just no reuse.
        self._lock = threading.Lock()

    # ── public ────────────────────────────────────────────────────────────────
    def invalidate(self) -> None:
        """Drop the running sum. Safe to call from any thread — it only clears, so
        a concurrent read either sees the old sum (and finishes with it) or the
        cleared one (and recomputes)."""
        self._sum = None
        self._points = None
        self._token = None
        if self._gpu is not None:
            self._gpu.invalidate()
            self._gpu = None

    def mean_frame(self, key, reader, cache, points, source_dtype, acc_dtype,
                   prof=None):
        """Mean of the frames at ``points`` (a list of nav-index tuples), rounded
        back to ``source_dtype`` for an integer source.

        Bit-identical to the serial ``acc = f0.astype(acc); acc += fi; acc /= n;
        rint; astype`` it replaces. Returns None if it can't serve the request, so
        the caller falls back to that loop."""
        n = len(points)
        if n == 0:
            return None
        pset = frozenset(points)

        def fetch(pt):
            return cache.get_frame(key, reader, pt, None)

        if len(pset) != n:
            # Duplicate nav points — the normal set arithmetic would miscount, and
            # this is NOT a corner case: dragging a span off the end of a movie
            # clamps every point to the last frame, so a 16-frame ROI becomes 16
            # copies of frame 39. Declining here cost 517 ms of serial re-summing
            # to display ONE frame (caught by movie_roi_drag_perf.spec.ts).
            return self._duplicate_mean(fetch, points, n, source_dtype,
                                        acc_dtype, prof)

        if not self._lock.acquire(blocking=False):
            # Another thread owns the running sum right now. Recompute into a
            # private accumulator instead of waiting — still row-band threaded, it
            # just can't reuse or update the shared sum.
            return self._stateless(fetch, points, n, source_dtype, acc_dtype, prof)
        try:
            return self._locked_mean(fetch, key, reader, points, pset, n,
                                     source_dtype, acc_dtype, prof)
        finally:
            self._lock.release()

    def _duplicate_mean(self, fetch, points, n, source_dtype, acc_dtype, prof):
        """Mean over a point list containing repeats, as ``sum(count_u * frame_u)/n``
        over the UNIQUE points — one read and one pass each instead of n of both.

        Exact, and bit-identical to the serial loop's repeated additions, ONLY for
        an integer source: every value is an exact integer in the accumulator, so
        adding a frame k times and multiplying it by k give the same float. That is
        not true in general for floats, which therefore take the plain n-pass
        accumulate below.

        Touches no shared state (no running sum to reuse — the multiplicities would
        have to be tracked through every slide), so it needs no lock."""
        # Counter keeps first-insertion order, so items[0] is points[0] and the
        # probe read below doubles as its frame — fetching it twice would make a
        # fully-clamped span touch two frames to read one.
        items = list(Counter(points).items())
        probe = fetch(items[0][0])
        shape = np.shape(probe)
        bounds, _ = self._bounds(shape, source_dtype)
        try:
            if not np.issubdtype(np.dtype(source_dtype), np.integer):
                out = self._recompute(fetch, points, probe, shape, bounds, n,
                                      source_dtype, acc_dtype, publish=False)
                how = "dup-plain"
            else:
                acc = np.empty(shape, dtype=acc_dtype)
                _run_bands(lambda lo, hi, f=probe, k=items[0][1]: np.multiply(
                    f[lo:hi], k, out=acc[lo:hi], casting="unsafe"), bounds)
                for pt, k in items[1:]:
                    f = fetch(pt)
                    _run_bands(lambda lo, hi, f=f, k=k: np.add(
                        acc[lo:hi], f[lo:hi] * k, out=acc[lo:hi],
                        casting="unsafe"), bounds)
                out = self._empty_out(shape, source_dtype, acc_dtype)
                _run_bands(lambda lo, hi: _finalize_band(acc, out, lo, hi, n,
                                                         source_dtype), bounds)
                how = f"dup x{len(items)}"
        except Exception as e:
            log.debug("duplicate-point region integrate failed: %s", e)
            return None
        if prof is not None:
            prof.done(f"array-cache region x{n} {how}")
        return out

    def _stateless(self, fetch, points, n, source_dtype, acc_dtype, prof):
        probe = fetch(points[0])
        shape = np.shape(probe)
        bounds, _nbands = self._bounds(shape, source_dtype)
        try:
            out = self._recompute(fetch, points, probe, shape, bounds, n,
                                  source_dtype, acc_dtype, publish=False)
        except Exception as e:
            log.debug("stateless region integrate failed: %s", e)
            return None
        if prof is not None:
            prof.done(f"array-cache region x{n} full-unshared")
        return out

    def _bounds(self, shape, source_dtype):
        frame_bytes = int(np.prod(shape)) * np.dtype(source_dtype).itemsize
        nbands = _thread_count() if frame_bytes >= THREAD_MIN_BYTES else 1
        return (_band_bounds(shape[0], nbands) if shape else [(0, 1)]), nbands

    def _locked_mean(self, fetch, key, reader, points, pset, n, source_dtype,
                     acc_dtype, prof):
        # Plan BEFORE touching any frame: the incremental path already knows the
        # frame shape (it is the running sum's), so probing points[0] up front
        # would add a third frame touch to what should be exactly two.
        token = (key, id(reader), np.dtype(acc_dtype))
        incremental = self._plan_incremental(token, pset, source_dtype)
        probe = None
        if incremental is not None:
            shape = self._shape()
        else:
            probe = fetch(points[0])
            shape = np.shape(probe)
        frame_bytes = int(np.prod(shape)) * np.dtype(source_dtype).itemsize
        bounds, nbands = self._bounds(shape, source_dtype)

        try:
            if incremental is not None:
                entering, leaving = incremental
                if self._gpu is not None:
                    out = self._gpu.apply_delta(fetch, entering, leaving, n)
                else:
                    out = self._apply_delta(fetch, entering, leaving, bounds, n,
                                            source_dtype, acc_dtype)
                how = f"incremental +{len(entering)}-{len(leaving)}"
            else:
                self._start_recompute(frame_bytes, source_dtype, acc_dtype)
                if self._gpu is not None:
                    out = self._gpu.recompute(fetch, points, n)
                else:
                    out = self._recompute(fetch, points, probe, shape, bounds, n,
                                          source_dtype, acc_dtype)
                how = "full"
            self._token, self._points = token, pset
        except Exception as e:
            # A half-applied delta must never be reused as a running sum. A device
            # failure drops the GPU accumulator too, so the retry lands on the CPU.
            self.invalidate()
            log.debug("region integrate failed, falling back to the serial loop: %s", e)
            return None
        if prof is not None:
            prof.done(f"array-cache region x{n} {how}"
                      f"{' gpu' if self._gpu is not None else ''}"
                      f"{'' if nbands == 1 or self._gpu is not None else f' t{nbands}'}")
        return out

    # ── internals ─────────────────────────────────────────────────────────────
    def _shape(self):
        """Frame shape of whichever running sum is live (host or device)."""
        if self._gpu is not None and self._gpu.has_sum:
            return self._gpu.shape
        return self._sum.shape

    def _have_sum(self) -> bool:
        if self._gpu is not None:
            return self._gpu.has_sum
        return self._sum is not None

    def _start_recompute(self, frame_bytes, source_dtype, acc_dtype) -> None:
        """Choose the backend for a from-scratch window. Only ever swapped here —
        an incremental step must stay on whichever side already holds the sum,
        because the two live in different memories."""
        self._sum = None
        if self._gpu is not None:
            self._gpu.invalidate()
        try:
            from .region_sum_gpu import make_gpu_accumulator
            self._gpu = make_gpu_accumulator(frame_bytes, source_dtype, acc_dtype)
        except Exception as e:
            log.debug("gpu region accumulator unavailable: %s", e)
            self._gpu = None

    def _plan_incremental(self, token, pset, source_dtype):
        """(entering, leaving) when updating the running sum beats recomputing it,
        else None.

        Requires an INTEGER source: float frames cannot be subtracted back out
        exactly, and a silently drifting sum is far worse than a slow one.

        The token deliberately does NOT carry the frame shape — ``id(reader)``
        already implies it (a reader serves exactly one signal/data pair), and
        carrying it would force a probe read just to build the token. A shape
        change that somehow slipped past would raise on the first banded subtract
        and be caught by mean_frame's invalidate-and-decline."""
        if not self._have_sum() or self._points is None or self._token != token:
            return None
        if not np.issubdtype(np.dtype(source_dtype), np.integer):
            return None
        entering = pset - self._points
        leaving = self._points - pset
        # Touching every frame twice would be worse than one clean pass.
        if len(entering) + len(leaving) >= len(pset):
            return None
        return sorted(entering), sorted(leaving)

    def _apply_delta(self, fetch, entering, leaving, bounds, n, source_dtype,
                     acc_dtype):
        """Update the running sum in place, then finalize. Leaving frames are
        subtracted BEFORE entering ones are added so an intermediate sum can never
        exceed the full window's bound (which is what keeps float32 exact)."""
        acc = self._sum
        out = self._empty_out(acc.shape, source_dtype, acc_dtype)
        for pt in leaving:
            f = fetch(pt)
            _run_bands(lambda lo, hi, f=f: np.subtract(
                acc[lo:hi], f[lo:hi], out=acc[lo:hi]), bounds)
        for pt in entering:
            f = fetch(pt)
            _run_bands(lambda lo, hi, f=f: np.add(
                acc[lo:hi], f[lo:hi], out=acc[lo:hi]), bounds)
        _run_bands(lambda lo, hi: _finalize_band(acc, out, lo, hi, n,
                                                 source_dtype), bounds)
        return out

    def _recompute(self, fetch, points, first, shape, bounds, n, source_dtype,
                   acc_dtype, publish=True):
        """Full accumulate, ONE frame resident at a time.

        The per-frame band fan-out has a barrier per frame, which is what keeps
        peak memory at one frame + one accumulator instead of an N-frame stack
        (the Memory-Safety rule). The barrier is ~50 us against ~8 MiB of work per
        band, so it costs nothing measurable."""
        acc = np.empty(shape, dtype=acc_dtype)
        _run_bands(lambda lo, hi: np.copyto(acc[lo:hi], first[lo:hi],
                                            casting="unsafe"), bounds)
        for pt in points[1:]:
            f = fetch(pt)
            _run_bands(lambda lo, hi, f=f: np.add(
                acc[lo:hi], f[lo:hi], out=acc[lo:hi]), bounds)
        out = self._empty_out(shape, source_dtype, acc_dtype)
        _run_bands(lambda lo, hi: _finalize_band(acc, out, lo, hi, n,
                                                 source_dtype), bounds)
        if publish:
            self._sum = acc          # not on the contended (stateless) path
        return out

    @staticmethod
    def _empty_out(shape, source_dtype, acc_dtype):
        return np.empty(shape, dtype=_out_dtype(source_dtype, acc_dtype))

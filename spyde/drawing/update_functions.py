"""
update_functions.py

Module containing functions to update a plot based on a selector.  These functions are
called on the move or change events of a selector.

"""

import contextlib
import itertools
import logging
import sys
import threading
import time
import numpy as np
import dask
import dask.array as da
import distributed
from distributed import Future

from scipy import fft

log = logging.getLogger(__name__)

# Guards get-or-create of the per-signal cache lock below.
_CACHE_LOCK_GUARD = threading.Lock()


def _cache_lock_ctx(signal):
    """A per-signal lock guarding the ``CachedDaskArray`` critical section
    (cancel → cancel_surrounding → get_chunk → submit) in
    :func:`update_from_navigation_selection`.

    The cache's block bookkeeping (``core_cached_blocks`` / ``surrounding_*``)
    and its chunk futures are mutated/cancelled there with no internal lock. Two
    navigator updates that share one signal's cache must not run it concurrently:
    one thread cancelling a stale ``write_shared_array`` future (or surrounding
    prefetch) while another promotes that block surrounding→core and submits a
    ``get_inds`` on top of it cancels the block future its dependents need — so
    the image future dies and the frame never loads. Serialising this section
    keeps the greedy future-cancel correct. (Per-selector serialisation already
    covers a single navigator; this also covers two selectors on one signal.)
    """
    with _CACHE_LOCK_GUARD:
        lk = getattr(signal, "_spyde_nav_cache_lock", None)
        if lk is None:
            lk = threading.Lock()
            try:
                signal._spyde_nav_cache_lock = lk
            except Exception:
                lk = None
    return lk if lk is not None else contextlib.nullcontext()

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spyde.drawing.selectors import BaseSelector
    from spyde.drawing.plots.plot import Plot
from multiprocessing import shared_memory
import os as _os

_SHARED_MEMORY_SUPPORTED = True

# Per-frame navigator diagnostics ('NAV-DEBUG enter' / timing) fire on EVERY
# crosshair move. At DEBUG with a fast drag they flood the stdout IPC pipe
# (shared with figure pushes) and add visible lag. Gate them behind an opt-in
# env flag so a normal DEBUG session stays responsive; set SPYDE_NAV_TIMING=1
# to turn the navigator trace back on.
_NAV_TIMING = _os.environ.get("SPYDE_NAV_TIMING") == "1"

# Per-frame UPDATE PROFILE: logs ONE compact timing line per navigator update, at
# INFO (so it reaches the Log panel without full DEBUG). It breaks the per-frame
# cost into stages — read (cache/disk), dtype round, prefetch prime, LOD decimate,
# contrast levels, transport (anyplotlib set_data → base64 → stdout emit) — so a
# "the update is slow" report shows exactly WHICH stage dominates. Toggle it LIVE
# from the Log panel's "Profile" button (or SPYDE_NAV_PROFILE=1 at startup) — the
# state lives in backend.debug_flags.nav_profile_on(), read fresh each frame. Kept
# separate from _NAV_TIMING (the noisy per-move index/cache trace). See NavProfile.
from de_shell.debug_flags import nav_profile_on as _nav_profile_on


class NavProfile:
    """Accumulates per-stage timings for one navigator update and logs a single
    compact line. No-op unless nav profiling is on, so it's free in normal use.

    Usage:
        prof = NavProfile("SIG", indices)
        with prof.stage("read"): frame = ...
        with prof.stage("transport"): plot.update_data(frame)
        prof.done(extra="cache_hit")   # emits the line
    """

    __slots__ = ("_on", "_label", "_idx", "_stages", "_t0", "_frame_shape")

    def __init__(self, label: str, indices=None) -> None:
        self._on = _nav_profile_on()
        self._label = label
        self._idx = None
        self._stages: "list[tuple[str, float]]" = []
        self._t0 = time.perf_counter() if self._on else 0.0
        self._frame_shape = None
        if self._on and indices is not None:
            try:
                self._idx = np.asarray(indices).ravel().tolist()
            except Exception:
                self._idx = None

    def stage(self, name: str):
        """Context manager timing one stage. Returns a nullcontext if profiling
        is off, so callers pay nothing."""
        if not self._on:
            return contextlib.nullcontext()
        return _StageTimer(self, name)

    def _record(self, name: str, dt: float) -> None:
        self._stages.append((name, dt))

    def set_frame(self, arr) -> None:
        if self._on:
            self._frame_shape = getattr(arr, "shape", None)

    def done(self, extra: str = "") -> None:
        if not self._on:
            return
        total = (time.perf_counter() - self._t0) * 1e3
        parts = "  ".join(f"{n}={dt*1e3:.1f}" for n, dt in self._stages)
        idx = f" idx={self._idx}" if self._idx is not None else ""
        shp = f" frame={self._frame_shape}" if self._frame_shape is not None else ""
        ex = f" {extra}" if extra else ""
        # INFO so a "report this during normal use" line reaches stderr / the Log
        # panel. One line per update; the total plus the per-stage ms in order.
        log.info("[NAV-PROFILE] %s total=%.1fms  %s%s%s%s",
                 self._label, total, parts, idx, shp, ex)


class _StageTimer:
    __slots__ = ("_prof", "_name", "_t")

    def __init__(self, prof: "NavProfile", name: str) -> None:
        self._prof = prof
        self._name = name
        self._t = 0.0

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._prof._record(self._name, time.perf_counter() - self._t)
        return False


def _nav_cache_was_hit(signal, indices) -> bool:
    """Profile-line-only variant of :func:`_nav_cache_is_resident`: skip the probe
    entirely unless nav profiling is on (the classifier needs the real probe every
    read; the profile line does not). Used only to tag a nav-update timing line as
    a cache HIT vs a cold MISS."""
    if not _nav_profile_on():
        return False
    return _nav_cache_is_resident(signal, indices)


class _MoviePrefetcher:
    """Warm the OS page cache for the frames a movie scrub is about to reach.

    A cold single-frame read of a large in-situ movie is disk-bound (~50 ms);
    once the file pages are in the OS cache a re-read is ~18 ms.
    After the navigator paints frame ``t`` this reads a few upcoming frames
    (``t±1 … t±radius``) on a single background daemon thread, purely to trigger
    the page-in — so a steady scrub/playback finds each next frame already warm.

    Safety: it reads the **given lazy dask array directly**
    (``arr[i].compute(scheduler="synchronous")``), NOT the ``CachedDaskArray`` the
    navigator read uses — so it never touches hyperspy's (non-concurrency-safe)
    cache bookkeeping (CLAUDE.md §4). The OS page cache it warms is process-global
    and thread-safe. Latest-center-wins: a new ``prime`` replaces the pending
    target set, so a fast scrub doesn't pile up stale reads.

    It serves both the raw movie read (cheap tier) AND a DERIVED view (rebin/crop)
    scrubbed through the async tier: priming with the derived array reads
    ``derived[i]``, which pulls that output frame's SOURCE chunk through disk — so
    the source pages the next re-decode needs are already warm. (Reading a derived
    neighbour re-runs its transform too, but that CPU is discarded background work;
    the disk warm is the win.)
    """

    def __init__(self, radius: int = 3) -> None:
        self._radius = radius
        self._lock = threading.Lock()
        self._raw = None            # the raw dask array (movie frames)
        self._center = 0
        self._n = 0
        self._pending = False
        self._wake = threading.Event()
        self._thread = None

    def prime(self, raw, center: int, n_time: int) -> None:
        """Queue a prefetch around ``center`` (latest-wins). No-op if disabled."""
        with self._lock:
            self._raw = raw
            self._center = int(center)
            self._n = int(n_time)
            self._pending = True
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="movie-prefetch", daemon=True)
                self._thread.start()
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                self._wake.clear()
                if not self._pending:
                    continue
                self._pending = False
                raw, center, n = self._raw, self._center, self._n
            if raw is None or n <= 0:
                continue
            # Read outward from the center: t+1, t-1, t+2, … to `radius`.
            order = []
            for d in range(1, self._radius + 1):
                order.append(center + d)
                order.append(center - d)
            for i in order:
                if self._wake.is_set():
                    break               # a newer center arrived — abandon this set
                if 0 <= i < n:
                    try:
                        # Touch the frame to page it into the OS cache. The result
                        # is discarded; we only want the disk read to have happened.
                        raw[int(i)].compute(scheduler="synchronous")
                    except Exception as e:
                        log.debug("movie prefetch of frame %d failed: %s", i, e)


# One prefetcher for the whole process (movie navigation is serial).
_movie_prefetcher = _MoviePrefetcher()


class _BlockPrefetcher:
    """Warm the nav-CHUNK block a 2-D drag is heading into, off the dispatcher.

    With the block cache in place a region drag is ~5 ms/step — EXCEPT on the few
    steps that cross into an undecoded nav-chunk, which cost ~100 ms (measured: 4
    decodes across a 120-step squiggle, median 103 ms vs 5.5 ms otherwise). Those
    are the only remaining stalls, and they are predictable: a drag has velocity,
    so the chunk it is about to enter can be decoded before it is asked for.

    Extrapolates the last step one ROI-length ahead, and decodes that block through
    the SAME reader — so the result lands in the plot's BlockCache and the
    dispatcher's next read is a hit. Latest-target-wins (a newer position replaces
    the pending one), single daemon thread, and every failure is swallowed: this is
    pure speculation, so being wrong must cost nothing but wasted background work.

    Distinct from _MoviePrefetcher, which warms the OS page cache for 1-D time
    scrubs; this warms the DECODED-block cache for 2-D nav. Both are latest-wins.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._target = None         # (plot, signal, data, point)
        self._pending = False
        self._wake = threading.Event()
        self._thread = None

    def prime(self, plot, signal, data, prev_point, point, reach: int) -> None:
        """Queue a speculative decode of the block ``reach`` positions along the
        travel direction. No-op without a previous point (no velocity yet)."""
        if plot is None or prev_point is None or point is None:
            return
        try:
            if len(prev_point) != len(point) or len(point) < 2:
                return              # 2-D nav only; 1-D is _MoviePrefetcher's job
            delta = [int(point[k]) - int(prev_point[k]) for k in range(len(point))]
            if not any(delta):
                return              # not moving — nothing to guess
            nav_shape = data.shape[:len(point)]
            ahead = tuple(
                int(np.clip(int(point[k]) + delta[k] * reach, 0, nav_shape[k] - 1))
                for k in range(len(point)))
        except Exception:
            return
        with self._lock:
            self._target = (plot, signal, data, ahead)
            self._pending = True
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="nav-block-prefetch", daemon=True)
                self._thread.start()
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                self._wake.clear()
                if not self._pending:
                    continue
                self._pending = False
                target = self._target
            if target is None:
                continue
            plot, signal, data, point = target
            try:
                from spyde.array_cache import (
                    get_local_frame, is_local_frame_resident,
                )
                if is_local_frame_resident(plot, signal, data, np.asarray(point)):
                    continue                    # already warm
                get_local_frame(plot, signal, data, np.asarray(point))
            except Exception as e:
                log.debug("nav block prefetch at %s failed: %s", point, e)


_block_prefetcher = _BlockPrefetcher()


class _InteractiveActivity:
    """Lets a heavy MAIN-PROCESS background disk reader YIELD to interactive
    navigation. For a large movie the threaded navigator sum reads the WHOLE file
    from disk on a background thread (tens of seconds); that read saturates disk
    bandwidth and starves the crosshair's own per-frame read, so the signal plot
    appears frozen while the navigator fills ("plot doesn't update while the
    navigator computes").

    The nav read `poke()`s this on every move; the background fill calls
    `wait_if_active()` between chunks, which blocks briefly while scrubbing is
    recent so the interactive frame read gets the disk first. The fill is only
    slowed while the user is actively moving — it resumes as soon as they pause.

    Scope: this only helps a reader that (a) runs in THIS process and (b) has a
    per-chunk loop to yield between — i.e. the THREADED progressive navigator fill
    (`_bg_nav`, no-cluster path). The DISTRIBUTED progressive fill reads on the
    Dask worker PROCESSES (a main-thread yield can't throttle them), and the
    single-shot VI fallback (`stream_progressive_to_plot`, client is None) is one
    blocking `.compute()` with nothing to yield between — neither is preempted by
    this. Pass a stop event to abort the wait promptly on teardown if needed."""

    def __init__(self, quiet_s: float = 0.35) -> None:
        self._quiet = quiet_s
        self._last = 0.0            # monotonic time of the last interactive poke
        self._lock = threading.Lock()

    def poke(self) -> None:
        with self._lock:
            self._last = time.monotonic()

    def wait_if_active(self, max_wait_s: float = 2.0, stop=None) -> None:
        """Block while interactive activity is recent (up to ``max_wait_s`` PER
        CALL so a continuous drag can't starve the fill forever — note the caller
        loops this per chunk, so under a sustained drag the fill advances one chunk
        per ``max_wait_s``). Returns immediately if ``stop`` is set, so a torn-down
        fill aborts the wait at once instead of lingering up to ``max_wait_s``."""
        deadline = time.monotonic() + max_wait_s
        while True:
            if stop is not None and stop.is_set():
                return
            with self._lock:
                idle = time.monotonic() - self._last
            if idle >= self._quiet or time.monotonic() >= deadline:
                return
            time.sleep(0.05)


# Process-wide: the navigator/VI fill yields the disk to active scrubbing.
_interactive_activity = _InteractiveActivity()


# ── Cheap vs expensive nav-read classification (Live-Display §3, tiered read) ──
# A CHEAP read is served synchronously on the serial dispatcher and paints in the
# same call (single point / small dwell-in-chunk / small region). An EXPENSIVE
# read (large region, cold cross-chunk LARGE frame, or a derived rebin/crop node
# whose per-frame graph is heavy) is submitted off the dispatcher via
# ComputeBackend.submit_graph so it never blocks the navigator and is cancelled on
# supersede. Thresholds are deliberately conservative — a false "cheap" is just a
# brief stall (recovered by the settle re-fire); a false "expensive" costs one
# callback hop on a frame that could've painted inline.
# A region integrating MORE than this many frames goes async — the ONE case that
# genuinely freezes the navigator if done on the dispatcher: a maxed 4D-STEM region
# (16×16 = 256 DP frames) is ~0.5–0.9 s synchronously. A maxed movie region (16
# frames of 512² ≈ 64 ms) and everything smaller stays synchronous + chunk-cached.
# (Single-point reads — including derived rebin/crop/.zspy views — are ALWAYS
# synchronous now: the transform recompute is only ~5–9 ms and the decoded-chunk
# cache makes dwell-in-chunk ~0 ms, so the async round-trip's ~15–40 ms + 4 thread
# hops was pure overhead. See spyde/array_cache.)
REGION_FRAME_CAP = 4              # ≤ this many frames: definitely synchronous
REGION_ASYNC_FRAME_CAP = 48      # > this many frames: async (would freeze otherwise)
REGION_BYTES_CAP = 128 * 2 ** 20  # …or more than this many total bytes → async
COLD_FRAME_CAP = 8 * 2 ** 20      # a cache-MISS single LARGE frame → async (rare)


def _classify_nav_read(current_signal, indices, data, frame_bytes, child=None):
    """Return ``"cheap"`` or ``"expensive"`` for this lazy nav read. O(1)-ish and
    side-effect-free — NEVER triggers a compute (it only inspects index shapes and
    the cache). Called on the serial dispatcher thread before the read, so an
    expensive read can be routed off-dispatcher instead of blocking it.

    Async fires ONLY for the reads that would genuinely FREEZE the navigator:
      * a large region — > REGION_ASYNC_FRAME_CAP frames or > REGION_BYTES_CAP bytes
        (a maxed 4D-STEM region is ~0.5–0.9 s synchronously);
      * a single read that MISSES the chunk cache AND whose frame is bigger than
        COLD_FRAME_CAP (a cold read of a huge frame on a cached signal is seconds).
    Everything else — single points (incl. DERIVED rebin/crop/.zspy views, which are
    now synchronous + decoded-chunk-cached) and small/medium regions — is cheap.
    Returns ``"cheap"`` on any probe failure — the synchronous path is always
    correct, just potentially slow; the async path is the optimisation."""
    try:
        idx = np.asarray(indices)
        is_region = idx.ndim > 1
        if is_region:
            # Size the region by the frames that would actually be READ, not by
            # its total extent: a dragged ROI shares most of its points with the
            # previous step, so an already-served region is nearly free. Falls
            # back to the total count when there's no cache to probe.
            n_read = _region_uncached_count(child, current_signal, data, idx)
            # "expensive" now means "run it OFF the dispatcher", NOT "skip the
            # cache" — the async path goes through get_local_frame too. So the
            # question is only whether the read would BLOCK long enough to make
            # the ROI stop tracking the cursor, and a read that is mostly cache
            # hits does not, however many points it has.
            #
            # Getting this wrong in either direction is visible: too eager and a
            # trivially-cheap region pays an async round-trip; too reluctant and a
            # 4D-STEM ROI blocks the serial dispatcher for tens of ms per drag
            # step, so the ROI lags behind the cursor even though the throughput
            # numbers look fine (that regression is why this comment exists).
            if n_read * frame_bytes > REGION_BYTES_CAP:
                return "expensive"
            # Many UNCACHED points is also blocking even when the bytes are small
            # — each one is a separate read with its own per-point overhead.
            if n_read > REGION_ASYNC_FRAME_CAP:
                return "expensive"
            return "cheap"

        # Single point. A cached signal's cold read of a HUGE frame is the only
        # single-point case worth the async round-trip; derived views (no cache) are
        # served synchronously from the decoded-chunk cache.
        #
        # "Cold" now means cold in BOTH caches. The per-frame read is served by the
        # plot's ArrayCache (spyde/array_cache), which leaves hyperspy's
        # CachedDaskArray empty — so _nav_cache_is_resident alone would call every
        # big frame cold and pay the async round-trip even for one that ArrayCache
        # can hand back as a dict lookup.
        cached_arr = getattr(current_signal, "cached_dask_array", None)
        if cached_arr is not None and frame_bytes > COLD_FRAME_CAP \
                and not _nav_cache_is_resident(current_signal, indices) \
                and not _array_cache_is_resident(child, current_signal, data, indices):
            return "expensive"
        return "cheap"
    except Exception:
        return "cheap"


def _region_uncached_count(child, current_signal, data, idx) -> int:
    """How many of a region's points would need a REAL read? Side-effect-free
    (never LRU-touches, never reads), so the classifier can call it before
    deciding sync-vs-async. Returns the full point count when there's no cache to
    probe or on any failure — the conservative answer that preserves the old
    extent-based behaviour."""
    n_pts = int(idx.shape[0])
    if child is None:
        return n_pts
    try:
        from spyde.array_cache import is_local_frame_resident
        n = 0
        for i in range(n_pts):
            if not is_local_frame_resident(child, current_signal, data, idx[i]):
                n += 1
        return n
    except Exception:
        return n_pts


def _array_cache_is_resident(child, current_signal, data, indices) -> bool:
    """Would the ArrayCache path serve ``indices`` without a real read? False for
    a missing plot / an ArrayCache-ineligible (opaque) signal / any probe failure
    — always the conservative answer, since a false "resident" only costs a brief
    stall the settle re-fire recovers from."""
    try:
        from spyde.array_cache import is_local_frame_resident
        return is_local_frame_resident(child, current_signal, data, indices)
    except Exception:
        return False


def _nav_cache_is_resident(signal, indices) -> bool:
    """Was the chunk for ``indices`` already resident in the signal's
    CachedDaskArray before this read? A resident block is a ~ms numpy slice; a
    non-resident one reads the chunk off disk. The classifier calls this every
    read (unlike the profile-gated :func:`_nav_cache_was_hit` wrapper). Cheap +
    side-effect-free — only inspects the cache's block-index list; returns False
    if it can't tell (no cache / probe failed), which makes a cold read classify
    expensive only when the frame is also large."""
    try:
        cache = getattr(signal, "cached_dask_array", None)
        if cache is None or not getattr(cache, "core_cached_block_inds", None):
            return False
        from hyperspy.misc.array_tools import _get_navigation_dimension_chunk_slice
        nav_dim = len(signal.axes_manager.navigation_axes)
        inds = np.asarray([row[:nav_dim] for row in np.atleast_2d(indices)])
        core, _surr, _by = _get_navigation_dimension_chunk_slice(
            inds, cache.array.chunks, cache.cache_padding)
        return all(c in cache.core_cached_block_inds for c in core)
    except Exception:
        return False


def _direct_read_frame(current_signal, selector, indices, prof, child=None):
    """Unified fast VIEW read: compute the requested nav slice DIRECTLY with the
    synchronous scheduler and return the ndarray — bypassing hyperspy's
    ``CachedDaskArray``/``get_index`` machinery, which adds ~160 ms/frame of pure
    overhead and balloons to seconds on a cold miss. A direct
    ``raw[idx].compute(scheduler="synchronous")`` of the same slice is ~2–30 ms and
    byte-identical (profiled: movie 179→25 ms, 4D-STEM DP 10→2 ms, region 9→7 ms).
    ``CachedDaskArray``/``get_index`` machinery, which adds ~160 ms/frame of pure
    overhead and balloons to seconds on a cold miss. A direct
    ``raw[idx].compute(scheduler="synchronous")`` of the same slice is ~2–30 ms and
    byte-identical (profiled: movie 179→25 ms, 4D-STEM DP 10→2 ms, region 9→7 ms).

    Handles BOTH navigator shapes, using the SAME index semantics as the eager
    branch below:
      * single point (``idx.ndim<=1``) — a movie frame OR a 4D-STEM diffraction
        pattern: ``data[tuple(point)]`` → native dtype, no rounding.
      * integrating region (``idx.ndim>1``, N nav points) — ``data[sl].mean(axis=0)``
        rounded back to an integer source dtype (parity with the old distributed
        ``weighted_mean_round_from_sums``).

    Works on ANY lazy signal including DERIVED views (rebin / crop / rechunk / .zspy)
    — those have no ``CachedDaskArray`` at all, so the direct read is the only path
    that serves them. Memory stays bounded: a single frame peaks ~1 frame even on a
    monolithic chunk, and a region mean is accumulated INCREMENTALLY (one frame at a
    time into a float sum) so peak stays ~1 frame regardless of region size — no cap.

    Returns None to fall through to the cached ``get_index`` read when it can't serve
    the request (eager data or any failure) — a safety net, not the primary path.
    Also primes the movie prefetcher + drives the profile line so the caller doesn't
    repeat that work.

    Concurrency: issued only from the serial ``_NavDispatcher`` thread; it never
    touches the ``CachedDaskArray`` bookkeeping, so the "(i,j) is not in list" hazard
    (the reason that path had to be serial) does not apply here (CLAUDE.md §4)."""
    try:
        data = getattr(current_signal, "data", None)
        # Needs a lazy dask array (has .compute + .chunks). Eager numpy is handled
        # by the eager branch; a Future-bearing array is the loading placeholder.
        if data is None or not hasattr(data, "compute") or not hasattr(data, "chunks"):
            return None

        idx = np.asarray(indices)
        item_bytes = data.dtype.itemsize
        nav_dim = current_signal.axes_manager.navigation_dimension
        frame_shape = data.shape[nav_dim:]
        frame_bytes = int(np.prod(frame_shape)) * item_bytes

        is_region = idx.ndim > 1
        with prof.stage("read"):
            # BOTH shapes go through the array cache first — a single point and an
            # integrating region are the same question ("give me these frames"), so
            # they share one path. A region used to bypass this entirely and pay a
            # dask compute PER POINT, each materialising the whole enclosing
            # nav-chunk to keep one frame: ~64 chunk decodes for the ~4 chunks an
            # 8x8 ROI actually spans (measured 2.9 s/step; through the cache it is
            # ~4 ms). Falls through to the plain compute below when the cache
            # declines (an opaque node, eager data, no signal_tree).
            frame = None
            if child is not None:
                try:
                    from spyde.array_cache import get_local_frame
                    cframe = get_local_frame(child, current_signal, data, idx, prof)
                    if cframe is not None:
                        frame = np.asarray(cframe)
                except Exception as _ce:
                    log.debug("array cache read failed, direct read: %s", _ce)
                    frame = None

            if frame is None and not is_region:
                point = tuple(int(v) for v in np.atleast_1d(idx))
                frame = np.asarray(data[point].compute(scheduler="synchronous"))
            elif frame is None:
                # Integrating region fallback: N nav points → frame-wise mean,
                # accumulated INCREMENTALLY (read one frame, add to a float
                # accumulator, free it) so peak memory is ~ONE frame, not the whole
                # block. Reading a 59-frame region as one vindex block peaked ~2 GB;
                # incremental peaks ~1 frame at the SAME speed (it's disk-bound
                # either way). No size cap needed — memory is bounded by construction.
                coords = [idx[:, k].astype(int) for k in range(idx.shape[1])]
                n_pts = int(idx.shape[0])
                acc = None
                for i in range(n_pts):
                    pt = tuple(int(coords[k][i]) for k in range(len(coords)))
                    f = np.asarray(data[pt].compute(scheduler="synchronous"))
                    if acc is None:
                        acc = f.astype(np.float64)
                    else:
                        acc += f
                mean = acc / n_pts
                # Parity with the old distributed region mean: round an integer
                # source's fractional mean back to its dtype.
                if np.issubdtype(data.dtype, np.integer):
                    mean = np.rint(mean).astype(data.dtype)
                frame = mean
        prof.set_frame(frame)

        # DIAGNOSTIC: resolution of the frame straight off disk (before ANY painting).
        # If an 82² centre crop has few distinct values, the FILE's frame is blocky —
        # not a display bug. Compare to the initial-load frame's count.
        try:
            _f = np.asarray(frame)
            if _f.ndim == 2 and min(_f.shape) >= 200 \
                    and log.isEnabledFor(logging.DEBUG):
                _cy, _cx = _f.shape[0] // 2, _f.shape[1] // 2
                _crop = _f[_cy - 41:_cy + 41, _cx - 41:_cx + 41]
                log.debug(
                    "[TILEDBG] _direct_read_frame OFF-DISK shape=%s dtype=%s "
                    "center82_distinct=%d region=%s",
                    _f.shape, _f.dtype, int(np.unique(_crop).size), is_region)
        except Exception as _de:
            log.debug("off-disk probe failed: %s", _de)

        # Read-ahead. TWO kinds, by nav shape:
        #  * 1-D time scrub → _movie_prefetcher warms the OS page cache for the
        #    neighbouring frames.
        #  * 2-D nav (4D-STEM point or region) → _block_prefetcher decodes the
        #    nav-CHUNK the drag is heading into. With the block cache resident that
        #    is the ONLY remaining stall (measured: 4 decodes in a 120-step
        #    squiggle, ~103 ms each vs ~5.5 ms otherwise), and it is predictable
        #    from the travel direction.
        with prof.stage("prefetch"):
            try:
                if nav_dim == 1 and idx.ndim <= 1:
                    n_time = int(data.shape[0])
                    center = int(np.atleast_1d(idx).ravel()[0])
                    _movie_prefetcher.prime(data, center, n_time)
                elif child is not None and nav_dim >= 2:
                    # Guess from the ROI's leading corner (a region) or the point.
                    cur = (tuple(int(v) for v in idx.min(axis=0)) if is_region
                           else tuple(int(v) for v in np.atleast_1d(idx)))
                    prev = getattr(child, "_prefetch_prev_point", None)
                    child._prefetch_prev_point = cur
                    # Reach one region-extent ahead so the block is decoded before
                    # the ROI's leading edge arrives; a point gets a short lookahead.
                    reach = int(np.ptp(idx[:, 0])) + 1 if is_region else 2
                    _block_prefetcher.prime(child, current_signal, data,
                                            prev, cur, max(2, reach))
            except Exception as _e:
                log.debug("prefetch prime failed: %s", _e)
        prof.done("direct")
        return frame
    except Exception as _e:
        # WARNING, not debug: this fallback is ~100x slower than the direct read
        # (measured 220 ms vs 2.8 ms on a real 4D-STEM scrub), so it silently
        # turning a fast path into a slow one is exactly the kind of failure that
        # must not hide at debug level. A user-driven profile capture showed 66 of
        # 223 reads taking it — including 26 that had ALREADY paid for a successful
        # array-cache read and then threw, discarding the result.
        log.warning("direct-read FAILED (falling back to the ~100x slower cached "
                    "get_index path): %r", _e, exc_info=True)
        return None


def _build_nav_lazy_slice(current_signal, indices):
    """Build the LAZY dask expression for a nav frame WITHOUT computing it — the
    async (expensive-tier) counterpart of _direct_read_frame's read. Same index
    semantics: a single point → ``data[point]`` (native dtype); an integrating
    region → ``mean over the points``, rounded back to an integer source dtype for
    parity with the synchronous path. Returns the lazy dask array, or None if the
    signal isn't a lazy dask array."""
    data = getattr(current_signal, "data", None)
    if data is None or not hasattr(data, "compute") or not hasattr(data, "chunks"):
        return None
    idx = np.asarray(indices)
    if idx.ndim <= 1:
        point = tuple(int(v) for v in np.atleast_1d(idx))
        return data[point]
    # Region: stack the selected frames and mean over them as a graph op (dask
    # schedules this chunk-wise; the extent cap bounds it to <=256 frames anyway).
    coords = [idx[:, k].astype(int) for k in range(idx.shape[1])]
    n_pts = int(idx.shape[0])
    frames = [data[tuple(int(coords[k][i]) for k in range(len(coords)))]
              for i in range(n_pts)]
    mean = da.stack(frames, axis=0).mean(axis=0)
    if np.issubdtype(data.dtype, np.integer):
        mean = da.round(mean).astype(data.dtype)
    return mean


def _submit_async_nav_read(child, current_signal, indices, settle, prof):
    """EXPENSIVE-tier nav read: submit the frame as a cancellable submit_graph
    future OFF the serial dispatcher, cancelling any prior in-flight read for this
    plot (supersede). Paints from the future's done-callback, marshalled to the
    main thread; holds the last good frame until it lands. Returns True if it armed
    an async read (caller must NOT paint), False to fall back to the sync path.

    All of the submit / cancel bookkeeping runs on the ONE serial dispatcher
    thread, so no lock and no generation counter are needed — a superseded read is
    identified by the ``child._nav_future is not fut`` identity check in the
    callback (Live-Display §3)."""
    session = getattr(child, "session", None)
    if session is None:
        return False
    try:
        backend = session.compute_backend
    except Exception as e:
        log.debug("no compute backend for async nav read: %s", e)
        return False
    if backend is None:
        # Session shut down (or no backend available) → fall through to the
        # synchronous read, which is always correct.
        return False

    # Prefer the CACHED read, just executed off the dispatcher. Async used to mean
    # "build a dask graph", which bypassed the array cache entirely — so a region
    # routed async could never warm and re-read everything on every drag step.
    # Going through get_local_frame instead means async and cached are no longer
    # mutually exclusive: the navigator stays responsive AND the blocks stay warm.
    # Falls back to the lazy dask slice when the cache declines (opaque node, eager
    # data, no signal_tree).
    _idx = np.asarray(indices)
    cached_read = None
    if child is not None and getattr(child, "signal_tree", None) is not None:
        try:
            from spyde.array_cache import get_local_frame
            data_ref = current_signal.data
            # Decide HERE, not in the callback: only take this path if the cache
            # will actually serve this signal. Discovering "declined" after the
            # round-trip would mean painting nothing for that move.
            if child.signal_tree.resolve_locality(current_signal):
                def cached_read(_sig=current_signal, _data=data_ref, _i=_idx):
                    out = get_local_frame(child, _sig, _data, _i)
                    return None if out is None else np.asarray(out)
        except Exception as e:
            log.debug("async cached-read setup failed: %s", e)
            cached_read = None

    lazy = None
    if cached_read is None:
        lazy = _build_nav_lazy_slice(current_signal, indices)
        if lazy is None:
            return False

    # Supersede: cancel the prior in-flight expensive read for this plot. A queued
    # future cancels cleanly; an already-running one runs to completion but its
    # callback then no-ops via the identity + sequence checks.
    prev = child._nav_future
    if prev is not None:
        try:
            prev.cancel()
        except Exception:
            pass

    # Monotonically increasing submit order for this plot, so a read that was
    # superseded AFTER it started running can be dropped instead of painting an
    # older frame over a newer one (see _apply). Assigned on the one serial
    # dispatcher thread, so a plain increment is enough — no lock.
    my_seq = getattr(child, "_nav_read_seq", 0) + 1
    child._nav_read_seq = my_seq

    try:
        fut = (backend.submit_nav_read(cached_read) if cached_read is not None
               else backend.submit_graph(lazy))
    except Exception as e:
        log.debug("submit for async nav read failed: %s", e)
        return False
    child._nav_future = fut
    # NB: we deliberately do NOT set child.current_data = fut. The nav read's
    # supersede/staleness is tracked solely by the _nav_future identity check
    # below (the PlotUpdateWorker only polls raw dask.distributed.Futures, not the
    # concurrent.futures.Future / adapter submit_graph returns, so it would never
    # act on it anyway). Leaving current_data on the last-good ndarray means an
    # interleaved non-nav repaint (e.g. axis recalibration) still paints a real
    # frame instead of no-oping on a Future — and holds the last frame (no flash)
    # until the async result lands.

    # Source-chunk warming for a 1-D DERIVED-view scrub (rebin/crop of a movie):
    # computing a neighbouring OUTPUT frame off-thread pulls its SOURCE chunk
    # through disk, so the OS page cache is warm when the user scrubs there and the
    # next re-decode reads warm pages (~18 ms) instead of cold (~50 ms). We prime
    # with the DERIVED array itself (reading derived[i] warms exactly the source
    # deps of frame i). Safe: the prefetcher reads via a plain
    # ``arr[i].compute(scheduler="synchronous")`` on a background daemon, never the
    # CachedDaskArray (§4). Single-point 1-D only — a region has no single "next".
    try:
        idx0 = np.asarray(indices)
        nav_dim = current_signal.axes_manager.navigation_dimension
        data = current_signal.data
        if nav_dim == 1 and idx0.ndim <= 1:
            center = int(np.atleast_1d(idx0).ravel()[0])
            _movie_prefetcher.prime(data, center, int(data.shape[0]))
    except Exception as e:
        log.debug("source-chunk warm prime failed: %s", e)

    def _apply(f, plot=child, expected=fut, want_settle=settle, seq=my_seq):
        # Superseded before we even ran → drop (a newer read owns the slot).
        if getattr(plot, "_nav_future", None) is not expected:
            return
        # MONOTONIC GUARD. The identity check above is not sufficient on its own:
        # fut.cancel() only takes effect on a QUEUED future, so a superseded read
        # that already STARTED runs to completion and can land after a newer one,
        # painting an older frame — the display visibly jumps backwards mid-drag.
        # Never paint a frame older than the newest one already painted.
        if seq < getattr(plot, "_nav_painted_seq", 0):
            return
        try:
            result = f.result()
        except Exception:
            # Cancelled / failed → hold the last good frame, but RELEASE the slot
            # (still ours) so a wedged dead future doesn't defeat the next read's
            # supersede-cancel. current_data was never set to the future, so a
            # non-nav repaint still paints the last-good ndarray.
            if getattr(plot, "_nav_future", None) is expected:
                plot._nav_future = None
            return
        if result is None:
            # The cached read declined (opaque node / eager data). Release the
            # slot and hold the last good frame — the settle re-fire will take
            # the synchronous path, which is always correct.
            if getattr(plot, "_nav_future", None) is expected:
                plot._nav_future = None
            return
        result = np.asarray(result)

        def _paint():
            if getattr(plot, "_nav_future", None) is not expected:
                return
            # Re-check on the main thread: a newer frame may have painted while
            # this one was queued for dispatch.
            if seq < getattr(plot, "_nav_painted_seq", 0):
                return
            plot._nav_painted_seq = seq
            plot._nav_future = None
            plot.current_data = result
            try:
                plot.update()
            except Exception as e:
                log.debug("async nav paint failed: %s", e)

        try:
            session._dispatch_to_main(_paint)
        except Exception as e:
            log.debug("dispatching async nav paint failed: %s", e)

    try:
        fut.add_done_callback(_apply)
    except Exception as e:
        log.debug("attaching async nav callback failed: %s", e)
        child._nav_future = None
        return False
    prof.done("async-submit")
    return True


def write_shared_array(data, shared_arr_name):
    dtype_bytes = data.dtype.str.encode('utf-8')
    dtype_length = len(dtype_bytes)
    ndim = data.ndim
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=shared_arr_name, create=False)
        buffer = shm.buf
        offset = 0
        buffer[offset:offset+4] = dtype_length.to_bytes(4, byteorder='little')
        offset += 4
        buffer[offset:offset+dtype_length] = dtype_bytes
        offset += dtype_length
        buffer[offset:offset+4] = ndim.to_bytes(4, byteorder='little')
        offset += 4
        for dim in data.shape:
            buffer[offset:offset+8] = dim.to_bytes(8, byteorder='little')
            offset += 8
        target_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf[offset:])
        target_arr[:] = data
    except Exception:
        # A failed write means the plot reads a stale/blank shm frame — surface
        # it (workers log to their own stream) rather than silently mispaint.
        log.warning("write_shared_array(%s) failed", shared_arr_name, exc_info=True)
    finally:
        if shm is not None:
            try:
                shm.close()
            except Exception as e:
                log.debug("closing shared-memory %s failed: %s", shared_arr_name, e)


def read_shared_array(shm):
    buffer = shm.buf
    offset = 0
    # Read dtype length
    dtype_length = int.from_bytes(buffer[offset:offset+4], byteorder='little')
    offset += 4
    # An unwritten buffer (e.g. a cancelled write_shared_array future the worker
    # tried to read anyway) has a zero-length / empty dtype header — np.dtype('')
    # then raises "Data type '' not understood". Treat it as "no data yet".
    if dtype_length <= 0 or dtype_length > 32:
        raise ValueError("shared-memory buffer not yet written (empty dtype header)")
    # Read dtype
    dtype_str = bytes(buffer[offset:offset+dtype_length]).decode('utf-8')
    if not dtype_str:
        raise ValueError("shared-memory buffer not yet written (empty dtype)")
    dtype = np.dtype(dtype_str)
    offset += dtype_length
    # Read ndim
    ndim = int.from_bytes(buffer[offset:offset+4], byteorder='little')
    offset += 4
    # Read shape
    shape = tuple(int.from_bytes(buffer[offset+i*8:offset+(i+1)*8], byteorder='little')
                  for i in range(ndim))
    offset += ndim * 8
    # Copy out of the shared buffer before returning — the caller's shm handle
    # may be closed (and the memoryview invalidated) before the array is used.
    arr = np.array(np.ndarray(shape, dtype=dtype, buffer=buffer[offset:]))
    return arr


def _try_async_expensive_nav_read(current_signal, selector, child, indices, prof) -> bool:
    """Classify this lazy nav read and, if EXPENSIVE, submit it asynchronously
    (returning True so the caller returns None / skips the synchronous paint). A
    CHEAP read returns False → the caller falls through to the fast synchronous
    _direct_read_frame / cached path unchanged.

    The ``settle`` flag (motion stopped → paint the resting frame at full res) is
    read from ``child._pending_settle``, stashed by _run_update just before it
    invokes the update fn (the fn signature has no settle parameter)."""
    try:
        data = getattr(current_signal, "data", None)
        if data is None or not hasattr(data, "compute") or not hasattr(data, "chunks"):
            return False
        nav_dim = current_signal.axes_manager.navigation_dimension
        frame_shape = data.shape[nav_dim:]
        frame_bytes = int(np.prod(frame_shape)) * data.dtype.itemsize
        if _classify_nav_read(current_signal, indices, data, frame_bytes,
                              child=child) != "expensive":
            return False
        settle = bool(getattr(child, "_pending_settle", False))
        return _submit_async_nav_read(child, current_signal, indices, settle, prof)
    except Exception as e:
        log.debug("async expensive nav-read classification failed: %s", e)
        return False


def _nav_readable_data(signal, data) -> bool:
    """Can *data* (a captured ``signal.data`` binding) be sliced with this
    signal's navigation coordinates at all?

    The tripwire case is hyperspy's ``_deepcopy_with_new_data``: it TRANSIENTLY
    rebinds ``self.data = None`` on the LIVE signal object while it deep-copies
    (the data setter's ``np.atleast_1d(np.asanyarray(None))`` makes that an
    ``array([None], dtype=object)``, shape ``(1,)``) — and EVERY hyperspy
    operation (arithmetic, comparison, ``sum``, ``deepcopy``) passes through it.
    The math console evaluates user expressions (``s1 + 0``, incl. the live
    preview's nav-refresh re-runs) against the SAME bound signal objects on the
    console thread, so a navigator update on the dispatcher thread can land
    inside that window: the signal still reports nav_shape (6, 6) but its data
    is the 1-element placeholder → "too many indices for array" — the
    second-signal IndexError, console flavour. ``_pending_future_data`` can't
    catch it because ``data[0]`` is None, not a future.

    A None / object-dtype / under-dimensioned ``.data`` is never a readable
    frame source, transient or not, so the caller skips the frame (returns
    None → the last good frame stays up, same as the pending-future skip)."""
    if data is None:
        return False
    if isinstance(data, np.ndarray) and data.dtype == object:
        # The wrapped-future shape was already skipped by _pending_future_data;
        # any OTHER object array is the deepcopy placeholder (or equally not a
        # frame source). Catches the 1-D-navigator case the ndim check below
        # can't (placeholder ndim 1 == nav_dim 1).
        return False
    ndim = getattr(data, "ndim", None)
    if ndim is None:
        return True   # not array-like; leave it to the branches (as before)
    try:
        nav_dim = int(signal.axes_manager.navigation_dimension)
    except Exception:
        return True
    return int(ndim) >= nav_dim


def _prepare_nav_indices(current_signal, indices, integrating: bool, data=None):
    """Transform RAW selector indices into DATA-ORDER, clamped array indices — the
    shared index-prep for the navigator read.

    Does exactly what ``update_from_navigation_selection`` does before it reads:
      * swap the innermost spatial (x, y) widget pair → (y, x) data order (for a
        2-D-signal navigator; the outer stack coords in a 5-D chain stay put),
      * mean-reduce a crosshair's point cloud to one integer nav point when NOT
        integrating (a region keeps all its points),
      * clamp every coordinate to the leading (navigation) data-axis sizes so a
        stale/larger-grid selector position can't IndexError.

    ``data`` (optional) is the caller's already-captured ``current_signal.data``
    binding — pass it so the clamp bounds and the eventual read use the SAME
    array even if ``.data`` is concurrently rebound (see ``_nav_readable_data``).

    Extracted so the MDI-overlay layer read (:mod:`spyde.actions.overlay`) resolves
    the SAME nav position as the base frame from the same raw selector indices.
    Returns the prepared ndarray (or None on failure)."""
    indices = np.asarray(indices)
    _has_spatial_nav = False
    try:
        _has_spatial_nav = current_signal.axes_manager.signal_dimension == 2
    except Exception:
        _has_spatial_nav = indices.ndim >= 1 and indices.shape[-1] == 2
    if _has_spatial_nav and indices.ndim >= 1 and indices.shape[-1] >= 2:
        indices = indices.copy()
        indices[..., -2:] = indices[..., -2:][..., ::-1]

    if not integrating:
        indices = np.mean(indices, axis=0).astype(int)

    try:
        data_obj = data if data is not None else current_signal.data
        data_shape = getattr(data_obj, "shape", None)
        if data_shape is None:
            nav_shape_xy = tuple(current_signal.axes_manager.navigation_shape)
            data_shape = tuple(reversed(nav_shape_xy))
        idx_arr = np.asarray(indices)
        if idx_arr.size and len(data_shape):
            ncoord = idx_arr.shape[-1] if idx_arr.ndim else 1
            n = min(ncoord, len(data_shape))
            limits = np.array([data_shape[i] - 1 for i in range(n)], dtype=idx_arr.dtype)
            before = idx_arr.copy()
            if limits.size == ncoord:
                indices = np.clip(idx_arr, 0, limits)
            else:
                clamped = idx_arr.copy()
                if idx_arr.ndim == 1:
                    clamped[:n] = np.clip(idx_arr[:n], 0, limits)
                else:
                    clamped[..., :n] = np.clip(idx_arr[..., :n], 0, limits)
                indices = clamped
                log.debug(
                    "NAV-DEBUG clamp ncoord=%d != bounds=%d; partial-clamped "
                    "data_shape=%s", ncoord, limits.size, data_shape,
                )
            if log.isEnabledFor(logging.DEBUG) and not np.array_equal(before, np.asarray(indices)):
                log.debug(
                    "NAV-DEBUG clamped out-of-range nav index %s -> %s "
                    "(data_shape=%s) — selector held a stale/larger-grid position",
                    before.tolist(), np.asarray(indices).tolist(), data_shape,
                )
    except Exception as e:
        log.debug("clamping nav indices failed: %s", e)
    return indices


def update_from_navigation_selection(
        selector: "BaseSelector",
        child: "Plot",
        indices,
        get_result: bool = False,
):
    """
    Update the plot based on the navigation selection. This is the most common update function for using some
    navigation selector (on a parent) and updating a child plot.

    Parameters
    ----------
    selector : BaseSelector
        The selector that triggered the update.
    child : Plot
        The child plot to update.
    indices : array-like
        The indices selected by the selector.
    get_result : bool
        Only meaningful in the eager/placeholder branches; the lazy branch always
        reads synchronously (see below). Kept for signature stability.
    """
    # Signal that the user is interacting NOW, so a heavy background disk fill
    # (the progressive navigator/VI sum) yields the disk to this frame read
    # instead of starving it — the "plot frozen while the VI computes" fix.
    _interactive_activity.poke()

    # Per-frame update profile (no-op unless SPYDE_NAV_PROFILE=1). Started here so
    # its `total` covers the whole read (index prep → cache read → dtype → prefetch);
    # the transport/paint half is timed separately in _run_update around update_data.
    _prof = NavProfile(getattr(child, "window_id", "SIG"), indices)

    # get the data from the signal tree based on the current indices

    current_signal = child.plot_state.current_signal

    # A signal whose `.data` is still a pending FUTURE has nothing to slice yet,
    # and that is independent of `_lazy` — so it must be checked here rather
    # than inside the lazy branch below.
    #
    # `signal.data = <future>` does not store the object: hyperspy's setter runs
    # `np.atleast_1d(np.asanyarray(value))`, so it lands as a length-1 OBJECT
    # array. Indexing that with nav coordinates raises
    # "too many indices for array: array is 1-dimensional, but 2 were indexed" —
    # the long-standing "second-signal IndexError", which the eager branch below
    # logs and re-raises.
    #
    # This was always latent: the progressive navigator fill parks a future on
    # the base navigator signal, and `PlotUpdateWorker` swaps in the real array
    # once it resolves. With a distributed Future that window was milliseconds,
    # so the race effectively never fired. The client-side `_AssembledFuture`
    # only reports done when the WHOLE dispatch finishes — ~50 s on a 977-chunk
    # movie — which widens the window enough that ordinary navigator moves land
    # inside it. Returning None leaves the last good frame up, which is what the
    # caller already does for a superseded read.
    if _pending_future_data(current_signal):
        log.debug("nav read skipped: %s still carries a pending future",
                  getattr(current_signal, "metadata", None) and
                  getattr(current_signal.metadata, "General", None) and
                  getattr(current_signal.metadata.General, "title", "<signal>"))
        return None

    # Capture `.data` ONCE for this read, and skip if the captured binding
    # cannot satisfy the nav indices. hyperspy transiently rebinds `.data` on
    # the LIVE signal object during ordinary operations (`_deepcopy_with_new_data`
    # parks a shape-(1,) `array([None], dtype=object)` placeholder while it
    # deep-copies), and the math console runs user expressions (`s1 + 0` — the
    # live preview's nav-refresh) against these SAME objects on the console
    # thread. Working from one captured reference (rebinds are GIL-atomic) plus
    # this guard makes that race structurally harmless: a mid-window read skips
    # the frame (the last good frame stays up, exactly like the pending-future
    # skip above); a post-capture swap still clamps + indexes the coherent
    # pre-swap array. See _nav_readable_data.
    data_now = getattr(current_signal, "data", None)
    if not _nav_readable_data(current_signal, data_now):
        log.debug(
            "nav read skipped: data %s cannot satisfy nav_shape %s — transient "
            "placeholder from a concurrent hyperspy op on this signal (e.g. a "
            "console evaluation), or a mis-shaped signal",
            getattr(data_now, "shape", type(data_now).__name__),
            tuple(current_signal.axes_manager.navigation_shape),
        )
        return None

    # Per-frame trace — gated behind SPYDE_NAV_TIMING because it fires on EVERY
    # crosshair move and floods the IPC log/panel at DEBUG (which itself adds lag).
    if _NAV_TIMING:
        log.debug(f"update_from_navigation_selection: indicies = {indices}, current_signal = {current_signal}, "
                  f"selector = {selector}, child = {child}, get_result = {get_result}")

    # ── NAV-DEBUG ───────────────────────────────────────────────────────────
    # Diagnostics for the "second signal" reports: IndexError on the DP update
    # and threaded-vs-distributed cache path. Opt-in (SPYDE_NAV_TIMING=1) because
    # it fires per crosshair move and floods the IPC log at DEBUG.
    if _NAV_TIMING and log.isEnabledFor(logging.DEBUG):
        try:
            _cache = getattr(current_signal, "cached_dask_array", None)
            _cli = getattr(_cache, "client", None) if _cache is not None else None
            _cli_set = getattr(_cache, "_client", None) is not None if _cache is not None else None
            _cli_kind = (
                "distributed" if (_cli is not None and type(_cli).__name__ == "Client")
                else ("THREADED/none" if _cli is None else type(_cli).__name__)
            )
            _dshape = getattr(getattr(current_signal, "data", None), "shape", None)
            log.debug(
                "NAV-DEBUG enter: sig=%s lazy=%s data.shape=%s nav_shape=%s "
                "sig_shape=%s raw_indices=%s integrating=%s cache.client=%s "
                "cache._client_set=%s",
                getattr(current_signal, "_signal_type", type(current_signal).__name__),
                getattr(current_signal, "_lazy", None),
                _dshape,
                tuple(current_signal.axes_manager.navigation_shape),
                tuple(current_signal.axes_manager.signal_shape),
                np.asarray(indices).tolist(),
                getattr(selector, "is_integrating", None),
                _cli_kind,
                _cli_set,
            )
        except Exception as _e:
            log.debug("NAV-DEBUG enter logging failed: %s", _e)

    # anyplotlib displays the navigator image un-transposed (imshow convention:
    # data axis 0 = rows = y = iy, axis 1 = cols = x = ix). The 2-D spatial
    # selector reports widget coords (cx = column, cy = row), i.e. (x, y) order,
    # so the SPATIAL pair must be swapped (x, y) → (y, x) to index data[iy, ix] —
    # otherwise a real-space pixel shows a transposed/wrong diffraction pattern
    # (and IndexError-then-clamp on a non-square scan).
    #
    # For a chained multi-navigator (a 5-D stack: outer index axis → spatial
    # scan → DP), the combined row is (outer…, x, y) — the outer navigator
    # coordinate(s) come FIRST (broadcast_rows_cartesian puts upstream selectors
    # first) and are ALREADY in data order; only the spatial (x, y) pair from the
    # innermost crosshair is in widget order. So swap just the LAST TWO columns,
    # not the whole row (reversing the whole row would scramble the stack axis
    # against x — the cause of "clamped [0,525,169] -> [0,299,169]" on a 5-D
    # stack: x=525 wrongly bounded by the y-axis). The swap applies whenever the
    # innermost navigator is 2-D spatial (signal_dimension == 2).
    #
    # The swap + mean-reduce + clamp is shared with the MDI-overlay layer read
    # (spyde.actions.overlay) so a layer resolves the SAME nav position from the
    # same raw selector indices — see _prepare_nav_indices.
    indices = _prepare_nav_indices(current_signal, indices,
                                   selector.is_integrating, data=data_now)

    if current_signal._lazy:
        if is_future_like(data_now[0]):
            current_img = np.ones(current_signal.axes_manager.signal_shape, dtype=np.int8)
            if current_img.ndim == 2:
                #make checkerboard pattern to indicate loading
                current_img[::2, ::2] = 0
        elif _try_async_expensive_nav_read(current_signal, selector, child, indices, _prof):
            # ── EXPENSIVE TIER (async + cancellable) ───────────────────────────
            # A large integrating region, a cold cross-chunk LARGE frame, or a
            # derived rebin/crop view whose transform re-runs every move is too
            # slow to compute synchronously on the serial dispatcher (it would
            # freeze the navigator). _try_async_expensive_nav_read classified it,
            # submitted the frame via ComputeBackend.submit_graph OFF the
            # dispatcher, and armed a done-callback that paints when it lands
            # (cancelling any superseded read). Return None so _run_update skips
            # the synchronous paint — the callback owns it. The last good frame
            # stays up until then (no flash). (Live-Display §3 tiered read.)
            return None
        elif (_direct := _direct_read_frame(
                current_signal, selector, indices, _prof, child=child)) is not None:
            # ── UNIFIED FAST VIEW READ: compute the slice DIRECTLY, bypass ──────
            # get_index. For EVERY navigator (movie frame, 4D-STEM diffraction
            # pattern, integrating region) hyperspy's CachedDaskArray adds ~160 ms
            # of overhead per frame (block bookkeeping, ghost padding,
            # surrounding-block prefetch, meshgrid indexing) and BALLOONS to seconds
            # on a cold miss — while a plain raw[idx].compute(scheduler="synchronous")
            # of the same slice is ~2–30 ms and byte-identical (profiled: movie
            # 179→25 ms, DP 10→2 ms, region 9→7 ms). It also serves DERIVED views
            # (rebin/crop/rechunk) that have no CachedDaskArray at all, and stays
            # memory-bounded (dask reads only the frame's deps). _direct_read_frame
            # did the read + prefetch + profile. get_index remains only as the
            # fall-through safety net below (eager data / oversized region).
            current_img = _direct
        else:
            # ── UNIFIED CACHED READ (synchronous, no distributed scheduler) ─────
            # This runs on the single serial _nav_dispatcher thread (see
            # base_selector), one update at a time, latest-position-wins: a newer
            # move overwrites the pending slot before a superseded one runs. So we
            # compute the frame RIGHT HERE, synchronously, and return the numpy
            # array — no distributed Future, no shared-memory buffer, no
            # PlotUpdateWorker poll. `update_data` paints an ndarray immediately.
            #
            # The speed comes from hyperspy's CachedDaskArray numpy chunk cache:
            # with the cache client UNSET, get_index takes its synchronous branch
            # (caches the loaded block, then slices/means it in numpy). A DP
            # navigator dwells within a nav chunk → ~1 ms cache hits; a movie is 1
            # frame/chunk → each move is a ~cold read of just that frame. This
            # matches the OLD distributed path's speed WITHOUT the scheduler
            # round-trip, the shm buffer, the client pinning, or the
            # _inflight_getinds juggling.
            #
            # Serial-only: the cache's block bookkeeping is not concurrency-safe
            # ("ValueError: (i, j) is not in list"); the dispatcher already
            # guarantees this is never re-entered concurrently, so no lock (§4).
            #
            # Force the SYNCHRONOUS cache path (fast: ~1-2 ms dwell-in-chunk hits
            # vs ~16 ms distributed). Setting _client=None requests it, AND
            # spyde.external.hyperspy.cached_dask_array makes the cache honour that
            # (the fork's client property otherwise adopts the app's global default
            # Client from this non-worker thread → a distributed round-trip on
            # every move; the pin alone was a no-op — see that patch). A fresh
            # nav cache starts with _client=None, so this is normally a re-assert.
            cached_arr = getattr(current_signal, "cached_dask_array", None)
            if cached_arr is not None:
                try:
                    cached_arr._client = None
                except Exception as _e:
                    log.debug("unsetting cache client failed: %s", _e)

            # Was this chunk already resident in the numpy cache? (a cache HIT is
            # ~ms; a MISS reads the chunk off disk). Recorded for the profile so a
            # slow report distinguishes "cold cross-chunk read" from "the cache is
            # fast but something downstream is slow".
            _cache_hit = _nav_cache_was_hit(current_signal, indices)

            with _prof.stage("read"):
                current_img = current_signal._get_cache_dask_chunk(
                    indices, get_result=True,
                )
                current_img = np.asarray(current_img)
            _prof.set_frame(current_img)

            # Dtype parity with the OLD distributed path. The synchronous cache
            # branch runs everything through np.mean, so it returns float64 for
            # BOTH a single point and a region — but the distributed path returned
            # the native frame dtype (single point → the raw frame; region →
            # weighted_mean_round_from_sums, which ROUNDS an integer result back to
            # its dtype). So for an INTEGER source, round the float64 back to the
            # frame dtype: a no-op on a single point's values, and the correct
            # rounded integer mean for a region — so the DP navigator shows the
            # SAME uint16 frame (same memory + contrast) it did before. Float
            # sources keep their (un-rounded) mean. (Pinned by test_nav_cached_read.py.)
            with _prof.stage("dtype"):
                try:
                    src_dtype = getattr(data_now, "dtype", None)
                    if (src_dtype is not None
                            and np.issubdtype(src_dtype, np.integer)
                            and np.issubdtype(current_img.dtype, np.floating)):
                        # A SINGLE point (crosshair): get_index returns the frame's
                        # own values as float64 — they are already EXACT integers,
                        # so np.rint is a mathematical no-op. Skip it: plain astype
                        # (truncation) gives the identical result at ~4x less cost
                        # (~40 ms vs ~160 ms on a 4k frame — the movie-scrub hot
                        # path; profiled). Only an INTEGRATING REGION produces
                        # fractional means that actually need rounding.
                        _idx0 = np.asarray(indices)
                        single_point = (_idx0.ndim <= 1) or _idx0.shape[0] == 1
                        if single_point:
                            current_img = current_img.astype(src_dtype, copy=False)
                        else:
                            current_img = np.rint(current_img).astype(src_dtype)
                except Exception as _e:
                    log.debug("nav frame round-to-dtype failed: %s", _e)

            # Read-ahead prefetch for a MOVIE scrub: warm the OS page cache for
            # the next few frames so the following move finds them warm (~18 ms
            # vs ~50 ms cold). Only for a 1-D (time) navigator on
            # a crosshair (single point): a 4D-STEM scan dwells in-chunk so its
            # cache already covers neighbours, and an integrating region has no
            # single "next frame". Reads the RAW dask array (not the CachedDaskArray)
            # so it never races the nav read's cache (§4).
            with _prof.stage("prefetch"):
                try:
                    am = current_signal.axes_manager
                    _idx = np.asarray(indices)
                    is_single = (not selector.is_integrating) or _idx.ndim <= 1
                    if (am.navigation_dimension == 1 and is_single
                            and hasattr(data_now, "shape")):
                        n_time = int(data_now.shape[0])
                        center = int(np.atleast_1d(_idx).ravel()[0])
                        _movie_prefetcher.prime(data_now, center, n_time)
                except Exception as _e:
                    log.debug("movie prefetch prime failed: %s", _e)
            _prof.done("cache=" + ("hit" if _cache_hit else "MISS"))
    else:
        # Eager (in-RAM) slice. `indices` is either a single nav point (1-D,
        # from a crosshair after the mean-reduce above) or a list of nav points
        # (2-D, from an integrating region). A single point yields one signal
        # frame directly; multiple points are averaged frame-wise.
        #
        # NB: the old `tuple(indices[i] ...)` form conflated "number of nav
        # coordinates" with "number of points to average", which collapsed a
        # 2-D-navigation diffraction pattern to 1-D. The Qt app never hit this
        # because it always loaded lazily (the Future branch above); eager
        # example datasets do.
        # Index the CAPTURED binding (data_now), never a fresh `.data` read —
        # the clamp above used its shape, so clamp and read stay coherent even
        # if a console-thread hyperspy op rebinds `.data` mid-read.
        idx = np.asarray(indices)
        try:
            with _prof.stage("read"):
                if idx.ndim <= 1:
                    point = tuple(int(v) for v in np.atleast_1d(idx))
                    current_img = data_now[point]
                else:
                    sl = tuple(idx[:, k].astype(int) for k in range(idx.shape[1]))
                    current_img = data_now[sl].mean(axis=0)
            _prof.set_frame(current_img)
        except Exception:
            log.exception(
                "NAV-DEBUG eager index RAISED: indices=%s data.shape=%s "
                "nav_shape=%s — the second-signal IndexError",
                idx.tolist(),
                getattr(data_now, "shape", None),
                tuple(current_signal.axes_manager.navigation_shape),
            )
            raise
        _prof.done("eager")
    return current_img


def get_fft(selector: "BaseSelector", child: "Plot", indices, get_result: bool = False):
    """
    Get the FFT of the image.

    Parameters
    ----------
    img : array-like
        The input image.

    Returns
    -------
    array-like
        The FFT of the input image.
    """
    # convert indices to image slice:
    max_x, max_y = np.max(indices, axis=0)
    min_x, min_y = np.min(indices, axis=0)

    img = selector.parent.image_item.image

    img_max_x = img.shape[0] - 1
    img_max_y = img.shape[1] - 1
    if max_x > img_max_x:
        max_x = img_max_x
    if max_y > img_max_y:
        max_y = img_max_y
    if min_x < 0:
        min_x = 0
    if min_y < 0:
        min_y = 0

    slice_x, slice_y = slice(min_x, max_x + 1), slice(min_y, max_y + 1)
    sliced_img = img[slice_x, slice_y]
    fft_img = fft.fftshift(fft.fft2(sliced_img))
    return fft_img.real


def compute_virtual_image_kernel(
    data: da.Array,
    mask: np.ndarray,
    client: distributed.Client,
    gpu_worker_address: "str | None",
) -> distributed.Future:
    """
    Compute a virtual image by masking and summing the last two (signal) axes.

    Equivalent to:
        np.sum(data * mask[np.newaxis, np.newaxis, ...], axis=(-1, -2))

    Works for any number of navigation axes (3D, 4D, 5D, 6D datasets).
    Signal axes must be the last two (HyperSpy convention).

    Broadcasting mask as a numpy array (not a dask array) means each worker
    multiplies its navigation chunk directly against the in-memory mask without
    any cross-chunk communication, then reduces over the last two axes within
    the chunk. This is O(n_nav_chunks) independent tasks with no shuffle.

    Parameters
    ----------
    data : dask array, shape (...nav..., ky, kx)
    mask : float32 numpy array, shape (ky, kx)
    client : dask distributed Client
    gpu_worker_address : str or None

    Returns
    -------
    distributed.Future resolving to np.ndarray of shape (...nav...)
    """
    mask = np.asarray(mask, dtype=np.float32)
    if gpu_worker_address:
        with dask.annotate(resources={"GPU": 1}):
            result = (data * mask).sum(axis=(-2, -1))
    else:
        result = (data * mask).sum(axis=(-2, -1))
    return client.compute(result)


class _ProgressiveFuture:
    """Future-like handle for a progressive chunk dispatch (duck-typed
    ``done``/``result``/``cancel``).

    The VI/FFT stream polls this itself and it is deliberately INVISIBLE to
    PlotUpdateWorker (no ``_spyde_future`` marker) — the stream paints from its
    shm buffer, and letting the worker also deliver the assembled array would
    push a second, unrelated frame onto the plot. The navigator fill wants the
    opposite and uses :class:`_AssembledFuture`.
    """

    def __init__(self, label: str = "progressive"):
        self.label = label
        self._done_evt = threading.Event()
        self._error: Exception | None = None
        self._value = None
        self._cancel_cb = None
        self.cancelled = False

    def done(self) -> bool:
        return self._done_evt.is_set()

    def result(self, timeout=None):
        if not self._done_evt.wait(timeout):
            raise TimeoutError("progressive compute not finished")
        if self._error is not None:
            raise self._error
        return self._value

    def cancel(self) -> None:
        self.cancelled = True
        cb = self._cancel_cb
        if cb is not None:
            try:
                cb()
            except Exception as e:
                log.debug("progressive cancel callback failed: %s", e)


_ASSEMBLED_SEQ = itertools.count()


def _pending_future_data(signal) -> bool:
    """True when *signal*'s ``.data`` is a future that has not resolved yet.

    Covers both shapes the future can take, because hyperspy's `data` setter
    runs ``np.atleast_1d(np.asanyarray(value))``: the bare object, and the
    length-1 OBJECT array it actually becomes. Mirrors
    ``plot_update_worker._future_from_signal``, which unwraps the same two.

    A RESOLVED future is not "pending" — `PlotUpdateWorker` will swap the real
    array in on its next tick, and until then there is nothing to slice either
    way; reporting it as pending simply skips one frame instead of raising.
    """
    data = getattr(signal, "data", None)
    if is_future_like(data):
        return True
    # `if data` on an ndarray raises "truth value is ambiguous" — test length.
    if isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
        try:
            return is_future_like(data.reshape(-1)[0])
        except Exception:
            return False
    return False


def is_future_like(obj) -> bool:
    """True for a dask ``distributed.Future`` OR a SpyDE duck-typed future.

    The progressive navigator fill no longer submits a second whole-array graph
    just to have a real ``distributed.Future`` to hand over: the per-chunk
    futures already computed every value, so the dispatcher assembles them
    client-side and returns an :class:`_AssembledFuture` (``_spyde_future``)
    with the same ``done()``/``result()``/``key`` surface.  Anything without the
    marker — a plain ndarray, a ``_ProgressiveFuture`` from the VI stream, the
    synchronous ``_SyncResult`` — is ignored exactly as before.

    NB ``signal.data = <future>`` does NOT store the object as-is: hyperspy's
    setter runs it through ``np.atleast_1d(np.asanyarray(...))``, so it lands as
    a length-1 OBJECT array.  Callers that test a signal's data must look at
    ``data[0]``, not ``data``.
    """
    return isinstance(obj, Future) or getattr(obj, "_spyde_future", False) is True


class _AssembledFuture(_ProgressiveFuture):
    """Future-like handle whose value is the CLIENT-SIDE ASSEMBLY of a chunk
    dispatch — the replacement for the second whole-array ``client.compute()``.

    The navigator fill used to submit every nav chunk AND then
    ``client.compute(result_array)``: a complete SECOND pass over the dataset
    computing values the chunks had already produced.  It existed only because
    ``PlotUpdateWorker`` isinstance-checks ``distributed.Future`` on
    ``plot.current_data`` / ``signal.data``.  The chunks ARE the result, so the
    dispatcher assembles them and this handle carries the assembly with the
    same ``done()``/``result()``/``key`` surface; the worker accepts it via the
    ``_spyde_future`` marker (see ``plot_update_worker._is_future``).

    ``key`` must be UNIQUE per instance: the worker dedups by
    ``(key, id(plot))``, so a shared constant would make a second fill on the
    same plot never emit.  It must also not contain ``write_shared_array``,
    which the worker routes to a shm read instead of ``result()``.
    """

    _spyde_future = True

    def __init__(self, label: str = "assembled"):
        super().__init__(label)
        self.key = f"spyde-{label}-{next(_ASSEMBLED_SEQ)}"


def _local_display_future(lazy_array, label: str = "display"):
    """Compute *lazy_array* in THIS process, off-thread, into an
    :class:`_AssembledFuture` — the same handle the cluster path hands back, so
    everything downstream (PlotUpdateWorker poll → ``Session._on_plot_ready`` →
    the ``window_computing`` stop) is identical either way."""
    fut = _AssembledFuture(label)

    def _run():
        try:
            fut._value = np.asarray(lazy_array.compute(scheduler="threads"))
        except Exception as e:                     # delivered via result()
            fut._error = e
            log.exception("local display compute failed")
        finally:
            # LAST, always: _on_plot_ready is what stops the window's
            # "Calculating…" overlay, and it only runs once this is set.
            fut._done_evt.set()

    threading.Thread(target=_run, daemon=True,
                     name=f"display-{label}").start()
    return fut


def graph_can_reach_workers(lazy_array) -> bool:
    """True if *lazy_array*'s graph can be sent to a distributed worker.

    The verdict must come from **distributed's own serializer**, not from a
    plain pickle: those two disagree, and on a real format. A lazy ``.hspy``
    graph holds an h5py object that ``cloudpickle`` refuses outright, yet
    ``distributed`` serializes it happily (it has handlers pickle does not) —
    so an earlier cloudpickle-only version of this function answered False for
    ``.hspy`` and would have pushed every HDF5 dataset off the cluster and into
    this process for no reason. Measured verdicts:

        format   cloudpickle   distributed   .. and in the app
        .tif     False         False         was broken, now fixed
        .hspy    False         True          always worked — must stay on the cluster
        .zspy    True          True          always worked

    ``cloudpickle`` is still asked FIRST, purely as a fast accept: it is the
    cheaper call, it never says True where distributed says False (it is the
    more permissive of the two only in the direction that matters here), and it
    settles the common ``.zspy``/``.mrc`` case without touching the noisy path
    below.

    Probing at all — rather than catching a failed ``client.compute()`` — is
    deliberate: that failure is not quiet. ``distributed.protocol.pickle`` logs
    it at ERROR with three chained tracebacks BEFORE raising, so a try/except
    cannot suppress it and every ``.tif`` open would spray tracebacks into the
    user's Log panel. The same logger is why the probe silences it here: this
    call is a QUESTION, not an error, and its answer is the return value.
    """
    try:
        import cloudpickle
        cloudpickle.dumps(lazy_array)
        return True                       # fast accept: .zspy, .mrc, plain dask
    except Exception:
        pass

    from distributed.protocol import serialize
    from distributed.protocol.serialize import to_serialize
    pickle_log = logging.getLogger("distributed.protocol.pickle")
    prior = pickle_log.level
    pickle_log.setLevel(logging.CRITICAL)
    try:
        serialize(to_serialize(lazy_array), on_error="raise")
        return True                       # .hspy lands here
    except Exception:
        return False                      # .tif lands here
    finally:
        pickle_log.setLevel(prior)


def compute_display_future(lazy_array, client, label: str = "display"):
    """Compute a lazy array FOR DISPLAY, on the cluster when the graph can get
    there and in this process when it cannot.

    **Not every lazy graph is picklable, and a graph that is not cannot go to a
    distributed worker at all.** Two of the formats SpyDE opens build exactly
    such a graph: rosettasciio's lazy TIFF reader closes over an open
    ``BufferedReader``, and an ``.hspy`` closes over an h5py object. Handing one
    to ``client.compute()`` raises *at submit time* —

        TypeError: Could not serialize object of type _HLGExprSequence
        ... cannot pickle 'BufferedReader' instances

    — and because that call sits between ``window_computing().start()`` and the
    ``_on_plot_ready`` that stops it, the raise ALSO stranded the spinner: a
    plain ``.tif`` opened as a permanently black window captioned
    "Calculating…", with "Failed to load …" on the status bar. (Only ``.zspy``
    is picklable, which is why the bug hid — that is the format the app writes
    itself and the one every lazy test fixture uses.)

    Computing it here instead is not a downgrade: this display path materialises
    ONE image that has to reach this process to be painted anyway, so the cluster
    buys a graph serialization and a result transfer and nothing else — the same
    reasoning ``ComputeBackend.submit_graph`` documents for the navigator read.
    The cluster is still preferred whenever the graph CAN go, because a lazy
    display array may be the tip of a large reduction whose inputs should stay on
    the workers.
    """
    if client is None or not graph_can_reach_workers(lazy_array):
        if client is not None:
            log.info("display graph holds an unpicklable handle (lazy .tif/.hspy "
                     "hold an open file object); computing it in-process")
        return _local_display_future(lazy_array, label)
    return client.compute(lazy_array)


class _StopFlag(list):
    """A ``[False]`` stopped_flag that ALSO reports a set ``threading.Event``.

    ``dispatch_chunks`` polls ``stopped_flag[0]``; SpyDE's progressive callers
    stop in three different ways — ``handle.cancel()``, a ``stop_event``
    (``_stop_progressive_stream`` sets one before it unlinks the shm), and
    ``BaseSignalTree.register_cancel`` writing ``flag[0] = True`` on tree close.
    Folding the event into ``__getitem__`` lets one flag serve all three.
    """

    def __init__(self, event=None):
        super().__init__([False])
        self._event = event

    def __getitem__(self, i):
        if i == 0 and self._event is not None and self._event.is_set():
            return True
        return super().__getitem__(i)


def _dispatched_progressive(result_array, nav_shape, client, on_chunk_done,
                            on_future, stop_event, label="navigator",
                            cap=None, future_cls=_AssembledFuture):
    """Progressive per-nav-chunk compute through the SHARED dispatcher
    (:func:`spyde.compute_dispatch.dispatch_chunks`).

    Serves BOTH progressive paths — the navigator fill and the VI/FFT stream —
    which had grown two hand-rolled submit loops with different bug sets.  What
    the shared dispatcher brings that neither loop had in full:

      * **Batched submit.** One ``client.compute(list)`` per top-up instead of a
        blocking scheduler round trip per chunk with the GIL held in the client
        process.  Measured end to end on a 977-frame / 15.3 GB in-situ movie
        (977 nav chunks, 4 workers — ``benchmark_nav_fill_dispatch.py``):
        **9.86 s** of submission during which NOTHING else in the backend could
        run (blank navigator, silent paint threads) and a first painted chunk at
        16.4 s, versus 0.00 s / **0.08 s** here.
      * **Backpressure.** A bounded in-flight window (half the cluster threads),
        so the scheduler is never handed the whole dataset at once — the
        prefetch-everything-then-spill pathology.  The navigator fill had NO
        window at all: it submitted all 977 up front.
      * **The stall watchdog + scheduler poke** the find-vectors batch depends
        on (task delivery freezes on the hidden Electron-spawned backend until
        client traffic arrives).
      * **No second whole-array graph** — see :class:`_AssembledFuture`.

    Observable behaviour is unchanged: chunks land progressively in submission
    order, ``on_chunk_done(chunk_result, nav_slices)`` fires from the dask
    done-callback thread IN THE CLIENT PROCESS (worker-side shm writes
    access-violate on Windows teardown), the result is assembled client-side,
    and ``cancel()`` still stops outstanding work immediately (via
    ``dispatch_chunks``' ``on_start`` hook, not on its next 0.5 s poll — a
    superseded VI stream must not keep computing through an ROI drag).

    ``dispatch_chunks`` BLOCKS, so it runs on a daemon thread and this returns
    immediately, exactly like the old submit loops did.
    """
    from spyde.compute_dispatch import dispatch_chunks

    nav_ndim = len(nav_shape)
    trailing = (slice(None),) * (result_array.ndim - nav_ndim)
    handle = future_cls(label)
    stopped = _StopFlag(stop_event)
    # Filled in by dispatch_chunks' on_start hook (see below); until then a
    # cancel still registers on the flag.
    handle._cancel_cb = lambda: stopped.__setitem__(0, True)

    if on_future is not None:
        # ONE registration instead of one per chunk (978 of them on the
        # 977-frame movie). BaseSignalTree.close -> _cancel_all_compute calls
        # .cancel() on a non-distributed future.
        try:
            on_future(handle)
        except Exception as e:
            log.debug("on_future(%s dispatch handle) registration failed: %s",
                      label, e)

    def _assemble(result, nav_slices, chunk_result):
        result[nav_slices + trailing] = chunk_result

    def _chunk_done(nav_slices, chunk_result):
        # dispatch_chunks calls (slices, result); the live-buffer callers have
        # always taken (result, slices) — flip here, not at every call site.
        if on_chunk_done is not None and not stopped[0]:
            on_chunk_done(chunk_result, nav_slices)

    def _on_start(request_stop):
        handle._cancel_cb = request_stop

    # An integer navigator sum can't hold NaN; only a stopped run ever leaves
    # the fill value visible (every chunk overwrites its own slice).
    fill = np.nan if np.issubdtype(result_array.dtype, np.floating) else 0

    def _run():
        try:
            arr = dispatch_chunks(
                client, result_array, nav_ndim,
                [],            # no GPU lane
                None,          # ONE UNPINNED lane over the whole cluster
                stopped_flag=stopped, assemble=_assemble, fill_value=fill,
                label=label, on_chunk_done=_chunk_done, cap=cap,
                lane_default_mode="off", on_start=_on_start,
            )
            handle._value = arr          # None when stopped
        except Exception as exc:
            handle._error = exc
        finally:
            handle._done_evt.set()

    threading.Thread(target=_run, daemon=True, name=f"{label}-dispatch").start()
    return handle


def compute_with_live_buffer(
    result_array: da.Array,
    nav_shape: tuple,
    client: distributed.Client,
    shm_name: str,
    on_chunk_done=None,
    on_future=None,
    windowed: bool = False,
    stop_event=None,
):
    """
    Progressive per-nav-chunk compute: calls ``on_chunk_done(chunk_result,
    nav_slices)`` from a Dask callback thread as each chunk completes.  The
    caller is responsible for marshalling back to the GUI thread.

    Returns a future-like handle (``done``/``result``/``cancel``) whose value is
    the CLIENT-SIDE ASSEMBLY of the chunks — NOT a ``distributed.Future`` over a
    second whole-array graph, which is what this used to return.  See
    :func:`_dispatched_progressive` and :class:`_AssembledFuture`.

    Shared memory is written from the GUI process (not from worker
    subprocesses) to avoid Windows access-violation crashes when the shm
    segment is torn down during test teardown while a worker is mid-write.

    Parameters
    ----------
    result_array : dask array, nav-shaped (signal axes already reduced)
    nav_shape    : tuple — full navigation shape
    client       : dask distributed Client (or None for synchronous path)
    shm_name     : str — name of a pre-existing SharedMemory segment to
                   update from the GUI side (via on_chunk_done)
    on_chunk_done : callable(chunk_result, nav_slices) | None
                   Called from a Dask callback thread as each chunk finishes.
    on_future     : callable(future) | None
                   Called synchronously with every Future created so the caller
                   can register them for cancellation on teardown. Not called on
                   the synchronous path. The navigator path now creates exactly
                   ONE handle (the dispatcher owns the per-chunk futures and
                   cancels them itself), so this fires once there.
    """
    if client is None:
        # Synchronous fallback: compute the whole array and write shm once
        result = result_array.compute()
        if shm_name:
            try:
                from multiprocessing import shared_memory as _shm_mod
                shm = _shm_mod.SharedMemory(name=shm_name, create=False)
                buf = np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf)
                buf[:] = result.astype(np.float32)
                shm.close()
            except Exception as e:
                # Live-buffer write is display-only; _SyncResult still returns the
                # real computed array, so a failure just skips the live preview.
                log.debug("synchronous live-buffer write to %s failed: %s", shm_name, e)

        class _SyncResult:
            def result(self): return result
            def done(self): return True
            def cancel(self): pass

        return _SyncResult()

    # Both progressive paths go through the SHARED dispatcher: batched submit +
    # bounded in-flight window + stall watchdog + client-side assembly, and no
    # second whole-array graph. See _dispatched_progressive.
    if windowed:
        # VI/FFT stream. Its handle is a plain _ProgressiveFuture, deliberately
        # invisible to PlotUpdateWorker — the stream paints from its own shm
        # buffer and must not also have the assembled array pushed at it.
        return _dispatched_progressive(
            result_array, nav_shape, client, on_chunk_done, on_future,
            stop_event, label="vi", future_cls=_ProgressiveFuture)

    # Navigator fill. Its handle IS the contract PlotUpdateWorker polls, so it
    # carries the _spyde_future marker (_AssembledFuture).
    return _dispatched_progressive(result_array, nav_shape, client,
                                   on_chunk_done, on_future, stop_event,
                                   label="navigator")


def ensure_live_buffer(nav_shape: tuple, shm_name: str) -> "shared_memory.SharedMemory":
    """
    Create (or recreate) a float32 shared memory segment for live display.

    Returns the SharedMemory object — the caller must keep a reference to
    prevent premature cleanup.  Call ``shm.unlink()`` when done.
    """
    from multiprocessing import shared_memory
    nbytes = int(np.prod(nav_shape)) * 4  # float32
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        if shm.size < nbytes:
            shm.close()
            shm.unlink()
            raise FileNotFoundError
        # Zero out existing buffer so old data doesn't show
        buf = np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf)
        buf[:] = np.nan
        return shm
    except (FileNotFoundError, Exception):
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=max(nbytes, 1))
            buf = np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf)
            buf[:] = np.nan
            return shm
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
            buf = np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf)
            buf[:] = np.nan
            return shm


def read_live_buffer(nav_shape: tuple, shm_name: str) -> np.ndarray:
    """Read current contents of live shared-memory buffer into a new array."""
    from multiprocessing import shared_memory
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        arr = np.array(np.ndarray(nav_shape, dtype=np.float32, buffer=shm.buf))
        shm.close()
        return arr
    except Exception:
        return np.full(nav_shape, np.nan, dtype=np.float32)


def stream_progressive_to_plot(plot, result_array, client, *, name="vi"):
    """Progressively compute a nav-shaped ``result_array`` and live-update
    ``plot`` as chunks land — so virtual images / FFTs fill in instead of
    blocking until the whole compute finishes.

    Mirrors the navigator's progressive compute (``compute_with_live_buffer`` +
    a poll loop that pushes partial frames). Any prior stream on ``plot`` is
    stopped first (ROI moves restart the compute). Returns the initial
    NaN-filled display array so the caller's selector can push a blank frame
    immediately; the poll loop then streams in the partial results.

    With ``client is None`` the helper falls back to a synchronous one-shot
    compute (no chunks) — still correct, just not progressive.
    """
    import threading
    import time as _time
    from psygnal import Signal

    nav_shape = tuple(result_array.shape)

    # Tear down any in-flight stream on this plot before starting a new one.
    _stop_progressive_stream(plot)

    if client is None:
        # Synchronous: no chunks land over time, so DON'T start a poll thread —
        # it races with the selector's own `update_data(blank)` (which runs right
        # after this returns) and the output ends up clobbered back to the blank
        # frame (the "virtual image is just black" bug). Compute and return the
        # real data so the caller's selector pushes it.
        try:
            return np.asarray(result_array.compute(), dtype=np.float32)
        except Exception:
            return np.zeros(nav_shape, dtype=np.float32)

    shm_name = f"spyde_{name}_{id(plot)}"
    shm = ensure_live_buffer(nav_shape, shm_name)

    # Blank frame shown immediately while the stream fills in (zeros, not NaN,
    # so the first push is a clean black frame rather than an all-NaN level calc).
    initial = np.zeros(nav_shape, dtype=np.float32)
    stop = threading.Event()

    # Marshal chunk writes off the Dask callback thread via a psygnal relay
    # (slot runs on the emitting thread; writing shm is GIL-safe).
    class _ChunkRelay:
        chunk_ready = Signal(object, object)

    relay = _ChunkRelay()

    def _write_chunk(chunk_result, nav_slices, _shape=nav_shape):
        try:
            buf = np.ndarray(_shape, dtype=np.float32, buffer=shm.buf)
            buf[nav_slices] = np.asarray(chunk_result, dtype=np.float32)
        except Exception as e:
            log.debug("writing progressive chunk %r to %s failed: %s",
                      nav_slices, shm_name, e)

    relay.chunk_ready.connect(_write_chunk)

    def _on_chunk(chunk_result, nav_slices):
        relay.chunk_ready.emit(chunk_result, nav_slices)

    # WINDOWED + CANCELLABLE (the "VI is a completely wild dask task" fix):
    # bounded in-flight chunks (no whole-dataset prefetch/spill), no duplicate
    # full-graph submission, and _stop_progressive_stream's future.cancel()
    # now actually stops EVERYTHING (the old per-chunk futures were
    # unregistered and ran to completion — every ROI drag tick stacked another
    # complete full-dataset pass on the cluster).
    future = compute_with_live_buffer(
        result_array, nav_shape, client, shm_name, on_chunk_done=_on_chunk,
        windowed=True, stop_event=stop,
    )

    levels = [None]

    def _poll_loop():
        # window_computing brackets the whole poll in try/finally so a
        # cancelled stream (a new ROI move calling _stop_progressive_stream,
        # or the compute erroring) still clears the renderer's floating
        # "Calculating…" overlay — see lifecycle.window_computing.
        from spyde.actions.lifecycle import window_computing
        computing = window_computing(getattr(plot, "window_id", None))
        computing.start()
        try:
            while not stop.is_set():
                try:
                    arr = read_live_buffer(nav_shape, shm_name)
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        lo, hi = float(finite.min()), float(finite.max())
                        if levels[0] is None:
                            levels[0] = (lo, hi if hi > lo else lo + 1)
                        elif hi > levels[0][1]:
                            levels[0] = (levels[0][0], hi)
                        plot.set_data(arr, levels=levels[0])
                except Exception as e:
                    log.debug("progressive %s poll paint failed: %s", name, e)
                if future.done():
                    break
                _time.sleep(0.1)
            # Final push of the completed buffer.
            try:
                arr = read_live_buffer(nav_shape, shm_name)
                if np.isfinite(arr).any():
                    plot.set_data(arr, levels=levels[0])
            except Exception as e:
                log.debug("progressive %s final paint failed: %s", name, e)
        finally:
            computing.stop()

    t = threading.Thread(target=_poll_loop, daemon=True, name=f"{name}-poll")
    t.start()

    plot._progressive_stream = {
        "future": future, "stop": stop, "shm": shm, "thread": t, "relay": relay,
    }
    return initial


def _stop_progressive_stream(plot) -> None:
    """Stop and clean up any progressive stream previously started on ``plot``."""
    st = getattr(plot, "_progressive_stream", None)
    if not st:
        return
    try:
        st["stop"].set()
    except Exception as e:
        log.debug("signalling progressive-stream stop failed: %s", e)
    try:
        fut = st.get("future")
        if fut is not None and hasattr(fut, "cancel"):
            fut.cancel()
    except Exception as e:
        log.debug("cancelling progressive-stream future failed: %s", e)
    try:
        shm = st.get("shm")
        if shm is not None:
            shm.close()
            shm.unlink()
    except Exception as e:
        log.debug("cleaning up progressive-stream shared memory failed: %s", e)
    plot._progressive_stream = None


def compute_line_profile_kernel(
    image: np.ndarray,
    roi,
    image_item,
    client: distributed.Client,
) -> distributed.Future:
    """Extract a 1D line profile from a 2D image via LineROI.getArrayRegion.

    Parameters
    ----------
    image : np.ndarray, shape (ny, nx)
        The currently displayed image (plot.image_item.image).
    roi : pyqtgraph.LineROI
    image_item : pyqtgraph.ImageItem
    client : dask distributed Client

    Returns
    -------
    distributed.Future resolving to np.ndarray shape (length_px,)

    Notes
    -----
    LineROI.getArrayRegion returns shape (length_px, width_px).
    nanmean over axis=1 collapses the perpendicular width to give the profile.
    """
    region = roi.getArrayRegion(image, image_item)   # (length_px, width_px)
    profile = np.nanmean(region, axis=1)             # (length_px,)
    return client.submit(lambda p=profile: p)


def compute_nav_line_sum_kernel(
    data: da.Array,
    ys: np.ndarray,
    xs: np.ndarray,
    client: distributed.Client,
    gpu_worker_address: "str | None",
) -> distributed.Future:
    """Compute the mean diffraction pattern over all nav pixels in a line strip.

    Parameters
    ----------
    data : dask array, shape (...nav..., nkx, nky)
        HyperSpy convention: last two axes are signal.
    ys : np.ndarray, shape (N,)
        Row (y) pixel indices of all nav pixels inside the strip.
    xs : np.ndarray, shape (N,)
        Column (x) pixel indices of all nav pixels inside the strip.
    client : dask distributed Client
    gpu_worker_address : str or None

    Returns
    -------
    distributed.Future resolving to np.ndarray shape (nkx, nky)
    """
    # Dask doesn't support multi-dimensional fancy indexing, so loop and vstack
    slices = [data[int(y), int(x)] for y, x in zip(ys, xs)]
    nav_slices = da.stack(slices, axis=0)  # (N, nkx, nky)
    resources = {"GPU": 1} if gpu_worker_address else {}
    with dask.annotate(resources=resources):
        result = da.mean(nav_slices, axis=0)
    return client.compute(result)

"""
update_functions.py

Module containing functions to update a plot based on a selector.  These functions are
called on the move or change events of a selector.

"""

import contextlib
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
from spyde.backend.debug_flags import nav_profile_on as _nav_profile_on


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
    """Best-effort: was the chunk for ``indices`` already resident in the signal's
    CachedDaskArray before this read? A HIT is ~ms (numpy slice of a resident
    block); a MISS reads the chunk off disk. Used only for the profile line —
    returns False if it can't tell (no cache yet / probe failed), never raises.
    Cheap and side-effect-free: it only inspects the cache's block-index list."""
    if not _nav_profile_on():
        return False
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


class _MoviePrefetcher:
    """Warm the OS page cache for the frames a movie scrub is about to reach.

    A cold single-frame read of a large in-situ movie is disk-bound (~50 ms);
    once the file pages are in the OS cache a re-read is ~18 ms (benchmarks.md).
    After the navigator paints frame ``t`` this reads a few upcoming frames
    (``t±1 … t±radius``) on a single background daemon thread, purely to trigger
    the page-in — so a steady scrub/playback finds each next frame already warm.

    Safety: it reads the **raw dask array directly**
    (``raw[i].compute(scheduler="synchronous")``), NOT the ``CachedDaskArray`` the
    navigator read uses — so it never touches hyperspy's (non-concurrency-safe)
    cache bookkeeping (CLAUDE.md §4). The OS page cache it warms is process-global
    and thread-safe. Latest-center-wins: a new ``prime`` replaces the pending
    target set, so a fast scrub doesn't pile up stale reads.
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


def _direct_read_frame(current_signal, selector, indices, prof):
    """Unified fast VIEW read: compute the requested nav slice DIRECTLY with the
    synchronous scheduler and return the ndarray — bypassing hyperspy's
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
            if not is_region:
                # Single point (crosshair): one frame, native dtype.
                point = tuple(int(v) for v in np.atleast_1d(idx))
                frame = np.asarray(
                    data[point].compute(scheduler="synchronous"))
            else:
                # Integrating region: N nav points → frame-wise mean, accumulated
                # INCREMENTALLY (read one frame, add to a float accumulator, free
                # it) so peak memory is ~ONE frame, not the whole block. Reading a
                # 59-frame region as one vindex block peaked ~2 GB; incremental
                # peaks ~1 frame at the SAME speed (it's disk-bound either way).
                # No size cap needed — memory is bounded by construction.
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

        # Read-ahead for a MOVIE scrub (1-D time nav, single point): warm the OS
        # page cache for the neighbouring frames off-thread. A 4D-STEM DP dwells in
        # small in-RAM-cheap chunks and an integrating region has no single "next
        # frame", so prefetch is movie-only.
        with prof.stage("prefetch"):
            try:
                if nav_dim == 1 and idx.ndim <= 1:
                    n_time = int(data.shape[0])
                    center = int(np.atleast_1d(idx).ravel()[0])
                    _movie_prefetcher.prime(data, center, n_time)
            except Exception as _e:
                log.debug("movie prefetch prime failed: %s", _e)
        prof.done("direct")
        return frame
    except Exception as _e:
        log.debug("direct-read failed, falling back to cached get_index: %s", _e)
        return None


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

def update_from_navigation_selection(
        selector: "BaseSelector",
        child: "Plot",
        indices,
        get_result: bool = False,
        cache_in_shared_memory: bool = True,
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
    cache_in_shared_memory : bool
        DEPRECATED / NO-OP. The lazy nav read is now a synchronous cached read that
        returns a numpy array directly (no distributed Future, no shared-memory
        buffer — see §3 of CLAUDE.md Live-Display). This parameter no longer has any
        effect; it remains only so existing callers don't break. Do not rely on it.
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

    # Per-frame trace — gated behind SPYDE_NAV_TIMING because it fires on EVERY
    # crosshair move and floods the IPC log/panel at DEBUG (which itself adds lag).
    if _NAV_TIMING:
        log.debug(f"update_from_navigation_selection: indicies = {indices}, current_signal = {current_signal}, "
                  f"selector = {selector}, child = {child}, get_result = {get_result},"
                  f" cache_in_shared_memory = {cache_in_shared_memory}")

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
    indices = np.asarray(indices)
    _has_spatial_nav = False
    try:
        _has_spatial_nav = current_signal.axes_manager.signal_dimension == 2
    except Exception:
        _has_spatial_nav = indices.ndim >= 1 and indices.shape[-1] == 2
    if _has_spatial_nav and indices.ndim >= 1 and indices.shape[-1] >= 2:
        indices = indices.copy()
        indices[..., -2:] = indices[..., -2:][..., ::-1]

    if not selector.is_integrating:
        indices = np.mean(indices, axis=0).astype(int)

    # Clamp nav indices to the leading (navigation) axis sizes. The signal behind
    # a plot can change (e.g. set_signal_type, or loading a SECOND signal) while a
    # selector still holds positions from the previous, larger nav grid — the
    # subsequent data[...] index would raise IndexError ("Index N out of bounds
    # for axis 0 with size N"). Clamping keeps the display valid until the
    # selector catches up to the new shape.
    #
    # Use the data array's leading-axis sizes as the AUTHORITATIVE per-coordinate
    # bound. `indices` here is (y, x, …) order (post-transpose), matching the data
    # axes; data_shape's leading nav axes are (ny, nx, …). A distributed signal's
    # `.data` may be a Future (no `.shape`) — fall back to axes_manager then.
    try:
        data_obj = current_signal.data
        data_shape = getattr(data_obj, "shape", None)
        if data_shape is None:
            # Future / unshaped — derive the nav-axis sizes from the axes manager
            # (navigation_shape is (x, y, …); reverse to (…, y, x) array order).
            nav_shape_xy = tuple(current_signal.axes_manager.navigation_shape)
            data_shape = tuple(reversed(nav_shape_xy))
        idx_arr = np.asarray(indices)
        if idx_arr.size and len(data_shape):
            # Last axis holds the per-point coordinates (one per leading data
            # axis); clamp each coordinate to its axis size.
            ncoord = idx_arr.shape[-1] if idx_arr.ndim else 1
            n = min(ncoord, len(data_shape))
            limits = np.array([data_shape[i] - 1 for i in range(n)], dtype=idx_arr.dtype)
            before = idx_arr.copy()
            if limits.size == ncoord:
                indices = np.clip(idx_arr, 0, limits)
            else:
                # ncoord != available bounds: clamp the coordinates we CAN bound
                # rather than skipping entirely (the old code skipped, which let
                # the IndexError through on a mismatched/changed signal).
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
        # Couldn't clamp (unexpected shape) — proceed with raw indices; a real
        # out-of-range index then surfaces as an IndexError downstream.
        log.debug("clamping nav indices failed: %s", e)

    if current_signal._lazy:
        if isinstance(current_signal.data[0], Future):
            current_img = np.ones(current_signal.axes_manager.signal_shape, dtype=np.int8)
            if current_img.ndim == 2:
                #make checkerboard pattern to indicate loading
                current_img[::2, ::2] = 0
        elif (_direct := _direct_read_frame(
                current_signal, selector, indices, _prof)) is not None:
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
            # _inflight_getinds juggling — see benchmarks.md "unified read".
            #
            # Serial-only: the cache's block bookkeeping is not concurrency-safe
            # ("ValueError: (i, j) is not in list"); the dispatcher already
            # guarantees this is never re-entered concurrently, so no lock (§4).
            #
            # Force the SYNCHRONOUS cache path (fast: ~1-2 ms dwell-in-chunk hits
            # vs ~16 ms distributed). Setting _client=None requests it, AND
            # heavy_imports._patch_cached_dask_client makes the cache honour that
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
            # sources keep their (un-rounded) mean. (benchmarks.md rounding gotcha.)
            with _prof.stage("dtype"):
                try:
                    src_dtype = getattr(current_signal.data, "dtype", None)
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
            # vs ~50 ms cold — benchmarks.md). Only for a 1-D (time) navigator on
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
                            and hasattr(current_signal.data, "shape")):
                        n_time = int(current_signal.data.shape[0])
                        center = int(np.atleast_1d(_idx).ravel()[0])
                        _movie_prefetcher.prime(current_signal.data, center, n_time)
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
        idx = np.asarray(indices)
        try:
            with _prof.stage("read"):
                if idx.ndim <= 1:
                    point = tuple(int(v) for v in np.atleast_1d(idx))
                    current_img = current_signal.data[point]
                else:
                    sl = tuple(idx[:, k].astype(int) for k in range(idx.shape[1]))
                    current_img = current_signal.data[sl].mean(axis=0)
            _prof.set_frame(current_img)
        except Exception:
            log.exception(
                "NAV-DEBUG eager index RAISED: indices=%s data.shape=%s "
                "nav_shape=%s — the second-signal IndexError",
                idx.tolist(),
                getattr(getattr(current_signal, "data", None), "shape", None),
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


def compute_with_live_buffer(
    result_array: da.Array,
    nav_shape: tuple,
    client: distributed.Client,
    shm_name: str,
    on_chunk_done=None,
) -> distributed.Future:
    """
    Progressive compute: submits one Future per nav chunk and calls
    ``on_chunk_done(chunk_result, nav_slices)`` from a Dask callback thread
    as each chunk completes.  The caller is responsible for marshalling
    back to the GUI thread (e.g. via a Qt Signal).

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
    """
    import itertools

    nav_ndim = len(nav_shape)
    nav_chunks = result_array.chunks

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

    # Build per-nav-chunk futures
    axes_ranges = []
    for axis_chunks in nav_chunks:
        positions, start = [], 0
        for size in axis_chunks:
            positions.append((start, size))
            start += size
        axes_ranges.append(positions)

    chunk_futures = []
    chunk_slices = []
    for combo in itertools.product(*axes_ranges):
        slices = tuple(slice(s, s + n) for s, n in combo)
        full_slice = slices + (slice(None),) * (result_array.ndim - nav_ndim)
        chunk_da = result_array[full_slice]
        fut = client.compute(chunk_da)
        chunk_futures.append(fut)
        chunk_slices.append(slices)

    # Attach callbacks — run in Dask callback threads, must be thread-safe
    def _make_cb(fut, nav_slices):
        def _cb(f):
            try:
                chunk_result = f.result()
                if on_chunk_done is not None:
                    on_chunk_done(chunk_result, nav_slices)
            except Exception as e:
                # This is the live-preview path only; a genuinely failed chunk
                # re-raises when the commit path calls full_future.result().
                log.debug("live chunk callback for %r failed: %s", nav_slices, e)
        return _cb

    if on_chunk_done is not None:
        for fut, nav_slices in zip(chunk_futures, chunk_slices):
            fut.add_done_callback(_make_cb(fut, nav_slices))

    # Return the whole-array future for progress indicator and commit path
    full_future = client.compute(result_array)
    return full_future


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

    future = compute_with_live_buffer(
        result_array, nav_shape, client, shm_name, on_chunk_done=_on_chunk
    )

    levels = [None]

    def _poll_loop():
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

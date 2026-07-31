"""
compute_dispatch.py — greedy dual-lane (GPU/CPU) chunk dispatcher for
heterogeneous clusters, shared by the batch actions (find_vectors,
orientation mapping).

Why this exists: dask's scheduler keeps ONE duration estimate per task
family, so it can never learn that the designated GPU worker finishes a
chunk ~30x faster than a CPU worker — chunk tasks get placed for data
locality and queue behind slow siblings while the GPU idles.  This
dispatcher takes over placement: per-nav-chunk slices are submitted with
*loose* worker restrictions, each lane keeps a bounded in-flight window,
and every completion pulls the next chunk from the shared pending queue,
so both lanes drain the same pool and finish together.

Hard-won rules baked in (measured on the 60 GB benchmark — see
benchmarks.md):
  - `scheduler_info(n_workers=-1)` everywhere (default truncates to 5!).
  - allow_other_workers=True: hard pins deadlock when a worker restarts.
  - Futures are HELD until the end (mid-run release of graphs that share
    input keys races the scheduler -> KeyError); per-chunk `postprocess`
    keeps the held results small.
  - Batched submissions amortise per-compute graph-cull cost.
  - Banded (2-row) order keeps ghost-sharing neighbours temporally close.
  - Stall watchdog + periodic lane refresh (workers register late while the
    app cluster scales 1 -> N in the background).
"""
from __future__ import annotations

import collections
import functools
import itertools
import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

# Never-set event used as an interruptible sleep. MEASURED on the Windows/
# Electron-spawned backend: `time.sleep(0.05)` in a poll loop froze for 15 s
# (woken only by process I/O — timer coalescing of the hidden child process)
# while `threading.Event.wait(timeout)` in the same process ticked exactly on
# schedule 120/120 times. Poll loops that must make progress use this.
_WAKE = threading.Event()


def reliable_sleep(seconds: float) -> None:
    """Sleep that keeps ticking on the throttled Electron-spawned backend
    (Event.wait-based — see _WAKE). Use instead of time.sleep in poll loops."""
    _WAKE.wait(seconds)


def split_workers_for_gpu(client, default_mode: str = "one") -> tuple:
    """
    Partition cluster workers into (gpu_addrs, cpu_addrs) per SPYDE_FV_GPU.

    Returns ([], []) when GPU-aware dispatching should be disabled:
    mode "off"/"all", no CUDA on this machine, or no split possible
    (the GPU lane and the CPU lane each need at least one worker).

    ``default_mode`` is the policy when SPYDE_FV_GPU is UNSET and must match
    the chunk fn's ``_gpu_task_allowed`` default for the method being
    dispatched — a mismatched lane split (sized for one GPU worker while every
    worker actually submits CUDA) was exactly the all-workers contention that
    made the GPU slower AND lagged the desktop (real-data A/B 2026-07-16:
    2 GPU workers beat 9 on throughput and smoothness).
    """
    mode = os.environ.get("SPYDE_FV_GPU", default_mode).lower()
    if mode in ("off", "all"):
        return [], []
    try:
        from numba import cuda as _nc
        if not _nc.is_available():
            return [], []
    except Exception:
        return [], []
    try:
        n_gpu = max(1, int(mode))
    except ValueError:
        n_gpu = 1
    try:
        info = client.scheduler_info(n_workers=-1)["workers"]
    except Exception:
        return [], []
    gpu_addrs, cpu_addrs = [], []
    for addr, w in info.items():
        name = str(w.get("name"))
        try:
            is_gpu = 1 <= int(name) <= n_gpu
        except (TypeError, ValueError):
            is_gpu = name == "1"
        (gpu_addrs if is_gpu else cpu_addrs).append(addr)
    if not gpu_addrs or not cpu_addrs:
        return [], []
    return gpu_addrs, cpu_addrs


def poke_scheduler(client, label: str = "poke") -> None:
    """Fire the empirically-proven "unstick" trio of client round-trips.

    On this Windows/Electron-spawned backend, submitted tasks can sit
    {waiting, processing} with every worker idle INDEFINITELY — task delivery
    only resumes after certain client↔scheduler/worker traffic (measured with
    _probe_fv_stall.spec.ts; standalone the same code is healthy, see
    repro_batch_stall.py). No SINGLE call type unsticks it reliably
    (stochastic across runs), but the full trio below did 4/4 times, ~4 s to
    completion each. The stall watchdogs call this every few seconds while a
    compute makes no progress — a no-op cost on a healthy cluster (it never
    fires) and bounded staleness on this one.
    """
    try:
        client.run_on_scheduler(
            lambda dask_scheduler: {
                s: sum(1 for t in dask_scheduler.tasks.values() if t.state == s)
                for s in {t.state for t in dask_scheduler.tasks.values()}})
        client.scheduler_info(n_workers=-1)
        client.call_stack()
        log.info("[%s] stall poke fired (scheduler+info+call_stack)", label)
    except Exception as e:
        log.debug("[%s] stall poke failed: %s", label, e)


# Memory-pressure hysteresis for the dispatch window (fractions of a worker's
# memory_limit): shrink the window when ANY worker crosses HOT (just under
# dask's own 0.6-spill/0.8-pause thresholds so we back off BEFORE the
# spill-to-disk thrash starts), restore it below COOL.
MEM_HOT_FRAC = 0.72
MEM_COOL_FRAC = 0.60


def _lane_cap(base_cap: int, threads: int, mem_hot: bool) -> int:
    """Effective in-flight window for a lane: the full cap normally, ~half the
    lane's threads while cluster memory is hot (never below 2 — the workers
    must keep making progress to drain the pressure)."""
    if not mem_hot:
        return base_cap
    return min(base_cap, max(2, threads // 2))


def _cluster_mem_frac(client) -> float:
    """Peak worker memory as a fraction of its limit (0.0 when unknown)."""
    info = client.scheduler_info(n_workers=-1).get("workers") or {}
    peak = 0.0
    for w in info.values():
        limit = float(w.get("memory_limit") or 0)
        if limit <= 0:
            continue
        used = float((w.get("metrics") or {}).get("memory", 0))
        peak = max(peak, used / limit)
    return peak


def dispatch_chunks(
    client,
    result_array,
    nav_dim: int,
    gpu_addrs: list,
    cpu_addrs: "list | None",
    stopped_flag=None,
    postprocess=None,
    fill_value=np.nan,
    stall_timeout_s: float = 600.0,
    submit_batch: int = 8,
    label: str = "dispatch",
    on_chunk_done=None,
    lane_default_mode: str = "one",
    gpu_only: bool = False,
    assemble=None,
    cap=None,
    on_start=None,
):
    """
    Compute `result_array` (a dask array with nav dims leading) chunk by
    chunk with explicit lane placement, assembling into one host ndarray.

    THIS IS THE ONLY PLACE THAT MAY SUBMIT PER-CHUNK COMPUTES IN A LOOP.
    Every other progressive-chunk path (navigator fill, VI/FFT stream, the
    single-lane find-vectors preview) routes through here — see
    ``test_chunk_dispatch_guard.py``, which fails the build if a new
    ``client.compute()`` appears inside a chunk loop elsewhere.  A hand-rolled
    loop always ends up missing at least one of the three properties below:
    batched submit, a bounded in-flight window, and the stall watchdog.

    cpu_addrs : list | None
        ``None`` means ONE UNPINNED LANE over the whole cluster — chunks are
        submitted with no ``workers=`` restriction, so the scheduler places
        them by data locality.  That is the plain progressive-chunk case (no
        GPU/CPU asymmetry to exploit) and it is what the navigator fill / VI
        stream want.  An empty LIST keeps its existing, different meaning: a
        lane that must NEVER be submitted to (the ``gpu_only`` pin — a
        ``workers=[]`` submit silently leaks chunks to any worker, which for
        neural is the 10-50x-slower torch-CPU path).
    cap : int | None
        Explicit in-flight window for the unpinned lane.  Default is HALF the
        cluster's threads (min 4): a progressive chunk is read-bound, so full
        saturation buys no throughput while every in-flight chunk pins a whole
        source chunk in worker RAM — a full-thread window was effectively no
        throttle at all on medium scans and pushed workers into spill.

    postprocess : callable(np.ndarray) -> np.ndarray | None
        Applied on the worker to each chunk's result before transfer (e.g.
        trimming NaN padding).  May shorten axis -2; the assembly writes the
        result into slots [0:n) of that axis, the rest keeps `fill_value`.
    assemble : callable(result, nav_slices, chunk_result) | None
        How a landed chunk is written into the output array.  The default is
        find_vectors' NaN-padded convention (slots [0:n) of axis -2), which
        assumes at least two trailing axes.  Segmentation's per-frame result is
        RAGGED — a variable number of particle rows plus a variable number of
        contours per frame — so it dispatches a 1-D **object** array (one
        ``(rows, contours)`` tuple per frame) and passes an assembler that just
        writes the slice.  Everything else about the lane machinery is
        identical, which is the point: one dispatcher, two payload shapes.
    on_start : callable(cancel) | None
        Called once, before the first submission, with a zero-argument
        ``cancel()`` that requests an IMMEDIATE stop: it flips the stop flag AND
        wakes the wait loop, so outstanding futures are cancelled right away
        instead of on the next 0.5 s poll.  Interactive callers need this — the
        VI/FFT stream restarts on every ROI drag tick and a superseded stream
        that keeps computing for another half second is exactly the "every drag
        tick stacks another dataset pass" pathology, only smaller.  Batch
        callers can ignore it and use ``stopped_flag`` alone.
    on_chunk_done : callable(nav_slices, chunk_result) | None
        Called from the Dask done-callback thread as each chunk lands, with
        the chunk's GLOBAL nav slice (a tuple of slices into the full nav
        grid) and the chunk's (post-processed) result.  Used to drive a live
        preview from the client side — counting/writing happens here, NOT in
        the dask graph, so the global location is always correct (slicing the
        array per-chunk resets block_info to local coords).  Must be
        thread-safe and never raise; exceptions are swallowed.

    Returns the assembled ndarray, or None when stopped via stopped_flag.
    Raises the first task exception encountered.
    """
    import dask as _dask

    nav_chunks = result_array.chunks[:nav_dim]
    axes_ranges = []
    for axis_chunks in nav_chunks:
        positions, start = [], 0
        for size in axis_chunks:
            positions.append((start, size))
            start += size
        axes_ranges.append(positions)
    chunk_slices = [
        tuple(slice(s, s + n) for s, n in combo)
        for combo in itertools.product(*axes_ranges)
    ]
    n_total = len(chunk_slices)
    trailing = (slice(None),) * (result_array.ndim - nav_dim)

    result = np.full(result_array.shape, fill_value, dtype=result_array.dtype)

    # cpu_addrs is None -> one UNPINNED lane over the whole cluster (see the
    # docstring).  `lanes["cpu"]` stays [] in that mode and the empty-lane guard
    # in _submit_next is bypassed for it, so `[]` keeps meaning "never submit".
    unpinned = cpu_addrs is None
    cpu_addrs = [] if unpinned else cpu_addrs

    # Thread counts only SIZE the windows — a client that can't answer (a test
    # double, a scheduler hiccup) must degrade to the conservative default, not
    # fail the whole dispatch.
    try:
        info = client.scheduler_info(n_workers=-1)["workers"]
    except Exception as e:
        log.debug("[%s] scheduler_info unavailable (%s); using default windows",
                  label, e)
        info = {}
    gpu_threads = sum(int(info[a].get("nthreads", 1)) for a in gpu_addrs if a in info)
    if unpinned:
        cpu_threads = sum(int(w.get("nthreads", 1)) for w in info.values())
    else:
        cpu_threads = sum(int(info[a].get("nthreads", 1)) for a in cpu_addrs if a in info)
    # GPU lane gets a deeper window (chunks overlap loads/transfers/kernels on
    # per-thread streams); CPU margin stays small — every in-flight chunk pins
    # its inputs on a worker and over-prefetching pushes workers into spill.
    if cap is not None:
        cpu_cap = max(1, int(cap))
    elif unpinned:
        cpu_cap = max(4, cpu_threads // 2)
    else:
        cpu_cap = max(2, cpu_threads + 2)
    caps = {
        "gpu": max(2, 2 * gpu_threads + 2),
        "cpu": cpu_cap,
    }
    # _lane_cap halves THIS number under memory pressure. For a pinned lane it
    # is the lane's thread count (the natural sizing); the unpinned lane has no
    # thread-based sizing, so feed it the window itself and hot => window/2.
    lane_threads = {"gpu": gpu_threads,
                    "cpu": cpu_cap if unpinned else cpu_threads}
    lanes = {"gpu": list(gpu_addrs), "cpu": list(cpu_addrs)}

    # Banded (2-row) submission order: vertical ghost-zone neighbours stay
    # temporally close so shared input tasks deduplicate while both futures
    # are alive instead of being recomputed a row later.
    def _band_key(i):
        pos = []
        rem = i
        for ar in reversed(axes_ranges):
            pos.append(rem % len(ar))
            rem //= len(ar)
        pos.reverse()
        iy = pos[-2] if len(pos) >= 2 else 0
        ix = pos[-1]
        return tuple(pos[:-2]) + (iy // 2, ix, iy % 2)

    lock = threading.Lock()
    done_event = threading.Event()
    pending = collections.deque(sorted(range(n_total), key=_band_key))
    futures: set = set()
    completed_futures: list = []  # held until the end — see module docstring
    outstanding = {"gpu": 0, "cpu": 0}
    state = {"completed": 0, "error": None, "last_progress": time.time(),
             "lane_done": {"gpu": 0, "cpu": 0}, "mem_hot": False}

    def _submit_next(lane):
        """Top up `lane` with a batch of pending chunks (lock held).

        MEMORY BACKPRESSURE: while cluster memory is hot (see the wait-loop
        sampler) the effective window shrinks to ~half the lane's threads —
        the loading/producing side pauses and lets in-flight chunks drain
        instead of buffering the workers into spill. Top-ups are only ever
        DEFERRED (the pending deque is untouched, every completion re-calls
        this), so the throttle cannot wedge the dispatch."""
        if state["error"] is not None or not pending:
            return
        if stopped_flag is not None and stopped_flag[0]:
            return
        if not lanes[lane] and not (unpinned and lane == "cpu"):
            # Empty lane (gpu_only dispatch: torch-CPU inference is 10-50x
            # slower than the GPU batch, so the CPU lane would only stretch
            # the tail — and `workers=[] + allow_other_workers` would leak
            # chunks to ANY worker). Never submit here.
            # The UNPINNED lane is the deliberate exception: it has no address
            # list precisely because it wants no placement restriction.
            return
        # NB `lane_cap`, not `cap` — `cap` is the caller's window override and is
        # already folded into `caps` above; shadowing it here would read as if
        # the parameter were being recomputed per top-up.
        lane_cap = _lane_cap(caps[lane], lane_threads[lane], state["mem_hot"])
        n = min(submit_batch, len(pending), lane_cap - outstanding[lane])
        if n <= 0:
            return
        idxs = [pending.popleft() for _ in range(n)]
        if postprocess is not None:
            delayeds = [
                _dask.delayed(postprocess)(
                    result_array[chunk_slices[i] + trailing]
                )
                for i in idxs
            ]
        else:
            delayeds = [result_array[chunk_slices[i] + trailing] for i in idxs]
        # ONE batched client.compute per top-up, never one per chunk: a
        # per-chunk submit is a blocking scheduler round trip with the GIL held
        # in the client process (14.2 ms each measured on a real memmap graph —
        # 13.8 s of dead UI before the first of 977 navigator chunks painted).
        # `client.compute(list)` returns futures in the SAME ORDER as the input.
        #
        # gpu_only dispatch pins tasks HARD to the lane: with
        # allow_other_workers=True the list is only a soft preference, and a
        # busy GPU lane silently leaks chunks onto CPU workers — for neural
        # that is the 10-50x-slower torch-CPU path (the e2e batch timed out
        # exactly this way). Dual-lane mode keeps the soft placement.
        if unpinned and lane == "cpu":
            futs = client.compute(delayeds)
        else:
            futs = client.compute(
                delayeds, workers=lanes[lane], allow_other_workers=not gpu_only,
            )
        if not isinstance(futs, (list, tuple)):
            futs = [futs]          # a 1-element submit can come back bare
        for i, fut in zip(idxs, futs):
            futures.add(fut)
            outstanding[lane] += 1
            fut.add_done_callback(
                functools.partial(_on_chunk_future_done, idx=i, lane=lane)
            )

    def _on_chunk_future_done(fut, idx, lane):
        # An exception escaping this callback would strand the dispatcher
        # in an infinite wait — catch everything and convert to an error.
        try:
            chunk_result = fut.result()
            # Disjoint nav slices — safe to write without holding the lock.
            if assemble is not None:
                assemble(result, chunk_slices[idx], chunk_result)
            else:
                n_found = chunk_result.shape[-2]
                result[chunk_slices[idx] + (slice(0, n_found), slice(None))] = \
                    chunk_result
            if on_chunk_done is not None:
                # Live preview: hand the caller this chunk's GLOBAL nav slice
                # and its result so it can paint/write shm at the right place.
                try:
                    on_chunk_done(chunk_slices[idx], chunk_result)
                except Exception as e:
                    log.debug("[%s] on_chunk_done failed: %s", label, e)
        except Exception as exc:
            with lock:
                futures.discard(fut)
                if state["error"] is None and not (
                    stopped_flag is not None and stopped_flag[0]
                ):
                    state["error"] = exc
            done_event.set()
            return
        with lock:
            futures.discard(fut)
            completed_futures.append(fut)
            outstanding[lane] -= 1
            state["completed"] += 1
            state["last_progress"] = time.time()
            state["lane_done"][lane] += 1
            try:
                _submit_next(lane)
            except Exception as exc:
                if state["error"] is None:
                    state["error"] = exc
                done_event.set()
                return
            if state["completed"] >= n_total or (not pending and not futures):
                done_event.set()

    t0 = time.time()
    if stopped_flag is None:
        stopped_flag = [False]
    if on_start is not None:
        # Hand the caller a zero-latency stop: flipping the flag alone would
        # only take effect on the next 0.5 s wait-loop poll.
        def _request_stop(_flag=stopped_flag, _evt=done_event):
            _flag[0] = True
            _evt.set()
        try:
            on_start(_request_stop)
        except Exception as e:
            log.debug("[%s] on_start hook failed: %s", label, e)
    with lock:
        for lane in ("gpu", "cpu"):
            while outstanding[lane] < caps[lane] and pending:
                before = outstanding[lane]
                _submit_next(lane)
                if outstanding[lane] == before:
                    break
        if not futures:
            done_event.set()

    last_lane_refresh = time.time()
    last_poke = time.time()
    last_mem_check = time.time()
    while not done_event.wait(timeout=0.5):
        if stopped_flag is not None and stopped_flag[0]:
            break
        # Memory backpressure sampler (~3 s): while any worker sits above
        # MEM_HOT_FRAC of its limit, _submit_next holds the window at ~half
        # the lane threads so the producing side stops outrunning the
        # consumers (the "loading steps" throttle). Hysteresis avoids
        # flapping; the hot→cool transition kicks a top-up pass because
        # completions alone may be sparse at the shrunken window.
        if time.time() - last_mem_check > 3.0:
            last_mem_check = time.time()
            try:
                peak = _cluster_mem_frac(client)
                with lock:
                    was_hot = state["mem_hot"]
                    state["mem_hot"] = (peak > MEM_HOT_FRAC if not was_hot
                                        else peak > MEM_COOL_FRAC)
                    now_hot = state["mem_hot"]
                    if not now_hot and was_hot:
                        for lane in ("gpu", "cpu"):
                            while (outstanding[lane]
                                   < _lane_cap(caps[lane], lane_threads[lane], False)
                                   and pending):
                                before = outstanding[lane]
                                _submit_next(lane)
                                if outstanding[lane] == before:
                                    break
                if now_hot != was_hot:
                    log.info("[%s] memory %s (peak worker %.0f%% of limit) — "
                             "dispatch window %s", label,
                             "HOT" if now_hot else "cool", peak * 100,
                             "shrunk" if now_hot else "restored")
            except Exception as e:
                log.debug("[%s] memory pressure sample failed: %s", label, e)
        # No-progress watchdog poke (see poke_scheduler): on the frozen-
        # delivery pathology, tasks sit assigned-but-undelivered until client
        # traffic arrives — re-poke every 5 s of no progress until they move.
        now = time.time()
        if now - state["last_progress"] > 5.0 and now - last_poke > 5.0:
            last_poke = now
            poke_scheduler(client, label)
        # Lane refresh: fold in workers that registered after dispatch start.
        # The unpinned lane has nothing to refresh — it never restricts
        # placement, so a late worker picks up chunks on its own.
        if not unpinned and time.time() - last_lane_refresh > 5.0:
            last_lane_refresh = time.time()
            try:
                # Same unset-default as the initial split — a refresh must not
                # silently change the lane policy mid-run.
                new_gpu, new_cpu = split_workers_for_gpu(client, lane_default_mode)
                if gpu_only:
                    new_cpu = []
                if new_gpu and (new_cpu or gpu_only):
                    with lock:
                        if set(new_cpu) != set(lanes["cpu"]) or \
                                set(new_gpu) != set(lanes["gpu"]):
                            lanes["gpu"][:] = new_gpu
                            lanes["cpu"][:] = new_cpu
                            w_info = client.scheduler_info(n_workers=-1)["workers"]
                            new_cpu_threads = sum(
                                int(w_info[a].get("nthreads", 1))
                                for a in new_cpu if a in w_info
                            )
                            lane_threads["cpu"] = new_cpu_threads
                            caps["cpu"] = max(2, new_cpu_threads + 2)
                            log.debug(
                                "[%s] dispatcher lanes refreshed: GPU=%d CPU=%d workers",
                                label, len(new_gpu), len(new_cpu),
                            )
                            while (outstanding["cpu"] < caps["cpu"]
                                   and pending):
                                before = outstanding["cpu"]
                                _submit_next("cpu")
                                if outstanding["cpu"] == before:
                                    break
            except Exception as e:
                # Submission hiccup — the stall watchdog below still fires the
                # user-facing error if dispatch genuinely makes no progress.
                log.debug("dispatcher submit pass failed: %s", e)
        with lock:
            stalled = bool(futures) and (
                time.time() - state["last_progress"] > stall_timeout_s
            )
            n_stuck = len(futures)
        if stalled:
            with lock:
                if state["error"] is None:
                    state["error"] = RuntimeError(
                        f"{label} dispatcher stalled: {n_stuck} chunk "
                        f"task(s) made no progress for {stall_timeout_s:.0f}s "
                        f"(worker restarted or task unschedulable?)"
                    )
            break

    def _cleanup(cancel_outstanding):
        # No further submissions can happen now, so releasing the held
        # futures (and their shared input keys) is race-free.
        with lock:
            outstanding_futs = list(futures)
            held = list(completed_futures)
            completed_futures.clear()
        if cancel_outstanding:
            for fut in outstanding_futs:
                try:
                    fut.cancel()
                except Exception as e:
                    log.debug("cancelling outstanding dispatch future failed: %s", e)
        for fut in held:
            try:
                fut.release()
            except Exception as e:
                log.debug("releasing held dispatch future failed: %s", e)

    if stopped_flag is not None and stopped_flag[0]:
        _cleanup(cancel_outstanding=True)
        return None

    if state["error"] is not None:
        _cleanup(cancel_outstanding=True)
        raise state["error"]

    _cleanup(cancel_outstanding=False)
    dt = max(time.time() - t0, 1e-9)
    log.debug(
        "[%s] dispatcher done: %d GPU + %d CPU chunks in %.1f s (%.2f chunks/s)",
        label, state['lane_done']['gpu'], state['lane_done']['cpu'], dt, n_total / dt,
    )
    return result

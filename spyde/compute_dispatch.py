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


def split_workers_for_gpu(client) -> tuple:
    """
    Partition cluster workers into (gpu_addrs, cpu_addrs) per SPYDE_FV_GPU.

    Returns ([], []) when GPU-aware dispatching should be disabled:
    mode "off"/"all", no CUDA on this machine, or no split possible
    (the GPU lane and the CPU lane each need at least one worker).
    """
    mode = os.environ.get("SPYDE_FV_GPU", "one").lower()
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


def dispatch_chunks(
    client,
    result_array,
    nav_dim: int,
    gpu_addrs: list,
    cpu_addrs: list,
    stopped_flag=None,
    postprocess=None,
    fill_value=np.nan,
    stall_timeout_s: float = 600.0,
    submit_batch: int = 8,
    label: str = "dispatch",
    on_chunk_done=None,
):
    """
    Compute `result_array` (a dask array with nav dims leading) chunk by
    chunk with explicit lane placement, assembling into one host ndarray.

    postprocess : callable(np.ndarray) -> np.ndarray | None
        Applied on the worker to each chunk's result before transfer (e.g.
        trimming NaN padding).  May shorten axis -2; the assembly writes the
        result into slots [0:n) of that axis, the rest keeps `fill_value`.
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

    info = client.scheduler_info(n_workers=-1)["workers"]
    gpu_threads = sum(int(info[a].get("nthreads", 1)) for a in gpu_addrs if a in info)
    cpu_threads = sum(int(info[a].get("nthreads", 1)) for a in cpu_addrs if a in info)
    # GPU lane gets a deeper window (chunks overlap loads/transfers/kernels on
    # per-thread streams); CPU margin stays small — every in-flight chunk pins
    # its inputs on a worker and over-prefetching pushes workers into spill.
    caps = {
        "gpu": max(2, 2 * gpu_threads + 2),
        "cpu": max(2, cpu_threads + 2),
    }
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
             "lane_done": {"gpu": 0, "cpu": 0}}

    def _submit_next(lane):
        """Top up `lane` with a batch of pending chunks (lock held)."""
        if state["error"] is not None or not pending:
            return
        if stopped_flag is not None and stopped_flag[0]:
            return
        n = min(submit_batch, len(pending), caps[lane] - outstanding[lane])
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
        futs = client.compute(
            delayeds, workers=lanes[lane], allow_other_workers=True,
        )
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
    while not done_event.wait(timeout=0.5):
        if stopped_flag is not None and stopped_flag[0]:
            break
        # Lane refresh: fold in workers that registered after dispatch start
        if time.time() - last_lane_refresh > 5.0:
            last_lane_refresh = time.time()
            try:
                new_gpu, new_cpu = split_workers_for_gpu(client)
                if new_gpu and new_cpu:
                    with lock:
                        if set(new_cpu) != set(lanes["cpu"]) or \
                                set(new_gpu) != set(lanes["gpu"]):
                            lanes["gpu"][:] = new_gpu
                            lanes["cpu"][:] = new_cpu
                            w_info = client.scheduler_info(n_workers=-1)["workers"]
                            caps["cpu"] = max(2, sum(
                                int(w_info[a].get("nthreads", 1))
                                for a in new_cpu if a in w_info
                            ) + 2)
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

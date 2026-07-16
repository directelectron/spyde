"""
orchestrate.py — dask/distributed batch orchestration for find_vectors.

The public batch entry point ``_do_compute_vectors`` and its supporting chunk
dispatch: nav-chunk sizing, balanced chunking, the GPU-aware dual-lane
dispatcher, the single-lane live-count compute, and packing the padded peak
result into a SpyDEDiffractionVectors flat buffer.

MEMORY SAFETY (see CLAUDE.md): ``_do_compute_vectors`` must NEVER call
.compute()/.result() on the full signal dataset — only the small padded-peaks
output or per-chunk ghost slices.  ``_nav_chunk_size`` keeps each ghost-padded
chunk inside a fixed RAM/VRAM budget.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.fft import next_fast_len

from spyde.actions.find_vectors.chunk import _find_vectors_chunk
from spyde.actions.find_vectors.detectors import (
    DEFAULT_DOG_SIGMA1,
    DEFAULT_DOG_SIGMA2,
    METHOD_NXCORR,
    _auto_beamstop_from_signal,
    _find_vectors_single_frame,  # noqa: F401  (referenced in docstrings)
    _get_disk_fft,
    _make_disk,
)
from spyde.actions.find_vectors.gpu_runtime import MAX_PEAKS

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Nav chunk sizing (memory-safety critical)
# ─────────────────────────────────────────────────────────────────────────────

def _nav_chunk_size(
    sigma: float, max_ram_mb: float = 200, sig_shape: tuple = (256, 256),
    vram_mb: float = 0.0,
) -> int:
    """
    Compute the nav chunk size so the ghost-padded chunk fits within the memory budget.

    When vram_mb > 0 (GPU path) the budget is VRAM, allowing much larger chunks and
    fewer Dask task dispatches.  For CPU the budget is max_ram_mb.
    """
    depth = int(np.ceil(3 * sigma))
    sig_pixels = sig_shape[0] * sig_shape[1]
    budget = vram_mb if vram_mb > 0 else max_ram_mb
    max_padded = int(np.sqrt(budget * 1e6 / (sig_pixels * 4)))
    # Keep the ghost-padded chunk side <= 255 so the flattened nav grid
    # (side + 2*depth)^2 stays under the CUDA gridDim.z limit of 65535.
    max_side = 255 - 2 * depth
    # Floor the core at 3*depth: a tiny core under a big halo multiplies IO
    # and blur work by ((core+2*depth)/core)^2 — e.g. 512x512 patterns with
    # sigma=1 gave core=4, a 6.2x overhead and 4096 chunks per 256x256 scan.
    # The floor caps overlap overhead at ~2.8x; pushing it lower needs bigger
    # ghost blocks, and per-task RAM (ghost + blur copy, x n_threads) is what
    # drives workers into spill on RAM-limited machines.
    core = max(max_padded - 2 * depth, 3 * depth)
    return max(depth + 1, min(core, max_side))


def _balanced_nav_chunks(dim: int, target: int, depth: int) -> tuple:
    """
    Explicit chunk sizes for one nav dimension: ~target each, but never
    smaller than the ghost depth (map_overlap would silently rechunk and
    desync the declared output chunks), with any short remainder folded
    into the last chunk.
    """
    target = max(int(target), int(depth), 1)
    dim = int(dim)
    if dim <= target:
        return (dim,)
    n = dim // target
    chunks = [target] * n
    rem = dim - n * target
    if rem:
        if rem < max(int(depth), 1):
            chunks[-1] += rem
        else:
            chunks.append(rem)
    return tuple(chunks)


def _split_workers_for_gpu(client) -> tuple:
    """Lane split per SPYDE_FV_GPU — shared implementation in compute_dispatch."""
    from spyde.compute_dispatch import split_workers_for_gpu
    return split_workers_for_gpu(client)


def _compact_padded_chunk(arr: np.ndarray) -> np.ndarray:
    """
    Trim the NaN-padded MAX_PEAKS axis of one chunk result down to the
    longest actual peak list in the chunk.

    The dispatcher must hold every chunk future until the run ends —
    releasing futures mid-run while sibling graphs that share input keys are
    still being submitted races the scheduler (KeyError on forgotten keys).
    Compacting makes holding them cheap: ~30 real peaks instead of 512 slots.
    """
    valid = np.isfinite(arr[..., 0])
    n_max = int(valid.sum(axis=-1).max()) if valid.size else 0
    return np.ascontiguousarray(arr[..., :max(1, n_max), :])




def _compute_chunks_with_live_counts(
    client, result_array, nav_dim, on_chunk_done, stopped_flag=None,
):
    """Single-lane distributed compute with a live per-chunk preview.

    Submits one Future per nav chunk (so chunks land progressively, each with
    its correct global nav slice) and calls ``on_chunk_done(nav_slices, chunk)``
    from the Dask done-callback thread as each completes — mirroring the VI
    progressive path.  Assembles the full padded result and returns it, or
    ``None`` if stopped.  Used when there is exactly one compute lane but a live
    count map is wanted; the dual-lane GPU dispatcher handles its own case.
    """
    import itertools

    nav_chunks = result_array.chunks[:nav_dim]
    axes_ranges = []
    for axis_chunks in nav_chunks:
        positions, start = [], 0
        for size in axis_chunks:
            positions.append((start, size))
            start += size
        axes_ranges.append(positions)
    trailing = (slice(None),) * (result_array.ndim - nav_dim)

    result = np.full(result_array.shape, np.nan, dtype=result_array.dtype)
    futures = []
    chunk_slices = []
    for combo in itertools.product(*axes_ranges):
        nav_sl = tuple(slice(s, s + n) for s, n in combo)
        fut = client.compute(result_array[nav_sl + trailing])
        futures.append(fut)
        chunk_slices.append(nav_sl)

    for fut, nav_sl in zip(futures, chunk_slices):
        def _cb(f, _sl=nav_sl):
            try:
                block = f.result()
            except Exception as e:
                log.debug("live-count chunk %r failed: %s", _sl, e)
                return
            result[_sl + trailing] = block
            try:
                on_chunk_done(_sl, block)
            except Exception as e:
                log.debug("on_chunk_done for %r failed: %s", _sl, e)
        fut.add_done_callback(_cb)

    from spyde.compute_dispatch import poke_scheduler, reliable_sleep
    pending = list(futures)
    n_done_prev = 0
    last_progress = time.time()
    last_poke = time.time()
    while pending:
        if stopped_flag is not None and stopped_flag[0]:
            for fut in pending:
                try:
                    fut.cancel()
                except Exception as e:
                    log.debug("cancelling live-count future failed: %s", e)
            return None
        pending = [f for f in pending if not f.done()]
        n_done = len(futures) - len(pending)
        now = time.time()
        if n_done != n_done_prev:
            n_done_prev = n_done
            last_progress = now
        elif now - last_progress > 5.0 and now - last_poke > 5.0:
            # No-progress watchdog (frozen task delivery — see poke_scheduler).
            last_poke = now
            poke_scheduler(client, "find_vectors-live")
        if pending:
            # NB reliable_sleep, NOT time.sleep — time.sleep froze for 15 s on
            # the throttled Electron-spawned backend, so this loop (and its
            # watchdog above) never ran on schedule.
            reliable_sleep(0.05)
    return result


def _dispatch_chunks_gpu_aware(
    client,
    result_array,
    nav_dim: int,
    gpu_addrs: list,
    cpu_addrs: list,
    stopped_flag=None,
    on_chunk_done=None,
):
    """Greedy dual-lane chunk dispatch — shared implementation in
    compute_dispatch; vectors-specific NaN-slot compaction via postprocess."""
    from spyde.compute_dispatch import dispatch_chunks
    return dispatch_chunks(
        client, result_array, nav_dim, gpu_addrs, cpu_addrs,
        stopped_flag=stopped_flag, postprocess=_compact_padded_chunk,
        fill_value=np.nan, label="find_vectors", on_chunk_done=on_chunk_done,
    )


def _do_compute_vectors(
    signal, params: dict, main_window=None, signal_tree=None,
    shm_name: str = None,
    beamstop_mask: np.ndarray = None,
    on_chunk_done=None,
    stopped_flag=None,
    client=None,
):
    """
    Batch compute via dask.array.map_overlap.

    Nav-space Gaussian blur is applied with ghost zones (depth=ceil(3σ)) so
    chunk boundaries are handled correctly, then _find_vectors_single_frame
    is applied to every pattern.  Results are collected as a flat buffer with
    per-position offsets.

    signal.data may be a numpy array or a lazy dask array.  For numpy inputs
    map_overlap operates on a trivially-chunked array (one chunk = full nav
    block) so no data is copied.  NEVER call .compute() on the full dataset.

    Submission: with a distributed client and a designated GPU worker the
    per-chunk futures go through _dispatch_chunks_gpu_aware (greedy dual-lane
    placement — see its docstring); otherwise the whole graph is one dask
    future.  Either way a passthrough map_blocks stage writes per-chunk peak
    counts into the shared-memory buffer as chunk tasks finish, so a GUI
    polling the buffer sees the count image update live.

    Parameters
    ----------
    shm_name : str | None
        Pre-existing float32 SharedMemory segment; chunk counts are written
        into it from worker tasks while the compute runs, plus a final
        authoritative write when the compute completes.
    on_chunk_done : callable(nav_slice_2d, count_subarray) | None
        Called once after the full compute completes with the full-nav slice.
    stopped_flag : list[bool] | None
        Polled while waiting on the future; setting it cancels the compute
        and returns None.
    client : distributed.Client | None
        Explicit Dask client (the spyde.api / script path). Takes precedence
        over the in-app main_window/signal_tree lookups; with everything None
        the compute falls back to the local threaded scheduler.
    """
    import functools
    import dask.array as da
    from spyde.signals.diffraction_vectors import (
        N_COLS, COL_KX, COL_KY, COL_TIME, COL_INTENSITY, SpyDEDiffractionVectors
    )
    import time

    tic = time.time()

    nav_dim = signal.axes_manager.navigation_dimension
    sig_dim = signal.axes_manager.signal_dimension
    sig_ax = signal.axes_manager.signal_axes
    sig_shape = signal.axes_manager.signal_shape  # (ky, kx) in HS order

    sigma = float(params["sigma"])
    depth_px = int(np.ceil(3 * sigma))

    method = str(params.get("method", METHOD_NXCORR)).lower()
    kernel_r = int(params["kernel_radius"])
    threshold = float(params["threshold"])
    min_dist = int(params["min_distance"])
    subpixel = bool(params.get("subpixel", True))
    dog_sigma1 = float(params.get("dog_sigma1", DEFAULT_DOG_SIGMA1))
    dog_sigma2 = float(params.get("dog_sigma2", DEFAULT_DOG_SIGMA2))
    # Neural-method knobs: registry model id ("" → the registry default), the
    # calibrated local-norm high-pass scale (see find_vectors_neural), and the
    # optional scan-neighbour persistence refine (batch-only — needs neighbours).
    model_id = str(params.get("model_id") or "").strip() or None
    bg_sigma = float(params.get("bg_sigma") or 12.0)
    persistence = bool(params.get("persistence", False))
    log.debug("[do_compute_vectors] START method=%s thr=%s md=%s sigma=%s "
              "nav_dim=%s sig_shape=%s lazy=%s beamstop=%s", method, threshold,
              min_dist, sigma, nav_dim, tuple(sig_shape),
              getattr(signal, "_lazy", "?"),
              beamstop_mask is not None)

    ky_scale = float(sig_ax[1].scale)
    ky_offset = float(sig_ax[1].offset)
    kx_scale = float(sig_ax[0].scale)
    kx_offset = float(sig_ax[0].offset)

    # ── Chunk size: fixed ~100 MB RAM budget per ghost-padded chunk ──────────
    # Uniform small chunks keep every worker's memory bounded, stream a steady
    # supply of tasks (GPU included), and let the live count map fill in
    # progressively.  Sizing chunks to free VRAM collapsed small datasets into
    # a single chunk — a full rechunk-shuffle through one worker, no live
    # updates — and exploded per-worker RAM on large datasets.
    chunk_nav = _nav_chunk_size(sigma, max_ram_mb=100, sig_shape=sig_shape)

    raw = signal.data
    nav_shape_full = raw.shape[:nav_dim]
    nav_2d_shape = nav_shape_full[-2:]
    n_nav_y, n_nav_x = nav_2d_shape

    # Ghost depth cannot exceed the spatial nav extent (map_overlap rejects
    # depth > axis size); tiny grids just lose cross-boundary blur support.
    depth_px = max(0, min(depth_px, min(nav_2d_shape) - 1))

    # ── Auto beam-stop detection ──────────────────────────────────────────────
    # When no mask was supplied and the user asked for one (or it's requested by
    # default), detect a physical beam stop from a SPARSE sample of patterns
    # (never the full dataset — memory rule).  The stop is static, so a few
    # hundred frames give a clean low-intensity mask; dilated to clear the rim.
    if beamstop_mask is None and params.get("beamstop_auto", False):
        try:
            beamstop_dilate = int(params.get("beamstop_dilate", 5))
            beamstop_mask = _auto_beamstop_from_signal(
                signal, nav_dim, dilate=beamstop_dilate
            )
            if beamstop_mask is not None:
                log.debug("[find_vectors] auto beam-stop: %d px masked",
                          int(beamstop_mask.sum()))
        except Exception as e:
            log.debug("[find_vectors] auto beam-stop detection failed: %s", e)
            beamstop_mask = None

    # Disk stats: passed to chunk_fn so workers don't recompute them.
    _disk = _make_disk(kernel_r)
    _n = _disk.shape[0] * _disk.shape[1]
    _t_mean = float(_disk.mean())
    _t_std = float(np.sqrt(np.sum((_disk - _t_mean) ** 2) / _n))
    disk_stats = (_n, _t_mean, _t_std)
    # CPU fallback also needs disk_fft
    pH = next_fast_len(sig_shape[0] + 2 * kernel_r)
    pW = next_fast_len(sig_shape[1] + 2 * kernel_r)
    disk_fft = _get_disk_fft(kernel_r, pH, pW)

    # ── Build chunked dask array ──────────────────────────────────────────────
    # Leading (time) dims are chunked at 1 so a 5D chunk has the same memory
    # footprint as a 4D one.  Every spatial chunk must also be >= depth_px so
    # map_overlap never rechunks internally (that would desync out_chunks).
    # float32 conversion happens per chunk inside chunk_fn — never materialise
    # a converted copy of the full dataset here.
    min_chunk = max(depth_px, 1)
    nav_chunks_tuple = (1,) * (nav_dim - 2) + tuple(
        _balanced_nav_chunks(d, chunk_nav, depth_px) for d in nav_2d_shape
    )
    sig_chunks_tuple = tuple(s for s in raw.shape[nav_dim:])
    if isinstance(raw, np.ndarray):
        da_data = da.from_array(raw, chunks=nav_chunks_tuple + sig_chunks_tuple)
    else:
        # Avoid a rechunk shuffle when the stored chunking is already usable:
        # spatial nav chunks within [depth, ~2x budget], leading dims chunked
        # at 1, signal axes unchunked.  Alignment beats the theoretical chunk
        # budget: rechunking to a misaligned "better" size makes every ghost
        # block gather split pieces from many source chunks — measured 2.3x
        # slower (419 s vs 184 s on the 64 GiB benchmark) from the transfers
        # and memory churn alone.
        keep_limit = min(max(2 * chunk_nav, chunk_nav + 2 * depth_px),
                         255 - 2 * depth_px)
        nav_ok = all(
            min(c) >= min_chunk and max(c) <= keep_limit
            for c in raw.chunks[max(0, nav_dim - 2):nav_dim]
        ) and all(
            max(c) <= 1 for c in raw.chunks[:max(0, nav_dim - 2)]
        )
        sig_ok = all(len(c) == 1 for c in raw.chunks[nav_dim:])
        if nav_ok and sig_ok:
            da_data = raw
        else:
            log.debug("Rechunking to %s and %s", nav_chunks_tuple, sig_chunks_tuple)
            da_data = raw.rechunk(nav_chunks_tuple + sig_chunks_tuple)

    toc  = time.time()
    log.debug("Prepared dask array with chunks %s in %.1f s", da_data.chunks, toc - tic)

    # ── Build the map_overlap graph ───────────────────────────────────────────
    # trim=False: chunk_fn receives the full ghost-padded block and handles
    # trimming itself so the blur can use all ghost rows for correct boundaries.
    # drop_axis removes (ky, kx); new_axis adds (MAX_PEAKS, 3).
    # Output: (nav_y, nav_x, MAX_PEAKS, 3) [4D] or (t, ny, nx, MAX_PEAKS, 3) [5D].
    # Ghost zones only on the two spatial nav dims — the leading (time) axis
    # has blur sigma 0, so overlapping it would be pure overhead (and the
    # chunk fn does not trim it).
    depth_dict = {i: 0 for i in range(nav_dim)}
    depth_dict[nav_dim - 2] = depth_px
    depth_dict[nav_dim - 1] = depth_px
    sig_axes_idx = list(range(nav_dim, nav_dim + sig_dim))
    new_axes_idx = list(range(nav_dim, nav_dim + 2))
    out_chunks = da_data.chunks[:nav_dim] + ((MAX_PEAKS,), (3,))

    chunk_fn = functools.partial(
        _find_vectors_chunk,
        depth_px=depth_px,
        nav_dim=nav_dim,
        sigma=sigma,
        kernel_r=kernel_r,
        threshold=threshold,
        min_dist=min_dist,
        subpixel=subpixel,
        beamstop_mask=beamstop_mask,
        disk_fft=disk_fft,
        disk_stats=disk_stats,
        method=method,
        dog_sigma1=dog_sigma1,
        dog_sigma2=dog_sigma2,
        model_id=model_id,
        bg_sigma=bg_sigma,
        persistence=persistence,
    )

    # Resolve the distributed client up front — needed both to decide on GPU
    # routing and to submit the future below. An explicit `client=` argument
    # wins (the spyde.api / script path); the main_window/signal_tree lookups
    # are the in-app path.
    if client is None and signal_tree is not None:
        client = getattr(signal_tree, "client", None)
    dask_manager = getattr(main_window, "dask_manager", None) if main_window else None
    if client is None and dask_manager is not None:
        client = getattr(dask_manager, "client", None)

    # The cluster is built asynchronously on a background thread, so an early
    # Find Vectors run can arrive before `client` exists.  In the live app
    # (a dask_manager is present) WAIT for it — we run on a worker thread, so
    # blocking here doesn't freeze the UI — rather than silently degrading to
    # the local threaded scheduler below (slow, no GPU, no progressive preview;
    # that path is meant only for the no-cluster unit-test case).
    #
    # Exception: when Dask is disabled (SPYDE_NO_DASK=1, the migrated-test mode)
    # the DaskManager exists but is never started, so its client stays None
    # forever.  Don't wait there — fall straight through to the threaded
    # scheduler, which is exactly the no-cluster path that mode wants.
    import os
    if os.environ.get("SPYDE_NO_DASK") == "1":
        dask_manager = None
    if client is None and dask_manager is not None:
        log.debug("[find_vectors] waiting for Dask client to come up…")
        from spyde.compute_dispatch import reliable_sleep
        deadline = time.time() + 180.0
        while client is None and time.time() < deadline:
            if stopped_flag is not None and stopped_flag[0]:
                return None
            reliable_sleep(0.1)
            client = getattr(dask_manager, "client", None)
        if client is None:
            log.warning("[find_vectors] Dask client never came up — "
                        "falling back to local threaded compute")

    # Only annotate with the GPU resource when at least one worker actually
    # advertises it.  Annotating without such workers leaves every chunk task
    # stuck in no-worker state — the compute never starts.
    gpu_workers = False
    if client is not None:
        try:
            workers_info = client.scheduler_info(n_workers=-1)["workers"]
            gpu_workers = any(
                (w.get("resources") or {}).get("GPU")
                for w in workers_info.values()
            )
        except Exception:
            gpu_workers = False

    import dask
    import contextlib
    tic = time.time()
    annotate_ctx = (
        dask.annotate(resources={"GPU": 1}) if gpu_workers
        else contextlib.nullcontext()
    )
    with annotate_ctx:
        # meta= prevents dask from calling chunk_fn on empty arrays in the
        # client process for type inference.
        peaks_padded = da.map_overlap(
            chunk_fn,
            da_data,
            depth=depth_dict,
            boundary="reflect",
            dtype=np.float32,
            trim=False,
            drop_axis=sig_axes_idx,
            new_axis=new_axes_idx,
            chunks=out_chunks,
            meta=np.empty((0,) * (nav_dim + 2), dtype=np.float32),
        )
    toc = time.time()
    log.debug("Built Dask graph with map_overlap in %.1f s", toc - tic)

    # ── Live count map: count + write each chunk from the CLIENT side ────────
    # Counting/writing happens HERE in the client process — driven by an
    # on_chunk_done(nav_slices, block) callback as each chunk's result lands —
    # NOT inside the dask graph.  An in-graph map_blocks stage reading
    # block_info["array-location"] writes to the WRONG location on the
    # GPU-aware path: that dispatcher slices the array per-chunk, so each
    # sub-array's block location resets to (0, 0) and every chunk clobbers the
    # top-left of the buffer.  The global nav slice the callback receives is
    # always correct, and the write stays on the client (no worker-subprocess
    # shm write → no Windows teardown access-violation; cf. the VI path).
    if shm_name is not None:
        from multiprocessing import shared_memory as _shm_mod

        def _live_count_chunk(nav_slices, block):
            """Count finite peaks per nav position in this chunk's result and
            write them to the live shm buffer at the chunk's GLOBAL location.
            ``nav_slices`` indexes only the nav axes; ``block`` is the padded
            (nav…, n_peaks, cols) chunk result.  Best-effort, never raises."""
            try:
                counts = np.isfinite(block[..., 0]).sum(axis=-1).astype(np.float32)
                shm = _shm_mod.SharedMemory(name=shm_name, create=False)
                try:
                    buf = np.ndarray(tuple(nav_shape_full), dtype=np.float32,
                                     buffer=shm.buf)
                    buf[tuple(nav_slices)] = counts
                finally:
                    shm.close()
            except Exception as e:
                log.debug("live count-map shm write for %r failed: %s",
                          nav_slices, e)
    else:
        _live_count_chunk = None

    # ── Submit as a dask future — only this (background) thread blocks ───────
    if stopped_flag is not None and stopped_flag[0]:
        return None

    log.debug("computing...")
    tic = time.time()
    if client is not None:
        # GPU-aware dual-lane dispatcher when the cluster has a designated
        # GPU worker: dask's locality-driven placement starves the (much
        # faster) GPU worker, so we place per-chunk futures ourselves.
        gpu_addrs, cpu_addrs = _split_workers_for_gpu(client)
        if gpu_addrs and cpu_addrs:
            log.debug(
                f"[find_vectors] dispatcher lanes: "
                f"GPU={len(gpu_addrs)} worker(s), CPU={len(cpu_addrs)} worker(s)"
            )
            result_padded = _dispatch_chunks_gpu_aware(
                client, peaks_padded, nav_dim, gpu_addrs, cpu_addrs,
                stopped_flag=stopped_flag, on_chunk_done=_live_count_chunk,
            )
            if result_padded is None:
                return None
        elif _live_count_chunk is not None:
            # Single-lane distributed path WITH a live preview: submit one
            # future per nav chunk so chunks land (and paint) progressively,
            # each with its correct global slice — same mechanism as the VI
            # progressive path (compute_with_live_buffer).
            result_padded = _compute_chunks_with_live_counts(
                client, peaks_padded, nav_dim, _live_count_chunk,
                stopped_flag=stopped_flag,
            )
            if result_padded is None:
                return None
        else:
            from spyde.compute_dispatch import poke_scheduler, reliable_sleep
            future = client.compute(peaks_padded)
            t_wait = time.time()
            last_poke = time.time()
            while not future.done():
                if stopped_flag is not None and stopped_flag[0]:
                    try:
                        future.cancel()
                    except Exception as e:
                        log.debug("cancelling find-vectors compute future failed: %s", e)
                    return None
                # No-progress watchdog (frozen task delivery — poke_scheduler).
                now = time.time()
                if now - t_wait > 5.0 and now - last_poke > 5.0:
                    last_poke = now
                    poke_scheduler(client, "find_vectors-monolithic")
                reliable_sleep(0.1)
            result_padded = future.result()
    else:
        # No distributed client (e.g. unit tests): local threaded scheduler,
        # pinned explicitly — a bare .compute() silently runs on any ambient
        # distributed Client (dask's global default when one exists).
        # This computes the small padded-peaks output, never the raw dataset.
        result_padded = peaks_padded.compute(scheduler="threads")
    toc = time.time()
    log.debug("Computed vectors in %.1f s", toc - tic)

    if stopped_flag is not None and stopped_flag[0]:
        return None

    # ── Unpack padded result into flat buffer (vectorised) ───────────────────
    # result_padded shape: (nav_y, nav_x, MAX_PEAKS, 3)  [4D]
    #                   or (t, nav_y, nav_x, MAX_PEAKS, 3) [5D]
    # Valid peaks have finite ky (col 0); NaN-padded slots are ignored.
    # np.nonzero returns C-order indices, so the flat buffer comes out already
    # sorted outermost-nav-dim first as SpyDEDiffractionVectors requires.
    valid = np.isfinite(result_padded[..., 0])
    nz = np.nonzero(valid)
    peaks_flat = result_padded[valid]              # (N_total, 3)
    N_total = peaks_flat.shape[0]

    flat_buffer = np.zeros((N_total, N_COLS), dtype=np.float32)
    if nav_dim == 2:
        iy_idx, ix_idx = nz[0], nz[1]
        flat_buffer[:, COL_TIME] = -1.0
    else:
        t_idx, iy_idx, ix_idx = nz[0], nz[1], nz[2]
        flat_buffer[:, COL_TIME] = t_idx.astype(np.float32)
    flat_buffer[:, 0] = ix_idx
    flat_buffer[:, 1] = iy_idx
    flat_buffer[:, COL_KX] = peaks_flat[:, 1] * kx_scale + kx_offset
    flat_buffer[:, COL_KY] = peaks_flat[:, 0] * ky_scale + ky_offset
    flat_buffer[:, COL_INTENSITY] = peaks_flat[:, 2]

    # ── Final shm write / completion callback ─────────────────────────────────
    if shm_name is not None or on_chunk_done is not None:
        # counts_full has shape nav_shape_full: (n_y, n_x) for 4D or
        # (n_t, n_y, n_x) for 5D — matches the buffer allocated by the action.
        counts_full = valid.sum(axis=-1)
        count_map = (counts_full.sum(axis=0) if nav_dim == 3
                     else counts_full).astype(np.int32)
        if shm_name is not None:
            from multiprocessing import shared_memory as _shm_mod
            try:
                shm_handle = _shm_mod.SharedMemory(name=shm_name, create=False)
                shm_buf = np.ndarray(nav_shape_full, dtype=np.float32, buffer=shm_handle.buf)
                shm_buf[:] = counts_full.astype(np.float32)
                shm_handle.close()
            except Exception as e:
                log.debug("final count-map shm write to %s failed: %s", shm_name, e)
        if on_chunk_done is not None:
            on_chunk_done((slice(None), slice(None)), count_map)

    log.debug("[do_compute_vectors] DONE: %d vectors over %s nav positions "
              "(%.2f per pattern)", N_total, int(np.prod(nav_shape_full)),
              N_total / max(1, int(np.prod(nav_shape_full))))

    # Snapshot the scan-step (navigation) calibration as lightweight axis records
    # so a saved/loaded vectors file reconstructs a calibrated scan grid without
    # the source dataset (the live Find-Vectors path still copies these from the
    # source onto the result tree; this is for the standalone round-trip).
    from spyde.signals.diffraction_vectors import _AxisLite
    nav_ax = [
        _AxisLite(scale=float(ax.scale), offset=float(ax.offset),
                  size=int(ax.size), units=str(getattr(ax, "units", "") or ""),
                  name=str(getattr(ax, "name", "") or ""))
        for ax in signal.axes_manager.navigation_axes
    ]
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat_buffer,
        full_nav_shape=nav_shape_full,
        sig_shape=sig_shape,
        sig_axes=sig_ax,
        kernel_radius_px=float(kernel_r),
        kernel_radius_data=float(kernel_r) * sig_ax[0].scale,
        params=dict(params),
        nav_axes=nav_ax,
    )


def _copy_nav_axes_to(source_signal, target_signal):
    """Copy navigation axis calibration from source to target."""
    src_axes = source_signal.axes_manager.navigation_axes
    tgt_axes = target_signal.axes_manager.navigation_axes
    for i, ax in enumerate(src_axes):
        if i < len(tgt_axes):
            tgt_axes[i].scale = ax.scale
            tgt_axes[i].offset = ax.offset
            tgt_axes[i].units = ax.units
            tgt_axes[i].name = ax.name

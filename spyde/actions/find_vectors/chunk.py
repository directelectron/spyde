"""
chunk.py — ghost-block chunk pipeline for find_vectors.

The per-chunk pipeline that map_overlap drives: nav-space Gaussian blur over a
ghost-padded block, ghost trim, then the per-frame detector.  Holds the GPU
chunk implementation (single H2D, device-resident blur + NXCORR + peak
kernels) with its warmup/serialisation wrapper, plus the DoG and CPU-fallback
chunk paths and the NavBlur ghost helper they share.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from spyde.actions.find_vectors.detectors import (
    DEFAULT_DOG_SIGMA1,
    DEFAULT_DOG_SIGMA2,
    METHOD_DOG,
    METHOD_NXCORR,
    _find_vectors_single_frame,
    _find_vectors_single_frame_dog,
    _make_disk,
    _with_raw_intensity,
)
from spyde.actions.find_vectors.gpu_runtime import (
    MAX_PEAKS,
    _GPU_MAX_WARM_FAILURES,
    _GPU_NATIVE_DTYPES,
    _cupy_available,
    _cupy_state,
    _get_thread_stream,
    _gpu_cache_lock,
    _gpu_disk_cache,
    _gpu_exec_lock,
    _gpu_pool_get,
    _gpu_pool_put,
    _gpu_serial_mode,
    _gpu_slots,
    _gpu_task_allowed,
    _gpu_warm_failures,
    _gpu_warmed,
    _gpu_warmup_lock,
    _interprocess_warmup_lock,
    _pinned_pool_get,
    _pinned_pool_put,
    _reset_gpu_state,
    _stream_ptr,
)
from spyde.actions.find_vectors.kernels import (
    _GPU_KERNELS_AVAILABLE,
    _convert_f32_kernel,
    _frame_reduce_kernel,
    _gaussian_blur_1d_kernel,
    _local_max_kernel,
    _nxcorr_fft_cupy,
    _nxcorr_reflect_kernel,
    _nxcorr_tiled_kernel,
    _subpixel_parabola_kernel,
    _trim_copy_kernel,
)

log = logging.getLogger(__name__)


def _nav_blur_trim(ghost_block, depth_px, nav_dim, sigma):
    """Nav-space Gaussian blur (sigma over the 2 spatial nav dims, 0 elsewhere)
    of a ghost-padded block, then trim the ghost zones.  Shared by the NXCORR CPU
    fallback and the DoG path."""
    from scipy.ndimage import gaussian_filter as _gf
    sigma_tuple = tuple([0.0] * (nav_dim - 2) + [sigma, sigma, 0.0, 0.0])
    blurred = _gf(np.asarray(ghost_block, dtype=np.float32), sigma=sigma_tuple)
    trim = [slice(None)] * ghost_block.ndim
    for d in (nav_dim - 2, nav_dim - 1):
        s = blurred.shape[d]
        lo = depth_px if depth_px < s else 0
        hi = s - depth_px if depth_px < s else s
        trim[d] = slice(lo, hi)
    return blurred[tuple(trim)]


def _dog_block(b4d, sigma1, sigma2, threshold, min_dist, subpixel, beamstop_mask):
    """Run the DoG detector on a (ny, nx, KY, KX) block → NaN-padded
    (ny, nx, MAX_PEAKS, 3).  Batches on the torch GPU when available; otherwise
    the numpy per-frame core."""
    out = np.full((b4d.shape[0], b4d.shape[1], MAX_PEAKS, 3), np.nan, dtype=np.float32)
    flat = b4d.reshape(-1, b4d.shape[2], b4d.shape[3])
    peaks_list = None
    try:
        from spyde.actions.find_vectors_torch import (
            torch_gpu_device, find_vectors_dog_torch_batch)
        if torch_gpu_device() is not None:
            peaks_list = find_vectors_dog_torch_batch(
                flat, sigma1, sigma2, threshold, min_dist,
                subpixel=subpixel, beamstop_mask=beamstop_mask)
            # GPU returns the SNR in col 2; replace with raw frame intensity
            # (disk-mean over ~σ₂ footprint — robust, avoids per-frame banding).
            _r = max(1.0, float(np.ceil(sigma2)))
            peaks_list = [_with_raw_intensity(flat[i], p, radius=_r)
                          for i, p in enumerate(peaks_list)]
    except Exception as _e:
        log.warning("[find_vectors] torch DoG GPU path failed (%s); CPU per-frame", _e)
        peaks_list = None
    if peaks_list is None:
        peaks_list = [
            _find_vectors_single_frame_dog(
                frame, sigma1, sigma2, threshold, min_dist,
                subpixel=subpixel, beamstop_mask=beamstop_mask)[2]
            for frame in flat
        ]
    for i, peaks in enumerate(peaks_list):
        iy, ix = divmod(i, b4d.shape[1])
        n = min(len(peaks), MAX_PEAKS)
        if n > 0:
            out[iy, ix, :n, :] = peaks[:n]
    return out


def _find_vectors_chunk_dog(
    ghost_block, depth_px, nav_dim, sigma,
    sigma1, sigma2, threshold, min_dist, subpixel, beamstop_mask,
):
    """DoG variant of _find_vectors_chunk: nav-blur + trim, then per-frame DoG
    band-pass (GPU-batched when torch CUDA is present).  Same output structure
    as the NXCORR chunk fn."""
    t_start = time.perf_counter()
    blurred = _nav_blur_trim(ghost_block, depth_px, nav_dim, sigma)
    nav_shape = blurred.shape[:nav_dim]
    ny, nx = nav_shape[-2:]
    if nav_dim == 2:
        result = _dog_block(blurred, sigma1, sigma2, threshold, min_dist,
                            subpixel, beamstop_mask)
        core_shape = result.shape[:2]
    else:
        n_lead = nav_shape[0]
        out = np.full((n_lead, ny, nx, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        for t in range(n_lead):
            out[t] = _dog_block(blurred[t], sigma1, sigma2, threshold, min_dist,
                                subpixel, beamstop_mask)
        result = out
        core_shape = (n_lead, ny, nx)
    log.debug("[find_vectors] DoG chunk core=%s total=%.0fms",
              tuple(int(s) for s in core_shape),
              (time.perf_counter() - t_start) * 1e3)
    return result


def _find_vectors_chunk(
    ghost_block: np.ndarray,
    depth_px: int,
    nav_dim: int,
    sigma: float,
    kernel_r: int,
    threshold: float,
    min_dist: int,
    subpixel: bool,
    beamstop_mask,
    disk_fft,
    disk_stats,
    method: str = METHOD_NXCORR,
    dog_sigma1: float = DEFAULT_DOG_SIGMA1,
    dog_sigma2: float = DEFAULT_DOG_SIGMA2,
) -> np.ndarray:
    """
    Full pipeline for one ghost-padded nav chunk.

    Passed to dask.array.map_overlap with trim=False.  Receives the ghost-padded
    block from map_overlap, does everything on GPU when available:

        CPU → [H2D] → GPU blur (nav-space Gaussian, real-space separable)
                     → NXCORR kernel (xcorr + window stats + normalise)
                     → local-max kernel (NMS)
                     → subpixel CoM kernel
                     → [D2H sparse peaks]
        → pack into (nav_y, nav_x, MAX_PEAKS, 3) NaN-padded float32

    Falls back to scipy + CPU per-frame loop if CUDA is unavailable or this
    worker is not the designated GPU worker (see _gpu_task_allowed /
    SPYDE_FV_GPU).  Both paths print one timing line per chunk.

    The ghost zone covers depth_px = ceil(3σ) nav rows/cols on each edge.
    The blur uses these ghost rows so chunk-boundary blur values are correct;
    we trim them back before NXCORR so no ghost pixels appear in the output.

    For 5D (nav_dim=3) the leading dimension is time; we process each t-slice
    as an independent 4D block and stack into (t, ny, nx, MAX_PEAKS, 3).

    Returns
    -------
    float32 ndarray (nav_y, nav_x, MAX_PEAKS, 3)          [4D]
                 or (t, nav_y, nav_x, MAX_PEAKS, 3)        [5D]
    """
    # Zero-size blocks (dask meta inference calls the chunk fn on empty
    # arrays in the CLIENT process) — return an empty result of the right
    # structure; an empty grid is an invalid CUDA launch.
    if ghost_block.size == 0:
        if nav_dim == 2:
            return np.empty((0, 0, MAX_PEAKS, 3), dtype=np.float32)
        return np.empty((0, 0, 0, MAX_PEAKS, 3), dtype=np.float32)

    # ── DoG band-pass detector ────────────────────────────────────────────────
    # The numba-CUDA NXCORR kernels below are disk-matched-filter specific, so
    # DoG takes its own route: nav-blur + trim (CPU), then the per-frame
    # band-pass batched on the torch GPU (Pascal CUDA) when available, numpy
    # otherwise.  Both share the same nav-blur as NXCORR.
    if str(method).lower() == METHOD_DOG:
        return _find_vectors_chunk_dog(
            ghost_block, depth_px, nav_dim, sigma,
            dog_sigma1, dog_sigma2, threshold, min_dist, subpixel,
            beamstop_mask,
        )

    # ── Try GPU path ──────────────────────────────────────────────────────────
    if _GPU_KERNELS_AVAILABLE and _gpu_task_allowed():
        try:
            from numba import cuda as _cuda
            cuda_ok = _cuda.is_available()
        except Exception:
            cuda_ok = False
        if cuda_ok:
            for _attempt in range(2):
                try:
                    return _find_vectors_chunk_gpu(
                        ghost_block, depth_px, nav_dim, sigma,
                        kernel_r, threshold, min_dist, subpixel,
                        beamstop_mask, disk_stats,
                    )
                except Exception as exc:
                    try:
                        from distributed import get_worker
                        _wname = f"worker {get_worker().name}"
                    except Exception:
                        _wname = "main process"
                    if _attempt == 0:
                        # The first CUDA touch in a fresh dask worker thread
                        # can fail transiently (driver init / cache-load
                        # race) — settles immediately after.  Retry once.
                        log.warning("[find_vectors] GPU attempt failed on %s (%r) — retrying", _wname, exc)
                        time.sleep(0.5)
                    else:
                        # A silent fallback hides real throughput problems.
                        import traceback as _tb
                        log.warning("[find_vectors] GPU path failed on %s (%r) — falling back to CPU", _wname, exc)
                        _tb.print_exc()

    # ── CPU fallback ──────────────────────────────────────────────────────────
    from scipy.ndimage import gaussian_filter as _gf

    t_start = time.perf_counter()
    sigma_tuple = tuple([0.0] * (nav_dim - 2) + [sigma, sigma, 0.0, 0.0])
    # asarray: no copy when the block is already float32
    blurred = _gf(np.asarray(ghost_block, dtype=np.float32), sigma=sigma_tuple)
    blur_ms = (time.perf_counter() - t_start) * 1e3

    # Trim ghost zones — only the two spatial nav dims are ghosted
    # (map_overlap depth is 0 on any leading time axis).
    trim = [slice(None)] * ghost_block.ndim
    for d in (nav_dim - 2, nav_dim - 1):
        s = blurred.shape[d]
        lo = depth_px if depth_px < s else 0
        hi = s - depth_px if depth_px < s else s
        trim[d] = slice(lo, hi)
    blurred = blurred[tuple(trim)]

    nav_shape = blurred.shape[:nav_dim]
    ny, nx = nav_shape[-2:]

    def _cpu_block(b4d):
        out = np.full((b4d.shape[0], b4d.shape[1], MAX_PEAKS, 3), np.nan, dtype=np.float32)
        flat = b4d.reshape(-1, b4d.shape[2], b4d.shape[3])
        # Fast path: batch the whole block's NXCORR on the torch GPU (Apple-MPS on
        # a MacBook, CUDA otherwise). Reached when the numba.cuda kernels above
        # aren't available — exactly the Mac case the user wants accelerated. The
        # surface + peak set match the numpy reference to float precision (the GPU
        # only accelerates the correlation; peak detection stays identical).
        peaks_list = None
        try:
            from spyde.actions.find_vectors_torch import (
                torch_gpu_device, find_vectors_torch_batch)
            if torch_gpu_device() is not None:
                peaks_list = find_vectors_torch_batch(
                    flat, kernel_r, threshold, min_dist,
                    subpixel=subpixel, beamstop_mask=beamstop_mask)
                # torch returns NXCORR positions + correlation score; replace the
                # value column with the raw frame intensity at each peak (disk-mean
                # over the kernel footprint — robust, avoids per-frame banding).
                peaks_list = [_with_raw_intensity(flat[i], p, radius=kernel_r)
                              for i, p in enumerate(peaks_list)]
        except Exception as _e:
            log.warning("[find_vectors] torch GPU path failed (%s); CPU per-frame", _e)
            peaks_list = None
        if peaks_list is None:
            peaks_list = [
                _find_vectors_single_frame(
                    frame, kernel_r, threshold, min_dist,
                    subpixel=subpixel, beamstop_mask=beamstop_mask,
                    _disk_fft=disk_fft, _disk_stats=disk_stats)[2]
                for frame in flat
            ]
        for i, peaks in enumerate(peaks_list):
            iy, ix = divmod(i, b4d.shape[1])
            n = min(len(peaks), MAX_PEAKS)
            if n > 0:
                out[iy, ix, :n, :] = peaks[:n]
        return out

    t_find = time.perf_counter()
    if nav_dim == 2:
        result = _cpu_block(blurred)
        core_shape = result.shape[:2]
    else:
        n_lead = nav_shape[0]
        out = np.full((n_lead, ny, nx, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        for t in range(n_lead):
            out[t] = _cpu_block(blurred[t])
        result = out
        core_shape = (n_lead, ny, nx)
    find_ms = (time.perf_counter() - t_find) * 1e3
    total_ms = (time.perf_counter() - t_start) * 1e3
    log.debug(
        f"[find_vectors] CPU chunk core={tuple(int(s) for s in core_shape)} "
        f"sig=({ghost_block.shape[-2]},{ghost_block.shape[-1]}) "
        f"blur={blur_ms:.0f}ms find={find_ms:.0f}ms total={total_ms:.0f}ms"
    )
    return result


def _gpu_warmup_probe():
    """
    Compile and exercise every CUDA kernel on a tiny synthetic chunk, for
    both the float32 and the uint16 (device-convert) paths.

    Run under the warmup locks so (a) a cursed first context costs a few
    milliseconds instead of a real chunk, and (b) no kernel specialisation
    is ever first-compiled outside the locks by concurrent task threads.
    """
    kernel_r = 2
    d = _make_disk(kernel_r)
    n = d.shape[0] * d.shape[1]
    t_mean = float(d.mean())
    t_std = float(np.sqrt(np.sum((d - t_mean) ** 2) / n))
    base = np.zeros((6, 6, 16, 16), dtype=np.float32)
    base[:, :, 8, 8] = 100.0
    for block in (base, (base * 40).astype(np.uint16)):
        _find_vectors_chunk_gpu_impl(
            block, 1, 2, 1.0, kernel_r, 0.5, 2, True, None, (n, t_mean, t_std),
        )


def _find_vectors_chunk_gpu(*args, **kwargs) -> np.ndarray:
    """Thin wrapper around _find_vectors_chunk_gpu_impl that serialises the
    first use per process (kernel compile + context init are not safe to
    race from multiple threads) and recovers from transient init failures."""
    if _gpu_warmed[0]:
        if _gpu_serial_mode():
            with _gpu_exec_lock:
                return _find_vectors_chunk_gpu_impl(*args, **kwargs)
        return _find_vectors_chunk_gpu_impl(*args, **kwargs)

    with _gpu_warmup_lock:
        if not _gpu_warmed[0]:
            if _gpu_warm_failures[0] >= _GPU_MAX_WARM_FAILURES:
                raise RuntimeError(
                    "GPU disabled in this process after repeated warmup failures"
                )
            from numba import cuda as _cuda
            # The first CUDA context in a fresh dask worker can come up
            # broken (kernel launches fail with INVALID_VALUE) and only
            # works after a full cuda.close() + re-init.  Retry the cheap
            # probe a few times with resets in between.
            last_exc = None
            for attempt in range(3):
                try:
                    with _interprocess_warmup_lock():
                        try:
                            _cuda.current_context()
                        except Exception:
                            _reset_gpu_state()
                            time.sleep(0.5)
                            _cuda.current_context()
                        with _gpu_exec_lock:
                            _gpu_warmup_probe()
                    if attempt:
                        log.debug("[find_vectors] GPU warmup succeeded on attempt %d", attempt + 1)
                    _gpu_warmed[0] = True
                    break
                except Exception as exc:
                    last_exc = exc
                    _reset_gpu_state()
                    time.sleep(1.0)
            else:
                _gpu_warm_failures[0] += 1
                raise last_exc

    if _gpu_serial_mode():
        with _gpu_exec_lock:
            return _find_vectors_chunk_gpu_impl(*args, **kwargs)
    return _find_vectors_chunk_gpu_impl(*args, **kwargs)


def _find_vectors_chunk_gpu_impl(
    ghost_block: np.ndarray,
    depth_px: int,
    nav_dim: int,
    sigma: float,
    kernel_r: int,
    threshold: float,
    min_dist: int,
    subpixel: bool,
    beamstop_mask,
    disk_stats,
) -> np.ndarray:
    """
    Device-resident GPU implementation of _find_vectors_chunk.

    Single H2D transfer of the ghost-padded block, then entirely on-device:
      1. Separable Gaussian blur in nav-space (two 1D kernel passes)
      2. Trim-compaction of the core into contiguous (N, KY, KX) frames
      3. Beamstop fill (per-frame unmasked-mean reduction + fill kernel)
      4. Per-frame std reduction for the NXCORR denominator floor
      5. NXCORR with reflected indexing (no host reflect-pad round trip)
         + local-max + subpixel kernels
      6. D2H only the tiny stats vectors and the sparse padded peak result

    For 5D, processes each t-slice sequentially on GPU (same device context).
    Prints one timing line per chunk so GPU vs CPU throughput can be compared.
    """
    import os
    from numba import cuda as _cuda

    t_start = time.perf_counter()
    timings = {"stage": 0.0, "h2d": 0.0, "blur": 0.0, "stats": 0.0,
               "nxcorr": 0.0, "peaks": 0.0, "d2h": 0.0, "pack": 0.0}

    # Per-stage device syncs make the stage timings accurate but act as
    # barriers — they prevent the GPU worker's threads from overlapping their
    # chunks.  Off by default; SPYDE_FV_TIMING=1 enables.
    stage_timing = os.environ.get("SPYDE_FV_TIMING", "") not in ("", "0", "off")

    # Per-thread stream: chunk tasks on different worker threads overlap
    # their H2D/D2H and kernels instead of serialising on the legacy default
    # stream.  CuPy work joins the same stream via ExternalStream.
    stream = _get_thread_stream()
    use_fft = _cupy_available()
    nx_path = ["fft" if use_fft else "numba"]

    def _sync_if_timing():
        if stage_timing:
            stream.synchronize()

    # Native integer data is uploaded as-is and cast to float32 on device:
    # half the H2D bytes for 16-bit detectors and no host astype pass.
    # float32 passes straight through (ascontiguousarray is a no-op copy-wise);
    # anything else (float64, big-endian) converts on the host.
    if ghost_block.dtype in _GPU_NATIVE_DTYPES:
        block_host = np.ascontiguousarray(ghost_block)
        convert_on_device = True
    else:
        block_host = np.ascontiguousarray(ghost_block, dtype=np.float32)
        convert_on_device = False

    nav_shape_ghost = block_host.shape[:nav_dim]
    KY, KX = block_host.shape[-2], block_host.shape[-1]
    HW = KY * KX

    # ── Pre-compute 1D Gaussian kernel (CPU, tiny) ────────────────────────────
    if sigma > 0:
        radius = int(np.ceil(3 * sigma))
        xs = np.arange(-radius, radius + 1, dtype=np.float32)
        kern_cpu = np.exp(-0.5 * (xs / sigma) ** 2).astype(np.float32)
        kern_cpu /= kern_cpu.sum()
    else:
        radius = 0
        kern_cpu = np.ones(1, dtype=np.float32)
    kern_d = _cuda.to_device(kern_cpu, stream=stream)

    # ── Upload disk kernel once per kernel_r (lock: shared across threads) ───
    with _gpu_cache_lock:
        if kernel_r not in _gpu_disk_cache:
            _gpu_disk_cache[kernel_r] = _cuda.to_device(_make_disk(kernel_r))
        disk_d = _gpu_disk_cache[kernel_r]

    n_disk, t_mean, t_std = disk_stats
    n_disk = np.int32(n_disk)
    t_mean = np.float32(t_mean)
    t_std  = np.float32(t_std)
    kr     = np.int32(kernel_r)
    kr_win = np.int32(kernel_r + 1)
    thr    = np.float32(threshold)
    min_d  = np.int32(min_dist)
    iH, iW = np.int32(KY), np.int32(KX)

    # Beamstop mask: uploaded once per chunk.  A 1x1 dummy keeps the reduction
    # kernel signature uniform when no mask is active (use_mask=0 short-circuits
    # before indexing it).
    have_mask = beamstop_mask is not None and beamstop_mask.any()
    if have_mask:
        mask_d = _cuda.to_device(beamstop_mask.astype(np.uint8))
        n_unmasked = max(1, HW - int(beamstop_mask.sum()))
    else:
        mask_d = _cuda.to_device(np.zeros((1, 1), dtype=np.uint8))
        n_unmasked = HW

    import warnings as _w

    def _device_section(block4d_cpu):
        """Device-resident pipeline for one (ny_ghost, nx_ghost, KY, KX) block.
        Runs under the GPU slot semaphore; returns host-side results only."""
        NY_g, NX_g = block4d_cpu.shape[0], block4d_cpu.shape[1]

        bx, by = 16, 16
        # x/y span signal pixels, z spans nav positions (z <= 65535)
        grid_sig_nav = (
            int(np.ceil(KX / bx)),
            int(np.ceil(KY / by)),
            NY_g * NX_g,
        )

        # H2D — the only full-block transfer in the whole pipeline.
        # Integer data goes up in its native width and is cast on device.
        # All device buffers come from the pool (no cudaMalloc per chunk).
        raw_d = None
        tmp_d = None
        t0 = time.perf_counter()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            if convert_on_device:
                raw_d = _gpu_pool_get(block4d_cpu.shape, block4d_cpu.dtype)
                raw_d.copy_to_device(block4d_cpu, stream=stream)
                src_d = _gpu_pool_get(block4d_cpu.shape, np.float32)
                _convert_f32_kernel[grid_sig_nav, (bx, by, 1), stream](raw_d, src_d)
            else:
                src_d = _gpu_pool_get(block4d_cpu.shape, np.float32)
                src_d.copy_to_device(block4d_cpu, stream=stream)
        _sync_if_timing()
        timings["h2d"] += time.perf_counter() - t0

        # ── Gaussian blur: two separable 1D passes ────────────────────────────
        t0 = time.perf_counter()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            if sigma > 0:
                tmp_d = _gpu_pool_get(block4d_cpu.shape, np.float32)
                grid_blur = grid_sig_nav
                block_blur = (bx, by, 1)
                _gaussian_blur_1d_kernel[grid_blur, block_blur, stream](
                    src_d, tmp_d, kern_d, np.int32(radius), np.int32(0)
                )  # blur along nav_y into tmp
                _gaussian_blur_1d_kernel[grid_blur, block_blur, stream](
                    tmp_d, src_d, kern_d, np.int32(radius), np.int32(1)
                )  # blur along nav_x back into src_d
        _sync_if_timing()
        timings["blur"] += time.perf_counter() - t0

        # ── Trim bounds (compacted on device by _trim_copy_kernel) ───────────
        lo = depth_px
        hi_y = NY_g - depth_px
        hi_x = NX_g - depth_px
        if lo >= hi_y or lo >= hi_x:
            lo, hi_y, hi_x = 0, NY_g, NX_g
        NY = hi_y - lo
        NX = hi_x - lo
        N = NY * NX

        grid_px = (int(np.ceil(KX / 16)), int(np.ceil(KY / 16)), N)
        blk_px = (16, 16, 1)

        t0 = time.perf_counter()
        sums_d = None
        sumsq_d = None
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            # Compact the trimmed core into contiguous (N, KY, KX) frames —
            # device-to-device; the strided ghost view never touches the host.
            frames_d = _gpu_pool_get((N, KY, KX), np.float32)
            _trim_copy_kernel[grid_px, blk_px, stream](
                src_d, frames_d, np.int32(lo), np.int32(lo), np.int32(NX), iH, iW
            )

            # Beam stop = PEAK REJECTION, not an image edit. We intentionally do
            # NOT fill the masked pixels here (a fill creates a sharp boundary
            # step the correlator scores as rim spots). The frame is left
            # UNMODIFIED; peaks inside the mask are dropped after detection (see
            # the `~beamstop_mask` exclusion below), so the stop contributes no
            # detections without an artificial edge.
        _sync_if_timing()
        timings["stats"] += time.perf_counter() - t0

        raw_corr_d = None
        peak_mask_d = _gpu_pool_get((N, KY, KX), np.uint8)

        # ── NXCORR: CuPy/cuFFT when available, numba kernels otherwise ───────
        t0 = time.perf_counter()
        raw_corr_obj = None
        if use_fft:
            try:
                raw_corr_obj = _nxcorr_fft_cupy(
                    frames_d, int(kernel_r), disk_stats, stream
                )
                nx_path[0] = "fft"
            except Exception as exc:
                # Don't retry CuPy on every chunk of this process
                _cupy_state["ok"] = False
                nx_path[0] = "numba"
                log.warning("[find_vectors] CuPy NXCORR failed (%r) — "
                            "falling back to numba kernels", exc)

        if raw_corr_obj is None:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                # Per-frame global std for the denominator floor
                if sums_d is None:
                    sums_d = _gpu_pool_get((N,), np.float32)
                    sumsq_d = _gpu_pool_get((N,), np.float32)
                _frame_reduce_kernel[N, 256, stream](
                    frames_d, mask_d, np.int32(0), sums_d, sumsq_d,
                    np.int32(HW), iW,
                )
                sums = sums_d.copy_to_host(stream=stream)
                sumsqs = sumsq_d.copy_to_host(stream=stream)
                stream.synchronize()
                mean = sums / HW
                var = np.maximum(sumsqs / HW - mean * mean, 0.0)
                stds = np.sqrt(var).astype(np.float32)
                stds[stds == 0.0] = 1.0
                global_stds_d = _cuda.to_device(stds, stream=stream)

                raw_corr_d = _gpu_pool_get((N, KY, KX), np.float32)
                # Tiled shared-memory kernel when the halo fits the 64x64
                # tile (kr_win <= 24); naive reflected-index kernel beyond.
                if int(kr_win) <= 24:
                    _nxcorr_tiled_kernel[grid_px, blk_px, stream](
                        frames_d, disk_d, raw_corr_d, global_stds_d,
                        n_disk, t_mean, t_std, kr, kr_win, iH, iW,
                    )
                else:
                    _nxcorr_reflect_kernel[grid_px, blk_px, stream](
                        frames_d, disk_d, raw_corr_d, global_stds_d,
                        n_disk, t_mean, t_std, kr, kr_win, iH, iW,
                    )
            raw_corr_obj = raw_corr_d
        _sync_if_timing()
        timings["nxcorr"] += time.perf_counter() - t0

        peaks_out_d = None
        n_peaks_d = None
        t0 = time.perf_counter()
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            # raw_corr_obj is a numba device array or a CuPy array — numba
            # kernels accept both via the CUDA array interface, and both were
            # produced on `stream`, so ordering is preserved.
            _local_max_kernel[grid_px, blk_px, stream](
                raw_corr_obj, peak_mask_d, thr, min_d, iH, iW,
            )
            if subpixel:
                peaks_out_d = _gpu_pool_get((N, MAX_PEAKS, 3), np.float32)
                n_peaks_d = _gpu_pool_get((N,), np.int32)
                n_peaks_d.copy_to_device(np.zeros(N, dtype=np.int32),
                                         stream=stream)
                # Parabolic vertex (matches the CPU/torch subpixel) — more
                # accurate than the old centre-of-mass kernel, which was the
                # least-accurate of the three subpixel paths.
                _subpixel_parabola_kernel[grid_px, blk_px, stream](
                    raw_corr_obj, peak_mask_d, peaks_out_d, n_peaks_d,
                    iH, iW,
                )
        _sync_if_timing()
        timings["peaks"] += time.perf_counter() - t0

        # D2H — only sparse results
        t0 = time.perf_counter()
        if subpixel:
            peaks_out = peaks_out_d.copy_to_host(stream=stream)
            n_peaks   = n_peaks_d.copy_to_host(stream=stream)
        else:
            peak_mask = peak_mask_d.copy_to_host(stream=stream)
            if raw_corr_d is not None:
                raw_corr = raw_corr_d.copy_to_host(stream=stream)
            else:
                import cupy as _cp
                with _cp.cuda.ExternalStream(_stream_ptr(stream)):
                    raw_corr = _cp.asnumpy(raw_corr_obj)
        # One sync per chunk: host reads below are safe and the pooled
        # buffers are idle, so cross-thread reuse on other streams is safe.
        stream.synchronize()
        timings["d2h"] += time.perf_counter() - t0

        _gpu_pool_put(raw_d, src_d, tmp_d, frames_d, sums_d, sumsq_d,
                      raw_corr_d, peak_mask_d, peaks_out_d, n_peaks_d)

        if subpixel:
            return NY, NX, N, peaks_out, n_peaks, None, None
        return NY, NX, N, None, None, peak_mask, raw_corr

    def _process_4d(block4d_cpu):
        """One block: stage into pinned memory, device section under the GPU
        slot semaphore, then the CPU pack outside it — staging and packing
        overlap another thread's GPU ownership."""
        # Stage into a pinned buffer BEFORE taking a GPU slot: the memcpy is
        # CPU work, and a pinned source makes the H2D truly asynchronous.
        t0 = time.perf_counter()
        pinned = _pinned_pool_get(block4d_cpu.shape, block4d_cpu.dtype)
        if pinned is not None:
            np.copyto(pinned, block4d_cpu)
            upload_src = pinned
        else:
            upload_src = block4d_cpu
        timings["stage"] += time.perf_counter() - t0

        try:
            with _gpu_slots():
                (NY, NX, N, peaks_out, n_peaks,
                 peak_mask, raw_corr) = _device_section(upload_src)
        finally:
            # Safe to recycle: the device section ends with a stream sync,
            # so the DMA from this buffer has completed.
            _pinned_pool_put(pinned)

        # Pack into (NY, NX, MAX_PEAKS, 3) NaN-padded
        t0 = time.perf_counter()
        out = np.full((NY, NX, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        min_d2 = int(min_dist) * int(min_dist)
        # Raw frames for intensity sampling. `block4d_cpu` is the FULL ghost
        # block (NY_g x NX_g), but the device section trimmed the ghost zone:
        # peak index i in [0, N) is over the CORE grid (NY x NX) starting at
        # (lo, lo). Index the ghost block at the core-shifted position, else
        # intensities are read from the wrong frame and grow more misaligned
        # per core row — chunk-aligned intensity tearing (positions stay
        # correct). Only bites when depth_px > 0 (nav blur on), so sigma=0
        # tests don't catch it.
        NY_g_local, NX_g_local = block4d_cpu.shape[0], block4d_cpu.shape[1]
        # Mirror the device section's trim bounds (see _device_section): lo is
        # the ghost depth, with the same too-small-block fallback to 0.
        lo = depth_px
        if lo >= NY_g_local - depth_px or lo >= NX_g_local - depth_px:
            lo = 0
        host_frames = block4d_cpu.reshape(-1, KY, KX)
        for i in range(N):
            iy, ix = divmod(i, NX)
            host_i = (iy + lo) * NX_g_local + (ix + lo)
            if subpixel:
                np_i = min(int(n_peaks[i]), MAX_PEAKS)
                frame_peaks = peaks_out[i, :np_i].copy() if np_i > 0 else None
            else:
                yx = np.argwhere(peak_mask[i] > 0)
                if len(yx) == 0:
                    continue
                scores = raw_corr[i][yx[:, 0], yx[:, 1]]
                frame_peaks = np.column_stack([yx.astype(np.float32), scores])

            if frame_peaks is None or len(frame_peaks) == 0:
                continue

            # Beamstop exclusion on surviving peaks
            if have_mask:
                ky_px = np.clip(frame_peaks[:, 0].astype(int), 0, KY - 1)
                kx_px = np.clip(frame_peaks[:, 1].astype(int), 0, KX - 1)
                frame_peaks = frame_peaks[~beamstop_mask[ky_px, kx_px]]

            # Greedy NMS
            if len(frame_peaks) > 1:
                # Deterministic order: the GPU subpixel kernel fills slots via
                # atomics in arbitrary order, so tie-break equal scores by
                # position to keep results reproducible run-to-run.
                order = np.lexsort(
                    (frame_peaks[:, 1], frame_peaks[:, 0], -frame_peaks[:, 2])
                )
                frame_peaks = frame_peaks[order]
                kept = np.ones(len(frame_peaks), dtype=bool)
                for j in range(len(frame_peaks)):
                    if not kept[j]:
                        continue
                    dy = frame_peaks[j + 1:, 0] - frame_peaks[j, 0]
                    dx = frame_peaks[j + 1:, 1] - frame_peaks[j, 1]
                    kept[j + 1:][(dy * dy + dx * dx) <= min_d2] = False
                frame_peaks = frame_peaks[kept]

            # Value column → raw experimental intensity at each peak (after NMS,
            # which used the corr score, matching the CPU path). Disk-mean over the
            # kernel footprint — robust, avoids the per-frame single-pixel banding.
            frame_peaks = _with_raw_intensity(host_frames[host_i], frame_peaks,
                                              radius=kernel_r)
            n = min(len(frame_peaks), MAX_PEAKS)
            if n > 0:
                out[iy, ix, :n, :] = frame_peaks[:n]
        timings["pack"] += time.perf_counter() - t0

        return out

    # ── Dispatch: 4D or 5D ────────────────────────────────────────────────────
    if nav_dim == 2:
        result = _process_4d(block_host)
        core_shape = result.shape[:2]
    else:
        n_lead = nav_shape_ghost[0]
        # Ghost block is (t, ny_ghost, nx_ghost, KY, KX) for 5D
        ny_ghost = nav_shape_ghost[1]
        nx_ghost = nav_shape_ghost[2]
        ny = max(1, ny_ghost - 2 * depth_px)
        nx = max(1, nx_ghost - 2 * depth_px)
        out5 = np.full((n_lead, ny, nx, MAX_PEAKS, 3), np.nan, dtype=np.float32)
        for t in range(n_lead):
            out5[t] = _process_4d(block_host[t])
        result = out5
        core_shape = (n_lead, ny, nx)

    total_ms = (time.perf_counter() - t_start) * 1e3
    if stage_timing:
        log.debug(
            f"[find_vectors] GPU chunk core={tuple(int(s) for s in core_shape)} sig=({KY},{KX}) "
            f"path={nx_path[0]} "
            + " ".join(f"{k}={v * 1e3:.0f}ms" for k, v in timings.items())
            + f" total={total_ms:.0f}ms"
        )
    else:
        # Without per-stage syncs only host-blocking stages are meaningful.
        log.debug(
            f"[find_vectors] GPU chunk core={tuple(int(s) for s in core_shape)} sig=({KY},{KX}) "
            f"path={nx_path[0]} "
            f"stage={timings['stage'] * 1e3:.0f}ms h2d={timings['h2d'] * 1e3:.0f}ms "
            f"d2h={timings['d2h'] * 1e3:.0f}ms "
            f"pack={timings['pack'] * 1e3:.0f}ms total={total_ms:.0f}ms"
        )
    return result

"""
gpu_runtime.py — numba-CUDA / CuPy runtime infrastructure for find_vectors.

Buffer + pinned-host pools, per-thread CUDA streams, CuPy availability probe,
GPU slot/latch/warmup state and the GPU task-allow policy, plus the context
reset used to recover from a poisoned CUDA context.

Pure infrastructure: no peak-finding math lives here (see kernels.py /
detectors.py / chunk.py).  Mutable module-level state (pools, latches) is
shared by kernels.py and chunk.py via direct imports of this module.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

# Maximum peaks per frame in GPU subpixel output buffer
MAX_PEAKS: int = 512

# Cache of device-side disk kernel arrays keyed by kernel_r
_gpu_disk_cache: dict = {}

# Native-endian integer dtypes the GPU converts to float32 on device.
# Uploading the raw integers halves H2D bytes for 16-bit detectors and removes
# the host-side astype pass.  float64 and big-endian (e.g. mrc '>u2') data are
# converted on the host instead.
_GPU_NATIVE_DTYPES = {
    np.dtype(t) for t in (np.uint8, np.int8, np.uint16, np.int16,
                          np.uint32, np.int32)
}

# ── Device buffer pool ────────────────────────────────────────────────────────
# numba-cuda has no caching allocator: every device_array() is a cudaMalloc,
# and allocation/free are context-wide sync points.  With several chunk tasks
# in flight on the GPU worker's threads the malloc/free storm serialises the
# whole device.  Chunks within a run share a handful of shapes, so a small
# free-list pool removes nearly all allocations.
_GPU_POOL_MAX_PER_KEY = 8
_gpu_buffer_pool: dict = {}
_gpu_pool_bytes: list = [0]
_gpu_pool_max_bytes: list = [None]  # resolved lazily from device VRAM
_gpu_pool_lock = threading.Lock()


def _gpu_pool_cap() -> int:
    """Pool byte cap: half of total VRAM (a too-small cap rejects returns and
    the resulting cudaFree storm device-syncs every thread)."""
    if _gpu_pool_max_bytes[0] is None:
        cap = 2_000_000_000
        try:
            from numba import cuda as _cuda
            _free, total = _cuda.current_context().get_memory_info()
            cap = int(total * 0.5)
        except Exception as e:
            log.debug("VRAM probe failed, using default GPU pool cap: %s", e)
        _gpu_pool_max_bytes[0] = cap
    return _gpu_pool_max_bytes[0]


def _gpu_pool_get(shape, dtype):
    """Reuse a pooled device array of this shape/dtype, or allocate one."""
    from numba import cuda as _cuda
    key = (tuple(int(s) for s in shape), np.dtype(dtype).str)
    with _gpu_pool_lock:
        lst = _gpu_buffer_pool.get(key)
        if lst:
            arr = lst.pop()
            _gpu_pool_bytes[0] -= arr.nbytes
            return arr
    return _cuda.device_array(shape, dtype=dtype)


def _gpu_pool_put(*arrays):
    """Return device arrays to the pool (over-cap buffers are just dropped)."""
    with _gpu_pool_lock:
        for arr in arrays:
            if arr is None:
                continue
            key = (tuple(int(s) for s in arr.shape), np.dtype(arr.dtype).str)
            lst = _gpu_buffer_pool.setdefault(key, [])
            if (len(lst) < _GPU_POOL_MAX_PER_KEY
                    and _gpu_pool_bytes[0] + arr.nbytes <= _gpu_pool_cap()):
                lst.append(arr)
                _gpu_pool_bytes[0] += arr.nbytes


# ── Pinned (page-locked) host staging buffers ─────────────────────────────────
# H2D copies from pageable numpy (what dask hands us) block the calling
# thread and run at reduced bandwidth.  Staging through a reused pinned
# buffer makes copy_to_device(..., stream=) truly asynchronous: the call
# returns immediately and the DMA overlaps another chunk's kernels.
# cudaHostAlloc is expensive, so buffers are pooled; pinned memory is wired
# RAM, so total allocation is capped and failure degrades to pageable.
_PINNED_POOL_MAX_PER_KEY = 4
_PINNED_POOL_MAX_BYTES = 3_000_000_000
_pinned_pool: dict = {}
_pinned_alloc_bytes = [0]  # allocated total: pooled + in flight
_pinned_failed = [False]


def _pinned_pool_get(shape, dtype):
    """A reused page-locked staging buffer, or None when unavailable."""
    if _pinned_failed[0]:
        return None
    key = (tuple(int(s) for s in shape), np.dtype(dtype).str)
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    with _gpu_pool_lock:
        lst = _pinned_pool.get(key)
        if lst:
            return lst.pop()
        if _pinned_alloc_bytes[0] + nbytes > _PINNED_POOL_MAX_BYTES:
            return None
        _pinned_alloc_bytes[0] += nbytes
    try:
        from numba import cuda as _cuda
        return _cuda.pinned_array(shape, dtype=dtype)
    except Exception:
        with _gpu_pool_lock:
            _pinned_alloc_bytes[0] -= nbytes
        _pinned_failed[0] = True
        log.debug("[find_vectors] pinned allocation failed — using pageable H2D")
        return None


def _pinned_pool_put(arr):
    if arr is None:
        return
    key = (tuple(int(s) for s in arr.shape), np.dtype(arr.dtype).str)
    with _gpu_pool_lock:
        lst = _pinned_pool.setdefault(key, [])
        if len(lst) < _PINNED_POOL_MAX_PER_KEY:
            lst.append(arr)
        else:
            _pinned_alloc_bytes[0] -= arr.nbytes  # dropped — freed by GC


# ── Per-thread CUDA streams ───────────────────────────────────────────────────
# Each chunk task thread gets its own stream: one chunk's H2D/D2H overlaps
# another's kernels instead of everything serialising on the legacy default
# stream.  CuPy work is bound to the same stream via ExternalStream so the
# whole per-chunk pipeline stays in-order on one queue.
_thread_streams = threading.local()
# Bumped by _reset_gpu_state so threads recreate streams after cuda.close()
_gpu_context_gen = [0]
# Fixed pool of streams handed out round-robin.  Creating a stream per thread
# leaks CuPy per-stream arenas and plan caches when threads churn (each dead
# thread's arena pins VRAM until a full pool reset) — a bounded set of
# long-lived streams keeps device state constant regardless of thread count.
_GPU_STREAM_POOL_SIZE = 4
_gpu_stream_pool: list = []
_gpu_stream_rr = [0]


def _get_thread_stream():
    s = getattr(_thread_streams, "stream", None)
    if s is not None and getattr(_thread_streams, "gen", -1) == _gpu_context_gen[0]:
        return s
    from numba import cuda as _cuda
    with _gpu_pool_lock:
        if _gpu_stream_pool and _gpu_stream_pool[0][0] != _gpu_context_gen[0]:
            _gpu_stream_pool.clear()  # context was reset — streams are dead
        if len(_gpu_stream_pool) < _GPU_STREAM_POOL_SIZE:
            s = _cuda.stream()
            _gpu_stream_pool.append((_gpu_context_gen[0], s))
        else:
            _gpu_stream_rr[0] = (_gpu_stream_rr[0] + 1) % _GPU_STREAM_POOL_SIZE
            s = _gpu_stream_pool[_gpu_stream_rr[0]][1]
    _thread_streams.stream = s
    _thread_streams.gen = _gpu_context_gen[0]
    return s


def _stream_ptr(stream) -> int:
    """Raw cudaStream_t pointer of a numba stream (for CuPy ExternalStream)."""
    h = stream.handle
    return int(getattr(h, "value", h) or 0)


# ── CuPy / cuFFT availability ─────────────────────────────────────────────────
_gpu_cache_lock = threading.Lock()
_gpu_disk_fft_conj_cache: dict = {}
_cupy_state = {"checked": False, "ok": False}


def _cupy_available() -> bool:
    """True when CuPy with a working CUDA runtime is importable.  Set
    SPYDE_FV_GPU_FFT=0 to force the brute-force numba NXCORR kernels."""
    import os
    if os.environ.get("SPYDE_FV_GPU_FFT", "") in ("0", "off"):
        return False
    if not _cupy_state["checked"]:
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
            _cupy_state["ok"] = True
        except Exception:
            _cupy_state["ok"] = False
        _cupy_state["checked"] = True
    return _cupy_state["ok"]


# ── GPU task-allow policy + warmup / serialisation latches ────────────────────
# First-use serialisation: concurrent first-time kernel cache-loading /
# compilation and CUDA context initialisation from multiple worker threads
# has produced corrupt launches (CUDA_ERROR_INVALID_VALUE) — run the first
# chunk in each process alone, then go fully concurrent.  The first CUDA
# touch in a fresh dask worker can also fail transiently and leave numba's
# context state poisoned, so warmup resets it and only counts a SUCCESSFUL
# chunk as warmed; after repeated failures the GPU is disabled per-process.
_gpu_warmup_lock = threading.Lock()
_gpu_warmed = [False]
_gpu_warm_failures = [0]
_GPU_MAX_WARM_FAILURES = 3

# Optional serialisation of GPU chunk execution across this process's task
# threads.  With per-thread streams the default is full concurrency (the
# historical "concurrent launch" failures traced back to dask meta-inference
# launching empty grids, not to numba thread-safety); SPYDE_FV_GPU_SERIAL=1
# restores the old whole-chunk lock as an escape hatch.
_gpu_exec_lock = threading.Lock()

# Per-thread streams allow chunk overlap, but unbounded concurrency thrashes
# VRAM (each in-flight 512^2 chunk holds ~1 GB of buffers, FFT temporaries
# and per-thread cuFFT plan workspaces).  The semaphore bounds how many
# chunks may occupy the device section at once; the CPU pack stage runs
# outside it.  Tune with SPYDE_FV_GPU_CONC (default 2).
_gpu_slots_state: dict = {"sem": None, "n": None}


def _gpu_task_allowed() -> bool:
    """
    Decide whether the current chunk task may use the GPU.

    With opportunistic GPU use, every dask worker funnels its chunks through
    the single device, the kernels serialise, and the whole cluster collapses
    to GPU throughput while the CPU cores idle.  Default policy: exactly ONE
    worker (LocalCluster worker name "1") uses the GPU; all others run the
    CPU path in parallel, so total throughput is GPU rate + CPU rate.

    Override with the SPYDE_FV_GPU environment variable:
        "one" (default) — GPU on worker "1" only
        "<N>" (integer) — GPU on workers "1".."N" (overlaps one chunk's
                          H2D/pack stages with another's kernels)
        "all"           — every worker may use the GPU (single-GPU contention)
        "off"           — CPU everywhere
    Outside a distributed worker (threaded scheduler, tests) the GPU is allowed.
    """
    import os
    if _gpu_warm_failures[0] >= _GPU_MAX_WARM_FAILURES and not _gpu_warmed[0]:
        return False  # GPU disabled in this process after warmup failures
    mode = os.environ.get("SPYDE_FV_GPU", "one").lower()
    if mode == "off":
        return False
    if mode == "all":
        return True
    try:
        n_gpu_workers = max(0, int(mode))
    except ValueError:
        n_gpu_workers = 1  # "one" or anything unparseable
    try:
        from distributed import get_worker
        name = str(get_worker().name)
    except Exception:
        return True  # not running on a dask worker
    try:
        return 1 <= int(name) <= n_gpu_workers
    except ValueError:
        return name == "1"  # non-integer worker names: single GPU worker


def _gpu_slots():
    import os
    try:
        n = max(1, int(os.environ.get("SPYDE_FV_GPU_CONC", "2")))
    except ValueError:
        n = 2
    with _gpu_cache_lock:
        if _gpu_slots_state["sem"] is None or _gpu_slots_state["n"] != n:
            _gpu_slots_state["sem"] = threading.BoundedSemaphore(n)
            _gpu_slots_state["n"] = n
        return _gpu_slots_state["sem"]


def _gpu_serial_mode() -> bool:
    import os
    return os.environ.get("SPYDE_FV_GPU_SERIAL", "") not in ("", "0", "off")


@contextlib.contextmanager
def _interprocess_warmup_lock():
    """
    Cross-process file lock held while a process compiles / cache-loads the
    CUDA kernels (its first chunk).  numba's on-disk kernel cache is not safe
    against concurrent writers on Windows: several worker processes compiling
    a cold cache simultaneously produce torn reads and corrupt launches
    (CUDA_ERROR_INVALID_VALUE).  Serialising first chunks across processes
    makes the cache single-writer; steady-state chunks never touch this.
    """
    import os
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "spyde_fv_cuda_warmup.lock")
    fd = None
    deadline = time.time() + 300.0
    while fd is None:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Remove stale locks (e.g. a killed worker)
            try:
                if time.time() - os.path.getmtime(path) > 120.0:
                    os.remove(path)
                    continue
            except OSError as e:
                log.debug("stale-lock check on %s failed: %s", path, e)
            if time.time() > deadline:
                break  # give up on locking rather than deadlock
            time.sleep(0.1)
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
                os.remove(path)
            except OSError as e:
                log.debug("releasing lock file %s failed: %s", path, e)


def _reset_gpu_state():
    """Tear down numba's CUDA context state and our device-side caches so the
    next attempt starts from a clean slate (cached handles die with the
    context)."""
    try:
        from numba import cuda as _cuda
        _cuda.close()
    except Exception as e:
        log.debug("numba CUDA context teardown failed: %s", e)
    with _gpu_pool_lock:
        _gpu_buffer_pool.clear()
        _gpu_pool_bytes[0] = 0
    _gpu_disk_cache.clear()
    with _gpu_cache_lock:
        _gpu_disk_fft_conj_cache.clear()
    _gpu_pool_max_bytes[0] = None
    with _gpu_pool_lock:
        _pinned_pool.clear()
        _pinned_alloc_bytes[0] = 0
    # Invalidate per-thread streams created on the dead context
    _gpu_context_gen[0] += 1

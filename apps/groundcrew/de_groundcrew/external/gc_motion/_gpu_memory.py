"""Return CuPy-held VRAM to the driver between GPU operations.

CuPy's default memory pool caches freed blocks, and the cuFFT plan cache pins a
per-shape workspace that is allocated FROM that pool. Neither returns to the driver
on its own, and the workspace STACKS across operations (load -> motion -> CTF ...).
On the 8 GB dev card that retained memory starves a later single-image display FFT
(the "movie resident -> take single image -> OOM" bug).

Call release_gpu_memory() in a worker's finally block after results are on the host.
No-op without CuPy; never raises (safe in finally).

ORDER IS LOAD-BEARING (measured 2026-07-18, notes/runs/2026-07-18-vram-occupancy-measurement.md):
the plan-cache workspace pins split pool blocks, so free_all_blocks() BEFORE the clear
returns ~nothing. Clear the plan cache FIRST, then free the pools.
"""


def release_gpu_memory(cp=None):
    """Free CuPy plan cache + default pool + pinned pool back to the driver.

    `cp`: inject a cupy-like module (tests). If None, cupy is imported lazily;
    a missing/broken CuPy makes this a silent no-op.
    """
    if cp is None:
        try:
            import cupy as cp  # noqa: PLC0415 — lazy by design (frozen-build safe)
        except Exception:
            return
    try:
        # 1. FIRST: clear the cuFFT plan cache -> releases its workspace back to the pool.
        cp.fft.config.get_plan_cache().clear()
        # 2. THEN: return the now-free pool blocks to the driver.
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        # cleanup is best-effort; it must never propagate into a worker's finally
        pass

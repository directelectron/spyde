"""
heavy_imports.py — single-flight import of the heavy analysis stack.

hyperspy + pyxem have internal circular imports; importing them CONCURRENTLY
from two threads (the startup prewarm and a data-load thread) can surface
``cannot import name … from partially initialized module 'pyxem.signals'``
and permanently poison ``sys.modules`` for the session. This became likely
once the backend ran at full speed (the Electron tick fix removed the frozen
timers that accidentally serialized startup).

Every thread that touches hyperspy/pyxem for the first time calls
``ensure_heavy_imports()`` first: one thread performs the import, the others
wait on the lock, and subsequent calls are a no-op check.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_DONE = False

# ── torch / CUDA prewarm (non-blocking) ──────────────────────────────────────
# torch's import is ~3 s on an idle box and MUCH worse while the disk is busy
# (e.g. the navigator fill of a fresh 16 GB movie) — and the GPU tile backend
# used to pay it lazily ON THE PAINTER THREAD at the first large-frame paint,
# which is exactly when the fill saturates the disk: the signal panel stayed
# black for tens of seconds. Prewarm it in the background instead; consumers
# poll ``torch_cuda_ready()`` and take a CPU path until it flips.
_TORCH_LOCK = threading.Lock()
_TORCH_STARTED = False
_TORCH_READY = False        # torch imported AND a CUDA device is usable
# Set when the prewarm thread finishes, WHETHER OR NOT a GPU was found. Distinct
# from _TORCH_READY: a consumer that only needs torch-CPU (the drift preview
# solve) cares that the ~3 s IMPORT is done, not whether CUDA exists.
_TORCH_IMPORT_DONE = threading.Event()


def torch_imported() -> bool:
    """True iff the torch import has finished. Never blocks (see
    :func:`torch_cuda_ready` on why touching a mid-import torch is a stall)."""
    return _TORCH_IMPORT_DONE.is_set()


def wait_for_torch(timeout: float = 60.0) -> bool:
    """Block until torch is imported. Returns False on timeout.

    **Never call this on the asyncio main thread or the nav dispatcher** — it can
    block for the full ~3 s import, and both of those are latency-critical. It
    exists so a WORKER can absorb that cost before an interactive path needs
    torch: the drift caret waits here on a worker so the first preview step,
    which runs on the shared dispatcher, can never be the thing that pays it.
    """
    prewarm_torch_cuda()                 # idempotent; starts one if none running
    if _TORCH_IMPORT_DONE.wait(timeout):
        return True
    log.info("torch still importing after %.0fs; caller falls back to numpy", timeout)
    return False


def torch_cuda_ready() -> bool:
    """True iff torch is FULLY imported with a usable CUDA device. Never blocks:
    while a prewarm is IN FLIGHT this reads only the flag — touching a
    mid-import torch (sys.modules holds the partial module) blocks the caller
    on the import lock until the import finishes, which is EXACTLY the
    painter-thread stall this machinery exists to avoid (measured 11 s while
    the navigator fill saturated the disk). The synchronous resolve below runs
    only when NO prewarm was ever started — i.e. torch was fully imported by
    someone else up front (the GPU test subprocess does this)."""
    global _TORCH_READY
    if _TORCH_READY:
        return True
    if not _TORCH_STARTED:
        import sys
        t = sys.modules.get("torch")
        if t is not None:
            try:
                if t.cuda.is_available():
                    _TORCH_READY = True
                    return True
            except Exception:
                pass
            return False
        prewarm_torch_cuda()
    return _TORCH_READY


def prewarm_torch_cuda() -> None:
    """Import torch + initialise the CUDA context on a background daemon thread
    (idempotent). ~3 s of background work at startup instead of a first-paint
    stall; harmless no-op on a CPU-only box (ready simply stays False).

    Skipped under pytest: torch-CUDA work inside the pytest process segfaults on
    Windows (see CLAUDE.md) — GPU correctness tests run in a subprocess."""
    global _TORCH_STARTED
    with _TORCH_LOCK:
        if _TORCH_STARTED:
            return
        _TORCH_STARTED = True
    import os
    if "PYTEST_CURRENT_TEST" in os.environ:
        log.debug("torch prewarm skipped under pytest")
        _TORCH_IMPORT_DONE.set()     # never leave a waiter hanging
        return

    def _warm():
        global _TORCH_READY
        try:
            import torch
            if torch.cuda.is_available():
                # Touch the device so the CUDA primary context is built now —
                # the first tensor op otherwise pays it (~0.3-5 s).
                torch.zeros(1, device="cuda")
                torch.cuda.synchronize()
                _TORCH_READY = True
                log.info("torch CUDA prewarmed: %s", torch.cuda.get_device_name(0))
            else:
                log.info("torch imported; no CUDA device — GPU paths stay off")
        except Exception as e:
            log.info("torch/CUDA prewarm failed (CPU paths only): %s", e)
        finally:
            # Set even on failure: a waiter must not hang because torch is
            # missing, it must fall through to the numpy path.
            _TORCH_IMPORT_DONE.set()

    threading.Thread(target=_warm, daemon=True, name="torch-prewarm").start()


def ensure_heavy_imports() -> None:
    global _DONE
    if _DONE:
        return
    with _LOCK:
        if _DONE:
            return
        import hyperspy.api  # noqa: F401
        try:
            import pyxem  # noqa: F401
        except Exception as e:
            # pyxem is required by the diffraction paths but a plain-imaging
            # session can live without it — don't fail the whole import gate.
            log.warning("pyxem import failed during heavy-import warmup: %s", e)
        # Apply all upstream monkey-patches now — AFTER the upstream import, BEFORE
        # first use. Their defined home is spyde.external (one subpackage per
        # upstream package). This replaces the old inline _patch_cached_dask_client
        # call; that name is kept below as a back-compat shim.
        from spyde.external import apply_all
        apply_all()
        _DONE = True


def _patch_cached_dask_client() -> None:
    """Back-compat shim — the CachedDaskArray.client patch now lives in
    :mod:`spyde.external.hyperspy.cached_dask_array` (see that module for the
    full WHAT/WHY/WHEN-TO-REMOVE writeup). Kept so existing imports
    (``from spyde.backend.heavy_imports import _patch_cached_dask_client``, e.g.
    ``test_cache_client_patch.py``) keep working. Idempotent."""
    from spyde.external.hyperspy.cached_dask_array import apply
    apply()

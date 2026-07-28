"""device_lock.py — THE process-wide accelerator serialisation lock.

Apple-MPS (Metal) crashes when two Python threads submit work to the device at
the same time: the Metal compute-context / shader-library bookkeeping inside
``libtorch_cpu`` is not thread-safe, and a concurrent submission corrupts the
command encoder. The failure is an **uncatchable native SIGSEGV** — no
``try``/``except`` sees it, the whole backend process dies, and the user gets
"Analysis backend stopped". Observed faulting frames (macOS 26.4, torch 2.13):

    at::native::relu_mps_ -> MetalShaderLibrary::exec_unary_kernel
    at::native::zero_ -> fill_mps_kernel -> [AGXG13GFamilyComputeContext
                                             setComputePipelineState:]

Reproduced standalone in ~30 lines: 4 threads running the same small conv/ReLU
net on MPS segfaults (or hangs) within a few dozen iterations; the identical
loop with one shared lock runs clean indefinitely.

**A lock only works if EVERY participant takes it.** SpyDE has several
independent torch users that can run on worker threads at the same time — the
neural spot detector (batch, single-frame preview, and calibration), the torch
NXCORR/DoG peak finders, and the batched vector-orientation fit. They must all
serialise against ONE lock object, which is what this module owns. Before this
existed the lock lived in ``find_vectors_torch`` and only the *batch* neural
path and the NXCORR paths took it, so a live navigator preview or an
orientation fit ran concurrently with them and took the process down.

Scope: MPS only. On CUDA the driver is thread-safe and concurrent streams are a
deliberate throughput win (the find_vectors GPU lane runs several submitters);
serialising there would be a pure regression. So ``accelerator_lock`` is a null
context off MPS, and the CUDA behaviour is byte-for-byte unchanged.

The lock is **reentrant**: nested acquisitions on one thread are normal here
(e.g. the neural batch takes the device lock and then the ``_gpu_slots``
semaphore, and helpers may re-enter), and an ``RLock`` keeps that from
self-deadlocking.
"""
from __future__ import annotations

import contextlib
import logging
import threading

log = logging.getLogger(__name__)

# THE lock. Reentrant so nesting on one thread is safe. Module-global, so every
# importer in this process shares the single object.
DEVICE_LOCK = threading.RLock()


def is_mps(device) -> bool:
    """True when ``device`` (a torch.device, a string, or None) is Apple-MPS."""
    if device is None:
        return False
    return getattr(device, "type", str(device)) == "mps"


def mps_sync() -> None:
    """Drain the Metal command queue. Called before releasing the lock so the
    device is quiesced at the hand-off — releasing while kernels are still in
    flight would let the next thread start submitting into a live encoder, which
    is the very race the lock exists to prevent. Best-effort: a torch build
    without ``torch.mps.synchronize`` (or with MPS uninitialised) just no-ops."""
    try:
        import torch

        mps = getattr(torch, "mps", None)
        sync = getattr(mps, "synchronize", None)
        if sync is not None:
            sync()
    except Exception as e:  # noqa: BLE001 - never let a sync failure escape
        log.debug("mps_sync skipped: %s", e)


@contextlib.contextmanager
def accelerator_lock(device=None, *, sync: bool = True):
    """Serialise a block of torch work against every other MPS user in-process.

    ``device`` is the torch device the block will run on; when it is not MPS this
    is a null context (see module docstring — CUDA concurrency is intentional).
    Pass ``device=None`` to mean "unknown/any", which also skips locking.

    ``sync=False`` skips the drain on release — only for a block that has already
    synchronised (or read a tensor back to the host, which syncs implicitly).
    """
    if not is_mps(device):
        yield
        return
    with DEVICE_LOCK:
        try:
            yield
        finally:
            if sync:
                mps_sync()

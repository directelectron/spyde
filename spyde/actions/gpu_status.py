"""
gpu_status.py — surfaces the GPU diagnostics that vector_orientation_gpu already
computes (select_device / gpu_available / gpu_unavailable_reason /
torch_available) to the Help -> GPU Status dialog, so a silent CPU fallback is
never a mystery (see DISTRIBUTION_PLAN.md Sec 3d). No new detection logic —
this module only packages the existing functions into an emitted message.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def get_gpu_status(session, plot, payload) -> None:
    """Staged handler: report torch/GPU availability. Emits ``gpu_status_result``.

    Runs inline (no thread) — every call here is a cheap ``torch.cuda`` /
    ``torch.backends.mps`` attribute check, not a compute, so there is no
    blocking-the-main-loop concern (see vector_orientation_gpu.select_device).
    """
    # Imported lazily (not at module scope) so `emit` is re-resolved against
    # spyde.backend.ipc on every call — a module-level `from ipc import emit`
    # binds once at first import, which is stale for any test that re-patches
    # ipc.emit per-fixture after this module is already cached in sys.modules.
    from spyde.backend.ipc import emit
    from spyde.actions.vector_orientation_gpu import (
        select_device, gpu_available, gpu_unavailable_reason, torch_available,
    )

    device = None
    torch_version = None
    if torch_available():
        try:
            import torch
            torch_version = torch.__version__
            dev = select_device()
            device = dev.type if dev is not None else None
        except Exception as e:
            log.debug("gpu_status device introspection failed: %s", e)

    emit({
        "type": "gpu_status_result",
        "torch_available": torch_available(),
        "torch_version": torch_version,
        "device": device,
        "gpu_available": gpu_available(),
        "reason": gpu_unavailable_reason(),
    })

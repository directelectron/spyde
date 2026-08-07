"""_device.py — which torch device the EBSD paths run on.

Its own module (rather than living in ``indexing.py``) so ``preprocess`` and
``refine`` can share it without importing the indexer, and so there is exactly
one place to change the policy.

The policy matches ``spyde.fitting.engine.default_device`` and
``vector_orientation_gpu.select_device``: CUDA > MPS > CPU. It is duplicated
rather than imported from those because a compute package must not depend on
``spyde/actions/`` (actions/README.md), and fitting and EBSD are independent
domains with their own override knobs.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def default_device() -> str:
    """CUDA > MPS > CPU, overridable with ``SPYDE_EBSD_DEVICE``.

    Unlike the fitting engine — whose ``n <= 20`` parameter blocks make it a
    stream of tiny kernels — dictionary indexing IS GEMM-bound (one big
    ``E @ Dᵀ``), so this is the path that actually gains from an accelerator.
    Measured on an M1 Air (8 GPU cores), float32, best of 3 after a warmup:

    |     P |     D | detector | CPU     | MPS     | MPS/CPU |
    |-------|-------|----------|---------|---------|---------|
    |   256 |  1000 |    60^2  |    4.7 ms |    9.3 ms | 0.50x |
    |  1024 |  5000 |    60^2  |   54.7 ms |   64.0 ms | 0.85x |
    |  4096 | 20000 |    60^2  |  727 ms   |  685 ms   | 1.06x |
    |  4096 | 20000 |    80^2  | 1286 ms   | 1112 ms   | 1.16x |

    Monotonic in problem size and crossing 1.0 at production scale, which is the
    shape a GEMM-bound path should have: fixed per-launch overhead amortised
    against work that grows. Small indexes stay faster on the CPU and that is
    fine — they are already milliseconds. Do NOT add a size threshold on the
    strength of this table; it is one (small) GPU, and a Pro/Max/Ultra part
    moves every row. Measure with ``SPYDE_EBSD_DEVICE`` instead.

    NB these are single-shot timings of a warm process. The first call on MPS
    also pays Metal context creation (~0.5 s), which is why the un-warmed
    version of this measurement reported 0.02x and was misleading.
    """
    forced = os.environ.get("SPYDE_EBSD_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception as e:                                # pragma: no cover
        log.debug("probing the accelerator failed: %s", e)
    return "cpu"


def resolve_dtype(device: str, dtype: str) -> str:
    """Metal has no float64 — coerce here rather than raise deep in a tile loop.

    Indexing already defaults to float32 (NCC does not need more), so this only
    bites a caller that asks for double explicitly.
    """
    if str(device).startswith("mps") and dtype == "float64":
        log.debug("MPS does not support float64; running in float32")
        return "float32"
    return dtype

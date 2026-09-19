"""torch_device.py — which accelerator to use, and making it safe to use.

Selection (CUDA > Apple-MPS > CPU) and the one-off CUDA autograd warm-up, kept
apart from :mod:`spyde.device_lock`, which is about serialising access to the
device once chosen.

These lived in the batched vector-orientation fit, because that is where they
were first needed. They are not about orientation mapping — EBSD refinement,
the GPU status dialog and the fitting engine all ask the same two questions —
so they outlived the fit that introduced them.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

def select_device():
    """Best available torch device: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.

    torch gives one batched/autograd API across all three, so the same fit runs
    GPU-accelerated on Windows/Linux+CUDA and on Apple-Silicon Macs via Metal,
    and still-batched (far faster than the serial scipy path) on plain CPU.
    Returns a torch.device or None if torch isn't importable.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except Exception:
        return None


def gpu_available() -> bool:
    """True when a hardware-accelerated torch device (CUDA or Apple-MPS) exists.

    CPU-only torch returns False here — the batched torch path still *runs* on
    CPU (and is used as the fast fallback), but this gate is what the UI uses to
    decide whether to expect interactive speed.
    """
    dev = select_device()
    return dev is not None and dev.type in ("cuda", "mps")


def torch_available() -> bool:
    """True when the batched torch path can run at all (any device, incl. CPU)."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def gpu_unavailable_reason() -> str:
    """Human-readable reason the accelerated path is off — for surfacing in the
    UI/logs so a silent fall-through to the slow CPU fit is never a mystery."""
    try:
        import torch
    except Exception as e:
        return f"torch not importable ({e})"
    try:
        if torch.cuda.is_available():
            return "CUDA available"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "Apple MPS available"
        return ("torch present but no CUDA/MPS device "
                f"(torch {torch.__version__}; CPU-only build or no GPU visible)")
    except Exception as e:
        return f"device check raised: {e}"


_AUTOGRAD_WARMED = False


def warmup_autograd() -> None:
    """Initialise the CUDA autograd engine on the *calling* thread.

    torch's CUDA autograd backward segfaults on Windows the first time it runs
    on a thread whose engine hasn't been initialised. The GPU compute runs on a
    daemon worker, so we run one trivial backward on the GUI/main thread first
    (idempotent) — afterwards the worker thread's backward is safe.
    """
    global _AUTOGRAD_WARMED
    if _AUTOGRAD_WARMED:
        return
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.zeros(1, device="cuda", requires_grad=True)
            (x * 2).sum().backward()
        _AUTOGRAD_WARMED = True
    except Exception as e:
        log.debug("CUDA autograd warmup skipped: %s", e)

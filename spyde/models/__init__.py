"""SpyDE's vendored neural disk detector (SpotUNet) + an upgradeable model registry.

The architecture/inference code (``unet``, ``preprocess``, ``decode``, ``infer``)
is a self-contained copy of the ``yoloDiffraction`` research project so SpyDE ships
and runs without it. ``registry`` resolves which checkpoint to load — a bundled
default now, with additional/upgraded models registered (and Hugging-Face-hosted)
later, see ``RELEASING.md``.

Typical use from the find-vectors detector:

    from spyde import models
    model, device = models.get_model()              # default, cached
    peaks = models.detect(model, frame, device)     # (N,3) [y,x,score]
    batch = models.detect_batch(model, frames, device)
"""
from __future__ import annotations

from .infer import (
    auto_bg_sigma,
    bg_sigma_from_peak_size,
    calibrate,
    detect,
    detect_batch,
    enable_mps_cpu_fallback,
    load_model,
    mps_batch_allowed,
)
from .registry import (
    available_models,
    default_model_id,
    demote_cached_models_to_cpu,
    ensure_local,
    get_cpu_model,
    get_model,
    is_cached,
    list_models,
    refresh_remote_registry,
)
from .unet import SpotUNet

__all__ = [
    "SpotUNet",
    "load_model",
    "detect",
    "detect_batch",
    "calibrate",
    "auto_bg_sigma",
    "bg_sigma_from_peak_size",
    "get_model",
    "get_cpu_model",
    "list_models",
    "available_models",
    "default_model_id",
    "ensure_local",
    "is_cached",
    "refresh_remote_registry",
    # Apple-MPS neural-batch gates (used by find_vectors_neural / orchestrate).
    # These are referenced as ``models.<name>`` and MUST be exported here — an
    # unexported name AttributeErrors only on Mac (where dev_is_mps is True), so
    # a Windows/CUDA dev box never surfaces the miss.
    "mps_batch_allowed",
    "enable_mps_cpu_fallback",
    "demote_cached_models_to_cpu",
]

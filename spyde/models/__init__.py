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
    load_model,
)
from .registry import (
    available_models,
    default_model_id,
    ensure_local,
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
    "list_models",
    "available_models",
    "default_model_id",
    "ensure_local",
    "is_cached",
    "refresh_remote_registry",
]

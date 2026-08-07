"""spyde.fitting — HyperSpy model fitting, made fast.

**Not to be confused with** :mod:`spyde.models`, which is the neural
disk-detector package (SpotUNet + the Hugging Face weight registry). Unrelated
meanings of the word "model"; do not merge them.

Layout:

``spec``
    :class:`~spyde.fitting.spec.ModelSpec` — a serialisable model description
    that round-trips against HyperSpy's ``BaseModel.as_dictionary()``. The
    contract shared by HyperSpy, the batched engine and the UI.

``components``
    Batched, autograd-differentiable torch ports of HyperSpy's components,
    each evaluating every nav position at once.

``engine``
    The batched Levenberg-Marquardt fit over the whole navigation grid.

Why this exists: HyperSpy's ``multifit`` fits one pixel at a time, measured at
~110 spectra/s on this box — about 10 minutes for a 256x256 spectrum image.
The engine packs the whole grid into ``(P, C)`` and solves it as one batched
problem, following the same playbook as
``spyde/actions/vector_orientation_gpu.py``.

Correctness is defined by HyperSpy: the engine must reproduce ``multifit``'s
parameters on the same data. See GitHub #50.
"""
from __future__ import annotations

from spyde.fitting.spec import (
    ComponentSpec,
    ModelSpec,
    ParameterSpec,
    spec_from_component,
)

__all__ = ["ModelSpec", "ComponentSpec", "ParameterSpec", "spec_from_component"]

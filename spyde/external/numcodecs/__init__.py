"""
spyde.external.numcodecs — patches to numcodecs, the codec layer zarr reads
compressed chunks through.

Each module here documents WHAT / WHY / WHEN-TO-REMOVE and exposes an idempotent,
guarded ``apply()``. Importing this package self-registers those ``apply()``
callables with :mod:`spyde.external`.
"""
from __future__ import annotations

from spyde.external import register
from spyde.external.numcodecs.blosc_threads import apply as _apply_blosc_threads

register("numcodecs", _apply_blosc_threads)

__all__ = ["_apply_blosc_threads"]

"""
spyde.external.hyperspy — patches to the hyperspy fork
(``github.com/cssfrancis/hyperspy``, pinned in pyproject.toml).

Each module here documents WHAT / WHY / WHEN-TO-REMOVE and exposes an idempotent,
guarded ``apply()``. Importing this package self-registers those ``apply()``
callables with :mod:`spyde.external`.
"""
from __future__ import annotations

from spyde.external import register
from spyde.external.hyperspy.cached_dask_array import apply as _apply_cached_dask_array

register("hyperspy", _apply_cached_dask_array)

__all__ = ["_apply_cached_dask_array"]

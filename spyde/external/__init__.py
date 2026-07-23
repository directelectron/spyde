"""
spyde.external — a defined, discoverable home for monkey-patches to upstream
packages.

Modelled on hyperspy's own ``external`` module pattern: one **subpackage per
upstream package** we patch (``spyde.external.hyperspy``,
``spyde.external.rosettasciio``, …), each holding one module per distinct
deficiency it works around.

Why this exists
---------------
Patches to third-party code used to be scattered inline across the backend
(``heavy_imports._patch_cached_dask_client``, a per-page TIFF chunker in
``_session_files``, a ``psutil`` net-io stub in ``dask_manager``). Nobody could
answer "what do we patch upstream, and why, and when can we drop it?" without
grepping. This package is the single answer to that question.

The contract every patch module follows
----------------------------------------
Each patch module exposes an **idempotent** ``apply()`` that:

* documents, in its module docstring, exactly WHAT upstream deficiency it
  patches, WHY, and WHEN it can be removed (with an upstream issue/PR link when
  one exists);
* is safe to call twice (guards with a sentinel attribute / early-return);
* is **defensive** — if the upstream shape it depends on has changed (attribute
  gone, signature moved), it logs a warning and returns ``False`` rather than
  raising, so a rosettasciio/hyperspy bump can never crash startup;
* returns ``True`` if it applied (or was already applied), ``False`` if it
  skipped/failed.

Each patch module registers itself via :func:`register`. :func:`apply_all`
walks the registry and calls every ``apply()``. It is wired into the existing
startup gate — ``spyde.backend.heavy_imports.ensure_heavy_imports`` — so patches
land **after** the upstream import but **before** first use, exactly where
``_patch_cached_dask_client`` used to run.

Adding a new upstream patch
---------------------------
1. Create ``spyde/external/<upstream>/<deficiency>.py`` with a module docstring
   (WHAT / WHY / WHEN-TO-REMOVE) and an idempotent, guarded ``apply()``.
2. At import time, call ``register(<upstream>, apply)`` (see the existing
   modules — they self-register on import of the subpackage ``__init__``).
3. Ensure the subpackage is imported from :func:`apply_all` (add it to
   ``_PATCH_SUBPACKAGES`` below).

Nothing else — ``ensure_heavy_imports`` already calls ``apply_all``.
"""
from __future__ import annotations

import importlib
import logging
from typing import Callable

log = logging.getLogger(__name__)

# ordered (upstream_name, apply_callable) registry, populated by register()
_REGISTRY: list[tuple[str, Callable[[], bool]]] = []
_APPLIED = False

# Subpackages whose import triggers their patch modules to self-register.
# apply_all() imports these first so the registry is populated before it walks
# it — this keeps registration lazy (no upstream import at `import spyde.external`
# time) while still discoverable from one list.
_PATCH_SUBPACKAGES = (
    "spyde.external.hyperspy",
    "spyde.external.rosettasciio",
)


def register(upstream: str, apply_fn: Callable[[], bool]) -> None:
    """Register an idempotent ``apply()`` for *upstream* (e.g. ``"hyperspy"``).

    Called at import time by each patch subpackage. Duplicate registrations of
    the SAME callable are ignored so re-importing a subpackage is harmless."""
    for name, fn in _REGISTRY:
        if fn is apply_fn:
            return
    _REGISTRY.append((upstream, apply_fn))


def apply_all(force: bool = False) -> None:
    """Apply every registered upstream patch (idempotent).

    Imports each patch subpackage (which self-registers its ``apply()``), then
    calls them in registration order. Called from
    ``heavy_imports.ensure_heavy_imports`` after hyperspy/pyxem import but before
    first use. Safe to call more than once — each ``apply()`` is itself
    idempotent, and after the first successful pass this is a no-op unless
    ``force=True`` (used by tests)."""
    global _APPLIED
    if _APPLIED and not force:
        return
    for modname in _PATCH_SUBPACKAGES:
        try:
            importlib.import_module(modname)
        except Exception as e:  # pragma: no cover - defensive
            log.warning("spyde.external: could not import patch subpackage %s: %s",
                        modname, e)
    for upstream, apply_fn in list(_REGISTRY):
        try:
            ok = apply_fn()
            log.debug("spyde.external: %s patch %s -> %s",
                      upstream, getattr(apply_fn, "__module__", "?"),
                      "applied" if ok else "skipped")
        except Exception as e:  # pragma: no cover - a patch must never crash startup
            log.warning("spyde.external: %s patch %s raised (skipping): %s",
                        upstream, getattr(apply_fn, "__module__", "?"), e)
    _APPLIED = True

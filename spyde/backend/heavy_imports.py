"""
heavy_imports.py — single-flight import of the heavy analysis stack.

hyperspy + pyxem have internal circular imports; importing them CONCURRENTLY
from two threads (the startup prewarm and a data-load thread) can surface
``cannot import name … from partially initialized module 'pyxem.signals'``
and permanently poison ``sys.modules`` for the session. This became likely
once the backend ran at full speed (the Electron tick fix removed the frozen
timers that accidentally serialized startup).

Every thread that touches hyperspy/pyxem for the first time calls
``ensure_heavy_imports()`` first: one thread performs the import, the others
wait on the lock, and subsequent calls are a no-op check.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_DONE = False


def ensure_heavy_imports() -> None:
    global _DONE
    if _DONE:
        return
    with _LOCK:
        if _DONE:
            return
        import hyperspy.api  # noqa: F401
        try:
            import pyxem  # noqa: F401
        except Exception as e:
            # pyxem is required by the diffraction paths but a plain-imaging
            # session can live without it — don't fail the whole import gate.
            log.warning("pyxem import failed during heavy-import warmup: %s", e)
        _DONE = True

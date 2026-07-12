"""
log_stream.py — forward Python ``logging`` records to the Electron app-log panel.

A single :class:`IPCLogHandler` is attached to the root logger at startup. Each
record it accepts is turned into a ``{"type": "log", ...}`` IPC message (over the
same PLOTAPP stdout channel ``emit`` uses) and appended to a bounded ring buffer
so a freshly-opened panel can backfill recent history.

Verbosity is controlled at runtime from the frontend (the level switcher) via the
``set_log_level`` staged handler. To keep the panel useful rather than flooded,
records from third-party loggers (dask/distributed/matplotlib/…) are only
forwarded at WARNING and above; ``spyde.*`` records are forwarded at the selected
level. So picking DEBUG shows SpyDE's own debug trail without drowning in library
chatter, while warnings/errors from anywhere always surface.

No Qt. The handler logs to the structured IPC channel; ordinary logging still
goes to stderr via whatever other handlers are configured.
"""
from __future__ import annotations

import collections
import logging
import threading
import traceback

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_handler: "IPCLogHandler | None" = None


# Map a logger name to a short, stable AREA tag so the log panel can group and
# filter by subsystem (the user copies only the relevant area's lines). Ordered
# longest/most-specific prefix first; first match wins. Extend freely as new
# subsystems get their own loggers.
_AREA_RULES = (
    ("spyde.dask_manager", "dask"),
    ("spyde.compute_backend", "dask"),
    ("spyde.drawing.update_functions", "navigator"),
    ("spyde.drawing.selectors", "navigator"),
    ("spyde.drawing.plots", "plots"),
    ("spyde.drawing.live_overlay", "overlay"),
    ("spyde.drawing", "drawing"),
    ("spyde.signal_tree", "navigator"),
    ("spyde.actions.find_vectors", "vectors"),
    ("spyde.actions.vector_orientation", "orientation"),
    ("spyde.actions.orientation", "orientation"),
    ("spyde.actions", "actions"),
    ("spyde.signals", "signals"),
    ("spyde.workers", "workers"),
    ("spyde.backend", "backend"),
    ("spyde.mdi_manager", "ui"),
    ("spyde.qt", "ui"),
    ("spyde.live", "instrument"),
    ("spyde", "spyde"),
    ("anyplotlib.tile", "plots"),
    ("anyplotlib", "plots"),
    ("distributed", "dask"),
    ("dask", "dask"),
    ("hyperspy", "hyperspy"),
    ("rsciio", "io"),
    ("pyxem", "pyxem"),
)


def _area_for(name: str) -> str:
    """Short subsystem tag for a logger name (e.g. 'navigator', 'dask')."""
    for prefix, area in _AREA_RULES:
        if name == prefix or name.startswith(prefix + "."):
            return area
    # Fall back to the top-level package so unmapped third-party loggers still
    # get a usable, filterable tag instead of nothing.
    return name.split(".", 1)[0] or "other"


def _coerce_level(level) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


class IPCLogHandler(logging.Handler):
    """A logging handler that streams records to the Electron app-log panel."""

    def __init__(self, level: int = logging.INFO, maxlen: int = 2000):
        super().__init__(level)
        self.buffer: collections.deque = collections.deque(maxlen=maxlen)
        # Re-entrancy guard: forwarding a record calls ipc.emit, and any logging
        # that happens *inside* that path would recurse back into this handler.
        self._guard = threading.local()

    # ── record → IPC ─────────────────────────────────────────────────────────
    def _entry(self, record: logging.LogRecord) -> dict:
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        if record.exc_info:
            msg = msg + "\n" + "".join(traceback.format_exception(*record.exc_info))
        return {
            "type": "log",
            "level": record.levelname,
            "name": record.name,
            "area": _area_for(record.name),
            "msg": msg,
            "time": record.created,
        }

    def _accept(self, record: logging.LogRecord) -> bool:
        # Always surface warnings+; otherwise only SpyDE's own loggers, so the
        # panel isn't flooded by third-party INFO/DEBUG at low levels.
        if record.levelno >= logging.WARNING:
            return True
        return record.name == "spyde" or record.name.startswith("spyde.")

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._guard, "active", False):
            return
        if not self._accept(record):
            return
        try:
            self._guard.active = True
            entry = self._entry(record)
            self.buffer.append(entry)
            from spyde.backend.ipc import emit as _emit
            _emit(entry)
        except Exception:
            # Logging must never crash the app, and handleError() would print to
            # stderr (noise); swallow deliberately here — this IS the log path.
            pass
        finally:
            self._guard.active = False


def install(level="INFO") -> IPCLogHandler:
    """Attach the singleton IPC log handler to the root logger (idempotent).

    Sets the root level so records at the chosen verbosity actually reach the
    handler. Call once at backend startup, after ``redirect_stray_stdout``.
    """
    global _handler
    if _handler is not None:
        return _handler
    lv = _coerce_level(level)
    _handler = IPCLogHandler(level=lv)
    root = logging.getLogger()
    root.addHandler(_handler)
    # Root defaults to WARNING; lift it so INFO/DEBUG records propagate to us.
    if root.level == logging.NOTSET or root.level > lv:
        root.setLevel(lv)
    return _handler


def set_level(level) -> int:
    """Set the live verbosity (root logger + handler). Returns the numeric level."""
    lv = _coerce_level(level)
    if _handler is not None:
        _handler.setLevel(lv)
    logging.getLogger().setLevel(lv)
    return lv


def emit_backfill() -> None:
    """Re-emit buffered records so a freshly-opened panel shows recent history."""
    if _handler is None:
        return
    from spyde.backend.ipc import emit
    emit({"type": "log_backfill", "entries": list(_handler.buffer)})


# ── staged handler (session.py dispatch: fn(session, plot, payload)) ───────────

def set_log_level(session, plot, payload) -> None:
    """Frontend level switcher → set verbosity, backfill history, confirm level."""
    lv = set_level(payload.get("level", "INFO"))
    emit_backfill()
    from spyde.backend.ipc import emit
    emit({"type": "log_level", "level": logging.getLevelName(lv)})

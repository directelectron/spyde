"""
ipc.py — stdin/stdout JSON-lines protocol between the Python backend and Electron.

All messages from Python to Electron are prefixed with "PLOTAPP:" and contain
JSON on a single line, matching the protocol anyplotlib._electron already uses.

Messages from Electron arrive on stdin as JSON lines (no prefix).

Usage
-----
    from spyde.backend.ipc import emit, IPC

    # send a message to Electron
    emit({"type": "status", "text": "Dask ready"})

    # read messages (async, called from the asyncio event loop)
    async for msg in IPC.messages():
        handle(msg)
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from typing import Any

# Logging goes to stderr (default), never the PLOTAPP stdout protocol channel.
log = logging.getLogger(__name__)

_stdout_lock = threading.Lock()

# Capture the *real* stdout at import time. This is the dedicated protocol
# channel; emit() always writes here even after stray prints are redirected.
_PROTOCOL_OUT = sys.stdout


def _write_line(line: str) -> None:
    """Write one protocol line, flushed immediately (thread-safe)."""
    with _stdout_lock:
        _PROTOCOL_OUT.write(line)
        _PROTOCOL_OUT.flush()


def redirect_stray_stdout() -> None:
    """Send all `print()` output to stderr so it can never interleave with the
    PLOTAPP protocol on stdout, while keeping BOTH protocol emitters — spyde's
    own ``emit`` and anyplotlib's ``_electron.emit`` — pointed at the real
    stdout protocol channel.

    anyplotlib._electron.emit writes to ``sys.stdout`` dynamically, so simply
    redirecting sys.stdout would send its state_update/event_json messages to
    stderr (where the runner never parses them). We therefore monkeypatch
    anyplotlib's emit to share spyde's locked protocol channel, then redirect
    sys.stdout so stray prints go to stderr.

    Call once at backend startup, after _PROTOCOL_OUT is captured.
    """
    try:
        import anyplotlib._electron as _ael

        def _shared_emit(obj: dict) -> None:
            _write_line("PLOTAPP:" + json.dumps(obj, default=str) + "\n")

        _ael.emit = _shared_emit
    except Exception as e:
        log.debug("redirecting anyplotlib emit to shared protocol channel failed: %s", e)

    sys.stdout = sys.stderr


def emit(obj: dict[str, Any]) -> None:
    """Write a PLOTAPP: message to the protocol channel (thread-safe, flushed
    immediately from any thread)."""
    _write_line("PLOTAPP:" + json.dumps(obj, default=str) + "\n")


def emit_status(text: str) -> None:
    emit({"type": "status", "text": text})


def emit_error(text: str) -> None:
    emit({"type": "error", "text": text})


def emit_progress(done: int, total: int, label: str = "") -> None:
    emit({"type": "progress", "done": done, "total": total, "label": label})


async def read_messages(loop: asyncio.AbstractEventLoop | None = None):
    """
    Async generator that yields parsed JSON dicts from stdin.
    Each line on stdin must be a valid JSON object.
    Exits when stdin closes.

    Implementation note (cross-platform): stdin is read on a dedicated daemon
    thread that pushes raw lines into an ``asyncio.Queue``, rather than via
    ``loop.connect_read_pipe(sys.stdin)`` — the latter raises
    ``OSError: [WinError 6] The handle is invalid`` under Windows'
    ``ProactorEventLoop`` (it can't register a console/pipe stdin handle with the
    IOCP), which silently broke every Electron→backend message on Windows. A
    blocking ``readline`` on a thread works identically on Windows, macOS, Linux.

    Encoding note: we read the BINARY stream (``sys.stdin.buffer``) and decode
    each line as UTF-8 ourselves. ``sys.stdin`` (the text layer) decodes with the
    platform-default encoding — cp1252 on Windows — so a UTF-8 payload from
    Electron carrying any non-ASCII character (e.g. ``εxx`` strain labels, the
    ``Å`` unit) was mojibake'd: the bytes ``0xCE 0xB5`` (UTF-8 ``ε``) decoded to
    ``Îµ`` and downstream string matches (tile_views' label lookup) silently
    failed. Reading bytes + explicit UTF-8 decode is correct on every platform.
    """
    if loop is None:
        loop = asyncio.get_event_loop()

    q: asyncio.Queue[str | None] = asyncio.Queue()
    # Prefer the raw byte stream; fall back to the text stream (some test
    # harnesses replace sys.stdin with a StringIO that has no .buffer).
    stream = getattr(sys.stdin, "buffer", None)
    binary = stream is not None
    if not binary:
        stream = sys.stdin

    def _pump() -> None:
        try:
            while True:
                raw = stream.readline()
                if not raw:   # EOF — pipe closed by Electron
                    break
                if binary:
                    raw = raw.decode("utf-8", errors="replace")
                loop.call_soon_threadsafe(q.put_nowait, raw)
        except Exception as e:
            log.debug("stdin pump stopped: %s", e)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=_pump, daemon=True, name="spyde-stdin-pump").start()

    while True:
        raw = await q.get()
        if raw is None:   # EOF sentinel
            break
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            log.debug("skipping non-JSON line from frontend: %r", line[:200])

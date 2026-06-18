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
import sys
import threading
from typing import Any

_stdout_lock = threading.Lock()

# Capture the *real* stdout at import time. This is the dedicated protocol
# channel; emit() always writes here even after stray prints are redirected.
_PROTOCOL_OUT = sys.stdout


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
            line = "PLOTAPP:" + json.dumps(obj, default=str) + "\n"
            with _stdout_lock:
                _PROTOCOL_OUT.write(line)
                _PROTOCOL_OUT.flush()

        _ael.emit = _shared_emit
    except Exception:
        pass

    sys.stdout = sys.stderr


def emit(obj: dict[str, Any]) -> None:
    """Write a PLOTAPP: message to the protocol channel (thread-safe)."""
    line = "PLOTAPP:" + json.dumps(obj, default=str) + "\n"
    with _stdout_lock:
        _PROTOCOL_OUT.write(line)
        _PROTOCOL_OUT.flush()


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
    """
    if loop is None:
        loop = asyncio.get_event_loop()

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line_bytes = await reader.readline()
        if not line_bytes:
            break
        line = line_bytes.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            pass

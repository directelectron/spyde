"""
app.py — asyncio event loop replacing QApplication.exec().

Launched by spyde/__main__.py.  Reads JSON messages from stdin (sent by
Electron), routes them to Session, and lets Session push replies via stdout.
"""
from __future__ import annotations

import asyncio
import os
import sys


async def _main() -> None:
    from spyde.backend.ipc import read_messages, emit, emit_status
    from spyde.backend.session import Session

    cpu_count = os.cpu_count() or 4
    if cpu_count < 4:
        workers, threads = 1, 1
    elif cpu_count <= 16:
        workers = max(1, cpu_count // 2 - 1)
        threads = 2
    else:
        workers = max(1, cpu_count // 4 - 1)
        threads = 4

    session = Session(n_workers=workers, threads_per_worker=threads)
    session.start_dask()

    emit({"type": "ready"})

    loop = asyncio.get_event_loop()
    async for msg in read_messages(loop):
        msg_type = msg.get("type")
        try:
            if msg_type == "action":
                session.dispatch_action(msg)
            elif msg_type == "quit":
                break
            else:
                print(f"[backend] unknown message type: {msg_type}")
        except Exception as e:
            from spyde.backend.ipc import emit_error
            emit_error(str(e))

    session.shutdown()


def run() -> None:
    asyncio.run(_main())

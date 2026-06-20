"""
app.py — asyncio event loop replacing QApplication.exec().

Launched by spyde/__main__.py.  Reads JSON messages from stdin (sent by
Electron), routes them to Session, and lets Session push replies via stdout.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

log = logging.getLogger(__name__)


def _prewarm_anyplotlib() -> None:
    """Warm anyplotlib's one-time costs off the critical path: the first
    `subplots`+`imshow` (~120 ms of module/JIT init) and the shared-ESM bundle
    write. Runs in a daemon thread so it never blocks startup; the first real
    dataset-load figures are then fast."""
    def _warm() -> None:
        try:
            import numpy as np
            import anyplotlib as apl
            from anyplotlib.embed import build_standalone_html
            fig, ax = apl.subplots(1, 1)
            a = ax[0][0] if isinstance(ax, list) else ax
            a.imshow(np.zeros((4, 4), dtype="float32"))
            build_standalone_html(fig, fig_id="prewarm", resizable=False)
            try:
                from spyde.drawing.plots.plot import _shared_esm_url
                esm = str(getattr(fig, "_esm", "") or "")
                if esm:
                    _shared_esm_url(esm)   # write the shared bundle to disk now
            except Exception as e:
                log.debug("prewarm shared-ESM write failed: %s", e)
        except Exception as e:
            log.debug("anyplotlib prewarm failed: %s", e)
    import threading
    threading.Thread(target=_warm, daemon=True, name="anyplotlib-prewarm").start()


async def _main() -> None:
    from spyde.backend.ipc import read_messages, emit, emit_status, redirect_stray_stdout
    from spyde.backend.session import Session

    # Keep stdout exclusively for the PLOTAPP protocol; stray prints → stderr.
    redirect_stray_stdout()

    # Stream logging records to the Electron app-log panel (level switchable at
    # runtime from the frontend). Installed after stdout is the protocol channel.
    from spyde.backend.log_stream import install as _install_log_stream
    _install_log_stream(level="INFO")

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

    # Tests (and headless smoke runs) skip the heavy Dask cluster.
    if os.environ.get("SPYDE_NO_DASK") != "1":
        session.start_dask()

    # Prewarm: the FIRST anyplotlib figure pays a one-time ~120 ms cost (module
    # init + first imshow) and the shared-ESM bundle write. Do it off-thread at
    # startup so the first dataset-load windows aren't slow. (Warm windows are
    # ~50 ms; this moves the cold cost off the user's critical path.)
    _prewarm_anyplotlib()

    emit({"type": "ready"})

    loop = asyncio.get_event_loop()
    async for msg in read_messages(loop):
        msg_type = msg.get("type")
        try:
            if msg_type == "action":
                session.dispatch_action(msg)
            elif msg_type == "figure_event":
                _dispatch_figure_event(msg)
            elif msg_type == "resize":
                _resize_figure(msg)
            elif msg_type == "quit":
                break
            else:
                log.warning("[backend] unknown message type: %s", msg_type)
        except Exception as e:
            from spyde.backend.ipc import emit_error
            emit_error(str(e))

    session.shutdown()


def _dispatch_figure_event(msg: dict) -> None:
    """Forward a frontend interaction event to the anyplotlib figure."""
    fig_id = msg.get("fig_id")
    event_json = msg.get("event_json")
    if fig_id is None or event_json is None:
        return
    import anyplotlib._electron as _ael
    _ael.dispatch_event(fig_id, event_json)


def _resize_figure(msg: dict) -> None:
    """Apply an MDI subwindow resize to the anyplotlib figure layout."""
    fig_id = msg.get("fig_id")
    if fig_id is None:
        return
    import anyplotlib._electron as _ael
    _ael.resize_figure(fig_id, int(msg.get("width", 600)), int(msg.get("height", 400)))


def run() -> None:
    asyncio.run(_main())

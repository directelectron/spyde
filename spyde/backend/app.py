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


def _prewarm_io() -> None:
    """Warm the file-I/O + diffraction stacks off the critical path.

    The FIRST ``hs.load`` of a session lazily imports the RosettaSciIO readers +
    format machinery (~6 s here, longer on a cold disk), and the first
    ``_add_signal`` pulls in pyxem (~2 s). Without prewarming, the first file the
    user opens pays all of that on the load thread — the window sits on
    "Reading …" for many seconds and looks hung (the dialog/windows only appear
    once those imports finish). Do it once at startup in a daemon thread so the
    first real open is instant. Idempotent: Python caches the modules, so the
    user's later ``hs.load`` just reuses them."""
    def _warm() -> None:
        import tempfile, os as _os
        try:
            import numpy as np
            import hyperspy.api as hs
            # A tiny real round-trip forces the hspy/zspy reader import path that
            # the first user load would otherwise pay for.
            tmp = _os.path.join(tempfile.gettempdir(), "spyde_prewarm.hspy")
            try:
                hs.signals.Signal2D(np.zeros((2, 2), dtype="float32")).save(tmp, overwrite=True)
                hs.load(tmp, lazy=True)
            finally:
                try:
                    _os.remove(tmp)
                except OSError:
                    pass
        except Exception as e:
            log.debug("io prewarm (hyperspy) failed: %s", e)
        try:
            import pyxem  # noqa: F401  — first import is ~2 s; warm it now.
        except Exception as e:
            log.debug("io prewarm (pyxem) failed: %s", e)
    import threading
    threading.Thread(target=_warm, daemon=True, name="io-prewarm").start()


async def _main() -> None:
    from spyde.backend.ipc import read_messages, emit, emit_status, redirect_stray_stdout
    from spyde.backend.session import Session

    # Keep stdout exclusively for the PLOTAPP protocol; stray prints → stderr.
    redirect_stray_stdout()

    # Guarantee Dask workers die with this process. The backend is an Electron
    # subprocess; if it is force-killed / crashes / Electron dies, the normal
    # shutdown() never runs and the cluster's worker+nanny grandchildren orphan
    # (every run leaked ~n_workers idle python.exe). A Windows kill-on-close Job
    # Object makes the OS reap the whole process tree no matter how we exit.
    # Installed BEFORE the cluster starts so workers inherit the job. No-op /
    # best-effort off Windows — the graceful shutdown() path below still runs.
    try:
        from spyde.backend.process_guard import install_kill_on_close
        install_kill_on_close()
    except Exception as e:
        log.debug("process guard install failed: %s", e)

    # Stream logging records to the Electron app-log panel (level switchable at
    # runtime from the frontend). Installed after stdout is the protocol channel.
    # SPYDE_LOG_LEVEL overrides the initial level (used by E2E tests / debugging).
    from spyde.backend.log_stream import install as _install_log_stream
    _init_level = os.environ.get("SPYDE_LOG_LEVEL", "INFO")
    _install_log_stream(level=_init_level)

    # When a level is forced via env (tests), also tee logs to STDERR so a
    # parent process (Playwright) can capture the [REDRAW]/NAV-DEBUG trace —
    # stderr is NOT the PLOTAPP protocol channel (stdout is). No-op normally.
    if "SPYDE_LOG_LEVEL" in os.environ:
        _h = logging.StreamHandler(sys.stderr)
        _h.setLevel(getattr(logging, _init_level.upper(), logging.INFO))
        _h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(_h)
        logging.getLogger().setLevel(getattr(logging, _init_level.upper(), logging.INFO))

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
    else:
        session.skip_dask()    # open the dask-ready gate; no cluster will start

    # Prewarm: the FIRST anyplotlib figure pays a one-time ~120 ms cost (module
    # init + first imshow) and the shared-ESM bundle write. Do it off-thread at
    # startup so the first dataset-load windows aren't slow. (Warm windows are
    # ~50 ms; this moves the cold cost off the user's critical path.)
    _prewarm_anyplotlib()
    # Warm the file-reader + pyxem imports too, so the FIRST file open isn't slow
    # (the "stuck on Reading…" until a second load — really first-call import lag).
    _prewarm_io()

    emit({"type": "ready"})

    loop = asyncio.get_event_loop()
    # Let the plot poller marshal result-applies onto this (main) thread.
    session.set_main_loop(loop)
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

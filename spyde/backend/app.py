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
            from spyde.backend.heavy_imports import ensure_heavy_imports
            ensure_heavy_imports()   # single-flight (races the load threads)
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


def _compute_worker_plan(cpu_count: int, fraction: float | None = None) -> tuple[int, int]:
    """Cluster size (n_workers, threads_per_worker) from the machine's logical
    cores.

    The compute budget is ``fraction`` of the logical cores — default **0.75**,
    deliberately NOT all of them (user feedback 2026-07-16: a batch saturating
    100% of cores stuttered the frontend even with the workers at below-normal
    priority; leaving ~a quarter of the machine for the Electron UI + the
    backend event loop + the painter/dispatcher threads keeps it fluid).
    Override with SPYDE_COMPUTE_FRACTION (clamped to 0.1–1.0) for throughput
    A/B runs — the priority throttle (_lower_worker_priority) has its own
    SPYDE_WORKER_PRIORITY opt-out.
    """
    if fraction is None:
        try:
            fraction = float(os.environ.get("SPYDE_COMPUTE_FRACTION", "0.75"))
        except ValueError:
            fraction = 0.75
    fraction = min(1.0, max(0.1, fraction))
    budget = max(1, int(cpu_count * fraction))
    if cpu_count < 4:
        return 1, 1
    threads = 2 if cpu_count <= 16 else 4
    return max(1, budget // threads), threads


async def _main() -> None:
    from spyde.backend.ipc import read_messages, emit, emit_status, redirect_stray_stdout
    from spyde.backend.session import Session

    # Keep stdout exclusively for the PLOTAPP protocol; stray prints → stderr.
    redirect_stray_stdout()

    # FIRST: real timer interrupts. Windows throttles timers for this hidden
    # Electron child, freezing every timer-driven wait in the process (dask
    # task delivery, poll loops) until I/O arrives — the "computes only finish
    # when you click" bug. See process_guard.unthrottle_windows_timers.
    try:
        from spyde.backend.process_guard import unthrottle_windows_timers
        unthrottle_windows_timers()
    except Exception as e:
        log.warning("timer unthrottle failed: %s", e)

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

    # Persisted in-app compute settings (max RAM / CPU use / GPU feeders — the
    # DaskMonitor popover) land in the environment BEFORE the worker plan and
    # cluster build; an explicitly-set env var still wins.
    from spyde.backend.compute_config import apply_persisted_compute_env
    apply_persisted_compute_env()
    workers, threads = _compute_worker_plan(os.cpu_count() or 4)

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
    # Warm torch + the CUDA context: the GPU tile backend (first large signal
    # frame) otherwise pays the ~3 s import ON THE PAINTER THREAD — much worse
    # while the navigator fill saturates the disk (black signal panel).
    from spyde.backend.heavy_imports import prewarm_torch_cuda
    prewarm_torch_cuda()

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
            elif "command" in msg:
                # Math-console commands arrive as flat {"command": "console_*", …}
                # (the console IPC contract), not wrapped in a "type":"action"
                # envelope. Route them straight to the session's console handler.
                _dispatch_console(session, msg)
            else:
                log.warning("[backend] unknown message type: %s", msg_type)
        except Exception as e:
            from spyde.backend.ipc import emit_error
            emit_error(str(e))

    session.shutdown()


def _dispatch_console(session, msg: dict) -> None:
    """Route a flat console command ({"command": "console_exec"|"console_create_window"|
    "console_complete", …}) to the session's console engine. The engine queues the
    work on its own daemon thread, so this returns immediately (the asyncio loop is
    never blocked on user code)."""
    command = msg.get("command")
    if command == "console_exec":
        session.console.submit_exec(str(msg.get("code", "")), int(msg.get("exec_id", 0)))
    elif command == "console_create_window":
        session.console.create_window(str(msg.get("name", "")))
    elif command == "console_preview":
        session.console.submit_preview(
            str(msg.get("code", "")), int(msg.get("preview_id", 0)),
            bool(msg.get("auto", True)),
        )
    elif command == "console_complete":
        session.console.submit_complete(
            str(msg.get("prefix", "")), int(msg.get("complete_id", 0))
        )
    elif command == "console_remove_var":
        session.console.remove_var(str(msg.get("name", "")))
    else:
        log.warning("[backend] unknown console command: %s", command)


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

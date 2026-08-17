"""
app.py — the asyncio backend loop, minus anything app-specific.

Reads JSON messages from stdin (Electron writes them), routes each to the
session, and lets the session push replies through ``de_shell.ipc.emit``. This
is the process's main thread: everything that touches a figure has to be
marshalled back onto it (``SessionBase._dispatch_to_main``).

Usage from an app's ``__main__``::

    from de_shell.app import run
    run(build_session=lambda: MySession(), app_packages=("de_groundcrew",))

Message routing is fixed for the three the shell owns — ``action``,
``figure_event``, ``resize``, plus ``quit`` and the ``tick`` no-op — and open
for everything else: pass ``on_message`` to handle app-specific envelopes (SpyDE
routes its flat ``{"command": "console_*"}`` messages that way).

The `tick` no-op matters more than it looks. Windows throttles timer delivery to
a hidden child process so hard that its waits can freeze until process I/O
arrives, so Electron writes a 0.5 Hz tick purely to keep this process
schedulable. Dropping it as "an unknown message" would log a warning twice a
second forever.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Callable, Iterable

from de_shell import ipc, log_stream

log = logging.getLogger(__name__)


def _dispatch_figure_event(msg: dict) -> None:
    """Forward a frontend interaction event to the anyplotlib figure."""
    fig_id = msg.get("fig_id")
    event_json = msg.get("event_json")
    if fig_id is None or event_json is None:
        return
    import anyplotlib._electron as _ael
    _ael.dispatch_event(fig_id, event_json)


def _resize_figure(msg: dict) -> None:
    """Apply a window resize to the anyplotlib figure layout."""
    fig_id = msg.get("fig_id")
    if fig_id is None:
        return
    import anyplotlib._electron as _ael
    w, h = int(msg.get("width", 600)), int(msg.get("height", 400))
    log.debug("resize figure %s -> %dx%d", fig_id, w, h)
    _ael.resize_figure(fig_id, w, h)


def _install_logging(app_packages: Iterable[str], log_level_env: str) -> None:
    """Stream logs to the frontend panel, and tee to stderr when asked.

    The tee is what makes a backend failure visible to Playwright: ``emit`` goes
    down the PLOTAPP stdout channel, which the Electron main process consumes,
    so without this a backend that dies mid-test dies silently.
    """
    level = os.environ.get(log_level_env)
    if level:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))
    log_stream.install(level=level or "INFO")


async def _main(
    build_session: Callable[[], object],
    app_packages: Iterable[str],
    log_level_env: str,
    on_message: Callable[[object, dict], bool] | None,
    on_ready: Callable[[object], None] | None,
) -> None:
    ipc.redirect_stray_stdout()
    _install_logging(app_packages, log_level_env)

    session = build_session()

    if on_ready is not None:
        on_ready(session)

    ipc.emit({"type": "ready"})

    loop = asyncio.get_event_loop()
    # Let background workers marshal their result-applies onto this thread.
    session.set_main_loop(loop)

    async for msg in ipc.read_messages(loop):
        msg_type = msg.get("type")
        try:
            if msg_type == "action":
                session.dispatch_action(msg)
            elif msg_type == "figure_event":
                _dispatch_figure_event(msg)
            elif msg_type == "resize":
                _resize_figure(msg)
            elif msg_type == "tick":
                pass          # see the module docstring — deliberately silent
            elif msg_type == "quit":
                break
            elif on_message is not None and on_message(session, msg):
                pass          # the app claimed it
            else:
                log.warning("[backend] unknown message type: %s", msg_type)
        except Exception as e:
            log.exception("handling %s failed", msg_type)
            ipc.emit_error(str(e))

    session.shutdown()


def run(
    build_session: Callable[[], object],
    *,
    app_packages: Iterable[str] = (),
    log_level_env: str = "DE_LOG_LEVEL",
    on_message: Callable[[object, dict], bool] | None = None,
    on_ready: Callable[[object], None] | None = None,
) -> None:
    """Run the backend loop until the frontend says quit.

    Parameters
    ----------
    build_session
        Builds the app's Session. Called once, on the main thread, before the
        loop starts.
    app_packages
        The app's own top-level package names, for log routing.
    log_level_env
        Environment variable holding the initial log level. Per-app so two
        installed apps can be made verbose independently.
    on_message
        Handles a message the shell does not know. Return True if claimed.
    on_ready
        Runs after the session is built and before ``ready`` is emitted —
        the place for prewarming and starting background services.
    """
    asyncio.run(_main(build_session, app_packages, log_level_env, on_message, on_ready))

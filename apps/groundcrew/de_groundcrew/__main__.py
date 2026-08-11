"""
__main__.py — the backend entry point Electron spawns (`python -m de_groundcrew`).

Compare with SpyDE's: the whole file is a session factory and a call into the
shell's loop, because everything else — stdin/stdout framing, log streaming,
figure event routing, the tick no-op, shutdown — belongs to `de_shell.app`.
"""
from __future__ import annotations

import multiprocessing
import os


def _settings_dir() -> str:
    """Where settings.json lives.

    `GROUNDCREW_SETTINGS_DIR` is the e2e escape hatch (Electron refuses to launch
    with HOME redirected, so a test cannot just move the home directory). The
    per-app name matters: pointing this at SpyDE's `~/.spyde` would have two
    apps writing each other's preferences.
    """
    return os.environ.get("GROUNDCREW_SETTINGS_DIR") or os.path.join(
        os.path.expanduser("~"), ".de-groundcrew")


def main() -> None:
    # Same reason SpyDE calls it: a frozen build re-executes this module in each
    # spawned child, and without freeze_support that forks the whole app again.
    multiprocessing.freeze_support()

    from de_shell.app import run
    from de_groundcrew.session import GroundCrewSession

    run(
        build_session=lambda: GroundCrewSession(settings_dir=_settings_dir()),
        app_packages=("de_groundcrew",),
        log_level_env="GROUNDCREW_LOG_LEVEL",
        # Open the viewer and start free-running BEFORE `ready` is emitted, so
        # the first frame is already on its way when the window appears.
        on_ready=lambda session: session.start(),
    )


if __name__ == "__main__":
    main()

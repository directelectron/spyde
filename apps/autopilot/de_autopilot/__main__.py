"""
__main__.py — the backend entry point Electron spawns (`python -m de_autopilot`).

As short as Ground Crew's, for the same reason: stdin/stdout framing, log
streaming, figure event routing, the tick no-op and shutdown all belong to
`de_shell.app`.
"""
from __future__ import annotations

import multiprocessing
import os


def _settings_dir() -> str:
    """Where settings.json lives. `AUTOPILOT_SETTINGS_DIR` is the e2e escape
    hatch; the per-app default keeps this app out of the others' preferences."""
    return os.environ.get("AUTOPILOT_SETTINGS_DIR") or os.path.join(
        os.path.expanduser("~"), ".de-autopilot")


def main() -> None:
    # A frozen build re-executes this module in each spawned child; without
    # freeze_support that forks the whole app again.
    multiprocessing.freeze_support()

    from de_shell.app import run
    from de_autopilot.session import AutopilotSession

    run(
        build_session=lambda: AutopilotSession(settings_dir=_settings_dir()),
        app_packages=("de_autopilot",),
        log_level_env="AUTOPILOT_LOG_LEVEL",
        # Open the viewer and publish the recipe before `ready`, so the UI has
        # something to show the moment the window appears. Unlike Ground Crew
        # this does NOT start acquiring — a recipe runs when asked.
        on_ready=lambda session: session.start(),
    )


if __name__ == "__main__":
    main()

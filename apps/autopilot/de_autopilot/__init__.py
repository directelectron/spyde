"""
de_autopilot — Direct Electron Autopilot.

Unattended acquisition: the operator describes a sequence of steps once and the
app executes it, reporting progress and keeping the last acquired frame on
screen. Where Ground Crew is a set of manual controls, this is a queue with a
run button — which is why it exists as a separate app rather than a Ground Crew
tab.

The desktop plumbing (asyncio loop, PLOTAPP IPC, log streaming, session,
figures) comes from `de_shell`. What lives here is the recipe model and its
runner.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Logger-name → log-panel area, registered with the shell at import. Without it
# the shell filters this app's sub-WARNING records out of the panel — it only
# lets its own and the app's DECLARED packages through below WARNING.
_LOG_AREA_RULES = (
    ("de_autopilot.recipe", "recipe"),
    ("de_autopilot.session", "session"),
    ("de_autopilot", "autopilot"),
)


def _register_log_areas() -> None:
    try:
        from de_shell.log_stream import register_area_rules
        register_area_rules(_LOG_AREA_RULES, verbose_packages=("de_autopilot",))
    except Exception as exc:  # never block import on a logging-config hiccup
        log.warning("Autopilot log-area registration skipped: %s", exc)


_register_log_areas()

__version__ = "0.1.0"

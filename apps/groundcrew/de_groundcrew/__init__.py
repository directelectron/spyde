"""
de_groundcrew — Direct Electron Ground Crew.

Camera and hardware control against the DE Server (Mission Control) SDK. The
Electron rewrite of the PySide6 app: fixed panes rather than SpyDE's MDI
workspace, and everything live and in memory — there is no dataset on disk to
be lazy about.

The desktop plumbing (asyncio loop, PLOTAPP IPC, log streaming, the window
registry, settings) comes from `de_shell`. What lives here is the camera.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Logger-name → log-panel area, registered with the shell at import (the same
# contract SpyDE uses; see de_shell.log_stream.register_area_rules). Without
# this, Ground Crew's own sub-WARNING records would be filtered out of the
# panel, because the shell only lets its own and the app's declared packages
# through below WARNING.
_LOG_AREA_RULES = (
    ("de_groundcrew.instrument", "camera"),
    ("de_groundcrew.tile", "camera"),
    ("de_groundcrew.session", "session"),
    ("de_groundcrew", "groundcrew"),
)


def _register_log_areas() -> None:
    try:
        from de_shell.log_stream import register_area_rules
        register_area_rules(_LOG_AREA_RULES, verbose_packages=("de_groundcrew",))
    except Exception as exc:  # never block import on a logging-config hiccup
        log.warning("Ground Crew log-area registration skipped: %s", exc)


_register_log_areas()

__version__ = "0.1.0"

"""
debug_flags.py — runtime-toggleable diagnostic switches.

These are cheap boolean flags that gate opt-in instrumentation (e.g. the per-frame
navigator update profile). Each can be seeded from an env var at import (so it can
be on from process start) AND toggled live from the UI via the ``set_debug_flag``
action — no restart needed. Keeping them in ONE module means a single source of
truth that both the read side (``update_functions``) and the paint side (``plot``)
read through ``nav_profile_on()``.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# Per-frame navigator update profile: one INFO line per move with the stage
# breakdown (read / dtype / prefetch / lod / levels / transport). Seed from the
# env so it can be on from startup; toggle live with set_debug_flag("nav_profile").
_nav_profile: bool = os.environ.get("SPYDE_NAV_PROFILE") == "1"


def nav_profile_on() -> bool:
    """True when per-frame update profiling is active. Read this each frame (it's
    a cheap module-global lookup) so a live toggle takes effect immediately."""
    return _nav_profile


def set_flag(name: str, value: bool) -> bool:
    """Set a debug flag by name. Returns the new value. Unknown names are ignored
    (return False) so a stale UI can't crash the backend."""
    global _nav_profile
    if name in ("nav_profile", "profile"):
        _nav_profile = bool(value)
        log.info("[debug] nav_profile = %s (per-frame update timing %s)",
                 _nav_profile, "ON" if _nav_profile else "off")
        return _nav_profile
    log.debug("[debug] unknown debug flag %r ignored", name)
    return False


def get_flags() -> dict:
    """Current flag states — for the UI to reflect the toggle on connect."""
    return {"nav_profile": _nav_profile}


def set_debug_flag(session, plot, payload) -> None:
    """Staged action: the UI's debug toggle → set a flag + echo the new state.

    Payload: ``{"name": "nav_profile", "value": true|false}``. Echoes
    ``{"type": "debug_flags", ...}`` so the button can reflect the state, and (when
    turning nav profiling ON) makes sure INFO records reach the Log panel so the
    profile lines are actually visible without touching the level dropdown."""
    name = payload.get("name", "nav_profile")
    value = bool(payload.get("value", False))
    set_flag(name, value)
    if name in ("nav_profile", "profile") and value:
        # The [NAV/PAINT-PROFILE] lines log at INFO; ensure the handler forwards
        # INFO (the default is INFO, but a user may have raised it to WARNING).
        try:
            from spyde.backend.log_stream import set_level
            import logging
            if logging.getLogger().getEffectiveLevel() > logging.INFO:
                set_level("INFO")
        except Exception as e:
            log.debug("raising log level to INFO for profiling failed: %s", e)
    from spyde.backend.ipc import emit
    emit({"type": "debug_flags", **get_flags()})

"""
de_shell — the Python half of the DE desktop-app shell.

The substrate three applications share: SpyDE (offline EM analysis),
de-groundcrew (live camera/hardware control) and de-autopilot (automated
acquisition). It answers "how do I be a desktop app with a Python brain and
pictures in it?" — the asyncio stdin/stdout loop, the PLOTAPP IPC protocol, log
streaming, the window/figure registry, the action + staged-wizard framework, and
the anyplotlib plotting wrapper.

It answers nothing about what the data IS. No HyperSpy, no Dask, no
RosettaSciIO, no pyxem — de-groundcrew and de-autopilot are live, in-memory
applications and must not acquire those dependencies transitively. That
constraint is what fixes the boundary; see ARCHITECTURE-SPLIT.md.
"""
from __future__ import annotations

__all__ = ["ipc", "log_stream", "process_guard", "debug_flags", "compute"]

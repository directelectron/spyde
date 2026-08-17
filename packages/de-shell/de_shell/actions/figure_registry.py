"""
figure_registry.py — per-window keep-alive for bare anyplotlib figures.

Result windows that are NOT registered ``Plot``s emit raw ``figure`` messages
whose Python-side figure objects must be kept referenced, or their widget
callbacks are garbage-collected while the window is still open. Historically
each module kept its own append-only ``_ALIVE`` list, which leaked every figure
for the process lifetime.

This registry keys the references by ``window_id`` and is evicted from the
session's ``_forget_window``, so a figure lives exactly as long as its window.

Apps hang their own per-window state off the same eviction via
:func:`register_evictor`, rather than this module reaching into them — which is
what it used to do (a hardcoded import of SpyDE's ``actions.views``).
"""
from __future__ import annotations

from typing import Any, Callable

_FIGS: dict[int, list[Any]] = {}

#: App callbacks run when a window is forgotten. See `register_evictor`.
_EVICTORS: list[Callable[[int], None]] = []


def register_evictor(fn: Callable[[int], None]) -> None:
    """Register ``fn(window_id)`` to run whenever a window is forgotten.

    For app state keyed by window id that must die with the window — SpyDE's
    per-window chip-view arrays, for instance. Registering the same function
    twice is a no-op, so a module can call this at import without guarding.
    """
    if fn not in _EVICTORS:
        _EVICTORS.append(fn)


def keep_alive(window_id: int, fig: Any) -> None:
    """Keep *fig* referenced until *window_id*'s window is forgotten."""
    _FIGS.setdefault(int(window_id), []).append(fig)


def forget_window(window_id: int) -> None:
    """Drop every reference held for *window_id*, and run the app's evictors."""
    wid = int(window_id)
    _FIGS.pop(wid, None)
    for fn in _EVICTORS:
        try:
            fn(wid)
        except Exception:
            # Teardown must not fail: this runs while a window is going away,
            # and one app's bookkeeping error should not strand the rest.
            pass

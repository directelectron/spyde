"""
figure_registry.py — per-window keep-alive for bare anyplotlib figures.

Result windows that are NOT registered ``Plot``s (strain map, IPF 3-D/key/
density, tiled view comparisons) emit raw ``figure`` messages whose Python-side
figure objects must be kept referenced or their widget callbacks are
garbage-collected while the window is still open. Historically each module kept
its own append-only ``_ALIVE`` list, which leaked every figure for the process
lifetime.

This registry keys the references by ``window_id`` and is evicted from
``Session._forget_window``, so a figure lives exactly as long as its window.
"""
from __future__ import annotations

from typing import Any

_FIGS: dict[int, list[Any]] = {}


def keep_alive(window_id: int, fig: Any) -> None:
    """Keep *fig* referenced until *window_id*'s window is forgotten."""
    _FIGS.setdefault(int(window_id), []).append(fig)


def forget_window(window_id: int) -> None:
    """Drop every reference held for *window_id* (figures + view data)."""
    _FIGS.pop(int(window_id), None)
    # views.py keeps per-window chip-view arrays; evict those too. Lazy import —
    # views may never have been loaded in this session.
    try:
        from spyde.actions import views
        views._VIEW_DATA.pop(int(window_id), None)
    except Exception:
        pass

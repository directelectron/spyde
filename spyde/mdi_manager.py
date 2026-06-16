"""
mdi_manager.py — MDI window lifecycle coordinator.

The Qt QMdiArea implementation lives in _qt_main_legacy.py (Phase 4 reference).
This module provides the backend-facing MDI coordinator used by Session: it
tracks open plot windows, handles window creation requests, and emits Electron
IPC messages when windows are opened/closed/focused.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING
from uuid import uuid4

from psygnal import Signal

if TYPE_CHECKING:
    from spyde.backend.session import Session
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow
    from spyde.drawing.plots.multiplot_manager import MultiplotManager
    from spyde.signal_tree import BaseSignalTree


class MDIManager:
    """
    Owns MDI window lifecycle.

    In the Electron architecture, actual window chrome lives in the frontend.
    This class tracks open windows, assigns IDs, and emits IPC messages so
    Electron can render/close/tile SubWindows.
    """

    subwindow_activated = Signal(object)   # PlotWindow

    def __init__(self, session: "Session") -> None:
        self.session = session
        self.plot_subwindows: list["PlotWindow"] = []
        self.signal_trees: list["BaseSignalTree"] = []
        self._navigator_drag_payloads: dict[str, dict] = {}

    # ── Public interface ───────────────────────────────────────────────────────

    def add_plot_window(
        self,
        *,
        is_navigator: bool = False,
        plot_manager: "MultiplotManager | None" = None,
        signal_tree: "BaseSignalTree | None" = None,
    ) -> "PlotWindow":
        from spyde.drawing.plots.plot_window import PlotWindow
        from spyde.backend.ipc import emit

        window_id = self.session.next_window_id()
        pw = PlotWindow(
            is_navigator=is_navigator,
            session=self.session,
            signal_tree=signal_tree,
            plot_manager=plot_manager,
            window_id=window_id,
        )
        self.plot_subwindows.append(pw)

        emit({
            "type": "window_opened",
            "window_id": window_id,
            "is_navigator": is_navigator,
            "title": signal_tree.root.metadata.get_item("General.title", default="Signal")
                if signal_tree else "",
        })
        return pw

    def windows_for_tree(self, tree: "BaseSignalTree") -> list["PlotWindow"]:
        return [pw for pw in list(self.plot_subwindows)
                if getattr(pw, "signal_tree", None) is tree]

    def close_signal_tree(self, tree: "BaseSignalTree") -> None:
        if tree is None or getattr(tree, "_spyde_closing", False):
            return
        tree._spyde_closing = True
        try:
            for pw in self.windows_for_tree(tree):
                try:
                    pw._spyde_tree_teardown = True
                    pw.close_window()
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception(
                        "close_signal_tree: window close failed")
            try:
                tree.close()
            except Exception:
                pass
            if tree in self.signal_trees:
                self.signal_trees.remove(tree)
            if (getattr(self.session, "current_selected_signal_tree", None) is tree):
                self.session.current_selected_signal_tree = None
        finally:
            tree._spyde_closing = False

    def active_plot(self) -> "Plot | None":
        """Return the currently focused plot, or None."""
        # In Electron, focus is tracked by the frontend; the session receives
        # an "activate_window" IPC message that calls on_window_activated().
        return self._active_plot

    def active_plot_window(self) -> "PlotWindow | None":
        return self._active_pw

    _active_plot: "Plot | None" = None
    _active_pw: "PlotWindow | None" = None

    def on_window_activated(self, window_id: int) -> None:
        """Called by Session when Electron reports a window focus change."""
        pw = next(
            (p for p in self.plot_subwindows if getattr(p, "window_id", None) == window_id),
            None,
        )
        if pw is None:
            return
        self._active_pw = pw
        self._active_plot = pw.current_plot_item
        self.subwindow_activated.emit(pw)

    # ── Navigator drag payloads ────────────────────────────────────────────────

    def register_navigator_drag_payload(self, signal, nav_manager) -> str:
        token = uuid4().hex
        self._navigator_drag_payloads[token] = {
            "signal": signal,
            "nav_manager": nav_manager,
        }
        return token

    # ── Tile / organize ───────────────────────────────────────────────────────

    def tile_active_windows(self) -> None:
        """Send tile layout to Electron for the active tree's windows."""
        from spyde.backend.ipc import emit

        active_tree = (
            self._active_pw.signal_tree
            if self._active_pw is not None
            else None
        )
        shown = [
            pw for pw in self.plot_subwindows
            if pw.signal_tree is active_tree and getattr(pw, "visible", True)
        ]
        n = len(shown)
        if n == 0:
            return
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        tile_data = []
        for i, pw in enumerate(shown):
            row = i // cols
            col = i % cols
            tile_data.append({
                "window_id": pw.window_id,
                "col": col,
                "row": row,
                "cols": cols,
                "rows": rows,
            })
        emit({"type": "tile_windows", "layout": tile_data})

    def organize_active_windows(self) -> None:
        """Send organize-no-resize layout to Electron."""
        from spyde.backend.ipc import emit

        active_tree = (
            self._active_pw.signal_tree
            if self._active_pw is not None
            else None
        )
        shown = [
            pw for pw in self.plot_subwindows
            if pw.signal_tree is active_tree and getattr(pw, "visible", True)
        ]
        emit({"type": "organize_windows", "window_ids": [pw.window_id for pw in shown]})

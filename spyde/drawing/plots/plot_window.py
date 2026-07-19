"""
plot_window.py — PlotWindow: a logical container for one or more Plot objects.

Replaces the Qt QMdiSubWindow.  In the Electron architecture this maps to one
SubWindow component in the React MDI, which can contain multiple anyplotlib
iframes laid out side-by-side.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.multiplot_manager import MultiplotManager
    from spyde.drawing.selectors import BaseSelector
    from spyde.drawing.plots.plot_states import PlotState
    from spyde.signal_tree import BaseSignalTree
    from spyde.backend.session import Session

logger = logging.getLogger(__name__)


class PlotWindow:
    """
    Logical container for one or more Plot objects sharing the same SubWindow.

    Attributes
    ----------
    window_id : int
        Unique ID used for Electron IPC (SubWindow key in React MDI).
    plots : list[Plot]
        All Plot instances in this window.
    current_plot_item : Plot | None
        The currently focused Plot.
    signal_tree : BaseSignalTree | None
        The signal tree this window belongs to.
    is_navigator : bool
        Whether this window shows navigation (virtual image) plots.
    parent_selector : BaseSelector | None
        The selector that drives this window's child plots.
    """

    def __init__(
        self,
        is_navigator: bool = False,
        plot_manager: "MultiplotManager | None" = None,
        signal_tree: "BaseSignalTree | None" = None,
        session: "Session | None" = None,
        window_id: int = 0,
        parent_selector: "BaseSelector | None" = None,
    ):
        self.window_id = window_id
        self.is_navigator = is_navigator
        self.signal_tree = signal_tree
        self.session = session
        self.multiplot_manager = plot_manager
        self.parent_selector = parent_selector
        self.owner_plot_window: "PlotWindow | None" = None
        self.controlling_action = None
        self.visibility_gate = None

        self.plots: list[Plot] = []
        self._current_plot_item: "Plot | None" = None
        self._primary_plot_item: "Plot | None" = None

        # Electron state
        self.visible: bool = True
        self._spyde_closed: bool = False

    @property
    def current_plot_item(self) -> "Plot | None":
        return self._current_plot_item or (self.plots[0] if self.plots else None)

    # ── Plot management ────────────────────────────────────────────────────────

    def add_new_plot(self) -> "Plot":
        """Create a new Plot inside this window."""
        from spyde.drawing.plots.plot import Plot

        plot = Plot(
            signal_tree=self.signal_tree,
            is_navigator=self.is_navigator,
            multiplot_manager=self.multiplot_manager,
            plot_window=self,
            session=self.session,
        )
        plot.window_id = self.window_id
        self.plots.append(plot)
        if self._primary_plot_item is None:
            self._primary_plot_item = plot
        self._current_plot_item = plot
        return plot

    def insert_new_plot(self, drop_pos=None) -> "Plot":
        """Insert a plot at a drop position (Electron handles layout)."""
        return self.add_new_plot()

    # ── Visibility ─────────────────────────────────────────────────────────────

    def show(self) -> None:
        self.visible = True
        from spyde.backend.ipc import emit
        emit({"type": "window_visibility", "window_id": self.window_id, "visible": True})

    def hide(self) -> None:
        self.visible = False
        from spyde.backend.ipc import emit
        emit({"type": "window_visibility", "window_id": self.window_id, "visible": False})

    def isVisible(self) -> bool:
        return self.visible

    def raise_(self) -> None:
        from spyde.backend.ipc import emit
        emit({"type": "window_raise", "window_id": self.window_id})

    def lower(self) -> None:
        from spyde.backend.ipc import emit
        emit({"type": "window_lower", "window_id": self.window_id})

    # ── Geometry (Electron manages actual pixels) ──────────────────────────────

    def move(self, x: int, y: int) -> None:
        from spyde.backend.ipc import emit
        emit({"type": "window_move", "window_id": self.window_id, "x": x, "y": y})

    def resize(self, w: int, h: int) -> None:
        from spyde.backend.ipc import emit
        emit({"type": "window_resize", "window_id": self.window_id, "width": w, "height": h})

    def setGeometry(self, x: int, y: int, w: int, h: int) -> None:
        from spyde.backend.ipc import emit
        emit({"type": "window_geometry",
              "window_id": self.window_id, "x": x, "y": y, "width": w, "height": h})

    # ── Graphics effects ───────────────────────────────────────────────────────

    def setGraphicsEffect(self, effect) -> None:
        opacity = 1.0
        if effect is not None:
            try:
                opacity = effect.opacity()
            except Exception as e:
                logger.debug("reading graphics-effect opacity failed: %s", e)
        from spyde.backend.ipc import emit
        emit({"type": "window_opacity", "window_id": self.window_id, "opacity": opacity})

    # ── Close ──────────────────────────────────────────────────────────────────

    def close_window(self) -> None:
        if self._spyde_closed:
            return
        self._spyde_closed = True
        for plot in list(self.plots):
            try:
                plot.close()
            except Exception as e:
                logger.debug("closing child plot on window close failed: %s", e)
        self.plots.clear()
        if (self.session is not None
                and hasattr(self.session, "mdi_manager")):
            try:
                self.session.mdi_manager.plot_subwindows = [
                    pw for pw in self.session.mdi_manager.plot_subwindows
                    if pw is not self
                ]
            except Exception as e:
                logger.debug("removing window from mdi_manager registry failed: %s", e)
        from spyde.backend.ipc import emit
        emit({"type": "window_closed", "window_id": self.window_id})

    def close(self) -> None:
        self.close_window()

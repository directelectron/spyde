"""
plot_states.py — PlotState and MultiImageManager.

PlotState tracks the visualization state for a (Plot, Signal) pair:
contrast, colormap, selectors, and toolbar configuration.

In the Electron architecture, toolbars are React components in the frontend.
PlotState sends toolbar configuration via IPC when initialized or when the
state becomes active.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from hyperspy.signal import BaseSignal

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot


class _StubToolbar:
    """Shim so code that calls toolbar methods doesn't raise AttributeError."""

    def __init__(self):
        self.action_widgets: dict = {}

    def hide(self) -> None: pass
    def show(self) -> None: pass
    def raise_(self) -> None: pass
    def isVisible(self) -> bool: return False
    def num_actions(self) -> int: return 0
    def set_size(self) -> None: pass
    def close(self) -> None: pass
    def setAttribute(self, *a, **kw) -> None: pass
    def actions(self): return []
    def add_action(self, *a, **kw): return None, None
    def add_action_widget(self, *a, **kw): return None
    def register_action_plot_item(self, *a, **kw): return None
    def register_action_plot_window(self, *a, **kw): return None


class PlotState:
    """
    Visualization state for a (Plot, Signal) pair.

    Attributes that are still meaningful (contrast, colormap, selectors)
    are preserved.  Toolbar objects are _StubToolbar shims; real toolbar
    state lives in Electron.  IPC messages are emitted when a state becomes
    active/inactive so the frontend can show/hide the right toolbar buttons.
    """

    def __init__(
        self,
        signal: BaseSignal,
        plot: "Plot",
        dimensions: Optional[int] = None,
        dynamic: bool = True,
    ):
        self.current_signal = signal
        self.plot = plot

        self.min_percentile = 100
        self.max_percentile = 0
        self.min_level = 0.0
        self.max_level = 1.0
        self.colormap = "gray"
        self.gamma = 1.0
        self.dynamic = dynamic
        self.dimensions = (
            dimensions
            if dimensions is not None
            else signal.axes_manager.signal_dimension
        )

        # Selectors
        self.plot_selectors: list = []
        self.signal_tree_selectors: list = []
        self.plot_selectors_children: list = []
        self.signal_tree_selectors_children: list = []

        # Stub toolbars — real toolbars are Electron components (Phase 4)
        self.toolbar_top = _StubToolbar()
        self.toolbar_bottom = _StubToolbar()
        self.toolbar_left = _StubToolbar()
        self.toolbar_right = _StubToolbar()

        # Send toolbar config to Electron
        self._send_toolbar_config()

    def __repr__(self) -> str:
        return (
            f"<PlotState signal={self.current_signal}, "
            f"dimensions={self.dimensions}, dynamic={self.dynamic}>"
        )

    # ── Toolbar config ─────────────────────────────────────────────────────────

    def _send_toolbar_config(self) -> None:
        """Serialize TOOLBAR_ACTIONS for this state and emit to Electron."""
        try:
            from spyde.drawing.toolbars.plot_control_toolbar import (
                get_toolbar_config_for_plot,
            )
            config = get_toolbar_config_for_plot(self)
        except Exception:
            config = []

        window_id = getattr(getattr(self.plot, "plot_window", None), "window_id", None)
        if window_id is None:
            return
        try:
            from spyde.backend.ipc import emit
            emit({
                "type": "toolbar_config",
                "window_id": window_id,
                "plot_id": id(self.plot),
                "toolbar_actions": config,
            })
        except Exception as e:
            log.debug("sending toolbar config failed: %s", e)

    # ── Visibility ─────────────────────────────────────────────────────────────

    def show_toolbars(self) -> None:
        window_id = getattr(getattr(self.plot, "plot_window", None), "window_id", None)
        if window_id is None:
            return
        try:
            from spyde.backend.ipc import emit
            emit({"type": "toolbars_show", "window_id": window_id, "plot_id": id(self.plot)})
        except Exception as e:
            log.debug("emitting toolbars_show failed: %s", e)

    def hide_toolbars(self) -> None:
        window_id = getattr(getattr(self.plot, "plot_window", None), "window_id", None)
        if window_id is None:
            return
        try:
            from spyde.backend.ipc import emit
            emit({"type": "toolbars_hide", "window_id": window_id, "plot_id": id(self.plot)})
        except Exception as e:
            log.debug("emitting toolbars_hide failed: %s", e)

    def update_toolbars(self) -> None:
        self._send_toolbar_config()

    def rebuild_toolbars(self) -> None:
        self._send_toolbar_config()
        self.show_toolbars()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.hide_toolbars()
        self.toolbar_top = None
        self.toolbar_bottom = None
        self.toolbar_left = None
        self.toolbar_right = None


class MultiImageManager:
    """
    Manages multiple images within a single plotting context (grid / overlay).

    Represents virtual images, channels, or multi-spectral channels that can
    be toggled or overlaid in the same Axes.
    """

    def __init__(self, plot_states: List["PlotState"], plot: "Plot"):
        self.plot_states = plot_states
        self.plot = plot

    def add_plot_state(self, plot_state: "PlotState") -> None:
        if plot_state not in self.plot_states:
            self.plot_states.append(plot_state)

    def remove_plot_state(self, plot_state: "PlotState") -> None:
        if plot_state in self.plot_states:
            self.plot_states.remove(plot_state)

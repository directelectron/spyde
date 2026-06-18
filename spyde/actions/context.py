"""
context.py — ActionContext: the adapter passed to action functions.

In the Qt architecture, action functions received a `RoundedToolBar` and read
`.plot`, `.plot_window`, `.action_widgets`, etc.  In the Electron architecture
the toolbar UI lives in the frontend, so this lightweight object provides the
same attribute surface backed by the new Plot/PlotWindow/Session objects.

Parameter values that the old CaretParams popouts collected are now collected by
the Electron parameter panel and arrive in the action payload — they are exposed
here as ``self.params`` and forwarded as kwargs to the action function.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow
    from spyde.backend.session import Session


def src_plot_tree(session, plot):
    """Resolve the source signal plot + its tree for a staged-wizard handler:
    the given *plot*, or the first non-navigator signal plot in the session.
    Returns ``(plot, tree)`` (either may be ``None``)."""
    src = plot or next(
        (p for p in session._plots if not p.is_navigator and p.plot_state is not None),
        None,
    )
    tree = getattr(src, "signal_tree", None) if src is not None else None
    return src, tree


def current_signal(src):
    """The signal currently displayed by *src* (its plot_state's current_signal)."""
    ps = getattr(src, "plot_state", None) if src is not None else None
    return getattr(ps, "current_signal", None) if ps is not None else None


class ActionContext:
    """Adapter exposing the old toolbar interface over the new objects."""

    def __init__(
        self,
        plot: "Plot",
        params: dict[str, Any] | None = None,
        action_name: str = "",
    ):
        self.plot = plot
        self.params = params or {}
        self.action_name = action_name

        # Per-plot persistent action state (FFT windows, toggle groups, etc.).
        # Stored on the plot so it survives across action invocations.
        if not hasattr(plot, "_action_widgets"):
            plot._action_widgets = {}
        self.action_widgets = plot._action_widgets

    # ── Old toolbar attribute surface ──────────────────────────────────────────

    @property
    def plot_window(self) -> "PlotWindow | None":
        return self.plot.plot_window

    @property
    def session(self) -> "Session | None":
        return self.plot.session

    # ── Stateful action registration (replaces toolbar.register_*) ─────────────

    def register_action_plot_item(self, action_name: str, item, key: str) -> None:
        slot = self.action_widgets.setdefault(action_name, {})
        slot.setdefault("plot_items", {})[key] = item

    def register_action_plot_window(self, action_name: str, plot_window, key: str) -> None:
        slot = self.action_widgets.setdefault(action_name, {})
        slot.setdefault("plot_windows", {})[key] = plot_window

    def add_action_widget(self, action_name: str, widget=None, layout=None) -> None:
        slot = self.action_widgets.setdefault(action_name, {})
        slot["widget"] = widget
        slot["layout"] = layout

    def actions(self) -> list:
        """No Qt actions in the Electron toolbar — return empty list."""
        return []

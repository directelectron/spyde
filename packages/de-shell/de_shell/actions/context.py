"""
context.py — ActionContext: the adapter passed to action functions.

An action function is handed one of these instead of reaching for the UI. It
carries the clicked plot, the parameter values the frontend's panel collected
(forwarded as kwargs), and a per-plot scratch dict for state that must outlive a
single invocation (an FFT window, a toggle group, a widget the action added).

Everything it touches is duck-typed — ``plot.plot_window``, ``plot.session`` —
so it does not care what kind of plot or session an app has.
"""
from __future__ import annotations

from typing import Any


class ActionContext:
    """The attribute surface an action function is written against."""

    def __init__(self, plot, params: dict[str, Any] | None = None,
                 action_name: str = ""):
        self.plot = plot
        self.params = params or {}
        self.action_name = action_name

        # Per-plot persistent action state (FFT windows, toggle groups, …).
        # Stored ON THE PLOT so it survives across action invocations — an
        # ActionContext is built fresh for each one.
        if not hasattr(plot, "_action_widgets"):
            plot._action_widgets = {}
        self.action_widgets = plot._action_widgets

    # ── Plot / session access ─────────────────────────────────────────────────

    @property
    def plot_window(self):
        return self.plot.plot_window

    @property
    def session(self):
        return self.plot.session

    # ── Stateful action registration ──────────────────────────────────────────

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
        """The toolbar lives in the frontend, so there are no host-side action
        objects to return. Kept because action code written against the old Qt
        toolbar still calls it."""
        return []

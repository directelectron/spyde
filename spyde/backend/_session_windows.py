"""
_session_windows.py — WindowManagerMixin extracted from session.py.

Plot/window registration, lookup, teardown (close window/plot/tree, selector
cleanup, per-window state forget), and figure resize / event dispatch.

NOTE: MDIManager is intentionally NOT folded in here — Session keeps
``self.mdi_manager`` and this mixin just calls it via ``self.mdi_manager``.
Folding it would add risk (its own API + ``session._mg``-style usages on other
objects) for no benefit; deferred per the 5f plan.

The mixin only USES ``self.<attr>`` (``self._plots``, ``self.signal_trees``,
``self.mdi_manager``, ``self._action_artifacts`` …) set up by the final Session.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from spyde.backend import ipc

if TYPE_CHECKING:
    from spyde.signal_tree import BaseSignalTree
    from spyde.drawing.plots.plot import Plot

log = logging.getLogger(__name__)


class WindowManagerMixin:
    def register_plot(self, plot: "Plot") -> None:
        self._plots.append(plot)

    def unregister_plot(self, plot: "Plot") -> None:
        self._plots = [p for p in self._plots if p is not plot]

    def next_window_id(self) -> int:
        wid = self._next_window_id
        self._next_window_id += 1
        return wid

    def _plot_by_window_id(self, window_id: int):
        for p in self._plots:
            if getattr(p, "window_id", None) == window_id:
                return p
        return None

    def register_window_controller(self, window_id: int, controller) -> None:
        """Give a non-Plot window (bare `figure` emit) a dispatch + teardown
        identity. See the WindowController protocol in spyde/actions/registry.py.
        _forget_window pops the controller and calls its close()."""
        self._window_controllers[window_id] = controller

    def controller_by_window_id(self, window_id: int | None):
        if window_id is None:
            return None
        return self._window_controllers.get(window_id)

    def _close_window(self, window_id: int) -> None:
        plot = self._plot_by_window_id(window_id)
        if plot is None:
            # A controller-backed window (strain map, IPF views…) has no Plot;
            # _forget_window closes its controller and emits window_closed.
            # Otherwise the backend already dropped it (or never had it) — still
            # tell the renderer to remove the window so the UI doesn't get stuck.
            self._forget_window(window_id)
            return
        try:
            tree = getattr(plot, "signal_tree", None)
            if tree is None:
                self._close_plot(plot)
                return
            # Scoping (per spec): the NAVIGATOR's X closes the whole tree (all
            # signals share its dataset); a signal window's X closes ONLY that
            # signal (and its selectors / action popouts). A lone signal plot
            # with no navigator falls through to closing its tree once empty.
            if getattr(plot, "is_navigator", False):
                self._close_tree(tree)
            else:
                self._close_signal_plot(plot, tree)
        except Exception as e:
            log.warning("close_window failed: %s", e)

    def _close_plot(self, plot) -> None:
        """Tear down a single plot and tell the renderer to drop its window."""
        wid = getattr(plot, "window_id", None)
        self._cleanup_plot_selectors(plot)
        try:
            plot.close()
        finally:
            self.unregister_plot(plot)
        self._forget_window(wid)

    def _close_signal_plot(self, plot, tree) -> None:
        """Close a single non-navigator signal window, leaving the rest of the
        tree open. Cleans up the plot's selectors / source ROI, then drops the
        tree entirely if nothing is left open."""
        self._close_plot(plot)
        try:
            if plot in getattr(tree, "signal_plots", []):
                tree.signal_plots.remove(plot)
        except Exception as e:
            log.debug("removing plot from tree.signal_plots failed: %s", e)
        # If no windows of this tree remain open, retire the tree.
        remaining = [p for p in self._plots if getattr(p, "signal_tree", None) is tree]
        if not remaining and tree in self.signal_trees:
            try:
                tree.close()
            except Exception as e:
                log.debug("retiring tree on last window close failed: %s", e)
            self.signal_trees.remove(tree)

    def _cleanup_plot_selectors(self, plot) -> None:
        """Close any selectors owned by / driving this plot, so closing a virtual
        image (etc.) also removes its source ROI from the parent plot."""
        # The selector on the PARENT plot that drives this output window.
        try:
            pw = getattr(plot, "plot_window", None)
            parent_sel = getattr(pw, "parent_selector", None)
            if parent_sel is not None and hasattr(parent_sel, "close"):
                parent_sel.close()
        except Exception as e:
            log.debug("closing parent selector failed: %s", e)
        # Selectors living on this plot itself.
        try:
            state = getattr(plot, "plot_state", None)
            for attr in ("plot_selectors", "signal_tree_selectors"):
                for sel in list(getattr(state, attr, []) or []):
                    if hasattr(sel, "close"):
                        try:
                            sel.close()
                        except Exception as e:
                            log.debug("closing plot selector failed: %s", e)
        except Exception as e:
            log.debug("iterating plot selectors for cleanup failed: %s", e)

    def _close_tree(self, tree: "BaseSignalTree") -> None:
        if tree not in self.signal_trees:
            return
        # Collect every plot/window belonging to this tree BEFORE teardown.
        plots = [p for p in self._plots if getattr(p, "signal_tree", None) is tree]
        window_ids = sorted({
            p.window_id for p in plots if getattr(p, "window_id", None) is not None
        })
        try:
            tree.close()
        except Exception as e:
            log.debug("closing tree in _close_tree failed: %s", e)
        for p in plots:
            self._cleanup_plot_selectors(p)
            try:
                p.close()
            except Exception as e:
                log.debug("closing plot in _close_tree failed: %s", e)
            self.unregister_plot(p)
        if tree in self.signal_trees:
            self.signal_trees.remove(tree)
        for wid in window_ids:
            self._forget_window(wid)

    def _forget_window(self, window_id: int | None) -> None:
        """Drop per-window backend state and tell the renderer to remove it."""
        if window_id is None:
            return
        # A controller-backed window tears itself down through its controller
        # (strain overlay + reference window, IPF nav hooks…), whatever the
        # close path was (✕, tree close, wizard stop).
        ctrl = self._window_controllers.pop(window_id, None)
        if ctrl is not None and hasattr(ctrl, "close"):
            try:
                ctrl.close()
            except Exception as e:
                log.debug("window controller close failed: %s", e)
        # Figures kept alive for this window (bare-figure emits) die with it.
        from spyde.actions.figure_registry import forget_window as _figs_forget
        _figs_forget(window_id)
        if hasattr(self, "_nav_selectors"):
            self._nav_selectors.pop(window_id, None)
        # Prune the MDIManager's PlotWindow tracking so closed windows don't leak.
        try:
            self.mdi_manager.remove_plot_window(window_id)
        except Exception as e:
            log.debug("pruning plot window from MDIManager failed: %s", e)
        # Drop any action-artifact entries that source from or output to this
        # window so a re-run starts clean and a closed output isn't "active".
        for k in [k for k, v in self._action_artifacts.items()
                  if k[0] == window_id or window_id in v.get("out_wids", [])]:
            self._action_artifacts.pop(k, None)
            # Tell the source window's toolbar to un-highlight the action.
            ipc.emit({"type": "action_active", "window_id": k[0], "name": k[1], "active": False})
        ipc.emit({"type": "window_closed", "window_id": window_id})

    def _resize_figure(self, window_id: int, width: int | None, height: int | None) -> None:
        plot = self._plot_by_window_id(window_id)
        if plot is None or width is None or height is None:
            return
        try:
            import anyplotlib._electron as _el
            fig_id = getattr(plot, "fig_id", None)
            if fig_id is not None:
                _el.resize_figure(fig_id, int(width), int(height))
        except Exception as e:
            log.warning("resize_figure failed: %s", e)

    def _dispatch_figure_event(self, window_id: int, event_json: str | None) -> None:
        if event_json is None:
            return
        plot = self._plot_by_window_id(window_id)
        if plot is None:
            return
        try:
            import anyplotlib._electron as _el
            fig_id = getattr(plot, "fig_id", None)
            if fig_id is not None:
                _el.dispatch_event(fig_id, event_json)
        except Exception as e:
            log.warning("dispatch_figure_event failed: %s", e)

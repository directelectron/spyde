from __future__ import annotations
import math
from typing import TYPE_CHECKING
from uuid import uuid4

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QEvent, QObject, Signal
from PySide6.QtWidgets import QGraphicsOpacityEffect

import pyqtgraph as pg

from spyde.actions.base import NAVIGATOR_DRAG_MIME

if TYPE_CHECKING:
    from spyde.__main__ import MainWindow
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_window import PlotWindow
    from spyde.drawing.plots.multiplot_manager import MultiplotManager
    from spyde.signal_tree import BaseSignalTree


class MDIManager(QObject):
    """Owns MDI subwindow lifecycle, 3-state visibility, and drag-and-drop."""

    subwindow_activated = Signal(object)  # PlotWindow

    def __init__(self, mdi_area: QtWidgets.QMdiArea, main_window: "MainWindow", parent=None):
        super().__init__(parent)
        self.mdi_area = mdi_area
        self.main_window = main_window
        self.plot_subwindows: list["PlotWindow"] = []
        self.signal_trees: list["BaseSignalTree"] = []
        self._navigator_drag_payloads: dict[str, dict] = {}
        self._navigator_drag_over_active = False
        self._navigator_placeholder = None
        self._navigator_placeholder_rect = None
        self._in_subwindow_activation = False

        self.mdi_area.subWindowActivated.connect(self._on_subwindow_activated)
        self.mdi_area.setAcceptDrops(True)
        self.mdi_area.installEventFilter(self)

    # ── Public interface ──────────────────────────────────────────────────────

    def add_plot_window(
        self,
        *,
        is_navigator: bool = False,
        plot_manager: "MultiplotManager | None" = None,
        signal_tree: "BaseSignalTree | None" = None,
    ) -> "PlotWindow":
        from spyde.drawing.plots.plot_window import PlotWindow
        from PySide6.QtCore import Qt

        screen_size = self.main_window.screen_size
        pw = PlotWindow(
            is_navigator=is_navigator,
            main_window=self.main_window,
            signal_tree=signal_tree,
            plot_manager=plot_manager,
        )
        pw.resize(screen_size.height() // 3, screen_size.height() // 3)
        self.mdi_area.addSubWindow(pw)
        try:
            pw.setWindowFlags(pw.windowFlags() | Qt.WindowType.FramelessWindowHint)
            pw.setStyleSheet("QMdiSubWindow { border: none; }")
        except Exception:
            pass
        pw.show()
        self.plot_subwindows.append(pw)
        return pw

    def windows_for_tree(self, tree: "BaseSignalTree") -> list["PlotWindow"]:
        """Every plot window belonging to a signal tree — its navigator/signal
        plots AND any action-spawned previews (VI previews, IPF, vector-OM maps),
        which all carry .signal_tree. The single source of truth for what must be
        torn down when the tree goes away."""
        return [pw for pw in list(self.plot_subwindows)
                if getattr(pw, "signal_tree", None) is tree]

    def close_signal_tree(self, tree: "BaseSignalTree") -> None:
        """Authoritatively tear down a whole signal tree: every associated plot
        window (and its toolbars/PlotStates/selectors), then the tree itself.

        This is the ONE place that does the tree-level bookkeeping. PlotWindow
        close paths delegate here so closing any window of a tree (e.g. the
        navigator) removes the entire graph rather than orphaning previews,
        toolbars, or the tree entry. Idempotent + re-entrancy guarded.
        """
        if tree is None or getattr(tree, "_spyde_closing", False):
            return
        tree._spyde_closing = True
        try:
            for pw in self.windows_for_tree(tree):
                try:
                    # tell the window not to re-trigger tree teardown
                    pw._spyde_tree_teardown = True
                    pw.close_window()
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception(
                        "close_signal_tree: window close failed")
            # release any tree-held resources (toolbars are closed per-window)
            try:
                tree.close()
            except Exception:
                pass
            if tree in self.signal_trees:
                self.signal_trees.remove(tree)
            if (getattr(self.main_window, "current_selected_signal_tree", None)
                    is tree):
                self.main_window.current_selected_signal_tree = None
        finally:
            tree._spyde_closing = False

    def active_plot(self) -> "Plot | None":
        sub = self.mdi_area.activeSubWindow()
        from spyde.drawing.plots.plot_window import PlotWindow
        if not isinstance(sub, PlotWindow):
            return None
        return sub.current_plot_item

    def active_plot_window(self) -> "PlotWindow | None":
        sub = self.mdi_area.activeSubWindow()
        from spyde.drawing.plots.plot_window import PlotWindow
        if not isinstance(sub, PlotWindow):
            return None
        return sub

    def _active_tree_windows(self):
        """Visible subwindows belonging to the active subwindow's signal tree."""
        from spyde.drawing.plots.plot_window import PlotWindow
        active = self.mdi_area.activeSubWindow()
        if not isinstance(active, PlotWindow):
            return []
        active_tree = active.signal_tree
        return [
            pw for pw in self.plot_subwindows
            if pw.signal_tree is active_tree and pw.isVisible()
        ]

    def tile_active_windows(self) -> None:
        """Tile (resize + position) the active tree's windows into a grid."""
        shown = self._active_tree_windows()
        n = len(shown)
        if n == 0:
            return
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        mdi_rect = self.mdi_area.rect()
        margin = 6
        cell_w = (mdi_rect.width() - margin * (cols + 1)) // cols
        cell_h = (mdi_rect.height() - margin * (rows + 1)) // rows
        for i, pw in enumerate(shown):
            row = i // cols
            col = i % cols
            pw.setGeometry(
                margin + col * (cell_w + margin),
                margin + row * (cell_h + margin),
                cell_w,
                cell_h,
            )

    def organize_active_windows(self) -> None:
        """Arrange the active tree's windows into a non-overlapping grid WITHOUT
        resizing them — each keeps its current size, only its position changes.
        Laid out left-to-right, top-to-bottom; rows advance by the tallest
        window so far and wrap when they'd overflow the MDI width."""
        shown = self._active_tree_windows()
        if not shown:
            return
        mdi_rect = self.mdi_area.rect()
        margin = 6
        x = margin
        y = margin
        row_h = 0
        for pw in shown:
            w, h = pw.width(), pw.height()
            # wrap to a new row if this window would overflow the right edge
            if x > margin and x + w + margin > mdi_rect.width():
                x = margin
                y += row_h + margin
                row_h = 0
            pw.move(x, y)
            x += w + margin
            row_h = max(row_h, h)

    def register_navigator_drag_payload(self, signal, nav_manager) -> str:
        token = uuid4().hex
        self._navigator_drag_payloads[token] = {
            "signal": signal,
            "nav_manager": nav_manager,
        }
        return token

    def auto_position_near_owner(self, pw: "PlotWindow") -> None:
        owner = pw.owner_plot_window
        if owner is None:
            return
        mdi_rect = self.mdi_area.rect()
        gap = 8
        x = owner.x() + owner.width() + gap
        y = owner.y()
        if x + pw.width() <= mdi_rect.width():
            pw.move(x, y)
            return
        x = owner.x()
        y = owner.y() + owner.height() + gap
        if y + pw.height() <= mdi_rect.height():
            pw.move(x, y)
            return
        x = min(x, max(0, mdi_rect.width() - pw.width()))
        y = min(y, max(0, mdi_rect.height() - pw.height()))
        pw.move(x, y)

    # ── Subwindow activation ─────────────────────────────────────────────────

    def _on_subwindow_activated(self, window: "PlotWindow") -> None:
        if self._in_subwindow_activation:
            return
        self._in_subwindow_activation = True
        try:
            self._on_subwindow_activated_impl(window)
        finally:
            self._in_subwindow_activation = False

    def _on_subwindow_activated_impl(self, window: "PlotWindow") -> None:
        from spyde.drawing.plots.plot_window import PlotWindow
        if window is None or not isinstance(window, PlotWindow):
            return

        plot = window.current_plot_item
        if plot is None:
            return

        self._update_toolbar_visibility(window)
        self._update_3state_visibility(window)
        self.subwindow_activated.emit(window)

    def _update_toolbar_visibility(self, window: "PlotWindow") -> None:
        if window.signal_tree is not None and window.signal_tree.navigator_plot_manager is not None:
            active_plots = [
                w.current_plot_item
                for w in window.signal_tree.navigator_plot_manager.all_plot_windows
                if w.isVisible()
            ]
        else:
            active_plots = [window.current_plot_item]

        for plt in active_plots:
            if getattr(plt, "plot_state", None) is not None:
                plt.plot_state.show_toolbars()
            if hasattr(plt, "show_selector_control_widget"):
                plt.show_selector_control_widget()

        for pw in self.plot_subwindows:
            for plt in pw.plots:
                if plt in active_plots:
                    continue
                if getattr(plt, "plot_state", None) is not None:
                    plt.plot_state.hide_toolbars()

    def _update_3state_visibility(self, window: "PlotWindow") -> None:
        active_tree = window.signal_tree
        for pw in self.plot_subwindows:
            if getattr(pw, "_spyde_closed", False):
                continue
            same_tree = pw.signal_tree is active_tree
            is_action_preview = pw.owner_plot_window is not None
            action = getattr(pw, "controlling_action", None)
            try:
                action_wants_visible = (
                    action is None or not action.isCheckable() or action.isChecked()
                )
            except RuntimeError:
                # C++ QAction object was deleted (toolbar/PlotState torn down);
                # clear the stale reference and treat as no controlling action.
                pw.controlling_action = None
                action_wants_visible = True
            gate = getattr(pw, "visibility_gate", None)
            if action_wants_visible and callable(gate):
                try:
                    action_wants_visible = bool(gate())
                except Exception:
                    pass
            if same_tree and action_wants_visible:
                if not pw.isVisible():
                    pw.show()
                pw.setGraphicsEffect(None)
                pw.raise_()        # focused tree comes to the front
            elif same_tree and not action_wants_visible:
                pw.hide()
            elif is_action_preview and not pw.isVisible():
                # Already hidden action-preview windows from other trees stay hidden;
                # visible ones are dimmed (not forcibly hidden) so commits don't
                # unexpectedly close live preview windows.
                pass
            elif is_action_preview:
                # Dim AND send to the back: out-of-focus trees recede both in
                # opacity and z-order so the active tree's windows read clearly.
                effect = QGraphicsOpacityEffect(pw)
                effect.setOpacity(0.65)
                pw.setGraphicsEffect(effect)
                pw.lower()
            else:
                if not pw.isVisible():
                    pw.show()
                effect = QGraphicsOpacityEffect(pw)
                effect.setOpacity(0.65)
                pw.setGraphicsEffect(effect)
                pw.lower()

    # ── Drag-and-drop event filter ───────────────────────────────────────────

    def eventFilter(self, obj, event: QEvent) -> bool:
        if event is None:
            return False
        if obj is self.mdi_area:
            et = event.type()
            if et in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
                mime = event.mimeData()
                if mime is not None and mime.hasFormat(NAVIGATOR_DRAG_MIME):
                    active_sub = self.mdi_area.activeSubWindow()
                    if active_sub is not None:
                        try:
                            pos = event.position().toPoint()
                        except Exception:
                            pos = event.pos()
                        if active_sub.geometry().contains(pos):
                            if not self._navigator_drag_over_active:
                                self._navigator_enter()
                            else:
                                self._navigator_move(pos)
                            self._navigator_drag_over_active = True
                            event.acceptProposedAction()
                            return True
                        if self._navigator_drag_over_active:
                            self._navigator_leave()
                            self._navigator_drag_over_active = False
                        return False
                paths = self._extract_file_paths(mime)
                if any(self._is_supported_file(p) for p in paths):
                    event.acceptProposedAction()
                    return True

            elif et == QEvent.Type.Drop:
                mime = event.mimeData()
                if mime is not None and mime.hasFormat(NAVIGATOR_DRAG_MIME):
                    active_sub = self.mdi_area.activeSubWindow()
                    try:
                        pos = event.position().toPoint()
                    except Exception:
                        pos = event.pos()
                    if active_sub is not None and active_sub.geometry().contains(pos):
                        self._navigator_drag_over_active = False
                        self._navigator_drop(pos, mime)
                        event.acceptProposedAction()
                        return True
                    if self._navigator_drag_over_active:
                        self._navigator_drag_over_active = False
                    return False
                paths = self._extract_file_paths(mime)
                if any(self._is_supported_file(p) for p in paths):
                    self.main_window._handle_drop_files(paths)
                    event.acceptProposedAction()
                    return True

            elif et == QEvent.Type.DragLeave:
                if self._navigator_drag_over_active:
                    self._navigator_leave()
                    self._navigator_drag_over_active = False

        return super().eventFilter(obj, event)

    def _navigator_enter(self) -> None:
        placeholder = pg.PlotItem()
        placeholder.setTitle("Drop Navigator Here", color="#888888")
        placeholder.hideAxis("left")
        placeholder.hideAxis("bottom")
        rect = pg.QtWidgets.QGraphicsRectItem()
        rect.setBrush(pg.mkBrush((100, 100, 255, 100)))
        rect.setPen(pg.mkPen((100, 100, 255), width=2))
        placeholder.addItem(rect)
        self._navigator_placeholder = placeholder
        self._navigator_placeholder_rect = rect

    def _navigator_move(self, pos: QtCore.QPointF) -> None:
        active = self.active_plot_window()
        if active is None or self._navigator_placeholder is None:
            return
        active._build_new_layout(drop_pos=pos, plot_to_add=self._navigator_placeholder)
        if self._navigator_placeholder_rect is not None:
            vb = self._navigator_placeholder.getViewBox()
            self._navigator_placeholder_rect.setRect(vb.rect())

    def _navigator_leave(self) -> None:
        active = self.active_plot_window()
        if active is not None:
            active.set_graphics_layout_widget(active.previous_subplots_pos)

    def _navigator_drop(self, pos: QtCore.QPointF, mime_data) -> None:
        active = self.active_plot_window()
        if active is None:
            return
        nav_plot = active.insert_new_plot(drop_pos=pos)
        token = mime_data.data(NAVIGATOR_DRAG_MIME).data().decode("utf-8")
        payload = self._navigator_drag_payloads.pop(token, None)
        if payload is None:
            return
        signal = payload["signal"]
        for navigation_signal in nav_plot.signal_tree.navigator_signals.values():
            nav_plot.multiplot_manager.add_plot_states_for_navigation_signals(navigation_signal)
        nav_plot.set_plot_state(signal=signal[0])
        active.previous_subplots_pos = {}
        active.previous_subplot_added = None

    def _is_supported_file(self, path: str) -> bool:
        import os
        from spyde.__main__ import SUPPORTED_EXTS
        try:
            return os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXTS)
        except Exception:
            return False

    def _extract_file_paths(self, mime) -> list[str]:
        import os
        paths = []
        if mime is None:
            return paths
        if mime.hasUrls():
            for url in mime.urls():
                if url.isLocalFile():
                    p = url.toLocalFile()
                    if p:
                        paths.append(p)
        elif mime.hasText():
            for chunk in mime.text().split():
                if os.path.isfile(chunk):
                    paths.append(chunk)
        return paths

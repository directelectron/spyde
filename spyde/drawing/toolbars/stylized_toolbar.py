from typing import Optional, Callable

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import Qt

from spyde.qt.style import ACCENT_SOFT

from typing import TYPE_CHECKING
while TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot
    from spyde.drawing.plots.plot_states import PlotState


class StylizedToolBar(QtWidgets.QToolBar):
    """
    A QToolBar with rounded corners and a semi-transparent background.

    This toolbar is designed to be used alongside a Plot widget, allowing for floating
    tools around some plot area. ("top", "bottom", "left", "right")

    Parameters
    ----------
    title : str
        The title of the toolbar.
    plot : Plot, optional
        The associated Plot widget to position the toolbar next to.
    parent : QWidget, optional
        The parent widget.
    radius : int, optional
        The corner radius for the rounded corners. Default is 8.
    moveable : bool, optional
        Whether the toolbar is moveable by the user. Default is False.
    position : str, optional
        The position of the toolbar relative to the plot ("top", "bottom", "left", "right").
        Default is "top".
    """

    def __init__(
        self,
        title: str,
        plot_state: "PlotState" = None,
        parent: Optional[QtWidgets.QWidget] = None,
        radius: int = 8,
        moveable: bool = False,
        position: str = "top",
        exclusive_checkable_actions: bool = True,
    ):
        # Ensure we never parent directly to QMainWindow; prefer content area
        parent = self._resolve_container_parent(parent)

        # The plot state is the parent context for this toolbar
        # Each plot state is tied to a particular signal and plot instance via the PlotState
        self.plot_state = plot_state
        self.layout_padding = (0, 0, 0, 0)  # left, top, right, bottom

        # Normalize position like "top-left" -> primary side
        def _normalize_position(pos: str) -> str:
            if not isinstance(pos, str):
                return "top"
            p = pos.lower()
            if "left" in p:
                return "left"
            if "right" in p:
                return "right"
            if "bottom" in p:
                return "bottom"
            return "top"

        self.exclusive_checkable_actions: bool = bool(exclusive_checkable_actions)

        norm_pos = _normalize_position(position)
        vertical = norm_pos not in ["top", "bottom"]
        self.position = norm_pos
        super().__init__(title, parent)

        if plot_state is None:
            self.plot_window = None
            self.plot: Optional[Plot] = None
        else:
            self.plot: Plot = self.plot_state.plot
            self.plot_window = self.plot.plot_window

        self._radius = float(radius)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(False)
        self.setContentsMargins(0, 0, 0, 0)
        self.action_widgets: dict[str, dict] = {}

        # set up fixed style
        self.setOrientation(
            QtCore.Qt.Orientation.Vertical
            if vertical
            else QtCore.Qt.Orientation.Horizontal
        )

        # Compact buttons
        self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setIconSize(QtCore.QSize(18, 18))

        # Hover/pressed background for tool buttons.  This
        self.setStyleSheet(
            "QToolBar {"
            "  background: transparent;"
            "  border: none;"
            "  padding: 4px;"
            "  margin: 0px;"
            "}"
            "QToolButton {"
            "  border: none;"
            "  margin: 2px;"
            "  background: transparent;"
            "  padding: 4px;"
            "  border-radius: 6px;"
            "}"
            "QToolButton:hover {"
            "  background-color: rgba(255, 255, 255, 40);"
            "}"
            "QToolButton:pressed {"
            "  background-color: rgba(255, 255, 255, 64);"
            "}"
            # checked = soft orange accent so an active tool reads as active
            # (plain white would be indistinguishable from hover)
            f"QToolButton:checked {{"
            f"  background-color: {ACCENT_SOFT};"
            f"}}"
        )
        self.setMovable(moveable)

        # Move/link guards and margin
        self._move_sync = False
        self._margin = 8
        self.set_size()

        # Track geometry to keep the toolbar and spawned widgets positioned
        self._position_tracker = self._make_position_tracker(self.move_next_to_plot)
        self.installEventFilter(self._position_tracker)
        if (
            container := self._resolve_container_parent(self.parentWidget())
        ) is not None:
            container.installEventFilter(self._position_tracker)
        if self.plot_window is not None:
            self.plot_window.installEventFilter(self._position_tracker)

    def set_size(self) -> None:
        """Fix the toolbar size to its sizeHint and schedule initial placement next to the plot."""
        # Lock size so it doesn't change when moved
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        size = self.sizeHint()
        self.setFixedSize(size)

        # Initial placement
        QtCore.QTimer.singleShot(0, self.move_next_to_plot)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        # Sub-pixel align for crisp 1px stroke at any DPR
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, self._radius, self._radius)

        # Fill
        p.setBrush(QtGui.QColor(30, 30, 30, 240))
        # 1px cosmetic pen to keep edge sharp on HiDPI
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 120))
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)

        p.drawPath(path)
        super().paintEvent(ev)

    @staticmethod
    def _resolve_container_parent(
        parent: Optional[QtWidgets.QWidget],
    ) -> Optional[QtWidgets.QWidget]:
        # Prefer a content container instead of QMainWindow for overlays
        if isinstance(parent, QtWidgets.QMainWindow):
            cw = parent.centralWidget()
            if isinstance(cw, QtWidgets.QMdiArea):
                return cw.viewport()
            return cw or parent
        return parent

    def move_next_to_plot(self) -> None:
        """Anchor the toolbar adjacent to its Plot according to self.position and re-place visible popouts."""
        if self.plot_window is None:
            return

        parent = self._resolve_container_parent(self.parentWidget())
        if parent is None:
            return
        plot_global_tl = self.plot_window.mapToGlobal(QtCore.QPoint(0, 0))

        if self.position == "left":
            desired_global = QtCore.QPoint(
                plot_global_tl.x() - self.width() - self._margin,
                plot_global_tl.y() + self._margin,
            )
        elif self.position == "right":
            desired_global = QtCore.QPoint(
                plot_global_tl.x() + self.plot_window.width() + self._margin,
                plot_global_tl.y() + self._margin,
            )
        elif self.position == "top":
            desired_global = QtCore.QPoint(
                plot_global_tl.x() + self._margin,
                plot_global_tl.y() - self.height() - self._margin,
            )
        else:  # "bottom"
            desired_global = QtCore.QPoint(
                plot_global_tl.x() + self._margin,
                plot_global_tl.y() + self.plot_window.height() + self._margin,
            )

        desired_in_parent = parent.mapFromGlobal(desired_global)

        self._move_sync = True
        try:
            self.move(desired_in_parent)
            self.raise_()
        finally:
            self._move_sync = False

        # Reposition any open widgets
        for data in self.action_widgets.values():
            fn = data.get("position_fn")
            w = data.get("widget")
            if callable(fn) and w is not None and w.isVisible():
                QtCore.QTimer.singleShot(0, fn)

    @staticmethod
    def _make_position_tracker(callback: Callable[[], None]) -> QtCore.QObject:
        """Create an event filter object that triggers callback on move/resize/show of watched widgets."""

        class _Tracker(QtCore.QObject):
            def eventFilter(self, obj, event):
                if event.type() in (
                    QtCore.QEvent.Type.Move,
                    QtCore.QEvent.Type.Resize,
                    QtCore.QEvent.Type.Show,
                ):
                    QtCore.QTimer.singleShot(0, callback)
                return False

        return _Tracker()

    def _remove_event_filter_safe(self, tracker: QtCore.QObject) -> None:
        parent = self._resolve_container_parent(self.parentWidget())
        try:
            self.removeEventFilter(tracker)
        except Exception:
            pass
        if parent is not None:
            try:
                parent.removeEventFilter(tracker)
            except Exception:
                pass
        if self.plot is not None:
            try:
                self.plot.removeEventFilter(tracker)
            except Exception:
                pass

from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Union, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

if TYPE_CHECKING:
    from spyde.drawing.plots.plot import Plot

from spyde.drawing.toolbars.toolbar import RoundedToolBar
from spyde.qt.style import ACCENT_SOFT

class PopoutToolBar(RoundedToolBar):
    """
    A floating RoundedToolBar with a caret tip, intended to be used as a popout anchored to a parent toolbar action.
    It shares the same style as RoundedToolBar but draws a caret and reserves space for it, like CaretGroup.
    """

    def __init__(
        self,
        title: str,
        plot: "Plot" = None,
        parent: Optional[QtWidgets.QWidget] = None,
        radius: int = 8,
        moveable: bool = False,
        position: str = "bottom",
        reposition_function: Optional[Callable[[], None]] = None,
        *,
        side: str = "auto",
        caret_base: int = 14,
        caret_depth: int = 8,
        padding: int = 8,
        parent_toolbar: Optional[RoundedToolBar] = None,
    ):

        # Popout does not need plot tracking; pass plot=None to avoid auto-placement
        self._side = (
            side if side in ("top", "bottom", "left", "right", "auto") else "auto"
        )
        self._caret_base = int(caret_base)
        self._caret_depth = int(caret_depth)
        self._padding = int(0)
        self._reposition_function = reposition_function
        self.parent_toolbar = parent_toolbar

        # TODO: Should the plot state be the same as the parent toolbar's?
        super().__init__(
            title,
            plot_state=None,
            parent=parent,
            radius=radius,
            moveable=moveable,
            position=position,
        )
        # Visual params for caret bubble
        self._pen_color = QtGui.QColor(255, 255, 255, 120)
        self._bg_color = QtGui.QColor(30, 30, 30, 240)
        # Transparent background; we fully paint our bubble

        top_margin = 2 + self._caret_depth
        self.setStyleSheet(
            f"QToolBar {{"
            "  background: transparent;"
            "  border: none;"
            "  padding: 4px;"
            "  margin: 0px;"
            "}"
            f"QToolButton {{"
            "  border: none;"
            f"  margin: {top_margin}px 2px 2px 2px;"
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
            # checked = soft orange accent, same as RoundedToolBar
            f"QToolButton:checked {{"
            f"  background-color: {ACCENT_SOFT};"
            f"}}"
        )

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Allow dynamic growth; RoundedToolBar.__init__ fixed the size – undo that here.
        self._unlock_fixed_size()
        self._margin = 1
        self.layout_padding = (
            self._padding,
            self._padding,
            self._padding,
            self._padding,
        )  # left, top, right, bottom

    def _unlock_fixed_size(self):
        # Remove fixed-size constraints introduced by RoundedToolBar.set_size()
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.setMinimumSize(0, 18)  # minimum height to fit buttons
        self.setMaximumSize(16777215, 16777215)

    # Ensure subtoolbars grow to fit their actions (don't lock to fixed size)
    def _refresh_fixed_size(self):
        self.adjustSize()

    # Optional helper if caller updates content later
    def content_updated(self):
        self.adjustSize()
        self.update()

    def set_side(self, side: str) -> None:
        """Set caret side ('top'/'bottom'/'left'/'right') and trigger repaint."""
        if side not in ("top", "bottom", "left", "right"):
            return
        if self._side != side:
            self._side = side
            # self._update_margins()
            self.update()

    def sizeHint(self) -> QtCore.QSize:
        """
        Size hint shouldn't account for caret space.
        """
        s = super().sizeHint()
        if self._side in ("top", "bottom"):
            return QtCore.QSize(s.width(), s.height())
        return QtCore.QSize(s.width(), s.height())

    def _update_margins(self) -> None:
        """Re-apply internal content margins based on caret side & depth."""
        l = r = t = b = self._padding
        if self._side == "top":
            t += self._caret_depth
        elif self._side == "bottom":
            b += self._caret_depth
        elif self._side == "left":
            l += self._caret_depth
        elif self._side == "right":
            r += self._caret_depth
        self.setContentsMargins(l, t, r, b)

    def _bubble_rect(self) -> QtCore.QRectF:
        """Return the QRectF of the rounded bubble excluding the caret triangle area."""
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if self._side == "top":
            rect.adjust(0, self._caret_depth, 0, 0)
        elif self._side == "bottom":
            rect.adjust(0, 0, 0, -self._caret_depth)
        elif self._side == "left":
            rect.adjust(self._caret_depth, 0, 0, 0)
        elif self._side == "right":
            rect.adjust(0, 0, -self._caret_depth, 0)
        return rect

    def _caret_polygon(self, bubble: QtCore.QRectF) -> QtGui.QPolygonF:
        """Generate the caret polygon pointing toward the anchor side."""
        base = float(self._caret_base)
        depth = float(self._caret_depth)
        if self._side in ("top", "bottom"):
            cx = bubble.center().x()
            x1 = cx - base / 2.0
            x2 = cx + base / 2.0
            if self._side == "top":
                y = bubble.top()
                return QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x1, y),
                        QtCore.QPointF(x2, y),
                        QtCore.QPointF(cx, y - depth),
                    ]
                )
            else:
                y = bubble.bottom()
                return QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x1, y),
                        QtCore.QPointF(x2, y),
                        QtCore.QPointF(cx, y + depth),
                    ]
                )
        else:
            cy = bubble.center().y()
            y1 = cy - base / 2.0
            y2 = cy + base / 2.0
            if self._side == "left":
                x = bubble.left()
                return QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x, y1),
                        QtCore.QPointF(x, y2),
                        QtCore.QPointF(x - depth, cy),
                    ]
                )
            else:
                x = bubble.right()
                return QtGui.QPolygonF(
                    [
                        QtCore.QPointF(x, y1),
                        QtCore.QPointF(x, y2),
                        QtCore.QPointF(x + depth, cy),
                    ]
                )

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        bubble = self._bubble_rect()

        bubble_path = QtGui.QPainterPath()
        bubble_path.addRoundedRect(bubble, self._radius, self._radius)

        caret_poly = self._caret_polygon(bubble)
        caret_path = QtGui.QPainterPath()
        caret_path.addPolygon(caret_poly)

        # simplify paths to eliminate seam between bubble and caret
        bubble_path.addPath(caret_path)
        path = bubble_path.simplified()

        p.setBrush(self._bg_color)
        pen = QtGui.QPen(self._pen_color)
        pen.setWidthF(1.0)
        pen.setCosmetic(True)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPath(path)
        # Paint actions/toolbuttons
        QtWidgets.QToolBar.paintEvent(self, ev)
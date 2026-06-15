from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QIcon
from spyde.drawing.toolbars.plot_control_toolbar import resolve_icon_path
from spyde.qt.style import (
    RADIUS, GOOD, GOOD_HOVER, DANGER, DANGER_HOVER, ACCENT, ACCENT_BORDER,
    SmoothButton,
)

# New imports for painting/tinting
from PySide6 import QtGui


class FramelessSubWindow(QtWidgets.QMdiSubWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # small rounded gradient border params
        self._border_radius = 8
        self._border_width = 2

        self._moving = False
        self._move_start = None
        self._resize_start = None

        self.title_bar = QtWidgets.QWidget(self)
        self.title_bar.setFixedHeight(30)
        self.title_bar.setMinimumHeight(30)
        self.title_bar.setMinimumWidth(200)
        self.title_bar.setStyleSheet(
            "QWidget {"
            "  background: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
            "    stop:0 #3a3a3a, stop:1 #2b2b2b);"
            "  border-top-left-radius: 6px;"
            "  border-top-right-radius: 6px;"
            "  border-left: 1px solid rgba(211, 211, 211, 170);"
            "  border-right: 1px solid rgba(211, 211, 211, 170);"
            "  border-top: 1px solid rgba(211, 211, 211, 170);"
            "}"
            "QPushButton {"
            "  background-color: transparent;"
            "  border: none;"
            f"  border-radius: {RADIUS};"
            "}"
            "QPushButton:hover {"
            "  background-color: rgba(255,255,255,18);"
            "}"
            "QPushButton:pressed {"
            "  background-color: rgba(255,255,255,28);"
            "}"
            "QLabel {"
            "  border: none;"
            "}"
        )
        self.title_bar_layout = QtWidgets.QHBoxLayout(self.title_bar)
        self.title_bar_layout.setContentsMargins(5, 5, 5, 5)
        self.title_bar_layout.setSpacing(4)

        self._status_placeholder = QtWidgets.QWidget(self.title_bar)
        self._status_placeholder.setFixedSize(24, 24)
        self._status_placeholder.setStyleSheet("background: transparent;")
        self._status_placeholder.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.title_bar_layout.addWidget(self._status_placeholder)

        self.title_bar.compute_button = SmoothButton(
            "Compute", self.title_bar,
            fill=ACCENT, fill_hover=ACCENT_BORDER, fill_pressed=ACCENT_BORDER,
            fill_disabled="rgba(80,80,80,120)",
            border="rgba(255,255,255,60)",
            text_disabled="rgba(255,255,255,80)",
            font_px=11,
        )
        self.title_bar.compute_button.setFixedHeight(18)
        self.title_bar.compute_button.hide()
        self.title_bar_layout.addWidget(self.title_bar.compute_button)

        self.title_bar.commit_button = SmoothButton(
            "Commit", self.title_bar,
            fill=GOOD, fill_hover=GOOD_HOVER, fill_pressed=GOOD_HOVER,
            fill_disabled="rgba(80,80,80,120)",
            border="rgba(255,255,255,60)",
            text_disabled="rgba(255,255,255,80)",
            font_px=11,
        )
        self.title_bar.commit_button.setFixedHeight(18)
        self.title_bar.commit_button.hide()
        self.title_bar_layout.addWidget(self.title_bar.commit_button)

        self.title_bar.stop_button = SmoothButton(
            "Stop", self.title_bar,
            fill=DANGER, fill_hover=DANGER_HOVER, fill_pressed=DANGER_HOVER,
            fill_disabled="rgba(80,80,80,120)",
            border="rgba(255,255,255,60)",
            text_disabled="rgba(255,255,255,80)",
            font_px=11,
        )
        self.title_bar.stop_button.setFixedHeight(18)
        self.title_bar.stop_button.hide()
        self.title_bar_layout.addWidget(self.title_bar.stop_button)

        self.title_label = QtWidgets.QLabel("", self.title_bar)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: #ffffff;")
        self.title_bar_layout.addWidget(self.title_label, stretch=1)

        self.minimize_button = QtWidgets.QPushButton(self.title_bar)
        self._icon_minimize = QIcon(resolve_icon_path("qt/assets/icons/minimize.svg"))
        self._icon_maximize = QIcon(resolve_icon_path("qt/assets/icons/maximize.svg"))
        self._icon_close = QIcon(resolve_icon_path("qt/assets/icons/close.svg"))

        self.minimize_button.setFixedSize(20, 20)
        self.minimize_button.clicked.connect(self.toggle_minimize)
        self.minimize_button.setIcon(self._icon_minimize)
        self.minimize_button.setIconSize(QtCore.QSize(12, 12))

        self.maximize_button = QtWidgets.QPushButton(self.title_bar)
        self.maximize_button.setIcon(self._icon_maximize)
        self.maximize_button.setCheckable(True)
        self.maximize_button.setChecked(False)
        self.maximize_button.setFixedSize(20, 20)
        self.maximize_button.clicked.connect(self.toggle_maximize)
        self.maximize_button.setIconSize(QtCore.QSize(12, 12))

        self.close_button = QtWidgets.QPushButton(self.title_bar)
        self.close_button.setIcon(self._icon_close)
        self.close_button.setFixedSize(20, 20)
        self.close_button.clicked.connect(self.close)
        self.close_button.setIconSize(QtCore.QSize(12, 12))

        self.title_bar.mousePressEvent = self.start_move
        self.title_bar.mouseMoveEvent = self.move_window
        self.title_bar.mouseReleaseEvent = self.end_move

        self.title_bar_layout.addWidget(self.minimize_button)
        self.title_bar_layout.addWidget(self.maximize_button)
        self.title_bar_layout.addWidget(self.close_button)

        self.setLayout(QtWidgets.QVBoxLayout())

        self.layout().setContentsMargins(3, 3, 3, 3)
        self.layout().addWidget(self.title_bar)
        self.layout().setSpacing(0)
        self.old_size = self.size()
        self.is_minimized = False
        self.setMouseTracking(True)
        self.title_bar.setMouseTracking(True)
        self.title_label.setMouseTracking(True)
        self.setMouseTracking(True)
        self.plot_widget = None
        self.installEventFilter(self)
        for w in self.findChildren(QtWidgets.QWidget):
            w.installEventFilter(self)

        self._resizing_top = False
        self._resizing_bottom = False
        self._resizing_left = False
        self._resizing_right = False

        # Edge snap state
        self._is_dragging = False
        self._snap_preview = None  # QGraphicsRectItem for preview
        self._snap_threshold = -1  # pixels from edge to trigger snap
        self._snap_zone = None  # 'top', 'bottom', 'left', 'right', or None
        self._drag_start_pos = None # Global pos where drag started
        self._snap_state = None  # 'top', 'bottom', 'left', 'right', 'top-left', etc.

    def eventFilter(self, obj, event):
        # Only forward mouse events from the subwindow frame itself or the title bar.
        # Forwarding from every child widget causes O(n_children) dispatch per
        # mouse-move which makes ROI drags and window drags visibly laggy.
        if obj is self or obj is self.title_bar:
            if event.type() == QtCore.QEvent.Type.MouseMove:
                self.mouseMoveEvent(event)
            elif event.type() == QtCore.QEvent.Type.MouseButtonPress:
                self.mousePressEvent(event)
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                self.mouseReleaseEvent(event)
        return super().eventFilter(obj, event)

    def toggle_minimize(self):
        # Minimize the window
        if self.is_minimized:
            self.resize(self.old_size)
            self.is_minimized = False
        else:
            self.is_minimized = True
            self.old_size = self.size()
            self.resize(QtCore.QSize(220, 75))

    def toggle_maximize(self):
        # Toggle between maximized and normal window states
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()


    @property
    def resizing(self):
        return (
            self._resizing_top
            or self._resizing_bottom
            or self._resizing_left
            or self._resizing_right
        )

    def _get_occupied_zones(self) -> set:
        """Return set of zones occupied by other snapped windows."""
        if self.main_window is None:
            return set()

        occupied = set()
        for subwin in self.main_window.mdi_area.subWindowList():
            if subwin is self or not hasattr(subwin, '_snap_state'):
                continue
            if subwin._snap_state:
                occupied.add(subwin._snap_state)
        print("occupied zones:", occupied)
        return occupied

    def _calculate_snap_zone(self, local_pos: QtCore.QPoint, mdi_rect: QtCore.QRect) -> str:
        """Calculate which zone to snap to, accounting for occupied zones."""
        occupied = self._get_occupied_zones()

        # Determine primary zone
        at_bottom = local_pos.y() >= mdi_rect.height() - self._snap_threshold
        at_top = local_pos.y() <= self._snap_threshold
        at_left = local_pos.x() <= self._snap_threshold
        at_right = local_pos.x() >= mdi_rect.width() - self._snap_threshold

        # Check for quadrant snapping (corners with existing snaps)
        if at_bottom and 'top' in occupied:
            if at_left or local_pos.x() < mdi_rect.width() / 2:
                return 'bottom-left'
            else:
                return 'bottom-right'

        if at_top and 'bottom' in occupied:
            if at_left or local_pos.x() < mdi_rect.width() / 2:
                return 'top-left'
            else:
                return 'top-right'

        if at_left and 'right' in occupied:
            if at_top or local_pos.y() < mdi_rect.height() / 2:
                return 'top-left'
            else:
                return 'bottom-left'

        if at_right and 'left' in occupied:
            if at_top or local_pos.y() < mdi_rect.height() / 2:
                return 'top-right'
            else:
                return 'bottom-right'

        # Primary zones (no conflicts)
        if at_bottom:
            return 'bottom'
        if at_top:
            return 'top'
        if at_left:
            return 'left'
        if at_right:
            return 'right'

        return None

    def start_move(self, event):
        # Start dragging the window
        if event.button() == Qt.MouseButton.LeftButton and not self.resizing:
            self._moving = True
            self._move_start = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )

    def move_window(self, event):
        # Move the window while dragging
        if self._moving:
            self.move(event.globalPosition().toPoint() - self._move_start)

    def end_move(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._moving = False
            # Snap back inside the MDI area if out of bounds
            container = None
            try:
                mdi = self.mdiArea()
                if mdi is not None:
                    container = mdi.viewport()
            except Exception:
                pass
            if container is None:
                container = self.parentWidget()
            if container is not None:
                pw, ph = container.width(), container.height()
                w, h = self.width(), self.height()
                max_x = max(0, pw - w)
                max_y = max(0, ph - h)
                new_x = min(max(self.x(), 0), max_x)
                new_y = min(max(self.y(), 0), max_y)
                if new_x != self.x() or new_y != self.y():
                    self.move(new_x, new_y)

    def mouseMoveEvent(self, event):
        margins = 7  # Resize margin
        rect = self.rect()

        # Map to local coords regardless of source widget
        gp = event.globalPosition().toPoint()
        pos = self.mapFromGlobal(gp)

        x, y = pos.x(), pos.y()
        w, h = rect.width(), rect.height()
        if self._is_dragging and self.main_window is not None:
            mdi_rect = self.main_window.mdi_area.rect()
            try:
                global_pos = event.globalPosition().toPoint()
            except AttributeError:
                global_pos = event.globalPos()

            local_pos = self.main_window.mdi_area.mapFromGlobal(global_pos)

            # Use new zone calculation
            new_zone = self._calculate_snap_zone(local_pos, mdi_rect)

            if new_zone != self._snap_zone:
                self._snap_zone = new_zone
                self._update_snap_preview(mdi_rect)

        # Update cursor shape only when not resizing and only on change
        if not self.resizing:
            desired = Qt.CursorShape.ArrowCursor
            if x < margins and y < margins:
                desired = Qt.CursorShape.SizeFDiagCursor  # Top-left corner
            elif x > w - margins and y < margins:
                desired = Qt.CursorShape.SizeBDiagCursor  # Top-right corner
            elif x < margins and y > h - margins:
                desired = Qt.CursorShape.SizeBDiagCursor  # Bottom-left corner
            elif x > w - margins and y > h - margins:
                desired = Qt.CursorShape.SizeFDiagCursor  # Bottom-right corner
            elif x < margins:
                desired = Qt.CursorShape.SizeHorCursor  # Left edge
            elif x > w - margins:
                desired = Qt.CursorShape.SizeHorCursor  # Right edge
            elif y < margins:
                desired = Qt.CursorShape.SizeVerCursor  # Top edge
            elif y > h - margins:
                desired = Qt.CursorShape.SizeVerCursor  # Bottom edge

            if self.cursor().shape() != desired:
                self.setCursor(QCursor(desired))

        if self.resizing:
            # Non-incremental deltas from the press position
            gp = event.globalPosition().toPoint()
            dx = gp.x() - self._press_global_pos.x()
            dy = gp.y() - self._press_global_pos.y()

            # Skip if no delta to avoid redundant work
            if dx == 0 and dy == 0:
                return

            init_geo = self._initial_geo
            new_x, new_y = init_geo.x(), init_geo.y()
            new_width, new_height = init_geo.width(), init_geo.height()

            if self._resizing_top:
                new_y = init_geo.y() + dy
                new_height = init_geo.height() - dy
            if self._resizing_bottom:
                new_height = init_geo.height() + dy
            if self._resizing_left:
                new_x = init_geo.x() + dx
                new_width = init_geo.width() - dx
            if self._resizing_right:
                new_width = init_geo.width() + dx

            # Enforce minimum size with proper anchoring

            # Min w is 220, min h is 75
            min_w, min_h = 220, 75
            if new_width < min_w:
                if self._resizing_left:
                    new_x = init_geo.x() + (init_geo.width() - min_w)
                new_width = min_w
            if new_height < min_h:
                if self._resizing_top:
                    new_y = init_geo.y() + (init_geo.height() - min_h)
                new_height = min_h

            # Avoid redundant geometry updates
            geo = self.geometry()
            if (new_x, new_y, new_width, new_height) == (
                geo.x(),
                geo.y(),
                geo.width(),
                geo.height(),
            ):
                return

            self.setUpdatesEnabled(False)
            try:
                self.setGeometry(new_x, new_y, new_width, new_height)
            finally:
                self.setUpdatesEnabled(True)

    def mousePressEvent(self, event):
        margins = 10  # Resize margin
        rect = self.rect()

        if event.button() == Qt.MouseButton.LeftButton:
            self._is_dragging = True
            self._drag_start_pos = event.globalPosition().toPoint()
            p = self.mapFromGlobal(event.globalPosition().toPoint())
            x, y = p.x(), p.y()
            w, h = rect.width(), rect.height()

            self._resizing_top = self._resizing_bottom = False
            self._resizing_left = self._resizing_right = False

            if x < margins and y < margins:
                self._resizing_top = True
                self._resizing_left = True  # Top-left corner
            elif x > w - margins and y < margins:
                self._resizing_top = True
                self._resizing_right = True  # Top-right corner
            elif x < margins and y > h - margins:
                self._resizing_bottom = True
                self._resizing_left = True  # Bottom-left corner
            elif x > w - margins and y > h - margins:
                self._resizing_bottom = True
                self._resizing_right = True  # Bottom-right corner
            elif x < margins:
                self._resizing_left = True
            elif x > w - margins:
                self._resizing_right = True
            elif y < margins:
                self._resizing_top = True
            elif y > h - margins:
                self._resizing_bottom = True

            if self.resizing:
                # Store starting global pos and geometry for non-incremental resizing
                self._press_global_pos = event.globalPosition().toPoint()
                self._initial_geo = self.geometry()

    def mouseReleaseEvent(self, event):
        # Stop resizing the window
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._is_dragging:
            self._is_dragging = False

            if self._snap_zone is not None:
                self._apply_snap(self._snap_zone)
            else:
                # Unsnapped - clear snap state
                self._snap_state = None

            self._clear_snap_preview()
            self._snap_zone = None
        if event.button() == Qt.MouseButton.LeftButton:
            self._resizing_top = False
            self._resizing_bottom = False
            self._resizing_left = False
            self._resizing_right = False

    def _update_snap_preview(self, mdi_rect: QtCore.QRect) -> None:
        """Show/update preview box with quadrant support."""
        self._clear_snap_preview()

        if self._snap_zone is None:
            return

        preview_rect = self._calculate_snap_rect(self._snap_zone, mdi_rect)

        self._snap_preview = QtWidgets.QWidget(self.main_window.mdi_area)
        self._snap_preview.setGeometry(preview_rect)
        self._snap_preview.setStyleSheet(
            "background-color: rgba(100, 150, 255, 50); "
            "border: 2px solid rgba(100, 150, 255, 200);"
        )
        self._snap_preview.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._snap_preview.show()
        self._snap_preview.raise_()

    def _calculate_snap_rect(self, zone: str, mdi_rect: QtCore.QRect) -> QtCore.QRect:
        """Calculate snap rectangle for a given zone."""
        margin = 50
        half_w = (mdi_rect.width() - 2 * margin) // 2
        half_h = (mdi_rect.height() - 2 * margin) // 2

        # Full-side snaps
        if zone == 'bottom':
            return QtCore.QRect(margin, mdi_rect.height() // 2,
                                mdi_rect.width() - 2 * margin, half_h)
        elif zone == 'top':
            return QtCore.QRect(margin, margin,
                                mdi_rect.width() - 2 * margin, half_h)
        elif zone == 'left':
            return QtCore.QRect(margin, margin,
                                half_w, mdi_rect.height() - 2 * margin)
        elif zone == 'right':
            return QtCore.QRect(mdi_rect.width() // 2, margin,
                                half_w, mdi_rect.height() - 2 * margin)

        # Quadrant snaps
        elif zone == 'top-left':
            return QtCore.QRect(margin, margin, half_w, half_h)
        elif zone == 'top-right':
            return QtCore.QRect(mdi_rect.width() // 2, margin, half_w, half_h)
        elif zone == 'bottom-left':
            return QtCore.QRect(margin, mdi_rect.height() // 2, half_w, half_h)
        elif zone == 'bottom-right':
            return QtCore.QRect(mdi_rect.width() // 2, mdi_rect.height() // 2,
                                half_w, half_h)

        return QtCore.QRect()

    def _clear_snap_preview(self) -> None:
        """Remove snap preview widget."""
        if self._snap_preview is not None:
            self._snap_preview.deleteLater()
            self._snap_preview = None

    def _apply_snap(self, zone: str) -> None:
        """Apply snap geometry and store snap state."""
        if self.main_window is None:
            return

        mdi_rect = self.main_window.mdi_area.rect()
        target_rect = self._calculate_snap_rect(zone, mdi_rect)

        self._snap_state = zone
        self.setGeometry(target_rect)
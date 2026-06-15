"""
Custom dark title bar for the main window.

On Windows 10 the dark theme leaves a jarring white native title bar. We go
frameless and draw our own dark bar instead — but keep the OS doing resize,
snap and window shadows via the DWM / WM_NCHITTEST trick (the same approach
VS Code, Discord and similar apps use), so we don't have to reimplement those
by hand and break native behaviour.

Off Windows the native frame is fine (no white-bar problem), so
`install_custom_titlebar` is a no-op there and the app keeps its OS frame.

The bar embeds the existing QMenuBar (File/View/Help) plus the "complete-app"
controls the user asked for — Collapse Sidebar, Organize Windows — and themed
minimise / maximise / close buttons. Everything is built from qt.style tokens.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from spyde.qt.style import (
    SURFACE_TITLEBAR, TEXT, TEXT_DIM, BORDER_FAINT, FILL_HOVER, DANGER_HOVER,
    ACCENT_SOFT, FONT_SMALL,
)

_ICONS = (Path(__file__).resolve().parent / "assets" / "icons").as_posix()

# Resize border thickness (px) used by the native hit-test. A bit generous so
# every edge is comfortable to grab, not just the corners.
_BORDER = 8
_TITLEBAR_H = 32


class _WinButton(QtWidgets.QPushButton):
    """Square window-control button (min/max/close) with an SVG glyph."""

    def __init__(self, icon_name: str, parent=None, *, danger: bool = False):
        super().__init__(parent)
        self.setFixedSize(46, _TITLEBAR_H)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setIcon(QtGui.QIcon(f"{_ICONS}/{icon_name}"))
        self.setIconSize(QtCore.QSize(12, 12))
        hover = DANGER_HOVER if danger else FILL_HOVER
        self.setStyleSheet(
            "QPushButton { border: none; background: transparent; }"
            f"QPushButton:hover {{ background: {hover}; }}"
        )


def _feature_button(tooltip: str, icon_name: str, parent) -> QtWidgets.QToolButton:
    """A small flat icon-only button for the title-bar feature actions."""
    btn = QtWidgets.QToolButton(parent)
    btn.setIcon(QtGui.QIcon(f"{_ICONS}/{icon_name}"))
    btn.setIconSize(QtCore.QSize(16, 16))
    btn.setToolTip(tooltip)
    btn.setFixedSize(_TITLEBAR_H, _TITLEBAR_H)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet(
        f"QToolButton {{ background: transparent; border: none; "
        f"border-radius: 4px; }}"
        f"QToolButton:hover {{ background: {ACCENT_SOFT}; }}"
    )
    return btn


class SpydeTitleBar(QtWidgets.QWidget):
    """The dark bar: [icon] [menu] ... [Collapse][Organize] [– □ ✕]."""

    def __init__(self, main_window: "QtWidgets.QMainWindow", menubar: QtWidgets.QMenuBar):
        super().__init__(main_window)
        self._win = main_window
        self.setObjectName("spydeTitleBar")
        self.setFixedHeight(_TITLEBAR_H)
        self.setStyleSheet(
            f"#spydeTitleBar {{ background: {SURFACE_TITLEBAR}; "
            f"border-bottom: 1px solid {BORDER_FAINT}; }}"
        )

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(4)

        # Menu bar at the far left (no app glyph/title — keep the bar clean).
        # Transparent so it sits on the dark bar.
        menubar.setStyleSheet(
            f"QMenuBar {{ background: transparent; color: {TEXT_DIM}; }}"
            f"QMenuBar::item {{ background: transparent; padding: 4px 8px; }}"
            f"QMenuBar::item:selected {{ background: {ACCENT_SOFT}; "
            f"color: {TEXT}; border-radius: 4px; }}"
        )
        menubar.setFixedHeight(_TITLEBAR_H)
        lay.addWidget(menubar)

        lay.addStretch(1)

        # Feature buttons (icon-only, tooltip on hover)
        self.collapse_btn = _feature_button(
            "Collapse Sidebar", "sidebar.svg", self)
        self.organize_btn = _feature_button(
            "Organize Windows", "organize.svg", self)
        lay.addWidget(self.collapse_btn)
        lay.addWidget(self.organize_btn)
        # Clear separation between the feature buttons and the window controls.
        lay.addSpacing(24)

        # Window controls
        self.min_btn = _WinButton("minimize.svg", self)
        self.max_btn = _WinButton("maximize.svg", self)
        self.close_btn = _WinButton("close.svg", self, danger=True)
        self.min_btn.clicked.connect(main_window.showMinimized)
        self.max_btn.clicked.connect(self._toggle_max)
        self.close_btn.clicked.connect(main_window.close)
        for b in (self.min_btn, self.max_btn, self.close_btn):
            lay.addWidget(b)

    def _toggle_max(self):
        if self._win.isMaximized():
            self._win.showNormal()
        else:
            self._win.showMaximized()

    # Double-click the empty bar area to maximise/restore (native behaviour).
    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._toggle_max()
        super().mouseDoubleClickEvent(e)

    # Fallback dragging for platforms without the native hit-test path.
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not _has_native_hittest():
            handle = self._win.windowHandle()
            if handle is not None:
                handle.startSystemMove()
        super().mousePressEvent(e)


def _has_native_hittest() -> bool:
    return sys.platform == "win32"


def install_custom_titlebar(main_window: "QtWidgets.QMainWindow") -> "SpydeTitleBar | None":
    """Replace the native title bar with the dark SpydeTitleBar.

    Returns the bar (so the caller can wire button signals), or None if we
    leave the native frame in place (non-Windows, or offscreen test platform).
    """
    # Skip under pytest: the suite reuses one session MainWindow and pokes
    # win.mdi_area / win.menuBar() directly, so reparenting those into a custom
    # frame breaks that contract. Also keep the native frame off Windows (no
    # white-bar issue there) and under the offscreen CI platform.
    if "pytest" in sys.modules or os.environ.get("SPYDE_NO_CUSTOM_TITLEBAR"):
        return None
    plat = QtGui.QGuiApplication.platformName() if QtGui.QGuiApplication.instance() else ""
    if sys.platform != "win32" or plat == "offscreen":
        return None

    menubar = main_window.menuBar()
    bar = SpydeTitleBar(main_window, menubar)

    # Frameless, but ask Windows (via the event filter below) to keep handling
    # resize/snap/shadow.
    main_window.setWindowFlags(
        main_window.windowFlags() | Qt.WindowType.FramelessWindowHint)

    # Occupy the menu-bar slot: that strip spans the FULL window width above
    # both the dock areas and the central widget — i.e. exactly where a real
    # title bar sits. (Wrapping only the central widget left the bar beside the
    # Plot Control dock, so the window buttons looked offset from it.)
    main_window.setMenuWidget(bar)
    main_window._spyde_titlebar = bar

    # Extend the DWM frame by 1px so Windows keeps drawing the drop shadow and
    # honours snap/Aero for our otherwise-frameless window.
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = int(main_window.winId())
        margins = wintypes.RECT(1, 1, 1, 1)
        ctypes.windll.dwmapi.DwmExtendFrameIntoClientArea(
            hwnd, ctypes.byref(margins))
    except Exception:
        pass
    return bar


# Win32 hit-test result codes.
_HTCLIENT = 1
_HTCAPTION = 2
_HTLEFT, _HTRIGHT, _HTTOP, _HTBOTTOM = 10, 11, 12, 15
_HTTOPLEFT, _HTTOPRIGHT, _HTBOTTOMLEFT, _HTBOTTOMRIGHT = 13, 14, 16, 17


def handle_win_nchittest(main_window, message_ptr):
    """Resolve WM_NCHITTEST for a frameless window so Windows resizes from the
    edges and drags from the empty title-bar area natively.

    Call from MainWindow.nativeEvent on WM_NCHITTEST; returns the HT* code, or
    None to let default handling proceed. Windows-only.
    """
    import ctypes
    from ctypes import wintypes

    msg = ctypes.wintypes.MSG.from_address(int(message_ptr))
    # lParam packs screen x,y (signed 16-bit each).
    x = ctypes.c_short(msg.lParam & 0xFFFF).value
    y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
    pt = main_window.mapFromGlobal(QtCore.QPoint(x, y))
    w, h = main_window.width(), main_window.height()

    if main_window.isMaximized():
        on_left = on_right = on_top = on_bottom = False
    else:
        on_left = pt.x() < _BORDER
        on_right = pt.x() > w - _BORDER
        on_top = pt.y() < _BORDER
        on_bottom = pt.y() > h - _BORDER

    if on_top and on_left:
        return _HTTOPLEFT
    if on_top and on_right:
        return _HTTOPRIGHT
    if on_bottom and on_left:
        return _HTBOTTOMLEFT
    if on_bottom and on_right:
        return _HTBOTTOMRIGHT
    if on_left:
        return _HTLEFT
    if on_right:
        return _HTRIGHT
    if on_top:
        return _HTTOP
    if on_bottom:
        return _HTBOTTOM

    # Drag region: the title bar, but NOT its interactive children (menu /
    # buttons), which must receive their own clicks.
    bar = getattr(main_window, "_spyde_titlebar", None)
    if bar is not None and pt.y() < bar.height():
        child = bar.childAt(bar.mapFrom(main_window, pt))
        if child is None:
            return _HTCAPTION
    return _HTCLIENT

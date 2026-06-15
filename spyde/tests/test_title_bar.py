"""
Custom title bar (spyde.qt.title_bar).

These build the bar directly on a throwaway QMainWindow rather than through the
session fixture — install_custom_titlebar deliberately no-ops under pytest (it
reparents the menu/MDI area, which would break the reused session window), so
here we test the widget + the hit-test helper in isolation. The hit-test test
is Windows-only.
"""
import sys
import pytest
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QMainWindow, QMdiArea

from spyde.qt.title_bar import SpydeTitleBar


def _app():
    return QApplication.instance() or QApplication([])


def _win_with_bar():
    _app()
    win = QMainWindow()
    win.menuBar().addMenu("File")
    win.menuBar().addMenu("View")
    win.setCentralWidget(QMdiArea())
    bar = SpydeTitleBar(win, win.menuBar())
    return win, bar


class TestTitleBar:
    def test_has_all_controls(self):
        win, bar = _win_with_bar()
        for attr in ("collapse_btn", "organize_btn", "min_btn", "max_btn", "close_btn"):
            assert hasattr(bar, attr), f"missing {attr}"
        win.close()

    def test_feature_buttons_emit(self):
        win, bar = _win_with_bar()
        fired = {"collapse": 0, "organize": 0}
        bar.collapse_btn.clicked.connect(lambda: fired.__setitem__("collapse", 1))
        bar.organize_btn.clicked.connect(lambda: fired.__setitem__("organize", 1))
        bar.collapse_btn.click()
        bar.organize_btn.click()
        assert fired == {"collapse": 1, "organize": 1}
        win.close()

    def test_feature_buttons_are_icon_only(self):
        win, bar = _win_with_bar()
        # SVG icons, no text label, tooltips for discoverability
        assert not bar.collapse_btn.icon().isNull()
        assert not bar.organize_btn.icon().isNull()
        assert bar.collapse_btn.text() == ""
        assert bar.organize_btn.text() == ""
        assert bar.collapse_btn.toolTip() == "Collapse Sidebar"
        assert bar.organize_btn.toolTip() == "Organize Windows"
        win.close()


class TestOrganizeWindows:
    """organize_active_windows repositions but must NOT resize (unlike tile)."""

    def test_organize_preserves_sizes(self, monkeypatch):
        _app()
        from spyde.mdi_manager import MDIManager
        mdi = QMdiArea()
        mdi.resize(1000, 800)

        class _FakeWin:
            def __init__(self, w, h):
                self._w, self._h = w, h
                self._x, self._y = 0, 0
            def width(self):
                return self._w
            def height(self):
                return self._h
            def move(self, x, y):
                self._x, self._y = x, y
            def isVisible(self):
                return True

        wins = [_FakeWin(300, 200), _FakeWin(250, 150), _FakeWin(400, 300)]
        mgr = MDIManager.__new__(MDIManager)   # skip __init__ (needs MainWindow)
        mgr.mdi_area = mdi
        monkeypatch.setattr(mgr, "_active_tree_windows", lambda: wins)

        mgr.organize_active_windows()

        # sizes unchanged
        assert [(w.width(), w.height()) for w in wins] == [
            (300, 200), (250, 150), (400, 300)]
        # all moved to a non-negative, non-overlapping-ish layout (first at origin
        # margin, second to its right)
        assert wins[0]._x >= 0 and wins[0]._y >= 0
        assert wins[1]._x > wins[0]._x  # laid left-to-right

    def test_maximize_toggles(self):
        win, bar = _win_with_bar()
        win.show()
        QApplication.processEvents()
        assert not win.isMaximized()
        bar._toggle_max()
        QApplication.processEvents()
        assert win.isMaximized()
        bar._toggle_max()
        QApplication.processEvents()
        assert not win.isMaximized()
        win.close()

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows hit-test only")
    def test_nchittest_regions(self):
        import ctypes
        from PySide6 import QtCore
        from spyde.qt.title_bar import (
            handle_win_nchittest, _HTTOPLEFT, _HTCAPTION, _HTCLIENT)
        win, bar = _win_with_bar()
        win._spyde_titlebar = bar
        win.resize(800, 600)
        win.show()
        QApplication.processEvents()

        def _hit(local_x, local_y):
            g = win.mapToGlobal(QtCore.QPoint(local_x, local_y))
            msg = ctypes.wintypes.MSG()
            msg.message = 0x0084  # WM_NCHITTEST
            msg.lParam = (g.y() << 16) | (g.x() & 0xFFFF)
            return handle_win_nchittest(win, ctypes.addressof(msg))

        assert _hit(1, 1) == _HTTOPLEFT          # corner -> resize
        assert _hit(400, 8) == _HTCAPTION        # empty bar -> drag
        assert _hit(400, 300) == _HTCLIENT       # body -> client
        win.close()

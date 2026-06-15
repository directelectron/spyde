"""Tests for action-owned window lifecycle: close interception, visibility
gates, and protection against re-showing torn-down subwindow shells."""

from PySide6 import QtGui


class TestOnCloseRequest:
    def test_close_request_intercepts_close(self, qtbot, window):
        """A window with on_close_request set must not be torn down by close()."""
        win = window["window"]
        pw = win.add_plot_window(is_navigator=False, signal_tree=None)
        called = []

        def _handler():
            called.append(True)
            pw.hide()

        pw.on_close_request = _handler
        pw.close()

        assert called, "on_close_request was not invoked"
        assert not getattr(pw, "_spyde_closed", False), (
            "close_window must not run when on_close_request intercepts"
        )
        # The inner container must survive so a later show() displays content,
        # not just the title bar.
        assert pw.container.isVisibleTo(pw), (
            "Container was hidden/closed despite close interception"
        )
        pw.close_window()  # explicit cleanup for the session window

    def test_close_without_handler_tears_down(self, qtbot, window):
        win = window["window"]
        pw = win.add_plot_window(is_navigator=False, signal_tree=None)
        pw.close()
        assert getattr(pw, "_spyde_closed", False), (
            "close_window must mark the window as torn down"
        )
        assert pw not in win.plot_subwindows


class TestClosedShellProtection:
    def test_action_binding_does_not_reshow_closed_window(self, qtbot, window):
        """A torn-down window must not be re-shown by its action binding —
        QMdiSubWindow close hides the inner container, so re-showing the shell
        would display only the title bar."""
        from spyde.drawing.toolbars.utils import _bind_action_to_plot_item

        win = window["window"]
        pw = win.add_plot_window(is_navigator=False, signal_tree=None)
        act = QtGui.QAction("Test Action")
        act.setCheckable(True)
        _bind_action_to_plot_item(act, pw)

        pw.close()
        assert not pw.isVisible()

        act.setChecked(True)
        assert not pw.isVisible(), (
            "Action toggle re-showed a closed window shell"
        )


class TestVisibilityGate:
    def test_gate_blocks_and_allows_auto_show(self, qtbot, stem_4d_dataset):
        """_update_3state_visibility must respect a window's visibility_gate."""
        win = stem_4d_dataset["window"]
        nav_window, sig_window = stem_4d_dataset["subwindows"][:2]
        tree = nav_window.signal_tree

        pw = win.add_plot_window(is_navigator=False, signal_tree=tree)
        pw.owner_plot_window = nav_window
        gate_open = [False]
        pw.visibility_gate = lambda: gate_open[0]
        pw.hide()

        win.mdi_area.setActiveSubWindow(nav_window)
        qtbot.wait(100)
        assert not pw.isVisible(), (
            "Window with a closed visibility_gate was auto-shown"
        )

        gate_open[0] = True
        win.mdi_area.setActiveSubWindow(sig_window)
        qtbot.wait(100)
        assert pw.isVisible(), (
            "Window with an open visibility_gate was not auto-shown"
        )
        pw.close_window()

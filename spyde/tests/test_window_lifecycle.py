import pytest
from spyde.drawing.plots.plot_window import PlotWindow


class TestCloseWindowLifecycle:
    def test_close_window_removes_from_tracking(self, qtbot, stem_4d_dataset):
        win = stem_4d_dataset["window"]
        initial_count = len(win.plot_subwindows)
        assert initial_count >= 2

        # Close the first subwindow
        pw = win.plot_subwindows[0]
        pw.close()
        qtbot.waitUntil(
            lambda: len(win.plot_subwindows) < initial_count,
            timeout=3000,
        )
        assert pw not in win.plot_subwindows

    def test_close_nav_window_removes_signal_tree(self, qtbot, stem_4d_dataset):
        win = stem_4d_dataset["window"]
        initial_tree_count = len(win.signal_trees)
        assert initial_tree_count >= 1

        # The nav window is the first subwindow (level-1)
        nav_pw = win.plot_subwindows[0]
        nav_pw.close()
        qtbot.waitUntil(
            lambda: len(win.signal_trees) < initial_tree_count,
            timeout=3000,
        )
        assert len(win.signal_trees) == initial_tree_count - 1


class TestCloseSignalTree:
    """MDIManager.close_signal_tree tears down the WHOLE tree graph, incl.
    action-spawned preview windows, with no orphans."""

    @pytest.fixture(autouse=True)
    def setup(self, stem_4d_dataset, qtbot):
        from PySide6.QtWidgets import QApplication
        self.QApplication = QApplication
        self.win = stem_4d_dataset["window"]
        self.qtbot = qtbot
        self.mdi = self.win.mdi_manager

    def test_close_removes_all_tree_windows_and_entry(self):
        tree = self.win.signal_trees[0]
        # an action-style preview window on the same tree must be torn down too
        preview = self.win.add_plot_window(is_navigator=False, signal_tree=tree)
        self.QApplication.processEvents()
        assert preview in self.mdi.windows_for_tree(tree)

        self.mdi.close_signal_tree(tree)
        self.QApplication.processEvents()

        assert tree not in self.win.signal_trees
        leftover = [pw for pw in self.win.plot_subwindows
                    if getattr(pw, "signal_tree", None) is tree]
        assert leftover == [], f"orphaned windows: {leftover}"

    def test_close_is_idempotent(self):
        tree = self.win.signal_trees[0]
        self.mdi.close_signal_tree(tree)
        self.mdi.close_signal_tree(tree)   # must not raise / double-remove
        assert tree not in self.win.signal_trees

    def test_closing_navigator_tears_down_previews(self):
        tree = self.win.signal_trees[0]
        preview = self.win.add_plot_window(is_navigator=False, signal_tree=tree)
        self.QApplication.processEvents()
        npm = tree.navigator_plot_manager
        if npm is None or not npm.plot_windows:
            pytest.skip("no navigator window for this fixture")
        nav = list(npm.plot_windows.keys())[0]
        nav.close_window()
        self.QApplication.processEvents()
        assert tree not in self.win.signal_trees
        assert preview not in self.win.plot_subwindows

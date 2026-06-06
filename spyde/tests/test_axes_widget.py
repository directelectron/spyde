"""Tests for the Plot Axes sidebar widget."""
from PySide6 import QtWidgets

from spyde.drawing.plots.plot import Plot


class TestAxesWidget:
    def test_axes_widget_populated_on_activation(self, qtbot, stem_4d_dataset):
        """Activating a plot subwindow must populate the Plot Axes sidebar."""
        win = stem_4d_dataset["window"]
        nav_window, sig_window = win.plot_subwindows

        win.mdi_area.setActiveSubWindow(nav_window)
        qtbot.wait(300)

        # The axes_layout must contain at least one QGroupBox
        assert win.axes_layout is not None
        assert win.axes_layout.count() > 0, "axes_layout is empty after subwindow activation"

        # At least one child must be a QGroupBox (Navigation Axes or Signal Axes)
        groups = [
            win.axes_layout.itemAt(i).widget()
            for i in range(win.axes_layout.count())
            if win.axes_layout.itemAt(i).widget() is not None
        ]
        assert any(isinstance(g, QtWidgets.QGroupBox) for g in groups), (
            "No QGroupBox found in axes_layout — axes groups were not added"
        )

    def test_axes_widget_has_navigation_axes_group(self, qtbot, stem_4d_dataset):
        """The axes sidebar must always include a Navigation Axes group."""
        win = stem_4d_dataset["window"]
        nav_window = win.plot_subwindows[0]

        win.mdi_area.setActiveSubWindow(nav_window)
        qtbot.wait(300)

        titles = [
            win.axes_layout.itemAt(i).widget().title()
            for i in range(win.axes_layout.count())
            if isinstance(win.axes_layout.itemAt(i).widget(), QtWidgets.QGroupBox)
        ]
        assert "Navigation Axes" in titles, (
            f"Expected 'Navigation Axes' group, found: {titles}"
        )

    def test_axes_widget_has_signal_axes_for_signal_plot(self, qtbot, stem_4d_dataset):
        """Activating a signal (non-navigator) plot must add a Signal Axes group."""
        win = stem_4d_dataset["window"]
        nav_window, sig_window = win.plot_subwindows

        # Activate the signal plot window
        win.mdi_area.setActiveSubWindow(sig_window)
        qtbot.wait(300)

        titles = [
            win.axes_layout.itemAt(i).widget().title()
            for i in range(win.axes_layout.count())
            if isinstance(win.axes_layout.itemAt(i).widget(), QtWidgets.QGroupBox)
        ]
        assert "Signal Axes" in titles, (
            f"Expected 'Signal Axes' group for signal plot, found: {titles}"
        )

    def test_axes_widget_updates_on_different_signal(self, qtbot, stem_4d_dataset):
        """Switching between nav and signal windows must refresh the axes widget."""
        win = stem_4d_dataset["window"]
        nav_window, sig_window = win.plot_subwindows

        # Activate nav window — should have Navigation Axes but no Signal Axes
        win.mdi_area.setActiveSubWindow(nav_window)
        qtbot.wait(300)
        nav_titles = [
            win.axes_layout.itemAt(i).widget().title()
            for i in range(win.axes_layout.count())
            if isinstance(win.axes_layout.itemAt(i).widget(), QtWidgets.QGroupBox)
        ]

        # Activate signal window — should add Signal Axes
        win.mdi_area.setActiveSubWindow(sig_window)
        qtbot.wait(300)
        sig_titles = [
            win.axes_layout.itemAt(i).widget().title()
            for i in range(win.axes_layout.count())
            if isinstance(win.axes_layout.itemAt(i).widget(), QtWidgets.QGroupBox)
        ]

        assert "Navigation Axes" in nav_titles
        assert "Signal Axes" in sig_titles

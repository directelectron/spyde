from PySide6 import QtCore
from PySide6.QtWidgets import QDockWidget, QVBoxLayout, QWidget


class ControlDockWidget(QDockWidget):
    """
    A dockable widget for controlling stage movements and other live controls.

    This widget is designed to be docked on the left or right side of the main window and
    be composed of a Vertical layout of various control widgets, such as stage controls.


    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Control")
        self.setFloating(True)
        self.setObjectName("ControlDockWidget")
        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.setMinimumWidth(210)

        self.setMaximumWidth(300)
        self.setMinimumHeight(150)
        self.setVisible(False)
        self.setContentsMargins(5, 5, 5, 5)
        # Dark theme, from the shared tokens (was a hardcoded dark block).
        from spyde.qt.style import SURFACE_PANEL, BORDER_FAINT
        self.setStyleSheet(
            f"QDockWidget {{ background-color: {SURFACE_PANEL}; "
            f"border: 1px solid {BORDER_FAINT}; }}"
        )
        # Create a central widget to hold the layout
        central_widget = QWidget()
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)
        self.setWidget(central_widget)

    def add_widget(self, widget):
        self.layout.addWidget(widget)

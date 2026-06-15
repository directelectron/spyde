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
        # Dark theme (was a light-theme leftover: #f0f0f0 in a #141414 app)
        self.setStyleSheet(
            """
            QDockWidget {
                background-color: #141414;
                border: 1px solid rgba(255,255,255,40);
            }
            """
        )
        # Create a central widget to hold the layout
        central_widget = QWidget()
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)
        self.setWidget(central_widget)

    def add_widget(self, widget):
        self.layout.addWidget(widget)

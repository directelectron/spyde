from functools import partial

import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QSpinBox,
    QLabel,
    QPushButton,
    QDialogButtonBox,
)
import hyperspy.api as hs

from PySide6 import QtWidgets

import dask.array as da
import pyxem


class AddNavigator(QDialog):
    """
    Add a navigator to some SignalTree.

    This is used to add a simple navigator like a virtual image (if it isn't automatically loaded).

    It can also be used to load in-situ data which will resample to match the time-stamp of some dataset
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Navigator")
        self.selected_source = None  # (\`"file"\`, path) or (\`"signal"\`, id)

        layout = QVBoxLayout(self)

        self.load_btn = QPushButton("Load from File", self)
        self.select_btn = QPushButton("Select Signal", self)

        layout.addWidget(self.load_btn)
        layout.addWidget(self.select_btn)

        self.load_btn.clicked.connect(self._on_load_from_file)
        self.select_btn.clicked.connect(self._on_select_signal)

    def _on_load_from_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select file", "", "All files (*)"
        )
        if path:
            self.selected_source = ("file", path)
            self.accept()

    def _on_select_signal(self):
        # Stub: implement a signal picker as needed
        self.selected_source = ("signal", None)
        self.accept()

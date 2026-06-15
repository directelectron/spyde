from __future__ import annotations

import json
from typing import TYPE_CHECKING

from PySide6 import QtWidgets
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox

if TYPE_CHECKING:
    pass


class WorkflowViewDialog(QDialog):
    """Read-only dialog showing the steps in a saved workflow."""

    def __init__(self, steps: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Workflow Steps")
        self.resize(480, 320)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("The following transformations will be applied in order:"))

        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        lines = []
        for i, step in enumerate(steps, 1):
            name = step.get("transformation", "<unknown>")
            kwargs = step.get("kwargs", {})
            kw_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            lines.append(f"{i}. {name}({kw_str})")
        text.setPlainText("\n".join(lines))
        layout.addWidget(text)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

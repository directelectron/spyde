"""GPU Status dialog (Help → GPU Status…).

Shows the GPU diagnostics from spyde.gpu_setup and, when an accelerator exists
but torch is CPU-only, offers a "Set up GPU" button that reinstalls the correct
torch wheel via uv on a worker thread (streaming progress).
"""
from __future__ import annotations

import threading

from PySide6 import QtCore, QtWidgets

from spyde import gpu_setup
from spyde.qt.style import make_button, LABEL_QSS, SURFACE_PANEL, BORDER_FAINT


class GpuStatusDialog(QtWidgets.QDialog):
    _line = QtCore.Signal(str)
    _done = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPU Status")
        self.setMinimumWidth(460)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        self._summary = QtWidgets.QLabel()
        self._summary.setStyleSheet(LABEL_QSS)
        self._summary.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._summary.setWordWrap(True)
        v.addWidget(self._summary)

        # uv install log (hidden until a setup run starts)
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(160)
        self._log.setStyleSheet(
            f"background: {SURFACE_PANEL}; border: 1px solid {BORDER_FAINT};")
        self._log.hide()
        v.addWidget(self._log)

        row = QtWidgets.QHBoxLayout()
        self._setup_btn = make_button("Set up GPU", self)
        self._refresh_btn = make_button("Refresh", self)
        self._close_btn = make_button("Close", self)
        row.addWidget(self._setup_btn)
        row.addWidget(self._refresh_btn)
        row.addStretch(1)
        row.addWidget(self._close_btn)
        v.addLayout(row)

        self._setup_btn.clicked.connect(self._on_setup)
        self._refresh_btn.clicked.connect(self._refresh)
        self._close_btn.clicked.connect(self.accept)
        self._line.connect(self._append_log)
        self._done.connect(self._on_setup_done)

        self._refresh()

    def _refresh(self):
        try:
            lines = gpu_setup.summary_lines()
            d = gpu_setup.detect()
        except Exception as e:
            self._summary.setText(f"GPU diagnostics failed: {e}")
            self._setup_btn.setEnabled(False)
            return
        self._summary.setText("\n".join(lines))
        # offer setup only when it would actually help
        self._setup_btn.setEnabled(bool(d.get("needs_gpu_wheel")))
        self._setup_btn.setToolTip(
            "" if d.get("needs_gpu_wheel")
            else "Already using the best available device — nothing to set up.")

    def _append_log(self, text: str):
        self._log.appendPlainText(text)

    def _on_setup(self):
        self._setup_btn.setEnabled(False)
        self._refresh_btn.setEnabled(False)
        self._log.show()
        self._log.clear()
        self._append_log("Installing the GPU-correct torch wheel via uv…")

        def _run():
            res = gpu_setup.ensure_backend(progress=self._line.emit)
            self._done.emit(res)

        threading.Thread(target=_run, daemon=True).start()

    def _on_setup_done(self, res: dict):
        self._append_log("")
        self._append_log(res.get("message", "done"))
        self._refresh_btn.setEnabled(True)
        self._refresh()

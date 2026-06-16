"""In-app update applier for uv-managed installs.

When SpyDE was installed via the NSIS/uv-managed path, an update can be applied
without sending the user to a browser: fetch the new source bundle + lock for
the target tag, drop them into the install root, and `uv sync` (incremental,
GPU-correct wheel). This dialog streams that progress and offers a restart.

Portable / dev builds don't get here — check_for_updates() opens the download
page for those instead.
"""
from __future__ import annotations

import io
import os
import sys
import threading
import urllib.request
import zipfile

from PySide6 import QtCore, QtWidgets

from spyde.qt.style import make_button, SURFACE_PANEL, BORDER_FAINT
from spyde import updater


class UpdateApplyDialog(QtWidgets.QDialog):
    _line = QtCore.Signal(str)
    _done = QtCore.Signal(bool, str)

    def __init__(self, tag: str, source_zip_url: str, parent=None):
        super().__init__(parent)
        self._tag = tag
        self._url = source_zip_url
        self.setWindowTitle(f"Install SpyDE {tag}")
        self.setMinimumWidth(480)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        v.addWidget(QtWidgets.QLabel(
            f"Updating to {tag}. This downloads the new version and rebuilds "
            "the environment via uv (the GPU-correct PyTorch is handled "
            "automatically). You can keep working; restart when it finishes."))

        self._log = QtWidgets.QPlainTextEdit(readOnly=True)
        self._log.setMaximumHeight(200)
        self._log.setStyleSheet(
            f"background: {SURFACE_PANEL}; border: 1px solid {BORDER_FAINT};")
        v.addWidget(self._log)

        row = QtWidgets.QHBoxLayout()
        self._start_btn = make_button("Install Now", self)
        self._restart_btn = make_button("Restart SpyDE", self)
        self._restart_btn.setEnabled(False)
        self._close_btn = make_button("Close", self)
        row.addWidget(self._start_btn)
        row.addWidget(self._restart_btn)
        row.addStretch(1)
        row.addWidget(self._close_btn)
        v.addLayout(row)

        self._start_btn.clicked.connect(self._on_start)
        self._restart_btn.clicked.connect(self._on_restart)
        self._close_btn.clicked.connect(self.reject)
        self._line.connect(self._log.appendPlainText)
        self._done.connect(self._on_done)

    def _on_start(self):
        self._start_btn.setEnabled(False)
        self._line.emit(f"Downloading {self._tag}…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            root = updater._install_root()
            if root is None:
                self._done.emit(False, "not a uv-managed install")
                return
            # 1. fetch + extract the source zip over the install root (keeps the
            #    .venv; only project files change).
            req = urllib.request.Request(
                self._url, headers={"User-Agent": "SpyDE-updater"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            self._line.emit("Extracting…")
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                # GitHub source zips nest everything under a top dir; strip it.
                names = z.namelist()
                top = names[0].split("/")[0] + "/" if names else ""
                for name in names:
                    if name.endswith("/"):
                        continue
                    rel = name[len(top):] if top and name.startswith(top) else name
                    if not rel or rel.startswith(".venv/"):
                        continue
                    dest = os.path.join(root, rel.replace("/", os.sep))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with z.open(name) as src, open(dest, "wb") as f:
                        f.write(src.read())
            # 2. uv sync the new lock.
            self._line.emit("Rebuilding environment (uv sync)…")
            res = updater.apply_uv_sync(progress=self._line.emit)
            self._done.emit(res.get("ok", False), res.get("message", "done"))
        except Exception as e:
            self._done.emit(False, f"update failed: {e}")

    def _on_done(self, ok: bool, message: str):
        self._line.emit("")
        self._line.emit(message)
        self._restart_btn.setEnabled(ok)
        if not ok:
            self._start_btn.setEnabled(True)

    def _on_restart(self):
        # Relaunch via the installed launcher and quit.
        root = updater._install_root()
        try:
            launcher = os.path.join(root, "SpyDE.exe")
            if os.path.exists(launcher):
                os.startfile(launcher)  # noqa: S606 (Windows launcher)
            else:
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()


class PyPIUpgradeDialog(QtWidgets.QDialog):
    """Upgrade a PyPI-installed SpyDE via uv/pip (future PyPI release path)."""
    _line = QtCore.Signal(str)
    _done = QtCore.Signal(bool, str)

    def __init__(self, tag: str, prerelease: bool = False, parent=None):
        super().__init__(parent)
        self._prerelease = prerelease
        self.setWindowTitle(f"Upgrade to SpyDE {tag}")
        self.setMinimumWidth(460)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)
        v.addWidget(QtWidgets.QLabel(
            f"Upgrading SpyDE to {tag} via uv/pip. Restart when it finishes."))
        self._log = QtWidgets.QPlainTextEdit(readOnly=True)
        self._log.setMaximumHeight(200)
        self._log.setStyleSheet(
            f"background: {SURFACE_PANEL}; border: 1px solid {BORDER_FAINT};")
        v.addWidget(self._log)
        row = QtWidgets.QHBoxLayout()
        self._start = make_button("Upgrade Now", self)
        self._close = make_button("Close", self)
        row.addWidget(self._start)
        row.addStretch(1)
        row.addWidget(self._close)
        v.addLayout(row)
        self._start.clicked.connect(self._on_start)
        self._close.clicked.connect(self.reject)
        self._line.connect(self._log.appendPlainText)
        self._done.connect(self._on_done)

    def _on_start(self):
        self._start.setEnabled(False)
        self._line.emit("Upgrading…")

        def _run():
            res = updater.apply_pypi_upgrade(
                prerelease=self._prerelease, progress=self._line.emit)
            self._done.emit(res.get("ok", False), res.get("message", "done"))

        threading.Thread(target=_run, daemon=True).start()

    def _on_done(self, ok: bool, message: str):
        self._line.emit("")
        self._line.emit(message)

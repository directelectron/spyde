from __future__ import annotations

import pytest
from typing import Dict, Union, List, Iterator

from PySide6.QtWidgets import QApplication, QMdiArea

from spyde.qt.shared import open_window as _open_window
from spyde.qt.shared import create_data as _create_data
from spyde.__main__ import MainWindow
from spyde.drawing.plots.plot import Plot
from spyde.signal_tree import BaseSignalTree


# ---------------------------------------------------------------------------
# Session-scoped window: one MainWindow + Dask cluster per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _session_window():
    """Create a single MainWindow for the entire test session.

    Dask startup is asynchronous; we do NOT wait for ``win.client`` here
    because the session fixture runs before any ``qtbot`` is active and
    ``QTest.qWait`` cannot reliably deliver cross-thread queued signals in
    that context.  Instead, each per-test dataset fixture (which has
    ``qtbot``) lets ``MainWindow.create_signal_tree`` handle the wait via its
    own ``while self.client is None: QApplication.processEvents()`` loop.
    """
    win = _open_window()
    yield win
    win.close()


# ---------------------------------------------------------------------------
# Per-test reset: close all subwindows, clear tracking lists
# ---------------------------------------------------------------------------

def _reset_window(win: MainWindow) -> MainWindow:
    """Close all MDI subwindows and clear signal/plot tracking. Returns win."""
    from PySide6.QtTest import QTest

    def _noop(*args, **kwargs):
        pass

    # Patch close_window on all subwindows to prevent cascade errors during removal
    for sw in list(win.mdi_area.subWindowList()):
        try:
            sw.close_window = _noop
        except (AttributeError, RuntimeError):
            pass

    # Close signal trees before clearing so they can do their own cleanup
    for st in list(win.signal_trees):
        try:
            st.close()
        except (AttributeError, RuntimeError):
            pass

    win.plot_subwindows.clear()
    win.signal_trees.clear()
    # Drop the dock-panel rebuild memo: id()s of freed trees/signals can be
    # reused across tests, which would otherwise let a stale key falsely skip a
    # rebuild for a brand-new (same-id) signal.
    dm = getattr(win, "dock_manager", None)
    if dm is not None:
        dm._panels_built_for = None

    # removeSubWindow actually removes from subWindowList(); closeAllSubWindows only hides
    for sw in list(win.mdi_area.subWindowList()):
        try:
            win.mdi_area.removeSubWindow(sw)
            sw.hide()
        except (AttributeError, RuntimeError):
            pass

    # Wait up to 2s for all subwindows to actually close
    deadline = 2000
    elapsed = 0
    while win.mdi_area.subWindowList() and elapsed < deadline:
        QApplication.processEvents()
        QTest.qWait(50)
        elapsed += 50

    QApplication.processEvents()
    return win


# ---------------------------------------------------------------------------
# Dataset fixtures — reuse session window, reset before each test
# ---------------------------------------------------------------------------

@pytest.fixture()
def window(qtbot, _session_window) -> Iterator[Dict[str, Union[MainWindow, QMdiArea, List[Plot], List[BaseSignalTree]]]]:
    win = _reset_window(_session_window)
    qtbot.waitUntil(lambda: win.isVisible(), timeout=2000)
    yield {
        "window": win,
        "mdi_area": win.mdi_area,
        "subwindows": win.mdi_area.subWindowList(),
        "signal_trees": getattr(win, "signal_trees", []),
    }


@pytest.fixture()
def tem_2d_dataset(qtbot, _session_window) -> Iterator[Dict[str, Union[MainWindow, QMdiArea, List[Plot], List[BaseSignalTree]]]]:
    win = _reset_window(_session_window)
    _create_data(win, "Image")
    qtbot.waitUntil(lambda: len(win.mdi_area.subWindowList()) == 1, timeout=5000)
    yield {
        "window": win,
        "mdi_area": win.mdi_area,
        "subwindows": win.mdi_area.subWindowList(),
        "signal_trees": getattr(win, "signal_trees", []),
    }


@pytest.fixture()
def insitu_tem_2d_dataset(qtbot, _session_window) -> Iterator[Dict[str, Union[MainWindow, QMdiArea, List[Plot], List[BaseSignalTree]]]]:
    win = _reset_window(_session_window)
    _create_data(win, "Insitu TEM")
    qtbot.waitUntil(lambda: len(win.mdi_area.subWindowList()) == 2, timeout=5000)
    yield {
        "window": win,
        "mdi_area": win.mdi_area,
        "subwindows": win.mdi_area.subWindowList(),
        "signal_trees": getattr(win, "signal_trees", []),
    }


@pytest.fixture()
def stem_4d_dataset(qtbot, _session_window) -> Iterator[Dict[str, Union[MainWindow, QMdiArea, List[Plot], List[BaseSignalTree]]]]:
    win = _reset_window(_session_window)
    _create_data(win, "4D STEM")
    qtbot.waitUntil(lambda: len(win.mdi_area.subWindowList()) == 2, timeout=5000)
    yield {
        "window": win,
        "mdi_area": win.mdi_area,
        "subwindows": win.mdi_area.subWindowList(),
        "signal_trees": getattr(win, "signal_trees", []),
    }


@pytest.fixture()
def stem_5d_dataset(qtbot, _session_window) -> Iterator[Dict[str, Union[MainWindow, QMdiArea, List[Plot], List[BaseSignalTree]]]]:
    win = _reset_window(_session_window)
    _create_data(win, "5D STEM")
    qtbot.waitUntil(lambda: len(win.mdi_area.subWindowList()) == 3, timeout=10000)
    yield {
        "window": win,
        "mdi_area": win.mdi_area,
        "subwindows": win.mdi_area.subWindowList(),
        "signal_trees": getattr(win, "signal_trees", []),
    }


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """True if nvidia-smi detects at least one GPU."""
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, timeout=3,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False

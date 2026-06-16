from __future__ import annotations
import sys
import os
import json
from collections import deque
from typing import Union
from functools import partial
import webbrowser
from time import perf_counter

from PySide6.QtGui import QAction, QIcon, QBrush
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSplashScreen,
    QMainWindow,
    QApplication,
    QMessageBox,
    QDialog,
    QFileDialog,
)
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QPixmap, QColor

import hyperspy.api as hs
import pyxem.data
from hyperspy.signal import BaseSignal

from spyde.live.camera_control_widget import CameraControlWidget
from spyde.live.control_dock_widget import ControlDockWidget
from spyde.live.particle_scanning import ParticleScanControlWidget
from spyde.live.stage_control_widget import StageControlWidget
from spyde.live.stem_control_widget import StemControlWidget
from spyde.live.reference_control_widget import ReferenceControlWidget
from spyde.misc.dialogs import DatasetSizeDialog, CreateDataDialog, MovieExportDialog, WorkflowViewDialog
from spyde.drawing.plots.plot import Plot
from spyde.drawing.plots.plot_window import PlotWindow
from spyde.signal_tree import BaseSignalTree
from spyde.external.pyqtgraph.histogram_widget import (
    HistogramLUTWidget,
    HistogramLUTItem,
)
from spyde.workers.plot_update_worker import PlotUpdateWorker
from spyde.drawing.colormaps import COLORMAPS
from spyde.dask_manager import DaskManager
from spyde.dock_manager import DockManager
from spyde.mdi_manager import MDIManager
from spyde.qt.style import APP_QSS, SURFACE_CANVAS

SUPPORTED_EXTS = (".hspy", ".zspy", ".mrc", ".tif", ".tiff", ".de5")  # extend as needed


class StartupTimer:
    """Context manager that prints how long a startup step took."""

    def __init__(self, label: str) -> None:
        self.label = label
        self._start = 0.0

    def __enter__(self) -> "StartupTimer":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = (perf_counter() - self._start) * 1000.0
        status = "failed" if exc_type else "completed"
        print(f"[startup] {self.label} {status} in {elapsed_ms:.1f} ms")
        return False


def log_startup_time(label: str) -> StartupTimer:
    """Convenience factory for StartupTimer instances."""
    return StartupTimer(label)


class MainWindow(QMainWindow):
    """
    A class to manage the main window of the application.
    """

    def __init__(self, app=None):
        super().__init__()
        self.btn_reset = None
        self.btn_auto = None
        self.app = app
        self.metadata_group = None  # type: Union[QtWidgets.QGroupBox, None]
        self.metadata_layout = None  # type: Union[QtWidgets.QVBoxLayout, None]

        self.axes_group = None  # type: Union[QtWidgets.QGroupBox, None]
        self.axes_layout = None  # type: Union[QtWidgets.QVBoxLayout, None]
        cpu_count = os.cpu_count()
        print("CPU Count:", cpu_count)
        if cpu_count is None or cpu_count < 4:
            workers = 1  # Don't overdo it on small systems
            threads_per_worker = 1
        else:
            # take roughly 3/4s of the available cores
            if cpu_count <= 16:
                workers = (cpu_count // 2) - 1
                threads_per_worker = 2
            else:
                workers = (cpu_count // 4) - 1  # For very large systems, limit workers
                threads_per_worker = 4
        # get screen size and set window size to 3/4 of the screen size
        self.dock_widget = None
        self.control_widget = None

        screen = QApplication.primaryScreen()
        self.screen_size = screen.size()
        self.resize(
            self.screen_size.width() * 3 // 4, self.screen_size.height() * 3 // 4
        )
        self.histogram = None
        self._histogram_image_item = None  # track bound ImageItem to avoid LUT resets

        # center the main window on the screen
        self.move(
            (self.screen_size.width() - self.width()) // 2,
            (self.screen_size.height() - self.height()) // 2,
        )
        # create an MDI area
        self.mdi_area = QtWidgets.QMdiArea()
        self.mdi_area.setBackground(QBrush(QColor(SURFACE_CANVAS)))
        self.setCentralWidget(self.mdi_area)

        # settings and recent menu
        self.settings = QtCore.QSettings("spyde", "SpyDE")
        self.recent_menu = None

        self._pending_signal_queue: deque = deque()  # thread-safe deque for cross-thread signal delivery
        # Temporary empty list; replaced by mdi_manager.plot_subwindows after MDIManager is created
        self.plot_subwindows: list["PlotWindow"] = []
        self.signal_trees: list["BaseSignalTree"] = []

        self.create_menu()
        self.setMouseTracking(True)

        self.selectors_layout = None
        self.s_list_widget = None
        self.file_dialog = None

        # Start a background worker thread to poll plot Futures
        self._update_thread = QtCore.QThread(self)
        self._plot_update_worker = PlotUpdateWorker(
            lambda: [p for plots in self.plot_subwindows for p in plots.plots],
            interval_ms=5,
        )
        self._plot_update_worker.moveToThread(self._update_thread)
        self._update_thread.started.connect(self._plot_update_worker.start)
        self._plot_update_worker.plot_ready.connect(self.on_plot_future_ready)
        self._plot_update_worker.signal_ready.connect(self.on_signal_future_ready)
        self._plot_update_worker.debug_print.connect(lambda msg: print(msg))

        with log_startup_time("Plot update worker thread start"):
            self._update_thread.start()

        if self.app is not None:
            # Use Fusion style on non-macOS
            if sys.platform != "darwin":
                QtWidgets.QApplication.setStyle("Fusion")
                with log_startup_time("Apply application stylesheet"):
                    # Single source of truth for the theme: spyde/qt/style.py
                    self.app.setStyleSheet(APP_QSS)
        else:
            self.mdi_area.setStyleSheet(f"background-color: {SURFACE_CANVAS};")

        self.current_selected_signal_tree = None  # type: Union[BaseSignalTree, None]
        self._pending_navigator_assignment = None
        with log_startup_time("Plot control dock creation"):
            self.dock_manager = DockManager(main_window=self, parent=self)
            # expose selectors_layout for Plot.show_selector_control_widget compatibility
            self.selectors_layout = self.dock_manager.selectors_layout
            self.histogram = self.dock_manager.histogram
            self.cmap_selector = self.dock_manager.cmap_selector
            self.metadata_layout = self.dock_manager.metadata_layout
            self.axes_layout = self.dock_manager.axes_layout
            self.btn_auto = self.dock_manager.btn_auto
            self.btn_reset = self.dock_manager.btn_reset
            # expose dock/control widgets for backward compat
            self.dock_widget = self.dock_manager.dock_widget
            self.control_widget = self.dock_manager.control_widget


        self.cursor_readout = QtWidgets.QLabel("x: -, y: -, value: -")
        self.statusBar().addPermanentWidget(self.cursor_readout)

        self.mdi_manager = MDIManager(mdi_area=self.mdi_area, main_window=self, parent=self)
        # Share the lists so existing code can still do win.plot_subwindows / win.signal_trees
        self.plot_subwindows = self.mdi_manager.plot_subwindows
        self.signal_trees = self.mdi_manager.signal_trees
        self.mdi_manager.subwindow_activated.connect(self.dock_manager.on_active_plot_changed)

        print(f"Starting Dask LocalCluster with {workers} workers, {threads_per_worker} threads per worker")
        self.dask_manager = DaskManager(
            n_workers=workers,
            threads_per_worker=threads_per_worker,
            parent=self,
        )
        self.dask_manager.ready.connect(self._on_dask_ready)
        self.dask_manager.start()
        if self.app is not None:
            self.app.aboutToQuit.connect(self.dask_manager.shutdown)
            self.app.aboutToQuit.connect(self._shutdown_update_thread)

        # Custom dark title bar (Windows only; no-op elsewhere / offscreen).
        # Replaces the white native Win10 bar and hosts the menu plus the
        # Collapse Sidebar / Organize Windows controls.
        self._spyde_titlebar = None
        from spyde.qt.title_bar import install_custom_titlebar
        bar = install_custom_titlebar(self)
        if bar is not None:
            bar.collapse_btn.clicked.connect(self._toggle_sidebar)
            bar.organize_btn.clicked.connect(self.organize_active_windows)

        # Check for updates shortly after startup (non-blocking, silent unless
        # an update is found). Disable via SPYDE_NO_UPDATE_CHECK or QSettings.
        if (self.app is not None
                and not os.environ.get("SPYDE_NO_UPDATE_CHECK")
                and self.settings.value("updates/check_on_startup", True, type=bool)):
            QtCore.QTimer.singleShot(
                2500, lambda: self.check_for_updates(silent=True))

    def _toggle_sidebar(self):
        """Collapse / restore the Plot Control sidebar from the title bar."""
        self.dock_manager.toggle_plot_control()

    def nativeEvent(self, event_type, message):
        """Frameless-window native handling on Windows so the OS still does
        edge-resize, snap and Aero shadow (delegated to title_bar). No-op when
        the custom title bar isn't installed.

        - WM_NCCALCSIZE: claim the full window as client area (hides the native
          frame while keeping WS_THICKFRAME for native resize/snap).
        - WM_NCHITTEST: report the resize edges / draggable caption.
        """
        if (getattr(self, "_spyde_titlebar", None) is not None
                and event_type == b"windows_generic_MSG"):
            try:
                import ctypes
                from spyde.qt.title_bar import (
                    WM_NCCALCSIZE, WM_NCHITTEST,
                    handle_win_nccalcsize, handle_win_nchittest)
                msg = ctypes.wintypes.MSG.from_address(int(message))
                if msg.message == WM_NCCALCSIZE and msg.wParam:
                    if handle_win_nccalcsize(message):
                        return True, 0
                elif msg.message == WM_NCHITTEST:
                    ht = handle_win_nchittest(self, message)
                    if ht is not None:
                        return True, ht
            except Exception:
                pass
        return super().nativeEvent(event_type, message)

    @QtCore.Slot()
    def _on_dask_ready(self):
        print("MainWindow: Dask ready.")

    @property
    def plots(self) -> list[Plot]:
        """Get a flat list of all Plot instances in all plot windows."""
        all_plots = []
        for pw in self.plot_subwindows:
            all_plots.extend(pw.plots)
        return all_plots

    @property
    def navigation_selectors(self):
        selectors = []
        for s in self.signal_trees:
            if s.navigator_plot_manager is not None:
                selectors.extend(s.navigator_plot_manager.all_navigation_selectors)
        return selectors

    @QtCore.Slot(object, object, object)
    def on_plot_future_ready(self, plot: Plot, result: object, future: object) -> None:
        """
        Receive finished compute results from the worker and apply them on the GUI thread.

        Parameters:
            plot: Plot to update.
            result: Either the computed data or an Exception.
            future: The Future object that completed (used for identity staleness check).
        """
        if isinstance(result, Exception):
            print(f"Plot update failed: {result}")
            return
        try:
            # Use identity (`is`) not id() — id() can be reused by GC for a newer
            # future allocated at the same address, falsely passing the staleness check.
            if plot.current_data is not future:
                return
            plot.current_data = result
            plot.update()
        except Exception as e:
            print(f"Failed to update plot: {e}")

    @QtCore.Slot(object, object, object)
    def on_signal_future_ready(self,
                               signal: BaseSignal,
                               result: object,
                               plot: Plot) -> None:
        """
        Receive finished compute results from the worker and apply them on the GUI thread.

        Parameters:


        Parameters
        ----------
        signal: Signal to update.
        result: Either the computed data or an Exception.
        plot:
            The targeted plot to update
        """
        if isinstance(result, Exception):
            print(f"signal update failed: {result}")
            return
        try:
            signal.data = result
            signal._lazy = False
            signal._assign_subclass()
            plot.parent_selector.delayed_update_data(update_contrast=True, force=True)
        except Exception as e:
            print(f"Failed to update signal: {e}")


    def create_menu(self):
        """
        Create the menu bar for the main window.
        """
        menubar = self.menuBar()

        # Add File Menu
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        open_create_data_dialog = QAction("Create Data...", self)

        open_create_data_dialog.triggered.connect(self.create_data)
        file_menu.addAction(open_create_data_dialog)

        self.recent_menu = file_menu.addMenu("Open Recent")
        self._update_recent_menu()

        example_data = file_menu.addMenu("Load Example Data...")

        names = [
            "mgo_nanocrystals",
            "small_ptychography",
            "zrnb_precipitate",
            "pdcusi_insitu",
            "sped_ag",
            "fe_multi_phase_grains",

        ]
        for n in names:
            action = example_data.addAction(n)
            action.triggered.connect(partial(self.load_example_data, n))

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_current_signal)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        save_workflow_action = QAction("Save Workflow...", self)
        save_workflow_action.triggered.connect(self.save_workflow)
        file_menu.addAction(save_workflow_action)

        apply_workflow_action = QAction("Apply Workflow...", self)
        apply_workflow_action.triggered.connect(self.apply_workflow)
        file_menu.addAction(apply_workflow_action)

        file_menu.addSeparator()

        export_file = QAction("Export Current Signal...", self)
        export_file.triggered.connect(self.export_current_signal)
        file_menu.addAction(export_file)

        # Add View Menu
        view_menu = menubar.addMenu("View")

        # Add a view to open the dask dashboard
        view_dashboard_action = QAction("Open Dask Dashboard", self)
        view_dashboard_action.triggered.connect(self.open_dask_dashboard)
        view_menu.addAction(view_dashboard_action)

        view_plot_control_action = QAction("Toggle Plot Control Dock", self)
        view_plot_control_action.triggered.connect(lambda: self.dock_manager.toggle_plot_control())
        view_menu.addAction(view_plot_control_action)

        view_camera_control_action = QAction("Toggle Instrument Control Dock", self)
        view_camera_control_action.triggered.connect(lambda: self.dock_manager.toggle_instrument_control())
        view_menu.addAction(view_camera_control_action)

        tile_action = QAction("Tile Active Windows", self)
        tile_action.triggered.connect(self.tile_active_windows)
        tile_action.setShortcut("Ctrl+T")
        view_menu.addAction(tile_action)

        # Add Help Menu
        help_menu = menubar.addMenu("Help")

        check_updates_action = QAction("Check for Updates…", self)
        check_updates_action.triggered.connect(
            lambda: self.check_for_updates(silent=False))
        help_menu.addAction(check_updates_action)

        gpu_status_action = QAction("GPU Status…", self)
        gpu_status_action.triggered.connect(self.show_gpu_status)
        help_menu.addAction(gpu_status_action)

        help_menu.addSeparator()

        about_action = QAction("About SpyDE", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_gpu_status(self):
        from spyde.misc.dialogs.gpu_status_dialog import GpuStatusDialog
        GpuStatusDialog(parent=self).exec()

    def show_about(self):
        from spyde._build_info import build_info
        from spyde import gpu_setup
        info = build_info()
        try:
            backend = gpu_setup.detect()["backend"].upper()
        except Exception:
            backend = "?"
        QMessageBox.about(
            self, "About SpyDE",
            f"<b>SpyDE</b> {info['version']}<br>"
            f"build {info['git_sha']} · {info['channel']}<br>"
            f"compute backend: {backend}<br><br>"
            "Visualization &amp; analysis for electron microscopy "
            "(TEM / STEM / 4D-STEM / EELS).")

    def check_for_updates(self, silent: bool = False) -> None:
        """Check GitHub Releases for a newer version on a worker thread.

        silent=True (startup): only surfaces a prompt if an update is found and
        the user hasn't skipped that version. silent=False (Help menu): always
        reports the outcome.
        """
        class _Relay(QtCore.QObject):
            done = QtCore.Signal(object)
        relay = _Relay(self)
        self._update_relay = relay

        def _present(info):
            from spyde.updater import UpdateInfo  # noqa: F401
            if info.error and not silent:
                QMessageBox.information(
                    self, "Check for Updates",
                    f"Could not check for updates.\n\n{info.error}")
                return
            if not info.available:
                if not silent:
                    QMessageBox.information(
                        self, "Check for Updates",
                        f"SpyDE {info.current} is up to date.")
                return
            # An update is available.
            skipped = self.settings.value("updates/skip_version", "", type=str)
            if silent and skipped == info.latest:
                return
            from spyde.updater import is_uv_managed
            can_apply = is_uv_managed()
            box = QMessageBox(self)
            box.setWindowTitle("Update Available")
            box.setText(
                f"SpyDE {info.latest} is available "
                f"(you have {info.current}).")
            if info.notes:
                box.setDetailedText(info.notes)
            if can_apply:
                get_btn = box.addButton("Install & Restart",
                                        QMessageBox.AcceptRole)
            else:
                get_btn = box.addButton("Get Update", QMessageBox.AcceptRole)
            box.addButton("Remind Me Later", QMessageBox.RejectRole)
            skip_btn = box.addButton("Skip This Version",
                                     QMessageBox.DestructiveRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is get_btn:
                if can_apply:
                    self._apply_update(info)
                else:
                    import webbrowser
                    webbrowser.open(info.url or
                                    "https://github.com/CSSFrancis/spyde/releases")
            elif clicked is skip_btn:
                self.settings.setValue("updates/skip_version", info.latest)

        relay.done.connect(_present)

        def _run():
            try:
                from spyde.updater import check
                relay.done.emit(check())
            except Exception as e:
                from spyde.updater import UpdateInfo
                relay.done.emit(UpdateInfo(
                    available=False, current="?", error=str(e)))

        import threading
        threading.Thread(target=_run, daemon=True).start()

    def _apply_update(self, info) -> None:
        """Open the in-app applier (uv-managed installs only)."""
        from spyde.misc.dialogs.update_dialog import UpdateApplyDialog
        from spyde.updater import GITHUB_REPO
        tag = info.latest
        # GitHub source archive for the tag → unpacked over the install root.
        src_zip = f"https://github.com/{GITHUB_REPO}/archive/refs/tags/{tag}.zip"
        UpdateApplyDialog(tag, src_zip, parent=self).exec()

    def export_current_signal(self):
        plot = self._active_plot()
        if not isinstance(plot, Plot):
            QMessageBox.warning(self, "Error", "No active plot window to export from.")
            return
        MovieExportDialog(plot=plot, parent=self).exec()

    # ── Save current signal ──────────────────────────────────────────────────

    def _current_signal(self):
        """Return the HyperSpy signal currently displayed in the active plot, or None."""
        plot = self._active_plot()
        if not isinstance(plot, Plot):
            return None
        if plot.plot_state is None:
            return None
        return plot.plot_state.current_signal

    def save_current_signal(self):
        """File → Save: write the currently displayed signal to .hspy or .zspy."""
        signal = self._current_signal()
        if signal is None:
            QMessageBox.warning(self, "Save", "No active signal to save.")
            return

        suggested = ""
        title = signal.metadata.get_item("General.title", default="")
        if title:
            suggested = title.replace(" ", "_") + ".hspy"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Signal",
            suggested,
            "HyperSpy Files (*.hspy);;Zarr Files (*.zspy);;All Files (*)",
        )
        if not path:
            return

        try:
            import dask
            with dask.config.set(scheduler="synchronous"):
                signal.save(path, overwrite=True)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return

        QMessageBox.information(self, "Saved", f"Signal saved to:\n{path}")

    # ── Workflow save / apply ────────────────────────────────────────────────

    def _workflow_steps_for_signal(self, signal) -> list[dict] | None:
        """
        Walk the signal tree to find the node for *signal* and collect the
        ordered chain of transformations from root → that node.

        Returns a list of step dicts, or None if the node isn't found.
        """
        for tree in self.signal_trees:
            node = tree.get_node(signal)
            if node is None:
                continue
            # Collect path from node back to root
            steps = []
            current = node
            while current is not None and current.parent is not None:
                step = {
                    "transformation": current.transformation or current.name,
                    "args": list(current.args),
                    "kwargs": current.kwargs,
                }
                steps.append(step)
                current = current.parent
            steps.reverse()
            return steps
        return None

    def save_workflow(self):
        """File → Save Workflow: serialize the transformation chain to a JSON file."""
        signal = self._current_signal()
        if signal is None:
            QMessageBox.warning(self, "Save Workflow", "No active signal to derive a workflow from.")
            return

        steps = self._workflow_steps_for_signal(signal)
        if steps is None:
            QMessageBox.warning(self, "Save Workflow", "Could not locate signal in any signal tree.")
            return
        if not steps:
            QMessageBox.information(self, "Save Workflow", "This signal has no transformations — it is the root signal.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Workflow",
            "workflow.json",
            "Workflow Files (*.json);;All Files (*)",
        )
        if not path:
            return

        workflow = {"version": 1, "steps": steps}
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(workflow, fh, indent=2, default=str)
        except Exception as e:
            QMessageBox.critical(self, "Save Workflow failed", str(e))
            return

        QMessageBox.information(self, "Saved", f"Workflow ({len(steps)} step(s)) saved to:\n{path}")

    def apply_workflow(self):
        """File → Apply Workflow: load a workflow JSON and apply it to the active signal."""
        signal = self._current_signal()
        if signal is None:
            QMessageBox.warning(self, "Apply Workflow", "Open a signal first, then apply a workflow to it.")
            return

        # Find which tree owns this signal so we can call add_transformation
        owning_tree = None
        for tree in self.signal_trees:
            if tree.get_node(signal) is not None:
                owning_tree = tree
                break
        if owning_tree is None:
            QMessageBox.warning(self, "Apply Workflow", "Could not locate signal in any signal tree.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Workflow",
            "",
            "Workflow Files (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                workflow = json.load(fh)
        except Exception as e:
            QMessageBox.critical(self, "Load Workflow failed", str(e))
            return

        steps = workflow.get("steps", [])
        if not steps:
            QMessageBox.information(self, "Apply Workflow", "The workflow file contains no steps.")
            return

        # Show a preview dialog so the user can confirm before applying
        dlg = WorkflowViewDialog(steps=steps, parent=self)
        if dlg.exec() != WorkflowViewDialog.DialogCode.Accepted:
            return

        # Apply each step in sequence; each step's output becomes the input of the next
        current_signal = signal
        errors = []
        for step in steps:
            transformation = step.get("transformation")
            kwargs = step.get("kwargs", {})
            if not transformation:
                continue
            try:
                current_signal = owning_tree.add_transformation(
                    current_signal,
                    method=transformation,
                    **kwargs,
                )
                if current_signal is None:
                    errors.append(f"'{transformation}' returned None — stopping.")
                    break
            except Exception as e:
                errors.append(f"'{transformation}': {e}")
                break

        if errors:
            QMessageBox.warning(
                self,
                "Workflow errors",
                "Workflow stopped early:\n" + "\n".join(errors),
            )
        else:
            QMessageBox.information(
                self,
                "Workflow applied",
                f"Applied {len(steps)} transformation(s) to the signal.",
            )

    ### Handling Recent File opens ###

    def _add_to_recent(self, path: str) -> None:
        """Add a path to the recent-files list (persisted via QSettings)."""
        try:
            recent = self.settings.value("recentFiles", [])
            # QSettings may return a single string if only one item stored
            if isinstance(recent, str):
                recent = [recent]
            recent = list(recent or [])
            if path in recent:
                recent.remove(path)
            recent.insert(0, path)
            # cap recent list
            recent = recent[:10]
            self.settings.setValue("recentFiles", recent)
            self._update_recent_menu()
        except Exception:
            pass

    def _update_recent_menu(self) -> None:
        """Rebuild the Open Recent submenu from QSettings."""
        if self.recent_menu is None:
            return
        self.recent_menu.clear()
        recent = self.settings.value("recentFiles", [])
        if isinstance(recent, str):
            recent = [recent]
        recent = list(recent or [])
        if not recent:
            act = QAction("No recent files", self)
            act.setEnabled(False)
            self.recent_menu.addAction(act)
            return
        for path in recent:
            # show only the filename in the menu but keep full path in the triggered slot
            display = os.path.basename(path) if os.path.basename(path) else path
            act = QAction(display, self)
            act.setToolTip(path)
            act.triggered.connect(partial(self.open_recent, path))
            self.recent_menu.addAction(act)
        self.recent_menu.addSeparator()
        clear_act = QAction("Clear Recent", self)
        clear_act.triggered.connect(self._clear_recent)
        self.recent_menu.addAction(clear_act)

    def open_recent(self, path: str) -> None:
        """Open a recent file (called by the recent menu actions)."""
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Open Recent", f"File not found: {path}")
            # remove missing entry
            try:
                recent = self.settings.value("recentFiles", [])
                if isinstance(recent, str):
                    recent = [recent]
                recent = list(recent or [])
                if path in recent:
                    recent.remove(path)
                    self.settings.setValue("recentFiles", recent)
                    self._update_recent_menu()
            except Exception:
                pass
            return
        # reuse existing loading path
        self._create_signals([path])
        # ensure the opened file is moved to the top of recent
        self._add_to_recent(path)

    def _clear_recent(self) -> None:
        """Clear the recent-files list."""
        try:
            self.settings.remove("recentFiles")
            self._update_recent_menu()
        except Exception:
            pass

    def open_dask_dashboard(self) -> None:
        """
        Open the Dask dashboard in a new window.
        """
        if self.dask_manager.client:
            dashboard_url = self.dask_manager.client.dashboard_link
            webbrowser.open(dashboard_url)
        else:
            QMessageBox.warning(self, "Error", "Dask client is not initialized.")

    def create_data(self):
        dialog = CreateDataDialog(self)
        print("Creating Data")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            print("Dialog accepted")
            data, navigators = dialog.get_data()
            print("Data created")
            if data is not None:
                self.add_signal(data, navigators=navigators)

    def _create_signals(self, file_paths: list[str]) -> None:
        """Internal helper to load multiple file paths into signals and add them."""
        for file_path in file_paths:
            kwargs = {"lazy": True}
            if file_path.endswith(".mrc"):
                dialog = DatasetSizeDialog(self, filename=file_path)
                print("Opening Dataset Size Dialog for .mrc file")
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    x_size = dialog.x_input.value()
                    y_size = dialog.y_input.value()
                    time_size = dialog.time_input.value()
                    kwargs["navigation_shape"] = tuple(
                        [val for val in (x_size, y_size, time_size) if val > 1]
                    )
                    print(f"{kwargs['navigation_shape']}")
                else:
                    print("Dialog cancelled")
                    return
                # .mrc always have 2 signal axes.  Maybe needs changed for eels.
                if len(kwargs["navigation_shape"]) == 3:
                    kwargs["chunks"] = (
                        (1,) + ("auto",) * (len(kwargs["navigation_shape"]) - 1)
                    ) + (-1, -1)
                else:
                    kwargs["chunks"] = (("auto",) * len(kwargs["navigation_shape"])) + (
                        -1,
                        -1,
                    )

                print(f"chunks: {kwargs['chunks']}")
            if "navigation_shape" in kwargs and kwargs["navigation_shape"] == ():
                kwargs.pop("navigation_shape")
                kwargs.pop("chunks")
            print("Loading signal from file:", file_path, "with kwargs:", kwargs)
            import os as _os
            self.statusBar().showMessage(
                f"Loading {_os.path.basename(file_path)}…")
            QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
            try:
                # tifffile-backed lazy arrays embed an open BufferedReader that
                # cannot be pickled by Dask's distributed scheduler.  Load the
                # TIFF eagerly and then convert to a serializable dask array via
                # as_lazy(), which calls da.from_array on the in-memory numpy data.
                is_tiff = file_path.lower().endswith((".tif", ".tiff"))
                if is_tiff:
                    tiff_kwargs = {k: v for k, v in kwargs.items() if k != "lazy"}
                    signal = hs.load(file_path, lazy=False, **tiff_kwargs)
                    signal = signal.as_lazy()
                else:
                    signal = hs.load(file_path, **kwargs)
                # fix MRC loading in rsciio.
                if (signal.axes_manager.signal_dimension + signal.axes_manager.navigation_dimension) == 2:
                    signal = signal.transpose(2)
                if kwargs.get("lazy", False) or is_tiff:
                    if signal.axes_manager.navigation_dimension == 1:
                        signal.cache_pad = 3
                    elif signal.axes_manager.navigation_dimension == 2:
                        signal.cache_pad = 2
                print("Signal loaded:", signal)
                print("Signal shape:", signal.data.shape)
                print("Signal Chunks:", signal.data.chunks)
            finally:
                # Load done (or failed); drop our wait cursor. add_signal sets
                # its own around the tree build.
                QApplication.restoreOverrideCursor()
            self.add_signal(signal)
            try:
                self._add_to_recent(file_path)
            except Exception:
                print("Failed to add to recent files list")

    def open_file(self):
        self.file_dialog = QFileDialog()
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        self.file_dialog.setNameFilter("Supported Files (*.hspy *.zspy *.mrc *.tif *.tiff);;"
                                       "HyperSpy Files (*.hspy *.zspy);;"
                                       "mrc Files (*.mrc);;"
                                       "TIFF Files (*.tif *.tiff)")

        if self.file_dialog.exec():
            file_paths = self.file_dialog.selectedFiles()
            if file_paths:
                self._create_signals(file_paths)

    def add_signal(self, signal, navigators=None, selector_type=None,
                   navigator_override=None) -> None:
        """Add a signal to the main window.

        This will "plant" a new seed for a signal tree and set up the associated plots.

        Parameters
        ----------
        signal : hs.signals.BaseSignal
            The hyperspy signal to add.

        """
        print("Creating Signal Tree for signal")

        # Wait for the Dask client if it's still starting — but without the old
        # modal "Please wait" dialog that made the app feel frozen. A wait
        # cursor + status message keeps it responsive (events still pump) and
        # explains the delay. add_signal stays synchronous, which the
        # programmatic/open paths and tests rely on.
        QApplication.setOverrideCursor(Qt.CursorShape.BusyCursor)
        try:
            if self.dask_manager.client is None:
                self.statusBar().showMessage(
                    "Starting compute backend… your data will open shortly.")
                while self.dask_manager.client is None:
                    QApplication.processEvents()
            self.statusBar().showMessage("Opening signal…")
            signal_tree = BaseSignalTree(
                root_signal=signal, main_window=self,
                distributed_client=self.dask_manager.client,
                selector_type=selector_type, navigator_override=navigator_override,
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()
        self.signal_trees.append(signal_tree)
        print("Signal Tree Created")
        if navigators is not None:
            for i, nav in enumerate(navigators):
                title = nav.metadata.get_item(
                    "General.title", default="navigation_" + str(i)
                )
                if title == "":
                    title = "navigation_" + str(i)
                print("Adding navigator signal:", title)
                signal_tree.add_navigator_signal(title, nav)

        if signal.metadata.get_item("General.virtual_images", False):
            for key, item in signal.metadata.General.virtual_images:
                print("Adding virtual image navigator signal:", key)
                signal_tree.add_navigator_signal(key, item)

    @QtCore.Slot(object)
    def _add_signal_from_thread(self, signal):
        """Thread-safe slot to add a committed virtual image signal."""
        self.add_signal(signal)

    @QtCore.Slot()
    def _flush_pending_signals(self):
        """Drain the thread-safe pending-signal queue on the GUI thread."""
        while self._pending_signal_queue:
            sig = self._pending_signal_queue.popleft()
            self.add_signal(sig)

    def load_example_data(self, name):
        """
        Load example data for testing purposes.
        """
        signal = getattr(pyxem.data, name)(allow_download=True, lazy=True)

        if name == "sped_ag":
            signal.axes_manager.signal_axes.set(offset =-0.374196254*4, scale=0.00668207597*4)
        self.add_signal(signal)
        print("Example data loaded:", name)

    def _auto_position_near_owner(self, pw: "PlotWindow") -> None:
        """Delegate to MDIManager."""
        self.mdi_manager.auto_position_near_owner(pw)

    def tile_active_windows(self) -> None:
        self.mdi_manager.tile_active_windows()

    def organize_active_windows(self) -> None:
        """Title-bar "Organize": reposition windows into a grid, keep their
        sizes (unlike Tile, which resizes them)."""
        self.mdi_manager.organize_active_windows()

    def add_plot_window(
        self,
        is_navigator: bool = False,
        plot_manager=None,
        signal_tree=None,
        *args,
        **kwargs,
    ) -> "PlotWindow":
        return self.mdi_manager.add_plot_window(
            is_navigator=is_navigator,
            plot_manager=plot_manager,
            signal_tree=signal_tree,
        )

    def set_cursor_readout(
        self, x=None, y=None, xpix=None, ypix=None, value=None
    ) -> None:
        """Update status bar readout with cursor coordinates and data value."""

        def _fmt(v):
            if v is None:
                return "-"
            try:
                return f"{float(v):.4g}"
            except Exception:
                return str(v)

        txt = f"x: {_fmt(x)} ({xpix}), y: {_fmt(y)} ({ypix}), value: {_fmt(value)}"
        if hasattr(self, "cursor_readout") and self.cursor_readout is not None:
            self.cursor_readout.setText(txt)




    def on_subwindow_activated(self, window: "PlotWindow") -> None:
        """Delegate to MDIManager (called from Plot and PlotWindow)."""
        self.mdi_manager._on_subwindow_activated(window)

    # ── Navigator drag/drop delegators (used by tests and legacy callers) ────

    def navigator_enter(self) -> None:
        self.mdi_manager._navigator_enter()

    def navigator_move(self, pos) -> None:
        self.mdi_manager._navigator_move(pos)

    def navigator_leave(self) -> None:
        self.mdi_manager._navigator_leave()

    def navigator_drop(self, pos, mime_data) -> None:
        self.mdi_manager._navigator_drop(pos, mime_data)

    def _active_plot(self):
        return self.mdi_manager.active_plot()

    def _active_plot_window(self):
        return self.mdi_manager.active_plot_window()


    def _handle_drop_files(self, paths: list[str]) -> None:
        files = [p for p in paths if os.path.isfile(p) and p.lower().endswith(SUPPORTED_EXTS)]
        if files:
            self._create_signals(files)

    def register_navigator_drag_payload(self, signal, nav_manager) -> str:
        return self.mdi_manager.register_navigator_drag_payload(signal, nav_manager)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.dask_manager.shutdown()
        self._shutdown_update_thread()
        super().closeEvent(event)

    def _shutdown_update_thread(self) -> None:
        worker = getattr(self, "_plot_update_worker", None)
        thread = getattr(self, "_update_thread", None)

        if worker is not None:
            QtCore.QMetaObject.invokeMethod(
                worker, "stop", QtCore.Qt.ConnectionType.QueuedConnection
            )
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(2000)

    def close(self) -> None:
        self._shutdown_update_thread()
        try:
            self.dask_manager.shutdown()
        except Exception:
            pass
        super().close()


def _asset(filename: str) -> str:
    """Return the absolute path to a bundled asset regardless of how the app was launched."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def main() -> MainWindow:
    with log_startup_time("QApplication startup"):
        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("SpyDE")

    # Splash screen — use package-relative path so it works when bundled
    splash_path = _asset("SpydeDark.png")
    pixmap = QPixmap(splash_path).scaled(
        300,
        300,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )

    splash = QSplashScreen(pixmap, Qt.WindowType.FramelessWindowHint)
    splash.show()
    splash.raise_()
    app.processEvents()

    with log_startup_time("MainWindow construction"):
        main_window = MainWindow(app=app)

    main_window.setWindowTitle("SpyDE")

    # Platform-appropriate window / taskbar icon
    if sys.platform == "darwin":
        icon_path = _asset("icon.icns")
    elif sys.platform == "win32":
        icon_path = _asset("Spyde.ico")
    else:  # Linux / other
        icon_path = _asset("Spyde.png")

    main_window.setWindowIcon(QIcon(icon_path))
    main_window.show()
    splash.finish(main_window)

    app.exec()
    return main_window


if __name__ == "__main__":
    # multiprocessing.freeze_support()
    sys.exit(main())

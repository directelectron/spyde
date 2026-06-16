"""
session.py — Python-side session coordinator.

Owns: signal trees, Dask cluster, plot registration, file I/O, action dispatch.
All communication with Electron goes through ipc.emit().
"""
from __future__ import annotations

import json
import os
import threading
from typing import TYPE_CHECKING, Any

import hyperspy.api as hs
from hyperspy.signal import BaseSignal

from spyde.backend.ipc import emit, emit_status, emit_error, emit_progress
from spyde.dask_manager import DaskManager
from spyde.workers.plot_update_worker import PlotUpdateWorker

if TYPE_CHECKING:
    from spyde.signal_tree import BaseSignalTree
    from spyde.drawing.plots.plot import Plot

SUPPORTED_EXTS = (".hspy", ".zspy", ".mrc", ".tif", ".tiff", ".de5")

_DEFAULT_EXAMPLE_NAMES = (
    "mgo_nanocrystals",
    "small_ptychography",
    "zrnb_precipitate",
    "pdcusi_insitu",
    "sped_ag",
    "fe_multi_phase_grains",
)


class Session:
    """
    Top-level coordinator.  One instance per app lifetime.

    The Electron frontend talks to this object exclusively through IPC messages
    routed by app.py.  The session talks back via ipc.emit().
    """

    def __init__(self, n_workers: int, threads_per_worker: int) -> None:
        self.signal_trees: list[BaseSignalTree] = []
        self._plots: list[Plot] = []  # all open Plot objects (anyplotlib-backed)
        self._next_window_id = 0
        self._recent_files: list[str] = []
        self.current_selected_signal_tree = None

        # MDI manager
        from spyde.mdi_manager import MDIManager
        self.mdi_manager = MDIManager(session=self)

        # Dask
        self.dask_manager = DaskManager(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
        )
        self.dask_manager.ready.connect(self._on_dask_ready)
        self.dask_manager.error.connect(self._on_dask_error)

        # Plot update poller
        self._plot_worker = PlotUpdateWorker(
            get_plots_callable=lambda: list(self._plots),
            interval_ms=5,
        )
        self._plot_worker.plot_ready.connect(self._on_plot_ready)
        self._plot_worker.signal_ready.connect(self._on_signal_ready)
        self._plot_worker.debug_print.connect(lambda msg: print(msg))
        self._plot_worker.start()

        # Settings
        self._settings_path = os.path.join(
            os.path.expanduser("~"), ".spyde", "settings.json"
        )
        self._settings: dict[str, Any] = self._load_settings()

    # ── Startup ────────────────────────────────────────────────────────────────

    def start_dask(self) -> None:
        self.dask_manager.start()

    def _on_dask_ready(self) -> None:
        emit_status("Dask cluster ready")
        emit({"type": "dask_ready", "dashboard": self.dask_manager.client.dashboard_link})

    def _on_dask_error(self, msg: str) -> None:
        emit_error(f"Dask startup failed: {msg}")

    # ── File operations ────────────────────────────────────────────────────────

    def open_file(self, path: str) -> None:
        """Load a HyperSpy-compatible file and open it in the MDI."""
        if not os.path.isfile(path):
            emit_error(f"File not found: {path}")
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in SUPPORTED_EXTS:
            emit_error(f"Unsupported file type: {ext}")
            return

        emit_status(f"Loading {os.path.basename(path)}…")
        threading.Thread(
            target=self._load_file_thread,
            args=(path,),
            daemon=True,
            name=f"load-{os.path.basename(path)}",
        ).start()

    def _load_file_thread(self, path: str) -> None:
        try:
            signal = hs.load(path, lazy=True)
            if not isinstance(signal, list):
                signal = [signal]
            for sig in signal:
                self._add_signal(sig, source_path=path)
            self._add_recent(path)
            emit({"type": "recent_files", "paths": self._recent_files[:20]})
        except Exception as e:
            emit_error(f"Failed to load {os.path.basename(path)}: {e}")

    def load_example_data(self, name: str) -> None:
        import pyxem.data as _pxd

        emit_status(f"Loading example: {name}…")
        threading.Thread(
            target=self._load_example_thread,
            args=(name,),
            daemon=True,
            name=f"example-{name}",
        ).start()

    def _load_example_thread(self, name: str) -> None:
        try:
            import pyxem.data as _pxd
            loader = getattr(_pxd, name, None)
            if loader is None:
                emit_error(f"Unknown example dataset: {name}")
                return
            sig = loader()
            self._add_signal(sig, source_path=None)
        except Exception as e:
            emit_error(f"Failed to load example {name}: {e}")

    def _add_signal(self, signal: BaseSignal, source_path: str | None = None) -> None:
        """Create a signal tree + plots for a loaded signal."""
        from spyde.signal_tree import BaseSignalTree
        from spyde.drawing.plots.plot import Plot

        client = self.dask_manager.client
        tree = BaseSignalTree(
            root_signal=signal,
            session=self,
            distributed_client=client,
        )
        self.signal_trees.append(tree)

        # Open the MDI windows for this tree
        tree.open()

        title = signal.metadata.get_item("General.title", default=None)
        if title is None and source_path:
            title = os.path.splitext(os.path.basename(source_path))[0]

        emit_status(f"Loaded: {title or 'Signal'}")

    # ── Plot / window management ───────────────────────────────────────────────

    def add_plot_window(
        self,
        *,
        is_navigator: bool = False,
        signal_tree=None,
        plot_manager=None,
    ):
        """Delegate to MDIManager — the single place that creates PlotWindows."""
        return self.mdi_manager.add_plot_window(
            is_navigator=is_navigator,
            signal_tree=signal_tree,
            plot_manager=plot_manager,
        )

    def register_plot(self, plot: "Plot") -> None:
        self._plots.append(plot)

    def unregister_plot(self, plot: "Plot") -> None:
        self._plots = [p for p in self._plots if p is not plot]

    def next_window_id(self) -> int:
        wid = self._next_window_id
        self._next_window_id += 1
        return wid

    # ── Plot update callbacks ──────────────────────────────────────────────────

    def _on_plot_ready(self, plot, result, future) -> None:
        if isinstance(result, Exception):
            print(f"Plot update failed: {result}")
            return
        try:
            if plot.current_data is not future:
                return
            plot.current_data = result
            plot.update()
        except Exception as e:
            print(f"Failed to update plot: {e}")

    def _on_signal_ready(self, signal, result, plot) -> None:
        if isinstance(result, Exception):
            print(f"Signal update failed: {result}")
            return
        try:
            signal.data = result
            signal._lazy = False
            signal._assign_subclass()
            plot.parent_selector.delayed_update_data(update_contrast=True, force=True)
        except Exception as e:
            print(f"Failed to update signal: {e}")

    # ── Action dispatch ────────────────────────────────────────────────────────

    def dispatch_action(self, msg: dict) -> None:
        """Route an action message from Electron to the appropriate handler."""
        action = msg.get("action")
        payload = msg.get("payload", {})
        window_id = msg.get("window_id")

        plot = self._plot_by_window_id(window_id) if window_id is not None else None

        if action == "open_file":
            self.open_file(payload["path"])
        elif action == "load_example":
            self.load_example_data(payload["name"])
        elif action == "save_signal":
            self._save_signal(payload.get("path"), plot)
        elif action == "set_colormap":
            self._set_colormap(plot, payload.get("name"))
        elif action == "set_clim":
            self._set_clim(plot, payload.get("vmin"), payload.get("vmax"))
        elif action == "close_window":
            self._close_window(window_id)
        elif action == "resize_figure":
            self._resize_figure(window_id, payload.get("width"), payload.get("height"))
        elif action == "figure_event":
            self._dispatch_figure_event(window_id, payload.get("event_json"))
        else:
            print(f"Unknown action: {action}")

    def _plot_by_window_id(self, window_id: int):
        for p in self._plots:
            if getattr(p, "window_id", None) == window_id:
                return p
        return None

    def _save_signal(self, path: str | None, plot) -> None:
        if plot is None or path is None:
            emit_error("Save: no active plot or path")
            return
        signal = getattr(getattr(plot, "plot_state", None), "current_signal", None)
        if signal is None:
            emit_error("Save: no signal in active plot")
            return
        try:
            import dask
            with dask.config.set(scheduler="synchronous"):
                signal.save(path, overwrite=True)
            emit({"type": "saved", "path": path})
        except Exception as e:
            emit_error(f"Save failed: {e}")

    def _set_colormap(self, plot, name: str | None) -> None:
        if plot is None or name is None:
            return
        try:
            plot.set_colormap(name)
        except Exception as e:
            print(f"set_colormap failed: {e}")

    def _set_clim(self, plot, vmin, vmax) -> None:
        if plot is None:
            return
        try:
            plot.set_clim(vmin, vmax)
        except Exception as e:
            print(f"set_clim failed: {e}")

    def _close_window(self, window_id: int) -> None:
        plot = self._plot_by_window_id(window_id)
        if plot is None:
            return
        try:
            tree = getattr(plot, "signal_tree", None)
            if tree is not None:
                self._close_tree(tree)
            else:
                plot.close()
                self.unregister_plot(plot)
        except Exception as e:
            print(f"close_window failed: {e}")

    def _close_tree(self, tree: "BaseSignalTree") -> None:
        if tree not in self.signal_trees:
            return
        try:
            tree.close()
        except Exception:
            pass
        self.signal_trees.remove(tree)
        emit({"type": "windows_closed", "tree_id": id(tree)})

    def _resize_figure(self, window_id: int, width: int | None, height: int | None) -> None:
        plot = self._plot_by_window_id(window_id)
        if plot is None or width is None or height is None:
            return
        try:
            import anyplotlib._electron as _el
            fig_id = getattr(plot, "fig_id", None)
            if fig_id is not None:
                _el.resize_figure(fig_id, int(width), int(height))
        except Exception as e:
            print(f"resize_figure failed: {e}")

    def _dispatch_figure_event(self, window_id: int, event_json: str | None) -> None:
        if event_json is None:
            return
        plot = self._plot_by_window_id(window_id)
        if plot is None:
            return
        try:
            import anyplotlib._electron as _el
            fig_id = getattr(plot, "fig_id", None)
            if fig_id is not None:
                _el.dispatch_event(fig_id, event_json)
        except Exception as e:
            print(f"dispatch_figure_event failed: {e}")

    # ── Settings & recent files ────────────────────────────────────────────────

    def _load_settings(self) -> dict:
        try:
            with open(self._settings_path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_settings(self) -> None:
        os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
        with open(self._settings_path, "w", encoding="utf-8") as fh:
            json.dump(self._settings, fh, indent=2)

    def _add_recent(self, path: str) -> None:
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._settings["recent_files"] = self._recent_files[:20]
        try:
            self._save_settings()
        except Exception:
            pass

    def get_recent_files(self) -> list[str]:
        return list(self._recent_files[:20])

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._plot_worker.stop()
        self.dask_manager.shutdown()

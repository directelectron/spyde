"""
session.py — Python-side session coordinator.

Owns: signal trees, Dask cluster, plot registration, file I/O, action dispatch.
All communication with Electron goes through ipc.emit().
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from hyperspy.signal import BaseSignal

from spyde.backend.ipc import emit, emit_status, emit_error, emit_progress
from spyde.backend._session_axes import AxesEditorMixin
from spyde.backend._session_actions import (
    ActionRouterMixin, _TEST_ACTIONS, _TEST_ACTIONS_ENABLED,
)
from spyde.backend._session_files import (
    FileLoaderMixin,
    SUPPORTED_EXTS, _DIR_DATASET_EXTS, _DEFAULT_EXAMPLE_NAMES, _EXAMPLE_CALIBRATION,
    _path_ext, _is_supported_dataset_path, _dataset_size_bytes,
    _apply_example_calibration,
)
from spyde.backend._session_testharness import TestHarnessMixin
from spyde.backend.tutorial_data import TutorialDataMixin
from spyde.backend._session_windows import WindowManagerMixin
from spyde.dask_manager import DaskManager
from spyde.workers.plot_update_worker import PlotUpdateWorker

log = logging.getLogger(__name__)

# Per-frame navigator/redraw trace logs ([REDRAW2] APPLY/DROP) are gated behind
# this — they fire on every painted frame and flood the IPC log at DEBUG. Match
# the same env switch used in base_selector / update_functions / plot_update_worker.
_NAV_TIMING = os.environ.get("SPYDE_NAV_TIMING") == "1"

# _TEST_ACTIONS / _TEST_ACTIONS_ENABLED live in _session_actions; the staged
# action table lives in spyde.actions.registry (STAGED_HANDLERS).

if TYPE_CHECKING:
    from spyde.signal_tree import BaseSignalTree
    from spyde.drawing.plots.plot import Plot

# SUPPORTED_EXTS / _path_ext / _is_supported_dataset_path / _dataset_size_bytes /
# _DIR_DATASET_EXTS / _DEFAULT_EXAMPLE_NAMES / _EXAMPLE_CALIBRATION /
# _apply_example_calibration now live in _session_files (re-imported above so
# `from spyde.backend.session import _path_ext` etc. still resolve).


class Session(
    AxesEditorMixin,
    ActionRouterMixin,
    FileLoaderMixin,
    TestHarnessMixin,
    TutorialDataMixin,
    WindowManagerMixin,
):
    """
    Top-level coordinator.  One instance per app lifetime.

    The Electron frontend talks to this object exclusively through IPC messages
    routed by app.py.  The session talks back via ipc.emit().
    """

    def __init__(self, n_workers: int, threads_per_worker: int) -> None:
        self.signal_trees: list[BaseSignalTree] = []
        self._plots: list[Plot] = []  # all open Plot objects (anyplotlib-backed)
        self._next_window_id = 0
        self._active_window_id: int | None = None  # focused window (for save etc.)
        self._recent_files: list[str] = []
        self._example_temp_paths: list[str] = []  # temp .zspy dirs to clean up
        # (src_window_id, action_name) -> {"selector", "out_wids"} so deselecting
        # a toolbar action can hide the output window + ROI it created.
        self._action_artifacts: dict[tuple[int, str], dict] = {}
        # window_id -> controller for windows that are NOT registered Plots
        # (bare `figure` emits: strain map, IPF views…). See the WindowController
        # protocol in spyde/actions/registry.py. _forget_window closes + evicts.
        self._window_controllers: dict[int, object] = {}
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
        # Gate: set once the cluster is ready (or when dask is skipped). A file /
        # example load fired before the cluster exists waits on this instead of
        # racing ahead with a None client (which errored "Folder not found" /
        # silently produced no navigator). See _await_dask.
        self._dask_ready = threading.Event()
        # Tests and headless scripts construct Session directly, bypassing
        # app.py's SPYDE_NO_DASK branch — honour the env var here too, or a
        # load thread blocks _await_dask's full timeout on a cluster that will
        # never start (the test_nav_shape_prompt "busy never cleared" CI hang).
        if os.environ.get("SPYDE_NO_DASK") == "1":
            self._dask_ready.set()

        # Plot update poller. `dispatch` marshals the result-APPLY onto the main
        # asyncio thread (set later via set_main_loop, once the loop exists) — the
        # poll thread only detects done futures + reads shm; plot.update()/push
        # runs on the main thread, like the Qt app's queued plot_ready slot.
        self._main_loop = None
        # Lazily-built ComputeBackend for the EXPENSIVE-tier navigator read (large
        # region / cold cross-chunk / derived rebin-crop view): submit_graph gives
        # a cancellable async read so an expensive frame never blocks the serial
        # nav dispatcher. Threaded (no-cluster) mode gets its own small pool so
        # nav reads don't queue behind other pool work; distributed mode uses the
        # live client. Rebuilt when the client identity changes. See compute_backend.
        self._compute_backend = None
        self._compute_backend_client = None   # identity of the client the cached backend wraps
        self._nav_executor = None             # ThreadPoolExecutor, created lazily in no-cluster mode
        # Set by shutdown() so a nav update still draining on the process-global
        # dispatcher thread can't recreate the nav executor after teardown.
        self._closed = False
        self._plot_worker = PlotUpdateWorker(
            get_plots_callable=lambda: list(self._plots),
            interval_ms=5,
            dispatch=self._dispatch_to_main,
        )
        self._plot_worker.plot_ready.connect(self._on_plot_ready)
        self._plot_worker.signal_ready.connect(self._on_signal_ready)
        self._plot_worker.debug_print.connect(lambda msg: log.debug(msg))
        self._plot_worker.start()

        # Settings
        self._settings_path = os.path.join(
            os.path.expanduser("~"), ".spyde", "settings.json"
        )
        self._settings: dict[str, Any] = self._load_settings()
        # Restore the persisted recent-files list (capped to match _add_recent).
        try:
            self._recent_files = list(self._settings.get("recent_files", []))[:20]
        except Exception as e:
            log.debug("restoring recent files from settings failed: %s", e)
        # update_channel/skip_version mirror the Electron-side updater's choice
        # here so they're inspectable/debuggable from the Python side too (the
        # Electron main process is the one that actually acts on them).
        self._update_channel: str = (
            self._settings.get("update_channel") if self._settings.get("update_channel") in ("stable", "beta")
            else "stable"
        )

    # ── Startup ────────────────────────────────────────────────────────────────

    def start_dask(self) -> None:
        self.dask_manager.start()

    def skip_dask(self) -> None:
        """Eager / no-dask mode (SPYDE_NO_DASK): the cluster never starts, so open
        the gate immediately — a load must NOT wait forever for a `ready` that will
        never fire."""
        self._dask_ready.set()

    def _await_dask(self, timeout: float = 120.0) -> bool:
        """Block the calling (load) thread until the Dask cluster is ready, so a
        file/example opened during startup waits for the cluster instead of racing
        ahead with a None client. Returns True if ready, False on timeout. Safe in
        no-dask mode (the gate is pre-set by skip_dask). Never call on the main
        asyncio thread — only from the load worker threads."""
        if self._dask_ready.is_set():
            return True
        emit_status("Waiting for the compute cluster to start…")
        return self._dask_ready.wait(timeout)

    def set_main_loop(self, loop) -> None:
        """Register the main asyncio loop so the plot poller can marshal the
        result-apply onto this (main) thread. Call from app._main once the loop
        is running.

        NB the process's frozen-timer pathology (waits only wake on process
        I/O — see runner.ts's backend tick) is healed by Electron's 0.5 Hz
        stdin tick; an in-process wake ticker was tried and is useless here
        because its own sleep freezes the same way."""
        self._main_loop = loop

    def _dispatch_to_main(self, fn) -> None:
        """Schedule fn() on the main asyncio thread (the plot poller calls this to
        apply a finished future's result). Falls back to running inline if no loop
        is registered yet (early startup / tests)."""
        loop = self._main_loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(fn)
                return
            except Exception as e:
                log.debug("dispatch_to_main failed, running inline: %s", e)
        fn()

    @property
    def compute_backend(self):
        """A ComputeBackend for the EXPENSIVE-tier navigator read (submit_graph).

        Distributed when a Dask client exists (already async + cancellable via the
        adapter); otherwise a small dedicated ThreadPoolExecutor so an expensive
        nav frame computes off the serial dispatcher thread and a superseded one
        can be cancelled while queued. Cached and rebuilt only when the client
        identity changes (client appears once the cluster is ready, or disappears
        on shutdown) — so the same backend is reused across scrubbing.

        Returns None once the session is shut down: the process-global nav
        dispatcher (and its settle timers) can still fire a queued update after
        teardown, and we must NOT lazily spawn a fresh executor then (it would leak
        and defeat shutdown's cleanup). A None here makes _submit_async_nav_read
        fall through to the synchronous read, which is always correct."""
        if self._closed:
            return None
        from spyde.compute_backend import ComputeBackend
        client = self.dask_manager.client if self.dask_manager is not None else None
        if client is not self._compute_backend_client or self._compute_backend is None:
            if client is not None:
                self._compute_backend = ComputeBackend(client=client)
            else:
                if self._nav_executor is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self._nav_executor = ThreadPoolExecutor(
                        max_workers=2, thread_name_prefix="nav-read"
                    )
                self._compute_backend = ComputeBackend(executor=self._nav_executor)
            self._compute_backend_client = client
        return self._compute_backend

    def _on_dask_ready(self) -> None:
        self._dask_ready.set()           # release any load waiting on the cluster
        emit_status("Dask cluster ready")
        emit({"type": "dask_ready", "dashboard": self.dask_manager.client.dashboard_link})
        # Live compute telemetry for the StatusBar HUD (worker CPU/mem/queues +
        # GPU util) — see backend/dask_stats.py. Stopped in shutdown().
        try:
            from spyde.backend.dask_stats import DaskStatsSampler
            self._dask_stats = DaskStatsSampler(
                lambda: getattr(self.dask_manager, "client", None))
            self._dask_stats.start()
        except Exception as e:
            log.debug("dask stats sampler failed to start: %s", e)

    def _on_dask_error(self, msg: str) -> None:
        emit_error(f"Dask startup failed: {msg}")

    def _add_signal(
        self,
        signal: BaseSignal,
        source_path: str | None = None,
        navigator_override: BaseSignal | None = None,
        selector_type=None,
        enable_nav_sidecar: bool = True,
    ):
        """Create a signal tree + plots for a loaded signal. Returns the tree.

        NB: callers on fresh threads must not race the startup prewarm's
        hyperspy/pyxem import (partially-initialized-module poisoning) —
        ensure_heavy_imports() below single-flights it.

        ``navigator_override`` supplies a pre-built navigator (e.g. a vectors
        count-map) so the base navigator is NOT recomputed from the full
        dataset — essential for the breaking transformations (Find Vectors).
        """
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()
        from spyde.signal_tree import BaseSignalTree
        from spyde.drawing.plots.plot import Plot

        client = self.dask_manager.client
        # Only a real on-disk origin enables the navigator sidecar cache
        # (test/example loaders pass pseudo-paths like "test_data"; a STACK's
        # navigator depends on every member, not just paths[0] → disabled).
        disk_path = (source_path if enable_nav_sidecar and source_path
                     and os.path.exists(source_path) else None)

        # Resolve the dataset name and stamp it onto the signal BEFORE building the
        # tree — the tree's constructor (_initialize_initial_plots) creates the
        # plots and emits their `figure` messages, whose `title` field (the window
        # header + breadcrumb Name) and in-panel title strip both read
        # General.title. Stamping after would leave the header at the "Signal"/
        # "Navigator" fallback even though we know the filename.
        title = signal.metadata.get_item("General.title", default=None)
        # hyperspy may return an empty string or a `<undefined>` sentinel for an
        # unset title, not None — treat any of those as "no title".
        if (title is None or str(title).strip() in ("", "<undefined>")) and source_path:
            title = os.path.splitext(os.path.basename(source_path))[0]
            if title:
                try:
                    signal.metadata.set_item("General.title", title)
                except Exception as e:
                    log.debug("stamping General.title failed: %s", e)

        tree = BaseSignalTree(
            root_signal=signal,
            session=self,
            distributed_client=client,
            selector_type=selector_type,
            navigator_override=navigator_override,
            source_path=disk_path,
        )
        self.signal_trees.append(tree)

        # Open the MDI windows for this tree
        tree.open()

        # Emit metadata + axes for the sidebar, tagged with this tree's windows.
        try:
            from spyde.metadata_extract import build_metadata_dict
            emit({
                "type": "metadata",
                "window_ids": self._tree_window_ids(tree),
                "metadata": build_metadata_dict(tree),
            })
        except Exception as e:
            log.warning("metadata emit failed: %s", e)
        self._emit_axes(tree)
        self._emit_signal_type(tree)
        # The Workflow panel is always-on: push the (initial, single-node) tree
        # for every window of this tree right away — it grows with transforms.
        self._reemit_signal_tree(tree)
        # …and the navigator chip strip (shown once a tree has ≥2 navigators).
        try:
            from spyde.actions.navigator_views import emit_navigator_options
            emit_navigator_options(tree)
        except Exception as e:
            log.debug("navigator options emit failed: %s", e)
        try:
            from spyde.actions.composition import emit_composition
            emit_composition(tree, self._tree_window_ids(tree))
        except Exception as e:
            log.warning("composition emit failed: %s", e)

        emit_status(f"Loaded: {title or 'Signal'}")
        self._notify_console_trees_changed()
        return tree

    def _notify_console_trees_changed(self) -> None:
        """Refresh the math console's signal bindings after a tree is added /
        closed. Only pokes the console if it has ALREADY been created — never
        force-creates the engine (and its heavy hyperspy import) just because a
        dataset loaded. The refresh is posted onto the console thread, so this is a
        cheap non-blocking call safe to make from the main thread OR a load thread."""
        con = getattr(self, "_console", None)
        if con is not None:
            try:
                con.refresh_bindings()
            except Exception as e:
                log.debug("console binding refresh failed: %s", e)

    # Signal types offered in the sidebar dropdown (HyperSpy/pyxem). "" = the
    # generic BaseSignal/Signal2D with no specialised type.
    _SIGNAL_TYPES = (
        "",
        "electron_diffraction",
        "diffraction",
        "electron_microscope",
        "EELS",
        "EDS_TEM",
        "EDS_SEM",
        "hologram",
    )

    def _emit_signal_type(self, tree) -> None:
        """Tell the sidebar the active signal's current HyperSpy ``signal_type``
        and the list of types it can be switched to."""
        try:
            stype = tree.root.metadata.get_item("Signal.signal_type", default="") or ""
            emit({
                "type": "signal_type_info",
                "window_ids": self._tree_window_ids(tree),
                "current": stype,
                "options": list(self._SIGNAL_TYPES),
            })
        except Exception as e:
            log.warning("signal_type emit failed: %s", e)

    def _set_signal_type(self, plot, signal_type: str) -> None:
        """Apply a new HyperSpy ``signal_type`` to the active plot's current
        signal (re-casts the signal class), then re-emit metadata/axes/type so
        the sidebar + downstream actions reflect the change."""
        if plot is None or getattr(plot, "signal_tree", None) is None:
            return
        tree = plot.signal_tree
        try:
            sig = plot.plot_state.current_signal if plot.plot_state else tree.root
            sig.set_signal_type(signal_type or "")
        except Exception as e:
            emit_error(f"Could not set signal type to {signal_type!r}: {e}")
            return
        # Re-broadcast the dependent sidebar panels.
        try:
            from spyde.metadata_extract import build_metadata_dict
            emit({
                "type": "metadata",
                "window_ids": self._tree_window_ids(tree),
                "metadata": build_metadata_dict(tree),
            })
        except Exception as e:
            log.debug("metadata re-emit after signal-type change failed: %s", e)
        self._emit_signal_type(tree)
        # Re-send the toolbar config: available actions are gated on the signal
        # class / signal_type (toolbars.yaml signal_class / signal_types), so a
        # type change must refresh the toolbar (e.g. diffraction actions appear
        # when the signal becomes electron_diffraction).
        for sp in list(getattr(tree, "signal_plots", []) or []):
            try:
                st = getattr(sp, "plot_state", None)
                if st is not None and hasattr(st, "_send_toolbar_config"):
                    st._send_toolbar_config()
            except Exception as e:
                log.debug("re-sending toolbar after signal-type change failed: %s", e)

    def _tree_window_ids(self, tree) -> list[int]:
        return sorted({
            p.window_id for p in self._plots
            if getattr(p, "signal_tree", None) is tree and p.window_id is not None
        })

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

    def register_nav_selector(self, window_id: int, selector) -> None:
        """Track a navigator's selectors so the dock can toggle each between
        crosshair and integration modes. The FIRST selector of a window stays
        the window-keyed fallback (back-compat callers address by window id);
        every selector is also addressable by its ``selector_id`` (the dock's
        per-row key — one navigator can carry several selectors)."""
        if not hasattr(self, "_nav_selectors"):
            self._nav_selectors = {}
        if not hasattr(self, "_nav_selectors_by_id"):
            self._nav_selectors_by_id = {}
        self._nav_selectors.setdefault(window_id, selector)
        self._nav_selectors_by_id[id(selector)] = (window_id, selector)

    def set_selector_mode(self, window_id: int, integrate: bool,
                          selector_id: int | None = None) -> None:
        """Switch a navigator selector between crosshair and integrating mode.
        ``selector_id`` addresses one selector of a multi-selector navigator;
        without it the window's first selector is used."""
        sel = None
        if selector_id is not None:
            window_id, sel = getattr(self, "_nav_selectors_by_id", {}).get(
                selector_id, (window_id, None))
        if sel is None:
            sel = getattr(self, "_nav_selectors", {}).get(window_id)
        if sel is None or not hasattr(sel, "set_integrating"):
            return
        try:
            sel.set_integrating(bool(integrate))
            # No title here — the dock merges by selector_id and keeps the
            # title/colour from the creation-time selector_info.
            emit({
                "type": "selector_info",
                "window_id": window_id,
                "selector_id": id(sel),
                "color": getattr(sel, "color", None),
                "mode": "integrate" if integrate else "crosshair",
            })
        except Exception as e:
            log.warning("set_selector_mode failed: %s", e)

    def _select_signal_node(self, plot, signal_id) -> None:
        """Switch to the signal-tree node with the given id (the id(node.signal)
        emitted in the signal_tree message). The pick can come from ANY of the
        tree's windows (the Workflow panel shows the tree for navigators too),
        so search all of the tree's signal plots for the one holding the node."""
        if plot is None or signal_id is None:
            return
        tree = getattr(plot, "signal_tree", None)
        cands = [plot] + list(getattr(tree, "signal_plots", []) or [])
        for p in cands:
            for sig in list(getattr(p, "plot_states", {}) or {}):
                if id(sig) == signal_id:
                    from spyde.actions.lifecycle import show_tree_node
                    show_tree_node(p, tree, sig)
                    emit({"type": "status", "text": "Switched signal node"})
                    return

    def _reemit_signal_tree(self, plot_or_tree) -> None:
        """Push the workflow tree to EVERY window of the tree (signal plots and
        navigators alike) so the dock's Workflow section is populated whichever
        of the tree's windows has focus. Called on tree creation and after every
        transform / node switch. No-op if the tree isn't available yet."""
        tree = plot_or_tree
        if tree is not None and not hasattr(tree, "root_node"):
            tree = getattr(plot_or_tree, "signal_tree", None)
        root_node = getattr(tree, "root_node", None) if tree is not None else None
        if root_node is None:
            return

        def node_to_dict(node):
            return {
                "name": node.name, "signal_id": id(node.signal),
                "children": [node_to_dict(c) for c in node.children.values()],
            }

        # Active node = what the tree's signal plot displays (prefer the plot
        # we were called with when it has a state).
        active = None
        cands = ([plot_or_tree] if hasattr(plot_or_tree, "plot_state") else []) \
            + list(getattr(tree, "signal_plots", []) or [])
        for p in cands:
            st = getattr(p, "plot_state", None)
            sig = getattr(st, "current_signal", None) if st is not None else None
            if sig is not None:
                active = id(sig)
                break
        payload = node_to_dict(root_node)
        window_ids = self._tree_window_ids(tree) or \
            ([getattr(plot_or_tree, "window_id", None)]
             if getattr(plot_or_tree, "window_id", None) is not None else [])
        for wid in window_ids:
            emit({
                "type": "signal_tree", "window_id": wid,
                "tree": payload, "active_signal_id": active, "visible": True,
            })

    # ── Plot update callbacks ──────────────────────────────────────────────────

    def _on_plot_ready(self, plot, result, future) -> None:
        # Runs on the MAIN thread (marshaled from the poll worker via
        # _dispatch_to_main). A superseded future (newer navigator position already
        # in flight) is no longer the one the plot wants — drop its result silently.
        # This also covers a torn shared-memory read, whose result is a ValueError;
        # it's expected under the latest-wins model, not an error.
        if plot.current_data is not future:
            if _NAV_TIMING:
                log.debug("[REDRAW2] DROP win=%s (current_data superseded "
                          "future %s)", getattr(plot, "window_id", None),
                          getattr(future, "key", None))
            return
        if isinstance(result, Exception):
            log.debug("[REDRAW2] DROP win=%s (exception/torn read): %s",
                      getattr(plot, "window_id", None), result)
            return
        try:
            plot.current_data = result
            plot.update()
            if _NAV_TIMING:
                log.debug("[REDRAW2] APPLY win=%s future=%s",
                          getattr(plot, "window_id", None), getattr(future, "key", None))
        except Exception as e:
            log.warning("Failed to update plot: %s", e)

    def _on_signal_ready(self, signal, result, plot) -> None:
        if isinstance(result, Exception):
            log.warning("Signal update failed: %s", result)
            return
        try:
            signal.data = result
            signal._lazy = False
            signal._assign_subclass()
            sel = getattr(plot, "parent_selector", None)
            if sel is not None:
                sel.delayed_update_data(update_contrast=True, force=True)
            else:
                # No selector (e.g. a navigatorless plot, or selector init failed)
                # — just repaint with the freshly computed data.
                plot.needs_auto_level = True
                plot.update()
        except Exception as e:
            log.warning("Failed to update signal: %s", e)

    def _set_colormap(self, plot, name: str | None) -> None:
        if plot is None or name is None:
            return
        try:
            plot.set_colormap(name)
        except Exception as e:
            log.warning("set_colormap failed: %s", e)

    def _set_clim(self, plot, vmin, vmax) -> None:
        if plot is None:
            return
        try:
            plot.set_clim(vmin, vmax)
        except Exception as e:
            log.warning("set_clim failed: %s", e)

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
        except Exception as e:
            log.debug("saving recent-files settings failed: %s", e)

    def set_update_channel(self, channel: str) -> None:
        """Persist the update channel ('stable' or 'beta') to settings.json.

        This mirrors the choice the Electron main process's autoUpdater
        actually acts on (electron/src/main/updater.ts) — kept here too so the
        preference is visible/debuggable from the Python side and survives a
        settings.json inspection independent of Electron's own storage.
        """
        if channel not in ("stable", "beta"):
            log.warning("ignoring invalid update_channel %r", channel)
            return
        self._update_channel = channel
        self._settings["update_channel"] = channel
        try:
            self._save_settings()
        except Exception as e:
            log.debug("saving update_channel setting failed: %s", e)

    def get_recent_files(self) -> list[str]:
        return list(self._recent_files[:20])

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        pb = getattr(self, "_playback", None)
        if pb is not None:
            try:
                pb.shutdown()
            except Exception as e:
                log.debug("playback shutdown failed: %s", e)
        con = getattr(self, "_console", None)
        if con is not None:
            try:
                con.shutdown()
            except Exception as e:
                log.debug("console shutdown failed: %s", e)
        self._closed = True   # block compute_backend from recreating _nav_executor
        stats = getattr(self, "_dask_stats", None)
        if stats is not None:
            try:
                stats.stop()
            except Exception as e:
                log.debug("dask stats sampler stop failed: %s", e)
        self._plot_worker.stop()
        if self._nav_executor is not None:
            try:
                self._nav_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                log.debug("nav executor shutdown failed: %s", e)
            self._nav_executor = None
        self.dask_manager.shutdown()
        for tmpdir in self._example_temp_paths:
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                log.debug("removing example temp dir %s failed: %s", tmpdir, e)
        self._example_temp_paths.clear()


# ── staged handler (dispatch_action's _STAGED_HANDLERS: fn(session, plot, payload)) ──

def dispatch_set_update_channel(session: Session, plot, payload: dict) -> None:
    """Renderer's channel radio (stable/beta) -> persist to settings.json."""
    session.set_update_channel(str(payload.get("channel", "stable")))

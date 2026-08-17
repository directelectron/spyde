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

from de_shell.ipc import emit, emit_status, emit_error, emit_progress
from de_shell.session import SessionBase
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
    SessionBase,
):
    """
    Top-level coordinator.  One instance per app lifetime.

    The Electron frontend talks to this object exclusively through IPC messages
    routed by app.py.  The session talks back via ipc.emit().
    """

    def __init__(self, n_workers: int, threads_per_worker: int) -> None:
        # The shell's half: window/plot registry, main-loop marshalling, and the
        # settings store. SPYDE_SETTINGS_DIR lets tests (e2e in particular —
        # Electron refuses to launch if HOME is redirected to a scratch dir, so
        # overriding the whole-process home isn't viable there) point
        # settings.json at a throwaway directory without touching the real
        # user's ~/.spyde. Unset in normal use. The shell takes the resolved
        # directory rather than deriving it: the fallback and the override
        # variable are per-app, and guessing would let one app write into
        # another's preferences.
        SessionBase.__init__(
            self,
            settings_dir=os.environ.get("SPYDE_SETTINGS_DIR") or os.path.join(
                os.path.expanduser("~"), ".spyde"
            ),
        )

        self.signal_trees: list[BaseSignalTree] = []
        self._example_temp_paths: list[str] = []  # temp .zspy dirs to clean up
        # (src_window_id, action_name) -> {"selector", "out_wids"} so deselecting
        # a toolbar action can hide the output window + ROI it created.
        self._action_artifacts: dict[tuple[int, str], dict] = {}
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

        # Lazily-built ComputeBackend for the EXPENSIVE-tier navigator read (large
        # region / cold cross-chunk / derived rebin-crop view): submit_graph gives
        # a cancellable async read so an expensive frame never blocks the serial
        # nav dispatcher. Threaded (no-cluster) mode gets its own small pool so
        # nav reads don't queue behind other pool work; distributed mode uses the
        # live client. Rebuilt when the client identity changes. See compute_backend.
        self._compute_backend = None
        self._compute_backend_client = None   # identity of the client the cached backend wraps
        self._nav_executor = None             # ThreadPoolExecutor, created lazily in no-cluster mode

        # Plot update poller. `dispatch` marshals the result-APPLY onto the main
        # asyncio thread (SessionBase.set_main_loop registers it once the loop
        # exists) — the poll thread only detects done futures + reads shm;
        # plot.update()/push runs on the main thread.
        self._plot_worker = PlotUpdateWorker(
            get_plots_callable=lambda: list(self._plots),
            interval_ms=5,
            dispatch=self._dispatch_to_main,
        )
        self._plot_worker.plot_ready.connect(self._on_plot_ready)
        self._plot_worker.signal_ready.connect(self._on_signal_ready)
        self._plot_worker.debug_print.connect(lambda msg: log.debug(msg))
        self._plot_worker.start()

    # Settings, recent files, the update channel and the first-run flag are all
    # loaded by SessionBase.__init__ above — see de_shell/session.py.

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

        # Guided-walkthrough bookkeeping. Between `tutorial_session_begin` and
        # `tutorial_close_all` (sent by the in-app Tour on open/exit), record
        # every tree the walkthrough causes to appear — the tutorial dataset
        # itself AND everything derived from it during the tour (a find-vectors
        # result, a virtual image, an orientation map: they all land here, since
        # `commit_result_tree`/`vector_virtual_imaging` create their trees
        # through this same method). `tutorial_close_all` then closes the whole
        # set, so a finished tour leaves a clean workspace rather than a pile of
        # dummy-data windows.
        #
        # A tree backed by a real FILE ON DISK is the user's own data and is
        # never recorded — opening your own dataset mid-tour is safe.
        if getattr(self, "_tutorial_session_active", False):
            on_disk = bool(source_path) and os.path.exists(source_path)
            if not on_disk:
                if getattr(self, "_tutorial_session_trees", None) is None:
                    self._tutorial_session_trees = []
                self._tutorial_session_trees.append(tree)

        # Open the MDI windows for this tree
        tree.open()

        # Emit metadata + axes for the sidebar, tagged with this tree's windows.
        self._emit_metadata(tree)
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
        self._emit_metadata(tree)
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

    def set_selector_sum(self, window_id: int, frames: int,
                         selector_id: int | None = None) -> None:
        """Set how many navigation positions a POINT selector sums.

        1 is a plain crosshair. Higher keeps the single pointer and widens what
        it reads — n frames of an in-situ movie summed for signal, or the
        exposure of a sparse event stream chosen without turning the slider
        into a draggable range (which is what Integrate is for).
        """
        sel = None
        if selector_id is not None:
            window_id, sel = getattr(self, "_nav_selectors_by_id", {}).get(
                selector_id, (window_id, None))
        if sel is None:
            sel = getattr(self, "_nav_selectors", {}).get(window_id)
        if sel is None or not hasattr(sel, "sum_frames"):
            return
        try:
            from spyde.actions import csb_raw_frame
            # 0 means "go BELOW a plane" — show one raw camera frame. It is a
            # different integration, not a slice of the loaded plane stack, so
            # it swaps the selector's producer rather than widening the window.
            # The width itself stays 1: raw is a single frame by definition.
            raw = int(frames) == csb_raw_frame.RAW
            csb_raw_frame.install(sel, raw)
            sel.sum_frames = 1 if raw else max(1, int(frames))
            # Re-read at the new width immediately; the pointer has not moved,
            # so nothing else would trigger it. update_data() takes no `force`
            # — passing one raised, and since the raise happened BEFORE the
            # emit below, the width applied but the dock never heard about it
            # and its badge kept reading the old value.
            if hasattr(sel, "update_data"):
                sel.update_data()
            emit({
                "type": "selector_info",
                "window_id": window_id,
                "selector_id": id(sel),
                "color": getattr(sel, "color", None),
                "sum_frames": csb_raw_frame.RAW if raw else int(sel.sum_frames),
            })
        except Exception as e:
            log.warning("set_selector_sum failed: %s", e)

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
                    # The origin crosshair captured THIS plot's current signal
                    # axes; after the switch those belong to the node the user
                    # just left, so dragging it would recalibrate the wrong
                    # signal. Tear it down (and tell the dock, so the "+" goes
                    # with it — button ON ⟺ crosshair alive).
                    self._clear_offset_crosshair(p)
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
        #
        # window_computing stop: this is the resolution point for the ONE
        # future add_signal_plot's window_computing.start() bracketed (the
        # initial-frame client.compute for a lazy signal). Every branch below
        # (superseded/dropped/applied/exception) is a terminal outcome for
        # THIS future, so unconditionally emit the stop here — a no-op on the
        # renderer if this window was never actually tracked as computing (a
        # plain nav-read plot_ready, unrelated to add_signal_plot).
        from spyde.actions.lifecycle import window_computing
        window_computing(getattr(plot, "window_id", None)).stop()
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

    def _auto_clim(self, plot, mode: str = "robust") -> None:
        """Re-derive the display range from the data on screen (Auto / Reset).

        Duck-typed like set_clim/set_colormap: a Plot and a bare-figure
        controller (strain map, …) each implement ``auto_clim`` in their own
        terms — robust percentiles of the frame, or a symmetric strain scale.
        ``mode`` is "robust" (Auto) or "full" (Reset, the whole data range).
        """
        if plot is None:
            return
        fn = getattr(plot, "auto_clim", None)
        if not callable(fn):
            log.debug("auto_clim: %s does not support it", type(plot).__name__)
            return
        try:
            fn(mode)
        except Exception as e:
            log.warning("auto_clim failed: %s", e)

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
        # Stop the trees' in-flight compute BEFORE the executors go away.
        #
        # BaseSignalTree.close() exists for exactly this — it flips every
        # registered stopped_flag, cancels every registered future, and sets
        # _nav_stop so the progressive-navigator thread bails on its next poll —
        # but shutdown() never called it. That did not show while shutdown()
        # slept half a second on its way out: the background fill usually
        # finished inside the sleep. With that gone, a live `_bg_nav` submits
        # into an executor that is already shutting down and logs "threaded
        # navigator compute failed" during interpreter teardown.
        #
        # Cancel-then-teardown is the same order close() itself uses, and it is
        # the order that matters: reversing it is what produces the raise.
        for tree in list(getattr(self, "signal_trees", []) or []):
            try:
                tree.close()
            except Exception as e:
                log.debug("closing signal tree during shutdown failed: %s", e)
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
        # The compute backend keeps its OWN local nav pool in distributed mode
        # (submit_graph never uses the cluster for an interactive read).
        backend = self._compute_backend
        if backend is not None:
            try:
                backend.shutdown_nav_pool()
            except Exception as e:
                log.debug("compute-backend nav pool shutdown failed: %s", e)
        self.dask_manager.shutdown()
        for tmpdir in self._example_temp_paths:
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                log.debug("removing example temp dir %s failed: %s", tmpdir, e)
        self._example_temp_paths.clear()
        # The shell's half last: it only clears the window-controller registry
        # and latches _closed, and SpyDE's teardown above still needs both.
        super().shutdown()


# ── staged handler (dispatch_action's _STAGED_HANDLERS: fn(session, plot, payload)) ──

def dispatch_set_update_channel(session: Session, plot, payload: dict) -> None:
    """Renderer's channel radio (stable/beta) -> persist to settings.json."""
    session.set_update_channel(str(payload.get("channel", "stable")))


def get_first_run(session: Session, plot, payload: dict) -> None:
    """Staged handler: report whether the welcome tour has been seen yet.

    Payload-less request/response, mirroring get_gpu_status — the renderer
    calls this once on boot and emits `mark_tutorial_seen` when it opens the
    welcome tour so it never auto-shows again. Emits `first_run_result`.
    """
    emit({"type": "first_run_result", "first_run": session.first_run})


def dispatch_mark_tutorial_seen(session: Session, plot, payload: dict) -> None:
    """Renderer opened (or dismissed) the welcome tour -> persist tutorial_seen
    so it never auto-opens again. Always available (not gated)."""
    session.mark_tutorial_seen()

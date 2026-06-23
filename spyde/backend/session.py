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
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import hyperspy.api as hs
from hyperspy.signal import BaseSignal

from spyde.backend.ipc import emit, emit_status, emit_error, emit_progress
from spyde.dask_manager import DaskManager
from spyde.workers.plot_update_worker import PlotUpdateWorker

log = logging.getLogger(__name__)

# Per-frame navigator/redraw trace logs ([REDRAW2] APPLY/DROP) are gated behind
# this — they fire on every painted frame and flood the IPC log at DEBUG. Match
# the same env switch used in base_selector / update_functions / plot_update_worker.
_NAV_TIMING = os.environ.get("SPYDE_NAV_TIMING") == "1"

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

# Staged-wizard actions → "module.function". All share the (session, plot,
# payload) signature, so `dispatch_action` routes them through one lazy-import
# branch instead of a copy-pasted elif per handler.
# pyxem ships some examples with a half-scale reciprocal calibration; the legacy
# app corrected these per-dataset (so the kx/ky axes + scale bar read in Å⁻¹
# correctly). Restore that override on example load.
_EXAMPLE_CALIBRATION = {
    # sped_ag: pyxem default scale/offset are exactly half the true values.
    "sped_ag": dict(scale=0.00668207597 * 4, offset=-0.374196254 * 4),
}


def _apply_example_calibration(sig, name: str) -> None:
    cal = _EXAMPLE_CALIBRATION.get(name)
    if cal is None:
        return
    try:
        for ax in sig.axes_manager.signal_axes:
            ax.scale = cal["scale"]
            ax.offset = cal["offset"]
    except Exception as e:
        log.debug("applying calibration to signal axes failed: %s", e)


_STAGED_HANDLERS = {
    "om_generate_library": "spyde.actions.orientation_action.om_generate_library",
    "om_refine":           "spyde.actions.orientation_action.om_refine",
    "om_run":              "spyde.actions.orientation_action.om_run",
    "fv_preview":          "spyde.actions.find_vectors_action.fv_preview",
    "fv_tune":             "spyde.actions.find_vectors_action.fv_tune",
    "fv_run":              "spyde.actions.find_vectors_action.fv_run",
    "fv_stop":             "spyde.actions.find_vectors_action.fv_stop",
    "vom_generate_library": "spyde.actions.vector_orientation_om.vom_generate_library",
    "vom_refine":          "spyde.actions.vector_orientation_om.vom_refine",
    "vom_run":             "spyde.actions.vector_orientation_om.vom_run",
    "strain_run":          "spyde.actions.strain_action.strain_run",
    "strain_set_component": "spyde.actions.strain_action.strain_set_component",
    "strain_set_cif":      "spyde.actions.strain_action.strain_set_cif",
    "strain_set_rings":    "spyde.actions.strain_action.strain_set_rings",
    "ipf_set_direction":   "spyde.actions.ipf_view.ipf_set_direction",
    "tile_views":          "spyde.actions.views.tile_views",
    "set_composition":     "spyde.actions.composition.set_composition",
    "cod_search":          "spyde.actions.composition.cod_search",
    "cod_pick":            "spyde.actions.composition.cod_pick",
    "czb_auto":            "spyde.actions.center_zero_beam.czb_auto",
    "czb_manual_start":    "spyde.actions.center_zero_beam.czb_manual_start",
    "czb_manual":          "spyde.actions.center_zero_beam.czb_manual",
    "czb_manual_stop":     "spyde.actions.center_zero_beam.czb_manual_stop",
    "set_log_level":       "spyde.backend.log_stream.set_log_level",
}


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

        # Plot update poller. `dispatch` marshals the result-APPLY onto the main
        # asyncio thread (set later via set_main_loop, once the loop exists) — the
        # poll thread only detects done futures + reads shm; plot.update()/push
        # runs on the main thread, like the Qt app's queued plot_ready slot.
        self._main_loop = None
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

    # ── Startup ────────────────────────────────────────────────────────────────

    def start_dask(self) -> None:
        self.dask_manager.start()

    def set_main_loop(self, loop) -> None:
        """Register the main asyncio loop so the plot poller can marshal the
        result-apply onto this (main) thread. Call from app._main once the loop
        is running."""
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

        # A large file's lazy load is a one-time cold-cache disk read (reading the
        # header + building the dask graph for an 11 GB MRC can take tens of
        # seconds the FIRST time; the OS cache makes the next open instant). Say
        # so, and flag a busy state so the frontend can show a spinner instead of
        # looking hung. Emit the busy flag FIRST so it paints before the read.
        try:
            size_gb = os.path.getsize(path) / 1e9
        except OSError:
            size_gb = 0.0
        name = os.path.basename(path)
        hint = " (first open of a large file can take a while)" if size_gb >= 1 else ""
        emit({"type": "loading", "busy": True, "text": f"Reading {name}…{hint}"})
        emit_status(f"Reading {name}…{hint}")
        threading.Thread(
            target=self._load_file_thread,
            args=(path,),
            daemon=True,
            name=f"load-{name}",
        ).start()

    @staticmethod
    def _signal_spanning_chunks(sig, nav_chunk: int = 32):
        """Chunks for a lazy 2-D-signal dataset where each chunk holds WHOLE
        signal frames (``-1`` on the signal axes) and a small contiguous nav
        block.  Returns a ``chunks`` tuple to re-load with, or None if the
        dataset isn't a navigated 2-D-signal or its signal axes are already
        whole.

        Why: RosettaSciIO auto-chunks a 4-D MRC as a balanced cube (e.g.
        (90,90,90,90)) that SPLITS the signal axes — so reading one diffraction
        pattern ``data[iy,ix]`` pulls a 131 MB chunk spanning 90x90 nav
        positions and partial frames.  Whole-signal chunks make single-frame
        navigator access read one contiguous chunk, and the navigator sum is
        uniform across chunk boundaries."""
        try:
            am = sig.axes_manager
            nav_dim = am.navigation_dimension
            sig_dim = am.signal_dimension
            data = sig.data
            if sig_dim != 2 or nav_dim < 1 or not hasattr(data, "chunks"):
                return None
            # Signal axes already whole (one chunk each)? nothing to do.
            sig_chunks = data.chunks[nav_dim:]
            if all(len(c) == 1 for c in sig_chunks):
                return None
            nav_shape = data.shape[:nav_dim]
            nav = tuple(min(nav_chunk, int(n)) for n in nav_shape)
            return nav + (-1,) * sig_dim
        except Exception as e:
            log.debug("computing signal-spanning chunks failed: %s", e)
            return None

    def _load_file_thread(self, path: str) -> None:
        try:
            signal = hs.load(path, lazy=True)
            if not isinstance(signal, list):
                signal = [signal]
            # Re-load with whole-signal chunks when the reader split the signal
            # axes (cheap: a lazy reload only rebuilds the dask graph, ~0 s — it
            # does NOT read or shuffle data, unlike a rechunk()).
            for i, sig in enumerate(signal):
                ch = self._signal_spanning_chunks(sig)
                if ch is not None:
                    try:
                        reloaded = hs.load(path, lazy=True, chunks=ch)
                        signal = reloaded if isinstance(reloaded, list) else [reloaded]
                        log.debug("re-loaded %s with whole-signal chunks %s",
                                  os.path.basename(path), ch)
                    except Exception as e:
                        log.debug("re-load with signal-spanning chunks failed "
                                  "(%s); using reader default", e)
                    break
            # File read done — clear the busy flag (a nav-shape prompt or the
            # opened windows take over from here).
            emit({"type": "loading", "busy": False, "text": ""})
            # A single navigated signal (e.g. a 4D-STEM MRC scan) → let the user
            # confirm/override the navigation shape and set the real step size
            # (calibration) before opening. Everything else opens directly.
            if len(signal) == 1 and self._wants_nav_prompt(signal[0]):
                self._prompt_nav_shape(signal[0], path)
                return
            for sig in signal:
                self._add_signal(sig, source_path=path)
            self._add_recent(path)
            emit({"type": "recent_files", "paths": self._recent_files[:20]})
        except Exception as e:
            emit({"type": "loading", "busy": False, "text": ""})
            emit_error(f"Failed to load {os.path.basename(path)}: {e}")

    @staticmethod
    def _wants_nav_prompt(sig: BaseSignal) -> bool:
        """True for a signal where confirming the scan shape + step size is
        useful: a 2-D-signal dataset that is either already navigated (4D-STEM)
        or a flat stack of images (nav-dim 1) that the user may want to fold into
        a 2-D scan grid."""
        try:
            am = sig.axes_manager
            return am.signal_dimension == 2 and am.navigation_dimension >= 1
        except Exception:
            return False

    def _prompt_nav_shape(self, sig: BaseSignal, path: str) -> None:
        """Stash the loaded (lazy) signal and ask the frontend to confirm the
        navigation shape + step size. The reply arrives as the
        ``confirm_nav_shape`` action → :meth:`_confirm_nav_shape`."""
        am = sig.axes_manager
        nav_shape = list(am.navigation_shape)          # (x, y[, …]) display order
        n_patterns = int(np.prod(nav_shape)) if nav_shape else 0
        nav_axes = am.navigation_axes
        scale = float(nav_axes[0].scale) if nav_axes else 1.0
        units = (nav_axes[0].units if nav_axes else "") or ""
        if units in ("<undefined>",):
            units = ""
        self._pending_load = (sig, path)
        emit({
            "type": "nav_shape_prompt",
            "nav_shape": nav_shape,        # inferred (display x, y) order
            "n_patterns": n_patterns,      # total frames → factor options for a stack
            "signal_shape": list(am.signal_shape),
            "scale": scale,
            "units": units or "nm",
            "filename": os.path.basename(path),
        })

    def _confirm_nav_shape(self, payload: dict) -> None:
        """Apply the user's chosen navigation shape + step size to the stashed
        signal, then open it. ``nav_shape`` is in display (x, y) order; an empty
        / null shape means 'keep as loaded'."""
        pending = getattr(self, "_pending_load", None)
        if pending is None:
            return
        sig, path = pending
        self._pending_load = None
        try:
            nav_shape = payload.get("nav_shape") or None      # display (x, y) order
            step = payload.get("step_size")
            units = payload.get("units") or "nm"
            sig = self._apply_nav_shape(sig, nav_shape, step, units)
        except Exception as e:
            emit_error(f"Could not apply navigation shape: {e}")
            # Fall back to opening the signal as-loaded so the user isn't stuck.
        self._add_signal(sig, source_path=path)
        self._add_recent(path)
        emit({"type": "recent_files", "paths": self._recent_files[:20]})

    @staticmethod
    def _apply_nav_shape(sig: BaseSignal, nav_shape, step, units):
        """Reshape the navigation space to ``nav_shape`` (display x,y order) if it
        differs from the current shape, and calibrate the navigation axes to
        ``step``/``units``. Lazy-safe: the reshape is a dask-array view — the data
        is never materialised."""
        am = sig.axes_manager
        cur = list(am.navigation_shape)
        if nav_shape and [int(n) for n in nav_shape] != cur:
            # HyperSpy data layout is (nav reversed) + signal. nav_shape is
            # display (x, y), so the data's nav block is its reverse → (…, y, x).
            new_nav_data = tuple(reversed([int(n) for n in nav_shape]))
            sig_data = tuple(reversed([int(s) for s in am.signal_shape]))
            total_nav = int(np.prod(new_nav_data))
            cur_total = int(np.prod(cur)) if cur else 0
            if cur_total and total_nav != cur_total:
                raise ValueError(
                    f"nav shape {tuple(nav_shape)} ({total_nav} frames) ≠ "
                    f"{cur_total} frames in the data"
                )
            reshaped = sig.data.reshape(new_nav_data + sig_data)
            # Build a fresh signal of the same class so axes/metadata are
            # consistent with the new shape (rather than poking axes_manager).
            new = sig.__class__(reshaped)
            if getattr(sig, "_lazy", False) and not getattr(new, "_lazy", False):
                new = new.as_lazy()
            try:
                new.metadata = sig.metadata.deepcopy()
                stype = sig.metadata.get_item("Signal.signal_type", "")
                if stype:
                    new.set_signal_type(stype)
            except Exception as e:
                log.debug("carrying metadata across reshape failed: %s", e)
            sig, am = new, new.axes_manager
        # Calibrate the navigation axes' step size + units.
        if step:
            for ax in am.navigation_axes:
                ax.scale = float(step)
                ax.units = units
        return sig

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
            sig = self._load_example_lazy(loader)
            _apply_example_calibration(sig, name)
            self._add_signal(sig, source_path=None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            emit_error(f"Failed to load example {name}: {e}")

    def _load_example_lazy(self, loader) -> BaseSignal:
        """Load a pyxem example as a LAZY (Dask-backed) signal.

        pyxem example loaders forward ``**kwargs`` to ``hs.load``, so passing
        ``lazy=True`` reads the already-downloaded file straight off disk as a
        dask array — no eager 668 MB materialise, no zspy re-save. Falls back to
        an in-place ``as_lazy()`` wrap only if a loader doesn't honour ``lazy``.
        """
        for kwargs in ({"allow_download": True, "lazy": True}, {"lazy": True}):
            try:
                sig = loader(**kwargs)
            except TypeError:
                continue
            return sig if getattr(sig, "_lazy", False) else sig.as_lazy()
        # Loader doesn't accept those kwargs at all → eager once, wrap lazy.
        try:
            sig = loader(allow_download=True)
        except TypeError:
            sig = loader()
        return sig if getattr(sig, "_lazy", False) else sig.as_lazy()

    def _to_lazy(self, sig: BaseSignal, name: str) -> BaseSignal:
        """Return a lazy (Dask-backed) version of an in-memory *sig* via an
        in-place ``as_lazy()`` wrap (no disk round-trip)."""
        if getattr(sig, "_lazy", False):
            return sig
        try:
            return sig.as_lazy()
        except Exception:
            return sig

    def _add_signal(
        self,
        signal: BaseSignal,
        source_path: str | None = None,
        navigator_override: BaseSignal | None = None,
        selector_type=None,
    ):
        """Create a signal tree + plots for a loaded signal. Returns the tree.

        ``navigator_override`` supplies a pre-built navigator (e.g. a vectors
        count-map) so the base navigator is NOT recomputed from the full
        dataset — essential for the breaking transformations (Find Vectors).
        """
        from spyde.signal_tree import BaseSignalTree
        from spyde.drawing.plots.plot import Plot

        client = self.dask_manager.client
        tree = BaseSignalTree(
            root_signal=signal,
            session=self,
            distributed_client=client,
            selector_type=selector_type,
            navigator_override=navigator_override,
        )
        self.signal_trees.append(tree)

        # Open the MDI windows for this tree
        tree.open()

        title = signal.metadata.get_item("General.title", default=None)
        if title is None and source_path:
            title = os.path.splitext(os.path.basename(source_path))[0]

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
        try:
            from spyde.actions.composition import emit_composition
            emit_composition(tree, self._tree_window_ids(tree))
        except Exception as e:
            log.warning("composition emit failed: %s", e)

        emit_status(f"Loaded: {title or 'Signal'}")
        return tree

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

    def _emit_axes(self, tree) -> None:
        try:
            from spyde.metadata_extract import build_axes_list
            emit({
                "type": "axes_info",
                "window_ids": self._tree_window_ids(tree),
                "axes": build_axes_list(tree),
            })
        except Exception as e:
            log.warning("axes emit failed: %s", e)

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
        """Track a navigator's composite selector by its window id so the dock
        can toggle its crosshair/integration mode."""
        if not hasattr(self, "_nav_selectors"):
            self._nav_selectors = {}
        self._nav_selectors[window_id] = selector

    def set_selector_mode(self, window_id: int, integrate: bool) -> None:
        """Switch a navigator selector between crosshair and integrating mode."""
        sel = getattr(self, "_nav_selectors", {}).get(window_id)
        if sel is None or not hasattr(sel, "set_integrating"):
            return
        try:
            sel.set_integrating(bool(integrate))
            emit({
                "type": "selector_info",
                "window_id": window_id,
                "mode": "integrate" if integrate else "crosshair",
                "title": "Navigator",
            })
        except Exception as e:
            log.warning("set_selector_mode failed: %s", e)

    def _select_signal_node(self, plot, signal_id) -> None:
        """Switch *plot* to display the signal-tree node with the given id
        (emitted by toggle_signal_tree as id(node.signal))."""
        if plot is None or signal_id is None:
            return
        for sig in list(getattr(plot, "plot_states", {}).keys()):
            if id(sig) == signal_id:
                plot.set_plot_state(sig)
                self._reemit_signal_tree(plot)
                emit({"type": "status", "text": "Switched signal node"})
                return

    def _reemit_signal_tree(self, plot) -> None:
        """Re-push the workflow tree for *plot* (refreshes after a new node is
        added by a transform, and highlights the active node). No-op if the tree
        isn't available yet."""
        tree = getattr(plot, "signal_tree", None) if plot is not None else None
        root_node = getattr(tree, "root_node", None) if tree is not None else None
        if root_node is None:
            return

        def node_to_dict(node):
            return {
                "name": node.name, "signal_id": id(node.signal),
                "children": [node_to_dict(c) for c in node.children.values()],
            }
        try:
            active = id(plot.plot_state.current_signal)
        except Exception:
            active = None
        emit({
            "type": "signal_tree", "window_id": getattr(plot, "window_id", None),
            "tree": node_to_dict(root_node), "active_signal_id": active, "visible": True,
        })

    def _set_axis(self, plot, payload: dict) -> None:
        """Edit one axis property of the active window's root signal and
        recalibrate every plot in its tree. Writes back to the real
        axes_manager so the change is reflected in the dataset."""
        if plot is None:
            return
        tree = getattr(plot, "signal_tree", None)
        if tree is None:
            return
        index = payload.get("index")
        field = payload.get("field")
        value = payload.get("value")
        if index is None or field not in ("name", "units", "scale", "offset"):
            return
        try:
            axes = tree.root.axes_manager._axes
            if not (0 <= int(index) < len(axes)):
                return
            ax = axes[int(index)]
            if field == "scale":
                try:
                    new_scale = float(value)
                except (TypeError, ValueError):
                    return  # ignore non-numeric input mid-typing
                # Keep the ORIGIN PIXEL fixed when the scale changes: the pixel
                # where data == 0 is pixel0 = -offset/scale; to pin that same
                # pixel under the new scale, offset must scale with it:
                #   offset_new = offset_old * (scale_new / scale_old).
                # So the (0,0) point (e.g. the crosshair-marked centre) does not
                # drift when the user recalibrates the pixel size.
                old_scale = float(ax.scale)
                old_offset = float(ax.offset)
                ax.scale = new_scale
                if old_scale != 0.0:
                    ax.offset = old_offset * (new_scale / old_scale)
            elif field == "offset":
                try:
                    ax.offset = float(value)
                except (TypeError, ValueError):
                    return  # ignore non-numeric input mid-typing
            else:
                setattr(ax, field, str(value))
        except Exception as e:
            log.warning("set_axis failed: %s", e)
            return

        # Recalibrate: re-push every plot in the tree (re-reads the axes →
        # updated scale bar / extent) and re-emit the table + metadata.
        for p in list(self._plots):
            if getattr(p, "signal_tree", None) is tree:
                try:
                    p.update()
                except Exception as e:
                    log.debug("re-emitting plot update failed: %s", e)
        self._emit_axes(tree)
        try:
            from spyde.metadata_extract import build_metadata_dict
            emit({
                "type": "metadata",
                "window_ids": self._tree_window_ids(tree),
                "metadata": build_metadata_dict(tree),
            })
        except Exception as e:
            log.debug("re-emitting metadata failed: %s", e)

    def _set_offset_crosshair(self, plot, payload: dict) -> None:
        """Toggle a draggable "set origin" crosshair on the ACTIVE plot.

        The crosshair edits the offsets of the axes the active plot is drawn
        against, so it reads (0, 0) at the crosshair position:
          • signal plot    → the two SIGNAL axes' offsets
          • navigator plot → the two NAVIGATION axes' offsets
        Offsets are in real (calibrated) units; the tool starts at the current
        origin so it begins at the existing offset.

        payload {"on": True}  → drop the crosshair and update offsets as it moves.
                {"on": False} → remove the crosshair.
        """
        if plot is None:
            return
        tree = getattr(plot, "signal_tree", None)
        if tree is None:
            return
        on = bool(payload.get("on", False))
        plot2d = getattr(plot, "_plot2d", None)

        # always clear any existing crosshair first (idempotent). Keyed per-plot
        # so a signal-plot tool and a navigator-plot tool don't clobber each
        # other; store on the plot, not the shared tree.
        old = getattr(plot, "_offset_cross", None)
        if old is not None:
            # remove_widget() deletes the widget AND re-pushes the panel, so the
            # crosshair disappears on the FIRST toggle-off. A bare widget.hide()
            # only emits a targeted event that a later repaint overwrites, so the
            # ROI lingered until a second click (the reported "needs 2x").
            try:
                if plot2d is not None and hasattr(plot2d, "remove_widget"):
                    plot2d.remove_widget(old)
                else:
                    old.hide()
            except Exception as e:
                log.debug("removing offset crosshair failed: %s", e)
                try:
                    old.hide()
                except Exception:
                    pass
            plot._offset_cross = None
        if not on:
            return
        if plot2d is None:
            return

        # The axes the ACTIVE plot is drawn against: navigation axes for a
        # navigator, signal axes otherwise (mirrors Plot._axes_info / scale bar).
        try:
            if getattr(plot, "is_navigator", False):
                edit_ax = tree.root.axes_manager.navigation_axes
            else:
                edit_ax = plot.plot_state.current_signal.axes_manager.signal_axes
        except Exception as e:
            log.debug("offset crosshair axes lookup failed: %s", e)
            return
        if len(edit_ax) < 2:
            return
        ax_x, ax_y = edit_ax[0], edit_ax[1]
        w, h = int(ax_x.size), int(ax_y.size)
        # Start the crosshair on the pixel that is currently the origin
        # (data == 0): pixel = -offset/scale, expressed back in data coords for
        # the widget.  If the data origin is off-image, fall back to the centre.
        def _origin_data():
            sx, ox = float(ax_x.scale), float(ax_x.offset)
            sy, oy = float(ax_y.scale), float(ax_y.offset)
            pxi = (-ox / sx) if sx else w / 2.0
            pyi = (-oy / sy) if sy else h / 2.0
            if not (0 <= pxi <= w and 0 <= pyi <= h):
                pxi, pyi = w / 2.0, h / 2.0
            return pxi * sx + ox, pyi * sy + oy
        cx0, cy0 = _origin_data()
        try:
            cross = plot2d.add_crosshair_widget(cx=cx0, cy=cy0, color="#ffae57")
        except Exception as e:
            log.debug("offset crosshair add failed: %s", e)
            return
        plot._offset_cross = cross

        # Capture the calibration at toggle-on time as the FIXED reference for
        # converting the widget's data coords → pixel.  The widget reports data
        # coords under whatever offset is current, but we mutate the offset every
        # move; deriving the pixel from the live (mutating) offset would feed back
        # and drift.  Anchoring to the reference offset keeps a stationary
        # crosshair mapping to a stationary pixel across repeated applies.
        ref = {"sx": float(ax_x.scale), "ox": float(ax_x.offset),
               "sy": float(ax_y.scale), "oy": float(ax_y.offset)}

        def _apply(final: bool):
            # Recover the PIXEL the crosshair sits on (using the reference
            # calibration), then set each offset so that pixel maps to data 0:
            # offset_new = -pixel * scale.  Stable across repeated move events.
            try:
                sx, sy = ref["sx"], ref["sy"]
                px = (float(cross.cx) - ref["ox"]) / sx if sx else 0.0
                py = (float(cross.cy) - ref["oy"]) / sy if sy else 0.0
                ax_x.offset = -px * sx
                ax_y.offset = -py * sy
            except Exception as e:
                log.debug("offset crosshair update failed: %s", e)
                return
            # Live: re-emit the axes table so the dock shows the new offsets as
            # the user drags.  Defer the HOST-plot re-push (which rewrites the
            # displayed extent, and would shift the widget under the cursor) to
            # pointer-up so dragging stays smooth.  Only the host plot is
            # re-pushed — NOT every plot in the tree: re-pushing a navigator that
            # is progressively filling clobbers its live buffer, and editing one
            # plot's axes doesn't change the other's calibration.
            self._emit_axes(tree)
            if final:
                try:
                    plot.update()
                except Exception as e:
                    log.debug("re-pushing host plot after offset set failed: %s", e)
                # The displayed extent now reflects the new offset, so the widget
                # sits at data coord 0.  Re-anchor the reference to the new
                # calibration so a SUBSEQUENT drag is interpreted correctly.
                ref["ox"], ref["oy"] = float(ax_x.offset), float(ax_y.offset)

        def _on_event(event=None):
            etype = getattr(event, "type", None) or getattr(event, "name", None)
            _apply(final=(etype == "pointer_up"))

        try:
            cross.add_event_handler(_on_event, "pointer_move", "pointer_up")
        except Exception as e:
            log.debug("offset crosshair handler bind failed: %s", e)
        # Emit the current axes once so the dock reflects the starting state, but
        # do NOT mutate the offset at toggle-on: the crosshair already starts at
        # the existing origin, so the offset is unchanged until the user drags.
        self._emit_axes(tree)

    def _update_vi(self, window_id: int, name: str, params: dict) -> None:
        """A per-VI caret edit — apply new detector params and recompute that
        virtual image live."""
        art = self._action_artifacts.get((window_id, name))
        if not art:
            return
        act = art.get("action")
        if act is not None and hasattr(act, "update_live_params"):
            act.update_live_params(params)
            # A detector-type change rebuilds the selector — refresh the ref so
            # removal closes the current ROI.
            new_sel = getattr(act, "_selector", None)
            if new_sel is not None:
                art["selector"] = new_sel
        # Keep the source plot's VI list + the renderer chip in sync.
        src = self._plot_by_window_id(window_id)
        item = None
        for it in getattr(src, "_vi_items", []) or []:
            if it.get("name") == name:
                it.update({k: v for k, v in params.items()})
                item = it
                break
        if item is not None:
            emit({
                "type": "sub_item", "window_id": window_id,
                "action": item.get("parent_action", "Virtual Imaging"),
                "name": name, "color": item.get("color"),
                "vtype": item.get("type"), "calculation": item.get("calculation"),
                "active": True,
            })

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

    def _load_test_data(self) -> None:
        """Load a synthetic 4D-STEM dataset (no file, no Dask, no download).

        Test-only entry point so Playwright can exercise the full live
        navigator→signal interaction deterministically. Each nav position has a
        distinct single bright pixel so a selector move produces a visibly
        different diffraction pattern.
        """
        import numpy as np
        nav, sig = (8, 8), (32, 32)
        data = np.zeros(nav + sig, dtype=np.float32)
        for i in range(nav[0]):
            for j in range(nav[1]):
                data[i, j, (i * 4) % 32, (j * 4) % 32] = 255.0
                data[i, j, 16, 16] = 60.0  # faint common center
        s = hs.signals.Signal2D(data)
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type(electron_diffraction) on synthetic data failed: %s", e)
        self._add_signal(s, source_path="test_data")

    def _load_test_data_lazy(self) -> None:
        """Synthetic LAZY 4D-STEM data — exercises the lazy+Dask path (Future
        compute, worker-thread display) that the eager `_load_test_data` doesn't.
        The central disk intensity varies per nav position so a virtual image of
        it is clearly structured (not uniform/black)."""
        import numpy as np
        nav, sig = (8, 8), (32, 32)
        yy, xx = np.mgrid[0:32, 0:32]
        disk = ((xx - 16) ** 2 + (yy - 16) ** 2 <= 20).astype(np.float32)
        data = np.zeros(nav + sig, dtype=np.float32)
        for i in range(nav[0]):
            for j in range(nav[1]):
                data[i, j] = disk * (50.0 + i * 15.0 + j * 10.0)
        s = hs.signals.Signal2D(data).as_lazy()
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type(electron_diffraction) on synthetic data failed: %s", e)
        # CALIBRATE the signal axes (scale != 1, beam-centred). This is the real
        # scenario that exposed the "VI is just black" mask bug — anyplotlib ROI
        # widgets report PIXEL coords, so the detector mask must be built in pixel
        # space, not physical units. A scale=1 dataset hides that class of bug, so
        # the lazy test data is deliberately calibrated to guard against it.
        for ax in s.axes_manager.signal_axes:
            ax.scale = 0.1
            ax.offset = -(ax.size / 2.0) * 0.1
            ax.units = "1/nm"
        self._add_signal(s, source_path="test_data_lazy")

    def _load_test_data_lazy_chunked(self) -> None:
        """Test-only: LAZY 4D-STEM with MULTIPLE navigation chunks, so a crosshair
        drag crosses chunk boundaries and exercises the real distributed
        future→shm→PlotUpdateWorker→paint path (the in-chunk synthetic 8×8 is one
        chunk and never round-trips a worker). Each nav position has a single
        bright pixel at a position that varies with (iy, ix), so a frame change is
        unambiguous. nav=(24,24), signal=(32,32), nav chunks of 8 → a 3×3 chunk
        grid; signal axes span the full frame (storage-aligned)."""
        import numpy as np
        import dask.array as da
        ny, nx, ky, kx = 24, 24, 32, 32
        data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
        for i in range(ny):
            for j in range(nx):
                data[i, j, (i * 1) % ky, (j * 1) % kx] = 255.0
                data[i, j, 16, 16] = 60.0  # faint common centre
        dask_data = da.from_array(data, chunks=(8, 8, ky, kx))  # 3×3 nav chunk grid
        s = hs.signals.Signal2D(dask_data).as_lazy()
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type on chunked lazy data failed: %s", e)
        self._add_signal(s, source_path="test_data_lazy_chunked")

    def _test_nav_drag(self, targets: list) -> None:
        """Test-only: drive the navigator crosshair through a list of (x, y) nav
        cells server-side and report, per move, whether the SIGNAL plot's painted
        data actually changed. Bypasses the iframe widget harness so a Playwright
        test can deterministically exercise the distributed future→shm→worker→
        paint path. Emits {"type":"nav_drag_result", ...}.

        For each target: set the crosshair widget position, fire the selector,
        then poll the signal plot's current_data (the painted numpy frame) until
        it changes or a timeout. Records CHANGED / NO-CHANGE + the DP's argmax so
        we can see WHICH frame painted.
        """
        import time as _time
        try:
            tree = self.signal_trees[-1] if self.signal_trees else None
            if tree is None or tree.navigator_plot_manager is None:
                emit({"type": "nav_drag_result", "error": "no navigator tree"})
                return
            mgr = tree.navigator_plot_manager
            pw = next(iter(mgr.navigation_selectors.keys()))
            sel = mgr.navigation_selectors[pw][0]
            cross = getattr(sel, "_crosshair_selector", sel)
            child = next(iter(sel.children.keys()))

            def _frame_sig():
                d = getattr(child, "current_data", None)
                if isinstance(d, np.ndarray) and d.size:
                    return (int(d.argmax()), float(d.sum()))
                return None

            def _cd_kind():
                d = getattr(child, "current_data", None)
                return type(d).__name__

            results = []
            prev = _frame_sig()
            for (x, y) in targets:
                try:
                    cross._widget.cx = float(x)
                    cross._widget.cy = float(y)
                except Exception as e:
                    results.append({"x": x, "y": y, "changed": False, "err": str(e)})
                    continue
                sel.delayed_update_data(force=True)
                cur = prev
                t_end = _time.monotonic() + 3.0
                while _time.monotonic() < t_end:
                    cur = _frame_sig()
                    if cur is not None and cur != prev:
                        break
                    _time.sleep(0.03)
                changed = cur is not None and cur != prev
                results.append({"x": x, "y": y, "changed": bool(changed),
                                "sig": cur, "prev": prev, "cd_kind": _cd_kind()})
                prev = cur
            n_changed = sum(1 for r in results if r.get("changed"))
            emit({"type": "nav_drag_result", "total": len(targets),
                  "changed": n_changed, "results": results})
            log.info("[REDRAW] test_nav_drag: %d/%d moves changed the DP",
                     n_changed, len(targets))
        except Exception as e:
            log.exception("test_nav_drag failed")
            emit({"type": "nav_drag_result", "error": str(e)})

    def _load_test_vectors(self) -> None:
        """Test-only: load a small calibrated 4D-STEM stack (two disks per
        pattern) and run Find Diffraction Vectors on it, so the vectors-image
        window opens cleanly (no picker, no wizard occlusion). Lets Playwright
        exercise the downstream vector actions (Vector Virtual Imaging / Vector
        Orientation Mapping) E2E."""
        import numpy as np
        import hyperspy.api as hs
        nav, sig = (6, 6), (32, 32)
        yy, xx = np.mgrid[0:32, 0:32]
        # Four disks per pattern (≥4 vectors) so the downstream Vector
        # Orientation per-pattern fit actually runs (not skipped for too-few).
        spots = [(16, 16), (23, 9), (8, 21), (22, 24)]
        pat = np.zeros(sig, dtype=np.float32)
        for sxx, syy in spots:
            pat += ((xx - sxx) ** 2 + (yy - syy) ** 2 <= 7).astype(np.float32)
        data = np.zeros(nav + sig, dtype=np.float32)
        for i in range(nav[0]):
            for j in range(nav[1]):
                data[i, j] = pat * 100.0
        s = hs.signals.Signal2D(data)
        try:
            s.set_signal_type("electron_diffraction")
        except Exception as e:
            log.debug("set_signal_type(electron_diffraction) on synthetic data failed: %s", e)
        for ax in s.axes_manager.signal_axes:
            ax.scale = 0.1
            ax.offset = -(ax.size / 2.0) * 0.1
            ax.units = "1/nm"
        self._add_signal(s, source_path="test_vectors")

        src = next((p for p in self._plots
                    if not p.is_navigator and p.plot_state is not None), None)
        if src is None:
            emit_error("load_test_vectors: no active signal")
            return
        from spyde.actions.context import ActionContext
        from spyde.actions.find_vectors_action import find_diffraction_vectors
        ctx = ActionContext(plot=src, params={}, action_name="Find Diffraction Vectors")
        find_diffraction_vectors(
            ctx, sigma=1.0, kernel_radius=5, threshold=0.4,
            min_distance=3, subpixel=True,
        )

    def _run_test_orientation(self, plot) -> None:
        """Test-only Orientation Mapping with a built-in Al phase (no CIF dialog),
        so the full OM workflow can be exercised E2E (incl. lazy data) without a
        file picker. Mirrors `orientation_action.orientation_mapping`."""
        src = plot or next(
            (p for p in self._plots if not p.is_navigator and p.plot_state is not None),
            None,
        )
        if src is None:
            emit_error("run_test_orientation: no active signal")
            return
        tree = getattr(src, "signal_tree", None)
        if tree is None:
            emit_error("run_test_orientation: no signal tree")
            return

        def _work():
            try:
                from orix.crystal_map import Phase
                from diffpy.structure import Atom, Lattice, Structure
                from spyde.actions.orientation_action import run_orientation
                structure = Structure(
                    atoms=[Atom("Al", [0, 0, 0])],
                    lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90),
                )
                phase = Phase(name="Al", space_group=225, structure=structure)
                run_orientation(
                    self, tree.root, tree, [phase],
                    dict(accelerating_voltage=200.0, resolution=8.0),
                    dict(n_best=3, gamma=0.5), src_dp_plot=src,
                )
            except Exception as e:
                emit_error(f"run_test_orientation failed: {e}")
                log.exception("run_test_orientation failed")

        threading.Thread(target=_work, daemon=True, name="test-orientation").start()

    # ── Action dispatch ────────────────────────────────────────────────────────

    def dispatch_action(self, msg: dict) -> None:
        """Route an action message from Electron to the appropriate handler."""
        action = msg.get("action")
        payload = msg.get("payload", {})
        window_id = msg.get("window_id")

        plot = self._plot_by_window_id(window_id) if window_id is not None else None

        if action == "load_test_data":
            self._load_test_data()
        elif action == "load_test_data_lazy":
            self._load_test_data_lazy()
        elif action == "load_test_data_lazy_chunked":
            self._load_test_data_lazy_chunked()
        elif action == "test_nav_drag":
            # Run on a BACKGROUND thread: the drag loop sleeps/polls, and if it ran
            # on the main asyncio thread it would block loop.call_soon_threadsafe —
            # i.e. the very main-thread applies it's trying to observe.
            threading.Thread(
                target=self._test_nav_drag, args=(payload.get("targets") or [],),
                daemon=True, name="test-nav-drag",
            ).start()
        elif action == "load_test_vectors":
            self._load_test_vectors()
        elif action in _STAGED_HANDLERS:
            # Staged-wizard handlers (Orientation / Find-Vectors / Vector-OM /
            # Center-Zero-Beam) share the (session, plot, payload) signature and
            # are imported lazily so their heavy deps load only on first use.
            import importlib
            mod, fn = _STAGED_HANDLERS[action].rsplit(".", 1)
            getattr(importlib.import_module(mod), fn)(self, plot, payload)
        elif action == "run_test_orientation":
            # Test-only: run Orientation Mapping with a built-in Al phase (no CIF
            # dialog) on the active signal, so the E2E workflow can be driven
            # headlessly / in Playwright on lazy data.
            self._run_test_orientation(plot)
        elif action == "set_selector_mode":
            self.set_selector_mode(window_id, bool(payload.get("integrate")))
        elif action == "select_signal_node":
            self._select_signal_node(plot, payload.get("signal_id"))
        elif action == "set_axis":
            self._set_axis(plot, payload)
        elif action == "set_offset_crosshair":
            self._set_offset_crosshair(plot, payload)
        elif action == "set_overlay":
            self._set_overlay(plot, payload.get("name"),
                              bool(payload.get("visible", True)))
        elif action == "set_action_active":
            self._set_action_active(
                window_id, payload.get("name"), bool(payload.get("active"))
            )
        elif action == "update_vi":
            self._update_vi(window_id, payload.get("name"), payload.get("params", {}))
        elif action == "open_file":
            self.open_file(payload["path"])
        elif action == "confirm_nav_shape":
            self._confirm_nav_shape(payload)
        elif action == "set_signal_type":
            self._set_signal_type(plot, payload.get("signal_type", ""))
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
        elif action == "toolbar_action":
            self._dispatch_toolbar_action(
                plot, payload.get("name"), payload.get("params", {})
            )
        else:
            log.warning("Unknown action: %s", action)

    def _dispatch_toolbar_action(self, plot, name: str, params: dict) -> None:
        """Invoke a YAML-configured toolbar action by name on *plot*.

        The action function is resolved from TOOLBAR_ACTIONS and called with an
        ActionContext, so the same functions that ran under the Qt toolbar run
        here unchanged.  Parameter values collected by the Electron parameter
        panel arrive in *params* and are forwarded as kwargs.
        """
        if plot is None or not name:
            emit_error("Toolbar action: no active plot or action name")
            return

        # Actions whose modules still carry the Qt/interactive implementation and
        # haven't been ported to the host-agnostic template yet. Clicking them
        # gives a clear message instead of a confusing Qt-without-QApplication
        # traceback. (Virtual Imaging / FFT / Line Profile / Rebin ARE ported.)
        NOT_YET_PORTED: set = set()
        if name in NOT_YET_PORTED:
            emit_error(f"'{name}' is not yet available in the Electron build.")
            return

        try:
            import importlib
            from spyde import TOOLBAR_ACTIONS
            from spyde.actions.context import ActionContext

            meta = TOOLBAR_ACTIONS["functions"].get(name)
            if meta is None:
                # Sub-toolbar action (e.g. "add_virtual_image") — search the
                # subfunctions of every top-level action.
                for parent in TOOLBAR_ACTIONS["functions"].values():
                    subs = parent.get("subfunctions", {}) or {}
                    if name in subs:
                        meta = subs[name]
                        break
            if meta is None:
                emit_error(f"Unknown toolbar action: {name}")
                return
            module_path, _, attr = meta["function"].rpartition(".")
            target = getattr(importlib.import_module(module_path), attr)
            ctx = ActionContext(plot=plot, params=params, action_name=name)

            # A target may be either an Action subclass (template style) or a
            # plain function (legacy style). Both receive the same ActionContext.
            from spyde.actions.action import Action
            if isinstance(target, type) and issubclass(target, Action):
                result = target(ctx).run(**params)
            else:
                result = target(ctx, action_name=name, **params)
            self._track_action_artifacts(plot, name, result)
        except Exception as e:
            emit_error(f"Action '{name}' failed: {e}")
            log.exception("Action '%s' failed", name)

    def _track_action_artifacts(self, src_plot, name: str, result) -> None:
        """Remember the selector + output windows a RegionAction created so the
        toolbar can mark the action 'active' and hide them again on deselect."""
        if result is None or not hasattr(result, "active_children"):
            return
        src_wid = getattr(src_plot, "window_id", None)
        if src_wid is None:
            return
        out_wids = sorted({
            c.window_id for c in getattr(result, "active_children", [])
            if getattr(c, "window_id", None) is not None
        })
        self._action_artifacts[(src_wid, name)] = {"selector": result, "out_wids": out_wids}
        emit({"type": "action_active", "window_id": src_wid, "name": name, "active": True})

    def _set_overlay(self, plot, name: str, visible: bool) -> None:
        """Show/hide the live DP overlay(s) tied to a toolbar action — the marker
        overlay is only drawn while its action (caret) is SELECTED. The overlay
        still tracks the navigator while hidden, so re-selecting redraws the
        current frame."""
        tree = getattr(plot, "signal_tree", None) if plot is not None else None
        if tree is None or not name:
            return
        overlays = []
        if name == "Find Diffraction Vectors":
            overlays.append(getattr(tree, "_vector_overlay", None))
        elif name == "Orientation Mapping":
            overlays.append(getattr(tree, "_orientation_overlay", None))
            wiz = getattr(tree, "_om_wizard", None)
            if wiz:
                overlays.append(wiz.get("overlay"))
        elif name == "Vector Orientation Mapping":
            wiz = getattr(tree, "_vom_wizard", None)
            if wiz:
                overlays.append(wiz.get("overlay"))
        for ov in overlays:
            if ov is not None and hasattr(ov, "set_visible"):
                try:
                    ov.set_visible(visible)
                except Exception as e:
                    log.debug("toggling overlay visibility failed: %s", e)

    def _set_action_active(self, window_id: int, name: str, active: bool) -> None:
        """Deselecting an action hides the output window + ROI selector it made
        (Qt parity: an unchecked toolbar action removes its artifacts)."""
        key = (window_id, name)
        art = self._action_artifacts.get(key)
        if active or art is None:
            return
        # Closing each output plot also cleans its source ROI (parent_selector).
        for wid in art.get("out_wids", []):
            p = self._plot_by_window_id(wid)
            if p is not None:
                self._close_plot(p)
        try:
            art["selector"].close()
        except Exception as e:
            log.debug("closing action selector failed: %s", e)
        self._action_artifacts.pop(key, None)
        emit({"type": "action_active", "window_id": window_id, "name": name, "active": False})
        # If this was a virtual-image chip, drop it from the source plot's list
        # and tell the sub-toolbar to remove the chip.
        src = self._plot_by_window_id(window_id)
        if src is not None and hasattr(src, "_vi_items"):
            src._vi_items = [it for it in src._vi_items if it.get("name") != name]
        emit({"type": "sub_item", "window_id": window_id,
              "action": "Virtual Imaging", "name": name, "active": False})

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
            log.warning("set_colormap failed: %s", e)

    def _set_clim(self, plot, vmin, vmax) -> None:
        if plot is None:
            return
        try:
            plot.set_clim(vmin, vmax)
        except Exception as e:
            log.warning("set_clim failed: %s", e)

    def _close_window(self, window_id: int) -> None:
        plot = self._plot_by_window_id(window_id)
        if plot is None:
            # Backend already dropped it (or never had it) — still tell the
            # renderer to remove the window so the UI doesn't get stuck.
            emit({"type": "window_closed", "window_id": window_id})
            return
        try:
            tree = getattr(plot, "signal_tree", None)
            if tree is None:
                self._close_plot(plot)
                return
            # Scoping (per spec): the NAVIGATOR's X closes the whole tree (all
            # signals share its dataset); a signal window's X closes ONLY that
            # signal (and its selectors / action popouts). A lone signal plot
            # with no navigator falls through to closing its tree once empty.
            if getattr(plot, "is_navigator", False):
                self._close_tree(tree)
            else:
                self._close_signal_plot(plot, tree)
        except Exception as e:
            log.warning("close_window failed: %s", e)

    def _close_plot(self, plot) -> None:
        """Tear down a single plot and tell the renderer to drop its window."""
        wid = getattr(plot, "window_id", None)
        self._cleanup_plot_selectors(plot)
        try:
            plot.close()
        finally:
            self.unregister_plot(plot)
        self._forget_window(wid)

    def _close_signal_plot(self, plot, tree) -> None:
        """Close a single non-navigator signal window, leaving the rest of the
        tree open. Cleans up the plot's selectors / source ROI, then drops the
        tree entirely if nothing is left open."""
        self._close_plot(plot)
        try:
            if plot in getattr(tree, "signal_plots", []):
                tree.signal_plots.remove(plot)
        except Exception as e:
            log.debug("removing plot from tree.signal_plots failed: %s", e)
        # If no windows of this tree remain open, retire the tree.
        remaining = [p for p in self._plots if getattr(p, "signal_tree", None) is tree]
        if not remaining and tree in self.signal_trees:
            try:
                tree.close()
            except Exception as e:
                log.debug("retiring tree on last window close failed: %s", e)
            self.signal_trees.remove(tree)

    def _cleanup_plot_selectors(self, plot) -> None:
        """Close any selectors owned by / driving this plot, so closing a virtual
        image (etc.) also removes its source ROI from the parent plot."""
        # The selector on the PARENT plot that drives this output window.
        try:
            pw = getattr(plot, "plot_window", None)
            parent_sel = getattr(pw, "parent_selector", None)
            if parent_sel is not None and hasattr(parent_sel, "close"):
                parent_sel.close()
        except Exception as e:
            log.debug("closing parent selector failed: %s", e)
        # Selectors living on this plot itself.
        try:
            state = getattr(plot, "plot_state", None)
            for attr in ("plot_selectors", "signal_tree_selectors"):
                for sel in list(getattr(state, attr, []) or []):
                    if hasattr(sel, "close"):
                        try:
                            sel.close()
                        except Exception as e:
                            log.debug("closing plot selector failed: %s", e)
        except Exception as e:
            log.debug("iterating plot selectors for cleanup failed: %s", e)

    def _close_tree(self, tree: "BaseSignalTree") -> None:
        if tree not in self.signal_trees:
            return
        # Collect every plot/window belonging to this tree BEFORE teardown.
        plots = [p for p in self._plots if getattr(p, "signal_tree", None) is tree]
        window_ids = sorted({
            p.window_id for p in plots if getattr(p, "window_id", None) is not None
        })
        try:
            tree.close()
        except Exception as e:
            log.debug("closing tree in _close_tree failed: %s", e)
        for p in plots:
            self._cleanup_plot_selectors(p)
            try:
                p.close()
            except Exception as e:
                log.debug("closing plot in _close_tree failed: %s", e)
            self.unregister_plot(p)
        if tree in self.signal_trees:
            self.signal_trees.remove(tree)
        for wid in window_ids:
            self._forget_window(wid)

    def _forget_window(self, window_id: int | None) -> None:
        """Drop per-window backend state and tell the renderer to remove it."""
        if window_id is None:
            return
        if hasattr(self, "_nav_selectors"):
            self._nav_selectors.pop(window_id, None)
        # Drop any action-artifact entries that source from or output to this
        # window so a re-run starts clean and a closed output isn't "active".
        for k in [k for k, v in self._action_artifacts.items()
                  if k[0] == window_id or window_id in v.get("out_wids", [])]:
            self._action_artifacts.pop(k, None)
            # Tell the source window's toolbar to un-highlight the action.
            emit({"type": "action_active", "window_id": k[0], "name": k[1], "active": False})
        emit({"type": "window_closed", "window_id": window_id})

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
            log.warning("resize_figure failed: %s", e)

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
            log.warning("dispatch_figure_event failed: %s", e)

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

    def get_recent_files(self) -> list[str]:
        return list(self._recent_files[:20])

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        self._plot_worker.stop()
        self.dask_manager.shutdown()
        for tmpdir in self._example_temp_paths:
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                log.debug("removing example temp dir %s failed: %s", tmpdir, e)
        self._example_temp_paths.clear()

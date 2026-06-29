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
from spyde.backend._session_axes import AxesEditorMixin
from spyde.backend._session_actions import (
    ActionRouterMixin, _STAGED_HANDLERS, _TEST_ACTIONS, _TEST_ACTIONS_ENABLED,
)
from spyde.backend._session_testharness import TestHarnessMixin
from spyde.backend._session_windows import WindowManagerMixin
from spyde.dask_manager import DaskManager
from spyde.workers.plot_update_worker import PlotUpdateWorker

log = logging.getLogger(__name__)

# Per-frame navigator/redraw trace logs ([REDRAW2] APPLY/DROP) are gated behind
# this — they fire on every painted frame and flood the IPC log at DEBUG. Match
# the same env switch used in base_selector / update_functions / plot_update_worker.
_NAV_TIMING = os.environ.get("SPYDE_NAV_TIMING") == "1"

# _TEST_ACTIONS / _TEST_ACTIONS_ENABLED and _STAGED_HANDLERS now live in
# _session_actions (re-imported above so any `session._STAGED_HANDLERS` access
# still resolves).

if TYPE_CHECKING:
    from spyde.signal_tree import BaseSignalTree
    from spyde.drawing.plots.plot import Plot

SUPPORTED_EXTS = (".hspy", ".zspy", ".mrc", ".tif", ".tiff", ".de5")

# Extensions whose "file" is actually a DIRECTORY (a Zarr store). `.zspy` (and a
# bare `.zarr`) is a nested-group folder, not a single file — so `os.path.isfile`
# is False and `os.path.getsize` raises on it. Treat these as a present dataset
# when the directory exists.
_DIR_DATASET_EXTS = (".zspy", ".zarr")


def _path_ext(path: str) -> str:
    """Lowercased extension, working for both files and `.zspy`/`.zarr` dirs."""
    return os.path.splitext(path)[1].lower()


def _is_supported_dataset_path(path: str) -> bool:
    """True if ``path`` is a loadable dataset — a regular file, OR a directory
    store (`.zspy`/`.zarr`). A plain `os.path.isfile` wrongly rejects the latter."""
    ext = _path_ext(path)
    if ext in _DIR_DATASET_EXTS:
        return os.path.isdir(path)
    return os.path.isfile(path)


def _dataset_size_bytes(path: str) -> float:
    """Size in bytes — `os.path.getsize` for a file; a (bounded) recursive walk for
    a `.zspy`/`.zarr` directory store (used only to decide the 'large file' hint,
    so a best-effort sum is fine)."""
    try:
        if os.path.isdir(path):
            total = 0
            for root, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            return float(total)
        return float(os.path.getsize(path))
    except OSError:
        return 0.0

_DEFAULT_EXAMPLE_NAMES = (
    "mgo_nanocrystals",
    "small_ptychography",
    "zrnb_precipitate",
    "pdcusi_insitu",
    "sped_ag",
    "fe_multi_phase_grains",
)

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


class Session(AxesEditorMixin, ActionRouterMixin, TestHarnessMixin, WindowManagerMixin):
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
        # Restore the persisted recent-files list (capped to match _add_recent).
        try:
            self._recent_files = list(self._settings.get("recent_files", []))[:20]
        except Exception as e:
            log.debug("restoring recent files from settings failed: %s", e)

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
        """Load a HyperSpy-compatible file and open it in the MDI.

        ``path`` may be a regular file OR a directory store (``.zspy``/``.zarr``,
        which are Zarr nested-group folders, not single files)."""
        ext = _path_ext(path)
        if ext not in SUPPORTED_EXTS:
            emit_error(f"Unsupported file type: {ext}")
            return
        if not _is_supported_dataset_path(path):
            kind = "Folder" if ext in _DIR_DATASET_EXTS else "File"
            emit_error(f"{kind} not found: {path}")
            return

        # A large file's lazy load is a one-time cold-cache disk read (reading the
        # header + building the dask graph for an 11 GB MRC can take tens of
        # seconds the FIRST time; the OS cache makes the next open instant). Say
        # so, and flag a busy state so the frontend can show a spinner instead of
        # looking hung. Emit the busy flag FIRST so it paints before the read.
        size_gb = _dataset_size_bytes(path) / 1e9
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

    def _open_if_dense_vectors(self, sig, path: str) -> bool:
        """If *sig* is a saved dense-vectors carrier, reconstruct the vectors and
        open a Find-Vectors result tree. Returns True if it handled the file."""
        try:
            from spyde.signals.dense_diffraction_vectors import (
                is_dense_vectors_signal, from_dense_signal,
            )
            if not is_dense_vectors_signal(sig):
                return False
            vecs = from_dense_signal(sig)
            from spyde.actions.find_vectors_action import build_vectors_result_tree
            title = sig.metadata.get_item("General.title", default=None) \
                or os.path.splitext(os.path.basename(path))[0]
            build_vectors_result_tree(self, vecs, title=title)
            emit_status(f"Loaded {int(len(vecs.flat_buffer))} diffraction vectors")
            return True
        except Exception as e:
            emit_error(f"Failed to load diffraction vectors: {e}")
            log.exception("dense-vectors load failed for %s", path)
            return True   # we recognised it; don't fall through to image open

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
            # A saved diffraction-vectors result (dense flat-buffer carrier, see
            # spyde.signals.dense_diffraction_vectors) reopens as a Find-Vectors
            # result tree — reconstruct the vectors and rebuild the rendered-disk
            # window with the vector toolbar actions, NOT as raw image data.
            if len(signal) == 1 and self._open_if_dense_vectors(signal[0], path):
                self._add_recent(path)
                emit({"type": "recent_files", "paths": self._recent_files[:20]})
                return
            # A single navigated signal (e.g. a 4D-STEM MRC scan) → let the user
            # confirm/override the navigation shape and set the real step size
            # (calibration) before opening. But a SELF-DESCRIBING HyperSpy format
            # (.zspy/.zarr/.hspy) was written by us with correct axes — its shape
            # and calibration are already right, so open it straight away (no
            # prompt). The prompt is only for raw formats (.mrc/.tif) where the
            # scan grid is ambiguous.
            if (len(signal) == 1
                    and not self._is_self_describing(path)
                    and self._wants_nav_prompt(signal[0])):
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
    def _is_self_describing(path: str) -> bool:
        """True for HyperSpy-native formats that store full axes (shape +
        calibration) — .zspy/.zarr/.hspy. These reload with correct dimensions, so
        the scan-shape prompt (for ambiguous raw .mrc/.tif) should be skipped."""
        return _path_ext(path) in (".zspy", ".zarr", ".hspy")

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

    def open_stack(self, paths: "list[str]") -> None:
        """Stack several same-shaped datasets (e.g. a series of 4D-STEM MRC scans)
        into a single dataset with ONE extra leading navigation axis (a generic
        index 0,1,2,…). Files are stacked in the order given (selection order).

        Each file is loaded lazily and its per-file ``_info.txt`` (scan shape +
        calibration, handled by the MRC reader) is honoured. If the files don't all
        share the same nav/signal shape they are cropped to the common minimum and
        the user is warned. Everything stays lazy — a dask stack is a graph op, no
        data is read or materialised here."""
        paths = [p for p in (paths or []) if p]
        if len(paths) < 2:
            emit_error("Load Stack needs at least two files.")
            return
        bad = [p for p in paths
               if _path_ext(p) not in SUPPORTED_EXTS
               or not _is_supported_dataset_path(p)]
        if bad:
            emit_error(f"Cannot stack — missing/unsupported: "
                       f"{', '.join(os.path.basename(b) for b in bad)}")
            return
        names = [os.path.basename(p) for p in paths]
        emit({"type": "loading", "busy": True,
              "text": f"Stacking {len(paths)} files…"})
        emit_status(f"Stacking {len(paths)} files…")
        threading.Thread(
            target=self._load_stack_thread,
            args=(paths, names),
            daemon=True,
            name=f"load-stack-{len(paths)}",
        ).start()

    def _load_stack_thread(self, paths: "list[str]", names: "list[str]") -> None:
        import dask.array as da
        try:
            sigs = []
            for p in paths:
                s = hs.load(p, lazy=True)
                if isinstance(s, list):
                    if len(s) != 1:
                        raise ValueError(
                            f"{os.path.basename(p)} holds {len(s)} signals; "
                            "stack only supports single-signal files."
                        )
                    s = s[0]
                # Re-load with whole-signal chunks if the reader split the signal
                # axes — a lazy reload is ~0 s (rebuilds the graph, no data move),
                # and stacking members that are ALREADY signal-spanning-chunked
                # avoids ever rechunking the (huge) 5-D stack afterward. See the
                # "storage-aligned chunking — never rechunk live" rule in CLAUDE.md.
                ch = self._signal_spanning_chunks(s)
                if ch is not None:
                    try:
                        r = hs.load(p, lazy=True, chunks=ch)
                        s = r[0] if isinstance(r, list) else r
                    except Exception as e:
                        log.debug("stack member %s signal-chunk reload failed: %s",
                                  os.path.basename(p), e)
                sigs.append(s)

            # All members must share the SAME ndim/axis layout to stack coherently.
            ndims = {s.data.ndim for s in sigs}
            if len(ndims) != 1:
                raise ValueError(
                    "Files have different dimensionality "
                    f"({sorted(ndims)}); cannot stack."
                )
            ref_am = sigs[0].axes_manager
            if ref_am.signal_dimension != 2:
                raise ValueError(
                    "Load Stack expects 2-D-signal datasets (e.g. 4D-STEM); "
                    f"got signal_dimension={ref_am.signal_dimension}."
                )

            # Crop every member to the common (minimum) shape per axis, warning if
            # any file actually had to be cropped. Shapes are full array shapes
            # (nav-reversed + signal), so a per-axis min keeps axes aligned.
            shapes = [tuple(int(d) for d in s.data.shape) for s in sigs]
            common = tuple(min(dim) for dim in zip(*shapes))
            cropped_files = []
            for i, s in enumerate(sigs):
                if shapes[i] != common:
                    cropped_files.append(names[i])
                    slicer = tuple(slice(0, c) for c in common)
                    sigs[i] = s.__class__(s.data[slicer])
                    if getattr(s, "_lazy", False) and not getattr(sigs[i], "_lazy", False):
                        sigs[i] = sigs[i].as_lazy()
            if cropped_files:
                emit_status(
                    f"Stack: cropped {len(cropped_files)} file(s) to common "
                    f"shape {common}: {', '.join(cropped_files)}"
                )
                log.warning("Load Stack cropped to common shape %s: %s",
                            common, cropped_files)

            # Stack the dask arrays along a NEW leading axis (becomes the slowest
            # navigation axis). da.stack on lazy arrays is a pure graph op.
            arrs = [s.data for s in sigs]
            stacked = da.stack(arrs, axis=0)

            # Build a fresh signal of the members' class so the signal axes + type
            # carry over; the new leading axis defaults to a generic index.
            ref = sigs[0]
            new = ref.__class__(stacked)
            if not getattr(new, "_lazy", False):
                new = new.as_lazy()
            try:
                new.metadata = ref.metadata.deepcopy()
                stype = ref.metadata.get_item("Signal.signal_type", "")
                if stype:
                    new.set_signal_type(stype)
            except Exception as e:
                log.debug("carrying metadata onto stack failed: %s", e)

            # Copy each EXISTING navigation/signal axis's calibration from the
            # reference (the new leading stack axis keeps its default index scale).
            # new nav axes are the old nav axes shifted by one (the stack axis is
            # navigation axis 0 in hyperspy's reversed nav order → display-last).
            try:
                self._carry_axes_from_reference(new, ref)
            except Exception as e:
                log.debug("carrying axis calibration onto stack failed: %s", e)

            # NOTE: do NOT rechunk the stacked 5-D array here — that would shuffle
            # the whole multi-GB stack. Members were already reloaded with
            # signal-spanning chunks above, and da.stack adds a size-1 chunk on the
            # new leading axis, so the stack is navigator-friendly out of the box.

            emit({"type": "loading", "busy": False, "text": ""})

            # Open directly — NO nav-shape prompt. Each member's _info.txt already
            # gave the correct scan shape + calibration (carried over above); the
            # only new axis is the stack index, which is intentionally a generic
            # 0,1,2,… (the user can rename/calibrate it in the Axes dock). Running
            # the single-file prompt here would also wrongly stamp the scan step
            # onto the stack axis (it calibrates every nav axis).
            self._add_signal(new, source_path=paths[0])
            emit_status(
                f"Stacked {len(paths)} files → "
                f"{tuple(new.axes_manager.navigation_shape)} nav "
                f"× {tuple(new.axes_manager.signal_shape)} signal"
            )
        except Exception as e:
            emit({"type": "loading", "busy": False, "text": ""})
            emit_error(f"Failed to stack files: {e}")

    @staticmethod
    def _carry_axes_from_reference(new: BaseSignal, ref: BaseSignal) -> None:
        """Copy scale/offset/units/name from the reference member's axes onto the
        matching axes of the stacked signal. The stack added ONE leading axis
        (hyperspy navigation axis index 0), so each reference axis maps to the
        new signal's axis at the next higher index."""
        new_axes = list(new.axes_manager._axes)
        ref_axes = list(ref.axes_manager._axes)
        # new_axes[0] is the stack axis (leave as default index). The remaining
        # new axes line up 1:1 with the reference's axes, in order.
        for ref_ax, new_ax in zip(ref_axes, new_axes[1:]):
            try:
                new_ax.scale = ref_ax.scale
                new_ax.offset = ref_ax.offset
                new_ax.units = ref_ax.units
                if getattr(ref_ax, "name", None):
                    new_ax.name = ref_ax.name
            except Exception:
                pass

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

    def _resolve_save_plot(self, plot):
        """Pick the plot to save: the one passed (from its window), else the
        active window's, else the sole signal plot. The File→Save menu sends no
        window id (it can't know the focused window), so fall back gracefully."""
        if plot is not None:
            return plot
        if self._active_window_id is not None:
            p = self._plot_by_window_id(self._active_window_id)
            if p is not None:
                return p
        # Last resort: if exactly one plot carries a signal, save that.
        with_sig = [p for p in self._plots
                    if getattr(getattr(p, "plot_state", None), "current_signal", None)
                    is not None]
        return with_sig[0] if len(with_sig) == 1 else None

    @staticmethod
    def _vectors_for_plot(plot):
        """The :class:`SpyDEDiffractionVectors` attached to *plot*'s tree, or None.

        A Find-Vectors result tree carries ``tree.diffraction_vectors``; this is
        how Save knows to write the dense vectors carrier instead of the rendered
        image, and lets any vector-aware code reach the result from a plot."""
        tree = getattr(plot, "signal_tree", None)
        return getattr(tree, "diffraction_vectors", None) if tree is not None else None

    def _save_signal(self, path: str | None, plot) -> None:
        if path is None:
            emit_error("Save: no path given")
            return
        plot = self._resolve_save_plot(plot)
        if plot is None:
            emit_error("Save: click a signal window first, then Save.")
            return
        signal = getattr(getattr(plot, "plot_state", None), "current_signal", None)
        if signal is None:
            emit_error("Save: no signal in the active window")
            return
        # A Find-Vectors result window's signal is the lazy RENDERED-disk image
        # (to_rendered_dask). Saving THAT would compute + serialise every rendered
        # frame (huge, slow) and lose the actual vectors. Instead save the dense
        # vectors carrier (flat buffer + calibration) — tiny, instant, lossless,
        # and reloads back into a vectors result tree. Detect via the tree's
        # attached `diffraction_vectors`.
        vecs = self._vectors_for_plot(plot)
        if vecs is not None:
            from spyde.signals.dense_diffraction_vectors import to_dense_signal
            signal = to_dense_signal(vecs)
        # Default to the .zspy Zarr folder store: if the user typed a name with no
        # extension (or an unknown one), append .zspy rather than letting hyperspy
        # guess or error. A writable extension the user explicitly chose is kept.
        _WRITABLE_EXTS = (".zspy", ".hspy", ".zarr", ".tif", ".tiff")
        if _path_ext(path) not in _WRITABLE_EXTS:
            path = path + ".zspy"

        name = os.path.basename(path)
        # Saving a lazy multi-GB dataset to Zarr is a real compute (it reads the
        # whole array). Run it OFF the asyncio event loop so the UI stays live,
        # and show a busy indicator (start → finish) — the "nice save dialog".
        emit({"type": "loading", "busy": True, "text": f"Saving {name}…"})
        emit_status(f"Saving {name}…")
        threading.Thread(
            target=self._save_signal_thread,
            args=(signal, path, name),
            daemon=True,
            name=f"save-{name}",
        ).start()

    def _save_signal_thread(self, signal, path: str, name: str) -> None:
        try:
            import dask
            # Prefer the distributed cluster when the data is lazy and a client is
            # up — the store runs on the workers (true off-load, progress visible
            # on the dashboard) instead of the GUI process. Otherwise a threaded
            # local save. Either way this is on a daemon thread, never the loop.
            client = getattr(self.dask_manager, "client", None)
            is_lazy = bool(getattr(signal, "_lazy", False))
            t0 = time.time()
            if is_lazy and client is not None:
                # hyperspy writes the Zarr store lazily; computing under the
                # distributed scheduler dispatches the chunk writes to workers.
                with dask.config.set(scheduler=client):
                    signal.save(path, overwrite=True)
            else:
                with dask.config.set(scheduler="synchronous"):
                    signal.save(path, overwrite=True)
            dt = time.time() - t0
            emit({"type": "loading", "busy": False, "text": ""})
            emit({"type": "saved", "path": path})
            emit_status(f"Saved {name} ({dt:.1f}s)")
        except Exception as e:
            emit({"type": "loading", "busy": False, "text": ""})
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

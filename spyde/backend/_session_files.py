"""
_session_files.py — FileLoaderMixin + file helpers extracted from session.py.

Owns file/stack/example loading, the nav-shape prompt round-trip, and signal
saving. The module-level file helpers (``_path_ext``, ``_is_supported_dataset_path``,
``_dataset_size_bytes``, ``SUPPORTED_EXTS``, …) live here and are re-exported from
``session.py`` so ``from spyde.backend.session import _path_ext`` still works.

The mixin only USES ``self.<attr>`` / ``self.<method>`` (``self._add_signal``,
``self._add_recent``, ``self._plot_by_window_id`` …) provided by the final Session.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np
import hyperspy.api as hs
from hyperspy.signal import BaseSignal

from spyde.backend import ipc
from spyde.backend.ipc import emit_status, emit_error

log = logging.getLogger(__name__)

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
    except (AttributeError, KeyError, TypeError) as e:
        # Applying a KNOWN example dataset's hardcoded calibration should always
        # succeed; a failure here is a regression (hyperspy axes API change,
        # unexpected signal shape) worth surfacing, not swallowing at debug level.
        log.warning("applying calibration to signal axes failed: %s", e)


class FileLoaderMixin:
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
        ipc.emit({"type": "loading", "busy": True, "text": f"Reading {name}…{hint}"})
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

    # Target bytes per chunk for whole-signal-frame chunking. ~64 MB keeps a
    # single-frame read cheap without a per-chunk explosion: a 128x128 uint16 DP
    # (32 KB) packs ~2000 frames/chunk (capped at NAV_CHUNK_MAX below), while an
    # 8k x 8k uint8/float frame (64-256 MB) forces exactly ONE frame per chunk.
    _CHUNK_TARGET_BYTES = 64 << 20  # 64 MB
    _NAV_CHUNK_MAX = 32             # never span more than this many nav positions

    @classmethod
    def _signal_spanning_chunks(cls, sig, nav_chunk: "int | None" = None):
        """Chunks for a lazy 2-D-signal dataset where each chunk holds WHOLE
        signal frames (``-1`` on the signal axes) and a small contiguous nav
        block.  Returns a ``chunks`` tuple to re-load with, or None if the
        dataset isn't a navigated 2-D-signal or its chunking is already fine.

        Why: RosettaSciIO auto-chunks a 4-D MRC as a balanced cube (e.g.
        (90,90,90,90)) that SPLITS the signal axes — so reading one diffraction
        pattern ``data[iy,ix]`` pulls a 131 MB chunk spanning 90x90 nav
        positions and partial frames.  Whole-signal chunks make single-frame
        navigator access read one contiguous chunk, and the navigator sum is
        uniform across chunk boundaries.

        The nav-block size is **adapted to the frame byte size** so a chunk stays
        near ``_CHUNK_TARGET_BYTES`` regardless of frame resolution. The old flat
        ``nav_chunk=32`` was tuned for 128-256 px DP frames; for an in-situ movie
        of 8k x 8k images it would make a 512 MB chunk (32 x 64 MB) and read half
        a gigabyte to show one frame — see benchmarks.md "movie playback". Now a
        big frame gets 1 frame/chunk, a small DP many. Pass ``nav_chunk`` to force
        a value (tests)."""
        try:
            am = sig.axes_manager
            nav_dim = am.navigation_dimension
            sig_dim = am.signal_dimension
            data = sig.data
            if sig_dim != 2 or nav_dim < 1 or not hasattr(data, "chunks"):
                return None

            # Frame byte size drives the adaptive nav-block target.
            sig_shape = data.shape[nav_dim:]
            frame_bytes = int(np.prod(sig_shape)) * data.dtype.itemsize
            if nav_chunk is None:
                target = max(1, cls._CHUNK_TARGET_BYTES // max(1, frame_bytes))
                nav_chunk = int(min(cls._NAV_CHUNK_MAX, target))
                # A MOVIE time navigator reads ONE frame per move and jumps around
                # in time, so packing several frames per chunk (e.g. 4 for a 16 MB
                # frame → a 64 MB read to show 1 frame) is wasted I/O — each move
                # crosses a chunk. Force 1 frame/chunk for a movie so a scrub reads
                # exactly the frame it shows. (A 4D-STEM DP navigator dwells within
                # a chunk, so there the multi-frame pack is a genuine cache win.)
                if cls._is_movie_time_axis(sig):
                    nav_chunk = 1

            sig_chunks = data.chunks[nav_dim:]
            sig_whole = all(len(c) == 1 for c in sig_chunks)
            # Current nav-block size (frames per chunk on the fastest-varying nav
            # axis the reader split); used to decide if a re-chunk is warranted.
            cur_nav0 = data.chunks[0][0] if data.chunks and data.chunks[0] else 1

            # If the signal axes are ALREADY whole AND the reader's nav block is
            # no bigger than our target, the chunking is fine — don't rebuild the
            # graph for nothing (the common self-describing / already-good case).
            if sig_whole and cur_nav0 <= nav_chunk:
                return None

            nav_shape = data.shape[:nav_dim]
            nav = tuple(min(nav_chunk, int(n)) for n in nav_shape)
            return nav + (-1,) * sig_dim
        except Exception as e:
            log.debug("computing signal-spanning chunks failed: %s", e)
            return None

    def _load_file_thread(self, path: str) -> None:
        try:
            # Wait for the Dask cluster (see _load_example_thread) — a file opened
            # during startup queues here rather than racing ahead with no client.
            self._await_dask()
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
            ipc.emit({"type": "loading", "busy": False, "text": ""})
            # A saved diffraction-vectors result (dense flat-buffer carrier, see
            # spyde.signals.dense_diffraction_vectors) reopens as a Find-Vectors
            # result tree — reconstruct the vectors and rebuild the rendered-disk
            # window with the vector toolbar actions, NOT as raw image data.
            if len(signal) == 1 and self._open_if_dense_vectors(signal[0], path):
                self._add_recent(path)
                ipc.emit({"type": "recent_files", "paths": self._recent_files[:20]})
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
                self._maybe_set_insitu_signal_type(sig)
                self._add_signal(sig, source_path=path)
            self._add_recent(path)
            ipc.emit({"type": "recent_files", "paths": self._recent_files[:20]})
        except Exception as e:
            ipc.emit({"type": "loading", "busy": False, "text": ""})
            emit_error(f"Failed to load {os.path.basename(path)}: {e}")

    @staticmethod
    def _is_self_describing(path: str) -> bool:
        """True for HyperSpy-native formats that store full axes (shape +
        calibration) — .zspy/.zarr/.hspy. These reload with correct dimensions, so
        the scan-shape prompt (for ambiguous raw .mrc/.tif) should be skipped."""
        return _path_ext(path) in (".zspy", ".zarr", ".hspy")

    # Axis names identifying a MOVIE / sequential-stack leading axis (NOT a
    # foldable spatial scan): a TIME axis (DE-MRC non-scanning movies →
    # name="time", units="sec") OR a generic stack index (name "z"/"index"/
    # "stack"/"frame"/"slice", which DE movies use). For any of these, folding
    # the stack into a 2-D scan grid or stamping a spatial nm step is wrong.
    _TIME_AXIS_NAMES = ("time", "t", "frame", "frames", "z", "index", "stack",
                        "slice", "image", "images")
    _TIME_AXIS_UNITS = ("s", "sec", "secs", "second", "seconds", "ms",
                        "millisecond", "milliseconds", "us", "µs", "min", "minute")
    # A movie frame is a real IMAGE; a raw 4D-STEM DP stack has small detector
    # frames. Frames at least this size (either axis) are treated as movie images
    # even if the axis name is uninformative — a fold-to-grid prompt on a
    # 2048²+ "diffraction pattern" is never what the user wants.
    _MOVIE_MIN_FRAME_PX = 1024

    @classmethod
    def _is_movie_time_axis(cls, sig: BaseSignal) -> bool:
        """True when the dataset is an in-situ MOVIE / image stack: nav-dim 1 with
        2-D image frames, whose single navigation axis is a time/stack axis (by
        name or units) OR whose frames are large images (≥ _MOVIE_MIN_FRAME_PX).
        Such a dataset opens straight as a movie — NOT folded into a 2-D scan grid,
        and NOT given a spatial step on its sequence axis (the prompt does both)."""
        try:
            am = sig.axes_manager
            if am.signal_dimension != 2 or am.navigation_dimension != 1:
                return False
            ax = am.navigation_axes[0]
            name = str(getattr(ax, "name", "") or "").strip().lower()
            units = str(getattr(ax, "units", "") or "").strip().lower()
            if units in ("<undefined>",):
                units = ""
            if name in cls._TIME_AXIS_NAMES or units in cls._TIME_AXIS_UNITS:
                return True
            # Large image frames → a movie regardless of the (uninformative) name.
            sh = tuple(int(s) for s in am.signal_shape)
            return bool(sh) and max(sh) >= cls._MOVIE_MIN_FRAME_PX
        except Exception:
            return False

    @classmethod
    def _wants_nav_prompt(cls, sig: BaseSignal) -> bool:
        """True for a signal where confirming the scan shape + step size is
        useful: a 2-D-signal dataset that is either already navigated (4D-STEM)
        or a flat stack of images (nav-dim 1) that the user may want to fold into
        a 2-D scan grid.

        A calibrated in-situ MOVIE (nav-dim 1, time axis) is EXCLUDED — it opens
        straight as a movie with its time navigator, no fold-to-grid prompt (which
        would otherwise stamp a spatial nm step on the time axis)."""
        try:
            am = sig.axes_manager
            if not (am.signal_dimension == 2 and am.navigation_dimension >= 1):
                return False
            if cls._is_movie_time_axis(sig):
                return False
            return True
        except Exception:
            return False

    @classmethod
    def _maybe_set_insitu_signal_type(cls, sig: BaseSignal) -> None:
        """Tag a freshly-loaded signal as ``insitu`` (see spyde.signals.insitu)
        when it's recognised as an in-situ MOVIE by the same
        ``_is_movie_time_axis`` check that already decides chunking + skips the
        nav-shape prompt for it — this reuses that existing movie-detection
        condition rather than inventing a new heuristic. Typing it drives the
        Play/Fast Forward toolbar gate (spyde/toolbars.yaml ``signal_types:
        [insitu]``). No-op (best-effort) if the signal already carries a more
        specific signal_type someone set deliberately, or on any error."""
        try:
            if not cls._is_movie_time_axis(sig):
                return
            current = str(getattr(sig, "_signal_type", "") or "")
            if current and current != "insitu":
                return
            sig.set_signal_type("insitu")
        except Exception as e:
            log.debug("auto-tagging in-situ movie signal_type failed: %s", e)

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
        ipc.emit({
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
        ipc.emit({"type": "recent_files", "paths": self._recent_files[:20]})

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
        ipc.emit({"type": "loading", "busy": True,
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

            ipc.emit({"type": "loading", "busy": False, "text": ""})

            # Open directly — NO nav-shape prompt. Each member's _info.txt already
            # gave the correct scan shape + calibration (carried over above); the
            # only new axis is the stack index, which is intentionally a generic
            # 0,1,2,… (the user can rename/calibrate it in the Axes dock). Running
            # the single-file prompt here would also wrongly stamp the scan step
            # onto the stack axis (it calibrates every nav axis).
            self._add_signal(new, source_path=paths[0], enable_nav_sidecar=False)
            emit_status(
                f"Stacked {len(paths)} files → "
                f"{tuple(new.axes_manager.navigation_shape)} nav "
                f"× {tuple(new.axes_manager.signal_shape)} signal"
            )
        except Exception as e:
            ipc.emit({"type": "loading", "busy": False, "text": ""})
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
        from spyde.backend.example_download import (
            DownloadCancelled, patched_example_downloader,
        )
        try:
            # Wait for the Dask cluster before registering the signal — _add_signal
            # builds the navigator compute, which needs the client. A load fired
            # during startup queues here instead of racing ahead with a None client.
            self._await_dask()
            import pyxem.data as _pxd
            loader = getattr(_pxd, name, None)
            if loader is None:
                emit_error(f"Unknown example dataset: {name}")
                return
            # Progress + cancel for the (possible) pooch download: the renderer
            # shows a toast with a bar and a Cancel button (download_progress /
            # download_done messages; download_cancel action). A cache hit never
            # downloads → no toast.
            try:
                with patched_example_downloader(f"example:{name}", name):
                    sig = self._load_example_lazy(loader)
            except DownloadCancelled:
                emit_status(f"Download cancelled: {name}")
                return
            _apply_example_calibration(sig, name)
            self._maybe_set_insitu_signal_type(sig)
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

    # ── Save ─────────────────────────────────────────────────────────────────

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
        ipc.emit({"type": "loading", "busy": True, "text": f"Saving {name}…"})
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
            ipc.emit({"type": "loading", "busy": False, "text": ""})
            ipc.emit({"type": "saved", "path": path})
            emit_status(f"Saved {name} ({dt:.1f}s)")
        except Exception as e:
            ipc.emit({"type": "loading", "busy": False, "text": ""})
            emit_error(f"Save failed: {e}")

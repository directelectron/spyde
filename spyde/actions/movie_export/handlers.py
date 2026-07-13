"""
handlers.py — the Movie Export staged wizard handlers (wizard key ``mvx``).

Gate: an in-situ movie tree (root ``_signal_type == "insitu"``). The wizard reads
the time axis via the playback conventions (``navigation_axes[0].scale/.units`` →
seconds), probes for a bundled ffmpeg, seeds defaults from the clicked plot's
colormap / contrast, and drives :func:`spyde.actions.movie_export.pipeline.export_movie`
on a worker thread — memory-safe (one lazy frame slice at a time), generation-
guarded, cancellable (per-frame check + the tree's cancel registry), with
partial-file cleanup on cancel/failure.

Message contracts (the renderer is written against these EXACT shapes):

* ``mvx_state`` — the authoritative wizard state (see :meth:`MovieExportState.emit`):
    {"type":"mvx_state","window_id":int,"ffmpeg_ok":bool,"running":bool,
     "n_frames":int,"time":{"scale_s":float,"units":str},
     "params":{"fps","downsample","stride","t_start","t_end","cmap","clim",
               "timestamp","scalebar","annotations":[...]},
     "traces":[{"id","label","color","units"}]}
* ``mvx_done`` — export success: {"type":"mvx_done","path":str,"frames":int}.
* errors / status via ``emit_error`` / ``emit_status`` / ``emit_progress``.

All handlers share the uniform ``fn(session, plot, payload)`` signature and are
registered in :data:`spyde.actions.registry.STAGED_HANDLERS`.
"""
from __future__ import annotations

import logging
import os

from spyde.backend import ipc
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.lifecycle import bump_generation, is_current, run_on_worker
from spyde.actions.playback import _units_to_seconds
from spyde.actions.movie_export import traces as _traces

log = logging.getLogger(__name__)

# ── the wizard parameter schema (single source of truth; test_wizard_schemas) ──
# Same dict spec as toolbars.yaml `parameters:`. Resolved host-agnostically via
# registry.wizard_parameters("mvx"). `annotations` and `clim` are structured
# lists the wizard mutates via mvx_tune (not simple form controls), so they are
# NOT declared here — only the scalar/enum/bool controls the param panel renders.
DEFAULTS = dict(fps=10, downsample=1, stride=1, cmap="gray",
                timestamp=True, scalebar=True)

PARAMETERS = {
    "fps": {
        "name": "Frame rate (fps)", "type": "int", "default": DEFAULTS["fps"],
        "min": 1, "max": 60,
    },
    "downsample": {
        "name": "Downsample ×", "type": "int", "default": DEFAULTS["downsample"],
        "min": 1, "max": 8,
    },
    "stride": {
        "name": "Frame stride", "type": "int", "default": DEFAULTS["stride"],
        "min": 1, "max": 20,
    },
    "cmap": {
        "name": "Colormap", "type": "enum", "default": DEFAULTS["cmap"],
        "choices": ["gray", "viridis", "magma", "inferno", "plasma",
                    "cividis", "hot", "jet"],
    },
    "timestamp": {
        "name": "Timestamp", "type": "bool", "default": DEFAULTS["timestamp"],
    },
    "scalebar": {
        "name": "Scale bar", "type": "bool", "default": DEFAULTS["scalebar"],
    },
}

# fps auto-seed is clamped to a sane playback band (real-time can be absurd).
_FPS_MIN, _FPS_MAX = 5, 60


# ── toolbar parent (opens the wizard) ────────────────────────────────────────────

def export_movie_action(ctx, action_name: str = "Export Movie", **kwargs):
    """Parent toolbar action — a no-op; the Electron toolbar opens the staged
    Movie Export wizard (which drives the ``mvx_*`` handlers) instead. Mirrors
    the Center-Zero-Beam / Vector-Orientation parent pattern."""
    return None


# ── ffmpeg probe ────────────────────────────────────────────────────────────────

def _ffmpeg_ok() -> bool:
    """True when a usable ffmpeg binary is available (imageio-ffmpeg's bundled
    static binary in the normal case)."""
    try:
        import imageio_ffmpeg
        return bool(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as e:
        log.debug("ffmpeg probe failed: %s", e)
        return False


# ── the per-session wizard state ─────────────────────────────────────────────────

class MovieExportState:
    """Backend state for one open Movie Export wizard. Owned by the source tree
    (``tree._mvx_state``). Holds the render params, the captured traces, the
    running flag + cancel hook, and the source tree / plot references."""

    def __init__(self, session, tree, plot, window_id):
        self.session = session
        self.tree = tree
        self.plot = plot
        self.window_id = int(window_id) if window_id is not None else -1
        self.params: dict = {}
        self.traces: list[_traces.TraceSpec] = []
        self.running = False
        self._cancel_flag: list | None = None   # [False] shared with the loop

    # ── time-axis reads (playback conventions) ──────────────────────────────────

    def n_frames(self) -> int:
        try:
            return int(self.tree.root.axes_manager.navigation_shape[0])
        except Exception:
            return 0

    def scale_seconds(self) -> float:
        """Per-frame time step in SECONDS (0.0 when uncalibrated)."""
        try:
            ax = self.tree.root.axes_manager.navigation_axes[0]
            scale = float(getattr(ax, "scale", 0.0) or 0.0)
            if scale <= 0.0:
                return 0.0
            return scale * _units_to_seconds(getattr(ax, "units", None))
        except Exception:
            return 0.0

    def time_units(self) -> str:
        try:
            ax = self.tree.root.axes_manager.navigation_axes[0]
            u = str(getattr(ax, "units", "") or "")
            return "" if u in ("<undefined>", "") else u
        except Exception:
            return ""

    def sig_scale_units(self) -> tuple[float, str]:
        """The signal x-axis scale + units (drives the scale bar; scale<=0 or an
        uncalibrated unit → no scale bar)."""
        try:
            ax = self.tree.root.axes_manager.signal_axes[0]
            scale = float(getattr(ax, "scale", 0.0) or 0.0)
            u = str(getattr(ax, "units", "") or "")
            if u in ("<undefined>", "px", ""):
                return (0.0, "")
            return (scale if scale > 0 else 0.0, u)
        except Exception:
            return (0.0, "")

    def raw(self):
        """The root's raw array (lazy dask or numpy) — the pipeline slices it one
        frame at a time; NEVER computed whole."""
        return self.tree.root.data

    # ── defaults ─────────────────────────────────────────────────────────────────

    def seed_defaults(self) -> None:
        n = self.n_frames()
        scale_s = self.scale_seconds()
        # fps from real-time (like playback) clamped to a sane band.
        if scale_s > 0:
            fps = int(round(1.0 / scale_s))
        else:
            fps = DEFAULTS["fps"]
        fps = max(_FPS_MIN, min(_FPS_MAX, fps))
        cmap, clim = self._plot_cmap_clim()
        self.params = {
            "fps": fps,
            "downsample": DEFAULTS["downsample"],
            "stride": DEFAULTS["stride"],
            "t_start": 0,
            "t_end": max(0, n - 1),
            "cmap": cmap,
            "clim": clim,                                # None = auto
            "timestamp": DEFAULTS["timestamp"],
            "scalebar": bool(self.sig_scale_units()[0] > 0),
            "annotations": [],
        }

    def _plot_cmap_clim(self):
        cmap = DEFAULTS["cmap"]
        clim = None
        plot = self.plot
        try:
            ps = getattr(plot, "plot_state", None)
            if ps is not None and getattr(ps, "colormap", None):
                cmap = str(ps.colormap)
        except Exception:
            pass
        try:
            lv = getattr(plot, "_last_levels", None)
            if lv is not None:
                clim = [float(lv[0]), float(lv[1])]
        except Exception:
            clim = None
        return cmap, clim

    # ── state emission ───────────────────────────────────────────────────────────

    def state(self) -> dict:
        return {
            "type": "mvx_state",
            "window_id": self.window_id,
            "ffmpeg_ok": _ffmpeg_ok(),
            "running": bool(self.running),
            "n_frames": self.n_frames(),
            "time": {"scale_s": self.scale_seconds(), "units": self.time_units()},
            "params": dict(self.params),
            "traces": [t.state() for t in self.traces],
        }

    def emit(self) -> None:
        ipc.emit(self.state())


# ── state resolution ─────────────────────────────────────────────────────────────

def _resolve_tree_plot(session, plot, payload):
    """Resolve (plot, tree) for a wizard action: prefer the clicked plot's tree,
    then the window_id's plot, then the first in-situ tree in the session."""
    src, tree = _src_plot_tree(session, plot)
    if tree is None:
        wid = payload.get("window_id")
        if wid is not None:
            p = session._plot_by_window_id(int(wid)) \
                if hasattr(session, "_plot_by_window_id") else None
            if p is not None:
                src = p
                tree = getattr(p, "signal_tree", None)
    if tree is None:
        for t in getattr(session, "signal_trees", []) or []:
            if _is_insitu(t):
                tree, src = t, (src or None)
                break
    return src, tree


def _is_insitu(tree) -> bool:
    """True when the tree root is an in-situ movie. Absent ``_signal_type`` (bare
    test fakes) is permissive — only an explicit non-insitu type disqualifies
    (mirrors playback's gate)."""
    root = getattr(tree, "root", None) if tree is not None else None
    st = getattr(root, "_signal_type", None)
    if st is None:
        return True
    return st == "insitu"


def _state_for(session, tree) -> "MovieExportState | None":
    return getattr(tree, "_mvx_state", None) if tree is not None else None


# ── handlers ─────────────────────────────────────────────────────────────────────

def mvx_open(session, plot, payload) -> None:
    """Open the Movie Export wizard on an in-situ tree. Refuses a non-insitu
    tree; reports missing ffmpeg (status error + ``ffmpeg_ok:false``)."""
    src, tree = _resolve_tree_plot(session, plot, payload)
    if tree is None:
        ipc.emit_error("Movie Export: no active dataset.")
        return
    if not _is_insitu(tree):
        ipc.emit_error("Movie Export is only available for in-situ movies.")
        return
    # StrictMode double-mount guard: bump the run generation FIRST.
    bump_generation(tree, "_mvx_open_gen")
    st = MovieExportState(session, tree, src, payload.get("window_id"))
    st.seed_defaults()
    tree._mvx_state = st
    if not _ffmpeg_ok():
        ipc.emit_error("Movie Export: ffmpeg not found (imageio-ffmpeg missing). "
                       "Timestamp/annotation preview works; encoding is disabled.")
    st.emit()


def mvx_tune(session, plot, payload) -> None:
    """Debounced live re-tune of the render params (fps / downsample / stride /
    time range / cmap / clim / timestamp / scalebar / annotations)."""
    _src, tree = _resolve_tree_plot(session, plot, payload)
    st = _state_for(session, tree)
    if st is None:
        return
    p = st.params
    for key in ("fps", "downsample", "stride", "t_start", "t_end"):
        if key in payload and payload[key] is not None:
            try:
                p[key] = int(payload[key])
            except Exception:
                pass
    if "cmap" in payload and payload["cmap"]:
        p["cmap"] = str(payload["cmap"])
    if "clim" in payload:
        cl = payload["clim"]
        p["clim"] = ([float(cl[0]), float(cl[1])]
                     if cl and len(cl) == 2 and cl[0] is not None else None)
    for key in ("timestamp", "scalebar"):
        if key in payload:
            p[key] = bool(payload[key])
    if "annotations" in payload:
        p["annotations"] = list(payload["annotations"] or [])
    # Clamp the time range to the dataset.
    n = st.n_frames()
    p["t_start"] = max(0, min(int(p.get("t_start", 0)), max(0, n - 1)))
    p["t_end"] = max(p["t_start"], min(int(p.get("t_end", n - 1)), max(0, n - 1)))
    st.emit()


def mvx_add_trace(session, plot, payload) -> None:
    """Capture a 1-D plot window's signal as a trace (drag a 1-D window pill onto
    the wizard's trace slot). The source must be a 1-D plot."""
    _src, tree = _resolve_tree_plot(session, plot, payload)
    st = _state_for(session, tree)
    if st is None:
        return
    swid = payload.get("source_window_id")
    src_plot = (session._plot_by_window_id(int(swid))
                if swid is not None and hasattr(session, "_plot_by_window_id")
                else None)
    if src_plot is None:
        ipc.emit_error("Add trace: source window not found.")
        return
    color = _traces.color_for_index(len(st.traces))
    spec = _traces.capture_from_plot(src_plot, color=color)
    if spec is None:
        ipc.emit_error("Add trace: source window is not a 1-D plot.")
        return
    st.traces.append(spec)
    st.emit()


def mvx_remove_trace(session, plot, payload) -> None:
    _src, tree = _resolve_tree_plot(session, plot, payload)
    st = _state_for(session, tree)
    if st is None:
        return
    tid = payload.get("trace_id")
    st.traces = [t for t in st.traces if t.id != tid]
    st.emit()


def mvx_run(session, plot, payload) -> None:
    """Render the movie to ``payload['path']``. Refuses while already running;
    runs the pipeline on a worker thread with a generation guard, per-frame cancel
    check, tree cancel-registry + ``mvx_cancel`` hook, progress, and partial-file
    cleanup on cancel/failure. On success emits ``mvx_done``."""
    from spyde.backend.ipc import emit_error, emit_status, emit_progress
    from spyde.actions.movie_export.pipeline import export_movie, _Cancelled

    _src, tree = _resolve_tree_plot(session, plot, payload)
    st = _state_for(session, tree)
    if st is None:
        emit_error("Movie Export: wizard is not open.")
        return
    if st.running:
        emit_error("Movie Export: a render is already in progress.")
        return
    path = payload.get("path")
    if not path:
        emit_error("Movie Export: no output path.")
        return
    if not _ffmpeg_ok() and not str(path).lower().endswith(".gif"):
        emit_error("Movie Export: ffmpeg not available — cannot encode mp4.")
        return

    # Generation guard + cancel wiring.
    gen = bump_generation(tree, "_mvx_run_gen")
    flag = [False]
    st._cancel_flag = flag
    st.running = True
    # Register with the tree's cancel registry so closing the tree stops the render.
    try:
        tree.register_cancel(flag=flag)
    except Exception as e:
        log.debug("mvx register_cancel failed: %s", e)

    raw = st.raw()
    params = dict(st.params)
    n_frames = st.n_frames()
    scale_s = st.scale_seconds()
    sig_scale_x, sig_units = st.sig_scale_units()
    traces = list(st.traces)
    st.emit()   # running=True

    def should_cancel():
        # Stop when THIS run is superseded, the flag is set, or the tree closed.
        return (flag[0] or not is_current(tree, "_mvx_run_gen", gen))

    def progress(done, total):
        try:
            emit_progress(done, total, f"Encoding movie {done}/{total}")
        except Exception:
            pass

    emit_status("Encoding movie…")

    def _work():
        return export_movie(
            raw, path=path, params=params, n_frames=n_frames,
            scale_s=scale_s, sig_scale_x=sig_scale_x, sig_units=sig_units,
            traces=traces, should_cancel=should_cancel, progress=progress,
        )

    def _done(frames):
        _unregister(tree, flag)
        st.running = False
        st._cancel_flag = None
        ipc.emit({"type": "mvx_done", "path": str(path), "frames": int(frames)})
        emit_status(f"Movie exported: {frames} frames.")
        st.emit()

    def _finish_error(exc):
        # State mutation + cleanup + emit — runs on the MAIN thread (marshalled),
        # so it never races the main-thread handlers (mvx_cancel / mvx_close /
        # mvx_run) that touch st.running / st._cancel_flag / the cancel registry.
        _unregister(tree, flag)
        st.running = False
        st._cancel_flag = None
        _cleanup_partial(path)
        if isinstance(exc, _Cancelled):
            emit_status("Movie export cancelled.")
        else:
            emit_error(f"Movie export failed: {exc}")
        st.emit()

    def _on_error(exc):
        # Runs on the WORKER thread — marshal ALL state mutation to the main loop,
        # exactly like _done (which run_on_worker dispatches). Same no-loop inline
        # fallback run_on_worker uses (bare test session → run inline) so tests still
        # observe the emission synchronously.
        disp = getattr(session, "_dispatch_to_main", None)
        (disp(lambda: _finish_error(exc)) if disp is not None
         else _finish_error(exc))

    run_on_worker(session, _work, name="movie-export",
                  on_done=_done, on_error=_on_error)


def mvx_cancel(session, plot, payload) -> None:
    """Request cancellation of an in-flight render."""
    _src, tree = _resolve_tree_plot(session, plot, payload)
    st = _state_for(session, tree)
    if st is None:
        return
    # Bump the generation (supersedes the running loop's is_current check) AND
    # flip the shared flag so the per-frame poll stops promptly.
    bump_generation(tree, "_mvx_run_gen")
    if st._cancel_flag is not None:
        st._cancel_flag[0] = True


def mvx_close(session, plot, payload) -> None:
    """Wizard unmounted: cancel any run and clear the state."""
    _src, tree = _resolve_tree_plot(session, plot, payload)
    st = _state_for(session, tree)
    # Bump FIRST so an in-flight open/run is superseded (StrictMode).
    if tree is not None:
        bump_generation(tree, "_mvx_open_gen")
        bump_generation(tree, "_mvx_run_gen")
    if st is not None and st._cancel_flag is not None:
        st._cancel_flag[0] = True
    if tree is not None:
        tree._mvx_state = None


# ── helpers ──────────────────────────────────────────────────────────────────────

def _unregister(tree, flag) -> None:
    try:
        tree.unregister_cancel(flag=flag)
    except Exception as e:
        log.debug("mvx unregister_cancel failed: %s", e)


def _cleanup_partial(path) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        log.debug("removing partial movie file failed: %s", e)

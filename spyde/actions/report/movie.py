"""
movie.py — the Movie BLOCK backend: an editable, persistent in-situ movie cell in
the report/presentation document, plus its full-screen editor session.

This REPLACES the old per-plot ``mvx`` caret wizard (``spyde/actions/movie_export/
handlers.py``). It reuses the same MEMORY-SAFE render engine —
``movie_export/pipeline.py`` (one lazy frame slice at a time; the full dataset is
NEVER computed), ``encoder.py`` (mp4/gif writer), ``traces.py`` — but the render
state now lives on a report :class:`~spyde.actions.report.model.MovieSpec` (a
``movie`` cell), so it persists in the ``.spyde-report`` zip and can sit inside a
Report OR a Presentation.

Two layers of handler, all ``fn(session, plot, payload)`` (registry.py):

* Document ops (mirror the ``report_*`` cell handlers): ``report_add_movie_cell``
  (empty card OR seeded from a dropped in-situ plot), ``report_set_movie_source``.
  They mutate the :class:`ReportDoc` and re-emit ``report_state``.
* Editor session ops (``movie_*``): ``movie_open`` / ``movie_close`` bind/unbind a
  live :class:`MovieEditSession` (keyed by cell id on the ReportManager).

THE MODEL: a movie IS the source in-situ tree's LIVE 2-D signal figure + its 1-D
time navigator — the SAME data + navigator machinery. The editor does NOT render
its own frames: it surfaces the tree's REAL signal figure (the renderer re-parents
that ``fig_id``'s iframe into the editor area — a fig_id is 1:1 with an iframe, so
the MDI iframe is superseded while the editor is open, then restored on close) and
scrubs by driving the REAL navigator (``movie_scrub`` → the playback primitive
``translate_pixels(t-cur)`` + ``delayed_update_data(force=True)``). So the signal
figure repaints through the real lazy nav pipeline (+ GPU tile mode), and overlays
annotate the live signal plot exactly like a 2-D figure — a movie annotation can be
time-gated. Export (``movie_export``) is a SEPARATE PIL-baked renderer of the same
MovieSpec.

Message contracts (the renderer is written against these EXACT shapes):

* ``movie_state`` — the authoritative editor state (:meth:`MovieEditSession.state`):
    {"type":"movie_state","cell_id":str,"open":bool,"has_source":bool,
     "ffmpeg_ok":bool,"running":bool,"n_frames":int,
     "time":{"scale_s":float,"units":str},"sig":{"scale_x":float,"units":str},
     "source_title":str,"params":{...},"annotations":[...],"text_overlays":[...],
     "freezes":[...],"crop":[x0,y0,x1,y1]|null,"out_size":[w,h]|null,
     "frame_size":[w,h],
     "signal_fig_id":str|null,"signal_window_id":int|null,"nav_fig_id":str|null,
     "current_index":int}
* ``movie_done`` — export success: {"type":"movie_done","cell_id":str,"path":str,
     "frames":int}.
* errors / status / progress via ``emit_error`` / ``emit_status`` / ``emit_progress``.
"""
from __future__ import annotations

import io
import logging
import os

import numpy as np

from spyde.backend import ipc
from spyde.actions.lifecycle import bump_generation, is_current, run_on_worker
from spyde.actions.playback import _units_to_seconds
from spyde.actions.movie_export import traces as _traces
from spyde.actions.movie_export.pipeline import (
    export_movie, render_single_frame, _Cancelled,
)

log = logging.getLogger(__name__)

# Preview frames are downsampled so the base editor preview stays small over the
# stdout JSON channel (a big in-situ frame is 2048² — a full-res base64 PNG per
# scrub tick would swamp the pipe). The editor zoom uses the source detail later;
# for Phase 1 the preview caps its longest edge here (in the ORIGINAL frame's px,
# BEFORE the spec downsample — so the effective preview downsample is the larger
# of the spec's and what this cap needs).
_PREVIEW_MAX_EDGE = 720

# Default render params for a freshly-created movie cell with no live source yet
# (seeded properly from the signal's time axis + the plot's cmap on movie_open).
_DEFAULT_PARAMS = dict(
    fps=12, downsample=1, stride=1, cmap="gray", clim=None,
    timestamp=True, scalebar=True, t_start=0, t_end=0,
)

# fps auto-seed clamp band (real-time can be absurd) — mirrors the old wizard.
_FPS_MIN, _FPS_MAX = 5, 60


def _ffmpeg_ok() -> bool:
    """True when a usable ffmpeg binary is available (imageio-ffmpeg's bundled
    static binary in the normal case)."""
    try:
        import imageio_ffmpeg
        return bool(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as e:
        log.debug("ffmpeg probe failed: %s", e)
        return False


# ── the per-editor session ────────────────────────────────────────────────────

class MovieEditSession:
    """Backend state for one open full-screen Movie editor. Owned by the
    :class:`ReportManager` keyed by cell id (``mgr._movie_sessions[cell_id]``).

    Holds the resolved LIVE source plot/tree (from the cell's
    ``MovieSpec.source`` SignalRef), the running flag + cancel hook, and the base
    text-overlay trace captures. The editable state (params / annotations /
    text_overlays / freezes / crop / out_size) lives on the cell's
    :class:`MovieSpec` — this session reads/writes THROUGH the cell so a tune is
    already persisted (a save just writes the spec).
    """

    def __init__(self, session, mgr, cell, src_plot, tree):
        self.session = session
        self.mgr = mgr
        self.cell = cell
        self.plot = src_plot            # the resolved live source plot (or None)
        self.tree = tree               # the resolved live source tree (or None)
        self.running = False
        self._cancel_flag: list | None = None
        # cell_id-scoped generation owner for the run/open StrictMode guard.
        self._gen_owner = cell
        # The LIVE windows the editor surfaces: the tree's 2-D signal plot (its
        # anyplotlib figure is re-parented into the editor) + its 1-D time
        # navigator selector (the scrubber drives it). Bound in bind_live_windows.
        self.signal_plot = None         # the 2-D signal Plot
        self.time_selector = None       # the 1-D time navigator selector
        self.nav_plot = None            # the 1-D navigator Plot (shown beside, opt)
        # Draggable anyplotlib annotation widgets on the live signal plot, keyed by
        # the annotation's index in cell.movie.annotations. Rebuilt on every spec
        # change so a widget drag → persist → re-emit → resync stays consistent.
        self._ann_widgets: dict = {}    # {ann_index: widget}
        self._widget_handlers: list = []  # keep-alive for the drag handlers
        self._text_overlay_widgets: dict = {}  # {i: (label_widget, overlay_dict)}
        self._overlay_traces: dict = {}   # captured TraceSpec by SignalRef identity
        self._crop_widget = None          # the draggable crop rect (when crop mode on)
        self._overlay_layer = None        # the live 2nd-image overlay layer (or None)

    # ── time / signal axis reads (playback conventions) ──────────────────────────

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

    def frame_size(self) -> tuple[int, int]:
        """The source frame's (w, h) in signal px — the editor's coordinate space
        for crop/annotation placement."""
        try:
            ss = self.tree.root.axes_manager.signal_shape   # (x, y) fast-first
            return (int(ss[0]), int(ss[1]))
        except Exception:
            return (0, 0)

    def raw(self):
        """The root's raw array (lazy dask or numpy) — sliced one frame at a time;
        NEVER computed whole."""
        return self.tree.root.data

    # ── live windows (the tree's real signal figure + time navigator) ────────────

    def bind_live_windows(self) -> None:
        """Find the tree's 2-D signal plot + 1-D time navigator selector so the
        editor can surface the REAL signal figure and scrub the REAL navigator.

        The signal plot is ``tree.signal_plots[0]``; the time selector is the tree's
        1-D navigator selector (reusing playback's discovery). Best-effort: leaves
        the fields None on a tree that hasn't built its windows yet."""
        try:
            plots = list(getattr(self.tree, "signal_plots", []) or [])
            self.signal_plot = plots[0] if plots else self.plot
        except Exception:
            self.signal_plot = self.plot
        # The 1-D time navigator selector — reuse the playback controller's finder,
        # which honours the in-situ gate + the 1-D-line/range widget test.
        try:
            pb = self.session.playback           # lazy-created property
            pb.set_preferred_tree(self.tree)
            sel, _tree = pb._time_selector()
            self.time_selector = sel
            # The navigator plot hosting the selector (shown beside, optional).
            parent = getattr(sel, "parent", None)
            self.nav_plot = getattr(parent, "current_plot_item", None) \
                if parent is not None else None
        except Exception as e:
            log.debug("movie bind time selector failed: %s", e)

    # ── annotation widgets on the live signal figure ─────────────────────────────

    def _signal_plot2d(self):
        """The live anyplotlib Plot2D of the signal figure (None until the iframe
        loads)."""
        return getattr(self.signal_plot, "_plot2d", None) if self.signal_plot else None

    def sync_overlay_widgets(self) -> None:
        """(Re)build the DRAGGABLE anyplotlib annotation widgets on the live signal
        plot from ``cell.movie.annotations`` — text → ``add_label_widget``, rect →
        ``add_rectangle_widget`` (source-pixel coords == image-pixel widget coords
        for a signal frame, so no data-coord conversion). A ``pointer_up`` handler
        persists the moved geometry back to the spec + re-emits state. Non-fatal if
        the figure iframe hasn't loaded (retried on the next scrub / tune)."""
        p2 = self._signal_plot2d()
        if p2 is None:
            return
        # Drop the previous widgets (a rebuild supersedes them wholesale).
        try:
            for w in self._ann_widgets.values():
                p2._widgets.pop(getattr(w, "id", None), None)
        except Exception as e:
            log.debug("movie clear overlay widgets failed: %s", e)
        self._ann_widgets = {}
        self._widget_handlers = []
        anns = (self.cell.movie.annotations if self.cell.movie else None) or []
        for i, ann in enumerate(anns):
            if not isinstance(ann, dict):
                continue
            kind = str(ann.get("kind", ""))
            color = str(ann.get("color", "#ffcc00") or "#ffcc00")
            try:
                if kind == "text":
                    xy = ann.get("xy", [8, 8])
                    w = p2.add_label_widget(
                        x=float(xy[0]), y=float(xy[1]),
                        text=str(ann.get("text", "") or "Label"),
                        fontsize=int(ann.get("size", 18) or 18),
                        color=color, show_handles=False)
                elif kind in ("rect", "circle"):
                    xy = ann.get("xy", [8, 8])
                    wh = ann.get("wh", [40, 40])
                    w = p2.add_rectangle_widget(
                        x=float(xy[0]), y=float(xy[1]),
                        w=float(wh[0]), h=float(wh[1]),
                        color=color, linewidth=float(ann.get("width", 3) or 3),
                        show_handles=True)
                else:
                    continue
            except Exception as e:
                log.debug("movie add overlay widget (%s) failed: %s", kind, e)
                continue
            self._ann_widgets[i] = w
            handler = _make_movie_widget_handler(self, i, kind)
            try:
                w.add_event_handler(handler, "pointer_up")
                self._widget_handlers.append(handler)
            except Exception as e:
                log.debug("movie wire widget drag failed: %s", e)
        # 1-D-signal-as-text overlays: a (non-draggable) label showing the signal's
        # LIVE value at the current frame, e.g. "T = 812.3 °C". Refreshed on scrub.
        self._text_overlay_widgets = {}
        overlays = self.rebuild_text_traces()
        cur = self.current_index()
        scale_s = self.scale_seconds()
        n = self.n_frames()
        for i, ov in enumerate(overlays):
            try:
                xy = ov.get("xy", [8, 8])
                lw = p2.add_label_widget(
                    x=float(xy[0]), y=float(xy[1]),
                    text=_format_overlay_value(ov, cur, scale_s, n),
                    fontsize=int(ov.get("size", 18) or 18),
                    color=str(ov.get("color", "#ffffff") or "#ffffff"),
                    show_handles=False)
                self._text_overlay_widgets[i] = (lw, ov)
            except Exception as e:
                log.debug("movie add text-overlay widget failed: %s", e)
        try:
            p2._push()
        except Exception as e:
            log.debug("movie widget push failed: %s", e)

    def refresh_text_overlays(self) -> None:
        """Update each 1-D-signal-as-text label to the value at the CURRENT frame
        (called after a scrub so the live value tracks the navigator). Cheap: just
        re-sets the widget text + pushes."""
        widgets = getattr(self, "_text_overlay_widgets", None)
        if not widgets:
            return
        p2 = self._signal_plot2d()
        if p2 is None:
            return
        cur = self.current_index()
        scale_s = self.scale_seconds()
        n = self.n_frames()
        changed = False
        for lw, ov in widgets.values():
            try:
                txt = _format_overlay_value(ov, cur, scale_s, n)
                if getattr(lw, "text", None) != txt:
                    lw.text = txt
                    changed = True
            except Exception as e:
                log.debug("movie text-overlay refresh failed: %s", e)
        if changed:
            try:
                p2._push()
            except Exception as e:
                log.debug("movie text-overlay push failed: %s", e)

    def show_crop_widget(self, on: bool) -> None:
        """Toggle a draggable CROP rectangle on the live figure. Seeded to the
        current crop (or the full frame); its ``pointer_up`` persists the rect to
        ``spec.crop`` (source px == widget px). Off → removes the widget."""
        p2 = self._signal_plot2d()
        if p2 is None:
            return
        # Drop any prior crop widget first.
        cw = getattr(self, "_crop_widget", None)
        if cw is not None:
            try:
                p2._widgets.pop(getattr(cw, "id", None), None)
            except Exception:
                pass
            self._crop_widget = None
        if not on:
            try:
                p2._push()
            except Exception:
                pass
            return
        fw, fh = self.frame_size()
        crop = (self.cell.movie.crop if self.cell.movie else None) or [0, 0, fw, fh]
        try:
            w = p2.add_rectangle_widget(
                x=float(crop[0]), y=float(crop[1]),
                w=float(crop[2] - crop[0]), h=float(crop[3] - crop[1]),
                color="#89b4fa", linewidth=3, show_handles=True)
            handler = _make_movie_crop_handler(self)
            w.add_event_handler(handler, "pointer_up")
            self._crop_widget = w
            self._crop_handler = handler
            p2._push()
        except Exception as e:
            log.debug("movie show crop widget failed: %s", e)

    def clear_overlay_widgets(self) -> None:
        p2 = self._signal_plot2d()
        if p2 is not None:
            try:
                for w in self._ann_widgets.values():
                    p2._widgets.pop(getattr(w, "id", None), None)
                for lw, _ov in getattr(self, "_text_overlay_widgets", {}).values():
                    p2._widgets.pop(getattr(lw, "id", None), None)
                cw = getattr(self, "_crop_widget", None)
                if cw is not None:
                    p2._widgets.pop(getattr(cw, "id", None), None)
                ol = getattr(self, "_overlay_layer", None)
                if ol is not None:
                    try:
                        p2.remove_layer(ol)
                    except Exception:
                        pass
                p2._push()
            except Exception as e:
                log.debug("movie clear overlay widgets failed: %s", e)
        self._ann_widgets = {}
        self._widget_handlers = []
        self._text_overlay_widgets = {}
        self._crop_widget = None
        self._overlay_layer = None

    def unbind_live_windows(self) -> None:
        """Restore the surfaced windows to normal (the renderer re-parents the
        signal figure back to its MDI window on movie_close). Clears the annotation
        widgets off the live plot; no backend figure was created, so there's nothing
        else to evict — just drop the references."""
        self.clear_overlay_widgets()
        self.signal_plot = None
        self.time_selector = None
        self.nav_plot = None

    def signal_fig_id(self):
        """The re-parentable ``fig_id`` of the tree's live 2-D signal figure (the
        editor mounts an iframe for this id; a fig_id is 1:1 with an iframe, so the
        MDI window's iframe for the same id is superseded while the editor is open,
        then restored on close)."""
        return getattr(self.signal_plot, "fig_id", None) if self.signal_plot else None

    def signal_window_id(self):
        return getattr(self.signal_plot, "window_id", None) if self.signal_plot else None

    def nav_fig_id(self):
        return getattr(self.nav_plot, "fig_id", None) if self.nav_plot else None

    def current_index(self) -> int:
        """The current time index (from the navigator selector position)."""
        sel = self.time_selector
        if sel is None:
            return 0
        try:
            return int(self.session.playback._current_index(sel))
        except Exception:
            return 0

    def scrub_to(self, t: int) -> None:
        """Drive the REAL navigator to time index *t* — the exact playback
        primitive (relative ``translate_pixels`` + ``delayed_update_data(force)``).
        Pauses playback so the clock isn't fighting the scrub. The live signal
        figure repaints through the real lazy pipeline."""
        sel = self.time_selector
        if sel is None:
            return
        try:
            self.session.playback.pause()
        except Exception as e:
            log.debug("movie scrub pause failed: %s", e)
        try:
            cur = self.current_index()
            sel.translate_pixels(int(t) - int(cur))
            sel.delayed_update_data(force=True)
            # Update any 1-D-signal-as-text overlays to the new frame's value, and
            # the 2nd-image overlay layer to the new frame.
            self.refresh_text_overlays()
            if getattr(self, "_overlay_layer", None) is not None:
                self.sync_overlay_image_layer()
        except Exception as e:
            log.debug("movie scrub_to(%s) failed: %s", t, e)

    @property
    def has_source(self) -> bool:
        return self.tree is not None

    # ── defaults ─────────────────────────────────────────────────────────────────

    def seed_defaults(self) -> None:
        """Seed the cell's MovieSpec params from the live source (fps from the
        real-time scale, cmap/clim from the plot). Only fills MISSING keys — an
        existing (reloaded) spec keeps the user's tuning."""
        spec = self.cell.movie
        if spec is None:
            from spyde.actions.report.model import MovieSpec
            spec = MovieSpec()
            self.cell.movie = spec
        p = dict(_DEFAULT_PARAMS)
        p.update(spec.params or {})       # keep any persisted params
        n = self.n_frames()
        scale_s = self.scale_seconds()
        if not p.get("fps") or "fps" not in (spec.params or {}):
            p["fps"] = (max(_FPS_MIN, min(_FPS_MAX, int(round(1.0 / scale_s))))
                        if scale_s > 0 else _DEFAULT_PARAMS["fps"])
        if "cmap" not in (spec.params or {}):
            p["cmap"] = self._plot_cmap()
        if "clim" not in (spec.params or {}):
            p["clim"] = self._plot_clim()
        if "t_end" not in (spec.params or {}) or not p.get("t_end"):
            p["t_end"] = max(0, n - 1)
        if "scalebar" not in (spec.params or {}):
            p["scalebar"] = bool(self.sig_scale_units()[0] > 0)
        spec.params = p

    def _plot_cmap(self) -> str:
        try:
            ps = getattr(self.plot, "plot_state", None)
            if ps is not None and getattr(ps, "colormap", None):
                return str(ps.colormap)
        except Exception:
            pass
        return _DEFAULT_PARAMS["cmap"]

    def _plot_clim(self):
        try:
            lv = getattr(self.plot, "_last_levels", None)
            if lv is not None:
                return [float(lv[0]), float(lv[1])]
        except Exception:
            pass
        return None

    # ── text-overlay trace captures ──────────────────────────────────────────────

    def rebuild_text_traces(self) -> list:
        """For every 1-D-signal-as-text overlay on the spec, attach a live
        ``_trace`` (a captured :class:`TraceSpec`) so export/preview can resample
        the current value. Prefers a trace CAPTURED AT ADD TIME + cached on the
        session (``_overlay_traces`` keyed by the overlay's SignalRef identity — a
        snapshot survives the source window closing, like a report figure); falls
        back to re-resolving the SignalRef. Returns copies with ``_trace`` attached
        (ephemeral — never serialized); an unresolvable overlay paints label + dash."""
        from spyde.actions.report.model import SignalRef
        spec = self.cell.movie
        cache = getattr(self, "_overlay_traces", None) or {}
        out = []
        for ov in (spec.text_overlays or []):
            ov = dict(ov)
            ref = ov.get("source")
            tr = None
            key = _overlay_key(ref)
            if key is not None and key in cache:
                tr = cache[key]
            elif isinstance(ref, dict):
                try:
                    src = SignalRef.from_dict(ref).resolve(self.session)
                    if src is not None:
                        tr = _traces.capture_from_plot(src)
                        if tr is not None and key is not None:
                            cache[key] = tr           # memoize the snapshot
                except Exception as e:
                    log.debug("text-overlay trace resolve failed: %s", e)
            if tr is not None:
                ov["_trace"] = tr
            out.append(ov)
        self._overlay_traces = cache
        return out

    def resolve_overlay_image(self):
        """Resolve the cell's 2nd-image overlay to a render-ready dict for
        ``export_movie`` (its lazy ``raw`` array + alpha/cmap/clim/blend/time_range),
        or None. Reads the overlay's source SignalRef → live tree → root data
        (sliced one frame at a time in the pipeline). Best-effort; None on failure."""
        from spyde.actions.report.model import SignalRef
        oi = self.cell.movie.overlay_image if self.cell.movie else None
        if not isinstance(oi, dict):
            return None
        ref = oi.get("source")
        if not isinstance(ref, dict):
            return None
        try:
            src = SignalRef.from_dict(ref).resolve(self.session)
            tree = getattr(src, "signal_tree", None) if src is not None else None
            raw_over = tree.root.data if tree is not None else None
            if raw_over is None:
                return None
            return {
                "raw": raw_over,
                "alpha": float(oi.get("alpha", 0.5) or 0.5),
                "cmap": str(oi.get("cmap", "magma") or "magma"),
                "clim": oi.get("clim"),
                "blend": str(oi.get("blend", "over") or "over"),
                "time_range": oi.get("time_range"),
                "downsample": int(oi.get("downsample", self.cell.movie.params.get("downsample", 1) or 1)),
            }
        except Exception as e:
            log.debug("resolve overlay image failed: %s", e)
            return None

    def sync_overlay_image_layer(self) -> None:
        """Composite the 2nd-image overlay as a LIVE layer on the signal figure (a
        translucent ``add_layer`` of its current frame), so the editor shows it too.
        Refreshed on scrub. Best-effort."""
        p2 = self._signal_plot2d()
        if p2 is None:
            return
        # Drop any prior overlay layer.
        prev = getattr(self, "_overlay_layer", None)
        if prev is not None:
            try:
                p2.remove_layer(prev)
            except Exception:
                try:
                    p2._layers = [ly for ly in getattr(p2, "_layers", [])
                                  if ly is not prev]
                    p2._push()
                except Exception:
                    pass
            self._overlay_layer = None
        oi = self.resolve_overlay_image()
        if oi is None:
            return
        try:
            frame = self.read_overlay_frame(oi, self.current_index())
            if frame is None:
                return
            self._overlay_layer = p2.add_layer(
                np.asarray(frame, dtype=np.float32),
                cmap=oi["cmap"], alpha=oi["alpha"], clim=oi.get("clim"))
        except Exception as e:
            log.debug("overlay-image layer add failed: %s", e)

    def read_overlay_frame(self, oi: dict, t: int):
        """Read ONE frame of the overlay image at time *t* (memory-safe slice). A
        3-D overlay is a time stack (index by *t*); a 2-D overlay is a static image
        (returned directly, same on every frame)."""
        raw_over = oi.get("raw")
        if raw_over is None:
            return None
        try:
            if int(getattr(raw_over, "ndim", 2)) >= 3:
                sl = raw_over[min(int(t), int(raw_over.shape[0]) - 1)]
            else:
                sl = raw_over
            comp = getattr(sl, "compute", None)
            if comp is not None:
                try:
                    sl = comp(scheduler="synchronous")
                except TypeError:
                    sl = comp()
            return np.asarray(sl)
        except Exception as e:
            log.debug("overlay frame read failed: %s", e)
            return None

    def cache_overlay_trace(self, ref: dict, trace) -> None:
        """Cache a trace captured at add-time keyed by its SignalRef identity, so
        the live value survives the source window closing."""
        key = _overlay_key(ref)
        if key is None or trace is None:
            return
        cache = getattr(self, "_overlay_traces", None)
        if cache is None:
            cache = {}
            self._overlay_traces = cache
        cache[key] = trace

    def output_info(self) -> dict:
        """The AUTHORITATIVE exported {frames, w, h} — computed with the SAME
        pipeline logic the export uses (freeze-expanded frame count, even-crop, and
        the optional out_size override), so the renderer shows the true output size
        instead of re-deriving it (and drifting on odd edges / freezes)."""
        spec = self.cell.movie
        p = dict(spec.params or {})
        n = self.n_frames()
        try:
            from spyde.actions.movie_export.pipeline import frame_indices_with_freezes
            fps = float(p.get("fps", 12) or 12)
            idxs = frame_indices_with_freezes(
                n, p.get("t_start", 0), p.get("t_end", n - 1),
                p.get("stride", 1), spec.freezes or [], fps)
            frames = len(idxs)
        except Exception:
            frames = 0
        # Output W×H: (crop or full) → /downsample → even-crop (−1 on an odd edge).
        fw, fh = self.frame_size()
        crop = spec.crop
        w = (crop[2] - crop[0]) if crop and len(crop) == 4 else fw
        h = (crop[3] - crop[1]) if crop and len(crop) == 4 else fh
        k = max(1, int(p.get("downsample", 1) or 1))
        w, h = w // k, h // k
        w, h = w - (w % 2), h - (h % 2)         # even-crop for H.264
        if spec.out_size and len(spec.out_size) == 2 and spec.out_size[0] and spec.out_size[1]:
            w = int(spec.out_size[0]) - (int(spec.out_size[0]) % 2)
            h = int(spec.out_size[1]) - (int(spec.out_size[1]) % 2)
        return {"frames": int(frames), "w": int(max(0, w)), "h": int(max(0, h))}

    # ── state emission ───────────────────────────────────────────────────────────

    def state(self) -> dict:
        spec = self.cell.movie
        params = dict(spec.params or {})
        fw, fh = self.frame_size()
        src_title = ""
        if spec.source is not None:
            src_title = str(spec.source.title or spec.source.tree_node or "")
        return {
            "type": "movie_state",
            "cell_id": self.cell.id,
            "open": True,
            "has_source": self.has_source,
            "ffmpeg_ok": _ffmpeg_ok(),
            "running": bool(self.running),
            "n_frames": self.n_frames(),
            "time": {"scale_s": self.scale_seconds(), "units": self.time_units()},
            "sig": {"scale_x": self.sig_scale_units()[0],
                    "units": self.sig_scale_units()[1]},
            "source_title": src_title,
            "params": params,
            "annotations": [dict(a) for a in (spec.annotations or [])],
            "text_overlays": [_public_overlay(o) for o in (spec.text_overlays or [])],
            "freezes": [dict(f) for f in (spec.freezes or [])],
            "crop": (list(spec.crop) if spec.crop else None),
            "out_size": (list(spec.out_size) if spec.out_size else None),
            "frame_size": [fw, fh],
            # The authoritative exported {frames, w, h} (freeze-expanded, even-crop,
            # out_size-aware) so the renderer readout can't drift from the export.
            "output_info": self.output_info(),
            # The live windows the editor surfaces: the tree's REAL 2-D signal
            # figure (re-parented into the editor) + the 1-D navigator (shown
            # beside, optional). A fig_id is 1:1 with an iframe, so mounting it in
            # the editor supersedes the MDI iframe while open (restored on close).
            "signal_fig_id": self.signal_fig_id(),
            "signal_window_id": self.signal_window_id(),
            "nav_fig_id": self.nav_fig_id(),
            "current_index": self.current_index(),
        }

    def emit(self) -> None:
        ipc.emit(self.state())


def _public_overlay(ov: dict) -> dict:
    """A text-overlay dict WITHOUT the ephemeral ``_trace`` (for the wire /
    serialization)."""
    return {k: v for k, v in ov.items() if k != "_trace"}


def _overlay_key(ref):
    """A stable identity for a text-overlay's source SignalRef (for the trace
    cache) — the tree_uid + title + shape, tolerant of a partial dict."""
    if not isinstance(ref, dict):
        return None
    return (ref.get("tree_uid"), ref.get("title"),
            tuple(ref.get("shape") or ()))


def _overlay_value_at(tr, frame: int, scale_s: float, n_frames: int):
    """The 1-D-signal-as-text value at movie *frame*. If the trace has ONE point
    per movie frame (the common per-frame column, e.g. a temperature log), read it
    by FRAME INDEX directly; otherwise resample by physical time (``frame*scale_s``)
    against the trace's own x-axis. ``None`` when unavailable."""
    import numpy as _np
    if tr is None or not hasattr(tr, "resample"):
        return None
    try:
        y = _np.asarray(getattr(tr, "y", None))
        if y is not None and y.size == int(n_frames) and n_frames > 0:
            return float(y[max(0, min(int(frame), int(n_frames) - 1))])
        return float(tr.resample(_np.array([float(frame) * scale_s]))[0])
    except Exception:
        return None


def _format_overlay_value(ov: dict, frame: int, scale_s: float,
                          n_frames: int = 0) -> str:
    """Format a 1-D-signal-as-text overlay's value at *frame* as label text (e.g.
    ``"T = 812.3 °C"``); a dash when no trace is resolved."""
    label = str(ov.get("label", "") or "")
    units = str(ov.get("units", "") or "")
    fmt = str(ov.get("fmt", "") or "")
    val = _overlay_value_at(ov.get("_trace"), frame, scale_s, n_frames)
    if val is None:
        return f"{label} = —" if label else "—"
    if fmt:
        try:
            return fmt.format(label=label, value=val, units=units)
        except (KeyError, IndexError, ValueError):
            pass
    return f"{label} = {val:.2f} {units}".strip()


def _make_movie_crop_handler(st):
    """A ``pointer_up`` handler for the crop rectangle: persists the dragged rect
    (source px) to ``cell.movie.crop`` as ``[x0, y0, x1, y1]`` and re-emits state."""
    def _on_drag_end(event):
        try:
            widget = getattr(event, "source", None)
            g = getattr(widget, "_data", None) if widget is not None else None
            if not isinstance(g, dict) or st.cell.movie is None:
                return
            x0 = int(round(float(g.get("x", 0))))
            y0 = int(round(float(g.get("y", 0))))
            x1 = x0 + int(round(float(g.get("w", 0))))
            y1 = y0 + int(round(float(g.get("h", 0))))
            fw, fh = st.frame_size()
            x0 = max(0, min(x0, fw)); x1 = max(0, min(x1, fw))
            y0 = max(0, min(y0, fh)); y1 = max(0, min(y1, fh))
            if x1 - x0 < 2 or y1 - y0 < 2:
                return
            crop = [x0, y0, x1, y1]
            if st.cell.movie.crop != crop:
                st.cell.movie.crop = crop
                st.mgr.dirty = True
                st.emit()
        except Exception as e:
            log.debug("movie crop persist failed: %s", e)
    return _on_drag_end


def _make_movie_widget_handler(st, ann_index: int, kind: str):
    """A module-level closure (NOT a bound method — anyplotlib sets
    ``fn._event_types`` on the handler) returning a ``pointer_up`` handler that
    persists ONE dragged annotation widget's new geometry back to
    ``cell.movie.annotations[ann_index]``. The widget geometry is in IMAGE pixels,
    which for a signal frame ARE source pixels — so it maps straight to the movie
    annotation's ``xy`` / ``wh`` (no data-coord conversion, unlike a figure panel).
    Re-emits ``movie_state`` so the timeline / renderer mirror stays in sync."""

    def _on_drag_end(event):
        try:
            widget = getattr(event, "source", None)
            g = getattr(widget, "_data", None) if widget is not None else None
            if not isinstance(g, dict):
                return
            cell = st.cell
            anns = (cell.movie.annotations if cell.movie else None) or []
            if not (0 <= ann_index < len(anns)):
                return
            ann = anns[ann_index]
            changed = False
            if kind == "text":
                nx, ny = int(round(float(g.get("x", 0)))), int(round(float(g.get("y", 0))))
                if ann.get("xy") != [nx, ny]:
                    ann["xy"] = [nx, ny]
                    changed = True
            elif kind in ("rect", "circle"):
                nx, ny = int(round(float(g.get("x", 0)))), int(round(float(g.get("y", 0))))
                nw, nh = int(round(float(g.get("w", 0)))), int(round(float(g.get("h", 0))))
                if ann.get("xy") != [nx, ny] or ann.get("wh") != [nw, nh]:
                    ann["xy"] = [nx, ny]
                    ann["wh"] = [nw, nh]
                    changed = True
            if changed:
                st.mgr.dirty = True
                st.emit()
        except Exception as e:
            log.debug("movie widget drag persist failed (idx %s): %s", ann_index, e)

    return _on_drag_end


# ── manager access ─────────────────────────────────────────────────────────────

def _sessions(mgr) -> dict:
    d = getattr(mgr, "_movie_sessions", None)
    if d is None:
        d = {}
        mgr._movie_sessions = d
    return d


def _manager(session):
    from spyde.actions.report.handlers import _manager as _rm
    return _rm(session)


def _ensure_open(session):
    from spyde.actions.report.handlers import _ensure_open as _eo
    return _eo(session)


def _resolve_source_plot(session, source_window_id):
    from spyde.actions.report.handlers import _resolve_source_plot as _rsp
    return _rsp(session, source_window_id)


def _is_insitu_plot(plot) -> bool:
    """True when *plot*'s tree root is an in-situ movie (the movie source gate).
    Permissive on a bare test fake with no ``_signal_type`` (only an explicit
    non-insitu type disqualifies), mirroring the old wizard's gate."""
    tree = getattr(plot, "signal_tree", None) if plot is not None else None
    root = getattr(tree, "root", None) if tree is not None else None
    st = getattr(root, "_signal_type", None)
    return st is None or st == "insitu"


def _first_insitu_plot(session):
    """The first in-situ MOVIE plot in the session — prefer a 2-D signal window
    (the movie frames) over the 1-D time navigator, but either resolves to the same
    tree root, so a navigator is an acceptable fallback. None if no in-situ tree is
    loaded."""
    plots = list(getattr(session, "_plots", []) or [])
    nav = None
    for p in plots:
        tree = getattr(p, "signal_tree", None)
        root = getattr(tree, "root", None) if tree is not None else None
        if root is None or getattr(root, "_signal_type", None) != "insitu":
            continue
        if getattr(p, "is_navigator", False):
            nav = nav or p
        else:
            return p
    return nav


# ── document handlers (mutate the ReportDoc, emit report_state) ─────────────────

def report_add_movie_cell(session, plot, payload) -> None:
    """Add a MOVIE cell to the report.

    Empty (a placeholder drop-zone) when no source is given — the sidebar "Movie"
    card path. Seeded from a live in-situ plot when ``source_window_id`` is given
    (or the active window): builds the ``MovieSpec.source`` SignalRef via
    ``SignalRef.from_plot`` and marks the cell ready.

    ``payload``: ``{source_window_id?, caption?, index?, slide_break?}``. Emits
    ``report_state``.
    """
    from spyde.actions.report.model import Cell, MovieSpec, SignalRef, new_cell_id
    mgr = _ensure_open(session)
    caption = str(payload.get("caption", "") or "")
    cell = Cell(id=new_cell_id(), cell_type="movie", caption=caption,
                movie=MovieSpec(), placeholder=True)
    if payload.get("slide_break") is not None:
        cell.slide_break = bool(payload.get("slide_break"))

    # Seed a source if a window was named (or the active one, for a drop/capture).
    wid = payload.get("source_window_id")
    if wid is not None or payload.get("from_active"):
        src = _resolve_source_plot(session, wid)
        # from_active: if the active window isn't an in-situ movie (it may be the
        # report dock, a navigator, or a different dataset), fall back to the first
        # in-situ MOVIE window in the session — the common "I have a movie open,
        # make a movie block" case shouldn't require focusing the right window.
        if (src is None or not _is_insitu_plot(src)) and payload.get("from_active"):
            src = _first_insitu_plot(session)
        if src is not None and _is_insitu_plot(src):
            cell.movie.source = SignalRef.from_plot(src)
            cell.placeholder = False
        elif src is not None and wid is not None:
            ipc.emit_error("Movie: the dropped window is not an in-situ movie.")
    _insert_cell(mgr.doc, cell, payload.get("index"))
    mgr.dirty = True
    mgr.emit_state()
    # Tell the renderer which cell to open in the full-screen editor (the card path
    # sets open:true so a fresh Movie card jumps straight into the editor).
    if payload.get("open"):
        ipc.emit({"type": "movie_edit_open", "cell_id": cell.id})


def report_set_movie_source(session, plot, payload) -> None:
    """Assign / replace a movie cell's source signal from a dropped in-situ window
    (``{cell_id, source_window_id?}``). Clears the placeholder. Emits
    ``report_state`` and, if an editor session is open for the cell, refreshes it."""
    from spyde.actions.report.model import SignalRef
    mgr = _manager(session)
    if not mgr.open:
        ipc.emit_error("report_set_movie_source: no open report.")
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None or cell.cell_type != "movie":
        ipc.emit_error("report_set_movie_source: not a movie cell.")
        return
    src = _resolve_source_plot(session, payload.get("source_window_id"))
    if src is None:
        ipc.emit_error("report_set_movie_source: source window not found.")
        return
    if not _is_insitu_plot(src):
        ipc.emit_error("Movie: the source window is not an in-situ movie.")
        return
    if cell.movie is None:
        from spyde.actions.report.model import MovieSpec
        cell.movie = MovieSpec()
    cell.movie.source = SignalRef.from_plot(src)
    cell.placeholder = False
    mgr.dirty = True
    mgr.emit_state()
    # If the editor is open for this cell, re-resolve + reseed + re-emit its state.
    st = _sessions(mgr).get(cell.id)
    if st is not None:
        st.plot = src
        st.tree = getattr(src, "signal_tree", None)
        st.seed_defaults()
        st.emit()


# ── editor session handlers ─────────────────────────────────────────────────────

def movie_open(session, plot, payload) -> None:
    """Open the editor for a movie cell (``{cell_id}``).

    A movie IS the source in-situ tree's LIVE 2-D signal figure + its 1-D time
    navigator — the SAME data + navigator machinery. The editor doesn't build a
    second figure: it surfaces the tree's REAL signal figure (the renderer
    re-parents that ``fig_id``'s iframe into the editor area) and scrubs by driving
    the REAL navigator. So overlays annotate the live signal plot exactly like a
    2-D figure, and scrubbing repaints through the real lazy pipeline + tile mode.

    Emits ``movie_state`` carrying the signal figure's ``fig_id`` + the nav
    ``fig_id`` so the renderer can mount both. A movie with no resolvable source
    opens in the pick-a-signal state (has_source:false)."""
    mgr = _manager(session)
    if not mgr.open:
        ipc.emit_error("movie_open: no open report.")
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None or cell.cell_type != "movie":
        ipc.emit_error("movie_open: not a movie cell.")
        return
    if cell.movie is None:
        from spyde.actions.report.model import MovieSpec
        cell.movie = MovieSpec()
    bump_generation(cell, "_movie_open_gen")
    prev = _sessions(mgr).pop(cell.id, None)
    if prev is not None:
        _teardown_session(session, prev)
    src = cell.movie.source.resolve(session) if cell.movie.source else None
    tree = getattr(src, "signal_tree", None) if src is not None else None
    st = MovieEditSession(session, mgr, cell, src, tree)
    if st.has_source:
        st.seed_defaults()
        st.bind_live_windows()
        st.sync_overlay_widgets()      # draggable annotation widgets on the figure
        st.sync_overlay_image_layer()  # the 2nd-image overlay layer (if any)
    _sessions(mgr)[cell.id] = st
    if not _ffmpeg_ok():
        ipc.emit_status("Movie: ffmpeg not found — preview works, mp4 encoding "
                        "is disabled (gif still works).")
    st.emit()


def _teardown_session(session, st) -> None:
    """Cancel any in-flight export and unbind the live windows (restore the signal
    figure to its MDI window)."""
    if st is None:
        return
    if st._cancel_flag is not None:
        st._cancel_flag[0] = True
    try:
        st.unbind_live_windows()
    except Exception as e:
        log.debug("movie unbind live windows failed: %s", e)


def movie_scrub(session, plot, payload) -> None:
    """Scrub the movie to time index ``{cell_id, t}`` by driving the tree's REAL
    1-D time navigator (the same primitive playback uses:
    ``translate_pixels(t - cur)`` + ``delayed_update_data(force=True)``). The live
    signal figure repaints through the real lazy nav pipeline (+ tile mode). Pauses
    playback first so the clock isn't fighting the scrub."""
    mgr = _manager(session)
    if not mgr.open:
        return
    st = _sessions(mgr).get(payload.get("cell_id"))
    if st is None or not st.has_source:
        return
    t = int(payload.get("t", 0))
    n = st.n_frames()
    t = max(0, min(t, max(0, n - 1)))
    st.scrub_to(t)


def movie_close(session, plot, payload) -> None:
    """Close the editor for a cell (``{cell_id}``): cancel any in-flight export,
    unbind the live windows, and drop the session."""
    mgr = _manager(session)
    if not mgr.open:
        return
    cell_id = payload.get("cell_id")
    cell = mgr.doc.cell_by_id(cell_id)
    if cell is not None:
        bump_generation(cell, "_movie_open_gen")
        bump_generation(cell, "_movie_run_gen")
    st = _sessions(mgr).pop(cell_id, None)
    _teardown_session(session, st)


def movie_tune(session, plot, payload) -> None:
    """Wholesale-replace the editable movie state from the editor (debounced live).
    Writes THROUGH to the cell's MovieSpec (so it's already persisted), clamps the
    time range to the dataset, marks the report dirty, and re-emits ``movie_state``.

    ``payload`` may carry any of: ``params`` (a dict merged into spec.params —
    fps/downsample/stride/cmap/clim/timestamp/scalebar/t_start/t_end),
    ``annotations``, ``text_overlays``, ``freezes`` (each a full list replacement),
    ``crop`` ([x0,y0,x1,y1] or null), ``out_size`` ([w,h] or null)."""
    mgr = _manager(session)
    if not mgr.open:
        return
    cell_id = payload.get("cell_id")
    st = _sessions(mgr).get(cell_id)
    cell = mgr.doc.cell_by_id(cell_id)
    if cell is None or cell.cell_type != "movie" or cell.movie is None:
        return
    spec = cell.movie
    if "params" in payload and isinstance(payload["params"], dict):
        p = dict(spec.params or {})
        incoming = payload["params"]
        for key in ("fps", "downsample", "stride", "t_start", "t_end"):
            if key in incoming and incoming[key] is not None:
                try:
                    p[key] = int(incoming[key])
                except (TypeError, ValueError):
                    pass
        if incoming.get("cmap"):
            p["cmap"] = str(incoming["cmap"])
        if "clim" in incoming:
            cl = incoming["clim"]
            p["clim"] = ([float(cl[0]), float(cl[1])]
                         if cl and len(cl) == 2
                         and cl[0] is not None and cl[1] is not None else None)
        for key in ("timestamp", "scalebar"):
            if key in incoming:
                p[key] = bool(incoming[key])
        # Clamp the time range to the dataset (n_frames known only with a source).
        n = st.n_frames() if st is not None and st.has_source else None
        if n:
            p["t_start"] = max(0, min(int(p.get("t_start", 0)), max(0, n - 1)))
            p["t_end"] = max(p["t_start"],
                             min(int(p.get("t_end", n - 1)), max(0, n - 1)))
        spec.params = p
    if "annotations" in payload:
        spec.annotations = list(payload["annotations"] or [])
    if "text_overlays" in payload:
        spec.text_overlays = [_public_overlay(dict(o))
                              for o in (payload["text_overlays"] or [])]
    if "freezes" in payload:
        spec.freezes = list(payload["freezes"] or [])
    if "crop" in payload:
        cr = payload["crop"]
        spec.crop = ([int(v) for v in cr] if cr and len(cr) == 4 else None)
    if "out_size" in payload:
        os_ = payload["out_size"]
        spec.out_size = ([int(v) for v in os_] if os_ and len(os_) == 2 else None)
    mgr.dirty = True
    # An annotation change → rebuild the draggable widgets on the live figure so a
    # newly-added / removed / retimed overlay is immediately editable on the figure.
    if st is not None and "annotations" in payload:
        st.sync_overlay_widgets()
    if st is not None:
        st.emit()
    # Also refresh the sidebar card state (dirty flag / caption may matter).
    mgr.emit_state()


def movie_drop_window(session, plot, payload) -> None:
    """Route a window dropped onto the movie figure by its dimensionality: a 1-D
    plot → a 1-D-signal-as-text overlay; a 2-D image → the 2nd-image overlay. So the
    user just drops any window and the editor does the right thing.
    ``{cell_id, source_window_id, xy?}``."""
    src = _resolve_source_plot(session, payload.get("source_window_id"))
    if src is None:
        ipc.emit_error("Movie drop: source window not found.")
        return
    data = getattr(src, "current_data", None)
    ndim = np.asarray(data).ndim if isinstance(data, np.ndarray) else 0
    if ndim == 1:
        movie_add_text_overlay(session, plot, payload)
    elif ndim == 2:
        movie_add_overlay_image(session, plot, payload)
    else:
        ipc.emit_error("Movie drop: unsupported window (need a 1-D signal or 2-D image).")


def movie_add_overlay_image(session, plot, payload) -> None:
    """Set the 2nd-image overlay from a dropped 2-D (in-situ) window
    (``{cell_id, source_window_id, alpha?, cmap?, blend?}``). Composited over the
    base frame in the live figure + the export. A ``{clear: true}`` payload removes
    it. Emits ``movie_state``."""
    from spyde.actions.report.model import SignalRef
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None or cell.cell_type != "movie" or cell.movie is None:
        return
    st = _sessions(mgr).get(cell.id)
    if payload.get("clear"):
        cell.movie.overlay_image = None
        mgr.dirty = True
        if st is not None:
            st.sync_overlay_image_layer()
            st.emit()
        mgr.emit_state()
        return
    src = _resolve_source_plot(session, payload.get("source_window_id"))
    if src is None:
        ipc.emit_error("Add overlay image: source window not found.")
        return
    frame = getattr(src, "current_data", None)
    if not isinstance(frame, np.ndarray) or frame.ndim != 2:
        ipc.emit_error("Add overlay image: source is not a 2-D image.")
        return
    cell.movie.overlay_image = {
        "source": SignalRef.from_plot(src).to_dict(),
        "alpha": float(payload.get("alpha", 0.5) or 0.5),
        "cmap": str(payload.get("cmap", "magma") or "magma"),
        "blend": str(payload.get("blend", "over") or "over"),
        "time_range": payload.get("time_range"),
    }
    mgr.dirty = True
    if st is not None:
        st.sync_overlay_image_layer()      # show it on the figure
        st.emit()
    mgr.emit_state()


def movie_crop_mode(session, plot, payload) -> None:
    """Toggle the draggable CROP rectangle on the live figure (``{cell_id, on}``).
    Dragging it persists ``spec.crop``; a ``{clear: true}`` payload resets the crop
    to the full frame."""
    mgr = _manager(session)
    if not mgr.open:
        return
    st = _sessions(mgr).get(payload.get("cell_id"))
    if st is None or not st.has_source:
        return
    if payload.get("clear"):
        if st.cell.movie is not None:
            st.cell.movie.crop = None
            mgr.dirty = True
        st.show_crop_widget(False)
        st.emit()
        mgr.emit_state()
        return
    st.show_crop_widget(bool(payload.get("on", True)))


def movie_add_text_overlay(session, plot, payload) -> None:
    """Capture a dropped 1-D plot window as a 1-D-signal-as-text overlay
    (``{cell_id, source_window_id, xy?, label?}``). Records the source SignalRef +
    a label/units snapshot on the spec; the live value is resampled at render
    time. Emits ``movie_state``."""
    from spyde.actions.report.model import SignalRef
    mgr = _manager(session)
    if not mgr.open:
        return
    cell = mgr.doc.cell_by_id(payload.get("cell_id"))
    if cell is None or cell.cell_type != "movie" or cell.movie is None:
        return
    src = _resolve_source_plot(session, payload.get("source_window_id"))
    if src is None:
        ipc.emit_error("Add text overlay: source window not found.")
        return
    tr = _traces.capture_from_plot(src)
    if tr is None:
        ipc.emit_error("Add text overlay: source window is not a 1-D plot.")
        return
    xy = payload.get("xy") or [8, 8 + 26 * len(cell.movie.text_overlays)]
    ov = {
        "source": SignalRef.from_plot(src).to_dict(),
        "label": str(payload.get("label") or tr.label or "value"),
        "units": tr.units,
        "fmt": "",
        "xy": [int(xy[0]), int(xy[1])],
        "size": 18,
        "color": "#ffffff",
    }
    cell.movie.text_overlays = list(cell.movie.text_overlays or []) + [ov]
    mgr.dirty = True
    st = _sessions(mgr).get(cell.id)
    if st is not None:
        st.cache_overlay_trace(ov["source"], tr)   # snapshot survives window close
        st.sync_overlay_widgets()                  # show the live value on the figure
        st.emit()
    mgr.emit_state()


def movie_export(session, plot, payload) -> None:
    """Render the movie to ``{cell_id, path}`` on a worker thread — memory-safe,
    generation-guarded, per-frame cancellable, with partial-file cleanup and a
    poster re-bake on success. Emits progress + ``movie_done``."""
    from spyde.backend.ipc import emit_error, emit_status, emit_progress
    mgr = _manager(session)
    if not mgr.open:
        emit_error("movie_export: no open report.")
        return
    cell_id = payload.get("cell_id")
    st = _sessions(mgr).get(cell_id)
    if st is None or not st.has_source:
        emit_error("movie_export: no source signal.")
        return
    if st.running:
        emit_error("Movie export: a render is already in progress.")
        return
    path = payload.get("path")
    if not path:
        emit_error("Movie export: no output path.")
        return
    if not _ffmpeg_ok() and not str(path).lower().endswith(".gif"):
        emit_error("Movie export: ffmpeg not available — cannot encode mp4 "
                   "(a .gif export still works).")
        return

    cell = st.cell
    gen = bump_generation(cell, "_movie_run_gen")
    flag = [False]
    st._cancel_flag = flag
    st.running = True
    tree = st.tree
    try:
        if tree is not None and hasattr(tree, "register_cancel"):
            tree.register_cancel(flag=flag)
    except Exception as e:
        log.debug("movie register_cancel failed: %s", e)

    raw = st.raw()
    # The pipeline reads crop / annotations / freezes FROM the params dict, but the
    # movie block stores them as separate MovieSpec fields — inject them here.
    params = dict(cell.movie.params or {})
    params["crop"] = cell.movie.crop
    params["annotations"] = list(cell.movie.annotations or [])
    params["freezes"] = list(cell.movie.freezes or [])
    n_frames = st.n_frames()
    scale_s = st.scale_seconds()
    sig_scale_x, sig_units = st.sig_scale_units()
    overlays = st.rebuild_text_traces()
    overlay_img = st.resolve_overlay_image()   # the 2nd-image composite (or None)
    st.emit()   # running=True

    def should_cancel():
        return flag[0] or not is_current(cell, "_movie_run_gen", gen)

    def progress(done, total):
        try:
            emit_progress(done, total, f"Encoding movie {done}/{total}")
        except Exception:
            pass

    emit_status("Encoding movie…")

    def _work():
        frames = export_movie(
            raw, path=path, params=params, n_frames=n_frames, scale_s=scale_s,
            sig_scale_x=sig_scale_x, sig_units=sig_units,
            text_overlays=overlays, overlay=overlay_img,
            should_cancel=should_cancel, progress=progress)
        # Bake a poster from the first rendered frame (memory-safe single frame)
        # so the report card + saved zip show a representative still — WITH the
        # overlays so the thumbnail matches the movie.
        poster = _bake_poster(raw, params, n_frames, scale_s,
                              sig_scale_x, sig_units,
                              text_overlays=overlays, overlay=overlay_img)
        return (frames, poster)

    def _done(result):
        frames, poster = result
        _unregister(tree, flag)
        st.running = False
        st._cancel_flag = None
        if poster is not None:
            mgr._baked[cell.id] = poster
        ipc.emit({"type": "movie_done", "cell_id": cell.id,
                  "path": str(path), "frames": int(frames)})
        emit_status(f"Movie exported: {frames} frames.")
        st.emit()
        mgr.emit_state()

    def _finish_error(exc):
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
        disp = getattr(session, "_dispatch_to_main", None)
        (disp(lambda: _finish_error(exc)) if disp is not None
         else _finish_error(exc))

    run_on_worker(session, _work, name="movie-export",
                  on_done=_done, on_error=_on_error)


def movie_cancel(session, plot, payload) -> None:
    """Request cancellation of an in-flight export (``{cell_id}``)."""
    mgr = _manager(session)
    if not mgr.open:
        return
    cell_id = payload.get("cell_id")
    cell = mgr.doc.cell_by_id(cell_id)
    st = _sessions(mgr).get(cell_id)
    if cell is not None:
        bump_generation(cell, "_movie_run_gen")
    if st is not None and st._cancel_flag is not None:
        st._cancel_flag[0] = True


# ── helpers ──────────────────────────────────────────────────────────────────────

def _insert_cell(doc, cell, index) -> None:
    from spyde.actions.report.handlers import _insert_cell as _ic
    _ic(doc, cell, index)


def _bake_poster(raw, params, n_frames, scale_s, sig_scale_x, sig_units,
                 text_overlays=None, overlay=None):
    """Render the movie's FIRST in-range frame to a capped-size PNG (the card
    poster + saved-zip still) — INCLUDING the annotation / 1-D-text / 2nd-image
    overlays so the thumbnail matches the exported movie. Memory-safe single frame.
    None on failure."""
    try:
        from spyde.actions.movie_export.pipeline import frame_indices
        idxs = frame_indices(n_frames, params.get("t_start", 0),
                             params.get("t_end", n_frames - 1),
                             params.get("stride", 1))
        if not idxs:
            return None
        t0 = idxs[0]
        pp = dict(params)
        # Cap the poster edge like a preview.
        fw = fh = 0
        crop = pp.get("crop")
        if crop and len(crop) == 4:
            fw = abs(int(crop[2]) - int(crop[0]))
            fh = abs(int(crop[3]) - int(crop[1]))
        longest = max(fw, fh, 1)
        need = int(pp.get("downsample", 1) or 1)
        while longest and longest // need > _PREVIEW_MAX_EDGE:
            need += 1
        pp["downsample"] = max(int(pp.get("downsample", 1) or 1), need)
        # The 1-D-text values at the poster frame + a static overlay for compositing.
        tvals = None
        overlays = list(text_overlays or [])
        if overlays:
            from spyde.actions.movie_export.pipeline import _overlay_value_at
            tvals = [_overlay_value_at(o.get("_trace"), t0, scale_s, n_frames)
                     for o in overlays]
        img = render_single_frame(raw, t0, params=pp, n_frames=n_frames,
                                  scale_s=scale_s, sig_scale_x=sig_scale_x,
                                  sig_units=sig_units, text_overlays=overlays,
                                  text_values=tvals, overlay=overlay)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        log.debug("movie poster bake failed: %s", e)
        return None


def _unregister(tree, flag) -> None:
    try:
        if tree is not None and hasattr(tree, "unregister_cancel"):
            tree.unregister_cancel(flag=flag)
    except Exception as e:
        log.debug("movie unregister_cancel failed: %s", e)


def _cleanup_partial(path) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        log.debug("removing partial movie file failed: %s", e)

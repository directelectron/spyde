"""
plot.py — anyplotlib-backed Plot object.

Replaces the old pg.PlotItem subclass with a plain Python class that owns an
anyplotlib Figure (registered with the Electron bridge for iframe rendering).

A Plot is either 2D (imshow) or 1D (line) based on the signal dimensionality
of its active PlotState.  Switching PlotStates may change the display type.
"""
from __future__ import annotations

import time
from math import floor, log10
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

import anyplotlib as apl
import anyplotlib._electron as _electron
from anyplotlib.embed import build_standalone_html

from spyde.drawing.colormaps import COLORMAPS, DEFAULT_COLORMAP

if TYPE_CHECKING:
    from spyde.drawing.plots.plot_states import PlotState
    from spyde.drawing.plots.plot_window import PlotWindow
    from spyde.drawing.plots.multiplot_manager import MultiplotManager
    from spyde.drawing.selectors import BaseSelector
    from spyde.signal_tree import BaseSignalTree
    from spyde.backend.session import Session

import logging
logger = logging.getLogger(__name__)

import threading as _threading


class _NavPainter:
    """A single serial daemon thread that PAINTS navigator frames off the
    `_NavDispatcher` thread, newest-wins per plot.

    Why: the nav READ must stay serial on the dispatcher (hyperspy cache safety),
    but PAINTING a frame is a `set_data` → binary-uint8 → stdout write that can take
    ~8–70 ms (transport of a large frame, or a cold chunk decode landing). Doing that
    inline on the dispatcher BLOCKS reading the next slider position, so the slider
    "catches" instead of tracking the cursor while the display lags behind. Moving
    the paint here lets the dispatcher immediately read the LATEST position; a stale
    frame queued behind a newer one is DROPPED (newest-wins) before it paints, so the
    display converges on the cursor's final position.

    This is NOT the retired read self-pacing/buffer-ring (those coalesced the READ
    and made it worse). This is a newest-wins single-slot PAINT decouple — exactly
    the "display lags, slider stays live" behaviour. stdout PLOTBIN writes stay
    serialized because there is ONE painter thread. Latest-wins by `id(plot)`."""

    def __init__(self) -> None:
        self._lock = _threading.Lock()
        self._pending: "dict[int, tuple]" = {}   # id(plot) -> (plot, ndarray)
        self._wake = _threading.Event()
        self._thread = _threading.Thread(
            target=self._run, name="nav-paint", daemon=True)
        self._thread.start()

    def submit(self, plot, data) -> None:
        """Queue (or replace) this plot's pending paint. Newest wins."""
        with self._lock:
            self._pending[id(plot)] = (plot, data)
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                jobs = list(self._pending.values())
                self._pending.clear()
                self._wake.clear()
            for plot, data in jobs:
                try:
                    plot.current_data = data
                    plot._set_array(data)
                except Exception as e:
                    logger.debug("nav paint failed: %s", e)


# One painter for the whole process — the single serial lane nav frames paint on.
_nav_painter = _NavPainter()

import time as _time
# Per-frame PAINT profile (the transport/render half of a navigator update): logs
# one line per _set_array with the paint stages — contrast levels and transport
# (anyplotlib set_data → base64/binary → stdout emit, usually the dominant cost for a
# large frame). Pairs with the [NAV-PROFILE] read line from
# update_functions so a "slow update" report shows the WHOLE pipeline (read +
# paint). Toggle live from the Log panel's "Profile" button; state lives in
# backend.debug_flags (read fresh each frame). No-op when off.
from spyde.backend.debug_flags import nav_profile_on as _nav_profile_on


_SUPERSCRIPTS = {
    "^{-1}": "⁻¹", "^-1": "⁻¹",
    "^{-2}": "⁻²", "^-2": "⁻²",
    "^{2}": "²", "^2": "²", "^{3}": "³", "^3": "³",
}


def _clean_units(u) -> str:
    """Turn a HyperSpy/LaTeX-ish units string into clean unicode for display.

    e.g. ``$A^{-1}$`` or pyxem's ``$\\AA^{-1}$`` → ``Å⁻¹`` (raw LaTeX showed
    literally in anyplotlib axis labels / scale bar). Strips ``$`` math
    delimiters and braces, expands the LaTeX ångström macro (``\\AA`` /
    ``\\angstrom`` → ``Å``), maps common superscripts, and uses the
    reciprocal-ångström convention (A⁻¹ → Å⁻¹).
    """
    s = str(u).strip().replace("$", "")
    # LaTeX ångström macro(s) → the Å glyph, BEFORE the superscript/`A⁻¹` maps so
    # pyxem's `$\AA^{-1}$` resolves to Å⁻¹ (not a literal backslash on the plot).
    s = s.replace("\\AA", "Å").replace("\\angstrom", "Å").replace("\\text{Å}", "Å")
    for k, v in _SUPERSCRIPTS.items():
        s = s.replace(k, v)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("A⁻¹", "Å⁻¹")   # bare A⁻¹ → Å⁻¹
    return s


_SHARED_ESM_PATH: str | None = None


def _shared_esm_url(esm: str) -> str:
    """Write the anyplotlib JS bundle to a single shared file once and return its
    file:// URL.

    Every figure HTML otherwise inlines the full ~368 KB bundle in a *blob* URL
    that is unique per figure — so Chromium's V8 code cache never reuses the
    compiled bytecode and every iframe re-parses the whole bundle (the "big
    drag"). A stable shared URL lets the code cache kick in across iframes and
    shrinks each figure's HTML from ~370 KB to a few KB.
    """
    global _SHARED_ESM_PATH
    import hashlib
    import os
    import tempfile

    if _SHARED_ESM_PATH and os.path.exists(_SHARED_ESM_PATH):
        return "file://" + _SHARED_ESM_PATH
    digest = hashlib.sha1(esm.encode("utf-8")).hexdigest()[:12]
    path = os.path.join(tempfile.gettempdir(), f"spyde_figure_esm_{digest}.js")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(esm)
    _SHARED_ESM_PATH = path
    return "file://" + path


def finalize_figure_html(fig, fig_id) -> str:
    """Build the standalone HTML for an anyplotlib figure and apply SpyDE's shared
    post-processing: swap the inlined JS bundle for the shared file URL (V8 code
    cache), force dark mode (`#widget-root`), and add the click-to-front focus
    relay. Shared by :meth:`Plot._ensure_figure` and the IPF 3-D figure builder."""
    import json as _json

    html = build_standalone_html(fig, fig_id=fig_id, resizable=False)
    try:
        esm = str(getattr(fig, "_esm", "") or "")
        if esm:
            embedded = _json.dumps(esm)
            shared = _shared_esm_url(esm)
            html = html.replace(
                f"const esmSource = {embedded};", "const esmSource = null;", 1)
            html = html.replace(
                "import(blobUrl)", f"import({_json.dumps(shared)})", 1)
    except Exception as e:
        logger.debug("shared-esm optimization skipped: %s", e)

    # The standalone template pins html/body to the figure's INITIAL px size with
    # overflow:hidden (sized for a fixed docs/notebook embed). SpyDE drives the
    # size live via resize_figure (fig_width/height → the panels re-layout), so
    # when the subwindow is dragged LARGER than that initial size the grown figure
    # was clipped to the old body box. Make html/body/#widget-root fill the iframe
    # (100%) so the figure always fills — and is never clipped by — the subwindow.
    dark = ("<style>html,body{background:#1e1e2e !important;color-scheme:dark;"
            "width:100% !important;height:100% !important;overflow:hidden}"
            "#widget-root{background:#1e1e2e !important;"
            "width:100% !important;height:100% !important;display:block !important}"
            "</style>")
    focus = ("<script>window.addEventListener('pointerdown',function(){"
             "try{window.parent.postMessage({type:'spyde_focus',figId:%r},'*');}"
             "catch(e){}},true);</script>" % str(fig_id))
    return html.replace("<body>", dark + focus + "<body>", 1)


class Plot:
    """
    A single plot panel backed by an anyplotlib Figure.

    Each Plot corresponds to one subplot (Axes) in a Figure.  For a
    PlotWindow with multiple plots the Figures are separate iframes that
    Electron lays out side-by-side.
    """

    def __init__(
        self,
        signal_tree: "BaseSignalTree",
        is_navigator: bool = False,
        multiplot_manager: "MultiplotManager | None" = None,
        plot_window: "PlotWindow | None" = None,
        session: "Session | None" = None,
    ):
        self.is_navigator = is_navigator
        self.signal_tree = signal_tree
        self.multiplot_manager = multiplot_manager
        self.plot_window = plot_window
        self.session = session or (signal_tree.session if signal_tree else None)

        # Display state
        self.needs_update_range: bool | None = None
        self.needs_auto_level: bool = True
        # Last contrast levels applied. Held across navigator-driven frames so the
        # diffraction pattern's brightness doesn't rescale (flash) on every move —
        # contrast only re-computes on an explicit auto-level (new data / request).
        self._last_levels: tuple[float, float] | None = None
        self.current_data: np.ndarray | object | None = None

        # PlotState management
        self.plot_state: "PlotState | None" = None
        self.plot_states: Dict = {}

        # Shared memory for the navigator→DP fast path (allocated lazily). ONE
        # buffer per plot: the navigator update submits a write_shared_array future
        # that writes the latest frame here, and the poll worker reads it locally
        # when the future completes (bypassing the slow TCP transfer of the frame).
        # Overlapping writes during a fast drag are fine — only the LATEST future's
        # result is applied (the staleness guard in _on_plot_ready), so a frame
        # clobbered by a newer write was going to be dropped anyway.
        self._shared_memory: SharedMemory | None = None
        # write_future.key -> get_inds Future, held alive until the write lands so
        # client-side GC doesn't release the get_inds dep and cancel the chain
        # (see update_from_navigation_selection).
        self._inflight_getinds: Dict = {}
        # The single in-flight EXPENSIVE-tier navigator read for this plot (a
        # cancellable submit_graph future). A newer nav position cancels it before
        # submitting its own, so a slow region / derived-view read never blocks the
        # serial dispatcher and a superseded frame is dropped (Live-Display §3
        # tiered read). None when the current read is cheap/synchronous.
        self._nav_future = None
        # Per-plot LRU of decoded output nav-chunks for the synchronous read — makes
        # dwelling within a chunk on a derived (rebin/crop/.zspy) view a ~0 ms numpy
        # slice instead of re-decoding the whole source chunk every move. Cleared
        # when the displayed node changes / on close. (Import here to avoid a
        # module-load cycle: update_functions imports drawing types.)
        from spyde.drawing.update_functions import _NavChunkCache
        self._nav_chunk_cache = _NavChunkCache()
        # GPU tile backend for a LARGE signal frame — reduces overview/detail tiles on
        # the GPU. Lazily built on the first large frame (_maybe_gpu_tile); reset on a
        # node switch / close so a new signal rebuilds it. None = not tiling yet.
        self._gpu_tile_backend = None

        # anyplotlib figure + plot objects
        self._fig: apl.Figure | None = None
        self._axes: apl.Axes | None = None
        self._plot2d: apl.Plot2D | None = None
        self._plot1d: apl.Plot1D | None = None
        self.fig_id: str | None = None

        # Scale bar state
        self._scale_bar_enabled = False

        # Unified view-bar tagging: when this plot is one of a window's named
        # views (e.g. an IPF-Z map or a strain εxx map) the figure carries a
        # chip label + representation kind so the frontend can build the
        # chip-strip selector. None = an ordinary (untagged) figure.
        self.view_label: str | None = None
        self.view_kind: str = "2d"

        # Window ID (set when registered with MDIManager)
        self.window_id: int | None = (
            getattr(plot_window, "window_id", None)
        )

        # Register with session
        if self.session is not None:
            self.session.register_plot(self)

    # ── anyplotlib figure ──────────────────────────────────────────────────────

    def _ensure_figure(self, dims: int = 2) -> None:
        """Create (or recreate) the anyplotlib Figure for the given dimensionality."""
        from spyde.backend.ipc import emit

        if self._fig is not None:
            return  # already created

        self._fig, axes_obj = apl.subplots(1, 1)
        self._axes = axes_obj[0][0] if isinstance(axes_obj, list) else axes_obj

        if dims == 2:
            # gpu="auto": large scalar images render on the GPU (WebGPU texture +
            # shader LUT + zoom/upsample); small images, RGB, and no-GPU machines
            # fall back to Canvas2D transparently. The GPU handles downscale/zoom,
            # so we can send FULL-res frames (no LOD decimation) when it's active.
            # SPYDE_GPU_IMAGE=0 forces the Canvas2D path — the e2e GPU-parity spec
            # uses it to render the CPU reference of the same scene.
            import os as _os
            _gpu_mode = ("off" if _os.environ.get("SPYDE_GPU_IMAGE", "")
                         .lower() in ("0", "off", "false") else "auto")
            self._plot2d = self._axes.imshow(
                np.zeros((10, 10), dtype=np.float32),
                cmap=DEFAULT_COLORMAP,
                gpu=_gpu_mode,
            )
            # Viewport LOD is owned by anyplotlib's tile mode (set_data(tile="auto")):
            # it emits view_changed to ITSELF and runs the overview + detail-tile loop
            # internally. SpyDE no longer registers a view handler here.
        else:
            self._plot1d = self._axes.plot(np.zeros(10))

        # Register with Electron bridge so it can dispatch events back
        self.fig_id = _electron.register(self._fig)

        # Build standalone HTML, embedding the SAME fig_id so the iframe stamps
        # its outgoing awi_event messages with an id that matches the registry —
        # otherwise selector-drag events can't be routed back to this figure and
        # the signal plot never updates.
        #
        # resizable=False hides anyplotlib's own corner resize triangle; sizing
        # is driven by the SubWindow (react-rnd) resize → resizeFigure IPC.
        html = finalize_figure_html(self._fig, self.fig_id)
        title = ""
        if self.signal_tree is not None:
            try:
                title = self.signal_tree.root.metadata.get_item(
                    "General.title", default=""
                )
            except Exception as e:
                logger.debug("reading plot title from metadata failed: %s", e)

        # Navigator: report the real-space image aspect (width/height) so the
        # frontend sizes the window to it. Without this, a wide scan (e.g. sped_ag
        # 208×64) gets aspect-letterboxed into a strip and the crosshair/axes no
        # longer line up with the image ("compressed / selector off").
        nav_aspect = None
        if self.is_navigator and self.signal_tree is not None:
            try:
                nav_shape = self.signal_tree.root.axes_manager.navigation_shape
                if len(nav_shape) >= 2 and nav_shape[0] > 0 and nav_shape[1] > 0:
                    nav_aspect = float(nav_shape[0]) / float(nav_shape[1])
            except Exception:
                nav_aspect = None

        self._figure_html = html
        self._figure_aspect = nav_aspect
        emit({
            "type": "figure",
            "fig_id": self.fig_id,
            "window_id": self.window_id,
            "html": html,
            "title": title or ("Navigator" if self.is_navigator else "Signal"),
            "is_navigator": self.is_navigator,
            "aspect": nav_aspect,
            "view_label": self.view_label,
            "view_kind": self.view_kind if self.view_label else None,
        })

    def set_view_tag(self, label: str, kind: str = "2d") -> None:
        """Tag this plot's figure as a named view (chip ``label`` + ``kind``) and
        RE-EMIT the figure message so the frontend's chip strip picks it up. The
        iframe is keyed by fig_id, so the metadata updates without a reload."""
        from spyde.backend.ipc import emit
        self.view_label = label
        self.view_kind = kind
        if self.fig_id is None or getattr(self, "_figure_html", None) is None:
            return
        emit({
            "type": "figure", "fig_id": self.fig_id, "window_id": self.window_id,
            "html": self._figure_html, "title": label, "is_navigator": False,
            "aspect": getattr(self, "_figure_aspect", None),
            "view_label": label, "view_kind": kind,
        })

    # A LARGE signal frame (either edge ≥ this) is tiled by anyplotlib, but SpyDE
    # supplies its OWN GpuTileBackend so the overview/detail reduction runs on the GPU
    # (the CPU area-mean of a 4k frame is ~62 ms — the movie-scrub wall). Matches
    # anyplotlib's own TILE_THRESHOLD so the gate agrees with what set_data would tile.
    _GPU_TILE_MIN_EDGE = 1024

    def _maybe_gpu_tile(self, data, clim, axes, units) -> bool:
        """Route a LARGE signal frame through anyplotlib tile mode backed by a GPU
        reducer. First large frame → enable_tile(GpuTileBackend); every subsequent one
        → swap the backend source + update_tile_source (keeps zoom/detail). Returns
        True if it took the tile path, False for a small frame (caller falls through to
        the plain set_data)."""
        if data.ndim != 2 or max(data.shape[:2]) < self._GPU_TILE_MIN_EDGE:
            return False
        p2 = self._plot2d
        if p2 is None or not hasattr(p2, "enable_tile"):
            return False
        try:
            from spyde.drawing.plots.gpu_tile_backend import GpuTileBackend
            _t0 = _time.perf_counter()
            arr = np.ascontiguousarray(data)
            vmin, vmax = clim
            if getattr(self, "_gpu_tile_backend", None) is None:
                # First large frame: build the GPU backend + enable tiling. Set the
                # contrast BEFORE enable so the overview quantises over it, then apply
                # the calibrated axes/extent (one push each; every later frame reuses
                # the same axes so no re-push).
                self._gpu_tile_backend = GpuTileBackend(arr, origin=p2._origin)
                _t1 = _time.perf_counter()
                try:
                    p2.set_clim(float(vmin), float(vmax))
                except Exception:
                    pass
                _t2 = _time.perf_counter()
                p2.enable_tile(self._gpu_tile_backend, integration_method="mean")
                logger.debug("[TILEDBG] tile timings: backend_ctor=%.1fms "
                             "set_clim=%.1fms enable=%.1fms",
                             (_t1 - _t0) * 1e3, (_t2 - _t1) * 1e3,
                             (_time.perf_counter() - _t2) * 1e3)
                if axes is not None:
                    try:
                        # set_extent carries the units so the tiled/GPU image
                        # draws physical tick gutters + the scale bar (set_extent
                        # sets has_axes=True). Without units here a calibrated
                        # signal shows ticks but no scale bar.
                        p2.set_extent(axes[0], axes[1], units=units)
                        self._last_extent_key = (
                            float(axes[0][0]), float(axes[0][-1]),
                            float(axes[1][0]), float(axes[1][-1]),
                            int(data.shape[0]), int(data.shape[1]))
                    except Exception as e:
                        logger.debug("gpu-tile set_extent failed: %s", e)
                logger.info("[TILEDBG] GPU tile ENABLED win=%s frame=%s cuda=%s",
                            self.window_id, arr.shape,
                            self._gpu_tile_backend._torch is not None)
            else:
                # Subsequent frame (movie scrub): swap the GPU source + re-sample the
                # overview/detail in place, keeping the zoom/subselection. Contrast is
                # windowed in the LUT (see anyplotlib set_clim tile path) — a display-
                # only move, so we don't re-quantise here.
                self._gpu_tile_backend.set_array(arr)
                if (vmin, vmax) != (self._plot2d._state.get("display_min"),
                                    self._plot2d._state.get("display_max")):
                    self._plot2d._state["display_min"] = float(vmin)
                    self._plot2d._state["display_max"] = float(vmax)
                # Re-apply the calibrated axes ONLY when they actually changed
                # (an Axes-editor edit of scale/offset/units) — the common movie
                # scrub keeps the same extent and skips this so there's one push
                # per frame. Without this, a scale/units edit on a tiled plot
                # updates the model but the ticks/scale bar never move.
                if axes is not None:
                    ext_key = (float(axes[0][0]), float(axes[0][-1]),
                               float(axes[1][0]), float(axes[1][-1]),
                               int(data.shape[0]), int(data.shape[1]))
                    if (ext_key != getattr(self, "_last_extent_key", None)
                            or units != self._plot2d._state.get("units")):
                        try:
                            p2.set_extent(axes[0], axes[1], units=units)
                            self._last_extent_key = ext_key
                        except Exception as e:
                            logger.debug("gpu-tile re-extent failed: %s", e)
                p2.update_tile_source()
            return True
        except Exception as e:
            logger.info("[TILEDBG] GPU tile path FAILED, falling back: %s", e)
            self._gpu_tile_backend = None
            return False

    # ── Public display interface ───────────────────────────────────────────────

    def update(self) -> None:
        """Push current_data to the anyplotlib figure."""
        data = self.current_data
        if data is None:
            return
        if isinstance(data, np.ndarray):
            self._set_array(data)
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug("[REDRAW] update() NO-OP: current_data is %s, not ndarray "
                         "(win=%s) — nothing painted", type(data).__name__,
                         self.window_id)

    def update_data(self, data_or_future) -> None:
        """Set current_data (may be ndarray or dask Future)."""
        self.current_data = data_or_future
        if isinstance(data_or_future, np.ndarray):
            self.update()

    def enqueue_paint(self, data: np.ndarray) -> None:
        """Paint an ndarray frame OFF the caller's thread via the serial newest-wins
        painter (see _NavPainter). Used by the nav dispatcher so a slow paint doesn't
        block reading the next slider position. current_data is set now (so a
        concurrent read sees the latest intent); the painter sets it again + paints.
        Newest-wins: a frame superseded before it paints is dropped."""
        self.current_data = data
        _nav_painter.submit(self, data)

    def set_data(self, data: np.ndarray, levels=None) -> None:
        """Directly push new array data (called from progressive compute poll)."""
        self.current_data = data
        self._set_array(data, levels=levels)

    def set_transform_active(self, active: bool) -> None:
        """Enter/leave Find-Vectors transform view. While active, the navigator's
        raw-frame paint to this plot is suppressed (the overlay drives the image
        via :meth:`set_transform_image`)."""
        self._fv_transform_active = bool(active)
        logger.debug("[plot] transform-view lock %s (window %s)",
                     "ACTIVE" if active else "released", self.window_id)

    def set_transform_image(self, data: np.ndarray, levels=None) -> None:
        """Paint the detector transform image, bypassing the transform-view lock
        (this is the one paint allowed while the lock is on)."""
        self._fv_paint_token = True
        try:
            self.current_data = data
            self._set_array(data, levels=levels)
        finally:
            self._fv_paint_token = False

    def set_overlay_mask(self, mask: "np.ndarray | None",
                         color: str = "#ff4444", alpha: float = 0.4) -> None:
        """Draw (or clear) a translucent boolean mask over the displayed image.

        Used to show the detected beam-stop region during Find-Vectors. The mask
        is composited client-side in the anyplotlib iframe — no recompute, no
        new image push. Pass ``mask=None`` to clear it.

        The mask must match the displayed image's (H, W); if it doesn't (e.g. a
        stale beam-stop from a different signal), the overlay is cleared rather
        than raising.
        """
        if self._plot2d is None:
            return
        try:
            if mask is None:
                self._plot2d.set_overlay_mask(None, color=color, alpha=alpha)
                logger.debug("[plot] overlay mask cleared (window %s)",
                             self.window_id)
                return
            arr = np.asarray(mask)
            h = self._plot2d._state.get("image_height")
            w = self._plot2d._state.get("image_width")
            if h and w and arr.shape != (h, w):
                logger.debug(
                    "[plot] overlay mask shape %s != image %sx%s — clearing",
                    arr.shape, h, w)
                self._plot2d.set_overlay_mask(None, color=color, alpha=alpha)
                return
            self._plot2d.set_overlay_mask(arr, color=color, alpha=alpha)
            logger.debug("[plot] overlay mask set: %d px (window %s)",
                         int(np.count_nonzero(arr)), self.window_id)
        except Exception as e:
            logger.debug("[plot] set_overlay_mask failed: %s", e)

    def _display_axes(self):
        """The axis objects that calibrate THIS plot's displayed image.

        A NAVIGATOR plot shows the navigation space, so its display axes are the
        ROOT signal's *navigation* axes (the objects the Axes editor's set_axis
        mutates) — NOT its current_signal's signal axes, which are a decoupled
        copy on the derived `root.sum(signal_axes)` navigator signal that never
        tracks later edits. Reading the root nav axes here is what makes a
        nav-axis rename/rescale reach the navigator panel on the set_axis
        re-push. A SIGNAL plot uses its current_signal's signal axes as before
        (those ARE the mutated root/node axis objects). Mirrors the is_navigator
        branch already in enable_scale_bar / _set_offset_crosshair."""
        if self.is_navigator:
            return list(self.signal_tree.root.axes_manager.navigation_axes)
        return list(self.plot_state.current_signal.axes_manager.signal_axes)

    def _axes_info(self, data: np.ndarray):
        """Return (axes, units) for the displayed 2-D image from this plot's
        display axes, so anyplotlib draws calibrated ticks + a scale bar.

        The scale bar auto-renders whenever units are physical (not 'px') and a
        scale is present — matching the Qt scale-bar overlay.
        """
        try:
            axes = self._display_axes()
            if len(axes) < 2 or data.ndim != 2:
                return None, "px"
            x_ax, y_ax = axes[0], axes[1]
            units = str(x_ax.units)
            if units in ("<undefined>", "px", ""):
                return None, "px"
            xa = np.asarray(x_ax.axis)
            ya = np.asarray(y_ax.axis)
            return [xa, ya], _clean_units(units)
        except Exception:
            return None, "px"

    def _axes_info_1d(self, data: np.ndarray):
        """Return (x_axis, x_units, y_label) for a displayed 1-D signal so the
        anyplotlib line plot draws a calibrated x-axis + an x/y-axis label.

        x-axis: the current signal's single signal axis (`.axis` gives the
        expanded coordinate array; `.units`/`.name` label it). y-label: the
        signal's `metadata.Signal.quantity` when set (hyperspy's convention),
        else a generic "Intensity". Returns (None, "", y_label) when there's no
        1-D signal so the caller falls back to an uncalibrated push.
        """
        y_label = "Intensity"
        try:
            sig = self.plot_state.current_signal
            # Action-declared y-label wins (e.g. a line profile's "Intensity"),
            # then the signal's own quantity metadata, then the default.
            act_label = getattr(self, "_y_label_override", None)
            if act_label:
                y_label = str(act_label)
            else:
                try:
                    q = sig.metadata.get_item("Signal.quantity", default="")
                    if q:
                        y_label = _clean_units(q)
                except Exception:
                    pass
            # A 1-D NAVIGATOR (e.g. a movie's time axis) is calibrated by the
            # ROOT's navigation axis, not the derived nav signal's copy — see
            # _display_axes — so a nav-axis edit reaches the line plot.
            axes = self._display_axes()
            if len(axes) < 1 or data.ndim != 1:
                return None, "", y_label
            x_ax = axes[0]
            # hyperspy uses a `<undefined>` sentinel object (str() == "<undefined>"
            # but not == the string) for unset units/name — stringify before the
            # "is it meaningful?" check so the sentinel is treated as empty.
            units = str(x_ax.units)
            x_units = ("" if units in ("<undefined>", "px", "")
                       else _clean_units(units))
            # Prefix the axis name when present ("Energy (eV)" style) so the
            # label reads meaningfully; fall back to bare units.
            name = str(getattr(x_ax, "name", "") or "")
            if name and name not in ("<undefined>", ""):
                x_units = (f"{name} ({x_units})" if x_units else name)
            xa = np.asarray(x_ax.axis, dtype=float)
            if xa.shape[0] != data.shape[0]:
                # Axis/data length mismatch (e.g. a placeholder frame) — don't
                # push a bad x-axis; keep the default 0..N-1.
                return None, x_units, y_label
            return xa, x_units, y_label
        except Exception:
            return None, "", y_label

    def _plot_title(self) -> str:
        """The dataset name to show as the panel title (drawn in anyplotlib's
        always-reserved title strip). Reads the signal's General.title; empty
        string means no title (the strip stays blank)."""
        try:
            sig = self.plot_state.current_signal
            t = sig.metadata.get_item("General.title", default="")
            return str(t or "")
        except Exception:
            return ""

    def _apply_plot_title(self) -> None:
        """Push the current signal's title into the anyplotlib panel so the name
        renders inside the plot (not only on the Electron window chrome). Called
        on figure creation / node switch — cheap, and safe to repeat."""
        p2 = getattr(self, "_plot2d", None)
        p1 = getattr(self, "_plot1d", None)
        target = p2 if p2 is not None else p1
        if target is None or not hasattr(target, "set_title"):
            return
        try:
            title = self._plot_title()
            if title != (target._state.get("title") or ""):
                target.set_title(title)
        except Exception as e:
            logger.debug("applying plot title failed: %s", e)

    def _is_navigated_frame(self) -> bool:
        """True when this plot shows a frame sliced from a NAVIGATED dataset (a
        diffraction pattern under the navigator), where adjacent frames share a
        contrast range — so the contrast should be held across navigation. False
        for output plots (virtual image / FFT / line profile), whose every
        recompute is a brand-new image that must re-auto-level."""
        try:
            if self.is_navigator:
                return False
            sig = self.plot_state.current_signal
            return sig.axes_manager.navigation_dimension > 0
        except Exception:
            return False

    def _set_array(self, data: np.ndarray, levels=None) -> None:
        # Paint-side per-frame profile — read the live toggle once per frame.
        _NAV_PROFILE = _nav_profile_on()
        _p0 = _time.perf_counter() if _NAV_PROFILE else 0.0
        _pt = {} if _NAV_PROFILE else None       # stage -> seconds
        _in_shape = getattr(data, "shape", None)
        # Never try to paint a Future or a Future-bearing object array (a lazy
        # navigator's data before its progressive compute lands) — anyplotlib's
        # set_data does np.asarray(data, dtype=float), which raises on a Future.
        if not isinstance(data, np.ndarray) or data.dtype == object:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[plot] _set_array skipped non-numeric data %s (win=%s)",
                             type(data).__name__ if not isinstance(data, np.ndarray)
                             else f"object-ndarray{data.shape}", self.window_id)
            return
        dims = data.ndim

        # DIAGNOSTIC: how much REAL detail is in the array handed to us? shape says
        # 4096² but if a small 82² center crop has only ~36 distinct values the ARRAY
        # is blocky (a low-res frame stored at 4096, not true 4k). Logs the crop's
        # distinct-value count so we can tell the INITIAL frame from a SCRUBBED one —
        # if only scrubbed frames are blocky, the nav read is the source, not tiling.
        if (dims == 2 and not self.is_navigator and min(data.shape) >= 200
                and logger.isEnabledFor(logging.DEBUG)):
            try:
                cy, cx = data.shape[0] // 2, data.shape[1] // 2
                crop = data[cy - 41:cy + 41, cx - 41:cx + 41]
                logger.debug(
                    "[TILEDBG] _set_array IN win=%s shape=%s dtype=%s "
                    "center82_distinct=%d full_distinct(sampled)=%d min=%.4g "
                    "max=%.4g navigator=%s",
                    self.window_id, tuple(data.shape), data.dtype,
                    int(np.unique(crop).size),
                    int(np.unique(data[::16, ::16]).size),
                    float(data.min()), float(data.max()), self.is_navigator)
            except Exception as _e:
                logger.debug("tiledbg in-frame probe failed: %s", _e)

        # Transform-view lock: while the Find-Vectors preview shows the DoG /
        # correlation image on this signal plot, the navigator's RAW-frame paint
        # must not overwrite it. The overlay sets `_fv_transform_active` and
        # paints through `set_transform_image` (which sets `_fv_paint_token` to
        # bypass this guard); every other paint of a navigated frame is dropped
        # so the navigator move recomputes the transform (via the overlay's
        # index hook) instead of flashing the raw pattern back.
        if (getattr(self, "_fv_transform_active", False)
                and not getattr(self, "_fv_paint_token", False)
                and dims == 2 and self._is_navigated_frame()):
            logger.debug("[REDRAW] _set_array DROPPED (transform-view lock) win=%s "
                         "shape=%s", self.window_id, tuple(data.shape))
            return

        # Hold the previous frame instead of flashing a degenerate one.
        #
        # On a cross-chunk navigator move over the distributed cluster, the new
        # diffraction pattern arrives via the shared-memory buffer; a torn/early
        # read of that reused buffer (or a transient loading placeholder) can be
        # all-zeros, which paints as a black flash. For a NAVIGATED DP we already
        # hold contrast across frames, so an all-zero frame is never a real
        # pattern — drop it and keep the last good image until a valid frame
        # lands (the latest-future guard then paints the correct one). Guarded to
        # navigated frames only: a genuinely empty output (blank VI/FFT) must
        # still paint.
        try:
            if (dims == 2 and self._last_levels is not None
                    and self._is_navigated_frame() and data.size):
                # all-zeros torn/early shm read, OR the int8 "loading" checkerboard
                # placeholder (values 0/1) — neither is a real pattern.
                if not np.any(data) or (
                        data.dtype == np.int8 and float(data.max()) <= 1):
                    logger.debug("[REDRAW] _set_array DROPPED (degenerate zeros/"
                                 "checkerboard) win=%s dtype=%s", self.window_id,
                                 data.dtype)
                    return
        except Exception as e:
            logger.debug("zero-frame hold check failed: %s", e)

        # RGB(A) image (H, W, 3|4) — e.g. an IPF orientation map. anyplotlib
        # renders it directly; there are no scalar levels / histogram.
        if dims == 3 and data.shape[-1] in (3, 4):
            self._ensure_figure(2)
            if self._plot2d is not None:
                self._plot2d.set_data(data)
            return

        self._ensure_figure(dims)

        # Keep the dataset name in the panel's title strip. Cheap (a compare;
        # only pushes when the title actually changed) and covers the case where
        # the name was stamped AFTER the first set_plot_state (dynamic signal
        # plots paint their first real frame later than the node switch).
        self._apply_plot_title()

        # NO SpyDE-side level-of-detail / decimation. anyplotlib's tile mode owns ALL
        # downscaling for a large frame: set_data(tile="auto") sends a downsampled
        # overview as the base and streams a hi-res detail tile of the visible region
        # on zoom/pan (native pixels, no full-frame transfer). We just hand it the full
        # frame every time — no stride, no thumbnail copy, no native/display split.

        _dbg = logger.isEnabledFor(logging.DEBUG)
        if _dbg:
            logger.debug("[TILEDBG] _set_array stage: pre-checks done (win=%s)",
                         self.window_id)
        if dims == 2 and self._plot2d is not None:
            _lv = _time.perf_counter() if _NAV_PROFILE else 0.0
            if self.needs_auto_level:
                # New data / explicit request → recompute contrast + histogram.
                if (self._is_navigated_frame() and self._last_levels is not None
                        and logger.isEnabledFor(logging.DEBUG)):
                    logger.debug("[plot-paint] SIG RE-AUTO-LEVEL on navigated frame "
                                 "(was %s) — this is a contrast flash", self._last_levels)
                vmin, vmax = self._robust_levels(data, signal=not self.is_navigator)
                self.needs_auto_level = False
                self._emit_histogram(data, vmin, vmax)
            elif levels is not None:
                vmin, vmax = levels
            elif self._last_levels is not None and self._is_navigated_frame():
                # NAVIGATOR move on a 4D dataset (adjacent frames share a range):
                # HOLD the contrast so the DP doesn't flash brighter/darker as you
                # drag. Output plots (virtual image / FFT / line profile) are NOT
                # navigated — each recompute is a brand-new image whose range can
                # differ a lot, so they re-auto-level instead (holding made the VI
                # look wrong as the detector ROI moved).
                vmin, vmax = self._last_levels
            else:
                vmin, vmax = self._robust_levels(data, signal=not self.is_navigator)
                self._emit_histogram(data, vmin, vmax)
            self._last_levels = (vmin, vmax)
            if _NAV_PROFILE:
                _pt["levels"] = _time.perf_counter() - _lv
            if _dbg:
                logger.debug("[TILEDBG] _set_array stage: levels+histogram done "
                             "(%.1fms, win=%s)",
                             (_time.perf_counter() - _lv) * 1e3, self.window_id)

            # Hand the FULL frame to anyplotlib. A signal image uses tile="auto"
            # (anyplotlib tiles it above its threshold: overview base + on-zoom detail
            # tile, all downscaling on its side). A NAVIGATOR image must NOT tile — its
            # 2-D selector maps clicks by displayed-pixel coords, so tile=False keeps a
            # 1:1 full-frame image. The clim rides the SAME push (atomic, no contrast
            # flash from a second set_clim).
            tile_mode = False if self.is_navigator else "auto"
            axes, units = self._axes_info(data)
            clim = (vmin, vmax)
            _st = _time.perf_counter() if _NAV_PROFILE else 0.0
            # GPU tile backend: for a LARGE signal frame, do the overview/detail
            # reduction on the GPU instead of the CPU (the ~62 ms numpy area-mean of a
            # 4k frame is the movie-scrub wall). SpyDE injects its own TileBackend via
            # enable_tile so anyplotlib keeps no torch dep. If the frame isn't big
            # enough to tile (or it's a navigator), fall through to the plain set_data.
            if (tile_mode == "auto"
                    and self._maybe_gpu_tile(data, clim, axes, units)):
                if _NAV_PROFILE:
                    _pt["transport"] = _time.perf_counter() - _st
                    stages = "  ".join(f"{k}={v*1e3:.1f}" for k, v in _pt.items())
                    logger.info(
                        "[PAINT-PROFILE] SIG-TILE total=%.1fms  %s  in=%s",
                        (_time.perf_counter() - _p0) * 1e3, stages, _in_shape)
                return
            if axes is not None:
                self._plot2d.set_data(data, x_axis=axes[0], y_axis=axes[1],
                                      units=units, clim=clim, tile=tile_mode)
                # set_data updates x_axis/units but does NOT recompute scale_x/
                # scale_y from the new axes — so the auto scale bar would size off
                # the stale init scale. set_extent recomputes it, BUT it pushes
                # again; skip it when the extent is unchanged (every navigated
                # frame shares the same axes) so we keep ONE push per frame.
                ext_key = (float(axes[0][0]), float(axes[0][-1]),
                           float(axes[1][0]), float(axes[1][-1]), int(data.shape[0]),
                           int(data.shape[1]))
                if ext_key != getattr(self, "_last_extent_key", None):
                    self._last_extent_key = ext_key
                    try:
                        self._plot2d.set_extent(axes[0], axes[1])
                    except Exception as e:
                        logger.debug("set_extent failed: %s", e)
            else:
                self._plot2d.set_data(data, clim=clim, tile=tile_mode)
            if _NAV_PROFILE:
                _pt["transport"] = _time.perf_counter() - _st
                out_shape = tuple(data.shape)
                stages = "  ".join(f"{k}={v*1e3:.1f}" for k, v in _pt.items())
                # INFO so it reaches stderr / the Log panel for a "normal usage"
                # report. total = whole _set_array; transport = the anyplotlib
                # set_data(+extent) base64+stdout push (usually the biggest stage).
                logger.info(
                    "[PAINT-PROFILE] %s total=%.1fms  %s  in=%s out=%s nav=%s",
                    "NAV" if self.is_navigator else "SIG",
                    (_time.perf_counter() - _p0) * 1e3, stages,
                    _in_shape, out_shape, self.is_navigator)
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    a = np.asarray(data, dtype=np.float64)
                    # Content fingerprint: if this CHANGES per frame, distinct
                    # images ARE being pushed to set_data (so a "frozen" display is
                    # downstream — the iframe/render). If it REPEATS while the
                    # crosshair moves, the SAME frame is being re-applied (the data
                    # path is delivering stale frames).
                    _h = int(a.tobytes().__hash__() & 0xFFFFFF) if a.size else 0
                    logger.debug(
                        "[plot-paint] %s shape=%s hash=%06x mean=%.3g std=%.3g "
                        "max=%.3g navigated=%s",
                        "NAV" if self.is_navigator else "SIG", tuple(a.shape),
                        _h, a.mean(), a.std(), a.max(), self._is_navigated_frame())
                except Exception:
                    pass
        elif dims == 1 and self._plot1d is not None:
            # Calibrate the line plot from the signal's own axis: pass the x
            # coordinate array + x-axis label (units/name) and a y-axis label so
            # the panel shows real ticks and labels — and re-reads them every
            # paint, so an Axes-editor edit (scale/offset/units) propagates via
            # the existing p.update() re-push. Falls back to a bare push when
            # there's no calibrated 1-D signal.
            xa, x_units, y_label = self._axes_info_1d(data)
            if xa is not None:
                self._plot1d.set_data(data, x_axis=xa, units=x_units,
                                      y_units=y_label)
            else:
                self._plot1d.set_data(data, units=x_units, y_units=y_label)

    def _emit_histogram(self, data: np.ndarray, vmin: float, vmax: float,
                        threshold: float = None) -> None:
        """Send a histogram of the current image to the sidebar for this window.

        Only emitted on auto-level (new data / contrast), not on every selector
        drag, so it doesn't flood the channel. ``threshold`` (when given) draws a
        dotted marker line on the histogram (the Find-Vectors detector threshold).
        """
        if self.window_id is None:
            return
        try:
            # Subsample a large frame the same way _robust_levels does (≤ ~512²
            # samples): a 64-bin histogram from 262k samples is display-identical
            # to the full-frame one, while a full pass over a 16 M-px movie frame
            # is memory-bandwidth-bound — measured 11.7 s (!) for this block while
            # the navigator fill's worker processes saturated the machine (the
            # signal panel sat black exactly that long). Integer frames skip the
            # isfinite mask entirely (nothing to mask; it copies 16 MB for fun).
            sub = data
            if sub.ndim == 2 and max(sub.shape) > 512:
                sub = sub[::max(1, sub.shape[0] // 512),
                          ::max(1, sub.shape[1] // 512)]
            if np.issubdtype(sub.dtype, np.integer):
                finite = sub.ravel()
            else:
                finite = sub[np.isfinite(sub)]
            if finite.size == 0:
                return
            counts, edges = np.histogram(finite, bins=64)
            from spyde.backend.ipc import emit
            msg = {
                "type": "histogram",
                "window_id": self.window_id,
                "counts": counts.astype(int).tolist(),
                "edges": [float(e) for e in edges],
                "vmin": float(vmin),
                "vmax": float(vmax),
            }
            # threshold marker: None clears it (raw view), a value draws the line
            msg["threshold"] = None if threshold is None else float(threshold)
            emit(msg)
            logger.debug("[plot] histogram emit window=%s vmin=%.3g vmax=%.3g "
                         "threshold=%s n=%d", self.window_id, vmin, vmax,
                         msg["threshold"], int(finite.size))
        except Exception as e:
            logger.debug("histogram emit failed: %s", e)

    @staticmethod
    def _robust_levels(img: np.ndarray, signal: bool = False) -> tuple:
        """Robust display range. For a *signal* (diffraction) plot the central
        beam is orders of magnitude brighter than the diffraction spots, so a
        99.5% upper clip still saturates everything but the beam. Use a tighter
        2–99% band there so the faint spots are visible; navigator/VI images keep
        the wide min–99.5% range (they have no saturating central spike)."""
        try:
            sy = max(1, img.shape[0] // 512)
            sx = max(1, img.shape[1] // 512) if img.ndim > 1 else 1
            data = np.asarray(img[::sy, ::sx] if img.ndim > 1 else img[::sy],
                              dtype=np.float64)
            data = data[np.isfinite(data)]
            if data.size == 0:
                return 0.0, 1.0
            if signal:
                mn = float(np.percentile(data, 2.0))
                mx = float(np.percentile(data, 99.0))
            else:
                mn = float(data.min())
                mx = float(np.percentile(data, 99.5))
            if mx <= mn:
                mx = float(data.max())
            if mx <= mn:
                mx = mn + 1.0
            return mn, mx
        except Exception:
            return 0.0, 1.0

    def set_colormap(self, name: str) -> None:
        resolved = COLORMAPS.get(name, name)
        if self._plot2d is not None:
            self._plot2d.set_colormap(resolved)
        if self.plot_state is not None:
            self.plot_state.colormap = name

    def set_clim(self, vmin: float | None, vmax: float | None) -> None:
        if self._plot2d is not None and vmin is not None and vmax is not None:
            self._plot2d.set_clim(float(vmin), float(vmax))
            # Remember the user's contrast so navigator frames keep it (instead of
            # snapping back to the auto levels held from the first frame).
            self._last_levels = (float(vmin), float(vmax))

    def set_gamma(self, gamma: float) -> None:
        if self.plot_state is not None:
            self.plot_state.gamma = float(gamma)
        # anyplotlib doesn't have gamma directly; approximate via display range warp
        # (full gamma support to be added in Phase 4 if needed)

    # ── Scale bar ──────────────────────────────────────────────────────────────

    def enable_scale_bar(self, enabled: bool = True) -> None:
        """Toggle scale bar overlay (visual only; no Qt objects)."""
        self._scale_bar_enabled = enabled
        if not enabled or self._plot2d is None:
            return
        try:
            if self.is_navigator:
                axes = self.signal_tree.root.axes_manager.navigation_axes
            else:
                axes = self.plot_state.current_signal.axes_manager.signal_axes
            if not axes:
                return
            x_range = axes[0].scale * axes[0].size
            target = x_range / 5
            if not np.isfinite(target) or target <= 0:
                target = 1.0
            exp = floor(log10(target))
            base = 10 ** exp
            norm = target / base
            nice = (1.0 if norm < 1.5 else 2.0 if norm < 2.5 else
                    2.5 if norm < 3.5 else 5.0 if norm < 7.5 else 10.0)
            nice_length = nice * base
            units = axes[0].units or ""
            # anyplotlib has no scale-bar primitive yet; surface the length via
            # the x-axis label as a lightweight stand-in.
            if hasattr(self._plot2d, "set_scale_bar"):
                self._plot2d.set_scale_bar(nice_length, units)
        except Exception as e:
            logger.debug("enable_scale_bar: %s", e)

    # ── PlotState management ───────────────────────────────────────────────────

    def add_plot_state(
        self,
        signal,
        dimensions: int = 2,
        dynamic: bool = True,
    ) -> "PlotState":
        from spyde.drawing.plots.plot_states import PlotState

        state = PlotState(
            signal=signal,
            plot=self,
            dimensions=dimensions,
            dynamic=dynamic,
        )
        self.plot_states[signal] = state
        # Activate the first state so the figure (iframe) is created and emitted
        # immediately — every plot renders as soon as it has a state, regardless
        # of which code path (nav / signal / child / FFT) created it.
        if self.plot_state is None:
            self.set_plot_state(signal)
        return state

    def set_plot_state(self, signal) -> None:
        old_state = self.plot_state
        self.needs_auto_level = True
        # The displayed node is changing — drop cached decoded chunks of the old
        # node so they don't occupy the budget (keys are per-signal-id, so they'd
        # never be returned for the new node anyway).
        if self._nav_chunk_cache is not None:
            self._nav_chunk_cache.clear()
        # Rebuild the GPU tile backend for the new node (its logical size / dtype may
        # differ; a stale backend would swap a mismatched frame into the old tiling).
        self._gpu_tile_backend = None

        if old_state is not None:
            old_state.hide_toolbars()
            for sel in (old_state.plot_selectors + old_state.signal_tree_selectors):
                sel.widget.hide()

        new_state = self.plot_states.get(signal)
        if new_state is None:
            return

        self.plot_state = new_state
        dims = new_state.dimensions
        self._ensure_figure(dims)
        new_state.show_toolbars()
        # Show the dataset name inside the panel (title strip), now that the
        # active signal is known. Re-applied on every node switch.
        self._apply_plot_title()

        # Static plots (e.g. the navigator image) display the signal's own data
        # directly. Dynamic plots (driven by a selector) are filled by the
        # selector instead. For lazy signals the data is a Future here and is
        # pushed later by the progressive compute, so only push real arrays.
        if not new_state.dynamic:
            data = getattr(new_state.current_signal, "data", None)
            # Require a REAL numeric array. A lazy navigator's data can be a dask
            # Future (distributed nav compute) — or, subtly, a length-1 OBJECT
            # ndarray *wrapping* a Future (shape (1,), ndim 1, dtype=object). Both
            # must be skipped here (the progressive nav compute paints the result
            # later); painting them sends a Future into anyplotlib's
            # np.asarray(data, dtype=float) → "float() argument ... not 'Future'".
            if (isinstance(data, np.ndarray) and data.ndim == dims
                    and data.dtype != object):
                self.current_data = data
                self.needs_auto_level = True
                self.update()

    # ── Shared memory ──────────────────────────────────────────────────────────

    def _buffer_nbytes(self) -> int:
        """Size the per-plot shared buffer to the displayed signal FRAME, not a
        max-possible 8K×8K image. The old fixed 256 MB buffer (8192²×4) was
        allocated for every lazy plot (nav, signal, every virtual image / FFT) —
        spiking RAM into the GBs and making subwindow spawn slow (zeroing 256 MB).
        A 256×256 diffraction pattern needs ~0.5 MB.
        """
        try:
            sig = self.plot_state.current_signal
            shape = tuple(sig.axes_manager.signal_shape)
            n = int(np.prod(shape)) if shape else 0
            if n > 0:
                # 8 bytes/elem (float64 worst case) + header/margin; 1 MB floor.
                return max(n * 8 + 8192, 1 << 20)
        except Exception as e:
            logger.debug("frame-size probe failed, using 4 MB shm fallback: %s", e)
        return 4 << 20  # 4 MB fallback when the frame size isn't known yet

    @property
    def shared_memory(self) -> SharedMemory:
        if self._shared_memory is None:
            self._shared_memory = SharedMemory(
                name=f"plot_buffer{id(self)}",
                create=True,
                size=self._buffer_nbytes(),
            )
        # The frame can grow (signal-type / shape change) after the buffer was
        # first sized — grow it so write_shared_array doesn't overflow.
        nbytes = self._buffer_nbytes()
        if self._shared_memory.size < nbytes:
            try:
                self._shared_memory.close(); self._shared_memory.unlink()
            except Exception as e:
                logger.debug("growing plot shm buffer failed to free old: %s", e)
            self._shared_memory = SharedMemory(
                name=f"plot_buffer{id(self)}", create=True, size=nbytes,
            )
        return self._shared_memory

    # ── Toolbar / selector helpers ────────────────────────────────────────────

    @property
    def toolbars(self) -> list:
        if self.plot_state is None:
            return []
        return [
            self.plot_state.toolbar_top,
            self.plot_state.toolbar_bottom,
            self.plot_state.toolbar_left,
            self.plot_state.toolbar_right,
        ]

    @property
    def parent_selector(self) -> "BaseSelector | None":
        if self.plot_window is None:
            return None
        return getattr(self.plot_window, "parent_selector", None)

    @property
    def main_window(self):
        """Compatibility shim — returns session for code that still uses main_window."""
        return self.session

    # ── Axis helpers (compatibility with old code) ────────────────────────────

    def addItem(self, item) -> None:
        """No-op compatibility — items are now anyplotlib widgets, not pg items."""
        pass

    def removeItem(self, item) -> None:
        pass

    def normalize_axes(self) -> None:
        pass

    def update_range(self) -> None:
        pass

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        # Stop any in-flight progressive (chunked) compute streaming into us.
        try:
            from spyde.drawing.update_functions import _stop_progressive_stream
            _stop_progressive_stream(self)
        except Exception as e:
            logger.debug("stopping progressive stream on plot close failed: %s", e)
        if self.session is not None:
            self.session.unregister_plot(self)
        # Drop any held get_inds futures (let the scheduler release them).
        try:
            self._inflight_getinds.clear()
        except Exception:
            pass
        # Cancel any in-flight expensive-tier navigator read.
        nf = self._nav_future
        if nf is not None:
            try:
                nf.cancel()
            except Exception:
                pass
            self._nav_future = None
        # Release cached decoded chunks.
        if self._nav_chunk_cache is not None:
            try:
                self._nav_chunk_cache.clear()
            except Exception:
                pass
        if self._shared_memory is not None:
            try:
                self._shared_memory.close()
                self._shared_memory.unlink()
            except Exception as e:
                logger.debug("releasing plot shared memory on close failed: %s", e)
            self._shared_memory = None
        if self.fig_id is not None:
            try:
                del _electron._figures[self.fig_id]
            except Exception as e:
                logger.debug("dropping figure registry entry on close failed: %s", e)

    def hide(self) -> None:
        from spyde.backend.ipc import emit
        emit({"type": "window_visibility", "window_id": self.window_id, "visible": False})

    def show(self) -> None:
        from spyde.backend.ipc import emit
        emit({"type": "window_visibility", "window_id": self.window_id, "visible": True})

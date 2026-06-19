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


_SUPERSCRIPTS = {
    "^{-1}": "⁻¹", "^-1": "⁻¹",
    "^{-2}": "⁻²", "^-2": "⁻²",
    "^{2}": "²", "^2": "²", "^{3}": "³", "^3": "³",
}


def _clean_units(u) -> str:
    """Turn a HyperSpy/LaTeX-ish units string into clean unicode for display.

    e.g. ``$A^{-1}$`` → ``Å⁻¹`` (raw LaTeX showed literally in anyplotlib axis
    labels / scale bar). Strips ``$`` math delimiters and braces, maps common
    superscripts, and uses the reciprocal-ångström convention (A⁻¹ → Å⁻¹).
    """
    s = str(u).strip().replace("$", "")
    for k, v in _SUPERSCRIPTS.items():
        s = s.replace(k, v)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("A⁻¹", "Å⁻¹")   # A⁻¹ → Å⁻¹
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

    dark = ("<style>html,body{background:#1e1e2e !important;color-scheme:dark}"
            "#widget-root{background:#1e1e2e !important}</style>")
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

        # Shared memory for large arrays (allocated lazily)
        self._shared_memory: SharedMemory | None = None
        self._pending_shm_future = None

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
            self._plot2d = self._axes.imshow(
                np.zeros((10, 10), dtype=np.float32),
                cmap=DEFAULT_COLORMAP,
            )
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
            except Exception:
                pass

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

    # ── Public display interface ───────────────────────────────────────────────

    def update(self) -> None:
        """Push current_data to the anyplotlib figure."""
        data = self.current_data
        if data is None:
            return
        if isinstance(data, np.ndarray):
            self._set_array(data)

    def update_data(self, data_or_future) -> None:
        """Set current_data (may be ndarray or dask Future)."""
        self.current_data = data_or_future
        if isinstance(data_or_future, np.ndarray):
            self.update()

    def set_data(self, data: np.ndarray, levels=None) -> None:
        """Directly push new array data (called from progressive compute poll)."""
        self.current_data = data
        self._set_array(data, levels=levels)

    def _axes_info(self, data: np.ndarray):
        """Return (axes, units) for the displayed 2-D image from the current
        signal's signal axes, so anyplotlib draws a calibrated scale bar.

        The scale bar auto-renders whenever units are physical (not 'px') and a
        scale is present — matching the Qt scale-bar overlay.
        """
        try:
            sig = self.plot_state.current_signal
            sig_axes = sig.axes_manager.signal_axes
            if len(sig_axes) < 2 or data.ndim != 2:
                return None, "px"
            x_ax, y_ax = sig_axes[0], sig_axes[1]
            units = x_ax.units
            if not units or units in ("<undefined>", "px", ""):
                return None, "px"
            return [np.asarray(x_ax.axis), np.asarray(y_ax.axis)], _clean_units(units)
        except Exception:
            return None, "px"

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
        dims = data.ndim

        # RGB(A) image (H, W, 3|4) — e.g. an IPF orientation map. anyplotlib
        # renders it directly; there are no scalar levels / histogram.
        if dims == 3 and data.shape[-1] in (3, 4):
            self._ensure_figure(2)
            if self._plot2d is not None:
                self._plot2d.set_data(data)
            return

        self._ensure_figure(dims)

        if dims == 2 and self._plot2d is not None:
            if self.needs_auto_level:
                # New data / explicit request → recompute contrast + histogram.
                vmin, vmax = self._robust_levels(data)
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
                vmin, vmax = self._robust_levels(data)
                self._emit_histogram(data, vmin, vmax)
            self._last_levels = (vmin, vmax)
            axes, units = self._axes_info(data)
            if axes is not None:
                self._plot2d.set_data(data.astype(np.float32),
                                      x_axis=axes[0], y_axis=axes[1], units=units)
                # set_data updates x_axis/units but does NOT recompute scale_x/
                # scale_y from the new axes — so the auto scale bar would size off
                # the stale init scale. set_extent recomputes the physical scale.
                try:
                    self._plot2d.set_extent(axes[0], axes[1])
                except Exception as e:
                    logger.debug("set_extent failed: %s", e)
            else:
                self._plot2d.set_data(data.astype(np.float32))
            self._plot2d.set_clim(vmin, vmax)
        elif dims == 1 and self._plot1d is not None:
            self._plot1d.set_data(data)

    def _emit_histogram(self, data: np.ndarray, vmin: float, vmax: float) -> None:
        """Send a histogram of the current image to the sidebar for this window.

        Only emitted on auto-level (new data / contrast), not on every selector
        drag, so it doesn't flood the channel.
        """
        if self.window_id is None:
            return
        try:
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                return
            counts, edges = np.histogram(finite, bins=64)
            from spyde.backend.ipc import emit
            emit({
                "type": "histogram",
                "window_id": self.window_id,
                "counts": counts.astype(int).tolist(),
                "edges": [float(e) for e in edges],
                "vmin": float(vmin),
                "vmax": float(vmax),
            })
        except Exception as e:
            logger.debug("histogram emit failed: %s", e)

    @staticmethod
    def _robust_levels(img: np.ndarray) -> tuple:
        try:
            sy = max(1, img.shape[0] // 512)
            sx = max(1, img.shape[1] // 512) if img.ndim > 1 else 1
            data = np.asarray(img[::sy, ::sx] if img.ndim > 1 else img[::sy],
                              dtype=np.float64)
            data = data[np.isfinite(data)]
            if data.size == 0:
                return 0.0, 1.0
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

        # Static plots (e.g. the navigator image) display the signal's own data
        # directly. Dynamic plots (driven by a selector) are filled by the
        # selector instead. For lazy signals the data is a Future here and is
        # pushed later by the progressive compute, so only push real arrays.
        if not new_state.dynamic:
            data = getattr(new_state.current_signal, "data", None)
            if isinstance(data, np.ndarray) and data.ndim == dims:
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
        except Exception:
            pass
        return 4 << 20  # 4 MB fallback when the frame size isn't known yet

    @property
    def shared_memory(self) -> SharedMemory:
        if self._shared_memory is None:
            self._shared_memory = SharedMemory(
                name=f"plot_buffer{id(self)}",
                create=True,
                size=self._buffer_nbytes(),
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
        except Exception:
            pass
        if self.session is not None:
            self.session.unregister_plot(self)
        if self._shared_memory is not None:
            try:
                self._shared_memory.close()
                self._shared_memory.unlink()
            except Exception:
                pass
            self._shared_memory = None
        if self.fig_id is not None:
            try:
                del _electron._figures[self.fig_id]
            except Exception:
                pass

    def hide(self) -> None:
        from spyde.backend.ipc import emit
        emit({"type": "window_visibility", "window_id": self.window_id, "visible": False})

    def show(self) -> None:
        from spyde.backend.ipc import emit
        emit({"type": "window_visibility", "window_id": self.window_id, "visible": True})

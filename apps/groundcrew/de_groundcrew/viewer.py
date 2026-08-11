"""
viewer.py — the live image pane.

One anyplotlib figure, created once and then repainted in place. Creating a
figure emits its HTML to the renderer, which mounts it in an iframe keyed by
`fig_id`; every subsequent frame is a `set_data` push down the same channel, so
the iframe is never rebuilt and the user's zoom survives the next frame.

This is deliberately much smaller than SpyDE's `Plot`. SpyDE's carries the array
cache, the tiered navigator read, overlay layers and tile mode, all of which
exist because it displays multi-gigabyte lazy datasets. A camera hands over a
frame that is already in RAM, so none of it applies. When the shell eventually
grows a shared figure wrapper, THIS is the shape it should be, with SpyDE's
extras layered on top — not the other way round.

It satisfies the WindowController protocol (a `close()` and a window id), so the
shell's window registry can own its lifetime.
"""
from __future__ import annotations

import logging

import numpy as np

import anyplotlib as apl
import anyplotlib._electron as _electron
from anyplotlib.embed import build_standalone_html

from de_shell.ipc import emit
from de_shell.plotting.colormaps import DEFAULT_COLORMAP

log = logging.getLogger(__name__)


class LiveViewer:
    """A single always-on image pane."""

    def __init__(self, window_id: int, title: str = "Live view") -> None:
        self.window_id = window_id
        self.title = title
        #: Assigned by anyplotlib in `open()`. It must be the id `_electron.register`
        #: hands back — a made-up string leaves the figure unregistered, so no
        #: trait observers are attached and `set_data` emits NOTHING. The symptom
        #: is a figure that mounts, sizes and titles correctly and then stays on
        #: its opening placeholder forever, with no error anywhere.
        self.fig_id: str | None = None
        self._fig = None
        self._axes = None
        self._plot2d = None
        self._colormap = DEFAULT_COLORMAP
        self._closed = False
        #: Set once the first real frame has been painted. Until then the figure
        #: holds a placeholder, and levels have to be computed rather than kept.
        self._has_frame = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self, shape: tuple[int, int]) -> None:
        """Create the figure and send it to the renderer. Idempotent."""
        if self._fig is not None:
            return
        h, w = shape
        self._fig, axes_obj = apl.subplots(1, 1)
        self._axes = axes_obj[0][0] if isinstance(axes_obj, list) else axes_obj
        # Placeholder at the real frame size so the pane is laid out correctly
        # before the first exposure lands — a 10×10 stand-in would resize the
        # view a moment later and make the window visibly jump.
        self._plot2d = self._axes.imshow(
            np.zeros((h, w), dtype=np.float32), cmap=self._colormap, gpu="auto")
        self._plot2d.set_title(self.title)

        # Register BEFORE building the HTML: registration attaches the trait
        # observers that turn every later `set_data` into a push, and it mints
        # the fig_id the renderer routes those pushes by — so the HTML has to be
        # built with the same id.
        self.fig_id = _electron.register(self._fig)

        html = build_standalone_html(self._fig, fig_id=self.fig_id, resizable=False)
        emit({
            "type": "figure",
            "fig_id": self.fig_id,
            "window_id": self.window_id,
            "html": html,
            "title": self.title,
            "is_navigator": False,
            "aspect": (w / h) if h else None,
        })

    def close(self) -> None:
        """WindowController.close — idempotent, and never raises during teardown."""
        if self._closed:
            return
        self._closed = True
        self._plot2d = None
        self._axes = None
        self._fig = None

    # ── Painting ──────────────────────────────────────────────────────────────

    def show(self, frame: np.ndarray) -> None:
        """Paint one frame. Main thread only."""
        if self._closed or self._plot2d is None:
            return
        self._plot2d.set_data(frame, clim=self._levels(frame))
        self._has_frame = True

    def set_colormap(self, name: str) -> None:
        if self._closed or self._plot2d is None:
            return
        self._colormap = name
        try:
            self._plot2d.set_colormap(name)
        except Exception as e:
            log.debug("set_colormap(%s) failed: %s", name, e)

    @staticmethod
    def _levels(frame: np.ndarray) -> tuple[float, float]:
        """Robust display range for one frame.

        Percentiles, not min/max: a single hot pixel (which a real detector
        always has) would otherwise set the ceiling and render the whole image
        black. Recomputed per frame because the scene is live — holding levels
        fixed is a separate feature, not a default.
        """
        lo, hi = np.percentile(frame, (2.0, 98.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(frame)), float(np.max(frame))
        if hi <= lo:
            hi = lo + 1.0    # a uniform frame still needs a non-degenerate range
        return float(lo), float(hi)

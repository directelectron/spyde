"""
figure.py — a single image pane, created once and repainted in place.

The smallest correct anyplotlib figure a shell app can own. Creating one emits
its HTML to the renderer, which mounts it in an iframe keyed by ``fig_id``;
every later frame is a ``set_data`` push down the same channel, so the iframe is
never rebuilt and the user's zoom survives the next frame.

**This is deliberately the small shape, not SpyDE's.** SpyDE's ``Plot`` carries
the array cache, the tiered navigator read, overlay layers and tile mode — all
of which exist because it displays multi-gigabyte lazy datasets on disk. A live
camera hands over a frame that is already in RAM, so none of it applies. The
shared wrapper is the in-memory core; an app that needs the out-of-core
machinery layers it on top rather than the shell carrying it for everyone.

It satisfies the WindowController protocol (a ``window_id`` and a ``close()``),
so a session's window registry can own its lifetime.

Two things here are easy to get wrong and cost a debugging session each; both
are enforced rather than documented-and-hoped:

* **Registration mints the fig_id.** ``_electron.register(fig)`` attaches the
  trait observers that turn a later ``set_data`` into a push, and returns the id
  the renderer routes those pushes by. Inventing an id instead leaves the figure
  unregistered: it mounts, sizes and titles correctly, and then never updates,
  with no error anywhere.
* **Levels are robust by default.** One hot pixel — which every real detector
  has — sets the ceiling under a plain min/max and renders the whole image
  black.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

import anyplotlib as apl
import anyplotlib._electron as _electron
from anyplotlib.embed import build_standalone_html

from de_shell.ipc import emit
from de_shell.plotting.colormaps import DEFAULT_COLORMAP

log = logging.getLogger(__name__)


def robust_levels(frame: np.ndarray,
                  low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    """A display range for *frame* that a hot pixel cannot wreck.

    Percentiles rather than min/max, with two fallbacks that matter in practice:
    a frame that is all-NaN (or whose percentiles collapse) falls back to
    min/max, and a UNIFORM frame still gets a non-degenerate range, because a
    zero-width window renders as a solid block and is indistinguishable from a
    broken decode.
    """
    arr = np.asarray(frame)
    with warnings.catch_warnings():
        # An all-NaN frame is a case this function HANDLES (see below), so
        # numpy's "All-NaN slice encountered" is noise, not a signal — and it
        # would fire on every frame of a detector that is not yet streaming.
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            # nanpercentile, not percentile: a single NaN makes the latter
            # return NaN for the whole frame.
            lo, hi = np.nanpercentile(arr, (low, high))
        except Exception:
            lo, hi = np.nan, np.nan

    # Non-finite means there was nothing to measure (an all-NaN frame, or an
    # empty one). Only THEN fall back to the extremes.
    if not np.isfinite(lo) or not np.isfinite(hi):
        finite = arr[np.isfinite(arr)] if arr.size else arr
        lo, hi = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)

    # Collapsed percentiles mean the BULK of the frame is one value — a flat
    # field, or a flat field plus a few outliers. Widen by a hair and stop.
    #
    # Falling back to min/max here would be actively wrong, and was: on a
    # uniform frame carrying one hot pixel, min/max restores exactly the outlier
    # this function exists to reject, and the image renders black again.
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


class FigureView:
    """One always-on image pane.

    Parameters
    ----------
    window_id
        The window this figure belongs to, as minted by the session.
    title
        Shown above the image and used as the window title.
    colormap
        Initial colormap name.
    gpu
        Passed through to ``imshow``. ``"auto"`` renders large scalar images on
        the GPU and falls back to Canvas2D for small ones, RGB, and machines
        without it; ``"off"`` forces Canvas2D (what a CPU-reference screenshot
        test wants).
    """

    def __init__(self, window_id: int, title: str = "", *,
                 colormap: str = DEFAULT_COLORMAP, gpu: str = "auto") -> None:
        self.window_id = window_id
        self.title = title
        #: Assigned by `open()` from `_electron.register` — see the module note.
        self.fig_id: str | None = None
        self._fig = None
        self._axes = None
        self._plot2d = None
        self._colormap = colormap
        self._gpu = gpu
        self._closed = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._plot2d is not None and not self._closed

    def open(self, shape: tuple[int, int], *, is_navigator: bool = False) -> str | None:
        """Create the figure and send it to the renderer. Idempotent.

        *shape* is the frame size to lay out for. Passing the REAL size (rather
        than a small stand-in) matters: a placeholder of a different shape makes
        the pane resize the moment the first frame lands, which reads as the
        window jumping.

        Returns the fig_id.
        """
        if self._fig is not None:
            return self.fig_id
        h, w = int(shape[0]), int(shape[1])
        self._fig, axes_obj = apl.subplots(1, 1)
        self._axes = axes_obj[0][0] if isinstance(axes_obj, list) else axes_obj
        self._plot2d = self._axes.imshow(
            np.zeros((h, w), dtype=np.float32), cmap=self._colormap, gpu=self._gpu)
        if self.title:
            self._plot2d.set_title(self.title)

        # Register BEFORE building the HTML: registration attaches the trait
        # observers AND mints the id, so the HTML must be built with that id.
        self.fig_id = _electron.register(self._fig)
        html = build_standalone_html(self._fig, fig_id=self.fig_id, resizable=False)

        emit({
            "type": "figure",
            "fig_id": self.fig_id,
            "window_id": self.window_id,
            "html": html,
            "title": self.title,
            "is_navigator": is_navigator,
            "aspect": (w / h) if h else None,
        })
        return self.fig_id

    def close(self) -> None:
        """WindowController.close — idempotent, and never raises during teardown."""
        if self._closed:
            return
        self._closed = True
        self._plot2d = None
        self._axes = None
        self._fig = None

    # ── Painting ──────────────────────────────────────────────────────────────

    def show(self, frame: np.ndarray, *,
             clim: tuple[float, float] | None = None) -> bool:
        """Paint one frame. **Main thread only** — see the threading contract in
        ``de_shell.actions.lifecycle``.

        ``clim=None`` re-derives a robust range per frame, which is what a live
        scene wants; pass an explicit range to hold contrast steady across
        frames. Returns whether the paint landed, so a caller that must know
        (a live preview, a test) is not left inferring it from a counter that
        increments either way.
        """
        if not self.is_open:
            return False
        try:
            self._plot2d.set_data(
                frame, clim=clim if clim is not None else robust_levels(frame))
            return True
        except Exception as e:
            log.debug("painting figure %s failed: %s", self.fig_id, e)
            return False

    def set_colormap(self, name: str) -> None:
        self._colormap = name
        if not self.is_open:
            return
        try:
            self._plot2d.set_colormap(name)
        except Exception as e:
            log.debug("set_colormap(%s) failed: %s", name, e)

    def set_title(self, title: str) -> None:
        self.title = title
        if not self.is_open:
            return
        try:
            self._plot2d.set_title(title)
        except Exception as e:
            log.debug("set_title(%s) failed: %s", title, e)

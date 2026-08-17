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

import numpy as np

import anyplotlib as apl
import anyplotlib._electron as _electron
from anyplotlib.embed import build_standalone_html

from de_shell.ipc import emit
from de_shell.plotting.colormaps import DEFAULT_COLORMAP

log = logging.getLogger(__name__)


#: Subsample any axis longer than this before measuring levels. Percentiles over
#: 16 M pixels cost tens of ms and land on the paint path; a ≤512² sample gives
#: the same answer to well within a display level.
LEVEL_SAMPLE = 512


def robust_levels(frame: np.ndarray, *,
                  low: float | None = 2.0, high: float = 98.0,
                  sample: int = LEVEL_SAMPLE) -> tuple[float, float]:
    """A display range for *frame* that a hot pixel cannot wreck.

    Percentiles rather than min/max: one saturated pixel — which every real
    detector has — otherwise sets the ceiling and renders everything else black.

    ``low=None`` uses the true minimum instead of a low percentile. That is the
    right choice for an image with no saturating spike (a navigator, a virtual
    image), where clipping the floor throws away real dynamic range; an image
    that DOES have one (a diffraction pattern, whose central beam is orders of
    magnitude brighter than the spots) wants both ends clipped.

    Lifted from SpyDE's ``Plot._robust_levels``, which had already paid for the
    subsampling and the non-finite handling.
    """
    arr = np.asarray(frame)
    try:
        sy = max(1, arr.shape[0] // sample)
        sx = max(1, arr.shape[1] // sample) if arr.ndim > 1 else 1
        data = np.asarray(arr[::sy, ::sx] if arr.ndim > 1 else arr[::sy],
                          dtype=np.float64)
        data = data[np.isfinite(data)]
        if data.size == 0:
            # Nothing measurable — an all-NaN frame, or an empty one.
            return 0.0, 1.0
        # No warnings.catch_warnings() here, deliberately. `data` is already
        # finite-filtered above, so numpy's all-NaN RuntimeWarning cannot fire —
        # and this runs on the PAINT path, where entering a context manager that
        # saves and restores global warning state on every frame is pure cost.
        lo = float(np.percentile(data, low)) if low is not None else float(data.min())
        hi = float(np.percentile(data, high))
        # Collapsed percentiles mean the BULK of the frame is one value. Fall
        # back to the true maximum — load-bearing for a SPARSE image, e.g. a
        # count map that is >99.5% zeros with a few bright spots: the percentile
        # is zero there, and without this the spots all saturate against a
        # 1-wide window instead of scaling properly.
        #
        # The cost is that a perfectly FLAT frame carrying one hot pixel scales
        # to that pixel and renders dark. That is the right trade: a sparse real
        # image is common and a flat synthetic one is not, and any frame with
        # genuine variation never reaches this branch.
        if hi <= lo:
            hi = float(data.max())
        # Still collapsed: genuinely uniform. Widen by a hair rather than return
        # a zero-width window, which renders as a solid block and is
        # indistinguishable from a broken decode.
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi
    except Exception:
        return 0.0, 1.0


#: Figure chrome background. The apps are dark; anyplotlib's template is not.
FIGURE_BACKGROUND = "#1e1e2e"


def fill_iframe_html(html: str, *, background: str = FIGURE_BACKGROUND,
                     extra_head: str = "") -> str:
    """Make a standalone figure FILL its iframe, and match the app's theme.

    Load-bearing for any app that drives figure size with
    ``_electron.resize_figure``. anyplotlib's standalone template pins
    ``html``/``body`` to the figure's INITIAL pixel size with
    ``overflow:hidden`` — correct for a fixed docs or notebook embed. A shell app
    resizes the figure live, so once the pane is larger than that initial size
    the grown figure is CLIPPED to the old body box: the image spills past the
    panel and the bottom is cut off, while everything around it looks fine.

    ``extra_head`` is injected alongside, for an app that needs its own script in
    the frame (SpyDE relays a pointerdown to bring its subwindow to the front).
    """
    style = (f"<style>html,body{{background:{background} !important;color-scheme:dark;"
             "width:100% !important;height:100% !important;overflow:hidden}"
             f"#widget-root{{background:{background} !important;"
             "width:100% !important;height:100% !important;display:block !important}"
             "</style>")
    return html.replace("<body>", style + extra_head + "<body>", 1)


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
        self._tiled = False

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
        html = fill_iframe_html(
            build_standalone_html(self._fig, fig_id=self.fig_id, resizable=False))

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
        self._tiled = False
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

    # ── Tiled display ─────────────────────────────────────────────────────────

    def enable_tile(self, backend, *, integration_method: str = "mean") -> bool:
        """Render through a :class:`TileBackend` instead of pushed frames.

        anyplotlib then owns the loop: it shows a downsampled overview as the
        base and, on its own debounced ``view_changed``, asks the backend for a
        hi-res tile of just the visible region at panel resolution. So the
        source can be larger than anything worth sending whole — an 8192²
        detector, or a camera that is only ever asked for the crop on screen.

        Backends are duck-typed (``full_shape``, ``dtype``, ``origin``,
        ``extent()``, ``sample()``); the shell requires no particular class.

        Returns whether tiling was enabled — an anyplotlib without the tile API
        is a soft failure, and the caller can fall back to :meth:`show`.
        """
        if not self.is_open or not hasattr(self._plot2d, "enable_tile"):
            return False
        try:
            self._plot2d.enable_tile(backend, integration_method=integration_method)
            self._tiled = True
            return True
        except Exception as e:
            log.warning("enable_tile failed, falling back to pushed frames: %s", e)
            return False

    @property
    def is_tiled(self) -> bool:
        return self._tiled

    def refresh_tile(self) -> bool:
        """Re-read the CURRENT view from the backend — the live-data path.

        The zoom and pan persist across the refresh, which is the contract that
        matters for a live camera: new pixels arrive without the user's
        viewport being reset out from under them.
        """
        if not self.is_open or not self._tiled:
            return False
        try:
            self._plot2d.update_tile_source()
            return True
        except Exception as e:
            log.debug("refresh_tile failed: %s", e)
            return False

    def set_clim(self, vmin: float, vmax: float) -> bool:
        """Set the display range without touching the pixels.

        The separate call matters in tile mode: :meth:`show` is never called
        there, so there is no ``clim=`` argument to ride along with. Left
        unset, a tiled plot keeps whatever range the placeholder passed to
        ``imshow`` established — which renders real data as a uniform white or
        black panel.
        """
        if not self.is_open:
            return False
        try:
            self._plot2d.set_clim(float(vmin), float(vmax))
            return True
        except Exception as e:
            log.debug("set_clim(%s, %s) failed: %s", vmin, vmax, e)
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

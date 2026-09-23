"""
figure_window.py — opening a bare-figure window.

A bare-figure window is an anyplotlib figure emitted as a raw ``figure``
message rather than through a registered ``Plot`` (``actions/README.md`` §6).
Every one needs the same steps: register the figure with the renderer bridge,
turn it into HTML, keep the Python figure referenced for as long as the window
lives, and emit the message. :func:`register_figure` and :func:`emit_figure` are
those steps; :class:`ImageGrid` is the common window built on them — a grid of
grey image panels that are repainted and retitled in place.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

#: A bare figure never receives ``resize_figure`` (that path resolves a
#: registered ``Plot``), so its INITIAL px size is the one it keeps and anything
#: outside it is CLIPPED by the subwindow. The renderer sizes a new window from
#: the ``aspect`` field as ``inner_h = clamp(460 / aspect, 130, 300)`` then
#: ``inner_w = inner_h * aspect`` (``MDIArea.windowSize``). At the height cap
#: the first clamp is active for any aspect below 460/300, so a figure exactly
#: :data:`FIGURE_HEIGHT` tall lands pixel-for-pixel in its window at any width
#: up to 460 — pick the width, derive the aspect.
#:
#: The width is the renderer's OWN default. A wider window no longer fits beside
#: the movie, so the free-slot packer wraps it to the next row — on top of the
#: caret, which is an overlay the packer cannot see.
FIGURE_WIDTH = 340
FIGURE_HEIGHT = 300


def figure_geometry(width: int = FIGURE_WIDTH) -> tuple[tuple[int, int], float]:
    """``(figsize, aspect)`` that opens a bare-figure window with no clipping."""
    width = int(min(460, max(190, width)))
    return (width, FIGURE_HEIGHT), width / float(FIGURE_HEIGHT)


def register_figure(figure, *, standalone: bool = False) -> tuple[str, str]:
    """Register *figure* with the renderer bridge → ``(figure_id, html)``."""
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    figure_id = _electron.register(figure)
    return figure_id, finalize_figure_html(figure, figure_id, standalone=standalone)


def emit_figure(window_id: int, figure, title: str, *, registered=None,
                is_navigator: bool = False, rename: bool = False,
                **fields) -> str:
    """Show *figure* in window *window_id* and return its figure id.

    ``registered`` is the ``(figure_id, html)`` pair from :func:`register_figure`
    when the figure was built and registered elsewhere; otherwise it is
    registered here. The figure stays referenced until the window is forgotten.
    ``fields`` are extra keys of the ``figure`` message (``view``, ``aspect``, …).

    A ``figure`` message does not rename an existing window, so ``rename`` also
    emits a ``window_title``.
    """
    from de_shell import ipc
    from de_shell.actions.figure_registry import keep_alive

    figure_id, html = registered or register_figure(figure)
    keep_alive(int(window_id), figure)
    ipc.emit({"type": "figure", "fig_id": figure_id, "window_id": int(window_id),
              "html": html, "title": title, "is_navigator": is_navigator,
              **fields})
    if rename:
        ipc.emit({"type": "window_title", "window_ids": [int(window_id)],
                  "title": title})
    return figure_id


class ImageGrid:
    """A ``rows × cols`` grid of grey image panels, keyed by name.

    ``images`` maps each panel's key to its first image, in reading order. The
    panels are repainted and retitled in place; a failure to do either is
    logged and skipped, since these windows are evidence, not results.
    """

    def __init__(self, rows: int, cols: int, images: dict):
        import anyplotlib as apl

        figsize, self.aspect = figure_geometry()
        self.figure, axes = apl.subplots(rows, cols, figsize=figsize)
        self.panels = {
            key: axis.imshow(np.asarray(image, np.float32), cmap="gray")
            for axis, (key, image)
            in zip(np.array(axes, dtype=object).ravel(), images.items())
        }

    def set_data(self, key: str, image) -> None:
        try:
            self.panels[key].set_data(np.asarray(image, np.float32))
        except Exception as exc:
            log.debug("painting the %s panel failed: %s", key, exc)

    def set_titles(self, titles: dict) -> None:
        for key, title in titles.items():
            try:
                self.panels[key].set_title(title)
            except Exception as exc:
                log.debug("titling the %s panel failed: %s", key, exc)

    def show(self, window_id: int, title: str, *, rename: bool = False) -> str:
        """Emit the grid as window *window_id*; returns the figure id."""
        return emit_figure(window_id, self.figure, title, rename=rename,
                           aspect=float(self.aspect))

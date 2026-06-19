"""
views.py — unified per-window "views" (the chip-strip selector + tiling).

A result window can hold several NAMED views of the same navigation field —
e.g. strain εxx / εyy / εxy, or IPF X / Y / Z, or several virtual images. Each
is emitted as a figure tagged with a ``view_label`` (the chip text) and a
``view_kind`` ("2d"/"3d") so the frontend builds one chip strip per window:
single-click shows a view, ⌘-click TILES several to compare.

Tiling uses anyplotlib's native side-by-side axes: ``tile_views`` rebuilds ONE
figure with ``subplots(1, N)`` (``sharex/sharey`` → linked pan/zoom) and a
linked crosshair on every panel — not N separate iframes. The per-view source
arrays are stashed by ``register_views`` so the tiled figure can be rebuilt for
any selected subset.
"""
from __future__ import annotations

import numpy as np

# Keep emitted figures alive past the emit (the registry holds a weak ref only).
_ALIVE: list = []

# window_id → {"images": {label: np.ndarray}, "order": [label,...],
#              "cmap": str, "levels": (lo, hi) | None}
# The source arrays for each named view, so a tiled (multi-axis) figure can be
# rebuilt for any selected subset without recomputing.
_VIEW_DATA: dict[int, dict] = {}

# The figure that carries a tiled comparison is tagged with this reserved
# view_label so the frontend (a) shows it when ≥2 chips are selected and
# (b) excludes it from the chip strip.
TILED_LABEL = "__tiled__"


def register_views(window_id: int, items, *, cmap: str = "gray", levels=None) -> None:
    """Stash the source arrays for a window's named views so ``tile_views`` can
    rebuild a side-by-side figure for any subset. ``items`` = list of
    ``(label, image)``."""
    images = {}
    order = []
    for label, image in items:
        images[label] = np.asarray(image)
        order.append(label)
    _VIEW_DATA[int(window_id)] = {
        "images": images, "order": order, "cmap": cmap, "levels": levels,
    }


def _imshow_view(ax, image, cmap, levels):
    """imshow one view into ``ax`` (RGB as-is, else scalar+cmap+clim) → Plot2D."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        return ax.imshow(img)
    p = ax.imshow(img.astype(np.float32), cmap=cmap)
    if levels is not None:
        try:
            p.set_clim(float(levels[0]), float(levels[1]))
        except Exception:
            pass
    return p


def emit_view_figure(window_id: int, image, label: str, *, kind: str = "2d",
                     cmap: str = "gray", levels=None) -> str | None:
    """Emit a single-axis map figure tagged as the named view ``label``. Returns
    the fig id (or None on failure)."""
    try:
        import anyplotlib as apl
        import anyplotlib._electron as _electron
        from spyde.drawing.plots.plot import finalize_figure_html
        from spyde.backend.ipc import emit

        fig, axes = apl.subplots(1, 1)
        ax = axes[0][0] if isinstance(axes, list) else axes
        _imshow_view(ax, image, cmap, levels)

        fig_id = _electron.register(fig)
        html = finalize_figure_html(fig, fig_id)
        _ALIVE.append(fig)
        emit({
            "type": "figure", "fig_id": fig_id, "window_id": window_id,
            "html": html, "title": label, "is_navigator": False,
            "view_label": label, "view_kind": kind,
        })
        return fig_id
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("emit_view_figure(%s) failed: %s", label, e)
        return None


def _link_crosshairs(widgets) -> None:
    """Link N anyplotlib crosshair widgets: moving one moves them all (the
    "linked selector"). A re-entrancy guard stops the set→event→set feedback."""
    if len(widgets) < 2:
        return
    state = {"busy": False}

    def make(src):
        def handler(_ev=None):
            if state["busy"]:
                return
            state["busy"] = True
            try:
                cx, cy = src.get("cx"), src.get("cy")
                for w in widgets:
                    if w is src:
                        continue
                    try:
                        w.set(cx=cx, cy=cy)
                    except Exception:
                        pass
            finally:
                state["busy"] = False
        return handler

    for w in widgets:
        h = make(w)
        for et in ("pointer_up", "pointer_settled"):
            try:
                w.add_event_handler(h, et)
            except Exception:
                pass


def build_tiled_figure(window_id: int, labels):
    """Build ONE figure with the selected views as side-by-side axes (anyplotlib
    ``subplots(1, N)``, shared pan/zoom + linked crosshairs). Returns
    ``(fig, fig_id, html, ordered_labels)`` or ``None`` if no data is registered."""
    data = _VIEW_DATA.get(int(window_id))
    if not data:
        return None
    # Preserve the window's canonical view order regardless of click order.
    sel = [l for l in data["order"] if l in set(labels)]
    pairs = [(l, data["images"][l]) for l in sel if l in data["images"]]
    if not pairs:
        return None

    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    cmap, levels = data.get("cmap", "gray"), data.get("levels")
    fig, axes = apl.subplots(1, len(pairs), sharex=True, sharey=True)
    arr = np.array(axes, dtype=object).ravel()
    widgets = []
    for ax, (label, image) in zip(arr, pairs):
        p = _imshow_view(ax, image, cmap, levels)
        try:
            ax.set_title(label)
        except Exception:
            pass
        try:
            h, w = np.asarray(image).shape[:2]
            widgets.append(p.add_crosshair_widget(cx=w / 2.0, cy=h / 2.0))
        except Exception:
            pass
    _link_crosshairs(widgets)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    _ALIVE.append(fig)
    return fig, fig_id, html, sel


def emit_tiled_figure(window_id: int, labels) -> str | None:
    """Build + emit the side-by-side tiled figure for ``window_id``. Tagged with
    the reserved ``__tiled__`` view_label so the frontend swaps it in while ≥2
    chips are selected (and the FIGURE reducer replaces a prior tiled figure)."""
    built = build_tiled_figure(window_id, labels)
    if built is None:
        return None
    _fig, fig_id, html, sel = built
    try:
        from spyde.backend.ipc import emit
        emit({
            "type": "figure", "fig_id": fig_id, "window_id": window_id,
            "html": html, "title": " / ".join(sel), "is_navigator": False,
            "view_label": TILED_LABEL, "view_kind": "tiled",
        })
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("emit_tiled_figure failed: %s", e)
        return None
    return fig_id


def tile_views(session, plot, payload) -> None:
    """Staged handler: (re)build the side-by-side comparison figure for the
    window's selected views. ``payload['labels']`` is the selected chip set."""
    labels = payload.get("labels") or []
    if len(labels) < 2:
        return                      # a single view shows its own figure
    window_id = getattr(plot, "window_id", None)
    if window_id is None:
        return
    emit_tiled_figure(int(window_id), labels)

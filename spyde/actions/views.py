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

import logging

import numpy as np

from spyde.actions.figure_registry import keep_alive

logger = logging.getLogger(__name__)

# window_id → {"images": {label: np.ndarray}, "order": [label,...],
#              "cmap": str, "levels": (lo, hi) | None}
# The source arrays for each named view, so a tiled (multi-axis) figure can be
# rebuilt for any selected subset without recomputing.
_VIEW_DATA: dict[int, dict] = {}

# The figure that carries a tiled comparison is tagged with this reserved
# view_label so the frontend (a) shows it when ≥2 chips are selected and
# (b) excludes it from the chip strip.
TILED_LABEL = "__tiled__"


def _wire_pick(plot2d, image, pick_hook):
    """Put a white crosshair on *plot2d* that reports the picked ``(iy, ix)``
    scan pixel to *pick_hook*. Returns the widget (or None).

    A window whose views are several maps of ONE navigation field wants the
    same pick on every one of them — otherwise the interaction silently dies
    when the user switches chips (the IPF explorer follows the orientation map's
    crosshair, so picking had to keep working on IPF-X and IPF-Y too, and on the
    ⌘-tiled comparison)."""
    if pick_hook is None or plot2d is None:
        return None
    try:
        widget = plot2d.add_crosshair_widget(color="#ffffff")
    except Exception as e:
        logger.debug("adding a view pick crosshair failed: %s", e)
        return None
    return _wire_pick_widget(widget, image, pick_hook)


def _wire_pick_widget(widget, image, pick_hook):
    """Report an EXISTING crosshair widget's position to *pick_hook* as an
    ``(iy, ix)`` scan pixel, bounds-checked against *image*."""
    if widget is None or pick_hook is None:
        return None
    h, w = np.asarray(image).shape[:2]

    def _on_pick(_ev=None):
        try:
            ix = int(round(float(widget.cx)))
            iy = int(round(float(widget.cy)))
        except Exception:
            return
        if 0 <= iy < h and 0 <= ix < w:
            try:
                pick_hook(iy, ix)
            except Exception as e:
                logger.debug("view pick hook failed: %s", e)

    try:
        widget.add_event_handler(_on_pick, "pointer_up")
    except Exception as e:
        logger.debug("wiring the view pick handler failed: %s", e)
    return widget


def register_views(window_id: int, items, *, cmap: str = "gray", levels=None,
                   append: bool = False, pick_hook=None) -> None:
    """Stash the source arrays for a window's named views so ``tile_views`` can
    rebuild a side-by-side figure for any subset. ``items`` = list of
    ``(label, image)``.

    ``append=True`` MERGES into what the window already has instead of replacing
    it — needed when two independent attachers contribute chips to one window
    (an EBSD result window carries the IPF-X/Y/Z projections *and* the
    NCC / Similarity / ADP quality maps; whichever ran second used to erase the
    other's arrays, so ⌘-tiling silently dropped half the chips)."""
    prev = _VIEW_DATA.get(int(window_id)) if append else None
    images = dict(prev["images"]) if prev else {}
    order = list(prev["order"]) if prev else []
    for label, image in items:
        images[label] = np.asarray(image)
        if label not in order:
            order.append(label)
    _VIEW_DATA[int(window_id)] = {
        "images": images, "order": order,
        "cmap": (prev or {}).get("cmap", cmap) if append else cmap,
        "levels": (prev or {}).get("levels", levels) if append else levels,
        # Carried so the TILED comparison figure gets the same pick behaviour
        # its single-view siblings have.
        "pick_hook": pick_hook or ((prev or {}).get("pick_hook") if append else None),
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
        except Exception as e:
            logger.debug("set_clim on view image failed: %s", e)
    return p


def emit_view_figure(window_id: int, image, label: str, *, kind: str = "2d",
                     cmap: str = "gray", levels=None, pick_hook=None,
                     key=None) -> str | None:
    """Emit a single-axis map figure tagged as the named view ``label``. Returns
    the fig id (or None on failure).

    ``pick_hook(iy, ix)`` adds a white pick crosshair to this view (see
    :func:`_wire_pick`).

    ``key`` is an optional ``(rgba, labels)`` pinned image overlay
    (``Plot2D.add_key``). Each chip is its own FIGURE, so a key belongs to the
    figure it annotates — attaching it here is what makes the IPF colour key
    follow whichever projection chip is on screen, which the old separate
    ``view="ipf_key"`` figure had to fake by floating an iframe over the whole
    window."""
    try:
        import anyplotlib as apl
        import anyplotlib._electron as _electron
        from spyde.drawing.plots.plot import finalize_figure_html
        from de_shell.ipc import emit

        fig, axes = apl.subplots(1, 1)
        ax = axes[0][0] if isinstance(axes, list) else axes
        p = _imshow_view(ax, image, cmap, levels)
        _wire_pick(p, image, pick_hook)
        if key is not None and getattr(p, "add_key", None) is not None:
            rgba, key_labels = key
            try:
                from spyde.actions.ipf_view import (
                    IPF_KEY_BG, IPF_KEY_BORDER, IPF_KEY_SIZE,
                )
                p.add_key(rgba, corner="bottom-right", size=IPF_KEY_SIZE,
                          hover_only=True, labels=key_labels, name="ipf_key",
                          bgcolor=IPF_KEY_BG, border=IPF_KEY_BORDER)
            except Exception as e:
                logger.debug("add_key on view %s failed: %s", label, e)

        fig_id = _electron.register(fig)
        html = finalize_figure_html(fig, fig_id)
        keep_alive(window_id, fig)
        emit({
            "type": "figure", "fig_id": fig_id, "window_id": window_id,
            "html": html, "title": label, "is_navigator": False,
            "view_label": label, "view_kind": kind,
        })
        return fig_id
    except Exception as e:
        logger.debug("emit_view_figure(%s) failed: %s", label, e)
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
                    except Exception as e:
                        logger.debug("linking tiled crosshair failed: %s", e)
            finally:
                state["busy"] = False
        return handler

    for w in widgets:
        h = make(w)
        for et in ("pointer_up", "pointer_settled"):
            try:
                w.add_event_handler(h, et)
            except Exception as e:
                logger.debug("wiring tiled crosshair %s handler failed: %s", et, e)


def build_tiled_figure(window_id: int, labels):
    """Build ONE figure with the selected views as side-by-side axes (anyplotlib
    ``subplots(1, N)``, shared pan/zoom + linked crosshairs). Returns
    ``(fig, fig_id, html, ordered_labels)`` or ``None`` if no data is registered."""
    data = _VIEW_DATA.get(int(window_id))
    if not data:
        return None
    # Preserve the window's canonical view order regardless of click order.
    # Match on NFC-normalised labels so a composed/decomposed Unicode mismatch
    # (a label round-tripped through JSON/IPC) can't silently drop a view.
    import unicodedata
    def _nfc(s):
        return unicodedata.normalize("NFC", s)
    wanted = {_nfc(l) for l in labels}
    sel = [l for l in data["order"] if _nfc(l) in wanted]
    pairs = [(l, data["images"][l]) for l in sel if l in data["images"]]
    if not pairs:
        return None

    import anyplotlib as apl
    import anyplotlib._electron as _electron
    from spyde.drawing.plots.plot import finalize_figure_html

    cmap, levels = data.get("cmap", "gray"), data.get("levels")
    pick_hook = data.get("pick_hook")
    fig, axes = apl.subplots(1, len(pairs), sharex=True, sharey=True)
    arr = np.array(axes, dtype=object).ravel()
    widgets = []
    for ax, (label, image) in zip(arr, pairs):
        p = _imshow_view(ax, image, cmap, levels)
        try:
            ax.set_title(label)
        except Exception as e:
            logger.debug("set_title on tiled view failed: %s", e)
        try:
            h, w = np.asarray(image).shape[:2]
            widgets.append(p.add_crosshair_widget(cx=w / 2.0, cy=h / 2.0))
        except Exception as e:
            logger.debug("adding crosshair to tiled view failed: %s", e)
    _link_crosshairs(widgets)
    if pick_hook is not None and widgets:
        # ONE hook on the linked set: _link_crosshairs already mirrors the drag
        # onto the others, so wiring every widget would fire N times per pick.
        _wire_pick_widget(widgets[0], pairs[0][1], pick_hook)

    fig_id = _electron.register(fig)
    html = finalize_figure_html(fig, fig_id)
    keep_alive(int(window_id), fig)
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
        from de_shell.ipc import emit
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
        # Bare-figure windows (the VOM unified strain window, IPF views) have
        # no registered Plot, so dispatch resolves plot=None — fall back to the
        # window id the dispatcher injects into every staged payload.
        window_id = payload.get("window_id")
    if window_id is None:
        return
    emit_tiled_figure(int(window_id), labels)

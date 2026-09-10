"""MDI live image layering: another window's image drawn over this one.

A user drops a signal window's pill onto another open signal window to draw its
image as a translucent, colormapped layer over the target's base image, with a
per-layer colormap, alpha, clim and visibility. A layer is an overlay child of
the node the target displays, added with ``BaseSignalTree.add_overlay``: its
function is the source frame, its group is one anyplotlib ``Layer``, and its
appearance rides every value. The navigator drives it like every other overlay,
so the frame comes from the source window's own readers and blocks, the read
takes the tier that frame's own read takes, and the push runs on the painter
thread. An integrating region reaches the source whole, so a layer integrates
the positions the base image integrates.

The four staged handlers (``overlay_add`` / ``overlay_set`` / ``overlay_remove``
/ ``overlay_query``) share the uniform ``fn(session, plot, payload)`` signature
and are registered in :data:`spyde.actions.registry.STAGED_HANDLERS`. ``plot`` is
the TARGET plot (resolved from ``payload["window_id"]``); ``source_window_id``
names the plot supplying the layer image.

Layers refuse TILE mode: a tiled plot streams detail tiles of a single image and
cannot composite independent layers.
"""
from __future__ import annotations

import logging

import numpy as np

from de_shell import ipc

log = logging.getLogger(__name__)

#: The one group a layer overlay draws with.
LAYER_GROUP = "layer"

# A small palette cycled across the layers added to one plot so a 2nd/3rd overlay
# is visually distinct from the base (which is usually gray/viridis).
_LAYER_CMAP_CYCLE = ["magma", "cividis", "plasma", "inferno", "cool", "spring"]


def layer_frame(frame, *, appearance: dict) -> dict:
    """The source window's frame, drawn as an image layer with ``appearance``.

    Anything that is not a 2-D image cannot composite over the base, so it
    clears the layer rather than pushing a frame anyplotlib would reject.
    """
    frame = np.asarray(frame)
    if frame.ndim != 2 or frame.dtype == object:
        return {}
    return {LAYER_GROUP: {"data": frame, **appearance}}


# ── the layers of a plot ──────────────────────────────────────────────────────


def layer_nodes(plot) -> list:
    """The layer overlays drawn on ``plot``, oldest first."""
    tree = getattr(plot, "signal_tree", None)
    state = getattr(plot, "plot_state", None)
    signal = getattr(state, "current_signal", None) if state is not None else None
    if tree is None or signal is None:
        return []
    return [node for node in tree.overlay_children(signal)
            if LAYER_GROUP in node.groups
            and getattr(node.signal, "target_plot", None) is plot]


def layer_appearance(node) -> dict:
    """A layer's colormap, alpha and clim, the record its dock row shows and
    its next value carries."""
    return dict(node.signal._map_recipe.static.get("appearance") or {})


def layer_source_plot(node):
    """The plot whose image a layer draws."""
    return getattr(node.signal, "source_plot", None)


def _layer_state(node) -> dict:
    appearance = layer_appearance(node)
    clim = appearance.get("clim")
    if clim is not None:
        clim = [float(clim[0]), float(clim[1])]
    return {
        "id": node.name,
        "title": _source_title(layer_source_plot(node)),
        "cmap": str(appearance.get("cmap", "magma")),
        "alpha": float(appearance.get("alpha", 0.5)),
        "clim": clim,
        "visible": bool(node.visible),
    }


def _emit_layers_state(target_plot) -> None:
    """Emit the authoritative ``layers_state`` for one target plot. The
    renderer's Plot-Control "Layers" section is written against this shape."""
    window_id = getattr(target_plot, "window_id", None)
    if window_id is None:
        return
    ipc.emit({
        "type": "layers_state",
        "window_id": int(window_id),
        "layers": [_layer_state(node) for node in layer_nodes(target_plot)],
    })


def _find_layer(target_plot, layer_id):
    for node in layer_nodes(target_plot):
        if node.name == layer_id:
            return node
    return None


def _source_title(source_plot) -> str:
    if source_plot is None:
        return "Layer"
    try:
        signal = source_plot.plot_state.current_signal
        title = str(signal.metadata.get_item("General.title", default="") or "")
        if title:
            return title
    except Exception as e:
        log.debug("reading a layer source's title failed: %s", e)
    return str(getattr(source_plot, "view_label", None) or "Layer")


# ── teardown ──────────────────────────────────────────────────────────────────


def drop_all_layers(target_plot) -> None:
    """Remove every layer drawn on ``target_plot``. Called when it closes, and
    when its image shape changes so nothing can composite over it any more."""
    tree = getattr(target_plot, "signal_tree", None)
    if tree is None:
        return
    for node in layer_nodes(target_plot):
        try:
            tree.remove_overlay(node)
        except Exception as e:
            log.debug("dropping layer %r failed: %s", node.name, e)


def drop_layers_for_source(session, source_plot) -> None:
    """Remove every layer (on ANY target plot) reading from ``source_plot``, so
    closing the source leaves no target compositing an image nobody updates.
    Re-emits ``layers_state`` for each affected target."""
    if session is None:
        return
    for target in list(getattr(session, "_plots", []) or []):
        tree = getattr(target, "signal_tree", None)
        removed = False
        for node in layer_nodes(target):
            if layer_source_plot(node) is not source_plot or tree is None:
                continue
            try:
                tree.remove_overlay(node)
                removed = True
            except Exception as e:
                log.debug("dropping a source-closed layer failed: %s", e)
        if removed:
            _emit_layers_state(target)


# ── handlers ───────────────────────────────────────────────────────────────────


def overlay_add(session, plot, payload) -> None:
    """Add ``source_window_id``'s current image as a live layer over the target
    ``window_id`` plot. Validates: identical signal-frame ``(H, W)``; refuses a
    tiled target; refuses source IS target. Seeds the layer with the source's
    CURRENT frame and emits ``layers_state``."""
    if plot is None:
        ipc.emit_error("overlay_add: target window not found.")
        return
    source_window_id = payload.get("source_window_id")
    source = (session._plot_by_window_id(int(source_window_id))
              if source_window_id is not None else None)
    if source is None:
        ipc.emit_error("overlay_add: source window not found.")
        return
    if source is plot:
        ipc.emit_status("Cannot layer a window onto itself.")
        return

    plot2d = getattr(plot, "_plot2d", None)
    if plot2d is None:
        ipc.emit_status("Overlay: target is not an image plot.")
        return
    if getattr(plot2d, "_tile_on", False):
        ipc.emit_status(
            "Overlay unavailable: the target image is in tile mode (large frame). "
            "Layers are only supported on non-tiled images.")
        return

    base = getattr(plot, "displayed_data", None)
    source_frame = getattr(source, "displayed_data", None)
    if not isinstance(base, np.ndarray) or base.ndim != 2:
        ipc.emit_status("Overlay: target has no 2-D image to layer onto.")
        return
    if not isinstance(source_frame, np.ndarray) or source_frame.ndim != 2:
        ipc.emit_status("Overlay: source has no 2-D image to layer.")
        return
    if source_frame.shape != base.shape:
        ipc.emit_status(
            f"Overlay refused: frame shapes differ "
            f"({source_frame.shape} vs {base.shape}).")
        return

    tree = getattr(plot, "signal_tree", None)
    signal = getattr(getattr(plot, "plot_state", None), "current_signal", None)
    if tree is None or signal is None:
        ipc.emit_status("Overlay: the target window has no signal to draw on.")
        return

    appearance = {
        "cmap": _LAYER_CMAP_CYCLE[len(layer_nodes(plot)) % len(_LAYER_CMAP_CYCLE)],
        "alpha": 0.5,
        "clim": None,
    }
    try:
        node = tree.add_overlay(
            signal, layer_frame, name=LAYER_GROUP,
            groups={LAYER_GROUP: (LAYER_GROUP, {})},
            static={"appearance": appearance},
            source=True, source_plot=source, target_plot=plot,
            # No tier of its own: a layer is the source window's frame, so it
            # takes whichever tier reading that frame takes. A resident one
            # draws in the same pass as the base image.
            expensive=None, follows_region=True)
    except Exception as e:
        ipc.emit_error(f"overlay_add failed: {e}")
        return
    # Draw the source's current frame now: the layer must appear on the drop,
    # not on the next navigator move.
    plot.enqueue_overlay(node, layer_frame(source_frame, appearance=appearance))
    _emit_layers_state(plot)


def overlay_set(session, plot, payload) -> None:
    """Update a layer's appearance (any subset of cmap / alpha / clim / visible)."""
    if plot is None:
        return
    node = _find_layer(plot, payload.get("layer_id"))
    if node is None:
        return
    tree = plot.signal_tree
    appearance = layer_appearance(node)
    if payload.get("cmap"):
        appearance["cmap"] = str(payload["cmap"])
    if payload.get("alpha") is not None:
        try:
            appearance["alpha"] = float(payload["alpha"])
        except (TypeError, ValueError):
            pass
    if "clim" in payload:
        clim = payload["clim"]
        if clim is None:
            appearance["clim"] = None
        else:
            try:
                appearance["clim"] = [float(clim[0]), float(clim[1])]
            except (TypeError, ValueError, IndexError):
                pass
    tree.replace_overlay_static(node, appearance=appearance)
    if payload.get("visible") is not None:
        tree.set_overlay_visible(node, bool(payload["visible"]))
    _emit_layers_state(plot)


def overlay_remove(session, plot, payload) -> None:
    """Remove one layer from the target plot."""
    if plot is None:
        return
    node = _find_layer(plot, payload.get("layer_id"))
    if node is None:
        return
    plot.signal_tree.remove_overlay(node)
    _emit_layers_state(plot)


def overlay_query(session, plot, payload) -> None:
    """Re-emit the target plot's ``layers_state`` (renderer refresh / reconnect)."""
    if plot is None:
        return
    _emit_layers_state(plot)

"""
overlay.py — MDI live image layering (Report Builder Phase 2, Part 2).

A user can drop a signal-window pill onto ANOTHER open signal window to draw its
image as a translucent, colormapped LAYER over the target's base image (per-layer
colormap + alpha + clim + visibility). The layers are anyplotlib ``Layer``s
(``plot2d.add_layer``), composited client-side with the SAME zoom/pan transform as
the base — and they track the base image LIVE: when the target's displayed frame
changes because the navigator moved, each visible layer re-reads its OWN source at
the same navigation indices and refreshes.

The four staged handlers (``overlay_add`` / ``overlay_set`` / ``overlay_remove`` /
``overlay_query``) share the uniform ``fn(session, plot, payload)`` signature and
are registered in :data:`spyde.actions.registry.STAGED_HANDLERS`. ``plot`` is the
TARGET plot (resolved from ``payload["window_id"]``); ``source_window_id`` names the
plot supplying the layer image.

Live-nav contract (the sensitive part — see CLAUDE.md "Live-Display Core Patterns"
and "Thread Safety"):

* The nav READ for a layer runs on the SAME serial ``_NavDispatcher`` thread as the
  base read (``refresh_plot_layers`` is called from ``_run_update`` right after the
  base frame is enqueued). It uses the cheap SYNCHRONOUS cached read only — if the
  base move was routed to the async/expensive tier the layer refresh is skipped, and
  the selector's settle re-fire (which runs the cheap path for the resting position)
  catches the layers up. No new threads, no locks.
* The layer PUSH (``Layer.set_data`` → ``_push`` → stdout) is DECOUPLED onto the
  same newest-wins painter thread that paints the base frame (``_NavPainter``): the
  dispatcher stashes the freshly-read layer frames on the target plot and the painter
  applies them right after ``_set_array``, so stdout writes stay serialized and a slow
  push never blocks reading the next slider position.
* The non-layered path is ZERO-COST: ``refresh_plot_layers`` returns immediately when
  ``plot._layers`` is empty (the common case).

Layers refuse TILE mode (a tiled plot streams detail tiles of a single image and
cannot composite independent layers) — ``overlay_add`` emits a status and no-ops.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from spyde.backend import ipc

log = logging.getLogger(__name__)

# A small palette cycled across the layers added to one plot so a 2nd/3rd overlay
# is visually distinct from the base (which is usually gray/viridis).
_LAYER_CMAP_CYCLE = ["magma", "cividis", "plasma", "inferno", "cool", "spring"]


@dataclass
class PlotLayer:
    """One live overlay layer on a target :class:`~spyde.drawing.plots.plot.Plot`.

    Holds the layer id (the anyplotlib ``Layer.id``), a reference to the SOURCE
    plot supplying the image, the appearance (cmap/alpha/clim/visible), and the
    anyplotlib ``Layer`` handle. ``source_plot`` is a weak-ish direct ref; if the
    source's tree closes the layer is dropped by :func:`drop_layers_for_source`."""
    layer_id: str
    source_plot: object
    cmap: str = "magma"
    alpha: float = 0.5
    clim: "list | None" = None
    visible: bool = True
    handle: object = None                       # anyplotlib Layer
    title: str = ""

    def to_state(self) -> dict:
        clim = None
        if self.clim is not None and self.clim[0] is not None and self.clim[1] is not None:
            clim = [float(self.clim[0]), float(self.clim[1])]
        return {
            "id": self.layer_id,
            "title": self.title,
            "cmap": self.cmap,
            "alpha": float(self.alpha),
            "clim": clim,
            "visible": bool(self.visible),
        }


# ── frame reading (cheap synchronous tier only) ───────────────────────────────


def _read_source_frame(source_plot, indices, integrating=False):
    """Read the SOURCE plot's frame at the SAME navigation ``indices`` the target
    is showing, CHEAPLY + SYNCHRONOUSLY on the caller's (dispatcher) thread. Returns
    a numpy array, or None to skip this refresh (source not paintable, or the read
    would be expensive — the settle re-fire will catch it up).

    ``indices`` is the RAW selector index array (same object handed to the base
    ``update_from_navigation_selection``); this re-runs the identical transpose /
    mean-reduce / clamp / dtype logic against the source's own signal, so a layer on
    a same-tree source always resolves the same nav position as the base.
    ``integrating`` mirrors the driving selector's mode so a REGION read integrates
    the same nav points the base frame does (a crosshair reduces to one point).

    For a lazy source it forces the SYNCHRONOUS cached/direct read (never the async
    tier) so it can never spawn a background future on the source plot or block the
    dispatcher on a heavy graph — a large-region / cold-huge read simply returns None
    and is picked up by the settle re-fire.
    """
    try:
        ps = getattr(source_plot, "plot_state", None)
        if ps is None:
            return None
        current_signal = ps.current_signal
    except Exception:
        return None

    try:
        from spyde.drawing.update_functions import (
            NavProfile, _classify_nav_read, _direct_read_frame,
            _prepare_nav_indices,
        )
    except Exception as e:
        log.debug("overlay layer read imports failed: %s", e)
        return None

    try:
        idx = _prepare_nav_indices(current_signal, indices, integrating=integrating)
    except Exception as e:
        log.debug("overlay layer index prep failed: %s", e)
        return None
    if idx is None:
        return None

    prof = NavProfile(getattr(source_plot, "window_id", "LAYER"), idx)

    try:
        if getattr(current_signal, "_lazy", False):
            data = getattr(current_signal, "data", None)
            if data is None or not hasattr(data, "compute") or not hasattr(data, "chunks"):
                return None
            # Skip the loading placeholder (Future-bearing data).
            try:
                from distributed import Future
                if isinstance(data[0], Future):
                    return None
            except Exception:
                pass
            nav_dim = current_signal.axes_manager.navigation_dimension
            frame_shape = data.shape[nav_dim:]
            frame_bytes = int(np.prod(frame_shape)) * data.dtype.itemsize
            # Expensive reads (large region / cold huge frame) are NOT done inline
            # for a layer — skip; the settle re-fire runs the cheap path.
            if _classify_nav_read(current_signal, idx, data, frame_bytes) == "expensive":
                return None
            frame = _direct_read_frame(current_signal, None, idx, prof, child=source_plot)
            if frame is None:
                # Fall back to the synchronous cached read (base signal path).
                cached_arr = getattr(current_signal, "cached_dask_array", None)
                if cached_arr is not None:
                    try:
                        cached_arr._client = None
                    except Exception:
                        pass
                frame = np.asarray(
                    current_signal._get_cache_dask_chunk(idx, get_result=True))
                src_dtype = getattr(current_signal.data, "dtype", None)
                if (src_dtype is not None and np.issubdtype(src_dtype, np.integer)
                        and np.issubdtype(frame.dtype, np.floating)):
                    frame = frame.astype(src_dtype, copy=False)
            return np.asarray(frame)
        # Eager (in-RAM) slice — one nav point, or a region mean (same semantics as
        # the base eager read in update_from_navigation_selection).
        idx_arr = np.asarray(idx)
        data = current_signal.data
        if idx_arr.ndim <= 1:
            point = tuple(int(v) for v in np.atleast_1d(idx_arr))
            return np.asarray(data[point])
        sl = tuple(idx_arr[:, k].astype(int) for k in range(idx_arr.shape[1]))
        mean = np.asarray(data[sl]).mean(axis=0)
        if np.issubdtype(data.dtype, np.integer):
            mean = np.rint(mean).astype(data.dtype)
        return mean
    except Exception as e:
        log.debug("overlay layer frame read failed: %s", e)
        return None


# ── the post-paint hook (dispatcher-thread read → painter-thread push) ────────


def refresh_plot_layers(target_plot, indices, integrating=False) -> None:
    """Post-paint hook: refresh every VISIBLE layer of ``target_plot`` from its own
    source at ``indices``. Called from ``BaseSelector._run_update`` right after the
    base frame is enqueued, on the serial ``_NavDispatcher`` thread. ``integrating``
    mirrors the driving selector's mode (region vs crosshair).

    ZERO-COST when the plot has no layers (the common case). For each visible layer
    it does a cheap SYNCHRONOUS read of the source frame HERE (dispatcher thread),
    then hands the frames to the plot's painter thread to push (``set_data``) — so
    the stdout pushes stay serialized and off the dispatcher, mirroring the base
    frame's read/paint split (Live-Display §2/§3, Thread-Safety)."""
    layers = getattr(target_plot, "_layers", None)
    if not layers:
        return
    frames: "list[tuple[object, np.ndarray]]" = []
    base = getattr(target_plot, "current_data", None)
    base_shape = base.shape[:2] if isinstance(base, np.ndarray) and base.ndim >= 2 else None
    for layer in list(layers):
        if not layer.visible or layer.handle is None:
            continue
        src = layer.source_plot
        if src is None:
            continue
        frame = _read_source_frame(src, indices, integrating=integrating)
        if frame is None:
            continue
        frame = np.asarray(frame)
        # Only 2-D scalar frames can layer (matches the base image); an RGB / 1-D
        # source can't overlay a 2-D image.
        if frame.ndim != 2:
            continue
        # Guard a source whose frame shape drifted from the base (e.g. a node
        # switch on the source): a mismatch would raise in add/set_data.
        if base_shape is not None and frame.shape != base_shape:
            continue
        frames.append((layer.handle, frame))
    if frames:
        _enqueue_layer_push(target_plot, frames)


def _enqueue_layer_push(target_plot, frames) -> None:
    """Stash freshly-read layer frames on the plot and wake the painter to apply
    them (``Layer.set_data``) after the base ``_set_array`` — keeping every stdout
    push serialized on the one painter thread. Newest-wins: a later refresh replaces
    the pending frames before the painter runs (mirrors the base paint decouple)."""
    try:
        target_plot._pending_layer_frames = frames
        # Re-submit the plot's CURRENT base frame to the painter so it wakes and
        # drains _pending_layer_frames (the painter applies base then layers). If
        # there's no base frame yet, push the layers directly (rare).
        base = getattr(target_plot, "current_data", None)
        if isinstance(base, np.ndarray):
            target_plot.enqueue_paint(base)
        else:
            _apply_pending_layer_frames(target_plot)
    except Exception as e:
        log.debug("enqueue layer push failed: %s", e)


def _apply_pending_layer_frames(target_plot) -> None:
    """Apply (and clear) any pending layer frames on ``target_plot`` by calling each
    layer handle's ``set_data``. Called on the PAINTER thread from ``_NavPainter``
    right after ``_set_array`` (so pushes stay serialized), and as a direct fallback
    when there's no base frame. Safe to call with nothing pending."""
    frames = getattr(target_plot, "_pending_layer_frames", None)
    if not frames:
        return
    target_plot._pending_layer_frames = None
    for handle, frame in frames:
        try:
            handle.set_data(frame)
        except Exception as e:
            log.debug("layer set_data failed: %s", e)


# ── teardown ──────────────────────────────────────────────────────────────────


def drop_all_layers(target_plot) -> None:
    """Remove every layer from ``target_plot`` (called on plot close). Drops the
    anyplotlib handles and clears the list so no dangling refs remain."""
    layers = getattr(target_plot, "_layers", None)
    if not layers:
        return
    p2 = getattr(target_plot, "_plot2d", None)
    for layer in list(layers):
        try:
            if layer.handle is not None:
                layer.handle.remove()
            elif p2 is not None:
                p2.remove_layer(layer.layer_id)
        except Exception as e:
            log.debug("dropping layer on close failed: %s", e)
    target_plot._layers = []
    target_plot._pending_layer_frames = None


def drop_layers_for_source(session, source_plot) -> None:
    """Drop every layer (on ANY target plot) whose source is ``source_plot`` — called
    when the source's window/tree is closing so no target keeps a dangling source ref
    or a stale composited image. Re-emits ``layers_state`` for each affected target."""
    if session is None:
        return
    for target in list(getattr(session, "_plots", []) or []):
        layers = getattr(target, "_layers", None)
        if not layers:
            continue
        keep = []
        removed = False
        for layer in layers:
            if layer.source_plot is source_plot:
                try:
                    if layer.handle is not None:
                        layer.handle.remove()
                except Exception as e:
                    log.debug("removing source-closed layer failed: %s", e)
                removed = True
            else:
                keep.append(layer)
        if removed:
            target._layers = keep
            _emit_layers_state(target)


# ── layers_state emission ──────────────────────────────────────────────────────


def _emit_layers_state(target_plot) -> None:
    """Emit the authoritative ``layers_state`` for one target plot. The renderer's
    Plot-Control "Layers" section is written against this EXACT shape."""
    wid = getattr(target_plot, "window_id", None)
    if wid is None:
        return
    layers = getattr(target_plot, "_layers", None) or []
    ipc.emit({
        "type": "layers_state",
        "window_id": int(wid),
        "layers": [ly.to_state() for ly in layers],
    })


def _find_layer(target_plot, layer_id):
    for ly in getattr(target_plot, "_layers", None) or []:
        if ly.layer_id == layer_id:
            return ly
    return None


def _source_title(source_plot) -> str:
    try:
        sig = source_plot.plot_state.current_signal
        t = str(sig.metadata.get_item("General.title", default="") or "")
        if t:
            return t
    except Exception:
        pass
    lbl = getattr(source_plot, "view_label", None)
    return str(lbl or "Layer")


# ── handlers ───────────────────────────────────────────────────────────────────


def overlay_add(session, plot, payload) -> None:
    """Add ``source_window_id``'s current image as a live layer over the target
    ``window_id`` plot. Validates: identical signal-frame ``(H, W)``; refuses a
    tiled target; refuses source IS target. Seeds the layer with the source's CURRENT
    frame and emits ``layers_state``."""
    if plot is None:
        ipc.emit_error("overlay_add: target window not found.")
        return
    src_wid = payload.get("source_window_id")
    source = session._plot_by_window_id(int(src_wid)) if src_wid is not None else None
    if source is None:
        ipc.emit_error("overlay_add: source window not found.")
        return
    if source is plot:
        ipc.emit_status("Cannot layer a window onto itself.")
        return

    p2 = getattr(plot, "_plot2d", None)
    if p2 is None:
        ipc.emit_status("Overlay: target is not an image plot.")
        return
    # Refuse tile mode — a tiled plot cannot composite independent layers.
    if getattr(p2, "_tile_on", False):
        ipc.emit_status(
            "Overlay unavailable: the target image is in tile mode (large frame). "
            "Layers are only supported on non-tiled images.")
        return

    base = getattr(plot, "current_data", None)
    src_frame = getattr(source, "current_data", None)
    if not isinstance(base, np.ndarray) or base.ndim != 2:
        ipc.emit_status("Overlay: target has no 2-D image to layer onto.")
        return
    if not isinstance(src_frame, np.ndarray) or src_frame.ndim != 2:
        ipc.emit_status("Overlay: source has no 2-D image to layer.")
        return
    if src_frame.shape != base.shape:
        ipc.emit_status(
            f"Overlay refused: frame shapes differ "
            f"({src_frame.shape} vs {base.shape}).")
        return

    if not hasattr(plot, "_layers") or plot._layers is None:
        plot._layers = []
    cmap = _LAYER_CMAP_CYCLE[len(plot._layers) % len(_LAYER_CMAP_CYCLE)]
    try:
        handle = p2.add_layer(np.asarray(src_frame), cmap=cmap, alpha=0.5)
    except RuntimeError as e:
        # add_layer raises in tile mode (already guarded) — belt & suspenders.
        ipc.emit_status(f"Overlay unavailable: {e}")
        return
    except Exception as e:
        ipc.emit_error(f"overlay_add failed: {e}")
        return

    layer = PlotLayer(
        layer_id=handle.id, source_plot=source, cmap=cmap, alpha=0.5,
        clim=None, visible=True, handle=handle, title=_source_title(source))
    plot._layers.append(layer)
    _emit_layers_state(plot)


def overlay_set(session, plot, payload) -> None:
    """Update a layer's appearance (any subset of cmap / alpha / clim / visible)."""
    if plot is None:
        return
    layer = _find_layer(plot, payload.get("layer_id"))
    if layer is None or layer.handle is None:
        return
    kw = {}
    if "cmap" in payload and payload["cmap"]:
        layer.cmap = str(payload["cmap"])
        kw["cmap"] = layer.cmap
    if "alpha" in payload and payload["alpha"] is not None:
        try:
            layer.alpha = float(payload["alpha"])
            kw["alpha"] = layer.alpha
        except (TypeError, ValueError):
            pass
    if "visible" in payload and payload["visible"] is not None:
        layer.visible = bool(payload["visible"])
        kw["visible"] = layer.visible
    if "clim" in payload:
        clim = payload["clim"]
        if clim is None:
            layer.clim = None
            kw["clim"] = None
        else:
            try:
                layer.clim = [float(clim[0]), float(clim[1])]
                kw["clim"] = (layer.clim[0], layer.clim[1])
            except (TypeError, ValueError, IndexError):
                pass
    if kw:
        try:
            layer.handle.set(**kw)
        except Exception as e:
            log.debug("overlay_set layer.set failed: %s", e)
    _emit_layers_state(plot)


def overlay_remove(session, plot, payload) -> None:
    """Remove one layer from the target plot."""
    if plot is None:
        return
    layer = _find_layer(plot, payload.get("layer_id"))
    if layer is None:
        return
    try:
        if layer.handle is not None:
            layer.handle.remove()
    except Exception as e:
        log.debug("overlay_remove failed: %s", e)
    plot._layers = [ly for ly in (plot._layers or []) if ly is not layer]
    _emit_layers_state(plot)


def overlay_query(session, plot, payload) -> None:
    """Re-emit the target plot's ``layers_state`` (renderer refresh / reconnect)."""
    if plot is None:
        return
    _emit_layers_state(plot)

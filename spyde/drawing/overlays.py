"""Evaluating a plot's overlay children at the navigator's position.

One call from the navigator dispatcher, right after the base frame is enqueued.
Every overlay of the displayed node is evaluated at the same prepared index the
base frame used, through the same readers, and handed to the painter thread.

A cheap overlay is evaluated inline on the dispatcher, like the base frame. An
expensive one is submitted as a single future on the session's compute backend,
superseded by identity, and painted from its done callback. Neither path holds
a lock or starts a thread: the dispatcher is already serial, and the painter is
already newest-wins.
"""
from __future__ import annotations

import logging

import numpy as np

from spyde.array_cache import reader_for_overlay

log = logging.getLogger(__name__)


def refresh_overlays(plot, indices, settle: bool = False,
                     integrating: bool = False) -> None:
    """Evaluate and draw every visible overlay child of the node ``plot``
    displays, at the selector position ``indices``.

    Runs on the navigator dispatcher thread. Costs nothing when the displayed
    node has no overlay children, which is the common case. ``settle`` is the
    resting-position re-fire flag the base read takes; an overlay treats it as
    an ordinary evaluation. ``integrating`` is the selector's own mode, which
    a node declaring ``follows_region`` reads with so its source integrates
    the same positions the base frame does.
    """
    state = getattr(plot, "plot_state", None)
    signal = getattr(state, "current_signal", None) if state is not None else None
    tree = getattr(plot, "signal_tree", None)
    if signal is None or tree is None:
        return
    children = tree.overlay_children(signal)
    if not children:
        return

    from spyde.drawing.update_functions import _prepare_nav_indices

    # Overlays follow one position: the crosshair, or the centre of a region.
    prepared = _prepare_nav_indices(signal, indices, integrating=False)
    if prepared is None:
        return
    index = tuple(int(v) for v in np.atleast_1d(np.asarray(prepared)).ravel())
    region = None
    if integrating:
        region = _prepare_nav_indices(signal, indices, integrating=True)

    for node in children:
        if not node.visible:
            continue
        # An overlay put on one window (an image layer) draws there only; one
        # that belongs to the node draws on every window showing it.
        target = getattr(node.signal, "target_plot", None)
        if target is not None and target is not plot:
            continue
        at = region if (node.follows_region and region is not None) else index
        try:
            reader = reader_for_overlay(plot, node)
            if _runs_off_the_dispatcher(plot, node, at):
                _submit_overlay(plot, tree, node, reader, at)
            else:
                plot.enqueue_overlay(node, reader.read_frame(at))
        except Exception as e:
            log.debug("overlay %s did not evaluate at %s: %s", node.name, at, e)


def _runs_off_the_dispatcher(plot, node, index) -> bool:
    """Whether ``node``'s function runs on the overlay lane rather than inline.

    A node that names its own tier is taken at its word. One that leaves it
    open takes the tier of the frame it reads, so a source frame already
    decoded is drawn in the same pass as the base and a cold one arrives from
    the callback."""
    if node.expensive is not None:
        return bool(node.expensive)

    from spyde.drawing.update_functions import _classify_nav_read

    source_plot = getattr(node.signal, "source_plot", None) or plot
    state = getattr(source_plot, "plot_state", None)
    source = getattr(state, "current_signal", None)
    data = getattr(source, "data", None)
    if source is None or data is None:
        return False
    navigation_dimension = int(source.axes_manager.navigation_dimension)
    frame_shape = data.shape[navigation_dimension:]
    frame_bytes = int(np.prod(frame_shape)) * data.dtype.itemsize
    return _classify_nav_read(source, index, data, frame_bytes,
                              child=source_plot) == "expensive"


def refresh_overlays_for(tree) -> None:
    """Re-evaluate the tree's overlays at the navigator's current position.

    A forced navigator update, the same call a node switch makes: no overlay
    remembers its own last index, so a parameter or visibility change is
    redrawn by re-running the position the navigator is already on.
    """
    manager = getattr(tree, "navigator_plot_manager", None)
    if manager is None:
        return
    for selectors in getattr(manager, "navigation_selectors", {}).values():
        for selector in selectors:
            try:
                selector.delayed_update_data(force=True)
            except Exception as e:
                log.debug("re-slicing the navigator to redraw overlays "
                          "failed: %s", e)


def _submit_overlay(plot, tree, node, reader, index) -> None:
    """Run an expensive overlay's function off the dispatcher as one future.

    The previous future for this node is cancelled first; a queued one cancels
    cleanly, and one already running is discarded when it lands because the
    callback finds a newer future in the node's slot. All of the bookkeeping
    happens on the one dispatcher thread, so identity is the whole mechanism.
    """
    session = getattr(tree, "session", None)
    backend = getattr(session, "compute_backend", None) if session is not None else None
    if backend is None:
        log.debug("overlay %s needs a compute backend and there is none", node.name)
        return

    futures = plot._overlay_futures
    previous = futures.get(id(node))
    if previous is not None:
        try:
            previous.cancel()
        except Exception:
            pass

    future = backend.submit_overlay(lambda: reader.read_frame(index))
    futures[id(node)] = future

    def paint_when_done(finished, expected=future):
        if futures.get(id(node)) is not expected:
            return
        try:
            value = finished.result()
        except Exception as e:
            log.debug("overlay %s did not evaluate: %s", node.name, e)
            if futures.get(id(node)) is expected:
                del futures[id(node)]
            return
        if futures.get(id(node)) is not expected:
            return
        del futures[id(node)]
        plot.enqueue_overlay(node, value)

    try:
        future.add_done_callback(paint_when_done)
    except Exception as e:
        log.debug("overlay %s could not be armed: %s", node.name, e)
        futures.pop(id(node), None)

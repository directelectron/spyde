"""
particle_tree.py — segmentation spawns a NEW SignalTree. Plan §0.6.

A segmentation is not a property of the movie it came from; it is a derived
dataset computed *from* it, exactly like a strain map or an orientation map. So
this module builds a **particle tree** rather than hanging a ``particles``
attribute off the source.

    ParticleTree
      root signal : lazy LABEL MOVIE — same nav/signal shape as the source,
                    each frame painted from stored contours on demand
      tree.particles   : SpyDEParticles          (the CSR store)
      tree.source_node : the signal it was computed from
      tree.nav_map     : source nav indices → particle frame index
      navigator        : count(t) / mean size(t) / event lanes

Three things this buys, none of which the attribute form does:

* **It answers Wave D by construction.** Particles found on a 4D-STEM virtual
  image record the node they came from and which parent nav positions each one
  covers, so "the mean diffraction pattern for this particle" is a slice of the
  source's parent rather than a guess about which grid the coordinates belong to.
* **The label movie is a dataset**, so scrubbing, saving, the report builder and
  the movie editor all work on it with no special case.
* **Re-segmenting does not destroy the previous result** — two parameter choices
  are two sibling trees you can compare, which is what the signal tree is for.

The label movie is **lazy and never materialised**. A 4096² int32 label image is
64 MB *per frame*; the contours are the truth and ``render_frame`` paints one
frame when something asks for it. See ``spyde/signals/particles.py``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: ``_signal_type`` carried by a particle tree's root. Toolbar entries gate on
#: this to offer particle actions (track, export, per-particle DP) on the result
#: and nowhere else.
PARTICLE_SIGNAL_TYPE = "particles"


def _label_movie(particles, *, chunk_frames: int = 1):
    """A lazy ``(n_frames, h, w)`` int32 label movie backed by the contours.

    One frame per chunk, deliberately: each nav move then reads exactly the frame
    it needs, which is both the cheapest possible read and the same access
    granularity a real in-situ movie has (CLAUDE.md Live-Display §1). Painting is
    done inside the dask graph, so no frame exists until something asks for it.
    """
    import dask
    import dask.array as da

    h, w = particles.frame_shape
    n = particles.n_frames

    def _one(block_info=None):
        # block_info tells us which frame this block is; there is exactly one
        # frame per block, so the slice start IS the frame index.
        t = 0 if block_info is None else int(block_info[None]["array-location"][0][0])
        try:
            return particles.render_frame(t, value="track")[None, ...]
        except Exception as exc:                       # pragma: no cover
            log.debug("[particles] frame %d render failed: %s", t, exc)
            return np.zeros((1, h, w), np.int32)

    with dask.config.set(scheduler="threads"):
        return da.map_blocks(
            _one, dtype=np.int32,
            chunks=((chunk_frames,) * n, (h,), (w,)),
            meta=np.zeros((0, 0, 0), np.int32),
        )


def _navigator_traces(particles, events=None) -> dict[str, np.ndarray]:
    """The three navigator lanes, as plain arrays.

    ``count`` is integer data and the renderer must draw it as a STEP — a straight
    interpolation between frames puts a nucleation's visual transition half a frame
    early, so an event at frame 8 reads as 7 (plan C3).
    """
    traces: dict[str, np.ndarray] = {
        "count": particles.count_series(),
        "size": particles.property_series("area", "mean"),
    }
    if events is not None:
        from spyde.particles.track import event_counts
        ec = event_counts(events, particles.n_frames)
        for kind, arr in ec.items():
            traces[f"event_{kind}"] = np.asarray(arr, np.float32)
    return traces


def open_particle_tree(session, *, particles, source_node, source_tree=None,
                       title: str | None = None, events=None,
                       nav_map=None, params: dict[str, Any] | None = None,
                       provenance: dict[str, Any] | None = None):
    """Create the particle tree for a finished segmentation.

    Parameters
    ----------
    particles
        The :class:`~spyde.signals.particles.SpyDEParticles` store.
    source_node
        The signal segmentation ran on. Recorded on the tree so Wave D can walk
        back to its parent; **not** used to hold the result.
    source_tree
        The tree *source_node* belongs to, when the caller knows it. Recorded for
        provenance; the particle tree is a sibling, not a child.
    events
        Optional event list from :func:`spyde.particles.track.link`. When given,
        the navigator gains an event lane.
    nav_map
        Optional ``(n_frames,)`` int array mapping each particle frame back to a
        source navigation index. ``None`` means identity, which is the case for a
        movie; a 4D-STEM virtual image passes the parent's grid.

    Returns
    -------
    The new tree, with ``particles``, ``source_node``, ``nav_map`` and
    ``nav_traces`` attached.
    """
    import hyperspy.api as hs

    from spyde.actions.commit import open_result_tree

    n = particles.n_frames
    h, w = particles.frame_shape
    title = title or f"Particles — {particles.n_particles} in {n} frames"

    lazy = _label_movie(particles)
    sig = hs.signals.Signal2D(lazy).as_lazy()
    sig.data = lazy

    # Carry the source's calibration so a particle's centroid means the same
    # thing on this tree as it did on the movie. Without it the label movie is
    # in pixels while every measured property is in nm, which is the exact class
    # of unit mismatch the linker's docstring warns about.
    _copy_axes(source_node, sig, particles)

    tree = open_result_tree(
        session, title=title, signal=sig,
        signal_type=PARTICLE_SIGNAL_TYPE,
        provenance=provenance or {"action": "segment_particles",
                                  "params": dict(params or {})},
    )

    tree.particles = particles
    tree.source_node = source_node
    tree.source_tree = source_tree
    tree.nav_map = (np.arange(n, dtype=np.int64) if nav_map is None
                    else np.asarray(nav_map, np.int64))
    tree.particle_events = list(events or ())
    tree.nav_traces = _navigator_traces(particles, events)
    return tree


def _copy_axes(source_node, sig, particles) -> None:
    """Give the label movie the source's calibration, best-effort.

    Best-effort on purpose: a wrong calibration is worse than none, so every step
    is guarded and a failure leaves the axis at its default rather than half-
    applied. The signal axes fall back to the particle store's own ``scale``,
    which is what the measurements were computed with.
    """
    try:
        src_sig = getattr(source_node, "axes_manager", None)
        if src_sig is not None:
            for dst, src in zip(sig.axes_manager.signal_axes,
                                source_node.axes_manager.signal_axes):
                dst.scale, dst.units = float(src.scale), src.units
            nav = source_node.axes_manager.navigation_axes
            if nav:
                dnav = sig.axes_manager.navigation_axes[0]
                dnav.name, dnav.units = nav[0].name, nav[0].units
                dnav.scale, dnav.offset = float(nav[0].scale), float(nav[0].offset)
            return
    except Exception as exc:
        log.debug("[particles] copying source axes failed: %s", exc)
    try:
        for ax in sig.axes_manager.signal_axes:
            ax.scale, ax.units = float(particles.scale), particles.units
    except Exception as exc:                            # pragma: no cover
        log.debug("[particles] fallback axis calibration failed: %s", exc)


def particle_nav_positions(tree, index: int):
    """Source navigation indices covered by global particle *index*.

    This is the Wave D seam: a particle found on a derived node (a 4D-STEM virtual
    image) needs to name positions in the PARENT's navigation grid before anything
    can average diffraction patterns over it.

    Returns an ``(m, k)`` int array of navigation indices, where *k* is the source
    navigation dimensionality. For a movie (1-D nav) that is the single frame the
    particle lives in; for a virtual image it is every scan position under the
    particle's mask.
    """
    particles = tree.particles
    row = particles.flat_buffer[int(index)]
    from spyde.signals.particles import COL

    t = int(row[COL["t"]])
    frame = int(tree.nav_map[t]) if t < len(tree.nav_map) else t

    if not particles.has_masks:
        # No outlines stored: the best we can name is the frame itself.
        return np.asarray([[frame]], np.int64)

    mask, (y0, x0, _y1, _x1) = particles.mask_at(int(index))
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.asarray([[frame]], np.int64)

    nav_dim = _source_nav_dim(tree)
    if nav_dim <= 1:
        # A movie: the particle's pixels are SIGNAL coordinates, not navigation
        # ones, so the only navigation index involved is the frame.
        return np.asarray([[frame]], np.int64)
    # A virtual image: the particle's pixels ARE navigation positions.
    return np.stack([ys + y0, xs + x0], axis=-1).astype(np.int64)


def _source_nav_dim(tree) -> int:
    src = getattr(tree, "source_node", None)
    try:
        return int(src.axes_manager.navigation_dimension)
    except Exception:
        return 1

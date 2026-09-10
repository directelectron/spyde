"""Reader kind 6: a mapped node's frame, from the mapped function on the
PARENT's frame, with no dask in the loop.

A node made by a hyperspy ``map`` (``center_direct_beam``, an azimuthal
integration, a per-pattern filter) carries a :class:`FrameRecipe` on its
signal, recorded by ``spyde.external.hyperspy.map_recipe``. Frame *i* of such a
node is the mapped function applied to frame *i* of its source, which is
exactly what hyperspy computes inside every block. Asking dask for that one
frame instead computes the WHOLE enclosing block. Measured on a real 5-D .zspy
(64x64 navigation chunks of 128^2 uint16), constant-shift centring:

    first frame in a chunk through the dask block    2196 ms
    the same frame through the recipe                0.55 ms   (bit-identical)

This reader is the consumer. It checks that a node's recipe chain is rooted at
the node's TREE parent, then evaluates the chain per frame on top of the
parent's own reader, so the parent's block cache does the decoding and the
function is the only cost.

Correctness gate: the chain must reach the tree parent through recipes alone.
A method that mapped an intermediate the tree never saw, or a lazy
per-position argument with no recipe of its own, declines to the block path.
A wrong frame is worse than a slow one; the parity tests
(``test_recipe_reader.py``) are the proof that what is served equals the block.

Threading: ``read_frame`` runs on the navigator dispatcher thread and on the
overlay layer's warm thread. The reader holds no mutable state, and the mapped
function already has to be pure for hyperspy to run it on worker threads.
"""
from __future__ import annotations

import itertools

import numpy as np

from spyde.external.hyperspy.map_recipe import FrameRecipe, recipe_for


def chain_reaches(signal, parent_signal) -> bool:
    """True when ``signal``'s recipe, and every recipe it depends on, is rooted
    at ``parent_signal``: one frame of ``signal`` is a function of one frame of
    ``parent_signal`` and nothing else."""
    recipe = recipe_for(signal)
    return recipe is not None and _rooted_at(recipe, parent_signal, set())


def _rooted_at(recipe: FrameRecipe, parent_signal, seen: set) -> bool:
    if id(recipe) in seen:
        return False
    seen.add(id(recipe))
    for value in recipe.iterating.values():
        if isinstance(value, np.ndarray) or _per_position_source(value) is not None:
            continue
        nested = recipe_for(value)
        if nested is None or not _rooted_at(nested, parent_signal, seen):
            return False
    if recipe.source is None:
        # The function reads no frame, so there is no chain to root.
        return True
    if recipe.source is parent_signal:
        return True
    nested = recipe_for(recipe.source)
    return nested is not None and _rooted_at(nested, parent_signal, seen)


def _per_position_source(value):
    """``value``'s ``at(*navigation_index)`` when it has one, else None. A
    store that answers one position at a time is an iterating argument in its
    own right; it needs no recipe because it already holds the values."""
    at = getattr(value, "at", None)
    return at if callable(at) else None


def _missing_parent_frame(indices):
    raise ValueError("this recipe reads a source frame, but no parent reader "
                     "was resolved to supply one")


def _source_frame(recipe: FrameRecipe, indices, parent_signal, parent_frame):
    """One frame of ``recipe``'s source at ``indices``."""
    if recipe.source is parent_signal:
        return parent_frame(indices)
    return evaluate(recipe_for(recipe.source), indices, parent_signal, parent_frame)


def _navigation_sizes(signal, count: int) -> tuple[int, ...]:
    """The first ``count`` navigation axis sizes of ``signal``, in data order."""
    shape = getattr(getattr(signal, "data", None), "shape", None)
    if shape is not None and len(shape) >= count:
        return tuple(int(n) for n in shape[:count])
    navigation_shape = signal.axes_manager.navigation_shape
    return tuple(int(n) for n in reversed(navigation_shape))[:count]


def navigation_depths(depth, count: int) -> tuple[int, ...]:
    """One neighbourhood radius per navigation axis, from an int (the same
    radius everywhere) or a per-axis tuple, outermost first."""
    if isinstance(depth, (tuple, list)):
        radii = tuple(int(v) for v in depth)
        return radii + (0,) * (count - len(radii))
    return (int(depth),) * count


def _source_window(recipe: FrameRecipe, indices, depths, parent_signal,
                   parent_frame):
    """The stack of source frames over ``[index - depth, index + depth]`` on
    each navigation axis, clipped to the navigation grid, and the requested
    position's index inside that stack."""
    sizes = _navigation_sizes(recipe.source, len(indices))
    spans, centre = [], []
    for axis, position in enumerate(indices):
        low = max(0, position - depths[axis])
        high = min(sizes[axis] - 1, position + depths[axis])
        spans.append(range(low, high + 1))
        centre.append(position - low)
    frames = []
    for point in itertools.product(*spans):
        frame = _source_frame(recipe, point, parent_signal, parent_frame)
        if frame is None:
            raise ValueError(f"no source frame at {point} for the window "
                             f"around {tuple(indices)}")
        frames.append(frame)
    window_shape = tuple(len(span) for span in spans)
    window = np.stack(frames).reshape(window_shape + np.shape(frames[0]))
    return window, tuple(centre)


def evaluate(recipe: FrameRecipe, indices, parent_signal, parent_frame):
    """The value of ``recipe`` at ``indices``, reading the chain's root frame
    with ``parent_frame(indices)``. Mirrors ``process_function_blockwise``
    step for step: per-position arguments are squeezed and a 0-d one becomes a
    scalar; the result is written into a freshly allocated frame of the
    recorded dtype, so a float result lands in an integer frame by the same
    truncating assignment the block path uses.

    The recipe chooses one of three calls: no source frame at all (``source``
    is None), a navigation neighbourhood (``depth`` above zero, which passes
    the window and the centre index ahead of the arguments), or one frame.
    With no recorded ``output_shape`` there is no frame to cast into: a ragged
    map output comes back as an array of per-position values, and a recipe
    that describes no dask array at all comes back exactly as the function
    returned it, None included."""
    if parent_frame is None:
        parent_frame = _missing_parent_frame

    per_position = {}
    for key, value in recipe.iterating.items():
        at = _per_position_source(value)
        if at is not None:
            per_position[key] = at(*indices)
            continue
        if isinstance(value, np.ndarray):
            argument = value[indices]
        else:
            argument = evaluate(recipe_for(value), indices, parent_signal, parent_frame)
        argument = np.squeeze(argument)
        per_position[key] = argument[()] if argument.shape == () else argument

    # An array of points is an integrating region, which the source reduces in
    # one call; there is no single position to take a neighbourhood around.
    region = isinstance(indices, np.ndarray) and indices.ndim > 1
    depths = (0,) if region else navigation_depths(recipe.depth, len(indices))
    if recipe.source is None:
        result = recipe.function(**per_position, **recipe.static)
    elif any(depths):
        window, centre = _source_window(recipe, indices, depths, parent_signal,
                                        parent_frame)
        result = recipe.function(window, centre, **per_position, **recipe.static)
    else:
        frame = _source_frame(recipe, indices, parent_signal, parent_frame)
        result = recipe.function(frame, **per_position, **recipe.static)

    if recipe.output_shape is None:
        if recipe.output_name is None:
            # A display recipe: the value is whatever the function returns,
            # which for an overlay is a dict of one value per group.
            return result
        # A ragged map output is still an array of per-position values, just
        # one with no frame shape to cast it into.
        return None if result is None else np.asarray(result)
    result = np.asarray(result)
    out = np.empty(recipe.output_shape, recipe.output_dtype)
    out[...] = result.reshape(out.shape)
    return out


class RecipeReader:
    """Frames of a mapped node, evaluated from the parent's frames.

    Holds no cache of its own: the parent's reader (and the plot's BlockCache
    behind it) decodes, and output frames land in the plot's ArrayCache like
    any other. No ``sum_points``: a mapped function is not linear in general,
    so a region is the per-frame loop the cache already provides."""

    def __init__(self, signal, data, parent_signal, parent_reader):
        self.signal = signal
        self.data = data
        self.recipe = recipe_for(signal)
        self._parent_signal = parent_signal
        self._parent_reader = parent_reader
        self._nav_ndim = signal.axes_manager.navigation_dimension

    @property
    def frame_bytes(self) -> int:
        if self.data is None:
            # A display recipe has no array behind it, so nothing to budget.
            return 0
        frame_shape = self.data.shape[self._nav_ndim:]
        return int(np.prod(frame_shape)) * self.data.dtype.itemsize

    def is_chunk_resident(self, indices) -> bool:
        """Cheap iff the PARENT's block is resident; the function is sub-ms."""
        probe = getattr(self._parent_reader, "is_chunk_resident", None)
        return bool(probe(indices)) if probe is not None else False

    def read_frame(self, indices):
        """The recipe's value at ``indices``, or None when the function
        returned None (no value at this position).

        An array of points instead of one position is an integrating region,
        which reaches the source read whole: the source integrates it exactly
        as it does for a base frame, and the function is applied to the
        result."""
        index = np.asarray(indices)
        point = (index if index.ndim > 1
                 else tuple(int(v) for v in index[:self._nav_ndim]))
        parent_frame = (self._parent_reader.read_frame
                        if self._parent_reader is not None else None)
        return evaluate(self.recipe, point, self._parent_signal, parent_frame)

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
        if isinstance(value, np.ndarray):
            continue
        nested = recipe_for(value)
        if nested is None or not _rooted_at(nested, parent_signal, seen):
            return False
    if recipe.source is parent_signal:
        return True
    nested = recipe_for(recipe.source)
    return nested is not None and _rooted_at(nested, parent_signal, seen)


def evaluate(recipe: FrameRecipe, indices, parent_signal, parent_frame) -> np.ndarray:
    """One frame of ``recipe`` at ``indices``, reading the chain's root frame
    with ``parent_frame(indices)``. Mirrors ``process_function_blockwise``
    step for step: per-position arguments are squeezed and a 0-d one becomes a
    scalar; the result is written into a freshly allocated frame of the
    recorded dtype, so a float result lands in an integer frame by the same
    truncating assignment the block path uses."""
    source = recipe.source
    if source is parent_signal:
        frame = parent_frame(indices)
    else:
        frame = evaluate(recipe_for(source), indices, parent_signal, parent_frame)

    per_position = {}
    for key, value in recipe.iterating.items():
        if isinstance(value, np.ndarray):
            argument = value[indices]
        else:
            argument = evaluate(recipe_for(value), indices, parent_signal, parent_frame)
        argument = np.squeeze(argument)
        per_position[key] = argument[()] if argument.shape == () else argument

    result = np.asarray(recipe.function(frame, **per_position, **recipe.static))
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
        frame_shape = self.data.shape[self._nav_ndim:]
        return int(np.prod(frame_shape)) * self.data.dtype.itemsize

    def is_chunk_resident(self, indices) -> bool:
        """Cheap iff the PARENT's block is resident; the function is sub-ms."""
        probe = getattr(self._parent_reader, "is_chunk_resident", None)
        return bool(probe(indices)) if probe is not None else False

    def read_frame(self, indices: tuple[int, ...]) -> np.ndarray:
        point = tuple(int(v) for v in indices[:self._nav_ndim])
        return evaluate(self.recipe, point, self._parent_signal,
                        self._parent_reader.read_frame)

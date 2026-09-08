"""
Patch: a lazy ``BaseSignal.map`` output remembers HOW it was made, so ONE frame
of it can be evaluated without computing its dask block.

WHAT
----
Wraps ``hyperspy.signal.BaseSignal.map``. A call that returns a NEW lazy signal
gets ``result._map_recipe = FrameRecipe(...)``: the mapped function, its
constant keyword arguments, its per-position keyword arguments (the signals
hyperspy iterates alongside the data), the signal the map was called on, and
the output frame shape and dtype. An in-place map (hyperspy's default) removes
any recipe the signal carried, because its data no longer matches it.

The wrapper observes only. The call, the graph it builds and the signal it
returns are unchanged, so nothing that computes through dask can see a
difference.

WHY
---
``map`` builds ``da.blockwise(process_function_blockwise, ...)``, and that block
function is a loop (hyperspy/misc/utils.py)::

    for index in np.ndindex(chunk_nav_shape):
        output_array[index] = function(data[index], **per_position[index], **constants)

Frame *i* of the output is a closed expression over frame *i* of the source.
Dask's unit of work is the block, though, so ``output.data[i]`` decodes the
enclosing source chunk and runs the function over every frame in it to hand
back one. Measured on a real 5-D .zspy (64x64 navigation chunks of 128^2
uint16) with a constant-shift ``center_direct_beam`` node: 2196 ms per chunk
crossing through the block, 0.55 ms through the recipe, bit-identical.

The navigator read path is the only consumer
(``spyde.array_cache.readers.recipe``). Navigator sums, saves and find-vectors
keep using the dask graph.

WHEN TO REMOVE
--------------
When hyperspy records this itself. The natural home is ``_map_iterate`` right
after the blockwise array is built, plus a ``LazySignal`` method that evaluates
one navigation position through the chain of recipes down to the store. This
module is the bridge and the proposal; no upstream issue yet.
"""
from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

log = logging.getLogger(__name__)

RECIPE_ATTRIBUTE = "_map_recipe"


@dataclass(frozen=True)
class FrameRecipe:
    """How to build ONE frame of a mapped signal from ONE frame of its source::

        frame(i) = function(source_frame(i), **{key: iterating[key] at i}, **static)

    written into ``np.empty(output_shape, output_dtype)``, which is the same
    call and the same cast hyperspy's ``process_function_blockwise`` makes at
    every position of a block.

    ``iterating`` values are navigation-shaped numpy arrays, snapshotted BEFORE
    the map call because ``map`` rebinds an eager argument signal to a lazy
    copy, or lazy signals, which may carry a recipe of their own.
    ``output_name`` is the dask name of the array this recipe describes; a
    signal whose data was swapped afterwards no longer matches it
    (:func:`recipe_for`).
    """
    function: Callable
    static: Mapping[str, Any]
    iterating: Mapping[str, Any]
    source: Any
    output_name: str
    output_shape: tuple[int, ...]
    output_dtype: np.dtype


def recipe_for(signal) -> FrameRecipe | None:
    """The recipe describing ``signal``'s CURRENT data, or None."""
    recipe = getattr(signal, RECIPE_ATTRIBUTE, None)
    if recipe is None:
        return None
    data = getattr(signal, "data", None)
    if getattr(data, "name", None) != recipe.output_name:
        return None
    return recipe


def _split_kwargs(signal, kwargs):
    """The split hyperspy's ``map`` makes of the function's keyword arguments:
    a Signal with the mapped signal's navigation shape iterates per position; a
    Signal with navigation shape () or (1,) is squeezed into a constant;
    everything else is a constant."""
    from hyperspy.signal import BaseSignal

    navigation_shape = signal.axes_manager.navigation_shape
    iterating, static = {}, {}
    for key, value in kwargs.items():
        if isinstance(value, BaseSignal):
            shape = value.axes_manager.navigation_shape
            if shape == navigation_shape:
                iterating[key] = value if value._lazy else np.asarray(value.data)
                continue
            if shape in ((), (1,)):
                static[key] = np.squeeze(value.data)
                continue
        static[key] = value
    return iterating, static


def _describable(result, ragged) -> bool:
    """A new LAZY signal with signal axes is the only output a frame recipe
    describes. Eager output is already indexable; a ragged or scalar-per-
    position output has no frame."""
    return (result is not None
            and bool(getattr(result, "_lazy", False))
            and not ragged
            and result.axes_manager.signal_dimension > 0
            and isinstance(getattr(result.data, "name", None), str))


def apply() -> bool:
    """Idempotently wrap ``BaseSignal.map`` so each lazy output carries its
    :class:`FrameRecipe`. Returns True if the wrapper is in place, False if the
    upstream shape changed and it was skipped."""
    try:
        from hyperspy.signal import BaseSignal
    except Exception as e:  # upstream moved or renamed the class
        log.warning("spyde.external.hyperspy: BaseSignal import failed, map "
                    "outputs will not carry frame recipes: %s", e)
        return False

    original = BaseSignal.map
    if getattr(original, "_spyde_map_recipe", False):
        return True
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError) as e:
        log.warning("spyde.external.hyperspy: cannot read BaseSignal.map's "
                    "signature; skipping the frame-recipe wrapper: %s", e)
        return False
    if "inplace" not in signature.parameters or "kwargs" not in signature.parameters:
        log.warning("spyde.external.hyperspy: BaseSignal.map no longer takes "
                    "inplace/**kwargs; skipping the frame-recipe wrapper")
        return False

    @functools.wraps(original)
    def map_with_recipe(self, function, *args, **kwargs):
        try:
            bound = signature.bind(self, function, *args, **kwargs)
        except TypeError:
            return original(self, function, *args, **kwargs)  # hyperspy raises its own error
        inplace = bound.arguments.get("inplace", True)
        ragged = bound.arguments.get("ragged")
        iterating, static = _split_kwargs(self, bound.arguments.get("kwargs", {}))
        result = original(self, function, *args, **kwargs)
        if inplace:
            # This signal's data was replaced; a recipe for the old data is wrong.
            if hasattr(self, RECIPE_ATTRIBUTE):
                delattr(self, RECIPE_ATTRIBUTE)
            return result
        if _describable(result, ragged):
            navigation_dimension = result.axes_manager.navigation_dimension
            setattr(result, RECIPE_ATTRIBUTE, FrameRecipe(
                function=function, static=static, iterating=iterating, source=self,
                output_name=result.data.name,
                output_shape=tuple(int(n) for n in result.data.shape[navigation_dimension:]),
                output_dtype=np.dtype(result.data.dtype),
            ))
        return result

    map_with_recipe._spyde_map_recipe = True
    map_with_recipe._spyde_original = original
    BaseSignal.map = map_with_recipe
    log.debug("spyde.external.hyperspy: BaseSignal.map outputs now carry frame recipes")
    return True

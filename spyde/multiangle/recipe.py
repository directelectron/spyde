"""
recipe.py — what a composed multi-angle node needs to rebuild any one frame.

A composed node's frames are a function of its MEMBERS' frames: summing them, or
picking one. The dask array built by :mod:`spyde.multiangle.compose` says so
too, but asking dask for one frame of it makes dask do a chunk's worth of work
(``spyde/array_cache/readers/recipe.py`` measures 2196 ms for the equivalent
question on a mapped node). So a composed node also carries this recipe, and the
display reads through it instead: N member frames, each from that member's own
reader with its own block cache, added.

That is the same trade the rest of ``spyde/array_cache`` makes — read at the
granularity the ACCESS needs, not the granularity the storage happens to use.
Here it is a strong version of it, because a composed frame's members live in N
separate files and dask would otherwise pull a nav-chunk out of every one of
them to answer for a single scan position.

This module is pure data: it records the recipe and looks it up. Evaluating one
is :mod:`spyde.array_cache.readers.multiangle`, which lives over there because
it needs the reader machinery and the plot's shared block cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Attribute the recipe is recorded under. A private attribute on the signal,
#: matching how a hyperspy `map` output carries its own per-frame recipe.
_RECIPE_ATTRIBUTE = "_multiangle_recipe"


@dataclass(frozen=True)
class MultiAngleRecipe:
    """How to rebuild one frame of a composed node from its members.

    Parameters
    ----------
    members
        The member signals, in member order — the same order the model's offsets
        are in. Held so each can resolve its OWN reader, which is what makes a
        composed read cost one member-frame read per member rather than a dask
        graph evaluation.
    model
        The :class:`~spyde.multiangle.model.MultiAngleModel` whose offsets say
        where each member's copy of a position lives.
    has_angle_axis
        True for the 5-D stack node, whose indices lead with an angle; False for
        the summed node, whose frame is every member added together.
    dtype
        The accumulator the sum is taken in. Ignored when *has_angle_axis*,
        where a frame is one member's own pixels and keeps their type.
    stack
        The 5-D signal the members are planes of, when they are — a node
        rebuilt from a saved acquisition rather than composed from files.
        The reader then reads every member from the stack's ONE store reader
        instead of through a lazy slice per member, which is the difference
        between decoding a chunk and asking dask to (measured 25 ms against
        55-80 ms per member on a real stack).
    """

    members: tuple
    model: Any
    has_angle_axis: bool
    dtype: Any = None
    member_indices: tuple | None = None
    stack: Any = None

    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def indices(self) -> tuple:
        """Each held member's index IN THE MODEL.

        Not the same as its position in :attr:`members` whenever the node holds
        a SUBSET — a per-shell sum holds members 3 and 4, and their offsets are
        the model's entries 3 and 4, not 0 and 1. Reading the offsets by
        enumeration instead serves a frame from the wrong scan position, which
        is invisible because the frame is real, just not the one asked for.
        """
        if self.member_indices is not None:
            return tuple(int(index) for index in self.member_indices)
        return tuple(range(len(self.members)))


def attach_recipe(signal, recipe: MultiAngleRecipe):
    """Record *recipe* on *signal* and return the signal.

    Set on the signal object rather than in ``metadata`` deliberately: it holds
    live member signals, which are not serialisable and must not be written into
    a saved file's metadata tree.
    """
    object.__setattr__(signal, _RECIPE_ATTRIBUTE, recipe)
    return signal


def recipe_for(signal) -> MultiAngleRecipe | None:
    """The recipe recorded on *signal*, or None when it is not a composed node."""
    recipe = getattr(signal, _RECIPE_ATTRIBUTE, None)
    return recipe if isinstance(recipe, MultiAngleRecipe) else None

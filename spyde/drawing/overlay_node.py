"""The signal an overlay node holds.

An overlay is a child of the node a window displays whose value at the
navigator's position is drawn on that window: circles on the diffraction
pattern, a band set, a curve on a spectrum, an image layer. It has no array of
its own, so instead of data it carries a :class:`FrameRecipe` with no output
name: the function to call, its static and per-position arguments, and the
navigation neighbourhood it reads.

``axes_manager`` is the parent signal's, which is what makes the navigation
index prepared for the base frame the right index here too.
"""
from __future__ import annotations


class OverlaySignal:
    """A node's display value, evaluated one navigation position at a time.

    ``source_plot`` is the plot whose readers supply the source frame. It is
    None for an overlay that reads the frame of the window it draws on; it is
    the other window's plot for a layer sourced from elsewhere, so that
    window's decoded blocks serve the read instead of a second copy.
    """

    data = None

    def __init__(self, parent_signal, recipe, source_plot=None):
        self.axes_manager = parent_signal.axes_manager
        self._map_recipe = recipe
        self.source_plot = source_plot

    def __repr__(self) -> str:  # pragma: no cover
        return f"OverlaySignal(function={self._map_recipe.function!r})"

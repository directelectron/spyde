"""
RegionMap — the label image(s) a segmentation produced, as a signal type.

The result of Segment is a tree whose signal is the instance label map (one
per field, so a movie becomes a label movie) and whose ``tree.regions`` holds
the measured table. The distinct signal type lets the toolbar keep Segment off
its own output and gate region-only actions on. Registered in
``spyde/hyperspy_extension.yaml``.
"""
from __future__ import annotations

from hyperspy._signals.signal2d import LazySignal2D, Signal2D

SIGNAL_TYPE = "regions"


class RegionMap(Signal2D):
    _signal_type = SIGNAL_TYPE


class LazyRegionMap(LazySignal2D):
    _signal_type = SIGNAL_TYPE

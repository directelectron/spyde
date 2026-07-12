"""
InSitu — a HyperSpy signal type for time-series (in-situ) MOVIES.

An in-situ movie is a dataset with a 1-D navigation (time) axis and 2-D signal
axes — e.g. a DE-MRC / camera sequence of ``n_frames x H x W`` images. It
*displays* like any 2-D-signal navigated dataset, but its navigator is a TIME
axis being scrubbed/played through, not a spatial scan grid. We give it a
distinct signal type so the toolbar gating can turn on movie-only controls
(Play / Fast Forward) for it and keep them off ordinary navigated data (e.g. a
4D-STEM scan's 2-D spatial navigator).

Subclasses HyperSpy's Signal2D so it keeps all the standard 2-D display
behaviour; only the signal_type differs. Registered as a HyperSpy extension
(see spyde/hyperspy_extension.yaml) so `set_signal_type` and save/load work.
"""
from __future__ import annotations

from hyperspy._signals.signal2d import Signal2D, LazySignal2D

SIGNAL_TYPE = "insitu"


class InSitu(Signal2D):
    """Time-series (in-situ) movie: 1-D time navigation, 2-D image signal (eager)."""
    _signal_type = SIGNAL_TYPE


class LazyInSitu(LazySignal2D):
    """Time-series (in-situ) movie: 1-D time navigation, 2-D image signal (lazy)."""
    _signal_type = SIGNAL_TYPE

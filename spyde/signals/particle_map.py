"""
ParticleMap — the signal type a segmentation result carries.

The root of a particle tree (plan §0.6) is a **label movie**: same nav/signal
shape as the movie it was segmented from, each frame painted on demand from the
stored contours, with pixel values carrying track ids. It displays like any
navigated 2-D signal; only the signal type differs.

The type exists so toolbar gating can offer particle actions — track, export,
per-particle diffraction — on a segmentation result and *nowhere else*. That is
the same job ``insitu`` does for Play / Fast-Forward, and it is why the plan puts
the result on its own tree: gating becomes a plain signal-type check instead of a
hunt up the parent chain for someone else's attribute.

Registered as a HyperSpy extension (see ``spyde/hyperspy_extension.yaml``) so
``set_signal_type`` and save/load work.
"""
from __future__ import annotations

from hyperspy._signals.signal2d import LazySignal2D, Signal2D

SIGNAL_TYPE = "particles"


class ParticleMap(Signal2D):
    """Per-frame particle label map (eager)."""
    _signal_type = SIGNAL_TYPE


class LazyParticleMap(LazySignal2D):
    """Per-frame particle label map (lazy) — the normal case.

    Lazy is not an optimisation here, it is the design: a materialised label
    movie is 64 MB *per frame* at 4096², so frames are painted from contours only
    when something asks for one.
    """
    _signal_type = SIGNAL_TYPE

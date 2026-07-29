"""
spyde.particles — particle segmentation, measurement and tracking.

See ``DRIFT_AND_PARTICLES_PLAN.md`` (repo root) for the full design. The shape of
this package is its most important property: **three interchangeable ways to
produce a foreground probability map, and one shared downstream stage.**

    frame ──► [ classical | scribble | prompt ] ──► probability / mask
                                                          │
                                  split_instances()  (watershed)
                                                          │
                                  measure_frame()    (regionprops → units)
                                                          │
                                  SpyDEParticles     (CSR, per frame)
                                                          │
                                  link()             (Hungarian) → tracks

The three engines are not alternatives to choose between — they are three ways to
fill the first box, and they compose (plan §0.4). Everything after the first box is
written once, in :mod:`spyde.particles.classical` (the split) and
:mod:`spyde.particles.measure`.
"""
from __future__ import annotations

from spyde.particles.classical import (
    THRESHOLD_METHODS,
    SegmentParams,
    segment_frame,
    split_instances,
    threshold_mask,
)
from spyde.particles.measure import measure_frame

__all__ = [
    "SegmentParams",
    "THRESHOLD_METHODS",
    "segment_frame",
    "split_instances",
    "threshold_mask",
    "measure_frame",
]

"""
spyde.particles — particle segmentation, measurement and tracking.

See ``DRIFT_AND_PARTICLES_PLAN.md`` (repo root) for the full design. The shape of
this package is its most important property: **interchangeable ways to produce a
foreground probability map, and one shared downstream stage.**

    frame ──► [ scribble | (prompt, B4) ] ──────► probability / mask
                                                          │
                                  split_instances()  (watershed)
                                                          │
                                  measure_frame()    (props + hull → units)
                                                          │
                                  SpyDEParticles     (CSR, per frame)
                                                          │
                                  link()             (Hungarian) → tracks + events

The engines are not alternatives to choose between — they are ways to fill the
first box, and they compose (plan §0.4): a promptable model's masks become
scribble training labels via :func:`~spyde.particles.scribble.masks_to_labels`.
Everything after the first box is written once, in
:mod:`~spyde.particles.instances` (the instance split) and
:mod:`~spyde.particles.measure`.

**A third engine used to fill that box and was DELETED**: a port of
ParticleSpy's classical threshold pipeline. On low-contrast in-situ data a
global threshold has no bimodal histogram to find, so it returned the support
film as thousands of instances and no parameter recovered the real particles —
and because otsu is computed from whatever array it is handed, the caret's
preview crop and the batch's full frame disagreed, so tuning the preview did not
control the run. :mod:`~spyde.particles.instances` records the measurements.

torch is imported **lazily**, inside the functions that need it — importing this
package must never pay for CUDA init. :func:`~spyde.particles.features.gpu_available`
answers the capability question without loading it.

.. note::
   Importing this package currently costs ~6 s, because ``measure.py`` and
   ``track.py`` read the column schema from :mod:`spyde.signals.particles`, and
   importing anything under ``spyde.signals`` executes that package's ``__init__``,
   which pulls in **hyperspy** (5.5 s of the 6.3 s; torch is NOT involved). That is
   free in the running app — the backend loads hyperspy at startup regardless — but
   it does mean a script that only wants image segmentation pays for a signal
   framework it never touches, which is at odds with the "constructible standalone"
   contract in :mod:`spyde.signals`. Fixing it means deferring the ``insitu`` import
   in ``spyde/signals/__init__.py``; left alone deliberately, since that file is
   shared with other waves.
"""
from __future__ import annotations

from spyde.particles.instances import (
    SegmentParams,
    merge_close_instances,
    split_instances,
)
from spyde.particles.features import (
    DEFAULT_RANK_RADII,
    DEFAULT_SIGMAS,
    FeatureSpec,
    PreparedFrame,
    feature_names,
    feature_stack,
    feature_tensor,
    gpu_available,
    prepare_frame,
    sample_features,
    select_device,
)
from spyde.particles.measure import measure_frame
from spyde.particles.scribble import (
    DEFAULT_CLASSES,
    UNLABELLED,
    LabelStore,
    ScribbleClass,
    ScribbleClassifier,
    default_classes,
    masks_to_labels,
    random_forest_reference,
)
from spyde.particles.track import (
    EVENT_KINDS,
    LinkParams,
    LinkResult,
    ParticleEvent,
    event_counts,
    frame_indices,
    link,
    sample_frame_positions,
)

__all__ = [
    # the shared instance split
    "SegmentParams",
    "merge_close_instances",
    "split_instances",
    # measurement
    "measure_frame",
    # feature stack
    "FeatureSpec",
    "PreparedFrame",
    "DEFAULT_SIGMAS",
    "DEFAULT_RANK_RADII",
    "prepare_frame",
    "feature_tensor",
    "feature_stack",
    "feature_names",
    "sample_features",
    "select_device",
    "gpu_available",
    # scribble engine
    "ScribbleClass",
    "ScribbleClassifier",
    "LabelStore",
    "DEFAULT_CLASSES",
    "UNLABELLED",
    "default_classes",
    "masks_to_labels",
    "random_forest_reference",
    # tracking
    "link",
    "LinkParams",
    "LinkResult",
    "ParticleEvent",
    "EVENT_KINDS",
    "event_counts",
    "frame_indices",
    "sample_frame_positions",
]

"""
spyde.segmentation — trainable segmentation of images, movies and scans.

The user paints a few strokes (:class:`Labels`), a linear classifier over a
small feature bank learns them (:class:`PixelClassifier`), and the same
classifier then labels every field: a single image, each frame of an in-situ
movie, or the real-space navigator of a 4-D STEM scan (:mod:`fields`). The
instances are cut from the probability map (:func:`split_instances`) and
measured (:func:`measure_instances`); :func:`segment_field` is the whole
pipeline for one field.

Nothing here knows about sessions, plots or trees, and nothing materialises a
stack — a field is read when it is segmented and dropped afterwards.
"""
from __future__ import annotations

import numpy as np

from spyde.segmentation.classifier import PixelClassifier
from spyde.segmentation.features import FeatureBank, Normalisation
from spyde.segmentation.fields import FieldSource, field_source
from spyde.segmentation.instances import split_instances
from spyde.segmentation.labels import (
    BACKGROUND, BOUNDARY, CLASSES, PARTICLE, LabelClass, Labels,
)
from spyde.segmentation.measure import COLUMNS, measure_instances


def segment_field(field: np.ndarray, classifier: PixelClassifier, *,
                  scale: float = 1.0, **split) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Classify, split and measure one field.

    ``split`` is passed to :func:`split_instances`. Returns ``(labels, table)``:
    the ``(H, W)`` int32 instance labels and the measured columns.
    """
    field = np.asarray(field)
    particle, boundary = classifier.foreground(field)
    labels = split_instances(particle, boundary, **split)
    return labels, measure_instances(labels, field, scale=scale)


__all__ = [
    "PixelClassifier", "FeatureBank", "Normalisation",
    "FieldSource", "field_source",
    "Labels", "LabelClass", "CLASSES", "PARTICLE", "BACKGROUND", "BOUNDARY",
    "split_instances", "measure_instances", "COLUMNS",
    "segment_field",
]

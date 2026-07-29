"""spyde.data — example and test datasets.

Two halves, deliberately separate:

``synthetic``
    Pure generator functions: numpy in, a HyperSpy signal out. No download, no
    file, no Session. Small enough for a Playwright spec, and — the load-bearing
    property — **asymmetric and crisp**, so a mirrored axis, a stale frame or a
    transposed nav grid is visible in a screenshot rather than hiding behind
    symmetric test data (the ``si_grains`` / ``movie`` precedent).

    Every generator stamps its ground truth into
    ``metadata.Spyde.synthetic`` so a test can assert against the values the
    data was built from instead of a golden file.

Real / downloadable datasets (4D STEM, EELS, EDS, EBSD, in-situ) land here too
— that is Wave 5 (GitHub #80). The loaders that expose all of this to the UI
live in ``spyde/backend/tutorial_data.py`` (user-reachable) and
``spyde/backend/_session_testharness.py`` (test-gated).
"""
from __future__ import annotations

from spyde.data.synthetic import (
    atom_lattice,
    ebsd_patterns,
    eds_si,
    eels_si,
    ground_truth,
    particle_movie,
    particle_truth_at,
)

__all__ = ["eels_si", "eds_si", "ebsd_patterns", "atom_lattice",
           "particle_movie", "particle_truth_at", "ground_truth"]

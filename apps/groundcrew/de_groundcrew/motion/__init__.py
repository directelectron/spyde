"""
motion — Ground Crew's motion correction, on this app's Qt-free plumbing.

The compute is **not** implemented here. It is vendored verbatim in
`de_groundcrew.external.gc_motion` from the upstream repo, and `driver` is the
thin layer that calls it with `progress` / `should_cancel` callbacks instead of
Qt signals. See that package's docstring for why, and MOTION.md for the feature
inventory.

Everything this module exports is re-exported from `driver`, so callers never
reach into the vendored code directly — which keeps the seam small enough that
re-syncing upstream is a file copy.
"""
from de_groundcrew.motion.driver import (
    Cancelled, MODES, ORIENTATION_LABELS, align_stack, apply_orientation,
    classify_gain_tier, correct_local_motion, load_gain, load_movie_stack,
    log_fft, match_gain_to_frame, rank_gain_orientations, save_image,
)

__all__ = [
    "Cancelled", "MODES", "ORIENTATION_LABELS",
    "align_stack", "correct_local_motion",
    "load_movie_stack", "load_gain", "save_image", "log_fft",
    "apply_orientation", "match_gain_to_frame",
    "rank_gain_orientations", "classify_gain_tier",
]

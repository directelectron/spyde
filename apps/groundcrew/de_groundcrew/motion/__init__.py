"""
motion — the old Ground Crew's S.T.A.C.K. motion correction, Qt-free.

Recovered from a deleted branch of the PySide6 app and ported; MOTION.md is the
feature inventory and records what changed and what was deliberately left out.

Phase 1 (`align`) is whole-frame drift correction; Phase 2 (`local`) is the
patch-based residual field, and runs on Phase 1's output. Phase 3 (CTF) was
planned in the old repo but never written, so there is nothing to port.
"""
from de_groundcrew.motion.align import (
    Cancelled, REFERENCES, align_stack, apply_shift_fourier, bandpass_filter,
    cross_correlate, smooth_shifts, upsampled_dft)
from de_groundcrew.motion.frames import (
    ORIENTATION_LABELS, apply_orientation, bin_image, load_gain,
    load_movie_stack, log_fft, match_gain_to_frame, save_image,
    validate_gain_orientation)
from de_groundcrew.motion.local import (
    apply_local_shifts, build_cosine_blend_weights, correct_local_motion,
    correlate_patches, evaluate_motion_field, fit_motion_field,
    generate_patch_grid, smooth_patch_shifts)

__all__ = [
    "Cancelled", "REFERENCES", "ORIENTATION_LABELS",
    "align_stack", "apply_shift_fourier", "bandpass_filter", "cross_correlate",
    "smooth_shifts", "upsampled_dft",
    "apply_orientation", "bin_image", "load_gain", "load_movie_stack",
    "log_fft", "match_gain_to_frame", "save_image", "validate_gain_orientation",
    "apply_local_shifts", "build_cosine_blend_weights", "correct_local_motion",
    "correlate_patches", "evaluate_motion_field", "fit_motion_field",
    "generate_patch_grid", "smooth_patch_shifts",
]

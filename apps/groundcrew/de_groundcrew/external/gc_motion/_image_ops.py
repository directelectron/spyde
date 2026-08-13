"""
Shared low-level image operations for DE Ground Crew workers.

Single source of truth for the pure-numpy array helpers that were previously
copy-pasted across workers/motion_correction_worker.py and
workers/calibration_worker.py. Those modules now re-export these names, so
existing imports (tests, ui.stack_panel, ui.calibration_panel, batch scripts)
that do `from workers.motion_correction_worker import _bin_image, ...` keep
working unchanged.

NOTE: _motion_correction_v2.py keeps its own private copies on purpose —
it is the standalone algorithm prototype under cryoSPARC validation and is left
untouched until that validation completes. Do not repoint it here without an
equivalence check + sign-off (see notes/audit_2026-06-15/phase1_plan.md).

Pure CPU numpy only; no GPU, no Qt, no I/O.
"""
import numpy as np


def _bin_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Bin image by summing factor x factor blocks (preserves total counts).

    factor == 1 returns the input unchanged — a fast path that is value-identical
    to the block-sum (a no-op reshape/sum for factor 1).
    """
    if factor == 1:
        return img
    h, w = img.shape
    nh = (h // factor) * factor
    nw = (w // factor) * factor
    return img[:nh, :nw].reshape(
        nh // factor, factor, nw // factor, factor
    ).sum(axis=(1, 3))


def _apply_orientation(img: np.ndarray, idx: int) -> np.ndarray:
    """Apply one of 8 orientation transforms to a 2D array."""
    if idx == 0: return img
    if idx == 1: return np.rot90(img, 1)
    if idx == 2: return np.rot90(img, 2)
    if idx == 3: return np.rot90(img, 3)
    if idx == 4: return np.fliplr(img)
    if idx == 5: return np.flipud(img)
    if idx == 6: return img.T
    if idx == 7: return np.rot90(img, 2).T
    return img


def _match_gain_to_frame(gain: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    """Resize gain to match frame dimensions (handles super-res gain with standard data)."""
    gh, gw = gain.shape
    if gh == frame_h and gw == frame_w:
        return gain
    # Super-res gain (2x each dimension) -> bin by averaging
    if gh == frame_h * 2 and gw == frame_w * 2:
        return _bin_image(gain, 2) / 4.0  # sum -> average
    # Other integer ratios
    ratio_h = gh // frame_h
    ratio_w = gw // frame_w
    if ratio_h == ratio_w and ratio_h >= 2 and gh == frame_h * ratio_h and gw == frame_w * ratio_w:
        return _bin_image(gain, ratio_h) / float(ratio_h * ratio_w)
    raise ValueError(
        f"Gain dimensions ({gh}x{gw}) don't match frame ({frame_h}x{frame_w}). "
        f"Expected same size or integer multiple."
    )


# --- gain orientation labels (single source of truth; re-exported by motion_correction_worker) ---
ORIENTATION_LABELS = [
    "Identity", "Rot90", "Rot180", "Rot270",
    "FlipH", "FlipV", "Transpose", "Transverse",
]

# --- gain-orientation floor (audit F3). sep = median(scores) / best; high => one orientation fits. ---
# Thresholds calibrated on real Apollo+Tundra+cross-camera gains (n=219 pairings), Linux run
# 2026-07-07 — see notes/runs/2026-07-06-gain-orientation-characterization.md. Distributions (summed):
# correct median 6.10 (p5 2.36); clean-wrong median 1.007, max 1.050; the app feeds SUMMED movies.
# SCOPE (same-model blind spot): this catches wrong camera-TYPE / corrupted / random / wrong-ORIENTATION
# gains, NOT a gain from a different UNIT of the SAME camera model (same-model sensors share row/col
# readout structure, which the row_std+col_std metric cancels regardless of unit — those score ~2.0).
# ⚠ THE THRESHOLDS ASSUME ALL EIGHT ORIENTATIONS ARE SCORED. sep is
# median(scores)/best over the 8-element list; drop to 4 (e.g. "skip
# shape-changing orientations on a non-square sensor" -- the backlog's first
# fix option for the non-square crash below) and the median is a DIFFERENT
# ORDER STATISTIC: the boundaries need re-measuring against the n=219 corpus,
# not inheriting (final-review P-7). Nothing else in this file would make
# that visible to whoever makes the change.
GAIN_SEP_FAIL: float = 1.07   # clean-wrong gains top out at sep=1.050; 1.07 gives margin below correct.
GAIN_SEP_OK:   float = 1.5    # correct-gain median 6.10; 1.5 => ~2.2% false-warn, all marginal low-exposure.


def rank_gain_orientations(frame, gain):
    """Score all 8 gain orientations and the field separation.

    Returns (scores, sep): scores = [(score, label, idx), ...] ascending by score (best first,
    identical to the prior in-run computation); sep = median(scores) / score_best (+inf if
    score_best == 0). High sep => one orientation clearly wins (gain fits this sensor); sep ~ 1 =>
    all orientations tied (content-wrong gain, or too little signal to register the fixed pattern).
    """
    fh, fw = frame.shape
    bfactor = max(1, max(fh, fw) // 1024)
    gain_matched = _match_gain_to_frame(gain, fh, fw)
    frame_b = _bin_image(frame, bfactor) if bfactor > 1 else frame.copy()
    gain_b = _bin_image(gain_matched, bfactor) if bfactor > 1 else gain_matched.copy()

    scores = []
    for i in range(8):
        corrected = frame_b * _apply_orientation(gain_b, i)
        row_std = float(np.std(corrected.mean(axis=1)))
        col_std = float(np.std(corrected.mean(axis=0)))
        scores.append((row_std + col_std, ORIENTATION_LABELS[i], i))
    scores.sort()

    score_best = scores[0][0]
    score_median = float(np.median([s for s, _, _ in scores]))
    sep = float("inf") if score_best == 0 else score_median / score_best
    return scores, sep


def classify_gain_tier(sep, sep_ok=GAIN_SEP_OK, sep_fail=GAIN_SEP_FAIL):
    """Map a separation ratio to a tier: 'ok' (>= sep_ok), 'fail' (< sep_fail), else 'weak'."""
    if sep >= sep_ok:
        return "ok"
    if sep < sep_fail:
        return "fail"
    return "weak"


def apply_reference_correction(frame, dark, gain, mode,
                               dark_orient: int = 0, gain_orient: int = 0,
                               n_frames: int = 1):
    """Apply offline dark/gain correction to a single 2D image.

    mode "none"      -> copy of frame
    mode "gain"      -> frame * gain                    (no dark; data already dark-corrected)
    mode "dark"      -> frame - n_frames * dark
    mode "dark+gain" -> (frame - n_frames * dark) * gain

    Dark is a PER-FRAME reference: on a summed movie of n_frames it is
    subtracted n_frames times. Gain is multiplicative and factor-free
    (sum(f_i * g) == (sum f_i) * g). Single images use n_frames=1.
    Each reference is oriented (_apply_orientation) + size-matched
    (_match_gain_to_frame); a size-incompatible ref raises ValueError.
    Returns a NEW float32 array; never mutates `frame`.
    """
    out = frame.astype(np.float32, copy=True)
    if mode == "none":
        return out
    h, w = out.shape

    if mode in ("dark", "dark+gain"):
        if dark is None:
            raise ValueError("dark correction requires a dark reference")
        d = _apply_orientation(dark, dark_orient)
        d = _match_gain_to_frame(d, h, w)
        out -= n_frames * d

    if mode in ("gain", "dark+gain"):
        if gain is None:
            raise ValueError("gain correction requires a gain reference")
        g = _apply_orientation(gain, gain_orient)
        g = _match_gain_to_frame(g, h, w)
        out *= g

    return out

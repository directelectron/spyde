"""
align.py — whole-frame motion correction (the old app's Phase 1).

Ported from `workers/motion_correction_worker.py`. The algorithm follows
**Unblur** (Grant & Grigorieff 2015) and **MotionCor2** (Zheng et al. 2017),
and four of its choices are deliberate enough to be worth stating, because each
is the kind of thing a well-meaning rewrite "corrects" into something worse:

**Cross-correlation, not phase correlation.** Phase correlation whitens the
spectrum, which throws away the amplitude information that carries the signal
on a low-dose frame. Unblur and MotionCor2 both keep plain cross-correlation
for exactly this reason.

**Alignment runs on RAW frames, never gain-corrected ones.** On sparse counting
data most pixels are zero, and multiplying by a gain reference destroys the
correlation signal. Gain is applied only to the final full-resolution sum.

**Two passes.** Pass 1 aligns against one chosen frame; those shifts build a
refined reference from the whole stack, and pass 2 re-aligns against that. A
single frame is a noisy reference; the summed one is not.

**Shifts are applied as a Fourier phase ramp.** Exact for band-limited data and
free of the interpolation blur that a spatial-domain shift introduces — which
matters when the whole point is to recover resolution.

CuPy is not used here. The old app had a GPU path with a numpy fallback; this
app has no GPU dependency (see MOTION.md §3) and the numpy path is the one the
old app used on every machine without an NVIDIA card. The FFT calls in this
module are the seam if that changes.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline

from de_groundcrew.motion.frames import (
    apply_orientation, bin_image, log_fft, match_gain_to_frame)

#: Subpixel refinement factor for whole-frame alignment — shifts resolve to
#: 1/20 px. The old app's value.
UPSAMPLE_FACTOR = 20

#: Bandpass passband in fractional Nyquist units. The low cut removes uneven
#: illumination, which otherwise dominates the correlation peak.
LOW_FREQ, HIGH_FREQ = 0.01, 0.5

#: Reference frame choices, mapped from the UI's wording.
REFERENCES = ("central", "first", "average")

Progress = Callable[[str], None]
ShouldCancel = Callable[[], bool]


def _noop(_msg: str) -> None: ...
def _never() -> bool: return False


class Cancelled(Exception):
    """Raised when a caller's `should_cancel` asked for a stop."""


# ── Primitives ────────────────────────────────────────────────────────────────

def bandpass_filter(shape, low_freq: float = 0.005,
                    high_freq: float = 0.5) -> np.ndarray:
    """Cosine-edge bandpass in Fourier space, in fractional Nyquist units."""
    h, w = shape
    fy = np.fft.fftfreq(h).astype(np.float64)
    fx = np.fft.fftfreq(w).astype(np.float64)
    freq_norm = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) / 0.5

    lo_width = max(low_freq * 0.5, 0.002)
    hi_width = max((1.0 - high_freq) * 0.5, 0.01)

    filt = np.ones_like(freq_norm)
    lo = freq_norm < low_freq
    lo_edge = (freq_norm >= low_freq - lo_width) & (freq_norm < low_freq)
    filt[lo] = 0.0
    if np.any(lo_edge):
        filt[lo_edge] = 0.5 * (1.0 + np.cos(
            np.pi * (freq_norm[lo_edge] - low_freq) / lo_width))
    hi = freq_norm > high_freq
    hi_edge = (freq_norm > high_freq) & (freq_norm <= high_freq + hi_width)
    filt[hi] = 0.0
    if np.any(hi_edge):
        filt[hi_edge] = 0.5 * (1.0 + np.cos(
            np.pi * (freq_norm[hi_edge] - high_freq) / hi_width))
    return filt.astype(np.float64)


def upsampled_dft(data, upsampled_region_size, upsample_factor, axis_offsets):
    """Upsampled DFT by matrix multiplication (Guizar-Sicairos et al. 2008).

    `data` is ``conj(F1) * F2`` in Fourier space. Follows scikit-image's
    reference implementation; the ``+2j`` sign is the inverse transform,
    because this evaluates cc(τ) from CC(k).
    """
    ups = upsampled_region_size

    row_idx = np.arange(ups, dtype=np.float64) - axis_offsets[0]
    freq_row = np.fft.ifftshift(
        np.arange(data.shape[0], dtype=np.float64) - data.shape[0] // 2)
    kernel_r = np.exp(+2j * np.pi / (data.shape[0] * upsample_factor)
                      * row_idx[:, None] * freq_row[None, :])

    col_idx = np.arange(ups, dtype=np.float64) - axis_offsets[1]
    freq_col = np.fft.ifftshift(
        np.arange(data.shape[1], dtype=np.float64) - data.shape[1] // 2)
    kernel_c = np.exp(+2j * np.pi / (data.shape[1] * upsample_factor)
                      * col_idx[:, None] * freq_col[None, :])

    return kernel_r @ data @ kernel_c.T


def cross_correlate(ref: np.ndarray | None, frame: np.ndarray | None, *,
                    upsample_factor: int = UPSAMPLE_FACTOR,
                    bandpass: bool = True,
                    ref_fft=None, frame_fft=None, bp_filter=None,
                    ) -> tuple[float, float]:
    """Subpixel ``(dy, dx)`` to align *frame* onto *ref*.

    Positive dy means the frame sits below the reference. Pre-computed FFTs may
    be passed to avoid re-transforming a reference that is reused across a
    whole stack — that is most of the cost.
    """
    F2 = frame_fft if frame_fft is not None else np.fft.fft2(
        np.asarray(frame, dtype=np.float32))
    F1 = ref_fft if ref_fft is not None else np.fft.fft2(
        np.asarray(ref, dtype=np.float32))

    if bandpass:
        bp = bp_filter if bp_filter is not None else bandpass_filter(
            F2.shape, low_freq=LOW_FREQ, high_freq=HIGH_FREQ)
        if ref_fft is None:
            F1 = F1 * bp
        if frame_fft is None:
            F2 = F2 * bp

    CC = np.conj(F1) * F2
    cc = np.real(np.fft.ifft2(CC))
    h, w = cc.shape
    peak_y, peak_x = divmod(int(np.argmax(cc)), w)
    # Unwrap: a peak past the halfway point is a negative shift.
    if peak_y > h // 2:
        peak_y -= h
    if peak_x > w // 2:
        peak_x -= w

    if upsample_factor <= 1:
        return float(peak_y), float(peak_x)

    ups_region = int(np.ceil(upsample_factor * 1.5))
    dft_shift = int(np.floor(ups_region / 2.0))
    upsampled = upsampled_dft(
        CC, upsampled_region_size=ups_region, upsample_factor=upsample_factor,
        axis_offsets=[dft_shift - peak_y * upsample_factor,
                      dft_shift - peak_x * upsample_factor])
    uy, ux = divmod(int(np.argmax(np.abs(upsampled))), ups_region)
    return (float(peak_y + (uy - dft_shift) / upsample_factor),
            float(peak_x + (ux - dft_shift) / upsample_factor))


def apply_shift_fourier(frame: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Shift by a Fourier phase ramp — exact for band-limited data, no blur."""
    f = np.asarray(frame, dtype=np.float32)
    h, w = f.shape
    fy = np.fft.fftfreq(h).astype(np.float32).reshape(-1, 1)
    fx = np.fft.fftfreq(w).astype(np.float32).reshape(1, -1)
    # complex64, NOT float32 — and this is a BUG FIX, not a transcription.
    # The original wrote `np.float32(-2j * np.pi)`, which casts a complex
    # constant to a float: NumPy 2 raises, and NumPy 1 truncated it to the real
    # part, 0.0. The phase ramp was therefore exp(0) == 1 and the CPU shift did
    # NOTHING. The GPU branch used `_cp.complex64` and was correct, so on a
    # CUDA machine — the only kind it was developed on — nobody saw it.
    phase = np.exp(np.complex64(-2j * np.pi) * (fy * dy + fx * dx))
    return np.real(np.fft.ifft2(np.fft.fft2(f) * phase)).astype(np.float32)


def smooth_shifts(shifts) -> np.ndarray:
    """Cubic-spline smoothing of a shift trajectory.

    Specimen drift is physically smooth; per-frame correlation noise is not.
    Under four frames there is nothing to fit, so the raw values pass through.
    """
    n = len(shifts)
    if n < 4:
        return np.asarray(shifts, dtype=np.float64)
    t = np.arange(n, dtype=np.float64)
    return CubicSpline(t, shifts)(t)


# ── The pass ──────────────────────────────────────────────────────────────────

def align_stack(stack: np.ndarray, *, gain: np.ndarray | None = None,
                gain_orientation: int = 0, bin_factor: int = 2,
                reference: str = "central", throw: int = 0,
                unaligned_sum: np.ndarray | None = None,
                progress: Progress = _noop,
                should_cancel: ShouldCancel = _never) -> dict:
    """Whole-frame motion correction. Returns the old app's result dictionary.

    *throw* discards that many leading frames before aligning — early frames
    carry the beam-induced initial burst, and including them drags the whole
    trajectory. MotionCor2's ``-Throw``. At least two frames are always kept.
    """
    if reference not in REFERENCES:
        # Not a silent fallback to "central": the old app mapped UI wording to
        # these strings in the panel, so a typo became a wrong reference with
        # no complaint.
        raise ValueError(f"reference must be one of {REFERENCES}, got {reference!r}")

    def check() -> None:
        if should_cancel():
            raise Cancelled()

    full = np.asarray(stack)
    n_total = full.shape[0]
    throw = max(0, min(int(throw), max(0, n_total - 2)))
    stack = full[throw:] if throw else full
    n_frames, fh, fw = stack.shape
    if n_frames < 2:
        raise ValueError(f"need at least 2 frames to align, got {n_frames}")

    oriented_gain = None
    if gain is not None:
        oriented_gain = match_gain_to_frame(
            apply_orientation(gain, gain_orientation), fh, fw)

    # Bin the RAW frames. Gain correction here would destroy the correlation
    # signal on sparse counting data — it is applied to the final sum only.
    progress("Binning raw frames for alignment…")
    binned = []
    for i in range(n_frames):
        check()
        f = stack[i]
        binned.append((bin_image(f, bin_factor) if bin_factor > 1 else f
                       ).astype(np.float32))

    if unaligned_sum is None:
        progress("Computing unaligned sum…")
        unaligned_sum = np.mean(stack, axis=0, dtype=np.float32)
        check()
        if oriented_gain is not None:
            unaligned_sum = unaligned_sum * oriented_gain

    # Central is MotionCor2's default: beam-induced motion has an initial
    # burst, so the middle frame sits closest to the average specimen position
    # and minimises the total shift.
    if reference == "central":
        ref = binned[n_frames // 2]
    elif reference == "first":
        ref = binned[0]
    else:
        ref = np.mean(binned, axis=0).astype(np.float32)

    bp = bandpass_filter(binned[0].shape, low_freq=LOW_FREQ,
                         high_freq=HIGH_FREQ).astype(np.float32)
    ref_fft = np.fft.fft2(ref.astype(np.float32)) * bp

    def _pass(label: str, reference_fft) -> tuple[list[float], list[float]]:
        ys, xs = [], []
        for i in range(n_frames):
            check()
            dy, dx = cross_correlate(None, binned[i], ref_fft=reference_fft,
                                     bp_filter=bp)
            ys.append(dy); xs.append(dx)
            progress(f"{label}: frame {i + 1}/{n_frames}  dy={dy:.2f} dx={dx:.2f}")
        return ys, xs

    progress("Pass 1: estimating shifts…")
    ys1, xs1 = _pass("Pass 1", ref_fft)
    sy1, sx1 = smooth_shifts(ys1), smooth_shifts(xs1)

    # A single frame is a noisy reference. Sum the stack through pass 1's
    # shifts and re-align against that.
    progress("Pass 1: building refined reference…")
    ref_sum = np.zeros_like(binned[0], dtype=np.float32)
    for i in range(n_frames):
        check()
        ref_sum += apply_shift_fourier(binned[i], -sy1[i], -sx1[i])
    refined = ref_sum / n_frames

    progress("Pass 2: refining shifts…")
    ys2, xs2 = _pass("Pass 2", np.fft.fft2(refined) * bp)
    sy, sx = smooth_shifts(ys2), smooth_shifts(xs2)

    scale = float(bin_factor) if bin_factor > 1 else 1.0
    final_y, final_x = sy * scale, sx * scale

    progress("Applying shifts to full-resolution frames…")
    aligned_sum = np.zeros((fh, fw), dtype=np.float32)
    for i in range(n_frames):
        check()
        frame = stack[i].astype(np.float32)
        if oriented_gain is not None:
            frame = frame * oriented_gain
        aligned_sum += apply_shift_fourier(frame, -final_y[i], -final_x[i])
        progress(f"Summing aligned frame {i + 1}/{n_frames}")
    aligned_sum /= n_frames

    progress("Computing aligned FFT…")
    return {
        "aligned_sum": aligned_sum,
        "unaligned_sum": np.asarray(unaligned_sum, dtype=np.float32),
        "aligned_fft": log_fft(aligned_sum),
        "shifts_y_raw": [s * scale for s in ys2],
        "shifts_x_raw": [s * scale for s in xs2],
        "shifts_y_smooth": final_y.tolist(),
        "shifts_x_smooth": final_x.tolist(),
        "n_frames": n_frames,
        "bin_factor": bin_factor,
        "throw": throw,
    }

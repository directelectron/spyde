"""
translation.py — rigid (translation-only) drift solve, the cryo-EM way.

This is the MotionCor2 / Unblur family method, and it makes three choices that
are the *opposite* of naive phase correlation. Each one is a correction of a
real defect in the implementation this replaced.

**1. Band-pass-weighted PLAIN cross-correlation — never whitened.**
Microscopy noise is Poisson-dominated: noise power is roughly flat in frequency
while signal power falls off, so SNR *decreases* with q. Pure phase correlation
divides the cross-power spectrum by ``|R|``, which normalises every band to unit
weight — handing the bands that are pure noise at low dose exactly as much say
as the bands that carry the image. That is backwards, and the symptom was a
magnitude FLOOR constant needed to stop near-empty bins being amplified to unit
weight by their own rounding error. Here the spectrum is instead *weighted* by a
FIXED band-pass — a function of frequency alone, never of the data's own
magnitude, so an empty bin stays empty instead of being promoted to unit weight.
See :func:`_bandpass`.

The peak is taken from ``real(ifft2(...))``, not ``abs(...)``. A true match is a
positive real peak; ``abs`` also promotes spurious imaginary structure, which is
free extra chances to pick the wrong lag.

**What plain correlation costs, measured.** Discarding the magnitude normalisation
is not free, and it is worth knowing where the bill lands before changing anything
here. Plain correlation is weighted by AMPLITUDE, so it is dominated by whatever
is brightest; phase correlation equalises every feature, so it is dominated by
whatever is most numerous. On a whole frame of low-dose micrograph texture that is
a clear win for plain correlation (see ``benchmark_drift_translation.py``). On a
SMALL ROI containing a few bright objects that move differently from the
background it is a clear loss: on ``particle_movie``'s default half-frame box the
old phase solver recovered the stamped drift to 0.37 px and this one manages
~5 px, because the bright particles outweigh the low-contrast support film that
actually carries the stage motion. ``skimage.registration.phase_cross_correlation``
with ``normalization=None`` reproduces the same error on the same crops, so this is
a property of the estimator, not of this implementation. The band-pass defaults
below are chosen to claw back as much of that as a fixed filter can.

**2. Banded PAIRWISE measurement + LEAST SQUARES, not a chain.**
A sequential running-reference chain produces exactly ONE measurement per
unknown shift, so a bad registration propagates with nothing to outvote it.
Here every pair ``(i, j)`` with ``0 < j - i <= band`` is measured, giving ~``band``
constraints per unknown, and the per-frame positions come from an over-determined
least-squares solve with an explicit gauge (frame 0 at the origin). Cryo-EM uses
*all* pairs; an in-situ movie runs to thousands of frames, so O(N²) is not
viable and the band is the compromise. Outliers are down-weighted by IRLS on the
pairwise residual — the consensus of the other pairs is what rejects a bad
measurement, rather than an absolute quality threshold. See
:func:`_solve_positions`.

**3. The correlations BATCH.** The ``band`` pairs that close on frame *j* all
share the same moving spectrum, so they are one ``(B, gh, gw)`` product and ONE
batched inverse FFT — not ``band`` separate calls. Correlation runs on a BINNED
grid (``corr_size``), which is both what bounds the cost at 4096² and a genuine
low-pass: the drift signal lives at coarse spatial frequencies, so binning costs
nothing but buys ``bin²``. The sub-pixel refinement then works in full-frame
units (``upsample * bin`` on the grid), so the reported resolution is
``1/upsample`` of a FULL pixel regardless of binning.

Memory is bounded by construction (CLAUDE.md Memory-Safety rule): at most
``band + 1`` *binned* spectra are resident plus one full-resolution frame, and
the batched product is sub-batched under :data:`_MAX_BATCH_BYTES`. A 3000 × 4096²
movie never has more than a few hundred MB in flight.

Kept from the previous implementation because it was already right: the
Guizar-Sicairos upsampled matrix-multiply DFT for the sub-pixel peak (evaluating
the correlation on a ~12×12 window beats zero-padding to a 32768² transform),
the streaming frame source, and the numpy/torch operator adapter whose parity
test is what protects the GPU path.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np

from spyde.drift.frames import frame_source
from spyde.drift.model import DriftModel

log = logging.getLogger(__name__)

# ── sub-pixel refinement ─────────────────────────────────────────────────────

# Guizar-Sicairos: the refinement window spans 1.5 upsampled pixels either side
# of the current estimate. Matches skimage's `phase_cross_correlation`.
_UPSAMPLED_REGION_FACTOR = 1.5

# The refinement is a LADDER, not one jump: each stage raises the upsample factor
# by at most this ratio, so the window stays ~12 px wide however fine the target.
#
# This matters once correlation is binned. A 4096² frame binned to 512 needs
# `upsample * bin` = 8 * 8 = 64 upsampled grid steps to resolve 1/8 of a FULL
# pixel; in one stage that is a 96-wide window, and the matmul cost is linear in
# the window width. Two stages of 12 cost 8x less and land in the same place —
# stage 1 localises to 1/8 grid px, so stage 2 only has to search +/- that.
_REFINE_STEP = 8.0

# ── correlation grid ─────────────────────────────────────────────────────────

#: Target edge (px) of the binned grid the correlation runs on. The frame is
#: box-mean binned by an integer factor until its longer edge is at or below
#: this, which bounds BOTH the FFT cost and the resident spectra at any frame
#: size. At the default ``max_shift`` a 4096² movie correlates on 512² tiles —
#: 64x less arithmetic per pair, which is what makes measuring `band` pairs per
#: frame cheaper than the ONE full-resolution correlation the chain used to do.
#: The bin is also capped by the search range (:data:`_MIN_SEARCH_GRID`) and by
#: :data:`_MIN_GRID`, so this is a target rather than a promise.
#:
#: Binning is also a low-pass, and it is the one that matters: a box mean over b
#: pixels raises the Poisson SNR per grid pixel by b and deletes the sub-b-pixel
#: band outright. Measured on the benchmark movie at 8 e/px, unbinned correlation
#: comes back 17 px wrong and bin-4 comes back 0.19 px wrong at the SAME filter
#: settings. Drift lives at coarse spatial frequencies, so nothing is lost: the
#: discarded band is the one with the worst SNR. Sub-pixel accuracy is unaffected
#: because the refinement target is scaled by the bin factor
#: (see :data:`_REFINE_STEP`).
#:
#: **Where this stops being enough:** a SMALL frame at very low dose. 256² at
#: 2 e/px only reaches bin 2 here and comes back ~13 px wrong; 512² and up reach
#: bin 4-16 and stay inside 0.5 px at the same dose. Lower ``corr_size`` for such
#: a movie — that is the knob, and it is why this is a parameter.
DEFAULT_CORR_SIZE = 256

#: Never bin below this many grid pixels per axis — a correlation grid smaller
#: than the refinement window has nowhere to put a peak.
_MIN_GRID = 64

#: Smallest coarse search box, in grid pixels either side of zero. Binning shrinks
#: `max_shift` measured in grid pixels, so an unbounded bin would eventually leave
#: a 1-pixel search; this is what stops that (see :func:`_bin_factor`).
_MIN_SEARCH_GRID = 4

#: Ceiling on the batched ``(B, gh, gw)`` cross-power product. The band is
#: sub-batched to stay under it, so ``corr_size=0`` (no binning) on a 4096²
#: movie degrades to smaller batches rather than to an out-of-memory.
_MAX_BATCH_BYTES = 256 * 1024 * 1024

# ── the band-pass ────────────────────────────────────────────────────────────

#: Gaussian low-pass 1/e point, in cycles per GRID pixel (0.5 = grid Nyquist).
#: This is MotionCor2's ``-Bft`` in a scale-free form: ``exp(-B q²/4)`` with q in
#: cycles/px is ``exp(-(q/q_lp)²)`` for ``q_lp = 2/sqrt(B)``, so the default here
#: corresponds to B ≈ 8 px² on the binned grid — a gentle roll-off, because the
#: binning (:data:`DEFAULT_CORR_SIZE`) has already done the coarse low-passing.
#:
#: The weight is applied to EACH spectrum, so the cross-power product sees it
#: squared and the effective 1/e point of the correlation is ``lowpass/sqrt(2)``.
DEFAULT_LOWPASS = 0.5

#: High-pass 1/e point, cycles per grid pixel. ``1 - exp(-(q/q_hp)²)``, which is
#: exactly zero at DC. Two jobs: remove the mean (a plain, unwhitened
#: cross-correlation is otherwise dominated by the product of the two means) and
#: suppress the smooth illumination gradient, which is the one large-scale
#: feature in a micrograph that does NOT move with the sample.
DEFAULT_HIGHPASS = 0.08

#: Spectral tilt: the weight carries a ``q**tilt`` factor, so the band-pass is
#: ``q**tilt · exp(-(q/lowpass)²)`` and the correlation sees the square of that.
#:
#: **This is the low cut, and it is doing most of the work.** A Gaussian high-pass
#: only notches a few bins around DC; a natural image's power spectrum falls like
#: ``q**-2`` or steeper, so without a tilt the plain correlation peak is as broad
#: as the scene's own autocorrelation and its position is pulled around by any
#: smooth change in content. ``tilt=1`` per spectrum is a gradient filter — the
#: classic "correlate the edges" registration — and 0.6 is most of the way there.
#:
#: **It is NOT whitening.** ``q**tilt`` is a fixed function of frequency; it
#: applies the same weight whatever the data contains, so it cannot amplify an
#: empty bin above a full one. That distinction is the entire reason
#: :data:`_PHASE_FLOOR` no longer needs to exist.
#:
#: **The ceiling on it is noise, and it is sharp.** Measured on a 512²/1024²
#: synthetic movie with INDEPENDENT Poisson noise (the benchmark's, not the unit
#: fixtures' — those shift one base image, so their pixel noise is a common-mode
#: fiducial that flatters any high-q weighting):
#:
#:   dose 8 e/px, no binning:   tilt 0.5 → 0.6 px rms,  tilt 0.75 → 9.9 px rms
#:   dose 8 e/px, bin 4:        tilt 0.5 → 0.19 px,     tilt 1.0 → 0.19 px
#:
#: i.e. the tilt is only safe once the noisiest band is gone, and binning is what
#: removes it. Raising this without lowering :data:`DEFAULT_CORR_SIZE` is the way
#: to reintroduce exactly the failure phase correlation was blamed for.
DEFAULT_TILT = 0.75

# ── the least-squares solve ──────────────────────────────────────────────────

#: Default half-width of the measurement band: every pair ``(i, j)`` with
#: ``0 < j - i <= band``. Cryo-EM measures ALL pairs; at the thousands of frames
#: an in-situ movie reaches that is O(N²) and not viable, so the band is the
#: compromise — and how far it has to reach is the single least obvious number
#: here.
#:
#: **A band accumulates bias; all-pairs does not.** The old solver registered
#: every frame against one reference, so a per-measurement bias showed up once.
#: A banded system chains ``N/band`` independent links from frame 0 to frame N,
#: and ANY systematic per-pair error multiplies along that chain. Measured on the
#: 240-frame 2048² benchmark (0.17 px/frame, i.e. 2 px across a 12-frame band):
#:
#:   band 12  ->  1.46 px rms,  recovered/true = 0.90
#:   band 24  ->  0.60 px rms
#:   band 48  ->  0.09 px rms
#:
#: The error is a proportional SHRINK, not noise, and it has one cause: a plain
#: correlation peak sits on the shoulder of the window/content envelope, which
#: pulls it toward zero, and that pull is proportionally largest when the shift
#: being measured is a small fraction of a (binned) pixel. A wide band fixes both
#: halves at once — the shift across it is bigger, so the bias is relatively
#: smaller, AND there are fewer links to multiply it along.
#:
#: 48 is therefore the default even though the literature's rule of thumb is
#: 10-20: it costs ~1.4x the time of 12 (the per-frame read and FFT dominate, not
#: the pairs) and buys 16x the accuracy on real-scale data. Raise it further for
#: a very slow drift; the ceiling is memory, since ``band + 1`` binned spectra
#: stay resident.
DEFAULT_BAND = 48

#: IRLS passes over the solved system. Bisquare converges in 2-3; more only
#: costs banded solves, which are microseconds.
_IRLS_ITERS = 3

#: Tukey bisquare tuning constant (95% efficiency at the Gaussian).
_TUKEY_C = 4.685

#: Robust scale floor, in pixels. Without it a *perfect* fit (synthetic data,
#: residuals ~1e-4 px) drives the scale to zero and every pair that is one ULP
#: off consensus gets rejected. This is the sub-pixel noise floor below which
#: disagreement is not evidence of anything.
_SCALE_FLOOR_PX = 0.02

#: Weights are floored rather than zeroed. A frame whose every pair is rejected
#: would otherwise leave its position unconstrained and the banded system
#: singular; a tiny weight keeps it determined (by measurements known to be bad,
#: which is the honest answer for a frame with nothing else to go on).
_WEIGHT_FLOOR = 1e-3

#: Smallest alignment ROI worth correlating.
_MIN_ROI = 16


# ── operator adapters ────────────────────────────────────────────────────────
# The algorithm below is written once against this interface. `_TorchOps` is the
# production path; `_NumpyOps` is the reference the parity test pins it against.
# Everything is BATCHED over a leading pair axis — that is the whole point of the
# adapter now, and why the FFT/argmax/matmul entries take (B, gh, gw).

class _NumpyOps:
    name = "numpy"

    def __init__(self, device=None):
        self.device = None

    def to_backend(self, a):
        return np.asarray(a, dtype=np.float32)

    def from_numpy(self, a):
        return np.asarray(a)

    def bin_mean(self, a, b: int):
        gh, gw = a.shape[-2] // b, a.shape[-1] // b
        v = a[..., :gh * b, :gw * b]
        return v.reshape(v.shape[:-2] + (gh, b, gw, b)).mean(axis=(-3, -1))

    def mean(self, a):
        return np.mean(a)

    def fft2(self, a):
        return np.fft.fft2(a).astype(np.complex64)

    def ifft2(self, a):
        return np.fft.ifft2(a)

    def conj(self, a):
        return np.conj(a)

    def real(self, a):
        return np.real(a)

    def exp_i(self, a):
        """``exp(-2j*pi*a)`` for a real array — the one complex exp we need."""
        return np.exp(-2j * math.pi * np.asarray(a, dtype=np.float64)).astype(
            np.complex64)

    def fftfreq(self, n):
        return np.fft.fftfreq(n).astype(np.float32)

    def arange(self, n):
        return np.arange(n, dtype=np.float32)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def swap_last(self, a):
        return np.swapaxes(a, -1, -2)

    def masked_max(self, cc, mask):
        """Per-batch masked argmax + peak/mean/std, returned as numpy."""
        b = cc.shape[0]
        flat = np.where(mask, cc, -np.inf).reshape(b, -1)
        idx = np.argmax(flat, axis=1)
        peak = flat[np.arange(b), idx]
        raw = cc.reshape(b, -1)
        return idx, peak, raw.mean(axis=1), raw.std(axis=1)

    def argmax_flat(self, a):
        return np.argmax(a.reshape(a.shape[0], -1), axis=1)

    def to_numpy(self, a):
        return np.asarray(a)


class _TorchOps:
    name = "torch"

    def __init__(self, device):
        import torch
        self._torch = torch
        self.device = device

    def to_backend(self, a):
        t = self._torch
        return t.as_tensor(np.ascontiguousarray(a, dtype=np.float32),
                           device=self.device)

    def from_numpy(self, a):
        t = self._torch
        return t.as_tensor(np.ascontiguousarray(a, dtype=np.float32),
                           device=self.device)

    def bin_mean(self, a, b: int):
        gh, gw = a.shape[-2] // b, a.shape[-1] // b
        v = a[..., :gh * b, :gw * b]
        return v.reshape(tuple(v.shape[:-2]) + (gh, b, gw, b)).mean(dim=(-3, -1))

    def mean(self, a):
        return self._torch.mean(a)

    def fft2(self, a):
        return self._torch.fft.fft2(a).to(self._torch.complex64)

    def ifft2(self, a):
        return self._torch.fft.ifft2(a)

    def conj(self, a):
        return self._torch.conj(a)

    def real(self, a):
        return self._torch.real(a)

    def exp_i(self, a):
        t = self._torch
        ph = (-2.0 * math.pi) * a
        return t.complex(t.cos(ph), t.sin(ph)).to(t.complex64)

    def fftfreq(self, n):
        return self._torch.fft.fftfreq(n, device=self.device,
                                       dtype=self._torch.float32)

    def arange(self, n):
        return self._torch.arange(n, device=self.device,
                                  dtype=self._torch.float32)

    def matmul(self, a, b):
        return self._torch.matmul(a, b)

    def swap_last(self, a):
        return a.transpose(-1, -2)

    def masked_max(self, cc, mask):
        t = self._torch
        b = cc.shape[0]
        flat = cc.reshape(b, -1)
        masked = flat.masked_fill(~mask.reshape(1, -1), float("-inf"))
        peak, idx = masked.max(dim=1)
        return (idx.detach().cpu().numpy(),
                peak.detach().cpu().numpy(),
                flat.mean(dim=1).detach().cpu().numpy(),
                flat.std(dim=1).detach().cpu().numpy())

    def argmax_flat(self, a):
        t = self._torch
        return t.argmax(a.reshape(a.shape[0], -1), dim=1).detach().cpu().numpy()

    def to_numpy(self, a):
        return a.detach().cpu().numpy()


def _resolve_ops(device: str | None):
    """Pick the backend: ``cuda`` > ``mps`` > **torch CPU** > numpy.

    **torch CPU beats numpy even with no GPU**, and by a lot: ``np.fft.fft2`` is
    single-threaded while ``torch.fft.fft2`` uses every core, and the FFTs are
    the whole cost of this solver. numpy is kept as an **explicitly** selectable
    reference path (``device="numpy"``), which is what the backend-parity test
    pins the torch path against.
    """
    if device == "numpy":
        return _NumpyOps(None)
    try:
        import torch
    except Exception:
        return _NumpyOps(None)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and \
                torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    try:
        return _TorchOps(torch.device(device))
    except Exception as exc:      # pragma: no cover — bad device string
        log.warning("[drift] torch device %r unusable (%s); using numpy", device, exc)
        return _NumpyOps(None)


# ── windows, filters and masks (built once per solve) ────────────────────────

#: Default Tukey taper fraction. 0.35 tapers the outer ~17% at each edge and
#: leaves the middle two thirds at unit weight. See :func:`_taper2d` for why this is NOT 1.0 (a
#: full Hann window) and why it is no longer 0.25.
DEFAULT_TAPER_ALPHA = 0.35


def _tukey1d(n: int, alpha: float) -> np.ndarray:
    """Tukey (cosine-tapered) window. ``alpha=0`` rectangular, ``alpha=1`` Hann."""
    if n < 2:
        return np.ones(max(1, n), dtype=np.float32)
    alpha = float(min(1.0, max(0.0, alpha)))
    if alpha <= 0.0:
        return np.ones(n, dtype=np.float32)
    x = np.arange(n, dtype=np.float64) / (n - 1)
    w = np.ones(n, dtype=np.float64)
    lo = x < alpha / 2.0
    hi = x > 1.0 - alpha / 2.0
    w[lo] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[lo] - alpha / 2.0)))
    w[hi] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[hi] - 1.0 + alpha / 2.0)))
    return w.astype(np.float32)


def _taper2d(ops, h: int, w: int, alpha: float):
    """Separable Tukey taper, built in numpy once per solve then moved on-device.

    **Why a Tukey taper and NOT a full Hann window.** Some apodisation is needed:
    a feature entering or leaving at the frame edge otherwise correlates against
    the *border discontinuity* rather than the sample. But a full Hann window
    (``alpha=1``) reweights the entire frame, and once the drift is large the two
    frames have different content under the taper — which manufactures a spurious
    correlation peak that can outrank the true one.

    Measured on the synthetic particle movie, frame 23 (true drift ``(6.0, 2.9)``):
    with a full Hann window the strongest peak sits at ``(-19, 19)`` scoring 0.121
    while the TRUE peak scores only 0.088, so the solve returns ``(-19.2, 18.6)``
    — a 25 px error, worse than not correcting at all. **skimage's
    ``phase_cross_correlation`` returns the same wrong answer on the same windowed
    input**, so this is a property of full-frame windowing, not of either
    implementation. Tapering only the outer edge leaves the interior comparable
    between frames and recovers 0.06 px.

    Do not "simplify" this back to a Hann window.

    **Why the default moved from 0.25 to 0.5.** A plain (unwhitened) correlation
    is far more sensitive to the frame edge than phase correlation was: the peak
    is broad, so the window's own autocorrelation envelope — which falls off with
    lag — multiplies it and drags the maximum toward zero, and content entering or
    leaving at the border feeds that envelope. The bias is a consistent
    UNDER-estimate, which is what makes it dangerous: the drift curve looks the
    right shape and is simply too small. On the ROI fixture in
    ``test_drift_translation.py`` (a 56x64 box with blobs crossing its edge)
    alpha=0.25 gives a 1.5 px error and alpha=0.5 gives 0.08 px. Above ~0.6 it
    turns back around as the taper starts eating the signal itself.
    """
    win = _tukey1d(h, alpha)[:, None] * _tukey1d(w, alpha)[None, :]
    return ops.to_backend(win)


def _bandpass(ops, h: int, w: int, lowpass: float, highpass: float, tilt: float = 0.0):
    """The correlation weight ``W(q)``, in place of phase whitening.

    ``W = q**tilt · exp(-(q/lowpass)²) · (1 - exp(-(q/highpass)²))`` with ``q`` in
    cycles per grid pixel. The Gaussian leg is MotionCor2's B-factor written
    scale-free; the ``q**tilt`` leg is the low cut that a DC notch alone cannot
    provide (:data:`DEFAULT_TILT`); the high-pass leg is exactly zero at DC.

    Every leg is a function of FREQUENCY ONLY. That is the whole difference from
    the phase normalisation this replaces: no bin's weight depends on how much
    energy that bin happens to contain, so there is no bin whose own rounding
    error can be promoted to unit weight and hence no floor constant to tune.

    Applied to each spectrum rather than to the product, so it costs N
    multiplies instead of N·band and the correlation sees ``W²`` — which is why
    the effective cut-offs are the stated ones divided by sqrt(2).
    """
    fy = ops.fftfreq(h).reshape(h, 1)
    fx = ops.fftfreq(w).reshape(1, w)
    q2 = fy * fy + fx * fx
    if lowpass and lowpass > 0:
        weight = _np_exp(ops, -q2 / float(lowpass) ** 2)
    else:
        weight = q2 * 0.0 + 1.0
    if tilt:
        weight = weight * _pow(ops, q2, 0.5 * float(tilt))
    if highpass and highpass > 0:
        weight = weight * (1.0 - _np_exp(ops, -q2 / float(highpass) ** 2))
    elif not tilt:
        # A plain cross-correlation MUST lose DC or the product of the two means
        # swamps everything. Zeroing the bin is the minimum the filter can do.
        weight = _zero_dc(ops, weight)
    return weight


def _pow(ops, a, e):
    if ops.name == "torch":
        return ops._torch.pow(a, e)
    return np.power(a, e)


def _np_exp(ops, a):
    if ops.name == "torch":
        return ops._torch.exp(a)
    return np.exp(a)


def _zero_dc(ops, weight):
    if ops.name == "torch":
        weight = weight.clone()
        weight[0, 0] = 0.0
        return weight
    weight = np.array(weight, copy=True)
    weight[0, 0] = 0.0
    return weight


def _shift_mask(ops, h: int, w: int, max_shift, min_shift, binf: int):
    """Which correlation-grid bins are admissible shifts.

    The correlation is un-shifted, so bin ``k`` means shift ``k`` for
    ``k <= n//2`` and ``k - n`` above that — in GRID pixels, hence the ``binf``
    scaling to compare against bounds the caller gave in full-frame pixels.
    """
    def axis_shifts(n: int):
        k = np.arange(n)
        return np.where(k > n // 2, k - n, k).astype(np.float64) * float(binf)

    sy = axis_shifts(h)
    sx = axis_shifts(w)
    mask = np.ones((h, w), dtype=bool)
    if max_shift is not None:
        mask &= (np.abs(sy) <= float(max_shift))[:, None]
        mask &= (np.abs(sx) <= float(max_shift))[None, :]
    if min_shift is not None:
        r = np.hypot(sy[:, None], sx[None, :])
        mask &= r >= float(min_shift)
    if not mask.any():
        raise ValueError(
            "max_shift/min_shift exclude every possible shift "
            f"(max_shift={max_shift}, min_shift={min_shift}, grid={h}x{w}, "
            f"bin={binf})"
        )
    if ops.name == "torch":
        return ops._torch.as_tensor(mask, device=ops.device)
    return mask


# ── the batched correlation ──────────────────────────────────────────────────

def _dft_kernel(ops, region: int, n: int, u: float, offsets: np.ndarray):
    """``(B, region, n)`` matrix-multiply DFT kernel with a per-pair offset.

    ``kernel[b, m, f] = exp(-2πi (m - offset[b]) · fftfreq(n)[f] / u)``. NOTE the
    ``/u``: these are frequencies on the UPSAMPLED grid. Without it the window
    still evaluates and still finds a peak, but at ``1/u`` of the intended
    resolution, so every recovered shift lands on a multiple of ``1/u`` and the
    refinement silently does nothing.
    """
    m = ops.arange(region).reshape(1, region, 1)
    off = ops.from_numpy(np.asarray(offsets, np.float32)).reshape(-1, 1, 1)
    freq = ops.fftfreq(n).reshape(1, 1, n) / float(u)
    return ops.exp_i((m - off) * freq)


def _upsampled_peak(ops, product, region: int, u: float,
                    off_y: np.ndarray, off_x: np.ndarray):
    """``real`` inverse DFT of *product* on a ``region × region`` window per pair.

    One kernel matmul per axis. ``conj`` before the forward kernel is how the
    forward-kernel matmul evaluates an INVERSE transform; ``real(conj(z)) ==
    real(z)``, so the sign of the imaginary part does not matter and the value
    compared here is the same positive real peak the coarse step maximises.
    """
    data = ops.conj(product)                                   # (B, gh, gw)
    gh, gw = int(data.shape[-2]), int(data.shape[-1])
    kx = _dft_kernel(ops, region, gw, u, off_x)                # (B, r, gw)
    tmp = ops.matmul(kx, ops.swap_last(data))                  # (B, r, gh)
    ky = _dft_kernel(ops, region, gh, u, off_y)                # (B, r, gh)
    out = ops.matmul(ky, ops.swap_last(tmp))                   # (B, r_y, r_x)
    return ops.real(out)


def _correlate_batch(ops, refs, mov, mask, upsample: float, binf: int,
                     predict=None, tight=None):
    """Measure ``len(refs)`` relative shifts against one moving spectrum.

    Returns ``(shifts_full, quality)`` — ``(B, 2)`` in FULL-frame pixels and
    ``(B,)`` peak z-scores. ``shifts_full[b]`` is the correction to ADD to the
    moving frame to register it onto ``refs[b]`` (the ``DriftModel`` sign).

    *predict*, when given, is a ``(B, 2)`` prior in GRID pixels: the product is
    de-ramped by it so the residual peak sits near the origin and the search can
    use the tight *tight* mask. This is the ``-Iter`` refinement pass; a phase
    ramp is exact even for sub-pixel priors and costs no resampling.
    """
    stacked = _stack(ops, refs)                                # (B, gh, gw)
    product = stacked * ops.conj(mov).reshape(1, *mov.shape)
    gh, gw = int(product.shape[-2]), int(product.shape[-1])
    b = int(product.shape[0])

    if predict is not None:
        fy = ops.fftfreq(gh).reshape(1, gh, 1)
        fx = ops.fftfreq(gw).reshape(1, 1, gw)
        py = ops.from_numpy(np.asarray(predict[:, 0], np.float32)).reshape(b, 1, 1)
        px = ops.from_numpy(np.asarray(predict[:, 1], np.float32)).reshape(b, 1, 1)
        product = product * ops.conj(ops.exp_i(py * fy + px * fx))
        search = tight if tight is not None else mask
    else:
        search = mask

    cc = ops.real(ops.ifft2(product))
    idx, peak, mean, std = ops.masked_max(cc, search)
    py_i = (idx // gw).astype(np.float64)
    px_i = (idx % gw).astype(np.float64)
    dy = np.where(py_i > gh // 2, py_i - gh, py_i)
    dx = np.where(px_i > gw // 2, px_i - gw, px_i)

    # Sub-pixel ladder. `upsample` is in FULL pixels; on the grid the target is
    # `upsample * binf` because one grid pixel is `binf` full pixels.
    target = float(upsample) * float(binf)
    u = 1.0
    while u < target - 1e-9:
        u_new = min(target, u * _REFINE_STEP)
        region = max(3, int(math.ceil(_UPSAMPLED_REGION_FACTOR * (u_new / u))))
        dftshift = float(region // 2)
        off_y = (dftshift - dy * u_new).astype(np.float32)
        off_x = (dftshift - dx * u_new).astype(np.float32)
        fine = _upsampled_peak(ops, product, region, u_new, off_y, off_x)
        flat = ops.argmax_flat(fine)
        my = (flat // region).astype(np.float64)
        mx = (flat % region).astype(np.float64)
        dy = dy + (my - dftshift) / u_new
        dx = dx + (mx - dftshift) / u_new
        u = u_new

    if predict is not None:
        dy = dy + np.asarray(predict[:, 0], np.float64)
        dx = dx + np.asarray(predict[:, 1], np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        quality = np.where(std > 0, (peak - mean) / np.maximum(std, 1e-20), 0.0)
    shifts = np.stack([dy, dx], axis=1) * float(binf)
    return shifts.astype(np.float64), np.asarray(quality, np.float64)


def _stack(ops, arrays):
    if ops.name == "torch":
        return ops._torch.stack(list(arrays), dim=0)
    return np.stack(list(arrays), axis=0)


# ── the least-squares solve ──────────────────────────────────────────────────

def _bisquare(r: np.ndarray, scale: float) -> np.ndarray:
    """Tukey bisquare weights — REDESCENDING, which is the point.

    A Huber weight only tapers an outlier's influence; a pure-noise frame's
    measurement is not a mild outlier, it is meaningless, and bisquare takes it
    to (almost) zero. The weight floor keeps the banded system non-singular for a
    frame all of whose pairs are rejected.
    """
    z = r / max(float(scale), 1e-12) / _TUKEY_C
    w = np.where(z < 1.0, (1.0 - z * z) ** 2, 0.0)
    return np.maximum(w, _WEIGHT_FLOOR)


def _solve_banded(n_frames: int, pi, pj, c, w, band: int):
    """Solve the weighted normal equations for per-frame positions.

    The system is ``p_j - p_i = c_ij`` for every measured pair, which is the
    incidence matrix of a banded graph — so ``AᵀWA`` is that graph's weighted
    LAPLACIAN and is banded with the same half-width. That is the whole reason
    thousands of frames are cheap: the solve is ``scipy.linalg.solveh_banded`` on
    a ``(band+1, N-1)`` array, not a dense ``(N·band, N)`` lstsq (which at
    N = 3000 would be a 700 MB matrix).

    **Gauge: frame 0 at the origin.** The incidence matrix has a one-dimensional
    null space (adding a constant to every position changes nothing), so the
    system is singular until one position is pinned. Dropping row/column 0 both
    fixes the gauge and makes the remaining matrix positive definite. Frame 0
    getting exactly ``(0, 0)`` is therefore structural, not a special case.
    """
    from scipy.linalg import solveh_banded

    n = int(n_frames)
    nv = n - 1
    if nv <= 0:
        return np.zeros((n, 2), np.float64)

    pi = np.asarray(pi, np.int64)
    pj = np.asarray(pj, np.int64)
    c = np.asarray(c, np.float64)
    w = np.asarray(w, np.float64)

    u = int(min(int(band), max(0, nv - 1)))
    ab = np.zeros((u + 1, nv), np.float64)
    diag = np.zeros(nv, np.float64)
    rhs = np.zeros((nv, 2), np.float64)

    inner = pi >= 1
    if np.any(inner):
        np.add.at(diag, pi[inner] - 1, w[inner])
        np.add.at(rhs, pi[inner] - 1, -(w[inner][:, None] * c[inner]))
        d = (pj[inner] - pi[inner]).astype(np.int64)
        np.add.at(ab, (u - d, pj[inner] - 1), -w[inner])
    np.add.at(diag, pj - 1, w)
    np.add.at(rhs, pj - 1, w[:, None] * c)

    ridge = 1e-9 * float(np.max(diag)) if diag.size and np.max(diag) > 0 else 1e-12
    for attempt in range(3):
        ab[u] = diag + ridge
        try:
            return np.vstack([np.zeros((1, 2)), solveh_banded(ab, rhs, lower=False)])
        except Exception as exc:                     # pragma: no cover — rare
            log.debug("[drift] banded solve failed (ridge %g): %s", ridge, exc)
            ridge = max(ridge, 1e-12) * 1e4
    raise ValueError(
        "the drift least-squares system is singular — the measured pairs do not "
        "connect every frame to frame 0 (try a larger `band`)")


def _weights_from(p, pi, pj, c):
    r = p[pj] - p[pi] - c
    resid = np.hypot(r[:, 0], r[:, 1])
    scale = max(1.4826 * float(np.median(resid)), _SCALE_FLOOR_PX)
    return _bisquare(resid, scale), resid


def _solve_positions(n_frames: int, pi, pj, c, band: int, seed=None):
    """IRLS least squares. Returns ``(positions, weights, residual_norms)``.

    The robustness the deleted sharpness gate was approximating comes from HERE,
    and it is a different (better) criterion: a pair is down-weighted because it
    disagrees with what every *other* pair says about the same two frames, not
    because its correlation peak fell below an absolute threshold. With ~``band``
    constraints per unknown, one bad measurement is outvoted; the old chain gave
    it no opposition at all.

    **The seed is not optional in practice.** IRLS only converges to the robust
    optimum from a starting point the outliers have not already captured, and a
    plain unweighted solve is not one: measured on the corrupt-frame fixture, the
    four pairs touching a pure-noise frame dragged its free position 20 px, which
    put ~4 px of residual on the *good* pairs too — so the median residual (the
    robust scale) came out at ~3.5 px and the 4.685σ cut rejected NOTHING. Seeding
    from the streaming median trajectory (:func:`_measure_pass`, which is already
    robust because it takes the median over ~band predecessors) leaves the good
    pairs at ~0 residual and the bad ones at 8-25 px, and the first reweight
    separates them cleanly.
    """
    pi = np.asarray(pi, np.int64)
    pj = np.asarray(pj, np.int64)
    c = np.asarray(c, np.float64)
    m = pi.size
    if m == 0:
        return np.zeros((int(n_frames), 2)), np.zeros(0), np.zeros(0)

    if seed is not None:
        w, _ = _weights_from(np.asarray(seed, np.float64), pi, pj, c)
    else:
        w = np.ones(m, np.float64)
    p = _solve_banded(n_frames, pi, pj, c, w, band)
    for _ in range(int(_IRLS_ITERS)):
        w_new, _ = _weights_from(p, pi, pj, c)
        if np.allclose(w_new, w, atol=1e-6):
            w = w_new
            break
        w = w_new
        p = _solve_banded(n_frames, pi, pj, c, w, band)
    r = p[pj] - p[pi] - c
    return p, w, np.hypot(r[:, 0], r[:, 1])


# ── ROI ──────────────────────────────────────────────────────────────────────

def _validate_roi(roi, full_h: int, full_w: int):
    """Normalise and bounds-check an ``(y0, x0, h, w)`` alignment ROI.

    Rejects rather than clamps. A silently shrunk ROI would correlate on a
    different region than the one the user dragged, and the drift curve would be
    wrong in a way nothing on screen could explain.
    """
    if roi is None:
        return None
    try:
        y0, x0, h, w = (int(v) for v in roi)
    except (TypeError, ValueError):
        raise ValueError(
            f"roi must be (y0, x0, h, w) in pixels; got {roi!r}") from None
    if h < _MIN_ROI or w < _MIN_ROI:
        raise ValueError(
            f"roi is {h}x{w} px; the correlation needs at least "
            f"{_MIN_ROI}x{_MIN_ROI} to locate a peak at all")
    if y0 < 0 or x0 < 0 or y0 + h > full_h or x0 + w > full_w:
        raise ValueError(
            f"roi (y0={y0}, x0={x0}, h={h}, w={w}) falls outside the "
            f"{full_h}x{full_w} frame")
    return (y0, x0, h, w)


def _bin_factor(h: int, w: int, corr_size: int, max_shift=None) -> int:
    """Integer box-mean bin taking the longer edge to ~``corr_size``.

    **Never 1 for a frame that can afford 2**, and that floor is load-bearing
    rather than tidy. The band-pass (:data:`DEFAULT_TILT`) deliberately weights
    the upper part of the correlation grid's band, and at bin 1 that band IS the
    frame's own Nyquist — which in an EM movie is detector noise, not sample.
    Measured on a 256² synthetic movie at the shipped filter: **13.4 px** rms
    unbinned, **0.16 px** at bin 2. Binning 2x costs nothing (the refinement
    ladder recovers the sub-pixel part) and doubles the Poisson SNR per grid
    pixel, so the floor is pure gain wherever the frame is big enough for it.

    Small frames — the test fixtures, and any ROI under ~128 px — stay at bin 1
    because :data:`_MIN_GRID` forbids the alternative. They are correspondingly
    the case where the filter is closest to its safe limit.
    """
    if corr_size is None or int(corr_size) <= 0:
        return 1
    b = max(2, int(max(h, w) // int(corr_size)))
    if max_shift is not None and max_shift > 0:
        # The coarse peak is searched on the grid, so `max_shift/bin` is the
        # search box in grid pixels. Bin it down to nothing and there is no box
        # left to find a peak in — cap the bin instead of the search.
        b = min(b, max(1, int(float(max_shift) // _MIN_SEARCH_GRID)))
    while b > 1 and min(h // b, w // b) < _MIN_GRID:
        b -= 1
    return max(1, b)


# ── public solve ─────────────────────────────────────────────────────────────

def solve_translation(
    data,
    *,
    upsample: int = 8,
    max_shift: float | None = 32.0,
    min_shift: float | None = None,
    band: int = DEFAULT_BAND,
    roi: tuple[int, int, int, int] | None = None,
    apodize: bool | float = True,
    lowpass: float = DEFAULT_LOWPASS,
    highpass: float = DEFAULT_HIGHPASS,
    tilt: float = DEFAULT_TILT,
    corr_size: int = DEFAULT_CORR_SIZE,
    refine_iters: int = 0,
    smooth: float = 0.0,
    device: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    on_shift: Callable[[int, float, float, float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    provenance: dict[str, Any] | None = None,
) -> DriftModel:
    """Solve rigid drift for a frame stack. Returns a :class:`DriftModel`.

    Parameters
    ----------
    data
        A HyperSpy signal (1-D nav, 2-D signal), a 3-D numpy/dask array, or a
        sequence of 2-D frames. Read one frame at a time — never materialised.
    upsample
        Sub-pixel factor, in FULL-frame pixels. ``8`` resolves to 1/8 px, which
        is past the ~0.05 px accuracy floor noise sets on real data. Binning does
        not dilute it: the refinement target is scaled by the bin factor.
    max_shift
        Reject correlation peaks implying a larger RELATIVE shift between a pair,
        in pixels. Guards against a spurious peak from a periodic lattice — the
        failure mode where a crystalline sample locks onto the wrong lattice
        translation. NB this now bounds the shift between two frames at most
        ``band`` apart, not the total excursion, so it can be much tighter than
        the drift over the whole movie.
    min_shift
        Exclude peaks *smaller* than this. Rarely useful now that no frame is
        correlated against a reference containing itself.
    band
        Measure every pair ``(i, j)`` with ``0 < j - i <= band``. The unknowns are
        then ~``band``-times over-determined, which is what lets the solve outvote
        a bad measurement. Larger is more robust and costs proportionally more
        correlations; ``1`` degenerates to a chain and has no redundancy at all.
    roi
        ``(y0, x0, h, w)`` in pixels — correlate on this sub-region only. The
        returned shifts still apply to the WHOLE frame; a translation is a
        translation regardless of which window you measured it in.

        This is not merely a speed switch, it is often the more CORRECT answer.
        Whole-frame correlation averages over everything that moved, so on an
        in-situ movie where the sample is genuinely evolving — particles growing,
        drifting, appearing — the sample's own motion contaminates the estimate of
        the stage's. Restricting to a static, feature-rich landmark (a support
        film edge, a fiducial, a stationary grain) measures the stage and nothing
        else.

        **The ROI is FIXED in frame coordinates**, so the landmark drifts within
        it. That is fine while the drift is small compared with the box, and it is
        why the box wants to be comfortably larger than the total excursion.
    apodize
        Edge taper before transforming. ``True`` uses a Tukey window with
        ``alpha=DEFAULT_TAPER_ALPHA``; a float sets alpha explicitly
        (``1.0`` = full Hann, which is a trap — see :func:`_taper2d`);
        ``False`` disables it.
    lowpass, highpass, tilt
        The band-pass weight ``q**tilt · exp(-(q/lowpass)²) ·
        (1 - exp(-(q/highpass)²))``, with the cut-offs in cycles per
        correlation-grid pixel. See :data:`DEFAULT_LOWPASS`,
        :data:`DEFAULT_HIGHPASS` and :data:`DEFAULT_TILT` — the last one is the
        setting to understand before touching any of them. This REPLACES phase
        whitening; there is no ``normalize`` switch, deliberately.
    corr_size
        Bin the frames so the longer edge is about this many pixels before
        correlating (0 disables binning). See :data:`DEFAULT_CORR_SIZE`.
    refine_iters
        Extra measurement passes (MotionCor2's ``-Iter``). Each re-measures every
        pair with the current solution applied as a Fourier phase ramp, so the
        residual peak sits near the origin and the search is tight. Defaults to 0
        because the upsampled-DFT refinement already evaluates the continuous
        peak — the extra pass costs a second read of the movie and, measured on
        the test stacks and the 2048² benchmark, does not move the error.
    smooth
        Savitzky-Golay smoothing of the SOLVED trajectory; the window length in
        frames (see :func:`_smooth_trajectory` for why not a Gaussian). 0 (off) by
        default, and it stays off because it did not improve any gate: the
        least-squares solve is already an averaging estimator over ~``band``
        measurements per frame, so the jitter a smoother would remove has largely
        been removed already — while a genuine stage jump WOULD get spread over its
        neighbours. It exists for a movie whose per-frame dose is so low that the
        residual is comparable with the drift itself.
    device
        ``None`` auto-selects CUDA/MPS then falls back to torch CPU; ``"numpy"``
        forces the reference path; or an explicit torch device string.
    progress, cancel
        ``progress(done, total)`` is called as frames complete.
        ``cancel()`` returning True aborts; frames not yet reached keep NaN
        shifts, so a cancelled solve is detectable rather than silently partial.
    on_shift
        ``on_shift(index, dy, dx, quality)`` per frame, as each is measured, so a
        UI can draw the drift curve **while** it solves.

        The streamed value is a PROVISIONAL, causal estimate — the median of
        ``position[i] + measured(i → j)`` over the pairs already closed on frame
        *j*. The returned array is the global least-squares solution, which is not
        available until every pair has been measured. They agree closely (the
        provisional estimate uses the same measurements) but they are not the same
        number, and they cannot be: a global solve has no meaningful prefix.

        Called on the solver thread, so a UI implementation must marshal.

    Notes
    -----
    Frame 0 is the gauge and always gets exactly ``(0, 0)``.
    """
    if upsample < 1:
        raise ValueError(f"upsample must be >= 1; got {upsample}")
    band = int(band)
    if band < 1:
        raise ValueError(f"band must be >= 1; got {band}")

    n_frames, get_frame, (full_h, full_w) = frame_source(data)
    crop = _validate_roi(roi, full_h, full_w)
    src_h, src_w = (full_h, full_w) if crop is None else (crop[2], crop[3])
    binf = _bin_factor(src_h, src_w, corr_size, max_shift)
    gh, gw = src_h // binf, src_w // binf
    ops = _resolve_ops(device)

    alpha = (DEFAULT_TAPER_ALPHA if apodize is True
             else (0.0 if apodize is False else float(apodize)))

    from spyde.device_lock import accelerator_lock

    # MPS is not thread-safe and every torch user in the process shares ONE lock
    # (CLAUDE.md § GPU Computing). A null context off MPS, so CUDA keeps its
    # stream concurrency.
    with accelerator_lock(ops.device):
        window = _taper2d(ops, gh, gw, alpha) if alpha > 0 else None
        weight = _bandpass(ops, gh, gw, lowpass, highpass, tilt)
        mask = _shift_mask(ops, gh, gw, max_shift, min_shift, binf)
        # The refinement pass searches only a few grid pixels around the
        # prediction — that is the point of predicting.
        tight = _shift_mask(ops, gh, gw, 3.0 * binf, None, binf)

        def spectrum(i: int):
            raw = get_frame(i)
            if crop is not None:
                y0, x0, ch, cw = crop
                raw = raw[y0:y0 + ch, x0:x0 + cw]
            f = ops.to_backend(raw)
            if binf > 1:
                f = ops.bin_mean(f, binf)
            # MEAN SUBTRACT BEFORE WINDOWING, and it is load-bearing for a plain
            # (unwhitened) cross-correlation. The taper is FIXED in frame
            # coordinates, so multiplying a frame with a large DC pedestal by it
            # stamps the same bright-to-dark border ramp into every frame — a
            # feature that does not move, correlates with itself at zero lag, and
            # biases every measurement toward zero. Phase correlation hid this by
            # discarding magnitude; plain correlation does not, and it showed up
            # immediately as a 0.125 px bias on whole-pixel synthetic truth and a
            # 2 px error on a tapered 56x64 ROI. Removing the mean first leaves the
            # taper acting on zero-mean structure, and both go back to exact.
            f = f - ops.mean(f)
            if window is not None:
                f = f * window
            return ops.fft2(f) * weight

        # Sub-batch so the (B, gh, gw) product stays under the byte ceiling even
        # with binning disabled on a 4096² movie.
        per_pair = max(1, gh * gw * 8)
        max_b = max(1, int(_MAX_BATCH_BYTES // per_pair))

        state = _PassState(n_frames, band)
        passes = 1 + max(0, int(refine_iters))
        positions = None
        w_pairs = np.zeros(0)
        resid = np.zeros(0)
        for it in range(passes):
            predictor = positions if it > 0 else None
            done, provisional = _measure_pass(
                ops, spectrum, state, n_frames, band, mask, tight, upsample,
                binf, max_b,
                predict_from=predictor,
                progress=progress if it == 0 else None,
                on_shift=on_shift if it == 0 else None,
                cancel=cancel,
            )
            state.reached = done
            if state.count == 0:
                break
            positions, w_pairs, resid = _solve_positions(
                n_frames, state.pi[:state.count], state.pj[:state.count],
                state.c[:state.count], band,
                seed=positions if positions is not None else provisional)
            if done < n_frames:
                break

    shifts = np.full((n_frames, 2), np.nan, dtype=np.float32)
    per_frame = np.full((n_frames,), np.nan, dtype=np.float32)
    n_pairs = int(state.count)
    ls_rms = float("nan")
    ls_max = float("nan")
    down = 0
    if n_pairs and positions is not None:
        reach = int(state.reached)
        shifts[:reach] = positions[:reach].astype(np.float32)
        if smooth and float(smooth) > 0:
            shifts[:reach] = _smooth_trajectory(shifts[:reach], float(smooth))
        pi = state.pi[:n_pairs]
        pj = state.pj[:n_pairs]
        acc = np.zeros(n_frames, np.float64)
        cnt = np.zeros(n_frames, np.int64)
        np.add.at(acc, pi, resid ** 2)
        np.add.at(acc, pj, resid ** 2)
        np.add.at(cnt, pi, 1)
        np.add.at(cnt, pj, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_frame[:] = np.where(cnt > 0, np.sqrt(acc / np.maximum(cnt, 1)),
                                    np.nan)
        ls_rms = float(np.sqrt(np.mean(resid ** 2)))
        ls_max = float(np.max(resid))
        down = int(np.count_nonzero(w_pairs < 0.5))
    elif n_frames:
        shifts[0] = (0.0, 0.0)
        per_frame[0] = 0.0

    params = {
        "upsample": int(upsample),
        "max_shift": None if max_shift is None else float(max_shift),
        "min_shift": None if min_shift is None else float(min_shift),
        "band": int(band),
        "apodize": float(alpha),
        "lowpass": float(lowpass),
        "highpass": float(highpass),
        "tilt": float(tilt),
        "corr_size": int(corr_size),
        "corr_bin": int(binf),
        "corr_grid": [int(gh), int(gw)],
        "refine_iters": int(max(0, refine_iters)),
        "smooth": float(smooth),
        "backend": ops.name,
        "n_frames": int(n_frames),
        "n_pairs": n_pairs,
        "pairs_downweighted": down,
        "ls_residual_px": ls_rms,
        "ls_residual_max_px": ls_max,
        "frame_shape": [int(full_h), int(full_w)],
        "roi": None if crop is None else [int(v) for v in crop],
    }
    return DriftModel(
        shifts=shifts,
        kind="rigid",
        reference="least-squares",
        residuals=per_frame,
        params=params,
        provenance=provenance,
    )


class _PassState:
    """Growable pair store, reused across refinement passes.

    Preallocated to ``n·band`` because that is the exact upper bound and the
    arrays are tiny (a 3000-frame movie at band 12 is 36k pairs = 0.9 MB),
    which keeps the measurement loop allocation-free.
    """

    __slots__ = ("pi", "pj", "c", "count", "reached")

    def __init__(self, n_frames: int, band: int = DEFAULT_BAND):
        cap = max(1, int(n_frames)) * max(1, int(band))
        self.pi = np.zeros(cap, np.int64)
        self.pj = np.zeros_like(self.pi)
        self.c = np.zeros((self.pi.size, 2), np.float64)
        self.count = 0
        self.reached = 0

    def reset(self) -> None:
        self.count = 0

    def _grow(self, extra: int) -> None:
        need = self.count + extra
        if need <= self.pi.size:
            return
        cap = max(need, self.pi.size * 2)
        for name in ("pi", "pj"):
            old = getattr(self, name)
            new = np.zeros(cap, old.dtype)
            new[:self.count] = old[:self.count]
            setattr(self, name, new)
        old_c = self.c
        new_c = np.zeros((cap, 2), np.float64)
        new_c[:self.count] = old_c[:self.count]
        self.c = new_c

    def add(self, i, j, c) -> None:
        n = len(i)
        self._grow(n)
        s = slice(self.count, self.count + n)
        self.pi[s] = i
        self.pj[s] = j
        self.c[s] = c
        self.count += n


def _measure_pass(ops, spectrum, state: _PassState, n_frames: int, band: int,
                  mask, tight, upsample, binf, max_b, *, predict_from=None,
                  progress=None, on_shift=None, cancel=None):
    """Stream the movie once, measuring every banded pair.

    Returns ``(frames_reached, provisional)`` — the second is the causal median
    trajectory, which is both what ``on_shift`` streams and (crucially) the robust
    seed the IRLS solve needs; see :func:`_solve_positions`.

    The ring holds at most ``band`` spectra, so the resident set is
    ``band + 1`` BINNED spectra plus the one full-resolution frame being read —
    bounded whatever the movie length (CLAUDE.md Memory-Safety rule). Every pair
    closing on frame *j* shares the same moving spectrum, so they go through one
    batched product and ONE inverse FFT.
    """
    state.reset()
    ring: list[tuple[int, Any]] = []
    prov = np.zeros((n_frames, 2), np.float64)
    reached = 0

    for j in range(n_frames):
        if cancel is not None and cancel():
            log.info("[drift] cancelled at frame %d/%d", j, n_frames)
            break
        spec = spectrum(j)
        if ring:
            idx = np.array([r[0] for r in ring], np.int64)
            cand = []
            qual_all = []
            for lo in range(0, len(ring), max_b):
                sub = ring[lo:lo + max_b]
                pred = None
                if predict_from is not None:
                    rel = (predict_from[j] - predict_from[[r[0] for r in sub]])
                    pred = (rel / float(binf)).astype(np.float64)
                sh, q = _correlate_batch(
                    ops, [r[1] for r in sub], spec, mask, upsample, binf,
                    predict=pred, tight=tight)
                cand.append(sh)
                qual_all.append(q)
            shifts = np.concatenate(cand, axis=0)
            quality = np.concatenate(qual_all, axis=0)
            state.add(idx, np.full(idx.size, j, np.int64), shifts)
            # Provisional, causal estimate for the live trace: the MEDIAN of what
            # each already-placed predecessor says about this frame. A median, not
            # a mean, so one bad predecessor cannot drag the curve.
            prov[j] = np.median(prov[idx] + shifts, axis=0)
            q_rep = float(np.median(quality))
        else:
            prov[j] = 0.0
            q_rep = float("inf")

        ring.append((j, spec))
        if len(ring) > band:
            ring.pop(0)
        reached = j + 1

        if on_shift is not None:
            on_shift(j, float(prov[j, 0]), float(prov[j, 1]), q_rep)
        if progress is not None:
            progress(reached, n_frames)
    return reached, prov


def _smooth_trajectory(shifts: np.ndarray, window: float) -> np.ndarray:
    """Savitzky-Golay smoothing along the frame axis, NaN-safe. Off by default.

    **Savitzky-Golay and not a Gaussian**, because a stage trajectory is a smooth
    RAMP and a Gaussian kernel is a low-pass that does not preserve one: with any
    edge policy it flattens both ends, and measured on a straight 0.8 px/frame
    ramp it pulled the first and last points 0.46 px off truth — a systematic
    error introduced by the thing meant to reduce error. A quadratic SG filter
    reproduces any locally-quadratic trend exactly, including at the boundaries,
    so it removes jitter without bending the curve.

    The gauge is re-applied afterwards: smoothing moves frame 0 like any other
    point, and frame 0 is the origin BY DEFINITION (see :func:`_solve_banded`).
    """
    from scipy.signal import savgol_filter

    out = np.array(shifts, np.float64, copy=True)
    ok = np.isfinite(out).all(axis=1)
    n = int(ok.sum())
    win = int(round(max(3.0, float(window))))
    if win % 2 == 0:
        win += 1
    if n < 5 or win < 5 or win > n:
        return shifts
    for axis in (0, 1):
        out[ok, axis] = savgol_filter(out[ok, axis], win, 2, mode="interp")
    first = np.argmax(ok) if ok.any() else 0
    out[ok] -= out[first]
    return out.astype(np.float32)

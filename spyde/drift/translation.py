"""
translation.py — rigid (translation-only) drift solve. Plan step A1.

Algorithm: FFT phase correlation with a **running Fourier average** reference and
Guizar-Sicairos matrix-multiply DFT upsampling for the sub-pixel peak.

Three things here are deliberate and worth reading before changing them.

**1. It streams.** One frame is resident at a time (plus the accumulated reference
FFT, which is frame-sized). A 3000 × 4096² movie is tens of GB; nothing here ever
holds more than a few hundred MB. This is the CLAUDE.md Memory-Safety rule.

**2. The reference is accumulated in FOURIER space, aligned by a phase ramp.**
To add frame *i* to the running average *already aligned*, we multiply its FFT by
``exp(-2πi(dy·fy + dx·fx))`` rather than resampling the frame and re-transforming.
A translation is exactly a phase ramp in the Fourier domain, so this is not an
approximation — it is free, it is exact even for sub-pixel shifts, and it avoids
the interpolation blur that resample-then-average would accumulate over thousands
of frames. That blur is the reason a naive running average degrades as the stack
gets longer.

**3. Sub-pixel refinement is a small matmul, not a padded inverse FFT.** Zero-
padding the cross-correlation to get ``1/upsample`` resolution costs an FFT of
``(H·u, W·u)`` — for u=8 on a 4096² frame that is a 32768² transform. The
matrix-multiply DFT evaluates the correlation only on the ~12×12 window around
the coarse peak, which is what the refinement actually needs.

The torch and numpy paths run the *same* algorithm through a small operator
adapter, so ``test_drift_translation.py`` can assert they agree bit-closely — that
parity test is what protects the GPU path, since it has no independent reference.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np

from spyde.drift.frames import frame_source
from spyde.drift.model import DriftModel

log = logging.getLogger(__name__)

# Guizar-Sicairos: the refinement window spans 1.5 upsampled pixels either side of
# the coarse peak. Matches skimage's `phase_cross_correlation` so the two agree.
_UPSAMPLED_REGION_FACTOR = 1.5

# Magnitude FLOOR for phase normalisation — NOT an additive epsilon.
#
# Pure phase correlation divides the cross-power spectrum by its own magnitude,
# which is only meaningful where there is signal. Bins whose magnitude is
# numerically zero must be left alone; dividing them by a tiny epsilon amplifies
# rounding noise to UNIT magnitude, and since there are far more empty bins than
# populated ones, that noise then dominates the inverse transform.
#
# This is not theoretical. With an additive `1e-12` the solver recovered the
# synthetic particle movie's drift to 25 px (worse than not correcting at all)
# as soon as apodisation was enabled — because windowing concentrates spectral
# energy and pushes many more bins down into the numerical floor. With the floor
# below it: 0.06 px. Matches skimage's `100 * finfo(float32).eps`.
_PHASE_FLOOR = 100.0 * float(np.finfo(np.float32).eps)      # ~1.19e-5

# A frame whose correlation peak is weaker than this fraction of the running
# MEDIAN peak is kept out of the accumulated reference.
#
# The running-average reference exists to be robust to one bad frame, but folding
# every frame in unconditionally does the opposite: a dropped / blanked / saturated
# frame has a broadband spectrum, so after phase normalisation it contributes as
# much to the reference as a good frame and drags every subsequent registration
# with it. Measured on a 5-frame stack with one frame replaced by pure noise, the
# two frames AFTER the bad one came back ~3.9 px wrong; with this rejection they
# are correct and only the bad frame itself is wrong.
#
# Both constants come from measurement, not taste. Peak strength relative to the
# running median, measured across four stacks:
#
#   worst NATURAL frame (clean sub-pixel stack, erratic peaks)   0.388
#   a frame replaced by pure noise                              0.007
#
# So there is a ~50x gap to put a threshold in, and 0.25 sits inside it with margin
# both ways. 0.5 was tried first and produced a FALSE rejection on the clean
# sub-pixel stack — which is why this is not simply "half".
#
# _REJECT_MIN_SAMPLES is 1, not 3, and that is deliberate: a short stack cannot
# afford a warm-up. On a 5-frame stack the bad frame arrives before three good ones
# have been seen, so a 3-sample warm-up let it into the reference and the rule never
# fired (measured: 0 rejections, and the two frames after it came back 3.9 px wrong).
# With a 1-sample warm-up those two frames are recovered EXACTLY.
#
# The asymmetry justifies being aggressive: keeping a good frame OUT of the
# reference only slows the averaging, while letting a bad frame IN corrupts every
# registration after it. A rejected frame still gets its own shift reported.
#
# Windowing the median over the last N accepted frames was tried and made NO
# difference at N=3, 5 or unbounded — the natural decay in peak strength as the
# reference averages more frames is not steep enough to matter. Don't add it back.
_REJECT_FRACTION = 0.25
_REJECT_MIN_SAMPLES = 1

# Smallest alignment ROI worth correlating. Below roughly this the upsampled
# refinement window (1.5 x upsample, so 12 px at the default) approaches the box
# itself and the peak has nowhere to sit.
_MIN_ROI = 16


# ── operator adapters ────────────────────────────────────────────────────────
# The algorithm below is written once against this interface. `_TorchOps` is the
# production path; `_NumpyOps` is the reference the parity test pins it against.

class _NumpyOps:
    name = "numpy"

    def __init__(self, device=None):
        self.device = None

    def to_backend(self, a):
        return np.asarray(a, dtype=np.float32)

    def fft2(self, a):
        return np.fft.fft2(a).astype(np.complex64)

    def ifft2(self, a):
        return np.fft.ifft2(a)

    def conj(self, a):
        return np.conj(a)

    def abs(self, a):
        return np.abs(a)

    def clamp_min(self, a, floor):
        return np.maximum(a, floor)

    def fftfreq(self, n):
        return np.fft.fftfreq(n).astype(np.float32)

    def arange(self, n):
        return np.arange(n, dtype=np.float32)

    def exp(self, a):
        return np.exp(a)

    def masked_argmax(self, mag, mask):
        m = np.where(mask, mag, -np.inf)
        flat = int(np.argmax(m))
        return divmod(flat, mag.shape[1])

    def argmax2d(self, mag):
        flat = int(np.argmax(mag))
        return divmod(flat, mag.shape[1])

    def tensordot_last(self, kernel, data):
        """``kernel @ data`` contracting kernel's axis 1 with data's LAST axis."""
        return np.tensordot(kernel, data, axes=(1, -1))

    def scalar(self, a) -> float:
        return float(a)

    def mean_abs(self, a) -> float:
        return float(np.mean(np.abs(a)))

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
        return t.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=self.device)

    def fft2(self, a):
        return self._torch.fft.fft2(a).to(self._torch.complex64)

    def ifft2(self, a):
        return self._torch.fft.ifft2(a)

    def conj(self, a):
        return self._torch.conj(a)

    def abs(self, a):
        return self._torch.abs(a)

    def clamp_min(self, a, floor):
        return self._torch.clamp(a, min=float(floor))

    def fftfreq(self, n):
        return self._torch.fft.fftfreq(n, device=self.device, dtype=self._torch.float32)

    def arange(self, n):
        return self._torch.arange(n, device=self.device, dtype=self._torch.float32)

    def exp(self, a):
        return self._torch.exp(a)

    def masked_argmax(self, mag, mask):
        t = self._torch
        m = mag.masked_fill(~mask, float("-inf"))
        flat = int(t.argmax(m.reshape(-1)).item())
        return divmod(flat, mag.shape[1])

    def argmax2d(self, mag):
        t = self._torch
        flat = int(t.argmax(mag.reshape(-1)).item())
        return divmod(flat, mag.shape[1])

    def tensordot_last(self, kernel, data):
        return self._torch.tensordot(kernel, data, dims=([1], [data.ndim - 1]))

    def scalar(self, a) -> float:
        return float(a.item()) if hasattr(a, "item") else float(a)

    def mean_abs(self, a) -> float:
        return float(self._torch.mean(self._torch.abs(a)).item())

    def to_numpy(self, a):
        return a.detach().cpu().numpy()


def _resolve_ops(device: str | None):
    """Pick the backend: ``cuda`` > ``mps`` > **torch CPU** > numpy.

    **torch CPU beats numpy even with no GPU**, and by a lot — measured on this
    box, 120 × 512² frames at upsample=8::

        numpy      6.57 s    18 frames/s
        torch cpu  0.86 s   139 frames/s   (7.7x)
        torch cuda 0.42 s   284 frames/s   (16x)

    The reason is mundane: ``np.fft.fft2`` is single-threaded, ``torch.fft.fft2``
    uses every core. A per-frame FFT is the entire cost of this solver, so that
    one difference is the whole gap. An earlier version of this function preferred
    numpy on CPU-only machines on the assumption that torch's dispatch overhead
    would dominate at one frame at a time; that assumption was wrong by 7.7x.

    numpy is kept as an **explicitly** selectable reference path (``device=
    "numpy"``), which is what the backend-parity test pins the torch path against.
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


# ── windows and masks (built once per solve) ──────────────────────────────────

#: Default Tukey taper fraction. 0.25 tapers the outer ~12.5% at each edge and
#: leaves the middle 75% at unit weight. See :func:`_taper2d` for why this is
#: NOT 1.0 (a full Hann window).
DEFAULT_TAPER_ALPHA = 0.25


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
    """
    win = _tukey1d(h, alpha)[:, None] * _tukey1d(w, alpha)[None, :]
    return ops.to_backend(win)


def _shift_mask(ops, h: int, w: int, max_shift: float | None,
                min_shift: float | None):
    """Which cross-correlation bins are admissible shifts.

    The correlation is un-shifted, so bin ``k`` means shift ``k`` for
    ``k <= n//2`` and ``k - n`` above that. Both bounds are separable, so this is
    an outer product of two 1-D masks — built once and reused for every frame.
    """
    def axis_mask(n: int):
        k = np.arange(n)
        s = np.where(k > n // 2, k - n, k).astype(np.float64)
        ok = np.ones(n, dtype=bool)
        if max_shift is not None:
            ok &= np.abs(s) <= float(max_shift)
        return ok, s

    oky, sy = axis_mask(h)
    okx, sx = axis_mask(w)
    mask = oky[:, None] & okx[None, :]
    if min_shift is not None:
        # Exclude the near-zero-shift core. Useful when the reference already
        # contains this frame (a running average does), because the trivial
        # self-correlation peak at the origin can then outrank the real one.
        r = np.hypot(sy[:, None], sx[None, :])
        mask &= r >= float(min_shift)
    if not mask.any():
        raise ValueError(
            "max_shift/min_shift exclude every possible shift "
            f"(max_shift={max_shift}, min_shift={min_shift}, frame={h}x{w})"
        )
    if ops.name == "torch":
        return ops._torch.as_tensor(mask, device=ops.device)
    return mask


# ── the correlation ──────────────────────────────────────────────────────────

def _upsampled_dft(ops, data, region_size: int, upsample: float, offsets):
    """Evaluate the inverse DFT of *data* on a small upsampled window.

    Mirrors ``skimage.registration._masked_phase_cross_correlation._upsampled_dft``:
    one kernel matmul per axis, contracting the last axis each time.
    """
    out = data
    shape = tuple(data.shape)
    for axis in (1, 0):                       # last axis first
        n = shape[axis]
        off = offsets[axis]
        # NOTE the /upsample: this is `np.fft.fftfreq(n, upsample)` — frequencies
        # scaled to the UPSAMPLED grid. Without it the window still evaluates and
        # still finds a peak, but at 1/upsample of the intended resolution, so
        # every recovered shift lands on a multiple of 1/upsample and the
        # refinement silently does nothing.
        freq = ops.fftfreq(n) / float(upsample)
        kern = (ops.arange(region_size).reshape(region_size, 1) - float(off)) * \
               freq.reshape(1, n)
        if ops.name == "torch":
            kern = ops.exp(ops._torch.complex(
                ops._torch.zeros_like(kern), -2.0 * math.pi * kern))
        else:
            kern = np.exp(-2j * math.pi * kern).astype(np.complex64)
        out = ops.tensordot_last(kern, out)
    return out


def _peak_shift(ops, ref_fft, mov_fft, mask, upsample: float,
                normalize: bool) -> tuple[float, float, float]:
    """Return ``(dy, dx, sharpness)`` registering *mov* onto *ref*.

    Sign matches ``skimage.registration.phase_cross_correlation`` and
    ``scipy.ndimage.shift``: the result is the correction to ADD to the moving
    frame (see :mod:`spyde.drift.model`).
    """
    product = ref_fft * ops.conj(mov_fft)
    if normalize:
        # Phase correlation proper: discard magnitude, keep only phase. Gives a
        # far sharper peak than plain cross-correlation on images whose spectra
        # are dominated by low frequencies, which every real micrograph is.
        # The divisor is FLOORED, not offset — see _PHASE_FLOOR.
        product = product / ops.clamp_min(ops.abs(product), _PHASE_FLOOR)

    cc = ops.ifft2(product)
    mag = ops.abs(cc)
    py, px = ops.masked_argmax(mag, mask)

    h, w = int(mag.shape[0]), int(mag.shape[1])
    peak = ops.scalar(mag[py, px])
    baseline = ops.mean_abs(cc)
    sharpness = float(peak / baseline) if baseline > 0 else float("nan")

    dy = float(py - h) if py > h // 2 else float(py)
    dx = float(px - w) if px > w // 2 else float(px)

    if upsample and upsample > 1:
        u = float(upsample)
        dy = round(dy * u) / u
        dx = round(dx * u) / u
        region = int(math.ceil(u * _UPSAMPLED_REGION_FACTOR))
        dftshift = float(region // 2)
        offsets = (dftshift - dy * u, dftshift - dx * u)
        fine = _upsampled_dft(ops, ops.conj(product), region, u, offsets)
        fmag = ops.abs(fine)
        my, mx = ops.argmax2d(fmag)
        dy += (my - dftshift) / u
        dx += (mx - dftshift) / u

    return dy, dx, sharpness


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


def _accept_into_reference(sharpness: float, accepted: list[float],
                           enabled: bool) -> bool:
    """Whether this frame is credible enough to join the running reference.

    Always True until there are :data:`_REJECT_MIN_SAMPLES` accepted frames to
    take a median over — with nothing to compare against, rejecting would just be
    guessing. See :data:`_REJECT_FRACTION`.
    """
    if not enabled or len(accepted) < _REJECT_MIN_SAMPLES:
        return True
    if not math.isfinite(sharpness):
        return False
    return sharpness >= _REJECT_FRACTION * float(np.median(accepted))


def _phase_ramp(ops, h: int, w: int, dy: float, dx: float):
    """FFT multiplier that translates a frame by ``(dy, dx)`` exactly.

    ``F{f(y - dy, x - dx)}(k) = exp(-2πi(dy·fy + dx·fx))·F{f}(k)``. This is why
    the running reference needs no resampling — see the module docstring.
    """
    fy = ops.fftfreq(h).reshape(h, 1)
    fx = ops.fftfreq(w).reshape(1, w)
    ph = dy * fy + dx * fx
    if ops.name == "torch":
        return ops.exp(ops._torch.complex(
            ops._torch.zeros_like(ph), -2.0 * math.pi * ph))
    return np.exp(-2j * math.pi * ph).astype(np.complex64)


# ── public solve ─────────────────────────────────────────────────────────────

def solve_translation(
    data,
    *,
    upsample: int = 8,
    max_shift: float | None = 32.0,
    min_shift: float | None = None,
    reference: str = "running",
    roi: tuple[int, int, int, int] | None = None,
    apodize: bool | float = True,
    normalize: bool = True,
    reject_outliers: bool = True,
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
        Sub-pixel factor. ``8`` resolves to 1/8 px, which is well past the
        ~0.05 px accuracy floor set by noise on real data.
    max_shift
        Reject correlation peaks implying a larger per-frame shift, in pixels.
        Guards against a spurious peak from a periodic lattice — the failure mode
        where a crystalline sample locks onto the wrong lattice translation and
        the drift curve jumps by exactly one lattice spacing.
    min_shift
        Exclude peaks *smaller* than this. Off by default; see
        :func:`_shift_mask`.
    reference
        ``"running"`` — running Fourier average (default, robust to one bad
        frame); ``"sequential"`` — register to the previous frame and accumulate
        (handles large excursions, accumulates error); ``"first"`` or
        ``"fixed:<i>"`` — one fixed reference frame.
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
        else. It is also how a user can rescue a dataset where the field of view
        is mostly featureless.

        **The ROI is FIXED in frame coordinates**, so the landmark drifts within
        it. That is fine while the drift is small compared with the box, and it is
        why the box wants to be comfortably larger than the total excursion —
        the caret's preview exists so this is judged by eye rather than guessed.
        A box smaller than the drift will lose the landmark and the solve will
        wander.
    apodize
        Edge taper before transforming. ``True`` uses a Tukey window with
        ``alpha=DEFAULT_TAPER_ALPHA``; a float sets alpha explicitly
        (``1.0`` = full Hann, which is a trap — see :func:`_taper2d`);
        ``False`` disables it.
    normalize
        True phase correlation (unit-magnitude spectrum). Sharper peak.
    reject_outliers
        ``running`` mode only: keep a frame out of the accumulated reference when
        its correlation peak is not credible (see :data:`_REJECT_FRACTION`). Its
        own shift is still reported — only the reference is protected. This is
        what makes "robust to one bad frame" true rather than aspirational.
    device
        ``None`` auto-selects CUDA/MPS then falls back to numpy; ``"numpy"``
        forces the reference path; or an explicit torch device string.
    progress, cancel
        ``progress(done, total)`` is called as frames complete.
        ``cancel()`` returning True aborts; frames not yet reached keep NaN
        shifts, so a cancelled solve is detectable rather than silently partial.
    on_shift
        ``on_shift(index, dy, dx, sharpness)`` per frame, as each is solved.

        This exists so a UI can draw the drift curve **while** it solves, which
        ``progress`` cannot support: it carries only a count, and the shift array
        is solver-local until the return. Splitting the solve into chunks and
        concatenating would not be equivalent either — the running Fourier
        reference accumulates across the whole stack, so a restarted solve gives a
        different (worse) answer. A callback is the only way to stream the trace
        without changing the result.

        Called on the solver thread, so a UI implementation must marshal.

    Notes
    -----
    Frame 0 is the origin by definition and always gets ``(0, 0)``.
    """
    if reference not in ("running", "sequential", "first") and \
            not reference.startswith("fixed:"):
        raise ValueError(
            f"unknown reference {reference!r}; expected 'running', 'sequential', "
            "'first' or 'fixed:<index>'"
        )
    if upsample < 1:
        raise ValueError(f"upsample must be >= 1; got {upsample}")

    n_frames, get_frame, (full_h, full_w) = frame_source(data)
    crop = _validate_roi(roi, full_h, full_w)
    h, w = (full_h, full_w) if crop is None else (crop[2], crop[3])
    ops = _resolve_ops(device)

    shifts = np.full((n_frames, 2), np.nan, dtype=np.float32)
    sharp = np.full((n_frames,), np.nan, dtype=np.float32)

    from spyde.device_lock import accelerator_lock

    # MPS is not thread-safe and every torch user in the process shares ONE lock
    # (CLAUDE.md § GPU Computing). A null context off MPS, so CUDA keeps its
    # stream concurrency. Held across the solve rather than per-frame: the solve
    # runs on a worker thread and per-frame acquire/release would be pure
    # overhead at thousands of frames.
    with accelerator_lock(ops.device):
        alpha = (DEFAULT_TAPER_ALPHA if apodize is True
                 else (0.0 if apodize is False else float(apodize)))
        window = _taper2d(ops, h, w, alpha) if alpha > 0 else None
        mask = _shift_mask(ops, h, w, max_shift, min_shift)

        def frame_fft(i: int):
            raw = get_frame(i)
            if crop is not None:
                y0, x0, ch, cw = crop
                raw = raw[y0:y0 + ch, x0:x0 + cw]
            f = ops.to_backend(raw)
            if window is not None:
                f = f * window
            return ops.fft2(f)

        fixed_index = 0
        if reference.startswith("fixed:"):
            fixed_index = int(reference.split(":", 1)[1])
            if not 0 <= fixed_index < n_frames:
                raise ValueError(
                    f"fixed reference index {fixed_index} outside 0..{n_frames - 1}"
                )

        first = frame_fft(fixed_index if reference.startswith("fixed:") else 0)
        shifts[0] = (0.0, 0.0)
        sharp[0] = np.inf if n_frames else np.nan
        if on_shift is not None and n_frames:
            on_shift(0, 0.0, 0.0, float("inf"))

        ref_fft = first          # running accumulator / fixed reference
        ref_count = 1
        prev_fft = first         # sequential mode
        cumulative = np.zeros(2, dtype=np.float64)
        accepted_sharp: list[float] = []   # peak strengths folded into the reference
        rejected = 0

        if progress is not None:
            progress(1, n_frames)

        for i in range(1, n_frames):
            if cancel is not None and cancel():
                log.info("[drift] cancelled at frame %d/%d", i, n_frames)
                break

            mov = frame_fft(i)

            if reference == "sequential":
                dy, dx, s = _peak_shift(ops, prev_fft, mov, mask, upsample, normalize)
                cumulative += (dy, dx)
                shifts[i] = cumulative
                prev_fft = mov
            else:
                dy, dx, s = _peak_shift(ops, ref_fft, mov, mask, upsample, normalize)
                shifts[i] = (dy, dx)
                if reference == "running" and _accept_into_reference(
                        s, accepted_sharp, reject_outliers):
                    # Fold the ALIGNED frame in via a phase ramp — exact, and no
                    # resampling blur accumulates over the stack.
                    aligned = mov * _phase_ramp(ops, h, w, dy, dx)
                    ref_fft = (ref_fft * ref_count + aligned) / (ref_count + 1)
                    ref_count += 1
                    accepted_sharp.append(s)
                elif reference == "running":
                    rejected += 1
                    log.debug("[drift] frame %d kept out of the reference "
                              "(peak %.2f vs median %.2f)", i, s,
                              float(np.median(accepted_sharp)))
            sharp[i] = s

            if on_shift is not None:
                on_shift(i, float(shifts[i, 0]), float(shifts[i, 1]), float(s))
            if progress is not None:
                progress(i + 1, n_frames)

    params = {
        "upsample": int(upsample),
        "max_shift": None if max_shift is None else float(max_shift),
        "min_shift": None if min_shift is None else float(min_shift),
        "reference": reference,
        "apodize": float(alpha),
        "normalize": bool(normalize),
        "reject_outliers": bool(reject_outliers),
        "rejected_from_reference": int(rejected),
        "backend": ops.name,
        "n_frames": int(n_frames),
        "frame_shape": [int(full_h), int(full_w)],
        "roi": None if crop is None else [int(v) for v in crop],
    }
    return DriftModel(
        shifts=shifts,
        kind="rigid",
        reference=reference,
        residuals=sharp,
        params=params,
        provenance=provenance,
    )

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

def _hann2d(ops, h: int, w: int):
    """Separable Hann window.

    Without apodisation a feature entering or leaving at the frame edge correlates
    against the *border discontinuity* rather than the sample, which reads as a
    spurious jump in the drift curve exactly when something interesting is moving
    through the field of view.
    """
    n = ops.arange(h)
    m = ops.arange(w)
    wy = 0.5 - 0.5 * _cos(ops, 2.0 * math.pi * n / max(1, h - 1))
    wx = 0.5 - 0.5 * _cos(ops, 2.0 * math.pi * m / max(1, w - 1))
    return wy.reshape(h, 1) * wx.reshape(1, w)


def _cos(ops, a):
    if ops.name == "torch":
        return ops._torch.cos(a)
    return np.cos(a)


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
        eps = 1e-12
        product = product / (ops.abs(product) + eps)

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
    apodize: bool = True,
    normalize: bool = True,
    device: str | None = None,
    progress: Callable[[int, int], None] | None = None,
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
    apodize
        Apply a Hann window before transforming. See :func:`_hann2d`.
    normalize
        True phase correlation (unit-magnitude spectrum). Sharper peak.
    device
        ``None`` auto-selects CUDA/MPS then falls back to numpy; ``"numpy"``
        forces the reference path; or an explicit torch device string.
    progress, cancel
        ``progress(done, total)`` is called as frames complete.
        ``cancel()`` returning True aborts; frames not yet reached keep NaN
        shifts, so a cancelled solve is detectable rather than silently partial.

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

    n_frames, get_frame, (h, w) = frame_source(data)
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
        window = _hann2d(ops, h, w) if apodize else None
        mask = _shift_mask(ops, h, w, max_shift, min_shift)

        def frame_fft(i: int):
            f = ops.to_backend(get_frame(i))
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

        ref_fft = first          # running accumulator / fixed reference
        ref_count = 1
        prev_fft = first         # sequential mode
        cumulative = np.zeros(2, dtype=np.float64)

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
                if reference == "running":
                    # Fold the ALIGNED frame in via a phase ramp — exact, and no
                    # resampling blur accumulates over the stack.
                    aligned = mov * _phase_ramp(ops, h, w, dy, dx)
                    ref_fft = (ref_fft * ref_count + aligned) / (ref_count + 1)
                    ref_count += 1
            sharp[i] = s

            if progress is not None:
                progress(i + 1, n_frames)

    params = {
        "upsample": int(upsample),
        "max_shift": None if max_shift is None else float(max_shift),
        "min_shift": None if min_shift is None else float(min_shift),
        "reference": reference,
        "apodize": bool(apodize),
        "normalize": bool(normalize),
        "backend": ops.name,
        "n_frames": int(n_frames),
        "frame_shape": [int(h), int(w)],
    }
    return DriftModel(
        shifts=shifts,
        kind="rigid",
        reference=reference,
        residuals=sharp,
        params=params,
        provenance=provenance,
    )

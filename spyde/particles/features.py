"""
features.py — the torch feature stack the learned engines classify. Plan step B2.

ParticleSpy's ``trainable_parameters`` set (which is Weka Trainable Segmentation's
set, which is ilastik's set): gaussian blur, difference-of-gaussians, median /
minimum / maximum rank filters, Sobel gradient magnitude, Hessian eigenvalues,
Laplacian, and membrane projections. Every channel is computed in torch, so the
same code runs on CPU, CUDA and Apple-MPS, and the classifier head
(:mod:`spyde.particles.scribble`) never leaves the device.

Three properties are load-bearing and are what the implementation is shaped
around.

**1. One pass over the frame, not one pass per feature.** The gaussian pyramid is
computed once (separably — two 1-D convolutions per sigma instead of one 2-D one)
and then *every* other family is derived from it: difference-of-gaussians is a
subtraction of two cached blurs, the Laplacian is the trace of the Hessian's own
second derivatives, the two Hessian eigenvalues come from the same three
derivative images, and median/minimum/maximum come from one unfolded window each.
:class:`_Pass` is the memo that makes that true — a channel never triggers work
another channel already did. Measured on a 512² float32 frame, CPU, default spec
(36 channels): **54 ms shared vs 156 ms** when each channel recomputes its own
intermediates.

**2. The fine scales are load-bearing for SMALL particles** (plan §0.9) — and the
measurement is not the one you would guess. It is not about *detection*: on the
two deliberately faint probes in ``particle_movie()`` (r=4 and r=3) a coarse
``(4, 8)`` stack still finds both. It is about what is then *measured* of them.
Mean absolute error in recovered radius over the seven isolated particles, with
identical labels and an identical head, and the r=3 probe on its own::

    sigmas (0.5, 1, 2, 4, 8)   13.4 %    r=3 probe  -12 %
    sigmas (1, 2, 4, 8)        13.2 %    r=3 probe  -15 %
    sigmas (2, 4, 8)           20.4 %    r=3 probe  -28 %
    sigmas (4, 8)              25.5 %    r=3 probe  -44 %

So the floor must not be raised above ~1 px, which is what plan §0.9's "not
without a documented sensitivity measurement" amounts to here — a found particle
measured 44% too small is worse than an honest miss, because it enters the size
distribution. 0.5 buys nothing over 1.0 on this fixture and is kept anyway: it is
2 ms of a 71 ms 512² stack, and it is what a genuinely 1–2 px feature needs.

**3. Memory is bounded by row-banding, not by hope.** The full stack is
``C·H·W·4`` bytes — 2.4 GB for 36 channels at 4096², which is the plan's stated
frame size. So the stack is produced in **row bands with a halo** (:func:`
map_feature_bands`), and the two callers that matter never materialise the whole
thing: training samples the labelled pixels out of each band and drops it, and
:meth:`~spyde.particles.scribble.ScribbleClassifier.predict_proba` writes one
band's probabilities and drops it. With a halo of at least the largest filter
radius the banded result is *identical* to the unbanded one — pinned by a test.

NaN input
---------
A drift-corrected frame carries a NaN-padded border
(:mod:`spyde.drift.warp`), and every convolution propagates NaN outward, which
would erase a band of real data. :func:`prepare_frame` fills the non-finite
pixels with the finite **minimum** — the same choice, for the same reason, as
:func:`spyde.particles.classical._prepare`: the padding then reads as background,
the one value guaranteed not to classify as a particle. It also returns the
validity mask, and the classifier is required to force those pixels to zero
probability (plan trap 2).
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, NamedTuple

import numpy as np

from spyde.device_lock import accelerator_lock

log = logging.getLogger(__name__)

#: Gaussian kernels are truncated at this many sigma, matching
#: ``scipy.ndimage.gaussian_filter``'s default so the two agree numerically
#: (``test_particles_scribble.py`` pins that against scipy with ``mode="mirror"``,
#: which is what torch's ``reflect`` padding is).
_TRUNCATE = 4.0

#: Default scales, in pixels, octave-spaced. The **floor** is what matters — see
#: the module docstring for the measured cost of raising it. The top end (8) is
#: what gives the head a local-background reference, which is how it separates a
#: faint particle from a bright patch of support film.
DEFAULT_SIGMAS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)

#: Default rank-filter window radii, in pixels (window is ``2r+1`` square).
DEFAULT_RANK_RADII: tuple[int, ...] = (1, 2)

#: Projections available across the rotated membrane responses (Weka's set).
MEMBRANE_PROJECTIONS: tuple[str, ...] = ("sum", "mean", "std", "median",
                                         "max", "min")

#: Target working-set size for one row band. Not a hard cap — a band is always at
#: least ``4·halo`` rows tall, because a band shorter than its own halo would
#: recompute more halo than payload.
BAND_BYTES: int = 256 << 20

#: Robust per-frame statistics are estimated from at most this many pixels.
#: ``np.percentile`` on a 4096² frame is a full sort (~1.5 s here) and would
#: dominate the whole interaction budget; a strided subsample of 10⁶ pixels puts
#: the median and IQR well inside their own sampling noise. Same trade-off, and
#: the same reasoning, as the subsampled first-paint histogram in ``plot.py``.
_STAT_SAMPLE_MAX = 1_000_000


# ── spec ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeatureSpec:
    """Which features to compute, and at which scales.

    Frozen and serialisable: :meth:`to_dict` / :meth:`from_dict` round-trip
    through JSON, which is what lets a wizard parameter schema drive it and what
    lets a trained recipe be saved next to its weights
    (:meth:`spyde.particles.scribble.ScribbleClassifier.save`). A model whose
    spec no longer matches its weights is meaningless, so the two are never
    stored apart.

    Parameters
    ----------
    sigmas
        Gaussian scales in pixels, ascending. Drives ``gaussian``,
        ``difference_of_gaussians``, ``sobel``, ``hessian`` and ``laplacian``.
    rank_radii
        Window radii for ``median`` / ``minimum`` / ``maximum``, in pixels.
    membrane
        Weka's membrane projections: a thin line kernel rotated through 180° and
        projected. **Off by default**, which is a deliberate deviation from
        ParticleSpy's set rather than an omission — the family exists to find
        *elongated* structures (its name is literal; it was built for neuron
        membranes), it is the most expensive channel here per channel (adds 33 ms
        to a 71 ms 512² total for four channels, so ~8 ms/channel against ~2 ms
        for everything else), and on the compact blobs this feature is for it did
        not change the faint-probe result either way. Turn it on for fibres,
        films or lattice fringes, where it earns its cost.
    membrane_thickness
        Line width in pixels. Rounded up to an odd number so the line is centred.
    normalize_frame
        Robustly standardise the frame (median / IQR over finite pixels) before
        featurising. On by default because it is what makes a saved recipe
        transferable: without it every channel carries the dataset's absolute
        intensity scale, so a model trained on a float image in 0..1 predicts
        nothing at all on the same sample recorded as uint16 counts.
    """

    sigmas: tuple[float, ...] = DEFAULT_SIGMAS
    intensity: bool = True
    gaussian: bool = True
    difference_of_gaussians: bool = True
    sobel: bool = True
    hessian: bool = True
    laplacian: bool = True
    rank_radii: tuple[int, ...] = DEFAULT_RANK_RADII
    median: bool = True
    minimum: bool = True
    maximum: bool = True
    membrane: bool = False
    membrane_patch: int = 19
    membrane_thickness: int = 1
    membrane_rotations: int = 12
    membrane_projections: tuple[str, ...] = ("mean", "std", "max", "min")
    normalize_frame: bool = True

    def __post_init__(self) -> None:
        # Coerce lists (which is what `from_dict` and a JSON parameter schema
        # hand over) to tuples, so the dataclass stays hashable and comparable.
        object.__setattr__(self, "sigmas",
                           tuple(float(s) for s in self.sigmas))
        object.__setattr__(self, "rank_radii",
                           tuple(int(r) for r in self.rank_radii))
        object.__setattr__(self, "membrane_projections",
                           tuple(str(p) for p in self.membrane_projections))
        if any(s <= 0 for s in self.sigmas):
            raise ValueError(f"sigmas must be positive; got {self.sigmas}")
        if tuple(sorted(self.sigmas)) != self.sigmas:
            raise ValueError(f"sigmas must be ascending; got {self.sigmas}")
        if any(r < 1 for r in self.rank_radii):
            raise ValueError(f"rank_radii must be >= 1; got {self.rank_radii}")
        bad = set(self.membrane_projections) - set(MEMBRANE_PROJECTIONS)
        if bad:
            raise ValueError(
                f"unknown membrane projection(s) {sorted(bad)}; expected any of "
                f"{', '.join(MEMBRANE_PROJECTIONS)}"
            )
        if self.membrane and self.membrane_patch % 2 == 0:
            raise ValueError(
                f"membrane_patch must be odd so the line is centred; got "
                f"{self.membrane_patch}"
            )
        if self.membrane and self.membrane_rotations < 1:
            raise ValueError(
                f"membrane_rotations must be >= 1; got {self.membrane_rotations}")
        if not self.channel_names():
            raise ValueError(
                "this FeatureSpec produces no channels at all — every family is "
                "disabled, or the enabled ones have empty sigmas/rank_radii")

    # -- serialisation ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-safe dict (tuples become lists)."""
        out = asdict(self)
        for key in ("sigmas", "rank_radii", "membrane_projections"):
            out[key] = list(out[key])
        return out

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FeatureSpec":
        """Rebuild from :meth:`to_dict`, ignoring keys this build does not know.

        Unknown keys are dropped rather than raising: a spec written by a newer
        build should still load with the features this one understands, and the
        alternative is that a saved recipe becomes unopenable on a downgrade.
        Missing keys take their defaults.
        """
        if not d:
            return cls()
        fields = set(cls.__dataclass_fields__)
        unknown = set(d) - fields
        if unknown:
            log.debug("FeatureSpec.from_dict ignoring unknown keys: %s",
                      sorted(unknown))
        return cls(**{k: v for k, v in d.items() if k in fields})

    def replace(self, **kw: Any) -> "FeatureSpec":
        """A copy with *kw* overridden (the frozen-dataclass setter)."""
        return replace(self, **kw)

    # -- shape -----------------------------------------------------------------

    def channel_names(self) -> list[str]:
        """Channel names, in output order.

        Derived from the same :func:`_channel_plan` the tensor builder walks, so
        the names cannot drift out of step with the channels — there is exactly
        one definition of the order.
        """
        return [name for _family, name, _args in _channel_plan(self)]

    @property
    def n_channels(self) -> int:
        return len(_channel_plan(self))

    @property
    def halo(self) -> int:
        """Largest radius, in pixels, that any enabled filter reaches.

        This is the row overlap :func:`map_feature_bands` needs for a banded
        stack to equal an unbanded one. The ``+1`` on the gaussian radius is the
        3-tap Sobel / second-difference stencil applied *after* the blur.
        """
        r = 1
        if self.sigmas and (self.gaussian or self.difference_of_gaussians or
                            self.sobel or self.hessian or self.laplacian):
            r = max(r, _gauss_radius(max(self.sigmas)) + 1)
        if self.rank_radii and (self.median or self.minimum or self.maximum):
            r = max(r, max(self.rank_radii))
        if self.membrane:
            r = max(r, self.membrane_patch // 2)
        return int(r)


# ── the channel plan: THE definition of channel order and channel names ──────

def _fmt(x: float) -> str:
    return f"{float(x):g}"


def _channel_plan(spec: FeatureSpec) -> list[tuple[str, str, tuple]]:
    """``(family, name, args)`` for every channel, in output order.

    Both :meth:`FeatureSpec.channel_names` and :class:`_Pass` walk this list, so a
    new family is added in one place and the names follow automatically. Grouping
    by family rather than interleaving by sigma costs nothing — the shared
    intermediates are memoised on :class:`_Pass`, so revisiting sigma 2 for the
    Hessian after visiting it for the Sobel does not recompute its blur.
    """
    plan: list[tuple[str, str, tuple]] = []
    if spec.intensity:
        plan.append(("intensity", "intensity", ()))
    if spec.gaussian:
        plan += [("gaussian", f"gaussian_s{_fmt(s)}", (s,)) for s in spec.sigmas]
    if spec.difference_of_gaussians:
        plan += [("dog", f"dog_s{_fmt(a)}_s{_fmt(b)}", (a, b))
                 for a, b in zip(spec.sigmas, spec.sigmas[1:])]
    if spec.sobel:
        plan += [("sobel", f"sobel_s{_fmt(s)}", (s,)) for s in spec.sigmas]
    if spec.laplacian:
        plan += [("laplacian", f"laplacian_s{_fmt(s)}", (s,)) for s in spec.sigmas]
    if spec.hessian:
        for s in spec.sigmas:
            plan.append(("hessian_major", f"hessian_major_s{_fmt(s)}", (s,)))
            plan.append(("hessian_minor", f"hessian_minor_s{_fmt(s)}", (s,)))
    for stat in ("median", "minimum", "maximum"):
        if getattr(spec, stat):
            plan += [(stat, f"{stat}_r{r}", (r,)) for r in spec.rank_radii]
    if spec.membrane:
        plan += [("membrane", f"membrane_{p}", (p,))
                 for p in spec.membrane_projections]
    return plan


# ── device ───────────────────────────────────────────────────────────────────

def import_torch():
    """``import torch``, with a message that says whose problem it is if it fails.

    torch is a core SpyDE dependency, so a failure here is a broken environment
    rather than a missing extra — and the bare ``ModuleNotFoundError`` from six
    frames down does not say that. Shared with
    :mod:`spyde.particles.scribble` rather than duplicated there.
    """
    try:
        import torch
        return torch
    except Exception as exc:                              # pragma: no cover
        raise ImportError(
            "spyde.particles needs torch, which is a core SpyDE dependency but "
            f"failed to import: {exc}"
        ) from exc



def select_device(prefer: str | None = None):
    """Best torch device for the feature stack: CUDA → Apple-MPS → CPU.

    Parameters
    ----------
    prefer
        An explicit device string (``"cpu"``, ``"cuda"``, ``"mps"``) to force.
        Tests force ``"cpu"``: torch-CUDA work segfaults under the pytest process
        on Windows (CLAUDE.md), which is a harness interaction rather than a code
        defect, so the GPU path is exercised in a subprocess instead.
    """
    torch = import_torch()
    if prefer is not None:
        return torch.device(prefer)
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
    except Exception as exc:                              # pragma: no cover
        log.debug("device probe failed (%s); using CPU", exc)
    return torch.device("cpu")


def gpu_available() -> bool:
    """True when a hardware-accelerated torch device exists (CUDA or MPS).

    CPU-only torch is False here even though the stack runs perfectly well on it
    — this gate is about whether to *expect* interactive speed, matching
    ``vector_orientation_gpu.gpu_available``.
    """
    try:
        return select_device().type in ("cuda", "mps")
    except Exception:                                     # pragma: no cover
        return False


# ── frame preparation ────────────────────────────────────────────────────────

class PreparedFrame(NamedTuple):
    """A frame ready to featurise, plus the validity mask the caller must honour.

    ``image``
        float32, non-finite pixels replaced (see :func:`prepare_frame`), and
        robustly standardised when the spec asks for it.
    ``valid``
        bool, True where the *source* pixel was finite. The classifier forces
        zero foreground probability outside this — plan trap 2.
    """

    image: np.ndarray
    valid: np.ndarray


def _robust_stats(values: np.ndarray) -> tuple[float, float]:
    """``(centre, spread)`` — median and IQR/1.349, from a strided subsample.

    IQR/1.349 is the normal-consistent robust sigma. Robust rather than
    mean/std because a frame *full of particles* has a heavy bright tail, and a
    mean/std standardisation would then move the background level around with the
    particle coverage — so the same physical background would present differently
    at t=0 and t=end, which is exactly the drift a learned head must not see.
    """
    v = values
    if v.size > _STAT_SAMPLE_MAX:
        v = v[:: int(np.ceil(v.size / _STAT_SAMPLE_MAX))]
    centre = float(np.median(v))
    q1, q3 = np.percentile(v, [25.0, 75.0])
    spread = float(q3 - q1) / 1.349
    if not (spread > 0):
        # Constant (or near-constant) frame: fall back to the std, then to 1 so
        # the standardisation is a no-op rather than a division by zero.
        spread = float(np.std(v)) or 1.0
    return centre, spread


def prepare_frame(frame, spec: FeatureSpec | None = None) -> PreparedFrame:
    """Fill non-finite pixels, optionally standardise, and report validity.

    Parameters
    ----------
    frame
        2-D array. May contain NaN (a drift-corrected border does).
    spec
        Only ``normalize_frame`` is consulted.

    Returns
    -------
    PreparedFrame

    Notes
    -----
    Non-finite pixels are filled with the finite **minimum**, not with zero and
    not with the mean: the padding has to read as background, and the minimum is
    the one value that cannot threshold or classify as a particle. This mirrors
    :func:`spyde.particles.classical._prepare` deliberately — two engines that
    disagree about what the padding *is* would disagree about the frame border
    for no reason a user could see.

    The robust statistics are computed over finite pixels **only**, before the
    fill. Filling first would let a large NaN border pull the median down and
    rescale the whole frame by how much of it was padding.
    """
    spec = spec or FeatureSpec()
    img = np.asarray(frame, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError(f"frame must be 2-D; got shape {img.shape}")
    if min(img.shape) < 4:
        raise ValueError(
            f"frame must be at least 4x4 to filter; got {img.shape}")

    valid = np.isfinite(img)
    if valid.all():
        finite = img.reshape(-1)
    else:
        img = img.copy()
        finite = img[valid]
        if finite.size == 0:
            raise ValueError("frame has no finite pixels")
        img[~valid] = finite.min()

    if spec.normalize_frame:
        centre, spread = _robust_stats(finite)
        img = (img - np.float32(centre)) / np.float32(spread)

    return PreparedFrame(np.ascontiguousarray(img, dtype=np.float32), valid)


def _as_prepared(frame, spec: FeatureSpec) -> PreparedFrame:
    return frame if isinstance(frame, PreparedFrame) else prepare_frame(frame, spec)


# ── separable convolution primitives ─────────────────────────────────────────

def _gauss_radius(sigma: float) -> int:
    """Kernel half-width, matching ``scipy.ndimage``'s ``int(truncate*sd + 0.5)``."""
    return max(1, int(_TRUNCATE * float(sigma) + 0.5))


def _gauss_kernel(torch, sigma: float, radius: int, device, dtype):
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def _pad_edges(t, ry: int, rx: int):
    """Pad ``(1, 1, h, w)`` by *ry* rows and *rx* columns, reflect where legal.

    Padding is torch's ``reflect``, which is ``scipy.ndimage``'s ``mirror`` (NOT
    scipy's ``reflect``, which duplicates the edge sample); the parity test
    against scipy passes ``mode="mirror"`` for exactly this reason.

    ``reflect`` requires the pad to be strictly smaller than the dimension, and
    the kernels here are not small: a sigma-8 gaussian has radius 32 and a
    19x19 membrane patch radius 9, either of which exceeds a small frame. So the
    reflect is taken as far as it is legal and the remainder is **replicated**.
    Clamping the kernel radius instead was the first implementation, and it is
    worse in a way that is invisible until it matters: it silently applies a
    *different, narrower* filter than the one the FeatureSpec asked for, so the
    same spec means different things on different frame sizes — and a saved recipe
    would then not reproduce on a crop.
    """
    import torch.nn.functional as F
    h, w = int(t.shape[-2]), int(t.shape[-1])
    ry_ok, rx_ok = min(ry, h - 1), min(rx, w - 1)
    if ry_ok or rx_ok:
        t = F.pad(t, (rx_ok, rx_ok, ry_ok, ry_ok), mode="reflect")
    over_y, over_x = ry - ry_ok, rx - rx_ok
    if over_y or over_x:
        t = F.pad(t, (over_x, over_x, over_y, over_y), mode="replicate")
    return t


def _sep_conv(torch, t, ky, kx):
    """Separable correlation of ``(1, 1, h, w)`` *t* with 1-D kernels.

    ``F.conv2d`` is a correlation, not a convolution, so a kernel is applied as
    written — no flip. Either kernel may be ``None`` to skip that axis.
    """
    import torch.nn.functional as F
    if ky is not None:
        t = _pad_edges(t, (ky.numel() - 1) // 2, 0)
        t = F.conv2d(t, ky.view(1, 1, -1, 1))
    if kx is not None:
        t = _pad_edges(t, 0, (kx.numel() - 1) // 2)
        t = F.conv2d(t, kx.view(1, 1, 1, -1))
    return t


_MEMBRANE_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _membrane_kernels(patch: int, thickness: int, rotations: int) -> np.ndarray:
    """``(rotations, patch, patch)`` line kernels spanning 180°, sum-normalised.

    Built with ``scipy.ndimage.rotate`` once per configuration and cached. The
    rotation is a spline resample of a tiny image, so this is not expensive
    (measured 0.98 ms for the default twelve 19x19 kernels), but it sits on the
    interaction path — every retrain re-featurises every labelled frame — and 1 ms
    per frame has no reason to recur.

    Sum-normalised so a projection stays in the same units as the intensity
    channel. Weka does not normalise; unnormalised, the ``sum`` projection is
    ``patch`` times larger than every other channel, which makes the head's first
    layer spend its capacity on rescaling.
    """
    key = (int(patch), int(thickness), int(rotations))
    cached = _MEMBRANE_CACHE.get(key)
    if cached is not None:
        return cached

    from scipy.ndimage import rotate

    base = np.zeros((patch, patch), dtype=np.float32)
    c = patch // 2
    half = max(1, int(thickness)) // 2
    base[:, c - half: c + half + 1] = 1.0

    out = []
    for i in range(rotations):
        k = base if i == 0 else rotate(
            base, 180.0 * i / rotations, reshape=False, order=1,
            mode="constant", cval=0.0)
        k = np.clip(np.asarray(k, dtype=np.float32), 0.0, None)
        s = float(k.sum())
        out.append(k / s if s > 0 else k)
    stacked = np.stack(out)
    _MEMBRANE_CACHE[key] = stacked
    return stacked


# ── one pass over a frame (or a row band) ────────────────────────────────────

class _Pass:
    """Shared intermediates for one image, computed at most once each.

    This is where "one pass over the frame, not one pass per feature" actually
    lives. Anything more than one channel needs — the blur at a sigma, that
    blur's three second derivatives, an unfolded rank window, the rotated
    membrane responses — is memoised here, so the 36 default channels cost 5
    gaussian blurs, 5 derivative triples, 2 rank passes and nothing else.
    """

    def __init__(self, image, spec: FeatureSpec, device):
        torch = import_torch()
        self._torch = torch
        self.spec = spec
        self.device = device
        arr = np.ascontiguousarray(image, dtype=np.float32)
        self.h, self.w = int(arr.shape[0]), int(arr.shape[1])
        self.img = torch.as_tensor(arr, device=device)
        self._t = self.img.view(1, 1, self.h, self.w)
        self._blur: dict[float, Any] = {}
        self._second: dict[float, tuple] = {}
        self._rank: dict[tuple[int, str], Any] = {}
        self._membrane: Any = None

    # -- primitives ----------------------------------------------------------

    def blur(self, sigma: float):
        """Gaussian blur at *sigma* as ``(1, 1, h, w)``. Memoised."""
        got = self._blur.get(sigma)
        if got is None:
            torch = self._torch
            k = _gauss_kernel(torch, sigma, _gauss_radius(sigma), self.device,
                              self.img.dtype)
            got = _sep_conv(torch, self._t, k, k)
            self._blur[sigma] = got
        return got

    def second(self, sigma: float):
        """``(dyy, dxx, dxy)`` of the blur at *sigma*. Memoised.

        Second differences ``[1, -2, 1]`` and a cross term
        ``[-½, 0, ½] ⊗ [-½, 0, ½]``. **Not** skimage's ``hessian_matrix``, which
        applies ``np.gradient`` twice and therefore uses a 5-tap
        ``[¼, 0, -½, 0, ¼]`` — a wider, softer stencil that skips the immediate
        neighbours entirely. Both are valid discrete Hessians; the compact one is
        used because it keeps the fine-scale response localised, which is what the
        smallest sigmas are in the set for. Exact skimage parity is not a
        requirement here — a channel is an input to a learned head, not a
        published measurement.
        """
        got = self._second.get(sigma)
        if got is None:
            torch = self._torch
            b = self.blur(sigma)
            d2 = torch.tensor([1.0, -2.0, 1.0], device=self.device,
                              dtype=self.img.dtype)
            d1 = torch.tensor([-0.5, 0.0, 0.5], device=self.device,
                              dtype=self.img.dtype)
            dyy = _sep_conv(torch, b, d2, None)
            dxx = _sep_conv(torch, b, None, d2)
            dxy = _sep_conv(torch, b, d1, d1)
            got = (dyy, dxx, dxy)
            self._second[sigma] = got
        return got

    def rank(self, radius: int, stat: str):
        """Median / minimum / maximum over a ``(2r+1)²`` window. Memoised."""
        got = self._rank.get((int(radius), stat))
        if got is None:
            self._compute_rank(int(radius))
            got = self._rank[(int(radius), stat)]
        return got

    def _compute_rank(self, radius: int) -> None:
        """ONE ``unfold`` per radius serves all three rank statistics.

        ``unfold`` materialises ``(2r+1)²·h·w`` floats — 1.7 GB for r=2 on a
        4096² frame — which is why the whole stack is banded (:func:`
        map_feature_bands`); the window is dropped as soon as the enabled
        statistics are reduced out of it, so only one is ever alive.

        ``max_pool2d(stride=1)`` was the obvious alternative for min/max and is
        **much slower**, which is the opposite of what its fused-reduction
        implementation suggests. Measured on a 512² frame, CPU::

            r=1   max_pool2d 18.4 ms   unfold amin+amax  2.4 ms
            r=2   max_pool2d 46.9 ms   unfold amin+amax  5.3 ms

        torch's pooling kernels are tuned for the strided, downsampling case; at
        stride 1 they re-read every window from scratch. Making the window
        separable (two 1-D pools — a square min/max genuinely is separable, and
        the results are bit-identical) only got r=2 from 46.9 to 23.7 ms, still
        4x the unfold. And the unfold is *shared* with the median, which needs the
        window materialised regardless, so the whole rank family costs one pass:
        170 ms → 27 ms for the default two radii.
        """
        import torch.nn.functional as F
        spec = self.spec
        k = 2 * int(radius) + 1
        u = F.unfold(_pad_edges(self._t, radius, radius),
                     kernel_size=(k, k))                          # (1, k*k, h*w)
        shape = (1, 1, self.h, self.w)
        if spec.median:
            self._rank[(radius, "median")] = u.median(dim=1).values.view(shape)
        if spec.minimum:
            self._rank[(radius, "minimum")] = u.amin(dim=1).view(shape)
        if spec.maximum:
            self._rank[(radius, "maximum")] = u.amax(dim=1).view(shape)

    def membrane(self):
        """``(1, R, h, w)`` responses to the rotated line kernels. Memoised."""
        if self._membrane is None:
            torch = self._torch
            import torch.nn.functional as F
            spec = self.spec
            k = _membrane_kernels(spec.membrane_patch, spec.membrane_thickness,
                                  spec.membrane_rotations)
            weight = torch.as_tensor(k, device=self.device,
                                     dtype=self.img.dtype).unsqueeze(1)
            r = spec.membrane_patch // 2
            self._membrane = F.conv2d(_pad_edges(self._t, r, r), weight)
        return self._membrane

    # -- channels ------------------------------------------------------------

    def channel(self, family: str, args: tuple):
        """One ``(h, w)`` channel tensor for a :func:`_channel_plan` entry."""
        if family == "intensity":
            return self.img
        if family == "gaussian":
            return self.blur(args[0])[0, 0]
        if family == "dog":
            return (self.blur(args[0]) - self.blur(args[1]))[0, 0]
        if family == "sobel":
            return self._sobel(args[0])
        if family == "laplacian":
            dyy, dxx, _ = self.second(args[0])
            return (dyy + dxx)[0, 0]
        if family in ("hessian_major", "hessian_minor"):
            major, minor = self._hessian_eigs(args[0])
            return major if family == "hessian_major" else minor
        if family in ("median", "minimum", "maximum"):
            return self.rank(args[0], family)[0, 0]
        if family == "membrane":
            return self._membrane_projection(args[0])
        raise ValueError(f"unknown feature family {family!r}")  # pragma: no cover

    def _sobel(self, sigma: float):
        """Sobel gradient magnitude of the blur at *sigma*.

        skimage's kernels exactly: derivative ``[1, 0, -1]`` against smoothing
        ``[1, 2, 1]/4``, and the magnitude divided by ``sqrt(2)`` as
        ``skimage.filters.sobel`` does. The constant is irrelevant once the head
        standardises its inputs, but matching it means a reader can compare a
        channel against skimage without wondering about a factor.
        """
        torch = self._torch
        b = self.blur(sigma)
        der = torch.tensor([1.0, 0.0, -1.0], device=self.device,
                           dtype=self.img.dtype)
        smo = torch.tensor([1.0, 2.0, 1.0], device=self.device,
                           dtype=self.img.dtype) / 4.0
        gy = _sep_conv(torch, b, der, smo)
        gx = _sep_conv(torch, b, smo, der)
        return (torch.sqrt(gy * gy + gx * gx) / math.sqrt(2.0))[0, 0]

    def _hessian_eigs(self, sigma: float):
        """``(major, minor)`` eigenvalues of the 2×2 Hessian, signed.

        Analytic, not ``linalg.eigvalsh``: for a symmetric 2×2 the eigenvalues are
        ``tr/2 ± sqrt((tr/2)² - det)``, a handful of elementwise ops over the
        image. ``eigvalsh`` batched over h·w 2×2 matrices was measured at 125 ms
        on a 512² frame against **0.32 ms** for the closed form — a factor of 390,
        so this is not an optimisation but the only viable option.

        "Major"/"minor" are by signed value (major ≥ minor), which is what
        distinguishes a bright blob (both strongly negative) from a ridge (one
        negative, one near zero). Ordering by |value| would merge those two.
        """
        torch = self._torch
        dyy, dxx, dxy = self.second(sigma)
        half_tr = 0.5 * (dyy + dxx)
        det = dyy * dxx - dxy * dxy
        disc = torch.sqrt(torch.clamp(half_tr * half_tr - det, min=0.0))
        return (half_tr + disc)[0, 0], (half_tr - disc)[0, 0]

    def _membrane_projection(self, projection: str):
        resp = self.membrane()[0]                      # (R, h, w)
        if projection == "sum":
            return resp.sum(dim=0)
        if projection == "mean":
            return resp.mean(dim=0)
        if projection == "std":
            return resp.std(dim=0, unbiased=False)
        if projection == "median":
            return resp.median(dim=0).values
        if projection == "max":
            return resp.amax(dim=0)
        if projection == "min":
            return resp.amin(dim=0)
        raise ValueError(                              # pragma: no cover
            f"unknown membrane projection {projection!r}")


def _band_stack(image, spec: FeatureSpec, device):
    """``(C, h, w)`` float32 tensor for one image (or row band)."""
    torch = import_torch()
    p = _Pass(image, spec, device)
    plan = _channel_plan(spec)
    out = torch.empty((len(plan), p.h, p.w), device=device, dtype=torch.float32)
    for i, (family, _name, args) in enumerate(plan):
        out[i] = p.channel(family, args)
    return out


# ── banding ──────────────────────────────────────────────────────────────────

def band_rows_for(spec: FeatureSpec, width: int,
                  budget_bytes: int = BAND_BYTES) -> int:
    """How many rows one band should cover, given the per-band memory budget.

    The working set is bigger than the output stack — the pass also holds one
    blur and three derivative images per sigma, plus an unfolded median window —
    so the divisor counts those too rather than only ``n_channels``. Getting this
    wrong does not produce a wrong answer, only a larger peak allocation, but the
    plan's frame size leaves no headroom to be casual about it.
    """
    working = spec.n_channels + 4 * len(spec.sigmas) + 8
    if (spec.median or spec.minimum or spec.maximum) and spec.rank_radii:
        # One unfolded window is alive at a time (see `_Pass._compute_rank`), so
        # it is the LARGEST radius that sets the peak, not the sum over radii.
        working += (2 * max(spec.rank_radii) + 1) ** 2
    rows = int(max(1, budget_bytes // max(1, working * int(width) * 4)))
    return max(rows, 4 * spec.halo)


def map_feature_bands(
    frame,
    spec: FeatureSpec | None = None,
    *,
    device=None,
    fn: Callable[[int, int, Any], None],
    band_rows: int | None = None,
) -> None:
    """Compute the stack in row bands and hand each to *fn*.

    Calls ``fn(y0, y1, stack)`` where ``stack`` is a ``(C, y1-y0, W)`` tensor for
    output rows ``y0:y1``. Each band is featurised with ``spec.halo`` extra rows
    of real data above and below and then cropped, so **the banded result equals
    the unbanded one exactly** for every row — the halo replaces what reflect
    padding would otherwise have invented at a band boundary. Pinned by
    ``test_particles_scribble.py::TestBanding``.

    A callback rather than a generator on purpose: this holds the process-wide
    accelerator lock (:func:`spyde.device_lock.accelerator_lock`) for the whole
    traversal, and a generator abandoned by its consumer would hold that lock
    until garbage collection.
    """
    spec = spec or FeatureSpec()
    prepared = _as_prepared(frame, spec)
    img = prepared.image
    h, w = img.shape
    if device is None:
        device = select_device()

    halo = spec.halo
    rows = int(band_rows or band_rows_for(spec, w))

    with accelerator_lock(device):
        if rows >= h:
            fn(0, h, _band_stack(img, spec, device))
            return
        y = 0
        while y < h:
            y1 = min(h, y + rows)
            ky0, ky1 = max(0, y - halo), min(h, y1 + halo)
            stack = _band_stack(img[ky0:ky1], spec, device)
            fn(y, y1, stack[:, y - ky0: y1 - ky0])
            y = y1


def feature_tensor(frame, spec: FeatureSpec | None = None, *, device=None):
    """``(C, H, W)`` float32 torch tensor of every channel in *spec*.

    Parameters
    ----------
    frame
        2-D array, or a :class:`PreparedFrame` from :func:`prepare_frame`.
    device
        Torch device; ``None`` calls :func:`select_device`.

    Notes
    -----
    This materialises the whole stack: ``C·H·W·4`` bytes, which is 2.4 GB for the
    36 default channels at 4096². That is fine for the interactive path (one
    displayed frame, and the display is tiled well below that) and wrong for a
    batch run over big frames — those go through :func:`map_feature_bands` or
    :func:`sample_features`, which never hold more than one band.
    """
    spec = spec or FeatureSpec()
    prepared = _as_prepared(frame, spec)
    torch = import_torch()
    if device is None:
        device = select_device()
    h, w = prepared.image.shape

    # The allocation is a device submission too, so it goes inside the lock — the
    # reentrant acquisition in `map_feature_bands` is free.
    with accelerator_lock(device):
        out = torch.empty((spec.n_channels, h, w), device=device,
                          dtype=torch.float32)

        def take(y0: int, y1: int, stack) -> None:
            out[:, y0:y1] = stack

        map_feature_bands(prepared, spec, device=device, fn=take)
    return out


def feature_stack(frame, spec: FeatureSpec | None = None, *,
                  device=None) -> np.ndarray:
    """:func:`feature_tensor` as a ``(C, H, W)`` numpy array.

    The door the sklearn RandomForest parity reference comes through: the gate is
    "same labels, same features, does the torch head agree with the forest", so
    both sides must read the *identical* channels, not merely equivalent ones.
    """
    if device is None:
        device = select_device()
    # The device->host copy is a submission as well, hence the (reentrant) lock.
    with accelerator_lock(device):
        return feature_tensor(frame, spec, device=device).detach().cpu().numpy()


def sample_features(frame, index, spec: FeatureSpec | None = None, *,
                    device=None):
    """``(k, C)`` float32 tensor of the stack sampled at *k* pixels.

    Parameters
    ----------
    index
        Flat pixel indices into a ``(H, W)`` frame, ``(k,)`` integer; or ``(k, 2)``
        ``(y, x)`` pairs.

    Notes
    -----
    This is how training reads its data, and why training does not care how big
    the frame is: the stack is produced band by band and only the sampled rows
    survive, so the peak allocation is one band plus ``k·C`` floats. Scribbles are
    thousands of pixels, so ``k·C`` is under a megabyte however large the frame.
    """
    spec = spec or FeatureSpec()
    prepared = _as_prepared(frame, spec)
    torch = import_torch()
    if device is None:
        device = select_device()

    h, w = prepared.image.shape
    idx = np.asarray(index)
    if idx.ndim == 2 and idx.shape[1] == 2:
        flat = idx[:, 0].astype(np.int64) * w + idx[:, 1].astype(np.int64)
    else:
        flat = idx.reshape(-1).astype(np.int64)
    if flat.size and (flat.min() < 0 or flat.max() >= h * w):
        raise IndexError(
            f"pixel index outside 0..{h * w - 1} for a {h}x{w} frame")

    ys, xs = np.divmod(flat, w)
    with accelerator_lock(device):          # see `feature_tensor` — allocation too
        out = torch.empty((flat.size, spec.n_channels), device=device,
                          dtype=torch.float32)

        def take(y0: int, y1: int, stack) -> None:
            sel = np.flatnonzero((ys >= y0) & (ys < y1))
            if not sel.size:
                return
            ty = torch.as_tensor(ys[sel] - y0, device=device, dtype=torch.long)
            tx = torch.as_tensor(xs[sel], device=device, dtype=torch.long)
            rows = torch.as_tensor(sel, device=device, dtype=torch.long)
            out[rows] = stack[:, ty, tx].t()

        map_feature_bands(prepared, spec, device=device, fn=take)
    return out


def feature_names(spec: FeatureSpec | None = None) -> list[str]:
    """Channel names for *spec*, in the order :func:`feature_stack` returns them."""
    return (spec or FeatureSpec()).channel_names()

"""
fast_engine.py — the full-frame scribble engine. Replaces the 34-channel stack +
64-wide MLP of :mod:`spyde.particles.scribble` with something that can segment a
whole 4096² frame at interactive rates.

It is a drop-in for :class:`~spyde.particles.scribble.ScribbleClassifier`: same
constructor keywords, same ``fit`` / ``predict_foreground_boundary`` /
``predict_class_proba`` / ``save`` / ``load``, same report keys, same
``ScribbleClass`` list with its ``particle`` and ``boundary`` flags. So
``split_instances``, ``measure_frame``, the batch fan-out and the caret are all
unchanged — only the first box of the pipeline in ``spyde/particles/__init__``
is different.

Why it is shaped like this — every point is a measurement on a TITAN X (Pascal),
fp32, 4096² frames, and several of them are counter-intuitive.

**The MLP refit dominated the old design, not the convolutions.** One full
interactive cycle at 1024² measured 2487 ms: featurise 68 ms (2.7%), 300-epoch
full-batch Adam refit 1938 ms (78%), predict 98 ms, split 358 ms. So the levers
are the head and the training loop, not the filter bank.

**A wide head is unaffordable at full frame.** ``hidden=64`` materialises a
(1, 64, 4096, 4096) fp32 activation — 4.3 GB, ~22 ms of DRAM traffic before any
arithmetic. Measured head cost alone at 4096²: 113 ms for ``Linear`` over a
reshaped ``(H·W, C)`` matrix, 74 ms for ``Conv2d`` 1×1. ``hidden=0`` — a plain
linear head, i.e. one 1×1 conv — is 2.8× on the whole pipeline, and an epoch
sweep found training accuracy 1.0 at every epoch count from 10 to 300, so width
was never buying separability.

**Produce K filter planes in TWO multi-output conv passes**, not K one-channel
separable blurs: cuDNN is poor at 1-channel work (one separable blur at 4096²
measured 9.3 ms against a ~1 ms bandwidth floor). ``Conv2d(1, K, (1,k))`` then
depthwise ``Conv2d(K, K, (k,1), groups=K)`` gives it the shape it wants.

**"Extremely small kernels" is right for the discriminative filters and WRONG for
the background reference.** On real in-situ counting data (``InSituElectrochem
Growth``) the particles are 6.8 nm median (~15 px) at a local contrast of 1.07
counts — **per-pixel CNR 0.169**. One pixel carries almost nothing, and because
the field has large-scale thickness variation, "darker than the frame" is not
"darker than its surroundings": two stroke-siting attempts that compared
intensities globally produced an inverted segmentation. So a large local
background reference is mandatory — but it is computed on a 4×-decimated image
(:meth:`FastFeatureBank._background`), so a σ=25 reference gives a 321 px
receptive field at roughly small-kernel cost.

**Smooth the LOGITS, not the image.** At CNR 0.17 an independent decision per
pixel shatters a real particle into specks and inflates the instance count
(7095 → 4842 at the same threshold once the decision variable is blurred with
σ=3). It is one separable pass over ``n_classes`` planes and it is a matched
filter on the quantity actually being thresholded.

Two levers that do NOT work on this card, so do not plan around them:
``torch.compile``/inductor **refuses to run on Pascal** ("too old to be
supported"), and hand-rolled shift-and-accumulate convolution is *slower* than
cuDNN (0.42–0.58×). Both should pay off on Ampere or newer, where fp16/TF32 also
becomes available.

Normalisation
-------------
:class:`FrameNorm` is computed **once per frame** and passed into every band and
every training sample. This is load-bearing, not tidiness: ``prepare_frame`` in
the old stack derived robust median/IQR from whatever array it was handed, so a
band or crop normalised by its own statistics. Measured on an inhomogeneous
frame, a 128² tile featurised standalone differed from the same region of the
full-frame stack by **5.536** on channels whose true range is [−1.399, 0.954] —
4× outside the real range. Uncorrected that makes a preview a different
computation from the committed run, which is exactly the failure recorded for the
deleted classical engine (preview otsu 120.0 vs full-frame 146.0).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable

import numpy as np

from spyde.device_lock import accelerator_lock
from spyde.particles.features import import_torch, select_device
from spyde.particles.scribble import (
    LabelStore,
    ScribbleClass,
    _frame_getter,
)

log = logging.getLogger(__name__)

#: Discriminative scales, in pixels. Small — they are matched to the particle,
#: and the local-background job is done by :data:`DEFAULT_BG_SIGMA` instead.
DEFAULT_SIGMAS: tuple[float, ...] = (1.0, 2.0, 4.0)

#: Local background reference, in pixels of the FULL-resolution frame. Computed
#: on a decimated image, so this is cheap despite being large.
DEFAULT_BG_SIGMA: float = 25.0

#: Decimation factor for the background reference. 4 makes a σ=25 reference cost
#: ~1/16 of what it would at full resolution.
BG_DECIMATE: int = 4

#: Gaussian σ applied to the class logits before argmax. See the module
#: docstring — this is a matched filter on the decision variable, not a blur of
#: the image, and it is what stops a CNR~0.2 frame fragmenting into specks.
DEFAULT_LOGIT_SMOOTH: float = 3.0

#: Kernels are truncated at this many σ.
_TRUNCATE: float = 3.0

#: Robust statistics are estimated from at most this many pixels. A full
#: ``np.percentile`` over 16.7 M px is a ~35 ms sort and would dominate the
#: per-frame budget; 10⁵ puts median and IQR well inside their own sampling noise.
_STAT_SAMPLE_MAX: int = 100_000


# ── normalisation ────────────────────────────────────────────────────────────

class FrameNorm:
    """Robust centre/scale for ONE frame, computed once and then fixed.

    Every band, every crop and the training sampler standardise by the same two
    numbers, which is what makes a preview and the committed run the same
    computation. See the module docstring for the measured failure when they are
    allowed to differ.
    """

    __slots__ = ("centre", "scale")

    def __init__(self, centre: float, scale: float) -> None:
        self.centre = float(centre)
        self.scale = float(scale) if abs(scale) > 1e-6 else 1.0

    @classmethod
    def from_frame(cls, frame, max_sample: int = _STAT_SAMPLE_MAX) -> "FrameNorm":
        a = np.asarray(frame)
        flat = a.reshape(-1)
        finite = flat[np.isfinite(flat)] if flat.dtype.kind == "f" else flat
        if not finite.size:
            return cls(0.0, 1.0)
        step = max(1, finite.size // max_sample)
        s = finite[::step].astype(np.float32)
        q1, med, q3 = np.percentile(s, (25, 50, 75))
        return cls(float(med), float(q3 - q1))

    def apply(self, t):
        return (t - self.centre) / self.scale

    def to_dict(self) -> dict:
        return {"centre": self.centre, "scale": self.scale}


def _gauss1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


def _dgauss1d(sigma: float, radius: int) -> np.ndarray:
    """First derivative of a gaussian: the smoothed gradient in one pass."""
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    d = -(x / (sigma ** 2)) * g
    d -= d.mean()
    return (d / (np.abs(d).sum() / 2.0)).astype(np.float32)


# ── the feature bank ─────────────────────────────────────────────────────────

class FastFeatureBank:
    """Small separable bank + a decimated background reference.

    Channels, in this order (``channel_names`` is the definition):

    ``intensity``            the normalised frame
    ``gaussian_s{σ}``        one per σ
    ``dog_s{a}_s{b}``        differences of adjacent blurs
    ``highpass``             ``I − G_σmin * I``
    ``gradmag``              ``|∇ G_σmin * I|``
    ``localstd``             ``sqrt(G*(I²) − (G*I)²)`` — the noise profile
    ``background``           the decimated reference
    ``contrast``             ``background − G_σmin * I``
    ``contrast_over_noise``  ``contrast / localstd`` — a per-pixel CNR channel
    """

    def __init__(self, sigmas=DEFAULT_SIGMAS, *, bg_sigma: float = DEFAULT_BG_SIGMA,
                 bg_decimate: int = BG_DECIMATE, device=None) -> None:
        torch = import_torch()
        self.sigmas = tuple(float(s) for s in sigmas)
        self.bg_sigma = float(bg_sigma)
        self.bg_decimate = int(bg_decimate)
        self.device = select_device(device) if not hasattr(device, "type") else device
        self.n_sigma = len(self.sigmas)

        self.radius = max(1, int(math.ceil(_TRUNCATE * max(self.sigmas))))
        r = self.radius
        rows = [_gauss1d(s, r) for s in self.sigmas]
        rows.append(_dgauss1d(min(self.sigmas), r))
        self.K = len(rows)
        t = torch.as_tensor(np.stack(rows), device=self.device)
        self.w_row = t.view(self.K, 1, 1, 2 * r + 1).contiguous()
        self.w_col = t.view(self.K, 1, 2 * r + 1, 1).contiguous()

        if self.bg_sigma:
            sd = self.bg_sigma / self.bg_decimate
            rd = max(1, int(math.ceil(_TRUNCATE * sd)))
            kd = torch.as_tensor(_gauss1d(sd, rd), device=self.device)
            self.bg_r = rd
            self.bg_row = kd.view(1, 1, 1, 2 * rd + 1).contiguous()
            self.bg_col = kd.view(1, 1, 2 * rd + 1, 1).contiguous()

    # -- geometry ------------------------------------------------------------

    @property
    def halo(self) -> int:
        """Rows of context a band needs on each side to equal the unbanded result."""
        h = 2 * self.radius
        if self.bg_sigma:
            h = max(h, 2 * self.bg_r * self.bg_decimate + 2 * self.bg_decimate)
        return int(h)

    def channel_names(self) -> list[str]:
        def f(s):
            return f"{s:g}".replace(".", "p")
        names = ["intensity"]
        names += [f"gaussian_s{f(s)}" for s in self.sigmas]
        if self.n_sigma > 1:
            names += [f"dog_s{f(a)}_s{f(b)}"
                      for a, b in zip(self.sigmas, self.sigmas[1:])]
        names += ["highpass", "gradmag", "localstd"]
        if self.bg_sigma:
            names += ["background", "contrast", "contrast_over_noise"]
        return names

    @property
    def n_channels(self) -> int:
        return len(self.channel_names())

    # -- evaluation ----------------------------------------------------------

    def _background(self, img):
        """Decimate → blur → resample. A large reference at a small price."""
        import torch.nn.functional as F
        d = self.bg_decimate
        small = F.avg_pool2d(img, d, d)
        r = self.bg_r
        small = F.conv2d(small, self.bg_row, padding=(0, r))
        small = F.conv2d(small, self.bg_col, padding=(r, 0))
        return F.interpolate(small, size=img.shape[-2:], mode="bilinear",
                             align_corners=False)

    def __call__(self, img):
        """``img``: normalised ``(1,1,H,W)`` float32 → ``(1,C,H,W)``."""
        import torch
        import torch.nn.functional as F
        r, ns = self.radius, self.n_sigma

        a = F.conv2d(img, self.w_row, padding=(0, r))              # (1,K,H,W)
        A = F.conv2d(a, self.w_col, groups=self.K, padding=(r, 0))
        sq = img * img
        b = F.conv2d(sq, self.w_row[:1], padding=(0, r))
        B = F.conv2d(b, self.w_col[:1], groups=1, padding=(r, 0))

        blur = A[:, :ns]
        gx = A[:, ns:ns + 1]
        gy = F.conv2d(a[:, :1], self.w_col[ns:ns + 1], groups=1, padding=(r, 0))

        feats = [img, blur]
        if ns > 1:
            feats.append(blur[:, :-1] - blur[:, 1:])
        feats.append(img - blur[:, :1])
        feats.append(torch.sqrt(gx * gx + gy * gy + 1e-12))
        var = (B - blur[:, :1] * blur[:, :1]).clamp_min(0.0)
        sd = torch.sqrt(var + 1e-12)
        feats.append(sd)
        if self.bg_sigma:
            bg = self._background(img)
            contrast = bg - blur[:, :1]
            feats += [bg, contrast, contrast / (sd + 1e-3)]
        return torch.cat(feats, dim=1)

    def to_dict(self) -> dict:
        return {"sigmas": list(self.sigmas), "bg_sigma": self.bg_sigma,
                "bg_decimate": self.bg_decimate}


# ── the classifier ───────────────────────────────────────────────────────────

class FastScribbleClassifier:
    """Linear (or optionally shallow) per-pixel classifier over
    :class:`FastFeatureBank`, evaluated as 1×1 convolutions.

    API-compatible with :class:`spyde.particles.scribble.ScribbleClassifier`.

    Parameters
    ----------
    sigmas, bg_sigma
        The feature bank. See the module docstring for why the discriminative
        scales are small and the background reference is large.
    hidden
        Hidden width. **0 (linear) is the default and is what makes full-frame
        interactive** — see the module docstring for the 4.3 GB activation a
        64-wide layer materialises at 4096².
    logit_smooth
        Gaussian σ applied to the logits before argmax.
    epochs, lr, weight_decay, seed, device
        As the old engine. ``device=None`` auto-selects.
    """

    def __init__(self, spec=None, *, sigmas=DEFAULT_SIGMAS,
                 bg_sigma: float = DEFAULT_BG_SIGMA, hidden: int = 0,
                 logit_smooth: float = DEFAULT_LOGIT_SMOOTH, epochs: int = 300,
                 lr: float = 0.05, weight_decay: float = 1e-4, seed: int = 0,
                 device=None, band_rows: int = 1024) -> None:
        # `spec` is accepted positionally for drop-in compatibility; only its
        # sigmas are meaningful to this engine.
        if spec is not None and getattr(spec, "sigmas", None):
            sigmas = tuple(spec.sigmas)
        self.device = select_device(device) if not hasattr(device, "type") else device
        self.bank = FastFeatureBank(sigmas, bg_sigma=bg_sigma, device=self.device)
        self.hidden = int(hidden)
        self.logit_smooth = float(logit_smooth)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.band_rows = int(band_rows)
        self.classes: list[ScribbleClass] = []
        self._net = None
        self._mean = None
        self._std = None
        self._ls = None
        self.report: dict[str, Any] = {}
        if self.logit_smooth:
            self._build_logit_kernel()

    # -- state ---------------------------------------------------------------

    def _build_logit_kernel(self) -> None:
        torch = import_torch()
        r = max(1, int(math.ceil(_TRUNCATE * self.logit_smooth)))
        k = torch.as_tensor(_gauss1d(self.logit_smooth, r), device=self.device)
        self._ls = (r, k.view(1, 1, 1, 2 * r + 1).contiguous(),
                    k.view(1, 1, 2 * r + 1, 1).contiguous())

    @property
    def is_trained(self) -> bool:
        return self._net is not None

    @property
    def spec(self):
        """The feature bank, for callers that introspect ``clf.spec``."""
        return self.bank

    @property
    def halo(self) -> int:
        h = self.bank.halo
        if self._ls is not None:
            h += self._ls[0]
        return h

    @property
    def particle_class_ids(self) -> list[int]:
        return [c.id for c in self.classes if c.particle]

    @property
    def boundary_class_ids(self) -> list[int]:
        return [c.id for c in self.classes if c.boundary]

    @property
    def has_boundary(self) -> bool:
        return bool(self.boundary_class_ids)

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "this FastScribbleClassifier has not been trained — call fit() "
                "with a LabelStore first")

    # -- training ------------------------------------------------------------

    def _build_net(self, c_in: int, n_cls: int):
        torch = import_torch()
        torch.manual_seed(self.seed)
        if self.hidden:
            return torch.nn.Sequential(
                torch.nn.Conv2d(c_in, self.hidden, 1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(self.hidden, n_cls, 1)).to(self.device)
        return torch.nn.Conv2d(c_in, n_cls, 1).to(self.device)

    def _sample(self, frame: np.ndarray, flat_idx: np.ndarray, norm: FrameNorm,
                pad: int = 32):
        """Feature vectors at ``flat_idx``, featurising only their bounding box.

        This is why training does not care how big the frame is: a scribble
        occupies a few hundred pixels of a 16.7 M-pixel frame, so the box around
        it is a rounding error. Measured: 6 stroke boxes = 25.6 ms, 0.6% of a
        4096² frame, against 105 ms to featurise the whole thing.
        """
        torch = import_torch()
        h, w = frame.shape
        ys, xs = np.divmod(flat_idx.astype(np.int64), w)
        pad_h = self.halo + pad
        y0, y1 = max(0, int(ys.min()) - pad_h), min(h, int(ys.max()) + pad_h + 1)
        x0, x1 = max(0, int(xs.min()) - pad_h), min(w, int(xs.max()) + pad_h + 1)
        sub = np.ascontiguousarray(frame[y0:y1, x0:x1])
        t = torch.as_tensor(sub, device=self.device, dtype=torch.float32)[None, None]
        t = torch.nan_to_num(t, nan=float(np.nanmin(sub)) if sub.size else 0.0)
        f = self.bank(norm.apply(t))[0]
        ty = torch.as_tensor(ys - y0, device=self.device, dtype=torch.long)
        tx = torch.as_tensor(xs - x0, device=self.device, dtype=torch.long)
        return f[:, ty, tx].t().contiguous()

    def fit(self, store: LabelStore, frames, *,
            progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Train on every labelled pixel in *store*.

        Signature and report keys match the old engine so the caret, the training
        report line and the batch hand-off are unchanged.
        """
        torch = import_torch()
        get_frame = _frame_getter(frames)
        t_frames = sorted(store.labelled_frames())
        if not t_frames:
            raise RuntimeError("nothing has been painted yet")

        self.classes = [ScribbleClass(c.id, c.name, c.colour, c.particle, c.boundary)
                        for c in store.classes]

        t0 = time.perf_counter()
        xs_, ys_, w_ = [], [], []
        with accelerator_lock(self.device):
            for i, t in enumerate(t_frames):
                idx, cls = store.at(t)
                frame = np.asarray(get_frame(t))
                norm = FrameNorm.from_frame(frame)
                xs_.append(self._sample(frame, idx, norm))
                ys_.append(torch.as_tensor(cls.astype(np.int64), device=self.device))
                # Weight every STROKE equally rather than every pixel: the old
                # engine balanced per CLASS, so within a class a long stroke
                # outvoted a short one and painting more of one particle silently
                # reweighted the model toward it.
                sid = store.stroke_ids(t) if hasattr(store, "stroke_ids") else None
                if sid is None:
                    w_.append(torch.ones(len(idx), device=self.device))
                else:
                    s = torch.as_tensor(np.asarray(sid), device=self.device)
                    _, inv, cnt = torch.unique(s, return_inverse=True,
                                               return_counts=True)
                    w_.append(1.0 / cnt[inv].to(torch.float32))
                if progress:
                    progress(i + 1, len(t_frames))
            X = torch.cat(xs_, 0)
            y_raw = torch.cat(ys_, 0)
            wpx = torch.cat(w_, 0)
            featurise_s = time.perf_counter() - t0

            present = sorted({int(v) for v in y_raw.unique().tolist()})
            if len(present) < 2:
                raise RuntimeError(
                    "at least two classes must be painted before training "
                    f"(only class {present} has any pixels)")
            self.classes = [c for c in self.classes if c.id in present]
            remap = {cid: i for i, cid in enumerate(present)}
            y = torch.as_tensor([remap[int(v)] for v in y_raw.tolist()],
                                device=self.device, dtype=torch.long)

            self._mean = X.mean(0, keepdim=True)
            std = X.std(0, unbiased=False, keepdim=True)
            self._std = torch.where(std > 1e-6, std, torch.ones_like(std))
            Xn = (X - self._mean) / self._std

            t1 = time.perf_counter()
            n_cls = len(present)
            self._net = self._build_net(X.shape[1], n_cls)
            counts = torch.bincount(y, minlength=n_cls).to(Xn.dtype)
            cls_w = X.shape[0] / (n_cls * counts.clamp_min(1.0))
            sample_w = wpx * cls_w[y]
            sample_w = sample_w / sample_w.mean()
            opt = torch.optim.Adam(self._net.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)
            # (1, C, 1, N) so the very same 1x1-conv head trains and predicts
            Xc = Xn.t().reshape(1, X.shape[1], 1, -1)
            lossf = torch.nn.CrossEntropyLoss(reduction="none")
            loss = None
            for _ in range(self.epochs):
                opt.zero_grad(set_to_none=True)
                out = self._net(Xc).reshape(n_cls, -1).t()
                loss = (lossf(out, y) * sample_w).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred = self._net(Xc).reshape(n_cls, -1).t().argmax(1)
                acc = float((pred == y).float().mean().item())
            fit_s = time.perf_counter() - t1

        self.report = {
            "device": str(self.device),
            "engine": "fast",
            "n_pixels": int(X.shape[0]),
            "n_channels": int(X.shape[1]),
            "n_classes": n_cls,
            "has_boundary": self.has_boundary,
            "labelled_frames": [int(t) for t in t_frames],
            "pixels_per_class": {int(c.id): int((y == remap[c.id]).sum().item())
                                 for c in self.classes},
            "loss": float(loss.item()),
            "train_accuracy": acc,
            "featurise_s": featurise_s,
            "fit_s": fit_s,
        }
        log.info("fast engine trained: %d px x %d ch, %d classes, acc %.4f "
                 "(featurise %.2fs, fit %.2fs)", X.shape[0], X.shape[1], n_cls,
                 acc, featurise_s, fit_s)
        return dict(self.report)

    # -- prediction ----------------------------------------------------------

    def _smooth_logits(self, lg):
        import torch.nn.functional as F
        r, row, col = self._ls
        n = lg.shape[1]
        lg = F.conv2d(lg, row.expand(n, 1, 1, 2 * r + 1), groups=n, padding=(0, r))
        return F.conv2d(lg, col.expand(n, 1, 2 * r + 1, 1), groups=n,
                        padding=(r, 0))

    def _logits_banded(self, frame: np.ndarray, norm: FrameNorm | None = None):
        """Class logits for a whole frame, produced band by band.

        Nothing full-frame is materialised beyond the (n_classes, H, W) result —
        the C-channel stack only ever exists for one band.
        """
        torch = import_torch()
        self._require_trained()
        a = np.asarray(frame)
        finite = np.isfinite(a) if a.dtype.kind == "f" else np.ones(a.shape, bool)
        if not finite.all():
            a = np.where(finite, a, np.nanmin(a[finite]) if finite.any() else 0)
        norm = norm or FrameNorm.from_frame(a)
        H, W = a.shape
        halo = self.halo
        n_cls = len(self.classes)
        out = torch.empty((n_cls, H, W), device=self.device, dtype=torch.float32)
        step = max(64, self.band_rows)
        mean = self._mean.view(1, -1, 1, 1)
        std = self._std.view(1, -1, 1, 1)
        with accelerator_lock(self.device), torch.no_grad():
            src = torch.as_tensor(np.ascontiguousarray(a), device=self.device)
            for y0 in range(0, H, step):
                y1 = min(H, y0 + step)
                b0, b1 = max(0, y0 - halo), min(H, y1 + halo)
                t = src[b0:b1].to(torch.float32)[None, None]
                f = (self.bank(norm.apply(t)) - mean) / std
                lg = self._net(f)
                if self._ls is not None:
                    lg = self._smooth_logits(lg)
                out[:, y0:y1] = lg[0, :, y0 - b0:y0 - b0 + (y1 - y0)]
        return out, finite

    def predict_class_proba(self, frame, norm: FrameNorm | None = None):
        """``(n_classes, H, W)`` float32 softmax. Invalid pixels are zero."""
        torch = import_torch()
        lg, finite = self._logits_banded(frame, norm)
        with torch.no_grad():
            p = torch.softmax(lg, dim=0).cpu().numpy()
        p[:, ~finite] = 0.0
        return p

    def predict_foreground_boundary(self, frame, norm: FrameNorm | None = None):
        """``(foreground, boundary)``, each ``(H, W)`` float32 in [0, 1].

        One pass gives both, which is what lets ``split_instances`` take its
        connected-components route. ``boundary`` is ``None`` when no boundary
        class carries trained weight.
        """
        torch = import_torch()
        lg, finite = self._logits_banded(frame, norm)
        ids = [c.id for c in self.classes]
        with torch.no_grad():
            p = torch.softmax(lg, dim=0)
            fg_rows = [i for i, c in enumerate(self.classes) if c.particle]
            bd_rows = [i for i, c in enumerate(self.classes) if c.boundary]
            fg = (p[fg_rows].sum(0) if fg_rows
                  else torch.zeros_like(p[0])).cpu().numpy()
            bnd = p[bd_rows].sum(0).cpu().numpy() if bd_rows else None
        fg[~finite] = 0.0
        if bnd is not None:
            bnd[~finite] = 0.0
        del ids
        return fg, bnd

    # -- persistence ---------------------------------------------------------

    def save(self, path: str) -> str:
        """One ``.npz`` with the weights, the bank and the standardisation.

        ``allow_pickle=False`` on the way back in — this file crosses to dask
        workers, and a trained head must never be a pickle.
        """
        self._require_trained()
        sd = {k: v.detach().cpu().numpy() for k, v in self._net.state_dict().items()}
        np.savez(
            path,
            _engine=np.array("fast"),
            _sigmas=np.asarray(self.bank.sigmas, dtype=np.float64),
            _bg_sigma=np.asarray(self.bank.bg_sigma),
            _bg_decimate=np.asarray(self.bank.bg_decimate),
            _hidden=np.asarray(self.hidden),
            _logit_smooth=np.asarray(self.logit_smooth),
            _band_rows=np.asarray(self.band_rows),
            _mean=self._mean.detach().cpu().numpy(),
            _std=self._std.detach().cpu().numpy(),
            _class_ids=np.asarray([c.id for c in self.classes], np.int64),
            _class_names=np.asarray([c.name for c in self.classes]),
            _class_colours=np.asarray([c.colour for c in self.classes]),
            _class_particle=np.asarray([c.particle for c in self.classes]),
            _class_boundary=np.asarray([c.boundary for c in self.classes]),
            **{f"w_{k}": v for k, v in sd.items()},
        )
        return path

    @classmethod
    def load(cls, path: str, *, device=None) -> "FastScribbleClassifier":
        torch = import_torch()
        z = np.load(path, allow_pickle=False)
        obj = cls(sigmas=tuple(z["_sigmas"].tolist()),
                  bg_sigma=float(z["_bg_sigma"]),
                  hidden=int(z["_hidden"]),
                  logit_smooth=float(z["_logit_smooth"]),
                  band_rows=int(z["_band_rows"]),
                  device=device)
        obj.bank.bg_decimate = int(z["_bg_decimate"])
        obj.classes = [
            ScribbleClass(int(i), str(n), str(c), bool(p), bool(b))
            for i, n, c, p, b in zip(z["_class_ids"], z["_class_names"],
                                     z["_class_colours"], z["_class_particle"],
                                     z["_class_boundary"])]
        obj._mean = torch.as_tensor(z["_mean"], device=obj.device)
        obj._std = torch.as_tensor(z["_std"], device=obj.device)
        n_ch = obj._mean.shape[1]
        obj._net = obj._build_net(n_ch, len(obj.classes))
        sd = {k[2:]: torch.as_tensor(z[k], device=obj.device)
              for k in z.files if k.startswith("w_")}
        obj._net.load_state_dict(sd)
        obj._net.eval()
        return obj

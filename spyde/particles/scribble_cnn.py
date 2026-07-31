"""
scribble_cnn.py — PROTOTYPE. A small CNN trained on the same scribbles, as an
A/B alternative to :class:`spyde.particles.scribble.ScribbleClassifier`.

**This is a prototype and is deliberately not wired into anything.** It is not
in ``spyde.particles.__init__``, no action dispatches it, no caret exposes it
and the batch path does not know it exists. It exists so the two engines can be
measured against each other on identical scribbles
(``spyde/tests/benchmark_scribble_cnn.py``); the shipped engine is untouched.

What it replaces, and what it does not
--------------------------------------
The shipped engine is two stages: 36 hand-crafted channels
(:mod:`spyde.particles.features`) and a per-pixel MLP over them. This replaces
**both** with one small U-Net that reads the raw (robustly standardised) frame
and emits per-class logits at input resolution. Everything downstream is
unchanged, and that is the whole point of the design:

* :meth:`predict_foreground_boundary` returns the same ``(foreground,
  boundary)`` pair, so :func:`spyde.particles.classical.split_instances` takes
  its connected-components route exactly as it already does for the MLP — no new
  instance decoder, no star-convex head, no second downstream path to maintain.
* The class list is :class:`~spyde.particles.scribble.ScribbleClass`, so
  ``particle`` and ``boundary`` flags mean what they already mean, and several
  particle phases still sum into one foreground map.
* Non-finite input pixels are forced to zero probability in every class, the
  same NaN-border contract (plan trap 2) — both engines call
  :func:`spyde.particles.features.prepare_frame`, so they see a bit-identical
  input image and a bit-identical validity mask.

Training on sparse scribbles
----------------------------
A scribble is a few thousand pixels in a 16.7 M-pixel frame, so two things
follow and both are load-bearing:

**The loss is masked.** Unlabelled pixels carry ``ignore_index`` and contribute
nothing. A dense loss would need a dense target, which does not exist — the user
painted strokes, not a segmentation.

**Training runs on CROPS around the scribbles, never on whole frames.** This is
what makes the train time tolerable and it is also the memory-safety rule
(CLAUDE.md): a movie is never materialised, one frame is read at a time, and
only the crop windows that actually contain labelled pixels are ever pushed
through the net. Whole-frame training at 4096² would be ~0.3 s *per step* for
the small net; a batch of eight 128² crops is ~1/16 of one frame.

Inference is TILED
------------------
Measured on this dev box (TITAN X Pascal, fp32, 4096²): ``base=32, levels=3``
costs **0.737 s tiled at 16×1024²** against **7.72 s whole-frame** — a 10×
difference that is memory pressure on a 12 GB card, not arithmetic. So the
whole-frame path is not offered above :data:`TILE_ABOVE_PIXELS`.

Three measured traps that shaped this file
------------------------------------------
1. **fp16 is SLOWER on Pascal** (fp16:fp32 rate 1:64). ``autocast`` measured
   7.72 → 17.9 s and 14.5 → 22.5 s. There is deliberately no AMP option here.
2. **Level count matters more than parameter count.** ``base=32, levels=2``
   measured **25 s** whole-frame — far worse than ``levels=3`` with 4× the
   parameters — because fewer downsamples leaves more work at full resolution.
   So :data:`CONFIGS` names the two configurations worth running and a caller
   who invents a third should measure it before believing it.
3. **Tile anything above the tiny net.** See above.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable

import numpy as np

from spyde.device_lock import accelerator_lock
from spyde.particles.features import import_torch, prepare_frame, select_device
from spyde.particles.scribble import LabelStore, ScribbleClass, _frame_getter

log = logging.getLogger(__name__)

#: The two configurations worth running, per the forward-pass sweep in the
#: module docstring. ``(base, levels)`` → the label the benchmark reports.
CONFIGS: dict[str, tuple[int, int]] = {
    "tiny": (16, 2),      # 117 k params, 0.289 s whole-frame at 4096²
    "small": (32, 3),     # 1.9 M params, 0.737 s tiled at 4096²
}

#: Above this many pixels a frame is always tiled for inference (see the module
#: docstring — 10× on the small net, and it only gets worse with frame size).
TILE_ABOVE_PIXELS: int = 2048 * 2048

#: Default inference tile edge, in pixels. 1024 is what the 16×1024² measurement
#: used and it is a multiple of every ``2**levels`` in :data:`CONFIGS`.
TILE_EDGE: int = 1024

#: Default training crop edge. A multiple of 8, so it is legal for ``levels`` up
#: to 3 without padding.
CROP_EDGE: int = 128

#: Ignored target value for an unlabelled pixel. Not
#: :data:`~spyde.particles.scribble.UNLABELLED` by coincidence — it is the same
#: -1, and ``cross_entropy(ignore_index=-1)`` is what makes the sparse loss work.
IGNORE = -1


# ── the net ──────────────────────────────────────────────────────────────────

_SEG_UNET = None


def _seg_unet_class():
    """Define (once) the ``SpotUNet`` subclass with a segmentation head.

    Built lazily rather than at module scope because ``spyde.models.unet``
    imports torch eagerly, and everything else in :mod:`spyde.particles` is
    careful not to (:func:`spyde.particles.features.import_torch`).
    """
    global _SEG_UNET
    if _SEG_UNET is not None:
        return _SEG_UNET

    import torch.nn as nn

    from spyde.models.unet import SpotUNet

    class SegUNet(SpotUNet):
        """``SpotUNet``'s encoder/decoder with a K-class 1×1 head.

        The body is the vendored one, unmodified — that is the budget (no
        stardist, no cellpose, no SAM), and it is why the forward-pass costs
        measured for ``SpotUNet`` transfer here directly. Only the heads change:
        the spot detector's ``(heatmap, offset)`` pair is replaced by one
        ``Conv2d(base, K, 1)`` emitting per-class logits at input resolution.
        The spot heads are DELETED rather than left unused — otherwise they sit
        in the optimiser's parameter list receiving no gradient and are saved
        with the model.
        """

        def __init__(self, n_classes: int, base: int = 16, levels: int = 2):
            super().__init__(in_ch=1, base=base, levels=levels)
            del self.head_hm, self.head_off
            self.head_seg = nn.Conv2d(base, int(n_classes), 1)
            nn.init.zeros_(self.head_seg.bias)

        def forward(self, x):
            import torch
            feats = []
            h = x
            for i, enc in enumerate(self.enc):
                h = enc(h if i == 0 else self.pool(h))
                feats.append(h)
            d = feats[-1]
            for j, (up, dec) in enumerate(zip(self.up, self.dec)):
                d = dec(torch.cat([up(d), feats[self.levels - 1 - j]], 1))
            return self.head_seg(d)

    _SEG_UNET = SegUNet
    return SegUNet


def build_net(n_classes: int, *, base: int = 16, levels: int = 2, seed: int = 0):
    """A :class:`~spyde.models.unet.SpotUNet` body with a K-class 1×1 head.

    Seeded through ``fork_rng`` for the same reason
    :func:`spyde.particles.scribble._build_mlp` is: the global torch RNG is
    shared with everything else in the process, so initialising from it would
    make "same seed, same labels, same model" depend on what ran first — and
    would silently shift every other consumer's random stream. ``devices=[]``
    forks the CPU generator only; forking CUDA's would initialise the CUDA
    context as a side effect.
    """
    torch = import_torch()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return _seg_unet_class()(int(n_classes), int(base), int(levels))


# ── crop planning ────────────────────────────────────────────────────────────

def crop_windows(ys: np.ndarray, xs: np.ndarray, shape: tuple[int, int],
                 crop: int, mult: int) -> list[tuple[int, int, int, int]]:
    """``(y0, y1, x0, x1)`` windows covering every labelled pixel, deduplicated.

    Each labelled pixel is assigned to the ONE window that contains it most
    centrally (a half-crop grid, snapped), and the unique windows are returned.
    Centrally, because the U-Net's receptive field is what gives a pixel its
    context: a labelled pixel pinned to a crop's edge is classified from half the
    surroundings it will have at inference time, and that mismatch is invisible
    in the training loss.

    The window edge is clamped to a multiple of *mult* (``2**levels``) so the
    pooling stack divides evenly, and to the frame, so a small frame trains as
    one whole-frame crop rather than as a reflect-padded fiction.
    """
    h, w = int(shape[0]), int(shape[1])
    ch = min(int(crop), (h // mult) * mult)
    cw = min(int(crop), (w // mult) * mult)
    if ch < mult or cw < mult:
        raise ValueError(
            f"a {h}x{w} frame is too small to train a {mult}-divisible crop "
            f"from; use fewer levels")

    # Half-crop grid: round the pixel to the nearest window CENTRE, then clamp
    # the resulting top-left into the frame.
    def tops(v: np.ndarray, span: int, extent: int) -> np.ndarray:
        step = max(1, span // 2)
        t = np.rint((v - span / 2.0) / step).astype(np.int64) * step
        return np.clip(t, 0, extent - span)

    ty, tx = tops(ys, ch, h), tops(xs, cw, w)
    seen = {(int(a), int(b)) for a, b in zip(ty, tx)}
    return [(y, y + ch, x, x + cw) for y, x in sorted(seen)]


# ── the head ─────────────────────────────────────────────────────────────────

class ScribbleCNN:
    """Prototype CNN pixel classifier — same output contract as the MLP engine.

    Parameters
    ----------
    base, levels
        U-Net width and downsample count. See :data:`CONFIGS`; ``levels`` is the
        knob that decides both the receptive field and the cost, and the two
        do not trade off the way parameter count suggests (module docstring
        trap 2).
    crop
        Training crop edge, px. Training never touches a whole frame.
    steps, batch, lr, weight_decay
        Adam over crop mini-batches. ``steps`` is optimiser steps and not epochs
        on purpose: the number of crops depends on how much the user painted and
        how far apart, so "epochs" is not a fixed amount of work and would make
        the train time depend on the scribble layout in a way the user cannot
        predict.
    augment
        Random dihedral (flip/transpose) augmentation per crop per step. Nearly
        free and material at this label count — a few thousand labelled pixels
        is a very small training set for a conv net.
    tile
        Inference tile edge, px. See the module docstring: above
        :data:`TILE_ABOVE_PIXELS` this is not optional.
    device
        ``None`` auto-selects CUDA/MPS/CPU. Pass ``"cpu"`` under pytest —
        torch-CUDA segfaults in that process on Windows (CLAUDE.md).
    """

    def __init__(
        self,
        *,
        base: int = 16,
        levels: int = 2,
        crop: int = CROP_EDGE,
        steps: int = 300,
        batch: int = 8,
        lr: float = 3e-3,
        weight_decay: float = 1e-4,
        augment: bool = True,
        tile: int = TILE_EDGE,
        seed: int = 0,
        device=None,
    ) -> None:
        self.base = int(base)
        self.levels = int(levels)
        self.crop = int(crop)
        self.steps = int(steps)
        self.batch = int(batch)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.augment = bool(augment)
        self.tile = int(tile)
        self.seed = int(seed)
        self.device = select_device(device) if not hasattr(device, "type") else device
        self.classes: list[ScribbleClass] = []
        self._net = None
        self.report: dict[str, Any] = {}

    # -- state ---------------------------------------------------------------

    @property
    def mult(self) -> int:
        """Pixel multiple the pooling stack requires: ``2**levels``."""
        return 1 << self.levels

    @property
    def is_trained(self) -> bool:
        return self._net is not None

    @property
    def particle_class_ids(self) -> list[int]:
        return [c.id for c in self.classes if c.particle]

    @property
    def boundary_class_ids(self) -> list[int]:
        return [c.id for c in self.classes if c.boundary]

    @property
    def has_boundary(self) -> bool:
        return bool(self.boundary_class_ids)

    def num_params(self) -> int:
        self._require_trained()
        return int(sum(p.numel() for p in self._net.parameters()))

    def _require_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(
                "this ScribbleCNN has not been trained — call fit() with a "
                "LabelStore first")

    # -- training ------------------------------------------------------------

    def fit(self, store: LabelStore, frames, *,
            progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Train on every labelled pixel in *store*, from crops around them.

        Same signature and same *frames* vocabulary as
        :meth:`spyde.particles.scribble.ScribbleClassifier.fit` — a callable, a
        ``{t: frame}`` mapping, a 3-D stack or a HyperSpy signal — so the two
        engines are driven identically by the benchmark and by any future caret.
        One frame is read at a time and never held.

        Returns the training report: per-class pixel counts, crop count, final
        loss and training accuracy over the labelled pixels, the wall-clock split
        between preparing crops and optimising, and the device.
        """
        torch = import_torch()
        import torch.nn.functional as F

        if len(store) == 0:
            raise ValueError("nothing painted yet — the label store is empty")

        get_frame = _frame_getter(frames)
        t_frames = store.labelled_frames()

        counts_by_id: dict[int, int] = {}
        for t in t_frames:
            _idx, cls = store.at(t)
            for v, n in zip(*np.unique(cls, return_counts=True)):
                counts_by_id[int(v)] = counts_by_id.get(int(v), 0) + int(n)
        present = sorted(counts_by_id)
        if len(present) < 2:
            raise ValueError(
                f"only one class is painted (id {present[0]}) — a classifier "
                "needs at least two, e.g. a particle and some background")
        self.classes = [store.class_by_id(c) for c in present]
        col = {cid: k for k, cid in enumerate(present)}
        n_out = len(present)

        # CUDA's autograd engine has to be initialised on the thread that will
        # run backward, or the first backward segfaults on Windows (CLAUDE.md
        # § GPU Computing). `fit` is the dispatch point, so warm it here rather
        # than hoping the caller did.
        _warmup_autograd(self.device)

        t0 = time.perf_counter()
        crops_x: list[np.ndarray] = []
        crops_y: list[np.ndarray] = []
        h, w = store.frame_shape
        for i, t in enumerate(t_frames):
            idx, cls = store.at(t)
            if not idx.size:
                continue
            frame = np.asarray(get_frame(t))
            if tuple(frame.shape) != tuple(store.frame_shape):
                raise ValueError(
                    f"frame {t} is {frame.shape} but the label store holds "
                    f"{store.frame_shape} labels — the flat indices would land "
                    "in the wrong pixels")
            # Identical preparation to the MLP engine: same NaN fill, same
            # robust standardisation, so an A/B is about the head and not about
            # what the two were shown.
            img = prepare_frame(frame).image
            ys, xs = np.divmod(idx, w)
            target = np.full((h, w), IGNORE, dtype=np.int64)
            target[ys, xs] = [col[int(c)] for c in cls]
            for (y0, y1, x0, x1) in crop_windows(ys, xs, (h, w), self.crop,
                                                 self.mult):
                sub = target[y0:y1, x0:x1]
                if not (sub >= 0).any():         # pragma: no cover — defensive
                    continue
                crops_x.append(np.ascontiguousarray(img[y0:y1, x0:x1]))
                crops_y.append(np.ascontiguousarray(sub))
            if progress is not None:
                progress(i + 1, len(t_frames))
        if not crops_x:                          # pragma: no cover — defensive
            raise ValueError("no training crop contained a labelled pixel")

        # ONE lock acquisition around the whole fit, for the same reason
        # `ScribbleClassifier.fit` takes one: every line below submits to the
        # device and MPS needs all of them serialised. Null context off MPS.
        with accelerator_lock(self.device):
            X = torch.as_tensor(np.stack(crops_x)[:, None],
                                device=self.device, dtype=torch.float32)
            Y = torch.as_tensor(np.stack(crops_y), device=self.device)
            n_crops = int(X.shape[0])

            counts = torch.zeros(n_out, dtype=torch.float32, device=self.device)
            for cid, n in counts_by_id.items():
                counts[col[cid]] = float(n)
            # Class-balanced, exactly as the MLP is: a user paints a few dabs on
            # particles and sweeps whole regions of background, and unweighted
            # cross-entropy on a 40:1 split learns "background".
            weight = counts.sum() / (n_out * counts.clamp_min(1.0))

            t1 = time.perf_counter()
            self._net = build_net(n_out, base=self.base, levels=self.levels,
                                  seed=self.seed).to(self.device)
            self._net.train()
            opt = torch.optim.Adam(self._net.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)
            rng = np.random.default_rng(self.seed)
            batch = min(self.batch, n_crops)

            loss = float("nan")
            # Pin backward to this thread: torch's multithreaded autograd
            # engine segfaults under CUDA on Windows off the main thread.
            prev_mt = True
            try:
                torch.autograd.set_multithreading_enabled(False)
                prev_mt = False
                for step in range(self.steps):
                    sel = rng.choice(n_crops, size=batch,
                                     replace=batch > n_crops)
                    xb, yb = X[sel], Y[sel]
                    if self.augment:
                        xb, yb = _augment(torch, xb, yb, rng)
                    opt.zero_grad(set_to_none=True)
                    out = self._net(xb)
                    lo = F.cross_entropy(out, yb, weight=weight,
                                         ignore_index=IGNORE)
                    lo.backward()
                    opt.step()
                    loss = float(lo.detach())
            finally:
                if not prev_mt:
                    torch.autograd.set_multithreading_enabled(True)

            self._net.eval()
            with torch.no_grad():
                hit = tot = 0
                for k in range(0, n_crops, max(1, batch)):
                    xb, yb = X[k:k + batch], Y[k:k + batch]
                    pred = self._net(xb).argmax(dim=1)
                    m = yb >= 0
                    hit += int((pred[m] == yb[m]).sum())
                    tot += int(m.sum())
                acc = hit / max(1, tot)
        t_fit = time.perf_counter() - t1

        self.report = {
            "engine": "cnn",
            "device": str(self.device),
            "base": self.base,
            "levels": self.levels,
            "params": self.num_params(),
            "n_pixels": int(sum(counts_by_id.values())),
            "n_classes": n_out,
            "n_crops": n_crops,
            "crop": [int(X.shape[-2]), int(X.shape[-1])],
            "steps": self.steps,
            "batch": batch,
            "has_boundary": bool([c for c in self.classes if c.boundary]),
            "labelled_frames": list(t_frames),
            "pixels_per_class": {str(cid): int(n)
                                 for cid, n in sorted(counts_by_id.items())},
            "loss": loss,
            "train_accuracy": acc,
            "crops_s": t1 - t0,
            "fit_s": t_fit,
        }
        if progress is not None:
            progress(len(t_frames), len(t_frames))
        log.info("[scribble-cnn] base=%d levels=%d trained on %d px in %d crops,"
                 " %d classes: acc %.3f (crops %.2f s, fit %.2f s, %s)",
                 self.base, self.levels, self.report["n_pixels"], n_crops,
                 n_out, acc, self.report["crops_s"], t_fit, self.device)
        return self.report

    # -- prediction ----------------------------------------------------------

    def predict_class_proba(self, frame) -> np.ndarray:
        """``(K, H, W)`` float32 softmax over the trained classes.

        Column *k* is ``self.classes[k]``, and non-finite source pixels get
        probability 0 in every class — the same contract, verbatim, as
        :meth:`spyde.particles.scribble.ScribbleClassifier.predict_class_proba`,
        so a caller cannot tell the two engines apart by their output shape or
        their NaN handling.
        """
        self._require_trained()
        torch = import_torch()
        prepared = prepare_frame(frame)
        img = prepared.image
        h, w = img.shape
        k = len(self.classes)

        with accelerator_lock(self.device):
            out = np.empty((k, h, w), dtype=np.float32)
            self._net.eval()
            with torch.no_grad():
                for (y0, y1, x0, x1, py0, py1, px0, px1) in self._tiles(h, w):
                    sub = img[y0:y1, x0:x1]
                    t = torch.as_tensor(sub, device=self.device,
                                        dtype=torch.float32)[None, None]
                    t, (pad_b, pad_r) = _pad_to_multiple(torch, t, self.mult)
                    p = torch.softmax(self._net(t), dim=1)[0]
                    if pad_b or pad_r:
                        p = p[:, :y1 - y0, :x1 - x0]
                    out[:, py0:py1, px0:px1] = (
                        p[:, py0 - y0:py1 - y0, px0 - x0:px1 - x0]
                        .detach().cpu().numpy())

        out[:, ~prepared.valid] = 0.0
        return out

    def _tiles(self, h: int, w: int):
        """``(read window, write window)`` pairs for tiled inference.

        Each tile is read with a halo of the receptive field and written without
        it, so the tiled result matches an untiled one everywhere except where a
        halo runs off the frame — which is where reflect padding would have
        invented the same context anyway. A frame small enough to fit is one
        tile with no halo at all.
        """
        edge = self.tile
        if h * w <= TILE_ABOVE_PIXELS and h <= edge and w <= edge:
            yield (0, h, 0, w, 0, h, 0, w)
            return
        halo = _halo(self.levels)
        for py0 in range(0, h, edge):
            py1 = min(h, py0 + edge)
            y0, y1 = max(0, py0 - halo), min(h, py1 + halo)
            for px0 in range(0, w, edge):
                px1 = min(w, px0 + edge)
                x0, x1 = max(0, px0 - halo), min(w, px1 + halo)
                yield (y0, y1, x0, x1, py0, py1, px0, px1)

    def predict_foreground_boundary(
        self, frame
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """``(foreground, boundary)`` from ONE pass over the frame.

        *boundary* is None when no trained class is marked
        :attr:`~spyde.particles.scribble.ScribbleClass.boundary` — None and not
        a zero map, because :func:`spyde.particles.classical.split_instances`
        reads the two differently ("no boundary taught, use the watershed"
        versus "a boundary was taught and this frame has none of it").
        """
        self._require_trained()
        proba = self.predict_class_proba(frame)
        wanted = [k for k, c in enumerate(self.classes) if c.particle]
        if not wanted:
            raise RuntimeError(
                "no trained class is marked as a particle class, so there is no "
                "foreground to report — set ScribbleClass.particle on at least "
                f"one of {[c.name for c in self.classes]} and retrain")
        edge = [k for k, c in enumerate(self.classes) if c.boundary]
        return (_sum_planes(proba, wanted),
                _sum_planes(proba, edge) if edge else None)

    def predict_proba(self, frame) -> np.ndarray:
        """``(H, W)`` float32 foreground probability — the plan §0.2 spine's
        input, interchangeable with the MLP engine's."""
        return self.predict_foreground_boundary(frame)[0]

    def predict_boundary_proba(self, frame) -> np.ndarray | None:
        return self.predict_foreground_boundary(frame)[1]

    def segment(self, frame, params=None) -> np.ndarray:
        """Probability → labelled instances via the SHARED split.

        Byte-for-byte the same forwarder the MLP engine has, including passing
        the boundary down when one was taught. That is the design claim this
        prototype exists to test: a CNN that predicts the boundary class plugs
        into machinery that already exists.
        """
        from spyde.particles.classical import SegmentParams, split_instances
        fg, bnd = self.predict_foreground_boundary(frame)
        return split_instances(fg, params or SegmentParams(), boundary=bnd)


# ── helpers ──────────────────────────────────────────────────────────────────

def _halo(levels: int) -> int:
    """Tile overlap, px: roughly the net's receptive field, rounded up.

    Two 3×3 convolutions per level, doubling in stride each level down, plus the
    decoder's mirror — ~30 px at ``levels=2`` and ~120 px at ``levels=3``. Padded
    to a comfortable multiple rather than derived exactly: a halo that is too
    small shows as seams, a halo that is too large costs a few percent.
    """
    return int(32 * (1 << max(0, int(levels) - 2)))


def _pad_to_multiple(torch, t, mult: int):
    """Reflect-pad an ``(N, C, h, w)`` tensor's bottom/right up to *mult*.

    Bottom/right only, so the output's ``[:h, :w]`` is the input's pixels at the
    same indices — a symmetric pad would shift every coordinate and silently
    move the whole prediction by a pixel.
    """
    import torch.nn.functional as F
    h, w = int(t.shape[-2]), int(t.shape[-1])
    pad_b, pad_r = (-h) % mult, (-w) % mult
    if pad_b or pad_r:
        t = F.pad(t, (0, pad_r, 0, pad_b), mode="reflect")
    return t, (pad_b, pad_r)


def _augment(torch, xb, yb, rng):
    """Random dihedral transform of a crop batch (whole batch at once).

    Per-batch rather than per-crop: it is one indexing op instead of eight, the
    batch is re-sampled every step anyway so a crop still sees every transform
    over a run, and at this step count the difference is unmeasurable.
    """
    if rng.random() < 0.5:
        xb, yb = torch.flip(xb, (-1,)), torch.flip(yb, (-1,))
    if rng.random() < 0.5:
        xb, yb = torch.flip(xb, (-2,)), torch.flip(yb, (-2,))
    if xb.shape[-1] == xb.shape[-2] and rng.random() < 0.5:
        xb, yb = xb.transpose(-1, -2), yb.transpose(-1, -2)
    return xb.contiguous(), yb.contiguous()


def _sum_planes(proba: np.ndarray, cols: list[int]) -> np.ndarray:
    """Sum the given planes of a ``(K, H, W)`` softmax, float32.

    The one-column case is a plain slice for the reason
    :func:`spyde.particles.scribble._sum_planes` documents: fancy-indexing a
    4096² map builds a copy before reducing it, measured at 113 ms for the pair.
    """
    if len(cols) == 1:
        plane = proba[cols[0]]
        return plane if plane.dtype == np.float32 else plane.astype(np.float32)
    return proba[cols].sum(axis=0).astype(np.float32)


_AUTOGRAD_WARMED = False


def _warmup_autograd(device) -> None:
    """One trivial backward on the CALLING thread before the real one.

    torch's CUDA autograd backward segfaults on Windows the first time it runs
    on a thread whose engine has not been initialised. Same mitigation, for the
    same reason, as :func:`spyde.actions.vector_orientation_gpu.warmup_autograd`
    — duplicated rather than imported so this prototype does not reach into the
    orientation-mapping package.
    """
    global _AUTOGRAD_WARMED
    if _AUTOGRAD_WARMED or getattr(device, "type", str(device)) != "cuda":
        return
    try:
        torch = import_torch()
        x = torch.zeros(1, device=device, requires_grad=True)
        (x * 2).sum().backward()
        _AUTOGRAD_WARMED = True
    except Exception as e:                                    # pragma: no cover
        log.debug("CUDA autograd warmup skipped: %s", e)

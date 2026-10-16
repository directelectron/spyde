"""
classifier.py — a linear per-pixel classifier over the feature bank.

The head is one 1×1 convolution: the SAME module trains on the sampled stroke
pixels (laid out as a ``(1, C, 1, N)`` tensor) and predicts a whole band of a
field. Width was measured never to buy separability (training accuracy 1.0 at
every epoch count from 10 to 300), while a 64-wide hidden layer materialises a
4.3 GB activation at 4096².

Training featurises ONLY the bounding boxes of the strokes, so it does not care
how big the field is. Prediction runs band by band with a halo, so the
C-channel stack only ever exists for one band. The class logits are smoothed
with a small Gaussian before the softmax: at a contrast-to-noise of ~0.2 an
independent decision per pixel shatters a real particle into specks (7095 →
4842 instances at the same threshold once the decision variable is blurred).

The classifier pickles and unpickles as plain numpy state and rebuilds its
tensors on first use, so a lazy label movie can close over it and be computed
in any process.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

from spyde.device_lock import accelerator_lock
from spyde.segmentation.features import (
    DEFAULT_BACKGROUND_SIGMA, DEFAULT_SIGMAS, TRUNCATE, FeatureBank,
    Normalisation, align_down, gaussian_kernel, select_device,
)
from spyde.segmentation.labels import BOUNDARY, PARTICLE, Labels

#: Gaussian sigma applied to the class logits before the softmax.
DEFAULT_LOGIT_SMOOTH: float = 3.0


class PixelClassifier:
    """Scribble-trained per-pixel classifier.

    Parameters
    ----------
    sigmas, background_sigma
        The feature bank (:class:`~spyde.segmentation.features.FeatureBank`).
    logit_smooth
        Gaussian sigma on the logits, in pixels. 0 disables.
    epochs, learning_rate, weight_decay, seed
        The Adam fit over the sampled stroke pixels.
    device
        A torch device or name; ``None`` picks CUDA, MPS, then CPU.
    band_rows
        Rows per prediction band.
    """

    def __init__(self, *, sigmas=DEFAULT_SIGMAS,
                 background_sigma: float = DEFAULT_BACKGROUND_SIGMA,
                 logit_smooth: float = DEFAULT_LOGIT_SMOOTH, epochs: int = 300,
                 learning_rate: float = 0.05, weight_decay: float = 1e-4,
                 seed: int = 0, device=None, band_rows: int = 1024) -> None:
        self.sigmas = tuple(float(sigma) for sigma in sigmas)
        self.background_sigma = float(background_sigma)
        self.logit_smooth = float(logit_smooth)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.band_rows = int(band_rows)
        self._device_name = None if device is None else str(device)
        self.classes: list[int] = []          # class ids with trained weight, in row order
        self.report: dict[str, Any] = {}
        self._weights: dict[str, np.ndarray] | None = None   # numpy state, the truth
        self._device = None
        self._bank = None
        self._net = None
        self._feature_mean = None
        self._feature_std = None
        self._smooth_kernels = None

    # -- state ---------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        return self._weights is not None

    @property
    def device(self):
        if self._device is None:
            self._device = select_device(self._device_name)
        return self._device

    @property
    def bank(self) -> FeatureBank:
        if self._bank is None:
            self._bank = FeatureBank(self.sigmas, background_sigma=self.background_sigma,
                                     device=self.device)
        return self._bank

    @property
    def halo(self) -> int:
        halo = self.bank.halo
        if self.logit_smooth:
            halo += int(math.ceil(TRUNCATE * self.logit_smooth))
        return halo

    def to_dict(self) -> dict[str, Any]:
        """Everything a trained classifier is, as JSON-safe values and numpy arrays."""
        if not self.is_trained:
            raise RuntimeError("the classifier has not been trained yet")
        return {
            "sigmas": list(self.sigmas), "background_sigma": self.background_sigma,
            "logit_smooth": self.logit_smooth, "band_rows": self.band_rows,
            "classes": list(self.classes), "report": dict(self.report),
            **{name: np.asarray(array) for name, array in self._weights.items()},
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any], *, device=None) -> "PixelClassifier":
        classifier = cls(sigmas=state["sigmas"], background_sigma=state["background_sigma"],
                         logit_smooth=state["logit_smooth"], band_rows=state["band_rows"],
                         device=device)
        classifier.classes = [int(class_id) for class_id in state["classes"]]
        classifier.report = dict(state.get("report") or {})
        classifier._weights = {name: np.asarray(state[name])
                               for name in ("feature_mean", "feature_std", "weight", "bias")}
        return classifier

    def save(self, path: str) -> None:
        state = self.to_dict()
        report = state.pop("report")
        np.savez(path, report=np.array(_jsonable(report)), **state)

    @classmethod
    def load(cls, path: str, *, device=None) -> "PixelClassifier":
        import json
        with np.load(path, allow_pickle=False) as archive:
            state = {name: archive[name] for name in archive.files}
        state["report"] = json.loads(str(state.pop("report")))
        state["sigmas"] = state["sigmas"].tolist()
        for name in ("background_sigma", "logit_smooth", "band_rows"):
            state[name] = state[name].item()
        return cls.from_dict(state, device=device)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        for name in ("_device", "_bank", "_net", "_feature_mean", "_feature_std",
                     "_smooth_kernels"):
            state[name] = None
        return state

    # -- training ------------------------------------------------------------

    def fit(self, labels: Labels, get_field: Callable[[int], np.ndarray], *,
            on_progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Train on every painted pixel in *labels*; returns the fit report.

        Classes with no painted pixels are dropped from the head, so a boundary
        class that was never painted costs nothing and predicts nothing.
        """
        import torch
        fields = labels.painted_fields()
        if not fields:
            raise RuntimeError("nothing has been painted yet")
        started = time.perf_counter()
        features, targets, weights = [], [], []
        with accelerator_lock(self.device):
            for done, field in enumerate(fields, start=1):
                indices, class_ids, strokes = labels.at(field)
                image = np.asarray(get_field(field))
                features.append(self._sample(image, indices, Normalisation.from_field(image)))
                targets.append(torch.as_tensor(class_ids.astype(np.int64), device=self.device))
                weights.append(self._stroke_weights(strokes))
                if on_progress is not None:
                    on_progress(done, len(fields))
            sampled = torch.cat(features, 0)
            raw_targets = torch.cat(targets, 0)
            pixel_weights = torch.cat(weights, 0)
            featurise_seconds = time.perf_counter() - started

            present = sorted(int(value) for value in raw_targets.unique().tolist())
            if len(present) < 2:
                raise RuntimeError("paint at least two classes before training "
                                   f"(only class {present} has any pixels)")
            self.classes = present
            row_of = {class_id: row for row, class_id in enumerate(present)}
            target = torch.as_tensor([row_of[int(value)] for value in raw_targets.tolist()],
                                     device=self.device, dtype=torch.long)

            mean = sampled.mean(0, keepdim=True)
            std = sampled.std(0, unbiased=False, keepdim=True)
            std = torch.where(std > 1e-6, std, torch.ones_like(std))
            standardised = (sampled - mean) / std

            # The head is trained as a Linear layer over the (N, C) sample
            # matrix and predicted as the equivalent 1x1 convolution: the same
            # weights, and a matmul over a few hundred rows is far cheaper on
            # a CPU than a convolution shaped (1, C, 1, N).
            fit_started = time.perf_counter()
            torch.manual_seed(self.seed)
            n_classes = len(present)
            head = torch.nn.Linear(sampled.shape[1], n_classes).to(self.device)
            class_counts = torch.bincount(target, minlength=n_classes).to(sampled.dtype)
            class_weight = sampled.shape[0] / (n_classes * class_counts.clamp_min(1.0))
            sample_weight = pixel_weights * class_weight[target]
            sample_weight = sample_weight / sample_weight.mean()
            optimiser = torch.optim.Adam(head.parameters(), lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
            cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")
            loss = None
            for _ in range(self.epochs):
                optimiser.zero_grad(set_to_none=True)
                loss = (cross_entropy(head(standardised), target) * sample_weight).mean()
                loss.backward()
                optimiser.step()
            with torch.no_grad():
                predicted = head(standardised).argmax(1)
                accuracy = float((predicted == target).float().mean().item())
            fit_seconds = time.perf_counter() - fit_started

        self._weights = {
            "feature_mean": mean.detach().cpu().numpy().reshape(-1),
            "feature_std": std.detach().cpu().numpy().reshape(-1),
            "weight": head.weight.detach().cpu().numpy().reshape(n_classes, -1),
            "bias": head.bias.detach().cpu().numpy().reshape(-1),
        }
        self._net = None                 # rebuilt from the numpy state on first predict
        self.report = {
            "device": str(self.device),
            "n_pixels": int(sampled.shape[0]),
            "n_channels": int(sampled.shape[1]),
            "classes": list(present),
            "pixels_per_class": {int(class_id): int((target == row_of[class_id]).sum().item())
                                 for class_id in present},
            "painted_fields": [int(field) for field in fields],
            "loss": float(loss.item()),
            "train_accuracy": accuracy,
            "featurise_seconds": featurise_seconds,
            "fit_seconds": fit_seconds,
        }
        return dict(self.report)

    def _sample(self, image: np.ndarray, flat_indices: np.ndarray,
                normalisation: Normalisation, pad: int = 32):
        """Feature vectors at *flat_indices*, featurising only their bounding box.

        A scribble occupies a few hundred pixels of a 16.7 M-pixel frame, so
        the box around it is a rounding error against featurising the frame
        (measured: 26 ms for six stroke boxes against 105 ms for a 4096² frame).
        """
        import torch
        height, width = image.shape
        ys, xs = np.divmod(flat_indices.astype(np.int64), width)
        margin = self.halo + pad
        # The crop starts on the decimation grid so its background reference
        # is the full field's, pixel for pixel — otherwise training sees
        # features prediction never produces.
        y0, y1 = align_down(int(ys.min()) - margin), min(height, int(ys.max()) + margin + 1)
        x0, x1 = align_down(int(xs.min()) - margin), min(width, int(xs.max()) + margin + 1)
        crop = _finite(np.ascontiguousarray(image[y0:y1, x0:x1]))
        tensor = torch.as_tensor(crop, device=self.device, dtype=torch.float32)[None, None]
        stack = self.bank(normalisation.apply(tensor))[0]
        rows = torch.as_tensor(ys - y0, device=self.device, dtype=torch.long)
        columns = torch.as_tensor(xs - x0, device=self.device, dtype=torch.long)
        return stack[:, rows, columns].t().contiguous()

    def _stroke_weights(self, strokes: np.ndarray):
        import torch
        stroke = torch.as_tensor(strokes, device=self.device)
        _unique, inverse, counts = torch.unique(stroke, return_inverse=True,
                                                return_counts=True)
        return 1.0 / counts[inverse].to(torch.float32)

    # -- prediction ----------------------------------------------------------

    def _ensure_net(self) -> None:
        import torch
        if self._net is not None:
            return
        if not self.is_trained:
            raise RuntimeError("the classifier has not been trained yet")
        weight = self._weights["weight"]
        n_classes, n_channels = weight.shape
        net = torch.nn.Conv2d(n_channels, n_classes, 1).to(self.device)
        with torch.no_grad():
            net.weight.copy_(torch.as_tensor(weight, device=self.device).view(n_classes, n_channels, 1, 1))
            net.bias.copy_(torch.as_tensor(self._weights["bias"], device=self.device))
        net.eval()
        self._net = net
        self._feature_mean = torch.as_tensor(self._weights["feature_mean"],
                                             device=self.device).view(1, -1, 1, 1)
        self._feature_std = torch.as_tensor(self._weights["feature_std"],
                                            device=self.device).view(1, -1, 1, 1)
        if self.logit_smooth:
            radius = max(1, int(math.ceil(TRUNCATE * self.logit_smooth)))
            kernel = torch.as_tensor(gaussian_kernel(self.logit_smooth, radius),
                                     device=self.device)
            width = 2 * radius + 1
            self._smooth_kernels = (radius,
                                    kernel.view(1, 1, 1, width).expand(n_classes, 1, 1, width),
                                    kernel.view(1, 1, width, 1).expand(n_classes, 1, width, 1))

    def _smooth(self, logits):
        import torch.nn.functional as functional
        radius, row, column = self._smooth_kernels
        n_classes = logits.shape[1]
        logits = functional.conv2d(logits, row, groups=n_classes, padding=(0, radius))
        return functional.conv2d(logits, column, groups=n_classes, padding=(radius, 0))

    def _logits(self, field: np.ndarray):
        """``(n_classes, H, W)`` logits, band by band, plus the finite-pixel mask."""
        import torch
        self._ensure_net()
        image = np.asarray(field)
        finite = np.isfinite(image) if image.dtype.kind == "f" else np.ones(image.shape, bool)
        image = _finite(image)
        normalisation = Normalisation.from_field(image)
        height, width = image.shape
        halo = self.halo
        step = max(64, self.band_rows)
        out = torch.empty((len(self.classes), height, width), device=self.device,
                          dtype=torch.float32)
        with accelerator_lock(self.device), torch.no_grad():
            source = torch.as_tensor(np.ascontiguousarray(image), device=self.device)
            for y0 in range(0, height, step):
                y1 = min(height, y0 + step)
                band0, band1 = align_down(y0 - halo), min(height, y1 + halo)
                band = source[band0:band1].to(torch.float32)[None, None]
                stack = (self.bank(normalisation.apply(band)) - self._feature_mean) / self._feature_std
                logits = self._net(stack)
                if self._smooth_kernels is not None:
                    logits = self._smooth(logits)
                out[:, y0:y1] = logits[0, :, y0 - band0:y1 - band0]
        return out, finite

    def probabilities(self, field: np.ndarray) -> np.ndarray:
        """``(n_classes, H, W)`` float32 softmax, rows in :attr:`classes` order.
        Non-finite pixels of the field are zero in every class."""
        import torch
        logits, finite = self._logits(field)
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=0).cpu().numpy()
        probabilities[:, ~finite] = 0.0
        return probabilities

    def foreground(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """``(particle_probability, boundary_probability)`` for one field.

        Both are ``(H, W)`` float32 in ``[0, 1]``. The boundary map is ``None``
        when no boundary stroke carries trained weight, which is what routes the
        instance split onto the watershed.
        """
        probabilities = self.probabilities(field)
        particle = probabilities[self.classes.index(PARTICLE)]
        boundary = (probabilities[self.classes.index(BOUNDARY)]
                    if BOUNDARY in self.classes else None)
        return particle, boundary


def _finite(image: np.ndarray) -> np.ndarray:
    """Replace non-finite pixels (a drift-corrected border) by the field minimum."""
    if image.dtype.kind != "f":
        return image
    finite = np.isfinite(image)
    if finite.all():
        return image
    fill = float(image[finite].min()) if finite.any() else 0.0
    return np.where(finite, image, fill)


def _jsonable(value):
    import json
    return json.dumps(value, default=lambda item: item.tolist()
                      if hasattr(item, "tolist") else str(item))

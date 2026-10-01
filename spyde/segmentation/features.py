"""
features.py — the per-pixel feature bank behind the trainable segmentation.

Every pixel of a field is described by a short vector: the normalised
intensity, a few Gaussian blurs and their differences, a high-pass, the
gradient magnitude, the local standard deviation (the noise profile), and a
LARGE local background reference with the contrast against it. A linear head
over these channels (:mod:`spyde.segmentation.classifier`) is enough to
separate scribble classes; the width of the head was measured never to buy
separability, only cost.

Two decisions here were measured on real in-situ data and are load-bearing:

* **The discriminative kernels are small, the background reference is large.**
  Particles in low-dose in-situ movies carry a per-pixel contrast-to-noise of
  ~0.17, so one pixel says almost nothing and "darker than the frame" is not
  "darker than its surroundings" on a field with thickness variation. The
  background reference (σ = 25 px, a 300 px receptive field) is computed on a
  4x-decimated image, so it costs about a sixteenth of a full-resolution blur.
* **One normalisation per field.** The robust centre/scale is computed ONCE
  from the whole field and applied to every band and every training sample.
  A band normalised by its own statistics differs from the same region of the
  full field by 4x the real channel range, which makes a preview a different
  computation from the committed run.

torch is imported lazily so importing this package never pays for CUDA.
"""
from __future__ import annotations

import math

import numpy as np

#: Discriminative scales, in pixels. Small: matched to the particle.
DEFAULT_SIGMAS: tuple[float, ...] = (1.0, 2.0, 4.0)

#: Local background reference, in pixels of the full-resolution field.
DEFAULT_BACKGROUND_SIGMA: float = 25.0

#: Decimation factor for the background reference.
BACKGROUND_DECIMATION: int = 4

#: Kernels are truncated at this many sigma.
TRUNCATE: float = 3.0

#: Robust statistics are estimated from at most this many pixels: a full
#: percentile over 16.7 M pixels is a ~35 ms sort, and 1e5 samples put the
#: median and quartiles well inside their own sampling noise.
STATISTICS_SAMPLE: int = 100_000


def select_device(preferred=None):
    """The torch device the segmentation runs on: CUDA, then MPS, then CPU."""
    import torch
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Normalisation:
    """Robust centre and scale of ONE field, fixed for the life of the field.

    Median and inter-quartile range, sampled. Every band, every crop and the
    training sampler standardise by the same two numbers (see the module
    docstring).
    """

    __slots__ = ("centre", "scale")

    def __init__(self, centre: float, scale: float) -> None:
        self.centre = float(centre)
        self.scale = float(scale) if abs(scale) > 1e-6 else 1.0

    @classmethod
    def from_field(cls, field) -> "Normalisation":
        flat = np.asarray(field).reshape(-1)
        if flat.dtype.kind == "f":
            flat = flat[np.isfinite(flat)]
        if not flat.size:
            return cls(0.0, 1.0)
        step = max(1, flat.size // STATISTICS_SAMPLE)
        sample = flat[::step].astype(np.float32)
        lower, median, upper = np.percentile(sample, (25, 50, 75))
        return cls(median, upper - lower)

    def apply(self, tensor):
        return (tensor - self.centre) / self.scale


def align_down(value: int, multiple: int = BACKGROUND_DECIMATION) -> int:
    """*value* rounded down onto the decimation grid (never below 0)."""
    return max(0, int(value) - int(value) % int(multiple))


def gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).astype(np.float32)


def gaussian_derivative_kernel(sigma: float, radius: int) -> np.ndarray:
    """First derivative of a Gaussian: a smoothed gradient in one pass."""
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    gaussian = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    derivative = -(x / (sigma ** 2)) * gaussian
    derivative -= derivative.mean()
    return (derivative / (np.abs(derivative).sum() / 2.0)).astype(np.float32)


class FeatureBank:
    """Separable Gaussian bank plus a decimated background reference.

    Channels, in this order (:meth:`channel_names` is the definition)::

        intensity              the normalised field
        gaussian_<sigma>       one per sigma
        difference_<a>_<b>     differences of adjacent blurs
        highpass               field - gaussian_<smallest sigma>
        gradient               |grad(gaussian_<smallest sigma>)|
        local_std              sqrt(G * I^2 - (G * I)^2), the noise profile
        background             the decimated large-scale reference
        contrast               background - gaussian_<smallest sigma>
        contrast_over_noise    contrast / local_std

    The blurs are produced as K planes in two multi-output convolutions (a row
    pass then a depthwise column pass) rather than K one-channel blurs: cuDNN
    is poor at one-channel work, and one separable blur alone measured 9 ms at
    4096² against a ~1 ms bandwidth floor.
    """

    def __init__(self, sigmas=DEFAULT_SIGMAS, *,
                 background_sigma: float = DEFAULT_BACKGROUND_SIGMA,
                 device=None) -> None:
        import torch
        self.sigmas = tuple(float(sigma) for sigma in sigmas)
        self.background_sigma = float(background_sigma)
        self.device = device if hasattr(device, "type") else select_device(device)

        self.radius = max(1, int(math.ceil(TRUNCATE * max(self.sigmas))))
        rows = [gaussian_kernel(sigma, self.radius) for sigma in self.sigmas]
        rows.append(gaussian_derivative_kernel(min(self.sigmas), self.radius))
        stacked = torch.as_tensor(np.stack(rows), device=self.device)
        width = 2 * self.radius + 1
        self.n_kernels = len(rows)
        self.row_kernels = stacked.view(self.n_kernels, 1, 1, width).contiguous()
        self.column_kernels = stacked.view(self.n_kernels, 1, width, 1).contiguous()

        self.background_radius = 0
        if self.background_sigma:
            decimated_sigma = self.background_sigma / BACKGROUND_DECIMATION
            self.background_radius = max(1, int(math.ceil(TRUNCATE * decimated_sigma)))
            kernel = torch.as_tensor(
                gaussian_kernel(decimated_sigma, self.background_radius),
                device=self.device)
            width = 2 * self.background_radius + 1
            self.background_row = kernel.view(1, 1, 1, width).contiguous()
            self.background_column = kernel.view(1, 1, width, 1).contiguous()

    @property
    def halo(self) -> int:
        """Rows of context a band needs on each side to equal the unbanded result."""
        halo = 2 * self.radius
        if self.background_sigma:
            halo = max(halo, 2 * BACKGROUND_DECIMATION * (self.background_radius + 1))
        return int(halo)

    def channel_names(self) -> list[str]:
        def tag(sigma: float) -> str:
            return f"{sigma:g}".replace(".", "p")

        names = ["intensity"]
        names += [f"gaussian_{tag(sigma)}" for sigma in self.sigmas]
        names += [f"difference_{tag(a)}_{tag(b)}"
                  for a, b in zip(self.sigmas, self.sigmas[1:])]
        names += ["highpass", "gradient", "local_std"]
        if self.background_sigma:
            names += ["background", "contrast", "contrast_over_noise"]
        return names

    @property
    def n_channels(self) -> int:
        return len(self.channel_names())

    def _background(self, image):
        """Decimate, blur, resample: a large reference at a small price.

        The image is padded to a whole number of decimation cells and
        resampled by the exact factor, so a cell always covers the same rows.
        A band or a crop therefore reproduces the full-field value as long as
        it starts on the decimation grid — see :func:`align_down`.
        """
        import torch.nn.functional as functional
        decimation = BACKGROUND_DECIMATION
        height, width = image.shape[-2:]
        pad_rows, pad_columns = (-height) % decimation, (-width) % decimation
        padded = image
        if pad_rows or pad_columns:
            padded = functional.pad(image, (0, pad_columns, 0, pad_rows), mode="replicate")
        small = functional.avg_pool2d(padded, decimation, decimation)
        radius = self.background_radius
        small = functional.conv2d(small, self.background_row, padding=(0, radius))
        small = functional.conv2d(small, self.background_column, padding=(radius, 0))
        full = functional.interpolate(small, scale_factor=decimation, mode="bilinear",
                                      align_corners=False)
        return full[..., :height, :width]

    def __call__(self, image):
        """``image``: a normalised ``(1, 1, H, W)`` float32 tensor → ``(1, C, H, W)``."""
        import torch
        import torch.nn.functional as functional
        radius, n_sigma = self.radius, len(self.sigmas)

        rows = functional.conv2d(image, self.row_kernels, padding=(0, radius))
        planes = functional.conv2d(rows, self.column_kernels, groups=self.n_kernels,
                                   padding=(radius, 0))
        squared_rows = functional.conv2d(image * image, self.row_kernels[:1],
                                         padding=(0, radius))
        squared_blur = functional.conv2d(squared_rows, self.column_kernels[:1],
                                         padding=(radius, 0))

        blurs = planes[:, :n_sigma]
        gradient_x = planes[:, n_sigma:n_sigma + 1]
        gradient_y = functional.conv2d(rows[:, :1], self.column_kernels[n_sigma:n_sigma + 1],
                                       padding=(radius, 0))

        channels = [image, blurs]
        if n_sigma > 1:
            channels.append(blurs[:, :-1] - blurs[:, 1:])
        channels.append(image - blurs[:, :1])
        channels.append(torch.sqrt(gradient_x * gradient_x + gradient_y * gradient_y + 1e-12))
        variance = (squared_blur - blurs[:, :1] * blurs[:, :1]).clamp_min(0.0)
        local_std = torch.sqrt(variance + 1e-12)
        channels.append(local_std)
        if self.background_sigma:
            background = self._background(image)
            contrast = background - blurs[:, :1]
            channels += [background, contrast, contrast / (local_std + 1e-3)]
        return torch.cat(channels, dim=1)

"""GPU tile backend for anyplotlib's large-image tile mode.

anyplotlib's tile display (overview base + on-zoom detail tile) asks a ``TileBackend``
to ``sample(region → out_w×out_h)``. The default ``NumpyTileBackend`` area-means on the
CPU: reading all 16 M pixels of a 4096² frame once is memory-bandwidth-bound at ~62 ms,
which is the whole cost of a movie-scrub frame (the overview rebuild). This backend does
the reduction on the **GPU** instead: the frame is uploaded ONCE (a persistent CUDA
tensor, ~4 ms), then every ``sample`` runs a torch box-mean/adaptive-pool on-device and
copies back only the small (~1 MP) result — the 16 MP frame never crosses the wire per
sample.

Injected by SpyDE via ``Plot2D.enable_tile(backend)`` so anyplotlib keeps NO torch
dependency (the ``TileBackend`` Protocol is exactly this seam). Guarded by
``torch.cuda.is_available()`` with a numpy fallback, so it's correct on a CPU-only /
no-GPU box and lights up on a real GPU.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Reuse anyplotlib's vetted CPU reduction for the fallback / non-GPU path — one source
# of truth for the "mean|subsample|max" block semantics + the ragged-edge handling.
from anyplotlib.plot2d._tile_backend import _box_reduce, _nearest_resize


def _torch_cuda():
    """Return the torch module iff the BACKGROUND prewarm already finished with a
    usable CUDA device, else None — NEVER triggers the import itself. torch's
    import is ~3 s idle and tens of seconds while the disk is saturated (e.g. the
    navigator fill of a fresh 16 GB movie); paying it here — on the painter
    thread, at the first large-frame paint — kept the signal panel black for the
    whole stall. The first frames take the numpy fallback (~62 ms overview) and
    ``set_array`` re-checks per frame, so the display upgrades to the GPU as soon
    as the prewarm lands."""
    try:
        from spyde.backend.heavy_imports import torch_cuda_ready
        if torch_cuda_ready():        # non-blocking; kicks the prewarm if needed
            import torch              # already imported by the prewarm — instant
            return torch
    except Exception as e:  # torch missing / broken CUDA
        logger.debug("gpu tile backend: torch/cuda unavailable (%s)", e)
    return None


class GpuTileBackend:
    """A :class:`anyplotlib.plot2d._tile_backend.TileBackend` that reduces on the GPU.

    Owns the source frame; ``set_array`` swaps it and uploads to the GPU once.
    ``sample`` box-means (or subsamples/maxes) the requested region on-device and
    returns a small 2-D numpy array. Falls back to the numpy ``_box_reduce`` when no
    CUDA device is present.
    """

    def __init__(self, array: np.ndarray, extent=None, origin: str = "upper") -> None:
        a = np.asarray(array)
        if a.ndim != 2:
            raise ValueError(f"GpuTileBackend needs a 2-D array, got {a.shape}")
        self._a = a
        self._extent = tuple(float(v) for v in extent) if extent is not None else None
        self._origin = origin
        self._torch = _torch_cuda()
        self._dev = self._torch.device("cuda") if self._torch is not None else None
        self._gt = None                 # persistent GPU tensor of the current frame
        self._upload()

    # ── source management ──────────────────────────────────────────────────────
    def _upload(self) -> None:
        """Upload the current frame to the GPU as a float32 tensor (once per frame).
        float32 so the on-device mean is a plain average with no integer-overflow
        bookkeeping; a 4096² float32 frame is 64 MB — fine on any modern GPU, and the
        transfer (~4 ms) is paid ONCE per frame, not per sample."""
        if self._torch is None:
            self._gt = None
            return
        try:
            t = self._torch.from_numpy(np.ascontiguousarray(self._a))
            self._gt = t.to(self._dev, dtype=self._torch.float32, non_blocking=True)
        except Exception as e:
            logger.debug("gpu tile upload failed, CPU fallback: %s", e)
            self._torch = None
            self._gt = None

    def set_array(self, array: np.ndarray) -> None:
        a = np.asarray(array)
        if a.ndim != 2:
            raise ValueError(f"GpuTileBackend needs a 2-D array, got {a.shape}")
        self._a = a
        if self._torch is None:
            # The backend may have been built before the background torch/CUDA
            # prewarm finished (first frames ran on the numpy fallback) —
            # upgrade to the GPU as soon as it's ready. Cheap once ready.
            self._torch = _torch_cuda()
            self._dev = (self._torch.device("cuda")
                         if self._torch is not None else None)
            if self._torch is not None:
                logger.info("[TILEDBG] GPU tile backend upgraded to CUDA "
                            "(prewarm landed)")
        self._upload()

    # ── TileBackend protocol ───────────────────────────────────────────────────
    @property
    def full_shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def origin(self):
        return self._origin

    def extent(self):
        return self._extent

    def sample(self, x0, x1, y0, y1, out_w, out_h, method="mean"):
        h, w = self._a.shape
        x0 = int(max(0, min(w, x0))); x1 = int(max(x0 + 1, min(w, x1)))
        y0 = int(max(0, min(h, y0))); y1 = int(max(y0 + 1, min(h, y1)))
        out_w = int(max(1, out_w)); out_h = int(max(1, out_h))

        if self._torch is None or self._gt is None:
            # No GPU → the exact numpy path (byte-identical to NumpyTileBackend).
            region = self._a[y0:y1, x0:x1]
            if method == "subsample":
                sy = max(1, (y1 - y0) // out_h); sx = max(1, (x1 - x0) // out_w)
                return _nearest_resize(region[::sy, ::sx], out_h, out_w)
            return _box_reduce(region, out_h, out_w, "max" if method == "max" else "mean")

        try:
            return self._sample_gpu(x0, x1, y0, y1, out_w, out_h, method)
        except Exception as e:
            logger.debug("gpu sample failed, CPU fallback: %s", e)
            region = self._a[y0:y1, x0:x1]
            if method == "subsample":
                sy = max(1, (y1 - y0) // out_h); sx = max(1, (x1 - x0) // out_w)
                return _nearest_resize(region[::sy, ::sx], out_h, out_w)
            return _box_reduce(region, out_h, out_w, "max" if method == "max" else "mean")

    def _sample_gpu(self, x0, x1, y0, y1, out_w, out_h, method):
        """On-device region reduce → small numpy array. All the arithmetic is on the
        resident GPU tensor; only the (out_h, out_w) result crosses back to the CPU."""
        torch = self._torch
        reg = self._gt[y0:y1, x0:x1]                        # view, no copy (on GPU)
        rh, rw = reg.shape

        if method == "subsample":
            sy = max(1, rh // out_h); sx = max(1, rw // out_w)
            small = reg[::sy, ::sx]
            out = _resize_nearest_torch(torch, small, out_h, out_w)
        elif rh >= out_h and rw >= out_w:
            # DOWNSAMPLE: adaptive pooling gives an exact area reduction to (out_h,
            # out_w) in one fused kernel — no reshape, no ragged-block bookkeeping.
            pool = torch.nn.functional.adaptive_max_pool2d if method == "max" \
                else torch.nn.functional.adaptive_avg_pool2d
            out = pool(reg[None, None], (out_h, out_w))[0, 0]
        else:
            # UPSAMPLE (region smaller than the panel — a deep zoom): show native
            # texels, nearest (crisp), matching the numpy _nearest_resize path.
            out = _resize_nearest_torch(torch, reg, out_h, out_w)

        arr = out.detach().to("cpu").numpy()
        # dtype parity with NumpyTileBackend: mean → float32; subsample/max keep the
        # source dtype so the display quantises identically.
        if method == "mean":
            return arr.astype(np.float32, copy=False)
        return arr.astype(self._a.dtype, copy=False)


def _resize_nearest_torch(torch, t, out_h, out_w):
    """Nearest-neighbour resize a 2-D GPU tensor to (out_h, out_w), matching numpy's
    ``_nearest_resize`` index math (so GPU and CPU paths agree pixel-for-pixel)."""
    h, w = t.shape
    if (h, w) == (out_h, out_w):
        return t
    yi = (torch.arange(out_h, device=t.device) * h // max(1, out_h)).clamp(0, h - 1)
    xi = (torch.arange(out_w, device=t.device) * w // max(1, out_w)).clamp(0, w - 1)
    return t.index_select(0, yi).index_select(1, xi)

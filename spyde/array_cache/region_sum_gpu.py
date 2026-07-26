"""Optional torch-CUDA accumulator for an integrating ROI — the same running sum
:class:`~spyde.array_cache.region_sum.RegionIntegrator` keeps on the host, kept in
device memory instead.

The CPU path is already row-band threaded (~46 ms for a 16-frame 4096² ROI, down
from ~500 ms serial). This trades that for ~2 host→device frame transfers plus a
device add: measured 7.4 ms of arithmetic, ~20 ms/step end-to-end once the entering
frame's disk read is counted.

**It is a strict alternative implementation of arithmetic that already has a
correct answer**, so the bar is bit-identical output, not "close enough":
  * accumulate in float32 — the same dtype ``_region_accum_dtype`` picks, and only
    when it picks float32. A float64 accumulator (int32/uint32/float64 sources) is
    declined outright: fp64 is 1/32 rate on the dev box's Pascal card, so it would
    be both slower and a second numerical regime to keep in parity.
  * ``torch.round`` is round-half-to-EVEN, the same rule as ``np.rint`` (verified:
    [0.5, 1.5, 2.5, -0.5] → [0, 2, 2, -0] on both).
  * the cast back to the source dtype happens ON DEVICE, so only the final frame
    crosses back — and it crosses back already in the source dtype, not as a
    float32 that the host would have to convert.
Element-wise float32 add is IEEE-754 on both sides and the per-pixel summation
order is identical, so the results agree exactly; ``test_region_integrator_gpu.py``
asserts that with ``array_equal`` rather than trusting the argument.

VRAM stays bounded: ONE device accumulator (64 MB for a 4096² float32) plus the
transient frame being uploaded. Frames are NOT kept resident — a device-side ring
of a 16-frame 4k window would be 512 MB competing with find-vectors/orientation
work for the same card, for a saving of one ~5 ms upload per drag step.

Availability goes through ``heavy_imports.torch_cuda_ready()`` (non-blocking, never
triggers the ~3 s torch import itself) exactly like ``GpuTileBackend`` — the first
frames run on the CPU path and it upgrades once the prewarm lands.
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

# Below this the transfers dominate the arithmetic they replace — a 32 KB
# diffraction pattern is far quicker to sum on the host than to ship to the card.
GPU_MIN_FRAME_BYTES = 4 << 20


def gpu_region_enabled() -> bool:
    """``SPYDE_GPU_REGION``: 1/true to enable, 0/false to disable. Defaults to
    DISABLED — the threaded CPU path already puts a drag step inside the paint
    budget, so this is opt-in until it is proven on more hardware than one Pascal
    card."""
    return os.environ.get("SPYDE_GPU_REGION", "0").lower() in ("1", "true", "yes")


def _torch_cuda():
    """The torch module iff a background prewarm already finished with a usable
    CUDA device, else None. Never triggers the import (see GpuTileBackend)."""
    try:
        from spyde.backend.heavy_imports import torch_cuda_ready
        if torch_cuda_ready():
            import torch
            return torch
    except Exception as e:
        log.debug("gpu region accumulate unavailable: %s", e)
    return None


def make_gpu_accumulator(frame_bytes: int, source_dtype, acc_dtype):
    """A :class:`GpuRegionAccumulator` for this read, or None to use the CPU path.

    Declines (in this order) when: disabled by env, the frame is too small to be
    worth a transfer, the accumulator would have to be float64, the source dtype
    has no torch equivalent, or no CUDA device is ready."""
    if not gpu_region_enabled() or int(frame_bytes) < GPU_MIN_FRAME_BYTES:
        return None
    if np.dtype(acc_dtype) != np.float32:
        return None
    torch = _torch_cuda()
    if torch is None:
        return None
    if _torch_dtype(torch, source_dtype) is None:
        return None
    try:
        return GpuRegionAccumulator(torch, source_dtype)
    except Exception as e:
        log.debug("gpu region accumulator construction failed: %s", e)
        return None


def _torch_dtype(torch, np_dtype):
    """The torch dtype matching a numpy one, or None. ``uint16``/``uint32`` only
    exist on newer torch; returning None there just routes the read to the CPU."""
    return getattr(torch, np.dtype(np_dtype).name, None)


class GpuRegionAccumulator:
    """Device-resident running sum with the same two operations the host
    accumulator has. Raises on any device error — the caller invalidates and
    falls back to the CPU path, which is always correct."""

    def __init__(self, torch, source_dtype) -> None:
        self._torch = torch
        self._dev = torch.device("cuda")
        self._source_dtype = np.dtype(source_dtype)
        self._out_dtype = _torch_dtype(torch, source_dtype)
        self._integer = np.issubdtype(self._source_dtype, np.integer)
        self._sum = None                       # device float32 tensor

    # ── state ─────────────────────────────────────────────────────────────────
    @property
    def has_sum(self) -> bool:
        return self._sum is not None

    @property
    def shape(self):
        return None if self._sum is None else tuple(self._sum.shape)

    def invalidate(self) -> None:
        self._sum = None

    # ── operations ────────────────────────────────────────────────────────────
    def recompute(self, fetch, points, n):
        acc = None
        for pt in points:
            t = self._upload(fetch(pt))        # always a fresh device tensor
            if acc is None:
                acc = t
            else:
                acc.add_(t)
        # Published only once the whole window is in — a half-built sum must never
        # become the running sum a later delta is applied to.
        self._sum = acc
        return self._finalize(n)

    def apply_delta(self, fetch, entering, leaving, n):
        """Leaving frames subtracted BEFORE entering ones are added — the same
        ordering rule the host path uses to keep every partial sum inside
        float32's exact-integer range."""
        acc = self._sum
        for pt in leaving:
            acc.sub_(self._upload(fetch(pt)))
        for pt in entering:
            acc.add_(self._upload(fetch(pt)))
        return self._finalize(n)

    # ── internals ─────────────────────────────────────────────────────────────
    def _upload(self, frame):
        """Host frame → device float32. ``from_numpy`` shares memory, and cached
        frames are contractually read-only, so this only ever reads from it; the
        copy happens in ``.to(device)``. A non-writable buffer (the pread reader's
        zero-copy frame) is copied first — torch refuses to wrap one."""
        arr = np.asarray(frame)
        if not arr.flags.writeable:
            arr = arr.copy()
        return self._torch.from_numpy(arr).to(
            self._dev, dtype=self._torch.float32, non_blocking=True)

    def _finalize(self, n):
        """``round(sum/n)`` cast back to the source dtype ON DEVICE, so the frame
        crosses back once and already in its final dtype."""
        out = self._sum / n
        if self._integer:
            out = self._torch.round(out).to(self._out_dtype)
        res = out.detach().to("cpu").numpy()
        if res.dtype != self._source_dtype and self._integer:
            res = res.astype(self._source_dtype)
        return res

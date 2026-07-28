"""A torch-CUDA accumulator for CSB, alongside the vendored cupy one.

``_core``'s GPU path is a CuPy ``RawModule`` kernel. SpyDE's GPU stack is
**torch** — it is what the fitting engine, the vector-orientation fit and the
EBSD indexing all run on, and it is what the installer pins with CUDA — so on a
SpyDE machine ``gpu_available()`` is False for want of a package nobody
installed, and a 125 M-event movie integrates on the CPU at ~0.7 s per plane
when the GPU could do it in milliseconds.

This module adds the missing lane without touching ``_core``/``_sparse``, which
are vendored verbatim from de-csb and want to stay diffable against upstream.

The kernel, in one line
-----------------------
A CSB event word IS its intra-block raster index, and block ``b``'s events
belong in ``acc[b*tile_size : (b+1)*tile_size]``. So the whole accumulate is a
scatter-add of ones at ``block_id * tile_size + word`` — no custom kernel
needed, just ``index_add_``, which is exactly the shape CLAUDE.md's GPU notes
say to reach for instead of a per-item loop.

Memory is the one thing to be careful about. ``torch.bincount`` over the full
accumulator would allocate a fresh int64 the size of the frame (536 MB at
8192²) on every call; ``index_add_`` into a persistent int32 accumulator
allocates only ``ones`` at the EVENT count instead (~10 MB for one plane).
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def torch_gpu_available() -> bool:
    """True when torch can see a CUDA device."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception as e:
        log.debug("torch CUDA unavailable: %s", e)
        return False


class TorchAccumulator:
    """Duck-compatible with :class:`.._core.CSBAccumulator` for the summation
    path (``reset`` / ``add_frames`` / ``synchronize`` / ``image``)."""

    def __init__(self, csb, device=None, dtype=None):
        import torch

        self.csb = csb
        self.device = torch.device(device or "cuda")
        self.dtype = dtype or torch.int32
        n = int(csb.blocks_per_frame) * int(csb.tile_size)
        self._acc = torch.zeros(n, dtype=self.dtype, device=self.device)
        # Block index within a frame for each (frame, block) segment, in the
        # payload's readout order. Uploaded once and reused every call.
        self._seg_block = torch.arange(int(csb.blocks_per_frame),
                                       dtype=torch.int64, device=self.device)
        self._counts = torch.as_tensor(np.asarray(csb.counts, np.int64),
                                       device=self.device)

    # -- accumulate --------------------------------------------------------
    def reset(self) -> None:
        self._acc.zero_()

    def add_frames(self, f0: int, f1: int, payload=None, stream=None) -> None:
        """Histogram the events of frames ``[f0, f1)`` into the accumulator."""
        import torch

        cs = self.csb
        bpf = int(cs.blocks_per_frame)
        n_seg = (int(f1) - int(f0)) * bpf
        if n_seg <= 0:
            return
        seg0 = int(f0) * bpf
        counts = self._counts[seg0:seg0 + n_seg]
        n_ev = int(counts.sum().item())
        if n_ev == 0:
            return

        # The payload words for this window are contiguous: `starts` are
        # absolute file offsets and the frames are consecutive. `cs.events` is
        # the memory-mapped <u2 word stream.
        w0 = int(cs.starts[seg0])
        # int32 across PCIe, widened to int64 on the device. index_add_ needs
        # int64 indices, but sending them as int64 doubles the transfer — and
        # this path is transfer-bound, not arithmetic-bound (the whole-movie
        # sum is SLOWER on the GPU than on numba for exactly that reason).
        words_np = np.asarray(cs.events[w0:w0 + n_ev]).astype(np.int32)
        words = torch.from_numpy(words_np).to(self.device, non_blocking=True)

        # Each event's block, then its slot in the tiled accumulator.
        block_of_seg = self._seg_block.repeat(int(f1) - int(f0))
        block_id = torch.repeat_interleave(block_of_seg, counts)
        idx = block_id * int(cs.tile_size) + words.long()
        self._acc.index_add_(
            0, idx, torch.ones(n_ev, dtype=self.dtype, device=self.device))

    def synchronize(self, stream=None) -> None:
        import torch
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # -- read out ----------------------------------------------------------
    def _untile(self):
        """Block-major accumulator -> padded 2-D image, still on device.

        Mirrors ``_core.CSBAccumulator._untile`` exactly; the two must agree or
        the same file reads differently depending on which backend answered.
        """
        cs = self.csb
        bh, bw = cs.csb_block_height, cs.csb_block_width
        if cs.csb_order == 1:                                   # row-major
            v = self._acc.reshape(cs.blocks_per_height, cs.blocks_per_width, bh, bw)
            v = v.permute(0, 2, 1, 3)
        else:                                                   # column-major
            v = self._acc.reshape(cs.blocks_per_width, cs.blocks_per_height, bh, bw)
            v = v.permute(1, 2, 0, 3)
        return v.reshape(cs.padded_height, cs.padded_width)

    def image(self, dtype=None) -> np.ndarray:
        cs = self.csb
        v = self._untile()[:cs.frame_height, :cs.frame_width]
        out = v.contiguous().cpu().numpy()
        return out if dtype is None else out.astype(dtype)

    def preview(self, bin_factor: int = 4, dtype=np.float32) -> np.ndarray:
        if bin_factor < 1:
            raise ValueError("bin_factor must be >= 1")
        cs = self.csb
        h = (cs.frame_height // bin_factor) * bin_factor
        w = (cs.frame_width // bin_factor) * bin_factor
        v = self._untile()[:h, :w].contiguous()
        v = v.reshape(h // bin_factor, bin_factor, w // bin_factor, bin_factor)
        return v.sum(dim=(1, 3), dtype=__import__("torch").int64).cpu().numpy(
        ).astype(dtype)

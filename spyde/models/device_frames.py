"""Move a chunk's frames to the device into buffers allocated once per worker.

A Find Vectors batch sends one storage chunk after another, all the same shape,
through the neural detector. Each chunk used to be cast to float32 on the host
and then copied into a freshly allocated device tensor: 177 ms for 1024 frames
of 256 x 256 uint16, against 25 ms here. Instead each concurrent worker keeps:

- a device buffer in the frames' own dtype, which the upload lands in, so an
  integer frame crosses the bus at its own size rather than as float32;
- a device float32 buffer, written by one cast on the device: the frames the
  detector and the centre steps read.

The buffers grow when a chunk holds more frames than any before and are reused
otherwise. ``release_idle`` (the post-batch memory trim) drops them.

The upload reads the chunk where it lies, in ordinary pageable memory. Copying
it into a page-locked buffer first makes the transfer itself faster, but the
extra host copy costs more than that saves (1024 x 256 x 256 uint16: 33 ms
through a page-locked buffer against 25 ms direct).

On Apple-MPS host and device share memory and integer casts there are less
proven, so the frames are cast on the host and copied into the float32 buffer.
"""
from __future__ import annotations

import contextlib
import threading

import numpy as np
import torch

# Source dtypes cast to float32 on a CUDA device; anything else on the host.
# Every one of these casts exactly as numpy's ``astype(np.float32)`` does.
_DEVICE_CAST_DTYPES = {
    np.dtype(np.uint8), np.dtype(np.int8), np.dtype(np.uint16), np.dtype(np.int16),
    np.dtype(np.uint32), np.dtype(np.int32), np.dtype(np.int64),
    np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64),
}


def _fitting(buffer, shape, dtype, device):
    """``buffer`` when it can hold ``shape`` (frame count first) of ``dtype`` on
    ``device``, else a new buffer of exactly that size."""
    if (buffer is not None and buffer.dtype == dtype and buffer.device.type == device.type
            and tuple(buffer.shape[1:]) == tuple(shape[1:]) and buffer.shape[0] >= shape[0]):
        return buffer
    return torch.empty(tuple(shape), dtype=dtype, device=device)


class FrameBuffers:
    """One worker's upload buffers; see the module docstring."""

    def __init__(self):
        self._raw = None
        self._float = None

    def upload(self, frames: np.ndarray, device) -> torch.Tensor:
        """``frames`` ``(N, H, W)`` as float32 on ``device``: a view of this
        worker's buffer, valid until its next ``upload``."""
        frames = np.asarray(frames)
        count = frames.shape[0]
        self._float = _fitting(self._float, frames.shape, torch.float32, device)
        target = self._float[:count]
        if device.type != "cuda" or frames.dtype not in _DEVICE_CAST_DTYPES:
            target.copy_(torch.as_tensor(np.asarray(frames, np.float32)))
        elif frames.dtype == np.float32:
            target.copy_(torch.from_numpy(frames))
        else:
            source = torch.from_numpy(frames)
            self._raw = _fitting(self._raw, frames.shape, source.dtype, device)
            raw = self._raw[:count]
            raw.copy_(source)
            target.copy_(raw)
        return target


_idle: list[FrameBuffers] = []
_idle_lock = threading.Lock()


@contextlib.contextmanager
def leased_frame_buffers():
    """One worker's ``FrameBuffers`` for the duration of the block. They go
    back for the next chunk, so there are only ever as many as there were
    chunks on the device at once."""
    with _idle_lock:
        buffers = _idle.pop() if _idle else FrameBuffers()
    try:
        yield buffers
    finally:
        with _idle_lock:
            _idle.append(buffers)


def release_idle() -> None:
    """Drop every buffer not in use, so the memory trim can hand it back."""
    with _idle_lock:
        _idle.clear()

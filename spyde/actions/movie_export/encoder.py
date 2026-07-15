"""
encoder.py — the movie-export WRITER SEAM.

WHY THIS SEAM EXISTS (the plan's locked decision)
--------------------------------------------------
The whole frame pipeline (lazy slice → LUT → PIL annotations → trace inset) is
*encoder-agnostic*: it produces plain ``(H, W, 3)`` uint8 RGB frames and hands
them to a minimal writer interface — ``open_writer(path, fps, size)`` returning
an object with ``.append(rgb_uint8)`` and ``.close()``. Isolating the encoder
behind this one function means the codec is a **one-module swap**.

v1 chooses **imageio-ffmpeg** deliberately:

* it is ALREADY a spyde dependency (zero new weight),
* its wheel bundles a static ffmpeg binary — no system install, no CLI on PATH,
* H.264 mp4 is what pastes into PowerPoint / Slack / Word.

``matplotlib.animation`` was rejected (it shells out to ffmpeg anyway, behind a
slow per-frame figure render). If ffmpeg ever has to go (PyAV, renderer-side
WebCodecs, …) only this module changes — the pipeline, LUT, annotations, and
trace compositing stay put.

A ``.gif`` path routes to a Pillow-based GIF writer through the SAME seam (small
clips only; palette-quantised, duration derived from fps).
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# H.264 quality (imageio scale 0..10, higher = better). 7 is a good
# size/quality balance for scientific movies.
_H264_QUALITY = 7


class Writer(Protocol):
    """Minimal video-writer seam: append RGB uint8 frames, then close."""

    def append(self, rgb: np.ndarray) -> None: ...

    def close(self) -> None: ...


class _ImageioWriter:
    """H.264 mp4 (or any imageio-ffmpeg codec) writer.

    ``macro_block_size=1`` disables imageio's automatic pad-to-16 (we already
    crop to even dimensions in the pipeline, so the size the caller passes is the
    size that lands in the file — no silent letterboxing)."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        import imageio
        # size is (W, H); imageio infers it from the first appended frame, but we
        # keep it for the GIF path and for validation.
        self._w, self._h = int(size[0]), int(size[1])
        self._writer = imageio.get_writer(
            path, fps=float(fps), codec="libx264",
            quality=_H264_QUALITY, macro_block_size=1,
            pixelformat="yuv420p", ffmpeg_log_level="error",
        )

    def append(self, rgb: np.ndarray) -> None:
        self._writer.append_data(np.ascontiguousarray(rgb, dtype=np.uint8))

    def close(self) -> None:
        try:
            self._writer.close()
        except Exception as e:
            log.debug("closing imageio writer failed: %s", e)


class _GifWriter:
    """Pillow-based GIF writer (small clips). Accumulates frames and writes the
    animated GIF on ``close`` (Pillow needs all frames to build one palette /
    the append-images list). Duration per frame = 1000/fps ms."""

    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        self._path = path
        self._duration_ms = max(1, int(round(1000.0 / max(1e-6, float(fps)))))
        self._frames: list = []

    def append(self, rgb: np.ndarray) -> None:
        from PIL import Image
        self._frames.append(Image.fromarray(
            np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB"))

    def close(self) -> None:
        if not self._frames:
            return
        # Adaptive palette per frame keeps colormaps looking right; loop=0 = ∞.
        first, rest = self._frames[0], self._frames[1:]
        first.save(self._path, save_all=True, append_images=rest,
                   duration=self._duration_ms, loop=0, optimize=False)
        self._frames = []


def open_writer(path: str, fps: float, size: tuple[int, int]) -> Writer:
    """Open a video writer for *path* at *fps* and pixel *size* ``(W, H)``.

    ``.gif`` → the Pillow GIF writer; anything else → imageio-ffmpeg H.264. The
    returned object satisfies the :class:`Writer` protocol (``append`` / ``close``).
    """
    if str(path).lower().endswith(".gif"):
        return _GifWriter(path, fps, size)
    return _ImageioWriter(path, fps, size)

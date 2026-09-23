"""Synthetic CSB files and the image their events must integrate to (#91).

The format documents itself (`_core.py` module docstring): a 108-byte header,
then ONE uint16 per event giving its raster position inside its own block,
then a footer of one uint16 event-count per block. So a file carrying real
events is a header plus a payload, and because the mapping

    y = block_y_origin + word // block_w
    x = block_x_origin + word %  block_w

is two lines of arithmetic, the expected image is computable in plain numpy.
Tests compare the reader against `expected_image`, derived INDEPENDENTLY of
it, never against a blob recorded from it.
"""
from __future__ import annotations

import math
import struct

import numpy as np

from spyde.external.rsciio_csb._core import CSB_MAGIC

W, H, BW, BH, NF = 64, 48, 16, 16, 4       # 4x3 blocks, divides evenly
OW, OH = 70, 50                             # ...and a size that does NOT


def grid(width, height, block_w, block_h):
    """Blocks per width and per height; an edge block may be partial."""
    return math.ceil(width / block_w), math.ceil(height / block_h)


def block_origin(b, width, height, block_w, block_h, order):
    """(y, x) of block *b*.

    Deliberately a second implementation of CSBFile.block_origin rather than a
    call to it — the point is to check the reader against the documented rule,
    and a shared helper would agree with itself no matter what either did.
    """
    blocks_per_width, blocks_per_height = grid(width, height, block_w, block_h)
    if order == 1:                                  # row-major
        by, bx = divmod(b, blocks_per_width)
    else:                                           # column-major (default)
        bx, by = divmod(b, blocks_per_height)
    return by * block_h, bx * block_w


def csb_bytes(events=None, *, width=W, height=H, frames=NF, block_w=BW,
              block_h=BH, order=0, us_per_frame=390.0, magic=CSB_MAGIC,
              camera_sn=4242, kv=200):
    """Bytes of a CSB carrying *events* (none by default).

    ``events`` maps ``(frame, block_index)`` to a list of intra-block words.
    The payload and the count footer are written in the reader's own order —
    ``frame * blocks_per_frame + block`` — which is what ``frame_slice`` and
    ``frame_events`` index by.

    A degenerate dimension is written as given (the reader must reject it),
    with an empty block table.
    """
    events = events or {}
    degenerate = min(width, height, frames, block_w, block_h) < 1
    blocks_per_frame = 0 if degenerate else math.prod(
        grid(width, height, block_w, block_h))

    words: list[int] = []
    counts: list[int] = []
    for f in range(0 if degenerate else frames):
        for b in range(blocks_per_frame):
            block_words = list(events.get((f, b), ()))
            words.extend(block_words)
            counts.append(len(block_words))

    hdr = bytearray(108)
    struct.pack_into("<H", hdr, 0, magic)
    struct.pack_into("<H", hdr, 2, 1)                        # file_version
    struct.pack_into("<H", hdr, 4, width)
    struct.pack_into("<H", hdr, 6, height)
    struct.pack_into("<I", hdr, 8, frames)
    struct.pack_into("<f", hdr, 12, 0.025)                   # ang_per_pix
    struct.pack_into("<f", hdr, 16, us_per_frame)
    struct.pack_into("<H", hdr, 20, block_w)
    struct.pack_into("<H", hdr, 22, block_h)
    struct.pack_into("<Q", hdr, 24, 108)                     # data offset
    struct.pack_into("<Q", hdr, 32, 108 + len(words) * 2)    # lengths offset
    struct.pack_into("<H", hdr, 40, order)
    struct.pack_into("<H", hdr, 42, camera_sn)
    struct.pack_into("<H", hdr, 62, kv)

    raw = (bytes(hdr)
           + np.asarray(words, "<u2").tobytes()
           + np.asarray(counts, "<u2").tobytes())
    # A degenerate dimension leaves an empty table and a 108-byte file, which
    # trips the length guard BEFORE the dimension validation a test wants to
    # reach. Pad past it so those cases fail for the reason under test.
    return raw + b"\x00" * max(0, 110 - len(raw))


def expected_image(events, f0, f1, *, width=W, height=H, block_w=BW,
                   block_h=BH, order=0):
    """What frames [f0, f1) must integrate to.

    Events outside the real frame are DROPPED: edge blocks keep the full
    stride, so the accumulator is padded and cropped at readout.
    """
    img = np.zeros((height, width), np.int64)
    for (f, b), block_words in events.items():
        if not (f0 <= f < f1):
            continue
        oy, ox = block_origin(b, width, height, block_w, block_h, order)
        for w in block_words:
            y, x = oy + w // block_w, ox + w % block_w
            if 0 <= y < height and 0 <= x < width:
                img[y, x] += 1
    return img


def write_csb(tmp_path, events, name="ev.csb", **kw):
    """Write `csb_bytes(events, **kw)` under *tmp_path*; returns the path."""
    p = tmp_path / name
    p.write_bytes(csb_bytes(events, **kw))
    return str(p)

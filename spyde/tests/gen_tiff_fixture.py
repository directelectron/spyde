"""Write the .tif fixtures the TIFF-load e2e opens (``tiff_load.spec.ts``).

A REAL file on disk is the whole point: the bug under test only exists once
rosettasciio actually opens a TIFF, because that is when the lazy graph closes
over an open ``BufferedReader`` and stops being picklable. A synthetic
``da.from_array`` stand-in serializes fine and would prove nothing.

Usage::

    python -m spyde.tests.gen_tiff_fixture <out-dir>

writes ``single.tif`` (one page — the no-navigation display path) and
``stack.tif`` (eight pages — the progressive navigator-fill path).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import tifffile


def write_fixtures(out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    # One page. Structured, not noise: a flat frame would paint an even grey
    # that a "did anything render?" pixel check cannot tell from a placeholder.
    y, x = np.mgrid[0:256, 0:256]
    single = (1000 + 400 * np.exp(-((y - 128) ** 2 + (x - 128) ** 2) / 2000.0)
              + rng.normal(0, 20, (256, 256))).astype("float32")
    single_p = os.path.join(out_dir, "single.tif")
    tifffile.imwrite(single_p, single)

    # Eight pages, each with its band in a DIFFERENT row, so the per-frame sums
    # differ and a filled navigator is visibly eight distinct levels rather than
    # one flat bar (which is what a fill that never ran also looks like).
    stack = (rng.random((8, 128, 128)) * 500).astype("uint16")
    for i in range(8):
        stack[i, 12 * i:12 * i + 8, :] = 60000
    stack_p = os.path.join(out_dir, "stack.tif")
    tifffile.imwrite(stack_p, stack)
    return single_p, stack_p


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for p in write_fixtures(out):
        print(p)

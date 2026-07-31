"""test_live_fill_poller.py — the progressive-fill poller must not repaint
unchanged data.

``lifecycle.live_fill_poller`` polls a shared-memory buffer that a distributed
compute fills chunk by chunk, and hands each snapshot to a paint callback. It
used to paint on EVERY tick regardless of whether the buffer had changed.

Measured on a real 256x256-nav fill: **~195 paints for ~26 distinct buffer
states**. Two painters run concurrently and neither knows about the other (this
poller on its interval, plus the per-chunk relay), so during the long stretch
before the first chunk lands they were both repainting an ALL-NaN array several
times a second. Each paint is a full `_set_array` + levels + histogram + binary
push, on the same process that has to submit and collect the compute.

The digest is over raw BYTES rather than a value comparison, because
``NaN != NaN``: an `==` check would call every tick a change during exactly the
unfilled phase that hurts most. That case has its own test below.
"""
from __future__ import annotations

import time

import numpy as np

from spyde.actions import lifecycle


class TestLiveFillPollerDedupe:
    """The poller must not repaint a buffer that has not changed.

    Measured on a real 256x256-nav fill: ~195 paints for ~26 distinct buffer
    states. Two painters run concurrently (this poller on its interval, and the
    per-chunk relay) and neither knows about the other, so during the long
    stretch before the first chunk lands they repainted an ALL-NaN array several
    times a second — each one a full `_set_array` + levels + histogram + binary
    push on the thread that has to submit and collect the compute.
    """

    def _run(self, frames, interval=0.01):
        """Drive the poller over a scripted sequence of buffer states."""
        import spyde.drawing.update_functions as uf
        import numpy as _np
        seen = []
        seq = list(frames)
        idx = {"i": 0}

        def fake_read(shape, name):
            i = min(idx["i"], len(seq) - 1)
            idx["i"] += 1
            return _np.asarray(seq[i], dtype=_np.float32)

        real = uf.read_live_buffer
        uf.read_live_buffer = fake_read
        try:
            stop = lifecycle.live_fill_poller(
                (3,), "fake", seen.append, interval=interval)
            deadline = time.monotonic() + 2.0
            while idx["i"] < len(seq) and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.05)
            stop()
        finally:
            uf.read_live_buffer = real
        return seen

    def test_an_unchanged_buffer_is_painted_once(self):
        import numpy as _np
        same = [1.0, 2.0, 3.0]
        seen = self._run([same] * 8)
        assert len(seen) == 1, (
            f"{len(seen)} paints for one buffer state — the poller is "
            "repainting unchanged data")

    def test_an_all_nan_buffer_is_painted_once(self):
        """NaN != NaN under `==`, so a naive comparison would call every tick a
        change — which is exactly the pre-first-chunk case that hurt most."""
        import numpy as _np
        nan3 = [_np.nan, _np.nan, _np.nan]
        seen = self._run([nan3] * 8)
        assert len(seen) == 1, f"{len(seen)} paints of an unchanged NaN buffer"

    def test_real_changes_still_paint(self):
        seen = self._run([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                          [9.0, 2.0, 3.0], [9.0, 2.0, 3.0],
                          [9.0, 9.0, 3.0]])
        assert len(seen) == 3, (
            f"expected 3 paints for 3 distinct states, got {len(seen)}")

"""
Adaptive storage-spanning chunking for in-situ movies (Phase 1).

``Session._signal_spanning_chunks`` must size the nav block by FRAME BYTES so a
single-frame navigator read stays near ~64 MB regardless of frame resolution:

  * a large image frame (in-situ movie) → **1 frame per chunk** (never 32 ×
    256 MB = an 8 GB chunk),
  * a small diffraction pattern → many frames per chunk (capped),
  * an already-well-chunked self-describing dataset → no rebuild (returns None).

See benchmarks.md "In-situ movie playback" for why: the reader chunks 8 frames ×
full 4096² = a 128 MB read per single frame, and the old flat nav_chunk=32 made
it worse.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs

from spyde.backend.session import Session


def _lazy_2d_signal(n_frames, frame, dtype=np.float32, chunks=None):
    """A lazy nav-dim-1 stack of 2-D frames with an explicit (bad) chunking."""
    shape = (n_frames,) + frame
    arr = da.zeros(shape, dtype=dtype, chunks=chunks or (n_frames,) + frame)
    s = hs.signals.Signal2D(arr).as_lazy()
    return s


class TestAdaptiveChunking:
    def test_large_frame_gets_one_frame_per_chunk(self):
        # 4k×4k uint8 = 16 MB/frame → 64 MB target → 4 frames; but if the
        # signal axes are split we must still cap the nav block. Use 8k×8k
        # float32 = 256 MB/frame → exactly 1 frame/chunk.
        # Force a signal-split so the function re-chunks.
        s = _lazy_2d_signal(4, (8192, 8192), dtype=np.float32,
                            chunks=(4, 4096, 8192))  # signal axis 0 split
        ch = Session._signal_spanning_chunks(s)
        assert ch is not None
        # nav block == 1 (256 MB frame > 64 MB target), signal axes whole.
        assert ch == (1, -1, -1)

    def test_medium_frame_caps_at_target(self):
        # 4k×4k uint8 = 16 MB/frame → 64/16 = 4 frames/chunk (signal split so it
        # re-chunks).
        s = _lazy_2d_signal(20, (4096, 4096), dtype=np.uint8,
                            chunks=(20, 2048, 4096))
        ch = Session._signal_spanning_chunks(s)
        assert ch is not None
        assert ch == (4, -1, -1)

    def test_small_dp_packs_many_but_capped(self):
        # 128×128 uint16 = 32 KB/frame → target would be ~2000, capped to 32.
        # Signal axes split so it re-chunks.
        s = _lazy_2d_signal(500, (128, 128), dtype=np.uint16,
                            chunks=(500, 64, 128))
        ch = Session._signal_spanning_chunks(s)
        assert ch is not None
        assert ch == (32, -1, -1)   # _NAV_CHUNK_MAX

    def test_whole_signal_and_small_nav_block_is_left_alone(self):
        # Signal axes already whole AND the reader's nav block (2) is <= our
        # target (4 for a 16 MB frame) → no rebuild.
        s = _lazy_2d_signal(20, (4096, 4096), dtype=np.uint8,
                            chunks=(2, 4096, 4096))
        ch = Session._signal_spanning_chunks(s)
        assert ch is None

    def test_whole_signal_but_oversized_nav_block_is_recut(self):
        # Signal axes whole but the reader packed 32 large frames per chunk
        # (32 × 16 MB = 512 MB) — that's too big for one-frame reads, recut to 4.
        s = _lazy_2d_signal(64, (4096, 4096), dtype=np.uint8,
                            chunks=(32, 4096, 4096))
        ch = Session._signal_spanning_chunks(s)
        assert ch == (4, -1, -1)

    def test_2d_nav_4dstem_still_spans_signal(self):
        # A 4D-STEM scan (2 nav dims) with split signal axes still re-chunks to
        # whole signal frames — the movie change must not regress this.
        arr = da.zeros((10, 12, 128, 128), dtype=np.uint16, chunks=(10, 12, 64, 128))
        s = hs.signals.Signal2D(arr).as_lazy()
        ch = Session._signal_spanning_chunks(s)
        assert ch is not None
        # 32 KB/frame → capped nav block on BOTH nav axes, whole signal.
        assert ch[-2:] == (-1, -1)
        assert all(c == 32 for c in ch[:2]) or all(c <= 32 for c in ch[:2])

    def test_non_navigated_2d_image_returns_none(self):
        arr = da.zeros((512, 512), dtype=np.float32, chunks=(256, 512))
        s = hs.signals.Signal2D(arr).as_lazy()
        assert Session._signal_spanning_chunks(s) is None

    def test_explicit_nav_chunk_override_is_honoured(self):
        s = _lazy_2d_signal(20, (4096, 4096), dtype=np.uint8,
                            chunks=(20, 2048, 4096))
        ch = Session._signal_spanning_chunks(s, nav_chunk=1)
        assert ch == (1, -1, -1)


class TestMovieFixture:
    def test_movie_opens_with_1d_time_navigator(self, movie_dataset):
        session = movie_dataset["window"]
        assert len(session.signal_trees) == 1
        root = session.signal_trees[0].root
        am = root.axes_manager
        assert am.navigation_dimension == 1
        assert am.signal_dimension == 2
        # The time axis kept its calibration (sec, not stamped nm).
        tax = am.navigation_axes[0]
        assert tax.name == "time"
        assert str(tax.units) == "sec"
        # Stayed lazy — no materialise of the movie.
        assert root._lazy is True

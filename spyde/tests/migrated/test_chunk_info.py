"""The dock's chunk viewer payload (``build_chunk_info``).

The viewer is dask's block picture restyled, PLUS the judgement dask cannot make:
whether one chunk holds whole SIGNAL frames. That flag drives the whole display
(the red frame, the warning), and the ruinous case — chunks that split the signal
axes, RosettaSciIO's balanced-cube default on some readers — is exactly what no
bundled fixture produces, so it is pinned here rather than in the e2e.
"""
import dask.array as da
import hyperspy.api as hs
import numpy as np

from spyde.metadata_extract import build_chunk_info


class _Tree:
    """Minimal stand-in exposing what build_chunk_info reads."""
    def __init__(self, root):
        self.root = root
        self.signal_plots = []


def _lazy(shape, chunks):
    data = da.zeros(shape, chunks=chunks, dtype=np.uint16)
    return hs.signals.Signal2D(data).as_lazy()


class TestChunkInfo:
    def test_eager_data_has_no_chunking(self):
        s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), dtype=np.uint16))
        assert build_chunk_info(_Tree(s)) is None

    def test_storage_aligned_chunks_are_not_split(self):
        info = build_chunk_info(_Tree(_lazy((24, 24, 32, 32), (8, 8, 32, 32))))
        assert info["signal_split"] is False
        assert info["nav_ndim"] == 2
        assert info["counts"] == [3, 3, 1, 1]
        assert info["n_chunks"] == 9
        # One block = 8×8 nav × the whole 32×32 frame, uint16.
        assert info["chunk_bytes"] == 8 * 8 * 32 * 32 * 2
        assert info["nbytes"] == 24 * 24 * 32 * 32 * 2

    def test_a_balanced_cube_splits_the_signal(self):
        """The navigator-killer: chunks that cut the signal axes, so showing one
        frame reads several blocks."""
        info = build_chunk_info(_Tree(_lazy((24, 24, 32, 32), (8, 8, 16, 16))))
        assert info["signal_split"] is True
        assert info["counts"] == [3, 3, 2, 2]

    def test_a_one_frame_per_chunk_movie_is_not_split(self):
        """An in-situ movie is 1 frame per chunk — many blocks, none of them
        splitting a frame."""
        info = build_chunk_info(_Tree(_lazy((6, 64, 64), (1, 64, 64))))
        assert info["signal_split"] is False
        assert info["nav_ndim"] == 1
        assert info["counts"] == [6, 1, 1]

    def test_long_chunk_lists_are_truncated_but_counted(self):
        """A per-frame-chunked movie has one entry per frame. The viewer draws a
        few dozen at most, so the payload carries a bounded sample and the REAL
        count beside it."""
        info = build_chunk_info(_Tree(_lazy((400, 16, 16), (1, 16, 16))))
        assert info["counts"][0] == 400
        assert len(info["chunks"][0]) == 128

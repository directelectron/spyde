"""``nav_chunk_edge`` and the composed navigation chunk built on it (#187)."""
import math

import dask.array as da
import numpy as np
import pytest

from spyde.backend._session_files import nav_chunk_edge
from spyde.backend._session_multiangle import composed_nav_chunk


class TestNavChunkEdge:
    @pytest.mark.parametrize("frame_bytes, target_bytes, n_nav_dims, expected", [
        (256 * 256 * 2, 64 << 20, 1, 512),
        (256 * 256 * 2, 64 << 20, 2, 22),
        (256 * 256 * 2, 64 << 20, 3, 8),
        (4096 * 4096 * 4, 64 << 20, 2, 1),
        (8192 * 8192 * 8, 64 << 20, 2, 1),   # one frame is over the target
        (1, 64, 3, 4),                        # a float cube root gives 3.99...
    ])
    def test_representative_sizes(self, frame_bytes, target_bytes, n_nav_dims,
                                  expected):
        assert nav_chunk_edge(frame_bytes, target_bytes, n_nav_dims) == expected

    def test_is_the_exact_integer_root(self):
        for frames in range(1, 5000):
            assert nav_chunk_edge(1, frames, 2) == math.isqrt(frames)
            cube = nav_chunk_edge(1, frames, 3)
            assert cube ** 3 <= frames < (cube + 1) ** 3


class TestComposedNavChunkUnchanged:
    """Expected values computed with the ``math.isqrt`` rule this replaced."""

    @pytest.mark.parametrize("detector, dtype, members, expected, expected_10mib", [
        ((256, 256), np.uint16, 4, 20, 6),
        ((507, 501), np.uint16, 4, 10, 3),
        ((128, 128), np.uint8, 1, 80, 25),
        ((64, 64), np.uint16, 2, 80, 25),
        ((4096, 4096), np.float32, 2, 1, 1),
    ])
    def test_matches_the_previous_rule(self, detector, dtype, members, expected,
                                       expected_10mib):
        arrays = [da.zeros((4, 4) + detector, dtype=dtype,
                           chunks=(1, 1) + detector) for _ in range(members)]
        assert composed_nav_chunk(arrays) == expected
        assert composed_nav_chunk(arrays, target_bytes=10 * 1024 ** 2) == expected_10mib

    def test_unknown_frames_fall_back(self):
        assert composed_nav_chunk([]) == 32
        assert composed_nav_chunk([object()]) == 32

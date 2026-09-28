"""CSB event decoding, integration, binning and backend parity (#91).

`test_csb_reader.py` covers everything reachable WITHOUT event payload —
registration, the header, block geometry, rejection. This covers the rest,
which until now ran only in `csb_movie.spec.ts` against a multi-gigabyte
stream that no runner has, so it skipped itself on every CI run.

The fixtures come from `_csb_synthetic.py`, whose `expected_image` computes
what a range of frames must integrate to from the format's documented event
mapping alone. Every assertion here compares the reader against a value
derived INDEPENDENTLY of it, never against a blob recorded from it.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.external.rsciio_csb._core import CSBFile
from spyde.external.rsciio_csb._sparse import SparseCSB
from spyde.tests.migrated._csb_synthetic import (
    BH, BW, H, NF, OH, OW, W, expected_image, grid, write_csb,
)


#: Events at hand-picked places: block 0 twice on one pixel (they must add,
#: not overwrite) plus a neighbour, the LAST raster slot of an interior block
#: (the off-by-one a mid-block event cannot catch), and events spread across
#: frames so an integration range has something to include AND exclude.
EVENTS = {
    (0, 0): [0, 0, 1],
    (0, 5): [BW * BH - 1],
    (1, 0): [BW + 2],
    (2, 7): [0, 5],
    (3, 11): [BW * BH - 1],
}


class TestEventDecoding:
    def test_a_frame_lands_where_the_arithmetic_says(self, tmp_path):
        ds = SparseCSB(write_csb(tmp_path, EVENTS), backend="cpu-numpy")
        assert np.array_equal(ds.integrate_frames(0, 1, 1, np.int64),
                              expected_image(EVENTS, 0, 1))

    def test_repeated_events_on_one_pixel_add(self, tmp_path):
        ds = SparseCSB(write_csb(tmp_path, EVENTS), backend="cpu-numpy")
        got = ds.integrate_frames(0, 1, 1, np.int64)
        assert got[0, 0] == 2, "two events on one pixel must accumulate"

    def test_the_reference_decoder_agrees(self, tmp_path):
        """frame_events is the documented per-event path; it must land on the
        same pixels as the accumulator."""
        f = CSBFile(write_csb(tmp_path, EVENTS))
        ys, xs = f.frame_events(0)
        img = np.zeros((H, W), np.int64)
        for y, x in zip(np.asarray(ys), np.asarray(xs)):
            if 0 <= y < H and 0 <= x < W:
                img[y, x] += 1
        assert np.array_equal(img, expected_image(EVENTS, 0, 1))

    @pytest.mark.parametrize("order", [0, 1])
    def test_block_ordering(self, tmp_path, order):
        """csb_order flips block index -> origin. A square block grid would
        hide the difference, so the fixture's grid is 4x3."""
        ev = {(0, 1): [0], (0, 4): [0]}
        ds = SparseCSB(write_csb(tmp_path, ev, order=order, name=f"o{order}.csb"),
                       backend="cpu-numpy")
        assert np.array_equal(ds.integrate_frames(0, 1, 1, np.int64),
                              expected_image(ev, 0, 1, order=order))

    def test_a_partial_edge_block_is_cropped_not_wrapped(self, tmp_path):
        """70x50 with 16x16 blocks pads to 80x64. An event in the padding must
        vanish, not wrap onto a real pixel — invisible on a frame that divides
        evenly, which is why the even case above is not enough on its own."""
        bpw, bph = grid(OW, OH, BW, BH)            # 5 x 4
        edge = bpw * bph - 1                         # bottom-right, mostly padding
        ev = {(0, edge): [0, BW * BH - 1]}           # one real pixel, one padded
        ds = SparseCSB(write_csb(tmp_path, ev, width=OW, height=OH,
                                name="edge.csb"), backend="cpu-numpy")
        got = ds.integrate_frames(0, 1, 1, np.int64)
        exp = expected_image(ev, 0, 1, width=OW, height=OH)
        assert exp.sum() == 1, "fixture must put exactly one event in the padding"
        assert got.shape == (OH, OW)
        assert np.array_equal(got, exp)


class TestIntegration:
    @pytest.mark.parametrize("f0,f1", [(0, 1), (0, 2), (1, 3), (0, NF), (3, NF)])
    def test_a_range_sums_those_frames_and_no_others(self, tmp_path, f0, f1):
        ds = SparseCSB(write_csb(tmp_path, EVENTS), backend="cpu-numpy")
        assert np.array_equal(ds.integrate_frames(f0, f1, 1, np.int64),
                              expected_image(EVENTS, f0, f1))

    def test_ranges_are_additive(self, tmp_path):
        ds = SparseCSB(write_csb(tmp_path, EVENTS), backend="cpu-numpy")
        whole = ds.integrate_frames(0, NF, 1, np.int64)
        halves = (ds.integrate_frames(0, 2, 1, np.int64)
                  + ds.integrate_frames(2, NF, 1, np.int64))
        assert np.array_equal(whole, halves)

    @pytest.mark.parametrize("bad", [(-1, 2), (0, 0), (2, 1), (0, NF + 1)])
    def test_a_bad_range_is_refused(self, tmp_path, bad):
        ds = SparseCSB(write_csb(tmp_path, EVENTS), backend="cpu-numpy")
        with pytest.raises(ValueError, match="bad frame range"):
            ds.integrate_frames(bad[0], bad[1], 1, np.int64)

    @pytest.mark.parametrize("bin_factor", [2, 4])
    def test_binning_equals_box_summing_the_unbinned(self, tmp_path, bin_factor):
        ds = SparseCSB(write_csb(tmp_path, EVENTS), backend="cpu-numpy")
        full = ds.integrate_frames(0, NF, 1, np.int64)
        got = ds.integrate_frames(0, NF, bin_factor, np.int64)
        h = (H // bin_factor) * bin_factor
        w = (W // bin_factor) * bin_factor
        exp = full[:h, :w].reshape(h // bin_factor, bin_factor,
                                   w // bin_factor, bin_factor).sum(axis=(1, 3))
        assert np.array_equal(got, exp)

    def test_plane_counts_match_the_decoded_frames(self, tmp_path):
        """The free navigator must equal what integrating each plane gives —
        it exists precisely so the overview is NOT built by reducing every
        plane, so nothing else would catch it drifting."""
        from spyde.external.rsciio_csb._api import plane_counts
        ds = SparseCSB(write_csb(tmp_path, EVENTS), backend="cpu-numpy")
        bounds = [(f, f + 1) for f in range(NF)]
        got = np.asarray(plane_counts(ds, bounds))
        exp = np.array([expected_image(EVENTS, f, f + 1).sum() for f in range(NF)])
        assert np.array_equal(got, exp)


class TestBackendParity:
    """`_core.py`'s docstring claims "All three CPU paths are bit-identical to
    one another and to the GPU path". Nothing checked it until now."""

    @pytest.mark.parametrize("backend", ["cpu-numba", "gpu"])
    def test_matches_numpy_bit_for_bit(self, tmp_path, backend):
        from spyde.external.rsciio_csb import _core
        if backend == "cpu-numba" and not _core.numba_available():
            pytest.skip("numba not available")
        if backend == "gpu" and not _core.gpu_available():
            pytest.skip("no CuPy/CUDA device")
        path = write_csb(tmp_path, EVENTS)
        ref = SparseCSB(path, backend="cpu-numpy").integrate_frames(0, NF, 1, np.int64)
        got = SparseCSB(path, backend=backend).integrate_frames(0, NF, 1, np.int64)
        assert np.array_equal(np.asarray(got), ref), f"{backend} diverged from numpy"

    def test_torch_matches_numpy_bit_for_bit(self, tmp_path):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        from spyde.external.rsciio_csb._api import _TorchSparseCSB
        path = write_csb(tmp_path, EVENTS)
        ref = SparseCSB(path, backend="cpu-numpy").integrate_frames(0, NF, 1, np.int64)
        got = _TorchSparseCSB(path).integrate_frames(0, NF, 1, np.int64)
        assert np.array_equal(np.asarray(got), ref), "torch diverged from numpy"

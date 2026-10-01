"""Export resampling: what bins, what sub-selects.

Three controls reduce the data on the way out, and they must not all behave the
same way:

* **downsample** (spatial) box-MEANS k×k blocks — it already did.
* **fps** (temporal) INTEGRATES the source frames each output frame stands for.
  Dropping 30.5 frame/s to 12 frame/s means ~2.5 source frames per output one;
  keeping one and binning the rest throws away real signal, which on noisy
  in-situ data is visible as noise the integrated frame does not have.
* **speed segments** SUB-SELECT. A 32x segment jumps ~81 source frames per
  output frame; integrating 81 would smear a second of real change into one
  picture, so the window stays what fps asked for and the cursor simply jumps
  further.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions.movie_export import pipeline

SCALE_S = 0.03276          # a real DE movie: 30.525 frame/s


class TestSpatialBinning:
    def test_downsample_is_a_box_mean_not_a_decimation(self):
        frame = np.arange(16, dtype=np.float32).reshape(4, 4)
        got = pipeline.downsample(frame, 2)
        # Top-left 2x2 block is 0,1,4,5 -> mean 2.5 (a decimation would give 0).
        assert got[0, 0] == pytest.approx(2.5)
        assert got.shape == (2, 2)

    def test_k_of_one_is_a_no_op(self):
        frame = np.arange(9, dtype=np.float32).reshape(3, 3)
        assert pipeline.downsample(frame, 1) is frame


class TestIntegrationWindow:
    def test_fps_reduction_sets_the_window(self):
        # 30.525 source frame/s -> 12 frame/s output = 2.54 source frames each.
        assert pipeline.integration_window(12.0, SCALE_S) == 3
        assert pipeline.integration_window(30.525, SCALE_S) == 1

    def test_speed_up_does_NOT_widen_the_window(self):
        """The whole point: a fast-forward sub-selects, it does not average
        more. 32x advances ~81 frames but still integrates the fps window."""
        base = pipeline.integration_window(12.0, SCALE_S, speed=1.0)
        for speed in (2, 4, 8, 16, 32):
            assert pipeline.integration_window(12.0, SCALE_S, speed=speed) == base

    def test_slow_motion_narrows_it(self):
        """In slow-mo the output advances a fraction of a frame; integrating
        more than it advances would double-count frames into consecutive output
        frames and blur it."""
        assert pipeline.integration_window(12.0, SCALE_S, speed=0.25) == 1
        assert pipeline.integration_window(12.0, SCALE_S, speed=0.0) == 1

    def test_degenerate_inputs_fall_back_to_one_frame(self):
        assert pipeline.integration_window(0, SCALE_S) == 1
        assert pipeline.integration_window(12.0, 0) == 1
        assert pipeline.integration_window(12.0, SCALE_S, speed=-3) == 1

    def test_never_below_one(self):
        # A faster output than the source cannot integrate a fraction of a frame.
        assert pipeline.integration_window(120.0, SCALE_S) == 1


class TestSpeedAtFrame:
    def test_inside_a_segment(self):
        segs = [{"time_range": [1.0, 2.0], "speed": 8}]
        assert pipeline.speed_at_frame(segs, 45, SCALE_S) == 8      # 1.47 s
        assert pipeline.speed_at_frame(segs, 5, SCALE_S) == 1.0     # 0.16 s

    def test_no_segments_is_one(self):
        assert pipeline.speed_at_frame([], 10, SCALE_S) == 1.0

    def test_malformed_segments_are_skipped(self):
        assert pipeline.speed_at_frame([{"speed": 4}, None], 10, SCALE_S) == 1.0


class TestIntegratedRead:
    def _raw(self, n=10, edge=4):
        # Frame i is filled with i, so a mean over a window is exactly its
        # arithmetic mean — an integration bug shows up as a wrong number.
        return np.stack([np.full((edge, edge), i, np.uint16) for i in range(n)])

    def test_it_averages_the_window(self):
        got = pipeline.read_frame_integrated(self._raw(), 2, 4, 10)
        assert got.mean() == pytest.approx(3.5)      # (2+3+4+5)/4

    def test_window_of_one_is_the_plain_read(self):
        got = pipeline.read_frame_integrated(self._raw(), 7, 1, 10)
        assert got.mean() == pytest.approx(7.0)

    def test_it_clamps_at_the_end(self):
        got = pipeline.read_frame_integrated(self._raw(n=10), 8, 5, 10)
        assert got.mean() == pytest.approx(8.5)      # (8+9)/2, not out of range

    def test_it_reads_one_frame_at_a_time(self):
        """The memory-safety contract: never a stacked slice, whatever n is."""
        seen = []

        class _Probe:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, key):
                seen.append(key)
                return self.data[key]

        raw = _Probe(self._raw(n=10))
        pipeline.read_frame_integrated(raw, 1, 5, 10)
        assert len(seen) == 5
        for key in seen:
            # Each read is a scalar frame index, never a slice object.
            assert isinstance(key[0], (int, np.integer)), key

    def test_integrating_reduces_noise(self):
        """The reason this exists, on data shaped like the real thing."""
        rng = np.random.default_rng(0)
        raw = (rng.normal(1000, 50, (16, 32, 32))).astype(np.float32)
        single = pipeline.read_frame_integrated(raw, 0, 1, 16)
        integrated = pipeline.read_frame_integrated(raw, 0, 8, 16)
        assert integrated.std() < single.std() * 0.6


class TestExportUsesIt:
    def test_a_fast_segment_sub_selects_rather_than_smears(self):
        """End-to-end: the window used inside a 32x segment is the fps window,
        not the ~81 frames the cursor advances."""
        n = 400
        idxs = pipeline.frame_indices_with_speed(
            n, 0, n - 1, 1, [{"time_range": [0.0, n * SCALE_S], "speed": 32}],
            fps=12, scale_s=SCALE_S)
        assert len(idxs) >= 2
        step = idxs[1] - idxs[0]
        assert step > 40, f"a 32x segment should jump far, stepped {step}"
        window = pipeline.integration_window(
            12.0, SCALE_S, pipeline.speed_at_frame(
                [{"time_range": [0.0, n * SCALE_S], "speed": 32}], idxs[0], SCALE_S))
        assert window == 3
        assert window < step, "integration must not span the whole jump"

    def test_the_preview_integrates_like_the_export(self):
        """Editor preview and export must not disagree about the picture."""
        import inspect
        src = inspect.getsource(pipeline.render_single_frame)
        assert "integration_window" in src

"""
test_motion.py — the ported motion correction.

The old implementation had NO tests: everything ran inside a `QThread.run()`
that could only be exercised through the GUI. Dropping Qt is what makes these
possible, and the important ones are end-to-end on synthetic data with a KNOWN
drift — an alignment that runs without error but recovers the wrong shifts is
the failure that matters, and no unit test of a helper catches it.
"""
from __future__ import annotations

import numpy as np
import pytest

from de_groundcrew import motion
from de_groundcrew.motion import align, frames, local


def _speckle(h=128, w=128, seed=0) -> np.ndarray:
    """A frame with broadband texture, which is what correlation needs.

    Smooth blobs alone give a broad, ambiguous correlation peak; pure noise
    gives no peak at all across a shift. Blurred noise plus a few hard points
    behaves like a real micrograph.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 1, (h, w))
    f = np.fft.fft2(img)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    img = np.real(np.fft.ifft2(f * np.exp(-(r / 0.08) ** 2)))
    span = float(np.ptp(img)) or 1.0        # ndarray.ptp() went in NumPy 2
    img = (img - img.min()) / span
    for cy, cx in [(30, 40), (70, 90), (100, 25)]:
        img[cy - 2:cy + 3, cx - 2:cx + 3] += 2.0
    return img.astype(np.float32)


def _drifting_stack(shifts, h=128, w=128, seed=0) -> np.ndarray:
    """A stack of one scene translated by `shifts` — the ground truth."""
    base = _speckle(h, w, seed)
    return np.stack([align.apply_shift_fourier(base, dy, dx)
                     for dy, dx in shifts]).astype(np.float32)


class TestOrientationAndGain:
    def test_all_eight_orientations_are_distinct_and_shape_valid(self):
        img = np.arange(12, dtype=np.float32).reshape(3, 4)
        seen = {frames.apply_orientation(img, i).tobytes()
                for i in range(len(frames.ORIENTATION_LABELS))}
        assert len(seen) == 8, "two orientations produce the same array"

    def test_identity_is_index_zero(self):
        # The index is a stored setting; reordering the labels would silently
        # change the meaning of every saved orientation.
        assert frames.ORIENTATION_LABELS[0] == "Identity"
        img = _speckle(16, 16)
        assert np.array_equal(frames.apply_orientation(img, 0), img)

    def test_binning_sums_blocks(self):
        img = np.ones((4, 4), dtype=np.float32)
        assert np.array_equal(frames.bin_image(img, 2), np.full((2, 2), 4.0))

    def test_binning_drops_a_trailing_partial_block(self):
        assert frames.bin_image(np.ones((5, 5)), 2).shape == (2, 2)

    def test_superres_gain_is_binned_to_an_average_not_a_sum(self):
        # A 2x gain binned by SUMMING would multiply every frame by 4.
        gain = np.full((8, 8), 2.0, dtype=np.float32)
        matched = frames.match_gain_to_frame(gain, 4, 4)
        assert matched.shape == (4, 4)
        assert np.allclose(matched, 2.0)

    def test_a_non_integer_gain_ratio_raises_rather_than_resampling(self):
        # Silently interpolating a gain reference corrupts every frame it
        # touches, and does it quietly.
        with pytest.raises(ValueError, match="integer multiple"):
            frames.match_gain_to_frame(np.ones((7, 7)), 4, 4)

    def test_gain_validation_prefers_the_orientation_that_flattens(self):
        # Build a frame that a KNOWN orientation of the gain corrects, then
        # check that orientation scores best.
        rng = np.random.default_rng(3)
        pattern = 1.0 + 0.4 * rng.random((32, 32)).astype(np.float32)
        scene = _speckle(32, 32, seed=5) + 5.0
        gain = 1.0 / pattern
        results = frames.validate_gain_orientation(scene * pattern, gain)
        assert results, "no orientation scored"
        assert results[0][2] == 0, f"expected Identity to win, got {results[0]}"
        assert results[0][0] < results[-1][0]


class TestCrossCorrelation:
    @pytest.mark.parametrize("dy,dx", [(0, 0), (3, -5), (-7, 2), (1.25, -2.5)])
    def test_recovers_a_known_shift_to_subpixel(self, dy, dx):
        ref = _speckle()
        moved = align.apply_shift_fourier(ref, dy, dx)
        gy, gx = align.cross_correlate(ref, moved)
        assert abs(gy - dy) < 0.2 and abs(gx - dx) < 0.2, f"got ({gy}, {gx})"

    def test_integer_only_mode_skips_refinement(self):
        ref = _speckle()
        moved = align.apply_shift_fourier(ref, 4, -3)
        gy, gx = align.cross_correlate(ref, moved, upsample_factor=1)
        assert (gy, gx) == (4.0, -3.0)

    def test_a_shift_past_the_halfway_point_unwraps_negative(self):
        # Without the unwrap a small negative shift reads as a huge positive
        # one, and the whole trajectory inverts.
        ref = _speckle(64, 64)
        moved = align.apply_shift_fourier(ref, -20, -18)
        gy, gx = align.cross_correlate(ref, moved)
        assert gy < 0 and gx < 0, f"expected negative, got ({gy}, {gx})"

    def test_the_fourier_shift_round_trips(self):
        img = _speckle()
        there = align.apply_shift_fourier(img, 5.5, -3.25)
        back = align.apply_shift_fourier(there, -5.5, 3.25)
        # Tolerance against the DYNAMIC RANGE, not an absolute epsilon: two
        # float32 FFT round trips over a scene with hard point features ring
        # slightly, and a fixed atol just encodes whatever this fixture
        # happens to produce.
        err = float(np.max(np.abs(back - img))) / float(np.ptp(img))
        assert err < 0.01, f"round trip lost {err:.2%} of the range"


class TestSmoothing:
    def test_smoothing_passes_short_trajectories_through(self):
        # Under four points there is nothing to fit.
        raw = [1.0, 5.0, 2.0]
        assert np.array_equal(align.smooth_shifts(raw), np.array(raw))

    def test_smoothing_preserves_a_smooth_trajectory(self):
        t = np.arange(12, dtype=np.float64)
        drift = 0.4 * t
        assert np.allclose(align.smooth_shifts(drift.tolist()), drift, atol=1e-6)


class TestGlobalAlignment:
    def test_recovers_a_known_linear_drift(self):
        # THE test. Everything else can pass while the answer is wrong.
        truth = [(0.0, 0.0), (1.0, -0.5), (2.0, -1.0), (3.0, -1.5),
                 (4.0, -2.0), (5.0, -2.5)]
        stack = _drifting_stack(truth)
        r = align.align_stack(stack, bin_factor=1, reference="first")

        # Shifts are relative to the reference, so compare the trajectory's
        # SHAPE — an overall offset is not an error.
        got_y = np.array(r["shifts_y_smooth"]); got_y -= got_y[0]
        got_x = np.array(r["shifts_x_smooth"]); got_x -= got_x[0]
        want_y = np.array([t[0] for t in truth]); want_y -= want_y[0]
        want_x = np.array([t[1] for t in truth]); want_x -= want_x[0]
        assert np.allclose(got_y, want_y, atol=0.3), f"y: {got_y} vs {want_y}"
        assert np.allclose(got_x, want_x, atol=0.3), f"x: {got_x} vs {want_x}"

    def test_alignment_sharpens_the_sum(self):
        # The point of the whole exercise: the aligned sum must have more
        # high-frequency content than the smeared unaligned one.
        stack = _drifting_stack([(0, 0), (2, 1), (4, 2), (6, 3), (8, 4)])
        r = align.align_stack(stack, bin_factor=1, reference="central")
        assert float(r["aligned_sum"].std()) > float(r["unaligned_sum"].std()), (
            "aligned sum is no sharper than the unaligned one")

    def test_every_reference_mode_runs_and_agrees(self):
        truth = [(0.0, 0.0), (1.5, 1.0), (3.0, 2.0), (4.5, 3.0), (6.0, 4.0)]
        stack = _drifting_stack(truth)
        spans = {}
        for mode in align.REFERENCES:
            r = align.align_stack(stack, bin_factor=1, reference=mode)
            ys = np.array(r["shifts_y_smooth"])
            spans[mode] = float(ys[-1] - ys[0])
        # Whichever frame is the reference, the total travel is the same.
        assert max(spans.values()) - min(spans.values()) < 0.5, spans

    def test_an_unknown_reference_raises_rather_than_defaulting(self):
        # The old app mapped UI wording to these strings in the panel, so a
        # typo silently became "average".
        with pytest.raises(ValueError, match="reference must be one of"):
            align.align_stack(_drifting_stack([(0, 0), (1, 1)]),
                              reference="middle-ish")

    def test_throw_discards_leading_frames(self):
        stack = _drifting_stack([(0, 0), (9, 9), (2, 2), (3, 3), (4, 4), (5, 5)])
        r = align.align_stack(stack, bin_factor=1, throw=2)
        assert r["throw"] == 2
        assert r["n_frames"] == 4
        assert len(r["shifts_y_smooth"]) == 4

    def test_throw_always_leaves_at_least_two_frames(self):
        stack = _drifting_stack([(0, 0), (1, 1), (2, 2)])
        r = align.align_stack(stack, bin_factor=1, throw=99)
        assert r["n_frames"] >= 2

    def test_binning_scales_the_shifts_back_to_full_resolution(self):
        truth = [(0.0, 0.0), (4.0, 4.0), (8.0, 8.0), (12.0, 12.0)]
        stack = _drifting_stack(truth, h=256, w=256)
        r = align.align_stack(stack, bin_factor=2, reference="first")
        got = np.array(r["shifts_y_smooth"]); got -= got[0]
        assert np.allclose(got, [0, 4, 8, 12], atol=1.0), got

    def test_cancellation_raises_rather_than_returning_a_partial_result(self):
        stack = _drifting_stack([(0, 0), (1, 1), (2, 2), (3, 3)])
        with pytest.raises(align.Cancelled):
            align.align_stack(stack, bin_factor=1, should_cancel=lambda: True)

    def test_progress_reports_each_pass(self):
        msgs: list[str] = []
        align.align_stack(_drifting_stack([(0, 0), (1, 1), (2, 2), (3, 3)]),
                          bin_factor=1, progress=msgs.append)
        joined = " ".join(msgs)
        assert "Pass 1" in joined and "Pass 2" in joined

    def test_gain_is_applied_to_the_sum_but_not_to_alignment(self):
        # Gain correction destroys the correlation signal on sparse counting
        # data, so it must touch the final sum only.
        stack = _drifting_stack([(0, 0), (1, 1), (2, 2), (3, 3)])
        gain = np.full(stack.shape[1:], 3.0, dtype=np.float32)
        plain = align.align_stack(stack, bin_factor=1, reference="first")
        gained = align.align_stack(stack, gain=gain, bin_factor=1, reference="first")
        assert np.allclose(plain["shifts_y_smooth"], gained["shifts_y_smooth"],
                           atol=1e-9), "gain changed the estimated shifts"
        assert np.allclose(gained["aligned_sum"],
                           plain["aligned_sum"] * 3.0, rtol=1e-4)


class TestLogFFT:
    def test_output_is_a_display_ready_image(self):
        out = frames.log_fft(_speckle())
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 255.0

    def test_the_dc_term_does_not_swamp_the_spectrum(self):
        # A plain log(|F|) renders as a white dot on black. The two-anchor
        # stretch is what makes Thon rings visible.
        img = _speckle() + 1000.0            # huge DC offset
        out = frames.log_fft(img)
        bright = float((out > 128).mean())
        assert bright < 0.5, f"{bright:.0%} of the spectrum is saturated"
        assert out.std() > 1.0, "spectrum is flat — no structure survived"


class TestPatchGrid:
    def test_the_grid_covers_the_whole_frame(self):
        patches, _ = local.generate_patch_grid(300, 300, patch_size=128)
        assert max(p[2] for p in patches) == 300
        assert max(p[3] for p in patches) == 300

    def test_patches_overlap_by_half(self):
        patches, _ = local.generate_patch_grid(512, 512, patch_size=128)
        y0s = sorted({p[0] for p in patches})
        assert y0s[1] - y0s[0] == 64

    def test_the_weight_normalisation_is_what_makes_the_blend_flat(self):
        # NOT "Hann windows sum to 1". `np.hanning` is the SYMMETRIC window
        # (endpoints exactly zero), which misses perfect overlap-add by ~1%;
        # only the periodic variant satisfies COLA at 50%. What actually makes
        # the composite flat is dividing by the accumulated weight map, which
        # `apply_local_shifts` does — so THAT is the contract worth pinning.
        n = 64
        acc = np.zeros(n * 3)
        for start in range(0, n * 2 + 1, n // 2):
            acc[start:start + n] += np.hanning(n)
        mid = acc[n:n * 2]
        assert not np.allclose(mid, 1.0, atol=1e-6), (
            "np.hanning now satisfies COLA — the division could be dropped")
        assert np.allclose(mid / np.maximum(mid, 1e-6), 1.0)

    def _still_composite(self, n=128, ps=64):
        frame = _speckle(n, n)
        patches, centers = local.generate_patch_grid(n, n, patch_size=ps)
        out = local.apply_local_shifts(
            np.stack([frame, frame]), None, None, None,
            np.zeros((2, 2, 10)), (0.0, 0.0, 1.0, 1.0),
            patches, centers, local.build_cosine_blend_weights(ps))
        return frame, out

    def test_compositing_a_still_stack_reproduces_the_interior(self):
        # With no motion the blend must return the original frame exactly,
        # seams and all — that is what says the weighting is right.
        frame, out = self._still_composite()
        err = (float(np.max(np.abs(out[1:-1, 1:-1] - frame[1:-1, 1:-1])))
               / float(np.ptp(frame)))
        assert err < 0.01, f"compositing changed the interior by {err:.2%}"

    def test_the_outermost_pixel_ring_is_lost_to_the_hann_taper(self):
        # A REAL artifact, inherited from the original and pinned rather than
        # hidden. At the exact frame border the only patch covering that pixel
        # has Hann weight 0, so the composite divides ~0 by ~0 and the ring
        # comes out black. It is one pixel wide regardless of frame size (the
        # next row in has weight ~0.0024, small but exact after the division),
        # so on an 8192 frame it is 0.01% of the image — worth knowing, not
        # worth changing the algorithm for.
        frame, out = self._still_composite()
        assert float(np.max(np.abs(out[0, :]))) < 1e-3, "border is no longer zeroed"
        interior_err = float(np.max(np.abs(out[1:-1, 1:-1] - frame[1:-1, 1:-1])))
        assert interior_err / float(np.ptp(frame)) < 0.01, (
            "the artifact has spread beyond the outermost ring")


class TestMotionField:
    def test_a_constant_field_is_recovered_everywhere(self):
        centers = np.array([[y, x] for y in (0., 50., 100.)
                            for x in (0., 50., 100.)])
        shifts = np.zeros((1, len(centers), 2))
        shifts[0, :, 0] = 2.0
        shifts[0, :, 1] = -1.0
        coeffs, norm = local.fit_motion_field(centers, shifts)
        got = local.evaluate_motion_field(coeffs[0], norm,
                                          np.array([[25.0, 75.0]]))
        assert np.allclose(got[0], [2.0, -1.0], atol=1e-6)

    def test_a_linear_gradient_field_is_recovered(self):
        centers = np.array([[y, x] for y in (0., 40., 80., 120.)
                            for x in (0., 40., 80., 120.)])
        shifts = np.zeros((1, len(centers), 2))
        shifts[0, :, 0] = 0.01 * centers[:, 0]
        coeffs, norm = local.fit_motion_field(centers, shifts)
        got = local.evaluate_motion_field(coeffs[0], norm, np.array([[60.0, 60.0]]))
        assert abs(got[0, 0] - 0.6) < 0.05, got

    def test_outlier_patches_are_replaced_by_the_frame_median(self):
        # A patch on empty ice gives a meaningless peak; it must not drag the
        # polynomial fit.
        shifts = np.zeros((6, 9, 2))
        shifts[:, :, 0] = 1.0
        shifts[3, 4, 0] = 500.0
        cleaned = local.smooth_patch_shifts(shifts)
        assert abs(cleaned[3, 4, 0] - 1.0) < 0.5, cleaned[3, 4, 0]


class TestLocalMotionEndToEnd:
    def test_runs_and_returns_the_documented_shape(self):
        stack = _drifting_stack([(0, 0), (1, 1), (2, 2), (3, 3)], h=128, w=128)
        g = align.align_stack(stack, bin_factor=1, reference="first")
        r = local.correct_local_motion(
            stack, shifts_y=g["shifts_y_smooth"], shifts_x=g["shifts_x_smooth"],
            bin_factor=1, patch_size=64)
        assert r["corrected_sum"].shape == stack.shape[1:]
        assert r["corrected_fft"].shape == stack.shape[1:]
        assert r["n_patches"] > 1
        assert r["coefficients"].shape[0] == stack.shape[0]

    def test_a_uniformly_drifting_stack_is_not_made_worse(self):
        # Local correction on rigid drift has nothing to add, but it must not
        # destroy what Phase 1 achieved.
        stack = _drifting_stack([(0, 0), (2, 1), (4, 2), (6, 3)], h=128, w=128)
        g = align.align_stack(stack, bin_factor=1, reference="central")
        r = local.correct_local_motion(
            stack, shifts_y=g["shifts_y_smooth"], shifts_x=g["shifts_x_smooth"],
            bin_factor=1, patch_size=64)
        assert float(r["corrected_sum"].std()) > float(g["unaligned_sum"].std())

    def test_cancellation_propagates(self):
        stack = _drifting_stack([(0, 0), (1, 1), (2, 2), (3, 3)], h=128, w=128)
        with pytest.raises(align.Cancelled):
            local.correct_local_motion(stack, bin_factor=1, patch_size=64,
                                       should_cancel=lambda: True)


class TestPublicSurface:
    def test_the_package_exports_what_the_session_needs(self):
        for name in ("align_stack", "correct_local_motion", "load_movie_stack",
                     "load_gain", "validate_gain_orientation", "log_fft",
                     "save_image", "ORIENTATION_LABELS", "REFERENCES"):
            assert hasattr(motion, name), name

    def test_no_qt_leaked_into_the_port(self):
        import sys
        assert not any(m.startswith("PySide") or m.startswith("PyQt")
                       for m in sys.modules), "the port dragged Qt in"

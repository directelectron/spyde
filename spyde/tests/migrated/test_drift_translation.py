"""
Tests for spyde.drift — rigid translation solve, warp, and DriftModel.

The acceptance gate from DRIFT_AND_PARTICLES_PLAN.md is numerical, not
structural: recover a synthetically applied shift to better than 0.1 px, and
agree with ``skimage.registration.phase_cross_correlation`` on the same data.
That is what most of this file asserts.

Qt-free and dask-free — these are pure-compute tests on small synthetic stacks.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.drift import DriftModel, coverage_mask, frame_source, shift_frame
from spyde.drift.translation import solve_translation

# The solver's own tolerance target. Sub-pixel ground truth on a smooth
# synthetic scene should land well inside this.
GATE_PX = 0.1


# ── synthetic data ───────────────────────────────────────────────────────────

def _scene(h=96, w=112, seed=3, noise=0.02):
    """A smooth, asymmetric, non-periodic scene.

    Asymmetric on purpose: a symmetric scene correlates equally well at several
    offsets, so a sign error or an axis swap would still pass. Non-periodic on
    purpose: a lattice invites the exact wrong-translation lock that ``max_shift``
    exists to prevent, which is a separate test.

    ``noise=0`` gives a band-limited scene, needed wherever a test resamples
    twice — bilinear interpolation legitimately destroys pixel-scale noise, so a
    round-trip assertion on a noisy scene measures interpolation loss, not the
    property under test.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w), dtype=np.float64)
    # A handful of gaussian blobs at irregular positions and widths.
    for cy, cx, amp, sig in [
        (0.28 * h, 0.22 * w, 1.0, 5.0),
        (0.61 * h, 0.44 * w, 0.7, 8.0),
        (0.38 * h, 0.73 * w, 0.9, 4.0),
        (0.79 * h, 0.66 * w, 0.5, 6.5),
        (0.17 * h, 0.58 * w, 0.6, 3.5),
    ]:
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))
    if noise:
        img += noise * rng.standard_normal((h, w))
    return img


def _shifted_stack(shifts, h=96, w=112, seed=3):
    """Stack whose frame i is the scene translated by ``-shifts[i]``.

    So the CORRECTION needed for frame i is ``+shifts[i]`` — matching the
    DriftModel sign convention. Built by Fourier phase ramp so sub-pixel truth is
    exact rather than interpolated, which keeps the 0.1 px gate meaningful.
    """
    base = _scene(h, w, seed)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    F = np.fft.fft2(base)
    frames = []
    for dy, dx in shifts:
        # Applying -shift here means +shift is the correction.
        ramp = np.exp(-2j * np.pi * (-dy * fy + -dx * fx))
        frames.append(np.real(np.fft.ifft2(F * ramp)))
    return np.stack(frames).astype(np.float32)


class TestFrameSource:
    def test_numpy_stack(self):
        arr = np.zeros((5, 8, 9), dtype=np.uint16)
        n, get, shape = frame_source(arr)
        assert n == 5 and shape == (8, 9)
        assert get(3).shape == (8, 9)

    def test_sequence_of_frames(self):
        seq = [np.zeros((4, 6)) for _ in range(3)]
        n, get, shape = frame_source(seq)
        assert n == 3 and shape == (4, 6)

    def test_rejects_2d(self):
        with pytest.raises(TypeError, match="3-D"):
            frame_source(np.zeros((8, 9)))

    def test_rejects_unknown(self):
        with pytest.raises(TypeError, match="cannot read frames"):
            frame_source(object())

    def test_hyperspy_signal_wrong_nav_dim_is_rejected(self):
        """A 4D-STEM scan is not a movie; say so instead of solving nonsense."""
        class _AM:
            navigation_dimension = 2
            signal_dimension = 2

        class _Sig:
            axes_manager = _AM()
            data = np.zeros((3, 3, 4, 4))

        with pytest.raises(TypeError, match="1-D navigation"):
            frame_source(_Sig())

    def test_dask_reads_one_frame_only(self):
        """The Memory-Safety rule, enforced: never compute the whole array."""
        da = pytest.importorskip("dask.array")
        arr = da.zeros((6, 8, 9), chunks=(1, 8, 9))
        n, get, shape = frame_source(arr)
        assert n == 6 and shape == (8, 9)
        called = {"full": 0}
        real_compute = da.Array.compute

        def guard(self, *a, **k):
            if self.shape == (6, 8, 9):
                called["full"] += 1
            return real_compute(self, *a, **k)

        try:
            da.Array.compute = guard
            f = get(2)
        finally:
            da.Array.compute = real_compute
        assert f.shape == (8, 9)
        assert called["full"] == 0, "sliced a frame but computed the whole stack"


class TestSolveTranslationAccuracy:
    def test_recovers_integer_shifts(self):
        truth = np.array([[0, 0], [3, -4], [-6, 2], [1, 7]], dtype=float)
        stack = _shifted_stack(truth)
        model = solve_translation(stack, device="numpy", upsample=8)
        assert np.allclose(model.shifts, truth, atol=GATE_PX), model.shifts

    def test_recovers_subpixel_shifts_inside_gate(self):
        """The headline acceptance gate: < 0.1 px on sub-pixel ground truth.

        The truth values are deliberately **off** the ``1/upsample`` grid. Shifts
        that happen to be multiples of 1/8 are recovered to 0.00000 px by an
        upsample=8 solve — which looks like a spectacular result and actually
        tests nothing, because the answer is exactly representable. Off-grid truth
        is what makes the tolerance meaningful.
        """
        truth = np.array(
            [[0, 0], [1.37, -2.83], [-3.06, 0.61], [4.19, 5.44], [-0.72, -1.28]],
            dtype=float,
        )
        # None of these may land on the upsampled grid, or the test is vacuous.
        assert not np.any(np.isclose(truth[1:] * 8, np.round(truth[1:] * 8)))

        stack = _shifted_stack(truth)
        model = solve_translation(stack, device="numpy", upsample=8,
                                  reference="first")
        err = np.abs(model.shifts - truth)
        assert err.max() < GATE_PX, f"max error {err.max():.4f} px\n{model.shifts}"

    def test_higher_upsample_reduces_error(self):
        """Off-grid error should shrink as the upsampled grid gets finer.

        This is the test that would have caught the `_upsampled_dft` bug where
        the frequency scaling was omitted: with that bug every result quantised to
        1/upsample regardless, so raising upsample changed the quantum but the
        error stayed the same order. Here it must genuinely improve.
        """
        truth = np.array([[0, 0], [2.31, -1.77], [-3.42, 4.09]], dtype=float)
        stack = _shifted_stack(truth)
        errs = {}
        for u in (2, 8, 32):
            m = solve_translation(stack, device="numpy", upsample=u,
                                  reference="first")
            errs[u] = float(np.abs(m.shifts - truth).max())
        assert errs[8] < errs[2], errs
        assert errs[32] <= errs[8] + 1e-4, errs

    def test_frame_zero_is_the_origin(self):
        stack = _shifted_stack(np.array([[0, 0], [2, 3]], dtype=float))
        model = solve_translation(stack, device="numpy")
        assert tuple(model.shifts[0]) == (0.0, 0.0)

    def test_agrees_with_skimage_reference(self):
        """Parity against the implementation we are replacing."""
        skreg = pytest.importorskip("skimage.registration")
        truth = np.array([[0, 0], [2.25, -3.5], [-1.75, 4.125]], dtype=float)
        stack = _shifted_stack(truth)
        model = solve_translation(stack, device="numpy", upsample=8,
                                  reference="first", apodize=False)
        for i in range(1, len(truth)):
            ref, _, _ = skreg.phase_cross_correlation(
                stack[0], stack[i], upsample_factor=8, normalization="phase")
            assert np.allclose(model.shifts[i], ref, atol=0.05), (
                f"frame {i}: ours={model.shifts[i]} skimage={ref}")

    def test_sequential_reference_accumulates(self):
        """Sequential mode must return CUMULATIVE shifts, not per-pair deltas."""
        truth = np.array([[0, 0], [2, 0], [4, 0], [6, 0]], dtype=float)
        stack = _shifted_stack(truth)
        model = solve_translation(stack, device="numpy", reference="sequential",
                                  upsample=4)
        assert np.allclose(model.shifts, truth, atol=GATE_PX), model.shifts

    def test_running_reference_survives_one_corrupt_frame(self):
        """Why 'running' is the default: a single bad frame must not poison it.

        The frames AFTER the corrupt one are what matters. The corrupt frame's own
        shift is meaningless by construction and is not asserted on.
        """
        truth = np.array([[0, 0], [2, 1], [4, 2], [6, 3], [8, 4]], dtype=float)
        stack = _shifted_stack(truth).copy()
        rng = np.random.default_rng(0)
        stack[2] = rng.standard_normal(stack.shape[1:]).astype(np.float32)  # garbage
        model = solve_translation(stack, device="numpy", upsample=8, max_shift=20)
        good = [1, 3, 4]
        err = np.abs(model.shifts[good] - truth[good]).max()
        assert err < 0.5, f"good frames drifted after a corrupt frame: {model.shifts}"
        assert model.params["rejected_from_reference"] >= 1, (
            "nothing was kept out of the reference, so this passed by luck rather "
            "than by the outlier rejection it is meant to exercise")

    def test_outlier_rejection_can_be_disabled(self):
        """And with it off, the corrupt frame really does poison the reference —
        which is what makes the test above non-vacuous."""
        truth = np.array([[0, 0], [2, 1], [4, 2], [6, 3], [8, 4]], dtype=float)
        stack = _shifted_stack(truth).copy()
        rng = np.random.default_rng(0)
        stack[2] = rng.standard_normal(stack.shape[1:]).astype(np.float32)
        model = solve_translation(stack, device="numpy", upsample=8, max_shift=20,
                                  reject_outliers=False)
        assert model.params["rejected_from_reference"] == 0
        good = [3, 4]
        assert np.abs(model.shifts[good] - truth[good]).max() > 1.0, (
            "the corrupt frame no longer poisons an unprotected reference — if the "
            "solver became robust some other way, check deliberately")

    def test_clean_stack_rejects_nothing(self):
        """The rejection must not fire on ordinary frame-to-frame variation."""
        truth = np.array([[0, 0], [1.5, 0.5], [3, 1], [4.5, 1.5], [6, 2]], float)
        model = solve_translation(_shifted_stack(truth), device="numpy",
                                  upsample=8, max_shift=20)
        assert model.params["rejected_from_reference"] == 0
        assert np.abs(model.shifts - truth).max() < GATE_PX


class TestSolveTranslationGuards:
    def test_max_shift_rejects_far_peak(self):
        """A shift beyond max_shift is clamped out of the search, not returned."""
        truth = np.array([[0, 0], [20, 0]], dtype=float)
        stack = _shifted_stack(truth)
        model = solve_translation(stack, device="numpy", max_shift=5, upsample=1)
        assert abs(model.shifts[1][0]) <= 5.0 + 1e-6, model.shifts

    def test_impossible_bounds_raise(self):
        stack = _shifted_stack(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="exclude every possible shift"):
            solve_translation(stack, device="numpy", max_shift=1, min_shift=50)

    def test_bad_reference_name_raises(self):
        stack = _shifted_stack(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="unknown reference"):
            solve_translation(stack, device="numpy", reference="nonsense")

    def test_fixed_index_out_of_range_raises(self):
        stack = _shifted_stack(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="outside"):
            solve_translation(stack, device="numpy", reference="fixed:99")

    def test_progress_reports_every_frame(self):
        stack = _shifted_stack(np.zeros((4, 2)))
        seen = []
        solve_translation(stack, device="numpy", progress=lambda d, t: seen.append((d, t)))
        assert seen[0] == (1, 4) and seen[-1] == (4, 4)

    def test_on_shift_streams_every_frame_as_it_solves(self):
        """The drift caret draws its curve live; `progress` cannot carry that.

        `progress` is only a count, and the shift array is solver-local until the
        return — so without this callback a UI can show a bar but not a trace.
        """
        truth = np.array([[0, 0], [2, 1], [4, 2], [6, 3]], dtype=float)
        seen = []
        model = solve_translation(_shifted_stack(truth), device="numpy",
                                  upsample=8, reference="first",
                                  on_shift=lambda i, dy, dx, s: seen.append((i, dy, dx)))
        assert [i for i, _, _ in seen] == list(range(len(truth))), (
            f"expected one callback per frame in order, got {seen}")
        streamed = np.array([[dy, dx] for _, dy, dx in seen])
        assert np.allclose(streamed, model.shifts, equal_nan=True), (
            "the streamed values disagree with the returned array")

    def test_on_shift_is_optional(self):
        stack = _shifted_stack(np.array([[0, 0], [1, 1]], float))
        assert solve_translation(stack, device="numpy").n_frames == 2

    def test_cancel_leaves_nan_not_a_silent_partial(self):
        stack = _shifted_stack(np.array([[0, 0], [1, 1], [2, 2], [3, 3]], float))
        calls = {"n": 0}

        def cancel():
            calls["n"] += 1
            return calls["n"] > 1

        model = solve_translation(stack, device="numpy", cancel=cancel)
        assert np.isnan(model.shifts[-1]).all(), (
            "a cancelled solve must be detectable, not quietly truncated")


class TestAlignmentROI:
    """Correlating on a sub-region. Not a speed switch — often the RIGHT answer.

    Whole-frame correlation averages over everything that moved, so on a movie
    where the sample itself evolves, the sample's motion contaminates the estimate
    of the stage's. Restricting to a static landmark measures the stage alone.
    """

    def test_roi_recovers_the_same_shift_as_the_full_frame(self):
        truth = np.array([[0, 0], [2.5, -1.75], [-3.25, 4.0]], dtype=float)
        stack = _shifted_stack(truth, h=96, w=112)
        full = solve_translation(stack, device="numpy", upsample=8,
                                 reference="first")
        roi = solve_translation(stack, device="numpy", upsample=8,
                                reference="first", roi=(20, 20, 56, 64))
        assert np.abs(roi.shifts - truth).max() < 0.3, roi.shifts
        assert np.abs(roi.shifts - full.shifts).max() < 0.3, (
            "the ROI solve disagrees with the full-frame solve on the same data")

    def test_shifts_apply_to_the_whole_frame(self):
        """A translation is a translation; the ROI only chooses where to measure."""
        truth = np.array([[0, 0], [3.0, -2.0]], dtype=float)
        stack = _shifted_stack(truth, h=96, w=112)
        model = solve_translation(stack, device="numpy", upsample=8,
                                  reference="first", roi=(30, 30, 40, 48))
        from spyde.drift import shift_frame
        aligned = shift_frame(stack[1], model.shifts[1], fill=0.0)
        core = (slice(40, -40), slice(40, -40))     # far OUTSIDE the ROI
        resid = np.abs(aligned[core] - stack[0][core]).max()
        raw = np.abs(stack[1][core] - stack[0][core]).max()
        assert resid < 0.25 * raw, (
            "correcting with an ROI-derived shift did not align the region "
            "outside the ROI")

    def test_roi_is_recorded_in_params(self):
        stack = _shifted_stack(np.zeros((2, 2)), h=64, w=64)
        m = solve_translation(stack, device="numpy", roi=(8, 8, 32, 32))
        assert m.params["roi"] == [8, 8, 32, 32]
        assert m.params["frame_shape"] == [64, 64], (
            "frame_shape must stay the FULL frame — the shifts apply to it")

    def test_out_of_bounds_roi_raises_rather_than_clamping(self):
        """A silently shrunk ROI would correlate on a region the user never chose."""
        stack = _shifted_stack(np.zeros((2, 2)), h=64, w=64)
        for bad in [(0, 0, 80, 32), (40, 40, 32, 32), (-4, 0, 32, 32)]:
            with pytest.raises(ValueError, match="outside"):
                solve_translation(stack, device="numpy", roi=bad)

    def test_tiny_roi_raises(self):
        stack = _shifted_stack(np.zeros((2, 2)), h=64, w=64)
        with pytest.raises(ValueError, match="at least"):
            solve_translation(stack, device="numpy", roi=(0, 0, 8, 8))

    def test_malformed_roi_raises(self):
        stack = _shifted_stack(np.zeros((2, 2)), h=64, w=64)
        with pytest.raises(ValueError, match=r"\(y0, x0, h, w\)"):
            solve_translation(stack, device="numpy", roi=(1, 2, 3))

    def test_roi_ignores_motion_outside_it(self):
        """The point of the feature, on data built to punish whole-frame."""
        rng = np.random.default_rng(3)
        h, w, n = 96, 128, 5
        base = _scene(h, w, noise=0.0)
        frames = []
        for t in range(n):
            f = base.copy()
            # A big bright square that moves the OTHER way, far from the ROI.
            y, x = 60, 78 + 5 * t
            f[y:y + 26, x:x + 26] += 3.0
            frames.append(f.astype(np.float32))
        stack = np.stack(frames)
        roi = solve_translation(stack, device="numpy", upsample=8,
                                reference="first", roi=(4, 4, 44, 56))
        full = solve_translation(stack, device="numpy", upsample=8,
                                 reference="first")
        # The landmark region is STATIC, so the ROI answer should be ~zero.
        assert np.abs(roi.shifts).max() < 0.6, (
            f"ROI solve drifted although its region never moved: {roi.shifts}")
        assert np.abs(full.shifts).max() > np.abs(roi.shifts).max(), (
            "the whole-frame solve was not contaminated by the moving square, so "
            "this fixture no longer demonstrates why the ROI exists")


class TestBackendParity:
    """The GPU path has no independent reference, so it is pinned to numpy."""

    def test_torch_matches_numpy(self):
        torch = pytest.importorskip("torch")
        truth = np.array([[0, 0], [2.5, -1.75], [-4.25, 3.5]], dtype=float)
        stack = _shifted_stack(truth)
        ref = solve_translation(stack, device="numpy", upsample=8)
        got = solve_translation(stack, device="cpu", upsample=8)
        assert got.params["backend"] == "torch"
        assert np.allclose(got.shifts, ref.shifts, atol=1e-2), (
            f"torch={got.shifts}\nnumpy={ref.shifts}")


class TestWarp:
    def test_integer_shift_is_exact_and_preserves_dtype(self):
        f = np.arange(24, dtype=np.uint16).reshape(4, 6)
        out = shift_frame(f, (1, 2), fill=0, preserve_dtype=True)
        assert out.dtype == np.uint16
        # The interior must be bit-identical — no resampling on a whole-pixel move.
        # Destination [1:, 2:] is fed by source [:-1, :-2].
        assert np.array_equal(out[1:, 2:], f[:-1, :-2])
        assert np.all(out[0, :] == 0) and np.all(out[:, :2] == 0)

    def test_nan_padding_marks_uncovered(self):
        f = np.ones((5, 5), dtype=np.float32)
        out = shift_frame(f, (2, 0))
        assert np.isnan(out[:2]).all()
        assert np.allclose(out[2:], 1.0)

    def test_subpixel_shift_interpolates_and_pads(self):
        f = _scene(32, 32).astype(np.float32)
        out = shift_frame(f, (0.5, -0.5))
        assert out.dtype == np.float32
        assert np.isnan(out[0]).all(), "top row needs off-frame data"
        assert np.isnan(out[:, -1]).all(), "right column needs off-frame data"
        assert np.isfinite(out[3:-3, 3:-3]).all()

    def test_nan_does_not_bleed_into_real_data(self):
        """Interpolating with NaN cval would smear it `order` px inward."""
        f = np.ones((16, 16), dtype=np.float32)
        out = shift_frame(f, (2.5, 0.0))
        assert np.isfinite(out[4:, :]).all(), "NaN bled past the padded border"
        assert np.allclose(out[5:-1, :], 1.0, atol=1e-5)

    def test_preserve_dtype_rejects_subpixel(self):
        f = np.zeros((4, 4), dtype=np.uint16)
        with pytest.raises(ValueError, match="whole-pixel"):
            shift_frame(f, (0.5, 0), fill=0, preserve_dtype=True)

    def test_round_trip_is_sign_symmetric(self):
        """Shifting out and back must land where it started.

        This pins the SIGN symmetry, not interpolation fidelity — so the scene is
        noise-free (see :func:`_scene`) and cubic interpolation is used. Two
        bilinear passes over pixel-scale noise would lose ~1% of amplitude for
        entirely legitimate reasons and tell us nothing about the sign.
        """
        f = _scene(64, 64, noise=0.0).astype(np.float32)
        moved = shift_frame(f, (3.25, -2.5), fill=0.0, order=3)
        back = shift_frame(moved, (-3.25, 2.5), fill=0.0, order=3)
        core = (slice(10, -10), slice(10, -10))
        err = np.abs(back[core] - f[core]).max()
        assert err < 0.01, f"round trip lost {err:.4f} — sign asymmetry?"

    def test_round_trip_beats_the_uncorrected_offset(self):
        """Sanity: the corrected result is far closer than the shifted one."""
        f = _scene(64, 64, noise=0.0).astype(np.float32)
        moved = shift_frame(f, (3.25, -2.5), fill=0.0)
        back = shift_frame(moved, (-3.25, 2.5), fill=0.0)
        core = (slice(10, -10), slice(10, -10))
        assert np.abs(back[core] - f[core]).max() < \
            0.1 * np.abs(moved[core] - f[core]).max()

    def test_rejects_non_finite_shift(self):
        with pytest.raises(ValueError, match="finite"):
            shift_frame(np.zeros((4, 4)), (np.nan, 0))

    def test_coverage_matches_finite_pixels(self):
        f = np.ones((20, 20), dtype=np.float32)
        for s in [(0, 0), (3, -2), (2.5, 1.25), (-4.75, 6.5)]:
            out = shift_frame(f, s)
            cov = coverage_mask((20, 20), s)
            assert np.array_equal(np.isfinite(out), cov), f"mismatch at shift {s}"


class TestDriftModel:
    def test_shape_validation(self):
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            DriftModel(shifts=np.zeros((4, 3)))

    def test_residual_length_validation(self):
        with pytest.raises(ValueError, match="residuals must be"):
            DriftModel(shifts=np.zeros((4, 2)), residuals=np.zeros(3))

    def test_is_integer(self):
        assert DriftModel(shifts=np.array([[0, 0], [2, -3]], float)).is_integer
        assert not DriftModel(shifts=np.array([[0, 0], [2.5, 0]], float)).is_integer

    def test_max_abs_shift_ignores_nan(self):
        m = DriftModel(shifts=np.array([[0, 0], [3, -7], [np.nan, np.nan]], float))
        assert m.max_abs_shift == 7.0

    def test_frame_conversions_are_inverses(self):
        m = DriftModel(shifts=np.array([[0, 0], [2.5, -1.5], [4, 3]], float))
        pos = np.array([[10.0, 12.0], [20.0, 22.0]])
        idx = np.array([1, 2])
        assert np.allclose(m.to_lab_frame(m.to_sample_frame(pos, idx), idx), pos)

    def test_to_sample_frame_removes_stage_motion(self):
        """A particle that only *appears* to move because the stage drifted."""
        m = DriftModel(shifts=np.array([[0, 0], [-5, 0], [-10, 0]], float))
        # Same physical spot, drifting downward in the raw frames.
        lab = np.array([[30.0, 40.0], [35.0, 40.0], [40.0, 40.0]])
        idx = np.array([0, 1, 2])
        sample = m.to_sample_frame(lab, idx)
        assert np.allclose(sample[:, 0], 30.0), sample

    def test_save_load_round_trip(self, tmp_path):
        m = DriftModel(
            shifts=np.array([[0, 0], [1.25, -2.5]], float),
            residuals=np.array([np.inf, 12.5], np.float32),
            params={"upsample": 8}, provenance={"action": "drift"},
            reference="running",
        )
        p = str(tmp_path / "d.npz")
        m.save(p)
        back = DriftModel.load(p)
        assert np.array_equal(back.shifts, m.shifts)
        assert back.params["upsample"] == 8
        assert back.provenance == {"action": "drift"}
        assert back.reference == "running"
        assert np.array_equal(back.residuals, m.residuals)

    def test_load_rejects_future_format(self, tmp_path):
        import json
        p = str(tmp_path / "bad.npz")
        np.savez_compressed(
            p, shifts=np.zeros((2, 2), np.float32),
            meta=np.array(json.dumps({"format_version": 999})))
        with pytest.raises(ValueError, match="unsupported DriftModel format"):
            DriftModel.load(p)


class TestEndToEnd:
    def test_solve_then_warp_aligns_the_stack(self):
        """The whole point: after correction, every frame agrees with frame 0."""
        truth = np.array(
            [[0, 0], [2.5, -3.0], [-4.25, 1.75], [6.0, 4.5]], dtype=float)
        stack = _shifted_stack(truth, h=80, w=80)
        model = solve_translation(stack, device="numpy", upsample=8,
                                  reference="first")

        core = (slice(12, -12), slice(12, -12))
        ref = stack[0][core]
        for i in range(1, len(truth)):
            aligned = shift_frame(stack[i], model.shifts[i], fill=0.0)
            resid = np.abs(aligned[core] - ref).max()
            raw = np.abs(stack[i][core] - ref).max()
            assert resid < raw * 0.2, (
                f"frame {i}: correction barely helped (resid={resid:.4f} "
                f"raw={raw:.4f}) — check the SIGN convention")

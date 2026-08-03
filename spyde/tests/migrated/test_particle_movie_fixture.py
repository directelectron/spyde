"""
The synthetic particle-movie fixture, and what it is the acceptance gate for.

``spyde.data.synthetic.particle_movie`` is Step 0 of DRIFT_AND_PARTICLES_PLAN.md:
every later step is checked against a number from this fixture rather than against
a golden file or a screenshot. So this file has two jobs:

1. **Pin the fixture itself.** If its ground truth drifts out of step with the
   pixels it generates, every downstream gate silently becomes meaningless.
2. **Run the end-to-end gates it exists to serve** — drift recovery and
   segmentation sensitivity — since a fixture nothing consumes is not verified.

The regression in :class:`TestTaperTrap` is the reason this file exists at all:
the fixture found a defect in the drift solver on its first run.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.data.synthetic import (
    DISSOLUTION_INDEX,
    MERGE_PAIR,
    NUCLEATION_INDEX,
    ground_truth,
    particle_movie,
    particle_truth_at,
)
from spyde.drift import solve_translation
from spyde.particles import SegmentParams, measure_frame
from spyde.signals.particles import COL
from spyde.tests.migrated._labels import labels_from


@pytest.fixture(scope="module")
def movie():
    """One build shared across the module — it takes ~1 s and is deterministic."""
    s = particle_movie()
    return s, ground_truth(s)


class TestFixtureShape:
    def test_is_an_insitu_signal(self, movie):
        s, _ = movie
        assert type(s).__name__ == "InSitu", (
            "the movie must cast to the insitu signal type or the Play / "
            "Fast-Forward toolbar gating never applies to it")

    def test_dimensions_and_dtype(self, movie):
        s, gt = movie
        assert s.data.shape == (gt["n_frames"], *tuple(gt["frame_shape"]))
        assert s.data.dtype == np.float32
        assert s.axes_manager.navigation_dimension == 1
        assert s.axes_manager.signal_dimension == 2

    def test_frames_are_non_square(self, movie):
        """A square fixture hides a transposed frame."""
        _, gt = movie
        ny, nx = tuple(gt["frame_shape"])
        assert ny != nx

    def test_time_axis_is_calibrated(self, movie):
        s, _ = movie
        tax = s.axes_manager.navigation_axes[0]
        assert tax.name == "time" and tax.units == "s"
        assert tax.scale == pytest.approx(0.05), (
            "real-time playback reads the time-axis scale; scale=1 makes the "
            "movie crawl at 1 fps")

    def test_signal_axes_are_calibrated(self, movie):
        s, gt = movie
        for ax in s.axes_manager.signal_axes:
            assert ax.units == "nm"
            assert ax.scale == pytest.approx(gt["scale"])

    def test_deterministic(self):
        a = particle_movie(n_frames=6)
        b = particle_movie(n_frames=6)
        assert np.array_equal(a.data, b.data)

    def test_seed_changes_only_the_noise_and_film(self):
        a = particle_movie(n_frames=6, seed=0)
        b = particle_movie(n_frames=6, seed=1)
        assert not np.array_equal(a.data, b.data)
        # The particles are analytic, so the ground truth is seed-independent.
        assert np.array_equal(ground_truth(a)["p_y0"], ground_truth(b)["p_y0"])


class TestAsymmetry:
    """Every one of these would pass on symmetric data while hiding a real bug."""

    def test_frame_differs_from_its_mirrors_and_transpose(self, movie):
        s, _ = movie
        f = s.data[0]
        assert not np.allclose(f, f[::-1]), "vertically symmetric — a flip would hide"
        assert not np.allclose(f, f[:, ::-1]), "horizontally symmetric"
        # Non-square, so a transpose cannot even be compared — that is the point.
        assert f.shape[0] != f.shape[1]

    def test_frames_differ_from_each_other(self, movie):
        """A stale-frame bug must be visible."""
        s, gt = movie
        for t in range(1, int(gt["n_frames"])):
            assert not np.allclose(s.data[t], s.data[t - 1]), f"frame {t} == {t-1}"

    def test_drift_axes_have_different_shapes(self, movie):
        """A swapped axis must show as a wrong-shaped curve, not a wrong number."""
        _, gt = movie
        dy, dx = np.asarray(gt["drift"]).T
        assert np.all(np.diff(dy) >= -1e-9), "dy should be monotonic"
        assert dx.min() < -0.5 and dx.max() > 0.5, "dx should swing both ways"


class TestGroundTruth:
    def test_drift_starts_at_zero(self, movie):
        _, gt = movie
        assert np.allclose(np.asarray(gt["drift"])[0], 0.0)

    def test_event_frames_are_in_range(self, movie):
        _, gt = movie
        n = int(gt["n_frames"])
        for key in ("nucleation_frame", "dissolution_frame", "merge_frame"):
            assert 0 < int(gt[key]) < n, f"{key}={gt[key]} outside 0..{n}"

    def test_counts_match_the_event_timeline(self, movie):
        """The count trace must step up at nucleation and down at dissolution."""
        _, gt = movie
        n = int(gt["n_frames"])
        counts = np.array([particle_truth_at(gt, t)[2].sum() for t in range(n)])
        nuc, dis = int(gt["nucleation_frame"]), int(gt["dissolution_frame"])
        assert counts[nuc] == counts[nuc - 1] + 1, "no step up at nucleation"
        assert counts[dis] == counts[dis - 1] - 1, "no step down at dissolution"

    def test_nucleating_particle_is_absent_then_present(self, movie):
        _, gt = movie
        i, nuc = NUCLEATION_INDEX, int(gt["nucleation_frame"])
        assert not particle_truth_at(gt, nuc - 1)[2][i]
        assert particle_truth_at(gt, nuc)[2][i]

    def test_dissolving_particle_is_present_then_absent(self, movie):
        _, gt = movie
        i, dis = DISSOLUTION_INDEX, int(gt["dissolution_frame"])
        assert particle_truth_at(gt, dis - 1)[2][i]
        assert not particle_truth_at(gt, dis)[2][i]

    def test_merge_pair_converges_and_overlaps_at_the_stamped_frame(self, movie):
        _, gt = movie
        a, b = tuple(gt["merge_pair"])
        radii = np.asarray(gt["p_radius"])
        touch = radii[a] + radii[b]
        mf = int(gt["merge_frame"])

        def gap(t):
            pos = particle_truth_at(gt, t)[0]
            return float(np.hypot(*(pos[a] - pos[b])))

        assert gap(0) > touch, "the merge pair already overlaps at t=0"
        assert gap(mf) <= touch, f"no overlap at the stamped merge_frame {mf}"
        assert gap(mf - 1) > touch, f"they already overlapped before frame {mf}"

    def test_faint_probes_are_faint_but_findable(self, movie):
        """The Section 0.9 gate: above noise, well below the bright particles."""
        _, gt = movie
        faint = np.asarray(gt["p_faint"], bool)
        amps = np.asarray(gt["p_amp"])
        noise = float(gt["noise"])
        assert faint.sum() == 2
        assert amps[faint].max() < 0.25 * amps[~faint].min(), "not actually faint"
        assert amps[faint].min() / noise > 4.0, "buried in noise, unfindable"

    def test_ground_truth_raises_on_a_plain_signal(self):
        import hyperspy.api as hs
        with pytest.raises(ValueError, match="no synthetic ground truth"):
            ground_truth(hs.signals.Signal2D(np.zeros((4, 4))))


class TestTruthMatchesPixels:
    """The fixture's stamped truth must describe the pixels it actually drew."""

    def test_every_present_particle_is_brighter_than_its_surroundings(self, movie):
        s, gt = movie
        for t in (0, 12, int(gt["n_frames"]) - 1):
            f = s.data[t]
            pos, radii, present = particle_truth_at(gt, t)
            faint = np.asarray(gt["p_faint"], bool)
            for i in np.flatnonzero(present):
                if faint[i]:
                    continue                     # covered by the sensitivity test
                cy, cx = int(round(pos[i, 0])), int(round(pos[i, 1]))
                if not (2 <= cy < f.shape[0] - 2 and 2 <= cx < f.shape[1] - 2):
                    continue                     # drifted out of frame
                centre = f[cy - 1:cy + 2, cx - 1:cx + 2].mean()
                assert centre > np.median(f) + 0.2, (
                    f"frame {t} particle {i} at ({cy},{cx}) is not bright")

    def test_absent_particles_leave_nothing_behind(self, movie):
        """A dissolved particle must actually be gone from the pixels."""
        s, gt = movie
        i = DISSOLUTION_INDEX
        dis = int(gt["dissolution_frame"])
        before, after = s.data[dis - 1], s.data[dis]
        pos = particle_truth_at(gt, dis)[0]
        cy, cx = int(round(pos[i, 0])), int(round(pos[i, 1]))
        w = 3
        assert (before[cy - w:cy + w, cx - w:cx + w].mean()
                > after[cy - w:cy + w, cx - w:cx + w].mean() + 0.3), (
            "the dissolving particle is still in the pixels after its death frame")


class TestDriftRecoveryGate:
    """Plan gate A1, run against the fixture rather than a hand-made stack."""

    def test_recovers_the_applied_drift(self, movie):
        s, gt = movie
        truth = np.asarray(gt["drift"])
        model = solve_translation(s.data, device="numpy", upsample=8,
                                  reference="first", max_shift=20)
        err = np.abs(model.shifts - truth)
        assert err.max() < 0.5, f"max drift error {err.max():.3f} px\n{model.shifts}"

    def test_running_reference_also_works(self, movie):
        s, gt = movie
        truth = np.asarray(gt["drift"])
        model = solve_translation(s.data, device="numpy", upsample=8, max_shift=20)
        assert np.abs(model.shifts - truth).max() < 0.5

    def test_correction_flattens_the_drift(self, movie):
        """End to end: solve, apply, and the film should stop moving."""
        from spyde.drift import shift_frame
        s, gt = movie
        model = solve_translation(s.data, device="numpy", upsample=8,
                                  reference="first", max_shift=20)
        core = (slice(20, -20), slice(20, -20))
        ref = s.data[0][core]
        last = int(gt["n_frames"]) - 1
        raw = float(np.abs(s.data[last][core] - ref).mean())
        fixed = shift_frame(s.data[last], model.shifts[last], fill=0.0)
        corrected = float(np.abs(fixed[core] - ref).mean())
        assert corrected < raw, (
            f"correction made it worse (raw {raw:.4f} -> {corrected:.4f}) — "
            "check the SIGN convention")


class TestTaperTrap:
    """A full Hann window destroys this registration. Do not re-enable it.

    This is a real defect the fixture caught on its first run: with ``apodize=1.0``
    the solve returns a 25 px error on a 6 px drift — worse than not correcting at
    all. It is NOT specific to our implementation; ``skimage``'s
    ``phase_cross_correlation`` returns the same wrong answer on the same windowed
    input, because a full-frame window reweights the two frames' content
    differently once the drift is large and manufactures a false peak.
    """

    def test_default_taper_is_a_partial_edge_taper(self):
        from spyde.drift.translation import DEFAULT_TAPER_ALPHA
        assert 0.0 < DEFAULT_TAPER_ALPHA < 0.6, (
            "the default must taper only the EDGE; alpha near 1.0 is the trap "
            "this class documents")

    def test_full_hann_is_dramatically_worse_than_the_default(self, movie):
        s, gt = movie
        truth = np.asarray(gt["drift"])
        good = solve_translation(s.data, device="numpy", upsample=8,
                                 reference="first", max_shift=20)
        hann = solve_translation(s.data, device="numpy", upsample=8,
                                 reference="first", max_shift=20, apodize=1.0)
        e_good = np.abs(good.shifts - truth).max()
        e_hann = np.abs(hann.shifts - truth).max()
        assert e_good < 0.5, f"the default taper regressed: {e_good:.3f} px"
        assert e_hann > 5.0 * e_good, (
            "full Hann is no longer catastrophic here — if the solver changed so "
            "that it is safe, this test has served its purpose and can go, but "
            f"check deliberately (default {e_good:.3f} px vs hann {e_hann:.3f} px)")

    def test_apodize_records_the_alpha_actually_used(self, movie):
        s, _ = movie
        assert solve_translation(s.data[:3], device="numpy",
                                 apodize=False).params["apodize"] == 0.0
        assert solve_translation(s.data[:3], device="numpy",
                                 apodize=0.3).params["apodize"] == pytest.approx(0.3)


class TestHarnessLoader:
    """``load_test_data_particles`` — the door the e2e specs come through."""

    def test_action_is_registered(self):
        from spyde.backend._session_actions import _TEST_ACTIONS
        assert "load_test_data_particles" in _TEST_ACTIONS, (
            "the action is not in _TEST_ACTIONS, so dispatch will reject it and "
            "every e2e spec silently loads nothing")

    def test_loads_lazily_one_frame_per_chunk(self, window):
        session = window["window"]
        session._load_test_data_particles({"frames": 6})
        assert len(window["signal_trees"]) == 1
        root = window["signal_trees"][0].root
        assert root._lazy, "must be lazy — an eager fixture skips the cache path"
        assert root.data.chunksize[0] == 1, (
            f"expected one frame per chunk, got {root.data.chunksize} — that is "
            "what makes each nav move a small cold read like a real .mrc")

    def test_signal_type_and_axes_survive_the_lazy_rewrap(self, window):
        session = window["window"]
        session._load_test_data_particles({"frames": 6})
        root = window["signal_trees"][0].root
        assert getattr(root, "_signal_type", None) == "insitu", (
            "the insitu cast was lost, so Play / Fast-Forward will not appear")
        tax = root.axes_manager.navigation_axes[0]
        assert tax.name == "time" and tax.scale == pytest.approx(0.05)
        assert root.axes_manager.signal_axes[0].units == "nm"

    def test_ground_truth_survives_the_lazy_rewrap(self, window):
        """The whole point of the fixture is its stamped truth — losing it in the
        re-wrap would leave every downstream gate asserting against nothing."""
        session = window["window"]
        session._load_test_data_particles({"frames": 6})
        gt = ground_truth(window["signal_trees"][0].root)
        assert gt["kind"] == "particle_movie"
        assert int(gt["nucleation_frame"]) == NUCLEATION_INDEX + 5  # 8
        assert np.asarray(gt["drift"]).shape == (6, 2)

    def test_eager_option(self, window):
        session = window["window"]
        session._load_test_data_particles({"frames": 4, "eager": True})
        assert not window["signal_trees"][0].root._lazy

    def test_opens_two_windows(self, window):
        """A 1-D-nav movie gives a navigator plus a signal window."""
        session = window["window"]
        session._load_test_data_particles({"frames": 6})
        assert len(session._plots) >= 2, (
            f"expected navigator + signal plots, got {len(session._plots)}")


class TestSegmentationOnTheFixture:
    """Plan gate B1/B5 against known radii, plus the Section 0.9 sensitivity gate."""

    def test_finds_every_bright_particle(self, movie):
        s, gt = movie
        t = 12                                   # all nine present
        labels = labels_from(s.data[t], min_size=25, blur=1.0)
        pos, _, present = particle_truth_at(gt, t)
        faint = np.asarray(gt["p_faint"], bool)
        want = np.flatnonzero(present & ~faint)
        found = 0
        for i in want:
            cy, cx = int(round(pos[i, 0])), int(round(pos[i, 1]))
            if labels[cy, cx] != 0:
                found += 1
        assert found == len(want), f"found {found} of {len(want)} bright particles"

    def test_default_sensitivity_misses_the_faint_probes(self, movie):
        """Establishes that the sensitivity gate below is not vacuous."""
        s, gt = movie
        t = 12
        labels = labels_from(s.data[t], min_size=25, blur=1.0)
        pos = particle_truth_at(gt, t)[0]
        faint = np.flatnonzero(np.asarray(gt["p_faint"], bool))
        hit = sum(labels[int(round(pos[i, 0])), int(round(pos[i, 1]))] != 0
                  for i in faint)
        assert hit < len(faint), (
            "the faint probes are already found at default sensitivity — they are "
            "not faint enough to test anything")

    def test_measured_radii_match_the_truth(self, movie):
        s, gt = movie
        t = 12
        labels = labels_from(s.data[t], min_size=25, blur=1.0)
        rows, _ = measure_frame(labels, s.data[t], t=t, scale=1.0)
        pos, radii, present = particle_truth_at(gt, t)
        faint = np.asarray(gt["p_faint"], bool)
        checked = 0
        for i in np.flatnonzero(present & ~faint):
            cy, cx = int(round(pos[i, 0])), int(round(pos[i, 1]))
            lbl = labels[cy, cx]
            if lbl == 0:
                continue
            row = rows[rows[:, COL["label"]] == lbl]
            if not len(row):
                continue
            # Only isolated particles: a merged pair's area is not one disc.
            if i in tuple(gt["merge_pair"]):
                continue
            got_r = float(row[0, COL["equiv_diameter"]]) / 2.0
            assert abs(got_r - radii[i]) < 0.35 * radii[i], (
                f"particle {i}: measured r={got_r:.2f} vs truth {radii[i]:.2f}")
            checked += 1
        assert checked >= 3, f"only checked {checked} particles — test too weak"

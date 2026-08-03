"""
The linker and the event stream — plan steps C1 and C2, gate
"recovers known trajectories, births, deaths and the merge exactly".

Structure follows what the gates actually are:

* :class:`TestTrajectoryGate` / :class:`TestEventGates` / :class:`TestDriftFrame`
  run against ``particle_movie()`` end to end — segment, measure, link — because
  the fixture is the acceptance gate for this wave and a linker validated only on
  hand-written coordinates is validated against nothing that segmentation produces.
* The remaining classes use small hand-built tables, where a two-particle
  ambiguity can be constructed exactly and the assertion is unambiguous.

One measured fact shapes several of these tests and is worth stating up front:
**the frame at which a merge becomes detectable is a property of the segmenter,
not of the linker.** The fixture's two discs geometrically overlap from frame 14
(``ground_truth(sig)["merge_frame"]``), but the segmenter keeps them apart until
frame 18 with watershed splitting on, and fuses them already at frame 12 with it
off. A linker sees detections, not discs; the honest gate is that it reports the
frame the detections became one, and that this frame brackets the geometric truth
as the segmentation changes.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.data.synthetic import (
    DISSOLUTION_INDEX,
    NUCLEATION_INDEX,
    ground_truth,
    particle_movie,
    particle_truth_at,
)
from spyde.drift.model import DriftModel
from spyde.particles import measure_frame
from spyde.particles.track import (
    EVENT_KINDS,
    LinkParams,
    LinkResult,
    ParticleEvent,
    event_counts,
    events_from_records,
    events_to_records,
    frame_indices,
    link,
    sample_frame_positions,
)
from spyde.signals.particles import COL, N_COLUMNS, SpyDEParticles
from spyde.tests.migrated._labels import labels_from

# The gate used for every fixture test. 3.0 nm = 6 px at the fixture's 0.5 nm/px,
# i.e. ~3x the fastest particle's 2.2 px/frame lab-frame step. Deliberately not the
# default 10.0: a test that only passes at a very loose gate would not notice a
# linker that pairs by luck. TestGateWidth covers the default.
FIXTURE_MAX_DIST = 3.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _segment_movie(sig, gt, params=None, scale=None) -> SpyDEParticles:
    """Segment + measure every frame → a real particle table for the linker.

    Deliberately not a synthetic table: the linker's input is whatever a real
    segmentation produces, including its misses (the two faint probes are never
    found by a global threshold) and its merges. *params* is a kwargs dict for
    :func:`~spyde.tests.migrated._labels.labels_from`.
    """
    params = dict(params) if params else dict(min_size=25, blur=1.0)
    scale = float(gt["scale"]) if scale is None else float(scale)
    rows, contours = [], []
    for t in range(int(gt["n_frames"])):
        labels = labels_from(sig.data[t], **params)
        r, c = measure_frame(labels, sig.data[t], t=t, scale=scale)
        rows.append(r)
        contours.append(c)
    return SpyDEParticles.from_frames(
        rows, frame_shape=tuple(gt["frame_shape"]), contours_per_frame=contours,
        scale=scale, units="nm" if scale != 1.0 else "px")


def _truth_of(res: LinkResult, gt, track: int, scale: float) -> tuple[int, float]:
    """``(truth particle index, distance in px)`` nearest a track's first detection.

    Maps a track id back onto the fixture's hand-written particle table so an
    assertion can say "the mover" rather than "track 2", which is an id the linker
    is free to renumber.
    """
    gi = int(res.track_first_index[track])
    t = int(res.frame_index[gi])
    pos = res.positions[gi] / scale
    truth, _radii, present = particle_truth_at(gt, t)
    d = np.hypot(*(truth - pos).T)
    d[~present] = np.inf
    return int(np.argmin(d)), float(d.min())


def _track_for(res: LinkResult, gt, truth_index: int, scale: float) -> int:
    matches = [tid for tid in range(res.n_tracks)
               if _truth_of(res, gt, tid, scale)[0] == truth_index]
    assert matches, f"no track corresponds to fixture particle {truth_index}"
    return matches[0]


def _table(frames, *, scale: float = 1.0, diameter: float = 4.0,
           areas=None) -> SpyDEParticles:
    """A minimal particle table from per-frame ``[(y, x), ...]`` lists.

    Only the columns the linker reads are filled — positions, ``equiv_diameter``
    (the adaptive merge radius) and optionally ``area`` (the similarity term). Every
    other column stays 0, which is exactly the point: the linker must not depend on
    anything a caller might not have measured.
    """
    blocks = []
    for t, pts in enumerate(frames):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        row = np.zeros((len(pts), N_COLUMNS), np.float32)
        row[:, COL["t"]] = t
        row[:, COL["label"]] = np.arange(1, len(pts) + 1)
        row[:, COL["y"]] = pts[:, 0] * scale
        row[:, COL["x"]] = pts[:, 1] * scale
        row[:, COL["equiv_diameter"]] = diameter * scale
        row[:, COL["track_id"]] = -1.0
        if areas is not None:
            row[:, COL["area"]] = np.asarray(areas[t], dtype=np.float32)
        blocks.append(row)
    return SpyDEParticles.from_frames(blocks, frame_shape=(128, 128), scale=scale,
                                      units="nm" if scale != 1.0 else "px")


@pytest.fixture(scope="module")
def movie():
    s = particle_movie()
    return s, ground_truth(s)


@pytest.fixture(scope="module")
def linked(movie):
    """Segment + measure + link the fixture once — shared by the gate classes."""
    s, gt = movie
    particles = _segment_movie(s, gt)
    return s, gt, particles, link(particles, max_dist=FIXTURE_MAX_DIST)


# ── gate 1: trajectories ─────────────────────────────────────────────────────

class TestTrajectoryGate:
    """"Recovers known trajectories exactly" — no fragmentation, sub-pixel error."""

    def test_one_track_per_bright_particle(self, linked):
        """7 tracks: 6 bright particles present at t=0, plus the nucleation.

        The two faint probes are not detected by the classical engine at default
        sensitivity (``test_particle_movie_fixture.py`` pins that), so the linker
        cannot and must not invent tracks for them.
        """
        _s, gt, _p, res = linked
        assert res.n_tracks == 7, (
            f"expected 7 tracks, got {res.n_tracks} — a count above 7 means a "
            f"trajectory fragmented (lengths {np.sort(res.track_lengths())})")

    def test_the_mover_is_one_unbroken_track(self, linked):
        s, gt, _p, res = linked
        tid = _track_for(res, gt, 2, float(gt["scale"]))     # the constant-velocity one
        traj = res.trajectory(tid)
        assert len(traj) == int(gt["n_frames"]), (
            f"the mover's track spans {len(traj)} of {gt['n_frames']} frames")
        assert np.array_equal(traj[:, 0], np.arange(int(gt["n_frames"]))), (
            "the mover's track has a frame gap")

    def test_mover_positions_match_the_truth(self, linked):
        """Sub-pixel: the fixture draws analytic soft discs, so centroids are exact.

        Measured with the classical segmenter: max error 0.175 px, RMS 0.073 px over
        24 frames. The 0.4 px bar is ~2x the measured worst case — tight enough that
        an off-by-one frame (which would show as ~1.3 px on this particle) fails.
        """
        _s, gt, _p, res = linked
        scale = float(gt["scale"])
        tid = _track_for(res, gt, 2, scale)
        err = []
        for frame, y, x in res.trajectory(tid):
            want = particle_truth_at(gt, int(frame))[0][2]
            err.append(np.abs(np.array([y, x]) / scale - want))
        err = np.asarray(err)
        assert err.max() < 0.4, f"max trajectory error {err.max():.3f} px"

    def test_every_bright_particle_gets_its_own_track(self, linked):
        """No two tracks may map onto the same fixture particle."""
        _s, gt, _p, res = linked
        scale = float(gt["scale"])
        owners = [_truth_of(res, gt, tid, scale)[0] for tid in range(res.n_tracks)]
        assert len(set(owners)) == len(owners), f"two tracks share a particle: {owners}"

    def test_track_lengths_match_the_event_timeline(self, linked):
        """Lengths, not just count: 24, 24, 24, 24 persistent + 16 + 18 + 16."""
        _s, gt, _p, res = linked
        assert sorted(res.track_lengths().tolist()) == [16, 16, 18, 24, 24, 24, 24], (
            f"unexpected track lengths {np.sort(res.track_lengths())}")

    def test_trajectory_positions_agree_with_the_buffer(self, linked):
        """A lab-frame link must report exactly the measured centroids."""
        _s, _gt, particles, res = linked
        stored = particles.flat_buffer[:, [COL["y"], COL["x"]]].astype(np.float64)
        assert np.array_equal(res.positions, stored)
        assert res.reference == "lab"


# ── gates 2-4: events ────────────────────────────────────────────────────────

class TestEventGates:
    """Births, deaths and the merge against the fixture's stamped event frames."""

    def test_exactly_one_nucleation_birth_at_the_known_frame(self, linked):
        """Frame-0 detections DO count as births (LinkParams.initial_births).

        So "the nucleation" is the one birth with ``frame > 0``, and the rest sit at
        frame 0. Pinning both halves is what makes the assertion meaningful: a
        linker that fragmented a track would add births at other frames.
        """
        _s, gt, _p, res = linked
        births = res.events_of("birth")
        later = [e for e in births if e.frame > 0]
        assert len(later) == 1, f"expected 1 nucleation, got {[e.frame for e in later]}"
        assert later[0].frame == int(gt["nucleation_frame"]) == 8
        assert all(e.frame == 0 for e in births if e not in later)
        assert len(births) == res.n_tracks, (
            "every track must have exactly one birth — that is what makes the "
            "event stream a complete description of the assignment")

    def test_the_nucleation_birth_is_the_nucleating_particle(self, linked):
        _s, gt, _p, res = linked
        scale = float(gt["scale"])
        birth = res.events_of("birth", exclude_initial=True)[0]
        assert _truth_of(res, gt, birth.tracks[0], scale)[0] == NUCLEATION_INDEX

    def test_exclude_initial_leaves_only_the_nucleation(self, linked):
        _s, gt, _p, res = linked
        got = res.events_of("birth", exclude_initial=True)
        assert [e.frame for e in got] == [int(gt["nucleation_frame"])]

    def test_initial_births_can_be_turned_off(self, linked):
        _s, _gt, particles, _res = linked
        res = link(particles, max_dist=FIXTURE_MAX_DIST, initial_births=False)
        assert [e.frame for e in res.events_of("birth")] == [8]

    def test_exactly_one_death_at_the_known_dissolution_frame(self, linked):
        """The death frame is the first frame the particle is GONE — frame 16 here,
        matching the fixture's ``death`` column, not the last frame it was seen."""
        _s, gt, _p, res = linked
        deaths = res.events_of("death")
        assert len(deaths) == 1, f"expected 1 death, got {[e.frame for e in deaths]}"
        assert deaths[0].frame == int(gt["dissolution_frame"]) == 16

    def test_the_death_is_the_dissolving_particle(self, linked):
        _s, gt, _p, res = linked
        death = res.events_of("death")[0]
        assert _truth_of(res, gt, death.tracks[0], float(gt["scale"]))[0] \
            == DISSOLUTION_INDEX

    def test_a_track_alive_in_the_final_frame_has_no_death(self, linked):
        """The movie ending is not a dissolution."""
        _s, _gt, _p, res = linked
        ended = {e.tracks[0] for e in res.events if e.kind in ("death", "merge")}
        for tid in range(res.n_tracks):
            if int(res.track_last_frame[tid]) == res.n_frames - 1:
                assert tid not in ended, f"track {tid} reaches the last frame yet ends"

    def test_the_merge_is_detected_once_and_involves_the_merge_pair(self, linked):
        _s, gt, _p, res = linked
        scale = float(gt["scale"])
        merges = res.events_of("merge")
        assert len(merges) == 1, f"expected 1 merge, got {[e.frame for e in merges]}"
        involved = {_truth_of(res, gt, t, scale)[0] for t in merges[0].tracks}
        assert involved == set(int(i) for i in gt["merge_pair"]), (
            f"merge involves fixture particles {involved}, expected "
            f"{tuple(gt['merge_pair'])}")

    def test_the_merge_lands_where_the_detections_actually_fuse(self, linked):
        """The robust form of the merge-frame gate.

        The geometric ``merge_frame`` (14) is when the discs overlap; the segmenter
        keeps them as two detections until 18. So the invariant that holds for ANY
        segmentation is that the merge event coincides with the frame the particle
        count drops because two detections became one — which is also the only frame
        a linker could possibly report.
        """
        _s, gt, particles, res = linked
        merge = res.events_of("merge")[0]
        counts = particles.count_series()
        assert counts[merge.frame] == counts[merge.frame - 1] - 1, (
            f"the merge at frame {merge.frame} is not where the count drops "
            f"({counts.astype(int)})")
        assert abs(merge.frame - int(gt["merge_frame"])) <= 4, (
            f"merge reported at {merge.frame}, geometric truth "
            f"{int(gt['merge_frame'])} — measured offset with this segmenter is +4")

    def test_the_merge_frame_follows_the_segmenter_not_the_linker(self, movie):
        """Watershed OFF fuses the pair 6 frames earlier, and the linker follows.

        This is the honest statement of the limit: the same linker on the same movie
        reports frame 18 with splitting on and frame 12 with it off, bracketing the
        geometric truth at 14. Anything that claims to pin the merge to 14 exactly
        is pinning a segmentation parameter, not the linker.
        """
        s, gt = movie
        with_ws = link(_segment_movie(s, gt), max_dist=FIXTURE_MAX_DIST)
        no_ws = link(_segment_movie(
            s, gt, dict(min_size=25, blur=1.0, watershed=False)),
            max_dist=FIXTURE_MAX_DIST)
        f_on = with_ws.events_of("merge")[0].frame
        f_off = no_ws.events_of("merge")[0].frame
        assert f_off < int(gt["merge_frame"]) < f_on, (
            f"expected the geometric merge frame {int(gt['merge_frame'])} to sit "
            f"between the two segmentations ({f_off} and {f_on})")

    def test_no_spurious_splits(self, linked):
        """The fixture contains no fragmentation, so any split is a false positive.

        Specifically: the nucleation at frame 8 appears 28 px from its nearest
        neighbour, well outside that neighbour's ~12 px body, so it must read as a
        birth and not as a split off it.
        """
        _s, _gt, _p, res = linked
        assert res.events_of("split") == []

    def test_every_event_kind_is_known_and_ordered(self, linked):
        _s, _gt, _p, res = linked
        assert all(e.kind in EVENT_KINDS for e in res.events)
        frames = [e.frame for e in res.events]
        assert frames == sorted(frames), "events are not in chronological order"

    def test_event_lane_counts(self, linked):
        """The C2 navigator lane: one trace per kind, spikes at the event frames."""
        _s, gt, _p, res = linked
        lane = res.event_counts()
        assert set(lane) == set(EVENT_KINDS)
        for trace in lane.values():
            assert trace.shape == (int(gt["n_frames"]),)
        assert lane["birth"][8] == 1 and lane["birth"][0] == 6
        assert lane["death"][16] == 1 and lane["death"].sum() == 1
        assert lane["merge"].sum() == 1
        assert lane["split"].sum() == 0

    def test_event_counts_ignores_out_of_range_frames(self):
        lane = event_counts([ParticleEvent(9, "birth", (0,), (0,))], n_frames=4)
        assert lane["birth"].sum() == 0


# ── gate 5: frame of reference ───────────────────────────────────────────────

class TestDriftFrame:
    """Linking on lab and on drift-corrected coordinates, and what each means."""

    def test_lab_frame_anchors_visibly_move(self, linked):
        """The premise of the whole gate: without correction the static particles
        travel with the stage, so a "did it move?" answer read off the lab frame is
        wrong by the drift amplitude."""
        _s, gt, _p, res = linked
        scale = float(gt["scale"])
        for anchor in (0, 1):
            tid = _track_for(res, gt, anchor, scale)
            tr = res.trajectory(tid)[:, 1:] / scale
            excursion = float(np.hypot(*(tr.max(0) - tr.min(0))))
            assert excursion > 5.0, (
                f"anchor {anchor} only moves {excursion:.2f} px in the lab frame — "
                "the fixture's drift is not being drawn")

    def test_drift_corrected_anchors_are_stationary(self, linked):
        """Measured: 9.3 px lab excursion collapses to 0.29-0.36 px corrected."""
        _s, gt, particles, _res = linked
        scale = float(gt["scale"])
        model = DriftModel(shifts=np.asarray(gt["drift"]))
        res = link(particles, max_dist=FIXTURE_MAX_DIST, drift=model)
        assert res.reference == "sample"
        for anchor in (0, 1):
            tid = _track_for(res, gt, anchor, scale)
            tr = res.trajectory(tid)[:, 1:] / scale
            excursion = float(np.hypot(*(tr.max(0) - tr.min(0))))
            assert excursion < 1.0, (
                f"anchor {anchor} still moves {excursion:.2f} px after correction")

    def test_a_solved_drift_model_works_as_well_as_the_truth(self, linked):
        """End to end with A1's own output rather than the stamped truth, since
        that is what the wizard will hand the linker."""
        from spyde.drift import solve_translation
        s, gt, particles, _res = linked
        scale = float(gt["scale"])
        model = solve_translation(s.data, device="numpy", upsample=8,
                                  reference="first", max_shift=20)
        res = link(particles, max_dist=FIXTURE_MAX_DIST, drift=model)
        for anchor in (0, 1):
            tid = _track_for(res, gt, anchor, scale)
            tr = res.trajectory(tid)[:, 1:] / scale
            assert float(np.hypot(*(tr.max(0) - tr.min(0)))) < 1.0

    def test_the_mover_still_moves_after_correction(self, linked):
        """Correction must remove the stage, not the sample. In the sample frame the
        mover's displacement is its own constant velocity, 1.30 px/frame over 23
        frames = 29.9 px."""
        _s, gt, particles, _res = linked
        scale = float(gt["scale"])
        model = DriftModel(shifts=np.asarray(gt["drift"]))
        res = link(particles, max_dist=FIXTURE_MAX_DIST, drift=model)
        tid = _track_for(res, gt, 2, scale)
        tr = res.trajectory(tid)[:, 1:] / scale
        assert float(np.hypot(*(tr[-1] - tr[0]))) == pytest.approx(29.9, abs=1.0)

    def test_both_references_find_the_same_tracks(self, linked):
        """Same track set and same events either way — only the coordinates differ.

        Track *ids* are not compared: the fixture's merge is symmetric, so which of
        the pair survives is a genuine tie that the change of coordinates can flip
        (documented in ``_nearest_continuing``).
        """
        _s, gt, particles, lab = linked
        model = DriftModel(shifts=np.asarray(gt["drift"]))
        sample = link(particles, max_dist=FIXTURE_MAX_DIST, drift=model)
        assert sample.n_tracks == lab.n_tracks
        assert np.array_equal(np.sort(sample.track_lengths()),
                              np.sort(lab.track_lengths()))
        assert [(e.frame, e.kind) for e in sample.events] == \
               [(e.frame, e.kind) for e in lab.events]

    def test_calibration_is_applied_exactly_once(self):
        """The trap this guards: ``DriftModel`` is in PIXELS, centroids are
        calibrated. Adding the shifts straight onto ``y``/``x`` leaves the anchor
        moving at ``1 - scale`` of its drift — plausible-looking and wrong."""
        scale = 0.25
        shift = np.array([[0.0, 0.0], [4.0, -3.0]])
        # A particle that sits still in the SAMPLE frame therefore appears at
        # -shift in the lab frame.
        particles = _table([[(10.0, 20.0)], [(10.0 - 4.0, 20.0 + 3.0)]], scale=scale)
        got = sample_frame_positions(particles, DriftModel(shifts=shift))
        assert got == pytest.approx(np.array([[10.0, 20.0], [10.0, 20.0]]) * scale)

    def test_a_short_drift_model_is_rejected(self):
        particles = _table([[(1.0, 1.0)]] * 4)
        with pytest.raises(ValueError, match="covers 2 frames"):
            sample_frame_positions(particles, DriftModel(shifts=np.zeros((2, 2))))

    def test_frame_indices_come_from_the_row_pointers(self, linked):
        _s, _gt, particles, res = linked
        assert np.array_equal(frame_indices(particles), res.frame_index)
        assert np.array_equal(res.frame_index,
                              particles.column("t").astype(np.int32))


# ── the gate itself ──────────────────────────────────────────────────────────

class TestDistanceGate:
    """An unmatched detection must be genuinely possible, not forced into a pair."""

    def test_a_detection_beyond_the_gate_starts_a_new_track(self):
        p = _table([[(0.0, 0.0)], [(0.0, 50.0)]])
        res = link(p, max_dist=10.0)
        assert res.n_tracks == 2
        assert [(e.frame, e.kind) for e in res.events] == \
               [(0, "birth"), (1, "birth"), (1, "death")]

    def test_the_gate_is_not_relaxed_to_keep_cardinality(self):
        """The failure mode a plain rectangular Hungarian has: with one track and one
        detection it MUST return a pair, so the gate has to be enforced afterwards.
        """
        p = _table([[(0.0, 0.0)], [(0.0, 11.0)]])
        assert link(p, max_dist=10.0).n_tracks == 2
        assert link(p, max_dist=12.0).n_tracks == 1

    def test_the_globally_cheapest_pairing_wins_over_the_greedy_one(self):
        """Nearest-neighbour chaining gets this wrong; an assignment does not.

        Costs (x only): A at 10, B at 13; detections at 5 and 11.

        ======  ====  ====
        \\        d=5   d=11
        ======  ====  ====
        A         5      1
        B         8      2
        ======  ====  ====

        Greedy takes the globally smallest edge A->11 (1) and then has to put B on
        5 (8), total 9. The optimum is A->5 (5) plus B->11 (2), total 7. Every
        per-track ``argmin`` scheme fails this; the assignment is why we can charge
        one track more so the pair costs less.
        """
        p = _table([[(0.0, 10.0), (0.0, 13.0)],
                    [(0.0, 5.0), (0.0, 11.0)]])
        res = link(p, max_dist=10.0)
        assert res.n_tracks == 2
        assert res.trajectory(0)[1, 2] == pytest.approx(5.0), (
            "track A took its own nearest detection — this is a greedy match, not "
            "an assignment")
        assert res.trajectory(1)[1, 2] == pytest.approx(11.0)

    def test_a_near_gate_pair_is_still_linked_rather_than_left_over(self):
        """The other half of the gate contract: unmatched must be *possible*, not
        *preferred*.

        A sits on top of a detection (cost 0) while B's only admissible partner
        costs 10 out of a 10.5 gate. Both links must still be made — a solver whose
        sentinel were too small, or one that stopped at the obvious cheap pair,
        would leave B unmatched and invent a death plus a birth.
        """
        p = _table([[(0.0, 0.0), (0.0, 20.0)], [(0.0, 0.0), (0.0, 10.0)]])
        res = link(p, max_dist=10.5)
        assert res.n_tracks == 2
        assert res.events_of("death") == [] and \
            res.events_of("birth", exclude_initial=True) == []

    def test_a_frame_with_no_detections_does_not_link_across_it(self):
        p = _table([[(0.0, 0.0)], [], [(0.0, 0.0)]])
        res = link(p, max_dist=10.0, memory=0)
        assert res.n_tracks == 2
        assert [(e.frame, e.kind) for e in res.events] == \
               [(0, "birth"), (1, "death"), (2, "birth")]

    def test_max_dist_must_be_positive(self):
        for bad in (0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError, match="max_dist"):
                LinkParams(max_dist=bad)


class TestGateWidth:
    """The measured usable window for ``max_dist`` on the fixture."""

    @pytest.mark.parametrize("max_dist", [2.0, 3.0, 5.0, 10.0, 20.0, 40.0])
    def test_a_wide_range_of_gates_gives_the_same_answer(self, linked, max_dist):
        """Including the 10.0 default. 2.0 nm = 4 px is the lower edge."""
        _s, _gt, particles, _res = linked
        res = link(particles, max_dist=max_dist)
        assert res.n_tracks == 7
        assert len(res.events_of("death")) == 1
        assert len(res.events_of("merge")) == 1

    def test_too_tight_a_gate_fragments_and_is_visible_as_extra_deaths(self, linked):
        """Establishes that the range above is not vacuous — and shows what a badly
        set gate looks like, which is a wall of deaths rather than silence."""
        _s, _gt, particles, _res = linked
        res = link(particles, max_dist=0.6)          # 1.2 px, below the true step
        assert res.n_tracks > 7
        assert len(res.events_of("death")) > 1


# ── memory ───────────────────────────────────────────────────────────────────

class TestMemory:
    """A blinking detection must be ONE track, which is what memory is for."""

    def test_memory_zero_splits_a_blinking_detection(self):
        p = _table([[(0.0, 0.0)], [(0.0, 1.0)], [], [(0.0, 3.0)]])
        res = link(p, max_dist=10.0, memory=0)
        assert res.n_tracks == 2
        assert [(e.frame, e.kind) for e in res.events] == \
               [(0, "birth"), (2, "death"), (3, "birth")]

    def test_memory_one_bridges_a_single_missing_frame(self):
        p = _table([[(0.0, 0.0)], [(0.0, 1.0)], [], [(0.0, 3.0)]])
        res = link(p, max_dist=10.0, memory=1)
        assert res.n_tracks == 1
        assert [e.kind for e in res.events] == ["birth"]

    def test_memory_one_does_not_bridge_two_missing_frames(self):
        p = _table([[(0.0, 0.0)], [], [], [(0.0, 3.0)]])
        res = link(p, max_dist=10.0, memory=1)
        assert res.n_tracks == 2

    def test_memory_two_bridges_two_missing_frames(self):
        p = _table([[(0.0, 0.0)], [], [], [(0.0, 3.0)]])
        assert link(p, max_dist=10.0, memory=2).n_tracks == 1

    def test_the_gate_is_measured_from_the_last_seen_position(self):
        """Memory does not widen the search radius, so a fast particle that blinks
        can still fall outside it. Documented behaviour, pinned here."""
        p = _table([[(0.0, 0.0)], [], [(0.0, 12.0)]])
        assert link(p, max_dist=10.0, memory=1).n_tracks == 2

    def test_a_bridged_track_has_a_frame_gap_not_an_interpolated_row(self):
        p = _table([[(0.0, 0.0)], [], [(0.0, 2.0)]])
        res = link(p, max_dist=10.0, memory=1)
        assert np.array_equal(res.trajectory(0)[:, 0], [0, 2])
        assert res.track_lengths().tolist() == [2]
        assert int(res.track_last_frame[0]) == 2

    def test_memory_on_the_fixture_repairs_a_punched_hole(self, linked):
        """The fixture has no dropouts, so one is introduced: delete the mover's
        detection at frame 10. memory=0 fragments it into two tracks with a spurious
        death; memory=1 recovers the single track and the spurious death vanishes.
        """
        _s, gt, particles, _res = linked
        scale = float(gt["scale"])
        want = particle_truth_at(gt, 10)[0][2]
        rows, contours = [], []
        for t in range(particles.n_frames):
            blk = particles.at(t).copy()
            keep = np.ones(len(blk), bool)
            if t == 10:
                d = np.hypot(*(blk[:, [COL["y"], COL["x"]]] / scale - want).T)
                keep[int(np.argmin(d))] = False
            rows.append(blk[keep])
            contours.append([particles.contour_at(gi)
                             for gi, k in zip(particles.indices_at(t), keep) if k])
        holed = SpyDEParticles.from_frames(
            rows, frame_shape=particles.frame_shape, contours_per_frame=contours,
            scale=scale, units="nm")

        cold = link(holed, max_dist=FIXTURE_MAX_DIST, memory=0)
        warm = link(holed, max_dist=FIXTURE_MAX_DIST, memory=1)
        assert cold.n_tracks == 8 and len(cold.events_of("death")) == 2
        assert warm.n_tracks == 7 and len(warm.events_of("death")) == 1
        assert 10 not in [e.frame for e in warm.events_of("death")]

    def test_memory_must_not_be_negative(self):
        with pytest.raises(ValueError, match="memory"):
            LinkParams(memory=-1)


# ── property similarity ──────────────────────────────────────────────────────

class TestPropertySimilarity:
    """The optional weighting, and the promise that the GATE stays distance-only."""

    def test_off_by_default(self):
        assert LinkParams().property_weight == 0.0

    def test_distance_alone_takes_the_wrong_pair_here(self):
        """The setup: the distance-optimal pairing crosses the areas over.

        A(area 100) at 0 and B(area 10) at 10; detections at 4 (area 10) and 6
        (area 100). Distance prefers A->4, B->6 (total 8) over the area-consistent
        A->6, B->4 (total 12), so this is a case where the property term has to
        change the answer or it is doing nothing.
        """
        p = _table([[(0.0, 0.0), (0.0, 10.0)], [(0.0, 4.0), (0.0, 6.0)]],
                   areas=[[100.0, 10.0], [10.0, 100.0]])
        res = link(p, max_dist=10.0, property_weight=0.0)
        assert res.trajectory(0)[1, 2] == pytest.approx(4.0)

    def test_a_weighted_property_flips_it(self):
        """0.245 is the analytic crossover for this geometry; 0.5 clears it."""
        p = _table([[(0.0, 0.0), (0.0, 10.0)], [(0.0, 4.0), (0.0, 6.0)]],
                   areas=[[100.0, 10.0], [10.0, 100.0]])
        res = link(p, max_dist=10.0, property_weight=0.5, properties=("area",))
        assert res.trajectory(0)[1, 2] == pytest.approx(6.0), (
            "the property term did not change the assignment")
        assert res.n_tracks == 2

    def test_the_gate_stays_on_distance_alone(self):
        """A perfect property match cannot pull a pair inside the gate, and a bad
        one cannot push a pair out of it. Otherwise ``max_dist`` silently means
        "max_dist minus however dissimilar these two look"."""
        far = _table([[(0.0, 0.0)], [(0.0, 30.0)]], areas=[[50.0], [50.0]])
        assert link(far, max_dist=10.0, property_weight=5.0).n_tracks == 2
        near = _table([[(0.0, 0.0)], [(0.0, 9.0)]], areas=[[1000.0], [1.0]])
        assert link(near, max_dist=10.0, property_weight=5.0,
                    properties=("area",)).n_tracks == 1

    def test_a_missing_property_does_not_dilute_the_weight(self):
        """``intensity_mean`` is NaN when nothing measured it. Averaging it in as
        zero would halve a requested weight of 1.0 without saying so."""
        p = _table([[(0.0, 0.0), (0.0, 10.0)], [(0.0, 4.0), (0.0, 6.0)]],
                   areas=[[100.0, 10.0], [10.0, 100.0]])
        p.flat_buffer[:, COL["intensity_mean"]] = np.nan
        only_area = link(p, max_dist=10.0, property_weight=0.5, properties=("area",))
        with_nan = link(p, max_dist=10.0, property_weight=0.5,
                        properties=("area", "intensity_mean"))
        assert np.array_equal(only_area.track_id, with_nan.track_id)
        assert with_nan.trajectory(0)[1, 2] == pytest.approx(6.0)

    def test_the_fixture_is_unaffected_by_the_weighting(self, linked):
        """Positions on real segmented data are unambiguous, which is why the term
        is off by default: it cannot help here and it can only add noise."""
        _s, _gt, particles, res = linked
        for w in (0.25, 1.0, 5.0):
            other = link(particles, max_dist=FIXTURE_MAX_DIST, property_weight=w)
            assert np.array_equal(other.track_id, res.track_id), f"weight {w}"

    def test_unknown_property_is_rejected_at_construction(self):
        with pytest.raises(KeyError, match="not_a_column"):
            LinkParams(properties=("not_a_column",))

    def test_negative_weight_is_rejected(self):
        with pytest.raises(ValueError, match="property_weight"):
            LinkParams(property_weight=-0.1)


# ── merge / split rule ───────────────────────────────────────────────────────

class TestMergeAndSplitRule:
    """The post-pass, on geometries small enough to reason about exactly."""

    def test_two_tracks_onto_one_detection_is_a_merge(self):
        p = _table([[(0.0, 0.0), (0.0, 6.0)],
                    [(0.0, 0.5), (0.0, 5.5)],
                    [(0.0, 3.0)]], diameter=8.0)
        res = link(p, max_dist=10.0)
        merges = res.events_of("merge")
        assert len(merges) == 1 and merges[0].frame == 2
        assert set(merges[0].tracks) == {0, 1}
        assert res.events_of("death") == [], "a merge must replace the death"

    def test_one_track_into_two_detections_is_a_split(self):
        p = _table([[(0.0, 3.0)],
                    [(0.0, 3.0)],
                    [(0.0, 0.5), (0.0, 5.5)]], diameter=8.0)
        res = link(p, max_dist=10.0)
        splits = res.events_of("split")
        assert len(splits) == 1 and splits[0].frame == 2
        assert res.events_of("birth", exclude_initial=True) == [], (
            "a split must replace the birth")

    def test_a_lone_disappearance_far_from_anything_is_a_death(self):
        p = _table([[(0.0, 0.0), (0.0, 60.0)], [(0.0, 0.0)]], diameter=8.0)
        res = link(p, max_dist=10.0)
        assert [e.kind for e in res.events_of("death")] == ["death"]
        assert res.events_of("merge") == []

    def test_a_lone_appearance_far_from_anything_is_a_birth(self):
        p = _table([[(0.0, 0.0)], [(0.0, 0.0), (0.0, 60.0)]], diameter=8.0)
        res = link(p, max_dist=10.0)
        assert len(res.events_of("birth", exclude_initial=True)) == 1
        assert res.events_of("split") == []

    def test_the_adaptive_radius_scales_with_the_particle(self):
        """A disappearance 9 units from a survivor is a merge for a 20-unit body and
        a death for a 4-unit one. A fixed radius cannot be right for both."""
        frames = [[(0.0, 0.0), (0.0, 9.0)], [(0.0, 0.0)]]
        assert link(_table(frames, diameter=20.0), max_dist=10.0).events_of("merge")
        assert link(_table(frames, diameter=4.0), max_dist=10.0).events_of("death")

    def test_merge_dist_can_be_overridden(self):
        frames = [[(0.0, 0.0), (0.0, 9.0)], [(0.0, 0.0)]]
        p = _table(frames, diameter=4.0)
        assert link(p, max_dist=10.0, merge_dist=20.0).events_of("merge")
        assert link(p, max_dist=10.0, merge_dist=2.0).events_of("death")

    def test_split_dist_can_be_overridden(self):
        frames = [[(0.0, 0.0)], [(0.0, 0.0), (0.0, 9.0)]]
        p = _table(frames, diameter=4.0)
        assert link(p, max_dist=10.0, split_dist=20.0).events_of("split")
        assert link(p, max_dist=10.0, split_dist=2.0).events_of(
            "birth", exclude_initial=True)

    def test_a_zero_diameter_does_not_disable_the_post_pass(self):
        """The floor. An unmeasured ``equiv_diameter`` would otherwise give a zero
        radius, so no merge could ever be detected and the failure would be silent.
        """
        p = _table([[(0.0, 0.0), (0.0, 1.0)], [(0.0, 0.5)]], diameter=0.0)
        assert link(p, max_dist=10.0).events_of("merge")

    def test_the_floor_is_in_pixels_not_calibrated_units(self):
        """A 2 px floor must stay 2 px when the axis is calibrated at 0.1 nm/px,
        otherwise the post-pass gets 20x looser on a finer calibration."""
        frames = [[(0.0, 0.0), (0.0, 1.5)], [(0.0, 0.75)]]
        for scale in (0.1, 1.0, 10.0):
            res = link(_table(frames, diameter=0.0, scale=scale), max_dist=100.0)
            assert res.events_of("merge"), f"scale {scale}"

    def test_a_disappearance_next_to_a_NEWCOMER_is_not_a_merge(self):
        """The survivor must have existed the frame before. Without that
        requirement, a death happening to coincide with an unrelated birth nearby
        would be reported as a merge into it."""
        # Frame 3 keeps the far particle and introduces a NEW detection 4 units from
        # where the first one died — inside the 8-unit merge radius but outside the
        # 2-unit link gate, so the first track genuinely ends and the newcomer is
        # genuinely new. The buffer order is reversed at frame 3 so that row order
        # cannot stand in for the "existed before" test.
        p = _table([[(0.0, 0.0), (0.0, 40.0)],
                    [(0.0, 1.0), (0.0, 40.0)],
                    [(0.0, 2.0), (0.0, 40.0)],
                    [(0.0, 40.0), (0.0, 6.0)]], diameter=8.0)
        res = link(p, max_dist=2.0)
        assert [(e.frame, e.kind) for e in res.events_of("death")] == [(3, "death")]
        assert res.events_of("merge") == []
        assert [(e.frame, e.kind)
                for e in res.events_of("birth", exclude_initial=True)] == [(3, "birth")]
        assert res.events_of("split") == [], (
            "the newcomer is 34 units from the only continuing track, so it cannot "
            "be a fragment of it either")

    def test_events_survive_a_long_uneventful_stretch(self):
        """The post-pass skips building its per-frame lookup on frames that no
        candidate event needs, which is most of a long stable movie. A merge is the
        case that would break if the skip were off by one, because it is the only
        event that reads the PREVIOUS frame's map — so nothing happens for seven
        frames and then two detections become one.
        """
        frames = [[(0.0, 0.0), (0.0, 6.0)]] * 8 + [[(0.0, 3.0)]]
        res = link(_table(frames, diameter=8.0), max_dist=10.0)
        merges = res.events_of("merge")
        assert len(merges) == 1 and merges[0].frame == 8
        assert set(merges[0].tracks) == {0, 1}

    def test_a_late_split_survives_the_same_skip(self):
        frames = [[(0.0, 3.0)]] * 8 + [[(0.0, 0.5), (0.0, 5.5)]]
        res = link(_table(frames, diameter=8.0), max_dist=10.0)
        assert [(e.frame, e.kind) for e in res.events_of("split")] == [(8, "split")]

    def test_merge_event_particle_indices_point_at_real_rows(self):
        p = _table([[(0.0, 0.0), (0.0, 6.0)],
                    [(0.0, 0.5), (0.0, 5.5)],
                    [(0.0, 3.0)]], diameter=8.0)
        res = link(p, max_dist=10.0)
        e = res.events_of("merge")[0]
        assert len(e.particles) == len(e.tracks) == 2
        absorbed, survivor = e.particles
        assert int(res.frame_index[absorbed]) == e.frame - 1, (
            "the absorbed track has no row at the merge frame — its last one is the "
            "only row that can identify it")
        assert int(res.frame_index[survivor]) == e.frame
        assert res.track_id[survivor] == e.tracks[1]


# ── determinism, degenerate input, bookkeeping ───────────────────────────────

class TestDeterminism:
    def test_the_same_table_links_identically_twice(self, linked):
        _s, _gt, particles, res = linked
        again = link(particles, max_dist=FIXTURE_MAX_DIST)
        assert np.array_equal(again.track_id, res.track_id)
        assert again.events == res.events

    def test_a_rebuilt_table_links_identically(self, movie):
        """Not just the same object twice — the same pixels through the whole
        pipeline, so a dict- or set-ordering dependency anywhere would show."""
        s, gt = movie
        a = link(_segment_movie(s, gt), max_dist=FIXTURE_MAX_DIST)
        b = link(_segment_movie(s, gt), max_dist=FIXTURE_MAX_DIST)
        assert np.array_equal(a.track_id, b.track_id)
        assert a.events == b.events

    def test_ids_are_contiguous_from_zero(self, linked):
        _s, _gt, _p, res = linked
        assert res.track_id.min() == 0
        assert np.array_equal(np.unique(res.track_id), np.arange(res.n_tracks))

    def test_ids_are_ordered_by_first_appearance(self, linked):
        """Which is why they are stable: the numbering depends on the buffer order,
        not on iteration order anywhere in the loop."""
        _s, _gt, _p, res = linked
        assert np.all(np.diff(res.track_first_frame) >= 0)
        assert np.all(np.diff(res.track_first_index) > 0)

    def test_a_symmetric_tie_is_broken_reproducibly(self):
        """Two tracks exactly equidistant from one detection: whichever survives, it
        must be the same one every run."""
        p = _table([[(0.0, 0.0), (0.0, 4.0)], [(0.0, 2.0)]], diameter=8.0)
        first = link(p, max_dist=10.0)
        for _ in range(5):
            other = link(p, max_dist=10.0)
            assert np.array_equal(other.track_id, first.track_id)
            assert other.events == first.events


class TestDegenerate:
    def test_no_frames_at_all(self):
        res = link(SpyDEParticles.from_frames([], frame_shape=(8, 8)))
        assert res.n_tracks == 0 and res.events == [] and res.n_frames == 0
        assert res.positions.shape == (0, 2)

    def test_every_frame_empty(self):
        p = SpyDEParticles.from_frames([np.zeros((0, N_COLUMNS), np.float32)] * 5,
                                       frame_shape=(8, 8))
        res = link(p)
        assert res.n_tracks == 0 and res.events == []
        assert res.event_counts()["birth"].shape == (5,)

    def test_one_empty_frame_in_the_middle(self):
        p = _table([[(0.0, 0.0)], [], [(0.0, 0.0)]])
        res = link(p, max_dist=5.0)
        assert res.n_tracks == 2
        assert [e.frame for e in res.events_of("death")] == [1]

    def test_a_single_frame_gives_births_and_no_deaths(self):
        """Nothing to link, and the movie ending is not a dissolution — so a
        one-frame table is all births."""
        res = link(_table([[(0.0, 0.0), (5.0, 5.0)]]))
        assert res.n_tracks == 2
        assert [e.kind for e in res.events] == ["birth", "birth"]

    def test_everything_vanishes_mid_movie(self):
        p = _table([[(0.0, 0.0), (0.0, 20.0)],
                    [(0.0, 0.0), (0.0, 20.0)],
                    [], [], []])
        res = link(p, max_dist=5.0)
        assert res.n_tracks == 2
        assert sorted(e.frame for e in res.events_of("death")) == [2, 2]
        assert res.events_of("merge") == []

    def test_a_particle_appearing_only_in_the_last_frame(self):
        p = _table([[(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0), (0.0, 40.0)]])
        res = link(p, max_dist=5.0)
        assert res.n_tracks == 2
        assert [e.frame for e in res.events_of("birth", exclude_initial=True)] == [2]
        assert res.events_of("death") == []

    def test_a_table_that_never_got_masks_still_links(self):
        """``store_masks=False`` is the default for long movies, so the linker must
        not need contours."""
        p = _table([[(0.0, 0.0)], [(0.0, 1.0)]])
        assert p.contours is None
        assert link(p, max_dist=5.0).n_tracks == 1


class TestResultBookkeeping:
    def test_apply_writes_the_track_id_column(self, linked):
        _s, _gt, particles, res = linked
        copy = SpyDEParticles(flat_buffer=particles.flat_buffer.copy(),
                              t_offsets=particles.t_offsets.copy(),
                              frame_shape=particles.frame_shape,
                              scale=particles.scale, units=particles.units)
        assert not copy.has_tracks
        res.apply(copy)
        assert copy.has_tracks
        assert np.array_equal(copy.column("track_id").astype(np.int32), res.track_id)

    def test_link_does_not_mutate_unless_asked(self, linked):
        _s, _gt, particles, _res = linked
        before = particles.column("track_id").copy()
        link(particles, max_dist=FIXTURE_MAX_DIST)
        assert np.array_equal(particles.column("track_id"), before)

    def test_apply_true_writes_through(self):
        p = _table([[(0.0, 0.0)], [(0.0, 1.0)]])
        res = link(p, max_dist=5.0, apply=True)
        assert np.array_equal(p.column("track_id").astype(np.int32), res.track_id)

    def test_apply_rejects_a_mismatched_table(self, linked):
        _s, _gt, _p, res = linked
        with pytest.raises(ValueError, match="different link"):
            res.apply(_table([[(0.0, 0.0)]]))

    def test_track_indices_are_chronological(self, linked):
        _s, _gt, _p, res = linked
        for tid in range(res.n_tracks):
            idx = res.track_indices(tid)
            assert np.all(res.track_id[idx] == tid)
            assert np.all(np.diff(res.frame_index[idx]) > 0)

    def test_track_indices_cover_every_particle_exactly_once(self, linked):
        _s, _gt, _p, res = linked
        seen = np.concatenate([res.track_indices(t) for t in range(res.n_tracks)])
        assert np.array_equal(np.sort(seen), np.arange(res.n_particles))

    def test_track_endpoints_agree_with_the_trajectories(self, linked):
        _s, _gt, _p, res = linked
        for tid in range(res.n_tracks):
            traj = res.trajectory(tid)
            assert traj[0, 0] == res.track_first_frame[tid]
            assert traj[-1, 0] == res.track_last_frame[tid]
            assert np.array_equal(traj[0, 1:], res.positions[res.track_first_index[tid]])
            assert np.array_equal(traj[-1, 1:], res.positions[res.track_last_index[tid]])

    def test_track_at_matches_a_boolean_scan(self, linked):
        """`track_at` uses searchsorted for speed; it must agree with the naive form."""
        _s, _gt, _p, res = linked
        for t in range(res.n_frames):
            assert np.array_equal(res.track_at(t),
                                  res.track_id[res.frame_index == t])

    def test_track_lengths_count_detections_not_span(self):
        p = _table([[(0.0, 0.0)], [], [(0.0, 1.0)]])
        res = link(p, max_dist=5.0, memory=1)
        assert res.track_lengths().tolist() == [2]
        assert int(res.track_last_frame[0]) - int(res.track_first_frame[0]) == 2

    def test_out_of_range_track_raises(self, linked):
        _s, _gt, _p, res = linked
        with pytest.raises(IndexError):
            res.track_indices(res.n_tracks)

    def test_unknown_event_kind_raises(self, linked):
        _s, _gt, _p, res = linked
        with pytest.raises(ValueError, match="unknown event kind"):
            res.events_of("explosion")

    def test_unknown_link_parameter_raises(self, linked):
        _s, _gt, particles, _res = linked
        with pytest.raises(TypeError, match="max_distance"):
            link(particles, max_distance=5.0)

    def test_params_and_kwargs_compose(self, linked):
        _s, _gt, particles, _res = linked
        res = link(particles, LinkParams(max_dist=99.0), max_dist=FIXTURE_MAX_DIST)
        assert res.params["max_dist"] == FIXTURE_MAX_DIST

    def test_params_are_recorded_for_provenance(self, linked):
        _s, _gt, _p, res = linked
        assert res.params["max_dist"] == FIXTURE_MAX_DIST
        assert res.params["memory"] == 0
        assert res.params["merge_dist"] is None

    def test_repr_names_the_reference_and_the_event_counts(self, linked):
        _s, _gt, _p, res = linked
        text = repr(res)
        assert "reference='lab'" in text and "'merge': 1" in text


class TestEventSerialisation:
    def test_round_trips_through_records(self, linked):
        _s, _gt, _p, res = linked
        assert events_from_records(events_to_records(res.events)) == res.events

    def test_records_are_json_safe(self, linked):
        import json
        _s, _gt, _p, res = linked
        text = json.dumps(events_to_records(res.events))
        assert json.loads(text)[0]["kind"] == "birth"

    def test_result_to_dict_is_json_safe(self, linked):
        import json
        _s, _gt, _p, res = linked
        d = json.loads(json.dumps(res.to_dict()))
        assert d["n_tracks"] == 7 and d["reference"] == "lab"
        assert len(d["events"]) == len(res.events)

    def test_events_are_hashable_and_immutable(self, linked):
        """Three surfaces share this record (navigator lane, table, report embed),
        so one of them must not be able to rewrite it."""
        import dataclasses
        _s, _gt, _p, res = linked
        assert len(set(res.events)) == len(res.events)
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.events[0].frame = 3

    def test_params_to_dict_is_json_safe(self):
        import json
        assert json.loads(json.dumps(LinkParams().to_dict()))["memory"] == 0

"""
The particle overlay (plan B9) and the stacked navigator lanes (C2/C3).

Built on a REAL segmentation of the synthetic movie — ``labels_from`` →
``measure_frame`` → ``SpyDEParticles.from_frames`` → ``link`` — rather than a
hand-made container, because three of the claims here are only meaningful
against real geometry: the calibrated→pixel conversion (the fixture is
``scale=0.5`` nm/px, so a missing division is visible), the nearest-centroid hit
test, and the split/merge round trip through ``measure_frame``.

Five things are worth more than the rest:

:class:`TestPixelConversion`
    Centroids are CALIBRATED and must be divided by ``particles.scale``;
    contours are already PIXELS and must not be. Both are right at scale 1 and
    wrong everywhere else, which is exactly how ``masks._signal_k_grids``'s bug
    class survives review.
:class:`TestTrails`
    **A dead track draws no head dot.** The dot means "the particle is HERE
    NOW"; on a track that has died — or one inside its ``memory`` gap — it reads
    as a real particle the segmenter stopped filling (plan C3).
:class:`TestEdits`
    Delete / merge / split mutate the store IN PLACE (the lazy label movie closes
    over that object) and are recorded on the tree AND in provenance.
:class:`TestNavigatorLanes`
    The count lane is integer data and is emitted as STEP data — a straight
    interpolation puts a nucleation at 7 when the event is at 8.
:class:`TestLifecycle`
    The overlay lives on the tree and ``BaseSignalTree.close()`` reaps it.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import spyde.data.synthetic as sy
from spyde.actions import particle_overlay as po
from spyde.actions.particle_tree import open_particle_tree
from spyde.particles import (
    LinkParams,
    link,
    measure_frame,
)
from spyde.signals.particles import COL, N_COLUMNS, SpyDEParticles
from spyde.tests.migrated._labels import labels_from

N_FRAMES = 8


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def built():
    """A real segmentation + link of the fixture movie (copied from
    ``test_particle_tree.py``'s ``built`` — same door, same numbers)."""
    s = sy.particle_movie(n_frames=N_FRAMES)
    gt = sy.ground_truth(s)
    scale = float(gt["scale"])
    per_frame, contours = [], []
    for t in range(N_FRAMES):
        lab = labels_from(s.data[t], min_size=25, blur=1.0)
        rows, cs = measure_frame(lab, s.data[t], t=t, scale=scale)
        per_frame.append(rows)
        contours.append(cs)
    parts = SpyDEParticles.from_frames(
        per_frame, frame_shape=tuple(gt["frame_shape"]),
        contours_per_frame=contours, scale=scale, units="nm")
    res = link(parts, LinkParams(max_dist=10.0))
    res.apply(parts)
    return s, gt, parts, res


def _fresh(built):
    """A private copy of the built store — the edit tests mutate it in place."""
    _s, _gt, parts, _res = built
    return SpyDEParticles(
        parts.flat_buffer.copy(), parts.t_offsets.copy(), parts.frame_shape,
        contours=parts.contours.copy(), contour_offsets=parts.contour_offsets.copy(),
        scale=parts.scale, units=parts.units)


@pytest.fixture
def make_tree(window, built):
    """Build particle trees and CLOSE them on teardown.

    Not tidiness: the label movie is lazy, so a live tree runs a progressive
    navigator compute on a daemon thread. Leaving one running past the end of the
    test lets the interpreter finalise underneath it, which prints a truncated
    traceback that looks like a failure and is not. ``close()`` sets the tree's
    ``_nav_stop`` and is re-entrancy guarded, so a test that closes its own tree
    is unaffected.
    """
    session = window["window"]
    s, _gt, parts, res = built
    made = []

    def build(*, events=True, **kwargs):
        tree = open_particle_tree(session, particles=parts, source_node=s,
                                  events=(res.events if events else None), **kwargs)
        made.append(tree)
        assert _wait(lambda: po._first_signal_plot(tree) is not None)
        assert _wait(lambda: po._first_nav_plot(tree) is not None)
        return tree

    yield build
    for tree in made:
        try:
            tree.close()
        except Exception:
            pass


def _detached(particles, **kwargs) -> po.ParticleOverlay:
    """An overlay with no plot: every payload/selection/edit method works without
    a figure, which is what makes the geometry assertable headlessly."""
    return po.ParticleOverlay(None, particles, **kwargs)


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _csr_is_consistent(particles) -> None:
    """Every invariant ``SpyDEParticles.__post_init__`` would have enforced."""
    assert particles.flat_buffer.shape[1] == N_COLUMNS
    assert int(particles.t_offsets[0]) == 0
    assert int(particles.t_offsets[-1]) == len(particles.flat_buffer)
    assert np.all(np.diff(particles.t_offsets) >= 0)
    if particles.contours is not None:
        assert particles.contour_offsets.size == len(particles.flat_buffer) + 1
        assert int(particles.contour_offsets[-1]) == len(particles.contours)
    # The ``t`` column and the CSR block a row sits in must still agree.
    frames = np.repeat(np.arange(particles.n_frames), np.diff(particles.t_offsets))
    assert np.array_equal(frames, particles.flat_buffer[:, COL["t"]].astype(np.int64))


# ── colour ───────────────────────────────────────────────────────────────────

class TestColorCycle:
    def test_track_zero_is_blue(self):
        assert po.track_color(0) == "#89b4fa"
        assert po.TRACK_COLORS[0] == "#89b4fa"

    def test_cycle_is_stable_and_wraps_at_six(self):
        assert len(po.TRACK_COLORS) == 6
        for tid in range(24):
            assert po.track_color(tid) == po.TRACK_COLORS[tid % 6]
        # Stable means a pure function of the id — same answer every call, and
        # the same answer for ids six apart.
        assert po.track_color(3) == po.track_color(9) == po.track_color(15)

    def test_the_six_accents_are_distinct(self):
        assert len(set(po.TRACK_COLORS)) == 6

    def test_untracked_is_grey_not_a_seventh_accent(self):
        assert po.track_color(-1) == po.UNTRACKED_COLOR
        assert po.UNTRACKED_COLOR not in po.TRACK_COLORS

    def test_fade_appends_an_alpha_byte(self):
        assert po.fade("#89b4fa", 1.0) == "#89b4faff"
        assert po.fade("#89b4fa", 0.0) == "#89b4fa00"
        assert len(po.fade("#89b4fa", 0.5)) == 9

    def test_trail_alphas_run_newest_brightest(self):
        alphas = po.trail_alphas(4)
        assert alphas[0] == 1.0
        assert alphas == sorted(alphas, reverse=True)
        assert min(alphas) > 0.0, "the oldest trail step must still be visible"

    def test_real_tracks_land_on_distinct_colours(self, built):
        _s, _gt, parts, res = built
        assert res.n_tracks == 6
        colors = {po.track_color(t) for t in range(res.n_tracks)}
        assert colors == set(po.TRACK_COLORS)


# ── the calibrated → pixel conversion ────────────────────────────────────────

class TestPixelConversion:
    def test_fixture_scale_is_not_one(self, built):
        """The whole point of testing on this fixture."""
        _s, _gt, parts, _res = built
        assert parts.scale == 0.5

    def test_centroids_are_divided_by_scale(self, built):
        _s, _gt, parts, _res = built
        rows = parts.at(0)
        px = po.centroids_px(rows, parts.scale)
        assert np.allclose(px[:, 0], rows[:, COL["x"]] / 0.5)
        assert np.allclose(px[:, 1], rows[:, COL["y"]] / 0.5)
        # And they land INSIDE the frame in pixel space (a missing division would
        # put every marker in the top-left quadrant of a 96x112 frame).
        h, w = parts.frame_shape
        assert px[:, 0].max() < w and px[:, 1].max() < h

    def test_marker_offsets_are_x_then_y(self, built):
        """A property row is (y, x); a marker offset is (x, y)."""
        _s, _gt, parts, _res = built
        row = parts.at(0)[:1]
        px = po.centroids_px(row, parts.scale)
        assert px[0, 0] == pytest.approx(row[0, COL["x"]] / 0.5)
        assert px[0, 1] == pytest.approx(row[0, COL["y"]] / 0.5)
        assert px[0, 0] != pytest.approx(px[0, 1])

    def test_contours_are_NOT_divided(self, built):
        """Contours are stored in pixels already — dividing them by scale would
        shrink every outline to a quarter of its body and leave it detached from
        the centroid it belongs to."""
        _s, _gt, parts, _res = built
        gi = int(parts.indices_at(0)[0])
        raw = parts.contour_at(gi)
        poly = po.contour_xy(parts, gi)
        assert np.allclose(poly[:, 0], raw[:, 1])       # x = column
        assert np.allclose(poly[:, 1], raw[:, 0])       # y = row

    def test_centroid_sits_inside_its_own_outline(self, built):
        """The conversion is only right if the two agree in ONE space."""
        _s, _gt, parts, _res = built
        for gi in parts.indices_at(0):
            poly = po.contour_xy(parts, int(gi))
            if len(poly) < 3:
                continue
            cx, cy = po.centroids_px(parts.flat_buffer[int(gi):int(gi) + 1],
                                     parts.scale)[0]
            assert poly[:, 0].min() - 1 <= cx <= poly[:, 0].max() + 1
            assert poly[:, 1].min() - 1 <= cy <= poly[:, 1].max() + 1

    def test_a_scale_one_store_needs_no_division(self, built):
        """Sanity: at scale=1 the two conventions coincide, which is exactly why
        a scale=1 fixture cannot catch this class of bug."""
        _s, _gt, parts, _res = built
        rows = parts.at(0)
        assert np.allclose(po.centroids_px(rows, 1.0)[:, 0], rows[:, COL["x"]])

    def test_click_is_converted_through_the_PLOT_axes(self, built):
        """A click carries PHYSICAL xdata/ydata; markers are in pixels.

        The conversion goes through the displayed signal's axes, not through
        ``particles.scale`` — the store's calibration and the plot's need not be
        the same number.
        """
        _s, _gt, parts, _res = built

        class _Axis:
            def __init__(self, scale, offset):
                self.scale, self.offset = scale, offset

        class _AxesManager:
            signal_axes = (_Axis(0.25, 3.0), _Axis(0.25, 3.0))

        class _Signal:
            axes_manager = _AxesManager()

        class _State:
            current_signal = _Signal()

        class _Plot:
            plot_state = _State()

        class _Event:
            xdata, ydata = 3.0 + 0.25 * 40, 3.0 + 0.25 * 12

        overlay = po.ParticleOverlay(_Plot(), parts)
        assert overlay._event_px(_Event()) == pytest.approx((40.0, 12.0))

    def test_uncalibrated_plot_is_an_identity(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)

        class _Event:
            xdata, ydata = 17.0, 5.0
        assert overlay._event_px(_Event()) == pytest.approx((17.0, 5.0))


# ── the payload ──────────────────────────────────────────────────────────────

class TestPayload:
    def test_every_particle_is_filled_exactly_once(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.set_frame(3)
        payload = overlay._payload(3)
        drawn = sum(len(payload[f"fill{i}"]["vertices_list"])
                    for i in range(len(po.TRACK_COLORS) + 1))
        assert drawn == len(parts.at(3))

    def test_fill_group_is_the_track_colour_bucket(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        payload = overlay._payload(2)
        for gi in parts.indices_at(2):
            tid = int(parts.flat_buffer[gi, COL["track_id"]])
            bucket = tid % len(po.TRACK_COLORS)
            polys = payload[f"fill{bucket}"]["vertices_list"]
            expected = po.contour_xy(parts, int(gi))
            assert any(p.shape == expected.shape and np.allclose(p, expected)
                       for p in polys), f"particle {gi} is not in bucket {bucket}"

    def test_untracked_rows_go_to_the_grey_bucket(self, built):
        parts = _fresh(built)
        parts.flat_buffer[:, COL["track_id"]] = -1.0
        overlay = _detached(parts)
        payload = overlay._payload(0)
        grey = len(po.TRACK_COLORS)
        assert len(payload[f"fill{grey}"]["vertices_list"]) == len(parts.at(0))
        for i in range(grey):
            assert payload[f"fill{i}"]["vertices_list"] == []

    def test_labels_are_empty_until_something_is_selected(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        assert overlay._payload(0)["labels"]["texts"] == []
        assert overlay._payload(0)["selected"]["vertices_list"] == []

    def test_selection_gets_an_outline_and_a_readout(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        gi = int(parts.indices_at(0)[1])
        overlay.select([gi])
        payload = overlay._payload(0)
        assert len(payload["selected"]["vertices_list"]) == 1
        assert len(payload["labels"]["texts"]) == 1
        text = payload["labels"]["texts"][0]
        assert "track" in text and "area" in text

    def test_hover_labels_without_selecting(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.hovered = int(parts.indices_at(0)[0])
        payload = overlay._payload(0)
        assert len(payload["labels"]["texts"]) == 1
        assert payload["selected"]["vertices_list"] == [], \
            "hover must label, not outline — that is the selection's job"

    def test_a_selection_in_another_frame_is_not_drawn(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.select([int(parts.indices_at(5)[0])])
        payload = overlay._payload(0)
        assert payload["selected"]["vertices_list"] == []
        assert payload["labels"]["texts"] == []

    def test_out_of_range_frame_draws_nothing(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        payload = overlay._payload(999)
        for i in range(len(po.TRACK_COLORS) + 1):
            assert payload[f"fill{i}"]["vertices_list"] == []


# ── trails and the head dot ──────────────────────────────────────────────────

class TestTrails:
    def _heads(self, payload) -> dict[int, int]:
        return {i: len(payload[f"head{i}"]["offsets"])
                for i in range(len(po.TRACK_COLORS) + 1)}

    def _segments(self, payload, bucket) -> int:
        return sum(len(payload[f"trail{bucket}_{s}"]["segments"])
                   for s in range(po.TRAIL_FADE_STEPS))

    def test_trails_off_draws_nothing(self, built):
        _s, _gt, parts, _res = built
        payload = _detached(parts, show_trails=False)._payload(5)
        assert sum(self._heads(payload).values()) == 0
        assert all(self._segments(payload, i) == 0
                   for i in range(len(po.TRACK_COLORS) + 1))

    def test_a_live_track_gets_a_head_dot(self, built):
        _s, _gt, parts, _res = built
        payload = _detached(parts, show_trails=True)._payload(5)
        assert sum(self._heads(payload).values()) == len(parts.at(5))

    def test_head_dot_sits_on_the_centroid_in_pixels(self, built):
        _s, _gt, parts, _res = built
        payload = _detached(parts, show_trails=True)._payload(5)
        gi = int(parts.indices_at(5)[0])
        bucket = int(parts.flat_buffer[gi, COL["track_id"]]) % len(po.TRACK_COLORS)
        expected = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        heads = np.asarray(payload[f"head{bucket}"]["offsets"])
        assert np.allclose(heads[0], expected)

    def test_trail_spans_the_window_and_fades(self, built):
        _s, _gt, parts, _res = built
        payload = _detached(parts, show_trails=True, trail_frames=5)._payload(6)
        gi = int(parts.indices_at(6)[0])
        bucket = int(parts.flat_buffer[gi, COL["track_id"]]) % len(po.TRACK_COLORS)
        # 5 frames of window → at most 4 segments per track.
        assert 1 <= self._segments(payload, bucket) <= 4
        # More than one fade step is populated, i.e. the ramp is actually used.
        used = [s for s in range(po.TRAIL_FADE_STEPS)
                if len(payload[f"trail{bucket}_{s}"]["segments"])]
        assert len(used) >= 2

    def test_a_DEAD_track_draws_no_head_dot(self, built):
        """The plan's C3 note, asserted directly.

        Kill one track after frame 4 (delete its later detections), then look at
        frame 6 — still inside the trailing window, so the fading line is there,
        but the dot must be gone: it would read as a particle that is present
        now, which is precisely what it is not.
        """
        parts = _fresh(built)
        dead = int(parts.at(0)[0, COL["track_id"]])
        doomed = [int(gi) for t in range(5, parts.n_frames)
                  for gi in parts.indices_at(t)
                  if int(parts.flat_buffer[gi, COL["track_id"]]) == dead]
        assert doomed, "the fixture track never reached the later frames"
        po.delete_particles(parts, doomed)

        overlay = _detached(parts, show_trails=True, trail_frames=8)
        payload = overlay._payload(6)
        bucket = dead % len(po.TRACK_COLORS)
        assert self._segments(payload, bucket) > 0, \
            "the dead track's trail should still fade out"
        assert len(payload[f"head{bucket}"]["offsets"]) == 0, \
            "a dead track drew a head dot — it reads as a live particle"
        # Every OTHER track is unaffected.
        live = sum(len(payload[f"head{i}"]["offsets"])
                   for i in range(len(po.TRACK_COLORS) + 1) if i != bucket)
        assert live == len(parts.at(6))

    def test_a_track_inside_its_memory_GAP_draws_no_head_dot(self, built):
        """Same rule, the other half: a track with no detection at *t* has no
        current position, so it gets no dot even though it is not dead."""
        parts = _fresh(built)
        tid = int(parts.at(0)[1, COL["track_id"]])
        gap = [int(gi) for gi in parts.indices_at(4)
               if int(parts.flat_buffer[gi, COL["track_id"]]) == tid]
        assert gap
        po.delete_particles(parts, gap)

        payload = _detached(parts, show_trails=True, trail_frames=8)._payload(4)
        bucket = tid % len(po.TRACK_COLORS)
        assert self._segments(payload, bucket) > 0
        assert len(payload[f"head{bucket}"]["offsets"]) == 0

    def test_untracked_rows_get_no_trail(self, built):
        """An untracked row has no trajectory to draw — inventing one by
        colour-bucket would join unrelated detections into a fake track."""
        parts = _fresh(built)
        parts.flat_buffer[:, COL["track_id"]] = -1.0
        payload = _detached(parts, show_trails=True)._payload(5)
        assert sum(self._heads(payload).values()) == 0


# ── selection ────────────────────────────────────────────────────────────────

class TestSelection:
    def test_select_by_index(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        want = [int(i) for i in parts.indices_at(2)[:2]]
        assert overlay.select(want) == want
        assert overlay.selected == want

    def test_select_drops_out_of_range_indices(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        assert overlay.select([0, 10_000, -5]) == [0]

    def test_select_by_track(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.set_frame(4)
        tid = int(parts.at(4)[2, COL["track_id"]])
        picked = overlay.select_track(tid)
        assert len(picked) == 1
        assert int(parts.flat_buffer[picked[0], COL["track_id"]]) == tid
        assert int(parts.flat_buffer[picked[0], COL["t"]]) == 4

    def test_click_picks_the_nearest_centroid(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.set_frame(1)
        gi = int(parts.indices_at(1)[3])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        assert overlay.pick(cx + 0.5, cy - 0.5) == gi

    def test_click_on_empty_space_selects_nothing(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.set_frame(1)
        assert overlay.pick(1.0, 1.0) is None

    def test_click_handler_selects_through_the_plot_axes(self, built):
        """End-to-end for the click path: a physical xdata/ydata → a selection."""
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.set_frame(1)
        overlay._groups = {"sentinel": object()}    # make the handler non-inert
        gi = int(parts.indices_at(1)[2])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]

        class _Event:      # an uncalibrated plot → xdata IS the pixel column
            xdata, ydata = float(cx), float(cy)
        overlay._on_click(_Event())
        assert overlay.selected == [gi]

    def test_region_selects_in_bulk(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.set_frame(0)
        pts = po.centroids_px(parts.at(0), parts.scale)
        # A box around the left half of the frame.
        picked = overlay.select_region(0, 0, 60, 200)
        expected = int(((pts[:, 0] >= 0) & (pts[:, 0] <= 60)).sum())
        assert len(picked) == expected >= 1
        assert len(picked) < len(parts.at(0)), \
            "the box should not have caught every particle"

    def test_region_is_order_insensitive(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.set_frame(0)
        a = overlay.select_region(10, 10, 80, 90)
        b = overlay.select_region(80, 90, 10, 10)
        assert a == b

    def test_clear_selection(self, built):
        _s, _gt, parts, _res = built
        overlay = _detached(parts)
        overlay.select([0, 1])
        assert overlay.clear_selection() == []

    def test_on_select_callback_fires(self, built):
        _s, _gt, parts, _res = built
        seen = []
        overlay = _detached(parts, on_select=lambda: seen.append(1))
        overlay.select([0])
        assert seen == [1]


# ── editing ──────────────────────────────────────────────────────────────────

class TestEdits:
    def test_delete_removes_the_row_and_keeps_the_CSR_valid(self, built):
        parts = _fresh(built)
        before = parts.n_particles
        gi = int(parts.indices_at(3)[1])
        tid = int(parts.flat_buffer[gi, COL["track_id"]])
        assert po.delete_particles(parts, [gi]) == 1
        assert parts.n_particles == before - 1
        _csr_is_consistent(parts)
        assert tid not in parts.at(3)[:, COL["track_id"]].astype(int)

    def test_delete_mutates_the_store_IN_PLACE(self, built):
        """The lazy label movie closes over this object (particle_tree §0.6), so
        an edit that rebound ``tree.particles`` would leave the open window
        rendering the pre-edit contours forever."""
        parts = _fresh(built)
        buffer_before = parts.flat_buffer
        po.delete_particles(parts, [0])
        assert parts.flat_buffer is not buffer_before, "buffers are rebuilt…"
        # …but the STORE is the same object, which is what the movie holds.
        overlay = _detached(parts)
        assert overlay.particles is parts

    def test_delete_keeps_contours_paired_with_their_rows(self, built):
        parts = _fresh(built)
        survivor = int(parts.indices_at(0)[2])
        expected = parts.contour_at(survivor).copy()
        po.delete_particles(parts, [int(parts.indices_at(0)[0])])
        _csr_is_consistent(parts)
        assert np.array_equal(parts.contour_at(survivor - 1), expected), \
            "a deletion re-paired an outline with the wrong property row"

    def test_delete_through_the_overlay_records_the_edit(self, built):
        parts = _fresh(built)

        class _Tree:
            particles = parts
        tree = _Tree()
        overlay = _detached(parts)
        overlay.tree = tree
        gi = int(parts.indices_at(2)[0])
        overlay.select([gi])
        assert overlay.delete() == 1
        assert len(tree.particle_edits) == 1
        record = tree.particle_edits[0]
        assert record["kind"] == "delete" and record["indices"] == [gi]
        assert parts.provenance["edits"][0]["kind"] == "delete"
        assert overlay.selected == []

    def test_split_makes_two_and_conserves_the_body(self, built):
        parts = _fresh(built)
        gi = int(parts.indices_at(0)[0])
        area = float(parts.flat_buffer[gi, COL["area"]])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        before = parts.n_particles

        a, b = po.split_particle(parts, gi, ((cx, cy - 50), (cx, cy + 50)))
        assert parts.n_particles == before + 1
        _csr_is_consistent(parts)
        halves = parts.flat_buffer[[a, b], COL["area"]]
        assert halves.sum() == pytest.approx(area, rel=0.02)
        assert min(halves) > 0

    def test_split_keeps_the_parent_track_on_the_larger_half(self, built):
        parts = _fresh(built)
        gi = int(parts.indices_at(0)[0])
        parent = int(parts.flat_buffer[gi, COL["track_id"]])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        # An off-centre cut, so the halves are clearly unequal.
        a, b = po.split_particle(parts, gi, ((cx + 2, cy - 50), (cx + 2, cy + 50)))
        tracks = parts.flat_buffer[[a, b], COL["track_id"]].astype(int)
        areas = parts.flat_buffer[[a, b], COL["area"]]
        keeper = int(np.argmax(areas))
        assert tracks[keeper] == parent
        assert tracks[1 - keeper] == -1, \
            "which fragment continues the track is a re-link's answer, not a guess"

    def test_split_pairs_each_new_row_with_its_OWN_outline(self, built):
        """The add path through ``_splice``: a mis-gathered contour block would
        pair a plausible outline with the wrong row, which draws as nonsense
        rather than raising."""
        parts = _fresh(built)
        gi = int(parts.indices_at(0)[0])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        a, b = po.split_particle(parts, gi, ((cx, cy - 50), (cx, cy + 50)))
        for index in (a, b):
            row = parts.flat_buffer[index]
            contour = parts.contour_at(index)
            assert len(contour) >= 3
            # The outline must sit inside its own row's bounding box (+1 px, the
            # tracer runs on a padded crop).
            assert contour[:, 0].min() >= row[COL["bbox_y0"]] - 1
            assert contour[:, 0].max() <= row[COL["bbox_y1"]] + 1
            assert contour[:, 1].min() >= row[COL["bbox_x0"]] - 1
            assert contour[:, 1].max() <= row[COL["bbox_x1"]] + 1
        # And the neighbours' outlines are untouched by the insertion.
        _csr_is_consistent(parts)

    def test_split_that_misses_raises(self, built):
        parts = _fresh(built)
        gi = int(parts.indices_at(0)[0])
        with pytest.raises(ValueError, match="does not divide"):
            po.split_particle(parts, gi, ((0.0, 0.0), (0.0, 10.0)))

    def test_merge_round_trips_a_split(self, built):
        """Split then merge the halves back: one row, the original area."""
        parts = _fresh(built)
        gi = int(parts.indices_at(0)[0])
        area = float(parts.flat_buffer[gi, COL["area"]])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        a, b = po.split_particle(parts, gi, ((cx, cy - 50), (cx, cy + 50)))
        before = parts.n_particles

        merged = po.merge_particles(parts, [a, b])
        assert parts.n_particles == before - 1
        _csr_is_consistent(parts)
        assert float(parts.flat_buffer[merged, COL["area"]]) == pytest.approx(area, rel=0.02)

    def test_merge_keeps_the_largest_bodys_track(self, built):
        parts = _fresh(built)
        gi = int(parts.indices_at(0)[0])
        parent = int(parts.flat_buffer[gi, COL["track_id"]])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        a, b = po.split_particle(parts, gi, ((cx + 2, cy - 50), (cx + 2, cy + 50)))
        merged = po.merge_particles(parts, [a, b])
        assert int(parts.flat_buffer[merged, COL["track_id"]]) == parent

    def test_merging_particles_that_do_not_touch_raises(self, built):
        """A boolean union cast to int32 would measure two distant discs as ONE
        region with a centroid in the empty space between them."""
        parts = _fresh(built)
        a, b = (int(i) for i in parts.indices_at(0)[:2])
        with pytest.raises(ValueError, match="do not touch"):
            po.merge_particles(parts, [a, b])

    def test_merging_across_frames_raises(self, built):
        parts = _fresh(built)
        a = int(parts.indices_at(0)[0])
        b = int(parts.indices_at(1)[0])
        with pytest.raises(ValueError, match="one frame"):
            po.merge_particles(parts, [a, b])

    def test_merge_needs_two(self, built):
        parts = _fresh(built)
        with pytest.raises(ValueError, match="at least two"):
            po.merge_particles(parts, [0])

    def test_edits_are_stamped_into_provenance_for_reproducibility(self, built):
        parts = _fresh(built)

        class _Tree:
            _commit_provenance = {"action": "segment_particles"}
        tree = _Tree()
        overlay = _detached(parts)
        overlay.tree = tree
        gi = int(parts.indices_at(0)[0])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        overlay.select([gi])
        overlay.split(gi, ((cx, cy - 50), (cx, cy + 50)))
        overlay.merge(overlay.selected)

        kinds = [e["kind"] for e in tree.particle_edits]
        assert kinds == ["split", "merge"]
        # Both surfaces carry the log: the tree (so a re-run sees it) and the
        # store's provenance (so a saved file still reproduces the result).
        assert [e["kind"] for e in parts.provenance["edits"]] == kinds
        assert [e["kind"] for e in tree._commit_provenance["edits"]] == kinds

    def test_pending_edits_is_the_re_run_seam(self, built):
        """A re-segmentation rebuilds the store from the raw frames; unless it
        reads this list first, every correction is silently discarded."""
        parts = _fresh(built)

        class _Tree:
            pass
        tree = _Tree()
        assert po.pending_edits(tree) == []
        overlay = _detached(parts)
        overlay.tree = tree
        overlay.select([int(parts.indices_at(0)[0])])
        overlay.delete()
        edits = po.pending_edits(tree)
        assert len(edits) == 1 and edits[0]["revision"] == 1
        # JSON-safe: it crosses the IPC and gets saved with the store.
        import json
        json.dumps(edits)

    def test_edit_bumps_the_revision(self, built):
        parts = _fresh(built)
        overlay = _detached(parts)
        assert overlay.revision == 0
        overlay.select([int(parts.indices_at(0)[0])])
        overlay.delete()
        assert overlay.revision == 1

    def test_frame_provider_fills_the_intensity_columns(self, built):
        source, _gt, _parts, _res = built
        parts = _fresh(built)
        overlay = _detached(parts, frame_provider=lambda t: np.asarray(source.data[t]))
        gi = int(parts.indices_at(0)[0])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        overlay.select([gi])
        a, b = overlay.split(gi, ((cx, cy - 50), (cx, cy + 50)))
        assert np.isfinite(parts.flat_buffer[a, COL["intensity_mean"]])

    def test_without_a_frame_provider_intensity_stays_NaN(self, built):
        """NaN rather than an invented number — the same rule ``measure_frame``
        applies when it runs with no intensity image."""
        parts = _fresh(built)
        overlay = _detached(parts)
        gi = int(parts.indices_at(0)[0])
        cx, cy = po.centroids_px(parts.flat_buffer[gi:gi + 1], parts.scale)[0]
        a, _b = overlay.split(gi, ((cx, cy - 50), (cx, cy + 50)))
        assert not np.isfinite(parts.flat_buffer[a, COL["intensity_mean"]])


# ── navigator lanes ──────────────────────────────────────────────────────────

class TestNavigatorLanes:
    def test_count_lane_is_emitted_as_STEP_data(self):
        """Plan C3: a straight interpolation between frames puts a nucleation's
        transition half a frame early, so 8 reads as 7."""
        counts = np.array([0, 0, 3, 3, 5], np.float32)
        x, y = po.step_trace(counts)
        # Every sample is HELD to the next x before it jumps: consecutive pairs
        # share a y, and the jump happens exactly at the frame boundary.
        assert y.tolist() == [0, 0, 0, 0, 3, 3, 3, 3, 5, 5]
        assert x.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
        rise = int(np.argmax(np.diff(y) > 0)) + 1
        assert x[rise] == 2.0, "the count must rise AT frame 2, not between 1 and 2"

    def test_step_trace_honours_a_calibrated_time_axis(self):
        counts = np.array([1, 2, 3], np.float32)
        x, _y = po.step_trace(counts, np.arange(3) * 0.05)
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(0.15)

    def test_step_trace_is_empty_for_an_empty_lane(self):
        x, y = po.step_trace([])
        assert x.size == 0 and y.size == 0

    def test_a_continuous_lane_is_NOT_stepped(self, built):
        """Mean size is continuous; only the integer lane is a staircase. The
        lane builder plots it straight, so its length is n, not 2n."""
        _s, _gt, parts, _res = built
        size = parts.property_series("area", "mean")
        assert size.shape == (N_FRAMES,)

    def test_event_points_carry_one_row_per_kind(self, built):
        _s, _gt, _parts, res = built
        pts = po._event_points(res.events, "birth", 1.0, 0.0)
        assert len(pts) == len(res.events_of("birth"))
        assert set(pts[:, 1]) == {float(po.EVENT_ROWS["birth"])}
        assert po._event_points(res.events, "death", 1.0, 0.0).shape == (0, 2)

    def test_event_rows_and_colours_cover_every_kind(self):
        from spyde.particles.track import EVENT_KINDS
        assert set(po.EVENT_COLORS) == set(EVENT_KINDS)
        assert set(po.EVENT_ROWS) == set(EVENT_KINDS)
        assert len(set(po.EVENT_ROWS.values())) == len(EVENT_KINDS), \
            "two kinds sharing a row would overplot each other"
        assert len(set(po.EVENT_COLORS.values())) == len(EVENT_KINDS)
        assert po.EVENT_COLORS["birth"] == "#a6e3a1"     # green
        assert po.EVENT_COLORS["death"] == "#f38ba8"     # red
        assert po.EVENT_COLORS["merge"] == "#cba6f7"     # mauve
        assert po.EVENT_COLORS["split"] == "#f9e2af"     # yellow

    def test_event_points_use_the_calibrated_time_axis(self, built):
        _s, _gt, _parts, res = built
        pts = po._event_points(res.events, "birth", 0.05, 0.0)
        frames = [e.frame for e in res.events_of("birth")]
        assert np.allclose(pts[:, 0], np.asarray(frames) * 0.05)

    def test_publish_emits_a_stacked_three_row_navigator(self, window, make_tree):
        session = window["window"]
        messages = window["messages"]
        tree = make_tree()

        messages.clear()
        assert po.publish_navigator_lanes(session, tree) is True
        figures = [m for m in messages if m.get("type") == "figure"
                   and m.get("view_kind") == "stacked"]
        assert figures, "no stacked lane figure emitted"
        figure = figures[-1]
        assert figure["is_navigator"] is True
        assert figure["window_id"] == po._first_nav_plot(tree).window_id
        for lane in (po.LANE_COUNT, po.LANE_SIZE, po.LANE_EVENTS):
            assert lane in figure["title"]

    def test_lanes_reuse_the_shared_stacked_cursor(self, window, make_tree):
        """The reusable half of ``navigator_views``: one logical time cursor
        wired to the tree's REAL 1-D navigation selector, with a line per row."""
        session = window["window"]
        tree = make_tree()
        window_id = po._first_nav_plot(tree).window_id
        assert _wait(lambda: session._nav_selectors.get(window_id) is not None)

        po.publish_navigator_lanes(session, tree)
        cursor = session._stacked_nav_cursors.get(window_id)
        assert cursor is not None
        assert len(cursor.widgets) == 3, "one draggable line per lane"
        assert cursor._index_hook in session._nav_selectors[window_id].index_hooks

    def test_lanes_are_registered_as_named_navigators(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        po.publish_navigator_lanes(session, tree)
        assert {po.LANE_COUNT, po.LANE_SIZE} <= set(tree.navigator_signals)

    def test_republishing_replaces_the_prior_cursor(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        window_id = po._first_nav_plot(tree).window_id
        assert _wait(lambda: session._nav_selectors.get(window_id) is not None)

        po.publish_navigator_lanes(session, tree)
        first = session._stacked_nav_cursors[window_id]
        po.publish_navigator_lanes(session, tree)
        second = session._stacked_nav_cursors[window_id]
        assert second is not first and first._closed is True
        assert first._index_hook not in session._nav_selectors[window_id].index_hooks

    def test_a_tree_without_traces_publishes_nothing(self, window, make_tree):
        session = window["window"]
        tree = make_tree(events=False)
        tree.nav_traces = {}
        assert po.publish_navigator_lanes(session, tree) is False


# ── lifecycle on a real tree ─────────────────────────────────────────────────

class TestLifecycle:
    def test_attach_puts_the_overlay_on_the_tree(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        assert overlay is not None
        assert tree._particle_overlay is overlay
        assert overlay._groups, "no marker groups were created"

    def test_marker_group_count_is_one_per_colour(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        colours = len(po.TRACK_COLORS) + 1
        expected = colours * 2 + colours * po.TRAIL_FADE_STEPS + 2
        assert len(overlay._groups) == expected

    def test_draw_order_is_trails_then_fills_then_heads_then_labels(self, window, make_tree):
        """Creation order IS draw order, and anyplotlib flattens the registry by
        marker TYPE. A head dot underneath its own particle's fill is not a head
        dot, so the type order has to come out lines → polygons → circles → texts.
        """
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        registry = po._first_signal_plot(tree)._plot2d.markers
        assert list(registry) == ["lines", "polygons", "circles", "texts"]
        # Within the polygons type the selected outline is added after the fills.
        polygon_names = list(registry["polygons"].keys())
        assert polygon_names[-1].endswith("_selected")

    def test_attached_overlay_pushes_polygons_for_the_current_frame(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        overlay.set_frame(2)
        drawn = sum(len(overlay._groups[f"fill{i}"]._data.get("vertices_list", []))
                    for i in range(len(po.TRACK_COLORS) + 1))
        assert drawn == len(tree.particles.at(2))

    def test_reattaching_does_not_stack_two_overlays(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        plot = po._first_signal_plot(tree)
        first = po.attach_particle_overlay(plot, tree.particles, tree)
        second = po.attach_particle_overlay(plot, tree.particles, tree)
        assert second is not first
        assert tree._particle_overlay is second
        assert first._groups == {}, "the prior overlay's markers were left behind"

    def test_navigator_hook_is_attached_and_detached(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        assert overlay._selectors, "the overlay found no navigator selector"
        selectors = list(overlay._selectors)
        assert all(overlay._on_indices in sel.index_hooks for sel in selectors)
        overlay.remove()
        assert all(overlay._on_indices not in sel.index_hooks for sel in selectors)

    def test_a_nav_move_redraws_the_new_frame(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        overlay._on_indices(np.array([5]))
        assert overlay._frame == 5
        drawn = sum(len(overlay._groups[f"fill{i}"]._data.get("vertices_list", []))
                    for i in range(len(po.TRACK_COLORS) + 1))
        assert drawn == len(tree.particles.at(5))

    def test_a_superseded_payload_never_lands(self, window, make_tree):
        """Latest-wins: a payload built for an older generation is dropped."""
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        overlay.set_frame(3)
        stale = overlay._payload(0)
        current = overlay._gen
        overlay._apply(stale, current - 1)
        drawn = sum(len(overlay._groups[f"fill{i}"]._data.get("vertices_list", []))
                    for i in range(len(po.TRACK_COLORS) + 1))
        assert drawn == len(tree.particles.at(3))

    def test_teardown_bumps_the_generation_first(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        gen = overlay._gen
        overlay.remove()
        assert overlay._gen > gen

    def test_tree_close_reaps_the_overlay(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        selectors = list(overlay._selectors)
        tree.close()
        assert getattr(tree, "_particle_overlay", None) is None
        assert overlay._groups == {}
        assert all(overlay._on_indices not in sel.index_hooks for sel in selectors)

    def test_region_widget_appears_and_selects_in_bulk(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        plot = po._first_signal_plot(tree)
        overlay = po.attach_particle_overlay(plot, tree.particles, tree)
        assert overlay._region_widget is None

        overlay.set_region_select(True)
        assert overlay._region_widget is not None
        # The default box covers the middle half of the frame, so it catches some
        # particles but not all of them.
        assert 0 < len(overlay.selected) <= len(tree.particles.at(overlay._frame))

        overlay.set_region_select(False)
        assert overlay._region_widget is None
        assert plot._plot2d.list_widgets() == []

    def test_hidden_overlay_still_follows_the_navigator(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        overlay = po.attach_particle_overlay(
            po._first_signal_plot(tree), tree.particles, tree)
        overlay.set_visible(False)
        overlay._on_indices(np.array([4]))
        assert overlay._frame == 4
        drawn = sum(len(overlay._groups[f"fill{i}"]._data.get("vertices_list", []))
                    for i in range(len(po.TRACK_COLORS) + 1))
        assert drawn == 0, "a hidden overlay drew markers"
        overlay.set_visible(True)
        drawn = sum(len(overlay._groups[f"fill{i}"]._data.get("vertices_list", []))
                    for i in range(len(po.TRACK_COLORS) + 1))
        assert drawn == len(tree.particles.at(4))


# ── the actual push seam ─────────────────────────────────────────────────────

class TestMarkersActuallyShip:
    """Every assertion above — ``TestLifecycle`` included — reads
    ``overlay._groups[key]._data``, the builder's OWN copy of what it MEANT to
    draw. ``_push_groups`` (particle_overlay.py) mutates those dicts and then
    calls ``plot2d._push_markers()`` exactly once, wrapped in a bare
    ``except Exception: log.debug(...)`` — so a caller that gets the mutation
    right and drops (or has anyplotlib silently swallow) the push would pass
    every assertion above and draw nothing. This is the exact vacuity class that
    shipped a real overlay bug before (project lore: "assert on bytes that SHIP,
    not what the caller built").

    These tests spy the real seam — ``plot._plot2d._push_markers`` — and read
    back ``plot._plot2d.markers.to_wire_list()``, the wire list that actually
    reaches the renderer. Modelled on
    ``test_particles_wizard.py::TestRasterOverlayAboveThreshold``, which pins the
    same standard for the raster overlay: a REAL ``Plot2D``, not a stub of
    ``set_overlay_mask``/``_push_markers`` that only records what the caller
    handed it.
    """

    @staticmethod
    def _wire(plot) -> dict[str, dict]:
        """The wire list as it last actually SHIPPED.

        Deliberately reads ``plot2d._state["markers"]`` — written ONLY inside
        the real ``_push_markers()`` (``self._state["markers"] =
        self.markers.to_wire_list()``) — rather than calling
        ``plot2d.markers.to_wire_list()`` directly. Calling it directly would
        re-derive the list fresh from the registry's live ``group._data``
        dicts, which ``_push_groups`` mutates UNCONDITIONALLY (its first loop)
        regardless of whether the following ``pusher()`` call ever landed —
        exactly the vacuity this class exists to avoid. A dropped push leaves
        ``_state["markers"]`` stale; it does not leave the registry stale.
        """
        return {w["name"]: w for w in plot._plot2d._state.get("markers") or []}

    @staticmethod
    def _spy(plot, monkeypatch) -> list:
        """Counts real calls to the push seam, still performing the real push."""
        calls: list = []
        real = plot._plot2d._push_markers

        def spy():
            calls.append(1)
            return real()

        monkeypatch.setattr(plot._plot2d, "_push_markers", spy)
        return calls

    def test_set_frame_ships_exactly_one_push(self, window, make_tree, monkeypatch):
        tree = make_tree()
        plot = po._first_signal_plot(tree)
        po.attach_particle_overlay(plot, tree.particles, tree)
        calls = self._spy(plot, monkeypatch)
        tree._particle_overlay.set_frame(5)
        assert calls == [1], f"set_frame shipped {len(calls)} pushes, want 1"

    def test_redraw_ships_exactly_one_push(self, window, make_tree, monkeypatch):
        tree = make_tree()
        plot = po._first_signal_plot(tree)
        overlay = po.attach_particle_overlay(plot, tree.particles, tree)
        calls = self._spy(plot, monkeypatch)
        overlay._redraw()
        assert calls == [1], f"_redraw shipped {len(calls)} pushes, want 1"

    def test_a_nav_move_ships_exactly_one_push(self, window, make_tree, monkeypatch):
        """The path a real drag actually takes: ``_on_indices`` ->
        ``_request_redraw`` -> ``_apply`` -> ``_push_groups`` ->
        ``plot2d._push_markers()``."""
        tree = make_tree()
        plot = po._first_signal_plot(tree)
        overlay = po.attach_particle_overlay(plot, tree.particles, tree)
        calls = self._spy(plot, monkeypatch)
        overlay._on_indices(np.array([6]))
        assert calls == [1], f"a nav move shipped {len(calls)} pushes, want 1"

    def _assert_shipped_fills_match(self, plot, overlay, t: int) -> None:
        """Every shipped fill polygon's COORDINATES match ``_payload(t)``.

        Not a count check: this fixture has the same particle COUNT (6) in
        every frame, so a stale push (an old frame's data still sitting in
        ``_state["markers"]``) would pass a bare ``len(...) == len(...)``
        check by coincidence. The particles drift between frames, so their
        contour coordinates do not coincide — comparing them is what actually
        distinguishes "frame t shipped" from "some other frame shipped".
        """
        wire = self._wire(plot)
        expected = overlay._payload(t)
        n_colors = len(po.TRACK_COLORS) + 1
        for i in range(n_colors):
            shipped = wire[f"particles_fill_{i}"]["vertices_list"]
            want = expected[f"fill{i}"]["vertices_list"]
            assert len(shipped) == len(want), (
                f"bucket {i}: shipped {len(shipped)} polygons, frame {t} has "
                f"{len(want)}")
            for got, exp in zip(shipped, want):
                assert np.allclose(np.asarray(got), np.asarray(exp)), (
                    f"bucket {i}: a shipped polygon's coordinates do not match "
                    f"frame {t} — the wire list is showing a different frame")

    def test_the_shipped_wire_list_carries_the_frames_fills(self, window, make_tree):
        tree = make_tree()
        plot = po._first_signal_plot(tree)
        overlay = po.attach_particle_overlay(plot, tree.particles, tree)
        overlay.set_frame(5)
        self._assert_shipped_fills_match(plot, overlay, 5)

    def test_the_shipped_wire_list_updates_on_a_nav_move(self, window, make_tree):
        tree = make_tree()
        plot = po._first_signal_plot(tree)
        overlay = po.attach_particle_overlay(plot, tree.particles, tree)
        overlay._on_indices(np.array([6]))
        self._assert_shipped_fills_match(plot, overlay, 6)


# ── the staged actions ───────────────────────────────────────────────────────

class TestStagedActions:
    def test_every_part_handler_resolves(self):
        from spyde.actions import registry
        keys = [k for k in registry.STAGED_HANDLERS if k.startswith("part_")]
        assert {"part_open", "part_close", "part_select", "part_delete",
                "part_merge", "part_split", "part_lanes"} <= set(keys)
        for key in keys:
            assert callable(registry.resolve_staged(key))

    def test_the_caret_schema_resolves(self):
        from spyde.actions import registry
        schema = registry.wizard_parameters("part")
        assert schema and schema["trail_frames"]["default"] == po.DEFAULT_TRAIL_FRAMES

    def test_open_attaches_and_close_tears_down(self, window, make_tree):
        session = window["window"]
        tree = make_tree()
        plot = po._first_signal_plot(tree)

        po.part_open(session, plot, {})
        assert getattr(tree, "_particle_overlay", None) is not None
        po.part_close(session, plot, {})
        assert getattr(tree, "_particle_overlay", None) is None

    def test_select_emits_the_selection_for_the_renderer(self, window, make_tree, built):
        session = window["window"]
        messages = window["messages"]
        _s, _gt, parts, _res = built
        tree = make_tree()
        plot = po._first_signal_plot(tree)
        po.part_open(session, plot, {})

        gi = int(parts.indices_at(0)[0])
        messages.clear()
        po.part_select(session, plot, {"indices": [gi]})
        sent = [m for m in messages if m.get("type") == "particle_selection"]
        assert sent, "no particle_selection message emitted"
        payload = sent[-1]
        assert payload["indices"] == [gi]
        record = payload["particles"][0]
        assert record["index"] == gi
        assert record["color"] == po.track_color(record["track_id"])
        assert "area" in record and "circularity" in record
        po.part_close(session, plot, {})

    def test_selection_message_is_valid_JSON_even_with_NaN_properties(self, window, make_tree):
        """``ipc.emit`` dumps with ``allow_nan=True``, so a bare ``NaN`` token
        would make ``JSON.parse`` throw and the renderer lose the whole message —
        and NaN is the normal value for an unmeasured intensity column."""
        import json
        session = window["window"]
        messages = window["messages"]
        tree = make_tree()
        plot = po._first_signal_plot(tree)
        po.part_open(session, plot, {})
        gi = int(tree.particles.indices_at(0)[0])
        buffer = tree.particles.flat_buffer
        saved = float(buffer[gi, COL["intensity_mean"]])
        buffer[gi, COL["intensity_mean"]] = np.nan
        try:
            messages.clear()
            po.part_select(session, plot, {"indices": [gi]})
            sent = [m for m in messages if m.get("type") == "particle_selection"][-1]
            assert sent["particles"][0]["intensity_mean"] is None
            assert "NaN" not in json.dumps(sent)
        finally:
            # The store is module-scoped and shared by every test in this file.
            buffer[gi, COL["intensity_mean"]] = saved
            po.part_close(session, plot, {})

    def test_open_without_particles_errors_rather_than_hanging(self, window):
        """No segmentation running and no particles: say so, don't wait forever.

        ``wait_for_particles`` returns False with no event loop; the harness
        Session HAS one, so the wait starts on a daemon thread and only emits
        the error once it gives up — after its GRACE window (6 s default). An
        assertion made immediately after ``part_open`` returns is testing
        nothing but "an attach hasn't landed yet by coincidence": an overlay
        that attached 500 ms later (a race, or a future grace-window bug) would
        pass undetected. Wait for the actual emitted error instead — the real
        contract is "don't hang AND say why", not "don't attach instantly".
        """
        session = window["window"]
        messages = window["messages"]
        session._load_test_data_particles({"frames": 4})
        assert _wait(lambda: _signal_plot(session) is not None)
        plot = _signal_plot(session)
        messages.clear()
        po.part_open(session, plot, {})
        assert _wait(lambda: any(
            m.get("type") == "error"
            and "needs a segmentation result" in m.get("text", "")
            for m in messages)), "no error emitted after the grace window"
        assert getattr(plot.signal_tree, "_particle_overlay", None) is None

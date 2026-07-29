"""
Tests for spyde.particles (classical engine + measurement) and
spyde.signals.particles (the CSR container).

Acceptance gates from DRIFT_AND_PARTICLES_PLAN.md exercised here:

* B5 measure — matches ``regionprops`` on synthetic shapes of known area and
  eccentricity, and physical units are right under a non-unit axis scale.
* B1 classical — separates touching particles; the shared ``split_instances``
  behaves for a probability map as well as a boolean mask.
* A7 edges — no particle is ever detected in the NaN-padded border a
  drift-corrected frame carries. This is called out in the plan as the single most
  likely integration bug between the two features.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.particles import (
    THRESHOLD_METHODS,
    SegmentParams,
    measure_frame,
    segment_frame,
    split_instances,
    threshold_mask,
)
from spyde.signals.particles import COL, COLUMNS, N_COLUMNS, SpyDEParticles


# ── synthetic scenes ─────────────────────────────────────────────────────────

def _disc(shape, cy, cx, r, amp=1.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r)


def _field(h=120, w=140, radii=((30, 30, 9), (30, 100, 6), (85, 40, 12),
                               (90, 105, 7)), amp=1.0, bg=0.05):
    img = np.full((h, w), bg, dtype=np.float32)
    for cy, cx, r in radii:
        img += _disc((h, w), cy, cx, r, amp).astype(np.float32)
    return img


def _touching(h=80, w=120):
    """Two discs of radius 14 whose edges overlap — the watershed's whole job."""
    img = np.full((h, w), 0.05, dtype=np.float32)
    img += _disc((h, w), 40, 48, 14).astype(np.float32)
    img += _disc((h, w), 40, 72, 14).astype(np.float32)
    return np.clip(img, 0, 1.2)


class TestSegmentParams:
    def test_rejects_unknown_threshold(self):
        with pytest.raises(ValueError, match="unknown threshold"):
            SegmentParams(threshold="magic")

    def test_rejects_sensitivity_out_of_range(self):
        with pytest.raises(ValueError, match="sensitivity must be in 0..1"):
            SegmentParams(sensitivity=1.5)

    def test_rejects_even_local_size_for_local_methods(self):
        """Bumping it silently would make the caret disagree with what ran."""
        with pytest.raises(ValueError, match="local_size must be odd"):
            SegmentParams(threshold="sauvola", local_size=30)
        SegmentParams(threshold="otsu", local_size=30)   # global: irrelevant


class TestThresholding:
    @pytest.mark.parametrize("method", THRESHOLD_METHODS)
    def test_every_method_runs_and_finds_the_discs(self, method):
        # Blurred, so the histogram has real structure. A hard-edged synthetic
        # field is essentially two delta spikes, which several legitimate methods
        # cannot work with (see test_minimum_reports_actionably_on_a_spiky_field).
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(_field(), 2.0)
        p = SegmentParams(threshold=method, local_size=31)
        mask = threshold_mask(img.astype(np.float32), p)
        assert mask.shape == img.shape and mask.dtype == bool
        # All four discs are bright and large; any sane threshold finds signal.
        assert mask.sum() > 100, f"{method} found almost nothing"

    def test_minimum_reports_actionably_on_a_spiky_field(self):
        """skimage raises a bare RuntimeError; the user needs to know what to do."""
        img = _field().astype(np.float32)      # hard edges → one-spike histogram
        with pytest.raises(ValueError, match="requires a clearly bimodal"):
            threshold_mask(img, SegmentParams(threshold="minimum"))

    def test_sensitivity_half_is_exactly_the_plain_method(self):
        """0.5 must be a no-op offset, or the ParticleSpy parity gate is meaningless."""
        from skimage.filters import threshold_otsu
        img = _field().astype(np.float32)
        mask = threshold_mask(img, SegmentParams(threshold="otsu", sensitivity=0.5))
        assert np.array_equal(mask, img > threshold_otsu(img))

    def test_higher_sensitivity_never_shrinks_the_mask(self):
        img = _field(amp=0.4, bg=0.1).astype(np.float32)
        prev = -1
        for s in (0.1, 0.3, 0.5, 0.7, 0.9):
            n = int(threshold_mask(img, SegmentParams(sensitivity=s)).sum())
            assert n >= prev, f"sensitivity {s} shrank the mask ({n} < {prev})"
            prev = n

    def test_invert_finds_dark_particles(self):
        img = 1.0 - _field(amp=1.0, bg=0.05)        # dark discs on bright ground
        labels = segment_frame(img, SegmentParams(invert=True, min_size=30))
        assert labels.max() == 4, f"found {labels.max()} dark particles, expected 4"


class TestSplitInstances:
    def test_separates_touching_discs(self):
        img = _touching()
        labels = segment_frame(img, SegmentParams(watershed=True, min_size=40))
        assert labels.max() == 2, (
            f"watershed merged the pair into {labels.max()} region(s)")

    def test_without_watershed_they_merge(self):
        """Confirms the previous test is actually measuring the watershed."""
        img = _touching()
        labels = segment_frame(img, SegmentParams(watershed=False, min_size=40))
        assert labels.max() == 1

    def test_does_not_oversplit_a_single_round_disc(self):
        """The plateau trap: peak_local_max on a flat distance maximum returns
        several coincident peaks and watershed cuts one disc into wedges."""
        img = np.full((80, 80), 0.05, np.float32)
        img += _disc((80, 80), 40, 40, 20).astype(np.float32)
        labels = segment_frame(img, SegmentParams(watershed=True, min_size=50))
        assert labels.max() == 1, f"one disc split into {labels.max()} pieces"

    def test_accepts_a_probability_map(self):
        """The scribble/prompt engines hand over float probabilities, not masks."""
        prob = np.zeros((60, 60), np.float32)
        prob[10:25, 10:25] = 0.9
        prob[35:50, 35:50] = 0.8
        prob[5:8, 50:53] = 0.3           # below 0.5 — must be ignored
        labels = split_instances(prob, SegmentParams(min_size=20))
        assert labels.max() == 2

    def test_min_size_discards_small(self):
        prob = np.zeros((60, 60), bool)
        prob[10:30, 10:30] = True        # 400 px
        prob[45:48, 45:48] = True        # 9 px
        assert split_instances(prob, SegmentParams(min_size=100)).max() == 1
        assert split_instances(prob, SegmentParams(min_size=5)).max() == 2

    def test_a_tiny_particle_survives_the_watershed(self):
        """§0.9 regression: nothing in the split step may delete a small particle.

        The original marker step filtered markers by AREA (ParticleSpy's
        ``watershed_size``). A 3x3 particle's local-maximum marker is ONE pixel, so
        any area floor erased it and the particle vanished — silently, since the
        frame still had a plausible count.
        """
        prob = np.zeros((60, 60), bool)
        prob[10:30, 10:30] = True        # large
        prob[45:48, 45:48] = True        # tiny, 9 px
        labels = split_instances(prob, SegmentParams(min_size=5, watershed=True))
        assert labels.max() == 2, "the tiny particle was dropped by the split step"
        assert labels[46, 46] != 0

    def test_max_size_discards_large(self):
        prob = np.zeros((60, 60), bool)
        prob[5:55, 5:55] = True          # 2500 px
        assert split_instances(prob, SegmentParams(max_size=1000)).max() == 0

    def test_clear_border_drops_edge_touching(self):
        prob = np.zeros((60, 60), bool)
        prob[0:10, 0:10] = True          # touches the border
        prob[25:40, 25:40] = True
        p = SegmentParams(min_size=20, clear_border=True, watershed=False)
        assert split_instances(prob, p).max() == 1

    def test_labels_are_sequential_with_no_gaps(self):
        prob = np.zeros((80, 80), bool)
        for i, (y, x) in enumerate([(5, 5), (5, 40), (40, 5), (40, 40)]):
            prob[y:y + 12, x:x + 12] = True
        labels = split_instances(prob, SegmentParams(min_size=20, watershed=False))
        present = np.unique(labels)
        assert np.array_equal(present, np.arange(present.size))

    def test_empty_input_gives_empty_labels(self):
        labels = split_instances(np.zeros((20, 20), bool), SegmentParams())
        assert labels.max() == 0 and labels.dtype == np.int32

    def test_rejects_3d(self):
        with pytest.raises(ValueError, match="foreground must be 2-D"):
            split_instances(np.zeros((2, 4, 4), bool), SegmentParams())


class TestNaNBorder:
    """Plan trap #2 / gate A7 — the drift↔segmentation seam."""

    def test_nan_border_yields_no_particles_there(self):
        from spyde.drift import shift_frame
        img = _field()
        shifted = shift_frame(img, (12, -15))       # NaN band top and right
        assert np.isnan(shifted).any(), "test setup produced no NaN border"

        labels = segment_frame(shifted, SegmentParams(min_size=30))
        nan_mask = ~np.isfinite(shifted)
        assert not np.any(labels[nan_mask]), (
            "found a particle inside the NaN-padded border — the padding was "
            "coerced to a value that thresholds as signal")

    def test_nan_does_not_erase_real_data_near_the_border(self):
        """The opposite failure: propagating NaN through the filters wipes a band."""
        from spyde.drift import shift_frame
        img = _field()
        shifted = shift_frame(img, (5, 0))
        labels = segment_frame(shifted, SegmentParams(gaussian=2.0, min_size=30))
        assert labels.max() == 4, (
            f"found {labels.max()} of 4 particles — NaN bled through the blur")


class TestMeasure:
    def test_area_matches_regionprops(self):
        from skimage.measure import regionprops
        prob = np.zeros((80, 80), bool)
        prob[10:30, 10:40] = True         # exactly 600 px
        labels = split_instances(prob, SegmentParams(min_size=10, watershed=False))
        rows, _ = measure_frame(labels)
        assert rows.shape[1] == N_COLUMNS
        ref = regionprops(labels)[0]
        assert rows[0, COL["area"]] == pytest.approx(ref.area)
        assert rows[0, COL["area"]] == pytest.approx(600)

    def test_circularity_of_a_disc_is_near_one(self):
        labels = _disc((120, 120), 60, 60, 30).astype(np.int32)
        rows, _ = measure_frame(labels)
        # A pixelated disc's perimeter is slightly over-estimated, so ~0.9-1.05.
        assert 0.85 < rows[0, COL["circularity"]] < 1.1, rows[0, COL["circularity"]]

    def test_eccentricity_of_a_disc_is_near_zero(self):
        labels = _disc((120, 120), 60, 60, 25).astype(np.int32)
        rows, _ = measure_frame(labels)
        assert rows[0, COL["eccentricity"]] < 0.2

    def test_calibration_scales_length_and_area_correctly(self):
        prob = np.zeros((60, 60), bool)
        prob[10:30, 10:30] = True        # 400 px, 20 px across
        labels = split_instances(prob, SegmentParams(min_size=10, watershed=False))
        r1, _ = measure_frame(labels, scale=1.0)
        r2, _ = measure_frame(labels, scale=0.5)     # 0.5 nm/px
        assert r2[0, COL["area"]] == pytest.approx(r1[0, COL["area"]] * 0.25)
        assert r2[0, COL["perimeter"]] == pytest.approx(r1[0, COL["perimeter"]] * 0.5)
        assert r2[0, COL["y"]] == pytest.approx(r1[0, COL["y"]] * 0.5)

    def test_circularity_is_scale_invariant(self):
        """Dimensionless quantities must NOT pick up a scale factor."""
        labels = _disc((100, 100), 50, 50, 22).astype(np.int32)
        a, _ = measure_frame(labels, scale=1.0)
        b, _ = measure_frame(labels, scale=0.137)
        assert a[0, COL["circularity"]] == pytest.approx(b[0, COL["circularity"]])
        assert a[0, COL["eccentricity"]] == pytest.approx(b[0, COL["eccentricity"]])

    def test_intensity_excludes_nan(self):
        labels = np.zeros((40, 40), np.int32)
        labels[10:20, 10:20] = 1
        inten = np.full((40, 40), 5.0)
        inten[12, 12] = np.nan
        rows, _ = measure_frame(labels, inten)
        assert rows[0, COL["intensity_mean"]] == pytest.approx(5.0), (
            "NaN leaked into the mean, or was coerced to 0 and dragged it down")

    def test_background_ring_ignores_neighbouring_particles(self):
        """A touching neighbour's body must not be measured as this one's background."""
        labels = np.zeros((40, 60), np.int32)
        labels[15:25, 10:20] = 1
        labels[15:25, 21:31] = 2         # 1 px gap — inside a 3 px ring
        inten = np.zeros((40, 60))
        inten[labels == 1] = 10.0
        inten[labels == 2] = 99.0        # a bright neighbour
        inten[labels == 0] = 1.0
        rows, _ = measure_frame(labels, inten, background_ring=3)
        bg = rows[0, COL["background"]]
        assert bg == pytest.approx(1.0), (
            f"background {bg} — the neighbour's 99 leaked in")

    def test_min_area_px_filters_before_returning(self):
        labels = np.zeros((40, 40), np.int32)
        labels[5:25, 5:25] = 1           # 400
        labels[30:33, 30:33] = 2         # 9
        rows, contours = measure_frame(labels, min_area_px=100)
        assert len(rows) == 1 and len(contours) == 1

    def test_contours_match_rows_one_to_one(self):
        labels = np.zeros((60, 60), np.int32)
        labels[5:20, 5:20] = 1
        labels[30:50, 30:55] = 2
        rows, contours = measure_frame(labels)
        assert len(rows) == len(contours) == 2
        for c in contours:
            assert c.ndim == 2 and c.shape[1] == 2 and c.dtype == np.int16

    def test_single_pixel_region_still_gets_a_contour(self):
        """Degenerate regions must not break the 1:1 correspondence."""
        labels = np.zeros((20, 20), np.int32)
        labels[10, 10] = 1
        rows, contours = measure_frame(labels)
        assert len(rows) == 1 and len(contours) == 1 and len(contours[0]) >= 3

    def test_empty_label_image(self):
        rows, contours = measure_frame(np.zeros((20, 20), np.int32))
        assert rows.shape == (0, N_COLUMNS) and contours == []

    def test_track_id_starts_unassigned(self):
        labels = np.zeros((30, 30), np.int32)
        labels[5:20, 5:20] = 1
        rows, _ = measure_frame(labels)
        assert rows[0, COL["track_id"]] == -1

    def test_intensity_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="intensity shape"):
            measure_frame(np.zeros((10, 10), np.int32), np.zeros((8, 8)))


# ── the container ────────────────────────────────────────────────────────────

def _build(n_frames=4, seed=1):
    """A small SpyDEParticles built through the real segment→measure path."""
    rng = np.random.default_rng(seed)
    per_frame, contours = [], []
    for t in range(n_frames):
        img = np.full((90, 90), 0.05, np.float32)
        # A growing number of particles per frame, so count_series is non-trivial.
        for i in range(t + 1):
            cy = 20 + 25 * (i % 3)
            cx = 20 + 25 * (i // 3) + int(rng.integers(0, 3))
            img += _disc((90, 90), cy, cx, 8).astype(np.float32)
        labels = segment_frame(img, SegmentParams(min_size=30))
        rows, cs = measure_frame(labels, img, t=t, scale=0.5)
        per_frame.append(rows)
        contours.append(cs)
    return SpyDEParticles.from_frames(
        per_frame, frame_shape=(90, 90), contours_per_frame=contours,
        scale=0.5, units="nm")


class TestSpyDEParticles:
    def test_builds_and_reports_shape(self):
        p = _build()
        assert p.n_frames == 4
        assert p.n_particles == 1 + 2 + 3 + 4
        assert p.has_masks and not p.has_tracks

    def test_csr_slice_is_o1_and_correct(self):
        p = _build()
        for t in range(p.n_frames):
            blk = p.at(t)
            assert len(blk) == t + 1
            assert np.all(blk[:, COL["t"]] == t)

    def test_count_series_matches_offsets(self):
        p = _build()
        assert np.array_equal(p.count_series(), np.array([1, 2, 3, 4], np.float32))

    def test_property_series_is_nan_on_empty_frames(self):
        """An empty frame has no mean size; zero would draw a fake event spike."""
        rows = np.zeros((2, N_COLUMNS), np.float32)
        rows[:, COL["area"]] = [10.0, 20.0]
        p = SpyDEParticles.from_frames(
            [rows, np.zeros((0, N_COLUMNS), np.float32)], frame_shape=(10, 10))
        s = p.property_series("area", "mean")
        assert s[0] == pytest.approx(15.0)
        assert np.isnan(s[1])

    def test_property_series_reductions(self):
        rows = np.zeros((3, N_COLUMNS), np.float32)
        rows[:, COL["area"]] = [1.0, 2.0, 6.0]
        p = SpyDEParticles.from_frames([rows], frame_shape=(10, 10))
        assert p.property_series("area", "sum")[0] == pytest.approx(9.0)
        assert p.property_series("area", "max")[0] == pytest.approx(6.0)
        assert p.property_series("area", "median")[0] == pytest.approx(2.0)

    def test_unknown_reduce_and_column_raise(self):
        p = _build()
        with pytest.raises(ValueError, match="unknown reduce"):
            p.property_series("area", "bogus")
        with pytest.raises(KeyError, match="unknown column"):
            p.column("nope")

    def test_frame_index_bounds(self):
        p = _build()
        with pytest.raises(IndexError, match="outside"):
            p.at(99)

    def test_render_frame_paints_the_right_count(self):
        p = _build()
        img = p.render_frame(2)
        assert img.shape == (90, 90) and img.dtype == np.int32
        painted = np.unique(img)
        painted = painted[painted > 0]
        assert painted.size == 3, f"painted {painted.size} of 3 particles"

    def test_render_frame_by_index_supports_hit_testing(self):
        p = _build()
        img = p.render_frame(1, value="index")
        vals = np.unique(img)
        vals = vals[vals > 0] - 1
        assert set(vals.tolist()) == set(p.indices_at(1).tolist())

    def test_render_frame_rejects_unknown_value(self):
        p = _build()
        with pytest.raises(ValueError, match="unknown value"):
            p.render_frame(0, value="colour")

    def test_render_without_masks_raises_clearly(self):
        rows = np.zeros((1, N_COLUMNS), np.float32)
        p = SpyDEParticles.from_frames([rows], frame_shape=(10, 10))
        assert not p.has_masks
        with pytest.raises(ValueError, match="store_masks=False"):
            p.render_frame(0)

    def test_mask_at_is_cropped_to_bbox(self):
        p = _build()
        m, (y0, x0, y1, x1) = p.mask_at(0)
        assert m.shape == (y1 - y0, x1 - x0)
        assert m.any(), "empty mask for a real particle"
        assert m.size < 90 * 90, "mask was not cropped"

    def test_validation_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match=r"flat_buffer must be"):
            SpyDEParticles(np.zeros((3, 5)), np.array([0, 3]), (10, 10))
        with pytest.raises(ValueError, match="t_offsets must span"):
            SpyDEParticles(np.zeros((3, N_COLUMNS)), np.array([0, 2]), (10, 10))
        with pytest.raises(ValueError, match="non-decreasing"):
            SpyDEParticles(np.zeros((3, N_COLUMNS)), np.array([0, 3, 1, 3]), (10, 10))

    def test_contours_without_offsets_rejected(self):
        with pytest.raises(ValueError, match="both be set or both None"):
            SpyDEParticles(np.zeros((1, N_COLUMNS)), np.array([0, 1]), (10, 10),
                           contours=np.zeros((4, 2), np.int16))

    def test_from_frames_rejects_mismatched_contours(self):
        rows = np.zeros((2, N_COLUMNS), np.float32)
        with pytest.raises(ValueError, match="outlines must correspond 1:1"):
            SpyDEParticles.from_frames(
                [rows], frame_shape=(10, 10),
                contours_per_frame=[[np.zeros((4, 2), np.int16)]])   # 1 for 2

    def test_from_frames_rejects_frame_count_mismatch(self):
        rows = np.zeros((1, N_COLUMNS), np.float32)
        with pytest.raises(ValueError, match="contours_per_frame has"):
            SpyDEParticles.from_frames([rows, rows], frame_shape=(10, 10),
                                       contours_per_frame=[[]])

    def test_save_load_round_trip(self, tmp_path):
        p = _build()
        path = str(tmp_path / "p.npz")
        p.save(path)
        back = SpyDEParticles.load(path)
        assert np.array_equal(back.flat_buffer, p.flat_buffer)
        assert np.array_equal(back.t_offsets, p.t_offsets)
        assert np.array_equal(back.contours, p.contours)
        assert back.frame_shape == p.frame_shape
        assert back.scale == p.scale and back.units == "nm"
        # And the reloaded object still renders.
        assert back.render_frame(1).max() > 0

    def test_save_load_without_masks(self, tmp_path):
        rows = np.zeros((2, N_COLUMNS), np.float32)
        p = SpyDEParticles.from_frames([rows], frame_shape=(8, 8))
        path = str(tmp_path / "nomask.npz")
        p.save(path)
        back = SpyDEParticles.load(path)
        assert not back.has_masks

    def test_load_rejects_future_format(self, tmp_path):
        import json
        path = str(tmp_path / "bad.npz")
        np.savez_compressed(
            path, flat_buffer=np.zeros((0, N_COLUMNS), np.float32),
            t_offsets=np.array([0]),
            meta=np.array(json.dumps({"format_version": 999,
                                      "columns": list(COLUMNS),
                                      "frame_shape": [4, 4]})))
        with pytest.raises(ValueError, match="unsupported SpyDEParticles format"):
            SpyDEParticles.load(path)

    def test_load_rejects_changed_column_layout(self, tmp_path):
        import json
        path = str(tmp_path / "cols.npz")
        np.savez_compressed(
            path, flat_buffer=np.zeros((0, N_COLUMNS), np.float32),
            t_offsets=np.array([0]),
            meta=np.array(json.dumps({"format_version": 1,
                                      "columns": ["t", "label"],
                                      "frame_shape": [4, 4]})))
        with pytest.raises(ValueError, match="column layout changed"):
            SpyDEParticles.load(path)

    def test_to_csv_writes_a_header_and_every_row(self, tmp_path):
        p = _build()
        path = str(tmp_path / "p.csv")
        p.to_csv(path)
        with open(path) as fh:
            lines = fh.read().strip().splitlines()
        assert lines[0].split(",") == list(COLUMNS)
        assert len(lines) == p.n_particles + 1

    def test_repr_is_informative(self):
        assert "particles over" in repr(_build())

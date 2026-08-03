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
    SegmentParams,
    measure_frame,
    split_instances,
)
from spyde.signals.particles import COL, COLUMNS, N_COLUMNS, SpyDEParticles
from spyde.tests.migrated._labels import labels_from


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
    """What is left to validate once detection is not this dataclass's job.

    It used to guard ``threshold`` / ``sensitivity`` / ``local_size``; all three
    went with the classical engine. The remaining fields are the split and the
    size filter, and they still have values that make no sense.
    """

    def test_rejects_negative_size_bounds(self):
        with pytest.raises(ValueError, match="size bounds must be"):
            SegmentParams(min_size=-1)
        with pytest.raises(ValueError, match="size bounds must be"):
            SegmentParams(max_size=-5)

    def test_rejects_a_sub_pixel_marker_separation(self):
        """0 would ask peak_local_max for markers closer than one pixel."""
        with pytest.raises(ValueError, match="min_separation must be"):
            SegmentParams(min_separation=0)

    def test_the_defaults_are_constructible(self):
        p = SegmentParams()
        assert p.watershed is True and p.min_size == 20

    def test_a_deleted_classical_param_is_refused(self):
        """`particles_action._coerce` drops unknown keys before they get here,
        so this is the backstop rather than the front line — but a stale
        provenance dict must never silently construct a DIFFERENT segmentation."""
        with pytest.raises(TypeError):
            SegmentParams(sensitivity=0.9)


class TestSplitInstances:
    def test_separates_touching_discs(self):
        img = _touching()
        labels = labels_from(img, watershed=True, min_size=40)
        assert labels.max() == 2, (
            f"watershed merged the pair into {labels.max()} region(s)")

    def test_without_watershed_they_merge(self):
        """Confirms the previous test is actually measuring the watershed."""
        img = _touching()
        labels = labels_from(img, watershed=False, min_size=40)
        assert labels.max() == 1

    def test_does_not_oversplit_a_single_round_disc(self):
        """The plateau trap: peak_local_max on a flat distance maximum returns
        several coincident peaks and watershed cuts one disc into wedges."""
        img = np.full((80, 80), 0.05, np.float32)
        img += _disc((80, 80), 40, 40, 20).astype(np.float32)
        labels = labels_from(img, watershed=True, min_size=50)
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


def _pair_mask(h=80, w=120, r=14, cy=40, cx1=48, cx2=72):
    """The two overlapping discs of :func:`_touching`, as a boolean mask."""
    return (_disc((h, w), cy, cx1, r).astype(bool)
            | _disc((h, w), cy, cx2, r).astype(bool))


def _seam(mask, width=2):
    """The join between the two bodies: the geometric stand-in for what a
    trained boundary class predicts, and for the strokes a user paints along it."""
    from scipy import ndimage as ndi
    ws = split_instances(mask, SegmentParams(min_size=20))
    # A boundary is the seam BETWEEN two instances, never the outline of one —
    # a mask of outlines teaches a head to shrink every body and split nothing,
    # which is the failure this helper's shape exists to avoid reproducing.
    grown = [ndi.binary_dilation(ws == i, iterations=width)
             for i in range(1, int(ws.max()) + 1)]
    seam = np.zeros(mask.shape, bool)
    for i in range(len(grown)):
        for j in range(i + 1, len(grown)):
            seam |= grown[i] & grown[j]
    return seam & mask


class TestBoundarySplit:
    """The connected-components route: ``split_instances(fg, p, boundary=...)``.

    This exists for speed — a taught boundary lets the split skip the distance
    transform, the marker/elevation upsample and the watershed, together 1.62 s
    of a 1.78 s split at 4096². So the bar is not "it produces something": it
    has to produce **what the watershed produced**, or the speed is not worth
    having.
    """

    def test_splits_touching_particles(self):
        mask = _pair_mask()
        p = SegmentParams(min_size=40)
        assert split_instances(mask, p, boundary=_seam(mask)).max() == 2

    def test_matches_the_watershed_count_and_areas(self):
        """The gate: same count, the same pixels claimed in total, and the same
        areas to within a fraction of a percent.

        NOT bit-identical, and the difference is inherent rather than a bug: the
        watershed cuts at the point equidistant from two markers, while reclaim
        grows both sides one pixel per pass and breaks a tie toward the higher
        label id. On this pair that moves the cut by four pixels — 0.7% of a 590
        px body — while the union of the two instances is exactly the same set
        of pixels. Asserting bit-equality here would be asserting that two
        different algorithms agree by luck.
        """
        mask = _pair_mask()
        p = SegmentParams(min_size=40)
        ws = split_instances(mask, p)
        bd = split_instances(mask, p, boundary=_seam(mask))

        def areas(lab):
            c = np.bincount(lab.ravel())[1:]
            return np.sort(c[c > 0])

        a_ws, a_bd = areas(ws), areas(bd)
        assert bd.max() == ws.max() == 2
        assert (bd > 0).sum() == (ws > 0).sum(), (
            "the two routes claimed a different amount of foreground")
        assert np.array_equal(bd > 0, ws > 0), (
            "the two routes disagree about which pixels belong to a particle")
        rel = np.abs(a_bd - a_ws) / a_ws
        assert rel.max() < 0.02, (
            f"boundary areas {a_bd.tolist()} differ from watershed "
            f"{a_ws.tolist()} by {rel.max() * 100:.1f}%")

    def test_the_distance_transform_and_watershed_never_run(self, monkeypatch):
        """The whole point. If either is still called the route saves nothing,
        and a passing count test would hide that completely."""
        from scipy import ndimage as ndi
        from skimage import segmentation as skseg

        mask = _pair_mask()
        seam = _seam(mask)                       # built BEFORE the spies go in —
        # `_seam` runs a watershed itself, standing in for the user's eye.
        called = []
        monkeypatch.setattr(ndi, "distance_transform_edt",
                            lambda *a, **k: called.append("edt"))
        monkeypatch.setattr(skseg, "watershed",
                            lambda *a, **k: called.append("watershed"))

        split_instances(mask, SegmentParams(min_size=40), boundary=seam)
        assert called == [], f"the boundary route still ran {called}"

    def test_no_boundary_falls_back_to_the_watershed(self):
        """A user who has never painted a boundary must not silently get worse
        splitting — so an absent boundary is the old behaviour, bit for bit."""
        mask = _pair_mask()
        p = SegmentParams(min_size=40)
        ws = split_instances(mask, p)
        assert np.array_equal(split_instances(mask, p, boundary=None), ws)

    def test_an_all_false_boundary_also_falls_back(self):
        """"The class exists but nothing is painted yet" is the same situation as
        "there is no boundary", and it is the common one mid-session. Treating an
        empty mask as a real boundary would hand watershed's job to plain
        connected components and merge every touching pair."""
        mask = _pair_mask()
        p = SegmentParams(min_size=40)
        ws = split_instances(mask, p)
        empty = split_instances(mask, p, boundary=np.zeros_like(mask))
        assert np.array_equal(empty, ws)

    def test_isolated_particles_are_untouched(self):
        """An isolated body has no seam through it, so its core IS the body and
        the two routes cannot disagree."""
        mask = np.zeros((80, 120), bool)
        mask |= _disc((80, 120), 25, 25, 12).astype(bool)
        mask |= _disc((80, 120), 25, 90, 12).astype(bool)
        p = SegmentParams(min_size=40)
        seam = _seam(mask)
        assert not seam.any(), "these discs do not touch; there is no seam"
        assert np.array_equal(split_instances(mask, p, boundary=seam),
                              split_instances(mask, p))

    def test_a_probability_boundary_is_thresholded_like_the_foreground(self):
        mask = _pair_mask()
        seam = _seam(mask)
        soft = np.where(seam, 0.9, 0.1).astype(np.float32)
        p = SegmentParams(min_size=40)
        assert np.array_equal(split_instances(mask, p, boundary=soft),
                              split_instances(mask, p, boundary=seam))

    def test_a_weak_probability_boundary_is_ignored(self):
        """Below 0.5 everywhere is no boundary at all — and must therefore fall
        back rather than run the fast route on an empty seam."""
        mask = _pair_mask()
        soft = np.where(_seam(mask), 0.3, 0.05).astype(np.float32)
        p = SegmentParams(min_size=40)
        assert np.array_equal(split_instances(mask, p, boundary=soft),
                              split_instances(mask, p))

    def test_a_mismatched_boundary_shape_raises(self):
        with pytest.raises(ValueError, match="must describe the same frame"):
            split_instances(np.zeros((10, 10), bool), SegmentParams(),
                            boundary=np.zeros((10, 12), bool))

    def test_rejects_a_3d_boundary(self):
        with pytest.raises(ValueError, match="boundary must be 2-D"):
            split_instances(np.zeros((10, 10), bool), SegmentParams(),
                            boundary=np.zeros((2, 10, 10), bool))

    def test_a_tiny_particle_survives_the_boundary_route(self):
        """§0.9 again, on the new path: the split step may never delete a small
        body. A 3x3 particle has no seam through it, so it must come out whole."""
        mask = np.zeros((60, 60), bool)
        mask[10:30, 10:30] = True
        mask[45:48, 45:48] = True                    # 9 px
        bnd = np.zeros((60, 60), bool)
        bnd[19:21, 10:30] = True                     # a seam across the big one
        labels = split_instances(mask, SegmentParams(min_size=5), boundary=bnd)
        assert labels[46, 46] != 0, "the tiny particle was dropped"
        assert labels.max() == 3, "the seam should have cut the large body in two"


class TestBoundaryReclaim:
    """Growing the instances back over the seam — what keeps the areas honest."""

    def test_the_seam_is_fully_reclaimed_by_default(self):
        """Default is grow-to-convergence, so every foreground pixel reachable
        from a core ends up owned by one."""
        mask = _pair_mask()
        labels = split_instances(mask, SegmentParams(min_size=40),
                                 boundary=_seam(mask, width=3))
        assert int((labels > 0).sum()) == int(mask.sum()), (
            "some foreground was left unassigned with reclaim running to "
            "convergence")

    def test_capping_the_passes_leaves_part_of_the_seam_unassigned(self):
        """Confirms the previous test measures something: with one pass a wide
        seam cannot be closed, so the areas come out low."""
        mask = _pair_mask()
        seam = _seam(mask, width=3)
        capped = split_instances(
            mask, SegmentParams(min_size=40, boundary_reclaim=1), boundary=seam)
        assert int((capped > 0).sum()) < int(mask.sum())

    def test_the_count_is_the_same_however_many_passes_run(self):
        """Reclaim moves pixels between instances; it must never create or
        destroy one. That is what makes the cap a quality knob and not a
        correctness one."""
        mask = _pair_mask()
        seam = _seam(mask, width=3)
        counts = {
            int(split_instances(mask,
                                SegmentParams(min_size=40, boundary_reclaim=k),
                                boundary=seam).max())
            for k in (1, 2, 3, 5, 0)
        }
        assert counts == {2}, f"the pass count changed the particle count: {counts}"

    def test_foreground_with_no_core_at_all_stays_unassigned(self):
        """A body entirely covered by boundary belongs to no instance, and
        inventing an owner for it would be worse than leaving it out."""
        mask = np.zeros((40, 40), bool)
        mask[5:25, 5:25] = True                      # a real body
        mask[32:35, 32:35] = True                    # fully fenced in below
        bnd = np.zeros((40, 40), bool)
        bnd[32:35, 32:35] = True
        labels = split_instances(mask, SegmentParams(min_size=4), boundary=bnd)
        assert labels.max() == 1
        assert not labels[32:35, 32:35].any()


class TestFinalizeLabels:
    """``_finalize_labels`` fuses the size filter and the sequential relabel."""

    def test_identical_to_the_chain_it_replaces(self):
        """It reads the raster twice instead of six times, so it has to be
        proven equal to the obvious version rather than merely similar."""
        from spyde.particles.instances import (_drop_large, _drop_small,
                                               _finalize_labels,
                                               _relabel_sequential)
        rng = np.random.default_rng(0)
        for trial in range(60):
            lab = rng.integers(0, 12, size=(24, 24)).astype(np.int32)
            if trial % 3 == 0:
                lab[lab == 5] = 0                    # punch a gap in the ids
            p = SegmentParams(min_size=int(rng.integers(0, 10)),
                              max_size=int(rng.integers(0, 60)))
            ref = lab
            if p.max_size > 0:
                ref = _drop_large(ref, p.max_size)
            if p.min_size > 0:
                ref = _drop_small(ref, p.min_size)
            ref = _relabel_sequential(ref)
            assert np.array_equal(_finalize_labels(lab, p), ref), (
                f"trial {trial}: min_size={p.min_size} max_size={p.max_size}")


class TestNaNBorder:
    """Plan trap #2 / gate A7 — the drift↔segmentation seam."""

    def test_nan_border_yields_no_particles_there(self):
        from spyde.drift import shift_frame
        img = _field()
        shifted = shift_frame(img, (12, -15))       # NaN band top and right
        assert np.isnan(shifted).any(), "test setup produced no NaN border"

        labels = labels_from(shifted, min_size=30)
        nan_mask = ~np.isfinite(shifted)
        assert not np.any(labels[nan_mask]), (
            "found a particle inside the NaN-padded border — the padding was "
            "coerced to a value that thresholds as signal")

    def test_nan_does_not_erase_real_data_near_the_border(self):
        """The opposite failure: propagating NaN through the filters wipes a band."""
        from spyde.drift import shift_frame
        img = _field()
        shifted = shift_frame(img, (5, 0))
        labels = labels_from(shifted, blur=2.0, min_size=30)
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
        labels = labels_from(img, min_size=30)
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
        """A REORDERED/renamed layout is unreadable — but see the test below:
        columns merely APPENDED since are not a layout change."""
        import json
        from spyde.signals.particles import FORMAT_VERSION
        path = str(tmp_path / "cols.npz")
        np.savez_compressed(
            path, flat_buffer=np.zeros((0, 2), np.float32),
            t_offsets=np.array([0]),
            meta=np.array(json.dumps({"format_version": FORMAT_VERSION,
                                      "columns": ["label", "t"],   # swapped
                                      "frame_shape": [4, 4]})))
        with pytest.raises(ValueError, match="column layout changed"):
            SpyDEParticles.load(path)

    def test_load_migrates_a_file_written_before_a_column_was_appended(self, tmp_path):
        """An older file must still open.

        New columns go at the END precisely so this is possible. Refusing the
        file would make every previously-saved particle result unopenable in
        order to add one DERIVED number — and it is derived, so it never needed
        to have been stored.

        The migrated `score` is 1.0, not 0.0: absent evidence is not evidence of
        a bad particle, and the caret's confidence filter must not silently
        delete instances measured before the score existed.
        """
        import json
        from spyde.signals.particles import COLUMNS as _COLS, COL as _COL
        old_cols = _COLS[:-1]                      # everything before `score`
        path = str(tmp_path / "old.npz")
        buf = np.zeros((3, len(old_cols)), np.float32)
        buf[:, 0] = [0, 0, 1]                      # t
        np.savez_compressed(
            path, flat_buffer=buf, t_offsets=np.array([0, 2, 3]),
            meta=np.array(json.dumps({"format_version": 1,
                                      "columns": list(old_cols),
                                      "frame_shape": [4, 4]})))
        p = SpyDEParticles.load(path)
        assert p.flat_buffer.shape == (3, N_COLUMNS)
        assert np.all(p.flat_buffer[:, _COL["score"]] == 1.0), (
            "migrated particles must not be filtered away by the confidence "
            "slider — they carry no evidence, not bad evidence")

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


# The drift-padded NaN border used to be guarded HERE, against the classical
# engine's `_prepare`: it asserted the fill polarity followed `invert`, because
# filling with the finite minimum while inverting made the padding the brightest
# thing in the frame and it segmented as one 240 px edge-hugging instance.
#
# That function is gone with the engine, and the concern moved to the path that
# actually runs: `features.prepare_frame` fills non-finite pixels with the finite
# minimum AND returns a validity mask, and the classifier forces those pixels to
# zero probability (plan trap 2 / gate A7). It is covered end-to-end, on a real
# warped frame with a trained head, by
# `test_particles_scribble.py::TestNaNBorder` — which asserts the stronger thing
# this class could not: no INSTANCE lands in the padding, the padding reports
# UNLABELLED rather than a class, and real data next to the border still
# classifies (a filter propagating NaN outward would blank ~33 px of it).


class TestConfidenceScore:
    """One slider that means "fewer / more particles" on every engine.

    The reported failure was 547 instances where ~30 were real, and six Advanced
    knobs none of which says "fewer particles": min size, max size, split
    touching, min separation, marker smoothing, drop-edge. Size and shape cannot
    fix it — over-split support-film texture is often SMALL and ROUND, which is
    exactly what a size/circularity filter keeps.

    Contrast-to-noise against the instance's own dilated background ring is the
    statistic that does separate them: a real particle sits well away from its
    surroundings, while a fragment of textured film is by construction the same
    brightness as the texture around it.
    """

    @staticmethod
    def _field():
        """~30 genuinely dark particles in a noisy support film, plus ~300
        labelled fragments OF that film — the reported failure, in miniature."""
        rng = np.random.default_rng(0)
        h = w = 256
        frame = np.full((h, w), 200.0, np.float32)
        frame += rng.normal(0, 6.0, (h, w)).astype(np.float32)
        labels = np.zeros((h, w), np.int32)
        truth, nxt = {}, 1
        y, x = np.mgrid[0:h, 0:w]
        for _ in range(30):
            cy, cx, r = rng.uniform(15, h - 15), rng.uniform(15, w - 15), rng.uniform(5, 10)
            m = (y - cy) ** 2 + (x - cx) ** 2 < r * r
            frame[m] = 120.0 + rng.normal(0, 5.0, int(m.sum())).astype(np.float32)
            labels[m] = nxt; truth[nxt] = True; nxt += 1
        for _ in range(300):
            cy, cx, r = rng.uniform(8, h - 8), rng.uniform(8, w - 8), rng.uniform(3, 6)
            m = ((y - cy) ** 2 + (x - cx) ** 2 < r * r) & (labels == 0)
            if m.sum() < 20:
                continue
            labels[m] = nxt; truth[nxt] = False; nxt += 1
        return frame, labels, truth

    def test_the_score_separates_particles_from_textured_film(self):
        from spyde.particles.measure import measure_frame
        from spyde.signals.particles import COL

        frame, labels, truth = self._field()
        rows, _ = measure_frame(labels, frame, t=0)
        scores = rows[:, COL["score"]]
        real = np.array([truth.get(int(l), False)
                         for l in rows[:, COL["label"]].astype(int)])
        assert real.sum() and (~real).sum(), "the fixture built only one population"
        # A gap, not merely a difference in means: the slider is only usable if
        # SOME threshold cleanly separates them.
        assert np.percentile(scores[real], 10) > np.percentile(scores[~real], 90), (
            f"no separating threshold exists: real p10="
            f"{np.percentile(scores[real], 10):.3f} vs texture p90="
            f"{np.percentile(scores[~real], 90):.3f}")

    def test_filtering_keeps_rows_and_contours_aligned(self):
        """A misalignment draws one particle's outline on another's row."""
        from spyde.actions.particles_action import filter_by_score
        from spyde.signals.particles import COL, N_COLUMNS

        rows = np.zeros((5, N_COLUMNS), np.float32)
        rows[:, COL["score"]] = [0.05, 0.2, 0.6, 0.95, 0.99]
        rows[:, COL["label"]] = [1, 2, 3, 4, 5]
        contours = [np.full((3, 2), i, np.int16) for i in range(5)]
        kept_rows, kept_contours = filter_by_score(rows, contours, 0.5)
        assert len(kept_rows) == len(kept_contours) == 3
        # the SURVIVING contours must be the ones belonging to the kept rows
        assert [int(c[0, 0]) for c in kept_contours] == [2, 3, 4]

    def test_zero_is_the_old_behaviour_untouched(self):
        from spyde.actions.particles_action import filter_by_score
        from spyde.signals.particles import COL, N_COLUMNS

        rows = np.zeros((3, N_COLUMNS), np.float32)
        rows[:, COL["score"]] = [0.0, 0.5, 1.0]
        cont = [np.zeros((2, 2), np.int16)] * 3
        out_rows, out_cont = filter_by_score(rows, cont, 0.0)
        assert out_rows is rows and out_cont is cont, (
            "the default must be a no-op, not a copy")

    def test_unmeasurable_particles_are_kept_not_hidden(self):
        """No background ring => no evidence => not marginal.

        Scoring an unmeasurable instance 0 would let the slider silently delete
        particles it knows nothing about, which is the opposite of what a
        'hide the marginal ones' control should do.
        """
        from spyde.particles.measure import particle_scores
        from spyde.signals.particles import COL, N_COLUMNS

        rows = np.zeros((1, N_COLUMNS), np.float32)
        rows[:, COL["intensity_mean"]] = 100.0
        rows[:, COL["background"]] = np.nan          # ring fell outside the frame
        rows[:, COL["intensity_std"]] = np.nan
        assert particle_scores(rows)[0] == 1.0


class TestPerComponentWatershed:
    """Large frames watershed each component in its own bbox.

    A whole-frame watershed allocates several full-frame rasters — measured at
    4096², `split_instances` alone peaks at 546 MB of an 852 MB total. With
    dask's `threads_per_worker=4` that is ~3.4 GB of concurrent peak per worker,
    which drove a 9.24 GiB worker into a pause/resume/restart loop with
    "unmanaged memory" warnings (the arrays are ours, inside the task, so dask
    can neither see nor spill them).

    Cropping is EXACT, not an approximation: a connected component is surrounded
    by background by definition, so a 1 px pad contains every pixel the distance
    transform, the markers and the watershed can depend on, and no watershed can
    flow between two components that do not touch.
    """

    @staticmethod
    def _field(n, k, seed=0):
        rng = np.random.default_rng(seed)
        fg = np.zeros((n, n), bool)
        y, x = np.mgrid[0:n, 0:n]
        for _ in range(k):
            cy, cx = rng.uniform(20, n - 20), rng.uniform(20, n - 20)
            r = rng.uniform(8, 22)
            fg |= (y - cy) ** 2 + (x - cx) ** 2 < r * r     # overlapping on purpose
        return fg

    def test_identical_to_the_whole_frame_route_when_neither_decimates(self):
        """The exactness claim, where it can be checked directly.

        Below `_SPLIT_DECIMATE_ABOVE` both routes compute the split geometry at
        full resolution, so they must agree pixel for pixel. Above it they do
        NOT, and that is decimation rather than a cropping error — see the next
        test.
        """
        import spyde.particles.instances as C
        from spyde.particles.instances import SegmentParams, split_instances

        fg = self._field(1024, 90)
        p = SegmentParams(min_size=5, watershed=True)
        assert C._split_factor(fg.shape, p) == 1, "fixture must not decimate"

        orig = C._COMPONENT_ROUTE_PX
        try:
            C._COMPONENT_ROUTE_PX = 1 << 60          # force whole-frame
            whole = split_instances(fg, p)
            C._COMPONENT_ROUTE_PX = 0                # force per-component
            comp = split_instances(fg, p)
        finally:
            C._COMPONENT_ROUTE_PX = orig

        assert whole.max() == comp.max(), (
            f"instance COUNT differs with no decimation in play: "
            f"{whole.max()} vs {comp.max()}")
        # Ids may be numbered differently; the PARTITION must be the same.
        for v in np.unique(whole):
            if v == 0:
                continue
            m = whole == v
            ids = np.unique(comp[m])
            assert len(ids) == 1 and ids[0] != 0, (
                "a whole-frame instance was split across the crop boundary")

    def test_a_component_spanning_a_crop_is_not_broken_up(self):
        """One long diagonal particle — the shape a naive band split ruins."""
        import spyde.particles.instances as C
        from spyde.particles.instances import SegmentParams, split_instances

        n = 512
        fg = np.zeros((n, n), bool)
        for i in range(40, n - 40):
            fg[i - 3:i + 3, i - 3:i + 3] = True       # corner to corner
        orig = C._COMPONENT_ROUTE_PX
        try:
            C._COMPONENT_ROUTE_PX = 0
            lab = split_instances(fg, SegmentParams(min_size=5, watershed=True))
        finally:
            C._COMPONENT_ROUTE_PX = orig
        assert lab.max() >= 1
        # every foreground pixel belongs to a labelled instance
        assert (lab[fg] > 0).all(), "pixels were dropped at a crop edge"

    def test_labels_are_globally_unique_across_components(self):
        """Each crop labels 1..k locally; the offset into the global space is
        where an off-by-one silently merges two particles."""
        import spyde.particles.instances as C
        from spyde.particles.instances import SegmentParams, split_instances

        fg = np.zeros((256, 256), bool)
        fg[20:60, 20:60] = True          # three well-separated squares
        fg[20:60, 120:160] = True
        fg[120:160, 20:60] = True
        orig = C._COMPONENT_ROUTE_PX
        try:
            C._COMPONENT_ROUTE_PX = 0
            lab = split_instances(fg, SegmentParams(min_size=5, watershed=True))
        finally:
            C._COMPONENT_ROUTE_PX = orig
        assert lab.max() == 3, f"expected 3 instances, got {lab.max()}"
        assert len(set(np.unique(lab)) - {0}) == 3

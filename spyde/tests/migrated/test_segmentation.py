"""
spyde.segmentation — the compute contracts, on the synthetic particle movie.

Every classifier test runs on the CPU: torch-CUDA work segfaults under the
pytest process on Windows (CLAUDE.md). The wiring is device-agnostic.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from spyde.data.synthetic import ground_truth, particle_movie, particle_truth_at
from spyde.segmentation import (
    BACKGROUND, BOUNDARY, PARTICLE, Labels, PixelClassifier, field_source,
    measure_instances, segment_field, split_instances,
)
from spyde.segmentation.features import FeatureBank, Normalisation, align_down
from spyde.segmentation.labels import stroke_indices

BRIGHT = (0, 1)          # the two static anchors
FAINT_PAINTED = 8        # one of the two faint probes is painted...
FAINT_UNPAINTED = 7      # ...and the other has to be FOUND, not memorised


def paint_scribbles(labels: Labels, truth: dict, *, field: int = 0,
                    particles=BRIGHT + (FAINT_PAINTED,), boundary: bool = False) -> None:
    """A short stroke through each named particle and two through the film."""
    positions, radii, _present = particle_truth_at(truth, field)
    for index in particles:
        y, x = positions[index]
        labels.paint(field, [[y, x - 1], [y, x + 1]], PARTICLE,
                     radius=max(1.0, radii[index] * 0.4))
    labels.paint(field, [[4, 4], [4, 100]], BACKGROUND, radius=2)
    labels.paint(field, [[50, 55], [60, 55]], BACKGROUND, radius=2)
    if boundary:
        labels.paint(field, [[84, 70], [84, 72]], BOUNDARY, radius=1.5)


@pytest.fixture(scope="module")
def movie():
    signal = particle_movie(n_frames=8)
    return signal, ground_truth(signal), field_source(signal)


@pytest.fixture(scope="module")
def trained(movie):
    signal, truth, source = movie
    labels = Labels(source.shape)
    paint_scribbles(labels, truth)
    classifier = PixelClassifier(device="cpu")
    report = classifier.fit(labels, source.get)
    return classifier, report, labels


def _centroids_px(table: dict, scale: float) -> np.ndarray:
    return np.stack([table["y"] / scale, table["x"] / scale], axis=-1)


class TestLabels:
    def test_a_fast_stroke_paints_a_continuous_line(self):
        """The widget reports one point per animation frame, so a fast stroke
        jumps many pixels between samples; dabbing only at the samples leaves
        a dotted line."""
        indices = stroke_indices((64, 64), [[10, 5], [10, 55]], radius=1.0)
        ys, xs = np.divmod(indices, 64)
        assert set(xs[ys == 10]) == set(range(4, 57))

    def test_repainting_replaces_the_class_and_erasing_removes(self):
        labels = Labels((32, 32))
        labels.paint(0, [[8, 8], [8, 12]], PARTICLE, radius=1)
        labels.paint(0, [[8, 10], [8, 12]], BACKGROUND, radius=1)
        indices, classes, strokes = labels.at(0)
        assert classes[np.isin(indices, 8 * 32 + np.array([10, 11, 12]))].tolist() \
            == [BACKGROUND] * 3
        assert len(set(strokes.tolist())) == 2, "each stroke keeps its own id"
        labels.erase(0, [[8, 8], [8, 12]], radius=1)
        assert len(labels) == 0 and labels.painted_fields() == []

    def test_counts_and_round_trip(self):
        labels = Labels((40, 48))
        labels.paint(2, [[10, 10], [10, 20]], PARTICLE, radius=2)
        labels.paint(2, [[30, 10], [30, 20]], BACKGROUND, radius=2)
        counts = labels.counts()
        assert counts[PARTICLE] > 0 and counts[BACKGROUND] > 0 and counts[BOUNDARY] == 0
        clone = Labels.from_dict(labels.to_dict())
        assert clone.to_dict() == labels.to_dict()
        clone.paint(2, [[20, 20], [20, 22]], BOUNDARY, radius=1)
        assert clone.at(2)[2].max() == 2, "stroke ids continue after a reload"


class TestFeatureBank:
    def test_a_band_reproduces_the_full_field_exactly(self, movie):
        """Prediction is banded with a halo; a band that starts on the
        decimation grid must give the SAME logits as the whole field — not
        merely close, or the halo is too small."""
        _signal, _truth, source = movie
        field = source.get(0)
        whole = PixelClassifier(device="cpu", band_rows=4096)
        banded = PixelClassifier(device="cpu", band_rows=64)
        labels = Labels(source.shape)
        labels.paint(0, [[24, 20], [24, 24]], PARTICLE, radius=2)
        labels.paint(0, [[4, 4], [4, 60]], BACKGROUND, radius=2)
        whole.fit(labels, source.get)
        banded._weights, banded.classes = whole._weights, list(whole.classes)
        np.testing.assert_array_equal(whole.probabilities(field), banded.probabilities(field))

    def test_a_stroke_crop_features_match_the_full_field(self, movie):
        """Training featurises the stroke's bounding box only; those vectors
        must equal the full-field stack at the same pixels, or the head is
        trained on features prediction never produces."""
        import torch
        _signal, _truth, source = movie
        field = source.get(0).astype(np.float32)
        classifier = PixelClassifier(device="cpu")
        indices = stroke_indices(source.shape, [[70, 26], [70, 34]], radius=2)
        normalisation = Normalisation.from_field(field)
        crop = classifier._sample(field, indices, normalisation).numpy()
        stack = classifier.bank(normalisation.apply(torch.as_tensor(field)[None, None]))[0]
        ys, xs = np.divmod(indices, source.shape[1])
        full = stack[:, ys, xs].t().numpy()
        np.testing.assert_allclose(crop, full, rtol=1e-5, atol=1e-5)

    def test_normalisation_makes_the_features_scale_invariant(self, movie):
        """The same sample recorded on a different intensity scale must
        classify the same: one robust centre/scale per field."""
        import torch
        _signal, _truth, source = movie
        field = source.get(0)
        bank = FeatureBank(device="cpu")
        one = bank(Normalisation.from_field(field).apply(torch.as_tensor(field)[None, None]))
        scaled = field * 7.0 + 3.0
        two = bank(Normalisation.from_field(scaled).apply(torch.as_tensor(scaled)[None, None]))
        np.testing.assert_allclose(one.numpy(), two.numpy(), rtol=1e-3, atol=1e-3)

    def test_align_down_lands_on_the_decimation_grid(self):
        assert [align_down(v) for v in (0, 3, 4, 5, 9, -2)] == [0, 0, 4, 4, 8, 0]


class TestClassifier:
    def test_finds_every_particle_including_the_unpainted_faint_probe(self, movie, trained):
        """The sensitivity gate: two bright anchors and ONE faint probe are
        painted; the other faint probe, at ~7x the noise, must be found from
        what the classifier learnt, not from memory."""
        _signal, truth, source = movie
        classifier, report, _labels = trained
        assert report["train_accuracy"] > 0.95
        assert report["classes"] == [PARTICLE, BACKGROUND]
        labels, table = segment_field(source.get(0), classifier, scale=1.0, min_size=10)
        positions, _radii, present = particle_truth_at(truth, 0)
        found = _centroids_px(table, 1.0)
        for index in np.flatnonzero(present):
            distance = np.hypot(*(found - positions[index]).T).min()
            assert distance < 1.5, f"particle {index} not found (nearest {distance:.2f} px)"
        assert np.hypot(*(found - positions[FAINT_UNPAINTED]).T).min() < 1.5
        # The discs are ~9% of the frame; the logit smoothing dilates the mask
        # of these small soft-edged particles, so the bound is loose. What it
        # catches is a classifier that called the film 'particle'.
        assert (labels > 0).mean() < 0.35, "the mask covers the film — it learnt 'everything'"

    def test_a_nan_border_gets_zero_probability_and_no_instance(self, movie, trained):
        """A drift-corrected frame carries a NaN border; nothing may be found in it."""
        _signal, _truth, source = movie
        classifier, _report, _labels = trained
        field = source.get(0).astype(np.float32)
        field[:12, :] = np.nan
        particle, boundary = classifier.foreground(field)
        assert boundary is None, "no boundary stroke, so no boundary map"
        assert not particle[:12].any()
        labels = split_instances(particle, min_size=5)
        assert not labels[:12].any()

    def test_the_fit_is_deterministic(self, movie, trained):
        _signal, _truth, source = movie
        _classifier, _report, labels = trained
        first = PixelClassifier(device="cpu").fit(labels, source.get)
        second = PixelClassifier(device="cpu").fit(labels, source.get)
        assert first["loss"] == second["loss"]

    def test_save_load_and_pickle_predict_identically(self, movie, trained, tmp_path):
        _signal, _truth, source = movie
        classifier, _report, _labels = trained
        field = source.get(3)
        expected = classifier.probabilities(field)
        classifier.save(str(tmp_path / "head.npz"))
        loaded = PixelClassifier.load(str(tmp_path / "head.npz"), device="cpu")
        np.testing.assert_array_equal(loaded.probabilities(field), expected)
        cloned = pickle.loads(pickle.dumps(classifier))
        np.testing.assert_array_equal(cloned.probabilities(field), expected)

    def test_a_painted_boundary_gives_a_boundary_map(self, movie):
        _signal, truth, source = movie
        labels = Labels(source.shape)
        paint_scribbles(labels, truth, boundary=True)
        classifier = PixelClassifier(device="cpu")
        classifier.fit(labels, source.get)
        assert BOUNDARY in classifier.classes
        particle, boundary = classifier.foreground(source.get(0))
        assert boundary is not None and boundary.shape == source.shape

    def test_one_painted_class_is_refused(self, movie):
        _signal, _truth, source = movie
        labels = Labels(source.shape)
        labels.paint(0, [[24, 20], [24, 24]], PARTICLE, radius=2)
        with pytest.raises(RuntimeError, match="two classes"):
            PixelClassifier(device="cpu").fit(labels, source.get)


def _touching_discs(shape=(64, 96), centres=((32, 30), (32, 58)), radius=15):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    mask = np.zeros(shape, bool)
    for cy, cx in centres:
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return mask


class TestSplit:
    def test_boundary_and_watershed_agree_on_touching_discs(self):
        """The boundary route must give the watershed's count and claim the
        same foreground pixels, or its speed is not worth having."""
        mask = _touching_discs()
        seam = np.zeros(mask.shape, bool)
        seam[:, 43:46] = True
        by_watershed = split_instances(mask, min_size=0)
        by_boundary = split_instances(mask, seam & mask, min_size=0)
        assert by_watershed.max() == 2 and by_boundary.max() == 2
        np.testing.assert_array_equal(by_boundary > 0, by_watershed > 0)
        areas = lambda labels: sorted(np.bincount(labels.ravel())[1:])   # noqa: E731
        assert np.allclose(areas(by_boundary), areas(by_watershed), rtol=0.03)

    def test_a_single_disc_is_not_oversplit(self):
        """peak_local_max on a flat distance maximum returns several coincident
        peaks; the watershed must not cut one disc into wedges."""
        mask = _touching_discs(centres=((32, 48),), radius=20)
        assert split_instances(mask, min_size=0).max() == 1

    def test_a_tiny_particle_survives_the_watershed(self):
        """Nothing in the split may delete a small particle by itself; only
        ``min_size`` does, and that is the user's choice."""
        mask = _touching_discs(centres=((32, 48),), radius=20)
        mask[5:8, 5:8] = True
        assert split_instances(mask, min_size=0).max() == 2
        assert split_instances(mask, min_size=20).max() == 1

    def test_labels_are_dense_from_one_after_the_size_filter(self):
        mask = _touching_discs()
        mask[2:4, 2:4] = True                     # a 4-pixel speck
        assert sorted(np.unique(split_instances(mask, min_size=3)).tolist()) == [0, 1, 2, 3]
        assert sorted(np.unique(split_instances(mask, min_size=5)).tolist()) == [0, 1, 2]


class TestMeasure:
    def test_calibration_scales_lengths_once_and_areas_twice(self):
        labels = _touching_discs(centres=((32, 48),), radius=10).astype(np.int32)
        pixels = measure_instances(labels, scale=1.0)
        calibrated = measure_instances(labels, scale=0.5)
        assert calibrated["area"][0] == pytest.approx(pixels["area"][0] * 0.25)
        assert calibrated["equivalent_diameter"][0] == pytest.approx(
            pixels["equivalent_diameter"][0] * 0.5)
        assert calibrated["eccentricity"][0] == pixels["eccentricity"][0], "dimensionless"
        assert calibrated["y"][0] == pytest.approx(32 * 0.5, abs=0.5)

    def test_intensity_ignores_non_finite_pixels(self):
        labels = np.zeros((20, 20), np.int32)
        labels[5:10, 5:10] = 1
        image = np.full((20, 20), 2.0)
        image[5:7, 5:10] = np.nan
        table = measure_instances(labels, image)
        assert table["intensity_mean"][0] == 2.0 and table["intensity_std"][0] == 0.0
        assert table["intensity_max"][0] == 2.0

    def test_no_instances_gives_empty_columns(self):
        table = measure_instances(np.zeros((8, 8), np.int32))
        assert all(len(column) == 0 for column in table.values())


class TestFields:
    def test_an_image_is_one_field(self, movie):
        signal, _truth, _source = movie
        source = field_source(signal.inav[0])
        assert (source.count, source.shape, source.is_movie) == (1, (96, 112), False)

    def test_a_movie_reads_one_frame_at_a_time(self, movie):
        signal, _truth, _source = movie
        lazy = signal.as_lazy()
        source = field_source(lazy)
        assert source.count == 8 and source.is_movie
        np.testing.assert_array_equal(source.get(3), signal.data[3])

    def test_a_scan_must_be_segmented_on_its_navigator(self):
        import hyperspy.api as hs
        scan = hs.signals.Signal2D(np.zeros((4, 5, 8, 8), np.float32))
        with pytest.raises(TypeError, match="navigator"):
            field_source(scan)
        assert field_source(np.zeros((4, 5), np.float32)).count == 1

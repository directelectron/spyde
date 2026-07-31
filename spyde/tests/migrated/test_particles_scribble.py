"""
The torch feature stack and the scribble classifier — DRIFT_AND_PARTICLES_PLAN.md
step 3 (B2 + B3), and the acceptance gates it exists to pass.

Four of these classes are the plan's gates rather than ordinary unit tests, and
they are the reason this file is worth its runtime:

:class:`TestRandomForestParity`
    Plan B3: the torch head must agree with ``sklearn``'s RandomForest — what
    ParticleSpy and ilastik actually use — on **identical labels and identical
    feature channels**. Agreement by IoU, measured 0.94.
:class:`TestSensitivityGate`
    Plan §0.9, the headline gate for the whole feature: the two deliberately
    faint low-contrast probes in ``particle_movie()`` must be found. It is kept
    non-vacuous by :class:`TestTheClassicalBaselineMissesThem`, which shows the
    classical engine finds neither.
:class:`TestNaNBorder`
    Plan trap 2 / A7: no foreground inside a drift-corrected frame's NaN padding.
:class:`TestInteractionBudget`
    Plan B3's hard budget: train + apply to one visible frame under ~1 s on CPU.

Everything runs on **CPU explicitly** (``select_device("cpu")``): torch-CUDA work
segfaults under the pytest process on Windows (CLAUDE.md), which is a harness
interaction and not a code defect, so the GPU path is left to the real app and to
a subprocess check.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from spyde.data.synthetic import ground_truth, particle_movie, particle_truth_at
from spyde.drift.warp import shift_frame
from spyde.particles import SegmentParams, segment_frame, split_instances
from spyde.particles import features as feat
from spyde.particles.features import (
    DEFAULT_SIGMAS,
    FeatureSpec,
    PreparedFrame,
    band_rows_for,
    feature_stack,
    feature_tensor,
    map_feature_bands,
    prepare_frame,
    sample_features,
    select_device,
)
from spyde.particles.scribble import (
    UNLABELLED,
    LabelStore,
    ScribbleClass,
    ScribbleClassifier,
    default_classes,
    masks_to_labels,
    random_forest_reference,
)

#: The frame every gate is measured on: all nine particles are present at t=12.
FRAME_T = 12

DEVICE = select_device("cpu")


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def movie():
    """One build shared across the module — ~1 s and deterministic."""
    s = particle_movie()
    return s, ground_truth(s)


@pytest.fixture(scope="module")
def geom(movie):
    """``(positions, radii, present, faint, shape)`` at :data:`FRAME_T`."""
    _s, gt = movie
    pos, radii, present = particle_truth_at(gt, FRAME_T)
    faint = np.asarray(gt["p_faint"], bool)
    return pos, radii, present, faint, tuple(gt["frame_shape"])


def _clear_of_particles(shape, pos, radii, present, pad: float):
    """True where no present particle is within *pad* px — where background may
    be painted without accidentally labelling a particle as film."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    keep = np.ones((h, w), bool)
    for i in np.flatnonzero(present):
        keep &= ((yy - pos[i, 0]) ** 2 + (xx - pos[i, 1]) ** 2) > (radii[i] + pad) ** 2
    return keep


def paint_scribbles(geom, *, include_faint=(8,), n_bright=4,
                    t: int = FRAME_T) -> LabelStore:
    """The scribbles a user would actually paint on one frame.

    Four dabs on bright particles, a background stroke at each of those
    particles' boundaries, four background sweeps across the film, and — unless
    *include_faint* is emptied — one dab on the SMALLER of the two faint probes
    (index 8, r=3). Index 7 (r=4) is never painted, so it is a genuinely held-out
    detection in :class:`TestSensitivityGate`.

    The boundary strokes matter and are not padding: without them the head has
    never seen a not-quite-particle pixel, its masks come out visibly fat, and
    agreement with the RandomForest reference falls from 0.94 to 0.69 (measured).
    That is a fact about labelling, not about either classifier.
    """
    pos, radii, present, faint, shape = geom
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    store = LabelStore(frame_shape=shape, classes=default_classes())

    bright = [i for i in np.flatnonzero(present) if not faint[i]][:n_bright]
    for i in bright:
        store.paint_disc(t, pos[i, 0], pos[i, 1], max(1.5, radii[i] * 0.5), 0)
    for i in include_faint:
        store.paint_disc(t, pos[i, 0], pos[i, 1], 1.5, 0)

    far = _clear_of_particles(shape, pos, radii, present, 3.0)
    for (y0, x0, y1, x1) in ((4, 4, 8, 100), (h - 8, 4, h - 4, 100),
                             (40, 4, 60, 8), (20, 40, 40, 46)):
        sweep = np.zeros((h, w), bool)
        sweep[y0:y1, x0:x1] = True
        store.paint(t, sweep & far, 1)

    touching = _clear_of_particles(shape, pos, radii, present, 0.0)
    for i in bright:
        d2 = (yy - pos[i, 0]) ** 2 + (xx - pos[i, 1]) ** 2
        ring = (d2 > (radii[i] + 2.5) ** 2) & (d2 <= (radii[i] + 4.0) ** 2)
        store.paint(t, ring & touching, 1)
    return store


@pytest.fixture(scope="module")
def labels(geom):
    return paint_scribbles(geom)


@pytest.fixture(scope="module")
def trained(movie, labels):
    s, _gt = movie
    clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
    clf.fit(labels, {FRAME_T: s.data[FRAME_T]})
    return clf


@pytest.fixture(scope="module")
def proba(movie, trained):
    s, _gt = movie
    return trained.predict_proba(s.data[FRAME_T])


def hit(prob_or_labels, pos, i) -> bool:
    """Did the map fire at particle *i*'s ground-truth centre?"""
    cy, cx = int(round(pos[i, 0])), int(round(pos[i, 1]))
    v = prob_or_labels[cy, cx]
    return bool(v > 0.5) if np.issubdtype(np.asarray(v).dtype, np.floating) \
        else bool(v != 0)


# ── FeatureSpec ──────────────────────────────────────────────────────────────

class TestFeatureSpec:
    def test_the_scale_floor_is_fine(self):
        """Plan §0.9. What raising this costs is measured in
        :meth:`TestSensitivityGate.test_a_coarse_stack_undersizes_the_small_particles`
        — this only pins that it has not been raised without going and looking."""
        assert min(DEFAULT_SIGMAS) <= 1.0, (
            "the default scale floor has been raised above 1 px; that costs the "
            "smallest particles' measured radius (see the features module "
            "docstring) and needs a fresh sensitivity measurement, not a guess")
        assert sorted(DEFAULT_SIGMAS) == list(DEFAULT_SIGMAS)

    def test_default_channel_count_and_names(self):
        spec = FeatureSpec()
        names = spec.channel_names()
        assert len(names) == spec.n_channels == 36
        assert len(set(names)) == len(names), "duplicate channel names"
        # Every promised family is present.
        for token in ("intensity", "gaussian_s", "dog_s", "sobel_s", "laplacian_s",
                      "hessian_major_s", "hessian_minor_s", "median_r",
                      "minimum_r", "maximum_r"):
            assert any(n.startswith(token) for n in names), f"missing {token}"

    def test_names_match_the_stack_for_every_configuration(self, movie):
        """The names and the tensor come from one plan; this is what stops them
        drifting apart when a family is added."""
        s, _gt = movie
        frame = s.data[0][:32, :32]
        for spec in (FeatureSpec(),
                     FeatureSpec(membrane=True),
                     FeatureSpec(sigmas=(1.0,), median=False, minimum=False,
                                 maximum=False, membrane=True),
                     FeatureSpec(intensity=False, gaussian=False, sobel=False,
                                 hessian=False, laplacian=False,
                                 difference_of_gaussians=False)):
            stack = feature_stack(frame, spec, device=DEVICE)
            assert stack.shape[0] == len(spec.channel_names()) == spec.n_channels

    def test_round_trips_through_a_dict(self):
        spec = FeatureSpec(sigmas=(0.7, 1.4), rank_radii=(3,), membrane=True,
                           membrane_projections=("sum", "min"))
        d = spec.to_dict()
        assert json.loads(json.dumps(d)) == d, "not JSON-safe"
        assert FeatureSpec.from_dict(d) == spec

    def test_from_dict_ignores_unknown_keys(self):
        """A recipe written by a newer build must still open here."""
        d = FeatureSpec().to_dict()
        d["some_future_feature"] = True
        assert FeatureSpec.from_dict(d) == FeatureSpec()

    def test_from_dict_of_nothing_is_the_default(self):
        assert FeatureSpec.from_dict(None) == FeatureSpec()
        assert FeatureSpec.from_dict({}) == FeatureSpec()

    def test_replace_and_hashable(self):
        spec = FeatureSpec()
        assert spec.replace(median=False).median is False
        assert spec.median is True, "replace mutated the original"
        assert hash(spec) == hash(FeatureSpec())

    @pytest.mark.parametrize("kw, match", [
        ({"sigmas": (2.0, 1.0)}, "ascending"),
        ({"sigmas": (0.0, 1.0)}, "positive"),
        ({"rank_radii": (0,)}, "rank_radii"),
        ({"membrane_projections": ("nope",)}, "membrane projection"),
        ({"membrane": True, "membrane_patch": 18}, "odd"),
        ({"membrane": True, "membrane_rotations": 0}, "rotations"),
    ])
    def test_validation(self, kw, match):
        with pytest.raises(ValueError, match=match):
            FeatureSpec(**kw)

    def test_a_spec_with_no_channels_at_all_raises(self):
        with pytest.raises(ValueError, match="no channels"):
            FeatureSpec(intensity=False, gaussian=False,
                        difference_of_gaussians=False, sobel=False, hessian=False,
                        laplacian=False, median=False, minimum=False,
                        maximum=False)

    def test_halo_covers_the_largest_filter(self):
        spec = FeatureSpec()
        # sigma 8 truncated at 4 sigma is radius 32, plus the 3-tap derivative.
        assert spec.halo == 33
        assert FeatureSpec(sigmas=(1.0,), rank_radii=(7,)).halo == 7
        assert FeatureSpec(sigmas=(1.0,), rank_radii=(1,), membrane=True,
                           membrane_patch=19).halo == 9


# ── the feature stack ────────────────────────────────────────────────────────

class TestFeatureParity:
    """Every channel against the scipy/skimage filter it re-implements.

    ``normalize_frame=False`` throughout, because the whole point is comparing the
    *filters*, not the standardisation. Padding is compared with scipy's
    ``mirror``, which is what torch's ``reflect`` is (scipy's own ``reflect``
    duplicates the edge sample and is a different thing).
    """

    @pytest.fixture(scope="class")
    def img(self, movie):
        s, _gt = movie
        return np.ascontiguousarray(s.data[FRAME_T], dtype=np.float32)

    @pytest.fixture(scope="class")
    def stack(self, img):
        spec = FeatureSpec(normalize_frame=False, membrane=True)
        arr = feature_stack(img, spec, device=DEVICE)
        return spec, arr, spec.channel_names()

    def test_shape_and_dtype(self, img, stack):
        spec, arr, _names = stack
        assert arr.shape == (spec.n_channels, *img.shape)
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()

    def test_intensity_channel_is_the_frame(self, img, stack):
        _spec, arr, names = stack
        assert np.array_equal(arr[names.index("intensity")], img)

    @pytest.mark.parametrize("sigma", DEFAULT_SIGMAS)
    def test_gaussian_matches_scipy(self, img, stack, sigma):
        from scipy.ndimage import gaussian_filter
        _spec, arr, names = stack
        ref = gaussian_filter(img, float(sigma), mode="mirror")
        got = arr[names.index(f"gaussian_s{sigma:g}")]
        assert np.abs(got - ref).max() < 1e-5

    def test_difference_of_gaussians_is_the_difference(self, stack):
        _spec, arr, names = stack
        got = arr[names.index("dog_s1_s2")]
        ref = arr[names.index("gaussian_s1")] - arr[names.index("gaussian_s2")]
        assert np.array_equal(got, ref)

    @pytest.mark.parametrize("radius", (1, 2))
    def test_rank_filters_match_scipy(self, img, stack, radius):
        from scipy.ndimage import maximum_filter, median_filter, minimum_filter
        _spec, arr, names = stack
        k = 2 * radius + 1
        for stat, fn in (("median", median_filter), ("minimum", minimum_filter),
                         ("maximum", maximum_filter)):
            ref = fn(img, size=k, mode="mirror")
            got = arr[names.index(f"{stat}_r{radius}")]
            assert np.array_equal(got, ref), f"{stat} r={radius}"

    def test_sobel_matches_skimage(self, img, stack):
        from scipy.ndimage import gaussian_filter
        from skimage.filters import sobel
        _spec, arr, names = stack
        ref = sobel(gaussian_filter(img, 1.0, mode="mirror"), mode="mirror")
        assert np.abs(arr[names.index("sobel_s1")] - ref).max() < 1e-5

    def test_laplacian_is_the_hessian_trace(self, stack):
        """Not decoration: the Laplacian channel is *derived* from the Hessian's
        own second derivatives, which is what makes it free."""
        _spec, arr, names = stack
        lap = arr[names.index("laplacian_s2")]
        tr = (arr[names.index("hessian_major_s2")]
              + arr[names.index("hessian_minor_s2")])
        assert np.abs(lap - tr).max() < 1e-4

    def test_hessian_eigenvalues_are_ordered_and_signed(self, stack):
        _spec, arr, names = stack
        major = arr[names.index("hessian_major_s2")]
        minor = arr[names.index("hessian_minor_s2")]
        assert (major >= minor - 1e-6).all(), "major must be the larger SIGNED value"
        # A bright blob is a maximum, so BOTH curvatures are negative at its centre.
        assert minor.min() < 0 < major.max()

    @pytest.mark.parametrize("radius, matched_sigma", [(1.5, 1.0), (3.0, 2.0),
                                                       (8.0, 4.0)])
    def test_hessian_picks_a_blob_out_at_its_own_scale(self, radius,
                                                      matched_sigma):
        """Scale space, and the whole reason the sigma set spans a range.

        A disc of radius *r* registers as a curvature minimum most strongly at
        sigma ≈ ``r/sqrt(2)``, so which scale fires *is* the size measurement the
        head has available. A stack whose floor is too high sees a small particle
        only in its coarse channels, which is why it then measures it too big —
        see :meth:`TestSensitivityGate.
        test_a_coarse_stack_undersizes_the_small_particles`.

        Checked at the centre of a synthetic disc rather than on the fixture: a
        fixture particle of radius 7 has a *flat* centre at sigma 2, so the plain
        "is it a curvature minimum at the centre" assertion is scale-dependent and
        would be measuring the wrong thing.
        """
        n = 64
        yy, xx = np.mgrid[0:n, 0:n]
        img = (((yy - 32) ** 2 + (xx - 32) ** 2) <= radius ** 2).astype(np.float32)
        spec = FeatureSpec(normalize_frame=False)
        names = spec.channel_names()
        arr = feature_stack(img, spec, device=DEVICE)
        centre = {s: float(arr[names.index(f"hessian_minor_s{s:g}")][32, 32])
                  for s in DEFAULT_SIGMAS}
        assert min(centre, key=lambda s: centre[s]) == matched_sigma, centre
        assert centre[matched_sigma] < 0, "a bright blob must be a minimum"

    def test_membrane_projections_are_ordered(self, stack):
        _spec, arr, names = stack
        mx = arr[names.index("membrane_max")]
        mn = arr[names.index("membrane_min")]
        mean = arr[names.index("membrane_mean")]
        assert (mn <= mean + 1e-6).all() and (mean <= mx + 1e-6).all()
        assert (arr[names.index("membrane_std")] >= 0).all()

    def test_membrane_responds_to_a_line_not_a_blob(self):
        """The family's whole purpose, and why it is off by default."""
        line = np.zeros((48, 48), np.float32)
        line[24, 4:44] = 1.0
        blob = np.zeros((48, 48), np.float32)
        yy, xx = np.mgrid[0:48, 0:48]
        blob[(yy - 24) ** 2 + (xx - 24) ** 2 <= 9] = 1.0
        spec = FeatureSpec(normalize_frame=False, membrane=True)
        names = spec.channel_names()
        k = names.index("membrane_std")
        line_std = feature_stack(line, spec, device=DEVICE)[k][24, 24]
        blob_std = feature_stack(blob, spec, device=DEVICE)[k][24, 24]
        assert line_std > 3 * blob_std, (
            f"membrane std does not discriminate a line ({line_std:.4f}) from a "
            f"blob ({blob_std:.4f})")

    def test_deterministic(self, img):
        spec = FeatureSpec()
        a = feature_stack(img, spec, device=DEVICE)
        b = feature_stack(img, spec, device=DEVICE)
        assert np.array_equal(a, b)

    def test_runs_on_the_cpu_with_no_gpu(self, img, monkeypatch):
        """CI and most user machines have no accelerator."""
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(feat, "select_device", lambda prefer=None:
                            torch.device("cpu"))
        assert feature_stack(img[:32, :32]).shape[0] == FeatureSpec().n_channels


class TestPrepareFrame:
    def test_non_finite_pixels_take_the_finite_minimum(self):
        """Matching ``classical._prepare``: the padding must read as background,
        which is the one value that cannot classify as a particle."""
        img = np.linspace(2.0, 6.0, 64, dtype=np.float32).reshape(8, 8)
        img[0, :3] = np.nan
        img[7, 7] = np.inf
        finite_min = float(img[np.isfinite(img)].min())
        prepared = prepare_frame(img, FeatureSpec(normalize_frame=False))
        assert prepared.image[0, :3].tolist() == [finite_min] * 3
        assert prepared.image[7, 7] == pytest.approx(finite_min)
        assert np.isfinite(prepared.image).all()
        # Not zero and not the mean: the padding must read as background, and on a
        # frame whose values are all well above zero, zero-fill would instead read
        # as a hole and the mean as ordinary signal.
        assert prepared.image.min() == pytest.approx(finite_min)

    def test_valid_mask_marks_exactly_the_finite_source_pixels(self):
        img = np.ones((8, 8), np.float32)
        img[3, 4] = np.nan
        prepared = prepare_frame(img)
        assert prepared.valid.sum() == 63
        assert not prepared.valid[3, 4]

    def test_robust_statistics_ignore_the_padding(self):
        """Filling first would let a large NaN border rescale the whole frame by
        how much of it was padding."""
        rng = np.random.default_rng(0)
        img = rng.normal(100.0, 5.0, (64, 64)).astype(np.float32)
        padded = img.copy()
        padded[:32] = np.nan
        a = prepare_frame(img).image[32:]
        b = prepare_frame(padded).image[32:]
        # Same finite content in rows 32+, so the same standardisation of it.
        assert np.abs(a - b).max() < 0.15

    def test_normalisation_makes_the_stack_scale_invariant(self, movie):
        """The property that makes a saved recipe transferable: the same sample
        recorded on a different intensity scale must featurise the same."""
        s, _gt = movie
        frame = s.data[FRAME_T]
        a = feature_stack(frame, FeatureSpec(), device=DEVICE)
        b = feature_stack(frame * 1000.0 + 7.0, FeatureSpec(), device=DEVICE)
        rms = np.sqrt(np.mean((a - b) ** 2))
        assert rms < 1e-3, f"features are not scale invariant (rms {rms:.4g})"

    def test_a_constant_frame_does_not_divide_by_zero(self):
        out = prepare_frame(np.full((8, 8), 3.0, np.float32))
        assert np.isfinite(out.image).all()

    @pytest.mark.parametrize("bad, match", [
        (np.zeros((4, 4, 4), np.float32), "must be 2-D"),
        (np.zeros((3, 8), np.float32), "at least 4x4"),
        (np.full((8, 8), np.nan, np.float32), "no finite pixels"),
    ])
    def test_errors(self, bad, match):
        with pytest.raises(ValueError, match=match):
            prepare_frame(bad)

    def test_accepts_an_already_prepared_frame(self, movie):
        s, _gt = movie
        prepared = prepare_frame(s.data[0])
        assert isinstance(prepared, PreparedFrame)
        a = feature_stack(prepared, device=DEVICE)
        b = feature_stack(s.data[0], device=DEVICE)
        assert np.array_equal(a, b)


class TestSmallFramesAndPadding:
    """A kernel wider than the image must not change which kernel is applied.

    ``F.pad(mode="reflect")`` refuses a pad larger than the dimension, and the
    kernels here are not small — a sigma-8 gaussian has radius 32 and a 19x19
    membrane patch radius 9. The first implementation clamped the *radius*, which
    silently substituted a narrower filter and so made a ``FeatureSpec`` mean
    different things on different frame sizes; a saved recipe would then not
    reproduce on a crop. The padding degrades to replicate instead.
    """

    @pytest.mark.parametrize("shape", [(4, 4), (5, 9), (8, 8), (20, 17)])
    @pytest.mark.parametrize("spec", [
        FeatureSpec(),
        FeatureSpec(membrane=True),                # 19x19 patch on a 4x4 frame
        FeatureSpec(rank_radii=(9,)),              # 19x19 window on a 4x4 frame
    ], ids=["default", "membrane", "wide_rank"])
    def test_a_frame_smaller_than_the_kernels_still_works(self, shape, spec):
        img = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        stack = feature_stack(img, spec, device=DEVICE)
        assert stack.shape == (spec.n_channels, *shape)
        assert np.isfinite(stack).all()

    def test_the_kernel_is_the_same_one_on_a_crop(self, movie):
        """The property clamping broke: interior pixels must featurise identically
        whether or not the frame around them is big enough for the kernel."""
        s, _gt = movie
        spec = FeatureSpec(sigmas=(4.0,), normalize_frame=False, median=False,
                           minimum=False, maximum=False)
        big = np.ascontiguousarray(s.data[FRAME_T], dtype=np.float32)
        whole = feature_stack(big, spec, device=DEVICE)
        crop = feature_stack(big[20:76, 20:92], spec, device=DEVICE)
        # 16 px in from the crop's edge is beyond the kernel's reach either way.
        assert np.abs(whole[:, 36:60, 36:76] - crop[:, 16:40, 16:56]).max() < 1e-4

    def test_the_normalisation_still_works_on_an_all_nan_border(self):
        img = np.full((32, 32), np.nan, np.float32)
        img[10:20, 10:20] = np.random.default_rng(1).standard_normal((10, 10))
        prepared = prepare_frame(img)
        assert prepared.valid.sum() == 100
        assert np.isfinite(feature_stack(prepared, device=DEVICE)).all()


class TestBanding:
    """Plan §0.1: nothing may assume the frame fits in memory.

    The banded stack must be IDENTICAL to the unbanded one, not merely close — a
    halo of at least the largest filter radius replaces what reflect padding would
    otherwise invent at a band boundary, so there is nothing to be approximate
    about. Anything less than exact means the halo is too small.
    """

    @pytest.fixture(scope="class")
    def img(self, movie):
        s, _gt = movie
        return np.ascontiguousarray(s.data[FRAME_T], dtype=np.float32)

    @pytest.mark.parametrize("rows", (16, 33, 40, 500))
    def test_banded_equals_unbanded_exactly(self, img, rows):
        spec = FeatureSpec()
        whole = feature_stack(img, spec, device=DEVICE)
        got = np.zeros_like(whole)
        seen = []

        def take(y0, y1, stack):
            seen.append((y0, y1))
            got[:, y0:y1] = stack.detach().cpu().numpy()

        map_feature_bands(img, spec, device=DEVICE, fn=take, band_rows=rows)
        assert seen[0][0] == 0 and seen[-1][1] == img.shape[0]
        assert [a for a, _ in seen[1:]] == [b for _, b in seen[:-1]], "gap or overlap"
        assert np.array_equal(got, whole), (
            f"banded stack differs at band_rows={rows}: max "
            f"{np.abs(got - whole).max():.3g} — the halo is too small")

    def test_band_rows_stays_above_the_halo(self):
        spec = FeatureSpec()
        # A tiny budget must still not produce a band shorter than its own halo,
        # which would recompute more halo than payload.
        assert band_rows_for(spec, 4096, budget_bytes=1) >= 4 * spec.halo
        # A generous budget covers a whole 4096-row frame in one band.
        assert band_rows_for(spec, 4096, budget_bytes=8 << 30) >= 4096

    def test_sample_features_matches_the_full_stack(self, img):
        spec = FeatureSpec()
        whole = feature_stack(img, spec, device=DEVICE)
        rng = np.random.default_rng(3)
        flat = rng.choice(img.size, size=400, replace=False)
        got = sample_features(img, flat, spec, device=DEVICE).detach().cpu().numpy()
        ys, xs = np.divmod(flat, img.shape[1])
        assert np.array_equal(got, whole[:, ys, xs].T)

    def test_sample_features_accepts_yx_pairs(self, img):
        spec = FeatureSpec(sigmas=(1.0,), median=False, minimum=False,
                           maximum=False)
        yx = np.array([[5, 7], [40, 90], [0, 0]])
        a = sample_features(img, yx, spec, device=DEVICE).detach().cpu().numpy()
        flat = yx[:, 0] * img.shape[1] + yx[:, 1]
        b = sample_features(img, flat, spec, device=DEVICE).detach().cpu().numpy()
        assert np.array_equal(a, b)

    def test_sample_features_rejects_an_out_of_range_index(self, img):
        with pytest.raises(IndexError, match="outside"):
            sample_features(img, np.array([img.size]), device=DEVICE)

    def test_feature_tensor_lives_on_the_requested_device(self, img):
        t = feature_tensor(img[:32, :32], device=DEVICE)
        assert t.device.type == "cpu" and t.dtype.is_floating_point


# ── label store ──────────────────────────────────────────────────────────────

class TestLabelStore:
    def test_paint_and_read_back(self):
        store = LabelStore(frame_shape=(16, 20))
        mask = np.zeros((16, 20), bool)
        mask[2:5, 3:6] = True
        assert store.paint(0, mask, 0) == 9
        assert len(store) == 9
        assert store.labelled_frames() == [0]
        lm = store.label_map(0)
        assert (lm[2:5, 3:6] == 0).all()
        assert (lm[8:, :] == UNLABELLED).all()

    def test_labels_accumulate_across_frames(self):
        """Plan B3: paint on frame 0 and frame 400, both train one model."""
        store = LabelStore(frame_shape=(16, 20))
        store.paint_disc(0, 4, 4, 2, 0)
        store.paint_disc(400, 9, 9, 2, 1)
        assert store.labelled_frames() == [0, 400]
        assert len(store) == len(store.at(0)[0]) + len(store.at(400)[0])
        assert set(store.counts()) >= {0, 1}

    def test_repainting_a_pixel_changes_its_class(self):
        """Last write wins, which is what a brush does. A `np.unique` that kept
        the FIRST occurrence would silently ignore a correction."""
        store = LabelStore(frame_shape=(8, 8))
        store.paint(0, [(3, 3)], 0)
        store.paint(0, [(3, 3)], 1)
        assert len(store) == 1
        assert store.label_map(0)[3, 3] == 1

    def test_erase_removes_rather_than_reassigns(self):
        store = LabelStore(frame_shape=(8, 8))
        store.paint(0, [(1, 1), (1, 2)], 0)
        store.erase(0, [(1, 1)])
        assert len(store) == 1
        assert store.label_map(0)[1, 1] == UNLABELLED

    def test_erasing_everything_drops_the_frame(self):
        store = LabelStore(frame_shape=(8, 8))
        store.paint(0, [(1, 1)], 0)
        store.erase(0, [(1, 1)])
        assert store.labelled_frames() == []

    def test_paint_stroke_is_continuous(self):
        """The brush widget emits one sample per pointer frame, so a fast stroke
        jumps many pixels; dabbing only at the samples leaves a dotted line."""
        store = LabelStore(frame_shape=(32, 32))
        store.paint_stroke(0, [(5, 2), (5, 28)], 0, brush=1.0)
        row = store.label_map(0)[5]
        painted = np.flatnonzero(row != UNLABELLED)
        assert painted.min() <= 2 and painted.max() >= 28
        assert np.all(np.diff(painted) == 1), f"gaps in the stroke: {painted}"

    def test_paint_stroke_honours_brush_width(self):
        store = LabelStore(frame_shape=(32, 32))
        thin = store.paint_stroke(0, [(10, 5), (10, 25)], 0, brush=1.0)
        store.clear()
        fat = store.paint_stroke(0, [(10, 5), (10, 25)], 0, brush=7.0)
        assert fat > 3 * thin

    def test_out_of_frame_coordinates_are_dropped_not_wrapped(self):
        """A stroke running off the edge is normal; wrapping it would paint the
        opposite side of the image."""
        store = LabelStore(frame_shape=(10, 10))
        store.paint(0, [(-3, 4), (4, 40), (5, 5)], 0)
        assert len(store) == 1
        assert store.label_map(0)[5, 5] == 0

    def test_a_stroke_off_the_edge_paints_only_what_is_inside(self):
        store = LabelStore(frame_shape=(16, 16))
        store.paint_stroke(0, [(8, -6), (8, 6)], 0, brush=3.0)
        lm = store.label_map(0)
        # The stroke ends at x=6 with a radius-1.5 brush, so nothing past x=7.
        assert (lm[:, 8:] == UNLABELLED).all()
        assert (lm[8, 0:6] == 0).all(), "the in-frame part of the stroke is missing"

    def test_counts_lists_classes_with_no_pixels(self):
        """Which is how the caret shows a class is under-trained."""
        store = LabelStore(frame_shape=(8, 8))
        store.paint_disc(0, 4, 4, 2, 0)
        counts = store.counts()
        # 0 particle, 1 support film, 2 vacuum, 3 boundary.
        assert set(counts) == {0, 1, 2, 3}
        assert counts[1] == counts[2] == counts[3] == 0
        assert store.n_classes_used == 1

    def test_add_and_remove_classes(self):
        store = LabelStore(frame_shape=(8, 8))
        extra = store.add_class("beam stop", "#ff0000")
        assert extra.id == 4 and store.class_by_id(4).name == "beam stop"
        store.paint(0, [(1, 1)], 4)
        store.paint(0, [(2, 2)], 0)
        store.remove_class(4)
        assert 4 not in store.class_ids
        assert len(store) == 1, "removing a class must drop its pixels too"

    def test_removing_the_only_labelled_class_drops_the_frame(self):
        store = LabelStore(frame_shape=(8, 8))
        store.paint(0, [(1, 1)], 0)
        store.remove_class(0)
        assert store.labelled_frames() == []

    def test_unknown_class_raises(self):
        store = LabelStore(frame_shape=(8, 8))
        with pytest.raises(KeyError, match="no class with id"):
            store.paint(0, [(1, 1)], 99)

    def test_duplicate_class_ids_raise(self):
        with pytest.raises(ValueError, match="duplicate class ids"):
            LabelStore(frame_shape=(8, 8),
                       classes=[ScribbleClass(0, "a"), ScribbleClass(0, "b")])
        store = LabelStore(frame_shape=(8, 8))
        with pytest.raises(ValueError, match="already exists"):
            store.add_class("dup", id=0)

    def test_mask_of_the_wrong_shape_raises(self):
        store = LabelStore(frame_shape=(8, 8))
        with pytest.raises(ValueError, match="frame_shape"):
            store.paint(0, np.ones((4, 4), bool), 0)

    def test_round_trips_through_a_dict(self):
        store = LabelStore(frame_shape=(16, 24))
        store.paint_disc(0, 5, 5, 2, 0)
        store.paint_stroke(7, [(1, 1), (1, 20)], 1, brush=2.0)
        d = store.to_dict()
        assert json.loads(json.dumps(d)) == d, "not JSON-safe"
        back = LabelStore.from_dict(d)
        assert back.frame_shape == store.frame_shape
        assert back.labelled_frames() == store.labelled_frames()
        assert back.counts() == store.counts()
        for t in store.labelled_frames():
            assert np.array_equal(back.label_map(t), store.label_map(t))

    def test_clear_frame_and_clear(self):
        store = LabelStore(frame_shape=(8, 8))
        store.paint(0, [(1, 1)], 0)
        store.paint(1, [(2, 2)], 0)
        store.clear_frame(0)
        assert store.labelled_frames() == [1]
        store.clear()
        assert len(store) == 0


# ── the prompt-model bootstrap ───────────────────────────────────────────────

class TestMasksToLabels:
    """Plan §0.4: prompt masks become scribble labels with no painting at all."""

    @staticmethod
    def _disc(shape, cy, cx, r):
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

    def test_interior_is_particle_and_surroundings_are_background(self):
        m = self._disc((64, 64), 20, 20, 6)
        store = masks_to_labels(m, t=3)
        assert store.labelled_frames() == [3]
        lm = store.label_map(3)
        assert lm[20, 20] == 0, "mask centre is not the particle class"
        assert (lm[m] != 1).all(), "background was painted inside the mask"
        assert (lm == 1).sum() > 0, "no background ring"

    def test_the_boundary_itself_is_left_unlabelled(self):
        """Where the prompt model is least certain and where a particle's own soft
        edge lives — labelling it either way biases every instance's size."""
        m = self._disc((64, 64), 32, 32, 8)
        lm = masks_to_labels(m, gap=2, erode=1).label_map(0)
        yy, xx = np.mgrid[0:64, 0:64]
        r = np.sqrt((yy - 32) ** 2 + (xx - 32) ** 2)
        rim = (r > 8.0) & (r <= 9.0)
        assert (lm[rim] == UNLABELLED).all()

    def test_the_ring_never_covers_another_mask(self):
        """Otherwise particle A is taught as background for particle B."""
        a = self._disc((64, 64), 32, 24, 5)
        b = self._disc((64, 64), 32, 34, 5)
        lm = masks_to_labels(np.stack([a, b])).label_map(0)
        assert (lm[a] != 1).all() and (lm[b] != 1).all()

    def test_a_tiny_mask_survives_the_erosion(self):
        """Plan §0.9 applied here: a 3 px particle is exactly the object this
        feature exists for, and eroding it away would defeat the bootstrap."""
        m = np.zeros((32, 32), bool)
        m[16, 16] = True
        lm = masks_to_labels(m, erode=1).label_map(0)
        assert lm[16, 16] == 0

    def test_accumulates_into_an_existing_store(self):
        store = masks_to_labels(self._disc((64, 64), 20, 20, 5), t=0)
        n0 = len(store)
        masks_to_labels(self._disc((64, 64), 44, 44, 5), t=9, store=store)
        assert store.labelled_frames() == [0, 9]
        assert len(store) > n0

    def test_a_trained_model_finds_the_prompted_particles(self, movie, geom):
        """The end of the §0.4 handoff: masks in, dense segmentation out."""
        s, _gt = movie
        pos, radii, present, faint, shape = geom
        bright = [i for i in np.flatnonzero(present) if not faint[i]][:3]
        masks = np.stack([self._disc(shape, pos[i, 0], pos[i, 1], radii[i])
                          for i in bright])
        store = masks_to_labels(masks, t=FRAME_T)
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
        clf.fit(store, {FRAME_T: s.data[FRAME_T]})
        prob = clf.predict_proba(s.data[FRAME_T])
        assert all(hit(prob, pos, i) for i in bright)
        assert (prob > 0.5).mean() < 0.25, "foreground has swallowed the frame"

    @pytest.mark.parametrize("kw, match", [
        ({"gap": 8, "background_dilation": 4}, "background_dilation"),
        ({"particle_class": 99}, "no class with id"),
    ])
    def test_errors(self, kw, match):
        m = self._disc((32, 32), 16, 16, 4)
        with pytest.raises((ValueError, KeyError), match=match):
            masks_to_labels(m, **kw)

    def test_a_4d_input_raises(self):
        with pytest.raises(ValueError, match=r"\(h, w\) or \(n, h, w\)"):
            masks_to_labels(np.zeros((2, 2, 8, 8), bool))

    def test_a_store_of_the_wrong_shape_raises(self):
        store = LabelStore(frame_shape=(16, 16))
        with pytest.raises(ValueError, match="the store holds"):
            masks_to_labels(np.ones((32, 32), bool), store=store)


# ── training ─────────────────────────────────────────────────────────────────

class TestTraining:
    def test_report_describes_what_was_trained(self, trained, labels):
        rep = trained.report
        assert rep["n_pixels"] == len(labels)
        assert rep["n_channels"] == FeatureSpec().n_channels
        assert rep["labelled_frames"] == labels.labelled_frames()
        assert rep["train_accuracy"] > 0.95
        assert rep["featurise_s"] > 0 and rep["fit_s"] > 0
        assert set(rep["pixels_per_class"]) == {"0", "1"}

    def test_only_painted_classes_become_head_columns(self, trained):
        """The default store offers three classes; only two were painted, and a
        column that never sees a positive example would emit noise."""
        assert [c.id for c in trained.classes] == [0, 1]
        assert trained.particle_class_ids == [0]

    def test_multi_class_prediction(self, movie, geom):
        s, _gt = movie
        pos, radii, present, _faint, shape = geom
        store = paint_scribbles(geom)
        dark = np.zeros(shape, bool)
        dark[0:6, 30:90] = True
        store.paint(FRAME_T, dark & _clear_of_particles(shape, pos, radii,
                                                        present, 3.0), 2)
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
        rep = clf.fit(store, {FRAME_T: s.data[FRAME_T]})
        assert rep["n_classes"] == 3
        lbl = clf.predict_labels(s.data[FRAME_T])
        assert set(np.unique(lbl).tolist()) == {0, 1, 2}
        # The particle probability is still only the particle class.
        prob = clf.predict_proba(s.data[FRAME_T])
        assert prob.max() <= 1.0 and (prob > 0.5).mean() < 0.25

    def test_progress_is_reported_per_labelled_frame(self, movie, geom):
        s, _gt = movie
        store = paint_scribbles(geom)
        store.paint_disc(20, 30, 30, 3, 1)
        calls = []
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0, epochs=20)
        clf.fit(store, s.data, progress=lambda d, t: calls.append((d, t)))
        assert calls[-1] == (2, 2)
        assert all(t == 2 for _d, t in calls)

    def test_trains_from_a_hyperspy_signal_without_materialising_it(self, movie,
                                                                   geom):
        """`frames` goes through `drift.frames.frame_source`, which reads one
        frame at a time — the CLAUDE.md memory-safety rule."""
        s, _gt = movie
        store = paint_scribbles(geom)
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0, epochs=20)
        assert clf.fit(store, s)["n_pixels"] == len(store)

    def test_an_empty_store_raises(self, movie):
        s, _gt = movie
        clf = ScribbleClassifier(device=DEVICE)
        with pytest.raises(ValueError, match="nothing painted"):
            clf.fit(LabelStore(frame_shape=s.data[0].shape), s.data)

    def test_one_class_only_raises(self, movie):
        s, _gt = movie
        store = LabelStore(frame_shape=s.data[0].shape)
        store.paint_disc(0, 20, 20, 3, 0)
        clf = ScribbleClassifier(device=DEVICE)
        with pytest.raises(ValueError, match="only one class"):
            clf.fit(store, s.data)

    def test_a_frame_of_the_wrong_shape_raises(self, movie, geom):
        """A flat index means nothing without the shape; silently accepting a
        different one would scatter the training samples."""
        s, _gt = movie
        store = paint_scribbles(geom)
        clf = ScribbleClassifier(device=DEVICE)
        with pytest.raises(ValueError, match="label store holds"):
            clf.fit(store, {FRAME_T: s.data[FRAME_T][:64, :64]})

    def test_predicting_before_training_raises(self, movie):
        s, _gt = movie
        with pytest.raises(RuntimeError, match="not been trained"):
            ScribbleClassifier(device=DEVICE).predict_proba(s.data[0])

    def test_no_particle_class_is_a_clear_error(self, movie, geom):
        """A softmax over three backgrounds is a valid model with no foreground;
        say so instead of returning a zero map."""
        s, _gt = movie
        pos, radii, present, faint, shape = geom
        store = LabelStore(frame_shape=shape,
                           classes=[ScribbleClass(0, "film", particle=False),
                                    ScribbleClass(1, "vacuum", particle=False)])
        store.paint(FRAME_T, _clear_of_particles(shape, pos, radii, present, 3.0)
                    & (np.mgrid[0:shape[0], 0:shape[1]][0] < 20), 0)
        store.paint(FRAME_T, _clear_of_particles(shape, pos, radii, present, 3.0)
                    & (np.mgrid[0:shape[0], 0:shape[1]][0] > 80), 1)
        clf = ScribbleClassifier(device=DEVICE, epochs=20)
        clf.fit(store, {FRAME_T: s.data[FRAME_T]})
        with pytest.raises(RuntimeError, match="marked as a particle class"):
            clf.predict_proba(s.data[FRAME_T])

    def test_class_probabilities_sum_to_one_where_valid(self, movie, trained):
        s, _gt = movie
        p = trained.predict_class_proba(s.data[FRAME_T])
        assert np.abs(p.sum(axis=0) - 1.0).max() < 1e-4

    def test_segment_forwards_to_the_shared_instance_split(self, movie, trained,
                                                           geom):
        """Plan §0.2: this engine stops at a probability map; the instance stage
        is written once, in classical.split_instances."""
        s, _gt = movie
        pos, _radii, present, _faint, _shape = geom
        lab = trained.segment(s.data[FRAME_T], SegmentParams(min_size=8))
        assert lab.dtype == np.int32
        assert lab.max() >= int(present.sum()) - 1     # the merge pair may join
        for i in np.flatnonzero(present):
            assert hit(lab, pos, i), f"particle {i} missing from the instances"


# ── the acceptance gates ─────────────────────────────────────────────────────

class TestTheClassicalBaselineMissesThem:
    """Makes :class:`TestSensitivityGate` non-vacuous, in this file.

    ``test_particle_movie_fixture.py::TestSegmentationOnTheFixture`` already pins
    that the classical engine misses the faint probes at its default sensitivity;
    this repeats the measurement next to the gate it justifies, because a gate
    whose baseline lives in another file is one rename away from being vacuous.
    """

    def test_classical_finds_the_bright_particles(self, movie, geom):
        s, _gt = movie
        pos, _radii, present, faint, _shape = geom
        lab = segment_frame(s.data[FRAME_T],
                            SegmentParams(min_size=25, gaussian=1.0))
        want = np.flatnonzero(present & ~faint)
        assert all(hit(lab, pos, i) for i in want)

    def test_classical_finds_neither_faint_probe(self, movie, geom):
        s, _gt = movie
        pos, _radii, _present, faint, _shape = geom
        lab = segment_frame(s.data[FRAME_T],
                            SegmentParams(min_size=25, gaussian=1.0))
        found = [int(i) for i in np.flatnonzero(faint) if hit(lab, pos, i)]
        assert found == [], (
            f"the classical engine now finds faint probes {found}; the §0.9 gate "
            "below is no longer measuring anything and the fixture's "
            "faint_amplitude should be lowered")

    def test_a_looser_threshold_does_not_rescue_it(self, movie, geom):
        """Why §0.9 says the learned classifier is the primary path and threshold
        tuning is not: by the time the threshold is loose enough to include the
        faint probes it has merged the film into one giant region."""
        s, _gt = movie
        pos, _radii, _present, faint, _shape = geom
        rescued = False
        for sens in (0.6, 0.7, 0.8, 0.9, 1.0):
            lab = segment_frame(
                s.data[FRAME_T],
                SegmentParams(min_size=25, gaussian=1.0, sensitivity=sens))
            got = [int(i) for i in np.flatnonzero(faint) if hit(lab, pos, i)]
            if len(got) == 2:
                # Only counts as a rescue if the frame is still a segmentation.
                fg = float((lab > 0).mean())
                rescued = fg < 0.30
        assert not rescued, (
            "raising the classical sensitivity now finds both faint probes "
            "without flooding the frame — if that is a real improvement this "
            "test has served its purpose, but check deliberately")


def paint_seam(store, geom, t: int = FRAME_T, width: int = 2):
    """Add boundary strokes along the joins between touching particles.

    The seam BETWEEN two bodies, never the outline of a lone one. That
    distinction is the whole art of using this class and getting it wrong is
    silent: a head taught particle outlines shrinks every body and splits
    nothing, which measured as a MERGED touching pair and a 40% area loss on the
    fixture's merge frame.
    """
    from scipy import ndimage as ndi

    pos, radii, present, _faint, shape = geom
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    idx = list(np.flatnonzero(present))
    grown = [ndi.binary_dilation(
        ((yy - pos[i, 0]) ** 2 + (xx - pos[i, 1]) ** 2) <= radii[i] ** 2,
        iterations=width) for i in idx]
    seam = np.zeros((h, w), bool)
    for a in range(len(grown)):
        for b in range(a + 1, len(grown)):
            seam |= grown[a] & grown[b]
    if seam.any():
        store.paint(t, seam, 3)
    return store


class TestBoundaryClass:
    """The ilastik third class, and the route it unlocks.

    It is a **performance** feature: a taught boundary lets ``split_instances``
    skip the distance transform and the watershed — 1.78 s down to 0.33 s at
    4096². So these check both that it is wired up and that turning it on
    does not cost detection — a faster segmentation that finds fewer particles
    is a regression, not an optimisation (plan §0.9).
    """

    def test_the_default_class_set_has_one(self):
        classes = default_classes()
        edge = [c for c in classes if c.boundary]
        assert len(edge) == 1 and edge[0].name == "boundary"
        assert not edge[0].particle, (
            "a seam is not part of a body; counting it as foreground would glue "
            "back together exactly the particles it separates")

    def test_a_class_cannot_be_both_particle_and_boundary(self):
        with pytest.raises(ValueError, match="both particle and boundary"):
            ScribbleClass(0, "confused", particle=True, boundary=True)

    def test_round_trips_through_a_dict(self):
        c = ScribbleClass(3, "boundary", "#f38ba8", boundary=True)
        assert ScribbleClass.from_dict(c.to_dict()) == c

    def test_a_dict_from_before_the_class_existed_still_loads(self):
        """A session or a saved model written by an older build has no
        ``boundary`` key, and must come back as the particle/background-only
        setup it was — not fail, and not silently become a boundary."""
        old = {"id": 0, "name": "particle", "colour": "#f9a03f", "particle": True}
        c = ScribbleClass.from_dict(old)
        assert c.particle and not c.boundary

    def test_untrained_boundary_reports_none(self, trained, movie):
        """The default fixture paints no seam, so there is no boundary — and the
        answer must be None rather than an all-zero map. The two mean different
        things to the split: "use the watershed" versus "a boundary was taught
        and this frame has none", which would leave every touching pair merged.
        """
        s, _gt = movie
        assert not trained.has_boundary
        assert trained.boundary_class_ids == []
        fg, bnd = trained.predict_foreground_boundary(s.data[FRAME_T])
        assert bnd is None
        assert fg.shape == s.data[FRAME_T].shape

    def test_a_trained_boundary_is_reported_and_predicted(self, movie, geom):
        s, _gt = movie
        store = paint_seam(paint_scribbles(geom), geom)
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
        report = clf.fit(store, {FRAME_T: s.data[FRAME_T]})
        assert report["has_boundary"] is True
        assert clf.has_boundary and clf.boundary_class_ids == [3]
        _fg, bnd = clf.predict_foreground_boundary(s.data[FRAME_T])
        assert bnd is not None and bnd.shape == tuple(geom[4])
        assert (bnd > 0.5).any(), "a trained boundary class predicted nothing"

    def test_segment_takes_the_boundary_route_when_one_is_trained(
            self, movie, geom, monkeypatch):
        """The wizard calls ``segment``; this is what makes the fast route
        automatic without the caret having to know about it."""
        from skimage import segmentation as skseg

        s, _gt = movie
        store = paint_seam(paint_scribbles(geom), geom)
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
        clf.fit(store, {FRAME_T: s.data[FRAME_T]})

        called = []
        monkeypatch.setattr(skseg, "watershed",
                            lambda *a, **k: called.append("watershed"))
        clf.segment(s.data[FRAME_T], SegmentParams(min_size=10))
        assert called == [], "segment ran the watershed despite a trained boundary"

    def test_segment_still_uses_the_watershed_without_one(self, trained, movie,
                                                          monkeypatch):
        """A user who never paints a boundary must not silently get worse
        splitting — the fallback is the whole safety of making this automatic."""
        from skimage import segmentation as skseg

        s, _gt = movie
        real = skseg.watershed
        called = []
        monkeypatch.setattr(skseg, "watershed", lambda *a, **k: (
            called.append("watershed"), real(*a, **k))[1])
        trained.segment(s.data[FRAME_T], SegmentParams(min_size=10))
        assert called == ["watershed"]

    def test_the_faint_probes_survive_the_boundary_route(self, movie, geom):
        """**Plan §0.9 on the new path.** The boundary class exists to make the
        split cheap; if it costs a faint detection it is not worth having."""
        s, _gt = movie
        pos, _radii, _present, faint, _shape = geom
        store = paint_seam(paint_scribbles(geom), geom)
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
        clf.fit(store, {FRAME_T: s.data[FRAME_T]})
        labels = clf.segment(s.data[FRAME_T], SegmentParams(min_size=5))
        missed = [int(i) for i in np.flatnonzero(faint)
                  if not hit(labels, pos, i)]
        assert not missed, f"the boundary route lost faint probe(s) {missed}"

    def test_the_boundary_route_agrees_with_the_watershed_on_the_fixture(
            self, movie, geom):
        """Same classifier, both routes: the count and the median area must
        agree, or the speed is not worth having."""
        s, _gt = movie
        store = paint_seam(paint_scribbles(geom), geom)
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
        clf.fit(store, {FRAME_T: s.data[FRAME_T]})
        p = SegmentParams(min_size=5)
        fg, bnd = clf.predict_foreground_boundary(s.data[FRAME_T])
        by_boundary = split_instances(fg, p, boundary=bnd)
        by_watershed = split_instances(fg, p)

        def areas(lab):
            c = np.bincount(lab.ravel())[1:]
            return np.sort(c[c > 0])

        assert int(by_boundary.max()) == int(by_watershed.max()), (
            f"boundary found {by_boundary.max()} particles, watershed "
            f"{by_watershed.max()}")
        a_b, a_w = areas(by_boundary), areas(by_watershed)
        assert abs(np.median(a_b) - np.median(a_w)) / np.median(a_w) < 0.15, (
            f"median area {np.median(a_b)} vs watershed {np.median(a_w)}")


class TestSensitivityGate:
    """**Plan §0.9 — the headline gate for the whole feature.**

    Trained on eleven scribbles (four dabs on bright particles, one dab on the
    smaller faint probe, four background sweeps and four boundary rings), the
    classifier must find BOTH faint probes and keep every bright one.

    Note what the scribbles do and do not contain. Faint probe **8** (r=3) is
    painted; probe **7** (r=4) is not, so its detection is genuinely held out.
    :class:`TestBrightOnlyLabelsDoNotGeneralise` records the boundary of the
    claim: with *no* faint example at all, neither this head nor the RandomForest
    reference finds them, so at least one faint scribble is required. That is a
    property of the problem — an 8x contrast extrapolation — not of the head.
    """

    def test_finds_both_faint_probes(self, movie, geom, proba):
        _s, _gt = movie
        pos, _radii, _present, faint, _shape = geom
        scores = {int(i): float(proba[int(round(pos[i, 0])),
                                     int(round(pos[i, 1]))])
                  for i in np.flatnonzero(faint)}
        assert all(v > 0.5 for v in scores.values()), (
            f"faint probes not found: {scores}")

    def test_the_held_out_faint_probe_is_found(self, geom, proba):
        """Probe 7 is never painted. This is the part that is not memorisation."""
        pos = geom[0]
        assert proba[int(round(pos[7, 0])), int(round(pos[7, 1]))] > 0.5

    def test_every_bright_particle_is_still_found(self, geom, proba):
        """Sensitivity that costs the easy detections is not sensitivity."""
        pos, _radii, present, faint, _shape = geom
        for i in np.flatnonzero(present & ~faint):
            assert hit(proba, pos, i), f"lost bright particle {i}"

    def test_the_foreground_has_not_swallowed_the_frame(self, geom, proba):
        """A classifier that calls everything a particle would pass every test
        above. The nine discs cover ~9% of the frame."""
        fraction = float((proba > 0.5).mean())
        assert 0.05 < fraction < 0.20, (
            f"foreground is {fraction:.1%} of the frame; the true particle "
            "coverage is ~9%")

    def test_the_instance_count_is_right(self, movie, geom, proba):
        _s, _gt = movie
        present = geom[2]
        lab = split_instances(proba, SegmentParams(min_size=8))
        assert int(present.sum()) - 1 <= lab.max() <= int(present.sum()) + 1, (
            f"found {lab.max()} instances, expected ~{int(present.sum())}")

    def test_it_generalises_to_frames_it_never_saw(self, movie, trained):
        """Labels came from t=12 alone. Every other frame has different drift,
        different particles present, and — at t=8 onwards — a nucleated one."""
        s, gt = movie
        for t in (0, 6, 18, int(gt["n_frames"]) - 1):
            prob = trained.predict_proba(s.data[t])
            pos, _radii, present = particle_truth_at(gt, t)
            faint = np.asarray(gt["p_faint"], bool)
            h, w = s.data[t].shape
            missed = []
            for i in np.flatnonzero(present):
                cy, cx = int(round(pos[i, 0])), int(round(pos[i, 1]))
                if not (0 <= cy < h and 0 <= cx < w):
                    continue                      # drifted out of frame
                if prob[cy, cx] <= 0.5:
                    missed.append((int(i), bool(faint[i])))
            assert not missed, f"frame {t}: missed {missed}"

    def test_a_coarse_stack_undersizes_the_small_particles(self, movie, geom,
                                                          labels):
        """What the fine scales actually buy — and it is NOT detection.

        Plan §0.9 says small-object detection needs the fine scales and forbids
        coarsening "without a documented sensitivity measurement". This is that
        measurement, and it corrects the guess: a coarse ``(4, 8)`` stack still
        *finds* both faint probes. What it loses is their size. Measured mean
        absolute error in recovered radius over the isolated particles:

            (0.5, 1, 2, 4, 8)  13 %      (2, 4, 8)  20 %      (4, 8)  26 %

        A particle found and then measured 44% too small (the r=3 probe on the
        ``(4, 8)`` stack) is worse than an honest miss, because it silently enters
        the size distribution the whole feature exists to produce.
        """
        from spyde.particles import measure_frame
        from spyde.signals.particles import COL

        s, gt = movie
        pos, radii, present, _faint, _shape = geom
        merge = tuple(int(v) for v in gt["merge_pair"])

        def radius_error(sigmas):
            clf = ScribbleClassifier(FeatureSpec(sigmas=sigmas), device=DEVICE,
                                     seed=0)
            clf.fit(labels, {FRAME_T: s.data[FRAME_T]})
            lab = split_instances(clf.predict_proba(s.data[FRAME_T]),
                                 SegmentParams(min_size=8))
            rows, _c = measure_frame(lab, s.data[FRAME_T], t=FRAME_T, scale=1.0)
            errs = []
            for i in np.flatnonzero(present):
                if i in merge:
                    continue                      # a merged pair is not one disc
                lbl = lab[int(round(pos[i, 0])), int(round(pos[i, 1]))]
                row = rows[rows[:, COL["label"]] == lbl]
                assert lbl and len(row), f"sigmas={sigmas}: lost particle {i}"
                got = float(row[0, COL["equiv_diameter"]]) / 2.0
                errs.append(abs(got - radii[i]) / radii[i])
            return float(np.mean(errs))

        fine = radius_error(DEFAULT_SIGMAS)
        coarse = radius_error((4.0, 8.0))
        assert fine < 0.20, f"default stack radius error {fine:.1%}"
        assert coarse > 1.5 * fine, (
            f"a coarse (4, 8) stack now measures radii as well as the default one "
            f"({coarse:.1%} vs {fine:.1%}) — if real, the sigma floor could be "
            "raised, but re-measure deliberately before doing it")

    def test_the_nucleating_particle_appears_at_its_known_frame(self, movie,
                                                               trained):
        """§0.9's actual motivation: missing a particle's first appearance
        destroys the nucleation event."""
        s, gt = movie
        i, nuc = int(gt["nucleation_index"]), int(gt["nucleation_frame"])
        before = trained.predict_proba(s.data[nuc - 1])
        after = trained.predict_proba(s.data[nuc])
        pos_b = particle_truth_at(gt, nuc - 1)[0]
        pos_a = particle_truth_at(gt, nuc)[0]
        assert before[int(round(pos_b[i, 0])), int(round(pos_b[i, 1]))] < 0.5
        assert after[int(round(pos_a[i, 0])), int(round(pos_a[i, 1]))] > 0.5


class TestBrightOnlyLabelsAreNotEnough:
    """Records the boundary of the §0.9 claim, and that it is not the head's fault.

    Trained on bright particles only — no faint example anywhere in the labels —
    the head finds **1 of the 2** faint probes (0.67 for the r=3 one, 0.013 for the
    r=4 one) and the sklearn RandomForest reference finds **0 of 2** (both exactly
    0.0). Adding a *single seven-pixel dab* on the r=3 probe takes the head to
    2 of 2 at 0.98 and 1.00.

    **Those exact counts depend on where the background scribbles land**, so the
    assertion below is the robust `< 2` rather than `== 1`. An independent probe
    with different background dabs got **0 of 2** bright-only and 1 of 2 with the
    dab — same conclusion, different numbers. Read the figures above as one
    observation, not as a constant; the claim being pinned is only that bright-only
    labels are *not sufficient*.

    Two things worth having in the record:

    * The plan's §0.9 gate holds as written, with a clarification: "trained on a
      few scribbles" must include **at least one faint example**. That is what a
      user does anyway — they paint what they can see, and they can see these —
      and it is why the wizard's class list shows per-class pixel counts (plan
      B7): under-training a class is the failure mode, and the counts are how you
      notice.
    * The forest's 0.0 is not noise, it is structural. A tree ensemble cannot
      predict outside the leaves its training data reached, so a contrast 8x below
      anything it was shown is simply unreachable; the MLP extrapolates its
      decision boundary and gets one of the two for free. So for *sensitivity*
      specifically — which §0.9 makes the priority — the shipped head is better
      than the reference it is checked against, not merely faster.
    """

    @pytest.fixture(scope="class")
    def bright_only(self, movie, geom):
        s, _gt = movie
        store = paint_scribbles(geom, include_faint=())
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
        clf.fit(store, {FRAME_T: s.data[FRAME_T]})
        return store, clf, clf.predict_proba(s.data[FRAME_T])

    def test_the_head_does_not_find_both(self, geom, bright_only):
        pos, _radii, _present, faint, _shape = geom
        _store, _clf, prob = bright_only
        scores = {int(i): float(prob[int(round(pos[i, 0])),
                                    int(round(pos[i, 1]))])
                  for i in np.flatnonzero(faint)}
        found = [i for i, v in scores.items() if v > 0.5]
        assert len(found) < 2, (
            f"bright-only labels now generalise to BOTH faint probes ({scores}) — "
            "if that is real, TestSensitivityGate can drop its faint scribble and "
            "become a much stronger claim; check deliberately")

    def test_the_bright_particles_are_still_all_found(self, geom, bright_only):
        pos, _radii, present, faint, _shape = geom
        _store, _clf, prob = bright_only
        assert all(hit(prob, pos, i) for i in np.flatnonzero(present & ~faint))

    def test_the_random_forest_reference_finds_neither(self, movie, geom,
                                                       bright_only):
        """The forest cannot extrapolate past its leaves; the MLP can. This is the
        one place the shipped head beats its own reference."""
        s, _gt = movie
        pos, _radii, _present, faint, _shape = geom
        store, _clf, _prob = bright_only
        _rf, predict = random_forest_reference(
            store, {FRAME_T: s.data[FRAME_T]}, FeatureSpec(), device=DEVICE)
        rp = predict(s.data[FRAME_T])
        assert max(float(rp[int(round(pos[i, 0])), int(round(pos[i, 1]))])
                   for i in np.flatnonzero(faint)) < 0.5

    def test_one_extra_dab_is_all_it_takes(self, movie, geom, bright_only):
        """The interaction this feature is actually for: seven more labelled
        pixels, one retrain, both probes found."""
        s, _gt = movie
        pos, _radii, _present, faint, _shape = geom
        store, _clf, _prob = bright_only
        with_faint = paint_scribbles(geom)
        assert 0 < len(with_faint) - len(store) < 20, (
            "the extra scribble is no longer a single small dab, so this stopped "
            "being a statement about how little labelling is needed")
        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
        clf.fit(with_faint, {FRAME_T: s.data[FRAME_T]})
        prob = clf.predict_proba(s.data[FRAME_T])
        assert all(hit(prob, pos, i) for i in np.flatnonzero(faint))


class TestRandomForestParity:
    """**Plan B3's acceptance gate**: agreement with the reference implementation.

    ``sklearn.ensemble.RandomForestClassifier`` is what ParticleSpy and ilastik
    use. Both sides read the *identical* channels through the *identical* sampler
    (``random_forest_reference`` lives in ``scribble.py`` for exactly that
    reason), so a disagreement is about the head and nothing else.

    Measured IoU of the two foreground masks on the fixture: **0.94**. The gate is
    set at 0.75 — the margin is deliberate, because the objects are small discs
    whose IoU is dominated by a one-pixel boundary difference, and a tighter bar
    would fail on an epoch-count change that means nothing (0.94 at 300 epochs,
    0.90 at 200, 0.86 at 500).
    """

    @pytest.fixture(scope="class")
    def forest(self, movie, labels):
        s, _gt = movie
        _rf, predict = random_forest_reference(
            labels, {FRAME_T: s.data[FRAME_T]}, FeatureSpec(), device=DEVICE)
        return predict(s.data[FRAME_T])

    def test_foreground_masks_agree_by_iou(self, proba, forest):
        a, b = proba > 0.5, forest > 0.5
        iou = float((a & b).sum()) / max(1, int((a | b).sum()))
        assert iou > 0.75, (
            f"IoU {iou:.3f} against the RandomForest reference on identical "
            "labels and identical features")

    def test_both_agree_on_every_ground_truth_particle(self, geom, proba, forest):
        pos, _radii, present, _faint, _shape = geom
        for i in np.flatnonzero(present):
            assert hit(proba, pos, i) == hit(forest, pos, i), (
                f"the head and the forest disagree about particle {i}")

    def test_they_cover_a_similar_fraction_of_the_frame(self, proba, forest):
        """A high IoU with wildly different coverage would mean one is a subset of
        the other, which is a different kind of agreement."""
        fa, fb = float((proba > 0.5).mean()), float((forest > 0.5).mean())
        assert abs(fa - fb) < 0.05, f"coverage {fa:.3f} vs {fb:.3f}"


class TestNaNBorder:
    """**Plan trap 2 / gate A7**: no particle is ever detected in NaN padding.

    A drift-corrected frame keeps its full size with uncovered pixels set to NaN
    (``spyde.drift.warp``). Segmentation that ignores that finds a large "particle"
    along the edge, which then nucleates a spurious track — the single most likely
    integration bug in this feature.
    """

    @pytest.fixture(scope="class")
    def warped(self, movie):
        s, _gt = movie
        out = shift_frame(s.data[FRAME_T], (7.0, -5.0))     # NaN on two edges
        assert not np.isfinite(out).all(), "no NaN padding to test"
        return out

    def test_probability_is_exactly_zero_in_the_padding(self, trained, warped):
        prob = trained.predict_proba(warped)
        bad = ~np.isfinite(warped)
        assert float(prob[bad].max()) == 0.0

    def test_no_instance_lands_in_the_padding(self, trained, warped):
        prob = trained.predict_proba(warped)
        lab = split_instances(prob, SegmentParams(min_size=8))
        assert set(np.unique(lab[~np.isfinite(warped)]).tolist()) == {0}

    def test_predict_labels_reports_unlabelled_not_a_class(self, trained, warped):
        """-1, not the background class: an invalid pixel is not a measurement."""
        lbl = trained.predict_labels(warped)
        bad = ~np.isfinite(warped)
        assert set(np.unique(lbl[bad]).tolist()) == {UNLABELLED}
        assert (lbl[~bad] != UNLABELLED).all()

    def test_real_data_next_to_the_padding_still_classifies(self, trained, warped,
                                                            movie):
        """The other half of the contract: NaN must not erase a band of real data.
        A filter that propagated NaN outward would blank ~33 px (the halo) inside
        the border, which is most of a 96-row frame."""
        prob = trained.predict_proba(warped)
        good = np.isfinite(warped)
        assert float(prob[good].max()) > 0.5, "everything valid came back empty"
        assert float((prob[good] > 0.5).mean()) > 0.02

    def test_the_shifted_particles_are_found_at_their_shifted_positions(
            self, movie, trained, warped, geom):
        pos, _radii, present, _faint, _shape = geom
        prob = trained.predict_proba(warped)
        h, w = warped.shape
        found = 0
        for i in np.flatnonzero(present):
            cy, cx = int(round(pos[i, 0] + 7.0)), int(round(pos[i, 1] - 5.0))
            if 0 <= cy < h and 0 <= cx < w and np.isfinite(warped[cy, cx]):
                found += prob[cy, cx] > 0.5
        assert found >= 6, f"only {found} particles survived the warp"


class TestDeterminism:
    """Same seed + same labels → the same model. A user who changed nothing must
    not see the segmentation change."""

    def test_two_fits_with_the_same_seed_are_identical(self, movie, labels):
        s, _gt = movie
        outs = []
        for _ in range(2):
            clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0,
                                     epochs=60)
            clf.fit(labels, {FRAME_T: s.data[FRAME_T]})
            outs.append(clf.predict_proba(s.data[FRAME_T]))
        assert np.array_equal(*outs)

    def test_the_seed_actually_does_something(self, movie, labels):
        s, _gt = movie
        outs = []
        for seed in (0, 1):
            clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=seed,
                                     epochs=60)
            clf.fit(labels, {FRAME_T: s.data[FRAME_T]})
            outs.append(clf.predict_proba(s.data[FRAME_T]))
        assert not np.array_equal(*outs)

    def test_initialisation_does_not_consume_the_global_rng(self, movie, labels):
        """The global torch stream is shared with the neural detector and every
        dask worker; drawing from it would make 'same seed' depend on what else
        ran first."""
        import torch
        torch.manual_seed(1234)
        before = torch.randn(3)
        torch.manual_seed(1234)
        ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0, epochs=5).fit(
            labels, {FRAME_T: movie[0].data[FRAME_T]})
        after = torch.randn(3)
        assert torch.equal(before, after)


class TestSaveLoad:
    """A recipe is the spec *and* the weights *and* the standardisation; any of
    them alone predicts nonsense, so they live in one file."""

    def test_round_trip_predicts_identically(self, movie, trained, tmp_path):
        s, _gt = movie
        path = str(tmp_path / "model.npz")
        trained.save(path)
        back = ScribbleClassifier.load(path, device=DEVICE)
        assert np.array_equal(back.predict_proba(s.data[FRAME_T]),
                              trained.predict_proba(s.data[FRAME_T]))

    def test_carries_the_spec_and_the_classes(self, trained, tmp_path):
        path = str(tmp_path / "model.npz")
        trained.save(path)
        back = ScribbleClassifier.load(path, device=DEVICE)
        assert back.spec == trained.spec
        assert [c.to_dict() for c in back.classes] == \
               [c.to_dict() for c in trained.classes]
        assert back.hidden == trained.hidden and back.seed == trained.seed

    def test_loads_without_pickle(self, trained, tmp_path):
        """A model file must not be able to execute code on load."""
        path = str(tmp_path / "model.npz")
        trained.save(path)
        with np.load(path, allow_pickle=False) as z:
            assert "meta" in z.files and "feature_mean" in z.files

    def test_a_recipe_applies_to_a_differently_scaled_dataset(self, movie,
                                                              trained, tmp_path):
        """What ``normalize_frame`` buys: the same sample recorded as counts
        rather than as a normalised float still segments."""
        s, _gt = movie
        path = str(tmp_path / "model.npz")
        trained.save(path)
        back = ScribbleClassifier.load(path, device=DEVICE)
        rescaled = s.data[FRAME_T] * 4096.0 + 100.0
        a = back.predict_proba(s.data[FRAME_T]) > 0.5
        b = back.predict_proba(rescaled) > 0.5
        iou = float((a & b).sum()) / max(1, int((a | b).sum()))
        assert iou > 0.95, f"IoU across an intensity rescale is only {iou:.3f}"

    def test_saving_before_training_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not been trained"):
            ScribbleClassifier(device=DEVICE).save(str(tmp_path / "x.npz"))

    def test_a_future_format_version_is_refused(self, trained, tmp_path):
        path = tmp_path / "model.npz"
        trained.save(str(path))
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays.pop("meta").item()))
        meta["format_version"] = 99
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        with pytest.raises(ValueError, match="format version"):
            ScribbleClassifier.load(str(path), device=DEVICE)

    def test_a_spec_that_no_longer_matches_the_weights_is_refused(self, trained,
                                                                 tmp_path):
        """The failure this guards is silent otherwise: a channel count mismatch
        would only show as a wrong segmentation."""
        path = tmp_path / "model.npz"
        trained.save(str(path))
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays.pop("meta").item()))
        meta["spec"]["median"] = False
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
        with pytest.raises(ValueError, match="come apart"):
            ScribbleClassifier.load(str(path), device=DEVICE)


class TestInteractionBudget:
    """**Plan B3's hard budget**: train + apply to the visible frame under ~1 s.

    Measured here on the 96x112 fixture, CPU, default 36-channel spec: **0.49 s**
    total = 14 ms featurise + 457 ms for the 300 Adam steps + 20 ms to apply. The
    fit is dominated by per-step dispatch overhead rather than arithmetic —
    1.5 ms/step at any torch thread count from 1 to 24 — so it is a *fixed* cost,
    independent of frame size and of how much was painted.

    The bar here is 2.0 s, not 1.0 s: this is a shared CI box and a *timing*
    assertion at the measured value is a flake generator. The number that matters
    is the one above; this test exists to catch an order-of-magnitude regression.
    """

    def test_train_plus_apply_is_interactive(self, movie, labels):
        s, _gt = movie
        frame = s.data[FRAME_T]
        # Warm torch first: the FIRST op in a fresh process pays a one-time ~1 s
        # of thread-pool and kernel-selection cost that no user ever sees twice.
        ScribbleClassifier(FeatureSpec(), device=DEVICE, epochs=5).fit(
            labels, {FRAME_T: frame})

        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, seed=0)
        t0 = time.perf_counter()
        clf.fit(labels, {FRAME_T: frame})
        t1 = time.perf_counter()
        clf.predict_proba(frame)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, (
            f"train+apply took {elapsed:.2f} s (fit {t1 - t0:.2f} s); the plan's "
            "budget is ~1 s and the measured value on this box is 0.49 s")

    def test_applying_to_one_frame_is_a_fraction_of_the_budget(self, movie,
                                                              trained):
        """Scrubbing the navigator re-applies without re-training, so this is the
        number that has to stay small."""
        s, _gt = movie
        trained.predict_proba(s.data[0])              # warm
        t0 = time.perf_counter()
        trained.predict_proba(s.data[1])
        assert time.perf_counter() - t0 < 0.5


# ── the MPS device-serialisation contract ────────────────────────────────────

class _FakeDev:
    """Stands in for ``torch.device('mps')``; only ``.type`` is consulted. Same
    double as ``test_device_lock.py`` uses, so the contract is exercised on any
    machine without touching Metal."""

    def __init__(self, type_="mps"):
        self.type = type_

    def __str__(self):
        return self.type


def _lock_is_held_by_me() -> bool:
    from spyde.device_lock import DEVICE_LOCK
    got = []

    def probe():
        got.append(DEVICE_LOCK.acquire(blocking=False))
        if got[-1]:
            DEVICE_LOCK.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return not got[0]


class _LockSpy:
    """A stand-in for ``accelerator_lock`` that records entries and nesting depth.

    Two things need pinning and neither is visible from the real lock on a CPU box
    (where it is a null context): that the block is entered **with the right
    device**, and that every device submission happens **inside** it. So the spy
    records the devices it was called with and exposes :attr:`inside`, which the
    patched call sites report. The real-lock behaviour on a fake MPS device is
    pinned separately in ``test_device_lock.py``.
    """

    def __init__(self):
        self.devices: list = []
        self.depth = 0

    def __call__(self, device=None, **kw):
        import contextlib

        @contextlib.contextmanager
        def ctx():
            self.devices.append(device)
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1

        return ctx()

    @property
    def inside(self) -> bool:
        return self.depth > 0


class TestDeviceLock:
    """Every torch call site in these two modules takes ``accelerator_lock``.

    A lock only works if EVERY participant takes it — the last crash of this class
    existed because one path skipped it (CLAUDE.md § GPU Computing). The shared
    lock's *identity* and the real-lock behaviour are pinned in
    ``test_device_lock.py``; these tests pin the *nesting*, i.e. that no device
    submission escapes the block.
    """

    def test_feature_bands_hold_the_real_lock_on_mps(self, monkeypatch):
        """``map_feature_bands`` is the ONE torch entry point in features.py —
        ``feature_tensor``, ``feature_stack`` and ``sample_features`` all go
        through it — so this one acquisition covers the whole module."""
        held = []
        monkeypatch.setattr(feat, "_band_stack",
                            lambda *a, **k: held.append(_lock_is_held_by_me()))
        map_feature_bands(np.zeros((8, 8), np.float32), FeatureSpec(),
                          device=_FakeDev(), fn=lambda *a: None)
        assert held == [True], "the feature stack ran unserialised on MPS"
        assert not _lock_is_held_by_me(), "lock leaked"

    def test_no_lock_off_mps(self):
        """CUDA concurrency is a deliberate throughput win; serialising it would
        be a pure regression."""
        held = []
        map_feature_bands(np.zeros((8, 8), np.float32),
                          FeatureSpec(sigmas=(1.0,), median=False, minimum=False,
                                      maximum=False),
                          device=DEVICE,
                          fn=lambda *a: held.append(_lock_is_held_by_me()))
        assert held and not any(held)

    @pytest.mark.parametrize("call", [
        lambda img: feature_tensor(img, device=DEVICE),
        lambda img: feature_stack(img, device=DEVICE),
        lambda img: sample_features(img, np.array([0, 5]), device=DEVICE),
    ])
    def test_every_features_entry_point_takes_the_lock(self, call, monkeypatch):
        spy = _LockSpy()
        monkeypatch.setattr(feat, "accelerator_lock", spy)
        call(np.zeros((16, 16), np.float32))
        assert spy.devices and all(d is DEVICE for d in spy.devices)
        assert spy.depth == 0, "lock not released"

    def test_the_fit_takes_the_lock_around_every_submission(self, movie, labels,
                                                            monkeypatch):
        """The featurise, the optimiser loop and the ``.to(device)`` inside
        ``_build_mlp`` — a cold head load is the specific hole CLAUDE.md names."""
        from spyde.particles import scribble as scr

        s, _gt = movie
        spy = _LockSpy()
        inside = []
        monkeypatch.setattr(scr, "accelerator_lock", spy)
        for name in ("sample_features", "_build_mlp"):
            real = getattr(scr, name)
            monkeypatch.setattr(scr, name, lambda *a, _r=real, **k: (
                inside.append(spy.inside), _r(*a, **k))[1])

        clf = ScribbleClassifier(FeatureSpec(), device=DEVICE, epochs=5)
        clf.fit(labels, {FRAME_T: s.data[FRAME_T]})
        assert len(inside) >= 2 and all(inside), (
            "some of the fit's device work ran outside the lock")
        assert spy.devices == [clf.device], (
            f"expected ONE acquisition with the fit's device; got {spy.devices}")
        assert spy.depth == 0

    def test_predict_takes_the_lock_around_every_submission(self, movie, trained,
                                                            monkeypatch):
        """The interactive path — it fires on every navigator move, which is
        exactly the concurrency that took the backend down last time."""
        from spyde.particles import scribble as scr

        s, _gt = movie
        spy = _LockSpy()
        inside = []
        real = scr.map_feature_bands
        monkeypatch.setattr(scr, "accelerator_lock", spy)
        monkeypatch.setattr(scr, "map_feature_bands", lambda *a, **k: (
            inside.append(spy.inside), real(*a, **k))[1])
        trained.predict_class_proba(s.data[FRAME_T])
        assert inside == [True]
        assert spy.devices == [trained.device]
        assert spy.depth == 0

    @pytest.mark.parametrize("call", [
        "predict_foreground_boundary", "predict_proba",
        "predict_boundary_proba", "segment",
    ])
    def test_every_boundary_accessor_takes_the_lock(self, movie, trained, call,
                                                    monkeypatch):
        """The boundary route added three new public doors onto the device.

        A lock only works if every participant takes it, and the last crash of
        this class happened precisely because a newly added entry point reached
        the device through an existing helper and nobody re-checked that the
        helper's lock covered it. So all four doors are pinned, not just the one
        the wizard happens to call today — and ONE acquisition each, because
        reading two maps out of one softmax must not featurise twice.
        """
        from spyde.particles import scribble as scr

        s, _gt = movie
        spy = _LockSpy()
        monkeypatch.setattr(scr, "accelerator_lock", spy)
        getattr(trained, call)(s.data[FRAME_T])
        assert spy.devices == [trained.device], (
            f"{call} took {len(spy.devices)} acquisitions; expected exactly one")
        assert spy.depth == 0, f"{call} leaked the lock"

    def test_save_and_load_take_the_lock(self, trained, tmp_path, monkeypatch):
        """Both move tensors across the host/device boundary, which is a
        submission — ``save`` reading weights back and ``load`` pushing them out."""
        from spyde.particles import scribble as scr

        path = str(tmp_path / "m.npz")
        spy = _LockSpy()
        monkeypatch.setattr(scr, "accelerator_lock", spy)
        trained.save(path)
        assert spy.devices == [trained.device]
        spy.devices.clear()
        ScribbleClassifier.load(path, device=DEVICE)
        assert len(spy.devices) == 1
        assert spy.depth == 0

    def test_the_random_forest_reference_takes_the_lock_too(self, movie, labels,
                                                            monkeypatch):
        """It is a test helper, but it still submits torch work from whatever
        thread calls it, and an unlocked path is an unlocked path."""
        from spyde.particles import scribble as scr

        s, _gt = movie
        spy = _LockSpy()
        monkeypatch.setattr(scr, "accelerator_lock", spy)
        _rf, predict = random_forest_reference(
            labels, {FRAME_T: s.data[FRAME_T]},
            FeatureSpec(sigmas=(2.0,), median=False, minimum=False,
                        maximum=False),
            n_estimators=4, device=DEVICE)
        predict(s.data[FRAME_T])
        assert len(spy.devices) == 2 and all(d is DEVICE for d in spy.devices)
        assert spy.depth == 0

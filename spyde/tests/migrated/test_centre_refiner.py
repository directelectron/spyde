"""The neural method's centre stage (``spyde.models.centre_refine``).

After detection, every disk is cut out of the raw frame at native resolution and
re-placed by a refiner: the mask centroid, or a refiner network from the model
registry. These tests pin the refiner interface, the mask centroid's accuracy on
unevenly filled disks, the registry's detector/refiner split, and that the batch
and the single-frame preview place every disk identically.

Everything in-process uses a stand-in detector (no torch forward). The real
network runs in ONE subprocess (``RESULT_JSON <mode> {...}``, then
``os._exit(0)``), as in ``test_vector_confidence.py``: torch teardown can
segfault inside the pytest process on Windows.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erfc

from spyde.models import centre_refine
from spyde.models.centre_refine import (
    CENTRE_DECODE, CENTRE_MASK_CENTROID, MaskCentroidRefiner, extract_crops,
    refine_centres, refiner_for,
)


def uneven_disks(shape, centres, radius, rng, gradient=0.3, lobe=0.6, amplitude=150.0,
                 background=10.0):
    """A frame of disks with a soft edge and an uneven fill — a tilt across the
    disk plus an off-centre bright lobe, pointing a different way per disk (the
    dynamical-scattering signature) — under Poisson noise. The truth is the
    outline centre."""
    rows, columns = np.mgrid[:shape[0], :shape[1]].astype(np.float64)
    frame = np.full(shape, background)
    for y, x in centres:
        distance = np.hypot(rows - y, columns - x)
        edge = 0.5 * erfc((distance - radius) / (np.sqrt(2) * 0.6))
        angle = rng.uniform(0, 2 * np.pi)
        along = ((columns - x) * np.cos(angle) + (rows - y) * np.sin(angle)) / radius
        bright = np.exp(-((columns - x - 0.45 * radius * np.cos(angle)) ** 2
                          + (rows - y - 0.45 * radius * np.sin(angle)) ** 2)
                        / (2 * (0.35 * radius) ** 2))
        fill = np.clip(1 + gradient * along, 0, None) + lobe * bright
        frame += amplitude * edge * fill
    return rng.poisson(frame).astype(np.float32)


def disk_lattice(size, radius, rng, spacing=None):
    spacing = spacing or 3.2 * radius
    margin = radius + 6
    points = np.arange(margin, size - margin, spacing)
    centres = np.array([(y, x) for y in points for x in points], np.float64)
    return centres + rng.uniform(-0.5, 0.5, centres.shape)


class _ShiftRefiner:
    """A refiner that moves every disk by a fixed amount, declines the ones
    named, and optionally reports an uncertainty."""

    def __init__(self, shift, declined=(), sigma=None):
        self.shift = np.asarray(shift, np.float64)
        self.declined = set(declined)
        self.sigma = sigma
        self.batches = []

    def crop_half_width(self, spot_radius):
        return int(np.ceil(spot_radius)) + 4

    def refine(self, crops, centres, spot_radius):
        self.batches.append(len(crops))
        out = centres + self.shift
        for index in self.declined:
            if index < len(out):
                out[index] = np.nan
        sigma = None if self.sigma is None else np.full(len(crops), self.sigma, np.float32)
        return out, sigma


class TestRefinerInterface:
    def test_crops_are_native_windows_with_the_detection_in_crop_pixels(self):
        frame = np.arange(40 * 50, dtype=np.float32).reshape(40, 50)
        crops, local = extract_crops(frame, np.array([[10.3, 20.6]]), 3)
        assert crops.shape == (1, 7, 7) and crops.dtype == np.float32
        # nearest pixel (10, 21) sits at the crop centre; the sub-pixel part is kept
        assert crops[0, 3, 3] == frame[10, 21]
        np.testing.assert_allclose(local[0], [3.3, 2.6])

    def test_a_crop_past_the_frame_edge_is_zero_there(self):
        frame = np.ones((20, 20), np.float32)
        crops, _ = extract_crops(frame, np.array([[0.0, 19.0]]), 3)
        assert (crops[0, :3] == 0).all() and (crops[0, :, 4:] == 0).all()
        assert (crops[0, 3:, :4] == 1).all()

    def test_a_refined_centre_replaces_the_detection(self):
        frame = np.zeros((64, 64), np.float32)
        refiner = _ShiftRefiner([0.4, -0.3])
        positions, sigmas = refine_centres([frame], [np.array([[20.0, 30.0]])], 6.0, refiner)
        np.testing.assert_allclose(positions[0], [[20.4, 29.7]], atol=1e-5)
        assert sigmas is None                       # this refiner reports none

    def test_a_declined_or_far_moved_disk_keeps_its_detection(self):
        """NaN declines; a move beyond half a spot radius is not trusted."""
        frame = np.zeros((64, 64), np.float32)
        detections = np.array([[20.0, 20.0], [40.0, 40.0]])
        declined = refine_centres([frame], [detections], 6.0,
                                  _ShiftRefiner([0.5, 0.0], declined={0}))[0][0]
        np.testing.assert_allclose(declined, [[20.0, 20.0], [40.5, 40.0]])
        too_far = refine_centres([frame], [detections], 6.0, _ShiftRefiner([3.5, 0.0]))[0][0]
        np.testing.assert_allclose(too_far, detections)

    def test_the_refiners_uncertainty_is_carried_where_it_moved_a_disk(self):
        frame = np.zeros((64, 64), np.float32)
        _, sigmas = refine_centres([frame], [np.array([[20.0, 20.0], [40.0, 40.0]])], 6.0,
                                   _ShiftRefiner([0.2, 0.0], declined={1}, sigma=0.05))
        np.testing.assert_allclose(sigmas[0][0], 0.05)
        assert np.isnan(sigmas[0][1])

    def test_a_chunk_is_refined_in_batches_and_matches_frame_by_frame(self):
        rng = np.random.default_rng(3)
        frames = [uneven_disks((96, 96), disk_lattice(96, 5, rng), 5, rng) for _ in range(5)]
        detections = [disk_lattice(96, 5, np.random.default_rng(i)) for i in range(5)]
        refiner = MaskCentroidRefiner()
        together, _ = refine_centres(frames, detections, 5.0, refiner, batch_size=7)
        for frame, points, chunk_result in zip(frames, detections, together):
            alone, _ = refine_centres([frame], [points], 5.0, refiner)
            np.testing.assert_array_equal(alone[0], chunk_result)

    def test_the_batch_never_holds_much_more_than_its_size(self):
        frames = [np.zeros((64, 64), np.float32)] * 6
        detections = [np.full((5, 2), 32.0)] * 6
        refiner = _ShiftRefiner([0.0, 0.0])
        refine_centres(frames, detections, 4.0, refiner, batch_size=8)
        assert refiner.batches == [10, 10, 10]      # whole frames, flushed at >= 8

    def test_a_frame_with_no_detections_is_left_alone(self):
        positions, _ = refine_centres([np.zeros((8, 8), np.float32)], [np.zeros((0, 2))], 3.0,
                                      MaskCentroidRefiner())
        assert positions[0].shape == (0, 2)

    def test_the_choices_resolve_to_refiners(self, caplog):
        assert refiner_for(None) is None
        assert refiner_for("") is None
        assert refiner_for(CENTRE_DECODE) is None
        assert isinstance(refiner_for(CENTRE_MASK_CENTROID), MaskCentroidRefiner)
        with caplog.at_level(logging.WARNING):
            assert refiner_for("no-such-refiner") is None
        assert "keeping the detector's centres" in caplog.text


class TestMaskCentroid:
    @pytest.mark.parametrize("radius", [4.0, 6.0, 10.0])
    def test_recovers_sub_pixel_centres_of_unevenly_filled_disks(self, radius):
        """Seeds up to 1 px off, disks tilted 30 % across and carrying a lobe
        60 % above the fill. The mask centroid lands 0.24-0.26 px RMS from the
        outline centre for R = 4-10 (what is left is the uneven fill moving
        the soft edge), against 0.8 px for the seed and 0.47-1.1 px for a
        brightness-weighted centroid, which follows the lobe."""
        rng = np.random.default_rng(int(radius))
        size = 160
        truth = disk_lattice(size, radius, rng)
        frame = uneven_disks((size, size), truth, radius, rng)
        seeds = truth + rng.uniform(-1, 1, truth.shape)
        # the estimator itself, so the small-disk gate is off for R = 4
        refined, sigmas = refine_centres([frame], [seeds], radius,
                                         MaskCentroidRefiner(min_spot_radius=0))
        assert sigmas is None

        def rms(points):
            return float(np.sqrt(np.mean(np.sum((points - truth) ** 2, 1))))

        crops, local = extract_crops(frame, seeds, int(np.ceil(radius)) + 5)
        grid = np.arange(crops.shape[1])
        distance = np.hypot(grid[None, :, None] - local[:, 0, None, None],
                            grid[None, None, :] - local[:, 1, None, None])
        weight = (crops - 10.0) * (distance <= radius + 1)
        total = weight.sum((1, 2))
        brightness_in_crop = np.stack([(weight * grid[None, :, None]).sum((1, 2)) / total,
                                       (weight * grid[None, None, :]).sum((1, 2)) / total], 1)
        brightness_centres = seeds + (brightness_in_crop - local)

        assert rms(refined[0]) < 0.3
        assert rms(refined[0]) < 0.4 * rms(seeds)
        assert rms(refined[0]) < 0.6 * rms(brightness_centres)

    def test_small_disks_keep_the_detector_centres(self, monkeypatch):
        """Below 5 px (the gate every refiner shares by default) the stage is
        skipped: on SPED-Ag's 3 px disks the mask centroid scattered the peaks
        enough to fail the CIF-referenced orientation check."""
        from spyde.models.centre_refine import DEFAULT_MIN_SPOT_RADIUS

        assert MaskCentroidRefiner().min_spot_radius == DEFAULT_MIN_SPOT_RADIUS == 5.0
        rng = np.random.default_rng(9)
        truth = disk_lattice(96, 3.0, rng)
        frame = uneven_disks((96, 96), truth, 3.0, rng)
        seeds = truth + 0.6
        refiner = MaskCentroidRefiner()
        called = []
        monkeypatch.setattr(refiner, "refine", lambda *a: called.append(1))
        kept, sigmas = refine_centres([frame], [seeds], 3.0, refiner)
        assert not called and sigmas is None
        np.testing.assert_allclose(kept[0], seeds, atol=1e-5)

        ungated, _ = refine_centres([frame], [seeds], 3.0, MaskCentroidRefiner(min_spot_radius=0))
        assert np.abs(ungated[0] - seeds).max() > 0.1
        moved, _ = refine_centres([frame], [seeds], DEFAULT_MIN_SPOT_RADIUS, MaskCentroidRefiner())
        assert np.abs(moved[0] - seeds).max() > 0.1          # at the threshold it runs

    def test_the_network_refiners_share_the_gate(self, tmp_path):
        from spyde.models import centre_network
        from spyde.models.centre_refine import DEFAULT_MIN_SPOT_RADIUS

        path = tmp_path / "stub.pt"
        _write_stub_refiner(path)
        assert centre_network.load_refiner(path, "cpu").min_spot_radius == DEFAULT_MIN_SPOT_RADIUS
        assert centre_network.load_refiner(
            path, "cpu", contract={"min_spot_radius": 0}).min_spot_radius == 0

    def test_a_crop_with_no_disk_is_declined(self):
        crops = np.random.default_rng(0).poisson(10.0, (2, 21, 21)).astype(np.float32)
        crops[1] = 10.0
        centres, _ = MaskCentroidRefiner().refine(crops, np.full((2, 2), 10.0), 5.0)
        assert np.isnan(centres).all()

    def test_brightness_does_not_matter(self):
        rng = np.random.default_rng(5)
        truth = np.array([[32.3, 31.6]])
        frame = uneven_disks((64, 64), truth, 6.0, rng)
        seeds = truth + 0.7
        dim = refine_centres([frame], [seeds], 6.0, MaskCentroidRefiner())[0][0]
        bright = refine_centres([frame * 50], [seeds], 6.0, MaskCentroidRefiner())[0][0]
        np.testing.assert_allclose(dim, bright, atol=1e-4)


# ── the registry's detector / refiner split ────────────────────────────────────

def _write_stub_refiner(path, base=4):
    """A randomly initialised refiner network checkpoint, in the training
    code's format (state dict + base + crop contract)."""
    import torch

    from spyde.models.centre_network import CentreNet

    torch.manual_seed(0)
    torch.save({"state_dict": CentreNet(base).state_dict(), "base": base,
                "crop_radius": 10.0, "crop_half": 16}, path)


STUB_ID = "centre-stub-v1"
F3_ID = "centre-fast-f3-v1"
F5_ID = "centre-fast-f5-v1"
FAST_IDS = (F3_ID, F5_ID)
P4_ID = "centre-partner-p4-v1"
BUNDLED_REFINERS = (F3_ID, F5_ID, P4_ID)
#: Per bundled fast refiner: its crop geometry and the checkpoint's sigma_scale.
FAST_REFINERS = {F3_ID: dict(crop=(8.0, 13), sigma_scale=1.1276),
                 F5_ID: dict(crop=(10.0, 16), sigma_scale=0.9776)}
_TESTS = Path(__file__).resolve().parents[1]
#: Parity fixtures: the CPU output of the training code's ``fastref.refine``
#: with the calibrated checkpoint, R = 11. "sparse" is 8 low-count frames (112
#: disks) where F5's head reads high and it declines 67; "gan" is 8 real GaN
#: frames with E7 seeds (92 disks), where neither declines any.
#: (model, fixture file, largest centre difference allowed, px) — the GaN
#: frames carry counts up to ~1e4, where the float32 resampling differs from
#: the training code's gather by up to 8e-5 px.
FAST_FIXTURES = {
    "F3-sparse": (F3_ID, _TESTS / "f3_parity_reference.npz", 5e-5),
    "F5-sparse": (F5_ID, _TESTS / "f5_parity_reference.npz", 5e-5),
    "F5-gan": (F5_ID, _TESTS / "f5_parity_reference_gan.npz", 1e-4),
}


@pytest.fixture
def stub_registry(tmp_path, monkeypatch):
    """A user manifest holding one stub refiner, whose default (wrongly) names
    the refiner — it must still never become the detector default."""
    from spyde.models import registry

    weights = tmp_path / "centre_stub.pt"
    _write_stub_refiner(weights)
    (tmp_path / "registry.json").write_text(json.dumps({
        "default": STUB_ID,
        "models": [{
            "id": STUB_ID, "kind": "refiner", "label": "Centre stub",
            "version": 1, "arch": {"base": 4},
            "input": {"crop_radius": 8.0, "crop_half": 12,
                      "normalisation": "ring-median/disk-p95"},
            "source": {"type": "hf", "file": "centre_stub.pt"},
        }],
    }))
    monkeypatch.setattr(registry, "user_models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(registry, "_resolve_hf", lambda source: str(tmp_path / source["file"]))
    monkeypatch.setattr(registry, "_REFINER_CACHE", {})
    registry._invalidate_manifest()
    yield registry
    registry._invalidate_manifest()


class TestRefinerRegistry:
    def test_the_detector_list_never_offers_a_refiner(self, stub_registry):
        available = stub_registry.available_models()
        assert STUB_ID not in [m["id"] for m in available["models"]]
        refiners = {m["id"]: m for m in available["refiners"]}
        assert set(refiners) == {STUB_ID, *BUNDLED_REFINERS}
        assert refiners[STUB_ID]["label"] == "Centre stub"
        assert available["default"] != STUB_ID
        assert stub_registry.default_model_id() in [m["id"] for m in available["models"]]

    def test_the_bundled_manifest_ships_the_fast_refiners_and_not_as_detectors(self, monkeypatch):
        from spyde.models import registry

        monkeypatch.setattr(registry, "_load_user_manifest", lambda: None)
        registry._invalidate_manifest()
        try:
            available = registry.available_models()
            assert [m["id"] for m in available["refiners"]] == list(BUNDLED_REFINERS)
            assert not set(BUNDLED_REFINERS) & {m["id"] for m in available["models"]}
            assert available["default"] not in BUNDLED_REFINERS
            assert all(registry.is_cached(model_id) for model_id in BUNDLED_REFINERS)
        finally:
            registry._invalidate_manifest()

    def test_asking_for_a_refiner_as_the_detector_gives_the_default_detector(
            self, stub_registry, monkeypatch):
        from spyde.models import infer

        loaded = []
        monkeypatch.setattr(stub_registry, "_MODEL_CACHE", {})
        monkeypatch.setattr(infer, "load_model",
                            lambda path, arch=None, device=None: loaded.append(path) or ("m", "cpu"))
        stub_registry.get_model(STUB_ID)
        assert loaded and not loaded[0].endswith("centre_stub.pt")

    def test_a_refiner_loads_with_its_registry_contract(self, stub_registry):
        refiner = stub_registry.get_refiner(STUB_ID, "cpu")
        assert (refiner.crop_radius, refiner.crop_half) == (8.0, 12)
        assert refiner is stub_registry.get_refiner(STUB_ID, "cpu")     # cached
        crops, local = extract_crops(np.ones((64, 64), np.float32), np.array([[30.0, 30.0]]),
                                     refiner.crop_half_width(6.0))
        centres, sigma = refiner.refine(crops, local, 6.0)
        assert centres.shape == (1, 2) and sigma.shape == (1,)

    @pytest.mark.parametrize("crop_half, annulus", [(16, [0.78125, 1.0]), (20, [0.78, 1.0])])
    def test_each_checkpoint_gets_its_own_crop(self, tmp_path, crop_half, annulus):
        """R1 crops 33 px with the annulus at 12.5-16 px; R3 on crops 41 px with
        the annulus at 15.6-20 px. Both come from the registry's input contract."""
        import torch

        from spyde.models import centre_network

        path = tmp_path / f"half{crop_half}.pt"
        _write_stub_refiner(path)
        refiner = centre_network.load_refiner(
            path, "cpu", contract={"crop_half": crop_half, "annulus": annulus})
        assert refiner.crop_half == crop_half and refiner.annulus == tuple(annulus)
        assert refiner.crop_half_width(6.0) == int(np.ceil(crop_half / 10 * 6.0 + 0.5 * 6.0 + 0.5)) + 1

        shapes = []
        refiner.net.register_forward_pre_hook(lambda module, inputs: shapes.append(inputs[0].shape))
        crops, local = extract_crops(np.ones((96, 96), np.float32), np.array([[48.0, 48.0]]),
                                     refiner.crop_half_width(6.0))
        refiner.refine(crops, local, 6.0)
        assert {tuple(shape[-2:]) for shape in shapes} == {(2 * crop_half + 1,) * 2}

        # the background is the median of the contract's annulus, and only of it
        offsets = np.arange(-crop_half, crop_half + 1)
        distance = np.hypot(offsets[:, None], offsets[None, :])
        ring = (distance > annulus[0] * crop_half) & (distance <= annulus[1] * crop_half)
        sampled = np.where(ring, 7.0, 0.0) + np.where(distance <= 10, 20.0, 0.0)
        normalised = refiner._normalise(torch.as_tensor(sampled[None, None], dtype=torch.float32))
        assert float(normalised[0, 0][torch.as_tensor(ring)].abs().max()) == 0.0

    def test_a_local_checkpoint_can_be_named_by_path(self, tmp_path, monkeypatch):
        """``"source": {"type": "file", "path": ...}``: try a checkpoint that is
        still being trained without publishing it."""
        from spyde.models import registry

        weights = tmp_path / "trial.pt"
        _write_stub_refiner(weights)
        (tmp_path / "registry.json").write_text(json.dumps({"models": [
            {"id": "trial", "kind": "refiner", "arch": {"base": 4},
             "source": {"type": "file", "path": str(weights)}},
            {"id": "missing", "kind": "refiner",
             "source": {"type": "file", "path": str(tmp_path / "nope.pt")}}]}))
        monkeypatch.setattr(registry, "user_models_dir", lambda: str(tmp_path))
        monkeypatch.setattr(registry, "_REFINER_CACHE", {})
        registry._invalidate_manifest()
        try:
            assert registry.is_cached("trial") and not registry.is_cached("missing")
            assert registry.get_refiner("trial", "cpu").crop_half == 16
            assert refiner_for("missing") is None
        finally:
            registry._invalidate_manifest()

    def test_a_detector_is_not_a_refiner(self, stub_registry):
        with pytest.raises(ValueError):
            stub_registry.get_refiner(stub_registry.default_model_id(), "cpu")

    def test_an_unknown_normalisation_is_refused(self, tmp_path):
        from spyde.models import centre_network

        path = tmp_path / "stub.pt"
        _write_stub_refiner(path)
        with pytest.raises(ValueError, match="normalisation"):
            centre_network.load_refiner(path, "cpu", contract={"normalisation": "zscore"})


# ── the network refiner's inference: mirrors, sigma, re-crop ───────────────────

class _ThresholdNet:
    """A stand-in segmenter: covered where the normalised crop is above 0.3.
    Mirror-equivariant, as a well-trained network should be."""

    def __call__(self, x):
        return 40.0 * (x - 0.3)


class _BiasedNet(_ThresholdNet):
    """Adds coverage toward +x in whatever crop it is shown, so its mirrored
    answers disagree by a known amount."""

    def __init__(self, bias):
        self.bias = bias

    def __call__(self, x):
        import torch

        columns = torch.linspace(-1, 1, x.shape[-1])
        return super().__call__(x) + self.bias * columns


class _CentreSeekingNet(_ThresholdNet):
    """Coverage fades away from the crop centre: a network that leans toward the
    middle of its crop, which is what a re-crop pass corrects."""

    def __call__(self, x):
        import torch

        offsets = torch.linspace(-1, 1, x.shape[-1])
        distance = torch.sqrt(offsets[:, None] ** 2 + offsets[None, :] ** 2)
        return super().__call__(x) - 100.0 * distance ** 2


def _one_disk(offset, radius=6.0, seed=0, uneven=True):
    """A disk at a sub-pixel position, unevenly filled unless asked otherwise,
    and a detection ``offset`` px away from it."""
    rng = np.random.default_rng(seed)
    truth = np.array([[48.37, 47.81]])
    frame = uneven_disks((96, 96), truth, radius, rng, gradient=0.4 if uneven else 0.0,
                         lobe=0.8 if uneven else 0.0, amplitude=400.0)
    return frame, truth, truth + np.asarray(offset)


def _network(net, **options):
    from spyde.models.centre_network import NetworkRefiner

    return NetworkRefiner(net, "cpu", crop_half=20, **options)


class TestNetworkRefinerInference:
    def test_mirrored_answers_are_flipped_back_into_the_crops_frame(self):
        """The disk sits 1.2 px down and 0.8 px left of the detection and is
        lit unevenly. A mirror-equivariant network gives the same offset in all
        four views once each is flipped back, so the spread (and sigma) is ~0;
        a sign error in any un-mirroring would put that view ~2 px away."""
        import torch

        frame, truth, seed = _one_disk([-1.2, 0.8])
        refiner = _network(_ThresholdNet(), passes=1, mirror_mean=False)
        crops, local = extract_crops(frame, seed, refiner.crop_half_width(6.0))
        step = 6.0 / refiner.crop_radius
        sampled = refiner._normalise(refiner._resample(
            torch.as_tensor(crops), torch.as_tensor(local, dtype=torch.float32), step))
        views, _, _ = refiner._views(sampled, mirrored=True)
        assert views.shape == (4, 1, 2)
        assert float(views[0].norm()) * step > 1.0             # the disk is well off-centre
        assert float((views - views[0]).abs().max()) * step < 1e-3

        positions, sigmas = refine_centres([frame], [seed], 6.0, refiner)
        assert np.hypot(*(positions[0] - truth).T)[0] < 0.5 * np.hypot(*(seed - truth).T)[0]
        assert sigmas[0][0] < 1e-2

    def test_sigma_is_six_times_the_mirror_spread(self):
        import torch

        frame, _, seed = _one_disk([0.3, -0.2])
        refiner = _network(_BiasedNet(2.0), passes=1, mirror_mean=False)
        crops, local = extract_crops(frame, seed, refiner.crop_half_width(6.0))
        _, sigma = refiner.refine(crops, local, 6.0)

        step = 6.0 / refiner.crop_radius
        sampled = refiner._normalise(refiner._resample(
            torch.as_tensor(crops), torch.as_tensor(local, dtype=torch.float32), step))
        views, _, _ = refiner._views(sampled, mirrored=True)
        spread = float((views - views.mean(0)).norm(dim=2).mean()) * step
        assert spread > 0.01
        assert sigma[0] == pytest.approx(6.0 * spread, rel=1e-4)

    def test_an_unsure_disk_is_declined_and_keeps_its_detection(self):
        """sigma above a quarter of the spot radius: NaN from the refiner, and the
        stage keeps the detected centre and the detector's own sigma."""
        frame, _, seed = _one_disk([0.3, -0.2])
        refiner = _network(_BiasedNet(60.0), passes=1)
        crops, local = extract_crops(frame, seed, refiner.crop_half_width(6.0))
        centres, sigma = refiner.refine(crops, local, 6.0)
        assert np.isnan(centres).all() and np.isnan(sigma).all()

        positions, sigmas = refine_centres([frame], [seed], 6.0, refiner)
        np.testing.assert_allclose(positions[0], seed, atol=1e-5)
        assert np.isnan(sigmas[0]).all()

    def test_the_mirror_mean_is_a_contract_option(self):
        frame, _, seed = _one_disk([0.3, -0.2])
        crops, local = extract_crops(frame, seed, 20)
        unbiased = _network(_ThresholdNet(), passes=1, mirror_mean=False).refine(crops, local, 6.0)
        single = _network(_BiasedNet(2.0), passes=1, mirror_mean=False).refine(crops, local, 6.0)
        mean = _network(_BiasedNet(2.0), passes=1, mirror_mean=True).refine(crops, local, 6.0)
        # the bias pushes the plain view along +x; the x-mirrors cancel it in the mean
        assert single[0][0, 1] - unbiased[0][0, 1] > 0.05
        assert abs(mean[0][0, 1] - unbiased[0][0, 1]) < 0.2 * (single[0][0, 1] - unbiased[0][0, 1])
        assert mean[1][0] == pytest.approx(single[1][0], rel=1e-4)

    def test_sampling_the_crop_equals_sampling_the_whole_frame(self):
        """The refiner resamples each disk's small native crop, never the frame.
        That must give exactly what bilinear sampling of the whole frame gives
        (zeros outside it), which is how the network's training crops are cut —
        including disks whose grid runs off the frame edge."""
        import torch
        import torch.nn.functional as F

        rng = np.random.default_rng(21)
        refiner = _network(_ThresholdNet())
        radius = 6.0
        step = radius / refiner.crop_radius
        frame = rng.normal(50, 20, (90, 70)).astype(np.float32)
        centres = np.column_stack([rng.uniform(-2, 92, 64), rng.uniform(-2, 72, 64)])
        crops, local = extract_crops(frame, centres, refiner.crop_half_width(radius))
        from_crops = refiner._resample(torch.as_tensor(crops),
                                       torch.as_tensor(local, dtype=torch.float32), step)

        offsets = np.arange(-refiner.crop_half, refiner.crop_half + 1) * step
        rows = centres[:, 0, None] + offsets[None]
        columns = centres[:, 1, None] + offsets[None]
        grid = np.stack([np.broadcast_to(columns[:, None, :], (64, len(offsets), len(offsets))) / 69 * 2 - 1,
                         np.broadcast_to(rows[:, :, None], (64, len(offsets), len(offsets))) / 89 * 2 - 1], -1)
        from_frame = F.grid_sample(torch.as_tensor(frame)[None, None].expand(64, 1, -1, -1),
                                   torch.as_tensor(grid, dtype=torch.float32), mode="bilinear",
                                   padding_mode="zeros", align_corners=True)
        # float32 grid coordinates: ~1e-5 of the signal (measured 8e-6)
        assert float((from_crops - from_frame).abs().max()) < 3e-5 * float(from_frame.abs().max())

    def test_no_tensor_is_ever_the_size_of_a_frame_per_disk(self, monkeypatch):
        """Memory guard: with 512 x 512 frames, everything the refiner samples
        or runs through the network is crop-sized."""
        import torch.nn.functional as F

        from spyde.models import centre_network

        seen = []
        original = F.grid_sample

        def spy(source, grid, *args, **kwargs):
            seen.append(tuple(source.shape))
            return original(source, grid, *args, **kwargs)

        monkeypatch.setattr(centre_network.F, "grid_sample", spy)
        rng = np.random.default_rng(22)
        truth = disk_lattice(512, 6.0, rng, spacing=40.0)
        frames = [uneven_disks((512, 512), truth, 6.0, rng)] * 2
        refiner = _network(_ThresholdNet())
        refine_centres(frames, [truth, truth], 6.0, refiner)
        crop = 2 * refiner.crop_half_width(6.0) + 1
        assert seen and all(shape[-2:] == (crop, crop) for shape in seen)
        assert all(shape[0] <= 2 * len(truth) for shape in seen)

    def test_a_re_crop_pass_corrects_a_network_that_leans_to_the_crop_centre(self):
        errors = {}
        for passes in (1, 2, 3):
            frame, truth, seed = _one_disk([1.5, -1.0], uneven=False)
            refiner = _network(_CentreSeekingNet(), passes=passes)
            positions, _ = refine_centres([frame], [seed], 6.0, refiner)
            errors[passes] = float(np.hypot(*(positions[0] - truth).T)[0])
        assert errors[1] > 0.3                     # one pass is pulled toward the seed
        assert errors[2] < 0.5 * errors[1]
        assert errors[3] <= errors[2] + 1e-6


# ── the fast refiners (F3, F5): their own layout, a sigma head, a radius gate ──

def _fast_reference(fixture="F3-sparse"):
    """Frames, per-frame seeds and the centres + sigma the training code's
    ``fastref.refine`` gave for them on the CPU (see ``FAST_FIXTURES``)."""
    reference = np.load(FAST_FIXTURES[fixture][1])
    split = np.cumsum(reference["counts"])[:-1]
    return (list(reference["frames"].astype(np.float32)),
            np.split(reference["seeds"][:, :2], split), float(reference["radius"]),
            reference["centres"], reference["sigma"])


def _load_fast(model_id):
    """A fresh load of a bundled fast refiner from its registry entry (not the
    shared cache, which these tests patch and hook)."""
    from spyde.models import centre_network, registry

    registry._invalidate_manifest()
    entry = registry._entry(model_id)
    refiner = centre_network.load_refiner(registry._resolve_weights(entry), "cpu",
                                          arch=entry.get("arch"), contract=entry.get("input"))
    refiner.model_id = model_id
    refiner.fixture = "F3-sparse" if model_id == F3_ID else "F5-gan"
    return refiner


@pytest.fixture(params=FAST_IDS)
def fast_refiner(request):
    return _load_fast(request.param)


class TestFastRefiner:
    def test_the_bundled_checkpoint_loads_in_its_own_layout(self, fast_refiner):
        from spyde.models.centre_network import FastCentreNet

        assert isinstance(fast_refiner.net, FastCentreNet)
        expected = FAST_REFINERS[fast_refiner.model_id]["crop"]
        assert (fast_refiner.crop_radius, fast_refiner.crop_half) == expected
        assert fast_refiner.normalisation == "mean" and fast_refiner.uncertainty == "head"
        assert (fast_refiner.recrop_over, fast_refiner.min_spot_radius) == (0.15, 5.0)
        # sigma is scaled by R / crop_radius of THIS checkpoint and its own calibration
        assert fast_refiner.sigma_scale == pytest.approx(
            FAST_REFINERS[fast_refiner.model_id]["sigma_scale"], abs=1e-4)

    @pytest.mark.parametrize("fixture", sorted(FAST_FIXTURES))
    def test_matches_the_training_code_on_the_cpu(self, fixture):
        """Measured: centres within 3e-5 px on the sparse frames and 8e-5 px on
        the real GaN frames; sigma within 2e-5 px. On real GaN F5 declines none
        of the 92 disks (median sigma 0.106 R); on the sparse frames its head
        reads high and it declines 67 of 112, as the training code does."""
        model_id, _, tolerance = FAST_FIXTURES[fixture]
        refiner = _load_fast(model_id)
        frames, seeds, radius, centres, sigma = _fast_reference(fixture)
        positions, sigmas = refine_centres(frames, seeds, radius, refiner)
        np.testing.assert_allclose(np.concatenate(positions), centres, rtol=0, atol=tolerance)
        got = np.concatenate(sigmas)
        assert (np.isnan(got) == np.isnan(sigma)).all()
        np.testing.assert_allclose(got[np.isfinite(got)], sigma[np.isfinite(sigma)],
                                   rtol=0, atol=5e-5)

    def test_one_view_and_a_second_pass_only_for_disks_that_moved(self, fast_refiner):
        """The sigma head needs no mirrors; the re-crop pass sees only the disks
        the first pass moved by more than 0.15 R."""
        frames, seeds, radius, _, _ = _fast_reference(fast_refiner.fixture)
        batches = []
        fast_refiner.net.register_forward_pre_hook(lambda module, inputs: batches.append(len(inputs[0])))
        refine_centres(frames, seeds, radius, fast_refiner)
        total = sum(len(s) for s in seeds)
        assert batches[0] == total
        assert len(batches) <= 2 and (len(batches) == 1 or batches[1] < total)

    def test_small_disks_keep_the_detector_centres(self, fast_refiner, monkeypatch):
        """Below the contract's 5 px spot radius the stage is skipped outright."""
        called = []
        monkeypatch.setattr(fast_refiner, "refine", lambda *a: called.append(1))
        frames, seeds, _, _, _ = _fast_reference(fast_refiner.fixture)
        positions, sigmas = refine_centres(frames, seeds, 4.0, fast_refiner)
        assert not called and sigmas is None
        for got, seed in zip(positions, seeds):
            np.testing.assert_array_equal(got, seed.astype(np.float32))

    def test_a_disk_whose_head_sigma_is_too_large_is_declined(self, fast_refiner, monkeypatch):
        frames, seeds, radius, _, _ = _fast_reference(fast_refiner.fixture)
        monkeypatch.setattr(fast_refiner, "max_sigma_fraction", 1e-6)
        positions, sigmas = refine_centres(frames[:1], seeds[:1], radius, fast_refiner)
        np.testing.assert_allclose(positions[0], seeds[0], atol=1e-5)
        assert np.isnan(sigmas[0]).all()


# ── the Friedel-partner refiner (P4): a second channel at the mirror point ─────

P4_REFERENCE = Path(__file__).resolve().parents[1] / "p4_parity_reference.npz"


def _p4_reference():
    """The 8 real GaN frames of F5's fixture, F5's refined centres as the
    incoming positions (the shipped cascade), and the training code's
    ``partner_refine.refine_centres_with_partner`` output on the CPU."""
    frames = np.load(FAST_FIXTURES["F5-gan"][1])["frames"].astype(np.float32)
    reference = np.load(P4_REFERENCE)
    incoming = np.split(reference["incoming"], np.cumsum(reference["counts"])[:-1])
    return (list(frames), incoming, float(reference["radius"]), reference["centres"],
            reference["sigma"])


@pytest.fixture
def p4_refiner():
    return _load_fast(P4_ID)


class TestPartnerRefiner:
    def test_the_input_channels_come_from_the_first_convolution(self, tmp_path):
        import torch

        from spyde.models import centre_network, registry

        for model_id, channels in ((F5_ID, 1), (P4_ID, 2)):
            state = torch.load(registry._resolve_weights(registry._entry(model_id)),
                               weights_only=True)["state_dict"]
            assert centre_network.input_channels(state) == channels
        path = tmp_path / "two_channels.pt"
        torch.save({"state_dict": centre_network.FastCentreNet(4, 2, channels=2).state_dict(),
                    "base": 4, "levels": 2}, path)
        assert centre_network.load_refiner(path, "cpu").needs_partner
        assert not _load_fast(F5_ID).needs_partner

    def test_the_bundled_checkpoint_loads_with_its_radius_range(self, p4_refiner):
        assert p4_refiner.needs_partner and p4_refiner.uncertainty == "head"
        assert (p4_refiner.min_spot_radius, p4_refiner.max_spot_radius) == (6.0, 16.0)
        assert (p4_refiner.crop_radius, p4_refiner.crop_half) == (10.0, 16)

    def test_matches_the_training_code_on_the_cpu(self, p4_refiner):
        """Measured: identical centres, sigma within 4e-7 px (the task's bar
        is 0.01 px)."""
        frames, incoming, radius, centres, sigma = _p4_reference()
        positions, sigmas = refine_centres(frames, incoming, radius, p4_refiner)
        np.testing.assert_allclose(np.concatenate(positions), centres, rtol=0, atol=0.01)
        got = np.concatenate(sigmas)
        assert (np.isnan(got) == np.isnan(sigma)).all()
        np.testing.assert_allclose(got[np.isfinite(got)], sigma[np.isfinite(sigma)], rtol=0, atol=0.01)

    @pytest.mark.parametrize("radius", [5.9, 16.1])
    def test_disks_outside_its_radius_range_keep_their_centres(self, p4_refiner, monkeypatch, radius):
        """Below 6 px a version trained smaller made SPED-Ag's in-grain speckle
        worse; above 16 px it is untrained. Either way the incoming centres
        (F5's, in the cascade) are kept without running the network."""
        called = []
        monkeypatch.setattr(p4_refiner, "refine", lambda *a: called.append(1))
        frames, incoming, _, _, _ = _p4_reference()
        positions, sigmas = refine_centres(frames, incoming, radius, p4_refiner)
        assert not called and sigmas is None
        for got, before in zip(positions, incoming):
            np.testing.assert_array_equal(got, before.astype(np.float32))

    def test_the_second_window_is_cut_at_the_mirror_point(self, p4_refiner, monkeypatch):
        from spyde.models.centre_refine import frame_beams

        frames, incoming, radius, _, _ = _p4_reference()
        seen = {}

        def spy(crops, centres, spot_radius, partner_crops=None, partner_centres=None):
            seen.update(partner_crops=np.asarray(partner_crops), partner_centres=partner_centres)
            return np.full((len(crops), 2), np.nan), None

        monkeypatch.setattr(p4_refiner, "refine", spy)
        refine_centres(frames[:1], incoming[:1], radius, p4_refiner)
        beam = frame_beams(incoming[:1], frames[0].shape, radius)[0]
        expected, expected_local = extract_crops(frames[0], 2 * beam - incoming[0],
                                                 p4_refiner.crop_half_width(radius))
        np.testing.assert_array_equal(seen["partner_crops"], expected)
        np.testing.assert_allclose(seen["partner_centres"], expected_local)

    def test_a_partner_network_refuses_to_run_without_the_mirror_window(self, p4_refiner):
        with pytest.raises(ValueError, match="mirror window"):
            p4_refiner.refine(np.zeros((1, 41, 41), np.float32), np.full((1, 2), 20.0), 11.0)

    def test_the_beam_is_the_frames_centre_of_friedel_symmetry(self):
        from spyde.models.centre_refine import frame_beams

        rng = np.random.default_rng(3)
        centre = np.array([250.3, 247.8])
        g = rng.uniform(-80, 80, (6, 2))
        # the direct beam plus six Friedel pairs about it
        symmetric = np.concatenate([centre[None], centre + g, centre - g]) + rng.normal(0, 0.2, (13, 2))
        sparse = symmetric[1:4]                               # under four disks
        beams = frame_beams([symmetric, sparse, symmetric + 1.0], (507, 502), 11.0)
        np.testing.assert_allclose(beams[0], centre, atol=0.2)
        np.testing.assert_allclose(beams[2], centre + 1.0, atol=0.2)
        np.testing.assert_allclose(beams[1], np.median(beams[[0, 2]], 0))   # median of the rest
        alone = frame_beams([sparse], (507, 502), 11.0)
        np.testing.assert_allclose(alone[0], [253.5, 251.0])                 # the frame centre

    def test_windows_cut_on_a_device_equal_the_numpy_ones(self):
        """The GPU path gathers from the frames moved to the device; on any
        device the windows are exactly :func:`extract_crops`'s."""
        import torch

        from spyde.models.centre_refine import _extract_crops_on_device

        rng = np.random.default_rng(4)
        frames = [rng.normal(50, 20, (90, 70)).astype(np.float32) for _ in range(3)]
        centres = np.column_stack([rng.uniform(-3, 93, 30), rng.uniform(-3, 73, 30)])
        frame_of = rng.integers(0, 3, 30)
        crops, local = _extract_crops_on_device(torch.as_tensor(np.stack(frames)), frame_of,
                                                centres, 9)
        for i in range(30):
            expected, expected_local = extract_crops(frames[frame_of[i]], centres[i:i + 1], 9)
            np.testing.assert_array_equal(crops[i].numpy(), expected[0])
            np.testing.assert_allclose(local[i], expected_local[0])


# ── the Find Vectors wiring, with a stand-in detector ──────────────────────────

RADIUS = 5.0


def _scan(rng, frames=4, size=80):
    truth = [disk_lattice(size, RADIUS, rng) for _ in range(frames)]
    stack = np.stack([uneven_disks((size, size), t, RADIUS, rng) for t in truth])
    return stack, truth


@pytest.fixture
def standin_detector(monkeypatch):
    """``models.detect`` / ``detect_batch`` stand-ins returning each frame's true
    disks shifted by a fixed 0.8 px (columns ``[y, x, score, width]``), so the
    centre stage has a known error to remove."""
    from spyde import models

    truths = {}

    def _rows(frame):
        truth = truths[frame.tobytes()]
        rows = np.zeros((len(truth), 4), np.float32)
        rows[:, :2] = truth + 0.8
        rows[:, 2], rows[:, 3] = 0.9, 1.5
        return rows

    def fake_detect(model, frame, device, **_):
        return _rows(np.asarray(frame, np.float32))

    def fake_detect_batch(model, frames, device, **_):
        return [_rows(np.asarray(f, np.float32)) for f in frames]

    monkeypatch.setattr(models, "get_model", lambda mid=None: (None, types.SimpleNamespace(type="cpu")))
    monkeypatch.setattr(models, "detect", fake_detect)
    monkeypatch.setattr(models, "detect_batch", fake_detect_batch)
    return truths


def _register(truths, stack, truth):
    for frame, t in zip(stack, truth):
        truths[np.asarray(frame, np.float32).tobytes()] = t


class TestFindVectorsWiring:
    @pytest.mark.parametrize("batched", [True, False])
    def test_batch_and_preview_place_every_disk_identically(self, standin_detector, monkeypatch,
                                                            batched):
        """The chunk path (one refiner batch for the block — or, off the GPU
        lane, frame by frame) and the single-frame preview give the same
        record, and the mask centroid really moved the disks back."""
        import spyde.actions.find_vectors_neural as neural
        import spyde.actions.find_vectors_torch as find_vectors_torch

        stack, truth = _scan(np.random.default_rng(11))
        _register(standin_detector, stack, truth)
        monkeypatch.setattr(find_vectors_torch, "torch_gpu_device",
                            (lambda: "fake-gpu") if batched else (lambda: None))
        block = neural._neural_block(stack.reshape(2, 2, *stack.shape[1:]), 0.3, 3, True, None,
                                     None, spot_radius=RADIUS, centre_refiner=CENTRE_MASK_CENTROID)
        for index, (frame, t) in enumerate(zip(stack, truth)):
            preview = neural._find_vectors_single_frame_neural(
                frame, 0.3, 3, spot_radius=RADIUS, centre_refiner=CENTRE_MASK_CENTROID)[2]
            row = block[index // 2, index % 2]
            row = row[np.isfinite(row[:, 0])]
            np.testing.assert_array_equal(row, preview)
            error = np.hypot(*(preview[:, :2] - t).T)
            assert np.sqrt(np.mean(error ** 2)) < 0.3       # the stand-in was 1.13 px off

    def test_decode_leaves_the_detections_where_they_were(self, standin_detector):
        import spyde.actions.find_vectors_neural as neural

        stack, truth = _scan(np.random.default_rng(12), frames=1)
        _register(standin_detector, stack, truth)
        peaks = neural._find_vectors_single_frame_neural(
            stack[0], 0.3, 3, spot_radius=RADIUS, centre_refiner=CENTRE_DECODE)[2]
        np.testing.assert_allclose(peaks[:, :2], truth[0] + 0.8, atol=1e-5)

    def test_the_preview_dispatch_passes_the_choice(self, standin_detector, monkeypatch):
        from spyde.actions.find_vectors import _find_peaks_single_frame
        from spyde.actions.vector_overlay import _detector_params

        seen = []
        monkeypatch.setattr(centre_refine, "refiner_for",
                            lambda choice, device=None: seen.append(choice) or None)
        stack, truth = _scan(np.random.default_rng(13), frames=1)
        _register(standin_detector, stack, truth)
        params = _detector_params({"method": "neural", "spot_radius": RADIUS,
                                   "centre_refiner": CENTRE_MASK_CENTROID, "threshold": 0.3})
        _find_peaks_single_frame(stack[0], params)
        assert seen == [CENTRE_MASK_CENTROID]

    def test_the_wizard_defaults_to_the_decode(self):
        from spyde.actions.find_vectors_action import DEFAULTS, _coerce

        assert DEFAULTS["centre_refiner"] == CENTRE_DECODE
        assert _coerce({"centre_refiner": CENTRE_MASK_CENTROID})["centre_refiner"] == \
            CENTRE_MASK_CENTROID

    def test_the_wizard_dropdown_offers_the_built_in_choices(self):
        """The TSX dropdown's built-in values are the backend's choice names."""
        from pathlib import Path

        source = (Path(__file__).resolve().parents[3] / "electron" / "src" / "renderer" / "src"
                  / "components" / "FindVectorsWizard.tsx").read_text(encoding="utf-8")
        assert f"value: '{CENTRE_DECODE}'" in source
        assert f"value: '{CENTRE_MASK_CENTROID}'" in source
        assert "useState(saved?.centreRefiner ?? 'decode')" in source

    def test_the_full_dataset_is_never_computed(self, standin_detector, monkeypatch):
        """The centre stage reads crops from the chunk it is handed, never the
        whole lazy dataset: a guard on ``dask.array.Array.compute`` raises for
        the full shape, and every refiner call sees one chunk's frames."""
        from unittest.mock import patch

        import dask.array as da
        import hyperspy.api as hs

        from spyde.actions.find_vectors import _do_compute_vectors

        rng = np.random.default_rng(14)
        stack, truth = _scan(rng, frames=36, size=64)
        _register(standin_detector, stack, truth)
        lazy = hs.signals.Signal2D(da.from_array(stack.reshape(6, 6, 64, 64),
                                                 chunks=(3, 3, 64, 64)))
        full_shape = lazy.data.shape
        frames_per_call = []
        original_refine = centre_refine.refine_centres

        def spy_refine(frames, *args, **kwargs):
            frames_per_call.append(len(frames))
            return original_refine(frames, *args, **kwargs)

        original_compute = da.Array.compute

        def guarded_compute(self, *args, **kwargs):
            if self.shape == full_shape:
                raise AssertionError(f"full-dataset compute {self.shape}")
            return original_compute(self, *args, **kwargs)

        monkeypatch.setattr(centre_refine, "refine_centres", spy_refine)
        with patch.object(da.Array, "compute", guarded_compute):
            vectors = _do_compute_vectors(
                lazy, dict(method="neural", sigma=0.0, kernel_radius=5, threshold=0.3,
                           min_distance=3, subpixel=True, spot_radius=RADIUS,
                           centre_refiner=CENTRE_MASK_CENTROID), None, None)
        assert vectors.nav_shape == (6, 6)
        assert frames_per_call and max(frames_per_call) < 36
        assert sum(frames_per_call) == 36


# ── the real network, in one subprocess ────────────────────────────────────────

_DRIVER = textwrap.dedent(r"""
    import json, os, sys, tempfile
    import numpy as np
    sys.path.insert(0, os.environ["SPYDE_TEST_DIR"])
    from test_centre_refiner import uneven_disks, disk_lattice, _write_stub_refiner, STUB_ID

    def frames_and_truth(n=4, size=128, radius=6.0, seed=0):
        rng = np.random.default_rng(seed)
        truth = [disk_lattice(size, radius, rng, spacing=24.0) for _ in range(n)]
        return np.stack([uneven_disks((size, size), t, radius, rng) for t in truth]), truth

    def install_stub():
        from spyde.models import registry
        folder = tempfile.mkdtemp()
        _write_stub_refiner(os.path.join(folder, "centre_stub.pt"))
        with open(os.path.join(folder, "registry.json"), "w") as handle:
            json.dump({"models": [{"id": STUB_ID, "kind": "refiner", "arch": {"base": 4},
                                   "source": {"type": "hf", "file": "centre_stub.pt"}}]}, handle)
        registry.user_models_dir = lambda: folder
        registry._resolve_hf = lambda source: os.path.join(folder, source["file"])
        registry._invalidate_manifest()

    def run_mode(mode):
        from spyde import models
        import spyde.actions.find_vectors_neural as neural
        import spyde.actions.find_vectors_torch as find_vectors_torch
        import torch
        # Everything on the CPU: the batch path and the preview then share one
        # model and one device, so any difference is the code's, not the device's.
        models.get_model = models.get_cpu_model
        find_vectors_torch.torch_gpu_device = lambda: torch.device("cpu")
        install_stub()
        out = {}
        frames, truth = frames_and_truth()
        if mode == "parity":
            for choice in ("mask-centroid", STUB_ID, "centre-fast-f3-v1", "centre-fast-f5-v1"):
                block = neural._neural_block(frames.reshape(2, 2, *frames.shape[1:]), 0.3, 3,
                                             True, None, None, spot_radius=6.0,
                                             centre_refiner=choice)
                worst, moved = 0.0, []
                for index, frame in enumerate(frames):
                    preview = neural._find_vectors_single_frame_neural(
                        frame, 0.3, 3, spot_radius=6.0, centre_refiner=choice)[2]
                    decode = neural._find_vectors_single_frame_neural(
                        frame, 0.3, 3, spot_radius=6.0)[2]
                    row = block[index // 2, index % 2]
                    row = row[np.isfinite(row[:, 0])]
                    assert row.shape == preview.shape, (row.shape, preview.shape)
                    worst = max(worst, float(np.nanmax(np.abs(row - preview))))
                    moved.append(np.hypot(*(preview[:, :2] - decode[:, :2]).T))
                out[choice] = dict(max_abs_difference=worst,
                                   mean_move=float(np.mean(np.concatenate(moved))))
        elif mode == "f3_cuda":
            if not torch.cuda.is_available():
                return {"skipped": True}
            from test_centre_refiner import _fast_reference, FAST_FIXTURES
            from spyde.models import registry
            from spyde.models.centre_refine import refine_centres
            torch.nn.functional.linear(torch.zeros(1, 1, device="cuda"),
                                       torch.zeros(1, 1, device="cuda"))
            for fixture, (model_id, _, _) in FAST_FIXTURES.items():
                frames, seeds, radius, centres, sigma = _fast_reference(fixture)
                positions, sigmas = refine_centres(frames, seeds, radius,
                                                   registry.get_refiner(model_id, "cuda"))
                got = np.concatenate(sigmas)
                out[fixture] = dict(
                    centre_difference=float(np.abs(np.concatenate(positions) - centres).max()),
                    sigma_difference=float(np.nanmax(np.abs(got - sigma))),
                    declined_match=bool((np.isnan(got) == np.isnan(sigma)).all()))
            from test_centre_refiner import _p4_reference, P4_ID
            frames, incoming, radius, centres, sigma = _p4_reference()
            positions, sigmas = refine_centres(frames, incoming, radius,
                                               registry.get_refiner(P4_ID, "cuda"))
            got = np.concatenate(sigmas)
            out["P4-gan"] = dict(
                centre_difference=float(np.abs(np.concatenate(positions) - centres).max()),
                sigma_difference=float(np.nanmax(np.abs(got - sigma))),
                declined_match=bool((np.isnan(got) == np.isnan(sigma)).all()))
        elif mode == "accuracy":
            for choice in ("decode", "mask-centroid"):
                errors = []
                for frame, t in zip(frames, truth):
                    peaks = neural._find_vectors_single_frame_neural(
                        frame, 0.3, 3, spot_radius=6.0, centre_refiner=choice)[2]
                    d = np.hypot(peaks[:, None, 0] - t[None, :, 0], peaks[:, None, 1] - t[None, :, 1])
                    nearest = d.min(0)
                    errors.append(nearest[nearest < 3])
                e = np.concatenate(errors)
                out[choice] = dict(n=int(len(e)), rms=float(np.sqrt(np.mean(e ** 2))))
        return out

    for mode in sys.argv[1:]:
        print("RESULT_JSON", mode, json.dumps(run_mode(mode)), flush=True)
    sys.stdout.flush()
    os._exit(0)
""")


@pytest.fixture(scope="module")
def network_results():
    import os
    from pathlib import Path

    environment = dict(os.environ, SPYDE_TEST_DIR=str(Path(__file__).parent))
    proc = subprocess.run([sys.executable, "-c", _DRIVER, "parity", "accuracy", "f3_cuda"],
                          capture_output=True, text=True, timeout=900, env=environment)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            _, mode, payload = line.split(" ", 2)
            out[mode] = json.loads(payload)
    if len(out) < 3:
        pytest.fail(f"refiner driver failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return out


class TestWithTheDetectorNetwork:
    def test_batch_and_preview_agree_for_every_refiner(self, network_results):
        for choice, result in network_results["parity"].items():
            assert result["max_abs_difference"] < 1e-4, choice

    def test_the_refiners_actually_move_the_disks(self, network_results):
        """The mask centroid moves the detector's disks by a fraction of a pixel
        on average; the random stub network moves some (or declines them)."""
        result = network_results["parity"]
        assert 0.02 < result[CENTRE_MASK_CENTROID]["mean_move"] < 1.0

    def test_the_fast_refiners_match_the_training_code_on_cuda(self, network_results):
        """Against the CPU reference, so the device's own float32 rounding adds
        to the comparison: measured up to 8e-5 px."""
        results = network_results["f3_cuda"]
        if results.get("skipped"):
            pytest.skip("no CUDA device")
        for fixture in FAST_FIXTURES:
            result = results[fixture]
            assert result["declined_match"], fixture
            assert result["centre_difference"] < 1e-4, fixture
            assert result["sigma_difference"] < 5e-5, fixture

    def test_the_partner_refiner_matches_the_training_code_on_cuda(self, network_results):
        """Windows cut on the GPU; measured 3e-5 px (centres) against the CPU
        reference."""
        result = network_results["f3_cuda"]
        if result.get("skipped"):
            pytest.skip("no CUDA device")
        assert result["P4-gan"]["declined_match"]
        assert result["P4-gan"]["centre_difference"] < 0.01
        assert result["P4-gan"]["sigma_difference"] < 0.01

    def test_the_mask_centroid_beats_the_decode_on_uneven_disks(self, network_results):
        result = network_results["accuracy"]
        assert result[CENTRE_MASK_CENTROID]["n"] == result[CENTRE_DECODE]["n"] > 50
        assert result[CENTRE_MASK_CENTROID]["rms"] < result[CENTRE_DECODE]["rms"]

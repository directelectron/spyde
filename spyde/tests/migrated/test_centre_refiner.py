"""The neural method's centre steps (``spyde.models.centre_refine``).

After detection every disk is re-placed on the raw frame at native resolution:
the refine step (the bundled centre network F5), then, when the Friedel-partner
checkbox is on, the step that also reads each disk's Friedel-mirror window (P4)
on top of F5's centres. These tests pin the refiner interface, both networks
against the training code, P4's radius range, the registry's detector/refiner
split, and the Find Vectors wiring: the order of the steps, the two switches,
and that the batch and the single-frame preview place every disk identically.

The wiring tests use a stand-in detector (no detector forward). The real
detector runs in ONE subprocess (``RESULT_JSON <mode> {...}``, then
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
    CENTRE_MODEL, FRIEDEL_MODEL, extract_crops, frame_beams, load_step, refine_centres,
)

_TESTS = Path(__file__).resolve().parents[1]
#: Parity fixtures: the CPU output of the training code with the bundled
#: checkpoints, R = 11. F5: "sparse" is 8 low-count frames (112 disks), where
#: F5's sigma head reads high and it declines 67; "gan" is 8 real GaN frames
#: with detector seeds (92 disks), where it declines none. P4: the same GaN
#: frames with F5's refined centres as the incoming positions.
F5_FIXTURES = {
    # (fixture file, largest centre difference allowed, px) — the GaN frames
    # carry counts up to ~1e4, where float32 resampling differs from the
    # training code's gather by up to 8e-5 px.
    "sparse": (_TESTS / "f5_parity_reference.npz", 5e-5),
    "gan": (_TESTS / "f5_parity_reference_gan.npz", 1e-4),
}
P4_REFERENCE = _TESTS / "p4_parity_reference.npz"


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
    named, and optionally reports an uncertainty and a radius range."""

    def __init__(self, shift, declined=(), sigma=None, min_spot_radius=0.0,
                 max_spot_radius=np.inf):
        self.shift = np.asarray(shift, np.float64)
        self.declined = set(declined)
        self.sigma = sigma
        self.min_spot_radius = min_spot_radius
        self.max_spot_radius = max_spot_radius
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
        positions, sigmas = refine_centres([frame], [np.array([[20.0, 30.0]])], 6.0,
                                           _ShiftRefiner([0.4, -0.3]))
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

    def test_outside_the_refiners_radius_range_nothing_runs(self):
        frame = np.zeros((64, 64), np.float32)
        detections = [np.array([[20.0, 20.0]])]
        for radius in (5.9, 16.1):
            refiner = _ShiftRefiner([0.4, 0.0], min_spot_radius=6.0, max_spot_radius=16.0)
            positions, sigmas = refine_centres([frame], detections, radius, refiner)
            assert refiner.batches == [] and sigmas is None
            np.testing.assert_array_equal(positions[0], detections[0])

    def test_a_chunk_is_refined_in_batches_and_matches_frame_by_frame(self):
        rng = np.random.default_rng(3)
        frames = [uneven_disks((96, 96), disk_lattice(96, 6, rng), 6, rng) for _ in range(5)]
        detections = [disk_lattice(96, 6, np.random.default_rng(i)) for i in range(5)]
        refiner = _load(CENTRE_MODEL)
        together, _ = refine_centres(frames, detections, 6.0, refiner, batch_size=7)
        for frame, points, chunk_result in zip(frames, detections, together):
            alone, _ = refine_centres([frame], [points], 6.0, refiner)
            np.testing.assert_allclose(alone[0], chunk_result, rtol=0, atol=1e-4)

    def test_the_batch_never_holds_much_more_than_its_size(self):
        frames = [np.zeros((64, 64), np.float32)] * 6
        detections = [np.full((5, 2), 32.0)] * 6
        refiner = _ShiftRefiner([0.0, 0.0])
        refine_centres(frames, detections, 4.0, refiner, batch_size=8)
        assert refiner.batches == [10, 10, 10]      # whole frames, flushed at >= 8

    def test_a_frame_with_no_detections_is_left_alone(self):
        positions, _ = refine_centres([np.zeros((8, 8), np.float32)], [np.zeros((0, 2))], 6.0,
                                      _ShiftRefiner([0.3, 0.0]))
        assert positions[0].shape == (0, 2)

    def test_a_step_that_cannot_load_is_skipped_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert load_step("no-such-network") is None
        assert "keeping the centres" in caplog.text

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

    def test_the_beam_is_the_frames_centre_of_friedel_symmetry(self):
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


# ── the two bundled networks ───────────────────────────────────────────────────

def _load(model_id):
    """A fresh CPU load of a bundled network from its registry entry (not the
    shared cache, which these tests patch and hook)."""
    from spyde.models import centre_network, registry

    registry._invalidate_manifest()
    entry = registry._entry(model_id)
    return centre_network.load_refiner(registry._resolve_weights(entry), "cpu",
                                       arch=entry.get("arch"), contract=entry.get("input"))


def _f5_reference(fixture):
    """Frames, per-frame seeds and the centres + sigma the training code gave
    for them on the CPU (see ``F5_FIXTURES``)."""
    reference = np.load(F5_FIXTURES[fixture][0])
    split = np.cumsum(reference["counts"])[:-1]
    return (list(reference["frames"].astype(np.float32)),
            np.split(reference["seeds"][:, :2], split), float(reference["radius"]),
            reference["centres"], reference["sigma"])


def _p4_reference():
    """The 8 real GaN frames of F5's fixture, F5's refined centres as the
    incoming positions, and the training code's output on the CPU."""
    frames = np.load(F5_FIXTURES["gan"][0])["frames"].astype(np.float32)
    reference = np.load(P4_REFERENCE)
    incoming = np.split(reference["incoming"], np.cumsum(reference["counts"])[:-1])
    return (list(frames), incoming, float(reference["radius"]), reference["centres"],
            reference["sigma"])


@pytest.fixture
def f5():
    return _load(CENTRE_MODEL)


@pytest.fixture
def p4():
    return _load(FRIEDEL_MODEL)


class TestRefineStep:
    def test_the_bundled_checkpoint_loads_with_its_contract(self, f5):
        from spyde.models.centre_network import FastCentreNet

        assert isinstance(f5.net, FastCentreNet) and not f5.needs_partner
        assert (f5.crop_radius, f5.crop_half) == (10.0, 16)
        assert (f5.recrop_over, f5.min_spot_radius, f5.max_spot_radius) == (0.15, 5.0, np.inf)
        assert f5.sigma_scale == pytest.approx(0.9776, abs=1e-4)

    @pytest.mark.parametrize("fixture", sorted(F5_FIXTURES))
    def test_matches_the_training_code_on_the_cpu(self, f5, fixture):
        """Measured: centres within 3e-5 px on the sparse frames and 8e-5 px on
        the GaN frames; sigma within 2e-5 px."""
        _, tolerance = F5_FIXTURES[fixture]
        frames, seeds, radius, centres, sigma = _f5_reference(fixture)
        positions, sigmas = refine_centres(frames, seeds, radius, f5)
        np.testing.assert_allclose(np.concatenate(positions), centres, rtol=0, atol=tolerance)
        got = np.concatenate(sigmas)
        assert (np.isnan(got) == np.isnan(sigma)).all()
        np.testing.assert_allclose(got[np.isfinite(got)], sigma[np.isfinite(sigma)],
                                   rtol=0, atol=5e-5)

    def test_a_second_pass_only_for_disks_that_moved(self, f5):
        """The re-crop pass sees only the disks the first moved by over 0.15 R."""
        frames, seeds, radius, _, _ = _f5_reference("gan")
        batches = []
        f5.net.register_forward_pre_hook(lambda module, inputs: batches.append(len(inputs[0])))
        refine_centres(frames, seeds, radius, f5)
        total = sum(len(s) for s in seeds)
        assert batches[0] == total
        assert len(batches) <= 2 and (len(batches) == 1 or batches[1] < total)

    def test_small_disks_keep_the_detector_centres(self, f5, monkeypatch):
        """Below 5 px the step is skipped: on SPED-Ag's ~3 px disks refining
        made the in-grain speckle worse."""
        called = []
        monkeypatch.setattr(f5, "refine", lambda *a: called.append(1))
        frames, seeds, _, _, _ = _f5_reference("gan")
        positions, sigmas = refine_centres(frames, seeds, 4.9, f5)
        assert not called and sigmas is None
        for got, seed in zip(positions, seeds):
            np.testing.assert_array_equal(got, seed.astype(np.float32))

    def test_a_disk_whose_sigma_is_too_large_is_declined(self, f5, monkeypatch):
        frames, seeds, radius, _, _ = _f5_reference("gan")
        monkeypatch.setattr(f5, "max_sigma_fraction", 1e-6)
        positions, sigmas = refine_centres(frames[:1], seeds[:1], radius, f5)
        np.testing.assert_allclose(positions[0], seeds[0], atol=1e-5)
        assert np.isnan(sigmas[0]).all()

    def test_no_tensor_is_ever_the_size_of_a_frame_per_disk(self, f5, monkeypatch):
        """Memory guard: with 512 x 512 frames, everything the network samples
        is window-sized."""
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
        refine_centres(frames, [truth, truth], 6.0, f5)
        window = 2 * f5.crop_half_width(6.0) + 1
        assert seen and all(shape[-2:] == (window, window) for shape in seen)
        assert all(shape[0] <= 2 * len(truth) for shape in seen)


class TestFriedelStep:
    def test_the_input_channels_come_from_the_first_convolution(self, tmp_path):
        import torch

        from spyde.models import centre_network, registry

        for model_id, channels in ((CENTRE_MODEL, 1), (FRIEDEL_MODEL, 2)):
            state = torch.load(registry._resolve_weights(registry._entry(model_id)),
                               weights_only=True)["state_dict"]
            assert centre_network.input_channels(state) == channels
        path = tmp_path / "two_channels.pt"
        torch.save({"state_dict": centre_network.FastCentreNet(4, 2, channels=2).state_dict(),
                    "base": 4, "levels": 2}, path)
        assert centre_network.load_refiner(path, "cpu").needs_partner

    def test_the_bundled_checkpoint_loads_with_its_radius_range(self, p4):
        assert p4.needs_partner
        assert (p4.min_spot_radius, p4.max_spot_radius) == (6.0, 16.0)
        assert (p4.crop_radius, p4.crop_half) == (10.0, 16)

    def test_matches_the_training_code_on_the_cpu(self, p4):
        """Measured: identical centres, sigma within 4e-7 px."""
        frames, incoming, radius, centres, sigma = _p4_reference()
        positions, sigmas = refine_centres(frames, incoming, radius, p4)
        np.testing.assert_allclose(np.concatenate(positions), centres, rtol=0, atol=0.01)
        got = np.concatenate(sigmas)
        assert (np.isnan(got) == np.isnan(sigma)).all()
        np.testing.assert_allclose(got[np.isfinite(got)], sigma[np.isfinite(sigma)], rtol=0, atol=0.01)

    @pytest.mark.parametrize("radius", [5.9, 16.1])
    def test_disks_outside_6_to_16_px_keep_the_refine_steps_centres(self, p4, monkeypatch, radius):
        """Below 6 px a version trained smaller made SPED-Ag's in-grain speckle
        worse; above 16 px it is untrained. Either way F5's centres stay."""
        called = []
        monkeypatch.setattr(p4, "refine", lambda *a: called.append(1))
        frames, incoming, _, _, _ = _p4_reference()
        positions, sigmas = refine_centres(frames, incoming, radius, p4)
        assert not called and sigmas is None
        for got, before in zip(positions, incoming):
            np.testing.assert_array_equal(got, before.astype(np.float32))

    def test_the_second_window_is_cut_at_the_mirror_point(self, p4, monkeypatch):
        frames, incoming, radius, _, _ = _p4_reference()
        seen = {}

        def spy(crops, centres, spot_radius, partner_crops=None, partner_centres=None):
            seen.update(partner_crops=np.asarray(partner_crops), partner_centres=partner_centres)
            return np.full((len(crops), 2), np.nan), None

        monkeypatch.setattr(p4, "refine", spy)
        refine_centres(frames[:1], incoming[:1], radius, p4)
        beam = frame_beams(incoming[:1], frames[0].shape, radius)[0]
        expected, expected_local = extract_crops(frames[0], 2 * beam - incoming[0],
                                                 p4.crop_half_width(radius))
        np.testing.assert_array_equal(seen["partner_crops"], expected)
        np.testing.assert_allclose(seen["partner_centres"], expected_local)

    def test_it_refuses_to_run_without_the_mirror_window(self, p4):
        with pytest.raises(ValueError, match="mirror window"):
            p4.refine(np.zeros((1, 41, 41), np.float32), np.full((1, 2), 20.0), 11.0)


# ── the registry's detector / refiner split ────────────────────────────────────

STUB_ID = "centre-stub-v1"


def _write_stub_network(path, base=4):
    """A randomly initialised centre network checkpoint in the training code's
    format (state dict + layout + window contract)."""
    import torch

    from spyde.models.centre_network import FastCentreNet

    torch.manual_seed(0)
    torch.save({"state_dict": FastCentreNet(base, 2).state_dict(), "base": base, "levels": 2,
                "crop_radius": 10.0, "crop_half": 16}, path)


@pytest.fixture
def stub_registry(tmp_path, monkeypatch):
    """A user manifest holding one stub refiner, whose default (wrongly) names
    the refiner — it must still never become the detector default."""
    from spyde.models import registry

    _write_stub_network(tmp_path / "centre_stub.pt")
    (tmp_path / "registry.json").write_text(json.dumps({
        "default": STUB_ID,
        "models": [{
            "id": STUB_ID, "kind": "refiner", "label": "Centre stub",
            "version": 1, "arch": {"base": 4},
            "input": {"crop_radius": 8.0, "crop_half": 12},
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
        detectors = [m["id"] for m in available["models"]]
        assert not {STUB_ID, CENTRE_MODEL, FRIEDEL_MODEL} & set(detectors)
        assert available["default"] != STUB_ID
        assert stub_registry.default_model_id() in detectors

    def test_the_bundled_manifest_ships_both_networks(self, monkeypatch):
        from spyde.models import registry

        monkeypatch.setattr(registry, "_load_user_manifest", lambda: None)
        registry._invalidate_manifest()
        try:
            refiners = [m["id"] for m in registry.list_models(registry.KIND_REFINER)]
            assert refiners == [CENTRE_MODEL, FRIEDEL_MODEL]
            assert all(registry.is_cached(model_id) for model_id in refiners)
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

    def test_a_detector_is_not_a_refiner(self, stub_registry):
        with pytest.raises(ValueError):
            stub_registry.get_refiner(stub_registry.default_model_id(), "cpu")


# ── the Find Vectors wiring, with a stand-in detector ──────────────────────────

RADIUS = 7.0
F5_SHIFT = np.array([-0.5, -0.3])
P4_SHIFT = np.array([-0.2, -0.4])


def _scan(rng, frames=4, size=96):
    truth = [disk_lattice(size, RADIUS, rng) for _ in range(frames)]
    stack = np.stack([uneven_disks((size, size), t, RADIUS, rng) for t in truth])
    return stack, truth


@pytest.fixture
def standin_detector(monkeypatch):
    """``models.detect`` / ``detect_batch`` stand-ins returning each frame's true
    disks shifted by a fixed 0.8 px (columns ``[y, x, score, width]``)."""
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


@pytest.fixture
def standin_steps(monkeypatch):
    """Both steps as known shifts with their real radius ranges, recording the
    order they ran in."""
    loaded = []
    steps = {CENTRE_MODEL: _ShiftRefiner(F5_SHIFT, sigma=0.05, min_spot_radius=5.0),
             FRIEDEL_MODEL: _ShiftRefiner(P4_SHIFT, sigma=0.03, min_spot_radius=6.0,
                                          max_spot_radius=16.0)}

    def fake_load(model_id, device=None):
        loaded.append(model_id)
        return steps[model_id]

    monkeypatch.setattr(centre_refine, "load_step", fake_load)
    return loaded


def _register(truths, stack, truth):
    for frame, t in zip(stack, truth):
        truths[np.asarray(frame, np.float32).tobytes()] = t


def _preview(frame, radius=RADIUS, **steps):
    import spyde.actions.find_vectors_neural as neural

    return neural._find_vectors_single_frame_neural(frame, 0.3, 3, spot_radius=radius, **steps)[2]


class TestFindVectorsWiring:
    def test_the_friedel_step_runs_on_the_refine_steps_centres(self, standin_detector,
                                                                standin_steps):
        stack, truth = _scan(np.random.default_rng(12), frames=1)
        _register(standin_detector, stack, truth)
        peaks = _preview(stack[0], refine_centres=True, friedel_partner=True)
        assert standin_steps == [CENTRE_MODEL, FRIEDEL_MODEL]
        np.testing.assert_allclose(peaks[:, :2], truth[0] + 0.8 + F5_SHIFT + P4_SHIFT, atol=1e-4)
        np.testing.assert_allclose(peaks[:, 4], 0.03)          # the last step's sigma

    def test_with_the_friedel_step_off_only_the_refine_step_runs(self, standin_detector,
                                                                 standin_steps):
        stack, truth = _scan(np.random.default_rng(12), frames=1)
        _register(standin_detector, stack, truth)
        peaks = _preview(stack[0], refine_centres=True, friedel_partner=False)
        assert standin_steps == [CENTRE_MODEL]
        np.testing.assert_allclose(peaks[:, :2], truth[0] + 0.8 + F5_SHIFT, atol=1e-4)
        np.testing.assert_allclose(peaks[:, 4], 0.05)

    def test_small_disks_keep_the_refine_steps_centres_and_sigma(self, standin_detector,
                                                                 standin_steps):
        """At 5.5 px F5 runs and P4 (6-16 px) declines the whole frame."""
        stack, truth = _scan(np.random.default_rng(12), frames=1)
        _register(standin_detector, stack, truth)
        peaks = _preview(stack[0], radius=5.5, refine_centres=True, friedel_partner=True)
        np.testing.assert_allclose(peaks[:, :2], truth[0] + 0.8 + F5_SHIFT, atol=1e-4)
        np.testing.assert_allclose(peaks[:, 4], 0.05)

    def test_with_refinement_off_the_detections_stay(self, standin_detector, standin_steps):
        """The Friedel step is on top of the refine step, never on its own."""
        stack, truth = _scan(np.random.default_rng(12), frames=1)
        _register(standin_detector, stack, truth)
        peaks = _preview(stack[0], refine_centres=False, friedel_partner=True)
        assert standin_steps == []
        np.testing.assert_allclose(peaks[:, :2], truth[0] + 0.8, atol=1e-5)

    @pytest.mark.parametrize("batched", [True, False])
    def test_batch_and_preview_place_every_disk_identically(self, standin_detector, monkeypatch,
                                                            batched):
        """The chunk path (one batch per step for the block — or, off the GPU
        lane, frame by frame) and the single-frame preview give the same
        record, with the real networks on the CPU."""
        import spyde.actions.find_vectors_neural as neural
        import spyde.actions.find_vectors_torch as find_vectors_torch

        stack, truth = _scan(np.random.default_rng(11))
        _register(standin_detector, stack, truth)
        monkeypatch.setattr(find_vectors_torch, "torch_gpu_device",
                            (lambda: "fake-gpu") if batched else (lambda: None))
        monkeypatch.setattr(centre_refine, "load_step", lambda model_id, device=None: _load(model_id))
        steps = dict(refine_centres=True, friedel_partner=True)
        block = neural._neural_block(stack.reshape(2, 2, *stack.shape[1:]), 0.3, 3, True, None,
                                     None, spot_radius=RADIUS, **steps)
        moved = []
        for index, frame in enumerate(stack):
            preview = _preview(frame, **steps)
            row = block[index // 2, index % 2]
            row = row[np.isfinite(row[:, 0])]
            np.testing.assert_allclose(row, preview, rtol=0, atol=1e-4)
            moved.append(np.hypot(*(preview[:, :2] - (truth[index] + 0.8)).T))
        assert np.mean(np.concatenate(moved)) > 0.05            # the networks moved the disks

    def test_the_preview_dispatch_passes_both_switches(self, standin_detector, standin_steps):
        from spyde.actions.find_vectors import _find_peaks_single_frame
        from spyde.actions.vector_overlay import _detector_params

        stack, truth = _scan(np.random.default_rng(13), frames=1)
        _register(standin_detector, stack, truth)
        for friedel, expected in ((True, [CENTRE_MODEL, FRIEDEL_MODEL]), (False, [CENTRE_MODEL])):
            standin_steps.clear()
            params = _detector_params({"method": "neural", "spot_radius": RADIUS,
                                       "threshold": 0.3, "refine_centres": True,
                                       "friedel_partner": friedel})
            _find_peaks_single_frame(stack[0], params)
            assert standin_steps == expected

    def test_the_wizard_runs_both_steps_by_default(self):
        from spyde.actions.find_vectors_action import DEFAULTS, _coerce

        assert DEFAULTS["refine_centres"] is True and DEFAULTS["friedel_partner"] is True
        assert _coerce({"friedel_partner": False})["friedel_partner"] is False
        assert _coerce({})["friedel_partner"] is True

    def test_the_networks_are_made_local_before_the_batch(self, monkeypatch):
        from spyde import models
        from spyde.actions.find_vectors_action import _ensure_model_local

        fetched = []
        monkeypatch.setattr(models, "is_cached", lambda model_id: True)
        monkeypatch.setattr(models, "ensure_local", fetched.append)
        _ensure_model_local({"method": "neural", "refine_centres": True, "friedel_partner": True})
        assert fetched == [None, CENTRE_MODEL, FRIEDEL_MODEL]
        fetched.clear()
        _ensure_model_local({"method": "neural", "refine_centres": True, "friedel_partner": False})
        assert fetched == [None, CENTRE_MODEL]

    def test_the_wizard_has_one_checkbox_on_by_default(self):
        """The Friedel step is a checkbox, on by default; there is no choice of
        refine step."""
        source = (Path(__file__).resolve().parents[3] / "electron" / "src" / "renderer" / "src"
                  / "components" / "FindVectorsWizard.tsx").read_text(encoding="utf-8")
        assert 'testid="fv-friedel"' in source
        assert "useState(saved?.friedelPartner ?? true)" in source
        assert "friedel_partner: neural && v.friedelPartner" in source
        assert "fv-centre" not in source

    def test_the_full_dataset_is_never_computed(self, standin_detector, standin_steps,
                                                monkeypatch):
        """The centre steps read windows from the chunk they are handed, never
        the whole lazy dataset: a guard on ``dask.array.Array.compute`` raises
        for the full shape, and every step call sees one chunk's frames."""
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
                           refine_centres=True, friedel_partner=True), None, None)
        assert vectors.nav_shape == (6, 6)
        assert frames_per_call and max(frames_per_call) < 36
        assert sum(frames_per_call) == 2 * 36                   # both steps, every frame


# ── the real detector, and the networks on CUDA, in one subprocess ─────────────

_DRIVER = textwrap.dedent(r"""
    import json, os, sys
    import numpy as np
    sys.path.insert(0, os.environ["SPYDE_TEST_DIR"])
    from test_centre_refiner import uneven_disks, disk_lattice

    def run_mode(mode):
        from spyde import models
        import spyde.actions.find_vectors_neural as neural
        import spyde.actions.find_vectors_torch as find_vectors_torch
        import torch
        out = {}
        if mode == "parity":
            # Everything on the CPU: the batch path and the preview then share
            # one model and one device, so any difference is the code's.
            models.get_model = models.get_cpu_model
            find_vectors_torch.torch_gpu_device = lambda: torch.device("cpu")
            from spyde.models import centre_refine, registry
            centre_refine.load_step = lambda model_id, device=None: registry.get_refiner(model_id, "cpu")
            rng = np.random.default_rng(0)
            truth = [disk_lattice(128, 7.0, rng, spacing=24.0) for _ in range(4)]
            frames = np.stack([uneven_disks((128, 128), t, 7.0, rng) for t in truth])
            for name, steps in (("refine", dict(refine_centres=True)),
                                ("refine+friedel", dict(refine_centres=True, friedel_partner=True))):
                block = neural._neural_block(frames.reshape(2, 2, *frames.shape[1:]), 0.3, 3,
                                             True, None, None, spot_radius=7.0, **steps)
                worst, moved = 0.0, []
                for index, frame in enumerate(frames):
                    preview = neural._find_vectors_single_frame_neural(
                        frame, 0.3, 3, spot_radius=7.0, **steps)[2]
                    decode = neural._find_vectors_single_frame_neural(
                        frame, 0.3, 3, spot_radius=7.0)[2]
                    row = block[index // 2, index % 2]
                    row = row[np.isfinite(row[:, 0])]
                    assert row.shape == preview.shape, (row.shape, preview.shape)
                    worst = max(worst, float(np.nanmax(np.abs(row - preview))))
                    moved.append(np.hypot(*(preview[:, :2] - decode[:, :2]).T))
                out[name] = dict(max_abs_difference=worst,
                                 mean_move=float(np.mean(np.concatenate(moved))))
        elif mode == "cuda":
            if not torch.cuda.is_available():
                return {"skipped": True}
            from test_centre_refiner import _f5_reference, _p4_reference, F5_FIXTURES
            from spyde.models import registry
            from spyde.models.centre_refine import CENTRE_MODEL, FRIEDEL_MODEL, refine_centres
            torch.nn.functional.linear(torch.zeros(1, 1, device="cuda"),
                                       torch.zeros(1, 1, device="cuda"))
            cases = {f"F5-{name}": (CENTRE_MODEL, _f5_reference(name)) for name in F5_FIXTURES}
            frames, incoming, radius, centres, sigma = _p4_reference()
            cases["P4-gan"] = (FRIEDEL_MODEL, (frames, incoming, radius, centres, sigma))
            for case, (model_id, (frames, seeds, radius, centres, sigma)) in cases.items():
                positions, sigmas = refine_centres(frames, seeds, radius,
                                                   registry.get_refiner(model_id, "cuda"))
                got = np.concatenate(sigmas)
                out[case] = dict(
                    centre_difference=float(np.abs(np.concatenate(positions) - centres).max()),
                    sigma_difference=float(np.nanmax(np.abs(got - sigma))),
                    declined_match=bool((np.isnan(got) == np.isnan(sigma)).all()))
        return out

    for mode in sys.argv[1:]:
        print("RESULT_JSON", mode, json.dumps(run_mode(mode)), flush=True)
    sys.stdout.flush()
    os._exit(0)
""")


@pytest.fixture(scope="module")
def network_results():
    import os

    environment = dict(os.environ, SPYDE_TEST_DIR=str(Path(__file__).parent))
    proc = subprocess.run([sys.executable, "-c", _DRIVER, "parity", "cuda"],
                          capture_output=True, text=True, timeout=900, env=environment)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            _, mode, payload = line.split(" ", 2)
            out[mode] = json.loads(payload)
    if len(out) < 2:
        pytest.fail(f"centre-step driver failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return out


class TestWithTheDetectorNetwork:
    def test_batch_and_preview_agree_with_and_without_the_friedel_step(self, network_results):
        for name, result in network_results["parity"].items():
            assert result["max_abs_difference"] < 1e-4, name

    def test_the_steps_actually_move_the_disks(self, network_results):
        for name, result in network_results["parity"].items():
            assert 0.02 < result["mean_move"] < 2.0, name

    def test_the_networks_match_the_training_code_on_cuda(self, network_results):
        """Against the CPU reference, so the device's own float32 rounding adds
        to the comparison: measured up to 8e-5 px for F5 and 3e-5 px for P4."""
        results = network_results["cuda"]
        if results.get("skipped"):
            pytest.skip("no CUDA device")
        for case, result in results.items():
            assert result["declined_match"], case
            tolerance = 0.01 if case.startswith("P4") else 1e-4
            assert result["centre_difference"] < tolerance, case
            assert result["sigma_difference"] < max(tolerance, 5e-5), case

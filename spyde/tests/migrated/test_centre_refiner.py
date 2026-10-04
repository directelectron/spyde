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
        refined, sigmas = refine_centres([frame], [seeds], radius, MaskCentroidRefiner())
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
        assert [m["id"] for m in available["refiners"]] == [STUB_ID]
        assert available["refiners"][0]["label"] == "Centre stub"
        assert available["default"] != STUB_ID
        assert stub_registry.default_model_id() in [m["id"] for m in available["models"]]

    def test_the_bundled_manifest_alone_has_no_refiner_yet(self, monkeypatch):
        from spyde.models import registry

        monkeypatch.setattr(registry, "_load_user_manifest", lambda: None)
        registry._invalidate_manifest()
        try:
            assert registry.available_models()["refiners"] == []
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
        assert centres.shape == (1, 2) and sigma is None

    def test_a_detector_is_not_a_refiner(self, stub_registry):
        with pytest.raises(ValueError):
            stub_registry.get_refiner(stub_registry.default_model_id(), "cpu")

    def test_an_unknown_normalisation_is_refused(self, tmp_path):
        from spyde.models import centre_network

        path = tmp_path / "stub.pt"
        _write_stub_refiner(path)
        with pytest.raises(ValueError, match="normalisation"):
            centre_network.load_refiner(path, "cpu", contract={"normalisation": "zscore"})


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
            for choice in ("mask-centroid", STUB_ID):
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
    proc = subprocess.run([sys.executable, "-c", _DRIVER, "parity", "accuracy"],
                          capture_output=True, text=True, timeout=900, env=environment)
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            _, mode, payload = line.split(" ", 2)
            out[mode] = json.loads(payload)
    if len(out) < 2:
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

    def test_the_mask_centroid_beats_the_decode_on_uneven_disks(self, network_results):
        result = network_results["accuracy"]
        assert result[CENTRE_MASK_CENTROID]["n"] == result[CENTRE_DECODE]["n"] > 50
        assert result[CENTRE_MASK_CENTROID]["rms"] < result[CENTRE_DECODE]["rms"]

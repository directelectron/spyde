"""The neural method's third stage: refine each frame's disks together
(``spyde.models.symmetry_refine``).

These tests pin:

* the stage interface (declines, the half-radius cap, sigma, the device lock);
* the Friedel-pair baseline on synthetic patterns with known centres;
* that an amorphous sample skips the stage;
* that the batch and the single-frame preview agree;
* that the stage only ever sees one chunk of positions, never pixels;
* the registry's ``"kind": "symmetry"`` entries.

A stand-in detector replaces the network, so nothing here runs torch.
"""
from __future__ import annotations

import json
import logging
import types

import numpy as np
import pytest

from spyde.models import symmetry_refine
from spyde.models.symmetry_refine import (
    SAMPLE_AMORPHOUS, SAMPLE_CRYSTALLINE, SYMMETRY_FRIEDEL, SYMMETRY_OFF,
    FrameDisks, FriedelRefiner, beam_from_records, refine_frames, symmetry_refiner_for,
)

RADIUS = 6.0
G1, G2 = np.array([0.4, 17.0]), np.array([15.5, -3.0])


def lattice(centre, extent=2):
    """Reflections ``h g1 + k g2`` about ``centre``: every one has its Friedel
    partner, and index ``len // 2`` is the direct beam."""
    return np.array([centre + h * G1 + k * G2
                     for h in range(-extent, extent + 1) for k in range(-extent, extent + 1)])


def noisy_frame(rng, centre=(60.3, 58.7)):
    truth = lattice(np.asarray(centre))
    sigma = rng.uniform(0.05, 0.5, len(truth))
    observed = truth + rng.normal(0, 1, truth.shape) * sigma[:, None]
    return truth, observed, sigma


class _Shift:
    """A symmetry refiner that moves every disk by a fixed amount."""

    def __init__(self, shift, declined=(), sigma=None, device=None):
        self.shift, self.declined, self.sigma, self.device = np.asarray(shift), set(declined), sigma, device
        self.calls = []

    def refine_frames(self, frames, spot_radius):
        self.calls.append(len(frames))
        out = []
        for frame in frames:
            moved = np.asarray(frame.positions, float) + self.shift
            for index in self.declined:
                moved[index] = np.nan
            sigma = None if self.sigma is None else np.full(len(moved), self.sigma)
            out.append((moved, sigma))
        return out


def _disks(points, sigma=None):
    points = np.asarray(points, float)
    return FrameDisks(points, np.full(len(points), np.nan) if sigma is None else sigma,
                      np.ones(len(points)), beam=points[len(points) // 2])


class TestInterface:
    def test_a_refined_position_replaces_the_incoming_one(self):
        positions, sigmas = refine_frames([_disks([[10, 10], [20, 20]])], RADIUS, _Shift([0.5, 0]))
        np.testing.assert_allclose(positions[0], [[10.5, 10], [20.5, 20]])
        assert sigmas is None

    def test_declined_or_far_moved_disks_keep_their_position(self):
        frame = _disks([[10, 10], [20, 20]])
        kept = refine_frames([frame], RADIUS, _Shift([0.5, 0], declined={0}))[0][0]
        np.testing.assert_allclose(kept, [[10, 10], [20.5, 20]])
        too_far = refine_frames([frame], RADIUS, _Shift([3.0, 0]))[0][0]     # = 0.5 R
        np.testing.assert_allclose(too_far, [[10, 10], [20, 20]])

    def test_the_refiners_sigma_is_carried_only_where_it_moved_a_disk(self):
        _, sigmas = refine_frames([_disks([[10, 10], [20, 20]])], RADIUS,
                                  _Shift([0.2, 0], declined={1}, sigma=0.07))
        assert sigmas[0][0] == pytest.approx(0.07) and np.isnan(sigmas[0][1])

    def test_the_stage_takes_the_accelerator_lock_for_the_refiners_device(self, monkeypatch):
        import contextlib

        import spyde.device_lock as device_lock

        seen = []

        @contextlib.contextmanager
        def spy(device=None, **_):
            seen.append(device)
            yield

        monkeypatch.setattr(device_lock, "accelerator_lock", spy)
        refine_frames([_disks([[10, 10]])], RADIUS, _Shift([0, 0], device="mps"))
        assert seen == ["mps"]

    def test_the_small_disk_gate(self):
        refiner = _Shift([0.5, 0])
        refiner.min_spot_radius = 5.0
        positions, sigmas = refine_frames([_disks([[10, 10]])], 3.0, refiner)
        assert refiner.calls == [] and sigmas is None
        np.testing.assert_allclose(positions[0], [[10, 10]])

    def test_the_beam_is_the_brightest_disk(self):
        records = np.array([[1, 2, 5.0], [3, 4, 50.0], [5, 6, 7.0]])
        np.testing.assert_allclose(beam_from_records(records), [3, 4])
        assert beam_from_records(np.zeros((0, 5))) is None

    def test_the_choices_resolve(self, caplog):
        assert symmetry_refiner_for(None) is None
        assert symmetry_refiner_for(SYMMETRY_OFF) is None
        assert isinstance(symmetry_refiner_for(SYMMETRY_FRIEDEL), FriedelRefiner)
        with caplog.at_level(logging.WARNING):
            assert symmetry_refiner_for("no-such-model") is None
        assert "skipping the symmetry stage" in caplog.text


class TestFriedelBaseline:
    def test_pairs_become_exactly_symmetric_about_the_fitted_centre(self):
        rng = np.random.default_rng(0)
        truth, observed, sigma = noisy_frame(rng)
        frame = FrameDisks(observed, sigma, np.ones(len(observed)), beam=observed[12])
        (refined, refined_sigma), = FriedelRefiner().refine_frames([frame], RADIUS)
        paired = np.isfinite(refined[:, 0])
        assert paired.sum() == len(truth) - 1                  # all but the beam
        centre = refined[paired].mean(0)
        np.testing.assert_allclose(centre, [60.3, 58.7], atol=0.15)
        mirrored = 2 * centre - refined[paired]
        nearest = np.hypot(mirrored[:, None, 0] - refined[paired][None, :, 0],
                           mirrored[:, None, 1] - refined[paired][None, :, 1]).min(1)
        assert nearest.max() < 1e-9

    def test_it_reduces_the_error_on_noisy_pairs(self):
        """Over 200 frames with per-disk noise of known sigma (0.05-0.5 px), the
        sigma-weighted symmetric estimate cuts the RMS error by about a quarter
        (measured 0.40 -> 0.30 px on one frame)."""
        rng = np.random.default_rng(1)
        before, after = [], []
        for _ in range(200):
            truth, observed, sigma = noisy_frame(rng, centre=rng.uniform(50, 70, 2))
            frame = FrameDisks(observed, sigma, np.ones(len(observed)), beam=observed[12])
            refined = refine_frames([frame], RADIUS, FriedelRefiner())[0][0]
            ring = np.hypot(*(truth - truth[12]).T) > 2 * RADIUS
            before.append(np.sum((observed - truth)[ring] ** 2, 1))
            after.append(np.sum((refined - truth)[ring] ** 2, 1))
        rms_before = np.sqrt(np.mean(np.concatenate(before)))
        rms_after = np.sqrt(np.mean(np.concatenate(after)))
        assert rms_after < 0.85 * rms_before

    def test_the_better_known_partner_wins(self):
        centre = np.array([50.0, 50.0])
        g = np.array([0.0, 20.0])
        observed = np.array([centre, centre + g + [0.6, 0], centre - g, centre + [20, 0], centre - [20, 0]])
        sigma = np.array([0.1, 1.0, 0.05, 0.1, 0.1])
        frame = FrameDisks(observed, sigma, np.ones(5), beam=centre)
        (refined, refined_sigma), = FriedelRefiner().refine_frames([frame], RADIUS)
        # the well-known -g (0.05 px) pins g; the poor +g (1 px) is pulled onto it
        assert abs(refined[1, 0] - centre[0]) < 0.05
        assert refined_sigma[1] == pytest.approx(1 / np.sqrt(1 / 1.0 ** 2 + 1 / 0.05 ** 2), rel=1e-5)

    def test_a_disk_without_a_partner_is_declined_and_kept(self):
        rng = np.random.default_rng(2)
        truth, observed, sigma = noisy_frame(rng)
        lonely = observed[:-1]                             # drop one partner
        frame = FrameDisks(lonely, sigma[:-1], np.ones(len(lonely)), beam=lonely[12])
        (refined, _), = FriedelRefiner().refine_frames([frame], RADIUS)
        assert np.isnan(refined[0]).all()                  # index 0 pairs with the dropped one
        positions, _ = refine_frames([frame], RADIUS, FriedelRefiner())
        np.testing.assert_allclose(positions[0][0], lonely[0], atol=1e-5)

    def test_without_sigma_the_partners_weigh_the_same_and_no_sigma_is_reported(self):
        rng = np.random.default_rng(3)
        _, observed, _ = noisy_frame(rng)
        frame = FrameDisks(observed, np.full(len(observed), np.nan), np.ones(len(observed)),
                           beam=observed[12])
        (refined, refined_sigma), = FriedelRefiner().refine_frames([frame], RADIUS)
        assert np.isfinite(refined[:, 0]).sum() == len(observed) - 1
        assert np.isnan(refined_sigma).all()


class TestSampleType:
    def test_amorphous_turns_the_stage_off_at_the_choke_point(self):
        from spyde.actions.find_vectors_action import DEFAULTS, _coerce

        assert DEFAULTS["symmetry_refiner"] == SYMMETRY_OFF
        assert DEFAULTS["sample_type"] == SAMPLE_CRYSTALLINE
        assert _coerce({"symmetry_refiner": "friedel"})["symmetry_refiner"] == "friedel"
        coerced = _coerce({"symmetry_refiner": "friedel", "sample_type": "amorphous"})
        assert coerced["symmetry_refiner"] == SYMMETRY_OFF
        assert _coerce({"sample_type": "glass"})["sample_type"] == SAMPLE_CRYSTALLINE

    def test_the_wizard_offers_the_backends_choices(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[3] / "electron" / "src" / "renderer" / "src"
                  / "components" / "FindVectorsWizard.tsx").read_text(encoding="utf-8")
        for value in (SYMMETRY_OFF, SYMMETRY_FRIEDEL, SAMPLE_CRYSTALLINE, SAMPLE_AMORPHOUS):
            assert f"value: '{value}'" in source
        assert "useState(saved?.symmetryRefiner ?? 'off')" in source
        assert "useState<SampleType>(saved?.sampleType ?? 'crystalline')" in source


# ── the Find Vectors wiring, with a stand-in detector ──────────────────────────

def _scan(rng, frames=4, size=128):
    """Frames of a lattice with a bright direct beam; the stand-in detector
    reports every disk with per-disk noise, so the stage has work to do."""
    from scipy.special import erfc

    rows, columns = np.mgrid[:size, :size].astype(float)
    stack, truths = [], []
    for _ in range(frames):
        truth = lattice(np.array([size / 2, size / 2]) + rng.uniform(-1, 1, 2))
        image = np.full((size, size), 5.0)
        for index, (y, x) in enumerate(truth):
            distance = np.hypot(rows - y, columns - x)
            amplitude = 400.0 if index == len(truth) // 2 else 60.0
            image += amplitude * 0.5 * erfc((distance - 4.0) / 0.8)
        stack.append(rng.poisson(image).astype(np.float32))
        truths.append(truth)
    return np.stack(stack), truths


@pytest.fixture
def standin_detector(monkeypatch):
    from spyde import models

    found = {}

    def rows(frame):
        truth = found[np.asarray(frame, np.float32).tobytes()]
        noise = np.random.default_rng(int(truth[0, 0] * 1000)).normal(0, 0.3, truth.shape)
        out = np.zeros((len(truth), 4), np.float32)
        out[:, :2] = truth + noise
        out[:, 2], out[:, 3] = 0.9, 1.5
        return out

    monkeypatch.setattr(models, "get_model", lambda mid=None: (None, types.SimpleNamespace(type="cpu")))
    monkeypatch.setattr(models, "detect", lambda model, frame, device, **_: rows(frame))
    monkeypatch.setattr(models, "detect_batch",
                        lambda model, frames, device, **_: [rows(f) for f in frames])
    return found


class TestFindVectorsWiring:
    @pytest.mark.parametrize("batched", [True, False])
    def test_batch_and_preview_agree(self, standin_detector, monkeypatch, batched):
        import spyde.actions.find_vectors_neural as neural
        import spyde.actions.find_vectors_torch as find_vectors_torch

        stack, truths = _scan(np.random.default_rng(5))
        for frame, truth in zip(stack, truths):
            standin_detector[frame.tobytes()] = truth
        monkeypatch.setattr(find_vectors_torch, "torch_gpu_device",
                            (lambda: "fake-gpu") if batched else (lambda: None))
        block = neural._neural_block(stack.reshape(2, 2, *stack.shape[1:]), 0.3, 3, True, None,
                                     None, spot_radius=4.0, symmetry_refiner=SYMMETRY_FRIEDEL)
        moved = []
        for index, frame in enumerate(stack):
            preview = neural._find_vectors_single_frame_neural(
                frame, 0.3, 3, spot_radius=4.0, symmetry_refiner=SYMMETRY_FRIEDEL)[2]
            plain = neural._find_vectors_single_frame_neural(frame, 0.3, 3, spot_radius=4.0)[2]
            row = block[index // 2, index % 2]
            row = row[np.isfinite(row[:, 0])]
            np.testing.assert_array_equal(row, preview)
            moved.append(np.hypot(*(preview[:, :2] - plain[:, :2]).T))
        assert np.mean(np.concatenate(moved)) > 0.05

    def test_amorphous_never_runs_the_stage(self, standin_detector, monkeypatch):
        from spyde.actions.find_vectors import _find_peaks_single_frame
        from spyde.actions.find_vectors_action import _coerce
        from spyde.actions.vector_overlay import _detector_params

        calls = []
        monkeypatch.setattr(symmetry_refine, "symmetry_refiner_for",
                            lambda choice, device=None: calls.append(choice) or None)
        stack, truths = _scan(np.random.default_rng(6), frames=1)
        standin_detector[stack[0].tobytes()] = truths[0]
        for sample, expected in ((SAMPLE_CRYSTALLINE, SYMMETRY_FRIEDEL), (SAMPLE_AMORPHOUS, SYMMETRY_OFF)):
            calls.clear()
            params = _detector_params(_coerce({"method": "neural", "spot_radius": 4.0,
                                               "symmetry_refiner": SYMMETRY_FRIEDEL,
                                               "sample_type": sample}))
            _find_peaks_single_frame(stack[0], params)
            assert calls == [expected]

    def test_the_stage_sees_one_chunk_of_positions_and_never_the_dataset(self, standin_detector,
                                                                         monkeypatch):
        from unittest.mock import patch

        import dask.array as da
        import hyperspy.api as hs

        from spyde.actions.find_vectors import _do_compute_vectors

        stack, truths = _scan(np.random.default_rng(7), frames=36, size=96)
        for frame, truth in zip(stack, truths):
            standin_detector[frame.tobytes()] = truth
        lazy = hs.signals.Signal2D(da.from_array(stack.reshape(6, 6, 96, 96), chunks=(3, 3, 96, 96)))
        full_shape = lazy.data.shape
        seen = []
        original = FriedelRefiner.refine_frames

        def spy(self, frames, spot_radius):
            seen.append(len(frames))
            for frame in frames:          # positions and per-disk numbers only, never pixels
                assert frame.positions.ndim == 2 and frame.positions.shape[1] == 2
                assert frame.embeddings is None
            return original(self, frames, spot_radius)

        original_compute = da.Array.compute

        def guarded(self, *args, **kwargs):
            if self.shape == full_shape:
                raise AssertionError(f"full-dataset compute {self.shape}")
            return original_compute(self, *args, **kwargs)

        monkeypatch.setattr(FriedelRefiner, "refine_frames", spy)
        with patch.object(da.Array, "compute", guarded):
            vectors = _do_compute_vectors(
                lazy, dict(method="neural", sigma=0.0, kernel_radius=4, threshold=0.3,
                           min_distance=3, subpixel=True, spot_radius=4.0,
                           symmetry_refiner=SYMMETRY_FRIEDEL), None, None)
        assert vectors.nav_shape == (6, 6)
        assert seen and max(seen) < 36 and sum(seen) == 36


# ── the registry's symmetry models ─────────────────────────────────────────────

class _StubSymmetryModel:
    def __init__(self, path, device, arch, contract):
        self.path, self.device, self.arch, self.contract = path, device, arch, contract

    def refine_frames(self, frames, spot_radius):
        return [(np.asarray(f.positions, float), None) for f in frames]


@pytest.fixture
def symmetry_registry(tmp_path, monkeypatch):
    from spyde.models import registry

    weights = tmp_path / "symmetry_stub.pt"
    weights.write_bytes(b"stub")
    (tmp_path / "registry.json").write_text(json.dumps({"models": [
        {"id": "symmetry-stub", "kind": "symmetry", "label": "Symmetry stub",
         "arch": {"layout": "stub"}, "input": {"max_disks": 128, "features": ["position", "sigma"]},
         "source": {"type": "file", "path": str(weights)}},
        {"id": "symmetry-unknown", "kind": "symmetry", "arch": {"layout": "nope"},
         "source": {"type": "file", "path": str(weights)}},
    ]}))
    monkeypatch.setattr(registry, "user_models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(registry, "_REFINER_CACHE", {})
    monkeypatch.setitem(registry.SYMMETRY_LAYOUTS, "stub", _StubSymmetryModel)
    registry._invalidate_manifest()
    yield registry
    registry._invalidate_manifest()


class TestSymmetryRegistry:
    def test_symmetry_models_are_listed_apart(self, symmetry_registry):
        available = symmetry_registry.available_models()
        assert [m["id"] for m in available["symmetry"]] == ["symmetry-stub", "symmetry-unknown"]
        for other in ("models", "refiners"):
            assert not {"symmetry-stub", "symmetry-unknown"} & {m["id"] for m in available[other]}
        assert available["default"] not in ("symmetry-stub", "symmetry-unknown")

    def test_a_layout_loads_with_its_contract(self, symmetry_registry):
        model = symmetry_registry.get_symmetry_refiner("symmetry-stub", "cpu")
        assert isinstance(model, _StubSymmetryModel)
        assert model.contract == {"max_disks": 128, "features": ["position", "sigma"]}
        assert model is symmetry_registry.get_symmetry_refiner("symmetry-stub", "cpu")
        assert isinstance(symmetry_refiner_for("symmetry-stub", "cpu"), _StubSymmetryModel)

    def test_an_unknown_layout_skips_the_stage(self, symmetry_registry, caplog):
        with caplog.at_level(logging.WARNING):
            assert symmetry_refiner_for("symmetry-unknown", "cpu") is None
        assert "unknown layout" in caplog.text

    def test_a_refiner_is_not_a_symmetry_model(self, symmetry_registry):
        with pytest.raises(ValueError):
            symmetry_registry.get_symmetry_refiner("centre-fast-f3-v1", "cpu")

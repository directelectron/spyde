"""Adapting a centre network to one scan (``spyde.models.centre_adapt``).

The network here is a stand-in with one trainable number: coverage where the
normalised crop is above a threshold. Started too high, it covers only the
bright part of an unevenly lit disk, so its centres lean toward each disk's
lobe — a per-disk bias that varies across the scan, which is what adaptation
exists to remove. Started at 0.3 on evenly lit disks it has nothing to fix.
Everything runs on the CPU in seconds.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from scipy.special import erfc

from spyde.models.centre_adapt import (
    AdaptSettings, accept, adapt_centre_refiner, scan_index, varying,
)
from spyde.models.centre_network import NetworkRefiner

RADIUS = 6.0
SIZE = 128
G1, G2 = np.array([0.5, 21.0]), np.array([19.0, -4.0])


class Threshold(torch.nn.Module):
    def __init__(self, start):
        super().__init__()
        self.threshold = torch.nn.Parameter(torch.tensor(float(start)))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return 20.0 * (x - self.threshold)


def make_scan(rows, columns, uneven=True, extent=2, seed=0):
    rng = np.random.default_rng(seed)
    grid_y, grid_x = np.mgrid[:SIZE, :SIZE].astype(float)
    frames, truths = [], []
    for _ in range(rows * columns):
        centre = np.array([SIZE / 2, SIZE / 2]) + rng.uniform(-1, 1, 2)
        truth = np.array([centre + h * G1 + k * G2
                          for h in range(-extent, extent + 1) for k in range(-extent, extent + 1)])
        image = np.full((SIZE, SIZE), 5.0)
        for index, (y, x) in enumerate(truth):
            distance = np.hypot(grid_y - y, grid_x - x)
            edge = 0.5 * erfc((distance - RADIUS) / (np.sqrt(2) * 0.6))
            fill = 1.0
            if uneven:
                angle = rng.uniform(0, 2 * np.pi)
                along = ((grid_x - x) * np.cos(angle) + (grid_y - y) * np.sin(angle)) / RADIUS
                lobe = np.exp(-((grid_x - x - 0.45 * RADIUS * np.cos(angle)) ** 2
                                + (grid_y - y - 0.45 * RADIUS * np.sin(angle)) ** 2)
                              / (2 * (0.35 * RADIUS) ** 2))
                fill = np.clip(1 + 0.4 * along, 0, None) + 0.8 * lobe
            amplitude = 2000.0 if index == len(truth) // 2 else 300.0
            image += amplitude * edge * fill
        frames.append(rng.poisson(image).astype(np.float32))
        truths.append(truth)
    return np.stack(frames), truths


def detector_for(frames, truths):
    lookup = {frame.tobytes(): truth for frame, truth in zip(frames, truths)}
    rng = np.random.default_rng(5)

    def detect(stack):
        return [lookup[f.tobytes()] + rng.normal(0, 0.7, lookup[f.tobytes()].shape) for f in stack]

    return detect


def row_chunks(frames, rows, columns, rows_per_chunk=3, seen=None):
    for top in range(0, rows, rows_per_chunk):
        block = frames[top * columns:(top + rows_per_chunk) * columns]
        if seen is not None:
            seen.append(len(block))
        yield block, np.repeat(np.arange(top, min(top + rows_per_chunk, rows)), columns)


def refiner(start):
    return NetworkRefiner(Threshold(start), torch.device("cpu"), crop_radius=10.0, crop_half=16,
                          normalisation="mean", uncertainty="mirror", passes=1, min_spot_radius=0)


SETTINGS = AdaptSettings(budget_seconds=60, max_steps=150, learning_rate=0.01)
ROWS, COLUMNS = 15, 12


@pytest.fixture(scope="module")
def biased_scan():
    return make_scan(ROWS, COLUMNS, uneven=True)


@pytest.fixture(scope="module")
def even_scan():
    return make_scan(ROWS, COLUMNS, uneven=False, seed=1)


def run(scan, start, settings=SETTINGS, **kwargs):
    frames, truths = scan
    base = refiner(start)
    adapted, report = adapt_centre_refiner(row_chunks(frames, ROWS, COLUMNS), RADIUS, base,
                                           detector_for(frames, truths), (SIZE, SIZE), settings,
                                           **kwargs)
    return base, adapted, report


class TestAdaptation:
    def test_a_scan_varying_bias_is_reduced_and_accepted(self, biased_scan):
        """Measured: varying Friedel error 0.51 -> 0.07 px, lattice 0.92 ->
        0.14 px, on the 36 held-out frames (every fifth row)."""
        base, adapted, report = run(biased_scan, start=0.9)
        assert report.accepted and adapted is not None
        assert report.held_out_frames == 3 * COLUMNS and report.train_frames == 12 * COLUMNS
        assert report.after["friedel"] < 0.5 * report.before["friedel"]
        assert report.after["lattice"] < 0.5 * report.before["lattice"]
        assert float(adapted.net.threshold) < 0.9
        assert float(base.net.threshold) == pytest.approx(0.9)          # the base is untouched
        assert 0 < report.steps <= SETTINGS.max_steps

    def test_a_scan_with_nothing_to_fix_is_rejected(self, even_scan):
        _, adapted, report = run(even_scan, start=0.3)
        assert not report.accepted and adapted is None and report.declined is None
        assert report.after["friedel"] > SETTINGS.friedel_ratio * report.before["friedel"]

    def test_a_sparse_scan_is_declined_before_training(self):
        frames, truths = make_scan(ROWS, COLUMNS, uneven=True, extent=0)   # one disk a frame
        base = refiner(0.9)
        adapted, report = adapt_centre_refiner(row_chunks(frames, ROWS, COLUMNS), RADIUS, base,
                                               detector_for(frames, truths), (SIZE, SIZE), SETTINGS)
        assert adapted is None and not report.accepted
        assert report.declined and "four reflections" in report.declined
        assert report.steps == 0 and report.train_seconds == 0

    def test_it_can_be_cancelled(self, biased_scan):
        cancel = [False]

        def progress(stage, done, total):
            if stage == "training" and done >= 5:
                cancel[0] = True

        base, adapted, report = run(biased_scan, start=0.9, progress=progress, cancel=cancel)
        assert report.cancelled and adapted is None and not report.accepted
        assert float(base.net.threshold) == pytest.approx(0.9)

    def test_the_scan_is_read_a_chunk_at_a_time(self, biased_scan):
        frames, truths = biased_scan
        seen = []
        adapt_centre_refiner(row_chunks(frames, ROWS, COLUMNS, seen=seen), RADIUS, refiner(0.9),
                             detector_for(frames, truths), (SIZE, SIZE),
                             AdaptSettings(budget_seconds=1, max_steps=1))
        assert seen == [3 * COLUMNS] * 5

    def test_kept_crops_are_capped(self, biased_scan):
        """Only per-disk crops of at most train_cap + held_out_cap frames are
        held, whatever the scan's size."""
        settings = AdaptSettings(budget_seconds=1, max_steps=1, train_cap=40, held_out_cap=25,
                                 min_train=10, min_held_out=5)
        _, _, report = run(biased_scan, start=0.9, settings=settings)
        assert report.train_frames + report.held_out_frames <= 65

    def test_the_report_is_json(self, biased_scan):
        _, _, report = run(biased_scan, start=0.9, settings=AdaptSettings(budget_seconds=1, max_steps=2))
        json.dumps(report.to_dict())


class TestScanIndex:
    def test_reflections_pairs_and_lattice_are_found(self, biased_scan):
        frames, truths = biased_scan
        index = scan_index([t + 0.0 for t in truths], (SIZE, SIZE), RADIUS)
        assert len(index.reflections) == 24                 # 5 x 5 lattice minus the beam
        assert len(index.pairs) == 12
        assert np.isfinite(index.hk[:, 0]).all()
        assert all((a >= 0).sum() == 24 for a in index.index)

    def test_varying_ignores_a_constant_offset(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0, 0.1, (500, 6, 2)) + np.arange(6)[None, :, None]
        assert varying(values) == pytest.approx(0.1, rel=0.15)


class TestAcceptance:
    before = dict(friedel=1.0, lattice=1.0, noise=0.05, refined=0.99)

    @pytest.mark.parametrize("after, expected", [
        (dict(friedel=0.9, lattice=1.0, noise=0.05, refined=0.99), True),
        (dict(friedel=0.99, lattice=1.0, noise=0.05, refined=0.99), False),   # gain < 2 %
        (dict(friedel=0.9, lattice=1.05, noise=0.05, refined=0.99), False),   # lattice worse
        (dict(friedel=0.9, lattice=1.0, noise=0.08, refined=0.99), False),    # noise added
    ])
    def test_the_three_tests(self, after, expected):
        assert accept(self.before, after, AdaptSettings()) is expected

    def test_a_missing_measure_skips_its_test_and_both_missing_rejects(self):
        nan = float("nan")
        no_lattice = dict(self.before, lattice=nan)
        assert accept(no_lattice, dict(friedel=0.9, lattice=nan, noise=0.05, refined=0.99),
                      AdaptSettings())
        nothing = dict(self.before, friedel=nan, lattice=nan)
        assert not accept(nothing, dict(nothing), AdaptSettings())

    def test_the_thresholds_are_parameters(self):
        after = dict(friedel=0.97, lattice=1.0, noise=0.05, refined=0.99)
        assert accept(self.before, after, AdaptSettings())
        assert not accept(self.before, after, AdaptSettings(friedel_ratio=0.95))

    def test_a_refined_fraction_drop_is_reported_not_enforced_by_default(self):
        after = dict(friedel=0.9, lattice=1.0, noise=0.05, refined=0.90)
        assert accept(self.before, after, AdaptSettings())
        assert not accept(self.before, after, AdaptSettings(max_refined_drop=0.02))


# ── adapted refiners in the registry ───────────────────────────────────────────

@pytest.fixture
def user_folder(tmp_path, monkeypatch):
    from spyde.models import registry

    monkeypatch.setattr(registry, "user_models_dir", lambda: str(tmp_path))
    monkeypatch.setattr(registry, "_REFINER_CACHE", {})
    registry._invalidate_manifest()
    yield tmp_path, registry
    registry._invalidate_manifest()


class TestAdaptedRegistry:
    def _adapted(self, registry):
        base = registry.get_refiner("centre-fast-f5-v1", "cpu")
        import copy
        adapted = copy.copy(base)
        adapted.net = copy.deepcopy(base.net)
        with torch.no_grad():
            next(adapted.net.parameters()).add_(0.01)
        return adapted

    def test_an_adapted_refiner_is_written_listed_and_loadable(self, user_folder):
        folder, registry = user_folder
        adapted = self._adapted(registry)
        adapted.sigma_scale, adapted.max_shift_fraction = 1.5, 0.75
        entry = registry.register_adapted(adapted, "centre-fast-f5-v1", "GaN MQW", {"accepted": True},
                                          session="this")
        assert entry["parent"] == "centre-fast-f5-v1" and entry["scope"] == {"dataset": "GaN MQW"}
        assert entry["label"] == "Fast centre network F5 adapted to GaN MQW"
        assert entry["report"] == {"accepted": True} and not entry["saved"]
        listed = {m["id"]: m for m in registry.available_models()["refiners"]}
        assert listed[entry["id"]]["parent"] == "centre-fast-f5-v1"
        registry._REFINER_CACHE.clear()               # as a compute worker would load it
        loaded = registry.get_refiner(entry["id"], "cpu")
        for a, b in zip(loaded.net.state_dict().values(), adapted.net.state_dict().values()):
            assert torch.equal(a, b)
        assert loaded.crop_half == adapted.crop_half
        # the recalibrated scale and the copy's own move limit travel with it
        assert (loaded.sigma_scale, loaded.max_shift_fraction) == (1.5, 0.75)

    def test_unsaved_ones_from_another_session_are_pruned(self, user_folder):
        folder, registry = user_folder
        old = registry.register_adapted(self._adapted(registry), "centre-fast-f5-v1", "a", {}, session="old")
        kept = registry.register_adapted(self._adapted(registry), "centre-fast-f5-v1", "b", {}, session="old")
        current = registry.register_adapted(self._adapted(registry), "centre-fast-f5-v1", "c", {},
                                            session="now")
        assert registry.keep_adapted(kept["id"])
        assert registry.prune_unsaved_adapted("now") == [old["id"]]
        ids = {m["id"] for m in registry.list_models(kind=None)}
        assert old["id"] not in ids and {kept["id"], current["id"]} <= ids
        assert not (folder / "adapted" / f"{old['id']}.pt").exists()
        bundled = {m["id"] for m in registry.available_models()["refiners"]}
        assert {"centre-fast-f3-v1", "centre-fast-f5-v1"} <= bundled


class TestAction:
    def _call(self, monkeypatch, payload):
        import types

        import spyde.actions.centre_adapt_action as action

        sent = []
        monkeypatch.setattr(action, "emit", lambda message: sent.append(message))
        signal = types.SimpleNamespace(axes_manager=types.SimpleNamespace(
            navigation_dimension=2, signal_dimension=2))
        tree = types.SimpleNamespace(root=signal)
        monkeypatch.setattr(action, "_src_plot_tree", lambda session, plot: (object(), tree))
        monkeypatch.setattr(action, "_current_signal", lambda src: signal)
        action.fv_adapt_centre(None, types.SimpleNamespace(window_id=7), dict(payload, window_id=7))
        return [m for m in sent if m.get("type") == "fv_adapt_result"]

    def test_the_decode_and_mask_centroid_cannot_be_adapted(self, monkeypatch):
        for choice in ("decode", "mask-centroid"):
            (result,) = self._call(monkeypatch, {"method": "neural", "centre_refiner": choice,
                                                 "spot_radius": 8})
            assert not result["accepted"] and "centre network" in result["declined"]

    def test_spots_under_the_networks_gate_are_declined(self, monkeypatch):
        (result,) = self._call(monkeypatch, {"method": "neural", "centre_refiner": "centre-fast-f5-v1",
                                             "spot_radius": 3})
        assert not result["accepted"] and "5 px" in result["declined"]

    def test_the_lazy_scan_is_read_in_storage_row_chunks_never_whole(self, monkeypatch):
        """The action's reader against a lazy scan: each read is one stored row
        chunk, and the whole dataset is never computed."""
        from unittest.mock import patch

        import dask.array as da

        import spyde.actions.centre_adapt_action as action

        data = da.zeros((10, 6, 32, 32), chunks=(4, 6, 32, 32), dtype=np.uint16)
        signal = type("Signal", (), {"data": data})()
        original = da.Array.compute

        def guarded(self, *args, **kwargs):
            if self.shape == data.shape:
                raise AssertionError("full-dataset compute")
            return original(self, *args, **kwargs)

        with patch.object(da.Array, "compute", guarded):
            reads = [(frames.shape, rows.tolist()) for frames, rows in action._row_chunks(signal)]
        assert [shape for shape, _ in reads] == [(24, 32, 32), (24, 32, 32), (12, 32, 32)]
        assert reads[2][1] == [8] * 6 + [9] * 6


# ── the adapted copy's own move limit and sigma calibration ────────────────────

class ThresholdWithSigma(Threshold):
    """The stand-in with a sigma head: one trainable log sigma for every disk."""

    def __init__(self, start, log_sigma=-1.0):
        super().__init__(start)
        self.log_sigma = torch.nn.Parameter(torch.tensor(float(log_sigma)))

    def forward(self, x):
        return super().forward(x), self.log_sigma.expand(len(x))


class TestRoundSix:
    def test_the_refined_fraction_does_not_drop_and_the_copy_carries_its_move_limit(self, biased_scan):
        _, adapted, report = run(biased_scan, start=0.9)
        assert report.accepted
        assert report.after["refined"] >= report.before["refined"]
        assert adapted.max_shift_fraction == pytest.approx(SETTINGS.moved_limit) == 0.75

    def test_a_refiners_own_move_limit_is_honoured(self):
        from spyde.models.centre_refine import refine_centres

        class Shift:
            def __init__(self, limit=None):
                if limit is not None:
                    self.max_shift_fraction = limit

            def crop_half_width(self, spot_radius):
                return 16

            def refine(self, crops, centres, spot_radius):
                return centres + [0.6 * spot_radius, 0.0], None

        frame = np.zeros((64, 64), np.float32)
        detection = [np.array([[30.0, 30.0]])]
        kept = refine_centres([frame], detection, 6.0, Shift())[0][0]
        moved = refine_centres([frame], detection, 6.0, Shift(limit=0.75))[0][0]
        np.testing.assert_allclose(kept, [[30.0, 30.0]])
        np.testing.assert_allclose(moved, [[33.6, 30.0]], atol=1e-5)

    def test_the_move_limit_and_sigma_scale_travel_with_the_checkpoint(self, tmp_path):
        from spyde.models import centre_network, registry

        path = tmp_path / "adapted.pt"
        checkpoint = torch.load(registry._resolve_weights(registry._entry("centre-fast-f5-v1")),
                                weights_only=True)
        checkpoint.update(decline_moved_over=0.75, sigma_scale=1.7)
        torch.save(checkpoint, path)
        loaded = centre_network.load_refiner(path, "cpu")
        assert (loaded.max_shift_fraction, loaded.sigma_scale) == (0.75, 1.7)
        overridden = centre_network.load_refiner(path, "cpu", contract={"decline_moved_over": 0.6})
        assert overridden.max_shift_fraction == 0.6
        assert centre_network.load_refiner(
            registry._resolve_weights(registry._entry("centre-fast-f5-v1")), "cpu"
        ).max_shift_fraction == 0.5

    def test_the_friedel_scale_fit_recovers_a_known_factor(self):
        """Pairs whose midpoints scatter with the predicted sigma fit to 1; the
        same scatter with sigmas predicted half as large fits to 2."""
        from spyde.models.centre_adapt import friedel_sigma_scale

        rng = np.random.default_rng(0)
        frames, pairs = 400, 6
        partner = np.array([[2 * k + 1, 2 * k] for k in range(pairs)]).ravel()
        g = rng.uniform(-40, 40, (pairs, 2))
        sigma = rng.uniform(0.1, 0.5, (frames, 2 * pairs))
        positions, sigmas, index = [], [], []
        for n in range(frames):
            truth = np.concatenate([np.stack([60 + gk, 60 - gk]) for gk in g])
            positions.append(truth + rng.normal(0, 1, truth.shape) * sigma[n][:, None])
            sigmas.append(sigma[n])
            index.append(np.arange(2 * pairs))
        assert friedel_sigma_scale(positions, sigmas, index, partner, 2 * pairs) == pytest.approx(1.0, rel=0.1)
        halved = [s / 2 for s in sigmas]
        assert friedel_sigma_scale(positions, halved, index, partner, 2 * pairs) == pytest.approx(2.0, rel=0.1)

    def test_the_sigma_scale_is_recalibrated(self, biased_scan):
        frames, truths = biased_scan
        base = NetworkRefiner(ThresholdWithSigma(0.9), torch.device("cpu"), crop_radius=10.0,
                              crop_half=16, normalisation="mean", uncertainty="head", passes=1,
                              min_spot_radius=0, max_sigma_fraction=10.0, sigma_scale=1.3)
        adapted, report = adapt_centre_refiner(row_chunks(frames, ROWS, COLUMNS), RADIUS, base,
                                               detector_for(frames, truths), (SIZE, SIZE), SETTINGS)
        assert report.sigma["trained_scale"] == 1.3
        assert math.isfinite(report.sigma["friedel_scale_base"])
        assert math.isfinite(report.sigma["friedel_scale_adapted"])
        expected = 1.3 * report.sigma["friedel_scale_adapted"] / report.sigma["friedel_scale_base"]
        assert report.sigma["recalibrated_scale"] == pytest.approx(expected)
        assert adapted is not None and adapted.sigma_scale == pytest.approx(expected)
        assert base.sigma_scale == 1.3

"""GUI-wiring test for the neural (SpotUNet) find-vectors method.

Exercises the full dispatch + chunk + CSR-packing path with the neural detector,
forcing the CPU branch (``torch_gpu_device`` → None) so it's deterministic and
avoids the torch-CUDA-under-pytest segfault on Windows (see CLAUDE.md). Confirms:

  - the wizard default method is now ``neural`` and the model registry resolves
    the bundled model,
  - ``fv_models`` emits the available-models payload for the Model dropdown,
  - ``fv_refresh_models`` refreshes the remote registry and re-emits the list,
  - the one-shot auto-calibration (``_emit_calibration``) emits ``fv_calibration``
    once, caches on the tree and respects the run generation,
  - ``bg_sigma`` reaches the single-frame preview dispatch (preview/batch parity),
  - ``fv_run`` with ``method="neural"`` produces a vectors window with
    ``tree.diffraction_vectors`` attached.

Detection *quality* is not asserted (the network's accuracy is benchmarked
separately on real data) — only that the wiring produces a valid vectors image.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=40.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def _diffraction_4d(nav=(4, 5), sig=(32, 32), scale=0.1):
    """A small 4D-STEM stack with a few bright Gaussian disks per pattern (so the
    neural detector reliably fires)."""
    from scipy.ndimage import gaussian_filter
    data = np.zeros(nav + sig, dtype=np.float32)
    base = np.zeros(sig, np.float32)
    for (cy, cx) in [(16, 16), (8, 22), (24, 9)]:
        base[cy, cx] = 400.0
    base = gaussian_filter(base, 2.0)
    for idx in np.ndindex(*nav):
        data[idx] = base
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = 0.0
        ax.units = "1/nm"
    return s


class TestFindVectorsNeural:
    def test_neural_default_and_models_payload(self):
        from spyde.actions.find_vectors_action import _coerce, fv_models
        from spyde.models import default_model_id

        # Neural is the default method out of the box.
        assert _coerce({})["method"] == "neural"
        assert abs(_coerce({})["threshold"] - 0.3) < 1e-9

        # fv_models emits the registry's available models (for the Model dropdown).
        import spyde.actions.find_vectors_action as fva
        captured = []
        orig = fva.emit
        fva.emit = lambda msg: captured.append(msg)
        try:
            class _P:
                window_id = 3
            fv_models(None, _P(), {"window_id": 3})
        finally:
            fva.emit = orig
        assert captured and captured[0]["type"] == "fv_models"
        assert captured[0]["default"] == default_model_id()
        assert any(m["id"] == default_model_id() for m in captured[0]["models"])

    def test_coerce_carries_bg_sigma(self):
        from spyde.actions.find_vectors_action import _coerce
        assert _coerce({})["bg_sigma"] == 12.0
        assert _coerce({"bg_sigma": 8})["bg_sigma"] == 8.0

    def test_coerce_neural_no_nav_blur_and_spot_radius(self):
        """Nav blur is NEVER applied for neural (forced to 0 even if sent), it
        defaults to 0 for every method, and spot_radius coerces through."""
        from spyde.actions.find_vectors_action import _coerce
        assert _coerce({})["sigma"] == 0.0                       # default off
        assert _coerce({"method": "neural", "sigma": 2.5})["sigma"] == 0.0
        assert _coerce({"method": "nxcorr", "sigma": 2.5})["sigma"] == 2.5
        assert _coerce({})["spot_radius"] == 0.0                 # 0 = auto
        assert _coerce({"spot_radius": 7})["spot_radius"] == 7.0

    def test_fv_refresh_models_emits(self, monkeypatch):
        """fv_refresh_models pulls the remote registry (offline-safe) and re-emits
        the fv_models payload with refreshed:true for the wizard status line."""
        import spyde.actions.find_vectors_action as fva
        from spyde import models as smodels

        called = []

        def _fake_refresh():
            called.append(1)
            return smodels.available_models()

        monkeypatch.setattr(smodels, "refresh_remote_registry", _fake_refresh)
        captured = []
        monkeypatch.setattr(fva, "emit", lambda msg: captured.append(msg))

        class _P:
            window_id = 7

        # session=None → run_on_worker executes inline (bare-stub path).
        fva.fv_refresh_models(None, _P(), {"window_id": 7})
        assert called, "refresh_remote_registry never called"
        assert captured and captured[0]["type"] == "fv_models"
        assert captured[0]["refreshed"] is True
        assert captured[0]["window_id"] == 7
        assert captured[0]["models"], "merged manifest lost its models"

    def test_emit_calibration_wiring(self, monkeypatch):
        """fv_calibration is emitted once with the calibrated values, cached on
        the tree (no recompute on caret reopen) and dropped on a stale generation."""
        import spyde.actions.find_vectors_action as fva
        import spyde.actions.find_vectors_neural as fvn
        from spyde.actions.lifecycle import bump_generation

        calls = []

        def _fake_cal(frames, *, sigma=1.0, model_id=None, spot_radius=None):
            calls.append(len(frames))
            return {"bg_sigma": 8.0, "thresh": 0.22, "scale_factor": 1.0,
                    "confidence": 0.5}

        monkeypatch.setattr(fvn, "calibrate_neural", _fake_cal)
        captured = []
        monkeypatch.setattr(fva, "emit", lambda m: captured.append(m))

        class _Tree:
            pass

        tree = _Tree()
        tree.root = _diffraction_4d()
        gen = bump_generation(tree, "_fv_run_gen")

        class _P:
            window_id = 5

        fva._emit_calibration(_P(), tree, {"sigma": 1.0, "model_id": ""}, gen)
        assert captured and captured[0]["type"] == "fv_calibration"
        assert captured[0]["window_id"] == 5
        assert captured[0]["bg_sigma"] == 8.0
        assert captured[0]["thresh"] == 0.22
        assert captured[0]["confidence"] == 0.5
        assert calls and calls[0] >= 1, "no sample frames reached calibrate"
        assert tree._fv_calibration["bg_sigma"] == 8.0

        # Cached: a second wizard-open re-emits WITHOUT recomputing.
        fva._emit_calibration(_P(), tree, {"sigma": 1.0}, gen)
        assert len(calls) == 1 and len(captured) == 2

        # Stale generation (wizard closed while calibrating): nothing emitted.
        bump_generation(tree, "_fv_run_gen")
        fva._emit_calibration(_P(), tree, {"sigma": 1.0}, gen)
        assert len(captured) == 2

    def test_preview_dispatch_passes_bg_sigma(self, monkeypatch):
        """The live-preview dispatch must forward bg_sigma to the neural detector —
        otherwise preview and batch run with DIFFERENT high-pass scales."""
        import spyde.actions.find_vectors_neural as fvn
        from spyde.actions.find_vectors import _find_peaks_single_frame

        seen = {}

        def _fake_single(frame, threshold, min_distance, *, subpixel=True,
                         beamstop_mask=None, model_id=None, bg_sigma=12.0,
                         spot_radius=None):
            seen.update(threshold=threshold, bg_sigma=bg_sigma,
                        model_id=model_id, spot_radius=spot_radius)
            z = np.zeros(frame.shape, np.float32)
            return z, z, np.zeros((0, 3), np.float32)

        monkeypatch.setattr(fvn, "_find_vectors_single_frame_neural", _fake_single)
        frame = np.zeros((16, 16), np.float32)
        _find_peaks_single_frame(
            frame, {"method": "neural", "bg_sigma": 7.5, "threshold": 0.25,
                    "spot_radius": 6})
        assert seen["bg_sigma"] == 7.5
        assert seen["spot_radius"] == 6.0
        assert abs(seen["threshold"] - 0.25) < 1e-9

    def test_spot_size_drives_model_scale(self, monkeypatch):
        """The Spot-size override reaches models.detect as spot_diameter=2·radius
        (the canonical-rescale factor then derives from it, not the estimate)."""
        import spyde.actions.find_vectors_neural as fvn
        from spyde import models as smodels

        seen = {}
        monkeypatch.setattr(smodels, "get_model", lambda mid=None: (None, "cpu"))

        def _fake_detect(model, f, device, thresh=0.3, min_distance=4,
                         auto_scale=True, bg_sigma=12.0, spot_diameter=None):
            seen.update(spot_diameter=spot_diameter)
            return np.zeros((0, 3), np.float32)

        monkeypatch.setattr(smodels, "detect", _fake_detect)
        frame = np.zeros((16, 16), np.float32)
        fvn._find_vectors_single_frame_neural(frame, spot_radius=6.0)
        assert seen["spot_diameter"] == 12.0
        fvn._find_vectors_single_frame_neural(frame)          # no override → auto
        assert seen["spot_diameter"] is None

    def test_gpu_policy_modes(self, monkeypatch):
        """SPYDE_FV_GPU governs the neural path too: off → CPU everywhere;
        unset → the caller's default (neural passes "all")."""
        from spyde.actions.find_vectors.gpu_runtime import _gpu_task_allowed

        monkeypatch.delenv("SPYDE_FV_GPU", raising=False)
        assert _gpu_task_allowed(default_mode="all") is True
        assert _gpu_task_allowed() is True          # outside a worker: allowed

        monkeypatch.setenv("SPYDE_FV_GPU", "off")
        assert _gpu_task_allowed(default_mode="all") is False
        assert _gpu_task_allowed() is False

    def test_neural_block_respects_gpu_off(self, monkeypatch):
        """SPYDE_FV_GPU=off must keep the batched torch path off even when a
        GPU device is present — every frame goes through the per-frame CPU
        detector instead."""
        import spyde.actions.find_vectors_neural as fvn
        import spyde.actions.find_vectors_torch as fvt
        from spyde import models as smodels

        monkeypatch.setenv("SPYDE_FV_GPU", "off")
        monkeypatch.setattr(fvt, "torch_gpu_device", lambda: "fake-gpu")
        monkeypatch.setattr(smodels, "get_model", lambda mid=None: (None, "cpu"))

        batched = []
        monkeypatch.setattr(smodels, "detect_batch",
                            lambda *a, **k: batched.append(1) or [])
        single = []

        def _fake_single(frame, threshold, min_distance, **k):
            single.append(1)
            z = np.zeros(frame.shape, np.float32)
            return z, z, np.zeros((0, 3), np.float32)

        monkeypatch.setattr(fvn, "_find_vectors_single_frame_neural", _fake_single)

        b4d = np.zeros((2, 2, 32, 32), np.float32)
        out = fvn._neural_block(b4d, 0.3, 4, True, None, None)
        assert out.shape[:2] == (2, 2)
        assert not batched, "detect_batch ran despite SPYDE_FV_GPU=off"
        assert len(single) == 4, "per-frame CPU fallback did not run"

    def test_chunk_passes_persistence(self, monkeypatch):
        """The map_overlap chunk fn forwards the persistence flag to the neural
        chunk (it was silently dropped before — refine.py was dead code)."""
        import spyde.actions.find_vectors_neural as fvn
        from spyde.actions.find_vectors import MAX_PEAKS
        from spyde.actions.find_vectors.chunk import _find_vectors_chunk

        seen = {}

        def _fake_chunk(ghost_block, depth_px, nav_dim, sigma, threshold,
                        min_dist, subpixel, beamstop_mask, model_id=None,
                        bg_sigma=12.0, persistence=False, spot_radius=None):
            seen.update(persistence=persistence, bg_sigma=bg_sigma,
                        spot_radius=spot_radius)
            return np.full((ghost_block.shape[0], ghost_block.shape[1],
                            MAX_PEAKS, 3), np.nan, np.float32)

        monkeypatch.setattr(fvn, "_find_vectors_chunk_neural", _fake_chunk)
        block = np.zeros((2, 2, 8, 8), np.float32)
        _find_vectors_chunk(block, 0, 2, 0.0, 5, 0.3, 4, True, None, None, None,
                            method="neural", bg_sigma=6.0, persistence=True,
                            spot_radius=7.0)
        assert seen == {"persistence": True, "bg_sigma": 6.0, "spot_radius": 7.0}

    def test_default_device_chain(self, monkeypatch):
        """CUDA → MPS → CPU fallback chain (Macs previously never got MPS)."""
        import torch
        from spyde.models import infer

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if getattr(torch.backends, "mps", None) is not None:
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
            assert infer._default_device().type == "mps"
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert infer._default_device().type == "cpu"

    def test_neural_run_cpu(self, monkeypatch):
        # Force the CPU branch — deterministic, and no torch-CUDA under pytest.
        import spyde.actions.find_vectors_torch as fvt
        monkeypatch.setattr(fvt, "torch_gpu_device", lambda: None)

        from spyde.backend.session import Session
        from spyde.actions.find_vectors_action import fv_run

        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_diffraction_4d())
            time.sleep(0.4)
            src = _signal_plot(session)
            assert src is not None
            before = len(session.signal_trees)

            # persistence=True also exercises the stage-2 refine
            # (models/refine.py): identical frames → full neighbour
            # persistence → real peaks survive the filter.
            fv_run(session, src, {
                "method": "neural", "sigma": 1.0, "threshold": 0.3,
                "min_distance": 4, "subpixel": True, "model_id": "",
                "persistence": True,
            })
            assert _wait(lambda: len(session.signal_trees) == before + 1), \
                "neural vectors window never opened"
            vtree = session.signal_trees[-1]
            assert _wait(lambda: getattr(vtree, "diffraction_vectors", None) is not None), \
                "diffraction_vectors never attached"
            # A valid (possibly empty) count map of the right nav shape.
            cm = vtree.diffraction_vectors.count_map()
            assert cm.shape == (4, 5)
        finally:
            session.shutdown()

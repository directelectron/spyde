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
        unset → the caller's default (neural passes "4" —
        NEURAL_GPU_LANE_DEFAULT; outside a dask worker any worker-gated mode
        allows)."""
        from spyde.actions.find_vectors.gpu_runtime import _gpu_task_allowed
        from spyde.actions.find_vectors_neural import NEURAL_GPU_LANE_DEFAULT

        assert NEURAL_GPU_LANE_DEFAULT == "4"

        monkeypatch.delenv("SPYDE_FV_GPU", raising=False)
        assert _gpu_task_allowed(default_mode="4") is True   # not on a worker
        assert _gpu_task_allowed(default_mode="all") is True
        assert _gpu_task_allowed() is True          # outside a worker: allowed

        monkeypatch.setenv("SPYDE_FV_GPU", "off")
        assert _gpu_task_allowed(default_mode="4") is False
        assert _gpu_task_allowed() is False

    def test_lane_split_default_mode(self, monkeypatch):
        """split_workers_for_gpu honours the per-method unset-default so the
        lane sizing matches which workers actually submit CUDA."""
        from spyde.compute_dispatch import split_workers_for_gpu

        monkeypatch.delenv("SPYDE_FV_GPU", raising=False)

        class _C:
            def scheduler_info(self, n_workers=None):
                return {"workers": {
                    f"tcp://{i}": {"name": i} for i in range(4)
                }}

        # Force the numba-availability gate open for the unit test.
        import spyde.compute_dispatch as cd
        import types
        fake_numba = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True))
        monkeypatch.setitem(__import__("sys").modules, "numba", fake_numba)
        monkeypatch.setitem(__import__("sys").modules, "numba.cuda",
                            fake_numba.cuda)

        gpu1, cpu1 = cd.split_workers_for_gpu(_C(), "one")
        gpu2, cpu2 = cd.split_workers_for_gpu(_C(), "2")
        assert len(gpu1) == 1 and len(cpu1) == 3
        assert len(gpu2) == 2 and len(cpu2) == 2

    def test_mps_neural_lane_pins_single_worker(self, monkeypatch):
        """On Mac, the neural run is pinned to ONE worker process (worker '1' as
        the sole GPU lane, empty CPU lane) so only a single Metal context is ever
        live — the concurrent-context abort surface. Worker '0' is CPU-only."""
        import spyde.actions.find_vectors.orchestrate as o

        class _C:
            def scheduler_info(self, n_workers=None):
                return {"workers": {
                    f"tcp://{i}": {"name": i} for i in range(4)
                }}

        gpu, cpu = o._mps_neural_lane(_C())
        assert gpu == ["tcp://1"], "neural MPS lane must pin to worker '1'"
        assert cpu == [], "neural MPS lane must leave the CPU lane empty (gpu_only)"

        # Worker '1' absent (degenerate <4-core cluster): return empty so the
        # caller falls back to the single-lane path (no concurrency there anyway).
        class _C0:
            def scheduler_info(self, n_workers=None):
                return {"workers": {"tcp://0": {"name": 0}}}

        assert o._mps_neural_lane(_C0()) == ([], [])

    def test_mps_neural_enabled_gate(self, monkeypatch):
        """_mps_neural_enabled: True only on darwin + a torch MPS device + not the
        =0 escape hatch; False off Mac or when MPS is unavailable."""
        import spyde.actions.find_vectors.orchestrate as o
        import spyde.actions.find_vectors_torch as fvt
        import torch

        monkeypatch.delenv("SPYDE_NEURAL_MPS_BATCH", raising=False)
        from spyde.models import infer as _infer
        monkeypatch.setattr(_infer.sys, "platform", "darwin")
        monkeypatch.setattr(o.sys, "platform", "darwin", raising=False)
        monkeypatch.setattr(fvt, "torch_gpu_device", lambda: torch.device("mps"))
        assert o._mps_neural_enabled() is True

        # Escape hatch disables it even with MPS present.
        monkeypatch.setenv("SPYDE_NEURAL_MPS_BATCH", "0")
        assert o._mps_neural_enabled() is False
        monkeypatch.delenv("SPYDE_NEURAL_MPS_BATCH", raising=False)

        # No MPS device → False (CUDA/CPU lane logic runs instead).
        monkeypatch.setattr(fvt, "torch_gpu_device", lambda: None)
        assert o._mps_neural_enabled() is False

        # Off Mac → always False regardless of device.
        monkeypatch.setattr(o.sys, "platform", "win32", raising=False)
        monkeypatch.setattr(fvt, "torch_gpu_device", lambda: torch.device("mps"))
        assert o._mps_neural_enabled() is False

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

    # ── Mac (Apple-MPS) crash-avoidance layer ─────────────────────────────────
    # These never touch a real MPS device (this is a Windows box with no Metal):
    # they monkeypatch sys.platform='darwin' + a fake MPS device and force CPU
    # everywhere a real device would be needed.

    def test_mps_batch_uses_mps_by_default_on_mac(self, monkeypatch):
        """On Mac with MPS available, the BATCH runs on MPS by DEFAULT
        (SPYDE_NEURAL_MPS_BATCH unset) — detect_batch is invoked on the MPS
        device with the MPS model. MPS is validated + single-worker-pinned
        (orchestrate._mps_neural_lane), so this is the shipped default now."""
        import torch
        import spyde.actions.find_vectors_neural as fvn
        import spyde.actions.find_vectors_torch as fvt
        from spyde import models as smodels

        monkeypatch.setattr(fvn.sys, "platform", "darwin", raising=False)
        monkeypatch.delenv("SPYDE_NEURAL_MPS_BATCH", raising=False)
        # mps_batch_allowed reads sys.platform inside spyde.models.infer.
        from spyde.models import infer as _infer
        monkeypatch.setattr(_infer.sys, "platform", "darwin")

        mps_dev = torch.device("mps")
        monkeypatch.setattr(smodels, "get_model",
                            lambda mid=None: ("MPS_MODEL", mps_dev))
        monkeypatch.setattr(fvt, "torch_gpu_device", lambda: mps_dev)

        seen = {}

        def _fake_detect_batch(model, frames, device, **k):
            seen["model"] = model
            seen["device"] = device
            return [np.zeros((0, 3), np.float32) for _ in range(len(frames))]

        monkeypatch.setattr(smodels, "detect_batch", _fake_detect_batch)

        b4d = np.zeros((2, 2, 16, 16), np.float32)
        fvn._neural_block(b4d, 0.3, 4, True, None, None)
        assert seen["device"].type == "mps", "Mac batch did not use MPS by default"
        assert seen["model"] == "MPS_MODEL", "Mac batch did not use the MPS model"

    def test_mps_batch_cpu_escape_hatch_on_mac(self, monkeypatch):
        """SPYDE_NEURAL_MPS_BATCH=0 on Mac forces the batch onto CPU — the escape
        hatch for hardware/torch builds where MPS still misbehaves. get_cpu_model
        supplies the CPU model and detect_batch runs on the CPU device."""
        import torch
        import spyde.actions.find_vectors_neural as fvn
        import spyde.actions.find_vectors_torch as fvt
        from spyde import models as smodels

        monkeypatch.setattr(fvn.sys, "platform", "darwin", raising=False)
        from spyde.models import infer as _infer
        monkeypatch.setattr(_infer.sys, "platform", "darwin")
        monkeypatch.setenv("SPYDE_NEURAL_MPS_BATCH", "0")

        mps_dev = torch.device("mps")
        cpu_dev = torch.device("cpu")
        monkeypatch.setattr(smodels, "get_model",
                            lambda mid=None: ("MPS_MODEL", mps_dev))
        monkeypatch.setattr(smodels, "get_cpu_model",
                            lambda mid=None: ("CPU_MODEL", cpu_dev))
        monkeypatch.setattr(fvt, "torch_gpu_device", lambda: mps_dev)

        seen = {}

        def _fake_detect_batch(model, frames, device, **k):
            seen["model"] = model
            seen["device"] = device
            return [np.zeros((0, 3), np.float32) for _ in range(len(frames))]

        monkeypatch.setattr(smodels, "detect_batch", _fake_detect_batch)

        b4d = np.zeros((2, 2, 16, 16), np.float32)
        fvn._neural_block(b4d, 0.3, 4, True, None, None)
        assert seen["device"].type == "cpu", "escape hatch did not force CPU batch"
        assert seen["model"] == "CPU_MODEL", "escape hatch did not use the CPU model"

    def test_mps_batch_allowed_platform_gate(self, monkeypatch):
        """mps_batch_allowed: True off-Mac unconditionally; on Mac True by DEFAULT
        (MPS validated + single-worker-pinned), False only via the =0 escape
        hatch."""
        from spyde.models import infer

        monkeypatch.setattr(infer.sys, "platform", "win32")
        monkeypatch.delenv("SPYDE_NEURAL_MPS_BATCH", raising=False)
        assert infer.mps_batch_allowed() is True         # off-Mac: no-op gate

        monkeypatch.setattr(infer.sys, "platform", "darwin")
        assert infer.mps_batch_allowed() is True          # Mac default: MPS batch
        monkeypatch.setenv("SPYDE_NEURAL_MPS_BATCH", "1")
        assert infer.mps_batch_allowed() is True           # explicit on
        monkeypatch.setenv("SPYDE_NEURAL_MPS_BATCH", "0")
        assert infer.mps_batch_allowed() is False          # escape hatch: CPU
        monkeypatch.setenv("SPYDE_NEURAL_MPS_BATCH", "off")
        assert infer.mps_batch_allowed() is False          # escape hatch: CPU

    def test_mps_fallback_env_set_on_darwin(self, monkeypatch):
        """enable_mps_cpu_fallback sets PYTORCH_ENABLE_MPS_FALLBACK=1 on Mac
        (fix 1) and is a no-op off Mac. __main__ and the worker plugin both
        apply the same env early."""
        from spyde.models import infer

        # Off Mac: never touches the env.
        monkeypatch.setattr(infer.sys, "platform", "win32")
        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
        infer.enable_mps_cpu_fallback()
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in infer.os.environ

        # On Mac: sets it to 1.
        monkeypatch.setattr(infer.sys, "platform", "darwin")
        infer.enable_mps_cpu_fallback()
        assert infer.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"

        # __main__ helper does the same (and also SPYDE_FV_GPU_CONC=1 — fix 3).
        import spyde.__main__ as m
        import sys as _sys
        monkeypatch.setattr(_sys, "platform", "darwin")
        monkeypatch.delenv("SPYDE_FV_GPU_CONC", raising=False)
        m._set_mac_neural_env()
        assert infer.os.environ["SPYDE_FV_GPU_CONC"] == "1"

    def test_worker_plugin_sets_mac_env(self, monkeypatch):
        """The dask worker plugin applies the Mac neural env in each worker
        process (fix 1 + fix 3) — PYTORCH_ENABLE_MPS_FALLBACK=1 and
        SPYDE_FV_GPU_CONC=1 on darwin; nothing off Mac."""
        import spyde.dask_manager as dm

        monkeypatch.setattr(dm.sys, "platform", "win32", raising=False)
        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
        monkeypatch.delenv("SPYDE_FV_GPU_CONC", raising=False)
        dm._apply_mac_neural_env()
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in dm.os.environ

        monkeypatch.setattr(dm.sys, "platform", "darwin", raising=False)
        dm._apply_mac_neural_env()
        assert dm.os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        assert dm.os.environ["SPYDE_FV_GPU_CONC"] == "1"

    def test_gpu_conc_one_on_mac_serializes(self, monkeypatch):
        """With SPYDE_FV_GPU_CONC=1 (set on Mac), _gpu_slots admits ONE thread —
        the neural forward is serialised process-wide (fix 3)."""
        from spyde.actions.find_vectors.gpu_runtime import _gpu_slots

        monkeypatch.setenv("SPYDE_FV_GPU_CONC", "1")
        sem = _gpu_slots()
        assert sem.acquire(blocking=False) is True
        # A second acquire must fail — only one forward at a time.
        assert sem.acquire(blocking=False) is False
        sem.release()

    def test_detect_batch_runtimeerror_cpu_retry(self, monkeypatch):
        """A catchable RuntimeError on the non-CPU device forward → the sub-batch
        retries on CPU, the model+device flip to CPU for the rest of the call,
        and correct peaks come back (fix 2). No real MPS needed — we use torch's
        'meta' device (has no hardware requirement, and device.type != 'cpu' so
        the retry branch fires) and a fake model that raises on the first
        (non-CPU) forward then succeeds on CPU."""
        import torch
        from spyde.models import infer

        H = W = 16
        calls = {"devices": []}

        class _FakeModel:
            levels = 2

            def __init__(self):
                self._device = torch.device("meta")

            def to(self, dev):
                self._device = torch.device(dev)
                return self

            def eval(self):
                return self

            def __call__(self, x):
                calls["devices"].append(self._device.type)
                if self._device.type != "cpu":
                    raise RuntimeError("device op not implemented (simulated)")
                # CPU: return a heatmap with one bright spot + zero offsets.
                B = x.shape[0]
                hm = torch.full((B, 1, H, W), -6.0)
                hm[:, 0, 8, 8] = 6.0                 # sigmoid → ~1 confidence
                off = torch.zeros((B, 2, H, W))
                return hm, off

        model = _FakeModel()
        # Avoid poking the real registry cache in the demotion hook.
        from spyde.models import registry as _reg
        monkeypatch.setattr(_reg, "demote_cached_models_to_cpu", lambda: None)

        frames = np.zeros((3, H, W), np.float32)
        frames[:, 8, 8] = 100.0
        # auto_scale False keeps the frame at native size (no scipy zoom / estimate).
        out = infer.detect_batch(model, frames, torch.device("meta"),
                                 thresh=0.3, min_distance=2, auto_scale=False)
        assert len(out) == 3
        # The forward was attempted on the non-CPU device (raised) then on cpu.
        assert calls["devices"][0] != "cpu"
        assert "cpu" in calls["devices"]
        assert model._device.type == "cpu", "model did not flip to CPU"
        # Correct peak decoded near (8,8) on every frame.
        for p in out:
            assert len(p) >= 1
            assert abs(p[0, 0] - 8) < 2 and abs(p[0, 1] - 8) < 2

    def test_neural_gpu_demote_flag(self, monkeypatch):
        """demote_neural_gpu pins this process to CPU (fix 4 per-process flag);
        SPYDE_FV_GPU=off also reads as demoted."""
        import spyde.actions.find_vectors_neural as fvn

        monkeypatch.delenv("SPYDE_FV_GPU", raising=False)
        fvn._NEURAL_GPU_DEMOTED[0] = False
        assert fvn._neural_gpu_demoted() is False
        fvn.demote_neural_gpu()
        assert fvn._neural_gpu_demoted() is True
        fvn._NEURAL_GPU_DEMOTED[0] = False           # reset module state
        monkeypatch.setenv("SPYDE_FV_GPU", "off")
        assert fvn._neural_gpu_demoted() is True

    def test_worker_death_detected(self):
        """orchestrate._is_worker_death recognises a dask worker-process death
        so fix 4's CPU-recovery retry fires only on that class of failure."""
        from spyde.actions.find_vectors.orchestrate import _is_worker_death

        class KilledWorker(Exception):
            pass

        assert _is_worker_death(KilledWorker("boom")) is True
        assert _is_worker_death(RuntimeError(
            "dispatcher stalled: worker restarted or task unschedulable")) is True
        assert _is_worker_death(ValueError("bad shape")) is False

    def test_dask_spawn_method_pinned(self):
        """Fix 5: DaskManager pins the worker multiprocessing method to 'spawn'
        so a perf change can't reintroduce the fork+Metal crash."""
        from spyde.models import infer  # noqa: F401 (ensures package imports)
        import inspect
        import spyde.dask_manager as dm

        src = inspect.getsource(dm.DaskManager._run)
        assert "multiprocessing-method" in src and '"spawn"' in src

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

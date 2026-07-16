"""In-app compute limits (backend/compute_config.py): clamping, env + settings
persistence, worker-plan recompute, cluster restart, and startup env loading.
"""
from __future__ import annotations

import json
import os

import pytest


class _FakeDaskManager:
    def __init__(self):
        self.restarts = []
        self.client = object()

    def restart(self, n_workers=None, threads_per_worker=None):
        self.restarts.append((n_workers, threads_per_worker))


class _FakeEvent:
    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True


class _FakeSession:
    """Bare stub — run_on_worker executes inline when _dispatch_to_main is absent."""

    def __init__(self):
        self._settings = {}
        self.saved = 0
        self.dask_manager = _FakeDaskManager()
        self._dask_ready = _FakeEvent()

    def _save_settings(self):
        self.saved += 1


class TestComputeConfigure:
    def test_apply_sets_env_persists_and_restarts(self, monkeypatch):
        import spyde.backend.compute_config as cc
        for env in ("SPYDE_MEM_FRACTION", "SPYDE_COMPUTE_FRACTION", "SPYDE_FV_GPU"):
            monkeypatch.delenv(env, raising=False)
        statuses = []
        import spyde.backend.ipc as ipc
        monkeypatch.setattr(ipc, "emit_status", lambda t: statuses.append(t))

        s = _FakeSession()
        cc.compute_configure(s, None, {
            "mem_fraction": 0.5, "compute_fraction": 0.5, "gpu_workers": "8",
        })

        assert os.environ["SPYDE_MEM_FRACTION"] == "0.5"
        assert os.environ["SPYDE_COMPUTE_FRACTION"] == "0.5"
        assert os.environ["SPYDE_FV_GPU"] == "8"
        assert s._settings["compute"] == {
            "mem_fraction": 0.5, "compute_fraction": 0.5, "gpu_workers": "8"}
        assert s.saved == 1
        assert s._dask_ready.cleared           # loads wait during the restart
        # Restarted with the plan recomputed from the NEW compute fraction.
        from spyde.backend.app import _compute_worker_plan
        assert s.dask_manager.restarts == [
            _compute_worker_plan(os.cpu_count() or 4)]
        assert statuses and "Restarting compute cluster" in statuses[0]

    def test_values_clamped_and_bad_gpu_defaulted(self, monkeypatch):
        import spyde.backend.compute_config as cc
        for env in ("SPYDE_MEM_FRACTION", "SPYDE_COMPUTE_FRACTION", "SPYDE_FV_GPU"):
            monkeypatch.delenv(env, raising=False)
        import spyde.backend.ipc as ipc
        monkeypatch.setattr(ipc, "emit_status", lambda t: None)

        s = _FakeSession()
        cc.compute_configure(s, None, {
            "mem_fraction": 5.0, "compute_fraction": 0.0, "gpu_workers": "banana",
        })
        assert os.environ["SPYDE_MEM_FRACTION"] == "0.8"      # hi clamp
        assert os.environ["SPYDE_COMPUTE_FRACTION"] == "0.1"  # lo clamp
        assert os.environ["SPYDE_FV_GPU"] == cc.GPU_DEFAULT   # nonsense → default

    def test_empty_payload_is_a_noop(self, monkeypatch):
        import spyde.backend.compute_config as cc
        s = _FakeSession()
        cc.compute_configure(s, None, {})
        assert s.saved == 0 and s.dask_manager.restarts == []


class TestPersistedEnv:
    def test_startup_loads_settings_env_wins(self, monkeypatch, tmp_path):
        import spyde.backend.compute_config as cc
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps({"compute": {
            "mem_fraction": 0.4, "compute_fraction": 0.5, "gpu_workers": "8"}}))
        monkeypatch.setattr(cc, "_settings_path", lambda: str(settings))
        monkeypatch.delenv("SPYDE_MEM_FRACTION", raising=False)
        monkeypatch.delenv("SPYDE_FV_GPU", raising=False)
        monkeypatch.setenv("SPYDE_COMPUTE_FRACTION", "0.9")   # explicit env wins

        cc.apply_persisted_compute_env()
        assert os.environ["SPYDE_MEM_FRACTION"] == "0.4"
        assert os.environ["SPYDE_COMPUTE_FRACTION"] == "0.9"
        assert os.environ["SPYDE_FV_GPU"] == "8"

    def test_current_config_reads_env(self, monkeypatch):
        from spyde.backend.compute_config import current_config
        monkeypatch.setenv("SPYDE_MEM_FRACTION", "0.4")
        monkeypatch.setenv("SPYDE_COMPUTE_FRACTION", "0.5")
        monkeypatch.setenv("SPYDE_FV_GPU", "8")
        cfg = current_config()
        assert cfg == {"mem_fraction": 0.4, "compute_fraction": 0.5,
                       "gpu_workers": "8"}

"""Dask monitor telemetry (backend/dask_stats.py) + the worker priority throttle.

No cluster: build_stats is pure, the sampler runs against a fake client, the
GPU probe is exercised with a stubbed subprocess, and the priority drop gets a
recording fake process.
"""
from __future__ import annotations

import time

import pytest


_FAKE_INFO = {
    "workers": {
        "tcp://1": {"name": 1, "memory_limit": 8_000_000_000,
                    "metrics": {"cpu": 97.4, "memory": 2_000_000_000,
                                "executing": 3, "ready": 5}},
        "tcp://0": {"name": 0, "memory_limit": 8_000_000_000,
                    "metrics": {"cpu": 12.0, "memory": 1_000_000_000,
                                "executing": 1, "ready": 0}},
    },
}


class TestBuildStats:
    def test_shape_and_sums(self):
        from spyde.backend.dask_stats import build_stats
        msg = build_stats(_FAKE_INFO, gpu={"util": 96.0, "vram_used": 3000,
                                           "vram_total": 8192}, host_cpu=88.26)
        assert msg["type"] == "dask_stats"
        assert [w["name"] for w in msg["workers"]] == ["0", "1"]   # sorted
        assert msg["workers"][1]["cpu"] == 97.4
        assert msg["tasks"] == {"executing": 4, "queued": 5}
        assert msg["gpu"]["util"] == 96.0
        assert msg["host_cpu"] == 88.3

    def test_tolerates_missing_fields(self):
        from spyde.backend.dask_stats import build_stats
        msg = build_stats({}, gpu=None, host_cpu=None)
        assert msg["workers"] == [] and msg["tasks"] == {"executing": 0, "queued": 0}
        assert "gpu" not in msg and "host_cpu" not in msg


class TestGpuProbe:
    def test_disables_after_first_failure(self, monkeypatch):
        import spyde.backend.dask_stats as ds
        calls = []

        def _boom(*a, **k):
            calls.append(1)
            raise FileNotFoundError("no nvidia-smi")

        monkeypatch.setattr(ds.subprocess, "run", _boom)
        probe = ds.GpuProbe()
        assert probe.sample() is None
        assert probe.sample() is None            # second call: no subprocess
        assert len(calls) == 1

    def test_parses_nvidia_smi(self, monkeypatch):
        import spyde.backend.dask_stats as ds

        class _R:
            returncode = 0
            stdout = b"96, 3000, 8192\n"
            stderr = b""

        monkeypatch.setattr(ds.subprocess, "run", lambda *a, **k: _R())
        assert ds.GpuProbe().sample() == {"util": 96.0, "vram_used": 3000,
                                          "vram_total": 8192}


class TestSampler:
    def test_emits_from_fake_client(self, monkeypatch):
        import spyde.backend.dask_stats as ds
        msgs = []
        monkeypatch.setattr(ds, "emit", lambda m: msgs.append(m))

        class _FakeClient:
            def scheduler_info(self, n_workers=None):
                assert n_workers == -1           # the 5-worker-truncation trap
                return _FAKE_INFO

        sampler = ds.DaskStatsSampler(lambda: _FakeClient(), interval=0.05)
        sampler._gpu._dead = True                # no subprocess in the test
        sampler.start()
        deadline = time.time() + 5.0
        while not msgs and time.time() < deadline:
            time.sleep(0.02)
        sampler.stop()
        assert msgs and msgs[0]["type"] == "dask_stats"
        assert msgs[0]["tasks"]["executing"] == 4


class TestWorkerPriority:
    def test_drops_to_below_normal(self, monkeypatch):
        import sys
        from spyde.dask_manager import _lower_worker_priority
        import psutil

        monkeypatch.delenv("SPYDE_WORKER_PRIORITY", raising=False)
        seen = []

        class _FakeProc:
            def nice(self, value=None):
                if value is None:
                    return 0
                seen.append(value)

        assert _lower_worker_priority(_FakeProc()) is True
        if sys.platform == "win32":
            assert seen == [psutil.BELOW_NORMAL_PRIORITY_CLASS]
        else:
            assert seen == [10]

    def test_opt_out(self, monkeypatch):
        from spyde.dask_manager import _lower_worker_priority
        monkeypatch.setenv("SPYDE_WORKER_PRIORITY", "normal")

        class _FakeProc:
            def nice(self, value=None):        # pragma: no cover — must not run
                raise AssertionError("priority touched despite opt-out")

        assert _lower_worker_priority(_FakeProc()) is False

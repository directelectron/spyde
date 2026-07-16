"""Dask monitor telemetry (backend/dask_stats.py) + the worker priority throttle.

No cluster: build_stats is pure, the sampler runs against a fake client, the
GPU probe is exercised with a stubbed subprocess, and the priority drop gets a
recording fake process.
"""
from __future__ import annotations

import time

import numpy as np
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
                                           "vram_total": 8192},
                          host_cpu=88.26, host_mem=52.04)
        assert msg["type"] == "dask_stats"
        assert [w["name"] for w in msg["workers"]] == ["0", "1"]   # sorted
        assert msg["workers"][1]["cpu"] == 97.4
        assert msg["tasks"] == {"executing": 4, "queued": 5}
        assert msg["gpu"]["util"] == 96.0
        assert msg["host_cpu"] == 88.3
        assert msg["host_mem"] == 52.0

    def test_tolerates_missing_fields(self):
        from spyde.backend.dask_stats import build_stats
        msg = build_stats({}, gpu=None, host_cpu=None)
        assert msg["workers"] == [] and msg["tasks"] == {"executing": 0, "queued": 0}
        assert "gpu" not in msg and "host_cpu" not in msg


class TestGpuProbe:
    def test_missing_nvidia_smi_disables_permanently(self, monkeypatch):
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

    def test_transient_failure_keeps_last_reading(self, monkeypatch):
        """The "GPU % goes away" bug: one slow nvidia-smi under load must NOT
        kill the probe — the sample is skipped and the last reading holds."""
        import spyde.backend.dask_stats as ds

        class _R:
            returncode = 0
            stdout = b"96, 3000, 8192\n"
            stderr = b""

        seq = [_R(), ds.subprocess.TimeoutExpired("nvidia-smi", 3.0), _R()]

        def _run(*a, **k):
            item = seq.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        monkeypatch.setattr(ds.subprocess, "run", _run)
        probe = ds.GpuProbe()
        first = probe.sample()
        assert first is not None and first["util"] == 96.0
        assert probe.sample() == first            # timeout → last reading held
        assert probe.sample() is not None         # and the probe recovers
        assert not probe._dead

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


class TestNavBlurNoCopy:
    def test_sigma_zero_returns_raw_block_no_copy(self):
        """Nav blur OFF must NOT convert/copy the chunk: the old path made two
        full float32 copies of every uint16 chunk (the 34 GiB batch churn)."""
        from spyde.actions.find_vectors.chunk import _nav_blur_trim
        block = (np.random.rand(4, 4, 8, 8) * 100).astype(np.uint16)
        out = _nav_blur_trim(block, 0, 2, 0.0)
        assert out is block                      # same object — zero copies
        assert out.dtype == np.uint16

    def test_sigma_positive_still_blurs(self):
        from scipy.ndimage import gaussian_filter
        from spyde.actions.find_vectors.chunk import _nav_blur_trim
        block = (np.random.rand(6, 6, 8, 8) * 100).astype(np.uint16)
        out = _nav_blur_trim(block, 1, 2, 0.5)
        ref = gaussian_filter(block.astype(np.float32),
                              sigma=(0.5, 0.5, 0, 0))[1:-1, 1:-1]
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, ref, rtol=1e-6)


class TestMemoryTrim:
    def test_trim_runs_and_reports_rss(self):
        from spyde.backend.dask_stats import _trim_process_memory
        rss = _trim_process_memory()
        assert isinstance(rss, int) and rss > 0

    def test_dask_trim_action(self, monkeypatch):
        import spyde.backend.dask_stats as ds
        ran = []

        class _FakeClient:
            def run(self, fn):
                ran.append(fn)
                return {}

        class _FakeSession:
            class dask_manager:
                client = _FakeClient()

        statuses = []
        import spyde.backend.ipc as ipc
        monkeypatch.setattr(ipc, "emit_status", lambda t: statuses.append(t))
        # session lacks _dispatch_to_main → run_on_worker executes inline.
        ds.dask_trim(_FakeSession(), None, {})
        assert ran and ran[0] is ds._trim_process_memory
        assert statuses and "Memory trimmed" in statuses[0]


class TestMemoryBackpressure:
    def test_lane_cap_shrinks_when_hot(self):
        from spyde.compute_dispatch import _lane_cap
        assert _lane_cap(34, 32, mem_hot=False) == 34
        assert _lane_cap(34, 32, mem_hot=True) == 16     # ~half the threads
        assert _lane_cap(10, 4, mem_hot=True) == 2       # floor keeps progress
        assert _lane_cap(10, 2, mem_hot=True) == 2

    def test_cluster_mem_frac(self):
        from spyde.compute_dispatch import _cluster_mem_frac

        class _C:
            def scheduler_info(self, n_workers=None):
                return {"workers": {
                    "a": {"memory_limit": 100, "metrics": {"memory": 30}},
                    "b": {"memory_limit": 100, "metrics": {"memory": 80}},
                    "c": {"memory_limit": 0, "metrics": {"memory": 999}},  # ignored
                }}

        assert _cluster_mem_frac(_C()) == 0.8

    def test_worker_memory_limit(self, monkeypatch):
        from spyde.dask_manager import _worker_memory_limit
        monkeypatch.delenv("SPYDE_MEM_FRACTION", raising=False)
        total = 64 * 1024 ** 3
        # Default: HALF the machine split across workers (was 80%).
        assert _worker_memory_limit(total, 8) == int(total * 0.5) // 8
        assert _worker_memory_limit(total, 8, fraction=0.8) == int(total * 0.8) // 8
        assert _worker_memory_limit(total, 8, fraction=0.05) == int(total * 0.2) // 8  # clamp
        monkeypatch.setenv("SPYDE_MEM_FRACTION", "0.25")
        assert _worker_memory_limit(total, 8) == int(total * 0.25) // 8


class TestWorkerPlan:
    def test_budget_is_75_percent_of_cores(self, monkeypatch):
        monkeypatch.delenv("SPYDE_COMPUTE_FRACTION", raising=False)
        from spyde.backend.app import _compute_worker_plan
        # 16 logical cores → 12-thread budget → 6 workers × 2 threads (75%).
        assert _compute_worker_plan(16) == (6, 2)
        # 32 cores → 24-thread budget → 6 workers × 4 threads (75%).
        assert _compute_worker_plan(32) == (6, 4)
        # 8 cores → 6-thread budget → 3 × 2.
        assert _compute_worker_plan(8) == (3, 2)
        # Tiny machines keep the 1×1 floor.
        assert _compute_worker_plan(2) == (1, 1)

    def test_fraction_override_and_clamp(self, monkeypatch):
        from spyde.backend.app import _compute_worker_plan
        assert _compute_worker_plan(16, fraction=1.0) == (8, 2)     # full machine
        assert _compute_worker_plan(16, fraction=0.0) == (1, 2)     # clamped to 0.1
        monkeypatch.setenv("SPYDE_COMPUTE_FRACTION", "0.5")
        assert _compute_worker_plan(16) == (4, 2)
        monkeypatch.setenv("SPYDE_COMPUTE_FRACTION", "junk")
        assert _compute_worker_plan(16) == (6, 2)                   # bad env → 0.75


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

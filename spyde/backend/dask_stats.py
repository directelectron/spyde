"""dask_stats.py — live compute telemetry for the in-app Dask monitor.

A heavy find-vectors/OM batch pins every worker (and the GPU) at ~100%; the
only visibility used to be the external Dask dashboard. This module samples
the cluster every couple of seconds and streams a compact ``dask_stats``
message that the renderer's StatusBar HUD (``DaskMonitor.tsx``) renders as a
live CPU/GPU/task readout with a per-worker breakdown — so "what is slow /
going wrong" is answerable at a glance, without leaving the app.

- Worker CPU/memory/queue depths come from ONE ``client.scheduler_info()``
  round-trip (``n_workers=-1`` — the default silently truncates to 5 workers,
  see the dask-pitfalls note).
- GPU utilisation/VRAM comes from ``nvidia-smi`` (pynvml isn't shipped); a
  probe failure disables further GPU queries so a GPU-less machine pays the
  subprocess cost exactly once.
- The sampler is a daemon thread owned by the Session: started on
  ``_on_dask_ready``, stopped in ``shutdown()``. Emitting only ~every 2 s and
  only compact numbers keeps the stdout protocol traffic negligible.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time

from spyde.backend.ipc import emit

log = logging.getLogger(__name__)

SAMPLE_INTERVAL_S = 2.0


def build_stats(info: dict, gpu: dict | None, host_cpu: float | None,
                host_mem: float | None = None) -> dict:
    """Shape one ``dask_stats`` message from ``client.scheduler_info()`` output
    (pure — unit-testable without a cluster)."""
    workers = []
    executing = queued = 0
    for addr, w in (info.get("workers") or {}).items():
        m = w.get("metrics") or {}
        n_exec = int(m.get("executing", 0))
        n_ready = int(m.get("ready", 0))
        executing += n_exec
        queued += n_ready
        workers.append({
            "name": str(w.get("name", addr)),
            "cpu": round(float(m.get("cpu", 0.0)), 1),
            "mem": int(m.get("memory", 0)),
            "mem_limit": int(w.get("memory_limit", 0) or 0),
            "executing": n_exec,
            "ready": n_ready,
        })
    workers.sort(key=lambda w: w["name"])
    msg = {
        "type": "dask_stats",
        "workers": workers,
        "tasks": {"executing": executing, "queued": queued},
    }
    if gpu is not None:
        msg["gpu"] = gpu
    if host_cpu is not None:
        msg["host_cpu"] = round(float(host_cpu), 1)
    if host_mem is not None:
        msg["host_mem"] = round(float(host_mem), 1)
    return msg


def _trim_process_memory() -> int:
    """Reclaim what THIS process can hand back: collect garbage, release
    torch's cached CUDA blocks, and on Windows push the working set out to
    standby (``EmptyWorkingSet`` — RSS drops immediately; touched pages soft-
    fault back in). Runs in workers via ``client.run`` and in the backend
    directly. Returns the post-trim RSS in bytes (-1 if unreadable)."""
    import ctypes
    import gc
    import sys

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            ctypes.windll.psapi.EmptyWorkingSet(
                ctypes.windll.kernel32.GetCurrentProcess())
        except Exception:
            pass
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return -1


def trim_cluster_memory(session) -> None:
    """AUTOMATIC post-batch trim (no UI — user feedback: a manual "Trim
    memory" button second-guessing dask is odd). Why it runs at all: dask's
    spill/pause thresholds act on process RSS, so allocator retention left by
    a batch's churn silently eats the NEXT compute's headroom (the
    fv → VI → fv chronic-spill chain). Resetting RSS right after a batch
    keeps dask's accounting honest. Best-effort; touched pages soft-fault
    back in."""
    try:
        client = getattr(getattr(session, "dask_manager", None), "client", None)
        if client is not None:
            # wait=False: fire-and-forget. A blocking run() across all workers
            # right at app close wedged teardown (the batch thread sat in the
            # RPC while shutdown killed the stdin tick).
            client.run(_trim_process_memory, wait=False)
        _trim_process_memory()
    except Exception as e:
        log.debug("[dask-stats] post-batch trim failed: %s", e)


class GpuProbe:
    """``nvidia-smi`` utilisation/VRAM sampler.

    Only a MISSING nvidia-smi (no GPU / no driver) disables the probe
    permanently — so a GPU-less machine pays the subprocess cost exactly once.
    Transient failures (the user-visible "GPU % goes away" bug: a saturated
    machine can make one nvidia-smi call exceed its timeout) just skip that
    sample and keep the LAST reading, so the HUD holds steady under exactly
    the load it exists to show."""

    def __init__(self):
        self._dead = False
        self._last: dict | None = None

    def sample(self) -> dict | None:
        if self._dead:
            return None
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, timeout=3.0,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode(errors="replace")[:200])
            line = result.stdout.decode().strip().splitlines()[0]
            util, used, total = [float(x) for x in line.split(",")]
            self._last = {"util": util, "vram_used": int(used),
                          "vram_total": int(total)}
            return self._last
        except FileNotFoundError:
            log.debug("[dask-stats] no nvidia-smi — GPU probe disabled")
            self._dead = True
            self._last = None
            return None
        except Exception as e:
            log.debug("[dask-stats] GPU sample skipped (%s)", e)
            return self._last


class DaskStatsSampler:
    """Daemon thread: sample the cluster + GPU every ``interval`` s and emit
    ``dask_stats``. ``client_getter`` re-resolves the client each tick so a
    late-starting or restarted cluster is picked up automatically."""

    def __init__(self, client_getter, interval: float = SAMPLE_INTERVAL_S):
        self._client_getter = client_getter
        self._interval = float(interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu = GpuProbe()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="dask-stats")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # psutil is a hard dep of distributed, so it's always importable here.
        import psutil
        psutil.cpu_percent(interval=None)      # prime the host-CPU counter
        while not self._stop.wait(self._interval):
            try:
                client = self._client_getter()
                if client is None:
                    continue
                info = client.scheduler_info(n_workers=-1)
                emit(build_stats(info, self._gpu.sample(),
                                 psutil.cpu_percent(interval=None),
                                 psutil.virtual_memory().percent))
            except Exception as e:
                # Cluster shutting down / transient RPC error — keep sampling;
                # the stop() in Session.shutdown ends the thread.
                log.debug("[dask-stats] sample failed: %s", e)

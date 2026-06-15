from __future__ import annotations
import logging
import subprocess

from PySide6 import QtCore
from PySide6.QtCore import QObject, Signal, Slot
import psutil

from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


def _probe_gpus() -> int:
    """Return number of NVIDIA GPUs detected via nvidia-smi. Returns 0 on any failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            timeout=3,
        )
        if result.returncode != 0:
            return 0
        lines = [l for l in result.stdout.decode().strip().splitlines() if l.strip()]
        return len(lines)
    except Exception:
        return 0


class _DaskClusterWorker(QObject):
    finished = Signal(object, object, object)  # cluster, client, gpu_worker_address
    error = Signal(Exception)

    def __init__(self, n_workers: int, threads_per_worker: int, parent=None):
        super().__init__(parent)
        self.n_workers = n_workers
        self.threads_per_worker = threads_per_worker
        self._stopped = False

    @Slot()
    def start(self):
        if self._stopped:
            return
        # Pre-calculate per-worker memory based on the *final* worker count.
        # Without this, LocalCluster(n_workers=1) assigns 100% of RAM to the
        # first worker; scaled-up workers inherit that same spec, so each one
        # claims the full system memory instead of 1/n_workers of it.
        # Reserve ~80 % of RAM (matching Dask's own default headroom) and
        # divide by the target worker count.
        total_mem = psutil.virtual_memory().total
        memory_per_worker = int(total_mem * 0.80) // max(self.n_workers, 1)
        logger.info(
            "Dask memory per worker: %.1f GB (total=%.1f GB, workers=%d)",
            memory_per_worker / 1024**3,
            total_mem / 1024**3,
            self.n_workers,
        )

        # Start with 1 worker so the client is usable immediately (~300ms),
        # then scale to full count in the background while the app loads.
        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=self.threads_per_worker,
            memory_limit=memory_per_worker,
        )
        client = Client(cluster)
        n_gpus = _probe_gpus()
        gpu_worker_address = "gpu_available" if n_gpus > 0 else None
        self.finished.emit(cluster, client, gpu_worker_address)
        # Scale up after the client signal is delivered
        if self.n_workers > 1:
            cluster.scale(self.n_workers)

    @Slot()
    def stop(self):
        self._stopped = True


class DaskManager(QObject):
    """Owns the Dask LocalCluster and Client lifecycle."""

    ready = Signal()  # emitted once the client is available

    def __init__(self, n_workers: int, threads_per_worker: int, parent=None):
        super().__init__(parent)
        self._client: Client | None = None
        self._cluster: LocalCluster | None = None
        self._gpu_worker_address: str | None = None
        self._heavy_compute_workers: list[str] | None = None
        self._n_workers = n_workers
        self._threads_per_worker = threads_per_worker
        self._dask_thread: QtCore.QThread | None = None
        self._dask_worker: _DaskClusterWorker | None = None

    @property
    def client(self) -> Client | None:
        return self._client

    @property
    def heavy_workers(self) -> list[str] | None:
        return self._heavy_compute_workers

    @property
    def gpu_worker_address(self) -> str | None:
        return self._gpu_worker_address

    def start(self) -> None:
        """Start the Dask cluster in a background thread."""
        self._dask_thread = QtCore.QThread(self)
        self._dask_worker = _DaskClusterWorker(
            n_workers=self._n_workers,
            threads_per_worker=self._threads_per_worker,
        )
        self._dask_worker.moveToThread(self._dask_thread)
        self._dask_thread.started.connect(self._dask_worker.start)
        self._dask_worker.finished.connect(self._on_dask_ready)
        self._dask_worker.error.connect(self._on_dask_error)
        self._dask_thread.finished.connect(self._dask_worker.deleteLater)
        self._dask_thread.start()

    @Slot(object, object, object)
    def _on_dask_ready(self, cluster, client, gpu_worker_address=None):
        self._cluster = cluster
        self._client = client
        self._gpu_worker_address = gpu_worker_address
        print(f"Dask cluster ready. Dashboard: {client.dashboard_link}")
        worker_keys = list(client.scheduler_info(n_workers=-1)["workers"].keys())
        heavy = worker_keys[1:]
        self._heavy_compute_workers = heavy if heavy else None
        self._dask_thread.quit()
        self._dask_thread.wait(2000)
        self.ready.emit()

    @Slot(Exception)
    def _on_dask_error(self, exc):
        print(f"Failed to start Dask cluster: {exc}")
        self._dask_thread.quit()
        self._dask_thread.wait(2000)

    def shutdown(self) -> None:
        """Gracefully shut down the Dask client and cluster."""
        import logging as _logging
        import time
        import multiprocessing as mp
        import gc

        print("Shutting down Dask cluster and client...")
        for name in ("distributed", "distributed.comm", "distributed.comm.tcp"):
            lg = _logging.getLogger(name)
            lg.setLevel(_logging.CRITICAL)
            lg.propagate = False
            try:
                lg.handlers.clear()
            except Exception:
                lg.handlers = []
            lg.addHandler(_logging.NullHandler())

        client = self._client
        if client is not None:
            try:
                try:
                    client.close(timeout="2s")
                except TypeError:
                    try:
                        client.close(timeout=2)
                    except Exception:
                        client.close()
                except Exception:
                    try:
                        client.close()
                    except Exception:
                        pass
            finally:
                self._client = None

        cluster = self._cluster
        if cluster is not None:
            try:
                try:
                    cluster.scale(0)
                except Exception:
                    pass
                try:
                    cluster.close(timeout="2s")
                except TypeError:
                    cluster.close(timeout=2)
                except Exception:
                    cluster.close()
            except Exception:
                pass
            finally:
                self._cluster = None

        time.sleep(0.5)
        try:
            for child in mp.active_children():
                try:
                    child.terminate()
                    child.join(timeout=0.5)
                except Exception:
                    try:
                        child.kill()
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            gc.collect()
        except Exception:
            pass

from __future__ import annotations
import logging
import subprocess
import threading

from psygnal import Signal
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


class DaskManager:
    """Owns the Dask LocalCluster and Client lifecycle."""

    ready = Signal()
    error = Signal(str)

    def __init__(self, n_workers: int, threads_per_worker: int):
        self._client: Client | None = None
        self._cluster: LocalCluster | None = None
        self._gpu_worker_address: str | None = None
        self._heavy_compute_workers: list[str] | None = None
        self._n_workers = n_workers
        self._threads_per_worker = threads_per_worker
        self._thread: threading.Thread | None = None

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
        self._thread = threading.Thread(target=self._run, daemon=True, name="dask-startup")
        self._thread.start()

    def _run(self) -> None:
        try:
            total_mem = psutil.virtual_memory().total
            memory_per_worker = int(total_mem * 0.80) // max(self._n_workers, 1)
            logger.info(
                "Dask memory per worker: %.1f GB (total=%.1f GB, workers=%d)",
                memory_per_worker / 1024**3,
                total_mem / 1024**3,
                self._n_workers,
            )

            cluster = LocalCluster(
                n_workers=1,
                threads_per_worker=self._threads_per_worker,
                memory_limit=memory_per_worker,
            )
            client = Client(cluster)
            n_gpus = _probe_gpus()
            gpu_worker_address = "gpu_available" if n_gpus > 0 else None

            self._cluster = cluster
            self._client = client
            self._gpu_worker_address = gpu_worker_address
            logger.info("Dask cluster ready. Dashboard: %s", client.dashboard_link)

            worker_keys = list(client.scheduler_info(n_workers=-1)["workers"].keys())
            heavy = worker_keys[1:]
            self._heavy_compute_workers = heavy if heavy else None

            if self._n_workers > 1:
                cluster.scale(self._n_workers)

            self.ready.emit()
        except Exception as exc:
            logger.exception("Failed to start Dask cluster")
            self.error.emit(str(exc))

    def shutdown(self) -> None:
        """Gracefully shut down the Dask client and cluster."""
        import logging as _logging
        import time
        import multiprocessing as mp
        import gc

        logger.info("Shutting down Dask cluster and client...")
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
                    except Exception as e:
                        logger.debug("Dask client close failed during shutdown: %s", e)
            finally:
                self._client = None

        cluster = self._cluster
        if cluster is not None:
            try:
                try:
                    cluster.scale(0)
                except Exception as e:
                    logger.debug("scaling Dask cluster to 0 failed: %s", e)
                try:
                    cluster.close(timeout="2s")
                except TypeError:
                    cluster.close(timeout=2)
                except Exception:
                    cluster.close()
            except Exception as e:
                logger.debug("Dask cluster close failed during shutdown: %s", e)
            finally:
                self._cluster = None

        time.sleep(0.5)
        # Reap multiprocessing children we own directly.
        try:
            for child in mp.active_children():
                try:
                    child.terminate()
                    child.join(timeout=0.5)
                except Exception as e:
                    logger.debug("terminating worker %r failed, killing: %s", child, e)
                    try:
                        child.kill()
                    except Exception as e2:
                        logger.debug("killing worker %r failed: %s", child, e2)
        except Exception as e:
            logger.debug("reaping Dask worker children failed: %s", e)
        # Dask spawns each worker under a NANNY process, so the real worker is a
        # GRANDCHILD that mp.active_children() never lists. Walk the full process
        # subtree via psutil and reap anything still alive, so a graceful quit
        # doesn't leak workers even without the OS job-object guard.
        try:
            me = psutil.Process()
            kids = me.children(recursive=True)
            for c in kids:
                try:
                    c.terminate()
                except Exception as e:
                    logger.debug("terminating subprocess %s failed: %s", c.pid, e)
            gone, alive = psutil.wait_procs(kids, timeout=1.0)
            for c in alive:
                try:
                    c.kill()
                except Exception as e:
                    logger.debug("killing surviving subprocess %s failed: %s", c.pid, e)
        except Exception as e:
            logger.debug("reaping Dask process subtree failed: %s", e)
        try:
            gc.collect()
        except Exception as e:
            logger.debug("gc.collect() during Dask shutdown failed: %s", e)

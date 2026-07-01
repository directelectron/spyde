from __future__ import annotations
import logging
import subprocess
import threading

from psygnal import Signal
import psutil

from dask.distributed import Client, LocalCluster

logger = logging.getLogger(__name__)


def _neutralize_slow_net_io_counters(probe_timeout: float = 2.0) -> None:
    """Work around a Windows hang where ``psutil.net_io_counters()`` blocks for
    ~70s, freezing Dask cluster startup.

    Dask's ``distributed.SystemMonitor`` (created inside ``Scheduler.__init__``)
    calls ``psutil.net_io_counters()`` UNCONDITIONALLY at construction and on
    every monitor tick — and it has NO dask-config switch to disable it (unlike
    disk/cpu/gil). On this dev machine (AzureAD-joined, many virtual NICs) that
    syscall enumerates network adapters and blocks ~72s in the Electron-spawned
    backend process — the scheduler build hung there until something else kicked
    the network stack (observed: opening the file dialog). Confirmed by a
    faulthandler stack dump: SystemMonitor.__init__ → psutil.net_io_counters.

    We probe the call once on a worker thread with a short timeout. If it's slow,
    we monkeypatch ``psutil.net_io_counters`` to return the last good sample (or
    zeros) instantly. The net-io numbers are dashboard-only telemetry, so a
    static/zero value is harmless — we trade a graph nobody watches at startup
    for a cluster that starts in seconds. If the probe is fast, we leave psutil
    untouched. Idempotent.
    """
    if getattr(psutil, "_spyde_net_io_patched", False):
        return

    result: dict = {}

    def _probe():
        try:
            result["val"] = psutil.net_io_counters()
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=_probe, name="net-io-probe", daemon=True)
    t.start()
    t.join(timeout=probe_timeout)

    if not t.is_alive() and "val" in result:
        # Fast and healthy — leave the real implementation in place.
        return

    # Slow (still running) or errored: install a non-blocking stub. Capture a
    # zero-filled sample of the right namedtuple type so consumers keep working.
    last_good = result.get("val")
    if last_good is None:
        try:
            zero_type = type(psutil.net_io_counters.__wrapped__()) \
                if hasattr(psutil.net_io_counters, "__wrapped__") else None
        except Exception:
            zero_type = None
        # Build a zero snetio via the public namedtuple fields we know.
        try:
            from collections import namedtuple
            snetio = namedtuple(
                "snetio",
                ["bytes_sent", "bytes_recv", "packets_sent", "packets_recv",
                 "errin", "errout", "dropin", "dropout"],
            )
            last_good = snetio(0, 0, 0, 0, 0, 0, 0, 0)
        except Exception:
            last_good = None

    def _fast_net_io_counters(pernic=False, nowrap=True, _last=last_good):
        # pernic=True wants a dict; distributed only calls the scalar form.
        return {} if pernic else _last

    psutil.net_io_counters = _fast_net_io_counters
    psutil._spyde_net_io_patched = True
    logger.warning(
        "[dask] psutil.net_io_counters() did not return within %.1fs — patched "
        "it to a non-blocking stub so Dask's SystemMonitor can't freeze cluster "
        "startup (net-io dashboard telemetry disabled; harmless).",
        probe_timeout,
    )


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

    ready = Signal()           # scheduler + client usable (workers may still be spawning)
    workers_ready = Signal()   # all requested workers registered (heavy-worker split set)
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
        import time
        self._start_t0 = time.monotonic()
        logger.info("[dask] start() called — launching cluster on background thread")
        self._thread = threading.Thread(target=self._run, daemon=True, name="dask-startup")
        self._thread.start()

    def _run(self) -> None:
        import time
        t0 = getattr(self, "_start_t0", time.monotonic())

        def _elapsed() -> str:
            return f"{time.monotonic() - t0:.1f}s"

        try:
            logger.info("[dask] _run begin (t+%s): building LocalCluster", _elapsed())
            total_mem = psutil.virtual_memory().total
            memory_per_worker = int(total_mem * 0.80) // max(self._n_workers, 1)
            logger.info(
                "[dask] memory per worker: %.1f GB (total=%.1f GB, workers=%d)",
                memory_per_worker / 1024**3,
                total_mem / 1024**3,
                self._n_workers,
            )

            # ROOT-CAUSE FIX for the ~72s cluster-startup stall on this Windows
            # box: Dask's SystemMonitor (built inside Scheduler.__init__) calls
            # psutil.net_io_counters(), which blocked ~72s in the Electron backend
            # process (AzureAD-joined, many virtual NICs). Neutralize it BEFORE
            # building the cluster. See _neutralize_slow_net_io_counters.
            _neutralize_slow_net_io_counters()

            # Build the SCHEDULER first (n_workers=0) so the client + dashboard
            # come up fast, THEN spawn workers and let them register in the
            # background — so the app is usable in seconds even if worker spawn is
            # slow (e.g. packaged Windows, where each worker re-execs the bundle).
            t_sched = time.monotonic()
            logger.info("[dask] calling LocalCluster(n_workers=0) … (t+%s)", _elapsed())
            cluster = LocalCluster(
                n_workers=0,
                threads_per_worker=self._threads_per_worker,
                memory_limit=memory_per_worker,
            )
            logger.info(
                "[dask] scheduler up in %.1fs (t+%s); connecting Client",
                time.monotonic() - t_sched, _elapsed(),
            )
            client = Client(cluster)
            n_gpus = _probe_gpus()
            gpu_worker_address = "gpu_available" if n_gpus > 0 else None

            self._cluster = cluster
            self._client = client
            self._gpu_worker_address = gpu_worker_address

            # Scheduler + client are usable NOW — report ready immediately so the
            # app stops waiting. Workers spawn next (logged with their own timing).
            logger.info(
                "[dask] scheduler READY (t+%s), gpus=%d. Dashboard: %s — "
                "spawning %d workers in background…",
                _elapsed(), n_gpus, client.dashboard_link, self._n_workers,
            )
            self.ready.emit()

            t_workers = time.monotonic()
            cluster.scale(self._n_workers)
            try:
                client.wait_for_workers(self._n_workers, timeout=180)
            except Exception as e:
                logger.warning(
                    "[dask] only %d/%d workers registered within timeout (t+%s): %s",
                    self.worker_count(), self._n_workers, _elapsed(), e,
                )

            n_up = self.worker_count()
            logger.info(
                "[dask] %d/%d workers up after %.1fs (t+%s)",
                n_up, self._n_workers, time.monotonic() - t_workers, _elapsed(),
            )

            # scheduler_info can momentarily return a dict WITHOUT a "workers" key
            # (or with none registered yet) — never let that KeyError abort startup,
            # which left the cluster with the scheduler up but 0 workers (loads then
            # hung forever waiting on a worker). Tolerate it: no heavy split, the
            # navigator/compute still run on whatever workers exist.
            try:
                info = client.scheduler_info(n_workers=-1) or {}
                worker_keys = list((info.get("workers") or {}).keys())
            except Exception as e:
                logger.warning("[dask] scheduler_info failed (t+%s): %s", _elapsed(), e)
                worker_keys = []
            heavy = worker_keys[1:]
            self._heavy_compute_workers = heavy if heavy else None
            logger.info(
                "[dask] heavy-compute workers: %d of %d (worker 0 reserved for "
                "the live navigator); cluster fully READY (t+%s)",
                len(heavy), len(worker_keys), _elapsed(),
            )
            # Re-emit so listeners that gate heavy-worker routing on ready get the
            # final state too (ready already fired once when the scheduler came up
            # so the UI unblocked early; this second emit is idempotent for them).
            self.workers_ready.emit()
        except Exception as exc:
            logger.exception("[dask] FAILED to start Dask cluster (t+%s)", _elapsed())
            self.error.emit(str(exc))

    def worker_count(self) -> int:
        """True number of registered workers. Uses ``n_workers=-1`` because plain
        ``scheduler_info()`` truncates its worker list to 5."""
        client = self._client
        if client is None:
            return 0
        try:
            return len(client.scheduler_info(n_workers=-1).get("workers", {}))
        except Exception:
            return 0

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

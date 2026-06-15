"""
benchmark_workflow.py — end-to-end stage-timed benchmark of SpyDE workflows.

Times every stage a user actually experiences, against any (large) dataset:

  disk        raw sequential read speed of the source drive (the physics floor)
  cluster     dask LocalCluster startup (app-equivalent worker count)
  open        lazy file open, app-style chunking (signal axes unchunked)
  navigator   sum over signal axes — what file-open computes; equals the
              "as fast as data can be loaded" baseline
  frame       single diffraction-pattern fetch latency (navigation feel)
  vimage      virtual image ((data * mask).sum over signal axes)
  vectors     full find-diffraction-vectors run with auto parameters,
              including graph-build time, time-to-first-chunk, progress curve

Throughout each stage a sampler polls worker memory + spill metrics, and a
GIL-heartbeat thread measures the longest stall the main thread suffered —
a direct proxy for how long the GUI would freeze in the real app (graph
construction happens in a thread of the GUI process and holds the GIL).

Usage:
  python spyde/tests/benchmark_workflow.py PATH --nav 256 256 [options]

Options:
  --nav Y X         navigation shape of the mrc movie (app asks via dialog)
  --workers N       dask workers (default: app rule from cpu count)
  --threads T       threads per worker (default: app rule)
  --quick           crop navigation to 64x64 for the vectors stage
  --skip-vectors    skip the find_vectors stage
  --gpu MODE        SPYDE_FV_GPU value (one|off|all|N); default "one"
  --json PATH       also dump results as JSON

Results print as a table; paste-ready for benchmarks.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Monitors
# ─────────────────────────────────────────────────────────────────────────────

class GilHeartbeat:
    """Measures the longest main-process stall (GIL hold) during a stage.

    A daemon thread sleeps 20 ms at a time; any gap far beyond that means
    some thread held the GIL (e.g. dask graph construction) — in the GUI
    app that exact gap is a frozen interface.
    """

    def __init__(self):
        self._max_gap = 0.0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._max_gap = 0.0
        self._stop.clear()

        def _run():
            last = time.perf_counter()
            while not self._stop.is_set():
                time.sleep(0.02)
                now = time.perf_counter()
                gap = now - last
                if gap > self._max_gap:
                    self._max_gap = gap
                last = now

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def max_stall_ms(self) -> float:
        return self._max_gap * 1e3


class ClusterMemorySampler:
    """Polls scheduler worker metrics: peak cluster RSS and spilled bytes."""

    def __init__(self, client, interval: float = 1.0):
        self.client = client
        self.interval = interval
        self.peak_memory = 0
        self.peak_spilled = 0
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        try:
            info = self.client.scheduler_info(n_workers=-1)["workers"]
        except Exception:
            return
        mem = 0
        spilled = 0
        for w in info.values():
            metrics = w.get("metrics", {}) or {}
            mem += int(metrics.get("memory", 0))
            spill_info = metrics.get("spilled_bytes") or metrics.get("spilled_nbytes") or 0
            if isinstance(spill_info, dict):  # newer distributed: {"memory": .., "disk": ..}
                spilled += int(spill_info.get("disk", 0))
            else:
                spilled += int(spill_info)
        self.peak_memory = max(self.peak_memory, mem)
        self.peak_spilled = max(self.peak_spilled, spilled)

    def __enter__(self):
        self._stop.clear()

        def _run():
            while not self._stop.is_set():
                self._sample()
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()


class ShmProgressSampler:
    """Samples the live count-map shm buffer: time-to-first-chunk + curve."""

    def __init__(self, nav_shape, shm_name, interval: float = 0.25):
        self.nav_shape = nav_shape
        self.shm_name = shm_name
        self.interval = interval
        self.samples = []  # (t, fraction_finite)
        self._stop = threading.Event()
        self._thread = None
        self.t0 = None

    def __enter__(self):
        from spyde.drawing.update_functions import read_live_buffer
        self._stop.clear()
        self.t0 = time.perf_counter()

        def _run():
            while not self._stop.is_set():
                arr = read_live_buffer(self.nav_shape, self.shm_name)
                frac = float(np.isfinite(arr).mean())
                self.samples.append((time.perf_counter() - self.t0, frac))
                self._stop.wait(self.interval)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def time_to_first_chunk(self):
        for t, frac in self.samples:
            if frac > 0:
                return t
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────────

def stage_disk(path: str, results: dict, read_gb: float = 2.0):
    """Raw sequential read speed from a deep offset (avoids OS cache)."""
    size = os.path.getsize(path)
    block = 64 * 1024 * 1024
    n_blocks = max(1, int(read_gb * 1e9 / block))
    with open(path, "rb", buffering=0) as f:
        f.seek(size // 3)
        t0 = time.perf_counter()
        total = 0
        for _ in range(n_blocks):
            total += len(f.read(block))
        dt = time.perf_counter() - t0
    results["disk"] = {
        "time_s": dt,
        "MB_per_s": total / 1e6 / dt,
        "file_GB": size / 1e9,
        "full_pass_floor_s": (size / 1e6) / (total / 1e6 / dt),
    }


def stage_cluster(n_workers, threads, results: dict):
    from dask.distributed import Client, LocalCluster
    import psutil
    mem_per_worker = int(psutil.virtual_memory().total * 0.80) // max(n_workers, 1)
    t0 = time.perf_counter()
    cluster = LocalCluster(
        n_workers=n_workers, threads_per_worker=threads,
        memory_limit=mem_per_worker,
    )
    client = Client(cluster)
    # Workers register asynchronously — wait so every stage sees the full
    # cluster (the app instead scales 1 -> N in the background).
    client.wait_for_workers(n_workers, timeout=120)
    dt = time.perf_counter() - t0
    results["cluster"] = {
        "time_s": dt, "n_workers": n_workers, "threads_per_worker": threads,
        "memory_limit_GB": mem_per_worker / 1e9,
    }
    return cluster, client


def stage_open(path: str, nav_shape, results: dict):
    """Lazy open with the app's mrc kwargs (signal axes unchunked)."""
    import hyperspy.api as hs
    kwargs = {"lazy": True}
    if path.lower().endswith(".mrc") and nav_shape:
        kwargs["navigation_shape"] = tuple(nav_shape[::-1])  # HS x-first order
        kwargs["chunks"] = ("auto",) * len(nav_shape) + (-1, -1)
    with GilHeartbeat() as hb:
        t0 = time.perf_counter()
        s = hs.load(path, **kwargs)
        dt = time.perf_counter() - t0
    results["open"] = {
        "time_s": dt,
        "shape": list(s.data.shape),
        "dtype": str(s.data.dtype),
        "chunks_nav": [list(c)[:3] for c in s.data.chunks[:-2]],
        "sig_chunked": any(len(c) > 1 for c in s.data.chunks[-2:]),
        "main_thread_stall_ms": hb.max_stall_ms,
    }
    return s


def stage_navigator(s, client, results: dict, label="navigator"):
    """Sum over signal axes — the file-open navigator / data-rate baseline."""
    nbytes = s.data.nbytes
    nav = s.data.sum(axis=(-2, -1))
    with ClusterMemorySampler(client) as mem, GilHeartbeat() as hb:
        t0 = time.perf_counter()
        fut = client.compute(nav)
        graph_dt = time.perf_counter() - t0
        out = fut.result()
        dt = time.perf_counter() - t0
    results[label] = {
        "time_s": dt,
        "graph_submit_s": graph_dt,
        "GB_per_s": nbytes / 1e9 / dt,
        "peak_cluster_mem_GB": mem.peak_memory / 1e9,
        "spilled_GB": mem.peak_spilled / 1e9,
        "main_thread_stall_ms": hb.max_stall_ms,
    }
    return out


def stage_frame(s, client, results: dict):
    """Single-pattern fetch latency at a few positions (navigation feel)."""
    ny, nx = s.data.shape[0], s.data.shape[1]
    times = []
    for (iy, ix) in [(0, 0), (ny // 2, nx // 2), (ny - 1, nx - 1)]:
        t0 = time.perf_counter()
        client.compute(s.data[iy, ix]).result()
        times.append(time.perf_counter() - t0)
    results["frame"] = {
        "first_ms": times[0] * 1e3,
        "median_ms": float(np.median(times)) * 1e3,
    }


def stage_vimage(s, client, results: dict):
    """Virtual image: (data * mask).sum over signal axes, like the app."""
    ky, kx = s.data.shape[-2], s.data.shape[-1]
    yy, xx = np.ogrid[:ky, :kx]
    mask = (((yy - ky / 2) ** 2 + (xx - kx / 2) ** 2) <= (ky / 6) ** 2).astype(np.float32)
    nbytes = s.data.nbytes
    vi = (s.data * mask).sum(axis=(-2, -1))
    with ClusterMemorySampler(client) as mem, GilHeartbeat() as hb:
        t0 = time.perf_counter()
        fut = client.compute(vi)
        graph_dt = time.perf_counter() - t0
        fut.result()
        dt = time.perf_counter() - t0
    results["vimage"] = {
        "time_s": dt,
        "graph_submit_s": graph_dt,
        "GB_per_s": nbytes / 1e9 / dt,
        "peak_cluster_mem_GB": mem.peak_memory / 1e9,
        "spilled_GB": mem.peak_spilled / 1e9,
        "main_thread_stall_ms": hb.max_stall_ms,
    }


def stage_vectors(s, client, results: dict, quick: bool):
    """Full find-vectors run with auto params, instrumented end to end."""
    from spyde.actions.find_vectors import _auto_params, _do_compute_vectors
    from spyde.drawing.update_functions import ensure_live_buffer

    if quick:
        s = s.inav[:64, :64]

    nav_dim = s.axes_manager.navigation_dimension
    nav_shape_2d = tuple(s.data.shape[:nav_dim])[-2:]
    nbytes = s.data.nbytes

    # Auto params from the centre frame (what the caret does on open)
    t0 = time.perf_counter()
    iy, ix = nav_shape_2d[0] // 2, nav_shape_2d[1] // 2
    frame = np.asarray(client.compute(s.data[iy, ix]).result(), dtype=np.float32)
    params = _auto_params(frame)
    autoparam_dt = time.perf_counter() - t0

    class _Tree:
        pass
    tree = _Tree()
    tree.client = client

    shm_name = "spyde_bench_fv"
    shm = ensure_live_buffer(nav_shape_2d, shm_name)
    try:
        with ClusterMemorySampler(client) as mem, GilHeartbeat() as hb, \
                ShmProgressSampler(nav_shape_2d, shm_name) as prog:
            t0 = time.perf_counter()
            vecs = _do_compute_vectors(s, params, None, tree, shm_name=shm_name)
            dt = time.perf_counter() - t0
    finally:
        shm.close()
        try:
            shm.unlink()
        except Exception:
            pass

    n_vec = int(len(vecs.flat_buffer)) if vecs is not None else -1
    results["vectors"] = {
        "time_s": dt,
        "GB_per_s": nbytes / 1e9 / dt,
        "auto_params_s": autoparam_dt,
        "params": {k: (float(v) if not isinstance(v, bool) else v)
                   for k, v in params.items()},
        "n_vectors": n_vec,
        "time_to_first_chunk_s": prog.time_to_first_chunk,
        "peak_cluster_mem_GB": mem.peak_memory / 1e9,
        "spilled_GB": mem.peak_spilled / 1e9,
        "main_thread_stall_ms": hb.max_stall_ms,
        "quick_64x64": quick,
        "gpu_mode": os.environ.get("SPYDE_FV_GPU", "one"),
        "progress_curve": [(round(t, 1), round(f, 3)) for t, f in prog.samples[::4]],
    }
    return vecs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def app_worker_rule():
    """Replicates MainWindow's worker-count rule."""
    cpu = os.cpu_count() or 4
    if cpu < 4:
        return 1, 1
    if cpu <= 16:
        return (cpu // 2) - 1, 2
    return (cpu // 4) - 1, 4


def fmt_row(name, d):
    t = d.get("time_s")
    gbs = d.get("GB_per_s")
    stall = d.get("main_thread_stall_ms")
    spill = d.get("spilled_GB")
    mem = d.get("peak_cluster_mem_GB")
    parts = [f"{name:<10}"]
    parts.append(f"{t:8.1f} s" if t is not None else " " * 10)
    parts.append(f"{gbs:6.2f} GB/s" if gbs else " " * 11)
    parts.append(f"stall {stall:7.0f} ms" if stall is not None else "")
    parts.append(f"mem {mem:5.1f} GB" if mem else "")
    parts.append(f"SPILL {spill:5.1f} GB" if spill else "")
    return "  ".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--nav", nargs=2, type=int, default=None,
                    metavar=("Y", "X"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-vectors", action="store_true")
    ap.add_argument("--skip-disk", action="store_true")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["SPYDE_FV_GPU"] = args.gpu

    nw, tpw = app_worker_rule()
    if args.workers:
        nw = args.workers
    if args.threads:
        tpw = args.threads

    results = {"path": args.path, "argv": sys.argv[1:]}
    t_all = time.perf_counter()

    if not args.skip_disk:
        print("== disk ==", flush=True)
        stage_disk(args.path, results)
        print(fmt_row("disk", results["disk"]),
              f"-> {results['disk']['MB_per_s']:.0f} MB/s, full pass floor "
              f"{results['disk']['full_pass_floor_s']:.0f} s", flush=True)

    print("== cluster ==", flush=True)
    cluster, client = stage_cluster(nw, tpw, results)
    print(fmt_row("cluster", results["cluster"]), flush=True)

    try:
        print("== open ==", flush=True)
        s = stage_open(args.path, args.nav, results)
        print(fmt_row("open", results["open"]),
              f"chunks_nav={results['open']['chunks_nav']}"
              f" sig_chunked={results['open']['sig_chunked']}", flush=True)

        print("== navigator (data-rate baseline) ==", flush=True)
        stage_navigator(s, client, results)
        print(fmt_row("navigator", results["navigator"]), flush=True)

        print("== frame latency ==", flush=True)
        stage_frame(s, client, results)
        print(f"frame       first {results['frame']['first_ms']:.0f} ms, "
              f"median {results['frame']['median_ms']:.0f} ms", flush=True)

        print("== virtual image ==", flush=True)
        stage_vimage(s, client, results)
        print(fmt_row("vimage", results["vimage"]), flush=True)

        if not args.skip_vectors:
            print("== find vectors ==", flush=True)
            stage_vectors(s, client, results, quick=args.quick)
            v = results["vectors"]
            print(fmt_row("vectors", v),
                  f"first-chunk {v['time_to_first_chunk_s'] and round(v['time_to_first_chunk_s'], 1)} s,"
                  f" {v['n_vectors']} vectors", flush=True)
    finally:
        try:
            client.close()
            cluster.close()
        except Exception:
            pass

    results["total_s"] = time.perf_counter() - t_all
    print(f"\nTOTAL {results['total_s']:.0f} s")

    if args.json_path:
        with open(args.json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print("json ->", args.json_path)


if __name__ == "__main__":
    main()

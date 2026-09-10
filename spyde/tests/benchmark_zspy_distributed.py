"""What a distributed batch pays to decode a .zspy, with and without blosc's
thread pool in the worker processes.

Run directly, not under pytest::

    SPYDE_ZSPY_BENCH=D:\data\movie.zspy uv run --no-sync python -m spyde.tests.benchmark_zspy_distributed

Starts a ``LocalCluster`` shaped as the application would start it on this
machine twice, once with the worker plugin
(which applies ``spyde.external.numcodecs.blosc_threads``) and once with the
patch switched off in the workers, and times the same two computes the in-process
benchmark uses over the same far-corner square of chunks: a navigator sum and a
Find Vectors DoG batch. Each compute runs twice and the second time is reported,
so the disk read and the first CUDA context are not in the comparison.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

PATH_VARIABLE = "SPYDE_ZSPY_BENCH"
CHUNKS_PER_AXIS = 4


def _cluster_shape():
    """The worker count, threads per worker and blosc pool size per worker the
    application itself would start on this machine."""
    from spyde.backend.app import _compute_worker_plan
    cpu_count = os.cpu_count() or 1
    n_workers, threads = _compute_worker_plan(cpu_count)
    return n_workers, threads, max(1, cpu_count // n_workers)


def _region(path):
    import dask.array as da
    data = da.from_zarr(os.path.join(path, "Experiments", "__unnamed__", "data"))
    chunk = tuple(int(c[0]) for c in data.chunks[:2])
    extents = []
    for axis in range(2):
        length = chunk[axis] * CHUNKS_PER_AXIS
        start = max(0, int(data.shape[axis]) - length)
        extents.append(slice(start, start + length))
    return data[tuple(extents)]


def _second_of_two(compute):
    for _ in range(2):
        started = time.perf_counter()
        compute()
        elapsed = (time.perf_counter() - started) * 1e3
    return elapsed


def _dog(region):
    from spyde.actions.find_vectors.chunk import _find_vectors_chunk_dog
    from spyde.actions.find_vectors.detectors import (
        DEFAULT_DOG_SIGMA1, DEFAULT_DOG_SIGMA2, DEFAULT_DOG_THRESHOLD,
    )
    from spyde.actions.find_vectors.gpu_runtime import MAX_PEAKS
    return region.map_blocks(
        _find_vectors_chunk_dog, 0, 2, 0.0,
        DEFAULT_DOG_SIGMA1, DEFAULT_DOG_SIGMA2, DEFAULT_DOG_THRESHOLD, 3,
        False, None, dtype=np.float32,
        chunks=(region.chunks[0], region.chunks[1], (MAX_PEAKS,), (3,)))


def _worker_reports_pool(dask_worker=None):
    import numcodecs.blosc as blosc
    from spyde.external.numcodecs.blosc_threads import MARKER
    return bool(getattr(blosc.decompress, MARKER, False)), blosc.use_threads


def _measure(path, patched):
    from distributed import Client, LocalCluster
    from spyde.dask_manager import _WorkerTuningPlugin

    n_workers, threads, pool = _cluster_shape()
    # Set the switch explicitly both ways, so a shell that exports it off
    # cannot turn the comparison into off against off.
    env = {"SPYDE_BLOSC_THREADS": str(pool) if patched else "0"}
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads,
                           processes=True, dashboard_address=None, env=env)
    client = Client(cluster)
    try:
        client.register_plugin(_WorkerTuningPlugin(blosc_threads=pool))
        pools = client.run(_worker_reports_pool)
        region = _region(path)
        total = region.sum(axis=(-2, -1))
        sum_ms = _second_of_two(lambda: total.compute(scheduler=client))
        peaks = _dog(region)
        dog_ms = _second_of_two(lambda: peaks.compute(scheduler=client))
        return {"n_workers": n_workers, "threads": threads, "pool": pool,
                "workers_with_pool": sum(1 for v in pools.values() if v[0]),
                "navigator_sum_ms": round(sum_ms), "dog_batch_ms": round(dog_ms)}
    finally:
        client.close()
        cluster.close()


def main():
    path = os.environ.get(PATH_VARIABLE)
    if not path:
        print(f"set {PATH_VARIABLE} to a .zspy dataset to run this benchmark")
        return
    n_workers, threads, pool = _cluster_shape()
    print(f"file    {path}")
    print(f"cluster {n_workers} worker processes x {threads} threads (the "
          f"application's own plan for this machine), blosc pool {pool} per "
          f"worker, {CHUNKS_PER_AXIS}x{CHUNKS_PER_AXIS} chunks")
    for patched in (False, True):
        result = _measure(path, patched)
        print(f"workers {'with' if patched else 'without'} the pool: "
              f"{result['workers_with_pool']}/{n_workers} report it on; "
              f"navigator sum {result['navigator_sum_ms']} ms; "
              f"DoG batch {result['dog_batch_ms']} ms", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)

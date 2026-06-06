"""
benchmark_virtual_image.py
==========================
End-to-end benchmark for virtual image computation from a 4D-STEM MRC file.

Measures wall time and effective throughput for:
  1. Raw numpy  -- sequential read + masked sum, single-threaded (ceiling reference)
  2. Dask threads  -- current spyde path (scheduler='threads')
  3. GPU sequential  -- open+readinto per row-band, GPU einsum
  4. Dask + GPU  -- dask reads chunks, GPU reduces each chunk
  5. Dask distributed  -- LocalCluster with multiple workers

Run with:
    .venv/Scripts/python spyde/tests/benchmark_virtual_image.py
    .venv/Scripts/python spyde/tests/benchmark_virtual_image.py --path D:/data/file.mrc
"""

from __future__ import annotations

import argparse
import struct
import time

import dask.array as da
import hyperspy.api as hs
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(path: str):
    with open(path, "rb") as f:
        hdr = f.read(1024)
    nx, ny, nz = struct.unpack_from("3i", hdr, 0)
    ext = struct.unpack_from("i", hdr, 92)[0]
    return nx, ny, nz, 1024 + ext


def _masks(ky, kx, bf_r=32, adf_r1=64, adf_r2=128):
    cy, cx = ky // 2, kx // 2
    yy, xx = np.ogrid[-cy : ky - cy, -cx : kx - cx]
    r2 = (yy ** 2 + xx ** 2).astype(np.float32)
    bf  = (r2 <= bf_r ** 2).astype(np.float32)
    adf = ((r2 >= adf_r1 ** 2) & (r2 <= adf_r2 ** 2)).astype(np.float32)
    return bf, adf


def _row(label: str, t: float, gb: float, ref: float):
    return (
        f"  {label:<35s} {t:7.2f}s  {gb/t*1000:6.0f} MB/s  "
        f"{ref/t:5.2f}x"
    )


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def method_numpy_sequential(path, offset, n_nav_y, n_nav_x, ky, kx,
                             bf_mask, adf_mask, row_size=11):
    """Single-threaded sequential read + numpy masked sum. Disk bandwidth ceiling."""
    buf = np.empty((row_size * n_nav_x, ky, kx), dtype=np.float32)
    bf_out  = np.empty((n_nav_y, n_nav_x), dtype=np.float32)
    adf_out = np.empty((n_nav_y, n_nav_x), dtype=np.float32)

    t_read = t_compute = 0.0
    for rs in range(0, n_nav_y, row_size):
        re = min(rs + row_size, n_nav_y)
        n  = re - rs
        chunk = buf[: n * n_nav_x]
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            f.seek(offset + rs * n_nav_x * ky * kx * 4)
            f.readinto(chunk)
        t_read += time.perf_counter() - t0
        chunk = chunk.reshape(n, n_nav_x, ky, kx)
        t0 = time.perf_counter()
        bf_out[rs:re]  = (chunk * bf_mask).sum(axis=(2, 3))
        adf_out[rs:re] = (chunk * adf_mask).sum(axis=(2, 3))
        t_compute += time.perf_counter() - t0

    return bf_out, adf_out, t_read, t_compute


def method_dask_threads(sig, bf_mask, adf_mask):
    """Dask threads scheduler -- current spyde path."""
    arr = sig.data
    t0 = time.perf_counter()
    vi_bf  = da.einsum("yxij,ij->yx", arr, bf_mask)
    vi_adf = da.einsum("yxij,ij->yx", arr, adf_mask)
    both   = da.stack([vi_bf, vi_adf])
    t_graph = time.perf_counter() - t0
    n_tasks = len(dict(both.__dask_graph__()))

    t0 = time.perf_counter()
    result = both.compute(scheduler="threads")
    t_compute = time.perf_counter() - t0

    return result[0], result[1], t_graph, n_tasks, t_compute


def method_dask_overhead(sig):
    """Measure pure dask scheduling overhead with in-memory data."""
    import dask
    arr = sig.data

    # Materialise first chunk to RAM
    first_chunk = np.array(arr[:11, :11].compute(scheduler="synchronous"))
    d_chunk = da.from_array(first_chunk, chunks=first_chunk.shape)

    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        d_chunk.sum(axis=(2, 3)).compute(scheduler="synchronous")
    overhead_ms = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        first_chunk.sum(axis=(2, 3))
    numpy_ms = (time.perf_counter() - t0) / N * 1000

    return overhead_ms, numpy_ms


def method_gpu_sequential(path, offset, n_nav_y, n_nav_x, ky, kx,
                           bf_mask, adf_mask, row_size=11, device="cuda"):
    """Sequential read into pinned buffer, GPU einsum per row-band."""
    dev = torch.device(device)
    # Stay in float32 for H2D speed; use Kahan-style double-precision accumulation
    # on CPU after the fact if needed. For virtual images float32 is sufficient.
    bf_t  = torch.from_numpy(bf_mask).to(dev)
    adf_t = torch.from_numpy(adf_mask).to(dev)
    buf   = torch.empty((row_size * n_nav_x, ky, kx), dtype=torch.float32).pin_memory()
    # Accumulate in float64 on CPU to match numpy reference
    bf_acc  = np.zeros((n_nav_y, n_nav_x), dtype=np.float64)
    adf_acc = np.zeros((n_nav_y, n_nav_x), dtype=np.float64)
    stream  = torch.cuda.Stream(device=dev)

    t_read = t_h2d = t_compute = 0.0
    torch.cuda.synchronize()

    for rs in range(0, n_nav_y, row_size):
        re = min(rs + row_size, n_nav_y)
        n  = re - rs
        chunk_np = buf[: n * n_nav_x].numpy()

        t0 = time.perf_counter()
        with open(path, "rb") as f:
            f.seek(offset + rs * n_nav_x * ky * kx * 4)
            f.readinto(chunk_np)
        t_read += time.perf_counter() - t0

        with torch.cuda.stream(stream):
            t0 = time.perf_counter()
            chunk_t = buf[: n * n_nav_x].to(dev, non_blocking=True).view(n, n_nav_x, ky, kx)
            t_h2d += time.perf_counter() - t0

            t0 = time.perf_counter()
            bf_row  = torch.einsum("nxij,ij->nx", chunk_t, bf_t)
            adf_row = torch.einsum("nxij,ij->nx", chunk_t, adf_t)
            t_compute += time.perf_counter() - t0

        stream.synchronize()
        bf_acc[rs:re]  = bf_row.cpu().numpy().astype(np.float64)
        adf_acc[rs:re] = adf_row.cpu().numpy().astype(np.float64)

    return bf_acc, adf_acc, t_read, t_h2d, t_compute


def method_dask_gpu(sig, bf_mask, adf_mask, device="cuda"):
    """
    Dask reads chunks (threads scheduler), GPU reduces each chunk via map_blocks.

    For I/O-bound workloads (disk is the bottleneck) this is slower than
    dask threads alone because:
    - 576 dask tasks each acquire the GIL to launch a CUDA kernel
    - CUDA launches from multiple threads serialize on the GPU command queue
    - The H2D transfer + kernel launch overhead dominates the tiny reduction time

    GPU reduction only wins when the compute per chunk is much larger than the
    H2D transfer overhead -- e.g. peak-finding (NXCORR) rather than dot-product.
    """
    import threading
    dev  = torch.device(device)
    bf_t  = torch.from_numpy(bf_mask).to(dev)
    adf_t = torch.from_numpy(adf_mask).to(dev)
    # Lock to serialize GPU launches across dask threads
    gpu_lock = threading.Lock()

    def _gpu_reduce_both(chunk, block_id=None):
        arr = np.ascontiguousarray(chunk)
        with gpu_lock:
            t = torch.from_numpy(arr).to(dev, non_blocking=False)
            bf_r  = torch.einsum("yxij,ij->yx", t, bf_t).cpu().numpy()
            adf_r = torch.einsum("yxij,ij->yx", t, adf_t).cpu().numpy()
        # Return stacked (2, cy, cx) -- we'll split after compute
        return np.stack([bf_r, adf_r], axis=0)

    arr = sig.data

    t0 = time.perf_counter()
    # Each chunk (cy, cx, KY, KX) -> (2, cy, cx)
    reduced = da.map_blocks(
        _gpu_reduce_both,
        arr,
        dtype=np.float32,
        new_axis=0,
        chunks=(2,) + tuple(arr.chunks[i] for i in range(2)),
        drop_axis=(2, 3),
    )
    t_graph = time.perf_counter() - t0
    n_tasks  = len(dict(reduced.__dask_graph__()))

    t0 = time.perf_counter()
    out = reduced.compute(scheduler="threads")
    t_compute = time.perf_counter() - t0

    return out[0], out[1], t_graph, n_tasks, t_compute


def method_dask_distributed(sig, bf_mask, adf_mask, n_workers=4,
                             threads_per_worker=4):
    """Dask LocalCluster with multiple workers."""
    from dask.distributed import Client, LocalCluster

    t0 = time.perf_counter()
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit="8GB",
    )
    client  = Client(cluster)
    t_setup = time.perf_counter() - t0

    arr    = sig.data
    vi_bf  = da.einsum("yxij,ij->yx", arr, bf_mask)
    vi_adf = da.einsum("yxij,ij->yx", arr, adf_mask)
    both   = da.stack([vi_bf, vi_adf])
    n_tasks = len(dict(both.__dask_graph__()))

    t0 = time.perf_counter()
    result = client.compute(both).result()
    t_compute = time.perf_counter() - t0

    client.close()
    cluster.close()
    return result[0], result[1], t_setup, n_tasks, t_compute


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(path: str):
    kx, ky, nz, offset = _header(path)
    n_nav = int(nz ** 0.5)
    assert n_nav * n_nav == nz, f"Non-square nav grid: nz={nz}"
    n_nav_y = n_nav_x = n_nav
    gb = nz * ky * kx * 4 / 1e9

    print()
    print("=" * 65)
    print("VIRTUAL IMAGE BENCHMARK")
    print(f"  File:  {path}")
    print(f"  Shape: ({n_nav_y}, {n_nav_x}, {ky}, {kx})  {gb:.1f} GB")
    print(f"  GPU:   {torch.cuda.get_device_name(0)}")
    print("=" * 65)
    print()

    bf_mask, adf_mask = _masks(ky, kx)
    print(f"  BF mask:  {bf_mask.sum():.0f} px  (r={32})")
    print(f"  ADF mask: {adf_mask.sum():.0f} px  (r={64}-{128})")
    print()

    # ── Dask overhead probe ────────────────────────────────────────────────
    print("--- Dask scheduling overhead probe ---")
    sig = hs.load(path, lazy=True)
    overhead_ms, numpy_ms = method_dask_overhead(sig)
    n_chunks = np.prod([len(c) for c in sig.data.chunks])
    chunk_mb = np.prod(sig.data.chunks[0][0:1] + sig.data.chunks[1][0:1]) * ky * kx * 4 / 1e6
    print(f"  n_chunks={n_chunks}  chunk_size={chunk_mb:.0f} MB")
    print(f"  Dask task overhead (in-memory sum):  {overhead_ms:.2f} ms/task")
    print(f"  Numpy sum same chunk:                {numpy_ms:.2f} ms")
    print(f"  Dask scheduling overhead for VI ({n_chunks} tasks): "
          f"{n_chunks * overhead_ms / 1000:.2f}s")
    print(f"  Numpy compute only (all chunks):     "
          f"{n_chunks * numpy_ms / 1000:.2f}s")
    print()

    times = {}

    # ── Method 1: numpy sequential ────────────────────────────────────────
    print("--- 1. Raw numpy sequential (read ceiling) ---")
    t_wall_start = time.perf_counter()
    bf, adf, t_r, t_c = method_numpy_sequential(
        path, offset, n_nav_y, n_nav_x, ky, kx, bf_mask, adf_mask
    )
    t_wall = time.perf_counter() - t_wall_start
    times["numpy"] = t_wall
    print(f"  Total: {t_wall:.2f}s  read={t_r:.2f}s  compute={t_c:.2f}s")
    print(f"  Throughput: {gb/t_wall*1000:.0f} MB/s  BF mean={bf.mean():.1f}")
    ref = t_wall
    print()

    # ── Method 2: dask threads ────────────────────────────────────────────
    print("--- 2. Dask threads (current spyde path) ---")
    t_wall_start = time.perf_counter()
    bf2, adf2, t_graph, n_tasks, t_compute = method_dask_threads(sig, bf_mask, adf_mask)
    t_wall = time.perf_counter() - t_wall_start
    times["dask_threads"] = t_wall
    print(f"  Graph build: {t_graph*1000:.1f} ms  tasks={n_tasks}")
    print(f"  Compute:     {t_compute:.2f}s")
    print(f"  Total:       {t_wall:.2f}s  {gb/t_wall*1000:.0f} MB/s  {ref/t_wall:.2f}x")
    print(f"  BF mean={bf2.mean():.1f}  match numpy: {np.allclose(bf, bf2, rtol=1e-3)}")
    print()

    # ── Method 3: GPU sequential ──────────────────────────────────────────
    print("--- 3. GPU sequential (read -> pinned -> GPU einsum) ---")
    t_wall_start = time.perf_counter()
    bf3, adf3, t_r, t_h2d, t_c = method_gpu_sequential(
        path, offset, n_nav_y, n_nav_x, ky, kx, bf_mask, adf_mask
    )
    t_wall = time.perf_counter() - t_wall_start
    times["gpu_seq"] = t_wall
    print(f"  read={t_r:.2f}s  H2D={t_h2d:.2f}s  compute={t_c:.2f}s")
    print(f"  Total: {t_wall:.2f}s  {gb/t_wall*1000:.0f} MB/s  {ref/t_wall:.2f}x")
    print(f"  BF mean={bf3.mean():.1f}  match numpy: {np.allclose(bf, bf3, rtol=1e-2)}")
    print()

    # ── Method 4: Dask + GPU ──────────────────────────────────────────────
    print("--- 4. Dask threads + GPU reduce per chunk ---")
    t_wall_start = time.perf_counter()
    try:
        bf4, adf4, t_graph, n_tasks, t_compute = method_dask_gpu(sig, bf_mask, adf_mask)
        t_wall = time.perf_counter() - t_wall_start
        times["dask_gpu"] = t_wall
        print(f"  Graph build: {t_graph*1000:.1f} ms  tasks={n_tasks}")
        print(f"  Compute:     {t_compute:.2f}s")
        print(f"  Total:       {t_wall:.2f}s  {gb/t_wall*1000:.0f} MB/s  {ref/t_wall:.2f}x")
        print(f"  BF mean={bf4.mean():.1f}  match numpy: {np.allclose(bf, bf4, rtol=1e-2)}")
    except Exception as e:
        print(f"  FAILED: {e}")
        times["dask_gpu"] = float("inf")
    print()

    # ── Method 5: Dask distributed ────────────────────────────────────────
    print("--- 5. Dask distributed LocalCluster (4 workers x 4 threads) ---")
    t_wall_start = time.perf_counter()
    bf5, adf5, t_setup, n_tasks, t_compute = method_dask_distributed(
        sig, bf_mask, adf_mask, n_workers=4, threads_per_worker=4
    )
    t_wall = time.perf_counter() - t_wall_start
    times["dask_dist"] = t_wall
    print(f"  Cluster setup: {t_setup:.2f}s")
    print(f"  Graph tasks:   {n_tasks}")
    print(f"  Compute:       {t_compute:.2f}s")
    print(f"  Total:         {t_wall:.2f}s  {gb/t_wall*1000:.0f} MB/s  {ref/t_wall:.2f}x")
    print(f"  BF mean={bf5.mean():.1f}  match numpy: {np.allclose(bf, bf5, rtol=1e-3)}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 65)
    print("SUMMARY  (ref = raw numpy)")
    print(f"  {'Method':<35s} {'Time':>7s}  {'MB/s':>7s}  {'Speedup':>7s}")
    print("  " + "-" * 60)
    labels = {
        "numpy":       "1. Raw numpy sequential",
        "dask_threads":"2. Dask threads (spyde)",
        "gpu_seq":     "3. GPU sequential",
        "dask_gpu":    "4. Dask+GPU",
        "dask_dist":   "5. Dask distributed",
    }
    for key, label in labels.items():
        t = times[key]
        if t == float("inf"):
            print(f"  {label:<35s} {'FAILED':>7s}")
        else:
            print(f"  {label:<35s} {t:7.2f}s  {gb/t*1000:7.0f}  {ref/t:7.2f}x")
    print("=" * 65)
    print()

    print("KEY OBSERVATIONS:")
    t_np = times["numpy"]
    t_dt = times["dask_threads"]
    t_gs = times["gpu_seq"]
    print(f"  Disk read ceiling (numpy):   {gb/t_np*1000:.0f} MB/s  -- single-thread open+readinto")
    print(f"  Dask threads parallelises reads: {t_np/t_dt:.1f}x speedup over single-thread")
    if times["dask_gpu"] != float("inf"):
        t_dg = times["dask_gpu"]
        print(f"  Dask+GPU vs Dask threads:    {t_dt/t_dg:.2f}x  (GPU compute saves "
              f"{(t_dt-t_dg):.1f}s)")
    print(f"  GPU sequential vs Dask threads: {t_dt/t_gs:.2f}x")
    dd = times["dask_dist"]
    print(f"  Distributed vs threads:      {t_dt/dd:.2f}x  (includes {method_dask_distributed.__doc__.split('.')[-1]} setup)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=r"D:/data/sample.mrc",
    )
    args = parser.parse_args()
    run(args.path)

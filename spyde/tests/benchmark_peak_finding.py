"""
benchmark_peak_finding.py
=========================
Benchmark diffraction vector finding on a real 4D-STEM MRC file.

Uses HyperSpy / rosettasciio for loading (new read_binary_distributed path).
Compares:
  1. CPU serial          -- dask reads one chunk, CPU peak-finds sequentially
  2. CPU dask threads    -- dask reads + peak-finds in parallel thread pool
  3. GPU per-chunk       -- dask reads chunk, GPU does NXCORR, CPU does NMS
  4. GPU pipelined       -- blur overlaps with GPU compute via dask graph

Run with MSVC on PATH so the CUDA kernel can compile:
    .venv/Scripts/python spyde/tests/benchmark_peak_finding.py
    .venv/Scripts/python spyde/tests/benchmark_peak_finding.py --path D:/data/file.mrc
"""

from __future__ import annotations

import argparse
import time

import dask.array as da
import hyperspy.api as hs
import numpy as np
import torch
import torch.nn.functional as F
import setuptools  # must import before torch cpp_extension on Python 3.12


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_signal(path):
    t0 = time.perf_counter()
    sig = hs.load(path, lazy=True)
    t_load = time.perf_counter() - t0
    return sig, t_load


def make_cpu_tools(kr, ky, kx, thresh, min_d):
    from scipy.fft import next_fast_len
    from spyde.actions.find_vectors import _find_vectors_single_frame, _get_disk_fft, _make_disk
    pH = next_fast_len(ky + 2 * kr)
    pW = next_fast_len(kx + 2 * kr)
    disk_fft = _get_disk_fft(kr, pH, pW)
    disk = _make_disk(kr)
    nd = disk.size
    tm = float(disk.mean())
    ts = float(np.sqrt(np.sum((disk - tm) ** 2) / nd))
    disk_stats = (nd, tm, ts)

    def find_cpu(frame):
        from spyde.actions.find_vectors import _find_vectors_single_frame
        _, _, p = _find_vectors_single_frame(
            frame, kr, thresh, min_d,
            subpixel=False, _disk_fft=disk_fft, _disk_stats=disk_stats,
        )
        return p

    return find_cpu


def make_gpu_blur(sigma, ky, kx, device="cuda"):
    """
    Build GPU nav-blur function using separable depthwise conv2d.

    Groups=KY*KX depthwise conv avoids the conv1d batch-size limit (~32k)
    on Pascal GPUs.  Reflect-pad via index_select so it matches the
    manual reflect+convolve approach used on CPU.
    """
    import math
    dev = torch.device(device)
    r = int(math.ceil(3 * sigma))
    k = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma) ** 2); k /= k.sum()
    C = ky * kx
    kH = k.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous().to(dev)
    kW = k.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous().to(dev)

    def _safe_reflect_idx(n, r_pad, device):
        """Reflect-pad indices that handle edge chunks where n <= r_pad."""
        if n == 1:
            # Can't reflect a single element -- replicate
            top = torch.zeros(r_pad, dtype=torch.long)
            bot = torch.zeros(r_pad, dtype=torch.long)
        elif n <= r_pad:
            # Wrap-around: clamp to valid range
            top_vals = [min(i, n - 1) for i in range(r_pad, 0, -1)]
            bot_vals = [min(n - 2 + (r_pad - i), n - 1) for i in range(r_pad)]
            top = torch.tensor(top_vals, dtype=torch.long)
            bot = torch.tensor(bot_vals, dtype=torch.long)
        else:
            top = torch.arange(r_pad, 0, -1)
            bot = torch.arange(n - 2, max(n - 2 - r_pad, -1), -1)
        return torch.cat([top, torch.arange(n), bot]).to(device)

    def blur_chunk(ct):
        """ct: (cy, cx, KY, KX) GPU tensor -> blurred (cy, cx, KY, KX) GPU tensor."""
        cy, cx = ct.shape[0], ct.shape[1]
        x = ct.permute(2, 3, 0, 1).reshape(1, C, cy, cx)
        iy = _safe_reflect_idx(cy, r, dev)
        ix = _safe_reflect_idx(cx, r, dev)
        xh  = F.conv2d(x[:, :, iy, :], kH, groups=C, padding=0)
        out = F.conv2d(xh[:, :, :, ix], kW, groups=C, padding=0)
        return out.reshape(ky, kx, cy, cx).permute(2, 3, 0, 1).contiguous()

    return blur_chunk, r


def make_gpu_tools(kr, kr_win, ky, kx, thresh, min_d):
    from spyde.actions.find_vectors_pipeline import _get_cuda_module, _make_disk_np, _greedy_nms_batch
    mod = _get_cuda_module()
    dev = torch.device("cuda")

    disk_np = _make_disk_np(kr)
    nd = disk_np.size
    tm = float(disk_np.mean())
    ts = float(np.sqrt(np.sum((disk_np - tm) ** 2) / nd))
    mod.upload_disk(
        torch.from_numpy(disk_np.reshape(2 * kr + 1, 2 * kr + 1)), tm, ts
    )

    def find_gpu_chunk(flat_np):
        """(N, KY, KX) float32 numpy -> list of (n_peaks, 3) arrays."""
        frames_t = torch.from_numpy(flat_np).to(dev)
        frames_pad = F.pad(
            frames_t.unsqueeze(1), (kr_win,) * 4, mode="reflect"
        ).squeeze(1)
        raw_corr, peak_mask = mod.nxcorr_forward(
            frames_pad, ky, kx, kr, kr_win, thresh, min_d
        )
        torch.cuda.synchronize()
        corr_np = raw_corr.cpu().numpy()
        mask_np = peak_mask.cpu().numpy()
        return _greedy_nms_batch(corr_np, mask_np, min_d)

    return find_gpu_chunk


# ---------------------------------------------------------------------------
# Benchmark methods
# ---------------------------------------------------------------------------

def method_cpu_serial(sig, kr, thresh, min_d, sigma, n_chunks_limit=None):
    """
    Process each dask chunk sequentially on CPU.
    Measures: read time + blur time + peak-find time separately.
    """
    from scipy.ndimage import gaussian_filter

    find_cpu = make_cpu_tools(kr, *sig.axes_manager.signal_shape[::-1], thresh, min_d)
    arr = sig.data
    chunks_y = arr.chunks[0]
    chunks_x = arr.chunks[1]

    n_total = 0
    t_read = t_blur = t_peaks = 0.0
    chunk_count = 0

    y0 = 0
    for cy in chunks_y:
        x0 = 0
        for cx in chunks_x:
            if n_chunks_limit and chunk_count >= n_chunks_limit:
                break

            t0 = time.perf_counter()
            chunk = arr[y0:y0+cy, x0:x0+cx].compute(scheduler="synchronous")
            t_read += time.perf_counter() - t0

            t0 = time.perf_counter()
            blurred = gaussian_filter(chunk.astype(np.float32), sigma=(sigma, sigma, 0, 0))
            t_blur += time.perf_counter() - t0

            t0 = time.perf_counter()
            flat = blurred.reshape(-1, *blurred.shape[2:])
            for frame in flat:
                p = find_cpu(frame)
                n_total += len(p)
            t_peaks += time.perf_counter() - t0

            chunk_count += 1
            x0 += cx
        y0 += cy
        if n_chunks_limit and chunk_count >= n_chunks_limit:
            break

    return n_total, t_read, t_blur, t_peaks, chunk_count


def method_cpu_dask_threads(sig, kr, thresh, min_d, sigma, n_chunks_limit=None):
    """
    Dask threads -- one task per dask chunk (read + blur + peak-find per chunk).

    n_workers is capped to n_chunks so threads don't compete for GIL on
    a smaller task graph than the thread pool size.
    """
    KY, KX = sig.axes_manager.signal_shape[::-1]

    def _process_chunk(chunk):
        """chunk: (cy, cx, KY, KX) -> (cy, cx) int32 peak counts"""
        from scipy.ndimage import gaussian_filter as gf
        from spyde.actions.find_vectors import _find_vectors_single_frame, _get_disk_fft, _make_disk
        from scipy.fft import next_fast_len
        b = gf(chunk.astype(np.float32), sigma=(sigma, sigma, 0, 0))
        cy, cx = chunk.shape[:2]
        counts = np.zeros((cy, cx), dtype=np.int32)
        kry, kx_ = b.shape[2], b.shape[3]
        pH = next_fast_len(kry + 2 * kr)
        disk_fft = _get_disk_fft(kr, pH, pH)
        disk = _make_disk(kr)
        nd = disk.size; tm = float(disk.mean()); ts = float(np.sqrt(np.sum((disk-tm)**2)/nd))
        for iy in range(cy):
            for ix in range(cx):
                _, _, p = _find_vectors_single_frame(
                    b[iy, ix], kr, thresh, min_d,
                    subpixel=False, _disk_fft=disk_fft, _disk_stats=(nd, tm, ts),
                )
                counts[iy, ix] = len(p)
        return counts

    arr = sig.data
    if n_chunks_limit is not None:
        n_y = min(n_chunks_limit, len(arr.chunks[0]))
        y_end = sum(arr.chunks[0][:n_y])
        arr = arr[:y_end]

    n_chunks = np.prod([len(c) for c in arr.chunks])

    t0 = time.perf_counter()
    count_map = da.map_blocks(_process_chunk, arr, dtype=np.int32, drop_axis=(2, 3))
    t_graph = time.perf_counter() - t0
    n_tasks = len(dict(count_map.__dask_graph__()))

    import dask
    t0 = time.perf_counter()
    # Cap workers to n_chunks -- more threads just thrash for small graphs
    with dask.config.set(scheduler="threads", num_workers=min(n_chunks, 16)):
        result = count_map.compute()
    t_compute = time.perf_counter() - t0

    return result.sum(), t_graph, n_tasks, t_compute


def method_gpu_dask(sig, kr, thresh, min_d, sigma, n_chunks_limit=None):
    """
    Dask reads chunks (threads scheduler), GPU does NXCORR per chunk.

    Each map_blocks task:
      1. Blur chunk on CPU (sigma over nav dims)
      2. H2D transfer to GPU
      3. NXCORR kernel (xcorr + window stats + normalise + local max)
      4. D2H peak mask (sparse)
      5. CPU greedy NMS

    The GPU serialisation lock ensures only one CUDA launch at a time
    (CUDA context is not thread-safe for concurrent launches from multiple
    Python threads without explicit stream management).
    """
    import threading
    from scipy.ndimage import gaussian_filter as _gf
    from spyde.actions.find_vectors_pipeline import _get_cuda_module, _make_disk_np, _greedy_nms_batch

    KY, KX = sig.axes_manager.signal_shape[::-1]
    kr_win = kr + 1
    mod = _get_cuda_module()
    dev = torch.device("cuda")

    disk_np = _make_disk_np(kr)
    nd = disk_np.size; tm = float(disk_np.mean()); ts = float(np.sqrt(np.sum((disk_np-tm)**2)/nd))
    mod.upload_disk(torch.from_numpy(disk_np.reshape(2*kr+1, 2*kr+1)), tm, ts)

    gpu_lock = threading.Lock()

    def _process_chunk_gpu(chunk):
        """chunk: (cy, cx, KY, KX) -> (cy, cx) int32 peak counts"""
        from scipy.ndimage import gaussian_filter as gf
        from spyde.actions.find_vectors_pipeline import _greedy_nms_batch
        b = gf(chunk.astype(np.float32), sigma=(sigma, sigma, 0, 0))
        cy, cx = chunk.shape[:2]
        flat = b.reshape(-1, KY, KX)

        with gpu_lock:
            frames_t = torch.from_numpy(flat).to(dev)
            frames_pad = F.pad(
                frames_t.unsqueeze(1), (kr_win,) * 4, mode="reflect"
            ).squeeze(1)
            raw_corr, peak_mask = mod.nxcorr_forward(
                frames_pad, KY, KX, kr, kr_win, thresh, min_d
            )
            torch.cuda.synchronize()
            corr_np = raw_corr.cpu().numpy()
            mask_np = peak_mask.cpu().numpy()

        peaks_list = _greedy_nms_batch(corr_np, mask_np, min_d)
        counts = np.array([len(p) for p in peaks_list], dtype=np.int32).reshape(cy, cx)
        return counts

    arr = sig.data
    if n_chunks_limit is not None:
        n_y = min(n_chunks_limit, len(arr.chunks[0]))
        y_end = sum(arr.chunks[0][:n_y])
        arr = arr[:y_end]

    t0 = time.perf_counter()
    count_map = da.map_blocks(
        _process_chunk_gpu,
        arr,
        dtype=np.int32,
        drop_axis=(2, 3),
    )
    t_graph = time.perf_counter() - t0
    n_tasks = len(dict(count_map.__dask_graph__()))

    import dask
    n_chunks = np.prod([len(c) for c in arr.chunks])
    t0 = time.perf_counter()
    # GPU launches are serialised by gpu_lock, so more threads than chunks
    # just add overhead; cap at n_chunks
    with dask.config.set(scheduler="threads", num_workers=min(n_chunks, 16)):
        result = count_map.compute()
    t_compute = time.perf_counter() - t0

    return result.sum(), t_graph, n_tasks, t_compute


def method_gpu_blur_dask(sig, kr, thresh, min_d, sigma, n_chunks_limit=None):
    """
    Full-GPU pipeline: dask reads chunk -> H2D -> GPU blur -> GPU NXCORR -> CPU NMS.

    The GPU blur uses depthwise separable conv2d over the nav dimensions,
    which is 48x faster than CPU gaussian_filter and removes the blur from the
    critical path.  With blur on GPU, the per-chunk breakdown becomes:
        read ~0.19s  |  GPU (H2D + blur + NXCORR) ~0.18s
    The bottleneck flips to disk I/O.

    With multiple dask threads, read for chunk N+1 overlaps with GPU
    processing chunk N, giving near-disk-speed throughput.
    """
    import threading
    from spyde.actions.find_vectors_pipeline import _get_cuda_module, _make_disk_np, _greedy_nms_batch

    KY, KX = sig.axes_manager.signal_shape[::-1]
    kr_win = kr + 1
    mod = _get_cuda_module()
    dev = torch.device("cuda")

    disk_np = _make_disk_np(kr)
    nd = disk_np.size; tm = float(disk_np.mean())
    ts = float(np.sqrt(np.sum((disk_np - tm) ** 2) / nd))
    mod.upload_disk(torch.from_numpy(disk_np.reshape(2 * kr + 1, 2 * kr + 1)), tm, ts)

    blur_fn, _ = make_gpu_blur(sigma, KY, KX, device="cuda")
    gpu_lock = threading.Lock()

    def _process_chunk_gpu_blur(chunk):
        from spyde.actions.find_vectors_pipeline import _greedy_nms_batch
        cy, cx = chunk.shape[:2]
        flat = chunk.astype(np.float32)
        with gpu_lock:
            ct = torch.from_numpy(flat).to(dev)        # H2D
            blurred = blur_fn(ct)                       # GPU blur
            flat_t = blurred.reshape(-1, KY, KX)
            fp = F.pad(flat_t.unsqueeze(1), (kr_win,) * 4, mode='reflect').squeeze(1)
            rc, pm = mod.nxcorr_forward(fp, KY, KX, kr, kr_win, thresh, min_d)
            torch.cuda.synchronize()
            corr_np = rc.cpu().numpy()
            mask_np = pm.cpu().numpy()
        peaks_list = _greedy_nms_batch(corr_np, mask_np, min_d)
        return np.array([len(p) for p in peaks_list], dtype=np.int32).reshape(cy, cx)

    arr = sig.data
    if n_chunks_limit is not None:
        n_y = min(n_chunks_limit, len(arr.chunks[0]))
        y_end = sum(arr.chunks[0][:n_y])
        arr = arr[:y_end]

    t0 = time.perf_counter()
    count_map = da.map_blocks(_process_chunk_gpu_blur, arr, dtype=np.int32, drop_axis=(2, 3))
    t_graph = time.perf_counter() - t0
    n_tasks = len(dict(count_map.__dask_graph__()))

    import dask
    n_chunks = np.prod([len(c) for c in arr.chunks])
    t0 = time.perf_counter()
    with dask.config.set(scheduler="threads", num_workers=min(n_chunks, 16)):
        result = count_map.compute()
    t_compute = time.perf_counter() - t0

    return result.sum(), t_graph, n_tasks, t_compute


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(path, kr=14, thresh=0.2, min_d=28, sigma=1.5, n_chunks=8):
    print()
    print("=" * 65)
    print("PEAK FINDING BENCHMARK")
    print(f"  File:     {path}")
    print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    print(f"  kr={kr}  threshold={thresh}  min_dist={min_d}  sigma={sigma}")
    print(f"  Chunks to process: {n_chunks} (of 576 total)")
    print("=" * 65)
    print()

    sig, t_load = load_signal(path)
    KY, KX = sig.axes_manager.signal_shape[::-1]
    n_nav = sig.axes_manager.navigation_shape
    total_chunks = np.prod([len(c) for c in sig.data.chunks])
    chunk_pats = sig.data.chunks[0][0] * sig.data.chunks[1][0]
    chunk_mb = chunk_pats * KY * KX * 4 / 1e6
    gb = np.prod(sig.data.shape) * 4 / 1e9

    print(f"Signal: {sig.data.shape}  {gb:.1f} GB")
    print(f"Chunks: {[(len(c), c[0]) for c in sig.data.chunks]}")
    print(f"  {total_chunks} chunks x {chunk_mb:.0f} MB = {chunk_pats} pats/chunk")
    print(f"  Graph build: {t_load*1000:.0f} ms")
    print()

    # Per-chunk breakdown
    print("--- Per-chunk timing breakdown ---")
    from scipy.ndimage import gaussian_filter
    t0 = time.perf_counter()
    chunk_data = sig.data[:11, :11].compute(scheduler="synchronous")
    t_read_1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    blurred_1 = gaussian_filter(chunk_data.astype(np.float32), sigma=(sigma, sigma, 0, 0))
    t_blur_1 = time.perf_counter() - t0

    find_cpu = make_cpu_tools(kr, KY, KX, thresh, min_d)
    flat_1 = blurred_1.reshape(-1, KY, KX)
    t0 = time.perf_counter()
    n_pk = sum(len(find_cpu(f)) for f in flat_1)
    t_cpu_1 = time.perf_counter() - t0

    find_gpu = make_gpu_tools(kr, kr + 1, KY, KX, thresh, min_d)
    # warmup
    find_gpu(flat_1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    peaks_gpu = find_gpu(flat_1)
    t_gpu_1 = time.perf_counter() - t0
    n_pk_gpu = sum(len(p) for p in peaks_gpu)

    # GPU blur timing
    blur_fn, _ = make_gpu_blur(sigma, KY, KX)
    chunk_t_gpu = torch.from_numpy(chunk_data.astype(np.float32)).to("cuda")
    blur_fn(chunk_t_gpu); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        blur_fn(chunk_t_gpu); torch.cuda.synchronize()
    t_gpu_blur_1 = (time.perf_counter() - t0) / 10

    find_gpu = make_gpu_tools(kr, kr + 1, KY, KX, thresh, min_d)
    blurred_gpu = blur_fn(chunk_t_gpu).reshape(-1, KY, KX).cpu().numpy()
    find_gpu(blurred_gpu); torch.cuda.synchronize()
    t0 = time.perf_counter()
    peaks_gpu = find_gpu(blurred_gpu); torch.cuda.synchronize()
    t_gpu_1 = time.perf_counter() - t0
    n_pk_gpu = sum(len(p) for p in peaks_gpu)

    print(f"  Read:              {t_read_1:.3f}s   ({chunk_mb/t_read_1:.0f} MB/s)")
    print(f"  Blur (CPU):        {t_blur_1:.3f}s")
    print(f"  Blur (GPU):        {t_gpu_blur_1*1000:.0f} ms  ({t_blur_1/t_gpu_blur_1:.0f}x faster)")
    print(f"  CPU peak-find:     {t_cpu_1:.3f}s   ({t_cpu_1/chunk_pats*1000:.2f} ms/pat)  {n_pk} peaks")
    print(f"  GPU peak-find:     {t_gpu_1:.3f}s   ({t_gpu_1/chunk_pats*1000:.2f} ms/pat)  {n_pk_gpu} peaks")
    print(f"  GPU full pipeline: {t_gpu_blur_1 + t_gpu_1:.3f}s  (blur+NXCORR, excludes H2D)")
    print(f"  Bottleneck (CPU):  {'READ' if t_read_1>max(t_blur_1,t_cpu_1) else 'BLUR' if t_blur_1>t_cpu_1 else 'PEAKS'}")
    print(f"  Bottleneck (GPU):  {'READ' if t_read_1>(t_gpu_blur_1+t_gpu_1) else 'GPU compute'}")
    print()

    times = {}

    # Method 1: CPU serial
    print(f"--- 1. CPU serial ({n_chunks} chunks) ---")
    t_wall = time.perf_counter()
    n_pk, t_r, t_bl, t_pk, nc = method_cpu_serial(sig, kr, thresh, min_d, sigma, n_chunks)
    t_wall = time.perf_counter() - t_wall
    times["cpu_serial"] = t_wall / nc
    pats = nc * chunk_pats
    print(f"  {nc} chunks, {pats} patterns, {n_pk} peaks")
    print(f"  Read: {t_r:.2f}s  Blur: {t_bl:.2f}s  Peaks: {t_pk:.2f}s")
    print(f"  Total: {t_wall:.2f}s  ({t_wall/nc:.2f}s/chunk)")
    print(f"  Extrapolated full: {times['cpu_serial']*total_chunks/60:.1f} min")
    print()

    # Method 2: CPU dask threads
    print(f"--- 2. CPU dask threads ({n_chunks} chunks) ---")
    t_wall = time.perf_counter()
    n_pk2, t_graph, n_tasks, t_compute = method_cpu_dask_threads(
        sig, kr, thresh, min_d, sigma, n_chunks
    )
    t_wall = time.perf_counter() - t_wall
    times["cpu_dask"] = t_wall / n_chunks
    print(f"  Graph: {t_graph*1000:.1f} ms  tasks={n_tasks}")
    print(f"  Compute: {t_compute:.2f}s  Total: {t_wall:.2f}s  ({t_wall/n_chunks:.2f}s/chunk)")
    print(f"  {n_pk2} peaks  Extrapolated: {times['cpu_dask']*total_chunks/60:.1f} min")
    print(f"  Speedup vs serial: {times['cpu_serial']/times['cpu_dask']:.2f}x")
    print()

    # Method 3: GPU + dask (CPU blur)
    print(f"--- 3. GPU NXCORR + CPU blur + dask ({n_chunks} chunks) ---")
    t_wall = time.perf_counter()
    n_pk3, t_graph, n_tasks, t_compute = method_gpu_dask(
        sig, kr, thresh, min_d, sigma, n_chunks
    )
    t_wall = time.perf_counter() - t_wall
    times["gpu_dask"] = t_wall / n_chunks
    print(f"  Graph: {t_graph*1000:.1f} ms  tasks={n_tasks}")
    print(f"  Compute: {t_compute:.2f}s  Total: {t_wall:.2f}s  ({t_wall/n_chunks:.2f}s/chunk)")
    print(f"  {n_pk3} peaks  Extrapolated: {times['gpu_dask']*total_chunks/60:.1f} min")
    print(f"  Speedup vs serial: {times['cpu_serial']/times['gpu_dask']:.2f}x  vs CPU dask: {times['cpu_dask']/times['gpu_dask']:.2f}x")
    print()

    # Method 4: GPU + GPU blur + dask
    print(f"--- 4. GPU blur + GPU NXCORR + dask ({n_chunks} chunks) ---")
    t_wall = time.perf_counter()
    n_pk4, t_graph, n_tasks, t_compute = method_gpu_blur_dask(
        sig, kr, thresh, min_d, sigma, n_chunks
    )
    t_wall = time.perf_counter() - t_wall
    times["gpu_blur_dask"] = t_wall / n_chunks
    print(f"  Graph: {t_graph*1000:.1f} ms  tasks={n_tasks}")
    print(f"  Compute: {t_compute:.2f}s  Total: {t_wall:.2f}s  ({t_wall/n_chunks:.2f}s/chunk)")
    print(f"  {n_pk4} peaks  Extrapolated: {times['gpu_blur_dask']*total_chunks/60:.1f} min")
    print(f"  Speedup vs serial: {times['cpu_serial']/times['gpu_blur_dask']:.2f}x")
    print(f"  Speedup vs GPU (CPU blur): {times['gpu_dask']/times['gpu_blur_dask']:.2f}x")
    print()

    # Summary
    print("=" * 65)
    print("SUMMARY")
    print(f"  {'Method':<38} {'s/chunk':>7}  {'Full est':>8}  {'Speedup':>7}")
    print("  " + "-" * 65)
    for key, label in [
        ("cpu_serial",    "1. CPU serial"),
        ("cpu_dask",      "2. CPU dask threads"),
        ("gpu_dask",      "3. GPU NXCORR + CPU blur + dask"),
        ("gpu_blur_dask", "4. GPU blur + GPU NXCORR + dask"),
    ]:
        t = times[key]
        ext = t * total_chunks / 60
        spd = times["cpu_serial"] / t
        print(f"  {label:<38} {t:7.2f}s  {ext:7.1f} min  {spd:7.2f}x")
    print("=" * 65)
    print()
    print("Per-chunk bottleneck analysis:")
    print(f"  read={t_read_1:.2f}s  cpu_blur={t_blur_1:.2f}s  gpu_blur={t_gpu_blur_1*1000:.0f}ms")
    print(f"  cpu_peaks={t_cpu_1:.2f}s  gpu_peaks={t_gpu_1:.2f}s")
    print(f"  GPU full pipeline: {t_gpu_blur_1 + t_gpu_1:.2f}s  (vs read: {t_read_1:.2f}s)")
    if t_read_1 > t_gpu_blur_1 + t_gpu_1:
        print("  => READ-LIMITED: GPU is fast enough, disk is the bottleneck")
    else:
        print("  => GPU-LIMITED: GPU compute slower than disk")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=r"D:/data/sample.mrc",
    )
    parser.add_argument("--kr",     type=int,   default=14)
    parser.add_argument("--thresh", type=float, default=0.2)
    parser.add_argument("--min_d",  type=int,   default=28)
    parser.add_argument("--sigma",  type=float, default=1.5)
    parser.add_argument("--chunks", type=int,   default=8,
                        help="Number of dask chunks to process (default 8)")
    args = parser.parse_args()
    run(args.path, args.kr, args.thresh, args.min_d, args.sigma, args.chunks)

"""
find_vectors_pipeline.py -- Pipelined blur + GPU peak-finding for large 4D-STEM datasets.

Key insight: MRC files store 4D-STEM as a flat (N_patterns, KY, KX) sequence.
Dask's 4D chunking is correct for lazy signal operations but catastrophic for
sequential reads -- it fragments what is a 1 MB/pattern sequential layout into
76x76 pixel signal tiles, causing 49x read amplification.

This pipeline bypasses dask for I/O and reads the file as a flat memmap
(N, KY, KX), achieving 800+ MB/s vs ~200 MB/s through dask.

Architecture: two threads, one bounded queue
--------------------------------------------
Thread A (reader/blurrer):
    memmap the file, iterate nav chunks of ~nav_chunk_size x nav_chunk_size patterns
    For each nav chunk:
      1. np.array(mm[slice]) -- sequential read at disk speed
      2. scipy gaussian_filter with reflect ghost zones (nav dims only)
      3. Trim ghost zones, split into GPU sub-batches
      4. Push pinned numpy arrays onto blur_queue (bounded to 2 -- back-pressure)

Thread B (GPU consumer):
    For each sub-batch on blur_queue:
      1. async H2D (pinned -> GPU)
      2. CUDA kernel: fused xcorr + window stats + NXCORR normalise
      3. Local-max kernel: enforce min_distance
      4. D2H peak mask (sparse)
      5. CPU greedy NMS
      6. Write to output list

The queue bound ensures the reader stays at most one chunk ahead of the GPU,
capping RAM usage at ~2 * nav_chunk * KY * KX * 4 bytes.
"""

from __future__ import annotations

import threading
import queue
import time
import os
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_disk_np(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (y ** 2 + x ** 2 <= r ** 2).astype(np.float32)
    disk /= disk.sum()
    return disk


def _greedy_nms_batch(
    corr_np: np.ndarray,  # (B, KY, KX)
    mask_np: np.ndarray,  # (B, KY, KX) bool
    min_d: int,
) -> list[np.ndarray]:
    out = []
    min_d2 = min_d * min_d
    for i in range(corr_np.shape[0]):
        yx = np.argwhere(mask_np[i])
        if len(yx) == 0:
            out.append(np.zeros((0, 3), dtype=np.float32))
            continue
        scores = corr_np[i, yx[:, 0], yx[:, 1]]
        if len(yx) > 1:
            order = np.argsort(-scores)
            yx = yx[order]; scores = scores[order]
            kept = np.ones(len(yx), dtype=bool)
            for j in range(len(yx)):
                if not kept[j]:
                    continue
                dy = yx[j + 1:, 0] - yx[j, 0]
                dx = yx[j + 1:, 1] - yx[j, 1]
                kept[j + 1:][(dy * dy + dx * dx) <= min_d2] = False
            yx = yx[kept]; scores = scores[kept]
        out.append(np.column_stack([yx.astype(np.float32), scores.astype(np.float32)]))
    return out


# ---------------------------------------------------------------------------
# CUDA kernel
# ---------------------------------------------------------------------------

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define MAX_DISK_ELEMS 2500
__constant__ float c_disk[MAX_DISK_ELEMS];
__constant__ int   c_disk_kH;
__constant__ int   c_disk_kW;
__constant__ int   c_n_disk;
__constant__ float c_t_mean;
__constant__ float c_t_std;

void upload_disk(const float* disk_host, int kH, int kW, float t_mean, float t_std) {
    int n = kH * kW;
    cudaMemcpyToSymbol(c_disk,    disk_host, n * sizeof(float));
    cudaMemcpyToSymbol(c_disk_kH, &kH,       sizeof(int));
    cudaMemcpyToSymbol(c_disk_kW, &kW,       sizeof(int));
    cudaMemcpyToSymbol(c_n_disk,  &n,        sizeof(int));
    cudaMemcpyToSymbol(c_t_mean,  &t_mean,   sizeof(float));
    cudaMemcpyToSymbol(c_t_std,   &t_std,    sizeof(float));
}

#define TILE 32

__global__ void nxcorr_kernel(
    const float* __restrict__ frames,    // (N, PH, PW) reflect-padded by kr_win
    float*       __restrict__ raw_corr,  // (N, H, W)
    bool*        __restrict__ peak_mask, // (N, H, W)
    int N, int H, int W,
    int kr, int kr_win,
    float inv_n_disk, float inv_n_win,
    float threshold
) {
    const int PH = H + 2 * kr_win;
    const int PW = W + 2 * kr_win;
    const int n      = blockIdx.z;
    const int ty     = threadIdx.y;
    const int tx     = threadIdx.x;
    const int out_y0 = blockIdx.y * TILE;
    const int out_x0 = blockIdx.x * TILE;
    const int halo   = kr_win;
    const int TPH    = TILE + 2 * halo;
    const int TPW    = TPH;

    extern __shared__ float smem[];

    // Cooperative tile load
    const int n_smem    = TPH * TPW;
    const int n_threads = TILE * TILE;
    const int fbase     = n * PH * PW;
    for (int idx = ty * TILE + tx; idx < n_smem; idx += n_threads) {
        int sr = idx / TPW, sc = idx % TPW;
        int pr = out_y0 + sr, pc = out_x0 + sc;
        smem[idx] = (pr < PH && pc < PW) ? frames[fbase + pr * PW + pc] : 0.f;
    }
    __syncthreads();

    const int out_y = out_y0 + ty;
    const int out_x = out_x0 + tx;
    if (out_y >= H || out_x >= W) return;

    // xcorr: disk centre at smem (ty+halo, tx+halo)
    float xcorr_val = 0.f;
    for (int dr = 0; dr < c_disk_kH; dr++)
        for (int dc = 0; dc < c_disk_kW; dc++)
            xcorr_val += c_disk[dr * c_disk_kW + dc]
                       * smem[(ty + halo + dr - kr) * TPW + (tx + halo + dc - kr)];

    // window stats: (2*kr_win+1)^2 box, top-left at smem (ty, tx)
    float sum1 = 0.f, sum2 = 0.f;
    const int kwin = 2 * kr_win + 1;
    for (int dr = 0; dr < kwin; dr++)
        for (int dc = 0; dc < kwin; dc++) {
            float v = smem[(ty + dr) * TPW + (tx + dc)];
            sum1 += v; sum2 += v * v;
        }

    float wm  = sum1 * inv_n_win;
    float wv  = fmaxf(sum2 * inv_n_win - wm * wm, 0.f);
    float ws  = sqrtf(wv);
    float den = ws * c_t_std;
    float num = xcorr_val * inv_n_disk - wm * c_t_mean;
    float sc2 = (den >= 1e-8f) ? fmaxf(-1.f, fminf(1.f, num / den)) : 0.f;

    int oidx       = n * H * W + out_y * W + out_x;
    raw_corr[oidx] = sc2;
    peak_mask[oidx] = (sc2 >= threshold);
}

__global__ void local_max_kernel(
    const float* __restrict__ raw_corr,
    bool*        __restrict__ peak_mask,
    int N, int H, int W, int min_d, float threshold
) {
    const int n     = blockIdx.z;
    const int out_y = blockIdx.y * TILE + threadIdx.y;
    const int out_x = blockIdx.x * TILE + threadIdx.x;
    if (out_y >= H || out_x >= W) return;
    int idx = n * H * W + out_y * W + out_x;
    if (!peak_mask[idx]) return;
    float center = raw_corr[idx];
    for (int ny = max(0, out_y - min_d); ny <= min(H-1, out_y + min_d); ny++)
        for (int nx = max(0, out_x - min_d); nx <= min(W-1, out_x + min_d); nx++)
            if (raw_corr[n * H * W + ny * W + nx] > center) {
                peak_mask[idx] = false; return;
            }
}

void upload_disk_py(torch::Tensor disk_cpu, float t_mean, float t_std) {
    int kH = disk_cpu.size(0), kW = disk_cpu.size(1);
    TORCH_CHECK(kH * kW <= MAX_DISK_ELEMS, "Disk too large for constant memory");
    upload_disk(disk_cpu.data_ptr<float>(), kH, kW, t_mean, t_std);
}

std::vector<torch::Tensor> nxcorr_forward(
    torch::Tensor frames_padded, int H, int W,
    int kr, int kr_win, float threshold, int min_d
) {
    int N = frames_padded.size(0);
    float inv_n_disk = 1.f / float((2*kr+1)*(2*kr+1));
    float inv_n_win  = 1.f / float((2*kr_win+1)*(2*kr_win+1));
    auto opts      = frames_padded.options();
    auto raw_corr  = torch::empty({N, H, W}, opts);
    auto peak_mask = torch::zeros({N, H, W}, opts.dtype(torch::kBool));
    int halo = kr_win, TPH = TILE + 2*halo, TPW = TPH;
    int shmem = TPH * TPW * sizeof(float);
    dim3 block(TILE, TILE, 1);
    dim3 grid((W+TILE-1)/TILE, (H+TILE-1)/TILE, N);
    nxcorr_kernel<<<grid, block, shmem>>>(
        frames_padded.data_ptr<float>(), raw_corr.data_ptr<float>(),
        peak_mask.data_ptr<bool>(), N, H, W, kr, kr_win,
        inv_n_disk, inv_n_win, threshold);
    local_max_kernel<<<grid, block>>>(
        raw_corr.data_ptr<float>(), peak_mask.data_ptr<bool>(),
        N, H, W, min_d, threshold);
    return {raw_corr, peak_mask};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upload_disk",    &upload_disk_py);
    m.def("nxcorr_forward", &nxcorr_forward);
}
"""

_CUDA_MODULE = None


def _get_cuda_module():
    global _CUDA_MODULE
    if _CUDA_MODULE is not None:
        return _CUDA_MODULE
    import torch
    from torch.utils.cpp_extension import load_inline
    try:
        import setuptools       # noqa: F401
        import distutils._msvccompiler  # noqa: F401 -- patches distutils attr on Python 3.12
    except ImportError:
        pass
    cap = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")
    print("Compiling pipeline CUDA kernel...", flush=True)
    _CUDA_MODULE = load_inline(
        name="nxcorr_pipeline",
        cpp_sources="",
        cuda_sources=_CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    print("Done.", flush=True)
    return _CUDA_MODULE


# ---------------------------------------------------------------------------
# Open the MRC file as a flat memmap -- the right I/O layer
# ---------------------------------------------------------------------------

def _open_mrc_memmap(path: str) -> tuple[np.ndarray, int, int]:
    """
    Return (mm, KY, KX) where mm is a read-only memmap shaped (N, KY, KX).

    MRC stores 4D-STEM as NZ=N_patterns planes of (NY, NX) = (KY, KX).
    The header is always 1024 bytes (extended header size at byte 92).
    """
    import struct
    with open(path, 'rb') as f:
        hdr = f.read(1024)
    nx, ny, nz = struct.unpack_from('3i', hdr, 0)
    mode = struct.unpack_from('i', hdr, 12)[0]
    ext = struct.unpack_from('i', hdr, 92)[0]
    dtype_map = {0: np.uint8, 1: np.int16, 2: np.float32, 6: np.uint16}
    dtype = dtype_map.get(mode, np.float32)
    offset = 1024 + ext
    mm = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(nz, ny, nx))
    return mm, ny, nx  # KY=ny, KX=nx


# ---------------------------------------------------------------------------
# GPU worker thread
# ---------------------------------------------------------------------------

class _GpuWorker:
    """
    Consumes (flat_indices, pinned_frames) from in_queue, pushes
    (flat_indices, peaks_list) to out_queue.
    Sentinel: in_queue.put(None).
    """

    def __init__(self, kr, kr_win, threshold, min_d, ky, kx, device="cuda"):
        self.kr = kr; self.kr_win = kr_win
        self.threshold = threshold; self.min_d = min_d
        self.ky = ky; self.kx = kx; self.dev = device
        self.in_queue: queue.Queue = queue.Queue(maxsize=2)
        self.out_queue: queue.Queue = queue.Queue()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def push(self, flat_indices, pinned):
        self.in_queue.put((flat_indices, pinned))

    def stop(self):
        self.in_queue.put(None)
        self._t.join()

    def _run(self):
        import torch, torch.nn.functional as F
        mod = _get_cuda_module()
        dev = torch.device(self.dev)
        stream = torch.cuda.Stream(device=self.dev)

        # Upload disk to constant memory once
        disk_np = _make_disk_np(self.kr)
        nd = disk_np.size
        tm = float(disk_np.mean())
        ts = float(np.sqrt(np.sum((disk_np - tm) ** 2) / nd))
        mod.upload_disk(
            torch.from_numpy(disk_np.reshape(2*self.kr+1, 2*self.kr+1)),
            tm, ts,
        )

        while True:
            item = self.in_queue.get()
            if item is None:
                break
            flat_indices, frames_np = item
            with torch.cuda.stream(stream):
                frames_t = torch.from_numpy(frames_np).to(dev, non_blocking=True)
                frames_pad = F.pad(
                    frames_t.unsqueeze(1),
                    (self.kr_win,)*4, mode="reflect",
                ).squeeze(1)
                raw_corr, peak_mask = mod.nxcorr_forward(
                    frames_pad, self.ky, self.kx,
                    self.kr, self.kr_win, self.threshold, self.min_d,
                )
                corr_cpu = raw_corr.cpu().numpy()
                mask_cpu = peak_mask.cpu().numpy()
            stream.synchronize()
            peaks_list = _greedy_nms_batch(corr_cpu, mask_cpu, self.min_d)
            self.out_queue.put((flat_indices, peaks_list))


# ---------------------------------------------------------------------------
# Reader/blurrer thread
# ---------------------------------------------------------------------------

def _reader_thread(
    mm: np.ndarray,       # memmap (N_flat, KY, KX)  -- flat row-major nav order
    n_nav_y: int,
    n_nav_x: int,
    nav_chunk: int,       # number of nav rows per chunk
    depth_px: int,        # ghost zone depth for blur
    sigma: float,
    gpu_batch: int,
    out_queue: queue.Queue,
):
    """
    Reads row-band chunks from the memmap, blurs, and pushes sub-batches.

    Key design: nav is stored row-major, so a band of `nav_chunk` full rows
    is a single contiguous slice mm[y0*n_nav_x : y1*n_nav_x].  This is the
    fastest possible read -- one np.array() call per chunk at full disk speed.
    We never split on x, so x ghost zones are handled by reflect-padding after
    the read rather than requiring extra file I/O.

    Ghost zones in y: read py0..py1 rows (clamped at edges), blur, trim back.
    Ghost zones in x: already covered because we read full rows.
    """
    from scipy.ndimage import gaussian_filter
    import torch

    KY, KX = mm.shape[1], mm.shape[2]
    sigma_tuple = (sigma, sigma, 0.0, 0.0)

    for y0 in range(0, n_nav_y, nav_chunk):
        y1  = min(y0 + nav_chunk, n_nav_y)
        # Ghost zone rows (clamped)
        py0 = max(0, y0 - depth_px)
        py1 = min(n_nav_y, y1 + depth_px)
        gy  = py1 - py0   # number of rows including ghost

        # Single contiguous read: all columns for py0..py1 rows
        flat_s = py0 * n_nav_x
        flat_e = py1 * n_nav_x
        chunk_raw = np.array(mm[flat_s:flat_e], dtype=np.float32)  # (gy*n_nav_x, KY, KX)
        chunk_raw = chunk_raw.reshape(gy, n_nav_x, KY, KX)

        # Nav-space Gaussian blur -- sigma applied over (row, col) nav dims only
        chunk_blurred = gaussian_filter(chunk_raw, sigma=sigma_tuple)

        # Trim y ghost zones; x is full width so no trim needed
        vy0 = y0 - py0
        valid = chunk_blurred[vy0:vy0 + (y1 - y0)]          # (cy, n_nav_x, KY, KX)
        cy = valid.shape[0]

        flat_indices = np.arange(y0 * n_nav_x, y1 * n_nav_x, dtype=np.int64)
        flat_frames  = valid.reshape(-1, KY, KX)             # (cy*n_nav_x, KY, KX)

        for start in range(0, len(flat_frames), gpu_batch):
            end    = min(start + gpu_batch, len(flat_frames))
            batch  = flat_frames[start:end].copy()
            pinned = torch.from_numpy(batch).pin_memory().numpy()
            out_queue.put((flat_indices[start:end], pinned))

    out_queue.put(None)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def find_vectors_pipelined(
    path_or_signal,
    kernel_radius: int,
    threshold: float,
    min_distance: int,
    sigma: float = 1.5,
    kernel_window_pad: int = 1,
    nav_chunk_size: int = 64,
    gpu_batch_patterns: int = 2000,
    device: str = "cuda",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> "SpyDEDiffractionVectors":
    """
    Pipelined MRC read + nav blur + GPU NXCORR peak-finding.

    Parameters
    ----------
    path_or_signal : str or HyperSpy Signal2D
        Path to .mrc file, or a HyperSpy signal whose metadata contains the
        file path. If a signal is passed its axes calibration is used.
    kernel_radius : int
        Diffraction disk radius in pixels.
    threshold : float
        NXCORR threshold (-1 to 1).
    min_distance : int
        Minimum peak separation in pixels.
    sigma : float
        Nav-space Gaussian sigma in scan pixels.
    kernel_window_pad : int
        Extra pixels on the statistics window for robustness.
    nav_chunk_size : int
        Side length of nav chunks in scan pixels.  Each chunk is
        nav_chunk_size^2 patterns.  Tune so the blurred chunk fits in RAM.
        Default 64 -> 64^2 = 4096 patterns = ~4 GB for 512x512 float32.
    gpu_batch_patterns : int
        Patterns per GPU kernel launch.  Tune to fit in VRAM.
        Rule of thumb: leave 2 GB headroom from total VRAM.
        For 12 GB VRAM, 512x512 patterns: ~2000.
    device : str
        CUDA device.
    on_progress : callable(n_done, n_total) | None

    Returns
    -------
    SpyDEDiffractionVectors
    """
    from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors

    # Resolve path and axes info
    if isinstance(path_or_signal, str):
        path = path_or_signal
        mm, KY, KX = _open_mrc_memmap(path)
        N_flat = mm.shape[0]
        # Assume square nav grid
        n_nav = int(round(N_flat ** 0.5))
        assert n_nav * n_nav == N_flat, \
            f"Non-square nav grid ({N_flat} patterns); pass a HyperSpy signal instead"
        n_nav_y = n_nav_x = n_nav
        sig_ax = None
        ky_scale = kx_scale = 1.0
        ky_offset = kx_offset = 0.0
    else:
        sig = path_or_signal
        # Get path from signal metadata or compute
        try:
            path = sig.metadata.General.original_filename
        except Exception:
            path = None

        nav_shape = tuple(sig.axes_manager.navigation_shape[::-1])
        sig_shape = tuple(sig.axes_manager.signal_shape[::-1])
        n_nav_y, n_nav_x = nav_shape
        KY, KX = sig_shape
        sig_ax = sig.axes_manager.signal_axes
        ky_scale  = float(sig_ax[1].scale)
        ky_offset = float(sig_ax[1].offset)
        kx_scale  = float(sig_ax[0].scale)
        kx_offset = float(sig_ax[0].offset)

        if path and os.path.exists(path):
            mm, _, _ = _open_mrc_memmap(path)
        else:
            # Fall back: materialise from dask (slow, but works)
            print("Warning: no MRC path found, materialising from dask (slow)...",
                  flush=True)
            mm = sig.data.compute().astype(np.float32).reshape(
                n_nav_y * n_nav_x, KY, KX
            )

    N_total = n_nav_y * n_nav_x
    kr = int(kernel_radius)
    kr_win = kr + int(kernel_window_pad)
    min_d = int(min_distance)
    depth_px = int(np.ceil(3 * sigma))
    nav_chunk = int(nav_chunk_size)

    peaks_by_pos: list[Optional[np.ndarray]] = [None] * N_total

    # Queue between reader and GPU worker (bounded = back-pressure)
    blur_queue: queue.Queue = queue.Queue(maxsize=2)

    gpu_worker = _GpuWorker(kr, kr_win, threshold, min_d, KY, KX, device)

    reader = threading.Thread(
        target=_reader_thread,
        args=(mm, n_nav_y, n_nav_x, nav_chunk, depth_px, sigma,
              gpu_batch_patterns, blur_queue),
        daemon=True,
    )
    reader.start()

    # Relay: blur_queue -> gpu_worker (both bounded, relay prevents deadlock)
    def _relay():
        while True:
            item = blur_queue.get()
            if item is None:
                gpu_worker.in_queue.put(None)
                return
            gpu_worker.push(*item)

    threading.Thread(target=_relay, daemon=True).start()

    # Collect results
    n_done = 0
    reader_finished = threading.Event()

    def _watch_reader():
        reader.join()
        reader_finished.set()

    threading.Thread(target=_watch_reader, daemon=True).start()

    while n_done < N_total:
        try:
            result = gpu_worker.out_queue.get(timeout=0.5)
        except queue.Empty:
            if reader_finished.is_set() and gpu_worker.in_queue.empty():
                try:
                    result = gpu_worker.out_queue.get_nowait()
                except queue.Empty:
                    break
            continue
        flat_indices, peaks_list = result
        for i, fi in enumerate(flat_indices):
            peaks_by_pos[int(fi)] = peaks_list[i]
            n_done += 1
        if on_progress:
            on_progress(n_done, N_total)

    gpu_worker.stop()

    # Pack CSR buffer
    counts = np.array(
        [len(p) if p is not None else 0 for p in peaks_by_pos], dtype=np.int64
    )
    offsets = np.zeros(N_total + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    N_peaks = int(offsets[-1])
    flat_buffer = np.zeros((N_peaks, 5), dtype=np.float32)

    for fi in range(N_total):
        peaks = peaks_by_pos[fi]
        if peaks is None or len(peaks) == 0:
            continue
        s, e = offsets[fi], offsets[fi + 1]
        iy, ix = divmod(fi, n_nav_x)
        flat_buffer[s:e, 0] = ix
        flat_buffer[s:e, 1] = iy
        flat_buffer[s:e, 2] = peaks[:, 1] * kx_scale + kx_offset
        flat_buffer[s:e, 3] = peaks[:, 0] * ky_scale + ky_offset
        flat_buffer[s:e, 4] = peaks[:, 2]

    nav_shape_out = (n_nav_y, n_nav_x)
    return SpyDEDiffractionVectors(
        flat_buffer=flat_buffer,
        offsets=offsets,
        nav_shape=nav_shape_out,
        full_nav_shape=nav_shape_out,
        sig_shape=(KY, KX),
        sig_axes=sig_ax,
        kernel_radius_px=float(kr),
        kernel_radius_data=float(kr) * (kx_scale if sig_ax else 1.0),
        params=dict(
            sigma=sigma, kernel_radius=kr, threshold=threshold,
            min_distance=min_d, kernel_window_pad=kernel_window_pad,
        ),
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(path_or_signal, kernel_radius=14, threshold=0.3, min_distance=28,
              sigma=1.5, nav_chunk_size=64, gpu_batch_patterns=2000):
    import torch, hyperspy.api as hs

    if isinstance(path_or_signal, str):
        sig = hs.load(path_or_signal, lazy=True)
        path = path_or_signal
    else:
        sig = path_or_signal
        try:
            path = sig.metadata.General.original_filename
        except Exception:
            path = None

    nav_shape = tuple(sig.axes_manager.navigation_shape[::-1])
    sig_shape  = tuple(sig.axes_manager.signal_shape[::-1])
    N = nav_shape[0] * nav_shape[1]
    KY, KX = sig_shape

    print(f"\n{'='*60}")
    print(f"PIPELINED FIND-VECTORS BENCHMARK")
    print(f"  Signal: {sig.data.shape}  ({N} patterns, {KY}x{KX})")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  kr={kernel_radius}  thr={threshold}  min_dist={min_distance}")
    print(f"  sigma={sigma}  nav_chunk={nav_chunk_size}  gpu_batch={gpu_batch_patterns}")
    print(f"{'='*60}\n")

    # Single-frame CPU time for comparison
    from spyde.actions.find_vectors import _find_vectors_single_frame, _get_disk_fft, _make_disk
    from scipy.fft import next_fast_len
    from scipy.ndimage import gaussian_filter as gf
    kr = kernel_radius
    pH = next_fast_len(KY+2*kr); pW = next_fast_len(KX+2*kr)
    dfft = _get_disk_fft(kr, pH, pW)
    disk = _make_disk(kr); nd=disk.size; tm=float(disk.mean()); ts=float(np.sqrt(np.sum((disk-tm)**2)/nd))
    mm_tmp, _, _ = _open_mrc_memmap(path) if path else (None, None, None)
    if mm_tmp is not None:
        frame = mm_tmp[N//2].astype(np.float32)
    else:
        frame = sig.inav[nav_shape[0]//2, nav_shape[1]//2].data.compute().astype(np.float32)
    for _ in range(3):
        _find_vectors_single_frame(frame, kr, threshold, min_distance,
                                   subpixel=False, _disk_fft=dfft, _disk_stats=(nd,tm,ts))
    t0=time.perf_counter()
    for _ in range(10):
        _find_vectors_single_frame(frame, kr, threshold, min_distance,
                                   subpixel=False, _disk_fft=dfft, _disk_stats=(nd,tm,ts))
    cpu_ms = (time.perf_counter()-t0)/10*1000
    cpu_total = cpu_ms * N / 1000
    print(f"CPU single-thread: {cpu_ms:.1f} ms/pattern -> {cpu_total:.0f}s extrapolated")

    # Disk speed
    if path:
        t0=time.perf_counter()
        mm2, _, _ = _open_mrc_memmap(path)
        sample = np.array(mm2[:1000])
        disk_mb_s = sample.nbytes/1e6/(time.perf_counter()-t0)
        print(f"Disk speed (1000 patterns): {disk_mb_s:.0f} MB/s")
    print()

    n_done = [0]
    t_last = [time.perf_counter()]
    t_start = time.perf_counter()

    def _progress(done, total):
        n_done[0] = done
        now = time.perf_counter()
        if now - t_last[0] > 5.0:
            elapsed = now - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else float('inf')
            print(f"  {done:6d}/{total}  {rate:.0f} pat/s  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)
            t_last[0] = now

    vecs = find_vectors_pipelined(
        path_or_signal,
        kernel_radius=kernel_radius,
        threshold=threshold,
        min_distance=min_distance,
        sigma=sigma,
        nav_chunk_size=nav_chunk_size,
        gpu_batch_patterns=gpu_batch_patterns,
        on_progress=_progress,
    )

    elapsed = time.perf_counter() - t_start
    ms_per = elapsed / N * 1000
    print(f"\nDone: {elapsed:.1f}s total  ({ms_per:.2f} ms/pattern)  "
          f"{cpu_total/elapsed:.1f}x faster than single-thread CPU")
    print(f"Vectors found: {len(vecs.flat_buffer)} total  "
          f"({len(vecs.flat_buffer)/N:.2f} mean/pattern)")
    print(f"{'='*60}\n")
    return vecs

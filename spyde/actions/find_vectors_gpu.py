"""
find_vectors_gpu.py -- GPU-accelerated diffraction vector finding.

Two implementations are provided:

1. find_vectors_pytorch(data, ...)
   Pure PyTorch -- no custom kernels.  Batch-processes the entire (N, KY, KX)
   dataset through cuDNN conv2d (xcorr) and avg_pool2d (window stats) in a
   handful of kernel launches.  The disk kernel is a 1-channel weight stored
   in GPU SRAM between calls; it's small enough (361 floats) to stay resident
   in L2 cache for the lifetime of the batch.

2. find_vectors_cuda(data, ...)
   Custom CUDA kernel via torch.utils.cpp_extension.load_inline.
   - __constant__ memory for the disk kernel (1444 bytes -- survives across
     kernel launches without re-upload as long as the module is loaded).
   - Shared-memory tiling (TILE=32, halo=kr_win): each 32x32 output tile loads
     a (32+2*kr_win)2 region into smem once, then every thread walks the
     kr_win window from smem rather than global memory.  For kr_win=10 this
     cuts global-memory reads by ~(2*kr_win+1)^2 / 1 ~= 441x per output pixel.
   - Fused normalization: xcorr + window stats + NXCORR score in one pass,
     eliminating two intermediate (N, KY, KX) allocations.
   - Peak mask written to a bool tensor; NMS runs on CPU (peaks are sparse).

Both paths return the same structured output as _find_vectors_single_frame:
a list of (N_peaks, 3) float32 arrays [ky_px, kx_px, score] per nav position.

Usage
-----
    from spyde.actions.find_vectors_gpu import find_vectors_pytorch, find_vectors_cuda, benchmark

    # data: (N_nav_y, N_nav_x, KY, KX) float32 numpy array (already nav-blurred)
    peaks_list = find_vectors_pytorch(data, kernel_radius=9, threshold=0.2, min_distance=18)
    peaks_list = find_vectors_cuda(data, kernel_radius=9, threshold=0.2, min_distance=18)

    benchmark(data, kernel_radius=9, threshold=0.2, min_distance=18)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_disk_np(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (y ** 2 + x ** 2 <= r ** 2).astype(np.float32)
    disk /= disk.sum()
    return disk


def _greedy_nms(peaks_px: np.ndarray, scores: np.ndarray, min_d: int) -> np.ndarray:
    """CPU greedy NMS -- fine because N_peaks per frame is tiny (< 100)."""
    if len(peaks_px) <= 1:
        return np.arange(len(peaks_px), dtype=np.int64)
    order = np.argsort(-scores)
    peaks_px = peaks_px[order]
    kept = np.ones(len(peaks_px), dtype=bool)
    min_d2 = min_d * min_d
    for i in range(len(peaks_px)):
        if not kept[i]:
            continue
        dy = peaks_px[i + 1:, 0] - peaks_px[i, 0]
        dx = peaks_px[i + 1:, 1] - peaks_px[i, 1]
        kept[i + 1:][(dy * dy + dx * dx) <= min_d2] = False
    return order[kept]


def _peaks_from_mask(
    raw_corr_np: np.ndarray,   # (N, KY, KX)
    mask_np: np.ndarray,       # (N, KY, KX) bool
    min_distance: int,
    subpixel: bool = False,
) -> list[np.ndarray]:
    """Convert a boolean peak mask into a list of (n_peaks, 3) arrays."""
    N = raw_corr_np.shape[0]
    out = []
    for i in range(N):
        yx = np.argwhere(mask_np[i])
        if len(yx) == 0:
            out.append(np.zeros((0, 3), dtype=np.float32))
            continue
        scores = raw_corr_np[i, yx[:, 0], yx[:, 1]]
        kept_idx = _greedy_nms(yx, scores, min_distance)
        yx_kept = yx[kept_idx]
        sc_kept = scores[kept_idx]
        out.append(np.column_stack([
            yx_kept.astype(np.float32), sc_kept.astype(np.float32)
        ]))
    return out


# ---------------------------------------------------------------------------
# Implementation 1: Pure PyTorch
# ---------------------------------------------------------------------------

def find_vectors_pytorch(
    data: np.ndarray,
    kernel_radius: int,
    threshold: float,
    min_distance: int,
    kernel_window_pad: int = 1,
    subpixel: bool = False,
    device: str = "cuda",
    batch_size: int = 0,
) -> list[np.ndarray]:
    """
    Batch NXCORR peak-finding using PyTorch operations.

    Parameters
    ----------
    data : (N_y, N_x, KY, KX) or (N, KY, KX) float32 ndarray
        Nav-space blurred diffraction patterns.
    kernel_radius : int
        Disk template radius in pixels.
    threshold : float
        NXCORR threshold in (-1, 1).
    min_distance : int
        Minimum peak separation in pixels (applied in CPU NMS after GPU mask).
    kernel_window_pad : int
        Extra pixels added to the statistics window beyond kernel_radius.
    subpixel : bool
        Not yet implemented for GPU path (always returns integer positions).
    device : str
        'cuda' or 'cpu'.
    batch_size : int
        Number of frames per GPU batch.  0 = all at once.

    Returns
    -------
    List of (n_peaks, 3) float32 arrays [ky_px, kx_px, score], one per frame
    in raster order.
    """
    import torch
    import torch.nn.functional as F

    orig_shape = data.shape
    if data.ndim == 4:
        N_y, N_x, KY, KX = data.shape
        frames_np = data.reshape(-1, KY, KX)
    else:
        frames_np = data
        KY, KX = frames_np.shape[1], frames_np.shape[2]
    N = frames_np.shape[0]

    dev = torch.device(device)
    kr = int(kernel_radius)
    kr_win = kr + int(kernel_window_pad)
    kW_win = 2 * kr_win + 1
    min_d = int(min_distance)

    # -- Build disk kernel (stays on GPU for lifetime of call) --------------
    disk_np = _make_disk_np(kr)
    n_disk = disk_np.size
    t_mean = float(disk_np.mean())
    t_std = float(np.sqrt(np.sum((disk_np - t_mean) ** 2) / n_disk))
    # Shape: (1, 1, kH, kW) for F.conv2d
    disk_t = torch.from_numpy(disk_np).to(dev).view(1, 1, 2 * kr + 1, 2 * kr + 1)

    bs = N if batch_size == 0 else int(batch_size)
    all_masks = []
    all_corrs = []

    for start in range(0, N, bs):
        end = min(start + bs, N)
        chunk_np = frames_np[start:end]  # (B, KY, KX)
        B = chunk_np.shape[0]

        frames_t = torch.from_numpy(chunk_np).to(dev).unsqueeze(1)  # (B,1,KY,KX)

        # -- Step 1: xcorr via conv2d (reflect-padded) ---------------------
        # F.pad with reflect then conv2d with padding=0 is equivalent to
        # match_template with reflect boundary.
        frames_pad = F.pad(frames_t, (kr, kr, kr, kr), mode="reflect")
        # conv2d: weight=(1,1,kH,kW), input=(B,1,KY+2kr,KX+2kr) -> (B,1,KY,KX)
        xcorr = F.conv2d(frames_pad, disk_t, padding=0).squeeze(1)  # (B,KY,KX)

        # -- Step 2: window stats via avg_pool2d ---------------------------
        # avg_pool2d computes local mean in one fused CUDA kernel.
        # Reflect-pad by kr_win for the (possibly larger) stats window.
        frames_stat = F.pad(frames_t, (kr_win, kr_win, kr_win, kr_win), mode="reflect")
        win_mean = F.avg_pool2d(
            frames_stat, kernel_size=kW_win, stride=1, padding=0
        ).squeeze(1)  # (B,KY,KX)
        win_sq_mean = F.avg_pool2d(
            frames_stat ** 2, kernel_size=kW_win, stride=1, padding=0
        ).squeeze(1)

        win_var = (win_sq_mean - win_mean ** 2).clamp_(min=0.0)
        win_std = win_var.sqrt_()

        # -- Step 3: normalise ----------------------------------------------
        numerator = xcorr / n_disk - win_mean * t_mean
        denom = win_std * t_std
        valid = denom >= 1e-8
        raw_corr = torch.where(
            valid,
            (numerator / denom.clamp(min=1e-8)).clamp_(-1.0, 1.0),
            torch.zeros_like(numerator),
        )

        # -- Step 4: local max mask -----------------------------------------
        # max_pool2d with padding=min_d reproduces scipy's maximum_filter.
        raw_4d = raw_corr.unsqueeze(1)
        local_max = F.max_pool2d(
            raw_4d, kernel_size=2 * min_d + 1, stride=1, padding=min_d
        ).squeeze(1)
        peak_mask = (raw_corr == local_max) & (raw_corr >= threshold)

        all_masks.append(peak_mask.cpu().numpy())
        all_corrs.append(raw_corr.cpu().numpy())

    mask_np = np.concatenate(all_masks, axis=0)   # (N, KY, KX)
    corr_np = np.concatenate(all_corrs, axis=0)   # (N, KY, KX)

    return _peaks_from_mask(corr_np, mask_np, min_d, subpixel=subpixel)


# ---------------------------------------------------------------------------
# Implementation 2: Custom CUDA kernel
# ---------------------------------------------------------------------------

_CUDA_MODULE = None   # cached compiled module

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Disk kernel in __constant__ memory -- 1444 bytes for kr=9 (19x19).
// Max supported: kr=12 (25x25 = 625 floats = 2500 bytes).
// Survives across kernel launches without re-upload.
#define MAX_DISK_ELEMS 625
__constant__ float c_disk[MAX_DISK_ELEMS];
__constant__ int   c_disk_kH;     // 2*kr+1
__constant__ int   c_disk_kW;     // 2*kr+1
__constant__ int   c_n_disk;      // kH*kW
__constant__ float c_t_mean;
__constant__ float c_t_std;

// -------------------------------------------------------------------
// upload_disk_to_constant: called once from Python before kernel runs
// -------------------------------------------------------------------
void upload_disk(
    const float* disk_host,  // host pointer, size kH*kW
    int kH, int kW,
    float t_mean, float t_std
) {
    int n = kH * kW;
    cudaMemcpyToSymbol(c_disk,   disk_host, n * sizeof(float));
    cudaMemcpyToSymbol(c_disk_kH, &kH,      sizeof(int));
    cudaMemcpyToSymbol(c_disk_kW, &kW,      sizeof(int));
    cudaMemcpyToSymbol(c_n_disk, &n,        sizeof(int));
    cudaMemcpyToSymbol(c_t_mean, &t_mean,   sizeof(float));
    cudaMemcpyToSymbol(c_t_std,  &t_std,    sizeof(float));
}

// -------------------------------------------------------------------
// nxcorr_kernel: fused xcorr + window-stats + normalise
//
// Grid:  (ceil(W/TILE), ceil(H/TILE), N)
// Block: (TILE, TILE, 1)
//
// Shared memory layout (one float tile):
//   size = (TILE + 2*kr_win) * (TILE + 2*kr_win) floats
//   Loaded once per block from the reflect-padded input.
//
// The same smem tile is used for BOTH xcorr (walks c_disk kernel over
// the TILE+2*kr halo) and window stats (walks the kr_win window).
// Because kr_win = kr+pad >= kr, the tile halo covers both.
// -------------------------------------------------------------------
#define TILE 32

__global__ void nxcorr_kernel(
    const float* __restrict__ frames,  // (N, H, W) reflect-padded by kr_win
    float*       __restrict__ raw_corr,// (N, H, W) output
    bool*        __restrict__ peak_mask,// (N, H, W) output
    int N, int H, int W,               // original (unpadded) dims
    int kr, int kr_win,                // kernel radius, stats-window radius
    float inv_n_disk, float inv_n_win,
    float threshold
) {
    // Padded frame dims
    const int PH = H + 2 * kr_win;
    const int PW = W + 2 * kr_win;

    const int n  = blockIdx.z;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    // Top-left corner of this block's output tile in output coords
    const int out_y0 = blockIdx.y * TILE;
    const int out_x0 = blockIdx.x * TILE;

    // Halo size: must cover both kr (for xcorr) and kr_win (for stats).
    // Since kr_win >= kr, halo = kr_win.
    const int halo    = kr_win;
    const int TILE_PH = TILE + 2 * halo;
    const int TILE_PW = TILE + 2 * halo;

    // Shared memory: one tile of the reflect-padded input
    extern __shared__ float smem[];  // TILE_PH * TILE_PW floats

    // -- Cooperative load of the smem tile ------------------------------
    // Each thread loads multiple elements if TILE_PH*TILE_PW > TILE*TILE.
    const int n_smem   = TILE_PH * TILE_PW;
    const int n_threads = TILE * TILE;
    const int frame_base = n * PH * PW;

    for (int idx = ty * TILE + tx; idx < n_smem; idx += n_threads) {
        int sr = idx / TILE_PW;  // row in smem tile
        int sc = idx % TILE_PW;  // col in smem tile
        // Map smem position to padded-frame position
        int pr = out_y0 + sr;    // already offset by 0 because padded frame
        int pc = out_x0 + sc;
        float val = 0.f;
        if (pr < PH && pc < PW)
            val = frames[frame_base + pr * PW + pc];
        smem[idx] = val;
    }
    __syncthreads();

    const int out_y = out_y0 + ty;
    const int out_x = out_x0 + tx;
    if (out_y >= H || out_x >= W) return;

    // smem origin for this thread's output pixel:
    // The halo offset in smem is (halo, halo) relative to out_y0, out_x0.
    // Thread (ty, tx) reads smem at (ty + halo - offset, tx + halo - offset).
    // For xcorr the disk is centred at smem (ty+halo, tx+halo); disk offset is
    // -(kr..kr).  With halo=kr_win>=kr this always stays in [0, TILE_PH).

    // -- Step 1: xcorr (sum disk * smem patch) -------------------------
    float xcorr_val = 0.f;
    {
        int disk_idx = 0;
        // The disk is (c_disk_kH x c_disk_kW), centred at (kr, kr).
        // In smem the disk centre is at (ty + halo, tx + halo).
        // Disk offset: delta = -kr .. kr.  Smem index: (ty+halo+delta_r, tx+halo+delta_c).
        // Note: halo >= kr so indices are always >= 0.
        for (int dr = 0; dr < c_disk_kH; dr++) {
            int sr = ty + halo + (dr - kr);
            for (int dc = 0; dc < c_disk_kW; dc++) {
                int sc = tx + halo + (dc - kr);
                xcorr_val += c_disk[disk_idx++] * smem[sr * TILE_PW + sc];
            }
        }
    }

    // -- Step 2: window stats (sum over kr_win window) -----------------
    float sum1 = 0.f, sum2 = 0.f;
    {
        // Window is (kW_win x kW_win) = (2*kr_win+1)^2, centred at (ty+halo, tx+halo).
        for (int dr = 0; dr < 2 * kr_win + 1; dr++) {
            int sr = ty + (dr);   // ty + 0..2*kr_win -> smem row ty..ty+2*kr_win
            for (int dc = 0; dc < 2 * kr_win + 1; dc++) {
                int sc = tx + (dc);
                float v = smem[sr * TILE_PW + sc];
                sum1 += v;
                sum2 += v * v;
            }
        }
    }

    // -- Step 3: normalise ----------------------------------------------
    float win_mean = sum1 * inv_n_win;
    float win_var  = fmaxf(sum2 * inv_n_win - win_mean * win_mean, 0.f);
    float win_std  = sqrtf(win_var);
    float denom    = win_std * c_t_std;
    float num      = xcorr_val * inv_n_disk - win_mean * c_t_mean;
    float score    = (denom >= 1e-8f) ? fmaxf(-1.f, fminf(1.f, num / denom)) : 0.f;

    int out_idx = n * H * W + out_y * W + out_x;
    raw_corr[out_idx]  = score;
    peak_mask[out_idx] = (score >= threshold);  // refined in local-max pass
}

// -------------------------------------------------------------------
// local_max_kernel: sets peak_mask[i] = true iff score[i] is the
// strict local maximum in a (2*min_d+1)^2 neighbourhood.
// Grid/block same as nxcorr_kernel.
// -------------------------------------------------------------------
__global__ void local_max_kernel(
    const float* __restrict__ raw_corr, // (N, H, W)
    bool*        __restrict__ peak_mask,// (N, H, W)  in/out
    int N, int H, int W,
    int min_d, float threshold
) {
    const int n     = blockIdx.z;
    const int out_y = blockIdx.y * TILE + threadIdx.y;
    const int out_x = blockIdx.x * TILE + threadIdx.x;
    if (out_y >= H || out_x >= W) return;

    int idx = n * H * W + out_y * W + out_x;
    if (!peak_mask[idx]) return;  // already below threshold

    float center = raw_corr[idx];
    int y0 = max(0, out_y - min_d);
    int y1 = min(H - 1, out_y + min_d);
    int x0 = max(0, out_x - min_d);
    int x1 = min(W - 1, out_x + min_d);

    for (int ny = y0; ny <= y1; ny++) {
        for (int nx = x0; nx <= x1; nx++) {
            if (raw_corr[n * H * W + ny * W + nx] > center) {
                peak_mask[idx] = false;
                return;
            }
        }
    }
}

// -------------------------------------------------------------------
// Python-visible entry points
// -------------------------------------------------------------------
void upload_disk_py(torch::Tensor disk_cpu, float t_mean, float t_std) {
    // disk_cpu: (kH, kW) float32 CPU tensor
    int kH = disk_cpu.size(0);
    int kW = disk_cpu.size(1);
    TORCH_CHECK(kH * kW <= MAX_DISK_ELEMS,
        "Disk kernel too large for constant memory (max ", MAX_DISK_ELEMS, " elements)");
    upload_disk(disk_cpu.data_ptr<float>(), kH, kW, t_mean, t_std);
}

std::vector<torch::Tensor> nxcorr_forward(
    torch::Tensor frames_padded,  // (N, PH, PW) float32 CUDA -- reflect-padded by kr_win
    int H, int W,                 // original dims
    int kr, int kr_win,
    float threshold, int min_d
) {
    const int N  = frames_padded.size(0);
    const float inv_n_disk = 1.f / float((2*kr+1)*(2*kr+1));
    const float inv_n_win  = 1.f / float((2*kr_win+1)*(2*kr_win+1));

    auto opts = frames_padded.options();
    auto raw_corr  = torch::empty({N, H, W}, opts);
    auto peak_mask = torch::zeros({N, H, W}, opts.dtype(torch::kBool));

    const int halo    = kr_win;
    const int TILE_PH = TILE + 2 * halo;
    const int TILE_PW = TILE + 2 * halo;
    const int shmem   = TILE_PH * TILE_PW * sizeof(float);

    dim3 block(TILE, TILE, 1);
    dim3 grid(
        (W + TILE - 1) / TILE,
        (H + TILE - 1) / TILE,
        N
    );

    nxcorr_kernel<<<grid, block, shmem>>>(
        frames_padded.data_ptr<float>(),
        raw_corr.data_ptr<float>(),
        peak_mask.data_ptr<bool>(),
        N, H, W, kr, kr_win,
        inv_n_disk, inv_n_win, threshold
    );

    // Second pass: enforce local-maximum condition
    local_max_kernel<<<grid, block>>>(
        raw_corr.data_ptr<float>(),
        peak_mask.data_ptr<bool>(),
        N, H, W, min_d, threshold
    );

    return {raw_corr, peak_mask};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upload_disk",    &upload_disk_py,  "Upload disk kernel to constant memory");
    m.def("nxcorr_forward", &nxcorr_forward,  "Fused NXCORR kernel (xcorr + stats + normalise)");
}
"""

_CUDA_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

void upload_disk_py(torch::Tensor disk_cpu, float t_mean, float t_std);
std::vector<torch::Tensor> nxcorr_forward(
    torch::Tensor frames_padded,
    int H, int W, int kr, int kr_win,
    float threshold, int min_d
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("upload_disk",    &upload_disk_py);
    m.def("nxcorr_forward", &nxcorr_forward);
}
"""


def _get_cuda_module():
    """Compile and cache the CUDA extension (JIT, first call only)."""
    global _CUDA_MODULE
    if _CUDA_MODULE is not None:
        return _CUDA_MODULE

    import os
    import torch
    from torch.utils.cpp_extension import load_inline

    # Python 3.12 removed stdlib distutils; setuptools re-adds it but torch's
    # cpp_extension does `distutils._msvccompiler` as an attribute lookup.
    # Importing the submodule first populates the attribute on the package object.
    try:
        import setuptools  # noqa: F401 — must come before distutils import
        import distutils._msvccompiler  # noqa: F401 — populates distutils._msvccompiler attr
    except ImportError:
        pass

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}")

    print("Compiling NXCORR CUDA kernel (first call only)...", flush=True)
    _CUDA_MODULE = load_inline(
        name="nxcorr_cuda",
        cpp_sources="",
        cuda_sources=_CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    print("CUDA kernel compiled.", flush=True)
    return _CUDA_MODULE


def find_vectors_cuda(
    data: np.ndarray,
    kernel_radius: int,
    threshold: float,
    min_distance: int,
    kernel_window_pad: int = 1,
    subpixel: bool = False,
    device: str = "cuda",
    batch_size: int = 0,
) -> list[np.ndarray]:
    """
    Batch NXCORR using the custom fused CUDA kernel.

    The disk kernel is uploaded to __constant__ memory once and reused for
    all frames.  Each block loads a shared-memory tile covering the output
    TILExTILE region plus a halo of kr_win on each side, then computes xcorr
    and window stats entirely from smem -- no repeated global-memory reads.

    Parameters mirror find_vectors_pytorch().
    """
    import torch
    import torch.nn.functional as F

    mod = _get_cuda_module()
    dev = torch.device(device)

    orig_shape = data.shape
    if data.ndim == 4:
        N_y, N_x, KY, KX = data.shape
        frames_np = data.reshape(-1, KY, KX)
    else:
        frames_np = data
        KY, KX = frames_np.shape[1], frames_np.shape[2]
    N = frames_np.shape[0]

    kr = int(kernel_radius)
    kr_win = kr + int(kernel_window_pad)
    min_d = int(min_distance)

    # -- Upload disk to constant memory (no-op on subsequent calls if same kr) --
    disk_np = _make_disk_np(kr)
    n_disk = disk_np.size
    t_mean = float(disk_np.mean())
    t_std = float(np.sqrt(np.sum((disk_np - t_mean) ** 2) / n_disk))
    disk_cpu = torch.from_numpy(disk_np.reshape(2 * kr + 1, 2 * kr + 1))
    mod.upload_disk(disk_cpu, t_mean, t_std)

    bs = N if batch_size == 0 else int(batch_size)
    all_masks = []
    all_corrs = []

    for start in range(0, N, bs):
        end = min(start + bs, N)
        chunk_np = frames_np[start:end].astype(np.float32)
        B = chunk_np.shape[0]

        frames_t = torch.from_numpy(chunk_np).to(dev)  # (B, KY, KX)

        # Reflect-pad by kr_win -- the kernel reads this padded frame from smem
        frames_pad = F.pad(
            frames_t.unsqueeze(1),
            (kr_win, kr_win, kr_win, kr_win),
            mode="reflect",
        ).squeeze(1)  # (B, KY+2*kr_win, KX+2*kr_win)

        raw_corr, peak_mask = mod.nxcorr_forward(
            frames_pad, KY, KX, kr, kr_win, threshold, min_d
        )

        all_masks.append(peak_mask.cpu().numpy())
        all_corrs.append(raw_corr.cpu().numpy())

    mask_np = np.concatenate(all_masks, axis=0)
    corr_np = np.concatenate(all_corrs, axis=0)

    return _peaks_from_mask(corr_np, mask_np, min_d, subpixel=subpixel)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(
    data: np.ndarray,
    kernel_radius: int = 9,
    threshold: float = 0.2,
    min_distance: int = 18,
    n_warmup: int = 2,
    n_repeat: int = 5,
    run_cuda_kernel: bool = True,
):
    """Compare CPU, PyTorch-GPU, and custom-CUDA-kernel throughput."""
    import torch
    from spyde.actions.find_vectors import (
        _find_vectors_single_frame, _get_disk_fft, _make_disk,
        _nav_chunk_size,
    )
    from scipy.ndimage import gaussian_filter

    N_y, N_x, KY, KX = data.shape
    N = N_y * N_x
    kr = int(kernel_radius)

    print(f"\n{'='*64}")
    print(f"FIND-VECTORS GPU BENCHMARK")
    print(f"  Data:    {data.shape}  ({N} patterns, {KY}x{KX} each)")
    print(f"  kr={kr}  threshold={threshold}  min_dist={min_distance}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*64}")

    # -- CPU baseline -------------------------------------------------------
    from scipy.fft import next_fast_len
    pH = next_fast_len(KY + 2 * kr)
    pW = next_fast_len(KX + 2 * kr)
    disk_fft = _get_disk_fft(kr, pH, pW)
    disk = _make_disk(kr)
    n_d = disk.size
    tm = float(disk.mean())
    ts = float(np.sqrt(np.sum((disk - tm) ** 2) / n_d))
    ds = (n_d, tm, ts)
    frame = data[N_y // 2, N_x // 2].astype(np.float32)

    for _ in range(3):
        _find_vectors_single_frame(frame, kr, threshold, min_distance,
                                   subpixel=False, _disk_fft=disk_fft, _disk_stats=ds)
    t0 = time.perf_counter()
    for _ in range(50):
        _find_vectors_single_frame(frame, kr, threshold, min_distance,
                                   subpixel=False, _disk_fft=disk_fft, _disk_stats=ds)
    cpu_ms = (time.perf_counter() - t0) / 50 * 1000
    cpu_total_s = cpu_ms * N / 1000
    print(f"\n[CPU  ] {cpu_ms:.3f} ms/frame  ->  {cpu_total_s:.1f} s total")

    def _time_gpu(fn, label):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            fn()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_repeat
        ms_per_frame = elapsed / N * 1000
        speedup = cpu_total_s / elapsed
        print(f"[{label:<6}] {ms_per_frame:.3f} ms/frame  ->  {elapsed:.2f} s total  "
              f"({speedup:.1f}x CPU)")
        return elapsed

    # -- PyTorch GPU --------------------------------------------------------
    print()
    _time_gpu(
        lambda: find_vectors_pytorch(data, kr, threshold, min_distance),
        "PyTorch"
    )

    # -- Custom CUDA kernel -------------------------------------------------
    if run_cuda_kernel:
        try:
            _time_gpu(
                lambda: find_vectors_cuda(data, kr, threshold, min_distance),
                "CUDA"
            )
        except Exception as e:
            print(f"[CUDA  ] Failed: {e}")

    # -- Correctness spot-check ---------------------------------------------
    print(f"\n{'-'*64}")
    print("Correctness check (frame [N_y//2, N_x//2]):")
    sample = data[N_y // 2:N_y // 2 + 1, N_x // 2:N_x // 2 + 1].astype(np.float32)

    _, _, cpu_peaks = _find_vectors_single_frame(
        sample[0, 0], kr, threshold, min_distance,
        subpixel=False, _disk_fft=disk_fft, _disk_stats=ds,
    )
    pt_peaks = find_vectors_pytorch(sample, kr, threshold, min_distance)[0]
    print(f"  CPU peaks:     {len(cpu_peaks)}")
    print(f"  PyTorch peaks: {len(pt_peaks)}")

    if run_cuda_kernel:
        try:
            cu_peaks = find_vectors_cuda(sample, kr, threshold, min_distance)[0]
            print(f"  CUDA peaks:    {len(cu_peaks)}")
        except Exception as e:
            print(f"  CUDA peaks:    error -- {e}")

    print(f"{'='*64}\n")

"""
Benchmark: virtual vector imaging — numpy vs PyTorch CPU vs PyTorch CUDA vs custom CUDA kernel.

Run with:
    python spyde/tests/benchmark_virtual_vector_imaging.py

Reports mean ± std (ms) per call for each method at several dataset sizes.
The goal is < 10 ms for "live" drag updates.
"""

from __future__ import annotations

import time
import numpy as np

# ── Synthetic dataset ─────────────────────────────────────────────────────────

def _make_flat_buffer(nav_shape, n_per_pos, seed=42):
    rng = np.random.default_rng(seed)
    nav_y, nav_x = nav_shape
    n_nav = nav_y * nav_x
    N = n_nav * n_per_pos

    flat = np.empty((N, 6), dtype=np.float32)
    nav_idx = np.arange(n_nav, dtype=np.int64)
    iy_all = np.repeat(nav_idx // nav_x, n_per_pos)
    ix_all = np.repeat(nav_idx % nav_x, n_per_pos)
    flat[:, 0] = ix_all
    flat[:, 1] = iy_all
    flat[:, 2] = rng.uniform(-1, 1, N).astype(np.float32)  # kx
    flat[:, 3] = rng.uniform(-1, 1, N).astype(np.float32)  # ky
    flat[:, 4] = -1.0                                        # time: 4D
    flat[:, 5] = rng.uniform(0.1, 1.0, N).astype(np.float32)  # intensity

    counts = np.full(n_nav, n_per_pos, dtype=np.int64)
    offsets = np.zeros(n_nav + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return flat, offsets, nav_shape


# ── Method implementations ────────────────────────────────────────────────────

def vvi_numpy(flat, nav_shape, cx, cy, r_outer, r_inner=0.0):
    """Pure numpy: vectorised distance + np.add.at scatter."""
    if len(flat) == 0:
        return np.zeros(nav_shape, dtype=np.float32)
    kx = flat[:, 2]; ky = flat[:, 3]
    dist2 = (kx - cx)**2 + (ky - cy)**2
    mask = dist2 <= r_outer * r_outer
    if r_inner > 0:
        mask &= dist2 > r_inner * r_inner
    nav_y, nav_x = nav_shape
    out = np.zeros(nav_y * nav_x, dtype=np.float32)
    if mask.any():
        flat_nav = flat[mask, 1].astype(np.int32) * nav_x + flat[mask, 0].astype(np.int32)
        np.add.at(out, flat_nav, flat[mask, 5])  # COL_INTENSITY
    return out.reshape(nav_shape)


def vvi_torch_cpu(flat_t, nav_shape, cx, cy, r_outer, r_inner=0.0):
    """PyTorch CPU: same logic as numpy but using torch ops."""
    import torch
    kx = flat_t[:, 2]; ky = flat_t[:, 3]
    dist2 = (kx - cx)**2 + (ky - cy)**2
    mask = dist2 <= r_outer * r_outer
    if r_inner > 0:
        mask &= dist2 > r_inner * r_inner
    nav_y, nav_x = nav_shape
    out = torch.zeros(nav_y * nav_x, dtype=torch.float32)
    if mask.any():
        flat_nav = (flat_t[mask, 1].to(torch.int64) * nav_x
                    + flat_t[mask, 0].to(torch.int64))
        out.scatter_add_(0, flat_nav, flat_t[mask, 5])  # COL_INTENSITY
    return out.reshape(nav_shape).numpy()


def vvi_torch_cuda(flat_gpu, nav_shape, cx, cy, r_outer, r_inner=0.0):
    """
    PyTorch CUDA: distance filter on GPU, scatter_add_ on GPU.
    All intermediate tensors stay on device; only the final (nav_y, nav_x) comes back.
    """
    import torch
    kx = flat_gpu[:, 2]; ky = flat_gpu[:, 3]
    dist2 = (kx - cx)**2 + (ky - cy)**2
    mask = dist2 <= r_outer * r_outer
    if r_inner > 0:
        mask &= dist2 > r_inner * r_inner
    nav_y, nav_x = nav_shape
    out = torch.zeros(nav_y * nav_x, dtype=torch.float32, device=flat_gpu.device)
    if mask.any():
        flat_nav = (flat_gpu[mask, 1].to(torch.int64) * nav_x
                    + flat_gpu[mask, 0].to(torch.int64))
        out.scatter_add_(0, flat_nav, flat_gpu[mask, 5])  # COL_INTENSITY
    return out.reshape(nav_shape).cpu().numpy()


# ── Custom CUDA kernel via torch.utils.cpp_extension ─────────────────────────

_VVI_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * vvi_kernel: each thread handles one vector.
 *
 * flat   : (N, 6) float32  — [nav_x, nav_y, kx, ky, time, intensity]
 * out    : (nav_y * nav_x,) float32  — accumulation target
 * cx, cy : ROI centre
 * r2_out : outer radius squared
 * r2_in  : inner radius squared (0 = filled disk)
 * nav_x  : navigation grid width (for flat index)
 */
__global__ void vvi_kernel(
    const float* __restrict__ flat,
    float*       __restrict__ out,
    int N, float cx, float cy,
    float r2_out, float r2_in, int nav_x
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float kx = flat[i * 6 + 2];
    float ky = flat[i * 6 + 3];
    float d2 = (kx - cx)*(kx - cx) + (ky - cy)*(ky - cy);
    if (d2 > r2_out || d2 <= r2_in) return;

    int ix  = (int)flat[i * 6 + 0];
    int iy  = (int)flat[i * 6 + 1];
    float v = flat[i * 6 + 5];  // COL_INTENSITY
    atomicAdd(&out[iy * nav_x + ix], v);
}

torch::Tensor vvi_cuda(
    torch::Tensor flat,   // (N, 6) float32 CUDA
    int nav_y, int nav_x,
    float cx, float cy,
    float r_outer, float r_inner
) {
    int N = flat.size(0);
    auto out = torch::zeros({nav_y * nav_x}, flat.options());
    if (N == 0) return out.reshape({nav_y, nav_x});

    const int block = 256;
    const int grid  = (N + block - 1) / block;
    vvi_kernel<<<grid, block>>>(
        flat.data_ptr<float>(), out.data_ptr<float>(),
        N, cx, cy,
        r_outer * r_outer, r_inner * r_inner, nav_x
    );
    return out.reshape({nav_y, nav_x});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vvi_cuda", &vvi_cuda);
}
"""

_VVI_MODULE = None

def _get_vvi_module():
    global _VVI_MODULE
    if _VVI_MODULE is not None:
        return _VVI_MODULE
    import torch, os
    from torch.utils.cpp_extension import load_inline
    cap = torch.cuda.get_device_capability(0)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")
    print("  Compiling VVI CUDA kernel...", flush=True)
    _VVI_MODULE = load_inline(
        name="vvi_kernel",
        cpp_sources="",
        cuda_sources=_VVI_CUDA_SOURCE,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    print("  Done.", flush=True)
    return _VVI_MODULE


def vvi_cuda_kernel(flat_gpu, nav_shape, cx, cy, r_outer, r_inner=0.0):
    """Custom CUDA kernel: one thread per vector, atomicAdd into nav grid."""
    import torch
    mod = _get_vvi_module()
    nav_y, nav_x = nav_shape
    out = mod.vvi_cuda(flat_gpu, nav_y, nav_x, float(cx), float(cy),
                       float(r_outer), float(r_inner))
    torch.cuda.synchronize()
    return out.cpu().numpy()


# ── Timing helper ─────────────────────────────────────────────────────────────

def _bench(fn, n_warmup=3, n_runs=20):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return times.mean(), times.std()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmarks():
    import torch

    cuda_ok = torch.cuda.is_available()
    print(f"\n{'='*70}")
    print("Virtual Vector Imaging — method benchmark")
    if cuda_ok:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}\n")

    configs = [
        # (nav_shape,      n_per_pos, label)
        ((64,  64),   5,  " 64×64,  5v/pos =   20k vectors"),
        ((128, 128),  5,  "128×128, 5v/pos =   82k vectors"),
        ((256, 256),  5,  "256×256, 5v/pos =  328k vectors"),
        ((256, 256), 15,  "256×256,15v/pos =  983k vectors"),
        ((512, 512),  5,  "512×512, 5v/pos = 1.3M  vectors"),
    ]

    roi_configs = [
        (0.0, 0.0, 0.5, 0.0,  "large disk  (r=0.5, ~50% hits)"),
        (0.0, 0.0, 0.5, 0.3,  "annulus     (0.3<r<0.5)       "),
        (0.0, 0.0, 0.05, 0.0, "small disk  (r=0.05, ~0.3% hits)"),
    ]

    for nav_shape, n_per_pos, cfg_label in configs:
        print(f"\n── {cfg_label} ──")
        flat, offsets, _ = _make_flat_buffer(nav_shape, n_per_pos)

        flat_t_cpu = torch.from_numpy(flat)
        if cuda_ok:
            flat_t_gpu = flat_t_cpu.cuda()
            # Pre-compile kernel with a tiny call
            _get_vvi_module()
            vvi_cuda_kernel(flat_t_gpu, (4, 4), 0.0, 0.0, 5.0)

        for cx, cy, r_out, r_in, roi_label in roi_configs:
            print(f"  ROI: {roi_label}")

            mu, sd = _bench(lambda: vvi_numpy(flat, nav_shape, cx, cy, r_out, r_in))
            print(f"    numpy CPU:       {mu:6.2f} ± {sd:.2f} ms")

            mu, sd = _bench(lambda: vvi_torch_cpu(flat_t_cpu, nav_shape, cx, cy, r_out, r_in))
            print(f"    torch CPU:       {mu:6.2f} ± {sd:.2f} ms")

            if cuda_ok:
                # Warm GPU (first call includes H2D latency)
                vvi_torch_cuda(flat_t_gpu, nav_shape, cx, cy, r_out, r_in)
                mu, sd = _bench(lambda: vvi_torch_cuda(flat_t_gpu, nav_shape, cx, cy, r_out, r_in))
                print(f"    torch CUDA:      {mu:6.2f} ± {sd:.2f} ms")

                vvi_cuda_kernel(flat_t_gpu, nav_shape, cx, cy, r_out, r_in)
                mu, sd = _bench(lambda: vvi_cuda_kernel(flat_t_gpu, nav_shape, cx, cy, r_out, r_in))
                print(f"    custom kernel:   {mu:6.2f} ± {sd:.2f} ms")

    print(f"\n{'='*70}\n")

    # ── 5D benchmark: 256x256x500 ─────────────────────────────────────────────
    print("5D benchmark: 256x256 nav, 500 time steps, 5 vecs/pos")
    print("  (163M total vectors — per-frame query only)\n")

    nav_shape_5d = (256, 256)
    n_t = 500
    n_per = 5
    nav_y, nav_x = nav_shape_5d
    n_nav = nav_y * nav_x
    N_5d = n_nav * n_per * n_t
    rng = np.random.default_rng(0)
    flat_5d = np.empty((N_5d, 6), dtype=np.float32)
    nav_idx = np.arange(n_nav, dtype=np.int64)
    iy_all = np.repeat(nav_idx // nav_x, n_per * n_t)
    ix_all = np.repeat(nav_idx % nav_x, n_per * n_t)
    flat_5d[:, 0] = ix_all
    flat_5d[:, 1] = iy_all
    flat_5d[:, 2] = rng.uniform(-1, 1, N_5d).astype(np.float32)
    flat_5d[:, 3] = rng.uniform(-1, 1, N_5d).astype(np.float32)
    flat_5d[:, 4] = np.tile(np.repeat(np.arange(n_t, dtype=np.float32), n_per), n_nav)
    flat_5d[:, 5] = rng.uniform(0.1, 1.0, N_5d).astype(np.float32)

    from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors
    counts_5d = np.full(n_nav, n_per * n_t, dtype=np.int64)
    offsets_5d = np.zeros(n_nav + 1, dtype=np.int64)
    np.cumsum(counts_5d, out=offsets_5d[1:])
    vecs_5d = SpyDEDiffractionVectors(
        flat_buffer=flat_5d, offsets=offsets_5d,
        nav_shape=nav_shape_5d, full_nav_shape=(n_t, *nav_shape_5d),
        sig_shape=(256, 256), sig_axes=None,
        kernel_radius_px=4.0, kernel_radius_data=0.04,
    )

    print("  Building KDTree...", flush=True)
    t_build = time.perf_counter()
    vecs_5d.build_kdtree()
    print(f"  KDTree built in {(time.perf_counter()-t_build)*1000:.0f} ms\n")

    for roi_label, r_out in [("large disk r=0.5", 0.5), ("small disk r=0.05", 0.05)]:
        print(f"  ROI: {roi_label}")
        # Warm up
        vecs_5d.virtual_image_from_roi(0.0, 0.0, r_out, t=250)
        mu, sd = _bench(lambda: vecs_5d.virtual_image_from_roi(0.0, 0.0, r_out, t=250))
        print(f"    numpy per-frame (t=250): {mu:6.2f} ± {sd:.2f} ms")
        if vecs_5d._kdtree is not None:
            vecs_5d.virtual_image_from_kdtree(0.0, 0.0, r_out, t=250)
            mu, sd = _bench(lambda: vecs_5d.virtual_image_from_kdtree(0.0, 0.0, r_out, t=250))
            print(f"    kdtree per-frame (t=250):{mu:6.2f} ± {sd:.2f} ms")

    print()
    t0 = time.perf_counter()
    vecs_5d.virtual_image_series(0.0, 0.0, 0.5)
    print(f"  virtual_image_series (all 500 frames, r=0.5): {(time.perf_counter()-t0)*1000:.0f} ms")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    run_benchmarks()

"""Quick probe: GPU blur timing and correctness, then full GPU pipeline."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import math, numpy as np, time, torch, torch.nn.functional as F
import hyperspy.api as hs
from scipy.ndimage import gaussian_filter, convolve1d
import setuptools
from spyde.actions.find_vectors_pipeline import _get_cuda_module, _make_disk_np, _greedy_nms_batch

path = r'D:/data/sample.mrc'
sig = hs.load(path, lazy=True)
KY = KX = 512; sigma = 1.5; KR = 14; KR_WIN = 15; dev = torch.device('cuda')

chunk = sig.data[:11, :11].compute(scheduler='synchronous').astype(np.float32)
cy_ = cx_ = 11

def make_k1d(sigma, device=None):
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2); k /= k.sum()
    return k.to(device) if device else k, r


def blur_nav_gpu(ct, kH, kW, r):
    """
    Separable Gaussian blur over the first two (nav) dims of (cy, cx, KY, KX).

    Uses depthwise F.conv2d with KY*KX input channels (one per signal pixel).
    Reflect-pad via index_select so the boundary condition matches
    scipy.ndimage.gaussian_filter with manual reflect-padding.

    conv1d has a batch-size limit on Pascal (~32k); conv2d groups has no such
    limit, making this the correct approach for 512x512 signals.
    """
    cy, cx, KY, KX = ct.shape
    C = KY * KX
    # (cy, cx, KY, KX) -> (1, KY*KX, cy, cx)
    x = ct.permute(2, 3, 0, 1).reshape(1, C, cy, cx)
    # Reflect-pad indices (scipy 'reflect' = exclude-boundary mirror)
    iy = torch.cat([torch.arange(r, 0, -1), torch.arange(cy),
                    torch.arange(cy - 2, cy - 2 - r, -1)]).to(ct.device)
    ix = torch.cat([torch.arange(r, 0, -1), torch.arange(cx),
                    torch.arange(cx - 2, cx - 2 - r, -1)]).to(ct.device)
    # Blur H (cy) then W (cx)
    xh  = F.conv2d(x[:, :, iy, :], kH, groups=C, padding=0)   # (1,C,cy,cx)
    out = F.conv2d(xh[:, :, :, ix], kW, groups=C, padding=0)   # (1,C,cy,cx)
    return out.reshape(KY, KX, cy, cx).permute(2, 3, 0, 1).contiguous()


k1d, r = make_k1d(sigma, dev)
kH = k1d.view(1, 1, -1, 1).expand(KY * KX, 1, -1, 1).contiguous()
kW = k1d.view(1, 1, 1, -1).expand(KY * KX, 1, 1, -1).contiguous()

chunk_t = torch.from_numpy(chunk).to(dev)
out = blur_nav_gpu(chunk_t, kH, kW, r); torch.cuda.synchronize()

# Correctness: compare to manual pad + scipy convolve
k_np = k1d.cpu().numpy()
pad_y = np.pad(chunk, ((r, r), (0, 0), (0, 0), (0, 0)), mode='reflect')
cpu_blur_y = convolve1d(pad_y, k_np, axis=0, mode='constant', cval=0.0)[r:-r]
pad_x = np.pad(cpu_blur_y, ((0, 0), (r, r), (0, 0), (0, 0)), mode='reflect')
cpu_ref = convolve1d(pad_x, k_np, axis=1, mode='constant', cval=0.0)[:, r:-r]

diff = float(np.abs(cpu_ref - out.cpu().numpy()).max())
print(f'GPU blur max diff vs CPU (reflect pad + convolve): {diff:.5f}  (float32: <0.01 expected)')

# Timing
N_REP = 20
t0 = time.perf_counter()
for _ in range(5):
    pad_y = np.pad(chunk, ((r,r),(0,0),(0,0),(0,0)), mode='reflect')
    tmp = convolve1d(pad_y, k_np, axis=0, mode='constant', cval=0.0)[r:-r]
    pad_x = np.pad(tmp, ((0,0),(r,r),(0,0),(0,0)), mode='reflect')
    convolve1d(pad_x, k_np, axis=1, mode='constant', cval=0.0)[:,r:-r]
t_cpu_blur = (time.perf_counter() - t0) / 5

t0 = time.perf_counter()
for _ in range(N_REP):
    blur_nav_gpu(chunk_t, kH, kW, r)
    torch.cuda.synchronize()
t_gpu_blur = (time.perf_counter() - t0) / N_REP

print(f'CPU blur:  {t_cpu_blur * 1000:.0f} ms')
print(f'GPU blur:  {t_gpu_blur * 1000:.0f} ms  speedup={t_cpu_blur / t_gpu_blur:.1f}x')

# Full GPU pipeline
mod = _get_cuda_module()
disk_np = _make_disk_np(KR)
nd = disk_np.size; tm = float(disk_np.mean())
ts = float(np.sqrt(np.sum((disk_np - tm) ** 2) / nd))
mod.upload_disk(torch.from_numpy(disk_np.reshape(2 * KR + 1, 2 * KR + 1)), tm, ts)

chunk_pin = torch.from_numpy(chunk.copy()).pin_memory()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N_REP):
    ct = chunk_pin.to(dev, non_blocking=True)
    bl = blur_nav_gpu(ct, kH, kW, r)
    flat_t = bl.reshape(-1, KY, KX)
    fp = F.pad(flat_t.unsqueeze(1), (KR_WIN,) * 4, mode='reflect').squeeze(1)
    rc, pm = mod.nxcorr_forward(fp, KY, KX, KR, KR_WIN, 0.2, 28)
    torch.cuda.synchronize()
t_gpu_full = (time.perf_counter() - t0) / N_REP

n_pk = sum(len(p) for p in _greedy_nms_batch(rc.cpu().numpy(), pm.cpu().numpy(), 28))
print(f'GPU full (H2D+blur+NXCORR):  {t_gpu_full * 1000:.0f} ms  {n_pk} peaks')

t_read = 0.19; t_cpu_peaks = 5.4; n_chunks = 576
print()
print(f'Per-chunk:  read={t_read:.2f}s  cpu_blur={t_cpu_blur:.2f}s  cpu_peaks={t_cpu_peaks:.2f}s')
print(f'            gpu_blur={t_gpu_blur * 1000:.0f}ms  gpu_full={t_gpu_full:.2f}s')
bottleneck = "READ" if t_read > t_gpu_full else "GPU compute"
print(f'Bottleneck with GPU+blur: {bottleneck}  ({max(t_read, t_gpu_full):.2f}s/chunk)')
print(f'Full dataset: {max(t_read, t_gpu_full) * n_chunks / 60:.1f} min (ideal overlap)')
print(f'             {t_gpu_full * n_chunks / 60:.1f} min (GPU serial, no overlap)')

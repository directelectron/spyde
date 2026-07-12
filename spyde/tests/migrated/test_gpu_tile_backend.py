"""GpuTileBackend — SpyDE's GPU reducer for anyplotlib tile mode.

The default NumpyTileBackend area-means a 4096² frame on the CPU (~62 ms — the movie-
scrub wall). GpuTileBackend uploads the frame once and reduces overview/detail tiles on
the GPU (~10 ms swap+mean, ~2 ms sample-only), returning a small numpy array.

The numpy-fallback path (no CUDA) is tested in-process and must be byte-identical to
NumpyTileBackend. The real GPU correctness/perf is exercised in a SUBPROCESS because
torch-CUDA work segfaults under the pytest process on Windows (a harness interaction,
not a code bug — see CLAUDE.md); it's skipped when no CUDA device is present.
"""
import os
os.environ.setdefault("SPYDE_NO_DASK", "1")

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from spyde.drawing.plots.gpu_tile_backend import GpuTileBackend
from anyplotlib.plot2d._tile_backend import NumpyTileBackend, TileBackend


class TestProtocolAndFallback:
    """Runs on ANY machine (uses the numpy fallback when there's no GPU)."""

    def test_is_tile_backend(self):
        g = GpuTileBackend(np.zeros((2048, 2048), np.float32))
        assert isinstance(g, TileBackend)

    def test_geometry_matches_source(self):
        a = np.zeros((3000, 4096), np.uint16)
        g = GpuTileBackend(a, origin="lower")
        assert g.full_shape == (3000, 4096)
        assert g.dtype == np.uint16
        assert g.origin == "lower"
        assert g.extent() is None

    def test_mean_matches_numpy(self):
        rng = np.random.RandomState(0)
        a = rng.randint(0, 4000, (4096, 4096)).astype(np.uint16)
        g = GpuTileBackend(a)
        n = NumpyTileBackend(a)
        gm = g.sample(0, 4096, 0, 4096, 1024, 1024, "mean")
        nm = n.sample(0, 4096, 0, 4096, 1024, 1024, "mean")
        assert gm.shape == (1024, 1024)
        # GPU (adaptive_avg_pool) vs numpy reshape-mean agree to <1 code; on CPU
        # fallback they're byte-identical.
        assert np.allclose(gm, nm, atol=1.0)

    def test_subsample_matches_numpy(self):
        rng = np.random.RandomState(1)
        a = rng.randint(0, 255, (2048, 3000)).astype(np.uint8)
        g = GpuTileBackend(a)
        n = NumpyTileBackend(a)
        gs = g.sample(0, 3000, 0, 2048, 512, 512, "subsample")
        ns = n.sample(0, 3000, 0, 2048, 512, 512, "subsample")
        assert np.array_equal(gs, ns)

    def test_set_array_swaps_source(self):
        g = GpuTileBackend(np.zeros((2048, 2048), np.float32))
        g.set_array(np.full((2048, 2048), 5.0, np.float32))
        m = g.sample(0, 2048, 0, 2048, 64, 64, "mean")
        assert np.allclose(m, 5.0, atol=1e-3)

    def test_detail_region_upsample(self):
        # A deep-zoom detail tile: region SMALLER than the panel → native pixels,
        # nearest, matching numpy.
        rng = np.random.RandomState(2)
        a = rng.randint(0, 255, (4096, 4096)).astype(np.uint8)
        g = GpuTileBackend(a)
        n = NumpyTileBackend(a)
        # 100×100 native region → 274 output (upsample)
        gt = g.sample(1000, 1100, 1000, 1100, 274, 274, "mean")
        nt = n.sample(1000, 1100, 1000, 1100, 274, 274, "mean")
        assert gt.shape == (274, 274)
        assert np.allclose(gt, nt, atol=1.0)


# ── GPU correctness in a subprocess (torch-CUDA segfaults under pytest) ─────────
# Mirrors test_vector_orientation_gpu.py: RESULT_JSON prefix + flush + os._exit(0)
# so the parent reliably sees the result before the torch/CUDA teardown crash.
_GPU_PROBE = textwrap.dedent("""
    import json, os, sys, numpy as np
    os.environ['SPYDE_NO_DASK'] = '1'
    out = {}
    try:
        import torch
        out['cuda'] = bool(torch.cuda.is_available())
    except Exception as e:
        out['cuda'] = False; out['err'] = str(e)
    if out.get('cuda'):
        from spyde.drawing.plots.gpu_tile_backend import GpuTileBackend
        from anyplotlib.plot2d._tile_backend import NumpyTileBackend
        a = np.random.RandomState(0).randint(0, 4000, (4096, 4096)).astype(np.uint16)
        g = GpuTileBackend(a); n = NumpyTileBackend(a)
        out['active'] = g._torch is not None
        gm = g.sample(0, 4096, 0, 4096, 1024, 1024, "mean")
        nm = n.sample(0, 4096, 0, 4096, 1024, 1024, "mean")
        out['shape'] = list(gm.shape)
        out['maxdiff'] = float(np.abs(gm - nm).max())
    print("RESULT_JSON", json.dumps(out)); sys.stdout.flush()
    os._exit(0)
""")


def test_gpu_mean_correct_in_subprocess():
    import json
    proc = subprocess.run([sys.executable, "-c", _GPU_PROBE],
                          capture_output=True, text=True, timeout=180)
    line = next((l for l in proc.stdout.splitlines()
                 if l.startswith("RESULT_JSON")), None)
    assert line, (f"no RESULT_JSON (rc={proc.returncode})\n"
                  f"stdout={proc.stdout!r}\nstderr={proc.stderr[-600:]!r}")
    res = json.loads(line[len("RESULT_JSON "):])
    if not res.get("cuda"):
        pytest.skip(f"no GPU: {res.get('err', 'cuda unavailable')}")
    assert res["active"] is True, "GPU backend not active despite cuda"
    assert res["shape"] == [1024, 1024]
    assert res["maxdiff"] < 1.0, f"GPU mean diverged from numpy: {res['maxdiff']}"

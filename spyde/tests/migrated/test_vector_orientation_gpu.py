"""
Batched-GPU vector orientation fit (vector_orientation_gpu).

The GPU path packs the whole field onto the GPU and fits every pattern's pose
(theta, log-strain, beam-shift) in one batched torch optimisation — no dask,
no per-pattern Python loop.

Harness note: torch's CUDA autograd backward segfaults when run *inside* the
pytest process on Windows (a pytest/torch interaction — the same compute runs
correctly in plain Python and in the real app, alongside QApplication + Dask).
So ONE subprocess runs all three modes sequentially (each fresh subprocess
pays interpreter + torch import + cold CUDA init, ~5-7 s) and prints one
tagged ``RESULT_JSON <mode> {...}`` line per mode; the tests keep their
separate assertions on the parsed results.  The driver hard-exits after
printing, so partially-emitted results survive a teardown crash.
Skipped entirely when CUDA / torch GPU is unavailable.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest


def _cuda_available() -> bool:
    """CUDA specifically, NOT ``gpu_available()`` — that is True on Apple MPS
    too, and torch work under the pytest harness on the hosted macOS runners
    aborts the interpreter: SIGABRT immediately after these three subprocesses
    return their results, on main and on every branch since.  Same harness-
    interaction class as the Windows CUDA segfault this file's subprocess
    pattern exists for, and the same call the EBSD wizard tests make
    (``SPYDE_EBSD_DEVICE=cpu``).

    The Mac accelerator path keeps its real coverage: ``test_device_lock.py``
    pins that every torch call site serialises through the shared device lock,
    and the fit runs in the real app.  ``SPYDE_GPU_TESTS=1`` forces these on
    anyway (the escape hatch for reproducing on a Mac).
    """
    if os.environ.get("SPYDE_GPU_TESTS") == "1":
        from spyde.actions.vector_orientation_gpu import gpu_available
        return gpu_available()
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_available(),
    reason="needs CUDA (torch under pytest aborts on the macOS runners)")


# Driver script run ONCE in a subprocess. Builds synthetic vectors + a
# single-template library, runs the batched GPU fit for every mode given on
# argv, prints one tagged JSON summary line per mode.
_DRIVER = textwrap.dedent("""
    import json, sys
    import numpy as np

    # Prime cublasLt with a tiny F.linear BEFORE any other GPU work: on Pascal
    # (torch cu124) the FIRST cublasLt init that happens after cuDNN conv work
    # fails with CUBLAS_STATUS_NOT_INITIALIZED — a long multi-mode process is
    # exactly where that ordering can arise.
    import torch
    if torch.cuda.is_available():
        import torch.nn.functional as F
        F.linear(torch.zeros(1, 1, device="cuda"),
                 torch.zeros(1, 1, device="cuda"))
        torch.cuda.synchronize()
    from spyde.actions import vector_orientation as vo
    from spyde.actions.vector_orientation_gpu import compute_vector_orientation_gpu
    from spyde.signals.diffraction_vectors import (
        SpyDEDiffractionVectors, _build_nav_offsets, N_COLS,
        COL_NAV_X, COL_NAV_Y, COL_KX, COL_KY, COL_TIME, COL_INTENSITY)

    TEMPLATE = np.array([
        [0.05, 0.0], [-0.05, 0.0], [0.0, 0.05], [0.0, -0.05],
        [0.05, 0.05], [-0.05, -0.05], [0.05, -0.05], [-0.05, 0.05]],
        dtype=np.float32)

    def stub_library():
        return vo.TemplateLibrary(
            spots_xy=[TEMPLATE.copy()],
            spots_I=[np.ones(len(TEMPLATE), np.float32)],
            template_quats=np.array([[1.0, 0, 0, 0]]),
            template_phase=np.array([0], np.int16),
            phases_meta=[{"name": "x", "point_group": "m-3m"}],
            cache={}, radial_range=(0.0, 0.16), r_max=0.16)

    def make_vecs(strain=None, ny=4, nx=4):
        A = np.eye(2, dtype=np.float32) if strain is None else (np.eye(2, dtype=np.float32) + strain)
        spots = TEMPLATE @ A.T
        rows = []
        for iy in range(ny):
            for ix in range(nx):
                for kx, ky in spots:
                    r = np.zeros(N_COLS, np.float32)
                    r[COL_NAV_X]=ix; r[COL_NAV_Y]=iy; r[COL_KX]=kx; r[COL_KY]=ky
                    r[COL_TIME]=-1.0; r[COL_INTENSITY]=1.0
                    rows.append(r)
        flat = np.array(rows, np.float32)
        off = _build_nav_offsets(flat, (ny, nx))
        class Ax:
            scale=0.01; offset=-0.16; size=32; units="1/A"; name="k"
        return SpyDEDiffractionVectors(
            flat_buffer=flat, nav_offsets=off, nav_shape=(ny, nx),
            full_nav_shape=(ny, nx), sig_shape=(32, 32),
            sig_axes=[Ax(), Ax()], kernel_radius_px=3.0, kernel_radius_data=0.03)

    def run_mode(mode):
        out = {}
        if mode == "strain":
            E = np.array([[0.015, 0.005],[0.005,-0.010]], np.float32)
            res = compute_vector_orientation_gpu(
                make_vecs(strain=E), stub_library(),
                {"strain_cap":0.05,"sink_bw":0.04}, t=None)
            out["exx"]=float(np.nanmedian(res.strain[...,0]))
            out["eyy"]=float(np.nanmedian(res.strain[...,1]))
            out["exy"]=float(np.nanmedian(res.strain[...,2]))
            out["finite"]=bool(np.isfinite(res.strain[...,0]).all())
            out["nav"]=list(res.nav_shape)
        elif mode == "stop":
            res = compute_vector_orientation_gpu(
                make_vecs(), stub_library(), {"strain_cap":0.05,"sink_bw":0.04},
                t=None, stopped_flag=[True])
            out["is_none"]= res is None
        elif mode == "progress":
            # Use multiprocessing shared_memory directly — importing the GUI
            # helpers (spyde.drawing.update_functions) pulls in pyqtgraph/Qt, which
            # is unsafe to combine with torch CUDA in this same process.
            from multiprocessing import shared_memory
            sh = shared_memory.SharedMemory(create=True, size=4*4*12*4)
            buf = np.ndarray((4,4,12), np.float32, buffer=sh.buf); buf[:] = np.nan
            seen=[]
            res = compute_vector_orientation_gpu(
                make_vecs(), stub_library(), {"strain_cap":0.05,"sink_bw":0.04},
                t=None, progress=lambda d,t: seen.append((d,t)),
                shm_name=sh.name)
            out["reached_100"]= bool(seen and seen[-1][0]==seen[-1][1])
            out["buf_painted"]= bool(np.isfinite(buf[...,9]).any())
            out["not_none"]= res is not None
            sh.close(); sh.unlink()
        return out

    for mode in sys.argv[1:]:
        print("RESULT_JSON", mode, json.dumps(run_mode(mode)))
        sys.stdout.flush()
    # torch + CUDA + shared-memory teardown segfaults at interpreter exit on
    # Windows (harmless, post-result). Hard-exit so the parent sees rc==0.
    import os
    os._exit(0)
""")

_MODES = ("strain", "progress", "stop")


@pytest.fixture(scope="module")
def gpu_results():
    """Run the driver once for all modes; return {mode: parsed JSON dict}."""
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, *_MODES],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    results = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            _tag, mode, payload = line.split(" ", 2)
            results[mode] = json.loads(payload)
    missing = [m for m in _MODES if m not in results]
    assert not missing, (
        f"driver emitted no result for {missing}:\n{proc.stdout}\n{proc.stderr}")
    return results


def test_gpu_recovers_known_strain(gpu_results):
    out = gpu_results["strain"]
    assert out["nav"] == [4, 4]
    assert out["finite"]
    assert abs(out["exx"] - 0.015) < 5e-3, out
    assert abs(out["eyy"] - (-0.010)) < 5e-3, out
    assert abs(out["exy"] - 0.005) < 5e-3, out


def test_gpu_progress_and_shm_preview(gpu_results):
    out = gpu_results["progress"]
    assert out["not_none"]
    assert out["reached_100"]
    assert out["buf_painted"]


def test_gpu_stop_flag_aborts(gpu_results):
    out = gpu_results["stop"]
    assert out["is_none"]

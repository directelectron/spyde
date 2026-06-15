"""
Batched-GPU vector orientation fit (vector_orientation_gpu).

The GPU path packs the whole field onto the GPU and fits every pattern's pose
(theta, log-strain, beam-shift) in one batched torch optimisation — no dask,
no per-pattern Python loop.

Harness note: torch's CUDA autograd backward segfaults when run *inside* the
pytest process on Windows (a pytest/torch interaction — the same compute runs
correctly in plain Python and in the real app, alongside QApplication + Dask).
So each test runs the compute in a **subprocess** and checks its JSON result.
Skipped entirely when CUDA / torch GPU is unavailable.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from spyde.actions.vector_orientation_gpu import gpu_available

pytestmark = pytest.mark.skipif(
    not gpu_available(), reason="CUDA / torch GPU not available")


# Driver script body shared by the subprocesses. Builds synthetic vectors +
# a single-template library, runs the batched GPU fit, prints a JSON summary.
_DRIVER = textwrap.dedent("""
    import json, sys
    import numpy as np
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

    mode = sys.argv[1]
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

    print("RESULT_JSON", json.dumps(out))
    sys.stdout.flush()
    # torch + CUDA + shared-memory teardown segfaults at interpreter exit on
    # Windows (harmless, post-result). Hard-exit so the parent sees rc==0.
    import os
    os._exit(0)
""")


def _run(mode):
    """Run the driver in a subprocess; return the parsed JSON result dict."""
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, mode],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"subprocess failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT_JSON"))
    return json.loads(line[len("RESULT_JSON "):])


def test_gpu_recovers_known_strain():
    out = _run("strain")
    assert out["nav"] == [4, 4]
    assert out["finite"]
    assert abs(out["exx"] - 0.015) < 5e-3, out
    assert abs(out["eyy"] - (-0.010)) < 5e-3, out
    assert abs(out["exy"] - 0.005) < 5e-3, out


def test_gpu_progress_and_shm_preview():
    out = _run("progress")
    assert out["not_none"]
    assert out["reached_100"]
    assert out["buf_painted"]


def test_gpu_stop_flag_aborts():
    out = _run("stop")
    assert out["is_none"]

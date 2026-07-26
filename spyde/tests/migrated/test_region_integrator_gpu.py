"""GPU region accumulate — bit-parity with the CPU path.

The GPU accumulator is a second implementation of arithmetic that already had a
correct answer, so the ONLY thing that justifies it is producing exactly the same
frame. A region mean that is one count off would be invisible until it mattered,
so this asserts array_equal, never allclose.

Harness note (CLAUDE.md): torch-CUDA work segfaults *inside* the pytest process on
Windows — a pytest/torch interaction, not a code bug. So the compute runs in a
**subprocess** that prints JSON and ``os._exit(0)``s past torch's teardown. Skipped
when no CUDA device is present.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _cuda_available(), reason="CUDA / torch GPU not available")


_DRIVER = textwrap.dedent("""
    import json, os, sys
    import torch          # BEFORE spyde, so torch_cuda_ready() resolves inline
    import numpy as np

    os.environ["SPYDE_GPU_REGION"] = "1"
    from spyde.array_cache import ArrayCache, RegionIntegrator
    from spyde.array_cache.nav_read import _region_accum_dtype
    import spyde.array_cache.region_sum_gpu as rsg

    # Parity is what's under test, not the size heuristic — drop the minimum so
    # small (fast) frames still exercise the device path. The heuristic itself is
    # asserted in test_region_integrator.py, which needs no GPU.
    rsg.GPU_MIN_FRAME_BYTES = 0

    EDGE, W, STEPS = 512, 16, 6

    class FakeReader:
        def __init__(self, stack):
            self.stack = stack; self.data = stack
        @property
        def frame_bytes(self):
            return int(np.prod(self.stack.shape[1:])) * self.stack.dtype.itemsize
        def read_frame(self, indices):
            return self.stack[tuple(int(v) for v in indices)]

    def serial_mean(stack, pts):
        n = len(pts)
        acc_dtype = _region_accum_dtype(stack.dtype, n)
        acc = None
        for p in pts:
            f = stack[tuple(int(v) for v in p)]
            acc = np.asarray(f, dtype=acc_dtype).copy() if acc is None else acc + f
        acc = acc / n
        if np.issubdtype(stack.dtype, np.integer):
            acc = np.rint(acc).astype(stack.dtype)
        return acc

    def drive(stack, gpu):
        os.environ["SPYDE_GPU_REGION"] = "1" if gpu else "0"
        integ, cache = RegionIntegrator(), ArrayCache()
        cache.ensure_budget_for(W, FakeReader(stack).frame_bytes)
        reader = FakeReader(stack)
        acc_dt = _region_accum_dtype(stack.dtype, W)
        outs = []
        for k in range(STEPS):
            pts = [(k + i,) for i in range(W)]
            outs.append(integ.mean_frame("k", reader, cache, pts, stack.dtype,
                                         acc_dt, None))
        return outs, integ._gpu is not None

    rng = np.random.default_rng(0)
    res = {}

    # uint16 — the movie case; exercises full recompute THEN incremental slides.
    stack = rng.integers(0, 65535, (W + STEPS, EDGE, EDGE)).astype(np.uint16)
    gpu_out, used_gpu = drive(stack, True)
    cpu_out, _ = drive(stack, False)
    res["gpu_engaged"] = bool(used_gpu)
    res["u16_steps"] = len(gpu_out)
    res["u16_identical_to_cpu"] = all(
        np.array_equal(a, b) for a, b in zip(cpu_out, gpu_out))
    res["u16_identical_to_serial"] = all(
        np.array_equal(g, serial_mean(stack, [(k + i,) for i in range(W)]))
        for k, g in enumerate(gpu_out))
    res["u16_dtype"] = str(gpu_out[0].dtype)

    # saturated uint16 at the 16-point cap: n*max = 1,048,560, the float32 limit.
    sat = np.full((W + STEPS, 128, 128), 65535, np.uint16); sat[::2] = 65534
    g, _ = drive(sat, True)
    res["saturated_exact"] = all(
        np.array_equal(x, serial_mean(sat, [(k + i,) for i in range(W)]))
        for k, x in enumerate(g))

    # uint8 and int16 sources round-trip through the on-device cast.
    for name, dt in (("u8", np.uint8), ("i16", np.int16)):
        s = rng.integers(0, 100, (W + STEPS, 128, 128)).astype(dt)
        gg, _ = drive(s, True)
        res[name + "_identical"] = all(
            np.array_equal(x, serial_mean(s, [(k + i,) for i in range(W)]))
            for k, x in enumerate(gg))
        res[name + "_dtype"] = str(gg[0].dtype)

    # float64 accumulator (int32 source) must DECLINE the GPU — fp64 is a second
    # numerical regime we deliberately don't maintain on-device.
    i32 = rng.integers(0, 10 ** 6, (W + STEPS, 128, 128)).astype(np.int32)
    _, used = drive(i32, True)
    res["int32_declines_gpu"] = not used

    print("RESULT " + json.dumps(res))
    sys.stdout.flush()
    os._exit(0)
""")


@pytest.fixture(scope="module")
def gpu_result(tmp_path_factory):
    script = tmp_path_factory.mktemp("gpuregion") / "driver.py"
    # utf-8 explicitly: the driver has em-dashes in its comments and Windows'
    # default cp1252 write makes a file CPython then refuses to parse.
    script.write_text(_DRIVER, encoding="utf-8")
    env = dict(os.environ, SPYDE_NO_DASK="1")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                          text=True, timeout=900, env=env)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")),
                None)
    assert line, (f"driver produced no result\n"
                  f"--- stdout ---\n{proc.stdout[-3000:]}\n"
                  f"--- stderr ---\n{proc.stderr[-3000:]}")
    return json.loads(line[len("RESULT "):])


class TestGpuRegionParity:
    def test_the_gpu_path_actually_engaged(self, gpu_result):
        """Guards every other assertion here: if the accumulator silently declined,
        they would all pass by comparing the CPU path against itself."""
        assert gpu_result["gpu_engaged"] is True

    def test_matches_the_cpu_backend_exactly(self, gpu_result):
        assert gpu_result["u16_steps"] > 1
        assert gpu_result["u16_identical_to_cpu"] is True

    def test_matches_the_serial_reference_exactly(self, gpu_result):
        assert gpu_result["u16_identical_to_serial"] is True
        assert gpu_result["u16_dtype"] == "uint16"

    def test_saturated_uint16_stays_exact_at_the_cap(self, gpu_result):
        """float32 holds a 16-frame uint16 sum exactly (1,048,560 < 2**24) — the
        device subtract must not lose a count either."""
        assert gpu_result["saturated_exact"] is True

    @pytest.mark.parametrize("kind,dtype", [("u8", "uint8"), ("i16", "int16")])
    def test_other_integer_sources_round_trip(self, gpu_result, kind, dtype):
        assert gpu_result[kind + "_identical"] is True
        assert gpu_result[kind + "_dtype"] == dtype

    def test_float64_accumulator_declines_the_gpu(self, gpu_result):
        assert gpu_result["int32_declines_gpu"] is True

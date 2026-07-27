"""benchmark_fitting.py — batched engine vs HyperSpy multifit, at real scale.

Run DIRECTLY (not under pytest — torch-CUDA segfaults in the pytest process on
Windows, and this is slow by design)::

    uv run python -m spyde.tests.benchmark_fitting
    uv run python -m spyde.tests.benchmark_fitting --nav 128 --skip-hyperspy

The reference number this exists to beat, measured on the dev box: HyperSpy
``multifit`` on a 1024-channel EELS SI runs at ~110 spectra/s single-threaded,
i.e. ~10 minutes for a 256x256 spectrum image.

Reporting rules (CLAUDE.md, Benchmarking):

* real dataset at real scale — the synthetic EELS SI from ``spyde.data``, which
  is a power law plus three real core-loss edges, not a toy gaussian;
* ``torch.cuda.synchronize()`` around the timed region, since kernels are async;
* discard the first GPU run (cold CUDA init + kernel JIT is a one-time ~5 s);
* time each stage separately, so "it's slow" points at a stage rather than a
  vague total.
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _build(nav, n_channels):
    from spyde.data import eels_si
    s = eels_si(nav=(nav, nav), n_channels=n_channels)
    x = s.axes_manager.signal_axes[0].axis
    return s, np.asarray(x, float)


def _seed_model(signal):
    """Power-law background + one smeared STEP per edge.

    ``Erf`` (an error function) is the right shape here and a Gaussian is not:
    a core-loss edge is a step up at the onset, and a peak cannot represent it.
    A Gaussian-per-edge model is unfittable — measured, both HyperSpy and the
    batched engine exhaust their iteration budgets on it and the benchmark ends
    up timing the iteration cap instead of convergence. Without exspy's real
    ``EELSCLEdge`` (#63) this is the closest well-posed model.
    """
    from hyperspy.components1d import Erf, PowerLaw
    from spyde.data.synthetic import EELS_EDGES

    m = signal.create_model()
    comps = [PowerLaw()]
    for onset in EELS_EDGES.values():
        e = Erf()
        e.origin.value, e.sigma.value, e.A.value = onset, 4.0, 2e3
        comps.append(e)
    m.extend(comps)
    m[0].A.value, m[0].r.value = 1e8, 3.0
    m[0].origin.free = False
    return m


def _time(fn, *, sync=False):
    if sync:
        import torch
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if sync:
        import torch
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav", type=int, default=64,
                    help="navigation grid edge (nav x nav spectra)")
    ap.add_argument("--channels", type=int, default=1024)
    ap.add_argument("--skip-hyperspy", action="store_true",
                    help="skip the reference (it is the slow one)")
    args = ap.parse_args()

    from spyde.fitting import ModelSpec
    from spyde.fitting.engine import default_device, fit_batched

    P = args.nav * args.nav
    print(f"building {args.nav}x{args.nav} = {P} spectra x {args.channels} ch …")
    s, x = _build(args.nav, args.channels)
    spec = ModelSpec.from_model(_seed_model(s))
    n_free = int(spec.free_mask().sum())
    print(f"model: {[c.kind for c in spec]}  ({n_free} free parameters)\n")

    results = {}

    # -- reference ---------------------------------------------------------
    if not args.skip_hyperspy:
        m = _seed_model(s)
        print("hyperspy multifit … (this is the slow one)")
        _, dt = _time(lambda: m.multifit(optimizer="lm", show_progressbar=False))
        results["hyperspy multifit"] = dt
        print(f"  {dt:8.2f} s   {P/dt:8.1f} spectra/s\n")

    # -- batched, CPU ------------------------------------------------------
    print("batched engine (cpu) …")
    r_cpu, dt = _time(lambda: fit_batched(spec, s.data, x, device="cpu"))
    results["batched cpu"] = dt
    print(f"  {dt:8.2f} s   {P/dt:8.1f} spectra/s   "
          f"converged {r_cpu.convergence_rate:.1%}  iters {r_cpu.n_iter}\n")

    # -- batched, GPU ------------------------------------------------------
    if default_device() == "cuda":
        print("batched engine (cuda) … [discarding the cold run]")
        fit_batched(spec, s.data[:2, :2], x, device="cuda")      # warm up
        r_gpu, dt = _time(lambda: fit_batched(spec, s.data, x, device="cuda"),
                          sync=True)
        results["batched cuda"] = dt
        print(f"  {dt:8.2f} s   {P/dt:8.1f} spectra/s   "
              f"converged {r_gpu.convergence_rate:.1%}  iters {r_gpu.n_iter}")
        agree = np.nanmax(np.abs(r_gpu.values - r_cpu.values)
                          / np.maximum(np.abs(r_cpu.values), 1e-12))
        print(f"  max relative CPU/GPU disagreement: {agree:.2e}\n")
    else:
        print("no CUDA device — skipping the GPU leg\n")

    # -- summary -----------------------------------------------------------
    print("=" * 62)
    base = results.get("hyperspy multifit")
    for name, dt in results.items():
        line = f"{name:22s} {dt:8.2f} s   {P/dt:9.1f} spectra/s"
        if base and name != "hyperspy multifit":
            line += f"   {base/dt:7.1f}x"
        print(line)
    if base:
        for name, dt in results.items():
            if name == "hyperspy multifit":
                continue
            print(f"\n256x256 (65536 px) extrapolated: "
                  f"{name} {dt/P*65536:.1f} s  vs  hyperspy "
                  f"{base/P*65536/60:.1f} min")


if __name__ == "__main__":
    main()

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


def _build(nav, n_channels, dataset):
    """The two cases are DIFFERENT KINDS of benchmark and both are needed.

    ``eds`` is **well-posed**: the data is gaussian peaks on a smooth
    background, and the component library expresses gaussians exactly, so the
    fit can actually converge and the convergence rate is a real quality
    signal.

    ``eels`` is **deliberately hard**: real core-loss edges are tabulated GOS
    shapes, and nothing in the stock component set represents one, so the best
    available model (a smeared step) leaves structured residuals and neither
    HyperSpy nor the batched engine converges. Timing is still comparable —
    both do the same work — but do not read its convergence rate as a quality
    claim. That case is what exspy's real ``EELSCLEdge`` (#63) is for.
    """
    if dataset == "eels":
        from spyde.data import eels_si
        s = eels_si(nav=(nav, nav), n_channels=n_channels)
    else:
        from spyde.data import eds_si
        s = eds_si(nav=(nav, nav), n_channels=n_channels)
    x = s.axes_manager.signal_axes[0].axis
    return s, np.asarray(x, float)


def _seed_model(signal, dataset="eels"):
    """Background + one component per spectral feature.

    EELS gets ``Erf`` (a smeared step) per edge, because a core-loss edge is a
    step up at the onset and a peak cannot represent it — a Gaussian-per-edge
    model is simply unfittable, and measured, both HyperSpy and the batched
    engine burn their whole iteration budget on it.

    EDS gets a ``Gaussian`` per K-alpha line, which is exactly what the data
    is, so this one converges.
    """
    from hyperspy.components1d import Erf, Gaussian, PowerLaw

    if dataset == "eds":
        from spyde.data.synthetic import EDS_LINES
        m = signal.create_model()
        while len(m):
            m.remove(m[0])
        comps = [PowerLaw()]
        for lines in EDS_LINES.values():
            g = Gaussian()
            g.centre.value, g.sigma.value, g.A.value = lines[0][1], 0.09, 3e3
            comps.append(g)
        m.extend(comps)
        m[0].A.value, m[0].r.value = 1e4, 1.0
        m[0].origin.free = False
        return m

    from spyde.data.synthetic import EELS_EDGES

    m = signal.create_model()
    # create_model() PRE-POPULATES for a recognised signal type: with exspy
    # installed this is an EELSModel that already carries a background, so
    # appending our own silently produced a model with TWO PowerLaws — a
    # degenerate fit, and not the model the batched side was given. Both sides
    # must fit the SAME model or the comparison measures nothing.
    # (ModelSpec.to_model does the same clear, for the same reason.)
    while len(m):
        m.remove(m[0])
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
    ap.add_argument("--dataset", choices=("eels", "eds"), default="eds",
                    help="eds is well-posed (convergence is meaningful); "
                         "eels is deliberately hard (timing only)")
    ap.add_argument("--skip-hyperspy", action="store_true",
                    help="skip the reference (it is the slow one)")
    args = ap.parse_args()

    from spyde.fitting import ModelSpec
    from spyde.fitting.engine import default_device, fit_batched

    P = args.nav * args.nav
    print(f"building {args.nav}x{args.nav} = {P} spectra x {args.channels} ch …")
    s, x = _build(args.nav, args.channels, args.dataset)
    spec = ModelSpec.from_model(_seed_model(s, args.dataset))
    n_free = int(spec.free_mask().sum())
    print(f"model: {[c.kind for c in spec]}  ({n_free} free parameters)\n")

    results = {}

    # -- reference ---------------------------------------------------------
    if not args.skip_hyperspy:
        m = _seed_model(s, args.dataset)
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

    # -- seeded (coarse -> propagate -> refine) ----------------------------
    from spyde.fitting.seeding import fit_seeded
    dev = default_device()
    print(f"seeded ({dev}) …")
    r_seed, dt = _time(lambda: fit_seeded(spec, s.data, x, stride=4,
                                          device=dev),
                       sync=(dev == "cuda"))
    results[f"seeded {dev}"] = dt
    print(f"  {dt:8.2f} s   {P/dt:8.1f} spectra/s   "
          f"converged {r_seed.convergence_rate:.1%}  "
          f"(seeds {r_seed.seed_converged:.0%} of {r_seed.n_seeds})\n")

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

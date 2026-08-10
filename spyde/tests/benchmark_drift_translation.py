"""
benchmark_drift_translation.py — rigid drift solve, OLD vs NEW, at real scale.

Run it directly, never under pytest (CLAUDE.md § Benchmarking)::

    python -m spyde.tests.benchmark_drift_translation
    python -m spyde.tests.benchmark_drift_translation --size 4096 --frames 300
    python -m spyde.tests.benchmark_drift_translation --backends numpy,cpu,cuda

**Why the movie is synthesised rather than downloaded.** The thing being measured
is frames/s at 2048²-4096² over hundreds of frames — tens of GB. Generating it
means the ground-truth drift is EXACT, so accuracy and speed come out of the same
run and a speed win that costs accuracy is visible in the same table. That is the
whole point: the previous solver had no benchmark at all, so "284 frames/s at
512²" was the only number anyone had, and it said nothing about what a 4096²
movie costs.

**The movie is a real file, read one frame at a time.** It is written once to a
uint16 memmap in the OS temp directory (reused on later runs, keyed by its
parameters) and handed to the solvers as a ``np.memmap``, so the streaming read
path is exercised and nothing ever holds the stack. A 300 × 4096² movie is 10 GB
on disk and a few hundred MB resident — which is exactly the case the
Memory-Safety rule exists for.

**The synthetic scene is noise-realistic, and that matters more than it sounds.**
Frames carry INDEPENDENT Poisson noise at a chosen dose over a 1/f-ish micrograph
texture. The unit-test fixtures in ``test_drift_translation.py`` shift ONE base
image, so their pixel noise is identical in every frame — a perfect
high-frequency fiducial that no real movie has, and one that flatters any
correlation weighting that leans on the top of the band. Tuning the filter on
those fixtures alone is over-fitting; the ``--dose`` sweep here is the check.

The OLD solver is loaded from git rather than vendored (``--old-ref``), so this
file cannot drift out of sync with what it claims to compare against.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import types

import numpy as np

# The branch point this work started from: the last commit with the phase-
# correlation / running-reference solver.
DEFAULT_OLD_REF = "origin/feat/drift-rigid"


# ── the movie ────────────────────────────────────────────────────────────────

def _base_image(n: int, seed: int = 0, feature_px: float = 6.0) -> np.ndarray:
    """A micrograph-like scene: 1/f^1.5 texture plus a few bright particles."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    q = np.hypot(f[:, None], f[None, :])
    q[0, 0] = 1e-6
    amp = q ** -1.5 * np.exp(-(q * feature_px) ** 2)
    ph = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    img = np.real(np.fft.ifft2(np.fft.fft2(ph) * amp))
    img -= img.min()
    img /= max(img.max(), 1e-12)
    yy, xx = np.mgrid[0:n, 0:n]
    for _ in range(16):
        cy, cx = rng.uniform(0.1, 0.9, 2) * n
        s = rng.uniform(3, 12)
        img += 0.8 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))
    return (img / img.max()).astype(np.float64)


def _truth_shifts(n_frames: int, amp: float, seed: int = 1) -> np.ndarray:
    """A smooth stage creep plus a slow random walk — the shape a stage makes."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_frames)
    dy = amp * t ** 1.3 + 0.05 * amp * np.sin(6.0 * t)
    dx = -0.6 * amp * t + 0.08 * amp * np.sin(4.0 * t + 1.0)
    dy += 0.1 * rng.standard_normal(n_frames).cumsum()
    dx += 0.1 * rng.standard_normal(n_frames).cumsum()
    s = np.stack([dy, dx], 1)
    return s - s[0]


def make_movie(size: int, frames: int, dose: float, amp: float, seed: int = 0):
    """Write (or reuse) the movie as a uint16 memmap. Returns ``(memmap, truth)``.

    Frame *i* is the base image translated by ``-truth[i]`` (so ``+truth[i]`` is
    the correction, matching ``DriftModel``) and then sampled with independent
    Poisson noise at *dose* electrons per pixel.
    """
    key = hashlib.sha1(
        f"{size}-{frames}-{dose}-{amp}-{seed}-v1".encode()).hexdigest()[:12]
    path = os.path.join(tempfile.gettempdir(), f"spyde-drift-bench-{key}.raw")
    truth = _truth_shifts(frames, amp)
    if os.path.exists(path) and os.path.getsize(path) == frames * size * size * 2:
        return np.memmap(path, np.uint16, "r", shape=(frames, size, size)), truth

    print(f"  synthesising {frames} x {size}^2 uint16 "
          f"({frames * size * size * 2 / 1e9:.1f} GB) -> {path}")
    base = _base_image(size, seed)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    F = np.fft.fft2(base)
    rng = np.random.default_rng(seed + 99)
    # Build under a .partial name and rename at the end. `np.memmap(mode="w+")`
    # allocates the WHOLE file up front, so a synthesis interrupted half-way
    # leaves a file of exactly the right size whose tail is zeros — and the reuse
    # check above is a size check, so the next run would benchmark against a movie
    # that is half black and report it as real. A 4 GB synthesis takes ten minutes;
    # interrupting one is not a hypothetical.
    tmp = path + ".partial"
    mm = np.memmap(tmp, np.uint16, "w+", shape=(frames, size, size))
    t0 = time.perf_counter()
    try:
        for i, (dy, dx) in enumerate(truth):
            img = np.real(np.fft.ifft2(F * np.exp(-2j * np.pi * (-dy * fy + -dx * fx))))
            np.clip(img, 0.0, None, out=img)
            mm[i] = np.minimum(rng.poisson(dose * img), 65535).astype(np.uint16)
            if (i + 1) % 25 == 0:
                print(f"    {i + 1}/{frames}  {time.perf_counter() - t0:.0f}s")
        mm.flush()
    finally:
        del mm
    os.replace(tmp, path)
    return np.memmap(path, np.uint16, "r", shape=(frames, size, size)), truth


# ── the old solver, loaded from git ──────────────────────────────────────────

def load_old_solver(ref: str):
    """Import the pre-rewrite ``translation.py`` from *ref* as a module.

    Vendoring a copy would rot; reading it out of git cannot. Returns None (with
    a printed reason) if the ref is unavailable, so the benchmark still runs as a
    single-solver profile on a checkout without the history.
    """
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        src = subprocess.run(
            ["git", "show", f"{ref}:spyde/drift/translation.py"],
            cwd=here, capture_output=True, check=True).stdout.decode("utf-8")
    except Exception as exc:
        print(f"  [skip] old solver at {ref!r} unavailable: {exc}")
        return None
    mod = types.ModuleType("spyde_drift_translation_legacy")
    mod.__file__ = f"<git:{ref}:spyde/drift/translation.py>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


# ── timing ───────────────────────────────────────────────────────────────────

def _sync(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch
        torch.cuda.synchronize()
    except Exception:
        pass


def time_solve(fn, device: str, runs: int = 2):
    """Run *fn* *runs* times. Returns ``(cold_s, warm_s, model)``.

    The first run carries cold CUDA init and kernel JIT (a one-time ~5 s), so it
    is reported separately rather than averaged in — CLAUDE.md § Benchmarking.
    """
    times = []
    model = None
    for _ in range(max(1, runs)):
        _sync(device)
        t0 = time.perf_counter()
        model = fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times[0], min(times[1:]) if len(times) > 1 else times[0], model


def read_throughput(data, n: int = 40) -> float:
    """Frames/s for the READ alone — the floor no solver can beat.

    **This is a WARM number and is reported as such.** The movie has just been
    written (or read by a previous run), so it is in the OS page cache and this
    measures RAM, not disk. Purging it would make the figure honest about cold
    I/O but would also make every solver timing below a disk benchmark instead of
    a compute one, which is not what this file is for. Both solvers see the same
    warm cache, so the COMPARISON is sound; treat the absolute GB/s as an upper
    bound.
    """
    from spyde.drift.frames import frame_source
    n_frames, get_frame, _ = frame_source(data)
    idx = np.linspace(0, n_frames - 1, min(n, n_frames)).round().astype(int)
    t0 = time.perf_counter()
    acc = 0.0
    for i in idx:
        acc += float(get_frame(int(i))[0, 0])
    dt = time.perf_counter() - t0
    return len(idx) / dt if dt > 0 else float("nan")


def accuracy(model, truth) -> dict:
    got = np.asarray(model.shifts, np.float64)
    ok = np.isfinite(got).all(axis=1)
    err = got[ok] - truth[ok]
    r = np.hypot(err[:, 0], err[:, 1])
    return {"rms": float(np.sqrt(np.mean(r ** 2))),
            "p95": float(np.percentile(r, 95)),
            "max": float(np.max(r))}


# ── main ─────────────────────────────────────────────────────────────────────

def available_backends(requested: str) -> list[str]:
    want = [b.strip() for b in requested.split(",") if b.strip()]
    out = []
    for b in want:
        if b == "numpy":
            out.append(b)
            continue
        try:
            import torch
        except Exception:
            print(f"  [skip] backend {b!r}: torch not importable")
            continue
        if b == "cuda" and not torch.cuda.is_available():
            print("  [skip] backend 'cuda': no CUDA device")
            continue
        if b == "mps" and not getattr(torch.backends, "mps",
                                      types.SimpleNamespace(is_available=lambda: False)
                                      ).is_available():
            print("  [skip] backend 'mps': unavailable")
            continue
        out.append(b)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=2048, help="frame edge, px")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--dose", type=float, default=40.0,
                    help="electrons/px; sets the Poisson SNR")
    ap.add_argument("--drift", type=float, default=48.0,
                    help="total drift excursion, px")
    ap.add_argument("--backends", default="numpy,cpu,cuda")
    ap.add_argument("--runs", type=int, default=2, help="timed runs per solver")
    # None = whatever the solver ships as its default. A hard-coded number
    # here would silently benchmark a configuration nobody runs, which is
    # exactly what happened the first time this was used in anger.
    ap.add_argument("--band", type=int, default=None)
    ap.add_argument("--upsample", type=int, default=8)
    ap.add_argument("--max-shift", type=float, default=96.0,
                    help="must cover the WHOLE excursion for the old solver "
                         "(it registers against a reference); the new one only "
                         "needs the within-band relative shift")
    ap.add_argument("--old-ref", default=DEFAULT_OLD_REF)
    ap.add_argument("--skip-old", action="store_true")
    args = ap.parse_args(argv)

    print(f"drift benchmark - {args.frames} x {args.size}^2 uint16, "
          f"dose {args.dose:g} e/px, drift {args.drift:g} px")
    data, truth = make_movie(args.size, args.frames, args.dose, args.drift)
    print(f"  read floor: {read_throughput(data):.1f} frames/s "
          f"({read_throughput(data) * args.size ** 2 * 2 / 1e9:.2f} GB/s)")

    from spyde.drift.translation import solve_translation as solve_new
    old = None if args.skip_old else load_old_solver(args.old_ref)

    backends = available_backends(args.backends)
    rows = []
    for device in backends:
        band_kw = {} if args.band is None else {"band": int(args.band)}

        def _new(dev=device):
            return solve_new(data, device=dev, upsample=args.upsample,
                             max_shift=args.max_shift, **band_kw)
        cold, warm, model = time_solve(_new, device, args.runs)
        rows.append(("new", device, cold, warm, accuracy(model, truth),
                     dict(model.params)))
        print(f"  new/{device:5s} cold {cold:7.2f}s  warm {warm:7.2f}s  "
              f"{args.frames / warm:7.1f} f/s")

        if old is not None:
            def _old(dev=device):
                return old.solve_translation(
                    data, device=dev, upsample=args.upsample,
                    max_shift=args.max_shift, reference="running")
            cold, warm, model = time_solve(_old, device, args.runs)
            rows.append(("old", device, cold, warm, accuracy(model, truth), {}))
            print(f"  old/{device:5s} cold {cold:7.2f}s  warm {warm:7.2f}s  "
                  f"{args.frames / warm:7.1f} f/s")

    print()
    print(f"{'solver':>6} {'device':>7} {'cold s':>8} {'warm s':>8} {'frames/s':>9} "
          f"{'rms px':>8} {'p95 px':>8} {'max px':>8}")
    for name, device, cold, warm, acc, params in rows:
        print(f"{name:>6} {device:>7} {cold:8.2f} {warm:8.2f} "
              f"{args.frames / warm:9.1f} "
              f"{acc['rms']:8.3f} {acc['p95']:8.3f} {acc['max']:8.3f}")

    new_params = next((p for n, _d, _c, _w, _a, p in rows if n == "new" and p), {})
    if new_params:
        print()
        print(f"  new solver: correlation grid {new_params['corr_grid']} "
              f"(bin {new_params['corr_bin']}), band {new_params['band']}, "
              f"{new_params['n_pairs']} pairs, "
              f"{new_params['pairs_downweighted']} down-weighted, "
              f"LS residual {new_params['ls_residual_px']:.3f} px "
              f"(max {new_params['ls_residual_max_px']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

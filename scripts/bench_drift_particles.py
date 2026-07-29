#!/usr/bin/env python
"""
The drift + particles numbers, in one run.

    python scripts/bench_drift_particles.py

Prints a table suitable for pasting into ``benchmarks.md``. Every number here is a
number some decision in ``DRIFT_AND_PARTICLES_PLAN.md`` rests on, so re-run it
after touching the solver or the feature stack rather than trusting the recorded
values — they were measured on one machine on one day.

Deliberately NOT a pytest: these are slow, machine-dependent, and a benchmark that
fails CI because a runner was busy teaches nothing. The tests assert CORRECTNESS;
this reports COST.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _t(fn, repeat: int = 3):
    """Best of *repeat*, discarding the first run (cold CUDA init / kernel JIT)."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_drift_backends(rows: list[str]) -> None:
    from spyde.drift.translation import solve_translation

    rng = np.random.default_rng(0)
    for (nf, n) in [(60, 256), (60, 512)]:
        base = rng.standard_normal((n, n)).astype(np.float32)
        stack = np.repeat(base[None], nf, axis=0)
        rows.append(f"\n### {nf} frames x {n}^2, upsample=8")
        rows.append("")
        rows.append("| backend | time | frames/s |")
        rows.append("|---|---|---|")
        for dev in ("numpy", "cpu", "cuda"):
            try:
                dt = _t(lambda d=dev: solve_translation(stack, device=d, upsample=8),
                        repeat=2)
                rows.append(f"| {dev} | {dt:.2f} s | {nf / dt:.0f} |")
            except Exception as exc:
                rows.append(f"| {dev} | unavailable | {type(exc).__name__} |")


def bench_drift_accuracy(rows: list[str]) -> None:
    """Error vs upsample, on truth deliberately OFF the 1/upsample grid.

    On-grid truth is recovered exactly at any upsample, which looks superb and
    tests nothing — that mistake hid a real bug in the refinement once already.
    """
    from spyde.drift.translation import solve_translation
    from spyde.tests.migrated.test_drift_translation import _shifted_stack

    truth = np.array([[0, 0], [1.37, -2.83], [-3.06, 0.61], [4.19, 5.44],
                      [-0.72, -1.28]])
    stack = _shifted_stack(truth)
    rows.append("\n### Sub-pixel accuracy vs upsample (off-grid truth)")
    rows.append("")
    rows.append("| upsample | max error |")
    rows.append("|---|---|")
    for u in (1, 2, 8, 32, 64):
        m = solve_translation(stack, device="numpy", upsample=u, reference="first")
        rows.append(f"| {u} | {np.abs(m.shifts - truth).max():.3f} px |")


def bench_fixture(rows: list[str]) -> None:
    import spyde.data.synthetic as sy
    from spyde.drift.translation import solve_translation

    rows.append("\n### Synthetic particle-movie fixture")
    rows.append("")
    build = _t(lambda: sy.particle_movie(), repeat=2)
    s = sy.particle_movie()
    gt = sy.ground_truth(s)
    truth = np.asarray(gt["drift"])
    m = solve_translation(s.data, device="numpy", upsample=8, reference="first",
                          max_shift=20)
    err = np.abs(m.shifts - truth)
    solve = _t(lambda: solve_translation(s.data, device="numpy", upsample=8,
                                         reference="first", max_shift=20), repeat=2)
    rows.append("| stage | value |")
    rows.append("|---|---|")
    rows.append(f"| build (24 x 96x112) | {build * 1e3:.0f} ms |")
    rows.append(f"| drift solve | {solve * 1e3:.0f} ms |")
    rows.append(f"| drift error (max / mean) | {err.max():.3f} / {err.mean():.3f} px |")
    rows.append(f"| frames rejected from reference | "
                f"{m.params['rejected_from_reference']} |")


def bench_segment(rows: list[str]) -> None:
    import spyde.data.synthetic as sy
    from spyde.particles import SegmentParams, measure_frame, segment_frame

    s = sy.particle_movie()
    frame = s.data[12]
    p = SegmentParams(min_size=25, gaussian=1.0)
    seg = _t(lambda: segment_frame(frame, p))
    labels = segment_frame(frame, p)
    meas = _t(lambda: measure_frame(labels, frame, t=12, scale=0.5))
    rows.append("\n### Classical segment + measure, one 96x112 frame")
    rows.append("")
    rows.append("| stage | time | frames/s |")
    rows.append("|---|---|---|")
    rows.append(f"| segment_frame | {seg * 1e3:.1f} ms | {1 / seg:.0f} |")
    rows.append(f"| measure_frame | {meas * 1e3:.1f} ms | {1 / meas:.0f} |")
    rows.append(f"| combined | {(seg + meas) * 1e3:.1f} ms | {1 / (seg + meas):.0f} |")
    rows.append("")
    rows.append(f"Extrapolated to 3000 frames: **{3000 * (seg + meas):.0f} s** "
                f"single-threaded. The plan's target is minutes, so this is the "
                f"number to watch as frame size grows (cost is per-pixel, and a "
                f"4096^2 frame is 1600x this one's area).")


def bench_optional(rows: list[str]) -> None:
    """Stages whose modules may not have landed yet — reported as absent, not fatal."""
    try:
        from spyde.particles import features  # noqa: F401
    except Exception:
        rows.append("\n### Feature stack — not implemented yet")
        return
    import spyde.data.synthetic as sy
    from spyde.particles.features import FeatureSpec, compute_features

    s = sy.particle_movie()
    frame = s.data[12]
    spec = FeatureSpec()
    dt = _t(lambda: compute_features(frame, spec, device="cpu"))
    rows.append("\n### Torch feature stack, one 96x112 frame (CPU)")
    rows.append("")
    rows.append(f"| compute_features | {dt * 1e3:.1f} ms |")
    rows.append("|---|---|")


def main() -> int:
    rows: list[str] = ["# drift + particles benchmark", ""]
    for fn in (bench_drift_accuracy, bench_drift_backends, bench_fixture,
               bench_segment, bench_optional):
        try:
            fn(rows)
        except Exception as exc:
            rows.append(f"\n### {fn.__name__} FAILED: {type(exc).__name__}: {exc}")
    text = "\n".join(rows) + "\n"
    sys.stdout.write(text)
    sys.stdout.flush()
    # torch/CUDA teardown can crash the interpreter on Windows after CUDA work
    # (CLAUDE.md); the output is already flushed, so leave immediately.
    os._exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
One command that verifies the whole drift + particles feature.

(Named check_ rather than verify_ because .gitignore carries a `verify_*.py` rule
for throwaway ad-hoc scripts, which silently swallowed this file for several
commits — it ran fine locally and simply was not in the repo.)

    python scripts/check_drift_particles.py            # python only (fast)
    python scripts/check_drift_particles.py --all      # + typecheck + e2e
    python scripts/check_drift_particles.py --e2e      # + e2e only
    python scripts/check_drift_particles.py --bench     # + report the numbers

Exists because this feature spans two languages, three test tiers and a separate
repo, so "did I break it" was otherwise four commands with four different working
directories and one of them needed a build step first. Re-run it after every step
of DRIFT_AND_PARTICLES_PLAN.md.

Every stage is INDEPENDENT and the exit code is the worst result, so one broken
tier still reports the state of the others — the point is a status board, not a
fail-fast gate. Stages that cannot run (no node_modules, no browser) are reported
as SKIP, not as failure: a missing optional tool is not a regression.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ELECTRON = ROOT / "electron"

# The python suites this feature owns. Kept explicit rather than globbing
# `spyde/tests/migrated` so a new unrelated failure elsewhere does not read as a
# drift/particles regression — the full suite is a separate stage below.
FEATURE_SUITES = [
    "spyde/tests/migrated/test_drift_translation.py",
    "spyde/tests/migrated/test_particles_core.py",
    "spyde/tests/migrated/test_particle_movie_fixture.py",
    "spyde/tests/migrated/test_particles_scribble.py",
    "spyde/tests/migrated/test_particles_track.py",
    "spyde/tests/migrated/test_particle_tree.py",
    "spyde/tests/migrated/test_particles_wizard.py",
    "spyde/tests/migrated/test_drift_wizard.py",
    "spyde/tests/migrated/test_particle_overlay.py",
    "spyde/tests/migrated/test_particle_lifecycle.py",
]

# Playwright specs this feature owns.
FEATURE_SPECS = [
    "tests/particles_workflow.spec.ts",
    "tests/drift_workflow.spec.ts",
]

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_C = {PASS: "\033[32m", FAIL: "\033[31m", SKIP: "\033[33m"}
_R = "\033[0m"


def _colour(state: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return state
    return f"{_C.get(state, '')}{state}{_R}"


def _python() -> str:
    """The venv interpreter if there is one, else whatever is running us."""
    for rel in ("Scripts/python.exe", "bin/python"):
        cand = ROOT / ".venv" / rel
        if cand.exists():
            return str(cand)
    return sys.executable


class Runner:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.results: list[tuple[str, str, float, str]] = []

    def stage(self, name: str, cmd: list[str], *, cwd: Path = ROOT,
              skip_if: str | None = None, env: dict[str, str] | None = None) -> str:
        if skip_if:
            self.results.append((name, SKIP, 0.0, skip_if))
            print(f"  {_colour(SKIP)}  {name}  ({skip_if})")
            return SKIP
        print(f"  ....  {name}", end="\r", flush=True)
        full_env = {**os.environ, **(env or {})}
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(cwd), env=full_env,
                              capture_output=not self.verbose, text=True)
        dt = time.perf_counter() - t0
        state = PASS if proc.returncode == 0 else FAIL
        detail = ""
        if state is FAIL and not self.verbose:
            tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
            detail = "\n".join("      " + ln for ln in tail[-25:])
        self.results.append((name, state, dt, detail))
        print(f"  {_colour(state)}  {name}  ({dt:.1f}s)      ")
        if detail:
            print(detail)
        return state

    def summary(self) -> int:
        print("\n" + "=" * 66)
        worst = 0
        for name, state, dt, _ in self.results:
            print(f"  {_colour(state)}  {name:<44} {dt:>6.1f}s")
            if state is FAIL:
                worst = 1
        n_fail = sum(1 for _, s, _, _ in self.results if s is FAIL)
        n_skip = sum(1 for _, s, _, _ in self.results if s is SKIP)
        n_pass = sum(1 for _, s, _, _ in self.results if s is PASS)
        print("=" * 66)
        print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
        return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="every stage")
    ap.add_argument("--e2e", action="store_true", help="include Playwright specs")
    ap.add_argument("--typecheck", action="store_true", help="include tsc")
    ap.add_argument("--full-suite", action="store_true",
                    help="the whole python suite, not just this feature's")
    ap.add_argument("--bench", action="store_true",
                    help="print the feature's benchmark numbers")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="stream child output instead of capturing it")
    args = ap.parse_args()
    if args.all:
        args.e2e = args.typecheck = args.full_suite = args.bench = True

    py = _python()
    r = Runner(args.verbose)
    print(f"\nverifying drift + particles\n  python: {py}\n")

    # ── python ───────────────────────────────────────────────────────────────
    existing = [s for s in FEATURE_SUITES if (ROOT / s).exists()]
    missing = [s for s in FEATURE_SUITES if not (ROOT / s).exists()]
    if existing:
        r.stage("python  feature suites",
                [py, "-m", "pytest", *existing, "-q", "--no-header",
                 "-p", "no:cacheprovider"],
                env={"SPYDE_NO_DASK": "1"})
    for s in missing:
        # Not yet written — a step of the plan that has not landed. Visible, not fatal.
        r.stage(f"python  {Path(s).stem}", [], skip_if="not implemented yet")

    if args.full_suite:
        r.stage("python  full suite",
                [py, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                env={"SPYDE_NO_DASK": "1"})

    # ── frontend ─────────────────────────────────────────────────────────────
    has_npm = shutil.which("npm") is not None
    has_modules = (ELECTRON / "node_modules").is_dir()
    front_skip = (None if (has_npm and has_modules)
                  else "no npm" if not has_npm else "run npm install in electron/")

    if args.typecheck:
        r.stage("frontend typecheck", ["npm", "run", "typecheck"],
                cwd=ELECTRON, skip_if=front_skip)

    if args.e2e:
        specs = [s for s in FEATURE_SPECS if (ELECTRON / s).exists()]
        if front_skip:
            r.stage("e2e  feature specs", [], skip_if=front_skip)
        elif not specs:
            r.stage("e2e  feature specs", [], skip_if="no specs written yet")
        else:
            # test:build, not test — the harness launches out/main/index.js, so a
            # renderer change that was never built is invisible to Playwright and
            # the spec silently tests the previous bundle.
            r.stage("e2e  feature specs",
                    ["npm", "run", "test:build", "--", "--project=electron",
                     "--reporter=line", "--retries=0", *specs],
                    cwd=ELECTRON)

    # ── numbers ──────────────────────────────────────────────────────────────
    if args.bench:
        r.stage("bench  drift + fixture",
                [py, str(ROOT / "scripts" / "bench_drift_particles.py")],
                skip_if=None if (ROOT / "scripts" / "bench_drift_particles.py").exists()
                else "bench script not written yet")

    return r.summary()


if __name__ == "__main__":
    sys.exit(main())

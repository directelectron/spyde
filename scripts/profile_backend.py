"""Find the live SpyDE backend and dump every thread's stack with py-spy.

For diagnosing a STALL — the app is up but nothing is happening. A sampling
dump shows what each thread is actually executing, including native frames,
which is the one thing timestamps in a log cannot tell you. Run it WHILE the
stall is happening; a dump afterwards shows an idle process.

    python scripts/profile_backend.py              # list candidates, dump the live one
    python scripts/profile_backend.py --list       # just list, do not dump
    python scripts/profile_backend.py --pid 62280  # dump a specific pid
    python scripts/profile_backend.py --record 30  # 30 s flamegraph -> profile.svg

Picking the right process is the fiddly part and is why this exists: the venv
path contains "spyde", so every dask worker matches a naive name filter too.
The backend is the one running ``-m spyde`` with a real working set — the
0-byte ones are dead husks from previous runs.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def _pyspy() -> str:
    """The py-spy executable, preferring this project's venv."""
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(here, ".venv", "Scripts", "py-spy.exe"),
                 os.path.join(here, ".venv", "bin", "py-spy")):
        if os.path.exists(cand):
            return cand
    found = shutil.which("py-spy")
    if not found:
        sys.exit("py-spy not installed. Run:  uv pip install py-spy")
    return found


def _candidates() -> list[dict]:
    """Live `-m spyde` backends, biggest working set first.

    Uses WMI through PowerShell rather than psutil so this works even if the
    venv is mid-install; the command line is what distinguishes a backend from
    a dask worker (both are python.exe under a path containing "spyde").
    """
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
        "Where-Object { $_.CommandLine -match '-m\\s+spyde|spyde\\\\__main__' } | "
        "ForEach-Object { '{0}|{1}|{2}' -f $_.ProcessId, "
        "[math]::Round($_.WorkingSetSize/1MB), $_.ThreadCount }"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception as exc:
        sys.exit(f"could not enumerate processes: {exc}")
    rows = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3 and parts[0].isdigit():
            rows.append({"pid": int(parts[0]), "mb": int(parts[1]),
                         "threads": int(parts[2])})
    return sorted(rows, key=lambda r: -r["mb"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, help="dump this pid instead of guessing")
    ap.add_argument("--list", action="store_true", help="list candidates only")
    ap.add_argument("--record", type=int, metavar="SECONDS",
                    help="record a flamegraph for N seconds -> profile.svg")
    args = ap.parse_args()

    pyspy = _pyspy()
    rows = _candidates()
    if not rows:
        sys.exit("no running `-m spyde` backend found — is the app open?")

    print("SpyDE backend candidates (biggest working set first):")
    for r in rows:
        note = "  <-- almost certainly the live one" if r is rows[0] and r["mb"] > 200 else (
            "  (dead husk from an earlier run)" if r["mb"] < 50 else "")
        print(f"  pid {r['pid']:>7}   {r['mb']:>7} MB   {r['threads']:>3} threads{note}")
    if args.list:
        return

    pid = args.pid or rows[0]["pid"]
    if rows[0]["mb"] < 50 and not args.pid:
        sys.exit("\nthe largest candidate is under 50 MB — they all look dead. "
                 "Open the app, then re-run.")

    if args.record:
        cmd = [pyspy, "record", "-p", str(pid), "-d", str(args.record),
               "-o", "profile.svg", "--subprocesses", "--idle"]
        print(f"\nrecording {args.record}s from pid {pid} -> profile.svg")
    else:
        cmd = [pyspy, "dump", "-p", str(pid)]
        print(f"\ndumping every thread of pid {pid}\n")
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

"""
benchmark_particles_batch.py — whole-movie segmentation at REAL frame size.

The report that started this was "scribble segmentation over 900 frames of
4096x4096 is far too slow, and the GPU is hardly used, as are the CPUs". This
harness measures that end to end: the per-stage cost of ONE 4096^2 frame, then
the whole-movie throughput serially and through the dual-lane dispatcher
(:mod:`spyde.particles.batch`), and projects the 900-frame wall clock from both.

Run it on a REAL in-situ movie (CLAUDE.md § Benchmarking: real dataset, real
scale, end to end). A synthetic 96x112 fixture validates correctness and hides
every cost that actually bites here — ``measure_frame`` alone is 2 ms per
PARTICLE, and a real 4096^2 growth frame has 26 566 of them.

Not run under pytest (it is minutes long, and torch-CUDA segfaults under the
pytest process on Windows). Run it directly::

    .venv/Scripts/python -m spyde.tests.benchmark_particles_batch
    .venv/Scripts/python -m spyde.tests.benchmark_particles_batch --frames 36
    .venv/Scripts/python -m spyde.tests.benchmark_particles_batch --stages-only
    .venv/Scripts/python -m spyde.tests.benchmark_particles_batch --serial
    .venv/Scripts/python -m spyde.tests.benchmark_particles_batch --serial --purge-cache

``--frames`` is a frame COUNT at the real frame SIZE; the 900-frame figure is
extrapolated from it and labelled as such. Twelve frames per worker is enough
for the lanes to reach steady state.

Scribble is the only engine now — the classical engine (``segment_frame``) this
file used to A/B against was deleted from this branch, and ``resolve_engine``
refuses ``method="classical"`` outright (see ``test_particles_batch.py``,
``TestEngineSpec.test_there_is_no_classical_engine_any_more``). What remains of
that comparison is the one that still matters: the SAME resolved scribble
engine run serially (``--serial``) vs. through the dual-lane dispatcher.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import numpy as np

# Candidate real in-situ movies on this dev box (first that exists wins) — the
# same list benchmark_movie_playback.py uses.
_CANDIDATES = [
    r"C:\Users\CarterFrancis\Downloads\20251117_88075_run3 some growth_1236_movie.mrc",
    r"C:\Users\CarterFrancis\Downloads\20251117_88074_run1_9104_movie.mrc",
    r"C:\Users\CarterFrancis\Downloads\20241002_07954_movie.mrc",
]

TARGET_FRAMES = 900          # the user's movie length, for the projection


def _default_path() -> str | None:
    for p in _CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _fmt_hms(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def purge_cache(path: str) -> None:
    """Evict *path*'s pages from the Windows file cache before an arm.

    Same trick as ``benchmark_nav_fill_dispatch.purge_cache`` (see
    ``benchmarks.md``'s page-cache trap): opening a handle with
    ``FILE_FLAG_NO_BUFFERING`` makes the cache manager drop the cached view of
    the file, so the NEXT read pays real disk I/O rather than RAM. Without
    this, running the serial arm before the batch arm warms the exact pages
    the batch arm then reads for free — a "batch is way faster" result that is
    partly a page-cache donation from the arm that ran first, not the
    dispatcher. Best-effort: a failed purge only weakens the A/B, so it warns
    rather than raising.
    """
    if os.name != "nt":
        try:
            os.system("sync")
        except Exception:
            pass
        return
    GENERIC_READ = 0x80000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x20000000
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    h = k32.CreateFileW(
        ctypes.c_wchar_p(path), ctypes.c_uint(GENERIC_READ),
        ctypes.c_uint(FILE_SHARE_READ | FILE_SHARE_WRITE), None,
        ctypes.c_uint(OPEN_EXISTING), ctypes.c_uint(FILE_FLAG_NO_BUFFERING), None,
    )
    if h in (None, -1, 2 ** 64 - 1):
        print(f"  [purge] CreateFileW failed ({ctypes.get_last_error()})")
        return
    k32.CloseHandle(ctypes.c_void_p(h))


# ── the scribble head ────────────────────────────────────────────────────────

def train_scribble(frame, labels, *, device=None, crop: int = 1024):
    """A realistically-trained head, from pseudo-scribbles on a 1024^2 crop.

    Painted by hand in the app; synthesised here from an otsu segmentation of
    the same frame (``labels`` — see ``labels_from`` in
    ``spyde/tests/migrated/_labels.py``) so the benchmark is reproducible.
    What matters for TIMING is the feature spec and the frame size, both of
    which are the real ones — the particular strokes only move the particle
    count.
    """
    from scipy import ndimage as ndi

    from spyde.particles.features import FeatureSpec, select_device
    from spyde.particles.scribble import (LabelStore, ScribbleClassifier,
                                          default_classes)

    h, w = frame.shape
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    lab = labels[y0:y0 + crop, x0:x0 + crop]
    img = frame[y0:y0 + crop, x0:x0 + crop]

    fg = lab > 0
    core = ndi.binary_erosion(fg, iterations=1)
    far = ~ndi.binary_dilation(fg, iterations=6)

    def sub(mask, k=4000):
        idx = np.flatnonzero(np.asarray(mask).reshape(-1))
        return idx[:: max(1, idx.size // k)] if idx.size > k else idx

    store = LabelStore(frame_shape=(crop, crop), classes=default_classes())
    store.paint(0, sub(core), 0)
    store.paint(0, sub(far), 1)
    dev = select_device(device)
    if getattr(dev, "type", str(dev)) == "cuda":
        # Prime cuBLASLt before the fit's conv feature stack: on Pascal +
        # cu124 + Windows its FIRST initialisation fails with
        # CUBLAS_STATUS_NOT_INITIALIZED when it happens after cuDNN conv work
        # (minimal pair, 2026-08-07), and the MLP's first linear is exactly
        # that late first use.
        import torch
        torch.nn.functional.linear(torch.zeros(1, 1, device=dev),
                                   torch.zeros(1, 1, device=dev))
    clf = ScribbleClassifier(FeatureSpec(), device=dev, seed=0)
    rep = clf.fit(store, {0: img})
    print(f"  scribble head: {rep['n_pixels']} px / {rep['n_classes']} classes, "
          f"acc {rep['train_accuracy']:.3f}, boundary={rep['has_boundary']}, "
          f"device {clf.device}")
    return clf


# ── per-stage, one frame ─────────────────────────────────────────────────────

def stage_profile(frame, sp, clf) -> None:
    """Where one 4096^2 frame's time goes, stage by stage.

    Scribble is the only engine now (see the module docstring) — this used to
    also time the classical ``segment_frame`` arm alongside it; that arm and
    the engine it measured are both gone.
    """
    from spyde.particles.instances import split_instances
    from spyde.particles.measure import _contours, _fill_intensity
    from spyde.particles import measure_frame

    print(f"\n== one frame, {frame.shape[0]}x{frame.shape[1]} {frame.dtype} ==")

    import torch
    sync = (lambda: torch.cuda.synchronize()) if clf.device.type == "cuda" \
        else (lambda: None)
    sync()
    t0 = time.perf_counter()
    fg, bnd = clf.predict_foreground_boundary(frame)
    sync()
    t1 = time.perf_counter()
    labels = split_instances(fg, sp, boundary=bnd)
    t2 = time.perf_counter()
    rows, _cs = measure_frame(labels, frame, t=0, scale=1.0)
    t3 = time.perf_counter()
    print(f"  scribble  : predict {t1-t0:6.2f}s  split {t2-t1:6.2f}s  "
          f"measure {t3-t2:7.2f}s  n={len(rows)}   total {t3-t0:7.2f}s")

    # measure_frame is the stage the whole-movie cost turns on once a real frame
    # has thousands of particles, so break it down — BOTH ways, because all three
    # of its stages have been replaced and the interesting number is the ratio,
    # not either column alone (`benchmarks.md` § "Vectorising measure_frame").
    from spyde.particles.measure import property_table, warm_kernels
    from spyde.signals.particles import N_COLUMNS

    warm_kernels()      # never time a numba compile as if it were the work
    inten = np.asarray(frame, np.float64)
    for fast in (False, True):
        t0 = time.perf_counter()
        tbl = property_table(labels, fast=fast)
        t1 = time.perf_counter()
        n = len(tbl["label"])
        r = np.zeros((n, N_COLUMNS), np.float32)
        keep = np.ones(n, bool)
        _fill_intensity(r, labels, inten, tbl, keep, 3, fast=fast)
        t2 = time.perf_counter()
        _contours(labels, tbl, fast=fast)
        t3 = time.perf_counter()
        tag = "vectorised" if fast else "regionprops"
        print(f"  measure_frame internals, {tag:11s} ({n} particles): "
              f"table {t1-t0:6.2f}s  intensity {t2-t1:5.2f}s  "
              f"contours {t3-t2:5.2f}s  sum {t3-t0:6.2f}s")


# ── whole-movie ──────────────────────────────────────────────────────────────

def run_serial(data, spec, n, scale) -> tuple[float, int]:
    """The retired shape: one thread, one frame at a time."""
    from spyde.particles.batch import resolve_engine
    from spyde.particles.measure import measure_frame

    engine, dev = resolve_engine(spec)
    t0 = time.perf_counter()
    total = 0
    for t in range(n):
        frame = np.asarray(data[t])
        if hasattr(frame, "compute"):
            frame = frame.compute()
        labels = engine(frame)
        rows, _cs = measure_frame(labels, frame, t=t, scale=scale)
        total += len(rows)
    return time.perf_counter() - t0, total


def run_batch(data, spec, n, scale, client) -> tuple[float, int]:
    from spyde.particles.batch import segment_movie

    t0 = time.perf_counter()
    rows, _cs, done = segment_movie(data, spec, n_frames=n, scale=scale,
                                    store_masks=True, client=client)
    dt = time.perf_counter() - t0
    assert done == n, f"only {done}/{n} frames landed"
    _worker_stage_summary(client)
    return dt, sum(len(r) for r in rows)


def _worker_stage_summary(client) -> None:
    """Per-lane in-cluster stage costs, taken off the workers themselves.

    ``engine/f`` and ``measure/f`` here are what a frame cost INSIDE the
    cluster; comparing them with the single-frame profile above is the whole
    diagnosis — a stage that is 2x its solo cost is contended, and a lane whose
    frames/s is far below its stage sum is starved rather than slow.
    """
    from spyde.particles.batch import drain_stage_log

    per_dev: dict[str, list] = {}
    try:
        for recs in client.run(drain_stage_log).values():
            for _t0, n, dev, t_eng, t_meas, t_wall in recs:
                per_dev.setdefault(dev, []).append((n, t_eng, t_meas, t_wall))
    except Exception as exc:
        print(f"  (worker telemetry unavailable: {exc})")
        return
    if not per_dev:
        return
    print(f"  {'lane':>6} {'frames':>7} {'engine/f':>9} {'measure/f':>10} "
          f"{'block/f':>8}")
    for dev, rows in sorted(per_dev.items()):
        nf = sum(r[0] for r in rows) or 1
        print(f"  {dev:>6} {nf:>7} {sum(r[1] for r in rows)/nf:9.2f} "
              f"{sum(r[2] for r in rows)/nf:10.2f} "
              f"{sum(r[3] for r in rows)/nf:8.2f}")


def make_cluster(n_workers: int, threads: int):
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=0, threads_per_worker=threads,
                           processes=True)
    client = Client(cluster)
    cluster.scale(n_workers)
    client.wait_for_workers(n_workers, timeout=180)

    def _enable_telemetry():
        import logging as _lg
        _lg.getLogger("spyde.particles.batch").setLevel(_lg.INFO)

    client.run(_enable_telemetry)
    return cluster, client


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=_default_path())
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--start", type=int, default=10)
    ap.add_argument("--serial", action="store_true",
                    help="also time the serial loop (slow)")
    ap.add_argument("--stages-only", action="store_true")
    ap.add_argument("--gpu-lane", default=None,
                    help="SPYDE_FV_GPU override for this run (one/N/all/off)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--conc", default=None,
                    help="SPYDE_FV_GPU_CONC override (device slots per process)")
    ap.add_argument("--purge-cache", action="store_true",
                    help="evict the movie's cached pages (Windows) before each "
                         "arm, so serial and batch see the same cold I/O instead "
                         "of the second arm inheriting whatever the first one "
                         "warmed")
    args = ap.parse_args(argv)

    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s",
                        stream=sys.stdout)
    # The dispatcher's own summary line is the lane split — GPU chunks vs CPU
    # chunks — which is the number that says whether the lanes are balanced.
    logging.getLogger("spyde.compute_dispatch").setLevel(logging.DEBUG)
    logging.getLogger("spyde.particles.batch").setLevel(logging.INFO)
    if args.conc:
        os.environ["SPYDE_FV_GPU_CONC"] = str(args.conc)

    if args.gpu_lane:
        os.environ["SPYDE_FV_GPU"] = str(args.gpu_lane)
    if not args.path or not os.path.exists(args.path):
        print("no in-situ movie found; pass --path", file=sys.stderr)
        return 2

    import hyperspy.api as hs
    from spyde.backend.app import _compute_worker_plan
    from spyde.particles import SegmentParams
    from spyde.particles.batch import EngineSpec, save_engine_model

    print(f"movie: {args.path}")
    s = hs.load(args.path, lazy=True)
    # Load the way the APP loads. RosettaSciIO auto-chunks a big MRC as a
    # balanced cube — a real 977 x 4096^2 movie arrives as (511, 511, 511),
    # which SPLITS the signal axes, so one frame spans 64 blocks and 8.5 GB.
    # `Session._signal_spanning_chunks` re-loads every movie with whole-signal
    # chunks (free: a lazy reload only rebuilds the graph); benchmarking the
    # reader default instead would measure a shuffle the app never performs.
    from spyde.backend.session import Session
    ch = Session._signal_spanning_chunks(s)
    if ch is not None:
        print(f"  reader chunks {s.data.chunks[0][:2]}... split the signal "
              f"axes — re-loading with {ch} (as the app does)")
        s = hs.load(args.path, lazy=True, chunks=ch)
    raw = s.data
    print(f"  shape {raw.shape} {raw.dtype}  nav chunks "
          f"{raw.chunks[0][:4]}{'...' if len(raw.chunks[0]) > 4 else ''}  "
          f"signal chunks {tuple(c[0] for c in raw.chunks[1:])}")

    sp_kwargs = dict(min_size=20)
    sp = SegmentParams(**sp_kwargs)
    t0 = args.start
    frame = np.asarray(raw[t0].compute())

    # Scribble is the only engine (see module docstring): train it from a
    # label image derived by Otsu + the shared instance split — exactly what
    # the deleted classical engine did (`spyde/tests/migrated/_labels.py`,
    # which the migrated tests use for the same reason). Only the STROKES this
    # produces are synthetic; the frame, its size and the feature spec are
    # real, which is what timing here cares about.
    from spyde.tests.migrated._labels import labels_from
    print("\n== training the scribble head (labels: otsu + split_instances) ==")
    labels = labels_from(frame, **sp_kwargs)
    clf = train_scribble(frame, labels)
    model_path = save_engine_model(clf)

    stage_profile(frame, sp, clf)
    if args.stages_only:
        return 0

    workers, threads = _compute_worker_plan(os.cpu_count() or 4)
    workers = args.workers or workers
    threads = args.threads or threads
    lane = os.environ.get("SPYDE_FV_GPU", "<unset -> 4>")
    print(f"\n== cluster: {workers} workers x {threads} threads, "
          f"SPYDE_FV_GPU={lane} ==")
    cluster, client = make_cluster(workers, threads)

    n = int(args.frames)
    sub = raw[t0:t0 + n]
    spec = EngineSpec(method="scribble", params=sp_kwargs, model_path=model_path)
    try:
        print(f"\n== scribble: {n} frames of "
              f"{frame.shape[0]}x{frame.shape[1]} ==")
        if args.serial:
            if args.purge_cache:
                purge_cache(args.path)
            dt, total = run_serial(sub, spec, n, 1.0)
            print(f"  serial : {dt:7.1f}s  {n/dt:6.3f} frames/s  "
                  f"{total} particles  ->  900 frames = "
                  f"{_fmt_hms(dt / n * TARGET_FRAMES)}")
        if args.purge_cache:
            purge_cache(args.path)
        elif args.serial:
            print("  NOTE: no --purge-cache — the batch arm below may be "
                  "reading pages the serial arm already faulted into RAM, so "
                  "part of any speed-up is a warm page cache, not the "
                  "dispatcher. Re-run with --purge-cache for a cold A/B.")
        dt, total = run_batch(sub, spec, n, 1.0, client)
        print(f"  batch  : {dt:7.1f}s  {n/dt:6.3f} frames/s  "
              f"{total} particles  ->  900 frames = "
              f"{_fmt_hms(dt / n * TARGET_FRAMES)}")
    finally:
        client.close()
        cluster.close()
    return 0


if __name__ == "__main__":
    _rc = main()
    # torch/CUDA teardown crashes on exit on Windows (CLAUDE.md); the numbers
    # are already printed, so leave before it runs rather than sys.exit(main()).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc or 0)

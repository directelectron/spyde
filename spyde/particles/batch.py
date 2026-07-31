"""
batch.py — whole-movie segmentation, fanned out over the cluster.

The batch run used to be a plain serial ``for t in range(n_frames)`` on ONE
worker thread: read a frame, segment it, measure it, repeat. On the target
workload — 900 frames of 4096² — that is 90 minutes with one core busy, the
other 47 idle and the GPU idle 78% of the time. This module replaces that loop
with the **dual-lane dispatch SpyDE already uses for find-vectors**, because
segmentation is the same shape of problem: independent per-frame work, one
device, many cores.

Why the find-vectors mechanism and not a pipeline
-------------------------------------------------
The obvious reading is that the two engines want different parallelism — the
classical engine is pure scipy so it should fan out, while the scribble engine
holds ONE torch model on ONE GPU so fanning eight workers at it would mean eight
CUDA contexts fighting over one device. That reading is wrong, and SpyDE already
has the answer: :func:`~spyde.actions.find_vectors.gpu_runtime._gpu_task_allowed`.

Dask fans every chunk out over every worker as normal. Each task then asks
whether **this** worker may touch the GPU: workers named ``"1".."N"`` may, and
the rest run the CPU path. ``SPYDE_FV_GPU`` (the **GPU feeders** control in the
DaskMonitor popover, :mod:`spyde.backend.compute_config`) overrides the policy
for segmentation exactly as it does for find-vectors, and
:data:`PARTICLE_GPU_LANE_DEFAULT` is the unset-default.

So the two engines do NOT need different architectures. Both go through one
fan-out; the lane policy decides who uses the GPU, and the classical engine
simply never asks for it — which means it uses every worker, since a lane it
never asks about cannot exclude it.

Why the lane default is "one" and NOT the neural path's "4"
-----------------------------------------------------------
``default_mode`` is per-CALLER by design, and segmentation's was measured rather
than inherited. GPU-only dispatch, empty CPU lane, 60 frames of a real 977 x
4096² movie:

    GPU feeders   1       2       4
    throughput    0.270   0.252   0.260 frames/s

**Flat.** More processes on the device neither help nor much hurt, because the
device is not the constraint — see the next section for what is. So
segmentation keeps ``"one"``, on the same footing as the numba NXCORR kernels:
nothing is bought by spreading it, and one process is the arrangement that
cannot go wrong (Windows has no MPS server, and the feature stack sizes each row
band against *free* VRAM, which several processes then divide — ``GPU_BAND_BYTES``
records that overshooting there is catastrophic rather than merely slower). An
explicit ``SPYDE_FV_GPU`` still overrides, which is the user's call.

*History, because the first answer here was wrong and confidently so:* the
initial measurement said four feeders made a frame **13x** slower (110 s of
predict+split against 8 s). That run also had five CPU-lane workers each running
a 48-thread torch predict, so it measured contention and not the lane count. A
lane number taken from a contended run is exactly the kind of default that then
looks load-bearing forever. Re-measure with the other lane empty.

Why the CPU lane IS excluded, which the isolated numbers argue against
----------------------------------------------------------------------
Neural find-vectors dispatches ``gpu_only=True`` because torch-CPU inference is
10-50x the GPU batch. Segmentation looks like it should be the exception. One
4096² scribble predict on CPU:

    torch threads   1      2      4      8     16     48
    wall            65.8   35.1   18.8   11.2   7.0    9.0 s
    core-seconds    65.8   70.2   75.2   89.8  111.6  430.5

Against 1.6 s on the GPU that is 41x in core-seconds — but a GPU-lane frame only
occupies ~4.4 core-seconds of the machine (the split and the measurement, which
are CPU work on both lanes), so on paper 48 cores of CPU lane are worth about as
much as the GPU and running both is close to a doubling.

It is not. Measured in the cluster, the CPU lane contributed 0.16 frames/s and
cost the GPU lane 0.5 — its frames went from 5.8 s to 25-29 s — and the run went
0.270 -> 0.222 frames/s. The two lanes are not additive because they compete for
the same cores and the same memory bandwidth. So the CPU lane is **off by
default** (:func:`cpu_lane_enabled`, ``SPYDE_SEG_CPU_LANE=1`` to restore it),
and segmentation lands where the neural batch already is.

That thread table is also why every worker pins ``torch.set_num_threads(1)``
(:func:`_cap_torch_threads`): intra-op threading costs more total work the wider
it goes, and frame-level parallelism is free.

What used to limit this, and no longer does
--------------------------------------------
The honest ceiling WAS ``measure_frame``, and it bound twice: it was most of the
frame, and it held the **GIL**, so a worker's four task slots were worth one core
and the effective parallelism of a batch was the WORKER COUNT rather than the
worker × thread count.

Both halves are now gone, in two passes, and every one of them had to prove
parity before it was allowed to land (``benchmarks.md`` § "Vectorising
``measure_frame``"):

* ``regionprops_table`` → :mod:`spyde.particles.props` (every column that is a
  label-wise reduction) + :mod:`spyde.particles.hull` (``solidity``, a numba
  convex hull in exact integer arithmetic). 43.7 s → 1.08 s, every column
  bit-identical.
* ``_fill_intensity`` → :mod:`spyde.particles.intensity` (``bincount``
  statistics + a numba kernel for the background ring, which is per-particle and
  therefore not a partition of the raster). 4.9 s → 0.19 s, all four columns
  bit-identical on the real 26 566-region frame.
* ``_contours`` → :mod:`spyde.particles.contours` (marching squares and the
  segment assembly, every region in one ``prange``). 4.8 s → 0.15 s, and the
  FILLED polygon — what ``render_frame`` and ``mask_at`` actually consume — is
  identical on 26 566 of 26 566 regions.

``measure_frame`` is **52.6 s → 1.37 s**, and, because all three replacements are
numba ``nogil`` kernels and numpy ufuncs, four quadrants in four threads now scale
**2.82x** where the whole function managed 0.94x with the two Python loops still
in it. A worker's task slots are finally worth more than one core, so worker ×
thread is the unit of parallelism here rather than worker count.

Memory safety (CLAUDE.md)
-------------------------
Nothing here ever calls ``.compute()`` on the movie. Tasks are sized by BYTES
(:func:`frames_per_task`) so one task holds a bounded block of frames, and the
per-frame loop inside a task holds exactly one frame's labels at a time.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

#: Unset-default for the segmentation GPU lane, i.e. how many workers may submit
#: to the device when ``SPYDE_FV_GPU`` is not set. MEASURED, not inherited from
#: the neural path's "4" — 1/2/4 feeders are 0.270/0.252/0.260 frames/s on a real
#: 4096² movie, i.e. flat, so this takes the arrangement that cannot go wrong.
#: See the module docstring, including why the first measurement said 13×.
#:
#: This single value MUST be what both the per-worker gate (:func:`_gpu_allowed`
#: → ``_gpu_task_allowed``) and the client-side lane split (:func:`_dispatch` →
#: ``split_workers_for_gpu``) are given. When those two disagree, workers get
#: chunks they then refuse to run on the GPU — the bug ``gpu_runtime`` records
#: for the neural path, where the docstring said "2" while the code passed "4".
PARTICLE_GPU_LANE_DEFAULT = "one"

#: Target bytes of RAW FRAMES per dask task. Not a chunk-alignment number — a
#: working-set one: a task also holds an int32 label raster (4 bytes/px, i.e.
#: 64 MB for one 4096² frame) plus the float probability maps, and several tasks
#: run concurrently per worker. 64 MB of frames keeps a worker's peak near 1 GB.
#: Override with ``SPYDE_SEG_TASK_BYTES``.
TASK_BYTES: int = 64 << 20

#: Aim for at least this many tasks per worker, so a short movie still spreads
#: out instead of collapsing into one task per lane.
_TASKS_PER_WORKER = 4


def cpu_lane_enabled() -> bool:
    """Whether scribble chunks may also go to the non-GPU workers.

    **Off by default, and that is a measurement, not a guess.** In isolation the
    CPU lane looks like most of a doubling (the core-second table in the module
    docstring). In the cluster it is worse than nothing, because the two lanes
    compete for the same 48 cores and the same memory bandwidth — 60 frames of a
    real 4096² movie, one GPU worker, 8 CPU workers × 4 slots:

        CPU-lane predict, alone, 1 torch thread     66 s
        CPU-lane predict, 32 concurrent            88-177 s   (~2x, bandwidth)
        GPU-lane frame, machine otherwise idle     5.8 s
        GPU-lane frame, CPU lane running          25-29 s     (starved)

    The CPU lane contributed 0.16 frames/s and cost the GPU lane 0.5 — the run
    went from 0.270 frames/s GPU-only to 0.222. So segmentation lands where the
    neural detector already is (``gpu_only=True`` in orchestrate), for the same
    underlying reason: a torch path whose CPU cost is disproportionate does not
    belong in a second lane, it belongs out of the way of the first.

    ``SPYDE_SEG_CPU_LANE=1`` turns it back on — worth it on a machine with more
    memory bandwidth than GPU, or with several GPUs, neither of which is this
    one. It has no effect on the classical engine, which has no GPU lane at all
    and always runs on every worker.
    """
    return os.environ.get("SPYDE_SEG_CPU_LANE", "") not in ("", "0", "off")


# ── what a worker needs to rebuild the engine ────────────────────────────────

@dataclass(frozen=True)
class EngineSpec:
    """A picklable description of the engine, resolved to a callable per worker.

    The trained :class:`~spyde.particles.scribble.ScribbleClassifier` is shipped
    as a **file path**, not as an object: its weights live on the client's CUDA
    device, and pickling those would deserialise a CUDA tensor inside every
    worker process — a context per worker, created behind our back, before the
    lane policy has had any say. Shipping the path lets each worker load onto
    the device the lane policy chose for it, which is the same trick the neural
    detector plays with ``models.get_model(model_id)``.

    Parameters
    ----------
    method
        ``"classical"`` or ``"scribble"``.
    params
        ``SegmentParams`` keyword arguments. A plain dict so the spec pickles
        without importing the segmentation modules on the client.
    model_path
        Scribble only: an ``.npz`` written by ``ScribbleClassifier.save``.
    """

    method: str
    params: dict = field(default_factory=dict)
    model_path: str | None = None

    def segment_params(self):
        from spyde.particles.classical import SegmentParams
        return SegmentParams(**self.params)


#: Per-PROCESS ring of ``(t0, n, device, engine_s, measure_s, block_s)`` records,
#: one per task. A batch that is fast in isolation and slow in the cluster is
#: the NORMAL outcome — contention, oversubscription, a rechunk nobody asked for
#: — and the wall clock alone cannot tell you which. Drained by
#: :func:`drain_stage_log` (``client.run``), which is how
#: ``benchmark_particles_batch`` prints its per-lane table. Bounded, so a
#: 900-frame run cannot grow it without limit.
_STAGE_LOG: list = []
_STAGE_LOG_MAX = 4096

#: Per-PROCESS engine cache. Keyed by (model path, mtime, device) so a retrain
#: (a new temp file) never serves a stale head, and so the GPU and CPU lanes in
#: one process — which happens in the local thread-pool fallback — keep separate
#: models rather than moving one back and forth across devices.
_ENGINE_CACHE: dict[tuple, Any] = {}
_TORCH_THREADS_SET = [False]


def _cap_torch_threads() -> None:
    """Pin torch to ONE intra-op thread inside a dask worker, once.

    torch defaults to one intra-op thread per LOGICAL CORE, which inside a
    worker is a lie about how much of the machine this process owns: nine
    workers × four task slots each spawning 48 OpenMP threads is 1728 threads
    on 48 cores. But the reason for **one** rather than "the worker's thread
    budget" is stronger than avoiding oversubscription — measured on a 4096²
    predict, intra-op threading costs MORE total work the wider it goes (65.8
    core-seconds at 1 thread, 111.6 at 16, 430.5 at 48). Frame-level
    parallelism is free and thread-level is not, and dask's task slots already
    supply the former: one task per core is both the cheapest and the fastest
    arrangement.

    Deliberately a no-op outside a dask worker — the interactive preview and
    the training fit are single, latency-sensitive calls that should keep the
    whole machine.
    """
    if _TORCH_THREADS_SET[0]:
        return
    try:
        from distributed import get_worker
        get_worker()
    except Exception:
        return                     # not on a dask worker: leave torch alone
    _TORCH_THREADS_SET[0] = True
    try:
        import torch
        torch.set_num_threads(1)
    except Exception as exc:                                  # pragma: no cover
        log.debug("[seg-batch] capping torch threads failed: %s", exc)


def _warm_measure() -> None:
    """Compile every ``measure_frame`` kernel BEFORE the first frame is measured.

    Three now: ``solidity``'s convex hull (:mod:`spyde.particles.hull`, which
    replaced a per-region Qhull call — 30.2 s to 0.18 s at 26 566 regions), the
    background ring (:mod:`spyde.particles.intensity`) and the outlines
    (:mod:`spyde.particles.contours`). numba compiles each on first use, which is
    seconds, and paying that inside the first measured frame of a task makes that
    frame look like a regression and skews every per-stage number the benchmark
    prints. ``cache=True`` means they are normally loaded from disk rather than
    rebuilt. Cheap and idempotent: each module short-circuits once compiled.
    """
    try:
        from spyde.particles.measure import warm_kernels
        warm_kernels()
    except Exception as exc:                                  # pragma: no cover
        log.debug("[seg-batch] measure warmup skipped: %s", exc)


def _gpu_allowed() -> bool:
    """Whether THIS worker may use the GPU, per the shared lane policy.

    Delegates to find-vectors' ``_gpu_task_allowed`` rather than re-deriving the
    rule: it is the same ``SPYDE_FV_GPU`` setting, surfaced in the same
    DaskMonitor control, and a second implementation is a second thing to get
    out of step.
    """
    try:
        from spyde.actions.find_vectors.gpu_runtime import _gpu_task_allowed
        return bool(_gpu_task_allowed(default_mode=PARTICLE_GPU_LANE_DEFAULT))
    except Exception as exc:                                  # pragma: no cover
        log.debug("[seg-batch] GPU lane policy unavailable (%s); using CPU", exc)
        return False


def _gpu_slots():
    """The device-concurrency semaphore (``SPYDE_FV_GPU_CONC``), or a null
    context when find-vectors is not importable.

    Bounds how many frames occupy the device at once **per process**. The
    feature stack sizes each row band at a fraction of *free* VRAM
    (``features.band_budget_bytes``), and overshooting that is catastrophic
    rather than merely slow — so unbounded concurrency across a worker's task
    threads is exactly the allocator thrash ``GPU_BAND_BYTES`` documents.
    """
    try:
        from spyde.actions.find_vectors.gpu_runtime import _gpu_slots as _slots
        return _slots()
    except Exception:                                         # pragma: no cover
        import contextlib
        return contextlib.nullcontext()


def resolve_engine(spec: EngineSpec, *, force_cpu: bool = False):
    """``(engine, device_str)`` for this worker — the lane decision made once.

    ``engine(frame) -> int32 labels``. The classical engine never asks for the
    GPU, so it runs on every worker regardless of lane; the scribble engine
    loads its head onto CUDA/MPS only on a GPU-lane worker and onto the CPU
    everywhere else.
    """
    _cap_torch_threads()
    _warm_measure()
    method = str(spec.method).lower()
    if method == "classical":
        from spyde.particles.classical import segment_frame
        sp = spec.segment_params()
        return (lambda frame: segment_frame(frame, sp)), "cpu"

    if method != "scribble":
        raise ValueError(
            f"batch segmentation has no engine for method {spec.method!r}; "
            "expected 'classical' or 'scribble'")
    if not spec.model_path:
        raise ValueError(
            "the scribble engine needs a trained model — EngineSpec.model_path "
            "is empty (train the classifier before running the batch)")

    want_gpu = (not force_cpu) and _gpu_allowed()
    device = None if want_gpu else "cpu"
    try:
        mtime = os.path.getmtime(spec.model_path)
    except OSError:
        mtime = 0.0
    key = (spec.model_path, mtime, str(device))
    clf = _ENGINE_CACHE.get(key)
    if clf is None:
        from spyde.particles.scribble import ScribbleClassifier
        clf = ScribbleClassifier.load(spec.model_path, device=device)
        _ENGINE_CACHE[key] = clf
        log.info("[seg-batch] scribble head loaded on %s (gpu lane: %s)",
                 clf.device, want_gpu)
    sp = spec.segment_params()

    def engine(frame, _clf=clf, _sp=sp):
        # The device section only — the split that follows is numpy/scipy and
        # must NOT hold a device slot while it runs (that is what would turn
        # four GPU feeders back into one).
        with _gpu_slots():
            fg, bnd = _clf.predict_foreground_boundary(frame)
        from spyde.particles.classical import split_instances
        return split_instances(fg, _sp, boundary=bnd)

    return engine, str(clf.device)


# ── the per-task body ────────────────────────────────────────────────────────

def segment_block(block: np.ndarray, t0: int = 0, spec: EngineSpec | None = None,
                  *, scale: float = 1.0, store_masks: bool = True) -> np.ndarray:
    """Segment and measure every frame of one ``(n, h, w)`` block.

    Returns a 1-D **object** array of ``(rows, contours)``, one entry per frame,
    which is what the dispatcher assembles. An object payload rather than the
    NaN-padded fixed array find-vectors uses because a frame's result is doubly
    ragged: a variable number of particle rows AND a variable-length outline per
    particle. Padding to a fixed cap would either truncate a busy frame (a real
    4096² growth frame here measured **26 566** particles) or waste most of the
    transfer.

    Runs on a dask worker. Holds ONE frame's labels at a time — the block is
    already bounded by :func:`frames_per_task`.

    *t0* stamps the ``t`` column. The dispatch path leaves it at 0 and
    :func:`segment_movie` stamps the true index when the block lands, because a
    task **cannot** know its own global offset: ``dispatch_chunks`` slices the
    result array per chunk, and a ``map_blocks`` stage reading
    ``block_info["array-location"]`` then reports (0, 0) for every one of them —
    the bug ``orchestrate._do_compute_vectors`` records for the live count map.
    """
    from spyde.particles.measure import measure_frame

    block = np.asarray(block)
    if block.ndim != 3:
        raise ValueError(f"expected an (n, h, w) block; got shape {block.shape}")
    n = int(block.shape[0])
    out = np.empty((n,), dtype=object)
    if n == 0:
        return out                       # dask meta inference calls with size 0
    if spec is None:
        raise TypeError("segment_block needs an EngineSpec")

    engine, dev = resolve_engine(spec)
    t_eng = t_meas = 0.0
    n_particles = 0
    t_block = time.perf_counter()
    for i in range(n):
        frame = np.asarray(block[i])
        t_a = time.perf_counter()
        labels = _engine_with_cpu_fallback(engine, frame, spec)
        t_b = time.perf_counter()
        rows, contours = measure_frame(labels, frame, t=int(t0) + i,
                                       scale=float(scale))
        t_c = time.perf_counter()
        t_eng += t_b - t_a
        t_meas += t_c - t_b
        n_particles += len(rows)
        out[i] = (np.ascontiguousarray(rows, np.float32),
                  (list(contours) if store_masks else []))
    t_wall = time.perf_counter() - t_block
    if len(_STAGE_LOG) < _STAGE_LOG_MAX:
        _STAGE_LOG.append((int(t0), n, dev, t_eng, t_meas, t_wall))
    log.debug("[seg-batch] block %d..%d on %s: engine %.2fs measure %.2fs "
              "(block %.2fs, %d particles)", t0, t0 + n, dev, t_eng, t_meas,
              t_wall, n_particles)
    return out


def drain_stage_log() -> list:
    """Take and clear this process's per-task stage records. Run via
    ``client.run(drain_stage_log)`` to see the in-cluster stage costs."""
    out = list(_STAGE_LOG)
    _STAGE_LOG.clear()
    return out


def _map_block(block, spec=None, scale=1.0, store_masks=True):
    """``map_blocks`` entry point — module level so the graph pickles small."""
    if block.size == 0 or block.shape[0] == 0:
        return np.empty((0,), dtype=object)
    return segment_block(block, 0, spec, scale=scale, store_masks=store_masks)


def _engine_with_cpu_fallback(engine, frame, spec: EngineSpec):
    """Run *engine*, retrying the frame on the CPU if the device refuses it.

    Several GPU-lane workers each sizing a feature band against *free* VRAM can
    collectively overcommit a 12 GB card, and an out-of-memory frame must not
    take the whole 900-frame run down with it. A silent fallback would hide a
    throughput problem, so this logs loudly — the same bargain
    ``_find_vectors_chunk`` makes around its GPU path.
    """
    try:
        return engine(frame)
    except Exception as exc:
        if str(spec.method).lower() != "scribble":
            raise
        log.warning("[seg-batch] GPU segmentation failed (%r) — retrying this "
                    "frame on the CPU", exc)
        cpu_engine, _dev = resolve_engine(spec, force_cpu=True)
        return cpu_engine(frame)


# ── task sizing ──────────────────────────────────────────────────────────────

def frames_per_task(n_frames: int, frame_nbytes: int, *, n_workers: int = 1,
                    source_chunk: int | None = None) -> int:
    """How many frames one dask task should cover.

    Bounded three ways, and each bound is there for a different failure:

    * **by bytes** — a task holds its whole block plus a 4 byte/px label raster
      per frame in flight; ``TASK_BYTES`` keeps a worker's peak near 1 GB even
      with four task threads.
    * **by the source's own chunking** — never larger than one stored nav chunk,
      so a task never has to pull two chunks to serve one block (CLAUDE.md
      Live-Display §1: alignment, never a rechunk).
    * **by the cluster** — at least ``_TASKS_PER_WORKER`` tasks per worker, so a
      six-frame tutorial movie still spreads across the lanes instead of
      becoming one task that one worker runs alone.
    """
    try:
        budget = int(os.environ.get("SPYDE_SEG_TASK_BYTES", TASK_BYTES))
    except ValueError:
        budget = TASK_BYTES
    by_bytes = max(1, budget // max(1, int(frame_nbytes)))
    per = by_bytes
    if source_chunk:
        per = min(per, max(1, int(source_chunk)))
    spread = max(1, int(n_frames) // max(1, _TASKS_PER_WORKER * max(1, n_workers)))
    return int(max(1, min(per, spread)))


def _time_chunks(n_frames: int, per: int) -> tuple[int, ...]:
    per = max(1, int(per))
    full, rem = divmod(int(n_frames), per)
    out = [per] * full
    if rem:
        out.append(rem)
    return tuple(out) or (int(n_frames),)


# ── the run ──────────────────────────────────────────────────────────────────

def segment_movie(
    data,
    spec: EngineSpec,
    *,
    n_frames: int,
    get_frame: Callable[[int], np.ndarray] | None = None,
    scale: float = 1.0,
    store_masks: bool = True,
    client=None,
    stopped: list | None = None,
    on_frames: Callable[[int, int, list], None] | None = None,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], int] | None:
    """Segment every frame of *data*, in parallel. The batch run's whole body.

    Parameters
    ----------
    data
        The movie as a ``(n, h, w)`` numpy or dask array (``signal.data``).
    spec
        :class:`EngineSpec` — what each worker rebuilds the engine from.
    n_frames, get_frame
        The streaming accessor from ``frames_of``. ``get_frame`` is the serial
        fallback used when there is neither a cluster nor a dask array; the
        dispatch path never calls it.
    client
        A ``distributed.Client``, or None for the local thread-pool fallback
        (tests, ``SPYDE_NO_DASK=1``, a cluster that never came up).
    stopped
        One-element cancel flag, polled between blocks and honoured by the
        dispatcher.
    on_frames
        ``on_frames(t0, t1, results)`` as each block lands, where *results* is
        the list of ``(rows, contours)`` for frames ``t0:t1``. Called from a
        worker/callback thread — must not touch figures (CLAUDE.md threading).

    Returns
    -------
    ``(per_frame_rows, per_frame_contours, n_done)`` — both lists are exactly
    *n_frames* long, with EMPTY blocks where a cancelled run never reached, so
    the CSR store always spans the movie and a stopped run reads as "no
    particles after frame N" rather than as a shorter movie. A cancelled run
    still returns everything it did finish; the caller reads ``stopped`` to
    learn that it was cancelled.
    """
    from spyde.signals.particles import COL, N_COLUMNS

    n_frames = int(n_frames)
    results: list = [None] * n_frames
    done = [0]
    col_t = COL["t"]

    def _record(t0: int, block_out) -> None:
        # THE one place the frame index is written (see `segment_block`): a task
        # cannot know its own offset, so it stamps 0 and the true index is
        # applied here, where the dispatcher's global slice is authoritative.
        vals = list(block_out)
        for i, v in enumerate(vals):
            rows = v[0]
            if len(rows):
                rows[:, col_t] = float(t0 + i)
            results[t0 + i] = v
        done[0] += len(vals)
        if on_frames is not None:
            try:
                on_frames(t0, t0 + len(vals), vals)
            except Exception as exc:
                log.debug("[seg-batch] on_frames callback failed: %s", exc)

    lazy = type(data).__module__.startswith("dask.array") if data is not None else False
    arr_ok = data is not None and getattr(data, "ndim", 0) == 3

    if not (arr_ok and _dispatch(data, spec, n_frames, scale, store_masks,
                                 client, stopped, _record, lazy)):
        _serial(get_frame, spec, n_frames, scale, store_masks, stopped,
                _record)

    rows_out: list[np.ndarray] = []
    contours_out: list[list[np.ndarray]] = []
    for v in results:
        if v is None:
            rows_out.append(np.zeros((0, N_COLUMNS), np.float32))
            contours_out.append([])
        else:
            rows_out.append(v[0])
            contours_out.append(list(v[1]))
    return rows_out, contours_out, int(done[0])


def _n_workers(client) -> int:
    if client is None:
        return max(1, (os.cpu_count() or 4) // 4)
    try:
        # n_workers=-1: the default TRUNCATES to 5 (CLAUDE.md / benchmarks.md).
        return max(1, len(client.scheduler_info(n_workers=-1)["workers"]))
    except Exception:
        return 1


def _dispatch(data, spec, n_frames, scale, store_masks, client, stopped,
              record, lazy) -> bool:
    """Build the per-block graph and run it through the dual-lane dispatcher.

    Returns False when this movie cannot be dispatched, so ``segment_movie``
    falls back to the streaming accessor.
    """
    import dask.array as da

    if lazy and len(data.chunks) > 2 and any(len(c) > 1 for c in data.chunks[1:]):
        # The reader SPLIT the signal axes (RosettaSciIO's balanced-cube
        # auto-chunk: a real 977 x 4096² MRC arrives as (511, 511, 511)). A task
        # needs whole frames, and merging those axes while also cutting the time
        # axis down to one frame is a full P2P rechunk shuffle of the entire
        # movie — the exact thing CLAUDE.md Live-Display §1 forbids, and it was
        # measured here doing it: the dispatcher spent 90 s in stall pokes
        # waiting on `rechunk-merge-rechunk-transfer` before a single frame was
        # segmented.
        #
        # The fix belongs at LOAD time, where it is free — `hs.load(..., lazy=
        # True, chunks=(1, -1, -1))`, which is what `Session._signal_spanning_
        # chunks` already does for every movie the app opens. So this is not a
        # path the app reaches; it is the guard that keeps a hand-built signal
        # from silently shuffling multiple GB. Fall back to the streaming
        # accessor: slower, but correct and no worse than the serial loop was.
        log.warning(
            "[seg-batch] this movie's chunks %s split the signal axes, so a "
            "task cannot take whole frames without a full rechunk shuffle — "
            "segmenting serially instead. Re-load it with "
            "chunks=(1, -1, -1) to get the parallel path.", data.chunks)
        return False

    frame_nbytes = int(np.prod(data.shape[1:])) * int(data.dtype.itemsize)
    n_workers = _n_workers(client)
    source_chunk = None
    if lazy:
        try:
            source_chunk = int(max(data.chunks[0]))
        except Exception:
            source_chunk = None
    per = frames_per_task(n_frames, frame_nbytes, n_workers=n_workers,
                          source_chunk=source_chunk)
    chunks = _time_chunks(n_frames, per)

    if lazy:
        # Only the TIME axis is re-chunked, and only ever to a SMALLER size —
        # splitting a chunk is a slice, never the multi-GB shuffle CLAUDE.md
        # forbids. The signal axes are already whole (guarded above).
        da_data = data if tuple(data.chunks[0]) == chunks \
            else data.rechunk({0: chunks})
    else:
        da_data = da.from_array(data, chunks=(chunks, -1, -1))

    starts = np.concatenate([[0], np.cumsum(da_data.chunks[0])[:-1]]).astype(int)

    import functools
    block_fn = functools.partial(_map_block, spec=spec, scale=scale,
                                 store_masks=store_masks)
    result_array = da.map_blocks(
        block_fn, da_data, dtype=object, drop_axis=[1, 2],
        chunks=(da_data.chunks[0],), meta=np.empty((0,), dtype=object))

    def _assemble(out, nav_slices, chunk_result):
        out[nav_slices[0]] = chunk_result

    if client is None:
        _threaded(result_array, starts, stopped, record)
        return True

    from spyde.compute_dispatch import dispatch_chunks, split_workers_for_gpu

    scribble = str(spec.method).lower() == "scribble"
    lane_mode = PARTICLE_GPU_LANE_DEFAULT if scribble else "off"
    gpu_only = False
    gpu_addrs, cpu_addrs = ([], [])
    if lane_mode != "off":
        gpu_addrs, cpu_addrs = split_workers_for_gpu(client, lane_mode)
        if gpu_addrs and not cpu_lane_enabled():
            # GPU-ONLY, like the neural batch and for the same measured reason
            # (see cpu_lane_enabled). The lane list is then pinned HARD below,
            # because with allow_other_workers a busy GPU lane silently leaks
            # chunks onto the CPU workers — which is the slow path we just
            # decided against.
            cpu_addrs, gpu_only = [], True
    if not gpu_addrs:
        # No GPU lane (classical, no CUDA, or SPYDE_FV_GPU=off/all): every
        # worker is a plain CPU worker. Passing them ALL as the cpu lane is the
        # established shape for that case (orchestrate._retry_neural_on_cpu),
        # and keeps one code path instead of two.
        try:
            cpu_addrs = list(client.scheduler_info(n_workers=-1)["workers"])
        except Exception:
            cpu_addrs = []
        lane_mode = "off"
    if not cpu_addrs and not gpu_addrs:
        _threaded(result_array, starts, stopped, record)
        return True

    log.info("[seg-batch] %d frames in %d task(s); lanes GPU=%d CPU=%d "
             "(mode %s%s, method %s)", n_frames, len(da_data.chunks[0]),
             len(gpu_addrs), len(cpu_addrs), lane_mode,
             ", gpu-only" if gpu_only else "", spec.method)

    # `record` files every block from the done-callback as it lands, so the
    # dispatcher's own assembled array is redundant here and is dropped — and
    # its None-on-cancel return is NOT a "could not dispatch": the run was
    # handled, it just stopped early, and re-running it serially would be the
    # opposite of what cancel means.
    dispatch_chunks(
        client, result_array, 1, gpu_addrs, cpu_addrs,
        stopped_flag=stopped, fill_value=None, label="segment",
        on_chunk_done=lambda sl, blk: record(int(sl[0].start), blk),
        lane_default_mode=lane_mode, gpu_only=gpu_only, assemble=_assemble,
    )
    return True


def _threaded(result_array, starts, stopped, record) -> None:
    """Local fallback: a thread pool over the same blocks.

    Used for tests and ``SPYDE_NO_DASK=1``. Threads and not processes because
    the point here is only that the fallback is not the retired serial loop —
    real parallelism is the cluster's job, and the heavy stages (scipy's EDT,
    watershed, ``ndi.label``, torch) release the GIL.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_workers = max(1, min(8, (os.cpu_count() or 4) - 1))
    blocks = []
    for i, size in enumerate(result_array.chunks[0]):
        t0 = int(starts[i])
        blocks.append((t0, result_array[t0:t0 + int(size)]))
    with ThreadPoolExecutor(max_workers=n_workers,
                            thread_name_prefix="seg-block") as pool:
        futs = {pool.submit(b.compute, scheduler="threads"): t0
                for t0, b in blocks}
        for fut in as_completed(futs):
            if stopped is not None and stopped[0]:
                for f in futs:
                    f.cancel()
                return
            record(futs[fut], fut.result())


def _serial(get_frame, spec, n_frames, scale, store_masks, stopped,
            record) -> None:
    """Last resort: the original serial loop, for a frame source that is not an
    array at all (a callable / a sequence). Kept so ``segment_movie`` is total."""
    from spyde.particles.measure import measure_frame

    if get_frame is None:
        raise TypeError("segment_movie needs either a 3-D array or get_frame")
    engine, _dev = resolve_engine(spec)
    for t in range(int(n_frames)):
        if stopped is not None and stopped[0]:
            return
        frame = np.asarray(get_frame(t))
        labels = _engine_with_cpu_fallback(engine, frame, spec)
        rows, contours = measure_frame(labels, frame, t=t, scale=float(scale))
        record(t, [(np.ascontiguousarray(rows, np.float32),
                    list(contours) if store_masks else [])])


# ── shipping a trained head to the workers ───────────────────────────────────

def save_engine_model(classifier, directory: str | None = None) -> str:
    """Write *classifier* somewhere every worker can read it, return the path.

    A run-scoped temp file rather than a stable name: a retrain must not be
    served from a worker's cache under the same key, and two concurrent runs
    must not overwrite each other's head mid-flight.
    """
    import tempfile
    fd, path = tempfile.mkstemp(prefix="spyde-scribble-", suffix=".npz",
                                dir=directory)
    os.close(fd)
    os.unlink(path)                    # np.savez_compressed appends .npz itself
    base = path[:-4] if path.endswith(".npz") else path
    classifier.save(base)
    out = base + ".npz"
    log.debug("[seg-batch] scribble head written to %s (%.1f kB)", out,
              os.path.getsize(out) / 1e3)
    return out


def drop_engine_model(path: str | None) -> None:
    """Delete a model file written by :func:`save_engine_model`, best effort."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError as exc:
        log.debug("[seg-batch] removing %s failed: %s", path, exc)

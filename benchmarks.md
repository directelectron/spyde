# SpyDE Benchmarks

End-to-end workflow timings on real data, the dask limits they exposed, and
the workarounds now in the codebase.  Reproduce any number with:

```bash
python spyde/tests/benchmark_workflow.py "D:\...\20260331_125040_2756500_0_movie.mrc" --nav 256 256
# options: --quick (64x64 crop), --gpu off|one|N|all, --skip-vectors, --json out.json
```

The harness times every stage a user experiences, samples cluster memory and
disk spill throughout, and runs a GIL-heartbeat thread whose largest gap is a
direct proxy for GUI freeze (graph construction runs in a thread of the GUI
process and holds the GIL).

## Test system

| | |
|---|---|
| CPU | 48 logical cores (dask: 11 workers × 4 threads, app rule) |
| RAM | 137 GB (worker memory limit 10 GB each) |
| GPU | NVIDIA TITAN X (Pascal, 12 GB), numba-cuda kernels |
| Source drive | AVAGO MR9361-8i hardware RAID — **1.4–1.6 GB/s** sequential |
| Dataset | 60 GB `.mrc` movie, 256×256 scan × 512×512 float32 patterns (1 MB/pattern, 65 536 patterns) |

Note on caching: after a first pass, Windows keeps a large part of the file in
page cache, so steady-state read throughput measured ~3–3.7 GB/s — above raw
disk speed.  Cold-cache full passes are bounded by the **~49 s** disk floor.

## Headline results (final configuration)

| Stage | Time | Throughput | Notes |
|---|---|---|---|
| Raw disk read | — | 1.4 GB/s | physics floor: 49 s per full pass |
| Cluster startup | 3.4 s | — | LocalCluster construction; workers keep registering for a few more seconds |
| Lazy file open | 5.6 s | — | rsciio mrc header + dask array construction; happens once at File→Open |
| Navigator (sum) | 18.8–23.5 s | **2.9–3.7 GB/s** | the "as fast as data loads" baseline |
| Single frame fetch | 40–60 ms | — | navigation responsiveness |
| Virtual image | 23.1–23.4 s | **2.9–3.0 GB/s** | identical to navigator — pure data-rate bound ✔ |
| **Find vectors (full 60 GB)** | **186 s** | **0.37 GB/s** | 1.44 M vectors, auto params, **zero spill**, 576 chunks (GPU lane 246 / CPU 330) |
| Find vectors, time to first count-map paint | ~10 s | — | ≈ one chunk duration + initial loads (see below) |
| Largest GUI-thread stall during vectors | ~0.6–1.0 s | — | dask graph build holding the GIL; navigator/vimage stages stay under 200 ms |

**Interpretation.** Virtual imaging hits the data-rate wall — it cannot go
faster without faster storage.  Peak finding runs at 8× that wall-time, and
that gap is now **compute**, not dask overhead: NXCORR on a 512×512 pattern
costs ~76 ms on one CPU core (~5 000 core-seconds for the whole scan) and
~2.6 ms on the GPU, plus an intrinsic ~2.4× input amplification from the
ghost-zone overlap the nav-space blur requires.

## What each fix was worth (same dataset, measured)

| Configuration | Vectors time | Spill |
|---|---|---|
| chunk_nav=4 (100 MB budget, no floor): 4 096 chunks, 6.2× overlap, per-chunk submits | quick-crop only: 0.15 GB/s, 9.7 s before first task | — |
| Storage-aligned chunks (11×11 kept), but `scheduler_info` truncation → only 5 of 11 workers used | 184 s | 0 |
| Same, wide in-flight caps (threads + n_workers) | 262 s | 6.7 GB |
| Plain dask single-future, CPU only (`SPYDE_FV_GPU=off`) | 313 s | 9.3 GB |
| chunk floor 3×depth → chunk 9 **misaligned** with storage 11 → rechunk shuffle | 419–558 s | 6–7 GB |
| **Final: aligned chunks, all 11 workers, tight caps, batched submits, GPU lane** | **186 s** | **0** |

Two answers the table settles:

- *Is the custom dispatcher slower than dask's own scheduling?*  No — plain
  dask (no dispatcher, no GPU) is 313 s with the most spill.  The dispatcher
  costs one batched `client.compute` per 8 chunks (~ms) and buys GPU
  utilization that dask's scheduler structurally cannot (it keeps one duration
  estimate per task family, so it can't learn one worker is 30× faster).
- *Where did the spill come from?*  In-flight ghost blocks.  Each pending
  chunk pins a ~290 MB ghost-padded input on a worker; wide prefetch caps put
  workers past distributed's spill threshold (60 % of the limit) and the run
  slows ~40 %.  Caps are now `threads + 2` per lane.

## Dask limits found, and the workarounds (all in `find_vectors.py` unless noted)

1. **`Client.scheduler_info()` silently truncates to 5 workers**
   (distributed ≥ 2024, `n_workers=5` default).  Anything that enumerates
   workers from it — lane splits, `DaskManager.heavy_workers` — saw 5 of 11
   workers and quietly idled the rest.  *Workaround:* always call
   `scheduler_info(n_workers=-1)`.

2. **`map_blocks`/`map_overlap` without `meta=` executes the chunk function
   on zero-size arrays in the client process** for type inference.  Our GPU
   path launched an empty CUDA grid → `CUDA_ERROR_INVALID_VALUE` on every
   compute, misattributed for a long time to worker-side races.
   *Workaround:* pass `meta=`, and short-circuit `size == 0` blocks.

3. **Rechunking that crosses storage-chunk boundaries is a shuffle.**
   Rechunking 11×11-stored nav chunks to a "better" 9 or 12 made every
   ghost block gather split pieces from many source chunks: 419 s vs 186 s,
   plus spill.  *Workaround:* adopt the stored chunking whenever its size is
   within ~2× of the target (`keep_limit`); only rechunk pathological layouts
   (e.g. `hs.load(lazy=True)` defaults chunk the *signal* axes of mrc —
   (76,76,76,76) — which must be rechunked or, better, loaded right:
   the app's `chunks=("auto","auto",-1,-1)` is correct).

4. **Tiny cores under big ghost halos multiply IO and memory.**  The 100 MB
   chunk budget gave a core of 4 nav pixels under a depth-3 halo: 6.2×
   overlap overhead and 4 096 chunks.  *Workaround:* floor the core at
   3×depth (≤ 2.8× overhead) — but rule 3 outranks this floor.

5. **Per-`compute()` graph cost scales with chunk count.**  One submit per
   chunk took ~10 s before the first task ran (graph cull/optimize per call).
   *Workaround:* submit in batches of 8 (`SUBMIT_BATCH`) — one optimize pass
   per batch.

6. **Hard worker restrictions deadlock across worker restarts.**  A worker
   that hits its memory limit is restarted by the nanny under a *new
   address*; tasks pinned with `allow_other_workers=False` become
   unschedulable forever — count map full, Stop button forever, no vectors.
   *Workaround:* `allow_other_workers=True` (preference, not pin) plus a
   600 s no-progress watchdog that surfaces an error instead of hanging.

7. **Releasing futures mid-run races the scheduler when graphs share keys**
   (`KeyError` on forgotten keys during concurrent `update_graph`).
   *Workaround:* hold every chunk future until the run ends; make holding
   cheap by compacting each chunk result on the worker
   (`_compact_padded_chunk`: ~30 real peaks instead of 512 NaN slots).

8. **Cluster startup is asynchronous.**  `LocalCluster()` returns in ~3.5 s
   but workers keep registering afterwards (the app intentionally scales
   1 → N in the background).  The dispatcher refreshes its lanes every 5 s so
   late workers join mid-compute.

9. **Heavy graph construction freezes the GUI via the GIL.**  Measured worst
   stall ~1.0 s on the main thread during the vectors graph build (navigator
   and virtual image stay < 200 ms).  The ~8 s the user perceives between
   pressing Compute and "something happens" is **time-to-first-chunk**
   (initial loads + one chunk duration ≈ 10 s), not a GUI freeze —
   `[find_vectors] ui:<stage>` timings now print on every Compute click to
   verify in-app.

10. **GPU sharing.**  Multiple worker *processes* on one GPU time-slice
    CUDA contexts (WDDM) and get slower in aggregate; numba kernel launches
    from multiple threads of one process race (intermittent
    `INVALID_VALUE`).  *Workarounds:* one designated GPU worker
    (`SPYDE_FV_GPU`, default worker "1"), a process-level exec lock,
    `cache=True` kernels (first-chunk JIT 0.6 s instead of 3 s), and a
    serialized warmup probe that exercises every kernel on a tiny block.

## CuPy/cuFFT correlation + CUDA streams (follow-up, measured)

The GPU lane now computes NXCORR with batched cuFFT + integral images
(`_nxcorr_fft_cupy`, float64 accumulators — *more* accurate than the CPU's
float32 integral images) when CuPy is installed, falling back to the numba
kernels otherwise (`SPYDE_FV_GPU_FFT=0` forces the fallback).  Chunks run on
a fixed pool of CUDA streams instead of a process-wide lock, with a slot
semaphore (`SPYDE_FV_GPU_CONC`, default 2) bounding device-section
concurrency; the CPU pack stage runs outside the slots.

Measured (512² patterns, kr=8, 121-frame chunks, TITAN X):

| | chunks/s |
|---|---|
| numba tiled (post f32-accumulator fix) | 5.5 |
| cuFFT | 5.5 |
| cuFFT, 4 threads / 2 slots | 4.3–4.6 sustained |

Honest findings:

- **At typical radii on Pascal, cuFFT ≈ tiled brute force** — the tiled
  kernel got fast enough that kr≈8 sits at the crossover.  cuFFT's win is at
  **large radii** (taps grow as kr², FFT cost doesn't): beyond kr=23 the old
  path fell back to a naive kernel measured ~25× slower; cuFFT now covers
  that range at full speed.  Newer GPUs (much higher FP32 FFT throughput
  than Pascal) shift the crossover toward FFT.
- **Streams don't speed up this workload** (device already saturated by one
  chunk's kernels) but remove the serial lock, overlap the CPU pack with the
  next chunk's GPU work, and are the prerequisite for pinned-buffer H2D
  overlap later.
- **60 GB end-to-end is unchanged (192 s vs 186 s)** — at full-cluster scale
  the GPU lane is bounded by loading its input chunks on worker 1, not by
  kernel time.  The next real lever for the lane is feeding it (pinned-
  buffer async loads, or a reader thread on the GPU worker), not more FLOPs.

Two more pitfalls worth recording:

- **CuPy keeps per-stream arenas and per-thread cuFFT plan caches.**  A new
  stream per short-lived thread leaks VRAM until the device thrashes
  (measured progressive collapse 4.0 → 0.4 chunks/s across runs).  Fix:
  a fixed pool of 4 long-lived streams handed out round-robin.
- **A too-small device buffer pool is worse than none**: returns get
  rejected, every drop is a `cudaFree`, and frees are device-wide syncs.
  The pool cap is now half of total VRAM.

## Pinned-buffer async H2D (follow-up, measured)

Ghost blocks now stage through a pooled page-locked buffer
(`_pinned_pool_get`, capped at 3 GB, graceful pageable fallback) before
upload: the `np.copyto` into pinned memory happens *outside* the GPU slot
semaphore, and the subsequent `copy_to_device(..., stream=)` from pinned
memory is genuinely asynchronous — the worker thread is freed during the
DMA and the transfer overlaps other chunks' kernels.  Correctness verified
to 1e-4 px / 0.0 score deviation vs the CPU path on identical input.

Result on the 60 GB dataset: **187 s — unchanged within noise** (186–192 s
across all configurations of the final pipeline).  This confirms the lane
analysis: worker 1's chunk *loading* (dask read + ghost assembly tasks
sharing the worker's 4 threads) is the lane ceiling, not H2D latency or
kernel time.  The staging machinery is kept — it costs nothing, helps
compute-dominated datasets, and is the prerequisite for any future
reader-thread design that loads GPU-lane inputs directly into pinned
buffers (bypassing dask's pageable load tasks — the change that would
actually attack the load bound).

## Known remaining costs and future levers

- **Peak finding is compute-bound cluster-wide** at ~0.37 GB/s (8× the
  data-rate wall); the GPU lane specifically is **load-bound** on worker 1.
  Remaining levers, in order: the direct-read fast path below, a second GPU
  with `SPYDE_FV_GPU=2`, more worker-1 threads.

### Parked design: direct memmap→pinned reads for the GPU lane

The biggest remaining lever (est. 187 s → ~120–140 s on the 60 GB run).
Today a GPU-lane chunk's input is produced by ~10 dask tasks (storage-chunk
loads + ghost assembly) that compete with the GPU tasks for worker 1's four
threads and touch the bytes three times in pageable RAM (lane overhead
~0.58 s/chunk vs ~30 ms warm-cache raw read time).  For flat memmap-able
formats (mrc/raw) the whole subgraph can be replaced by ONE
`client.submit(_direct_gpu_chunk, file_meta, ghost_coords, ...)` task that:

1. opens a per-worker cached `np.memmap` (path/offset/dtype/shape carried as
   metadata, recoverable from the dask graph leaf or the loader),
2. reads the ghost-padded nav slice directly into a pooled pinned buffer
   (one copy, no assembly tasks, no sliver transfers), emulating
   `boundary="reflect"` at scan edges via index math,
3. runs the existing pinned→async-H2D→kernel pipeline and the shm count
   write, returning compacted peaks.

Dask stays the scheduler/transport; only the lane's IO pipeline is bypassed.
CPU lane and compressed formats (.hspy/zarr) keep the normal graph —
automatic fallback when no flat layout exists.  Must-haves before shipping:
equality test vs the dask path (reflect edges, float32 conversion) and the
memory-safety contract (each task reads only its ghost slice — Path B).
- GPU knobs: `SPYDE_FV_GPU` (lane policy), `SPYDE_FV_GPU_FFT=0` (force numba
  kernels), `SPYDE_FV_GPU_CONC` (device-section slots, default 2),
  `SPYDE_FV_GPU_SERIAL=1` (legacy whole-chunk lock), `SPYDE_FV_TIMING=1`
  (accurate per-stage GPU timings).
- **Lazy mrc open costs ~6 s** (header parse + graph construction) — paid at
  File→Open, worth profiling inside rsciio if it bothers.
- **Auto-chunking guidance:** store/convert 4D-STEM data with nav-only
  chunking near the ghost-block budget (e.g. (11,11,512,512) ≈ 121 MB for
  512² float32).  The pipeline adopts good stored chunking as-is.
- Numbers above are warm-cache; first-ever pass on a cold file adds up to
  ~49 s of pure disk time on this RAID (more on slower drives).

---

## Vector Orientation Mapping (2026-06-15)

Orientation + strain from sparse diffraction vectors (soft-assign + sink LM fit
over pose theta,A,t). Reproduce:

    python -m spyde.tests.benchmark_vector_orientation

Hardware: this machine (48 cores), single-core fits. Real data: sped_ag (Ag FCC
m-3m, mostly [100]), 1081-template library (res 1 deg, r_max 0.75).

### Per-pattern fit (10x14 sped_ag region)

| metric            | value           |
|-------------------|-----------------|
| time              | ~39 ms/pattern  |
| strain median     | 0.008           |
| residual          | 0.34 detector px |
| Friedel asymmetry | 0.0067 (low = well-centered) |

Full 64x208 scan: ~9 min single-core, ~1 min on the cluster (GPU/CPU lanes).

### Strategy comparisons (what helped, what didn't)

- **Warm-start propagation: WORSE** (§7e). 100 vs 39 ms/pat AND strain median
  0.046 vs 0.008 — the bounded affine absorbs a wrong neighbour-seed as spurious
  strain. Defaults OFF.
- **Friedel symmetry as a FIT CONSTRAINT: no help** (§7f). center pre-correction,
  pair denoising, symmetrized residual — all match the independent fit within
  noise. The affine already encodes centrosymmetry (A linear, t = beam center
  absorbs miscentering to >=3 px). Friedel kept only as the QC metric.
- **Edge-preserving strain-field smoothing: HELPS** (§7g). On a synthetic field
  with a grain boundary + noise: independent strain err 0.0056; median 3x3 0.0027
  (2x better, boundary preserved); Gaussian 0.0019 but over-blurs the boundary.
  Added as `VectorOrientationResult.smoothed_strain()` (median 3x3), default-on
  in the Run tab. This is the architecturally clean form of neighbour coupling:
  post-fit, edge-preserving, no wrong-branch absorption.

### Robustness: high-noise / few-spot field denoising (2026-06-15)

Which field method survives when independent per-pattern fits start to fail?
Synthetic strain field (gradient + grain boundary) swept over per-spot noise and
spot dropout; error to ground truth (`run_robustness`):

| noise/drop | independent | median | TV     | iterated |
|------------|-------------|--------|--------|----------|
| 0.010/0.0  | 0.0136      | 0.0056 | 0.0064 | 0.0264   |
| 0.020/0.0  | 0.0260      | 0.0147 | 0.0074 | 0.0260   |
| 0.020/0.3  | 0.0266      | 0.0137 | 0.0082 | 0.0278   |
| 0.035/0.3  | 0.0288      | 0.0138 | 0.0069 | 0.0283   |
| 0.050/0.4  | 0.0265      | 0.0121 | 0.0045 | 0.0274   |

- **TV (Chambolle) wins; the gap widens with noise** — 6x better than
  independent and 2.7x better than median at the worst case (0.05 noise, 40%
  dropout). Now the default for `smoothed_strain()`.
- **Iterated fit->TV->refit is WORST** — same failure as warm-start: per-pattern
  re-fitting lets the affine re-absorb noise, undoing the smoothing. Post-fit
  field denoising beats per-pattern joint coupling. The "full dataset fit" is the
  field-level TV solve, not per-pattern coupling.

### Nav-dimension denoise for peak finding: TV vs Gaussian (2026-06-15)

The pipeline blurs across the scan (nav) axes before NXCORR peak finding (adjacent
probe positions see near-identical patterns). TV vs Gaussian nav-denoise, synthetic
4D with two grains + sharp boundary + Poisson noise (`benchmark_peak_denoise.py`),
detection F1 vs known spots (tuned to F1=1.0 on clean data):

| dose            | none  | gaussian | TV    |
|-----------------|-------|----------|-------|
| medium          | 0.976 | 0.978    | 0.973 |
| low             | 0.974 | 0.978    | 0.979 |
| very_low (4 ct) | 0.643 | 0.977    | 0.982 |
| very_low bndry  | 0.650 | 0.984    | 0.992 |

- At adequate dose nav-denoise barely matters (NXCORR matched filter suffices) and
  both slightly worsen sub-pixel error.
- **At very low dose nav-denoise is essential** (F1 0.64 -> 0.98).
- **TV edges Gaussian, advantage concentrated at grain boundaries** (Gaussian
  smears orientations across the boundary; TV's edge-preserving prior respects
  it). Gap ~1-2% F1; Gaussian keeps better sub-pixel precision (0.48 vs 0.63 px).
- Verdict: keep Gaussian default; offer TV nav-denoise for low-dose data with
  sharp grain structure. Dose/microstructure-dependent, not a universal win.

## In-situ movie playback — per-frame stage timings (Phase 0, 2026-07-05)

`benchmark_movie_playback.py` on a **real Direct-Electron in-situ movie**
`20251117_88074_run1_9104_movie.mrc` = **(3618, 4096, 4096) uint8**, 60.7 GB —
a 3618-frame movie of **4k×4k image frames** (16.8 MB/frame), nav-dim 1 (time).
This is the case the 4D-STEM-oriented live-display path was NOT built for.
Frames sampled across the stack (crossing chunk boundaries); cold OS cache.

| stage | mean ms | what |
|---|---:|---|
| `memmap`    |  44 | `np.memmap mm[t]` -> RAM — the proposed playback read |
| `normalize` | 185 | -> uint8 (anyplotlib `set_data` / `_normalize_image`) |
| `getinds`   | 251 | hyperspy `_get_cache_dask_chunk` — **the current live-display call** (threaded) |
| `b64`       | 268 | + base64 encode (transport payload) |
| `compute`   | 275 | dask `raw[t].compute()` (threaded) |
| `json`      | 323 | + `json.dumps` PLOTAPP line (the full transport step) |

Frame in RAM **16.8 MB**; transport payload (b64-in-JSON) **22.4 MB/frame**
(scales to ~85 MB at 8k×8k).

**Findings (these drive the rewrite):**
- **Dask is the wrong tool for sequential movie reads.** `compute()` is **6.2×**
  the raw memmap read (275 vs 44 ms) and the current live-display `getinds` call
  is **~5.7×** (251 vs 44 ms) — because the reader auto-chunks 8 frames × full
  4096² = a **128 MB chunk read per single frame**. The plan's `nav_chunk=32`
  re-chunk would make it a **512 MB chunk** — worse. Confirms the "dask might be
  an issue" suspicion; motivates the direct `np.memmap` playback read (Phase 2)
  and per-frame-size-adaptive chunking (Phase 1).
- **Transport dominates the rest.** normalize->b64->json is **185->268->323 ms**
  and ships **22.4 MB/frame** of base64-in-JSON. Motivates the binary transport +
  GPU-shader colormap (Phases 4-5, in anyplotlib).
- **Current total ≈ getinds (251) + json transport (323) ≈ ~570 ms/frame -> under
  2 fps** on a 4k movie. Target (memmap 44 + binary + GPU render) removes ~500 ms
  of that. The ordering says: fix the read path AND the transport; normalize/LUT
  moving to the GPU removes the ~185 ms normalize too.

**Pure 8k×8k transport** (`--synthetic 8192`, no disk): normalize **729 ms** ->
b64 **1037 ms** -> json **1288 ms**, payload **89.5 MB/frame** (67.1 MB frame in
RAM). i.e. the transport chain ALONE is >1 s/frame (<1 fps) at 8k before any read
or render — the base64-in-JSON-over-stdout scheme cannot carry an 8k movie. This
is the hard case for the binary transport + GPU-shader colormap.

Run: `.venv/Scripts/python spyde/tests/benchmark_movie_playback.py --frames 20`
(`--path <file.mrc>`, or `--synthetic 8192` for pure 8k transport numbers).

### Pooled sync-graph movie read: futures + cancel WITHOUT distributed (2026-07-05)

The dilemma: the navigator needs **cancellable async futures** (latest-wins scrub)
but the distributed scheduler's per-frame round-trip is slow (251 ms above), while
the plain threaded `.compute()` is fast but blocking (no Future/cancel). Resolution
(`repro_movie_scrub.py` on the real 4k movie; `ComputeBackend.submit_graph`): submit
`lazy[t].compute(scheduler="synchronous")` to OUR ThreadPoolExecutor — the pool
gives the real `concurrent.futures.Future` (cancel + done_callback), the synchronous
scheduler walks the dask graph on that worker (no nested pool), no distributed hop.

| property | distributed (today) | **submit_graph (pool+sync)** |
|---|---|---|
| per-frame latency | 251 ms | **46 ms** (min 20) |
| cancel a queued scrub frame | yes | **yes (17/19 cancelled)** |
| async done-callback paint | yes | **yes** |
| scrub a lazy CROP (`inav/isig`) | yes | **yes, same path** (crop build 14.5 ms lazy; cropped frame read 48 ms) |

So we keep the full dask graph (crop/rebin/zspy all read through one path) and own
the async layer via the executor already in `ComputeBackend`. ~5.5x faster per frame
than the current distributed live-display call, and crop-then-scrub is free. This is
the basis for the Phase-2 movie navigator read.

### 4D-STEM DP nav A/B: DO NOT unify onto submit_graph (2026-07-05)

Gate benchmark before unifying the 4D-STEM diffraction-pattern navigator onto
submit_graph (`benchmark_nav_read_ab.py`, real 4D-STEM `20241219_29674_movie_movie.mrc`,
300×648 nav × 128² DP). Correctness: submit_graph of `raw[iy,ix]` (single) /
`raw[pts].mean(0)` **float64** (region) matches hyperspy `get_index(sum_data=True)`
exactly — the region result is the un-rounded float64 mean, NOT cast to frame dtype.

| case | current `get_index` | submit_graph |
|---|---:|---:|
| single point | **1.2 ms** | 24.6 ms |
| region (25 pts, integrating) | **3.5 ms** | 52.0 ms |

**The two navigator paths have OPPOSITE optimal strategies:**
- **4D-STEM DP**: navigation dwells WITHIN a nav chunk (adjacent probe positions
  share a 32×32 chunk). `get_index` caches the loaded chunk in numpy, so ~every move
  is a ~1 ms cache hit. submit_graph re-walks the graph + re-reads from disk each
  move (~24 ms) → **~15-20x slower**. Do NOT unify — it would badly regress the DP
  live display (CLAUDE.md §1-4).
- **Movie**: consecutive frames are in DIFFERENT chunks (1 frame/chunk), so the
  get_index chunk cache never helps; submit_graph's direct read wins (46 vs 251 ms).

**Conclusion (superseded — see below):** ~~keep distributed for DP, submit_graph for
movie~~. A better unification exists.

### The unified read: cached get_index MINUS distributed (2026-07-05)

Key realisation (from reading `CachedDaskArray.get_index`): the numpy chunk cache
that makes the DP navigator fast is **independent of the distributed scheduler**.
With `self.client` unset, `get_index` takes a **synchronous** branch
(`array_tools.py` ~L1067/1134) that caches blocks in numpy and slices/means them —
the SAME cache logic, no distributed hop. Measured on the 4D-STEM scan with the
cache client set to None: dwell-in-chunk **1.31 ms** (min 0.80), cross-chunk 45-100 ms.

So `cached_read` = `get_index` (no distributed client, synchronous chunk cache)
submitted to our own ThreadPoolExecutor gives a **cancellable async Future** AND the
cache hits, with NO distributed overhead. A/B (`benchmark_nav_read_ab.py`):

| case | distributed (today) | naive submit_graph | **cached_read (unified)** |
|---|---:|---:|---:|
| single DP | 1.6 ms | 25.0 ms | **1.0 ms** |
| region (25 pts) | 2.3 ms | 52.9 ms | **1.5 ms** |

All outputs match. This **unifies both navigators** on one path (cache logic kept,
distributed dropped): 4D-STEM DP dwells in chunk → ~1 ms hits; a movie is 1
frame/chunk → every move a ~46 ms cold read (already optimal). It's also SIMPLER
than today — no shm buffer, no distributed-client pinning, no `_inflight_getinds`
juggling — and must run on the serial `_NavDispatcher` (CLAUDE.md §4: the cache is
not concurrency-safe; the dispatcher already serialises). This is the Phase-2 design.

### The `_client=None` pin needs a property patch to actually take effect (2026-07-06)

A review caught that `cached_arr._client = None` **does not** force the synchronous
cache branch when a real cluster is up (the app default). The fork's
`CachedDaskArray.client` property, when `_client is None`, calls
`dask.distributed.get_client()` — which returns the process-global default `Client`
from any non-worker thread (it does NOT raise). So the nav read still went
distributed in the app; only `SPYDE_NO_DASK=1` tests hit the sync branch.

Measured on a real `LocalCluster` (default `Client`), 64×64×128² 4D-STEM, from a
`nav-dispatch` thread:

| | dwell-in-chunk | cross-chunk |
|---|---:|---:|
| `_client=None` alone (still distributed) | **~16 ms** | ~103 ms |
| + `_patch_cached_dask_client()` (true sync) | **~2 ms** (min 0.7) | (one chunk read) |

Fix: `heavy_imports._patch_cached_dask_client()` (applied in `ensure_heavy_imports`)
makes `.client` honour `_client=None` (no `get_client()` fallback). Verified with a
real cluster (2 ms dwell, correct frame) and in the app (DP nav 9/9 moves update).
The RETIREMENT of the shm/cancel machinery was always safe — it came from
**seriality + blocking**, not the branch; the patch just makes the read fast too.

### LOD decimation + read-ahead prefetch (Phase 3, 2026-07-05)

Two wins for a large-movie scrub, both benchmark-driven on the real 4k movie:

- **LOD (transport):** strided *reads* do NOT save disk I/O on a contiguous memmap
  (`raw[t,::4,::4]` = 45 ms vs 47 ms full — the OS reads full pages regardless), but
  decimating an already-read frame in numpy is ~1 ms and cuts the base64-in-JSON
  transport (the dominant per-frame cost) by the square of the stride. So
  `Plot._set_array` decimates any frame whose longest side > 1536 px (4096→1366,
  stride 3; 8192→1366, stride 6) before the wire, subsampling the axes identically
  to keep the scale bar calibrated. A DP frame (≤1536) is untouched.
- **Prefetch (cold read):** a warm (already-paged) frame re-reads in ~18 ms vs ~50 ms
  cold. `_MoviePrefetcher` reads t±1…t±3 on a background thread after each movie move
  to warm the OS page cache, so a steady scrub finds the next frame warm. It reads the
  RAW dask array (not the CachedDaskArray) so it never races the nav read's cache.

Verified in the real app: movie scrub 5/5 moves repaint a fresh decimated 4k frame
(scale bar correct) in ~23 s incl. cold load; LOD + prefetch coexist with no DP
regression.

### Real-cluster (`--distributed`) A/B + a rounding gotcha (2026-07-05)

Ran the A/B against a real `LocalCluster(processes=True)` (CLAUDE.md: won't run in
an agent sandbox — run directly). Cold reads through the cluster:

| case | distributed get_index | submit_graph | cached_read (no client) |
|---|---:|---:|---:|
| single | 13.3 ms | 23.9 ms | 18.9 ms |
| region-25 | 26.8 ms | 48.6 ms | 25.2 ms |

cached_read is competitive-to-better than the real distributed path even cold, and
~1 ms on warm dwell-in-chunk hits (threaded run) — so unifying does NOT regress the
DP navigator.

**Gotcha (must handle when unifying): the two get_index branches round the
integrating-region mean differently.**
- Distributed branch → `weighted_mean_round_from_sums` → for INTEGER dtype,
  `np.rint(mean).astype(dtype)` (rounded uint16, e.g. 106).
- No-client branch → `np.mean(arrays)` → **float64**, un-rounded (106.444...).

So a naive unify would change the DP navigator's region frames from rounded uint16
to float64 (shifting contrast/levels). The unified read must reproduce the
distributed rounding for integer data (round-to-dtype on the region mean). Small
fix, but it's a real display-correctness constraint — noted for the Phase-2 rewire.
(Single-frame reads match exactly; only the multi-point integrating region rounds.)

### First paint of a 16 GB movie + navigator sidecar (2026-07-10)

Repro: `D:\20251117_88075_run3 some growth_1236_movie.mrc` (977 x 4096 x 4096 uint8,
16.8 MB frames), fresh open with a running distributed navigator fill.

The signal panel sat BLACK long after the frame was read (`_set_array` entered at
~2.5 s): stage-timestamping showed the first paint's auto-level block took
**11.7 s** — `_emit_histogram` ran `np.isfinite` + `np.histogram` over the FULL
16 M-px frame while the fill's worker processes saturated memory bandwidth.
Subsampling the histogram input to ≤512² (like `_robust_levels`) + skipping the
isfinite mask for integer dtypes: **11,660 ms → 22 ms**. Second stall: lazy
`import torch` in `GpuTileBackend` on the painter thread (~3.3 s idle, worse under
load; and a sys.modules poke at a MID-IMPORT torch blocks on the import lock —
measured 11 s) → background `prewarm_torch_cuda()` at startup + flag-only
readiness + numpy-first tile that upgrades to CUDA when the prewarm lands.

End-to-end (first_paint_real_load.spec.ts, real file, dask):

| stage | before | after |
|---|---:|---:|
| signal panel first content (fresh open, fill running) | never (until nav move / fill end) | **5.3 s** (0.5 s after windows) |
| navigator on REOPEN (sidecar `<file>.spyde-nav.npz`) | ~52 s whole-file fill | **4.4 s** (real data at window-open) |
| navigator fill (977 x 16.8 MB read) | ~52 s | ~52 s + "Computing navigator… N%" status |

The fill is also DEFERRED until the signal plot's first frame lands
(`_start_nav_compute_after_first_frame`) so the first frame wins the disk, and the
completed fill writes the sidecar for the next open.

### ArrayCache reader kinds: read the store, not the dask graph (2026-07-24)

`spyde/array_cache/` replaces `_NavChunkCache` with a byte-budgeted frame cache
behind a `FrameReader` per backing kind. Which kind serves a signal is resolved
ONCE per node-select (`resolve.py`), so the per-frame path is just "read the
frame". Measured per frame on a 32x32x128x128 uint16 scan stored in 8x8-nav
chunks, warm page cache, frames spread one-per-chunk ("cold chunk") vs walking
new frames inside one already-touched chunk ("dwell"):

| backing | read strategy | cold chunk | dwell |
|---|---|---:|---:|
| `.zspy` | per frame from the store | 0.97 ms | 0.79 ms |
| `.zspy` | whole nav-chunk via dask (kind 4) | 1.64 ms | 0.009 ms |
| `.zspy` | **whole nav-chunk from the store (kind 2)** | **0.84 ms** | **0.002 ms** |
| `.hspy` | per frame from the store | 5.12 ms | 0.003 ms |
| `.hspy` | whole nav-chunk via dask (kind 4) | 5.97 ms | 0.009 ms |
| `.hspy` | **whole nav-chunk from the store (kind 3)** | **5.10 ms** | **0.002 ms** |

The non-obvious row is `.zspy` per-frame dwell: reading straight from the store
looks like the minimal read, but zarr re-decompresses the enclosing chunk for
EVERY frame — 88x worse than slicing a block that is already decoded, on the
pattern a 4D-STEM DP navigator spends all its time in. Reading the block
DIRECTLY from the store and memoizing it wins both columns (dask's route to the
same block costs ~0.8 ms of graph overhead). A CONTIGUOUS store, or a chunk
spanning 1 nav position (in-situ movie), or a block over
`MAX_BLOCK_BYTES` = 64 MB reads the frame alone — nothing to amortise.

Cost note: a raw `.mrc`/`.de5` (kind 1) skips all of this with one `os.pread`
per frame (`hasattr(os, "pread")`-gated; Windows takes the `np.memmap` slice).

Correctness trap worth remembering: a dask HighLevelGraph keeps every ancestor
layer, so a DERIVED view (rebin/crop of a memmap- or zarr-backed file — exactly
what the `local` locality tag makes cache-eligible) still carries the source's
`slice_memmap` / `original-array-*` tasks. Matching those served the RAW SOURCE
frame with the transform silently skipped (a nav crop read the wrong frame with
no error at all). Both finders now gate on `data.name` — the array must BE that
layer's output — plus a shape/dtype match.

### Vector-OM accuracy: what the coarse seed's asymmetry buys (2026-07-24)

Investigating "the vector orientation map results are a little off". Reference
throughout is the DENSE raw-OM template match on the same region/library/
calibration; the metric is IPF X/Y/Z colour agreement, which
`benchmark_om_parity` calls authoritative (the two result types carry different
quaternion conventions, so a cross-path misorientation angle is NOT comparable —
it reads ~50-60° even where the colours agree 100%).

**Where vector-OM actually stands** (sped_ag, 12x18 slab, 1081 Ag templates,
gamma=0.5): the production batched-torch path agrees with the dense reference at
**98% / 98% / 100%** (IPF X/Y/Z). The serial scipy path (`fit_pattern`, the
no-torch fallback) on the same vectors and library agrees at only
**25% / 4% / 67%** — the accuracy gap is between the two vector-OM paths, not
between vector-OM and the dense match.

**The trap: do NOT cosine-normalise the coarse-seed correlation.**
`_coarse_seed_batched` correlates polar signatures with a RAW inner product,
which looks like a bug — brighter templates score higher regardless of match
quality. Synthetic evidence says fix it; real data says the opposite:

| coarse seed | IPF-X | IPF-Y | IPF-Z | synthetic (patterns that ARE templates) |
|---|---:|---:|---:|---|
| raw inner product (shipped) | 98% | 98% | 100% | true template ranked top 1/40, field ~28° off |
| cosine-normalised | 0% | 28% | 0% | true template ranked top 40/40, field exact |

Real vector finding detects only the strong subset of reflections (~5-15 of ~25),
so dividing by the template norm charges the true template for spots that were
never detected and systematically favours sparse/low-multiplicity templates. The
synthetic test hands the fit every spot, giving both signatures equal support,
which is exactly the condition that makes cosine look right. Pinned by
`test_vector_orientation_seed.py`; the pre-existing GPU tests could not catch it
because they use a SINGLE-template library, where the argmax over templates is
trivially correct either way.

**FIXED: the scipy path resolved orientation from the wrong angle.**
`strain_from_pose` polar-decomposes `M = A.Rot(theta)` into `R.S` and reports
`S - I`, its docstring noting that "the rotation R is absorbed into the
orientation" — but nothing did that. LM freely splits the total rotation between
`theta` and the free 2x2 `A` (that freedom is *why* the strain needs the
decomposition), and `fit_pattern` resolved the quaternion from the bare `theta`,
so the reported in-plane orientation was short by whatever LM parked in `A`.
Textbook signature: zone axis right, in-plane wrong. Resolving from the
polar-decomposition rotation (`pose_in_plane_angle`) instead:

| variant (sped_ag 8x12, vs dense reference) | IPF-X | IPF-Y | IPF-Z |
|---|---:|---:|---:|
| scipy, own pyxem seed — before | 25% | 4% | 67% |
| scipy, own pyxem seed — after | 72% | 67% | 67% |
| scipy + the torch path's seed — before | 44% | 2% | 100% |
| scipy + the torch path's seed — after | 99% | 98% | 100% |
| batched torch path (unaffected) | 98% | 98% | 100% |

With a correct seed the scipy path now matches the production path exactly. The
batched torch path never had this bug: there `M = S.Rot(theta)` with `S` SPD by
construction, so its `theta` IS the physical angle. The residual gap in the
own-seed row is entirely the seed (IPF-Z 67%) — handing that path the batched
polar-histogram seed closes it, worth doing if the no-torch fallback or the live
refine overlay matters.

**Also measured and NOT shipped:** refining more than one coarse candidate. The
refine keeps the best-scoring candidate but is handed only the seed's top pick;
on synthetic ground truth `n_seed=1` returns a fit strictly worse than the true
one in 32/120 cases and `n_seed=3` fixes all 32. On real sped_ag it moved
agreement ~1 pt (25/4/67% -> 27/6/66%) for ~40% more time. Same for adding a
measured-side coverage term to the candidate score: neutral on real data. Both
reverted — the scipy path's accuracy limit is somewhere else, and that path is
the fallback anyway.

### Storage-aligned chunks at LOAD time, everywhere (2026-07-30)

RosettaSciIO hands the array to dask with `chunks="auto"`, and dask balances all
axes to hit its 128 MiB target with no idea which of them is navigation. On a
977 x 4096² uint8 in-situ MRC that is **(511, 511, 511)** — a 133 MB cube that
SPLITS the signal axes, so 4096 = 511x8 + 8 spreads every frame across **81
chunks**. `Session._signal_spanning_chunks` already computed the right answer
`(1, -1, -1)`; the bug was that only two of four load paths called it, and the
Examples path called it not at all.

Measured on that file:

| | reader default (511³) | `chunks=(1, -1, -1)` |
|---|---|---|
| nav chunks | 2 (each spanning ~8.5 GB of reads) | 977 |
| first nav chunk | **14.60 s** | **0.032 s** |
| whole nav sum | **24.37 s** | **4.52 s** (3.38 GB/s) |
| cost of the re-load | — | 0.08 s |

The re-load is a graph rebuild, not a data move — never `.rechunk()` instead.

Two things worth knowing before touching this:

* **`("auto", -1, -1)` is NOT strictly better than `(1, -1, -1)`.** Measured:
  auto gives 8 frames/chunk (134 MB) and wins the nav sum 4.84 s vs 5.57 s, but
  it makes every navigator scrub read 134 MB instead of 16.8 MB — an 8x
  regression on the interactive path, against a sum that runs once per file and
  is then cached in the `.spyde-nav.npz` sidecar. Hence the adaptive nav block.
* **Only `.mrc` honours `chunks=`.** Both `.hspy` and `.zspy` raise
  `TypeError: 'chunks' is an invalid keyword argument` — their chunking is fixed
  when the file is WRITTEN, so no load-time argument can reach it. That is why
  `load_aligned` catches and falls back: the fallback is load-bearing, and
  without it this change would make every hspy/zspy file fail to open.

### Navigator fill: the per-chunk submit loop vs the shared dispatcher (2026-07-30)

Real file: `20251117_88075_run3 some growth_1236_movie.mrc` — 977 x 4096² uint8,
15.27 GB, 1-D nav, loaded via `load_aligned` (1 frame/chunk => **977 nav chunks**).
Real `LocalCluster`, 4 worker processes x 2 threads. `--purge` evicts the file
from the Windows page cache (FILE_FLAG_NO_BUFFERING) before each run.
Harness: `spyde/tests/benchmark_nav_fill_dispatch.py`.

The old `compute_with_live_buffer` navigator branch did
`for slices in all_slices: client.compute(chunk)` — one blocking scheduler round
trip per nav chunk, all 977 up front, with the GIL held in the client process the
whole time — and then `client.compute(result_array)` again for the whole array.
It now routes through `compute_dispatch.dispatch_chunks` (batched submit, bounded
in-flight window, stall watchdog) and the result is ASSEMBLED from the chunks.

Cold (page cache purged before each run):

| | submit (client-side, GIL held) | first chunk painted | total fill |
|---|---|---|---|
| per-chunk loop | **9.86 s** | 16.41 s | 50.83 s |
| dispatch_chunks | **0.00 s** | **0.08 s** | 50.64 s |

Warm (one throwaway pass first, so I/O is held constant): submit 10.12 s -> 0.00 s,
total 50.21 s -> 46.53 s. Checksums MATCH in every run: the client-side assembly
is identical to the whole-array compute it replaced.

Three things this pins down:

* **The submit time is the bug.** 9.9 s during which nothing else in the backend
  process can run — the navigator sits blank and the paint threads go silent. It
  is client-side, so it is the same cold or warm. First-visible-pixel goes
  16.4 s -> 0.08 s (**200x**).
* **The bounded window costs nothing.** The window here is 4 (half of 8 cluster
  threads) versus the old path's 977-in-flight, and the total fill did not
  regress — it improved, because the old path also submitted the duplicate
  whole-array graph.
* **A progressive fill is ~7x a monolithic sum, and that is per-TASK overhead,
  not concurrency.** Warm, one `nav.compute()` of the whole graph is **6.4 s**;
  977 separate per-chunk futures are 46-50 s either way (~47 ms of scheduler
  round trip for a task whose work is ~6 ms). Unbounded submission does not fix
  it (50.2 s) and neither does a bigger window — if the progressive fill is ever
  worth optimising further, the lever is fewer, larger display chunks, not more
  in-flight ones.


## Rigid drift solve — phase correlation backends + accuracy vs upsample (2026-07-29)

`spyde/drift/translation.py` step A1. Synthetic stack built by Fourier phase ramp
so sub-pixel ground truth is exact (no interpolation in the truth).

**Accuracy** — 5 frames, 96×112, truth shifts deliberately OFF the `1/upsample`
grid (`1.37, -2.83, …`). This matters: shifts that happen to be multiples of
`1/upsample` come back at 0.00000 px, which looks superb and tests nothing.

| upsample | max error |
|---|---|
| 1 | 0.440 px |
| 2 | 0.220 px |
| 8 | **0.065 px** ← inside the 0.1 px acceptance gate |
| 32 | 0.015 px |
| 64 | 0.014 px (floor) |

Halves as expected until ~u=32, where it hits the scene's own noise floor. `u=8`
is the default: comfortably inside the gate at a fraction of the refinement cost.

**Throughput** — 120 frames × 512², upsample=8, warm (cold CUDA run discarded).

| backend | time | frames/s | |
|---|---|---|---|
| numpy | 6.57 s | 18 | reference path |
| torch **cpu** | 0.86 s | **139** | **7.7× numpy** |
| torch cuda | 0.42 s | 284 | 16× numpy |

**Do NOT default to numpy on a CPU-only machine.** `_resolve_ops` originally
preferred numpy when no GPU was present, reasoning that torch's per-call dispatch
would dominate at one frame at a time. Wrong by 7.7×: `np.fft.fft2` is
single-threaded and `torch.fft.fft2` uses every core, and a per-frame FFT is the
entire cost of this solver. Order is now cuda > mps > **torch cpu** > numpy, with
numpy kept only as the explicitly-selectable parity reference.

CUDA's 2× over torch-CPU is smaller than the batched-compute wins elsewhere in
SpyDE because this solver is deliberately *streaming* — one frame at a time, so
each frame pays a host→device transfer that a batched formulation would amortise.
That is the accepted trade for the Memory-Safety rule (a 3000 × 4096² movie is
tens of GB and cannot be batched wholesale). If the transfer ever dominates, the
fix is a bounded read-ahead of a few frames, not materialising the stack.

### Apodisation and reference robustness -- two traps the fixture found (2026-07-29)

Both found by `spyde.data.synthetic.particle_movie` on its FIRST run, which is the
argument for building a ground-truth fixture before the thing it grades.

**A full Hann window destroys the registration.** Synthetic movie, true drift at
frame 23 = `(6.0, 2.9)` px:

| taper alpha | max error over 24 frames |
|---|---|
| 0.00 (none) | 0.125 px |
| 0.10 | 0.125 px |
| **0.25 (default)** | **0.124 px** |
| 0.50 | 0.227 px |
| 1.00 (full Hann) | **25.25 px** |

At alpha=1 the strongest correlation peak sits at `(-19, 19)` scoring 0.121 while
the TRUE peak scores 0.088. **skimage's `phase_cross_correlation` returns the same
wrong answer on the same windowed input**, so this is a property of full-frame
windowing, not of either implementation: once the drift is large the window
reweights different content in each frame. `max_shift=20` does not save you -- the
false peak is inside the band. Taper the EDGE only.

**The running reference needed explicit outlier rejection.** Peak strength relative
to the running median:

| case | ratio |
|---|---|
| worst NATURAL frame (clean sub-pixel stack) | 0.388 |
| a frame replaced by pure noise | 0.007 |

A ~50x gap, so the threshold is 0.25. Without rejection, one noise frame in a
5-frame stack dragged the two frames AFTER it 3.9 px off. With it, those two are
recovered exactly and only the bad frame is wrong. Two settings that did not
survive measurement: a 3-sample warm-up (too slow -- the bad frame is already in
the reference on a short stack) and windowing the median over the last N accepted
(no difference at N=3, 5 or unbounded).

### Fixture + classical segmentation cost (2026-07-29)

| stage | time | note |
|---|---|---|
| build fixture (24 x 96x112) | 98 ms | eager, deterministic |
| drift solve, 24 frames | 32 ms | 0.124 px max error |
| `segment_frame`, one 96x112 frame | 3.7 ms | 272 frames/s |
| `measure_frame`, one 96x112 frame | 13.7 ms | 73 frames/s |
| combined | 17.4 ms | 58 frames/s |

**`measure_frame` is 3.7x the cost of `segment_frame`, which is the wrong way
round and is the thing to watch.** Segmentation is vectorised over pixels;
measurement still loops over PARTICLES to crop each bbox for the intensity ring and
to trace each contour. At 3000 frames of this size that is 52 s -- fine -- but the
plan's target frames are 2048-4096 square, and the loop cost grows with particle
count as well as pixels. If the combined figure misses the "minutes" target at real
scale, `measure_frame` is where to look first, not the segmenter.

Regenerate all of the above with `python scripts/bench_drift_particles.py`.
Verify the whole feature with `python scripts/verify_drift_particles.py --all`.

### Classical segmentation at 4096² — where the time goes (2026-07-30)

Reported as "just too slow for a 4k x 4k image". Stage profile of one frame:

| stage | time | share |
|---|---|---|
| `gaussian_filter(sigma=1)` | 0.46 s | 7% |
| `threshold_otsu` | 0.23 s | 4% |
| **`distance_transform_edt`** | **3.93 s** | **61%** |
| gaussian on the distance | 0.43 s | 7% |
| `peak_local_max` | 0.58 s | 9% |
| `watershed` | 0.60 s | 9% |
| **total** | **6.40 s** | |

**The distance transform is the cost, not watershed** — which is the opposite of
the intuition, and it is why "make watershed faster" would have been wasted work.

The EDT is used for exactly two things: seeding markers and giving watershed an
elevation. Neither needs full resolution, so above ~2 MP both are computed on a
decimated grid and the elevation is bilinearly upsampled (rescaled by the factor
to stay in pixel units). **Detection is untouched** — the threshold still runs at
full resolution, so *which* bodies are found is unchanged and §0.9's faint-particle
sensitivity is unaffected. Only the cut BETWEEN two touching bodies moves, by about
`factor` px.

| frame | full-res split | auto-decimated | speedup | count | median area |
|---|---|---|---|---|---|
| 1024² touching | 0.34 s | 0.30 s | 1.1× | 162 = 162 | 298 = 298 |
| 4096² touching | 7.09 s | 2.82 s | **2.5×** | 162 = 162 | 4762 = 4762 |
| 4096² isolated | 8.14 s | 2.80 s | **2.9×** | 81 = 81 | — |

Identical counts and identical median areas at 0.0% difference, on both touching
and isolated fields — the decimation is free in accuracy terms on this data.

**Turning "Split touching" OFF is a further 1.9×** (2.80 s → 1.45 s) and is the
right choice whenever particles are isolated, because watershed then has nothing
to do and the whole EDT is skipped. Decimating past 2 buys almost nothing
(2.80 → 2.73 s) since the EDT is no longer dominant once it is decimated at all.

Remaining levers, unmeasured, in the order they look worth trying:

1. **Tile + thread.** scipy's EDT and watershed are single-threaded and both
   release the GIL, so banding the frame with a halo of the largest particle
   radius should scale with cores — the `region_sum.py` precedent got 6.6× from
   exactly this shape. The halo makes the EDT correct at tile edges; watershed and
   the threshold tile cleanly.
2. **GPU EDT** (`cupy.ndimage.distance_transform_edt`). Only worth it if the frame
   is already on the device; a 64 MB round trip per frame is most of the win.
3. **Skip the EDT where nothing touches.** Connected components whose area matches
   a single-body prior do not need splitting at all; only ambiguous ones do.

#### The profile moved after decimation shipped — re-measure before optimising

Re-profiled 2026-07-30 on the same 4096² touching field (162 bodies, median area
4762 px) with the decimation in place. The EDT is **no longer the cost** — it is
4% of the frame. Optimising it further would have been wasted work, exactly as
optimising the watershed would have been before:

| stage | 4096² |
|---|---|
| `gaussian_filter(sigma=1)` | 0.42 s |
| `threshold_otsu` | 0.24 s |
| label + min_size pre-filter | 0.19 s |
| `distance_transform_edt` (decimated ×4) | **0.11 s** |
| marker smooth + `peak_local_max` | 0.05 s |
| upsample markers + elevation | 0.70 s |
| `watershed` | 0.67 s |
| size filter + sequential relabel | 0.33 s |
| **total** | **2.69 s** |

Two things that are *not* levers, measured rather than assumed:

* **`ndi.distance_transform_edt` and `skimage.watershed` do not thread well.**
  Per-connected-component processing (exact, since a masked watershed cannot flood
  between components — no halo or union-find needed) got 4096² from 1.53 s to
  0.39 s serial, but only to **0.17 s at 4 threads and got *worse* past that**
  (0.22 s at 16). `gaussian_filter` and `ndi.label` *do* release the GIL — banded
  they give 452→71 ms (bit-identical) and 92→21 ms — but the EDT/watershed pair
  saturates at ~2.2×. Row-band threading is not the `region_sum.py` story here.
* **`np.isin`/`np.unique` on the label raster cost more than the algorithm.**
  `_relabel_sequential`'s `np.unique` alone is a 139 ms full sort of 16.7 M
  elements. Fusing the size filter and the relabel into one `bincount` + one LUT
  gather (`_finalize_labels`) is 302 → 164 ms and bit-identical.

### The boundary class: making the split unnecessary (2026-07-30)

`split_instances` is shared by all three engines, so no engine-level work changes
what a big frame costs while every engine funnels into a 2 s watershed. The way
out is not to make the split faster but to make it **unnecessary**: the ilastik
convention paints particle / background / **boundary**, and a head that has been
shown the joins returns touching particles already separated. Instances are then
plain connected components and neither the EDT nor the watershed runs.

Measured at 4096², CUDA (TITAN X Pascal), 162-body touching field, scribble engine
trained on a 1024² crop:

| stage | watershed route | boundary route |
|---|---|---|
| predict (featurise + head + readback) | 0.96 s | 0.96 s |
| threshold | — | 0.02 s |
| `ndi.label(fg & ~boundary)` | — | 0.07 s |
| reclaim the seam | — | 0.08 s |
| EDT + markers + watershed | 1.62 s | **0 s** |
| size filter + relabel | 0.16 s | 0.16 s |
| **split subtotal** | **1.78 s** | **0.33 s** (5.4×) |
| **end to end** | **2.73 s** | **1.29 s** |

**Accuracy is better, not merely comparable**: the boundary route found n=162 —
the exact ground truth — where the watershed found 173 (11 spurious), at the same
median area (5670 vs 5668, +0.0%). On the `particle_movie()` fixture both routes
give the same count and the same areas, both faint §0.9 probes are still found,
and the deliberately-touching pair is split at the merge frame.

**The one thing that must be got right is what "boundary" is trained on.** It is
the seam BETWEEN two bodies, never the outline of one. A head taught outlines
learns "shrink everything": measured on the fixture's merge frame it MERGED the
touching pair and lost 40% of the median area. Training on 30 px of seam is
likewise useless — it took the fast route and returned 81 bodies where the
watershed found 162. The caret's per-class pixel counts are what surface this,
and `seg_train` now says which route the training selected.

#### Feature stack — the remaining floor

The split is no longer the cost; **featurising is**, at 0.68 s of the 1.29 s.
Two bit-identical fixes, measured at 4096² on CUDA:

| | old banding (143 rows, 29 bands) | device banding (574 rows, 8 bands) |
|---|---|---|
| median reduced along the strided axis | 1.27 s | 0.85 s |
| median reduced along a contiguous axis | 1.04 s | **0.68 s** |

* **Band size.** Every band re-featurises `halo` rows above and below, so a
  256 MB band at 4096² spends 36% of its work on halo. A device-sized band
  (`GPU_BAND_BYTES`, clamped to ¼ of *free* VRAM) cuts that to 11%. It is a
  ceiling and not a target because overshooting is catastrophic rather than
  merely slower: the same sweep at 1536 rows/band took **12.8 s** and at one band
  **26.7 s**, thrashing the allocator on a 12.9 GB card.
* **Median layout.** `F.unfold` returns `(1, k², h·w)`, so `median(dim=1)` reduces
  along the *strided* axis. Transposing first makes each window's taps adjacent:
  52.2 → 19.6 ms + 3.7 ms for the copy, on a 768×4096 band at r=2. Bit-identical
  (an odd-window median is a selection). Not worth it at r=1 (19.0 vs 20.4 ms),
  so it is gated on window size.

Remaining, unmeasured, in the order they look worth trying:

1. **A sorting network for the r=1 median** — measured 20.4 → **5.2 ms** per band,
   bit-identical, but 24 hand-written compare-exchanges. Worth ~80 ms/frame.
2. **The convolutions are ~30× off memory-bandwidth peak.** The 5-sigma gaussian
   pyramid moves ~134 MB per separable pass and should be ~4 ms on this card;
   it measures 118 ms. `F.pad` allocates a fresh padded copy per convolution and
   cuDNN is being handed 1-channel images. Batching the sigmas into the channel
   dimension is the obvious shape.
3. **Threading `_finalize_labels`** (158 ms: 93 ms `bincount` + 66 ms gather).
   Both are memory-bound and band cleanly; banded `bincount` measured 116 → 51 ms.

#### Two independent reproductions of the outline trap (2026-07-30)

The boundary route's hazard is not theoretical and it is not rare — it is what a
first attempt produces. Both of these were meant to be routine verification and
both hit it instead:

* **In the app** (`segment_wizard.spec.ts`, bundled 6-frame movie): one straight
  seam stroke, 135 px, took the preview from **9 particles to 2**. The caret
  reported `Trained on 1049 px, 3 classes · acc 1.000 · cuda · seam split` — a
  perfect training accuracy and the fast route, on a result 78% worse.
* **At 4096²** (648 touching discs, seam synthesised as `grey_dilation(lab) != lab`,
  i.e. a RING around each body rather than the join between two): the seam route
  returned **324 bodies — exactly half the ground truth**, every pair merged, at
  2.6× the correct median area. 1.31 M px (7.8% of the frame) came back as
  boundary. Speed was as advertised: split 2.66 s → 0.375 s (7.1×), end to end
  3.78 s → 1.49 s (2.5×).

The second one is the informative one: the ring is the *intuitive* reading of the
word "boundary", it is what an automated seam-builder writes on the first try, and
it produces a confidently wrong answer with a clean training accuracy. Speed and
correctness are independent here — the route was fast in both reproductions and
right in neither.

What is on screen today: the caret's persistent line says `seam split` vs
`watershed split`, and the strip's hover text says to paint the seam and not the
outline. Neither is a guard. **There is currently nothing that stops a wrongly
trained boundary from silently replacing a good answer with a bad one.**

### The batch run at real scale: 900 x 4096² (2026-07-30)

Reported as "scribble segmentation over 900 frames of 4096x4096 is far too slow,
and the GPU is hardly used, as are the CPUs". Measured on a REAL in-situ growth
movie (`20251117_88075_run3…mrc`, 977 x 4096² uint8, 15.3 GB) — 48 cores, one
TITAN X Pascal, 9 workers x 4 threads, which is what `_compute_worker_plan`
builds here.

#### Where one frame goes, and it is not where the previous sections say

| stage | classical | scribble |
|---|---|---|
| read one frame (lazy MRC, memmap) | 0.12 s | 0.12 s |
| segment / predict | 3.0 s | 1.5 s |
| split | (in segment) | 2.0 s |
| **measure** | **53.5 s** | **2.4 s** |
| **total** | **56.6 s** | **6.0 s** |
| particles found | 26 566 | 1 139 |

**`measure_frame` is the run.** The 2026-07-29 fixture note predicted this
("`measure_frame` is 3.7x the cost of `segment_frame`, which is the wrong way
round and is the thing to watch… if the combined figure misses the *minutes*
target at real scale, `measure_frame` is where to look first") and it is worse
than predicted, because the cost is per PARTICLE and a real frame has tens of
thousands. Inside it, at 26 566 particles:

| `regionprops_table` property | time |
|---|---|
| **solidity** | **29.4 s** |
| eccentricity | 11.3 s |
| major_axis_length | 11.5 s |
| minor_axis_length | 11.3 s |
| perimeter | 4.1 s |
| centroid | 2.1 s |
| equivalent_diameter_area | 1.2 s |
| area | 1.2 s |
| bbox | 1.1 s |
| **all together (shared intermediates)** | **43.8 s** |
| `_fill_intensity` | 4.9 s |
| `_contours` | 4.8 s |

`solidity` is a convex hull per region; the three axis/eccentricity properties
share one inertia tensor. Every scipy.ndimage equivalent of the cheap ones is
1-2 orders faster on the same raster (`bincount` area 0.13 s vs 1.2 s;
`find_objects` bbox 0.065 s vs 1.1 s; `center_of_mass` 1.0 s vs 2.1 s), so the
whole table is vectorisable in principle — but that changes what a particle's
measured properties ARE, so it is a proposal, not a drive-by.

#### It is GIL-bound, so PROCESSES are the unit of parallelism, not threads

`regionprops_table` releases the GIL essentially never. Four 2048² quadrants of
the same frame, in four threads of one process:

| threads | wall | speedup |
|---|---|---|
| 1 quadrant, serial | 10.4 s | — |
| 2 quadrants, 2 threads | 21.2 s | **0.99x** |
| 4 quadrants, 4 threads | 45.0 s | **0.93x** |
| 8 quadrants, 8 threads | 168.8 s | **0.49x** |

`_contours` (1.23 s -> 6.35 s for 4x the work) and `_fill_intensity`
(1.25 -> 5.74 s) are the same. So banding a frame across threads — the
`region_sum.py` trick — cannot work here, and a dask worker's four task slots
are worth one core, not four. **The effective parallelism of a segmentation
batch is the WORKER COUNT.**

#### The dual-lane fan-out (spyde/particles/batch.py)

| engine | frames | config | frames/s | 900-frame projection |
|---|---|---|---|---|
| scribble | — | serial (the retired loop) | 0.166 | 1h30m |
| scribble | 24 | dual lane, 4 GPU feeders, torch unpinned | 0.159 | 1h34m |
| scribble | 60 | dual lane, 1 GPU feeder, torch 1 thread | 0.222 | 1h07m |
| scribble | 60 | **GPU-only, 1 feeder** | **0.270** | **55m** |
| scribble | 60 | GPU-only, 2 feeders | 0.252 | 59m |
| scribble | 60 | GPU-only, 4 feeders | 0.260 | 58m |
| classical | — | serial | 0.0177 | 14h09m |
| classical | 36 | fan-out, 9 workers x 4 threads | 0.052 | 4h48m |

Classical gets **2.9x**, scribble **1.6x**. Both are far short of the 9-ish the
worker count allows, and the reason is the same in both: the frame's dominant
stage holds the GIL, so four task slots per worker do not add throughput, and
what is left competes for memory bandwidth.

#### Three things that are NOT true, measured

* **"Four GPU feeders keep the device fed" (the neural default) does not
  transfer.** The first measurement said 4 feeders made a frame 13x slower
  (110 s vs 8 s of predict+split) — but that run also had five CPU-lane workers
  running 48-thread torch predicts, so it was contention, not the lane count.
  Re-measured cleanly with an empty CPU lane, 1/2/4 feeders are 0.270 / 0.252 /
  0.260 frames/s: **flat**. More feeders neither help nor (much) hurt, because
  the device is not the constraint — a single worker process is, and it is
  GIL-bound. The segmentation lane default is `"one"` on that basis.
* **The CPU lane is worse than nothing for scribble**, which is the opposite of
  what the isolated numbers suggest. One CPU predict at 4096² costs 65.8
  core-seconds at 1 torch thread (35.1 s x 2 threads = 70.2, 18.8 x 4 = 75.2,
  11.2 x 8 = 89.8, 7.0 x 16 = 111.6, 9.0 x 48 = 430.5 — intra-op threading
  costs MORE work the wider it goes, so every worker pins
  `torch.set_num_threads(1)` and lets frame-level parallelism do the work).
  Against 1.6 s on the GPU that is 41x in core-seconds but only ~2x per frame,
  which looks like most of a doubling. In the cluster the CPU lane contributed
  0.16 frames/s and cost the GPU lane 0.5 (its frames went 5.8 s -> 25-29 s),
  so the run went 0.270 -> 0.222. GPU-only, as the neural batch already does.
* **Batching the gaussian sigmas into the channel dimension is SLOWER.** The
  "Feature stack — the remaining floor" note above proposed it as "the obvious
  shape" for the convolutions that sit ~30x off memory-bandwidth peak. Measured
  on a 574x4096 band and on a full 4096², one conv2d per axis with all five
  sigmas as output channels (zero-padded to the widest radius, `groups=5` on the
  second axis): **18.9 -> 22.0 ms** and **100.0 -> 135.1 ms**, i.e. **0.86x and
  0.74x**. It is bit-identical (`torch.equal` True — a zero tap contributes
  exactly 0.0), so the idea is sound and only the economics are wrong: padding
  every kernel to radius 32 turns 129 taps of work into 325. The 30x-off-peak
  observation stands; batching is not the way to collect it. NB the pyramid is
  ~0.15 s of a 6.0 s scribble frame, so the whole remaining prize there is 2.5%.

#### The trap this benchmark hit first, which the app does not

The first cluster run spent 90 s in stall pokes on a
`rechunk-merge-rechunk-transfer` before segmenting a single frame. RosettaSciIO
auto-chunks this movie as a balanced cube — `(511, 511, 511)` — which SPLITS the
signal axes, so one frame spans 64 blocks and 8.5 GB, and asking for whole
frames one at a time is a full P2P shuffle of the movie (CLAUDE.md
Live-Display §1). The app never sees this: `Session._signal_spanning_chunks`
re-loads every movie with `chunks=(1, -1, -1)`, free, at load. `batch._dispatch`
now REFUSES to rechunk the signal axes and falls back to the streaming accessor
with a warning naming the fix, and the benchmark loads the way the app loads.

### Vectorising `measure_frame`: 53.5 s -> 10.2 s, every column bit-identical (2026-07-30)

The section above says `measure_frame` **is** the run and that fixing it "changes
what a particle's properties ARE". It turns out it does not have to. Measured on
the same real frame (`20251117_88075_run3…mrc` frame 10, 4096², **26 566**
particles) with `spyde/tests/migrated/test_particles_props_parity.py` as the gate.

#### Almost every column is a label-wise reduction, and one is not

| `regionprops_table` column, alone | s | replaced by |
|---|---|---|
| solidity | **30.5** | `hull.convex_areas` — numba, exact integer hull |
| major_axis_length | 11.9 | second central moments (`bincount`) |
| minor_axis_length | 12.0 | " |
| eccentricity | 11.8 | " |
| perimeter | 4.1 | border-crossing weights, whole frame at once |
| centroid | 2.3 | `bincount` |
| equivalent_diameter_area | 1.5 | a function of `area` |
| area | 1.4 | `bincount` |
| bbox | 1.1 | `ndi.find_objects` |
| **whole table (shared intermediates)** | **43.7** | **1.09 s (40x)** |

`regionprops_table`'s cost is per REGION — a Python object, and for `solidity` two
Qhull calls and a polygon rasterisation, ~1.1 ms each, 26 566 times. The
arithmetic is nothing; the per-region overhead is everything.

#### The parity is not "close", it is the same numbers

Per column against `regionprops_table` on the real 26 566-region raster:

| column | agreement |
|---|---|
| label, area, bbox-* | **exact** (integers) |
| centroid-0/1 | **exact** — both sum exact integers in float64 |
| equivalent_diameter_area | **exact** (a function of `area`) |
| **solidity** | **exact** — `area_convex` matches on **26 566 / 26 566** regions, zero differing pixels |
| perimeter | 3.5e-16 relative |
| major/minor_axis_length | 1.0e-15 / 4.9e-15 relative |
| eccentricity | 1.6e-14 relative |

The float differences are summation ORDER and nothing else, and at the float32
resolution the property rows are stored in, **all 21 output columns and every
contour come out bit-identical** between the two paths.

Three things made that possible rather than lucky:

* **Central moments in skimage's own frame, two-pass.** `RegionProperties` takes
  `moments_central(image, centroid_local, …)` — about the LOCAL centroid, in the
  bbox crop's coordinates. Doing the algebraically-equal raw-to-central expansion
  in GLOBAL coordinates instead cancels, and that is where a first attempt lost
  eccentricity to 1.4e-5. Subtracting the bbox origin per pixel costs one gather.
* **The same 2x2 eigenproblem.** `np.linalg.eigvalsh` on a stacked `(N, 2, 2)` is
  the same LAPACK call skimage makes one at a time, so `4*sqrt(l1)` agrees to
  1e-15 instead of to a closed-form solver's 1e-9.
* **The hull in exact integers.** skimage's `convex_hull_image` replaces each
  pixel with the four diamond offsets `(r±0.5, c)`, `(r, c±0.5)`. **Double every
  coordinate and those are integers** — so the monotone chain's cross products and
  the inside-or-on test are `int64` comparisons with no tolerance to tune and no
  tie to lose. Reducing to the first/last pixel of each row first is not an
  approximation (a pixel between them is in their hull, so it is never a vertex).

`SPYDE_PARTICLE_PROPS=legacy` restores `regionprops_table`; it is what the parity
test compares against and what runs if numba cannot compile.

#### Where the frame now goes — the remaining floor moved, it did not vanish

| stage | before | after |
|---|---|---|
| property table | 43.7 s | **1.09 s** |
| `_fill_intensity` | 4.7 s | 4.7 s (untouched) |
| `_contours` | 4.7 s | 4.7 s (untouched) |
| **`measure_frame`** | **53.2 s** | **10.2 s (5.2x)** |

**`_fill_intensity` + `_contours` are now 92% of the measurement**, and both are
exactly what the property table used to be: a Python `for` loop over regions, one
`binary_dilation` and one `find_contours` per particle. The intensity statistics
are the same shape of `bincount` the moments turned out to be; the local
background RING (a dilation per particle, which overlapping neighbours can each
claim) and marching-squares contours are not, and are the reason they were left.

#### The GIL half of the prize did NOT land, and the reason is measurable

The point of removing `regionprops_table` was twofold — the per-frame cost, and
the fact that it never releases the GIL, so a worker's four task slots are worth
one core. Re-running the same threaded-quadrants experiment (four 2048² quadrants,
four threads, one process):

| what | 1 quadrant | 4 quadrants / 4 threads | scaling |
|---|---|---|---|
| property table, `regionprops_table` | 10.68 s | 33.79 s | 1.26x |
| property table, **vectorised** | 0.26 s | 0.43 s | **2.48x** |
| `measure_frame`, legacy | 12.78 s | 42.64 s | 1.20x |
| `measure_frame`, **vectorised** | 2.47 s | 10.53 s | **0.94x** |

The table itself now scales — 1.26x to 2.48x, and its own numba kernel is already
using every core inside one call, so 2.48x on top of that is the honest ceiling for
four threads. But **`measure_frame` as a whole still does not**, because the 9.4 s
that is left is the two Python loops, and they hold the GIL exactly as
`regionprops_table` did. So "the effective parallelism of a segmentation batch is
the WORKER COUNT" is still true, and it will stay true until `_fill_intensity` and
`_contours` go the same way.

#### End to end, same cluster, same movie, same frame count

`benchmark_particles_batch --frames 36 --engine both`, 9 workers x 4 threads, one
TITAN X Pascal — the identical configuration the 4h48m / 55m numbers above were
taken on.

| | one frame | throughput | **900 frames** |
|---|---|---|---|
| classical, before | 56.6 s | 0.052 frames/s | 4h48m |
| classical, **after** | **14.3 s** (3.1 segment + 11.2 measure) | **0.319 frames/s** | **47m** |
| scribble, before | 6.0 s | 0.270 frames/s | 55m |
| scribble, **after** | **4.7 s** (1.4 predict + 2.0 split + 1.3 measure) | **0.419 frames/s** | **36m** |

**Classical is 6.1x, and only 4.0x of that is the frame getting cheaper** — the
rest is the fan-out working better than it did. Serial classical is now
0.070 frames/s, so the cluster multiplies it by **4.6x** where it managed 2.9x
before: the property table releases the GIL (numba `nogil`, numpy ufuncs), so a
worker's four task slots are finally worth more than one core. Scribble gains
1.55x, which is all it can — it segments 1 139 particles per frame, not 26 566, so
measurement was never its bottleneck (2.4 s -> 1.3 s of a 4.7 s frame).

In-cluster per-frame stages (`drain_stage_log`), which say where the rest went:

| lane | frames | engine/f | measure/f | block/f |
|---|---|---|---|---|
| classical, cpu x9 | 36 | 7.32 s | 54.31 s | 61.63 s |
| scribble, cuda x1 | 36 | 7.47 s | 1.56 s | 9.03 s |

A classical frame measures in 11.2 s alone and **54.3 s** with 36 of them in
flight — a 4.9x contention factor, against 2.4x for the segmentation. That is the
signature of the remaining GIL-bound Python loops plus memory bandwidth, and it is
the next thing worth attacking: `_fill_intensity` and `_contours`.

**Minutes is still not reached.** 47m is 6x better and not 60x, and the arithmetic
says why: 900 frames in 5 minutes across 9 workers is ~3 s of wall per frame, and
one classical frame is 14.3 s of work of which 9.4 s is two Python `for` loops
over 26 566 regions. Vectorising those the way the property table was vectorised
is worth roughly another 2.5x on the frame **and** should lift the fan-out again,
which together is the difference between 47 minutes and ~10.

### The last two loops: `measure_frame` 10.2 s -> 1.37 s, and the fan-out unblocked (2026-07-30)

The section above ends by naming exactly what was left — "one classical frame is
14.3 s of work of which 9.4 s is two Python `for` loops over 26 566 regions" — and
predicting the prize: "roughly another 2.5x on the frame **and** should lift the
fan-out again, which together is the difference between 47 minutes and ~10." Both
halves landed. Same real frame (`20251117_88075_run3…mrc` frame 10, 4096²,
**26 566** particles), same box, same cluster.

#### `_fill_intensity`: three `bincount`s and one kernel

`intensity_mean` / `intensity_max` / `intensity_std` are label-wise reductions over
the foreground and go the way the moments went — `bincount` with weights for the
sums, one label-grouped `np.maximum.reduceat` for the max (`np.maximum.at` is an
unbuffered per-pixel ufunc call and is ~50x slower than the radix sort it avoids).
`intensity_std` is computed the way `np.std` computes it, mean first and then the
mean of squared deviations, NOT as `E[x²] - E[x]²` — algebraically equal, and it
cancels away the digits that matter exactly where a particle is bright and
uniform, which is the normal case.

`background` is the half that is not a reduction: it is the mean over the pixels a
**dilation of THIS particle by `ring`** adds and that belong to no particle. That
is a per-particle neighbourhood which overlapping neighbours may each claim, so it
is not a partition of the raster and no `bincount` expresses it. It keeps the
definition exactly — an iterated 4-connected dilation inside the same padded bbox
crop, which is what `binary_dilation`'s default structure and `border_value=0`
do — in a numba `prange` kernel with the GIL released, the way `hull.py` does for
the convex hull.

| | before | after |
|---|---|---|
| `_fill_intensity` at 26 566 regions | **4.92 s** | **0.187 s (26x)** |

**All four columns are bit-identical on the real frame** — `max |diff| = 0.0` on
all 26 566 rows for `intensity_mean`, `intensity_max`, `intensity_std` and
`background`, with the same NaN pattern, and the same again with a 64-row
NaN-padded border (the drift-corrected case) blanked into the frame. The pixel
SETS are identical by construction; only summation order differs, and at the
float32 resolution the rows are stored in that is nine orders below the last bit.

#### `_contours`: marching squares is a case table, and assembly is a walk

`find_contours` is Cython, but `_assemble_contours` — the part that joins its
segments into contours — is a pure-Python dict-and-deque walk, and the whole call
is ~177 us per region. Two observations collapse it:

* **On a BINARY mask at level 0.5, every vertex is an edge midpoint.**
  `_get_fraction` is `(0.5 - 0) / (1 - 0)` for every edge the case table actually
  uses, so a vertex is at `(i + 0.5, j)` or `(i, j + 0.5)` — it IS the crack
  between two 4-adjacent pixels, and can be named by an integer index with no
  floating point anywhere.
* **The segments form disjoint paths and cycles, and nothing else.** Each cell
  emits its segments oriented low-on-the-left, and a crack interior to the crop is
  shared by exactly two cells, appearing once as a tail and once as a head. So
  in-degree and out-degree are both <= 1, `_assemble_contours` recovers exactly the
  maximal chains, and following a `succ` array recovers the same ones in the same
  direction.

| | before | after |
|---|---|---|
| `_contours` at 26 566 regions | **4.77 s** | **0.149 s (32x)** |

##### The parity gate here is NOT vertex identity — and it is not "close enough" either

A closed contour is a CYCLE. skimage's assembly and a `succ`-following walk cut it
at different vertices, so the arrays differ **by a rotation while describing the
same shape**. Demanding bit-identical vertices rejects a correct implementation,
and that is where a first attempt stops and declares the loop untouchable.

The opposite conclusion is the more dangerous one, and it is also wrong: outlines
are not a display choice. `SpyDEParticles.render_frame` FILLS them to rebuild the
label movie, and `mask_at` fills one to produce the per-particle mask a mean
diffraction pattern is sliced with. A different contour is a different mask is a
different measurement. So the gate is the thing those two consume, and nothing
weaker:

> **`skimage.draw.polygon` on the new outline must select EXACTLY the same pixels
> as on the old one, for every region.** A boolean set equality — not a tolerance,
> not an IoU.

On the real frame: **26 566 / 26 566 regions fill to an identical pixel set**, and
the vertex COUNT matches on 26 566 / 26 566 as well. On 14 581 regions of random
thresholded noise the same holds, and additionally every closed contour is a
literal rotation of skimage's while every open one is bit-identical.

Two details that look like trivia and decide the answer:

* **`np.rint` is round-half-to-EVEN, and it is applied in CROP coordinates.** Every
  vertex here is a half-integer, so the rounding is entirely in the tie case and
  resolves on the PARITY of the crop-local coordinate — which depends on where the
  padded bbox happens to start. Two congruent particles at different positions
  therefore get genuinely different integer outlines. That is the behaviour on disk
  today; reproducing it means rounding in the crop frame and offsetting afterwards,
  never the reverse.
* **Which contour is "the" contour.** The caller takes `max(cs, key=len)`, and `cs`
  is ordered by `_assemble_contours`'s creation counter, which after every merge
  keeps the smaller of the two keys — so it equals the order of each contour's
  SMALLEST segment index. Ties in length break to the chain containing the earliest
  cell in raster order, and that is reproduced explicitly rather than left to
  whatever order a walk happens to discover.

#### Where the frame goes now

| stage | original | after props+hull | after this |
|---|---|---|---|
| property table | 43.7 s | 1.09 s | 1.08 s |
| `_fill_intensity` | 4.9 s | 4.9 s | **0.19 s** |
| `_contours` | 4.8 s | 4.8 s | **0.15 s** |
| **`measure_frame`** | **53.2 s** | **10.2 s** | **1.37 s (38x)** |

#### The GIL half of the prize, which is why this was worth doing at all

The previous pass got the property table to 2.48x in four threads but left
`measure_frame` as a whole at **0.94x**, "because the 9.4 s that is left is the two
Python loops, and they hold the GIL exactly as `regionprops_table` did". Same
experiment — four 2048² quadrants of the same frame, in one process:

| what | 1 quadrant | 4 quadrants / 4 threads | scaling |
|---|---|---|---|
| property table, `regionprops_table` | 10.86 s | 34.08 s | 1.27x |
| property table, vectorised | 0.27 s | 0.41 s | 2.63x |
| `measure_frame`, legacy | 12.75 s | 41.43 s | 1.23x |
| `measure_frame`, **vectorised** | **0.40 s** | **0.56 s** | **2.82x** |

`measure_frame` now scales BETTER than the property table alone, because all three
of its stages are numba `nogil` kernels or numpy ufuncs and the Python that remains
is per-FRAME rather than per-region. "The effective parallelism of a segmentation
batch is the WORKER COUNT" — asserted twice in the sections above — is no longer
true: a worker's four task slots are finally worth more than one core.

#### End to end, same cluster, same movie, same frame count

`benchmark_particles_batch --frames 36 --engine both`, 9 workers x 4 threads, one
TITAN X Pascal — the identical configuration every row above was taken on. One run
per engine.

| | one frame | throughput | **900 frames** |
|---|---|---|---|
| classical, original | 56.6 s | 0.052 frames/s | 4h48m |
| classical, after props+hull | 14.3 s | 0.319 frames/s | 47m |
| classical, **after this** | **5.1 s** (3.2 segment + 1.9 measure) | **1.152 frames/s** | **13m01s** |
| scribble, original | 6.0 s | 0.270 frames/s | 55m |
| scribble, after props+hull | 4.7 s | 0.419 frames/s | 36m |
| scribble, **after this** | **4.3 s** (1.4 predict + 1.9 split + 1.0 measure) | **0.499 frames/s** | **30m04s** |

**Classical is 3.6x on top of the previous pass and 22x on the original**, and only
2.8x of this pass is the frame getting cheaper — the rest is the fan-out finally
working. Scribble gains 1.19x, which is all it can: it segments 1 139 particles per
frame, not 26 566, so measurement was never its bottleneck (1.56 s -> 1.06 s
in-cluster).

In-cluster per-frame stages (`drain_stage_log`), which say where the rest went:

| lane | frames | engine/f | measure/f | block/f |
|---|---|---|---|---|
| classical, cpu x9, before | 36 | 7.32 s | **54.31 s** | 61.63 s |
| classical, cpu x9, **after** | 36 | 7.53 s | **3.97 s** | 11.51 s |
| scribble, cuda x1, **after** | 36 | 6.53 s | 1.06 s | 7.59 s |

A classical frame used to measure in 11.2 s alone and 54.3 s with 36 in flight — a
**4.9x** contention factor that was the signature of the GIL-bound loops. It now
measures in 1.4 s alone and 4.0 s in flight: **2.9x**, and what is left there is
memory bandwidth (nine workers each streaming a 16.7 MP frame), not a lock.

**Minutes is reached** — 13m01s, against the ~10m the previous section projected
from 47m. And the bottleneck has MOVED: `segment_frame` is now 7.5 s of the 11.5 s
in-cluster block and `measure_frame` is 1.4 s of a 5.1 s solo frame, so the next
thing worth attacking on the classical path is the segmentation, not the
measurement. Note also that the classical run finds **1 283 491 particles across 36
frames** and the scribble run 44 846 — the two engines are not measuring the same
scene, and their per-frame numbers are not comparable to each other, only to their
own previous rows.

### Navigator fill: the per-chunk submit loop vs the shared dispatcher (2026-07-30)

Real file: `20251117_88075_run3 some growth_1236_movie.mrc` — 977 x 4096² uint8,
15.27 GB, 1-D nav, loaded via `load_aligned` (1 frame/chunk => **977 nav chunks**).
Real `LocalCluster`, 4 worker processes x 2 threads. `--purge` evicts the file
from the Windows page cache (FILE_FLAG_NO_BUFFERING) before each run.
Harness: `spyde/tests/benchmark_nav_fill_dispatch.py`.

The old `compute_with_live_buffer` navigator branch did
`for slices in all_slices: client.compute(chunk)` — one blocking scheduler round
trip per nav chunk, all 977 up front, with the GIL held in the client process the
whole time — and then `client.compute(result_array)` again for the whole array.
It now routes through `compute_dispatch.dispatch_chunks` (batched submit, bounded
in-flight window, stall watchdog) and the result is ASSEMBLED from the chunks.

Cold (page cache purged before each run):

| | submit (client-side, GIL held) | first chunk painted | total fill |
|---|---|---|---|
| per-chunk loop | **9.86 s** | 16.41 s | 50.83 s |
| dispatch_chunks | **0.00 s** | **0.08 s** | 50.64 s |

Warm (one throwaway pass first, so I/O is held constant): submit 10.12 s -> 0.00 s,
total 50.21 s -> 46.53 s. Checksums MATCH in every run: the client-side assembly
is identical to the whole-array compute it replaced.

Three things this pins down:

* **The submit time is the bug.** 9.9 s during which nothing else in the backend
  process can run — the navigator sits blank and the paint threads go silent. It
  is client-side, so it is the same cold or warm. First-visible-pixel goes
  16.4 s -> 0.08 s (**200x**).
* **The bounded window costs nothing.** The window here is 4 (half of 8 cluster
  threads) versus the old path's 977-in-flight, and the total fill did not
  regress — it improved, because the old path also submitted the duplicate
  whole-array graph.
* **A progressive fill is ~7x a monolithic sum, and that is per-TASK overhead,
  not concurrency.** Warm, one `nav.compute()` of the whole graph is **6.4 s**;
  977 separate per-chunk futures are 46-50 s either way (~47 ms of scheduler
  round trip for a task whose work is ~6 ms). Unbounded submission does not fix
  it (50.2 s) and neither does a bigger window — if the progressive fill is ever
  worth optimising further, the lever is fewer, larger display chunks, not more
  in-flight ones.

---

## CNN scribble engine vs the shipped MLP — the prototype does NOT replace it

`python -m spyde.tests.benchmark_scribble_cnn` (CUDA, TITAN X Pascal, 300 steps
unless swept). Both engines train on the SAME `LabelStore` and are scored by one
evaluator, so neither gets a scoring path that could flatter it.

The question was whether one small U-Net over the raw frame could replace the
36 hand-crafted channels + per-pixel MLP. On these numbers: no.

### Train time — the interactive constraint

The caret's tuning loop re-fits on every stroke, so `fit` is the budget that
matters, and the shipped engine sets it at ~0.5-1.6 s.

| | fixture 96x112, 1.4k labelled px | realistic 2048², 34.8k labelled px |
|---|---|---|
| MLP (36ch + head) | **1.64 s** | **1.11 s** |
| CNN tiny b16/L2 | 2.53 s | 4.30 s |
| CNN small b32/L3 | 3.39 s | 7.23 s |

**4-6.5x slower to train**, and it grows with label count while the MLP's
shrinks (the MLP fits a per-pixel head; the CNN pays per crop — 176 of them).

### Quality — worse everywhere, and catastrophically so when labels are sparse

Frame 12 of `particle_movie()`, 9 true particles, against exact ground truth:

| route | engine | IoU | n found / 9 | faint | merge-split |
|---|---|---|---|---|---|
| watershed | MLP | **0.745** | **9** | 2/2 | True |
| watershed | CNN tiny | 0.332 | **78** | 2/2 | True |
| watershed | CNN small | 0.264 | **65** | 2/2 | True |
| boundary | MLP | 0.654 | **9** | 2/2 | False |
| boundary | CNN tiny | 0.647 | 11 | 2/2 | True |
| boundary | CNN small | 0.511 | 21 | 2/2 | True |

78 particles where there are 9 is not a tuning problem, it is a different
answer. On the realistic 2048² field (25x more labels) the gap nearly closes —
MLP 0.809 (n=413/404), CNN small 0.785 (n=434/404), CNN tiny 0.707 (n=388/404).

**That is the finding.** The CNN is LABEL-STARVED on a few strokes, which is
precisely the interactive scribble case it was meant to serve. It becomes
competitive only when given a field's worth of labels — by which point the MLP
is already better AND 6.5x faster to fit.

### Training is non-monotonic in steps — more training makes it worse

Fixture, foreground IoU:

| steps | 50 | 100 | 200 | 300 | 600 |
|---|---|---|---|---|---|
| tiny | 0.597 | 0.515 | 0.639 | **0.641** | 0.518 |
| small | 0.366 | 0.397 | 0.441 | **0.551** | 0.501 |

Both peak at 300 and fall by 600, and tiny's merge-split flips to False there.
So there is no "train it longer" fix available, and no knee to tune to — the
step count would have to be fitted per dataset, which is not something a caret
can ask a user for.

### The one CNN win: inference

| | MLP | CNN tiny | CNN small |
|---|---|---|---|
| fixture predict | 14-17 ms | **4-5 ms** | **4-5 ms** |
| realistic 2048² predict | 0.28 s | **0.17 s** | 0.29 s |

4096² fp32 forward, and note that tiling is not just a memory measure:

| | tiled-1024 | whole frame |
|---|---|---|
| tiny (117k params) | 0.367 s / 457 MiB | 0.296 s / 6465 MiB |
| small (1.93M params) | **0.901 s / 1045 MiB** | 8.781 s / 13127 MiB |

`small` whole-frame wants 13 GB on a 12 GB card, so it spills and runs **10x
slower** than tiled. Any future CNN path must tile — the whole-frame route is
only viable for `tiny`.

### Verdict

Not wired in, and it should stay that way. Inference is 1.6-3.5x cheaper, which
would matter for the 900-frame batch (55 min, above) — but only at quality
parity, and it is not close on sparse labels. If this is revisited, the thing to
attack is label efficiency (pretraining, heavier augmentation, or a loss that
does not reward over-segmentation), not step count or model size: `small` has
16x the parameters of `tiny` and is WORSE on the fixture.

---

## Non-rigid drift at scale — 4096² x hundreds of frames (2026-07-31)

`python -m spyde.tests.benchmark_drift_nonrigid --frames 300`, CUDA (TITAN X
Pascal), 120 steps.

The first fact is that the stack CANNOT be held: 300 x 4096² float32 is
**20.1 GB**. So the cost is two separate numbers that scale differently, and only
one of them is paid per frame. Quoting a single blended figure would hide the
one thing a caller has to decide — how much to decimate.

### The FIT is cheap — the whole movie at once

A drift field is smooth by construction (that IS the modelling assumption), so it
does not need full resolution to be measured. The fit is over a handful of
parameters per frame — 2 x n_knots, or 2 x gh x gw — and decimation is the
dominant knob:

| fit size | decimation | scan-knot | dense (6x6) |
|---|---|---|---|
| 128² | 32x | 2.73 s | 2.29 s |
| 256² | 16x | 2.41 s | 7.71 s |
| 512² | 8x | **9.08 s** | **28.39 s** |

That is for all 300 frames together, i.e. 8-95 ms per frame. Scan-knot barely
notices the resolution (it has ~6 parameters per frame); dense scales with it,
because a 6x6 grid bicubically upsampled to 512² is real work per step.

`_NONRIGID_FIT_SIDE = 512` is the conservative choice — more signal for the
correlation. 256² is 1.2x cheaper for scan-knot and **3.7x** for dense, and a
smooth field should be perfectly measurable there; that is worth testing against
recovery accuracy before anyone pays the 28 s.

### The APPLY is the expensive half, and it is PER FRAME

| | ms/frame at 4096² | 300 frames |
|---|---|---|
| scan-knot | **385 ms** | 115 s |
| dense | **432 ms** | 130 s |

**This is the number that matters operationally.** 385 ms is ~23x the 16.7 ms
60 fps budget, so a non-rigid corrected movie CANNOT be scrubbed frame-by-frame
the way a rigid one can — rigid applies as an `np.roll` for integer shifts and
preserves dtype exactly (see `DriftModel.is_integer`), while non-rigid resamples
every pixel through `grid_sample`. Two consequences worth stating plainly:

* For EXPORT / batch, ~2 minutes over a 300-frame movie is a reasonable price
  and is dominated by the per-frame resample, not the fit.
* For INTERACTIVE display, the corrected node needs the same treatment as any
  other expensive per-frame read — the tiered nav read routes it async
  (Live-Display §3), or the field is applied to a decimated view for scrubbing
  and only at full resolution on commit.

The fit is therefore NOT the thing to optimise. Even the slowest fit measured
(dense at 512², 28 s) is a quarter of the apply cost over the same movie.

### Reading the movie to fit it

The decimated read is one full streaming pass — `_decimated_stack` reads frames
ONE AT A TIME and strides each immediately, so the 20 GB is never resident. On a
real `.mrc` at ~3 GB/s that pass is ~7 s, i.e. comparable to the 128²/256² fits
and cheap against the apply. Strided rather than area-averaged on purpose: the
fit needs crisp gradients to correlate, and a box mean blurs exactly those.

### Making the apply faster — it is TRANSFER-bound, not compute-bound

The 385 ms above was a CPU number: `apply_nonrigid` had no `device` argument and
always ran on the host, on a machine whose GPU the *fit* was already using.
Profiling the stages at 4096² says where it goes and what is worth attacking:

| CPU stage | | CUDA | |
|---|---|---|---|
| build field | 68 ms | warp, frame resident | **7.9 ms** |
| warp (grid_sample) | 262 ms | + both host<->device copies | 41 ms |
| **total** | **392 ms** | | |

**The warp itself is 7.9 ms; the other ~33 ms is PCIe.** So micro-optimising the
resample buys almost nothing — the lever is not moving the data. Two things
follow, and the second is the interesting one:

* The field is now built on the DEVICE from the fitted parameters (a few hundred
  bytes) instead of on the host and shipped. Building it host-side would add
  134 MB to the very thing that already dominates.
* A batch pipeline that keeps frames RESIDENT pays only the 7.9 ms — **~2.4 s for
  a 300-frame movie** instead of ~14 s. That is the shape any future
  "correct the whole movie on the GPU" path should take; per-frame calls from
  host memory can never beat the copy.

After adding `device=` (auto: CUDA when present) and dropping a redundant 67 MB
`.copy()` on the way out — which was itself a fifth of the GPU path's cost:

| | ms/frame at 4096² | 300 frames | |
|---|---|---|---|
| CPU (was 392) | 278 ms | 83.5 s | the field build no longer round-trips numpy |
| **CUDA (default when present)** | **47.8 ms** | **14.4 s** | **5.8x** |

CPU/CUDA agreement is `max|diff| = 4.2e-04` on data in [0, 1] with identical NaN
masks — float32 `grid_sample` kernels differ slightly between backends, so this
is close-but-not-bit-identical, unlike the region-integrator's exact contract.
Fine for a resampled display frame; do not build an equality test on it.

MPS is deliberately NOT selected by `device=None`. The win here is one fused
kernel, and an unsolicited MPS submission contends for the shared device lock
that the neural/scribble paths hold. Opt in with `device="mps"`.

### The navigator fill was submitting ONE task at a time (2026-08-01)

Reported as "why is the computing navigator using a single worker" — the dask
dashboard showed the distributed backend live, the cluster idle, and tasks
arriving one by one. It got worse with dataset size (an 800 GB 4D-STEM scan and
a long movie both), which is the tell.

Not a placement bug. `dispatch_chunks` tops up on EVERY completion, so on the
**unpinned** lane `lane_cap - outstanding` is 1 in steady state and

    n = min(submit_batch, len(pending), lane_cap - outstanding[lane])

collapses to **n = 1**. `submit_batch=8` only ever applied to the first fill.
Every subsequent chunk was its own blocking scheduler round trip with the GIL
held in the client process — the exact cost #95 was written to remove,
reintroduced through the back door, and scaling with chunk count.

The window was never justified for this lane anyway:

* **It did not measure as backpressure.** Bounded 46-50 s vs unbounded 50.2 s on
  the same 977-chunk movie (above) — within noise.
* **`distributed` >= 2022.3 already does it**, queuing root tasks at the
  scheduler. We were duplicating the scheduler's job, worse, in our process.
* There is **no placement decision** to make on an unpinned lane, so there is
  nothing for a window to balance.

Fixed by priming with one small batch and then sending the rest in a single
submit. 977 chunks, 6 workers x 2 threads:

| | submits | first chunk | total |
|---|---|---|---|
| one-at-a-time (the bug) | **970** | 656 ms | 12.40 s |
| all-at-once | 1 | **1292 ms** | 5.03 s |
| **prime + bulk (shipped)** | **2** | **45 ms** | **5.28 s** |

**485x fewer round trips, 14.6x faster to first paint, 2.3x faster overall** —
and note the middle row. Going straight to one all-at-once submit is fastest in
total but DOUBLES time-to-first-chunk, because the client serialises the whole
graph before anything comes back. The progressive fill exists so the navigator
starts filling immediately, so that regression matters as much as the total; the
priming batch buys it back for one extra round trip. Measuring only wall-clock
would have shipped the wrong one.

On a synthetic graph each round trip is cheap, so these totals UNDERSTATE the
real gain: on a real memmap-backed graph a submit is ~14 ms (above), i.e. ~13.6 s
of GIL-held client time for 977 chunks.

**The dual-lane path keeps its window.** There a completion genuinely pulls the
next chunk so a ~30x-faster GPU lane and the CPU lane drain one pool and finish
together — real work stealing the scheduler cannot do, and the reason this
module exists. Only the unpinned lane changed.

### Batch segmentation paused dask workers — the per-FRAME peak, not the graph (2026-08-01)

Reported from a real run: workers pausing at 80% of a 9.24 GiB limit, restarting
at 95%, and `Unmanaged memory: 6.47 GiB`.

The graph was never the problem. `segment_movie` is a plain `map_blocks` over
time chunks and nothing calls `.compute()` on the movie — it is genuinely
embarrassingly parallel. What was wrong is that ONE unit of that parallel work
is enormous. Measured, 4096² frame, classical engine:

| stage | peak | time |
|---|---|---|
| input frame (float32) | 64 MB | — |
| `_prepare` | 24 MB | 0.04 s |
| threshold | 7 MB | 0.02 s |
| **`split_instances` (watershed)** | **546 MB** | **4.15 s** |
| **whole frame** | **852 MB** | 4.4 s |

The cluster runs `threads_per_worker=4`, so that is **~3.4 GB of concurrent peak
on one worker** before frames in flight or allocator fragmentation. And
"unmanaged" is the correct label: the rasters are ours, inside the task, so dask
can neither account for them nor spill them.

The watershed is 64% of the peak because the distance transform, its smoothed
copy, the marker labelling and the watershed each materialise a full-frame
float32/int32 raster.

**Fix: watershed each connected component in its own bbox.** This is EXACT, not
an approximation — a component is surrounded by background by definition, so a
1 px pad contains every pixel the distance transform, the markers and the
watershed can depend on, and no watershed can flow between components that do
not touch. It is the same shape `measure.py` already uses to trace contours.

| 4096², 400 overlapping discs | peak | time |
|---|---|---|
| whole-frame | 538 MB | 1.59 s |
| **per-component** | **272 MB** | **1.09 s** |

**2x less memory and 1.5x faster.** Faster because the distance transform now
runs over the particles' own bboxes instead of 16.7 M pixels of mostly
background.

**One behaviour change, and it is an improvement.** The counts differ on large
frames (395 vs 388 above) because `_split_factor` decimates the whole-frame
split geometry 4x at 4096² (the earlier 7.1 s -> 2.8 s optimisation), while a
per-component crop is a few hundred pixels and so gets factor 1 — full
resolution markers. At 1024², where neither route decimates, the two agree
PIXEL FOR PIXEL, which is what the parity test pins. So the per-component route
is more accurate as well as smaller; the decimation that bought the original
speedup is simply no longer needed on this path.

Gated at `_COMPONENT_ROUTE_PX` (4 MP): below it the per-crop bookkeeping costs
more than the memory it saves.

---

## Segmentation preview on a LOW-CONTRAST frame (2026-08-02)

Reported from a real in-situ movie: "14028 particles in this region", a solid
green preview window, and 4.1 s per tune. Reproduced on a synthetic stand-in —
noisy support film, 8 faint dark particles, 1024², `invert=True`, otsu.

**Otsu has no bimodal histogram to find here, so no caret knob rescues it:**

| settings | instances | coverage | block-ANY coverage at 1/4 | time |
|---|---|---|---|---|
| defaults (`min_size=20`, watershed) | 4873 | 39.0% | 51.1% | 810 ms |
| `min_size=200` | 208 | 14.4% | 16.5% | 595 ms |
| `min_size=2000` | 17 | 7.2% | 7.9% | 577 ms |
| watershed off | 751 | 39.8% | 52.3% | 67 ms |
| `gaussian=2` | 431 | 52.7% | 55.0% | 392 ms |
| rolling ball 64 + `gaussian=2` | 2280 | 26.4% | 39.4% | 11191 ms |
| `gaussian=2` + `min_size=200` + no watershed | 8 | 52.4% | 54.2% | 101 ms |

The last row is why `_threshold_failed` tests count AND coverage: 8 instances
looks like the 8 real particles, but at 52% coverage those 8 bodies are the
film. And note the block-ANY column — the overview reduction the tiled overlay
uses turns 39% coverage into 51%, so a shattered frame composites into a sheet.

**Where the preview's time goes** (same frame, warm):

| stage | over-segmented (n=4873) | filtered (n=17) |
|---|---|---|
| `segment_frame` (threshold + watershed + filter) | 706 ms | 569 ms |
| `measure_frame` (props + contours) | 647 ms | 62 ms |
| **total** | **1353 ms** | **631 ms** |

`watershed=True` is 700 ms of that vs 61 ms off — the split itself is ~639 ms
and is inherent to a mask covering 39% of the frame, so it is NOT the part to
optimise; the fix is to stop producing such a mask.

The contours are, though: they are the bulk of `measure_frame` at high instance
counts and are thrown away above the overlay's draw cap. `want_contours=False`
takes the preview **1513 -> 969 ms (36% faster)** at n=4873 with the measured
rows **bit-identical** (`np.array_equal`), so the count, histogram, median and
confidence filter are unaffected. The saving scales with instance count, so the
reported 14028-instance frame gains proportionally more.

Reproduce in the app: `load_test_data_particles {noise: 0.35, size: [1200,1200]}`
(`seg_oversegment.spec.ts`). At the default `noise=0.015` the fixture is clean,
a global threshold works on it, and none of this is visible.

## Apple-MPS for the 0.3.0 compute paths (2026-07-31)

**Test system:** MacBookAir10,1 (M1, 4 performance + 4 efficiency cores, 8-core
GPU), macOS 26.4, torch 2.13.0, `torch.get_num_threads() == 4`. This is the
*smallest* Apple GPU paired with a strong CPU — the pessimistic end for MPS, not
a typical Mac. Every row is float32 both sides (Metal has no float64).

### Does it work at all?

Yes. Every op these paths need runs on Metal: batched `matmul`, `linalg.solve`,
`linalg.cholesky_ex`, `cholesky_solve`, `diag_embed`, `topk`, `fft.rfft2/irfft2`
and `torch.func.vmap(jacfwd)`. The only failure is `float64`, which raises — hence
`resolve_dtype()` in `fitting/engine.py` and `ebsd/_device.py`.

### Accuracy: float32 vs float64, and MPS vs CPU

Max relative deviation from HyperSpy `multifit` on the same data:

| case | cpu f64 | cpu f32 | **mps f32** |
|---|---|---|---|
| Offset + Gaussian | 8.4e-8 | 9.3e-6 | **9.6e-6** |
| PowerLaw (A~1e6, r~3) | 2.1e-13 | 8.9e-8 | **7.8e-8** |

MPS tracks CPU-float32 to the last digit, so the device contributes essentially
nothing — the whole difference is dtype, and it is 4 orders inside the 1e-4
parity tolerance. EBSD indexing agrees with `kikuchipy.indexing`'s NCC to
**1.8e-7** (float32 rounding) with 100% best-match agreement, on CPU and MPS
alike; pinned by `test_ebsd_indexing.py::TestKikuchipyParity`.

### Two bugs this uncovered (both dtype, neither MPS-specific)

1. **`convergence_rate` collapsed to 0.00 in float32.** The `ftol/xtol/gtol`
   defaults are 1e-8; float32 epsilon is 1.19e-7, so all three tests were
   unreachable. Every position reported "did not converge" and burned all 60
   iterations while its parameters were correct to ~1e-5 — the user-visible
   symptom is a 0% coverage map on a good fit. Fixed by `floor_tolerances()`
   (10x eps, a no-op in float64).
2. **The gradient test is 0/0 on a near-perfect fit.** `|Jᵀr| / ||r||` is
   meaningless once the residual reaches the rounding floor: on CLEAN data in
   float32 an exact fit (chisq 1e-11, 8 correct digits) still reported 50%
   converged. Fixed with an absolute noise-floor test — a residual at the
   representation limit cannot be improved, so it IS converged.

### Speed

Removing the two per-iteration device->host syncs from the LM loop (`solvable.any()`
and the `converged.all()` early exit; one `.any()` costs 367 us on MPS against
30 us left on-device) was worth **4.3x** on a small fit — MPS 1303 ms -> 304 ms —
and helps CUDA for the same reason.

Raw float32 GEMM ceiling on this box: **MPS 1747 vs CPU 868 GFLOP/s = 2.0x.**

| path | size | CPU | MPS | MPS/CPU |
|---|---|---|---|---|
| fitting engine | P=64, C=1024 | 171 ms | 173 ms | 0.99x |
| fitting engine | P=1024 | 1.72 s | 2.48 s | 0.69x |
| fitting engine | P=16384 | 28.4 s | 41.6 s | 0.68x |
| EBSD indexing | P=256, D=1k, 60² | 4.7 ms | 9.3 ms | 0.50x |
| EBSD indexing | P=1024, D=5k, 60² | 54.7 ms | 64.0 ms | 0.85x |
| EBSD indexing | P=4096, D=20k, 60² | 727 ms | 685 ms | **1.06x** |
| EBSD indexing | P=4096, D=20k, 80² | 1286 ms | 1112 ms | **1.16x** |

**The split is the workload, not the backend.** EBSD indexing is one big
`E @ Dᵀ`, so it amortises launch overhead and crosses over at production scale,
rising with size exactly as a GEMM-bound path should. The fitting engine is the
opposite shape: `n <= 20` free parameters make `JᵀJ` a batch of *tiny* matrices,
so an LM iteration is many small kernels — the case CLAUDE.md's GPU section warns
about — and it stays below 1.0 here.

**Do not turn this table into a size threshold.** It is one small GPU; a
Pro/Max/Ultra part has 2-8x the GPU compute against the same per-launch cost and
moves every row. Both paths therefore prefer the accelerator and expose an
override (`SPYDE_FIT_DEVICE`, `SPYDE_EBSD_DEVICE`) so a real box can be measured
rather than guessed.

**Measurement trap:** the un-warmed version of the EBSD table reported 0.02x at
the smallest size and 0.72x at the largest, which reads as "MPS never wins". That
was Metal context creation (~0.5 s) landing inside the first timed call. Always
warm up before timing MPS.

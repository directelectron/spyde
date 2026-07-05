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

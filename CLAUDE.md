# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpyDE is a desktop application for visualizing and analyzing electron microscopy data (TEM, STEM, Cryo EM, 4D STEM, EELS). The UI is an **Electron + React/TypeScript** frontend; the compute/data engine is a **Python backend** (`spyde/backend/`) that the Electron main process spawns as a subprocess and talks to over stdin/stdout JSON lines (the `PLOTAPP:` protocol from `anyplotlib._electron`). Plots are rendered by **anyplotlib** (figures embedded as HTML in the renderer), not by a native Qt widget. It wraps HyperSpy and PyXEM, with Dask-based parallel computing and a signal transformation tree.

> **History note:** SpyDE began as a PySide6/pyqtgraph Qt app and was migrated to the Electron/anyplotlib architecture above. The old Qt code (`QMainWindow`, `QMdiArea`, `QThread`, pyqtgraph widgets, `spyde/qt/`) is **gone** — if you see "Qt"/"pyqtgraph" in a comment or an older memory, it's historical. Patterns described as "ported from the Qt app" mean the *algorithm/approach* was carried over, not the framework.

## Commands

**Install the Python backend (dev):**
```bash
pip install -e ".[tests]"   # or: uv pip install -e ".[tests]"
```

**Install the Electron frontend (dev):**
```bash
cd electron && npm install
```

**Run the app (dev):** from `electron/`, `npm run dev` (electron-vite; spawns the Python backend as a subprocess). Running `python -m spyde` alone launches only the **backend** (asyncio stdin/stdout loop) — useful for debugging the backend, but there's no UI without Electron.

**Run the Python tests** (Qt-free, build a real `Session`):
```bash
pytest spyde/tests/migrated/                                   # whole suite
pytest spyde/tests/migrated/test_navigator_race.py             # one file
pytest spyde/tests/migrated/test_navigator_race.py::TestNavigatorRace::test_x   # one test
```
Slow benchmarks live in `spyde/tests/benchmark_*.py` — run directly (`python -m spyde.tests.benchmark_<name>`), not under pytest.

**Run the Electron e2e (Playwright):** from `electron/`, `npm test` (or `npm run test:build` to build first).

**Build distributable:** from `electron/`, `npm run dist` (electron-vite build + `bundle:python` + electron-builder). See `electron/electron-builder.yml` and `DISTRIBUTION_PLAN.md`.

**Commits:** do NOT add Claude/AI as a co-author. Commit messages must not include a `Co-Authored-By: Claude …` (or `Claude-Session`) trailer. (Enforced via `includeCoAuthoredBy: false` + empty `attribution` in `~/.claude/settings.json`.)

## Dependencies

Python deps are in `pyproject.toml`. Key non-PyPI deps from custom forks (check `pyproject.toml` for the exact pinned branch — they move):
- `hyperspy` → `github.com/cssfrancis/hyperspy@slice-integrate2` (the navigator `CachedDaskArray` / `get_index` fixes live here — see Live-Display §4)
- `rosettasciio` → `github.com/cssfrancis/rosettasciio@win32-binary-read`

Frontend deps (Electron, React, electron-vite, Playwright) are in `electron/package.json`.

Supported file extensions: `.hspy`, `.zspy`, `.mrc`, `.tif`, `.tiff`, `.de5` (see `SUPPORTED_EXTS` in `session.py`).

## Architecture

### Entry Points
- `spyde/__main__.py` → `main()`: calls `multiprocessing.freeze_support()` then `spyde.backend.app.run()` (the asyncio backend). The Electron main process spawns this as a subprocess.
- `spyde/backend/app.py` → `run()` / `_main()`: the asyncio event loop that **replaces** `QApplication.exec()`. Reads JSON messages from stdin, dispatches them to the `Session`, and writes figure/stream messages to stdout.
- `main.py` (root): PyCrucible/frozen-app launcher wrapper (also calls `freeze_support()`) that delegates to `spyde.__main__.main()`.
- `electron/src/{main,preload,renderer}`: the Electron app — `main` (Node process, spawns Python + bridges IPC), `preload` (contextBridge), `renderer` (React/TS UI, the log panel, figure iframes).

### Session (`spyde/backend/session.py`)
`Session` is the Python-side coordinator (the old `MainWindow`'s role, minus Qt). It owns: the signal trees, the Dask cluster (via `DaskManager`), plot registration (`_plots`), file I/O, and action dispatch. All communication to Electron goes through `spyde/backend/ipc.py` `emit()`. It marshals worker results back onto the asyncio main thread via `set_main_loop()` + `_dispatch_to_main()`. Tests construct a `Session` directly.

### Signal Tree (`spyde/signal_tree.py`)
`BaseSignalTree` tracks a DAG of signal transformations. Each node is a HyperSpy `BaseSignal` with associated `Plot`(s). Non-breaking transformations (e.g. filtering, centering) update the current plot in-place; breaking transformations (e.g. azimuthal integration) create new branches. Users can navigate the tree to compare states.

### Drawing Layer (`spyde/drawing/`)
- `plots/plot.py`: `Plot` — wraps an **anyplotlib** figure (`anyplotlib._electron`); pushes image/line data to the embedded HTML view. Holds the per-plot shared-memory buffer and `current_data`.
- `plots/plot_window.py`: `PlotWindow` — a logical container for one or more `Plot`s (the renderer lays these out; there is no `QMdiSubWindow`).
- `plots/plot_states.py`: state machine governing how navigator and signal plots synchronize.
- `plots/multiplot_manager.py`: manages multi-panel layouts; `navigation_selectors` maps a navigator `PlotWindow` → its selectors.
- `selectors/`: 1D and 2D ROI/crosshair selectors (wrappers around **anyplotlib interactive widgets**) used to slice the HyperSpy navigation space. `base_selector.py` holds the `_NavDispatcher` (see Live-Display §2) and `event_handler_fn`.
- `toolbars/`: toolbar/button-bar/caret config that the renderer renders.
- `update_functions.py`: functions that compute what data to display given the current plot state (incl. `update_from_navigation_selection`, the navigator→DP path).

### Actions & toolbars (`spyde/actions/`)
**Read `spyde/actions/README.md` before adding or changing an action** — it is the contract: the action taxonomy (View / TransformAction / RegionAction / Wizard / Commit), the TWO dispatch paths (YAML toolbar via `ActionContext`, staged wizard via `registry.STAGED_HANDLERS` with `<key>_open/_close/_tune/_run/_commit` verbs), the lifecycle + ownership map, and copyable skeletons (`_template_action.py`). Framework modules:
- `action.py` / `wizard.py`: the template base classes (`TransformAction`, `RegionAction`, `WizardController`)
- `registry.py`: staged-action table + the WindowController protocol (bare-figure windows register in `session._window_controllers`)
- `lifecycle.py`: the shared basis set — `run_on_worker` (worker→main marshal), `bump_generation`/`is_current` (StrictMode/latest-wins guard), `wait_for_vectors` (the attach gap), `replace_tree_attr`, `paint_signal_plots`, `live_fill_poller`
- `commit.py`: `open_result_tree` (early/progressive window) + `commit_result_tree` (THE Commit action: new SignalTree with chip views + provenance)
- `figure_registry.py`: per-window figure keep-alive, evicted by `_forget_window`
- `find_vectors/`: Qt-free Find-Vectors compute package (the model for splitting heavy compute); `find_vectors_action.py` / `vector_overlay.py` are the interactive wiring
- `_common.py`: small shared helpers (`reciprocal_radius`, strain component constants, `widget_region`); `base.py` also defines `NAVIGATOR_DRAG_MIME`

### Compute Backend (`spyde/compute_backend.py`)
`ComputeBackend` provides a uniform `concurrent.futures.Future`-compatible interface over two modes:
- **Threaded** (default): `ThreadPoolExecutor` — low overhead, no Dask scheduler
- **Distributed**: wraps `dask.distributed.Client` futures via `_DistributedFutureAdapter`

Key methods: `.submit()`, `.compute()`, `.compute_chunks_progressive()` (streaming chunk results). Callers never import Dask directly; switch modes by swapping the backend instance.

### Workers (`spyde/workers/`)
- `plot_update_worker.py`: `PlotUpdateWorker` runs on a **plain daemon thread** (not a `QThread`); polls `dask.distributed.Future` objects, reads the completed result (from the per-plot shm buffer for the navigator path), and **marshals the apply onto the asyncio main thread** via the `dispatch` callback (`loop.call_soon_threadsafe`) → `Session._on_plot_ready` / `_on_signal_ready`. It emits via `psygnal` signals, not Qt signals.

### Live Instrument Control (`spyde/live/`)
WIP modules for live microscope control: camera, stage, STEM, TEM, particle scanning, reference.

### Signals (`spyde/signals/`)
- `diffraction_vectors.py`: `SpyDEDiffractionVectors` — GPU-optimized CSR flat-buffer container for ragged diffraction vectors. Stores `(nav_x, nav_y, kx, ky, intensity)` with an offsets array (row-pointers). Key methods: `.at()`, `.kxy_at()`, `.count_map()`, `.to_dense()` (cached), `.to_pyxem()`, `.cluster()`, `.get_strain_maps()`.

### Vector orientation mapping (`spyde/actions/`)
- `vector_orientation.py`: CPU reference — per-pattern scipy-LM fit of pose `(θ, A, t)` where `v ≈ A·Rot(θ)·g_template + t`. `_residual` (soft-assign + no-match sink + strain-band penalty) is the cost both paths must agree on. Strain via polar decomposition of `M = A·Rot(θ)`.
- `vector_orientation_gpu.py`: **the production path** — fits the *whole field at once* on the GPU (batched torch + Adam), no dask, no per-pattern loop. The vectors and library are tiny, so the entire scan is one batched optimisation. `compute_vector_orientation_gpu()` is dispatched first when `gpu_available()`; CPU is the fallback. On SpEd Ag (13k patterns × 1081 templates) it runs in ~8s. See the GPU Computing section for the non-obvious constraints baked into it.

### Backend IPC / logging (`spyde/backend/`)
- `ipc.py`: `emit()` / `emit_status` / `emit_error` / `emit_progress` — write JSON messages to stdout for the Electron main process to relay to the renderer.
- `log_stream.py`: tags each log record with a subsystem `area` (`_area_for` / `_AREA_RULES`) and streams it to the renderer's Log panel (which has search + area filter).
- `process_guard.py`: reaps orphaned Dask worker subprocesses on exit (Windows Job Object).

## Testing

Tests are **Qt-free** (no `pytest-qt`, no `QApplication`). They build a real `Session` (with a 1-worker Dask cluster) and assert on the JSON messages it emits + the signal-tree/plot state. Fixtures live in `spyde/tests/migrated/conftest.py`:

| Fixture | Data | Yields |
|---|---|---|
| `window` | empty session | `{window: Session, signal_trees, plots, messages}` |
| `tem_2d_dataset` | 2D image | same dict |
| `stem_4d_dataset` | 4D STEM (2D nav, 2D signal) | same dict |

`captured_messages` monkeypatches `ipc.emit` (bound into `session.py` at import) to capture outgoing messages. `window["window"]` is the `Session`; `window["plots"]` is `session._plots`. Each fixture calls `session.shutdown()` on teardown.

- **`torch`-CUDA work segfaults under the pytest process on Windows** (harness interaction, not a code bug — fine in the real app / plain Python). Run GPU correctness tests in a **subprocess** that prints a JSON result and `os._exit(0)` after (see `test_vector_orientation_gpu.py`). GUI-wiring tests that exercise the path should force the CPU branch (`monkeypatch gpu_available → False`).
- Distributed repros that spin a real `LocalCluster(processes=True)` likewise need a subprocess and won't run inside an agent sandbox — run them yourself (e.g. `uv run python -m spyde.tests.repro_write_cancelled`).
- Tests are written as classes with methods (e.g. `class TestActions` → `def test_center_direct_beam`).

## Verify by RUNNING THE APP — headless tests + typecheck are NOT verification

**A passing pytest suite and a clean `tsc` do NOT mean a UI feature works. They mean the code is structurally sound. They cannot see: duplicate windows piling up, a caret that never tears down, an overlay that draws in the wrong colour, a second window that never opens, a control that silently no-ops.** Any feature that adds/removes windows, draws overlays, toggles on an action, or wires the renderer↔backend MUST be verified by launching the real Electron app, driving it, and **looking at a screenshot** before you claim it works. The screenshot IS the test. If you have not looked at the pixels, say "built + headless-tested, needs your eyes" — never "it works."

**Do NOT hand-roll a launcher.** A proven, signal-based Playwright harness already exists — copy it, don't reinvent it (repeatedly writing throwaway `_electron` probe scripts with blind `waitForTimeout`s wasted an entire session and mis-diagnosed the harness's own noise as app bugs).

- **Harness:** `electron/tests/_harness.cjs` — `launchApp({dask:true, env})` waits for `[spyde backend] ready` + `dask_ready`; gives `backend.waitForLog`/`waitForMessage`, `waitForSubwindowCount`, `countColorPixels`, and `assertNoJsErrors`. **Copy the shape of `find_vectors_workflow.spec.ts`** (real Dask + bundled-synthetic data) for anything vectors/strain/orientation.
- **Load real-ish data the way a user does:** `backendAction(page, 'load_test_data_si_grains')` (bundled synthetic, crisp reciprocal lattice — find-vectors can actually detect spots) or `load_example {name}` (Examples menu; `zrnb_precipitate` etc., needs download+dask). `load_test_data*`/`load_test_vectors` are the fast bundled paths.
- **Screenshot each stage** to `electron/<name>_shots/NN-step.png` and Read them. A blank/black frame is a failure to launch or a stale placeholder, not success.
- **Backend `emit`/`emit_error`/`emit_status` do NOT reach Playwright stdout** (they're the `PLOTAPP:` line protocol, consumed by the main process). To see a backend error, either read `ctx.backend.logBuffer` at the end of the test, or set `SPYDE_LOG_LEVEL=WARNING` in `launchApp({env})` so `logging` tees to stderr (which the harness captures). Watching plain stdout for a status string will silently miss the error.
- **Run:** `npx playwright test tests/<spec>.spec.ts --project=electron --reporter=line --retries=0`. Kill strays first if flaky (`Get-Process electron,python | Stop-Process -Force`), but don't over-attribute flakiness to the app — a polluted local env (repeated relaunches, leftover processes) produces slow dask / port contention that is YOUR test setup, not a real bug. On this dev box a healthy `LocalCluster` scheduler starts in ~1 s.

**The find-vectors→downstream timing trap (this WILL bite):** `find_diffraction_vectors` opens its result window EARLY (count-map placeholder) but attaches `tree.diffraction_vectors` only when the streaming batch **finishes** (`_finalize`, which also emits `"Found N diffraction vectors"` and re-sends the toolbar config — the vector actions are `requires_vectors`-gated so they appear only then). An action that needs the vectors can fire in the gap and find `diffraction_vectors=None` on a tree that gets it seconds later. Don't gate a test on a fixed sleep; wait for the real completion signal (the `"Found"` status, or poll the attribute). Backend handlers self-wait via `lifecycle.wait_for_vectors` (strain/VOM/vector-VI all do; use `strict=True` when the handler gates on the clicked plot's own tree).

## Memory Safety Rule: Never Materialise Large Datasets

**`_do_compute_vectors` in `spyde/actions/find_vectors.py` must NEVER call `.compute()` or `.result()` on the full signal dataset.** Doing so loads hundreds of GB into RAM.

- For **numpy** data: the array is already in RAM — slice ghost-padded chunks directly.
- For **lazy dask** arrays: call `.compute()` on each small ghost-padded slice (`raw[py0:py1, px0:px1].compute()`) — never on `raw` itself.
- For **distributed Futures**: submit per-chunk tasks to the worker holding the future (Path B) — the worker does the slice locally, only small results return.

The 5D path slices by time index first (`raw[t, ...]`), producing a 4D chunk — use `sigma_tuple_2d_nav = (sigma, sigma, 0, 0)` for that blur, not the 5D `sigma_tuple`.

`test_find_vectors_memory.py` enforces this contract with 27 tests including a `patch.object` guard on `da.Array.compute` that raises if the full-dataset shape is ever computed.

## Thread Safety Constraints

- UI/figure updates must happen on the **asyncio main thread**. Background workers (e.g. `PlotUpdateWorker`, the `_NavDispatcher`) must marshal their results back via `Session._dispatch_to_main` (`loop.call_soon_threadsafe`) — never push to a `Plot`/emit IPC directly from the worker thread.
- Dask cluster startup is asynchronous (`DaskManager` builds it on a background thread and signals `ready` / `workers_ready`). Don't block the main loop waiting for it; submit compute only once a client exists.
- All navigator updates run serially on the single `_NavDispatcher` thread (latest-position-wins coalescing), so the hyperspy cache is never re-entered concurrently — no lock needed. See Live-Display §2 and §4.

## Live-Display Core Patterns (DO NOT "CLEAN UP" — they look hacky but every alternative tried is worse)

These three patterns are load-bearing for interactive performance. They look like
poor design and invite refactoring into something "proper" (a queue, a lock, a
rechunk, a direct return). **Every such attempt has made the app much worse**
(frozen navigators, stalled updates, multi-GB shuffles). Touch them only with a
benchmark on a real multi-GB scan and a specific reproduced bug — never on
aesthetic grounds.

### 1. Storage-aligned chunking — span the FULL signal dimension; never rechunk live

A 4D/5D-STEM dataset must be chunked so **each chunk holds whole signal frames**:
`(small_nav, small_nav, full_ky, full_kx)` (e.g. `(32, 32, 256, 256)`). The
navigator displays one diffraction pattern via `data[iy, ix]`, so a chunk that
**splits the signal axes** (RosettaSciIO's default auto-chunk is a balanced cube
like `(90,90,90,90)`) forces reading a 131 MB chunk spanning 90×90 nav positions
and *partial* frames to show one pattern — and the navigator sum is wrong/seamed
at chunk boundaries (partial-signal sums).

- **Fix at LOAD time**: `hs.load(path, lazy=True, chunks=(32,32,-1,-1))` — a lazy
  reload only rebuilds the dask graph (~0 s), it does NOT read or move data.
  `Session._signal_spanning_chunks` computes this and `_load_file_thread` reloads
  when the reader split the signal axes.
- **NEVER call `.rechunk()` on the full dataset to fix chunking** — that shuffles
  the entire multi-GB array through the scheduler. Storage-chunk *alignment* (load
  with the right chunks) beats any after-the-fact rechunk; see `benchmarks.md`
  (419 s vs 184 s when a "better" rechunk misaligned the ghost blocks).
- Batch computes (`_do_compute_vectors`, orientation) keep the stored chunking
  when it's already usable rather than rechunking to a theoretical optimum.

### 2. Navigator updates = ONE serial dispatcher + latest-position-wins — NOT a lock, NOT per-update threads

The navigator→signal update path must be **non-blocking** AND **non-concurrent**.
Every selector update runs on a single dedicated daemon thread, `_NavDispatcher`
(`base_selector.py`): `submit(selector)` coalesces by `id(selector)` (a newer
position replaces the queued one), and the worker runs one `_run_update` at a time.
The plot applies only the result of its current future (`plot.current_data is
future` staleness guard in `_on_plot_ready`), so superseded frames are dropped.

- **Why one thread, not per-update `threading.Timer`s:** concurrent updates raced
  hyperspy's `CachedDaskArray` block bookkeeping (`ValueError: (i, j) is not in
  list`). The serial dispatcher removes the concurrency at the source — so the
  cache is never re-entered and **no lock is needed**. (Earlier designs — an RLock
  held across the compute, then a generation counter `_gen_lock`/`_update_gen`/
  `is_stale_body`, then a `_cache_lock_ctx` — are all GONE. Don't reintroduce them.)
- `_run_update` commits `current_indices` **up front** then short-circuits an
  identical position, because the widget fires `pointer_move` + `pointer_up` =
  two submits per release.
- **Settle re-fire:** during a fast drag the in-flight futures get cancelled by
  latest-wins; `update_data` (re)arms ONE trailing timer (`_settle_timer`) that
  fires a single `force=True` update once motion stops, so the resting frame
  computes even if the user just holds still mid-drag. It has NO in-flight gate, so
  it cannot wedge.
- **Do NOT add self-pacing or a buffer ring** (skip-while-in-flight, per-future shm
  slots). Both were tried; both made it worse — a wedged gate / an infinite
  ~6-frame re-emit loop. Single shm buffer + serial dispatcher + latest-wins is it.
- A **cancelled** future is `done()` but its work never ran — never read its
  result/buffer (`read_shared_array` rejects an empty/torn buffer; `_on_plot_ready`
  drops superseded/torn results silently). Some churn is expected under latest-wins.
  But if EVERY get_inds/write future ends `cancelled`, that's the bug in §4 — fix it.
- Tests pin the contract: `test_navigator_race.py` (slow update must not block a
  newer one; stale result must not clobber), `test_shm_read_robust.py`.

### 3. Fast shared-memory display path — bypass TCP for the navigator image

For the distributed path, a chunk result is written into a per-plot **shared-memory
buffer** (`write_shared_array` / `read_shared_array`) and the `PlotUpdateWorker`
reads it locally when the future completes — instead of transferring the array over
the Dask TCP comm. This optimized navigator/VI pipeline's *approach* was carried
over from the original Qt app; the reused single buffer is race-safe because only
the LATEST future's result is applied (the `plot.current_data is future` staleness
guard in `_on_plot_ready`) — a frame clobbered by a newer write was going to be
dropped anyway. (Do NOT add a per-future buffer ring; see §2/§4.) The progressive
navigator (`signal_tree._start_progressive_nav_compute` +
`compute_with_live_buffer`) paints per-chunk into this buffer so a multi-GB
navigator fills top-to-bottom instead of blanking until the whole sum finishes.

- Don't replace the shm path with `future.result()` over TCP "for simplicity" —
  it's measurably slower on real scans and was deliberately built this way.
- Compute navigator display levels from the FULL accumulated finite data
  (robust 2–98% percentiles), with a final uniform repaint when the fill
  completes — per-chunk min/max levels make the contrast jump at chunk seams.

### 4. **The DP-stuck-on-a-stale-frame trap (cost a brutal multi-hour session — DO NOT re-derive)**

**Symptom:** dragging the navigator, the diffraction pattern (signal plot) freezes
on one frame; you may also see nothing in the Dask dashboard. The selector reports
changing indices and distinct `write_shared_array` futures fire per move, but
`[plot-paint] SIG` shows the **same content hash** every frame. It "used to work."

**There are TWO independent causes and you must fix BOTH. Diagnose, don't guess —
the path is: selector → `_run_update` → `update_from_navigation_selection` →
`CachedDaskArray.get_index(return_future=True)` (get_inds future) →
`client.submit(write_shared_array, get_inds_fut, …)` → shm → `PlotUpdateWorker` →
`_on_plot_ready`. Add `[plot-paint] hash=`, then a `[LIFE]` done-callback logging
`getinds_fut.status` + `write_fut.status`. If both are `cancelled`, it's the two
causes below. If the get_inds future is a NUMPY array (not a Future), it's the
ambient-client cause alone.**

1. **`CachedDaskArray._client` must be pinned to `DaskManager.client`.** The
   `.client` property falls back to `dask.distributed.get_client()` when `_client`
   is unset. Navigator updates run on the **`_NavDispatcher` thread** (a plain
   `threading.Thread`, not a Dask worker) where `get_client()` raises → returns
   `None` → the cache does a **silent synchronous threaded compute** (no dashboard
   tasks; `return_future=True` returns a numpy array, not a future; flapping
   client across threads corrupts the cached block state). FIX in
   `update_from_navigation_selection`, BEFORE `_get_cache_dask_chunk`:
   `cached_arr._client = child.main_window.dask_manager.client`.

2. **Do NOT `cancel_surrounding()` before the chunk request, and HOLD the get_inds
   future alive until its write lands.** Two things were cancelling EVERY get_inds
   (and its dependent `write_shared_array`) → state `cancelled` → the buffer kept
   the last surviving frame: (a) `cancel_surrounding()` cancels prefetch block
   futures, cascading into the core block the in-flight get_inds depends on —
   **removed that call**; (b) `current_img = fut` dropped the only client-side ref
   to the get_inds future, so distributed sent release-key and cancelled it before
   the write pulled its dependency — **now stashed on `plot._inflight_getinds[fut.key]`
   and released in the write's done-callback**. A cancelled future is `done()` but
   never ran; its shm write never happened.

Repros: `spyde/tests/repro_cache_client_thread.py` (prints `cache.client=THREADED/none`
from a timer thread) and `spyde/tests/repro_write_cancelled.py` (write future ends
`cancelled`). **Do NOT "fix" this with a buffer ring or self-pacing on `update_data`
— both were tried, both made it worse (infinite 6-frame re-emit loop / wedged
gate). Single shm buffer + serial dispatcher + latest-wins + the two fixes above is
the working design.**

## GPU Computing

The hot paths (vector finding, vector orientation mapping) are GPU-accelerated. The stack present in the dev env: `torch` (+CUDA), `cupy`, `numba.cuda`. Guard every GPU path with an availability check (`torch.cuda.is_available()`) and keep a working CPU fallback — CI and many user machines have no GPU.

**Batch the whole problem, don't loop.** The vectors and template library for a 4D-STEM scan are only a few MB. The win is transferring everything to the GPU once and running *every* nav position in lockstep as one batched tensor op — not dask, not processes, not a per-pattern Python loop. `vector_orientation_gpu.py` is the model: pack all P patterns → `(P, …)` tensors, one batched coarse seed, one batched Adam refine, one vectorised decode.

**Avoid per-item Python loops around tiny kernels.** The original coarse seed looped templates × angles in Python (hundreds of thousands of tiny kernel launches) → **289s** for a realistic library. Rewriting it as a polar-histogram angular cross-correlation (one batched FFT, no Python loop) → **1.6s**. When a GPU step is slow, the cause is almost always a Python loop launching small kernels or a blown-up intermediate tensor — not the arithmetic. Reach for FFTs / matmuls / `scatter_add_` over explicit loops, and **chunk the batch dimension** to bound the largest intermediate (e.g. the `(P,T,n_a)` correlation is chunked over patterns) rather than materialising it whole (a full `(P,T,…)` tensor OOMs).

**Windows + torch-CUDA-autograd gotchas (hard-won):**
- `backward()` segfaults when run off the **main thread** under CUDA on Windows. The GPU orientation fit therefore runs **inline on the main thread** (it's only ~1-2s of compute) with an `on_yield` callback (pumps the event loop / flushes pending work) so the UI stays responsive; it is *not* offloaded to a worker thread. The CPU fallback (numpy/scipy) is thread-safe and *does* run on a worker.
- Pin backward to the calling thread with `torch.autograd.set_multithreading_enabled(False)` around the refine loop.
- Yield *inside* the step loop (every ~12 steps), not just per anneal stage — otherwise the window freezes for seconds and the progress bar appears stuck. Drive the progress label from the compute's own `progress(done,total)` callback; do not derive % from a lagging live-preview cell count.

**Numerical traps that only show on real/strained data** (unit tests on uniform synthetic data won't catch these):
- *Rotation-branch ambiguity*: a centrosymmetric diffraction pattern is invariant under 180°, so the seed may pick θ≈±180° where an SPD-bounded stretch can't fit → garbage strain. Collapse the seed angle into `(−π/2, π/2]`.
- *Coarse-σ shrink bias*: at wide Gaussian σ the soft-assign cost is minimised by shrinking the template (spurious negative strain pinned at the cap). Fit a **rigid pose through the coarse stages and only release the strain DOF at the finest σ**, where the true strain is the global minimum.

## Benchmarking

**Always benchmark on a real dataset at real scale, end-to-end.** The canonical target is `pyxem.data.sped_ag()` — 208×64 = 13,312 patterns of 112×112 (a real 4D-STEM SpEd Ag scan). Synthetic 4×4 fixtures validate *correctness* but hide the costs that actually bite (per-Python-loop overhead, Vmax-padding blowup, library size). A method that's instant on 16 patterns can be minutes on 13k.

- Existing harnesses live in `spyde/tests/benchmark_*.py` (run directly with `python -m spyde.tests.benchmark_<name>`, not under pytest — they're slow). `benchmark_vector_orientation.py` builds the Ag library + sped_ag vectors and is the reference for the OM path. `benchmarks.md` records the numbers.
- **Time each stage separately** (vector finding / library build / orientation fit) — the user's "it's slow" is usually one stage, and conflating them hides the real bottleneck. Print `progress(done, total)` with timestamps to see whether a stage is progressing or genuinely stuck.
- For GPU timing, `torch.cuda.synchronize()` before/after the timed region (kernels are async) and **discard the first run** (cold CUDA init + kernel JIT is a one-time ~5s cost; report the warm steady-state too).
- `torch`-CUDA work **segfaults under the pytest process on Windows** (a harness interaction, not a code bug — it runs fine in plain Python and in the real app). So: run GPU correctness tests in a **subprocess** that prints a JSON result (see `test_vector_orientation_gpu.py`), and `os._exit(0)` after printing to skip the torch/CUDA teardown crash. GUI tests that exercise the *wiring* should force the CPU path (`monkeypatch gpu_available → False`).

## Configuration Files

- `spyde/*.yaml`: loaded at import time in `spyde/__init__.py` (toolbar and metadata widget configs)

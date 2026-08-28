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

**Build distributable:** from `electron/`, `npm run dist` (electron-vite build + `bundle:python` + electron-builder). See `electron/electron-builder.yml`.

**Commits:** do NOT add Claude/AI as a co-author. Commit messages must not include a `Co-Authored-By: Claude …` (or `Claude-Session`) trailer. (Enforced via `includeCoAuthoredBy: false` + empty `attribution` in `~/.claude/settings.json`.)

**Working notes and PR screenshots are NOT tracked:** plan / handoff / parity / checklist / architecture Markdown (`HANDOFF.md`, `docs/notes/*.md`, `OVERNIGHT.md`, `ARCHITECTURE-SPLIT.md`, and the like) is scratch belonging to ONE branch — do not `git add` it. A fact worth keeping goes in a test, a docstring, or the commit message, where it sits next to the code that makes it true and gets updated with it; a standalone note just rots into a confident description of an app that no longer exists, and a reader can't tell that from current guidance. Same for the PR screenshots under `docs/pr/**` — attach them to the GitHub PR or issue, which hosts uploads on its own CDN, rather than committing them. If such a file does get committed on a branch, **delete it as part of merging that branch** (or in a cleanup commit right after) — it must not survive into `main`. The only Markdown that belongs in the repo long-term is `README.md`, this file, `spyde/actions/README.md`, and `electron/tests/README.md`. (`.gitignore` covers the usual names, but it can't catch a new one — the rule is the thing.)

**The changelog is the exception, and it is not a working note:** `CHANGELOG.rst` plus the per-PR fragments under `upcoming_changes/*.rst` (towncrier — `upcoming_changes/README.rst` is the contract). These don't rot the way a plan document does, because a released entry describes a version that is now immutable — it is a record, not a claim about how the app currently works. **Add a fragment in any PR a user would notice**, and never delete either path when cleaning up a branch: the Prepare Release workflow consumes the fragments and writes the changelog.

## Code style

Code is writing. The science here is genuinely complex, so anything the code or
its comments add on top of that is cost paid by the next reader.

- **Follow the Zen of Python** (`import this`) strictly. Simple beats complex,
  explicit beats implicit, flat beats nested, readability counts.
- **Do not abbreviate names.** Write `navigation_index`, not `nav_idx`;
  `signal_tree`, not `st`. A name is the shortest documentation there is.
- **Prefer general terms to package-specific ones** in names and comments.
  Describe what is happening before reaching for a library's proper noun for it.
  "Reads the whole block from disk to return one frame" is readable by anyone;
  "the `CachedDaskArray` takes its synchronous branch" is readable only by
  someone with that package already in their head. Name the specific API when it
  is the subject, not as decoration.
- **Comments say why, once.** No narrative, no history, no restating the line
  below in prose. Contract docstrings are the place for behaviour a caller must
  know.

## Dependencies

Python deps are in `pyproject.toml`. Key non-PyPI deps from custom forks (check `pyproject.toml` for the exact pinned branch — they move):
- `hyperspy` → `github.com/cssfrancis/hyperspy@slice-integrate2` (the navigator `CachedDaskArray` / `get_index` cache logic lives here — see Live-Display §3)
- `rosettasciio` → `github.com/cssfrancis/rosettasciio@win32-binary-read`

Frontend deps (Electron, React, electron-vite, Playwright) are in `electron/package.json`.

**Electron is pinned EXACTLY, in `package.json` AND `electron/package.json`, and the two must agree.** This repo is an npm workspace, so `electron` hoists to the root and `electron/node_modules/electron` no longer exists; electron-builder then cannot read the installed version and falls back to parsing the spec, which it *refuses* if it is a range. That failure is release-only (nothing else runs electron-builder), so a widened range passes typecheck, unit and e2e and only breaks the tag build. Since Electron 42 there is also **no postinstall binary download** — `npm ci` leaves `node_modules/electron/dist` empty and the ~100 MB fetch happens lazily on the first `require('electron')`. CI pulls it explicitly (`npx install-electron`) so a failed download is an install error rather than a mystery test timeout.

Supported file extensions: `.hspy`, `.zspy`, `.mrc`, `.tif`, `.tiff`, `.de5`, `.csb` (see `SUPPORTED_EXTS` in `backend/_session_files.py`, re-exported from `session.py`). Adding one means updating that tuple **and** the Open-dialog `filters` in `electron/src/main/index.ts` (two places: the File menu and the `spyde:open-file` IPC handler) — the dialog will not offer an extension the tuple accepts.

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
- `plot_update_worker.py`: `PlotUpdateWorker` runs on a **plain daemon thread** (not a `QThread`); polls `dask.distributed.Future` objects for the **VI / signal-output** paths (the per-frame navigator read is now a synchronous cached read on the dispatcher — see Live-Display §3), reads the completed result (from the per-plot shm buffer where used), and **marshals the apply onto the asyncio main thread** via the `dispatch` callback (`loop.call_soon_threadsafe`) → `Session._on_plot_ready` / `_on_signal_ready`. It emits via `psygnal` signals, not Qt signals.

### Live Instrument Control (`spyde/live/`)
WIP modules for live microscope control: camera, stage, STEM, TEM, particle scanning, reference.

### Signals (`spyde/signals/`)
- `diffraction_vectors.py`: `SpyDEDiffractionVectors` — GPU-optimized CSR flat-buffer container for ragged diffraction vectors. Stores `(nav_x, nav_y, kx, ky, intensity)` with an offsets array (row-pointers). Key methods: `.at()`, `.kxy_at()`, `.count_map()`, `.to_dense()` (cached), `.to_pyxem()`, `.cluster()`, `.get_strain_maps()`.

### Vector orientation mapping (`spyde/actions/`)
- `vector_orientation.py`: CPU reference — per-pattern scipy-LM fit of pose `(θ, A, t)` where `v ≈ A·Rot(θ)·g_template + t`. `_residual` (soft-assign + no-match sink + strain-band penalty) is the cost both paths must agree on. Strain via polar decomposition of `M = A·Rot(θ)`.
- `vector_orientation_gpu.py`: **the production path** — fits the *whole field at once* on the GPU (batched torch + Adam), no dask, no per-pattern loop. The vectors and library are tiny, so the entire scan is one batched optimisation. `compute_vector_orientation_gpu()` is dispatched first when `gpu_available()`; CPU is the fallback. On SpEd Ag (13k patterns × 1081 templates) it runs in ~8s. See the GPU Computing section for the non-obvious constraints baked into it.

### CSB event-stream reader (`spyde/external/rsciio_csb/`, `spyde/actions/csb_*.py`)
A Direct Electron `.csb` file is a **sparse event stream, not a frame stack** — an image only exists once you pick a time window and integrate the events in it. So "opening" one means choosing an exposure, and the reader exists to make that choice cheap to change and cheap to scrub.
- `external/rsciio_csb/` is laid out as a **stock rsciio plugin** (`__init__` re-exporting `file_reader`, `_api.py`, `specifications.yaml`) and imports nothing from SpyDE — so it can be moved into RosettaSciIO as `rsciio/csb/` unchanged. `external/rosettasciio/csb_format.py` appends its spec to `rsciio.IO_PLUGINS` at runtime, which is what makes `hs.load("movie.csb", lazy=True)` resolve to it through ordinary extension dispatch. Delete both when the plugin lands upstream.
- **`lazy=True` returns one time plane per dask block**, each integrating only its own window. Load-bearing, not incidental — it is the same constraint as Live-Display §1: a graph that read a whole chunk to yield one frame cost the MRC path a 40× scrub regression.
- **The navigator is free**: per-plane total counts come from the block table alone, reading zero payload bytes. The reader publishes it as `original_metadata.csb.plane_counts` and `_session_files._reader_navigator` turns it into a calibrated navigator. Without it, building the overview means integrating the *entire movie* — the one thing this reader exists to avoid. It must carry the nav axis calibration, or a 1-D selector resolves every position to index 0 (`calibrated_nav_signal`).
- **The accumulator is NOT concurrency-safe** — `reset()` + `add_frames()` mutate one shared device/host buffer, so integrations take a lock rather than each getting an accumulator (at 8192² one buffer is 268 MB).
- `actions/csb_to_frames.py` re-cuts the stream at a different exposure; `actions/csb_raw_frame.py` swaps the selector's producer to show ONE raw camera frame (the dock's Point width "0"). Toolbar gating uses the `requires_original_metadata:` YAML key (`plot_control_toolbar._has_original_metadata`) — a format-specific gate, because a CSB movie is an ordinary `insitu` signal and `signal_type` cannot express it.

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
- **Load real-ish data the way a user does:** `backendAction(page, 'load_test_data_si_grains')` (bundled synthetic, crisp reciprocal lattice — find-vectors can actually detect spots) or `load_example {name}` (Examples menu; `zrnb_precipitate` etc., needs download+dask). `load_test_data*`/`load_test_vectors` are the fast bundled paths. `load_test_data_movie` (6 × 2048² uint16, 1 frame/chunk lazy, no file) is the synthetic in-situ movie for the GPU/tile path — asymmetric content (corner blocks, per-frame index band, fine checkerboard) makes mirror / stale-frame / blurry-tile bugs pixel-visible; `SPYDE_GPU_IMAGE=0` in `launchApp({env})` forces the Canvas2D reference render of the same scene (`gpu_image_parity.spec.ts` is the GPU-vs-CPU screenshot-parity + pan-direction + no-flash spec).
- **GPU render math belongs in anyplotlib's own suite first:** headless Playwright CAN run real WebGPU — `chromium.launch(channel="chromium", args=["--enable-unsafe-webgpu"])` (the default headless SHELL has no `navigator.gpu`; probe on a `file://` page — secure context) — and `page.screenshot()` captures the gpuCanvas there. The library-level GPU-vs-CPU parity suite is `anyplotlib/tests/test_plot2d/test_gpu_parity_playwright.py`; the SpyDE spec covers app integration (tile backend, binary transport, detail-tile round trip), not shader math.
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
- All navigator updates run serially on the single `_NavDispatcher` thread (latest-position-wins coalescing), so the hyperspy cache is never re-entered concurrently — no lock needed. See Live-Display §2 and §3.

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
  with the right chunks) beats any after-the-fact rechunk
  (419 s vs 184 s when a "better" rechunk misaligned the ghost blocks).
- Batch computes (`_do_compute_vectors`, orientation) keep the stored chunking
  when it's already usable rather than rechunking to a theoretical optimum.

### 2. Navigator updates = ONE serial dispatcher + latest-position-wins — NOT a lock, NOT per-update threads

The navigator→signal update path must be **non-blocking** AND **non-concurrent**.
Every selector update runs on a single dedicated daemon thread, `_NavDispatcher`
(`base_selector.py`): `submit(selector)` coalesces by `id(selector)` (a newer
position replaces the queued one), and the worker runs one `_run_update` at a time.
`_run_update` computes the frame synchronously (§3) and paints the returned ndarray;
superseded positions are dropped from the pending slot before they ever run, so no
in-flight staleness guard is needed for the nav read.

- **Why one thread, not per-update `threading.Timer`s:** concurrent updates raced
  hyperspy's `CachedDaskArray` block bookkeeping (`ValueError: (i, j) is not in
  list`). The serial dispatcher removes the concurrency at the source — so the
  cache is never re-entered and **no lock is needed**. (Earlier designs — an RLock
  held across the compute, then a generation counter `_gen_lock`/`_update_gen`/
  `is_stale_body`, then a `_cache_lock_ctx` — are all GONE. Don't reintroduce them.)
- `_run_update` commits `current_indices` **up front** then short-circuits an
  identical position, because the widget fires `pointer_move` + `pointer_up` =
  two submits per release.
- **Settle re-fire:** the `_settle_timer` (re)armed by `update_data` fires ONE
  `force=True` update once motion stops, so the resting frame computes even if the
  user holds still mid-drag. It has NO in-flight gate, so it cannot wedge. (Still
  useful: coalescing drops intermediate positions; the settle guarantees the final
  one runs.)
- **Do NOT add self-pacing or a buffer ring** (skip-while-in-flight, per-future
  slots). Both were tried; both made it worse — a wedged gate / an infinite ~6-frame
  re-emit loop. Serial dispatcher + latest-wins coalescing is it.
- Tests pin the contract: `test_navigator_race.py` (slow update must not block a
  newer one; stale result must not clobber), `test_nav_cached_read.py` (the unified
  read's single/region/dtype behaviour), `test_shm_read_robust.py`.

### 3. Navigator frame read — TIERED: synchronous cached read for CHEAP reads, async + cancellable for EXPENSIVE ones

A **cheap** read (single point / small dwell-in-chunk / small region) is computed
**synchronously, right on the `_NavDispatcher` thread**, in
`update_from_navigation_selection`: it calls
`current_signal._get_cache_dask_chunk(indices, get_result=True)` with the cache's
`_client` forced to `None` (or `_direct_read_frame` for a direct slice), and returns
the resulting **numpy array** directly. `Plot.update_data` paints an ndarray
immediately — no distributed Future, no shared-memory buffer, no `PlotUpdateWorker`
poll for the cheap nav path. This is the fast common case and stays fully synchronous.

- **One read path for points AND regions, through `spyde/array_cache/`.**
  `_direct_read_frame` calls `get_local_frame(child, signal, data, idx)` for BOTH a
  single crosshair point and an N-point integrating region; the resolved
  `FrameReader` decides granularity per backing. **Read granularity must match
  ACCESS granularity, not storage chunking** — that is the whole principle:
  `BinaryReader` (.mrc/.de5/raw) reads exactly the frames asked for and caches no
  block at all; `SourceArrayReader` (zarr/HDF5) and `LocalTransformReader` (derived
  views) must decode a whole nav-chunk to yield any frame, so they cache that block.
  When sub-chunked `.zspy` lands, that reader just stops populating blocks — no
  interface change.
- **A COMPRESSED chunk is ATOMIC — reading one frame costs the same as reading all
  of them.** Measured on a real `.zspy` with 537 MB (32×32 nav × 512² uint16)
  chunks: whole chunk 437 ms, an 8×16 sub-block 445 ms, **one frame 406 ms**. So
  "the chunk is too big to cache, read single frames instead" is a ~100× bug on
  compressed data (a ~50-frame ROI paid ~50 whole-chunk decodes: 2100–4250 ms vs
  50–100 ms on MRC). Sub-blocking is equally wrong — same decode, a fraction of
  the payload cached. Always cache the whole chunk and size the budget for it.
- **Derived views (rebin / signal-space crop) are computed PER FRAME in numpy from
  the PARENT's frame** — `readers/per_frame.py`, resolved in `nav_read` (it needs
  the signal tree, which `array_cache` otherwise stays agnostic of). Asking dask
  for one rebinned frame materialises the whole enclosing source chunk and re-runs
  the graph: **2403 ms**, versus **1.8 ms** once the parent's block is cached (the
  rebin itself is 1.8 ms; the parent's 435 ms decode is unavoidable and shared).
  `sum_points` sums the PARENT's frames and transforms ONCE — valid only because
  rebin and crop are LINEAR (529 → 69 ms on a 16×16 ROI). The gate is deliberately
  conservative: anything it can't reproduce exactly (unknown transform, non-integer
  rebin factor, changed nav grid, nav/time crop) returns None and falls back to
  `LocalTransformReader`. A silently wrong frame is far worse than a slow one.
- **Two caches, two granularities, both per-plot.** `Plot._array_cache`
  (`ArrayCache`, 256 MiB) holds decoded FRAMES; `Plot._block_cache` (`BlockCache`,
  **3 GiB**) holds decoded nav-CHUNK blocks shared by every reader that has one.
  The budget must hold the few STORAGE CHUNKS an ROI spans, and a chunk can be
  537 MB (see above), so a 16×16 ROI straddling 4 of them is ~2.1 GB.
  The block cache is why a region is cheap: an 8×8 ROI spans ≤4 nav-chunks, and a
  dragged ROI re-serves the ~75% of points it shares with the previous step from
  RAM. Both are cleared on node switch / close.
  **The block cache MUST be an LRU, not a one-entry memo** — that was the original
  bug: a single slot serves a dwell inside one chunk and nothing else, so crossing
  a boundary and returning re-decoded every move (measured 59× on `.zspy`, **1520×**
  on a rebinned view — a derived view re-runs the transform over the whole source
  chunk), and a region spanning several chunks thrashed on every drag step.
- **Verify a nav-read change with a REAL DRAG on LAZY data** (`region_drag_perf.spec.ts`).
  Two traps this caught that headless tests and a green suite did not: (1) the bundled
  `load_test_data_si_grains` is **EAGER**, so the whole cache path is skipped and the
  profile line says `eager` — the spec measures nothing; use
  `load_test_data_lazy_chunked` (lazy, 3×3 nav-chunk grid). (2) The tiered classifier
  can route the region async, which silently bypasses every cache below it. Run with
  `SPYDE_NAV_PROFILE=1` + `SPYDE_LOG_LEVEL=INFO` and check the `[NAV-PROFILE]` lines
  for `async-submit` and for `array-cache region` — a fast median with zero
  cache-served reads means the read never touched this machinery.
- **A region is NOT special-cased.** `get_local_frame` used to `return None` for
  `idx.ndim > 1`, so region integration bypassed all of the above and paid one dask
  `.compute()` per point, each materialising the enclosing nav-chunk to keep one
  frame — ~64 chunk decodes to read the ~4 chunks it actually spans. Measured on a
  64×64×256² 4D-STEM: **2850 ms → ~5 ms per drag step** (p95 ≤ 10 ms, i.e. inside
  the 16.7 ms/60 fps budget) on zspy, rebinned-zspy and MRC alike. Region means
  accumulate in float32 (exact: ≤256 frames × uint16 max < 2²⁴) and round back to an
  integer source dtype for parity with the old distributed
  `weighted_mean_round_from_sums`.
- **On LARGE frames a region integrate is ARITHMETIC-bound, not I/O-bound — every
  other bullet in this section models I/O and none of them explains it.** Measured on
  a 48 × 4096² uint16 `.mrc` movie, 16-frame ROI, one drag step: **660 ms, of which
  only ~166 ms was I/O** (16 × 10.4 ms memmap reads at 3.0 GiB/s). The other ~500 ms
  was plain numpy — `16 × acc(float32, 64 MiB) += frame(uint16, 32 MiB)` plus the
  `/n`, `rint`, `astype` tail, ~2.5 GiB of memory traffic through ONE core. So the
  two cache-shaped fixes both disappoint: sizing `ArrayCache` to hold the ROI gets
  660 → 460 ms, and routing the read async moves the 500 ms off the dispatcher
  without removing it. **Time the arithmetic with the frames already in RAM before
  reasoning about caches or sync-vs-async.** `spyde/array_cache/region_sum.py` fixes
  it two ways, together **660 → ~58 ms**:
  - **row-band threading** — numpy ufuncs release the GIL, so banding the frame and
    accumulating concurrently is ~6.6× (502 → 76 ms). Banding partitions PIXELS, so
    each pixel still sees its frames in the same order and the result is
    **bit-identical** — that is the whole contract (`test_region_integrator.py`
    asserts `array_equal`, never `allclose`). It saturates at ~8 threads: bandwidth-
    bound, not core-bound (48 cores gain nothing past that). `SPYDE_REGION_THREADS`.
  - **incremental ±1** — a slid ROI subtracts the leaving frames and adds the
    entering ones instead of re-summing all N. Enabled ONLY for an INTEGER source:
    every value is then exact in the accumulator and every partial sum stays inside
    float32's exact-integer range (leaving frames are subtracted BEFORE entering ones
    are added, so no intermediate can exceed the full-window bound). A FLOAT source
    cannot be subtracted back out without drift and always recomputes.
  - The full recompute streams ONE frame at a time into ONE accumulator with a
    per-frame band barrier — it never materialises an N-frame stack, so the
    Memory-Safety rule holds exactly as it did for the serial loop.
  - `ArrayCache` grows for a region (`ensure_budget_for`, ceiling 1 GiB, restored by
    `clear()`). The window alone is NOT enough headroom — the incremental path
    touches 2 frames per step, so the other N−2 age by one insert per step and the
    LEAVING frame gets evicted at almost exactly the step it is next needed
    (1.9 misses/step at N+1 slots vs 1.0 at N×1.5).
  - **`mean_frame` is NOT dispatcher-confined** — the expensive tier runs
    `get_local_frame` on a compute worker (`_submit_async_nav_read`) and
    `actions/overlay.py` warms off-thread — and the running sum is mutated IN PLACE.
    It is guarded by a **TRY-lock**: a contended caller recomputes into a private
    accumulator instead of waiting. Never make it a blocking lock; holding one across
    a compute is the retired `_cache_lock_ctx` wedge (§2).
  - An optional torch-CUDA accumulator (`region_sum_gpu.py`, `SPYDE_GPU_REGION=1`,
    **off by default**) is bit-identical (`torch.round` is round-half-to-even like
    `np.rint`; the cast back to the source dtype happens on-device). It is worth
    ~60 → 48 ms median / 102 → 60 ms p95 on the real path and is *slower* than the
    threaded CPU path when frames are already in RAM — it is PCIe-transfer-bound, not
    compute-bound. Measured on one Pascal card only, hence opt-in.
- **Focus demotes the block budget, it does NOT purge.** `set_active` →
  `Session._apply_focus_budgets` → `Plot.set_focused`: an unfocused plot shrinks to
  `UNFOCUSED_BUDGET_BYTES` (100 MiB) but KEEPS its working set, so clicking back is a
  numpy slice, not a cold re-decode. There is no idle timer (deliberately — it would
  need a thread and the caches are dispatcher-owned).
- **Do NOT route derived single-point reads async** — the transform recompute is only
  ~5–9 ms, while the async round-trip is 4 thread hops + 2 event-loop turns +
  ~15–40 ms; async was pure overhead there (the "everything is async-submits and
  slow" regression).
- **Tiered routing** (`_classify_nav_read`, called before the read): async
  (`_submit_async_nav_read` → `Session.compute_backend.submit_graph(lazy)` OFF the
  dispatcher, cancellable via `plot._nav_future`, paint from the done-callback) fires
  ONLY for reads that would genuinely FREEZE the navigator: a **large region**
  (`> REGION_ASYNC_FRAME_CAP=48` frames — a maxed 4D-STEM 16×16=256-frame region is
  ~0.5–0.9 s synchronously; a maxed movie 16-frame region is ~64 ms, stays sync) or
  `> REGION_BYTES_CAP`, or a **cold HUGE single frame on a CACHED signal**
  — where a region is sized by the **BYTES** of its **UNCACHED** frames
  (`_region_uncached_count`, a side-effect-free residency probe), NOT by point count
  and NOT by total extent. **This is load-bearing and was got wrong twice.** Async
  **bypasses the array cache**, and the async path never populates it — so any region
  routed async can *never warm*, and re-reads every frame on every drag step forever.
  A 4D-STEM ROI is many points of TINY frames (16×16 of 32 KB DPs = 256 points but
  only ~8 MB), so the old frame-COUNT cap (48) sent ordinary ROIs async: measured in
  the real app at **129 ms median / 380 ms p95 with ZERO cache-served reads**, vs
  **5.5 ms / 9.8 ms** once sized by bytes. The point cap now applies only when frames
  are individually large (`> COLD_FRAME_CAP`), i.e. the movie span it was tuned for.
  With no cache to probe it falls back to the total count (no regression)
  (`> COLD_FRAME_CAP` and not resident). The dispatcher returns `None` → `_run_update`
  skips the synchronous paint; the last good frame stays up until the async result lands
  (no flash). The async machinery is retained (audited) but fires far less often.
- **Region extent cap (Tier 0):** an integrating ROI/span is capped at
  `MAX_REGION_EXTENT_PER_DIM=16` nav positions PER navigation dimension, so a region
  read can never accidentally integrate a huge number of positions (worst case
  16×16=256 frames). The cap is enforced by the **anyplotlib widget itself**
  (`max_extent=`, ≥0.4.1): the ROI physically stops under the cursor mid-drag, the
  dragged edge pins and the opposite one stays put. SpyDE passes it in
  (`RectangleSelector` in image px; `LinearRegionSelector` in DATA units, so the cap
  is `MAX_REGION_EXTENT_PER_DIM * scale` and `_clamp_extent` re-derives it every
  update — the signal usually attaches AFTER the widget is built, so a cap fixed at
  construction is wrong for any calibrated axis).
  `_clamp_extent` remains as the fallback for geometry that never went through a
  drag (a programmatic set) — the widget's cap only applies to interactive drags.
  **Do not rely on it as the primary path**: it anchors on the lower edge, so it can move
  the edge the user is holding — which reads as the ROI jumping around. That, plus
  a default ROI size of half the image that then snapped down to the cap, was the
  "ROI is always huge and then clamps down / jumps" behaviour.
  See `test_region_extent_cap.py` and anyplotlib's `test_widget_max_extent.py`.
- **Paint is DECOUPLED from the read (slider stays live):** the read stays serial on the
  `_NavDispatcher`, but the PAINT (`set_data` → binary-uint8 → stdout, ~8–70 ms) runs on a
  separate serial newest-wins painter thread (`_NavPainter` in `plot.py`;
  `Plot.enqueue_paint` enqueues, `_run_update` calls it instead of painting inline). So a
  slow transport/decode doesn't block reading the NEXT slider position — the slider tracks
  the cursor and the display lags a frame behind + catches up (instead of the slider
  "catching" at slow frames). A frame superseded before it paints is DROPPED (newest-wins by
  `id(plot)`); stdout PLOTBIN writes stay serialized (one painter thread). This is NOT the
  retired READ self-pacing/buffer-ring — it's a newest-wins single-slot PAINT decouple. See
  `test_nav_paint_decouple.py`.
- **Large signal frames use anyplotlib TILE MODE (crisp zoom, no flash) — anyplotlib owns
  it, not SpyDE.** A big (≥1024 edge) signal frame is handed to anyplotlib's tiled display
  via `Plot._maybe_tile_signal` (in `_set_array`): it wraps the native frame in a
  `NumpyTileBackend` and calls `plot._plot2d.enable_tile(...)`. anyplotlib then owns the
  whole loop — it sends a downsampled OVERVIEW as the base (logical `image_width` = full
  size, `base_width` = overview px), reacts to its OWN debounced `view_changed`, and samples
  a hi-res detail tile of the visible region (1.25× over-fetch) at panel resolution. Each nav
  move swaps the frame via `update_tile_source(native)` — the zoom/subselection PERSIST while
  the pixels refresh (live-data contract). SpyDE no longer computes viewport crops or a base
  LOD for tiled frames. Why a tile not a full-res send: the ~400 ms full-res cost is
  dominated by normalising 16 M pixels (~100 ms) + the renderer receive; a tile processes
  only the ~1 MP the screen shows and scales to any image size. The default backend is
  numpy (a fast vectorised box-mean — GPU/torch measured not worth the dep for a ~1 MP
  visualisation sample, and numpy works on Mac too); a custom `TileBackend` (owns the source
  + sampling) can swap in for out-of-core/GPU. anyplotlib: `Plot2D.enable_tile` /
  `update_tile_source` / `set_detail` / `_detailUV`; `imshow(huge, tile="auto",
  integration_method="mean")` is the public API. See SpyDE `test_viewport_detail.py`;
  anyplotlib `test_tiled_imshow.py`, `test_tile_backend.py`, `test_detail_tile.py`.
- **Base frame SUBSAMPLES cheaply** (`data[::stride]`, ~1 ms): the base is just a
  thumbnail the GPU upscales for the zoomed-OUT overview — real detail always comes from
  the detail tile (which crops the NATIVE `current_data` on zoom, NOT this decimated copy).
  An area-MEAN of the base was ~70 ms on a uint16 4096² (the float32 cast of 16 M px alone
  is ~34 ms) and dominated the paint — not worth it for a thumbnail. `_lod_downsample` (a
  fast strided-add box-mean, ~40 ms, no full float cast) still exists but is used ONLY to
  cap an oversized detail-tile crop (deep zoom into a huge region). NB the LOD rebinds a
  LOCAL `data`; `current_data` stays the native frame so the tile crops native detail. See
  `test_lod_display.py`.

- **Why it's fast:** hyperspy's `CachedDaskArray` keeps the loaded chunk in a numpy
  cache. With **no client**, `get_index` takes its **synchronous** branch (cache the
  block, then slice/mean it in numpy) — ~1–2 ms dwell-in-chunk hits vs ~16 ms for the
  distributed round-trip (~100 ms cross-chunk). A 4D-STEM DP navigator dwells *within*
  a nav chunk → fast hits; an in-situ movie is 1 frame/chunk → each move is a small
  cold read of just that frame.
- **CRITICAL — `_client = None` alone is NOT enough** (this was a silent
  perf-only bug for a while). The fork's `CachedDaskArray.client` property, when
  `_client is None`, falls back to `dask.distributed.get_client()`, which returns the
  app's **process-global default `Client(cluster)`** from ANY thread (the
  `_NavDispatcher` thread included — it does NOT raise). So the pin was a no-op with a
  live cluster and every nav move still went distributed. `heavy_imports.
  _patch_cached_dask_client()` (applied in `ensure_heavy_imports`) removes that
  fallback so `_client = None` truly selects the synchronous branch. Tests never
  caught it because they run `SPYDE_NO_DASK=1` (no default client). See
  `test_cache_client_patch.py`.
- **Latest-wins — cheap vs expensive:** for a **cheap** read the serial dispatcher
  coalesces by `id(selector)`, so a superseded position is dropped from the pending slot
  before it ever runs — no in-flight compute to cancel. For an **expensive** read the
  dispatcher returns immediately after submitting, so a newer position must actively
  **cancel** the prior `plot._nav_future` (`fut.cancel()` in `_submit_async_nav_read`;
  a cheap paint also cancels+clears any stale `_nav_future` in `_run_update`). A queued
  future cancels cleanly; an already-running threaded one runs to completion but its
  callback no-ops via the `plot._nav_future is not fut` identity check (no lock, no
  generation counter — all on the one dispatcher thread). The settle re-fire still fires
  the (possibly expensive) authoritative read for the resting position at full res.
  See `test_nav_async_cancel.py`, `test_nav_rebin_no_block.py`, `test_nav_tiered_classify.py`.
- **dtype parity:** the synchronous branch returns float64 via `np.mean`. For an
  INTEGER source, round back to the frame dtype (no-op on a single point; correct
  rounded mean for an integrating region) so the DP shows the SAME uint16 frame +
  contrast the old distributed path did (`weighted_mean_round_from_sums`). This is
  the one non-obvious behaviour — see `test_nav_cached_read.py`.
- **Same slice semantics for everything lazy:** it's just "compute this lazy slice", so
  a cropped (`s.inav[..].isig[..]`) / rebinned / `.zspy` view scrubs through the same
  read — cheap ones synchronously, a heavy derived view via the async tier
  (`_build_nav_lazy_slice` builds the identical expression for both paths). A 1-D
  derived-view scrub also warms its **source** chunk off-thread (the movie prefetcher
  primed with the derived array) so the next re-decode reads warm pages.
- `write_shared_array` / `read_shared_array` and the shm buffer **remain** — but only
  for the **VI / progressive-navigator-fill** paths (`stream_progressive_to_plot`,
  `signal_tree._start_progressive_nav_compute`), NOT the per-frame nav read.
- Compute navigator (VI) display levels from the FULL accumulated finite data
  (robust 2–98% percentiles), with a final uniform repaint when the fill completes.

> **History (do NOT re-introduce):** the nav read USED to submit a distributed
> `get_inds` future + `write_shared_array` into a per-plot shm buffer, polled by
> `PlotUpdateWorker`. That path needed two hard-won fixes to avoid a DP frozen on a
> stale frame — pinning `CachedDaskArray._client` to the cluster and holding the
> get_inds future alive (`plot._inflight_getinds`) so distributed didn't
> release-key-cancel it. Both are now MOOT: the read is serial + blocking, so nothing
> is submitted to cancel and no future can be lost. The retired repros
> (`repro_cache_client_thread.py`, `repro_write_cancelled.py`) documented that dead
> machinery. Don't add a buffer ring or self-pacing (tried, both worse: wedged gate /
> infinite ~6-frame re-emit). NB: the safety came from **seriality + blocking**, not
> from which `get_index` branch runs — a serial blocking read is correct either way
> (the `_client=None` patch just makes it also FAST; see §3).
>
> **Tiered read (the expensive tier does NOT revive that machinery):** the async path
> for expensive reads uses ONE cancellable `submit_graph` frame future +
> `add_done_callback` → `_dispatch_to_main` paint. There is no `get_inds`, no
> `write_shared_array`, no shm ring, no `PlotUpdateWorker` poll, no self-pacing — a
> superseded read is cancelled by identity on the one dispatcher thread. The CHEAP tier
> is exactly the old synchronous read, untouched. So "expensive reads are async again"
> is restored WITHOUT the retired distributed per-frame path.

## GPU Computing

The hot paths (vector finding, vector orientation mapping) are GPU-accelerated. The stack present in the dev env: `torch` (+CUDA), `cupy`, `numba.cuda`. Guard every GPU path with an availability check (`torch.cuda.is_available()`) and keep a working CPU fallback — CI and many user machines have no GPU.

**Batch the whole problem, don't loop.** The vectors and template library for a 4D-STEM scan are only a few MB. The win is transferring everything to the GPU once and running *every* nav position in lockstep as one batched tensor op — not dask, not processes, not a per-pattern Python loop. `vector_orientation_gpu.py` is the model: pack all P patterns → `(P, …)` tensors, one batched coarse seed, one batched Adam refine, one vectorised decode.

**Avoid per-item Python loops around tiny kernels.** The original coarse seed looped templates × angles in Python (hundreds of thousands of tiny kernel launches) → **289s** for a realistic library. Rewriting it as a polar-histogram angular cross-correlation (one batched FFT, no Python loop) → **1.6s**. When a GPU step is slow, the cause is almost always a Python loop launching small kernels or a blown-up intermediate tensor — not the arithmetic. Reach for FFTs / matmuls / `scatter_add_` over explicit loops, and **chunk the batch dimension** to bound the largest intermediate (e.g. the `(P,T,n_a)` correlation is chunked over patterns) rather than materialising it whole (a full `(P,T,…)` tensor OOMs).

**Windows + torch-CUDA-autograd gotchas (hard-won):**
- `backward()` segfaults when run off the **main thread** under CUDA on Windows, the first time it runs on a thread whose autograd engine isn't initialised. Both OM handlers (`om_run`, `vom_run`) *do* dispatch the fit to a daemon worker via `run_on_worker`, so the two mitigations in the code are load-bearing: `warmup_autograd()` runs one trivial backward on the **dispatch thread** before the worker starts (CUDA-gated; a no-op on MPS/CPU), and the refine loop pins backward to its calling thread (next bullet). The fit takes an `on_yield` callback (pumps the event loop / flushes pending work) so the UI stays responsive. The CPU fallback (numpy/scipy) is thread-safe and needs neither.
- Pin backward to the calling thread with `torch.autograd.set_multithreading_enabled(False)` around the refine loop.
- Yield *inside* the step loop (every ~12 steps), not just per anneal stage — otherwise the window freezes for seconds and the progress bar appears stuck. Drive the progress label from the compute's own `progress(done,total)` callback; do not derive % from a lagging live-preview cell count.

**Mac + Apple-MPS is NOT thread-safe — every torch user takes ONE shared device lock (`spyde/device_lock.py`):**
- Two threads submitting to Metal at once corrupts the command encoder and raises an **uncatchable native SIGSEGV** — no `try`/`except` sees it, the backend process dies, and the user gets the "Analysis backend stopped" dialog. Observed faulting frames: `at::native::relu_mps_` → `MetalShaderLibrary::exec_unary_kernel`, and `at::native::zero_` → `fill_mps_kernel` → `[AGXG13GFamilyComputeContext setComputePipelineState:]`. Reproduces in ~30 lines: 4 threads running one small conv/ReLU net on MPS segfaults (or hangs) within a few dozen iterations; the same loop behind one lock runs clean.
- **`DEVICE_LOCK` in `spyde/device_lock.py` is THE lock** — a reentrant, process-wide `RLock`. Use `accelerator_lock(device)`, which is a **null context off MPS** (CUDA is thread-safe and its stream concurrency is a deliberate throughput win — never serialise it). Current holders: the neural batch / single-frame preview / calibration / cold model load (`models.infer.load_model`'s `.to(device)` + smoke-test forward), the torch NXCORR + DoG peak finders, and the batched vector-orientation fit.
- **A lock only works if EVERY participant takes it.** This crash existed because the lock was private to `find_vectors_torch` and only the neural *batch* and NXCORR paths took it — so the live preview (fires on navigator moves), the calibration, the cold model load, and the whole vector-orientation fit submitted unserialised. Adding a new torch call site without the lock silently re-opens it. `test_device_lock.py` pins the contract (shared-object identity + every entry point).
- **Long GPU work hands the device back at its yield points**, it does not hold the lock end-to-end: `compute_vector_orientation_gpu` wraps the caller's `on_yield` so each yield does `mps_sync()` → release → yield → re-acquire. So a concurrent preview waits one yield window (~12 refine steps), not the whole anneal. **Always `mps_sync()` before releasing** — handing off while kernels are still in flight lets the next thread submit into a live encoder, which is the race itself.
- The failure is **probabilistic per submission**, so a short stress test is NOT a regression gate: SpyDE's own preview path does heavy GIL-holding numpy work between short MPS bursts, and 6 threads × 40 frames with the lock disabled still completed cleanly here. It bites over a real multi-minute run (thousands of frames + a fit). Trust the lock-contract unit tests, not a crash-or-not stress run.

**Numerical traps that only show on real/strained data** (unit tests on uniform synthetic data won't catch these):
- *Rotation-branch ambiguity*: a centrosymmetric diffraction pattern is invariant under 180°, so the seed may pick θ≈±180° where an SPD-bounded stretch can't fit → garbage strain. Collapse the seed angle into `(−π/2, π/2]`.
- *Coarse-σ shrink bias*: at wide Gaussian σ the soft-assign cost is minimised by shrinking the template (spurious negative strain pinned at the cap). Fit a **rigid pose through the coarse stages and only release the strain DOF at the finest σ**, where the true strain is the global minimum.

## Benchmarking

**Always benchmark on a real dataset at real scale, end-to-end.** The canonical target is `pyxem.data.sped_ag()` — 208×64 = 13,312 patterns of 112×112 (a real 4D-STEM SpEd Ag scan). Synthetic 4×4 fixtures validate *correctness* but hide the costs that actually bite (per-Python-loop overhead, Vmax-padding blowup, library size). A method that's instant on 16 patterns can be minutes on 13k.

- Existing harnesses live in `spyde/tests/benchmark_*.py` (run directly with `python -m spyde.tests.benchmark_<name>`, not under pytest — they're slow). `benchmark_vector_orientation.py` builds the Ag library + sped_ag vectors and is the reference for the OM path.
- **Time each stage separately** (vector finding / library build / orientation fit) — the user's "it's slow" is usually one stage, and conflating them hides the real bottleneck. Print `progress(done, total)` with timestamps to see whether a stage is progressing or genuinely stuck.
- For GPU timing, `torch.cuda.synchronize()` before/after the timed region (kernels are async) and **discard the first run** (cold CUDA init + kernel JIT is a one-time ~5s cost; report the warm steady-state too).
- `torch`-CUDA work **segfaults under the pytest process on Windows** (a harness interaction, not a code bug — it runs fine in plain Python and in the real app). So: run GPU correctness tests in a **subprocess** that prints a JSON result (see `test_vector_orientation_gpu.py`), and `os._exit(0)` after printing to skip the torch/CUDA teardown crash. GUI tests that exercise the *wiring* should force the CPU path (`monkeypatch gpu_available → False`).

## Configuration Files

- `spyde/*.yaml`: loaded at import time in `spyde/__init__.py` (toolbar and metadata widget configs)

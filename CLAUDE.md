# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpyDE is a PySide6 GUI application for visualizing and analyzing electron microscopy data (TEM, STEM, Cryo EM, 4D STEM, EELS). It wraps HyperSpy and PyXEM with a custom MDI (Multiple Document Interface) window system, Dask-based parallel computing, and a signal transformation tree.

## Commands

**Install for development:**
```bash
pip install -e ".[tests]"
```

**Run the app:**
```bash
spyde
# or
python -m spyde
```

**Run tests:**
```bash
pytest spyde/tests/
```

**Run a single test file:**
```bash
pytest spyde/tests/test_actions.py
```

**Run a single test:**
```bash
pytest spyde/tests/test_actions.py::test_function_name
```

**Build distributable (Windows/macOS/Linux):**
```bash
pycrucible build
```
Release builds are triggered by pushing a version tag: `git tag v0.1.0 && git push origin v0.1.0`

## Distribution, Installer & Updates

Two delivery paths (see `DISTRIBUTION_PLAN.md`):
- **uv-managed installer (primary, Windows):** NSIS installer (`installer/spyde.nsi`)
  ships the project + bundled `uv` + a launcher stub to `%LOCALAPPDATA%\Programs\SpyDE`
  (per-user, no admin). `installer/launch.py` runs `uv sync --frozen --torch-backend=auto`
  on first run / after the lock changes, then `uv run main.py`. CI stages the payload
  with `tools/build_installer_payload.py`.
- **Portable single-exe (fallback):** the existing PyCrucible self-contained artifact.

Key modules:
- **Version is single-sourced** in `spyde/_version.py`; `pyproject.toml` reads it via
  setuptools dynamic version and `__init__` re-exports `__version__`. Bump that one line to release.
- `spyde/_build_info.py` carries version/sha/channel/build-date; CI stamps it via
  `tools/write_build_info.py` (dev checkout falls back to live git resolution).
- `spyde/updater.py`: `check()` against GitHub Releases (tolerant semver, skips
  pre-releases on `stable`); `is_uv_managed()` + `apply_uv_sync()` for in-place updates.
  Startup check fires 2.5s after launch (silent unless an update exists; disable via
  `SPYDE_NO_UPDATE_CHECK` or the QSettings toggle). Help → Check for Updates is the manual path.
- `spyde/gpu_setup.py`: `detect()`/`verify()`/`diagnostics()` and `ensure_backend()`
  (reinstalls the GPU-correct torch wheel via `uv pip install --torch-backend=auto`
  when an accelerator exists but torch is CPU-only). Help → GPU Status surfaces it;
  output must stay **cp1252-safe** (no unicode glyphs — they crash the Windows console).
- `tools/write_manifest.py` publishes `latest.json` with each release.

**Testing:** `spyde/tests/test_dist.py` covers version/build-info/updater-semver/gpu-setup
(offline, mocked network + uv). No Qt fixture needed.

## Dependencies

Key non-PyPI dependencies installed from custom forks:
- `hyperspy` → `github.com/cssfrancis/hyperspy@slice-integrate2`
- `rosettasciio` → `github.com/cssfrancis/rosettasciio@improve_mrc`

Supported file extensions: `.hspy`, `.mrc`, `.tif`, `.tiff`, `.de5`

## Architecture

### Entry Points
- `spyde/__main__.py` → `main()`: Creates `QApplication` + `MainWindow`
- `main.py` (root): PyCrucible launcher wrapper that calls `spyde.__main__.main()`

### MainWindow (`spyde/__main__.py`)
The central `QMainWindow` (~1500 lines). Contains:
- **MDI area** (`QMdiArea`): hosts `PlotWindow` subwindows
- **Dask** `LocalCluster` + `Client`: initialized in a background thread on startup
- **`PlotUpdateWorker`**: polls Dask futures on a background thread; emits results to update plots on the GUI thread
- **Dock widgets**: Plot Control (right), Instrument Control (left, for live microscopy)
- **`signal_trees`**: list of `BaseSignalTree` instances tracking all open datasets

### Signal Tree (`spyde/signal_tree.py`)
`BaseSignalTree` tracks a DAG of signal transformations. Each node is a HyperSpy `BaseSignal` with associated `Plot`(s). Non-breaking transformations (e.g. filtering, centering) update the current plot in-place; breaking transformations (e.g. azimuthal integration) create new branches. Users can navigate the tree to compare states.

### Drawing Layer (`spyde/drawing/`)
- `plots/plot.py`: `Plot` — wraps a pyqtgraph `ImageItem`/`PlotItem` with colormap, histogram, and navigator linkage
- `plots/plot_window.py`: `PlotWindow` — `QMdiSubWindow` containing one or more `Plot`s
- `plots/plot_states.py`: state machine governing how navigator and signal plots synchronize
- `plots/multiplot_manager.py`: manages multi-panel layouts
- `selectors/`: 1D and 2D ROI selectors (wrappers around pyqtgraph ROIs) used to slice the HyperSpy navigation space
- `toolbars/`: `PlotControlToolbar`, floating button bars, caret groups for UI controls
- `update_functions.py`: pure functions that compute what data to display given the current plot state

### Actions (`spyde/actions/`)
- `base.py`: base action framework; defines `NAVIGATOR_DRAG_MIME` for drag-and-drop between plots
- `hyper_signal_actions/`: YAML-configured HyperSpy signal transformation actions (filtering, FFT, azimuthal integration, etc.)
- `plot_actions/`: YAML-configured plot manipulation actions (colormap, zoom, export, etc.)
- `pyxem.py`: PyXEM-specific diffraction processing actions

### Compute Backend (`spyde/compute_backend.py`)
`ComputeBackend` provides a uniform `concurrent.futures.Future`-compatible interface over two modes:
- **Threaded** (default): `ThreadPoolExecutor` — low overhead, no Dask scheduler
- **Distributed**: wraps `dask.distributed.Client` futures via `_DistributedFutureAdapter`

Key methods: `.submit()`, `.compute()`, `.compute_chunks_progressive()` (streaming chunk results). Callers never import Dask directly; switch modes by swapping the backend instance.

### Workers (`spyde/workers/`)
- `plot_update_worker.py`: runs on a `QThread`; polls `dask.distributed.Future` objects and emits signals to update `Plot` objects on the Qt main thread

### Live Instrument Control (`spyde/live/`)
WIP widgets for live microscope control: camera, stage, STEM, TEM, particle scanning, reference. Housed in `ControlDockWidget`.

### Qt Utilities (`spyde/qt/`)
Shared Qt widgets and helpers. `spyde/qt/shared.py` contains `open_window()` and `create_data()` used by tests.

### Signals (`spyde/signals/`)
- `diffraction_vectors.py`: `SpyDEDiffractionVectors` — GPU-optimized CSR flat-buffer container for ragged diffraction vectors. Stores `(nav_x, nav_y, kx, ky, intensity)` with an offsets array (row-pointers). Key methods: `.at()`, `.kxy_at()`, `.count_map()`, `.to_dense()` (cached), `.to_pyxem()`, `.cluster()`, `.get_strain_maps()`.

### Vector orientation mapping (`spyde/actions/`)
- `vector_orientation.py`: CPU reference — per-pattern scipy-LM fit of pose `(θ, A, t)` where `v ≈ A·Rot(θ)·g_template + t`. `_residual` (soft-assign + no-match sink + strain-band penalty) is the cost both paths must agree on. Strain via polar decomposition of `M = A·Rot(θ)`.
- `vector_orientation_gpu.py`: **the production path** — fits the *whole field at once* on the GPU (batched torch + Adam), no dask, no per-pattern loop. The vectors and library are tiny, so the entire scan is one batched optimisation. `compute_vector_orientation_gpu()` is dispatched first when `gpu_available()`; CPU is the fallback. On SpEd Ag (13k patterns × 1081 templates) it runs in ~8s. See the GPU Computing section for the non-obvious constraints baked into it.

### External (`spyde/external/`)
- `pyqtgraph/histogram_widget.py`: customized `HistogramLUTWidget`/`HistogramLUTItem` extending pyqtgraph's histogram

## Testing

Tests use `pytest-qt` with a real `QApplication`. Fixtures in `spyde/conftest.py` create a `MainWindow` with synthetic data:

| Fixture | Data type | Expected subwindows |
|---|---|---|
| `window` | empty | 0 |
| `tem_2d_dataset` | 2D image | 1 |
| `insitu_tem_2d_dataset` | TEM + time | 2 |
| `stem_4d_dataset` | 4D STEM | 2 |
| `stem_5d_dataset` | 5D STEM | 3 |

Fixtures yield a dict with keys `window`, `mdi_area`, `subwindows`, `signal_trees`. On Linux CI, tests run under `xvfb-run` with `QT_QPA_PLATFORM=offscreen`.

The conftest uses a **session-scoped window** (`_session_window`) with a **per-test reset** (`_reset_window()`) that closes subwindows and clears tracking lists between tests — avoids paying Dask/Qt startup cost per test. `spyde/qt/shared.py` provides `open_window()`, `create_data()`, and `wait_until(predicate, timeout)` for test helpers. Tests are written as classes with methods (e.g. `class TestActions` → `def test_center_direct_beam`).

## Memory Safety Rule: Never Materialise Large Datasets

**`_do_compute_vectors` in `spyde/actions/find_vectors.py` must NEVER call `.compute()` or `.result()` on the full signal dataset.** Doing so loads hundreds of GB into RAM.

- For **numpy** data: the array is already in RAM — slice ghost-padded chunks directly.
- For **lazy dask** arrays: call `.compute()` on each small ghost-padded slice (`raw[py0:py1, px0:px1].compute()`) — never on `raw` itself.
- For **distributed Futures**: submit per-chunk tasks to the worker holding the future (Path B) — the worker does the slice locally, only small results return.

The 5D path slices by time index first (`raw[t, ...]`), producing a 4D chunk — use `sigma_tuple_2d_nav = (sigma, sigma, 0, 0)` for that blur, not the 5D `sigma_tuple`.

`test_find_vectors_memory.py` enforces this contract with 27 tests including a `patch.object` guard on `da.Array.compute` that raises if the full-dataset shape is ever computed.

## Thread Safety Constraints

- Qt UI updates must happen on the main thread. Workers (e.g. `PlotUpdateWorker`) communicate back via Qt signals, never direct method calls.
- Dask client startup is asynchronous. `MainWindow` uses a `wait loop + QApplication.processEvents()` before submitting compute work, not a blocking join.
- Navigator drag cancels stale futures before submitting new ones (frees workers immediately rather than queuing).

## GPU Computing

The hot paths (vector finding, vector orientation mapping) are GPU-accelerated. The stack present in the dev env: `torch` (+CUDA), `cupy`, `numba.cuda`. Guard every GPU path with an availability check (`torch.cuda.is_available()`) and keep a working CPU fallback — CI and many user machines have no GPU.

**Batch the whole problem, don't loop.** The vectors and template library for a 4D-STEM scan are only a few MB. The win is transferring everything to the GPU once and running *every* nav position in lockstep as one batched tensor op — not dask, not processes, not a per-pattern Python loop. `vector_orientation_gpu.py` is the model: pack all P patterns → `(P, …)` tensors, one batched coarse seed, one batched Adam refine, one vectorised decode.

**Avoid per-item Python loops around tiny kernels.** The original coarse seed looped templates × angles in Python (hundreds of thousands of tiny kernel launches) → **289s** for a realistic library. Rewriting it as a polar-histogram angular cross-correlation (one batched FFT, no Python loop) → **1.6s**. When a GPU step is slow, the cause is almost always a Python loop launching small kernels or a blown-up intermediate tensor — not the arithmetic. Reach for FFTs / matmuls / `scatter_add_` over explicit loops, and **chunk the batch dimension** to bound the largest intermediate (e.g. the `(P,T,n_a)` correlation is chunked over patterns) rather than materialising it whole (a full `(P,T,…)` tensor OOMs).

**Windows + torch-CUDA-autograd gotchas (hard-won):**
- `backward()` segfaults when run off the **main thread** under CUDA on Windows. The GPU orientation fit therefore runs **inline on the GUI/main thread** (it's only ~1-2s of compute) with an `on_yield=QApplication.processEvents` callback so the UI stays responsive; it is *not* offloaded to a worker thread. The CPU fallback (numpy/scipy) is thread-safe and *does* run on a worker.
- Pin backward to the calling thread with `torch.autograd.set_multithreading_enabled(False)` around the refine loop.
- Yield to Qt *inside* the step loop (every ~12 steps), not just per anneal stage — otherwise the window freezes for seconds and the progress bar appears stuck. Drive the progress label from the compute's own `progress(done,total)` callback; do not derive % from a lagging live-preview cell count.

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
- `spyde/actions/hyper_signal_actions/*.yaml`: declare available signal actions
- `spyde/actions/plot_actions/*.yaml`: declare available plot actions

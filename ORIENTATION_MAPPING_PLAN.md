# Orientation Mapping Workflow — Plan

> **Status (2026-06-12): built.**  Phase 1 + X/Y/Z selector + reduced 3D
> toggle + Integrate-ROI scatter are implemented and tested (133 tests
> green): `spyde/compute_dispatch.py` (shared dispatcher),
> `spyde/signals/orientation_map.py` (container + IPF helpers),
> `spyde/actions/orientation_compute.py` (batch compute),
> Run-page GUI in `spyde/actions/pyxem.py`.  Outstanding: in-app manual
> verification on real data, a `--orientation` benchmark stage, and adding
> `cupy-cuda12x`/`PyOpenGL` to optional extras in packaging.

Batch orientation mapping with the same workflow shape as Find Diffraction
Vectors: press Compute → chunked dispatch with the orientation map visibly
filling in → a lightweight result window (orientation map on the left, IPF
on the right) → a small standalone result that saves/loads without the raw
dataset.

## What already exists (reuse, don't rebuild)

| Piece | Where | Status |
|---|---|---|
| Phase/library setup UI (steps/tabs), gamma & radius params, IPF mask circles | `pyxem.py` `orientation_mapping` caret | done |
| Single-pattern matching (~5 ms) with precomputed polar templates | `_build_matching_cache`, `_get_best_fit_spots` | done |
| 2D IPF triangle + heatmap + marker widget (PlotWindow with swapped widget) | refine step, `_ipf_triangle_xy`, `_ipf_xy_for_rotation` | done |
| OrientationMap result unpacking (`(nav, n_best, 4)`: corr/index/mirror) | `_extract_orientation_outputs` | done |
| Chunked dispatcher, shm live buffers, lightweight result tree (`navigator_override`), save-button pattern, GPU lane plumbing | `find_vectors.py` | done — generalize, don't copy |
| All the dask pitfalls (meta=, `scheduler_info(n_workers=-1)`, storage-aligned chunks, held futures, loose restrictions, watchdog) | `benchmarks.md` | apply from day one |

## Architecture

### 1. Result container — `spyde/signals/orientation_map.py`

`SpyDEOrientationMap` (dataclass, analog of `SpyDEDiffractionVectors`):

- **Data**: `(nav_y, nav_x, n_best, 4)` float32 (pyxem layout: template index,
  correlation, in-plane angle, mirror) **plus** quaternions `(nav_y, nav_x,
  n_best, 4)` resolved from the library at compute time — so the container
  never needs the library again to answer orientation questions.
  Phase metadata (symmetry group, lattice, name) stored as a plain dict.
  Total for a 256² scan with n_best=5: ~5 MB — same "tiny standalone
  dataset" story as the vectors.
- **Methods**:
  - `ipf_color_map(direction="z") -> (ny, nx, 3) uint8` — orix
    `IPFColorKeyTSL`; the navigator image.
  - `correlation_map()`, `phase_map()` — alternate navigator images.
  - `ipf_xy(iy, ix, best=0)` / `ipf_xy_roi(sl) -> (M, 2)` — 2D stereographic
    coords (reuse `_ipf_xy_for_rotation`, vectorised for the ROI case).
  - `ipf_xyz(...)` — unit vectors in the fundamental sector for the 3D view.
  - `save(path)` / `load(path)` — compressed .npz + JSON phase dict
    (same pattern as vectors; orix Phase rebuilt from symmetry + lattice).

### 2. Batch compute — `_do_compute_orientations`

The find_vectors compute skeleton **minus ghost zones** (patterns are
independent → plain `map_blocks`, no overlap, no depth clamping):

- Chunk fn: polar-transform each pattern, correlate against the cached
  templates (`_build_matching_cache` output), keep n_best. The cache
  (templates, a few–tens of MB) is `client.scatter(..., broadcast=True)`'d
  once, not re-pickled per task.
- Chunking: adopt storage chunks as-is (alignment lesson); no ghost → no
  rechunk in the common case at all.
- Live display: shm buffer shaped `(nav_y, nav_x, 3)` float32 holding the
  IPF **RGB** so the user watches the actual orientation map paint in
  (passthrough `map_blocks` stage computes chunk RGB from the chunk's best
  rotations and writes its slab — same `_count_chunk_to_shm` pattern; needs
  the small generalisation of `ensure_live_buffer` to 3-channel).
- Dispatch: reuse `_dispatch_chunks_gpu_aware` (factor it + the worker-split
  helpers out of `find_vectors.py` into `spyde/compute_dispatch.py`; both
  actions import it). Phase 1 runs CPU-only on all lanes (the matcher is
  numpy); the GPU lane slot stays wired so a CuPy polar-correlation kernel
  can drop in later (Phase 3).
- All known guards: `meta=` + zero-size short-circuit, held futures +
  compaction (n_best slice is already compact — no NaN padding needed),
  stall watchdog, lane refresh, `scheduler_info(n_workers=-1)`.

Throughput expectation: ~5 ms/pattern × 65k patterns / 44 threads ≈ 8 s of
compute for the 60 GB benchmark dataset + IO ⇒ should land near virtual-
image time (~25–40 s), *not* find_vectors time — worth verifying with a new
`--orientation` stage in `benchmark_workflow.py`.

### 3. Result window pair

Created on Compute click exactly like find_vectors (instrumented `ui:`
timings, shm poll timer, Stop wiring, lightweight tree via
`navigator_override` — no signal copy, no navigator recompute):

- **Navigator (left): the orientation map.** During compute: poll the RGB
  shm buffer (whole-map refresh). On completion: `ipf_color_map()` becomes
  the navigator image/data. (Stretch: toolbar toggle navigator between
  IPF-Z / correlation / phase maps.)
- **Signal (right): the IPF.** Not an image plot — after `add_signal`, swap
  the signal PlotWindow's widget for the IPF widget (the refine step
  already does this swap for its floating IPF window; reuse that code as a
  shared `IPFWidget`): triangle outline + labels + a marker.
  Navigation hook = find_vectors `_read_position` pattern: crosshair move →
  `ipf_xy(iy, ix)` → marker moves. n_best > 1: best as solid marker,
  runners-up as faded markers sized by correlation (stretch).

### 4. 3D toggle (right toolbar action on the result plot)

- New toggle action ("3D IPF") on the result signal plot's toolbar
  (plot-action registration like existing toolbar actions).
- 3D view: `pyqtgraph.opengl.GLViewWidget` in a QStackedWidget with the 2D
  IPF; the toggle flips pages. Content: unit-sphere wireframe +
  fundamental-sector outline (vertices from orix
  `phase.point_group.fundamental_sector`) + `GLScatterPlotItem` for the
  point(s). Same navigation callback drives both views.
- Dependency: `pyqtgraph.opengl` requires PyOpenGL — add as optional
  extra; if missing, the toggle is disabled with a tooltip.

### 5. Integrate mode → all points in the ROI

When the navigator selector is switched to Integrate (region selection,
existing selector2d machinery):

- The navigation hook receives slices instead of a point →
  `ipf_xy_roi(sl)` / `ipf_xyz_roi(sl)` → scatter **all** orientations in
  the ROI on the current view (2D triangle or 3D sphere).
- Subsample above ~50k points (uniform stride) to keep the views
  interactive; optionally weight point alpha by correlation score.
- This is pure container + widget work — no recompute on drag, target
  <30 ms per ROI move (same throttle timer as the VVI caret).

### 6. Caret integration

Add a "Compute Map" step to the existing orientation_mapping caret (after
refine): Compute button, n_best spinbox, progress/status label, Stop via
the result nav window (find_vectors pattern), and "Save orientations…"
enabled on completion.

## Testing plan (first, per workflow)

1. **Container unit tests** (`test_orientation_map.py`): save/load
   roundtrip; `ipf_xy` against orix reference values for known rotations;
   ROI gather correctness; color map shapes/dtypes.
2. **Compute correctness**: synthetic dataset generated *from the library
   itself* (render patterns for known rotations) → map must recover the
   ground-truth orientations (within angular tolerance); chunked result ==
   single-chunk result; the memory-contract guard from
   `test_find_vectors_memory.py` adapted (no full-dataset compute; spy on
   `da.Array.compute`).
3. **Dispatcher reuse tests**: factoring `compute_dispatch.py` out must keep
   all 92 existing vector tests green — the refactor lands before any new
   feature code.
4. **GUI smoke** (where stable on Windows): caret builds, IPF widget swap,
   integrate-callback math (pure-function tested headless).
5. **Benchmark**: `--orientation` stage in `benchmark_workflow.py`; run on
   the 60 GB dataset; record in `benchmarks.md` next to vimage/vectors.

## Phasing

- **Phase 1**: dispatcher factor-out (tests green) → container + tests →
  batch compute + live RGB map → result window with 2D IPF point →
  save/load. *Usable end-to-end.*
- **Phase 2**: Integrate ROI scatter; 3D toggle.
- **Phase 3**: CuPy polar correlation for the GPU lane; navigator toggles
  (correlation/phase); n-best runner-up display; load-from-npz entry point
  (shared with the parked vectors loader).

## Decisions (Carter, 2026-06-12)

1. **n_best**: default 5, user-adjustable spinbox (more is better for IPF
   heat maps, at the cost of container size — range 1–20).
2. **Multi-phase**: in scope for Phase 1.  Signal view shows **one IPF per
   phase side by side**; the matched phase's IPF carries the marker.
3. **IPF reference direction**: X/Y/Z selector on the right toolbar.
4. **3D view**: *reduced* 3D (minimal sphere/sector), whose purpose is to
   show the **in-plane rotation the 2D IPF cannot**: the marker is an
   oriented glyph (point + tangent direction) encoding the full orientation,
   not just the beam direction.
5. **Compute placement**: new Compute button/step inside the existing
   orientation-mapping caret.

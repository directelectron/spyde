# Drift Correction & Particle Segmentation — Plan

Two features that share one spine: **align the movie → segment each frame →
measure → link into trajectories → show it moving.**

Reference points: [quantem `imaging/drift.py`][q] (scan model, Bézier knots,
non-rigid optimisation, directional Fourier merge) and [ParticleSpy][p]
(segmentation parameters, measured properties, trainable feature stack,
time-series linking). We take their **algorithms and parameter vocabulary** and
re-implement on the SpyDE stack: torch instead of skimage/scipy loops, batched
instead of per-frame, CSR instead of Python object lists.

[q]: https://github.com/electronmicroscopy/quantem/blob/dev/src/quantem/imaging/drift.py
[p]: https://github.com/ePSIC-DLS/particlespy

Prior art in this repo, in order of how much it saves us:
`signals/diffraction_vectors.py` (ragged CSR container — particles per frame is
the *same* data shape), `find_vectors/` (compute-package split + progressive,
cancellable result window — the interaction model this feature copies wholesale),
`vector_orientation_gpu.py` (batched-torch playbook + the Windows autograd traps),
`vector_overlay.py` (interactive per-point overlay with add/remove/state colouring),
`report/vectors_embed.py` (the interactive HTML explorer — §C4),
`models/registry.py` (upgradeable model delivery), `actions/README.md` (the
action contract — read before writing any action).

---

## 0. Decisions locked before writing code

### 0.1 Scale is the primary design driver

**Thousands of frames at 2048²–4096². Tens of GB.** This is not a movie that
fits in RAM, and it invalidates the obvious implementation of nearly every
stage. Consequences, applied everywhere below:

- **Nothing materialises the stack** — not the solver, not the aligned copy,
  not the segmenter, not the label images. CLAUDE.md's memory-safety rule,
  extended to two new packages.
- The drift solver **streams**: read frame → FFT → correlate → discard. Its
  output is an `(N, 2)` array, not a stack.
- The corrected movie is a **lazy view**, never a written copy (§0.7).
- Particle masks need a storage budget, not an assumption (§0.5).
- Every long-running stage is **progressive and cancellable** (§0.8).

### 0.2 One spine, three interchangeable mask sources

Every segmentation path produces the **same** intermediate — a per-frame
foreground probability map — and then shares one downstream stage:

```
 frame ──► [ classical | scribble | prompt ] ──► probability/mask
                                                      │
                              instance split (distance-transform watershed)
                                                      │
                                    measure (regionprops → physical units)
                                                      │
                                    SpyDEParticles (CSR, per frame)
                                                      │
                                    link (Hungarian) ──► tracks + events
```

The three engines are **not** competing implementations to choose between —
they are three ways to fill the first box, and they compose (§0.4). The
instance-split + measure stage is written once and is the only code that ever
touches skimage.

### 0.3 Package layout and dependencies

| Submodule | Contents | Extra |
|---|---|---|
| `spyde/drift/` | warp, scan/deformation models, translation/affine/non-rigid solvers | — (core) |
| `spyde/particles/` | feature stack, three engines, instance split, measure, track | — (core) |
| `spyde/signals/particles.py` | `SpyDEParticles` CSR container | — (core) |
| `spyde/actions/drift_action.py` | the Drift wizard | — |
| `spyde/actions/particles_action.py` | the Segment Particles wizard | — |
| `spyde/actions/particle_overlay.py` | live outlines / trails / editing | — |

torch, scikit-image, scipy and scikit-learn are **already core deps**, so the
compute needs no new dependency at all. Three new pieces of infrastructure:

- **`anyplotlib` gains a brush/freehand widget** (B0). An upstream contribution
  and a version bump, exactly like `max_extent` was added for the ROI cap. It
  gates B3's UI, so it goes first.
- **EfficientSAM-Ti as an optional download, not a dependency.** No
  `pip install` extra: the checkpoint is pulled on demand from Hugging Face
  through the **existing** `spyde/models/registry.py`, the same path SpotUNet
  weights already use. So the Prompt tab is present but shows a one-click
  "Download model (≈40 MB)" until the weights are cached in `~/.spyde/models`.
  Inference is plain torch, which is already core — there is nothing to install.
  This is strictly better than a `requires_package` extra: no reinstall, no
  environment surgery, and it upgrades without re-releasing SpyDE.
- **A generic table component** (`DataTable.tsx`) rather than a particle-specific
  grid. Columns, sort, selection and virtualised rows are all data-agnostic; the
  particle dock is its first consumer. Other features want it too — the vector
  list, the event log, per-phase OM statistics, the fit component list.

`spyde/models/` is the *neural disk detector* registry (RELEASE_0_3_0_PLAN
§0.1). We **extend** it to be architecture-generic rather than adding a second
registry — one model-delivery mechanism, one cache dir, one refresh path. See B4.

### 0.4 Scribble vs prompt: what each is for

| | Scribble classifier | Promptable (SAM-family) |
|---|---|---|
| Prior | none — learns *your* data | natural RGB photography |
| On EM contrast | adapts in ~30 s of painting | out of distribution; over-segments film texture, merges touching particles |
| Coverage | **dense** — every particle, every frame | one object per prompt |
| Cost per frame | one conv stack + a tiny head | one image embedding (expensive) + cheap per-prompt decode |
| Best at | batch measurement over a movie | "what is *this* thing", zero labels |

**The scribble classifier is the workhorse; the prompt model is the bootstrap
and the single-particle tool.** They compose, and the composition is the point:

> Click four particles with the prompt model. Those masks — plus their dilated
> surroundings as background — become the scribble classifier's training
> labels. Train (seconds), apply to all N frames. **No painting at all**, and
> the dense result is adapted to the data rather than to COCO.

So `prompt.py` exposes `masks_to_labels()` and the wizard's Prompt tab has a
**"Use as training labels"** button that hands off to the Scribble tab. That
handoff is a first-class feature, not a convenience.

The classical pipeline is always available (no model, no training) and is where
the shared instance-split lives, so it is never dead code.

### 0.5 Storage: `SpyDEParticles`, a CSR container — with a mask budget

Particles-per-frame is a **ragged per-navigation-position collection** — the
exact shape `SpyDEDiffractionVectors` already solves. Mirror it rather than
inventing a second pattern, and rather than ParticleSpy's list-of-Python-objects,
which does not survive 3000 frames.

```
flat_buffer : (N_total, K) float32
    columns: [t, label, y, x, area, equiv_diam, major, minor, perimeter,
              circularity, eccentricity, solidity, intensity_mean,
              intensity_max, intensity_std, background, bbox_y0, bbox_x0,
              bbox_y1, bbox_x1, track_id]
    sorted by t (outermost nav dim first) — same convention as vectors
nav_offsets : [t_offsets (n_t+1,)]  — CSR row pointers, O(1) frame slice
```

**Do the arithmetic before choosing a mask representation.** At the stated
scale — 3000 frames × ~500 particles = 1.5M particles:

| | per particle | total |
|---|---|---|
| property row (21 × float32) | 84 B | **126 MB** — fine, keep in RAM |
| bbox bitmap (packed 64² crop) | 512 B | **770 MB** — too big |
| contour polygon (~40 pts × 2 × f32) | 320 B | **480 MB** — still too big |
| contour, quantised int16 + RLE | ~80 B | **120 MB** — acceptable |

So: **properties always in RAM; masks are stored as quantised contours and are
optional.** A `store_masks=False` measure-only mode is the default for very long
movies, and a full-frame label image is never stored at any setting — a 4096²
int32 label image is 64 MB *per frame*. `render_frame(t)` reconstructs the
overlay for the displayed frame from contours on demand, the direct analogue of
`SpyDEDiffractionVectors.render_frame`.

What mirroring the vectors container buys for free:

- `count_map_series()` → **particle count vs time**, the movie's navigator trace.
- `render_frame(t)` → the overlay for the currently displayed frame.
- `open_result_tree` → result window opens **early and fills progressively**.
- `save()` / `load()` → particles are a standalone saveable mini-dataset.

### 0.6 Segmentation produces a NEW TREE, not an attribute

**This resolves the 4D-STEM attachment ambiguity, and it is the right shape
generally.** A segmentation is not a property of the source movie — it is a
derived dataset computed *from* it, exactly like a strain map or an orientation
map. So `seg_run` spawns a new `SignalTree` through the existing
`commit.open_result_tree` door:

```
ParticleTree
  root signal : lazy LABEL MOVIE — same nav/signal shape as the source,
                each frame rendered from the stored contours on demand
  tree.particles    : SpyDEParticles          (the CSR store)
  tree.source_node  : the node it was computed from
  tree.nav_map      : source nav indices → particle frame index (identity for a
                      movie; the parent's nav grid for a 4D-STEM virtual image)
  navigator         : stacked count(t) / size(t) / event lanes  (C2)
```

Why this is better than `source_tree.particles = …`:

- **It answers Wave D by construction.** Particles found on a 4D-STEM virtual
  image record the node they came from and the nav positions each particle
  covers, so "mean DP for this particle" is a slice of `source_node`'s parent
  rather than a guess about which grid the coordinates belong to.
- **The label movie is scrubbable, saveable and reportable** — it behaves like
  any other dataset, so the report builder, the movie editor and `save`/`load`
  need no special case.
- **Re-segmenting doesn't destroy the previous result.** Two parameter choices
  are two sibling trees you can compare, which is what the signal tree is for.
- Downstream actions gate on the *particle tree's* own type, so
  `requires_particles` is a plain signal-type check rather than a hunt up the
  parent chain.

The label movie is **lazy and never materialised** — the contours are the truth,
`render_frame(t)` paints one frame on demand (§0.5).

### 0.7 The drift-corrected movie is a LAZY VIEW, not a copy

- Solver output is a small **`DriftModel`** — `(N, 2)` shifts, or the warp
  parameters for the non-rigid case — stashed on the tree as `tree.drift` and
  stamped into provenance.
- The corrected node is a lazy per-frame warp added with
  `tree.add_transformation(...)`, composing with the existing
  `LocalTransformReader`, so nav scrubbing works unchanged on day one.

**Deferred, explicitly gated on review:** `array_cache/readers/per_frame.py`
has a deliberately conservative allowlist (`_rebin_fn`, `_crop_fn`) for
transforms it can reproduce exactly per frame. A rigid shift qualifies, and
adding it would make a drift-corrected movie scrub at parent-frame speed (the
CLAUDE.md rebin numbers: 2403 ms → 1.8 ms once the parent block is cached).
**This is a signal-tree read-path change and does not happen autonomously** —
benchmark, proposal, case-by-case review, then implement. Wave A ships without it.

### 0.8 The interaction contract: preview → progressive → cancellable

Locked, and it applies to both wizards:

1. **Scrub and see the result on a single frame before committing to a run.**
   Tuning happens on the displayed frame at full interactivity; nothing batch
   runs until asked.
2. **The run is progressive** — the result window opens immediately and the
   count-vs-time trace fills as frames complete, like the Find Vectors count map.
3. **The run is cancellable** — registered via `BaseSignalTree.register_cancel`
   so closing the tree or hitting stop kills in-flight compute.
4. **Target: minutes, not hours.** ~20–100 frames/s for segment + measure.

### 0.9 Detection sensitivity is the priority, not instance splitting

Given hundreds of frequently-touching particles, the instinct is to pour effort
into watershed splitting. **The steer is the opposite: faint, small, low-contrast
particles must be found at all** — missing a particle's first appearance destroys
the nucleation event, which is the most interesting thing in the movie.

Consequences:

- The learned classifier is the primary path, not threshold tuning — a
  threshold that catches a 3σ particle at t=0 is not the one that works at
  t=end, and no single global threshold spans a nucleation sequence.
- Expose **one sensitivity control**, not independent knobs. Sensitivity and
  separation trade off against each other (a threshold loose enough to catch
  faint particles also merges neighbours), so it should be one axis the user
  moves with live feedback, with splitting parameters secondary.
- Small-object detection needs the feature stack's fine scales — do not
  downsample frames for speed without a documented sensitivity measurement.

### 0.10 Scope: four data shapes, one code path

| Shape | Path |
|---|---|
| In-situ movie (1-D time nav + 2-D signal) | primary |
| **5-D STEM reduced to a virtual image** | **identical — it *is* an in-situ movie** (time, nav_y, nav_x) |
| Single 2-D image | same segment+measure, no nav to fill, no tracking |
| 4D-STEM virtual image | same, plus the diffraction linkage in Wave D |

Tracking is meaningless without a time axis, so it gates on `_signal_type ==
"insitu"` exactly as Play/Fast-Forward already does.

### 0.11 Baselines to measure BEFORE writing solver code

Every number is a *baseline to beat*, recorded in `benchmarks.md`:

| Stage | Reference | Target data |
|---|---|---|
| Rigid alignment | `skimage.registration.phase_cross_correlation` per pair | `load_test_data_movie`, then a real long `.mrc` |
| Non-rigid | quantem's scipy L-BFGS-B, per row | synthetic known warp |
| Feature stack | skimage filters on CPU | 2048² and 4096² frame |
| Scribble train+apply | sklearn RandomForest on the same stack | 2048², ~5k labelled px |
| Prompt latency | reference SAM predictor | 1024² embed + point decode |
| Full segment+measure | ParticleSpy `particle_analysis_series` | `pdcusi_insitu` |

Two traps that have already been paid for here:

- **Page cache.** Any drift benchmark reading a movie just written or just read
  is measuring RAM. Use `purge_cache` from `benchmark_mrc_access_patterns.py`
  and release live `np.memmap`/hyperspy handles first.
- **Time the arithmetic in RAM before reasoning about I/O.** The ROI-integrate
  work found 500 ms of 660 ms was single-threaded numpy, not disk. Warping a
  4096² frame has the same profile.

---

## Wave A — Drift correction (`spyde/drift/`)

**A1. Batched rigid translation.** FFTs in bounded frame-batches with
`torch.fft.rfft2`; cross-power spectrum against a **running Fourier average**
reference — quantem's stabilisation, and the locked default, so one bad frame
can't become the reference. Subpixel refinement by matrix-multiply DFT
upsampling (Guizar-Sicairos), default `upsample_factor=8` — a small dense matmul,
not a padded inverse FFT. `min_image_shift` / `max_image_shift` bounds as quantem
has them. Optional Hann/Tukey apodisation: a movie with a moving feature at the
edge otherwise correlates on the frame border. Output: `(N, 2)` shifts.

Sequential and fixed-reference modes exist behind the same solver but are not
the default.

**A2. Two warp parameterisations, one solver.** *Both physical causes are real*
— scan distortion happens, and so does local sample drift with no global
reference — so the model is selectable rather than assumed:

- **Scan-knot model** (quantem): fast/slow scan unit vectors from
  `scan_direction_degrees`, knots `[2, slow_dim, n_knots]`, Bézier basis → per-row
  coordinate map. `number_knots=1` is the documented default for uniform scan
  distortion. Correct when the distortion is a scanning artifact.
- **Dense control-point field**: a coarse 2-D grid of displacement control
  points with bending-energy regularisation — standard free-form deformation.
  Correct when the *sample* deforms, and when parts of the field move
  independently of each other.

They share the warp (A3), the solver (A5), and the regularisation machinery;
only the parameter→displacement map differs. Building both is roughly 30% more
than building one.

**A3. Differentiable KDE warp.** quantem resamples with a KDE scatter; in torch
that is `index_put_(accumulate=True)` over the four bilinear neighbours plus a
weight image for normalisation. Being differentiable is the whole point — it is
what makes A5 an autograd problem instead of a finite-difference one. Also
produces the coverage map A6 and the NaN-padding need.

**A4. Affine / linear drift search.** Grid search over parameter perturbations
(quantem: `num_tests=9` circular pattern, `step=0.01`, optional refine at finer
step). Batched — all 9 candidates in one call, not looped.

**A5. Non-rigid optimisation — the rewrite that justifies the port.** quantem
minimises with scipy L-BFGS-B, per row, with numerical gradients. Because A3 is
differentiable, we take an **analytic autograd gradient and optimise all rows
simultaneously** with `torch.optim.LBFGS` (or Adam + anneal, matching
`vector_orientation_gpu.py`'s proven schedule). Keep quantem's regularisation,
which is doing real work: Gaussian smoothing of residuals after polynomial trend
removal (`regularization_sigma_px=16`), max-displacement clamping, step damping
(0.8) over the 8 outer iterations.

> **Windows + torch-CUDA autograd — both mitigations are load-bearing.** The fit
> dispatches to a daemon worker via `run_on_worker`, and `backward()` segfaults
> the first time it runs on a thread whose autograd engine is uninitialised. So:
> `warmup_autograd()` on the dispatch thread before the worker starts, and
> `torch.autograd.set_multithreading_enabled(False)` around the refine loop.
> Yield every ~12 steps *inside* the loop or the window freezes; drive progress
> from the compute's own `progress(done, total)`.

**A6. Scan-rotation merge — LAST, and lowest priority.** No orthogonal-scan data
in hand, so this is built against synthetic ground truth or deferred entirely.
The spec is quantem's: directional Fourier filtering (bounded sine-squared
sigmoid on angle, `filter_midpoint`), cosine-tapered edge blending
(`mask_edge_blend=8`), `weight_thresh=0.1` coverage masking.

**A7. Edges: NaN pad + coverage mask.** Full frame size retained; uncovered
pixels are NaN and a per-frame coverage mask records validity. Nothing is
silently cropped or filled with invented data. **Downstream contract:**
segmentation must respect the coverage mask, or it will find "particles" in the
padding — this is the single most likely integration bug and gets an explicit test.

**A8. The Drift wizard** (`drift_` prefix, staged per `registry.py`).

```
drift_open        mount → current-frame preview + empty shift trace
drift_set_method  Rigid | Rigid+Affine | Non-rigid (scan-knot | dense field)
drift_tune        debounced re-tune of upsample / max-shift / regularisation
drift_run         solve on a worker, progressive shift-trace fill, cancellable
drift_commit      add the corrected node to the tree
drift_close       teardown
```

**Explicit only** — nothing runs on load.

**A narrow caret plus a separate Drift Check window.** The caret holds the method
tabs, parameters, progress bar and Commit. The check window holds what needs
pixels: **before/after sum images side by side** — an aligned stack sums sharp, a
misaligned one blurs — with the **shift-vs-time trace** (dx, dy) beneath, both
filling incrementally as the solve progresses. A 240 px caret cannot show a sum
image at a size where sharpness is judgeable, which is the whole point of the
check; and the window closes once you trust the result. Registered via
`own_window` + `figure_registry.keep_alive`, since a bare-figure window is not a
Plot (`actions/README.md` §6).

Declare the parameter schema as a `parameters` classattr **and** in
`registry._WIZARD_SCHEMAS` (`test_wizard_schemas.py` catches drift).

**A9. The corrected node.** `tree.drift = DriftModel(...)`, lazy per-frame warp
node, provenance stamped. Trajectories can then be reported in the **lab frame**
or the **sample frame** by adding or subtracting the model — the correct way to
answer "did the particle move, or did the stage?"

---

## Wave B — Particle segmentation (`spyde/particles/`)

**B0. anyplotlib brush primitive (upstream, first).** A freehand/brush widget:
`pointer_down` starts a stroke, `pointer_move` extends it with client-side
rendering, `pointer_up` emits the accumulated polyline. Client-side accumulation
is required — a per-move round trip over the PLOTAPP line protocol is 60
messages/s competing with the nav painter thread. Brush size and eraser are
widget properties. **Shift+drag paints**, leaving pan/zoom on the bare drag —
matching the existing Shift+click convention in Center Zero Beam, and avoiding a
mode that can be got stuck in.

**Controls live on a floating strip next to the plot, not in the caret.** While
painting you are looking at the image, so the things switched most often — active
class, brush size, eraser — sit under the cursor. The strip is colour swatches
only; class *names* and pixel counts stay in the caret (B7), which is the
authoritative list. Same component shape as the movie editor's overlay strip.

**B1. Classical baseline + the shared instance spine.** Port of ParticleSpy's
`segptcls.process`, keeping their parameter names so the caret is recognisable:
`rb_kernel` (rolling-ball via white-tophat), `gaussian`, `invert`, `threshold`
(otsu / mean / minimum / yen / isodata / li / local / local-otsu / niblack /
sauvola), `watershed`, `watershed_erosion`, `min_size`, `local_size`.

**One deliberate deviation from ParticleSpy's parameters:** their `watershed_size`
filters watershed markers by AREA. That works for their thresholded-distance
markers but is actively harmful with local-maximum markers — a 3×3 particle's
marker is ONE pixel, so any area floor erases it and the particle vanishes without
changing anything a user would notice. It is replaced by **`min_separation`** (the
minimum distance between markers) plus **`marker_smooth`**. No marker is ever
dropped for being small; that is §0.9 applied to the splitting step, and
`test_particles_core.py::test_a_tiny_particle_survives_the_watershed` pins it.

The second half — distance transform → `peak_local_max` → `watershed` →
`clear_border` → `remove_small_objects` — is factored out as
`split_instances(prob, params)` and is **shared by all three engines**. The only
module that imports skimage.

**B2. Torch feature stack** (`features.py`). ParticleSpy's `trainable_parameters`
set, batched on GPU: gaussian, difference-of-gaussians, median / min / max (via
`unfold`), Sobel, Hessian eigenvalues, Laplacian, membrane projections. One
`(C, H, W)` tensor per frame, separable convolutions where the kernel allows,
one pass over the frame rather than one pass per feature. **Fine scales are
mandatory** — they are what detects small faint particles (§0.9).

**B3. Scribble classifier** (`scribble.py`) — the workhorse.

- **Multi-class, user-defined**: add/name/colour classes freely — particle /
  carbon film / vacuum / beam-stop. A softmax head costs nothing over a sigmoid,
  and in EM "background" is genuinely two or three different things that a
  binary split confuses.
- **Labels accumulate across frames** into one training set, keyed by frame
  index, with a small list showing which frames carry labels so they can be
  revisited or cleared. Scrub to t=400, paint the newly-nucleated particle,
  retrain — earlier strokes are still there.
- Head is a **small torch MLP** (one hidden layer, class-balanced loss) so the
  whole path is one framework on the GPU.
  `sklearn.ensemble.RandomForestClassifier` on the identical feature stack stays
  in the test suite as the **parity reference** — it is what ParticleSpy and
  ilastik use, and agreement on the same labels is the acceptance gate.
- **Hard interaction budget: train + apply to the visible frame under ~1 s.**
  Train on labelled pixels only (thousands, not millions); apply to the visible
  frame only while tuning.

**B4. Promptable segmentation** (`prompt.py`) — the bootstrap.

| Interaction | Widget | Prompt |
|---|---|---|
| click a particle | crosshair / point | point |
| drag a box | `add_rectangle_widget` | box |
| draw around it | `add_polygon_widget` | polygon → bbox + dense mask hint |

> **Trap:** anyplotlib 2-D widgets report `cx/cy/x/y/w/h` in **image pixels**,
> no scale or offset applied. Building prompt coordinates in physical units gives
> an empty prompt on any calibrated axis — `masks.py::_signal_k_grids` documents
> exactly this bug class.

**EfficientSAM-Ti**, delivered as an **optional Hugging Face download** through
the **existing** `spyde/models/registry.py`, generalised from SpotUNet-specific
hyperparams to an `arch` field with a per-arch builder. Not a `pip` extra: the
Prompt tab always exists and shows a one-click "Download model (≈40 MB)" until
the weights are cached in `~/.spyde/models`. Inference is plain torch, already a
core dep, so nothing is installed and the model upgrades without re-releasing
SpyDE. The registry's manifest merge, offline fallback and refresh path all apply
unchanged.

Cost model: the image **embedding** is expensive (~100s of ms), the per-prompt
decode is milliseconds. Embed the current frame once on entering the Prompt tab,
cache by frame index, and every subsequent click is interactive.

`masks_to_labels()` + **"Use as training labels"** is the handoff to B3 (§0.4).

**B5. Measurement** (`measure.py`). ParticleSpy's property set, calibrated from
the signal axes so results are in nm/nm² not pixels: area, centroid, equivalent
circular diameter, major/minor axis, perimeter, circularity, eccentricity,
solidity, mean/max/std intensity, local background (mean over the dilated
boundary ring), bbox and bbox area. Vectorised via `regionprops_table`, never a
Python loop over regions.

**B6. `SpyDEParticles`** (§0.5) — container, `render_frame`, `count_map_series`,
`save`/`load`, `to_dataframe()` for CSV export, contour-based optional masks.

**B7. The Segment Particles wizard** (`seg_` prefix) — a **wide 2-column caret**
(330 px, using WizardShell's existing `width` override). Three tabs — Classical /
Scribble / Prompt — over a shared Preview and Run, honouring §0.8: preview on the
displayed frame, progressive fill, cancellable.

```
┌ Segment Particles ──────────────────────────── ✕ ┐
│ [Classical] [Scribble] [Prompt]                  │
├── params ──────────────┬── feedback ─────────────┤
│ Sensitivity   ▓▓▓▓▓░░  │ SIZE nm²  (histogram)   │
│ Min size          24   │ ▁▃▅█▆▃▁                 │
│ Split touching    on   │ 212 found · median 96   │
│ Store masks      off   ├── classes ──────────────┤
│                        │ ■ particle      1,204px │
│                        │ ■ carbon film     840px │
│                        │ ■ vacuum          612px │
│                        │ + add class             │
├──────────────────────────────────────────────────┤
│ [Train]  [Run all]                               │
│ 3 frames labelled · 4 classes                    │
└──────────────────────────────────────────────────┘
```

The right column is **feedback and class management**: the live size histogram
re-renders as sensitivity is dragged (so you see the distribution shift rather
than guessing), and below it the authoritative class list with per-class labelled-
pixel counts — which is how you notice a class is under-trained. One
**sensitivity** control front and centre (§0.9); splitting parameters secondary.
The floating strip (B0) mirrors the class colours for in-canvas switching.

**B8. Results surfaces — all four.**

1. **The particle tree** (§0.6) — the label movie, with stacked count(t) /
   size(t) / event-lane navigators. Saves, loads and reports like any dataset.
2. **Bottom dock, full width**, built on a new **generic `DataTable.tsx`**
   (§0.3) — one row per particle or per track, sortable by any column, click a
   row to highlight on the frame, with an Events tab beside the Table tab.
   Full width buys columns that don't truncate, and it reuses the Log panel's
   slot and show/hide behaviour. Costs vertical space, which is the accepted
   trade.
3. **Overlay property readout** — click a particle, see its properties in a
   popover on the frame.
4. **Histogram / scatter window** — ParticleSpy's `plot()`: histogram of one
   property, scatter of two, coloured by cluster. Reuses existing 1-D panels, and
   answers the size-distribution question that is usually the real one.

**B9. Overlay and editing** (`particle_overlay.py`, modelled on
`vector_overlay.py`).

- **Filled translucent, coloured by track ID.**
- **Labels on selection and hover only** — the selected particle gets its outline,
  ID and a property readout; everything else stays a plain fill. Quiet at 500
  particles, precise on demand. Always-on IDs were rejected: legible in a 9-particle
  mockup, a wall of numbers at real density.
- **Selection** three ways: click the particle on the frame (nearest-centroid
  hit test, as the strain reference-pixel picker does), click a table row, and
  **rubber-band a region** for bulk operations.
- **Manual correction in v1: delete + merge + split.** Delete drops a row; merge
  unions two masks and re-measures; split cuts along a drawn line and re-measures.
  Edits are recorded on the tree so a re-run does not silently discard them, and
  they are stamped into provenance so a corrected result is still reproducible.

---

## Wave C — Tracking, events, and showing motion

**C1. The linker** (`track.py`). Frame-to-frame assignment by
`scipy.optimize.linear_sum_assignment` on a cost matrix of centroid distance
(gated by `max_dist`), optionally weighted by property similarity — trackpy's
model, no new dependency. `memory=k` lets a track survive k frames of
non-detection.

To report trajectories in the sample frame, use **`DriftModel.to_sample_frame`,
which ADDS the shifts**. An earlier draft of this plan said "raw minus
`tree.drift`" — that is `to_lab_frame`, the inverse, and implementing it literally
*doubles* the drift instead of removing it. That is precisely the sign trap
`drift/model.py`'s docstring exists to warn about, and it got into the plan anyway;
always go through the named methods rather than writing the arithmetic out.

**Units:** `max_dist` is in the particles' **calibrated units** (nm), because the
measured centroids are. `DriftModel` is in **pixels**. One function owns that
seam (`sample_frame_positions`) — do not convert anywhere else. The caret must
render the unit label beside the field or users will type pixels.

**C2. Events on the navigator — the headline.** **birth** (nucleation), **death**
(dissolution), **merge** (coalescence), **split** (fragmentation).

Birth and death fall out of the assignment for free — an unmatched detection is a
birth, an unmatched track is a death. **Merge and split do NOT**, and an earlier
draft of this plan wrongly implied they did: a one-to-one assignment cannot
represent two-to-one, so they need an explicit post-pass with its own radius
parameter and its own failure modes. The implemented rule: a track that ends at
`t-1` whose last position is within a merge radius of a track present in *both*
`t-1` and `t` is a merge rather than a death (split is the mirror), with the radius
defaulting to the particle's own `equiv_diameter` so a large body absorbs from
further away. Merge/split **replace** the death/birth they explain, which gives the
checkable invariant `Δcount = births + splits − deaths − merges` — verified to hold
on every frame of the fixture.

They surface three ways:

- **Three stacked navigator lanes** on the particle tree — `count(t)`,
  `mean size(t)`, and a dedicated **event lane**. Each curve keeps its own
  y-scale (count and nm² have nothing in common, so a dual axis would squash
  one and invite misreading), and events get their own row with a colour per
  type — green birth, red death, mauve merge, yellow split — so you click
  straight to a nucleation instead of inferring it from a kink. Tallest option
  of the three considered; the stacked-navigator machinery already exists for
  in-situ playback.
- **A flash/badge on the frame** at a particle's birth or death frame during
  playback, so events are visible while watching rather than only on a timeline.
- **An Events tab in the table dock** — time, type, particle IDs — click to jump.
  Same `DataTable` component as the particle list (§0.3), so it is a column
  config, not a new panel. The rigorous path for counting events, and directly
  exportable.

**C3. Motion display.**

- **Trails: fading line + head dot.** The last N frames of each track fade with
  age, and a bright dot marks the current position so "now" is unambiguous —
  a bare fade leaves direction inferable only by close inspection of one track.
  N adjustable. One extra primitive per track.
- **Kymograph (v1), user-sortable** — tracks × time as an image, one row per
  track, coloured by a chosen property. Row order is a control, not a constant,
  matching the table dock's mental model: **by birth time** the leading edge's
  slope *is* the nucleation rate; **by lifetime** short-lived noise detections
  separate visually from real particles (segmentation QC); **by max area** ranks
  by size. Re-renders per sort, which is cheap on a tracks × time image.
- **Property vs time** — any measured property for the selected track(s),
  overlaid on a 1-D plot.
- **Committed maps** — `commit_result_tree` with count(t), total area(t), mean
  diameter(t) as chip views.

**C4. The deliverable: an interactive particles explorer in the report.**

Modelled directly on the existing **vectors HTML explorer**
(`report/vectors_embed.py`): anyplotlib widgets + the touch shim, self-contained,
scrollable through the movie **without embedding a huge video**. A collaborator
opens the report and scrubs the particles themselves.

> **Trap carried over from the vectors embed:** read recompute pixels via the
> **overlay** canvas — buffer assertions lie. That memory was paid for once.

Secondary: **movie-editor overlays with explicit labels** — outlines, trails and
text callouts on the live movie figure, so an exported movie shows *and names*
what changed. The movie editor already composites anyplotlib widgets on the live
plot, so this is wiring rather than new export code.

---

## Wave D — 4D-STEM linkage

Particles segmented on a virtual image live in the **nav space of the parent 4D
dataset**, not in the virtual image's own space. §0.6 resolves this: the particle
tree records `source_node` and `nav_map`, so the relationship is stored rather
than inferred. Three things then follow — all requested, all cheap given the CSR
store:

1. **Mean diffraction pattern per particle** — select a particle, get the
   averaged DP over its nav positions. Phase or orientation *per particle*. The
   thing no other tool does.
2. **Particle masks feed the existing vector/orientation actions** — a mask
   becomes a nav-space region, so Find Vectors or orientation mapping runs per
   particle rather than over the whole scan. The entire downstream pipeline is
   reused unchanged.
3. **Per-particle statistics from any nav-space map** — mean/std of strain,
   orientation or any virtual image within each particle. Turns every existing
   map into a per-particle table.

---

## Wave 0 — cross-cutting, do first

- **anyplotlib brush widget** (B0) — upstream, gates B3's UI.
- **Generic `DataTable.tsx`** (§0.3) — columns, sort, selection, virtualised rows,
  data-agnostic. The particle dock and the event log are its first two consumers.
- **`requires_particles` gate key** on `tree.particles`, mirroring
  `requires_vectors`. Both filter paths.
- **`lifecycle.wait_for_particles`**, mirroring `wait_for_vectors` — the
  find-vectors timing trap reproduces exactly here. `seg_run` opens its window
  early and attaches `tree.particles` only on **finalise**; any downstream action
  firing in that gap sees `None` on a tree that gets it seconds later. Gate on
  the real completion signal, never a sleep.
- **`spyde/models/registry.py` generalisation** to an `arch` field (B4).
- **Synthetic test data**: `load_test_data_particles` — a bundled in-situ movie
  with *known* ground truth: N particles of known radii, known per-frame rigid
  drift, one nucleation at a known t, one dissolution, one merge, plus faint
  low-contrast particles to exercise §0.9. Asymmetric and crisp per the
  `si_grains`/`movie` precedent so a mirrored overlay or an off-by-one frame is
  pixel-visible. **This is the acceptance gate for Waves A–C** — it makes every
  stage checkable against a number rather than a screenshot.
- **Docs**: one guide in `guides/`, dataset wired into the Examples menu.

---

## Traps — each previously paid for

1. **Never materialise the movie.** No `.compute()` / `.result()` on the full
   dataset in `spyde/drift/` or `spyde/particles/`. Mirror
   `test_find_vectors_memory.py`'s `patch.object` guard on `da.Array.compute`.
2. **NaN padding + coverage mask** (A7) — segmentation that ignores the mask will
   find particles in the padding. Explicit test.
3. **MPS device lock.** Every new torch call site takes `accelerator_lock(device)`
   — feature stack, scribble train and apply, prompt embedder *and* decoder,
   drift FFTs, non-rigid fit. A lock only works if every participant takes it;
   the last crash of this class existed because one path skipped it. Long fits
   hand the device back at yield points with `mps_sync()` **before** release.
   Extend `test_device_lock.py`.
4. **Windows CUDA autograd off the main thread** (A5) — `warmup_autograd()` +
   `set_multithreading_enabled(False)`. The failure is a hard segfault.
5. **Don't touch the nav read path.** The per-frame shift reader is a proposal
   gated on benchmark + review (§0.7), not part of Wave A.
6. **anyplotlib 2-D widget coords are image pixels** — prompts, scribbles and
   overlays build in pixel space, never `pixel*scale + offset`.
7. **Thread marshal.** Solvers and segmenters run on `run_on_worker`; plots,
   figures and IPC state are touched only on the asyncio main thread via
   `session._dispatch_to_main`. `emit_status`/`emit_error` are safe anywhere.
8. **Latest-wins.** Scribble re-tune, prompt clicks and drift re-solves can be
   superseded — `bump_generation`/`is_current`; teardown bumps first.
9. **StrictMode double-mount.** Both wizards use `useWizardLifecycle` *and* the
   backend generation guard; a double-fire test each.
10. **Report embed:** read recompute pixels via the overlay canvas (C4).
11. **GPU tests in a subprocess** on Windows; wiring tests force CPU with
    `monkeypatch gpu_available → False`.
12. **Page-cache benchmarks** (§0.11) — purge and release handles, or you time RAM.

---

## Acceptance gates

Where we replace a reference implementation, **numerical parity against it is the
test** — not "it converged", not "the screenshot looks right".

| Stage | Gate |
|---|---|
| A1 rigid | Recovers a synthetic shift to < 0.1 px; agrees with `phase_cross_correlation` on real frames |
| A5 non-rigid | Recovers a synthetic known warp, both parameterisations; residual ≤ quantem's on the same input |
| A7 edges | No particle is ever detected in NaN-padded region |
| B1 classical | Matches `segptcls.process` labels on identical parameters |
| B3 scribble | Matches the sklearn RandomForest reference on identical labels/features (IoU threshold) |
| B3 sensitivity | Detects the faint low-contrast particles in `load_test_data_particles` — the §0.9 priority made measurable |
| B5 measure | Matches `regionprops` on synthetic shapes; physical units correct under non-unit axis scale |
| C1 link | Recovers the known trajectories, births and deaths **exactly** (measured: mover is one track over all frames, 0.175 px max error; one birth at frame 8; one death at frame 16) |
| C2 merge | Exactly one merge, involving both merge-pair tracks, coinciding with the frame the count drops. **NOT** asserted at a fixed frame — see below |
| C1 gate | No link ever exceeds `max_dist`, at any `max_dist`; and `Δcount = births + splits − deaths − merges` on every frame |
| Scale | A full run on thousands of 2048² frames completes in minutes without exceeding a fixed memory ceiling |
| Perf | Every stage beats its §0.11 baseline, recorded in `benchmarks.md` |

### Why the merge frame is not a fixed number

The merge event's *frame* is a property of the **segmenter**, not the linker, and an
acceptance gate demanding a specific frame would only be satisfiable by tuning the
segmenter. Measured on the fixture, whose two merging discs first make geometric
contact at frame 14:

| | frame |
|---|---|
| discs' centres within `r₁+r₂` (geometric truth) | 14 |
| segmenter resolves ONE region, watershed **on** | 18 |
| segmenter resolves ONE region, watershed **off** | 12 |

Watershed exists to split touching particles, so it correctly keeps them apart for
four frames *past* first contact; without it the soft-edged tails connect two frames
*before*. The truth is bracketed, `12 < 14 < 18`, on one boolean. So the gate asserts
the invariants that genuinely belong to the linker — exactly one merge, the right two
tracks, coinciding with the count drop — and a separate test asserts the bracketing,
which is what proves the offset is a segmentation property rather than a linker bug.

## Verification standard

A green pytest run and a clean `tsc` are **not** verification for anything that
adds windows, draws overlays or wires renderer↔backend — and these waves are
almost entirely that. Each ships with a Playwright spec on
`electron/tests/_harness.cjs`, driven with `load_test_data_particles`, with
screenshots that were actually looked at. Specifically pixel-checked: brush
strokes land where the cursor was, outlines sit **on** the particles rather than
mirrored or offset by one frame, trails follow motion in the right direction, and
the navigator curves' kinks align with the known ground-truth event frames.

---

## Build order

Locked: **A1 → B → C**, then A2–A6, then D. Rigid alignment is the only part of
Wave A that Wave B depends on, and getting a working segment-and-track loop in
front of a real dataset early is worth more than a finished drift feature — the
segmentation parameters and the overlay are where the unknowns are. A6 (scan
merge) is last regardless, since there is no orthogonal-scan data to validate it.

| Step | Contents | Gate |
|---|---|---|
| **0** | brush widget, `DataTable`, `requires_particles`, `wait_for_particles`, `load_test_data_particles` | fixture ground truth is exact |
| **1** | A1 rigid translation + warp + `DriftModel` + corrected node | < 0.1 px on synthetic shift |
| **2** | B6 `SpyDEParticles`, B5 measure, B1 classical + instance split | parity vs `regionprops` / `segptcls` |
| **3** | B2 feature stack, B3 scribble, B7 wizard, B8 surfaces, B9 overlay | parity vs RandomForest; faint particles found |
| **4** | C1 linker, C2 events + navigator lanes, C3 trails/kymograph, C4 report embed | fixture trajectories and events exact |
| **5** | B4 EfficientSAM-Ti prompt + label handoff | one-click mask on a real particle |
| **6** | A2–A5 scan-knot + dense-field non-rigid | synthetic known warp |
| **7** | Wave D 4D-STEM linkage; A6 scan merge | per-particle mean DP correct |

---

## Resolved — the four questions this plan opened with

1. **Sequencing** → A1 → B → C, then A2–A6, then D. See Build order above.
2. **Prompt model** → **EfficientSAM-Ti**, as an optional Hugging Face download
   through the existing model registry rather than a `pip` extra (§0.3, B4).
3. **Wave D attachment** → segmentation spawns a **new tree** carrying
   `source_node` + `nav_map`, so the parent relationship is recorded rather than
   inferred (§0.6). The B6 container can be frozen.
4. **Table component** → **generic `DataTable.tsx`** (§0.3); the particle dock
   and the event log are its first two consumers.

No blocking questions remain. What is deliberately deferred, and why:

- The **per-frame shift reader** in `array_cache/readers/per_frame.py` — a
  signal-tree read-path change, so benchmark → proposal → case-by-case review
  before it is written (§0.7).
- **A6 scan-rotation merge** — no orthogonal-scan data to validate against.

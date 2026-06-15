# Vector Orientation Mapping — Design & Implementation Plan

Status: proposed (2026-06-14)
Owner: Carter Francis
Related: `ORIENTATION_MAPPING_PLAN.md` (dense path), `DIFFRACTION_VECTORS_PLAN.md`

## 1. Goal

Add orientation mapping that operates on **detected diffraction vectors**
(`SpyDEDiffractionVectors`) instead of dense diffraction patterns. Because a
pattern is now ~tens of `(kx, ky, intensity)` peaks rather than a full image,
the match is a *point-set* comparison. That smallness opens up:

- **Affine flexibility per pattern** — global scale (miscalibration), elastic
  strain (a bounded 2×2 affine), and in-plane rotation, fit jointly with the
  orientation.
- **Whole-field coupling** — neighboring patterns are nearly identical, so each
  pattern's fit can warm-start from its neighbor, making the search both faster
  and more robust than independent per-pattern matching.

Design steers from the user (2026-06-14):
- Output = **orientation + affine/strain** (not orientation alone).
- Coarse metric starts from pyxem #899 (sparse normalized cross-correlation on
  polar-unwrapped spots) but should be faster.
- **Affine in the inner loop**, bounded to ~5% strain.
- Use crystal symmetry / Friedel symmetry as a fit constraint and QC signal.
- Fit the **entire 4D dataset**, not strictly pattern-by-pattern.
- Reuse the dense OM library generation.
- Symmetry-as-constraint can come after a working rigid+affine core.

## 2. Background: what already exists

- `spyde/signals/diffraction_vectors.py` — `SpyDEDiffractionVectors`, CSR flat
  buffer `(nav_x, nav_y, kx, ky, time, intensity)`; `kxy_at`, `intensities_at`,
  `at`, `build_kdtree`, `n_time`, `nav_shape`.
- `spyde/actions/orientation_compute.py` — dense template matching: library
  generation (`generate_library_from_phases`), polar cache (`build_matching_cache`),
  `template_tables`, `resolve_quaternions`, batch driver `_do_compute_orientations`,
  live IPF-RGB shm preview.
- `spyde/signals/orientation_map.py` — `SpyDEOrientationMap`: quats/corr/phase/
  mirror per pattern, `ipf_color_map`, `correlation_map`, save/load. **No raw
  data or library needed after compute.**
- `spyde/actions/pyxem.py::_get_best_fit_spots` — already extracts a template's
  spots via `sim.get_simulation(lib_idx)` → `coords_dv.data[:, :2]` (kx,ky) +
  `coords_dv.intensity`, and applies the pyxem mirror/rotate/negate convention.
  **This is the exact primitive the vector matcher consumes.**
- Toolbar gating: YAML `requires_vectors: True` + `PlotState.rebuild_toolbars()`
  already wired (used by Vector Virtual Imaging). The vector-OM action reuses it.

## 3. Matching core (`spyde/actions/vector_orientation.py`, headless)

No Qt; unit-testable; importable on dask workers (mirrors `orientation_compute.py`).

### 3.1 Library → spot tables (reuse dense generation)
- Call `generate_library_from_phases(...)` (dense path) → `Simulation2D`.
- Precompute **once**: for each template orientation, its spot list
  `(kx, ky, I)` in calibrated Å⁻¹ (via `sim.get_simulation`, applying the same
  mirror/rotate/negate convention as `_get_best_fit_spots`). Store as a ragged
  set + a flat `(M_total, 4)` buffer `[tmpl_idx, kx, ky, I]` for vectorized ops.
- Also store the polar form `(r, θ, I)` per template for the coarse NCC.

### 3.2 Coarse orientation search (sparse NCC; pyxem #899 start)
- Measured vectors → polar `(r, θ, I)`.
- For each template, circular cross-correlation over θ between measured and
  template angular intensity profiles (binned in θ, weighted by I and by radial
  agreement). Argmax over θ gives the in-plane angle; peak value is the score.
- Vectorized over all templates in one pass (NumPy now; CuPy later — #899 wants
  GPU, no lazy). Keep top-N candidates.

### 3.3 Affine inner-loop refine (bounded ICP)
For each of the top-N coarse candidates:
1. Rotate template spots by the coarse in-plane angle.
2. **Pair**: nearest measured vector to each template spot within a tolerance
   `tol_data` (default ≈ kernel radius). KD-tree per template built once at
   library time; query the (few) measured vectors. Unmatched template spots
   (missing reflections) and unmatched vectors (spurious peaks) are dropped from
   the pair set but penalize the score.
3. **Solve** weighted least squares for the 2×2 affine `A` + translation `t`
   (beam-center residual) mapping template→measured over the matched pairs,
   weights = `min(I_template, I_measured)`.
4. **Bound**: decompose `A = R·(I+ε)` (polar); clamp symmetric strain `ε`
   eigenvalues to `[-0.05, 0.05]` (configurable); recombine.
5. Re-pair, repeat (≤ `max_iter`, default 5, or until pair set stable).
6. Score = intensity-weighted Gaussian overlap of refined pairs minus penalties
   for unmatched spots/vectors.
Best refined candidate wins; store orientation quat, score, phase, and `A`.

### 3.4 Friedel (g ↔ −g) symmetry — constraint + QC (in from the start, cheap)
- Diffraction is centrosymmetric: a real strain field maps g and −g symmetrically
  (|A·g| relation holds for ±g). The component of the fit that makes g and −g
  residuals *asymmetric* is **not** physical strain — it indicates skew in the
  vector finding (miscentered direct beam, detector distortion, subpixel bias).
- Implementation:
  - **QC metric** (always computed): `friedel_asymmetry` = mean over matched
    g/−g pairs of `||r(g) + r(−g)||` (residual vectors should cancel under
    centrosymmetry). High value flags skewed vector finding, surfaced per pattern
    and as a map.
  - **Constraint** (follow-on): symmetrize the affine solve by averaging g and
    −g contributions, so the recovered `ε` is the centrosymmetric part only.
- Full point-group symmetry regularizer (beyond Friedel) is deferred (§7).

### 3.5 Whole-field fit (the "fit the entire 4D dataset" answer)
- **Stage A — independent per-pattern** (baseline; what §3.2–3.4 give).
- **Stage B — warm-start propagation**: process patterns in a snake/space-filling
  order; seed each pattern's coarse candidate list + affine from the converged
  neighbor. Near-identical neighbors → ICP converges in 1–2 iters and stays on
  the correct orientation branch. Big speed + robustness win, low risk. **Built
  first after the baseline.**
- **Stage C — smoothness coupling** (optional toggle): a light TV-like
  regularizer penalizing orientation/strain jumps between neighbors, one extra
  sweep. Denoises the strain field. Deferred until B is validated.

## 4. Output container

Extend `SpyDEOrientationMap` (or a thin `SpyDEVectorOrientationMap` subclass):
- existing: quats, corr, phase_idx, mirror, IPF/RGB, correlation_map.
- **new per pattern**: affine `A` (2×2) → decomposed strain tensor
  `(εxx, εyy, εxy)`, rotation angle, dilatation; `friedel_asymmetry` QC scalar;
  matched/unmatched spot counts.
- **new maps**: `strain_map(component)`, `dilatation_map()`, `shear_map()`,
  `friedel_asymmetry_map()`, `n_matched_map()`. Save/load round-trips the affine.

## 5. UX (live single-pattern refine first)

Toolbar action on the vectors result tree, gated `requires_vectors: True`,
mirroring the dense OM caret (`spyde/actions/pyxem.py::orientation_mapping`):
- **Load** tab: CIF drop → phases; voltage.
- **Library** tab: resolution, min intensity → generate library (reuse dense).
- **Refine** tab (the "looks good immediately" piece): fits the pattern under the
  crosshair live; overlays matched template spots (green) on measured vectors
  (red); shows score, recovered strain (εxx/εyy/εxy), and `friedel_asymmetry`.
  Sliders: strain bound, match tolerance, top-N. Updates on crosshair move.
- **Run** tab: whole-field compute (Stage A→B) → `SpyDEOrientationMap` with
  orientation + strain maps; live IPF-RGB shm preview as today; Save.

## 6. Benchmarking & data quality

### 6.1 Synthetic ground-truth tests (unit, headless — built with the core)
- Generate spot sets from known orientations; apply known rotation + known affine
  (scale, uniaxial strain, shear) + Gaussian position noise + missing/spurious
  spots. Assert:
  - recovered orientation within tolerance (angular error < `resolution`),
  - recovered strain tensor within tolerance of the applied affine,
  - bounded-strain clamp respected,
  - **Friedel QC**: symmetric applied strain → low `friedel_asymmetry`; injected
    skew (shift one of a ±g pair) → high `friedel_asymmetry`.

### 6.2 Performance benchmarks (`spyde/tests/benchmark_vector_orientation.py`)
- Vary: n_templates (resolution), n_spots/pattern, nav size, top-N, max_iter.
- Report: coarse-search ms/pattern, refine ms/pattern, full-scan wall time,
  patterns/s. Compare independent (Stage A) vs warm-start (Stage B) — expect B
  to cut ICP iterations and wall time substantially.
- Compare against the dense OM wall time on the same scan as a sanity baseline
  (vectors should be much faster per pattern).
- Numbers recorded in `benchmarks.md` alongside the existing vector/Dask numbers.

### 6.3 Data-quality diagnostics (surfaced in the UI + as maps)
- `friedel_asymmetry_map` — high = skewed vector finding (beam center / distortion).
- `n_matched_map` / fraction of template spots matched — low = poor detection or
  wrong phase.
- correlation map — existing.
- residual-magnitude map — mean pair residual after affine fit.
- A "data quality" readout in the Refine tab summarizing these for the current
  pattern so issues are caught before a full-scan run.

## 7. Build order

1. **[DONE 2026-06-14] Matching core + tests** — `spyde/actions/vector_orientation.py`:
   `build_template_library` (reuses dense generation + per-template spots +
   coarse cache), `coarse_seed` (pyxem matcher on rasterized vectors),
   `fit_pattern` (soft-assign+sink LM over θ,A,t; bounded strain;
   `strain_from_pose` polar-decomposes rotation out so strain is rotation-free),
   `compute_vector_orientation` (whole-field driver, snake-order warm-start with
   residual-gated reseed), `VectorOrientationResult` with strain/dilatation/shear
   maps + Friedel QC. Tests: `spyde/tests/test_vector_orientation.py` — 15 pass
   (pose math, strain bound incl. reflection reject, clean/missing/spurious
   strain recovery, cap enforcement, Friedel QC both directions, batch driver
   uniform-field recovery + map accessors, Ag-library integration). Key fix
   beyond the probe: strain MUST be the polar-decomposition stretch of
   `A·Rot(θ)`, not `0.5(A+Aᵀ)−I`, or residual rotation leaks into strain.
2. **[DONE 2026-06-14] Live Refine caret** — `spyde/actions/vector_orientation_action.py`
   (`vector_orientation_mapping`), YAML-gated `requires_vectors`. 3-tab wizard
   (Load CIF → Library generate → Refine). Refine overlays fitted template spots
   (green ScatterPlotItem) on measured vectors (red) at the crosshair position,
   updating on crosshair move + slider changes (strain cap, tolerance/sink),
   with live strain (εxx/εyy/εxy %) + residual + Friedel readouts. Worker-thread
   fit marshalled to GUI via a relay. Tests:
   `spyde/tests/test_vector_orientation_gui.py` — 3 pass (gating, 3-tab wizard
   build, generate→refit→overlay with green+red spots drawn). Also added:
   `fit_pattern` seeds every template at angle 0 when the library has no coarse
   cache (single-template / small libs) instead of requiring the pyxem matcher.
   App boots with the action registered.
3. **[DONE 2026-06-14] Whole-field batch + maps** — `compute_vector_orientation`
   driver, `VectorOrientationResult.to_orientation_map()`/`ipf_color_map()` bridge
   to `SpyDEOrientationMap`, strain/dilatation/shear/residual/Friedel maps; caret
   Run tab spawns the batch (worker thread + live progress) and opens orientation
   IPF + strain + residual + Friedel-QC map windows. Tests: core
   `test_to_orientation_map_and_ipf`, GUI `test_run_computes_field_and_opens_maps`.
   Benchmark in §7e — **warm-start defaults OFF** (slower + less accurate on real
   data; independent ~59 ms/pat). 19 tests pass + app boots.
4. **[partial] Strain-map visualization** — strain component maps open as plot
   windows (autoscaled, symmetric levels). Richer viz (combined RGB strain, nav
   linkage) is a follow-on.
5. **Benchmarks** (§6.2) + `benchmarks.md` entry.
6. **Deferred**: full point-group symmetry regularizer (beyond Friedel);
   Stage C smoothness coupling; GPU/CuPy coarse search.

## 7b. Speed prototype findings (2026-06-14, measured)

Throwaway prototypes on `si_phase` (m-3m), reciprocal_radius=1.5, against a
synthetic strained pattern (template 7 + ~3% affine + noise). Single core,
this machine (48 cores, GPU available but unused in the probe).

**Datasets confirmed available** (pyxem.data): `si_grains_simple` (6×6×64×64 —
ideal fast-iteration), `simulated_strain` (32×32×512×512 — strain ground truth),
`si_grains`, `si_rotations_line`, `si_tilt`, `sample_with_g`, plus `sped_ag`
(real). Library: ~300 templates @2°, ~1081 @1°, ~4186 @0.5°; ~24 spots each;
`sim.get_simulation(i)` ≈ 0.08 ms/template (extract-all is a cheap one-time step).

**The affine refine is cheap — the coarse search dominates.**
- Bounded-affine ICP (6 iters, brute-force NN on ~24 spots): **0.07–1.5 ms/pattern.**
  Affine-in-the-inner-loop is affordable; it's ~5% of coarse cost. ✅ (validates
  the user's "faster search → more flexible fit" premise.)
- Naive angular-only NCC: 1.3–3.7 ms but **too lossy** — discards radial info,
  picks the wrong orientation. ❌
- Hand-rolled 2D-polar FFT-NCC: 180–686 ms/pattern AND still wrong — the per-pattern
  rfft over (n_templates, NR, NA) is the bottleneck. Don't hand-roll this. ❌
- pyxem `_mixed_matching_lib_to_polar` (the proven matcher): 13–187 ms/pattern
  (JIT-warmup variance), but **only correct if the sparse vectors are rasterized
  onto the matcher's exact (r,θ) grid** — my approximate binning misaligned with
  `get_slices2d`, so it MISSED. The matcher is right; the input rasterization must
  use the same radial axis `build_matching_cache` derives.

**Full-scan extrapolation** (coarse + 5 ICP, ÷10 for the cluster's GPU/CPU lanes):
256×256 @0.5° ≈ 84 s coarse-bound (pyxem matcher) — acceptable; @1° much less.
The ICP adds negligible time. So the design is viable; **the open work is the
coarse metric, not the affine.**

**Revised coarse-stage decision:** rasterize measured vectors onto the dense
matcher's polar grid and reuse pyxem `_mixed_matching_lib_to_polar` (proven,
already cached via `build_matching_cache`) rather than a bespoke sparse NCC.
Correctness hinges on matching the radial axis exactly (reuse
`signal.calibration.get_slices2d` outputs / the `radial_axis` from
`build_matching_cache`). A bespoke GPU sparse-NCC (#899's aspiration) is a later
optimization only if the pyxem matcher's per-pattern cost is the bottleneck after
the cluster split — the numbers say it likely isn't at ≤256² / ≥0.5°.

## 7c. Cost-function bake-off (2026-06-14, measured)

The method is a **continuous cost function + iterative (LM) fit** over pose
params `(θ, A, t)` where measured `v ≈ A·Rot(θ)·g_template + t` — NOT
rasterize-and-reuse-dense (that re-quantizes the exact sub-pixel vectors and
gains nothing over the raw-data path). The fit is local refinement; the coarse
stage only pins the discrete orientation branch.

Bake-off: synthetic template (~14 spots) + known 17° rotation + known symmetric
~3% strain + per-scenario noise/missing/spurious, fit by `scipy.least_squares`
with σ-annealing (0.3→0.12→0.05), seeded within ±3° of truth (coarse-search
resolution). Metric = max abs strain-tensor error (truth strain ≈ 0.03).

| scenario  | soft | hardNN+Huber | chamfer | **soft+sink** |
|-----------|------|-------------|---------|---------------|
| clean     | 0.029| 0.017       | 0.012   | 0.060 |
| noisy     | 0.014| **0.56 ✗**  | 0.048   | 0.059 |
| missing   | 0.011| 0.025       | **0.25 ✗** | 0.079 |
| spurious  | **0.76 ✗** | 0.030 | 0.074   | 0.064 |
| realistic | **0.65 ✗** | **0.48 ✗** | 0.084 | 0.069 |

Speed: soft/hardnn/chamfer 20–90 ms/fit (32–108 nfev); **soft+sink 4–5 ms/fit
(4 nfev)** — the confidence weighting smooths the cost surface so LM converges
almost immediately.

**Findings (these decide the design):**
1. **A good orientation seed is mandatory.** Starting LM from identity pose
   (no seed) gives garbage (strain errs 0.3–2.3) — the rotation/affine coupling
   creates local minima. Coarse stage MUST pin the branch first; the fit is
   refinement only.
2. **Plain soft-assign and hard-NN each win one scenario and catastrophically
   fail another** (the ✗). Not robust enough alone.
3. **Soft-assign + no-match sink (outlier rejection) is the winner**: never
   blows up (worst 0.079), and 5× faster. A measured vector with weak total
   soft-weight (spurious/far) down-weights its own residual, combining
   soft-assign's noise/missing tolerance with chamfer's spurious tolerance.
4. soft+sink's clean-case floor (0.060) is slightly high — the sink threshold
   over-suppresses good matches. Tunable; needs a sweep of the sink bandwidth so
   clean-case strain resolves to <0.01 while keeping spurious robustness.

**Decision:** cost term = **intensity-weighted soft-assign Gaussian with a
no-match sink**, fit by LM over `(θ, A, t)` with σ-annealing. Add a symmetric
(chamfer) template→measured term only if a "missing reflection" penalty proves
needed for phase discrimination. The dense pyxem matcher is used ONLY to seed the
coarse orientation branch (§7b), never for the strain fit.

Next prototype work before implementing the core:
- sweep sink bandwidth → clean-case strain < 0.01.
- add the bounded-strain projection (≤5%) and Friedel symmetrization into the
  residual, re-measure.
- validate the coarse→seed→fit handoff end-to-end on `simulated_strain` and
  `sped_ag` (currently the coarse seed was simulated as ±3° of truth).

## 7d. Real-data validation on sped_ag (2026-06-14, measured)

sped_ag = ElectronDiffraction2D 64×208 scan, 112×112 detector,
scale 0.01336 Å⁻¹/px, offset −0.7484, calibrated. Ag FCC (a=4.0853, m-3m),
mostly [100]. Library res=1° → 1081 templates; radial_range [0, 1.068] Å⁻¹.
Pipeline: peak-find vectors (drop direct beam) → pyxem matcher coarse seed on
rasterized vectors → soft-assign+sink LM refine over (θ, A, t).

**Two bugs the synthetic tests masked, fixed here:**
1. **Affine collapse to A≈0** (`eps≈−1.0` everywhere): with measured→template
   soft-assign, the fit shrank all templates onto a few vectors (degenerate
   minimum). FIX: flip assignment to **template→measured** (each template spot
   pulled to its soft-nearest measured vector) — now every template spot must
   land on data, collapse impossible.
2. **Reflection aliasing** (`eps≈−1.0` via det<0): when the seed was a mirror
   alias, the SVD strain clamp preserved the reflection. FIX: `project_strain_bound`
   forbids reflections (flip a singular value if det(U·Vt)<0), AND a strain
   **penalty band** inside the residual (singular values outside [1±cap]) keeps
   LM from exploring the collapse rather than fixing it after the fact.

**Coarse seed**: the toy angular-profile NCC was non-discriminating on
high-symmetry Ag (same template for all patterns). The real pyxem matcher on
vectors rasterized onto its EXACT (NA, NR) grid (radial_range from
`get_slices2d`, 3×3 Gaussian splat per spot) discriminates correctly — distinct
templates per orientation, scores 96–133.

**Results (36 valid patterns, sink_bw=0.04):**
- Orientation clusters: tmpl 168 = 36%, tmpl 125 = 19% (→ 55% two related
  orientations) + minority scatter. Matches "mostly [100]". ✅
- Strain |max|: median **0.025**, p90 0.072 — physical, no collapse. ✅
- Residual: median **0.018 Å⁻¹ = 1.4 detector px** — spots land ~1px from
  templates. ✅
- Friedel asymmetry: median **0.0065** — low ⇒ sped_ag vector finding is
  well-centered (QC metric validated on real data). ✅
- Speed: **42 ms/pattern** median (LM, 1 core). Full 64×208 ≈ 558 s 1-core →
  **~56 s on the cluster**. Interactive for live refine; fine for batch. ✅
- Sink bandwidth robust over 0.02–0.08; **0.035–0.05 most stable**; use 0.04.

**Method is validated end-to-end on real data.** Remaining before productionizing
the headless core: bounded-strain penalty weight `pen` and σ-anneal schedule are
hand-tuned (50.0; 0.06→0.03→0.015) — expose as params; the synthetic clean-case
floor (§7c) is resolved by the template→measured switch (real-data residual is
1.4px). Friedel is currently a QC readout only; the symmetrization *constraint*
(average g/−g contributions in the solve) is still deferred per §7a.

## 7e. Whole-field compute + warm-start benchmark (2026-06-14, measured)

Step 3 shipped: `compute_vector_orientation` driver + `VectorOrientationResult`
with `to_orientation_map()`/`ipf_color_map()` (bridges to SpyDEOrientationMap for
IPF/correlation maps) + strain/dilatation/shear/residual/Friedel maps; Run tab in
the caret (`vector_orientation_action.py`) spawns the batch on a worker thread with
live progress and opens orientation + strain + residual + Friedel-QC map windows.

**Warm-start benchmark on a 12×16 sped_ag sub-region (1081-template Ag library):**

| mode        | ms/pattern | strain median | residual | friedel |
|-------------|-----------|---------------|----------|---------|
| independent | **59**    | **0.008**     | 0.3 px   | 0.0067  |
| warm-start  | 100       | 0.046         | 0.3 px   | 0.0080  |

**Surprise finding: warm-start is currently SLOWER and LESS accurate** — the
opposite of the Stage-B premise. Two causes:
1. The bounded affine can fit a slightly-wrong neighbour-seeded orientation by
   absorbing the mismatch as **spurious strain** (residual stays low at 0.3px, so
   a residual-only reseed gate doesn't catch it → inflated strain median 0.046).
2. The residual-gated reseed then pays for BOTH the warm fit AND the cold
   fallback fit → slower overall.

**Decisions:**
- `warm_start` now defaults **False** (driver + caret checkbox). Independent
  fitting is fast enough (59 ms/pat → full 64×208 ≈ 9 min 1-core, ~1 min on the
  cluster) and more accurate.
- Warm-start reseed gate hardened: reject a warm fit when residual > 2σ OR strain
  magnitude > 0.8·cap (large strain flags a wrong branch the affine absorbed).
- Warm-start kept opt-in for very-large-library datasets where the cold coarse
  search dominates, but it must be validated per-dataset — it is NOT a universal
  win. This is the kind of thing the spatial-smoothness coupling (Stage C) would
  address more robustly than naive propagation.

## 7f. Friedel-as-constraint benchmark (2026-06-15, measured) — NEGATIVE RESULT

Benchmarked three Friedel formulations against the independent fit, on a 10×14
sped_ag region and on synthetic ground-truth (known 3% strain + per-spot noise +
beam-center offset). Harness: `spyde/tests/benchmark_vector_orientation.py`.

| condition                  | independent | friedel-center | friedel-denoise |
|----------------------------|-------------|----------------|-----------------|
| sped_ag strain median      | 0.0081      | 0.0088         | —               |
| sped_ag ms/pat             | 39          | 39             | —               |
| synth noise .004 off 0     | 0.0086      | 0.0086         | 0.0091          |
| synth noise .012 off 0     | 0.0218      | 0.0218         | 0.0226          |
| synth noise .004 off .04   | 0.0096      | 0.0096         | 0.0092          |
| synth noise .012 off .04   | 0.0217      | 0.0217         | 0.0238          |

Variants tried (all keep unpaired vectors, per the user constraint):
- **friedel-symmetrize-residual** (split residual: ±g halves match A·Rot·g without
  t, singles match +t): SLOWER (92 vs 39 ms) and a residual-scoring bug; abandoned.
- **friedel-center** (closed-form beam center from ±g midpoints, subtract before
  fit): same speed, no accuracy change.
- **friedel-denoise** (replace each ±g pair by its √2-cleaner half-vector): no
  accuracy change, slightly slower.

**Conclusion: Friedel symmetry as a FIT CONSTRAINT does not help.** Reason: the
affine model already encodes centrosymmetry — `A` is a linear map (so A·(−g) =
−A·g automatically) and the free translation `t` absorbs any constant beam-center
offset. The soft-assign least-squares already weighs all spots jointly, so
pre-averaging ±g pairs is mathematically redundant with fitting both. A constant
miscentering (tested to 0.04 Å⁻¹ = 3 px) is fit perfectly by `t`.

**What Friedel IS good for (kept):** the **QC metric** `friedel_asym` — it detects
when peak finding is skewed in a way the affine then *masks* as spurious strain
(non-constant distortion, subpixel bias that breaks ±g symmetry). That diagnostic
value stands; it's surfaced per-pattern and as a map. The explicit constraint is
dropped. (Where Friedel could still help: a spatially-varying distortion the
affine can't absorb — but that's a detector-correction problem upstream of the
per-pattern fit, not a per-fit constraint.)

## 7g. Full-4D / neighborhood fit (per-position x,y,θ,scale,strain) — TODO

User wants the field solved with each position carrying (x,y) + θ + scale + full
strain, neighbors coupled. The current per-pattern model already fits exactly
(x,y)=t, θ, and the 2×2 affine (scale+strain) — so the parameters are in place;
what's missing is the *field coupling*. Plan: benchmark (a) independent baseline
[done: 39 ms/pat, strain 0.008], (b) warm-start [done §7e: worse], (c) a
smoothness-coupled global solve on a small tile, (d) tile+propagate. The
smoothness-coupled solve (Stage C) is the promising remaining option — it
denoises the strain *field* (the noise floor §7f is per-pattern ~0.02; neighbor
averaging should cut it) without the wrong-branch absorption that sank warm-start.

**Field-coupling benchmark (2026-06-15, POSITIVE result).** Synthetic 20x20 field
with a known strain gradient + a grain boundary (eyy step 0.020) + per-spot noise,
fit independently then post-smoothed:

| method        | strain err to GT | boundary step (true 0.020) |
|---------------|------------------|----------------------------|
| independent   | 0.0056           | +0.0199 (sharp, good)      |
| gaussian s=1  | 0.0019 (3x)      | +0.0083 (over-blurs, bad)  |
| median 3x3    | 0.0027 (2x)      | +0.0119 (edge-preserving)  |

**Neighbor coupling DOES cut the per-pattern strain noise floor** — but the right
form is **edge-preserving post-smoothing (median / bilateral), NOT in-fit coupling
and NOT Gaussian.** Median 3x3 halves the noise while keeping grain boundaries;
Gaussian denoises more but smears real features. This is the architecturally clean
version of what warm-start tried: couple AFTER the independent fit (no wrong-branch
absorption), via an edge-preserving filter on the strain field.

DECISION: add optional edge-preserving strain-field smoothing as a post-process
on `VectorOrientationResult` (median 3x3), NOT in-fit coupling. The per-position
params (x,y=t, theta, scale+strain=A) are already all fit; the "full-4D" value is
this field denoising, which the benchmark confirms. A full global TV solve could do
marginally better but median captures most of the gain for ~zero cost.

## 7h. Global field-solve / robustness benchmark (2026-06-15) — TV WINS

Goal: a more robust strain method for high-noise / few-spot data. Built and
benchmarked three field-solve approaches (harness §run_robustness): median,
total-variation (Chambolle) denoising, and an iterated fit→TV→refit (per-pattern
re-fit warm-started from the smoothed field). Synthetic 12×12 field, smooth
strain + a grain boundary, swept noise + spot-dropout. Error to ground truth:

| noise/drop | independent | median | **TV** | iterated |
|------------|-------------|--------|--------|----------|
| 0.010/0.0  | 0.0136      | 0.0056 | 0.0064 | 0.0264   |
| 0.020/0.0  | 0.0260      | 0.0147 | 0.0074 | 0.0260   |
| 0.020/0.3  | 0.0266      | 0.0137 | 0.0082 | 0.0278   |
| 0.035/0.3  | 0.0288      | 0.0138 | 0.0069 | 0.0283   |
| 0.050/0.4  | 0.0265      | 0.0121 | **0.0045** | 0.0274 |

**TV is the clear winner and the gap WIDENS with noise/dropout** — at 0.05 noise +
40% spot dropout it's **6× better than independent** (0.0045 vs 0.0265) and 2.7×
better than median. Its piecewise-constant prior matches grain structure, so it
pools aggressively in flat regions while keeping boundaries.

**The iterated / joint per-pattern re-fit is the WORST** (~0.027, no better than
independent) — the SAME failure mode as warm-start (§7e): re-fitting per-pattern
lets the bounded affine re-absorb noise and undo the smoothing. **Post-fit field
denoising is strictly better than feeding smoothed seeds back into per-pattern
fits.** This also settles the "full-4D joint fit" question: the field-level TV
solve IS the full-dataset method, and it beats per-pattern coupling.

DECISIONS:
- `VectorOrientationResult.smoothed_strain(method="tv", weight=0.03)` — **TV is
  now the default** (median kept as `method="median"` + as the skimage-missing
  fallback). Caret Run-tab "Smooth strain field" checkbox now applies TV.
- **Do NOT build the iterated/global per-pattern joint solve** — benchmarked
  worse. A true global TV-regularized *pose* solve (one coupled optimization over
  all poses) might edge out post-hoc TV, but the data says the marginal gain isn't
  worth the cost: post-fit TV already gets 6× at high noise. Left as a future
  option only if a dataset proves post-hoc TV insufficient.

## 7i. Nav-dimension denoise for peak finding: TV vs Gaussian (2026-06-15)

Question: the pipeline blurs across the SCAN (nav) axes before NXCORR peak
finding (NavBlurCache: gaussian_filter sigma=(s,s,0,0)) — adjacent probe
positions see near-identical patterns, so averaging a detector pixel across
neighbours suppresses per-frame Poisson noise. Does nav-space *TV* denoising beat
nav-space *Gaussian*, especially at grain boundaries where Gaussian smears two
orientations together? Harness: `spyde/tests/benchmark_peak_denoise.py` (synthetic
4D, two grains + sharp boundary, Poisson noise; NXCORR detection F1 + subpixel
error vs known spots; tuned to F1=1.0 on clean data so noise has headroom).

| dose      | none F1 | gaussian F1 | TV F1 | notes |
|-----------|---------|-------------|-------|-------|
| medium    | 0.976   | 0.978       | 0.973 | already near-perfect; denoise unneeded |
| low       | 0.974   | 0.978       | 0.979 | "" |
| very_low (4 cts) | **0.643** | 0.977 | 0.982 | denoise ESSENTIAL here |
| very_low boundary | 0.650 | 0.984 | **0.992** | TV best at the boundary |

Boundary-stress (sigma=2 / weight=0.8, very low dose):
- gaussian: interior F1 0.979, **boundary 0.976** (degrades at the boundary —
  averages across grains)
- TV:       interior F1 0.988, **boundary 0.994** (best AT the boundary —
  edge-preserving prior respects the discontinuity)

**Findings:**
1. **At adequate dose (>=~10 cts/spot) nav-denoise barely changes detection** —
   NXCORR's matched filter already handles it; both denoisers slightly *worsen*
   sub-pixel position error (the blur shifts centroids).
2. **At very low dose nav-denoise is ESSENTIAL** — F1 0.64 → 0.98. This is the
   real win, and it's the regime that matters for beam-sensitive / low-dose data.
3. **TV vs Gaussian: TV is marginally better for detection rate, and its
   advantage is concentrated at grain boundaries** (Gaussian smears orientations
   together there; TV doesn't) — exactly the theoretical expectation. But the gap
   is small (~1-2% F1) and **Gaussian gives better sub-pixel precision**
   (0.48 vs 0.63 px at very-low dose).

**Recommendation:** keep Gaussian as the default nav-blur (faster, better
subpixel, fine away from boundaries); offer **TV nav-denoise as an option for
low-dose data with sharp grain structure**, where its boundary fidelity helps.
Not a clear universal win — dose- and microstructure-dependent. (NOT yet wired
into NavBlurCache; this is the evidence for whether to.)

## 9. SEM / low-kV 4D-STEM orientation mapping (2026-06-15)

Goal: make the vector-OM method work in an SEM (transmission / t-SEM, 5-30 kV).
Two physical effects vs TEM, and why the sparse-vector method handles them well.

### 9.1 Curved Ewald sphere (low kV)
Ewald radius = 1/lambda; lambda grows fast as kV drops (200 kV: 0.025 Å, 5 kV:
0.173 Å — 7x). Consequences:
- **Sparser patterns** (sphere intersects fewer reciprocal points) — exactly the
  regime the sparse vector method suits; dense polar correlation weakens.
- **Spot positions follow g = 2 sin(theta)/lambda**, not the small-angle
  g ≈ 2theta/lambda. The g²-dependent radial pull is ~7x bigger at 5 kV.
- **`max_excitation_error` is the key knob** — raise it at low kV to excite
  enough reflections. Now exposed in the caret Library tab (default 0.1).
- **The library is already kV-correct**: `generate_library_from_phases(
  accelerating_voltage=...)` passes kV to diffsims, which computes Ewald-correct
  positions + excitation-weighted intensities at any voltage. As long as the
  library kV matches the data, the curvature is baked into both sides and the
  match is valid; the linear affine A then only handles residual strain/scale.

### 9.2 Flat detector (gnomonic projection) — the careful correction
A flat camera records a reflection at radius R = L tan(2theta), but its
reciprocal magnitude is g = 2 sin(theta)/lambda. The naive linear calibration
g ∝ R over-estimates g at high angle by ~(R/L)²/8. Exact map (per-vector,
azimuth-preserving radial remap):
    2theta = atan(R/L);  g = (2/lambda) sin(theta)
Implemented in `spyde/actions/sem_geometry.py`:
- `electron_wavelength(kV)` — relativistic lambda (matches diffsims).
- `detector_radius_to_g(R, L, lambda)` — exact; reduces to linear as R/L->0.
- `correct_vectors_flat_detector(...)` — remaps measured vectors to Å⁻¹.
- `prepare_sem_vectors(...)` — center (Friedel) + correction in one call.
Tests (`test_sem_geometry.py`, 14 pass): exact forward/inverse round-trip,
small-angle linear limit, azimuth preserved, TEM near-noop, linear over-estimate
at high angle, end-to-end g recovery from forward-simulated detector spots.

### 9.3 No zero beam / can't center the pattern
SEM 4D-STEM often has no direct beam to mark the origin. The vector method is
structurally robust:
- The fit's **`t` (translation) IS the unknown center** — recovered as a fit
  output; the data never needs pre-centering. (Advantage over methods that
  pre-center on the direct beam.)
- The **coarse seed's polar transform is origin-dependent**, so it needs a
  center first. With no direct beam, **Friedel/centrosymmetry midpoints give
  it**: `friedel_center()` — mean midpoint of g/-g pairs — recovers the origin
  with no central spot, robust to unpaired/junk spots (iterated, permissive
  first round → tighten). NOTE: this flips the earlier §7f result — the Friedel
  constraint was *redundant* when data was already centered (sped_ag), but is
  *essential* in the no-zero-beam SEM case. Same code, instrument-dependent value.

### 9.4 Status & remaining work
DONE: `sem_geometry.py` (+14 tests); `max_excitation_error` exposed in the caret.
The library is already kV-parametrised. `prepare_sem_vectors` is ready to call.
TODO to fully close the loop: thread camera-length + kV from signal metadata
(Acquisition_instrument.TEM.camera_length / beam_energy) into a caret "SEM mode"
toggle that runs `prepare_sem_vectors` on the vectors before fitting; validate on
a real low-kV dataset (none in hand — synthetic round-trip is verified). The
detector-distortion case (spatially-varying, beyond the radial flat-detector
term) remains an upstream detector-calibration problem, out of scope here.

## 10. Consistency audit (2026-06-15)

Full audit of orientation-mapping + vector-finding after the SEM/TV/strain work.

**Consistent / clean (verified):**
- Column constants (`COL_NAV_X..COL_INTENSITY`, `N_COLS=6`) defined once in
  `diffraction_vectors.py`; all writers/readers use them. (Fixed: `_update_scatter`
  was using bare literals `rows[:,2/3]` + a stale 5-col docstring → now uses
  `COL_KX/COL_KY` and the correct 6-col doc.)
- OM column constants (`COL_LIB_IDX..COL_MIRROR`) defined once in
  `orientation_compute.py`; shared helpers (`template_tables`,
  `resolve_quaternions`, `phase_to_dict`, `build_matching_cache`) reused by the
  vector path — no duplication.
- All YAML toolbar actions + setup/subfunctions resolve to real callables
  (checked programmatically).
- Build-guard pattern (`_*_BUILT_TOOLBARS`) uniform across CZB/OM/FV/VOM carets.
- Overlay colours uniform: red = measured/detected, green = fitted template.
- Gating uniform: `signal_class` (Diffraction2D) for VI/OM, `requires_vectors`
  for the vector actions.
- No references to the deleted `find_vectors_gpu.py`.
- Friedel functions are distinct, not duplicated: `friedel_center` (centering,
  sem_geometry) vs `_friedel_asymmetry` (QC, vector_orientation).
- Caret params now seed from `vector_orientation.DEFAULTS` (was hardcoded
  `0.05/0.04` — removed the drift risk).

**Intentional divergences (documented in-code so they aren't "fixed" into bugs):**
- Vector lib spot extraction uses RAW `sim.get_simulation` coords; the dense
  overlay path applies pyxem's mirror/rotate/negate. Correct: the vector fit's
  free `Rot(θ)`+`A` absorb rotation/mirror, so the matcher needs un-transformed
  templates.
- `_resolve_one_quat` (vector) omits the mirror factor that `resolve_quaternions`
  (dense) applies — the vector fit doesn't track template mirror.

**RESOLVED 2026-06-15 — scene-coordinate convention.** Empirically determined
(against `_render_disks_block` + pyqtgraph col-major `imageAxisOrder`): the
vectors tree's rendered disk frame stores the array as `[ky(row), kx(col)]`, and
col-major maps array-axis-0 → scene-x, so a vector at (kx, ky) renders at
**scene-(x=ky, y=kx)**. Therefore the correct overlay is `pos=(ky, kx)` — which is
what the Find-Vectors `_update_scatter` already did. The **vector-OM caret was
wrong** (`-ky + center`, a flip) — FIXED: `_scene_xy` now returns `(ky, kx)`,
verified to land on the rendered disks (test). The dense OM caret's `-ky + center`
is correct *for the raw diffraction pattern* it displays (different image source),
so it is left as-is — the two families legitimately differ because they overlay
different images (rendered disks vs raw pattern).

## 11. Window lifecycle + vectors signal type (2026-06-15)

Two architectural fixes for bookkeeping that was being done ad-hoc.

### 11.1 DiffractionVectorsImage signal type (gating)
The Find-Vectors result tree's root is a rendered-disk *image* (Diffraction2D
subclass), so it matched the dense `signal_class: Diffraction2D` gate and wrongly
showed Virtual Imaging / Orientation Mapping / Find Diffraction Vectors. Fix: a
proper HyperSpy signal type **`spyde_diffraction_vectors_image`**
(`spyde/signals/diffraction_vectors_image.py`: `DiffractionVectorsImage` +
`LazyDiffractionVectorsImage`, subclassing pyxem `Diffraction2D` so it still
displays like one). Registered via `spyde/hyperspy_extension.yaml` + a
`hyperspy.extensions` entry point AND programmatically in `spyde/__init__.py`
(editable installs shadow the entry-point metadata, so the programmatic insert
into `ALL_EXTENSIONS` is what makes it work in dev — belt + suspenders). Find
Vectors now `set_signal_type("spyde_diffraction_vectors_image")` on the result.
Gating: dense actions get `exclude_signal_types: [spyde_diffraction_vectors_image]`;
vector actions get `signal_types: [spyde_diffraction_vectors_image]`. **Gate on
the _signal_type STRING, not isinstance** — Lazy/eager variants are sibling
classes (Lazy is NOT a subclass of the eager one) but share the signal_type, so
string gating covers both; isinstance silently missed the lazy result. (Long term:
upstream this type to pyxem alongside DiffractionVectors2D.)

### 11.2 MDIManager owns tree teardown (lifecycle)
Was leaky: `BaseSignalTree.close()` was a no-op; `PlotWindow.close_window` only
removed the tree at multiplot level==1 and never touched action-spawned previews
(VI/IPF/vector-OM map windows), which orphaned. Fix: **`MDIManager.close_signal_tree(tree)`
is the single authority** — `windows_for_tree(tree)` finds every window carrying
`.signal_tree` (incl. previews), closes each (which closes its PlotStates' 4
toolbars + selectors via the existing per-window path), then drops the tree +
clears `current_selected_signal_tree`. Re-entrancy guarded (`_spyde_closing` on
the tree, `_spyde_tree_teardown` on windows). `PlotWindow.close_window` delegates
to it when a level-1 (navigator) window closes. `BaseSignalTree.close()` now
actually releases its refs (signal_plots, navigator_plot_manager, attached
results). Tests: `test_window_lifecycle.py` (full teardown incl. previews,
idempotency, navigator-close cascade, other trees untouched).

## 8. Risks / open questions

- **Orientation ambiguity**: sparse spots + symmetry can make several
  orientations near-degenerate. Warm-start propagation and top-N refine mitigate;
  the correlation + n_matched maps expose where it's unreliable.
- **Coarse metric speed**: NumPy first; if patterns/s is inadequate, move the
  coarse NCC to CuPy (the dispatch infra from find_vectors/orientation_compute
  already splits GPU/CPU lanes).
- **Affine ↔ orientation coupling**: a large affine can mimic a small rotation.
  Bounding strain to ~5% and fixing the discrete orientation before the affine
  solve keeps them separable; revisit if degeneracy shows up in tests.
- **Beam-center**: the translation `t` in the affine absorbs residual
  miscentering; if it's large/structured, that's itself a QC flag.

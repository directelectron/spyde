# SpyDE 0.3.0 — Release Plan

Five waves, each independently shippable, each behind its own optional extra.
The tracker issue is **#48**; every wave and task below has a GitHub sub-issue.

Prior art this plan leans on, in order of how much it saves us:
`vector_orientation_gpu.py` (batched-torch playbook), `commit.py` +
`actions/views.py` (component-map toggle), `vector_overlay.py` (interactive
point overlay), `ipf_view.py`/`orientation_map.py` (orientation display),
`actions/README.md` (the action contract — read before writing any action).

---

## 0. Decisions locked before writing code

### 0.1 `spyde/models/` is already taken

It is the neural disk-detector package (SpotUNet + HF registry). The HyperSpy
model port therefore lives in **`spyde/fitting/`**. Do not "tidy" these
together — they are unrelated meanings of the word *model*.

### 0.2 Package layout — one submodule per domain

| Submodule | Wave | Extra | Contents |
|---|---|---|---|
| `spyde/fitting/` | 1 | — (core) | `ModelSpec`, torch component library, batched LM engine, seeding |
| `spyde/spectroscopy/` | 2 | `eels` | EELS/EDS composition → components, quantification |
| `spyde/ebsd/` | 3 | `ebsd` | pattern preprocessing, GPU dictionary indexing, refinement |
| `spyde/atoms/` | 4 | `atoms` | atom finding, sublattices, dumbbells, property maps |
| `spyde/data/` | 5 | — | example + synthetic datasets (owner: @CSSFrancis) |

Interactive wiring stays in `spyde/actions/` per the existing contract
(compute in its own package, actions outside it — `find_vectors/` is the model).
No package split; these are submodules of `spyde`.

### 0.3 Optional extras, gated at the toolbar

```toml
[project.optional-dependencies]
eels  = ["exspy>=0.3.2"]
ebsd  = ["kikuchipy>=0.11.0"]
atoms = ["atomap>=0.4.1"]
all   = ["spyde[eels,ebsd,atoms]"]
```

None of the three is installed today. Domain math comes from the real
libraries so our numbers match published results — **except** EBSD dictionary
indexing and refinement, which we implement in torch (§3) because that is the
whole point of Wave 3.

New YAML gate key **`requires_package`**, mirroring the existing
`requires_vectors`: a toolbar entry is hidden when the extra is absent, and the
caret shows a one-line install hint instead of failing on import. This is the
only framework change Wave 0 needs.

### 0.4 The fitting engine: batched GPU LM + seeded propagation

Measured baseline on this box — synthetic EELS SI, 1024 spectra × 1024
channels, power law + 2 gaussians, `multifit(optimizer="lm")`:

| | |
|---|---|
| single `fit()` | 17.6 ms |
| `multifit` | **10.83 s** = 10.6 ms/spectrum = **95 spectra/s** |
| extrapolated to 256×256 | **11.6 min** |

Single-threaded, one pixel at a time. That is the number to beat.

Free win found while measuring: **`numexpr` is not installed**, so every
hyperspy `Expression` component falls back to numpy (it warns on each model
build). Add it to core deps — costs nothing, speeds up the CPU reference path
and the live single-spectrum preview.

The engine is the `vector_orientation_gpu.py` playbook applied to curve
fitting: pack the whole nav grid into `(P, C)` on the GPU and run one batched
Levenberg–Marquardt, no Python loop over pixels. It works because **the
parameter count `n` is tiny** (≤ ~20) even when `P` and `C` are large: the
normal equations are `(P, n, n)` batched solves, which `torch.linalg.solve`
does natively. The only big intermediate is the `(P, C, n)` Jacobian, so
**chunk over `P`** to bound it (65536 × 2048 × 12 float32 = 6.4 GB whole —
chunked it is a fixed working set). This is the same "chunk the batch
dimension" rule the orientation code already follows.

SAMFire's insight — a neighbour's result is the best starting point — is kept
as the **seed source**, not as the scheduler: fit a strided coarse grid, then
propagate those parameters outward as initial guesses for one batched refine
over all pixels. We do not port SAMFire's per-pixel marker/strategy machinery;
that scheduler overhead is what makes it slow.

**Acceptance gate (non-negotiable):** the GPU engine must reproduce hyperspy's
`multifit` parameters within tolerance on the same data. We are replacing a
reference implementation, so parity against it is the test — not "it converged".
CPU (scipy) fallback stays for no-GPU machines and for the parity test itself.

---

## Wave 1 — Model fitting, 1D and 2D (`spyde/fitting/`)

The foundation. Waves 2 and 4 both fit models, so they consume this engine.

**1.1 Package skeleton + `ModelSpec`.** A serialisable description of a model
(components, parameters, bounds, free/fixed flags, signal range) that
round-trips against `hyperspy.model.BaseModel.as_dictionary()`. Storage mirrors
hyperspy — `m.store(name)` / `s.models.restore(name)` — so a saved `.hspy`
opens with its models intact in hyperspy and in SpyDE alike. Lives on the tree
as `tree.fit_models` (ownership map, `actions/README.md` §3).

**1.2 Torch component library.** Batched, autograd-differentiable ports of
`hyperspy.components1d`: Gaussian, GaussianHF, Lorentzian, Voigt, SplitVoigt,
PowerLaw, Offset, Polynomial, Exponential, Arctan, Erf, Doniach, SkewNormal,
Logistic, HeavisideStep, ScalableFixedPattern; `components2d`: Gaussian2D,
Expression. Each takes `(x, params)` → `(P, C)` fully batched. Per-component
parity test against the hyperspy component it mirrors.

**1.3 Batched LM engine.** `fit_batched(spec, data, ...)`: chunked over `P`,
Jacobian by autograd, bounded parameters, free/fixed masking, channel mask for
signal range, optional Poisson weighting (matters for EELS/EDS counts).
Benchmark harness in `spyde/tests/benchmark_fitting.py` reporting spectra/s
against the 95/s baseline. **Also implement variable projection**: components
linear in amplitude (most of them) are solved by batched linear least squares
given the nonlinear parameters, which shrinks `n` and improves conditioning —
this is what makes Wave 2's tabulated EELS edges tractable.

**1.4 Seeded propagation.** Coarse strided fit → propagate to neighbours as
initial guesses → one batched refine. Reports how many pixels converged.

**1.5 The Fit wizard.** Staged action (`fit_open/_add_component/_set_param/
_tune/_run/_commit/_close`) per the wizard protocol. Opens a caret group;
components are added line by line; live preview fits the *current* nav
position only, so it stays interactive. `fit_run` does the whole grid on a
worker with progress.

**1.6 Component picker with shape preview.** "Add component" shows what each
component looks like: the backend evaluates each candidate at default
parameters over the current signal axis and sends a small polyline; the
renderer draws it as a sparkline in the picker.

**1.7 Drag-to-adjust components on the plot.** Port of the anyplotlib
interactive-fitting example: `add_point_widget` for centre+amplitude,
`add_range_widget` for width, clicking a component's line toggles its widgets,
`pointer_move` re-evaluates the sum line. This is the primary way users tune a
model — sliders are the fallback, not the main path.

**1.8 Component area maps.** Integrated area under each component per pixel →
`commit_result_tree` with one view per component, so they toggle exactly like
εxx/εyy/εxy/ω (single click shows one, ⌘-click tiles several). This is direct
reuse of `actions/views.py`; no new display code.

**1.9 2D models.** Same engine, image-shaped data (`Gaussian2D`, `Expression`).
Lower priority within the wave but explicitly in scope.

---

## Wave 2 — EELS and EDS (`spyde/spectroscopy/`, extra `eels`) — depends on Wave 1

Scope distilled from the eXSpy example gallery: find EDS lines, EELS curve
fitting, background removal, Fourier-ratio deconvolution, residual plots,
quantification.

**2.1 exspy extra + signal-type gating** (`EELS`, `EDS_TEM`, `EDS_SEM` via the
existing `signal_types` key) + microscope-parameter panel (beam energy,
collection/convergence angle) since fits are meaningless without it.

**2.2 Composition → components.** Enter elements; auto-populate the model:
EELS gets background + an `EELSCLEdge` per ionisation edge in range; EDS gets a
gaussian per X-ray line with family ratios tied. This is the wave's headline
feature.

**2.3 Tabulated EELS edges in the engine.** `EELSCLEdge` is a GOS lookup, not
an analytic form. Interpolate the GOS table on-device and fit edge intensities
via the linear path from 1.3 — the shapes are fixed, the amplitudes are linear.

**2.4 Background removal + edge onset** as a RegionAction (drag the fit
window), reusing the existing region-selector machinery.

**2.5 Interactive drag fitting** — 1.7 applied to edges/lines, which is where
it pays off most.

**2.6 Quantification → maps.** EELS relative composition; EDS Cliff–Lorimer /
zeta-factor with absorption. Output is per-element maps → the same view-toggle
commit as 1.8.

**2.7 Fourier-ratio deconvolution, thickness map, residual view.**

---

## Wave 3 — EBSD (`spyde/ebsd/`, extra `ebsd`)

Independent of Waves 1–2. The reuse story here is unusually good: SpyDE already
depends on **orix** and already has IPF views, orientation maps and a 3D IPF
toolbar from the 4D-STEM work, so a CrystalMap feeds straight into existing
display code.

**3.1 kikuchipy extra, EBSD signal type, IO** (`.h5`, `.ang`, EDAX/Oxford/
Bruker via kikuchipy's readers) + `EBSDDetector` geometry panel.

**3.2 Pattern preprocessing**: static/dynamic background removal, neighbour
averaging, average dot-product map. These are per-pattern and embarrassingly
parallel — straight to torch.

**3.3 GPU dictionary indexing — the reason this wave exists.** Normalized
cross-correlation between `P` experimental and `D` dictionary patterns is, once
both are zero-mean/unit-norm, a single matmul `E @ Dᵀ` → `(P, D)`, followed by
top-k. Chunk over both `P` and `D` with a running top-k so the `(P, D)`
intermediate never materialises (65k × 100k float32 = 26 GB whole). kikuchipy
does this with dask; torch should be dramatically faster. Parity against
`kikuchipy.indexing` scores is the acceptance test.

**3.4 Orientation refinement.** Batched optimisation of `(φ1, Φ, φ2)` and
optionally the projection centre — the direct analogue of
`vector_orientation_gpu.py`. Needs differentiable master-pattern sampling
(sphere → detector bilinear interpolation) in torch; that is the real work of
this task. Watch the same numerical traps the vector-orientation work hit:
rotation-branch ambiguity and coarse-stage bias (CLAUDE.md, GPU Computing).

**3.5 CrystalMap → existing IPF display**, phase merging
(`merge_crystal_maps`), orientation similarity map.

---

## Wave 4 — Atom position mapping (`spyde/atoms/`, extra `atoms`) — uses Wave 1's engine

Atom refinement *is* a batched 2D gaussian fit, so 1.3 does the heavy lifting.

**4.1 atomap extra + initial atom finding** (`get_atom_positions`), sublattice
construction, nearest neighbours, zone axes.

**4.2 The three GUI functions, as anyplotlib overlays.** atomap's
`select_atoms_with_gui` / `add_atoms_with_gui` /
`toggle_atom_refine_position_with_gui` are matplotlib-based, so we reimplement
the *interaction* over our own overlay — which `vector_overlay.py` already does
for diffraction vectors (add/remove points, per-point state colouring). Polygon
select for phase separation; click to add/remove atoms; click to toggle an
atom's refine flag green↔red.

**4.3 Refinement**: centre-of-mass then batched 2D gaussian via the Wave 1
engine, honouring the per-atom refine flags.

**4.4 Dumbbell lattices** (`get_dumbbell_vector`, `Dumbbell_Lattice`) per the
atomap dumbbell guide.

**4.5 Property maps** — ellipticity, neighbour distance, monolayer spacing,
displacement — through the same view-toggle commit as 1.8.

---

## Wave 5 — Data module (`spyde/data/`) — owner @CSSFrancis

Datasets for 4D STEM, EELS, EDS, EBSD and in-situ. Waves 1–4 code against the
contract below and use synthetic stand-ins until real data lands, so this
never blocks them.

**Contract** — mirror `spyde/backend/tutorial_data.py`:

- **Bundled synthetic**, no download, fast enough for Playwright specs:
  `load_test_data_<modality>` on `TestHarnessMixin`. Must be *asymmetric and
  crisp* so bugs are pixel-visible (the `si_grains` / `movie` precedent).
- **Real, downloaded**: registered in the Examples menu, reached by
  `load_example {name}`.
- Each entry declares `signal_type`, nav/signal shape, calibration and, where
  relevant, the microscope parameters the fits need (§2.1) — a dataset that
  arrives uncalibrated cannot exercise the feature it is meant to test.
- Lazy loads must use signal-spanning chunks (CLAUDE.md Live-Display §1).

---

## Cross-cutting (Wave 0, do first)

- `requires_package` gate key + install-hint caret (§0.3).
- `[project.optional-dependencies]` block + `numexpr` into core deps (§0.4).
- CI: one matrix job installing `spyde[all]` so gated code is actually
  exercised; the default job proves the app still works with none of it.
- Docs: one guide per modality in `guides/`, and the data module's examples
  wired into the existing tutorial menu.

## Verification standard

Per CLAUDE.md: a green pytest run and a clean `tsc` are **not** verification for
anything that adds windows, draws overlays or wires renderer↔backend. Every
wave's UI work ships with a Playwright spec built on `electron/tests/_harness.cjs`
and screenshots that were actually looked at. Where a wave replaces a reference
implementation (1.3 vs `multifit`, 3.3 vs `kikuchipy.indexing`), numerical
parity against that reference is the acceptance test.

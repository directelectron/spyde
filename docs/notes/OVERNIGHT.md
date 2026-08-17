# Overnight Run — Parity Push + Harden

Plan (per user): **implement features first, then harden** (test/de-flake/document).
A/B method: **compare vs Qt source + docs** (can't launch the Qt GUI here). Test on
**sped_ag + Dask** where feasible. Kill stray Electron processes before each E2E run
(`pkill -9 -f out/main/index.js`; `npx playwright clear-cache` after kills).

## Features

- [x] **Center Zero Beam** — two tabs: Automatic (`czb_auto`, pyxem `get_direct_beam_position`+`center_direct_beam`, optional flat-field) + Manual (draggable crosshair → constant shift). `center_zero_beam.py` (Qt-free), `CenterZeroBeamWizard.tsx`. `test_center_zero_beam.py` (2: auto COM-centres, manual-from-crosshair) + `center_zero_beam.spec.ts` (real backend) + `spyde.spec.ts` UI. NOTE: `add_transformation` only REGISTERS the new PlotState — had to add `_display()` (set_plot_state + re-fire navigator) for the DP to actually show the centered result.
- [x] **Aesthetics: Axes table** — click-to-edit `EditableCell` (span→input on click, commit
      on blur/Enter, Esc reverts); dropped the size column; dataset shape/dtype now in the
      Metadata panel (`Dataset` section in `build_metadata_dict`); edits already re-push every
      plot (`_set_axis`). `test_axes_edit.py` + updated `selector.spec.ts` (click→input→commit).
- [x] **Multi-phase OM** — `om_generate_library` accepts `cif_paths` (list); `generate_library_from_phases`
      + `_do_compute_orientations` + `ipf_color_map`/`ipf_sphere_points` already handle >1 phase
      (verified on pyxem `fe_multi_phase_grains`). `OrientationWizard` Load tab now adds multiple
      `.cif` phases (`om-cif-list`). Live single-pattern refine overlay is single-phase only → skipped
      for multi-phase (whole-field Run is multi-phase). `_count_templates` handles the per-phase
      rotation list. `test_multiphase_orientation.py` (2-phase map + 2D IPF RGB all-coloured; multi-CIF
      handler). NOTE: a phase with no reflections inside the reciprocal radius → empty diffsims sim →
      `max()` ValueError; the dataset's recip radius must cover both phases.
- [x] **2D IPF map in anyplotlib** — the IPF window's 2D view is the RGB `ipf_color_map` via
      anyplotlib `imshow` (RGB path); verified clean (every pixel coloured) in the multi-phase test.
      Colour-key triangle legend → folded into IPF-explorer polish (#47).
- [x] **Workflow / signal-tree view polish** — redesigned `TreeNodes` (compact rows, depth
      guide-rail `└`, node dot, hover tint, ACTIVE-node highlight); section renamed "Workflow".
      Backend emits `active_signal_id`; `Session._reemit_signal_tree(plot)` re-pushes after a node
      switch / transform so new steps appear + highlight (CZB now shows its "Centered" node).
      `center_zero_beam.spec.ts` asserts the node; `signal-tree switcher` test still green.
- [~] **IPF explorer toggle polish** — the 2D⇄3D toggle + interactive (rotatable) `scatter3d`
      sphere were delivered in the IPF-toggle work; functionally complete. DEFERRED nicety: an IPF
      colour-key triangle legend on the 2D map (needs an extra backend-rendered key image).
- [x] **Stick windows** — edge-snap while dragging (`snapToPeers`, 9px capture, perpendicular
      overlap), dwell ~1.1s while edge-aligned → forms a stick group (🔗 badge), and dragging one
      stuck window moves the whole group (`onStuckMove` → `groupNudge`). `MDIArea` owns peer
      geometry + group membership (merge on overlap) + close cleanup. Additive to `SubWindow`
      (existing drag/resize/close tests still green). `stick_windows.spec.ts` (snap → badge →
      group-move). PARTIAL: group **resize** (shared edges resizing together) is NOT implemented —
      individual resize still works + the plot resizes (`onResize`→`resizeFigure`); group-resize is
      a follow-up.

## Harden

- [ ] De-flake the real-Dask Playwright tests (4 flaky under parallel launch).
- [ ] Coverage-gap review of the migrated + Playwright suites; document weak/misleading tests.
- [ ] Refactor/perf review of the codebase (data flow, structure, docs).
- [ ] A/B audit vs Qt source for each ported action; document discrepancies + root cause.

## Discrepancies / Issues found (running log)

- **Real-Dask Playwright tests are flaky under parallel launch** (om_wizard_lazy, vector_om_lazy,
  vector_vi_lazy, vi_lazy). They pass on retry; the cause is several Electron+LocalCluster
  instances starting at once (cluster-ready contention), NOT product bugs. To address in the
  harden phase: run the real-Dask specs with `--workers=1` / a serial project, or add a longer
  Dask-ready gate. `vi_lazy` doesn't touch any feature under test — pure environment flakiness.
- **Playwright "No tests found / test.beforeAll() did not expect" cascade** after a `kill -9` of
  electron/playwright: the transform cache corrupts. `npx playwright clear-cache` alone is NOT
  always enough — must ALSO `rm -rf node_modules/.cache`. Document this in the test runbook.
- **`add_transformation` does not switch the displayed signal** — it only REGISTERS the new
  PlotState. Any in-place transform (Center Zero Beam) must explicitly `set_plot_state(new)` +
  re-fire the navigator (`_display()` in `center_zero_beam.py`). Worth auditing other transform
  actions (Rebin/FFT) for the same gap.
- **Multi-phase library build fails if a phase has no reflections** within the data's reciprocal
  radius (empty diffsims sim → `max()` ValueError). Not guarded in `generate_library_from_phases`;
  consider a clear error instead of the raw numpy ValueError.

## A/B audit vs Qt source (parity gaps found)

Compared each Electron port against the Qt implementation in `pyxem.py` /
`find_vectors.py` / `vector_orientation_action.py`. Functional gaps (the
workflows produce correct results; these are missing knobs / depth):

- **Center Zero Beam — Auto methods.** Qt offers `cross_correlate / center_of_mass /
  blur / interpolate` (each with extra params: sigma, cc disk radii) + a
  `signal_slice` RectangleSelector ROI. Electron Auto exposes `center_of_mass`
  only (+ `half_square_width`). The others need their extra kwargs wired
  (`get_direct_beam_position` rejected them without sigma/radius in testing).
- **Center Zero Beam — Manual.** Qt fits a LINEAR PLANE from multiple clicked
  control points across the scan (so a tilted beam centre is corrected per
  position). Electron Manual applies a single crosshair as a CONSTANT shift.
  Adequate for a uniform offset; not for a beam-shift gradient.
- **Vector OM — Run controls.** Qt Run has `strain_cap`, `sink_bw`, warm-start,
  smoothing + a 12-channel live preview buffer (IPF X/Y/Z + εxx/εyy/εxy painted
  progressively). Electron exposes `strain_cap` + `smooth`; no `sink_bw` /
  warm-start sliders and no progressive preview (the maps appear when the run
  finishes).
- **Find Vectors — tuning extras.** Qt preview had a beam-stop mask (click a dark
  region) + a "show correlation image" overlay. Electron's live preview has the
  σ/radius/threshold/min-dist/subpixel sliders but not the beam-stop mask or the
  correlation-image overlay.
- **OM live refine — multi-phase.** The single-pattern best-match overlay
  (`best_match_spots`) is single-phase; multi-phase libraries skip the live
  refine (the whole-field Run IS multi-phase). Qt's refine was also effectively
  single-phase per pattern.
- **IPF colour-key legend.** Neither shows the stereographic-triangle colour key
  next to the 2-D IPF map (Qt drew one). Deferred nicety.

No CORRECTNESS discrepancies were found — the ported computes match the Qt
results (centering, vectors, orientation/strain, IPF colours, multi-phase).

## Codebase / test review notes

### Elegance refactor — DONE (audit of the Electron-support changes)
- **Frontend wizard shell** — extracted `WizardShell.tsx` (box + header + status +
  `TabRow` + `Field`/`NumInput`/`Slider`/`Check` primitives + the single shared
  `S` style object). The 4 carets (Orientation / Vector-OM / Find-Vectors / CZB)
  dropped their ~60-line duplicated `S` blocks and chrome → each is now just its
  step content (≈40–95 lines, down from 590 total with 4 copies of the styles).
- **Backend `_src_plot_tree` / `current_signal`** — were copy-pasted in 4 modules;
  centralised in `actions/context.py`, imported everywhere.
- **Staged-handler dispatch** — 13 identical `elif action == …: import; fn(self,
  plot, payload)` branches collapsed to one `_STAGED_HANDLERS` table + a single
  lazy-import branch in `session.py`.
- **Overlay base class** — `VectorOverlay` / `OrientationOverlay` /
  `FindVectorsPreviewOverlay` / `VectorOrientationOverlay` had byte-identical
  `attach`/`_on_indices`/`_push`/`remove` + calib→pixel math. Extracted a
  `_DPOverlay` base (attach + navigator wiring + push + show/hide + remove +
  `_to_px`/`_calibrate`); subclasses keep only `__init__` + `_offsets_for`
  (+ a `_marker_kwargs` override for the live-radius preview; the 2-group
  Vector-OM overlay overrides the push/attach). ~120 dup lines removed.
- Verified: 98 migrated headless + 60 Playwright still green; ruff F401 shows NO
  unused imports in the refactored modules (the remaining F401s are pre-existing
  in the legacy Qt files `pyxem.py` / `find_vectors.py`).

### Not done (noted for later)
- A shared `display_node(plot, signal)` helper (set_plot_state + re-fire navigator
  + re-emit workflow tree) — CZB does this inline; would DRY future transforms.
- A `@background_action` decorator for the handler bg-thread + emit_error boilerplate.
- Gaps: no Playwright test asserts group **resize**; CZB Auto only covers
  `center_of_mass`.
- Test hardening: serialised the Playwright suite (workers:1) to kill real-Dask
  flakiness + 150s headroom on the slow cold OM compute; documented the
  `rm -rf node_modules/.cache` + clear-cache recovery.

## Test status snapshot

- Start: 93 migrated headless, 57 Playwright (4 real-Dask flaky-on-parallel).
- **End: 98 migrated headless PASSED · 60 Playwright PASSED, 0 flaky** (serial workers:1, 5.3m).
  De-flake confirmed: the 4 previously-flaky real-Dask specs all passed clean.
- Runbook: the Playwright "No tests found / beforeAll" cascade is caused by a broad
  `pkill -9 -f Electron` corrupting the transform cache. Recovery: `rm -rf node_modules/.cache &&
  npx playwright clear-cache`. Only kill `out/main/index.js`, never a bare "Electron" match.

## Harden — done

- [x] De-flake real-Dask Playwright (serial workers:1 + 120s timeout → 0 flaky).
- [x] Coverage gaps + weak tests reviewed (see notes above; group-resize + CZB non-COM uncovered).
- [x] Refactor/perf notes captured (display_node helper, wizard shell, @background_action).
- [x] A/B audit vs Qt source — gaps documented (no correctness discrepancies).

## Follow-up fixes (post-overnight)

- [x] **CIF picker recents** — `CifRecents.tsx` (`useCifRecents` + `RecentCifs` chips,
      localStorage `spyde:cif-recents`, cap 8). Wired into OM (multi-phase) + Vector-OM
      Load tabs. `spyde.spec.ts` "CIF picker remembers recents".
- [x] **OM calibration + flip + Dask** — the "orientation vectors are flipped" report
      was a **calibration artifact**, NOT a transform bug. Round-trip diagnostic:
      as-is 9.1px (best) vs y-mirror 9.46 / x-mirror 9.39 / 180° 9.32 → `best_match_spots`
      is correct, left untouched. Real fixes: (1) restore `sped_ag` scale 0.0267 /
      offset -1.4968 (pyxem defaults are half-true; `_EXAMPLE_CALIBRATION` in
      `session.py`); (2) gamma defaults to 1.0; (3) `_do_compute_orientations` falls
      back to in-process `_threaded()` when a Dask worker dies (was crashing Compute
      Map with `FutureCancelledError`, so no IPF). See memory om-calibration-dask-resilience.
- [x] **Find Vectors result window display** — computed "— Vectors" window was an
      all-black placeholder with no markers. Root cause: `CachedDaskArray` captured the
      placeholder zeros at first render; swapping `tree.root.data` to `to_rendered_dask()`
      didn't invalidate it. Fix in `find_vectors_action._finalize`: clear
      `cached_dask_array` + `_clear_cache_dask_data()` after the swap, set
      `needs_auto_level`, and attach `_overlay_on_result` (red circles over the rendered
      disks, Qt parity). On REAL distributed data the lazy nav slice arrives async and
      could still stay black, so `_install_render_display` REPLACES the navigator slice fn
      with an in-process `vecs.render_frame(iy,ix)` (Qt's `_make_hooked` approach). Also
      `_clip_to_bounds` drops spurious out-of-detector vectors (pixel ~24000 → giant/off
      circles) + radius cap. Tests: `test_find_vectors_port::test_result_window_renders_vectors_and_overlays`
      + `find_vectors_result.spec.ts` (scoped bright+red pixel scan on the vectors signal iframe).

- [x] **Vector-OM "library not responding / IPF not displaying" (sped_ag audit)** —
      `vom_run` was using the SERIAL CPU fit (`compute_vector_orientation`): ~30 min on
      the full 13k-pattern scan, so the IPF "never displayed." Fixed: `_fit_field` now
      dispatches the BATCHED torch GPU path (`compute_vector_orientation_gpu`, the Qt
      production path) first, CPU fallback only if torch absent / GPU raises. Warm MPS:
      576 patterns GPU 12.8s vs CPU 50.3s. Added the missing **Refine tab** (strain-cap +
      tolerance sliders → `vom_refine` re-fits under the crosshair; strain readout streamed
      via a new `vom_fit` event + `VectorOrientationOverlay.on_fit`). Library build (~40-70s,
      `build_template_library`'s per-template `sim.get_simulation` loop) is inherent / shared
      with Qt — status message now sets the "~1 min" expectation.
      **Parity** (`benchmark_om_parity.py`, sped_ag 10×10, raw OM vs vector OM, varied region):
      IPF-Z **100% same colour**, IPF-X/Y **90% same** → the two methods AGREE (IPF-Z = beam
      axis perfect; X/Y in-plane 90%, the rest is the vector fit's in-plane branch ambiguity).
      Tests: `test_vector_orientation_om::test_fit_field_prefers_gpu_then_falls_back` (GPU-first
      dispatch + CPU fallback) + the wiring test forced to CPU; `vector_om_lazy.spec.ts` extended
      with the Refine readout check.

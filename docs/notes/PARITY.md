# SpyDE Electron Parity Checklist

Tracks parity between the legacy Qt/pyqtgraph app and the Electron/anyplotlib
rewrite. Each item has a **specific, checkable** verification — a passing test
or a concrete visual check — not "looks done".

Status: ✅ done & verified · 🟡 partial · ⬜ not started

## How to verify

- **Backend / data flow (headless):** `SPYDE_NO_DASK=1 uv run pytest spyde/tests/migrated/`
  and `uv run python -m spyde.tests._dataflow_probe`
- **UI + rendering (Playwright):** `cd electron && npm run build && npm test`
  — includes `visual.spec.ts`, which reads real `<canvas>` pixels and asserts
  non-black.
- **Visual A/B vs Qt:** run the legacy app (`git stash` / a Qt checkout) and the
  Electron app (`npm run dev`) side by side on the SAME dataset (e.g.
  `Examples → mgo_nanocrystals`) and compare each row below.

## Core display

| Feature                                       | Status | Verification                                              |
|-----------------------------------------------|---|-----------------------------------------------------------|
| App launches, MDI + right dock                | ✅ | `spyde.spec.ts` app-launch test                           |
| Navigator image shows real data               | ✅ | `test_dataflow::test_4d_navigator_fills_with_real_data`   |
| Diffraction pattern shows real data           | ✅ | `test_dataflow::test_4d_diffraction_pattern_fills`        |
| **Canvas actually renders non-black**         | ✅ | `visual.spec.ts` reads canvas pixels (max>40)             |
| **Data pushed before iframe load isn't lost** | ✅ | `visual.spec.ts` "THE RACE"                               |
| Selectors (crosshair/rect) visible on plot    | ✅ | overlay widgets in replayed panel state; visual           |
| Colormap change                               | ✅ | `spyde.spec.ts` set_colormap                              |
| Contrast / clim                               | ✅ | `spyde.spec.ts` set_clim                                  |
| Histogram in dock                             | ✅ | `spyde.spec.ts` + `test_dataflow::test_histogram_emitted` |
| Metadata panel in dock                        | ✅ | `spyde.spec.ts` + `test_dataflow::test_metadata_emitted`  |
| Navigator drag updates DP live                | ✅ | `selector.spec.ts` (real backend drag + pointer_move live) |
| Scale bar                                     | ✅ | calibrated from signal-axis units; `test_scale_bar.py` (+ set_extent fix so the bar length is correct) |
| Dark-mode plots                               | ✅ | `visual.spec.ts` "figures render in dark mode"            |
| No MDI background grid                         | ✅ | `spyde.spec.ts` "MDI area has no background grid"         |
| Windows draggable by title bar                | ✅ | `spyde.spec.ts` "a subwindow can be dragged…"            |
| Resize tied to window (no anyplotlib triangle)| ✅ | `resizable=False`; react-rnd resize → resizeFigure IPC    |
| Perf: shared JS bundle (no per-iframe reparse)| ✅ | HTML 370KB→9KB; `test_perf.py`                            |
| Tool bar                                      | ✅ | Floating bar BELOW the window, tracks it, reveal-on-hover (fades in/out, `pointerEvents:none` when hidden so it can't intercept). Carets/params open downward. `spyde.spec.ts` "reveals on hover" + all toolbar tests hover the window first |
| App top bar (drag window + sidebar toggle)    | ✅ | Frameless window was un-draggable — added a drag-region top bar (gradient + wordmark) with an icon-only sidebar toggle. `spyde.spec.ts` "top bar is a window-drag region" / "toggle hides and shows the sidebar" |
| Navigator pixel → real-space DP mapping       | ✅ | pyqtgraph displayed the navigator transposed; anyplotlib (imshow) does not, so the Qt index math `data[(cx,cy)]` was swapped → fixed to `data[iy,ix]`. `test_nav_orientation.py` (non-square scan) |
| Example load is lazy + instant                | ✅ | `pxd.<name>(allow_download=True, lazy=True)` reads the downloaded file as dask (no eager 668MB, no zspy re-save); display no longer blocks on the navigator (background threaded compute); `tree.client` is live so a tree opened before the cluster picks it up. ~15s→~1.7s. `test_lazy_loading.py::TestNonBlockingNavigator` |

## Toolbar actions

| Action | Status | Verification |
|---|---|---|
| Virtual Imaging (multi-VI sub-toolbar) | ✅ | Qt-parity: "Virtual Imaging" submenu → "+ Add Virtual Image" adds colour-cycled (red→green→blue→…) ROI VIs, each its own window, listed as removable chips. `test_virtual_imaging.py` + `spyde.spec.ts` + real-backend `selector.spec.ts` |
| FFT of region | ✅ | `test_template_actions::test_fft` |
| Line Profile | ✅ | `test_template_actions::test_line_profile` |
| Rebin | ✅ | `base.rebin2d` (TransformAction path) |
| Zoom / Reset | 🟡 | emits IPC; host-side view ops not fully wired |
| Center Zero Beam | ✅ | Electron-native `center_zero_beam.py` (Qt-free), two-tab `CenterZeroBeamWizard`: **Automatic** (`czb_auto` → pyxem `get_direct_beam_position(center_of_mass[, half_square_width])` + optional linear flat-field → `center_direct_beam` → "Centered" tree node, DP updates in place via `_display()`) + **Manual** (`czb_manual_start` drops a draggable crosshair → `czb_manual` applies a constant `centre−picked` shift → "Centered (Manual)"). `test_center_zero_beam.py` + `center_zero_beam.spec.ts` (real backend) + `spyde.spec.ts` UI. GAPS vs Qt (OVERNIGHT.md): non-COM methods, multi-point plane-fit manual |
| Multi-phase Orientation Mapping | ✅ | `om_generate_library` accepts multiple `.cif` (`cif_paths`); the compute (`generate_library_from_phases` + `_do_compute_orientations` + `ipf_color_map`/`ipf_sphere_points`) is multi-phase (verified on pyxem `fe_multi_phase_grains`). `OrientationWizard` Load tab adds multiple phases (`om-cif-list`). `test_multiphase_orientation.py` (2-phase map + clean 2-D IPF RGB). Live single-pattern refine is single-phase → skipped for multi-phase |
| Orientation Mapping | ✅ | Electron-native `orientation_action.py` reuses Qt-free `_do_compute_orientations`; CIF→library→match→IPF-Z map window + `tree.orientation_map`. **Staged 4-tab wizard** (`OrientationWizard.tsx`, Qt parity): **Load** (cif+voltage) → **Library** (resolution+min-intensity → `om_generate_library` builds the diffsims library, caches it on `tree._om_wizard`, attaches the live refine overlay, emits `om_library_ready` which unlocks Refine/Run) → **Refine** (gamma/min-int%/normalize sliders → debounced `om_refine` re-pushes best-match spots under the crosshair; move the navigator to preview other patterns) → **Run** (n_best → `om_run` reuses the cached library → IPF-Z window). **Live overlay:** best-match template spots on the source DP, tracking the navigator (`vector_overlay.OrientationOverlay` + Qt-free `orientation_compute.best_match_spots`, `_match_lock` guards the non-thread-safe numba kernel). **Ag Silver workflow:** `benchmark_orientation_ag.py` (real sped_ag, real `Silver__0011135.cif`: Load→Library(1081 templates)→Refine→Run, real `LocalCluster`) + `test_orientation_ag.py` (fast CI: cif→library→refine) + `test_orientation_wizard.py` (headless `om_generate_library`→`om_refine`→`om_run`) + lazy E2E (`om_wizard_lazy.spec.ts` real-Dask Generate→Compute→IPF, `orientation_lazy.spec.ts`, `spyde.spec.ts` wizard UI, `test_orientation_overlay::…lazy`) |
| Find Diffraction Vectors | ✅ | Electron-native `find_vectors_action.py` reuses the Qt-free, memory-safe `_do_compute_vectors`; builds the vectors-image tree + attaches `diffraction_vectors`. **Staged wizard** (`FindVectorsWizard.tsx`, Qt parity): opening the caret starts a **LIVE found-peaks preview** on the source DP — `fv_preview` attaches `vector_overlay.FindVectorsPreviewOverlay`, which runs `_find_vectors_single_frame` on the nav-blurred frame under the crosshair (memory-safe: only a small `ceil(3σ)` nav window is sliced/computed). The σ / disk-radius / threshold / min-distance / subpixel sliders re-run the preview live via `fv_tune` (debounced), and move-the-navigator previews other patterns; **Compute** (`fv_run`) runs the full-dataset batch → vectors window + persistent red-marker overlay (the preview is dropped). `test_find_vectors_port.py` + `test_vector_overlay.py` (calibrated) + `test_find_vectors_wizard.py` (headless preview→tune→run) + `vector_overlay.spec.ts` (real-Dask: live preview red markers BEFORE compute, then Compute → vectors window) + `spyde.spec.ts` wizard UI + `sped_ag` benchmark |
| Vector Virtual Imaging | ✅ | Electron-native `vector_virtual_imaging.py` SUBCLASSES the raw `VirtualImageAction` (RegionAction template) — only `reduce` differs: it builds each image in-memory from the CSR buffer (`vecs.virtual_image_from_roi_gpu`, calibrated detector ROI converted from anyplotlib image-pixel coords), not a Dask reduction. Reuses the multi-VI sub-toolbar verbatim (same `type`/`calculation` chip schema → `ViShape`/`ViCaret`/`SubToolbar`); `parent_action` routes per-VI caret edits to the right bar. `test_vector_virtual_imaging.py` (disk/annulus/rect, intensity/count, live shape change) + `vector_vi_lazy.spec.ts` (real-Dask: vectors window → ＋ → VI output window) |
| Vector Orientation Mapping | ✅ | Electron-native `vector_orientation_om.py` MIRRORS the staged OM wizard: `vom_generate_library` (`generate_library_from_phases` + `build_template_library`, cached on `tree._vom_wizard`) → `vom_run` (`compute_vector_orientation` sparse-vector fit on a bg thread → IPF-Z orientation window + εxx/εyy/εxy strain windows, attaches `tree.vector_orientation`). Frontend `VectorOrientationWizard.tsx` (Load/Library/Run) copied from `OrientationWizard`; `FloatingToolbar` routes it via `WIZARD_ACTIONS` (and opens it by name even though the action carries no params). **Live refine overlay** (`vector_overlay.VectorOrientationOverlay`): generating the library activates a per-pattern pose fit (`fit_pattern` → `project_spots`) drawing the fitted template (GREEN) over the measured vectors (RED) under the crosshair, tracking the navigator (lock-serialised). `test_vector_orientation_om.py` (generate→run→4 windows + result; live-overlay markers; run-without-library guard) + `vector_om_lazy.spec.ts` (real-Dask Silver-CIF: Generate → green+red live refine on the DP → Compute → IPF+strain) + `spyde.spec.ts` wizard UI |
| IPF 2D/3D explorer toggle | ✅ | Every IPF orientation window (dense OM + Vector OM) gets a **2D ⇄ 3D toggle**. The 3-D view is the anyplotlib IPF explorer: `SpyDEOrientationMap.ipf_sphere_points()` returns per-position reduced crystal directions ON the unit sphere (`(quat·d).in_fundamental_sector(pg)`) + their IPF RGB; `ipf_view.emit_ipf_3d` builds an `ax.scatter3d(... colors=rgb).set_sphere()` figure and emits it as a SECOND figure for the same window tagged `view:"3d"`. Frontend (`MDIArea`) shows a `2D`/`3D` toggle when a window has a `view:"3d"` figure and switches which iframe is visible (both stay mounted). Reducer guards: a `view` figure must NOT overwrite the window title/aspect. `finalize_figure_html` extracted from `Plot` for shared dark/esm/focus post-processing. `test_ipf_3d.py` (unit-sphere points + figure + emit) + `orientation_lazy.spec.ts` / `vector_om_lazy.spec.ts` (toggle visible, switch to 3D) |

## Windowing / UX (nice Qt features to port)

| Feature                             | Status | Notes                                                                    |
|-------------------------------------|--------|--------------------------------------------------------------------------|
| MDI drag/resize/focus/z-order       | ✅      | react-rnd; z-order test in `spyde.spec.ts`                               |
| Tile / organize windows             | 🟡     | menu emits; layout not applied                                           | (This should be something that we can improve upon likely with things like sticking subwindows together)
| Frameless custom title bar          | ⬜      | Qt had WS_THICKFRAME polish                                              | (Don't Do)
| Signal-tree navigator switcher      | ✅      | dock switcher renders nodes; click → select_signal_node; `spyde.spec.ts` |
| Workflow view (signal-tree polish)  | ✅      | redesigned compact `TreeNodes` (depth rail, node dot, hover, ACTIVE-node highlight via `active_signal_id`); `Session._reemit_signal_tree` re-pushes after a transform so new steps appear (CZB "Centered"). `center_zero_beam.spec.ts` + `signal-tree switcher` test |
| Stick windows (edge-snap grouping)  | ✅      | drag → `snapToPeers` aligns edges; ~1.1s dwell while aligned → stick group (🔗 badge); dragging one stuck window moves the group (`onStuckMove`→`groupNudge`). **v2:** resizing a stuck window LINKS the shared dimension — height when joined side-by-side, width when stacked — and slides partners to follow the moving edge so they never overlap (`onStuckResize`→BFS over the group's edge-adjacency→`rectOverride`); **shaking** a window (≥5 rapid horizontal reversals) breaks the group apart (`onShake`). `MDIArea` owns peer geom + groups. `stick_windows.spec.ts` (snap → badge → group-move → linked-height resize/no-overlap → shake-to-break) |
| Axes table (click-to-edit) + shape in metadata | ✅ | `EditableCell` (text→input on click, commit on blur/Enter); size column dropped → dataset shape/dtype in Metadata `Dataset` section; edits re-push plots immediately. `test_axes_edit.py` + `selector.spec.ts` |
| IPF 2D/3D explorer toggle            | ✅      | per-window 2D⇄3D toggle; 3D = interactive `scatter3d` on the unit sphere (`ipf_sphere_points`). `orientation_lazy.spec.ts` / `vector_om_lazy.spec.ts`. Deferred: colour-key triangle legend |
| Per-VI colored ROI + compute/commit | 🟡     | template does live recompute; no commit button                           | (We can likely improve upon this with a more modern UI)
| Instrument-control (left) dock      | ⬜      | live microscopy panels                                                   |. (Don't Do)
| Workflow view                       | ⬜      |                                                                          | (Don't Do yet)
| Movie export                        | ⬜      |                                                                          | (Don't Do yet)
| Drag-and-drop navigator assignment  | ⬜      | Qt `NAVIGATOR_DRAG_MIME`                                                 |


## Known gaps in self-checking (improve next)

- No **live-interaction** visual tests yet (drag a selector → DP updates on
  canvas). Current visual tests are static-render + replay.
- No **A/B pixel diff** against the Qt app. Manual side-by-side for now. (This isn't necessary)
- ✅ DONE Perf: shared JS bundle — figure HTML 370KB→9KB, bundle written once,
  V8 code-cache reuse across iframes. `test_perf.py`.
- ✅ DONE anyplotlib resize triangle hidden (`resizable=False`); resize tied to
  the SubWindow (react-rnd) → resizeFigure IPC.
- ✅ DONE windows draggable by their title bar (`dragHandleClassName`).
  `spyde.spec.ts`.
- ✅ DONE dark-mode plots (dark page bg + nativeTheme). `visual.spec.ts`.
- ✅ DONE navigator selector toggle in the right dock (Point ↔ Integrate, only
  one shows). `test_selector_mode.py` + `spyde.spec.ts`.
- ✅ DONE MDI background grid removed. `spyde.spec.ts`.


ALWAYS add a playwright E2E test for any new visual feature, and a headless test for any new data flow.
This will catch regressions immediately in CI.


## Electron-specific features (not in Qt)

- The toolbar is currently embedded in the PlotWindow.  I like the QT floating toolbar that tracks the PlotWindow but:
    - It can be a bit annoying to have it floating around and sometimes it can get lost behind other windows.
    - Things like the Caret and Parameter popouts don't have a clear implementation in Electron.
    - There is an opportunity to improve the UX here with a more modern design that is better suited to the Electron environment.
      - We can explore options like a sidebar, a ribbon, or a context-sensitive toolbar that appears when needed.
      - I like the idea of a sidebar extending down or on the right side of the window maybe better than the caret. 
- I like the idea of being able to "Stick" subwindows together so that they move and resize as a group.
  - This would be a nice improvement over the QT MDI behavior.
  - Something like if you hold two windows together for a few seconds they will stick side by side or one on top of the other and then they will move together until you "unstick" them.

## Bugs

### ✅ Resolved
- Close button did nothing — backend emitted an unhandled `windows_closed`/`tree_id`;
  now emits `window_closed` per window id. `test_close.py` + `selector.spec.ts`.
- Close button scoping — navigator X closes the whole tree, a signal X closes only
  itself (cleans its selectors/source ROI). `test_close.py`.
- Navigator Selector toggle lingered after a tree closed — `WINDOW_CLOSED` drops all
  per-window state. `selector.spec.ts`.
- Axes not editable in the sidebar — editable axes table (name/scale/offset/units →
  axes_manager, recalibrates). `test_axes.py` + `selector.spec.ts`.
- Examples loaded into memory (Dask unused) — `Session._to_lazy` persists to a temp
  .zspy + reloads lazily (frees RAM, compute via Dask). `test_lazy_loading.py`.
- Navigator-on-load chunked display — window appears ~0.4 s and fills chunk-by-chunk
  (lazy path / `_start_progressive_nav_compute`). Verified on a real cluster.
- Virtual image "just black" — two bugs: ROI was always green (`color` not forwarded to
  the selector `super()`), and on the LAZY path the output figure was built on the worker
  thread AFTER compute so its single push raced the iframe load. Fix: forward `color`;
  push the placeholder in `RegionAction.run` so the figure/iframe loads early.
  `vi_lazy.spec.ts` (real Dask) + `selector.spec.ts` (eager) assert the canvas is non-black.
- "Plots get highlighted" — global `user-select:none`; half-light plot — `#widget-root`
  dark so anyplotlib's `_isDarkBg` reads dark.
- **Lazy navigator → signal flashed / inconsistent** — the optimized distributed →
  shared-memory → plot pipeline is KEPT (Qt parity; race-safe via the
  `_on_plot_ready` latest-future guard + cancel-stale). The flashing was two unrelated
  bugs: (1) every nav frame re-auto-leveled (`update()` passed no levels, `_set_array`
  re-leveled whenever `levels is None`) so brightness jumped → now HOLD `_last_levels`
  across nav frames, re-level only on explicit auto-level / user clim
  (`test_contrast_hold.py`); (2) the navigator x/y swap showed the wrong/clamped frame
  while dragging (see the pixel→real-space row).
- **Figures cut off until resized** — the figure was created at anyplotlib's default
  (~640×480) inside a smaller window. Now the iframe reports its real box on load and
  the figure is resized to it (`MDIArea` onLoad → `resizeFigure`). Default windows are
  squarer (400×392) so square diffraction patterns / navigators sit tight.
- **Units rendered as raw LaTeX** (`$A^{-1}$`) — `_clean_units` normalises to `Å⁻¹`
  (strips `$`/braces, maps superscripts). `test_units.py`.
- **Non-square navigator compressed into a strip / crosshair misaligned** — a wide scan
  (sped_ag 208×64, nav scale 20.3) was aspect-letterboxed by anyplotlib into a centred
  strip while the axis ticks + crosshair spanned the full panel → the selector didn't
  line up with the image. Fix: the navigator figure reports its image aspect
  (`nav_shape[0]/nav_shape[1]`); the frontend sizes the window to it so the image fills
  (no letterbox, aligned selector). `SubWindow` adopts the size unless the user resized.
  `test_nav_aspect.py` + `spyde.spec.ts` "wide navigator is sized to its image aspect".
- **Lazy nav drag froze until release** — the nav→signal slice forced EVERY frame through
  the async future/worker round-trip (`return_future=True`); each was superseded before
  it resolved and dropped by the latest-future guard, so only the final frame painted.
  Fix: `_get_cache_dask_chunk(get_result=False, return_future=False)` → in-chunk (cached)
  frames return numpy and display synchronously (live drag); only cross-chunk misses go
  async through the shared-memory path. Verified with a real-cluster repro.
- **`_on_signal_ready` crashed** ("'Future' object has no attribute 'parent_selector'") —
  the worker ignored its `extra` arg and passed the future where the plot was expected.
  Fixed to emit `(signal, result, plot)`.
- **Navigator drag flooded Dask** — the selector update was a 2 ms DEBOUNCE → ~60
  computes/sec during a drag, clogging the cluster (the new chunk stuttered). Now a
  THROTTLE (`update_data` coalesces a burst into one fire per `live_delay`, min 40 ms →
  ~25/sec). `test_throttle.py`.
- **Virtual image looked wrong as the detector ROI moved** — the navigator contrast-HOLD
  (added for the DP) wrongly applied to OUTPUT plots too, so the VI kept the first
  frame's levels. Now contrast holds only for *navigated* frames
  (`current_signal.navigation_dimension > 0`); VI / FFT / line outputs re-auto-level each
  recompute. `test_contrast_hold.py`. Also: a lazy VI is a full-dataset reduction, so
  dragging the ROI now CANCELS the superseded compute before submitting the next.
- **OM verified on lazy E2E** — `test_orientation_overlay::test_orientation_end_to_end_on_lazy_data`
  (headless lazy) + `orientation_lazy.spec.ts` (real-Dask Playwright, `run_test_orientation`
  test action drives OM with a built-in Al phase → IPF-Z window opens). Find Vectors lazy
  E2E is `vector_overlay.spec.ts` (real-Dask, `load_test_data_lazy`): opening the staged
  wizard shows a LIVE found-peaks preview (red markers) BEFORE Compute, then Compute opens
  the vectors window + persistent overlay.
- **Virtual image black on REAL (calibrated) data** — ROOT CAUSE found in a self-audit.
  anyplotlib overlay widgets report `cx/cy/r` in *image-pixel* coords (extent only
  relabels ticks; see `_imgToCanvas2d`), but `masks._signal_k_grids` built the detector
  mask grid in *physical* units (`pixel*scale + offset`). On any calibrated axis
  (scale != 1) the ROI never overlapped the grid → empty mask → VI all zeros (black).
  Every synthetic test used scale=1, which hid it. Fix: build the mask in pixel space.
  `test_masks.py` (scale=0.1, asserts non-empty mask + calibration-invariance) and the
  lazy loader is now CALIBRATED so `vi_lazy.spec.ts` guards the visual case too.
- **Find Vectors / Orientation didn't overlay** — the ports only opened a result window;
  the Qt scatter overlay on the live DP was missing. Added `BaseSelector.index_hooks`
  (fired on every navigation change) + `actions/vector_overlay.py` (`VectorOverlay`,
  `OrientationOverlay`) that convert calibrated kx,ky → pixels and re-push markers as the
  navigator moves. `test_vector_overlay.py`, `test_orientation_overlay.py`,
  `vector_overlay.spec.ts`.

### ✅ Resolved (UI polish batch)
- Scale section → small min/max labels on the histogram (`clim-min`/`clim-max`). `spyde.spec.ts`.
- Metadata panel is now a 2-column grid.
- Histogram min/max lines thicker (w3 + grip caps) and taller (H 60→84).
- Orientation Mapping caret has tabs (generic `tab` param field; Library / Matching).
  `spyde.spec.ts`.
- Clicking inside a figure raises its window — the iframe swallows mousedown, so we raise on
  iframe-focus (window blur → activeElement). `spyde.spec.ts` (click-to-front).
- SubWindow creation perf: investigated — backend is ~18 ms (2 windows), shared-ESM HTML is
  ~10 KB, and a WARM window appears in ~50 ms (canvas ~80 ms). The cold cost is anyplotlib's
  first-figure init (~120 ms) — moved off the critical path with a startup `_prewarm_anyplotlib`.

- Sub-toolbars + caret move with the PlotWindow — they're parented to the window, and the
  open caret now STAYS open during a drag (click-away ignores mousedowns within the same
  subwindow) and tracks the window. `spyde.spec.ts` (caret-stays-open-while-dragging).

### ⬜ Open
- Caret group box: fine when wide, but may also extend vertically as long as things are
  grouped nicely. (Partly done — popout wraps to 2 rows + tabs; revisit if more grouping wanted.)



## To Do Overnight

Run this overnight and check off the remaining items. The goal is to make sure that you can run autonomously so really
focus on testing.  Especially focus on the visual tests and the A/B comparisons with the Qt app.  If you find any
discrepancies, make sure to document them clearly and investigate the root cause.  This is a critical step to ensure
that we have achieved parity between the two applications. Test using the sped_ag dataset as much as possible with 
dask.  It's okay to be slow and run longer tests but make sure any hanging electron processes are killed before you
start the next test to avoid interference.  If you find any issues with the tests themselves (e.g. they are flaky, 
or they are not actually testing what they claim to be testing), make sure to document that as well and propose improvements.

Create a .md and checklist if needed. Spawn extra agents to review if that helps keep on track.

Ask questions first.

- [] Improve the aesthetics. 
  - [ ] The Axes table should be more visually appealing. The qt application had nice text which when clicked on would allow you to edit. The size of the dataset should be in the metadata so that isn't needed. 
  - [ ] The any change to the axes should be reflected in the plot immediately. 
  - [] Implement the "Stick" feature for subwindows. This would allow users to group windows together so that they move and resize as a group.
    - [] When two windows edges are aligned for a few seconds they will "stick". Make sure you test various window sizes
    - [] Test window resizing and make sure the plots are resized correctly. 
- [] Feature Improvement:
  - [] The center zero beam feature needs to be improved. Have two tabs, automatic and manual.
      - [] Automatic: This will automatically find the zero beam position and center it. Use existing QT code and reach parity. 
      - [] Manual: This will allow the user to manually select the zero beam position. This can be done by moving a crosshair to the desired position.
- [] Workflow Improvement:
  - [] Implement a workflow view that allows users to see the steps they have taken and easily navigate back to previous steps. This could be a sidebar that shows a list of actions taken, with the ability to click on an action to revert to that state.
  - [] This is like the "Signal Tree Viewer" but it should be a little smoother and not as big.  The signal tree viewer is a good start but it can be improved to be more user-friendly and visually appealing. It should also allow users to easily navigate through their workflow and see the relationships between different steps.
  - [] The ipf explorer toggle is a good start but it can be improved to be more intuitive and visually appealing. It should allow users to easily switch between 2D and 3D views of the IPF data, and the 3D view should be interactive and allow users to explore the data in more detail.
  - [] Make sure multiple phases are supported in the orientation mapping workflow. pyxem has simulated test data https://pyxem.org/v0.21.0/examples/orientation_mapping/multi_phase_orientation.html#sphx-glr-examples-orientation-mapping-multi-phase-orientation-py which is useful. 
  - [] Make sure anyplotlib can actually plot a nice 2d ipf map.
- [] Review the codebase and identify any areas that can be refactored or improved for better performance and maintainability. This could include things like optimizing data flow, improving the structure of the code, or adding comments and documentation to make it easier to understand.
- [] Review the tests and identify any gaps in coverage or areas where the tests can be improved for better reliability and accuracy. This could include adding more test cases, improving the assertions, or refactoring the tests to be more maintainable.


TODO:
I think we need to find a good way to handle different navigation views effectively.  In the Qt version there was this drag/ drop approach which kind of was rough.  I like the toggle 2D/3D option and would LOVE to extend that.  That means that the different navigators loaded would all be different tab options.  You could either have the option to toggle the view (like the 2D <-> 3D) or toggle between views like the VDF VBF.  Or the different strain mesurements or the IPF X, IPF Y and IPF Z.  It would also be amazing to be able to visualize multiple at once. So if you select multiple images they tile nicely, have linked selectors so that is duplicated.  It's a very nice refinement of the QT version in a unified and reproducible fashion.  It needs to be elegant.  The code needs to be clean and reusable.  This is likely two slightly different things 1. a view selecting option like 2/3D 2. Swaping virtual images or reconstructions but possibly they could be combined.  
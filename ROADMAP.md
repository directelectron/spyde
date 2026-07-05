# SpyDE Roadmap & Improvement Audit

_Last updated: 2026-06-03_

---

## Current Feature Status

| Feature | Status | Notes |
|---|---|---|
| MDI window system | Mature | Navigator + signal pair, multi-panel |
| Signal tree navigation | Mature | Branching tree, toggle between nodes |
| Center Zero Beam | Mature | 5 methods + manual Shift+click picker |
| Virtual Imaging | Mature | Multi-ROI (annular/disk/rect), live preview, batch |
| Find Diffraction Vectors | Beta | Template match, NavBlurCache, flat-buffer storage |
| Orientation Mapping | Beta | 5-tab wizard; CIF loader partially incomplete |
| Line Profile | Mature | 1D extraction + navigator strip sum |
| Rebin / FFT | Mature | Wired, tested |
| Strain Mapping | Not exposed | Gateway code exists (`vecs.get_strain_maps()`); no UI |
| Vector Clustering | Not exposed | DBSCAN gateway exists; no UI |
| Azimuthal Integration | Not exposed | PyXEM `get_azimuthal_integral2d` not wired |
| Radial Integration | Not exposed | PyXEM `get_radial_profile` not wired |
| Phase / Grain Mapping | Partial | OM wizard produces phase index; no false-color map |
| Pole Figures | Partial | IPF heatmap only; no stereographic projection |
| EELS | Not planned | Out of scope for initial roadmap |
| Project Save / Load | Not implemented | |
| Undo / Redo | Not implemented | Signal tree is append-only |
| Custom colormaps | Not implemented | 5 hard-coded maps |

---

## Known Issues & Code Quality

### Print statement spam
Every major code path (`pyxem.py`, `plot.py`, `signal_tree.py`) uses `print()` for debug output. This floods stdout and makes profiling harder. Replace with `logging.getLogger(__name__)` throughout.

### Global toolbar guards — RESOLVED
The `_FV/_OM/_CZB_BUILT_TOOLBARS` module-level sets are gone. Per-window state
now lives in `session._window_controllers` / `spyde.actions.figure_registry`
(evicted by `_forget_window`), and wizard double-fire is handled by the
run/stop generation guard in `spyde.actions.lifecycle`. See
`spyde/actions/README.md` for the action framework.

### Shared memory disabled on Windows
`_SHARED_MEMORY_SUPPORTED = sys.platform != "win32"` in `update_functions.py`. Dask workers on Windows are separate processes and cannot share GUI-process memory. Data transfers fall back over TCP, which is slower. Worth benchmarking whether `threads` scheduler is faster than `processes` for the typical 4D STEM sizes users encounter.

### Magic numbers scattered throughout
- `ZOOM_STEP = 0.8` (base.py)
- `CROSSHAIR_SCREEN_PX = 10` (selectors)
- Debounce: 150 ms virtual image, 50 ms find vectors (inconsistent)
- Polar grid: `NR=100, NA=360` (orientation mapping)
- Color tuples hardcoded in virtual imaging

These should live in a single `constants.py` or be exposed as user preferences.

### Incomplete error surfacing
Most background-thread exceptions are caught and printed or put in a status label, but never raised as `QMessageBox.critical()` to the user. The orientation mapping wizard is the exception — it uses the message box correctly. Standardise this.

### Scale bar 2D only
`plot.py` has a TODO at line 344: scale bar is only drawn for 2D plots. 1D plots have no axis annotation.

### Navigator assumes 2D nav grid
`_preprocess_navigator` and most selector code assumes `navigation_dimension == 2`. 5D signals (time + 2D nav) work partially, but the time axis is never exposed as a controllable dimension in the UI.

---

## PyXEM Capability Roadmap

PyXEM's example gallery groups capabilities into the categories below. Each section describes what the tool does, what a good SpyDE implementation would look like, and what tests are needed.

---

### 1. Preprocessing

These are non-breaking transformations that refine the raw data before any analysis. They should all use `signal_tree.add_transformation()` and show up as children of the current node — not as new root trees.

#### 1a. Filtering
**PyXEM API**: `signal.gaussian_filter(sigma)`, `signal.median_filter()`, `signal.apply_filter()`

**UI**: Caret popout on the bottom toolbar with σ slider (for Gaussian) or kernel-size spinner (for median). Live preview by calling the filter on `plot.current_data` and calling `plot.image_item.setImage()` directly, then committing to the tree on Apply.

**Visualization**: Difference overlay (filtered − original) as a toggle to show what was removed.

**Tests**: Filter reduces noise metric; tree node is created; revert restores original.

#### 1b. Dead Pixel / Background Masking
**PyXEM API**: `signal.subtract_diffraction_background()`, circular/annular masks

**UI**: Paint-brush ROI tool or threshold-based auto-masker. Show the mask as a semi-transparent red overlay on the diffraction pattern.

**Tests**: Masked pixels are set to zero/NaN; navigator is unaffected.

#### 1c. Normalize Intensity
**PyXEM API**: `signal.normalize_intensity()`

**UI**: Simple Apply button. Optionally expose per-frame vs. global normalisation.

---

### 2. Azimuthal & Radial Integration

This converts 2D diffraction patterns into 1D profiles and is fundamental to STEM and EM powder diffraction.

**PyXEM API**:
- `signal.get_azimuthal_integral1d(npt, ...)` — 1D radial profile (q vs. intensity)
- `signal.get_azimuthal_integral2d(npt_rad, npt_azim, ...)` — 2D polar map (q vs. φ)
- `signal.get_radial_profile()` — simplified radial mean

**UI**:
- Bottom toolbar toggle "Azimuthal Integration"
- Caret: center (pre-filled from Center Zero Beam result), inner/outer q range, number of points, polarisation correction toggle
- Live preview: show the resulting 1D profile in a new subwindow that updates as the navigator moves (same pattern as virtual imaging)
- The 1D `Signal1D` result should be added as a child node in the signal tree (non-breaking transformation when nav dims are preserved)

**Visualization**:
- The 2D (q, φ) map is a powerful texture diagnostic — show as a pyqtgraph ImageItem with axis labels "q (Å⁻¹)" and "φ (°)"
- The 1D profile should overlay calibration markers (known d-spacings from a reference phase loaded in OM wizard)
- The navigator plot for the 1D result should show the azimuthally-averaged signal summed across q — gives a meaningful spatial map

**Tests**: Integration result shape = (nav_y, nav_x, npt); 1D profile peaks at correct q for a synthetic ring pattern.

---

### 3. Diffraction Vector Analysis (extending current Find Vectors)

The current `Find Diffraction Vectors` tool produces `SpyDEDiffractionVectors`. The downstream steps are:

#### 3a. Vector Clustering (DBSCAN)
**PyXEM API**: `vectors.cluster_vectors(distance_threshold)` → unique vectors with labels

**UI**: Toolbar action on the vectors result node. Caret: ε (distance), min_samples. Visualize the clustered vectors as a scatter plot in kx–ky space, coloured by cluster label. Use a qualitative colormap (tab10) for cluster IDs.

**Key visualization**: A standalone pyqtgraph scatter plot subwindow showing all `(kx, ky)` points from the entire dataset, coloured by cluster. This is the primary diagnostic for checking whether vectors represent real crystal reflections or noise.

**Tests**: N known vectors at N positions cluster into N groups with ε < inter-spot distance.

#### 3b. Unique Vectors & Spot Library
**UI**: After clustering, show the unique vectors overlaid on the mean diffraction pattern. Let the user manually delete outliers or merge close clusters. Export the resulting spot library as a reference for strain mapping.

#### 3c. Strain Mapping
**PyXEM API**: `vectors.get_strain_maps(reference_vectors, ...)` → `Signal2D` of ε_xx, ε_yy, ε_xy, θ

**UI**: "Strain Mapping" action on the vectors node. Caret: reference position picker (click a nav position to use its vectors as reference, or load from file), affine decomposition method. Compute button runs the batch calculation.

**Visualization**: This is the richest output in the whole pipeline:
- Four `Signal2D` maps: ε_xx, ε_yy, ε_xy, rotation θ — each as its own plot window
- A **diverging colormap** (RdBu) is essential for strain maps — positive/negative strain must be visually distinguishable. Currently SpyDE has no diverging colormap.
- A **vector field overlay** (quiver arrows) showing the local rotation θ on the navigator
- **Color scale locked across all four maps** (same min/max) for direct comparison

**Tests**: Known affine deformation → recovered strain tensor within tolerance.

#### 3d. Virtual Dark Field from Vectors
**PyXEM API**: `vectors.get_virtual_dark_field_images()` — integrate intensity around each unique vector position

**UI**: Add as a subfunction of the vectors toolbar. Shows one VDF image per unique vector, navigable as a stack.

---

### 4. Orientation Mapping (extending current OM wizard)

The current wizard handles template matching and IPF visualization. What's missing:

#### 4a. Pole Figures & Stereographic Projections
**PyXEM API**: `orientation_map.to_single_phase_orientations()` → orix `Rotation` → pole figure via `orix.plot`

**UI**: A new subwindow containing a stereographic projection rendered as a pyqtgraph `ScatterPlotItem` or `ImageItem` (pre-rendered via matplotlib → QPixmap, or a custom pyqtgraph implementation). Points are coloured by frequency. A selection ROI on the pole figure should highlight the corresponding nav positions.

**Visualization requirements**:
- Reference frame triangle (001, 101, 111) drawn as lines
- Scatter points sized by frequency, coloured by orientation group
- Bidirectional: selecting a region of the navigator should highlight points in the pole figure, and vice versa

#### 4b. Phase Maps
**UI**: After multi-phase OM, the phase index signal (0, 1, 2...) should be visualized with a discrete colormap — one colour per phase. Show a legend widget with phase name + colour. Allow toggling individual phases on/off.

#### 4c. Reliability / Confidence Metrics
**PyXEM API**: The correlation index from template matching gives a confidence score. Map it as a Signal2D — low confidence regions should be treated with caution.

**UI**: Show the correlation score as an overlay (semi-transparent grey where score < threshold). Let the user set the threshold with a slider.

#### 4d. Grain Boundary Detection
After orientation mapping, neighbouring pixels with misorientation > threshold are grain boundaries.

**UI**: Overlay coloured lines (white) on the IPF map at boundary positions. Caret slider for misorientation threshold.

---

### 5. Virtual Imaging Extensions

The current virtual imaging is solid. Additions:

#### 5a. Segmented Detectors (HAADF, ABF, iDPC, dDPC)
**PyXEM API**: `signal.get_integrated_differential_phase_contrast()` — splits annular detector into quadrants, computes differential signal

**UI**: New detector type "Segmented" in the virtual imaging submenu. Shows 4 quadrant sectors + DPC result. The DPC vector field should be displayed as a colour wheel map (hue = direction, brightness = magnitude) — requires a new HSV colormap mode.

#### 5b. 4D-STEM Centre of Mass (CoM)
**PyXEM API**: `signal.center_of_mass()` → two Signal2D maps (CoM_x, CoM_y)

**UI**: Button action that computes and opens two nav-space maps. Optionally display as a vector field overlay on the navigator.

#### 5c. Moment Maps
**PyXEM API**: `signal.get_variance()`, `signal.get_direct_beam_position()`

**UI**: Single-click actions that add moment maps as navigator signals in the current tree.

---

### 6. Visualization Infrastructure Improvements

These are cross-cutting improvements that benefit all features.

#### 6a. Diverging & Qualitative Colormaps
Strain maps require diverging colormaps (RdBu, coolwarm). Phase maps require qualitative colormaps (tab10, Paired). Currently only sequential maps exist.

**Implementation**: Add to `colormaps.py`:
- `"RdBu"` — diverging, centred at zero
- `"coolwarm"` — diverging, perceptually uniform
- `"tab10"` — 10-colour qualitative (for phases, clusters)
- `"hsv"` — for DPC colour wheel

Expose in the histogram widget colormap picker dropdown.

#### 6b. Locked Colour Scales Across Windows
For comparing strain components or phase-channel images, the min/max of the colour scale should be linkable across plot windows. Add a "Link contrast" toggle in the PlotControlToolbar.

#### 6c. Annotations & Measurements
- Distance measurement tool: click two points on a diffraction pattern → display Δ(kx, ky) and |k| in the status bar, convertible to d-spacing
- Angle measurement: three-point angle tool
- These are pure pyqtgraph `InfiniteLine` + `TextItem` overlays

#### 6d. kx–ky Scatter Plot Window
A dedicated 2D scatter plot in reciprocal space (not a nav-space image) for showing vector distributions, cluster labels, and unique reflections. Should support:
- Zoom / pan
- Colour by cluster label, intensity, or nav position
- Overlay of the simulated diffraction library positions (from OM wizard) as open circles

#### 6e. 1D Profile Overlays
When the line profile tool is active, draw a vertical dashed line on the 1D profile plot that follows the cursor position on the diffraction pattern, and vice versa. This makes it easy to identify peak positions.

#### 6f. Axis Label Improvements
- 1D signal plots should show axis tick labels and units (currently hidden)
- The scale bar on 2D plots should show units dynamically (Å⁻¹, nm, px)
- Navigator plots should show real-space units (nm, Å, µm)

---

### 7. Signal Tree UI

The signal tree is only navigable via small caret buttons — there is no visual representation of the tree structure.

**Planned**: A dock widget (or expandable panel in the existing Plot Control dock) showing the tree as a collapsible `QTreeWidget`. Each node shows: signal name, shape, transformation applied, and has a "Switch to" button. This replaces the current navigate-signal-tree button approach for deep trees.

---

### 8. Infrastructure

#### 8a. Replace print() with logging
Use Python's `logging` module throughout. Add a log-level selector to a settings dialog. This also makes test output readable.

#### 8b. Toolbar Guard Cleanup
Replace module-level `_*_BUILT_TOOLBARS` sets with per-toolbar attributes (`toolbar._fv_state` already exists; the guard should check that, not a global set).

#### 8c. Constants Module
Centralise all magic numbers into `spyde/constants.py`.

#### 8d. Project Save / Load
Serialize the signal tree to `.hspy` (hyperspy native) for each node, plus a JSON manifest describing the tree structure, node names, and transformation parameters. Load restores the tree and plot windows.

#### 8e. Settings Dialog
Expose: default colormap, debounce intervals, max RAM for batch compute, Dask worker count.

---

## Priority Order

### Near-term (next 1–3 months)
1. **Azimuthal integration** — highest-frequency user request for powder and amorphous STEM
2. **Diverging + qualitative colormaps** — needed by strain and phase tools; small implementation effort
3. **Vector clustering UI** — infrastructure already exists; just needs a toolbar action and scatter plot window
4. **Replace print() with logging** — pure cleanup, improves professionalism
5. **Fix toolbar guard globals** — correctness issue for power users who close/reopen plots

### Medium-term (3–6 months)
6. **Strain mapping UI** — gateway code exists; needs toolbar action, reference picker, and colour-scale linking
7. **kx–ky scatter plot window** — shared by clustering, unique vectors, and OM pole figures
8. **Segmented detector / DPC** — high scientific value for iDPC imaging
9. **Pole figures** — completes the OM workflow
10. **Phase map false-colour** — completes multi-phase OM output

### Longer-term (6–12 months)
11. **CoM / moment maps** — straightforward PyXEM calls, needs nav-space result display
12. **Signal tree dock widget** — quality-of-life improvement for complex processing chains
13. **Grain boundary overlay** — extends OM; requires misorientation calculation
14. **Project save / load** — enables reproducible workflows
15. **EELS support** — requires its own toolbar section and 1D-centric UI design

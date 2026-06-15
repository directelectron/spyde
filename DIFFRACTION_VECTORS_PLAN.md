# Diffraction Vector Finding — Feature Spec & Implementation Plan

## Overview

Add a **Find Diffraction Vectors** workflow to SpyDE that:

1. Provides a caret popout with real-space Gaussian blur (σ), disk kernel radius (linked to a draggable `CircleROI`), threshold, min-distance separation, and subpixel refinement toggle — all auto-populated from the current diffraction pattern.
2. Creates a live preview window showing the processed image (Gaussian → window-normalized cross-correlation → thresholded) with vector markers overlaid on both the transformed image and the raw diffraction pattern.
3. On "Compute", produces a new signal tree node whose rendering overlays circles (radius = kernel radius) with `+` centers on the parent diffraction pattern.
4. Backs the vectors in a flat-buffer nested-tensor layout (like PyTorch NestedTensor) as `SpyDEDiffractionVectors`, gating strain mapping, virtual imaging through vectors, and density-based clustering.

---

## Architecture Overview

```
ElectronDiffraction2D (4D-STEM: nav=[y,x], sig=[ky,kx])
  │                   or 5D-STEM: nav=[time,y,x], sig=[ky,kx]
  │
  ├── [Centered]          ← existing node
  │     │
  │     └── [Diffraction Vectors]  ← NEW node (SpyDEDiffractionVectors)
  │           │  rendering: circles(r=kernel_r) + '+' markers overlaid on parent
  │           │  data: flat buffer + offsets (CSR / PyTorch NestedTensor layout)
  │           │
  │           ├── [Strain Maps]       ← existing pyxem workflow, now gated here
  │           ├── [Virtual Images]    ← vector-based virtual image creation
  │           └── [Cluster Analysis]  ← DBSCAN / HDBSCAN on vector positions
```

---

## Timing Budget (benchmarked on development machine)

### Live Preview (per frame, warm cache)

| Operation | 128×128 sig | 256×256 sig |
|---|---|---|
| Nav blur — `NavBlurCache` lookup (cached chunk) | **~0 ms** | **~0 ms** |
| Nav blur — single-frame fallback (cold chunk) | ~0.4 ms | ~1.2 ms |
| `match_template` disk r=10 | ~1.5 ms | ~5.6 ms |
| `peak_local_max` | ~2.0 ms | ~6.5 ms |
| Subpixel CoM refinement | ~0.1 ms | ~0.1 ms |
| **Total (warm)** | **~4 ms** | **~13 ms** |
| **Total (cold / first frame after chunk load)** | **~4 ms** | **~14 ms** |

Live preview target: **<20 ms per frame** on a 256×256 pattern → 50 fps achievable on CPU.

### NavBlurCache background cost (one-time per chunk change)
| Chunk size | Signal size | Pad+blur time (async background) |
|---|---|---|
| 16×16 nav | 128×128 sig | ~120 ms |
| 16×16 nav | 256×256 sig | ~490 ms |
| 32×32 nav | 128×128 sig | ~400 ms |

These run in a daemon thread triggered by chunk-load events; the UI never waits for them.

### Batch Compute (16×16 nav, 128×128 sig)
- Nav blur (`map_overlap` on full dataset): ~620 ms
- Template match + subpixel (256 patterns): ~790 ms
- Flat buffer assembly: ~1 ms
- **Total**: ~1.4 s

---

## Part 1: The Real-Space Gaussian Blur — Two Paths

The nav-space Gaussian blur has two completely different implementations depending on whether it's serving the **live preview** (single frame, fast) or the **batch compute** (full dataset, correct at all boundaries).

### 1.1 How SpyDE's Plot System Already Loads Data

`CachedDaskArray` (in `hyperspy._signals.lazy`) is the chunk-caching layer that `update_from_navigation_selection` calls via `_get_cache_dask_chunk`. It maintains:

- **`core_cached_blocks`**: numpy arrays of the current navigation chunk(s) — already in memory
- **`surrounding_cached_blocks`**: numpy arrays of adjacent chunks — pre-fetched in the background (`cache_padding=1` when a Dask client is running)

This means when the user moves the navigator, the neighboring patterns are already resident in memory with **zero disk I/O**. The live preview can exploit this directly.

### 1.2 Live Preview Fast Path — `NavBlurCache`

Instead of blurring a single frame (which ignores neighbors) or triggering a full `map_overlap` compute (which re-reads from disk), the live preview uses a **`NavBlurCache`** that piggybacks on the existing chunk cache.

**Algorithm:**

```
On chunk change (new chunk loaded by CachedDaskArray):
  1. Fetch the raw chunk numpy array: shape (chunk_ny, chunk_nx, ky, kx)
  2. Reflect-pad by depth=ceil(3σ) in both nav dims:
         padded shape = (chunk_ny + 2·depth, chunk_nx + 2·depth, ky, kx)
  3. Apply gaussian_filter(padded, sigma=(σ, σ, 0, 0)) in a daemon thread
  4. Trim: blurred_chunk = blurred_padded[depth:-depth, depth:-depth]
     -> shape (chunk_ny, chunk_nx, ky, kx), correct at ALL positions

On nav position change (within cached chunk):
  if blurred_chunk ready:
      return blurred_chunk[iy_local, ix_local]   # O(1), zero copy
  else:
      # Cold: chunk just loaded, blur not done yet
      # Fallback: gaussian_filter on the single pattern only
      return gaussian_filter(raw_pattern, sigma=(σ, σ))  # ~1.2ms for 256×256
```

**Why this is correct**: The reflect-pad ensures that even edge patterns of the chunk see real (reflected) neighbor data rather than a hard boundary. The interior patterns see actual neighboring patterns from the chunk. Cross-chunk boundary accuracy is limited to reflection artifacts, which are negligible for the param-tuning use case.

**Why the async blur doesn't block the UI**: It runs in a daemon thread. For the first few nav moves after a chunk boundary (while blur is computing), the single-frame fallback takes ~1.2ms — fast enough that the user won't notice.

```python
# spyde/actions/find_vectors.py

class NavBlurCache:
    """
    Async per-chunk Gaussian blur cache for live diffraction vector preview.

    Hooks into the chunk-loading lifecycle:
    - Call update_chunk(chunk_array, chunk_id) when a new dask chunk becomes available.
    - Call get_blurred(iy_local, ix_local) to retrieve the nav-blurred pattern.
    """

    def __init__(self, sigma: float):
        self.sigma = sigma
        self._depth = int(np.ceil(3 * sigma))
        self._blurred: Optional[np.ndarray] = None   # (cy, cx, ky, kx)
        self._raw_chunk: Optional[np.ndarray] = None # (cy, cx, ky, kx)
        self._chunk_id: Optional[tuple] = None
        self._blur_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def update_chunk(self, chunk_array: np.ndarray, chunk_id: tuple):
        """Call when CachedDaskArray loads a new chunk. Starts async blur."""
        with self._lock:
            if chunk_id == self._chunk_id:
                return  # already have this chunk
            self._chunk_id = chunk_id
            self._raw_chunk = chunk_array
            self._blurred = None

        # Cancel previous blur thread (it will check chunk_id and exit early)
        t = threading.Thread(target=self._do_blur, args=(chunk_array, chunk_id), daemon=True)
        self._blur_thread = t
        t.start()

    def _do_blur(self, chunk_array: np.ndarray, chunk_id: tuple):
        from scipy.ndimage import gaussian_filter
        d = self._depth
        # Reflect-pad in nav dims so edge patterns see real (mirrored) neighbors
        padded = np.pad(chunk_array, ((d, d), (d, d), (0, 0), (0, 0)), mode='reflect')
        blurred_padded = gaussian_filter(padded, sigma=(self.sigma, self.sigma, 0, 0))
        trimmed = blurred_padded[d:-d, d:-d]
        with self._lock:
            if self._chunk_id == chunk_id:  # still the active chunk
                self._blurred = trimmed

    def get_blurred(self, iy_local: int, ix_local: int, raw_pattern: np.ndarray) -> np.ndarray:
        """
        Return nav-blurred pattern at (iy_local, ix_local).
        Uses cached blurred chunk if ready; falls back to single-frame blur.
        """
        from scipy.ndimage import gaussian_filter
        with self._lock:
            blurred = self._blurred
        if blurred is not None:
            return blurred[iy_local, ix_local]
        # Cold fallback: single-frame blur (ignores true neighbors, ~1.2ms)
        return gaussian_filter(raw_pattern, sigma=(self.sigma, self.sigma))

    def invalidate(self, sigma: float):
        """Call when σ changes; clears cache and updates sigma."""
        with self._lock:
            self.sigma = sigma
            self._depth = int(np.ceil(3 * sigma))
            self._blurred = None
            self._chunk_id = None
```

**Hooking into the chunk lifecycle**: The `NavBlurCache.update_chunk()` is called from the live-preview `_do_refit()` function. The current chunk can be obtained by accessing `signal.cached_dask_array` and checking which blocks are currently in `core_cached_blocks`. Since this is internal to hyperspy, the simpler approach is to call `signal._get_cache_dask_chunk(current_nav_indices)` which returns (or triggers) the chunk load, then read `signal.cached_dask_array.core_cached_blocks[0]` as a numpy array.

### 1.3 Batch Compute Path — `map_overlap` (Correct for Full Dataset)

The batch compute path uses the standard `dask.array.map_overlap` approach. This is correct at all boundaries (including dataset edges) because `map_overlap` loads ghost zones from disk before processing each chunk.

**Why `map_blocks` is wrong** — verified experimentally: with a spike at nav position [4,0] and a chunk boundary at row 4, `map_blocks` gives 0.43 at the neighbor cell [3,0]; `map_overlap` gives 7.06, matching scipy's reference of 7.06 on the full array.

```python
import dask.array as da
from scipy.ndimage import gaussian_filter

depth_px = int(np.ceil(3 * sigma_nav))
blurred = da.map_overlap(
    gaussian_filter,
    da_data,                          # (nav_y, nav_x, ky, kx)
    depth=(depth_px, depth_px, 0, 0), # ghost zones only in nav dims
    boundary='reflect',
    sigma=(sigma_nav, sigma_nav, 0, 0),
    dtype=np.float32,
)
```

Memory cost per chunk with ghost zones — `(C_y + 2·depth) × (C_x + 2·depth) × ky × kx × 4 bytes`:

| σ | depth | Chunk | Padded size | RAM/chunk (256×256 sig) |
|---|---|---|---|---|
| 1.0 | 3 | 32×32 | 38×38 | 150 MB |
| 1.5 | 5 | 16×16 | 26×26 | 176 MB |
| 2.0 | 6 | 16×16 | 28×28 | 204 MB |

```python
def _nav_chunk_size(sigma: float, max_ram_mb: float = 200, sig_shape: tuple = (256, 256)) -> int:
    depth = int(np.ceil(3 * sigma))
    sig_pixels = sig_shape[0] * sig_shape[1]
    max_padded = int(np.sqrt(max_ram_mb * 1e6 / (sig_pixels * 4)))
    return max(depth + 1, max_padded - 2 * depth)
```

### 1.4 Axis Selection for 4D vs 5D

In HyperSpy, for a signal of shape `(nav_0, ..., nav_k, sig_0, sig_1)`:
- The last two array axes are always the signal axes (ky, kx)
- The leading axes are navigation: for 4D-STEM `(nav_y, nav_x, ky, kx)`; for 5D-STEM `(time, nav_y, nav_x, ky, kx)`

Gaussian blur should target only the **spatial navigation axes** (the last two navigation axes), never the time axis and never the signal axes:

```python
nav_dim = signal.axes_manager.navigation_dimension  # 2 for 4D, 3 for 5D
sig_dim = signal.axes_manager.signal_dimension      # always 2

# sigma tuple: zeros for time (if present) and signal axes
sigma_tuple = tuple([0.0] * (nav_dim - 2) + [sigma_nav, sigma_nav] + [0.0] * sig_dim)
depth_tuple  = tuple([0] * (nav_dim - 2) + [depth_px, depth_px] + [0] * sig_dim)
```

For 4D: `sigma=(σ, σ, 0, 0)`, `depth=(d, d, 0, 0)`  
For 5D: `sigma=(0, σ, σ, 0, 0)`, `depth=(0, d, d, 0, 0)`

The **5D UI** shows a selector for which axes to blur (pre-selected to the last two navigation axes). This is a `QCheckBox` row in the caret, auto-built from `axes_manager.navigation_axes[:-2]`.

### 1.4 GPU Acceleration

When a GPU worker is available (`main_window.dask_manager.gpu_worker_address` is not None):

```python
# GPU path via CuPy (if installed)
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

def _nav_blur_gpu(data: np.ndarray, sigma_tuple: tuple) -> np.ndarray:
    """Apply navigation-space Gaussian blur on GPU."""
    arr = cp.asarray(data)
    out = gpu_gaussian(arr, sigma=sigma_tuple)
    return cp.asnumpy(out)
```

For the batch compute, the Dask scheduler dispatches the `map_overlap` task to the GPU worker if available. The live preview always runs on CPU (the GPU round-trip overhead negates the benefit for a single 256×256 frame at ~14ms).

---

## Part 2: Live Preview Pipeline

### 2.1 Core Compute Function (`spyde/actions/find_vectors.py`)

```python
def _find_vectors_single_frame(
    frame: np.ndarray,           # (ky, kx) float32 — already nav-blurred
    kernel_radius: int,          # disk kernel radius in pixels
    threshold: float,            # correlation threshold in [0, 1]
    min_distance: int,           # minimum peak separation (pixels)
    *,
    subpixel: bool = True,       # apply center-of-mass subpixel refinement
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        corr_map:  (ky, kx) thresholded correlation image for display
        raw_corr:  (ky, kx) pre-threshold correlation (full range [-1,1])
        peaks:     (N, 3) float32 — [ky_subpx, kx_subpx, intensity]
                   integer coords if subpixel=False
    """
    disk = _make_disk(kernel_radius)           # cached on first call per radius
    raw_corr = match_template(frame, disk, pad_input=True)
    corr_map = np.where(raw_corr >= threshold, raw_corr, 0.0)
    peaks_px = peak_local_max(corr_map, min_distance=min_distance, threshold_abs=threshold)

    if len(peaks_px) == 0:
        return corr_map, raw_corr, np.zeros((0, 3), dtype=np.float32)

    if subpixel:
        refined = _subpixel_com(raw_corr, peaks_px)
    else:
        refined = np.column_stack([peaks_px.astype(np.float32),
                                   raw_corr[peaks_px[:, 0], peaks_px[:, 1]]])
    return corr_map, raw_corr, refined


@functools.lru_cache(maxsize=16)
def _make_disk(radius: int) -> np.ndarray:
    """Build a normalized flat-disk kernel; cached by radius."""
    disk = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
    yy, xx = np.ogrid[-radius:radius+1, -radius:radius+1]
    disk[yy**2 + xx**2 <= radius**2] = 1.0
    disk /= disk.sum()
    return disk


def _subpixel_com(corr: np.ndarray, peaks_px: np.ndarray, half_win: int = 2) -> np.ndarray:
    """Center-of-mass subpixel refinement within ±half_win of each integer peak."""
    from scipy.ndimage import center_of_mass
    out = np.empty((len(peaks_px), 3), dtype=np.float32)
    for i, (py, px) in enumerate(peaks_px):
        y0, y1 = max(0, py - half_win), min(corr.shape[0], py + half_win + 1)
        x0, x1 = max(0, px - half_win), min(corr.shape[1], px + half_win + 1)
        patch = corr[y0:y1, x0:x1]
        dy, dx = center_of_mass(patch)
        out[i, 0] = y0 + dy   # ky subpixel
        out[i, 1] = x0 + dx   # kx subpixel
        out[i, 2] = float(corr[py, px])
    return out
```

The live preview uses `NavBlurCache.get_blurred(iy_local, ix_local, raw_pattern)` (see §1.2) which returns the correctly nav-blurred pattern from the cached blurred chunk, or falls back to a single-frame approximation if the async blur hasn't completed yet. The batch compute uses the full `map_overlap` path (§1.3).

### 2.2 Auto-population of Parameters

```python
def _auto_params(frame: np.ndarray) -> dict:
    """Estimate reasonable starting params from the current pattern."""
    # Kernel radius: 5% of shorter dimension, min 3 px
    r_px = max(3, int(min(frame.shape) * 0.05))
    return dict(
        sigma=1.5,
        kernel_radius=r_px,
        threshold=0.3,
        min_distance=2 * r_px,
        subpixel=True,
    )
```

---

## Part 3: UI — Caret Popout

### 3.1 Caret Structure

Following the virtual imaging / orientation mapping pattern:

```
[Find Vectors icon] → CaretGroup titled "Find Diffraction Vectors"
  ├── Row: "Real-space σ"    [slider + spinbox, range 0.1–10.0 px]
  ├── Row: "Kernel radius"   [slider + spinbox, range 1–50 px]  ← linked to CircleROI
  ├── Row: "Threshold"       [slider + spinbox, range 0.0–1.0]
  ├── Row: "Min distance"    [slider + spinbox, range 1–100 px]
  ├── [Subpixel CoM]         [QCheckBox, default ON]
  ├── [Live (ON)] [Compute]  ← button_row
  └── Status label: "N peaks found · X.Xms"
  
  [5D only: tab "Blur Axes"]
  └── [time ☐]  [y ☑]  [x ☑]   ← QCheckBoxes per nav axis
```

### 3.2 CircleROI Linking

A `CircleROI` at the diffraction origin, radius = kernel_radius in data units:

```python
r_data = kernel_radius_px * sig_ax[0].scale  # pixels → Å⁻¹
cx = sig_ax[1].size / 2.0 * sig_ax[1].scale + sig_ax[1].offset
cy = sig_ax[0].size / 2.0 * sig_ax[0].scale + sig_ax[0].offset

circle_roi = CircleROI(
    pos=(cx - r_data, cy - r_data),
    size=(2 * r_data, 2 * r_data),
    pen=mkPen("c", width=1.5),
)
plot.addItem(circle_roi)

def _roi_to_spinbox():
    r = circle_roi.size().x() / 2.0 / sig_ax[0].scale
    radius_spin.blockSignals(True)
    radius_spin.setValue(r)
    radius_spin.blockSignals(False)
    _schedule_recompute()

def _spinbox_to_roi(r_px):
    r_d = r_px * sig_ax[0].scale
    cx2, cy2 = (cx, cy)
    circle_roi.blockSignals(True)
    circle_roi.setPos(cx2 - r_d, cy2 - r_d)
    circle_roi.setSize((2 * r_d, 2 * r_d))
    circle_roi.blockSignals(False)
    _schedule_recompute()

circle_roi.sigRegionChanged.connect(_roi_to_spinbox)
radius_spin.valueChanged.connect(_spinbox_to_roi)
```

### 3.3 Live Preview Window

Two-panel `GraphicsLayoutWidget` MDI subwindow:

```
┌─────────────────────────────────────────────────┐
│   Correlation map (thresholded)  │  Raw pattern  │
│   [corr_map image]               │  [frame image]│
│   [+ markers at peaks]           │  [○ + markers]│
└─────────────────────────────────────────────────┘
```

```python
# Build preview MDI window
preview_window = main_window.add_plot_window(is_navigator=False, signal_tree=plot.signal_tree)
preview_window.setWindowTitle("Vector Finding — Preview")

glw = pg.GraphicsLayoutWidget()
left_plot = glw.addPlot(row=0, col=0, title="Correlation")
right_plot = glw.addPlot(row=0, col=1, title="Diffraction Pattern")

left_img  = pg.ImageItem()
right_img = pg.ImageItem()
left_plot.addItem(left_img)
right_plot.addItem(right_img)

# ScatterPlotItem for '+' markers on correlation image
corr_scatter = pg.ScatterPlotItem(symbol='+', size=12, pen=mkPen('c', width=1.5), brush=None)
# ScatterPlotItem with circle symbols on raw pattern
raw_scatter = pg.ScatterPlotItem(symbol='o', size=kernel_radius_px*2, pen=mkPen('c', width=1), brush=None)
raw_plus    = pg.ScatterPlotItem(symbol='+', size=8, pen=mkPen('c', width=1.5), brush=None)

left_plot.addItem(corr_scatter)
right_plot.addItem(raw_scatter)
right_plot.addItem(raw_plus)
```

**Relay pattern** (identical to orientation mapping):

```python
class _VectorRelay(QtCore.QObject):
    vectors_ready = QtCore.Signal(object, object, object)  # corr_map, raw_corr, peaks(N,3)

relay = _VectorRelay()

def _apply_results(corr_map, raw_corr, peaks):
    left_img.setImage(corr_map.T)
    raw_img_data = plot.current_data  # grab current frame for right panel
    if raw_img_data is not None:
        right_img.setImage(np.asarray(raw_img_data).T)
    spots = [{"pos": (p[1], p[0])} for p in peaks]  # kx, ky for scene coords
    corr_scatter.setData(spots)
    raw_scatter.setData(spots)
    raw_plus.setData(spots)
    status_label.setText(f"{len(peaks)} peaks · {elapsed_ms:.1f}ms")

relay.vectors_ready.connect(_apply_results)
```

### 3.4 Debounce + Generation Counter

50ms debounce timer (identical to orientation mapping):

```python
refit_timer = QTimer()
refit_timer.setInterval(50)
refit_timer.setSingleShot(True)
refit_generation = [0]

def _schedule_recompute():
    refit_timer.start()

def _do_refit():
    # Grab current nav indices and raw pattern on the GUI thread
    nav_indices = _get_current_nav_indices(plot)
    raw_frame = np.asarray(plot.current_data).copy()

    # Update NavBlurCache with the current chunk (triggers async blur if chunk changed)
    cached_dask = getattr(signal, 'cached_dask_array', None)
    if cached_dask is not None and cached_dask.core_cached_blocks:
        # core_cached_blocks[0] is a Future or numpy array for the current chunk
        block = cached_dask.core_cached_blocks[0]
        if not isinstance(block, Future):
            chunk_id = tuple(cached_dask.core_cached_block_inds[0])
            nav_blur_cache.update_chunk(block, chunk_id)

    refit_generation[0] += 1
    my_gen = refit_generation[0]
    sigma = sigma_spin.value()

    def _run():
        if refit_generation[0] != my_gen:
            return
        t0 = time.perf_counter()
        # Get nav-blurred pattern: O(1) from cache, or ~1.2ms single-frame fallback
        iy_local = nav_indices[0] % signal.data.chunks[0][0]  # local position in chunk
        ix_local = nav_indices[1] % signal.data.chunks[1][0]
        blurred = nav_blur_cache.get_blurred(iy_local, ix_local, raw_frame)
        corr_map, raw_corr, peaks = _find_vectors_single_frame(
            blurred, kernel_radius_spin.value(), threshold_spin.value(),
            min_distance_spin.value(), subpixel=subpixel_check.isChecked()
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if refit_generation[0] == my_gen:
            relay.vectors_ready.emit(corr_map, raw_corr, peaks, elapsed_ms)

    threading.Thread(target=_run, daemon=True).start()

refit_timer.timeout.connect(_do_refit)

for spin in [sigma_spin, radius_spin, threshold_spin, mindist_spin]:
    spin.valueChanged.connect(_schedule_recompute)
```

---

## Part 4: Batch Compute

### 4.1 Algorithm

```python
def _do_compute_vectors(signal, params, main_window, signal_tree):
    """
    Full batch compute:
    1. Build sigma/depth tuples from nav_dim
    2. Rechunk for map_overlap
    3. Apply nav Gaussian via map_overlap
    4. Collect blurred data
    5. Run template match + subpixel per frame (in worker thread)
    6. Assemble flat buffer on main thread
    7. Build SpyDEDiffractionVectors + count map signal
    8. Add to signal tree
    """
    data = signal.data  # dask or numpy array
    nav_dim = signal.axes_manager.navigation_dimension   # 2 (4D) or 3 (5D)
    sig_dim = signal.axes_manager.signal_dimension       # 2
    sig_shape = signal.data.shape[-2:]

    sigma = params["sigma"]
    depth_px = int(np.ceil(3 * sigma))
    sigma_tuple = tuple([0.0]*(nav_dim - 2) + [sigma, sigma] + [0.0]*sig_dim)
    depth_tuple  = tuple([0]*(nav_dim - 2) + [depth_px, depth_px] + [0]*sig_dim)

    # Determine chunk size so ghost-padded chunk fits in ~200 MB
    chunk_nav = _nav_chunk_size(sigma, max_ram_mb=200, sig_shape=sig_shape)

    # 4D: chunks=(chunk_nav, chunk_nav, ky, kx)
    # 5D: chunks=(1, chunk_nav, chunk_nav, ky, kx)  — one time step per chunk
    if nav_dim == 2:
        chunks = (chunk_nav, chunk_nav) + sig_shape
        nav_shape = signal.data.shape[:2]
    else:
        chunks = (1, chunk_nav, chunk_nav) + sig_shape
        nav_shape = signal.data.shape[:nav_dim]

    if not hasattr(data, 'dask'):
        da_data = da.from_array(data.astype(np.float32), chunks=chunks)
    else:
        da_data = data.astype(np.float32).rechunk(chunks)

    # Step 1: blurred is a lazy dask array; compute() loads chunks as needed
    blurred_lazy = da.map_overlap(
        gaussian_filter,
        da_data,
        depth=depth_tuple,
        boundary='reflect',
        sigma=sigma_tuple,
        dtype=np.float32,
    )

    # Step 2: Collect results per partition
    # Compute blurred_lazy into memory, then iterate frames
    # For large datasets: compute chunk-by-chunk using dask futures
    blurred = blurred_lazy.compute()   # triggers actual disk reads + blur

    # Step 3: Template match + subpixel, frame by frame
    # Flatten to (N_patterns, ky, kx) for uniform iteration
    flat_blurred = blurred.reshape(-1, sig_shape[0], sig_shape[1])
    n_patterns = flat_blurred.shape[0]
    kernel_r = params["kernel_radius"]
    threshold = params["threshold"]
    min_dist  = params["min_distance"]
    subpixel  = params["subpixel"]

    frame_results = []  # list of (N_i, 3) float32 arrays: [ky_sub, kx_sub, intensity]
    for i in range(n_patterns):
        _, _, peaks = _find_vectors_single_frame(
            flat_blurred[i], kernel_r, threshold, min_dist, subpixel=subpixel
        )
        frame_results.append(peaks)

    # Step 4: Assemble flat buffer on main thread
    sig_ax = signal.axes_manager.signal_axes
    ky_scale = sig_ax[1].scale; ky_offset = sig_ax[1].offset
    kx_scale = sig_ax[0].scale; kx_offset = sig_ax[0].offset

    counts  = np.array([len(r) for r in frame_results], dtype=np.int64)
    offsets = np.zeros(n_patterns + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    N_total = int(offsets[-1])

    flat_buffer = np.zeros((N_total, 5), dtype=np.float32)
    # columns: [nav_x, nav_y, kx_data, ky_data, intensity]
    # nav_shape is (nav_y, nav_x) for 4D, (time, nav_y, nav_x) for 5D
    nav_2d_shape = nav_shape[-2:]  # always (nav_y, nav_x)

    for flat_idx, peaks in enumerate(frame_results):
        if len(peaks) == 0:
            continue
        # Recover nav coordinates from flat_idx
        # For 4D: flat_idx = iy * nav_x + ix
        # For 5D: flat_idx = it * nav_y * nav_x + iy * nav_x + ix
        s, e = offsets[flat_idx], offsets[flat_idx + 1]
        iy = (flat_idx % (nav_2d_shape[0] * nav_2d_shape[1])) // nav_2d_shape[1]
        ix =  flat_idx % nav_2d_shape[1]
        ky_data = peaks[:, 0] * ky_scale + ky_offset
        kx_data = peaks[:, 1] * kx_scale + kx_offset
        flat_buffer[s:e, 0] = ix
        flat_buffer[s:e, 1] = iy
        flat_buffer[s:e, 2] = kx_data
        flat_buffer[s:e, 3] = ky_data
        flat_buffer[s:e, 4] = peaks[:, 2]  # intensity

    return SpyDEDiffractionVectors(
        flat_buffer=flat_buffer,
        offsets=offsets,
        nav_shape=nav_2d_shape,
        full_nav_shape=nav_shape,
        sig_shape=sig_shape,
        sig_axes=signal.axes_manager.signal_axes,
        kernel_radius_px=float(kernel_r),
        kernel_radius_data=float(kernel_r) * sig_ax[0].scale,
        params=params,
    )
```

### 4.2 Dask Worker Dispatch

The compute runs in a background thread (not blocking the GUI). Progress uses the existing `ComputeStatusIndicator` pattern from virtual imaging:

```python
def _on_compute_clicked():
    btn.setEnabled(False)
    status_label.setText("Computing…")

    def _run():
        vecs = _do_compute_vectors(signal, _get_params(), main_window, signal_tree)
        # Marshal count map signal to GUI thread via pending_signal_queue
        count_map = vecs.count_map()
        count_signal = hs.signals.Signal2D(count_map)
        count_signal.metadata.vectors = vecs
        count_signal.metadata.Signal.signal_type = "diffraction_vectors"
        _copy_nav_axes(signal, count_signal)
        main_window._pending_signal_queue.append(count_signal)
        QtCore.QMetaObject.invokeMethod(
            main_window, "_flush_pending_signals",
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        # _flush_pending_signals calls signal_tree.add_node(signal, count_signal, "Diffraction Vectors")

    threading.Thread(target=_run, daemon=True).start()
```

### 4.3 Large Dataset Strategy (Dask Futures)

For datasets that don't fit in RAM after blurring, the batch compute can be split into chunks of time steps or nav tiles and dispatched as Dask futures. The `frame_results` list is then built by collecting futures in order. This is the same polling pattern used by virtual imaging and orientation mapping.

---

## Part 5: `SpyDEDiffractionVectors` Data Layout

### 5.1 Design (PyTorch NestedTensor analogy)

PyXEM's `DiffractionVectors2D` stores ragged data in a numpy object array `(nav_y, nav_x)` where each element is a `(N_i, 2)` array — memory-inefficient and slow for slicing.

The new layout uses a **flat buffer + CSR offset array**:

```
flat_buffer: shape (N_total, 5)   float32
  columns: [nav_x, nav_y, kx_data, ky_data, intensity]

offsets: shape (n_patterns + 1,)  int64   (CSR row pointer)
  offsets[i]   = start of position i in flat_buffer
  offsets[-1]  = N_total

Slicing position (iy, ix):
  flat_idx = iy * nav_shape[1] + ix
  rows = flat_buffer[offsets[flat_idx] : offsets[flat_idx+1]]
```

### 5.2 Class Definition (`spyde/signals/diffraction_vectors.py`)

```python
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpyDEDiffractionVectors:
    flat_buffer: np.ndarray        # (N_total, 5) float32: [nav_x, nav_y, kx, ky, intensity]
    offsets: np.ndarray            # (n_patterns+1,) int64
    nav_shape: tuple               # (nav_y, nav_x)  — the 2D spatial nav grid
    full_nav_shape: tuple          # same as nav_shape for 4D; (time, nav_y, nav_x) for 5D
    sig_shape: tuple               # (ky_size, kx_size)
    sig_axes: object               # hyperspy AxesManager signal_axes
    kernel_radius_px: float
    kernel_radius_data: float      # in Å⁻¹
    params: dict = field(default_factory=dict)
    _dense_cache: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Indexing ─────────────────────────────────────────────────────────────

    def at(self, iy: int, ix: int) -> np.ndarray:
        """(N, 5) array at navigation position (iy, ix)."""
        i = iy * self.nav_shape[1] + ix
        return self.flat_buffer[self.offsets[i]:self.offsets[i+1]]

    def kxy_at(self, iy: int, ix: int) -> np.ndarray:
        """(N, 2) [kx, ky] in data units at (iy, ix)."""
        return self.at(iy, ix)[:, 2:4]

    def intensities_at(self, iy: int, ix: int) -> np.ndarray:
        return self.at(iy, ix)[:, 4]

    def count_map(self) -> np.ndarray:
        """(nav_y, nav_x) int32 — vector count at each position."""
        return np.diff(self.offsets).reshape(self.nav_shape).astype(np.int32)

    def flatten(self) -> np.ndarray:
        """Full (N_total, 5) flat buffer."""
        return self.flat_buffer

    # ── Dense conversion ─────────────────────────────────────────────────────

    def to_dense(self, fill_value: float = np.nan, max_vectors: int = None) -> np.ndarray:
        """(nav_y, nav_x, max_n, 5) dense array; cached after first call."""
        if self._dense_cache is not None:
            return self._dense_cache
        counts = np.diff(self.offsets)
        max_n  = max_vectors or int(counts.max())
        nav_y, nav_x = self.nav_shape
        dense = np.full((nav_y, nav_x, max_n, 5), fill_value, dtype=np.float32)
        for flat_idx in range(nav_y * nav_x):
            iy, ix = divmod(flat_idx, nav_x)
            s, e = self.offsets[flat_idx], self.offsets[flat_idx+1]
            n = e - s
            if n > 0:
                dense[iy, ix, :n] = self.flat_buffer[s:e]
        self._dense_cache = dense
        return dense

    # ── Unique vectors ────────────────────────────────────────────────────────

    def get_unique_vectors(self, distance_threshold: float = 0.01) -> np.ndarray:
        """(M, 2) [kx, ky] — unique vectors across entire scan."""
        kxy = self.flat_buffer[:, 2:4]
        if distance_threshold == 0:
            return np.unique(kxy, axis=0)
        # iterative distance-comparison (same algorithm as pyxem)
        from scipy.spatial.distance import cdist
        unique = list(kxy[:1])
        for v in kxy[1:]:
            dists = cdist([v], unique)[0]
            if dists.min() >= distance_threshold:
                unique.append(v)
        return np.array(unique, dtype=np.float32)

    # ── PyXEM compatibility ────────────────────────────────────────────────────

    def to_pyxem(self):
        """Convert to pyxem DiffractionVectors2D (object-array form)."""
        from pyxem.signals import DiffractionVectors2D
        nav_y, nav_x = self.nav_shape
        ragged = np.empty((nav_y, nav_x), dtype=object)
        for iy in range(nav_y):
            for ix in range(nav_x):
                ragged[iy, ix] = self.kxy_at(iy, ix)
        return DiffractionVectors2D(ragged)

    # ── Downstream gateways ───────────────────────────────────────────────────

    def get_strain_maps(self, unstrained_vectors: np.ndarray, distance: float = 0.5):
        """Delegate to pyxem after converting to DiffractionVectors2D."""
        dv = self.to_pyxem()
        return dv.get_strain_maps(unstrained_vectors, distance=distance)

    def cluster(self, eps: float = 0.02, min_samples: int = 5):
        """DBSCAN clustering on all kx,ky vectors. Returns labels array (N_total,)."""
        from sklearn.cluster import DBSCAN
        kxy = self.flat_buffer[:, 2:4]
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(kxy)

    # ── Markers for overlay rendering ─────────────────────────────────────────

    def spots_at(self, iy: int, ix: int) -> list:
        """Return list of pyqtgraph ScatterPlotItem spot dicts for (iy, ix)."""
        kxy = self.kxy_at(iy, ix)
        # scene coords: scene_x = ky, scene_y = kx  (pyqtgraph col-major)
        r_scene = self.kernel_radius_data * 2  # diameter for 'o' symbol size
        return [{"pos": (float(ky), float(kx)), "size": r_scene}
                for kx, ky in kxy]

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_ragged(cls, ragged: np.ndarray, nav_shape: tuple, **kwargs) -> SpyDEDiffractionVectors:
        """Build from pyxem-style (nav_y, nav_x) object array of (N_i, 2) [kx, ky] arrays."""
        nav_y, nav_x = nav_shape
        counts  = np.array([len(ragged[iy, ix]) for iy in range(nav_y) for ix in range(nav_x)], dtype=np.int64)
        offsets = np.zeros(nav_y * nav_x + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        N_total = int(offsets[-1])

        flat_buffer = np.zeros((N_total, 5), dtype=np.float32)
        for flat_idx in range(nav_y * nav_x):
            iy, ix = divmod(flat_idx, nav_x)
            s, e = offsets[flat_idx], offsets[flat_idx+1]
            if e > s:
                arr = ragged[iy, ix]  # (N, 2)
                flat_buffer[s:e, 0] = ix
                flat_buffer[s:e, 1] = iy
                flat_buffer[s:e, 2:4] = arr  # kx, ky
        return cls(flat_buffer=flat_buffer, offsets=offsets,
                   nav_shape=nav_shape, full_nav_shape=nav_shape, **kwargs)
```

---

## Part 6: Signal Tree Node & Overlay Rendering

### 6.1 Node Representation (Option A)

The vectors node stores a `(nav_y, nav_x)` count-map `Signal2D` with `metadata.vectors = SpyDEDiffractionVectors(...)`. This slots into `signal_tree.add_node()` without any changes to `SignalNode` or `BaseSignalTree`.

```python
import hyperspy.api as hs

count_signal = hs.signals.Signal2D(vecs.count_map().astype(np.float32))
count_signal.metadata.vectors = vecs
count_signal.metadata.Signal.signal_type = "diffraction_vectors"
# Copy navigation axes from parent
for i, ax in enumerate(signal.axes_manager.navigation_axes):
    count_signal.axes_manager.navigation_axes[i].scale  = ax.scale
    count_signal.axes_manager.navigation_axes[i].offset = ax.offset
    count_signal.axes_manager.navigation_axes[i].units  = ax.units
    count_signal.axes_manager.navigation_axes[i].name   = ax.name

signal_tree.add_node(signal, count_signal, "Diffraction Vectors")
```

### 6.2 Overlay on Signal Plot

When the user activates the vectors node in the signal tree, the signal plot switches to show the **parent diffraction pattern** with vector overlays. The existing `plot.set_current_signal()` machinery handles the parent image display. The overlay layer is added as pyqtgraph items:

```python
def _activate_vector_overlay(plot, vecs: SpyDEDiffractionVectors):
    scatter_circles = pg.ScatterPlotItem(
        symbol='o',
        pen=mkPen('c', width=1.0),
        brush=None,
    )
    scatter_plus = pg.ScatterPlotItem(
        symbol='+',
        size=8,
        pen=mkPen('c', width=1.5),
        brush=None,
    )
    plot.addItem(scatter_circles)
    plot.addItem(scatter_plus)

    def _update(nav_idx):
        iy, ix = nav_idx
        spots = vecs.spots_at(iy, ix)
        scatter_circles.setData(spots)
        scatter_plus.setData(spots)

    plot.sigNavigatorMoved.connect(_update)
    _update(plot.current_nav_index)
    return scatter_circles, scatter_plus
```

---

## Part 7: Action Registration

New YAML entry in `spyde/actions/hyper_signal_actions/` (or appended to the existing pyxem config):

```yaml
- name: "Find Diffraction Vectors"
  icon: "peak_finding.svg"
  function: "spyde.actions.find_vectors.find_diffraction_vectors"
  signal_types: ["electron_diffraction"]
  toggle: true
```

The `find_diffraction_vectors(toolbar, action_name, ...)` function follows the orientation mapping guard pattern:

```python
_FV_BUILT_TOOLBARS: set = set()

def find_diffraction_vectors(toolbar, action_name="Find Diffraction Vectors", *args, **kwargs):
    tid = id(toolbar)
    if tid in _FV_BUILT_TOOLBARS:
        return
    _FV_BUILT_TOOLBARS.add(tid)
    # ... build CaretGroup, ROI, preview window, state dict ...
```

---

## Part 8: PyXEM Upstream Suggestions

1. **`ElectronDiffraction2D.find_vectors_wncc(sigma_nav, kernel_radius, threshold, min_distance, subpixel=True)`** — new method exposing the window-normalized cross-correlation pipeline as a first-class operation, returning `DiffractionVectors2D`. The current `find_peaks(method='template_matching')` wraps this in an interactive widget that isn't scriptable cleanly.

2. **`ElectronDiffraction2D.filter()` doesn't use `map_overlap`** — the current implementation calls `func(self.data, **kwargs)` directly. For lazy datasets with `dask_image.ndfilters.gaussian_filter`, this works because `dask_image` uses `map_overlap` internally. But with `scipy.ndimage.gaussian_filter` on a dask array it silently produces wrong results at chunk boundaries. The method should warn or document this.

3. **`DiffractionVectors2D.from_flat_buffer(flat, offsets, nav_shape)`** — classmethod mirroring the CSR design above; submit as a PR to pyxem.

4. **`DiffractionVectors2D.get_strain_maps` blocks on lazy input** — calls `.compute()` internally without returning a lazy result; document as a breaking limitation or fix.

5. **`subpixel_refine` as a pipeline step** — the current API buries subpixel refinement in interactive find_peaks; expose as `DiffractionVectors2D.subpixel_refine(method='com'|'gaussian', half_win=2)` returning a new `DiffractionVectors2D`.

---

## Part 9: Tests

### 9.1 Unit Tests (`spyde/tests/test_find_vectors.py`)

```python
import numpy as np
import pytest
import functools
from scipy.ndimage import gaussian_filter
from spyde.actions.find_vectors import (
    _find_vectors_single_frame, _auto_params, _nav_chunk_size, _subpixel_com
)
from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors


# ── Core algorithm ────────────────────────────────────────────────────────────

def test_detects_known_peaks():
    frame = np.zeros((128, 128), dtype=np.float32)
    expected = [(30, 40), (80, 70), (60, 100)]
    for ky, kx in expected:
        frame[ky-3:ky+3, kx-3:kx+3] = 10.0
    frame = gaussian_filter(frame, sigma=1.5)
    _, _, peaks = _find_vectors_single_frame(frame, kernel_radius=5, threshold=0.3, min_distance=8)
    for ey, ex in expected:
        dists = np.hypot(peaks[:, 0] - ey, peaks[:, 1] - ex)
        assert dists.min() < 3, f"Peak at ({ey},{ex}) not found; got {peaks[:, :2]}"


def test_threshold_controls_count():
    frame = np.zeros((128, 128), dtype=np.float32)
    for ky, kx in [(20, 20), (60, 60), (100, 100)]:
        frame[ky-3:ky+3, kx-3:kx+3] = 10.0
    frame = gaussian_filter(frame, sigma=1.5)
    _, _, p_low  = _find_vectors_single_frame(frame, 5, 0.05, 8)
    _, _, p_high = _find_vectors_single_frame(frame, 5, 0.90, 8)
    assert len(p_low) >= len(p_high)


def test_min_distance_prevents_duplicates():
    frame = np.zeros((64, 64), dtype=np.float32)
    frame[30:34, 30:34] = 10.0
    frame[32:36, 32:36] = 10.0
    frame = gaussian_filter(frame, sigma=0.5)
    _, _, peaks = _find_vectors_single_frame(frame, 4, 0.2, 10)
    assert len(peaks) <= 1


def test_output_shapes():
    frame = np.random.rand(64, 64).astype(np.float32)
    corr, raw, peaks = _find_vectors_single_frame(frame, 4, 0.5, 5)
    assert corr.shape == frame.shape
    assert raw.shape == frame.shape
    assert peaks.ndim == 2 and peaks.shape[1] == 3


def test_zero_frame_no_peaks():
    frame = np.zeros((64, 64), dtype=np.float32)
    _, _, peaks = _find_vectors_single_frame(frame, 4, 0.3, 5)
    assert len(peaks) == 0


def test_subpixel_refinement_moves_peaks():
    """Subpixel CoM should shift integer peaks to fractional positions."""
    frame = np.zeros((64, 64), dtype=np.float32)
    # Off-center peak: blob centered at (30.3, 40.7)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            dist = np.hypot(dy - 0.3, dx - 0.7)
            frame[30 + dy, 40 + dx] = max(0, 5.0 - dist)
    frame = gaussian_filter(frame, sigma=0.5)
    _, _, peaks_sub = _find_vectors_single_frame(frame, 4, 0.1, 6, subpixel=True)
    _, _, peaks_int = _find_vectors_single_frame(frame, 4, 0.1, 6, subpixel=False)
    # Subpixel peaks should have non-integer coordinates
    assert len(peaks_sub) > 0
    assert any(p % 1 != 0 for p in peaks_sub[0, :2])
    # Integer peaks should be whole numbers
    assert all(p % 1 == 0 for p in peaks_int[0, :2])


def test_disk_kernel_cached():
    from spyde.actions.find_vectors import _make_disk
    d1 = _make_disk(8)
    d2 = _make_disk(8)
    assert d1 is d2  # lru_cache hit


def test_auto_params_valid_ranges():
    frame = np.random.rand(128, 128).astype(np.float32)
    p = _auto_params(frame)
    assert 0 < p["sigma"] <= 10
    assert 1 <= p["kernel_radius"] < 64
    assert 0 < p["threshold"] < 1
    assert p["min_distance"] >= 1
    assert isinstance(p["subpixel"], bool)


# ── NavBlurCache ──────────────────────────────────────────────────────────────

def test_nav_blur_cache_warm_hit():
    """After update_chunk, get_blurred returns the correctly blurred pattern."""
    from spyde.actions.find_vectors import NavBlurCache
    from scipy.ndimage import gaussian_filter

    sigma = 1.5
    cache = NavBlurCache(sigma=sigma)
    chunk = np.random.rand(8, 8, 64, 64).astype(np.float32)
    chunk[4, 4, 30, 30] = 20.0  # spike at center of chunk

    cache.update_chunk(chunk, chunk_id=(0, 0))
    cache._blur_thread.join()  # wait for async blur to finish

    result = cache.get_blurred(4, 4, raw_pattern=chunk[4, 4])
    # Reference: full-array blur
    ref = gaussian_filter(chunk, sigma=(sigma, sigma, 0, 0))[4, 4]
    np.testing.assert_allclose(result, ref, atol=1e-3)


def test_nav_blur_cache_cold_fallback():
    """Before async blur completes, get_blurred falls back to single-frame blur."""
    from spyde.actions.find_vectors import NavBlurCache
    from scipy.ndimage import gaussian_filter

    sigma = 1.5
    cache = NavBlurCache(sigma=sigma)
    chunk = np.random.rand(16, 16, 64, 64).astype(np.float32)
    cache._chunk_id = (0, 0)   # pretend chunk is loaded
    cache._raw_chunk = chunk
    cache._blurred = None      # but blur not done yet

    raw_pattern = chunk[8, 8]
    result = cache.get_blurred(8, 8, raw_pattern=raw_pattern)
    ref = gaussian_filter(raw_pattern, sigma=(sigma, sigma))
    np.testing.assert_allclose(result, ref, atol=1e-6)


def test_nav_blur_cache_invalidate_clears():
    """invalidate() clears cached state and updates sigma."""
    from spyde.actions.find_vectors import NavBlurCache
    cache = NavBlurCache(sigma=1.5)
    cache._blurred = np.zeros((8, 8, 64, 64), dtype=np.float32)
    cache._chunk_id = (0, 0)
    cache.invalidate(sigma=2.0)
    assert cache._blurred is None
    assert cache._chunk_id is None
    assert cache.sigma == 2.0


def test_nav_blur_cache_chunk_id_guards_stale_blur():
    """A blur started for chunk (0,0) should not overwrite results for chunk (1,0)."""
    from spyde.actions.find_vectors import NavBlurCache
    import time

    sigma = 1.5
    cache = NavBlurCache(sigma=sigma)
    chunk_a = np.zeros((4, 4, 16, 16), dtype=np.float32)
    chunk_b = np.ones((4, 4, 16, 16), dtype=np.float32)

    cache.update_chunk(chunk_a, (0, 0))
    # Immediately switch to chunk_b before blur of chunk_a finishes
    cache.update_chunk(chunk_b, (1, 0))
    cache._blur_thread.join()  # wait for blur_b to finish

    # Result should be for chunk_b (ones), not chunk_a (zeros)
    with cache._lock:
        assert cache._chunk_id == (1, 0)
        if cache._blurred is not None:
            assert cache._blurred.mean() > 0.5  # chunk_b was ones


def test_nav_blur_cache_edge_accuracy():
    """Edge patterns of the chunk should be within 5% of the full-array reference."""
    from spyde.actions.find_vectors import NavBlurCache
    from scipy.ndimage import gaussian_filter

    sigma = 1.5
    # Simulate: chunk is surrounded by actual data (not zeros)
    full = np.random.rand(24, 24, 32, 32).astype(np.float32)
    full[12, 12, 16, 16] = 20.0

    ref_blurred = gaussian_filter(full, sigma=(sigma, sigma, 0, 0))

    # NavBlurCache sees only the middle 8x8 chunk
    chunk = full[8:16, 8:16].copy()
    cache = NavBlurCache(sigma=sigma)
    cache.update_chunk(chunk, (0, 0))
    cache._blur_thread.join()

    # At the chunk EDGE (position 0,0 in local = global (8,8)):
    result_edge = cache.get_blurred(0, 0, raw_pattern=chunk[0, 0])
    ref_edge = ref_blurred[8, 8]
    # Reflect-pad gives different boundary than true neighbors; allow 5% error
    rel_err = np.max(np.abs(result_edge - ref_edge)) / (np.max(np.abs(ref_edge)) + 1e-6)
    assert rel_err < 0.05, f"Edge pattern error too large: {rel_err:.3f}"


def test_nav_blur_cache_speed():
    """Warm cache lookup must be sub-millisecond."""
    import time
    from spyde.actions.find_vectors import NavBlurCache

    cache = NavBlurCache(sigma=1.5)
    chunk = np.random.rand(16, 16, 256, 256).astype(np.float32)
    cache.update_chunk(chunk, (0, 0))
    cache._blur_thread.join()

    raw_pattern = chunk[8, 8]
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        cache.get_blurred(8, 8, raw_pattern)
    avg_ms = (time.perf_counter() - t0) / N * 1000
    assert avg_ms < 1.0, f"Warm cache lookup too slow: {avg_ms:.2f}ms"


# ── Chunk size calculation ────────────────────────────────────────────────────

def test_nav_chunk_size_respects_memory_limit():
    chunk = _nav_chunk_size(sigma=2.0, max_ram_mb=200, sig_shape=(128, 128))
    depth = int(np.ceil(3 * 2.0))
    ram_mb = (chunk + 2*depth)**2 * 128 * 128 * 4 / 1e6
    assert ram_mb <= 200, f"Chunk uses {ram_mb:.0f} MB, limit 200 MB"


def test_nav_chunk_size_larger_than_depth():
    for sigma in [0.5, 1.0, 2.0, 3.0]:
        chunk = _nav_chunk_size(sigma, max_ram_mb=200, sig_shape=(256, 256))
        depth = int(np.ceil(3 * sigma))
        assert chunk > depth, f"sigma={sigma}: chunk={chunk} <= depth={depth}"


# ── Navigation Gaussian blur with map_overlap ─────────────────────────────────

def test_map_overlap_correct_at_chunk_boundary():
    """Verify map_overlap produces the same result as full-array gaussian_filter."""
    import dask.array as da
    data = np.zeros((8, 8, 32, 32), dtype=np.float32)
    data[4, 0, 16, 16] = 100.0  # spike straddling chunk boundary at row 4

    sigma = 1.5
    depth = int(np.ceil(3 * sigma))
    da_data = da.from_array(data, chunks=(4, 4, 32, 32))

    result = da.map_overlap(
        gaussian_filter, da_data,
        depth=(depth, depth, 0, 0), boundary='reflect',
        sigma=(sigma, sigma, 0, 0), dtype=np.float32,
    ).compute()

    reference = gaussian_filter(data, sigma=(sigma, sigma, 0, 0))
    # Value at [3,0,16,16] must match reference (spike bleeds across boundary)
    np.testing.assert_allclose(result[3, 0, 16, 16], reference[3, 0, 16, 16], rtol=1e-4)


def test_map_overlap_wrong_without_overlap():
    """Confirm that map_blocks (no overlap) gives wrong result at chunk boundaries."""
    import dask.array as da
    data = np.zeros((8, 8, 32, 32), dtype=np.float32)
    data[4, 0, 16, 16] = 100.0

    sigma = 1.5
    da_data = da.from_array(data, chunks=(4, 4, 32, 32))

    wrong = da_data.map_blocks(gaussian_filter, sigma=(sigma, sigma, 0, 0), dtype=np.float32).compute()
    reference = gaussian_filter(data, sigma=(sigma, sigma, 0, 0))

    # The two should differ at the boundary
    assert abs(wrong[3, 0, 16, 16] - reference[3, 0, 16, 16]) > 0.1, \
        "Expected chunk boundary artifact but result matched reference"


def test_sigma_tuple_4d():
    """4D signal: sigma tuple is (s, s, 0, 0)."""
    import hyperspy.api as hs
    s = hs.signals.Signal2D(np.zeros((4, 4, 16, 16)))
    nav_dim = s.axes_manager.navigation_dimension  # 2
    sig_dim  = s.axes_manager.signal_dimension      # 2
    sigma_nav = 1.5
    sigma_tuple = tuple([0.0]*(nav_dim-2) + [sigma_nav, sigma_nav] + [0.0]*sig_dim)
    assert sigma_tuple == (1.5, 1.5, 0.0, 0.0)


def test_sigma_tuple_5d():
    """5D signal: sigma tuple is (0, s, s, 0, 0) — time axis gets zero."""
    import hyperspy.api as hs
    s = hs.signals.Signal2D(np.zeros((3, 4, 4, 16, 16)))
    nav_dim = s.axes_manager.navigation_dimension  # 3
    sig_dim  = s.axes_manager.signal_dimension      # 2
    sigma_nav = 1.5
    sigma_tuple = tuple([0.0]*(nav_dim-2) + [sigma_nav, sigma_nav] + [0.0]*sig_dim)
    assert sigma_tuple == (0.0, 1.5, 1.5, 0.0, 0.0)


# ── SpyDEDiffractionVectors ───────────────────────────────────────────────────

def _make_vecs(nav_shape=(4, 4), n_per_pos=3):
    nav_y, nav_x = nav_shape
    n_nav = nav_y * nav_x
    counts  = np.full(n_nav, n_per_pos, dtype=np.int64)
    offsets = np.zeros(n_nav + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    N = int(offsets[-1])
    flat = np.random.rand(N, 5).astype(np.float32)
    return SpyDEDiffractionVectors(
        flat_buffer=flat, offsets=offsets,
        nav_shape=nav_shape, full_nav_shape=nav_shape,
        sig_shape=(128, 128), sig_axes=None,
        kernel_radius_px=5.0, kernel_radius_data=0.05,
    )


def test_at_returns_correct_rows():
    vecs = _make_vecs((3, 3), n_per_pos=4)
    for iy in range(3):
        for ix in range(3):
            assert vecs.at(iy, ix).shape == (4, 5)


def test_kxy_at_correct_columns():
    vecs = _make_vecs()
    assert vecs.kxy_at(0, 0).shape == (3, 2)


def test_count_map():
    vecs = _make_vecs((4, 4), n_per_pos=3)
    cm = vecs.count_map()
    assert cm.shape == (4, 4)
    assert (cm == 3).all()


def test_to_dense_shape_and_cache():
    vecs = _make_vecs((2, 3), n_per_pos=5)
    d1 = vecs.to_dense()
    assert d1.shape == (2, 3, 5, 5)
    d2 = vecs.to_dense()
    assert d1 is d2  # cache hit


def test_flatten_full_buffer():
    vecs = _make_vecs((2, 2), n_per_pos=3)
    assert vecs.flatten().shape == (12, 5)


def test_from_ragged_roundtrip():
    nav_shape = (3, 4)
    nav_y, nav_x = nav_shape
    ragged = np.empty(nav_shape, dtype=object)
    for i in range(nav_y):
        for j in range(nav_x):
            n = np.random.randint(1, 8)
            ragged[i, j] = np.random.rand(n, 2).astype(np.float32)

    vecs = SpyDEDiffractionVectors.from_ragged(
        ragged, nav_shape,
        full_nav_shape=nav_shape, sig_shape=(128, 128),
        sig_axes=None, kernel_radius_px=5.0, kernel_radius_data=0.05,
    )
    for i in range(nav_y):
        for j in range(nav_x):
            assert len(vecs.at(i, j)) == len(ragged[i, j])


def test_to_pyxem_type():
    from pyxem.signals import DiffractionVectors2D
    vecs = _make_vecs()
    dv = vecs.to_pyxem()
    assert isinstance(dv, DiffractionVectors2D)


# ── Performance ───────────────────────────────────────────────────────────────

def test_single_frame_pipeline_under_20ms():
    import time
    frame = np.random.rand(256, 256).astype(np.float32)
    frame = gaussian_filter(frame, sigma=1.5)
    _find_vectors_single_frame(frame, 12, 0.3, 10)  # warm up
    t0 = time.perf_counter()
    for _ in range(10):
        _find_vectors_single_frame(frame, 12, 0.3, 10)
    avg_ms = (time.perf_counter() - t0) / 10 * 1000
    assert avg_ms < 20, f"Pipeline too slow: {avg_ms:.1f}ms (limit 20ms)"
```

### 9.2 Integration Tests (`spyde/tests/test_find_vectors_integration.py`)

```python
import pytest
import numpy as np


@pytest.mark.usefixtures("qapp")
def test_caret_builds_on_4d_dataset(stem_4d_dataset, qtbot):
    """find_diffraction_vectors caret builds without error."""
    window = stem_4d_dataset["window"]
    subwindows = stem_4d_dataset["subwindows"]
    signal_pw = next(sw for sw in subwindows if not sw.plot.is_navigator)
    toolbar = signal_pw.plot.toolbar

    from spyde.actions.find_vectors import find_diffraction_vectors
    find_diffraction_vectors(toolbar)
    assert hasattr(toolbar, "_fv_state")


@pytest.mark.usefixtures("qapp")
def test_compute_adds_vectors_node(stem_4d_dataset, qtbot):
    """Batch compute adds a SpyDEDiffractionVectors-backed node to the signal tree."""
    window = stem_4d_dataset["window"]
    trees = stem_4d_dataset["signal_trees"]
    tree = trees[0]
    n_before = sum(1 for _ in tree.walk())

    from spyde.actions.find_vectors import _do_compute_vectors
    from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors
    import hyperspy.api as hs

    signal = tree.root
    params = dict(sigma=1.0, kernel_radius=4, threshold=0.3, min_distance=8, subpixel=True)
    vecs = _do_compute_vectors(signal, params, window, tree)

    assert isinstance(vecs, SpyDEDiffractionVectors)
    assert vecs.flat_buffer.shape[1] == 5
    assert vecs.count_map().shape == signal.data.shape[:2]


@pytest.mark.usefixtures("qapp")
def test_sigma_tuple_5d(stem_5d_dataset):
    """5D dataset: sigma tuple has zero in time axis position."""
    trees = stem_5d_dataset["signal_trees"]
    signal = trees[0].root
    nav_dim = signal.axes_manager.navigation_dimension
    sig_dim  = signal.axes_manager.signal_dimension
    sigma = 1.5
    sigma_tuple = tuple([0.0]*(nav_dim-2) + [sigma, sigma] + [0.0]*sig_dim)
    assert sigma_tuple[0] == 0.0  # time axis = 0
    assert sigma_tuple[1] == sigma
    assert sigma_tuple[2] == sigma


@pytest.mark.usefixtures("qapp")
def test_chunk_boundary_blur_correctness_on_real_signal(stem_4d_dataset):
    """map_overlap result matches full-array gaussian_filter for a 4D signal."""
    import dask.array as da
    trees = stem_4d_dataset["signal_trees"]
    signal = trees[0].root
    data = np.asarray(signal.data).astype(np.float32)

    sigma = 1.5
    depth = int(np.ceil(3 * sigma))
    da_data = da.from_array(data, chunks=(4, 4) + data.shape[2:])

    overlap_result = da.map_overlap(
        gaussian_filter, da_data,
        depth=(depth, depth, 0, 0), boundary='reflect',
        sigma=(sigma, sigma, 0, 0), dtype=np.float32,
    ).compute()

    reference = gaussian_filter(data, sigma=(sigma, sigma, 0, 0))
    np.testing.assert_allclose(overlap_result, reference, rtol=1e-4,
                               err_msg="map_overlap blur differs from reference")
```

---

## Part 10: Implementation Sequence

### Phase 1 — Core algorithm + data class (no UI)
1. `spyde/signals/diffraction_vectors.py`: `SpyDEDiffractionVectors`
2. `spyde/actions/find_vectors.py`: `_find_vectors_single_frame`, `_make_disk` (cached), `_subpixel_com`, `_auto_params`, `_nav_chunk_size`, sigma/depth tuple helpers
3. Pass all unit tests from §9.1

### Phase 2 — Batch compute
4. `_do_compute_vectors`: `map_overlap` blur → template match → flat buffer assembly
5. Wire to `_pending_signal_queue` / `_flush_pending_signals`
6. Pass integration tests from §9.2

### Phase 3 — Live preview caret UI
7. `find_diffraction_vectors(toolbar, ...)`: `CaretGroup` + parameter rows + `CircleROI` + `QCheckBox` for subpixel
8. Debounce timer, generation counter, `_VectorRelay`, two-panel preview window
9. Auto-populate on caret open via `_auto_params`
10. 5D axis selection UI (`QCheckBox` per extra nav axis)

### Phase 4 — Signal tree overlay
11. `_activate_vector_overlay`: `ScatterPlotItem` circles + plus markers
12. Wire to `sigNavigatorMoved` / navigation index changes
13. Connect "Compute" button → `_on_compute_clicked` → background thread

### Phase 5 — Downstream gateways
14. Strain mapping toolbar action on vectors node
15. Virtual image from vectors toolbar action
16. Clustering toolbar action

### Phase 6 — Polish & GPU
17. GPU path (CuPy) for `_do_compute_vectors` nav blur step
18. Subpixel refinement with Gaussian fitting option (alternative to CoM)
19. Benchmark and profile on real 4D-STEM data

---

## Open Questions (Resolved)

1. **Nav blur scope for 5D**: Blur only spatial nav axes (last 2 of nav). Time axis gets σ=0. User can override via checkbox row in caret. ✓

2. **Ragged gather strategy**: Collect frame results into `frame_results = list of (N_i, 3) arrays` in main thread after `blurred.compute()`. Assemble `flat_buffer` on main thread. For very large datasets (blurred array > available RAM), process in time-step chunks using Dask futures with the existing progress polling pattern. ✓

3. **Signal tree node type**: Option A — count map `Signal2D` + `metadata.vectors`. Clean fit with existing `add_node()` / `update_plot_states()` infrastructure. ✓

4. **Subpixel refinement**: Center-of-mass in a ±2 px window around each integer peak. Toggle in caret (ON by default). Adds negligible time (<0.1ms for typical N peaks). ✓

5. **PyXEM upstream**: Submit `from_flat_buffer` + `find_vectors_wncc` as PR to pyxem. SpyDE carries locally until merged. ✓

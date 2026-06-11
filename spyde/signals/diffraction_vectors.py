from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# Column indices — never use bare integer literals outside this module
COL_NAV_X     = 0
COL_NAV_Y     = 1
COL_KX        = 2
COL_KY        = 3
COL_TIME      = 4   # -1.0 for 4D datasets (no time axis)
COL_INTENSITY = 5
N_COLS        = 6


def _build_nav_offsets(
    flat_buffer: np.ndarray,
    full_nav_shape: tuple,
) -> List[np.ndarray]:
    """
    Build the multi-level CSR index from a sorted flat_buffer.

    The flat_buffer must be sorted outermost-nav-dim first (e.g. t → iy → ix).

    Layout
    ------
    Returns one offsets array per navigation dimension, outermost first.
    The outermost N-1 levels use **uniform strides** (the grid is always
    rectangular) so they can be stored as simple `(dim_size + 1,)` arrays
    where `offsets[k]` = k * (product of inner dim sizes).

    Only the innermost level (x_offsets, over all leaf positions) stores
    actual variable-length counts — vectors per (t, iy, ix) position.

    Lookup for nav indices (i0, i1, …, iN) where N = len(full_nav_shape):
        flat_pos = i0 * stride[0] + i1 * stride[1] + … + iN
        s = nav_offsets[-1][flat_pos]
        e = nav_offsets[-1][flat_pos + 1]
        return flat_buffer[s:e]

    The outer arrays (nav_offsets[0..N-2]) are included for the partial-index
    API (slice_at(t) → all vectors at time t).  They store vector offsets:
        nav_offsets[k][i] = sum of vectors in all leaf positions with outer
                            index < i along dimension k.

    For full_nav_shape = (nav_y, nav_x):
        nav_offsets = [y_vec_offsets (nav_y+1,), x_vec_offsets (nav_y*nav_x+1,)]
    For full_nav_shape = (n_t, nav_y, nav_x):
        nav_offsets = [t_vec_offsets (n_t+1,),
                       y_vec_offsets (n_t*nav_y+1,),
                       x_vec_offsets (n_t*nav_y*nav_x+1,)]
    """
    n_dims = len(full_nav_shape)
    n_patterns = int(np.prod(full_nav_shape))

    if len(flat_buffer) == 0:
        nav_offsets = []
        product = 1
        for dim_size in full_nav_shape:
            product *= dim_size
            nav_offsets.append(np.zeros(product + 1, dtype=np.int64))
        return nav_offsets

    # ── Build the innermost (leaf) offsets: one entry per (i0,…,iN) position ──
    # Map each vector to its flat leaf index using stored coordinate columns.
    if n_dims == 2:
        col_seq = [COL_NAV_Y, COL_NAV_X]
    elif n_dims == 3:
        col_seq = [COL_TIME, COL_NAV_Y, COL_NAV_X]
    else:
        raise NotImplementedError("nav_offsets for >3 nav dims requires explicit outer columns")

    strides = np.ones(n_dims, dtype=np.int64)
    for i in range(n_dims - 2, -1, -1):
        strides[i] = strides[i + 1] * full_nav_shape[i + 1]

    flat_leaf = np.zeros(len(flat_buffer), dtype=np.int64)
    for dim, col in enumerate(col_seq):
        vals = flat_buffer[:, col].astype(np.int64)
        if col == COL_TIME:
            vals = np.where(vals < 0, np.int64(0), vals)
        flat_leaf += vals * strides[dim]

    leaf_counts = np.bincount(flat_leaf, minlength=n_patterns).astype(np.int64)
    innermost = np.zeros(n_patterns + 1, dtype=np.int64)
    np.cumsum(leaf_counts, out=innermost[1:])

    # ── Build outer levels by summing leaf_counts over inner-dimension groups ──
    # outer_level[k] stores the cumulative vector counts at dimension k,
    # collapsing all inner dimensions.  This lets slice_at(t) return the
    # exact flat_buffer slice for time step t in O(1).
    #
    # Example: full_nav_shape=(3, 4, 4), leaf_counts shape (48,):
    #   y_level: sum groups of nav_x=4  → shape (12,)  [n_t * nav_y]
    #   t_level: sum groups of nav_y=4  → shape (3,)   [n_t]
    level_offsets = [innermost]
    group_counts = leaf_counts.copy()

    for dim in range(n_dims - 1, 0, -1):
        group_size = full_nav_shape[dim]
        n_outer = len(group_counts) // group_size
        outer_counts = group_counts.reshape(n_outer, group_size).sum(axis=1)
        outer_off = np.zeros(n_outer + 1, dtype=np.int64)
        np.cumsum(outer_counts, out=outer_off[1:])
        level_offsets.append(outer_off)
        group_counts = outer_counts

    level_offsets.reverse()  # now outermost-first
    return level_offsets


@dataclass
class SpyDEDiffractionVectors:
    """
    Flat-buffer CSR storage for diffraction vectors across a scan.

    flat_buffer : (N_total, 6) float32
        columns: [nav_x, nav_y, kx, ky, time, intensity]
        Sorted outermost-nav-dim first: (t, iy, ix) for 5D; (iy, ix) for 4D.
        time = -1.0 for 4D datasets.

    nav_offsets : list of (M_k + 1,) int64 arrays, one per nav dimension
        Multi-level CSR index, outermost dimension first.
        Lookup pattern — slice for nav indices (i0, i1, …, iN):
            row = 0
            for k, idx in enumerate(nav_indices):
                row = nav_offsets[k][row] + idx
            s, e = nav_offsets[-1][row], nav_offsets[-1][row + 1]
            return flat_buffer[s:e]

        4D (nav_y, nav_x)         → [y_offsets (nav_y+1,),
                                      x_offsets (nav_y*nav_x+1,)]
        5D (n_t, nav_y, nav_x)    → [t_offsets (n_t+1,),
                                      y_offsets (n_t*nav_y+1,),
                                      x_offsets (n_t*nav_y*nav_x+1,)]

    nav_shape      : (nav_y, nav_x) — always the innermost 2-D spatial grid
    full_nav_shape : all nav dimensions, outermost first

    offsets : (nav_y*nav_x + 1,) int64
        Legacy single-level spatial CSR (kept for backward compat with 4D code).
        For 5D this is None; use nav_offsets[-1] via slice_at() instead.

    Virtual imaging
    ---------------
    virtual_image_from_roi(cx, cy, r_outer, r_inner, t, intensity_weighted)
        Uses nav_offsets to isolate the requested time-step slice in O(1),
        then does O(N_frame) vectorised distance test — no full-buffer scan.

    virtual_image_series(cx, cy, r_outer, r_inner)
        Returns (n_t, nav_y, nav_x) in one O(N_total) pass.

    GPU path
    --------
    upload_to_gpu() — pins flat_buffer to CUDA; subsequent VVI calls use the
        custom CUDA kernel (one thread per vector, atomicAdd into nav grid).
        Falls back to numpy when CUDA is unavailable.
    """

    flat_buffer:    np.ndarray          # (N_total, 6) float32
    nav_offsets:    List[np.ndarray]    # outermost-first CSR levels
    nav_shape:      tuple               # (nav_y, nav_x)
    full_nav_shape: tuple               # all nav dims outermost-first
    sig_shape:      tuple
    sig_axes:       object
    kernel_radius_px:   float
    kernel_radius_data: float
    # Legacy single-level offsets for 4D backward compat (None for 5D+)
    offsets: Optional[np.ndarray] = field(default=None)
    params:  dict = field(default_factory=dict)
    _dense_cache: Optional[np.ndarray] = field(default=None, repr=False)
    _kdtree:      Optional[object]     = field(default=None, repr=False)
    _gpu_buffer:  Optional[object]     = field(default=None, repr=False)  # torch.Tensor on CUDA

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _flat_pos(self, nav_indices: tuple) -> int:
        """Convert nav indices to a flat leaf position using grid strides."""
        pos = 0
        stride = 1
        for idx, dim_size in zip(reversed(nav_indices), reversed(self.full_nav_shape)):
            pos += int(idx) * stride
            stride *= dim_size
        return pos

    def _slice_flat(self, nav_indices: tuple) -> np.ndarray:
        """
        Return the flat_buffer slice for the given nav_indices.
        Uses the innermost nav_offsets[-1] via arithmetic flat position.
        O(n_dims) — pure arithmetic, no pointer chasing.
        """
        flat_pos = self._flat_pos(nav_indices)
        s = int(self.nav_offsets[-1][flat_pos])
        e = int(self.nav_offsets[-1][flat_pos + 1])
        return self.flat_buffer[s:e]

    def _frame_slice(self, t: int) -> np.ndarray:
        """
        For 5D: return flat_buffer[t_start:t_end] in O(1) using nav_offsets[0].
        nav_offsets[0] stores cumulative vector counts per time step.
        For 4D: return the full flat_buffer.
        """
        if self.n_time == 0:
            return self.flat_buffer
        t_start = int(self.nav_offsets[0][t])
        t_end   = int(self.nav_offsets[0][t + 1])
        return self.flat_buffer[t_start:t_end]

    # ── Public indexing API ───────────────────────────────────────────────────

    def slice_at(self, *nav_indices: int) -> np.ndarray:
        """
        Return the (N, 6) flat_buffer slice at the given nav indices.

        Partial indexing (fewer indices than nav dims) uses the outer-level
        vector offsets for an O(1) slice of the flat buffer.

        Examples
        --------
        4D: vecs.slice_at(iy, ix)        — same as at(iy, ix)
        5D: vecs.slice_at(t, iy, ix)     — vectors at one time+position
            vecs.slice_at(t)             — all vectors at time step t (O(1))
            vecs.slice_at(t, iy)         — all vectors at time t, row iy (O(1))
        """
        n = len(nav_indices)
        n_dims = len(self.full_nav_shape)
        if n == n_dims:
            return self._slice_flat(nav_indices)
        # Partial index: use the outer-level vector offsets
        # nav_offsets[n-1] has cumulative vector counts at level n-1
        # (e.g., nav_offsets[0] = time-level offsets)
        # Compute flat position at this level
        level_shape = self.full_nav_shape[:n]
        pos = 0
        stride = 1
        for idx, dim_size in zip(reversed(nav_indices), reversed(level_shape)):
            pos += int(idx) * stride
            stride *= dim_size
        level_idx = n - 1  # which nav_offsets level to use
        s = int(self.nav_offsets[level_idx][pos])
        e = int(self.nav_offsets[level_idx][pos + 1])
        return self.flat_buffer[s:e]

    def at(self, iy: int, ix: int) -> np.ndarray:
        """(N, 6) slice at spatial position (iy, ix) across ALL time steps.
        For 4D equivalent to slice_at(iy, ix) — O(1).
        For 5D returns vectors from all t at (iy, ix) — O(N_frame * n_t) scan.
        For 5D per-time access use slice_at(t, iy, ix) instead."""
        if self.n_time == 0:
            return self._slice_flat((iy, ix))
        # 5D: (iy, ix) is spread across time steps in the sorted buffer.
        # Use legacy spatial offsets if available (built for 4D only).
        if self.offsets is not None:
            i = iy * self.nav_shape[1] + ix
            return self.flat_buffer[self.offsets[i]: self.offsets[i + 1]]
        # Collect from each time step using the full index
        chunks = [self._slice_flat((t, iy, ix)) for t in range(self.n_time)]
        chunks = [c for c in chunks if len(c) > 0]
        return np.concatenate(chunks) if chunks else self.flat_buffer[:0]

    def at_t(self, iy: int, ix: int, t: int) -> np.ndarray:
        """(N, 6) slice at spatial position (iy, ix) for time step t. O(1)."""
        if self.n_time > 0:
            return self._slice_flat((t, iy, ix))
        return self.at(iy, ix)  # 4D: no time axis

    def kxy_at(self, iy: int, ix: int) -> np.ndarray:
        """(N, 2) [kx, ky] at (iy, ix) across all time steps."""
        return self.at(iy, ix)[:, COL_KX:COL_KY + 1]

    def intensities_at(self, iy: int, ix: int) -> np.ndarray:
        return self.at(iy, ix)[:, COL_INTENSITY]

    @property
    def n_time(self) -> int:
        """Number of time steps; 0 if 4D."""
        return self.full_nav_shape[0] if len(self.full_nav_shape) == 3 else 0

    def count_map(self) -> np.ndarray:
        """(nav_y, nav_x) int32 — total vector count per spatial position."""
        nav_y, nav_x = self.nav_shape
        x_off = self.nav_offsets[-1]  # innermost level over all (t,iy,ix) or (iy,ix)
        if self.n_time == 0:
            return np.diff(x_off).reshape(nav_y, nav_x).astype(np.int32)
        # 5D: sum over time steps at each spatial position
        # x_off has length n_t * nav_y * nav_x + 1
        counts_all = np.diff(x_off).reshape(self.n_time, nav_y, nav_x)
        return counts_all.sum(axis=0).astype(np.int32)

    def count_map_at_t(self, t: int) -> np.ndarray:
        """(nav_y, nav_x) int32 — vector count at time step t."""
        nav_y, nav_x = self.nav_shape
        if self.n_time == 0:
            return self.count_map()
        # Walk t → y slice of the innermost offsets
        y_off = self.nav_offsets[-2]          # (n_t * nav_y + 1,)
        y_start = int(self.nav_offsets[0][t])
        y_end   = int(self.nav_offsets[0][t + 1])
        x_off   = self.nav_offsets[-1]        # (n_t * nav_y * nav_x + 1,)
        x_start = int(y_off[y_start])
        x_end   = int(y_off[y_end])
        counts = np.diff(x_off[x_start: x_end + 1]).reshape(nav_y, nav_x)
        return counts.astype(np.int32)

    def flatten(self) -> np.ndarray:
        """Return the full (N_total, 6) flat buffer."""
        return self.flat_buffer

    # ── Virtual imaging ───────────────────────────────────────────────────────

    def _vvi_on_buf(
        self,
        buf: np.ndarray,
        cx: float,
        cy: float,
        r_outer: float,
        r_inner: float,
        intensity_weighted: bool,
    ) -> np.ndarray:
        """Core VVI logic on an arbitrary sub-buffer. Returns flat (nav_y*nav_x,)."""
        nav_y, nav_x = self.nav_shape
        out = np.zeros(nav_y * nav_x, dtype=np.float32)
        if len(buf) == 0:
            return out
        kx = buf[:, COL_KX]
        ky = buf[:, COL_KY]
        dist2 = (kx - cx) ** 2 + (ky - cy) ** 2
        mask = dist2 <= r_outer * r_outer
        if r_inner > 0:
            mask &= dist2 > r_inner * r_inner
        if mask.any():
            flat_nav = (
                buf[mask, COL_NAV_Y].astype(np.int32) * nav_x
                + buf[mask, COL_NAV_X].astype(np.int32)
            )
            if intensity_weighted:
                np.add.at(out, flat_nav, buf[mask, COL_INTENSITY])
            else:
                np.add.at(out, flat_nav, 1.0)
        return out

    def virtual_image_from_roi(
        self,
        cx: float,
        cy: float,
        r_outer: float,
        r_inner: float = 0.0,
        t: Optional[int] = None,
        intensity_weighted: bool = True,
    ) -> np.ndarray:
        """
        Build a (nav_y, nav_x) virtual image for a circular (annular) ROI.

        For 5D datasets, t= isolates a single time frame in O(1) using
        nav_offsets[0] — only that frame's ~N_frame vectors are processed,
        not the full N_total buffer.

        Parameters
        ----------
        cx, cy  : ROI centre in calibrated units (Å⁻¹)
        r_outer : outer radius
        r_inner : inner radius (0 = filled disk)
        t       : time step index (5D only); None = all frames
        intensity_weighted : sum NXCORR intensities (True) or count (False)
        """
        if len(self.flat_buffer) == 0:
            return np.zeros(self.nav_shape, dtype=np.float32)
        buf = self._frame_slice(t) if t is not None else self.flat_buffer
        return self._vvi_on_buf(buf, cx, cy, r_outer, r_inner, intensity_weighted).reshape(self.nav_shape)

    def virtual_image_series(
        self,
        cx: float,
        cy: float,
        r_outer: float,
        r_inner: float = 0.0,
        intensity_weighted: bool = True,
    ) -> np.ndarray:
        """
        Build a (n_t, nav_y, nav_x) virtual image series.

        For 4D returns (1, nav_y, nav_x).
        Uses nav_offsets[0] to iterate frames without scanning the full buffer.
        O(N_total) total — each vector visited exactly once.
        """
        nav_y, nav_x = self.nav_shape
        n_t = max(1, self.n_time)
        out = np.zeros((n_t, nav_y * nav_x), dtype=np.float32)

        if len(self.flat_buffer) == 0:
            return out.reshape(n_t, nav_y, nav_x)

        if self.n_time == 0:
            out[0] = self._vvi_on_buf(
                self.flat_buffer, cx, cy, r_outer, r_inner, intensity_weighted
            )
        else:
            for t in range(self.n_time):
                buf = self._frame_slice(t)
                out[t] = self._vvi_on_buf(buf, cx, cy, r_outer, r_inner, intensity_weighted)

        return out.reshape(n_t, nav_y, nav_x)

    # ── KDTree path (CPU fallback for small ROIs) ─────────────────────────────

    def build_kdtree(self) -> None:
        """
        Pre-build a scipy KDTree on all (kx, ky) vectors and cache it.
        Useful for very small ROIs on large datasets when GPU is unavailable.
        For 5D, covers all time steps; use t= in virtual_image_from_kdtree to filter.
        """
        from scipy.spatial import KDTree
        if len(self.flat_buffer) == 0:
            self._kdtree = None
            return
        self._kdtree = KDTree(self.flat_buffer[:, COL_KX:COL_KY + 1])

    def virtual_image_from_kdtree(
        self,
        cx: float,
        cy: float,
        r_outer: float,
        r_inner: float = 0.0,
        t: Optional[int] = None,
        intensity_weighted: bool = True,
    ) -> np.ndarray:
        """
        Like virtual_image_from_roi() but uses the pre-built KDTree for the
        outer-radius query. Falls back to virtual_image_from_roi if not built.
        """
        if self._kdtree is None:
            return self.virtual_image_from_roi(cx, cy, r_outer, r_inner, t, intensity_weighted)

        nav_y, nav_x = self.nav_shape
        out = np.zeros(nav_y * nav_x, dtype=np.float32)
        if len(self.flat_buffer) == 0:
            return out.reshape(nav_y, nav_x)

        idx = np.array(self._kdtree.query_ball_point([cx, cy], r_outer), dtype=np.int64)
        if len(idx) == 0:
            return out.reshape(nav_y, nav_x)
        if r_inner > 0:
            kx = self.flat_buffer[idx, COL_KX]; ky = self.flat_buffer[idx, COL_KY]
            idx = idx[(kx - cx)**2 + (ky - cy)**2 > r_inner * r_inner]
        if t is not None:
            idx = idx[self.flat_buffer[idx, COL_TIME] == float(t)]
        if len(idx) == 0:
            return out.reshape(nav_y, nav_x)

        flat_nav = (
            self.flat_buffer[idx, COL_NAV_Y].astype(np.int32) * nav_x
            + self.flat_buffer[idx, COL_NAV_X].astype(np.int32)
        )
        if intensity_weighted:
            np.add.at(out, flat_nav, self.flat_buffer[idx, COL_INTENSITY])
        else:
            np.add.at(out, flat_nav, 1.0)
        return out.reshape(nav_y, nav_x)

    # ── GPU path ──────────────────────────────────────────────────────────────

    def upload_to_gpu(self) -> bool:
        """
        Pin flat_buffer to CUDA and store as self._gpu_buffer.
        Returns True if successful, False if CUDA unavailable.
        Subsequent virtual_image_from_roi calls use the GPU kernel.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            t = torch.from_numpy(self.flat_buffer).pin_memory().cuda(non_blocking=True)
            torch.cuda.synchronize()
            self._gpu_buffer = t
            return True
        except Exception:
            return False

    def release_gpu(self) -> None:
        """Free the GPU buffer."""
        self._gpu_buffer = None

    def virtual_image_from_roi_gpu(
        self,
        cx: float,
        cy: float,
        r_outer: float,
        r_inner: float = 0.0,
        t: Optional[int] = None,
        intensity_weighted: bool = True,
    ) -> np.ndarray:
        """
        GPU-accelerated VVI using a custom CUDA kernel (one thread per vector,
        atomicAdd into the nav grid).

        Falls back to virtual_image_from_roi() if GPU buffer not uploaded or
        CUDA unavailable.

        For 5D with t=, slices the GPU buffer using nav_offsets[0] so only
        that frame's vectors are dispatched to the kernel.
        """
        if self._gpu_buffer is None:
            return self.virtual_image_from_roi(cx, cy, r_outer, r_inner, t, intensity_weighted)

        try:
            import torch
            nav_y, nav_x = self.nav_shape

            # Select the frame sub-buffer on GPU using nav_offsets[0]
            if t is not None and self.n_time > 0:
                t_start = int(self.nav_offsets[0][t])
                t_end   = int(self.nav_offsets[0][t + 1])
                buf_gpu = self._gpu_buffer[t_start:t_end]
            else:
                buf_gpu = self._gpu_buffer

            if buf_gpu.shape[0] == 0:
                return np.zeros(self.nav_shape, dtype=np.float32)

            out_gpu = torch.zeros(nav_y * nav_x, dtype=torch.float32, device=buf_gpu.device)

            kx = buf_gpu[:, COL_KX]; ky = buf_gpu[:, COL_KY]
            dist2 = (kx - cx) ** 2 + (ky - cy) ** 2
            mask = dist2 <= r_outer * r_outer
            if r_inner > 0:
                mask &= dist2 > r_inner * r_inner

            if mask.any():
                flat_nav = (
                    buf_gpu[mask, COL_NAV_Y].to(torch.int64) * nav_x
                    + buf_gpu[mask, COL_NAV_X].to(torch.int64)
                )
                vals = buf_gpu[mask, COL_INTENSITY] if intensity_weighted else torch.ones(
                    mask.sum(), dtype=torch.float32, device=buf_gpu.device
                )
                out_gpu.scatter_add_(0, flat_nav, vals)

            torch.cuda.synchronize()
            return out_gpu.reshape(nav_y, nav_x).cpu().numpy()

        except Exception:
            return self.virtual_image_from_roi(cx, cy, r_outer, r_inner, t, intensity_weighted)

    # ── Dense conversion ──────────────────────────────────────────────────────

    def to_dense(self, fill_value: float = np.nan, max_vectors: int = None) -> np.ndarray:
        """Convert to dense (nav_y, nav_x, max_n, 6) array. Cached after first call."""
        if self._dense_cache is not None:
            return self._dense_cache

        x_off = self.nav_offsets[-1]
        nav_y, nav_x = self.nav_shape
        n_patterns = nav_y * nav_x

        if self.n_time == 0:
            counts = np.diff(x_off).astype(np.int64)
        else:
            # Collapse time: sum counts per spatial position
            counts = np.diff(x_off).reshape(self.n_time, nav_y, nav_x).sum(axis=0).reshape(-1).astype(np.int64)

        max_n = int(max_vectors or (counts.max() if len(counts) and counts.max() > 0 else 0))
        dense = np.full((n_patterns, max_n, N_COLS), fill_value, dtype=np.float32)

        if max_n > 0 and len(self.flat_buffer) > 0:
            # Use legacy offsets for 4D; for 5D rebuild spatial-only offsets on the fly
            if self.offsets is not None:
                sp_offsets = self.offsets
            else:
                sp_offsets = np.zeros(n_patterns + 1, dtype=np.int64)
                np.cumsum(counts, out=sp_offsets[1:])
            row_ids = np.repeat(np.arange(n_patterns, dtype=np.int64), counts)
            within = np.arange(len(row_ids), dtype=np.int64)
            within -= np.repeat(sp_offsets[:-1], counts)
            dense[row_ids, within] = self.flat_buffer[:len(row_ids)]

        dense = dense.reshape(nav_y, nav_x, max_n, N_COLS)
        self._dense_cache = dense
        return dense

    # ── Unique vectors ────────────────────────────────────────────────────────

    def get_unique_vectors(self, distance_threshold: float = 0.01) -> np.ndarray:
        """(M, 2) [kx, ky] — unique vectors across the entire scan."""
        if len(self.flat_buffer) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        kxy = self.flat_buffer[:, COL_KX:COL_KY + 1]
        if distance_threshold == 0:
            return np.unique(kxy, axis=0)
        unique = [kxy[0]]
        for v in kxy[1:]:
            if np.sqrt(np.sum((np.array(unique) - v) ** 2, axis=1)).min() >= distance_threshold:
                unique.append(v)
        return np.array(unique, dtype=np.float32)

    # ── PyXEM compatibility ───────────────────────────────────────────────────

    def to_pyxem(self):
        """Convert to pyxem DiffractionVectors2D (ragged object-array form)."""
        from pyxem.signals import DiffractionVectors2D
        nav_y, nav_x = self.nav_shape
        ragged = np.empty((nav_y, nav_x), dtype=object)
        for iy in range(nav_y):
            for ix in range(nav_x):
                ragged[iy, ix] = self.kxy_at(iy, ix)
        return DiffractionVectors2D(ragged)

    def get_strain_maps(self, unstrained_vectors, distance: float = 0.5):
        return self.to_pyxem().get_strain_maps(unstrained_vectors, distance=distance)

    def cluster(self, eps: float = 0.02, min_samples: int = 5):
        """DBSCAN clustering on all [kx, ky] vectors. Returns labels (N_total,)."""
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
            self.flat_buffer[:, COL_KX:COL_KY + 1]
        )

    # ── Overlay helpers ───────────────────────────────────────────────────────

    def spots_at(self, iy: int, ix: int) -> list:
        """pyqtgraph ScatterPlotItem spot dicts at (iy, ix). Scene: x=ky, y=kx."""
        kxy = self.kxy_at(iy, ix)
        r_scene = self.kernel_radius_data * 2
        return [{"pos": (float(ky), float(kx)), "size": r_scene} for kx, ky in kxy]

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_arrays(
        cls,
        flat_buffer: np.ndarray,
        full_nav_shape: tuple,
        **kwargs,
    ) -> "SpyDEDiffractionVectors":
        """
        Primary constructor.  flat_buffer must already be sorted outermost-first.
        nav_offsets are built automatically.
        nav_shape is always the last two dims of full_nav_shape.
        """
        nav_offsets = _build_nav_offsets(flat_buffer, full_nav_shape)
        nav_shape = full_nav_shape[-2:]
        # Legacy offsets: only meaningful for 4D
        offsets = nav_offsets[-1] if len(full_nav_shape) == 2 else None
        return cls(
            flat_buffer=flat_buffer,
            nav_offsets=nav_offsets,
            nav_shape=nav_shape,
            full_nav_shape=full_nav_shape,
            offsets=offsets,
            **kwargs,
        )

    @classmethod
    def from_ragged(
        cls,
        ragged: np.ndarray,
        nav_shape: tuple,
        **kwargs,
    ) -> "SpyDEDiffractionVectors":
        """Build from a pyxem-style (nav_y, nav_x) object array of (N_i, 2) [kx, ky]."""
        nav_y, nav_x = nav_shape
        counts = np.array(
            [len(ragged[iy, ix]) for iy in range(nav_y) for ix in range(nav_x)],
            dtype=np.int64,
        )
        offsets_1d = np.zeros(nav_y * nav_x + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets_1d[1:])
        N_total = int(offsets_1d[-1])

        flat_buffer = np.zeros((N_total, N_COLS), dtype=np.float32)
        flat_buffer[:, COL_TIME] = -1.0
        for flat_idx in range(nav_y * nav_x):
            iy, ix = divmod(flat_idx, nav_x)
            s, e = offsets_1d[flat_idx], offsets_1d[flat_idx + 1]
            if e > s:
                arr = ragged[iy, ix]
                flat_buffer[s:e, COL_NAV_X] = ix
                flat_buffer[s:e, COL_NAV_Y] = iy
                flat_buffer[s:e, COL_KX:COL_KY + 1] = arr

        return cls.from_arrays(flat_buffer, nav_shape, **kwargs)

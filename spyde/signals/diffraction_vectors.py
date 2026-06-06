from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SpyDEDiffractionVectors:
    """
    Flat-buffer nested-tensor storage for diffraction vectors across a scan.

    Layout mirrors PyTorch NestedTensor / CSR sparse format:

        flat_buffer : (N_total, 5) float32
            columns: [nav_x, nav_y, kx_data, ky_data, intensity]

        offsets : (n_patterns + 1,) int64
            CSR row-pointer; position (iy, ix) maps to
            flat_idx = iy * nav_shape[1] + ix
            flat_buffer[offsets[flat_idx]:offsets[flat_idx+1]] are its vectors.

    Parameters
    ----------
    flat_buffer : np.ndarray  (N_total, 5) float32
    offsets     : np.ndarray  (n_patterns+1,) int64
    nav_shape   : (nav_y, nav_x)  — the 2-D spatial navigation grid
    full_nav_shape : same as nav_shape for 4D; (time, nav_y, nav_x) for 5D
    sig_shape   : (ky_size, kx_size)
    sig_axes    : hyperspy AxesManager signal_axes (or None for tests)
    kernel_radius_px   : float — disk radius used during peak finding (pixels)
    kernel_radius_data : float — disk radius in Å⁻¹
    params      : dict — full params snapshot (sigma, threshold, min_distance…)
    """

    flat_buffer: np.ndarray
    offsets: np.ndarray
    nav_shape: tuple
    full_nav_shape: tuple
    sig_shape: tuple
    sig_axes: object
    kernel_radius_px: float
    kernel_radius_data: float
    params: dict = field(default_factory=dict)
    _dense_cache: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def at(self, iy: int, ix: int) -> np.ndarray:
        """(N, 5) slice of flat_buffer at navigation position (iy, ix)."""
        i = iy * self.nav_shape[1] + ix
        return self.flat_buffer[self.offsets[i]: self.offsets[i + 1]]

    def kxy_at(self, iy: int, ix: int) -> np.ndarray:
        """(N, 2) [kx, ky] in data units at (iy, ix)."""
        return self.at(iy, ix)[:, 2:4]

    def intensities_at(self, iy: int, ix: int) -> np.ndarray:
        return self.at(iy, ix)[:, 4]

    def count_map(self) -> np.ndarray:
        """(nav_y, nav_x) int32 — vector count at each navigation position."""
        return np.diff(self.offsets).reshape(self.nav_shape).astype(np.int32)

    def flatten(self) -> np.ndarray:
        """Return the full (N_total, 5) flat buffer."""
        return self.flat_buffer

    # ── Dense conversion ──────────────────────────────────────────────────────

    def to_dense(self, fill_value: float = np.nan, max_vectors: int = None) -> np.ndarray:
        """
        Convert to dense (nav_y, nav_x, max_n, 5) array.

        Cached after first call.  Uses vectorised numpy scatter — no Python
        loop over nav positions, so it is O(N_total + nav_y*nav_x) instead of
        O(nav_y * nav_x * loop_overhead).

        fill_value pads positions with fewer than max_n vectors.
        """
        if self._dense_cache is not None:
            return self._dense_cache

        counts = np.diff(self.offsets).astype(np.int64)  # (n_patterns,)
        max_n = int(max_vectors or (counts.max() if len(counts) and counts.max() > 0 else 0))
        nav_y, nav_x = self.nav_shape
        n_patterns = nav_y * nav_x

        dense = np.full((n_patterns, max_n, 5), fill_value, dtype=np.float32)

        if max_n > 0 and len(self.flat_buffer) > 0:
            # Build a (N_total,) array of which pattern each vector belongs to
            # using np.repeat — this is the CSR row-expand operation, O(N_total).
            row_ids = np.repeat(np.arange(n_patterns, dtype=np.int64), counts)
            # Build a (N_total,) array of the within-row index (0, 1, 2, …) for
            # each vector.  np.arange over each run, concatenated via cumsum trick.
            within = np.arange(len(row_ids), dtype=np.int64)
            within -= np.repeat(self.offsets[:-1], counts)
            # Scatter into the dense array — a single advanced-indexing write
            dense[row_ids, within] = self.flat_buffer

        dense = dense.reshape(nav_y, nav_x, max_n, 5)
        self._dense_cache = dense
        return dense

    # ── Unique vectors ────────────────────────────────────────────────────────

    def get_unique_vectors(self, distance_threshold: float = 0.01) -> np.ndarray:
        """(M, 2) [kx, ky] — unique vectors across the entire scan."""
        if len(self.flat_buffer) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        kxy = self.flat_buffer[:, 2:4]
        if distance_threshold == 0:
            return np.unique(kxy, axis=0)
        # Iterative distance comparison (same algorithm as pyxem)
        unique = [kxy[0]]
        for v in kxy[1:]:
            dists = np.sqrt(np.sum((np.array(unique) - v) ** 2, axis=1))
            if dists.min() >= distance_threshold:
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

    # ── Downstream gateways ───────────────────────────────────────────────────

    def get_strain_maps(self, unstrained_vectors, distance: float = 0.5):
        """Delegate to pyxem after converting to DiffractionVectors2D."""
        dv = self.to_pyxem()
        return dv.get_strain_maps(unstrained_vectors, distance=distance)

    def cluster(self, eps: float = 0.02, min_samples: int = 5):
        """DBSCAN clustering on all [kx, ky] vectors. Returns labels (N_total,)."""
        from sklearn.cluster import DBSCAN
        kxy = self.flat_buffer[:, 2:4]
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(kxy)

    # ── Overlay helpers ───────────────────────────────────────────────────────

    def spots_at(self, iy: int, ix: int) -> list:
        """
        Return pyqtgraph ScatterPlotItem spot dicts for navigation position (iy, ix).

        Scene coordinate convention (pyqtgraph col-major):
            scene_x = ky,  scene_y = kx
        """
        kxy = self.kxy_at(iy, ix)
        r_scene = self.kernel_radius_data * 2  # diameter for 'o' symbol
        return [
            {"pos": (float(ky), float(kx)), "size": r_scene}
            for kx, ky in kxy
        ]

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_ragged(
        cls,
        ragged: np.ndarray,
        nav_shape: tuple,
        **kwargs,
    ) -> "SpyDEDiffractionVectors":
        """
        Build from a pyxem-style (nav_y, nav_x) object array of (N_i, 2) [kx, ky].
        """
        nav_y, nav_x = nav_shape
        flat_idx_iter = (
            (iy, ix) for iy in range(nav_y) for ix in range(nav_x)
        )
        counts = np.array(
            [len(ragged[iy, ix]) for iy in range(nav_y) for ix in range(nav_x)],
            dtype=np.int64,
        )
        offsets = np.zeros(nav_y * nav_x + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        N_total = int(offsets[-1])

        flat_buffer = np.zeros((N_total, 5), dtype=np.float32)
        for flat_idx, (iy, ix) in enumerate(flat_idx_iter):
            s, e = offsets[flat_idx], offsets[flat_idx + 1]
            if e > s:
                arr = ragged[iy, ix]  # (N, 2) — kx, ky
                flat_buffer[s:e, 0] = ix
                flat_buffer[s:e, 1] = iy
                flat_buffer[s:e, 2:4] = arr

        return cls(
            flat_buffer=flat_buffer,
            offsets=offsets,
            nav_shape=nav_shape,
            full_nav_shape=nav_shape,
            **kwargs,
        )

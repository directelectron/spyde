"""
orientation_map.py — SpyDEOrientationMap: standalone container for batch
orientation-mapping results.

Same philosophy as SpyDEDiffractionVectors: a small, self-contained result
(a few MB for a 256x256 scan) that can be saved, reloaded and fully
visualised — orientation map, IPF positions, in-plane rotation — without
the raw dataset or the simulation library.  Orientations are stored as
resolved quaternions (template rotation ⊗ in-plane rotation), so no library
lookup is ever needed after compute.

No Qt imports here — this module is used on dask workers and in tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

_DIRECTIONS = ("x", "y", "z")


def _direction_vector(direction: str):
    from orix.vector import Vector3d
    d = str(direction).lower()
    if d == "x":
        return Vector3d.xvector()
    if d == "y":
        return Vector3d.yvector()
    return Vector3d.zvector()


def orix_phase_from_dict(meta: dict):
    """Rebuild a minimal orix Phase from {'name', 'point_group'}."""
    from orix.crystal_map import Phase
    return Phase(name=meta.get("name", "phase"),
                 point_group=meta.get("point_group", "m-3m"))


def phase_to_dict(phase) -> dict:
    """Serialise the parts of an orix Phase the container needs."""
    pg = getattr(phase, "point_group", None)
    return {
        "name": str(getattr(phase, "name", "phase")),
        "point_group": str(getattr(pg, "name", "m-3m")),
    }


def ipf_xy_for_rotations(rotations, phase, direction: str = "z"):
    """
    Stereographic (x, y) of rotation * direction folded into the fundamental
    sector.  Vectorised: `rotations` may hold any number of orientations.

    Returns (x, y) float arrays of the broadcast shape.
    """
    from orix.projections import StereographicProjection
    vec = rotations * _direction_vector(direction)
    vec = vec.in_fundamental_sector(phase.point_group)
    x, y = StereographicProjection().vector2xy(vec)
    return np.atleast_1d(np.asarray(x, dtype=float)), \
        np.atleast_1d(np.asarray(y, dtype=float))


def ipf_triangle_xy(phase):
    """(xy_edges, label_xy, label_texts) outlining the fundamental sector.

    Same output as the refine-step helper in actions/pyxem.py — kept here so
    GUI code and tests can use it without importing Qt-heavy modules.
    """
    from orix.projections import StereographicProjection
    from pyxem.signals.indexation_results import (
        _closed_edges_in_hemisphere, _get_ipf_axes_labels,
    )
    s = StereographicProjection()
    sector = phase.point_group.fundamental_sector
    edges = _closed_edges_in_hemisphere(sector.edges, sector)
    ex, ey = s.vector2xy(edges)
    xy_edges = np.vstack((ex, ey)).T
    try:
        raw_labels = _get_ipf_axes_labels(sector.vertices,
                                          symmetry=phase.point_group)
        labels = [lbl.replace("$", "") for lbl in raw_labels]
        lx, ly = s.vector2xy(sector.vertices)
        lx = np.atleast_1d(np.array(lx, dtype=float))
        ly = np.atleast_1d(np.array(ly, dtype=float))
        cx, cy = float(np.mean(lx)), float(np.mean(ly))
        label_xy = np.vstack([lx + 0.25 * (lx - cx),
                              ly + 0.25 * (ly - cy)]).T
    except Exception:
        labels, label_xy = [], np.empty((0, 2))
    return xy_edges, label_xy, labels


@dataclass
class SpyDEOrientationMap:
    """
    Batch orientation-mapping result for a 2D scan.

    quats     : (ny, nx, n_best, 4) float32 — resolved orientations
                (w, x, y, z), best-first along n_best.
    corr      : (ny, nx, n_best) float32 — correlation scores.
    phase_idx : (ny, nx, n_best) int16 — index into `phases`.
    mirror    : (ny, nx, n_best) int8 — template mirror factor (+1/-1).
    phases    : list of {'name', 'point_group'} dicts (see phase_to_dict).
    nav_axes  : axis records duck-typing .scale/.offset/.units/.name
                (hyperspy axes or _AxisLite), or None.
    params    : compute parameters for provenance.
    """

    quats: np.ndarray
    corr: np.ndarray
    phase_idx: np.ndarray
    mirror: np.ndarray
    phases: List[dict]
    nav_axes: object = None
    params: dict = field(default_factory=dict)
    # Provenance record ({"action", "params", "spyde_version"}) — same dict
    # convention as commit._stamp_provenance (script/app interchangeable).
    provenance: Optional[dict] = field(default=None)
    _phase_cache: dict = field(default_factory=dict, repr=False)

    # ── Basic properties ──────────────────────────────────────────────────────

    @property
    def nav_shape(self) -> tuple:
        return tuple(self.quats.shape[:2])

    @property
    def n_best(self) -> int:
        return int(self.quats.shape[2])

    @property
    def n_phases(self) -> int:
        return len(self.phases)

    def orix_phase(self, i: int = 0):
        """Cached orix Phase for phase index i."""
        if i not in self._phase_cache:
            self._phase_cache[i] = orix_phase_from_dict(self.phases[i])
        return self._phase_cache[i]

    def _rotations(self, sel):
        from orix.quaternion import Rotation
        return Rotation(self.quats[sel])

    # ── Navigator images ──────────────────────────────────────────────────────

    def ipf_color_map(self, direction: str = "z") -> np.ndarray:
        """(ny, nx, 3) uint8 IPF color map (best match per position).

        Multi-phase: each position is colored by its matched phase's color
        key — same direction for all phases.
        """
        from orix.plot import IPFColorKeyTSL
        from orix.quaternion import Orientation, Rotation

        ny, nx = self.nav_shape
        rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
        best_phase = self.phase_idx[..., 0]
        d = _direction_vector(direction)
        for i in range(self.n_phases):
            mask = best_phase == i
            if not mask.any():
                continue
            phase = self.orix_phase(i)
            key = IPFColorKeyTSL(phase.point_group.laue, direction=d)
            ori = Orientation(Rotation(self.quats[mask, 0]),
                              symmetry=phase.point_group)
            colors = key.orientation2color(ori)
            rgb[mask] = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
        return rgb

    def ipf_sphere_points(self, direction: str = "z", max_points: int = 1_000_000):
        """Per-position reduced crystal directions ON THE UNIT SPHERE + their IPF
        colour, for the 3-D IPF explorer (anyplotlib ``scatter3d``).

        Returns ``(xyz (M, 3) float32, rgb (M, 3) uint8)`` for the best match at
        every position: ``xyz`` is ``(rotation · direction)`` folded into the
        point group's fundamental sector (the same point the 2-D IPF shows, but
        kept in 3-D), ``rgb`` is the matching IPF colour. The WebGPU instanced
        scatter path (``scatter3d(..., gpu=True)``) renders comfortably up to
        ~1M points, so every nav pixel is kept by default; only a scan bigger
        than ``max_points`` (a safety ceiling, not the interactive budget) is
        uniformly subsampled.
        """
        from orix.plot import IPFColorKeyTSL
        from orix.quaternion import Orientation, Rotation

        best_phase = self.phase_idx[..., 0].reshape(-1)
        best_q = self.quats[..., 0, :].reshape(-1, 4)
        d = _direction_vector(direction)
        xyz = np.full((best_q.shape[0], 3), np.nan, dtype=np.float32)
        rgb = np.zeros((best_q.shape[0], 3), dtype=np.uint8)
        for i in range(self.n_phases):
            mask = best_phase == i
            if not mask.any():
                continue
            pg = self.orix_phase(i).point_group
            rot = Rotation(best_q[mask])
            v = (rot * d).in_fundamental_sector(pg)
            xyz[mask] = np.asarray(v.data, dtype=np.float32)
            key = IPFColorKeyTSL(pg.laue, direction=d)
            colors = key.orientation2color(Orientation(rot, symmetry=pg))
            rgb[mask] = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

        valid = np.isfinite(xyz).all(axis=1)
        xyz, rgb = xyz[valid], rgb[valid]
        if len(xyz) > max_points:
            stride = int(np.ceil(len(xyz) / max_points))
            xyz, rgb = xyz[::stride], rgb[::stride]
        return xyz, rgb

    def correlation_map(self) -> np.ndarray:
        """(ny, nx) float32 best-match correlation."""
        return np.ascontiguousarray(self.corr[..., 0])

    def phase_map(self) -> np.ndarray:
        """(ny, nx) int16 best-match phase index."""
        return np.ascontiguousarray(self.phase_idx[..., 0])

    # ── IPF coordinates (2D stereographic) ────────────────────────────────────

    def ipf_xy(self, iy: int, ix: int, direction: str = "z"):
        """
        IPF positions of all n_best candidates at one scan position.

        Returns (xy (n_best, 2) float, phase_idx (n_best,) int, corr (n_best,)).
        Candidates of different phases project into their own phase's sector.
        """
        pidx = self.phase_idx[iy, ix].astype(int)
        out = np.zeros((self.n_best, 2), dtype=float)
        for i in np.unique(pidx):
            sel = pidx == i
            rot = self._rotations((iy, ix, sel))
            x, y = ipf_xy_for_rotations(rot, self.orix_phase(int(i)), direction)
            out[sel, 0] = x
            out[sel, 1] = y
        return out, pidx, self.corr[iy, ix].copy()

    def ipf_xy_roi(self, ys, xs, phase: int = 0, best_only: bool = True,
                   direction: str = "z", max_points: int = 50_000):
        """
        IPF positions for every position inside a navigation ROI.

        ys, xs : slices (Integrate-mode region)
        best_only : only the best candidate per position (False = all n_best,
            the 'heat map' mode)
        Returns (xy (M, 2), corr (M,)) for candidates matching `phase`,
        uniformly subsampled to at most max_points.
        """
        q = self.quats[ys, xs]
        c = self.corr[ys, xs]
        p = self.phase_idx[ys, xs]
        if best_only:
            q, c, p = q[..., :1, :], c[..., :1], p[..., :1]
        q = q.reshape(-1, 4)
        c = c.reshape(-1)
        p = p.reshape(-1)
        sel = p == phase
        q, c = q[sel], c[sel]
        if len(q) == 0:
            return np.empty((0, 2)), np.empty((0,))
        if len(q) > max_points:
            stride = int(np.ceil(len(q) / max_points))
            q, c = q[::stride], c[::stride]
        from orix.quaternion import Rotation
        x, y = ipf_xy_for_rotations(Rotation(q), self.orix_phase(phase),
                                    direction)
        return np.stack([x, y], axis=1), c

    # ── 3D IPF (beam direction + in-plane tangent) ────────────────────────────

    def ipf_xyz(self, iy: int, ix: int, n: int = 0, direction: str = "z"):
        """
        Full-orientation glyph data for the 3D view at one position.

        Returns (v (3,), tangent (3,), phase_idx int):
        v       — rotation * direction folded into the fundamental sector
                  (unit vector — the point on the sphere, same as the 2D IPF)
        tangent — the in-plane reference direction folded by the SAME
                  symmetry operation and projected onto the tangent plane at
                  v.  This is the information the 2D IPF discards: two
                  orientations with equal v but different in-plane rotation
                  get different tangents.
        """
        from orix.quaternion import Rotation
        from orix.vector import Vector3d

        rot = Rotation(self.quats[iy, ix, n])
        pidx = int(self.phase_idx[iy, ix, n])
        pg = self.orix_phase(pidx).point_group

        d = _direction_vector(direction)
        # In-plane reference: any direction orthogonal to d
        ref = Vector3d.xvector() if str(direction).lower() != "x" \
            else Vector3d.yvector()

        v = rot * d
        u = rot * ref
        v_fs = v.in_fundamental_sector(pg)

        # Find the symmetry operation that folded v into the sector and apply
        # the same one to u, so the tangent is consistent with the point.
        sym_v = (pg.outer(v)).unit
        target = v_fs.unit.data.ravel()
        diffs = np.linalg.norm(sym_v.data.reshape(-1, 3) - target, axis=1)
        op = pg[int(np.argmin(diffs))]
        u_fs = (op * u).unit

        v_arr = v_fs.unit.data.ravel()
        u_arr = u_fs.data.ravel()
        tangent = u_arr - np.dot(u_arr, v_arr) * v_arr
        norm = np.linalg.norm(tangent)
        if norm > 1e-9:
            tangent = tangent / norm
        return v_arr.astype(float), tangent.astype(float), pidx

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Compressed .npz — small, self-contained, no library needed."""
        axes_meta = []
        for ax in (self.nav_axes or []):
            axes_meta.append(dict(
                scale=float(getattr(ax, "scale", 1.0)),
                offset=float(getattr(ax, "offset", 0.0)),
                size=int(getattr(ax, "size", 0)),
                units=str(getattr(ax, "units", "") or ""),
                name=str(getattr(ax, "name", "") or ""),
            ))
        meta = dict(phases=self.phases, params=self.params,
                    nav_axes=axes_meta)
        np.savez_compressed(
            path,
            quats=self.quats,
            corr=self.corr,
            phase_idx=self.phase_idx,
            mirror=self.mirror,
            meta_json=np.frombuffer(
                json.dumps(meta, default=str).encode("utf-8"), dtype=np.uint8
            ),
        )

    @classmethod
    def load(cls, path: str) -> "SpyDEOrientationMap":
        from spyde.signals.diffraction_vectors import _AxisLite
        with np.load(path) as z:
            quats = z["quats"].astype(np.float32)
            corr = z["corr"].astype(np.float32)
            phase_idx = z["phase_idx"].astype(np.int16)
            mirror = z["mirror"].astype(np.int8)
            meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
        nav_axes = [_AxisLite(**a) for a in meta.get("nav_axes", [])] or None
        return cls(
            quats=quats, corr=corr, phase_idx=phase_idx, mirror=mirror,
            phases=list(meta.get("phases", [])),
            nav_axes=nav_axes,
            params=meta.get("params", {}) or {},
        )

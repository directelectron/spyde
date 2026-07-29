"""
particles.py — :class:`SpyDEParticles`, ragged per-frame particle storage.

Particles-per-frame is a **ragged per-navigation-position collection** — exactly
the shape :class:`spyde.signals.diffraction_vectors.SpyDEDiffractionVectors`
already solves. This mirrors that design rather than inventing a second pattern,
and rather than ParticleSpy's list-of-``Particle``-objects, which does not survive
the target scale.

Why not a list of objects, and why not full-frame label images
--------------------------------------------------------------
The target is thousands of frames with hundreds of particles each — around 1.5M
particles (DRIFT_AND_PARTICLES_PLAN.md §0.1). Do the arithmetic before choosing a
representation:

===========================================  ============  ==========
representation                                per particle  total
===========================================  ============  ==========
property row (21 x float32)                          84 B      126 MB
bbox bitmap (packed 64^2 crop)                      512 B      770 MB
contour polygon (~40 pts x 2 x float32)             320 B      480 MB
**contour polygon, int16**                       **~80 B**  **120 MB**
===========================================  ============  ==========

So: **properties always resident, outlines as int16 contours, and masks are
optional.** A full-frame ``int32`` label image is 64 MB *per frame* at 4096^2 and
is never stored at any setting — :meth:`SpyDEParticles.render_frame` paints one
frame on demand from its contours.

Contours are quantised to whole pixels. That is a **display** fidelity choice, not
a measurement one: every measured quantity lives in the property row, computed at
full precision from the original mask by :mod:`spyde.particles.measure`. Rounding
an outline for drawing cannot corrupt an area, because the area was never derived
from the outline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

FORMAT_VERSION = 1

#: Column layout of ``flat_buffer``. Order is load-bearing — it is the on-disk
#: layout. Append new columns at the END and bump ``FORMAT_VERSION``.
COLUMNS: tuple[str, ...] = (
    "t",                # frame index (float for a uniform buffer dtype)
    "label",            # per-frame instance label, 1-based (0 = background)
    "y", "x",           # centroid, calibrated units
    "area",
    "equiv_diameter",
    "major_axis", "minor_axis",
    "perimeter",
    "circularity",
    "eccentricity",
    "solidity",
    "intensity_mean", "intensity_max", "intensity_std",
    "background",       # mean intensity in the dilated boundary ring
    "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",   # pixel indices, half-open
    "track_id",         # -1 until the linker runs
)
COL: dict[str, int] = {name: i for i, name in enumerate(COLUMNS)}
N_COLUMNS = len(COLUMNS)

#: Columns a user can sort, histogram or colour by — i.e. real measurements,
#: excluding bookkeeping. Drives the table dock and the histogram window.
MEASURED_COLUMNS: tuple[str, ...] = (
    "area", "equiv_diameter", "major_axis", "minor_axis", "perimeter",
    "circularity", "eccentricity", "solidity",
    "intensity_mean", "intensity_max", "intensity_std", "background",
)

#: Which measured columns scale with length, area, or not at all — used to apply
#: pixel-size calibration exactly once, in one place.
_LENGTH_COLUMNS = ("y", "x", "equiv_diameter", "major_axis", "minor_axis", "perimeter")
_AREA_COLUMNS = ("area",)


@dataclass
class SpyDEParticles:
    """Ragged per-frame particle table with optional outlines.

    Parameters
    ----------
    flat_buffer
        ``(N_total, N_COLUMNS)`` float32, sorted by frame index ``t``.
    t_offsets
        ``(n_frames + 1,)`` int64 CSR row pointers. ``t_offsets[i]:t_offsets[i+1]``
        is frame *i*'s slice — an O(1) lookup, no search.
    frame_shape
        ``(h, w)`` of the source frames, in pixels.
    contours, contour_offsets
        Optional outlines: ``(M, 2)`` int16 ``(y, x)`` pixel coordinates and an
        ``(N_total + 1,)`` int64 index. ``None`` when segmentation ran with
        ``store_masks=False`` (the default for very long movies).
    scale, units
        Pixel size and its unit, so ``area`` is in ``units**2``. ``scale=1.0`` with
        ``units="px"`` means uncalibrated.
    """

    flat_buffer: np.ndarray
    t_offsets: np.ndarray
    frame_shape: tuple[int, int]
    contours: np.ndarray | None = None
    contour_offsets: np.ndarray | None = None
    scale: float = 1.0
    units: str = "px"
    params: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.flat_buffer = np.ascontiguousarray(self.flat_buffer, dtype=np.float32)
        if self.flat_buffer.ndim != 2 or self.flat_buffer.shape[1] != N_COLUMNS:
            raise ValueError(
                f"flat_buffer must be (N, {N_COLUMNS}); got {self.flat_buffer.shape}"
            )
        self.t_offsets = np.ascontiguousarray(self.t_offsets, dtype=np.int64)
        if self.t_offsets.ndim != 1 or self.t_offsets.size < 1:
            raise ValueError(f"t_offsets must be 1-D and non-empty; got {self.t_offsets.shape}")
        if int(self.t_offsets[0]) != 0 or int(self.t_offsets[-1]) != len(self.flat_buffer):
            raise ValueError(
                f"t_offsets must span 0..{len(self.flat_buffer)}; got "
                f"{int(self.t_offsets[0])}..{int(self.t_offsets[-1])}"
            )
        if np.any(np.diff(self.t_offsets) < 0):
            raise ValueError("t_offsets must be non-decreasing")
        self.frame_shape = (int(self.frame_shape[0]), int(self.frame_shape[1]))

        if (self.contours is None) != (self.contour_offsets is None):
            raise ValueError("contours and contour_offsets must both be set or both None")
        if self.contours is not None:
            self.contours = np.ascontiguousarray(self.contours, dtype=np.int16)
            if self.contours.ndim != 2 or self.contours.shape[1] != 2:
                raise ValueError(f"contours must be (M, 2); got {self.contours.shape}")
            self.contour_offsets = np.ascontiguousarray(self.contour_offsets, dtype=np.int64)
            if self.contour_offsets.size != len(self.flat_buffer) + 1:
                raise ValueError(
                    f"contour_offsets must be ({len(self.flat_buffer) + 1},); "
                    f"got {self.contour_offsets.shape}"
                )

    # ── shape ────────────────────────────────────────────────────────────────

    @property
    def n_particles(self) -> int:
        return int(len(self.flat_buffer))

    @property
    def n_frames(self) -> int:
        return int(self.t_offsets.size - 1)

    @property
    def has_masks(self) -> bool:
        return self.contours is not None

    @property
    def has_tracks(self) -> bool:
        """True once the linker has assigned track ids."""
        if self.n_particles == 0:
            return False
        return bool(np.any(self.flat_buffer[:, COL["track_id"]] >= 0))

    # ── per-frame access ─────────────────────────────────────────────────────

    def at(self, t: int) -> np.ndarray:
        """Frame *t*'s ``(n, N_COLUMNS)`` block. O(1) — a view, not a copy."""
        t = int(t)
        if not 0 <= t < self.n_frames:
            raise IndexError(f"frame {t} outside 0..{self.n_frames - 1}")
        return self.flat_buffer[self.t_offsets[t]:self.t_offsets[t + 1]]

    def indices_at(self, t: int) -> np.ndarray:
        """Global particle indices belonging to frame *t*."""
        t = int(t)
        if not 0 <= t < self.n_frames:
            raise IndexError(f"frame {t} outside 0..{self.n_frames - 1}")
        return np.arange(self.t_offsets[t], self.t_offsets[t + 1], dtype=np.int64)

    def column(self, name: str) -> np.ndarray:
        """One column across every particle."""
        try:
            return self.flat_buffer[:, COL[name]]
        except KeyError:
            raise KeyError(
                f"unknown column {name!r}; available: {', '.join(COLUMNS)}"
            ) from None

    def contour_at(self, index: int) -> np.ndarray:
        """``(k, 2)`` int16 ``(y, x)`` outline of global particle *index*."""
        if self.contours is None:
            raise ValueError(
                "no outlines stored (segmentation ran with store_masks=False)"
            )
        i = int(index)
        if not 0 <= i < self.n_particles:
            raise IndexError(f"particle {i} outside 0..{self.n_particles - 1}")
        return self.contours[self.contour_offsets[i]:self.contour_offsets[i + 1]]

    # ── navigator traces ─────────────────────────────────────────────────────

    def count_series(self) -> np.ndarray:
        """``(n_frames,)`` particle count per frame — the navigator's count lane.

        Straight from the CSR row pointers, so it is O(n_frames) regardless of how
        many particles there are.
        """
        return np.diff(self.t_offsets).astype(np.float32)

    def property_series(self, name: str, reduce: str = "mean") -> np.ndarray:
        """``(n_frames,)`` per-frame reduction of a column — e.g. mean size lane.

        Empty frames yield NaN rather than 0: a frame with no particles has no
        mean size, and plotting it as zero would draw a spurious spike down to the
        axis that reads as a real physical event.
        """
        col = self.column(name)
        fn = {"mean": np.mean, "sum": np.sum, "max": np.max,
              "min": np.min, "median": np.median, "std": np.std}.get(reduce)
        if fn is None:
            raise ValueError(
                f"unknown reduce {reduce!r}; expected mean/sum/max/min/median/std"
            )
        out = np.full(self.n_frames, np.nan, dtype=np.float32)
        for t in range(self.n_frames):
            s, e = self.t_offsets[t], self.t_offsets[t + 1]
            if e > s:
                vals = col[s:e]
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    out[t] = fn(finite)
        return out

    # ── rendering ────────────────────────────────────────────────────────────

    def render_frame(self, t: int, *, value: str = "label") -> np.ndarray:
        """Paint frame *t*'s particles into an ``int32`` label image.

        Built on demand and never cached here — a 4096^2 label image is 64 MB, so
        holding even a handful would dwarf the entire particle table. The caller
        (the overlay) keeps only the frame it is displaying.

        Parameters
        ----------
        value
            ``"label"`` fills each particle with its per-frame label;
            ``"track"`` fills with ``track_id + 1`` so the overlay can colour by
            identity across frames (0 stays background). ``"index"`` fills with the
            global particle index + 1, which is what a click-to-select hit test
            wants.
        """
        from skimage.draw import polygon as sk_polygon

        if self.contours is None:
            raise ValueError(
                "cannot render outlines: segmentation ran with store_masks=False"
            )
        h, w = self.frame_shape
        out = np.zeros((h, w), dtype=np.int32)
        for gi in self.indices_at(t):
            c = self.contour_at(gi)
            if len(c) < 3:
                continue
            rr, cc = sk_polygon(c[:, 0].astype(np.intp), c[:, 1].astype(np.intp),
                               shape=(h, w))
            if value == "label":
                fill = int(self.flat_buffer[gi, COL["label"]])
            elif value == "track":
                fill = int(self.flat_buffer[gi, COL["track_id"]]) + 1
            elif value == "index":
                fill = int(gi) + 1
            else:
                raise ValueError(
                    f"unknown value {value!r}; expected 'label', 'track' or 'index'"
                )
            out[rr, cc] = fill
        return out

    def mask_at(self, index: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """``(mask, (y0, x0, y1, x1))`` — one particle's boolean mask and its bbox.

        Cropped to the bounding box, so this stays small no matter the frame size.
        This is what Wave D's per-particle mean diffraction pattern slices with.
        """
        from skimage.draw import polygon as sk_polygon

        c = self.contour_at(index)
        row = self.flat_buffer[int(index)]
        y0, x0 = int(row[COL["bbox_y0"]]), int(row[COL["bbox_x0"]])
        y1, x1 = int(row[COL["bbox_y1"]]), int(row[COL["bbox_x1"]])
        h, w = max(1, y1 - y0), max(1, x1 - x0)
        m = np.zeros((h, w), dtype=bool)
        if len(c) >= 3:
            rr, cc = sk_polygon((c[:, 0] - y0).astype(np.intp),
                                (c[:, 1] - x0).astype(np.intp), shape=(h, w))
            m[rr, cc] = True
        return m, (y0, x0, y1, x1)

    # ── export ───────────────────────────────────────────────────────────────

    def to_dataframe(self):
        """A pandas DataFrame of every particle. Requires pandas at call time."""
        import pandas as pd
        return pd.DataFrame(self.flat_buffer, columns=list(COLUMNS))

    def to_csv(self, path: str) -> None:
        """Write the property table as CSV, with a units line in the header."""
        header = ",".join(COLUMNS)
        np.savetxt(
            path, self.flat_buffer, delimiter=",", header=header, comments="",
            fmt="%.6g",
        )

    # ── serialisation ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        meta = {
            "format_version": FORMAT_VERSION,
            "columns": list(COLUMNS),
            "frame_shape": list(self.frame_shape),
            "scale": float(self.scale),
            "units": self.units,
            "params": self.params,
            "provenance": self.provenance,
        }
        arrays = {
            "flat_buffer": self.flat_buffer,
            "t_offsets": self.t_offsets,
            "meta": np.array(json.dumps(meta)),
        }
        if self.contours is not None:
            arrays["contours"] = self.contours
            arrays["contour_offsets"] = self.contour_offsets
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "SpyDEParticles":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            ver = meta.get("format_version")
            if ver != FORMAT_VERSION:
                raise ValueError(
                    f"unsupported SpyDEParticles format version {ver!r} "
                    f"(this build reads {FORMAT_VERSION})"
                )
            saved_cols = tuple(meta.get("columns") or ())
            if saved_cols != COLUMNS:
                raise ValueError(
                    "column layout changed since this file was written "
                    f"({len(saved_cols)} columns on disk, {N_COLUMNS} expected)"
                )
            return cls(
                flat_buffer=z["flat_buffer"],
                t_offsets=z["t_offsets"],
                frame_shape=tuple(meta["frame_shape"]),
                contours=z["contours"] if "contours" in z.files else None,
                contour_offsets=(z["contour_offsets"]
                                 if "contour_offsets" in z.files else None),
                scale=meta.get("scale", 1.0),
                units=meta.get("units", "px"),
                params=meta.get("params") or {},
                provenance=meta.get("provenance"),
            )

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_frames(
        cls,
        per_frame: Sequence[np.ndarray],
        *,
        frame_shape: tuple[int, int],
        contours_per_frame: Sequence[Sequence[np.ndarray]] | None = None,
        scale: float = 1.0,
        units: str = "px",
        params: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> "SpyDEParticles":
        """Build from a per-frame list of ``(n_i, N_COLUMNS)`` property blocks.

        *contours_per_frame*, when given, must line up exactly with *per_frame* —
        one outline per row. A mismatch raises rather than silently pairing the
        wrong outline with a particle, which would draw plausible nonsense.
        """
        n_frames = len(per_frame)
        blocks: list[np.ndarray] = []
        counts: list[int] = []
        for i, blk in enumerate(per_frame):
            b = np.zeros((0, N_COLUMNS), np.float32) if blk is None or len(blk) == 0 \
                else np.asarray(blk, dtype=np.float32)
            if b.ndim != 2 or b.shape[1] != N_COLUMNS:
                raise ValueError(
                    f"frame {i}: expected (n, {N_COLUMNS}); got {b.shape}"
                )
            blocks.append(b)
            counts.append(len(b))

        flat = (np.concatenate(blocks, axis=0) if blocks
                else np.zeros((0, N_COLUMNS), np.float32))
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        contours = contour_offsets = None
        if contours_per_frame is not None:
            if len(contours_per_frame) != n_frames:
                raise ValueError(
                    f"contours_per_frame has {len(contours_per_frame)} frames, "
                    f"per_frame has {n_frames}"
                )
            polys: list[np.ndarray] = []
            for i, (cs, n) in enumerate(zip(contours_per_frame, counts)):
                cs = list(cs or ())
                if len(cs) != n:
                    raise ValueError(
                        f"frame {i}: {len(cs)} outlines for {n} particles — "
                        "outlines must correspond 1:1 with property rows"
                    )
                polys.extend(np.asarray(c, dtype=np.int16).reshape(-1, 2) for c in cs)
            contours = (np.concatenate(polys, axis=0) if polys
                        else np.zeros((0, 2), np.int16))
            contour_offsets = np.concatenate(
                [[0], np.cumsum([len(p) for p in polys])]).astype(np.int64)

        return cls(
            flat_buffer=flat, t_offsets=offsets, frame_shape=frame_shape,
            contours=contours, contour_offsets=contour_offsets,
            scale=scale, units=units, params=params or {}, provenance=provenance,
        )

    def __repr__(self) -> str:
        return (
            f"SpyDEParticles({self.n_particles} particles over {self.n_frames} "
            f"frames, {self.frame_shape[0]}x{self.frame_shape[1]} px, "
            f"masks={self.has_masks}, tracks={self.has_tracks})"
        )

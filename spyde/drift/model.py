"""
model.py — :class:`DriftModel`, the small serialisable result of a drift solve.

This is the ONLY thing a solve produces. It is deliberately tiny (an ``(N, 2)``
float32 array plus metadata) because the alternative — an aligned copy of the
movie — is tens of GB (see the module docstring in ``spyde/drift/__init__.py``).

Sign convention — read this before touching anything
-----------------------------------------------------
``shifts[i]`` is the **correction**: the ``(dy, dx)`` you ADD to frame *i* to
bring it into the reference frame. So::

    aligned_i = scipy.ndimage.shift(frame_i, model.shifts[i])

This matches ``skimage.registration.phase_cross_correlation``, whose docstring
defines its return as "the shift vector required to register moving_image with
reference_image", and matches ``scipy.ndimage.shift``'s own sign. Keeping all
three identical is why the convention is stated here rather than inferred at each
call site — an inverted sign produces a drift curve that looks entirely plausible
and doubles the drift instead of removing it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Bumped when the on-disk layout changes incompatibly.
FORMAT_VERSION = 1


@dataclass
class DriftModel:
    """The correction for a frame stack.

    Parameters
    ----------
    shifts
        ``(N, 2)`` float32, ``(dy, dx)`` per frame — the correction to ADD (see
        the module docstring on sign convention).
    kind
        ``"rigid"`` today. ``"affine"`` / ``"scan_knot"`` / ``"dense"`` are the
        non-rigid parameterisations planned in Wave A2–A5; they will carry their
        parameters in :attr:`extra` and keep ``shifts`` as the rigid component.
    reference
        How the reference was formed: ``"running"`` (running Fourier average),
        ``"sequential"`` (frame-to-frame, cumulative) or ``"fixed:<i>"``.
    residuals
        Optional ``(N,)`` per-frame correlation peak sharpness — a weak but free
        quality signal. NaN where not computed.
    params
        The solver arguments, for provenance and for re-running.
    """

    shifts: np.ndarray
    kind: str = "rigid"
    reference: str = "running"
    residuals: np.ndarray | None = None
    params: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.shifts = np.ascontiguousarray(self.shifts, dtype=np.float32)
        if self.shifts.ndim != 2 or self.shifts.shape[1] != 2:
            raise ValueError(
                f"shifts must be (N, 2); got {self.shifts.shape}"
            )
        if self.residuals is not None:
            self.residuals = np.ascontiguousarray(self.residuals, dtype=np.float32)
            if self.residuals.shape != (self.n_frames,):
                raise ValueError(
                    f"residuals must be ({self.n_frames},); got {self.residuals.shape}"
                )

    # ── basic properties ─────────────────────────────────────────────────────

    @property
    def n_frames(self) -> int:
        return int(self.shifts.shape[0])

    @property
    def max_abs_shift(self) -> float:
        """Largest single-axis correction, in pixels. Sizes the padded border."""
        if self.n_frames == 0:
            return 0.0
        return float(np.nanmax(np.abs(self.shifts)))

    @property
    def is_integer(self) -> bool:
        """True when every shift is a whole number of pixels.

        An integer-only model can be applied by ``np.roll`` and so **preserves
        the source dtype**; a sub-pixel model needs interpolation and therefore
        float output. Callers use this to avoid promoting a uint16 movie to
        float32 when they don't have to.
        """
        if self.n_frames == 0:
            return True
        finite = self.shifts[np.isfinite(self.shifts)]
        return bool(finite.size == 0 or np.all(finite == np.round(finite)))

    def shift_at(self, index: int) -> np.ndarray:
        """The ``(dy, dx)`` correction for frame *index*."""
        return self.shifts[int(index)]

    # ── frame-of-reference conversions ───────────────────────────────────────

    def to_sample_frame(self, positions: np.ndarray, frame_index) -> np.ndarray:
        """Map lab-frame ``(y, x)`` positions onto the corrected (sample) frame.

        *positions* is ``(M, 2)``; *frame_index* is a scalar or ``(M,)``. This is
        how a trajectory measured on RAW frames is reported as if the stage had
        never moved — the alternative to physically warping the movie first.
        """
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"positions must be (M, 2); got {pos.shape}")
        idx = np.asarray(frame_index, dtype=np.intp)
        return pos + self.shifts[idx].reshape(pos.shape)

    def to_lab_frame(self, positions: np.ndarray, frame_index) -> np.ndarray:
        """Inverse of :meth:`to_sample_frame`."""
        pos = np.asarray(positions, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"positions must be (M, 2); got {pos.shape}")
        idx = np.asarray(frame_index, dtype=np.intp)
        return pos - self.shifts[idx].reshape(pos.shape)

    # ── serialisation ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write to a compressed ``.npz``. Small enough to sit beside the data."""
        meta = {
            "format_version": FORMAT_VERSION,
            "kind": self.kind,
            "reference": self.reference,
            "params": self.params,
            "provenance": self.provenance,
            "extra": self.extra,
        }
        arrays = {"shifts": self.shifts, "meta": np.array(json.dumps(meta))}
        if self.residuals is not None:
            arrays["residuals"] = self.residuals
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "DriftModel":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"].item()))
            ver = meta.get("format_version")
            if ver != FORMAT_VERSION:
                raise ValueError(
                    f"unsupported DriftModel format version {ver!r} "
                    f"(this build reads {FORMAT_VERSION})"
                )
            return cls(
                shifts=z["shifts"],
                kind=meta.get("kind", "rigid"),
                reference=meta.get("reference", "running"),
                residuals=z["residuals"] if "residuals" in z.files else None,
                params=meta.get("params") or {},
                provenance=meta.get("provenance"),
                extra=meta.get("extra") or {},
            )

    def __repr__(self) -> str:
        return (
            f"DriftModel(kind={self.kind!r}, n_frames={self.n_frames}, "
            f"reference={self.reference!r}, max_abs_shift={self.max_abs_shift:.2f} px)"
        )

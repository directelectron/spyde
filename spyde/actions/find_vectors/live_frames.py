"""
live_frames.py — render a Find-Vectors position BEFORE the batch finalizes.

The batch produces one padded peaks block per nav chunk,
``(nav…, MAX_PEAKS, 3)`` with columns ``(ky_px, kx_px, intensity)``, and the
client already receives every one of them through the per-chunk callback that
drives the live count map. Only at the very end are they unpacked into the CSR
:class:`~spyde.signals.diffraction_vectors.SpyDEDiffractionVectors` whose
``render_frame`` paints the result window's diffraction pattern — so for the
whole run the signal plot has nothing to show even for positions whose vectors
are sitting in the client process.

:class:`LiveVectorFrames` keeps those blocks and renders any completed position
as the same flat disks ``render_frame`` will draw later, so the progressive
window's signal plot can be live (see ``spyde/actions/live_signal.py``).

Memory: each block is COMPACTED on arrival (``_compact_padded_chunk`` trims the
MAX_PEAKS padding to the chunk's real maximum — typically a few tens of slots of
512), and the whole store is capped, so it stays a small fraction of the padded
result the client already holds. It never touches the source dataset (CLAUDE.md
memory-safety rule) — it only ever holds peak coordinates.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from spyde.signals.diffraction_vectors import (
    COL_INTENSITY, COL_KX, COL_KY, COL_NAV_X, COL_NAV_Y, N_COLS,
    _render_disks_block,
)

log = logging.getLogger(__name__)

#: Cap on the retained (compacted) peak blocks. Well past a realistic scan's
#: peak count — a 13k-position SpEd scan with ~20 peaks/pattern compacts to a
#: few MB — but it bounds a pathological low-threshold run that finds hundreds
#: of peaks everywhere. Over budget, later blocks are simply not retained: they
#: still count as computed for the navigator, they just cannot be previewed.
MAX_STORE_BYTES = 192 * 1024 * 1024


def _compact(block: np.ndarray) -> np.ndarray:
    """Trim the MAX_PEAKS NaN padding to this block's real maximum peak count."""
    valid = np.isfinite(block[..., 0])
    n_max = int(valid.sum(axis=-1).max()) if valid.size else 0
    return np.ascontiguousarray(block[..., :max(1, n_max), :])


class LiveVectorFrames:
    """Client-side store of completed Find-Vectors peak blocks + a renderer.

    ``add(nav_slices, block)`` is called from the compute's per-chunk callback
    (a Dask done-callback thread); ``render(index)`` is called from the
    ``_NavDispatcher`` thread and from the preview's sampler. Both are cheap and
    thread-safe; ``render`` returns ``None`` for a position no retained block
    covers, which the preview treats as "not previewable, keep the last frame".
    """

    def __init__(self, *, sig_hw, kernel_radius_px: float):
        self.sig_hw = (int(sig_hw[0]), int(sig_hw[1]))
        self.kernel_radius_px = float(kernel_radius_px)
        self._blocks: list[tuple[tuple[slice, ...], np.ndarray]] = []
        self._bytes = 0
        self._lock = threading.Lock()

    # ── ingest ───────────────────────────────────────────────────────────────

    def add(self, nav_slices, block) -> bool:
        """Retain *block* for its global nav slice. Never raises."""
        try:
            arr = _compact(np.asarray(block))
            sl = tuple(nav_slices)
            with self._lock:
                if self._bytes + arr.nbytes > MAX_STORE_BYTES:
                    return False
                self._blocks.append((sl, arr))
                self._bytes += arr.nbytes
            return True
        except Exception as e:
            log.debug("retaining live vector block %r failed: %s", nav_slices, e)
            return False

    # ── render ───────────────────────────────────────────────────────────────

    def _locate(self, index: tuple[int, ...]):
        """The (block, local index) covering *index*, newest block first."""
        with self._lock:
            blocks = list(self._blocks)
        for sl, arr in reversed(blocks):
            if len(sl) != len(index):
                continue
            local = []
            for v, s in zip(index, sl):
                start = int(s.start or 0)
                stop = int(s.stop) if s.stop is not None else start + 1
                if not (start <= v < stop):
                    break
                local.append(v - start)
            else:
                return arr, tuple(local)
        return None, None

    def render(self, index) -> np.ndarray | None:
        """The (H, W) frame of flat disks at nav *index*, or ``None``.

        Identical rasterisation to
        :meth:`SpyDEDiffractionVectors.render_frame` — the same
        ``_render_disks_block`` with the same kernel radius — so the preview
        frame and the finalized frame agree pixel for pixel. The peaks are still
        in DETECTOR PIXELS here (calibration is applied only when the flat
        buffer is built), so the raster runs with unit scale / zero offset
        instead of round-tripping through the calibrated units.
        """
        try:
            idx = tuple(int(v) for v in index)
            arr, local = self._locate(idx)
            if arr is None:
                return None
            peaks = np.asarray(arr[local])              # (n_slots, 3)
            good = np.isfinite(peaks[:, 0])
            peaks = peaks[good]
            rows = np.zeros((peaks.shape[0], N_COLS), dtype=np.float32)
            if peaks.shape[0]:
                rows[:, COL_NAV_Y] = idx[-2]
                rows[:, COL_NAV_X] = idx[-1]
                rows[:, COL_KY] = peaks[:, 0]           # detector row (px)
                rows[:, COL_KX] = peaks[:, 1]           # detector column (px)
                rows[:, COL_INTENSITY] = peaks[:, 2]
            H, W = self.sig_hw
            return _render_disks_block(
                rows, (1, 1), (H, W), 1.0, 0.0, 1.0, 0.0,
                self.kernel_radius_px, (idx[-2], idx[-1]),
            )[0, 0]
        except Exception as e:
            log.debug("live vector render at %r failed: %s", index, e)
            return None

    def clear(self) -> None:
        with self._lock:
            self._blocks = []
            self._bytes = 0

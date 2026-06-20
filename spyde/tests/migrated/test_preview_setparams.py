"""Regression: tuning Find-Vectors params (set_params) must re-render the live
preview peaks IMMEDIATELY even when the navigator hasn't moved — a slider tweak
is synchronous (instant feedback), while the navigator drag stays async.
"""
import time

import numpy as np

from spyde.actions.vector_overlay import FindVectorsPreviewOverlay
from spyde.drawing.live_overlay import LiveOverlayEngine


class _FakeMarkerGroup:
    def __init__(self):
        self.offsets = []

    def set(self, **kw):
        if "offsets" in kw:
            self.offsets.append(np.asarray(kw["offsets"]))

    def remove(self):
        pass


class _Sig:
    """One nav pixel with disks close together so the found-peak count is
    sensitive to min_distance (the disk NXCORR ≈1, so threshold doesn't
    discriminate clean disks; min_distance does)."""
    def __init__(self):
        frame = np.zeros((1, 1, 64, 64), np.float32)
        yy, xx = np.mgrid[0:64, 0:64]
        for cy, cx in [(28, 28), (28, 36), (36, 28), (36, 36), (20, 32), (44, 32)]:
            frame[0, 0] += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.5 ** 2)))
        self.data = frame.astype(np.float32)


def _make_overlay():
    ov = FindVectorsPreviewOverlay(
        dp_plot=None, signal=_Sig(), sigma=0.0, kernel_radius=3,
        threshold=0.5, min_distance=3, subpixel=False,
    )
    ov._mg = _FakeMarkerGroup()
    ov._last_iyix = (0, 0)
    # The overlay normally runs the navigation path through the engine (thread
    # mode); set_params must still re-render synchronously regardless.
    ov._engine = LiveOverlayEngine(ov._offsets_for, ov._render_payload,
                                   mode="thread", name="fv")
    return ov


def test_set_params_rerenders_peaks_synchronously():
    ov = _make_overlay()
    try:
        n_before = len(ov._mg.offsets)
        ov.set_params(min_distance=2)             # close peaks resolved → many
        assert len(ov._mg.offsets) == n_before + 1, \
            "set_params did not re-render immediately (synchronously)"
        n_close = len(ov._mg.offsets[-1])

        ov.set_params(min_distance=30)            # merge → few
        assert len(ov._mg.offsets) == n_before + 2
        n_far = len(ov._mg.offsets[-1])

        assert n_close > 0
        assert n_far < n_close, \
            f"min_distance change didn't update the peaks ({n_close} → {n_far})"
    finally:
        ov._engine.stop()


def test_navigation_path_renders_via_engine():
    """The navigator path (_on_indices) goes through the async engine and still
    renders the preview peaks."""
    ov = _make_overlay()
    try:
        n_before = len(ov._mg.offsets)
        ov._on_indices(np.array([[0, 0]]))        # crosshair (cx, cy)
        t0 = time.time()
        while time.time() - t0 < 2.0 and len(ov._mg.offsets) <= n_before:
            time.sleep(0.01)
        assert len(ov._mg.offsets) > n_before, "navigation did not render peaks"
    finally:
        ov._engine.stop()

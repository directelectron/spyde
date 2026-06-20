"""
Subpixel accuracy of Find Vectors on synthetic disks at KNOWN subpixel centres.

The peak should land on the disk centre (sub-pixel), and the stored intensity
should track the raw disk intensity — not the correlation score.
"""
import numpy as np

from spyde.actions.find_vectors import _find_vectors_single_frame


def _soft_disk(frame, cy, cx, r, amp):
    """Add a smooth-edged disk centred at (cy, cx) (sub-pixel) of radius r."""
    H, W = frame.shape
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    frame += amp / (1.0 + np.exp((d - r) * 2.0))      # logistic edge ≈ filled disk


def _make_frame(size=112, r=5):
    """A few disks at known sub-pixel centres with DIFFERENT amplitudes."""
    rng = np.random.RandomState(0)
    f = (rng.rand(size, size) * 0.02).astype(np.float32)        # faint noise
    truth = [                                                   # (cy, cx, amp)
        (28.3, 40.7, 1.0),
        (60.8, 22.2, 0.55),
        (44.5, 78.9, 0.80),
        (82.1, 64.4, 0.40),
    ]
    for cy, cx, amp in truth:
        _soft_disk(f, cy, cx, r, amp)
    return f, r, truth


def _match(peaks, truth):
    """Nearest found peak (ky, kx) for each truth centre → (err_px, found_row)."""
    out = []
    for cy, cx, amp in truth:
        d = np.hypot(peaks[:, 0] - cy, peaks[:, 1] - cx)
        j = int(np.argmin(d))
        out.append((float(d[j]), peaks[j], amp))
    return out


class TestSubpixelAccuracy:
    def test_subpixel_is_applied_and_subpixel(self):
        """Position stays on the NXCORR surface (parabolic vertex); the
        refinement must actually move OFF the integer pixel and land within a
        pixel of the true disk."""
        f, r, truth = _make_frame()
        _c, _rc, peaks = _find_vectors_single_frame(
            f, kernel_radius=r, threshold=0.35, min_distance=8, subpixel=True)
        assert len(peaks) >= len(truth)
        # at least one peak carries a genuine fractional (sub-pixel) coordinate
        frac = np.abs(peaks[:, :2] - np.round(peaks[:, :2]))
        assert frac.max() > 0.05, "subpixel refinement was not applied (all integer)"
        errs = [e for e, _p, _a in _match(peaks, truth)]
        assert max(errs) < 1.25, f"NXCORR subpixel error too large: {errs}"

    def test_subpixel_beats_integer(self):
        f, r, truth = _make_frame()
        _c, _rc, p_sub = _find_vectors_single_frame(
            f, kernel_radius=r, threshold=0.35, min_distance=8, subpixel=True)
        _c, _rc, p_int = _find_vectors_single_frame(
            f, kernel_radius=r, threshold=0.35, min_distance=8, subpixel=False)
        e_sub = max(e for e, _p, _a in _match(p_sub, truth))
        e_int = max(e for e, _p, _a in _match(p_int, truth))
        assert e_sub < e_int, f"subpixel ({e_sub:.2f}) not better than integer ({e_int:.2f})"

    def test_intensity_tracks_raw_disk_amplitude(self):
        f, r, truth = _make_frame()
        _c, _rc, peaks = _find_vectors_single_frame(
            f, kernel_radius=r, threshold=0.35, min_distance=8, subpixel=True)
        matched = _match(peaks, truth)
        # stored intensity should be MONOTONIC in the true disk amplitude — the
        # bright disk's intensity > the faint disk's. (Correlation score is ~1
        # for all well-matched disks, so it would NOT be monotonic.)
        amps = [a for _e, _p, a in matched]
        ints = [float(p[2]) for _e, p, _a in matched]
        order_true = np.argsort(amps)
        order_found = np.argsort(ints)
        assert list(order_true) == list(order_found), (
            f"intensity not monotonic in disk amplitude: amps={amps} ints={ints}")

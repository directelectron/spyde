"""
Subpixel accuracy of Find Vectors on synthetic disks at KNOWN subpixel centres.

The peak should land on the disk centre (sub-pixel), and the stored intensity
should track the raw disk intensity — not the correlation score.
"""
import numpy as np
import pytest

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


def _gauss(cy, cx, size=64, s=2.2):
    """A SHARP, unique-peak Gaussian blob (unlike a flat-topped disk, whose
    correlation surface plateaus and makes argmax tie-break ambiguously)."""
    f = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    f += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s * s))).astype(np.float32)
    return f


class TestSubpixelUnbiased:
    """On a clean (non-plateau) peak the sub-pixel estimate must be UNBIASED to
    a fraction of a pixel — tight enough to catch a half-pixel coordinate
    convention slip, which the 1.25px logistic-disk tolerance above would hide.
    (A real investigation: the apparent 'disks slightly off' turned out to be a
    half-pixel imshow EXTENT in a diagnostic plot, not the compute — anyplotlib
    renders image pixel i and a data marker at i both at the pixel centre.)"""

    def test_subpixel_unbiased_on_gaussian(self):
        ey, ex = [], []
        for fy, fx in [(0.0, 0.0), (0.3, -0.3), (0.5, 0.5), (-0.4, 0.2), (0.7, -0.6)]:
            cy, cx = 32 + fy, 32 + fx
            _c, _rc, p = _find_vectors_single_frame(
                _gauss(cy, cx), kernel_radius=5, threshold=0.3,
                min_distance=8, subpixel=True)
            j = int(np.argmax(p[:, 2]))
            ey.append(float(p[j, 0] - cy))
            ex.append(float(p[j, 1] - cx))
        ey, ex = np.array(ey), np.array(ex)
        assert np.abs(ey).max() < 0.15 and np.abs(ex).max() < 0.15, (ey, ex)
        # no SYSTEMATIC bias — a half-pixel offset would show as a ~0.5 mean
        assert abs(ey.mean()) < 0.05 and abs(ex.mean()) < 0.05, (ey.mean(), ex.mean())

    def test_integer_peak_matches_skimage(self):
        """The NXCORR peak pixel must equal skimage.match_template (the reference
        the implementation claims equivalence to) — i.e. correctly centred."""
        match_template = pytest.importorskip("skimage.feature").match_template
        from spyde.actions.find_vectors import _make_disk
        templ = _make_disk(5)
        for cy, cx in [(30, 30), (25, 40), (33, 27)]:
            f = _gauss(cy, cx)
            sk = match_template(f, templ, pad_input=True)
            sy, sx = np.unravel_index(int(np.argmax(sk)), sk.shape)
            _c, rc, _p = _find_vectors_single_frame(
                f, kernel_radius=5, threshold=0.3, min_distance=8, subpixel=False)
            iy, ix = np.unravel_index(int(np.argmax(rc)), rc.shape)
            assert (iy, ix) == (sy, sx), f"code {(iy, ix)} != skimage {(sy, sx)}"

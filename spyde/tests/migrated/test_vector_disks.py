"""
Vector → disk-frame rendering and vector virtual imaging.

Covers the contract behind the live signal display:
  - render_frame draws each vector as a disk valued by its NXCORR intensity
  - the frame changes when the nav position (crosshair) changes
  - virtual_image_from_roi builds an intensity-weighted nav image from the
    vectors using the kernel radius
"""
import numpy as np
import pytest
import dask.array as da
import hyperspy.api as hs

from spyde.actions.find_vectors import _do_compute_vectors
from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY


def _params():
    return {
        "sigma": 0.5,
        "kernel_radius": 3,
        "threshold": 0.3,
        "min_distance": 3,
        "subpixel": False,
    }


def _make_4d_two_spots():
    """4D STEM where the bright spot sits at a different detector position
    depending on the nav column — so different nav positions yield different
    vector coordinates (and thus different rendered frames)."""
    ny, nx, ky, kx = 4, 4, 32, 32
    data = np.zeros((ny, nx, ky, kx), dtype=np.float32)
    for ix in range(nx):
        # spot drifts across the detector with the nav column
        cy = 8 + ix
        cx = 8 + ix
        data[:, ix, cy - 2:cy + 2, cx - 2:cx + 2] = 100.0
    da_data = da.from_array(data, chunks=(2, 2, ky, kx))
    s = hs.signals.Signal2D(da_data)
    for ax in s.axes_manager.signal_axes:
        ax.scale = 0.01
        ax.offset = -ky * 0.005
    return s


@pytest.fixture(scope="module")
def vecs():
    v = _do_compute_vectors(_make_4d_two_spots(), _params(), None, None)
    if len(v.flat_buffer) == 0:
        pytest.skip("No vectors found in synthetic data")
    return v


class TestDiskRendering:
    def test_frame_disk_value_is_intensity(self, vecs):
        frame = vecs.render_frame(0, 0)
        rows = vecs.at(0, 0)
        assert len(rows) > 0
        peak_inten = float(rows[:, COL_INTENSITY].max())
        # the brightest disk pixel equals the brightest vector's intensity
        assert np.isclose(frame.max(), peak_inten, atol=1e-4)
        # disks are non-trivial: more than one pixel lit (radius >= 1)
        assert (frame > 0).sum() > 1

    def test_frame_changes_with_nav_position(self, vecs):
        # spot drifts with nav column → different columns give different frames
        f0 = vecs.render_frame(0, 0)
        f3 = vecs.render_frame(0, 3)
        assert not np.array_equal(f0, f3), (
            "Rendered disk frame did not change between nav columns — "
            "the live display would show stale data on crosshair move."
        )

    def test_disk_centered_on_vector(self, vecs):
        frame = vecs.render_frame(0, 0)
        rows = vecs.at(0, 0)
        ax = vecs.sig_axes
        # brightest vector
        bi = int(np.argmax(rows[:, COL_INTENSITY]))
        cx = int(round((rows[bi, COL_KX] - ax[0].offset) / ax[0].scale))
        cy = int(round((rows[bi, COL_KY] - ax[1].offset) / ax[1].scale))
        # the disk centre pixel should be lit at (row=cy, col=cx)
        assert frame[cy, cx] > 0


class TestRenderRegion:
    """render_region: max-within-frame, sum-across-region — so a 1x1 region
    equals render_frame and a 2-position region equals the sum of two frames."""

    def test_1x1_region_equals_render_frame(self, vecs):
        for (iy, ix) in [(0, 0), (0, 3), (2, 1)]:
            frame = vecs.render_frame(iy, ix)
            reg = vecs.render_region(iy, iy + 1, ix, ix + 1)
            assert reg.shape == frame.shape
            np.testing.assert_array_equal(reg, frame)

    def test_two_position_region_equals_sum_of_frames(self, vecs):
        # A 1-row x 2-col region → sum of the two per-position frames.
        f0 = vecs.render_frame(0, 0)
        f1 = vecs.render_frame(0, 1)
        reg = vecs.render_region(0, 1, 0, 2)
        np.testing.assert_allclose(reg, f0 + f1, atol=1e-4)

    def test_full_region_sums_all_frames(self, vecs):
        ny, nx = vecs.nav_shape
        ref = np.zeros_like(vecs.render_frame(0, 0))
        for iy in range(ny):
            for ix in range(nx):
                ref += vecs.render_frame(iy, ix)
        reg = vecs.render_region(0, ny, 0, nx)
        np.testing.assert_allclose(reg, ref, atol=1e-4)

    def test_bounds_clamped_and_ordered(self, vecs):
        ny, nx = vecs.nav_shape
        # Backwards + out-of-range bounds are sorted and clamped, never crash.
        a = vecs.render_region(ny + 5, -3, nx + 2, -1)
        assert a.shape == vecs.render_frame(0, 0).shape
        # Degenerate (equal) bounds collapse to a single position == render_frame.
        b = vecs.render_region(1, 1, 2, 2)
        np.testing.assert_array_equal(b, vecs.render_frame(1, 2))

    def test_no_full_dataset_compute(self, vecs, monkeypatch):
        # Memory-safety: render_region must only touch the CSR buffer, never
        # trigger a dask compute on any lazy array.
        import dask.array as da
        called = {"n": 0}
        orig = da.Array.compute

        def _spy(self, *a, **k):
            called["n"] += 1
            return orig(self, *a, **k)

        monkeypatch.setattr(da.Array, "compute", _spy)
        vecs.render_region(0, 2, 0, 2)
        assert called["n"] == 0


class TestVectorVirtualImaging:
    def test_intensity_weighted_matches_manual_sum(self, vecs):
        # ROI over the whole detector → every vector counts.
        # Intensity-weighted VI must equal the per-position sum of intensities.
        cx = cy = 0.0
        r_out = 10.0
        img_count = vecs.virtual_image_from_roi(
            cx, cy, r_out, intensity_weighted=False
        )
        img_inten = vecs.virtual_image_from_roi(
            cx, cy, r_out, intensity_weighted=True
        )
        assert img_count.shape == vecs.nav_shape
        assert img_inten.shape == vecs.nav_shape
        assert img_count.sum() > 0 and img_inten.sum() > 0

        # Manual reference: sum intensity per (iy, ix) directly from the buffer
        ny, nx = vecs.nav_shape
        ref = np.zeros((ny, nx), dtype=np.float32)
        for iy in range(ny):
            for ix in range(nx):
                ref[iy, ix] = vecs.at(iy, ix)[:, COL_INTENSITY].sum()
        np.testing.assert_allclose(img_inten, ref, atol=1e-4)
        # count map equals per-position vector counts
        np.testing.assert_allclose(img_count, vecs.count_map(), atol=1e-4)

    def test_roi_radius_excludes_far_vectors(self, vecs):
        # a tiny ROI far from any spot picks up nothing
        empty = vecs.virtual_image_from_roi(
            cx=5.0, cy=5.0, r_outer=0.001, intensity_weighted=True
        )
        assert empty.sum() == 0

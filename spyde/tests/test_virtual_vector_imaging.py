"""
Tests for SpyDEDiffractionVectors virtual imaging methods.

Covers:
  - virtual_image_from_roi: correctness, intensity weighting, annular mask,
    edge cases (empty buffer, no hits, single position)
  - build_kdtree / virtual_image_from_kdtree: tree builds correctly, results
    match direct numpy path, fallback when tree not built
  - Performance: both paths must complete a full-scan image in <100 ms
  - Consistency: kdtree and direct paths agree to float32 tolerance
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_vecs(nav_shape=(8, 8), n_per_pos=5, kxy_scale=0.1, seed=42):
    """
    Uniform random vectors in [-1, 1] × [-1, 1] with known positions.
    nav_x = col index, nav_y = row index stored in cols 0, 1.
    """
    rng = np.random.default_rng(seed)
    nav_y, nav_x = nav_shape
    n_nav = nav_y * nav_x
    counts = np.full(n_nav, n_per_pos, dtype=np.int64)
    offsets = np.zeros(n_nav + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    N = int(offsets[-1])

    flat = np.zeros((N, 6), dtype=np.float32)
    for flat_idx in range(n_nav):
        iy, ix = divmod(flat_idx, nav_x)
        s, e = offsets[flat_idx], offsets[flat_idx + 1]
        flat[s:e, 0] = ix                                   # nav_x
        flat[s:e, 1] = iy                                   # nav_y
        flat[s:e, 2] = rng.uniform(-1, 1, n_per_pos)        # kx
        flat[s:e, 3] = rng.uniform(-1, 1, n_per_pos)        # ky
        flat[s:e, 4] = -1.0                                 # time: 4D
        flat[s:e, 5] = rng.uniform(0.1, 1.0, n_per_pos)    # intensity

    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat,
        full_nav_shape=nav_shape,
        sig_shape=(64, 64),
        sig_axes=None,
        kernel_radius_px=4.0,
        kernel_radius_data=0.04,
    )


def _make_vecs_at_known_positions(nav_shape=(4, 4)):
    """
    One vector per nav position, placed at a known kx/ky so ROI tests are exact.
    Vector at (iy, ix) is placed at kx = ix * 0.1, ky = iy * 0.1.
    Intensity = 1.0 everywhere.
    """
    nav_y, nav_x = nav_shape
    n_nav = nav_y * nav_x
    flat = np.zeros((n_nav, 6), dtype=np.float32)
    for flat_idx in range(n_nav):
        iy, ix = divmod(flat_idx, nav_x)
        flat[flat_idx, 0] = ix
        flat[flat_idx, 1] = iy
        flat[flat_idx, 2] = ix * 0.1   # kx
        flat[flat_idx, 3] = iy * 0.1   # ky
        flat[flat_idx, 4] = -1.0       # time: 4D
        flat[flat_idx, 5] = 1.0        # intensity
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat,
        full_nav_shape=nav_shape,
        sig_shape=(64, 64),
        sig_axes=None,
        kernel_radius_px=4.0,
        kernel_radius_data=0.04,
    )


def _make_empty_vecs(nav_shape=(4, 4)):
    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=np.zeros((0, 6), dtype=np.float32),
        full_nav_shape=nav_shape,
        sig_shape=(64, 64),
        sig_axes=None,
        kernel_radius_px=4.0,
        kernel_radius_data=0.04,
    )


# ── virtual_image_from_roi ────────────────────────────────────────────────────


def test_roi_output_shape():
    vecs = _make_vecs(nav_shape=(5, 7))
    img = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=2.0)
    assert img.shape == (5, 7)
    assert img.dtype == np.float32


def test_roi_empty_buffer_returns_zeros():
    vecs = _make_empty_vecs((3, 4))
    img = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=1.0)
    assert img.shape == (3, 4)
    assert (img == 0).all()


def test_roi_large_radius_covers_all_vectors():
    """ROI large enough to contain all vectors — every nav position with vectors
    should be non-zero."""
    vecs = _make_vecs(nav_shape=(4, 4), n_per_pos=3)
    img = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=10.0)
    assert (img > 0).all(), "All nav positions should have vectors in a large ROI"


def test_roi_zero_radius_no_hits():
    vecs = _make_vecs()
    img = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.0)
    assert (img == 0).all()


def test_roi_count_mode():
    """intensity_weighted=False should count vectors, not sum intensities."""
    vecs = _make_vecs(nav_shape=(3, 3), n_per_pos=4)
    img_count = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=10.0, intensity_weighted=False)
    img_intensity = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=10.0, intensity_weighted=True)
    # Count should be integer-valued
    assert np.allclose(img_count, img_count.astype(int))
    # Intensity sum != count sum (intensities are in (0.1, 1.0))
    assert not np.allclose(img_count, img_intensity)


def test_roi_annulus_excludes_inner():
    """Annular ROI with r_inner > 0 must exclude vectors inside r_inner."""
    vecs = _make_vecs(nav_shape=(4, 4), n_per_pos=20, seed=0)
    r_inner = 0.3
    r_outer = 0.8
    img_disk = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=r_outer, r_inner=0.0)
    img_annulus = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=r_outer, r_inner=r_inner)
    # Annulus must have <= disk counts everywhere
    assert (img_annulus <= img_disk + 1e-5).all()
    # And strictly less somewhere (statistically certain with 20 vectors per pos)
    assert img_annulus.sum() < img_disk.sum()


def test_roi_known_positions_exact():
    """
    With one vector per nav position at kx=ix*0.1, ky=iy*0.1:
    an ROI centred at (0.15, 0.15) with r=0.06 should hit exactly (iy=1,ix=1)
    which is at kx=0.1, ky=0.1, distance = sqrt(0.05^2 + 0.05^2) ≈ 0.071.
    That's > 0.06, so it misses. r=0.08 should hit it.
    """
    vecs = _make_vecs_at_known_positions((4, 4))
    # (iy=1, ix=1) has kx=0.1, ky=0.1; dist from (0.15,0.15) = sqrt(2)*0.05 ≈ 0.0707
    img_miss = vecs.virtual_image_from_roi(0.15, 0.15, r_outer=0.06)
    assert img_miss[1, 1] == 0.0, "Should miss (1,1) with r=0.06"

    img_hit = vecs.virtual_image_from_roi(0.15, 0.15, r_outer=0.08)
    assert img_hit[1, 1] > 0.0, "Should hit (1,1) with r=0.08"


def test_roi_sum_matches_manual_count():
    """Manual per-position count must match virtual_image_from_roi count mode."""
    nav_shape = (4, 5)
    vecs = _make_vecs(nav_shape=nav_shape, n_per_pos=10, seed=7)
    cx, cy, r = 0.0, 0.0, 0.5

    img = vecs.virtual_image_from_roi(cx, cy, r_outer=r, intensity_weighted=False)

    nav_y, nav_x = nav_shape
    expected = np.zeros(nav_shape, dtype=np.float32)
    for iy in range(nav_y):
        for ix in range(nav_x):
            rows = vecs.at(iy, ix)
            kx = rows[:, 2]; ky = rows[:, 3]
            dist2 = (kx - cx) ** 2 + (ky - cy) ** 2
            expected[iy, ix] = float((dist2 <= r * r).sum())

    np.testing.assert_array_equal(img, expected)


def test_roi_intensity_sum_matches_manual():
    """Manual per-position intensity sum must match virtual_image_from_roi."""
    nav_shape = (3, 4)
    vecs = _make_vecs(nav_shape=nav_shape, n_per_pos=8, seed=11)
    cx, cy, r = 0.2, -0.1, 0.4

    img = vecs.virtual_image_from_roi(cx, cy, r_outer=r, intensity_weighted=True)

    nav_y, nav_x = nav_shape
    expected = np.zeros(nav_shape, dtype=np.float32)
    for iy in range(nav_y):
        for ix in range(nav_x):
            rows = vecs.at(iy, ix)
            kx = rows[:, 2]; ky = rows[:, 3]
            dist2 = (kx - cx) ** 2 + (ky - cy) ** 2
            mask = dist2 <= r * r
            expected[iy, ix] = float(rows[mask, 5].sum())  # COL_INTENSITY

    np.testing.assert_allclose(img, expected, atol=1e-5)


def test_roi_non_square_nav():
    vecs = _make_vecs(nav_shape=(3, 7))
    img = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=5.0)
    assert img.shape == (3, 7)


def test_roi_single_nav_position():
    """1×1 nav grid — degenerate case."""
    offsets = np.array([0, 3], dtype=np.int64)
    flat = np.array([
        [0, 0, 0.1, 0.2, -1, 0.5],
        [0, 0, -0.1, 0.1, -1, 0.8],
        [0, 0, 0.5, 0.5, -1, 0.3],
    ], dtype=np.float32)
    vecs = SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat, full_nav_shape=(1, 1),
        sig_shape=(64, 64), sig_axes=None,
        kernel_radius_px=4.0, kernel_radius_data=0.04,
    )
    img = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.5)
    assert img.shape == (1, 1)
    assert img[0, 0] > 0


# ── build_kdtree / virtual_image_from_kdtree ──────────────────────────────────


def test_build_kdtree_sets_internal_tree():
    vecs = _make_vecs()
    assert vecs._kdtree is None
    vecs.build_kdtree()
    assert vecs._kdtree is not None


def test_build_kdtree_empty_buffer():
    vecs = _make_empty_vecs()
    vecs.build_kdtree()  # should not raise
    assert vecs._kdtree is None


def test_kdtree_matches_direct_large_roi():
    """Large ROI: kdtree and direct paths must agree exactly."""
    vecs = _make_vecs(nav_shape=(6, 6), n_per_pos=10, seed=3)
    vecs.build_kdtree()

    direct = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=5.0, intensity_weighted=True)
    tree = vecs.virtual_image_from_kdtree(0.0, 0.0, r_outer=5.0, intensity_weighted=True)
    np.testing.assert_allclose(direct, tree, atol=1e-5,
                               err_msg="KDTree and direct paths disagree (large ROI)")


def test_kdtree_matches_direct_small_roi():
    """Small ROI hitting a subset of positions."""
    vecs = _make_vecs(nav_shape=(8, 8), n_per_pos=15, seed=99)
    vecs.build_kdtree()

    for cx, cy, r in [(0.1, 0.1, 0.2), (-0.3, 0.4, 0.15), (0.0, 0.0, 0.05)]:
        direct = vecs.virtual_image_from_roi(cx, cy, r_outer=r, intensity_weighted=True)
        tree = vecs.virtual_image_from_kdtree(cx, cy, r_outer=r, intensity_weighted=True)
        np.testing.assert_allclose(direct, tree, atol=1e-5,
                                   err_msg=f"Mismatch at cx={cx},cy={cy},r={r}")


def test_kdtree_matches_direct_annulus():
    vecs = _make_vecs(nav_shape=(6, 6), n_per_pos=12, seed=55)
    vecs.build_kdtree()
    direct = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.7, r_inner=0.3)
    tree = vecs.virtual_image_from_kdtree(0.0, 0.0, r_outer=0.7, r_inner=0.3)
    np.testing.assert_allclose(direct, tree, atol=1e-5)


def test_kdtree_matches_direct_count_mode():
    vecs = _make_vecs(nav_shape=(5, 5), n_per_pos=8, seed=22)
    vecs.build_kdtree()
    direct = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.6, intensity_weighted=False)
    tree = vecs.virtual_image_from_kdtree(0.0, 0.0, r_outer=0.6, intensity_weighted=False)
    np.testing.assert_allclose(direct, tree, atol=1e-5)


def test_kdtree_fallback_when_not_built():
    """virtual_image_from_kdtree falls back to direct when tree is None."""
    vecs = _make_vecs(nav_shape=(4, 4), n_per_pos=5)
    assert vecs._kdtree is None
    img = vecs.virtual_image_from_kdtree(0.0, 0.0, r_outer=1.0)
    assert img.shape == (4, 4)
    expected = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=1.0)
    np.testing.assert_allclose(img, expected, atol=1e-5)


def test_kdtree_empty_buffer():
    vecs = _make_empty_vecs((4, 4))
    vecs.build_kdtree()
    img = vecs.virtual_image_from_kdtree(0.0, 0.0, r_outer=1.0)
    assert img.shape == (4, 4)
    assert (img == 0).all()


def test_kdtree_no_hits_in_roi():
    """ROI placed far from all vectors — result must be all zeros."""
    vecs = _make_vecs(nav_shape=(4, 4), n_per_pos=5)
    vecs.build_kdtree()
    img = vecs.virtual_image_from_kdtree(100.0, 100.0, r_outer=0.01)
    assert (img == 0).all()


# ── Performance ────────────────────────────────────────────────────────────────


def test_direct_roi_performance():
    """
    virtual_image_from_roi on a 64x64 nav with 10 vectors/pos (40k total vectors)
    must complete in < 100 ms — suitable for live drag updates.
    """
    nav_shape = (64, 64)
    vecs = _make_vecs(nav_shape=nav_shape, n_per_pos=10)

    # Warm up (JIT, import caches)
    vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.5)

    t0 = time.perf_counter()
    for _ in range(10):
        vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.5)
    avg_ms = (time.perf_counter() - t0) / 10 * 1000

    assert avg_ms < 100, f"Direct ROI too slow: {avg_ms:.1f} ms (limit 100 ms)"


def test_kdtree_build_performance():
    """KDTree build on 40k vectors must complete in < 2 s."""
    vecs = _make_vecs(nav_shape=(64, 64), n_per_pos=10)
    t0 = time.perf_counter()
    vecs.build_kdtree()
    elapsed_s = time.perf_counter() - t0
    assert elapsed_s < 2.0, f"KDTree build too slow: {elapsed_s:.2f}s"


def test_kdtree_query_performance():
    """
    virtual_image_from_kdtree after tree is built must be < 100 ms for a
    small ROI (few hits) — the point where the tree advantage shows.
    """
    nav_shape = (64, 64)
    vecs = _make_vecs(nav_shape=nav_shape, n_per_pos=10)
    vecs.build_kdtree()
    vecs.virtual_image_from_kdtree(0.0, 0.0, r_outer=0.1)  # warm up

    t0 = time.perf_counter()
    for _ in range(10):
        vecs.virtual_image_from_kdtree(0.0, 0.0, r_outer=0.1)
    avg_ms = (time.perf_counter() - t0) / 10 * 1000

    assert avg_ms < 100, f"KDTree query too slow: {avg_ms:.1f} ms (limit 100 ms)"


def test_large_dataset_direct_roi_under_budget():
    """
    256x256 nav, 5 vectors/pos = 327k vectors.  Direct ROI must stay < 100 ms.
    This is the realistic upper bound for a large 4D-STEM dataset.
    """
    nav_shape = (256, 256)
    vecs = _make_vecs(nav_shape=nav_shape, n_per_pos=5)
    vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.5)  # warm up

    t0 = time.perf_counter()
    vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.5)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 100, (
        f"Direct ROI on 256x256 nav too slow: {elapsed_ms:.1f} ms (limit 100 ms). "
        f"This will cause lag on live drag."
    )


# ── 5D time-aware VVI tests ───────────────────────────────────────────────────


def _make_5d_vecs(nav_shape=(4, 4), n_t=3, n_per_pos=5, seed=42):
    """
    5D flat buffer sorted (t, iy, ix) — outermost-first for multi-level CSR.
    full_nav_shape = (n_t, nav_y, nav_x).
    """
    rng = np.random.default_rng(seed)
    nav_y, nav_x = nav_shape
    n_nav = nav_y * nav_x
    n_total = n_nav * n_per_pos * n_t

    flat = np.zeros((n_total, 6), dtype=np.float32)
    idx = 0
    # Sort (t, iy, ix) — time outermost
    for t in range(n_t):
        for iy in range(nav_y):
            for ix in range(nav_x):
                for _ in range(n_per_pos):
                    flat[idx, 0] = ix
                    flat[idx, 1] = iy
                    flat[idx, 2] = rng.uniform(-1, 1)
                    flat[idx, 3] = rng.uniform(-1, 1)
                    flat[idx, 4] = float(t)
                    flat[idx, 5] = rng.uniform(0.1, 1)
                    idx += 1

    return SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat,
        full_nav_shape=(n_t, *nav_shape),
        sig_shape=(64, 64),
        sig_axes=None,
        kernel_radius_px=4.0,
        kernel_radius_data=0.04,
    )


def test_5d_n_time():
    vecs = _make_5d_vecs(nav_shape=(4, 4), n_t=5)
    assert vecs.n_time == 5


def test_5d_at_returns_all_time_steps():
    """at(iy, ix) should return vectors from ALL time steps."""
    n_t, n_per = 3, 5
    vecs = _make_5d_vecs(nav_shape=(4, 4), n_t=n_t, n_per_pos=n_per)
    rows = vecs.at(0, 0)
    assert len(rows) == n_t * n_per


def test_5d_at_t_filters_correctly():
    """at_t(iy, ix, t) should return only vectors for that time step."""
    n_t, n_per = 4, 3
    vecs = _make_5d_vecs(nav_shape=(3, 3), n_t=n_t, n_per_pos=n_per)
    for t in range(n_t):
        rows = vecs.at_t(0, 0, t)
        assert len(rows) == n_per, f"t={t}: expected {n_per} rows, got {len(rows)}"
        assert (rows[:, 4] == float(t)).all()


def test_5d_virtual_image_t_param_isolates_frame():
    """virtual_image_from_roi with t= should only see that time step's vectors."""
    n_t = 3
    vecs = _make_5d_vecs(nav_shape=(4, 4), n_t=n_t, n_per_pos=8)

    img_all = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=5.0, t=None)
    imgs_t = [vecs.virtual_image_from_roi(0.0, 0.0, r_outer=5.0, t=t)
              for t in range(n_t)]

    # Sum over individual time steps must equal the all-time image
    np.testing.assert_allclose(
        sum(imgs_t), img_all, atol=1e-5,
        err_msg="Sum of per-t images != all-time image"
    )


def test_5d_virtual_image_series_shape():
    n_t, nav_shape = 5, (4, 6)
    vecs = _make_5d_vecs(nav_shape=nav_shape, n_t=n_t, n_per_pos=4)
    series = vecs.virtual_image_series(0.0, 0.0, r_outer=5.0)
    assert series.shape == (n_t, *nav_shape)
    assert series.dtype == np.float32


def test_5d_virtual_image_series_matches_per_t():
    """Each slice of virtual_image_series must equal virtual_image_from_roi(t=t)."""
    n_t, nav_shape = 4, (5, 5)
    vecs = _make_5d_vecs(nav_shape=nav_shape, n_t=n_t, n_per_pos=6)
    vecs.build_kdtree()

    series = vecs.virtual_image_series(0.0, 0.0, r_outer=5.0)
    for t in range(n_t):
        img_t = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=5.0, t=t)
        np.testing.assert_allclose(
            series[t], img_t, atol=1e-5,
            err_msg=f"series[{t}] != virtual_image_from_roi(t={t})"
        )


def test_5d_kdtree_t_param():
    """virtual_image_from_kdtree with t= must match virtual_image_from_roi with t=."""
    n_t = 3
    vecs = _make_5d_vecs(nav_shape=(6, 6), n_t=n_t, n_per_pos=8, seed=77)
    vecs.build_kdtree()

    for t in range(n_t):
        direct = vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.8, t=t)
        tree = vecs.virtual_image_from_kdtree(0.0, 0.0, r_outer=0.8, t=t)
        np.testing.assert_allclose(direct, tree, atol=1e-5, err_msg=f"t={t} mismatch")


def test_5d_virtual_image_series_performance():
    """
    virtual_image_series on 256x256 nav x 500 time x 5 vecs/pos = 163M vectors
    is too large to generate in a test. Use a reduced 64x64x10 proxy and
    verify it stays < 500 ms — the series is one-shot, not a drag update.
    """
    nav_shape = (64, 64)
    n_t = 10
    vecs = _make_5d_vecs(nav_shape=nav_shape, n_t=n_t, n_per_pos=5)
    vecs.virtual_image_series(0.0, 0.0, r_outer=5.0)  # warm up

    t0 = time.perf_counter()
    vecs.virtual_image_series(0.0, 0.0, r_outer=5.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert elapsed_ms < 500, f"virtual_image_series too slow: {elapsed_ms:.1f} ms"


def test_5d_per_frame_roi_performance():
    """
    virtual_image_from_roi(t=t) on 64x64 nav x 500 time steps x 5 vecs/pos.
    Per-frame query must be < 100 ms using nav_offsets[0] for O(1) frame isolation.
    Buffer is sorted (t, iy, ix) — outermost-first.
    """
    nav_shape = (64, 64)
    n_t = 500
    n_per = 5
    rng = np.random.default_rng(0)
    nav_y, nav_x = nav_shape
    n_nav = nav_y * nav_x
    n_total = n_nav * n_per * n_t

    # Build sorted (t, iy, ix) buffer via vectorised broadcast
    t_idx = np.repeat(np.arange(n_t, dtype=np.float32), n_nav * n_per)
    nav_idx = np.tile(np.repeat(np.arange(n_nav, dtype=np.int64), n_per), n_t)
    iy_all = (nav_idx // nav_x).astype(np.float32)
    ix_all = (nav_idx % nav_x).astype(np.float32)

    flat = np.empty((n_total, 6), dtype=np.float32)
    flat[:, 0] = ix_all
    flat[:, 1] = iy_all
    flat[:, 2] = rng.uniform(-1, 1, n_total).astype(np.float32)
    flat[:, 3] = rng.uniform(-1, 1, n_total).astype(np.float32)
    flat[:, 4] = t_idx
    flat[:, 5] = rng.uniform(0.1, 1.0, n_total).astype(np.float32)

    vecs = SpyDEDiffractionVectors.from_arrays(
        flat_buffer=flat,
        full_nav_shape=(n_t, *nav_shape),
        sig_shape=(256, 256), sig_axes=None,
        kernel_radius_px=4.0, kernel_radius_data=0.04,
    )

    vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.5, t=0)  # warm up

    t0 = time.perf_counter()
    for _ in range(5):
        vecs.virtual_image_from_roi(0.0, 0.0, r_outer=0.5, t=250)
    avg_ms = (time.perf_counter() - t0) / 5 * 1000

    assert avg_ms < 100, (
        f"Per-frame VVI on 64x64x500 too slow: {avg_ms:.1f} ms (limit 100 ms). "
        f"Live time-scrubbing will lag."
    )

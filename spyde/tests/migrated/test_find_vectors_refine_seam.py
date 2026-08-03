"""The neighbour refine must not depend on how the scan was chunked.

``persistence`` scores a candidate by the FRACTION of that scan position's
neighbours showing the same peak. The detector runs per dask chunk, so a
position at a chunk edge used to be judged against 3 neighbours instead of 4 —
and the same evidence scores higher out of 3. Every chunk seam therefore grew a
stripe of extra vectors.

Measured on the PdCuSi in-situ series (47x39 scan, nav chunked 26 wide) it was
~+15% in the two boundary columns, and moving the boundary moved the stripe:
splitting at 13 and 26 put one at each, while a single nav chunk was flat.

The fix runs the detector on the ghost-PADDED grid when refining and trims the
result instead of the input, so an edge frame sees its real neighbours. This
pins the invariant that makes it right: chunking is a memory-layout decision and
must not change the answer.
"""
from __future__ import annotations

import numpy as np
import pytest
import hyperspy.api as hs


def _scan(ny=6, nx=12, ks=32, seed=0):
    """A scan with a persistent lattice plus per-frame noise blobs.

    The lattice recurs at every position (so the refine keeps it) while the
    noise does not (so the refine has something to drop) — without both, a
    chunk-edge difference in refine strength would be invisible.
    """
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:ks, 0:ks]

    def disk(cy, cx, amp, r=3):
        return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.float32) * amp

    data = np.zeros((ny, nx, ks, ks), dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            f = disk(ks // 2, ks // 2, 400.0)
            for dy, dx in ((-9, 0), (9, 0), (0, -9), (0, 9)):
                f += disk(ks // 2 + dy, ks // 2 + dx, 260.0)
            for _ in range(3):                       # transient, per-frame
                cy = rng.randint(6, ks - 6)
                cx = rng.randint(6, ks - 6)
                f += disk(cy, cx, 200.0, r=2)
            data[iy, ix] = f
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale, ax.offset, ax.units = 0.1, -1.6, "1/nm"
    return s


def _count_map(nav_chunk_x, spots_seed=0):
    from spyde.actions.find_vectors import _do_compute_vectors
    s = _scan(seed=spots_seed)
    ny, nx = s.data.shape[0], s.data.shape[1]
    ks = s.data.shape[2]
    s.data = s.data.reshape(ny, nx, ks, ks)
    import dask.array as da
    s = s.as_lazy()
    # Signal axes stay WHOLE (Live-Display §1); only the nav-x split moves.
    s.data = da.from_array(np.asarray(s.data), chunks=(ny, nav_chunk_x, ks, ks))
    params = dict(
        method="neural", model_id="", sigma=0.0, kernel_radius=8,
        spot_radius=8.0, min_distance=4, threshold=0.3, subpixel=True,
        bg_sigma=12.0, dog_sigma1=0.8, dog_sigma2=2.0,
        beamstop_auto=False, beamstop_dilate=5,
        persistence=True, show_transform=False,
    )
    vecs = _do_compute_vectors(s, params)
    assert vecs is not None
    return np.asarray(vecs.count_map(), np.int64)


class TestRefineIsChunkInvariant:
    def test_the_count_map_does_not_change_with_the_nav_chunking(self):
        """One chunk vs two vs four — the same scan must give the same map.

        Compared with a tolerance rather than bit-for-bit: the GPU detector
        fills peak slots via atomics and batches by chunk, so re-chunking can
        reorder a tie and shift a count by one even when nothing is wrong. The
        seam this guards was ~15%, far outside that.
        """
        whole = _count_map(12)          # nx=12: a single nav chunk
        split_6 = _count_map(6)         # boundary at x=6
        split_4 = _count_map(4)         # boundaries at x=4 and x=8

        for label, got in (("x=6", split_6), ("x=4/8", split_4)):
            diff = np.abs(got.astype(float) - whole.astype(float))
            assert diff.max() <= 1, (
                f"splitting the scan at {label} changed the vectors found by up "
                f"to {diff.max():.0f} per position (whole={whole.tolist()}, "
                f"split={got.tolist()})")
            assert diff.mean() <= 0.05, (
                f"splitting the scan at {label} shifted {diff.mean():.3f} "
                f"vectors per position on average — chunking is changing the "
                f"answer")

    def test_no_stripe_at_the_boundary_columns(self):
        """The failure mode, stated directly: the columns either side of a chunk
        boundary must not carry more vectors than their neighbours do."""
        split = _count_map(4)           # boundaries at x=4 and x=8
        col = split.mean(axis=0)
        for b in (4, 8):
            edge = 0.5 * (col[b - 1] + col[b])
            near = 0.5 * (col[b - 2] + col[b + 1])
            assert edge <= near * 1.05 + 0.5, (
                f"boundary columns {b - 1}/{b} carry {edge:.1f} vectors vs "
                f"{near:.1f} either side — the refine is seeing fewer "
                f"neighbours at the chunk edge (profile: {col.round(1).tolist()})")

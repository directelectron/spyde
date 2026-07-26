"""ArrayCache reader kinds 2+3: zarr+blosc (.zspy) and HDF5 (.hspy), served by
one SourceArrayReader (spyde/array_cache/readers/source_array.py).

Pins: find_source_array pulls the live h5py.Dataset / zarr.core.Array out of a
real hyperspy-saved lazy signal's da.from_array graph; frames read straight
from the store are byte-identical to a dask compute (1-D and 2-D nav); a plain
in-RAM ndarray source and every DERIVED view decline (a derived view still
carries the source layer in its graph — reading through it would silently
return the untransformed source frame); and resolve_reader's ordering picks this
kind for .hspy/.zspy but LocalTransformReader for a view over them.
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs

from spyde.array_cache.readers.local_transform import LocalTransformReader
from spyde.array_cache.readers.source_array import (
    SourceArrayReader,
    find_source_array,
)
from spyde.array_cache.resolve import resolve_reader


class _AxesManager:
    def __init__(self, nav_dim):
        self.navigation_dimension = nav_dim


class _Signal:
    def __init__(self, data, nav_dim):
        self.data = data
        self.axes_manager = _AxesManager(nav_dim)


def _saved_lazy(tmp_path, ext, shape=(8, 6, 16, 16), chunks=(4, 3, 16, 16)):
    data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    s = hs.signals.Signal2D(data)
    path = str(tmp_path / f"t.{ext}")
    s.save(path, chunks=chunks)
    return hs.load(path, lazy=True), data


class TestFindSourceArray:
    def test_hspy_yields_h5py_dataset(self, tmp_path):
        lazy, _ = _saved_lazy(tmp_path, "hspy")
        source = find_source_array(lazy.data)
        assert source is not None
        assert type(source).__module__.startswith("h5py")
        assert tuple(source.shape) == tuple(lazy.data.shape)

    def test_zspy_yields_zarr_array(self, tmp_path):
        lazy, _ = _saved_lazy(tmp_path, "zspy")
        source = find_source_array(lazy.data)
        assert source is not None
        assert type(source).__module__.startswith("zarr")
        assert tuple(source.shape) == tuple(lazy.data.shape)

    def test_in_ram_ndarray_source_declines(self):
        # Nothing to cache — the data is already fully resident.
        arr = da.from_array(np.zeros((4, 4, 4), np.uint16), chunks=(2, -1, -1))
        assert find_source_array(arr) is None

    def test_memmap_source_is_accepted(self, tmp_path):
        path = str(tmp_path / "mm.bin")
        base = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(6, 4, 4)
        base.tofile(path)
        mm = np.memmap(path, np.uint16, mode="r", shape=(6, 4, 4))
        arr = da.from_array(mm, chunks=(2, -1, -1))
        source = find_source_array(arr)
        assert isinstance(source, np.memmap)


class TestFrameParity:
    def test_hspy_frames_match_compute(self, tmp_path):
        lazy, data = _saved_lazy(tmp_path, "hspy")
        sig = _Signal(lazy.data, nav_dim=2)
        reader = SourceArrayReader(sig, lazy.data, find_source_array(lazy.data))
        for iy, ix in [(0, 0), (3, 2), (7, 5)]:
            np.testing.assert_array_equal(reader.read_frame((iy, ix)), data[iy, ix])

    def test_zspy_frames_match_compute(self, tmp_path):
        lazy, data = _saved_lazy(tmp_path, "zspy")
        sig = _Signal(lazy.data, nav_dim=2)
        reader = SourceArrayReader(sig, lazy.data, find_source_array(lazy.data))
        for iy, ix in [(0, 0), (3, 2), (7, 5)]:
            np.testing.assert_array_equal(reader.read_frame((iy, ix)), data[iy, ix])

    def test_1d_nav_movie_frames_match(self, tmp_path):
        shape, chunks = (6, 8, 8), (2, 8, 8)
        data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
        s = hs.signals.Signal2D(data)
        path = str(tmp_path / "movie.hspy")
        s.save(path, chunks=chunks)
        lazy = hs.load(path, lazy=True)
        sig = _Signal(lazy.data, nav_dim=1)
        reader = SourceArrayReader(sig, lazy.data, find_source_array(lazy.data))
        for i in range(shape[0]):
            np.testing.assert_array_equal(reader.read_frame((i,)), data[i])

    def test_returned_frame_owns_its_memory(self, tmp_path):
        """A cached frame must not be a view onto a bigger buffer (honest
        ArrayCache byte accounting) — matters for the memmap source."""
        path = str(tmp_path / "mm.bin")
        base = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(6, 4, 4)
        base.tofile(path)
        mm = np.memmap(path, np.uint16, mode="r", shape=(6, 4, 4))
        arr = da.from_array(mm, chunks=(2, -1, -1))
        sig = _Signal(arr, nav_dim=1)
        reader = SourceArrayReader(sig, arr, find_source_array(arr))
        frame = reader.read_frame((3,))
        assert frame.base is None
        np.testing.assert_array_equal(frame, base[3])


class TestBlockMemo:
    """A CHUNKED store decodes a whole chunk per access anyway, so the reader
    reads the nav-chunk once and slices frames out of it — that is what makes
    dwell-in-chunk ~0 ms instead of a per-frame re-decompress on .zspy."""

    def test_chunked_source_memoizes_the_nav_block(self, tmp_path):
        lazy, data = _saved_lazy(tmp_path, "zspy", chunks=(4, 3, 16, 16))
        sig = _Signal(lazy.data, nav_dim=2)
        reader = SourceArrayReader(sig, lazy.data, find_source_array(lazy.data))
        assert reader._nav_chunks == (4, 3)

        np.testing.assert_array_equal(reader.read_frame((1, 1)), data[1, 1])
        block = reader._memo[1]
        assert reader._memo[0] == (0, 0)
        assert block.shape[:2] == (4, 3)

        # Another frame in the same storage chunk reuses the block…
        np.testing.assert_array_equal(reader.read_frame((3, 2)), data[3, 2])
        assert reader._memo[1] is block
        # …a frame in a different chunk replaces it.
        np.testing.assert_array_equal(reader.read_frame((5, 4)), data[5, 4])
        assert reader._memo[0] == (1, 1)
        assert reader._memo[1] is not block

    def test_is_chunk_resident_tracks_the_memo(self, tmp_path):
        lazy, _ = _saved_lazy(tmp_path, "zspy", chunks=(4, 3, 16, 16))
        sig = _Signal(lazy.data, nav_dim=2)
        reader = SourceArrayReader(sig, lazy.data, find_source_array(lazy.data))

        assert reader.is_chunk_resident((1, 1)) is False
        reader.read_frame((1, 1))
        assert reader.is_chunk_resident((1, 1)) is True
        assert reader.is_chunk_resident((3, 2)) is True   # same storage chunk
        assert reader.is_chunk_resident((5, 4)) is False  # a different one

    def test_oversized_block_falls_back_to_frame_reads(self, tmp_path, monkeypatch):
        """A storage chunk spanning many nav positions of big frames would cost
        more RAM than the dwell win is worth — read the single frame instead."""
        import spyde.array_cache.readers.source_array as sa

        lazy, data = _saved_lazy(tmp_path, "zspy", chunks=(4, 3, 16, 16))
        monkeypatch.setattr(sa, "MAX_BLOCK_BYTES", 1)     # every block is "too big"
        sig = _Signal(lazy.data, nav_dim=2)
        reader = sa.SourceArrayReader(sig, lazy.data, find_source_array(lazy.data))
        assert reader._nav_chunks is None
        np.testing.assert_array_equal(reader.read_frame((3, 2)), data[3, 2])
        assert reader._memo is None
        assert reader.is_chunk_resident((3, 2)) is False

    def test_single_frame_chunk_does_not_memoize(self, tmp_path):
        """1 nav position per storage chunk (an in-situ movie) — the block IS
        the frame, so there is nothing to amortise."""
        shape, chunks = (6, 8, 8), (1, 8, 8)
        data = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
        path = str(tmp_path / "movie.zspy")
        hs.signals.Signal2D(data).save(path, chunks=chunks)
        lazy = hs.load(path, lazy=True)
        sig = _Signal(lazy.data, nav_dim=1)
        reader = SourceArrayReader(sig, lazy.data, find_source_array(lazy.data))
        assert reader._nav_chunks is None
        np.testing.assert_array_equal(reader.read_frame((4,)), data[4])


class TestDerivedViewsDecline:
    """REGRESSION (same class of bug as the binary reader's): a dask
    HighLevelGraph keeps ancestor layers, so a derived view still carries the
    source's original-array layer. Reading through it would return the
    UNTRANSFORMED source frame."""

    def test_nav_crop_declines_and_falls_back(self, tmp_path):
        lazy, _ = _saved_lazy(tmp_path, "hspy")
        crop = lazy.data[1:5, 2:5]
        assert find_source_array(crop) is None
        assert isinstance(resolve_reader(_Signal(crop, 2), crop), LocalTransformReader)

    def test_signal_rebin_declines_and_falls_back(self, tmp_path):
        lazy, _ = _saved_lazy(tmp_path, "zspy")
        reb = da.coarsen(np.mean, lazy.data, {2: 2, 3: 2})
        assert find_source_array(reb) is None
        assert isinstance(resolve_reader(_Signal(reb, 2), reb), LocalTransformReader)

    def test_derived_frames_are_the_transformed_ones(self, tmp_path):
        lazy, data = _saved_lazy(tmp_path, "hspy")
        reb = da.coarsen(np.mean, lazy.data, {2: 2, 3: 2})
        reader = resolve_reader(_Signal(reb, 2), reb)
        got = reader.read_frame((3, 2))
        expected = np.asarray(reb[3, 2].compute())
        assert got.shape == expected.shape != data.shape[2:]
        np.testing.assert_allclose(got, expected)


class TestResolveOrdering:
    def test_hspy_resolves_source_array_reader(self, tmp_path):
        lazy, _ = _saved_lazy(tmp_path, "hspy")
        assert isinstance(
            resolve_reader(_Signal(lazy.data, 2), lazy.data), SourceArrayReader)

    def test_zspy_resolves_source_array_reader(self, tmp_path):
        lazy, _ = _saved_lazy(tmp_path, "zspy")
        assert isinstance(
            resolve_reader(_Signal(lazy.data, 2), lazy.data), SourceArrayReader)

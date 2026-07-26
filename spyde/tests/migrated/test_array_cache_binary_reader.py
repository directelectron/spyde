"""ArrayCache reader kind 1: binary uncompressed (spyde/array_cache/readers/
binary.py + resolve.py), backed by rosettasciio's memmap_distributed
primitives rather than a re-derived header parser.

Pins: find_memmap_source correctly extracts (file, dtypes, shape, offset,
order, key) from a real hyperspy-loaded lazy signal's dask graph; a
BinaryReader built from those is byte-identical to a plain compute, on both
its pread fast path and its np.memmap fallback; multi-dim nav indexing is
correct; resolve_reader declines (falls back to LocalTransformReader) for
data with no memmap_distributed graph and for arbitrary-positions scans;
close()/close_all_readers release the open file descriptor.
"""
from __future__ import annotations

import os

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest

from rsciio.mrc._api import get_std_dtype_list
from rsciio.utils._distributed import memmap_distributed

from spyde.array_cache.readers.binary import BinaryReader, find_memmap_source
from spyde.array_cache.readers.local_transform import LocalTransformReader
from spyde.array_cache.resolve import resolve_reader

# ``os.pread`` is Unix-only. BinaryReader gates its zero-copy fast path on it and
# uses the memmap slice everywhere else (binary.py), so on Windows the fast path
# is CORRECTLY absent — tests that assert it unconditionally are asserting a
# platform, not a contract. The parity/fd tests below are split accordingly:
# the pread specifics are skipped, and the behaviour that must hold on EVERY
# platform (same bytes out, resources released) is asserted everywhere.
HAS_PREAD = hasattr(os, "pread")
requires_pread = pytest.mark.skipif(
    not HAS_PREAD, reason="os.pread is Unix-only; BinaryReader uses the memmap "
                          "fallback on this platform")


class _AxesManager:
    def __init__(self, nav_dim):
        self.navigation_dimension = nav_dim


class _Signal:
    def __init__(self, data, nav_dim):
        self.data = data
        self.axes_manager = _AxesManager(nav_dim)


def _write_synthetic_mrc(path, data, mode=6):
    """A minimal valid MRC (1024-byte standard header, no extended header) —
    data shape (NZ, NY, NX), mode 6 = uint16."""
    NZ, NY, NX = data.shape
    header = np.zeros(1, dtype=get_std_dtype_list("<"))
    header["NX"], header["NY"], header["NZ"] = NX, NY, NZ
    header["MODE"] = mode
    header["MX"], header["MY"], header["MZ"] = NX, NY, NZ
    header["Xlen"], header["Ylen"], header["Zlen"] = NX, NY, NZ
    header["MAPC"], header["MAPR"], header["MAPS"] = 1, 2, 3
    header["NEXT"] = 0
    header["NLABL"] = 0
    header["CMAP"] = b"MAP "
    header["STAMP"] = b"DA\x00\x00"
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(data.tobytes())


class TestFindMemmapSourceRealMRC:
    def test_extracts_correct_kwargs_and_offset(self, tmp_path):
        data = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(6, 4, 4)
        path = str(tmp_path / "synthetic.mrc")
        _write_synthetic_mrc(path, data)

        s = hs.load(path, lazy=True)
        kwargs = find_memmap_source(s.data)
        assert kwargs is not None
        assert kwargs["file"] == path
        assert kwargs["offset"] == 1024  # standard header, no extension
        assert tuple(int(v) for v in kwargs["shape"]) == (6, 4, 4)
        assert np.dtype(kwargs["dtypes"]) == np.uint16

    def test_byte_identical_to_compute(self, tmp_path):
        data = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(6, 4, 4)
        path = str(tmp_path / "synthetic.mrc")
        _write_synthetic_mrc(path, data)

        s = hs.load(path, lazy=True)
        kwargs = find_memmap_source(s.data)
        sig = _Signal(s.data, nav_dim=1)
        reader = BinaryReader(sig, s.data, kwargs)
        try:
            for i in range(6):
                expected = np.asarray(s.data[i].compute())
                got = reader.read_frame((i,))
                np.testing.assert_array_equal(got, expected)
        finally:
            reader.close()

    def test_no_memmap_source_for_non_memmap_array(self):
        arr = da.from_array(np.zeros((4, 4, 4), dtype=np.uint16), chunks=(1, -1, -1))
        assert find_memmap_source(arr) is None


class TestDerivedViewMustNotResolveBinary:
    """REGRESSION: a dask HighLevelGraph keeps every ancestor layer, so a
    DERIVED view of a memmap-backed file still carries the source's
    ``slice_memmap`` tasks. Matching those would serve the RAW SOURCE frame
    with the transform silently skipped — and rebin/crop are exactly the
    transforms the locality tag marks ArrayCache-eligible, so this is the new
    happy path, not a corner case. find_memmap_source must match on
    ``data.name`` (the array IS the memmap layer's output), not on any
    slice_memmap key anywhere in the graph."""

    def _memmap_4d(self, tmp_path):
        nav_y, nav_x, h, w = 4, 6, 8, 8
        data = np.arange(nav_y * nav_x * h * w, dtype=np.uint16).reshape(
            nav_y, nav_x, h, w)
        path = str(tmp_path / "raw4d.bin")
        data.tofile(path)
        arr = memmap_distributed(
            path, dtype=np.dtype(np.uint16), offset=0,
            shape=(nav_y, nav_x, h, w), chunks=(2, 2, -1, -1),
        )
        return arr, data

    def test_base_array_still_resolves_binary(self, tmp_path):
        arr, _ = self._memmap_4d(tmp_path)
        assert find_memmap_source(arr) is not None

    def test_nav_crop_declines(self, tmp_path):
        arr, _ = self._memmap_4d(tmp_path)
        crop = arr[1:3, 2:5]
        assert find_memmap_source(crop) is None, \
            "a cropped view would read the wrong frame through BinaryReader"
        sig = _Signal(crop, nav_dim=2)
        assert isinstance(resolve_reader(sig, crop), LocalTransformReader)

    def test_signal_rebin_declines(self, tmp_path):
        arr, _ = self._memmap_4d(tmp_path)
        reb = da.coarsen(np.mean, arr, {2: 2, 3: 2})
        assert find_memmap_source(reb) is None, \
            "a rebinned view would read raw un-binned frames through BinaryReader"
        sig = _Signal(reb, nav_dim=2)
        assert isinstance(resolve_reader(sig, reb), LocalTransformReader)

    def test_derived_read_through_resolve_reader_matches_compute(self, tmp_path):
        """End-to-end: whatever reader resolve_reader picks for a derived view,
        its frame must equal the view's own compute()."""
        arr, _ = self._memmap_4d(tmp_path)
        for view in (arr[1:3, 2:5], da.coarsen(np.mean, arr, {2: 2, 3: 2})):
            sig = _Signal(view, nav_dim=2)
            reader = resolve_reader(sig, view)
            try:
                got = reader.read_frame((0, 0))
                expected = np.asarray(view[0, 0].compute())
                assert got.shape == expected.shape
                np.testing.assert_allclose(got, expected)
            finally:
                close = getattr(reader, "close", None)
                if close is not None:
                    close()


class TestBinaryReaderFastPathVsFallback:
    def test_fast_path_disabled_without_os_pread(self, tmp_path, monkeypatch):
        """os.pread is Unix-only — on Windows the fast path must not be armed
        (it would raise AttributeError on every frame and silently degrade to a
        plain compute upstream); the memmap fallback serves instead."""
        data = np.arange(6 * 4 * 4, dtype=np.uint16).reshape(6, 4, 4)
        path = str(tmp_path / "synthetic.mrc")
        _write_synthetic_mrc(path, data)
        s = hs.load(path, lazy=True)
        kwargs = find_memmap_source(s.data)
        sig = _Signal(s.data, nav_dim=1)

        monkeypatch.delattr(os, "pread", raising=False)
        reader = BinaryReader(sig, s.data, kwargs)
        try:
            assert reader._fast_path is False
            assert reader._fd is None
            np.testing.assert_array_equal(reader.read_frame((2,)), data[2])
        finally:
            reader.close()

    @requires_pread
    def test_pread_matches_memmap_fallback(self, tmp_path):
        """The two read mechanisms must agree byte for byte."""
        data = np.random.RandomState(0).randint(0, 4000, (10, 8, 8)).astype(np.uint16)
        path = str(tmp_path / "synthetic.mrc")
        _write_synthetic_mrc(path, data)

        s = hs.load(path, lazy=True)
        kwargs = find_memmap_source(s.data)
        sig = _Signal(s.data, nav_dim=1)
        reader = BinaryReader(sig, s.data, kwargs)
        try:
            assert reader._fast_path is True
            for i in (0, 5, 9):
                fast = reader.read_frame((i,))
                reader._fast_path = False
                slow = reader.read_frame((i,))
                reader._fast_path = True
                np.testing.assert_array_equal(fast, slow)
                np.testing.assert_array_equal(fast, data[i])
        finally:
            reader.close()

    def test_frames_are_correct_on_any_platform(self, tmp_path):
        """Whichever mechanism this platform selects, the frames must be right.

        The parity test above can only run where os.pread exists; this one is the
        contract that has to hold everywhere, so Windows still has real coverage
        of BinaryReader's output rather than just a skip."""
        data = np.random.RandomState(1).randint(0, 4000, (10, 8, 8)).astype(np.uint16)
        path = str(tmp_path / "synthetic.mrc")
        _write_synthetic_mrc(path, data)

        s = hs.load(path, lazy=True)
        kwargs = find_memmap_source(s.data)
        sig = _Signal(s.data, nav_dim=1)
        reader = BinaryReader(sig, s.data, kwargs)
        try:
            assert reader._fast_path is HAS_PREAD, \
                "the fast path should track os.pread availability"
            for i in (0, 5, 9):
                np.testing.assert_array_equal(reader.read_frame((i,)), data[i])
        finally:
            reader.close()

    def test_frame_bytes(self, tmp_path):
        data = np.zeros((6, 8, 8), dtype=np.uint16)
        path = str(tmp_path / "synthetic.mrc")
        _write_synthetic_mrc(path, data)
        s = hs.load(path, lazy=True)
        kwargs = find_memmap_source(s.data)
        sig = _Signal(s.data, nav_dim=1)
        reader = BinaryReader(sig, s.data, kwargs)
        try:
            assert reader.frame_bytes == 8 * 8 * 2
        finally:
            reader.close()


class TestBinaryReaderMultiDimNav:
    def test_2d_nav_flat_index_matches_expected_frame(self, tmp_path):
        # Build a memmap_distributed-backed array directly (2-D nav) rather
        # than needing a real 4-D-STEM MRC file — same underlying graph shape
        # find_memmap_source/BinaryReader consume, just constructed by hand.
        nav_y, nav_x, h, w = 3, 5, 4, 4
        data = np.arange(nav_y * nav_x * h * w, dtype=np.uint16).reshape(nav_y, nav_x, h, w)
        path = str(tmp_path / "raw4d.bin")
        data.tofile(path)

        arr = memmap_distributed(
            path, dtype=np.dtype(np.uint16), offset=0,
            shape=(nav_y, nav_x, h, w), chunks=(1, 1, -1, -1),
        )
        kwargs = find_memmap_source(arr)
        assert kwargs is not None
        sig = _Signal(arr, nav_dim=2)
        reader = BinaryReader(sig, arr, kwargs)
        try:
            for iy, ix in [(0, 0), (1, 3), (2, 4)]:
                expected = data[iy, ix]
                got = reader.read_frame((iy, ix))
                np.testing.assert_array_equal(got, expected)
        finally:
            reader.close()


class TestBinaryReaderDeclinesArbitraryPositions:
    def test_positions_true_returns_none(self, tmp_path):
        n, h, w = 5, 4, 4
        data = np.arange(n * h * w, dtype=np.uint16).reshape(n, h, w)
        path = str(tmp_path / "raw.bin")
        data.tofile(path)
        # positions index into the leading (nav) dims only — shape[:-2] == (n,).
        positions = np.array([[0], [1], [2], [3], [4]])
        arr = memmap_distributed(
            path, dtype=np.dtype(np.uint16), offset=0,
            shape=(n, h, w), positions=positions, chunks=(1, -1, -1),
        )
        assert find_memmap_source(arr) is None


class TestResolveReaderFallback:
    def test_falls_back_to_local_transform_for_derived_view(self):
        base = np.random.RandomState(0).randint(0, 4000, (16, 8, 8)).astype(np.uint16)
        raw = da.from_array(base, chunks=(4, -1, -1))
        reb = da.coarsen(np.mean, raw, {1: 2, 2: 2})  # derived, no memmap graph
        sig = _Signal(reb, nav_dim=1)
        reader = resolve_reader(sig, reb)
        assert isinstance(reader, LocalTransformReader)

    def test_resolves_binary_reader_for_memmap_backed(self, tmp_path):
        data = np.zeros((4, 4, 4), dtype=np.uint16)
        path = str(tmp_path / "synthetic.mrc")
        _write_synthetic_mrc(path, data)
        s = hs.load(path, lazy=True)
        sig = _Signal(s.data, nav_dim=1)
        reader = resolve_reader(sig, s.data)
        try:
            assert isinstance(reader, BinaryReader)
        finally:
            reader.close()


class TestReaderCleanup:
    def _reader(self, tmp_path):
        data = np.zeros((4, 4, 4), dtype=np.uint16)
        path = str(tmp_path / "synthetic.mrc")
        _write_synthetic_mrc(path, data)
        s = hs.load(path, lazy=True)
        kwargs = find_memmap_source(s.data)
        return BinaryReader(_Signal(s.data, nav_dim=1), s.data, kwargs)

    @requires_pread
    def test_close_releases_fd(self, tmp_path):
        """On the pread path close() must actually release the descriptor —
        leaking one per opened node would exhaust the process's fd budget."""
        reader = self._reader(tmp_path)
        fd = reader._fd
        assert fd is not None
        reader.close()
        assert reader._fd is None
        with pytest.raises(OSError):
            os.pread(fd, 1, 0)

    def test_close_is_safe_on_any_platform(self, tmp_path):
        """close() has to be callable (and idempotent) wherever the reader runs —
        close_all_readers calls it on node switch and plot close regardless of
        which read mechanism was selected."""
        reader = self._reader(tmp_path)
        if not HAS_PREAD:
            assert reader._fd is None, "no fd is opened without the pread path"
        reader.close()
        assert reader._fd is None
        reader.close()          # idempotent — must not raise


class TestCloseAllReaders:
    def test_closes_and_clears_plot_readers(self, tmp_path):
        from spyde.array_cache import close_all_readers

        data = np.zeros((4, 4, 4), dtype=np.uint16)
        path = str(tmp_path / "synthetic.mrc")
        _write_synthetic_mrc(path, data)
        s = hs.load(path, lazy=True)
        kwargs = find_memmap_source(s.data)
        sig = _Signal(s.data, nav_dim=1)
        reader = BinaryReader(sig, s.data, kwargs)

        class _FakePlot:
            pass

        plot = _FakePlot()
        plot._local_transform_readers = {id(sig): reader}
        close_all_readers(plot)
        assert reader._fd is None
        assert len(plot._local_transform_readers) == 0

"""
Direct Electron writes ``.de5`` axis arrays as ``(N, 1)`` columns and stores the
datacube scan-first in C order. Stock rsciio's EMD reader dies on the former,
reads the whole datacube into RAM even for a lazy load, and transposes the
latter into a signal-only array. The patch in ``spyde.external.rosettasciio.de5``
fixes all three; these tests pin that contract against a synthetic file shaped
like the camera's output.
"""
import h5py
import numpy as np

from spyde.external.rosettasciio import de5 as de5_patch


def write_de5(path, nav=(3, 2), signal=(4, 5), direct_electron=True,
              chunks="frame", written=None):
    """A minimal ``.de5`` shaped like the camera's output, with column axes.

    ``chunks="frame"`` stores one frame per HDF5 chunk as the camera does;
    ``None`` stores a contiguous dataset. ``written`` limits the scan positions
    that receive data, which is what an acquisition stopped early looks like:
    the dataset is declared at the planned size and the rest is never written."""
    shape = nav + signal
    with h5py.File(path, "w") as file:
        experiment = file.create_group("4DSTEM_experiment")
        experiment.attrs["emd_group_type"] = np.array([2], dtype=np.int32)
        experiment.attrs["version_major"] = np.array([0], dtype=np.int32)
        experiment.attrs["version_minor"] = np.array([6], dtype=np.int32)
        if direct_electron:
            camera = experiment.create_group("DirectElectronInfo")
            camera.attrs["FramesPerSecond"] = np.float32(100.0)
        datacube = experiment.create_group("data/datacubes/datacube_0")
        datacube.attrs["emd_group_type"] = np.array([1], dtype=np.int32)
        datacube.attrs["metadata"] = np.array([0], dtype=np.int32)
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        h5_chunks = (1,) * len(nav) + signal if chunks == "frame" else chunks
        if written is None:
            datacube.create_dataset("data", data=data, chunks=h5_chunks)
        else:
            dataset = datacube.create_dataset(
                "data", shape=shape, dtype=np.uint16, chunks=h5_chunks)
            for position in written:
                dataset[position] = data[position]
            data = np.zeros(shape, dtype=np.uint16)
            for position in written:
                data[position] = dataset[position]
        names = ["R_y", "R_x", "Q_y", "Q_x"]
        for index, (size, name) in enumerate(zip(shape, names), start=1):
            axis = datacube.create_dataset(
                f"dim{index}", data=np.arange(size, dtype=np.int32).reshape(size, 1))
            axis.attrs["name"] = np.bytes_(name.encode())
            axis.attrs["units"] = np.bytes_(b"[pix]")
        experiment.create_group("metadata/metadata_0")
    return data


def dask_source(signal):
    """The object dask reads from for this lazy signal: the live HDF5 dataset
    when the load is lazy, an in-RAM ndarray when the reader materialised it."""
    layers = signal.data.dask.layers
    sources = [list(layer.values())[0]
               for name, layer in layers.items() if name.startswith("original-")]
    assert len(sources) == 1, "expected exactly one from_array source layer"
    return sources[0]


class TestDe5Patch:
    def test_apply_is_idempotent_and_flattens_column_axis(self):
        from rsciio.emd._emd_ncem import EMD_NCEM

        assert de5_patch.apply() is True
        assert de5_patch.apply() is True
        offset, scale = EMD_NCEM._parse_axis(np.arange(5, dtype=np.int32).reshape(5, 1))
        assert (offset, scale) == (0.0, 1.0)
        offset, scale = EMD_NCEM._parse_axis(np.array([2.0, 4.0, 6.0]))
        assert (offset, scale) == (2.0, 2.0)
        assert EMD_NCEM._parse_axis(np.array(["a", "b"])) == (0, 1)

    def test_direct_electron_datacube_loads_in_storage_order(self, tmp_path):
        import hyperspy.api as hs

        de5_patch.apply()
        path = tmp_path / "movie.de5"
        expected = write_de5(path)
        signal = hs.load(str(path), lazy=True)
        assert signal.data.shape == expected.shape
        assert signal.axes_manager.navigation_shape == (2, 3)
        assert signal.axes_manager.signal_shape == (5, 4)
        assert [axis.name for axis in signal.axes_manager._axes] == ["R_y", "R_x", "Q_y", "Q_x"]
        np.testing.assert_array_equal(signal.data.compute(), expected)
        np.testing.assert_array_equal(signal.inav[1, 2].data.compute(), expected[2, 1])

    def test_lazy_load_reads_from_the_file_not_from_ram(self, tmp_path):
        """A stopped-early acquisition: 8 x 8 declared, only the first row
        written. The lazy load must wrap the HDF5 dataset itself, in storage
        order with no transpose layers, so SpyDE's frame reader can read
        straight from the file and only the frames asked for."""
        import hyperspy.api as hs
        from spyde.array_cache.readers.source_array import find_source_array

        de5_patch.apply()
        path = tmp_path / "stopped_early.de5"
        written = [(0, column) for column in range(8)]
        expected = write_de5(path, nav=(8, 8), signal=(16, 16), written=written)
        signal = hs.load(str(path), lazy=True)
        source = dask_source(signal)
        assert isinstance(source, h5py.Dataset)
        assert find_source_array(signal.data) is source
        assert signal.data.shape == (8, 8, 16, 16)
        assert all(len(chunk) == 1 for chunk in signal.data.chunks[2:])
        np.testing.assert_array_equal(signal.inav[3, 0].data.compute(), expected[0, 3])
        assert not signal.inav[3, 5].data.compute().any()

    def test_contiguous_dataset_keeps_frames_whole(self, tmp_path):
        import hyperspy.api as hs

        de5_patch.apply()
        path = tmp_path / "contiguous.de5"
        expected = write_de5(path, nav=(3, 2), signal=(4, 5), chunks=None)
        signal = hs.load(str(path), lazy=True)
        assert isinstance(dask_source(signal), h5py.Dataset)
        assert signal.data.chunks == ((3,), (2,), (4,), (5,))
        np.testing.assert_array_equal(signal.data.compute(), expected)

    def test_lazy_chunks_are_64mb_blocks_of_whole_frames(self):
        class Dataset:
            def __init__(self, shape, dtype, chunks):
                self.shape, self.dtype, self.chunks = shape, np.dtype(dtype), chunks

        # The camera's real layout: 128 KB frames, one per HDF5 chunk. 64 MB is
        # 512 frames; the largest power-of-two square inside that is 16 x 16.
        camera = Dataset((128, 64, 256, 256), np.uint16, (1, 1, 256, 256))
        assert de5_patch.lazy_dataset_chunks(camera) == (16, 16, 256, 256)
        # 2 MB frames: 32 per block, so a 4 x 4 square.
        large = Dataset((12, 12, 1024, 1024), np.uint16, (1, 1, 1024, 1024))
        assert de5_patch.lazy_dataset_chunks(large) == (4, 4, 1024, 1024)
        # A block never exceeds the scan, and never exceeds 32 per axis.
        tiny = Dataset((8, 8, 16, 16), np.uint16, (1, 1, 16, 16))
        assert de5_patch.lazy_dataset_chunks(tiny) == (8, 8, 16, 16)
        huge_scan = Dataset((1000, 1000, 16, 16), np.uint16, (1, 1, 16, 16))
        assert de5_patch.lazy_dataset_chunks(huge_scan) == (32, 32, 16, 16)
        # A file chunked in 3 x 3 scan blocks gets a multiple of 3, not 16.
        blocked = Dataset((128, 64, 256, 256), np.uint16, (3, 3, 256, 256))
        assert de5_patch.lazy_dataset_chunks(blocked) == (15, 15, 256, 256)
        # A frame stack blocks along time; an image is one chunk.
        stack = Dataset((5000, 256, 256), np.uint16, (1, 256, 256))
        assert de5_patch.lazy_dataset_chunks(stack) == (32, 256, 256)
        assert de5_patch.lazy_dataset_chunks(Dataset((256, 256), np.uint16, None)) == (256, 256)

    def test_eager_load_is_unchanged(self, tmp_path):
        import hyperspy.api as hs

        de5_patch.apply()
        path = tmp_path / "eager.de5"
        expected = write_de5(path)
        signal = hs.load(str(path), lazy=False)
        assert isinstance(signal.data, np.ndarray)
        np.testing.assert_array_equal(signal.data, expected)

    def test_other_emd_files_keep_the_readers_layout(self, tmp_path):
        import hyperspy.api as hs

        de5_patch.apply()
        path = tmp_path / "generic.emd"
        expected = write_de5(path, direct_electron=False)
        signal = hs.load(str(path), lazy=True)
        assert signal.data.shape == expected.shape[::-1]
        assert signal.axes_manager.navigation_dimension == 0

    def test_restore_storage_order_leaves_images_alone(self):
        dictionary = {
            "data": np.zeros((4, 5)),
            "axes": [{"index_in_array": 0, "navigate": False},
                     {"index_in_array": 1, "navigate": False}],
        }
        de5_patch.restore_storage_order(dictionary)
        assert dictionary["data"].shape == (4, 5)
        assert [axis["navigate"] for axis in dictionary["axes"]] == [False, False]

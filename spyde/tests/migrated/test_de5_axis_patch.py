"""
Direct Electron writes ``.de5`` axis arrays as ``(N, 1)`` columns and stores the
datacube scan-first in C order. Stock rsciio's EMD reader dies on the former and
transposes the latter into a signal-only array. The patch in
``spyde.external.rosettasciio.de5`` fixes both; these tests pin that contract
against a synthetic file shaped like the camera's output.
"""
import h5py
import numpy as np

from spyde.external.rosettasciio import de5 as de5_patch


def write_de5(path, nav=(3, 2), signal=(4, 5), direct_electron=True):
    """A minimal ``.de5`` shaped like the camera's output, with column axes."""
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
        datacube.create_dataset("data", data=data, chunks=(1, 1) + signal)
        names = ["R_y", "R_x", "Q_y", "Q_x"]
        for index, (size, name) in enumerate(zip(shape, names), start=1):
            axis = datacube.create_dataset(
                f"dim{index}", data=np.arange(size, dtype=np.int32).reshape(size, 1))
            axis.attrs["name"] = np.bytes_(name.encode())
            axis.attrs["units"] = np.bytes_(b"[pix]")
        experiment.create_group("metadata/metadata_0")
    return data


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

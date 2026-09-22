"""File → Export to NumPy: what a window writes into its ``.npz``.

Each result type is committed the way its wizard commits it, exported from
its window, and read back with nothing but ``np.load``. The two things that
matter most are the negative ones: a window on a DATASET writes the frame it
shows and never the dataset, and a virtual image writes the image and not
the scan it was cut from — both share a tree with the raw data.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from spyde.actions.commit import commit_result_tree
from spyde.actions.export_numpy import Export, collect, read_meta, slug, write
from spyde.tests.migrated.conftest import _settle


def _signal_plot(session, tree=None):
    for plot in session._plots:
        if getattr(plot, "is_navigator", False):
            continue
        if tree is not None and plot.signal_tree is not tree:
            continue
        if getattr(getattr(plot, "plot_state", None), "current_signal", None) is not None:
            return plot
    raise AssertionError("no data window open")


def _calibrated_scan(ny=4, nx=5, scale=2.5, units="nm"):
    import hyperspy.api as hs
    sig = hs.signals.Signal2D(np.zeros((ny, nx, 3, 3), np.float32))
    for axis in sig.axes_manager.navigation_axes:
        axis.scale, axis.offset, axis.units = scale, 0.0, units
    return sig


def _exported(session, path, plot=None, window_id=None, timeout=20.0):
    """Run the export the way the menu does and wait for the file."""
    session._export_numpy(str(path), plot, window_id)
    written = str(path) if str(path).endswith((".npz", ".npy")) else str(path) + ".npz"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(written) and os.path.getsize(written) > 0:
            time.sleep(0.1)
            return written
        time.sleep(0.05)
    raise AssertionError(f"Export never produced {written}")


class TestKeys:
    """A key is something one can type after ``np.load``."""

    @pytest.mark.parametrize("label, key", [
        ("εxx (%)", "exx"),
        ("ω (°)", "omega"),
        ("Bx (mrad)", "Bx"),
        ("|B| (mrad)", "abs_B"),
        ("Direction (rad)", "Direction"),
        ("Virtual Image 1 (red)", "Virtual_Image_1_red"),
        ("IPF-Z", "IPF_Z"),
        ("Si — Strain", "Si_Strain"),
        ("", "array"),
        ("2theta", "_2theta"),
    ])
    def test_labels_become_identifiers(self, label, key):
        assert slug(label) == key

    def test_a_second_array_never_overwrites_the_first(self):
        export = Export()
        assert export.add("εxx (%)", np.ones(3)) == "exx"
        assert export.add("exx", np.zeros(3)) is None
        assert export.arrays["exx"].sum() == 3


class TestCommittedStrainTree:
    """The Strain window's Commit: εxx is the root, the others are child nodes."""

    def _commit(self, session):
        from spyde.actions._common import STRAIN_TITLES, strain_quantity
        maps = {"exx": np.full((4, 5), 1.2, np.float32),
                "eyy": np.full((4, 5), -0.5, np.float32),
                "exy": np.zeros((4, 5), np.float32),
                "omega": np.full((4, 5), 1.0, np.float32)}
        tree = commit_result_tree(
            session, title="Strain", primary=maps["exx"],
            primary_label=STRAIN_TITLES["exx"],
            views=[(STRAIN_TITLES[c], maps[c]) for c in ("eyy", "exy", "omega")],
            levels="auto_sym", cmap="coolwarm", source_signal=_calibrated_scan(),
            value_units={STRAIN_TITLES[c]: strain_quantity(c) for c in maps},
            provenance={"action": "Strain Mapping", "params": {"ref_yx": [0, 0]}},
        )
        _settle(session)
        return tree, maps

    def test_every_component_is_in_the_file_under_its_name(self, window, tmp_path):
        session = window["window"]
        tree, maps = self._commit(session)
        path = _exported(session, tmp_path / "strain.npz", _signal_plot(session, tree))

        with np.load(path) as archive:
            for component, expected in maps.items():
                assert np.array_equal(archive[component], expected), component
            assert archive["x_axis"].shape == (5,)
            assert archive["y_axis"].shape == (4,)
            assert archive["x_axis"][1] == pytest.approx(2.5)
        meta = read_meta(path)
        assert meta["primary"] == "exx"
        assert meta["arrays"]["exx"]["quantity"] == "εxx (%)"
        assert meta["arrays"]["omega"]["label"] == "ω"
        assert meta["axis_units"] == {"x": "nm", "y": "nm"}
        assert meta["provenance"]["action"] == "Strain Mapping"

    def test_npy_holds_the_shown_map_alone(self, window, tmp_path):
        session = window["window"]
        tree, maps = self._commit(session)
        path = _exported(session, tmp_path / "exx.npy", _signal_plot(session, tree))
        assert np.array_equal(np.load(path), maps["exx"])

    def test_no_extension_means_npz(self, window, tmp_path):
        session = window["window"]
        tree, _maps = self._commit(session)
        path = _exported(session, tmp_path / "strain", _signal_plot(session, tree))
        assert path.endswith("strain.npz")

    def test_the_menu_path_uses_the_active_window(self, window, tmp_path):
        session = window["window"]
        tree, maps = self._commit(session)
        session._active_window_id = _signal_plot(session, tree).window_id
        path = _exported(session, tmp_path / "active.npz")
        with np.load(path) as archive:
            assert np.array_equal(archive["eyy"], maps["eyy"])


class TestDpcResult:
    """A DPC Commit carries the result object; its field is in the file."""

    def _result(self):
        from spyde.actions.dpc import DpcResult
        rng = np.random.RandomState(1)
        field = rng.rand(4, 5, 2).astype(np.float32)
        return DpcResult(field=field, raw_shifts=field.copy(), reference=None,
                         units="mrad", mode="magnetic", rotation=12.0, flip=False,
                         reverse=False, rgb=np.zeros((4, 5, 3), np.uint8),
                         wheel=np.zeros((8, 8, 3), np.uint8), params={"method": "com"})

    def test_components_and_the_raw_field_round_trip(self, window, tmp_path):
        from spyde.actions import dpc as _dpc
        session = window["window"]
        result = self._result()
        titles = _dpc.component_titles("magnetic", "mrad")
        tree = commit_result_tree(
            session, title="DPC", primary=result.fx, primary_label=titles["fx"],
            views=[(titles[c], result.component(c)) for c in _dpc.COMPONENTS[1:]],
            attrs={"dpc_result": result}, source_signal=_calibrated_scan(),
            value_units={titles[c]: titles[c] for c in _dpc.COMPONENTS},
        )
        _settle(session)
        path = _exported(session, tmp_path / "dpc.npz", _signal_plot(session, tree))

        with np.load(path) as archive:
            assert np.allclose(archive["Bx"], result.fx)
            assert np.allclose(archive["By"], result.fy)
            assert np.allclose(archive["abs_B"], result.magnitude)
            assert set(archive.files) >= {"Direction", "Divergence", "Curl",
                                          "field", "raw_shifts"}
            assert np.array_equal(archive["field"], result.field)
        meta = read_meta(path)
        assert meta["dpc_result"]["units"] == "mrad"
        assert meta["dpc_result"]["rotation"] == 12.0
        assert meta["dpc_result"]["params"] == {"method": "com"}


class TestOrientationResult:
    """A vector-orientation window: an RGB picture on screen, the fit behind it."""

    def _result(self):
        from spyde.signals.orientation_map import VectorOrientationResult
        rng = np.random.RandomState(2)
        ny, nx = 4, 5
        quats = rng.rand(ny, nx, 4).astype(np.float32)
        return VectorOrientationResult(
            quats=quats, phase_idx=np.zeros((ny, nx), np.int16),
            theta=rng.rand(ny, nx).astype(np.float32),
            strain=rng.rand(ny, nx, 3).astype(np.float32) * 0.01,
            residual=rng.rand(ny, nx).astype(np.float32),
            friedel_asym=rng.rand(ny, nx).astype(np.float32),
            n_matched=np.full((ny, nx), 7, np.int16),
            coarse_score=rng.rand(ny, nx).astype(np.float32),
            phases_meta=[{"name": "Si", "point_group": "m-3m"}],
            nav_shape=(ny, nx), params={"library": "test"},
        )

    def test_the_picture_and_the_fit_are_both_in_the_file(self, window, tmp_path):
        session = window["window"]
        result = self._result()
        rgb = np.random.RandomState(3).randint(0, 255, (4, 5, 3)).astype(np.uint8)
        tree = commit_result_tree(
            session, title="Si — Orientation (IPF-Z)", primary=rgb,
            attrs={"vector_orientation": result}, source_signal=_calibrated_scan(),
        )
        _settle(session)
        path = _exported(session, tmp_path / "om.npz", _signal_plot(session, tree))

        with np.load(path) as archive:
            picture = archive[read_meta(path)["primary"]]
            assert picture.shape == (4, 5, 3)
            assert np.array_equal(picture, rgb)
            assert np.array_equal(archive["quats"], result.quats)
            assert np.array_equal(archive["strain"], result.strain)
            assert np.array_equal(archive["n_matched"], result.n_matched)
        meta = read_meta(path)
        assert meta["vector_orientation"]["phases_meta"][0]["name"] == "Si"
        assert meta["vector_orientation"]["nav_shape"] == [4, 5]

    def test_a_view_registered_on_the_window_rides_along(self, window, tmp_path):
        from spyde.actions.views import register_views
        session = window["window"]
        rgb = np.zeros((4, 5, 3), np.uint8)
        tree = commit_result_tree(session, title="Orientation", primary=rgb)
        _settle(session)
        plot = _signal_plot(session, tree)
        ipf_x = np.ones((4, 5, 3), np.uint8)
        register_views(plot.window_id, [("IPF-X", ipf_x)], append=True)

        path = _exported(session, tmp_path / "views.npz", plot)
        with np.load(path) as archive:
            assert np.array_equal(archive["IPF_X"], ipf_x)


class TestVirtualImage:
    """A live virtual image shares its tree with the scan it was cut from."""

    def test_the_image_is_written_and_the_scan_is_not(self, stem_4d_dataset, tmp_path):
        session = stem_4d_dataset["window"]
        source = _signal_plot(session)
        session._dispatch_toolbar_action(
            source, "add_virtual_image", {"type": "disk", "calculation": "mean"})
        _settle(session)
        artifact = session._action_artifacts[(source.window_id, "Virtual Image 1 (red)")]
        out_plot = session._plot_by_window_id(artifact["out_wids"][0])
        shown = np.asarray(out_plot.current_data)
        assert shown.shape == (4, 5)

        path = _exported(session, tmp_path / "vi.npz", out_plot)
        with np.load(path) as archive:
            assert np.array_equal(archive["Virtual_Image_1_red"], shown)
            assert all(archive[key].ndim <= 2 for key in archive.files), \
                "the scan itself leaked into a virtual-image export"
            # The image is over the scan, so its axes are the scan's.
            assert archive["x_axis"].shape == (5,)
            assert archive["y_axis"].shape == (4,)
        assert read_meta(path)["primary"] == "Virtual_Image_1_red"


class TestDatasetWindow:
    """A window on data exports the frame on screen and nothing more."""

    def test_only_the_shown_frame_of_a_scan(self, stem_4d_dataset, tmp_path):
        session = stem_4d_dataset["window"]
        plot = _signal_plot(session)
        shown = np.asarray(plot.current_data)

        path = _exported(session, tmp_path / "frame.npz", plot)
        with np.load(path) as archive:
            assert np.array_equal(archive["frame"], shown)
            assert max(archive[key].ndim for key in archive.files) == 2
            assert archive["x_axis"].shape == (16,)
        assert "navigation_index" in read_meta(path)

    def test_a_plain_image_is_exported_whole(self, tem_2d_dataset, tmp_path):
        session = tem_2d_dataset["window"]
        plot = _signal_plot(session)
        expected = np.asarray(session.signal_trees[0].root.data)
        path = _exported(session, tmp_path / "image.npz", plot)
        with np.load(path) as archive:
            key = read_meta(path)["primary"]
            assert np.array_equal(archive[key], expected)


class TestBareWindow:
    """The live Strain window has no Plot: its controller answers."""

    def test_the_strain_controller_exports_its_field(self, window, tmp_path):
        import anyplotlib as apl
        from spyde.actions.strain_action import StrainController
        from spyde.actions.strain_mapping import StrainField
        session = window["window"]
        fig, ax = apl.subplots()
        image = ax.imshow(np.zeros((4, 5), "f4"))
        controller = StrainController(None, image, window_id=4242, ref_yx=(0, 0),
                                      session=session,
                                      src_tree=type("T", (), {"root": None})())
        assert controller.export_arrays() == {}
        controller.field = StrainField(
            exx=np.full((4, 5), 0.012), eyy=np.full((4, 5), -0.005),
            exy=np.zeros((4, 5)), omega=np.full((4, 5), np.pi / 180.0),
            coverage=np.ones((4, 5)))
        session.register_window_controller(4242, controller)

        path = _exported(session, tmp_path / "live.npz", None, 4242)
        with np.load(path) as archive:
            assert np.allclose(archive["exx"], 1.2)
            assert np.allclose(archive["omega"], 1.0)
            assert np.allclose(archive["strain_exx"], 0.012)
            assert np.all(archive["coverage"] == 1)
        session._window_controllers.pop(4242, None)

    def test_a_window_with_nothing_says_so(self, window):
        session = window["window"]
        messages = window["messages"]
        messages.clear()
        session._export_numpy("nothing.npz", None, 9999)
        errors = [m for m in messages if m.get("type") == "error"]
        assert errors and "nothing to export" in errors[0].get("message", errors[0].get("text", ""))

    def test_collect_returns_none_without_a_window(self, window):
        assert collect(window["window"], None, None) is None

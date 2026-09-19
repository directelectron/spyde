"""
The composed arrays as calibrated HyperSpy signals.

Two things here are silent when wrong and so are pinned hardest: the angle axis
must stay an INDEX axis (a calibrated one would resolve many scrub positions to
one index, because the members are not evenly spaced in any coordinate), and
both origins must move by the crop (otherwise reciprocal space is offset from
the direct beam by the size of the crop, which looks plausible and corrupts
every g-vector taken from it).
"""
import dask.array as da
import numpy as np
import pytest
from hyperspy._signals.signal2d import LazySignal2D

from spyde.multiangle.compose import aligned_member
from spyde.multiangle.model import MultiAngleModel
from spyde.multiangle.recipe import recipe_for
from spyde.multiangle.signals import (
    ANGLE_AXIS_NAME, build_stack_signal, build_summed_signal,
    composite_navigator, shell_titles,
)

SCAN = (20, 24)
DETECTOR = (12, 12)
NAV_CHUNK = 8
N_MEMBERS = 4

SCAN_SCALE, SCAN_OFFSET = 2.0, 0.0
DETECTOR_SCALE, DETECTOR_OFFSET = 0.1, -0.6


@pytest.fixture
def acquisition():
    model = MultiAngleModel(
        paths=[f"member{index}.mrc" for index in range(N_MEMBERS)],
        tilts=np.array([1.0, 1.0, 0.5, 0.5]),
        azimuths=np.array([0.0, 180.0, 90.0, 270.0]),
        shell_ids=np.array([0, 0, 1, 1]),
        nav_offsets=np.array([(0, 0), (2, -1), (-3, 2), (1, 3)]),
        dp_offsets=np.array([(0, 0), (1, 1), (-2, 0), (0, -1)]),
    )
    generator = np.random.default_rng(11)
    members = []
    for _ in range(N_MEMBERS):
        raw = generator.integers(0, 3000, SCAN + DETECTOR, dtype=np.uint16)
        signal = LazySignal2D(
            da.from_array(raw, chunks=(NAV_CHUNK, NAV_CHUNK, -1, -1)))
        axes = signal.axes_manager._axes
        for axis, name in zip(axes[:2], ("y", "x")):
            axis.scale, axis.offset, axis.units, axis.name = (
                SCAN_SCALE, SCAN_OFFSET, "nm", name)
        for axis, name in zip(axes[2:], ("ky", "kx")):
            axis.scale, axis.offset, axis.units, axis.name = (
                DETECTOR_SCALE, DETECTOR_OFFSET, "1/A", name)
        members.append(signal)
    aligned = [aligned_member(member, model, index)
               for index, member in enumerate(members)]
    return aligned, members, model


class TestStackSignal:
    def test_it_keeps_the_angle_axis_and_stays_lazy(self, acquisition):
        aligned, members, model = acquisition
        stack = build_stack_signal(aligned, members, model)
        assert stack._lazy
        assert stack.axes_manager.navigation_dimension == 3
        assert stack.axes_manager.signal_dimension == 2
        assert stack.data.shape[0] == N_MEMBERS

    def test_the_angle_axis_is_an_index_axis(self, acquisition):
        """Calibrating it in degrees would be wrong: shells of differing tilt
        are not uniformly spaced, and a HyperSpy axis is uniform."""
        aligned, members, model = acquisition
        angle_axis = build_stack_signal(
            aligned, members, model).axes_manager._axes[0]
        assert angle_axis.name == ANGLE_AXIS_NAME
        assert angle_axis.scale == 1.0
        assert angle_axis.offset == 0.0
        assert angle_axis.size == N_MEMBERS

    def test_one_chunk_per_angle(self, acquisition):
        aligned, members, model = acquisition
        stack = build_stack_signal(aligned, members, model)
        assert stack.data.chunks[0] == (1,) * N_MEMBERS

    def test_the_acquisition_is_recorded_in_metadata(self, acquisition):
        aligned, members, model = acquisition
        recorded = build_stack_signal(
            aligned, members, model).metadata.get_item("Acquisition.multiangle")
        assert recorded["n_members"] == N_MEMBERS
        assert recorded["n_shells"] == 2
        assert recorded["tilts"] == [1.0, 1.0, 0.5, 0.5]

    def test_it_carries_a_recipe_with_the_angle_axis(self, acquisition):
        aligned, members, model = acquisition
        recipe = recipe_for(build_stack_signal(aligned, members, model))
        assert recipe is not None and recipe.has_angle_axis
        assert recipe.n_members == N_MEMBERS


class TestSummedSignal:
    def test_the_angle_axis_is_gone(self, acquisition):
        aligned, members, model = acquisition
        summed = build_summed_signal(aligned, members, model)
        assert summed.axes_manager.navigation_dimension == 2
        assert summed.axes_manager.signal_dimension == 2

    def test_it_sums_without_overflowing(self, acquisition):
        aligned, members, model = acquisition
        summed = build_summed_signal(aligned, members, model)
        assert summed.data.dtype == np.uint32

    def test_it_equals_the_stack_summed_over_angle(self, acquisition):
        aligned, members, model = acquisition
        stack = build_stack_signal(aligned, members, model)
        summed = build_summed_signal(aligned, members, model)
        assert np.array_equal(
            summed.data.compute(),
            stack.data.sum(axis=0, dtype=np.uint32).compute())

    def test_a_shell_sums_only_its_own_members(self, acquisition):
        aligned, members, model = acquisition
        shell_members = model.shells[1]
        shell = build_summed_signal(
            aligned, members, model, member_indices=shell_members,
            title="Summed 0.5°")
        expected = sum(aligned[index].astype(np.uint32)
                       for index in shell_members)
        assert np.array_equal(shell.data.compute(), expected.compute())
        assert recipe_for(shell).n_members == len(shell_members)

    def test_shell_titles_name_their_tilt(self, acquisition):
        _aligned, _members, model = acquisition
        assert set(shell_titles(model).values()) == {"Summed 1°", "Summed 0.5°"}


class TestCalibration:
    """The crop moves both origins; forgetting that is silent and corrupting."""

    def test_the_scan_origin_moves_by_the_crop(self, acquisition):
        aligned, members, model = acquisition
        summed = build_summed_signal(aligned, members, model)
        rows, columns = summed.axes_manager._axes[:2]
        assert rows.offset == SCAN_OFFSET + model.overlap_origin[0] * SCAN_SCALE
        assert columns.offset == (
            SCAN_OFFSET + model.overlap_origin[1] * SCAN_SCALE)

    def test_the_detector_origin_moves_by_the_crop(self, acquisition):
        aligned, members, model = acquisition
        summed = build_summed_signal(aligned, members, model)
        ky, kx = summed.axes_manager._axes[2:]
        assert ky.offset == pytest.approx(
            DETECTOR_OFFSET + model.detector_origin[0] * DETECTOR_SCALE)
        assert kx.offset == pytest.approx(
            DETECTOR_OFFSET + model.detector_origin[1] * DETECTOR_SCALE)

    def test_scales_and_units_survive(self, acquisition):
        aligned, members, model = acquisition
        for signal in (build_stack_signal(aligned, members, model),
                       build_summed_signal(aligned, members, model)):
            scan_and_detector = signal.axes_manager._axes[-4:]
            assert [axis.units for axis in scan_and_detector] == [
                "nm", "nm", "1/A", "1/A"]
            assert [axis.scale for axis in scan_and_detector] == [
                SCAN_SCALE, SCAN_SCALE, DETECTOR_SCALE, DETECTOR_SCALE]

    def test_both_nodes_share_one_calibration(self, acquisition):
        """A node switch must not move the scan under the user."""
        aligned, members, model = acquisition
        stack = build_stack_signal(aligned, members, model)
        summed = build_summed_signal(aligned, members, model)
        for stack_axis, summed_axis in zip(
                stack.axes_manager._axes[1:], summed.axes_manager._axes):
            assert stack_axis.scale == summed_axis.scale
            assert stack_axis.offset == summed_axis.offset


class TestCompositeNavigator:
    def test_it_is_the_members_navigators_on_the_common_grid(self, acquisition):
        _aligned, _members, model = acquisition
        generator = np.random.default_rng(5)
        navigators = [generator.random(SCAN) for _ in range(N_MEMBERS)]
        composed = composite_navigator(navigators, model)

        expected = np.zeros(model.overlap_shape(SCAN))
        for member_index, navigator in enumerate(navigators):
            rows, columns = model.nav_slices(member_index, SCAN)
            expected += navigator[rows, columns]
        assert np.allclose(composed, expected)

    def test_its_shape_is_the_common_region(self, acquisition):
        _aligned, _members, model = acquisition
        navigators = [np.ones(SCAN) for _ in range(N_MEMBERS)]
        assert composite_navigator(navigators, model).shape == \
            model.overlap_shape(SCAN)

    def test_a_navigator_count_mismatch_is_refused(self, acquisition):
        _aligned, _members, model = acquisition
        with pytest.raises(ValueError, match="describes"):
            composite_navigator([np.ones(SCAN)], model)

    def test_navigators_of_differing_shape_are_refused(self, acquisition):
        _aligned, _members, model = acquisition
        navigators = [np.ones(SCAN)] * (N_MEMBERS - 1) + [np.ones((4, 4))]
        with pytest.raises(ValueError, match="scan shape"):
            composite_navigator(navigators, model)


class TestMemberNavigatorPlanes:
    """The 5-D root's navigator must vary along the angle axis, or the angle
    navigator is a flat line with nothing to navigate."""

    def test_each_plane_is_that_member_on_the_common_grid(self, acquisition):
        from spyde.multiangle.signals import member_navigator_planes

        _aligned, _members, model = acquisition
        generator = np.random.default_rng(3)
        navigators = [generator.random(SCAN) for _ in range(N_MEMBERS)]
        planes = member_navigator_planes(navigators, model)
        assert planes.shape == (N_MEMBERS,) + model.overlap_shape(SCAN)
        for member_index, navigator in enumerate(navigators):
            rows, columns = model.nav_slices(member_index, SCAN)
            assert np.allclose(planes[member_index], navigator[rows, columns])

    def test_the_angle_profile_is_not_constant(self, acquisition):
        """The defect this replaced: identical planes reduce to a flat line."""
        from spyde.multiangle.signals import member_navigator_planes

        _aligned, _members, model = acquisition
        generator = np.random.default_rng(4)
        navigators = [generator.random(SCAN) * (index + 1)
                      for index in range(N_MEMBERS)]
        profile = member_navigator_planes(navigators, model).sum(axis=(1, 2))
        assert profile.std() > 0

    def test_the_planes_sum_to_the_composite(self, acquisition):
        """Two views of one thing: summing the angle axis is the composite."""
        from spyde.multiangle.signals import member_navigator_planes

        _aligned, _members, model = acquisition
        generator = np.random.default_rng(5)
        navigators = [generator.random(SCAN) for _ in range(N_MEMBERS)]
        assert np.allclose(
            member_navigator_planes(navigators, model).sum(axis=0),
            composite_navigator(navigators, model))


class TestAxisNames:
    """A composed axis is named for what it IS.

    A member's own names come from whichever reader opened it: an MRC with no
    sidecar reports its scan axes as '' and 'z'. Propagating that puts a scan
    axis called 'z' in front of the user when the role is known for certain.
    """

    def test_the_roles_are_stated_not_inherited(self, acquisition):
        aligned, members, model = acquisition
        for axis in members[model.reference].axes_manager._axes:
            axis.name = "z"
        summed = build_summed_signal(aligned, members, model)
        assert [axis.name for axis in summed.axes_manager._axes] == [
            "y", "x", "ky", "kx"]

    def test_the_stack_names_its_angle_axis_too(self, acquisition):
        aligned, members, model = acquisition
        stack = build_stack_signal(aligned, members, model)
        assert [axis.name for axis in stack.axes_manager._axes] == [
            ANGLE_AXIS_NAME, "y", "x", "ky", "kx"]

    def test_physical_calibration_still_comes_from_the_member(self, acquisition):
        """Only the NAME is stated — scale, offset and units carry real
        information and must not be invented."""
        aligned, members, model = acquisition
        summed = build_summed_signal(aligned, members, model)
        assert [axis.units for axis in summed.axes_manager._axes] == [
            "nm", "nm", "1/A", "1/A"]
        assert summed.axes_manager._axes[0].scale == SCAN_SCALE
        assert summed.axes_manager._axes[2].scale == DETECTOR_SCALE


class TestTheMembersMetadataIsCarried:
    """What the reference member knew, the composition has to know too.

    This was silently empty: the carry assigned over ``signal.metadata``, which
    is a read-only property, and the bare ``except`` around it turned the
    AttributeError into "no metadata at all". Nothing failed — the acquisition
    opened, looked right, and had no signal type, so every diffraction action
    was gated off it and the dataset could not be analysed.
    """

    def _reference(self, members):
        reference = members[0]
        reference.set_signal_type("electron_diffraction")
        reference.metadata.set_item(
            "Acquisition_instrument.TEM.beam_energy", 200.0)
        reference.metadata.set_item("Preprocessing.note", "carried")
        return reference

    def test_the_summed_node_keeps_the_signal_type(self, acquisition):
        aligned, members, model = acquisition
        self._reference(members)
        summed = build_summed_signal(aligned, members, model)
        assert summed.metadata.Signal.signal_type == "electron_diffraction"

    def test_the_stack_keeps_the_signal_type(self, acquisition):
        aligned, members, model = acquisition
        self._reference(members)
        stack = build_stack_signal(aligned, members, model)
        assert stack.metadata.Signal.signal_type == "electron_diffraction"

    def test_the_instrument_and_provenance_come_across(self, acquisition):
        aligned, members, model = acquisition
        self._reference(members)
        summed = build_summed_signal(aligned, members, model)
        assert summed.metadata.get_item(
            "Acquisition_instrument.TEM.beam_energy") == 200.0
        assert summed.metadata.get_item("Preprocessing.note") == "carried"

    def test_the_title_is_still_the_compositions_own(self, acquisition):
        aligned, members, model = acquisition
        reference = self._reference(members)
        reference.metadata.set_item("General.title", "one member")
        summed = build_summed_signal(aligned, members, model)
        assert summed.metadata.General.title == "Summed"

    def test_the_acquisition_record_survives_the_carry(self, acquisition):
        aligned, members, model = acquisition
        self._reference(members)
        summed = build_summed_signal(aligned, members, model)
        record = summed.metadata.get_item("Acquisition.multiangle")
        assert record["n_members"] == N_MEMBERS
        assert record["n_shells"] == 2

"""
Multi-Angle 4-D STEM: the acquisition assembled into ONE signal tree and opened.

``Session.open_multiangle`` turns N member files into::

    Aligned Stack   (angle, y, x | ky, kx)   the ROOT
    └── Summed      (y, x | ky, kx)          what the window shows first
        ├── Summed 0.5°
        └── Summed 1°

Three things about that are silent when wrong, so each gets its own class.

The **displayed node** is the summed one, not the root the tree was built from —
a user opening a multi-angle acquisition wants the sum, and the stack is the
step before it. The **frames** must stay right across a node switch, because a
5-D root and a 4-D child are navigated by indices of different length and a
mismatch paints a plausible-looking wrong frame rather than raising. And the
**navigator** must come from the members' own images: the composed array would
answer too, by reading every member in full to draw a thumbnail.

Expected pixel values here are computed by applying the model's offsets to the
raw synthetic members by hand, so the loader is never checked against its own
composition.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import dask.array as da
import numpy as np
import hyperspy.api as hs
import pytest

from spyde.backend._session_multiangle import NAV_CHUNK
from spyde.drawing.selectors.base_selector import _nav_dispatcher
from spyde.multiangle import make_multiangle
from spyde.multiangle.recipe import recipe_for
from spyde.multiangle.signals import composite_navigator, shell_titles
from spyde.tests.migrated._async import quiesce, wait_until, why_busy

SCAN = (24, 28)
DETECTOR = (24, 24)

# The navigation position every frame assertion is made at.
ANGLE_INDEX, SCAN_ROW, SCAN_COLUMN = 1, 5, 7


def _wait(pred, timeout=30.0):
    return wait_until(pred, timeout)


def _members_on_disk(directory, data):
    """The synthetic members written out, so the loader opens real files."""
    paths = []
    for index, member in enumerate(data.members):
        signal = hs.signals.Signal2D(member)
        signal.set_signal_type("electron_diffraction")
        path = str(directory / f"member{index:02d}.hspy")
        signal.save(path)
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def two_shells(tmp_path_factory):
    """Five members at two tilts — unequal shells, which is the general case."""
    data = make_multiangle(shells=((1.0, 3), (0.5, 2)), scan_shape=SCAN,
                           detector_shape=DETECTOR, seed=3)
    return data, _members_on_disk(tmp_path_factory.mktemp("two-shells"), data)


@pytest.fixture(scope="module")
def one_shell(tmp_path_factory):
    """Three members at ONE tilt — the acquisition that needs no shell nodes."""
    data = make_multiangle(shells=((1.0, 3),), scan_shape=SCAN,
                           detector_shape=DETECTOR, seed=5)
    return data, _members_on_disk(tmp_path_factory.mktemp("one-shell"), data)


@pytest.fixture(scope="module")
def frame_chunked(tmp_path_factory):
    """Three members in stores compressed FRAME BY FRAME, on a scan longer than
    one navigation chunk — the layout where the load-time alignment shows up in
    the composed chunk grid."""
    data = make_multiangle(shells=((1.0, 3),), scan_shape=(40, 44),
                           detector_shape=(16, 16), seed=7)
    directory = tmp_path_factory.mktemp("frame-chunked")
    paths = []
    for index, member in enumerate(data.members):
        signal = hs.signals.Signal2D(member)
        path = str(directory / f"member{index:02d}.zspy")
        signal.save(path, chunks=(1, 1, -1, -1))
        paths.append(path)
    return data, paths


def _signal_plot(session):
    """The diffraction-pattern window — the tree's only non-navigator plot."""
    return next(plot for plot in session._plots if not plot.is_navigator)


def _settle(session) -> None:
    """Wait for the session AND the navigator dispatcher to have nothing left.

    Idle has to hold TWICE for the dispatcher: a chained update (angle → real
    space → diffraction pattern) is submitted from inside the update it
    follows, so a single sample can land in the gap between the two.
    """
    assert quiesce(session), why_busy(session)

    def _idle_twice():
        if not _nav_dispatcher.idle():
            return False
        time.sleep(0.05)
        return _nav_dispatcher.idle()

    assert _wait(_idle_twice), "the navigator dispatcher never went idle"


def _navigation_selectors(tree):
    """``(angle_selector, scan_selector)`` — the chain, outermost first."""
    by_name = {type(selector).__name__: selector
               for selector in tree.navigator_plot_manager.all_navigation_selectors}
    return by_name["IntegratingSelector1D"], by_name["IntegratingSSelector2D"]


def _pin_navigators(tree, angle=ANGLE_INDEX, row=SCAN_ROW, column=SCAN_COLUMN):
    """Hold both navigators at a fixed position, so the expected frame is a
    known array rather than wherever the widgets happen to sit."""
    angle_selector, scan_selector = _navigation_selectors(tree)
    angle_selector.selector._get_selected_indices = \
        lambda: np.array([[int(angle)]])
    # A 2-D selector reports widget order (column, row).
    scan_selector.selector._get_selected_indices = \
        lambda: np.array([[int(column), int(row)]])
    scan_selector.selector._run_update(force=True)


def _open(session, acquisition):
    """Open the acquisition and wait until the Summed node is on screen."""
    data, paths = acquisition
    session.open_multiangle(paths, data.tilts.tolist(), data.azimuths.tolist(),
                            reference=data.reference)
    assert _wait(lambda: bool(session.signal_trees)
                 and "Summed" in session.signal_trees[0].root_node.children), \
        "the multi-angle tree never appeared"
    tree = session.signal_trees[0]
    summed = tree.root_node.children["Summed"].signal
    plot = _signal_plot(session)
    assert _wait(lambda: plot.plot_state is not None
                 and plot.plot_state.current_signal is summed), \
        "the Summed node never became the displayed one"
    _pin_navigators(tree)
    _settle(session)
    return tree, summed, plot


def _member_frame(member, model, member_index, row=SCAN_ROW, column=SCAN_COLUMN):
    """One RAW member's contribution to a composed frame, aligned by hand.

    Applying the offsets here rather than through ``compose`` keeps the expected
    value independent of the code that produced the node.
    """
    rows, columns = model.nav_slices(member_index, member.shape[:2])
    detector_rows, detector_columns = model.detector_slices(
        member_index, member.shape[2:])
    frame = member[rows.start + int(row), columns.start + int(column)]
    return frame[detector_rows, detector_columns]


def _summed_frame(data, model, member_indices=None):
    """The composed frame those members add up to, in the composed dtype."""
    if member_indices is None:
        member_indices = range(model.n_members)
    total = None
    for member_index in member_indices:
        frame = _member_frame(
            data.members[member_index], model, member_index).astype(np.uint32)
        total = frame if total is None else total + frame
    return total


class TestTheTreeShape:
    def test_the_root_is_the_aligned_stack(self, window, two_shells):
        """Summing is the transformation, so the stack is the parent — and the
        Workflow panel names it rather than calling it "root"."""
        data, _paths = two_shells
        tree, _summed, _plot = _open(window["window"], two_shells)

        assert tree.root_node.name == "Aligned Stack"
        assert tree.root.axes_manager.navigation_dimension == 3
        assert tree.root.data.shape[0] == data.n_members
        assert tree.root.axes_manager._axes[0].name == "angle"

    def test_the_summed_node_is_the_stacks_only_child(self, window, two_shells):
        tree, summed, _plot = _open(window["window"], two_shells)

        assert list(tree.root_node.children) == ["Summed"]
        assert summed.axes_manager.navigation_dimension == 2
        assert summed.axes_manager.signal_dimension == 2
        assert summed.data.shape == tree.root.data.shape[1:]

    def test_every_composed_node_is_tagged_local(self, window, two_shells):
        """A composed frame is one frame from each member added, which is what
        the locality gate asks — and what lets the multi-angle reader serve it
        instead of the composed dask graph."""
        tree, _summed, _plot = _open(window["window"], two_shells)

        composed = [node for node in tree.walk() if node.parent is not None]
        assert composed
        assert all(node.local is True for node in composed)
        assert all(tree.resolve_locality(node.signal) for node in composed)

    def test_the_nodes_stay_lazy(self, window, two_shells):
        tree, summed, _plot = _open(window["window"], two_shells)

        assert tree.root._lazy and summed._lazy
        assert summed.data.dtype == np.uint32     # uint16 x 5 cannot overflow it

    def test_per_shell_nodes_hang_off_the_summed_node(self, window, two_shells):
        """One child per shell, named for the tilt that shell was taken at."""
        tree, summed, _plot = _open(window["window"], two_shells)
        model = recipe_for(summed).model

        shell_nodes = tree.root_node.children["Summed"].children
        assert set(shell_nodes) == set(shell_titles(model).values())
        assert set(shell_nodes) == {"Summed 0.5°", "Summed 1°"}

    def test_a_single_shell_acquisition_gets_no_shell_children(self, window,
                                                               one_shell):
        """Its one shell IS the summed node; a child duplicating its parent is
        a node the user cannot tell apart from it."""
        tree, summed, _plot = _open(window["window"], one_shell)

        assert recipe_for(summed).model.n_shells == 1
        assert tree.root_node.children["Summed"].children == {}

    def test_a_shell_node_sums_only_its_own_members(self, window, two_shells):
        data, _paths = two_shells
        tree, summed, _plot = _open(window["window"], two_shells)
        model = recipe_for(summed).model

        for shell_id, member_indices in model.shells.items():
            node = tree.root_node.children["Summed"].children[
                shell_titles(model)[shell_id]]
            assert recipe_for(node.signal).n_members == len(member_indices)
            assert np.array_equal(
                np.asarray(node.signal.data[SCAN_ROW, SCAN_COLUMN]),
                _summed_frame(data, model, member_indices))

    def test_the_offsets_the_loader_solved_are_the_planted_ones(self, window,
                                                                two_shells):
        """The alignment has its own suite; this only pins that the loader
        feeds it the members' images the right way round."""
        data, _paths = two_shells
        _tree, summed, _plot = _open(window["window"], two_shells)
        model = recipe_for(summed).model

        assert np.array_equal(model.nav_offsets, data.nav_offsets)
        assert np.array_equal(model.dp_offsets, data.dp_offsets)


class TestWhatTheWindowDisplays:
    def test_the_summed_node_opens_displayed(self, window, two_shells):
        data, _paths = two_shells
        _tree, summed, plot = _open(window["window"], two_shells)
        model = recipe_for(summed).model

        assert plot.plot_state.current_signal is summed
        assert plot.current_data.shape == model.detector_shape(DETECTOR)
        assert np.array_equal(plot.current_data, _summed_frame(data, model))

    def test_switching_to_the_stack_reads_that_members_frame(self, window,
                                                             two_shells):
        data, _paths = two_shells
        session = window["window"]
        tree, summed, plot = _open(session, two_shells)
        model = recipe_for(summed).model

        session._select_signal_node(plot, id(tree.root))
        _settle(session)

        assert plot.plot_state.current_signal is tree.root
        assert np.array_equal(
            plot.current_data,
            _member_frame(data.members[ANGLE_INDEX], model, ANGLE_INDEX))

    def test_a_shell_node_displays_its_own_members_frame(self, window,
                                                         two_shells):
        data, _paths = two_shells
        session = window["window"]
        tree, summed, plot = _open(session, two_shells)
        model = recipe_for(summed).model

        # The shell that does NOT start at member 0 — a prefix shell happens to
        # re-index onto itself and would pass whatever the reader does.
        shell_id, member_indices = next(
            (shell_id, indices) for shell_id, indices in model.shells.items()
            if indices[0] != 0)
        shell = tree.root_node.children["Summed"].children[
            shell_titles(model)[shell_id]].signal

        session._select_signal_node(plot, id(shell))
        _settle(session)

        assert plot.plot_state.current_signal is shell
        assert np.array_equal(plot.current_data,
                              _summed_frame(data, model, member_indices))

    def test_switching_back_restores_the_summed_frame(self, window, two_shells):
        """The round trip is clean — the composed index the navigator chain
        builds still reads the node the window is on."""
        data, _paths = two_shells
        session = window["window"]
        tree, summed, plot = _open(session, two_shells)
        model = recipe_for(summed).model

        session._select_signal_node(plot, id(tree.root))
        _settle(session)
        session._select_signal_node(plot, id(summed))
        _settle(session)

        assert plot.plot_state.current_signal is summed
        # Waited for, not asserted after a fixed settle: the switch paints
        # from the navigator dispatcher, and on a loaded Windows runner the
        # frame landed after the settle twice.
        expected = _summed_frame(data, model)
        if not wait_until(
                lambda: plot.current_data is not None
                and np.array_equal(plot.current_data, expected), timeout=20.0):
            # Say what the window holds instead: one member's frame means the
            # root's paint landed after the switch back; the summed frame at
            # another position means the selector moved; None means nothing
            # painted at all. Each is a different bug.
            shown = plot.current_data
            members = [
                member_index for member_index in range(model.n_members)
                if shown is not None and np.array_equal(
                    shown, _member_frame(data.members[member_index], model,
                                         member_index))]
            raise AssertionError(
                "the summed frame never came back after switching away and "
                f"back: the window shows {'nothing' if shown is None else shown.shape}"
                f"{' — member ' + str(members) + chr(39) + 's own frame' if members else ''}; "
                f"current_signal is summed: {plot.plot_state.current_signal is summed}")



class TestTheNavigator:
    def test_each_angle_plane_is_that_member(self, window, two_shells):
        """The navigator costs nothing — it is the members' own real-space
        images on the common grid, not a reduction of the composed 5-D array,
        which would read every member in full.

        One plane PER MEMBER, not one image repeated: repeating it makes the
        1-D angle line a flat constant, which is a navigator with nothing to
        navigate. That was the original defect and it was only visible on
        screen, so the profile check below is the guard.
        """
        data, _paths = two_shells
        tree, summed, _plot = _open(window["window"], two_shells)
        model = recipe_for(summed).model

        images = [member.sum(axis=(-2, -1), dtype=np.float64)
                  for member in data.members]

        _angle_navigator, planes = tree.navigator_signals["base"]
        assert planes.data.shape[0] == model.n_members
        for angle in range(model.n_members):
            rows, columns = model.nav_slices(angle, SCAN)
            assert np.allclose(planes.data[angle], images[angle][rows, columns])

        # The 1-D angle navigator is these planes reduced; a flat one cannot be
        # dragged along, which is exactly what the user sees when it breaks.
        profile = planes.data.sum(axis=(1, 2))
        assert float(profile.std()) > 0.0

    def test_the_planes_sum_to_the_composite(self, window, two_shells):
        """The composed overview is still recoverable — it is the angle axis
        summed away, the same relationship the data itself has."""
        data, _paths = two_shells
        tree, summed, _plot = _open(window["window"], two_shells)
        model = recipe_for(summed).model

        images = [member.sum(axis=(-2, -1), dtype=np.float64)
                  for member in data.members]
        _angle_navigator, planes = tree.navigator_signals["base"]
        assert np.allclose(planes.data.sum(axis=0),
                           composite_navigator(images, model))


class TestTheMembersAreAlignedAsTheyAreRead:
    def test_the_composed_grid_carries_no_slivers(self, window, frame_chunked):
        """A member compressed frame by frame is rebuilt on a grid whose blocks
        line up with the other members' after cropping, so the composed array
        needs no rechunk. Cropping every member on a UNIFORM grid instead knocks
        the boundaries out of step by exactly the offsets, and the composition
        then unifies to the union of all of them — chunks one or two scan
        positions wide, which is the state this path exists to avoid.
        """
        _tree, summed, _plot = _open(window["window"], frame_chunked)

        for axis_chunks in summed.data.chunks[:2]:
            assert all(size == NAV_CHUNK for size in axis_chunks[:-1]), \
                f"the composed navigation grid was fragmented: {axis_chunks}"


class TestNothingIsMaterialised:
    def test_opening_never_computes_a_composed_array(self, window, two_shells):
        """The CLAUDE.md memory-safety rule, multiplied by N: a ``.compute()``
        on a composed array pulls a navigation chunk out of every member at
        once. Only the two small per-member reductions the alignment needs may
        be computed.
        """
        data, _paths = two_shells
        session = window["window"]
        computed_shapes = []
        compute_method, compute_function = da.Array.compute, da.compute

        def record_method(self, *args, **kwargs):
            computed_shapes.append(tuple(self.shape))
            return compute_method(self, *args, **kwargs)

        def record_function(*arrays, **kwargs):
            computed_shapes.extend(
                tuple(getattr(array, "shape", ())) for array in arrays)
            return compute_function(*arrays, **kwargs)

        with patch.object(da.Array, "compute", record_method), \
                patch.object(da, "compute", record_function):
            tree, summed, _plot = _open(session, two_shells)

        forbidden = {tuple(tree.root.data.shape), tuple(summed.data.shape),
                     tuple(data.members[0].shape)}
        assert not forbidden & set(computed_shapes), (
            f"the loader computed {forbidden & set(computed_shapes)}")


class TestRefusals:
    def _errors(self, messages):
        return [m["text"] for m in messages if m.get("type") == "error"]

    def test_one_member_is_not_an_acquisition(self, window, two_shells):
        data, paths = two_shells
        session = window["window"]

        session.open_multiangle(paths[:1], data.tilts[:1].tolist(),
                                data.azimuths[:1].tolist())

        assert any("at least two" in text
                   for text in self._errors(window["messages"]))
        assert not session.signal_trees

    def test_every_member_needs_a_tilt_and_an_azimuth(self, window, two_shells):
        """The shells are grouped by tilt, so an acquisition missing one is not
        one — refused before anything is read."""
        data, paths = two_shells
        session = window["window"]

        session.open_multiangle(paths, data.tilts[:-1].tolist(),
                                data.azimuths.tolist())

        assert any("tilt" in text for text in self._errors(window["messages"]))
        assert not session.signal_trees

    def test_a_reference_outside_the_members_is_refused(self, window, two_shells):
        data, paths = two_shells
        session = window["window"]

        session.open_multiangle(paths, data.tilts.tolist(),
                                data.azimuths.tolist(), reference=99)

        assert any("Reference member 99" in text
                   for text in self._errors(window["messages"]))
        assert not session.signal_trees


class TestTheRendererCanTriggerIt:
    def test_the_action_dispatch_opens_the_acquisition(self, window, two_shells):
        data, paths = two_shells
        session = window["window"]

        session.dispatch_action({
            "action": "open_multiangle",
            "payload": {"paths": paths, "tilts": data.tilts.tolist(),
                        "azimuths": data.azimuths.tolist(),
                        "reference": data.reference},
        })

        assert _wait(lambda: bool(session.signal_trees)
                     and "Summed" in session.signal_trees[0].root_node.children), \
            "open_multiangle never reached the loader"
        _settle(session)

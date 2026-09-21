"""
The STAGED multi-angle loader: Load datasets → Align real space → Align
reciprocal space → Commit.

``Session.open_multiangle`` does all four at once. This is the same job split
so the user can see and adjust each stage, and the things that can go wrong are
the seams between them, not the alignment — which has its own suite
(``test_multiangle_align.py``) and is not re-checked here.

Four of those seams are silent when they break, so each gets its own class.

**Probing must not read data.** The dialog is used by adding and removing files
repeatedly; a probe that computed anything would make that unusable on real
members, and on a small fixture it would look fine. A guard on ``compute``
itself is the only way to see it.

**One bad file must not sink the others.** A multi-select of ten with one typo
is the normal case, and "nothing opened" hides which one.

**The per-member reductions are computed once.** They are the expensive part of
this loader on real data, so re-running a solve with different parameters must
cost the solve alone.

**Commit must go through the SAME composition as the one-shot loader.** Two
doors onto one dataset that produce different trees is the bug this split
invites, so the test opens the same files both ways and compares.

Three more seams came with the tableaux, and they fail the same quiet way.

**The chosen virtual image must actually reach the solve.** A dropdown that is
read for the picture and ignored for the alignment looks exactly like one that
works, so one of the planted images is the same picture on every member: a
solve that really used it returns zeros, and a solve that used something else
cannot.

**A corner extent must re-measure that corner and nothing else.** The cache is
what makes the widget usable; without it every nudge re-reads all four corners
of every member, which is the full measurement the corners method exists to
avoid.

**A picture must not be re-encoded when nothing about it changed.** Every
action answers with the WHOLE snapshot, so a preview rebuilt per snapshot is a
PNG encode per click, per member.

**Both tableaux must be full before anything is solved.** A tableau is what the
user looks at to decide whether to run a stage; it cannot be the reward for
having run one. So the probe hands over to a pass that reads the members and
fills the grids — one member at a time, so the first tile lands early — and
everything afterwards reuses what it read. That pass now starts unprompted, so
it also has to stop on its own: closing the dialog mid-pass must not leave a
multi-gigabyte read running for a window that is gone.

**The real-space solve's window must not multiply or outlive the dialog.** It
is a bare figure window, so nothing closes it for free: three tries at the
alignment have to leave one window, a superseded solve none, and closing the
dialog must take it with it.

The members are real MRC files, written the way the test harness writes them
(``_write_minimal_mrc``), so the probe reads a real header and the loader takes
its real path rather than a numpy shortcut.
"""
from __future__ import annotations

import os
import threading
from unittest.mock import patch

import dask.array as da
import hyperspy.api as hs
import numpy as np
import pytest

from spyde.backend import _session_multiangle_loader as loader
from spyde.backend._session_multiangle_loader import (
    maped_add_files, maped_align_real, maped_align_reciprocal,
    maped_close_loader, maped_commit, maped_open_loader, maped_remove_member,
    maped_set_corner_extent, maped_set_member, maped_set_pair,
    maped_set_real_offset,
    maped_set_reference,
    maped_set_scan_shape, maped_set_virtual_image,
)
from spyde.backend._session_testharness import _write_minimal_mrc
from spyde.multiangle import make_multiangle
from spyde.tests.migrated._async import (
    call_on_loop,
    drain_loop, quiesce, wait_until, why_busy,
)
from spyde.tests.migrated.conftest import close_session, make_session

#: Big enough that phase correlation can register the synthetic scene — below
#: about this the real-space solve returns zeros for every member and the test
#: would be pinning the fixture's limits rather than the loader's behaviour.
SCAN = (24, 28)
DETECTOR = (16, 16)

#: MRC carries the detector shape but has no concept of a scan grid, so the
#: dialog's scan-shape field is what makes these members readable. The reader
#: takes the scan as ``(x, y)``, which is the order the field is in.
SCAN_XY = (SCAN[1], SCAN[0])


def _wait(pred, timeout=60.0):
    return wait_until(pred, timeout)


@pytest.fixture(scope="module")
def acquisition(tmp_path_factory):
    """Five members at two tilts, on disk as MRC, with planted offsets."""
    data = make_multiangle(shells=((1.0, 3), (0.5, 2)), scan_shape=SCAN,
                           detector_shape=DETECTOR, seed=3)
    directory = tmp_path_factory.mktemp("multiangle-loader")
    paths = []
    for index, member in enumerate(data.members):
        path = str(directory / f"member{index:02d}.mrc")
        _write_minimal_mrc(path, np.ascontiguousarray(member).reshape(
            -1, *member.shape[2:]))
        paths.append(path)
    return data, paths


def _last_state(messages) -> dict:
    """The most recent ``maped_state`` snapshot the loader emitted."""
    states = [m for m in messages
              if isinstance(m, dict) and m.get("type") == "maped_state"]
    assert states, "the loader emitted no maped_state at all"
    return states[-1]


def _probed(session):
    """Wait for the probe worker to fill the rows it appended."""
    assert _wait(lambda: session._multiangle_loader is not None
                 and not session._multiangle_loader.busy), \
        "the probe never finished"


def _add(session, paths, scan_shape=SCAN_XY):
    """Loaded the way the dialog loads: open, name the scan grid, add files."""
    maped_open_loader(session, None, {})
    if scan_shape is not None:
        maped_set_scan_shape(session, None, {"scan_shape": list(scan_shape)})
    maped_add_files(session, None, {"paths": list(paths)})
    _probed(session)


def _with_angles(session, acquisition):
    """Loaded, with each member's true tilt and azimuth set — the state a user
    reaches by typing them in."""
    data, paths = acquisition
    _add(session, paths)
    for index in range(len(paths)):
        maped_set_member(session, None, {
            "index": index,
            "tilt": float(data.tilts[index]),
            "azimuth": float(data.azimuths[index]),
        })


def _solve_both(session, messages, method="beam"):
    maped_align_real(session, None, {"params": {}})
    assert _wait(lambda: _last_state(messages)["real"]["solved"]), \
        "the real-space stage never solved"
    maped_align_reciprocal(session, None, {"method": method, "params": {}})
    assert _wait(lambda: _last_state(messages)["reciprocal"]["solved"]), \
        "the reciprocal stage never solved"


class TestProbing:
    def test_a_probe_reports_the_shape_dtype_and_size(self, window, acquisition):
        _data, paths = acquisition
        _add(window["window"], paths)

        state = _last_state(window["messages"])
        assert len(state["members"]) == len(paths)
        for index, member in enumerate(state["members"]):
            assert member["index"] == index
            assert member["path"] == paths[index]
            assert member["name"] == os.path.basename(paths[index])
            assert member["scan_shape"] == list(SCAN)
            assert member["detector_shape"] == list(DETECTOR)
            assert member["dtype"] == "uint16"
            assert member["size_bytes"] > 0
            assert member["error"] is None

    def test_a_probe_reads_no_frames(self, window, acquisition):
        """A lazy open is a header read. Anything that computes here would
        read the acquisition every time a file is added or removed.

        The tableau fill the probe hands over to DOES read — that is its whole
        job (see :class:`TestTheThumbnails`) — so it is held off here. Without
        that, this guard would fire inside the FILL's worker, where a failure
        is reported rather than raised, and the test would pass whether or not
        the probe itself had read anything.
        """
        _data, paths = acquisition

        def _refuse(*args, **kwargs):
            raise AssertionError("probing computed a dask array — it must "
                                 "read headers only")

        def _no_fill(session, state):
            """The fill minus the reading: it still ends the ``busy`` phase
            the probe hands to it, which is what ``_probed`` waits for."""
            state.busy = False
            loader._emit_state(state)

        with patch.object(loader, "_start_preview_fill", _no_fill), \
                patch.object(da.Array, "compute", _refuse), \
                patch.object(da, "compute", _refuse):
            _add(window["window"], paths)

        assert len(_last_state(window["messages"])["members"]) == len(paths)

    def test_a_bad_path_is_one_bad_row(self, window, acquisition):
        """The user needs to see WHICH file, with the rest of the list intact."""
        _data, paths = acquisition
        _add(window["window"], list(paths) + ["/no/such/member.mrc"])

        members = _last_state(window["messages"])["members"]
        assert len(members) == len(paths) + 1
        assert all(member["error"] is None for member in members[:-1])
        assert members[-1]["error"]
        assert members[-1]["name"] == "member.mrc"
        assert members[-1]["scan_shape"] is None

    def test_adding_files_says_it_is_busy_before_it_says_anything_else(
            self, window, acquisition):
        """The dialog's controls move only when this echoes, so an add that
        stayed quiet until the probe landed would read as a frozen dialog."""
        _data, paths = acquisition
        session = window["window"]
        maped_open_loader(session, None, {})
        maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})
        before = len(window["messages"])

        maped_add_files(session, None, {"paths": list(paths)})
        first = [m for m in window["messages"][before:]
                 if isinstance(m, dict) and m.get("type") == "maped_state"][0]
        assert first["busy"] is True
        assert len(first["members"]) == len(paths)
        assert first["members"][0]["scan_shape"] is None    # not probed yet

        _probed(session)
        assert _last_state(window["messages"])["busy"] is False

    def test_a_file_with_no_scan_grid_says_what_is_missing(self, window,
                                                           acquisition):
        """MRC records the frames and the detector but has no concept of a
        scan grid, so opening one without a scan shape is the error a user
        actually hits — and "3-D, expected 4-D" does not tell them what to do
        about it."""
        _data, paths = acquisition
        session = window["window"]
        maped_open_loader(session, None, {})           # no reader options
        maped_add_files(session, None, {"paths": paths[:1]})
        _probed(session)

        member = _last_state(window["messages"])["members"][0]
        assert "scan shape" in member["error"]
        assert member["detector_shape"] == list(DETECTOR)

    def test_an_errored_member_is_in_no_shell(self, window, acquisition):
        """A file that did not open has no tilt anyone has vouched for, so it
        is in no shell and counts towards no shell's membership."""
        _data, paths = acquisition
        _add(window["window"], list(paths) + ["/no/such/member.mrc"])

        state = _last_state(window["messages"])
        assert state["members"][-1]["shell"] is None
        assert all(len(paths) not in shell["members"]
                   for shell in state["shells"])

    def test_a_bad_row_blocks_aligning_until_it_is_removed(self, window,
                                                           acquisition):
        session = window["window"]
        _data, paths = acquisition
        _add(session, list(paths) + ["/no/such/member.mrc"])

        maped_align_real(session, None, {"params": {}})
        assert quiesce(session), why_busy(session)
        assert not _last_state(window["messages"])["real"]["solved"]

        maped_remove_member(session, None, {"index": len(paths)})
        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])


class TestTheScanShape:
    """A raw binary member records frames and a detector but no scan grid.

    The reader accepts any grid it is asked for — including one that does not
    fit the file — so every guard here is about not letting a typed number turn
    into a plausible wrong dataset.
    """

    def test_a_scan_shape_rescues_the_members_that_need_one(self, window,
                                                            acquisition):
        session = window["window"]
        _data, paths = acquisition
        _add(session, paths, scan_shape=None)

        state = _last_state(window["messages"])
        assert state["scan_shape"] is None
        assert all("scan shape" in member["error"]
                   for member in state["members"])

        maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})
        _probed(session)

        state = _last_state(window["messages"])
        assert state["scan_shape"] == list(SCAN_XY)
        assert all(member["error"] is None for member in state["members"])
        assert all(member["scan_shape"] == list(SCAN)
                   for member in state["members"])

    def test_clearing_it_puts_the_members_back(self, window, acquisition):
        """A shape entered wrongly has to be undoable without rebuilding the
        list."""
        session = window["window"]
        _data, paths = acquisition
        _add(session, paths)
        assert all(member["error"] is None
                   for member in _last_state(window["messages"])["members"])

        maped_set_scan_shape(session, None, {"scan_shape": None})
        _probed(session)

        state = _last_state(window["messages"])
        assert state["scan_shape"] is None
        assert all("scan shape" in member["error"]
                   for member in state["members"])

    def test_a_scan_shape_that_does_not_fit_the_file_is_refused(
            self, window, acquisition):
        """The reader does NOT check this: asked for a grid that does not fit,
        it returns that many positions anyway — reading the wrong frames, or
        past the end of the file."""
        session = window["window"]
        _data, paths = acquisition
        _add(session, paths, scan_shape=(10, 10))

        member = _last_state(window["messages"])["members"][0]
        assert "100 positions" in member["error"]
        assert str(SCAN[0] * SCAN[1]) in member["error"]
        assert member["scan_shape"] != [10, 10]

    def test_a_member_that_records_its_own_scan_grid_is_not_overridden(
            self, window, acquisition, tmp_path):
        """A Direct Electron ``.mrc`` and its ``_info.txt`` sidecar is the
        normal case, and the dialog's field is a fallback — so a file that
        already knows its scan grid must survive a wrong number in that
        field."""
        data, paths = acquisition
        self_describing = str(tmp_path / "self_describing.hspy")
        hs.signals.Signal2D(data.members[0]).save(self_describing)

        session = window["window"]
        _add(session, [self_describing] + list(paths), scan_shape=(10, 10))

        members = _last_state(window["messages"])["members"]
        assert members[0]["scan_shape"] == list(SCAN)
        assert members[0]["error"] is None
        assert all(member["error"] for member in members[1:])

    def test_the_field_stays_empty_when_nothing_needs_it(self, window,
                                                         acquisition, tmp_path):
        data, _paths = acquisition
        self_describing = []
        for index in range(2):
            path = str(tmp_path / f"self{index}.hspy")
            hs.signals.Signal2D(data.members[index]).save(path)
            self_describing.append(path)

        session = window["window"]
        _add(session, self_describing, scan_shape=None)
        maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})

        state = _last_state(window["messages"])
        assert "no member needs one" in state["message"]
        assert all(member["error"] is None for member in state["members"])
        assert all(member["scan_shape"] == list(SCAN)
                   for member in state["members"])

    def test_a_nonsense_scan_shape_is_refused_without_touching_the_members(
            self, window, acquisition):
        session = window["window"]
        _data, paths = acquisition
        _add(session, paths)

        maped_set_scan_shape(session, None, {"scan_shape": [0, 8]})
        state = _last_state(window["messages"])
        assert state["scan_shape"] == list(SCAN_XY)
        assert all(member["error"] is None for member in state["members"])


class TestAngles:
    def test_azimuths_spread_evenly_and_tilts_stay_zero(self, window,
                                                        acquisition):
        """A tilt cannot be guessed from a file's place in a list; an azimuth
        can, because a shell is normally stepped evenly around the ring."""
        _data, paths = acquisition
        _add(window["window"], paths)

        members = _last_state(window["messages"])["members"]
        assert [member["tilt"] for member in members] == [0.0] * len(paths)
        assert [member["azimuth"] for member in members] == pytest.approx(
            [360.0 * i / len(paths) for i in range(len(paths))])

    def test_a_typed_azimuth_survives_the_next_file(self, window, acquisition):
        """Only the loader's own guesses are re-spread — a number someone
        chose is never moved under them."""
        session = window["window"]
        _data, paths = acquisition
        _add(session, paths[:2])
        maped_set_member(session, None, {"index": 0, "azimuth": 42.0})
        maped_add_files(session, None, {"paths": paths[2:]})
        _probed(session)

        members = _last_state(window["messages"])["members"]
        assert members[0]["azimuth"] == 42.0
        assert len({member["azimuth"] for member in members[1:]}) == len(paths) - 1

    def test_a_file_can_arrive_already_placed(self, window, acquisition):
        """A file dropped onto a spot on the polar tableau already has its
        angles, and this action hands back nothing addressable — so they come
        WITH the path rather than in a second call the caller has to aim by
        matching its own filename in a snapshot."""
        data, paths = acquisition
        session = window["window"]
        maped_open_loader(session, None, {})
        maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})

        maped_add_files(session, None, {
            "paths": list(paths),
            "angles": [{"tilt": float(data.tilts[i]),
                        "azimuth": float(data.azimuths[i])}
                       for i in range(len(paths))]})
        _probed(session)

        members = _last_state(window["messages"])["members"]
        assert [member["tilt"] for member in members] == \
            [pytest.approx(float(t)) for t in data.tilts]
        assert [member["azimuth"] for member in members] == \
            [pytest.approx(float(a)) for a in data.azimuths]

    def test_the_angles_are_on_the_very_first_snapshot(self, window,
                                                       acquisition):
        """Applied as the row is appended, so there is nothing to wait for and
        nothing to match on — the ``busy`` snapshot already has them."""
        data, paths = acquisition
        session = window["window"]
        maped_open_loader(session, None, {})
        maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})
        before = len(window["messages"])

        maped_add_files(session, None, {
            "paths": list(paths),
            "angles": [{"tilt": 7.0, "azimuth": 11.0}] * len(paths)})
        first = [m for m in window["messages"][before:]
                 if isinstance(m, dict) and m.get("type") == "maped_state"][0]

        assert first["busy"] is True
        assert all(member["tilt"] == 7.0 and member["azimuth"] == 11.0
                   for member in first["members"])
        _probed(session)

    def test_a_placed_member_is_not_re_spread_or_re_read(self, window,
                                                         acquisition):
        """The placement is the user's, exactly as a typed one is: neither the
        file's own metadata nor the loader's even spread moves it back."""
        _data, paths = acquisition
        session = window["window"]
        maped_open_loader(session, None, {})
        maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})

        maped_add_files(session, None, {
            "paths": list(paths),
            "angles": [{"azimuth": 5.0}] + [None] * (len(paths) - 1)})
        _probed(session)

        members = _last_state(window["messages"])["members"]
        assert members[0]["azimuth"] == 5.0
        assert members[0]["tilt"] == 0.0        # nothing said, nothing guessed
        assert len({member["azimuth"] for member in members[1:]}) == \
            len(paths) - 1, "the unplaced members were still spread evenly"

    def test_an_unequal_angle_list_is_refused_and_the_files_still_land(
            self, window, acquisition):
        """Angles go with paths BY POSITION, so an unequal list cannot say
        which is which — and dropping the files too would leave the user with
        neither the members nor an explanation."""
        _data, paths = acquisition
        session = window["window"]
        maped_open_loader(session, None, {})
        maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})
        before = len(window["messages"])

        maped_add_files(session, None, {"paths": list(paths),
                                        "angles": [{"tilt": 9.0}]})
        _probed(session)

        errors = [m for m in window["messages"][before:]
                  if isinstance(m, dict) and m.get("type") == "error"]
        assert errors and "angles" in str(errors[0].get("text"))
        members = _last_state(window["messages"])["members"]
        assert len(members) == len(paths)
        assert all(member["tilt"] == 0.0 for member in members)

    def test_shells_group_by_tilt(self, window, acquisition):
        data, _paths = acquisition
        _with_angles(window["window"], acquisition)

        state = _last_state(window["messages"])
        assert [shell["tilt"] for shell in state["shells"]] == [0.5, 1.0]
        assert [shell["members"] for shell in state["shells"]] == [[3, 4],
                                                                   [0, 1, 2]]
        assert [member["shell"] for member in state["members"]] == \
            [int(v) for v in data.shell_ids]


class TestTheSolves:
    def test_real_space_recovers_the_planted_offsets(self, window, acquisition):
        data, _paths = acquisition
        _with_angles(window["window"], acquisition)

        maped_align_real(window["window"], None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])

        real = _last_state(window["messages"])["real"]
        assert real["offsets"] == data.nav_offsets.tolist()
        assert real["max_residual"] < 0.5

    @pytest.mark.parametrize("method", ["beam", "correlate"])
    def test_reciprocal_recovers_the_planted_offsets(self, window, acquisition,
                                                     method):
        """Both methods must work: ``correlate`` is what is left when there is
        no sharp direct beam for ``beam`` to fit."""
        data, _paths = acquisition
        _with_angles(window["window"], acquisition)

        maped_align_reciprocal(window["window"], None,
                               {"method": method, "params": {}})
        assert _wait(
            lambda: _last_state(window["messages"])["reciprocal"]["solved"])

        assert _last_state(window["messages"])["reciprocal"]["offsets"] == \
            data.dp_offsets.tolist()

    def test_the_real_space_tab_can_hand_the_solver_a_max_shift(
            self, window, acquisition):
        """The one solver knob the dialog exposes: the guard against a
        periodic lattice locking the correlation onto the wrong translation.
        The planted offsets reach 5 px, so a 6 px cap must still find them."""
        # Patched where `best_real_space` looks it up: the stage now chooses
        # between registration settings by what they do to the sum, and runs
        # the solver once per candidate. The knob still has to reach it.
        from spyde.multiangle import align as align_module

        data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        original = align_module.solve_real_space
        forwarded = {}

        def _capture(images, *, reference=0, **kwargs):
            forwarded.update(kwargs)
            return original(images, reference=reference, **kwargs)

        with patch.object(align_module, "solve_real_space", _capture):
            maped_align_real(session, None, {"params": {"max_shift": 6.0}})
            assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])

        assert forwarded["max_shift"] == 6.0
        assert _last_state(window["messages"])["real"]["offsets"] == \
            data.nav_offsets.tolist()

    def test_a_parameter_the_solver_does_not_take_is_dropped(self, window,
                                                             acquisition):
        """Forwarded, it would raise inside the worker and the user would see
        a failed stage rather than a control that does nothing."""
        _data, _paths = acquisition
        _with_angles(window["window"], acquisition)

        maped_align_real(window["window"], None,
                         {"params": {"wobble": 3, "max_shift": 8.0}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])

    def test_a_running_solve_says_it_is_busy(self, window, acquisition):
        _data, _paths = acquisition
        _with_angles(window["window"], acquisition)
        before = len(window["messages"])

        maped_align_real(window["window"], None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])

        states = [m for m in window["messages"][before:]
                  if isinstance(m, dict) and m.get("type") == "maped_state"]
        assert states[0]["busy"] is True
        assert states[-1]["busy"] is False

    def test_the_members_are_read_once_for_the_whole_dialog(self, window,
                                                            acquisition):
        """The tableau fill and both solves want the SAME per-member
        reduction, and it is the expensive thing in this loader.

        So it happens once — when the tableau is filled, right after the probe
        — and everything afterwards finds it done. This is the contract that
        makes filling the tableau eagerly affordable: the work moved earlier,
        it was not added.
        """
        _data, paths = acquisition
        session = window["window"]
        reductions = []
        real = loader._member_reductions

        def _counted(signal):
            reductions.append(signal)
            return real(signal)

        with patch.object(loader, "_member_reductions", _counted):
            _with_angles(session, acquisition)
            assert len(reductions) == len(paths), \
                "the tableau fill did not read every member"

            maped_align_real(session, None, {"params": {}})
            assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
            maped_align_real(session, None, {"params": {"upsample": 4}})
            assert quiesce(session), why_busy(session)
            maped_align_reciprocal(session, None, {"method": "beam"})
            assert _wait(
                lambda: _last_state(window["messages"])["reciprocal"]["solved"])

        assert len(reductions) == len(paths), \
            "a solve recomputed the reductions the tableau had already made"

    def test_an_unknown_reciprocal_method_is_refused(self, window, acquisition):
        _data, _paths = acquisition
        _with_angles(window["window"], acquisition)

        maped_align_reciprocal(window["window"], None, {"method": "vibes"})
        assert quiesce(window["window"]), why_busy(window["window"])
        assert not _last_state(window["messages"])["reciprocal"]["solved"]


class TestCommit:
    def test_can_commit_needs_two_members_and_both_stages(self, window,
                                                          acquisition):
        session = window["window"]
        _data, paths = acquisition

        _add(session, paths[:1])
        assert _last_state(window["messages"])["can_commit"] is False

        maped_add_files(session, None, {"paths": paths[1:]})
        _probed(session)
        assert _last_state(window["messages"])["can_commit"] is False

        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        assert _last_state(window["messages"])["can_commit"] is False

        maped_align_reciprocal(session, None, {"method": "beam"})
        assert _wait(
            lambda: _last_state(window["messages"])["reciprocal"]["solved"])
        assert _last_state(window["messages"])["can_commit"] is True

    def test_the_offsets_are_indexed_parallel_to_the_members(self, window,
                                                             acquisition):
        """Row *i* belongs to ``members[i]``, so a list that changed under
        them leaves them indexed against names that are no longer there."""
        session = window["window"]
        data, _paths = acquisition
        _with_angles(session, acquisition)
        _solve_both(session, window["messages"])

        state = _last_state(window["messages"])
        assert len(state["real"]["offsets"]) == len(state["members"])
        assert state["real"]["offsets"] == data.nav_offsets.tolist()
        # One residual per member — the worse of its two axes, which is what a
        # row beside a member's name has to answer.
        residuals = state["reciprocal"]["residuals"]
        assert len(residuals) == len(state["members"])
        assert all(isinstance(value, float) for value in residuals)
        assert max(residuals) == pytest.approx(
            state["reciprocal"]["max_residual"])

        maped_remove_member(session, None, {"index": 1})
        dropped = _last_state(window["messages"])
        assert dropped["real"]["offsets"] is None
        assert dropped["reciprocal"]["offsets"] is None
        assert dropped["can_commit"] is False

    def test_the_reference_is_null_until_a_member_can_be_one(self, window,
                                                             acquisition):
        _data, paths = acquisition
        session = window["window"]
        maped_open_loader(session, None, {})
        assert _last_state(window["messages"])["reference"] is None

        _add(session, paths)
        assert _last_state(window["messages"])["reference"] == 0

    def test_changing_the_reference_drops_both_solves(self, window,
                                                      acquisition):
        """An offset is a correction TOWARDS the reference, so every row of
        both arrays means something else afterwards."""
        session = window["window"]
        _with_angles(session, acquisition)
        _solve_both(session, window["messages"])

        maped_set_reference(session, None, {"index": 2})
        state = _last_state(window["messages"])
        assert state["reference"] == 2
        assert state["real"]["solved"] is False
        assert state["reciprocal"]["solved"] is False
        assert state["can_commit"] is False

    def test_committing_unsolved_opens_nothing(self, window, acquisition):
        session = window["window"]
        _data, paths = acquisition
        _add(session, paths)

        maped_commit(session, None, {})
        assert quiesce(session), why_busy(session)
        assert session.signal_trees == []

    def test_a_failed_commit_is_reported_after_the_dialog_closes(
            self, window, acquisition):
        """The renderer commits and CLOSES the dialog, so a failure that only
        updated an open dialog would be silent — and a dismissed dialog with
        no dataset and no explanation is the worst outcome available."""
        session = window["window"]
        _with_angles(session, acquisition)
        _solve_both(session, window["messages"])

        def _refuse(*args, **kwargs):
            raise RuntimeError("composition refused")

        with patch.object(loader, "compose_multiangle_tree", _refuse):
            maped_commit(session, None, {})
            maped_close_loader(session, None, {})
            after_close = len(window["messages"])
            assert quiesce(session), why_busy(session)
            drain_loop(session)

        later = [m for m in window["messages"][after_close:]
                 if isinstance(m, dict)]
        assert any(m.get("type") == "error"
                   and "composition refused" in str(m.get("text")) for m in later), \
            "the failure never reached the status bar"
        states = [m for m in later if m.get("type") == "maped_state"]
        assert states, "the failure never reached the dialog's own channel"
        assert "composition refused" in states[-1]["message"]
        assert states[-1]["busy"] is False
        assert states[-1]["members"] == [], \
            "reporting the failure put the dismissed member list back"
        assert session.signal_trees == []

    def test_committing_takes_the_alignment_window_with_it(self, window,
                                                           acquisition):
        """The alignment window is evidence for a decision, so it goes when the
        decision is made.

        Left open it is the ACTIVE plot, and being a bare figure it has no
        signal tree — so the Plot Control dock shows no workflow and the
        acquisition the user just opened is not the one in focus. That is
        exactly what it looked like in the app before this.
        """
        session = window["window"]
        _with_angles(session, acquisition)
        _solve_both(session, window["messages"])

        state = session._multiangle_loader
        assert state.aligned_window is not None, \
            "the real-space solve did not open its window"
        window_id = state.aligned_window.window_id

        maped_commit(session, None, {})
        _opened_tree(session, "the staged commit")
        assert _wait(lambda: not _last_state(window["messages"])["busy"])

        assert state.aligned_window is None
        assert session.controller_by_window_id(window_id) is None, \
            "the alignment window outlived the commit"

    def test_commit_builds_the_tree_the_one_shot_loader_does(self, window,
                                                             acquisition):
        """Two doors onto one dataset; the same tree has to come out of both."""
        data, paths = acquisition
        staged = window["window"]
        _with_angles(staged, acquisition)
        _solve_both(staged, window["messages"])

        maped_commit(staged, None, {})
        staged_tree = _opened_tree(staged, "the staged commit")
        assert _wait(lambda: not _last_state(window["messages"])["busy"]), \
            "the commit never reported that it had finished"
        assert "opened" in _last_state(window["messages"])["message"].lower()

        one_shot = make_session()
        try:
            one_shot.open_multiangle(
                paths, data.tilts.tolist(), data.azimuths.tolist(),
                reference=int(data.reference),
                reader_options={"navigation_shape": SCAN_XY})
            _compare_trees(staged_tree, _opened_tree(one_shot,
                                                     "open_multiangle"))
        finally:
            close_session(one_shot)


def _opened_tree(session, what: str):
    """The finished multi-angle tree.

    Waits for the SHELL nodes, not just for "Summed": the composition attaches
    them one after another on a worker thread, so a tree that has the summed
    node is not yet a tree that has all of them — and comparing there compares
    a half-built tree with a finished one.
    """
    def _built():
        if not session.signal_trees:
            return False
        children = session.signal_trees[0].root_node.children
        return "Summed" in children and len(children["Summed"].children) == 2

    assert _wait(_built), f"{what} never produced a finished tree"
    assert quiesce(session), why_busy(session)
    return session.signal_trees[0]


def _compare_trees(staged, one_shot) -> None:
    from spyde.multiangle.recipe import recipe_for

    assert staged.root_node.name == one_shot.root_node.name == "Aligned Stack"
    assert staged.root.data.shape == one_shot.root.data.shape
    assert list(staged.root_node.children) == list(one_shot.root_node.children)

    staged_summed = staged.root_node.children["Summed"]
    one_shot_summed = one_shot.root_node.children["Summed"]
    assert staged_summed.signal.data.shape == one_shot_summed.signal.data.shape
    assert sorted(staged_summed.children) == sorted(one_shot_summed.children)

    staged_model = recipe_for(staged_summed.signal).model
    one_shot_model = recipe_for(one_shot_summed.signal).model
    assert np.array_equal(staged_model.nav_offsets, one_shot_model.nav_offsets)
    assert np.array_equal(staged_model.dp_offsets, one_shot_model.dp_offsets)
    assert np.array_equal(staged_model.shell_ids, one_shot_model.shell_ids)


class TestClosing:
    def test_close_drops_the_state(self, window, acquisition):
        session = window["window"]
        _data, paths = acquisition
        _add(session, paths)
        assert session._multiangle_loader is not None

        maped_close_loader(session, None, {})
        assert session._multiangle_loader is None
        state = _last_state(window["messages"])
        assert state["members"] == []
        assert state["busy"] is False
        assert state["can_commit"] is False

    def test_a_solve_closed_mid_flight_never_lands(self, window, acquisition):
        """The dialog is gone; a result arriving afterwards would announce a
        state nothing is showing, and revive the loader the user dismissed."""
        session = window["window"]
        _with_angles(session, acquisition)

        maped_align_real(session, None, {"params": {}})
        maped_close_loader(session, None, {})
        after_close = len(window["messages"])

        assert quiesce(session), why_busy(session)
        later = [m for m in window["messages"][after_close:]
                 if isinstance(m, dict) and m.get("type") == "maped_state"]
        assert all(not state["real"]["solved"] for state in later), \
            "a superseded solve landed after the loader was closed"
        assert session._multiangle_loader is None


# ─────────────────────────────────────────────────────────────────────────────
# The tableaux: virtual images, scan corners, and the thumbnails of both
# ─────────────────────────────────────────────────────────────────────────────

#: Names planted on every member of :func:`virtual_image_acquisition`.
#:
#: ``HAADF`` is each member's TRUE scene crop, so aligning on it recovers the
#: planted offsets. ``Static`` is member 0's crop repeated on every member, so
#: aligning on it recovers zeros — a wrong answer, deliberately, and a
#: DISTINGUISHABLE one. Without it, "the dropdown drives the solve" could pass
#: on a loader that ignored the dropdown entirely.
TRUE_IMAGE = "HAADF"
STATIC_IMAGE = "Static"

#: On member 0 alone, so the offered list has to be an intersection.
LONE_IMAGE = "Only0"

#: Not the scan grid — an external camera frame is the real-world case. It must
#: not be offered: nothing can be registered against another member's copy.
OFF_GRID_IMAGE = "Elsewhere"


def _with_navigators(signal, data, index: int):
    """Plant the virtual images a Direct Electron ``.mrc`` would ship.

    The reader puts them in ``metadata._HyperSpy.navigators``; ``.hspy`` stores
    that node verbatim, which is what lets the loader's real discovery path be
    tested without a full Direct Electron sidecar set.
    """
    item = "_HyperSpy.navigators."
    signal.metadata.set_item(
        item + TRUE_IMAGE,
        hs.signals.Signal2D(data.images[index].astype(np.float32)))
    signal.metadata.set_item(
        item + STATIC_IMAGE,
        hs.signals.Signal2D(data.images[0].astype(np.float32)))
    signal.metadata.set_item(
        item + OFF_GRID_IMAGE,
        hs.signals.Signal2D(np.zeros((3, 4), dtype=np.float32)))
    if index == 0:
        signal.metadata.set_item(
            item + LONE_IMAGE,
            hs.signals.Signal2D(data.images[index].astype(np.float32)))
    return signal


@pytest.fixture(scope="module")
def virtual_image_acquisition(tmp_path_factory):
    """Members that SHIP their virtual images, plus one that ships none.

    Saved as ``.hspy`` so they are self-describing and the scan-shape field
    plays no part — this fixture is about what the members CARRY.
    """
    data = make_multiangle(shells=((1.0, 3), (0.5, 2)), scan_shape=SCAN,
                           detector_shape=DETECTOR, seed=3)
    directory = tmp_path_factory.mktemp("multiangle-loader-vi")
    paths = []
    for index, member in enumerate(data.members):
        path = str(directory / f"member{index:02d}.hspy")
        _with_navigators(hs.signals.Signal2D(member), data, index).save(path)
        paths.append(path)
    bare = str(directory / "bare.hspy")
    hs.signals.Signal2D(data.members[0]).save(bare)
    return data, paths, bare


def _add_carrying(session, paths):
    """Loaded the way the dialog loads a self-describing member."""
    _add(session, paths, scan_shape=None)


class TestVirtualImages:
    """Real space is aligned from a virtual image, chosen from what the data
    already carries. The failures are all silent ones: a name only one member
    has offers an alignment that cannot be made, an image of something other
    than the scan cannot be registered at all, and a selection that does not
    actually reach the solve looks exactly like one that does."""

    def test_the_names_a_member_carries_reach_the_snapshot(
            self, window, virtual_image_acquisition):
        _data, paths, _bare = virtual_image_acquisition
        _add_carrying(window["window"], paths)

        members = _last_state(window["messages"])["members"]
        assert members[0]["virtual_images"] == sorted(
            [TRUE_IMAGE, STATIC_IMAGE, LONE_IMAGE])
        assert all(member["virtual_images"] == sorted(
            [TRUE_IMAGE, STATIC_IMAGE]) for member in members[1:])

    def test_only_the_names_every_member_has_are_offered(
            self, window, virtual_image_acquisition):
        """A virtual image one member has cannot align a set."""
        _data, paths, _bare = virtual_image_acquisition
        _add_carrying(window["window"], paths)

        state = _last_state(window["messages"])
        assert state["available_virtual_images"] == sorted(
            [TRUE_IMAGE, STATIC_IMAGE])
        assert LONE_IMAGE not in state["available_virtual_images"]

    def test_an_image_that_is_not_the_scan_is_not_offered(
            self, window, virtual_image_acquisition):
        """It cannot be registered against another member's copy, so offering
        it would be offering a stage that fails after the click."""
        _data, paths, _bare = virtual_image_acquisition
        _add_carrying(window["window"], paths)

        state = _last_state(window["messages"])
        assert OFF_GRID_IMAGE not in state["available_virtual_images"]
        assert all(OFF_GRID_IMAGE not in member["virtual_images"]
                   for member in state["members"])

    def test_the_loader_picks_one_without_being_asked(
            self, window, virtual_image_acquisition):
        """The acquisition shipped it; reducing the whole detector to reinvent
        it is the expensive way to get a worse picture."""
        _data, paths, _bare = virtual_image_acquisition
        _add_carrying(window["window"], paths)

        assert _last_state(window["messages"])["virtual_image"] == TRUE_IMAGE

    def test_members_that_carry_none_leave_it_computed(self, window,
                                                       acquisition):
        _data, paths = acquisition
        _add(window["window"], paths)

        state = _last_state(window["messages"])
        assert state["virtual_image"] is None
        assert state["available_virtual_images"] == []

    def test_the_chosen_image_is_the_one_that_is_registered(
            self, window, virtual_image_acquisition):
        """``Static`` is the same picture on every member, so a solve that
        really used it returns zeros — which a solve that quietly used
        something else cannot."""
        data, paths, _bare = virtual_image_acquisition
        session = window["window"]
        _add_carrying(session, paths)

        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        assert _last_state(window["messages"])["real"]["offsets"] == \
            data.nav_offsets.tolist()

        maped_set_virtual_image(session, None, {"name": STATIC_IMAGE})
        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        assert _last_state(window["messages"])["real"]["offsets"] == \
            [[0, 0]] * len(paths)

    def test_aligning_on_a_virtual_image_reads_no_member(
            self, window, virtual_image_acquisition):
        """The whole point: the picture is already in the file, so the stage
        that used to be the expensive one becomes free."""
        _data, paths, _bare = virtual_image_acquisition
        session = window["window"]
        _add_carrying(session, paths)

        def _refuse(signal):
            raise AssertionError("the real-space stage reduced a member when "
                                 "the file already carried the image")

        with patch.object(loader, "_member_reductions", _refuse):
            maped_align_real(session, None, {"params": {}})
            assert _wait(
                lambda: _last_state(window["messages"])["real"]["solved"])

    def test_changing_it_drops_the_real_space_solve_alone(
            self, window, virtual_image_acquisition):
        """The offsets came from registering a different set of pictures. The
        reciprocal ones measured the detector, which this does not touch."""
        _data, paths, _bare = virtual_image_acquisition
        session = window["window"]
        _add_carrying(session, paths)
        _solve_both(session, window["messages"], method="corners")

        maped_set_virtual_image(session, None, {"name": STATIC_IMAGE})
        state = _last_state(window["messages"])
        assert state["virtual_image"] == STATIC_IMAGE
        assert state["real"]["solved"] is False
        assert state["real"]["offsets"] is None
        assert state["reciprocal"]["solved"] is True
        assert state["can_commit"] is False

    def test_computing_one_is_a_choice_too(self, window,
                                           virtual_image_acquisition):
        """``null`` means compute it, and it must stick — a loader that
        re-picked its own favourite on the next file drop would undo it."""
        _data, paths, _bare = virtual_image_acquisition
        session = window["window"]
        _add_carrying(session, paths[:3])

        maped_set_virtual_image(session, None, {"name": None})
        assert _last_state(window["messages"])["virtual_image"] is None

        maped_add_files(session, None, {"paths": paths[3:]})
        _probed(session)
        assert _last_state(window["messages"])["virtual_image"] is None

    def test_a_name_the_newcomer_lacks_falls_back_to_computing(
            self, window, virtual_image_acquisition):
        """Aligning the rest on it while the newcomer used something else
        would be a silently different measurement per row."""
        _data, paths, bare = virtual_image_acquisition
        session = window["window"]
        _add_carrying(session, paths)
        maped_set_virtual_image(session, None, {"name": STATIC_IMAGE})

        maped_add_files(session, None, {"paths": [bare]})
        _probed(session)

        state = _last_state(window["messages"])
        assert state["available_virtual_images"] == []
        assert state["virtual_image"] is None

    def test_an_unknown_name_is_refused(self, window,
                                        virtual_image_acquisition):
        _data, paths, _bare = virtual_image_acquisition
        session = window["window"]
        _add_carrying(session, paths)

        maped_set_virtual_image(session, None, {"name": "NotADetector"})
        assert _last_state(window["messages"])["virtual_image"] == TRUE_IMAGE

    def test_the_committed_tree_is_drawn_from_the_image_it_was_aligned_by(
            self, window, virtual_image_acquisition):
        """The images become the composed stack's per-member navigator planes.
        Drawn from one picture and aligned by another, the navigator would
        disagree with its own offsets."""
        data, paths, _bare = virtual_image_acquisition
        session = window["window"]
        _add_carrying(session, paths)
        for index in range(len(paths)):
            maped_set_member(session, None, {
                "index": index, "tilt": float(data.tilts[index]),
                "azimuth": float(data.azimuths[index])})
        _solve_both(session, window["messages"], method="corners")

        planted = []

        def _capture(session_, members_, model_, images_):
            planted.extend(images_)

        with patch.object(loader, "compose_multiangle_tree", _capture):
            maped_commit(session, None, {})
            assert _wait(lambda: not _last_state(window["messages"])["busy"])

        assert len(planted) == len(paths)
        for index, image in enumerate(planted):
            assert np.allclose(np.asarray(image, dtype=np.float64),
                               data.images[index], atol=1e-5)


class TestTheScanCorners:
    """Reciprocal alignment from four corner sums, each with its own extent."""

    def test_it_recovers_the_planted_offsets(self, window, acquisition):
        """The cheap measurement has to be a CORRECT measurement — reading a
        fraction of a percent of each member is only worth anything if the
        answer matches what the expensive stages find."""
        data, _paths = acquisition
        _with_angles(window["window"], acquisition)

        maped_align_reciprocal(window["window"], None, {"method": "corners"})
        assert _wait(
            lambda: _last_state(window["messages"])["reciprocal"]["solved"])

        assert _last_state(window["messages"])["reciprocal"]["offsets"] == \
            data.dp_offsets.tolist()

    def test_it_is_the_default(self, window, acquisition):
        """It is the fast one, so it is what a click with nothing chosen
        does."""
        assert loader.DEFAULTS["method"] == "corners"
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)

        def _refuse(*args, **kwargs):
            raise AssertionError("the default reciprocal stage reduced a "
                                 "member instead of reading its corners")

        with patch.object(loader, "_member_reductions", _refuse):
            maped_align_reciprocal(session, None, {})
            assert _wait(
                lambda: _last_state(window["messages"])["reciprocal"]["solved"])

    def test_the_sums_are_published_as_a_tableau(self, window, acquisition):
        _data, paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        maped_align_reciprocal(session, None, {"method": "corners"})
        assert _wait(
            lambda: _last_state(window["messages"])["reciprocal"]["solved"])

        corners = _last_state(window["messages"])["reciprocal"]["corners"]
        assert sorted(corners) == sorted(str(i) for i in range(len(paths)))
        for entry in corners.values():
            assert len(entry["previews"]) == 4
            assert all(preview.startswith("data:image/png;base64,")
                       for preview in entry["previews"])
            assert entry["extents"] == [loader.DEFAULT_CORNER_FRACTION] * 4

    def test_asking_with_no_extent_fills_only_what_is_missing(self, window,
                                                              acquisition):
        """The probe already fills the tableau, so this is a top-up.

        It still earns its place: it is how a caller re-asks for a corner that
        was thrown away, and it must re-sum that one rather than all four.
        """
        _data, paths = acquisition
        session = window["window"]
        _add(session, paths)
        before = _last_state(window["messages"])["reciprocal"]["corners"]["0"]
        assert all(preview is not None for preview in before["previews"])

        session._multiangle_loader.members[0].forget_corners(2)
        asked = []
        real = loader._corner_sums

        def _counted(signal, extents, corners):
            asked.append(list(corners))
            return real(signal, extents, corners)

        with patch.object(loader, "_corner_sums", _counted):
            maped_set_corner_extent(session, None,
                                    {"member": None, "corner": 2})
            assert _wait(lambda: not _last_state(window["messages"])["busy"])

        assert asked == [[2]]
        after = _last_state(window["messages"])["reciprocal"]["corners"]["0"]
        assert all(preview is not None for preview in after["previews"])
        assert after["extents"] == [loader.DEFAULT_CORNER_FRACTION] * 4

    def test_the_four_are_published_in_row_major_order(self, window,
                                                        acquisition):
        """Top left, top right, bottom left, bottom right — the reading order
        of a 2 x 2 grid, and a STATED contract.

        A renderer laying the four out as a grid and labelling them has no way
        to notice a reordering: every panel is a plausible diffraction pattern,
        so the picture would just be wrong. So each published preview is
        checked against a sum taken over that corner's own block, computed
        here rather than by the loader.
        """
        data, paths = acquisition
        session = window["window"]
        _add(session, paths)
        maped_set_corner_extent(session, None, {"member": None, "corner": 0})
        assert _wait(lambda: not _last_state(window["messages"])["busy"])

        assert loader.CORNER_NAMES == ("top left", "top right",
                                       "bottom left", "bottom right")
        member = session._multiangle_loader.members[0]
        published = _last_state(
            window["messages"])["reciprocal"]["corners"]["0"]["previews"]
        for corner in range(4):
            rows, columns = loader.corner_slice(
                SCAN, corner, member.corner_extents[corner])
            expected = data.members[0][rows, columns].sum(
                axis=(0, 1), dtype=np.float64)
            assert np.allclose(member.corner_patterns[corner], expected)
            assert published[corner] == loader._thumbnail(expected)

    def test_each_corner_has_its_own_extent(self, window, acquisition):
        """One fraction for all four cannot say "the top-left is vacuum and
        the bottom-right is the sample"."""
        _data, _paths = acquisition
        session = window["window"]
        _add(session, _paths)

        maped_set_corner_extent(session, None,
                                {"member": None, "corner": 3, "extent": 0.25})
        assert _wait(lambda: not _last_state(window["messages"])["busy"])

        corners = _last_state(window["messages"])["reciprocal"]["corners"]
        for entry in corners.values():
            assert entry["extents"] == [loader.DEFAULT_CORNER_FRACTION] * 3 \
                + [0.25]

    def test_one_member_can_differ_from_the_others(self, window, acquisition):
        _data, paths = acquisition
        session = window["window"]
        _add(session, paths)

        maped_set_corner_extent(session, None,
                                {"member": 1, "corner": 0, "extent": 0.2})
        assert _wait(lambda: not _last_state(window["messages"])["busy"])

        corners = _last_state(window["messages"])["reciprocal"]["corners"]
        assert corners["1"]["extents"][0] == 0.2
        assert all(corners[str(i)]["extents"][0]
                   == loader.DEFAULT_CORNER_FRACTION
                   for i in range(len(paths)) if i != 1)

    def test_changing_one_corner_re_sums_that_corner_alone(self, window,
                                                           acquisition):
        """The cache is the whole reason the tableau is usable: a slider that
        re-read three untouched corners of every member would make every drag
        step cost the full measurement."""
        _data, paths = acquisition
        session = window["window"]
        _add(session, paths)
        maped_set_corner_extent(session, None, {"member": None, "corner": 0})
        assert _wait(lambda: not _last_state(window["messages"])["busy"])

        asked = []
        real = loader._corner_sums

        def _counted(signal, extents, corners):
            asked.append(list(corners))
            return real(signal, extents, corners)

        with patch.object(loader, "_corner_sums", _counted):
            maped_set_corner_extent(
                session, None, {"member": 2, "corner": 1, "extent": 0.2})
            assert _wait(lambda: not _last_state(window["messages"])["busy"])

        assert asked == [[1]], \
            "a corner nobody moved was summed again"

    def test_re_solving_does_not_re_read_the_corners(self, window,
                                                     acquisition):
        """``quiesce``, not the ``solved`` flag — it is already true from the
        first solve, so waiting for it would not wait at all."""
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        maped_align_reciprocal(session, None, {"method": "corners"})
        assert _wait(
            lambda: _last_state(window["messages"])["reciprocal"]["solved"])

        asked = []
        real = loader._corner_sums

        def _counted(signal, extents, corners):
            asked.append(list(corners))
            return real(signal, extents, corners)

        with patch.object(loader, "_corner_sums", _counted):
            maped_align_reciprocal(session, None, {"method": "corners"})
            assert quiesce(session), why_busy(session)

        assert asked == [], "a cached corner sum was recomputed"

    def test_changing_an_extent_drops_the_reciprocal_solve_alone(
            self, window, acquisition):
        """A corner extent is an input to the reciprocal measurement in
        exactly the way the virtual image is an input to the real-space one.
        The real-space offsets are untouched: the scan did not move."""
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        _solve_both(session, window["messages"], method="corners")

        maped_set_corner_extent(session, None,
                                {"member": None, "corner": 2, "extent": 0.3})
        state = _last_state(window["messages"])
        assert state["reciprocal"]["solved"] is False
        assert state["real"]["solved"] is True
        assert state["can_commit"] is False

    def test_setting_the_same_extent_again_changes_nothing(self, window,
                                                           acquisition):
        """A slider that re-sends its resting value must not throw the solve
        away — the renderer echoes this module, so it will."""
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        _solve_both(session, window["messages"], method="corners")

        maped_set_corner_extent(
            session, None, {"member": None, "corner": 0,
                            "extent": loader.DEFAULT_CORNER_FRACTION})
        assert _wait(lambda: not _last_state(window["messages"])["busy"])
        assert _last_state(window["messages"])["reciprocal"]["solved"] is True

    @pytest.mark.parametrize("payload", [
        {"member": None, "corner": 4, "extent": 0.1},
        {"member": None, "corner": -1, "extent": 0.1},
        {"member": None, "corner": None, "extent": 0.1},
        {"member": None, "corner": 0, "extent": 0.0},
        {"member": None, "corner": 0, "extent": 0.9},
        {"member": None, "corner": 0, "extent": "wide"},
        {"member": 99, "corner": 0, "extent": 0.1},
    ])
    def test_a_nonsense_corner_is_refused_without_touching_the_members(
            self, window, acquisition, payload):
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        _solve_both(session, window["messages"], method="corners")

        maped_set_corner_extent(session, None, payload)
        assert quiesce(session), why_busy(session)

        state = _last_state(window["messages"])
        assert state["reciprocal"]["solved"] is True
        assert all(entry["extents"] == [loader.DEFAULT_CORNER_FRACTION] * 4
                   for entry in state["reciprocal"]["corners"].values())

    def test_the_other_two_methods_still_work(self, window, acquisition):
        """``corners`` is a third option, not a replacement — a blocked or
        diffuse centre still needs ``correlate``."""
        data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)

        for method in ("beam", "correlate"):
            maped_align_reciprocal(session, None, {"method": method})
            assert _wait(
                lambda: _last_state(window["messages"])["reciprocal"]["solved"])
            assert _last_state(window["messages"])["reciprocal"]["offsets"] \
                == data.dp_offsets.tolist()
            maped_set_reference(session, None, {"index": 0})   # drop the solve


class TestTheThumbnails:
    """The renderer cannot read an array, so every picture the loader decides
    something from is also published as a small PNG."""

    def test_both_tableaux_are_filled_by_the_probe_alone(self, window,
                                                         acquisition):
        """THE regression this class exists for.

        A tableau is what the user looks at to DECIDE whether to run a stage,
        so it cannot be the reward for having run one. Nothing in this test
        solves anything: files are added, and both grids have pictures in them.
        """
        _data, paths = acquisition
        _add(window["window"], paths)

        state = _last_state(window["messages"])
        assert state["real"]["solved"] is False, "this test must not solve"
        assert state["reciprocal"]["solved"] is False

        assert all(member["preview"].startswith("data:image/png;base64,")
                   for member in state["members"]), \
            "the Load tab's tableau was empty after the probe"
        corners = state["reciprocal"]["corners"]
        assert len(corners) == len(paths)
        assert all(all(preview is not None for preview in entry["previews"])
                   for entry in corners.values()), \
            "the reciprocal tableau was empty after the probe"

    def test_the_tiles_arrive_one_member_at_a_time(self, window, acquisition):
        """A ten-member acquisition shows its first tile in the time ONE
        member takes, not nothing until the last.

        Reading the members is the dominant cost of this loader on real data,
        so a pass that announced only its result would be a minute of a dialog
        that looks broken.

        Each member is held until the dialog has shown the one before it, so
        what is asserted is the interleaving the code produces and not
        whatever this machine's timing happened to produce — on a fixture this
        small the worker otherwise outruns the event loop and several members
        land in one snapshot.
        """
        _data, paths = acquisition
        session = window["window"]
        release = threading.Event()
        real = loader._member_reductions

        def _one_at_a_time(signal):
            release.wait(60.0)
            release.clear()
            return real(signal)

        def _tiles() -> int:
            return sum(1 for member in _last_state(window["messages"])["members"]
                       if member["preview"])

        maped_open_loader(session, None, {})
        maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})
        before = len(window["messages"])

        with patch.object(loader, "_member_reductions", _one_at_a_time):
            maped_add_files(session, None, {"paths": list(paths)})
            for expected in range(1, len(paths) + 1):
                release.set()
                assert _wait(lambda: _tiles() == expected), \
                    f"the dialog never showed exactly {expected} tile(s)"
                if expected < len(paths):
                    # Still holding a member, so the pass cannot have ended.
                    # After the LAST one it may already have, which is a race
                    # with nothing to say — the history is checked below.
                    assert _last_state(window["messages"])["busy"] is True
            _probed(session)

        states = [m for m in window["messages"][before:]
                  if isinstance(m, dict) and m.get("type") == "maped_state"]
        # Working, visibly, the whole way — a dialog that said it was idle
        # while it read would have every control live over a busy backend.
        assert all(state["busy"] for state in states[:-1])
        assert states[-1]["busy"] is False
        assert _tiles() == len(paths)

    def test_a_member_that_cannot_be_read_has_none(self, window, acquisition):
        """``null`` means "not measured", never "measured and blank" — so a
        renderer draws a placeholder rather than a black panel."""
        _data, paths = acquisition
        _add(window["window"], list(paths) + ["/no/such/member.mrc"])

        members = _last_state(window["messages"])["members"]
        assert members[-1]["preview"] is None
        assert all(member["preview"] for member in members[:-1])

    def test_a_carried_image_gives_one_without_reading_anything(
            self, window, virtual_image_acquisition):
        """A member that ships its virtual image has a tableau panel the
        moment it is probed — no solve, no reduction, no frames read."""
        _data, paths, _bare = virtual_image_acquisition
        _add_carrying(window["window"], paths)

        assert all(member["preview"].startswith("data:image/png;base64,")
                   for member in _last_state(window["messages"])["members"])

    def test_it_is_a_thumbnail_and_not_the_data(self, window,
                                               virtual_image_acquisition):
        """A full-size image per member would put megabytes through a
        line-oriented protocol on every snapshot, and every action here
        answers with the WHOLE snapshot."""
        import base64
        import io

        from PIL import Image

        _data, paths, _bare = virtual_image_acquisition
        _add_carrying(window["window"], paths)

        preview = _last_state(window["messages"])["members"][0]["preview"]
        image = Image.open(io.BytesIO(
            base64.b64decode(preview.split(",", 1)[1])))
        assert max(image.size) <= loader.PREVIEW_MAX_EDGE
        assert image.size == (SCAN[1], SCAN[0])     # never UPSCALED

    def test_one_is_not_rebuilt_when_nothing_relevant_changed(
            self, window, acquisition):
        """Re-running a solve re-reports every member; re-ENCODING every
        member's PNG each time is what the cache is there to stop.

        Waits on ``quiesce`` rather than on ``real.solved``: the flag is
        already true from the first solve, so waiting for it would return at
        once and leave the second solve running into the next test.
        """
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        # "solved" is announced before the first solve's thumbnails have all
        # been encoded; counting from here caught that solve's own late
        # encodes and called them a cache miss.
        assert quiesce(session), why_busy(session)

        encoded = []
        real = loader._thumbnail

        def _counted(array):
            encoded.append(array)
            return real(array)

        with patch.object(loader, "_thumbnail", _counted):
            maped_align_real(session, None, {"params": {"upsample": 4}})
            assert quiesce(session), why_busy(session)

        assert encoded == [], "a cached thumbnail was re-encoded"

    def test_changing_the_image_rebuilds_it(self, window,
                                            virtual_image_acquisition):
        _data, paths, _bare = virtual_image_acquisition
        session = window["window"]
        _add_carrying(session, paths)
        before = [member["preview"]
                  for member in _last_state(window["messages"])["members"]]

        state = session._multiangle_loader
        maped_set_virtual_image(session, None, {"name": STATIC_IMAGE})
        after = [member["preview"]
                 for member in _last_state(window["messages"])["members"]]

        assert after[0] == before[0], \
            "member 0's two images are the same picture"
        assert all(later != earlier
                   for earlier, later in zip(before[1:], after[1:]))
        assert len(set(after)) == 1, \
            "every member shows the SAME picture, which is what Static is"

    def test_a_bright_outlier_does_not_black_out_the_panel(self):
        """One saturated pixel on an electron detector is ordinary. A min/max
        stretch would put the whole scene in the bottom percent of the range
        and the panel would read as empty."""
        scene = np.linspace(100.0, 200.0, 64 * 64).reshape(64, 64)
        with_outlier = scene.copy()
        with_outlier[0, 0] = 1e6

        assert loader._stretched(with_outlier)[32:, :].mean() > 100

    def test_a_flat_image_is_grey_rather_than_black(self):
        """0-or-255 would read as data; grey says "this is flat"."""
        assert np.all(loader._stretched(np.full((8, 8), 7.0)) == 128)

    def test_nothing_to_draw_is_no_picture_at_all(self):
        assert loader._thumbnail(None) is None
        assert loader._thumbnail(np.zeros((0, 4))) is None
        assert loader._thumbnail(np.zeros((4, 4, 3))) is None

    def test_closing_the_dialog_stops_the_fill(self, window, acquisition):
        """Filling the tableau now starts UNPROMPTED, so it has to stop the
        same way a solve does.

        On a real acquisition each member is a pass over a multi-gigabyte
        file. A user who drops ten of them and immediately closes the dialog
        must not leave nine of those running for a window that is gone.
        """
        _data, paths = acquisition
        session = window["window"]
        started, release, read = (threading.Event(), threading.Event(), [])
        real = loader._member_reductions

        def _slow(signal):
            read.append(signal)
            started.set()
            release.wait(30.0)
            return real(signal)

        with patch.object(loader, "_member_reductions", _slow):
            maped_open_loader(session, None, {})
            maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})
            maped_add_files(session, None, {"paths": list(paths)})
            assert started.wait(60.0), "the tableau fill never started"

            maped_close_loader(session, None, {})
            release.set()
            assert quiesce(session), why_busy(session)

        assert len(read) == 1, \
            "the fill kept reading members after the dialog was closed"
        assert session._multiangle_loader is None

    def test_a_fill_closed_mid_flight_never_lands(self, window, acquisition):
        """Its snapshot would announce a state nothing is showing, and revive
        the member list the user dismissed."""
        _data, paths = acquisition
        session = window["window"]
        maped_open_loader(session, None, {})
        maped_set_scan_shape(session, None, {"scan_shape": list(SCAN_XY)})
        maped_add_files(session, None, {"paths": list(paths)})
        assert _wait(lambda: session._multiangle_loader is not None
                     and any(member.preview
                             for member in session._multiangle_loader.members))

        # Closed ON THE LOOP, as the app closes it: a handler called from the
        # test thread can bump the generation while a fill's callback is
        # already past its check on the loop, and that interleaving does not
        # exist in the app, where both run on the loop in turn.
        def _close():
            maped_close_loader(session, None, {})
            return len(window["messages"])

        after_close = call_on_loop(session, _close)
        assert quiesce(session), why_busy(session)
        drain_loop(session)

        later = [m for m in window["messages"][after_close:]
                 if isinstance(m, dict) and m.get("type") == "maped_state"]
        assert all(state["members"] == [] for state in later), \
            "a superseded tableau fill landed after the loader was closed"




class TestTheAlignedSumWindow:
    """The real-space solve's own evidence.

    A residual in pixels says how far from a whole pixel the answer landed; it
    does not say whether the members are the same region of the same sample.
    Summed on top of one another they do: aligned, the features stack and the
    sum is sharp; unaligned, it blurs. So the stage opens a window with both,
    and everything here is about that window not multiplying or outliving the
    dialog that owns it.
    """

    def _figures(self, messages):
        return [m for m in messages
                if isinstance(m, dict) and m.get("type") == "figure"
                and m.get("title") == loader.ALIGNED_WINDOW_TITLE]

    def test_the_solve_opens_it(self, window, acquisition):
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)

        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        assert quiesce(session), why_busy(session)

        figures = self._figures(window["messages"])
        assert len(figures) == 1, "the real-space stage opened no window"
        opened = session._multiangle_loader.aligned_window
        assert opened is not None
        assert figures[0]["window_id"] == opened.window_id
        assert session.controller_by_window_id(opened.window_id) is opened

    def test_re_running_moves_the_window_not_the_window_count(self, window,
                                                              acquisition):
        """Three tries at the alignment must leave one window, not three."""
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)

        for upsample in (1, 4, 8):
            maped_align_real(session, None, {"params": {"upsample": upsample}})
            assert quiesce(session), why_busy(session)

        assert len(self._figures(window["messages"])) == 1

    def test_closing_the_dialog_closes_it(self, window, acquisition):
        """It is the dialog's evidence for a stage nothing is standing in any
        more, so it goes when the dialog does — through the real close path,
        so the renderer removes it and its figure is released."""
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        assert quiesce(session), why_busy(session)
        window_id = session._multiangle_loader.aligned_window.window_id

        before = len(window["messages"])
        maped_close_loader(session, None, {})

        closed = [m for m in window["messages"][before:]
                  if isinstance(m, dict) and m.get("type") == "window_closed"]
        assert [m["window_id"] for m in closed] == [window_id]
        assert session.controller_by_window_id(window_id) is None

    def test_an_x_on_the_window_leaves_the_dialog_alone(self, window,
                                                        acquisition):
        """Closing the evidence must not close the loader, and re-running must
        then open it again rather than repainting a window that is gone."""
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        assert quiesce(session), why_busy(session)
        state = session._multiangle_loader

        session._forget_window(state.aligned_window.window_id)
        assert state.aligned_window is None
        assert session._multiangle_loader is state

        maped_align_real(session, None, {"params": {"upsample": 4}})
        assert quiesce(session), why_busy(session)
        assert len(self._figures(window["messages"])) == 2
        assert state.aligned_window is not None

    def test_a_superseded_solve_opens_nothing(self, window, acquisition):
        """The window is opened behind the stage's generation guard, so a
        solve whose result is dropped cannot leave a window behind."""
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)

        maped_align_real(session, None, {"params": {}})
        maped_close_loader(session, None, {})
        assert quiesce(session), why_busy(session)
        drain_loop(session)

        assert self._figures(window["messages"]) == []

    def test_aligning_is_what_makes_the_sum_sharp(self, window, acquisition):
        """The whole claim the window makes, as a number: the same members
        over the same region, summed with and without their offsets."""
        data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        state = session._multiangle_loader

        model = state.build_model(
            dp_offsets=np.zeros((len(state.members), 2), dtype=np.int64))
        evidence = loader._alignment_evidence(
            [member.image for member in state.members], model)

        assert evidence["aligned"].shape == evidence["unaligned"].shape, \
            "the two panels must be the same pixels or the numbers cannot be "\
            "compared"
        assert evidence["gain"] > 1.0
        assert np.array_equal(state.real.offsets, data.nav_offsets)

    def test_it_reads_no_member(self, window, acquisition):
        """Display only, and built from the real-space images already in hand.
        A window that read the 4-D arrays would put the loader's whole cost
        behind a picture."""
        _data, _paths = acquisition
        session = window["window"]
        _with_angles(session, acquisition)
        maped_align_real(session, None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        assert quiesce(session), why_busy(session)
        session._forget_window(
            session._multiangle_loader.aligned_window.window_id)
        before = len(window["messages"])

        def _refuse(*args, **kwargs):
            raise AssertionError("the aligned-sum window read a member")

        with patch.object(da.Array, "compute", _refuse), \
                patch.object(da, "compute", _refuse):
            loader._show_aligned_sum(session, session._multiangle_loader)

        assert len(self._figures(window["messages"][before:])) == 1


def _planted_pattern(size=512, beam=(250.0, 262.0), beam_peak=900.0,
                     reflection=(250.0, 346.0), reflection_peak=2600.0):
    """A zero beam near the middle and a BRIGHTER reflection 84 px away.

    84 px is where GaN (0002) lands on the real acquisition this came from,
    and the reflection outshining the zero beam is the ordinary case off zone
    axis. A centre of mass over the whole frame is dragged towards whichever
    reflections a tilt excites; this is the smallest thing that shows it.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    def blob(cy, cx, peak, width=6.0):
        return peak * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) /
                               (2.0 * width ** 2)))
    return (blob(*beam, beam_peak) + blob(*reflection, reflection_peak)
            + 1.0).astype(np.float32)


class TestTheBeamSearchRegion:
    """The reciprocal stage looks for the zero beam where it is told to."""

    def test_no_region_means_the_whole_pattern(self):
        assert loader.beam_region(None, (64, 64)) is None
        assert loader.beam_region({}, (64, 64)) is None

    def test_a_region_is_clipped_to_the_pattern(self):
        rows, columns = loader.beam_region(
            {"cy": 4.0, "cx": 60.0, "half": 10.0}, (64, 64))
        assert (rows.start, rows.stop) == (0, 15)
        assert (columns.start, columns.stop) == (50, 64)

    def test_a_region_off_the_pattern_is_refused_rather_than_empty(self):
        assert loader.beam_region(
            {"cy": -50.0, "cx": 32.0, "half": 4.0}, (64, 64)) is None

    def test_nonsense_is_ignored_rather_than_raised(self):
        assert loader.beam_region({"cy": 1.0}, (64, 64)) is None
        assert loader.beam_region({"cy": "a", "cx": 1, "half": 2}, (64, 64)) is None

    def test_the_whole_pattern_is_pulled_by_a_bright_reflection(self):
        """The defect, stated as a measurement rather than a story."""
        pattern = _planted_pattern()
        x, _y = loader._beam_position(pattern, dict(loader.DEFAULTS), None)
        centre_x = loader._beam_position(
            _planted_pattern(reflection_peak=0.0),
            dict(loader.DEFAULTS), None)[0]
        assert abs(x - centre_x) > 20.0

    def test_a_region_on_the_beam_measures_the_beam(self):
        pattern = _planted_pattern()
        roi = {"cy": 256.0, "cx": 256.0, "half": 32.0}
        x, y = loader._beam_position(pattern, dict(loader.DEFAULTS), roi)
        alone = loader._beam_position(
            _planted_pattern(reflection_peak=0.0), dict(loader.DEFAULTS), roi)
        assert abs(x - alone[0]) < 1.0
        assert abs(y - alone[1]) < 1.0

    def test_the_region_is_reported_in_the_full_patterns_frame(self):
        """Where the region sits must not change the answer.

        Not compared against the whole-pattern reading, which is not a
        reference: a centre of mass over the whole frame is dragged towards
        the middle by the background alone, which is half of why the region
        exists.
        """
        pattern = _planted_pattern(beam=(250.0, 262.0), reflection_peak=0.0)
        answers = [
            loader._beam_position(pattern, dict(loader.DEFAULTS), roi)
            for roi in ({"cy": 250.0, "cx": 262.0, "half": 40.0},
                        {"cy": 240.0, "cx": 250.0, "half": 60.0},
                        {"cy": 262.0, "cx": 275.0, "half": 50.0})]
        # Within a pixel, not exactly: a centre of mass is pulled towards its
        # OWN middle by the background, so where the region sits moves the
        # answer slightly. Forgetting the correction instead moves it by the
        # region's origin, which is tens of pixels.
        for x, y in answers[1:]:
            assert abs(x - answers[0][0]) < 1.5
            assert abs(y - answers[0][1]) < 1.5
        # And it is the planted beam: the finder reports centre − beam, so a
        # beam 6 px right of and 6 px above the middle of a 512 frame reads
        # (−6, +6).
        assert abs(answers[0][0] - -6.0) < 1.0
        assert abs(answers[0][1] - 6.0) < 1.0


class TestSettingTheBeamRegion:
    def test_it_is_placed_on_the_detector_once_members_are_probed(self):
        state = loader.MultiAngleLoaderState()
        state.members = [loader.LoaderMember(path="a", name="a",
                                             detector_shape=(512, 512))]
        loader._default_beam_roi(state)
        assert state.beam_roi == {"cy": 256.0, "cx": 256.0, "half": 64.0}

    def test_a_region_already_chosen_is_not_overwritten(self):
        state = loader.MultiAngleLoaderState()
        state.members = [loader.LoaderMember(path="a", name="a",
                                             detector_shape=(512, 512))]
        state.beam_roi = {"cy": 10.0, "cx": 20.0, "half": 5.0}
        loader._default_beam_roi(state)
        assert state.beam_roi == {"cy": 10.0, "cx": 20.0, "half": 5.0}

    def test_moving_it_throws_the_reciprocal_solve_away(self, window):
        session = window["window"]
        loader.maped_open_loader(session, None, {})
        state = session._multiangle_loader
        state.reciprocal.offsets = np.zeros((2, 2), dtype=np.int64)
        loader.maped_set_beam_roi(
            session, None, {"beam_roi": {"cy": 1.0, "cx": 2.0, "half": 3.0}})
        assert state.beam_roi == {"cy": 1.0, "cx": 2.0, "half": 3.0}
        assert not state.reciprocal.solved

    def test_resending_the_same_region_keeps_the_solve(self, window):
        session = window["window"]
        loader.maped_open_loader(session, None, {})
        state = session._multiangle_loader
        roi = {"cy": 1.0, "cx": 2.0, "half": 3.0}
        loader.maped_set_beam_roi(session, None, {"beam_roi": roi})
        state.reciprocal.offsets = np.zeros((2, 2), dtype=np.int64)
        loader.maped_set_beam_roi(session, None, {"beam_roi": dict(roi)})
        assert state.reciprocal.solved

    def test_null_searches_the_whole_pattern_again(self, window):
        session = window["window"]
        loader.maped_open_loader(session, None, {})
        state = session._multiangle_loader
        state.beam_roi = {"cy": 1.0, "cx": 2.0, "half": 3.0}
        loader.maped_set_beam_roi(session, None, {"beam_roi": None})
        assert state.beam_roi is None

    def test_a_region_with_no_size_is_refused(self, window):
        session = window["window"]
        loader.maped_open_loader(session, None, {})
        state = session._multiangle_loader
        loader.maped_set_beam_roi(
            session, None, {"beam_roi": {"cy": 1.0, "cx": 2.0, "half": 0.0}})
        assert state.beam_roi is None

    def test_the_snapshot_carries_it(self, window):
        session = window["window"]
        loader.maped_open_loader(session, None, {})
        state = session._multiangle_loader
        loader.maped_set_beam_roi(
            session, None, {"beam_roi": {"cy": 1.0, "cx": 2.0, "half": 3.0}})
        message = loader.state_message(state)
        assert message["beam_roi"] == {"cy": 1.0, "cx": 2.0, "half": 3.0}


class TestMovingAMemberByHand:
    """`maped_set_real_offset` overrides one member's offset.

    It exists because the solve reports an offset per axis whether or not the
    specimen determined one — see `vote_real_space`. Where it did not, the
    number came from noise and a person looking at the picture can do better.
    """

    def _solved(self, window, acquisition):
        _with_angles(window["window"], acquisition)
        maped_align_real(window["window"], None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])
        return _last_state(window["messages"])["real"]

    def test_the_solver_answer_is_kept_to_go_back_to(self, window, acquisition):
        real = self._solved(window, acquisition)
        assert real["solver_offsets"] == real["offsets"]

    def test_a_member_moves_and_the_solver_answer_does_not(
            self, window, acquisition):
        real = self._solved(window, acquisition)
        solver = [list(row) for row in real["offsets"]]
        wanted = [solver[1][0] + 3, solver[1][1] - 2]

        maped_set_real_offset(window["window"], None,
                              {"index": 1, "offset": wanted})
        moved = _last_state(window["messages"])["real"]
        assert moved["offsets"][1] == wanted
        assert moved["offsets"][0] == solver[0], "an untouched member moved"
        assert moved["solver_offsets"] == solver, \
            "moving by hand overwrote what it has to be able to restore"

    def test_none_puts_it_back(self, window, acquisition):
        real = self._solved(window, acquisition)
        solver = [list(row) for row in real["offsets"]]
        maped_set_real_offset(window["window"], None,
                              {"index": 1, "offset": [solver[1][0] + 4,
                                                      solver[1][1] + 4]})
        maped_set_real_offset(window["window"], None,
                              {"index": 1, "offset": None})
        assert _last_state(window["messages"])["real"]["offsets"] == solver

    def test_the_offset_is_absolute_not_a_step(self, window, acquisition):
        """Sending the same thing twice must land in the same place.

        A held arrow key produces a burst, and the renderer is free to drop
        all but the last. That is only safe because the payload says where to
        BE rather than how far to move.
        """
        real = self._solved(window, acquisition)
        wanted = [real["offsets"][1][0] + 2, real["offsets"][1][1] + 2]
        for _ in range(3):
            maped_set_real_offset(window["window"], None,
                                  {"index": 1, "offset": wanted})
        assert _last_state(window["messages"])["real"]["offsets"][1] == wanted

    def test_the_reference_cannot_move(self, window, acquisition):
        real = self._solved(window, acquisition)
        before = [list(row) for row in real["offsets"]]
        reference = _last_state(window["messages"])["reference"]
        before_count = len(window["messages"])

        maped_set_real_offset(window["window"], None,
                              {"index": reference, "offset": [5, 5]})
        errors = [m for m in window["messages"][before_count:]
                  if isinstance(m, dict) and m.get("type") == "error"]
        assert errors and "reference" in str(errors[0].get("text"))
        assert _last_state(window["messages"])["real"]["offsets"] == before

    def test_it_refuses_to_pull_the_members_apart(self, window, acquisition):
        """The one hard bound on a hand-set offset is that the members still
        overlap — `max_shift` guards the SOLVER against a wild correlation
        peak, and someone overriding it on purpose is not who that is for."""
        real = self._solved(window, acquisition)
        before = [list(row) for row in real["offsets"]]
        before_count = len(window["messages"])

        maped_set_real_offset(window["window"], None,
                              {"index": 1, "offset": [10_000, 10_000]})
        errors = [m for m in window["messages"][before_count:]
                  if isinstance(m, dict) and m.get("type") == "error"]
        assert errors and "common region" in str(errors[0].get("text"))
        assert _last_state(window["messages"])["real"]["offsets"] == before

    def test_it_will_not_move_anything_before_a_solve(self, window, acquisition):
        _with_angles(window["window"], acquisition)
        before = len(window["messages"])
        maped_set_real_offset(window["window"], None,
                              {"index": 1, "offset": [1, 1]})
        errors = [m for m in window["messages"][before:]
                  if isinstance(m, dict) and m.get("type") == "error"]
        assert errors and "align real space" in str(errors[0].get("text"))


class TestSizingAMemberIsFree:
    """A member's size must never be measured by walking its store.

    A frame-chunked .zspy is one file per frame, so measuring one means a
    `stat` per frame: 36 s each on a 256 x 256 scan, 145 s for four members —
    to put one number on four rows that already say the shape and the dtype.
    """

    def _member(self, tmp_path, shape=(4, 5, 6, 7), dtype="uint16"):
        import hyperspy.api as hs
        member = loader.LoaderMember(path=str(tmp_path / "member.zspy"),
                                     name="member.zspy")
        member.signal = hs.signals.Signal2D(
            np.zeros(shape, dtype=np.dtype(dtype)))
        return member

    def test_the_size_comes_from_the_shape_and_dtype(self, tmp_path):
        member = self._member(tmp_path)
        assert loader._member_size_bytes(member) == 4 * 5 * 6 * 7 * 2

    def test_it_does_not_walk_the_store(self, tmp_path, monkeypatch):
        """The guard: sizing a probed member must not touch the filesystem."""
        member = self._member(tmp_path)

        def refuse(path):
            raise AssertionError(f"walked {path} to size a probed member")

        monkeypatch.setattr(loader, "_dataset_size_bytes", refuse)
        monkeypatch.setattr(loader.os, "walk", refuse)
        assert loader._member_size_bytes(member) > 0

    def test_a_directory_that_would_not_probe_is_not_walked_either(
            self, tmp_path, monkeypatch):
        """Nothing probed means no shape to go on — and still no walk, because
        the walk is the cost this exists to avoid."""
        store = tmp_path / "unreadable.zspy"
        store.mkdir()
        member = loader.LoaderMember(path=str(store), name="unreadable.zspy")

        def refuse(path):
            raise AssertionError(f"walked {path}")

        monkeypatch.setattr(loader.os, "walk", refuse)
        assert loader._member_size_bytes(member) == 0

    def test_a_plain_file_is_still_measured(self, tmp_path):
        """One `stat` is not the problem, and it is the true size on disk."""
        path = tmp_path / "member.mrc"
        path.write_bytes(b"\0" * 2048)
        member = loader.LoaderMember(path=str(path), name="member.mrc")
        assert loader._member_size_bytes(member) == 2048


class TestComparingTwoMembers:
    """The pairwise view: one member against the reference, and their overlay.

    It exists because the sum over every member dilutes one member's move to
    a fraction, so the picture the nudge pad is judged by barely changes when
    the thing being nudged does.
    """

    def _solved(self, window, acquisition):
        _with_angles(window["window"], acquisition)
        maped_align_real(window["window"], None, {"params": {}})
        assert _wait(lambda: _last_state(window["messages"])["real"]["solved"])

    def test_opening_it_gives_three_pictures_and_a_gain(
            self, window, acquisition):
        self._solved(window, acquisition)
        maped_set_pair(window["window"], None, {"index": 1})
        pair = _last_state(window["messages"])["real"]["pair"]
        assert pair is not None and pair["index"] == 1
        for part in ("member", "reference", "overlay"):
            assert str(pair[part]).startswith("data:image/png;base64,"), part
        assert pair["gain"] is None or np.isfinite(pair["gain"])

    def test_the_overlay_is_in_colour(self, window, acquisition):
        """Red against cyan, so a misalignment is a coloured fringe whose SIDE
        says which way to press — not a sharpness the eye has to score."""
        import base64
        import io

        from PIL import Image

        self._solved(window, acquisition)
        maped_set_pair(window["window"], None, {"index": 1})
        pair = _last_state(window["messages"])["real"]["pair"]
        raw = base64.b64decode(pair["overlay"].split(",", 1)[1])
        assert Image.open(io.BytesIO(raw)).mode == "RGB"

    def test_nudging_redraws_it(self, window, acquisition):
        self._solved(window, acquisition)
        maped_set_pair(window["window"], None, {"index": 1})
        before = _last_state(window["messages"])["real"]["pair"]["overlay"]
        offsets = _last_state(window["messages"])["real"]["offsets"]
        maped_set_real_offset(window["window"], None,
                              {"index": 1,
                               "offset": [offsets[1][0] + 2, offsets[1][1] + 2]})
        after = _last_state(window["messages"])["real"]["pair"]["overlay"]
        assert after != before, "the comparison did not follow the member"

    def test_the_view_picks_its_own_image_without_dropping_the_solve(
            self, window, acquisition):
        """The loader's own image choice invalidates the real-space solve, by
        design — its offsets were measured on different pictures. The view has
        to be able to flip between images while the offsets it is editing stay
        exactly where they are."""
        self._solved(window, acquisition)
        offsets = _last_state(window["messages"])["real"]["offsets"]
        names = _last_state(window["messages"])["available_virtual_images"]
        maped_set_pair(window["window"], None,
                       {"index": 1, "image": names[0] if names else None})
        state = _last_state(window["messages"])
        assert state["real"]["solved"], "switching the view's image lost the solve"
        assert state["real"]["offsets"] == offsets

    def test_the_reference_is_not_a_pair(self, window, acquisition):
        self._solved(window, acquisition)
        reference = _last_state(window["messages"])["reference"]
        before = len(window["messages"])
        maped_set_pair(window["window"], None, {"index": reference})
        errors = [m for m in window["messages"][before:]
                  if isinstance(m, dict) and m.get("type") == "error"]
        assert errors and "reference" in str(errors[0].get("text"))

    def test_closing_it_clears_the_pictures(self, window, acquisition):
        self._solved(window, acquisition)
        maped_set_pair(window["window"], None, {"index": 1})
        maped_set_pair(window["window"], None, {"index": None})
        assert _last_state(window["messages"])["real"]["pair"] is None

    def test_a_new_solve_drops_it(self, window, acquisition):
        """Its pictures were drawn from offsets a re-solve has replaced."""
        self._solved(window, acquisition)
        maped_set_pair(window["window"], None, {"index": 1})
        maped_align_real(window["window"], None, {"params": {"upsample": 4}})
        assert quiesce(window["window"]), why_busy(window["window"])
        pair = _last_state(window["messages"])["real"]["pair"]
        assert pair is None or pair["index"] == 1


class TestTheRingAndTheShellsAgree:
    """The tableau groups members onto rings by a tilt tolerance written in
    the TSX; the backend groups them into shells by one written in Python.
    They drifted to 0.05 against 0.01, and two members then sat on one ring
    while the status line counted two shells. Parsing the TSX is the only
    place that disagreement shows."""

    def test_the_tsx_tolerance_is_the_shell_tolerance(self):
        import pathlib
        import re

        from spyde.multiangle.model import DEFAULT_SHELL_TOLERANCE

        path = (pathlib.Path(__file__).resolve().parents[3] / "electron"
                / "src" / "renderer" / "src" / "components"
                / "MultiAngleLoader.tsx")
        assert path.exists(), f"the loader moved: {path}"
        found = re.search(r"const TILT_TOLERANCE_DEG = ([0-9.]+)",
                          path.read_text(encoding="utf-8"))
        assert found, "TILT_TOLERANCE_DEG is no longer a plain constant"
        assert float(found.group(1)) == DEFAULT_SHELL_TOLERANCE, (
            f"the TSX draws rings at {found.group(1)}° while the backend "
            f"counts shells at {DEFAULT_SHELL_TOLERANCE}°")


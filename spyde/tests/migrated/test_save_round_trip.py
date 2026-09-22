"""Save a dataset, load it back, and check it is the same dataset.

Nothing in the suite did this. Every piece was covered — the writer, the
reader, the tree — and the thing a user actually does, save and reopen, was
not. Two bugs lived in that gap and both wrote a file that opened cleanly:

* saving from a NAVIGATOR window wrote the navigator. On a multi-angle
  acquisition that is one virtual image per member, so "save the aligned
  stack" produced 555 KB of thumbnails instead of 93 GB of data, under the
  name that was asked for, reporting success.
* the Crop action never passed its scan box on, so a scan crop came back
  uncropped — the full dataset, saved, with nothing to say it had ignored the
  request.

Both are invisible to a test that checks the writer or the reader alone. What
catches them is comparing what comes back with what went in.
"""
from __future__ import annotations

import os
import time

import hyperspy.api as hs
import numpy as np
import pytest

from spyde.tests.migrated.conftest import close_session, make_session, open_saved


def _saved(session, path, plot=None, timeout=120.0):
    """Run Save and wait for the file to appear and settle."""
    session._save_signal(str(path), plot)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            size = -1
            # A directory store appears before it is finished; wait for it to
            # stop growing rather than racing the writer.
            while time.time() < deadline:
                current = sum(
                    os.path.getsize(os.path.join(root, name))
                    for root, _dirs, names in os.walk(path) for name in names
                ) if os.path.isdir(path) else os.path.getsize(path)
                if current == size and current > 0:
                    return hs.load(str(path), lazy=True)
                size = current
                time.sleep(0.3)
        time.sleep(0.2)
    raise AssertionError(f"Save never produced {path}")


def _plots_of(session):
    return list(session._plots)


def _signal_plot(session):
    """The data window. Save resolves nothing when several plots are open and
    none is named, which is the menu's own path, not this test's subject."""
    for plot in session._plots:
        if (not getattr(plot, "is_navigator", False)
                and getattr(getattr(plot, "plot_state", None),
                            "current_signal", None) is not None):
            return plot
    raise AssertionError("no data window open")


class TestASavedDatasetComesBack:
    """The whole point of Save: what is written is what was open."""

    def test_a_four_dimensional_scan_round_trips(self, tmp_path,
                                                 stem_4d_dataset):
        session = stem_4d_dataset["window"]
        source = session.signal_trees[0].root_node.signal
        wanted = np.asarray(source.data)

        back = _saved(session, tmp_path / "scan.zspy",
                      _signal_plot(session))
        assert back.data.shape == wanted.shape, "a different shape came back"
        assert np.array_equal(np.asarray(back.data), wanted), \
            "the values that came back are not the ones that went in"

    def test_the_signal_type_survives(self, tmp_path, stem_4d_dataset):
        """Losing it gates off the whole diffraction toolchain, so a file that
        reopens as a plain image is not a file that round-tripped."""
        session = stem_4d_dataset["window"]
        back = _saved(session, tmp_path / "typed.zspy",
                      _signal_plot(session))
        assert back.metadata.Signal.signal_type == "electron_diffraction"

    def test_saving_from_the_navigator_writes_the_DATA(self, tmp_path,
                                                       stem_4d_dataset):
        """A navigator's signal is a picture OF the dataset, not the dataset.

        Saving from that window wrote the picture: on a 4-D scan a 2-D
        overview, which opens perfectly well and is not the data. The file was
        three orders of magnitude too small and said nothing about it.
        """
        session = stem_4d_dataset["window"]
        source = session.signal_trees[0].root_node.signal
        navigators = [plot for plot in _plots_of(session)
                      if getattr(plot, "is_navigator", False)]
        if not navigators:
            pytest.skip("this fixture opened no navigator window")

        back = _saved(session, tmp_path / "from_nav.zspy", navigators[0])
        assert back.data.ndim == source.data.ndim, (
            f"saving from the navigator wrote {back.data.ndim} dimensions "
            f"where the dataset has {source.data.ndim} — the navigator, "
            "not the data")
        assert np.array_equal(np.asarray(back.data),
                              np.asarray(source.data))


class TestCroppingThenSaving:
    """A crop that is ignored still saves, and still looks fine."""

    def _session_with(self, data):
        session = make_session()
        signal = hs.signals.Signal2D(data)
        signal.set_signal_type("electron_diffraction")
        session._add_signal(signal)
        return session

    def test_a_scan_crop_reaches_the_file(self, tmp_path):
        from spyde.actions.base import _crop_signal

        data = np.arange(6 * 8 * 4 * 4, dtype=np.uint16).reshape(6, 8, 4, 4)
        session = self._session_with(data)
        try:
            source = session.signal_trees[0].root_node.signal
            cropped = _crop_signal(source, scan_x0=1, scan_x1=6,
                                   scan_y0=2, scan_y1=5)
            assert cropped.data.shape == (3, 5, 4, 4), \
                "the scan box did not reach the crop"
            session._add_signal(cropped)
            # By what it SHOWS: indexing the plot list by position picks up
            # whatever another test left open.
            plot = next(p for p in session._plots
                        if getattr(getattr(p, "plot_state", None),
                                   "current_signal", None) is cropped)
            back = _saved(session, tmp_path / "cropped.zspy", plot)
            assert back.data.shape == (3, 5, 4, 4)
            assert np.array_equal(np.asarray(back.data),
                                  data[2:5, 1:6])
        finally:
            close_session(session)

    def test_the_action_declares_the_scan_box(self):
        """Declared in two places or dropped in silence: the dialog accepts a
        field the action does not list, and `build_kwargs` forwards only what
        it names."""
        from spyde.actions.base import CropAction

        for name in ("scan_x0", "scan_x1", "scan_y0", "scan_y1"):
            assert name in CropAction.parameters, \
                f"{name} is missing from CropAction.parameters"
        from unittest.mock import patch
        with patch.object(CropAction, "signal_tree", None):
            action = CropAction.__new__(CropAction)
            signal = hs.signals.Signal2D(
                np.zeros((4, 5, 4, 4), dtype=np.uint16))
            forwarded = action.build_kwargs(signal, scan_x0=1, scan_x1=4,
                                            scan_y0=1, scan_y1=3)
        assert forwarded["scan_x1"] == 4 and forwarded["scan_y1"] == 3, \
            "build_kwargs dropped the scan box"


class TestTheActionItselfCrops:
    """Through `run()`, the way the toolbar reaches it.

    The earlier tests here drove `_crop_signal` directly and passed while the
    toolbar's Crop quietly ignored its scan box: `build_kwargs` returned a
    dict that did not NAME the scan keys, so they were dropped between the
    resolved parameters and the transform. Neither end was wrong; the wiring
    between them was, and only a call to `run()` goes through it.

    Not `CropAction.parameters` — that supplies defaults, and
    `_resolved_params` merges `ctx.params` wholesale, so the toolbar's values
    arrive with or without it. Removing it leaves these tests passing;
    removing the keys from the returned dict fails them with the shape the
    user saw.
    """

    def _session_with(self, data):
        session = make_session()
        signal = hs.signals.Signal2D(data)
        signal.set_signal_type("electron_diffraction")
        session._add_signal(signal)
        return session

    def _data_plot(self, session):
        return next(plot for plot in session._plots
                    if not getattr(plot, "is_navigator", False)
                    and getattr(getattr(plot, "plot_state", None),
                                "current_signal", None) is not None)

    def test_running_crop_reduces_the_scan(self):
        from spyde.actions.base import CropAction
        from spyde.actions.context import ActionContext

        data = np.arange(6 * 8 * 4 * 4, dtype=np.uint16).reshape(6, 8, 4, 4)
        session = self._session_with(data)
        try:
            plot = self._data_plot(session)
            params = {"scan_x0": 1, "scan_x1": 6, "scan_y0": 2, "scan_y1": 5}
            action = CropAction(
                ActionContext(plot=plot, params=params, action_name="Crop"))
            new = action.run(**params)
            assert new is not None, "Crop produced no node"
            assert new.data.shape == (3, 5, 4, 4), (
                f"Crop returned {new.data.shape}; the scan box did not reach "
                "the transform")
            assert np.array_equal(np.asarray(new.data), data[2:5, 1:6])
        finally:
            close_session(session)

    def test_the_declared_parameters_are_what_gets_resolved(self):
        """`_resolved_params` seeds from `self.parameters`, so a field on the
        toolbar and absent there is accepted and dropped."""
        from spyde.actions.base import CropAction
        from spyde.actions.context import ActionContext

        data = np.zeros((4, 5, 4, 4), dtype=np.uint16)
        session = self._session_with(data)
        try:
            params = {"scan_x0": 1, "scan_x1": 4, "scan_y0": 1, "scan_y1": 3}
            action = CropAction(ActionContext(
                plot=self._data_plot(session), params=params,
                action_name="Crop"))
            resolved = action._resolved_params(params)
            for key, value in params.items():
                assert resolved.get(key) == value, (
                    f"{key} did not survive resolution — check it is in "
                    "CropAction.parameters as well as the toolbar")
        finally:
            close_session(session)

    def test_running_rebin_reduces_the_scan(self):
        from spyde.actions.base import Rebin2DAction
        from spyde.actions.context import ActionContext

        data = np.arange(8 * 10 * 4 * 4, dtype=np.uint16).reshape(8, 10, 4, 4)
        session = self._session_with(data)
        try:
            params = {"scale_x": 2, "scale_y": 2, "scan_x": 2, "scan_y": 2}
            action = Rebin2DAction(ActionContext(
                plot=self._data_plot(session), params=params,
                action_name="Rebin"))
            new = action.run(**params)
            assert new is not None and new.data.shape == (4, 5, 2, 2), (
                f"Rebin returned "
                f"{None if new is None else new.data.shape}; want (4, 5, 2, 2)")
        finally:
            close_session(session)


class TestTheCropCaretSendsWhatTheBackendTakes:
    """The caret is the ONLY thing that reaches Crop from the app.

    `CropWizard.tsx` is a hand-written panel with its own fields, so the
    toolbar's declared parameters never reach the UI for this action. The
    backend grew a scan box, every Python test passed, and the app went on
    cropping only the detector — because the caret sent `{x0, x1, y0, y1}`
    and nothing else. Reading the TSX is the only place that mismatch shows.
    """

    def _caret(self):
        import pathlib

        path = (pathlib.Path(__file__).resolve().parents[3] / "electron"
                / "src" / "renderer" / "src" / "components" / "CropWizard.tsx")
        assert path.exists(), f"the Crop caret moved: {path}"
        return path.read_text(encoding="utf-8")

    def test_the_caret_offers_the_scan_box(self):
        caret = self._caret()
        for testid in ("crop-scan-x0", "crop-scan-x1",
                       "crop-scan-y0", "crop-scan-y1"):
            assert testid in caret, (
                f"{testid} is not in CropWizard.tsx — the backend takes a "
                "scan box and the caret cannot send one")

    def test_the_caret_sends_the_scan_box(self):
        """Offering the fields is not sending them: they have to be in the
        payload `doCrop` hands to `toolbar_action`."""
        caret = self._caret()
        start = caret.index("const doCrop")
        payload = caret[start:caret.index("}", caret.index("toolbar_action",
                                                           start))]
        assert "scan" in payload, (
            "doCrop does not put the scan box in the params it sends; it "
            f"sends: {payload.strip()}")

    def test_what_it_sends_is_what_the_transform_takes(self):
        """Every key the caret sends has to be one `_crop_signal` names, or it
        is dropped without a word on the way through `build_kwargs`."""
        import inspect

        from spyde.actions.base import _crop_signal

        accepted = set(inspect.signature(_crop_signal).parameters)
        caret = self._caret()
        # The keys have to appear in the state `doCrop` spreads into its
        # payload, not merely somewhere in the file: "x0" is a substring of
        # "scan_x0", so a whole-file search passed with the detector box gone.
        fields = {"box": ("x0", "x1", "y0", "y1"),
                  "scan": ("scan_x0", "scan_x1", "scan_y0", "scan_y1")}
        start = caret.index("const doCrop")
        payload = caret[start:caret.index("}", caret.index("toolbar_action",
                                                           start))]
        for state, keys in fields.items():
            assert f"...{state}" in payload, (
                f"doCrop does not spread `{state}` into the params it sends")
            declared = caret[caret.index(f"useState") if state == "box"
                             else caret.index(f"const [{state}"):]
            declared = declared[:declared.index(")")]
            for key in keys:
                assert key in accepted, f"_crop_signal does not take {key}"
                assert f"{key}:" in declared, (
                    f"the caret's `{state}` state has no field {key}")


class TestAMultiAngleAcquisitionReopensWhole:
    """Saved and reopened, a composed acquisition is its TREE again.

    Otherwise it is a 5-D array: the alignment survives and everything built
    on it does not — no Summed node, no shells, no angle ring — and a dataset
    published that way hands the next person an array with no way to know what
    it is. Nothing extra is needed to rebuild them: the sums are reductions
    over the stack's own leading axis and the angles are in its metadata.
    """

    def _stack(self, members=4, shells=(0, 0, 1, 1)):
        from spyde.multiangle.synthetic import saved_stack

        signal, data = saved_stack(members=members, shells=shells, seed=7)
        signal.metadata.set_item(
            "Acquisition.multiangle.member_signal_type", "electron_diffraction")
        return signal, data

    def test_a_stack_is_recognised_and_a_sum_is_not(self):
        from spyde.signals.multiangle import is_multiangle_stack

        stack, _data = self._stack()
        assert is_multiangle_stack(stack)
        summed = hs.signals.Signal2D(np.zeros((5, 6, 4, 4), dtype=np.uint16))
        summed.metadata.set_item("Acquisition.multiangle",
                                 stack.metadata.get_item(
                                     "Acquisition.multiangle").as_dictionary())
        assert not is_multiangle_stack(summed), \
            "a 4-D SUM carries the same metadata and is not a stack"

    def test_reopening_rebuilds_the_nodes(self, tmp_path):
        stack, data = self._stack()
        path = tmp_path / "acquisition.zspy"
        stack.save(str(path))

        session = make_session()
        try:
            tree = open_saved(session, path)
            assert tree.root_node.name == "Aligned Stack", (
                f"the root is {tree.root_node.name!r}; the stack reopened as a "
                "plain dataset")
            names = set(tree.root_node.children)
            assert "Summed" in names, f"no Summed node, only {names}"
        finally:
            close_session(session)

    def test_the_summed_node_is_the_members_added(self, tmp_path):
        """Not merely present — the same numbers the composition produced."""
        from spyde.backend._session_multiangle import _sum_over_angles

        stack, data = self._stack()
        summed = _sum_over_angles(stack, range(data.shape[0]), "Summed")
        assert summed.data.shape == data.shape[1:]
        assert np.array_equal(np.asarray(summed.data),
                              data.astype(np.uint64).sum(axis=0))

    def test_the_sums_are_diffraction_again(self, tmp_path):
        """The stack's own type says multi-angle; the sums are patterns, and
        the whole diffraction toolchain is gated on them saying so."""
        from spyde.backend._session_multiangle import _sum_over_angles

        stack, data = self._stack()
        summed = _sum_over_angles(stack, range(data.shape[0]), "Summed")
        assert summed.metadata.Signal.signal_type == "electron_diffraction"


class TestTheStackTypeStaysDiffraction:
    """The type exists so a saved acquisition can be recognised. It EXTENDS
    ElectronDiffraction2D so that costs nothing: the toolchain is gated on the
    signal type, and a composed acquisition losing it once made every
    diffraction action disappear from the window."""

    def test_the_type_is_a_diffraction_signal(self):
        from pyxem.signals import Diffraction2D

        from spyde.signals.multiangle import MULTIANGLE_SIGNAL_TYPE

        signal = hs.signals.Signal2D(
            np.zeros((3, 4, 5, 6, 6), dtype=np.uint16))
        signal.set_signal_type(MULTIANGLE_SIGNAL_TYPE)
        assert isinstance(signal, Diffraction2D), (
            "a multi-angle stack is not a diffraction signal; every action "
            "gated on that is gone")

    def test_a_stack_is_recognised_by_its_type_alone(self):
        """The type plus the angle axis; the metadata need not say more."""
        from spyde.signals.multiangle import (
            MULTIANGLE_METADATA, MULTIANGLE_SIGNAL_TYPE, is_multiangle_stack,
        )

        signal = hs.signals.Signal2D(
            np.zeros((3, 4, 5, 6, 6), dtype=np.uint16))
        signal.metadata.set_item(MULTIANGLE_METADATA, {"n_members": 3})
        signal.set_signal_type(MULTIANGLE_SIGNAL_TYPE)
        assert is_multiangle_stack(signal)

    def test_an_older_file_without_the_type_is_still_recognised(self):
        """Every acquisition composed before the type existed."""
        from spyde.signals.multiangle import (
            MULTIANGLE_METADATA, is_multiangle_stack,
        )

        signal = hs.signals.Signal2D(
            np.zeros((3, 4, 5, 6, 6), dtype=np.uint16))
        signal.metadata.set_item(MULTIANGLE_METADATA, {"n_members": 3})
        signal.set_signal_type("electron_diffraction")
        assert is_multiangle_stack(signal)


class TestTheAngleRingComesBackToo:
    """The ring is driven by the MODEL, and the model lived in the recipe.

    A recipe is a runtime object attached when an acquisition is composed; it
    is not written to the file. So a reopened acquisition re-expanded into its
    nodes and then had no angles — `multiangle_model` asked the recipe, found
    none, and the ring refused with "needs a multi-angle acquisition" about a
    multi-angle acquisition.
    """

    def _saved_stack(self, tmp_path):
        from spyde.signals.multiangle import (
            MULTIANGLE_METADATA, MULTIANGLE_SIGNAL_TYPE,
        )

        from spyde.multiangle.synthetic import saved_stack

        signal, _data = saved_stack(shells=(0, 0, 1, 1), seed=7,
                                    stack_type=True)
        path = tmp_path / "acquisition.zspy"
        signal.save(str(path))
        return path

    def test_the_model_is_read_from_the_file(self, tmp_path):
        """Without a recipe: it is the metadata or nothing."""
        from spyde.multiangle.model import model_from_metadata

        path = self._saved_stack(tmp_path)
        back = hs.load(str(path), lazy=True)
        from spyde.multiangle.recipe import recipe_for
        assert recipe_for(back) is None, \
            "a file should not carry a recipe; this test proves nothing"
        model = model_from_metadata(back)
        assert model is not None, "the model did not survive the file"
        assert model.n_members == 4
        assert list(model.azimuths) == [0.0, 90.0, 180.0, 270.0]

    def test_the_ring_opens_on_a_reopened_acquisition(self, tmp_path):
        from spyde.actions.multiangle_navigator import (
            multiangle_model, open_multiangle_navigator, stack_signal,
        )

        path = self._saved_stack(tmp_path)
        session = make_session()
        try:
            tree = open_saved(session, path)

            assert multiangle_model(tree) is not None, (
                "the tree has no model, so the ring will refuse to draw for "
                "a multi-angle acquisition")
            assert stack_signal(tree) is not None, "no angle-axis node found"
            assert open_multiangle_navigator(session, tree) is not None, \
                "the angle ring did not open"
        finally:
            close_session(session)


class TestRebinSumsIntoTheNarrowestDtype:
    """hyperspy left to itself sums a uint16 into uint64 — four times the
    bytes that were asked for, and a width nothing downstream can widen for
    a further sum. Sixteen uint16 pixels fit uint32 exactly."""

    def test_the_kwargs_name_the_dtype(self):
        from spyde.actions.base import _rebin_dtype

        assert _rebin_dtype(np.uint16, [2, 2, 2, 2]) == np.dtype(np.uint32)
        assert _rebin_dtype(np.uint16, [1, 1, 2, 2]) == np.dtype(np.uint32)
        assert _rebin_dtype(np.float32, [2, 2, 2, 2]) == np.dtype(np.float32)
        assert _rebin_dtype(np.bool_, [2, 2, 2, 2]) is None, \
            "a dtype with no sum is left to hyperspy"

    def test_a_full_scale_scan_does_not_overflow(self):
        from spyde.actions.base import Rebin2DAction
        from spyde.actions.context import ActionContext

        data = np.full((4, 4, 4, 4), np.iinfo(np.uint16).max, dtype=np.uint16)
        session = make_session()
        signal = hs.signals.Signal2D(data)
        signal.set_signal_type("electron_diffraction")
        session._add_signal(signal)
        try:
            plot = next(p for p in session._plots
                        if not getattr(p, "is_navigator", False)
                        and getattr(getattr(p, "plot_state", None),
                                    "current_signal", None) is not None)
            params = {"scale_x": 2, "scale_y": 2, "scan_x": 2, "scan_y": 2}
            new = Rebin2DAction(ActionContext(
                plot=plot, params=params, action_name="Rebin")).run(**params)
            assert new.data.dtype == np.dtype(np.uint32), (
                f"rebin produced {new.data.dtype}; want uint32")
            assert int(np.asarray(new.data).max()) == 16 * 65535
        finally:
            close_session(session)


class TestAWideStackStillReopensWhole:
    """The failure exactly as it happened: a stack rebinned by the app was
    uint64, and the reopen fell back to a plain dataset — silently, because
    the only word of it went to the log."""

    def _stack(self, dtype):
        from spyde.multiangle.synthetic import saved_stack

        signal, data = saved_stack(shells=(0, 0, 0, 0), seed=11, dtype=dtype)
        signal.metadata.set_item(
            "Acquisition.multiangle.member_signal_type", "electron_diffraction")
        return signal, data

    def _open(self, session, path, needs_summed=True):
        return open_saved(session, path, needs_summed=needs_summed)

    def test_a_uint64_stack_reopens_as_a_tree(self, tmp_path):
        stack, data = self._stack(np.uint64)
        path = tmp_path / "wide.zspy"
        stack.save(str(path))
        session = make_session()
        try:
            tree = self._open(session, path)
            assert tree.root_node.name == "Aligned Stack", (
                f"the root is {tree.root_node.name!r}: a uint64 stack "
                "reopened as a plain dataset")
            summed = tree.root_node.children["Summed"].signal
            assert summed.data.dtype == np.dtype(np.uint64)
            assert np.array_equal(np.asarray(summed.data), data.sum(axis=0))
        finally:
            close_session(session)

    def test_the_summed_node_reads_through_the_stack(self, tmp_path):
        """One store reader for all the planes, not a lazy slice per member:
        measured 25 ms against 55-80 ms per member on a real stack."""
        from spyde.array_cache.readers.multiangle import build_multiangle_reader
        from spyde.multiangle.recipe import recipe_for

        stack, data = self._stack(np.uint16)
        path = tmp_path / "planes.zspy"
        stack.save(str(path))
        session = make_session()
        try:
            tree = self._open(session, path)
            root = tree.root_node.signal
            summed = tree.root_node.children["Summed"].signal
            assert recipe_for(summed).stack is root
            reader = build_multiangle_reader(summed, summed.data)
            assert reader is not None
            assert reader._stack_reader is not None, \
                "the planes were not read through the stack's own reader"
            assert all(member is None for member in reader._member_readers)
            for row in range(data.shape[1]):
                for column in range(data.shape[2]):
                    assert np.array_equal(
                        reader.read_frame((row, column)),
                        data[:, row, column].astype(np.uint32).sum(axis=0))
        finally:
            close_session(session)

    def test_a_failed_re_expansion_is_said_in_the_app(self, tmp_path,
                                                      monkeypatch):
        import spyde.backend._session_files as files
        import spyde.backend._session_multiangle as multiangle

        stack, _data = self._stack(np.uint16)
        path = tmp_path / "broken.zspy"
        stack.save(str(path))
        errors = []
        monkeypatch.setattr(files, "emit_error", errors.append)
        monkeypatch.setattr(multiangle, "rebuild_multiangle_tree",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("no accumulator")))
        session = make_session()
        try:
            tree = self._open(session, path, needs_summed=False)
            assert tree.root_node.name != "Aligned Stack"
            assert any("multi-angle" in message and "no accumulator" in message
                       for message in errors), errors
        finally:
            close_session(session)


class TestASavedFileIsNotChunkedInSlivers:
    """A scan cropped from mid-chunk was written in chunks two positions
    wide, because the writer takes each axis's FIRST dask block as the file's
    chunk size and a crop origin makes that block a remainder. Every third
    step of a drag on the saved file then decoded a chunk."""

    def test_the_largest_block_is_the_chunk(self):
        from spyde.backend._session_files import _uniform_save_chunks

        signal = hs.signals.Signal2D(np.zeros((30, 30, 4, 4), dtype=np.uint16)).as_lazy()
        signal.data = signal.data.rechunk((10, 10, 4, 4))
        assert _uniform_save_chunks(signal) is None,             "a regular grid is left to the writer"
        cropped = signal.inav[6:, :]
        assert cropped.data.chunks[1][0] == 4, cropped.data.chunks
        assert _uniform_save_chunks(cropped) == (10, 10, 4, 4)

    def test_the_file_gets_it(self, tmp_path):
        session = make_session()
        try:
            data = np.arange(30 * 30 * 4 * 4, dtype=np.uint16).reshape(30, 30, 4, 4)
            signal = hs.signals.Signal2D(data).as_lazy()
            signal.data = signal.data.rechunk((10, 10, 4, 4))
            cropped = signal.inav[6:, 3:]
            path = tmp_path / "cropped.zspy"
            session._save_signal_thread(cropped, str(path), "cropped")
            back = hs.load(str(path), lazy=True)
            assert back.data.chunksize == (10, 10, 4, 4), back.data.chunksize
            assert np.array_equal(np.asarray(back.data), data[3:, 6:])
        finally:
            close_session(session)


class TestASumIsNeverAStack:
    """A 4-D sum inherited the stack's type when the members carried none,
    was recognised as a stack on reopening, and the rebuild raised on it —
    which, now that a failed rebuild is an error in the app, told the user
    their ordinary dataset could not be re-expanded."""

    def _stack(self):
        """Members that carried no type: the record has no member type and
        the stack has the multi-angle one, the case the sum inherited."""
        from spyde.multiangle.synthetic import saved_stack
        from spyde.signals.multiangle import MULTIANGLE_SIGNAL_TYPE

        signal, _data = saved_stack(members=3, shells=(0, 0, 0),
                                    scan_shape=(4, 5), detector_shape=(6, 6),
                                    fill=1, signal_type="")
        signal.set_signal_type(MULTIANGLE_SIGNAL_TYPE)
        return signal

    def test_a_4d_signal_with_the_type_is_not_a_stack(self):
        from spyde.signals.multiangle import MULTIANGLE_SIGNAL_TYPE, is_multiangle_stack

        stack = self._stack()
        assert is_multiangle_stack(stack)
        summed = hs.signals.Signal2D(np.ones((4, 5, 6, 6), dtype=np.uint32))
        summed.metadata.set_item("Acquisition.multiangle",
                                 stack.metadata.get_item(
                                     "Acquisition.multiangle").as_dictionary())
        summed.set_signal_type(MULTIANGLE_SIGNAL_TYPE)
        assert not is_multiangle_stack(summed)

    def test_the_sum_of_typeless_members_is_diffraction(self):
        from spyde.backend._session_multiangle import _sum_over_angles

        summed = _sum_over_angles(self._stack(), range(3), "Summed")
        assert summed.metadata.Signal.signal_type == "electron_diffraction"
        assert summed.data.dtype == np.dtype(np.uint32), \
            "three uint16 members sum into uint32, not numpy's uint64"


class TestRebinRecordsTheReduction:
    def test_binned_by_is_cumulative(self):
        from spyde.actions.base import Rebin2DAction
        from spyde.actions.context import ActionContext
        from spyde.signals.multiangle import MULTIANGLE_METADATA

        session = make_session()
        signal = hs.signals.Signal2D(np.ones((2, 8, 8, 8, 8), dtype=np.uint16))
        signal.metadata.set_item(MULTIANGLE_METADATA, {
            "n_members": 2, "nav_offsets": [[0, 0], [3, -2]]})
        signal.set_signal_type("electron_diffraction")
        session._add_signal(signal)
        try:
            plot = next(p for p in session._plots
                        if not getattr(p, "is_navigator", False)
                        and getattr(getattr(p, "plot_state", None),
                                    "current_signal", None) is not None)
            params = {"scale_x": 2, "scale_y": 2, "scan_x": 2, "scan_y": 1}
            once = Rebin2DAction(ActionContext(
                plot=plot, params=params, action_name="Rebin")).run(**params)
            recorded = once.metadata.get_item(
                f"{MULTIANGLE_METADATA}.binned_by").as_dictionary()
            assert recorded == {"scan": [1, 2], "detector": [2, 2]}, recorded
            assert once.metadata.get_item(
                f"{MULTIANGLE_METADATA}.nav_offsets") == [[0, 0], [3, -2]], \
                "the offsets stay in the composition's pixels"
            from spyde.actions.lifecycle import show_tree_node
            show_tree_node(plot, session.signal_trees[0], once)
            twice = Rebin2DAction(ActionContext(
                plot=plot, params=params, action_name="Rebin")).run(**params)
            recorded = twice.metadata.get_item(
                f"{MULTIANGLE_METADATA}.binned_by").as_dictionary()
            assert recorded == {"scan": [1, 4], "detector": [4, 4]}, recorded
        finally:
            close_session(session)


class TestAReopenedTreeIsTheComposedOne:
    """Whatever the compose path gives, the reopen path gives too."""

    def _saved(self, tmp_path, planes=True, shells=(0, 0, 1, 1)):
        from spyde.multiangle.synthetic import saved_stack

        signal, data = saved_stack(
            shells=shells, scan_shape=(6, 8), seed=5, planes=planes,
            nav_offsets=[[0, 0], [1, -1], [0, 2], [-1, 0]])
        path = tmp_path / "acquisition.zspy"
        signal.save(str(path))
        return path, data

    def _open(self, session, path, needs_summed=True):
        return open_saved(session, path, needs_summed=needs_summed)

    def test_the_model_reports_the_recorded_offsets(self, tmp_path):
        from spyde.actions.multiangle_navigator import multiangle_model

        path, _data = self._saved(tmp_path)
        session = make_session()
        try:
            tree = self._open(session, path)
            model = multiangle_model(tree)
            assert model.nav_offsets.tolist() == [[0, 0], [1, -1], [0, 2], [-1, 0]], (
                "the model answered with the Summed node's zeroed offsets")
        finally:
            close_session(session)

    def test_the_shells_carry_the_composed_names(self, tmp_path):
        path, _data = self._saved(tmp_path)
        session = make_session()
        try:
            tree = self._open(session, path)
            summed = tree.root_node.children["Summed"]
            assert set(summed.children) == {"Summed 0.5°", "Summed 1°"}, \
                set(summed.children)
        finally:
            close_session(session)

    def test_the_stack_is_brought_up_to_the_current_type(self, tmp_path):
        from spyde.signals.multiangle import MULTIANGLE_METADATA, MULTIANGLE_SIGNAL_TYPE

        path, _data = self._saved(tmp_path)
        session = make_session()
        try:
            tree = self._open(session, path)
            root = tree.root_node.signal
            assert root.metadata.Signal.signal_type == MULTIANGLE_SIGNAL_TYPE
            assert root.metadata.get_item(
                f"{MULTIANGLE_METADATA}.member_signal_type") == "electron_diffraction"
            summed = tree.root_node.children["Summed"].signal
            assert summed.metadata.Signal.signal_type == "electron_diffraction"
        finally:
            close_session(session)

    def test_the_recorded_planes_are_the_navigator(self, tmp_path, monkeypatch):
        from spyde.backend.session import Session

        path, data = self._saved(tmp_path)
        seen = {}
        original = Session._add_signal

        def spy(self, signal, *args, **kwargs):
            seen["override"] = kwargs.get("navigator_override")
            return original(self, signal, *args, **kwargs)

        monkeypatch.setattr(Session, "_add_signal", spy)
        session = make_session()
        try:
            self._open(session, path)
            override = seen.get("override")
            assert override is not None, "the tree reduced the whole stack again"
            assert np.allclose(np.asarray(override.data),
                               data.sum(axis=(3, 4)).astype(np.float32))
        finally:
            close_session(session)

    def test_a_failure_after_the_tree_exists_does_not_open_it_twice(
            self, tmp_path, monkeypatch):
        import spyde.backend._session_multiangle as multiangle

        path, _data = self._saved(tmp_path)
        monkeypatch.setattr(multiangle, "_attach_node",
                            lambda *a, **k: (_ for _ in ()).throw(
                                ValueError("attach failed")))
        session = make_session()
        try:
            self._open(session, path, needs_summed=False)
            time.sleep(1.0)
            assert len(session.signal_trees) == 1, (
                f"{len(session.signal_trees)} trees for one file")
        finally:
            close_session(session)


class TestRebinRefusesByName:
    def test_the_last_detector_axis_is_checked_on_a_stack(self):
        from spyde.actions.base import Rebin2DAction
        from spyde.actions.context import ActionContext

        session = make_session()
        signal = hs.signals.Signal2D(np.ones((2, 4, 4, 7, 8), dtype=np.uint16))
        signal.set_signal_type("electron_diffraction")
        session._add_signal(signal)
        try:
            plot = next(p for p in session._plots
                        if not getattr(p, "is_navigator", False)
                        and getattr(getattr(p, "plot_state", None),
                                    "current_signal", None) is not None)
            params = {"scale_x": 2, "scale_y": 2, "scan_x": 1, "scan_y": 1}
            with pytest.raises(RuntimeError, match="detector .* is 7 px"):
                Rebin2DAction(ActionContext(
                    plot=plot, params=params, action_name="Rebin")).run(**params)
        finally:
            close_session(session)


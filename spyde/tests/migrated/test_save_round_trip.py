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

from spyde.tests.migrated.conftest import close_session, make_session


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
        for key in ("x0", "x1", "y0", "y1",
                    "scan_x0", "scan_x1", "scan_y0", "scan_y1"):
            assert key in accepted, f"_crop_signal does not take {key}"
            assert key in caret, f"the caret never mentions {key}"


class TestAMultiAngleAcquisitionReopensWhole:
    """Saved and reopened, a composed acquisition is its TREE again.

    Otherwise it is a 5-D array: the alignment survives and everything built
    on it does not — no Summed node, no shells, no angle ring — and a dataset
    published that way hands the next person an array with no way to know what
    it is. Nothing extra is needed to rebuild them: the sums are reductions
    over the stack's own leading axis and the angles are in its metadata.
    """

    def _stack(self, members=4, shells=(0, 0, 1, 1)):
        from spyde.signals.multiangle import MULTIANGLE_METADATA

        generator = np.random.default_rng(7)
        data = generator.integers(
            0, 400, (members, 5, 6, 4, 4), dtype=np.uint16)
        signal = hs.signals.Signal2D(data)
        signal.metadata.set_item(MULTIANGLE_METADATA, {
            "n_members": members,
            "n_shells": len(set(shells)),
            "tilts": [1.0 if shell else 0.5 for shell in shells],
            "azimuths": [i * 90.0 for i in range(members)],
            "shell_ids": list(shells),
            "reference": 0,
            "member_signal_type": "electron_diffraction",
        })
        signal.set_signal_type("electron_diffraction")
        return signal, np.asarray(data)

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
            session.open_file(str(path))
            deadline = time.time() + 60.0
            while time.time() < deadline and not session.signal_trees:
                time.sleep(0.2)
            assert session.signal_trees, "the file never opened"
            tree = session.signal_trees[0]
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
        """Without the shape check: a file may be opened lazily in pieces, and
        the type is the cheapest thing that identifies one."""
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

        data = np.random.default_rng(7).integers(
            0, 400, (4, 5, 6, 4, 4), dtype=np.uint16)
        signal = hs.signals.Signal2D(data)
        signal.metadata.set_item(MULTIANGLE_METADATA, {
            "n_members": 4, "n_shells": 2, "tilts": [0.5, 0.5, 1.0, 1.0],
            "azimuths": [0.0, 90.0, 180.0, 270.0], "shell_ids": [0, 0, 1, 1],
            "reference": 0, "nav_offsets": [[0, 0]] * 4,
            "dp_offsets": [[0, 0]] * 4, "paths": ["a", "b", "c", "d"],
            "member_signal_type": "electron_diffraction"})
        signal.set_signal_type(MULTIANGLE_SIGNAL_TYPE)
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
            session.open_file(str(path))
            deadline = time.time() + 60.0
            while time.time() < deadline and not session.signal_trees:
                time.sleep(0.2)
            assert session.signal_trees, "the file never opened"
            tree = session.signal_trees[0]

            assert multiangle_model(tree) is not None, (
                "the tree has no model, so the ring will refuse to draw for "
                "a multi-angle acquisition")
            assert stack_signal(tree) is not None, "no angle-axis node found"
            assert open_multiangle_navigator(session, tree) is not None, \
                "the angle ring did not open"
        finally:
            close_session(session)

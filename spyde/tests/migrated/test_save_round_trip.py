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
            back = _saved(session, tmp_path / "cropped.zspy",
                          _plots_of(session)[-1])
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

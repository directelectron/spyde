"""
Signal-type selector (right sidebar) — re-added from the Qt version.

Loading a signal emits `signal_type_info` (current type + selectable options) to
the sidebar; the dropdown's `set_signal_type` action re-casts the signal's
HyperSpy class and re-broadcasts the dependent panels.
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs
from spyde.tests.migrated.conftest import _settle, make_session


def _make_session():
    return make_session()


def _msgs(ms, t):
    return [m for m in ms if m.get("type") == t]


class TestSignalType:
    def test_load_emits_signal_type_info(self, captured_messages, monkeypatch):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((8, 8), dtype=np.float32))
            session._add_signal(s, source_path=None)
            _settle(session)
            info = _msgs(captured_messages, "signal_type_info")
            assert info, "no signal_type_info emitted on load"
            assert info[-1]["current"] == ""          # generic Signal2D
            assert "electron_diffraction" in info[-1]["options"]
        finally:
            session.shutdown()

    def test_set_signal_type_recasts_and_reemits(self, captured_messages, monkeypatch):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), dtype=np.float32))
            session._add_signal(s, source_path=None)
            _settle(session)
            captured_messages.clear()

            plot = session._plots[-1]
            session._set_signal_type(plot, "electron_diffraction")
            _settle(session)

            tree = session.signal_trees[-1]
            assert tree.root.metadata.get_item("Signal.signal_type") == "electron_diffraction"
            info = _msgs(captured_messages, "signal_type_info")
            assert info and info[-1]["current"] == "electron_diffraction", \
                "signal_type_info not re-emitted with the new type"
        finally:
            session.shutdown()

    def test_set_empty_type_reverts_to_generic(self, captured_messages, monkeypatch):
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            s = hs.signals.Signal2D(np.zeros((4, 4, 8, 8), dtype=np.float32))
            s.set_signal_type("electron_diffraction")
            session._add_signal(s, source_path=None)
            _settle(session)

            plot = session._plots[-1]
            session._set_signal_type(plot, "")
            _settle(session)
            assert (session.signal_trees[-1].root.metadata.get_item(
                "Signal.signal_type", default="") or "") == ""
        finally:
            session.shutdown()


class TestSignalTypeFromNavigator:
    """The dropdown acts on the dataset whichever of its windows has focus."""

    def test_set_from_navigator_recasts_the_dataset(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        messages = stem_4d_dataset["messages"]
        tree = session.signal_trees[-1]
        navigator = next(p for p in session._plots
                         if p.signal_tree is tree and p.is_navigator)
        navigator_image = navigator.plot_state.current_signal
        navigator_type = navigator_image.metadata.get_item(
            "Signal.signal_type", default="")
        messages.clear()

        session._set_signal_type(navigator, "")
        _settle(session)

        assert (tree.root.metadata.get_item("Signal.signal_type", default="")
                or "") == "", "the dataset kept its type"
        assert navigator_image.metadata.get_item(
            "Signal.signal_type", default="") == navigator_type, \
            "the navigator image was re-typed instead of the dataset"
        info = _msgs(messages, "signal_type_info")
        assert info and info[-1]["current"] == ""

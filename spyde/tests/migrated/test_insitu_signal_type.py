"""
InSitu signal type + Play/Fast Forward toolbar gating.

An in-situ movie (1-D time navigation, 2-D image signal) is tagged with the
``insitu`` HyperSpy signal type (spyde/signals/insitu.py) so the movie
playback controls (Play / Fast Forward — spyde/toolbars.yaml) only appear for
signals explicitly typed that way, not for every 1-D navigator (e.g. a
line-scan navigator carved out of a bigger dataset).

The subtlety: Play/Fast Forward are ``navigation: True`` — they gate on the
NAVIGATOR plot, whose own displayed signal is a DERIVED trace (the root
summed over its signal axes, see BaseSignalTree._initialize_navigator), not
the movie itself. plot_control_toolbar._gate_signal_type resolves the gate
against the signal TREE ROOT's ``_signal_type`` for navigation-only actions,
so the derived trace's own (generic) signal type never matters for this gate.
"""
from __future__ import annotations

import numpy as np
import hyperspy.api as hs

from spyde.signals.insitu import InSitu, LazyInSitu
from spyde.drawing.toolbars.plot_control_toolbar import get_toolbar_actions_for_plot


class TestInSituSignalType:
    """set_signal_type("insitu") round-trips to the InSitu/LazyInSitu classes."""

    def test_eager_round_trip(self):
        s = hs.signals.Signal2D(np.zeros((5, 8, 8), dtype=np.float32))
        s.set_signal_type("insitu")
        assert isinstance(s, InSitu)
        assert s._signal_type == "insitu"

    def test_lazy_round_trip(self):
        s = hs.signals.Signal2D(np.zeros((5, 8, 8), dtype=np.float32)).as_lazy()
        s.set_signal_type("insitu")
        assert isinstance(s, LazyInSitu)
        assert s._signal_type == "insitu"


class TestTestHarnessMovieLoaderTypesInSitu:
    """The synthetic movie loader (load_test_data_movie) types its root insitu."""

    def test_movie_root_is_insitu(self, window):
        session = window["window"]
        session.dispatch_action(
            {"action": "load_test_data_movie", "payload": {"size": 64, "frames": 3}}
        )
        import time
        time.sleep(0.5)
        assert len(session.signal_trees) == 1
        tree = session.signal_trees[0]
        assert tree.root._signal_type == "insitu"
        assert isinstance(tree.root, (InSitu, LazyInSitu))


class TestPlaybackToolbarGating:
    """Play / Fast Forward show up only on an insitu-typed 1-D navigator."""

    @staticmethod
    def _navigator_plot(session):
        return next((p for p in session._plots
                     if p.is_navigator and p.plot_state is not None), None)

    def test_movie_navigator_includes_playback_buttons(self, movie_dataset):
        session = movie_dataset["window"]
        tree = session.signal_trees[0]
        assert tree.root._signal_type == "insitu", \
            "movie_dataset fixture root should be insitu-typed"
        nav_plot = self._navigator_plot(session)
        assert nav_plot is not None, "movie fixture should have a navigator plot"
        names = get_toolbar_actions_for_plot(nav_plot.plot_state)[2]
        assert "Play" in names, f"Play missing from navigator toolbar: {names}"
        assert "Fast Forward" in names, \
            f"Fast Forward missing from navigator toolbar: {names}"

    def test_4d_stem_navigator_excludes_playback_buttons(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        assert tree.root._signal_type != "insitu"
        nav_plot = self._navigator_plot(session)
        assert nav_plot is not None, "4D-STEM fixture should have a navigator plot"
        names = get_toolbar_actions_for_plot(nav_plot.plot_state)[2]
        assert "Play" not in names, f"Play should be absent: {names}"
        assert "Fast Forward" not in names, f"Fast Forward should be absent: {names}"

    def test_plain_1d_nav_signal_excludes_playback_buttons(self, window):
        """A generic (non-insitu) 1-D-navigation signal must NOT get Play/Fast
        Forward just because its navigator is 1-D — the gate is on signal_type,
        not on navigator dimensionality."""
        session = window["window"]
        data = np.zeros((6, 16, 16), dtype=np.float32)
        s = hs.signals.Signal2D(data)   # nav-dim 1, signal-dim 2, NOT insitu
        assert s._signal_type != "insitu"
        session._add_signal(s, source_path=None)
        import time
        time.sleep(0.5)
        nav_plot = self._navigator_plot(session)
        assert nav_plot is not None, "1-D-nav fixture should have a navigator plot"
        names = get_toolbar_actions_for_plot(nav_plot.plot_state)[2]
        assert "Play" not in names, f"Play should be absent: {names}"
        assert "Fast Forward" not in names, f"Fast Forward should be absent: {names}"

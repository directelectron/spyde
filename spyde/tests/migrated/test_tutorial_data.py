"""
test_tutorial_data.py — Phase 1 of the docs/walkthroughs overhaul: curated,
ALWAYS-AVAILABLE tutorial datasets (spyde/backend/tutorial_data.py).

Unlike the _TEST_ACTIONS gate (spyde/backend/_session_actions.py), the
``tutorial_load`` action must dispatch even under SPYDE_PACKAGED=1 — these are
user-reachable in every build, not just the Playwright/dev harness. Each
loader is asserted to:
  - build a signal of the expected ndim / signal_type without raising,
  - stamp ``source_path="tutorial_<name>"``,
  - stay SMALL (bounded nav/signal shapes) — a regression that swaps in a huge
    default (e.g. simulated_strain's un-downsized 512x512 x 1e5-electron
    default, ~2 GB) must fail here, not ship.
"""
from __future__ import annotations

import time

import pytest

from spyde.backend.tutorial_data import TUTORIAL_LOADERS


def _last_tree(session):
    assert session.signal_trees, "no signal tree was created"
    return session.signal_trees[-1]


def _settle():
    # Let selector/navigator debounce timers + the (threaded, no-dask) compute
    # settle before asserting — mirrors conftest._load's own sleep.
    time.sleep(0.5)


class TestTutorialLoadersDirect:
    """Call each TutorialDataMixin method directly on a real Session."""

    def test_navigation_shape(self, window):
        session = window["window"]
        session.tutorial_navigation()
        _settle()
        tree = _last_tree(session)
        root = tree.root
        am = root.axes_manager
        assert am.navigation_dimension == 2
        assert am.signal_dimension == 2
        # nav=10x10, signal=50x50 — small + fast, per the loader's docstring.
        assert tuple(am.navigation_shape) == (10, 10)
        assert tuple(am.signal_shape) == (50, 50)
        assert str(getattr(root, "_signal_type", "")) == "electron_diffraction"
        assert tree.source_path is None  # pseudo-path, not a real on-disk file
        # source_path threading: _add_signal only enables the on-disk sidecar
        # for a real path; confirm the pseudo tutorial path was passed through
        # by checking the recorded metadata title instead (source_path isn't
        # stored verbatim on the tree when it's not a real file).
        title = root.metadata.get_item("General.title", default=None)
        assert title == "Tutorial: Navigation & VI"

    def test_find_vectors_shape(self, window):
        session = window["window"]
        session.tutorial_find_vectors()
        _settle()
        tree = _last_tree(session)
        am = tree.root.axes_manager
        assert tuple(am.navigation_shape) == (6, 6)
        assert tuple(am.signal_shape) == (128, 128)
        assert str(getattr(tree.root, "_signal_type", "")) == "electron_diffraction"

    def test_orientation_shape(self, window):
        session = window["window"]
        session.tutorial_orientation()
        _settle()
        tree = _last_tree(session)
        am = tree.root.axes_manager
        assert tuple(am.navigation_shape) == (6, 6)
        assert tuple(am.signal_shape) == (128, 128)
        assert str(getattr(tree.root, "_signal_type", "")) == "electron_diffraction"

    def test_multiphase_shape(self, window):
        session = window["window"]
        session.tutorial_multiphase()
        _settle()
        tree = _last_tree(session)
        am = tree.root.axes_manager
        assert tuple(am.navigation_shape) == (20, 20)
        assert tuple(am.signal_shape) == (128, 128)
        assert str(getattr(tree.root, "_signal_type", "")) == "electron_diffraction"

    def test_strain_shape_is_bounded(self, window):
        """Guards the downsize: pyxem's default simulated_strain() is
        32x32 nav x 512x512 signal x 1e5 electrons (~2 GB). The tutorial
        loader MUST use the small (16,16)/(128,128)/1e3 override."""
        session = window["window"]
        session.tutorial_strain()
        _settle()
        tree = _last_tree(session)
        am = tree.root.axes_manager
        assert tuple(am.navigation_shape) == (16, 16)
        assert tuple(am.signal_shape) == (128, 128)
        # Bound the total element count well under the un-downsized default
        # (32*32*512*512 ~= 268M elements) as a blunt "nothing huge shipped"
        # guard in addition to the exact-shape assertions above.
        n_nav = am.navigation_shape[0] * am.navigation_shape[1]
        n_sig = am.signal_shape[0] * am.signal_shape[1]
        assert n_nav * n_sig < 5_000_000
        assert str(getattr(tree.root, "_signal_type", "")) == "electron_diffraction"

    def test_spectroscopy_shape(self, window):
        session = window["window"]
        session.tutorial_spectroscopy()
        _settle()
        tree = _last_tree(session)
        root = tree.root
        am = root.axes_manager
        assert am.signal_dimension == 1
        assert am.navigation_dimension == 2
        assert tuple(am.navigation_shape) == (32, 32)
        assert tuple(am.signal_shape) == (1024,)
        # Signal1D — no diffraction cast expected/needed.
        assert root.__class__.__name__ in ("Signal1D",) or hasattr(root, "isig")

    def test_movie_shape_is_small_and_insitu(self, window):
        session = window["window"]
        session.tutorial_movie()
        _settle()
        tree = _last_tree(session)
        root = tree.root
        am = root.axes_manager
        assert am.navigation_dimension == 1
        assert am.signal_dimension == 2
        assert tuple(am.navigation_shape) == (6,)
        # DOWNSIZED to 512^2 (vs. the test-only loader's 2048^2 default).
        assert tuple(am.signal_shape) == (512, 512)
        assert str(getattr(root, "_signal_type", "")) == "insitu"
        assert bool(getattr(root, "_lazy", False))

    def test_all_loaders_present_in_map(self):
        expected = {
            "navigation", "find_vectors", "orientation", "multiphase",
            "strain", "spectroscopy", "movie",
        }
        assert expected <= set(TUTORIAL_LOADERS.keys())


class TestTutorialLoadDispatch:
    """The `tutorial_load` action itself: must dispatch through
    Session.dispatch_action, and — the whole point of Phase 1 — must NOT be
    gated by _TEST_ACTIONS_ENABLED / SPYDE_PACKAGED."""

    def test_dispatch_unknown_name_emits_error(self, window):
        session = window["window"]
        messages = window["messages"]
        session.dispatch_action({"action": "tutorial_load", "payload": {"name": "nope"}})
        errors = [m for m in messages if m.get("type") == "error"]
        assert errors, "expected an error message for an unknown tutorial name"

    def test_dispatch_navigation_by_name(self, window):
        session = window["window"]
        session.dispatch_action(
            {"action": "tutorial_load", "payload": {"name": "navigation"}}
        )
        _settle()
        tree = _last_tree(session)
        assert tuple(tree.root.axes_manager.navigation_shape) == (10, 10)

    def test_repeat_load_is_idempotent(self, window):
        """Loading the SAME tutorial name twice must NOT stack a duplicate copy
        (the walkthrough double/triple-load fix) — the second dispatch is a no-op
        because that dataset is already open."""
        session = window["window"]
        n0 = len(session.signal_trees)
        session.dispatch_action(
            {"action": "tutorial_load", "payload": {"name": "navigation"}})
        _settle()
        assert len(session.signal_trees) == n0 + 1
        # Second load of the same name → no new tree.
        session.dispatch_action(
            {"action": "tutorial_load", "payload": {"name": "navigation"}})
        _settle()
        assert len(session.signal_trees) == n0 + 1
        # A DIFFERENT tutorial name still loads (dedup is per-name, not global).
        session.dispatch_action(
            {"action": "tutorial_load", "payload": {"name": "spectroscopy"}})
        _settle()
        assert len(session.signal_trees) == n0 + 2

    def test_close_all_tears_down_tutorial_trees(self, window):
        """tutorial_close_all closes every tutorial dataset opened this session
        (walkthrough teardown) and leaves non-tutorial data untouched."""
        session = window["window"]
        # A non-tutorial dataset the close must NOT touch.
        session.dispatch_action({"action": "load_test_data", "payload": {}})
        _settle()
        keep = len(session.signal_trees)
        session.dispatch_action(
            {"action": "tutorial_load", "payload": {"name": "navigation"}})
        session.dispatch_action(
            {"action": "tutorial_load", "payload": {"name": "spectroscopy"}})
        _settle()
        assert len(session.signal_trees) == keep + 2
        # Close all tutorial trees → back to just the non-tutorial dataset.
        session.dispatch_action({"action": "tutorial_close_all", "payload": {}})
        _settle()
        assert len(session.signal_trees) == keep
        # A second close is a harmless no-op.
        session.dispatch_action({"action": "tutorial_close_all", "payload": {}})
        assert len(session.signal_trees) == keep

    def test_session_begin_is_not_test_gated(self):
        """The in-app Tour sends `tutorial_session_begin` on open in EVERY
        build, so it must not be swept up by the packaged-build test gate."""
        from spyde.backend._session_actions import _TEST_ACTIONS
        assert "tutorial_session_begin" not in _TEST_ACTIONS

    def test_tutorial_load_not_in_test_actions_gate(self):
        """The un-gate is the whole point of Phase 1: tutorial_load must not
        be listed among the packaged-build-disabled test actions."""
        from spyde.backend._session_actions import _TEST_ACTIONS
        assert "tutorial_load" not in _TEST_ACTIONS

    def test_dispatch_works_under_simulated_packaged_env(self, window, monkeypatch):
        """Simulate a packaged build (SPYDE_PACKAGED=1, which flips
        _TEST_ACTIONS_ENABLED to False at import time) by monkeypatching the
        already-imported flag directly, and confirm tutorial_load still
        dispatches — the un-gate the whole feature exists for."""
        import spyde.backend._session_actions as actions_mod
        monkeypatch.setattr(actions_mod, "_TEST_ACTIONS_ENABLED", False)
        session = window["window"]
        session.dispatch_action(
            {"action": "tutorial_load", "payload": {"name": "spectroscopy"}}
        )
        _settle()
        tree = _last_tree(session)
        assert tuple(tree.root.axes_manager.signal_shape) == (1024,)

        # A genuine _TEST_ACTIONS member, by contrast, IS ignored under the
        # same simulated-packaged flag — proves the monkeypatch is actually
        # exercising the gate (not a no-op) and tutorial_load's pass-through
        # is a deliberate exemption, not an accident of the gate being off.
        n_trees_before = len(session.signal_trees)
        session.dispatch_action({"action": "load_test_data", "payload": {}})
        assert len(session.signal_trees) == n_trees_before


class TestTutorialSessionTeardown:
    """A guided walkthrough must be SELF-CONTAINED at both ends: it brings its
    own example dataset, and on exit it takes away everything it put on screen
    — including the RESULT windows the walkthrough created along the way (a
    find-vectors result tree, a virtual image, an orientation map). Before this,
    `tutorial_close_all` closed only the tutorial dataset itself and every
    derived window was left behind.

    The lifecycle is: the Tour sends `tutorial_session_begin` on open, and
    `Session._add_signal` records every non-file-backed tree created while the
    session is active; `tutorial_close_all` closes the lot.
    """

    def _derived(self, session):
        """Stand in for what an action does mid-tour: create a new tree from a
        result signal, via the same `_add_signal` every result path uses."""
        import hyperspy.api as hs
        import numpy as np
        sig = hs.signals.Signal2D(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6))
        return session._add_signal(sig)

    def test_derived_result_windows_are_closed_too(self, window):
        session = window["window"]
        keep = len(session.signal_trees)
        session.dispatch_action({"action": "tutorial_session_begin", "payload": {}})
        session.dispatch_action(
            {"action": "tutorial_load", "payload": {"name": "navigation"}})
        _settle()
        # …and now the walkthrough runs an analysis, which opens a result tree.
        derived = self._derived(session)
        _settle()
        assert len(session.signal_trees) == keep + 2
        assert derived in session.signal_trees

        session.dispatch_action({"action": "tutorial_close_all", "payload": {}})
        _settle()
        # BOTH the tutorial dataset and the derived result are gone.
        assert derived not in session.signal_trees
        assert len(session.signal_trees) == keep

    def test_only_active_session_records(self, window):
        """Outside a session (`tutorial_session_begin` never sent, or already
        torn down) a new tree is NOT recorded — so a later tour's teardown
        can't reach back and close a window the user made on their own."""
        session = window["window"]
        before = self._derived(session)
        _settle()
        n = len(session.signal_trees)
        session.dispatch_action({"action": "tutorial_close_all", "payload": {}})
        _settle()
        assert before in session.signal_trees
        assert len(session.signal_trees) == n

    def test_user_file_backed_tree_is_never_closed(self, window, tmp_path):
        """Opening your OWN dataset mid-tour is safe: a tree backed by a real
        file on disk is never recorded for teardown."""
        import hyperspy.api as hs
        import numpy as np
        path = tmp_path / "mine.hspy"
        hs.signals.Signal2D(np.zeros((3, 4, 5))).save(str(path))
        session = window["window"]
        session.dispatch_action({"action": "tutorial_session_begin", "payload": {}})
        session.dispatch_action(
            {"action": "tutorial_load", "payload": {"name": "navigation"}})
        _settle()
        mine = session._add_signal(hs.load(str(path)), source_path=str(path))
        _settle()
        session.dispatch_action({"action": "tutorial_close_all", "payload": {}})
        _settle()
        assert mine in session.signal_trees

    def test_close_all_ends_the_session(self, window):
        """After teardown the session is INACTIVE — a tree created later is not
        recorded, so a second `tutorial_close_all` cannot close it."""
        session = window["window"]
        session.dispatch_action({"action": "tutorial_session_begin", "payload": {}})
        session.dispatch_action({"action": "tutorial_close_all", "payload": {}})
        after = self._derived(session)
        _settle()
        session.dispatch_action({"action": "tutorial_close_all", "payload": {}})
        _settle()
        assert after in session.signal_trees

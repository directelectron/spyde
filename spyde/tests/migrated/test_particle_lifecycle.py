"""
Teardown for everything this feature opens. Nothing may outlive its window.

The feature adds a lot of surfaces — a particle tree, the Drift Check window, the
dx/dy plot, the ROI preview, overlay widgets, navigator lanes, two wizards — and
each is a separate chance to leak. The mechanisms exist (``register_cancel``,
``replace_tree_attr``, ``own_window``, ``figure_registry``), but the presence of a
mechanism is not evidence it fires, so these tests close things and assert on what
is left.

**The list-drift hazard.** ``BaseSignalTree.close()`` tears down by iterating
hard-coded attribute NAME LISTS. A new wizard or result that nobody adds to those
lists is silently exempt — no error, no warning, just a controller that outlives
its tree. That already happened here: ``_seg_wizard`` / ``_drift_wizard`` were
absent from the wizard list, and ``particles`` / ``source_node`` / ``source_tree``
from the results list, so closing a particle tree kept a back-reference to the
source movie's lazy array alive and closing the source freed nothing.
:class:`TestCloseListsCoverThisFeature` exists to make the next omission fail
rather than leak.
"""
from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

import spyde.data.synthetic as sy
from spyde.actions import figure_registry
from spyde.actions.particle_tree import open_particle_tree
from spyde.particles import (
    LinkParams,
    link,
    measure_frame,
)
from spyde.signals.particles import SpyDEParticles
from spyde.tests.migrated._labels import labels_from

N_FRAMES = 5


@pytest.fixture(scope="module")
def parts():
    s = sy.particle_movie(n_frames=N_FRAMES)
    gt = sy.ground_truth(s)
    per_frame, contours = [], []
    for t in range(N_FRAMES):
        lab = labels_from(s.data[t], min_size=25, blur=1.0)
        rows, cs = measure_frame(lab, s.data[t], t=t, scale=float(gt["scale"]))
        per_frame.append(rows)
        contours.append(cs)
    p = SpyDEParticles.from_frames(
        per_frame, frame_shape=tuple(gt["frame_shape"]),
        contours_per_frame=contours, scale=float(gt["scale"]), units="nm")
    link(p, LinkParams(max_dist=10.0)).apply(p)
    return s, gt, p


class TestCloseListsCoverThisFeature:
    """Guards the hard-coded name lists in ``BaseSignalTree.close()``.

    These read the source rather than the behaviour on purpose: the failure mode
    is an attribute nobody REMEMBERED to list, and a behavioural test only catches
    the ones you thought to set.
    """

    def _close_src(self) -> str:
        import inspect

        from spyde import signal_tree
        src = inspect.getsource(signal_tree.BaseSignalTree.close)
        return src

    @pytest.mark.parametrize("attr", ["_seg_wizard", "_drift_wizard"])
    def test_new_wizards_are_torn_down(self, attr):
        assert attr in self._close_src(), (
            f"{attr} is not named in BaseSignalTree.close(), so a live wizard "
            "controller outlives its tree — silently, because close() iterates "
            "name lists and an unlisted attribute is simply skipped")

    @pytest.mark.parametrize("attr", [
        "particles", "_seg_pending_particles", "particle_events",
        "particle_edits", "nav_traces", "drift", "nav_map", "_seg_batch_running",
    ])
    def test_new_results_are_cleared(self, attr):
        # Match the quoted token, not a bare substring: "particles" is also a
        # substring of "_seg_pending_particles" and "drift" of "_drift_wizard",
        # so `attr in src` can pass even when `attr` itself is never listed.
        assert f'"{attr}"' in self._close_src(), f"{attr} survives tree.close()"

    @pytest.mark.parametrize("attr", ["source_node", "source_tree"])
    def test_back_references_to_the_source_are_cleared(self, attr):
        """The costly one: these hold the source movie's lazy array."""
        assert attr in self._close_src(), (
            f"{attr} survives tree.close(), so a particle tree pins the source "
            "movie's signal and closing the movie frees nothing")

    def test_particle_overlay_is_torn_down(self):
        assert "_particle_overlay" in self._close_src()


class TestParticleTreeTeardown:
    def test_close_clears_the_result_and_the_back_reference(self, window, parts):
        session = window["window"]
        s, _gt, p = parts
        tree = open_particle_tree(session, particles=p, source_node=s)
        assert tree.particles is p and tree.source_node is s
        tree.close()
        assert getattr(tree, "particles", None) is None
        assert getattr(tree, "source_node", None) is None
        assert getattr(tree, "source_tree", None) is None

    def test_close_drops_the_plots(self, window, parts):
        session = window["window"]
        s, _gt, p = parts
        tree = open_particle_tree(session, particles=p, source_node=s)
        tree.close()
        assert tree.signal_plots == []
        assert tree.navigator_plot_manager is None

    def test_the_source_signal_is_releasable_after_close(self, window, parts):
        """The leak this was really about, tested by weak reference.

        A tree that keeps `source_node` set pins the movie. Build a THROWAWAY
        source so the module fixture's own reference does not mask the result.
        """
        session = window["window"]
        _s, gt, p = parts
        throwaway = sy.particle_movie(n_frames=3)
        ref = weakref.ref(throwaway)
        tree = open_particle_tree(session, particles=p, source_node=throwaway)
        tree.close()
        del throwaway
        gc.collect()
        assert ref() is None, (
            "the source signal is still reachable after the particle tree was "
            "closed — something still holds source_node")

    def test_without_close_the_source_is_still_pinned(self, window, parts):
        """Non-vacuity for the test above.

        If this ALSO collected, the weakref test would be proving nothing about
        `close()` — it would just be showing that a local went out of scope.
        """
        session = window["window"]
        _s, _gt, p = parts
        throwaway = sy.particle_movie(n_frames=3)
        ref = weakref.ref(throwaway)
        tree = open_particle_tree(session, particles=p, source_node=throwaway)
        del throwaway
        gc.collect()
        assert ref() is not None, (
            "the source was collected without close() — so the companion test "
            "does not demonstrate that close() is what releases it")
        tree.close()

    def test_close_is_idempotent(self, window, parts):
        session = window["window"]
        s, _gt, p = parts
        tree = open_particle_tree(session, particles=p, source_node=s)
        tree.close()
        tree.close()          # must not raise


class TestPendingParticlesTeardown:
    """A run cancelled before finalize must not leave the placeholder behind."""

    def test_pending_store_is_cleared(self, window, parts):
        session = window["window"]
        s, _gt, p = parts
        tree = open_particle_tree(session, particles=p, source_node=s,
                                  attach=False)
        assert tree.particles is None
        assert tree._seg_pending_particles is p
        tree.close()
        assert getattr(tree, "_seg_pending_particles", None) is None

    def test_batch_flag_does_not_survive_as_a_phantom_run(self, window, parts):
        """`lifecycle.seg_batch_running` scans live trees; a closed one that kept
        the flag would make `wait_for_particles` wait on a run that has ended."""
        from spyde.actions.lifecycle import seg_batch_running
        session = window["window"]
        s, _gt, p = parts
        tree = open_particle_tree(session, particles=p, source_node=s)
        tree._seg_batch_running = True
        assert seg_batch_running(session)
        tree.close()
        assert not getattr(tree, "_seg_batch_running", False), (
            "close() did not clear _seg_batch_running")
        assert not seg_batch_running(session), (
            "a closed tree still reports a running segmentation batch")


class TestFigureRegistry:
    """Bare-figure windows must not pin their figures past teardown."""

    def test_forget_window_evicts_the_figure(self):
        marker = object()
        wid = 987654
        figure_registry.keep_alive(wid, marker)
        assert wid in figure_registry._FIGS
        figure_registry.forget_window(wid)
        assert wid not in figure_registry._FIGS, (
            "figure_registry still holds the window's figure after teardown")

    def test_forgetting_an_unknown_window_is_harmless(self):
        figure_registry.forget_window(123456789)

    def test_registry_holds_no_module_state_for_a_closed_window(self, window,
                                                                parts):
        """actions/README.md §3: figure_registry._FIGS is the ONLY module-level
        mutable state allowed, and it must be evicted by _forget_window."""
        session = window["window"]
        s, _gt, p = parts
        before = set(figure_registry._FIGS)
        tree = open_particle_tree(session, particles=p, source_node=s)
        wid = None
        for plot in tree.signal_plots or []:
            wid = getattr(plot, "window_id", None)
            if wid is not None:
                break
        tree.close()
        if wid is not None:
            session._forget_window(wid)
        leaked = set(figure_registry._FIGS) - before
        assert not leaked, f"figures left registered for windows {leaked}"


class TestSessionArtifacts:
    def test_forget_window_drops_controller_and_artifacts(self, window):
        session = window["window"]
        wid = 424242

        class _Ctrl:
            closed = False
            window_id = wid

            def close(self):
                _Ctrl.closed = True

        session.register_window_controller(wid, _Ctrl())
        session._action_artifacts[(wid, "Correct Drift")] = {"selector": None}
        assert wid in session._window_controllers

        session._forget_window(wid)
        assert wid not in session._window_controllers, "controller not dropped"
        assert _Ctrl.closed, "controller.close() was never called"
        assert not [k for k in session._action_artifacts if k[0] == wid], (
            "action artifacts survived the window")

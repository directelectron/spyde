"""
The particle tree (plan §0.6) and the Wave-0 framework around it.

Three separable claims, one per class:

* segmentation spawns a NEW tree carrying its provenance, rather than decorating
  the source — which is what makes Wave D's per-particle diffraction well-defined;
* the label movie is LAZY and stays lazy, because a materialised one is 64 MB per
  frame at the plan's target size;
* ``requires_particles`` gates identically in both toolbar filter paths, because
  a gate added to only one renders a button that never dispatches (or vice versa).
"""
from __future__ import annotations

import numpy as np
import pytest

import spyde.data.synthetic as sy
from spyde.actions.particle_tree import (
    PARTICLE_SIGNAL_TYPE,
    open_particle_tree,
    particle_nav_positions,
)
from spyde.particles import (
    LinkParams,
    link,
    measure_frame,
)
from spyde.signals.particles import COL, SpyDEParticles
from spyde.tests.migrated._labels import labels_from

N_FRAMES = 8


@pytest.fixture(scope="module")
def built():
    """A real segmentation of the fixture — not a hand-made container."""
    s = sy.particle_movie(n_frames=N_FRAMES)
    gt = sy.ground_truth(s)
    scale = float(gt["scale"])
    per_frame, contours = [], []
    for t in range(N_FRAMES):
        lab = labels_from(s.data[t], min_size=25, blur=1.0)
        rows, cs = measure_frame(lab, s.data[t], t=t, scale=scale)
        per_frame.append(rows)
        contours.append(cs)
    parts = SpyDEParticles.from_frames(
        per_frame, frame_shape=tuple(gt["frame_shape"]),
        contours_per_frame=contours, scale=scale, units="nm")
    res = link(parts, LinkParams(max_dist=10.0))
    res.apply(parts)
    return s, gt, parts, res


class TestTreeCreation:
    def test_spawns_a_new_tree_not_an_attribute(self, window, built):
        """The §0.6 decision, asserted directly."""
        session = window["window"]
        s, _gt, parts, res = built
        before = len(session.signal_trees)
        tree = open_particle_tree(session, particles=parts, source_node=s,
                                  events=res.events)
        assert len(session.signal_trees) == before + 1
        assert tree.particles is parts
        assert tree.source_node is s
        assert getattr(s, "particles", None) is None, (
            "the SOURCE signal was decorated — that is the design this replaces")

    def test_root_carries_the_particle_signal_type(self, window, built):
        session = window["window"]
        s, _gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        assert getattr(tree.root, "_signal_type", None) == PARTICLE_SIGNAL_TYPE

    def test_label_movie_matches_the_source_shape(self, window, built):
        session = window["window"]
        s, gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        assert tree.root.data.shape == (N_FRAMES, *tuple(gt["frame_shape"]))

    def test_provenance_is_stamped(self, window, built):
        session = window["window"]
        s, _gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s,
                                  params={"sensitivity": 0.5})
        prov = getattr(tree, "_commit_provenance", None) or {}
        assert prov.get("action") == "segment_particles"
        assert prov.get("params", {}).get("sensitivity") == 0.5

    def test_nav_map_defaults_to_identity(self, window, built):
        session = window["window"]
        s, _gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        assert np.array_equal(tree.nav_map, np.arange(N_FRAMES))

    def test_calibration_follows_the_source(self, window, built):
        """A centroid must mean the same thing on both trees."""
        session = window["window"]
        s, gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        assert tree.root.axes_manager.signal_axes[0].scale == \
            pytest.approx(float(gt["scale"]))
        assert tree.root.axes_manager.signal_axes[0].units == "nm"
        assert tree.root.axes_manager.navigation_axes[0].name == "time"


class TestLabelMovieStaysLazy:
    """A materialised label movie is 64 MB per frame at the plan's target size."""

    def test_root_is_lazy(self, window, built):
        session = window["window"]
        s, _gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        assert tree.root._lazy

    def test_one_frame_per_chunk(self, window, built):
        session = window["window"]
        s, _gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        assert tree.root.data.chunksize[0] == 1

    def test_computing_one_frame_does_not_compute_the_stack(self, window, built):
        """The Memory-Safety rule, enforced the way find_vectors enforces it."""
        import dask.array as da
        session = window["window"]
        s, _gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        full = tree.root.data.shape
        seen = {"full": 0}
        real = da.Array.compute

        def guard(self, *a, **k):
            if self.shape == full:
                seen["full"] += 1
            return real(self, *a, **k)

        try:
            da.Array.compute = guard
            frame = np.asarray(tree.root.data[3].compute())
        finally:
            da.Array.compute = real
        assert frame.shape == full[1:]
        assert seen["full"] == 0, "rendering one frame computed the whole movie"

    def test_rendered_frame_carries_track_ids(self, window, built):
        session = window["window"]
        s, _gt, parts, res = built
        tree = open_particle_tree(session, particles=parts, source_node=s,
                                  events=res.events)
        frame = np.asarray(tree.root.data[3].compute())
        painted = np.unique(frame)
        painted = painted[painted > 0]
        assert painted.size == len(parts.at(3)), (
            "painted a different number of particles than frame 3 holds")


class TestNavigatorTraces:
    def test_count_and_size_lanes_exist(self, window, built):
        session = window["window"]
        s, _gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        assert set(tree.nav_traces) >= {"count", "size"}
        assert tree.nav_traces["count"].shape == (N_FRAMES,)

    def test_event_lanes_appear_only_with_events(self, window, built):
        session = window["window"]
        s, _gt, parts, res = built
        without = open_particle_tree(session, particles=parts, source_node=s)
        assert not any(k.startswith("event_") for k in without.nav_traces)
        with_ev = open_particle_tree(session, particles=parts, source_node=s,
                                     events=res.events)
        assert any(k.startswith("event_") for k in with_ev.nav_traces)

    def test_count_lane_matches_the_store(self, window, built):
        session = window["window"]
        s, _gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        assert np.array_equal(tree.nav_traces["count"], parts.count_series())


class TestWaveDSeam:
    """`particle_nav_positions` is what makes per-particle diffraction definable."""

    def test_movie_particle_maps_to_its_frame(self, window, built):
        session = window["window"]
        s, _gt, parts, _res = built
        tree = open_particle_tree(session, particles=parts, source_node=s)
        gi = int(parts.indices_at(3)[0])
        nav = particle_nav_positions(tree, gi)
        assert nav.shape == (1, 1) and int(nav[0, 0]) == 3, (
            "on a MOVIE a particle's pixels are signal coordinates, so the only "
            "navigation index involved is the frame")

    def test_nav_map_is_honoured(self, window, built):
        session = window["window"]
        s, _gt, parts, _res = built
        shifted = np.arange(N_FRAMES) + 100
        tree = open_particle_tree(session, particles=parts, source_node=s,
                                  nav_map=shifted)
        gi = int(parts.indices_at(3)[0])
        assert int(particle_nav_positions(tree, gi)[0, 0]) == 103

    def test_survives_a_store_without_masks(self, window, built):
        """store_masks=False is the default for long movies — this must not raise."""
        session = window["window"]
        s, _gt, parts, _res = built
        bare = SpyDEParticles(parts.flat_buffer.copy(), parts.t_offsets.copy(),
                              parts.frame_shape, scale=parts.scale,
                              units=parts.units)
        assert not bare.has_masks
        tree = open_particle_tree(session, particles=bare, source_node=s)
        nav = particle_nav_positions(tree, 0)
        assert nav.shape == (1, 1)


class TestRequiresParticlesGate:
    """Both filter paths, because one alone is a button that never dispatches."""

    def _fake(self, has_particles: bool):
        class _Sig:
            _signal_type = "particles"

        class _Tree:
            particles = object() if has_particles else None
            diffraction_vectors = None
            root = _Sig()

        class _Plot:
            signal_tree = _Tree()

        class _State:
            plot = _Plot()
            current_signal = _Sig()
            dimensions = 2
            navigation = False
        return _State()

    def test_second_path_hides_and_shows(self):
        from spyde.drawing.toolbars.plot_control_toolbar import _action_matches_plot
        meta = {"requires_particles": True, "plot_dim": [1, 2]}
        assert not _action_matches_plot("X", meta, self._fake(False))
        assert _action_matches_plot("X", meta, self._fake(True))

    def test_ungated_actions_are_unaffected(self):
        from spyde.drawing.toolbars.plot_control_toolbar import _action_matches_plot
        meta = {"plot_dim": [1, 2]}
        assert _action_matches_plot("X", meta, self._fake(False))

    def test_both_paths_read_the_same_key(self):
        """Guards the §6 pitfall directly: the key must appear in BOTH filters."""
        import inspect
        from spyde.drawing.toolbars import plot_control_toolbar as mod
        src = inspect.getsource(mod)
        assert src.count("requires_particles") >= 4, (
            "requires_particles must be read in get_toolbar_actions_for_plot AND "
            "_action_matches_plot — one alone renders a button that never "
            "dispatches, or hides one that would have worked")


class TestWaitForParticles:
    def test_returns_false_without_an_event_loop(self):
        from spyde.actions.lifecycle import wait_for_particles

        class _S:
            _dispatch_to_main = None
        called = []
        started = wait_for_particles(_S(), None, lambda: called.append(1),
                                     what="Test")
        assert started is False and not called

    def test_fires_once_the_particles_land(self, built):
        import threading
        import time
        from spyde.actions.lifecycle import wait_for_particles

        _s, _gt, parts, _res = built

        class _Tree:
            particles = None

        class _Plot:
            signal_tree = _Tree()

        done = threading.Event()

        class _S:
            signal_trees = [_Tree()]

            @staticmethod
            def _dispatch_to_main(fn):
                fn()

        plot = _Plot()
        assert wait_for_particles(_S(), plot, done.set, what="Test", grace=30.0)
        time.sleep(0.25)
        assert not done.is_set(), "fired before the particles attached"
        plot.signal_tree.particles = parts
        assert done.wait(5.0), "never fired after the particles attached"

    def test_seg_batch_running_reads_the_tree_flag(self):
        from spyde.actions.lifecycle import seg_batch_running

        class _T:
            _seg_batch_running = False

        class _S:
            signal_trees = [_T()]
        s = _S()
        assert not seg_batch_running(s)
        s.signal_trees[0]._seg_batch_running = True
        assert seg_batch_running(s)

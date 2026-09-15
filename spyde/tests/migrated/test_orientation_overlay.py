"""
Orientation template overlay on the diffraction pattern (Qt parity).

After Orientation Mapping runs, the best-matching template's simulated spots must
be drawn on the SOURCE diffraction pattern and track the navigator — like the Qt
live-refine scatter. These tests verify the overlay attaches, produces marker
offsets in image-pixel coordinates, and re-pushes when the navigator moves.

The signal is calibrated with the direct beam at calibrated 0
(``offset = -(N/2)*scale``) — the standard centered-DP convention the spot→pixel
mapping (``px = (coord - offset)/scale``) relies on.
"""
from __future__ import annotations

import numpy as np
import hyperspy.api as hs

from spyde.array_cache import reader_for_overlay
from spyde.tests.migrated.conftest import _settle, close_session, make_session


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _make_phase():
    from orix.crystal_map import Phase
    from diffpy.structure import Atom, Lattice, Structure
    structure = Structure(
        atoms=[Atom("Al", [0, 0, 0])],
        lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90),
    )
    return Phase(name="Al", space_group=225, structure=structure)


def _centered_diffraction_4d(nav=(3, 4), sig=(32, 32), scale=1.0):
    """Calibrated in nm⁻¹ and meaning it: 1 nm⁻¹/px over 32 px is a half-extent
    of 16 nm⁻¹ = 1.6 Å⁻¹, which holds aluminium's {111} at 0.43 Å⁻¹. The scale
    was 0.1 under an nm⁻¹ label — a number that only made sense as Å⁻¹, and a
    library with no reflections in it once the label is read."""
    rng = np.random.RandomState(0)
    s = hs.signals.Signal2D(rng.rand(*nav, *sig).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = -(ax.size / 2.0) * scale     # beam at calibrated 0 (centre)
        ax.units = "nm^-1"
    return s


class TestOrientationOverlay:
    def test_overlay_attaches_and_draws_template_spots(self):
        from spyde.actions.orientation_action import run_orientation
        session = make_session()
        try:
            session._add_signal(_centered_diffraction_4d())
            _settle(session)
            src_tree = session.signal_trees[0]
            src_plot = _signal_plot(session)

            om = run_orientation(
                session, src_tree.root, src_tree, [_make_phase()],
                dict(accelerating_voltage=200.0, resolution=10.0),
                dict(n_best=3, gamma=0.5),
                src_dp_plot=src_plot,
            )
            assert om is not None

            node = getattr(src_tree, "_orientation_overlay", None)
            assert node is not None, "orientation overlay never attached"
            assert (id(node), "template") in src_plot._overlay_groups

            # It is a child of the node the window displays, which is what makes
            # a navigator move find it.
            displayed = src_plot.plot_state.current_signal
            assert node in src_tree.overlay_children(displayed)

            # Read a concrete nav position through the plot's readers, the way
            # the navigator refresh reads it, and inspect the offsets.
            reader = reader_for_overlay(src_plot, node)
            offsets = np.asarray(reader.read_frame((1, 1))["template"],
                                 dtype=np.float64)
            # There should be at least a few simulated spots, and every one must
            # land inside the 32x32 detector (pixel coords), not off-frame.
            assert len(offsets) > 0, "no template spots produced"
            assert offsets.shape[1] == 2
            assert offsets.min() >= -0.5 and offsets.max() <= 32.5
            assert np.isfinite(offsets).all()

            # Moving to two more positions both yield a valid (finite,
            # in-frame) push.
            for (iy, ix) in [(0, 0), (2, 3)]:
                off = np.asarray(reader.read_frame((iy, ix))["template"],
                                 dtype=np.float64)
                assert np.isfinite(off).all()
                if len(off):
                    assert off.min() >= -0.5 and off.max() <= 32.5
        finally:
            close_session(session)

    def test_orientation_end_to_end_on_lazy_data(self):
        """The full OM workflow must run on a LAZY signal: compute → IPF-Z window
        + attached map → live template overlay on the source DP."""
        from spyde.actions.orientation_action import run_orientation
        session = make_session()
        try:
            session._add_signal(_centered_diffraction_4d().as_lazy())
            _settle(session)
            src_tree = session.signal_trees[0]
            assert src_tree.root._lazy is True          # genuinely lazy
            src_plot = _signal_plot(session)
            before = len(session.signal_trees)

            om = run_orientation(
                session, src_tree.root, src_tree, [_make_phase()],
                dict(accelerating_voltage=200.0, resolution=10.0),
                dict(n_best=3, gamma=0.5), src_dp_plot=src_plot,
            )
            assert om is not None
            assert len(session.signal_trees) == before + 1   # IPF-Z window opened
            otree = session.signal_trees[-1]
            assert getattr(otree, "orientation_map", None) is om
            assert getattr(src_tree, "_orientation_overlay", None) is not None
        finally:
            close_session(session)

    def test_the_template_node_reads_a_centred_lazy_node_through_the_recipe(self):
        """Centre a lazy scan, then attach the template overlay to the centred
        node: both the overlay and the frame it matches must come through the
        plot's recipe readers, not a compute of the centred node's block."""
        import dask.array as da
        from spyde.actions.orientation_compute import build_matching_cache
        from spyde.actions.vector_overlay import attach_orientation_overlay
        from spyde.actions.center_zero_beam import czb_run
        from spyde.actions.orientation_compute import generate_library_from_phases
        from spyde.array_cache import reader_for_overlay
        from spyde.tests.migrated._async import wait_until

        session = make_session()
        try:
            lazy = _centered_diffraction_4d(nav=(4, 4)).as_lazy()
            lazy.data = lazy.data.rechunk((2, 2, -1, -1))
            session._add_signal(lazy)
            _settle(session)
            plot = _signal_plot(session)
            tree = plot.signal_tree

            before = plot.plot_state.current_signal
            czb_run(session, plot, {"method": "center_of_mass"})
            assert wait_until(
                lambda: plot.plot_state.current_signal is not before, 30)
            centred = plot.plot_state.current_signal

            sim = generate_library_from_phases([_make_phase()], 200.0, 10.0,
                                               1e-4, 1.0)
            node = attach_orientation_overlay(
                centred, sim, build_matching_cache(centred, sim), tree)

            reader = reader_for_overlay(plot, node)
            assert type(reader).__name__ == "RecipeReader", reader
            block_computes = []
            original = da.Array.compute

            def counting_compute(self, *args, **kwargs):
                if int(np.prod(self.shape[:2])) > 1:
                    block_computes.append(self.shape)
                return original(self, *args, **kwargs)

            reader.read_frame((1, 1))               # warm the parent's block
            parent = plot._local_transform_readers.get(id(centred))
            assert type(parent).__name__ == "RecipeReader", parent

            da.Array.compute = counting_compute
            try:
                spots = reader.read_frame((1, 1))["template"]
            finally:
                da.Array.compute = original
            assert block_computes == [], block_computes
            assert np.isfinite(spots).all()
        finally:
            close_session(session)

    # test_overlay_hook_registered_and_updates was folded into
    # test_overlay_attaches_and_draws_template_spots: identical eager session +
    # identical run_orientation(res=10) call, reading disjoint properties of
    # the same overlay (hook registration + two more positions).

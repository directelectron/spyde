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

import time

import numpy as np
import hyperspy.api as hs


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


def _centered_diffraction_4d(nav=(3, 4), sig=(32, 32), scale=0.1):
    rng = np.random.RandomState(0)
    s = hs.signals.Signal2D(rng.rand(*nav, *sig).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:
        ax.scale = scale
        ax.offset = -(ax.size / 2.0) * scale     # beam at calibrated 0 (centre)
        ax.units = "1/nm"
    return s


class TestOrientationOverlay:
    def test_overlay_attaches_and_draws_template_spots(self):
        from spyde.backend.session import Session
        from spyde.actions.orientation_action import run_orientation
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_centered_diffraction_4d())
            time.sleep(0.3)
            src_tree = session.signal_trees[0]
            src_plot = _signal_plot(session)

            om = run_orientation(
                session, src_tree.root, src_tree, [_make_phase()],
                dict(accelerating_voltage=200.0, resolution=10.0),
                dict(n_best=3, gamma=0.5),
                src_dp_plot=src_plot,
            )
            assert om is not None

            overlay = getattr(src_tree, "_orientation_overlay", None)
            assert overlay is not None, "orientation overlay never attached"
            assert overlay._mg is not None

            # Push a concrete nav position through the hook and inspect offsets.
            overlay._on_indices(np.array([[1, 1]]))
            offsets = np.asarray(overlay._mg._data["offsets"], dtype=np.float64)
            # There should be at least a few simulated spots, and every one must
            # land inside the 32x32 detector (pixel coords), not off-frame.
            assert len(offsets) > 0, "no template spots produced"
            assert offsets.shape[1] == 2
            assert offsets.min() >= -0.5 and offsets.max() <= 32.5
            assert np.isfinite(offsets).all()
        finally:
            session.shutdown()

    def test_orientation_end_to_end_on_lazy_data(self):
        """The full OM workflow must run on a LAZY signal: compute → IPF-Z window
        + attached map → live template overlay on the source DP."""
        from spyde.backend.session import Session
        from spyde.actions.orientation_action import run_orientation
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_centered_diffraction_4d().as_lazy())
            time.sleep(0.3)
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
            session.shutdown()

    def test_overlay_hook_registered_and_updates(self):
        from spyde.backend.session import Session
        from spyde.actions.orientation_action import run_orientation
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_centered_diffraction_4d())
            time.sleep(0.3)
            src_tree = session.signal_trees[0]
            src_plot = _signal_plot(session)
            run_orientation(
                session, src_tree.root, src_tree, [_make_phase()],
                dict(accelerating_voltage=200.0, resolution=10.0),
                dict(n_best=3, gamma=0.5),
                src_dp_plot=src_plot,
            )
            overlay = src_tree._orientation_overlay
            # The hook is registered on a navigator selector (real-drag path).
            sels = [s for s in src_tree.navigator_plot_manager.all_navigation_selectors
                    if overlay._on_indices in s.index_hooks]
            assert sels, "overlay hook not registered on any navigator selector"

            # Moving to two positions both yield a valid (finite, in-frame) push.
            for (iy, ix) in [(0, 0), (2, 3)]:
                overlay._on_indices(np.array([[ix, iy]]))
                off = np.asarray(overlay._mg._data["offsets"], dtype=np.float64)
                assert np.isfinite(off).all()
                if len(off):
                    assert off.min() >= -0.5 and off.max() <= 32.5
        finally:
            session.shutdown()

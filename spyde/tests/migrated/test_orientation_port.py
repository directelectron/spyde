"""
Orientation Mapping (Electron port) — end-to-end data flow.

`run_orientation` must build the template library, run the Qt-free compute core,
open an IPF-Z orientation-map window (RGB), and attach `tree.orientation_map`.
Uses a programmatically-built crystal phase (no CIF file needed) and a coarse
library so it runs quickly. The compute is memory-safe (per-chunk slices only).
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs

import pytest


def _make_phase():
    from orix.crystal_map import Phase
    from diffpy.structure import Atom, Lattice, Structure
    structure = Structure(
        atoms=[Atom("Al", [0, 0, 0])],
        lattice=Lattice(4.05, 4.05, 4.05, 90, 90, 90),
    )
    return Phase(name="Al", space_group=225, structure=structure)


def _diffraction_4d(nav=(3, 4), sig=(32, 32)):
    rng = np.random.RandomState(0)
    s = hs.signals.Signal2D(rng.rand(*nav, *sig).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    for ax in s.axes_manager.signal_axes:   # calibrate reciprocal space
        ax.scale = 0.1
        ax.units = "1/nm"
    return s


class TestOrientationPort:
    def test_orientation_builds_ipf_window_and_attaches_map(self):
        from spyde.backend.session import Session
        from spyde.actions.orientation_action import run_orientation
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_diffraction_4d())
            time.sleep(0.3)
            src_tree = session.signal_trees[0]
            before = len(session.signal_trees)

            om = run_orientation(
                session, src_tree.root, src_tree, [_make_phase()],
                dict(accelerating_voltage=200.0, resolution=10.0),
                dict(n_best=3, gamma=0.5),
            )
            assert om is not None
            # New orientation-map tree/window created + map attached.
            assert len(session.signal_trees) == before + 1
            otree = session.signal_trees[-1]
            assert getattr(otree, "orientation_map", None) is om

            # IPF-Z RGB map matches the scan grid and was pushed to the plot.
            ipf = om.ipf_color_map("z")
            assert ipf.shape == (3, 4, 3)
            sp = otree.signal_plots[0]
            assert isinstance(sp.current_data, np.ndarray)
            assert sp.current_data.shape == (3, 4, 3)   # RGB displayed
        finally:
            session.shutdown()

    def test_entry_rejects_missing_cif(self):
        from spyde.backend.session import Session
        from spyde.actions.context import ActionContext
        from spyde.actions.orientation_action import orientation_mapping
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            session._add_signal(_diffraction_4d())
            time.sleep(0.3)
            plot = next(p for p in session._plots
                        if not p.is_navigator and p.plot_state is not None)
            before = len(session.signal_trees)
            ctx = ActionContext(plot=plot, params={}, action_name="Orientation Mapping")
            orientation_mapping(ctx, cif_path=None)
            time.sleep(0.2)
            assert len(session.signal_trees) == before   # nothing created
        finally:
            session.shutdown()


def test_plot_renders_rgb_image():
    """Plot._set_array must route an (H, W, 3) RGB image to anyplotlib."""
    import numpy as np
    from spyde.drawing.plots.plot import Plot
    # Build a bare plot (no signal tree) and push an RGB frame.
    p = Plot.__new__(Plot)
    # Minimal attrs _set_array touches:
    p._plot2d = None
    p._plot1d = None
    p._fig = None
    p.fig_id = None
    calls = {}

    class _FakeP2D:
        def set_data(self, data, **k):
            calls["data"] = np.asarray(data)

    def _ensure(dims):
        p._plot2d = _FakeP2D()
    p._ensure_figure = _ensure  # type: ignore

    rgb = (np.random.rand(5, 6, 3) * 255).astype(np.uint8)
    Plot._set_array(p, rgb)
    assert "data" in calls and calls["data"].shape == (5, 6, 3)

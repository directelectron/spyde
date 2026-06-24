"""
5-D (stack/time) navigator chaining: the time axis must drive the DP.

The navigator for a 5-D dataset is a 2-level chain:
    time nav (1-D IntegratingSelector1D)  →  spatial nav (2-D IntegratingSSelector2D)  →  DP

Bug: IntegratingSelector1D is a COMPOSITE of two inner selectors (InfiniteLine +
LinearRegion). Each inner __init__ set ``child.plot_window.parent_selector =
<inner>``, and the LinearRegion (built second) won — so the spatial selector,
walking ``upstream_selectors()``, found the hidden region selector instead of the
active crosshair, composed the WRONG index, and the DP never tracked the time
axis (it only updated the 2-D real-space image). Fix: the composite points the
child's parent_selector at ITSELF (it delegates to the active sub-selector), and
the chain re-fire forces downstream recompute.
"""
from __future__ import annotations

import os

import numpy as np
import hyperspy.api as hs
import pytest


@pytest.fixture
def stack_5d_session():
    os.environ["SPYDE_NO_DASK"] = "1"
    from spyde.backend.session import Session
    sess = Session(n_workers=1, threads_per_worker=1)
    s = hs.signals.Signal2D(np.random.rand(2, 4, 5, 8, 8).astype(np.float32))
    s.set_signal_type("electron_diffraction")
    sess._add_signal(s, source_path=None)
    yield sess
    sess.shutdown()


def _selectors(sess):
    npm = sess.signal_trees[0].navigator_plot_manager
    return {type(x).__name__: x for x in npm.all_navigation_selectors}


class TestNavChain5D:
    def test_builds_time_and_spatial_selectors(self, stack_5d_session):
        sels = _selectors(stack_5d_session)
        assert "IntegratingSelector1D" in sels      # time (1-D line)
        assert "IntegratingSSelector2D" in sels     # spatial (2-D)

    def test_spatial_upstream_resolves_to_time_composite(self, stack_5d_session):
        sels = _selectors(stack_5d_session)
        s2 = sels["IntegratingSSelector2D"]
        up = [type(u).__name__ for u in s2.upstream_selectors()]
        # MUST be the composite, NOT a raw inner sub-selector (the bug).
        assert up == ["IntegratingSelector1D"]
        assert "LinearRegionSelector" not in up
        assert "InfiniteLineSelector" not in up

    def test_composed_index_includes_and_tracks_time(self, stack_5d_session):
        sels = _selectors(stack_5d_session)
        s1, s2 = sels["IntegratingSelector1D"], sels["IntegratingSSelector2D"]
        # Pin the spatial crosshair; vary the time selector's reported index.
        s2.selector._get_selected_indices = lambda: np.array([[2, 1]])
        s1.selector._get_selected_indices = lambda: np.array([[0]])
        at_t0 = s2.get_selected_indices().tolist()
        s1.selector._get_selected_indices = lambda: np.array([[1]])
        at_t1 = s2.get_selected_indices().tolist()
        # 3 coords [time, x, y]; the time component changes with the time axis.
        assert at_t0 == [[0, 2, 1]]
        assert at_t1 == [[1, 2, 1]]

    def test_chain_refire_forces_downstream(self, stack_5d_session):
        # The CHAIN re-fire must force=True so the deeper plot recomputes even
        # though the downstream selector's OWN widget didn't move.
        sels = _selectors(stack_5d_session)
        s1 = sels["IntegratingSelector1D"]
        s2 = sels["IntegratingSSelector2D"]
        forced = []
        # capture the force flag the chain passes downstream
        orig = s2.delayed_update_data
        s2.delayed_update_data = lambda force=False, **k: forced.append(force)
        try:
            # Run S1's update body directly; its child is the spatial nav window,
            # which gates → re-fires S2.
            s1.selector._get_selected_indices = lambda: np.array([[1]])
            s1.selector._run_update(force=True)
        finally:
            s2.delayed_update_data = orig
        assert forced and all(f is True for f in forced)

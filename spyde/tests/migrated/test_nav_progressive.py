"""
Lazy navigator load must be NON-BLOCKING.

The contract (see issue: large MRC scans on Windows):

  1. load the data lazily (no full materialise),
  2. create the navigator + signal PlotWindows,
  3. put a crosshair selector on the navigator,

…all near-instantly — and only THEN does the (slow) virtual navigation image
fill in progressively as chunks complete. Computing the navigator image must
never block steps 1–3 or freeze crosshair interaction.

These tests put an artificially SLOW navigator compute behind a lazy signal and
assert the windows + crosshair are ready long before the image finishes, and
that the crosshair can be moved (selecting a frame) while the nav image is still
computing.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest
from spyde.tests.migrated.conftest import make_session


def _make_session():
    return make_session()


# A blocking gate the fake navigator compute waits on, so the test controls
# exactly when the nav image is allowed to finish.
_GATE = threading.Event()
_NAV_COMPUTED = threading.Event()
# Armed only just before _add_signal, so HyperSpy's build-time metadata peeks
# (set_signal_type traverses a couple of blocks) don't trip the gate — we want
# to gate ONLY the navigator-image compute triggered by loading.
_ARMED = threading.Event()


def _slow_navigator_4d(nav=(16, 16), sig=(8, 8), free_pos=None, nav_chunks=None):
    """A lazy 4-D signal whose per-chunk read sleeps until the test releases the
    gate — simulating a large scan whose virtual-image sum is slow to compute.

    With ``free_pos=(iy, ix)`` (and a multi-chunk ``nav_chunks`` grid) the ONE
    block containing that nav position passes the gate freely: the navigator
    sum still blocks (it needs every other block), while a single-frame DP
    read of that position can complete.  Gating EVERY block was the fixture
    flaw that made 'DP while nav computes' untestable — the DP slice blocked
    at the same gate as the navigator sum."""
    ny, nx = nav
    # Identify the free block by its CONTENT (each frame is a constant, unique
    # value), not by block_info coordinates — dask's slicing pushdown can hand
    # the block function a rewritten task whose array-location is relative to
    # the sliced array, so coordinate keying silently gates the wrong blocks.
    free_value = (float(free_pos[0] * nx + free_pos[1] + 1)
                  if free_pos is not None else None)

    def _slow_block(block, block_info=None):
        if _ARMED.is_set():
            if free_value is not None and np.any(
                    np.asarray(block)[..., 0, 0] == free_value):
                return block              # the DP probe's block passes freely
            # The navigator sum traverses all blocks; block until released so the
            # nav image can't finish before the test checks the windows.
            _NAV_COMPUTED.set()
            _GATE.wait(timeout=10)
        return block

    base = np.zeros((ny, nx) + sig, dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            base[iy, ix] = float(iy * nx + ix + 1)
    arr = da.from_array(base, chunks=(nav_chunks or (ny, nx)) + sig)
    arr = arr.map_blocks(_slow_block, dtype=np.float32)
    s = hs.signals.Signal2D(arr).as_lazy()
    s.set_signal_type("electron_diffraction")
    return s


class TestProgressiveNavigator:
    def setup_method(self):
        _GATE.clear()
        _NAV_COMPUTED.clear()
        _ARMED.clear()

    def teardown_method(self):
        _ARMED.clear()
        _GATE.set()   # release any waiting compute so threads exit

    def test_windows_and_crosshair_ready_before_nav_image(self, monkeypatch):
        """Steps 1–3 (lazy load → windows → crosshair) complete while the
        navigator image is still blocked computing."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            s = _slow_navigator_4d()
            _ARMED.set()   # from here, any block traversal is the nav-image compute
            t0 = time.time()
            session._add_signal(s, source_path=None)
            setup_elapsed = time.time() - t0

            # The navigator image compute is gated for 10 s. _add_signal must
            # return FAST regardless — it runs the compute on a background thread
            # (the compute may have STARTED there, but setup must not WAIT on it).
            assert setup_elapsed < 5.0, (
                f"setup took {setup_elapsed:.1f}s — it blocked on the nav compute"
            )
            # The gate is still closed → the nav image has NOT finished, yet
            # setup already returned: proof the load didn't wait for it.
            assert not _GATE.is_set()

            # A navigator + signal window and a crosshair selector must exist.
            tree = session.signal_trees[-1]
            mgr = tree.navigator_plot_manager
            assert mgr is not None
            pw = next(iter(mgr.navigation_selectors.keys()))
            sel = mgr.navigation_selectors[pw][0]
            cross = getattr(sel, "_crosshair_selector", sel)
            assert cross._widget is not None, "no crosshair on the navigator"
        finally:
            _GATE.set()
            session.shutdown()

    def test_crosshair_selects_frame_while_nav_still_computing(self, monkeypatch):
        """The crosshair can move and select a diffraction pattern BEFORE the
        navigator virtual image has finished.

        (Formerly a dead non-strict xfail: the fixture gated EVERY block, so
        every DP read — including the load-time read at the crosshair's
        DEFAULT CENTER — blocked at the same gate as the navigator sum,
        wedging the serial dispatcher; the test could never pass and timed
        out its poll on every run.  The fixture now exempts the ONE 8x8
        nav-chunk holding the default center AND the probed position, so
        DP reads complete while the sum stays blocked on the other three
        chunks.)"""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        session = _make_session()
        try:
            nx = 16
            # 2x2 nav-chunk grid.  The free block must be the one holding the
            # crosshair's DEFAULT CENTER (8, 8): the selector fires a read
            # there at load, on the same serial dispatcher — if that read
            # gates, the probe below never runs.  Probe a different position
            # inside the same chunk.
            ix, iy = 10, 9                      # chunk (1, 1), like (8, 8)
            s = _slow_navigator_4d(nav=(16, nx), free_pos=(iy, ix),
                                   nav_chunks=(8, 8))
            _ARMED.set()
            session._add_signal(s, source_path=None)
            # The progressive fill is deferred until the first signal frame
            # paints, so wait on the event rather than a flat sleep.
            assert _NAV_COMPUTED.wait(timeout=10), "nav compute never started"
            assert not _GATE.is_set()   # still blocked mid-compute

            tree = session.signal_trees[-1]
            mgr = tree.navigator_plot_manager
            pw = next(iter(mgr.navigation_selectors.keys()))
            sel = mgr.navigation_selectors[pw][0]
            cross = getattr(sel, "_crosshair_selector", sel)

            # Move the crosshair and force an update — must work even though
            # the nav image is still blocked.
            cross._widget.cx = float(ix)
            cross._widget.cy = float(iy)
            sel.delayed_update_data(force=True)

            # Updates run on the serial nav-dispatcher (async); poll the
            # crosshair's child for the PROBE's frame (the load-time read may
            # already have painted the default-center frame).
            expected = float(iy * nx + ix + 1)
            child = next(iter(cross.children.keys()))
            data = None
            for _ in range(40):
                data = child.current_data
                if (isinstance(data, np.ndarray)
                        and abs(float(np.mean(data)) - expected) < 1e-3):
                    break
                time.sleep(0.05)
            assert isinstance(data, np.ndarray), "crosshair didn't select a DP frame"
            assert abs(float(np.mean(data)) - expected) < 1e-3
            # The nav image compute is still blocked at the gate — proving the
            # crosshair worked WHILE the navigator was mid-compute.
            assert not _GATE.is_set(), "gate released — nav compute not blocked"
        finally:
            _GATE.set()
            session.shutdown()

    def test_navigator_fills_progressively_without_cluster(self, monkeypatch):
        """Without a Dask cluster, the navigator must fill PER CHUNK (multiple
        paints with a growing finite-pixel count), not stay blank until the whole
        multi-GB sum finishes."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")

        # Count every navigator-plot paint (Plot.set_data) and how many pixels
        # were finite at each — wrapped on the CLASS before load so we catch the
        # background compute's incremental paints.
        from spyde.drawing.plots.plot import Plot
        paints: list[int] = []
        orig = Plot.set_data

        def _count(self, data, *a, **k):
            try:
                if getattr(self, "is_navigator", False):
                    paints.append(int(np.isfinite(np.asarray(data)).sum()))
            except Exception:
                pass
            return orig(self, data, *a, **k)

        monkeypatch.setattr(Plot, "set_data", _count)

        session = _make_session()
        try:
            ny = nx = 16
            base = np.zeros((ny, nx, 8, 8), dtype=np.float32)
            for iy in range(ny):
                for ix in range(nx):
                    base[iy, ix] = float(iy * nx + ix + 1)
            arr = da.from_array(base, chunks=(8, 8, 8, 8))   # 2×2 nav chunk grid
            s = hs.signals.Signal2D(arr).as_lazy()
            s.set_signal_type("electron_diffraction")

            session._add_signal(s, source_path=None)
            # Wait for the background per-chunk paints to land.
            for _ in range(60):
                if len(paints) >= 2 and paints[-1] >= ny * nx:
                    break
                time.sleep(0.1)

            assert len(paints) >= 2, (
                f"navigator painted {len(paints)}× — not progressive (expected per-chunk)"
            )
            # Finite pixels accumulate to the full 16×16 image.
            assert max(paints) >= ny * nx, (
                f"navigator never fully filled (max finite px {max(paints)} < {ny*nx})"
            )
        finally:
            session.shutdown()

    def test_navigator_has_no_holes_when_complete(self, monkeypatch):
        """The FINAL navigator must be hole-free (no leftover NaN). Holes came
        from the distributed poll loop breaking on future.done() before a final
        paint, and from chunk shm-writes racing the whole-array future. The fix
        paints the authoritative result on completion. Here we assert the
        end-state completeness on the (deterministic) threaded path; the
        distributed path now paints future.result() which is complete by
        construction."""
        monkeypatch.setenv("SPYDE_NO_DASK", "1")
        from spyde.drawing.plots.plot import Plot
        last_finite = [0]
        last_size = [0]
        orig = Plot.set_data

        # The session first, so the tracker can tell OUR navigator from anyone
        # else's. The patch is on the CLASS, so it sees every Plot in the
        # process — including one belonging to an earlier test whose poller
        # thread has not wound down yet. That is not hypothetical: this test
        # failed in CI with `assert 256 == (12 * 12)`, i.e. it measured a
        # leftover 16x16 navigator and never looked at its own.
        session = _make_session()

        def own(p):
            if any(p is q for q in session._plots):
                return True
            # Registration in _plots can trail the first paint, so also accept
            # a plot whose tree is one of ours.
            tree = getattr(p, "signal_tree", None)
            return tree is not None and any(tree is t for t in session.signal_trees)

        def _track(self, data, *a, **k):
            try:
                if getattr(self, "is_navigator", False) and own(self):
                    arr = np.asarray(data)
                    last_finite[0] = int(np.isfinite(arr).sum())
                    last_size[0] = int(arr.size)
            except Exception:
                pass
            return orig(self, data, *a, **k)

        monkeypatch.setattr(Plot, "set_data", _track)

        try:
            ny = nx = 12
            base = np.arange(ny * nx * 8 * 8, dtype=np.float32).reshape(ny, nx, 8, 8)
            arr = da.from_array(base, chunks=(4, 4, 8, 8))   # 3×3 nav chunk grid
            s = hs.signals.Signal2D(arr).as_lazy()
            s.set_signal_type("electron_diffraction")
            session._add_signal(s, source_path=None)

            # Wait for the fill to complete (last paint covers every pixel).
            for _ in range(80):
                if last_size[0] and last_finite[0] >= ny * nx:
                    break
                time.sleep(0.1)
            assert last_size[0] == ny * nx, "navigator size unexpected"
            assert last_finite[0] == ny * nx, (
                f"navigator left {ny*nx - last_finite[0]} NaN holes "
                f"({last_finite[0]}/{ny*nx} finite)"
            )
        finally:
            session.shutdown()

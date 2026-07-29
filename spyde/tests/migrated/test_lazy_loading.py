"""
Example datasets must load LAZILY (Dask-backed), not eagerly into RAM.

Regression: pyxem's example loaders return eager in-memory signals, so the app
"just loaded into memory" and Dask was never used. `Session._to_lazy` converts
them (persist to temp .zspy → lazy reload), and the lazy signal must then flow
through `_add_signal` even with no distributed client (headless/threaded).
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs


def _eager_4d(nav=(4, 5), sig=(8, 8)):
    rng = np.random.RandomState(0)
    return hs.signals.Signal2D(rng.rand(*nav, *sig).astype(np.float32))


class TestLazyExampleLoading:
    def test_to_lazy_converts_eager_signal(self, window):
        session = window["window"]
        s = _eager_4d()
        assert not s._lazy
        lazy = session._to_lazy(s, "synthetic")
        assert lazy._lazy is True
        assert isinstance(lazy.data, da.Array)
        # Data round-trips through the temp store intact.
        np.testing.assert_allclose(
            np.asarray(lazy.data.compute()), s.data, rtol=1e-5
        )

    def test_to_lazy_passthrough_when_already_lazy(self, window):
        session = window["window"]
        s = _eager_4d().as_lazy()
        out = session._to_lazy(s, "already-lazy")
        assert out is s  # no needless disk round-trip

    def test_lazy_signal_flows_through_add_signal_without_client(self, window):
        """A lazy signal must open (nav + signal windows) with no Dask client."""
        session = window["window"]
        assert session.dask_manager.client is None  # SPYDE_NO_DASK
        lazy = _eager_4d().as_lazy()
        session._add_signal(lazy, source_path=None)
        assert len(session.signal_trees) == 1
        tree = session.signal_trees[0]
        # Root stays lazy — we did NOT materialise the whole dataset.
        assert tree.root._lazy is True
        # Navigator + signal windows were created.
        assert len(session._plots) >= 2

    def test_load_example_opens_the_file_lazily(self, window, monkeypatch):
        """A downloaded example is opened as dask, never materialised — some of
        them are gigabytes, and the navigator reads a chunk at a time."""
        import spyde.backend.example_catalogue as ec
        session = window["window"]
        opened = []

        def fake_open(path):
            opened.append(path)
            return _eager_4d().as_lazy()

        monkeypatch.setattr(ec, "_open_lazily", fake_open)
        sig = session._load_example_lazy("/tmp/example.zspy")
        assert sig._lazy is True
        assert opened == ["/tmp/example.zspy"]

    def test_load_example_wraps_an_eager_reader(self, window, monkeypatch):
        """A reader that hands back an eager signal is wrapped via as_lazy() —
        still Dask-backed, still no disk round-trip."""
        import spyde.backend.example_catalogue as ec
        session = window["window"]
        monkeypatch.setattr(ec, "_open_lazily", lambda path: _eager_4d())
        sig = session._load_example_lazy("/tmp/example.hspy")
        assert sig._lazy is True
        assert isinstance(sig.data, da.Array)

    def test_load_example_takes_the_first_of_a_multi_signal_file(self, window,
                                                                 monkeypatch):
        """Some readers return a LIST of signals for one file; loading one must
        not hand a list to _add_signal."""
        import spyde.backend.example_catalogue as ec
        session = window["window"]
        monkeypatch.setattr(ec, "_open_lazily",
                            lambda path: [_eager_4d(), _eager_4d()])
        sig = session._load_example_lazy("/tmp/multi.h5")
        assert sig._lazy is True and not isinstance(sig, list)


class TestNonBlockingNavigator:
    """The display must NOT wait on the navigator compute, and a tree built
    before the cluster is ready must still pick it up (live `client`)."""

    def _nav_plot(self, session):
        return next((p for p in session._plots if p.is_navigator), None)

    def test_add_signal_returns_without_blocking_on_nav_compute(self, window, monkeypatch):
        import time
        import dask.array as da
        session = window["window"]
        assert session.dask_manager.client is None   # no cluster (SPYDE_NO_DASK)

        # Make any full dask compute slow. The navigator compute now runs on a
        # BACKGROUND thread, so _add_signal must still return promptly.
        orig = da.Array.compute
        def slow(self, *a, **k):
            time.sleep(1.0)
            return orig(self, *a, **k)
        monkeypatch.setattr(da.Array, "compute", slow)

        lazy = _eager_4d().as_lazy()
        t = time.time()
        session._add_signal(lazy, source_path=None)
        elapsed = time.time() - t
        assert elapsed < 0.8, f"_add_signal blocked on the navigator compute ({elapsed:.2f}s)"

    def test_navigator_fills_in_background_without_client(self, window):
        import time
        session = window["window"]
        lazy = _eager_4d(nav=(4, 5)).as_lazy()
        session._add_signal(lazy, source_path=None)
        nav = self._nav_plot(session)
        assert nav is not None
        # The background threaded compute fills the NaN placeholder with real data.
        filled = False
        for _ in range(80):                      # ≤8 s
            cd = nav.current_data
            if isinstance(cd, np.ndarray) and np.isfinite(np.asarray(cd)).any():
                filled = True
                break
            time.sleep(0.1)
        assert filled, "navigator never filled in the background"

    def test_client_property_reads_live_from_dask_manager(self, window):
        session = window["window"]
        lazy = _eager_4d().as_lazy()
        session._add_signal(lazy, source_path=None)
        tree = session.signal_trees[0]
        assert tree.client is None             # no cluster yet
        sentinel = object()
        session.dask_manager._client = sentinel   # cluster "comes up"
        assert tree.client is sentinel         # tree picks it up live
        session.dask_manager._client = None

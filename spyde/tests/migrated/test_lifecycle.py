"""
test_lifecycle.py — the shared action-lifecycle basis set (spyde/actions/lifecycle.py):
worker marshal, generation guard, the wait-for-vectors gap helper, overlay
replacement, signal-plot painting, progress emission, and the live-fill poller.
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from spyde.actions import lifecycle
from spyde.backend.session import Session


def _wait_until(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


class _LoopSession:
    """Minimal session driving the REAL Session._dispatch_to_main against a live
    asyncio loop on its own thread (copy of test_strain_threaded's harness)."""

    _dispatch_to_main = Session._dispatch_to_main
    set_main_loop = Session.set_main_loop

    def __init__(self):
        self._main_loop = None
        self.signal_trees: list = []
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="test-lifecycle-loop")
        self._thread.start()
        assert self._ready.wait(5.0), "event loop thread did not start"

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.set_main_loop(loop)
        self._ready.set()
        loop.run_forever()

    def stop(self):
        loop = self._main_loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=5.0)


@pytest.fixture
def loop_session():
    s = _LoopSession()
    yield s
    s.stop()


class TestRunOnWorker:
    def test_inline_without_session(self):
        """No session (bare handler test) → work + on_done run synchronously."""
        seen = []
        lifecycle.run_on_worker(None, lambda: 42, name="t",
                                on_done=lambda r: seen.append(r))
        assert seen == [42]

    def test_inline_error_calls_on_error(self):
        errs = []

        def boom():
            raise RuntimeError("nope")

        lifecycle.run_on_worker(None, boom, name="t",
                                on_error=lambda e: errs.append(e))
        assert len(errs) == 1 and isinstance(errs[0], RuntimeError)

    def test_marshals_on_done_to_loop_thread(self, loop_session):
        done = threading.Event()
        where = {}

        def on_done(result):
            where["thread"] = threading.current_thread().name
            where["result"] = result
            done.set()

        lifecycle.run_on_worker(loop_session, lambda: "ok", name="t", on_done=on_done)
        assert done.wait(10.0), "on_done never landed"
        assert where["result"] == "ok"
        assert where["thread"] == "test-lifecycle-loop"

    def test_worker_error_reaches_on_error(self, loop_session):
        errs = []
        got = threading.Event()

        def boom():
            raise ValueError("bad")

        lifecycle.run_on_worker(loop_session, boom, name="t",
                                on_error=lambda e: (errs.append(e), got.set()))
        assert got.wait(10.0)
        assert isinstance(errs[0], ValueError)


class TestGenerationGuard:
    def test_bump_and_is_current(self):
        class Owner:
            pass

        o = Owner()
        g1 = lifecycle.bump_generation(o, "_x_run_gen")
        assert g1 == 1 and lifecycle.is_current(o, "_x_run_gen", g1)
        g2 = lifecycle.bump_generation(o, "_x_run_gen")
        assert g2 == 2
        assert not lifecycle.is_current(o, "_x_run_gen", g1)
        assert lifecycle.is_current(o, "_x_run_gen", g2)


class _FakeTree:
    def __init__(self, vecs=None, running=False):
        self.diffraction_vectors = vecs
        self._fv_batch_running = running


class _FakePlot:
    def __init__(self, tree):
        self.signal_tree = tree


class TestResolveVectors:
    def test_prefers_plot_tree(self):
        t = _FakeTree(vecs="V")
        session = type("S", (), {"signal_trees": []})()
        tree, vecs = lifecycle.resolve_vectors(session, _FakePlot(t))
        assert tree is t and vecs == "V"

    def test_falls_back_to_any_tree(self):
        other = _FakeTree(vecs="W")
        session = type("S", (), {"signal_trees": [_FakeTree(), other]})()
        tree, vecs = lifecycle.resolve_vectors(session, _FakePlot(_FakeTree()))
        assert tree is other and vecs == "W"


class TestWaitForVectors:
    def test_no_loop_returns_false(self):
        session = type("S", (), {"signal_trees": []})()   # no _dispatch_to_main
        assert lifecycle.wait_for_vectors(session, None, lambda: None,
                                          what="Test") is False

    def test_fires_then_once_vectors_attach(self, loop_session):
        tree = _FakeTree(vecs=None, running=True)
        loop_session.signal_trees = [tree]
        fired = []
        done = threading.Event()

        started = lifecycle.wait_for_vectors(
            loop_session, _FakePlot(tree),
            lambda: (fired.append(1), done.set()),
            what="Test", grace=0.5, timeout=5.0)
        assert started is True
        time.sleep(0.3)                       # in the gap: nothing yet
        assert not fired
        tree.diffraction_vectors = "V"        # the batch finalizes
        assert done.wait(5.0), "then never fired after attach"
        time.sleep(0.3)
        assert fired == [1], "then fired more than once"

    def test_nothing_running_errors_after_grace(self, loop_session, captured_messages):
        tree = _FakeTree(vecs=None, running=False)
        loop_session.signal_trees = [tree]
        fired = []

        started = lifecycle.wait_for_vectors(
            loop_session, _FakePlot(tree), lambda: fired.append(1),
            what="Test action", grace=0.3, timeout=5.0)
        assert started is True
        assert _wait_until(lambda: any(
            m.get("type") == "error" and "Test action" in m.get("text", "")
            for m in captured_messages), 5.0), "no grace-expiry error emitted"
        assert not fired


class TestReplaceTreeAttr:
    def test_removes_old_and_sets_new(self):
        class Overlay:
            removed = 0

            def remove(self):
                Overlay.removed += 1

        tree = _FakeTree()
        tree._ov = Overlay()
        new = lifecycle.replace_tree_attr(tree, "_ov", lambda: "NEW")
        assert Overlay.removed == 1
        assert new == "NEW" and tree._ov == "NEW"

    def test_factory_failure_leaves_none(self):
        tree = _FakeTree()
        tree._ov = None

        def boom():
            raise RuntimeError("attach failed")

        assert lifecycle.replace_tree_attr(tree, "_ov", boom) is None
        assert tree._ov is None

    def test_none_factory_just_removes(self):
        removed = []
        tree = _FakeTree()
        tree._ov = type("O", (), {"remove": lambda self: removed.append(1)})()
        assert lifecycle.replace_tree_attr(tree, "_ov", None) is None
        assert removed == [1] and tree._ov is None


class _FakeSignalPlot:
    def __init__(self):
        self.needs_auto_level = None
        self.clim = None
        self.data = None

    def set_clim(self, lo, hi):
        self.clim = (lo, hi)

    def set_data(self, d):
        self.data = d


class TestPaintSignalPlots:
    def test_paints_with_levels_locked(self):
        tree = _FakeTree()
        tree.signal_plots = [_FakeSignalPlot(), _FakeSignalPlot()]
        m = np.ones((3, 3), np.float32)
        lifecycle.paint_signal_plots(tree, m, levels=(-2.0, 2.0))
        for sp in tree.signal_plots:
            assert sp.needs_auto_level is False
            assert sp.clim == (-2.0, 2.0)
            assert sp.data is m

    def test_paints_auto_level_without_levels(self):
        tree = _FakeTree()
        tree.signal_plots = [_FakeSignalPlot()]
        lifecycle.paint_signal_plots(tree, np.ones((2, 2)))
        sp = tree.signal_plots[0]
        assert sp.needs_auto_level is True and sp.clim is None


class TestProgressEmitter:
    def test_throttles_and_always_emits_final(self, captured_messages):
        progress = lifecycle.progress_emitter("Fitting…", min_interval=10.0)
        progress(1, 100)     # first emit
        progress(2, 100)     # throttled
        progress(100, 100)   # final — always emitted
        texts = [m.get("text") for m in captured_messages if m.get("type") == "status"]
        assert texts == ["Fitting… 1%", "Fitting… 100%"]


class TestLiveFillPoller:
    def test_none_shm_is_noop(self):
        stop = lifecycle.live_fill_poller((4, 4), None, lambda a: None)
        stop()   # must not raise

    def test_polls_until_stopped(self):
        from spyde.drawing.update_functions import ensure_live_buffer
        name = "spyde_test_lifecycle_poll"
        shm = ensure_live_buffer((4, 4), name)
        calls = []
        try:
            stop = lifecycle.live_fill_poller(
                (4, 4), name, lambda arr: calls.append(arr.shape), interval=0.05)
            assert _wait_until(lambda: len(calls) >= 2, 5.0), "poller never painted"
            assert calls[0] == (4, 4)
            stop()
            n = len(calls)
            time.sleep(0.3)
            assert len(calls) <= n + 1, "poller kept painting after stop()"
        finally:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

"""DaskManager.shutdown() must not reap a cluster that was never started.

The reap tail — settle sleep, psutil walk of the machine's whole process
tree, wait_procs, gc.collect() — exists to make sure a nanny's worker
GRANDCHILD dies with the app. Under SPYDE_NO_DASK (every pytest Session,
and any headless script) nothing was ever spawned, so it was reaping
nothing at a flat ~670ms a go. The fixtures build a Session per test, so
that was ~26 minutes of a ~28-minute suite, and it is what pushed the CI
matrix past its job timeout.

Both halves matter, so both are pinned here: skip when nothing ran, and
still reap when something did. The second is the one that protects users —
a leaked Dask worker outlives the app and holds its memory.

These stub the cluster rather than spinning one: a real ``LocalCluster``
costs ~15 s, which is precisely the kind of thing this change exists to
remove from the suite. The real path was verified out-of-band instead —
a 2-worker cluster, ``shutdown()`` taking 1.99 s (so the settle + psutil
walk + ``wait_procs`` all ran) and zero surviving children. Re-run that
by hand if you touch the reap; do not add it here.
"""
from __future__ import annotations

import time

import pytest

from spyde.dask_manager import DaskManager


class TestNothingStarted:
    def test_shutdown_is_immediate_when_start_was_never_called(self):
        dm = DaskManager(n_workers=1, threads_per_worker=1)
        t0 = time.perf_counter()
        dm.shutdown()
        elapsed = time.perf_counter() - t0
        # The old path slept 0.5s flat before it even began walking processes.
        assert elapsed < 0.2, (
            f"shutdown() took {elapsed:.3f}s with no cluster ever started — the "
            "settle/reap tail is running again; that is ~0.67s on every one of "
            "the suite's Session teardowns"
        )

    def test_it_does_not_touch_the_process_tree(self, monkeypatch):
        """The psutil walk is the expensive half on Windows (~30ms for a
        ppid_map of the whole box) and is meaningless with no children."""
        import psutil

        called: list[str] = []
        monkeypatch.setattr(
            psutil.Process, "children",
            lambda self, recursive=False: called.append("children") or [],
        )
        DaskManager(n_workers=1, threads_per_worker=1).shutdown()
        assert called == [], "shutdown() walked the process tree with no cluster"

    def test_repeated_shutdown_stays_cheap(self):
        """Session.shutdown() is called on every fixture teardown, and some
        paths call it twice; neither may pay the tail."""
        dm = DaskManager(n_workers=1, threads_per_worker=1)
        t0 = time.perf_counter()
        for _ in range(5):
            dm.shutdown()
        assert time.perf_counter() - t0 < 0.5


class TestSomethingStarted:
    """The half that protects users: if start() ever ran, reap unconditionally.

    Gated on `_thread` rather than on client/cluster, because start() builds
    the cluster on a BACKGROUND thread — a shutdown racing a startup can
    legitimately see both attributes still unset while a LocalCluster spawns.
    Skipping there would leak the workers it was about to create.
    """

    def _instrument(self, monkeypatch):
        import psutil

        seen: list[str] = []
        monkeypatch.setattr(
            psutil.Process, "children",
            lambda self, recursive=False: seen.append("children") or [],
        )
        monkeypatch.setattr(time, "sleep", lambda s: seen.append(f"sleep{s}"))
        return seen

    def test_a_started_manager_still_reaps(self, monkeypatch):
        dm = DaskManager(n_workers=1, threads_per_worker=1)
        # What start() leaves behind, without actually spawning a cluster.
        dm._thread = object()
        seen = self._instrument(monkeypatch)
        dm.shutdown()
        assert "children" in seen, (
            "a manager that called start() skipped the reap — a nanny's worker "
            "grandchild would outlive the app"
        )

    def test_a_live_cluster_still_reaps_even_without_the_thread(self, monkeypatch):
        """restart() and the test helpers set _cluster directly."""
        class _Cluster:
            def scale(self, n): pass
            def close(self, timeout=None): pass

        dm = DaskManager(n_workers=1, threads_per_worker=1)
        dm._cluster = _Cluster()
        seen = self._instrument(monkeypatch)
        dm.shutdown()
        assert "children" in seen

    @pytest.mark.parametrize("attr", ["_client", "_cluster"])
    def test_either_handle_alone_is_enough_to_reap(self, monkeypatch, attr):
        class _Handle:
            def scale(self, n): pass
            def close(self, timeout=None): pass

        dm = DaskManager(n_workers=1, threads_per_worker=1)
        setattr(dm, attr, _Handle())
        seen = self._instrument(monkeypatch)
        dm.shutdown()
        assert "children" in seen

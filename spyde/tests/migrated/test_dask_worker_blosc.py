"""The worker plugin gives each dask worker process blosc's thread pool.

Run in-process against the plugin's ``setup``: what it applies here is what a
worker applies, since the patch is a module-level wrapper either way.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def unpatched(monkeypatch):
    """numcodecs' blosc module with the wrappers removed, restored after."""
    import numcodecs.blosc as blosc
    from spyde.external.numcodecs import blosc_threads

    saved = {name: getattr(blosc, name)
             for name in ("compress", "decompress", "decompress_partial")
             if hasattr(blosc, name)}
    originals = {name: getattr(fn, "__wrapped__", fn) for name, fn in saved.items()}
    for name, fn in originals.items():
        monkeypatch.setattr(blosc, name, fn)
    monkeypatch.setattr(blosc, "use_threads", None)
    yield blosc, blosc_threads
    for name, fn in saved.items():
        setattr(blosc, name, fn)


class TestWorkerPluginBlosc:
    def test_setup_turns_the_pool_on_in_the_worker(self, monkeypatch, unpatched):
        blosc, blosc_threads = unpatched
        monkeypatch.delenv(blosc_threads.THREAD_COUNT_VARIABLE, raising=False)
        from spyde.dask_manager import _WorkerTuningPlugin

        _WorkerTuningPlugin().setup()
        assert getattr(blosc.decompress, blosc_threads.MARKER, False)
        assert blosc.use_threads is True

    def test_the_switch_reaches_the_worker_too(self, monkeypatch, unpatched):
        blosc, blosc_threads = unpatched
        monkeypatch.setenv(blosc_threads.THREAD_COUNT_VARIABLE, "0")
        from spyde.dask_manager import _WorkerTuningPlugin

        _WorkerTuningPlugin().setup()
        assert not getattr(blosc.decompress, blosc_threads.MARKER, False)
        assert blosc.use_threads is None

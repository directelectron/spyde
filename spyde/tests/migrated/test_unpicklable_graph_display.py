"""
A lazy graph that cannot be pickled must never be sent to a Dask worker.

RosettaSciIO's lazy TIFF reader closes its graph over an open ``BufferedReader``
and an ``.hspy`` closes over an h5py object, so neither can be serialized to a
distributed worker at all. ``client.compute()`` on one raises AT SUBMIT with
``Could not serialize object of type _HLGExprSequence``.

That broke opening a plain ``.tif`` outright: the raise landed between
``window_computing().start()`` and the ``_on_plot_ready`` that stops it, so the
file opened as a permanently black window captioned "Calculating…" with
"Failed to load …" on the status bar. It hid for so long because ``.zspy`` — the
format SpyDE writes itself, and the one every lazy fixture uses — IS picklable.

Two paths carry a freshly-loaded lazy array to a compute, and both are covered
here: the no-navigation display (``compute_display_future``) and the progressive
navigator fill (which must take its in-process branch, with an EXPLICIT threaded
scheduler — a bare ``.compute()`` resolves to the distributed scheduler whenever
a client exists, which silently shipped the unsendable graph anyway).
"""
from __future__ import annotations

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest
import tifffile

from spyde.drawing.update_functions import (
    compute_display_future, graph_can_reach_workers,
)


def _lazy_tif(tmp_path, name="img.tif", frames=None):
    """A lazy signal read back from a REAL .tif — the graph under test only
    exists when rsciio actually opens a file (a synthetic da.from_array is
    picklable and would prove nothing)."""
    data = np.random.default_rng(0).random((64, 64)).astype("float32") \
        if frames is None else \
        np.random.default_rng(0).random((frames, 32, 32)).astype("float32")
    p = tmp_path / name
    tifffile.imwrite(str(p), data)
    return hs.load(str(p), lazy=True), data


class _SpyClient:
    """Stands in for a distributed Client: records submissions and refuses an
    unpicklable graph exactly as the real one does."""

    def __init__(self):
        self.calls = 0

    def compute(self, arr):
        self.calls += 1
        if not graph_can_reach_workers(arr):
            raise TypeError("Could not serialize object of type _HLGExprSequence")
        fut = _DoneFuture()
        fut._value = np.asarray(arr.compute(scheduler="threads"))
        fut._done_evt.set()
        return fut


class _DoneFuture:
    _spyde_future = True
    key = "spy-client-future"

    def __init__(self):
        import threading
        self._done_evt = threading.Event()
        self._value = None
        self._error = None

    def done(self):
        return self._done_evt.is_set()

    def result(self, timeout=None):
        self._done_evt.wait(timeout)
        if self._error:
            raise self._error
        return self._value


class TestGraphCanReachWorkers:
    def test_lazy_tif_graph_cannot(self, tmp_path):
        sig, _ = _lazy_tif(tmp_path)
        assert graph_can_reach_workers(sig.data) is False

    def test_lazy_zspy_graph_can(self, tmp_path):
        p = tmp_path / "x.zspy"
        hs.signals.Signal2D(np.zeros((4, 4, 8, 8), "float32")).save(str(p))
        sig = hs.load(str(p), lazy=True)
        assert graph_can_reach_workers(sig.data) is True

    def test_plain_dask_array_can(self):
        assert graph_can_reach_workers(da.zeros((8, 8), chunks=(4, 4))) is True

    def test_lazy_hspy_can_even_though_plain_pickle_refuses_it(self, tmp_path):
        """THE case that says why the verdict must come from distributed's own
        serializer. An .hspy graph holds an h5py object that cloudpickle refuses
        outright — but distributed serializes it fine, and .hspy has always
        worked in the app. A cloudpickle-only probe answers False here and
        quietly demotes every HDF5 dataset to an in-process compute."""
        p = tmp_path / "x.hspy"
        hs.signals.Signal2D(np.zeros((4, 4, 8, 8), "float32")).save(str(p))
        data = hs.load(str(p), lazy=True).data

        import cloudpickle
        with pytest.raises(Exception):
            cloudpickle.dumps(data)          # the trap
        assert graph_can_reach_workers(data) is True   # the verdict that counts

    def test_the_probe_does_not_leave_distributed_logging_muted(self, tmp_path):
        """The probe silences distributed's serialization logger while it asks;
        it must put the level back, or a REAL serialization failure elsewhere
        goes unreported for the rest of the session."""
        import logging as _logging
        lg = _logging.getLogger("distributed.protocol.pickle")
        before = lg.level
        sig, _ = _lazy_tif(tmp_path)
        assert graph_can_reach_workers(sig.data) is False
        assert lg.level == before


class TestComputeDisplayFuture:
    def test_unpicklable_graph_never_reaches_the_client(self, tmp_path):
        """The regression: the submit must not even be attempted, because a
        failed one is NOT quiet — distributed logs it at ERROR with three
        chained tracebacks before raising."""
        sig, data = _lazy_tif(tmp_path)
        client = _SpyClient()
        fut = compute_display_future(sig.data, client, label="test")
        assert fut.result(timeout=30) == pytest.approx(data)
        assert client.calls == 0, "an unsendable graph was offered to the cluster"

    def test_picklable_graph_still_goes_to_the_client(self):
        """The cluster stays the default: a display array may be the tip of a
        large reduction whose inputs should stay on the workers."""
        client = _SpyClient()
        arr = da.ones((8, 8), chunks=(4, 4))
        fut = compute_display_future(arr, client, label="test")
        assert client.calls == 1
        assert np.array_equal(fut.result(timeout=30), np.ones((8, 8)))

    def test_no_client_computes_locally(self, tmp_path):
        sig, data = _lazy_tif(tmp_path)
        fut = compute_display_future(sig.data, None, label="test")
        assert fut.result(timeout=30) == pytest.approx(data)

    def test_local_future_is_visible_to_the_plot_worker(self, tmp_path):
        """It must carry the ``_spyde_future`` marker and a unique key, or
        PlotUpdateWorker never delivers it and the "Calculating…" overlay — which
        only _on_plot_ready stops — stays up forever."""
        from spyde.workers.plot_update_worker import _is_future
        sig, _ = _lazy_tif(tmp_path)
        a = compute_display_future(sig.data, None, label="test")
        b = compute_display_future(sig.data, None, label="test")
        assert _is_future(a) and _is_future(b)
        assert a.key != b.key

    def test_a_failing_compute_still_resolves_the_future(self):
        """An error must arrive through result(), not hang: the overlay stop is
        downstream of this future resolving either way."""
        def _boom(_block):
            raise ValueError("boom")

        # dtype= is required, or map_blocks infers it by CALLING _boom now and
        # the raise never reaches the compute this test is about.
        boom = da.ones((4, 4), chunks=(2, 2)).map_blocks(_boom, dtype="float32")
        fut = compute_display_future(boom, None, label="test")
        with pytest.raises(ValueError):
            fut.result(timeout=30)


class TestThreadedNavFillIsActuallyThreaded:
    def test_a_bare_compute_follows_the_configured_default(self):
        """Pins the hazard the explicit kwarg exists for: a bare ``.compute()``
        resolves to dask's CONFIGURED scheduler, and a live distributed Client
        installs itself as that default. Demonstrated here with the config knob
        the Client uses, rather than by standing up a cluster."""
        import dask
        calls = []

        def _spy(dsk, keys, **kw):
            calls.append(keys)
            return dask.get(dsk, keys, **kw)

        arr = da.ones((4, 4), chunks=(2, 2))
        with dask.config.set(scheduler=_spy):
            arr.compute()                      # bare: picks up the default
        assert calls, "a bare .compute() ignored the configured scheduler"
        calls.clear()
        with dask.config.set(scheduler=_spy):
            arr.compute(scheduler="threads")   # explicit: immune to it
        assert not calls, "an explicit scheduler= was overridden by the default"

    def test_nav_fill_names_the_threaded_scheduler(self):
        import inspect
        from spyde import signal_tree
        src = inspect.getsource(signal_tree.BaseSignalTree
                                ._start_progressive_nav_compute)
        assert 'compute(scheduler="threads")' in src, (
            "the in-process navigator fill must pass scheduler='threads' "
            "explicitly — a bare .compute() goes to the cluster when a client "
            "exists, which is exactly the graph that cannot be serialized")

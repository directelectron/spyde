"""One dispatcher for every per-chunk submission.

`compute_dispatch.dispatch_chunks` is the only place allowed to submit
per-nav-chunk computes in a loop, because a hand-rolled loop always ends up
missing at least one of its three properties:

  1. **batched submit** — `client.compute(list)` once per top-up, not a blocking
     scheduler round trip per chunk.  Measured end to end on the real 977-frame
     / 15.3 GB in-situ movie (977 nav chunks, 4 workers — see
     `benchmark_nav_fill_dispatch.py`): the per-chunk loop
     spent **9.86 s** in submission with the GIL held in the client process, so
     nothing else in the backend could run and the first navigator pixel
     appeared at **16.4 s**.  Through the dispatcher: submit 0.00 s, first pixel
     **0.08 s**, same total fill, identical checksum.
  2. **backpressure** — a bounded in-flight window, so the scheduler is never
     handed the whole dataset up front (the prefetch-then-spill pathology).
     The navigator fill submitted all 977 chunks at once.
  3. **the stall watchdog / scheduler poke** — task delivery freezes on the
     hidden Electron-spawned backend until client traffic arrives.

`TestNoPerChunkSubmitLoops` asserts on the SOURCE, deliberately, exactly like
`test_movie_chunking.TestEveryLoadPathIsAligned`: the failure mode is a new call
site that forgets, and no runtime assertion catches a path no test exercises.
"""
from __future__ import annotations

import ast
import pathlib
import threading

import dask.array as da
import numpy as np

import spyde


def _spyde_root() -> pathlib.Path:
    return pathlib.Path(spyde.__file__).parent


# Modules that may legitimately submit inside a loop.
_ALLOWED = {
    # THE dispatcher itself.
    "compute_dispatch.py",
}


class _SubmitInLoopVisitor(ast.NodeVisitor):
    """Flags `client.compute(...)` / `client.submit(...)` lexically inside a
    `for`/`while` body.

    Keyed on the RECEIVER NAME containing "client" (`client`, `self._client`,
    `self.client`, `dask_client`, ...). It has to be: a bare `.compute()` in a
    loop is the *correct*, memory-safe pattern for a small ghost-padded slice
    (CLAUDE.md's Memory Safety rule), so flagging every `.compute()` would be
    noise. A future call site that binds the client to a name with no "client"
    in it slips through — name it sensibly.
    """

    def __init__(self):
        self.hits: list[tuple[int, str]] = []
        self._loop_depth = 0

    def _visit_loop(self, node):
        self._loop_depth += 1
        self.generic_visit(node)
        self._loop_depth -= 1

    visit_For = _visit_loop
    visit_AsyncFor = _visit_loop
    visit_While = _visit_loop

    def visit_Call(self, node):
        if self._loop_depth > 0:
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr in ("compute", "submit"):
                owner = fn.value
                name = getattr(owner, "id", None) or getattr(owner, "attr", None)
                if name and "client" in str(name).lower():
                    self.hits.append((node.lineno, f"{name}.{fn.attr}()"))
        self.generic_visit(node)


class TestNoPerChunkSubmitLoops:
    def test_no_client_submit_inside_a_loop_outside_the_dispatcher(self):
        offenders = []
        for path in sorted(_spyde_root().rglob("*.py")):
            if path.name in _ALLOWED or "tests" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:                       # pragma: no cover
                continue
            v = _SubmitInLoopVisitor()
            v.visit(tree)
            for lineno, what in v.hits:
                offenders.append(f"{path.relative_to(_spyde_root())}:{lineno} {what}")
        assert not offenders, (
            "per-chunk submission inside a loop — route it through "
            "spyde.compute_dispatch.dispatch_chunks (batched submit + bounded "
            "in-flight window + stall watchdog):\n  " + "\n  ".join(offenders)
        )

    def test_the_guard_actually_catches_the_pattern(self):
        """A source assertion that matches nothing is worthless — pin that the
        visitor really fires on the shape it is supposed to forbid, and really
        does NOT fire on the memory-safe per-slice `.compute()`."""
        bad = ast.parse(
            "for sl in slices:\n"
            "    futs.append(self._client.compute(arr[sl]))\n"
        )
        v = _SubmitInLoopVisitor(); v.visit(bad)
        assert v.hits, "the guard would not catch a per-chunk client.compute loop"

        ok = ast.parse(
            "for sl in slices:\n"
            "    blocks.append(raw[sl].compute())\n"       # memory-safe slice
            "client.compute(whole)\n"                       # outside any loop
        )
        v2 = _SubmitInLoopVisitor(); v2.visit(ok)
        assert not v2.hits

    def test_the_navigator_fill_calls_the_dispatcher(self):
        """Behaviour tests below use a fake client; this pins that the REAL
        navigator branch is wired to the shared dispatcher at all."""
        import inspect
        from spyde.drawing import update_functions as uf

        src = inspect.getsource(uf._dispatched_progressive)
        assert "dispatch_chunks" in src
        assert "dispatch_chunks" in inspect.getsource(uf.compute_with_live_buffer) \
            or "_dispatched_progressive" in inspect.getsource(uf.compute_with_live_buffer)

    def test_no_second_whole_array_compute_in_the_nav_fill(self):
        """The fill used to end with `client.compute(result_array)` — a SECOND
        full pass over the dataset after every chunk had already been computed.
        The chunks ARE the result; they are assembled client-side."""
        import inspect
        from spyde.drawing import update_functions as uf

        src = inspect.getsource(uf.compute_with_live_buffer)
        assert "client.compute(result_array)" not in src


class _FakeFuture:
    """Fires its done-callbacks on demand so the test owns completion order."""

    def __init__(self, value_fn, key="chunk"):
        self._value_fn = value_fn
        self._cbs = []
        self.key = key
        self.cancelled = False
        self._done = False

    def add_done_callback(self, cb):
        self._cbs.append(cb)

    def done(self):
        return self._done

    def result(self):
        if self.cancelled:
            raise RuntimeError("cancelled")
        return self._value_fn()

    def cancel(self):
        self.cancelled = True

    def release(self):
        pass

    def fire(self):
        self._done = True
        for cb in self._cbs:
            cb(self)


class _FakeClient:
    """Records how many SUBMIT CALLS were made (not how many futures)."""

    def __init__(self, nthreads=4):
        self.created: list[_FakeFuture] = []
        self.submit_calls = 0
        self.submit_kwargs: list[dict] = []
        self._nthreads = nthreads

    def scheduler_info(self, n_workers=None):
        return {"workers": {"a": {"nthreads": self._nthreads,
                                  "memory_limit": 1 << 30,
                                  "metrics": {"memory": 1}}}}

    def compute(self, arrays, **kwargs):
        self.submit_calls += 1
        self.submit_kwargs.append(kwargs)
        single = not isinstance(arrays, (list, tuple))
        out = []
        for arr in ([arrays] if single else arrays):
            fut = _FakeFuture(lambda a=arr: a.compute(scheduler="synchronous"))
            self.created.append(fut)
            out.append(fut)
        return out[0] if single else out

    def run_on_scheduler(self, *a, **k):
        return {}

    def call_stack(self, *a, **k):
        return {}


def _run_nav_fill(nav=(16,), chunks=(1,), nthreads=4):
    """Drive `compute_with_live_buffer`'s navigator branch to completion."""
    from spyde.drawing.update_functions import compute_with_live_buffer

    src = da.arange(int(np.prod(nav)), dtype=np.float32).reshape(nav) \
            .rechunk(chunks)
    client = _FakeClient(nthreads=nthreads)
    seen: list = []
    futs: list = []
    handle = compute_with_live_buffer(
        src, nav, client, shm_name="",
        on_chunk_done=lambda res, sl: seen.append((res, sl)),
        on_future=futs.append,
    )
    # The dispatcher runs on its own thread; wait for the first batch to exist.
    _wait(lambda: client.created)
    fired = 0
    max_in_flight = 0
    while fired < len(client.created):
        max_in_flight = max(max_in_flight, len(client.created) - fired)
        client.created[fired].fire()
        fired += 1
        _wait(lambda f=fired: len(client.created) > f or handle.done(),
              timeout=2.0, required=False)
    handle.result(timeout=10.0)
    return src, client, handle, seen, futs, max_in_flight


def _wait(pred, timeout=10.0, required=True):
    end = threading.Event()
    import time as _t
    t0 = _t.monotonic()
    while _t.monotonic() - t0 < timeout:
        if pred():
            return True
        end.wait(0.005)
    if required:
        raise AssertionError("timed out waiting for condition")
    return False


class TestNavigatorFillThroughDispatcher:
    def test_batched_submit_prime_then_bulk(self):
        """16 one-frame nav chunks: a small priming batch, then the rest — and
        only a couple of scheduler round trips in total.

        This test used to assert ``max_in_flight <= 4`` as BACKPRESSURE, on the
        reasoning that the navigator fill must not "submit all 977 up front".
        That guard is retired deliberately, so here is the argument against my
        own fence:

        * It did not measure as backpressure. Measured: the bounded
          window at 46-50 s against unbounded at 50.2 s on the same 977-chunk
          movie — within noise. The window never bought throughput or safety
          that anyone demonstrated.
        * ``distributed`` >= 2022.3 QUEUES root tasks at the scheduler
          (worker-saturation), which is exactly this guard's job, done in the
          right process, without holding OUR GIL. We were duplicating it badly.
        * Keeping it was actively harmful: ``_submit_next`` runs on every
          completion, so ``lane_cap - outstanding`` was 1 in steady state and
          the batch collapsed to ONE task per submit — 970 blocking round trips
          for 977 chunks, which scaled with dataset size and read to users as
          "big datasets only use one worker".

        What replaces it: the assertion below that the whole job goes out in a
        couple of submits, and the scheduler's own queuing for memory. The
        DUAL-LANE path keeps its window (see the module docstring) because
        there a completion genuinely pulls the next chunk so a ~30x-faster GPU
        lane and the CPU lane finish together — that is real work stealing the
        scheduler cannot do.
        """
        src, client, handle, seen, futs, max_in_flight = _run_nav_fill(
            nav=(16,), chunks=(1,))
        assert len(seen) == 16                       # every chunk streamed
        assert len(client.created) == 16             # one future per chunk
        # The bug this replaced made ~one submit per chunk.
        assert client.submit_calls <= 3, (
            f"{client.submit_calls} submits for 16 chunks — the unpinned lane "
            f"is back to per-completion top-ups")
        np.testing.assert_array_equal(handle.result(), src.compute())

    def test_result_is_assembled_not_recomputed(self):
        """No extra whole-array submission: exactly one future per nav chunk."""
        src, client, handle, seen, futs, _ = _run_nav_fill(nav=(8,), chunks=(2,))
        assert len(client.created) == 4              # 4 chunks, no 5th graph
        np.testing.assert_array_equal(handle.result(), src.compute())

    def test_submits_unpinned(self):
        """The navigator has no GPU/CPU lane asymmetry — chunks must go out with
        NO `workers=` restriction so the scheduler places by data locality."""
        src, client, handle, seen, futs, _ = _run_nav_fill(nav=(4,), chunks=(1,))
        assert all("workers" not in k for k in client.submit_kwargs)

    def test_one_cancellable_handle_registered(self):
        """`on_future` used to fire once per chunk (978 registrations on the
        977-frame movie). It now fires once, with a handle whose cancel() stops
        the whole dispatch."""
        src, client, handle, seen, futs, _ = _run_nav_fill(nav=(4,), chunks=(1,))
        assert futs == [handle]
        assert hasattr(handle, "cancel")

    def test_cancel_stops_the_dispatch(self):
        """Tree close -> _cancel_all_compute -> handle.cancel() must stop the
        fill on the cluster, not let the whole dataset sum run to completion."""
        from spyde.drawing.update_functions import compute_with_live_buffer

        srcd = da.arange(64, dtype=np.float32).rechunk((1,))
        client = _FakeClient()
        handle = compute_with_live_buffer(srcd, (64,), client, shm_name="")
        _wait(lambda: client.created)
        n_before = len(client.created)
        assert n_before < 64                       # bounded, not all 64
        handle.cancel()
        _wait(lambda: all(f.cancelled for f in client.created))
        for f in list(client.created):
            f.fire()
        _wait(handle.done)
        assert len(client.created) == n_before      # no top-ups after cancel
        assert handle.result() is None

    def test_plot_update_worker_accepts_the_handle(self):
        """The whole reason the old code kept a duplicate whole-array future:
        PlotUpdateWorker isinstance-checked distributed.Future."""
        from spyde.workers.plot_update_worker import _is_future
        from spyde.drawing.update_functions import (
            _AssembledFuture, _ProgressiveFuture,
        )

        h = _AssembledFuture("navigator")
        assert _is_future(h)
        assert "write_shared_array" not in h.key
        # Unique per instance — the worker dedups by (key, id(plot)), so a
        # shared key would make a second fill on the same plot never emit.
        assert _AssembledFuture("navigator").key != h.key
        # The VI stream's handle must stay invisible to the worker.
        assert not _is_future(_ProgressiveFuture())
        assert not _is_future(np.zeros(3))
        assert not _is_future(None)

    def test_worker_finds_the_handle_through_hyperspys_data_setter(self):
        """`nav_signal.data = handle` does NOT store the handle: hyperspy's
        setter runs `np.atleast_1d(np.asanyarray(value))`, so it lands as a
        length-1 OBJECT array. `_future_from_signal` has to unwrap that — this
        is the path that de-lazifies the navigator signal when the fill lands.
        """
        import hyperspy.api as hs
        from spyde.workers.plot_update_worker import PlotUpdateWorker
        from spyde.drawing.update_functions import _AssembledFuture

        sig = hs.signals.Signal1D(np.zeros((4, 4), np.float32))
        h = _AssembledFuture("navigator")
        sig.data = h
        assert isinstance(sig.data, np.ndarray)      # the trap
        w = PlotUpdateWorker(lambda: [])
        assert w._future_from_signal(sig) is h

    def test_cancelled_handle_is_not_delivered(self):
        """A cancelled duck future has no result — pushing its None onto the
        plot/signal would blank them."""
        from spyde.workers.plot_update_worker import PlotUpdateWorker
        from spyde.drawing.update_functions import _AssembledFuture

        h = _AssembledFuture("navigator")
        h.cancel()
        h._done_evt.set()
        got = []
        w = PlotUpdateWorker(lambda: [])
        w._maybe_emit_future(h, lambda *a: got.append(a), plot=object())
        assert got == []

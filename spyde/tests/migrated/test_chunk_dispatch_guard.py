"""One dispatcher for every per-chunk submission.

`compute_dispatch.dispatch_chunks` is the only place allowed to submit
per-nav-chunk computes in a loop, because a hand-rolled loop always ends up
missing at least one of its three properties:

  1. **batched submit** — `client.compute(list)` once per top-up, not a blocking
     scheduler round trip per chunk.  Measured end to end on the real 977-frame
     / 15.3 GB in-situ movie (977 nav chunks, 4 workers — see
     `benchmark_nav_fill_dispatch.py` and benchmarks.md): the per-chunk loop
     spent **9.86 s** in submission with the GIL held in the client process, so
     nothing else in the backend could run and the first navigator pixel
     appeared at **16.4 s**.  Through the dispatcher: submit 0.00 s, first pixel
     **0.08 s**, same total fill, identical checksum.
  2. **the in-flight window, on the lane that earns it** — a PINNED dual-lane
     dispatch keeps a bounded window PER LANE, so every completion pulls
     exactly the next chunk from the shared pending queue and a ~30x-faster GPU
     lane drains the same pool as the CPU lane instead of finishing early and
     idling.  That is real work stealing the dask scheduler cannot do, and
     `TestPinnedLaneWindow` is what pins it.
     The UNPINNED lane deliberately has NO window any more: it primes with one
     small batch and sends the rest in a single submit.  As backpressure the
     window never measured (benchmarks.md: bounded 46-50 s vs unbounded 50.2 s
     on the same 977-chunk movie), `distributed` >= 2022.3 queues root tasks at
     the scheduler itself, and keeping it collapsed the batch to ONE task per
     submit — 970 blocking round trips for 977 chunks.  See
     `TestNavigatorFillThroughDispatcher.test_batched_submit_prime_then_bulk`.
  3. **the stall watchdog / scheduler poke** — task delivery freezes on the
     hidden Electron-spawned backend until client traffic arrives.

`TestNoPerChunkSubmitLoops` asserts on the SOURCE, deliberately, exactly like
`test_movie_chunking.TestEveryLoadPathIsAligned`: the failure mode is a new call
site that forgets, and no runtime assertion catches a path no test exercises.
"""
from __future__ import annotations

import ast
import collections
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
            "spyde.compute_dispatch.dispatch_chunks (batched submit + lane "
            "placement with a per-lane in-flight window + stall watchdog + one "
            "cancellable handle):\n  " + "\n  ".join(offenders)
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


def _submitted_and_armed(client, at_least: int = 1) -> bool:
    """``at_least`` futures exist AND every one of them has its done-callback.

    ``dispatch_chunks`` appends the futures FIRST and registers the callbacks in
    a following loop, so a predicate on the COUNT alone can wake inside that
    gap. Firing a future whose ``_cbs`` is still empty drops the completion on
    the floor — the dispatcher then never sees that chunk finish and the whole
    drain hangs, looking like a dispatcher bug rather than a test race.
    """
    futs = list(client.created)
    return len(futs) >= at_least and all(f._cbs for f in futs)


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
    # The dispatcher runs on its own thread; wait for the first batch to be
    # submitted AND armed (see _submitted_and_armed).
    _wait(lambda: _submitted_and_armed(client))
    fired = 0
    while fired < len(client.created):
        client.created[fired].fire()
        fired += 1
        _wait(lambda f=fired: len(client.created) > f or handle.done(),
              timeout=2.0, required=False)
    handle.result(timeout=10.0)
    return src, client, handle, seen, futs


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

        * It did not measure as backpressure. benchmarks.md has the bounded
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
        src, client, handle, seen, futs = _run_nav_fill(nav=(16,), chunks=(1,))
        assert len(seen) == 16                       # every chunk streamed
        assert len(client.created) == 16             # one future per chunk
        # The bug this replaced made ~one submit per chunk.
        assert client.submit_calls <= 3, (
            f"{client.submit_calls} submits for 16 chunks — the unpinned lane "
            f"is back to per-completion top-ups")
        np.testing.assert_array_equal(handle.result(), src.compute())

    def test_result_is_assembled_not_recomputed(self):
        """No extra whole-array submission: exactly one future per nav chunk."""
        src, client, handle, seen, futs = _run_nav_fill(nav=(8,), chunks=(2,))
        assert len(client.created) == 4              # 4 chunks, no 5th graph
        np.testing.assert_array_equal(handle.result(), src.compute())

    def test_submits_unpinned(self):
        """The navigator has no GPU/CPU lane asymmetry — chunks must go out with
        NO `workers=` restriction so the scheduler places by data locality."""
        src, client, handle, seen, futs = _run_nav_fill(nav=(4,), chunks=(1,))
        assert all("workers" not in k for k in client.submit_kwargs)

    def test_one_cancellable_handle_registered(self):
        """`on_future` used to fire once per chunk (978 registrations on the
        977-frame movie). It now fires once, with a handle whose cancel() stops
        the whole dispatch."""
        src, client, handle, seen, futs = _run_nav_fill(nav=(4,), chunks=(1,))
        assert futs == [handle]
        assert hasattr(handle, "cancel")

    def test_cancel_stops_the_dispatch(self):
        """Tree close -> _cancel_all_compute -> handle.cancel() must stop the
        fill on the cluster, not let the whole dataset sum run to completion."""
        from spyde.drawing.update_functions import compute_with_live_buffer

        srcd = da.arange(64, dtype=np.float32).rechunk((1,))
        client = _FakeClient()
        handle = compute_with_live_buffer(srcd, (64,), client, shm_name="")
        _wait(lambda: _submitted_and_armed(client))
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


# ── the PINNED (dual-lane) path: the window that was NOT retired ─────────────


class _LaneClient(_FakeClient):
    """A two-address cluster with a GPU lane and a CPU lane.

    Tags every future with the lane it was submitted to (read off the
    ``workers=`` kwarg ``dispatch_chunks`` passes) so the test can measure each
    lane's in-flight window separately — a single global count would not notice
    one lane running unbounded while the other stayed small.
    """

    GPU = "tcp://gpu-worker:1"
    CPU = "tcp://cpu-worker:1"

    def __init__(self, gpu_nthreads=2, cpu_nthreads=1):
        super().__init__()
        self._gpu_nthreads = gpu_nthreads
        self._cpu_nthreads = cpu_nthreads
        self.lane_of: dict = {}                  # id(future) -> "gpu" | "cpu"

    def scheduler_info(self, n_workers=None):
        base = {"memory_limit": 1 << 30, "metrics": {"memory": 1}}
        return {"workers": {
            self.GPU: dict(base, nthreads=self._gpu_nthreads),
            self.CPU: dict(base, nthreads=self._cpu_nthreads),
        }}

    def compute(self, arrays, **kwargs):
        first = len(self.created)
        futs = super().compute(arrays, **kwargs)
        lane = "gpu" if self.GPU in list(kwargs.get("workers") or []) else "cpu"
        for f in self.created[first:]:
            self.lane_of[id(f)] = lane
        return futs


def _run_pinned_dispatch(n_chunks=24, gpu_nthreads=2, cpu_nthreads=1,
                         timeout=30.0):
    """Drive `dispatch_chunks` on a PINNED dual-lane cluster, one completion at
    a time, recording the per-lane in-flight window and what each completion
    pulled.

    Firing a future runs the dispatcher's done-callback INLINE on this thread,
    and that callback tops the lane up under the dispatcher's own lock — so
    ``created`` has already grown by the time ``fire()`` returns and the
    measurement needs no sleeps.
    """
    from spyde.compute_dispatch import dispatch_chunks

    src = da.arange(n_chunks, dtype=np.float32).rechunk((1,))
    client = _LaneClient(gpu_nthreads=gpu_nthreads, cpu_nthreads=cpu_nthreads)
    seen: list = []
    out: dict = {}

    def _assemble(result, nav_slices, chunk_result):
        result[nav_slices] = chunk_result

    def _run():
        try:
            out["result"] = dispatch_chunks(
                client, src, 1, [client.GPU], [client.CPU],
                assemble=_assemble,
                on_chunk_done=lambda sl, res: seen.append((sl, res)),
                label="pinned-test",
                # "off" so the 5 s lane refresh can't consult this machine's
                # real CUDA availability and make the test host-dependent.
                lane_default_mode="off",
            )
        except BaseException as exc:                      # surfaced below
            out["error"] = exc

    th = threading.Thread(target=_run, daemon=True, name="pinned-dispatch")
    th.start()
    _wait(lambda: _submitted_and_armed(client))
    initial = len(client.created)

    steps: list = []
    fired = 0
    while fired < len(client.created):
        before = len(client.created)
        in_flight = collections.Counter(
            client.lane_of[id(f)] for f in client.created[fired:before])
        fut = client.created[fired]
        fut.fire()
        fired += 1
        steps.append({
            "lane": client.lane_of[id(fut)],
            "in_flight": in_flight,
            "pulled": [client.lane_of[id(f)]
                       for f in client.created[before:]],
        })
    th.join(timeout)
    assert not th.is_alive(), "the pinned dispatch never finished"
    if "error" in out:
        raise out["error"]
    return src, client, seen, out["result"], initial, steps


class TestPinnedLaneWindow:
    """The dual-lane in-flight window — the one the unpinned lane's retirement
    did NOT cover, and which nothing else in the suite exercises.

    `test_dask_stats.TestMemoryBackpressure` checks `_lane_cap`'s ARITHMETIC in
    isolation; both fake-client harnesses above run the UNPINNED lane, whose
    window is deliberately gone. So a change that extended the unpinned
    prime-then-bulk branch to a pinned lane — handing a ~30x-asymmetric cluster
    the whole pending queue up front and destroying the per-completion pull —
    passed every test in this file.
    """

    def test_each_lane_holds_its_window_and_pulls_one_chunk_per_completion(self):
        n = 24
        src, client, seen, result, initial, steps = _run_pinned_dispatch(
            n_chunks=n, gpu_nthreads=2, cpu_nthreads=1)

        # Windows as dispatch_chunks sizes them: the GPU lane gets a deeper one
        # (2*threads + 2) because chunks overlap loads/transfers/kernels; the
        # CPU margin stays small (threads + 2) because each in-flight chunk pins
        # its inputs in worker RAM.
        gpu_cap, cpu_cap = 2 * 2 + 2, 1 + 2

        # 1. The opening fill opens each lane's window and STOPS there — it does
        #    not hand the scheduler the whole job the way the unpinned lane does.
        assert initial == gpu_cap + cpu_cap, (
            f"the first fill submitted {initial} chunks for two lanes with "
            f"windows {gpu_cap}/{cpu_cap} — a pinned lane lost its window")
        assert initial < n

        # 2. Neither lane ever exceeds its own window, at any point in the drain.
        for i, st in enumerate(steps):
            assert st["in_flight"]["gpu"] <= gpu_cap, (
                f"step {i}: {st['in_flight']['gpu']} GPU chunks in flight, "
                f"window is {gpu_cap}")
            assert st["in_flight"]["cpu"] <= cpu_cap, (
                f"step {i}: {st['in_flight']['cpu']} CPU chunks in flight, "
                f"window is {cpu_cap}")

        # 3. THE point of the dual-lane window: while work remains, EACH
        #    completion pulls exactly ONE chunk from the shared pending queue,
        #    onto the lane that just freed a slot. That is how a fast lane keeps
        #    stealing from the same pool instead of finishing early and idling —
        #    placement the dask scheduler cannot do, because it keeps one
        #    duration estimate per task family.
        n_pulling = n - initial
        for i, st in enumerate(steps[:n_pulling]):
            assert st["pulled"] == [st["lane"]], (
                f"completion {i} on the {st['lane']} lane pulled "
                f"{st['pulled']} — expected exactly the next chunk, on that "
                f"same lane")
        for i, st in enumerate(steps[n_pulling:], start=n_pulling):
            assert st["pulled"] == [], (
                f"completion {i} submitted {st['pulled']} with an empty "
                f"pending queue")

        # 4. Both lanes really did drain the one pool, and the answer is right.
        lanes = collections.Counter(client.lane_of[id(f)] for f in client.created)
        assert lanes["gpu"] and lanes["cpu"], f"one lane did no work: {lanes}"
        assert len(client.created) == n                  # one future per chunk
        assert len(seen) == n                            # every chunk streamed
        np.testing.assert_array_equal(result, src.compute())

    def test_pinned_submits_carry_soft_lane_placement(self):
        """The measurement above is only about the PINNED path if the chunks
        actually went out pinned: every submit names its lane, and placement is
        SOFT (`allow_other_workers`) — a hard pin deadlocks when a worker
        restarts, and is reserved for the gpu_only neural dispatch."""
        src, client, seen, result, initial, steps = _run_pinned_dispatch(
            n_chunks=8)
        assert client.submit_kwargs
        assert all(k.get("workers") for k in client.submit_kwargs), (
            "a pinned dispatch submitted with no workers= restriction")
        assert all(k.get("allow_other_workers") is True
                   for k in client.submit_kwargs)
        assert {tuple(k["workers"]) for k in client.submit_kwargs} == {
            (client.GPU,), (client.CPU,)}

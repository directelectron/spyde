"""test_nav_cluster_handover.py — the threaded navigator fill hands over.

`_start_progressive_nav_compute` picks its path from `self.client`. The cluster
takes ~10 s to come up, so a file opened right after launch finds it None and
takes the THREADED branch — one background thread, one chunk at a time.

That choice used to be permanent. However many workers registered a second
later, the whole fill ran single-threaded, which on a long movie is the
difference between seconds and minutes and reads as "the cluster is idle".

Why it looked like a movie-only bug: a 4D-STEM scan goes through the nav-shape
prompt (a human round trip), so its cluster is always up by the time the tree is
built. A movie opens straight through and loses the race.

These tests DRIVE THE REAL LOOP. `BaseSignalTree._start_progressive_nav_compute`
is called with a duck-typed ``self`` (`_Tree`) whose ``client`` property flips
from None to a fake client after N completed chunks, over a real dask array that
RECORDS every block it computes on the threaded scheduler. The re-entry the
handover performs lands on the stub, so the test can assert what it was handed.

An earlier version of this file re-implemented the handover decision inside the
test and only grepped the production source for a substring. Every drift that
matters passed it: `>` widened to `>=` (handing over when the recompute costs
more than it saves), the `return` after the re-entry dropped (the threaded loop
then keeps computing the chunks the dispatcher is already redoing), or the check
moved BELOW the compute. All three are red here.
"""
from __future__ import annotations

import threading
import time

import dask.array as da
import numpy as np


# ── the smallest `self` the real threaded branch touches ─────────────────────


class _Plot:
    """A navigator plot. ``window_id=None`` makes ``lifecycle.window_computing``
    a documented no-op, so driving the real fill emits no IPC."""

    window_id = None

    def __init__(self):
        self.current_data = None
        self.needs_auto_level = False
        self.painted: list = []

    def set_data(self, arr, levels=None):
        self.painted.append((arr, levels))

    def _emit_histogram(self, *a, **k):
        pass

    def update(self):
        pass


class _NavPlotManager:
    def __init__(self, plot):
        self.plot_windows = {"nav-pw": object()}
        self.plots = {"nav-pw": [plot]}


class _NavSignal:
    def __init__(self):
        self.data = None


class _Tree:
    """Duck-typed ``self`` for the REAL ``BaseSignalTree`` fill.

    Only what the threaded branch actually touches is provided, deliberately: a
    refactor that reaches for a new attribute fails loudly here instead of
    quietly skipping the code under test.
    """

    def __init__(self, client_after: int, deep_targets=None):
        self._client_after = client_after     # chunks before the cluster "starts"
        self.cluster = object()               # stands in for a distributed Client
        self.chunks_done = 0                  # advanced by _emit_nav_progress
        self.handed_over_with = None          # (nav_dask, deep) at the re-entry
        self.handover_after_chunks = None
        self.plot = _Plot()
        self.nav_signal = _NavSignal()
        self.navigator_signals = {"base": [self.nav_signal]}
        self.navigator_plot_manager = _NavPlotManager(self.plot)
        self.source_path = None
        self.sidecar_saved: list = []
        self.deep_paints: list = []
        self._deep_targets = deep_targets
        self._pending_nav_dask = None
        self._pending_nav_deep = False

    # The seam under test: the cluster registers PART WAY THROUGH the fill.
    @property
    def client(self):
        return self.cluster if self.chunks_done >= self._client_after else None

    def _emit_nav_progress(self, frac):
        # The real loop calls this once per COMPLETED chunk, right after its own
        # `done_chunks += 1` — so this counter tracks `done_chunks` exactly.
        self.chunks_done += 1

    def _save_nav_sidecar(self, arr):
        self.sidecar_saved.append(arr)

    def _deep_nav_targets(self):
        return self._deep_targets

    def _paint_deep_nav(self, acc, targets, *, final: bool = False):
        self.deep_paints.append((acc.copy(), targets, final))

    def _commit_deep_nav(self, acc, nav_signals):
        nav_signals[0].data = acc

    def _start_progressive_nav_compute(self, nav_dask=None, deep=None):
        """THE RE-ENTRY. The real loop hands over by calling this on itself
        rather than switching path in place (the distributed branch owns
        cancellation, the sidecar save and the final repaint)."""
        self.handed_over_with = (nav_dask, deep)
        self.handover_after_chunks = self.chunks_done


def _counting_nav_dask(n_chunks: int, tail: tuple = (4,)):
    """A lazy nav array of ``n_chunks`` nav-chunks that records each block it
    computes, so the test can see how much work the THREADED path really did.

    The 1 ms per block also guarantees the fill thread is still alive when
    ``_run_real_fill`` goes looking for it.
    """
    rows = 2
    base = np.arange(rows * n_chunks * int(np.prod(tail)), dtype=np.float32)
    base = base.reshape((rows * n_chunks,) + tail)
    computed: list = []
    lock = threading.Lock()

    def _count(block):
        # dask calls the block fn once on an EMPTY array to infer meta at graph
        # construction time — that is not a chunk anyone asked for.
        if block.size:
            with lock:
                computed.append(block.shape)
            time.sleep(0.001)
        return block

    arr = da.from_array(base, chunks=(rows,) + tail).map_blocks(
        _count, dtype=np.float32)
    return arr, base, computed


def _run_real_fill(tree, nav_dask, deep=False, caplog=None, timeout=60.0):
    """Run the REAL fill on ``tree`` and wait for its background thread to end."""
    from spyde.signal_tree import BaseSignalTree

    before = {t.ident for t in threading.enumerate()}
    # The REAL production function, with the stub as `self` — not a copy of it.
    BaseSignalTree._start_progressive_nav_compute(tree, nav_dask, deep=deep)

    th = None
    deadline = time.monotonic() + 5.0
    while th is None and time.monotonic() < deadline:
        for t in threading.enumerate():
            if t.name == "nav-threaded" and t.ident not in before:
                th = t
                break
        else:
            time.sleep(0.002)
    assert th is not None, (
        "the fill never started its 'nav-threaded' thread — the THREADED "
        "branch was not taken (is `client` really None at entry?)")
    th.join(timeout)
    assert not th.is_alive(), "the navigator fill thread never finished"
    if caplog is not None:
        # The fill swallows exceptions into logger.exception; without this a
        # crashed fill reads as "it just didn't hand over".
        bad = [r for r in caplog.records if r.levelno >= 40]
        assert not bad, f"the fill logged an error: {[r.getMessage() for r in bad]}"
    return th


class TestHandover:
    def test_hands_over_to_the_dispatcher_once_the_cluster_appears(self, caplog):
        """The real loop must re-enter itself the FIRST time it sees a client,
        and must stop computing on the threaded path when it does."""
        from spyde.signal_tree import _NAV_HANDOVER_MIN_CHUNKS

        total, after = 20, 3
        # Isolate the client flip: the threshold is nowhere near binding here.
        assert total - after > _NAV_HANDOVER_MIN_CHUNKS
        nav_dask, _base, computed = _counting_nav_dask(total)
        tree = _Tree(client_after=after)

        _run_real_fill(tree, nav_dask, caplog=caplog)

        assert tree.handed_over_with is not None, (
            "the fill never handed over — the whole movie would run "
            "single-threaded however many workers registered meanwhile")
        handed_dask, handed_deep = tree.handed_over_with
        assert handed_dask is nav_dask, (
            "the handover re-entered with a DIFFERENT array than the fill was "
            "given; the dispatcher would fill the navigator from the wrong sum")
        assert handed_deep is False
        assert tree.handover_after_chunks == after, (
            f"handed over after {tree.handover_after_chunks} chunks, expected "
            f"{after} — the check runs BEFORE each chunk's compute, so it must "
            f"fire on the first iteration that sees a client")
        # The threaded path stops dead at the handover: the dispatcher is about
        # to recompute these chunks, and a loop that kept going would race it.
        assert len(computed) == after, (
            f"{len(computed)} chunks computed on the threaded path but the "
            f"handover happened after {after} — the loop kept computing past "
            f"the re-entry (a dropped `return`)")
        assert len(tree.plot.painted) == after      # progressive up to handover
        # The tail belongs to the distributed branch now, not to this thread.
        assert tree.sidecar_saved == []
        assert tree.nav_signal.data is None

    def test_does_not_hand_over_with_only_the_threshold_left(self, caplog):
        """Handing over recomputes what is already painted, so the threshold is
        strictly greater-than: with exactly ``_NAV_HANDOVER_MIN_CHUNKS`` left it
        costs more than it saves. This pins the boundary itself — the cluster is
        visible for all but the first iteration, so only the threshold can be
        holding the handover back."""
        from spyde.signal_tree import _NAV_HANDOVER_MIN_CHUNKS

        total = _NAV_HANDOVER_MIN_CHUNKS + 1     # remaining == MIN at done == 1
        nav_dask, base, computed = _counting_nav_dask(total)
        tree = _Tree(client_after=1)

        _run_real_fill(tree, nav_dask, caplog=caplog)

        assert tree.client is not None, "the cluster never became visible"
        assert tree.handed_over_with is None, (
            f"handed over with exactly {_NAV_HANDOVER_MIN_CHUNKS} chunks "
            f"remaining — the threshold must be strictly greater-than")
        # …and the threaded fill still finished the job properly.
        assert len(computed) == total
        np.testing.assert_array_equal(tree.nav_signal.data, base)
        assert len(tree.sidecar_saved) == 1

    def test_never_hands_over_without_a_cluster(self, caplog):
        total = 12
        nav_dask, base, computed = _counting_nav_dask(total)
        tree = _Tree(client_after=10_000)        # the cluster never arrives

        _run_real_fill(tree, nav_dask, caplog=caplog)

        assert tree.handed_over_with is None
        assert len(computed) == total
        np.testing.assert_array_equal(tree.nav_signal.data, base)
        assert len(tree.sidecar_saved) == 1

    def test_handover_preserves_the_deep_flag(self, caplog):
        """A 5-D+ fill computes the DEEP ``(…lead…, y, x)`` accumulator that
        drives BOTH navigators. Re-entering without ``deep`` would make the
        dispatcher path reduce that array to a single, wrong navigator."""
        total, after = 20, 2
        nav_dask, _base, computed = _counting_nav_dask(total, tail=(3, 4))
        targets = ("top-plot", "child-plot", "top-selector")
        tree = _Tree(client_after=after, deep_targets=targets)

        _run_real_fill(tree, nav_dask, deep=True, caplog=caplog)

        assert tree.handed_over_with is not None, "the deep fill never handed over"
        handed_dask, handed_deep = tree.handed_over_with
        assert handed_dask is nav_dask, (
            "the deep accumulator was reduced before the handover — the "
            "dispatcher must receive the SAME (…lead…, y, x) array")
        assert handed_deep is True, (
            "the re-entry dropped the deep flag; the dispatcher would then sum "
            "the accumulator into the wrong navigator")
        assert tree.handover_after_chunks == after
        assert len(computed) == after
        # The deep branch paints through _paint_deep_nav, never plot.set_data.
        assert len(tree.deep_paints) == after
        assert tree.plot.painted == []

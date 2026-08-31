"""
The live SIGNAL half of a progressively-filled result window.

A progressive action (Find Vectors) opens a navigator + signal window early and
fills the navigator block by block. These tests pin the contract that the signal
plot comes alive too:

  (a) each landing block paints one sample position's frame;
  (b) a position whose block has already landed renders on demand, so dragging
      the navigator over a computed region shows its real value immediately —
      while an un-computed position returns ``None`` so the last good frame
      stays up (no flash).

plus the pieces that make it safe: a deterministic-per-block sample position, a
render that agrees pixel-for-pixel with the finalized display, the hand-back on
close (which must NOT clobber the real render display), and the no-navigator
window (the Orientation/EBSD IPF map) being a documented no-op.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions.find_vectors.live_frames import LiveVectorFrames
from spyde.actions.live_signal import (
    ProgressiveSignalPreview, _block_sample_index, attach_signal_preview,
)
from spyde.signals.diffraction_vectors import SpyDEDiffractionVectors, _AxisLite
from spyde.tests.migrated._async import drain_loop, quiesce, why_busy


# ── helpers ───────────────────────────────────────────────────────────────────

def _block(ny, nx, n_slots, peaks):
    """A padded peaks block: NaN everywhere, ``peaks[(iy, ix)]`` filled in.

    Columns are the batch's own layout — (ky_px, kx_px, intensity)."""
    arr = np.full((ny, nx, n_slots, 3), np.nan, dtype=np.float32)
    for (iy, ix), rows in peaks.items():
        for k, row in enumerate(rows):
            arr[iy, ix, k] = row
    return arr


def _result_tree(session, nav=(4, 5), sig=(16, 16)):
    """A Find-Vectors-shaped result tree: lazy zero placeholder + a count-map
    navigator override, i.e. a real navigator + signal pair."""
    import dask.array as da
    import hyperspy.api as hs
    from spyde.actions.commit import open_result_tree
    from spyde.drawing.selectors import CrosshairSelector

    shape = tuple(nav) + tuple(sig)
    sig_placeholder = hs.signals.Signal2D(
        da.zeros(shape, chunks=shape, dtype=np.float32)).as_lazy()
    nav_sig = hs.signals.BaseSignal(np.zeros(nav, dtype=np.float32)).T
    return open_result_tree(session, title="Preview", signal=sig_placeholder,
                            navigator_override=nav_sig,
                            selector_type=CrosshairSelector)


def _nav_selector(tree):
    npm = tree.navigator_plot_manager
    return list(npm.all_navigation_selectors)[0]


# ── the deterministic sample position ─────────────────────────────────────────

class TestBlockSampling:
    def test_sample_is_inside_the_block(self):
        sl = (slice(4, 8), slice(10, 16))
        for _ in range(20):
            iy, ix = _block_sample_index(sl)
            assert 4 <= iy < 8 and 10 <= ix < 16

    def test_sample_is_deterministic_per_block(self):
        """Reproducible regardless of the (arbitrary) order blocks land in —
        that is what makes 'a random position from each block' testable."""
        sl = (slice(0, 3), slice(6, 9))
        assert _block_sample_index(sl) == _block_sample_index(sl)

    def test_different_blocks_sample_different_positions(self):
        picks = {_block_sample_index((slice(y, y + 4), slice(x, x + 4)))
                 for y in range(0, 16, 4) for x in range(0, 16, 4)}
        assert len(picks) == 16          # every block picks its own position

    def test_single_position_block(self):
        assert _block_sample_index((slice(2, 3), slice(5, 6))) == (2, 5)


# ── rendering a position from its raw peaks block ─────────────────────────────

class TestLiveVectorFrames:
    def test_renders_a_disk_at_the_peak(self):
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        store.add((slice(0, 2), slice(0, 2)),
                  _block(2, 2, 4, {(1, 0): [(5.0, 9.0, 3.0)]}))
        frame = store.render((1, 0))
        assert frame.shape == (16, 16)
        assert frame[5, 9] == pytest.approx(3.0)      # ky → row, kx → column
        assert frame[0, 0] == 0.0

    def test_unknown_position_renders_none(self):
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        store.add((slice(0, 2), slice(0, 2)), _block(2, 2, 4, {}))
        assert store.render((3, 3)) is None           # no block covers it

    def test_empty_position_renders_blank_not_none(self):
        """A computed position with no peaks is a BLACK frame, not 'unknown' —
        the user must be able to tell 'nothing here' from 'not computed yet'."""
        store = LiveVectorFrames(sig_hw=(8, 8), kernel_radius_px=1)
        store.add((slice(0, 1), slice(0, 1)), _block(1, 1, 4, {}))
        frame = store.render((0, 0))
        assert frame is not None and not frame.any()

    def test_matches_the_finalized_render(self):
        """The preview frame and the frame SpyDEDiffractionVectors renders after
        the batch must be identical — otherwise the display visibly jumps when
        the run finalizes."""
        peaks = [(5.0, 9.0, 3.0), (11.0, 4.0, 7.0)]
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        store.add((slice(1, 2), slice(2, 3)), _block(1, 1, 4, {(0, 0): peaks}))

        flat = np.zeros((len(peaks), 6), dtype=np.float32)
        flat[:, 0] = 2          # nav_x
        flat[:, 1] = 1          # nav_y
        flat[:, 2] = [p[1] for p in peaks]      # kx (unit-calibrated)
        flat[:, 3] = [p[0] for p in peaks]      # ky
        flat[:, 4] = -1.0
        flat[:, 5] = [p[2] for p in peaks]
        axes = [_AxisLite(scale=1.0, offset=0.0, size=16),
                _AxisLite(scale=1.0, offset=0.0, size=16)]
        vecs = SpyDEDiffractionVectors.from_arrays(
            flat_buffer=flat, full_nav_shape=(4, 5), sig_shape=(16, 16),
            sig_axes=axes, kernel_radius_px=2.0, kernel_radius_data=2.0)

        assert np.array_equal(store.render((1, 2)), vecs.render_frame(1, 2))

    def test_store_is_capped(self, monkeypatch):
        """A pathological run cannot grow the retained blocks without bound; an
        unretained block simply is not previewable."""
        import spyde.actions.find_vectors.live_frames as lf
        monkeypatch.setattr(lf, "MAX_STORE_BYTES", 16)
        store = LiveVectorFrames(sig_hw=(8, 8), kernel_radius_px=1)
        big = _block(4, 4, 8, {(0, 0): [(1.0, 1.0, 1.0)]})
        assert store.add((slice(0, 4), slice(0, 4)), big) is False
        assert store.render((0, 0)) is None

    def test_compaction_drops_the_padding(self):
        """512 NaN slots per position must not be retained — only the block's
        real maximum peak count."""
        store = LiveVectorFrames(sig_hw=(8, 8), kernel_radius_px=1)
        store.add((slice(0, 2), slice(0, 2)),
                  _block(2, 2, 512, {(0, 0): [(1.0, 1.0, 1.0)]}))
        assert store._bytes < 2 * 2 * 512 * 3 * 4 / 10


# ── the preview on a real navigator + signal window ───────────────────────────

class TestProgressiveSignalPreview:
    def test_attaches_to_a_navigator_plus_signal_window(self, window):
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        assert preview is not None
        assert tree._live_signal_preview is preview
        preview.close()

    def test_no_navigator_window_is_a_no_op(self, window):
        """The Orientation / EBSD IPF result is a single 2-D plot with no
        navigator — nothing to preview into, and that must not be an error."""
        from spyde.actions.commit import open_result_tree
        session = window["window"]
        ipf = open_result_tree(session, title="IPF",
                               data=np.zeros((6, 7), dtype=np.float32))
        assert attach_signal_preview(session, ipf, render=lambda i: None,
                                     nav_shape=(6, 7)) is None
        assert getattr(ipf, "_live_signal_preview", None) is None

    def test_uncomputed_position_returns_none(self, window):
        """No frame for a position the batch has not reached — _run_update skips
        the paint, so the last good frame stays up instead of flashing black."""
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        sel = _nav_selector(tree)
        assert preview.slice_fn(sel, None, [[1, 2]]) is None
        preview.close()

    def test_computed_position_renders_on_demand(self, window):
        """(b) — drag over an already-computed region and you get its value."""
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        blk = _block(2, 2, 4, {(0, 1): [(6.0, 7.0, 4.0)]})
        store.add((slice(2, 4), slice(0, 2)), blk)
        preview.note_block((slice(2, 4), slice(0, 2)))

        # indices are [[ix, iy]] → nav (iy=2, ix=1)
        frame = preview.slice_fn(_nav_selector(tree), None, [[1, 2]])
        assert frame is not None and frame[6, 7] == pytest.approx(4.0)
        # A position in a block that has NOT landed still returns None.
        assert preview.slice_fn(_nav_selector(tree), None, [[3, 0]]) is None
        preview.close()

    def test_served_and_declined_reads_are_counted(self, window):
        """The (b) counters the e2e spec asserts on.

        A pixel signature cannot tell "the drag was answered" apart from "a
        block landed just then" — both repaint the same plot — so the backend
        counts navigator-driven reads itself and the spec reads that count.

        Asserted as DELTAS around each call, not as absolute totals. The preview
        installs its slice function on the live navigator selector, so the
        session's own background updates (the initial paint, a settle re-fire)
        legitimately call it too and bump the same counters. Absolute totals made
        this test a race that only lost on a loaded CI machine; what it actually
        means to pin is that a declined read increments `reads_declined` by one
        and a served read increments `frames_served` by one.
        """
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        sel = _nav_selector(tree)

        def counts():
            return preview.frames_served, preview.reads_declined

        # Nothing computed yet → this read is declined, nothing served.
        served, declined = counts()
        assert preview.slice_fn(sel, None, [[1, 2]]) is None
        assert counts() == (served, declined + 1)

        store.add((slice(2, 4), slice(0, 2)),
                  _block(2, 2, 4, {(0, 1): [(6.0, 7.0, 4.0)]}))
        preview.note_block((slice(2, 4), slice(0, 2)))

        # Now the SAME position is answered from the computed region.
        served, declined = counts()
        assert preview.slice_fn(sel, None, [[1, 2]]) is not None
        assert preview.frames_served == served + 1
        # A position outside the landed block is still declined.
        served, declined = counts()
        assert preview.slice_fn(sel, None, [[3, 0]]) is None
        assert counts() == (served, declined + 1)
        preview.close()

    # Formerly a rerun-marked KNOWN FLAKE (1-2 of 13 CI jobs, never locally):
    # it asserted the PIXELS right after note_block, while
    # `lifecycle.paint_signal_plots` swallowed a failed `set_data` at DEBUG and
    # the attempt counter incremented around the call — so a paint that did
    # not land left `frames_painted == 1` asserting fine and
    # `current_data.max() == 0` failing.  paint_signal_plots now RETURNS the
    # success count and the preview records it as `frames_landed`, so the test
    # asserts the paint actually landed instead of inferring it from pixels a
    # concurrent navigator repaint can race. The paint is MARSHALLED (the
    # fixtures register a loop, as the app does), so wait for it rather than
    # reading the counter on the line after — it used to be readable there only
    # because a session with no loop ran the callback inline.
    def test_block_paints_a_sample_frame(self, window):
        """(a) — a landing block puts a frame on the signal plot."""
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        # Park the navigator away from the block so the SAMPLE path is what runs
        # (a parked navigator re-fires the selector instead — its own test).
        _nav_selector(tree).current_indices = np.array([[4, 0]])
        sl = (slice(2, 4), slice(3, 5))
        peaks = {(iy, ix): [(4.0, 4.0, 9.0)] for iy in range(2) for ix in range(2)}
        store.add(sl, _block(2, 2, 4, peaks))
        preview.note_block(sl)

        assert preview.frames_painted == 1
        assert preview.blocks_seen == 1
        assert quiesce(session), why_busy(session)
        assert preview.frames_landed == 1, \
            "the sample paint's set_data did not land on the signal plot"
        assert tree.signal_plots[0].current_data is not None
        preview.close()

    def test_sample_paints_are_throttled(self, window):
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        _nav_selector(tree).current_indices = np.array([[4, 0]])
        for iy in range(0, 4, 2):
            sl = (slice(iy, iy + 2), slice(0, 2))
            store.add(sl, _block(2, 2, 4, {(0, 0): [(4.0, 4.0, 9.0)]}))
            preview.note_block(sl)
        assert preview.blocks_seen == 2
        assert preview.frames_painted == 1        # the second was throttled
        assert preview.ready_count == 8           # …but readiness still landed
        preview.close()

    def test_user_drag_holds_off_the_sampler(self, window):
        """A landing block must not yank the frame away from a user who is
        dragging the navigator."""
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        _nav_selector(tree).current_indices = np.array([[4, 0]])
        preview.slice_fn(_nav_selector(tree), None, [[0, 0]])   # user read
        sl = (slice(0, 2), slice(0, 2))
        store.add(sl, _block(2, 2, 4, {(0, 0): [(4.0, 4.0, 9.0)]}))
        preview.note_block(sl)
        assert preview.frames_painted == 0
        preview.close()

    def test_parked_navigator_refires_instead_of_sampling(self, window):
        """Arrive at a position before it computes and it fills in when the
        block lands — the other half of 'a computed region is readable'."""
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        sel = _nav_selector(tree)
        fired = []
        sel.delayed_update_data = lambda force=False, **kw: fired.append(force)
        sel.current_indices = np.array([[1, 0]])          # nav (iy=0, ix=1)

        sl = (slice(0, 2), slice(0, 2))
        store.add(sl, _block(2, 2, 4, {(0, 1): [(4.0, 4.0, 9.0)]}))
        preview.note_block(sl)
        assert fired == [True]
        assert preview.frames_painted == 0     # the re-fire wins over the sample
        preview.close()

    def test_close_restores_the_original_slice_fn(self, window):
        session = window["window"]
        tree = _result_tree(session)
        sel = _nav_selector(tree)
        child = tree.signal_plots[0]
        original = sel.children[child]
        preview = attach_signal_preview(session, tree, render=lambda i: None,
                                        nav_shape=(4, 5))
        assert sel.children[child] is preview.slice_fn
        preview.close()
        assert sel.children[child] is original
        assert getattr(tree, "_live_signal_preview", None) is None

    def test_close_does_not_clobber_the_final_display(self, window):
        """_finalize installs the real render display BEFORE the preview closes;
        restoring the placeholder slice over it would paint the finished window
        black."""
        session = window["window"]
        tree = _result_tree(session)
        sel = _nav_selector(tree)
        child = tree.signal_plots[0]
        preview = attach_signal_preview(session, tree, render=lambda i: None,
                                        nav_shape=(4, 5))

        def final_fn(selector, plot, indices):
            return np.ones((16, 16), dtype=np.float32)

        sel.children[child] = final_fn          # what _install_render_display does
        preview.close()
        assert sel.children[child] is final_fn

    def test_out_of_bounds_index_is_not_ready(self, window):
        session = window["window"]
        tree = _result_tree(session)
        preview = attach_signal_preview(session, tree, render=lambda i: None,
                                        nav_shape=(4, 5))
        assert preview.is_ready((99, 99)) is False
        assert preview.is_ready((0,)) is False
        preview.close()


# ── the Find-Vectors wiring ───────────────────────────────────────────────────

class TestFindVectorsWiring:
    # Formerly rerun-marked, same flake family as
    # test_block_paints_a_sample_frame above (a paint that did not land while
    # the counters said it did).  Deflaked the same way: the assertion is now
    # on the recorded set_data success count (`frames_landed`), not on the
    # attempt counter.
    def test_batch_attaches_a_preview_and_feeds_it(self, stem_4d_dataset,
                                                   monkeypatch):
        """``_start_batch`` must hand the compute an ``on_chunk_block`` that
        renders into the result window's signal plot. The compute is stubbed out
        (no cluster in the test session) and its callback driven directly, which
        is exactly what a landing Dask chunk does."""
        import spyde.actions.find_vectors_action as fva
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        plot = tree.signal_plots[0]

        nav = tuple(int(s) for s in tree.root.data.shape[:2])
        sig_h = int(tree.root.axes_manager.signal_axes[1].size)
        sig_w = int(tree.root.axes_manager.signal_axes[0].size)
        captured: dict = {}

        def fake_compute(src, p, **kw):
            """Stand in for the batch: fire the per-chunk hand-off exactly like a
            landing Dask chunk, and record what the preview does with it. The
            assertions happen HERE — the batch's teardown closes the preview as
            soon as this returns."""
            on_block = kw.get("on_chunk_block")
            captured["on_chunk_block"] = on_block
            result_tree = session.signal_trees[-1]
            preview = captured["preview"] = result_tree._live_signal_preview
            if preview is None or not callable(on_block):
                captured["done"] = True
                return None
            # Park the navigator OUTSIDE the block below, so this exercises the
            # SAMPLE paint rather than the parked-selector re-fire (each has its
            # own unit test above).
            for sel in result_tree.navigator_plot_manager.all_navigation_selectors:
                sel.current_indices = np.array([[0, 0]])
            blk = np.full((nav[0] - 1, nav[1], 4, 3), np.nan, dtype=np.float32)
            blk[..., 0, :] = (2.0, 3.0, 5.0)         # ky=2, kx=3, intensity=5
            on_block((slice(1, nav[0]), slice(0, nav[1])), blk)
            # The paint is marshalled onto the loop, and everything below reads
            # counters the paint updates — so drain before reading, or `landed`
            # is captured before the paint has run. `drain_loop`, not `quiesce`:
            # this IS the batch worker, so the in-flight count can never be zero
            # from in here.
            drain_loop(session)
            captured["ready"] = preview.ready_count
            captured["frame"] = preview.slice_fn(None, None, [[1, 1]])
            captured["painted"] = preview.frames_painted
            captured["landed"] = preview.frames_landed
            captured["done"] = True
            return None                    # "no result" → no finalize, no error

        monkeypatch.setattr(fva, "_do_compute_vectors", fake_compute)
        monkeypatch.setattr(fva, "emit_error", lambda *a, **k: None)

        fva._start_batch(session, plot, tree, fva._coerce(dict(fva.DEFAULTS)))
        for _ in range(300):
            if captured.get("done"):
                break
            import time
            time.sleep(0.02)
        assert captured.get("done"), "the batch worker never ran"
        assert callable(captured["on_chunk_block"]), \
            "_start_batch must hand the compute an on_chunk_block callback"
        assert captured["preview"] is not None, \
            "the result window must get a live signal preview"
        assert captured["ready"] == (nav[0] - 1) * nav[1]
        assert captured["painted"] >= 1, "the landing block attempted no paint"
        assert captured["landed"] >= 1, \
            "the landing block's paint never landed (set_data failed/swallowed)"
        frame = captured["frame"]
        assert frame is not None and frame.shape == (sig_h, sig_w)
        assert frame[2, 3] == pytest.approx(5.0)

    def test_teardown_closes_the_preview(self, stem_4d_dataset, monkeypatch):
        """The batch's finally-block must hand the signal plot back, whatever
        the run did — a leaked preview would keep intercepting every navigator
        read on a finished window."""
        import spyde.actions.find_vectors_action as fva
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        done = []

        def fake_compute(src, p, **kw):
            done.append(True)
            raise RuntimeError("boom")

        monkeypatch.setattr(fva, "_do_compute_vectors", fake_compute)
        monkeypatch.setattr(fva, "emit_error", lambda *a, **k: None)
        fva._start_batch(session, tree.signal_plots[0], tree,
                         fva._coerce(dict(fva.DEFAULTS)))
        import time
        for _ in range(300):
            if done and getattr(session.signal_trees[-1],
                                "_live_signal_preview", None) is None:
                break
            time.sleep(0.02)
        assert done
        assert getattr(session.signal_trees[-1], "_live_signal_preview",
                       None) is None


class TestOrchestratorPlumbing:
    def test_on_chunk_block_is_optional(self):
        """The signature stays back-compatible — every existing caller passes
        neither the callback nor, in the api/script path, a shm name."""
        import inspect
        from spyde.actions.find_vectors import _do_compute_vectors
        sig = inspect.signature(_do_compute_vectors)
        assert sig.parameters["on_chunk_block"].default is None

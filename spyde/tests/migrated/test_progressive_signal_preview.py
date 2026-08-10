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
close (which must NOT clobber the real render display, pinned as the tree's
reader override), and the no-navigator window (the Orientation/EBSD IPF map)
being a documented no-op.
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
        assert preview.read_frame((2, 1)) is None
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

        frame = preview.read_frame((2, 1))
        assert frame is not None and frame[6, 7] == pytest.approx(4.0)
        # A position in a block that has NOT landed still returns None.
        assert preview.read_frame((0, 3)) is None
        preview.close()

    def test_served_and_declined_reads_are_counted(self, window):
        """The (b) counters the e2e spec asserts on.

        A pixel signature cannot tell "the drag was answered" apart from "a
        block landed just then" — both repaint the same plot — so the backend
        counts navigator-driven reads itself and the spec reads that count.

        Asserted as DELTAS around each call, not as absolute totals. The preview
        answers the live navigator's reads, so the session's own background
        updates (the initial paint, a settle re-fire) legitimately call it too
        and bump the same counters. Absolute totals made
        this test a race that only lost on a loaded CI machine; what it actually
        means to pin is that a declined read increments `reads_declined` by one
        and a served read increments `frames_served` by one.
        """
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))

        def counts():
            return preview.frames_served, preview.reads_declined

        # Nothing computed yet → this read is declined, nothing served.
        served, declined = counts()
        assert preview.read_frame((2, 1)) is None
        assert counts() == (served, declined + 1)

        store.add((slice(2, 4), slice(0, 2)),
                  _block(2, 2, 4, {(0, 1): [(6.0, 7.0, 4.0)]}))
        preview.note_block((slice(2, 4), slice(0, 2)))

        # Now the SAME position is answered from the computed region.
        served, declined = counts()
        assert preview.read_frame((2, 1)) is not None
        assert preview.frames_served == served + 1
        # A position outside the landed block is still declined.
        served, declined = counts()
        assert preview.read_frame((0, 3)) is None
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

    def test_user_drag_latches_off_the_sampler_for_good(self, window):
        """The first crosshair MOVE ends auto-sampling for the rest of the run.

        Not a timed hold: a hold hands the panel back the moment the user pauses
        to look at what they navigated to, which is exactly when it must not.
        Here two blocks land well after the user's read (the throttle interval is
        stepped over between them) and NEITHER may paint.
        """
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        sel = _nav_selector(tree)
        sel.current_indices = np.array([[4, 0]])
        preview._last_index = (0, 0)
        preview.read_frame((0, 1))                     # a MOVE: (0,0) → (0,1)
        assert preview._user_owns is True

        for iy in (0, 2):
            sl = (slice(iy, iy + 2), slice(0, 2))
            store.add(sl, _block(2, 2, 4, {(0, 0): [(4.0, 4.0, 9.0)]}))
            preview._last_paint = 0.0                  # step over the throttle
            preview.note_block(sl)
        assert preview.blocks_seen == 2
        assert preview.frames_painted == 0
        assert preview.ready_count == 8                # the fill itself is unaffected
        preview.close()

    def test_a_refire_at_the_same_position_does_not_latch(self, window):
        """Only a MOVE is user intent. The preview re-fires the parked selector
        when a block lands under it, and the selector's settle timer re-fires at
        a resting position — both are forced reads at an UNCHANGED index, and if
        either latched, one block landing under a resting crosshair would end
        auto-sampling without the user having touched anything."""
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        sel = _nav_selector(tree)
        sel.current_indices = np.array([[4, 0]])
        preview._last_index = (0, 1)
        for _ in range(3):
            preview.read_frame((0, 1))                 # same position, repeatedly
        assert preview._user_owns is False

        sl = (slice(2, 4), slice(3, 5))
        store.add(sl, _block(2, 2, 4, {(0, 0): [(4.0, 4.0, 9.0)]}))
        preview.note_block(sl)
        assert preview.frames_painted == 1             # auto-sampling still alive
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

    def test_close_releases_the_reader_override(self, window):
        session = window["window"]
        tree = _result_tree(session)
        preview = attach_signal_preview(session, tree, render=lambda i: None,
                                        nav_shape=(4, 5))
        assert tree.reader_override_for(tree.root) is preview
        preview.close()
        assert tree.reader_override_for(tree.root) is None
        assert getattr(tree, "_live_signal_preview", None) is None

    def test_close_does_not_clobber_the_final_display(self, window):
        """_finalize pins the real render display BEFORE the preview closes;
        releasing that override would paint the finished window black."""
        session = window["window"]
        tree = _result_tree(session)
        preview = attach_signal_preview(session, tree, render=lambda i: None,
                                        nav_shape=(4, 5))

        class _FinalReader:
            frame_bytes = 0

            def read_frame(self, indices):
                return np.ones((16, 16), dtype=np.float32)

        final = _FinalReader()
        tree.set_reader_override(tree.root, final)   # what _finalize does
        preview.close()
        assert tree.reader_override_for(tree.root) is final

    def test_out_of_bounds_index_is_not_ready(self, window):
        session = window["window"]
        tree = _result_tree(session)
        preview = attach_signal_preview(session, tree, render=lambda i: None,
                                        nav_shape=(4, 5))
        assert preview.is_ready((99, 99)) is False
        assert preview.is_ready((0,)) is False
        preview.close()


# ── the reader the navigator actually reads through ───────────────────────────

class TestPinnedOnTheDispatchedRead:
    """The preview must be the reader the signal plots' navigator read resolves,
    not merely *a* reader on the tree.

    This is the explanation the e2e failure was blamed on for a whole cycle: a
    drag served nothing, and "the preview never reached the dragged link" was
    the natural suspect. It was not — the drag never reached the backend at all
    (the harness had parked the crosshair from the backend, and the press
    hit-tested against a renderer state that no longer matched). Pinning the
    routing here means that explanation can be RULED OUT next time instead of
    investigated.
    """

    def test_the_signal_plots_read_through_the_preview(self, window):
        session = window["window"]
        tree = _result_tree(session)
        preview = attach_signal_preview(session, tree, render=lambda i: None,
                                        nav_shape=(4, 5))
        assert tree.signal_plots, "the result window has no signal plot"
        for plot in tree.signal_plots:
            assert tree.reader_override_for(tree.root, plot) is preview
        preview.close()

    def test_a_real_dispatcher_read_serves(self, window):
        """End to end through the selector: a forced update at a COMPUTED
        position must come back through the preview with a frame."""
        import time
        from spyde.drawing.selectors.base_selector import _nav_dispatcher
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        store.add((slice(2, 4), slice(0, 2)),
                  _block(2, 2, 4, {(0, 1): [(6.0, 7.0, 4.0)]}))
        preview.note_block((slice(2, 4), slice(0, 2)))

        served0 = preview.frames_served
        sel = _nav_selector(tree)
        sel.get_selected_indices = lambda: np.array([[1, 2]])   # [[ix, iy]]
        sel.delayed_update_data(force=True)
        end = time.monotonic() + 5.0
        while time.monotonic() < end and preview.frames_served == served0:
            time.sleep(0.02)
        assert preview.frames_served > served0, (
            "a dispatcher read at a computed position did not reach the preview"
        )
        assert _nav_dispatcher is not None      # the read ran on the real lane
        preview.close()


# ── decline narration ─────────────────────────────────────────────────────────

class TestDeclineNarration:
    """A declined navigator read is narrated at INFO with cumulative counts —
    the log-side counterpart of `reads_declined`, so a drag that serves NOTHING
    is distinguishable in the captured log from a drag whose reads never ran
    (the e2e spec's `declinedCount` parses this line, mirroring `servedCount`)."""

    @staticmethod
    def _decline_records(caplog):
        return [r for r in caplog.records
                if r.name == "spyde.actions.live_signal"
                and "navigator read declined" in r.getMessage()]

    def test_first_decline_always_logs_with_counts(self, window, caplog):
        import logging
        caplog.set_level(logging.INFO, logger="spyde.actions.live_signal")
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        sel = _nav_selector(tree)

        assert preview.read_frame((2, 1)) is None   # nothing computed
        recs = self._decline_records(caplog)
        assert len(recs) == 1
        msg = recs[0].getMessage()
        # WHERE the read landed + the parseable cumulative counts.
        assert "(2, 1)" in msg
        assert "0 served / 1 declined" in msg
        preview.close()

    def test_declines_are_throttled_but_counted(self, window, caplog):
        import logging
        caplog.set_level(logging.INFO, logger="spyde.actions.live_signal")
        session = window["window"]
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        sel = _nav_selector(tree)

        preview.read_frame((2, 1))
        preview.read_frame((2, 2))   # inside the throttle window
        assert preview.reads_declined == 2
        assert len(self._decline_records(caplog)) == 1   # narration throttled

        # Once the throttle window has passed, the next decline narrates the
        # CUMULATIVE count — the spec reads the last line, not a per-event sum.
        preview._last_decline_log = 0.0
        preview.read_frame((2, 3))
        recs = self._decline_records(caplog)
        assert len(recs) == 2
        assert "3 declined" in recs[-1].getMessage()
        preview.close()


# ── the ready-mask aim (test_aim_ready_position) ──────────────────────────────

class TestReadyWalkTarget:
    """The pure target picker behind the `test_aim_ready_position` backend
    action: near the right end of the longest ready run in one row (backed off
    a quarter-run so the walk starts INSIDE the computed region, not on its
    boundary), so a leftward crosshair walk of `ready_left` columns stays over
    computed data."""

    def test_nothing_ready_is_none(self):
        from spyde.backend._session_testharness import _ready_walk_target
        assert _ready_walk_target(np.zeros((4, 6), dtype=bool)) is None

    def test_near_the_right_end_of_the_longest_run(self):
        from spyde.backend._session_testharness import _ready_walk_target
        mask = np.zeros((3, 10), dtype=bool)
        mask[0, 1:4] = True         # run of 3
        mask[2, 2:9] = True         # run of 7 — the winner
        # right end 8, backed off 7//4=1 → start 7, 6 ready columns to its left
        assert _ready_walk_target(mask) == (2, 7, 6)

    def test_longest_run_within_a_row_wins_over_fragments(self):
        from spyde.backend._session_testharness import _ready_walk_target
        mask = np.zeros((2, 12), dtype=bool)
        mask[0, 0:2] = True
        mask[0, 5:9] = True         # longest single run (4) despite row 1's
        mask[1, 0:3] = True         # larger total (3 + 3 = 6, runs of 3)
        mask[1, 6:9] = True
        assert _ready_walk_target(mask) == (0, 7, 3)

    def test_full_row_backs_off_the_edge(self):
        """A fully-ready row must NOT start the walk at the image's right edge:
        the spec converts the target to screen pixels and rounds, so a start on
        the boundary can land one column outside the computed region."""
        from spyde.backend._session_testharness import _ready_walk_target
        mask = np.zeros((2, 12), dtype=bool)
        mask[1, :] = True           # run of 12, right end 11
        assert _ready_walk_target(mask) == (1, 8, 9)   # backed off 12//4=3

    def test_leading_stack_dims_collapse_to_index_zero(self):
        from spyde.backend._session_testharness import _ready_walk_target
        mask = np.zeros((3, 4, 6), dtype=bool)
        mask[0, 1, 2:5] = True      # visible to the spatial aim (3//4=0 backoff)
        mask[2, 3, 0:6] = True      # a later stack plane — ignored
        assert _ready_walk_target(mask) == (1, 4, 3)


class TestAimReadyPositionAction:
    """`test_aim_ready_position` on a real result tree.

    It REPORTS a walk target from the preview's ready mask — as a delta from
    where the crosshair already is — and moves nothing. Moving the widget from
    the backend is what the first version did, and the measured consequence was
    that the e2e spec's press on the redrawn crosshair delivered no pointer
    event at all: the renderer hit-tests against its OWN widget state, which a
    backend-side write had left it disagreeing with. So "the action must not
    touch the widget" is the contract, and these tests pin it.
    """

    @staticmethod
    def _aim_records(caplog):
        return [r for r in caplog.records if "[test-aim]" in r.getMessage()]

    @staticmethod
    def _prepared(session, caplog):
        tree = _result_tree(session)
        store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
        preview = attach_signal_preview(session, tree, render=store.render,
                                        nav_shape=(4, 5))
        sl = (slice(2, 4), slice(0, 2))
        peaks = {(iy, ix): [(4.0, 4.0, 9.0)] for iy in range(2) for ix in range(2)}
        store.add(sl, _block(2, 2, 4, peaks))
        preview.note_block(sl)
        return tree, preview

    def test_reports_a_ready_target_as_a_delta(self, window, caplog):
        import logging
        caplog.set_level(logging.INFO)
        session = window["window"]
        tree, preview = self._prepared(session, caplog)
        sel = _nav_selector(tree)
        sel.current_indices = np.array([[3, 1]])       # crosshair at iy=1, ix=3

        session.dispatch_action({"action": "test_aim_ready_position"})

        aimed = [r for r in self._aim_records(caplog)
                 if "walk target" in r.getMessage()]
        assert aimed, f"no aim narration; got: {caplog.text[-1500:]}"
        msg = aimed[0].getMessage()
        # Ready block rows 2..3 × cols 0..1 → longest run right end = (2, 1).
        assert "walk target iy=2 ix=1" in msg
        assert "from iy=1 ix=3" in msg
        # The DELTA is what the spec converts to a pointer move, plus the nav
        # shape it needs for the index→pixel scale.
        assert "dix=-2 diy=1" in msg
        assert "nav 5x4" in msg
        preview.close()

    def test_moves_nothing(self, window, caplog):
        """The whole point: no widget write, no dispatcher submit, no read."""
        import logging
        caplog.set_level(logging.INFO)
        session = window["window"]
        tree, preview = self._prepared(session, caplog)
        sel = _nav_selector(tree)
        cross = getattr(sel, "_crosshair_selector", sel)
        inner = getattr(cross, "selector", cross)
        w = getattr(inner, "_widget", None) or getattr(cross, "_widget", None)
        before = (float(w.cx), float(w.cy)) if w is not None else None
        served0, declined0 = preview.frames_served, preview.reads_declined

        session.dispatch_action({"action": "test_aim_ready_position"})

        if w is not None:
            assert (float(w.cx), float(w.cy)) == before
        assert (preview.frames_served, preview.reads_declined) == (served0,
                                                                   declined0)
        preview.close()

    def test_without_a_preview_it_narrates_and_does_not_raise(self, window,
                                                              caplog):
        import logging
        caplog.set_level(logging.INFO)
        session = window["window"]
        session.dispatch_action({"action": "test_aim_ready_position"})
        assert any("no live signal preview" in r.getMessage()
                   for r in self._aim_records(caplog))


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
            # own unit test above). The result window's own first navigator
            # update can run at any point in here, so pin what the widget
            # reports too, or that update commits the widget's position.
            outside = np.array([[0, 0]])
            for sel in result_tree.navigator_plot_manager.all_navigation_selectors:
                sel.get_selected_indices = lambda: outside
                sel.current_indices = outside
            # That same first update reads through the preview, which counts
            # as the user driving the navigator and holds the sample paint
            # back. Whether it lands before the block is scheduling, so take
            # the hold out of this test rather than lose the race.
            preview.user_hold = 0.0
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
            captured["frame"] = preview.read_frame((1, 1))
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


class TestResultTreeIsLockedWhileFilling:
    """The result tree is LOCKED for the duration of the batch — no actions, no
    new nodes — and released when the vectors attach.

    The lock is not housekeeping: it is what makes the preview's install-ONCE
    snapshot of the navigator→signal links correct by construction, so the
    interactive-fill read path needs no per-read re-check (which would put a
    branch on the Live-Display nav read, CLAUDE.md §3).
    """

    @staticmethod
    def _run_batch(session, tree, monkeypatch, compute):
        import time
        import spyde.actions.find_vectors_action as fva
        monkeypatch.setattr(fva, "_do_compute_vectors", compute)
        monkeypatch.setattr(fva, "emit_error", lambda *a, **k: None)
        fva._start_batch(session, tree.signal_plots[0], tree,
                         fva._coerce(dict(fva.DEFAULTS)))
        for _ in range(400):
            if len(session.signal_trees) > 1:
                break
            time.sleep(0.01)
        return session.signal_trees[-1]

    def test_locked_during_the_batch_and_released_after(self, stem_4d_dataset,
                                                        monkeypatch,
                                                        captured_messages):
        import threading
        import time
        from spyde.actions.lifecycle import tree_lock
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        inside = threading.Event()
        release = threading.Event()
        seen: dict = {}

        def fake_compute(src, p, **kw):
            result_tree = session.signal_trees[-1]
            seen["lock"] = tree_lock(result_tree)
            # A node-add on the locked tree is REFUSED, and says why.
            before = len(list(result_tree.walk()))
            result_tree.add_node(result_tree.root, result_tree.root, "rebin")
            seen["nodes_added"] = len(list(result_tree.walk())) - before
            inside.set()
            release.wait(10.0)
            return None                       # no result → no finalize

        result_tree = self._run_batch(session, tree, monkeypatch, fake_compute)
        assert inside.wait(10.0), "the batch worker never ran"
        assert seen["lock"] == "Find Diffraction Vectors"
        assert seen["nodes_added"] == 0, "a node was added to a locked tree"
        errs = [m for m in captured_messages if m.get("type") == "error"]
        assert any("still computing" in m.get("text", "") for m in errs), \
            f"the refusal was silent; errors were {errs}"

        release.set()
        end = time.monotonic() + 10.0
        while time.monotonic() < end and tree_lock(result_tree) is not None:
            time.sleep(0.02)
        assert tree_lock(result_tree) is None, \
            "the batch teardown left the result tree locked"
        # …and now a node CAN be added.
        before = len(list(result_tree.walk()))
        result_tree.add_node(result_tree.root, result_tree.root, "rebin")
        assert len(list(result_tree.walk())) == before + 1

    def test_a_toolbar_action_is_refused_while_locked(self, stem_4d_dataset,
                                                      captured_messages):
        """The dispatch gate, on the path a toolbar click takes. Both dispatch
        paths call the same helper, so this pins the wiring rather than the
        rule (which `test_lifecycle.TestResultTreeLock` owns)."""
        from spyde.actions.lifecycle import lock_tree, unlock_tree
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        plot = tree.signal_plots[0]
        ran: list = []
        lock_tree(tree, "Find Diffraction Vectors")
        try:
            session._dispatch_toolbar_action(plot, "Rebin", {})
        finally:
            unlock_tree(tree)
        assert not ran
        errs = [m for m in captured_messages if m.get("type") == "error"]
        assert any("still computing" in m.get("text", "") for m in errs), \
            f"the toolbar action was not refused; errors were {errs}"

    def test_a_staged_action_is_refused_while_locked(self, stem_4d_dataset,
                                                     captured_messages,
                                                     monkeypatch):
        from spyde.actions import registry
        from spyde.actions.lifecycle import lock_tree, unlock_tree
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        plot = tree.signal_plots[0]
        key = next(iter(registry.STAGED_HANDLERS))
        ran: list = []
        monkeypatch.setattr(registry, "resolve_staged",
                            lambda name: lambda *a, **k: ran.append(name))
        import spyde.backend._session_actions as sa
        monkeypatch.setattr(sa, "resolve_staged", registry.resolve_staged)

        lock_tree(tree, "Find Diffraction Vectors")
        try:
            session.dispatch_action({"action": key,
                                     "window_id": plot.window_id,
                                     "payload": {}})
        finally:
            unlock_tree(tree)
        assert ran == [], f"staged handler {key!r} ran on a locked tree"

        session.dispatch_action({"action": key, "window_id": plot.window_id,
                                 "payload": {}})
        assert ran == [key], "the staged handler stayed blocked after unlock"

    def test_toolbar_config_disables_every_action_while_locked(
            self, stem_4d_dataset, captured_messages):
        """The VISIBLE half of the lock: the buttons must render unavailable
        rather than look clickable and error.

        Disabled, not hidden — the `requires_vectors` family HIDES because those
        actions do not apply to the data, where the lock means "not yet", and a
        button that vanishes and comes back reads as a glitch. Every toolbar
        action is disabled because `_dispatch_toolbar_action` refuses every one
        of them (`registry.is_lock_exempt` covers STAGED verbs, which never
        appear in this config).
        """
        from spyde.actions.lifecycle import lock_tree, unlock_tree
        from spyde.drawing.toolbars.plot_control_toolbar import (
            get_toolbar_config_for_plot,
        )
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        state = tree.signal_plots[0].plot_state

        free = get_toolbar_config_for_plot(state)
        assert free, "the fixture window has no toolbar actions to test with"
        assert not any(a.get("disabled") for a in free)

        lock_tree(tree, "Find Diffraction Vectors")
        try:
            locked = get_toolbar_config_for_plot(state)
            assert {a["name"] for a in locked} == {a["name"] for a in free}, \
                "locking must DISABLE the actions, never remove them"
            assert all(a.get("disabled") is True for a in locked)
            assert all("Find Diffraction Vectors" in a["disabled_reason"]
                       for a in locked)
        finally:
            unlock_tree(tree)

        restored = get_toolbar_config_for_plot(state)
        assert not any(a.get("disabled") for a in restored)

    def test_locking_pushes_the_greyed_toolbar_immediately(
            self, stem_4d_dataset, captured_messages):
        """Taking the lock RE-SENDS the config, so the buttons grey out at the
        START of the fill — not only when the batch finishes and something else
        happens to re-send. Both the signal AND the navigator windows: the
        navigator's Add Selector would add the very navigator→signal link the
        lock exists to freeze."""
        from spyde.actions.lifecycle import lock_tree, unlock_tree
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]

        def configs():
            return [m for m in captured_messages
                    if m.get("type") == "toolbar_config"]

        n_before = len(configs())
        lock_tree(tree, "Find Diffraction Vectors")
        sent = configs()[n_before:]
        assert sent, "locking sent no toolbar config"
        assert all(a.get("disabled") for m in sent for a in m["toolbar_actions"])
        nav_ids = {id(p) for lst in
                   tree.navigator_plot_manager.plots.values() for p in lst}
        assert nav_ids & {m["plot_id"] for m in sent}, \
            "the navigator window's toolbar was not greyed"

        n_locked = len(configs())
        unlock_tree(tree)
        after = configs()[n_locked:]
        assert after, "unlocking sent no toolbar config"
        assert not any(a.get("disabled") for m in after
                       for a in m["toolbar_actions"])
        # Idempotent: a second unlock is not a transition, so no more traffic.
        n_after = len(configs())
        unlock_tree(tree)
        assert len(configs()) == n_after

    def test_read_only_and_teardown_verbs_are_exempt(self, stem_4d_dataset,
                                                     captured_messages,
                                                     monkeypatch):
        """`overlay_query` is fired by the RENDERER itself whenever the active
        window changes, so gating it put an unrequested error in the status bar
        on every progressive fill (observed) and left the dock's layer state
        stale. Teardown verbs are exempt for the mirror reason: a lock that can
        trap a caret open is worse than no lock."""
        from spyde.actions import registry
        from spyde.actions.lifecycle import lock_tree, unlock_tree
        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        plot = tree.signal_plots[0]
        ran: list = []
        monkeypatch.setattr(registry, "resolve_staged",
                            lambda name: lambda *a, **k: ran.append(name))
        import spyde.backend._session_actions as sa
        monkeypatch.setattr(sa, "resolve_staged", registry.resolve_staged)

        lock_tree(tree, "Find Diffraction Vectors")
        try:
            for key in ("overlay_query", "fv_close", "movie_cancel"):
                assert key in registry.STAGED_HANDLERS
                session.dispatch_action({"action": key,
                                         "window_id": plot.window_id,
                                         "payload": {}})
        finally:
            unlock_tree(tree)
        assert ran == ["overlay_query", "fv_close", "movie_cancel"]
        assert not [m for m in captured_messages
                    if m.get("type") == "error"
                    and "still computing" in m.get("text", "")]


class TestOrchestratorPlumbing:
    def test_on_chunk_block_is_optional(self):
        """The signature stays back-compatible — every existing caller passes
        neither the callback nor, in the api/script path, a shm name."""
        import inspect
        from spyde.actions.find_vectors import _do_compute_vectors
        sig = inspect.signature(_do_compute_vectors)
        assert sig.parameters["on_chunk_block"].default is None

"""
Can PR #126's nav-read guard starve the progressive vectors preview's
navigator reads? (CI: progressive_signal_preview.spec.ts once served ZERO
frames over a whole drag while the batch was parked — the suspicion was that
``update_from_navigation_selection``'s new ``_nav_readable_data`` skip fired on
every read mid-fill.)

Finding: NO — structurally impossible, and pinned here so it stays that way.

The guard lives INSIDE ``update_from_navigation_selection``
(update_functions.py, the capture-once + ``_nav_readable_data`` skip). But a
progressive result window's navigator→signal reads never run that function
while the preview is installed: ``ProgressiveSignalPreview.install()``
REPLACES the selector's child entry wholesale (``sel.children[child] =
self.slice_fn``, live_signal.py) and ``BaseSelector._run_update`` dispatches
``fn = self.children[child]`` — there is no shared upstream through the
guard. The preview's slice function renders from the retained peaks blocks
(``LiveVectorFrames``) and consults readiness (``is_ready``), never the
signal's ``.data`` — so even parking hyperspy's deepcopy placeholder
(``array([None], dtype=object)``, the exact state the guard skips on) on the
result tree's root for the WHOLE drive changes nothing: every dispatcher read
at a computed position still serves.

The guard applies exactly where #126 put it: the DEFAULT slice path — which is
what the selector falls back to after ``preview.close()``.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from spyde.actions.find_vectors.live_frames import LiveVectorFrames
from spyde.actions.live_signal import attach_signal_preview
from spyde.drawing.selectors.base_selector import _nav_dispatcher


def _guard_skip_records(caplog):
    """The #126 guard's own skip narration (logged at DEBUG in
    spyde.drawing.update_functions)."""
    return [r for r in caplog.records
            if r.name == "spyde.drawing.update_functions"
            and r.getMessage().startswith("nav read skipped")]


def _error_records(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def _wait_dispatcher_idle(timeout: float = 3.0) -> None:
    """Wait until the serial dispatcher has drained its pending queue (same
    helper as test_nav_second_signal_race.py)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with _nav_dispatcher._lock:
            empty = not _nav_dispatcher._pending
        if empty:
            time.sleep(0.05)   # let the in-flight job finish
            return
        time.sleep(0.01)


def _block(ny, nx, n_slots, peaks):
    """A padded peaks block: NaN everywhere, ``peaks[(iy, ix)]`` filled in."""
    arr = np.full((ny, nx, n_slots, 3), np.nan, dtype=np.float32)
    for (iy, ix), rows in peaks.items():
        for k, row in enumerate(rows):
            arr[iy, ix, k] = row
    return arr


def _result_tree(session, nav=(4, 5), sig=(16, 16)):
    """A Find-Vectors-shaped result tree: lazy zero placeholder signal + a
    count-map navigator override (same shape test_progressive_signal_preview
    builds)."""
    import dask.array as da
    import hyperspy.api as hs
    from spyde.actions.commit import open_result_tree
    from spyde.drawing.selectors import CrosshairSelector

    shape = tuple(nav) + tuple(sig)
    sig_placeholder = hs.signals.Signal2D(
        da.zeros(shape, chunks=shape, dtype=np.float32)).as_lazy()
    nav_sig = hs.signals.BaseSignal(np.zeros(nav, dtype=np.float32)).T
    return open_result_tree(session, title="GuardProbe", signal=sig_placeholder,
                            navigator_override=nav_sig,
                            selector_type=CrosshairSelector)


def _nav_selector(tree):
    npm = tree.navigator_plot_manager
    return list(npm.all_navigation_selectors)[0]


def _attach_ready_preview(session, tree):
    """Attach a preview with block (iy 2..3, ix 0..1) already computed, so nav
    positions [[ix=0..1, iy=2..3]] are servable."""
    store = LiveVectorFrames(sig_hw=(16, 16), kernel_radius_px=2)
    preview = attach_signal_preview(session, tree, render=store.render,
                                    nav_shape=(4, 5))
    store.add((slice(2, 4), slice(0, 2)),
              _block(2, 2, 4, {(0, 0): [(6.0, 7.0, 4.0)],
                               (0, 1): [(5.0, 9.0, 3.0)],
                               (1, 0): [(3.0, 2.0, 8.0)],
                               (1, 1): [(8.0, 8.0, 1.0)]}))
    preview.note_block((slice(2, 4), slice(0, 2)))
    return preview


def _drive(sel, ix, iy):
    """Run one forced navigator update through the REAL serial dispatcher at
    widget position [[ix, iy]] (the crosshair's raw report order)."""
    pos = np.array([[int(ix), int(iy)]])
    sel.get_selected_indices = lambda: pos
    sel.delayed_update_data(force=True)
    _wait_dispatcher_idle()


class TestGuardCannotStarvePreviewReads:
    """The hypothesis test: park / hammer the exact ``.data`` state the #126
    guard skips on, drive real dispatcher reads over computed positions, and
    count both the serves and the guard's own skip line."""

    def test_preview_serves_with_the_placeholder_parked_all_along(
            self, window, caplog):
        """Worst case for the hypothesis: the deepcopy placeholder is parked on
        the result signal's ``.data`` for the ENTIRE drag — every read lands
        'mid-rebind'. If the guard were anywhere in this path, every read would
        skip and serve nothing (the CI symptom). Instead every read serves and
        the guard never fires."""
        caplog.set_level(logging.DEBUG, logger="spyde.drawing.update_functions")
        session = window["window"]
        tree = _result_tree(session)
        preview = _attach_ready_preview(session, tree)
        sel = _nav_selector(tree)
        _wait_dispatcher_idle()

        root = tree.root
        old = root.data
        root.data = None       # hyperspy's transient: array([None], dtype=object)
        assert root.data.shape == (1,) and root.data.dtype == object
        served0, declined0 = preview.frames_served, preview.reads_declined
        n = 20
        try:
            for i in range(n):
                # alternate between two COMPUTED positions (nav iy=2..3, ix=0..1)
                _drive(sel, ix=i % 2, iy=2 + (i // 2) % 2)
        finally:
            root.data = old

        # >= not ==: note_block's parked-position re-fire can legitimately add
        # an extra serve at attach time. The claim is "nothing was starved":
        # every driven read served, none declined.
        assert preview.frames_served - served0 >= n, (
            f"only {preview.frames_served - served0}/{n} dispatcher reads served"
            f" ({preview.reads_declined - declined0} declined) — the preview"
            " path stalled")
        assert preview.reads_declined - declined0 == 0
        assert not _guard_skip_records(caplog), (
            "the #126 guard fired on the preview-owned read path: "
            + "\n".join(r.getMessage() for r in _guard_skip_records(caplog)))
        assert not _error_records(caplog), caplog.text[:2000]
        preview.close()

    def test_rebind_hammer_cannot_starve_serving(self, window, caplog):
        """The CI-shaped timing: a thread flips ``.data`` between the real
        binding and the deepcopy placeholder as fast as it can (denser than any
        real progressive fill's operations) while the dispatcher reads computed
        positions. Serving must not miss a single read; zero guard skips."""
        caplog.set_level(logging.DEBUG, logger="spyde.drawing.update_functions")
        session = window["window"]
        tree = _result_tree(session)
        preview = _attach_ready_preview(session, tree)
        sel = _nav_selector(tree)
        _wait_dispatcher_idle()

        root = tree.root
        real = root.data
        stop = threading.Event()

        def hammer():
            while not stop.is_set():
                root.data = None      # the _deepcopy_with_new_data transient
                root.data = real

        t = threading.Thread(target=hammer, daemon=True)
        served0 = preview.frames_served
        n = 40
        t.start()
        try:
            for i in range(n):
                _drive(sel, ix=i % 2, iy=2 + (i // 2) % 2)
        finally:
            stop.set()
            t.join(timeout=2.0)
            root.data = real
        _wait_dispatcher_idle()

        assert preview.frames_served - served0 >= n, (
            f"only {preview.frames_served - served0}/{n} reads served under the"
            " rebind hammer — the preview path can be starved")
        assert not _guard_skip_records(caplog), (
            "the #126 guard fired on the preview-owned read path")
        assert not _error_records(caplog), caplog.text[:2000]
        preview.close()

    def test_guard_lives_only_on_the_default_path_the_preview_replaced(
            self, window, caplog):
        """The seam, pinned from both sides with the SAME parked placeholder:

        - preview installed → the read is ``preview.slice_fn`` → serves;
        - preview closed → the selector falls back to the default
          ``update_from_navigation_selection`` → the #126 guard skips the frame
          (its designed behaviour), quietly.

        So the guard and the preview occupy the same slot ALTERNATELY — the
        guard is not upstream of the preview and cannot have caused a
        zero-served drag on a preview-owned window."""
        caplog.set_level(logging.DEBUG, logger="spyde.drawing.update_functions")
        session = window["window"]
        tree = _result_tree(session)
        preview = _attach_ready_preview(session, tree)
        sel = _nav_selector(tree)
        child = tree.signal_plots[0]
        assert sel.children[child] is preview.slice_fn
        _wait_dispatcher_idle()

        root = tree.root
        old = root.data
        root.data = None
        try:
            served0 = preview.frames_served
            _drive(sel, ix=1, iy=2)
            assert preview.frames_served >= served0 + 1
            assert not _guard_skip_records(caplog)

            preview.close()
            from spyde.drawing.update_functions import (
                update_from_navigation_selection)
            assert sel.children[child] is update_from_navigation_selection

            served1 = preview.frames_served
            _drive(sel, ix=1, iy=3)
            assert len(_guard_skip_records(caplog)) >= 1, (
                "the restored default path should have skipped the placeholder"
                " read via the #126 guard")
            assert preview.frames_served == served1   # nothing served it
            assert not _error_records(caplog), caplog.text[:2000]
        finally:
            root.data = old

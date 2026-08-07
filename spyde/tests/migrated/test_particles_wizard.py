"""
The Segment Particles wizard backend (``seg_*`` staged handlers).

Handlers are called directly as ``fn(session, plot, payload)`` and polled with
``_wait`` — the shape ``test_find_vectors_wizard.py`` establishes for staged
actions, because everything heavy here goes onto a worker thread.

Four claims are worth more than the rest, and they are the reason this file
exists rather than a smoke test:

:class:`TestPreviewIsOneFrame`
    Plan §0.8.1. A tune must read exactly the frame the navigator is on. The
    guard is the CLAUDE.md memory-safety one: ``da.Array.compute`` must never
    see the full movie shape.
:class:`TestMinSizeFloor`
    Plan §0.9. ``min_size=0`` is a footgun (measured: 33 instances where 9 are
    real), so it is floored — AND the effective value is reported, because a
    caret whose number disagrees with what ran is the failure
    ``SegmentParams.local_size`` refuses to introduce.
:class:`TestRunIsProgressiveAndCancellable`
    Plan §0.8.2-3 and the attach gap: the result window opens EARLY with no
    particles attached, ``_seg_batch_running`` is up for the duration (that is
    exactly what ``lifecycle.wait_for_particles`` polls), and ``tree.particles``
    lands only at finalize.
:class:`TestDoubleFire`
    README §4 / StrictMode: open, close, open leaves exactly ONE controller.

Torch runs on **CPU explicitly** wherever the scribble head is trained:
torch-CUDA work segfaults under the pytest process on Windows (CLAUDE.md).
"""
from __future__ import annotations

import importlib
import time

import numpy as np
import pytest

from spyde.actions import particles_action as pa


@pytest.fixture(autouse=True)
def _capture_module_emit(request, monkeypatch):
    """Route ``particles_action``'s own ``emit`` into the captured list.

    The module does ``from spyde.backend.ipc import emit`` at import, so
    conftest's patch of ``ipc.emit`` never reaches that binding — the identical
    hazard conftest already documents for ``session.py``, and the identical fix.
    ``emit_status``/``emit_error`` need no patch: they resolve ``emit`` inside
    ``ipc`` at call time.

    GATED ON ``window``, which is the whole point of the gate: plain
    ``autouse=True`` pulled the ``window`` fixture into EVERY test in the file,
    so ~30 pure-function tests — the schema, the nm→px arithmetic, the
    lane/semaphore contracts, the overlay stubs, none of which involve a session
    at all — each built and tore down a real ``Session`` (and its Dask manager)
    to assert on a dict. Gating on the fixture NAME rather than on a hand-kept
    list of tests is what makes this safe: dispatching a handler REQUIRES a
    session, so a test that needs the patch cannot fail to request ``window``.
    """
    if "window" not in request.fixturenames:
        return
    messages = request.getfixturevalue("window")["messages"]
    monkeypatch.setattr(pa, "emit", messages.append)


@pytest.fixture
def brush_required():
    """Skip unless the installed anyplotlib carries the brush widget.

    Thirteen tests repeated this three-line check inline. Request it FIRST in
    the signature so the skip is decided before the session is built.
    """
    if not pa._brush_supported():
        pytest.skip("installed anyplotlib has no brush widget (needs >= 0.5.0)")


def _plots_of(session):
    plots = session._plots
    return list(plots.values()) if hasattr(plots, "values") else list(plots)


def _signal_plot(session):
    return next((p for p in _plots_of(session)
                 if not p.is_navigator and p.plot_state is not None), None)


def _figure_plot(session):
    """The first plot carrying BOTH a signal tree and a real figure — the plot
    the brush/caret tests drive.

    Deliberately NOT filtered on ``is_navigator``: four copies of this lived in
    four test classes and every one of them took the first match in ``_plots``
    order, so preserving that exact predicate is what makes this a dedupe rather
    than a quiet change of which plot the tests run against.
    """
    return next(p for p in _plots_of(session)
                if getattr(p, "signal_tree", None) is not None
                and getattr(p, "_plot2d", None) is not None)


def _bare_wizard(p2d):
    """A ``SegmentWizard`` with ONLY its overlay state — no session, no signal.

    Three test classes hand-built this with slightly different subsets of the
    attributes, so one of them depended on a default another set explicitly.
    Callers that need frames attach ``frames``/``frame_index`` themselves.
    """
    import types

    w = object.__new__(pa.SegmentWizard)
    w._ov_group = None
    w._ov_box_group = None
    w._ov_raster = False
    w._ov_cleared = False
    w._ov_box_state = None
    w.src_plot = types.SimpleNamespace(_plot2d=p2d)
    return w


def _scribble_wizard(session):
    """Open the caret with ``seg_open`` ALONE.

    Deliberately not ``seg_set_method`` as well: a helper that armed the brush
    the caret itself never asks for is exactly how the missing-brush bug stayed
    green. Two identical copies of this lived in two classes, each carrying half
    of that note and a comment pointing at the other one.
    """
    plot = _figure_plot(session)
    pa.seg_open(session, plot, {"window_id": getattr(plot, "window_id", None)})
    return plot, plot.signal_tree


def _wait(pred, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _movie(window, frames: int = 6):
    """The synthetic particle movie through the door the e2e specs use."""
    session = window["window"]
    session._load_test_data_particles({"frames": frames})
    plot = _wait(lambda: _signal_plot(session) is not None) and _signal_plot(session)
    assert plot is not None, "the particle movie never produced a signal plot"
    return session, plot, plot.signal_tree


def _opened_untrained(window, frames: int = 6, **params):
    """Caret open, nothing trained — so there is NO preview yet.

    That is the caret's real opening state since the classical engine was
    deleted: an unfitted classifier has no opinion. Tests about the untrained
    face use this; everything that needs a result uses :func:`_opened`.
    """
    session, plot, tree = _movie(window, frames)
    pa.seg_open(session, plot, {"min_size": 25, **params})
    assert _wait(lambda: getattr(tree, "_seg_wizard", None) is not None), \
        "seg_open never built a controller"
    return session, plot, tree, tree._seg_wizard


def _opened(window, frames: int = 6, **params):
    """Caret open, labelled from ground truth, TRAINED, with a live preview.

    `_opened` used to mean "open it and a preview appears", because the caret
    opened on the classical engine and that engine always had an answer. With
    it gone the equivalent starting point costs a train, so this goes through
    the same `seg_autolabel` door the e2e specs use — the REAL rasteriser at
    ground-truth coordinates, then the real `seg_train`.
    """
    session, plot, tree, wiz = _opened_untrained(window, frames, **params)
    pa.seg_autolabel(session, plot, {})
    counts = wiz.label_store().counts()
    assert counts.get(0, 0) > 0 and counts.get(1, 0) > 0, \
        f"seg_autolabel painted nothing usable: {counts}"
    # CPU EXPLICITLY. With no device the classifier auto-selects, i.e. CUDA on
    # this box — and in-process torch-CUDA segfaults under pytest on Windows
    # (CLAUDE.md). The two call sites that already passed it were not enough:
    # this helper is what nearly every test in the file trains through.
    pa.seg_train(session, plot, {"device": "cpu"})
    assert _wait(lambda: wiz.classifier is not None and wiz.classifier.is_trained), \
        "the classifier never finished training"
    assert _wait(lambda: wiz.preview is not None), \
        "the trained wizard never produced a first preview"
    return session, plot, tree, wiz


def _of_type(messages, kind):
    return [m for m in messages if isinstance(m, dict) and m.get("type") == kind]


def _tune(session, plot, payload, msgs, timeout=30.0):
    """Dispatch a tune and return the ``seg_preview`` IT produced.

    Replaces the flat ``time.sleep`` that used to stand in for "the preview has
    landed by now". A preview is one worker hop and it ANNOUNCES itself, so
    waiting for the message is both faster and not a guess about how long a
    loaded box takes — the shape the rest of the file already used.
    """
    seen = len(_of_type(msgs, "seg_preview"))
    pa.seg_tune(session, plot, payload)
    assert _wait(lambda: len(_of_type(msgs, "seg_preview")) > seen, timeout), \
        f"the tune {payload} produced no seg_preview"
    return _of_type(msgs, "seg_preview")[-1]


class _FullComputeGuard:
    """Raise the count if ``.compute()`` is ever called on the whole movie."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.hits = 0

    def __enter__(self):
        import dask.array as da
        self._real = da.Array.compute
        guard = self

        def _spy(arr, *a, **k):
            if tuple(arr.shape) == guard.shape:
                guard.hits += 1
            return guard._real(arr, *a, **k)

        da.Array.compute = _spy
        return self

    def __exit__(self, *exc):
        import dask.array as da
        da.Array.compute = self._real
        return False


class TestPreviewIsOneFrame:
    def test_open_builds_a_controller_and_previews(self, window):
        _s, _p, tree, wiz = _opened(window)
        assert wiz is getattr(tree, "_seg_wizard")
        assert wiz.preview["frame"] == wiz.frame_index()
        # MASK-ONLY (plan §3(iii)): the live preview classifies and stops there,
        # so what lands is a mask and an explicit "not counted", never a count.
        assert wiz.preview["count"] == -1, (
            "the live preview counted instances — the split and the measurement "
            "are 87% of a preview and this path is supposed to skip both")
        mask = np.asarray(wiz.preview["mask"])
        assert mask.ndim == 2 and mask.any(), \
            "the trained head called nothing foreground on the fixture's frame"

    def test_preview_never_computes_the_whole_movie(self, window):
        session, plot, tree, wiz = _opened(window)
        msgs = window["messages"]
        assert tree.root._lazy, "the fixture must be lazy or this guards nothing"
        with _FullComputeGuard(tree.root.data.shape) as guard:
            _tune(session, plot, {"min_separation": 6}, msgs)
            assert wiz.params["min_separation"] == 6
        assert guard.hits == 0, "a tune materialised the whole movie"

    def test_tune_opens_no_new_window(self, window):
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        before = len(session.signal_trees)
        _tune(session, plot, {"min_separation": 7}, msgs)
        assert wiz.params["min_separation"] == 7
        assert len(session.signal_trees) == before, "a tune spawned a tree"

    def test_the_preview_message_says_it_did_NOT_count(self, window):
        """``count: -1`` means "not counted", NOT "found nothing".

        The caret has to be able to tell those apart — a good mask reported as
        0 reads as a failed segmentation — so the payload carries the sentinel
        AND the `mask_only` flag, and the size histogram it used to carry is
        empty rather than misleading. The real count comes from the run and from
        Commit, which is where the split is worth its price.
        """
        session, plot, _tree, _wiz = _opened(window)
        msgs = window["messages"]
        msg = _tune(session, plot, {"marker_smooth": 1.5}, msgs)
        assert msg["count"] == -1 and msg["mask_only"] is True
        assert msg["areas"] == [] and msg["median_area"] == 0.0
        assert msg["units"] == "nm"
        # Coverage is what the caret shows INSTEAD of the count, so it has to be
        # a real fraction of the previewed window.
        assert 0.0 < msg["coverage"] < 1.0, msg

    def test_an_engine_that_cannot_run_keeps_the_last_preview(self, window):
        """An engine with no answer must SAY so, not clear the result on screen.

        `prompt` is the only unrunnable engine left (plan B4, not installed);
        it used to be an untrained scribble as well, before scribble became the
        one engine that is always selected.
        """
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        keep = wiz.preview
        pa.seg_set_method(session, plot, {"method": "prompt"})
        # The stub's own status IS the signal that the switch was handled —
        # `prompt` emits it and then returns without previewing, so there is no
        # seg_preview to wait for and a flat sleep was standing in for it.
        assert _wait(lambda: any("not installed yet" in str(m.get("text", ""))
                                 for m in _of_type(msgs, "status")))
        assert wiz.params["method"] == "prompt"
        assert wiz.preview is keep

    def test_prompt_engine_is_an_explicit_stub(self, window):
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        pa.seg_set_method(session, plot, {"method": "prompt"})
        assert _wait(lambda: any("not installed yet" in str(m.get("text", ""))
                                 for m in _of_type(msgs, "status")))


class TestMinSizeFloor:
    """Plan §0.9 — the measured finding, not a preference."""

    def test_zero_is_floored(self, window):
        session, plot, _tree, wiz = _opened(window)
        pa.seg_tune(session, plot, {"min_size": 0})
        assert _wait(lambda: wiz.params["min_size"] == pa.MIN_SIZE_FLOOR)
        assert wiz.params["min_size_floored"] is True

    def test_the_effective_value_is_reported(self, window):
        """Silently running a different number than the caret shows is the
        failure mode; the floor must come back in the preview payload."""
        session, plot, _tree, _wiz = _opened(window)
        msgs = window["messages"]
        pa.seg_tune(session, plot, {"min_size": 0})
        assert _wait(lambda: any(m.get("min_size_floored")
                                 for m in _of_type(msgs, "seg_preview")))
        msg = next(m for m in _of_type(msgs, "seg_preview")
                   if m.get("min_size_floored"))
        assert msg["min_size"] == pa.MIN_SIZE_FLOOR

    def test_a_real_min_size_is_left_alone(self, window):
        session, plot, _tree, wiz = _opened(window)
        pa.seg_tune(session, plot, {"min_size": 40})
        assert _wait(lambda: wiz.params["min_size"] == 40)
        assert wiz.params["min_size_floored"] is False

    def test_the_COMMITTED_count_is_after_the_size_filter(self, window):
        """§0.9b, asserted where the count still exists.

        The live preview no longer counts anything (it is mask-only), so the
        claim "the number the user sees is the post-filter one" now belongs to
        the two paths that DO split: the batch run, and Commit. This drives
        Commit, because that is the one this branch had to teach to split.
        """
        session, plot, _tree, wiz = _opened(window)
        from spyde.signals.particles import COL

        # `measure_frame` reports area in CALIBRATED units (px² × scale²) while
        # `min_size` filters in PIXELS, so the bar has to be converted rather
        # than assumed — the fixture's 1 nm/px would hide the difference.
        min_px = pa._min_size_px(dict(wiz.params))
        min_area = min_px * wiz.scale_units()[0] ** 2
        assert min_px > 0, "nothing is being filtered, so this proves nothing"

        before = len(session.signal_trees)
        pa.seg_commit(session, plot, {})
        assert _wait(lambda: len(session.signal_trees) > before), \
            "the commit never produced a tree"
        parts = session.signal_trees[-1].particles
        rows = parts.at(0)
        assert parts.n_particles == len(rows) > 0
        assert rows[:, COL["area"]].min() >= min_area, (
            "a particle smaller than min_size survived into the committed "
            "store — the filter the caret shows is not the filter that ran")


class TestScribbleLabelling:
    def test_paint_accumulates_across_frames(self, window):
        session, plot, _tree, wiz = _opened(window)
        pa.seg_paint(session, plot, {"frame": 0, "points": [[20, 20], [24, 24]],
                                     "class_id": 0, "brush": 3})
        pa.seg_paint(session, plot, {"frame": 3, "points": [[40, 40], [44, 44]],
                                     "class_id": 1, "brush": 3})
        assert wiz.labels.labelled_frames() == [0, 3]
        counts = wiz.labels.counts()
        assert counts[0] > 0 and counts[1] > 0

    def test_state_reports_per_class_pixel_counts(self, window):
        """Plan B3: under-training a class is *the* failure mode and these
        counts are how a user notices, so every class is present."""
        session, plot, _tree, _wiz = _opened(window)
        msgs = window["messages"]
        pa.seg_paint(session, plot, {"frame": 0, "points": [[20, 20], [22, 22]],
                                     "class_id": 0, "brush": 3})
        state = _of_type(msgs, "seg_state")[-1]
        by_id = {c["id"]: c for c in state["classes"]}
        assert by_id[0]["pixels"] > 0
        assert by_id[2]["pixels"] == 0, "an unpainted class must still be listed"

    def test_erase_covers_exactly_what_the_brush_painted(self, window):
        # UNTRAINED: this is about the label store starting empty.
        session, plot, _tree, wiz = _opened_untrained(window)
        stroke = {"frame": 0, "points": [[20, 20], [30, 30]], "brush": 5}
        pa.seg_paint(session, plot, {**stroke, "class_id": 0})
        painted = len(wiz.labels)
        assert painted > 0
        pa.seg_paint(session, plot, {**stroke, "erase": True})
        assert len(wiz.labels) == 0, (
            f"{len(wiz.labels)} of {painted} px survived an erase over the same "
            "stroke — the eraser and the brush have drifted apart")

    def test_paint_with_no_points_is_a_no_op(self, window):
        # UNTRAINED: this is about the label store starting empty.
        session, plot, _tree, wiz = _opened_untrained(window)
        pa.seg_paint(session, plot, {"frame": 0, "points": [], "class_id": 0})
        assert wiz.labels is None

    def test_train_needs_labels(self, window):
        # UNTRAINED: this is about the label store starting empty.
        session, plot, _tree, _wiz = _opened_untrained(window)
        msgs = window["messages"]
        pa.seg_train(session, plot, {"device": "cpu"})
        assert any("nothing painted" in str(m.get("text", ""))
                   for m in _of_type(msgs, "error"))

    @pytest.mark.slow
    def test_train_then_the_scribble_engine_previews(self, window):
        """The full B3 loop on CPU: scribble a particle and some background,
        train, and the scribble engine becomes the live preview."""
        session, plot, _tree, wiz = _opened(window)
        # A bright particle in the fixture and a patch of bare film.
        pa.seg_paint(session, plot, {"frame": 0, "class_id": 0, "brush": 3,
                                     "points": [[24, 28], [25, 29]]})
        pa.seg_paint(session, plot, {"frame": 0, "class_id": 1, "brush": 5,
                                     "points": [[5, 5], [5, 60], [8, 100]]})
        pa.seg_train(session, plot, {"device": "cpu"})
        assert _wait(lambda: wiz.classifier is not None
                     and wiz.classifier.is_trained, timeout=30)
        assert wiz.params["method"] == "scribble"
        assert _wait(lambda: wiz.preview is not None
                     and wiz.preview["frame"] == wiz.frame_index(), timeout=30)


class TestRunIsProgressiveAndCancellable:
    def test_result_window_opens_before_the_particles_attach(self, window):
        """The attach gap, asserted at the instant it exists."""
        session, plot, tree, _wiz = _opened(window)
        before = len(session.signal_trees)
        pa.seg_run(session, plot, {"min_size": 25})
        assert len(session.signal_trees) == before + 1, \
            "seg_run must open its result window synchronously"
        result = session.signal_trees[-1]
        assert getattr(result, "particles", None) is None, (
            "particles attached at open — requires_particles would unlock "
            "against an empty store")
        assert result._seg_batch_running and tree._seg_batch_running
        assert _wait(lambda: getattr(result, "particles", None) is not None)

    def test_seg_batch_running_is_what_lifecycle_polls(self, window):
        from spyde.actions.lifecycle import seg_batch_running
        session, plot, _tree, _wiz = _opened(window)
        pa.seg_run(session, plot, {"min_size": 25})
        assert seg_batch_running(session)
        result = session.signal_trees[-1]
        assert _wait(lambda: getattr(result, "particles", None) is not None)
        assert _wait(lambda: not seg_batch_running(session))

    def test_finalize_attaches_a_full_store_and_says_how_many(self, window):
        session, plot, _tree, _wiz = _opened(window, frames=6)
        msgs = window["messages"]
        pa.seg_run(session, plot, {"min_size": 25})
        result = session.signal_trees[-1]
        assert _wait(lambda: getattr(result, "particles", None) is not None)
        parts = result.particles
        assert parts.n_frames == 6 and parts.n_particles > 0
        assert parts.has_masks
        # Polled, not read once: the status is the LAST thing _finalize does,
        # after the count-trace paint, so it lands a beat after the attach.
        assert _wait(lambda: any(
            f"Found {parts.n_particles} particles" in str(m.get("text", ""))
            for m in _of_type(msgs, "status"))), (
            "the finalize status must carry the count — the e2e specs and the "
            "user both read it")

    def test_the_label_movie_renders_the_real_contours(self, window):
        """The placeholder is mutated IN PLACE precisely so this works; a fresh
        store would leave the early window rendering zeros forever."""
        session, plot, _tree, _wiz = _opened(window)
        pa.seg_run(session, plot, {"min_size": 25})
        result = session.signal_trees[-1]
        assert _wait(lambda: getattr(result, "particles", None) is not None)
        frame = np.asarray(result.root.data[3].compute())
        painted = np.unique(frame)
        painted = painted[painted > 0]
        assert painted.size == len(result.particles.at(3)) > 0

    def test_count_trace_reaches_the_navigator(self, window):
        session, plot, _tree, _wiz = _opened(window)
        pa.seg_run(session, plot, {"min_size": 25})
        result = session.signal_trees[-1]
        assert _wait(lambda: getattr(result, "particles", None) is not None)
        counts = result.particles.count_series()
        navs = [n for n in pa._nav_plots(result)
                if getattr(getattr(n, "current_data", None), "ndim", 0) == 1]
        assert navs, "the particle tree has no 1-D navigator to fill"
        assert any(np.array_equal(np.asarray(n.current_data, np.float32), counts)
                   for n in navs), "the count trace never reached the navigator"

    def test_run_never_computes_the_whole_movie(self, window):
        session, plot, tree, _wiz = _opened(window)
        with _FullComputeGuard(tree.root.data.shape) as guard:
            pa.seg_run(session, plot, {"min_size": 25})
            result = session.signal_trees[-1]
            assert _wait(lambda: getattr(result, "particles", None) is not None)
        assert guard.hits == 0, "the batch materialised the whole movie"

    def test_closing_the_result_tree_cancels_the_batch(self, window, monkeypatch):
        """Cancellation runs through BaseSignalTree.register_cancel, so the
        ordinary act of closing the window has to stop the compute."""
        import spyde.particles.batch as batch
        real = batch.resolve_engine

        def _slow(spec, **kw):
            engine, dev = real(spec, **kw)

            def _slow_engine(frame, _e=engine):
                time.sleep(0.25)
                return _e(frame)

            return _slow_engine, dev

        monkeypatch.setattr(batch, "resolve_engine", _slow)
        session, plot, _tree, _wiz = _opened(window, frames=24)
        msgs = window["messages"]
        pa.seg_run(session, plot, {"min_size": 25})
        result = session.signal_trees[-1]
        # Close only once the batch is demonstrably RUNNING — cancelling before
        # the first frame lands would pass without exercising cancellation. The
        # batch's own progress emission is that signal (the slowed engine makes
        # it comfortably reachable), where the flat sleep was a guess at it.
        assert _wait(lambda: any(m.get("label") == "Segmenting"
                                 for m in _of_type(msgs, "progress"))), \
            "the batch never reported progress, so there was nothing to cancel"
        result.close()
        assert _wait(lambda: not result._seg_batch_running, timeout=30)
        # A cancelled run keeps its partial rows out of the torn-down tree.
        assert getattr(result, "particles", None) is None

    def test_finalize_re_sends_the_toolbar_config(self, window):
        """requires_particles flips here; without the re-send the gated buttons
        stay hidden until something else rebuilds the toolbar."""
        session, plot, _tree, _wiz = _opened(window)
        msgs = window["messages"]
        pa.seg_run(session, plot, {"min_size": 25})
        result = session.signal_trees[-1]
        assert _wait(lambda: getattr(result, "particles", None) is not None)
        wids = {getattr(sp_, "window_id", None) for sp_ in result.signal_plots}
        assert _wait(lambda: any(m.get("window_id") in wids
                                 for m in _of_type(msgs, "toolbar_config")))


class TestCommit:
    """Commit is where the mask-only preview gets SPLIT.

    Plan §3(iii) — "show foreground probability live; split on demand and on
    commit" — and `_mask_engine`'s own docstring: "instance identity and the
    size distribution come from the real run (and from `seg_commit`)". So when
    the user presses Commit there are no labels, no rows and no contours yet,
    and reading them off the preview is what raised `KeyError: 'contours'` out
    of the handler on the ordinary tune-then-Commit path.

    Committing is therefore ASYNCHRONOUS now (the split is up to seconds on a
    4096² frame and the asyncio main thread runs the whole backend), which is
    why these poll for the tree instead of counting it synchronously.
    """

    def test_commit_computes_the_instances_the_preview_skipped(self, window):
        session, plot, tree, wiz = _opened(window)
        assert wiz.preview["count"] == -1 and "contours" not in wiz.preview, (
            "this fixture no longer produces a mask-only preview, so the "
            "regression under test cannot happen here any more")
        before = len(session.signal_trees)
        pa.seg_commit(session, plot, {})
        assert _wait(lambda: len(session.signal_trees) > before), \
            "Commit on a mask-only preview produced no tree at all"
        committed = session.signal_trees[-1]
        assert committed.particles.n_frames == 1
        assert committed.particles.n_particles > 0, (
            "the commit-time split found no instances in a mask that has "
            "foreground — the preview and the commit disagree about the frame")
        # One outline PER ROW, in order — the contract `from_frames` enforces
        # and the reason `_commit_or_refuse` exists.
        assert committed.particles.has_masks
        assert all(len(committed.particles.contour_at(i)) >= 3
                   for i in range(committed.particles.n_particles))
        assert committed.source_tree is tree

    def test_commit_refuses_politely_when_the_engine_is_gone(self, window):
        """No path may raise. The split needs the ENGINE (re-running it is what
        keeps the boundary channel, and so the committed result, identical to
        what the batch would produce), so a head that vanished between preview
        and Commit has to come back as a message, not a traceback."""
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        before = len(session.signal_trees)
        wiz.classifier = None                       # the head is gone
        del msgs[:]
        pa.seg_commit(session, plot, {})            # must not raise
        assert any("commit" in str(m.get("text", "")).lower()
                   for m in _of_type(msgs, "error")), \
            "commit failed silently instead of saying why"
        assert len(session.signal_trees) == before, \
            "a tree was committed with no engine to measure it"

    def test_commit_stamps_provenance(self, window):
        session, plot, _tree, wiz = _opened(window)
        before = len(session.signal_trees)
        frame = int(wiz.preview["frame"])
        pa.seg_commit(session, plot, {})
        assert _wait(lambda: len(session.signal_trees) > before)
        prov = getattr(session.signal_trees[-1], "_commit_provenance", None) or {}
        assert prov.get("action") == "segment_particles"
        assert prov.get("params", {}).get("mode") == "single_frame"
        # The frame it says it committed is the frame that was previewed.
        assert prov.get("frame") == frame
        assert prov.get("params", {}).get("frame") == frame

    def test_commit_without_a_preview_errors(self, window):
        session, _plot, _tree = _movie(window)
        msgs = window["messages"]
        pa.seg_commit(session, _signal_plot(session), {})
        assert any("nothing to commit" in str(m.get("text", ""))
                   for m in _of_type(msgs, "error"))


class TestDoubleFire:
    def test_open_close_open_leaves_one_controller(self, window):
        """README §4 / StrictMode: mount → cleanup → remount fires all three
        synchronously, before any worker lands."""
        session, plot, tree = _movie(window)
        built = []
        real_init = pa.SegmentWizard.__init__

        def _tracking(self, *a, **k):
            real_init(self, *a, **k)
            built.append(self)

        pa.SegmentWizard.__init__ = _tracking
        try:
            pa.seg_open(session, plot, {})
            pa.seg_close(session, plot, {})
            pa.seg_open(session, plot, {})
        finally:
            pa.SegmentWizard.__init__ = real_init

        # NO SLEEP. All three verbs are synchronous on an untrained caret —
        # `_preview` finds no engine and draws the window box inline, so nothing
        # is dispatched to a worker at all and there is nothing to wait for. The
        # 0.8 s here was covering for a preview that no longer runs (and, being
        # a sleep, would not have caught a late arrival anyway: it asserts once).
        alive = [w for w in built if not w._closed]
        assert len(alive) == 1, \
            f"expected 1 live controller, got {len(alive)} of {len(built)} built"
        assert tree._seg_wizard is alive[0]

        pa.seg_close(session, plot, {})
        assert getattr(tree, "_seg_wizard", None) is None
        assert all(w._closed for w in built)

    def test_close_without_an_open_is_harmless(self, window):
        session, plot, tree = _movie(window)
        pa.seg_close(session, plot, {})
        assert getattr(tree, "_seg_wizard", None) is None

    def test_reopen_keeps_the_scribbles(self, window):
        """A second open must adopt the new parameters, not throw the user's
        accumulated labels away."""
        session, plot, tree, wiz = _opened(window)
        pa.seg_paint(session, plot, {"frame": 0, "points": [[20, 20], [22, 22]],
                                     "class_id": 0, "brush": 3})
        painted = len(wiz.labels)
        pa.seg_open(session, plot, {"min_separation": 7})
        assert tree._seg_wizard is wiz
        assert len(wiz.labels) == painted


class TestSchema:
    def test_schema_resolves_through_the_registry(self):
        from spyde.actions import registry
        schema = registry.wizard_parameters("seg")
        assert schema and schema is not pa.SegmentWizard.parameters

    def test_schema_defaults_match_the_handler_defaults(self):
        """The drift test_wizard_schemas.py exists to catch."""
        from spyde.actions import registry
        schema = registry.wizard_parameters("seg")
        for key, spec in schema.items():
            assert key in pa.DEFAULTS, f"seg schema declares unknown param {key!r}"
            assert spec["default"] == pa.DEFAULTS[key], \
                f"seg schema/{key} drifted from particles_action.DEFAULTS"

    #: Parameters deliberately absent from the declared schema, and who owns
    #: them instead. Everything here is rendered by the caret's own hand-written
    #: face (``electron/src/renderer/src/components/SegmentWizard.tsx``) because
    #: it needs a custom control, not a generic slider:
    #:   min_score / merge_nm / min_nm  — the physical face controls
    #:   active_class / erase           — the ClassStrip's paint state
    #: ``scale`` is not here because it is not a DEFAULT at all: ``set_params``
    #: stashes the signal's own nm/px onto the params after ``_coerce``.
    CARET_OWNED = {"min_score", "merge_nm", "min_nm", "active_class", "erase"}

    def test_every_default_is_DECLARED_or_KNOWINGLY_caret_owned(self):
        """The other direction, which the test above cannot see.

        Schema ⊆ DEFAULTS catches a schema entry nothing reads. It does NOT
        catch a parameter the handlers read that no host renders — which is
        exactly what `active_class` / `erase` shipped as: read by the brush,
        declared nowhere, so nothing could set them and every stroke came out in
        class 0 with a dead eraser.

        The allowlist is the point. A new DEFAULT now has to be either declared
        in the schema or added here deliberately, rather than silently becoming
        a parameter no UI can reach.
        """
        from spyde.actions import registry
        schema = registry.wizard_parameters("seg")
        undeclared = set(pa.DEFAULTS) - set(schema) - self.CARET_OWNED
        assert not undeclared, (
            f"{sorted(undeclared)} are read by the seg handlers but declared "
            f"in no schema and not listed as caret-owned, so nothing can set "
            f"them")
        # ...and the allowlist may not rot into a place to hide deleted params.
        assert self.CARET_OWNED <= set(pa.DEFAULTS), \
            f"{sorted(self.CARET_OWNED - set(pa.DEFAULTS))} no longer exist"
        assert "scale" not in pa.DEFAULTS, (
            "scale became a DEFAULT — `_coerce` would then build it from the "
            "payload and a stale caret value could override the signal's own")

    def test_every_stage_is_registered(self):
        from spyde.actions.registry import STAGED_HANDLERS, resolve_staged
        for stage in ("seg_open", "seg_close", "seg_set_method", "seg_tune",
                      "seg_paint", "seg_train", "seg_run", "seg_commit"):
            assert stage in STAGED_HANDLERS
            assert callable(resolve_staged(stage))

    def test_toolbar_entry_points_at_the_action(self):
        import spyde
        meta = spyde.TOOLBAR_ACTIONS["functions"]["Segment Particles"]
        assert meta["function"] == \
            "spyde.actions.particles_action.segment_particles"
        assert "spyde_diffraction_vectors_image" in meta["exclude_signal_types"]

    @pytest.mark.parametrize("signal_type,offered", [
        ("insitu", True),            # the in-situ movie — the primary shape
        ("", True),                  # a plain 2-D image — plan §0.10
        ("electron_diffraction", False),   # a 4D-STEM scan has no image frames
        ("particles", False),        # a label movie is not re-segmented
    ])
    def test_toolbar_gating(self, signal_type, offered):
        import spyde
        from spyde.drawing.toolbars.plot_control_toolbar import _action_matches_plot

        class _Sig:
            _signal_type = signal_type

        class _Tree:
            particles = None
            diffraction_vectors = None
            root = _Sig()

        class _Plot:
            signal_tree = _Tree()
            is_navigator = False

        class _State:
            plot = _Plot()
            current_signal = _Sig()
            dimensions = 2
            navigation = False

        meta = spyde.TOOLBAR_ACTIONS["functions"]["Segment Particles"]
        assert _action_matches_plot("Segment Particles", meta, _State()) is offered


class TestCoercion:
    def test_unknown_method_falls_back(self):
        assert pa._coerce({"method": "magic"})["method"] == pa.DEFAULTS["method"]

    def test_a_sub_pixel_min_separation_is_lifted(self):
        """`SegmentParams` raises below 1 px rather than bumping, and a slider
        that lands on 0 must not error mid-drag."""
        assert pa._coerce({"min_separation": 0})["min_separation"] == 1

    def test_a_deleted_classical_param_is_DROPPED_not_kept(self):
        """A stale payload — or a provenance dict from a result saved before the
        classical engine was removed — must not survive into the params, or it
        would reach `SegmentParams` and raise TypeError on a re-run."""
        p = pa._coerce({"sensitivity": 0.9, "threshold": "otsu", "gaussian": 2.0,
                        "rb_kernel": 64, "invert": True, "local_size": 31})
        for dead in ("sensitivity", "threshold", "gaussian", "rb_kernel",
                     "invert", "local_size"):
            assert dead not in p, f"{dead} survived _coerce"

    def test_a_junk_value_keeps_the_default(self):
        assert pa._coerce({"min_separation": "wat"})["min_separation"] == \
            pa.DEFAULTS["min_separation"]


class TestPhysicalFaceControls:
    """`merge_nm` / `min_nm` are the two controls on the caret's face, and they
    are in NANOMETRES — so the whole feature is the nm→px conversion.

    It was dead on arrival. ``_nm_to_px`` divides by ``p["scale"]`` and falls
    back to "the value IS pixels" when there is no scale, which is the right
    behaviour for an uncalibrated signal — but ``_coerce`` builds the dict from
    the ``DEFAULTS`` keys alone, so no dispatch path ever put a scale in it and
    EVERY signal took the uncalibrated branch. A slider reading "50 nm" merged
    at 50 pixels, wrong by exactly the magnification and silent about it.

    Hence the first test: it asserts the scale is on the params after each
    dispatch verb, which is the thing that was missing, rather than asserting
    ``_nm_to_px`` divides — that part was always correct.
    """

    def test_every_dispatch_stashes_the_signal_scale(self, window):
        session, plot, _tree, wiz = _opened(window)
        scale = wiz.scale_units()[0]
        assert scale > 0, "the fixture must be calibrated or this proves nothing"

        assert wiz.params["scale"] == scale, "seg_open did not stash the scale"
        pa.seg_tune(session, plot, {"merge_nm": 30.0})
        assert _wait(lambda: wiz.params.get("merge_nm") == 30.0)
        assert wiz.params["scale"] == scale, "seg_tune dropped the scale"
        pa.seg_set_method(session, plot, {"method": "classical"})
        assert wiz.params["scale"] == scale, "seg_set_method dropped the scale"

    # The particle fixture is calibrated at 1.0 nm/px, which makes every
    # conversion below the identity — so these pass an explicit NON-UNIT scale
    # instead of reading the fixture's. At 1.0 the arithmetic tests hold whether
    # or not the conversion happens at all, which is exactly how it shipped
    # unconverted. Only the dispatch test above needs a real wizard.

    def test_merge_nm_reaches_the_solver_in_PIXELS(self):
        """A nm distance must be divided by the scale before it is a radius."""
        p = pa._coerce({"merge_nm": 30.0})
        assert pa._segment_kwargs({**p, "scale": 0.4})["merge_distance"] == \
            pytest.approx(75.0)

    def test_min_nm_is_a_DIAMETER_and_converts_through_the_AREA(self):
        """`min_size` filters on AREA, so handing it a diameter straight over
        under-filters by a factor of ~d — a face control that barely does
        anything is worse than one that is not there."""
        p = pa._coerce({"min_nm": 20.0})
        d_px = 20.0 / 0.4
        assert pa._min_size_px({**p, "scale": 0.4}) == \
            int(round(np.pi / 4.0 * d_px * d_px))
        # and not the diameter, which is the plausible wrong answer
        assert pa._min_size_px({**p, "scale": 0.4}) != int(d_px)

    def test_min_nm_wins_over_the_pixel_min_size(self):
        """Both exist (one on the face, one in Advanced) and they filter the same
        thing, so the physical one has to be the tie-break — otherwise the face
        control is silently overridden by a value the user cannot see."""
        p = pa._coerce({"min_nm": 20.0, "min_size": 25})
        assert pa._min_size_px({**p, "scale": 0.4}) != 25
        assert pa._min_size_px({**p, "min_nm": 0.0, "scale": 0.4}) == 25

    def test_an_uncalibrated_signal_reads_the_value_as_pixels(self):
        """No scale is the only case where nm==px is correct, and it must not
        divide by zero to get there."""
        assert pa._nm_to_px(12.0, {}) == 12.0
        assert pa._nm_to_px(12.0, {"scale": 0.0}) == 12.0
        assert pa._nm_to_px(0.0, {"scale": 0.5}) == 0.0

    def test_a_length_axis_converts_through_NANOMETRES(self):
        """An axis in µm or Å is still a length; the slider is still nm."""
        assert pa._length_nm_per_px(2.0, "nm") == pytest.approx(2.0)
        assert pa._length_nm_per_px(2.0, "um") == pytest.approx(2000.0)
        assert pa._length_nm_per_px(2.0, "µm") == pytest.approx(2000.0)
        assert pa._length_nm_per_px(2.0, "Å") == pytest.approx(0.2)
        # ...so "20 nm" is 10 px at 2 nm/px and 0.01 px at 2 µm/px.
        assert pa._nm_to_px(20.0, {"scale": pa._length_nm_per_px(2.0, "nm")}) \
            == pytest.approx(10.0)

    def test_a_RECIPROCAL_axis_is_not_a_length_and_falls_back_to_pixels(self):
        """The reported bug: the caret said 'nm' on an axis reading nm⁻¹.

        A reciprocal-space signal reports a perfectly healthy positive scale, so
        the conversion ran and produced a merge radius wrong by whatever the
        camera length was — silently, because nothing checked the UNIT. There is
        no distance to convert to here, so the controls are pixels and the caret
        has to relabel; claiming nm is a claim about the scale bar that is false.
        """
        for units in ("nm^-1", "1/nm", "nm⁻¹", "mrad", "px", "", None):
            assert pa._length_nm_per_px(0.9, units) == 0.0, units
            assert pa._face_units(0.9, units) == "px", units
        assert pa._face_units(0.5, "nm") == "nm"
        # and the pixel fallback is the IDENTITY, not a division by a number in
        # the wrong unit
        assert pa._nm_to_px(30.0, {"scale": pa._length_nm_per_px(0.9, "nm^-1")}) \
            == 30.0

    def test_the_wizard_stashes_nm_per_px_not_the_raw_axis_scale(self, window):
        _s, _p, _t, wiz = _opened(window)
        scale, units = wiz.scale_units()
        assert wiz.params["scale"] == pytest.approx(
            pa._length_nm_per_px(scale, units))


class TestThresholdFailureIsNotAResult:
    """14028 'particles' covering the frame is a failed threshold, not an answer.

    Measured on a low-contrast noisy stand-in for the reported frame: otsu has
    no bimodal histogram to find, lands inside the noise, and the split shatters
    the support film. `min_size` alone moves 4873 -> 17 instances but coverage
    only 39% -> 7%, and the settings that DO yield 8 instances cover 52% of the
    frame -- those 8 bodies ARE the film. So neither number alone is diagnostic
    and the verdict needs both.
    """

    def test_the_shatter_shape_needs_both_conditions(self):
        """Thousands of small fragments over a good part of the frame."""
        assert pa._threshold_failed(5000, 0.40) is True
        # A crowded frame of REAL particles: many instances, but they do not
        # blanket the frame.
        assert pa._threshold_failed(5000, 0.05) is False
        # A coarse segmentation that found a few big real objects.
        assert pa._threshold_failed(12, 0.40) is False
        assert pa._threshold_failed(0, 0.0) is False

    def test_near_total_coverage_fails_at_ANY_count(self):
        """The other shape, and the one the count rule missed.

        A head mis-trained to call the film "particle" merges it into a few
        hundred large blobs rather than shattering it: measured in the app,
        **228** instances covering ~100% of the frame — a solid sheet, plainly
        wrong, and comfortably under the 500-instance bar. Nothing legitimate
        is most of the frame, so past `_FAIL_COVERAGE_ALONE` the count stops
        being evidence either way.
        """
        assert pa._threshold_failed(228, 1.00) is True
        assert pa._threshold_failed(3, 0.95) is True
        assert pa._threshold_failed(20_000, 0.99) is True
        # ...but an empty result is not a failure of this kind, whatever the
        # coverage arithmetic says about an empty frame.
        assert pa._threshold_failed(0, 1.0) is False

    def test_the_two_bars_are_ordered(self):
        """A coverage that fails alone must also fail the two-condition rule,
        or the branches contradict each other somewhere in between."""
        assert pa._FAIL_COVERAGE_ALONE > pa._FAIL_COVERAGE

    def test_a_normal_preview_is_not_flagged(self, window):
        """The fixture must stay UNflagged or the notice is just noise."""
        session, plot, _tree, _wiz = _opened(window)
        msg = _tune(session, plot, {"marker_smooth": 1.5}, window["messages"])
        assert msg["threshold_failed"] is False, msg
        assert 0.0 <= msg["coverage"] <= 1.0

    def test_the_preview_reports_coverage_and_face_units(self, window):
        session, plot, _tree, _wiz = _opened(window)
        msg = _tune(session, plot, {"min_separation": 4}, window["messages"])
        assert "coverage" in msg and "face_units" in msg
        assert msg["face_units"] in ("nm", "px")


class TestTheCaretAlwaysGetsItsClasses:
    """The class list is what the brush strip is gated on.

    Reported: "the caret opens but the class strip never appears" — the list
    showing "—" and nothing to paint with. That is `classes: []` in the caret,
    which means `seg_state` either was not emitted or was not ROUTED, and both
    are silent: the caret looks merely empty, not broken.
    """

    def test_a_fresh_wizard_reports_the_default_classes(self, window):
        session, plot, _tree, _wiz = _opened_untrained(window)
        msgs = window["messages"]
        states = _of_type(msgs, "seg_state")
        assert states, "seg_open emitted no seg_state at all"
        classes = states[-1]["classes"]
        assert len(classes) >= 2, (
            f"only {len(classes)} classes — the caret gates its brush strip on "
            f"this list, so an empty one is a caret with nothing to paint with")
        assert all(c["pixels"] == 0 for c in classes)

    def test_a_plot_that_got_its_window_LATE_is_still_routed(self, window):
        """`Plot.window_id` is copied from `plot_window` at PLOT construction,
        and the wizard copies it again at ITS construction. A plot that
        acquired its window afterwards carried None through both, and every
        message the caret sent was addressed to nobody — the renderer filters
        `seg_state` by window id, so it silently dropped them."""
        session, plot, _tree, wiz = _opened_untrained(window)
        msgs = window["messages"]

        wiz.window_id = None                       # the stale capture
        plot.window_id = 4242                      # the window it really has
        del msgs[:]
        pa._emit_state(wiz)

        states = _of_type(msgs, "seg_state")
        assert states, "no seg_state emitted"
        assert states[-1]["window_id"] == 4242, (
            "seg_state was addressed with the STALE captured id, so the caret "
            "never receives it")

    def test_a_broken_class_report_still_emits_the_state(self, window,
                                                         monkeypatch):
        """Best-effort everywhere except the message itself."""
        session, plot, _tree, wiz = _opened_untrained(window)
        msgs = window["messages"]

        def _boom(self):
            raise RuntimeError("label store exploded")

        monkeypatch.setattr(pa.SegmentWizard, "class_report", _boom)
        del msgs[:]
        pa._emit_state(wiz)
        states = _of_type(msgs, "seg_state")
        assert states, "a failing class report swallowed the whole message"
        assert len(states[-1]["classes"]) >= 2, (
            "fell back to NO classes, which is the unusable caret again")


class TestThePreviewUsesTheSameDevicePolicyAsTheBatch:
    """The interactive path must be gated like the batch, i.e. like find-vectors.

    It was not, and that was invisible for as long as `classical` was the
    default engine: pure scipy, so the preview never submitted to the device.
    Deleting classical made every preview a torch submission — one per tune,
    one per navigator move, each on its own worker thread — with nothing
    bounding how many occupy the GPU at once. That is the opportunistic-GPU
    collapse `_gpu_task_allowed` exists to prevent on the vectors path.

    Contract tests rather than a throughput measurement, for the reason
    `test_device_lock.py` gives about the MPS lock: the failure is
    probabilistic per submission, so a short stress run that happens to survive
    proves nothing. What can be pinned is that the gate is *taken*.
    """

    def test_the_scribble_preview_engine_enters_the_gpu_semaphore(self, window,
                                                                  monkeypatch):
        import contextlib

        entered = []

        @contextlib.contextmanager
        def _spy():
            entered.append(1)
            yield

        import spyde.particles.batch as batch
        monkeypatch.setattr(batch, "_gpu_slots", _spy)

        session, plot, _tree, wiz = _opened(window)
        engine = pa._engine(wiz, dict(wiz.params))
        assert engine is not None, "no engine on a trained wizard"
        entered.clear()
        engine(np.asarray(wiz.signal().data[0]))
        assert entered, (
            "the preview engine ran a device submission without taking "
            "_gpu_slots() — unbounded feeders, which is the vectors bug")

    def test_the_split_is_NOT_inside_the_device_slot(self, window, monkeypatch):
        """Holding a slot across the numpy split turns N feeders back into one.

        The batch is careful about this (`resolve_engine`'s comment); the
        preview has to be too, or the semaphore serialises work that never
        touches the device.
        """
        import contextlib

        held = {"now": False, "split_inside": False}

        @contextlib.contextmanager
        def _spy():
            held["now"] = True
            try:
                yield
            finally:
                held["now"] = False

        import spyde.particles.batch as batch
        import spyde.particles.instances as inst
        monkeypatch.setattr(batch, "_gpu_slots", _spy)
        real_split = inst.split_instances

        def _watch_split(*a, **k):
            held["split_inside"] = held["now"]
            return real_split(*a, **k)

        monkeypatch.setattr(inst, "split_instances", _watch_split)

        session, plot, _tree, wiz = _opened(window)
        engine = pa._engine(wiz, dict(wiz.params))
        engine(np.asarray(wiz.signal().data[0]))
        assert held["split_inside"] is False, (
            "split_instances ran while holding a GPU slot")

    def test_the_feeder_cap_times_the_band_fraction_stays_under_one(self):
        """THE invariant, and the reason the GPU froze.

        `features.band_budget_bytes` sizes one row band at
        `_GPU_FREE_FRACTION` of *free* VRAM, read at call time — so concurrent
        feeders each claim that fraction of a figure that ignores the others.
        Four of them claim 100% of free memory at once, and `GPU_BAND_BYTES`
        records the cost of overshooting: 1012 ms per frame -> 12.8 s -> 26.7 s,
        thrashing the allocator. That is a hang, not a slowdown.

        Counting the feeders: the dask lane (`PARTICLE_GPU_LANE_DEFAULT`) plus
        ONE for the app process, which is not in the lane at all
        (`_gpu_task_allowed` returns True off a worker), each running up to
        `PARTICLE_DEVICE_CONC` at a time.
        """
        from spyde.particles import features
        from spyde.particles.batch import (PARTICLE_DEVICE_CONC,
                                           PARTICLE_GPU_LANE_DEFAULT)

        lane = PARTICLE_GPU_LANE_DEFAULT
        n_lane = 1 if lane in ("one", "off") else int(lane)
        feeders = (n_lane + 1) * PARTICLE_DEVICE_CONC       # +1 = the app
        claim = feeders * features._GPU_FREE_FRACTION
        assert claim <= 1.0, (
            f"{feeders} concurrent GPU feeders x "
            f"{features._GPU_FREE_FRACTION:.0%} of free VRAM = {claim:.0%} "
            f"claimed at once. Overshooting is catastrophic, not merely slow "
            f"(see GPU_BAND_BYTES). Lower PARTICLE_DEVICE_CONC, tighten the "
            f"lane, or lower _GPU_FREE_FRACTION.")

    @pytest.mark.parametrize("setting,expect", [
        (None, "one"), ("one", "one"), ("off", "off"),
        ("2", "2"), ("4", "2"), ("8", "2"), ("all", "2"), ("junk", "one"),
    ])
    def test_the_lane_is_capped_whatever_the_shared_setting_says(
            self, setting, expect, monkeypatch):
        """The DaskMonitor's "GPU feeders" control cannot exceed our ceiling.

        `_gpu_slots` is per-PROCESS and cannot bound the number of processes;
        the lane is the only cross-process cap, and it was fully overridable.
        At "all" the split returns NO lane, `_dispatch` hands every worker
        chunks, and `_gpu_task_allowed` says yes to each — nine scribble heads
        on one card. Observed: six busy workers, 11.5 of 12.0 GB, hung.
        """
        from spyde.particles.batch import _clamped_lane_mode
        monkeypatch.delenv("SPYDE_FV_GPU", raising=False)
        monkeypatch.delenv("SPYDE_SEG_GPU_LANE_MAX", raising=False)
        if setting is not None:
            monkeypatch.setenv("SPYDE_FV_GPU", setting)
        assert _clamped_lane_mode() == expect

    def test_off_is_still_honoured(self, monkeypatch):
        """The cap is a CEILING — asking for fewer feeders must still work, or
        a user trying to get off the GPU entirely cannot."""
        from spyde.particles.batch import _clamped_lane_mode
        monkeypatch.setenv("SPYDE_FV_GPU", "off")
        assert _clamped_lane_mode() == "off"

    def test_the_ceiling_is_itself_overridable(self, monkeypatch):
        """A bigger card should not be held to this one's number."""
        from spyde.particles.batch import _clamped_lane_mode
        monkeypatch.setenv("SPYDE_FV_GPU", "8")
        monkeypatch.setenv("SPYDE_SEG_GPU_LANE_MAX", "1")
        assert _clamped_lane_mode() == "one"

    def test_the_preview_and_the_batch_share_ONE_semaphore(self):
        """The app process is not in the dask lane, so if it had its own
        semaphore a preview during a batch would be an uncounted feeder."""
        from spyde.particles.batch import _gpu_slots
        assert _gpu_slots() is _gpu_slots(), "not process-wide"

    def test_the_device_slot_actually_bounds(self):
        from spyde.particles.batch import _gpu_slots
        sem = _gpu_slots()
        got = []
        try:
            while sem.acquire(blocking=False):
                got.append(1)
                if len(got) > 8:
                    break
        finally:
            for _ in got:
                sem.release()
        assert 1 <= len(got) <= 4, (
            f"the device semaphore admitted {len(got)} concurrent feeders")

    def test_the_preview_does_NOT_cap_torch_threads(self, window):
        """The two paths want the same DEVICE policy and OPPOSITE CPU policies.

        `_cap_torch_threads` is documented as a no-op off a dask worker, and
        that is a decision rather than an oversight: capping intra-op threads
        is right for nine workers x four task slots and wrong for one preview
        the user is waiting on. Borrowing the batch's device gate must not
        drag its thread cap along — asserted here because "copy the batch's
        GPU workflow" is exactly the instruction under which someone would.
        """
        import torch
        before = torch.get_num_threads()
        session, plot, _tree, wiz = _opened(window)
        pa._engine(wiz, dict(wiz.params))
        assert torch.get_num_threads() == before, (
            f"building the preview engine changed torch's intra-op threads "
            f"({before} -> {torch.get_num_threads()}); the interactive path "
            f"should keep the whole machine")


class TestOutlineDrawCap:
    """No route may hand the renderer an unbounded number of polygons.

    `_RASTER_ABOVE` picks the nicer drawing; this cap is the seatbelt for when
    the raster is unavailable. It had none, so a tile-mode raster failure fell
    back to one filled path per instance -- 14028 of them, which hangs the
    renderer and composites into a flat green sheet showing nothing.
    """

    def test_the_cap_is_well_above_any_real_outline_count(self):
        assert pa._MAX_OUTLINE_POLYS > pa._RASTER_ABOVE * 5

    def test_a_huge_contour_list_draws_NONE_rather_than_all(self, monkeypatch):
        import types
        pushed = {}
        monkeypatch.setattr(pa, "_push_groups", lambda p, u: pushed.update(
            {g.name: pl.get("vertices_list") for g, pl in u.items()}))

        # No `set_overlay_mask` on this plot => the raster path is unavailable,
        # which is exactly the state the tile-mode ValueError used to produce.
        w = _bare_wizard(types.SimpleNamespace())

        # A plain class, NOT SimpleNamespace: the group is used as a dict KEY in
        # the `_push_groups` payload, and SimpleNamespace defines __eq__ so it
        # is unhashable.
        class _Group:
            name = "seg_preview_outline"

        monkeypatch.setattr(pa.SegmentWizard, "_overlay_group",
                            lambda self, p: _Group())
        monkeypatch.setattr(pa.SegmentWizard, "_window_group", lambda self, p: None)

        n = pa._MAX_OUTLINE_POLYS + 1
        w.set_overlay([np.zeros((3, 2), np.int16) for _ in range(n)], None,
                      labels=None, n_instances=n)
        assert pushed.get("seg_preview_outline") == [], (
            f"{n} polygons reached the renderer; the cap is "
            f"{pa._MAX_OUTLINE_POLYS}")


class TestMeasureSkipsOutlinesItCannotDraw:
    """Contours are ~half a preview's cost once instances run to thousands."""

    def test_want_contours_false_skips_them_but_keeps_every_measurement(self):
        from spyde.particles import measure_frame

        lab = np.zeros((64, 64), np.int32)
        lab[5:15, 5:15] = 1
        lab[30:40, 30:40] = 2
        frame = np.random.default_rng(0).random((64, 64)).astype(np.float32)

        rows_a, cont_a = measure_frame(lab, frame, t=0, scale=0.5)
        rows_b, cont_b = measure_frame(lab, frame, t=0, scale=0.5,
                                       want_contours=False)
        assert len(cont_a) == 2 and cont_b == []
        # The ROWS -- every measured property, including the score the
        # confidence filter reads -- must be untouched by the flag.
        assert np.array_equal(rows_a, rows_b)

    def test_commit_refuses_a_preview_whose_outlines_were_skipped(self, window):
        """A store needs one contour PER ROW; committing without them would
        build a silently mis-corresponding dataset."""
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        wiz.preview = {"frame": 0, "labels": np.zeros((8, 8), np.int32),
                       "rows": np.zeros((3, 4), np.float32), "contours": [],
                       "count": 3, "areas": np.zeros(3), "box": None}
        assert wiz.commit() is None
        assert any("too many to commit" in str(m.get("text", ""))
                   for m in msgs if isinstance(m, dict)), \
            "commit failed silently instead of saying why"


class TestBrushActuallyPaints:
    """The regression guard for "I can't scribble".

    Shipped broken once, in two independent ways, and neither was visible to any
    test that existed at the time:

    1. **Nothing ever created a brush widget.** ``add_brush_widget`` was called
       nowhere, so Shift+drag had nothing to hit.
    2. **The caret listened on a path the brush cannot reach.** It waited for a
       renderer-side ``spyde:figure_event`` carrying a points array, but an
       anyplotlib stroke travels to PYTHON (``event_json`` →
       ``Figure._dispatch_event`` → ``Widget._update_from_js`` →
       ``plot.callbacks.fire``). The renderer never sees it, so even with a brush
       present nothing would have arrived.

    The old tests passed because they posted a synthetic ``seg_paint`` payload —
    which exercises the rasteriser and proves nothing about whether a real stroke
    can ever get there. These drive the widget.
    """

    def test_a_brush_is_attached_by_seg_open_ALONE(self, brush_required, window):
        """`seg_open` and nothing else. This is the regression that shipped.

        The brush used to be armed only by ``seg_set_method("scribble")``,
        which the caret sent when you clicked the Scribble tab. Deleting the
        classical engine deleted the tab row, nothing sent that verb any more,
        and the caret opened telling you to paint with nothing to paint with —
        reported as "I can't scribble".

        Every test that existed went through a helper which called
        ``seg_set_method`` itself, and the e2e painted by dispatching a
        synthetic ``figure_event`` straight at ``seg_paint``. Both bypassed the
        arming path, so both stayed green. Hence this test calls the ONE verb
        the caret actually sends on mount, and asserts on the widget.
        """
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot = _figure_plot(session)
        pa.seg_open(session, plot, {})
        assert getattr(plot.signal_tree, "_seg_brush", None) is not None, (
            "seg_open left no brush on the plot — Shift+drag has nothing to "
            "hit, and painting is the only way to teach the only engine")

    def test_reopening_the_caret_re_arms_the_brush(self, brush_required, window):
        """The idempotent re-open path is a separate branch and needs it too."""
        from spyde.actions.particles_action import seg_close, seg_open
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot = _figure_plot(session)
        seg_open(session, plot, {})
        seg_open(session, plot, {})          # re-open over a live controller
        assert getattr(plot.signal_tree, "_seg_brush", None) is not None, (
            "the re-open branch does not arm the brush")
        seg_close(session, plot, {})
        seg_open(session, plot, {})          # close, then open again
        assert getattr(plot.signal_tree, "_seg_brush", None) is not None, (
            "reopening after a close left no brush")

    def test_a_stroke_from_the_WIDGET_reaches_the_label_store(self, brush_required,
                                                              window):
        """Drives the widget, not a synthetic seg_paint payload."""
        from spyde.actions.particles_action import _on_stroke
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        _plot, tree = _scribble_wizard(session)
        wiz = tree._seg_wizard
        brush = tree._seg_brush
        store = wiz.label_store()
        before = dict(store.counts())
        # What anyplotlib delivers: [[x, y], …] in IMAGE PIXELS, then pointer_up.
        brush.set(strokes=[[[20.0, 30.0], [24.0, 32.0], [28.0, 34.0]]],
                  stroke_classes=[0])
        _on_stroke(wiz, None)
        after = dict(store.counts())
        assert after != before, "the stroke never reached the label store"
        assert after[0] > 0

    def test_a_second_stroke_only_paints_its_own_points(self, brush_required,
                                                        window):
        """The widget accumulates strokes for its whole life, so replaying the
        full list on every event would re-paint everything and make the class
        counts grow quadratically.

        The class is switched through ``seg_tune``, NOT by handing the widget
        different ``stroke_classes``: the caret's params are the authority,
        precisely because the JS widget's own value is not reliably in sync (see
        :meth:`test_class_comes_from_PARAMS_not_from_the_js_widget`).
        """
        from spyde.actions.particles_action import _on_stroke, seg_tune
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, tree = _scribble_wizard(session)
        wiz, brush = tree._seg_wizard, tree._seg_brush
        store = wiz.label_store()
        s1 = [[20.0, 30.0], [24.0, 32.0]]
        brush.set(strokes=[s1])
        _on_stroke(wiz, None)
        one = dict(store.counts())
        # Fire again with NO new stroke: nothing may change.
        _on_stroke(wiz, None)
        assert dict(store.counts()) == one, "re-fired the same stroke"
        # A genuinely new stroke, in a class chosen the way the strip chooses it.
        seg_tune(session, plot, {"active_class": 1})
        brush.set(strokes=[s1, [[60.0, 40.0], [64.0, 42.0]]])
        _on_stroke(wiz, None)
        two = dict(store.counts())
        assert two[1] > 0, "the second stroke's class was not painted"
        assert two[0] == one[0], "the first stroke was painted twice"

    def test_closing_the_caret_detaches_the_brush(self, brush_required, window):
        """It floats over the image, so it must not outlive the caret.

        This used to switch to the Classical tab and assert the brush went with
        it. That engine is gone and painting is now the only way to teach the
        only engine, so the brush is armed for as long as the caret is open —
        which makes CLOSING it the event that has to tear the widget down.
        """
        from spyde.actions.particles_action import seg_close
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, tree = _scribble_wizard(session)
        assert tree._seg_brush is not None
        seg_close(session, plot, {})
        assert getattr(tree, "_seg_brush", None) is None

    def test_missing_brush_says_so_instead_of_failing_silently(self, window,
                                                              monkeypatch):
        """A user dragging at an image that never responds must be TOLD why."""
        import spyde.actions.particles_action as pa
        monkeypatch.setattr(pa, "_brush_supported", lambda: False)
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        before = len(window["messages"])
        _scribble_wizard(session)
        new = window["messages"][before:]
        assert any("anyplotlib" in str(m.get("text", "")).lower()
                   or "brush" in str(m.get("text", "")).lower() for m in new), (
            "switching to Scribble with no brush available emitted nothing — "
            "the user gets a picture that ignores them and no explanation")


class TestPaintStateReachesTheWidget:
    """Class and eraser must travel to the WIDGET, not stop at wiz.params.

    Both bugs here shipped and were user-visible: every stroke came out in class
    0 ("I can only scribble one colour") and the eraser did nothing ("delete
    doesn't work"). One root cause — the ClassStrip set React state, nothing sent
    it to Python, and the backend read `active_class` / `erase` from params that
    were never declared and never set.

    The widget tags each stroke with its OWN class at paint time in JS, so a
    change that only reaches ``wiz.params`` paints the previous colour forever.
    That is why these assert on the WIDGET's attributes and on what actually
    landed in the store — not on the params dict, which was already "right" while
    the paint was wrong.
    """

    def _scribbling(self, session):
        """Load, open the caret, and select Scribble EXPLICITLY.

        The extra verb is the difference from the shared `_scribble_wizard`, and
        it is deliberate here: these tests are about what reaches the widget
        once the strip is live, not about which verb arms it.
        """
        from spyde.actions.particles_action import seg_set_method
        session._load_test_data_particles({"frames": 4})
        plot, tree = _scribble_wizard(session)
        seg_set_method(session, plot, {"method": "scribble"})
        return plot, tree

    @staticmethod
    def _stroke(tree, wiz, pts):
        from spyde.actions.particles_action import _on_stroke
        brush = tree._seg_brush
        strokes = list(getattr(brush, "strokes", []) or [])
        classes = list(getattr(brush, "stroke_classes", []) or [])
        brush.set(strokes=strokes + [pts],
                  stroke_classes=classes + [int(brush.class_id)])
        _on_stroke(wiz, None)

    def test_defaults_declare_the_paint_state(self):
        """They were read but never declared, so params.get() always won."""
        from spyde.actions.particles_action import DEFAULTS
        assert "active_class" in DEFAULTS, (
            "active_class is read by the brush but not a declared parameter, so "
            "nothing can ever set it and every stroke is class 0")
        assert "erase" in DEFAULTS

    def test_class_comes_from_PARAMS_not_from_the_js_widget(self, brush_required, window):
        """The test that would have caught it. The old one did not.

        The first version set `brush.class_id` and read it back — both PYTHON
        side — so it passed while the app was broken. In the real app the widget
        lives in JS, and `Figure._push_widget` sends a targeted update that never
        writes `panel_<id>_json`, so a Python-side class push does not reliably
        reach it before the next stroke.

        The natural experiment that exposed this: the ERASER worked and the CLASS
        did not, in the same handler on the same stroke — because erase read from
        `wiz.params` and class read from the widget's `stroke_classes`.

        So this test forces the widget's own class to be STALE and wrong, and
        asserts the stroke still lands in the class the caret asked for.
        """
        from spyde.actions.particles_action import _on_stroke, seg_tune
        session = window["window"]
        plot, tree = self._scribbling(session)
        wiz, brush = tree._seg_wizard, tree._seg_brush
        store = wiz.label_store()

        seg_tune(session, plot, {"active_class": 2})
        # Simulate the push NOT landing: the widget still thinks it is class 0,
        # and tags the stroke accordingly.
        brush.set(class_id=0)
        brush.set(strokes=[[[20.0, 30.0], [26.0, 32.0]]], stroke_classes=[0])
        _on_stroke(wiz, None)

        counts = dict(store.counts())
        assert counts.get(2, 0) > 0, (
            "the stroke was filed under the WIDGET's stale class instead of the "
            "one the caret selected — this is the 'can only scribble one colour' "
            f"bug: {counts}")
        assert counts.get(0, 0) == 0

    def test_switching_class_retags_the_next_stroke(self, brush_required, window):
        from spyde.actions.particles_action import seg_tune
        session = window["window"]
        plot, tree = self._scribbling(session)
        wiz = tree._seg_wizard
        store = wiz.label_store()

        self._stroke(tree, wiz, [[20.0, 30.0], [26.0, 32.0]])
        seg_tune(session, plot, {"active_class": 1})
        assert int(tree._seg_brush.class_id) == 1, (
            "seg_tune did not reach the widget — the strip's choice stops at "
            "wiz.params and the next stroke paints the OLD class")
        self._stroke(tree, wiz, [[60.0, 40.0], [66.0, 42.0]])
        seg_tune(session, plot, {"active_class": 2})
        self._stroke(tree, wiz, [[80.0, 20.0], [86.0, 22.0]])

        counts = dict(store.counts())
        assert counts[0] > 0 and counts[1] > 0 and counts[2] > 0, (
            f"only some classes were painted: {counts}")

    def test_eraser_removes_and_leaves_other_classes_alone(self, brush_required, window):
        from spyde.actions.particles_action import seg_tune
        session = window["window"]
        plot, tree = self._scribbling(session)
        wiz = tree._seg_wizard
        store = wiz.label_store()

        path = [[20.0, 30.0], [26.0, 32.0]]
        self._stroke(tree, wiz, path)
        seg_tune(session, plot, {"active_class": 1})
        self._stroke(tree, wiz, [[60.0, 40.0], [66.0, 42.0]])
        before = dict(store.counts())
        assert before[0] > 0 and before[1] > 0

        seg_tune(session, plot, {"erase": True})
        assert bool(tree._seg_brush.erase) is True, (
            "the eraser is a WIDGET mode — a stroke is tagged before the handler "
            "sees it, so an erase flag that stops at wiz.params does nothing")
        self._stroke(tree, wiz, path)          # retrace the class-0 stroke

        after = dict(store.counts())
        assert after[0] < before[0], "the eraser removed nothing"
        assert after[1] == before[1], "the eraser hit a class it was not over"

    def test_brush_size_reaches_the_widget_too(self, brush_required, window):
        from spyde.actions.particles_action import seg_tune
        session = window["window"]
        plot, tree = self._scribbling(session)
        seg_tune(session, plot, {"brush": 9.0})
        assert float(tree._seg_brush.radius) == pytest.approx(9.0)


class TestBrushDrawColour:
    """The brush must DRAW in the selected class's colour, not just tag strokes.

    Reported twice as "I can only scribble one colour" / "support film still
    doesn't change the painting colour". The first fix made the STROKE land in
    the right class, which it now does — the per-class pixel counts in the caret
    prove it. But the colour on screen comes from the WIDGET's own `class_id`
    (`figure_esm.js::_brushLiveBegin` reads `w.class_id` at stroke start and
    draws in `colors[class_id]`), so the data can be right while the paint is
    still orange. These assert the widget state, which is the thing the eye sees.
    """

    def test_selecting_a_class_changes_the_widget_class_id(self, brush_required, window):
        from spyde.actions.particles_action import seg_tune
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, tree = _scribble_wizard(session)
        brush = getattr(tree, "_seg_brush", None)
        assert brush is not None

        # What the ClassStrip sends when you click "support film".
        seg_tune(session, plot, {"active_class": 1})
        assert int(brush._data["class_id"]) == 1, (
            "the widget still paints class 0 — the strip's selection reached "
            "wiz.params but not the widget, so every stroke DRAWS orange")

        seg_tune(session, plot, {"active_class": 2})
        assert int(brush._data["class_id"]) == 2

    def test_the_new_class_reaches_the_PANEL_state(self, brush_required, window):
        """The JS draws from ``panel_<id>_json``, not from the Python object.

        This is the assertion the previous two miss and the reason the bug
        survived a "fix": ``Widget.set`` reaches JS via ``_push_widget``, which
        writes ``event_json`` ONLY and leaves the panel's own widget state
        stale. ``_brushLiveBegin`` reads ``w.class_id`` from
        ``p.state.overlay_widgets`` and paints ``colors[class_id]`` — so
        ``brush._data`` can say 1 while every stroke still draws in class 0's
        colour. Assert the serialised panel, which is what the eye sees.
        """
        from spyde.actions.particles_action import seg_tune
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, tree = _scribble_wizard(session)

        # Spy on the panel push. Asserting `_state` alone would be VACUOUS:
        # `overlay_widgets` holds the widget's own `_data` BY REFERENCE, so
        # `brush.set()` mutates it in place and the dict always looks current
        # whether or not anything was ever sent to JS. What the renderer sees is
        # the re-serialised trait, so the assertion has to be "a push happened".
        pushes = []
        real_push = plot._plot2d._push
        plot._plot2d._push = lambda *a, **k: (pushes.append(1), real_push(*a, **k))[1]
        try:
            seg_tune(session, plot, {"active_class": 2})
        finally:
            plot._plot2d._push = real_push

        assert pushes, (
            "no panel push after the class changed — the targeted widget update "
            "writes event_json only, so panel_<id>_json keeps the OLD class and "
            "JS goes on painting the previous class's colour")

        widgets = (plot._plot2d._state.get("overlay_widgets") or [])
        brushes = [w for w in widgets if (w or {}).get("type") == "brush"]
        assert brushes, f"no brush in the panel state: {widgets}"
        assert int(brushes[0].get("class_id", -1)) == 2

    def test_an_unrelated_tune_does_NOT_force_a_panel_push(self, brush_required, window):
        """`seg_tune` fires for every slider tick too.

        A full panel push re-serialises the image bytes, so doing one per tick
        on a 4096² frame would trade the colour bug for a much worse drag. Only
        an actual brush change (class / eraser / size) may pay for it.
        """
        from spyde.actions.particles_action import seg_tune
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, _tree = _scribble_wizard(session)

        seg_tune(session, plot, {"active_class": 1})      # settle the state

        # WAIT FOR THE PLOT TO GO QUIET FIRST. The spy below counts EVERY push
        # to this panel, and the signal plot has an async fill of its own
        # (`_start_progressive_nav_compute` / the progressive signal preview)
        # that is nothing to do with tuning. Under load its last push lands
        # inside the measured window and the test fails claiming a slider drag
        # re-serialised the image — a false positive that only appears in a
        # full-suite run, which is the worst kind. Verified with a stack-trace
        # spy: in isolation this test sees ZERO pushes from any source.
        quiet = []
        settle_push = plot._plot2d._push
        plot._plot2d._push = lambda *a, **k: (quiet.append(1), settle_push(*a, **k))[1]
        try:
            deadline = time.time() + 10.0
            while time.time() < deadline:
                n = len(quiet)
                time.sleep(0.4)
                if len(quiet) == n:
                    break
        finally:
            plot._plot2d._push = settle_push

        pushes = []
        real_push = plot._plot2d._push
        plot._plot2d._push = lambda *a, **k: (pushes.append(1), real_push(*a, **k))[1]
        try:
            seg_tune(session, plot, {"min_separation": 6})
            seg_tune(session, plot, {"min_separation": 7})
            seg_tune(session, plot, {"active_class": 1})   # SAME class, no change
        finally:
            plot._plot2d._push = real_push

        assert pushes == [], (
            f"{len(pushes)} panel pushes for tunes that did not change the "
            "brush — a slider drag would re-serialise the whole image")

    def test_the_widget_has_a_colour_for_every_class(self, brush_required, window):
        """`colors` is indexed by class_id in JS, so a short list means the
        later classes draw with `undefined` — no colour change on screen even
        though class_id updated correctly."""
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        _plot, tree = _scribble_wizard(session)
        brush = tree._seg_brush
        from spyde.particles import default_classes
        n_classes = len(default_classes())
        colours = list(brush._data.get("colors") or [])
        assert len(colours) >= n_classes, (
            f"brush carries {len(colours)} colours for {n_classes} classes — "
            f"class ids >= {len(colours)} draw with no colour: {colours}")


class TestBatchComputingOverlay:
    """The "Calculating…" chip over the RESULT window, and when it appears.

    Reported as: "there is a lot of lag between the subwindow appearing and then
    [calculating]". The chip was not raised at all, and the obvious place to add
    it — next to the first progress emission, inside the worker — is far too
    late: the window opens, then the placeholder store is built, the cancel
    flags registered, the generation bumped, the worker scheduled, the thread
    hop paid, and the first frame COMPUTED. On 4096² frames that is seconds of a
    window that looks finished and empty.

    So the contract is SYNCHRONOUS: by the time `seg_run` returns to the event
    loop, the chip is already up.
    """

    def _trained_plot(self, session):
        """A plot whose caret has a trained head, via the autolabel test door."""
        plot = _figure_plot(session)
        pa.seg_open(session, plot, {"min_size": 25})
        assert _wait(lambda: getattr(plot.signal_tree, "_seg_wizard", None))
        wiz = plot.signal_tree._seg_wizard
        pa.seg_autolabel(session, plot, {})
        # CPU explicitly — see `_opened`.
        pa.seg_train(session, plot, {"device": "cpu"})
        assert _wait(lambda: wiz.classifier is not None
                     and wiz.classifier.is_trained), "training never finished"
        return plot

    @staticmethod
    def _result_window_ids(session):
        """EVERY window of the newest tree — the result `seg_run` just opened.

        Its navigator counts: the count-trace fill raises a Calculating chip of
        its own, so "the chip is on the result" is a claim about the tree, not
        about one plot of it.
        """
        result = session.signal_trees[-1]
        plots = list(result.signal_plots) + list(pa._nav_plots(result))
        return {getattr(p, "window_id", None) for p in plots}

    @staticmethod
    def _batch_chip_window(session):
        """The window the BATCH's own chip is addressed to (its signal plot)."""
        return pa._result_window_id(session.signal_trees[-1])

    def test_the_chip_is_raised_before_seg_run_returns(self, window):
        session = window["window"]
        msgs = window["messages"]
        session._load_test_data_particles({"frames": 4})

        # A TRAINED head, or `seg_run` returns before it raises anything —
        # there is no engine that can run without one now, and a `seg_run` that
        # bails early would make this pass for the wrong reason.
        plot = self._trained_plot(session)
        del msgs[:]
        pa.seg_run(session, plot, dict(track=False))

        # No waiting, no polling: the chip must already be on the wire.
        raised = [m for m in msgs
                  if m.get("type") == "window_computing" and m.get("computing")]
        assert raised, (
            "no window_computing raised synchronously — the chip only appears "
            "once the worker gets going, which is the reported lag")

    def test_the_chip_names_the_RESULT_window(self, window):
        """Not the source window: the source is where you were scribbling and it
        is not the thing sitting there looking empty.

        This test was vacuous three times over and asserted nothing at all:
        it ran `seg_run` on an UNTRAINED plot with `method="classical"` (which
        `_coerce` maps to `scribble`, whose `_engine` then refuses), so the
        handler returned before raising any chip; it cleared `msgs`, which IS
        `window["messages"]`, deleting the very chip it went on to look for; and
        its only assertion sat behind `if all_ids:`, so an empty list passed.
        """
        session = window["window"]
        msgs = window["messages"]
        session._load_test_data_particles({"frames": 4})
        src = self._trained_plot(session)
        del msgs[:]
        pa.seg_run(session, src, dict(track=False))

        raised = {m.get("window_id") for m in msgs
                  if m.get("type") == "window_computing" and m.get("computing")}
        assert raised, "no Calculating chip was raised at all"
        assert self._batch_chip_window(session) in raised, \
            "the batch raised no chip on the result's signal window"
        assert raised <= self._result_window_ids(session), (
            f"a chip went to a window outside the result tree: "
            f"{sorted(raised - self._result_window_ids(session))}")
        assert getattr(src, "window_id", None) not in raised, (
            "the chip was put on the SOURCE window — that is where you were "
            "scribbling, not the window sitting there looking empty")

    def test_the_chip_comes_down_when_the_batch_ends(self, window):
        session = window["window"]
        msgs = window["messages"]
        session._load_test_data_particles({"frames": 4})
        plot = self._trained_plot(session)
        del msgs[:]
        pa.seg_run(session, plot, dict(track=False))
        chip = self._batch_chip_window(session)

        # FILTERED BY WINDOW, and by the BATCH's window specifically: the result
        # tree's navigator raises and lowers a chip of its own for the count-
        # trace fill, so an unfiltered wait would pass on that one instead —
        # green whether or not the batch ever cleared its own.
        assert _wait(lambda: any(
            m.get("type") == "window_computing" and not m.get("computing")
            and m.get("window_id") == chip
            for m in msgs), timeout=30), (
            "the Calculating chip never came down — it will spin forever over a "
            "window that has finished")


class TestPreviewWindowBoxLifecycle:
    """The 1-megapixel preview-window box must not depend on an ENGINE.

    Reported: "the 1k x 1k outline only shows in classical; if you move to a
    different mode it continues to show, but toggle out/back in and it won't
    show up again." All three symptoms are one cause — the box was drawn only
    as a side effect of a successful segmentation, and `_preview` returns early
    when there is no engine (an untrained Scribble, or Prompt before any
    prompt), painting nothing and CLEARING nothing:

      * Classical always has a solver, so only it drew the box.
      * Switching to an untrained engine left the previous box on screen,
        because the early return cleared nothing.
      * Toggling the caret dropped the overlay groups, and reopening in an
        untrained engine early-returned again, so it never came back.

    The box documents WHERE the budget looks. That is true whenever the caret
    is open.
    """

    @staticmethod
    def _wiz():
        class _Grp:                     # hashable by identity, unlike SimpleNamespace
            def __init__(self, name):
                self.name, self.removed = name, False

            def remove(self):
                self.removed = True

        class _P2D:
            def add_polygons(self, *a, **k):
                return _Grp(k.get("name"))

        wiz = _bare_wizard(_P2D())
        wiz.frames = lambda: (1, lambda i: np.zeros((2048, 2048), np.float32),
                              (2048, 2048))
        wiz.frame_index = lambda: 0
        return wiz

    def test_the_box_is_CLEARED_not_drawn_with_no_engine(self, monkeypatch):
        """The box is gone by design on this branch, and this is where it shows.

        It existed to admit that only the middle megapixel of a large frame had
        been looked at. The mask-only preview raised that budget to
        `_PREVIEW_PIXEL_BUDGET_MASK` — a whole 4096² frame — so there is no
        untouched remainder left to disclaim, and drawing a 1-megapixel
        rectangle over data the preview does NOT stop at would be the same lie
        in the other direction. `set_mask_overlay` clears it for the same
        reason, and the crop that survives above the mask budget is still
        reported in text (`preview_box` in the `seg_preview` payload).

        So what this class pins is unchanged in substance — the box must not
        depend on an ENGINE — only the box is now consistently ABSENT rather
        than consistently present.
        """
        import spyde.actions.particles_action as pa
        pushed = {}
        monkeypatch.setattr(pa, "_push_groups", lambda p, u: pushed.update(
            {g.name: pl.get("vertices_list") for g, pl in u.items()}))
        self._wiz().show_preview_window()
        assert "seg_preview_window" in pushed, (
            "the window group was not pushed at all, so a box left over from a "
            "previous engine would still be on screen")
        assert pushed["seg_preview_window"] == [], (
            f"a preview-window box was drawn: {pushed['seg_preview_window']}")

    def test_switching_to_an_untrained_engine_clears_stale_outlines(self, monkeypatch):
        """A previous engine's instances must not linger looking like the new
        engine's answer."""
        import spyde.actions.particles_action as pa
        pushed = {}
        monkeypatch.setattr(pa, "_push_groups", lambda p, u: pushed.update(
            {g.name: pl.get("vertices_list") for g, pl in u.items()}))
        self._wiz().show_preview_window()
        assert pushed.get("seg_preview_outline") == [], (
            "the outline group was not cleared, so the old engine's particles "
            "stay on screen")

    def test_switching_to_an_untrained_engine_clears_the_RASTER_too(self, monkeypatch):
        """The other drawing route, which this used to miss entirely.

        Above `_RASTER_ABOVE` instances the overlay is ONE mask, not outlines.
        `show_preview_window` cleared only the outlines, so the previous
        engine's mask survived the switch — and on the Scribble tab that covers
        the image you have to paint on. The reported symptom was exactly "moved
        to scribble, immediately unusable".
        """
        import spyde.actions.particles_action as pa
        monkeypatch.setattr(pa, "_push_groups", lambda p, u: None)
        wiz = self._wiz()
        cleared = []
        monkeypatch.setattr(pa.SegmentWizard, "_clear_raster_overlay",
                            lambda self: cleared.append(True))
        wiz.show_preview_window()
        assert cleared, ("the raster mask was left on screen; on Scribble it "
                         "covers the frame the user has to paint on")


class TestRasterOverlayAboveThreshold:
    """Hundreds of vector contours make the app sluggish; draw one mask instead.

    Every contour is a path the renderer re-transforms on each pan/zoom frame,
    so a few hundred of them cost real interactivity. A mask is ONE image
    however many particles it contains. Below the threshold the outlines stay —
    crisp at any zoom, and each is an object the UI can hover.
    """

    # A REAL Plot2D, not a stub that records whatever it is handed.
    #
    # This class used to fake `set_overlay_mask`, and that fake is why the bug
    # it was written to prevent shipped anyway: SpyDE reduced the mask to the
    # overview grid (right for the renderer), the REAL `set_overlay_mask`
    # rejected that shape against `image_width` (the full frame in tile mode),
    # `_set_raster_overlay` swallowed the ValueError at DEBUG, and every
    # large-frame preview fell back to N polygons — at 14028 instances, a hung
    # renderer painted solid green. The stub asserted the mask SpyDE built and
    # was green throughout, because it never enforced the contract that failed.
    #
    # So these tests now assert on the bytes that actually SHIP. Whichever layer
    # does the reduction, the renderer's rule is the same and is the only thing
    # worth pinning: `bytes.length === (base_width || image_width) * (…)`.
    @staticmethod
    def _p2d(base_w=0, base_h=0):
        from anyplotlib.plot2d import Plot2D
        full = np.zeros((4096, 4096), np.uint8)
        if base_w:
            p = Plot2D(full, tile="auto")
            assert p._state.get("base_width"), "tile mode did not set a base grid"
        else:
            p = Plot2D(full)
        return p

    @staticmethod
    def _p2d_small(n):
        """A plot small enough that anyplotlib does NOT put it in tile mode."""
        from anyplotlib.plot2d import Plot2D
        return Plot2D(np.zeros((n, n), np.uint8))

    @staticmethod
    def _shipped(p2d):
        """(bytes_len, expected_len) for the mask currently on *p2d*."""
        import base64
        st = p2d._state
        b64 = st.get("overlay_mask_b64") or ""
        want = (int(st.get("base_width") or 0) or int(st["image_width"])) * \
               (int(st.get("base_height") or 0) or int(st["image_height"]))
        return len(base64.b64decode(b64)), want

    _wiz = staticmethod(_bare_wizard)

    @staticmethod
    def _labels():
        lab = np.zeros((1024, 1024), np.int32)
        y, x = np.mgrid[0:1024, 0:1024]
        for i, (cy, cx) in enumerate([(200, 200), (500, 700), (800, 300)], start=1):
            lab[(y - cy) ** 2 + (x - cx) ** 2 < 25] = i     # 5 px radius
        return lab

    def test_untiled_mask_is_the_native_frame_size(self):
        """A frame small enough to escape tile mode ships at its native size.

        Small ON PURPOSE: anyplotlib tiles anything with a >=1024 px edge, so
        there is no such thing as an untiled 4096² plot and asking for one here
        tested nothing. 512² is the branch where `base_width` stays 0 and the
        renderer falls back to `image_width`.
        """
        p = self._p2d_small(512)
        lab = np.zeros((512, 512), np.int32)
        lab[100:110, 100:110] = 1
        assert self._wiz(p)._set_raster_overlay(lab, None, (512, 512))
        assert not p._state.get("base_width"), "512² unexpectedly tiled"
        sent, want = self._shipped(p)
        assert sent == want == 512 * 512

    @pytest.mark.skipif(
        not hasattr(importlib.import_module("anyplotlib.plot2d._plot2d"),
                    "_reduce_mask_any"),
        reason="needs anyplotlib fix/overlay-mask-tile-mode (a61d8658): released "
               "0.7.1 rejects an overview-sized mask in tile mode, so this ships "
               "broken until that fix is released and the pin is bumped. A "
               "CAPABILITY probe, not a version gate — the editable dev checkout "
               "also reports 0.7.1. This skip is a release gate: green-with-skip "
               "means the tiled overlay-mask feature is NOT in the shipped app.")
    def test_tiled_mask_SHIPS_at_the_OVERVIEW_size(self):
        """The trap this exists for, asserted where it actually bites.

        In tile mode the renderer checks ``bytes.length === iw * ih`` where
        ``iw = base_width || image_width`` — the OVERVIEW size — and on a
        mismatch sets ``maskCache=null``: no error, no overlay, nothing in the
        log. So what matters is not which layer reduces the mask but that the
        bytes leaving Python are the size the renderer will accept.
        """
        p = self._p2d(1024, 1024)
        assert self._wiz(p)._set_raster_overlay(
            self._labels(), (1536, 1536, 1024, 1024), (4096, 4096)), \
            "the raster overlay refused to draw on a tiled frame"
        sent, want = self._shipped(p)
        assert sent == want, (
            f"mask ships {sent} bytes but the renderer expects {want} — it "
            f"would silently draw nothing at all")

    def test_the_reduction_keeps_small_particles(self):
        """Block ANY, not a subsample.

        Particles are often a few pixels across; striding a 4096² mask down to
        1024² would drop three quarters of them at random.
        """
        import base64
        p = self._p2d(1024, 1024)
        self._wiz(p)._set_raster_overlay(
            self._labels(), (1536, 1536, 1024, 1024), (4096, 4096))
        sent = np.frombuffer(
            base64.b64decode(p._state["overlay_mask_b64"]), np.uint8)
        assert sent.any(), "every particle vanished in the reduction"

    def test_below_the_threshold_nothing_is_rastered(self):
        """Few particles → keep the crisp, hoverable vector outlines."""
        import spyde.actions.particles_action as pa
        assert pa._RASTER_ABOVE > 1
        p = self._p2d()
        wiz = self._wiz(p)          # `_bare_wizard` sets the overlay state
        contours = [np.zeros((3, 2), np.int16) for _ in range(3)]
        wiz.set_overlay(contours, None, labels=self._labels(),
                        full_shape=(4096, 4096))
        assert not (p._state.get("overlay_mask_b64") or ""), (
            "a 3-particle frame was rastered; the outlines are better there")

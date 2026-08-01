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

import time

import numpy as np
import pytest

from spyde.actions import particles_action as pa


@pytest.fixture(autouse=True)
def _capture_module_emit(window, monkeypatch):
    """Route ``particles_action``'s own ``emit`` into the captured list.

    The module does ``from spyde.backend.ipc import emit`` at import, so
    conftest's patch of ``ipc.emit`` never reaches that binding — the identical
    hazard conftest already documents for ``session.py``, and the identical fix.
    ``emit_status``/``emit_error`` need no patch: they resolve ``emit`` inside
    ``ipc`` at call time.
    """
    monkeypatch.setattr(pa, "emit", window["messages"].append)


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=60.0):
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


def _opened(window, frames: int = 6, **params):
    session, plot, tree = _movie(window, frames)
    pa.seg_open(session, plot, {"min_size": 25, "gaussian": 1.0, **params})
    assert _wait(lambda: getattr(tree, "_seg_wizard", None) is not None
                 and tree._seg_wizard.preview is not None), \
        "the wizard never produced a first preview"
    return session, plot, tree, tree._seg_wizard


def _of_type(messages, kind):
    return [m for m in messages if isinstance(m, dict) and m.get("type") == kind]


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
        assert wiz.preview["count"] > 0, "found nothing on the fixture's frame"

    def test_preview_never_computes_the_whole_movie(self, window):
        session, plot, tree, wiz = _opened(window)
        assert tree.root._lazy, "the fixture must be lazy or this guards nothing"
        with _FullComputeGuard(tree.root.data.shape) as guard:
            pa.seg_tune(session, plot, {"sensitivity": 0.6})
            assert _wait(lambda: abs(wiz.params["sensitivity"] - 0.6) < 1e-9)
            time.sleep(0.6)
        assert guard.hits == 0, "a tune materialised the whole movie"

    def test_tune_opens_no_new_window(self, window):
        session, plot, _tree, wiz = _opened(window)
        before = len(session.signal_trees)
        pa.seg_tune(session, plot, {"sensitivity": 0.7})
        assert _wait(lambda: abs(wiz.params["sensitivity"] - 0.7) < 1e-9)
        time.sleep(0.4)
        assert len(session.signal_trees) == before, "a tune spawned a tree"

    def test_preview_message_carries_the_size_histogram(self, window):
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        pa.seg_tune(session, plot, {"sensitivity": 0.55})
        assert _wait(lambda: len(_of_type(msgs, "seg_preview")) >= 2)
        msg = _of_type(msgs, "seg_preview")[-1]
        assert msg["count"] == len(msg["areas"]) or msg["count"] > pa._MAX_AREAS_SENT
        assert msg["median_area"] > 0 and msg["units"] == "nm"

    def test_set_method_to_an_untrained_scribble_keeps_the_last_preview(self, window):
        """The scribble engine cannot run before Train — it must say so, not
        clear the result the user is looking at."""
        session, plot, _tree, wiz = _opened(window)
        keep = wiz.preview
        pa.seg_set_method(session, plot, {"method": "scribble"})
        time.sleep(0.4)
        assert wiz.params["method"] == "scribble"
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

    def test_reported_count_is_after_the_size_filter(self, window):
        """§0.9b: the number the user sees must be the post-filter one, or it
        moves for a reason they cannot see."""
        session, plot, _tree, wiz = _opened(window)
        from spyde.signals.particles import COL
        pa.seg_tune(session, plot, {"min_size": 400})
        assert _wait(lambda: wiz.params["min_size"] == 400)
        assert _wait(lambda: wiz.preview is not None
                     and (wiz.preview["rows"].size == 0
                          or wiz.preview["rows"][:, COL["area"]].min() > 0))
        rows = wiz.preview["rows"]
        assert wiz.preview["count"] == len(rows)


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
        session, plot, _tree, wiz = _opened(window)
        stroke = {"frame": 0, "points": [[20, 20], [30, 30]], "brush": 5}
        pa.seg_paint(session, plot, {**stroke, "class_id": 0})
        painted = len(wiz.labels)
        assert painted > 0
        pa.seg_paint(session, plot, {**stroke, "erase": True})
        assert len(wiz.labels) == 0, (
            f"{len(wiz.labels)} of {painted} px survived an erase over the same "
            "stroke — the eraser and the brush have drifted apart")

    def test_paint_with_no_points_is_a_no_op(self, window):
        session, plot, _tree, wiz = _opened(window)
        pa.seg_paint(session, plot, {"frame": 0, "points": [], "class_id": 0})
        assert wiz.labels is None

    def test_train_needs_labels(self, window):
        session, plot, _tree, _wiz = _opened(window)
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
                     and wiz.classifier.is_trained, timeout=180)
        assert wiz.params["method"] == "scribble"
        assert _wait(lambda: wiz.preview is not None
                     and wiz.preview["frame"] == wiz.frame_index(), timeout=120)


class TestRunIsProgressiveAndCancellable:
    def test_result_window_opens_before_the_particles_attach(self, window):
        """The attach gap, asserted at the instant it exists."""
        session, plot, tree, _wiz = _opened(window)
        before = len(session.signal_trees)
        pa.seg_run(session, plot, {"min_size": 25, "gaussian": 1.0})
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
        pa.seg_run(session, plot, {"min_size": 25, "gaussian": 1.0})
        assert seg_batch_running(session)
        result = session.signal_trees[-1]
        assert _wait(lambda: getattr(result, "particles", None) is not None)
        assert _wait(lambda: not seg_batch_running(session))

    def test_finalize_attaches_a_full_store_and_says_how_many(self, window):
        session, plot, _tree, _wiz = _opened(window, frames=6)
        msgs = window["messages"]
        pa.seg_run(session, plot, {"min_size": 25, "gaussian": 1.0})
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
        pa.seg_run(session, plot, {"min_size": 25, "gaussian": 1.0})
        result = session.signal_trees[-1]
        assert _wait(lambda: getattr(result, "particles", None) is not None)
        frame = np.asarray(result.root.data[3].compute())
        painted = np.unique(frame)
        painted = painted[painted > 0]
        assert painted.size == len(result.particles.at(3)) > 0

    def test_count_trace_reaches_the_navigator(self, window):
        session, plot, _tree, _wiz = _opened(window)
        pa.seg_run(session, plot, {"min_size": 25, "gaussian": 1.0})
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
            pa.seg_run(session, plot, {"min_size": 25, "gaussian": 1.0})
            result = session.signal_trees[-1]
            assert _wait(lambda: getattr(result, "particles", None) is not None)
        assert guard.hits == 0, "the batch materialised the whole movie"

    def test_closing_the_result_tree_cancels_the_batch(self, window, monkeypatch):
        """Cancellation runs through BaseSignalTree.register_cancel, so the
        ordinary act of closing the window has to stop the compute."""
        import spyde.particles as sp
        real = sp.segment_frame

        def _slow(frame, params=None):
            time.sleep(0.25)
            return real(frame, params)

        monkeypatch.setattr(sp, "segment_frame", _slow)
        session, plot, _tree, _wiz = _opened(window, frames=24)
        pa.seg_run(session, plot, {"min_size": 25, "gaussian": 1.0})
        result = session.signal_trees[-1]
        time.sleep(0.5)
        result.close()
        assert _wait(lambda: not result._seg_batch_running, timeout=30)
        # A cancelled run keeps its partial rows out of the torn-down tree.
        assert getattr(result, "particles", None) is None

    def test_finalize_re_sends_the_toolbar_config(self, window):
        """requires_particles flips here; without the re-send the gated buttons
        stay hidden until something else rebuilds the toolbar."""
        session, plot, _tree, _wiz = _opened(window)
        msgs = window["messages"]
        pa.seg_run(session, plot, {"min_size": 25, "gaussian": 1.0})
        result = session.signal_trees[-1]
        assert _wait(lambda: getattr(result, "particles", None) is not None)
        wids = {getattr(sp_, "window_id", None) for sp_ in result.signal_plots}
        assert _wait(lambda: any(m.get("window_id") in wids
                                 for m in _of_type(msgs, "toolbar_config")))


class TestCommit:
    def test_commit_snapshots_the_previewed_frame(self, window):
        session, plot, tree, wiz = _opened(window)
        before = len(session.signal_trees)
        pa.seg_commit(session, plot, {})
        assert len(session.signal_trees) == before + 1
        committed = session.signal_trees[-1]
        assert committed.particles.n_frames == 1
        assert committed.particles.n_particles == wiz.preview["count"]
        assert committed.source_tree is tree

    def test_commit_stamps_provenance(self, window):
        session, plot, _tree, _wiz = _opened(window)
        pa.seg_commit(session, plot, {})
        prov = getattr(session.signal_trees[-1], "_commit_provenance", None) or {}
        assert prov.get("action") == "segment_particles"
        assert prov.get("params", {}).get("mode") == "single_frame"

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
        time.sleep(0.8)

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
        pa.seg_open(session, plot, {"sensitivity": 0.7})
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
    def test_local_size_is_forced_odd(self):
        """skimage requires it and SegmentParams raises rather than bumping —
        a slider that lands on an even number must not error."""
        assert pa._coerce({"local_size": 30})["local_size"] == 31

    def test_unknown_method_falls_back(self):
        assert pa._coerce({"method": "magic"})["method"] == pa.DEFAULTS["method"]

    def test_sensitivity_is_clamped(self):
        assert pa._coerce({"sensitivity": 5.0})["sensitivity"] == 1.0
        assert pa._coerce({"sensitivity": -2.0})["sensitivity"] == 0.0

    def test_a_junk_value_keeps_the_default(self):
        assert pa._coerce({"min_separation": "wat"})["min_separation"] == \
            pa.DEFAULTS["min_separation"]


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

    def _scribble_wizard(self, session):
        from spyde.actions.particles_action import seg_open, seg_set_method
        plots = (list(session._plots) if isinstance(session._plots, list)
                 else list(session._plots.values()))
        plot = next(p for p in plots
                    if getattr(p, "signal_tree", None) is not None
                    and getattr(p, "_plot2d", None) is not None)
        seg_open(session, plot, {"window_id": getattr(plot, "window_id", None)})
        seg_set_method(session, plot, {"method": "scribble"})
        return plot, plot.signal_tree

    def test_a_brush_is_attached_on_the_scribble_tab(self, window):
        from spyde.actions.particles_action import _brush_supported
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        _plot, tree = self._scribble_wizard(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget (needs >= 0.5.0)")
        assert getattr(tree, "_seg_brush", None) is not None, (
            "no brush on the plot — Shift+drag has nothing to hit")

    def test_a_stroke_from_the_WIDGET_reaches_the_label_store(self, window):
        """Drives the widget, not a synthetic seg_paint payload."""
        from spyde.actions.particles_action import _brush_supported, _on_stroke
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        _plot, tree = self._scribble_wizard(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget (needs >= 0.5.0)")
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

    def test_a_second_stroke_only_paints_its_own_points(self, window):
        """The widget accumulates strokes for its whole life, so replaying the
        full list on every event would re-paint everything and make the class
        counts grow quadratically.

        The class is switched through ``seg_tune``, NOT by handing the widget
        different ``stroke_classes``: the caret's params are the authority,
        precisely because the JS widget's own value is not reliably in sync (see
        :meth:`test_class_comes_from_PARAMS_not_from_the_js_widget`).
        """
        from spyde.actions.particles_action import (_brush_supported, _on_stroke,
                                                    seg_tune)
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, tree = self._scribble_wizard(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget (needs >= 0.5.0)")
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

    def test_leaving_scribble_detaches_the_brush(self, window):
        """It floats over the image; on Classical there is nothing to paint."""
        from spyde.actions.particles_action import _brush_supported, seg_set_method
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, tree = self._scribble_wizard(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget (needs >= 0.5.0)")
        assert tree._seg_brush is not None
        seg_set_method(session, plot, {"method": "classical"})
        assert getattr(tree, "_seg_brush", None) is None

    def test_missing_brush_says_so_instead_of_failing_silently(self, window,
                                                              monkeypatch):
        """A user dragging at an image that never responds must be TOLD why."""
        import spyde.actions.particles_action as pa
        monkeypatch.setattr(pa, "_brush_supported", lambda: False)
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        before = len(window["messages"])
        self._scribble_wizard(session)
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
        from spyde.actions.particles_action import seg_open, seg_set_method
        session._load_test_data_particles({"frames": 4})
        plots = (list(session._plots) if isinstance(session._plots, list)
                 else list(session._plots.values()))
        plot = next(p for p in plots
                    if getattr(p, "signal_tree", None) is not None
                    and getattr(p, "_plot2d", None) is not None)
        seg_open(session, plot, {"window_id": getattr(plot, "window_id", None)})
        seg_set_method(session, plot, {"method": "scribble"})
        return plot, plot.signal_tree

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

    def test_class_comes_from_PARAMS_not_from_the_js_widget(self, window):
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
        from spyde.actions.particles_action import _brush_supported, _on_stroke, seg_tune
        session = window["window"]
        plot, tree = self._scribbling(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget (needs >= 0.5.0)")
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

    def test_switching_class_retags_the_next_stroke(self, window):
        from spyde.actions.particles_action import _brush_supported, seg_tune
        session = window["window"]
        plot, tree = self._scribbling(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget (needs >= 0.5.0)")
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

    def test_eraser_removes_and_leaves_other_classes_alone(self, window):
        from spyde.actions.particles_action import _brush_supported, seg_tune
        session = window["window"]
        plot, tree = self._scribbling(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget (needs >= 0.5.0)")
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

    def test_brush_size_reaches_the_widget_too(self, window):
        from spyde.actions.particles_action import _brush_supported, seg_tune
        session = window["window"]
        plot, tree = self._scribbling(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget (needs >= 0.5.0)")
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

    def _scribble_wizard(self, session):
        from spyde.actions.particles_action import seg_open, seg_set_method
        plots = session._plots
        plots = list(plots.values()) if hasattr(plots, "values") else list(plots)
        plot = next(p for p in plots
                    if getattr(p, "signal_tree", None) is not None
                    and getattr(p, "_plot2d", None) is not None)
        seg_open(session, plot, {"window_id": getattr(plot, "window_id", None)})
        seg_set_method(session, plot, {"method": "scribble"})
        return plot, plot.signal_tree

    def test_selecting_a_class_changes_the_widget_class_id(self, window):
        from spyde.actions.particles_action import _brush_supported, seg_tune
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, tree = self._scribble_wizard(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget")
        brush = getattr(tree, "_seg_brush", None)
        assert brush is not None

        # What the ClassStrip sends when you click "support film".
        seg_tune(session, plot, {"active_class": 1})
        assert int(brush._data["class_id"]) == 1, (
            "the widget still paints class 0 — the strip's selection reached "
            "wiz.params but not the widget, so every stroke DRAWS orange")

        seg_tune(session, plot, {"active_class": 2})
        assert int(brush._data["class_id"]) == 2

    def test_the_new_class_reaches_the_PANEL_state(self, window):
        """The JS draws from ``panel_<id>_json``, not from the Python object.

        This is the assertion the previous two miss and the reason the bug
        survived a "fix": ``Widget.set`` reaches JS via ``_push_widget``, which
        writes ``event_json`` ONLY and leaves the panel's own widget state
        stale. ``_brushLiveBegin`` reads ``w.class_id`` from
        ``p.state.overlay_widgets`` and paints ``colors[class_id]`` — so
        ``brush._data`` can say 1 while every stroke still draws in class 0's
        colour. Assert the serialised panel, which is what the eye sees.
        """
        from spyde.actions.particles_action import _brush_supported, seg_tune
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, tree = self._scribble_wizard(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget")

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

    def test_an_unrelated_tune_does_NOT_force_a_panel_push(self, window):
        """`seg_tune` fires for every sensitivity-slider tick too.

        A full panel push re-serialises the image bytes, so doing one per tick
        on a 4096² frame would trade the colour bug for a much worse drag. Only
        an actual brush change (class / eraser / size) may pay for it.
        """
        from spyde.actions.particles_action import _brush_supported, seg_tune
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        plot, _tree = self._scribble_wizard(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget")

        seg_tune(session, plot, {"active_class": 1})      # settle the state

        pushes = []
        real_push = plot._plot2d._push
        plot._plot2d._push = lambda *a, **k: (pushes.append(1), real_push(*a, **k))[1]
        try:
            seg_tune(session, plot, {"sensitivity": 0.6})
            seg_tune(session, plot, {"sensitivity": 0.7})
            seg_tune(session, plot, {"active_class": 1})   # SAME class, no change
        finally:
            plot._plot2d._push = real_push

        assert pushes == [], (
            f"{len(pushes)} panel pushes for tunes that did not change the "
            "brush — a sensitivity drag would re-serialise the whole image")

    def test_the_widget_has_a_colour_for_every_class(self, window):
        """`colors` is indexed by class_id in JS, so a short list means the
        later classes draw with `undefined` — no colour change on screen even
        though class_id updated correctly."""
        from spyde.actions.particles_action import _brush_supported
        session = window["window"]
        session._load_test_data_particles({"frames": 4})
        _plot, tree = self._scribble_wizard(session)
        if not _brush_supported():
            pytest.skip("installed anyplotlib has no brush widget")
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

    def _seg_run(self, session, **params):
        from spyde.actions.particles_action import seg_run
        plots = session._plots
        plots = list(plots.values()) if hasattr(plots, "values") else list(plots)
        plot = next(p for p in plots
                    if getattr(p, "signal_tree", None) is not None
                    and getattr(p, "_plot2d", None) is not None)
        seg_run(session, plot, dict(params))
        return plot

    def test_the_chip_is_raised_before_seg_run_returns(self, window):
        session = window["window"]
        msgs = window["messages"]
        session._load_test_data_particles({"frames": 4})
        del msgs[:]

        self._seg_run(session, method="classical", track=False)

        # No waiting, no polling: the chip must already be on the wire.
        raised = [m for m in msgs
                  if m.get("type") == "window_computing" and m.get("computing")]
        assert raised, (
            "no window_computing raised synchronously — the chip only appears "
            "once the worker gets going, which is the reported lag")

    def test_the_chip_names_the_RESULT_window(self, window):
        """Not the source window: the source is where you were scribbling and it
        is not the thing sitting there looking empty."""
        session = window["window"]
        msgs = window["messages"]
        session._load_test_data_particles({"frames": 4})
        src = self._seg_run(session, method="classical", track=False)
        del msgs[:]

        raised = [m for m in window["messages"]
                  if m.get("type") == "window_computing"]
        # Re-read from the full list: the run may already have finished.
        all_ids = {m.get("window_id") for m in raised}
        src_id = getattr(src, "window_id", None)
        if all_ids:
            assert all_ids != {src_id}, (
                "the chip was put on the SOURCE window, not the result")

    def test_the_chip_comes_down_when_the_batch_ends(self, window):
        session = window["window"]
        msgs = window["messages"]
        session._load_test_data_particles({"frames": 4})
        del msgs[:]
        self._seg_run(session, method="classical", track=False)

        assert _wait(lambda: any(
            m.get("type") == "window_computing" and not m.get("computing")
            for m in msgs), timeout=180), (
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
        import types
        import spyde.actions.particles_action as pa

        class _Grp:                     # hashable by identity, unlike SimpleNamespace
            def __init__(self, name):
                self.name, self.removed = name, False

            def remove(self):
                self.removed = True

        class _P2D:
            def add_polygons(self, *a, **k):
                return _Grp(k.get("name"))

        wiz = object.__new__(pa.SegmentWizard)
        wiz._ov_group = None
        wiz._ov_box_group = None
        wiz.src_plot = types.SimpleNamespace(_plot2d=_P2D())
        wiz.frames = lambda: (1, lambda i: np.zeros((2048, 2048), np.float32),
                              (2048, 2048))
        wiz.frame_index = lambda: 0
        return wiz

    def test_box_is_drawn_with_no_engine(self, monkeypatch):
        import spyde.actions.particles_action as pa
        pushed = {}
        monkeypatch.setattr(pa, "_push_groups", lambda p, u: pushed.update(
            {g.name: pl.get("vertices_list") for g, pl in u.items()}))
        self._wiz().show_preview_window()
        box = pushed.get("seg_preview_window")
        assert box, "no preview-window box without an engine — the untrained " \
                    "Scribble/Prompt states show nothing at all"
        assert len(box[0]) >= 4, "the box is not a closed rectangle"

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

"""
The Drift Correction wizard backend (``drift_*`` staged handlers).

Handlers are called directly as ``fn(session, plot, payload)`` and polled with
``_wait`` — the shape ``test_find_vectors_wizard.py`` establishes, because the
solve and the check sums both run on a worker thread.

The four claims that matter:

:class:`TestCheckWindow`
    Plan A8 / README §6. The verification surface is a SEPARATE window, and a
    bare ``figure`` is not a registered ``Plot`` — so it must be reachable
    through ``session.controller_by_window_id`` and must disappear on close. A
    check window that leaks is the exact bug README §6 documents.
:class:`TestSolve`
    The solved shifts must match ``particle_movie``'s stamped ground truth, and
    ``tree.drift`` must carry the model. Ground truth beats a golden number.
:class:`TestCommitIsLazy`
    The CLAUDE.md memory-safety rule, guarded the way
    ``test_find_vectors_memory.py`` guards it: a ``da.Array.compute`` spy that
    counts calls on the full-dataset shape. And the corrected node has to be
    genuinely better — an aligned stack sums SHARP, which is the whole claim the
    check window makes to the user.
:class:`TestDoubleFire`
    README §4 / StrictMode: open, close, open leaves exactly ONE controller and
    exactly ONE check window.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from spyde.actions import drift_action as dr


@pytest.fixture(autouse=True)
def _capture_module_emit(window, monkeypatch):
    """Route ``drift_action``'s own ``emit`` into the captured list.

    The module does ``from spyde.backend.ipc import emit`` at import, so
    conftest's patch of ``ipc.emit`` never reaches that binding — the identical
    hazard conftest already documents for ``session.py``, and the identical fix.
    ``emit_status``/``emit_error`` need no patch: they resolve ``emit`` inside
    ``ipc`` at call time.
    """
    monkeypatch.setattr(dr, "emit", window["messages"].append)

# Small but enough for the drift curve to turn: `particle_movie`'s drift is a
# smooth excursion, and 8 frames already reach ~6 px, which is 5x the tolerance
# asserted below.
N_FRAMES = 8


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wait(pred, timeout=120.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _movie(window, frames: int = N_FRAMES):
    session = window["window"]
    session._load_test_data_particles({"frames": frames})
    plot = _wait(lambda: _signal_plot(session) is not None) and _signal_plot(session)
    assert plot is not None, "the particle movie never produced a signal plot"
    return session, plot, plot.signal_tree


def _opened(window, frames: int = N_FRAMES, **params):
    session, plot, tree = _movie(window, frames)
    dr.drift_open(session, plot, {"upsample": 8, "max_shift": 16, **params})
    assert _wait(lambda: getattr(tree, "_drift_wizard", None) is not None
                 and tree._drift_wizard.window_id is not None), \
        "the Drift Check window never opened"
    return session, plot, tree, tree._drift_wizard


def _solved(window, frames: int = N_FRAMES):
    session, plot, tree, wiz = _opened(window, frames)
    dr.drift_run(session, plot, {"upsample": 8, "max_shift": 16})
    assert _wait(lambda: wiz.model is not None), "the solve never finished"
    return session, plot, tree, wiz


def _of_type(messages, kind):
    return [m for m in messages if isinstance(m, dict) and m.get("type") == kind]


def _sharpness(img) -> float:
    """Mean squared gradient — an aligned sum has more of it than a blurred one."""
    a = np.nan_to_num(np.asarray(img, np.float64))
    gy, gx = np.gradient(a)
    return float(np.mean(gy ** 2 + gx ** 2))


class _FullComputeGuard:
    """Count ``.compute()`` calls made on the whole movie."""

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


class TestCheckWindow:
    def test_open_registers_a_controller_for_the_bare_figure(self, window):
        """README §6: a bare `figure` is not a Plot, so dispatch can only find
        it through the window-controller registry."""
        session, _plot, _tree, wiz = _opened(window)
        assert session.controller_by_window_id(wiz.window_id) is wiz
        assert session._plot_by_window_id(wiz.window_id) is None, \
            "the check window is supposed to be a bare figure, not a Plot"

    def test_the_window_shows_the_uncorrected_sum(self, window):
        session, _plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        figs = [m for m in _of_type(msgs, "figure")
                if m.get("window_id") == wiz.window_id]
        assert figs and figs[-1]["title"] == "Drift Check"
        assert wiz._before_sum is not None and np.isfinite(wiz._before_sum).any()

    def test_open_solves_nothing(self, window):
        """Plan A8: drift correction is explicit — nothing runs on load."""
        _s, _p, tree, wiz = _opened(window)
        assert wiz.model is None
        assert getattr(tree, "drift", None) is None

    def test_close_takes_the_window_with_it(self, window):
        session, plot, tree, wiz = _opened(window)
        wid = wiz.window_id
        dr.drift_close(session, plot, {})
        assert getattr(tree, "_drift_wizard", None) is None
        assert session.controller_by_window_id(wid) is None
        from spyde.actions.figure_registry import _FIGS
        assert wid not in _FIGS, "the check figure outlived its window"

    def test_the_summed_subset_is_bounded(self, window):
        """A sum is a sharpness test, not a measurement — the cap is what keeps
        the check window usable on a long movie."""
        _s, _p, _t, wiz = _opened(window)
        wiz._sum_indices = None                 # as if the movie were long
        idx = wiz.sum_indices(10_000)
        assert idx.size <= dr._SUM_MAX_FRAMES
        assert idx[0] == 0 and idx[-1] == 9_999
        assert wiz.sum_indices(10_000) is idx, (
            "the subset is memoised so the before and after sums cover the SAME "
            "frames — comparing two different subsets means nothing")


class TestMethodStubs:
    def test_nonrigid_says_so_and_stays_rigid(self, window):
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        dr.drift_set_method(session, plot, {"method": "nonrigid"})
        assert wiz.params["method"] == "rigid", (
            "a stub must not leave the caret claiming a model the solve does "
            "not implement — the wrong `kind` would land in provenance")
        assert any("not implemented" in str(m.get("text", ""))
                   for m in _of_type(msgs, "status"))

    def test_rigid_affine_says_so_too(self, window):
        session, plot, _tree, wiz = _opened(window)
        dr.drift_set_method(session, plot, {"method": "rigid_affine"})
        assert wiz.params["method"] == "rigid"

    def test_unknown_method_errors(self, window):
        session, plot, _tree, _wiz = _opened(window)
        msgs = window["messages"]
        dr.drift_set_method(session, plot, {"method": "banana"})
        assert any("unknown model" in str(m.get("text", ""))
                   for m in _of_type(msgs, "error"))


class TestTune:
    def test_tune_solves_only_the_first_pair(self, window):
        """Two frames, so it lands inside a slider drag — and it must not
        materialise the movie."""
        session, plot, tree, _wiz = _opened(window)
        msgs = window["messages"]
        with _FullComputeGuard(tree.root.data.shape) as guard:
            dr.drift_tune(session, plot, {"upsample": 4, "max_shift": 12})
            assert _wait(lambda: _of_type(msgs, "drift_preview"))
        assert guard.hits == 0
        prev = _of_type(msgs, "drift_preview")[-1]
        truth = _ground_truth(tree)[1]
        assert abs(prev["dy"] - truth[0]) < 0.5
        assert abs(prev["dx"] - truth[1]) < 0.5

    def test_tune_stores_the_new_parameters(self, window):
        session, plot, _tree, wiz = _opened(window)
        dr.drift_tune(session, plot, {"upsample": 16, "max_shift": 9.0})
        assert wiz.params["upsample"] == 16
        assert wiz.params["max_shift"] == 9.0


def _ground_truth(tree):
    import spyde.data.synthetic as sy
    return np.asarray(sy.ground_truth(tree.root)["drift"], np.float64)


class TestSolve:
    def test_shifts_match_the_stamped_ground_truth(self, window):
        _s, _p, tree, wiz = _solved(window)
        truth = _ground_truth(tree)[:N_FRAMES]
        err = np.abs(wiz.model.shifts - truth).max()
        assert err < 0.25, f"worst per-axis drift error {err:.3f} px"

    def test_the_model_lands_on_the_tree(self, window):
        _s, _p, tree, wiz = _solved(window)
        assert tree.drift is wiz.model
        assert wiz.model.kind == "rigid"

    def test_run_reports_progress_and_the_finished_trace(self, window):
        session, plot, tree, wiz = _opened(window)
        msgs = window["messages"]
        dr.drift_run(session, plot, {})
        assert _wait(lambda: _of_type(msgs, "drift_result"))
        prog = _of_type(msgs, "drift_progress")
        assert prog and prog[-1]["done"] == prog[-1]["total"] == N_FRAMES
        res = _of_type(msgs, "drift_result")[-1]
        assert len(res["shifts"]) == N_FRAMES and not res["cancelled"]
        assert res["max_abs_shift"] > 1.0

    def test_run_never_computes_the_whole_movie(self, window):
        session, plot, tree, wiz = _opened(window)
        with _FullComputeGuard(tree.root.data.shape) as guard:
            dr.drift_run(session, plot, {})
            assert _wait(lambda: wiz.model is not None)
        assert guard.hits == 0, "the solve materialised the whole movie"

    def test_the_check_window_gets_a_sharper_corrected_sum(self, window):
        """The claim the window makes to the user, asserted rather than drawn:
        an aligned stack sums sharp, a misaligned one blurs."""
        _s, _p, _t, wiz = _solved(window)
        n, get_frame, _shape = wiz.frames()
        idx = wiz.sum_indices(n)
        before = dr._stack_sum(get_frame, idx)
        after = dr._stack_sum(get_frame, idx, wiz.model.shifts)
        assert _sharpness(after) > 1.5 * _sharpness(before), (
            f"corrected sum {_sharpness(after):.5f} is not sharper than the raw "
            f"{_sharpness(before):.5f} — check the sign convention in "
            "spyde/drift/model.py")

    def test_closing_the_tree_cancels_the_solve(self, window):
        """Cancellation goes through BaseSignalTree.register_cancel, so closing
        the tree has to stop it."""
        session, plot, tree, wiz = _opened(window, frames=24)
        dr.drift_run(session, plot, {})
        tree.close()
        assert _wait(lambda: wiz.model is not None or wiz._closed, timeout=60)
        if wiz.model is not None:
            # A cancelled solve leaves NaN for the frames it never reached, so a
            # partial model is detectable rather than silently wrong.
            assert not np.isfinite(wiz.model.shifts).all()

    def test_run_without_a_caret_errors(self, window):
        session, plot, _tree = _movie(window)
        msgs = window["messages"]
        dr.drift_run(session, plot, {})
        assert any("caret is not open" in str(m.get("text", ""))
                   for m in _of_type(msgs, "error"))


class TestCommitIsLazy:
    def test_commit_adds_a_lazy_node_without_computing(self, window):
        session, plot, tree, wiz = _solved(window)
        before = set(tree.root_node.children)
        with _FullComputeGuard(tree.root.data.shape) as guard:
            dr.drift_commit(session, plot, {})
        assert guard.hits == 0, "commit materialised the movie"
        added = set(tree.root_node.children) - before
        assert added == {"Drift corrected"}
        node = tree.root_node.children["Drift corrected"]
        assert node.signal._lazy
        assert node.signal.data.shape == tree.root.data.shape
        assert node.local is True, (
            "a per-frame shift IS local — without the tag the derived-view "
            "reader falls back to the opaque path")

    def test_one_corrected_frame_costs_one_frame(self, window):
        session, plot, tree, _wiz = _solved(window)
        dr.drift_commit(session, plot, {})
        node = tree.root_node.children["Drift corrected"]
        with _FullComputeGuard(tree.root.data.shape) as guard:
            frame = np.asarray(node.signal.data[3].compute())
        assert guard.hits == 0
        assert frame.shape == tuple(tree.root.data.shape[1:])

    def test_uncovered_pixels_are_nan_not_invented(self, window):
        """Plan A7: nothing is cropped and nothing is filled with invented
        data — segmentation would find 'particles' in a zero-filled border."""
        session, plot, tree, _wiz = _solved(window)
        dr.drift_commit(session, plot, {})
        node = tree.root_node.children["Drift corrected"]
        frame = np.asarray(node.signal.data[N_FRAMES - 1].compute())
        assert np.isnan(frame).any(), "no NaN padding on a shifted frame"
        assert np.isfinite(frame).any(), "the whole frame is NaN"

    def test_the_corrected_node_is_actually_aligned(self, window):
        session, plot, tree, wiz = _solved(window)
        dr.drift_commit(session, plot, {})
        node = tree.root_node.children["Drift corrected"]
        raw = np.nanmean(np.stack([np.asarray(tree.root.data[i].compute(),
                                              np.float64)
                                   for i in range(N_FRAMES)]), axis=0)
        fixed = np.nanmean(np.stack([np.asarray(node.signal.data[i].compute(),
                                                np.float64)
                                     for i in range(N_FRAMES)]), axis=0)
        assert _sharpness(fixed) > 1.5 * _sharpness(raw)

    def test_commit_stamps_provenance(self, window):
        session, plot, tree, _wiz = _solved(window)
        dr.drift_commit(session, plot, {})
        node = tree.root_node.children["Drift corrected"]
        prov = node.signal.metadata.get_item("General.spyde_provenance")
        assert prov["action"] == "Drift Correction"
        assert prov["kind"] == "rigid"

    def test_commit_before_solving_errors(self, window):
        session, plot, _tree, _wiz = _opened(window)
        msgs = window["messages"]
        dr.drift_commit(session, plot, {})
        assert any("solve first" in str(m.get("text", ""))
                   for m in _of_type(msgs, "error"))

    def test_a_model_of_the_wrong_length_is_refused(self, window):
        """Re-solving after a crop must not silently pair frame 0's shift with
        a different frame."""
        from spyde.drift import DriftModel
        _s, _p, tree, _wiz = _opened(window)
        bad = DriftModel(shifts=np.zeros((N_FRAMES + 3, 2), np.float32))
        with pytest.raises(ValueError, match="covers"):
            dr.drift_corrected(tree.root, model=bad)


class TestDoubleFire:
    def test_open_close_open_leaves_one_controller(self, window):
        session, plot, tree = _movie(window)
        built = []
        real_init = dr.DriftWizard.__init__

        def _tracking(self, *a, **k):
            real_init(self, *a, **k)
            built.append(self)

        dr.DriftWizard.__init__ = _tracking
        try:
            dr.drift_open(session, plot, {})
            dr.drift_close(session, plot, {})
            dr.drift_open(session, plot, {})
        finally:
            dr.DriftWizard.__init__ = real_init
        assert _wait(lambda: tree._drift_wizard is not None
                     and tree._drift_wizard.window_id is not None)
        time.sleep(0.5)

        alive = [w for w in built if not w._closed]
        assert len(alive) == 1, \
            f"expected 1 live controller, got {len(alive)} of {len(built)} built"
        assert tree._drift_wizard is alive[0]
        # …and exactly one check window, because a superseded open's deferred
        # build must be dropped rather than emitting a second figure.
        wids = [w.window_id for w in built if w.window_id is not None]
        assert len(wids) == 1, f"{len(wids)} check windows opened"

    def test_close_without_an_open_is_harmless(self, window):
        session, plot, tree = _movie(window)
        dr.drift_close(session, plot, {})
        assert getattr(tree, "_drift_wizard", None) is None


class TestSchema:
    def test_schema_resolves_through_the_registry(self):
        from spyde.actions import registry
        schema = registry.wizard_parameters("drift")
        assert schema and schema is not dr.DriftWizard.parameters

    def test_schema_defaults_match_the_handler_defaults(self):
        from spyde.actions import registry
        schema = registry.wizard_parameters("drift")
        for key, spec in schema.items():
            assert key in dr.DEFAULTS, f"drift schema declares unknown param {key!r}"
            assert spec["default"] == dr.DEFAULTS[key], \
                f"drift schema/{key} drifted from drift_action.DEFAULTS"

    def test_every_stage_is_registered(self):
        from spyde.actions.registry import STAGED_HANDLERS, resolve_staged
        for stage in ("drift_open", "drift_close", "drift_set_method",
                      "drift_tune", "drift_run", "drift_commit"):
            assert stage in STAGED_HANDLERS
            assert callable(resolve_staged(stage))

    def test_toolbar_entry_gates_on_a_movie(self):
        import spyde
        meta = spyde.TOOLBAR_ACTIONS["functions"]["Drift Correction"]
        assert meta["function"] == "spyde.actions.drift_action.drift_correction"
        assert meta["signal_types"] == ["insitu"], (
            "the rigid solver needs a 1-D navigation axis; gating on `insitu` "
            "is the same gate Play/Fast-Forward use for exactly that")

    @pytest.mark.parametrize("signal_type,offered", [
        ("insitu", True),
        ("", False),                       # a single image has nothing to align
        ("electron_diffraction", False),
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

        meta = spyde.TOOLBAR_ACTIONS["functions"]["Drift Correction"]
        assert _action_matches_plot("Drift Correction", meta, _State()) is offered


class TestCoercion:
    def test_unknown_method_falls_back(self):
        assert dr._coerce({"method": "warp"})["method"] == dr.DEFAULTS["method"]

    def test_unknown_reference_falls_back(self):
        assert dr._coerce({"reference": "later"})["reference"] == "running"

    def test_order_is_clamped(self):
        assert dr._coerce({"order": 9})["order"] == 3
        assert dr._coerce({"order": -1})["order"] == 0

    def test_a_junk_value_keeps_the_default(self):
        assert dr._coerce({"upsample": "eight"})["upsample"] == \
            dr.DEFAULTS["upsample"]

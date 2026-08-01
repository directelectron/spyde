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
    def test_nonrigid_is_selectable_now_that_it_is_implemented(self, window):
        """Non-rigid used to be a stub that silently reverted to rigid.

        It is implemented (``spyde.drift.nonrigid``), so selecting it must
        STICK — a caret that quietly reverts would put the wrong ``kind`` in
        provenance, which is what the stub test was guarding against.
        """
        session, plot, _tree, wiz = _opened(window)
        dr.drift_set_method(session, plot, {"method": "nonrigid"})
        assert wiz.params["method"] == "nonrigid"
        assert "nonrigid" not in dr._UNAVAILABLE

    def test_the_nonrigid_field_parameterisation_is_selectable(self, window):
        """Both parameterisations describe different physics; neither is a default
        the user should be stuck with."""
        session, plot, _tree, wiz = _opened(window)
        dr.drift_set_method(session, plot, {"method": "nonrigid"})
        for name in dr.NONRIGID_MODELS:
            wiz.params = dr._coerce({**wiz.params, "nonrigid_model": name})
            assert wiz.params["nonrigid_model"] == name

    def test_an_unknown_field_falls_back_rather_than_raising(self, window):
        session, plot, _tree, wiz = _opened(window)
        wiz.params = dr._coerce({**wiz.params, "nonrigid_model": "banana"})
        assert wiz.params["nonrigid_model"] == dr.DEFAULTS["nonrigid_model"]

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


class _Box:
    """Stand-in for an anyplotlib RectangleWidget (x/y/w/h in IMAGE PIXELS).

    The headless session does build a real ``_plot2d``, so the wizard's own box
    exists — but a test that wants a SPECIFIC region needs to place one, and
    dragging a real widget means faking pointer events. Swapping this in is the
    smaller lie, and it exercises the same ``roi_box()`` conversion.
    """

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, float(v))

    def hide(self):
        pass


class TestDiscoveryPreview:
    """The centrepiece: a draggable box + a drift-corrected sum of it over ~20
    frames, so the user sees whether alignment works BEFORE paying for the whole
    movie."""

    def test_open_previews_the_default_box(self, window):
        session, _plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        assert _wait(lambda: _of_type(msgs, "drift_preview")), \
            "opening the caret never produced a discovery preview"
        prev = _of_type(msgs, "drift_preview")[-1]
        assert prev["frames"] >= 2
        assert prev["gain"] > 1.0, (
            "the aligned sum of the default box is not sharper than the raw one "
            "— the preview cannot discriminate anything if it never improves")

    def test_the_preview_never_computes_the_whole_movie(self, window):
        session, plot, tree, _wiz = _opened(window)
        msgs = window["messages"]
        del msgs[:]
        with _FullComputeGuard(tree.root.data.shape) as guard:
            dr.drift_tune(session, plot, {"upsample": 4, "max_shift": 12})
            assert _wait(lambda: _of_type(msgs, "drift_preview"))
        assert guard.hits == 0

    def test_tune_stores_the_new_parameters(self, window):
        session, plot, _tree, wiz = _opened(window)
        dr.drift_tune(session, plot, {"upsample": 16, "max_shift": 9.0})
        assert wiz.params["upsample"] == 16
        assert wiz.params["max_shift"] == 9.0

    def test_the_preview_uses_the_box_even_with_the_toggle_off(self, window):
        """The toggle is the COMMITMENT (does the full solve restrict to the
        box); the preview is the QUESTION and always asks it about the box."""
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        assert wiz.params["use_roi"] is False
        wiz._roi_widget = _Box(10, 12, 60, 48)
        del msgs[:]
        dr.drift_tune(session, plot, {})
        assert _wait(lambda: _of_type(msgs, "drift_preview"))
        assert _of_type(msgs, "drift_preview")[-1]["roi"] == [12, 10, 48, 60]

    def test_the_box_is_read_in_image_pixels_as_y0_x0_h_w(self, window):
        """anyplotlib 2-D widgets report IMAGE PIXELS and solve_translation's
        roi is in pixels — the two meet with NO scale conversion."""
        _s, _p, _t, wiz = _opened(window)
        wiz._frame_shape = (96, 112)
        wiz._roi_widget = _Box(x=20, y=8, w=40, h=32)
        assert wiz.roi_box() == (8, 20, 32, 40)

    def test_the_box_is_clamped_into_the_frame(self, window):
        _s, _p, _t, wiz = _opened(window)
        wiz._frame_shape = (96, 112)
        wiz._roi_widget = _Box(x=100, y=90, w=400, h=400)
        y0, x0, h, w = wiz.roi_box()
        assert 0 <= y0 and 0 <= x0
        assert y0 + h <= 96 and x0 + w <= 112

    def test_a_box_below_the_solver_floor_is_refused(self, window):
        """solve_translation REJECTS a too-small roi rather than clamping it, so
        the caret must never hand it one."""
        from spyde.drift.translation import _MIN_ROI
        assert dr._ROI_MIN_PX >= _MIN_ROI
        _s, _p, _t, wiz = _opened(window)
        wiz._frame_shape = (96, 112)
        wiz._roi_widget = _Box(x=0, y=0, w=4, h=4)
        y0, x0, h, w = wiz.roi_box()
        assert h >= _MIN_ROI and w >= _MIN_ROI

    def test_a_superseded_preview_does_not_paint(self, window):
        """Latest-wins: a drag that outruns the solve must drop the stale
        result, not paint it over the newer one."""
        _s, _p, tree, wiz = _opened(window)
        gen = dr.bump_generation(tree, "_drift_preview_gen")
        painted = []
        wiz.show_preview = lambda res: painted.append(res)
        dr.bump_generation(tree, "_drift_preview_gen")     # a newer drag lands
        assert not dr.is_current(tree, "_drift_preview_gen", gen)
        assert painted == []

    def test_the_settle_timer_coalesces_a_drag(self, window):
        """The widget's pointer_move fires at renderer frame rate; only the
        RESTING geometry may solve."""
        _s, _p, _t, wiz = _opened(window)
        fired = []
        wiz._fire_preview = lambda: fired.append(1)
        for _ in range(20):
            wiz.schedule_preview(delay=0.05)
        assert fired == []
        assert _wait(lambda: len(fired) >= 1, timeout=3.0)
        time.sleep(0.2)
        assert len(fired) == 1, f"{len(fired)} solves for one drag"


class TestSharpnessNumber:
    """The gain has to be an ANSWER, not decoration: a landmark and a
    featureless patch must come out clearly different."""

    @staticmethod
    def _stack(n=16, size=140, pad=20):
        """Textured on the left half, flat on the right, drifting rigidly."""
        from scipy.ndimage import gaussian_filter, map_coordinates
        rng = np.random.default_rng(3)
        canvas = np.zeros((size + 2 * pad, size + 2 * pad), np.float32) + 1.0
        tex = gaussian_filter(rng.standard_normal(canvas.shape), 1.5) * 0.6
        half = size // 2 + pad
        canvas[:, :half] += tex[:, :half]
        drift = np.stack([np.linspace(0, 8.0, n), np.linspace(0, -5.0, n)], 1)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        frames = np.empty((n, size, size), np.float32)
        for t in range(n):
            dy, dx = drift[t]
            frames[t] = map_coordinates(canvas, [yy + pad - dy, xx + pad - dx],
                                        order=1, mode="nearest")
        frames += rng.normal(0, 0.02, frames.shape).astype(np.float32)
        return frames

    def test_a_landmark_beats_a_featureless_patch(self):
        frames = self._stack()
        idx = np.arange(frames.shape[0])
        params = dict(dr.DEFAULTS)
        good = dr.preview_alignment(frames.__getitem__, idx, (30, 5, 80, 55),
                                    params=params)
        bad = dr.preview_alignment(frames.__getitem__, idx, (30, 82, 80, 55),
                                   params=params)
        assert good["gain"] > 2.0, f"a real landmark only scored {good['gain']:.2f}"
        assert bad["gain"] < 1.0, f"a featureless patch scored {bad['gain']:.2f}"
        assert good["gain"] > 3 * bad["gain"]

    def test_the_nan_border_does_not_inflate_the_number(self):
        """A shifted frame's uncovered edge is NaN (plan A7). Zero-filling it
        manufactures a step whose gradient energy dwarfs the image's own — every
        ROI would look brilliant."""
        a = np.ones((32, 32), np.float32)
        a[:4, :] = np.nan
        assert dr._gradient_energy(a) == 0.0

    def test_the_two_sums_are_measured_on_the_same_pixels(self):
        raw = np.ones((16, 16), np.float32)
        aligned = raw.copy()
        aligned[:3, :] = np.nan
        both = np.isfinite(raw) & np.isfinite(aligned)
        assert dr._gradient_energy(raw, both) == dr._gradient_energy(aligned, both)

    def test_the_preview_sample_spans_the_whole_movie(self):
        """20 CONSECUTIVE frames of a long movie drift by almost nothing, so a
        contiguous window would say "looks fine" for every box."""
        idx = dr._preview_indices(3000, 20, 64 * 64 * 4)
        assert idx[0] == 0 and idx[-1] == 2999 and idx.size <= 20

    def test_the_sample_is_thinned_to_fit_the_byte_cap(self):
        """With no ROI the crop IS the frame — 20 × 4096² float32 is 1.3 GB."""
        idx = dr._preview_indices(3000, 20, 4096 * 4096 * 4)
        assert 2 <= idx.size < 20


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

    def test_use_roi_feeds_the_box_to_the_solver(self, window):
        """The toggle's whole job: the same rectangle the preview tested is the
        one the full solve correlates on."""
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        wiz._frame_shape = (96, 112)
        wiz._roi_widget = _Box(x=16, y=12, w=64, h=64)
        dr.drift_run(session, plot, {"use_roi": True})
        assert _wait(lambda: wiz.model is not None)
        assert wiz.model.params["roi"] == [12, 16, 64, 64]
        assert _of_type(msgs, "drift_result")[-1]["roi"] == [12, 16, 64, 64]


class TestTraceWindow:
    """The dy/dx curve is its OWN plot window, filled as the solve runs — not
    caret furniture (plan §0.9a)."""

    def test_the_solve_opens_a_second_figure_window(self, window):
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        dr.drift_run(session, plot, {})
        assert _wait(lambda: wiz.trace_window_id is not None)
        figs = [m for m in _of_type(msgs, "figure")
                if m.get("window_id") == wiz.trace_window_id]
        assert figs and figs[-1]["title"] == "Drift dy/dx"
        assert session.controller_by_window_id(wiz.trace_window_id) is wiz, \
            "a bare figure is only reachable through the controller registry"
        assert session._plot_by_window_id(wiz.trace_window_id) is None

    def test_it_fills_from_the_on_shift_stream(self, window):
        session, plot, _tree, wiz = _opened(window)
        msgs = window["messages"]
        dr.drift_run(session, plot, {})
        assert _wait(lambda: wiz.model is not None)
        assert _wait(lambda: int(wiz._trace.get("filled", 0)) == N_FRAMES)
        # …and the same batches went out as `drift_trace` messages.
        streamed = sum(len(m["points"]) for m in _of_type(msgs, "drift_trace"))
        assert streamed == N_FRAMES

    def test_the_trace_matches_the_model(self, window):
        _s, _p, _t, wiz = _solved(window)
        assert _wait(lambda: int(wiz._trace.get("filled", 0)) == N_FRAMES)
        np.testing.assert_allclose(wiz._trace["dy_data"], wiz.model.shifts[:, 0],
                                   atol=1e-5)
        np.testing.assert_allclose(wiz._trace["dx_data"], wiz.model.shifts[:, 1],
                                   atol=1e-5)

    def test_only_the_solved_prefix_is_pushed(self, window):
        """Pushing the NaN-padded whole array would leave anyplotlib's auto
        y-range looking at one finite point."""
        _s, _p, _t, wiz = _opened(window)
        wiz.trace_window_id = None
        wiz.open_trace_window(50)
        wiz.push_trace([(1, 3.0, -2.0), (2, 4.0, -3.0)])
        assert wiz._trace["filled"] == 3
        assert np.isnan(wiz._trace["dy_data"][3:]).all()

    def test_close_takes_both_windows(self, window):
        session, plot, tree, wiz = _solved(window)
        assert _wait(lambda: wiz.trace_window_id is not None)
        check, trace = wiz.window_id, wiz.trace_window_id
        dr.drift_close(session, plot, {})
        from spyde.actions.figure_registry import _FIGS
        for wid in (check, trace):
            assert session.controller_by_window_id(wid) is None
            assert wid not in _FIGS, f"figure {wid} outlived its window"


class TestDiscard:
    def test_discard_drops_the_model_and_the_trace_window(self, window):
        session, plot, tree, wiz = _solved(window)
        assert _wait(lambda: wiz.trace_window_id is not None)
        trace = wiz.trace_window_id
        dr.drift_discard(session, plot, {})
        assert wiz.model is None
        assert getattr(tree, "drift", None) is None
        assert wiz.trace_window_id is None
        assert session.controller_by_window_id(trace) is None
        assert wiz.window_id is not None, "Discard must not close the caret"

    def test_discard_stops_a_solve_in_flight(self, window):
        """Same user intent as Stop, so it is the same handler: bumping the run
        generation FIRST means a solve that finishes anyway never installs."""
        session, plot, tree, wiz = _opened(window, frames=24)
        dr.drift_run(session, plot, {})
        dr.drift_discard(session, plot, {})
        assert wiz._stop[0] is True
        time.sleep(1.0)
        assert wiz.model is None


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
                      "drift_tune", "drift_run", "drift_discard",
                      "drift_commit"):
            assert stage in STAGED_HANDLERS
            assert callable(resolve_staged(stage))

    def test_the_default_face_is_two_toggles(self):
        """§0.9a: everything that is not the task itself is tagged Advanced, so
        any host renders the same small face. The caret is the enforcement; this
        is the schema saying the same thing."""
        from spyde.actions import registry
        schema = registry.wizard_parameters("drift")
        face = [k for k, s in schema.items() if not s.get("tab")]
        assert face == ["use_roi", "reject_outliers"], \
            f"the caret's default face grew to {face}"

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

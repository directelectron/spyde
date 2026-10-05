"""
The Segment wizard backend (``seg_*`` staged handlers), on a real Session.

Handlers are called directly as ``fn(session, plot, payload)`` and polled with
``wait_until`` — training, previews and the run all land from worker threads.
Strokes go in through the real anyplotlib brush widget the wizard attaches
(``add_stroke`` is its Python door), so the stroke → label → retrain seam is
the one the app uses.

The three claims that matter, one per dataset shape:

* a MOVIE: the run labels every frame lazily, never computing the stack, and
  opens a result tree whose navigator is the count per frame;
* an IMAGE: the run commits one label map;
* a 4-D SCAN: opened on the navigator, the pixels are scan positions.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions import segment_action as sa
from spyde.data.synthetic import ground_truth, particle_truth_at
from spyde.segmentation import BACKGROUND, PARTICLE
from spyde.tests.migrated._async import wait_until


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    """torch-CUDA work segfaults under pytest on Windows; the wiring is device-agnostic."""
    monkeypatch.setenv(sa.DEVICE_VARIABLE, "cpu")


@pytest.fixture
def _capture_module_emit(window, monkeypatch):
    monkeypatch.setattr(sa, "emit", window["messages"].append)


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _navigator_plot(session):
    return next((p for p in session._plots if p.is_navigator), None)


def _of_type(messages, kind):
    return [m for m in messages if isinstance(m, dict) and m.get("type") == kind]


def _open(session, plot):
    sa.seg_open(session, plot, {})
    wizard = plot.signal_tree._seg_wizard
    assert wizard is not None and wizard._brush is not None, "no brush on the plot"
    return wizard


def _stroke(wizard, points_yx, class_id, radius=2.0):
    """Paint through the real brush: the widget's Python door, then the event."""
    wizard.params["active_class"], wizard.params["radius"] = class_id, radius
    wizard._brush.add_stroke([[x, y] for y, x in points_yx], class_id)
    wizard._on_stroke()


def _paint_movie(wizard, truth):
    positions, radii, _present = particle_truth_at(truth, 0)
    for index in (0, 1, 8):
        y, x = positions[index]
        _stroke(wizard, [[y, x - 1], [y, x + 1]], PARTICLE, radius=max(1.0, radii[index] * 0.4))
    _stroke(wizard, [[4, 4], [4, 100]], BACKGROUND)
    _stroke(wizard, [[50, 55], [60, 55]], BACKGROUND)


def _movie(window, frames=6):
    session = window["window"]
    session._load_test_data_particles({"frames": frames})
    assert wait_until(lambda: _signal_plot(session) is not None)
    plot = _signal_plot(session)
    return session, plot, plot.signal_tree


class _FullComputeGuard:
    """Count ``.compute()`` calls made on the whole movie."""

    def __init__(self, shape):
        self.shape, self.hits = tuple(shape), 0

    def __enter__(self):
        import dask.array as da
        self._real = da.Array.compute
        guard = self

        def _spy(array, *args, **kwargs):
            if tuple(array.shape) == guard.shape:
                guard.hits += 1
            return guard._real(array, *args, **kwargs)
        da.Array.compute = _spy
        return self

    def __exit__(self, *exc):
        import dask.array as da
        da.Array.compute = self._real
        return False


@pytest.mark.usefixtures("_capture_module_emit")
class TestOpenAndClose:
    def test_open_puts_a_brush_on_the_plot_and_reports_the_classes(self, window):
        session, plot, tree = _movie(window)
        wizard = _open(session, plot)
        state = _of_type(window["messages"], "seg_state")[-1]
        assert state["window_id"] == plot.window_id
        assert [c["name"] for c in state["classes"]] == ["particle", "background", "boundary"]
        assert state["n_fields"] == 6 and state["space"] == "signal" and not state["trained"]
        assert wizard.source.shape == (96, 112)

    def test_close_removes_the_brush_and_the_mask(self, window):
        session, plot, tree = _movie(window)
        wizard = _open(session, plot)
        cleared = []
        plot.set_overlay_mask = lambda mask, **kw: cleared.append(mask)
        sa.seg_close(session, plot, {})
        assert wizard._closed and wizard._brush is None
        assert tree._seg_wizard is None and cleared == [None]

    def test_open_close_open_leaves_one_controller(self, window):
        session, plot, tree = _movie(window)
        first = _open(session, plot)
        sa.seg_close(session, plot, {})
        second = _open(session, plot)
        assert second is not first and tree._seg_wizard is second
        assert first._brush is None and second._brush is not None

    def test_a_diffraction_pattern_window_is_refused(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        plot = _signal_plot(session)
        sa.seg_open(session, plot, {})
        assert getattr(plot.signal_tree, "_seg_wizard", None) is None
        assert any("navigator" in str(m.get("text", "")).lower()
                   for m in _of_type(stem_4d_dataset["messages"], "error"))


@pytest.mark.usefixtures("_capture_module_emit")
class TestPaintTrainPreview:
    def test_strokes_land_in_the_store_with_their_class(self, window):
        session, plot, _tree = _movie(window)
        wizard = _open(session, plot)
        _stroke(wizard, [[24, 20], [24, 24]], PARTICLE)
        _stroke(wizard, [[4, 4], [4, 60]], BACKGROUND)
        counts = wizard.labels.counts()
        assert counts[PARTICLE] > 0 and counts[BACKGROUND] > counts[PARTICLE]
        state = _of_type(window["messages"], "seg_state")[-1]
        assert {c["name"]: c["pixels"] for c in state["classes"]}["boundary"] == 0

    def test_train_previews_the_mask_and_new_strokes_retrain(self, window):
        session, plot, tree = _movie(window)
        truth = ground_truth(tree.root)
        wizard = _open(session, plot)
        _paint_movie(wizard, truth)
        masks = []
        plot.set_overlay_mask = lambda mask, **kw: masks.append(mask)
        sa.seg_train(session, plot, {})
        assert wait_until(lambda: wizard.classifier is not None and masks, timeout=60)
        first = wizard.classifier
        assert masks[-1].shape == (96, 112) and 0 < masks[-1].mean() < 0.3
        assert _of_type(window["messages"], "seg_state")[-1]["trained"]
        _stroke(wizard, [[70, 60], [70, 64]], BACKGROUND)
        assert wait_until(lambda: wizard.classifier is not first, timeout=60)

    def test_clear_forgets_the_strokes_and_the_classifier(self, window):
        session, plot, tree = _movie(window)
        wizard = _open(session, plot)
        _stroke(wizard, [[24, 20], [24, 24]], PARTICLE)
        sa.seg_clear(session, plot, {})
        assert len(wizard.labels) == 0 and wizard.classifier is None
        assert wizard._brush.n_strokes == 0


@pytest.mark.usefixtures("_capture_module_emit")
class TestRun:
    def _trained(self, window, frames=12):
        session, plot, tree = _movie(window, frames)
        wizard = _open(session, plot)
        _paint_movie(wizard, ground_truth(tree.root))
        sa.seg_train(session, plot, {})
        assert wait_until(lambda: wizard.classifier is not None, timeout=60)
        return session, plot, tree, wizard

    def test_a_movie_run_opens_a_lazy_label_movie_with_a_count_navigator(self, window):
        session, plot, tree, wizard = self._trained(window)
        truth = ground_truth(tree.root)
        before = len(session.signal_trees)
        # A classifier trained on frame 0's strokes alone picks up a few film
        # specks on the drifted later frames; a 30 px floor keeps the faint
        # probes (~30-90 px once smoothed) and drops the specks. Measured on
        # this fixture: every frame within one of the truth.
        with _FullComputeGuard(tree.root.data.shape) as guard:
            sa.seg_run(session, plot, {"min_size": 30})
            assert wait_until(lambda: wizard.result_tree is not None, timeout=120)
        assert guard.hits == 0, "the run computed the whole movie at once"
        result = wizard.result_tree
        assert len(session.signal_trees) == before + 1
        regions = result.regions
        assert regions.n_fields == 12 and regions.space == "signal" and regions.units == "nm"
        counts = regions.count_series()
        for frame in range(12):
            _pos, _radii, present = particle_truth_at(truth, frame)
            assert abs(int(counts[frame]) - int(present.sum())) <= 1, \
                f"frame {frame}: {int(counts[frame])} regions for {int(present.sum())} particles"
        assert result.root._lazy and result.root._signal_type == "regions"
        assert result.root.data.shape == tree.root.data.shape
        frame = np.asarray(result.root.data[2].compute())
        assert frame.dtype == np.int32 and frame.max() == counts[2]
        assert result.regions.provenance["labels"]["fields"], "the strokes travel with the result"
        done = _of_type(window["messages"], "seg_result")[-1]
        assert done["n_regions"] == regions.n_regions and not done["cancelled"]

    def test_a_single_image_run_commits_one_label_map(self, window):
        session, plot, tree, wizard = self._trained(window)
        image = tree.root.inav[0]
        image.metadata.General.title = "one frame"
        session._add_signal(image)
        assert wait_until(lambda: len(session.signal_trees) == 2)
        image_plot = next(p for p in session._plots
                          if not p.is_navigator and p.plot_state is not None
                          and p.signal_tree is session.signal_trees[-1])
        image_wizard = _open(session, image_plot)
        image_wizard.classifier = wizard.classifier          # same field shape
        assert not image_wizard.source.is_movie
        sa.seg_run(session, image_plot, {"min_size": 30})
        assert wait_until(lambda: image_wizard.result_tree is not None, timeout=60)
        result = image_wizard.result_tree
        assert result.regions.n_fields == 1 and 7 <= result.regions.n_regions <= 9
        assert result.root.data.shape == (96, 112) and not result.root._lazy
        assert result.root._signal_type == "regions"

    def test_the_run_can_be_stopped(self, window):
        session, plot, tree, wizard = self._trained(window)
        sa.seg_run(session, plot, {})
        sa.seg_stop(session, plot, {})
        assert wait_until(lambda: not wizard.running, timeout=60)
        assert wait_until(lambda: wizard.result_tree is not None, timeout=60)
        done = _of_type(window["messages"], "seg_result")[-1]
        assert done["cancelled"] or done["n_fields"] == 12


def _scan_with_a_grain(nav=(48, 56), sig=(8, 8)):
    """A 4-D scan whose navigator shows one bright grain: every scan position
    inside a disc carries a brighter pattern."""
    yy, xx = np.mgrid[0:nav[0], 0:nav[1]]
    inside = (yy - 24) ** 2 + (xx - 30) ** 2 <= 9 ** 2
    pattern = np.ones(sig, np.float32)
    data = np.ones(nav + sig, np.float32) * pattern
    data[inside] *= 4.0
    rng = np.random.default_rng(0)
    data += rng.random(data.shape, dtype=np.float32) * 0.2
    return data, inside


@pytest.mark.usefixtures("_capture_module_emit")
class TestScanNavigator:
    def test_the_navigator_pixels_are_scan_positions(self, window):
        import hyperspy.api as hs
        from spyde.tests.migrated.conftest import _settle
        session = window["window"]
        data, inside = _scan_with_a_grain()
        scan = hs.signals.Signal2D(data)
        scan.set_signal_type("electron_diffraction")
        for axis in scan.axes_manager.navigation_axes:
            axis.scale, axis.units = 2.0, "nm"
        session._add_signal(scan)
        _settle(session)
        navigator = _navigator_plot(session)
        assert wait_until(lambda: isinstance(navigator.current_data, np.ndarray)
                          and np.isfinite(navigator.current_data).all(), timeout=30)
        wizard = _open(session, navigator)
        assert wizard.space == "navigation" and wizard.source.shape == (48, 56)
        assert (wizard.scale, wizard.units) == (2.0, "nm")
        _stroke(wizard, [[24, 26], [24, 34]], PARTICLE)
        _stroke(wizard, [[4, 4], [4, 50]], BACKGROUND)
        _stroke(wizard, [[44, 4], [44, 50]], BACKGROUND)
        sa.seg_train(session, navigator, {})
        assert wait_until(lambda: wizard.classifier is not None, timeout=60)
        sa.seg_run(session, navigator, {"min_size": 10})
        assert wait_until(lambda: wizard.result_tree is not None, timeout=60)
        regions = wizard.result_tree.regions
        assert regions.space == "navigation" and regions.n_regions == 1
        table = regions.table_at(0)
        assert abs(table["y"][0] / 2.0 - 24) < 1.5 and abs(table["x"][0] / 2.0 - 30) < 1.5
        labels = np.asarray(wizard.result_tree.root.data)
        overlap = (labels > 0) & inside
        assert overlap.sum() > 0.8 * inside.sum(), "the region is the grain's scan positions"

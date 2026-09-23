"""`To Frames` end to end: a CSB opened through a real Session, re-cut (#91).

The synthetic stream is 100 raw frames of 1 ms. Opening it takes the reader's
default exposure — the movie split into 50 planes, i.e. 2 frames each — and
the re-cut asks for 3.5 ms, which is NOT a whole number of load planes, nor
even of raw frames. That is the case a boundary error hides in: at a whole
number of frames every plane is the same width and an off-by-one only moves
the last one.

The expectation never touches the reader. The documented rule (`_time_bounds`)
is that a plane holds the frames whose START TIME falls in its window, so frame
``f`` (starting at ``f`` ms) belongs to plane ``floor(f / 3.5) = (2 f) // 7`` —
integer arithmetic, no float edges. Each plane's image is then the plain-numpy
sum of its frames' events (`_csb_synthetic.expected_image`).
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.tests.migrated._async import quiesce, wait_until
from spyde.tests.migrated._csb_synthetic import (
    BH, BW, H, W, expected_image, grid, write_csb,
)

FRAMES = 100
US_PER_FRAME = 1000.0
EXPOSURE_MS = 3.5
LOAD_FRAMES_PER_PLANE = 2             # duration / DEFAULT_PLANES = 100 / 50


def _plane_of(frame: int) -> int:
    """The plane a raw frame lands in at a 3.5 ms exposure: floor(f / 3.5)."""
    return (2 * frame) // 7


N_PLANES = _plane_of(FRAMES - 1) + 1  # 29


def _expected_bounds():
    """[f0, f1) of every plane, from `_plane_of` alone."""
    frames_by_plane: dict[int, list[int]] = {}
    for f in range(FRAMES):
        frames_by_plane.setdefault(_plane_of(f), []).append(f)
    return [(fs[0], fs[-1] + 1) for _, fs in sorted(frames_by_plane.items())]


def _random_events(seed=0):
    """0-3 events in every (frame, block), so every frame carries a
    distinguishable pattern and moving any boundary by one frame shows."""
    rng = np.random.default_rng(seed)
    blocks_per_frame = int(np.prod(grid(W, H, BW, BH)))
    events = {}
    for f in range(FRAMES):
        for b in range(blocks_per_frame):
            n = int(rng.integers(0, 4))
            if n:
                events[(f, b)] = rng.integers(0, BW * BH, n).tolist()
    return events


EVENTS = _random_events()


@pytest.fixture
def cpu_only(monkeypatch):
    """Integrate on the CPU: torch-CUDA can segfault under pytest on Windows,
    and this file tests the action's wiring, not the GPU accumulator."""
    from spyde.external.rsciio_csb import _core, _torch
    monkeypatch.setattr(_torch, "torch_gpu_available", lambda: False)
    monkeypatch.setattr(_core, "gpu_available", lambda: False)


@pytest.fixture
def opened_csb(window, tmp_path, cpu_only):
    """The synthetic stream opened the way File > Open opens it."""
    session = window["window"]
    path = write_csb(tmp_path, EVENTS, name="movie.csb", frames=FRAMES,
                     us_per_frame=US_PER_FRAME)
    session._load_file_thread(path)
    assert len(session.signal_trees) == 1, "the CSB did not open"
    assert quiesce(session)
    return session, session.signal_trees[0]


def _signal_plot(session, tree):
    return next(p for p in session._plots
                if not p.is_navigator and getattr(p, "signal_tree", None) is tree)


def _to_frames(session, tree, **params):
    """Click To Frames on *tree*'s signal plot and wait for the new tree."""
    from spyde.actions.context import ActionContext
    from spyde.actions.csb_to_frames import csb_to_frames
    before = len(session.signal_trees)
    csb_to_frames(ActionContext(plot=_signal_plot(session, tree), params=params,
                                action_name="To Frames"), **params)
    assert wait_until(lambda: len(session.signal_trees) > before), \
        "To Frames never added a dataset"
    assert quiesce(session)
    return session.signal_trees[-1]


@pytest.fixture
def recut(opened_csb):
    session, source = opened_csb
    tree = _to_frames(session, source, exposure_ms=EXPOSURE_MS, bin=1,
                      backend="cpu-numpy")
    return session, source, tree


class TestTheSourceLoad:
    def test_the_load_exposure_is_two_frames(self, opened_csb):
        """Pins the premise: 3.5 ms is not a multiple of what was loaded."""
        _, source = opened_csb
        per_plane = source.root.original_metadata.csb.frames_per_plane
        assert set(per_plane) == {LOAD_FRAMES_PER_PLANE}
        assert (EXPOSURE_MS * 1e3 / US_PER_FRAME) % LOAD_FRAMES_PER_PLANE != 0


class TestRecutPlanes:
    def test_plane_count(self, recut):
        _, _, tree = recut
        assert tree.root.axes_manager.navigation_shape == (N_PLANES,)
        assert tree.root.data.shape == (N_PLANES, H, W)

    def test_the_planes_are_not_all_the_same_width(self, recut):
        """3.5 frames per plane alternates 4 and 3 (and the movie ends two
        frames into the last one) — a fixture that gave every plane the same
        width could not see a boundary shifted by one."""
        widths = [f1 - f0 for f0, f1 in _expected_bounds()]
        assert set(widths[:-1]) == {3, 4} and widths[-1] == 2

    def test_each_plane_is_the_sum_of_the_events_in_its_window(self, recut):
        _, _, tree = recut
        got = np.asarray(tree.root.data.compute())
        expected = np.zeros((N_PLANES, H, W), np.int64)
        for f in range(FRAMES):
            expected[_plane_of(f)] += expected_image(EVENTS, f, f + 1)
        assert np.array_equal(got, expected)

    def test_every_event_is_counted_exactly_once(self, recut):
        _, _, tree = recut
        total = sum(len(words) for words in EVENTS.values())
        assert float(tree.root.data.sum().compute()) == total

    def test_recorded_frame_bounds(self, recut):
        """original_metadata says which frames each plane integrated — without
        it a plane's intensity is uninterpretable."""
        _, _, tree = recut
        csb = tree.root.original_metadata.csb
        bounds = _expected_bounds()
        assert [tuple(b) for b in csb.plane_frame_bounds] == bounds
        assert list(csb.frames_per_plane) == [f1 - f0 for f0, f1 in bounds]
        assert csb.exposure_s == pytest.approx(EXPOSURE_MS * 1e-3)
        assert csb.bin == 1


class TestRecutIsAnOrdinaryMovie:
    def test_signal_type_is_insitu(self, recut):
        _, _, tree = recut
        assert tree.root.metadata.Signal.signal_type == "insitu"

    def test_time_axis_is_named_and_in_seconds(self, recut):
        _, _, tree = recut
        (axis,) = tree.root.axes_manager.navigation_axes
        assert axis.name == "time"
        assert axis.units == "s"
        assert axis.offset == pytest.approx(0.0)

    def test_each_plane_sits_at_its_own_start_time(self, recut):
        """A plane's time coordinate is within one raw frame of when its first
        frame started. The planes are 3 or 4 frames wide, so a uniform axis
        spaced by the FIRST gap (4 ms) drifts half a frame per plane and puts
        the last plane at 112 ms of a 100 ms movie."""
        _, _, tree = recut
        (axis,) = tree.root.axes_manager.navigation_axes
        frame_s = US_PER_FRAME * 1e-6
        starts = np.array([f0 for f0, _ in _expected_bounds()]) * frame_s
        coords = axis.offset + axis.scale * np.arange(N_PLANES)
        assert np.all(np.abs(coords - starts) < frame_s), (coords - starts)

    def test_signal_axes_are_calibrated(self, recut):
        _, _, tree = recut
        for axis in tree.root.axes_manager.signal_axes:
            assert axis.units == "Å"
            assert axis.scale == pytest.approx(0.025)

    def test_lazy_one_plane_per_block(self, recut):
        """Nothing is integrated until a plane is looked at, and a plane reads
        only its own window."""
        _, _, tree = recut
        assert tree.root._lazy
        assert set(tree.root.data.chunks[0]) == {1}


class TestRecutIsItsOwnDataset:
    def test_a_new_tree_beside_the_source(self, recut):
        session, source, tree = recut
        assert tree is not source
        assert session.signal_trees == [source, tree]
        assert source.root.axes_manager.navigation_shape == (FRAMES // 2,)

    def test_title_carries_the_exposure(self, recut):
        _, source, tree = recut
        base = source.root.metadata.General.title
        assert tree.root.metadata.General.title == f"{base} @ 3.5 ms"

    def test_navigator_is_the_event_count_per_plane(self, recut):
        """The free navigator: per-plane totals from the block table, which
        must equal the sum of each plane's independently computed image."""
        _, _, tree = recut
        navigator = tree.navigator_signals["base"]
        got = np.asarray(getattr(navigator, "data", navigator)).ravel()
        expected = np.zeros(N_PLANES)
        for (f, _block), words in EVENTS.items():
            expected[_plane_of(f)] += len(words)
        assert np.array_equal(got, expected)


class TestExposureKnobs:
    """fps, exposure and frame count are three ways of saying one thing."""

    def test_fps_matches_the_same_exposure(self, opened_csb):
        session, source = opened_csb
        tree = _to_frames(session, source, fps=1e3 / EXPOSURE_MS, bin=1,
                          backend="cpu-numpy")
        bounds = [tuple(b) for b in tree.root.original_metadata.csb.plane_frame_bounds]
        assert bounds == _expected_bounds()

    def test_frames_per_plane(self, opened_csb):
        session, source = opened_csb
        tree = _to_frames(session, source, frames_per_plane=3, bin=1,
                          backend="cpu-numpy")
        bounds = [tuple(b) for b in tree.root.original_metadata.csb.plane_frame_bounds]
        assert bounds == [(f, min(f + 3, FRAMES)) for f in range(0, FRAMES, 3)]


class TestRefusals:
    def test_a_non_csb_signal_is_refused(self, stem_4d_dataset):
        from spyde.actions.context import ActionContext
        from spyde.actions.csb_to_frames import csb_to_frames
        session = stem_4d_dataset["window"]
        plot = next(p for p in session._plots if not p.is_navigator)
        csb_to_frames(ActionContext(plot=plot, params={}, action_name="To Frames"),
                      fps=100.0)
        assert quiesce(session)
        assert len(session.signal_trees) == 1
        assert any(m.get("type") == "error" and "CSB" in str(m.get("text", ""))
                   for m in stem_4d_dataset["messages"] if isinstance(m, dict))

    def test_no_exposure_is_refused(self, opened_csb, window):
        session, source = opened_csb
        from spyde.actions.context import ActionContext
        from spyde.actions.csb_to_frames import csb_to_frames
        csb_to_frames(ActionContext(plot=_signal_plot(session, source), params={},
                                    action_name="To Frames"), backend="cpu-numpy")
        assert quiesce(session)
        assert len(session.signal_trees) == 1
        assert any(m.get("type") == "error" and "To Frames failed" in str(m.get("text", ""))
                   for m in window["messages"] if isinstance(m, dict))

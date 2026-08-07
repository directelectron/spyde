"""
Tests for ``spyde.particles.batch`` — the whole-movie segmentation fan-out.

The contract this file exists to pin, in the order it matters:

**1. A frame's particles must not change because it was computed in parallel.**
Everything else here is throughput, and throughput that alters the answer is
worthless. Asserted with ``array_equal`` against the serial loop, never
``allclose``. The reference is the SAME resolved engine run one frame at a time
on this thread, so the only thing that differs between it and the run is the
dispatch, and exact equality is the honest bar. (Across DEVICES it would not be:
a CUDA frame and a CPU frame differ in the last bits of a float32 convolution,
which is why the GPU/CPU agreement gate lives in ``test_particles_gpu.py``, in a
subprocess.)

**2. A cancelled run still spans the movie.** The CSR store is built from these
lists, so a short list is a shorter movie, not a stopped one.

**3. The frame index survives the dispatch.** A task cannot know its own global
offset — ``dispatch_chunks`` slices the result array per chunk, so a
``map_blocks`` stage reading ``block_info`` reports (0, 0) for every one of them
(the bug ``orchestrate`` records for the live count map). The index is stamped
where the global slice is authoritative, and a multi-frame block is where a
mistake would show.

**4. The lane policy is ONE value.** ``gpu_runtime`` records a real bug of this
class: the docstring said "2" while the code passed "4", so the client sized a
lane for one policy while the workers gated on another. The per-worker gate and
the client-side split must be given the same default.

**5. No rechunk shuffle, ever.** A movie whose reader split the signal axes must
fall back rather than shuffle multiple GB through the scheduler (CLAUDE.md
Live-Display §1).

Everything runs on the LOCAL thread-pool fallback (no cluster, no CUDA):
``SPYDE_NO_DASK=1`` is the migrated-test mode and torch-CUDA segfaults under
pytest on Windows.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.particles import measure_frame
from spyde.particles.batch import (
    EngineSpec,
    PARTICLE_GPU_LANE_DEFAULT,
    frames_per_task,
    resolve_engine,
    segment_block,
    segment_movie,
)
from spyde.signals.particles import COL, N_COLUMNS

PARAMS = dict(min_size=10)


def _disc(shape, cy, cx, r):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)


def _movie(n=7, h=90, w=110):
    """A short movie whose particle count CHANGES frame to frame.

    Deliberately not a repeated frame: identical frames would let a dispatcher
    that mixed up block offsets still pass every equality check here.
    """
    out = np.zeros((n, h, w), np.float32)
    for t in range(n):
        img = np.full((h, w), 0.05, np.float32)
        img += _disc((h, w), 25, 30, 8)
        img += _disc((h, w), 60, 70, 6 + (t % 3))
        if t % 2 == 0:
            img += _disc((h, w), 25, 85, 5)
        out[t] = np.clip(img, 0, 1.5)
    return out


def _serial(data, spec, scale=1.0):
    """The reference: the SAME engine, one frame at a time on this thread.

    It used to call the classical ``segment_frame`` directly. Comparing against
    the dispatcher's own resolved engine is both what survives that engine's
    removal and a sharper statement of contract 1 — the answer may not depend on
    how it was computed, so the only thing allowed to differ between reference
    and run is the dispatch.
    """
    engine, _dev = resolve_engine(spec, force_cpu=True)
    rows, contours = [], []
    for t in range(data.shape[0]):
        labels = engine(data[t])
        r, c = measure_frame(labels, data[t], t=t, scale=scale)
        rows.append(r)
        contours.append(c)
    return rows, contours


@pytest.fixture(scope="module")
def movie():
    return _movie()


@pytest.fixture(scope="module")
def spec(movie, tmp_path_factory):
    """A REAL trained scribble spec — the only engine there is.

    Trained once for the module on frame 0's discs: the disc interiors are the
    particle class, the corners are background. The head crosses to the workers
    as a FILE (see ``EngineSpec``), so this is also the shape a real run has.

    CPU explicitly: torch-CUDA work segfaults under the pytest process on
    Windows (CLAUDE.md).
    """
    from spyde.particles import (FeatureSpec, LabelStore, ScribbleClassifier,
                                 default_classes, select_device)

    frame = movie[0]
    h, w = frame.shape
    store = LabelStore(frame_shape=(h, w), classes=default_classes())
    for cy, cx, r in ((25, 30, 8), (60, 70, 6), (25, 85, 5)):
        store.paint_disc(0, cy, cx, max(1.5, r * 0.5), 0)
    background = np.zeros((h, w), bool)
    background[:10, :] = True
    background[-10:, :] = True
    background[:, :10] = True
    store.paint(0, background, 1)

    clf = ScribbleClassifier(FeatureSpec(), device=select_device("cpu"), seed=0)
    clf.fit(store, {0: frame})
    path = str(tmp_path_factory.mktemp("seg-batch") / "head.npz")
    clf.save(path)
    return EngineSpec(method="scribble", params=dict(PARAMS), model_path=path)


@pytest.fixture(scope="module")
def reference(movie, spec):
    return _serial(movie, spec)


class TestParallelIsBitIdentical:
    """Contract 1 — the answer may not depend on how it was computed."""

    def _assert_same(self, got, reference, label):
        rows, contours, done = got
        ref_rows, ref_contours = reference
        assert done == len(ref_rows), label
        assert len(rows) == len(ref_rows) == len(contours), label
        for t, (r, ref) in enumerate(zip(rows, ref_rows)):
            assert np.array_equal(r, ref), f"{label}: frame {t} rows differ"
        for t, (c, ref) in enumerate(zip(contours, ref_contours)):
            assert len(c) == len(ref), f"{label}: frame {t} outline count"
            for a, b in zip(c, ref):
                assert np.array_equal(a, b), f"{label}: frame {t} outline"

    def test_numpy_source(self, movie, reference, spec):
        got = segment_movie(movie, spec, n_frames=len(movie), client=None)
        self._assert_same(got, reference, "numpy")

    def test_lazy_source(self, movie, reference, spec):
        import dask.array as da
        lazy = da.from_array(movie, chunks=(2, -1, -1))
        got = segment_movie(lazy, spec, n_frames=len(movie), client=None)
        self._assert_same(got, reference, "dask")

    def test_get_frame_source(self, movie, reference, spec):
        """No array at all — a callable frame source still has to agree."""
        got = segment_movie(None, spec, n_frames=len(movie),
                            get_frame=lambda t: movie[t], client=None)
        self._assert_same(got, reference, "get_frame")

    def test_multi_frame_blocks_agree(self, movie, reference, spec,
                                      monkeypatch):
        """Several frames per task is the case where a block offset can be
        wrong without any single-frame test noticing."""
        monkeypatch.setenv("SPYDE_SEG_TASK_BYTES", str(1 << 30))
        got = segment_movie(movie, spec, n_frames=len(movie), client=None)
        self._assert_same(got, reference, "multi-frame blocks")

    def test_scale_is_applied_once(self, movie, spec):
        """A calibrated axis must not be applied twice by the parallel path."""
        rows, _c, _d = segment_movie(movie, spec, n_frames=len(movie),
                                     client=None, scale=2.5)
        ref, _ = _serial(movie, spec, scale=2.5)
        for t, (r, e) in enumerate(zip(rows, ref)):
            assert np.array_equal(r, e), f"frame {t}"


class TestFrameIndex:
    """Contract 3 — the ``t`` column, which no task can know for itself."""

    def test_every_row_carries_its_own_frame(self, movie, spec):
        rows, _c, _d = segment_movie(movie, spec, n_frames=len(movie),
                                     client=None)
        for t, r in enumerate(rows):
            assert len(r), f"frame {t} found nothing — the fixture is wrong"
            assert np.all(r[:, COL["t"]] == float(t)), (
                f"frame {t} rows are stamped {set(r[:, COL['t']].tolist())}")

    def test_multi_frame_block_indexes_within_itself(self, movie, spec,
                                                    monkeypatch):
        monkeypatch.setenv("SPYDE_SEG_TASK_BYTES", str(1 << 30))
        rows, _c, _d = segment_movie(movie, spec, n_frames=len(movie),
                                     client=None)
        for t, r in enumerate(rows):
            assert np.all(r[:, COL["t"]] == float(t)), f"frame {t}"

    def test_segment_block_stamps_from_its_offset(self, movie, spec):
        """Used directly (the serial/threaded paths), the block DOES stamp."""
        out = segment_block(movie[2:5], 2, spec)
        assert len(out) == 3
        for i, (rows, _c) in enumerate(out):
            assert np.all(rows[:, COL["t"]] == float(2 + i))


class TestProgress:
    """The progressive fill: blocks land out of order, so the callback has to
    carry the global offset rather than a running counter."""

    def test_on_frames_covers_every_frame_exactly_once(self, movie, spec):
        seen: list[int] = []

        def _on(t0, t1, vals):
            assert t1 - t0 == len(vals)
            seen.extend(range(t0, t1))

        segment_movie(movie, spec, n_frames=len(movie), client=None,
                      on_frames=_on)
        assert sorted(seen) == list(range(len(movie)))

    def test_a_failing_callback_does_not_fail_the_run(self, movie, spec):
        def _boom(t0, t1, vals):
            raise RuntimeError("the caret went away")

        rows, _c, done = segment_movie(movie, spec, n_frames=len(movie),
                                       client=None, on_frames=_boom)
        assert done == len(movie)

    def test_the_callback_sees_the_stamped_rows(self, movie, spec):
        """The live label movie renders from these, so they must already carry
        the true frame index — not the 0 the task stamped."""
        stamps: list[tuple[int, float]] = []

        def _on(t0, t1, vals):
            for i, (rows, _c) in enumerate(vals):
                if len(rows):
                    stamps.append((t0 + i, float(rows[0, COL["t"]])))

        segment_movie(movie, spec, n_frames=len(movie), client=None,
                      on_frames=_on)
        assert stamps and all(t == v for t, v in stamps), stamps


class TestCancellation:
    """Contract 2 — a stopped run is a movie with empty frames, not a short one."""

    def test_stopped_before_it_starts_still_spans_the_movie(self, movie, spec):
        rows, contours, done = segment_movie(
            movie, spec, n_frames=len(movie), client=None, stopped=[True])
        assert len(rows) == len(contours) == len(movie)
        assert done == 0
        assert all(r.shape == (0, N_COLUMNS) for r in rows)
        assert all(c == [] for c in contours)

    def test_unreached_frames_are_empty_blocks_not_missing_ones(self, movie,
                                                               spec):
        stopped = [False]
        seen = []

        def _on(t0, t1, vals):
            seen.append(t0)
            stopped[0] = True          # stop after the first block lands

        rows, contours, done = segment_movie(
            movie, spec, n_frames=len(movie), client=None, stopped=stopped,
            on_frames=_on)
        assert len(rows) == len(movie)
        assert done < len(movie)
        assert all(r.ndim == 2 and r.shape[1] == N_COLUMNS for r in rows)


class TestTaskSizing:
    """``frames_per_task`` is three bounds, and each is a different failure."""

    def test_bounded_by_bytes(self):
        # A 4096^2 uint8 frame is 16 MB; the 64 MB default holds four.
        assert frames_per_task(1000, 16 << 20, n_workers=1) == 4

    def test_never_exceeds_the_source_chunk(self):
        """Alignment beats the byte budget: a task that spans two stored chunks
        pulls both (CLAUDE.md Live-Display §1)."""
        assert frames_per_task(1000, 1 << 20, n_workers=1,
                               source_chunk=3) == 3

    def test_spreads_a_short_movie_across_the_cluster(self):
        """Six frames and nine workers must not become one task."""
        assert frames_per_task(6, 1 << 10, n_workers=9) == 1

    def test_never_zero(self):
        assert frames_per_task(1, 1 << 30, n_workers=64) == 1

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("SPYDE_SEG_TASK_BYTES", str(8 << 20))
        assert frames_per_task(1000, 1 << 20, n_workers=1) == 8

    def test_a_junk_env_value_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("SPYDE_SEG_TASK_BYTES", "lots")
        assert frames_per_task(1000, 16 << 20, n_workers=1) == 4


class TestNoRechunkShuffle:
    """Contract 5. A reader that split the signal axes (RosettaSciIO's
    balanced-cube auto-chunk) must NOT be fixed with a rechunk here — that is a
    full P2P shuffle of the movie. It falls back, and the answer stays right."""

    def test_split_signal_axes_falls_back_instead_of_rechunking(
            self, movie, reference, spec, monkeypatch):
        import dask.array as da

        split = da.from_array(movie, chunks=(3, 45, 55))
        assert any(len(c) > 1 for c in split.chunks[1:]), "fixture not split"

        def _no_rechunk(self, *a, **kw):        # pragma: no cover - must not run
            raise AssertionError("rechunked a movie instead of falling back")

        monkeypatch.setattr(da.Array, "rechunk", _no_rechunk)
        got = segment_movie(split, spec, n_frames=len(movie),
                            get_frame=lambda t: movie[t], client=None)
        rows, _c, done = got
        assert done == len(movie)
        for t, (r, ref) in enumerate(zip(rows, reference[0])):
            assert np.array_equal(r, ref), f"frame {t}"

    def test_an_already_aligned_movie_is_not_rechunked(self, movie, spec,
                                                      monkeypatch):
        """The app loads movies at one frame per chunk, which is already the
        task size — rebuilding that graph for nothing is pure overhead."""
        import dask.array as da

        calls = []
        real = da.Array.rechunk

        def _count(self, *a, **kw):
            calls.append(a)
            return real(self, *a, **kw)

        monkeypatch.setattr(da.Array, "rechunk", _count)
        lazy = da.from_array(movie, chunks=(1, -1, -1))
        segment_movie(lazy, spec, n_frames=len(movie), client=None)
        assert calls == []


class TestEngineSpec:
    def test_the_resolved_engine_labels_a_frame(self, movie, spec):
        engine, dev = resolve_engine(spec, force_cpu=True)
        assert dev  # a device string, whichever lane this worker is on
        labels = engine(movie[0])
        assert labels.dtype == np.int32
        assert labels.max() > 0, "the trained head found nothing at all"

    def test_resolving_twice_gives_the_same_answer(self, movie, spec):
        """The per-worker engine cache must not change what a frame segments to."""
        first, _ = resolve_engine(spec, force_cpu=True)
        second, _ = resolve_engine(spec, force_cpu=True)
        assert np.array_equal(first(movie[0]), second(movie[0]))

    def test_there_is_no_classical_engine_any_more(self):
        """Deleted, not deprecated — asking for it must fail loudly rather than
        silently falling back to something that behaves differently."""
        with pytest.raises(ValueError, match="no engine for method"):
            resolve_engine(EngineSpec(method="classical", params=dict(PARAMS)),
                          force_cpu=True)

    def test_scribble_without_a_model_says_so(self):
        with pytest.raises(ValueError, match="model_path"):
            resolve_engine(EngineSpec(method="scribble", params=dict(PARAMS)),
                          force_cpu=True)

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="no engine for method"):
            resolve_engine(EngineSpec(method="prompt"), force_cpu=True)

    def test_segment_params_round_trip(self, spec):
        p = spec.segment_params()
        assert p.min_size == PARAMS["min_size"]
        assert p.watershed is True                       # the dataclass default

    def test_a_stale_classical_param_does_not_reach_SegmentParams(self):
        """A result saved before the classical engine was removed carries
        ``sensitivity`` and friends in its provenance. Re-running from it must
        not explode — the action drops unknown keys, so the spec never sees
        them; this pins that the dataclass would indeed have refused."""
        from spyde.particles import SegmentParams

        with pytest.raises(TypeError):
            SegmentParams(sensitivity=0.6, min_size=10)

    def test_the_spec_pickles(self, spec):
        """It crosses to a worker process; a spec that only cloudpickles would
        work here and fail against a real scheduler."""
        import pickle
        assert pickle.loads(pickle.dumps(spec)) == spec


class TestLanePolicy:
    """Contract 4 — one value, given to both halves."""

    @pytest.mark.parametrize("setting", [None, "one", "2", "4", "8", "all"])
    def test_the_worker_gate_and_the_lane_split_agree(self, setting,
                                                      monkeypatch):
        """They must be given the SAME mode, whatever the shared setting says.

        `split_workers_for_gpu`'s docstring records the real bug this prevents:
        a lane split sized for one GPU worker while every worker actually
        submitted CUDA, which is all-workers contention wearing a lane's
        clothes.

        This used to assert that `_gpu_allowed` DELEGATES to
        `_gpu_task_allowed` with `PARTICLE_GPU_LANE_DEFAULT`. It no longer
        delegates, on purpose: segmentation caps the lane at
        `PARTICLE_GPU_LANE_MAX` regardless of the setting (a torch head plus a
        band sized against free VRAM does not tolerate what the numba kernels
        do), and reaching that through the shared helper would have meant
        mutating a process-global env var around a call made from several
        worker threads. So the AGREEMENT is asserted directly, which is what
        the test was ever about.
        """
        from spyde.particles import batch as batch_mod

        monkeypatch.delenv("SPYDE_FV_GPU", raising=False)
        monkeypatch.delenv("SPYDE_SEG_GPU_LANE_MAX", raising=False)
        if setting is not None:
            monkeypatch.setenv("SPYDE_FV_GPU", setting)

        mode = batch_mod._clamped_lane_mode()
        n_lane = 0 if mode == "off" else (1 if mode == "one" else int(mode))
        assert n_lane <= batch_mod.PARTICLE_GPU_LANE_MAX

        # The gate, per worker name, must admit exactly the lane's workers.
        class _W:
            def __init__(self, name): self.name = name

        for name in ("0", "1", "2", "3", "9"):
            monkeypatch.setattr("distributed.get_worker",
                                lambda n=name: _W(n), raising=False)
            allowed = batch_mod._gpu_allowed()
            expect = 1 <= int(name) <= n_lane
            assert allowed == expect, (
                f"setting={setting!r} mode={mode!r}: worker {name} gate says "
                f"{allowed}, the lane admits {expect}")

    def test_the_default_is_a_value_the_shared_policy_understands(self):
        """``SPYDE_FV_GPU`` accepts one/N/all/off; anything else silently
        becomes a single GPU worker, which would make the constant a lie."""
        assert (PARTICLE_GPU_LANE_DEFAULT in ("one", "all", "off")
                or PARTICLE_GPU_LANE_DEFAULT.isdigit())

    def test_off_is_honoured(self, monkeypatch):
        from spyde.particles import batch as batch_mod
        monkeypatch.setenv("SPYDE_FV_GPU", "off")
        assert batch_mod._gpu_allowed() is False

    def test_the_cpu_lane_is_opt_in(self, monkeypatch):
        from spyde.particles.batch import cpu_lane_enabled
        monkeypatch.delenv("SPYDE_SEG_CPU_LANE", raising=False)
        assert cpu_lane_enabled() is False
        monkeypatch.setenv("SPYDE_SEG_CPU_LANE", "1")
        assert cpu_lane_enabled() is True
        monkeypatch.setenv("SPYDE_SEG_CPU_LANE", "0")
        assert cpu_lane_enabled() is False


class TestDegenerateBlocks:
    def test_a_zero_length_block_is_not_an_error(self, spec):
        """dask calls the chunk fn on an empty array for meta inference."""
        out = segment_block(np.zeros((0, 8, 8), np.float32), 0, spec)
        assert out.shape == (0,)

    def test_a_non_3d_block_is_refused(self, spec):
        with pytest.raises(ValueError, match=r"\(n, h, w\)"):
            segment_block(np.zeros((8, 8), np.float32), 0, spec)

    def test_store_masks_off_drops_the_outlines_not_the_rows(self, movie, spec):
        rows, contours, _d = segment_movie(movie, spec, n_frames=len(movie),
                                           client=None, store_masks=False)
        assert any(len(r) for r in rows)
        assert all(c == [] for c in contours)


class TestDispatchChunksAssembleHook:
    """The shared dispatcher grew an ``assemble`` hook for this module; the
    default must stay exactly what find_vectors relies on."""

    def test_the_signature_defaults_to_the_padded_convention(self):
        import inspect
        from spyde.compute_dispatch import dispatch_chunks
        sig = inspect.signature(dispatch_chunks)
        assert sig.parameters["assemble"].default is None

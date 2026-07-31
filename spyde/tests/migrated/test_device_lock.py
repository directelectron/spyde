"""Pins the Apple-MPS device-serialisation contract.

Concurrent Metal submission from two threads is an uncatchable native SIGSEGV
that kills the whole backend process ("Analysis backend stopped"). The guard is
ONE process-wide lock — and a lock only works if EVERY torch user takes it.

The bug these tests exist for: the lock lived privately in ``find_vectors_torch``
and only the neural BATCH and the NXCORR paths took it, so the single-frame live
preview, the neural calibration, the cold model load, and the batched
vector-orientation fit all submitted to Metal unserialised. An orientation-mapping
run overlapping a preview took the backend down with:

    at::native::relu_mps_ -> MetalShaderLibrary::exec_unary_kernel
    at::native::zero_ -> fill_mps_kernel -> setComputePipelineState:

These tests are Qt-free, GPU-free and fast: they use a FAKE device whose ``.type``
is ``"mps"`` so the locking contract is exercised on any machine (CI included)
without touching Metal.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from spyde import device_lock
from spyde.device_lock import DEVICE_LOCK, accelerator_lock, is_mps


class _FakeDev:
    """Stands in for ``torch.device('mps')`` — only ``.type`` is consulted."""

    def __init__(self, type_="mps"):
        self.type = type_

    def __str__(self):
        return self.type


def _lock_is_held_by_me() -> bool:
    """True when THIS thread already owns DEVICE_LOCK. ``RLock.acquire`` from the
    owning thread always succeeds (it's reentrant), so probe from another thread."""
    got = []

    def probe():
        got.append(DEVICE_LOCK.acquire(blocking=False))
        if got[-1]:
            DEVICE_LOCK.release()

    t = threading.Thread(target=probe)
    t.start()
    t.join()
    return not got[0]


class TestSharedLockIdentity:
    """Every torch user must serialise against the SAME object."""

    def test_find_vectors_torch_reuses_the_shared_lock(self):
        from spyde.actions import find_vectors_torch

        assert find_vectors_torch._GPU_LOCK is DEVICE_LOCK

    def test_mps_forward_lock_is_the_shared_lock(self):
        import sys

        from spyde.actions.find_vectors_neural import _mps_forward_lock

        if sys.platform == "darwin":
            assert _mps_forward_lock() is DEVICE_LOCK
        else:
            # Off Mac the neural path deliberately keeps CUDA concurrency.
            assert _mps_forward_lock() is None

    def test_lock_is_reentrant(self):
        """Nesting on one thread is normal (batch takes the device lock and then
        the _gpu_slots semaphore); a plain Lock would self-deadlock."""
        with accelerator_lock(_FakeDev()):
            with accelerator_lock(_FakeDev()):
                assert _lock_is_held_by_me()


class TestAcceleratorLock:
    def test_noop_off_mps(self):
        """CUDA/CPU must NOT be serialised — concurrent CUDA streams are a
        deliberate throughput win, so locking there would be a pure regression."""
        for dev in (None, _FakeDev("cpu"), _FakeDev("cuda")):
            with accelerator_lock(dev):
                assert not _lock_is_held_by_me(), f"{dev} should not lock"

    def test_holds_lock_on_mps(self):
        with accelerator_lock(_FakeDev()):
            assert _lock_is_held_by_me()
        assert not _lock_is_held_by_me()

    def test_released_on_exception(self):
        with pytest.raises(RuntimeError):
            with accelerator_lock(_FakeDev()):
                raise RuntimeError("boom")
        assert not _lock_is_held_by_me()

    def test_syncs_before_release(self, monkeypatch):
        """The device must be quiesced at the hand-off: releasing while kernels
        are in flight lets the next thread submit into a live encoder."""
        calls = []
        monkeypatch.setattr(device_lock, "mps_sync", lambda: calls.append("sync"))
        with accelerator_lock(_FakeDev()):
            pass
        assert calls == ["sync"]

    def test_serialises_two_threads(self):
        """The actual invariant: two threads never inside the block at once."""
        overlap = []
        inside = [0]
        counter_lock = threading.Lock()

        def work():
            for _ in range(50):
                with accelerator_lock(_FakeDev()):
                    with counter_lock:
                        inside[0] += 1
                        if inside[0] > 1:
                            overlap.append(inside[0])
                    with counter_lock:
                        inside[0] -= 1

        ts = [threading.Thread(target=work) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert overlap == [], f"concurrent entry: {overlap}"


class TestOrientationFitTakesLock:
    """The vector-orientation fit runs on a worker thread and previously took no
    lock at all — the ``zero_``/``fill_mps_kernel`` crash."""

    def test_fit_runs_under_the_lock(self, monkeypatch):
        from spyde.actions import vector_orientation_gpu as vog

        held = []
        monkeypatch.setattr(vog, "select_device", lambda: _FakeDev())
        monkeypatch.setattr(
            vog, "_compute_vector_orientation_batched",
            lambda *a, **k: held.append(_lock_is_held_by_me()) or "result")

        assert vog.compute_vector_orientation_gpu(None, None) == "result"
        assert held == [True]
        assert not _lock_is_held_by_me(), "lock leaked after the fit"

    def test_yield_hands_the_device_back(self, monkeypatch):
        """The fit yields every ~12 refine steps; each yield must release the
        device so a concurrent preview waits one yield window, not the whole
        anneal — and must re-acquire it afterwards."""
        from spyde.actions import vector_orientation_gpu as vog

        observed = {}

        def fake_fit(*a, **k):
            # k["on_yield"] is the wrapper's releasing shim, which calls the
            # caller's on_yield in the middle of the hand-off window.
            assert _lock_is_held_by_me()
            k["on_yield"]()
            observed["held_after_yield"] = _lock_is_held_by_me()
            return "ok"

        user_calls = []
        monkeypatch.setattr(vog, "select_device", lambda: _FakeDev())
        monkeypatch.setattr(vog, "_compute_vector_orientation_batched", fake_fit)
        vog.compute_vector_orientation_gpu(
            None, None, on_yield=lambda: user_calls.append(
                _lock_is_held_by_me()))

        assert user_calls == [False], "device not released during the yield"
        assert observed["held_after_yield"] is True, "device not re-acquired"

    def test_no_deadlock_against_concurrent_users(self, monkeypatch):
        """Adding a lock risks deadlock, and the fit's release/re-acquire around
        each yield is the delicate part. Run the real yield protocol against
        preview-style acquirers on other threads and require completion."""
        from spyde.actions import vector_orientation_gpu as vog

        monkeypatch.setattr(vog, "select_device", lambda: _FakeDev())

        def fake_fit(*a, **k):
            for _ in range(40):
                k["on_yield"]()          # release -> hand off -> re-acquire
            return "ok"

        monkeypatch.setattr(vog, "_compute_vector_orientation_batched", fake_fit)
        done, errors = [], []

        def fit_thread():
            try:
                done.append(vog.compute_vector_orientation_gpu(None, None))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        def preview_thread():
            try:
                for _ in range(60):
                    with accelerator_lock(_FakeDev()):
                        pass
                done.append("preview")
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        ts = [threading.Thread(target=fit_thread)] + [
            threading.Thread(target=preview_thread) for _ in range(3)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=30)

        assert not [t for t in ts if t.is_alive()], "deadlock: thread never finished"
        assert errors == [], errors
        assert len(done) == 4
        assert not _lock_is_held_by_me(), "lock leaked"

    def test_off_mps_is_a_passthrough(self, monkeypatch):
        """CUDA path must be byte-for-byte unchanged: no lock, original on_yield
        passed straight through."""
        from spyde.actions import vector_orientation_gpu as vog

        seen = {}
        sentinel = object()

        def fake_fit(*a, **k):
            seen["on_yield"] = k.get("on_yield")
            seen["held"] = _lock_is_held_by_me()
            return "cuda"

        monkeypatch.setattr(vog, "select_device", lambda: _FakeDev("cuda"))
        monkeypatch.setattr(vog, "_compute_vector_orientation_batched", fake_fit)
        vog.compute_vector_orientation_gpu(None, None, on_yield=sentinel)

        assert seen["on_yield"] is sentinel
        assert seen["held"] is False


class TestNeuralPathsTakeLock:
    """The single-frame preview and the calibration were the unlocked neural
    entry points (the two concurrent ``relu_mps_`` threads in the crash)."""

    def test_single_frame_preview_locks(self, monkeypatch):
        from spyde import models
        from spyde.actions import find_vectors_neural as fvn

        held = []
        monkeypatch.setattr(models, "get_model",
                            lambda mid=None: (object(), _FakeDev()))

        def fake_detect(model, f, device, **kw):
            held.append(_lock_is_held_by_me())
            return np.zeros((0, 3), dtype=np.float32)

        monkeypatch.setattr(models, "detect", fake_detect)
        fvn._find_vectors_single_frame_neural(
            np.zeros((32, 32), dtype=np.float32), threshold=0.5, min_distance=3)
        assert held == [True], "preview forward ran unserialised on MPS"

    def test_calibration_locks(self, monkeypatch):
        from spyde import models
        from spyde.actions import find_vectors_neural as fvn

        held = []
        monkeypatch.setattr(models, "get_model",
                            lambda mid=None: (object(), _FakeDev()))

        def fake_calibrate(model, frames, device, **kw):
            held.append(_lock_is_held_by_me())
            return {"bg_sigma": 12.0, "thresh": 0.5, "scale_factor": 1.0,
                    "confidence": 0.9}

        monkeypatch.setattr(models, "calibrate", fake_calibrate)
        fvn.calibrate_neural([np.zeros((32, 32), dtype=np.float32)])
        assert held == [True], "calibration forwards ran unserialised on MPS"


class TestZeroThreeZeroPathsTakeLock:
    """The 0.3.0 compute packages — fitting, EBSD — are torch users too.

    They were written CUDA-or-CPU, so on a Mac they never reached Metal and the
    missing lock was invisible. Enabling MPS makes them submitters like any
    other, and CLAUDE.md's rule is the one that matters here: *a lock only works
    if EVERY participant takes it*, and a new torch call site added without one
    silently re-opens the crash.

    These assert the lock is taken on the COMPUTE PATH, using a real CPU run and
    a recording wrapper, so they guard in CI (which has no Metal) rather than
    skipping exactly where a regression would land. `accelerator_lock` is a null
    context off MPS, so what is pinned is that the call site EXISTS and is
    reached — which is the thing that gets forgotten.
    """

    @staticmethod
    def _spy(monkeypatch, module):
        """Wrap a module's `accelerator_lock` and record the devices it guards."""
        seen = []
        real = device_lock.accelerator_lock

        def spy(device=None, **kw):
            seen.append(device)
            return real(device, **kw)

        monkeypatch.setattr(module, "accelerator_lock", spy)
        return seen

    def test_fit_batched_locks_every_chunk(self, monkeypatch):
        from spyde.fitting import engine

        seen = self._spy(monkeypatch, engine)
        _fit_a_tiny_model(chunk=2)
        # 4 positions, chunk=2 -> one acquisition per chunk, not one overall:
        # a whole-scan fit must not hold the device end-to-end.
        assert len(seen) == 2, f"expected one lock per chunk, got {seen}"
        assert all(d == "cpu" for d in seen)

    def test_dictionary_index_locks(self, monkeypatch):
        pytest.importorskip("torch")
        from spyde.ebsd import indexing

        seen = self._spy(monkeypatch, indexing)
        rng = np.random.default_rng(0)
        indexing.dictionary_index(rng.normal(size=(6, 5, 5)).astype("float32"),
                                  rng.normal(size=(9, 5, 5)).astype("float32"),
                                  device="cpu")
        assert seen, "dictionary indexing submitted to the device unserialised"

    def test_the_live_single_pattern_match_locks(self, monkeypatch):
        """The band overlay's per-navigator-move match — the concurrent
        submitter most likely to overlap a running index."""
        pytest.importorskip("torch")
        from spyde.ebsd import indexing

        rng = np.random.default_rng(0)
        idx = indexing.SinglePatternIndexer(
            rng.normal(size=(9, 5, 5)).astype("float32"),
            rng.normal(size=(9, 3)), device="cpu")
        seen = self._spy(monkeypatch, indexing)
        idx.best(rng.normal(size=(5, 5)))
        assert seen, "live single-pattern match ran unserialised"

    def test_remove_background_locks(self, monkeypatch):
        pytest.importorskip("torch")
        from spyde.ebsd import preprocess

        seen = self._spy(monkeypatch, preprocess)
        rng = np.random.default_rng(0)
        # sigma small enough that the blur radius fits the pattern.
        preprocess.remove_background(
            rng.normal(size=(4, 16, 16)).astype("float32"), sigma=2.0,
            device="cpu")
        assert seen, "background removal ran unserialised"

    def test_average_dot_product_map_locks(self, monkeypatch):
        pytest.importorskip("torch")
        from spyde.ebsd import preprocess

        seen = self._spy(monkeypatch, preprocess)
        rng = np.random.default_rng(0)
        preprocess.average_dot_product_map(
            rng.normal(size=(3, 3, 6, 6)).astype("float32"), device="cpu")
        assert seen, "ADP ran unserialised"


def _fit_a_tiny_model(**kw):
    """A 4-position fit, small enough to be instant on the CPU."""
    import hyperspy.api as hs
    from hyperspy.components1d import Gaussian, Offset

    from spyde.fitting import ModelSpec
    from spyde.fitting.engine import fit_batched

    x = np.linspace(0.0, 50.0, 64)
    data = np.stack([5.0 + a * np.exp(-((x - 25.0) ** 2) / 18.0)
                     for a in (20.0, 30.0, 40.0, 50.0)])
    s = hs.signals.Signal1D(data)
    s.axes_manager.signal_axes[0].offset = x[0]
    s.axes_manager.signal_axes[0].scale = x[1] - x[0]
    m = s.create_model()
    m.extend([Offset(), Gaussian()])
    m[0].offset.value = 1.0
    m[1].A.value, m[1].centre.value, m[1].sigma.value = 30.0, 25.0, 2.0
    return fit_batched(ModelSpec.from_model(m), data, x, device="cpu", **kw)

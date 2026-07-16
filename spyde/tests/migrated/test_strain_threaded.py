"""
Threaded strain recompute path (`spyde.actions.strain_action.StrainController`).

The interactive strain field is computed OFF the asyncio main thread: a worker
thread runs the per-pixel scipy fit and marshals the figure-apply back via
``Session._dispatch_to_main`` (``loop.call_soon_threadsafe``), with a
generation counter dropping a superseded recompute (latest-wins).

The existing strain tests (test_strain_mapping.py::TestStrainAction) only cover
the INLINE fallback — they use a duck-typed session with no ``_dispatch_to_main``
so ``_recompute`` runs synchronously. These tests exercise the REAL threaded
branch: a session with a working ``_dispatch_to_main`` driven by a live asyncio
loop on a background thread.
"""
from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

import anyplotlib as apl
from spyde.actions.strain_action import StrainController
from spyde.backend.session import Session


# Square 1st+2nd ring reference lattice (non-collinear) — enough rings to fit.
_G_REF = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
                   [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])


def _strained(T, g=_G_REF):
    """Measured reciprocal vectors of a lattice under real deformation T."""
    M = np.linalg.inv(np.asarray(T, dtype=float)).T
    return g @ M.T


class _Vecs:
    """Duck-typed SpyDEDiffractionVectors: nav_shape + kxy_at + count_map."""
    nav_shape = (4, 4)

    def kxy_at(self, iy, ix):
        # mild εxx gradient with ix, reference (0,0) unstrained
        return _strained(np.array([[1.0 + 0.01 * ix, 0.0], [0.0, 1.0]]))

    def count_map(self):
        return np.full(self.nav_shape, len(_G_REF), dtype=int)


class _LoopSession:
    """A minimal session that drives the REAL Session._dispatch_to_main against a
    live asyncio loop running on its own thread — so the strain worker thread's
    marshal-back actually crosses threads (not the inline test fallback)."""

    # Bind the production marshal method onto this stand-in.
    _dispatch_to_main = Session._dispatch_to_main
    set_main_loop = Session.set_main_loop

    def __init__(self):
        self._main_loop = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="test-strain-loop")
        self._thread.start()
        assert self._ready.wait(5.0), "event loop thread did not start"

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.set_main_loop(loop)
        self._ready.set()
        loop.run_forever()

    def stop(self):
        loop = self._main_loop
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=5.0)


@pytest.fixture
def loop_session():
    s = _LoopSession()
    yield s
    s.stop()


class TestStrainThreaded:
    def test_recompute_applies_via_dispatch_thread(self, loop_session):
        """The worker thread computes the field and the apply lands on the loop
        thread (not the caller's thread) — proving the real marshal path runs."""
        fig, ax = apl.subplots()
        p = ax.imshow(np.zeros((4, 4), "f4"))

        applied = threading.Event()
        apply_thread = {}

        ctrl = StrainController(_Vecs(), p, ref_yx=(0, 0),
                                session=loop_session)

        # Wrap update_strain_view so we can observe WHICH thread applies + signal.
        import spyde.actions.strain_display as sd
        orig = sd.update_strain_view

        def _spy(plot, field, component, **kw):
            apply_thread["name"] = threading.current_thread().name
            orig(plot, field, component, **kw)
            applied.set()

        sd.update_strain_view = _spy
        try:
            # attach() calls _recompute() once (the genuine threaded branch).
            ctrl.attach()
            assert applied.wait(10.0), "threaded strain apply never landed"
        finally:
            sd.update_strain_view = orig

        assert ctrl.field is not None and ctrl.field.nav_shape == (4, 4)
        # The apply ran on the loop thread, NOT inline on the caller — i.e. it was
        # genuinely marshalled across threads by _dispatch_to_main.
        assert apply_thread["name"] == "test-strain-loop"

    def test_superseded_recompute_is_dropped(self, loop_session):
        """A slow in-flight recompute whose generation is bumped by a newer
        interaction must NOT apply its stale result (latest-wins)."""
        fig, ax = apl.subplots()
        p = ax.imshow(np.zeros((4, 4), "f4"))
        ctrl = StrainController(_Vecs(), p, ref_yx=(0, 0),
                                session=loop_session)
        # Establish the reference set so _recompute() actually fits (otherwise it
        # short-circuits on len(ref) < 2). Done BEFORE the mocks are installed.
        ctrl._set_full_reference(ctrl.vecs.kxy_at(0, 0))

        # Block the FIRST worker's compute (the gen captured at the first call)
        # until we've issued a newer recompute, so the first worker's _apply runs
        # with a stale generation and must be dropped. Deterministic: we only
        # release the gate AFTER the second recompute has been submitted.
        import spyde.actions.strain_mapping as sm
        orig_compute = sm.compute_strain_field
        gate = threading.Event()
        first_started = threading.Event()
        calls = {"n": 0}
        calls_lock = threading.Lock()

        def _slow(*a, **k):
            with calls_lock:
                calls["n"] += 1
                is_first = calls["n"] == 1
            if is_first:
                first_started.set()
                assert gate.wait(10.0), "gate never released"
            return orig_compute(*a, **k)

        applies = []
        import spyde.actions.strain_display as sd
        orig_apply = sd.update_strain_view

        def _count_apply(plot, field, component, **kw):
            applies.append(field)
            orig_apply(plot, field, component, **kw)

        sm.compute_strain_field = _slow
        sd.update_strain_view = _count_apply
        try:
            # First recompute (gen 1) — its worker blocks inside _slow.
            ctrl._recompute()
            assert first_started.wait(10.0), "first worker never entered compute"
            # Second recompute (gen 2) supersedes it. Its worker is the 2nd _slow
            # call (not first) → runs to completion and applies.
            ctrl._recompute()
            deadline_applies = _wait_until(lambda: len(applies) >= 1, 10.0)
            assert deadline_applies, "newer recompute never applied"
            # Now release the stale gen-1 worker; its _apply must be dropped.
            gate.set()
            # Give the stale worker a chance to (wrongly) apply, then confirm it
            # did not.
            import time
            time.sleep(0.5)
        finally:
            sm.compute_strain_field = orig_compute
            sd.update_strain_view = orig_apply

        # Exactly ONE apply (gen 2). The stale gen-1 result was dropped by the
        # generation-counter guard in _recompute's _apply.
        assert len(applies) == 1, f"expected 1 apply, got {len(applies)}"
        assert ctrl._recompute_gen == 2


def _wait_until(pred, timeout):
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()

"""The deterministic test-hold (backend/test_hold.py).

Two properties matter more than the feature itself, because this code ships:

  * with ``SPYDE_TEST_HOLD`` unset it must do NOTHING — production pays one
    module-level bool and no locks, no threads, no waits;
  * it must be impossible to wedge CI. Every wait is bounded, so a test that
    forgets to release costs `MAX_HOLD_S` once instead of hanging the job.
"""
from __future__ import annotations

import importlib
import os
import threading
import time

import pytest




@pytest.fixture(autouse=True)
def _disarm_after_each():
    """Re-import the module DISARMED after every test in this file.

    Load-bearing, and the reason is a bug this file caused: `_SPEC` is parsed
    once at import (that is what makes the unset case free), so a
    `monkeypatch.setenv` + `importlib.reload` leaves the module ARMED for the
    rest of the pytest session even though monkeypatch has faithfully restored
    the environment variable. `find_vectors_action` holds a module reference,
    so a later test's batch then hit a live hold point and parked for
    `MAX_HOLD_S` — surfacing as "the landing block painted no frame" and
    "assert 0.0 > 0" in tests that have nothing to do with holds, on whichever
    shard happened to run them afterwards.
    """
    yield
    import spyde.backend.test_hold as th
    os.environ.pop("SPYDE_TEST_HOLD", None)
    importlib.reload(th).reset()


def _fresh(monkeypatch, spec: "str | None"):
    """Re-import the module with ``SPYDE_TEST_HOLD`` set to *spec* — the spec is
    parsed once at import, which is what makes the unset case free. The autouse
    fixture above puts it back."""
    import spyde.backend.test_hold as th
    if spec is None:
        monkeypatch.delenv("SPYDE_TEST_HOLD", raising=False)
    else:
        monkeypatch.setenv("SPYDE_TEST_HOLD", spec)
    mod = importlib.reload(th)
    mod.reset()
    return mod


class TestInertInProduction:
    def test_unset_means_not_armed(self, monkeypatch):
        th = _fresh(monkeypatch, None)
        assert th.armed() is False

    def test_hold_point_returns_immediately(self, monkeypatch):
        th = _fresh(monkeypatch, None)
        t0 = time.monotonic()
        th.hold_point("fv-batch", 10, 10)       # would park if it were armed
        assert time.monotonic() - t0 < 0.05

    def test_release_of_an_unconfigured_hold_is_false_not_an_error(self, monkeypatch):
        th = _fresh(monkeypatch, None)
        assert th.release("fv-batch") is False

    def test_an_unnamed_point_is_not_held(self, monkeypatch):
        """Arming one point must not pause every other one."""
        th = _fresh(monkeypatch, "other@0.5")
        t0 = time.monotonic()
        th.hold_point("fv-batch", 10, 10)
        assert time.monotonic() - t0 < 0.05


class TestSpecParsing:
    def test_fraction_of_total(self, monkeypatch):
        th = _fresh(monkeypatch, "fv-batch@0.25")
        assert th.armed() is True
        assert th._SPEC == {"fv-batch": 0.25}

    def test_absolute_count(self, monkeypatch):
        th = _fresh(monkeypatch, "fv-batch@800")
        assert th._SPEC == {"fv-batch": 800.0}

    def test_several_points(self, monkeypatch):
        th = _fresh(monkeypatch, "a@0.5, b@100")
        assert th._SPEC == {"a": 0.5, "b": 100.0}

    def test_a_bare_name_holds_at_once(self, monkeypatch):
        th = _fresh(monkeypatch, "fv-batch")
        assert th._SPEC == {"fv-batch": 0.0}

    def test_rubbish_is_ignored_not_raised(self, monkeypatch):
        """A malformed spec must not take the backend down on startup."""
        th = _fresh(monkeypatch, "fv-batch@banana,,good@0.5")
        assert th._SPEC == {"good": 0.5}


class TestHolding:
    def test_below_the_threshold_does_not_park(self, monkeypatch):
        th = _fresh(monkeypatch, "fv-batch@0.5")
        t0 = time.monotonic()
        th.hold_point("fv-batch", 10, 100)      # 10% of 100, threshold 50
        assert time.monotonic() - t0 < 0.05

    def test_parks_at_the_threshold_and_resumes_on_release(self, monkeypatch):
        th = _fresh(monkeypatch, "fv-batch@0.25")
        done = threading.Event()

        def worker():
            th.hold_point("fv-batch", 25, 100)  # exactly at the threshold
            done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        # It must still be parked a beat later — that IS the window a test uses.
        assert not done.wait(0.4), "hold_point did not park"
        assert th.release("fv-batch") is True
        assert done.wait(5.0), "release did not let it continue"

    def test_EVERY_caller_parks_not_just_the_first(self, monkeypatch):
        """THE property that makes the hold actually pause a compute.

        Holding only the first caller was the obvious design and it silently
        does nothing useful: a per-chunk callback parks one thread while every
        later chunk sails through, so the batch runs to completion and the
        "paused" state the test wants to inspect never exists. Observed exactly
        that in the real spec — the hold logged `parked`, and the batch
        finalized anyway.
        """
        th = _fresh(monkeypatch, "fv-batch@0.1")
        monkeypatch.setattr(th, "MAX_HOLD_S", 5.0)
        through = []

        def worker(i):
            th.hold_point("fv-batch", 50 + i, 100)
            through.append(i)

        ts = [threading.Thread(target=worker, args=(i,), daemon=True)
              for i in range(4)]
        for t in ts:
            t.start()
        time.sleep(0.4)
        assert through == [], f"callers got through while parked: {through}"

        th.release("fv-batch")
        for t in ts:
            t.join(5.0)
        assert sorted(through) == [0, 1, 2, 3]

    def test_after_release_later_chunks_do_not_stutter(self, monkeypatch):
        th = _fresh(monkeypatch, "fv-batch@0.1")
        th.release("fv-batch")
        for done in (60, 70, 100):
            t0 = time.monotonic()
            th.hold_point("fv-batch", done, 100)
            assert time.monotonic() - t0 < 0.05, f"re-parked at {done}/100"

    def test_a_forgotten_release_cannot_hang_the_run(self, monkeypatch):
        th = _fresh(monkeypatch, "fv-batch@0.1")
        monkeypatch.setattr(th, "MAX_HOLD_S", 0.3)
        t0 = time.monotonic()
        th.hold_point("fv-batch", 50, 100)      # nobody ever releases it
        elapsed = time.monotonic() - t0
        assert 0.25 <= elapsed < 3.0, f"unbounded wait ({elapsed:.2f}s)"

    def test_release_before_the_hold_is_reached(self, monkeypatch):
        """Release/park ordering must not matter — the event latches."""
        th = _fresh(monkeypatch, "fv-batch@0.5")
        th.release("fv-batch")
        t0 = time.monotonic()
        th.hold_point("fv-batch", 50, 100)
        assert time.monotonic() - t0 < 0.5


class TestActionWiring:
    def test_the_release_action_is_registered_and_importable(self):
        """The e2e reaches this by name over IPC, so a typo in either the
        registry key or the dotted path is invisible until a spec hangs."""
        from spyde.actions.registry import STAGED_HANDLERS

        assert "test_hold_release" in STAGED_HANDLERS
        mod_path, _, fn_name = STAGED_HANDLERS["test_hold_release"].rpartition(".")
        mod = importlib.import_module(mod_path)
        assert callable(getattr(mod, fn_name))

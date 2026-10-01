"""
test_dask_ready_latch.py — PHASE 1 PROBE for SESSION_RESYNC_PLAN.md §8/§9.

Pins the claim the plan's correction rests on: **a dead cluster is
indistinguishable from a live one at the load gate**.

``Session._await_dask()`` blocks every file/example load until the Dask cluster
is ready, using ``self._dask_ready`` — a ``threading.Event``. An Event is a
LATCH: once ``set()`` it stays set until something explicitly clears it. Exactly
one place in the codebase clears this one (``compute_config.py``, on a
user-driven cluster restart). Nothing clears it when a cluster dies on its own.

So after a cluster death the gate still opens instantly and the load proceeds
against a dead client. "Re-loading data works" is therefore evidence about the
latch, NOT about the cluster — which is why the earlier claim that the cluster
survives a laptop sleep was withdrawn.

These are characterisation tests: they describe what the code does today so the
behaviour cannot change silently while §9 is being designed. They are not
asserting that the current behaviour is CORRECT — §9 proposes changing it.
"""
from __future__ import annotations

import pathlib
import re

from spyde.backend import session as session_mod


class TestTheGateIsALatch:
    def test_await_dask_returns_immediately_once_set(self, window):
        """The gate does not re-check anything — it reads the Event."""
        session = window["window"]
        assert session._dask_ready.is_set(), "fixture should start cluster-ready"
        # No cluster is consulted here; this is a pure Event read.
        assert session._await_dask(timeout=0.01) is True

    def test_a_dead_client_still_passes_the_gate(self, window):
        """THE finding. Tear the client out from under the session — the state a
        cluster death leaves behind — and the gate still opens instantly."""
        session = window["window"]
        mgr = session.dask_manager
        if mgr is not None:
            # Simulate the cluster being gone without touching the latch, which
            # is precisely what a death (as opposed to a restart) does.
            try:
                mgr.client = None
            except Exception:
                pass

        assert session._await_dask(timeout=0.01) is True, (
            "a load proceeds against a dead client — the gate cannot tell")

    def test_nothing_clears_the_latch_on_death(self):
        """Structural: the ONLY clear() is the user-driven restart path.

        If a liveness detector is added (§9) it will add a second clear() and
        this test will fail — deliberately. Update it then; the point is that
        the count cannot change by accident."""
        root = pathlib.Path(session_mod.__file__).resolve().parents[2]
        hits = []
        for path in (root / "spyde").rglob("*.py"):
            if "tests" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                if re.search(r"_dask_ready\s*\.\s*clear\s*\(", line):
                    hits.append(f"{path.relative_to(root)}: {line.strip()}")

        assert len(hits) == 1, (
            "expected exactly ONE place to clear the dask gate (the user-driven "
            f"cluster restart in compute_config.py); found:\n  " + "\n  ".join(hits))
        assert "compute_config" in hits[0], hits[0]

    def test_no_liveness_check_consumes_the_cluster(self):
        """Structural: nothing polls the cluster for health.

        Searches for the shapes a detector would take. If this starts failing,
        someone added one — good, but §9's traps (worker churn is not cluster
        death; never probe from the asyncio main thread) need reading first."""
        root = pathlib.Path(session_mod.__file__).resolve().parents[2]
        suspects = []
        for path in (root / "spyde").rglob("*.py"):
            if "tests" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if re.search(r"(cluster_alive|is_cluster_alive|_check_cluster|"
                         r"cluster_died|on_cluster_lost)", text):
                suspects.append(str(path.relative_to(root)))
        assert suspects == [], (
            "a cluster-liveness path now exists — see SESSION_RESYNC_PLAN §9 "
            f"before changing it: {suspects}")

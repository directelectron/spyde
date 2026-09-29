"""
A window that a compute is still filling shows its toolbar greyed out, and
gets it back on EVERY way the fill can end: success, failure and cancel.

The mechanism is the result-tree lock (``lifecycle.lock_tree``): a tree opened
through ``commit.open_result_tree(filling=...)`` (or committed early as a blank
with ``commit_result_tree(filling=...)``) is locked, its toolbar config carries
``disabled`` on every action, and the action that fills it releases it. These
tests drive each such fill with its compute stubbed out, so what they pin is
the lock's wiring, not the science: greyed during, live after, and never left
grey — a toolbar stuck grey is worse than none.
"""
from __future__ import annotations

import ast
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spyde.actions.lifecycle import tree_lock


def _toolbar(tree) -> list[dict]:
    from spyde.drawing.toolbars.plot_control_toolbar import get_toolbar_config_for_plot
    actions = []
    for plot in getattr(tree, "signal_plots", []) or []:
        actions += get_toolbar_config_for_plot(plot.plot_state)
    return actions


def _assert_greyed(tree, label: str) -> None:
    assert tree_lock(tree) == label
    actions = _toolbar(tree)
    assert actions, "the filling window has no toolbar to grey"
    enabled = [a["name"] for a in actions if not a.get("disabled")]
    assert not enabled, f"clickable while {label!r} fills: {enabled}"
    assert all(label in a["disabled_reason"] for a in actions)


def _assert_live(tree) -> None:
    assert tree_lock(tree) is None, "the fill left its window locked"
    assert not any(a.get("disabled") for a in _toolbar(tree)), \
        "the toolbar stayed grey after the fill ended"


def _wait(predicate, timeout=10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ── the shared entry points ───────────────────────────────────────────────────

class TestTheEntryPointsLock:
    def test_open_result_tree_locks_when_filling(self, window):
        from spyde.actions.commit import open_result_tree
        session = window["window"]
        tree = open_result_tree(session, title="Map",
                                data=np.zeros((4, 5), np.float32),
                                filling="Some Fill")
        _assert_greyed(tree, "Some Fill")

    def test_open_result_tree_without_filling_is_not_locked(self, window):
        from spyde.actions.commit import open_result_tree
        tree = open_result_tree(window["window"], title="Map",
                                data=np.zeros((4, 5), np.float32))
        _assert_live(tree)

    def test_an_early_committed_blank_locks_when_filling(self, window):
        from spyde.actions.commit import commit_result_tree
        blank = np.zeros((4, 5, 3), np.uint8)
        tree = commit_result_tree(window["window"], title="IPF", primary=blank,
                                  filling="Some Fill")
        _assert_greyed(tree, "Some Fill")

    def test_every_filling_caller_releases_the_lock(self):
        """A window opened locked and never released stays grey for the rest
        of the session. Every module that opens one must also call
        ``unlock_tree`` — checked from the source, so a new fill that forgets
        fails here instead of in front of a user."""
        root = Path(__file__).resolve().parents[2]
        offenders = []
        openers = 0
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            fills = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                in ("open_result_tree", "commit_result_tree")
                and any(k.arg == "filling" for k in node.keywords)
            ]
            if not fills:
                continue
            openers += 1
            calls = {getattr(n.func, "id", getattr(n.func, "attr", None))
                     for n in ast.walk(tree) if isinstance(n, ast.Call)}
            if "unlock_tree" not in calls:
                offenders.append(path.name)
        assert openers >= 4, "the fills that open a locked window moved"
        assert not offenders, f"opens a locked window, never unlocks: {offenders}"


# ── Find Diffraction Vectors: a FAILED batch ─────────────────────────────────

class TestFindVectorsFailure:
    def test_a_failed_batch_releases_the_window(self, stem_4d_dataset,
                                                monkeypatch):
        import spyde.actions.find_vectors_action as fva
        session = stem_4d_dataset["window"]
        source = session.signal_trees[0]
        seen = {}

        def failing_compute(src, p, **kw):
            seen["tree"] = session.signal_trees[-1]
            seen["lock"] = tree_lock(seen["tree"])
            raise RuntimeError("boom")

        monkeypatch.setattr(fva, "_do_compute_vectors", failing_compute)
        monkeypatch.setattr(fva, "emit_error", lambda *a, **k: None)
        fva._start_batch(session, source.signal_plots[0], source,
                         fva._coerce(dict(fva.DEFAULTS)))
        assert _wait(lambda: "tree" in seen), "the batch never ran"
        assert seen["lock"] == "Find Diffraction Vectors"
        assert _wait(lambda: tree_lock(seen["tree"]) is None)
        _assert_live(seen["tree"])


# ── EBSD Indexing ─────────────────────────────────────────────────────────────

class TestEbsdIndexing:
    @staticmethod
    def _run(session, monkeypatch, compute):
        import spyde.actions.ebsd_action as ebsd
        finalized = {}

        class _Painter:
            def __init__(self, *a, **k):
                pass

            def paint(self, *a, **k):
                pass

        monkeypatch.setattr(ebsd, "_nav_shape", lambda signal: (4, 5))
        monkeypatch.setattr(ebsd, "_lazy_scan", lambda signal: None)
        monkeypatch.setattr(ebsd, "_ProgressivePainter", _Painter)
        monkeypatch.setattr(ebsd, "_packed_index_graph", lambda *a, **k: None)
        monkeypatch.setattr(ebsd, "_compute_nav_chunks", compute)
        monkeypatch.setattr(ebsd, "_adp_graph", lambda *a, **k: SimpleNamespace(
            compute=lambda **k: np.zeros((4, 5), np.float32)))
        monkeypatch.setattr(ebsd, "_orientation_map", lambda *a, **k: "OM")
        monkeypatch.setattr(ebsd, "emit_error", lambda *a, **k: None)

        def _finalize(session_, tree, om, **kw):
            finalized["lock"] = tree_lock(tree)

        monkeypatch.setattr(ebsd, "_finalize_ipf_window", _finalize)
        source = session.signal_trees[0]
        return ebsd, source, finalized

    def test_greyed_while_indexing_live_after(self, stem_4d_dataset, monkeypatch):
        session = stem_4d_dataset["window"]
        seen = {}

        def compute(packed, stopped, **kw):
            seen["tree"] = session.signal_trees[-1]
            _assert_greyed(seen["tree"], "EBSD Indexing")
            return np.zeros((4, 5, 5), np.float32)

        ebsd, source, finalized = self._run(session, monkeypatch, compute)
        ebsd._run_indexing(session, source, SimpleNamespace(euler=None), keep=1,
                           refine=False, steps=0)
        assert "tree" in seen
        # Released BEFORE the finalize: it adds the quality maps as nodes,
        # which a locked tree refuses.
        assert finalized["lock"] is None
        _assert_live(seen["tree"])

    def test_a_failed_index_releases_the_window(self, stem_4d_dataset,
                                                monkeypatch):
        session = stem_4d_dataset["window"]
        seen = {}

        def compute(packed, stopped, **kw):
            seen["tree"] = session.signal_trees[-1]
            raise RuntimeError("boom")

        ebsd, source, finalized = self._run(session, monkeypatch, compute)
        with pytest.raises(RuntimeError):
            ebsd._run_indexing(session, source, SimpleNamespace(euler=None), keep=1,
                               refine=False, steps=0)
        assert not finalized
        _assert_live(seen["tree"])

    def test_a_cancelled_index_releases_the_window(self, stem_4d_dataset,
                                                   monkeypatch):
        session = stem_4d_dataset["window"]
        seen = {}

        def compute(packed, stopped, **kw):
            seen["tree"] = session.signal_trees[-1]
            stopped[0] = True
            return None

        ebsd, source, finalized = self._run(session, monkeypatch, compute)
        assert ebsd._run_indexing(session, source, SimpleNamespace(euler=None), keep=1,
                                  refine=False, steps=0) is None
        assert not finalized
        _assert_live(seen["tree"])


# ── Orientation Mapping (template match) ─────────────────────────────────────

class TestOrientationMapping:
    @staticmethod
    def _run(session, monkeypatch, compute):
        import spyde.actions.orientation_action as oa
        import spyde.actions.orientation_compute as oc
        finalized = {}
        monkeypatch.setattr(oc, "_do_compute_orientations", compute)

        def _finalize(tree, om, session=None):
            finalized["lock"] = tree_lock(tree)

        monkeypatch.setattr(oa, "_finalize_ipf_window", _finalize)
        source = session.signal_trees[0]
        return oa, source, finalized

    def test_greyed_while_matching_live_after(self, stem_4d_dataset,
                                              monkeypatch):
        session = stem_4d_dataset["window"]
        seen = {}

        def compute(src, sim, params, **kw):
            seen["tree"] = session.signal_trees[-1]
            _assert_greyed(seen["tree"], "Orientation Mapping")
            return "OM"

        oa, source, finalized = self._run(session, monkeypatch, compute)
        assert oa._compute_with_live_ipf(session, source.root, source,
                                         None, {}) == "OM"
        assert finalized["lock"] is None
        _assert_live(seen["tree"])

    def test_a_failed_match_releases_the_window(self, stem_4d_dataset,
                                                monkeypatch):
        session = stem_4d_dataset["window"]
        seen = {}

        def compute(src, sim, params, **kw):
            seen["tree"] = session.signal_trees[-1]
            raise RuntimeError("boom")

        oa, source, finalized = self._run(session, monkeypatch, compute)
        with pytest.raises(RuntimeError):
            oa._compute_with_live_ipf(session, source.root, source, None, {})
        assert not finalized
        _assert_live(seen["tree"])

    def test_a_cancelled_match_releases_the_window(self, stem_4d_dataset,
                                                   monkeypatch):
        session = stem_4d_dataset["window"]
        seen = {}

        def compute(src, sim, params, **kw):
            seen["tree"] = session.signal_trees[-1]
            kw["stopped_flag"][0] = True
            return None

        oa, source, finalized = self._run(session, monkeypatch, compute)
        assert oa._compute_with_live_ipf(session, source.root, source,
                                         None, {}) is None
        assert not finalized
        _assert_live(seen["tree"])


# ── Vector Orientation Mapping (Compute Maps) ────────────────────────────────

class TestVectorOrientationMapping:
    @staticmethod
    def _run(session, monkeypatch, fit):
        import spyde.actions.vector_orientation_om as vom
        import spyde.torch_device as torch_device
        source = session.signal_trees[0]
        source._vom_wizard = SimpleNamespace(fitter=object(), phases=[])
        source.diffraction_vectors = SimpleNamespace(nav_shape=(4, 5))
        built = {}

        class _Painter:
            def __init__(self, *a, **k):
                pass

            def on_band(self, *a, **k):
                pass

        def _build(session_, src, result, *, smooth=False, ipf_tree=None):
            built["lock"] = tree_lock(ipf_tree)

        monkeypatch.setattr(torch_device, "warmup_autograd", lambda: None)
        monkeypatch.setattr(vom, "_BandPainter", _Painter)
        monkeypatch.setattr(vom, "_fit_field", fit)
        monkeypatch.setattr(vom, "_build_result_windows", _build)
        monkeypatch.setattr(vom, "emit_error", lambda *a, **k: None)
        n_before = len(session.signal_trees)
        vom.vom_run(session, source.signal_plots[0], {"smooth": False})
        assert len(session.signal_trees) == n_before + 1, \
            "the orientation window did not open before the fit"
        return session.signal_trees[-1], built

    def test_greyed_while_fitting_live_after(self, stem_4d_dataset,
                                             monkeypatch):
        session = stem_4d_dataset["window"]
        inside = threading.Event()
        release = threading.Event()

        def fit(vecs, wiz, params, **kw):
            inside.set()
            release.wait(10.0)
            return "RESULT"

        ipf_tree, built = self._run(session, monkeypatch, fit)
        assert inside.wait(10.0), "the fit never ran"
        _assert_greyed(ipf_tree, "Vector Orientation Mapping")
        release.set()
        assert _wait(lambda: tree_lock(ipf_tree) is None)
        assert "lock" in built
        _assert_live(ipf_tree)

    def test_a_failed_fit_releases_the_window(self, stem_4d_dataset,
                                              monkeypatch):
        session = stem_4d_dataset["window"]

        def fit(vecs, wiz, params, **kw):
            raise RuntimeError("boom")

        ipf_tree, built = self._run(session, monkeypatch, fit)
        assert _wait(lambda: tree_lock(ipf_tree) is None)
        assert not built
        _assert_live(ipf_tree)

    def test_a_cancelled_fit_releases_the_window(self, stem_4d_dataset,
                                                 monkeypatch):
        session = stem_4d_dataset["window"]

        def fit(vecs, wiz, params, **kw):
            return None          # what a closed tree's fit returns

        ipf_tree, built = self._run(session, monkeypatch, fit)
        assert _wait(lambda: tree_lock(ipf_tree) is None)
        assert not built
        _assert_live(ipf_tree)

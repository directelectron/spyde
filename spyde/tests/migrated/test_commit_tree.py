"""
test_commit_tree.py — the two tree-spawning lifecycles (spyde/actions/commit.py):
open_result_tree (early-open / progressive fill) and commit_result_tree (the
Commit action), including provenance stamping, chip views, and eviction of the
per-window view data when the committed window closes.
"""
from __future__ import annotations

import numpy as np

from spyde.actions import views
from spyde.actions.commit import commit_result_tree, open_result_tree


class TestOpenResultTree:
    def test_opens_new_tree_with_title_and_provenance(self, window):
        session = window["window"]
        n0 = len(session.signal_trees)
        tree = open_result_tree(
            session, title="My Result",
            data=np.zeros((6, 7), np.float32),
            provenance={"action": "Test Action", "source_title": "Src"})
        assert len(session.signal_trees) == n0 + 1
        assert tree is session.signal_trees[-1]
        assert tree.root.metadata.General.title == "My Result"
        assert tree._commit_provenance == {"action": "Test Action", "source_title": "Src"}
        assert tree.root.metadata.General.spyde_provenance.as_dictionary()[
            "action"] == "Test Action"

    def test_accepts_prepared_signal_and_signal_type(self, window):
        import hyperspy.api as hs
        session = window["window"]
        sig = hs.signals.Signal2D(np.zeros((4, 4), np.float32))
        tree = open_result_tree(session, title="Typed", signal=sig,
                                signal_type="electron_diffraction")
        assert tree.root is sig or tree.root.metadata.General.title == "Typed"
        assert tree.root.metadata.General.title == "Typed"


class TestCommitResultTree:
    def _commit(self, session, **kw):
        exx = np.linspace(-0.02, 0.02, 20, dtype=np.float32).reshape(4, 5)
        eyy = -exx
        exy = np.zeros((4, 5), np.float32)
        return commit_result_tree(
            session, title="Strain", primary=exx, primary_label="εxx",
            views=[("εyy", eyy), ("εxy", exy)],
            provenance={"action": "Strain Mapping"}, **kw), exx

    def test_creates_exactly_one_tree_with_views(self, window):
        session = window["window"]
        n0 = len(session.signal_trees)
        (tree, exx) = self._commit(session)
        assert len(session.signal_trees) == n0 + 1

        sp = tree.signal_plots[0]
        wid = sp.window_id
        # Chip views registered with the symmetric locked scale.
        data = views._VIEW_DATA[wid]
        assert data["order"] == ["εxx", "εyy", "εxy"]
        lo, hi = data["levels"]
        assert lo == -hi and hi >= float(np.abs(exx).max())
        # The two extra views were emitted as tagged figures.
        tagged = [m for m in window["messages"]
                  if m.get("type") == "figure" and m.get("view_label") in ("εyy", "εxy")]
        assert len(tagged) == 2
        # Contrast locked (no auto-level) on the primary plot.
        assert sp.needs_auto_level is False

    def test_root_signal_carries_primary_data(self, window):
        session = window["window"]
        (tree, exx) = self._commit(session)
        assert np.allclose(np.asarray(tree.root.data), np.nan_to_num(exx))

    def test_attrs_and_provenance(self, window):
        session = window["window"]
        marker = object()
        tree = commit_result_tree(
            session, title="R", primary=np.zeros((3, 3), np.float32),
            attrs={"vector_orientation": marker},
            provenance={"action": "Vector Orientation Mapping"})
        assert tree.vector_orientation is marker
        assert tree._commit_provenance["action"] == "Vector Orientation Mapping"

    def test_rgb_primary_autolevels_without_clim(self, window):
        session = window["window"]
        rgb = np.zeros((4, 4, 3), np.uint8)
        tree = commit_result_tree(session, title="IPF", primary=rgb)
        sp = tree.signal_plots[0]
        assert sp.needs_auto_level is True

    def test_on_tree_hook_runs_and_failure_is_swallowed(self, window):
        session = window["window"]
        seen = []
        commit_result_tree(session, title="A", primary=np.zeros((2, 2)),
                           on_tree=lambda t: seen.append(t))
        assert len(seen) == 1
        # A raising hook must not break the commit.
        tree = commit_result_tree(session, title="B", primary=np.zeros((2, 2)),
                                  on_tree=lambda t: 1 / 0)
        assert tree is session.signal_trees[-1]

    def test_view_data_evicted_when_window_closes(self, window):
        session = window["window"]
        (tree, _) = self._commit(session)
        wid = tree.signal_plots[0].window_id
        assert wid in views._VIEW_DATA
        session._close_window(wid)
        assert wid not in views._VIEW_DATA

"""ArrayCache locality tag (spyde/array_cache/locality.py + SignalNode.local).

Pins: the ancestry-walk resolver on synthetic SignalNode chains (root
boundary, untagged defaults to opaque, an opaque ancestor poisons a
local-tagged descendant, memoization doesn't re-walk after the tree
mutates), and — through the real action/tree flow — that Rebin2DAction and
CropAction resolve local, and Center Zero Beam resolves local in automatic
(per-pattern) and manual (constant-shift) mode but opaque when
make_flat_field=True (a whole-scan plane fit).
"""
from __future__ import annotations

import time

import numpy as np
import hyperspy.api as hs

from spyde.signal_node import SignalNode
from spyde.array_cache.locality import resolve_locality
from spyde.tests.migrated.conftest import _settle, close_session, make_session
from spyde.tests.migrated._async import wait_until


def _wait(pred, timeout=25.0):
    return wait_until(pred, timeout)


def _node(name, parent, local=None):
    return SignalNode(signal=object(), name=name, parent=parent, local=local)


class TestResolveLocalityStandalone:
    def test_root_is_always_local(self):
        root = _node("root", parent=None)
        assert resolve_locality(root) is True

    def test_chain_of_local_tags_resolves_local(self):
        root = _node("root", parent=None)
        child = _node("child", parent=root, local=True)
        grandchild = _node("grandchild", parent=child, local=True)
        assert resolve_locality(grandchild) is True

    def test_untagged_node_defaults_opaque(self):
        root = _node("root", parent=None)
        child = _node("child", parent=root, local=None)  # never tagged
        assert resolve_locality(child) is False

    def test_untagged_ancestor_poisons_a_locally_tagged_descendant(self):
        root = _node("root", parent=None)
        untagged = _node("mid", parent=root, local=None)   # e.g. console-created
        child = _node("child", parent=untagged, local=True)
        assert resolve_locality(child) is False

    def test_explicit_opaque_ancestor_poisons_a_locally_tagged_descendant(self):
        root = _node("root", parent=None)
        opaque = _node("mid", parent=root, local=False)
        child = _node("child", parent=opaque, local=True)
        assert resolve_locality(child) is False

    def test_memoized_result_does_not_change_after_mutation(self):
        root = _node("root", parent=None)
        child = _node("child", parent=root, local=True)
        assert resolve_locality(child) is True
        # Mutate the tag AFTER resolving — must not affect the cached result
        # (resolved once per view-select, not re-walked per frame).
        child.local = False
        assert resolve_locality(child) is True
        assert child._resolved_local is True

    def test_resolving_a_descendant_also_memoizes_ancestors(self):
        root = _node("root", parent=None)
        mid = _node("mid", parent=root, local=True)
        leaf = _node("leaf", parent=mid, local=True)
        resolve_locality(leaf)
        assert mid._resolved_local is True
        assert root._resolved_local is True


class TestLocalityThroughActions:
    def test_rebin_resolves_local(self, stem_4d_dataset):
        from spyde.actions.base import Rebin2DAction

        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        act = Rebin2DAction.for_plot(plot, scale_x=2, scale_y=2)
        new = act.run()
        _settle(session)
        assert new is not None
        assert tree.resolve_locality(new) is True

    def test_crop_resolves_local(self, stem_4d_dataset):
        from spyde.actions.base import CropAction

        session = stem_4d_dataset["window"]
        tree = session.signal_trees[0]
        plot = next(p for p in session._plots
                    if not p.is_navigator and p.plot_state is not None)
        act = CropAction.for_plot(plot, x0=2, x1=14, y0=4, y1=10)
        new = act.run()
        _settle(session)
        assert new is not None
        assert tree.resolve_locality(new) is True


def _off_center_4d(nav=(3, 3), sig=(32, 32), beam=(18, 14)):
    yy, xx = np.mgrid[0:sig[0], 0:sig[1]]
    disk = ((xx - beam[0]) ** 2 + (yy - beam[1]) ** 2 <= 9).astype(np.float32)
    data = np.zeros(nav + sig, dtype=np.float32)
    for idx in np.ndindex(*nav):
        data[idx] = disk * 100.0
    s = hs.signals.Signal2D(data)
    s.set_signal_type("electron_diffraction")
    return s


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


class TestLocalityThroughCZB:
    def test_auto_per_pattern_resolves_local(self):
        from spyde.actions.center_zero_beam import czb_run

        session = make_session()
        try:
            session._add_signal(_off_center_4d())
            _settle(session)
            src = _signal_plot(session)
            before = src.plot_state.current_signal
            czb_run(session, src, {"method": "center_of_mass"})
            assert _wait(lambda: src.plot_state.current_signal is not before)
            tree = session.signal_trees[0]
            assert tree.resolve_locality(src.plot_state.current_signal) is True
        finally:
            close_session(session)

    def test_auto_flat_field_resolves_opaque(self):
        from spyde.actions.center_zero_beam import czb_run

        session = make_session()
        try:
            session._add_signal(_off_center_4d())
            _settle(session)
            src = _signal_plot(session)
            before = src.plot_state.current_signal
            czb_run(session, src, {"method": "center_of_mass", "make_flat_field": True})
            assert _wait(lambda: src.plot_state.current_signal is not before)
            tree = session.signal_trees[0]
            # A whole-scan plane fit couples every frame's shift together —
            # this specific run must resolve opaque even though CZB is usually local.
            assert tree.resolve_locality(src.plot_state.current_signal) is False
        finally:
            close_session(session)

    def test_manual_constant_shift_resolves_local(self):
        from spyde.actions.center_zero_beam import czb_open, czb_pick

        session = make_session()
        try:
            session._add_signal(_off_center_4d())
            _settle(session)
            src = _signal_plot(session)
            tree = src.signal_tree
            before = src.plot_state.current_signal

            czb_open(session, src, {})
            tree._czb_cross.set(cx=18.0, cy=14.0)
            czb_pick(session, src, {})
            assert _wait(lambda: src.plot_state.current_signal is not before)
            assert tree.resolve_locality(src.plot_state.current_signal) is True
        finally:
            close_session(session)

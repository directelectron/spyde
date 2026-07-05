"""Smoke test: the copyable action skeletons in _template_action.py must keep
importing and following the framework contracts (so the examples can't rot)."""
from __future__ import annotations

import numpy as np


class TestTemplateSkeletons:
    def test_imports_and_shapes(self):
        from spyde.actions import _template_action as t
        from spyde.actions.action import RegionAction, TransformAction
        from spyde.actions.wizard import WizardController

        assert issubclass(t.TemplateTransformAction, TransformAction)
        assert issubclass(t.TemplateRegionAction, RegionAction)
        assert issubclass(t.TemplateWizard, WizardController)
        assert t.TemplateWizard.key == "mywiz"
        for fn in (t.mywiz_open, t.mywiz_close, t.mywiz_commit):
            assert callable(fn)

    def test_wizard_gen_guard_and_idempotent_remove(self):
        from spyde.actions._template_action import TemplateWizard

        tree = type("T", (), {})()
        wiz = TemplateWizard(None, tree)
        g1 = wiz.guard()
        assert wiz.still(g1)
        wiz.cancel_inflight()
        assert not wiz.still(g1)

        tree._mywiz = wiz
        wiz.remove()
        assert wiz._closed and tree._mywiz is None
        wiz.remove()   # idempotent

    def test_wizard_commit_creates_tree(self, window):
        from spyde.actions._template_action import TemplateWizard
        session = window["window"]
        n0 = len(session.signal_trees)
        wiz = TemplateWizard(session, type("T", (), {})())
        wiz.result = np.zeros((4, 4), np.float32)
        tree = wiz.commit()
        assert tree is not None and len(session.signal_trees) == n0 + 1
        assert tree._commit_provenance["action"] == "My Wizard"

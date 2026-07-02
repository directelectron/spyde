"""
test_action_dispatch_state.py — dispatch-side state contracts:

- generically-dispatched Action subclasses store their instance in the action
  artifacts so per-item caret edits (update_vi → update_live_params) reach them
  (previously only the manual VI path set "action" — a latent no-op);
- a toolbar action that raises emits action_active:false so the renderer's
  toggle button doesn't stay lit with no backend artifact behind it.
"""
from __future__ import annotations

import time


def _signal_plot(session):
    for p in session._plots:
        if not p.is_navigator and p.plot_state is not None:
            return p
    return None


class TestActionInstanceTracked:
    def test_generic_dispatch_stores_action_instance(self, stem_4d_dataset):
        from spyde.actions.action import Action
        session = stem_4d_dataset["window"]
        plot = _signal_plot(session)

        session._dispatch_toolbar_action(plot, "FFT", {})
        time.sleep(0.3)

        art = session._action_artifacts.get((plot.window_id, "FFT"))
        assert art is not None, "FFT artifacts not tracked"
        assert isinstance(art.get("action"), Action), (
            "the Action instance must ride with its artifacts (update_vi needs it)")

    def test_update_vi_reaches_generic_action(self, stem_4d_dataset):
        session = stem_4d_dataset["window"]
        plot = _signal_plot(session)
        session._dispatch_toolbar_action(plot, "FFT", {})
        time.sleep(0.3)

        art = session._action_artifacts.get((plot.window_id, "FFT"))
        assert art is not None
        seen = []
        art["action"].update_live_params = lambda params: seen.append(params)

        session._update_vi(plot.window_id, "FFT", {"type": "rectangle"})
        assert seen == [{"type": "rectangle"}]


class TestFailedActionUnlightsButton:
    def test_exception_emits_action_active_false(self, stem_4d_dataset, monkeypatch):
        import spyde
        session = stem_4d_dataset["window"]
        plot = _signal_plot(session)
        messages = stem_4d_dataset["messages"]

        # A throwaway toolbar entry whose function always raises.
        monkeypatch.setitem(
            spyde.TOOLBAR_ACTIONS["functions"], "Boom Action",
            {"function": "spyde.tests.migrated.test_action_dispatch_state._boom"},
        )
        messages.clear()
        session._dispatch_toolbar_action(plot, "Boom Action", {})

        errors = [m for m in messages if m.get("type") == "error"]
        assert errors and "Boom Action" in errors[0]["text"]
        offs = [m for m in messages if m.get("type") == "action_active"
                and m.get("name") == "Boom Action" and m.get("active") is False]
        assert offs, "failed action must emit action_active:false to un-light the button"


def _boom(ctx, action_name=None, **params):
    raise RuntimeError("intentional test failure")

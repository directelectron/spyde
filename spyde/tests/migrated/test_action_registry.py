"""
test_action_registry.py — the staged-action registry (spyde/actions/registry.py),
the window-controller registry on Session, and figure_registry eviction.
"""
from __future__ import annotations

import pytest

from spyde.actions import figure_registry
from spyde.actions.registry import STAGED_HANDLERS, register_staged, resolve_staged


class TestStagedHandlersTable:
    def test_all_entries_are_dotted_paths(self):
        for name, dotted in STAGED_HANDLERS.items():
            mod, _, attr = dotted.rpartition(".")
            assert mod.startswith("spyde."), f"{name}: {dotted}"
            assert attr.isidentifier(), f"{name}: {dotted}"

    def test_resolve_staged_imports_handler(self):
        # Cheap module — log_stream has no heavy deps.
        fn = resolve_staged("set_log_level")
        assert callable(fn)

    def test_resolve_unknown_returns_none(self):
        assert resolve_staged("no_such_action") is None

    def test_register_staged_adds_entry(self):
        register_staged("_test_action_xyz", "spyde.backend.log_stream.set_log_level")
        try:
            assert callable(resolve_staged("_test_action_xyz"))
        finally:
            STAGED_HANDLERS.pop("_test_action_xyz", None)


class _FakeController:
    def __init__(self, window_id):
        self.window_id = window_id
        self.closed = 0

    def close(self):
        self.closed += 1


class TestWindowControllerRegistry:
    def test_register_and_lookup(self, window):
        session = window["window"]
        ctrl = _FakeController(4242)
        session.register_window_controller(4242, ctrl)
        assert session.controller_by_window_id(4242) is ctrl
        assert session.controller_by_window_id(9999) is None
        assert session.controller_by_window_id(None) is None

    def test_forget_window_closes_and_evicts(self, window):
        session = window["window"]
        ctrl = _FakeController(4242)
        session.register_window_controller(4242, ctrl)

        session._forget_window(4242)

        assert ctrl.closed == 1
        assert session.controller_by_window_id(4242) is None
        closed = [m for m in window["messages"]
                  if m.get("type") == "window_closed" and m.get("window_id") == 4242]
        assert closed, "renderer must be told the window is gone"

    def test_close_window_routes_controller_backed_window(self, window):
        """✕ on a bare-figure (non-Plot) window must tear down its controller."""
        session = window["window"]
        ctrl = _FakeController(555)
        session.register_window_controller(555, ctrl)

        session.dispatch_action({"action": "close_window", "window_id": 555})

        assert ctrl.closed == 1
        assert session.controller_by_window_id(555) is None

    def test_controller_close_exception_is_swallowed(self, window):
        session = window["window"]

        class _Boom(_FakeController):
            def close(self):
                super().close()
                raise RuntimeError("boom")

        ctrl = _Boom(7)
        session.register_window_controller(7, ctrl)
        session._forget_window(7)  # must not raise
        assert ctrl.closed == 1


class TestFigureRegistry:
    def test_keep_alive_and_forget(self):
        sentinel = object()
        figure_registry.keep_alive(31337, sentinel)
        assert sentinel in figure_registry._FIGS[31337]
        figure_registry.forget_window(31337)
        assert 31337 not in figure_registry._FIGS

    def test_forget_window_evicts_view_data(self):
        from spyde.actions import views
        views._VIEW_DATA[31338] = {"mats": []}
        figure_registry.keep_alive(31338, object())
        figure_registry.forget_window(31338)
        assert 31338 not in views._VIEW_DATA

    def test_forget_unknown_window_is_noop(self):
        figure_registry.forget_window(999999)  # must not raise

    def test_session_forget_window_evicts_figures(self, window):
        session = window["window"]
        figure_registry.keep_alive(888, object())
        session._forget_window(888)
        assert 888 not in figure_registry._FIGS

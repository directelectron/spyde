"""
test_actions.py — the shell's action framework, with no app around it.

These pin the behaviours the framework exists to guarantee: a worker result
lands on the main thread, a superseded run is recognisable, a replaced overlay
does not stack, and the registry resolves lazily. Each one is something an
action would otherwise re-roll and get subtly wrong.
"""
from __future__ import annotations

import threading

import pytest

from de_shell.actions import figure_registry
from de_shell.actions.lifecycle import (
    bump_generation, is_current, progress_emitter, replace_tree_attr,
    run_on_worker, window_computing,
)
from de_shell.actions.registry import (
    STAGED_HANDLERS, YAML_SCHEMA, register_staged, register_wizard_schema,
    resolve_staged, set_yaml_schema_resolver, wizard_keys, wizard_parameters,
    _WIZARD_SCHEMAS,
)


class _Owner:
    """Stand-in for whatever an app hangs generation counters off."""


class TestRunOnWorker:
    def test_runs_inline_without_a_session(self):
        # A bare test stub has no _dispatch_to_main. Running inline is what lets
        # handler tests observe the result immediately instead of sleeping.
        seen = []
        run_on_worker(None, lambda: 21 * 2, name="t", on_done=seen.append)
        assert seen == [42]

    def test_inline_errors_go_to_on_error_not_on_done(self):
        boom = RuntimeError("nope")
        errs, done = [], []

        def work():
            raise boom

        run_on_worker(None, work, name="t", on_done=done.append, on_error=errs.append)
        assert errs == [boom] and done == []

    def test_marshals_the_result_through_the_session(self):
        calls = []

        class Session:
            def _dispatch_to_main(self, fn):
                calls.append(fn)

        finished = threading.Event()
        seen = []
        run_on_worker(Session(), lambda: "result", name="t",
                      on_done=lambda r: (seen.append(r), finished.set()))
        # The apply is DEFERRED to the main thread, so nothing has run yet.
        for _ in range(200):
            if calls:
                break
            threading.Event().wait(0.01)
        assert calls, "worker never marshalled its result"
        assert seen == []
        calls[0]()
        assert seen == ["result"]

    def test_a_failing_worker_never_calls_on_done(self):
        class Session:
            def _dispatch_to_main(self, fn):
                fn()

        done, errs = [], []
        run_on_worker(Session(), lambda: 1 / 0, name="t",
                      on_done=done.append, on_error=errs.append)
        for _ in range(200):
            if errs:
                break
            threading.Event().wait(0.01)
        assert done == [] and isinstance(errs[0], ZeroDivisionError)


class TestGenerationGuard:
    def test_bump_increments_from_absent(self):
        o = _Owner()
        assert bump_generation(o, "_gen") == 1
        assert bump_generation(o, "_gen") == 2

    def test_only_the_latest_generation_is_current(self):
        o = _Owner()
        first = bump_generation(o, "_gen")
        second = bump_generation(o, "_gen")
        assert not is_current(o, "_gen", first)
        assert is_current(o, "_gen", second)

    def test_unknown_key_is_never_current(self):
        # The close-then-open StrictMode race relies on this: a worker holding a
        # generation for a counter that no longer exists must not proceed.
        assert not is_current(_Owner(), "_never_set", 1)


class TestReplaceTreeAttr:
    def test_removes_the_previous_value_before_attaching(self):
        order = []

        class Overlay:
            def __init__(self, tag): self.tag = tag
            def remove(self): order.append(f"remove-{self.tag}")

        o = _Owner()
        replace_tree_attr(o, "_ov", lambda: Overlay("a"))
        new = replace_tree_attr(o, "_ov", lambda: (order.append("build-b"), Overlay("b"))[1])
        # The old one must go FIRST — otherwise re-running an action stacks markers.
        assert order == ["remove-a", "build-b"]
        assert o._ov is new and new.tag == "b"

    def test_factory_none_just_removes(self):
        class Overlay:
            def remove(self): pass

        o = _Owner()
        replace_tree_attr(o, "_ov", Overlay)
        assert replace_tree_attr(o, "_ov", None) is None
        assert o._ov is None

    def test_a_failing_factory_leaves_none_rather_than_the_stale_value(self):
        class Overlay:
            def remove(self): pass

        o = _Owner()
        replace_tree_attr(o, "_ov", Overlay)

        def boom():
            raise RuntimeError("build failed")

        assert replace_tree_attr(o, "_ov", boom) is None
        assert o._ov is None, "a failed rebuild must not leave the old overlay attached"

    def test_a_failing_remove_does_not_block_the_replacement(self):
        class Bad:
            def remove(self): raise RuntimeError("cannot remove")

        o = _Owner()
        o._ov = Bad()
        assert replace_tree_attr(o, "_ov", lambda: "new") == "new"

    def test_value_without_remove_is_replaced_cleanly(self):
        o = _Owner()
        o._ov = "a plain value"
        assert replace_tree_attr(o, "_ov", lambda: "new") == "new"


class TestProgressAndOverlay:
    def test_first_line_emits_then_throttles_then_always_emits_100(self, monkeypatch):
        # The clock is driven explicitly. Using a huge min_interval instead
        # would make the FIRST call's behaviour depend on machine uptime
        # (the throttle compares against monotonic() with last=0.0), which is a
        # test that passes or fails according to when you last rebooted.
        import de_shell.actions.lifecycle as lc

        class Clock:
            t = 1000.0
            def monotonic(self): return self.t

        clock = Clock()
        monkeypatch.setattr(lc, "time", clock)
        lines = []
        monkeypatch.setattr("de_shell.ipc.emit_status", lines.append)

        progress = progress_emitter("Working", min_interval=0.5)
        progress(1, 10)                      # first call: emits immediately
        clock.t += 0.1
        progress(2, 10)                      # inside the window: throttled
        clock.t += 1.0
        progress(3, 10)                      # window elapsed: emits
        clock.t += 0.01
        progress(10, 10)                     # 100% always emits, throttle or not

        assert lines == ["Working 10%", "Working 30%", "Working 100%"]

    def test_zero_total_emits_nothing(self):
        # Guards a divide-by-zero on a compute that discovered it had no work.
        progress_emitter("Working")(0, 0)

    def test_window_computing_stops_even_on_exception(self, monkeypatch):
        events = []
        monkeypatch.setattr("de_shell.ipc.emit_window_computing",
                            lambda wid, on: events.append((wid, on)))
        with pytest.raises(RuntimeError):
            with window_computing(3):
                raise RuntimeError("compute failed")
        assert events == [(3, True), (3, False)], "the overlay must never stick"


class TestStagedRegistry:
    def test_register_and_resolve_lazily(self):
        register_staged("_t_action", "de_shell.log_stream.set_log_level")
        try:
            assert callable(resolve_staged("_t_action"))
        finally:
            STAGED_HANDLERS.pop("_t_action", None)

    def test_unknown_action_resolves_to_none(self):
        assert resolve_staged("_no_such_action") is None


class TestWizardSchemas:
    def test_schema_from_a_module_dict(self):
        register_wizard_schema("_t_wiz", "de_shell.plotting.colormaps", "COLORMAPS")
        try:
            assert "_t_wiz" in wizard_keys()
            assert wizard_parameters("_t_wiz")   # a copy of the dict
        finally:
            _WIZARD_SCHEMAS.pop("_t_wiz", None)

    def test_yaml_schema_without_a_resolver_fails_loudly(self, monkeypatch):
        # Silently returning {} would render a wizard with NO controls and look
        # like a missing schema rather than a missing wiring step.
        monkeypatch.setattr("de_shell.actions.registry._yaml_resolver", None)
        register_wizard_schema("_t_yaml", YAML_SCHEMA, "Some Action")
        try:
            with pytest.raises(RuntimeError, match="no resolver"):
                wizard_parameters("_t_yaml")
        finally:
            _WIZARD_SCHEMAS.pop("_t_yaml", None)

    def test_yaml_schema_goes_through_the_registered_resolver(self):
        seen = []

        def resolver(title):
            seen.append(title)
            return {"p": {"type": "int"}}

        from de_shell.actions import registry as reg
        prev = reg._yaml_resolver
        set_yaml_schema_resolver(resolver)
        register_wizard_schema("_t_yaml2", YAML_SCHEMA, "Some Action")
        try:
            assert wizard_parameters("_t_yaml2") == {"p": {"type": "int"}}
            assert seen == ["Some Action"]
        finally:
            _WIZARD_SCHEMAS.pop("_t_yaml2", None)
            set_yaml_schema_resolver(prev)

    def test_unknown_wizard_raises(self):
        with pytest.raises(KeyError):
            wizard_parameters("_no_such_wizard")


class TestFigureRegistry:
    def test_keep_alive_then_forget(self):
        sentinel = object()
        figure_registry.keep_alive(31337, sentinel)
        assert sentinel in figure_registry._FIGS[31337]
        figure_registry.forget_window(31337)
        assert 31337 not in figure_registry._FIGS

    def test_forget_unknown_window_is_a_noop(self):
        figure_registry.forget_window(999999)

    def test_registered_evictors_run(self):
        seen = []
        figure_registry.register_evictor(seen.append)
        try:
            figure_registry.forget_window(4242)
            assert seen == [4242]
        finally:
            figure_registry._EVICTORS.remove(seen.append)

    def test_a_throwing_evictor_does_not_break_teardown(self):
        def boom(_wid):
            raise RuntimeError("bookkeeping blew up")

        after = []
        figure_registry.register_evictor(boom)
        figure_registry.register_evictor(after.append)
        try:
            figure_registry.forget_window(4243)   # must not raise
            assert after == [4243], "a failing evictor must not stop the others"
        finally:
            figure_registry._EVICTORS.remove(boom)
            figure_registry._EVICTORS.remove(after.append)

    def test_registering_the_same_evictor_twice_is_idempotent(self):
        seen = []
        figure_registry.register_evictor(seen.append)
        figure_registry.register_evictor(seen.append)
        try:
            figure_registry.forget_window(4244)
            assert seen == [4244], "a module importing twice must not double-evict"
        finally:
            figure_registry._EVICTORS.remove(seen.append)

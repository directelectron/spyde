"""
Regression: anyplotlib's add_event_handler does ``fn._event_types = …`` on the
handler. Passing a BOUND METHOD raises ``'method' object has no attribute
'_event_types'`` → the selector widget init failed silently (no crosshair, dead
navigator, and _on_signal_ready crashed on a None parent_selector).

event_handler_fn() wraps a bound method in a plain function so the attribute can
be set. These tests use a faithful fake widget that, like real anyplotlib, sets
_event_types on the handler — so a bound method WOULD blow up without the wrapper.
"""
import functools

import pytest

from spyde.drawing.selectors.base_selector import event_handler_fn


class _FaithfulWidget:
    """Mimics anyplotlib: add_event_handler tags the handler with _event_types."""
    def __init__(self):
        self.handlers = []

    def add_event_handler(self, fn, *types, **kw):
        # This is the exact line that breaks on a bound method.
        fn._event_types = getattr(fn, "_event_types", set()) | set(types)
        self.handlers.append((fn, types))
        return fn


def test_bound_method_raises_on_faithful_widget():
    """Confirms the failure mode the wrapper exists to prevent."""
    class Obj:
        def _on(self, ev):
            return ev
    w = _FaithfulWidget()
    with pytest.raises(AttributeError):
        w.add_event_handler(Obj()._on, "pointer_up")


def test_event_handler_fn_makes_bound_method_registerable():
    class Obj:
        def __init__(self):
            self.calls = []
        def _on(self, ev):
            self.calls.append(ev)

    obj = Obj()
    w = _FaithfulWidget()
    cb = event_handler_fn(obj._on)          # wrap the bound method
    w.add_event_handler(cb, "pointer_move", "pointer_up")   # must NOT raise
    assert cb._event_types == {"pointer_move", "pointer_up"}

    # And it still dispatches to the original bound method.
    w.handlers[0][0]("EV")
    assert obj.calls == ["EV"]


def test_wrapper_preserves_name():
    class Obj:
        def _on_pointer_up(self, ev): ...
    cb = event_handler_fn(Obj()._on_pointer_up)
    assert cb.__name__ == "_on_pointer_up"

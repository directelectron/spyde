"""Stacked 1-D navigator lanes: the held lane is never pulled back mid-drag.

The failure this pins is an interaction bug you cannot see in a screenshot and
cannot see in a synchronous unit test either — it needs the write-back to
arrive AFTER the drag handler has returned, which is exactly what
``_dispatch_to_main`` does in the app.

The selector's index hook fires on the ``_NavDispatcher`` thread and is
marshalled onto the main thread, landing some milliseconds later. If that
write-back put the committed (index-quantised, and by then stale) position onto
the lane still under the pointer, the lane would snap back, the next
``pointer_move`` would drag it forward again, and the cursor would oscillate for
the whole drag. ``_LinkedNavCursor`` reads the selector's own widget — the
master — when the write-back lands, so every lane takes the latest position.
"""
from __future__ import annotations

import numpy as np
import pytest

from spyde.actions.navigator_views import _LinkedNavCursor


class _FakeWidget:
    """A VLine stand-in. A notifying ``set`` fires ``pointer_move`` like the
    real one; ``_notify=False`` does not."""

    def __init__(self, x=0.0):
        self._x = float(x)
        self._handlers: dict[str, list] = {}
        self._plot = object()

    def add_event_handler(self, fn, event_type):
        self._handlers.setdefault(event_type, []).append(fn)

    def get(self, key):
        assert key == "x"
        return self._x

    @property
    def x(self):
        return self._x

    def set(self, x=None, _notify=True, **_kw):
        if x is None:
            return
        self._x = float(x)
        if _notify:
            self._fire("pointer_move")

    def _fire(self, event_type):
        for fn in list(self._handlers.get(event_type, ())):
            fn()

    def drag_to(self, x: float):
        self._x = float(x)
        self._fire("pointer_move")

    def release(self):
        self._fire("pointer_up")


class _FakeSelector:
    def __init__(self):
        self.index_hooks: list = []
        self._widget = _FakeWidget()
        self.updates = 0

    def delayed_update_data(self, force=False):
        self.updates += 1

    def commit(self, index):
        """The dispatcher committing a frame: fire the hooks, as it does."""
        for hook in list(self.index_hooks):
            hook(np.array([index]))


class _FakeSession:
    """Defers marshalled work, so a test can land it at a chosen moment —
    the app's ``_dispatch_to_main`` is likewise asynchronous."""

    def __init__(self):
        self.pending: list = []

    def _dispatch_to_main(self, fn):
        self.pending.append(fn)

    def flush(self):
        pending, self.pending = self.pending, []
        for fn in pending:
            fn()


@pytest.fixture
def stacked():
    session = _FakeSession()
    widgets = [_FakeWidget(), _FakeWidget(), _FakeWidget()]
    sel = _FakeSelector()
    cursor = _LinkedNavCursor(session, sel, widgets, ("x",))
    return session, widgets, sel, cursor


class TestHeldLane:
    def test_drag_mirrors_to_the_other_lanes(self, stacked):
        _session, widgets, sel, _cursor = stacked
        widgets[0].drag_to(4.2)
        assert [w.x for w in widgets] == [4.2, 4.2, 4.2]
        assert sel._widget.x == pytest.approx(4.2)
        assert sel.updates == 1

    def test_late_writeback_does_not_move_the_held_line(self, stacked):
        """THE regression: the index hook lands after the handler returned."""
        session, widgets, sel, _cursor = stacked

        widgets[0].drag_to(4.2)   # user drags lane 0 to 4.2
        sel.commit(128)           # the dispatcher commits frame 128
        widgets[0].drag_to(4.9)   # user keeps dragging BEFORE the write-back lands
        session.flush()           # …and now it lands

        assert widgets[0].x == pytest.approx(4.9), (
            "the held line was yanked back to the committed position"
        )
        # The followers take the master's latest position, not the stale index.
        assert widgets[1].x == pytest.approx(4.9)
        assert widgets[2].x == pytest.approx(4.9)

    def test_release_settles_every_line(self, stacked):
        session, widgets, sel, _cursor = stacked
        widgets[0].drag_to(4.2)
        widgets[0].release()
        sel.commit(128)
        session.flush()
        assert [w.x for w in widgets] == [pytest.approx(4.2)] * 3

    def test_programmatic_move_with_no_drag_syncs_every_line(self, stacked):
        """Playback moves the selector with nobody dragging — all lanes follow."""
        session, widgets, sel, _cursor = stacked
        sel._widget.set(x=64 * 0.03276, _notify=False)
        sel.commit(64)
        session.flush()
        assert [w.x for w in widgets] == [pytest.approx(64 * 0.03276)] * 3

    def test_drag_does_not_re_enter_via_the_mirror(self, stacked):
        """Mirroring onto the other lanes must not echo back as their drags."""
        _session, widgets, sel, _cursor = stacked
        widgets[2].drag_to(1.5)
        assert sel.updates == 1

    def test_close_stops_syncing(self, stacked):
        session, widgets, sel, cursor = stacked
        widgets[0].drag_to(4.2)
        sel.commit(128)
        cursor.close()
        assert sel.index_hooks == []
        widgets[1].set(x=0.0, _notify=False)
        session.flush()
        assert widgets[1].x == 0.0

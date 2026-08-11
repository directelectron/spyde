"""
test_session_base.py — SessionBase, tested with NO app around it.

That is the point of these tests as much as the assertions are: they construct a
shell Session with nothing but a temp directory, so if SessionBase ever grows a
dependency on SpyDE (or on hyperspy/dask arriving through some import chain),
this file stops working long before de-groundcrew does.
"""
from __future__ import annotations

import json
import os

import pytest

from de_shell.session import SessionBase


@pytest.fixture
def session(tmp_path):
    return SessionBase(settings_dir=str(tmp_path))


class TestWindowRegistry:
    def test_window_ids_are_sequential_and_unique(self, session):
        assert [session.next_window_id() for _ in range(4)] == [0, 1, 2, 3]

    def test_register_and_unregister_plot(self, session):
        a, b = object(), object()
        session.register_plot(a)
        session.register_plot(b)
        assert session._plots == [a, b]
        session.unregister_plot(a)
        assert session._plots == [b]

    def test_unregister_is_identity_based_and_removes_every_copy(self, session):
        # register_plot does not dedupe, so unregister must not stop at the
        # first match — a double-registered plot left behind would keep being
        # polled after its window closed.
        class Equal:
            def __eq__(self, other): return True
            def __hash__(self): return 0
        a = Equal()
        b = Equal()
        session.register_plot(a)
        session.register_plot(a)
        session.register_plot(b)
        session.unregister_plot(a)
        assert len(session._plots) == 1 and session._plots[0] is b

    def test_unregister_unknown_plot_is_a_no_op(self, session):
        session.register_plot("keep")
        session.unregister_plot("never registered")
        assert session._plots == ["keep"]

    def test_plot_by_window_id(self, session):
        class P:
            def __init__(self, wid): self.window_id = wid
        p = P(7)
        session.register_plot(P(3))
        session.register_plot(p)
        assert session._plot_by_window_id(7) is p
        assert session._plot_by_window_id(99) is None

    def test_window_controller_registry(self, session):
        ctl = object()
        session.register_window_controller(5, ctl)
        assert session.controller_by_window_id(5) is ctl
        assert session.controller_by_window_id(6) is None
        # A None window id is the "no window in context" case, not a lookup.
        assert session.controller_by_window_id(None) is None


class TestMainLoopMarshalling:
    def test_runs_inline_when_no_loop_registered(self, session):
        # The property that makes a Session usable in a plain test: a worker
        # callback fired before the asyncio loop exists must still run, not be
        # silently dropped.
        seen = []
        session._dispatch_to_main(lambda: seen.append(1))
        assert seen == [1]

    def test_marshals_through_the_registered_loop(self, session):
        calls = []

        class FakeLoop:
            def call_soon_threadsafe(self, fn):
                calls.append(fn)

        session.set_main_loop(FakeLoop())
        ran = []
        session._dispatch_to_main(lambda: ran.append(1))
        assert ran == [] and len(calls) == 1   # deferred, not run inline
        calls[0]()
        assert ran == [1]

    def test_falls_back_inline_when_the_loop_rejects(self, session):
        # A closed loop raises from call_soon_threadsafe. Losing the callback
        # would strand whatever it was going to paint or free.
        class DeadLoop:
            def call_soon_threadsafe(self, fn):
                raise RuntimeError("event loop is closed")

        session.set_main_loop(DeadLoop())
        ran = []
        session._dispatch_to_main(lambda: ran.append(1))
        assert ran == [1]


class TestSettings:
    def test_missing_settings_file_is_not_an_error(self, tmp_path):
        s = SessionBase(settings_dir=str(tmp_path / "does-not-exist"))
        assert s._settings == {}
        assert s.get_recent_files() == []
        assert s.first_run is True

    def test_corrupt_settings_file_is_not_an_error(self, tmp_path):
        (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")
        s = SessionBase(settings_dir=str(tmp_path))
        assert s._settings == {}

    def test_recent_files_round_trip_and_are_most_recent_first(self, tmp_path):
        s = SessionBase(settings_dir=str(tmp_path))
        s._add_recent("/a")
        s._add_recent("/b")
        assert s.get_recent_files() == ["/b", "/a"]
        assert SessionBase(settings_dir=str(tmp_path)).get_recent_files() == ["/b", "/a"]

    def test_re_adding_moves_to_front_without_duplicating(self, tmp_path):
        s = SessionBase(settings_dir=str(tmp_path))
        for p in ("/a", "/b", "/c"):
            s._add_recent(p)
        s._add_recent("/a")
        assert s.get_recent_files() == ["/a", "/c", "/b"]

    def test_recent_files_are_capped(self, tmp_path):
        s = SessionBase(settings_dir=str(tmp_path))
        for i in range(SessionBase.MAX_RECENT + 10):
            s._add_recent(f"/f{i}")
        assert len(s.get_recent_files()) == SessionBase.MAX_RECENT
        on_disk = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
        assert len(on_disk["recent_files"]) == SessionBase.MAX_RECENT

    def test_settings_dir_is_created_on_first_write(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        s = SessionBase(settings_dir=str(nested))
        s._add_recent("/a")
        assert os.path.exists(nested / "settings.json")

    def test_update_channel_persists_and_rejects_junk(self, tmp_path):
        s = SessionBase(settings_dir=str(tmp_path))
        assert s._update_channel == "stable"
        s.set_update_channel("beta")
        assert SessionBase(settings_dir=str(tmp_path))._update_channel == "beta"
        s.set_update_channel("nightly")            # ignored, not persisted
        assert s._update_channel == "beta"

    def test_garbage_channel_on_disk_falls_back_to_stable(self, tmp_path):
        (tmp_path / "settings.json").write_text(
            json.dumps({"update_channel": "nightly"}), encoding="utf-8")
        assert SessionBase(settings_dir=str(tmp_path))._update_channel == "stable"

    def test_first_run_flips_once_and_is_idempotent(self, tmp_path):
        s = SessionBase(settings_dir=str(tmp_path))
        assert s.first_run is True
        s.mark_tutorial_seen()
        s.mark_tutorial_seen()
        assert s.first_run is False
        assert SessionBase(settings_dir=str(tmp_path)).first_run is False


class TestShutdown:
    def test_is_idempotent_and_clears_controllers(self, session):
        session.register_window_controller(1, object())
        session.shutdown()
        session.shutdown()
        assert session._closed is True
        assert session._window_controllers == {}

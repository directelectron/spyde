"""
test_first_run.py — tutorial_seen settings round-trip + the get_first_run /
mark_tutorial_seen staged handlers (Phase 4 of the docs overhaul: the
first-run welcome walkthrough). Mirrors test_update_gpu_status.py's shape
(same settings.json key, same isolation pattern) — see that file's
_isolate_settings docstring.
"""
from __future__ import annotations

import json
import os

from spyde.actions.registry import STAGED_HANDLERS


def _isolate_settings(session, tmp_path):
    """Point Session._settings_path at a throwaway file so the test never
    touches the real user's ~/.spyde/settings.json."""
    session._settings_path = os.path.join(tmp_path, ".spyde", "settings.json")
    session._settings = {}


class TestFirstRunFlag:
    def test_first_run_true_by_default(self, window):
        session = window["window"]
        assert session.first_run is True

    def test_mark_tutorial_seen_persists(self, window, tmp_path):
        session = window["window"]
        _isolate_settings(session, tmp_path)
        assert session.first_run is True

        session.mark_tutorial_seen()

        assert session.first_run is False
        assert session._settings["tutorial_seen"] is True
        with open(session._settings_path, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["tutorial_seen"] is True

    def test_mark_tutorial_seen_idempotent(self, window, tmp_path):
        session = window["window"]
        _isolate_settings(session, tmp_path)

        session.mark_tutorial_seen()
        mtime1 = os.path.getmtime(session._settings_path)
        session.mark_tutorial_seen()  # second call should be a cheap no-op
        mtime2 = os.path.getmtime(session._settings_path)

        assert session.first_run is False
        assert mtime1 == mtime2

    def test_restores_persisted_flag_on_restart(self, tmp_path, monkeypatch):
        """A fresh Session should read back a previously-persisted flag."""
        from spyde.backend.session import Session

        settings_path = os.path.join(tmp_path, ".spyde", "settings.json")
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, "w", encoding="utf-8") as fh:
            json.dump({"tutorial_seen": True}, fh)

        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            assert session.first_run is False
        finally:
            session.shutdown()

    def test_dispatch_mark_tutorial_seen_routes(self, window, tmp_path):
        """The renderer sends {action: 'mark_tutorial_seen'} when it auto-opens
        (or the user reopens from Help) the welcome tour."""
        session = window["window"]
        _isolate_settings(session, tmp_path)

        session.dispatch_action({"action": "mark_tutorial_seen", "payload": {}})

        assert session.first_run is False


class TestGetFirstRunAction:
    def test_staged_handler_registered(self):
        assert "get_first_run" in STAGED_HANDLERS
        assert "mark_tutorial_seen" in STAGED_HANDLERS

    def test_emits_first_run_result_true(self, window, tmp_path):
        session = window["window"]
        _isolate_settings(session, tmp_path)

        session.dispatch_action({"action": "get_first_run", "payload": {}})

        results = [m for m in window["messages"] if m.get("type") == "first_run_result"]
        assert len(results) == 1
        assert results[-1]["first_run"] is True

    def test_emits_first_run_result_false_after_marking(self, window, tmp_path):
        session = window["window"]
        _isolate_settings(session, tmp_path)
        session.mark_tutorial_seen()

        session.dispatch_action({"action": "get_first_run", "payload": {}})

        results = [m for m in window["messages"] if m.get("type") == "first_run_result"]
        assert results[-1]["first_run"] is False

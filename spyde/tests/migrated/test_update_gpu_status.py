"""
test_update_gpu_status.py — update_channel/skip_version settings round-trip and
the get_gpu_status staged handler (Help -> GPU Status / Check for Updates).
"""
from __future__ import annotations

import json
import os

import pytest

from spyde.actions.registry import STAGED_HANDLERS


def _isolate_settings(session, tmp_path):
    """Point Session._settings_path at a throwaway file so the test never
    touches the real user's ~/.spyde/settings.json."""
    session._settings_path = os.path.join(tmp_path, ".spyde", "settings.json")
    session._settings = {}


class TestUpdateChannelSettings:
    def test_defaults_to_stable(self, window):
        session = window["window"]
        assert session._update_channel == "stable"

    def test_set_update_channel_persists(self, window, tmp_path):
        session = window["window"]
        _isolate_settings(session, tmp_path)

        session.set_update_channel("beta")

        assert session._update_channel == "beta"
        with open(session._settings_path, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["update_channel"] == "beta"

    def test_set_update_channel_rejects_invalid(self, window, tmp_path):
        session = window["window"]
        _isolate_settings(session, tmp_path)

        session.set_update_channel("nightly")

        assert session._update_channel == "stable"
        assert not os.path.exists(session._settings_path)

    def test_dispatch_action_routes_to_settings(self, window, tmp_path):
        """The renderer sends {action: 'set_update_channel', payload: {channel: 'beta'}}."""
        session = window["window"]
        _isolate_settings(session, tmp_path)

        session.dispatch_action({"action": "set_update_channel", "payload": {"channel": "beta"}})

        assert session._update_channel == "beta"

    def test_restores_persisted_channel_on_restart(self, tmp_path, monkeypatch):
        """A fresh Session should read back a previously-persisted channel."""
        from spyde.backend.session import Session

        settings_path = os.path.join(tmp_path, ".spyde", "settings.json")
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, "w", encoding="utf-8") as fh:
            json.dump({"update_channel": "beta"}, fh)

        monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path))
        session = Session(n_workers=1, threads_per_worker=1)
        try:
            assert session._update_channel == "beta"
        finally:
            session.shutdown()


class TestGpuStatusAction:
    def test_staged_handler_registered(self):
        assert "get_gpu_status" in STAGED_HANDLERS

    def test_emits_gpu_status_result(self, window):
        session = window["window"]
        session.dispatch_action({"action": "get_gpu_status", "payload": {}})

        results = [m for m in window["messages"] if m.get("type") == "gpu_status_result"]
        assert len(results) == 1
        result = results[0]
        assert "torch_available" in result
        assert "gpu_available" in result
        assert "reason" in result
        assert isinstance(result["reason"], str) and result["reason"]

    def test_reports_no_gpu_without_torch(self, window, monkeypatch):
        import spyde.actions.vector_orientation_gpu as vog

        monkeypatch.setattr(vog, "torch_available", lambda: False)
        monkeypatch.setattr(vog, "gpu_available", lambda: False)
        monkeypatch.setattr(vog, "gpu_unavailable_reason", lambda: "torch not importable (mocked)")

        session = window["window"]
        session.dispatch_action({"action": "get_gpu_status", "payload": {}})

        results = [m for m in window["messages"] if m.get("type") == "gpu_status_result"]
        assert results[-1]["torch_available"] is False
        assert results[-1]["gpu_available"] is False
        assert results[-1]["reason"] == "torch not importable (mocked)"

"""Tests for the app-log IPC stream (spyde.backend.log_stream)."""
import logging

import pytest

from spyde.backend import log_stream
from spyde.backend.log_stream import IPCLogHandler


@pytest.fixture
def captured(monkeypatch):
    """Collect every message the handler emits over the IPC channel."""
    msgs = []
    monkeypatch.setattr("spyde.backend.ipc.emit", lambda obj: msgs.append(obj))
    return msgs


def _logger_with(handler, name="spyde.unit", level=logging.DEBUG):
    lg = logging.getLogger(name)
    lg.handlers = [handler]
    lg.setLevel(level)
    lg.propagate = False
    return lg


class TestIPCLogHandler:
    def test_record_becomes_log_message(self, captured):
        h = IPCLogHandler(level=logging.DEBUG)
        _logger_with(h).info("hello %d", 42)
        assert len(captured) == 1
        m = captured[0]
        assert m["type"] == "log"
        assert m["level"] == "INFO"
        assert m["name"] == "spyde.unit"
        assert m["msg"] == "hello 42"
        assert isinstance(m["time"], float)

    def test_handler_level_gates_records(self, captured):
        h = IPCLogHandler(level=logging.INFO)
        lg = _logger_with(h)
        lg.debug("too quiet")          # below handler level → dropped
        lg.warning("loud enough")
        assert [m["level"] for m in captured] == ["WARNING"]

    def test_thirdparty_info_is_filtered_but_warning_passes(self, captured):
        h = IPCLogHandler(level=logging.DEBUG)
        lg = _logger_with(h, name="distributed.worker")
        lg.info("library chatter")     # third-party INFO → dropped
        lg.warning("library problem")  # warnings always pass
        assert [m["level"] for m in captured] == ["WARNING"]

    def test_spyde_info_passes(self, captured):
        h = IPCLogHandler(level=logging.DEBUG)
        _logger_with(h, name="spyde.actions.find_vectors").info("found 5")
        assert [m["msg"] for m in captured] == ["found 5"]

    def test_exception_traceback_included(self, captured):
        h = IPCLogHandler(level=logging.DEBUG)
        lg = _logger_with(h)
        try:
            raise ValueError("boom")
        except ValueError:
            lg.exception("it failed")
        assert len(captured) == 1
        assert "it failed" in captured[0]["msg"]
        assert "Traceback" in captured[0]["msg"]
        assert "ValueError: boom" in captured[0]["msg"]

    def test_ring_buffer_bounded(self, captured):
        h = IPCLogHandler(level=logging.DEBUG, maxlen=5)
        lg = _logger_with(h)
        for i in range(20):
            lg.info("line %d", i)
        assert len(h.buffer) == 5
        assert h.buffer[-1]["msg"] == "line 19"

    def test_reentrancy_guard_blocks_recursion(self, monkeypatch):
        # If emit() itself logs (e.g. on failure), the guard must stop the record
        # from recursing back into the handler.
        h = IPCLogHandler(level=logging.DEBUG)
        lg = _logger_with(h)
        calls = {"n": 0}

        def _reentrant_emit(obj):
            calls["n"] += 1
            lg.error("logging from inside emit")   # would recurse without guard
        monkeypatch.setattr("spyde.backend.ipc.emit", _reentrant_emit)
        lg.info("trigger")
        assert calls["n"] == 1                     # exactly one emit, no recursion


class TestLevelControl:
    def test_set_level_changes_handler_and_root(self):
        h = IPCLogHandler(level=logging.INFO)
        root_before = logging.getLogger().level
        try:
            log_stream._handler = h
            lv = log_stream.set_level("DEBUG")
            assert lv == logging.DEBUG
            assert h.level == logging.DEBUG
            assert logging.getLogger().level == logging.DEBUG
        finally:
            log_stream._handler = None
            logging.getLogger().setLevel(root_before)

    def test_set_log_level_staged_handler_backfills(self, captured):
        h = IPCLogHandler(level=logging.DEBUG)
        _logger_with(h).info("earlier line")       # lands in the ring buffer
        root_before = logging.getLogger().level
        try:
            log_stream._handler = h
            captured.clear()
            log_stream.set_log_level(None, None, {"level": "WARNING"})
            kinds = [m["type"] for m in captured]
            assert "log_backfill" in kinds
            assert "log_level" in kinds
            backfill = next(m for m in captured if m["type"] == "log_backfill")
            assert backfill["entries"][-1]["msg"] == "earlier line"
            lvl = next(m for m in captured if m["type"] == "log_level")
            assert lvl["level"] == "WARNING"
        finally:
            log_stream._handler = None
            logging.getLogger().setLevel(root_before)


class TestInstall:
    def test_install_is_idempotent(self):
        root = logging.getLogger()
        n_before = len(root.handlers)
        root_level_before = root.level
        try:
            h1 = log_stream.install(level="INFO")
            h2 = log_stream.install(level="DEBUG")
            assert h1 is h2                         # singleton
            assert root.handlers.count(h1) == 1     # attached exactly once
        finally:
            root.removeHandler(log_stream._handler)
            log_stream._handler = None
            root.setLevel(root_level_before)
            assert len(root.handlers) == n_before

"""Distribution layer: version single-source, build info, updater, gpu_setup.

These are pure-logic / offline tests (no Qt fixture, no network) — the live
GitHub check and the uv install paths are exercised by mocking.
"""
import sys

import pytest


# ── version single-source ────────────────────────────────────────────────────

def test_version_single_source():
    import spyde
    from spyde._version import __version__ as v
    assert spyde.__version__ == v
    # pyproject reads spyde._version.__version__ — make sure it's a real semver
    parts = v.split(".")
    assert len(parts) >= 3 and all(p[0].isdigit() for p in parts[:3])


def test_build_info_dev_fallback():
    from spyde._build_info import build_info, version_string
    info = build_info()
    assert set(info) == {"version", "git_sha", "channel", "build_date"}
    assert info["version"]                # non-empty
    assert "SpyDE" in version_string()


# ── updater semver ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("cand,cur,expected", [
    ("v0.2.0", "0.1.0", True),
    ("0.1.1", "0.1.0", True),
    ("0.1.0", "0.1.0", False),
    ("0.1.0", "0.2.0", False),
    ("0.1.0-rc.1", "0.1.0", False),     # rc is older than final
    ("0.2.0", "0.1.0-rc.1", True),
    ("garbage", "0.1.0", False),
])
def test_is_newer(cand, cur, expected):
    from spyde.updater import is_newer
    assert is_newer(cand, cur) is expected


def test_parse_version_handles_v_prefix_and_pre():
    from spyde.updater import parse_version
    assert parse_version("v1.2.3")[:3] == (1, 2, 3)
    assert parse_version("1.2.3")[3] == float("inf")     # final sorts last
    assert parse_version("1.2.3-rc.2")[3] == 2
    assert parse_version("nope") is None


def test_check_handles_offline(monkeypatch):
    # force the network call to fail → graceful error, available False
    import spyde.updater as up

    def _boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(up.urllib.request, "urlopen", _boom)
    info = up.check()
    assert info.available is False
    assert info.error is not None


def test_check_finds_newer(monkeypatch):
    import io
    import json
    import spyde.updater as up

    releases = [
        {"tag_name": "v0.1.0", "html_url": "u0", "body": "old"},
        {"tag_name": "v9.9.9", "html_url": "u1", "body": "shiny"},
        {"tag_name": "v0.5.0-rc.1", "prerelease": True, "html_url": "u2"},
    ]

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=0):
        return _Resp(json.dumps(releases).encode())

    monkeypatch.setattr(up.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(up, "_current_version", lambda: "0.1.0")
    monkeypatch.setattr(up, "_current_channel", lambda: "stable")
    info = up.check()
    assert info.available is True
    assert info.latest == "v9.9.9"        # picks newest, skips the rc on stable
    assert info.url == "u1"


# ── gpu_setup ────────────────────────────────────────────────────────────────

def test_gpu_detect_shape():
    from spyde import gpu_setup
    d = gpu_setup.detect()
    for key in ("platform", "machine", "nvidia", "torch", "backend",
                "accelerated", "needs_gpu_wheel"):
        assert key in d
    assert d["backend"] in ("cuda", "mps", "cpu")
    assert isinstance(d["accelerated"], bool)


def test_gpu_summary_is_ascii():
    # summary_lines go to logs / a plain console — must be cp1252-safe on Windows
    from spyde import gpu_setup
    for line in gpu_setup.summary_lines():
        line.encode("cp1252")            # raises if a non-encodable glyph slipped in


def test_ensure_backend_noop_when_correct(monkeypatch):
    from spyde import gpu_setup
    monkeypatch.setattr(gpu_setup, "detect",
                        lambda: {"needs_gpu_wheel": False, "accelerated": True})
    res = gpu_setup.ensure_backend()
    assert res["ran"] is False and res["ok"] is True

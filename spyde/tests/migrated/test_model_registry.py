"""Registry hardening: ``ensure_local`` / ``is_cached`` / sha256 verification.

Pure file-level tests — no torch import, no network (bundled sources only).
The model-LOADING side (get_model, arch resolution, user-manifest merge) is
covered by ``test_neural_detect.py``'s subprocess driver.
"""
from __future__ import annotations

import hashlib
import os

import pytest


class TestModelRegistry:
    def test_ensure_local_bundled(self):
        from spyde.models import registry
        p = registry.ensure_local(None)              # default → bundled weight
        assert p and os.path.exists(p)

    def test_ensure_local_unknown_id(self):
        from spyde.models import registry
        assert registry.ensure_local("no-such-model") is None

    def test_is_cached_bundled(self):
        from spyde.models import registry
        assert registry.is_cached(None) is True
        assert registry.is_cached("spotunet-base16-v1") is True
        # Unknown ids resolve to the bundled default → present.
        assert registry.is_cached("no-such-model") is True

    def test_sha256_verify(self, tmp_path):
        from spyde.models import registry
        f = tmp_path / "w.pt"
        f.write_bytes(b"weights")
        good = hashlib.sha256(b"weights").hexdigest()
        registry._verify_sha256(str(f), good, "m")            # no raise
        registry._verify_sha256(str(f), good.upper(), "m")    # case-insensitive
        with pytest.raises(ValueError):
            registry._verify_sha256(str(f), "0" * 64, "m")

    def test_resolve_weights_checks_sha(self, tmp_path, monkeypatch):
        """A registry entry carrying ``sha256`` rejects a tampered file (get_model
        then falls back to the bundled default)."""
        from spyde.models import registry
        f = tmp_path / "w.pt"
        f.write_bytes(b"payload")
        monkeypatch.setattr(registry, "_resolve_bundled", lambda src: str(f))
        entry = {"id": "x", "sha256": "0" * 64,
                 "source": {"type": "bundled", "file": "w.pt"}}
        with pytest.raises(ValueError):
            registry._resolve_weights(entry)
        entry["sha256"] = hashlib.sha256(b"payload").hexdigest()
        assert registry._resolve_weights(entry) == str(f)

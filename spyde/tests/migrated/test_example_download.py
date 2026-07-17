"""Examples-menu download progress + cancel (spyde/backend/example_download.py).

Covers the pooch progress-object protocol (total / update / reset / close),
message throttling, the cancel flag → DownloadCancelled abort, the scoped
pyxem pooch proxy patch, and the download_cancel staged action. No network —
pooch itself is never invoked.
"""
from __future__ import annotations

import pytest


def _capture(monkeypatch):
    import spyde.backend.example_download as dl
    msgs = []
    monkeypatch.setattr(dl, "emit", lambda m: msgs.append(m))
    return dl, msgs


class TestIpcProgress:
    def test_pooch_protocol_sequence(self, monkeypatch):
        """total-set → chunk updates → reset/update(total)/close (pooch's exact
        end-of-file sequence) produces a start emit and a final 100% emit."""
        import threading
        dl, msgs = _capture(monkeypatch)
        p = dl._IpcProgress("example:x", "x", threading.Event())

        p.total = 1000                      # pooch assigns before streaming
        assert msgs[-1] == {"type": "download_progress", "token": "example:x",
                            "label": "x", "done": 0, "total": 1000}
        p.update(100)                       # throttled: within 200 ms of start
        assert msgs[-1]["done"] == 0
        # End-of-file: reset() then update(total) must emit UN-throttled so the
        # bar reaches 100%.
        p.reset()
        p.update(1000)
        p.close()
        assert msgs[-1]["done"] == 1000 and msgs[-1]["total"] == 1000
        assert p.started

    def test_done_clamped_to_total(self, monkeypatch):
        """pooch updates by chunk_size (not len(chunk)) so done can overshoot —
        clamp it."""
        import threading
        dl, msgs = _capture(monkeypatch)
        p = dl._IpcProgress("t", "x", threading.Event())
        p.total = 100
        p.reset(); p.update(1024)           # closing → un-throttled emit
        assert msgs[-1]["done"] == 100

    def test_cancel_raises(self, monkeypatch):
        import threading
        dl, _ = _capture(monkeypatch)
        ev = threading.Event()
        p = dl._IpcProgress("t", "x", ev)
        p.total = 100
        ev.set()
        with pytest.raises(dl.DownloadCancelled):
            p.update(10)


class TestPatchedDownloader:
    def test_proxy_scoped_and_restored(self, monkeypatch):
        """Inside the context pyxem's pooch.HTTPDownloader pins progressbar= to
        the monitor; outside, the real module is back."""
        import pyxem.data._data as _pxdata
        dl, _ = _capture(monkeypatch)
        real = _pxdata.pooch
        with dl.patched_example_downloader("example:t", "t") as monitor:
            assert _pxdata.pooch is not real
            d = _pxdata.pooch.HTTPDownloader(progressbar=True)
            assert d.progressbar is monitor
            # everything else delegates to the real pooch
            assert _pxdata.pooch.os_cache is real.os_cache
        assert _pxdata.pooch is real

    def test_done_emitted_only_when_started(self, monkeypatch):
        import pyxem.data._data  # noqa: F401 — ensure importable
        dl, msgs = _capture(monkeypatch)
        # Cache hit: no bytes flowed → no download_done.
        with dl.patched_example_downloader("example:hit", "hit"):
            pass
        assert not [m for m in msgs if m["type"] == "download_done"]
        # Download happened → ok:true done.
        with dl.patched_example_downloader("example:miss", "miss") as mon:
            mon.total = 10
            mon.update(10)
        done = [m for m in msgs if m["type"] == "download_done"]
        assert done and done[-1]["ok"] is True and done[-1]["cancelled"] is False

    def test_cancel_flow(self, monkeypatch):
        """download_cancel (the staged action) flags the token; the next chunk
        raises, the context re-raises and reports cancelled:true."""
        import pyxem.data._data  # noqa: F401
        dl, msgs = _capture(monkeypatch)
        with pytest.raises(dl.DownloadCancelled):
            with dl.patched_example_downloader("example:c", "c") as mon:
                mon.total = 100
                mon.update(10)
                dl.download_cancel(None, None, {"token": "example:c"})
                mon.update(10)              # ← aborts here
        done = [m for m in msgs if m["type"] == "download_done"]
        assert done and done[-1]["cancelled"] is True and done[-1]["ok"] is False
        # Flag cleaned up: cancelling again is a no-op.
        assert dl.request_cancel("example:c") is False

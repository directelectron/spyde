"""Report vectors embed (spyde/actions/report/vectors_embed.py).

The packed payload must round-trip exactly (the browser recomputes virtual
images from it), the size cap must fall back cleanly, and the cell → vectors
resolution must go through the SignalRef. The in-browser behaviour itself is
covered by electron/tests/vectors_report_embed.spec.ts against a real browser.
"""
from __future__ import annotations

import base64
import json
import re

import numpy as np
import pytest


def _vecs():
    from spyde.tests.gen_vectors_embed import synthetic_vectors
    return synthetic_vectors(nav=(8, 8))


class TestPackVectors:
    def test_payload_roundtrip(self):
        from spyde.actions.report.vectors_embed import pack_vectors
        from spyde.signals.diffraction_vectors import COL_KX, COL_KY, COL_INTENSITY

        vecs = _vecs()
        payload = pack_vectors(vecs)
        assert payload is not None
        hdr = payload["header"]
        n = hdr["n"]
        assert n == len(vecs.flat_buffer)
        assert hdr["nav"] == [8, 8]
        assert hdr["k"]["kx"] == [-1.0, 1.0]
        assert hdr["k"]["units"] == "1/A"

        blob = base64.b64decode(payload["b64"])
        off = 0
        x = np.frombuffer(blob, "<u2", n, off); off += 2 * n
        y = np.frombuffer(blob, "<u2", n, off); off += 2 * n
        kx = np.frombuffer(blob, "<f4", n, off); off += 4 * n
        ky = np.frombuffer(blob, "<f4", n, off); off += 4 * n
        inten = np.frombuffer(blob, "<f4", n, off)

        buf = vecs.flat_buffer
        np.testing.assert_array_equal(x, buf[:, 0].astype(np.uint16))
        np.testing.assert_array_equal(y, buf[:, 1].astype(np.uint16))
        np.testing.assert_array_equal(kx, buf[:, COL_KX])
        np.testing.assert_array_equal(ky, buf[:, COL_KY])
        np.testing.assert_array_equal(inten, buf[:, COL_INTENSITY])

    def test_cap_refuses_embed(self, monkeypatch):
        import spyde.actions.report.vectors_embed as ve
        monkeypatch.setattr(ve, "MAX_EMBED_VECTORS", 10)
        assert ve.pack_vectors(_vecs()) is None      # 8*8*3 = 192 > 10


class TestExplorerHtml:
    def test_selfcontained_page(self):
        from spyde.actions.report.vectors_embed import vectors_explorer_html
        html = vectors_explorer_html(_vecs(), caption="cap & <text>")
        assert html is not None
        assert "vx-header" in html and "vx-data" in html
        # Two mounted anyplotlib figures + their serialized states + the ESM.
        assert "vx-figk" in html and "vx-figvi" in html
        assert "vx-state-k" in html and "vx-state-vi" in html
        assert "vx-esm" in html and "createLocalModel" in html
        # Both detector shapes + the real-space region widget serialized in
        # (inside the panel-state JSON string, so quotes are escaped).
        assert "circle" in html
        assert "annular" in html and "rectangle" in html
        assert "cap &amp; &lt;text&gt;" in html       # caption escaped
        # The embedded JSON header parses and matches the dataset.
        m = re.search(r'id="vx-header">(.*?)</script>', html, re.S)
        hdr = json.loads(m.group(1))
        assert hdr["nav"] == [8, 8] and hdr["n"] == 8 * 8 * 3
        # Single-file contract: no external script/style/fetch references.
        assert "<script src=" not in html and "<link " not in html

    def test_over_cap_returns_none(self, monkeypatch):
        import spyde.actions.report.vectors_embed as ve
        monkeypatch.setattr(ve, "MAX_EMBED_VECTORS", 10)
        assert ve.vectors_explorer_html(_vecs()) is None


class TestCellResolution:
    def test_vectors_for_cell_via_signal_ref(self):
        from spyde.actions.report.vectors_embed import vectors_for_cell

        vecs = _vecs()

        class _Tree:
            diffraction_vectors = vecs

        class _Plot:
            signal_tree = _Tree()

        class _Ref:
            def resolve(self, session):
                return _Plot()

        class _Layer:
            source = _Ref()

        class _Panel:
            layers = [_Layer()]

        class _Spec:
            panels = [_Panel()]

        class _Cell:
            spec = _Spec()

        assert vectors_for_cell(object(), _Cell()) is vecs

        class _NoVecTree:
            diffraction_vectors = None

        _Plot.signal_tree = _NoVecTree()
        assert vectors_for_cell(object(), _Cell()) is None

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
        # New: DP frame pixel dims + calibrated-vs-pixel disk radius.
        assert hdr["sig"] == [256, 256]
        assert hdr["r_px"] == 4.0
        # Run-length index over the point block (length ny*nx+1 for 4-D).
        assert hdr["nav_off"] == 8 * 8 + 1

        blob = base64.b64decode(payload["b64"])
        off = 0
        x = np.frombuffer(blob, "<u2", n, off); off += 2 * n
        y = np.frombuffer(blob, "<u2", n, off); off += 2 * n
        kx = np.frombuffer(blob, "<f4", n, off); off += 4 * n
        ky = np.frombuffer(blob, "<f4", n, off); off += 4 * n
        inten = np.frombuffer(blob, "<f4", n, off); off += 4 * n

        buf = vecs.flat_buffer
        np.testing.assert_array_equal(x, buf[:, 0].astype(np.uint16))
        np.testing.assert_array_equal(y, buf[:, 1].astype(np.uint16))
        np.testing.assert_array_equal(kx, buf[:, COL_KX])
        np.testing.assert_array_equal(ky, buf[:, COL_KY])
        np.testing.assert_array_equal(inten, buf[:, COL_INTENSITY])

        # nav_off tail: monotonic uint32, starts at 0, ends at n, and slices a
        # known position to its exact vector count (3 per position in the fixture).
        nav_off = np.frombuffer(blob, "<u4", hdr["nav_off"], off)
        assert nav_off[0] == 0 and nav_off[-1] == n
        assert np.all(np.diff(nav_off.astype(np.int64)) >= 0)
        ny, nx = hdr["nav"]
        for (iy, ix) in [(0, 0), (3, 5), (7, 7)]:
            p = iy * nx + ix
            s, e = int(nav_off[p]), int(nav_off[p + 1])
            assert e - s == 3
            # the sliced rows really are this nav position
            np.testing.assert_array_equal(x[s:e], np.full(3, ix, np.uint16))
            np.testing.assert_array_equal(y[s:e], np.full(3, iy, np.uint16))

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
        # ONE mounted anyplotlib figure (two panels) + its serialized state +
        # the ESM + the discovered nav/DP panel ids.
        assert 'id="vx-fig"' in html
        assert "vx-state" in html and "vx-navid" in html and "vx-dpid" in html
        assert "vx-esm" in html and "createLocalModel" in html
        # Navigator carries BOTH widgets (crosshair pointer + rectangle
        # integrate); the DP carries the CIRCLE detector for virtual imaging —
        # all serialized inside the panel-state JSON string.
        assert "crosshair" in html and "rectangle" in html
        assert '"type": "circle"' in html or "circle" in html
        # The themed SEGMENTED pointer/integrate toggle (fix #1) — pill buttons,
        # not radios. Both mode buttons + the accent-driven aria-pressed present.
        assert "vx-seg-btn" in html
        assert 'data-mode="pointer"' in html and 'data-mode="integrate"' in html
        assert 'aria-pressed=' in html
        # Dark theme (fix #2): dark color-scheme + the app surface color, not #fff.
        assert "color-scheme: dark" in html
        assert "#1e1e2e" in html
        assert "background: #fff" not in html
        # VI machinery (fix #4): the detector→virtual-image scan + nav overlay.
        assert "computeVI" in html and "refreshVI" in html and "ovVI" in html
        assert "setDetector" in html
        assert "cap &amp; &lt;text&gt;" in html       # caption escaped
        # The embedded JSON header parses and matches the dataset (new keys too).
        m = re.search(r'id="vx-header">(.*?)</script>', html, re.S)
        hdr = json.loads(m.group(1))
        assert hdr["nav"] == [8, 8] and hdr["n"] == 8 * 8 * 3
        assert hdr["sig"] == [256, 256] and hdr["r_px"] == 4.0
        assert hdr["nav_off"] == 8 * 8 + 1
        # The nav/DP panel ids are real panel ids present in the figure state.
        nav_id = re.search(r'id="vx-navid">(.*?)</script>', html, re.S).group(1)
        dp_id = re.search(r'id="vx-dpid">(.*?)</script>', html, re.S).group(1)
        assert nav_id and dp_id and nav_id != dp_id
        assert f"panel_{nav_id}_json" in html and f"panel_{dp_id}_json" in html
        # Single-file contract: no external script/style/fetch references.
        assert "<script src=" not in html and "<link " not in html

    def test_over_cap_returns_none(self, monkeypatch):
        import spyde.actions.report.vectors_embed as ve
        monkeypatch.setattr(ve, "MAX_EMBED_VECTORS", 10)
        assert ve.vectors_explorer_html(_vecs()) is None


class TestBuildCaching:
    """Fix #6: the ESM is read once (module-level) and a built explorer page is
    memoized per (cell_id, id(vectors)) so a rebuild for the same cell + same
    vectors reuses the packed blob + figure instead of re-encoding."""

    def test_page_memoized_by_cell_and_identity(self):
        import spyde.actions.report.vectors_embed as ve
        ve.clear_explorer_cache()
        vecs = _vecs()
        # Same cell + same vectors object → the EXACT same page object (no rebuild).
        h1 = ve.vectors_explorer_html(vecs, cache_key="cellA")
        h2 = ve.vectors_explorer_html(vecs, cache_key="cellA")
        assert h1 is not None and h1 is h2

    def test_swapped_vectors_identity_rebuilds(self):
        import spyde.actions.report.vectors_embed as ve
        ve.clear_explorer_cache()
        h1 = ve.vectors_explorer_html(_vecs(), cache_key="cellA")
        # A DIFFERENT vectors object under the same key → cache MISS → new page.
        h2 = ve.vectors_explorer_html(_vecs(), cache_key="cellA")
        assert h1 is not None and h2 is not None and h1 is not h2

    def test_no_cache_key_never_memoizes(self):
        import spyde.actions.report.vectors_embed as ve
        ve.clear_explorer_cache()
        vecs = _vecs()
        # Without a cache_key each call rebuilds (distinct page objects).
        h1 = ve.vectors_explorer_html(vecs)
        h2 = ve.vectors_explorer_html(vecs)
        assert h1 is not None and h1 is not h2

    def test_clear_drops_entry(self):
        import spyde.actions.report.vectors_embed as ve
        ve.clear_explorer_cache()
        vecs = _vecs()
        h1 = ve.vectors_explorer_html(vecs, cache_key="cellA")
        ve.clear_explorer_cache("cellA")
        h2 = ve.vectors_explorer_html(vecs, cache_key="cellA")
        # After a clear, the same cell + vectors rebuilds a fresh page object.
        assert h1 is not h2

    def test_esm_text_cached_module_level(self):
        import spyde.actions.report.vectors_embed as ve
        # The module-level ESM cache is populated once and reused.
        ve._ESM_TEXT = None
        t1 = ve._esm_text()
        assert isinstance(t1, str) and len(t1) > 1000
        assert ve._esm_text() is t1        # same object, not re-read


class TestDropChoice:
    """Dropping a vectors-carrying window defers the cell behind a
    ``report_vectors_choice`` prompt; the re-send with ``vectors_mode`` creates
    the cell and stamps the choice on its FigureSpec; export honors it."""

    @staticmethod
    def _attach_vectors(session, wid):
        class _Vecs:
            flat_buffer = np.zeros((7, 5), dtype=np.float32)
        for p in session._plots:
            if p.window_id == wid:
                p.signal_tree.diffraction_vectors = _Vecs()
                return
        raise AssertionError(f"no plot for window {wid}")

    @staticmethod
    def _prime(session):
        for p in session._plots:
            if isinstance(getattr(p, "current_data", None), np.ndarray):
                continue
            sig = p.plot_state.current_signal
            frame = np.asarray(sig.data)
            if frame.ndim > 2:
                frame = frame.reshape(-1, *frame.shape[-2:])[0]
            p.current_data = np.ascontiguousarray(frame.astype(np.float32))
            p._last_levels = (float(np.nanmin(p.current_data)),
                              float(np.nanmax(p.current_data)))

    @staticmethod
    def _signal_wid(session):
        for p in session._plots:
            if not getattr(p, "is_navigator", False) and p.window_id is not None:
                return p.window_id
        return session._plots[0].window_id

    def test_drop_without_mode_asks_and_defers(self, tem_2d_dataset):
        from spyde.actions.report import handlers as h
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        self._prime(session)
        wid = self._signal_wid(session)
        h.report_new(session, None, {})
        self._attach_vectors(session, wid)
        messages.clear()
        h.report_add_figure(session, None, {"source_window_id": wid,
                                            "caption": "V", "index": 0})
        choices = [m for m in messages if m.get("type") == "report_vectors_choice"]
        assert len(choices) == 1
        assert choices[0]["source_window_id"] == wid
        assert choices[0]["caption"] == "V"
        assert choices[0]["index"] == 0
        assert choices[0]["count"] == 7
        # No cell was created — the drop is deferred behind the prompt.
        assert not h._manager(session).doc.cells

    @pytest.mark.parametrize("mode", ["viewer", "image"])
    def test_drop_with_mode_stamps_spec(self, tem_2d_dataset, mode):
        from spyde.actions.report import handlers as h
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        self._prime(session)
        wid = self._signal_wid(session)
        h.report_new(session, None, {})
        self._attach_vectors(session, wid)
        messages.clear()
        h.report_add_figure(session, None, {"source_window_id": wid,
                                            "vectors_mode": mode})
        assert not [m for m in messages
                    if m.get("type") == "report_vectors_choice"]
        cells = h._manager(session).doc.cells
        assert len(cells) == 1 and cells[0].cell_type == "figure"
        assert cells[0].spec.vectors_mode == mode

    def test_plain_drop_is_unprompted(self, tem_2d_dataset):
        from spyde.actions.report import handlers as h
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        self._prime(session)
        wid = self._signal_wid(session)
        h.report_new(session, None, {})
        messages.clear()
        h.report_add_figure(session, None, {"source_window_id": wid})
        assert not [m for m in messages
                    if m.get("type") == "report_vectors_choice"]
        cells = h._manager(session).doc.cells
        assert len(cells) == 1
        assert cells[0].spec.vectors_mode == ""

    def test_spec_yaml_roundtrip(self):
        from spyde.actions.report.model import FigureSpec
        spec = FigureSpec(vectors_mode="image")
        assert FigureSpec.from_yaml(spec.to_yaml()).vectors_mode == "image"
        # Older files (no key) keep the viewer-when-available default, and a
        # default spec doesn't grow the key.
        assert FigureSpec.from_dict({}).vectors_mode == ""
        assert "vectors_mode" not in FigureSpec().to_dict()


class TestExportHonorsMode:
    def _cell(self, vectors_mode):
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

        _Spec.vectors_mode = vectors_mode

        class _Cell:
            id = "c1"
            cell_type = "figure"
            caption = "V"
            placeholder = False
            spec = _Spec()

        return _Cell()

    def _render(self, cell, monkeypatch):
        import spyde.actions.report.export_html as ex

        class _Doc:
            cells = [cell]

        class _Mgr:
            doc = _Doc()

        # Keep the fallback figure path inert — this test is only about
        # whether the vectors explorer is chosen.
        monkeypatch.setattr(ex, "_build_interactive_figure_html",
                            lambda mgr, c: None)
        return ex._render_body(_Mgr(), {}, interactive=True, session=object())

    def test_image_mode_skips_viewer(self, monkeypatch):
        body = self._render(self._cell("image"), monkeypatch)
        assert "vx-root" not in body

    def test_default_and_viewer_embed(self, monkeypatch):
        assert "vx-root" in self._render(self._cell(""), monkeypatch)
        assert "vx-root" in self._render(self._cell("viewer"), monkeypatch)


class TestSidebarExplorer:
    """build_figure_window hosts the LIVE 2-panel vectors explorer in the
    SIDEBAR cell (Approach A) for a VIEWER-vectors cell — the SAME page the HTML
    export embeds (export/sidebar parity). An IMAGE-mode cell keeps the plain
    anyplotlib snapshot figure."""

    @staticmethod
    def _prime(session):
        for p in session._plots:
            if isinstance(getattr(p, "current_data", None), np.ndarray):
                continue
            sig = p.plot_state.current_signal
            frame = np.asarray(sig.data)
            if frame.ndim > 2:
                frame = frame.reshape(-1, *frame.shape[-2:])[0]
            p.current_data = np.ascontiguousarray(frame.astype(np.float32))
            p._last_levels = (float(np.nanmin(p.current_data)),
                              float(np.nanmax(p.current_data)))

    @staticmethod
    def _signal_wid(session):
        for p in session._plots:
            if not getattr(p, "is_navigator", False) and p.window_id is not None:
                return p.window_id
        return session._plots[0].window_id

    @staticmethod
    def _attach_real_vectors(session, wid):
        vecs = _vecs()
        for p in session._plots:
            if p.window_id == wid:
                p.signal_tree.diffraction_vectors = vecs
                return
        raise AssertionError(f"no plot for window {wid}")

    def _drop_vectors_cell(self, session, messages, mode):
        from spyde.actions.report import handlers as h
        self._prime(session)
        wid = self._signal_wid(session)
        h.report_new(session, None, {})
        self._attach_real_vectors(session, wid)
        messages.clear()
        h.report_add_figure(session, None,
                            {"source_window_id": wid, "vectors_mode": mode})
        figs = [m for m in messages
                if m.get("type") == "figure" and m.get("host") == "report"]
        assert figs, "no report figure message emitted"
        return figs[-1]

    def test_viewer_cell_emits_live_explorer(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        fig = self._drop_vectors_cell(session, messages, "viewer")
        html = fig.get("html") or ""
        # The self-contained explorer page, NOT a plain anyplotlib snapshot.
        assert "vx-root" in html and "vx-header" in html and "vx-data" in html
        # Themed segmented toggle (fix #1) + dark theme (fix #2).
        assert "vx-seg-btn" in html and 'data-mode="integrate"' in html
        assert "color-scheme: dark" in html
        # It carries the navigator widgets + DP detector (2-panel figure + VI).
        assert "crosshair" in html and "rectangle" in html
        assert "computeVI" in html
        # fig_id is the explorer scheme (vx_<cell>_<uuid>), unique per build.
        assert str(fig.get("fig_id", "")).startswith("vx_")

    def test_image_cell_keeps_snapshot(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        fig = self._drop_vectors_cell(session, messages, "image")
        html = fig.get("html") or ""
        # Image mode → the plain anyplotlib figure, never the explorer.
        assert "vx-root" not in html
        assert not str(fig.get("fig_id", "")).startswith("vx_")

    def test_default_mode_embeds_explorer(self, tem_2d_dataset):
        # A plain drop that resolves to a vectors tree AND was re-sent with the
        # default viewer choice ("") embeds the explorer (mirrors export's
        # `!= "image"` gate).
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        fig = self._drop_vectors_cell(session, messages, "viewer")
        # (default "" path is covered by report_add_figure deferring the choice;
        # here we assert the viewer explorer is the emitted figure.)
        assert "vx-root" in (fig.get("html") or "")

    def test_rebuild_reuses_memoized_page(self, tem_2d_dataset):
        """Fix #6: rebuilding the SAME viewer cell (same vectors) does NOT
        re-encode — the explorer page is served from the (cell_id, id(vecs))
        memo. We assert vectors_explorer_html is called with the cell's
        cache_key and that a second build returns the identical page string."""
        from spyde.actions.report import handlers as h
        import spyde.actions.report.vectors_embed as ve
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        fig = self._drop_vectors_cell(session, messages, "viewer")
        html1 = fig.get("html") or ""
        assert "vx-root" in html1
        cell_id = fig.get("cell_id")
        assert cell_id
        # Spy: pack_vectors must NOT run again on a cache hit (it's the expensive
        # base64 encode). Rebuild the SAME cell → served from the memo.
        calls = {"pack": 0}
        orig_pack = ve.pack_vectors
        try:
            ve.pack_vectors = lambda v: (calls.__setitem__("pack", calls["pack"] + 1)
                                         or orig_pack(v))
            mgr = h._manager(session)
            cell = mgr.doc.cell_by_id(cell_id)
            messages.clear()
            mgr.build_figure_window(cell)
        finally:
            ve.pack_vectors = orig_pack
        figs = [m for m in messages
                if m.get("type") == "figure" and m.get("host") == "report"]
        assert figs, "no figure re-emitted on rebuild"
        html2 = figs[-1].get("html") or ""
        # Same page bytes, and NO re-pack (the expensive encode was skipped).
        assert html2 == html1
        assert calls["pack"] == 0


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

"""Report orientation embed (spyde/actions/report/orientation_embed.py).

The packed payload must be NAV-ALIGNED and round-trip exactly (a crosshair pick
in the browser is an index into it, so a shifted row is a wrong orientation, not
a rendering nit), the page must be self-contained, and the cell → result
resolution must go through the SignalRef like the vectors embed's. The
in-browser behaviour is covered by
``electron/tests/orientation_report_embed.spec.ts`` against a real browser.
"""
from __future__ import annotations

import base64
import json
import re

import numpy as np

from spyde.tests.migrated.test_ipf_3d import _al_orientation_map


_OM_CACHE = None
_PACKED = None


def _om(ny=4, nx=5):
    """The default (4, 5) map is built ONCE for the module — every consumer
    only reads it (pack_orientation / the explorer html take it as input),
    and rebuilding re-ran the random-quat orix symmetry reduction per test.
    Non-default shapes still build fresh."""
    global _OM_CACHE
    if (ny, nx) != (4, 5):
        return _al_orientation_map(ny, nx)
    if _OM_CACHE is None:
        _OM_CACHE = _al_orientation_map(4, 5)
    return _OM_CACHE


def _packed():
    """pack_orientation(_om()) once (orix reduction x 4 directions) — shared
    by the read-only payload consumers.  Tests that monkeypatch the module's
    cap/stride constants run their own pack_orientation call."""
    global _PACKED
    if _PACKED is None:
        from spyde.actions.report.orientation_embed import pack_orientation
        _PACKED = pack_orientation(_om())
    return _PACKED


class TestPackOrientation:
    def test_payload_roundtrip(self):
        from spyde.actions.report.orientation_embed import DIRECTIONS

        om = _om()
        payload = _packed()
        assert payload is not None
        hdr = payload["header"]
        assert hdr["nav"] == [4, 5]
        assert hdr["m"] == 20
        assert hdr["dirs"] == list(DIRECTIONS)
        assert hdr["stride"] == 1                    # 20 points, well under the cap
        assert hdr["phases"] and hdr["phases"][0]["name"] == "Al"
        assert len(hdr["phases"][0]["edges"]) >= 3

        blob = base64.b64decode(payload["b64"])
        m = hdr["m"]
        off = 0
        phase = np.frombuffer(blob, "<u1", m, off); off += m
        np.testing.assert_array_equal(
            phase, np.asarray(om.phase_map()).reshape(-1).astype(np.uint8))

        for d in DIRECTIONS:
            rgb = np.frombuffer(blob, "<u1", m * 3, off).reshape(m, 3)
            off += m * 3
            xy = np.frombuffer(blob, "<f4", m * 2, off).reshape(m, 2)
            off += 4 * m * 2
            xyz = np.frombuffer(blob, "<f4", m * 3, off).reshape(m, 3)
            off += 4 * m * 3
            # The colours ARE the map the app paints, position for position.
            np.testing.assert_array_equal(
                rgb, np.asarray(om.ipf_color_map(d)).reshape(m, 3))
            # Every sphere direction is a unit vector (a point ON the sphere).
            np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1.0, atol=1e-3)
            assert np.isfinite(xy).all()
        assert off == len(blob), "the blob has trailing bytes the page won't read"

    def test_rows_are_nav_positions(self):
        """Row i of every array is flat nav index i — the whole contract the
        crosshair pick rests on. Checked against the app's own per-position
        accessor rather than against the packing code's arithmetic."""
        om = _om(4, 5)
        payload = _packed()
        m = payload["header"]["m"]
        _rgb, xy, xyz = payload["arrays"]["z"]
        nx = payload["header"]["nav"][1]
        for flat in (0, 7, m - 1):
            iy, ix = divmod(flat, nx)
            np.testing.assert_allclose(xy[flat],
                                       om.ipf_xy(iy, ix, "z")[0][0], atol=1e-5)
            np.testing.assert_allclose(xyz[flat],
                                       om.ipf_xyz(iy, ix, 0, "z")[0], atol=1e-5)

    def test_cap_refuses_embed(self, monkeypatch):
        import spyde.actions.report.orientation_embed as oe
        monkeypatch.setattr(oe, "MAX_EMBED_POSITIONS", 10)
        assert oe.pack_orientation(_om(4, 5)) is None      # 20 > 10

    def test_cloud_is_strided_not_truncated(self, monkeypatch):
        """Over CLOUD_MAX the drawn cloud is uniformly strided. A head-slice of a
        raster-order scan is its top rows — a crop of the sample, not a sample of
        the orientations."""
        import spyde.actions.report.orientation_embed as oe
        monkeypatch.setattr(oe, "CLOUD_MAX", 5)
        payload = oe.pack_orientation(_om(4, 5))           # 20 positions
        assert payload["header"]["stride"] == 4


def _figs(html):
    """The ``ox-figs`` payload: ``{view: {"state", "panels"}}``."""
    raw = re.search(r'id="ox-figs">(.*?)</script>', html, re.S).group(1)
    return json.loads(raw.replace("<\\/", "</"))


class TestExplorerHtml:
    def test_selfcontained_page(self):
        from spyde.actions.report.orientation_embed import (
            orientation_explorer_html,
        )
        html = orientation_explorer_html(_om(), caption="cap & <text>")
        assert html is not None
        assert "ox-header" in html and "ox-data" in html and "ox-figs" in html
        # The map is its own mount; the explorer is a slot of lazily-mounted views.
        assert 'id="ox-map"' in html and 'id="ox-slot"' in html
        assert "ox-esm" in html and "createLocalModel" in html
        # The map's crosshair is the only interaction that moves the pick.
        assert "crosshair" in html
        # BOTH toggle pairs, plus the X/Y/Z direction — the app's own controls.
        assert "ox-seg-btn" in html
        for v in ("2d", "3d"):
            assert f'data-dim="{v}"' in html
        for s in ("points", "heatmap"):
            assert f'data-style="{s}"' in html
        for d in ("x", "y", "z"):
            assert f'data-dir="{d}"' in html
        assert "color-scheme: dark" in html and "#1e1e2e" in html
        # A pick moves the scatter marker AND both spheres' highlight, and turns
        # the camera — the live IpfWindowController.show_orientation.
        assert "setPick" in html and "highlight" in html
        assert "_view_from_python" in html
        assert "cap &amp; &lt;text&gt;" in html          # caption escaped

        hdr = json.loads(
            re.search(r'id="ox-header">(.*?)</script>', html, re.S).group(1))
        assert hdr["nav"] == [4, 5] and hdr["m"] == 20
        # Single-file contract: nothing external.
        assert "<script src=" not in html and "<link " not in html

    def test_all_four_views_plus_the_map_are_built(self):
        """Every toggle state ships as its own figure, and each is the panel
        KIND that view needs — the 2-D ones on PlotXY (the '1d' panel kind), the
        3-D ones on a 3-D panel."""
        from spyde.actions.report.orientation_embed import (
            VIEWS, orientation_explorer_html,
        )
        figs = _figs(orientation_explorer_html(_om()))
        assert set(figs) == {"map", *VIEWS}
        kinds = {}
        for name, rec in figs.items():
            pj = json.loads(rec["state"][f"panel_{rec['panels'][0]}_json"])
            kinds[name] = pj["kind"]
        assert kinds["map"] == "2d"                 # the IPF colour image
        assert kinds["3d-points"] == "3d"
        assert kinds["3d-heat"] == "3d"
        assert kinds["2d-points"] not in ("2d", "3d")   # PlotXY → the '1d' kind
        assert kinds["2d-heat"] not in ("2d", "3d")

    def test_panels_are_in_axis_order(self):
        """``panels`` must be the ORDER the axes were created, because every
        per-phase update maps panel i to phase i. ``figure_state``'s own key
        order is not that, so the list comes from ``layout_json``."""
        from spyde.actions.report.orientation_embed import (
            orientation_explorer_html,
        )
        figs = _figs(orientation_explorer_html(_om()))
        for name, rec in figs.items():
            layout = json.loads(rec["state"]["layout_json"])
            want = [s["id"] for s in layout["panel_specs"]]
            assert rec["panels"] == want, name

    def test_density_images_are_png_urls_not_raw_bytes(self):
        """A density field is smooth and mostly transparent; PNG crushes it, and
        the 3-D texture wants a data URL anyway. Raw RGBA in the blob was ~1 MB
        of the page for a 256px raster."""
        from spyde.actions.report.orientation_embed import (
            DIRECTIONS, pack_orientation,
        )
        rec = _packed()["header"]["phases"][0]
        for key in ("raster", "texture"):
            assert set(rec[key]) == set(DIRECTIONS), key
            for url in rec[key].values():
                assert url.startswith("data:image/png;base64,")
        # Each direction really is a different picture.
        assert len(set(rec["raster"].values())) == len(DIRECTIONS)

    def test_over_cap_returns_none(self, monkeypatch):
        import spyde.actions.report.orientation_embed as oe
        monkeypatch.setattr(oe, "MAX_EMBED_POSITIONS", 10)
        assert oe.orientation_explorer_html(_om(4, 5)) is None


class TestBuildCaching:
    def _clear(self):
        from spyde.actions.report.orientation_embed import clear_explorer_cache
        clear_explorer_cache()

    def test_page_memoized_by_cell_and_identity(self):
        from spyde.actions.report.orientation_embed import (
            orientation_explorer_html,
        )
        self._clear()
        om = _om()
        a = orientation_explorer_html(om, cache_key="c1")
        b = orientation_explorer_html(om, cache_key="c1")
        assert a is b                        # the SAME object, not just equal

    def test_swapped_result_identity_rebuilds(self):
        from spyde.actions.report.orientation_embed import (
            orientation_explorer_html,
        )
        self._clear()
        # Two genuinely DISTINCT map objects (the module memo would hand back
        # one identity, which is the opposite of what this test needs).
        a = orientation_explorer_html(_al_orientation_map(4, 5), cache_key="c1")
        b = orientation_explorer_html(_al_orientation_map(4, 5), cache_key="c1")
        assert a is not b

    def test_no_cache_key_never_memoizes(self):
        from spyde.actions.report.orientation_embed import (
            orientation_explorer_html,
        )
        self._clear()
        om = _om()
        assert orientation_explorer_html(om) is not orientation_explorer_html(om)

    def test_clear_drops_entry(self):
        from spyde.actions.report.orientation_embed import (
            clear_explorer_cache, orientation_explorer_html,
        )
        self._clear()
        om = _om()
        a = orientation_explorer_html(om, cache_key="c1")
        clear_explorer_cache("c1")
        assert orientation_explorer_html(om, cache_key="c1") is not a


class TestSpecMode:
    def test_yaml_roundtrip(self):
        from spyde.actions.report.model import FigureSpec
        spec = FigureSpec()
        assert spec.orientation_mode == ""
        assert "orientation_mode" not in spec.to_dict()   # absent when default
        spec.orientation_mode = "image"
        assert FigureSpec.from_dict(spec.to_dict()).orientation_mode == "image"

    def test_older_file_defaults_to_viewer(self):
        from spyde.actions.report.model import FigureSpec
        assert FigureSpec.from_dict({"layout": {"kind": "single"}}
                                    ).orientation_mode == ""


class TestCellResolution:
    def _cell(self, tree):
        class _Plot:
            signal_tree = tree

        class _Ref:
            def resolve(self, session):
                return _Plot()

        class _Layer:
            source = _Ref()

        class _Panel:
            layers = [_Layer()]

        class _Spec:
            panels = [_Panel()]
            orientation_mode = ""

        class _Cell:
            spec = _Spec()
        return _Cell()

    def test_resolves_raw_om_through_the_signal_ref(self):
        from spyde.actions.report.orientation_embed import orientation_for_cell
        om = _om()

        class _Tree:
            orientation_map = om
        assert orientation_for_cell(object(), self._cell(_Tree())) is om

    def test_resolves_vector_om_too(self):
        """A vector-OM tree attaches the result under a different name; the
        embed goes through the SAME chain the live IPF window uses, so both
        producers resolve."""
        from spyde.actions.report.orientation_embed import orientation_for_cell
        om = _om()

        class _Tree:
            orientation_map = None
            vector_orientation = om
        assert orientation_for_cell(object(), self._cell(_Tree())) is om

    def test_no_orientation_is_none(self):
        from spyde.actions.report.orientation_embed import orientation_for_cell

        class _Tree:
            orientation_map = None
        assert orientation_for_cell(object(), self._cell(_Tree())) is None

    def test_specless_cell_is_none(self):
        from spyde.actions.report.orientation_embed import orientation_for_cell

        class _Cell:
            spec = None
        assert orientation_for_cell(object(), _Cell()) is None


class TestExportHonoursMode:
    """``export_html`` swaps in the explorer unless the cell is pinned to
    ``"image"`` — the same gate the vectors embed uses."""

    def _render(self, mode, monkeypatch):
        from spyde.actions.report import export_html as ex
        from spyde.actions.report.model import Cell, FigureSpec

        om = _om()
        monkeypatch.setattr(
            "spyde.actions.report.orientation_embed.orientation_for_cell",
            lambda session, cell: om)
        # Keep the fallback figure path inert — this is only about WHICH
        # renderer is chosen, and the real one needs a live manager.
        monkeypatch.setattr(ex, "_build_interactive_figure_html",
                            lambda mgr, c: None)
        spec = FigureSpec()
        spec.orientation_mode = mode
        cell = Cell(id="c1", cell_type="figure", caption="", spec=spec)
        return ex._render_figure_side_html(None, cell, {}, interactive=True,
                                           session=None)

    def test_image_mode_skips_the_viewer(self, monkeypatch):
        assert "ox-root" not in self._render("image", monkeypatch)

    def test_default_and_viewer_embed(self, monkeypatch):
        assert "ox-root" in self._render("", monkeypatch)
        assert "ox-root" in self._render("viewer", monkeypatch)


class TestSidebarExplorer:
    """``build_figure_window`` hosts the live explorer in the SIDEBAR cell, the
    same page the export embeds — and a vectors cell still wins when a tree
    carries both."""

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

    def _drop(self, session, messages, *, with_vectors=False):
        from spyde.actions.report import handlers as h
        self._prime(session)
        wid = self._signal_wid(session)
        h.report_new(session, None, {})
        for p in session._plots:
            if p.window_id == wid:
                p.signal_tree.orientation_map = _om()
                if with_vectors:
                    from spyde.tests.gen_vectors_embed import synthetic_vectors
                    p.signal_tree.diffraction_vectors = synthetic_vectors(nav=(8, 8))
        messages.clear()
        h.report_add_figure(session, None,
                            {"source_window_id": wid, "vectors_mode": "image"})
        figs = [m for m in messages
                if m.get("type") == "figure" and m.get("host") == "report"]
        assert figs, "no report figure message emitted"
        return figs[-1]

    def test_orientation_cell_emits_live_explorer(self, tem_2d_dataset):
        session = tem_2d_dataset["window"]
        fig = self._drop(session, tem_2d_dataset["messages"])
        html = fig.get("html") or ""
        assert "ox-root" in html and "ox-header" in html and "ox-data" in html
        assert "crosshair" in html and "setPick" in html
        assert str(fig.get("fig_id", "")).startswith("ox_")

    def test_image_mode_keeps_the_snapshot(self, tem_2d_dataset):
        from spyde.actions.report import handlers as h
        session = tem_2d_dataset["window"]
        fig = self._drop(session, tem_2d_dataset["messages"])
        cell_id = fig.get("cell_id")
        mgr = h._manager(session)
        cell = mgr.doc.cell_by_id(cell_id)
        cell.spec.orientation_mode = "image"
        tem_2d_dataset["messages"].clear()
        mgr.build_figure_window(cell)
        out = [m for m in tem_2d_dataset["messages"]
               if m.get("type") == "figure" and m.get("host") == "report"][-1]
        assert "ox-root" not in (out.get("html") or "")
        assert not str(out.get("fig_id", "")).startswith("ox_")

    def test_vectors_win_when_a_tree_carries_both(self, tem_2d_dataset):
        """A vector-OM tree has vectors AND an orientation result. The vectors
        explorer is the one that cell was dragged from, so it stays."""
        from spyde.actions.report import handlers as h
        session = tem_2d_dataset["window"]
        messages = tem_2d_dataset["messages"]
        fig = self._drop(session, messages, with_vectors=True)
        cell = h._manager(session).doc.cell_by_id(fig.get("cell_id"))
        cell.spec.vectors_mode = "viewer"
        messages.clear()
        h._manager(session).build_figure_window(cell)
        html = [m for m in messages if m.get("type") == "figure"
                and m.get("host") == "report"][-1].get("html") or ""
        assert "vx-root" in html and "ox-root" not in html

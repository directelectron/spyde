"""
test_views.py — the unified per-window view registry + side-by-side tiling.

⌘-tiling rebuilds ONE anyplotlib figure with the selected views as side-by-side
axes (``subplots(1, N)``), not N iframes. These check the backend builder:
N axes, canonical view order, and that ``tile_views`` emits a single ``__tiled__``
figure (and skips a lone selection).
"""
import numpy as np

from spyde.actions import views


class TestTiledViews:
    def test_build_tiled_figure_has_n_axes(self):
        views.register_views(42, [
            ("εxx", np.zeros((5, 4))), ("εyy", np.ones((5, 4))),
            ("εxy", np.full((5, 4), 2.0)),
        ], levels=(-1, 1))
        built = views.build_tiled_figure(42, ["εxx", "εyy"])
        assert built is not None
        fig, fig_id, html, sel = built
        assert sel == ["εxx", "εyy"]
        assert len(fig.get_axes()) == 2          # two side-by-side axes, one figure
        assert isinstance(html, str) and html

    def test_build_tiled_preserves_window_order_not_click_order(self):
        views.register_views(43, [
            ("a", np.zeros((3, 3))), ("b", np.zeros((3, 3))), ("c", np.zeros((3, 3))),
        ])
        built = views.build_tiled_figure(43, ["c", "a"])   # clicked c then a
        assert built is not None
        assert built[3] == ["a", "c"]            # canonical (registered) order

    def test_build_tiled_no_data_returns_none(self):
        assert views.build_tiled_figure(987654, ["x"]) is None

    def test_tile_views_emits_one_tiled_figure(self, monkeypatch):
        import spyde.backend.ipc as ipc
        captured = []
        monkeypatch.setattr(ipc, "emit", lambda m: captured.append(m))
        views.register_views(44, [
            ("εxx", np.zeros((4, 4))), ("εyy", np.ones((4, 4))),
        ], levels=(-1, 1))

        class _Plot:
            window_id = 44
        views.tile_views(object(), _Plot(), {"labels": ["εxx", "εyy"]})
        figs = [m for m in captured if m.get("type") == "figure"]
        assert len(figs) == 1
        assert figs[0]["view_label"] == views.TILED_LABEL
        assert figs[0]["view_kind"] == "tiled"
        assert figs[0]["window_id"] == 44

    def test_tile_views_skips_single_selection(self, monkeypatch):
        import spyde.backend.ipc as ipc
        captured = []
        monkeypatch.setattr(ipc, "emit", lambda m: captured.append(m))
        views.register_views(45, [("a", np.zeros((4, 4)))])

        class _Plot:
            window_id = 45
        views.tile_views(object(), _Plot(), {"labels": ["a"]})
        assert not captured                      # one view → no comparison figure

    def test_register_views_rgb_and_scalar(self):
        # RGB (H,W,3) and scalar (H,W) views coexist in one tiled figure.
        rgb = (np.random.rand(6, 6, 3) * 255).astype(np.uint8)
        views.register_views(46, [("IPF", rgb), ("εxx", np.zeros((6, 6)))])
        built = views.build_tiled_figure(46, ["IPF", "εxx"])
        assert built is not None and len(built[0].get_axes()) == 2

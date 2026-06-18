"""
The navigator figure must report the real-space image ASPECT (width/height) so
the frontend sizes the window to it. Without this a non-square scan (e.g. sped_ag
208×64) is aspect-letterboxed into a strip and the crosshair/axes stop lining up
with the image.
"""
from __future__ import annotations


def _nav_figure(messages):
    return next((m for m in messages
                 if m.get("type") == "figure" and m.get("is_navigator")), None)


def _sig_figure(messages):
    return next((m for m in messages
                 if m.get("type") == "figure" and not m.get("is_navigator")), None)


class TestNavigatorAspect:
    def test_navigator_reports_image_aspect(self, stem_4d_dataset):
        # fixture nav is (4, 5) → navigation_shape (5, 4) → aspect 5/4 = 1.25.
        fig = _nav_figure(stem_4d_dataset["messages"])
        assert fig is not None
        assert fig.get("aspect") is not None
        assert abs(float(fig["aspect"]) - (5.0 / 4.0)) < 1e-6

    def test_signal_figure_has_no_aspect(self, stem_4d_dataset):
        # Only the navigator is aspect-sized; the DP keeps the default window.
        fig = _sig_figure(stem_4d_dataset["messages"])
        assert fig is not None
        assert fig.get("aspect") in (None,)

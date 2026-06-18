"""Perf regression: figure HTML must not inline the full anyplotlib JS bundle."""
import json


def _figure_messages(messages):
    return [m for m in messages if m.get("type") == "figure"]


class TestSharedBundle:
    def test_figure_html_is_small(self, stem_4d_dataset):
        figs = _figure_messages(stem_4d_dataset["messages"])
        assert figs, "no figure emitted"
        for f in figs:
            # Was ~370 KB with the bundle inlined; shared-bundle HTML is a few KB.
            assert len(f["html"]) < 60_000, f"figure HTML too large: {len(f['html'])}"

    def test_figure_imports_shared_bundle(self, stem_4d_dataset):
        figs = _figure_messages(stem_4d_dataset["messages"])
        for f in figs:
            html = f["html"]
            assert "const esmSource = null" in html, "inline esm not removed"
            assert "spyde_figure_esm_" in html, "shared bundle not referenced"

    def test_shared_bundle_written_once(self, stem_4d_dataset):
        import glob
        import os
        import tempfile
        shared = glob.glob(os.path.join(tempfile.gettempdir(), "spyde_figure_esm_*.js"))
        assert shared, "shared bundle file not written"
        # The bundle is the real anyplotlib renderer (hundreds of KB).
        assert any(os.path.getsize(p) > 50_000 for p in shared)

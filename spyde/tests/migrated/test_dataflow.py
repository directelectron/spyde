"""Data-flow regression tests: real image data must reach the plots."""
import json


def _panel_pushes(messages):
    """All (width, height, display_max) image pushes from anyplotlib panels."""
    out = []
    for m in messages:
        if m.get("type") == "state_update" and str(m.get("key", "")).startswith("panel"):
            d = json.loads(m["value"]) if isinstance(m["value"], str) else m["value"]
            if "image_width" in d:
                out.append((d["image_width"], d["image_height"],
                            round(float(d.get("display_max", 0)), 1)))
    return out


class TestDataFlow:
    def test_2d_signal_opens_one_window(self, tem_2d_dataset):
        opened = {m["window_id"] for m in tem_2d_dataset["messages"]
                  if m.get("type") == "window_opened"}
        assert len(opened) == 1

    def test_2d_signal_displays_real_data(self, tem_2d_dataset):
        # A 32x32 image with non-zero contrast must be pushed (not the 10x10
        # zero placeholder).
        pushes = _panel_pushes(tem_2d_dataset["messages"])
        real = [p for p in pushes if p[0] == 32 and p[2] > 0]
        assert real, f"no real 32x32 push; got {set(pushes)}"

    def test_4d_stem_opens_navigator_and_signal(self, stem_4d_dataset):
        opened = {m["window_id"] for m in stem_4d_dataset["messages"]
                  if m.get("type") == "window_opened"}
        assert len(opened) == 2

    def test_4d_navigator_fills_with_real_data(self, stem_4d_dataset):
        # Navigator is the 5x4 brightness-gradient image.
        pushes = _panel_pushes(stem_4d_dataset["messages"])
        assert any(w == 5 and h == 4 and dmax > 1 for (w, h, dmax) in pushes), \
            f"navigator not filled; got {set(pushes)}"

    def test_4d_diffraction_pattern_fills_with_real_data(self, stem_4d_dataset):
        # The selector must slice a 16x16 DP (not collapse it to 1-D).
        pushes = _panel_pushes(stem_4d_dataset["messages"])
        assert any(w == 16 and h == 16 and dmax > 1 for (w, h, dmax) in pushes), \
            f"diffraction pattern not filled; got {set(pushes)}"

    def test_metadata_emitted(self, stem_4d_dataset):
        md = [m for m in stem_4d_dataset["messages"] if m.get("type") == "metadata"]
        assert md and "Instrument Metadata" in md[0]["metadata"]

    def test_histogram_emitted(self, stem_4d_dataset):
        hg = [m for m in stem_4d_dataset["messages"] if m.get("type") == "histogram"]
        assert hg and len(hg[0]["counts"]) == 64

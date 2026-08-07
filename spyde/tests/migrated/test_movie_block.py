"""
test_movie_block.py — the Movie BLOCK backend (spyde/actions/report/movie.py):
an editable, persistent in-situ movie cell + its full-screen editor session.

Covers the document handlers (add/set-source), the editor session (open/tune/
preview/export), the freeze frame-count math, the time-range clamp, and the
MEMORY-SAFETY guard (the preview + export must NEVER compute the full dataset —
only single-frame slices), all against a real Session with the ``movie_dataset``
fixture (an 8-frame lazy in-situ movie, 1 frame/chunk).
"""
from __future__ import annotations

import os
import time
from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest

from spyde.actions.report import handlers as H
from spyde.actions.report import movie as M
from spyde.tests.migrated.conftest import _settle


# ── helpers ──────────────────────────────────────────────────────────────────────

def _insitu_plot(session):
    for p in session._plots:
        tree = getattr(p, "signal_tree", None)
        root = getattr(tree, "root", None)
        if root is not None and getattr(root, "_signal_type", None) == "insitu" \
                and not getattr(p, "is_navigator", False):
            return p
    return None


def _latest(messages, mtype):
    for m in reversed(messages):
        if m.get("type") == mtype:
            return m
    return None


def _poll(messages, mtype, tries=300, delay=0.02):
    for _ in range(tries):
        m = _latest(messages, mtype)
        if m is not None:
            return m
        time.sleep(delay)
    return None


def _wait_until(pred, tries=200, delay=0.02):
    """Poll *pred* until true (the nav dispatcher runs the scrub asynchronously)."""
    for _ in range(tries):
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def _add_movie(session, messages):
    """New report + a movie cell seeded from the in-situ plot → the cell id."""
    plot = _insitu_plot(session)
    assert plot is not None, "no in-situ plot in the movie_dataset"
    H.report_new(session, None, {"type": "report"})
    M.report_add_movie_cell(session, None,
                            {"source_window_id": plot.window_id, "caption": "M"})
    rs = _latest(messages, "report_state")
    mc = next(c for c in rs["report"]["cells"] if c["cell_type"] == "movie")
    return mc["id"]


# ── document handlers ─────────────────────────────────────────────────────────────

class TestDocumentHandlers:
    def test_add_movie_cell_seeded_from_plot(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        rs = _latest(messages, "report_state")
        mc = next(c for c in rs["report"]["cells"] if c["id"] == cell_id)
        assert mc["cell_type"] == "movie"
        assert mc["caption"] == "M"
        assert mc["has_source"] is True
        assert mc["placeholder"] is False
        assert "movie" in mc and mc["movie"]["source"] is not None

    def test_add_empty_movie_cell_is_placeholder(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        H.report_new(session, None, {"type": "report"})
        M.report_add_movie_cell(session, None, {"caption": ""})
        rs = _latest(messages, "report_state")
        mc = next(c for c in rs["report"]["cells"] if c["cell_type"] == "movie")
        assert mc["placeholder"] is True
        assert mc["has_source"] is False

    def test_set_movie_source(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        plot = _insitu_plot(session)
        H.report_new(session, None, {"type": "report"})
        M.report_add_movie_cell(session, None, {})     # placeholder
        cell_id = next(c for c in _latest(messages, "report_state")["report"]["cells"]
                       if c["cell_type"] == "movie")["id"]
        M.report_set_movie_source(session, None,
                                  {"cell_id": cell_id,
                                   "source_window_id": plot.window_id})
        mc = next(c for c in _latest(messages, "report_state")["report"]["cells"]
                  if c["id"] == cell_id)
        assert mc["has_source"] is True and mc["placeholder"] is False

    def test_add_movie_cell_open_emits_edit_open(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        plot = _insitu_plot(session)
        H.report_new(session, None, {"type": "report"})
        M.report_add_movie_cell(session, None,
                                {"source_window_id": plot.window_id, "open": True})
        assert _latest(messages, "movie_edit_open") is not None


# ── editor session ─────────────────────────────────────────────────────────────────

class TestEditorSession:
    def test_open_emits_state_shape(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        st = _latest(messages, "movie_state")
        assert st is not None
        for k in ("cell_id", "has_source", "ffmpeg_ok", "running", "n_frames",
                  "time", "sig", "params", "annotations", "text_overlays",
                  "freezes", "speed_segments", "crop", "out_size", "frame_size",
                  "output_info", "signal_fig_id", "signal_window_id", "nav_fig_id",
                  "current_index"):
            assert k in st, f"movie_state missing {k!r}"
        assert set(st["output_info"]) == {"frames", "w", "h"}
        assert st["output_info"]["frames"] == 8      # full 0..7, no stride/freeze
        assert st["cell_id"] == cell_id
        assert st["has_source"] is True
        assert st["running"] is False
        assert st["n_frames"] == 8
        assert st["time"]["scale_s"] == pytest.approx(0.1)
        assert st["params"]["t_end"] == 7
        assert st["frame_size"][0] > 0
        # The editor surfaces the tree's LIVE signal figure (re-parented) — its
        # fig_id must be shipped so the renderer can mount it.
        assert st["signal_fig_id"], "movie_state must ship the live signal fig_id"
        assert st["signal_window_id"] is not None

    def test_tune_persists_onto_spec(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        M.movie_tune(session, None, {
            "cell_id": cell_id,
            "params": {"fps": 24, "cmap": "magma", "t_end": 5, "timestamp": False},
            "crop": [8, 8, 56, 56],
            "freezes": [{"t": 2, "hold_s": 0.5}],
            "annotations": [{"kind": "text", "time_range": [0, 1], "text": "hi",
                             "xy": [4, 4]}]})
        cell = session._report.doc.cell_by_id(cell_id)
        assert cell.movie.params["fps"] == 24
        assert cell.movie.params["cmap"] == "magma"
        assert cell.movie.params["timestamp"] is False
        assert cell.movie.crop == [8, 8, 56, 56]
        assert cell.movie.freezes == [{"t": 2, "hold_s": 0.5}]
        assert len(cell.movie.annotations) == 1

    def test_tune_half_open_clim_does_not_raise(self, movie_dataset):
        # A clim of [lo, None] (only one bound) must NOT raise TypeError and abort
        # the tune — it collapses to auto (None). (Review finding.)
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        M.movie_tune(session, None,
                     {"cell_id": cell_id, "params": {"clim": [5.0, None], "fps": 15}})
        cell = session._report.doc.cell_by_id(cell_id)
        assert cell.movie.params["clim"] is None      # half-open → auto
        assert cell.movie.params["fps"] == 15         # the rest of the tune applied

    def test_tune_clamps_time_range(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        # t_end past the dataset end (8 frames → max index 7) is clamped.
        M.movie_tune(session, None,
                     {"cell_id": cell_id, "params": {"t_start": 3, "t_end": 999}})
        cell = session._report.doc.cell_by_id(cell_id)
        assert cell.movie.params["t_end"] == 7
        assert cell.movie.params["t_start"] == 3

    def test_scrub_drives_real_navigator(self, movie_dataset):
        # Scrubbing the movie drives the tree's REAL 1-D time navigator (the same
        # playback primitive), so the navigator's current index moves — no bespoke
        # preview render. The live signal figure repaints through the real pipeline.
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        assert sess.time_selector is not None, "no time selector bound"
        assert sess.signal_plot is not None, "no signal plot bound"
        M.movie_scrub(session, None, {"cell_id": cell_id, "t": 5})
        _wait_until(lambda: sess.current_index() == 5)
        assert sess.current_index() == 5, "scrub did not move the navigator to 5"
        M.movie_scrub(session, None, {"cell_id": cell_id, "t": 2})
        _wait_until(lambda: sess.current_index() == 2)
        assert sess.current_index() == 2

    def test_scrub_clamps_to_range(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        M.movie_scrub(session, None, {"cell_id": cell_id, "t": 999})   # past the end
        _wait_until(lambda: sess.current_index() == 7)
        assert sess.current_index() == 7      # 8 frames → max index 7

    def test_close_drops_session(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        assert cell_id in M._sessions(session._report)
        M.movie_close(session, None, {"cell_id": cell_id})
        assert cell_id not in M._sessions(session._report)


# ── export (gif is ffmpeg-free; mp4 needs ffmpeg) ────────────────────────────────

class TestExport:
    def test_export_gif_emits_done_and_bakes_poster(self, movie_dataset, tmp_path):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        out = str(tmp_path / "out.gif")
        M.movie_export(session, None, {"cell_id": cell_id, "path": out})
        done = _poll(messages, "movie_done")
        assert done is not None, [m for m in messages if m.get("type") == "error"]
        assert done["path"] == out
        assert done["frames"] == 8            # full 0..7 range, no stride
        assert os.path.exists(out) and os.path.getsize(out) > 0
        # Poster baked into _baked and surfaced on report_state.
        mgr = session._report
        assert cell_id in mgr._baked
        assert mgr._baked[cell_id][:8] == b"\x89PNG\r\n\x1a\n"

    def test_freeze_grows_frame_count(self, movie_dataset, tmp_path):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        # 8 base frames; freeze t=3 hold 1.0s @ 10fps → +10 → 18.
        M.movie_tune(session, None,
                     {"cell_id": cell_id, "params": {"fps": 10},
                      "freezes": [{"t": 3, "hold_s": 1.0}]})
        out = str(tmp_path / "frz.gif")
        M.movie_export(session, None, {"cell_id": cell_id, "path": out})
        done = _poll(messages, "movie_done")
        assert done is not None, [m for m in messages if m.get("type") == "error"]
        assert done["frames"] == 18

    def test_export_respects_crop(self, movie_dataset, tmp_path):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        # A crop WITHIN the 32x32 source frame → a 20x20 output (even, no downsample).
        M.movie_tune(session, None, {"cell_id": cell_id, "crop": [6, 6, 26, 26]})
        out = str(tmp_path / "crop.gif")
        M.movie_export(session, None, {"cell_id": cell_id, "path": out})
        done = _poll(messages, "movie_done")
        assert done is not None
        import imageio
        with imageio.get_reader(out) as r:
            frame = r.get_data(0)
        # 20x20 crop (already even) → gif frame is 20x20, proving the crop applied.
        assert frame.shape[0] == 20 and frame.shape[1] == 20


# ── overlays (Phase 2: time-gated text, ROI, freeze, 1-D-signal-as-text) ──────────

class TestOverlays:
    def test_tune_persists_annotations_and_freezes(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        M.movie_tune(session, None, {
            "cell_id": cell_id,
            "annotations": [
                {"kind": "text", "text": "hi", "xy": [4, 4], "time_range": [0.1, 0.3]},
                {"kind": "rect", "xy": [8, 8], "wh": [20, 20], "color": "#ffcc00"}],
            "freezes": [{"t": 2, "hold_s": 0.5}]})
        cell = session._report.doc.cell_by_id(cell_id)
        assert len(cell.movie.annotations) == 2
        assert cell.movie.annotations[0]["kind"] == "text"
        assert cell.movie.annotations[1]["kind"] == "rect"
        assert cell.movie.freezes == [{"t": 2, "hold_s": 0.5}]
        # The editor state echoes them back.
        st = _latest(messages, "movie_state")
        assert len(st["annotations"]) == 2
        assert st["freezes"] == [{"t": 2, "hold_s": 0.5}]

    def test_annotation_widgets_drag_persists_to_spec(self, movie_dataset):
        # Text/rect annotations become DRAGGABLE anyplotlib widgets on the live
        # signal figure; a widget drag persists the moved geometry (source px) back
        # to the spec. (Widget coords == image px == source px for a signal frame.)
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        assert sess._signal_plot2d() is not None, "no live signal Plot2D"
        M.movie_tune(session, None, {"cell_id": cell_id, "annotations": [
            {"kind": "text", "text": "Label", "xy": [10, 20], "time_range": [0, 1]},
            {"kind": "rect", "xy": [5, 5], "wh": [30, 30], "time_range": [0, 1]}]})
        assert len(sess._ann_widgets) == 2, "annotation widgets not attached"

        class _Ev:
            pass
        # Drag the text widget.
        wt = sess._ann_widgets[0]
        wt._data = {"x": 40.0, "y": 50.0}
        ev = _Ev(); ev.source = wt
        sess._widget_handlers[0](ev)
        cell = session._report.doc.cell_by_id(cell_id)
        assert cell.movie.annotations[0]["xy"] == [40, 50]
        # Drag the rect widget.
        wr = sess._ann_widgets[1]
        wr._data = {"x": 12.0, "y": 14.0, "w": 22.0, "h": 26.0}
        ev2 = _Ev(); ev2.source = wr
        sess._widget_handlers[1](ev2)
        assert cell.movie.annotations[1]["xy"] == [12, 14]
        assert cell.movie.annotations[1]["wh"] == [22, 26]
        # Close clears the widgets off the live plot.
        M.movie_close(session, None, {"cell_id": cell_id})
        assert not sess._ann_widgets

    def test_overlay_image_composited_in_export(self, movie_dataset, tmp_path):
        # A 2nd-image overlay (a bright square) composites onto the base movie's
        # dark region — the exported frame brightens where the overlay lands.
        import hyperspy.api as hs
        import numpy as np
        import dask.array as da
        import imageio.v3 as iio
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        # A 2nd in-situ movie: a bright square bottom-left (base there is dark).
        ov = np.zeros((8, 32, 32), dtype=np.uint16)
        ov[:, 20:30, 2:12] = 1000
        osig = hs.signals.Signal2D(da.from_array(ov, chunks=(1, 32, 32))).as_lazy()
        osig.metadata.set_item("General.title", "overlay")
        osig.axes_manager.navigation_axes[0].scale = 0.1
        osig.set_signal_type("insitu")
        session._add_signal(osig, source_path=None)

        def _is_ov(p):
            tr = getattr(p, "signal_tree", None)
            root = getattr(tr, "root", None)
            return (root is not None
                    and getattr(root, "_signal_type", None) == "insitu"
                    and not getattr(p, "is_navigator", False)
                    and str(root.metadata.get_item("General.title", default="")) == "overlay")
        oplot = next(p for p in session._plots if _is_ov(p))

        out0 = str(tmp_path / "base.gif")
        M.movie_export(session, None, {"cell_id": cell_id, "path": out0})
        assert _poll(messages, "movie_done") is not None
        bl0 = float(np.asarray(iio.imread(out0))[0][20:30, 2:12].mean())

        M.movie_add_overlay_image(session, None, {
            "cell_id": cell_id, "source_window_id": oplot.window_id,
            "alpha": 0.8, "cmap": "gray"})
        cell = session._report.doc.cell_by_id(cell_id)
        assert cell.movie.overlay_image is not None
        out1 = str(tmp_path / "ov.gif")
        # Clear the movie_done marker so the poll waits for the NEW export.
        messages.clear()
        M.movie_export(session, None, {"cell_id": cell_id, "path": out1})
        assert _poll(messages, "movie_done") is not None
        bl1 = float(np.asarray(iio.imread(out1))[0][20:30, 2:12].mean())
        assert bl1 > bl0 + 30, f"overlay did not brighten ({bl1} vs {bl0})"
        # Clear removes it.
        M.movie_add_overlay_image(session, None, {"cell_id": cell_id, "clear": True})
        assert cell.movie.overlay_image is None

    def test_crop_offsets_annotation_coordinates_in_export(self, movie_dataset, tmp_path):
        # A time-gated annotation placed in source coords must land at (xy - crop
        # origin) in the cropped export (the review's crop-origin bug). We place a
        # bright rect at source (18,18) and crop [12,12,32,32]; in the 20x20 output
        # it must appear near (6,6), NOT off-frame at (18,18).
        import numpy as np
        import imageio.v3 as iio
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        # A red TEXT marker at source (14,14). With crop [12,12,32,32] it maps to
        # (2,2) in the 20x20 output (top-left); WITHOUT the origin fix it would draw
        # at (14,14) (bottom-right). Text has a black shadow, so the red channel in
        # the correct quadrant is unambiguous.
        M.movie_tune(session, None, {
            "cell_id": cell_id,
            "crop": [12, 12, 32, 32],
            "annotations": [{"kind": "text", "text": "XXXX", "xy": [14, 14],
                             "size": 8, "color": "#ff0000", "time_range": [0, 100]}]})
        out = str(tmp_path / "crop_ann.gif")
        M.movie_export(session, None, {"cell_id": cell_id, "path": out})
        assert _poll(messages, "movie_done") is not None
        f = np.asarray(iio.imread(out))[0]           # 20x20
        assert f.shape[0] == 20 and f.shape[1] == 20

        # Count strongly-red pixels (R high, G/B low — the marker) per quadrant.
        def red_px(reg):
            r = reg.astype(np.int32)
            return int(((r[..., 0] > 150) & (r[..., 1] < 100) & (r[..., 2] < 100)).sum())
        tl = red_px(f[0:12, 0:12])                   # where cropped (2,2) lands
        br = red_px(f[8:20, 8:20])                    # where an un-offset (14,14) lands
        assert tl > 0, "red marker not found in the top-left (crop origin not applied)"
        assert tl > br, f"crop origin not applied — red not in TL ({tl} vs {br})"

    def test_static_2d_overlay_image_composites(self, movie_dataset, tmp_path):
        # A STATIC 2-D image (not a time stack) as the 2nd-image overlay must
        # composite as one fixed image on every frame (review bug: rows-as-frames).
        import hyperspy.api as hs
        import numpy as np
        import imageio.v3 as iio
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        img2d = np.zeros((32, 32), dtype=np.uint16)
        img2d[20:30, 2:12] = 1000                    # bright square bottom-left
        static = hs.signals.Signal2D(img2d)          # a plain 2-D image (ndim==2 root)
        static.metadata.set_item("General.title", "static")
        session._add_signal(static, source_path=None)
        sp = next(p for p in session._plots
                  if getattr(getattr(p, "signal_tree", None), "root", None) is not None
                  and str(p.signal_tree.root.metadata.get_item("General.title", default="")) == "static"
                  and getattr(p, "current_data", None) is not None
                  and np.asarray(p.current_data).ndim == 2)
        M.movie_add_overlay_image(session, None, {
            "cell_id": cell_id, "source_window_id": sp.window_id,
            "alpha": 0.8, "cmap": "gray"})
        out = str(tmp_path / "static_ov.gif")
        M.movie_export(session, None, {"cell_id": cell_id, "path": out})
        assert _poll(messages, "movie_done") is not None
        f = np.asarray(iio.imread(out))[0]
        assert float(f[20:30, 2:12].mean()) > 60, "static overlay not composited"

    def test_speed_segment_slows_the_export(self, movie_dataset, tmp_path):
        # A 0.25× slow-mo segment over part of the movie makes the export LONGER
        # (more output frames) than the plain range. (Variable-speed timeline.)
        # The frame count is asserted from the authoritative movie_done + output_info
        # (a GIF re-encode may de-dup repeated frames on disk, so we don't count the
        # file). The output_info readout must match the exported frame count.
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        out0 = str(tmp_path / "base.gif")
        M.movie_export(session, None, {"cell_id": cell_id, "path": out0})
        assert _poll(messages, "movie_done")["frames"] == 8
        messages.clear()
        # 0.25× over the first half (t 0.0–0.3s at 0.1 s/frame → frames 0..3).
        M.movie_tune(session, None, {"cell_id": cell_id,
                                     "speed_segments": [{"time_range": [0.0, 0.3], "speed": 0.25}]})
        # The readout (output_info) reflects the slow-mo frame count.
        st = _latest(messages, "movie_state")
        assert st["output_info"]["frames"] > 8, "output_info did not grow with slow-mo"
        out1 = str(tmp_path / "slow.gif")
        M.movie_export(session, None, {"cell_id": cell_id, "path": out1})
        done = _poll(messages, "movie_done")
        assert done is not None
        assert done["frames"] > 8, f"slow-mo did not add frames ({done['frames']})"
        assert done["frames"] == st["output_info"]["frames"]   # readout == export

    def test_time_gated_annotation_shows_only_in_window(self, movie_dataset):
        # A text annotation's live widget is hidden outside its time_range and shown
        # inside it, as the scrubber crosses. (Live time-gating on the figure.)
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        M.movie_tune(session, None, {"cell_id": cell_id, "annotations": [
            {"kind": "text", "text": "X", "xy": [4, 4], "time_range": [0.3, 0.5]}]})
        w = sess._ann_widgets[0]
        M.movie_scrub(session, None, {"cell_id": cell_id, "t": 0})   # 0.0s, outside
        _wait_until(lambda: sess.current_index() == 0)
        assert w.visible is False
        M.movie_scrub(session, None, {"cell_id": cell_id, "t": 4})   # 0.4s, inside
        _wait_until(lambda: sess.current_index() == 4)
        assert w.visible is True

    def test_render_params_apply_to_live_figure(self, movie_dataset):
        # cmap / contrast / axes-toggle push onto the live signal plot.
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        p2 = sess._signal_plot2d()
        M.movie_tune(session, None, {"cell_id": cell_id,
                                     "params": {"clim": [10.0, 900.0], "axes": False}})
        assert p2._state.get("has_axes") is False
        assert float(p2._state.get("display_min")) == 10.0
        assert float(p2._state.get("display_max")) == 900.0

    def test_crop_zooms_the_live_figure(self, movie_dataset):
        # Applying a crop zooms the live figure into the region (set_view).
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        p2 = sess._signal_plot2d()
        M.movie_crop_mode(session, None, {"cell_id": cell_id, "on": True})
        w = sess._crop_widget
        w._data = {"x": 8.0, "y": 8.0, "w": 8.0, "h": 8.0}

        class _Ev:
            pass
        ev = _Ev(); ev.source = w
        sess._crop_handler(ev)
        assert sess.cell.movie.crop == [8, 8, 16, 16]
        assert float(p2._state.get("zoom", 1)) > 1.5   # zoomed in

    def test_play_emits_frames_and_stops(self, movie_dataset):
        import time
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        M.movie_play(session, None, {"cell_id": cell_id})
        # Poll for >=2 movie_frame events (early exit) instead of a flat 0.8 s
        # play window.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if len([m for m in messages
                    if m.get("type") == "movie_frame"]) >= 2:
                break
            time.sleep(0.02)
        M.movie_stop(session, None, {"cell_id": cell_id})
        frames = [m for m in messages if m.get("type") == "movie_frame"]
        assert len(frames) >= 2, "play did not emit movie_frame events"

    def test_crop_widget_on_figure_persists(self, movie_dataset):
        # Crop mode adds a draggable rect widget on the live figure; dragging it
        # persists spec.crop (source px), and clear resets it.
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        M.movie_crop_mode(session, None, {"cell_id": cell_id, "on": True})
        assert sess._crop_widget is not None, "no crop widget appeared"

        class _Ev:
            pass
        w = sess._crop_widget
        w._data = {"x": 6.0, "y": 6.0, "w": 20.0, "h": 20.0}
        ev = _Ev(); ev.source = w
        sess._crop_handler(ev)
        cell = session._report.doc.cell_by_id(cell_id)
        assert cell.movie.crop == [6, 6, 26, 26]
        M.movie_crop_mode(session, None, {"cell_id": cell_id, "clear": True})
        assert cell.movie.crop is None
        assert sess._crop_widget is None

    def test_add_text_overlay_from_1d_window(self, movie_dataset):
        import hyperspy.api as hs
        import numpy as np
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        # A 1-D signal window (e.g. a temperature trace).
        line = hs.signals.Signal1D(np.linspace(20, 900, 8).astype(np.float32))
        line.metadata.set_item("General.title", "temperature")
        session._add_signal(line, source_path=None)
        line_plot = next(p for p in session._plots
                         if getattr(p, "current_data", None) is not None
                         and np.asarray(p.current_data).ndim == 1)
        M.movie_add_text_overlay(session, None, {
            "cell_id": cell_id, "source_window_id": line_plot.window_id,
            "label": "T"})
        cell = session._report.doc.cell_by_id(cell_id)
        assert len(cell.movie.text_overlays) == 1
        ov = cell.movie.text_overlays[0]
        assert ov["label"] == "T"
        assert isinstance(ov.get("source"), dict)      # a SignalRef dict
        # The editor state ships the overlay WITHOUT the ephemeral _trace.
        st = _latest(messages, "movie_state")
        assert len(st["text_overlays"]) == 1
        assert "_trace" not in st["text_overlays"][0]

    def test_text_overlay_live_value_on_figure(self, movie_dataset):
        # A 1-D-signal-as-text overlay puts a LABEL widget on the live figure whose
        # text is the signal's value at the current frame (index-aligned per frame),
        # and it updates as you scrub. Uses a per-frame temperature column.
        import hyperspy.api as hs
        import numpy as np
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        # 8 frames → a per-frame temperature [100, 200, …, 800].
        temp = hs.signals.Signal1D(np.arange(1, 9, dtype=np.float32) * 100)
        temp.metadata.set_item("General.title", "temperature")
        temp.axes_manager.signal_axes[0].units = "C"
        session._add_signal(temp, source_path=None)

        def _is_temp(p):
            if getattr(p, "current_data", None) is None \
                    or np.asarray(p.current_data).ndim != 1:
                return False
            tr = getattr(p, "signal_tree", None)
            try:
                return str(tr.root.metadata.get_item(
                    "General.title", default="")) == "temperature"
            except Exception:
                return False
        tline = next(p for p in session._plots if _is_temp(p))
        M.movie_add_text_overlay(session, None, {
            "cell_id": cell_id, "source_window_id": tline.window_id, "label": "T"})
        sess = M._sessions(session._report)[cell_id]
        assert len(sess._text_overlay_widgets) == 1, "no live text-overlay widget"
        lw = sess._text_overlay_widgets[0][0]
        assert "100" in lw.text, lw.text          # frame 0 → 100
        M.movie_scrub(session, None, {"cell_id": cell_id, "t": 5})
        _wait_until(lambda: sess.current_index() == 5)
        sess.refresh_text_overlays()
        assert "600" in lw.text, lw.text          # frame 5 → 600

    def test_text_overlay_value_in_export(self, movie_dataset, tmp_path):
        """A 1-D-signal-as-text overlay resamples the source onto the movie time and
        renders the current value — the export must run without error and produce a
        movie (the value-drawing path is exercised)."""
        import hyperspy.api as hs
        import numpy as np
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        line = hs.signals.Signal1D(np.linspace(20, 900, 8).astype(np.float32))
        line.metadata.set_item("General.title", "T")
        session._add_signal(line, source_path=None)
        lp = next(p for p in session._plots
                  if getattr(p, "current_data", None) is not None
                  and np.asarray(p.current_data).ndim == 1)
        M.movie_add_text_overlay(session, None,
                                 {"cell_id": cell_id, "source_window_id": lp.window_id})
        out = str(tmp_path / "txt.gif")
        M.movie_export(session, None, {"cell_id": cell_id, "path": out})
        done = _poll(messages, "movie_done")
        assert done is not None, [m for m in messages if m.get("type") == "error"]
        assert done["frames"] == 8
        assert os.path.exists(out) and os.path.getsize(out) > 0


# ── OVERLAY SCOPING: the editor's live-plot widgets must not leak (laundry #6) ────
#
# Root cause: annotation/crop/overlay-image widgets are drawn on the tree's REAL
# signal Plot2D (the movie editor never builds its own figure — see movie.py's
# module docstring). `movie_close` already tore them down correctly
# (`_teardown_session` -> `unbind_live_windows` -> `clear_overlay_widgets`,
# covered by `test_annotation_widgets_drag_persists_to_spec` above). But every
# OTHER exit path that can drop a `MovieEditSession` did NOT route through that
# same teardown:
#   * `ReportManager._clear_movie_sessions` (report_new / report_close) only
#     flipped the export cancel flag and cleared the session dict — the widgets
#     stayed on the live plot.
#   * `report_remove_cell` had no branch for cell_type == "movie" at all.
# Both are fixed in handlers.py to route through movie.py's `_teardown_session`
# (the SAME function `movie_close` uses), so the widget cleanup can't drift
# between exit paths.

class TestOverlayScopingAcrossExitPaths:
    def _add_two_annotations(self, session, cell_id):
        M.movie_tune(session, None, {"cell_id": cell_id, "annotations": [
            {"kind": "text", "text": "Label", "xy": [10, 20], "time_range": [0, 1]},
            {"kind": "rect", "xy": [5, 5], "wh": [30, 30], "time_range": [0, 1]}]})

    def test_report_new_tears_down_widgets_but_keeps_cell_persistence(self, movie_dataset):
        # A "New" report (or Open, or explicit report_close) while the editor is
        # open must drop the widgets off the LIVE plot — the cell model itself
        # (in the OLD doc, which report_new discards) is a separate concern; what
        # matters here is the live Plot2D's widget registry is left clean, not
        # still painting a dead session's overlays.
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        p2 = sess._signal_plot2d()
        assert p2 is not None
        self._add_two_annotations(session, cell_id)
        widget_ids = [getattr(w, "id", None) for w in sess._ann_widgets.values()]
        assert len(widget_ids) == 2
        assert all(wid in p2._widgets for wid in widget_ids), \
            "annotation widgets must be registered on the live Plot2D"

        # New report (mirrors the "New"/"Open" sidebar action) — this is the
        # report-wide exit path that used to leak.
        H.report_new(session, None, {"type": "report"})

        assert cell_id not in M._sessions(session._report), \
            "the movie session must not survive a report_new"
        assert not sess._ann_widgets, "session's widget bookkeeping must be cleared"
        assert not any(wid in p2._widgets for wid in widget_ids), \
            "annotation widgets must be removed from the live Plot2D by report_new"

    def test_report_close_tears_down_widgets(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        p2 = sess._signal_plot2d()
        self._add_two_annotations(session, cell_id)
        widget_ids = [getattr(w, "id", None) for w in sess._ann_widgets.values()]

        from spyde.actions.report.handlers import report_close
        report_close(session, None, {})

        assert not sess._ann_widgets
        assert not any(wid in p2._widgets for wid in widget_ids)

    def test_remove_movie_cell_tears_down_widgets(self, movie_dataset):
        # Deleting the movie cell itself (report_remove_cell) while its editor
        # session is open must ALSO clear the live-plot widgets — previously
        # report_remove_cell had no branch for cell_type == "movie" at all, so
        # the session (and its widgets) just leaked silently.
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        p2 = sess._signal_plot2d()
        self._add_two_annotations(session, cell_id)
        widget_ids = [getattr(w, "id", None) for w in sess._ann_widgets.values()]
        assert all(wid in p2._widgets for wid in widget_ids)

        from spyde.actions.report.handlers import report_remove_cell
        report_remove_cell(session, None, {"cell_id": cell_id})

        assert cell_id not in M._sessions(session._report)
        assert session._report.doc.cell_by_id(cell_id) is None, \
            "the cell must be removed from the document"
        assert not sess._ann_widgets
        assert not any(wid in p2._widgets for wid in widget_ids)

    def test_reopen_after_close_restores_widgets_from_persisted_spec(self, movie_dataset):
        # PERSISTENCE STAYS: the annotations live on cell.movie.annotations
        # regardless of the editor's open/close state. Closing must remove the
        # live widgets; reopening must rebuild them from that SAME persisted
        # list (not lose them).
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        self._add_two_annotations(session, cell_id)
        cell = session._report.doc.cell_by_id(cell_id)
        assert len(cell.movie.annotations) == 2          # persisted on the spec

        M.movie_close(session, None, {"cell_id": cell_id})
        assert cell_id not in M._sessions(session._report)
        # The model survives the close untouched.
        cell_after_close = session._report.doc.cell_by_id(cell_id)
        assert len(cell_after_close.movie.annotations) == 2

        M.movie_open(session, None, {"cell_id": cell_id})
        sess2 = M._sessions(session._report)[cell_id]
        p2b = sess2._signal_plot2d()
        assert len(sess2._ann_widgets) == 2, "reopen must rebuild widgets from the spec"
        widget_ids2 = [getattr(w, "id", None) for w in sess2._ann_widgets.values()]
        assert all(wid in p2b._widgets for wid in widget_ids2)

    def test_view_outside_editor_after_close_shows_no_overlays(self, movie_dataset):
        # "Viewing the same tree's signal window normally" after the editor
        # closes must not show the movie's overlays — i.e. the live Plot2D the
        # rest of the app renders (sess.signal_plot._plot2d, the SAME object
        # WindowContent's iframe reflects) carries no leftover widgets.
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        live_plot = sess.signal_plot
        p2 = live_plot._plot2d
        self._add_two_annotations(session, cell_id)
        assert p2._widgets, "sanity: widgets present while editor open"

        M.movie_close(session, None, {"cell_id": cell_id})

        # live_plot is the SAME Plot the MDI window shows outside the editor —
        # its Plot2D must be clean.
        assert not p2._widgets, \
            "the live signal plot still carries movie overlay widgets after close"


# ── TILE MODE: the editor's frame updates must route through anyplotlib tile
#    mode for a LARGE frame (laundry #13) ──────────────────────────────────────
#
# The movie editor never builds its own figure/paint path — it drives the
# tree's REAL 1-D time-navigator selector (`sess.time_selector`), whose update
# fires the SAME `Plot._set_array` every other nav move uses (see
# `update_from_navigation_selection` -> `child.enqueue_paint` ->
# `Plot._set_array` -> `Plot._maybe_gpu_tile`). So a movie whose frames are
# already >= Plot._GPU_TILE_MIN_EDGE (1024 px edge) takes the SAME tile path
# scrubbing through the editor as scrubbing the plain MDI window; there is no
# second/bespoke tile implementation to fork. This is verified directly against
# `Plot._maybe_gpu_tile` (spy/monkeypatch) using a synthetic movie built the
# same way `load_test_data_movie` builds its (smaller, to keep the test fast)
# large-frame synthetic movie (`_build_movie_frames`, shared with the tutorial
# loader — see spyde/backend/tutorial_data.py).

def _big_movie_dataset(session, edge=1100, n_frames=3):
    """A lazy in-situ movie whose frames are >= Plot._GPU_TILE_MIN_EDGE (1024),
    1 frame/chunk (mirrors the real storage-aligned chunking a large in-situ
    movie loads with)."""
    import dask.array as da
    import hyperspy.api as hs
    from spyde.backend.tutorial_data import _build_movie_frames
    frames = _build_movie_frames(edge, n_frames).astype(np.float32)
    stack = da.from_array(frames, chunks=(1, edge, edge))
    s = hs.signals.Signal2D(stack).as_lazy()
    tax = s.axes_manager.navigation_axes[0]
    tax.name, tax.units, tax.scale = "time", "sec", 0.1
    s.set_signal_type("insitu")
    session._add_signal(s, source_path=None)
    _settle(session)


class TestTileModeInEditor:
    def test_scrub_in_editor_routes_through_gpu_tile(self, window):
        from spyde.drawing.plots.plot import Plot
        session, messages = window["window"], window["messages"]
        _big_movie_dataset(session)
        plot = _insitu_plot(session)
        assert plot is not None, "no big in-situ plot loaded"

        H.report_new(session, None, {"type": "report"})
        M.report_add_movie_cell(session, None,
                                {"source_window_id": plot.window_id, "caption": "big"})
        rs = _latest(messages, "report_state")
        cell_id = next(c for c in rs["report"]["cells"] if c["cell_type"] == "movie")["id"]

        calls = []
        orig = Plot._maybe_gpu_tile

        def spy(self, data, clim, axes, units):
            calls.append((self.window_id, tuple(data.shape)))
            return orig(self, data, clim, axes, units)

        with patch.object(Plot, "_maybe_gpu_tile", spy):
            M.movie_open(session, None, {"cell_id": cell_id})
            sess = M._sessions(session._report)[cell_id]
            assert sess.signal_plot is not None
            assert sess.signal_plot.window_id == plot.window_id

            before = len(calls)
            M.movie_scrub(session, None, {"cell_id": cell_id, "t": 1})
            _wait_until(lambda: sess.current_index() == 1)
            assert sess.current_index() == 1
            # The read (nav index) and the PAINT are decoupled (Live-Display §3:
            # a serial newest-wins _NavPainter thread runs the actual
            # _set_array/_maybe_gpu_tile call after the index updates) — wait
            # for the tile call itself, not just the index.
            _wait_until(lambda: len(calls) > before)

        assert len(calls) > before, \
            "scrubbing in the movie editor did not reach Plot._maybe_gpu_tile " \
            "(the tile flow) for a >=1024px frame"
        assert all(max(shape) >= 1024 for _wid, shape in calls[before:])
        # Reviewer hardening: prove tile mode actually ENGAGED (a backend was
        # installed), not merely that the entry point was reached.
        assert getattr(plot, "_gpu_tile_backend", None) is not None, \
            "tile entry point ran but no tile backend was installed"

    def test_scrub_in_editor_keeps_using_same_plot_as_outside_editor(self, window):
        # The editor must not fork a second display path: the Plot object it
        # scrubs through (sess.signal_plot) is the IDENTICAL object the MDI
        # window (outside the editor) paints — same _plot2d, same tile backend
        # state, so "zoom persists across scrub" falls out for free (anyplotlib
        # owns the tile loop on that one Plot2D; nothing here resets it).
        session, messages = window["window"], window["messages"]
        _big_movie_dataset(session)
        plot = _insitu_plot(session)
        H.report_new(session, None, {"type": "report"})
        M.report_add_movie_cell(session, None,
                                {"source_window_id": plot.window_id, "caption": "big"})
        rs = _latest(messages, "report_state")
        cell_id = next(c for c in rs["report"]["cells"] if c["cell_type"] == "movie")["id"]
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        assert sess.signal_plot is plot, \
            "the editor must surface the SAME live Plot the MDI window uses, " \
            "not a copy/second figure"
        assert sess._signal_plot2d() is plot._plot2d


# ── MEMORY SAFETY: neither preview nor export computes the full dataset ────────────

def _guarded_compute(real):
    """Raise if ``.compute()`` is ever called on the FULL (8, H, W) movie array —
    the preview/export must slice ONE frame at a time. A single-frame slice
    ``raw[t]`` has 2-D shape and is allowed."""
    def compute(self, *args, **kwargs):
        shape = tuple(self.shape)
        if len(shape) == 3 and shape[0] == 8:
            raise AssertionError(
                f"Full-dataset .compute() on shape {shape} — a movie preview/"
                f"export must slice one frame at a time.")
        return real(self, *args, **kwargs)
    return compute


class TestMemorySafety:
    def test_scrub_never_computes_full_array(self, movie_dataset):
        # Scrubbing drives the tree's real navigator, whose per-frame read is a
        # single-frame slice (the app's lazy nav pipeline) — never the full array.
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        sess = M._sessions(session._report)[cell_id]
        with patch.object(da.Array, "compute", _guarded_compute(da.Array.compute)):
            M.movie_scrub(session, None, {"cell_id": cell_id, "t": 5})
            _wait_until(lambda: sess.current_index() == 5)
        assert sess.current_index() == 5

    def test_export_never_computes_full_array(self, movie_dataset, tmp_path):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        cell_id = _add_movie(session, messages)
        M.movie_open(session, None, {"cell_id": cell_id})
        out = str(tmp_path / "safe.gif")
        with patch.object(da.Array, "compute", _guarded_compute(da.Array.compute)):
            M.movie_export(session, None, {"cell_id": cell_id, "path": out})
            done = _poll(messages, "movie_done")
        assert done is not None, [m for m in messages if m.get("type") == "error"]
        assert done["frames"] == 8

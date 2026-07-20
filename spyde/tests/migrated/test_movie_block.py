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
                  "freezes", "crop", "out_size", "frame_size",
                  "signal_fig_id", "signal_window_id", "nav_fig_id",
                  "current_index"):
            assert k in st, f"movie_state missing {k!r}"
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

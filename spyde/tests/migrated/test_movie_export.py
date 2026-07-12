"""
test_movie_export.py — Movie Export (Phase 4) backend tests.

Two layers:

* **Pipeline** (``spyde.actions.movie_export.pipeline`` / ``encoder`` / ``traces``):
  a small synthetic LAZY in-situ movie built in-file (12 × 64×64 uint16 dask, 1
  frame/chunk, distinct per-frame content, calibrated 0.1 s/frame time axis) is
  encoded to mp4 / GIF and read back. Covers: frame count vs stride + time range,
  downsample halving (even-cropped) dimensions, LUT non-blank + differing frames,
  time-gated annotations, trace-inset pixels, cancellation leaving NO partial
  file, ffmpeg-missing error, and the ``patch.object(da.Array.compute)`` guard
  asserting the FULL dataset shape is never computed.

* **Handlers** (``spyde.actions.movie_export.handlers``): mvx_open gate + state
  shape, add/remove trace, run→mvx_done, all against a real Session.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import dask.array as da
import hyperspy.api as hs
import pytest
import imageio

from spyde.actions.movie_export import pipeline as pl
from spyde.actions.movie_export import traces as tr
from spyde.actions.movie_export import handlers as h


# ── synthetic lazy in-situ movie (mirrors load_test_data_movie's construction) ──

N_FRAMES = 12
FRAME = 64
TIME_SCALE_S = 0.1


def _movie_frames():
    """(N_FRAMES, FRAME, FRAME) uint16 with DISTINCT per-frame content: a gradient
    base + a bright vertical band whose x-position encodes the frame index (so
    every frame differs and a stale frame is detectable)."""
    yy, xx = np.mgrid[0:FRAME, 0:FRAME].astype(np.float32)
    base = (xx / FRAME) * 3000.0 + (yy / FRAME) * 3000.0
    frames = np.empty((N_FRAMES, FRAME, FRAME), dtype=np.uint16)
    for t in range(N_FRAMES):
        f = base.copy()
        x0 = (t + 1) * FRAME // (N_FRAMES + 2)
        f[:, x0:x0 + FRAME // 16] = 60000.0        # frame-index band
        frames[t] = f.astype(np.uint16)
    return frames


def _lazy_movie_signal():
    frames = _movie_frames()
    stack = da.from_array(frames, chunks=(1, FRAME, FRAME))   # 1 frame/chunk
    s = hs.signals.Signal2D(stack).as_lazy()
    for ax in s.axes_manager.signal_axes:
        ax.scale, ax.units = 0.5, "nm"
    tax = s.axes_manager.navigation_axes[0]
    tax.name, tax.units, tax.scale = "time", "s", TIME_SCALE_S
    s.set_signal_type("insitu")
    return s


def _raw_lazy():
    return _lazy_movie_signal().data


def _base_params(**over):
    p = dict(fps=10, downsample=1, stride=1, t_start=0, t_end=N_FRAMES - 1,
             cmap="viridis", clim=None, timestamp=True, scalebar=True,
             annotations=[])
    p.update(over)
    return p


def _export(tmp_path, raw=None, params=None, **kw):
    raw = raw if raw is not None else _raw_lazy()
    params = params if params is not None else _base_params()
    path = str(tmp_path / kw.pop("name", "movie.mp4"))
    frames = pl.export_movie(
        raw, path=path, params=params, n_frames=N_FRAMES,
        scale_s=TIME_SCALE_S, sig_scale_x=0.5, sig_units="nm", **kw)
    return path, frames


# ── LUT / downsample units ───────────────────────────────────────────────────────

class TestLUT:
    def test_lut_shape_and_range(self):
        lut = pl.build_lut("viridis")
        assert lut.shape == (256, 3) and lut.dtype == np.uint8

    def test_apply_lut_non_blank_and_differs(self):
        lut = pl.build_lut("viridis")
        f0 = _movie_frames()[0].astype(np.float32)
        f5 = _movie_frames()[5].astype(np.float32)
        lo, hi = float(f0.min()), float(f0.max())
        rgb0 = pl.apply_lut(f0, lut, lo, hi)
        rgb5 = pl.apply_lut(f5, lut, lo, hi)
        assert rgb0.shape == (FRAME, FRAME, 3)
        assert int(rgb0.max()) > 0 and int(rgb0.std()) > 0     # not blank
        assert not np.array_equal(rgb0, rgb5)                  # frames differ

    def test_downsample_halves(self):
        f = _movie_frames()[0].astype(np.float32)
        d = pl.downsample(f, 2)
        assert d.shape == (FRAME // 2, FRAME // 2)


# ── encode + read back ───────────────────────────────────────────────────────────

class TestEncodeReadback:
    def test_mp4_exists_and_frame_count(self, tmp_path):
        path, frames = _export(tmp_path)
        assert os.path.exists(path)
        assert frames == N_FRAMES
        assert imageio.get_reader(path).count_frames() == N_FRAMES

    def test_stride(self, tmp_path):
        path, frames = _export(tmp_path, params=_base_params(stride=3),
                               name="stride.mp4")
        expected = len(range(0, N_FRAMES, 3))
        assert frames == expected
        assert imageio.get_reader(path).count_frames() == expected

    def test_time_range(self, tmp_path):
        path, frames = _export(tmp_path, params=_base_params(t_start=2, t_end=6),
                               name="range.mp4")
        assert frames == 5                                     # 2,3,4,5,6
        assert imageio.get_reader(path).count_frames() == 5

    def test_downsample_dimensions_even_cropped(self, tmp_path):
        path, _ = _export(tmp_path, params=_base_params(downsample=2),
                          name="down.mp4")
        frame = imageio.get_reader(path).get_data(0)
        h_px, w_px = frame.shape[:2]
        assert (h_px, w_px) == (FRAME // 2, FRAME // 2)        # 64→32, already even

    def test_odd_downsample_even_crop(self, tmp_path):
        # downsample 3 → 64//3 = 21 (odd) → even-cropped to 20.
        path, _ = _export(tmp_path, params=_base_params(downsample=3),
                          name="odd.mp4")
        frame = imageio.get_reader(path).get_data(0)
        assert frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0

    def test_gif_decodable(self, tmp_path):
        path, frames = _export(tmp_path, params=_base_params(fps=5),
                               name="movie.gif")
        assert os.path.exists(path) and frames == N_FRAMES
        rdr = imageio.get_reader(path)
        first = rdr.get_data(0)
        assert first.ndim == 3 and first.shape[-1] in (3, 4)


# ── annotations time-gating ──────────────────────────────────────────────────────

class TestAnnotations:
    def test_rect_only_in_frames_3_to_6(self, tmp_path):
        """A rect annotation gated to t in [0.3, 0.6] s (frames 3..6) must make
        those frames differ from a bare export, while other frames match."""
        # A big bright rect that clearly changes pixels.
        rect = {"kind": "rect", "xy": [8, 8], "wh": [40, 40],
                "color": "#ff0000", "width": 6,
                "time_range": [3 * TIME_SCALE_S, 6 * TIME_SCALE_S]}
        bare, _ = _export(tmp_path, params=_base_params(), name="bare.mp4")
        anno, _ = _export(tmp_path, params=_base_params(annotations=[rect]),
                          name="anno.mp4")
        rb = imageio.get_reader(bare)
        ra = imageio.get_reader(anno)
        gated = set(range(3, 7))
        for i in range(N_FRAMES):
            fb = rb.get_data(i).astype(np.int32)
            fa = ra.get_data(i).astype(np.int32)
            diff = np.abs(fb - fa).sum()
            if i in gated:
                assert diff > 0, f"frame {i} should show the gated rect"
            else:
                # ungated frames should be (near) identical — allow tiny codec noise
                assert diff < fb.size * 2, f"frame {i} changed but shouldn't"


# ── trace inset ──────────────────────────────────────────────────────────────────

class TestTraceInset:
    def _trace(self):
        x = np.linspace(0.0, (N_FRAMES - 1) * TIME_SCALE_S, N_FRAMES)
        y = np.sin(np.linspace(0, 4 * np.pi, N_FRAMES))
        return tr.TraceSpec(id="tr1", label="temp", color="#d62728",
                            units="s", x=x, y=y)

    def test_resample_clamps(self):
        spec = self._trace()
        out = spec.resample(np.array([-5.0, 0.0, 100.0]))
        assert out.shape == (3,)
        # clamps to endpoints outside range
        assert out[0] == pytest.approx(spec.y[0])
        assert out[-1] == pytest.approx(spec.y[-1])

    def test_inset_changes_pixels_in_inset_region(self, tmp_path):
        traces = [self._trace()]
        no_trace, _ = _export(tmp_path, params=_base_params(timestamp=False,
                                                            scalebar=False),
                              name="notrace.mp4")
        with_trace, _ = _export(tmp_path, params=_base_params(timestamp=False,
                                                             scalebar=False),
                                traces=traces, name="trace.mp4")
        fb = imageio.get_reader(no_trace).get_data(0).astype(np.int32)
        ft = imageio.get_reader(with_trace).get_data(0).astype(np.int32)
        assert fb.shape == ft.shape
        # Inset is bottom-left → compare the bottom-left quadrant.
        H, W = fb.shape[:2]
        blb = fb[H // 2:, : W // 2]
        blt = ft[H // 2:, : W // 2]
        assert np.abs(blb - blt).sum() > 0, "trace inset must change bottom-left pixels"


# ── cancellation ─────────────────────────────────────────────────────────────────

class TestCancellation:
    def test_cancel_leaves_no_partial_file(self, tmp_path):
        path = str(tmp_path / "cancelled.mp4")
        calls = {"n": 0}

        def should_cancel():
            calls["n"] += 1
            return calls["n"] > 3          # cancel after a few frames

        with pytest.raises(pl._Cancelled):
            pl.export_movie(
                _raw_lazy(), path=path, params=_base_params(), n_frames=N_FRAMES,
                scale_s=TIME_SCALE_S, sig_scale_x=0.5, sig_units="nm",
                should_cancel=should_cancel)
        # export_movie raises; the HANDLER removes the file — emulate that cleanup.
        h._cleanup_partial(path)
        assert not os.path.exists(path)


# ── ffmpeg missing ───────────────────────────────────────────────────────────────

class TestFfmpegMissing:
    def test_ffmpeg_ok_false_when_probe_raises(self, monkeypatch):
        import imageio_ffmpeg
        monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe",
                            lambda: (_ for _ in ()).throw(RuntimeError("no ffmpeg")))
        assert h._ffmpeg_ok() is False


# ── MEMORY SAFETY: the full dataset is NEVER computed ────────────────────────────

def _guarded_compute(real):
    """A drop-in replacement for ``da.Array.compute`` (a plain function so the
    descriptor protocol still binds ``self``) that raises if called on an array
    whose shape is the FULL dataset. A single-frame slice ``raw[t]`` has shape
    (FRAME, FRAME) — allowed; the full (N, FRAME, FRAME) is not."""

    def compute(self, *args, **kwargs):
        shape = tuple(self.shape)
        if len(shape) == 3 and shape[0] == N_FRAMES:
            raise AssertionError(
                f"Full-dataset .compute() called on shape {shape} — the movie "
                f"export must slice one frame at a time.")
        return real(self, *args, **kwargs)

    return compute


class TestMemorySafety:
    def test_never_computes_full_array(self, tmp_path):
        with patch.object(da.Array, "compute", _guarded_compute(da.Array.compute)):
            path, frames = _export(tmp_path, name="safe.mp4")
        assert frames == N_FRAMES
        assert imageio.get_reader(path).count_frames() == N_FRAMES

    def test_read_frame_is_single_frame(self):
        raw = _raw_lazy()
        f = pl.read_frame(raw, 5)
        assert f.shape == (FRAME, FRAME)
        # matches the numpy ground truth for that frame
        assert np.array_equal(f, _movie_frames()[5])


# ── handlers ─────────────────────────────────────────────────────────────────────

def _insitu_plot(session):
    for p in session._plots:
        tree = getattr(p, "signal_tree", None)
        root = getattr(tree, "root", None)
        if root is not None and getattr(root, "_signal_type", None) == "insitu" \
                and not getattr(p, "is_navigator", False):
            return p
    return None


class TestHandlers:
    def test_open_refuses_non_insitu(self, stem_4d_dataset):
        session, messages = stem_4d_dataset["window"], stem_4d_dataset["messages"]
        plot = next(p for p in session._plots if not p.is_navigator)
        h.mvx_open(session, plot, {"window_id": plot.window_id})
        errs = [m for m in messages if m.get("type") == "error"]
        assert any("in-situ" in str(m.get("text", "")).lower() for m in errs)
        assert getattr(plot.signal_tree, "_mvx_state", None) is None

    def test_open_emits_state_shape(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        plot = _insitu_plot(session)
        assert plot is not None
        h.mvx_open(session, plot, {"window_id": plot.window_id})
        states = [m for m in messages if m.get("type") == "mvx_state"]
        assert states, "no mvx_state emitted"
        st = states[-1]
        # exact contract keys
        for k in ("window_id", "ffmpeg_ok", "running", "n_frames", "time",
                  "params", "traces"):
            assert k in st, f"mvx_state missing {k!r}"
        assert set(st["time"]) == {"scale_s", "units"}
        for k in ("fps", "downsample", "stride", "t_start", "t_end", "cmap",
                  "clim", "timestamp", "scalebar", "annotations"):
            assert k in st["params"], f"params missing {k!r}"
        assert st["running"] is False
        assert st["n_frames"] == 8                      # movie_dataset has 8 frames
        assert st["time"]["scale_s"] == pytest.approx(0.1)
        assert st["params"]["t_end"] == 7

    def test_tune_updates_params(self, movie_dataset):
        session = movie_dataset["window"]
        plot = _insitu_plot(session)
        h.mvx_open(session, plot, {"window_id": plot.window_id})
        h.mvx_tune(session, plot, {"window_id": plot.window_id,
                                   "fps": 24, "downsample": 2, "stride": 2,
                                   "cmap": "magma", "timestamp": False})
        st = plot.signal_tree._mvx_state
        assert st.params["fps"] == 24 and st.params["downsample"] == 2
        assert st.params["stride"] == 2 and st.params["cmap"] == "magma"
        assert st.params["timestamp"] is False

    def test_add_and_remove_trace(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        plot = _insitu_plot(session)
        h.mvx_open(session, plot, {"window_id": plot.window_id})
        # Build a 1-D plot to add as a trace: a line-profile-style signal window.
        line = hs.signals.Signal1D(np.sin(np.linspace(0, 6, 40)).astype(np.float32))
        session._add_signal(line, source_path=None)
        line_plot = next(p for p in session._plots
                         if getattr(p, "current_data", None) is not None
                         and np.asarray(p.current_data).ndim == 1)
        h.mvx_add_trace(session, plot, {"window_id": plot.window_id,
                                        "source_window_id": line_plot.window_id})
        st = plot.signal_tree._mvx_state
        assert len(st.traces) == 1
        tid = st.traces[0].id
        state_msg = [m for m in messages if m.get("type") == "mvx_state"][-1]
        assert state_msg["traces"][0]["id"] == tid
        h.mvx_remove_trace(session, plot, {"window_id": plot.window_id,
                                           "trace_id": tid})
        assert len(plot.signal_tree._mvx_state.traces) == 0

    def test_run_emits_mvx_done(self, movie_dataset, tmp_path):
        import time
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        plot = _insitu_plot(session)
        h.mvx_open(session, plot, {"window_id": plot.window_id})
        if not h._ffmpeg_ok():
            pytest.skip("ffmpeg not available")
        path = str(tmp_path / "handler.mp4")
        # run_on_worker spawns a daemon thread (the session has _dispatch_to_main
        # but no running loop → the marshalled _done runs inline on that thread);
        # poll for the mvx_done emission.
        h.mvx_run(session, plot, {"window_id": plot.window_id, "path": path})
        done = []
        for _ in range(200):
            done = [m for m in messages if m.get("type") == "mvx_done"]
            if done:
                break
            time.sleep(0.05)
        assert done, "no mvx_done emitted"
        assert done[-1]["path"] == path
        assert done[-1]["frames"] == 8
        assert os.path.exists(path)

    def test_run_refuses_without_path(self, movie_dataset):
        session, messages = movie_dataset["window"], movie_dataset["messages"]
        plot = _insitu_plot(session)
        h.mvx_open(session, plot, {"window_id": plot.window_id})
        h.mvx_run(session, plot, {"window_id": plot.window_id})
        assert any(m.get("type") == "error" for m in messages)

    def test_close_clears_state(self, movie_dataset):
        session = movie_dataset["window"]
        plot = _insitu_plot(session)
        h.mvx_open(session, plot, {"window_id": plot.window_id})
        assert plot.signal_tree._mvx_state is not None
        h.mvx_close(session, plot, {"window_id": plot.window_id})
        assert plot.signal_tree._mvx_state is None

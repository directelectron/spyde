"""Burn an attached instrument channel into the movie's frames.

The movie editor's original text overlay needed a live 1-D plot window to drag
in. An instrument channel — an electrochemistry potential, a holder temperature
— is already per-frame on the tree once :mod:`spyde.insitu` has aligned it, so
it needs no window at all. This covers that second trace source (the
``from_metadata`` seam ``traces.py`` documented) and the editor plumbing on top.
"""
from __future__ import annotations

import numpy as np
import pytest
import hyperspy.api as hs

from spyde.actions.movie_export import traces as _traces


class _Tree:
    def __init__(self, n=50, scale=0.03276, channels=None):
        self.root = hs.signals.Signal2D(np.zeros((n, 4, 4), np.uint16))
        ax = self.root.axes_manager.navigation_axes[0]
        ax.name, ax.units, ax.scale = "time", "s", scale
        self.insitu_channels = channels if channels is not None else {
            "time/s": np.arange(n) * scale + 31.56,
            "Ewe/V": np.linspace(0.0, 0.8, n),
            "<I>/mA": np.linspace(-1e-3, 1e-3, n),
            "cycle number": np.ones(n),
            "I Range": np.full(n, 52.0),
            "mode": np.full(n, 2.0),
            "ox/red": np.zeros(n),
            "error": np.zeros(n),
        }


class TestFromInsituChannel:
    def test_captures_a_channel_on_the_movie_time_base(self):
        tree = _Tree(n=50, scale=0.03276)
        tr = _traces.from_insitu_channel(tree, "Ewe/V")
        assert tr is not None
        assert tr.label == "Ewe"
        assert tr.units == "V"
        assert tr.y.shape == (50,)
        # x is the MOVIE's own time axis, NOT the instrument clock — an overlay
        # is resampled against movie_times, and the instrument clock carries the
        # alignment lag (here 31.56 s) that would shift every value.
        assert tr.x[0] == pytest.approx(0.0)
        assert tr.x[-1] == pytest.approx(49 * 0.03276)

    def test_bracketed_average_names_are_cleaned(self):
        """EC-Lab writes "<I>/mA"; burnt into a frame that should read "I"."""
        tr = _traces.from_insitu_channel(_Tree(), "<I>/mA")
        assert (tr.label, tr.units) == ("I", "mA")

    def test_missing_channel_returns_none(self):
        assert _traces.from_insitu_channel(_Tree(), "Nope/V") is None

    def test_no_channels_at_all_returns_none(self):
        tree = _Tree(channels={})
        assert _traces.from_insitu_channel(tree, "Ewe/V") is None

    def test_nan_outside_the_record_is_preserved(self):
        """Frames the instrument never covered must stay NaN so the overlay
        paints a dash, not a fabricated value."""
        n = 20
        values = np.linspace(0, 1, n)
        values[-5:] = np.nan
        tree = _Tree(n=n, channels={"Ewe/V": values})
        tr = _traces.from_insitu_channel(tree, "Ewe/V")
        assert np.isnan(tr.y[-5:]).all()
        assert np.isfinite(tr.y[:-5]).all()

    def test_resample_onto_movie_times_is_the_identity(self):
        """The values are already per-frame, so resampling at the frame times
        must return them unchanged — the overlay path is shared with dragged
        1-D traces and must not shift an in-situ channel."""
        tree = _Tree(n=40, scale=0.05)
        tr = _traces.from_insitu_channel(tree, "Ewe/V")
        got = tr.resample(tr.x)
        assert got == pytest.approx(tr.y)


class TestChannelOptions:
    def test_lists_the_meaningful_channels(self):
        options = _traces.insitu_channel_options(_Tree())
        names = [o["channel"] for o in options]
        assert "Ewe/V" in names and "<I>/mA" in names
        # time/s is the axis, not a reading; flags are booleans that read as
        # "0"/"1" burnt into a frame.
        for skipped in ("time/s", "mode", "error", "ox/red"):
            assert skipped not in names

    def test_labels_and_units_are_split(self):
        options = {o["channel"]: o for o in _traces.insitu_channel_options(_Tree())}
        assert options["Ewe/V"]["label"] == "Ewe"
        assert options["Ewe/V"]["units"] == "V"

    def test_all_nan_channel_is_not_offered(self):
        tree = _Tree(n=10, channels={"Ewe/V": np.full(10, np.nan)})
        assert _traces.insitu_channel_options(tree) == []

    def test_no_insitu_data_offers_nothing(self):
        tree = _Tree()
        del tree.insitu_channels
        assert _traces.insitu_channel_options(tree) == []


class _FakeSpec:
    def __init__(self):
        self.text_overlays: list = []
        self.annotations: list = []
        self.params: dict = {"timestamp": False}


class _FakeEditSession:
    """The slice of MovieEditSession the burn-in add path touches."""
    from spyde.actions.report.movie import MovieEditSession
    add_burnin = MovieEditSession.add_burnin
    add_insitu_overlay = MovieEditSession.add_insitu_overlay
    burnin_sources = MovieEditSession.burnin_sources
    insitu_channel_options = MovieEditSession.insitu_channel_options
    _overlays_with_time = MovieEditSession._overlays_with_time
    set_timestamp_enabled = MovieEditSession.set_timestamp_enabled

    def __init__(self, tree):
        self.tree = tree

        class _Cell:
            pass
        self.cell = _Cell()
        self.cell.movie = _FakeSpec()

    def frame_size(self):
        return (512, 512)


class TestAddOverlay:
    def test_adds_an_overlay_bound_to_the_channel(self):
        st = _FakeEditSession(_Tree())
        assert st.add_insitu_overlay("Ewe/V")
        (ov,) = st.cell.movie.text_overlays
        assert ov["insitu_channel"] == "Ewe/V"
        assert ov["label"] == "Ewe" and ov["units"] == "V"
        # No SignalRef — the whole point is that no source window is needed.
        assert "source" not in ov

    def test_unknown_channel_is_refused(self):
        st = _FakeEditSession(_Tree())
        assert not st.add_insitu_overlay("Nope/V")
        assert st.cell.movie.text_overlays == []

    def test_a_flag_channel_is_refused(self):
        """Only what `insitu_channel_options` offers can be added."""
        st = _FakeEditSession(_Tree())
        assert not st.add_insitu_overlay("ox/red")

    def test_successive_overlays_stack_down_the_frame(self):
        st = _FakeEditSession(_Tree())
        st.add_insitu_overlay("Ewe/V")
        st.add_insitu_overlay("<I>/mA")
        ys = [o["xy"][1] for o in st.cell.movie.text_overlays]
        assert ys[1] < ys[0], "the second overlay should not land on the first"
        colors = {o["color"] for o in st.cell.movie.text_overlays}
        assert len(colors) == 2, "each overlay should take its own colour"

    def test_the_row_gap_scales_with_the_frame(self):
        """A fixed 30 px gap is a readable row on a 512 px frame and 0.7% of a
        4096 px one — two overlays then land on the same line and read as one
        duplicated text box."""
        from spyde.actions.report.movie import _overlay_row_step

        class _Big(_FakeEditSession):
            def frame_size(self):
                return (4096, 4096)

        st = _Big(_Tree())
        st.add_insitu_overlay("Ewe/V")
        st.add_insitu_overlay("<I>/mA")
        ys = [o["xy"][1] for o in st.cell.movie.text_overlays]
        assert ys[0] - ys[1] == _overlay_row_step(4096)
        assert ys[0] - ys[1] >= 180, "rows too close to tell apart on a 4k frame"
        # …and a small frame keeps a sane minimum rather than collapsing.
        assert _overlay_row_step(256) == 30


class TestBurnInLegibility:
    """What "the voltage doesn't show on export" actually was: drawn, but at an
    absolute 18 px beside a frame-relative timestamp, so a speck on a 4k movie —
    and formatted `.2f`, which renders a µA current as "0.00"."""

    def test_font_size_scales_with_the_output_frame(self):
        from spyde.actions.movie_export.pipeline import _overlay_font_px
        # The timestamp's own rule is out_h // 28; a default overlay should
        # land in the same ballpark rather than staying at a fixed 18 px.
        assert _overlay_font_px(18, 512) == 18
        assert _overlay_font_px(18, 1024) == 36
        assert _overlay_font_px(18, 1024) == pytest.approx(1024 // 28, abs=4)
        # …and never collapses to nothing on a tiny export.
        assert _overlay_font_px(18, 64) >= 10

    def test_font_size_survives_junk(self):
        from spyde.actions.movie_export.pipeline import _overlay_font_px
        assert _overlay_font_px(None, 512) == 18
        assert _overlay_font_px("big", 512) == 18

    def test_small_magnitudes_keep_their_significant_figures(self):
        from spyde.actions.movie_export.pipeline import _overlay_number
        assert _overlay_number(0.5603) == "0.56"
        # A µA-scale current in mA — ".2f" would render this as "0.00".
        assert _overlay_number(2.67e-4) == "0.000267"
        assert _overlay_number(-6.1e-3) == "-0.0061"
        assert _overlay_number(1.5e6) == "1.5e+06"

    def test_nan_and_junk_paint_a_dash(self):
        from spyde.actions.movie_export.pipeline import _overlay_number
        assert _overlay_number(float("nan")) == "—"
        assert _overlay_number(None) == "—"
        assert _overlay_number([1, 2]) == "—"

    def test_the_overlay_actually_lights_pixels_on_a_rendered_frame(self):
        """End-to-end through the real compose path — the check a green unit
        suite could not make, and the one that would have caught this."""
        from spyde.actions.movie_export import pipeline

        n, edge = 12, 256
        raw = np.full((n, edge, edge), 12000, np.uint16)
        tree = _Tree(n=n)
        tr = _traces.from_insitu_channel(tree, "Ewe/V")
        overlay = {
            "insitu_channel": "Ewe/V", "label": "Ewe", "units": "V",
            "xy": [12, int(edge * 0.85)], "size": 18, "color": "#ffcc00",
            "_trace": tr,
        }
        values = pipeline._resample_text_overlays(
            [overlay], np.arange(n) * 0.03276, src_indices=np.arange(n))
        img = pipeline.render_single_frame(
            raw, 5, params=dict(fps=12, downsample=1, stride=1, cmap="gray",
                                clim=None, timestamp=False, scalebar=False,
                                t_start=0, t_end=n - 1),
            n_frames=n, scale_s=0.03276, sig_scale_x=1.0, sig_units="nm",
            text_overlays=[overlay],
            text_values=[None if v is None else v[5] for v in values],
        )
        a = np.asarray(img)
        lit = np.count_nonzero((a[..., 0] > 180) & (a[..., 1] > 140) & (a[..., 2] < 90))
        assert lit > 60, f"the burnt-in overlay is invisible ({lit} px lit)"


class TestOneAddPath:
    """Static text, the clock and an instrument channel are the same object and
    go through ONE action. They used to have three add paths landing in two
    different lists across two timeline lanes."""

    def test_every_source_produces_a_text_overlay(self):
        st = _FakeEditSession(_Tree())
        for source in ("label", "time", "Ewe/V"):
            assert st.add_burnin(source), source
        kinds = [o.get("builtin") or o.get("insitu_channel")
                 for o in st.cell.movie.text_overlays]
        assert kinds == ["time", "label", "Ewe/V"] or set(kinds) == {
            "time", "label", "Ewe/V"}
        # …and nothing landed in `annotations`, which is for shapes now.
        assert st.cell.movie.annotations == []

    def test_they_all_carry_the_same_editable_fields(self):
        st = _FakeEditSession(_Tree())
        for source in ("label", "time", "Ewe/V"):
            st.add_burnin(source)
        for ov in st.cell.movie.text_overlays:
            for key in ("xy", "size", "color"):
                assert key in ov, f"{ov.get('builtin') or ov.get('insitu_channel')} lacks {key}"

    def test_the_source_list_covers_label_time_and_channels(self):
        sources = [s["source"] for s in _FakeEditSession(_Tree()).burnin_sources()]
        assert sources[:2] == ["label", "time"]
        assert "Ewe/V" in sources and "<I>/mA" in sources

    def test_only_one_clock(self):
        st = _FakeEditSession(_Tree())
        assert st.add_burnin("time")
        assert not st.add_burnin("time"), "a second timestamp makes no sense"

    def test_adding_twice_adds_exactly_two(self):
        """The reported bug was one add showing as two."""
        st = _FakeEditSession(_Tree())
        st.add_burnin("Ewe/V")
        st.add_burnin("Ewe/V")
        assert len(st.cell.movie.text_overlays) == 2

    def test_legacy_text_annotations_migrate_to_overlays(self):
        st = _FakeEditSession(_Tree())
        st.cell.movie.annotations = [
            {"kind": "text", "text": "Before", "xy": [10, 20], "size": 24,
             "color": "#ff0000", "time_range": [0.0, 1.0]},
            {"kind": "rect", "xy": [0, 0], "wh": [10, 10]},
        ]
        overlays = st._overlays_with_time(st.cell.movie)
        labels = [o for o in overlays if o.get("builtin") == "label"]
        assert len(labels) == 1
        assert labels[0]["text"] == "Before"
        assert labels[0]["xy"] == [10, 20]
        assert labels[0]["size"] == 24
        assert labels[0]["time_range"] == [0.0, 1.0]
        # The SHAPE stays an annotation — only text moved.
        assert [a["kind"] for a in st.cell.movie.annotations] == ["rect"]

    def test_a_static_label_draws_its_literal_text(self):
        from spyde.actions.movie_export import pipeline
        n, edge = 4, 96
        raw = np.full((n, edge, edge), 9000, np.uint16)
        ov = pipeline.label_overlay(edge, text="Hello", color="#ffcc00")
        img = pipeline.render_single_frame(
            raw, 1, params=dict(fps=12, downsample=1, stride=1, cmap="gray",
                                clim=None, timestamp=False, scalebar=False,
                                t_start=0, t_end=n - 1),
            n_frames=n, scale_s=0.5, sig_scale_x=1.0, sig_units="nm",
            text_overlays=[ov], text_values=[None])
        a = np.asarray(img)
        lit = np.count_nonzero((a[..., 0] > 180) & (a[..., 1] > 140) & (a[..., 2] < 90))
        assert lit > 20, "a static label overlay drew nothing"


class TestDragPersists:
    """Dragging a burn-in in the editor moved the WIDGET and left the spec's
    `xy` untouched, so the export drew it somewhere else — the editor and the
    movie disagreeing about where the timestamp and the voltage sit."""

    def _dragged(self, index, x, y):
        from spyde.actions.report.movie import _make_burnin_widget_handler
        st = _FakeEditSession(_Tree())
        st.add_burnin("time")
        st.add_burnin("Ewe/V")
        st.mgr = type("M", (), {"dirty": False})()
        st.emit = lambda: None

        class _Ev:
            source = type("W", (), {"_data": {"x": x, "y": y}})()
        _make_burnin_widget_handler(st, index)(_Ev())
        return st

    def test_a_drag_writes_the_new_position_to_the_spec(self):
        st = self._dragged(0, 3100.4, 120.6)
        assert st.cell.movie.text_overlays[0]["xy"] == [3100, 121]

    def test_it_moves_only_the_dragged_one(self):
        st = self._dragged(1, 900, 40)
        assert st.cell.movie.text_overlays[1]["xy"] == [900, 40]
        assert st.cell.movie.text_overlays[0]["xy"] != [900, 40]

    def test_it_marks_the_report_dirty(self):
        assert self._dragged(0, 10, 10).mgr.dirty is True

    def test_an_out_of_range_index_is_ignored(self):
        from spyde.actions.report.movie import _make_burnin_widget_handler
        st = _FakeEditSession(_Tree())
        st.add_burnin("time")
        st.mgr = type("M", (), {"dirty": False})()
        st.emit = lambda: None

        class _Ev:
            source = type("W", (), {"_data": {"x": 5, "y": 5}})()
        _make_burnin_widget_handler(st, 7)(_Ev())     # removed since the build
        assert st.cell.movie.text_overlays[0]["xy"] != [5, 5]

    def test_every_burn_in_label_gets_a_handler_wired(self):
        import inspect
        from spyde.actions.report.movie import MovieEditSession
        src = inspect.getsource(MovieEditSession.sync_overlay_widgets)
        tail = src.split("_text_overlay_widgets[i]", 1)[1]
        assert "_make_burnin_widget_handler" in tail
        assert 'add_event_handler(handler, "pointer_up")' in tail


class TestWidgetLeak:
    """`sync_overlay_widgets` cleared the text-overlay DICT but never removed
    the label widgets from the plot, so every resync stacked another copy of
    every label — one add read as two."""

    def test_resync_removes_the_previous_labels(self):
        import inspect
        from spyde.actions.report.movie import MovieEditSession
        src = inspect.getsource(MovieEditSession.sync_overlay_widgets)
        head = src.split("cur_sec", 1)[0]
        assert "_text_overlay_widgets" in head, (
            "text-overlay widgets are not popped off the plot before a rebuild"
        )
        assert head.count("p2._widgets.pop") >= 2


class TestTimestampIsAnOrdinaryOverlay:
    """The timestamp used to be a bool param drawn at a fixed spot with its own
    font rule and NO presence in the editor — so toggling it changed nothing on
    screen and it could not be moved, recoloured or time-gated like the values
    burnt in beside it. It is now a `builtin: "time"` text overlay.
    """

    def _session(self):
        st = _FakeEditSession(_Tree())
        st.cell.movie.params = {"timestamp": True}
        return st

    def test_a_legacy_param_migrates_to_a_real_overlay(self):
        from spyde.actions.movie_export.pipeline import has_time_overlay
        st = self._session()
        overlays = st._overlays_with_time(st.cell.movie)
        assert has_time_overlay(overlays)
        # …and it is persisted, so it is only synthesised once.
        assert has_time_overlay(st.cell.movie.text_overlays)

    def test_it_carries_the_same_fields_as_any_other_overlay(self):
        st = self._session()
        (ov,) = st._overlays_with_time(st.cell.movie)
        for key in ("label", "units", "xy", "size", "color"):
            assert key in ov, f"timestamp overlay is missing {key!r}"

    def test_toggling_off_removes_it_and_leaves_the_others(self):
        st = self._session()
        st._overlays_with_time(st.cell.movie)
        st.add_insitu_overlay("Ewe/V")
        st.set_timestamp_enabled(False)
        kinds = [o.get("builtin") or o.get("insitu_channel")
                 for o in st.cell.movie.text_overlays]
        assert kinds == ["Ewe/V"]
        st.set_timestamp_enabled(True)
        assert "time" in [o.get("builtin") for o in st.cell.movie.text_overlays]

    def test_removing_the_clip_STICKS(self):
        """The migration must run ONCE. Re-deriving it on every read left the
        legacy `params["timestamp"]` authoritative forever, so deleting the
        timestamp clip from the timeline was undone by the next read."""
        st = self._session()
        st._overlays_with_time(st.cell.movie)
        assert st.cell.movie.text_overlays
        st.cell.movie.text_overlays = []          # the timeline's remove
        again = st._overlays_with_time(st.cell.movie)
        assert [o.get("builtin") for o in again] == []

    def test_migration_does_not_re_add_after_a_toggle_off(self):
        st = self._session()
        st.set_timestamp_enabled(False)
        assert st._overlays_with_time(st.cell.movie) == []
        assert st.cell.movie.params["timestamp"] is False

    def test_legacy_text_annotations_migrate_only_once(self):
        st = _FakeEditSession(_Tree())
        st.cell.movie.annotations = [{"kind": "text", "text": "A", "xy": [1, 2]}]
        st._overlays_with_time(st.cell.movie)
        assert len(st.cell.movie.text_overlays) == 1
        st.cell.movie.text_overlays = []          # user deletes it
        assert st._overlays_with_time(st.cell.movie) == []

    def test_its_value_is_the_frames_own_time(self):
        """No trace to resolve — the editor label rendered a dash before."""
        from spyde.actions.report.movie import _format_overlay_value
        ov = {"builtin": "time", "label": "t", "units": "s"}
        assert _format_overlay_value(ov, 100, 0.03276, 200) == "t = 3.28 s"

    def test_the_burnt_in_frame_draws_it_from_the_overlay_path(self):
        from spyde.actions.movie_export import pipeline

        n, edge = 6, 128
        raw = np.full((n, edge, edge), 9000, np.uint16)
        ov = pipeline.time_overlay(edge, color="#ffcc00")
        params = dict(fps=12, downsample=1, stride=1, cmap="gray", clim=None,
                      timestamp=True, scalebar=False, t_start=0, t_end=n - 1)
        img = pipeline.render_single_frame(
            raw, 3, params=params, n_frames=n, scale_s=0.5,
            sig_scale_x=1.0, sig_units="nm",
            text_overlays=[ov], text_values=[None])
        a = np.asarray(img)
        lit = np.count_nonzero((a[..., 0] > 180) & (a[..., 1] > 140) & (a[..., 2] < 90))
        assert lit > 20, "the timestamp overlay drew nothing"

    def test_the_legacy_path_does_not_double_draw(self):
        """A migrated spec still has `timestamp: True`; the old fixed-position
        draw must stand down once the overlay exists, or the frame carries two
        timestamps."""
        import inspect
        from spyde.actions.movie_export import pipeline
        src = inspect.getsource(pipeline._compose_frame)
        assert "not has_time_overlay(text_overlays)" in src


class TestTimestampColour:
    def test_hex_is_parsed(self):
        from spyde.actions.movie_export.pipeline import _hex_to_rgb
        assert _hex_to_rgb("#ff9100") == (255, 145, 0)
        assert _hex_to_rgb("f90") == (255, 153, 0)

    def test_unset_or_junk_falls_back_to_white(self):
        from spyde.actions.movie_export.pipeline import _TS_COLOR, _hex_to_rgb
        for bad in (None, "", "not-a-colour", "#12345"):
            assert _hex_to_rgb(bad) == _TS_COLOR

    def test_the_default_params_carry_a_timestamp_colour(self):
        from spyde.actions.report.movie import _DEFAULT_PARAMS
        assert _DEFAULT_PARAMS["timestamp_color"] == "#ffffff"

    def test_the_pipeline_threads_the_colour_through(self):
        """The param must reach `_draw_timestamp`, not just sit in the dict."""
        import inspect
        from spyde.actions.movie_export import pipeline
        src = inspect.getsource(pipeline)
        assert 'p.get("timestamp_color")' in src
        assert "_draw_timestamp(img, t_sec, ts_font, ts_color)" in src

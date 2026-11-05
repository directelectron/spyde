"""Wiring: a movie's timestamps calibrate its time axis, and instrument data
recorded beside it attaches as navigator lanes.

The two behaviours pinned here are the ones a green reader suite cannot see:
that the loader PREFERS the timestamps sidecar over the reader's derived
period (which is wrong by ``sum_count**2`` for a summed DE movie), and that
auto-discovery stays quiet unless a record really does explain the movie.
"""
from __future__ import annotations

import numpy as np
import pytest
import hyperspy.api as hs

from spyde.backend._session_files import FileLoaderMixin
from spyde.insitu import attach as attach_mod
from spyde.insitu import de_movie
from spyde.tests.migrated.test_insitu_align import (
    CV_IDS, build_mpr, cv_records, write_movie, write_mpt,
)

N_FRAMES = 120
PERIOD = 0.03276


def movie_signal(n=N_FRAMES, frame=8):  # noqa: D401
    """A minimal in-situ movie signal: nav-dim 1, 2-D frames, time axis."""
    sig = hs.signals.Signal2D(np.zeros((n, frame, frame), np.uint16))
    ax = sig.axes_manager.navigation_axes[0]
    ax.name, ax.units = "time", "sec"
    ax.scale = 1.0 / (61.05006 * 2)      # what the MRC reader would have set
    return sig


def paired(tmp_path, *, n_ec=None, ec_dt=0.08):
    mrc = write_movie(tmp_path, n=N_FRAMES, period=PERIOD, epoch=1763387769)
    duration = (N_FRAMES - 1) * PERIOD
    if n_ec is None:
        n_ec = int(np.ceil(duration / ec_dt)) + 1
    (tmp_path / "r_02_CV_C01.mpr").write_bytes(
        build_mpr(ids=CV_IDS, records=cv_records(n=n_ec, dt_s=ec_dt))
    )
    return mrc


class _FakeTree:
    """Enough tree surface for attach: nav shape + navigator registration.

    ``navigator_signals`` stores a LIST, because that is what the real
    ``BaseSignalTree.add_navigator_signal`` stores — ``_preprocess_navigator``
    returns ``[signal]`` (or ``[navigator, signal]``), never the bare signal it
    was handed. A fake that stored the signal directly let a re-attach bug
    through this suite and all the way into the running app.
    """

    def __init__(self, n=N_FRAMES, source_path=None):
        self.root = movie_signal(n)
        self.navigator_signals: dict = {}
        self.source_path = source_path
        self.registered: list[tuple[str, np.ndarray]] = []

    def add_navigator_signal(self, name, signal):
        self.navigator_signals[name] = [signal]
        self.registered.append((name, np.asarray(signal.data)))

    def lane_data(self, name) -> np.ndarray:
        return np.asarray(self.navigator_signals[name][0].data)


class TestTimeAxisFromTimestamps:
    def test_timestamps_override_the_readers_period(self, tmp_path):
        """The reader's 1/(fps*sum) is 4× too fast for a 2-frame sum; the CSV
        is ground truth and must win."""
        mrc = write_movie(tmp_path, n=N_FRAMES, period=PERIOD)
        sig = movie_signal()
        before = float(sig.axes_manager.navigation_axes[0].scale)

        clock = FileLoaderMixin._apply_frame_timestamps(sig, mrc)

        assert clock is not None
        ax = sig.axes_manager.navigation_axes[0]
        assert ax.scale == pytest.approx(PERIOD, abs=1e-9)
        assert ax.units == "s"
        assert before / ax.scale == pytest.approx(0.25, rel=1e-3)  # was 4× fast

    def test_the_recorded_fps_is_corrected_too(self, tmp_path):
        """The metadata panel prefers the explicit fps KEY over the axis, so
        leaving the reader's camera rate there would report 61 fps beside a
        32.76 ms/frame axis. Fix the value, not the precedence."""
        from spyde.metadata_extract import build_metadata_dict

        mrc = write_movie(tmp_path, n=N_FRAMES, period=PERIOD)
        sig = movie_signal()
        sig.metadata.set_item("Acquisition_instrument.TEM.frames_per_second", 61.05006)

        FileLoaderMixin._apply_frame_timestamps(sig, mrc)

        recorded = sig.metadata.get_item(
            "Acquisition_instrument.TEM.frames_per_second")
        # Rounded for display — the metadata chip prints this number verbatim,
        # so "30.525030441931396 fps" is noise. The AXIS keeps full precision.
        assert recorded == pytest.approx(1 / PERIOD, rel=1e-4)
        assert len(str(recorded)) <= 8, f"unrounded fps on the chip: {recorded}"

        from spyde.tests.migrated.test_movie_metadata import _Tree
        panel = build_metadata_dict(_Tree(sig))["Movie / In-Situ"]
        assert "30.5" in panel["FPS"], panel["FPS"]
        assert "61" not in panel["FPS"], panel["FPS"]

    def test_frame_count_mismatch_keeps_the_period_but_not_the_mapping(self, tmp_path):
        """A stale sidecar from another autosave session still knows the
        camera's frame period — that is a camera property. What it cannot do is
        map ITS frames onto THIS movie's, so no clock is returned."""
        mrc = write_movie(tmp_path, n=N_FRAMES + 40, period=PERIOD)
        sig = movie_signal(n=N_FRAMES)

        clock = FileLoaderMixin._apply_frame_timestamps(sig, mrc)

        assert clock is None
        assert sig.axes_manager.navigation_axes[0].scale == pytest.approx(PERIOD)

    def test_no_sidecar_leaves_the_signal_untouched(self, tmp_path):
        (tmp_path / "bare_movie.mrc").write_bytes(b"")
        sig = movie_signal()
        before = float(sig.axes_manager.navigation_axes[0].scale)
        assert FileLoaderMixin._apply_frame_timestamps(
            sig, str(tmp_path / "bare_movie.mrc")) is None
        assert sig.axes_manager.navigation_axes[0].scale == before

    def test_non_movie_signals_are_skipped(self, tmp_path):
        """A 4D-STEM scan has no frame time base to fix."""
        mrc = write_movie(tmp_path, n=N_FRAMES, period=PERIOD)
        scan = hs.signals.Signal2D(np.zeros((4, 5, 8, 8), np.uint16))
        assert FileLoaderMixin._apply_frame_timestamps(scan, mrc) is None


class TestPixelSize:
    """RosettaSciIO decides imaging-vs-diffraction with ``camera_length != -1``,
    but an IMAGING exposure records 0 — so a TEM image is calibrated as
    diffraction (nm^-1) at the unset -1 pixel size."""

    def _sig(self):
        s = movie_signal(n=4, frame=8)
        # What the reader leaves on a mis-detected imaging exposure.
        for ax, name in zip(s.axes_manager.signal_axes, ("kx", "ky")):
            ax.scale, ax.units, ax.name = -1.0, "nm^-1", name
        return s

    def test_imaging_axes_are_renamed_out_of_reciprocal_space(self, tmp_path):
        """The reader names them kx/ky for the diffraction branch it wrongly
        took; nm axes called "kx" are just as wrong as nm^-1 ones."""
        mrc = write_movie(tmp_path, n=4, info_extra={
            "Instrument Project Camera Length (centimeters)": "0",
            "Specimen Pixel Size X (nanometers)": "1.14786",
            "Specimen Pixel Size Y (nanometers)": "1.14786",
        })
        sig = self._sig()
        FileLoaderMixin._apply_de_pixel_size(sig, mrc)
        assert [ax.name for ax in sig.axes_manager.signal_axes] == ["x", "y"]

    def test_a_user_named_axis_is_not_clobbered(self, tmp_path):
        mrc = write_movie(tmp_path, n=4, info_extra={
            "Instrument Project Camera Length (centimeters)": "0",
            "Specimen Pixel Size X (nanometers)": "1.14786",
            "Specimen Pixel Size Y (nanometers)": "1.14786",
        })
        sig = self._sig()
        for ax in sig.axes_manager.signal_axes:
            ax.name = "my axis"
        FileLoaderMixin._apply_de_pixel_size(sig, mrc)
        assert [ax.name for ax in sig.axes_manager.signal_axes] == ["my axis"] * 2

    def test_imaging_exposure_gets_nm_from_the_specimen_pixel_size(self, tmp_path):
        mrc = write_movie(tmp_path, n=4, info_extra={
            "Instrument Project Camera Length (centimeters)": "0",
            "Diffraction Pixel Size X": "-1",
            "Diffraction Pixel Size Y": "-1",
            "Specimen Pixel Size X (nanometers)": "1.14786",
            "Specimen Pixel Size Y (nanometers)": "1.14786",
        })
        sig = self._sig()
        assert FileLoaderMixin._apply_de_pixel_size(sig, mrc)
        for ax in sig.axes_manager.signal_axes:
            assert ax.scale == pytest.approx(1.14786)
            assert ax.units == "nm"

    def test_real_diffraction_keeps_reciprocal_units(self, tmp_path):
        mrc = write_movie(tmp_path, n=4, info_extra={
            "Instrument Project Camera Length (centimeters)": "80",
            "Diffraction Pixel Size X": "0.0031",
            "Diffraction Pixel Size Y": "0.0031",
            "Specimen Pixel Size X (nanometers)": "-1",
            "Specimen Pixel Size Y (nanometers)": "-1",
        })
        sig = self._sig()
        assert FileLoaderMixin._apply_de_pixel_size(sig, mrc)
        for ax in sig.axes_manager.signal_axes:
            assert ax.scale == pytest.approx(0.0031)
            assert ax.units == "nm^-1"

    def test_no_usable_pixel_size_changes_nothing(self, tmp_path):
        mrc = write_movie(tmp_path, n=4, info_extra={
            "Instrument Project Camera Length (centimeters)": "0",
            "Diffraction Pixel Size X": "-1",
            "Diffraction Pixel Size Y": "-1",
            "Specimen Pixel Size X (nanometers)": "-1",
            "Specimen Pixel Size Y (nanometers)": "-1",
        })
        sig = self._sig()
        assert not FileLoaderMixin._apply_de_pixel_size(sig, mrc)
        assert sig.axes_manager.signal_axes[0].units == "nm^-1"

    def test_a_1d_signal_is_left_alone(self, tmp_path):
        import hyperspy.api as hs
        mrc = write_movie(tmp_path, n=4, info_extra={
            "Specimen Pixel Size X (nanometers)": "1.1",
            "Specimen Pixel Size Y (nanometers)": "1.1",
        })
        line = hs.signals.Signal1D(np.zeros((4, 8), np.uint16))
        assert not FileLoaderMixin._apply_de_pixel_size(line, mrc)


class TestAutoDiscovery:
    def test_attaches_a_matching_record_as_navigator_lanes(self, tmp_path):
        mrc = paired(tmp_path)
        tree = _FakeTree(source_path=mrc)

        result = attach_mod.discover_and_attach(tree, mrc)

        assert result, result.reason
        names = [n for n, _ in tree.registered]
        assert attach_mod.POTENTIAL_LANE in names
        assert any(n.startswith(attach_mod.CURRENT_LANE) for n in names)
        for _, values in tree.registered:
            assert values.shape == (N_FRAMES,)
            assert np.isfinite(values).all(), "a lane must not carry NaN"
        assert result.alignment.method == "span"

    def test_lanes_carry_the_movies_time_calibration(self, tmp_path):
        """A lane's signal axis MUST be the movie's time axis. A 1-D selector
        maps a widget position to a frame with ``(x - offset) / scale`` read
        off the shown navigator's own axis, so an uncalibrated lane plots in
        frame index and its cursor resolves to the wrong frame."""
        mrc = paired(tmp_path)
        tree = _FakeTree(source_path=mrc)
        # The loader calibrates the root's time axis first; the lanes copy it.
        FileLoaderMixin._apply_frame_timestamps(tree.root, mrc)
        attach_mod.discover_and_attach(tree, mrc)

        root_ax = tree.root.axes_manager.navigation_axes[0]
        assert root_ax.scale == pytest.approx(PERIOD, abs=1e-9)
        assert tree.navigator_signals, "no lanes registered"
        for name in tree.navigator_signals:
            lane_ax = tree.navigator_signals[name][0].axes_manager.signal_axes[0]
            assert lane_ax.scale == pytest.approx(root_ax.scale), name
            assert lane_ax.offset == pytest.approx(root_ax.offset), name
            assert lane_ax.units == root_ax.units, name
            # …so the lane's x runs over the movie's DURATION in seconds, not
            # over its frame COUNT.
            span = lane_ax.scale * (N_FRAMES - 1)
            assert span == pytest.approx((N_FRAMES - 1) * PERIOD, rel=1e-6), name

    def test_current_lane_is_named_for_the_unit_it_uses(self, tmp_path):
        """A µA-scale current plotted as mA is an unreadable 0.0001 axis; the
        rescale is fine but the NAME has to say so."""
        mrc = paired(tmp_path)
        tree = _FakeTree(source_path=mrc)
        attach_mod.discover_and_attach(tree, mrc)
        current = [n for n, _ in tree.registered if n.startswith(attach_mod.CURRENT_LANE)]
        assert current == ["I (µA)"]

    def test_silent_when_nothing_matches_the_duration(self, tmp_path):
        """A folder of unrelated records must not produce a false attach."""
        mrc = paired(tmp_path, n_ec=12)     # far too short for the movie
        tree = _FakeTree(source_path=mrc)

        result = attach_mod.discover_and_attach(tree, mrc)

        assert not result
        assert tree.registered == []
        assert "none matching" in result.reason

    def test_silent_with_no_records_at_all(self, tmp_path):
        mrc = write_movie(tmp_path, n=N_FRAMES, period=PERIOD)
        tree = _FakeTree(source_path=mrc)
        result = attach_mod.discover_and_attach(tree, mrc)
        assert not result
        assert "no EC-Lab records" in result.reason

    def test_refuses_when_the_movie_has_no_time_base(self, tmp_path):
        (tmp_path / "bare_movie.mrc").write_bytes(b"")
        tree = _FakeTree()
        result = attach_mod.discover_and_attach(tree, str(tmp_path / "bare_movie.mrc"))
        assert not result
        assert "timestamps" in result.reason

    def test_stores_the_full_table_on_the_tree(self, tmp_path):
        """Only two channels become chips; everything else stays reachable."""
        mrc = paired(tmp_path)
        tree = _FakeTree(source_path=mrc)
        attach_mod.discover_and_attach(tree, mrc)
        assert set(tree.insitu_channels) >= {"time/s", "Ewe/V", "<I>/mA",
                                             "cycle number", "I Range"}
        assert tree.insitu_run is not None
        assert tree.insitu_alignment.covered_fraction == pytest.approx(1.0, abs=0.02)

    def test_a_frame_count_mismatch_refuses_to_map(self, tmp_path):
        mrc = paired(tmp_path)
        tree = _FakeTree(n=N_FRAMES - 5, source_path=mrc)   # movie shorter than CSV
        result = attach_mod.discover_and_attach(tree, mrc)
        assert not result
        assert "cannot map" in result.reason


class TestManualAttach:
    def test_attaches_an_explicitly_chosen_file(self, tmp_path):
        mrc = write_movie(tmp_path, n=N_FRAMES, period=PERIOD)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        duration = (N_FRAMES - 1) * PERIOD
        n_ec = int(np.ceil(duration / 0.08)) + 1
        ec = elsewhere / "r_02_CV_C01.mpr"
        ec.write_bytes(build_mpr(ids=CV_IDS, records=cv_records(n=n_ec)))
        tree = _FakeTree(source_path=mrc)

        result = attach_mod.attach_ec_file(tree, str(ec), movie_path=mrc)

        assert result, result.reason
        assert attach_mod.POTENTIAL_LANE in tree.navigator_signals

    def test_reads_the_ascii_export_too(self, tmp_path):
        mrc = write_movie(tmp_path, n=N_FRAMES, period=PERIOD)
        duration = (N_FRAMES - 1) * PERIOD
        n_ec = int(np.ceil(duration / 0.08)) + 1
        ec = write_mpt(tmp_path, "export.txt", n=n_ec)
        tree = _FakeTree(source_path=mrc)
        assert attach_mod.attach_ec_file(tree, ec, movie_path=mrc)

    def test_manual_override_of_the_lag(self, tmp_path):
        mrc = paired(tmp_path, n_ec=12)     # too short to auto-align
        tree = _FakeTree(source_path=mrc)
        ec = str(tmp_path / "r_02_CV_C01.mpr")

        assert not attach_mod.attach_ec_file(tree, ec, movie_path=mrc)
        result = attach_mod.attach_ec_file(
            tree, ec, movie_path=mrc, method="manual", lag_s=31.56
        )
        assert result, result.reason
        assert result.alignment.lag_s == pytest.approx(31.56)

    def test_reattaching_the_same_run_succeeds(self, tmp_path):
        """Auto-discovery runs on open, so picking the same record by hand
        afterwards is the NORMAL second call — it must read as success, not as
        'this run has no potential or current channel'."""
        mrc = paired(tmp_path)
        tree = _FakeTree(source_path=mrc)
        ec = str(tmp_path / "r_02_CV_C01.mpr")

        first = attach_mod.discover_and_attach(tree, mrc)
        second = attach_mod.attach_ec_file(tree, ec, movie_path=mrc)

        assert first and second, second.reason
        assert set(second.lanes) == set(first.lanes)
        # ONE plot state per lane, not one per attach.
        names = [n for n, _ in tree.registered]
        assert len(names) == len(set(names))

    def test_reattach_replaces_the_lane_values(self, tmp_path):
        """A different record under the same lane name must REPLACE the trace,
        not leave the previous run's on screen."""
        mrc = paired(tmp_path)
        tree = _FakeTree(source_path=mrc)
        attach_mod.discover_and_attach(tree, mrc)
        before = tree.lane_data(attach_mod.POTENTIAL_LANE).copy()

        duration = (N_FRAMES - 1) * PERIOD
        n_ec = int(np.ceil(duration / 0.08)) + 1
        other = tmp_path / "other_02_CV_C01.mpr"
        records = cv_records(n=n_ec)
        records["Ewe/V"] = records["Ewe/V"] * -1.0 - 0.25    # a distinguishable sweep
        other.write_bytes(build_mpr(ids=CV_IDS, records=records))

        assert attach_mod.attach_ec_file(tree, str(other), movie_path=mrc)
        after = tree.lane_data(attach_mod.POTENTIAL_LANE)
        assert not np.allclose(before, after), "the lane still shows the old run"
        assert after.shape == before.shape

    def test_a_run_with_no_e_or_i_says_what_it_does_have(self, tmp_path):
        mrc = write_movie(tmp_path, n=N_FRAMES, period=PERIOD)
        clock = de_movie.read_movie_clock(mrc)
        tree = _FakeTree(source_path=mrc)
        run = type("R", (), {})()
        # An OCV record carries <Ewe>/V, so strip to something that carries
        # neither a potential nor a current.
        from spyde.insitu.eclab import EcRun
        run = EcRun(path="x_01_OCV_C01.mpr", technique="Open Circuit Voltage",
                    start=None, time_s=np.linspace(0, clock.duration, 40),
                    channels={"cycle number": np.zeros(40)})
        from spyde.insitu.align import align_clocks
        result = attach_mod.attach_run(tree, clock, run, align_clocks(clock, run))
        assert not result
        assert "cycle number" in result.reason

    def test_unreadable_file_reports_rather_than_raises(self, tmp_path):
        mrc = write_movie(tmp_path, n=N_FRAMES, period=PERIOD)
        junk = tmp_path / "junk.mpr"
        junk.write_bytes(b"not an instrument record")
        tree = _FakeTree(source_path=mrc)
        result = attach_mod.attach_ec_file(tree, str(junk), movie_path=mrc)
        assert not result
        assert "could not read" in result.reason


class TestSessionEntryPoint:
    def test_mps_is_reported_as_a_recipe_not_data(self, window, tmp_path):
        """Picking the .mps is an easy mistake — it holds no samples. Say what
        it planned instead of failing blankly."""
        session = window["window"]
        mps = tmp_path / "seq.mps"
        mps.write_text(
            "EC-LAB SETTING FILE\n\nNumber of linked techniques : 2\n\n"
            "Technique : 1\nOpen Circuit Voltage\ntR (h:m:s) 0:00:30\n\n"
            "Technique : 2\nCyclic Voltammetry\nEi (V) 0,000\n",
            encoding="latin1",
        )
        session.load_insitu_data(str(mps))
        text = " ".join(str(m) for m in window["messages"])
        assert "settings file" in text
        assert "Cyclic Voltammetry" in text

    def test_missing_file_errors(self, window, tmp_path):
        session = window["window"]
        session.load_insitu_data(str(tmp_path / "nope.mpr"))
        text = " ".join(str(m) for m in window["messages"])
        assert "not found" in text.lower()

    def test_no_movie_open_is_reported(self, window, tmp_path):
        session = window["window"]
        (tmp_path / "r_02_CV_C01.mpr").write_bytes(
            build_mpr(ids=CV_IDS, records=cv_records(n=50))
        )
        session.load_insitu_data(str(tmp_path / "r_02_CV_C01.mpr"))
        text = " ".join(str(m) for m in window["messages"])
        assert "in-situ movie" in text.lower()

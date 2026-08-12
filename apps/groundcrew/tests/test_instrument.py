"""
test_instrument.py — the DE connection and the tile backend, against the REAL
simulated server.

These are integration tests on purpose. The whole viewer rests on `get_result`
behaving the way the tile backend assumes, and a mocked client would only assert
that the mock matches the assumption. Each test spawns deapi's `FakeServer` and
speaks the real protobuf protocol to it.

Two limits of the simulator, both established by inspecting its source rather
than guessed:

* it accepts exactly ONE connection, which is why `Instrument` has one; and
* it parses `center_x` / `center_y` off the wire and never uses them, so PAN is
  not asserted here. Zoom, output size and statistics are.
"""
from __future__ import annotations

import numpy as np
import pytest

from de_groundcrew.instrument import Instrument
from de_groundcrew.tile import DeapiTileBackend

pytestmark = pytest.mark.timeout(180)


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def instrument():
    """A connected Instrument talking to its own FakeServer.

    Module-scoped: spawning the server costs a second or two, and the tests
    below do not mutate state in ways that leak between them.
    """
    inst = Instrument(port=_free_port())
    info = inst.connect(fake=True).result(timeout=120)
    inst.info = info                      # type: ignore[attr-defined]
    yield inst
    inst.close()


@pytest.fixture(scope="module")
def acquired(instrument):
    """One acquisition, so `get_result` has something to return."""
    import time

    def _go(c):
        c["Frames Per Second"] = 20
        c["Exposure Time (seconds)"] = 0.2
        c.start_acquisition(1)
        while c.acquiring:
            time.sleep(0.02)
        return True

    instrument.call(_go).result(timeout=120)
    return instrument


class TestConnection:
    def test_connects_and_reports_the_camera(self, instrument):
        info = instrument.info
        assert info["cameras"], "no cameras reported"
        assert info["width"] > 0 and info["height"] > 0
        assert info["fake"] is True

    def test_reads_a_property(self, instrument):
        v = instrument.get("Frames Per Second").result(timeout=30)
        assert float(v) > 0

    def test_writes_then_reads_back(self, instrument):
        instrument.set("Frames Per Second", 25).result(timeout=30)
        assert float(instrument.get("Frames Per Second").result(timeout=30)) == 25

    def test_batch_read_is_one_round_trip(self, instrument):
        props = instrument.properties([
            "Frames Per Second", "Exposure Time (seconds)", "Server Software Version",
        ]).result(timeout=30)
        assert set(props) == {
            "Frames Per Second", "Exposure Time (seconds)", "Server Software Version"}
        assert all(v is not None for v in props.values())

    def test_an_unsupported_property_yields_None_rather_than_exploding(self, instrument):
        # A status board reads a dozen properties; one unsupported on this
        # camera must not take the whole poll down — nor be reported as a value.
        props = instrument.properties(
            ["Frames Per Second", "No Such Property At All"]).result(timeout=30)
        assert props["No Such Property At All"] is None
        assert props["Frames Per Second"] is not None

    def test_unsupported_is_distinguished_from_a_False_value(self, instrument):
        # `client[unknown]` returns False — the library's command-failed
        # sentinel — and False is also a legitimate value for a boolean
        # property. Deciding by VALUE would report every switch that happens to
        # be off as "missing", so the check is against the camera's own
        # property list.
        raw = instrument.call(lambda c: c["No Such Property At All"]).result(timeout=20)
        assert raw is False, "deapi changed its unknown-property sentinel"
        assert instrument.properties(
            ["No Such Property At All"]).result(timeout=20)["No Such Property At All"] is None

    def test_the_simulator_lacks_most_status_properties(self, instrument):
        # Pins a constraint on what can be BUILT against the fake server. The
        # status board and the top bar were designed against a dump of a REAL
        # server's 339 properties; the simulator exposes 73, and none of the
        # temperature, valve, vacuum or TEM-channel ones. They read None here,
        # which is correct behaviour and also means those surfaces cannot be
        # exercised without hardware or a richer simulator.
        names = set(instrument.property_names().result(timeout=30))
        assert "Frames Per Second" in names
        for absent in ("System Status", "Temperature - Detector (Celsius)",
                       "Camera Position Status", "Instrument Project Magnification"):
            assert absent not in names, (
                f"{absent!r} is now in the simulator — the status board can be "
                "developed against it; update this test and the scoping doc")

    def test_calling_from_the_io_thread_does_not_deadlock(self, instrument):
        # A nested DE call — a tile fetched from inside another DE call — would
        # deadlock against the single worker if `call` always submitted.
        def outer(c):
            assert instrument.on_io_thread
            return instrument.call(lambda c2: c2["Frames Per Second"]).result(timeout=5)

        assert float(instrument.call(outer).result(timeout=30)) > 0

    def test_close_is_idempotent(self):
        inst = Instrument(port=_free_port())
        inst.close()
        inst.close()

    def test_calls_after_close_fail_rather_than_hang(self):
        inst = Instrument(port=_free_port())
        inst.close()
        with pytest.raises(RuntimeError, match="closed"):
            inst.get("System Status").result(timeout=10)

    def test_second_connection_is_refused_explicitly(self, instrument):
        # Rather than half-work against a simulator that cannot serve it.
        with pytest.raises(NotImplementedError):
            instrument.open_reader()


class TestTileBackend:
    def test_protocol_surface(self, acquired):
        b = DeapiTileBackend(acquired, (acquired.info["height"], acquired.info["width"]))
        assert b.full_shape == (acquired.info["height"], acquired.info["width"])
        assert b.dtype == np.uint16
        assert b.origin == "upper"
        assert b.extent() is None

    def test_sample_returns_exactly_the_requested_size(self, acquired):
        h, w = acquired.info["height"], acquired.info["width"]
        b = DeapiTileBackend(acquired, (h, w))
        out = b.sample(0, w, 0, h, 256, 256)
        assert out.shape == (256, 256)
        assert out.dtype == np.uint16

    def test_a_non_square_tile_comes_back_non_square(self, acquired):
        h, w = acquired.info["height"], acquired.info["width"]
        b = DeapiTileBackend(acquired, (h, w))
        assert b.sample(0, w, 0, h, 320, 180).shape == (180, 320)

    def test_a_degenerate_region_does_not_divide_by_zero(self, acquired):
        h, w = acquired.info["height"], acquired.info["width"]
        b = DeapiTileBackend(acquired, (h, w))
        # x1 == x0 is reachable from a zero-width viewport mid-layout.
        assert b.sample(10, 10, 10, 10, 32, 32).shape == (32, 32)


class TestRequestMapping:
    """The region → `Attributes` arithmetic, asserted directly.

    Deliberately NOT tested through the simulator's pixels. Its synthetic scene
    is close to flat, so "zoom changed the picture" can fail for reasons that
    have nothing to do with the mapping — and its `center_x` is parsed and then
    ignored, so panning could never be asserted that way at all. Recording what
    was actually asked for tests the contract; the integration tests above
    already prove the call works.
    """

    class Recorder:
        """Stands in for the Instrument, capturing the Attributes it is given."""
        def __init__(self):
            self.seen = []

        @property
        def on_io_thread(self):
            return True

        def call(self, fn):
            from concurrent.futures import Future
            fut = Future()
            fut.set_result(fn(self))
            return fut

        # Client surface used by DeapiTileBackend._fetch
        def get_result(self, frame_type, pixel_format, attrs, hist):
            self.seen.append({
                "centerX": attrs.centerX, "centerY": attrs.centerY,
                "windowWidth": attrs.windowWidth, "windowHeight": attrs.windowHeight,
                "zoom": attrs.zoom, "frame_type": frame_type,
                "pixel_format": pixel_format, "bins": hist.bins,
            })
            attrs.imageMin, attrs.imageMax = 0.0, 100.0
            attrs.imageMean, attrs.imageStd = 50.0, 5.0
            return np.zeros((attrs.windowHeight, attrs.windowWidth), dtype=np.uint16)

    def _ask(self, x0, x1, y0, y1, out_w, out_h):
        rec = self.Recorder()
        DeapiTileBackend(rec, (1024, 1024)).sample(x0, x1, y0, y1, out_w, out_h)
        return rec.seen[-1]

    def test_window_is_the_OUTPUT_size(self):
        r = self._ask(0, 1024, 0, 1024, 256, 128)
        assert (r["windowWidth"], r["windowHeight"]) == (256, 128)

    def test_zoom_is_output_pixels_per_source_pixel(self):
        # THE contract the viewer rests on. source_extent = windowWidth / zoom,
        # measured against the real server: a 256-px window at zoom 0.25 came
        # back as the whole 1024² frame (correlation 0.98 vs a numpy reference).
        assert self._ask(0, 1024, 0, 1024, 256, 256)["zoom"] == pytest.approx(0.25)
        assert self._ask(0, 512, 0, 512, 256, 256)["zoom"] == pytest.approx(0.5)
        assert self._ask(0, 256, 0, 256, 256, 256)["zoom"] == pytest.approx(1.0)
        # Zoomed IN past 1:1 — one source pixel across several output pixels.
        assert self._ask(0, 128, 0, 128, 256, 256)["zoom"] == pytest.approx(2.0)

    def test_centre_is_the_middle_of_the_requested_region(self):
        # UNVERIFIED against hardware: the simulator ignores center_x/center_y
        # entirely (parsed off the wire, never used — requesting 200 returns a
        # byte-identical image to 0). This pins what we SEND; whether the server
        # agrees about the origin still needs a real camera.
        r = self._ask(256, 768, 128, 640, 256, 256)
        assert (r["centerX"], r["centerY"]) == (512, 384)

    def test_a_full_frame_request_centres_on_the_sensor(self):
        r = self._ask(0, 1024, 0, 1024, 512, 512)
        assert (r["centerX"], r["centerY"]) == (512, 512)

    def test_histogram_is_requested_with_the_pixels(self):
        assert self._ask(0, 1024, 0, 1024, 64, 64)["bins"] == 256

    def test_statistics_describe_the_PIXELS_not_the_attributes(self, acquired):
        # The bug this pins: `attrs.imageMin/Max/Mean/Std` came back 0 / 32768 /
        # 100 / 5 from the simulator — its configuration, not a measurement —
        # while the pixels in the SAME call ran 1…25 with a mean of 1.98.
        # Reporting the attributes would have printed "MAX 32768" under a
        # picture whose brightest pixel was 25. The histogram agreed with the
        # array, so the statistics are derived from it.
        h, w = acquired.info["height"], acquired.info["width"]
        b = DeapiTileBackend(acquired, (h, w))
        arr = b.sample(0, w, 0, h, 512, 512)
        s = b.last_stats
        assert s["max"] <= float(arr.max()) * 4 + 1, (
            f"reported max {s['max']} is unrelated to the pixels (max {arr.max()})")
        assert s["min"] <= float(arr.min()) + 1
        # Mean from binned counts is approximate; an order of magnitude is the
        # claim, and it is the claim that failed before.
        assert abs(s["mean"] - float(arr.mean())) < max(1.0, 0.5 * float(arr.mean()))

    def test_display_levels_are_inside_the_data_range(self, acquired):
        # Levels drive the clim; outside the data they render the frame a flat
        # white or black panel, which is exactly how this was found.
        h, w = acquired.info["height"], acquired.info["width"]
        b = DeapiTileBackend(acquired, (h, w))
        b.sample(0, w, 0, h, 256, 256)
        lo, hi = b.last_stats["levels"]
        assert hi > lo
        assert b.last_stats["min"] <= lo and hi <= b.last_stats["max"] + 1

    def test_statistics_ride_along_with_the_pixels(self, acquired):
        # The point of using get_result as the render backend: the stats strip
        # is free, not a second round trip.
        h, w = acquired.info["height"], acquired.info["width"]
        b = DeapiTileBackend(acquired, (h, w))
        b.sample(0, w, 0, h, 128, 128)
        s = b.last_stats
        assert s["max"] is not None and s["mean"] is not None
        assert s["max"] >= s["min"]


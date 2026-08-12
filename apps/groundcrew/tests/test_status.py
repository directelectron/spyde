"""
test_status.py — the status board's judging rules.

Pure functions over property dictionaries, so no server is involved. What these
pin is mostly NEGATIVE: the board must not show green for something it cannot
see, cannot judge, or does not recognise. Each of those is a different state
with a different fix, and collapsing them is the failure mode this file exists
to prevent.
"""
from __future__ import annotations

from de_groundcrew import status
from de_groundcrew.status import BAD, NO_CRITERIA, OK, UNREPORTED, WARN, build_report


def _all(value=None) -> dict:
    """Every property the board reads, set to *value*."""
    return {p: value for p in status.STATUS_PROPS}


def _card(report: dict, key: str) -> dict:
    return next(c for c in report["cards"] if c["key"] == key)


class TestNotKnowing:
    def test_a_server_that_reports_nothing_is_not_green(self):
        # The headline failure. With every property absent the board is all
        # grey — and an "overall: ok" beside it would read as a healthy camera.
        report = build_report(_all(None))
        assert report["summary"]["overall"] == UNREPORTED
        assert report["summary"]["reporting"] == 0
        assert all(c["state"] == UNREPORTED for c in report["cards"])

    def test_coverage_is_reported_so_a_thin_board_cannot_pose_as_a_full_one(self):
        values = _all(None)
        values["Server Software Version"] = "2.8.0"
        s = build_report(values)["summary"]
        assert s["reporting"] == 1
        assert s["total"] == len(status.CHECKS) > 1

    def test_an_unreported_property_names_itself(self):
        # So the fix is obvious: a missing property is a server gap, not a fault.
        card = _card(build_report(_all(None)), "vacuum")
        assert card["missing"] == ["Vacuum State"]
        assert "not reported" in card["detail"]

    def test_unjudgeable_readings_are_shown_but_not_passed(self):
        # Sensor-health thresholds are deferred, so the voltages are readings
        # with no verdict. They must never count as a passing check.
        values = _all(None)
        values["System Voltage - Main Sensor 1.8 V (mV)"] = 1802
        values["System Voltage - Main Sensor 3.3 VA (mV)"] = 3301
        values["System Voltage - Main Sensor 3.3 VD (mV)"] = 3298
        card = _card(build_report(values), "sensor_voltages")
        assert card["state"] == NO_CRITERIA
        assert "1802" in card["detail"]
        assert NO_CRITERIA in status.NOT_PASSING

    def test_unreported_and_no_criteria_are_distinct(self):
        # Both are grey; the fixes differ (a richer server vs a decision).
        assert UNREPORTED != NO_CRITERIA

    def test_a_rule_that_explodes_does_not_take_the_board_down(self):
        values = _all(None)
        values["Instrument Project Magnification"] = object()   # unformattable
        report = build_report(values)
        assert len(report["cards"]) == len(status.CHECKS)


class TestConnectionCheck:
    """The one check that must work when nothing else does.

    Its verdict is passed IN rather than derived from a property, because a
    camera that has stopped answering cannot be asked whether it is answering.
    deapi's FakeServer makes this concrete: `stop_acquisition()` never returns
    and takes the connection with it, so every property read after it hangs.
    """

    def test_an_unresponsive_camera_is_stated_not_left_blank(self):
        report = build_report({}, link={"state": BAD, "detail": "no response within 6s"})
        card = next(c for c in report["cards"] if c["key"] == "connection")
        assert card["state"] == BAD and "no response" in card["detail"]
        assert report["summary"]["overall"] == BAD

    def test_the_connection_card_comes_first(self):
        # It explains every other card being dark, so it must be read first.
        report = build_report({}, link={"state": BAD, "detail": "x"})
        assert report["cards"][0]["key"] == "connection"

    def test_a_responding_camera_does_not_mask_a_real_fault(self):
        values = _all(None)
        values["Instrument Project Magnification"] = -1
        report = build_report(values, link={"state": OK, "detail": "responding"})
        assert report["summary"]["overall"] == BAD

    def test_omitting_the_link_leaves_the_board_unchanged(self):
        # Callers that never race a deadline should not grow a phantom card.
        assert all(c["key"] != "connection" for c in build_report(_all(None))["cards"])


class TestTemChannel:
    def test_the_unset_sentinel_blames_the_channel_not_the_calibration(self):
        # The owner's correction. -1 means the DE↔TEM channel is not reporting;
        # "calibration required" would send an engineer to the wrong place.
        values = _all(None)
        values["Instrument Project Magnification"] = -1
        card = _card(build_report(values), "magnification")
        assert card["state"] == BAD
        assert "TEM" in card["detail"] and "calibrat" not in card["detail"].lower()

    def test_a_real_magnification_passes_and_shows_the_pixel_size(self):
        values = _all(None)
        values["Instrument Project Magnification"] = 20000
        values["Specimen Pixel Size X (nanometers)"] = 0.0521
        card = _card(build_report(values), "magnification")
        assert card["state"] == OK
        assert "20000" in card["detail"] and "0.0521" in card["detail"]

    def test_camera_length_uses_the_same_sentinel_rule(self):
        values = _all(None)
        values["Instrument Project Camera Length (centimeters)"] = -1.0
        assert _card(build_report(values), "camera_length")["state"] == BAD


class TestCorrections:
    def test_a_correction_left_off_warns_rather_than_fails(self):
        # Running without flatfield is a legitimate deliberate choice; the board
        # exists to stop it happening by ACCIDENT.
        values = _all(None)
        values["Image Processing - Flatfield Correction"] = False
        card = _card(build_report(values), "flatfield")
        assert card["state"] == WARN
        assert "NOT applied" in card["detail"]

    def test_a_correction_switched_on_passes(self):
        values = _all(None)
        values["Image Processing - Bad Pixel Correction"] = True
        assert _card(build_report(values), "bad_pixel")["state"] == OK

    def test_string_forms_of_on_and_off_are_both_understood(self):
        for raw, expected in (("On", OK), ("enabled", OK), ("off", WARN), ("0", WARN)):
            values = _all(None)
            values["Image Processing - Flatfield Correction"] = raw
            assert _card(build_report(values), "flatfield")["state"] == expected, raw

    def test_wording_nobody_anticipated_warns_instead_of_passing(self):
        values = _all(None)
        values["Image Processing - Flatfield Correction"] = "Partially"
        card = _card(build_report(values), "flatfield")
        assert card["state"] == WARN
        assert "Partially" in card["detail"]


class TestStatusStrings:
    def test_a_healthy_status_word_passes(self):
        values = _all(None)
        values["Temperature - Chilled Water Status"] = "OK"
        values["Temperature - Chilled Water (Celsius)"] = 18.3
        card = _card(build_report(values), "chilled_water")
        assert card["state"] == OK and "18.3" in card["detail"]

    def test_a_fault_word_fails(self):
        values = _all(None)
        values["Temperature - Chilled Water Status"] = "Over Temperature Alarm"
        assert _card(build_report(values), "chilled_water")["state"] == BAD

    def test_an_unrecognised_status_warns_and_says_so(self):
        # These are free text from the server. A word this board has never seen
        # is exactly where quietly showing green would be wrong.
        values = _all(None)
        values["Vacuum State"] = "Transitioning"
        card = _card(build_report(values), "vacuum")
        assert card["state"] == WARN and "unrecognised" in card["detail"]

    def test_a_retracted_camera_is_not_a_fault(self):
        # It is correct when the operator retracted it; nagging would train
        # people to ignore the board.
        values = _all(None)
        values["Camera Position Status"] = "Retracted"
        card = _card(build_report(values), "camera_position")
        assert card["state"] == OK and card["detail"] == "Retracted"

    def test_a_camera_mid_travel_warns(self):
        values = _all(None)
        values["Camera Position Status"] = "Inserting"
        assert _card(build_report(values), "camera_position")["state"] == WARN


class TestSummary:
    def test_one_failure_makes_the_whole_board_fail(self):
        values = _all(None)
        values["Server Software Version"] = "2.8.0"
        values["Instrument Project Magnification"] = -1
        assert build_report(values)["summary"]["overall"] == BAD

    def test_a_warning_shows_when_nothing_has_failed(self):
        values = _all(None)
        values["Image Processing - Flatfield Correction"] = False
        assert build_report(values)["summary"]["overall"] == WARN

    def test_all_reporting_checks_passing_is_green(self):
        values = _all(None)
        values["Server Software Version"] = "2.8.0"
        values["Firmware Version"] = "1.2.3"
        assert build_report(values)["summary"]["overall"] == OK


class TestPropertyList:
    def test_the_read_list_is_deduplicated(self):
        assert len(status.STATUS_PROPS) == len(set(status.STATUS_PROPS))

    def test_every_check_property_is_in_the_read_list(self):
        # Or the check would be permanently unreported for want of a read.
        for check in status.CHECKS:
            for p in check.props:
                assert p in status.STATUS_PROPS, f"{check.key} reads unlisted {p!r}"

    def test_optional_properties_belong_to_their_check(self):
        for check in status.CHECKS:
            assert set(check.optional) <= set(check.props), check.key

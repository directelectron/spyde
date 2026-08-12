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
from de_groundcrew.status import (
    BAD, NO_CRITERIA, OK, UNREPORTED, WARN, build_report)


def _all(value=None) -> dict:
    """Every property the board reads, set to *value*."""
    return {p: value for p in status.STATUS_PROPS}


def _card(report: dict, key: str) -> dict:
    return next(c for c in report["cards"] if c["key"] == key)


def _with(**pairs) -> dict:
    """An empty board with the named properties set. Keys are status module
    constants, so a renamed property breaks the test at the import, not with a
    mysterious `unreported`."""
    values = _all(None)
    values.update(pairs)
    return values


class TestNotKnowing:
    def test_a_server_that_reports_nothing_is_not_green(self):
        # The headline failure. With every property absent the board is all
        # grey — and an "overall: ok" beside it would read as a healthy camera.
        report = build_report(_all(None))
        assert report["summary"]["overall"] == UNREPORTED
        assert report["summary"]["reporting"] == 0
        assert all(c["state"] == UNREPORTED for c in report["cards"])

    def test_coverage_is_reported_so_a_thin_board_cannot_pose_as_a_full_one(self):
        s = build_report(_with(**{status.SERVER_VERSION: "2.8.0"}))["summary"]
        assert s["reporting"] == 1
        assert s["total"] == len(status.CARDS) > 1

    def test_an_unreported_card_names_the_properties_to_go_and_find(self):
        # "Amber" is useless if you cannot tell what to fix; a missing card must
        # say which property strings the server did not answer.
        card = _card(build_report(_all(None)), "vacuum")
        assert card["state"] == UNREPORTED
        assert status.VACUUM in card["missing"]
        # The ROWS name them; a fix line here would repeat the rows verbatim
        # and double the height of every dark card on the board.
        assert not card["fix"]
        assert any(r["label"] in status.VACUUM for r in card["rows"])

    def test_a_partly_reported_card_says_which_part_is_missing(self):
        card = _card(build_report(_with(**{status.VACUUM: "Ready"})), "vacuum")
        # Partial, not green and not grey: it HAS a live headline value, but an
        # essential reading is absent, so it must not pass.
        assert card["state"] == WARN
        assert card["big"] == "Ready"
        assert "Protection Cover Status" in card["fix"], (
            "the gap must be stated, not silently dropped")
        assert status.POSITION not in card["fix"], (
            "position is context, not the check — nagging about it trains people "
            "to ignore the line that matters")

    def test_unjudgeable_readings_are_shown_but_not_passed(self):
        # Sensor-health thresholds are deferred, so the voltages are readings
        # with no verdict. They must never count as a passing check.
        card = _card(build_report(_with(**{status.VOLTAGES[0]: 1802})), "sensor")
        assert card["state"] == NO_CRITERIA
        assert card["big"] == "Not assessed"
        assert any("1802" in r["value"] for r in card["rows"])
        assert NO_CRITERIA in status.NOT_PASSING

    def test_unreported_and_no_criteria_are_distinct(self):
        # Both are grey; the fixes differ (a richer server vs a decision).
        assert UNREPORTED != NO_CRITERIA

    def test_a_rule_that_explodes_does_not_take_the_board_down(self):
        report = build_report(_with(**{status.MAGNIFICATION: object()}))
        assert len(report["cards"]) == len(status.CARDS)


class TestTemChannel:
    def test_the_unset_sentinel_blames_the_channel_not_the_calibration(self):
        # The owner's correction. -1 means the DE↔TEM channel is not reporting;
        # "calibration required" would send an engineer to the wrong place.
        card = _card(build_report(_with(**{
            status.MAGNIFICATION: -1, status.CAMERA_LENGTH: -1.0,
            status.SPECIMEN_PX: -1.0})), "tem")
        assert card["state"] == BAD
        assert card["big"] == "No link"
        assert "not reporting to DE" in card["fix"]
        # The fix names the CONSEQUENCE (calibrated pixel sizes will be wrong)
        # but must not tell anyone to go and calibrate — that is the wrong job.
        assert "calibration required" not in card["fix"].lower()

    def test_a_real_magnification_passes_and_shows_the_pixel_size(self):
        card = _card(build_report(_with(**{
            status.MAGNIFICATION: 20000, status.CAMERA_LENGTH: 245,
            status.SPECIMEN_PX: 0.0521})), "tem")
        assert card["state"] == OK and card["big"] == "Reporting"
        values = " ".join(r["value"] for r in card["rows"])
        assert "20000" in values and "0.0521" in values and "245" in values

    def test_a_partial_sentinel_warns_rather_than_claiming_no_link(self):
        # One dead reading among live ones is a different situation from the
        # whole channel being down, and says so.
        card = _card(build_report(_with(**{
            status.MAGNIFICATION: 20000, status.CAMERA_LENGTH: -1})), "tem")
        assert card["state"] == WARN and card["big"] == "Partial"


class TestCooling:
    def test_absolute_zero_is_an_absent_sensor_not_cold_water(self):
        # The status string says OK while the value says -273.1: reporting them
        # separately is the point. Averaging them into one light is how this
        # gets shown as green.
        card = _card(build_report(_with(**{
            status.DETECTOR_T: -30.0, status.DETECTOR_STATUS: "OK",
            status.WATER_T: -273.1, status.WATER_STATUS: "OK"})), "cooling")
        assert card["state"] == WARN
        assert "absolute zero" in card["fix"]
        assert "sensor is absent" in card["fix"]

    def test_a_healthy_loop_passes_and_leads_with_the_temperature(self):
        card = _card(build_report(_with(**{
            status.DETECTOR_T: -30.0, status.DETECTOR_STATUS: "Cooled",
            status.WATER_T: 18.2, status.WATER_STATUS: "OK"})), "cooling")
        assert card["state"] == OK
        assert card["big"] == "-30.0 °C"
        assert card["big_tone"] == "cryo", "temperature is not a semantic status colour"
        assert not card["fix"], "a healthy card must not carry a fix line"

    def test_a_fault_word_fails(self):
        card = _card(build_report(_with(**{
            status.WATER_STATUS: "Over Temperature Alarm"})), "cooling")
        assert card["state"] == BAD


class TestCorrections:
    def test_a_correction_left_off_warns_rather_than_fails(self):
        # Running without flatfield is a legitimate deliberate choice; the board
        # exists to stop it happening by ACCIDENT.
        card = _card(build_report(_with(**{
            status.FLATFIELD: False, status.BAD_PIXEL: True})), "corrections")
        assert card["state"] == WARN
        assert "Flatfield" in card["fix"] and "accident" in card["fix"]

    def test_both_applied_passes_with_no_fix_line(self):
        card = _card(build_report(_with(**{
            status.FLATFIELD: True, status.BAD_PIXEL: True})), "corrections")
        assert card["state"] == OK and card["big"] == "Applied" and not card["fix"]

    def test_string_forms_of_on_and_off_are_both_understood(self):
        for raw, expected in (("On", OK), ("enabled", OK), ("off", WARN), ("0", WARN)):
            card = _card(build_report(_with(**{
                status.FLATFIELD: raw, status.BAD_PIXEL: raw})), "corrections")
            assert card["state"] == expected, raw

    def test_wording_nobody_anticipated_warns_instead_of_passing(self):
        card = _card(build_report(_with(**{
            status.FLATFIELD: "Partially", status.BAD_PIXEL: True})), "corrections")
        assert card["state"] == WARN


class TestVacuumAndPosition:
    def test_an_unrecognised_status_warns_and_does_not_pass(self):
        # These are free text from the server. A word this board has never seen
        # is exactly where quietly showing green would be wrong.
        card = _card(build_report(_with(**{status.VACUUM: "Transitioning"})), "vacuum")
        assert card["state"] == WARN

    def test_a_retracted_camera_is_not_a_fault(self):
        # It is correct when the operator retracted it; nagging would train
        # people to ignore the board.
        card = _card(build_report(_with(**{
            status.VACUUM: "Ready", status.COVER: "Open",
            status.POSITION: "Retracted"})), "vacuum")
        assert card["state"] == OK
        assert next(r for r in card["rows"] if r["label"] == "Position")["state"] == OK

    def test_a_camera_mid_travel_warns_on_its_row(self):
        card = _card(build_report(_with(**{
            status.VACUUM: "Ready", status.COVER: "Open",
            status.POSITION: "Inserting"})), "vacuum")
        assert next(r for r in card["rows"] if r["label"] == "Position")["state"] == WARN


class TestConnectionCheck:
    """The one check that must work when nothing else does.

    Its verdict is passed IN rather than derived from a property, because a
    camera that has stopped answering cannot be asked whether it is answering.
    """

    def test_an_unresponsive_camera_is_stated_not_left_blank(self):
        report = build_report({}, link={"state": BAD, "detail": "no response within 6s"})
        card = _card(report, "connection")
        assert card["state"] == BAD and card["big"] == "No response"
        assert "no response" in card["fix"]
        assert report["summary"]["overall"] == BAD

    def test_the_connection_card_comes_first(self):
        # It explains every other card being dark, so it must be read first.
        report = build_report({}, link={"state": BAD, "detail": "x"})
        assert report["cards"][0]["key"] == "connection"

    def test_a_responding_camera_does_not_mask_a_real_fault(self):
        report = build_report(_with(**{status.MAGNIFICATION: -1}),
                              link={"state": OK, "detail": "responding"})
        assert report["summary"]["overall"] == BAD

    def test_omitting_the_link_leaves_the_board_unchanged(self):
        assert all(c["key"] != "connection" for c in build_report(_all(None))["cards"])


class TestCardShape:
    def test_every_card_property_is_in_the_read_list(self):
        # Or the card would be permanently unreported for want of a read.
        for card in status.CARDS:
            for p in card.props:
                assert p in status.STATUS_PROPS, f"{card.key} reads unlisted {p!r}"

    def test_the_read_list_is_deduplicated(self):
        assert len(status.STATUS_PROPS) == len(set(status.STATUS_PROPS))

    def test_every_card_names_a_source_for_the_engineer(self):
        for card in status.CARDS:
            assert card.source, f"{card.key} has no source property family"

    def test_cards_always_carry_the_full_shape_the_ui_indexes(self):
        # The renderer reads these unconditionally; a missing key is a blank
        # pane rather than a caught error.
        for report in (build_report(_all(None)),
                       build_report(_with(**{status.VACUUM: "Ready"}))):
            for c in report["cards"]:
                assert set(c) >= {"key", "title", "source", "state", "big",
                                  "big_tone", "rows", "chips", "fix", "missing"}
                assert isinstance(c["rows"], list)
                for r in c["rows"]:
                    assert set(r) == {"label", "value", "state"}


class TestSummary:
    def test_one_failure_makes_the_whole_board_fail(self):
        assert build_report(_with(**{
            status.SERVER_VERSION: "2.8.0",
            status.MAGNIFICATION: -1, status.CAMERA_LENGTH: -1,
            status.SPECIMEN_PX: -1}))["summary"]["overall"] == BAD

    def test_a_warning_shows_when_nothing_has_failed(self):
        assert build_report(_with(**{
            status.FLATFIELD: False, status.BAD_PIXEL: True,
        }))["summary"]["overall"] == WARN

    def test_all_reporting_cards_passing_is_green(self):
        assert build_report(_with(**{
            status.SERVER_VERSION: "2.8.0", status.FIRMWARE: "1.4.2",
        }))["summary"]["overall"] == OK

"""
status.py — the camera status board, and what to do when the camera won't say.

Every check declares the properties it needs and how to read them. Reading is
one batched trip over the single connection; judging is pure functions on the
values, which is what makes this testable without a server.

## Not knowing is a state

The whole design turns on refusing to fake a green light. A card is:

``ok`` / ``warn`` / ``bad``
    The camera reported, and the value was judged against a rule.
``unreported``
    The server does not expose the property at all. Measured, not hypothetical:
    the simulator has 73 properties where a real server has 339, and 15 of the
    22 this board wants are missing from it.
``no_criteria``
    The camera reported a value and there is no agreed threshold for it yet.
    Sensor health is the case that matters — the owner deferred thresholds, so
    the voltages are shown as readings and explicitly not judged.

``unreported`` and ``no_criteria`` both mean "this is not a green light", and
they are kept apart because the fixes differ: one needs a richer server, the
other needs a decision. Neither is counted as passing. The summary reports
coverage — "9 of 14 checks reporting" — so a board that is green because it can
only see four things cannot be mistaken for a healthy camera.

## The TEM channel

`Instrument Project *` reading its unset sentinel (-1) does NOT mean the
microscope is uncalibrated — it means the DE↔TEM channel is not reporting. The
two have completely different fixes, so they are different messages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

#: A card that passes, warns, fails, was not reported, or has no rule yet.
OK, WARN, BAD, UNREPORTED, NO_CRITERIA = (
    "ok", "warn", "bad", "unreported", "no_criteria")

#: `Instrument Project *` values when the DE↔TEM channel is not reporting.
TEM_UNSET = (-1, -1.0)

Verdict = tuple[str, str]          # (state, detail)


@dataclass(frozen=True)
class Check:
    key: str
    label: str
    group: str
    props: tuple[str, ...]
    judge: Callable[[dict], Verdict]
    #: Properties that may be absent without making the card unreported — used
    #: where one reading is the check and the others are context.
    optional: tuple[str, ...] = field(default=())

    def run(self, values: dict) -> dict:
        missing = [p for p in self.props
                   if p not in self.optional and values.get(p) is None]
        if missing:
            state, detail = UNREPORTED, f"not reported by this server"
        else:
            try:
                state, detail = self.judge(values)
            except Exception as e:                  # a rule must never take the board down
                state, detail = UNREPORTED, f"could not be read ({e})"
        return {
            "key": self.key, "label": self.label, "group": self.group,
            "state": state, "detail": detail,
            "readings": {p: values.get(p) for p in self.props},
            "missing": missing,
        }


# ── Rules ─────────────────────────────────────────────────────────────────────

def _txt(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _status_word(v: Any, *, good: tuple[str, ...]) -> Verdict:
    """Judge a server status string by its wording.

    Unrecognised wording is reported as a warning rather than assumed fine:
    these strings are free text from the server, and a status this board has
    never seen is exactly the case where quietly showing green is wrong.
    """
    s = _txt(v)
    if not s:
        return UNREPORTED, "blank"
    low = s.lower()
    if any(g in low for g in good):
        return OK, s
    if any(w in low for w in ("error", "fault", "fail", "alarm", "over", "trip")):
        return BAD, s
    return WARN, f"{s} (unrecognised status)"


def _chilled_water(v: dict) -> Verdict:
    state, detail = _status_word(v["Temperature - Chilled Water Status"],
                                 good=("ok", "normal", "good", "ready"))
    temp = v.get("Temperature - Chilled Water (Celsius)")
    if temp is not None:
        detail = f"{detail} · {float(temp):.1f} °C"
    return state, detail


def _detector_temp(v: dict) -> Verdict:
    state, detail = _status_word(v["Temperature - Detector Status"],
                                 good=("ok", "normal", "stable", "ready", "cooled"))
    temp = v.get("Temperature - Detector (Celsius)")
    if temp is not None:
        detail = f"{detail} · {float(temp):.1f} °C"
    return state, detail


def _camera_position(v: dict) -> Verdict:
    # Position is not good or bad — a retracted camera is correct when the
    # operator retracted it. The card reports the state and flags only motion
    # or an error, so it does not nag about a deliberate choice.
    s = _txt(v["Camera Position Status"])
    low = s.lower()
    if any(w in low for w in ("moving", "inserting", "retracting", "unknown")):
        return WARN, s
    if any(w in low for w in ("error", "fault", "fail")):
        return BAD, s
    return OK, s


def _cover(v: dict) -> Verdict:
    return _status_word(v["Protection Cover Status"],
                        good=("open", "closed", "ok", "normal"))


def _vacuum(v: dict) -> Verdict:
    return _status_word(v["Vacuum State"], good=("ok", "normal", "ready", "vacuum", "pumped"))


def _on_off(prop: str, label: str) -> Callable[[dict], Verdict]:
    """A correction that is either applied or not.

    Off is a WARNING, never an error: running without flatfield is a legitimate
    thing to do deliberately. The board's job is to make it impossible to do by
    accident.
    """
    def judge(v: dict) -> Verdict:
        raw = v[prop]
        s = _txt(raw).lower()
        on = raw is True or s in ("on", "true", "1", "enabled", "yes")
        off = raw is False or s in ("off", "false", "0", "disabled", "no")
        if on:
            return OK, f"{label} applied"
        if off:
            return WARN, f"{label} NOT applied"
        return WARN, f"unrecognised setting: {_txt(raw)!r}"
    return judge


def _magnification(v: dict) -> Verdict:
    mag = v["Instrument Project Magnification"]
    try:
        as_num = float(mag)
    except (TypeError, ValueError):
        as_num = None
    if as_num is not None and as_num in TEM_UNSET:
        # The owner's correction: this is a dead channel, not a missing
        # calibration. Saying "calibration required" would send an engineer to
        # the wrong place entirely.
        return BAD, "DE↔TEM channel not reporting"
    px = v.get("Specimen Pixel Size X (nanometers)")
    detail = f"{mag}×"
    if px not in (None, *TEM_UNSET):
        detail += f" · {float(px):.4g} nm/px"
    return OK, detail


def _camera_length(v: dict) -> Verdict:
    cl = v["Instrument Project Camera Length (centimeters)"]
    try:
        as_num = float(cl)
    except (TypeError, ValueError):
        as_num = None
    if as_num is not None and as_num in TEM_UNSET:
        return BAD, "DE↔TEM channel not reporting"
    return OK, f"{cl} cm"


def _readings_only(label: str, props: tuple[str, ...]) -> Callable[[dict], Verdict]:
    """Report values without judging them.

    For sensor health: the owner deferred the thresholds, so there is no rule to
    apply. Showing the numbers is useful; colouring them green would be a claim
    nobody has made.

    Takes its property names explicitly because `judge` receives the WHOLE
    values dict, not the check's slice — iterating it would list every property
    the board read.
    """
    def judge(v: dict) -> Verdict:
        shown = [f"{p.rsplit(' - ', 1)[-1]} {v[p]}" for p in props if v.get(p) is not None]
        return NO_CRITERIA, (f"{label}: " + ", ".join(shown)) if shown else label
    return judge


def _software(v: dict) -> Verdict:
    parts = [f"server {v['Server Software Version']}"]
    fw = v.get("Firmware Version")
    if fw is not None:
        parts.append(f"firmware {fw}")
    return OK, " · ".join(parts)


# ── The board ─────────────────────────────────────────────────────────────────

_VOLTAGES = ("System Voltage - Main Sensor 1.8 V (mV)",
             "System Voltage - Main Sensor 3.3 VA (mV)",
             "System Voltage - Main Sensor 3.3 VD (mV)")
_SENSOR_TEMPS = ("System Temperature - ADC 1 (0.1 K)",
                 "System Temperature - Heat Exchanger (0.1 K)")

CHECKS: tuple[Check, ...] = (
    Check("chilled_water", "Chilled water", "Cooling",
          ("Temperature - Chilled Water Status", "Temperature - Chilled Water (Celsius)"),
          _chilled_water, optional=("Temperature - Chilled Water (Celsius)",)),
    Check("detector_temp", "Detector temperature", "Cooling",
          ("Temperature - Detector Status", "Temperature - Detector (Celsius)"),
          _detector_temp, optional=("Temperature - Detector (Celsius)",)),

    Check("camera_position", "Camera position", "Mechanics",
          ("Camera Position Status",), _camera_position),
    Check("cover", "Protection cover", "Mechanics",
          ("Protection Cover Status",), _cover),
    Check("vacuum", "Vacuum", "Mechanics", ("Vacuum State",), _vacuum),

    Check("bad_pixel", "Bad pixel correction", "Corrections",
          ("Image Processing - Bad Pixel Correction",),
          _on_off("Image Processing - Bad Pixel Correction", "Bad pixel correction")),
    Check("flatfield", "Flatfield correction", "Corrections",
          ("Image Processing - Flatfield Correction",),
          _on_off("Image Processing - Flatfield Correction", "Flatfield")),

    Check("magnification", "Magnification calibration", "Calibration",
          ("Instrument Project Magnification", "Specimen Pixel Size X (nanometers)"),
          _magnification, optional=("Specimen Pixel Size X (nanometers)",)),
    Check("camera_length", "Camera length", "Calibration",
          ("Instrument Project Camera Length (centimeters)",), _camera_length),

    Check("sensor_voltages", "Sensor voltages", "Sensor health",
          _VOLTAGES, _readings_only("rails", _VOLTAGES)),
    Check("sensor_temps", "Sensor temperatures", "Sensor health",
          _SENSOR_TEMPS, _readings_only("sensors", _SENSOR_TEMPS)),

    Check("software", "Software", "System",
          ("Server Software Version", "Firmware Version"),
          _software, optional=("Firmware Version",)),
)

#: Every property the board needs, de-duplicated — ONE batched read per refresh.
STATUS_PROPS: tuple[str, ...] = tuple(dict.fromkeys(
    p for c in CHECKS for p in c.props))

#: States that do not count as a passing check.
NOT_PASSING = (WARN, BAD, UNREPORTED, NO_CRITERIA)


def build_report(values: dict, *, link: dict | None = None) -> dict:
    """Judge every check and summarise. Pure — no I/O, no server.

    *link* describes the CONNECTION itself, which the caller judges rather than
    this module: it is the one check that must work when nothing else does, so
    it cannot depend on having read a property. Passing ``{"state": BAD, ...}``
    for an unresponsive camera turns an empty *values* from a silently blank
    board into an explicit "the camera stopped answering".
    """
    cards = [c.run(values) for c in CHECKS]
    if link is not None:
        cards.insert(0, {
            "key": "connection", "label": "Connection", "group": "System",
            "state": link.get("state", UNREPORTED),
            "detail": link.get("detail", ""), "readings": {}, "missing": [],
        })
    by_state: dict[str, int] = {}
    for c in cards:
        by_state[c["state"]] = by_state.get(c["state"], 0) + 1

    reporting = len(cards) - by_state.get(UNREPORTED, 0)
    if by_state.get(BAD):
        overall = BAD
    elif by_state.get(WARN):
        overall = WARN
    elif reporting == 0:
        # Nothing to be green ABOUT. Without this, a server that answers
        # nothing produces an all-grey board with a green headline.
        overall = UNREPORTED
    else:
        overall = OK

    return {
        "cards": cards,
        "summary": {
            "overall": overall,
            "counts": by_state,
            "reporting": reporting,
            "total": len(cards),
        },
    }

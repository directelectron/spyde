"""
status.py — the camera status board, and what to do when the camera won't say.

A go/no-go board you check before committing microscope time. Everything should
read green; the design's whole job is to make the one thing that is NOT read
instantly.

## One card per subsystem, not one per property

An engineer thinks in subsystems — cooling, vacuum, the TEM link — so a card is
a subsystem with several readings inside it, a headline value, and the property
family it came from. Splitting them one-per-property produced a wall of
identical tiles where nothing stood out, which defeats the point.

Every card carries:

``big``       the one value you read first, already judged
``rows``      the supporting readings, each with its OWN state
``source``    the property family, e.g. ``Temperature - *``. "Amber" is useless
              if you cannot tell what to go and fix.
``fix``       what to DO about it. Only present when something is wrong.

## Not knowing is a state

The board refuses to fake a green light. A card is:

``ok`` / ``warn`` / ``bad``
    The camera reported, and the value was judged against a rule.
``unreported``
    The server does not expose the properties at all. Measured, not
    hypothetical: the simulator has 73 properties where a real server has 339.
``no_criteria``
    It reported, and there is no agreed threshold yet. Sensor health is the
    case that matters — thresholds were deferred, so the voltages are shown as
    readings and explicitly not judged.

``unreported`` and ``no_criteria`` are kept apart because the fixes differ: one
needs a richer server, the other needs a decision. Neither counts as passing,
and the banner reports coverage, so a board that is green because it can only
see two things cannot be mistaken for a healthy camera.

## The TEM channel

`Instrument Project *` reading its unset sentinel (-1) does NOT mean the
microscope is uncalibrated — it means the DE↔TEM channel is not reporting.
Different fault, different fix, so it gets its own card and its own wording.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

OK, WARN, BAD, UNREPORTED, NO_CRITERIA = (
    "ok", "warn", "bad", "unreported", "no_criteria")

#: `Instrument Project *` values when the DE↔TEM channel is not reporting.
TEM_UNSET = (-1, -1.0)

#: States that do not count as a passing check.
NOT_PASSING = (WARN, BAD, UNREPORTED, NO_CRITERIA)

#: Worst-first, for reducing several row states into a card state.
#:
#: UNREPORTED ranks BELOW ok on purpose. A card with a live headline value and
#: one missing supporting row is PARTIAL, not unreported — letting the missing
#: row dominate turned "Vacuum: Ready" grey, which reads as no information at
#: all. The gap is carried by the fix line and by the ok -> warn bump in
#: `Card.run` instead. (An all-missing card never reaches here.)
_SEVERITY = {BAD: 4, WARN: 3, NO_CRITERIA: 2, OK: 1, UNREPORTED: 0}


# ── Property names ────────────────────────────────────────────────────────────

DETECTOR_T = "Temperature - Detector (Celsius)"
DETECTOR_STATUS = "Temperature - Detector Status"
WATER_T = "Temperature - Chilled Water (Celsius)"
WATER_STATUS = "Temperature - Chilled Water Status"
TEMP_CONTROL = "Temperature - Control"
VACUUM = "Vacuum State"
COVER = "Protection Cover Status"
POSITION = "Camera Position Status"
BAD_PIXEL = "Image Processing - Bad Pixel Correction"
FLATFIELD = "Image Processing - Flatfield Correction"
MAGNIFICATION = "Instrument Project Magnification"
CAMERA_LENGTH = "Instrument Project Camera Length (centimeters)"
SPECIMEN_PX = "Specimen Pixel Size X (nanometers)"
SERVER_VERSION = "Server Software Version"
FIRMWARE = "Firmware Version"

VOLTAGES = ("System Voltage - Main Sensor 1.8 V (mV)",
            "System Voltage - Main Sensor 3.3 VA (mV)",
            "System Voltage - Main Sensor 3.3 VD (mV)")
SENSOR_TEMPS = ("System Temperature - ADC 1 (0.1 K)",
                "System Temperature - Heat Exchanger (0.1 K)")


# ── Small helpers ─────────────────────────────────────────────────────────────

def _txt(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row(label: str, value: Any, state: str = OK, *, unit: str = "") -> dict:
    """One reading. A missing value renders as an em dash, never as zero."""
    if value is None:
        return {"label": label, "value": "—", "state": UNREPORTED}
    text = f"{value}{(' ' + unit) if unit else ''}"
    return {"label": label, "value": text, "state": state}


def _short(prop: str) -> str:
    """A property name with its family prefix dropped.

    The card header already carries the family (``Temperature - *``), so the
    rows repeating it wastes the width that the value needs. ``Instrument
    Project Magnification`` has no separator, so it is left whole.
    """
    return prop.rsplit(" - ", 1)[-1]


def _worst(*states: str) -> str:
    return max(states, key=lambda s: _SEVERITY.get(s, 0)) if states else UNREPORTED


def _status_word(v: Any, *, good: tuple[str, ...]) -> str:
    """Judge a server status string by its wording.

    Wording this board has never seen WARNS rather than passing: these are free
    text from the server, and an unknown status is exactly the case where
    quietly showing green is wrong.
    """
    s = _txt(v).lower()
    if not s:
        return UNREPORTED
    if any(w in s for w in ("error", "fault", "fail", "alarm", "trip", "over")):
        return BAD
    return OK if any(g in s for g in good) else WARN


def _switch(v: Any) -> bool | None:
    """True/False for an on-off property, None if the wording is unrecognised."""
    if v is True or v is False:
        return v
    s = _txt(v).lower()
    if s in ("on", "true", "1", "enabled", "yes", "applied"):
        return True
    if s in ("off", "false", "0", "disabled", "no"):
        return False
    return None


# ── Cards ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Card:
    key: str
    title: str
    source: str
    props: tuple[str, ...]
    build: Callable[[dict], dict]
    #: Properties that are CONTEXT, not the check. Still read and still shown,
    #: but their absence does not make the card announce a gap — a card that
    #: says "partly reported" because one supporting reading is missing trains
    #: people to ignore the line that matters.
    optional: tuple[str, ...] = ()

    def run(self, values: dict) -> dict:
        missing = [p for p in self.props if values.get(p) is None]
        essential_missing = [p for p in missing if p not in self.optional]

        if len(missing) == len(self.props):
            # Nothing at all to show. The ROWS name the properties — that is
            # what tells someone whether the server is old or a check is
            # pointed at the wrong name — so there is deliberately no `fix`
            # line here: it would repeat the rows verbatim and double the
            # height of every dark card on the board.
            return self._card(
                UNREPORTED, "Not reported", missing,
                rows=[_row(_short(p), None) for p in self.props])
        try:
            out = self.build(values)
        except Exception as e:                 # a rule must never take the board down
            return self._card(UNREPORTED, "Unreadable", missing,
                              fix=f"Could not be read: {e}")

        fix = out.get("fix", "")
        if essential_missing and not fix:
            fix = ("Not exposed by this server: " +
                   ", ".join(_short(p) for p in essential_missing) + ".")
        state = out.get("state", UNREPORTED)
        # Partial information is not a pass. The card keeps its live headline,
        # but an absent essential reading stops it showing green.
        if essential_missing and state == OK:
            state = WARN
        return self._card(state, out.get("big", ""), missing,
                          rows=out.get("rows", []), fix=fix,
                          chips=out.get("chips", []), tone=out.get("big_tone", ""))

    def _card(self, state: str, big: str, missing: list, *, rows=(), fix="",
              chips=(), tone="") -> dict:
        return {"key": self.key, "title": self.title, "source": self.source,
                "state": state, "big": big, "big_tone": tone,
                "rows": list(rows), "chips": list(chips), "fix": fix,
                "missing": list(missing)}


# ── Rules ─────────────────────────────────────────────────────────────────────

def _system(v: dict) -> dict:
    return {
        "state": OK, "big": "OK",
        "rows": [_row("Server", v.get(SERVER_VERSION)),
                 _row("Firmware", v.get(FIRMWARE))],
    }


def _tem_channel(v: dict) -> dict:
    mag, cl, px = (_num(v.get(MAGNIFICATION)), _num(v.get(CAMERA_LENGTH)),
                   _num(v.get(SPECIMEN_PX)))
    reported = [x for x in (mag, cl, px) if x is not None]
    unset = [x for x in reported if x in TEM_UNSET]

    if reported and len(unset) == len(reported):
        return {
            "state": BAD, "big": "No link",
            "rows": [_row("Magnification", v.get(MAGNIFICATION), BAD),
                     _row("Camera length", v.get(CAMERA_LENGTH), BAD),
                     _row("Specimen px", v.get(SPECIMEN_PX), BAD)],
            "fix": "Every Instrument Project property reads its unset sentinel — "
                   "the microscope is not reporting to DE. Scale bars and "
                   "calibrated pixel sizes will be wrong.",
        }
    st = WARN if unset else OK
    return {
        "state": st, "big": "Reporting" if st == OK else "Partial",
        "rows": [_row("Magnification", None if mag is None else f"×{mag:g}",
                      BAD if mag in TEM_UNSET else OK),
                 _row("Camera length", None if cl is None else f"{cl:g} cm",
                      BAD if cl in TEM_UNSET else OK),
                 _row("Specimen px", None if px is None else f"{px:.4g} nm",
                      BAD if px in TEM_UNSET else OK)],
        "fix": "" if st == OK else
               "Some Instrument Project properties read their unset sentinel.",
    }


def _cooling(v: dict) -> dict:
    det_t, water_t = _num(v.get(DETECTOR_T)), _num(v.get(WATER_T))
    det_state = _status_word(v.get(DETECTOR_STATUS),
                             good=("ok", "normal", "stable", "ready", "cooled"))
    water_state = _status_word(v.get(WATER_STATUS), good=("ok", "normal", "good", "ready"))

    fix = ""
    # A thermocouple reading absolute zero is an ABSENT SENSOR, not cold water.
    # Reporting the status and the value separately is the point: averaging
    # them into one light is how "OK · -273.1 °C" gets shown as green.
    if water_t is not None and water_t < -270:
        water_state = WARN
        fix = ("Chilled-water status says OK but the value is absolute zero — "
               "the sensor is absent, not the water.")
    if det_t is not None and det_t < -270:
        det_state = WARN
        fix = fix or "Detector temperature reads absolute zero — sensor absent."

    big = "—" if det_t is None else f"{det_t:.1f} °C"
    return {
        "state": _worst(det_state, water_state), "big": big, "big_tone": "cryo",
        "rows": [_row("Detector", v.get(DETECTOR_STATUS), det_state),
                 _row("Chilled water",
                      None if water_t is None else f"{_txt(v.get(WATER_STATUS)) or '—'} · {water_t:.1f} °C",
                      water_state),
                 _row("Control", v.get(TEMP_CONTROL))],
        "fix": fix,
    }


def _vacuum_cover(v: dict) -> dict:
    vac = _status_word(v.get(VACUUM), good=("ok", "normal", "ready", "vacuum", "pumped"))
    cover = _status_word(v.get(COVER), good=("open", "closed", "ok", "normal"))
    # Position is not good or bad — a retracted camera is correct when the
    # operator retracted it. Only motion or an error is worth a colour.
    pos_text = _txt(v.get(POSITION))
    pos = (WARN if any(w in pos_text.lower()
                       for w in ("moving", "inserting", "retracting", "unknown"))
           else OK)
    return {
        "state": _worst(vac, cover),
        "big": _txt(v.get(VACUUM)) or "—",
        "rows": [_row("Cover", v.get(COVER), cover),
                 _row("Position", v.get(POSITION), pos)],
    }


def _corrections(v: dict) -> dict:
    bp, ff = _switch(v.get(BAD_PIXEL)), _switch(v.get(FLATFIELD))

    def one(flag, raw):
        if flag is None:
            return (UNREPORTED, "—") if raw is None else (WARN, f"{_txt(raw)}?")
        return (OK, "Applied") if flag else (WARN, "Not applied")

    bp_state, bp_text = one(bp, v.get(BAD_PIXEL))
    ff_state, ff_text = one(ff, v.get(FLATFIELD))
    off = [n for n, f in (("bad pixel", bp), ("flatfield", ff)) if f is False]
    return {
        "state": _worst(bp_state, ff_state),
        "big": "Applied" if bp and ff else ("Partial" if (bp or ff) else "Not applied"),
        "rows": [_row("Bad pixel", bp_text, bp_state),
                 _row("Flatfield", ff_text, ff_state)],
        # Off is a WARNING, never an error: running without flatfield is a
        # legitimate deliberate choice. The board exists to stop it happening
        # by ACCIDENT.
        "fix": (f"{', '.join(off).capitalize()} correction is switched off. "
                "Deliberate is fine; by accident is not.") if off else "",
    }


def _sensor_health(v: dict) -> dict:
    rows = [_row(p.rsplit(" - ", 1)[-1], v.get(p), NO_CRITERIA)
            for p in (*VOLTAGES, *SENSOR_TEMPS) if v.get(p) is not None]
    return {
        "state": NO_CRITERIA, "big": "Not assessed", "rows": rows,
        "fix": "Readings only — no thresholds have been agreed for these yet, "
               "so they are not counted as a passing check.",
    }


CARDS: tuple[Card, ...] = (
    Card("system", "System", "Server Software Version",
         (SERVER_VERSION, FIRMWARE), _system, optional=(FIRMWARE,)),
    Card("tem", "DE ↔ TEM channel", "Instrument Project *",
         (MAGNIFICATION, CAMERA_LENGTH, SPECIMEN_PX), _tem_channel,
         optional=(SPECIMEN_PX,)),
    Card("cooling", "Cooling", "Temperature - *",
         (DETECTOR_T, DETECTOR_STATUS, WATER_T, WATER_STATUS, TEMP_CONTROL),
         _cooling, optional=(TEMP_CONTROL,)),
    Card("vacuum", "Vacuum & cover", "Vacuum State",
         (VACUUM, COVER, POSITION), _vacuum_cover, optional=(POSITION,)),
    Card("corrections", "Image corrections", "Image Processing - *",
         (BAD_PIXEL, FLATFIELD), _corrections),
    Card("sensor", "Sensor health", "System Voltage / Temperature - *",
         (*VOLTAGES, *SENSOR_TEMPS), _sensor_health),
)

#: Every property the board needs, de-duplicated — ONE batched read per refresh.
STATUS_PROPS: tuple[str, ...] = tuple(dict.fromkeys(
    p for c in CARDS for p in c.props))


def build_report(values: dict, *, link: dict | None = None) -> dict:
    """Judge every card and summarise. Pure — no I/O, no server.

    *link* describes the CONNECTION itself, which the caller judges rather than
    this module: it is the one check that must work when nothing else does, so
    it cannot depend on having read a property.
    """
    cards = []
    if link is not None:
        state = link.get("state", UNREPORTED)
        cards.append({
            "key": "connection", "title": "Connection", "source": "DE Server",
            "state": state, "big": "Responding" if state == OK else "No response",
            "rows": [], "chips": [],
            "fix": "" if state == OK else link.get("detail", ""),
            "missing": [],
        })
    cards += [c.run(values) for c in CARDS]

    counts: dict[str, int] = {}
    for c in cards:
        counts[c["state"]] = counts.get(c["state"], 0) + 1

    reporting = len(cards) - counts.get(UNREPORTED, 0)
    if counts.get(BAD):
        overall = BAD
    elif counts.get(WARN):
        overall = WARN
    elif reporting == 0:
        # Nothing to be green ABOUT. Without this, a server that answers
        # nothing produces an all-grey board with a green headline.
        overall = UNREPORTED
    else:
        overall = OK

    return {"cards": cards,
            "summary": {"overall": overall, "counts": counts,
                        "reporting": reporting, "total": len(cards)}}

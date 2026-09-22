"""
composition.py — what the sample is made of, as a list of PHASES, and the
Crystallography Open Database (COD) search that finds a phase its structure.

A phase is its elements (with optional atomic percentages) and, once one is
chosen, the ``.cif`` that indexes it. The phase list is the only thing a person
edits, and each technique reads the part it needs:

- EELS and EDS fit every element of every phase. They read the
  HyperSpy-canonical ``metadata.Sample.elements``, which is rewritten as the
  union of the phases on every edit — and an element removed from the phases
  loses its X-ray lines too, so it is no longer fitted.
- EBSD and 4D-STEM orientation mapping index against each phase's structure.

An element can be marked TRACE in its phase: the O in an Fe phase matters to
EELS and EDS, but the structure is still Fe's. Trace elements are left out of
the phase's COD search and kept when its structure is swapped.

The COD search is scoped to ONE phase because COD matches the elements exactly:
a Cu/Nb sample searched as one composition asks for a Cu-Nb compound and gets
nothing back, while fcc Cu and bcc Nb are one query each.

Phases are addressed by a stable ``id``, never by position: a structure
download can finish after the phase list has changed.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.parse
import urllib.request
import uuid

from de_shell.ipc import emit, emit_error, emit_status
from spyde.actions.context import src_plot_tree as _src_plot_tree
from spyde.actions.lifecycle import run_on_worker

log = logging.getLogger(__name__)

_COD_BASE = "https://www.crystallography.net/cod"
_COD_TIMEOUT = 20      # seconds — network call is short-circuited if COD is down
_MAX_RESULTS = 40

_PHASES_KEY = "Sample.spyde_phases"
_ELEMENTS_KEY = "Sample.elements"
# Atomic percentages as SpyDE stored them before phases existed. Read only to
# open such a file with its percentages on its one phase; removed once phases
# are written, so the file does not carry two answers that can disagree.
_LEGACY_PERCENTAGES_KEY = "Sample.composition"


# ── the phase list ────────────────────────────────────────────────────────────
def _as_mapping(value) -> dict:
    """*value* as a plain dict, or ``{}`` when it is not a mapping at all."""
    if hasattr(value, "as_dictionary"):
        value = value.as_dictionary()
    return dict(value) if isinstance(value, dict) else {}


def _symbols(value) -> list[str]:
    """An element list as metadata may hold it — a list, an array, or a single
    symbol — as unique strings in order."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    symbols: list[str] = []
    for symbol in value:
        symbol = str(symbol)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _clean_phase(raw, fallback_id=None) -> dict:
    """One phase in its stored shape. The dict round-trips through file
    metadata, so unknown keys are dropped, and percentages and trace marks are
    kept only for elements the phase contains."""
    raw = _as_mapping(raw)
    elements = _symbols(raw.get("elements"))
    percentages: dict[str, float] = {}
    for symbol, value in _as_mapping(raw.get("percentages")).items():
        if str(symbol) not in elements:
            continue
        try:
            percentages[str(symbol)] = float(value)
        except (TypeError, ValueError):
            continue
    trace = [symbol for symbol in elements if symbol in _symbols(raw.get("trace"))]
    structure_elements = raw.get("structure_elements")
    return {
        "id": str(raw.get("id") or fallback_id or uuid.uuid4().hex[:12]),
        "elements": elements,
        "percentages": percentages,
        "trace": trace,
        # The structure that indexes this phase. None until one is chosen — a
        # phase with elements but no structure is what COD is searched from.
        "cif_path": str(raw["cif_path"]) if raw.get("cif_path") else None,
        "label": str(raw["label"]) if raw.get("label") else None,
        "cod_id": str(raw["cod_id"]) if raw.get("cod_id") else None,
        # What the structure file contains, read when it was chosen. It tells a
        # phase whose elements came from its file apart from one whose elements
        # the person chose, when the structure is swapped.
        "structure_elements": (None if structure_elements is None
                               else _symbols(structure_elements)),
    }


def elements_of(phases) -> list[str]:
    """Every element of every phase, once each, in the order they were added."""
    elements: list[str] = []
    for phase in phases:
        for symbol in phase["elements"]:
            if symbol not in elements:
                elements.append(symbol)
    return elements


def major_elements(phase) -> list[str]:
    """A phase's elements without its trace ones — what its structure is made of."""
    return [symbol for symbol in phase["elements"] if symbol not in phase["trace"]]


def read_phases(tree) -> list[dict]:
    """The sample's phases.

    Every element in ``Sample.elements`` belongs to a phase. With no phase list
    — elements set by a file reader, or by SpyDE before phases existed — they
    are ONE phase. An element added some other way after phases exist (the
    console, exspy) joins Phase 1 as trace: fitted by EELS and EDS, and left
    out of Phase 1's structure search.
    """
    metadata = tree.root.metadata
    listed = _symbols(metadata.get_item(_ELEMENTS_KEY, None))
    stored = metadata.get_item(_PHASES_KEY, None)
    if not stored:
        if not listed:
            return []
        return [_clean_phase({
            "elements": listed,
            "percentages": metadata.get_item(_LEGACY_PERCENTAGES_KEY, None),
        }, "phase-1")]
    phases = [_clean_phase(phase, f"phase-{number}")
              for number, phase in enumerate(stored, start=1)]
    extras = [symbol for symbol in listed if symbol not in elements_of(phases)]
    if extras:
        first = phases[0]
        phases[0] = _clean_phase(dict(first, elements=first["elements"] + extras,
                                      trace=first["trace"] + extras))
    return phases


def write_phases(tree, phases) -> None:
    """Store *phases* and make the signal name exactly their elements."""
    from spyde.spectroscopy.composition import keep_only_elements

    cleaned = [_clean_phase(phase) for phase in phases]
    root = tree.root
    root.metadata.set_item(_PHASES_KEY, cleaned)
    keep_only_elements(root, elements_of(cleaned))
    if root.metadata.has_item(_LEGACY_PERCENTAGES_KEY):
        parent, _, name = _LEGACY_PERCENTAGES_KEY.rpartition(".")
        delattr(root.metadata.get_item(parent), name)


def phase_label(phase) -> str:
    """How a phase reads in a status line: its structure if it has one, else
    what it is made of."""
    return phase.get("label") or "-".join(phase.get("elements") or []) or "phase"


def emit_composition(session, tree) -> None:
    """Show the sample's phases in the dock of every window of *tree* — the
    navigator included, because the dock follows the focused window and on a
    scan that is usually the navigator."""
    phases = read_phases(tree)
    emit({
        "type": "composition",
        "window_ids": session._tree_window_ids(tree),
        "phases": phases,
        # For whatever only asks what the sample contains (the fit wizards).
        "elements": elements_of(phases),
    })


# ── staged handlers ───────────────────────────────────────────────────────────
def _sample(session, plot):
    """``(tree, phases)`` for the sample *plot* shows, or ``(None, [])``."""
    _source, tree = _src_plot_tree(session, plot)
    if tree is None:
        return None, []
    return tree, read_phases(tree)


def _phase_named(phases, payload, create=False):
    """The phase whose id is ``payload['phase']``, or None.

    With *create*, an id no phase has yet makes a new empty phase with it. The
    popout names a phase before it exists, so the first element clicked — or
    the first structure loaded — creates the phase already shown as selected,
    and a second click made before the reply finds that same phase.
    """
    phase_id = payload.get("phase")
    if not phase_id:
        return None
    for phase in phases:
        if phase["id"] == str(phase_id):
            return phase
    if not create:
        return None
    phase = _clean_phase({"id": phase_id})
    phases.append(phase)
    return phase


def _numbered(phases, phase) -> str:
    return f"Phase {phases.index(phase) + 1}: {phase_label(phase)}"


def _store(session, tree, phases, status) -> None:
    write_phases(tree, phases)
    emit_composition(session, tree)
    emit_status(status)


def add_phase(session, plot, payload) -> None:
    """Append a phase — empty, or made of ``payload['elements']`` — with the id
    ``payload['phase']`` when one is given. Adding an id that exists does
    nothing."""
    tree, phases = _sample(session, plot)
    if tree is None or _phase_named(phases, payload) is not None:
        return
    phases.append(_clean_phase({"id": payload.get("phase"),
                                "elements": payload.get("elements")}))
    _store(session, tree, phases, f"Added phase {len(phases)}")


def remove_phase(session, plot, payload) -> None:
    tree, phases = _sample(session, plot)
    phase = _phase_named(phases, payload)
    if tree is None or phase is None:
        return
    phases.remove(phase)
    _store(session, tree, phases, f"Removed {phase_label(phase)}")


def toggle_phase_element(session, plot, payload) -> None:
    """A click on the periodic table: put ``element`` in the phase, or take it
    out if it is already there.

    A toggle rather than the phase's whole element list, so that two clicks
    made before the first reply arrives cannot overwrite each other.
    """
    tree, phases = _sample(session, plot)
    phase = _phase_named(phases, payload, create=True)
    element = str(payload.get("element") or "")
    if tree is None or phase is None or not element:
        return
    if element in phase["elements"]:
        phase["elements"].remove(element)
    else:
        phase["elements"].append(element)
    _store(session, tree, phases, _numbered(phases, phase))


def set_phase_percentages(session, plot, payload) -> None:
    """Set the atomic percentages in ``payload['percentages']`` on the phase —
    ``{symbol: percent}``, where None clears one. Elements not named keep
    theirs, so two saves made before the first reply do not overwrite each
    other."""
    tree, phases = _sample(session, plot)
    phase = _phase_named(phases, payload)
    if tree is None or phase is None:
        return
    for symbol, percent in _as_mapping(payload.get("percentages")).items():
        if percent is None:
            phase["percentages"].pop(str(symbol), None)
        else:
            phase["percentages"][str(symbol)] = percent
    _store(session, tree, phases, _numbered(phases, phase))


def set_phase_trace(session, plot, payload) -> None:
    """Mark ``element`` as trace in the phase (``trace: true``), or not."""
    tree, phases = _sample(session, plot)
    phase = _phase_named(phases, payload)
    element = str(payload.get("element") or "")
    if tree is None or phase is None or element not in phase["elements"]:
        return
    trace = [symbol for symbol in phase["trace"] if symbol != element]
    if payload.get("trace"):
        trace.append(element)
    phase["trace"] = trace
    marked = "trace" if payload.get("trace") else "not trace"
    _store(session, tree, phases, f"Phase {phases.index(phase) + 1}: {element} is {marked}")


def elements_from_cif(path) -> list[str]:
    """The element symbols a ``.cif`` contains, or ``[]`` if it cannot be read.

    Failure is not an error: the phase keeps its structure and simply has no
    elements from it, which is the state it was in anyway.
    """
    try:
        from orix.crystal_map import Phase
        phase = Phase.from_cif(str(path))
        symbols: list[str] = []
        for atom in phase.structure:
            symbol = str(getattr(atom, "element", "") or "").strip()
            # diffpy writes isotopes and oxidation states ("Fe2+", "O2-").
            symbol = "".join(ch for ch in symbol if ch.isalpha()).capitalize()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        return symbols
    except Exception as e:
        log.debug("reading elements from %s failed: %s", path, e)
        return []


def _apply_structure(session, plot, payload, structure_elements, create) -> None:
    """Give the phase the structure in ``payload['cif_path']`` (whose elements
    are *structure_elements*), or clear it when that is empty.

    The phase's elements follow the file only when they were the file's to
    begin with: none yet, or exactly what the previous structure contained.
    Elements the person chose are kept — they may have said something the file
    cannot, like a solid solution. Trace elements are always kept.
    """
    tree, phases = _sample(session, plot)
    phase = _phase_named(phases, payload, create=create)
    if tree is None:
        return
    if phase is None:
        emit_error("That phase no longer exists, so its structure was not set.")
        return
    path = payload.get("cif_path")
    major = major_elements(phase)
    elements_came_from_file = (
        not major or set(major) == set(phase["structure_elements"] or ()))
    # An unreadable file says nothing about the elements, so it changes none.
    if path and structure_elements and elements_came_from_file:
        phase["elements"] = list(structure_elements) + [
            symbol for symbol in phase["trace"] if symbol not in structure_elements]
    if payload.get("label"):
        label = str(payload["label"])
    elif path:
        label = os.path.splitext(os.path.basename(str(path)))[0]
    else:
        label = None
    phase["cif_path"] = str(path) if path else None
    phase["label"] = label
    phase["cod_id"] = str(payload["cod_id"]) if payload.get("cod_id") else None
    phase["structure_elements"] = list(structure_elements) if path else None
    _store(session, tree, phases, _numbered(phases, phase))
    # An orientation wizard's IPF window shows the phases' triangles as soon
    # as they have a structure — see ipf_panel.
    from spyde.actions.ipf_panel import phases_changed
    try:
        phases_changed(session, tree)
    except Exception as e:
        log.debug("updating the IPF panel phases failed: %s", e)


def set_phase_structure(session, plot, payload) -> None:
    """Give the phase the ``.cif`` in ``payload['cif_path']``, or clear its
    structure when that is empty. A new phase id creates the phase."""
    path = payload.get("cif_path")
    if not path:
        _apply_structure(session, plot, payload, [], create=False)
        return
    # The first read imports the crystallography stack; keep it off the main thread.
    run_on_worker(
        session, lambda: elements_from_cif(path), name="read-cif",
        on_done=lambda elements: _apply_structure(session, plot, payload, elements,
                                                  create=True))


# ── COD structure search ───────────────────────────────────────────────────────
def _cod_query(elements) -> list[dict]:
    """Query the COD REST API for structures with EXACTLY ``elements`` and return
    raw result dicts. Raises on network/HTTP error (caller handles)."""
    els = [e for e in elements if e]
    if not els:
        return []
    n = len(els)
    params = [(f"el{i + 1}", el) for i, el in enumerate(els)]
    params += [("strictmin", str(n)), ("strictmax", str(n)), ("format", "json")]
    url = f"{_COD_BASE}/result.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "SpyDE/0.1"})
    with urllib.request.urlopen(req, timeout=_COD_TIMEOUT) as resp:
        data = resp.read().decode("utf-8", "replace")
    try:
        return json.loads(data) if data.strip() else []
    except json.JSONDecodeError:
        return []


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _tidy_results(raw) -> list[dict]:
    """Normalise + dedupe COD rows into compact picker entries (formula, phase,
    space group, a/b/c/α/β/γ). Dedupes near-identical redeterminations."""
    out, seen = [], set()
    for r in raw:
        a, b, c = _fnum(r.get("a")), _fnum(r.get("b")), _fnum(r.get("c"))
        if a is None or b is None or c is None:
            continue
        formula = (r.get("formula") or r.get("calcformula") or "").strip(" -") or "?"
        phase = (r.get("mineral") or r.get("commonname") or r.get("chemname") or "").strip()
        sg = (r.get("sg") or "").strip()
        key = (formula, sg, round(a, 2), round(b, 2), round(c, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "id": str(r.get("file", "")),
            "formula": formula,
            "phase": phase,
            "sg": sg,
            "sg_number": _fnum(r.get("sgNumber")),
            "a": a, "b": b, "c": c,
            "alpha": _fnum(r.get("alpha")), "beta": _fnum(r.get("beta")),
            "gamma": _fnum(r.get("gamma")),
            "volume": _fnum(r.get("vol")),
        })
    # Smaller, simpler cells first (the common phases people want).
    out.sort(key=lambda e: (e["volume"] or 1e9))
    return out[:_MAX_RESULTS]


def cod_search(session, plot, payload) -> None:
    """Search COD for structures made of exactly the phase's non-trace elements.

    Replies with a ``cod_results`` message carrying the same ``phase`` id, so
    the popout shows the results under the phase that asked.
    """
    source, tree = _src_plot_tree(session, plot)
    window_id = getattr(source, "window_id", None)
    phase = _phase_named(read_phases(tree), payload) if tree is not None else None
    elements = major_elements(phase) if phase is not None else []
    if not elements:
        emit_error("Give the phase a non-trace element before searching COD.")
        return

    def reply(results, error=None):
        emit({"type": "cod_results", "window_id": window_id, "phase": phase["id"],
              "elements": elements, "results": results, "error": error})

    def search():
        emit_status(f"Searching COD for {'-'.join(elements)} structures…")
        return _tidy_results(_cod_query(elements))

    def found(results):
        reply(results)
        emit_status(f"COD: {len(results)} structure(s) for {'-'.join(elements)}")

    def failed(error):
        log.debug("COD search failed: %s", error)
        reply([], "COD search failed (offline?)")
        emit_status("COD search failed — check your connection")

    run_on_worker(session, search, name="cod-search", on_done=found, on_error=failed)


def fetch_cod_cif(cod_id: str) -> str:
    """Download COD entry ``cod_id`` as a ``.cif`` to a temp file → its path."""
    cod_id = "".join(ch for ch in str(cod_id) if ch.isdigit())
    if not cod_id:
        raise ValueError("invalid COD id")
    url = f"{_COD_BASE}/{cod_id}.cif"
    req = urllib.request.Request(url, headers={"User-Agent": "SpyDE/0.1"})
    with urllib.request.urlopen(req, timeout=_COD_TIMEOUT) as resp:
        text = resp.read().decode("utf-8", "replace")
    if "loop_" not in text and "_cell_length_a" not in text:
        raise ValueError("COD response did not look like a CIF")
    path = os.path.join(tempfile.gettempdir(), f"cod_{cod_id}.cif")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def cod_pick(session, plot, payload) -> None:
    """Download COD entry ``cod_id`` and make it the phase's structure.

    The download and the file read run off the main thread; the phase — found
    again by id, since the list may have changed meanwhile — is updated back on
    it, like every other phase edit.
    """
    cod_id = payload.get("cod_id")
    phase_id = payload.get("phase")
    if not cod_id or not phase_id:
        emit_error("A COD pick needs both a COD id and a phase.")
        return
    label = payload.get("label") or f"COD {cod_id}"

    def download():
        path = fetch_cod_cif(cod_id)
        return path, elements_from_cif(path)

    def bind(result):
        path, structure_elements = result
        _apply_structure(session, plot, {
            "phase": phase_id, "cif_path": path, "label": label, "cod_id": str(cod_id),
        }, structure_elements, create=False)

    def failed(error):
        emit_error(f"Could not download COD {cod_id}: {error}")

    run_on_worker(session, download, name="cod-pick", on_done=bind, on_error=failed)

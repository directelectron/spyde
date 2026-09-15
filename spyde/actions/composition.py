"""
composition.py — sample composition (elements + percentages) and the
composition-driven "easy CIF" picker (Crystallography Open Database search).

Composition lives in the HyperSpy signal metadata at the canonical
``metadata.Sample.elements`` (list of symbols) + ``metadata.Sample.composition``
(``{symbol: atomic_percent}``) — the same place HyperSpy's EDS/EELS tooling
reads. The right dock shows it and a periodic-table popout edits it.

The composition then drives CIF picking: ``cod_search`` queries the COD REST API
for structures with exactly those elements and returns a tidy list (formula,
phase, space group, a/b/c/α/β/γ) to choose from; ``cod_pick`` downloads the
chosen ``.cif`` so it can be used as an orientation-mapping phase. No Qt.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.parse
import urllib.request

from de_shell.ipc import emit, emit_error, emit_status
from spyde.actions.context import src_plot_tree as _src_plot_tree

log = logging.getLogger(__name__)

_COD_BASE = "https://www.crystallography.net/cod"
_COD_TIMEOUT = 20      # seconds — network call is short-circuited if COD is down
_MAX_RESULTS = 40


# ── metadata read / write ──────────────────────────────────────────────────────
def read_composition(tree) -> tuple[list[str], dict[str, float]]:
    """Return ``(elements, percentages)`` from the tree's root signal metadata."""
    md = tree.root.metadata
    elements = list(md.get_item("Sample.elements", []) or [])
    comp_raw = md.get_item("Sample.composition", {}) or {}
    percentages: dict[str, float] = {}
    try:
        # HyperSpy stores a dict-like DictionaryTreeBrowser; normalise to floats.
        items = comp_raw.as_dictionary() if hasattr(comp_raw, "as_dictionary") else dict(comp_raw)
        for k, v in items.items():
            try:
                percentages[str(k)] = float(v)
            except (TypeError, ValueError) as e:
                log.debug("composition value %r=%r not numeric, skipping: %s", k, v, e)
    except Exception as e:
        log.debug("parsing composition metadata failed: %s", e)
    return [str(e) for e in elements], percentages


def write_composition(tree, elements, percentages=None) -> None:
    """Write ``Sample.elements`` (list) + ``Sample.composition`` (dict) to the
    tree's root signal metadata (the HyperSpy-canonical location)."""
    md = tree.root.metadata
    elements = [str(e) for e in elements]
    md.set_item("Sample.elements", elements)
    comp = {e: float(percentages.get(e)) for e in elements
            if percentages and percentages.get(e) is not None}
    md.set_item("Sample.composition", comp)


# ── phases ────────────────────────────────────────────────────────────────────
# A sample is made of PHASES, and a phase is two things that were previously
# kept apart: what it is made of, and the structure that indexes it. Keeping
# them apart is why a two-phase sample could not be described at all — the
# composition was one flat element list, so Cu-and-Nb read as "a compound of Cu
# and Nb" and COD was asked for a structure containing both (it returns
# nothing; the two elemental phases it should have found are one query each).
#
# ``Sample.elements`` / ``Sample.composition`` stay exactly as they were — the
# HyperSpy-canonical flat union that EELS edge suggestion and EDS quantification
# read (``spyde/spectroscopy/edges.py``, ``composition.py``). The phase list
# lives beside them, so nothing that already reads the canonical fields changes
# behaviour and a file written by an older SpyDE still opens.
_PHASES_KEY = "Sample.spyde_phases"


def _clean_phase(raw) -> dict:
    """One stored phase, normalised. Unknown keys are dropped rather than kept:
    this dict round-trips through file metadata, so it stays a fixed shape."""
    if hasattr(raw, "as_dictionary"):
        raw = raw.as_dictionary()
    raw = dict(raw or {})
    percentages = raw.get("percentages") or {}
    if hasattr(percentages, "as_dictionary"):
        percentages = percentages.as_dictionary()
    clean_pct = {}
    for symbol, value in dict(percentages).items():
        try:
            clean_pct[str(symbol)] = float(value)
        except (TypeError, ValueError):
            continue
    return {
        "elements": [str(e) for e in (raw.get("elements") or []) if e],
        "percentages": clean_pct,
        # The structure that indexes this phase, once one is chosen. None until
        # then — a phase whose composition is known but whose structure is not
        # is a normal, useful state (it is what you search COD from).
        "cif_path": str(raw["cif_path"]) if raw.get("cif_path") else None,
        "label": str(raw["label"]) if raw.get("label") else None,
        "cod_id": str(raw["cod_id"]) if raw.get("cod_id") else None,
    }


def read_phases(tree) -> list[dict]:
    """The sample's phases, outermost-first.

    A signal that predates phases — or one whose composition was set through the
    flat path — reports its composition as a SINGLE phase, so every caller can
    be written against the list and none needs to know which era the file is
    from.
    """
    try:
        stored = tree.root.metadata.get_item(_PHASES_KEY, None)
    except Exception as e:
        log.debug("reading phases failed: %s", e)
        stored = None
    if stored:
        return [_clean_phase(p) for p in stored]
    elements, percentages = read_composition(tree)
    if not elements:
        return []
    return [_clean_phase({"elements": elements, "percentages": percentages})]


def write_phases(tree, phases) -> None:
    """Store *phases* and fold their elements into the canonical flat fields.

    The sample's own element list is KEPT and added to, never replaced by the
    union across phases. An element can belong to the sample without belonging
    to any phase — the extra oxygen that is in neither structure being indexed
    against — and rebuilding ``Sample.elements`` from the phases alone would
    silently drop it the next time any phase was edited.

    Order is the sample's first, then anything a phase introduced, so the list
    reads the way the person building it added things.
    """
    cleaned = [_clean_phase(p) for p in phases]
    elements, percentages = read_composition(tree)
    elements = list(elements)
    for phase in cleaned:
        for symbol in phase["elements"]:
            if symbol not in elements:
                elements.append(symbol)
            percentages.setdefault(symbol, phase["percentages"].get(symbol))
    percentages = {k: v for k, v in percentages.items() if v is not None}
    md = tree.root.metadata
    md.set_item(_PHASES_KEY, cleaned)
    write_composition(tree, elements, percentages)


def phase_label(phase) -> str:
    """How a phase reads in a status line: its structure if it has one, else
    just what it is made of."""
    return phase.get("label") or "-".join(phase.get("elements") or []) or "phase"


def emit_composition(tree, window_ids) -> None:
    """Push the current composition to the dock for the given windows."""
    elements, percentages = read_composition(tree)
    emit({
        "type": "composition",
        "window_ids": list(window_ids),
        "elements": elements,
        "percentages": percentages,
        # The dock renders one group per phase and shows its structure beside
        # the chips; the flat fields above stay for anything that only wants
        # "what is this sample made of".
        "phases": read_phases(tree),
    })


def _window_ids_for(tree) -> list[int]:
    ids = []
    for sp in list(getattr(tree, "signal_plots", []) or []):
        wid = getattr(sp, "window_id", None)
        if wid is not None:
            ids.append(int(wid))
    return ids


def set_composition(session, plot, payload) -> None:
    """Staged handler: persist the chosen elements + percentages to metadata and
    echo the composition back to the dock. ``payload`` =
    ``{elements: [...], percentages: {El: pct}}``."""
    src, tree = _src_plot_tree(session, plot)
    if tree is None:
        return
    elements = [str(e) for e in (payload.get("elements") or [])]
    percentages = payload.get("percentages") or {}
    try:
        write_composition(tree, elements, percentages)
        # Un-ticking an element in the periodic table has to remove it from the
        # phases too. A phase is a SUBSET of the sample, so leaving it behind
        # would both contradict that and quietly put the element back the next
        # time any phase was written.
        phases = read_phases(tree)
        kept = [dict(p, elements=[e for e in p["elements"] if e in elements])
                for p in phases]
        if kept != phases:
            write_phases(tree, kept)
    except Exception as e:
        emit_error(f"Could not set composition: {e}")
        return
    emit_composition(tree, _window_ids_for(tree))
    pretty = ", ".join(
        f"{el} {percentages[el]:g}%" if percentages.get(el) is not None else el
        for el in elements
    )
    emit_status(f"Composition: {pretty}" if elements else "Composition cleared")


def _phases_and_tree(session, plot):
    """``(phases, tree)`` for a staged phase handler, or ``(None, None)``."""
    _src, tree = _src_plot_tree(session, plot)
    if tree is None:
        return None, None
    return read_phases(tree), tree


def _push_phases(tree, phases, status=None) -> None:
    write_phases(tree, phases)
    emit_composition(tree, _window_ids_for(tree))
    if status:
        emit_status(status)


def add_phase(session, plot, payload) -> None:
    """The ``&`` button: append a phase. ``payload`` may carry ``elements``
    (and ``percentages``) for it; an empty one is fine — the widget opens the
    periodic table on it next."""
    phases, tree = _phases_and_tree(session, plot)
    if tree is None:
        return
    phases.append(_clean_phase({
        "elements": payload.get("elements") or [],
        "percentages": payload.get("percentages") or {},
        "cif_path": payload.get("cif_path"),
        "label": payload.get("label"),
    }))
    _push_phases(tree, phases, f"Added phase {len(phases)}")


def remove_phase(session, plot, payload) -> None:
    """Drop one phase. Removing the last one clears the composition rather than
    leaving an empty list that reads as "no phases known"."""
    phases, tree = _phases_and_tree(session, plot)
    if tree is None:
        return
    index = int(payload.get("index", -1))
    if not (0 <= index < len(phases)):
        return
    dropped = phases.pop(index)
    _push_phases(tree, phases, f"Removed {phase_label(dropped)}")


def set_phase(session, plot, payload) -> None:
    """Set one phase's composition — the periodic table, scoped to a phase.

    Its structure is left alone: changing what a phase is made of does not by
    itself invalidate the .cif you chose for it, and silently dropping one
    would be worse than letting you see that they disagree.
    """
    phases, tree = _phases_and_tree(session, plot)
    if tree is None:
        return
    index = int(payload.get("index", -1))
    while index >= len(phases):        # setting phase N creates it
        phases.append(_clean_phase({}))
    if index < 0:
        return
    phases[index]["elements"] = [str(e) for e in (payload.get("elements") or [])]
    phases[index]["percentages"] = _clean_phase(
        {"percentages": payload.get("percentages") or {}})["percentages"]
    _push_phases(tree, phases, f"Phase {index + 1}: {phase_label(phases[index])}")


def elements_from_cif(path) -> list[str]:
    """The element symbols a ``.cif`` contains, or ``[]`` if it cannot be read.

    A structure file already knows what it is made of, so a phase added from one
    should not also have to be told. Failure is not an error: the phase keeps
    its structure and simply has no composition, which is the state it would
    have been left in anyway.
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


def set_phase_structure(session, plot, payload) -> None:
    """Bind a ``.cif`` to a phase — the file route into the phase widget. The
    COD route lands here too, through :func:`cod_pick`."""
    phases, tree = _phases_and_tree(session, plot)
    if tree is None:
        return
    index = int(payload.get("index", -1))
    while index >= len(phases):
        phases.append(_clean_phase({}))
    if index < 0:
        return
    path = payload.get("cif_path")
    phases[index]["cif_path"] = str(path) if path else None
    phases[index]["label"] = str(payload["label"]) if payload.get("label") else (
        os.path.splitext(os.path.basename(str(path)))[0] if path else None)
    phases[index]["cod_id"] = str(payload["cod_id"]) if payload.get("cod_id") else None
    # Take the composition from the file when the phase has none — it is in
    # there, and a phase that knows its structure but claims no elements reads
    # as a mistake. An existing composition is never overwritten: the user may
    # have said something the file cannot (a solid solution, a measured
    # percentage), and the file does not get to argue with that.
    if path and not phases[index]["elements"]:
        phases[index]["elements"] = elements_from_cif(path)
    _push_phases(tree, phases,
                 f"Phase {index + 1}: {phase_label(phases[index])}")


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
    """Staged handler: search the COD for structures matching ONE phase.

    Scoped to a phase (``payload['phase']``, an index) rather than to the whole
    sample, because the query asks for a structure containing EXACTLY these
    elements. A two-phase Cu/Nb sample searched as one composition asks for a
    Cu-Nb compound and gets nothing back; searched a phase at a time it finds
    fcc Cu and bcc Nb, which is what the sample actually contains.

    ``payload['elements']`` still overrides everything, and with neither the
    whole composition is used — the pre-phase behaviour, for a sample that
    really is one phase.
    """
    src, tree = _src_plot_tree(session, plot)
    window_id = getattr(src, "window_id", None) if src is not None else None
    phase_index = payload.get("phase")
    elements = [str(e) for e in (payload.get("elements") or [])]
    if not elements and tree is not None and phase_index is not None:
        phases = read_phases(tree)
        if 0 <= int(phase_index) < len(phases):
            elements = phases[int(phase_index)]["elements"]
    if not elements and tree is not None:
        elements, _ = read_composition(tree)
    if not elements:
        emit_error("Set a composition (elements) first to search structures.")
        return

    def _work():
        emit_status(f"Searching COD for {'-'.join(elements)} structures…")
        try:
            results = _tidy_results(_cod_query(elements))
        except Exception as e:
            log.debug("COD search failed: %s", e)
            emit({"type": "cod_results", "window_id": window_id,
                  "phase": phase_index, "elements": elements, "results": [],
                  "error": "COD search failed (offline?)"})
            emit_status("COD search failed — check your connection")
            return
        emit({"type": "cod_results", "window_id": window_id,
              "phase": phase_index, "elements": elements, "results": results})
        emit_status(f"COD: {len(results)} structure(s) for {'-'.join(elements)}")

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="cod-search")


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
    """Staged handler: download the chosen COD structure's CIF and tell the
    frontend its local path (the OM wizard adds it as a phase). ``payload`` =
    ``{cod_id, label}``. Runs off-thread (network)."""
    src, tree = _src_plot_tree(session, plot)
    window_id = getattr(src, "window_id", None) if src is not None else None
    cod_id = payload.get("cod_id")
    label = payload.get("label") or f"COD {cod_id}"
    phase_index = payload.get("phase")
    if not cod_id:
        return

    def _work():
        try:
            path = fetch_cod_cif(cod_id)
        except Exception as e:
            emit_error(f"Could not download COD {cod_id}: {e}")
            return
        # The download is BOUND to its phase, not just handed to whoever asked.
        # That is what makes the sample remember its own structures: the dock
        # can show them, and a wizard reads them off the sample instead of
        # keeping a private list that nothing else can see.
        if tree is not None and phase_index is not None:
            set_phase_structure(session, plot, {
                "index": int(phase_index), "cif_path": path,
                "label": label, "cod_id": str(cod_id),
            })
        emit({"type": "cod_cif_ready", "window_id": window_id,
              "phase": phase_index, "cod_id": str(cod_id),
              "path": path, "label": label})
        emit_status(f"Loaded structure {label}")

    from spyde.actions.lifecycle import run_on_worker
    run_on_worker(session, _work, name="cod-pick")

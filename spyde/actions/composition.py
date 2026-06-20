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
import threading
import urllib.parse
import urllib.request

from spyde.backend.ipc import emit, emit_error, emit_status
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


def emit_composition(tree, window_ids) -> None:
    """Push the current composition to the dock for the given windows."""
    elements, percentages = read_composition(tree)
    emit({
        "type": "composition",
        "window_ids": list(window_ids),
        "elements": elements,
        "percentages": percentages,
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
    except Exception as e:
        emit_error(f"Could not set composition: {e}")
        return
    emit_composition(tree, _window_ids_for(tree))
    pretty = ", ".join(
        f"{el} {percentages[el]:g}%" if percentages.get(el) is not None else el
        for el in elements
    )
    emit_status(f"Composition: {pretty}" if elements else "Composition cleared")


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
    """Staged handler: search the COD for structures matching the composition.
    ``payload['elements']`` overrides the stored composition. Runs off-thread
    (network) and emits ``cod_results``."""
    src, tree = _src_plot_tree(session, plot)
    window_id = getattr(src, "window_id", None) if src is not None else None
    elements = [str(e) for e in (payload.get("elements") or [])]
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
                  "elements": elements, "results": [],
                  "error": "COD search failed (offline?)"})
            emit_status("COD search failed — check your connection")
            return
        emit({"type": "cod_results", "window_id": window_id,
              "elements": elements, "results": results})
        emit_status(f"COD: {len(results)} structure(s) for {'-'.join(elements)}")

    threading.Thread(target=_work, daemon=True).start()


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
    src, _ = _src_plot_tree(session, plot)
    window_id = getattr(src, "window_id", None) if src is not None else None
    cod_id = payload.get("cod_id")
    label = payload.get("label") or f"COD {cod_id}"
    if not cod_id:
        return

    def _work():
        try:
            path = fetch_cod_cif(cod_id)
        except Exception as e:
            emit_error(f"Could not download COD {cod_id}: {e}")
            return
        emit({"type": "cod_cif_ready", "window_id": window_id,
              "cod_id": str(cod_id), "path": path, "label": label})
        emit_status(f"Loaded structure {label}")

    threading.Thread(target=_work, daemon=True).start()

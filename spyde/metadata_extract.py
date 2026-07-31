"""
metadata_extract.py — Qt-free metadata extraction.

Resolves METADATA_WIDGET_CONFIG against a signal tree into a plain
``{group: {label: "value units"}}`` dict the Electron sidebar can render.
Kept separate from signal_tree_presenter (which imports Qt) so the backend can
use it without pulling in PySide6.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from spyde import METADATA_WIDGET_CONFIG

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from spyde.signal_tree import BaseSignalTree


def read_metadata_prop(signal_tree: "BaseSignalTree", value: dict):
    """Resolve one config entry to (value, key). ``key`` is the writable
    metadata path, or None for derived attr/function props."""
    if "key" in value:
        return (
            signal_tree.root.metadata.get_item(
                item_path=value["key"], default=value.get("default", "--")
            ),
            value["key"],
        )
    if "attr" in value:
        return signal_tree.get_nested_attr(value["attr"]), None
    if "function" in value:
        fun = signal_tree.get_nested_attr(value["function"])
        return (fun() if callable(fun) else "--"), None
    return "--", None


def _clean(value) -> str:
    if value in (None, "<undefined>"):
        return ""
    return str(value)


def build_axes_list(signal_tree: "BaseSignalTree") -> list[dict]:
    """Return the root signal's axes as plain dicts for the sidebar table.

    One row per axis (navigation + signal), in array order. ``scale``/``offset``
    are ``None`` for non-uniform/functional axes (rendered read-only). The
    ``index`` is the stable handle the renderer sends back in ``set_axis``.
    """
    am = signal_tree.root.axes_manager
    rows: list[dict] = []
    for i, ax in enumerate(am._axes):
        scale = getattr(ax, "scale", None)
        offset = getattr(ax, "offset", None)
        rows.append({
            "index": i,
            "name": _clean(getattr(ax, "name", "")),
            "size": int(getattr(ax, "size", 0)),
            "scale": float(scale) if isinstance(scale, (int, float)) else None,
            "offset": float(offset) if isinstance(offset, (int, float)) else None,
            "units": _clean(getattr(ax, "units", "")),
            "navigate": bool(getattr(ax, "navigate", False)),
        })
    return rows


def build_metadata_editable(signal_tree: "BaseSignalTree") -> dict[str, dict[str, str]]:
    """Return ``{group: {prop: raw}}`` for the config-declared cells that are
    writable — i.e. have a ``key`` (a real ``metadata.set_item`` path), as
    opposed to a derived ``attr``/``function`` entry (Dtype, Dim.) which is
    computed, not stored, and so can't be written back. Cell PRESENCE gates
    which metadata-panel cells accept a click-to-edit (mirroring the axes
    table's per-field ``ax[field] != None`` check); the VALUE is the cell's
    current RAW metadata value as a string ("" when unset).

    The raw value matters: the display string ``build_metadata_dict`` produces
    has units baked in ("12000.5 x"), and committing that back would fail the
    float() parse in ``_set_metadata`` — so the renderer pre-fills the inline
    editor with THIS unit-free value while the cell keeps displaying the
    formatted one (the same value/``display`` split the axes table's
    EditableCell already uses for rounded scale/offset).

    The synthetic ``Dataset`` subsection (shape/dtype/chunking, appended in
    ``build_metadata_dict``) isn't config-driven at all, so it has no entry
    here and stays fully read-only."""
    editable: dict[str, dict[str, str]] = {}
    for subsection, props in METADATA_WIDGET_CONFIG["metadata_widget"].items():
        cells: dict[str, str] = {}
        for prop, value in props.items():
            if "key" not in value:
                continue
            current = signal_tree.root.metadata.get_item(
                item_path=value["key"], default=None
            )
            cells[prop] = _clean(current)
        if cells:
            editable[subsection] = cells
    return editable


#: Per dimension, the most chunk boundaries worth sending. A 1-frame-per-chunk
#: movie has one entry per frame — thousands of numbers nobody can draw and
#: nobody can read. The viewer draws the first ``_MAX_CHUNK_ENTRIES`` and says
#: how many there really are.
_MAX_CHUNK_ENTRIES = 128


def build_chunk_info(signal_tree: "BaseSignalTree") -> dict | None:
    """Describe the displayed node's dask chunking, or ``None`` if it is not lazy.

    Structured numbers, NOT rendered HTML: dask's own ``_repr_html_`` is a
    light-themed block of markup that would have to be restyled from the outside,
    and it knows nothing about which axes are NAVIGATION and which are SIGNAL —
    the distinction that decides whether this chunking is good or ruinous here (a
    chunk that splits the signal axes makes every navigator frame read several
    chunks; see the storage-alignment note in CLAUDE.md). So the backend sends
    the facts and the dock draws them in its own palette, flagging that split.
    """
    sig = _displayed_signal(signal_tree)
    data = getattr(sig, "data", None)
    chunks = getattr(data, "chunks", None)
    if data is None or chunks is None:
        return None
    try:
        am = sig.axes_manager
        # hyperspy stores navigation axes as the LEADING array dimensions.
        nav_ndim = int(am.navigation_dimension)
        itemsize = int(data.dtype.itemsize)
        per_dim = [[int(c) for c in dim] for dim in chunks]
        # The largest block the scheduler can hand out, which is the number that
        # decides whether a single-frame read is cheap.
        max_block = 1
        for dim in per_dim:
            max_block *= max(dim) if dim else 1
        n_chunks = 1
        for dim in per_dim:
            n_chunks *= len(dim)
        sig_dims = per_dim[nav_ndim:]
        return {
            "shape": [int(s) for s in data.shape],
            "chunks": [d[:_MAX_CHUNK_ENTRIES] for d in per_dim],
            "counts": [len(d) for d in per_dim],
            "names": [_clean(getattr(ax, "name", "")) for ax in am._axes],
            "nav_ndim": nav_ndim,
            "dtype": str(data.dtype),
            "itemsize": itemsize,
            "nbytes": int(np.prod(data.shape)) * itemsize,
            "chunk_bytes": int(max_block) * itemsize,
            "n_chunks": int(n_chunks),
            # THE thing worth knowing: does one chunk hold whole signal frames?
            "signal_split": any(len(d) > 1 for d in sig_dims),
        }
    except Exception as e:
        log.debug("building chunk info failed: %s", e)
        return None


def _displayed_signal(signal_tree: "BaseSignalTree"):
    """The signal actually on screen for this tree (which may be a derived node),
    falling back to the root."""
    for p in getattr(signal_tree, "signal_plots", []) or []:
        ps = getattr(p, "plot_state", None)
        if ps is not None and getattr(ps, "current_signal", None) is not None:
            return ps.current_signal
    return signal_tree.root


def build_metadata_info() -> dict[str, dict[str, dict[str, str]]]:
    """Return ``{group: {prop: {description, key, units}}}`` — the STATIC part of
    the config, for the dock's field-detail popover.

    The sidebar only has room for a curated summary (a four-across instrument
    row, three experiment fields), so every field also gets a detail box that
    explains what it is and where it is stored. That text already exists as the
    YAML ``description``; without it the popover could only repeat the number
    the user is already looking at. Config-derived and signal-independent, so no
    signal tree is needed to build it.
    """
    info: dict[str, dict[str, dict[str, str]]] = {}
    for subsection, props in METADATA_WIDGET_CONFIG["metadata_widget"].items():
        cells: dict[str, dict[str, str]] = {}
        for prop, value in props.items():
            cells[prop] = {
                "description": str(value.get("description", "")),
                # The stored path for a writable cell; an attr/function entry is
                # derived, and says so instead of naming a path that isn't one.
                "key": str(value.get("key", "")),
                "units": str(value.get("units", "")),
                "derived": "" if "key" in value else "yes",
            }
        info[subsection] = cells
    return info


def build_metadata_dict(signal_tree: "BaseSignalTree") -> dict[str, dict[str, str]]:
    """Return metadata for *signal_tree* as a nested plain dict."""
    subsections: dict[str, dict[str, str]] = {}
    for subsection, props in METADATA_WIDGET_CONFIG["metadata_widget"].items():
        subsections[subsection] = {}
        for prop, value in props.items():
            current_value, _ = read_metadata_prop(signal_tree, value)
            subsections[subsection][prop] = (
                f"{current_value} {value.get('units', '')}".strip()
            )

    # Dataset shape/dtype — surfaced here so the axes table doesn't need a size
    # column (the displayed signal node, which may differ from root).
    try:
        sig = _displayed_signal(signal_tree)
        am = sig.axes_manager
        nav = " × ".join(str(int(s)) for s in am.navigation_shape) or "—"
        sg = " × ".join(str(int(s)) for s in am.signal_shape) or "—"
        shape = f"nav {nav} · sig {sg}" if nav != "—" else f"sig {sg}"
        data = getattr(sig, "data", None)
        ds = {
            "Shape": shape,
            "Dtype": str(getattr(data, "dtype", "—")),
        }
        # Chunking — only meaningful for lazy (dask) data. Show the per-chunk
        # block size + size in MB so an oversized / signal-split chunking (the
        # navigator-killing default on some MRC readers) is visible at a glance.
        chunksize = getattr(data, "chunksize", None)
        if chunksize is not None:
            try:
                itemsize = data.dtype.itemsize
                mb = float(np.prod(chunksize)) * itemsize / 1e6
                ds["Chunks"] = " × ".join(str(int(c)) for c in chunksize) + f"  ({mb:.0f} MB)"
                ds["Lazy"] = "yes"
            except Exception as e:
                log.debug("formatting chunk info failed: %s", e)
        else:
            ds["Lazy"] = "no"
        subsections["Dataset"] = ds
    except Exception as e:
        log.debug("building Dataset metadata subsection failed: %s", e)

    # Movie fps / frame time: prefer the explicit metadata key (filled above). If
    # it's absent and the leading navigation axis is calibrated in a KNOWN time
    # unit, DERIVE fps = 1/(scale in seconds) so a calibrated movie shows real
    # numbers instead of "--". We require a recognised time UNIT (not just a
    # "time"-ish name) and convert it to seconds — deriving from an unconvertible
    # unit (or a bare uncalibrated name) would show a wrong fps, worse than "--".
    try:
        movie = subsections.get("Movie / In-Situ")
        if movie is not None:
            sig = signal_tree.root
            am = sig.axes_manager
            if am.navigation_dimension >= 1:
                ax = am.navigation_axes[0]
                units = str(getattr(ax, "units", "") or "").strip().lower()
                scale = float(getattr(ax, "scale", 0.0) or 0.0)
                # Unit → seconds factor. Covers every unit the loader treats as a
                # movie time axis (_session_files._TIME_AXIS_UNITS); an unlisted
                # unit yields no derivation (stays "--").
                to_seconds = {
                    "s": 1.0, "sec": 1.0, "secs": 1.0,
                    "second": 1.0, "seconds": 1.0,
                    "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
                    "us": 1e-6, "µs": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
                    "min": 60.0, "minute": 60.0, "minutes": 60.0,
                }.get(units)
                if to_seconds is not None and scale > 0:
                    per_frame_s = scale * to_seconds
                    if per_frame_s > 0:
                        # Only fill when the YAML key gave nothing ("-- …").
                        if movie.get("FPS", "").startswith("--"):
                            movie["FPS"] = f"{1.0 / per_frame_s:.3g} fps"
                        if movie.get("Frame time", "").startswith("--"):
                            movie["Frame time"] = f"{per_frame_s:.3g} s"
    except Exception as e:
        log.debug("deriving movie fps from time axis failed: %s", e)

    return subsections

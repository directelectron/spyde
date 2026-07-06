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
        sig = None
        for p in getattr(signal_tree, "signal_plots", []) or []:
            ps = getattr(p, "plot_state", None)
            if ps is not None and getattr(ps, "current_signal", None) is not None:
                sig = ps.current_signal
                break
        sig = sig if sig is not None else signal_tree.root
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

    # Movie fps / frame time: prefer the explicit metadata key (filled above), but
    # if it's absent and the leading navigation axis is a calibrated TIME axis
    # (an in-situ movie: name "time" / units in seconds), DERIVE fps = 1/scale and
    # the frame time = scale so a calibrated movie shows real numbers instead of "--".
    try:
        movie = subsections.get("Movie / In-Situ")
        if movie is not None:
            sig = signal_tree.root
            am = sig.axes_manager
            if am.navigation_dimension >= 1:
                ax = am.navigation_axes[0]
                name = str(getattr(ax, "name", "") or "").strip().lower()
                units = str(getattr(ax, "units", "") or "").strip().lower()
                scale = float(getattr(ax, "scale", 0.0) or 0.0)
                is_time = name in ("time", "t") or units in (
                    "s", "sec", "secs", "second", "seconds")
                # Only fill when the YAML key gave nothing (value starts with "--").
                if is_time and scale > 0:
                    per_frame_s = scale
                    if units in ("ms", "millisecond", "milliseconds"):
                        per_frame_s = scale / 1000.0
                    if movie.get("FPS", "").startswith("--"):
                        movie["FPS"] = f"{1.0 / per_frame_s:.3g} fps"
                    if movie.get("Frame time", "").startswith("--"):
                        movie["Frame time"] = f"{per_frame_s:.3g} s"
    except Exception as e:
        log.debug("deriving movie fps from time axis failed: %s", e)

    return subsections

"""
export_numpy.py — File → Export to NumPy: what a window shows, as .npz / .csv.

A strain map, an orientation map, a DPC field, a virtual image, a spectrum or
a line profile is a small result the user wants in a notebook, and the
``.zspy`` Save writes the whole signal tree for that. This writes one ``.npz``
holding every map and trace the window carries, keyed by a plain identifier
(``exx``, ``eyy``, ``Bx``, ``Virtual_Image_1_red``, ``Line_Profile``), plus
the calibrated axes and a ``meta_json`` record saying what each key is (its
label, its ``Signal.quantity``) and where it came from (the commit
provenance). ``np.load(path)`` is the whole reader. A ``.csv`` holds the
traces as columns against their axis, or one map as rows.

What goes in, in this order (a later source never overwrites an earlier key):

1. the array the window is showing, under the window's own key;
2. every other map node of the same tree, when the shown node is a map — a
   committed Strain / DPC / orientation tree is one window whose views are
   sibling nodes, and the file should hold all of them;
3. the chip views registered on the window (``spyde.actions.views``) — the
   IPF-X/Y/Z projections, the EBSD quality maps — which are figures, not nodes;
4. the raw fields of a result object on the tree (``dpc_result``,
   ``vector_orientation``, ``orientation_map``): quaternions, the fractional
   strain tensor, the un-rotated field.

A window on a DATASET (a scan with navigation axes, or anything lazy) exports
only the frame it shows — never the dataset. That is the CLAUDE.md memory-
safety rule; the ``.zspy`` Save is the door for the data itself.

A bare-figure window (the live Strain window, an IPF explorer) has no Plot; its
controller answers through an optional ``export_arrays()`` and, failing that,
the views registered on the window.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np

from spyde.backend.console import _sanitize_identifier

log = logging.getLogger(__name__)

#: Keys the archive itself uses: ``np.savez``'s own parameters, the record,
#: and the calibration. A label that slugs to one of these gets a suffix.
_RESERVED_KEYS = frozenset({"file", "allow_pickle", "meta_json", "x_axis", "y_axis"})

#: Symbols a map label uses that have no place in a numpy key. ``|B|`` is
#: handled by :func:`slug` before this table applies.
_TRANSLITERATE = {"ε": "e", "ω": "omega", "θ": "theta", "φ": "phi", "°": "deg",
                  "µ": "u", "μ": "u", "Å": "A", "—": " ", "–": " "}
_UNIT_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")
_ABS = re.compile(r"\|([^|]+)\|")


def slug(label: str) -> str:
    """A label as a numpy key one can type: ``"εxx (%)"`` → ``exx``,
    ``"|B| (mrad)"`` → ``abs_B``, ``"Virtual Image 1 (red)"`` →
    ``Virtual_Image_1_red``. The units are dropped only when the whole
    remainder is a unit parenthetical on a symbol; the meta record keeps the
    original label. The last step is the console's own identifier rule, so
    a title binds to the same name in a notebook as in the console."""
    text = str(label).strip()
    text = _ABS.sub(lambda m: f"abs_{m.group(1)}", text)
    if _UNIT_SUFFIX.search(text) and not _looks_like_a_name(text):
        text = _UNIT_SUFFIX.sub("", text)
    for symbol, ascii_form in _TRANSLITERATE.items():
        text = text.replace(symbol, ascii_form)
    text = _sanitize_identifier(text[:60], fallback="array")
    return text


def _looks_like_a_name(text: str) -> bool:
    """A parenthetical that distinguishes one thing from another — ``Virtual
    Image 1 (red)`` — is part of the name; one after a symbol — ``εxx (%)``,
    ``Bx (mrad)`` — is a unit. The test is whether what precedes it has a
    space in it."""
    head = _UNIT_SUFFIX.sub("", text)
    return " " in head.strip()


@dataclass
class Export:
    """What one window exports: ``arrays`` by key, ``labels`` recording each
    key's human label and quantity, and ``meta`` for the file-level record."""
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    labels: dict[str, dict] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    primary: str | None = None

    def add(self, key: str, array, *, label: str = "", quantity: str = "",
            source: str = "") -> str | None:
        """Add *array* under *key* (a slug of it), returning the key used, or
        None when *array* is not an array or the key is already taken. A key
        the archive itself uses is suffixed rather than refused."""
        if array is None or not hasattr(array, "__array__"):
            return None
        label = label or str(key)
        key = slug(key)
        if key in _RESERVED_KEYS:
            key = next(f"{key}_{n}" for n in range(2, 1000)
                       if f"{key}_{n}" not in self.arrays)
        if key in self.arrays:
            log.debug("export: %r already present, keeping the first", key)
            return None
        self.arrays[key] = np.asarray(array)
        self.labels[key] = {"label": str(label), "quantity": str(quantity or ""),
                            "source": source}
        return key


# ── Collecting ───────────────────────────────────────────────────────────────

def collect(session, plot=None, window_id=None) -> Export | None:
    """The :class:`Export` for *plot*, or for the bare-figure *window_id* when
    there is no plot. None when the window holds nothing exportable. A
    controller's ``export_arrays()`` maps a label to an array, or to an
    ``(array, quantity)`` pair; its failure propagates to the caller, which
    reports it — a window with a map on screen must never be reported as
    having nothing."""
    if plot is not None:
        return _collect_plot(session, plot)
    if window_id is None:
        return None
    export = Export()
    controller = _controller(session, window_id)
    exporter = getattr(controller, "export_arrays", None)
    if callable(exporter):
        for label, value in dict(exporter()).items():
            array, quantity = value if isinstance(value, tuple) else (value, "")
            key = export.add(label, array, label=label, quantity=quantity,
                             source="controller")
            if export.primary is None:
                export.primary = key
    _add_registered_views(export, window_id)
    export.meta.setdefault("title", str(getattr(controller, "title", "") or ""))
    return export if export.arrays else None


def _controller(session, window_id):
    lookup = getattr(session, "controller_by_window_id", None)
    if lookup is None:
        return None
    try:
        return lookup(int(window_id))
    except Exception:
        return None


def _collect_plot(session, plot) -> Export | None:
    export = Export()
    tree = getattr(plot, "signal_tree", None)
    signal = getattr(getattr(plot, "plot_state", None), "current_signal", None)
    node = tree.get_node(signal) if (tree is not None and signal is not None) else None
    shown = getattr(plot, "current_data", None)
    shown = np.asarray(shown) if hasattr(shown, "__array__") else None

    if node is not None and _is_map_or_trace(node.signal):
        # A result window: the shown node first (its painted array — an RGB
        # IPF map is painted over a zeros root), then every sibling map.
        export.primary = export.add(_node_label(node, tree), _painted_or(shown, node.signal),
                                    label=_node_label(node, tree),
                                    quantity=_quantity(node.signal), source="node")
        for other in tree.walk():
            if (other is node or getattr(other, "overlay", False)
                    or not _is_map_or_trace(other.signal)):
                continue
            export.add(_node_label(other, tree), other.signal.data,
                       label=_node_label(other, tree),
                       quantity=_quantity(other.signal), source="node")
        _add_axes(export, node.signal)
        export.meta["title"] = _title(tree.root_node.signal)
        # The chip views and the result object belong to the RESULT window.
        # A fit stamps its result on the source scan's tree too, and that
        # tree is shared with the scan's own windows and with every live
        # output cut from it: none of those is asking for the fit.
        _add_registered_views(export, getattr(plot, "window_id", None))
        _add_result_objects(export, tree)
        provenance = getattr(tree, "_commit_provenance", None)
        if provenance:
            export.meta["provenance"] = dict(provenance)
    elif shown is not None:
        # A window on a dataset, or a live output (a virtual image, a line
        # profile) whose signal is a placeholder outside its tree: the shown
        # frame or trace only.
        label = _artifact_name(session, getattr(plot, "window_id", None)) or "frame"
        export.primary = export.add(label, shown, label=label,
                                    quantity=_quantity(signal), source="displayed")
        if label == "frame":
            # A frame of the dataset: its own signal axes, and where it is.
            _add_axes(export, signal)
            index = _navigation_index(tree)
            if index is not None:
                export.meta["navigation_index"] = index
        elif tree is not None:
            # A virtual image is a picture over the SCAN, so it carries the
            # scan's navigation calibration, not its placeholder's pixels.
            _add_navigation_axes(export, tree.root_node.signal, shown.shape[:2])
        if tree is not None:
            export.meta["title"] = _title(tree.root_node.signal)
    return export if export.arrays else None


def _is_map_or_trace(signal) -> bool:
    """An in-memory image or trace: no navigation axes, 1-D, 2-D or RGB, not
    lazy. Anything else is a dataset, and a dataset is never exported here."""
    try:
        if getattr(signal, "_lazy", False):
            return False
        if signal.axes_manager.navigation_dimension != 0:
            return False
        data = signal.data
        return hasattr(data, "shape") and data.ndim in (1, 2, 3) and not _is_lazy_array(data)
    except Exception:
        return False


def _is_lazy_array(data) -> bool:
    return type(data).__module__.startswith("dask")


def _painted_or(shown, signal):
    """The array on screen when it is the node's picture (same height and
    width — the RGB paint of an IPF window), else the node's own data."""
    data = signal.data
    extent = min(data.ndim, 2)
    if shown is not None and shown.shape[:extent] == tuple(data.shape[:extent]):
        return shown
    return data


def _node_label(node, tree) -> str:
    """A child node is named for its view (``εyy``); the root carries the
    tree's title, so its quantity (``εxx (%)``) names it better when set."""
    if node is not tree.root_node:
        return str(node.name)
    quantity = _quantity(node.signal)
    return quantity or _title(node.signal)


def _quantity(signal) -> str:
    try:
        return str(signal.metadata.get_item("Signal.quantity", "") or "")
    except Exception:
        return ""


def _title(signal) -> str:
    try:
        return str(signal.metadata.get_item("General.title", "") or "")
    except Exception:
        return ""


def _add_axes(export: Export, signal) -> None:
    """The calibrated coordinate of every pixel column and row of a map or
    trace (its signal axes), so it replots on its own scale."""
    try:
        axes = list(signal.axes_manager.signal_axes)
    except Exception:
        return
    _put_axes(export, axes)


def _add_navigation_axes(export: Export, signal, shape) -> None:
    """The scan's navigation coordinates for a picture drawn over the scan
    (its height and width match the two spatial navigation axes)."""
    from spyde.actions.commit import spatial_navigation_axes
    axes = spatial_navigation_axes(signal)
    if len(axes) != 2:
        return
    x, y = axes
    if (int(y.size), int(x.size)) != tuple(int(n) for n in shape):
        return
    _put_axes(export, [x, y])


def _put_axes(export: Export, axes) -> None:
    """``x_axis`` / ``y_axis`` from up to two hyperspy axes, and their units
    as the plots show them: cleaned of LaTeX, blank where hyperspy's
    ``<undefined>`` sentinel stands in for none. The first writer wins."""
    from spyde.drawing.plots.plot import _clean_units
    if "x_axis" in export.arrays:
        return
    units = {}
    for name, axis in zip(("x", "y"), axes):
        try:
            export.arrays[f"{name}_axis"] = np.asarray(axis.axis, dtype=np.float64)
            raw = str(axis.units or "")
            units[name] = "" if raw == "<undefined>" else _clean_units(raw)
        except Exception as e:
            log.debug("export: reading the %s axis failed: %s", name, e)
    if units:
        export.meta["axis_units"] = units


def _navigation_index(tree) -> list | None:
    """Where the exported frame sits in the scan, in DATA order (row first),
    read from the navigator's selector: the axes manager's own index is never
    moved by a navigator drag. The position committed by the last move, or
    the widget's selection before any. An integrating region reports its
    centre."""
    from spyde.drawing.update_functions import _prepare_nav_indices
    manager = getattr(tree, "navigator_plot_manager", None)
    for selector in getattr(manager, "all_navigation_selectors", None) or ():
        try:
            indices = getattr(selector, "current_indices", None)
            if indices is None:
                indices = selector.get_selected_indices()
            prepared = _prepare_nav_indices(tree.root, indices, integrating=False)
        except Exception as e:
            log.debug("export: reading the navigation index failed: %s", e)
            continue
        if prepared is not None:
            return [int(v) for v in np.atleast_1d(np.asarray(prepared)).ravel()]
    return None


def _artifact_name(session, window_id) -> str | None:
    """The live output named for *window_id* — ``Virtual Image 1 (red)``."""
    if window_id is None:
        return None
    artifacts = getattr(session, "_action_artifacts", None) or {}
    for (_source, name), artifact in artifacts.items():
        if window_id in (artifact.get("out_wids") or []):
            return str(name)
    return None


def _add_registered_views(export: Export, window_id) -> None:
    if window_id is None:
        return
    from spyde.actions.views import _VIEW_DATA
    entry = _VIEW_DATA.get(int(window_id))
    if not entry:
        return
    labels = entry.get("value_labels") or {}
    for label in entry.get("order", []):
        image = entry["images"].get(label)
        if label.startswith("__"):
            continue
        key = export.add(label, image, label=label, quantity=labels.get(label, ""),
                         source="view")
        if export.primary is None:
            export.primary = key
    axes = entry.get("axes")
    if axes and "x_axis" not in export.arrays:
        # commit.navigation_extent's (x, y, units): already cleaned.
        try:
            x, y, units = axes
            export.arrays["x_axis"] = np.asarray(x, dtype=np.float64)
            export.arrays["y_axis"] = np.asarray(y, dtype=np.float64)
            export.meta["axis_units"] = {"x": str(units), "y": str(units)}
        except Exception as e:
            log.debug("export: view axes unusable: %s", e)


#: The result objects a tree can carry and the record their non-array
#: fields go under. Arrays are exported by field name; everything else that
#: serialises lands in ``meta_json[<record>]``.
_RESULT_ATTRIBUTES = ("dpc_result", "vector_orientation", "orientation_map")


def _add_result_objects(export: Export, tree) -> None:
    for attribute in _RESULT_ATTRIBUTES:
        result = getattr(tree, attribute, None)
        if result is None or not is_dataclass(result):
            continue
        record: dict[str, Any] = {}
        for member in fields(result):
            if member.name.startswith("_"):
                continue
            value = getattr(result, member.name, None)
            if hasattr(value, "__array__") and getattr(value, "ndim", 0) > 0:
                export.add(member.name, value, label=member.name, source=attribute)
            elif value is not None:
                record[member.name] = _jsonable(value)
        export.meta[attribute] = record


def _jsonable(value):
    """*value* as JSON can hold it: dataclasses and arrays become their
    fields and lists, anything else its string."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in fields(value)}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# ── Writing ──────────────────────────────────────────────────────────────────

def write(export: Export, path: str) -> str:
    """Write *export* to *path* and return the path written.

    ``.npz`` holds every array plus ``meta_json`` (a UTF-8 byte array, the
    convention :meth:`SpyDEOrientationMap.save` uses); ``.npy`` holds the
    primary array alone; ``.csv`` is :func:`write_csv`. No extension means
    ``.npz``."""
    extension = os.path.splitext(path)[1].lower()
    key = export.primary or next(iter(export.arrays))
    if extension == ".npy":
        np.save(path, export.arrays[key])
        return path
    if extension == ".csv":
        return write_csv(export, path)
    if extension != ".npz":
        path = path + ".npz"
    meta = dict(export.meta)
    meta["arrays"] = export.labels
    meta["primary"] = export.primary
    payload = json.dumps(meta, default=str).encode("utf-8")
    np.savez_compressed(path, meta_json=np.frombuffer(payload, dtype=np.uint8),
                        **export.arrays)
    return path


def read_meta(path: str) -> dict:
    """The ``meta_json`` record of an exported ``.npz``."""
    with np.load(path) as archive:
        return json.loads(bytes(archive["meta_json"]).decode("utf-8"))


def write_csv(export: Export, path: str) -> str:
    """A ``.csv`` for a spreadsheet: when the primary is a trace, one column
    per trace of its length (a spectrum, a line profile), the calibrated axis
    first and the keys as the header row; when it is a map, the map alone,
    one row per image row, its key in a comment line. A picture (RGB) has no
    CSV form; the ``.npz`` holds it."""
    key = export.primary or next(iter(export.arrays))
    primary = export.arrays[key]
    if primary.ndim == 1:
        columns, header = [], []
        axis = export.arrays.get("x_axis")
        if axis is not None and axis.shape == primary.shape:
            units = (export.meta.get("axis_units") or {}).get("x", "")
            columns.append(axis)
            header.append(f"x ({units})" if units else "x")
        for name, array in export.arrays.items():
            if name.endswith("_axis") or array.ndim != 1 or array.shape != primary.shape:
                continue
            columns.append(array)
            header.append(name)
        np.savetxt(path, np.column_stack(columns), delimiter=",",
                   header=",".join(header), comments="")
        return path
    if primary.ndim != 2:
        raise ValueError(f"{key} is a picture {primary.shape}; save it as .npz")
    label = export.labels.get(key, {})
    np.savetxt(path, primary, delimiter=",",
               header=f"{key}: {label.get('quantity') or label.get('label') or key}, "
                      f"{primary.shape[0]} rows x {primary.shape[1]} columns")
    return path

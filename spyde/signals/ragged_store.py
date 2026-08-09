"""
RaggedStore — ragged per-navigation-position column storage over a scan grid.

The shared base for SpyDE's genuinely ragged result families (diffraction
vectors; particles), extracted from ``SpyDEDiffractionVectors``. It owns the
NAV-SPACE storage model only — signal-space (detector axes, kernel radii,
rendering) stays with the owning subclass.

Storage model (the whole contract)
----------------------------------
offsets : (n_positions + 1,) int64 CSR row pointers over the C-ordered flat
    nav grid; ``offsets[p]:offsets[p+1]`` is position p's rows.
columns : one flat 1-D array per schema column, all length n_rows, dtype per
    schema. MAY be strided views into a single packed (n_rows, n_cols) array
    (``from_packed``) — the packed layout is then the subclass's public ABI;
    the base never copies or reorders a packed backing.
full_nav_shape : all nav dims, outermost first, rank >= 1.
nav_axes : axis records duck-typing .scale/.offset/.size/.units/.name; scan
    calibration only.

Because the grid is rectangular, every OUTER level of a multi-level CSR index
is a zero-copy strided view of the leaf level (``leaf[::prod(inner dims)]``) —
so the base stores ONE offsets array and derives the rest
(:func:`derive_levels`).

Lifecycle
---------
STRUCTURE-FROZEN after ``finalize()``: row count/order, offsets and the column
set never change; cell VALUES may be overwritten in place by the owning
subclass (cache invalidation is the subclass's problem, as today).

Threading
---------
``append_batch()`` is the only mutating call (mutex-guarded); every read
accessor is valid only after ``finalize()`` and thereafter lock-free. An
unfinalized store carries a ``threading.Lock`` and is therefore not picklable;
a finalized one drops it and pickles like a plain container.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

ColumnDef = Tuple[str, str]   # (name, numpy dtype string), e.g. ("kx", "f4")


@dataclass
class _AxisLite:
    """Minimal axis record so results loaded from disk can be rendered
    without HyperSpy axes objects (duck-types .scale/.offset/.size/.units/.name)."""
    scale: float = 1.0
    offset: float = 0.0
    size: int = 0
    units: str = ""
    name: str = ""


# ── grid arithmetic ──────────────────────────────────────────────────────────

def _grid_strides(full_nav_shape: Sequence[int]) -> np.ndarray:
    """int64 C-order strides of the nav grid, outermost first (innermost = 1)."""
    n = len(full_nav_shape)
    strides = np.ones(n, dtype=np.int64)
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * int(full_nav_shape[i + 1])
    return strides


def flat_leaf_index(index_arrays: Sequence[np.ndarray],
                    full_nav_shape: Sequence[int]) -> np.ndarray:
    """Per-row flat leaf position from the integer index columns.

    ``index_arrays`` holds one 1-D array per nav dim, outermost first. Values
    are cast to int64 TRANSIENTLY (stored columns are never cast — the f4
    packed ABI stays f4); negative values clamp to 0, which is what lets a
    subclass sentinel (the vectors' ``time = -1`` for "no axis") classify as
    position 0 rather than corrupt the bincount.
    """
    n_rows = int(len(index_arrays[0])) if index_arrays else 0
    strides = _grid_strides(full_nav_shape)
    flat = np.zeros(n_rows, dtype=np.int64)
    for vals, stride in zip(index_arrays, strides):
        v = np.asarray(vals).astype(np.int64)
        v = np.where(v < 0, np.int64(0), v)
        flat += v * int(stride)
    return flat


def build_leaf_offsets(index_arrays: Sequence[np.ndarray],
                       full_nav_shape: Sequence[int]) -> np.ndarray:
    """(n_positions + 1,) int64 CSR row pointers by bincount — O(n_rows), no
    sort, no copy. Rows must already be sorted by flat position for the
    resulting slices to be meaningful (counts are order-independent)."""
    n_positions = int(np.prod(np.asarray(full_nav_shape, dtype=np.int64)))
    flat = flat_leaf_index(index_arrays, full_nav_shape)
    counts = np.bincount(flat, minlength=n_positions).astype(np.int64)
    if counts.shape[0] != n_positions:
        raise ValueError(
            f"index columns address position {int(flat.max())} outside the "
            f"{tuple(full_nav_shape)} nav grid")
    leaf = np.zeros(n_positions + 1, dtype=np.int64)
    np.cumsum(counts, out=leaf[1:])
    return leaf


def derive_levels(leaf: np.ndarray,
                  full_nav_shape: Sequence[int]) -> List[np.ndarray]:
    """Multi-level CSR offsets, outermost first, derived from the leaf level.

    The rectangular-grid stride identity: level k is ``leaf[::prod(inner
    dims)]`` — a ZERO-COPY strided view, element-for-element identical to the
    historically materialised outer levels. The innermost entry is the leaf
    array itself (identity, not a view), preserving the 4-D
    ``offsets is nav_offsets[-1]`` alias downstream."""
    strides = _grid_strides(full_nav_shape)
    return [leaf[:: int(s)] for s in strides[:-1]] + [leaf]


class RaggedStore:
    """Ragged per-navigation-position column store over a rectangular scan grid.

    See the module docstring for the storage model and lifecycle contract.

    Construction paths:

    * ``RaggedStore(columns, offsets, full_nav_shape, ...)`` — FINALIZED from a
      dict of 1-D column arrays + valid offsets (zero-copy; arrays adopted).
    * ``from_packed(packed, full_nav_shape, index_columns=...)`` — FINALIZED
      from an already-sorted packed (n_rows, n_cols) single-dtype array.
    * ``streaming(...)`` then ``append_batch()`` × N then ``finalize()`` — THE
      one seam for chunked computes; batches may arrive in any chunk order.

    Subclasses declare ``columns_schema`` (ORDER load-bearing — for a packed
    backing it IS the column order) and may bump ``format_version`` under the
    append-only-columns rule. Exactly two persistence extension points exist:
    ``_save_extra`` / ``_load_extra``. Subclasses with their own constructor
    ABI (dataclass fields) keep their constructors and wire the base state over
    the same memory (see ``SpyDEDiffractionVectors.__post_init__``).
    """

    # ORDER load-bearing; append-only + bump format_version on change.
    columns_schema: ClassVar[Tuple[ColumnDef, ...]] = ()
    format_version: ClassVar[int] = 1

    def __init__(self, columns: Dict[str, np.ndarray], offsets: np.ndarray,
                 full_nav_shape: Sequence[int], *,
                 index_columns: Sequence[str] = (),
                 nav_axes: Sequence[object] = (),
                 params: Optional[dict] = None,
                 provenance: Optional[dict] = None):
        """Construct FINALIZED from dict-of-1D-arrays + valid offsets."""
        names = self._schema_names()
        if not names:
            raise TypeError(
                f"{type(self).__name__} declares no columns_schema")
        full_nav_shape = tuple(int(s) for s in full_nav_shape)
        if not full_nav_shape:
            raise ValueError("full_nav_shape must have rank >= 1")
        offsets = np.asarray(offsets, dtype=np.int64)
        n_positions = int(np.prod(np.asarray(full_nav_shape, dtype=np.int64)))
        if offsets.shape != (n_positions + 1,):
            raise ValueError(
                f"offsets shape {offsets.shape} != ({n_positions + 1},) for "
                f"nav grid {full_nav_shape}")
        n_rows = int(offsets[-1])
        cols: Dict[str, np.ndarray] = {}
        given = dict(columns)
        if set(given) != set(names):
            raise ValueError(
                f"columns {sorted(given)} do not match schema {list(names)}")
        for name in names:                      # schema order, zero-copy
            arr = np.asarray(given[name])
            if arr.ndim != 1 or arr.shape[0] != n_rows:
                raise ValueError(
                    f"column {name!r} must be 1-D of length {n_rows}, "
                    f"got shape {arr.shape}")
            cols[name] = arr
        self.full_nav_shape = full_nav_shape
        self.nav_axes = list(nav_axes or [])
        self.params = dict(params) if params else {}
        self.provenance = provenance
        self.offsets = offsets
        self._wire(columns=cols, packed=None, offsets=offsets, levels=None,
                   index_columns=self._check_index_columns(index_columns))

    # ── schema helpers ───────────────────────────────────────────────────────

    @classmethod
    def _schema_names(cls) -> Tuple[str, ...]:
        return tuple(n for n, _ in cls.columns_schema)

    @classmethod
    def _check_index_columns(cls, index_columns: Sequence[str]) -> Tuple[str, ...]:
        idx = tuple(index_columns)
        names = cls._schema_names()
        for c in idx:
            if c not in names:
                raise ValueError(f"index column {c!r} not in schema {list(names)}")
        return idx

    def _wire(self, *, columns: Optional[Dict[str, np.ndarray]],
              packed: Optional[np.ndarray], offsets: np.ndarray,
              levels: Optional[List[np.ndarray]],
              index_columns: Sequence[str]) -> None:
        """Install the FINALIZED read state. ``levels=None`` derives the outer
        levels as zero-copy strided views of *offsets* (the leaf)."""
        self._columns = columns
        self._packed = packed
        self._offsets = offsets
        self._levels = (levels if levels is not None
                        else derive_levels(offsets, self.full_nav_shape))
        self._index_columns = tuple(index_columns)
        self._finalized = True
        self._staged = None
        self._staged_packed = None
        self._staged_rows = 0
        self._append_lock = None     # locks don't pickle; frozen stores need none

    def _require_finalized(self) -> None:
        if not getattr(self, "_finalized", False):
            raise RuntimeError(
                f"{type(self).__name__} is not finalized — call finalize() "
                "before reading")

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_packed(cls, packed: np.ndarray, full_nav_shape: Sequence[int], *,
                    index_columns: Sequence[str] = (),
                    nav_axes: Sequence[object] = (),
                    params: Optional[dict] = None,
                    provenance: Optional[dict] = None) -> "RaggedStore":
        """FINALIZED from a packed (n_rows, n_cols) single-dtype array already
        sorted outermost-nav-first. Columns are zero-copy strided views;
        offsets by bincount over the index columns — O(n_rows), no sort, no
        copy (the packed array is ADOPTED, never reordered).

        Only valid for classes whose usable state is pure base state — a
        subclass with its own constructor ABI (dataclass fields) must build
        through its own constructors instead.
        """
        names = cls._schema_names()
        if not names:
            raise TypeError(f"{cls.__name__} declares no columns_schema")
        packed = np.asarray(packed)
        if packed.ndim != 2 or packed.shape[1] != len(names):
            raise ValueError(
                f"packed must be (n_rows, {len(names)}); got {packed.shape}")
        inst = cls.__new__(cls)
        inst.full_nav_shape = tuple(int(s) for s in full_nav_shape)
        if not inst.full_nav_shape:
            raise ValueError("full_nav_shape must have rank >= 1")
        inst.nav_axes = list(nav_axes or [])
        inst.params = dict(params) if params else {}
        inst.provenance = provenance
        idx = cls._check_index_columns(index_columns)
        if len(idx) != len(inst.full_nav_shape):
            raise ValueError(
                f"index_columns {idx} must name one column per nav dim "
                f"{inst.full_nav_shape}")
        index_arrays = [packed[:, names.index(c)] for c in idx]
        leaf = build_leaf_offsets(index_arrays, inst.full_nav_shape)
        inst.offsets = leaf
        inst._wire(columns=None, packed=packed, offsets=leaf, levels=None,
                   index_columns=idx)
        return inst

    # ── streaming fill: THE one seam for chunked computes ────────────────────

    @classmethod
    def streaming(cls, full_nav_shape: Sequence[int], *,
                  index_columns: Sequence[str],
                  nav_axes: Sequence[object] = (),
                  params: Optional[dict] = None,
                  provenance: Optional[dict] = None) -> "RaggedStore":
        """UNFINALIZED store accepting ``append_batch()``; reads raise until
        ``finalize()``."""
        names = cls._schema_names()
        if not names:
            raise TypeError(f"{cls.__name__} declares no columns_schema")
        inst = cls.__new__(cls)
        inst.full_nav_shape = tuple(int(s) for s in full_nav_shape)
        if not inst.full_nav_shape:
            raise ValueError("full_nav_shape must have rank >= 1")
        idx = cls._check_index_columns(index_columns)
        if len(idx) != len(inst.full_nav_shape):
            raise ValueError(
                f"index_columns {idx} must name one column per nav dim "
                f"{inst.full_nav_shape}")
        inst.nav_axes = list(nav_axes or [])
        inst.params = dict(params) if params else {}
        inst.provenance = provenance
        inst.offsets = None
        inst._columns = None
        inst._packed = None
        inst._offsets = None
        inst._levels = None
        inst._index_columns = idx
        inst._finalized = False
        inst._staged = []            # dict-of-columns batches
        inst._staged_packed = []     # packed (m, n_cols) blocks
        inst._staged_rows = 0
        inst._append_lock = threading.Lock()
        return inst

    def append_batch(self, batch) -> int:
        """Stage rows; returns the total number of rows staged so far.

        *batch* is either an ``(m, n_cols)`` packed block (single-dtype
        schema) or a dict of equal-length 1-D arrays covering EVERY schema
        column (the integer index columns included). Batch kinds cannot be
        mixed within one store. Thread-safe; callable from dask done-callbacks
        in ANY chunk order. O(1) list append — no offsets maintenance during
        the fill (deliberate: leaf offsets can't be maintained incrementally
        under out-of-order chunks; batch-then-build matches the historical
        one-shot build).
        """
        lock = getattr(self, "_append_lock", None)
        if getattr(self, "_finalized", False) or lock is None:
            raise RuntimeError(
                "append_batch() after finalize() — the structure is frozen")
        names = self._schema_names()
        with lock:
            if self._finalized:      # finalized while we waited for the lock
                raise RuntimeError(
                    "append_batch() after finalize() — the structure is frozen")
            if isinstance(batch, dict):
                if self._staged_packed:
                    raise TypeError("cannot mix dict and packed batches")
                if set(batch) != set(names):
                    raise ValueError(
                        f"batch columns {sorted(batch)} do not match schema "
                        f"{list(names)}")
                cols = {n: np.asarray(batch[n]) for n in names}
                lengths = {c.shape[0] for c in cols.values()}
                if any(c.ndim != 1 for c in cols.values()) or len(lengths) > 1:
                    raise ValueError("batch columns must be equal-length 1-D arrays")
                m = lengths.pop() if lengths else 0
                self._staged.append(cols)
            else:
                if self._staged:
                    raise TypeError("cannot mix dict and packed batches")
                block = np.asarray(batch)
                if block.ndim != 2 or block.shape[1] != len(names):
                    raise ValueError(
                        f"packed batch must be (m, {len(names)}); got {block.shape}")
                m = int(block.shape[0])
                self._staged_packed.append(block)
            self._staged_rows += int(m)
            return self._staged_rows

    def finalize(self) -> "RaggedStore":
        """Concatenate, STABLE-sort by flat nav position, build offsets, freeze.

        Idempotent. The O(n) already-sorted check keeps a sorted input
        byte-identical — and, for a SINGLE sorted packed batch, object-
        identical (the block is adopted, not copied), which is what keeps the
        ``from_arrays``/orchestrate path exactly as it was. Row order within a
        position = batch arrival order then in-batch order (stable) — what
        keeps overlay spot order and the report embed's CSR-contiguity
        assumption deterministic.
        """
        lock = getattr(self, "_append_lock", None)
        if getattr(self, "_finalized", False) or lock is None:
            return self
        with lock:
            if self._finalized:      # lost the race to another finalizer
                return self
            names = self._schema_names()
            n_positions = int(np.prod(
                np.asarray(self.full_nav_shape, dtype=np.int64)))
            if self._staged_packed:
                blocks = self._staged_packed
                packed = blocks[0] if len(blocks) == 1 else np.concatenate(blocks)
                index_arrays = [packed[:, names.index(c)]
                                for c in self._index_columns]
                flat = flat_leaf_index(index_arrays, self.full_nav_shape)
                if flat.size and np.any(np.diff(flat) < 0):
                    order = np.argsort(flat, kind="stable")
                    packed = packed[order]
                    flat = flat[order]
                leaf = self._leaf_from_sorted(flat, n_positions)
                columns = None
            else:
                cols: Dict[str, np.ndarray] = {}
                for name, dt in self.columns_schema:
                    parts = [b[name] for b in self._staged]
                    if len(parts) == 1:
                        cols[name] = parts[0]
                    elif parts:
                        cols[name] = np.concatenate(parts)
                    else:
                        cols[name] = np.zeros(0, dtype=np.dtype(dt))
                index_arrays = [cols[c] for c in self._index_columns]
                flat = flat_leaf_index(index_arrays, self.full_nav_shape)
                if flat.size and np.any(np.diff(flat) < 0):
                    order = np.argsort(flat, kind="stable")
                    cols = {n: c[order] for n, c in cols.items()}
                    flat = flat[order]
                leaf = self._leaf_from_sorted(flat, n_positions)
                columns = cols
                packed = None
            self.offsets = leaf
            self._wire(columns=columns, packed=packed, offsets=leaf,
                       levels=None, index_columns=self._index_columns)
        return self

    @staticmethod
    def _leaf_from_sorted(flat: np.ndarray, n_positions: int) -> np.ndarray:
        counts = np.bincount(flat, minlength=n_positions).astype(np.int64)
        if counts.shape[0] != n_positions:
            raise ValueError(
                f"index columns address position {int(flat.max())} outside "
                f"the {n_positions}-position nav grid")
        leaf = np.zeros(n_positions + 1, dtype=np.int64)
        np.cumsum(counts, out=leaf[1:])
        return leaf

    # ── index arithmetic (shared with subclasses; hot path — no guards) ──────

    def _flat_pos(self, nav_indices: tuple) -> int:
        """Convert nav indices to a flat leaf position using grid strides."""
        pos = 0
        stride = 1
        for idx, dim_size in zip(reversed(nav_indices),
                                 reversed(self.full_nav_shape)):
            pos += int(idx) * stride
            stride *= dim_size
        return pos

    def _row_range(self, nav_indices: tuple) -> Tuple[int, int]:
        """Half-open row range for one FULL nav index. O(n_dims) arithmetic."""
        p = self._flat_pos(nav_indices)
        lev = self._levels[-1]
        return int(lev[p]), int(lev[p + 1])

    def _prefix_row_range(self, nav_indices: tuple) -> Tuple[int, int]:
        """Half-open row range for a FULL or PREFIX nav index (partial indexing
        reads the outer CSR level — O(1) via the stride identity). Exactly the
        historical ``slice_at`` arithmetic, degenerate cases included."""
        n = len(nav_indices)
        if n == len(self.full_nav_shape):
            return self._row_range(nav_indices)
        level_shape = self.full_nav_shape[:n]
        pos = 0
        stride = 1
        for idx, dim_size in zip(reversed(nav_indices), reversed(level_shape)):
            pos += int(idx) * stride
            stride *= dim_size
        lev = self._levels[n - 1]
        return int(lev[pos]), int(lev[pos + 1])

    def _slice_flat(self, nav_indices: tuple) -> np.ndarray:
        """Packed rows for one FULL nav index (packed backing only)."""
        s, e = self._row_range(nav_indices)
        return self._packed[s:e]

    # ── access ───────────────────────────────────────────────────────────────

    def offset_levels(self) -> List[np.ndarray]:
        """Multi-level CSR offsets, outermost first, leaf last. For base-built
        stores the outer levels are zero-copy strided views of the leaf; a
        subclass wired over a stored list returns that list's entries verbatim."""
        self._require_finalized()
        return list(self._levels)

    def column(self, name: str) -> np.ndarray:
        """The full flat column — zero-copy (a strided view for a packed
        backing, the adopted array otherwise)."""
        self._require_finalized()
        if self._columns is not None:
            return self._columns[name]
        names = self._schema_names()
        try:
            i = names.index(name)
        except ValueError:
            raise KeyError(name) from None
        return self._packed[:, i]

    def at(self, *nav_index: int) -> Dict[str, np.ndarray]:
        """Zero-copy column views for ONE full nav index, in schema order.

        Subclasses with a pinned packed return (the vectors' ``(N, 6)`` slice)
        override this; the base contract is the dict form."""
        self._require_finalized()
        if len(nav_index) != len(self.full_nav_shape):
            raise ValueError(
                f"at() needs {len(self.full_nav_shape)} indices, "
                f"got {len(nav_index)}")
        s, e = self._row_range(nav_index)
        return {name: self.column(name)[s:e] for name in self._schema_names()}

    def slice_at(self, *nav_prefix: int) -> Dict[str, np.ndarray]:
        """All rows under an outer-index prefix, O(1) via the stride identity.
        Returns the dict-of-column-views form (packed subclasses override)."""
        self._require_finalized()
        s, e = self._prefix_row_range(nav_prefix)
        return {name: self.column(name)[s:e] for name in self._schema_names()}

    def flatten(self) -> np.ndarray:
        """The full packed (n_rows, n_cols) buffer (packed backing only)."""
        self._require_finalized()
        if self._packed is None:
            raise TypeError("flatten() requires a packed backing")
        return self._packed

    def counts(self) -> np.ndarray:
        """(n_positions,) int64 rows per leaf position — np.diff over the leaf
        offsets. O(positions), never O(rows)."""
        self._require_finalized()
        return np.diff(self._levels[-1])

    def count_map(self) -> np.ndarray:
        """(full_nav_shape) int64 rows per position. O(positions) via
        ``np.diff(offsets)`` — NOT implemented as ``map(None, 'count')``
        because count_map sits on the progressive-fill hot path; their
        equivalence is pinned by test instead."""
        return self.counts().reshape(self.full_nav_shape)

    def map(self, column: Optional[str], reducer="mean", *,
            fill: float = np.nan) -> np.ndarray:
        """(full_nav_shape) per-position reduction of a column.

        ``reducer`` is one of ``'sum'|'mean'|'max'|'min'|'median'|'std'|
        'count'`` or a callable ``(rows_i,) -> scalar``. ``'count'`` ignores
        *column* (``map(None, 'count') == count_map()`` — the pinned law) and
        returns int64; ``'sum'`` returns float64 with empty positions 0; every
        other reducer returns float64 with empty positions set to *fill*
        (NaN — a frame with no rows has no mean; zero would plot a fake
        event). 'median'/'std'/callable loop over non-empty positions in
        Python — O(positions) calls; the array reducers are vectorised.
        """
        self._require_finalized()
        counts = self.counts()
        n_positions = counts.shape[0]
        if reducer == "count":
            return counts.reshape(self.full_nav_shape)
        col = np.asarray(self.column(column))
        leaf = np.asarray(self._levels[-1])
        nonempty = counts > 0
        if reducer in ("sum", "mean"):
            pos_ids = np.repeat(np.arange(n_positions, dtype=np.int64), counts)
            sums = np.bincount(pos_ids, weights=col.astype(np.float64),
                               minlength=n_positions)
            if reducer == "sum":
                return sums.reshape(self.full_nav_shape)
            out = np.full(n_positions, fill, dtype=np.float64)
            out[nonempty] = sums[nonempty] / counts[nonempty]
            return out.reshape(self.full_nav_shape)
        if reducer in ("max", "min"):
            out = np.full(n_positions, fill, dtype=np.float64)
            if nonempty.any():
                # reduceat over the STARTS of non-empty positions only: empty
                # positions contribute no rows, so consecutive non-empty starts
                # delimit exactly one position's rows (reduceat's empty-segment
                # misbehaviour never comes into play).
                starts = leaf[:-1][nonempty]
                ufunc = np.maximum if reducer == "max" else np.minimum
                out[nonempty] = ufunc.reduceat(col.astype(np.float64), starts)
            return out.reshape(self.full_nav_shape)
        if reducer in ("median", "std") or callable(reducer):
            fn = {"median": np.median, "std": np.std}.get(reducer, reducer)
            out = np.full(n_positions, fill, dtype=np.float64)
            for p in np.nonzero(nonempty)[0]:
                out[p] = fn(col[leaf[p]:leaf[p + 1]])
            return out.reshape(self.full_nav_shape)
        raise ValueError(f"unknown reducer {reducer!r}")

    # ── persistence (versioned from day one; append-only columns) ────────────

    def _save_extra(self) -> Tuple[Dict[str, np.ndarray], dict]:
        """Subclass extension point: extra arrays + JSON-safe extra meta to
        persist alongside the base payload."""
        return {}, {}

    @classmethod
    def _load_extra(cls, arrays: Dict[str, np.ndarray], meta: dict) -> dict:
        """Subclass extension point: attributes (name -> value) to set on the
        loaded instance, built from what ``_save_extra`` persisted."""
        return {}

    def save(self, path: str) -> None:
        """Save to a compressed ``.npz`` — versioned, self-describing (schema +
        index columns + nav grid/axes in ``meta_json``), columns stored one
        array each so the append-only rule can pad older files on load."""
        self._require_finalized()
        import json
        arrays: Dict[str, np.ndarray] = {
            "offsets": np.ascontiguousarray(self._levels[-1]),
        }
        for name in self._schema_names():
            arrays[f"col_{name}"] = np.ascontiguousarray(self.column(name))
        extra_arrays, extra_meta = self._save_extra()
        for k, v in (extra_arrays or {}).items():
            arrays[f"extra_{k}"] = np.asarray(v)
        axes_meta = [
            dict(scale=float(ax.scale), offset=float(ax.offset),
                 size=int(ax.size),
                 units=str(getattr(ax, "units", "") or ""),
                 name=str(getattr(ax, "name", "") or ""))
            for ax in (self.nav_axes or [])
        ]
        meta = {
            "format_version": int(self.format_version),
            "class": type(self).__name__,
            "columns": [[n, d] for n, d in self.columns_schema],
            "index_columns": list(self._index_columns),
            "full_nav_shape": [int(s) for s in self.full_nav_shape],
            "nav_axes": axes_meta,
            "params": self.params or {},
            "provenance": self.provenance,
            "extra": extra_meta or {},
        }
        np.savez_compressed(
            path,
            meta_json=np.frombuffer(
                json.dumps(meta, default=str).encode("utf-8"), dtype=np.uint8),
            **arrays,
        )

    @classmethod
    def load(cls, path: str) -> "RaggedStore":
        """Load a store saved with :meth:`save`.

        An older file whose columns are a PREFIX of ``cls.columns_schema`` is
        padded with zero-filled arrays for the missing columns (the proven
        append-only rule); a file whose ``format_version`` exceeds the class's
        is rejected. nav_axes come back as :class:`_AxisLite` records.
        """
        import json
        with np.load(path) as z:
            meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
            file_version = int(meta.get("format_version", 0))
            if file_version > cls.format_version:
                raise ValueError(
                    f"{path}: format_version {file_version} is newer than "
                    f"{cls.__name__}.format_version {cls.format_version}")
            file_names = [n for n, _ in
                          (tuple(c) for c in meta.get("columns", []))]
            names = list(cls._schema_names())
            if file_names != names[:len(file_names)]:
                raise ValueError(
                    f"{path}: saved columns {file_names} are not a prefix of "
                    f"the {cls.__name__} schema {names} (append-only rule)")
            offsets = np.asarray(z["offsets"], dtype=np.int64)
            n_rows = int(offsets[-1]) if offsets.size else 0
            cols: Dict[str, np.ndarray] = {}
            for name, dt in cls.columns_schema:
                key = f"col_{name}"
                if key in z.files:
                    cols[name] = np.asarray(z[key])
                else:
                    cols[name] = np.zeros(n_rows, dtype=np.dtype(dt))
            extra_arrays = {k[len("extra_"):]: np.asarray(z[k])
                            for k in z.files if k.startswith("extra_")}
        nav_axes = [_AxisLite(**a) for a in meta.get("nav_axes", [])]
        inst = cls(
            cols, offsets, tuple(int(s) for s in meta["full_nav_shape"]),
            index_columns=tuple(meta.get("index_columns", ()) or ()),
            nav_axes=nav_axes,
            params=meta.get("params") or {},
            provenance=meta.get("provenance"),
        )
        for k, v in (cls._load_extra(extra_arrays,
                                     meta.get("extra", {}) or {}) or {}).items():
            setattr(inst, k, v)
        return inst

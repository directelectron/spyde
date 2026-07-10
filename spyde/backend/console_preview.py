"""
console_preview.py — the math-console live *preview* evaluator.

The renderer's console bar debounces a "preview" of the expression the user is
typing (``s1 > 100``, ``s1.data``, ``mask & beam``) and asks the backend to show
what ONE frame of that expression looks like AT the referenced signal's current
navigator position — a tiny thumbnail / sparkline / scalar — WITHOUT computing
the whole (possibly hundreds-of-GB) dataset.

Everything here runs on the console thread (``ConsoleSession._do_preview`` calls
``evaluate_preview``), so the namespace stays single-threaded exactly like every
other console verb (see console.py's threading note). numpy / dask are imported
INSIDE the functions — the console thread's ``_init_namespace`` has already paid
the heavy-import cost, so these are cache hits, but keeping them local means this
module imports cheaply even before hyperspy is warm.

Design — three layers of RAM safety (the memory-safety rule is a HARD invariant,
see CLAUDE.md "Never Materialise Large Datasets"):

1. **AST tier gate.** An ``auto`` (typing-debounced) preview only runs for a
   *pure, side-effect-free* expression whitelist (comparisons, arithmetic,
   subscripts, attribute access…). Anything with a call (``s1.sum()``,
   ``np.log(s1)``) needs an explicit Ctrl+Enter (``auto=False``), because a call
   can trigger arbitrary compute. ``is_auto_safe`` is that whitelist.

2. **Nav-slice BEFORE compute.** We ``eval`` the expression to a LAZY value
   (dask/hyperspy graphs are cheap to build), then slice ONE navigator frame at
   the referenced signal's live cursor position — so the graph we actually
   compute spans a single frame, never the full nav space.

3. **Cost guard on the CULLED graph, fail-closed.** Even one frame of a derived
   view can pull many source chunks. Before computing we cull the sliced graph to
   its own keys and estimate how many source chunks / bytes it touches; over the
   caps we refuse ("too expensive" / "result too large"). ANY failure in the
   probe refuses too (fail closed) — we would rather show nothing than risk
   materialising the dataset.

The compute itself is ``scheduler="synchronous"`` (same as the live nav read's
``_direct_read_frame``), so it needs no Dask cluster and blocks only the console
thread. The rendered payload is small: a strided 2-D uint8 thumbnail (base64), a
strided 1-D sparkline, or a scalar repr. Previews NEVER register out<N>/assign
chips and NEVER emit ``console_result`` — they are display-only (see
``ConsoleSession._do_preview``).
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io

# ── caps (all fail-closed — over any of these, refuse) ───────────────────────
MAX_SOURCE_CHUNKS = 8               # a preview may touch at most this many source chunks
MAX_EST_BYTES     = 256 * 2**20     # …and at most this many bytes of source data
MAX_GRAPH_TASKS   = 2048            # a graph with NO attributable source keys is bounded here
MAX_RESULT_BYTES  = 64 * 2**20      # the sliced result itself must be smaller than this
THUMB_MAX = 192                     # 2-D thumbnail max edge (px)
SPARK_MAX = 512                     # 1-D sparkline max points


# The AST-node whitelist for an AUTO (typing-debounced) preview: a value-only,
# side-effect-free expression. Concrete node types PLUS the abstract grouping
# bases (an operator/context node is an isinstance of these). A ``Call`` /
# comprehension / f-string / literal-container / lambda / walrus is deliberately
# NOT here → is_auto_safe returns False → needs an explicit Ctrl+Enter.
_AUTO_SAFE_NODES = (
    ast.Expression, ast.Name, ast.Constant, ast.BinOp, ast.Compare,
    ast.UnaryOp, ast.BoolOp, ast.Subscript, ast.Tuple, ast.Slice, ast.Attribute,
    # abstract bases so operator / context / comparison nodes pass:
    ast.expr_context, ast.operator, ast.cmpop, ast.boolop, ast.unaryop,
)


# ── AST classification ───────────────────────────────────────────────────────


def parse_expr(code):
    """Parse *code* as a single expression (``mode="eval"``). Returns the
    ``ast.Expression`` on success, or None for empty / incomplete / statement
    input (``x = 1`` is a statement, not an expression → None)."""
    try:
        return ast.parse(code, mode="eval")
    except (SyntaxError, ValueError):
        return None


def is_auto_safe(tree) -> bool:
    """True iff EVERY node in *tree* is in the value-only whitelist — i.e. the
    expression can be auto-previewed while the user types (no call, no
    comprehension, no container literal, no side effect). One disallowed node
    (a Call, IfExp, JoinedStr, List/Dict/Set, comprehension, Lambda, NamedExpr,
    Starred, Await…) makes the whole expression unsafe."""
    for node in ast.walk(tree):
        if not isinstance(node, _AUTO_SAFE_NODES):
            return False
    return True


def referenced_names(tree) -> list[str]:
    """The ``ast.Name`` ids referenced in *tree*, in first-appearance order
    (deduped, preserving the first occurrence). Used to find which bound signal
    supplies the navigator cursor position."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in names:
            names.append(node.id)
    return names


# ── navigator-position resolution ────────────────────────────────────────────


def _nav_indices_for(console, names):
    """Resolve the live navigator cursor (indices) + source nav shape from the
    first *name* that is a bound signal tree.

    Walks the referenced names; for the first that is a console binding, finds a
    Plot whose ``signal_tree is tree`` and whose ``parent_selector.current_indices``
    is set, and returns ``(indices_tuple, src_nav_shape)`` where:

      * ``indices_tuple`` — the integer nav position (a region selector's Nx-D
        ``current_indices`` is reduced to its integer centroid), and
      * ``src_nav_shape`` — the source signal's navigation shape IN DATA ORDER
        (``root.data.shape[:navigation_dimension]``, the same leading-axes
        convention ``_direct_read_frame`` uses).

    No locks: ``current_indices`` is rebound atomically on the dispatcher thread,
    so a torn read here just yields the OLD or the NEW tuple — both are valid
    positions. Any failure → ``(None, None)`` (the caller falls back to frame 0).
    """
    import numpy as np
    for name in names:
        tree = console._bindings.get(name)
        if tree is None:
            continue
        root = getattr(tree, "root", None)
        if root is None:
            continue
        # Find the plot driving this tree's navigator (the selector holding the
        # live cursor). Scan tolerantly — mirror _tree_window_ids in console.py.
        sel = None
        try:
            for p in list(getattr(console._session, "_plots", []) or []):
                if getattr(p, "signal_tree", None) is not tree:
                    continue
                ps = getattr(p, "parent_selector", None)
                if ps is None:
                    continue
                ci = getattr(ps, "current_indices", None)
                if ci is None:
                    continue
                sel = ps
                break
        except Exception:
            sel = None
        if sel is None:
            continue
        try:
            # Copy immediately so a concurrent rebind can't mutate under us.
            idx = np.asarray(sel.current_indices)
            if idx.ndim > 1:
                # An integrating region → collapse to its integer centroid.
                idx = np.rint(idx.mean(axis=0))
            indices = tuple(int(v) for v in np.atleast_1d(idx))
            nav_dim = int(root.axes_manager.navigation_dimension)
            src_nav_shape = tuple(int(s) for s in root.data.shape[:nav_dim])
            return indices, src_nav_shape
        except Exception:
            return None, None
    return None, None


# ── the preview evaluator ────────────────────────────────────────────────────


def evaluate_preview(console, code, *, auto) -> dict:
    """Evaluate *code* to a small preview payload (the ``console_preview_result``
    body, minus the ``type``/``preview_id``/``elapsed_ms`` stamps its caller adds).

    Returns a dict with a ``kind`` of ``scalar`` | ``sparkline`` | ``image`` |
    ``unavailable``. Runs on the console thread; touches only the namespace + the
    session's plot list (read-only), never a Session mutation.
    """
    import numpy as np

    tree = parse_expr(code)
    if tree is None:
        return {"kind": "unavailable", "reason": "incomplete expression"}

    # AUTO tier gate: a call / comprehension / literal-container needs Ctrl+Enter.
    if auto and not is_auto_safe(tree):
        return {"kind": "unavailable",
                "reason": "contains a call — Ctrl+Enter to preview"}

    # Evaluate to a (lazy) value. Swallow stdout so a stray print in the
    # expression can't pollute the protocol channel. On error: an AUTO preview
    # is fully quiet (the user is mid-type); a manual one surfaces the reason.
    try:
        compiled = compile(tree, "<preview>", "eval")
        with contextlib.redirect_stdout(io.StringIO()):
            value = eval(compiled, console._ns, console._ns)  # noqa: S307 — user console
    except Exception as e:  # noqa: BLE001 — any eval error → unavailable
        return {"kind": "unavailable",
                "reason": "" if auto else f"{type(e).__name__}: {e}"}

    # Classify: hyperspy signal → its .data + nav dim; dask/ndarray → array +
    # trailing-2-signal-axes convention (same as _materialise); scalar → repr.
    from spyde.backend.console import _kind_of, _safe_repr
    kind = _kind_of(value)

    if kind == "scalar":
        dtype = None
        dt = getattr(value, "dtype", None)
        if dt is not None:
            dtype = str(dt)
        return {"kind": "scalar", "text": _safe_repr(value)[:200], "dtype": dtype}

    if kind == "signal":
        arr = value.data
        try:
            nav_dim = int(value.axes_manager.navigation_dimension)
        except Exception:
            nav_dim = max(0, getattr(arr, "ndim", 2) - 2)
    elif kind in ("dask", "ndarray"):
        arr = value
        nav_dim = max(0, int(getattr(arr, "ndim", 0)) - 2)
    else:
        return {"kind": "unavailable",
                "reason": f"can't preview {type(value).__name__}"}

    full_shape = [int(s) for s in getattr(arr, "shape", ()) or ()]
    result_dtype = str(getattr(arr, "dtype", "")) or None

    # Slice ONE navigator frame at the live cursor (so we compute one frame, not
    # the whole nav space). Use the resolved cursor ONLY if the value's leading
    # nav axes match the SOURCE nav shape (a derived view could have re-shaped
    # them); otherwise fall back to frame 0.
    if nav_dim > 0:
        indices, src_nav_shape = _nav_indices_for(console, referenced_names(tree))
        if (indices is None
                or tuple(arr.shape[:nav_dim]) != tuple(src_nav_shape or ())):
            indices = (0,) * nav_dim
        # Clamp each index into range (a stale cursor from a bigger source).
        indices = tuple(
            max(0, min(int(indices[k]), int(arr.shape[k]) - 1))
            for k in range(nav_dim)
        )
        sliced = arr[indices]
    else:
        sliced = arr

    # Cost guard (dask only) — fail closed. A numpy result is already in RAM.
    reason = _cost_guard(sliced, console)
    if reason is not None:
        if auto and reason == "too expensive":
            # Nudge the user toward the manual path — a heavy view is what
            # Ctrl+Enter is for.
            reason = "too expensive — Ctrl+Enter to preview"
        return {"kind": "unavailable", "reason": reason,
                "shape": full_shape or None, "dtype": result_dtype}

    # Compute the (small, cost-guarded) frame. dask → synchronous scheduler
    # (same as the live nav read); numpy → already materialised.
    try:
        if hasattr(sliced, "compute") and hasattr(sliced, "chunks"):
            frame = np.asarray(sliced.compute(scheduler="synchronous"))
        else:
            frame = np.asarray(sliced)
    except Exception as e:  # noqa: BLE001
        return {"kind": "unavailable",
                "reason": "" if auto else f"{type(e).__name__}: {e}",
                "shape": full_shape or None, "dtype": result_dtype}

    frame = np.squeeze(frame)

    payload = _render(frame)
    # FULL pre-slice result shape + dtype on every non-scalar payload.
    if payload.get("kind") != "scalar":
        payload["shape"] = full_shape or None
        payload["dtype"] = result_dtype
    return payload


# ── cost guard (RAM-safety: refuse anything that would touch too much data) ──


def _cost_guard(sliced, console):
    """Return a refusal reason (str) if computing *sliced* would be too expensive
    / too large, else None. FAILS CLOSED: any exception in the probe → refuse
    ("can't estimate cost") — this is the memory-safety rule, we never guess in
    favour of computing.

    Only meaningful for dask arrays (a numpy result is already in RAM). We:
      * reject on the result size itself (MAX_RESULT_BYTES);
      * CULL the sliced graph to its own keys (cull, do NOT fuse/optimize —
        fusing renames keys and destroys source attribution);
      * tally, per bound source, how many of that source's chunk keys the culled
        graph references, times that source's per-chunk byte size;
      * reject over MAX_SOURCE_CHUNKS or MAX_EST_BYTES; and if NO key attributed
        to any known source, fall back to a raw graph-task count (MAX_GRAPH_TASKS).
    """
    import numpy as np

    # numpy result → in RAM already; no graph to probe.
    if not (hasattr(sliced, "compute") and hasattr(sliced, "chunks")):
        return None

    try:
        # Result-size guard.
        result_bytes = int(np.prod(sliced.shape)) * int(sliced.dtype.itemsize)
        if result_bytes > MAX_RESULT_BYTES:
            return "result too large"

        # Cull the graph to just the keys the result needs. Culling (unlike
        # fusion/optimization) preserves the source layer names, so a key like
        # ("array-abc", 0, 3) still attributes to that source. Try the
        # HighLevelGraph.cull API, else fall back to the low-level dict cull.
        import dask
        keys = set(dask.core.flatten(sliced.__dask_keys__()))
        graph = sliced.__dask_graph__()
        try:
            culled = graph.cull(keys)
        except Exception:
            # Low-level fallback: dask.base.cull on the materialised dict form.
            culled, _deps = dask.base.cull(dict(graph), list(keys))

        # Build {source dask-name: per-chunk nbytes} for every bound signal so we
        # can attribute a culled key back to the source it reads.
        src_chunk_bytes: dict[str, int] = {}
        for tree in console._bindings.values():
            root = getattr(tree, "root", None)
            if root is None:
                continue
            data = getattr(root, "data", None)
            name = getattr(data, "name", None)
            chunksize = getattr(data, "chunksize", None)
            if name is None or chunksize is None:
                continue
            try:
                chunk_nbytes = int(np.prod(chunksize)) * int(data.dtype.itemsize)
            except Exception:
                continue
            src_chunk_bytes[name] = chunk_nbytes

        # Tally source-chunk references across the culled graph's keys. A key is
        # either a tuple ``(name, i, j, …)`` or a bare string ``name``.
        counts: dict[str, int] = {}
        for key in culled.keys():
            if isinstance(key, tuple) and key:
                head = key[0]
            else:
                head = key
            if head in src_chunk_bytes:
                counts[head] = counts.get(head, 0) + 1

        n_src_chunks = sum(counts.values())
        est_bytes = sum(counts[nm] * src_chunk_bytes[nm] for nm in counts)

        if n_src_chunks > MAX_SOURCE_CHUNKS or est_bytes > MAX_EST_BYTES:
            return "too expensive"

        # No key attributed to a known source (a fully synthetic graph, e.g.
        # da.ones(...) + da.ones(...)) → bound by raw task count.
        if n_src_chunks == 0 and len(culled) > MAX_GRAPH_TASKS:
            return "too expensive"

        return None
    except Exception:
        # Fail closed — a probe we can't complete is treated as too expensive.
        return "can't estimate cost"


# ── rendering (small payloads only) ──────────────────────────────────────────


def _render(frame) -> dict:
    """Render a (squeezed) numpy *frame* to a preview payload by dimensionality:
    0-D → scalar text; 1-D → sparkline; 2-D → downsampled uint8 thumbnail;
    3-D+ → unavailable (we only preview a single frame)."""
    import numpy as np

    ndim = frame.ndim
    if ndim == 0:
        return {"kind": "scalar", "text": _repr_scalar(frame),
                "dtype": str(getattr(frame, "dtype", "")) or None}

    if ndim == 1:
        vec = np.asarray(frame)
        stride = max(1, len(vec) // SPARK_MAX)
        vec = vec[::stride][:SPARK_MAX]
        # NaN/inf → null: a raw JSON NaN literal breaks JSON.parse in the Electron
        # main process (MANDATORY — see the frontend contract).
        points = [float(v) if np.isfinite(v) else None for v in vec]
        return {"kind": "sparkline", "points": points}

    if ndim == 2:
        return _render_2d(frame)

    return {"kind": "unavailable", "reason": "3-D+ result"}


def _repr_scalar(value) -> str:
    """A short repr for a 0-D result (≤200 chars)."""
    try:
        return repr(value)[:200]
    except Exception as e:  # noqa: BLE001
        return f"<unreprable: {e}>"[:200]


def _robust_levels(img):
    """Robust (mn, mx) display range for a 2-D preview thumbnail: the 2–99.5%
    finite percentiles, with min/max and (0, 1) fallbacks for degenerate data.

    A small, local re-implementation of ``Plot._robust_levels``
    (spyde/drawing/plots/plot.py:953) — deliberately NOT imported, so the preview
    evaluator never pulls in the drawing stack on the console thread."""
    import numpy as np
    try:
        finite = np.asarray(img, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        mn = float(np.percentile(finite, 2.0))
        mx = float(np.percentile(finite, 99.5))
        if mx <= mn:
            mn, mx = float(finite.min()), float(finite.max())
        if mx <= mn:
            return 0.0, 1.0
        return mn, mx
    except Exception:
        return 0.0, 1.0


def _render_2d(img) -> dict:
    """Downsample a 2-D frame to a ≤THUMB_MAX-edge robust-normalized uint8
    thumbnail and base64-encode its row-major bytes → an ``image`` payload
    (``w`` = columns, ``h`` = rows; ``len(bytes) == w*h``)."""
    import numpy as np

    img = np.asarray(img)
    if img.dtype == np.bool_:
        img = img.astype(np.uint8)

    rows, cols = int(img.shape[0]), int(img.shape[1])
    sy = max(1, rows // THUMB_MAX)
    sx = max(1, cols // THUMB_MAX)
    img = img[::sy, ::sx]
    rows, cols = int(img.shape[0]), int(img.shape[1])

    mn, mx = _robust_levels(img)
    span = (mx - mn) or 1.0
    f = np.asarray(img, dtype=np.float64)
    norm = np.clip((f - mn) / span, 0.0, 1.0) * 255.0
    img8 = norm.astype(np.uint8)

    data_b64 = base64.b64encode(np.ascontiguousarray(img8).tobytes()).decode("ascii")
    return {"kind": "image", "w": cols, "h": rows, "data_b64": data_b64}

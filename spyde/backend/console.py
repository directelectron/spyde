"""
console.py — the SpyDE math console execution engine (a Jupyter-like cell runner
behind the renderer's one-line console bar).

The user types Python (``mask = s1 > 100``, ``np.random.rand(256, 256)``,
``s1 + s2``); the engine runs it with FULL cell semantics (persistent namespace,
imports, multi-statement code) and IPython-style last-expression echo: the code is
parsed with :mod:`ast`, the preceding statements are ``exec``'d, and — if the last
top-level node is an expression — that node is ``eval``'d and its value captured.
A non-None value becomes a draggable "chip" the renderer drops into the MDI to
create a new signal window.

Design constraints (see CLAUDE.md):

* **Lazy end-to-end.** Expressions on lazy hyperspy signals / dask arrays build
  graphs and are NEVER computed. The engine derives echo/metadata from
  ``shape``/``dtype``/``repr`` only (those reprs are already cheap + truncated).
  Materialising a chip goes through the SAME ``_add_signal`` path the file loaders
  use — creating the tree may launch the normal progressive-navigator compute
  (that's expected), but the engine itself adds no ``.compute()``/``.result()``.

* **Threading.** The asyncio main loop must never block on user code, so every
  execution runs on ONE dedicated daemon *console thread* (a serial queue —
  order preserved, a second exec while one runs is queued). The namespace is
  mutated ONLY on that thread, so no lock guards it. Binding refreshes requested
  from the main thread (tree add / close) are POSTED as tasks into the same queue
  — they too run on the console thread, keeping the namespace single-threaded.
  All IPC emits and any Session/tree mutation are marshalled back to the main
  asyncio thread via ``Session._dispatch_to_main`` (``loop.call_soon_threadsafe``),
  exactly like ``PlotUpdateWorker`` / ``lifecycle.run_on_worker`` do.
"""
from __future__ import annotations

import ast
import contextlib
import io
import keyword
import logging
import queue
import re
import threading
import traceback

from spyde.backend import ipc

log = logging.getLogger(__name__)

# Truncate value reprs / captured stdout to keep the IPC line small (the renderer
# only shows a preview). ~500 chars per the IPC contract.
_REPR_CAP = 500
_STDOUT_CAP = 4000
_TB_CAP = 8000

_IDENT_RE = re.compile(r"[^0-9a-zA-Z_]+")


def _sanitize_identifier(name: str, fallback: str = "sig") -> str:
    """Turn an arbitrary signal title into a valid Python identifier.

    Non-identifier runs collapse to ``_``; a leading digit / empty result / a
    Python keyword is prefixed so the name is always bindable and never shadows a
    keyword. Collision handling (``_2`` suffixes) is done by the caller, which
    knows the full set of names in use.
    """
    base = _IDENT_RE.sub("_", str(name or "")).strip("_")
    if not base:
        base = fallback
    if base[0].isdigit():
        base = f"_{base}"
    if keyword.iskeyword(base):
        base = f"{base}_"
    return base


class ConsoleSession:
    """The math-console execution engine, owned by :class:`Session`.

    One instance per session (lazily created via ``session.console``). Holds the
    persistent namespace, the result registry, the auto-exposed signal bindings,
    and the single console worker thread. Shut down via :meth:`shutdown` from
    ``Session.shutdown``.
    """

    def __init__(self, session) -> None:
        self._session = session
        # The persistent user namespace. Mutated ONLY on the console thread.
        self._ns: dict[str, object] = {}
        # Result registry: outN -> value, plus the ordered list of names the
        # console has registered (so console_vars can list chips). Touched only
        # on the console thread.
        self._out_counter = 0
        # name -> ("out"|"assign") source classification for chip listing.
        self._registered: dict[str, str] = {}
        # Ordered list of registered chip names (out<N> + assigned vars), newest
        # last, so the chip strip has a stable order.
        self._chip_order: list[str] = []
        # Current signal bindings: identifier -> the tree root signal. Rebuilt on
        # every tree add / close (refresh_bindings). Also touched only on the
        # console thread. Maps binding-name -> (signal, tree) so materialisation
        # and window_ids resolution are cheap.
        self._bindings: dict[str, object] = {}   # name -> tree (source=="signal")

        # Newest-wins guard for live previews. A plain int is read/written
        # atomically under the GIL (no lock): submit_preview stamps the latest
        # id, _do_preview no-ops if a newer preview (or an exec) has since bumped
        # it. submit_exec resets it to -1 so a queued preview loses to the exec.
        self._latest_preview_id = -1
        # The most recent AUTO preview as (code, preview_id) — re-run when a
        # navigator commits a new position (base_selector.NAV_CHANGE_HOOKS) so
        # the thumbnail tracks the cursor. Cleared by an EMPTY-code preview (the
        # frontend's "eye off / cell emptied" STOP) and by submit_exec.
        # Tuple/None rebinds are GIL-atomic (no lock), same as
        # _latest_preview_id.
        self._last_auto_preview: "tuple[str, int] | None" = None
        # Coalesce nav-change notifications: at most ONE nav_refresh queued at a
        # time. A drag fires dozens of commits; each refresh re-reads the LIVE
        # cursor when it runs, so dropped notifications lose nothing (the same
        # latest-wins philosophy as the nav dispatcher itself).
        self._nav_refresh_queued = False

        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="spyde-console"
        )
        self._thread.start()
        # Seed the namespace + expose current signals (posted onto the console
        # thread so the heavy hyperspy/dask import happens off the main loop).
        self._queue.put(("_init", None))
        # Track the navigator so the live preview follows the cursor. Fired on
        # the nav dispatcher thread; notify_nav_changed only enqueues (cheap).
        try:
            from spyde.drawing.selectors import base_selector
            base_selector.NAV_CHANGE_HOOKS.append(self.notify_nav_changed)
        except Exception:
            log.debug("nav-change hook registration failed", exc_info=True)

    # ── public API (called from the main asyncio thread) ─────────────────────

    def submit_exec(self, code: str, exec_id: int) -> None:
        """Queue a cell for execution on the console thread (returns immediately)."""
        if self._closed:
            return
        # An exec ALWAYS wins over any pending preview: reset the latest-preview
        # id BEFORE the put so every preview queued before this exec sees a
        # mismatched id in _do_preview and no-ops (stale). Set before enqueuing
        # so a preview submitted concurrently can't slip its id in after. Also
        # drop the nav-refresh expression — the cell the user just ran owns the
        # console now; navigator moves must not resurrect the old preview.
        self._latest_preview_id = -1
        self._last_auto_preview = None
        self._queue.put(("exec", (code, exec_id)))

    def submit_preview(self, code: str, preview_id: int, auto: bool) -> None:
        """Queue a live preview of *code* on the console thread (returns
        immediately). Previews share the SERIAL console queue, so the namespace
        stays single-threaded (the module invariant) — a preview never runs
        concurrently with an exec or another preview. ``_latest_preview_id`` is
        stamped to *preview_id* so a superseded preview (a newer keystroke, or an
        exec) is dropped as stale in _do_preview without ever computing.

        An EMPTY code is the frontend's STOP signal (eye toggled off / cell
        emptied): it clears the nav-refresh expression and evaluates nothing —
        no reply is emitted."""
        if self._closed:
            return
        pid = int(preview_id)
        self._latest_preview_id = pid
        if not code.strip():
            self._last_auto_preview = None
            return
        if auto:
            # Remember the expression so navigator moves re-run it (the preview
            # slices at the live cursor — see notify_nav_changed). Explicit
            # (Ctrl+Enter) one-shots are deliberately NOT nav-tracked: they may
            # contain arbitrary calls and be expensive per evaluation.
            self._last_auto_preview = (code, pid)
        self._queue.put(("preview", (code, pid, bool(auto))))

    def notify_nav_changed(self) -> None:
        """A navigator selector committed a genuinely NEW position (called on
        the nav dispatcher thread via ``base_selector.NAV_CHANGE_HOOKS``).
        Re-run the last AUTO preview so the thumbnail tracks the cursor.

        Coalesced: at most one refresh sits in the queue — the refresh re-reads
        the live cursor when it actually runs, so folding a burst of moves into
        one refresh loses nothing. Must stay cheap (enqueue only): it runs on
        the dispatcher thread, in the navigator's hot path."""
        if self._closed or self._last_auto_preview is None:
            return
        if self._nav_refresh_queued:
            return
        self._nav_refresh_queued = True
        self._queue.put(("nav_refresh", None))

    def submit_complete(self, prefix: str, complete_id: int) -> None:
        """Queue a completion request (runs on the console thread so it reads the
        live namespace without a lock)."""
        if self._closed:
            return
        self._queue.put(("complete", (prefix, complete_id)))

    def remove_var(self, name: str) -> None:
        """Remove a REGISTERED result chip (out<N> / assigned var) from the
        namespace + registry and re-emit console_vars — the chip's (×) button
        (the console_remove_var command). Only registry names are removable: a
        signal binding is owned by its tree (closing the window is its
        lifecycle), so an unknown / signal name no-ops."""
        if self._closed or not name:
            return
        self._queue.put(("remove_var", str(name)))

    def create_window(self, name: str) -> None:
        """Materialise the namespace/registry variable *name* as a new signal
        window (the console_create_window command / a chip drop). Runs the lookup
        on the console thread, then marshals the actual window creation onto the
        main thread."""
        if self._closed:
            return
        self._queue.put(("create_window", name))

    def refresh_bindings(self) -> None:
        """Re-derive the signal bindings from the session's current trees and
        re-emit console_vars. Called from the MAIN thread at the tree add / close
        seams; POSTED into the console queue so the namespace stays single-threaded.
        """
        if self._closed:
            return
        self._queue.put(("refresh", None))

    def bind_node(self, plot, signal_id) -> None:
        """Bind a specific WORKFLOW node (a mid-tree signal) into the console
        namespace under a fresh ``node<N>`` name and re-emit console_vars — the
        seam a Workflow-node → console drop uses. The node is resolved on the MAIN
        thread (tree walk by ``id(signal)``) so we don't touch tree state off it;
        the namespace mutation is posted to the console thread."""
        if self._closed or signal_id is None:
            return
        tree = getattr(plot, "signal_tree", None)
        if tree is None:
            return
        # Resolve the node's signal by id (SignalNode.signal_id == id(signal)).
        target = None
        node_name = "node"
        try:
            for node in tree.walk():
                sig = getattr(node, "signal", None)
                if sig is not None and id(sig) == int(signal_id):
                    target = sig
                    node_name = getattr(node, "name", "node") or "node"
                    break
        except Exception:
            target = None
        if target is None:
            return
        self._queue.put(("bind_node", (target, node_name)))

    def shutdown(self) -> None:
        self._closed = True
        try:
            from spyde.drawing.selectors import base_selector
            if self.notify_nav_changed in base_selector.NAV_CHANGE_HOOKS:
                base_selector.NAV_CHANGE_HOOKS.remove(self.notify_nav_changed)
        except Exception:
            pass
        self._queue.put(("_stop", None))

    # ── console worker thread ────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while True:
            try:
                kind, arg = self._queue.get()
            except Exception:
                break
            try:
                if kind == "_stop":
                    break
                elif kind == "_init":
                    self._init_namespace()
                    self._sync_bindings()
                    self._emit_vars()
                elif kind == "exec":
                    self._do_exec(*arg)
                elif kind == "preview":
                    self._do_preview(*arg)
                elif kind == "nav_refresh":
                    self._do_nav_refresh()
                elif kind == "complete":
                    self._do_complete(*arg)
                elif kind == "create_window":
                    self._do_create_window(arg)
                elif kind == "remove_var":
                    self._do_remove_var(arg)
                elif kind == "bind_node":
                    self._do_bind_node(*arg)
                elif kind == "refresh":
                    self._sync_bindings()
                    self._emit_vars()
            except Exception:
                log.exception("console worker task %r failed", kind)

    def _init_namespace(self) -> None:
        """Populate the persistent namespace with np / hs / da / show.

        hyperspy is imported through the project's single-flight heavy-import gate
        so the backend startup isn't slowed and the concurrent import poisoning is
        avoided (see heavy_imports). Runs on the console thread."""
        import numpy as np
        import dask.array as da
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()
        import hyperspy.api as hs

        self._ns.setdefault("__name__", "__console__")
        self._ns.setdefault("__builtins__", __builtins__)
        self._ns["np"] = np
        self._ns["numpy"] = np
        self._ns["da"] = da
        self._ns["hs"] = hs
        self._ns["show"] = self._show

    # ── binding sync (signal trees → identifiers + positional aliases) ───────

    def _sync_bindings(self) -> None:
        """Rebuild the auto-exposed signal bindings from ``session.signal_trees``.

        Each loaded tree's root is exposed under a sanitized identifier derived
        from its title (collisions get ``_2`` suffixes) AND a positional alias
        ``s1``, ``s2``, … in load order. Closed trees drop their bindings. Reads
        the (main-thread-owned) ``signal_trees`` list; list iteration of a plain
        Python list is atomic enough here, and the actual namespace mutation stays
        on this console thread. Runs on the console thread."""
        session = self._session
        trees = list(getattr(session, "signal_trees", []) or [])

        # Remember what the namespace currently holds for each signal-binding name
        # so we can drop STALE ones (a closed tree's s1) after rebuilding — but
        # never clobber a name the USER reassigned (tracked in _registered) or a
        # name that maps to a still-open tree below.
        prev_binding_names = set(self._bindings.keys())

        new_bindings: dict[str, object] = {}
        used: set[str] = set()
        # Reserve the fixed builtins so a signal can't shadow np/hs/da/show.
        reserved = {"np", "numpy", "da", "hs", "show"}
        for pos, tree in enumerate(trees, start=1):
            root = getattr(tree, "root", None)
            if root is None:
                continue
            title = None
            try:
                title = root.metadata.get_item("General.title", default=None)
            except Exception:
                title = None
            base = _sanitize_identifier(title or f"signal_{pos}", fallback=f"signal_{pos}")
            name = base
            n = 2
            while name in used or name in reserved:
                name = f"{base}_{n}"
                n += 1
            used.add(name)
            new_bindings[name] = tree
            # Positional alias in load order — always s1, s2, …
            alias = f"s{pos}"
            new_bindings.setdefault(alias, tree)
            used.add(alias)

        # Apply: expose each binding's root signal in the namespace. The signal
        # bindings do NOT overwrite a user-assigned variable of the same name that
        # the user created deliberately — but a fresh signal name that isn't a
        # user var should appear. We track which names are signal bindings so a
        # later refresh can cleanly replace them.
        for name, tree in new_bindings.items():
            root = getattr(tree, "root", None)
            if root is None:
                continue
            # If the user has assigned this exact name to something else, leave
            # their value in the live namespace but still record the binding so the
            # chip strip / window_ids reflect the signal — the positional alias
            # (s1…) is always authoritative for materialisation.
            if name not in self._registered:
                self._ns[name] = root

        # Drop namespace entries for signal bindings that vanished (a closed
        # tree's s1 / titled name) — but keep user-reassigned names and any name
        # that's still a live binding.
        for stale in prev_binding_names - set(new_bindings.keys()):
            if stale in self._registered:
                continue   # user reassigned this name → leave it
            self._ns.pop(stale, None)

        self._bindings = new_bindings

    # ── exec / eval (IPython last-expression echo) ───────────────────────────

    def _do_exec(self, code: str, exec_id: int) -> None:
        """Run one cell on the console thread and emit console_result + refreshed
        console_vars. Preceding statements are exec'd; a trailing top-level
        expression is eval'd and its value captured (Jupyter echo)."""
        import time
        t0 = time.perf_counter()
        stdout_buf = io.StringIO()
        ok = True
        err: str | None = None
        tb: str | None = None
        value = None
        has_value = False
        assigned_names: list[str] = []

        try:
            parsed = ast.parse(code, mode="exec")
        except SyntaxError as e:
            dt = (time.perf_counter() - t0) * 1000.0
            self._emit_result(
                exec_id, ok=False, value_repr="", stdout="",
                error=f"{type(e).__name__}: {e}",
                tb="".join(traceback.format_exception_only(type(e), e)),
                duration_ms=dt, result=None,
            )
            return

        body = list(parsed.body)
        eval_node = None
        if body and isinstance(body[-1], ast.Expr):
            eval_node = body.pop()

        # Names bound by top-level assignments (so they register as chips).
        assigned_names = _assigned_names(parsed.body if eval_node is None else body)

        try:
            with contextlib.redirect_stdout(stdout_buf):
                if body:
                    exec_code = compile(
                        ast.Module(body=body, type_ignores=[]),
                        "<console>", "exec",
                    )
                    exec(exec_code, self._ns, self._ns)
                if eval_node is not None:
                    eval_code = compile(
                        ast.Expression(body=eval_node.value),
                        "<console>", "eval",
                    )
                    value = eval(eval_code, self._ns, self._ns)
                    has_value = value is not None
        except Exception as e:  # noqa: BLE001 — surface ANY user error as ok:false
            ok = False
            err = f"{type(e).__name__}: {e}"
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))

        dt = (time.perf_counter() - t0) * 1000.0
        stdout_txt = stdout_buf.getvalue()

        # Register the echoed value as out<N> (only on success + non-None).
        result_meta = None
        if ok and has_value:
            out_name = self._register_out(value)
            result_meta = self._describe(value, out_name)
        # Register assigned names so their variables become chips too (even a
        # bare assignment with no echo). Re-register on reassignment.
        for nm in assigned_names:
            if nm in self._ns and not _is_builtin_binding(nm):
                self._register_assign(nm)

        self._emit_result(
            exec_id, ok=ok,
            value_repr=(_safe_repr(value) if (ok and has_value) else ""),
            stdout=stdout_txt[:_STDOUT_CAP],
            error=err, tb=(tb[:_TB_CAP] if tb else None),
            duration_ms=dt, result=result_meta,
        )
        # Refresh chips + bindings after every exec (a new var / out chip appeared,
        # and the user may have closed/opened data via show()).
        self._emit_vars()

    def _register_out(self, value) -> str:
        self._out_counter += 1
        name = f"out{self._out_counter}"
        self._ns[name] = value
        self._registered[name] = "out"
        if name not in self._chip_order:
            self._chip_order.append(name)
        return name

    def _register_assign(self, name: str) -> None:
        self._registered[name] = "assign"
        if name not in self._chip_order:
            self._chip_order.append(name)

    # ── live preview (display-only, never registers/echoes) ──────────────────

    def _do_preview(self, code: str, preview_id: int, auto: bool) -> None:
        """Evaluate a live preview of *code* and emit console_preview_result.

        Unlike _do_exec, a preview is DISPLAY-ONLY: it never registers an
        out<N>/assign chip, never mutates the registry, and never emits
        console_result. It builds the lazy expression, slices ONE navigator frame
        at the referenced signal's current cursor, cost-guards the culled graph,
        and computes only that (see console_preview). Stale previews (a newer
        keystroke or an exec has since bumped ``_latest_preview_id``) no-op so we
        don't waste a compute on a position/expression the user has moved past."""
        if preview_id != self._latest_preview_id:
            return   # superseded by a newer preview or an exec → drop it
        import time
        t0 = time.perf_counter()
        from spyde.backend import console_preview
        payload = console_preview.evaluate_preview(self, code, auto=auto)
        payload["type"] = "console_preview_result"
        payload["preview_id"] = preview_id
        payload["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        self._dispatch(lambda: ipc.emit(payload))

    def _do_nav_refresh(self) -> None:
        """Re-evaluate the last AUTO preview at the (new) navigator position —
        the console-thread half of notify_nav_changed. Clears the coalescing
        flag FIRST so a nav move landing during this evaluation queues the next
        refresh (rather than being silently swallowed). Re-emits with the SAME
        preview_id, so the frontend's newest-wins intake accepts it in place."""
        self._nav_refresh_queued = False
        last = self._last_auto_preview
        if last is None:
            return
        code, pid = last
        if pid != self._latest_preview_id:
            return   # user typed / ran a cell since — that flow owns the panel
        self._do_preview(code, pid, True)

    # ── completion ───────────────────────────────────────────────────────────

    def _do_complete(self, prefix: str, complete_id: int) -> None:
        """Return names matching *prefix*. Supports ONE level of attribute
        completion (``s1.su`` → members of ``s1`` starting with ``su``). Kept
        simple + exception-safe: attribute names come from ``dir()`` on the object
        (which may run descriptors) — guarded so a misbehaving attribute never
        breaks completion."""
        matches: list[str] = []
        try:
            if "." in prefix:
                obj_expr, _, attr_prefix = prefix.rpartition(".")
                # Only support a simple dotted chain of identifiers on the LHS
                # (no calls/subscripts) so we never evaluate side-effecting code.
                if obj_expr and all(
                    part.isidentifier() for part in obj_expr.split(".")
                ):
                    obj = self._resolve_dotted(obj_expr)
                    if obj is not None:
                        for attr in _safe_dir(obj):
                            if attr.startswith(attr_prefix) and not attr.startswith("__"):
                                matches.append(f"{obj_expr}.{attr}")
            else:
                names = set(self._ns.keys())
                names.update(self._registered.keys())
                # Drop internal dunders / the __sig_ mirror keys.
                for nm in sorted(names):
                    if nm.startswith("__"):
                        continue
                    if nm.startswith(prefix):
                        matches.append(nm)
        except Exception as e:
            log.debug("completion for %r failed: %s", prefix, e)
        matches = sorted(dict.fromkeys(matches))[:200]
        self._dispatch(lambda: ipc.emit({
            "type": "console_completions",
            "complete_id": complete_id,
            "matches": matches,
        }))

    def _resolve_dotted(self, expr: str):
        """Resolve ``a.b.c`` against the namespace using only ``getattr`` (no eval,
        no calls/subscripts). Returns the object or None."""
        parts = expr.split(".")
        obj = self._ns.get(parts[0])
        for part in parts[1:]:
            if obj is None:
                return None
            try:
                obj = getattr(obj, part)
            except Exception:
                return None
        return obj

    # ── show() helper + materialisation ──────────────────────────────────────

    def _show(self, x):
        """``show(x)`` in user code: materialise *x* as a new signal window — the
        SAME path a chip drop uses. Runs on the CONSOLE thread (user code), so it
        marshals the actual window creation onto the main thread and returns
        immediately (fire-and-forget). Returns None so it doesn't itself echo /
        register an out chip."""
        self._materialise(x, name=None, source_expr="show(...)")
        self._dispatch(lambda: ipc.emit_status("Opening console result…"))
        return None

    def _do_bind_node(self, signal, node_name: str) -> None:
        """Bind a workflow node's signal into the namespace under a fresh, valid
        identifier derived from the node name (``node`` → ``node``, ``node_2`` on
        collision), register it as an assign chip, and re-emit console_vars — the
        Workflow-node → console drop. Runs on the console thread."""
        base = _sanitize_identifier(node_name or "node", fallback="node")
        reserved = {"np", "numpy", "da", "hs", "show"}
        name = base
        n = 2
        # Don't collide with builtins, live bindings, or a DIFFERENT registered var.
        while (name in reserved or name in self._bindings
               or (name in self._registered and self._ns.get(name) is not signal)):
            name = f"{base}_{n}"
            n += 1
        self._ns[name] = signal
        self._register_assign(name)
        self._emit_vars()
        # Tell the renderer the bound name so it can insert it at the caret (the
        # Workflow-node → console drop; previously the name never reached the
        # frontend, so the drop bound the node but inserted nothing).
        self._dispatch(lambda: ipc.emit({"type": "console_node_bound", "name": name}))
        self._dispatch(lambda: ipc.emit_status(f"Console: bound {name}"))

    def _do_remove_var(self, name: str) -> None:
        """Handle console_remove_var (the chip's ×): drop a REGISTERED chip name
        from the registry + namespace and re-emit console_vars. Runs on the
        console thread. Signal bindings / unknown names no-op — the registry
        membership check is the guard (chips only exist for out<N>/assign)."""
        if name not in self._registered:
            return
        self._registered.pop(name, None)
        try:
            self._chip_order.remove(name)
        except ValueError:
            pass
        self._ns.pop(name, None)
        self._emit_vars()

    def _do_create_window(self, name: str) -> None:
        """Handle console_create_window: look up *name* in the registry/namespace
        (console thread) and materialise it (marshalled to main). Refuses politely
        if the name is unknown."""
        if not name:
            return
        # Prefer a positional alias / signal binding, else the user namespace /
        # registry. A signal binding materialises its tree's root (which stays
        # lazy); an out/assign value materialises the stored object.
        value = None
        if name in self._bindings:
            tree = self._bindings[name]
            value = getattr(tree, "root", None)
        if value is None:
            value = self._ns.get(name, None)
        if value is None:
            self._dispatch(lambda: ipc.emit_status(
                f"Console: nothing named {name!r} to open."))
            return
        self._materialise(value, name=name, source_expr=f"console: {name}")

    def _materialise(self, value, *, name: str | None, source_expr: str) -> None:
        """Turn *value* into a new signal window, marshalling the mutation onto the
        MAIN thread. hyperspy BaseSignal → open through ``_add_signal`` (the file
        loader path); numpy/dask array → wrap by dimensionality (lazy stays lazy);
        scalar / anything else → refuse politely. The engine adds NO compute — the
        tree's own navigator build may launch the normal progressive compute for a
        lazy signal, which is expected."""
        prepared = self._prepare_signal(value, name=name, source_expr=source_expr)
        if prepared is None:
            kind = _kind_of(value)
            self._dispatch(lambda: ipc.emit_status(
                f"Console: can't open a {kind} as a window."))
            return

        def _open():
            try:
                # _add_signal calls session._notify_console_trees_changed(), which
                # posts a binding refresh back onto the console thread — so the new
                # tree picks up a positional alias + chip automatically.
                self._session._add_signal(prepared, source_path=None)
            except Exception as e:  # noqa: BLE001
                ipc.emit_error(f"Console: failed to open window: {e}")
                log.exception("console materialise failed")

        self._dispatch(_open)

    def _prepare_signal(self, value, *, name: str | None, source_expr: str):
        """Build a hyperspy BaseSignal ready for ``_add_signal`` from *value*, or
        None if it can't be shown. Runs on the console thread (no session mutation
        — pure construction). Keeps lazy data lazy and never computes."""
        from spyde.backend.heavy_imports import ensure_heavy_imports
        ensure_heavy_imports()
        import numpy as np
        import hyperspy.api as hs
        from hyperspy.signal import BaseSignal

        title = name or "Console result"

        if isinstance(value, BaseSignal):
            # Open the signal as-is (lazy signals stay lazy). Stamp a title +
            # provenance without mutating the user's own object graph if we can —
            # a shallow title set is fine (the user handed it to us to display).
            sig = value
            try:
                cur = sig.metadata.get_item("General.title", default=None)
                if not cur and name:
                    sig.metadata.General.title = title
            except Exception as e:
                log.debug("setting console signal title failed: %s", e)
            self._stamp_provenance(sig, source_expr)
            return sig

        # numpy ndarray or dask array → wrap by dimensionality.
        import dask.array as da_mod
        is_dask = isinstance(value, da_mod.Array)
        is_ndarray = isinstance(value, np.ndarray)
        if not (is_dask or is_ndarray):
            return None

        ndim = getattr(value, "ndim", None)
        if ndim is None or ndim < 1:
            return None  # scalar array / 0-D → refuse (handled as "other")

        # Dimensionality → signal class + number of navigation axes.
        #   1-D → Signal1D
        #   2-D → Signal2D
        #   3-D → Signal2D with 1 nav axis (a stack of images / a movie)
        #   4-D → Signal2D with 2 nav axes (4D-STEM)
        #   >4-D → Signal2D spanning the trailing 2 axes as the signal.
        if ndim == 1:
            sig = hs.signals.Signal1D(value)
        elif ndim == 2:
            sig = hs.signals.Signal2D(value)
        else:
            # 3-D+ → 2-D signal (trailing two axes), the rest navigation.
            sig = hs.signals.Signal2D(value)
            try:
                # HyperSpy infers signal_dimension=2 for Signal2D and treats the
                # leading axes as navigation automatically for a 3-D+ array, so no
                # explicit axis surgery is needed — the trailing 2 axes are the
                # signal and the leading ndim-2 are navigation.
                pass
            except Exception:
                pass

        # Keep dask arrays lazy: hs.signals.Signal2D(dask_array) already yields a
        # lazy signal, but guard with as_lazy() so a non-lazy wrap is corrected
        # WITHOUT computing (as_lazy is a graph op).
        if is_dask and not getattr(sig, "_lazy", False):
            try:
                sig = sig.as_lazy()
            except Exception as e:
                log.debug("as_lazy() on console dask signal failed: %s", e)

        try:
            sig.metadata.General.title = title
        except Exception as e:
            log.debug("setting console array title failed: %s", e)
        self._stamp_provenance(sig, source_expr)
        return sig

    @staticmethod
    def _stamp_provenance(sig, source_expr: str) -> None:
        try:
            sig.metadata.set_item("General.notes", f"console: {source_expr}")
            sig.metadata.set_item(
                "General.spyde_provenance",
                {"action": "console", "source_expr": source_expr},
            )
        except Exception as e:
            log.debug("stamping console provenance failed: %s", e)

    # ── console_vars emit ────────────────────────────────────────────────────

    def _emit_vars(self) -> None:
        """Emit the FULL current variable list: signal bindings + registered chips
        (out<N> + assigned vars). Marshalled to the main thread. window_ids is
        non-null only for source=="signal"."""
        vars_list: list[dict] = []
        seen: set[str] = set()

        # Signal bindings first (positional aliases + titled names).
        for name, tree in self._bindings.items():
            if name in seen:
                continue
            seen.add(name)
            root = getattr(tree, "root", None)
            desc = self._describe(root, name)
            desc["source"] = "signal"
            desc["window_ids"] = self._tree_window_ids(tree)
            vars_list.append(desc)

        # Registered chips (out<N> + assigned names) that aren't signal bindings.
        for name in self._chip_order:
            if name in seen:
                continue
            if name not in self._ns:
                continue
            seen.add(name)
            desc = self._describe(self._ns.get(name), name)
            desc["source"] = self._registered.get(name, "out")
            desc["window_ids"] = None
            vars_list.append(desc)

        payload = {"type": "console_vars", "vars": vars_list}
        self._dispatch(lambda: ipc.emit(payload))

    def _tree_window_ids(self, tree):
        """The MDI window ids currently displaying *tree*. Reads the session's
        plot list (main-thread-owned, but a list read is safe here) so the
        renderer can map a dragged signal window → its console variable name."""
        try:
            ids = sorted({
                p.window_id for p in list(getattr(self._session, "_plots", []) or [])
                if getattr(p, "signal_tree", None) is tree
                and getattr(p, "window_id", None) is not None
            })
            return ids or None
        except Exception:
            return None

    # ── value description (lazy-safe: shape/dtype/repr only) ─────────────────

    def _describe(self, value, name: str) -> dict:
        """Return the ``result`` / ``vars`` metadata dict for *value* — kind,
        shape, dtype, lazy — derived from attributes ONLY (never computed)."""
        kind = _kind_of(value)
        shape = None
        dtype = None
        lazy = False
        try:
            if kind == "signal":
                data = getattr(value, "data", None)
                shp = getattr(data, "shape", None)
                shape = list(shp) if shp is not None else None
                dt = getattr(data, "dtype", None)
                dtype = str(dt) if dt is not None else None
                lazy = bool(getattr(value, "_lazy", False))
            elif kind in ("ndarray", "dask"):
                shp = getattr(value, "shape", None)
                shape = list(shp) if shp is not None else None
                dt = getattr(value, "dtype", None)
                dtype = str(dt) if dt is not None else None
                lazy = kind == "dask"
            elif kind == "scalar":
                # numpy scalars carry a dtype; python scalars don't.
                dt = getattr(value, "dtype", None)
                dtype = str(dt) if dt is not None else type(value).__name__
        except Exception as e:
            log.debug("describing console value %r failed: %s", name, e)
        return {
            "name": name, "kind": kind, "shape": shape,
            "dtype": dtype, "lazy": lazy,
        }

    # ── emit helpers (all marshalled to main) ────────────────────────────────

    def _emit_result(self, exec_id, *, ok, value_repr, stdout, error, tb,
                     duration_ms, result) -> None:
        payload = {
            "type": "console_result",
            "exec_id": exec_id,
            "ok": bool(ok),
            "value_repr": value_repr[:_REPR_CAP] if value_repr else "",
            "stdout": stdout or "",
            "error": error,
            "traceback": tb,
            "duration_ms": round(float(duration_ms), 3),
            "result": result,
        }
        self._dispatch(lambda: ipc.emit(payload))

    def _dispatch(self, fn) -> None:
        """Marshal *fn* (an IPC emit or a session mutation) onto the main asyncio
        thread via the session's established dispatcher."""
        try:
            self._session._dispatch_to_main(fn)
        except Exception:
            # Fall back to running inline (no loop yet / tests) — emits are
            # thread-safe (ipc holds its own stdout lock).
            fn()


# ── module helpers ──────────────────────────────────────────────────────────


def _assigned_names(nodes) -> list[str]:
    """Top-level names bound by simple assignments in *nodes* (for chip
    registration). Covers ``x = …``, ``x = y = …``, annotated ``x: T = …``, and
    augmented ``x += …``. Tuple/list unpacking targets are included."""
    names: list[str] = []

    def _add_target(target) -> None:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                _add_target(elt)

    for node in nodes:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                _add_target(tgt)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            _add_target(node.target)
        elif isinstance(node, ast.AugAssign):
            _add_target(node.target)
    # De-dup, preserve order.
    return list(dict.fromkeys(names))


def _is_builtin_binding(name: str) -> bool:
    return name in ("np", "numpy", "da", "hs", "show", "__name__", "__builtins__")


def _kind_of(value) -> str:
    """Classify a value for the IPC ``kind`` field: signal | ndarray | dask |
    scalar | other. Import-light + exception-safe (no heavy imports here — checks
    by module/attribute so it works even before hyperspy is imported)."""
    if value is None:
        return "other"
    # hyperspy BaseSignal — check by walking the MRO names so we don't force an
    # import here (kind_of is called from describe on the console thread, where
    # hyperspy is already imported, but stay defensive).
    try:
        for klass in type(value).__mro__:
            if klass.__name__ == "BaseSignal":
                return "signal"
    except Exception:
        pass
    mod = type(value).__module__ or ""
    cls = type(value).__name__
    if mod.startswith("dask") and cls == "Array":
        return "dask"
    if cls == "ndarray" and mod.startswith("numpy"):
        return "ndarray"
    # numpy scalar (0-d) or a python scalar.
    if isinstance(value, (int, float, complex, bool)):
        return "scalar"
    if mod.startswith("numpy") and getattr(value, "ndim", None) == 0:
        return "scalar"
    return "other"


def _safe_repr(value) -> str:
    """A lazy-safe repr: hyperspy/numpy/dask reprs are already cheap + truncated
    (they show shape/dtype, not the data), so a plain repr never triggers a
    compute. Guarded so a misbehaving __repr__ can't break the result emit."""
    try:
        return repr(value)
    except Exception as e:  # noqa: BLE001
        return f"<unreprable {type(value).__name__}: {e}>"


def _safe_dir(obj) -> list[str]:
    try:
        return [a for a in dir(obj)]
    except Exception:
        return []

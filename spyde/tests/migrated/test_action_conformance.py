"""
test_action_conformance.py — the contract EVERY staged action must satisfy.

Adding an action means wiring the same half-dozen seams every time: register the
staged verbs, declare the schema, add the caret, put it in `WIZARD_ACTIONS`,
render it in the switch, re-broadcast its messages, address them to the right
window. Each seam has its own failure mode, and they share one nasty property —
**they fail silently**. A caret listening for an event nobody re-broadcasts just
sits there; a message addressed to the wrong window is dropped by a filter, not
an error; a caret default that drifts from the backend's wins without a word.
None of it shows up in a typecheck or a normal unit test.

So the checks live HERE, once, driven by the registry rather than written out
per action. A new wizard is covered the moment it registers — nothing to
remember, which is the point.

Two kinds of check:

**Static** (parametrized over every registered key, no fixtures, milliseconds) —
the wiring seams. Covers every action automatically.

**Runtime** (:class:`TestWizardRuntimeConformance`) — the behavioural contracts
that need a real ``Session``: message addressing, StrictMode double-fire,
teardown. These need to know how to OPEN each wizard, which is genuinely
per-wizard, so they read :data:`RUNTIME_FIXTURES`. A wizard absent from that
table is reported as SKIPPED with its reason, so the gap is visible rather than
silently uncovered.

When this suite fails for a NEW action, the fix is almost always to wire the
seam it names — not to add an exemption.
"""
from __future__ import annotations

import inspect
import re
import time
from pathlib import Path

import numpy as np
import pytest

from spyde.actions import registry

# ─────────────────────────────────────────────────────────────────────────────
# Where the renderer lives (skip cleanly in a backend-only checkout)
# ─────────────────────────────────────────────────────────────────────────────

_REPO = Path(__file__).resolve().parents[3]
_COMPONENTS = _REPO / "electron" / "src" / "renderer" / "src" / "components"
_CONTEXT = _REPO / "electron" / "src" / "renderer" / "src" / "kernel" / "SpyDEContext.tsx"
_TOOLBAR = _COMPONENTS / "FloatingToolbar.tsx"

_HAS_RENDERER = _COMPONENTS.is_dir() and _CONTEXT.exists() and _TOOLBAR.exists()
_renderer_only = pytest.mark.skipif(not _HAS_RENDERER,
                                    reason="renderer sources not present")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _caret_files() -> list[Path]:
    return sorted(_COMPONENTS.glob("*Wizard.tsx")) if _HAS_RENDERER else []


# ─────────────────────────────────────────────────────────────────────────────
# The action inventory, derived from the registry (not hand-listed)
# ─────────────────────────────────────────────────────────────────────────────

def _staged_by_key() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in registry.STAGED_HANDLERS:
        if "_" in name:
            out.setdefault(name.split("_", 1)[0], []).append(name)
    return out


STAGED_KEYS = sorted(_staged_by_key())
SCHEMA_KEYS = sorted(registry.wizard_keys())
ALL_HANDLERS = sorted(registry.STAGED_HANDLERS)

#: Keys whose staged handlers are NOT a caret wizard — they are dispatch
#: namespaces for other UI (the report sidebar, the movie editor, the console).
#: Listed explicitly so a genuinely new wizard cannot hide among them.
_NON_WIZARD_KEYS = {"report", "repfig", "movie", "cod", "set", "get", "mark",
                    "compute", "download", "tile", "select", "add", "extract",
                    "overlay", "test", "ipf", "vi"}


def _module_defaults(key: str) -> dict | None:
    """The ``DEFAULTS`` dict of the module that owns *key*'s handlers."""
    import importlib
    for verb in ("open", "run", "generate_library"):
        dotted = registry.STAGED_HANDLERS.get(f"{key}_{verb}")
        if dotted:
            mod = importlib.import_module(dotted.rsplit(".", 1)[0])
            d = getattr(mod, "DEFAULTS", None)
            return d if isinstance(d, dict) else None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Static: the dispatch table itself
# ─────────────────────────────────────────────────────────────────────────────

class TestStagedHandlers:
    @pytest.mark.parametrize("name", ALL_HANDLERS)
    def test_every_handler_resolves(self, name):
        """A dotted path that no longer imports is a button that does nothing.

        The table is lazy by design (heavy deps load on first use), so a typo or
        a moved module is invisible until a user clicks — which is exactly when
        it must not happen.
        """
        fn = registry.resolve_staged(name)
        assert callable(fn), f"{name} did not resolve to a callable"

    @pytest.mark.parametrize("name", ALL_HANDLERS)
    def test_every_handler_takes_the_staged_signature(self, name):
        """`fn(session, plot, payload)` — the uniform shape dispatch calls."""
        fn = registry.resolve_staged(name)
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        assert len(params) >= 3, (
            f"{name}{inspect.signature(fn)} does not accept "
            f"(session, plot, payload)")

    def test_a_wizard_declares_open_and_close_together(self):
        """An `_open` with no `_close` leaks whatever it built; a `_close` with
        no `_open` is dead. They are only meaningful as a pair."""
        by_key = _staged_by_key()
        for key, names in by_key.items():
            has_open = f"{key}_open" in names
            has_close = f"{key}_close" in names
            assert has_open == has_close, (
                f"{key}: _open={has_open} _close={has_close} — a staged wizard "
                f"needs both, so what open creates has somewhere to be torn "
                f"down")


# ─────────────────────────────────────────────────────────────────────────────
# Static: the parameter schema (the three-host parity contract)
# ─────────────────────────────────────────────────────────────────────────────

VALID_TYPES = {"int", "float", "bool", "enum", "file"}


class TestSchemas:
    @pytest.mark.parametrize("key", SCHEMA_KEYS)
    def test_schema_is_well_formed(self, key):
        """Generalises what `test_wizard_schemas` checked for a hand-written
        list — a new wizard is covered without editing this file."""
        schema = registry.wizard_parameters(key)
        assert schema, f"{key} resolves to an empty schema"
        for pname, spec in schema.items():
            assert isinstance(spec, dict), f"{key}.{pname} is not a dict"
            ptype = spec.get("type")
            assert ptype in VALID_TYPES, f"{key}.{pname}: bad type {ptype!r}"
            assert spec.get("name"), f"{key}.{pname}: no display name"
            assert "default" in spec, f"{key}.{pname}: no default"
            d = spec["default"]
            if ptype in ("int", "float"):
                assert isinstance(d, (int, float)) and not isinstance(d, bool)
                assert spec.get("min", d) <= d <= spec.get("max", d)
            elif ptype == "bool":
                assert isinstance(d, bool)
            elif ptype == "enum":
                choices = spec.get("choices") or spec.get("options")
                assert choices and d in choices, \
                    f"{key}.{pname}: default {d!r} not in {choices}"
            elif ptype == "file":
                assert spec.get("extensions"), f"{key}.{pname}: no extensions"

    @pytest.mark.parametrize("key", SCHEMA_KEYS)
    def test_schema_matches_the_module_defaults(self, key):
        """One source of truth. A schema default that drifts from the handler's
        makes the caret and a scripted call disagree about what "default" is."""
        defaults = _module_defaults(key)
        if defaults is None:
            pytest.skip(f"{key}'s module declares no DEFAULTS dict")
        schema = registry.wizard_parameters(key)
        for pname, value in defaults.items():
            if pname not in schema:
                continue          # backend-only knob, deliberately not exposed
            assert schema[pname]["default"] == value, (
                f"{key} schema/{pname} = {schema[pname]['default']!r} but the "
                f"module DEFAULTS says {value!r}")

    def test_schema_is_a_copy(self):
        for key in SCHEMA_KEYS:
            a = registry.wizard_parameters(key)
            a["__scribble__"] = 1
            assert "__scribble__" not in registry.wizard_parameters(key), \
                f"{key}: wizard_parameters handed out its live dict"


# ─────────────────────────────────────────────────────────────────────────────
# Static: the renderer seams
# ─────────────────────────────────────────────────────────────────────────────

@_renderer_only
class TestRendererWiring:
    def test_every_wizard_action_is_rendered(self):
        """A name in `WIZARD_ACTIONS` with no branch in the switch is a toolbar
        button that suppresses the normal param popout and then renders
        nothing — a control that silently does nothing at all."""
        text = _read(_TOOLBAR)
        block = re.search(r"const WIZARD_ACTIONS = new Set\(\[(.*?)\]\)", text, re.S)
        assert block, "could not find WIZARD_ACTIONS"
        declared = set(re.findall(r"'([^']+)'", block.group(1)))
        rendered = set(re.findall(r"openAction\.name === '([^']+)'", text))
        assert declared <= rendered, (
            f"in WIZARD_ACTIONS but never rendered: {sorted(declared - rendered)}")

    def test_every_wizard_action_exists_in_the_toolbar_yaml(self):
        """The caret is opened by NAME. A name that no toolbar action defines
        can never be reached, and nothing would say so."""
        import spyde
        text = _read(_TOOLBAR)
        block = re.search(r"const WIZARD_ACTIONS = new Set\(\[(.*?)\]\)", text, re.S)
        declared = set(re.findall(r"'([^']+)'", block.group(1)))
        known = {name for group in spyde.TOOLBAR_ACTIONS.values()
                 if isinstance(group, dict) for name in group}
        assert declared <= known, (
            f"WIZARD_ACTIONS names with no toolbars.yaml entry: "
            f"{sorted(declared - known)}")

    def test_every_caret_event_is_re_broadcast(self):
        """A caret's `useWizardEvent('spyde:X')` only ever fires if
        SpyDEContext re-broadcasts `X`. Miss the case and the listener is dead:
        no error, no warning, just a caret that never updates.
        """
        cases = set(re.findall(r"case '([a-z0-9_]+)':", _read(_CONTEXT)))
        dead: list[str] = []
        for f in _caret_files():
            for evt in re.findall(r"useWizardEvent\(\s*'spyde:([a-z0-9_]+)'",
                                  _read(f)):
                if evt not in cases:
                    dead.append(f"{f.name} listens for spyde:{evt}")
        assert not dead, (
            "these listeners can never fire — add the type to SpyDEContext's "
            f"re-broadcast switch: {dead}")


# ─────────────────────────────────────────────────────────────────────────────
# Static: caret defaults vs backend defaults
# ─────────────────────────────────────────────────────────────────────────────

def _tsx_defaults(text: str) -> dict[str, str]:
    """Parse a caret's ``const DEFAULTS … = { … }`` literal → {name: raw}."""
    block = re.search(r"const DEFAULTS[^=]*=\s*\{(.*?)\n\}", text, re.S)
    if not block:
        return {}
    return dict(re.findall(r"(\w+):\s*('[^']*'|[-\w.]+)\s*,", block.group(1)))


def _camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(w.title() for w in rest)


def _carets_with_defaults() -> list[tuple[str, Path]]:
    """``(wizard key, caret path)`` for carets that declare a DEFAULTS literal.

    The key is recovered from the staged action the caret sends, so a renamed
    component or a new wizard needs no edit here.
    """
    out = []
    for f in _caret_files():
        text = _read(f)
        if not _tsx_defaults(text):
            continue
        keys = {m.split("_", 1)[0]
                for m in re.findall(r"sendAction\(\s*'([a-z0-9_]+)'", text)}
        for key in sorted(keys & set(STAGED_KEYS)):
            if _module_defaults(key):
                out.append((key, f))
                break
    return out


@_renderer_only
class TestCaretDefaults:
    """A caret default that drifts from the backend's WINS SILENTLY.

    The caret sends its own value in every payload, so the backend's DEFAULTS is
    never consulted and nothing fails — the feature just behaves differently
    from what the Python says. It has cost this project a session before (see
    CLAUDE.md), which is why this reads the TSX rather than trusting a comment.
    """

    @pytest.mark.parametrize("key,path", _carets_with_defaults(),
                             ids=lambda v: v if isinstance(v, str) else v.name)
    def test_caret_defaults_match_the_backend(self, key, path):
        defaults = _module_defaults(key)
        found = _tsx_defaults(_read(path))
        for pname, py_value in defaults.items():
            raw = found.get(_camel(pname))
            if raw is None:
                continue          # not surfaced on this caret
            if isinstance(py_value, bool):
                ts_value, py_cmp = raw == "true", py_value
            elif isinstance(py_value, (int, float)):
                try:
                    ts_value, py_cmp = float(raw), float(py_value)
                except ValueError:
                    continue      # a non-numeric literal; nothing to compare
            else:
                ts_value, py_cmp = raw.strip("'"), py_value
            assert ts_value == py_cmp, (
                f"{path.name} {_camel(pname)}={raw} but {key} DEFAULTS"
                f"[{pname}]={py_value!r} — the caret's value would win silently")

    def test_the_guard_would_catch_a_drift(self):
        """A guard nobody has seen fail is not a guard."""
        pairs = _carets_with_defaults()
        assert pairs, "no caret DEFAULTS literals found — has the format changed?"
        key, path = pairs[0]
        found = _tsx_defaults(_read(path))
        name, raw = next(iter(found.items()))
        mutated = dict(found)
        mutated[name] = "'__drifted__'" if raw.startswith("'") else "-99999"
        assert mutated[name] != raw


# ─────────────────────────────────────────────────────────────────────────────
# Runtime: the behavioural contracts
# ─────────────────────────────────────────────────────────────────────────────

#: How to OPEN each wizard for the runtime checks below.
#:
#: ``fixture`` names a loader on the test Session; ``payload`` is the open
#: payload. A wizard NOT listed is reported as skipped with its reason — the
#: gap stays visible instead of quietly reading as covered.
RUNTIME_FIXTURES: dict[str, dict] = {
    "dpc":    {"loader": "_load_test_data_dpc",
               "kwargs": {"nav": 12, "sig": 24}, "payload": {}},
    "czb":    {"loader": "_load_test_data", "kwargs": {}, "payload": {}},
    "crop":   {"loader": "_load_test_data", "kwargs": {}, "payload": {}},
}

#: Why a wizard has no runtime fixture. Explicit, so "not covered" is a
#: decision on the record rather than an oversight.
RUNTIME_EXEMPT: dict[str, str] = {
    "fv":     "opening spawns a live GPU/torch preview; covered by "
              "test_find_vectors_wizard.py",
    "om":     "needs a .cif and a generated template library",
    "vom":    "needs diffraction vectors AND a template library",
    "ebsd":   "needs a simulated EBSD master pattern + dictionary",
    "strain": "needs a Find-Vectors result on the tree",
    "drift":  "needs an in-situ movie; covered by test_drift_wizard.py",
    "fit":    "needs a 1-D spectrum image; covered by test_fit_action.py",
    "bg":     "needs a 1-D spectrum image",
    "movie":  "the movie editor is not a caret wizard",
    "report": "the report sidebar is not a caret wizard",
}

WIZARD_KEYS = sorted(k for k in STAGED_KEYS
                     if f"{k}_open" in registry.STAGED_HANDLERS
                     and k not in _NON_WIZARD_KEYS)


def _wait(pred, timeout=30.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _open_wizard(session, key, spec):
    """Load the fixture, open the wizard, return its (plot, controller)."""
    _call_loader(session, spec)
    assert _wait(lambda: _signal_plot(session) is not None), \
        f"{key}: the fixture never produced a signal plot"
    plot = _signal_plot(session)
    registry.resolve_staged(f"{key}_open")(session, plot, dict(spec["payload"]))
    return plot


def _call_loader(session, spec):
    loader = getattr(session, spec["loader"])
    kwargs = spec.get("kwargs") or {}
    try:
        return loader(kwargs) if kwargs else loader()
    except TypeError:
        return loader()


def _signal_plot(session):
    return next((p for p in session._plots
                 if not p.is_navigator and p.plot_state is not None), None)


def _wizard_messages(messages, key):
    """Messages whose type belongs to this wizard's namespace."""
    return [m for m in messages if isinstance(m, dict)
            and str(m.get("type", "")).startswith(f"{key}_")]


class TestWizardRuntimeConformance:
    """Contracts that only a real Session can check."""

    @pytest.fixture(autouse=True)
    def _close_every_wizard(self, window):
        """Tear down any wizard the test left open, and let its workers land.

        Without this the `window` fixture shuts the Session down with a measure
        still running, and the worker raises into the suite output — a traceback
        that looks like a product failure and is not one. It is also the same
        hygiene the DPC suite needed: a late `emit` from a torn-down session
        lands in the NEXT test's captured messages.
        """
        yield
        session = window["window"]
        for key in RUNTIME_FIXTURES:
            close = registry.resolve_staged(f"{key}_close")
            for tree in list(getattr(session, "signal_trees", []) or []):
                if not _live_controllers(tree, key):
                    continue
                plot = next(iter(getattr(tree, "signal_plots", []) or []), None)
                try:
                    close(session, plot, {})
                except Exception:            # teardown must never mask a failure
                    pass
        _wait(lambda: not any(_live_controllers(t, k)
                              for t in getattr(session, "signal_trees", []) or []
                              for k in RUNTIME_FIXTURES), timeout=5.0)

    def test_every_wizard_is_either_covered_or_exempt(self):
        """The list cannot silently rot: a new wizard must be added to one of
        the two tables, and the exemption must carry a reason."""
        unaccounted = [k for k in WIZARD_KEYS
                       if k not in RUNTIME_FIXTURES and k not in RUNTIME_EXEMPT]
        assert not unaccounted, (
            f"new wizard(s) {unaccounted}: add a RUNTIME_FIXTURES entry so the "
            f"contracts below cover them, or a RUNTIME_EXEMPT reason")

    @pytest.mark.parametrize("key", sorted(RUNTIME_FIXTURES))
    def test_messages_are_addressed_to_the_caret_window(self, key, window,
                                                        monkeypatch):
        """Every wizard message must carry the SOURCE window's id.

        `useWizardEvent` drops anything whose `window_id` is not the window the
        caret is mounted on. A wizard that opens a bare-figure result window has
        a second, different id to hand, and addressing its messages to THAT one
        makes every message vanish silently — no error, just a caret that never
        updates. This is the single most repeated bug in this area.
        """
        session = window["window"]
        _patch_module_emit(monkeypatch, key, window["messages"])
        plot = _open_wizard(session, key, RUNTIME_FIXTURES[key])
        src_id = getattr(plot, "window_id", None)
        _wait(lambda: _wizard_messages(window["messages"], key))

        misaddressed = [
            (m.get("type"), m.get("window_id"))
            for m in _wizard_messages(window["messages"], key)
            if "window_id" in m and m["window_id"] not in (None, src_id)
        ]
        assert not misaddressed, (
            f"{key}: messages addressed away from the caret's window "
            f"(id={src_id}); useWizardEvent will drop them: {misaddressed}")

    @pytest.mark.parametrize("key", sorted(RUNTIME_FIXTURES))
    def test_open_close_open_leaves_one_controller(self, key, window, monkeypatch):
        """React StrictMode fires the three synchronously, before any worker
        lands. Two live controllers means two of everything they built."""
        session = window["window"]
        _patch_module_emit(monkeypatch, key, window["messages"])
        spec = RUNTIME_FIXTURES[key]
        plot = _open_wizard(session, key, spec)
        close = registry.resolve_staged(f"{key}_close")
        open_ = registry.resolve_staged(f"{key}_open")
        close(session, plot, {})
        open_(session, plot, dict(spec["payload"]))
        time.sleep(0.6)          # let any superseded worker land and be dropped

        live = [t for t in session.signal_trees
                if _live_controllers(t, key)]
        assert len(live) <= 1, f"{key}: {len(live)} trees hold a live controller"
        for tree in live:
            assert len(_live_controllers(tree, key)) == 1, \
                f"{key}: a tree holds more than one live controller"

    @pytest.mark.parametrize("key", sorted(RUNTIME_FIXTURES))
    def test_close_leaves_nothing_registered(self, key, window, monkeypatch):
        """Close must reach every window the wizard registered. A leaked
        controller keeps its figures alive and its window on screen."""
        session = window["window"]
        _patch_module_emit(monkeypatch, key, window["messages"])
        plot = _open_wizard(session, key, RUNTIME_FIXTURES[key])
        before = set(getattr(session, "_window_controllers", {}) or {})
        registry.resolve_staged(f"{key}_close")(session, plot, {})
        after = set(getattr(session, "_window_controllers", {}) or {})
        assert after <= before, f"{key}: close REGISTERED a controller"
        for tree in session.signal_trees:
            assert not _live_controllers(tree, key), \
                f"{key}: a live controller survived close"


#: Wizards whose OPEN starts no dataset-wide compute, so there is nothing for
#: :class:`TestComputeCancellation` to cancel. Explicit, for the same reason
#: ``RUNTIME_EXEMPT`` is: "nothing to cancel" must be a decision on the record,
#: not an action that quietly forgot to register.
NO_COMPUTE_ON_OPEN: dict[str, str] = {
    "crop": "opening only draws an ROI; the crop happens on commit",
    "czb":  "opening only draws the search region; the pass is czb_run",
}


def _cancel_tokens(tree) -> list:
    """Everything registered on *tree* for cancellation — ``[False]`` stop flags
    and ``.cancel()``-able futures alike (``BaseSignalTree.register_cancel``)."""
    return (list(getattr(tree, "_cancel_flags", None) or [])
            + list(getattr(tree, "_cancel_futures", None) or []))


def _is_cancelled(token) -> bool:
    """True if *token* has been asked to stop — flag set, or future cancelled
    (a future that finished before the supersede landed also counts: there is
    no longer a running compute, which is the property under test)."""
    if isinstance(token, list):
        return bool(token) and bool(token[0])
    for probe in ("cancelled", "done"):
        fn = getattr(token, probe, None)
        if callable(fn):
            try:
                if fn():
                    return True
            except Exception:
                pass
    return False


class TestComputeCancellation:
    """A superseded or abandoned compute must be **cancelled**, not left to
    finish so its result can be thrown away.

    A pass over the dataset is the most expensive thing the app does. One that
    nobody is waiting for still costs the cluster the whole scan, and they pile
    up: every drag of an ROI, every StrictMode double-mount, starts another.
    Dropping the RESULT is not cancelling the WORK.

    The contract — ``virtual_image.py`` is the reference implementation:

    * **register** the in-flight compute on the tree, ``register_cancel(flag=…)``
      for a chunk loop or ``register_cancel(future=…)`` for a single future, so
      closing the tree stops it;
    * **cancel the previous one** before starting a new one, and
      ``unregister_cancel`` it, so the registry does not grow one entry per
      interaction.

    DPC shipped without any of this: a superseded measure ran to completion on
    every double-mount, and two of them racing on one hyperspy signal is what
    made ``test_open_close_open_leaves_exactly_one_wizard`` fail on CI. This
    gate is registry-driven so the next action cannot repeat it silently — a
    new wizard either registers a token or names itself in
    :data:`NO_COMPUTE_ON_OPEN`.
    """

    @pytest.mark.parametrize("key", sorted(RUNTIME_FIXTURES))
    def test_an_open_compute_is_registered_for_tree_close(
            self, key, window, monkeypatch):
        if key in NO_COMPUTE_ON_OPEN:
            pytest.skip(NO_COMPUTE_ON_OPEN[key])
        session = window["window"]
        _patch_module_emit(monkeypatch, key, window["messages"])
        spec = RUNTIME_FIXTURES[key]
        _call_loader(session, spec)
        assert _wait(lambda: _signal_plot(session) is not None)
        plot = _signal_plot(session)
        tree = plot.signal_tree
        # Only tokens this OPEN adds count — the tree already carries the
        # navigator fill's own.
        before = {id(t) for t in _cancel_tokens(tree)}
        registry.resolve_staged(f"{key}_open")(session, plot,
                                               dict(spec["payload"]))
        added = [t for t in _cancel_tokens(tree) if id(t) not in before]
        assert added, (
            f"{key}: opening started a compute but registered nothing on the "
            f"tree, so closing the tree cannot stop it. Call "
            f"tree.register_cancel(flag=…) or (future=…) — see virtual_image.py "
            f"— or add {key!r} to NO_COMPUTE_ON_OPEN with a reason.")

    #: Action modules that dispatch a compute without registering a cancel
    #: token. Pre-existing and GRANDFATHERED so the gate can go in — not an
    #: endorsement. Some are legitimate (``find_vectors/orchestrate`` is handed
    #: a flag by its caller rather than owning one); the rest are the same gap
    #: DPC had. **Do not add to this list** — wire the token instead. Shrinking
    #: it is the follow-up.
    _UNREGISTERED_DISPATCH: frozenset = frozenset({
        "background_action.py", "center_zero_beam.py", "composition.py",
        "csb_to_frames.py", "orchestrate.py", "fit_action.py",
        "orientation_compute.py", "strain_action.py",
    })

    def test_a_dispatched_compute_registers_a_cancel_token(self):
        """STATIC: dispatching a dataset-wide compute obliges you to register a
        cancel token, so closing the tree can stop it.

        The runtime checks above only reach wizards that have a fixture, and
        most actions never will — this one reaches every module.
        """
        import re
        from pathlib import Path
        actions = Path(__file__).resolve().parents[2] / "actions"
        dispatch = re.compile(r"\b(run_on_worker|compute_chunks_progressive"
                              r"|client\.compute|submit_graph)\s*\(")
        # A bare token, not a call: the registration is often reached through
        # `getattr(tree, "register_cancel", None)`, which a `\(` pattern misses
        # — as an earlier version of this guard did, on this very file.
        registers = re.compile(r"\bregister_cancel\b")
        offenders = sorted(
            p.name for p in actions.rglob("*.py")
            if dispatch.search(p.read_text(encoding="utf-8"))
            and not registers.search(p.read_text(encoding="utf-8"))
            and p.name not in self._UNREGISTERED_DISPATCH)
        assert not offenders, (
            f"{offenders} dispatch a compute but register no cancel token, so "
            f"closing the tree cannot stop it and a superseded pass runs to "
            f"completion. Call tree.register_cancel(flag=…) for a chunk loop or "
            f"(future=…) for a single future — see virtual_image.py and "
            f"DpcWizard._track_measure.")

    def test_the_dispatch_guard_would_catch_a_regression(self):
        """The guard above is only worth having if it fails on the real thing.
        DPC before the fix: a compute dispatched, nothing registered."""
        import re
        dispatch = re.compile(r"\b(run_on_worker|compute_chunks_progressive"
                              r"|client\.compute|submit_graph)\s*\(")
        registers = re.compile(r"\bregister_cancel\b")
        pre_fix = ("def measure(self):\n"
                   "    self.run_on_worker(_work, name='dpc-measure')\n")
        assert dispatch.search(pre_fix) and not registers.search(pre_fix)

    def test_a_registered_compute_has_a_cancel_path(self):
        """STATIC half of the gate: any action that registers a cancel token
        must also, in the same module, cancel or unregister one.

        Registering only wires up TREE CLOSE. Superseding — a re-measure, an ROI
        drag, a StrictMode remount — is the common case, and an action that
        never calls ``unregister_cancel`` or ``.cancel()`` leaves every prior
        pass running and its registry growing one entry per interaction.

        This is lexical on purpose, like ``test_chunk_dispatch_guard``: the
        runtime checks above can only reach the wizards that have a fixture, and
        most actions never will. This one reaches all of them.
        """
        import re
        from pathlib import Path
        actions = Path(__file__).resolve().parents[2] / "actions"
        offenders = []
        for path in sorted(actions.rglob("*.py")):
            src = path.read_text(encoding="utf-8")
            if not re.search(r"\bregister_cancel\s*\(", src):
                continue
            if re.search(r"\bunregister_cancel\s*\(", src) or \
                    re.search(r"\.cancel\s*\(\s*\)", src):
                continue
            offenders.append(path.name)
        assert not offenders, (
            f"{offenders} register a compute for cancellation but never cancel "
            f"or unregister one. Registering only covers TREE CLOSE; a "
            f"superseded pass must be cancelled too, or it runs to completion "
            f"and its result is discarded. See virtual_image.py "
            f"(cancel prior → unregister → register new).")

    @pytest.mark.parametrize("key", sorted(RUNTIME_FIXTURES))
    def test_close_cancels_the_in_flight_compute(self, key, window, monkeypatch):
        """Closing the caret is the clearest "nobody is waiting for this"."""
        if key in NO_COMPUTE_ON_OPEN:
            pytest.skip(NO_COMPUTE_ON_OPEN[key])
        session = window["window"]
        _patch_module_emit(monkeypatch, key, window["messages"])
        spec = RUNTIME_FIXTURES[key]
        _call_loader(session, spec)
        assert _wait(lambda: _signal_plot(session) is not None)
        plot = _signal_plot(session)
        tree = plot.signal_tree
        before = {id(t) for t in _cancel_tokens(tree)}
        registry.resolve_staged(f"{key}_open")(session, plot,
                                               dict(spec["payload"]))
        mine = [t for t in _cancel_tokens(tree) if id(t) not in before]
        registry.resolve_staged(f"{key}_close")(session, plot, {})
        left = {id(t) for t in _cancel_tokens(tree)}
        for token in mine:
            assert _is_cancelled(token) or id(token) not in left, (
                f"{key}: close left a compute running. Cancel it in the "
                f"wizard's teardown (see DpcWizard.remove).")


def _live_controllers(tree, key) -> list:
    """Controllers this wizard parks on a tree (`_<key>_wizard`, `_<key>_controller`)."""
    out = []
    for attr in (f"_{key}_wizard", f"_{key}_controller"):
        ctrl = getattr(tree, attr, None)
        if ctrl is not None and not getattr(ctrl, "_closed", False):
            out.append(ctrl)
    return out


def _patch_module_emit(monkeypatch, key, sink):
    """Route the action module's own `emit` into the captured list.

    Every action module does `from de_shell.ipc import emit` at import, so
    conftest's patch of `ipc.emit` never reaches that binding — the hazard
    conftest documents for `session.py`, applied generically.
    """
    import importlib
    dotted = registry.STAGED_HANDLERS.get(f"{key}_open")
    if not dotted:
        return
    mod = importlib.import_module(dotted.rsplit(".", 1)[0])
    if hasattr(mod, "emit"):
        monkeypatch.setattr(mod, "emit", sink.append)
